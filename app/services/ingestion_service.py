
import asyncio
from pathlib import Path

from sqlalchemy.orm import Session

from app.embeddings.ollama_embeddings import embedding_service
from app.ingestion.generic_pdf_processor import process_generic_pdf
from app.ingestion.generic_json_processor import process_generic_json
from app.repositories.source_repository import source_repository
from app.vector_store.chroma_store import chroma_store


BASE_DIR = Path(__file__).resolve().parents[2]

BATCH_SIZE = 32


class IngestionService:

    async def ingest_source(
        self,
        db: Session,
        source_id: int,
    ) -> dict:

        source = source_repository.get_by_id(
            db=db,
            source_id=source_id,
        )

        if source is None:
            raise ValueError("Source not found")

        # Only allow new ingestion or recovery of failed ingestion.
        if source.status not in {"pending", "failed"}:
            raise ValueError(
                f"Source {source_id} has status "
                f"'{source.status}'. "
                "Only pending or failed sources can be ingested."
            )

        file_path = Path(source.file_path)

        if not file_path.is_absolute():
            file_path = BASE_DIR / file_path

        file_path = file_path.resolve()

        if not file_path.is_file():
            raise FileNotFoundError(
                f"Source file not found: {file_path}"
            )

        extension = file_path.suffix.lower()

        if extension not in {".pdf", ".json"}:
            raise ValueError(
                "Only PDF and JSON sources are supported."
            )

        # Load existing document IDs without deleting anything.
        existing_ids = set(
            chroma_store.get_source_document_ids(
                source_id=source_id,
            )
        )

        # Never automatically resume a partially ingested PDF.
        # PDF processing may produce different chunk boundaries.
        if extension == ".pdf" and existing_ids:
            raise ValueError(
                "This PDF already contains documents in ChromaDB. "
                "Automatic PDF re-ingestion is blocked."
            )

        # Protect pending sources that unexpectedly contain data.
        if source.status == "pending" and existing_ids:
            raise ValueError(
                f"Pending source {source_id} already has "
                f"{len(existing_ids)} documents in ChromaDB. "
                "Manual review is required."
            )

        stored_count = 0

        try:
            if extension == ".pdf":
                pdf_chunks = await asyncio.to_thread(
                    process_generic_pdf,
                    str(file_path),
                )

                documents = [
                    {
                        "text": chunk["text"],
                        "metadata": {
                            "page": chunk["page"],
                            "chunk_index": chunk["chunk_index"],
                        },
                        "record_index": chunk["chunk_index"],
                    }
                    for chunk in pdf_chunks
                ]

            else:
                documents = await asyncio.to_thread(
                    process_generic_json,
                    str(file_path),
                )

            if not documents:
                raise ValueError(
                    "No usable documents found in source."
                )

            # Build all expected IDs and check for duplicates.
            expected_ids = set()
            missing_documents = []

            for document in documents:
                index = document["record_index"]

                document_id = (
                    f"source-{source_id}-chunk-{index}"
                )

                if document_id in expected_ids:
                    raise ValueError(
                        f"Duplicate document ID: {document_id}"
                    )

                expected_ids.add(document_id)

                if document_id not in existing_ids:
                    missing_documents.append(
                        (document_id, document)
                    )

            # Stop if ChromaDB has IDs that the current
            # processor/file does not produce.
            unexpected_ids = existing_ids - expected_ids

            if unexpected_ids:
                raise ValueError(
                    f"Source {source_id} has "
                    f"{len(unexpected_ids)} unexpected "
                    "document IDs in ChromaDB. "
                    "The source file or processing logic "
                    "may have changed. Manual review required."
                )

            total = len(documents)
            missing_count = len(missing_documents)

            print(
                f"[INGESTION] Source {source_id}: "
                f"{len(existing_ids)} existing, "
                f"{missing_count} missing, "
                f"{total} total.",
                flush=True,
            )

            # Mark processing only after validation.
            source_repository.update_status(
                db=db,
                source=source,
                new_status="processing",
            )

            for start in range(
                0,
                missing_count,
                BATCH_SIZE,
            ):
                batch = missing_documents[
                    start:start + BATCH_SIZE
                ]

                ids = []
                texts = []
                metadatas = []

                for document_id, document in batch:

                    metadata = {
                        "source_id": source_id,
                        "source_name": source.name,
                        "file_type": extension.lstrip("."),
                        **document["metadata"],
                    }

                    ids.append(document_id)
                    texts.append(document["text"])
                    metadatas.append(metadata)

                embeddings = await embedding_service.embed(
                    texts
                )

                if len(embeddings) != len(texts):
                    raise RuntimeError(
                        "Embedding count does not match "
                        "document count."
                    )

                chroma_store.add_documents(
                    ids=ids,
                    documents=texts,
                    embeddings=embeddings,
                    metadatas=metadatas,
                )

                stored_count += len(batch)

                print(
                    f"[INGESTION] Source {source_id}: "
                    f"{stored_count}/{missing_count} "
                    "missing documents stored.",
                    flush=True,
                )

            # Verify actual ChromaDB IDs before completion.
            final_ids = set(
                chroma_store.get_source_document_ids(
                    source_id=source_id,
                )
            )

            if final_ids != expected_ids:
                remaining = len(expected_ids - final_ids)
                unexpected = len(final_ids - expected_ids)

                raise RuntimeError(
                    "ChromaDB verification failed. "
                    f"Missing: {remaining}, "
                    f"Unexpected: {unexpected}."
                )

            source_repository.update_status(
                db=db,
                source=source,
                new_status="completed",
            )

            return {
                "source_id": source_id,
                "source_name": source.name,
                "status": "completed",
                "documents_already_present": len(existing_ids),
                "documents_stored": stored_count,
                "total_documents": len(final_ids),
            }

        except Exception:
            db.rollback()

            try:
                source_repository.update_status(
                    db=db,
                    source=source,
                    new_status="failed",
                )
            except Exception:
                db.rollback()

            raise


ingestion_service = IngestionService()
