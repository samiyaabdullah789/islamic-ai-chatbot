"""
Ingest registered sources and check ChromaDB, without one-off commands.

Usage (from the project root, venv active, Ollama running):

    python -m scripts.ingest_source --status
        Show every source: DB status, chunks in ChromaDB, file present.
        Read-only.

    python -m scripts.ingest_source 3
    python -m scripts.ingest_source 3 4 5
        Ingest these source IDs one after another. Stops at the first
        error. Stop the backend first, and start it again afterwards so
        the BM25 index is rebuilt.

    python -m scripts.ingest_source 3 --reset-missing
        For a source whose DB status says "completed" (or is stuck on
        "processing") but which has 0 chunks in ChromaDB: set its status
        back to "pending", then ingest it. This is the only option that
        changes a database record, and only for such sources.

Source 1 (Sahih al-Bukhari) is always refused, as in the API.
"""

import argparse
import asyncio
from pathlib import Path

from app.db.session import SessionLocal
from app.embeddings.ollama_embeddings import embedding_service
from app.repositories.source_repository import source_repository
from app.services.ingestion_service import ingestion_service
from app.vector_store.chroma_store import chroma_store

BASE_DIR = Path(__file__).resolve().parent.parent
PROTECTED_SOURCE_IDS = {1}


def chroma_count(source_id: int) -> int:
    return len(chroma_store.get_source_document_ids(source_id=source_id))


def resolve_path(file_path: str) -> Path:
    path = Path(file_path)
    return path if path.is_absolute() else BASE_DIR / path


def print_status() -> None:
    db = SessionLocal()
    try:
        sources = sorted(source_repository.get_all(db=db), key=lambda s: s.id)
        print(f"ChromaDB total documents: {chroma_store.collection.count()}\n")
        print(f"{'ID':>3}  {'DB status':<10} {'Chroma':>6}  {'File':<7} Name")
        for source in sources:
            count = chroma_count(source.id)
            exists = "ok" if resolve_path(source.file_path).is_file() else "MISSING"
            note = ""
            if source.status == "completed" and count == 0:
                note = "   <- completed in DB but empty in ChromaDB"
            elif source.status == "pending" and count > 0:
                note = "   <- pending in DB but has chunks"
            print(
                f"{source.id:>3}  {source.status:<10} {count:>6}  "
                f"{exists:<7} {source.name}{note}"
            )
    finally:
        db.close()


async def check_ollama() -> None:
    try:
        vectors = await embedding_service.embed(["embedding check"])
    except Exception as error:
        raise SystemExit(
            f"Ollama embeddings are not working ({error}). "
            "Check that Ollama is running and that the embedding model "
            "is available on the LLM_BASE_URL in .env."
        )
    print(f"Ollama embeddings OK ({len(vectors[0])} dimensions)")


async def ingest_one(source_id: int, reset_missing: bool) -> bool:
    if source_id in PROTECTED_SOURCE_IDS:
        print(f"Source {source_id} is protected from re-ingestion. Skipped.")
        return False

    db = SessionLocal()
    try:
        source = source_repository.get_by_id(db=db, source_id=source_id)
        if source is None:
            print(f"Source {source_id} not found.")
            return False

        before = chroma_count(source_id)
        print(f"\n=== Source {source_id}: {source.name}")
        print(f"DB status: {source.status} | chunks in ChromaDB: {before}")

        if not resolve_path(source.file_path).is_file():
            print(f"File not found: {source.file_path}")
            return False

        if source.status in {"completed", "processing"}:
            if before == 0 and reset_missing:
                source_repository.update_status(
                    db=db, source=source, new_status="pending"
                )
                print("Status reset to pending (it had 0 chunks in ChromaDB).")
            elif before == 0:
                print(
                    f"Status is '{source.status}' but ChromaDB has 0 chunks. "
                    "Run again with --reset-missing to re-ingest it."
                )
                return False
            else:
                print("Already ingested. Nothing to do.")
                return True

        result = await ingestion_service.ingest_source(db=db, source_id=source_id)
        after = chroma_count(source_id)
        print(f"Result: {result}")
        print(f"Chunks in ChromaDB for source {source_id}: {after}")
        if after != result.get("total_documents"):
            print("WARNING: ChromaDB count does not match the expected total.")
            return False
        return True
    except Exception as error:
        print(f"FAILED: {type(error).__name__}: {error}")
        return False
    finally:
        db.close()


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source_ids", nargs="*", type=int)
    parser.add_argument("--status", action="store_true", help="Show sources and ChromaDB counts")
    parser.add_argument(
        "--reset-missing",
        action="store_true",
        help="Re-ingest sources marked completed/processing that have 0 chunks",
    )
    args = parser.parse_args()

    if args.status or not args.source_ids:
        print_status()
        if not args.source_ids:
            return

    await check_ollama()
    for source_id in args.source_ids:
        if not await ingest_one(source_id, args.reset_missing):
            print(f"\nStopped at source {source_id}.")
            break
    print(f"\nChromaDB total documents: {chroma_store.collection.count()}")
    print("Start (or restart) the backend so the BM25 index is rebuilt.")


if __name__ == "__main__":
    asyncio.run(main())
