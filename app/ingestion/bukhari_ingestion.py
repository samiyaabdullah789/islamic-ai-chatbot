import asyncio

from app.embeddings.ollama_embeddings import embedding_service
from app.ingestion.pdf_processor import process_pdf
from app.vector_store.chroma_store import chroma_store


SOURCE_ID = 1
SOURCE_NAME = "Sahih al-Bukhari"

BATCH_SIZE = 32

# Temporary testing mode
TEST_MODE = False
TEST_LIMIT = 2


async def ingest_bukhari() -> None:
    print("Reading and processing Sahih al-Bukhari...")

    hadiths = process_pdf()

    # Remove records whose source PDF contains no Hadith text.
    usable_hadiths = [
        hadith
        for hadith in hadiths
        if hadith["text"].strip()
    ]

    empty_count = len(hadiths) - len(usable_hadiths)

    print("\n" + "=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    print(f"Extracted Hadiths: {len(hadiths)}")
    print(f"Usable Hadiths: {len(usable_hadiths)}")
    print(f"Skipped empty Hadiths: {empty_count}")

    # For the first test, ingest only a very small number.
    if TEST_MODE:
        usable_hadiths = usable_hadiths[:TEST_LIMIT]

        print(
            f"TEST MODE: Only first "
            f"{len(usable_hadiths)} Hadiths will be ingested."
        )

    total = len(usable_hadiths)

    if total == 0:
        print("No usable Hadiths found.")
        return

    for start in range(0, total, BATCH_SIZE):
        batch = usable_hadiths[
            start:start + BATCH_SIZE
        ]

        ids = []
        documents = []
        metadatas = []

        for hadith in batch:
            volume = hadith["volume"]
            book = hadith["book"]
            number = hadith["hadith_number"]
            narrator = hadith["narrator"] or ""

            document = hadith["text"].strip()

            # Stable ID prevents duplicate records
            # when the same source is re-ingested.
            document_id = (
                f"source-{SOURCE_ID}"
                f"-v{volume}"
                f"-b{book}"
                f"-h{number}"
            )

            metadata = {
                "source_id": SOURCE_ID,
                "source_name": SOURCE_NAME,
                "volume": volume,
                "book": book,
                "hadith_number": number,
                "narrator": narrator,
            }

            ids.append(document_id)
            documents.append(document)
            metadatas.append(metadata)

        print(
            f"\nEmbedding Hadiths "
            f"{start + 1}-"
            f"{start + len(batch)} "
            f"of {total}..."
        )

        embeddings = await embedding_service.embed(
            documents
        )

        chroma_store.add_documents(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        print(
            f"Stored {len(batch)} Hadiths "
            f"in ChromaDB."
        )

    print("\n" + "=" * 60)

    if TEST_MODE:
        print("TEST INGESTION COMPLETE")
    else:
        print("SAHIH AL-BUKHARI INGESTION COMPLETE")

    print("=" * 60)
    print(f"Source ID: {SOURCE_ID}")
    print(f"Stored Hadiths: {total}")


if __name__ == "__main__":
    asyncio.run(ingest_bukhari())