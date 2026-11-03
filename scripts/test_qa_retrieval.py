
import asyncio
import time

from app.embeddings.ollama_embeddings import embedding_service
from app.vector_store.chroma_client import chroma_client


TEST_QUESTION = (
    "What are the proofs for forgetful "
    "prostration in the Hanafi madhhab?"
)


async def main():
    collection = chroma_client.qa_collection

    print(f"[QA TEST] Collection: {collection.name}")
    print(f"[QA TEST] Total records: {collection.count()}")
    print(f"[QA TEST] Question: {TEST_QUESTION}")

    if collection.count() == 0:
        print("[ERROR] Q&A collection is empty")
        return

    start = time.perf_counter()

    embeddings = await embedding_service.embed(
        [TEST_QUESTION]
    )

    print(
        "[TIMING] Query embedding: "
        f"{time.perf_counter() - start:.2f}s",
        flush=True,
    )

    start = time.perf_counter()

    results = collection.query(
        query_embeddings=embeddings,
        n_results=min(5, collection.count()),
        where={"source_id": 2},
        include=["documents", "metadatas", "distances"],
    )

    print(
        "[TIMING] QA retrieval: "
        f"{time.perf_counter() - start:.2f}s",
        flush=True,
    )

    ids = results["ids"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for index, (record_id, metadata, distance) in enumerate(
        zip(ids, metadatas, distances),
        start=1,
    ):
        question = metadata.get("question", "")
        answer = metadata.get("answer", "")

        print(f"\n{'=' * 65}")
        print(f"RESULT #{index}")
        print(f"Document ID: {record_id}")
        print(f"Distance: {distance:.4f}")
        print(f"Question: {question}")
        print(f"Original Record ID: {metadata.get('original_record_id', 'N/A')}")
        print(f"Madhhab: {metadata.get('madhhab', '')}")
        print(f"Answer length: {len(answer)} characters")
        print(f"Answer preview: {answer[:500]}")
        print(f"{'=' * 65}")


if __name__ == "__main__":
    asyncio.run(main())
