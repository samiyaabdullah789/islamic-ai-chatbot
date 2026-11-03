
import asyncio
import time

from app.embeddings.ollama_embeddings import embedding_service
from app.rag.qa_hybrid_retriever import qa_hybrid_retriever
from app.vector_store.chroma_client import chroma_client


TEST_QUESTIONS = [
    "What are the proofs for forgetful prostration in the Hanafi madhhab?",
    "Is Witr prayer wajib in the Hanafi school?",
    "When should sajda al-sahw be performed?",
]


async def test_question(question: str):
    collection = chroma_client.qa_collection

    print("\n" + "=" * 75)
    print(f"QUESTION: {question}")
    print("=" * 75)

    # -----------------------------------
    # 1. ChromaDB-only search
    # -----------------------------------
    start = time.perf_counter()

    embeddings = await embedding_service.embed([question])

    dense_results = collection.query(
        query_embeddings=embeddings,
        n_results=min(5, collection.count()),
        include=["metadatas", "distances"],
    )

    print("\n[CHROMADB ONLY]")

    for rank, (metadata, distance) in enumerate(
        zip(
            dense_results["metadatas"][0],
            dense_results["distances"][0],
        ),
        start=1,
    ):
        metadata = metadata or {}

        print(
            f"{rank}. "
            f"{metadata.get('question', '')[:110]} "
            f"| ID: {metadata.get('original_record_id', 'N/A')} "
            f"| Distance: {distance:.4f}"
        )

    print(
        "[TIMING] Dense-only total: "
        f"{time.perf_counter() - start:.2f}s",
        flush=True,
    )

    # -----------------------------------
    # 2. Hybrid search
    # -----------------------------------
    results = await qa_hybrid_retriever.retrieve(
        query=question,
        limit=5,
        dense_limit=30,
        bm25_limit=30,
        source_id=None,
        madhhab=None,
    )

    print("\n[HYBRID SEARCH]")

    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. "
            f"{result['question'][:110]} "
            f"| ID: {result['original_record_id']} "
            f"| RRF: {result['rrf_score']:.5f}"
        )


async def main():
    count = chroma_client.qa_collection.count()

    print(f"[TEST] Total indexed Q&A records: {count}")

    if count == 0:
        print("[ERROR] Q&A collection is empty")
        return

    for question in TEST_QUESTIONS:
        await test_question(question)


if __name__ == "__main__":
    asyncio.run(main())
