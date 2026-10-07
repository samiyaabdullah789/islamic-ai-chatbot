import asyncio

from app.embeddings.ollama_embeddings import embedding_service
from app.vector_store.chroma_store import chroma_store


async def main():

    # Temporary test documents
    documents = [
        "Python is a programming language.",
        "PostgreSQL is a relational database.",
        "FastAPI is a Python framework for building APIs.",
    ]

    # Generate embeddings
    embeddings = await embedding_service.embed(documents)

    # Store documents in ChromaDB
    chroma_store.add_documents(
        ids=["test_1", "test_2", "test_3"],
        documents=documents,
        embeddings=embeddings,
        metadatas=[
            {"source": "temporary_test"},
            {"source": "temporary_test"},
            {"source": "temporary_test"},
        ],
    )

    # Generate embedding for the search question
    query_embedding = (
        await embedding_service.embed(
            ["Which framework is used to build Python APIs?"]
        )
    )[0]

    # Search ChromaDB
    results = chroma_store.search(
        query_embedding=query_embedding,
        limit=3,
    )

    # Display search results
    for document, distance in zip(
        results["documents"][0],
        results["distances"][0],
    ):
        print(f"Document: {document}")
        print(f"Distance: {distance}")
        print("---")


if __name__ == "__main__":
    asyncio.run(main())