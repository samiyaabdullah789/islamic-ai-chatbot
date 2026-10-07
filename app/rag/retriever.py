from app.embeddings.ollama_embeddings import embedding_service
from app.vector_store.chroma_store import chroma_store


class RAGRetriever:

    async def retrieve(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict]:

        # Convert user question into an embedding
        embeddings = await embedding_service.embed([query])
        query_embedding = embeddings[0]

        # Search relevant documents in ChromaDB
        results = chroma_store.search(
            query_embedding=query_embedding,
            limit=limit,
        )

        # Format retrieved documents
        documents = []

        for doc_id, content, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            documents.append(
                {
                    "id": doc_id,
                    "content": content,
                    "metadata": metadata,
                    "distance": distance,
                }
            )

        return documents


rag_retriever = RAGRetriever()