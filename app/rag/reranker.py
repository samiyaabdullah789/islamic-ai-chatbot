class RAGReranker:
    async def rerank(
        self,
        question: str,
        documents: list[dict],
        limit: int = 5,
    ) -> list[dict]:

        if not documents:
            return []

        # ChromaDB se retrieved documents use karenge.
        # Extra Ollama call ki zaroorat nahi.
        # Documents ka existing order preserve hoga.

        ranked_documents = documents[:limit]

        return ranked_documents


rag_reranker = RAGReranker()