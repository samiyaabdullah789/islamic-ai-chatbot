from app.llm.client import ollama_client


class RAGReranker:
    async def rerank(
        self,
        question: str,
        documents: list[dict],
        limit: int = 5,
    ) -> list[dict]:

        if not documents:
            return []

        candidates = "\n\n".join(
            f"[{index}]\n{document['content']}"
            for index, document in enumerate(documents)
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a relevance reranker for an Islamic knowledge "
                    "retrieval system. Your only task is to rank the provided "
                    "source passages by how directly they answer the user's "
                    "question.\n\n"
                    "Return only the passage numbers in best-to-worst order, "
                    "separated by commas.\n"
                    "Do not answer the question.\n"
                    "Do not explain your ranking.\n"
                    "Do not invent passage numbers."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Passages:\n{candidates}"
                ),
            },
        ]

        response = await ollama_client.generate(messages)

        ranked_indexes = []

        for value in response.strip().split(","):
            value = value.strip()

            if value.isdigit():
                index = int(value)

                if 0 <= index < len(documents):
                    ranked_indexes.append(index)

        ranked_documents = [
            documents[index]
            for index in ranked_indexes
        ]

        return ranked_documents[:limit]


rag_reranker = RAGReranker()