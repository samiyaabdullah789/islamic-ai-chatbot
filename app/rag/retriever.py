
import re

from app.embeddings.ollama_embeddings import embedding_service
from app.vector_store.chroma_store import chroma_store


class RAGRetriever:

    def build_search_queries(self, query: str) -> list[str]:
        queries = [query]
        words = set(re.findall(r"\b\w+\b", query.lower()))

        prayer_words = {
            "namaz", "salah", "salat", "prayer", "prayers"
        }
        virtue_words = {
            "importance", "virtue", "virtues",
            "reward", "rewards", "benefit", "benefits"
        }

        if (
            words.intersection(prayer_words)
            and words.intersection(virtue_words)
        ):
            queries.append(
                "reward virtues blessings of congregational "
                "prayer salah mosque"
            )

        return list(dict.fromkeys(queries))

    async def retrieve(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict]:

        search_queries = self.build_search_queries(query)
        embeddings = await embedding_service.embed(search_queries)

        documents_by_id = {}

        for query_embedding in embeddings:
            results = chroma_store.search(
                query_embedding=query_embedding,
                limit=limit,
            )

            for doc_id, content, metadata, distance in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                document = {
                    "id": doc_id,
                    "content": content,
                    "metadata": metadata or {},
                    "distance": distance,
                }

                previous = documents_by_id.get(doc_id)

                if (
                    previous is None
                    or distance < previous["distance"]
                ):
                    documents_by_id[doc_id] = document

        return sorted(
            documents_by_id.values(),
            key=lambda item: item["distance"],
        )


rag_retriever = RAGRetriever()
