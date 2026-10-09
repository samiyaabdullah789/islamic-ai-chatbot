import re

from app.embeddings.ollama_embeddings import embedding_service
from app.vector_store.chroma_store import chroma_store


class RAGRetriever:
    @staticmethod
    def _words(text: str) -> set[str]:
        return set(re.findall(r"\b\w+\b", text.lower()))

    def build_search_queries(self, query: str) -> list[str]:
        # One semantic query is the default. A second, broader query helps
        # with everyday wording without adding topic-specific shortcuts.
        query = query.strip()
        simplified = re.sub(
            r"\b(give|me|one|some|please|tell|show|share|what|does|is|the|a|an|about|in|islam|islamic|hadith|hadees|hadeeth|hadis)\b",
            " ", query, flags=re.IGNORECASE,
        )
        simplified = " ".join(simplified.split())
        queries = [query]
        if simplified and simplified.casefold() != query.casefold() and len(simplified) >= 5:
            queries.append(simplified)
        return queries

    async def retrieve(self, query: str, limit: int = 10) -> list[dict]:
        queries = self.build_search_queries(query)
        embeddings = await embedding_service.embed(queries)
        if len(embeddings) != len(queries):
            raise ValueError("Embedding count does not match search queries")

        documents_by_id: dict = {}
        for embedding in embeddings:
            results = chroma_store.search(query_embedding=embedding, limit=limit)
            ids = results.get("ids", [[]])[0]
            contents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]
            for doc_id, content, metadata, distance in zip(ids, contents, metadatas, distances):
                if not content or not str(content).strip():
                    continue
                doc = {
                    "id": doc_id,
                    "content": content,
                    "metadata": metadata or {},
                    "distance": distance,
                }
                previous = documents_by_id.get(doc_id)
                if previous is None or (
                    isinstance(distance, (int, float))
                    and (not isinstance(previous["distance"], (int, float))
                         or distance < previous["distance"])
                ):
                    documents_by_id[doc_id] = doc

        return sorted(
            documents_by_id.values(),
            key=lambda doc: doc["distance"] if isinstance(doc["distance"], (int, float)) else float("inf"),
        )


rag_retriever = RAGRetriever()