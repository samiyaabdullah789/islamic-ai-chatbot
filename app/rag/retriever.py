
import re

from app.embeddings.ollama_embeddings import embedding_service
from app.vector_store.chroma_store import chroma_store


class RAGRetriever:

    def build_search_queries(self, query: str) -> list[str]:

        queries = [query]
        normalized = query.lower()

        prayer_terms = {
            "namaz", "salah", "salat",
            "prayer", "prayers",
        }

        words = set(re.findall(r"\b\w+\b", normalized))

        if words.intersection(prayer_terms):

            if words.intersection({
                "importance", "virtue", "virtues",
                "benefit", "benefits", "reward", "rewards",
            }):
                queries.extend([
                    "virtues and rewards of performing salah prayer",
                    "importance and benefits of daily obligatory prayers",
                ])

        # Remove duplicate queries
        return list(dict.fromkeys(queries))

    async def retrieve(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict]:

        # 1. Prepare search queries
        search_queries = self.build_search_queries(query)

        # 2. Generate embeddings in one batch
        embeddings = await embedding_service.embed(
            search_queries
        )

        documents_by_id = {}

        # 3. Search ChromaDB for each query
        for query_embedding in embeddings:

            results = chroma_store.search(
                query_embedding=query_embedding,
                limit=limit,
            )

            ids = results.get("ids", [[]])[0]
            contents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for doc_id, content, metadata, distance in zip(
                ids,
                contents,
                metadatas,
                distances,
            ):

                document = {
                    "id": doc_id,
                    "content": content,
                    "metadata": metadata,
                    "distance": distance,
                }

                # Keep best distance for duplicate documents
                existing = documents_by_id.get(doc_id)

                if (
                    existing is None
                    or distance < existing["distance"]
                ):
                    documents_by_id[doc_id] = document

        # 4. Sort by semantic similarity
        documents = sorted(
            documents_by_id.values(),
            key=lambda doc: doc["distance"],
        )

        return documents


rag_retriever = RAGRetriever()
