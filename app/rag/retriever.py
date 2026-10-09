
import re

from app.embeddings.ollama_embeddings import embedding_service
from app.vector_store.chroma_store import chroma_store


class RAGRetriever:

    @staticmethod
    def _words(text: str) -> set[str]:
        return set(re.findall(r"\b\w+\b", text.lower()))

    def build_search_queries(self, query: str) -> list[str]:
        queries = [query]
        words = self._words(query)

        prayer = {
            "namaz", "salah", "salat", "prayer", "prayers"
        }
        virtues = {
            "importance", "virtue", "virtues",
            "reward", "rewards", "benefit", "benefits"
        }
        zakat = {"zakat", "zakah"}
        definition = {
            "what", "meaning", "definition", "define"
        }

        if words & prayer and words & virtues:
            queries.append(
                "reward of congregational prayer salah mosque"
            )

        elif words & zakat and words & definition:
            queries.append(
                "Zakat obligatory charity duty of Islam "
                "payment of Zakat poor needy"
            )

        return list(dict.fromkeys(queries))

    async def retrieve(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict]:

        search_queries = self.build_search_queries(query)

        embeddings = await embedding_service.embed(
            search_queries
        )

        if len(embeddings) != len(search_queries):
            raise ValueError(
                "Embedding count does not match search queries"
            )

        documents_by_id = {}

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
                ids, contents, metadatas, distances
            ):
                document = {
                    "id": doc_id,
                    "content": content or "",
                    "metadata": metadata or {},
                    "distance": distance,
                }

                previous = documents_by_id.get(doc_id)

                if (
                    previous is None
                    or (
                        isinstance(distance, (int, float))
                        and (
                            not isinstance(
                                previous["distance"],
                                (int, float),
                            )
                            or distance < previous["distance"]
                        )
                    )
                ):
                    documents_by_id[doc_id] = document

        return sorted(
            documents_by_id.values(),
            key=lambda doc: (
                doc["distance"]
                if isinstance(
                    doc["distance"], (int, float)
                )
                else float("inf")
            ),
        )


rag_retriever = RAGRetriever()
