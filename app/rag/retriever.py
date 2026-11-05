import math
import os
import re
import time
import unicodedata

from app.embeddings.ollama_embeddings import embedding_service
from app.vector_store.chroma_store import chroma_store


class RAGRetriever:
    """Retrieve diverse ChromaDB candidates for the existing reranker.

    Embedding inputs are intentionally unchanged. Adding a query prefix
    without checking how stored documents were embedded can reduce recall.
    """

    QUERY_FILLER_WORDS = {
        "give", "me", "one", "some", "please", "tell", "show",
        "share", "what", "the", "a", "an", "about", "islam", "islamic",
    }

    @staticmethod
    def _normalize(text: str) -> str:
        normalized = unicodedata.normalize("NFKC", str(text))
        return " ".join(normalized.strip().split())

    def build_search_queries(self, query: str) -> list[str]:
        original = self._normalize(query)
        if not original:
            return []

        words = re.findall(r"[^\W_]+", original, flags=re.UNICODE)
        simplified = " ".join(
            word for word in words
            if word.casefold() not in self.QUERY_FILLER_WORDS
        ).strip()

        queries = [original]
        if (
            simplified
            and len(simplified) >= 5
            and simplified.casefold() != original.casefold()
        ):
            queries.append(simplified)
        return queries

    @staticmethod
    def _valid_distance(distance: object) -> float:
        if (
            isinstance(distance, (int, float))
            and not isinstance(distance, bool)
            and math.isfinite(distance)
        ):
            return float(distance)
        return float("inf")

    @classmethod
    def _extract_results(cls, results: dict, query_index: int) -> list[dict]:
        if not isinstance(results, dict):
            raise ValueError("Invalid ChromaDB search response")

        ids = (results.get("ids") or [[]])[0] or []
        contents = (results.get("documents") or [[]])[0] or []
        metadatas = (results.get("metadatas") or [[]])[0] or []
        distances = (results.get("distances") or [[]])[0] or []

        documents = []
        for index, doc_id in enumerate(ids):
            content = contents[index] if index < len(contents) else ""
            metadata = metadatas[index] if index < len(metadatas) else {}
            distance = distances[index] if index < len(distances) else None

            if not str(content or "").strip():
                continue

            documents.append({
                "id": str(doc_id),
                "content": str(content),
                "metadata": metadata if isinstance(metadata, dict) else {},
                "distance": cls._valid_distance(distance),
                "query_index": query_index,
            })
        return documents

    @staticmethod
    def _merge_documents(
        documents_by_id: dict[str, dict],
        new_documents: list[dict],
    ) -> None:
        for document in new_documents:
            doc_id = document["id"]
            previous = documents_by_id.get(doc_id)
            if previous is None or document["distance"] < previous["distance"]:
                documents_by_id[doc_id] = document

    @staticmethod
    def _get_test_source_id() -> int | None:
        value = os.getenv("RAG_TEST_SOURCE_ID", "").strip()
        if not value:
            return None
        source_id = int(value)
        if source_id <= 0:
            raise ValueError("RAG_TEST_SOURCE_ID must be positive")
        return source_id

    @staticmethod
    def _build_filter(
        source_id: int | None,
        madhhab: str | None,
        general_question: bool,
    ) -> dict | None:
        conditions = []
        if source_id is not None:
            conditions.append({"source_id": source_id})
        if madhhab and not general_question:
            conditions.append({"madhhab": madhhab})
        if not conditions:
            return None
        return conditions[0] if len(conditions) == 1 else {"$and": conditions}

    async def retrieve(
        self,
        query: str,
        limit: int = 20,
        madhhab: str | None = None,
        general_question: bool = False,
    ) -> list[dict]:
        total_start = time.perf_counter()
        if limit <= 0:
            return []

        queries = self.build_search_queries(query)
        if not queries:
            return []

        test_source_id = self._get_test_source_id()
        where_filter = self._build_filter(
            test_source_id, madhhab, general_question
        )

        print(f"[RAG] Test source ID: {test_source_id or 'All'}", flush=True)
        print(f"[RAG] Selected Madhhab: {madhhab or 'General'}", flush=True)
        print(f"[RAG] ChromaDB filter: {where_filter}", flush=True)

        documents_by_id: dict[str, dict] = {}

        for query_index, search_query in enumerate(queries):
            print(
                f"[RAG] Search query {query_index + 1}: {search_query}",
                flush=True,
            )
            start = time.perf_counter()
            vectors = await embedding_service.embed([search_query])
            if len(vectors) != 1:
                raise ValueError("Query embedding count mismatch")
            print(
                f"[TIMING] Query {query_index + 1} embedding: "
                f"{time.perf_counter() - start:.2f}s",
                flush=True,
            )

            start = time.perf_counter()
            results = chroma_store.search(
                query_embedding=vectors[0],
                limit=limit,
                where=where_filter,
            )
            documents = self._extract_results(results, query_index)
            self._merge_documents(documents_by_id, documents)
            print(
                f"[TIMING] Query {query_index + 1} ChromaDB search: "
                f"{time.perf_counter() - start:.2f}s",
                flush=True,
            )
            print(
                f"[RAG] Query {query_index + 1} candidates: {len(documents)}",
                flush=True,
            )

        # Keep candidates from both formulations for the reranker.
        documents = sorted(
            documents_by_id.values(),
            key=lambda item: (item["distance"], item["id"]),
        )

        if test_source_id is not None:
            documents = [
                document for document in documents
                if str(document["metadata"].get("source_id")) == str(test_source_id)
            ]

        print(f"[RAG] Unique retrieved documents: {len(documents)}", flush=True)
        for index, document in enumerate(documents[:5], start=1):
            metadata = document["metadata"]
            print(
                f"[RAG] Candidate {index}: "
                f"source={metadata.get('source_name', 'Unknown')} | "
                f"source_id={metadata.get('source_id')} | "
                f"madhhab={metadata.get('madhhab', 'N/A')} | "
                f"distance={document['distance']:.4f}",
                flush=True,
            )
        print(
            f"[TIMING] Retriever total: {time.perf_counter() - total_start:.2f}s",
            flush=True,
        )
        return documents


rag_retriever = RAGRetriever()
