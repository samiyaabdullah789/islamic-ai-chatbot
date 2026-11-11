import math
import os
import re
import time
import unicodedata

from app.embeddings.ollama_embeddings import embedding_service
from app.vector_store.chroma_store import chroma_store


class RAGRetriever:
    """Semantic retrieval with safe query simplification and existing filters."""

    QUERY_FILLER_WORDS = {
        "give", "me", "please", "tell", "show", "share", "explain",
        "what", "which", "who", "how", "does", "do", "did", "the",
        "a", "an", "about", "of", "in", "is", "are", "was", "were",
        "to", "for", "on", "and", "according", "regarding", "teach",
        "teaches", "say", "says", "islam", "islamic",
    }

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(unicodedata.normalize("NFKC", str(text)).strip().split())

    def build_search_queries(self, query: str) -> list[str]:
        original = self._normalize(query)
        if not original:
            return []
        words = re.findall(r"[^\W_]+", original, flags=re.UNICODE)
        core = [word for word in words if word.casefold() not in self.QUERY_FILLER_WORDS]
        # Do not issue a second embedding/search for a single vague token.
        # The original question is always searched without modification.
        simplified = " ".join(core)
        if (
            len(core) >= 2
            and len(simplified) >= 8
            and simplified.casefold() != " ".join(words).casefold()
        ):
            return [original, simplified]
        return [original]

    @staticmethod
    def is_clearly_general_question(query: str) -> bool:
        normalized = " ".join(re.findall(
            r"[^\W_]+", unicodedata.normalize("NFKC", str(query)).casefold()
        ))
        if not normalized:
            return False
        fiqh_terms = {
            "permissible", "allowed", "forbidden", "haram", "halal",
            "obligatory", "mandatory", "required", "wajib", "fard",
            "farz", "sunnah", "mustahabb", "valid", "invalid",
            "ruling", "fatwa", "madhhab", "hanafi", "shafi", "maliki",
            "hanbali", "break", "invalidate", "conditions", "method",
        }
        if set(normalized.split()) & fiqh_terms:
            return False
        general_patterns = (
            r"\b(?:five|5) pillars of islam\b",
            r"\b(?:six|6) pillars of (?:iman|faith)\b",
            r"\b(?:who|what) (?:was|were|is|are) (?:the )?prophet\b",
            r"\b(?:who|what) (?:was|were|is|are) (?:the )?messenger\b",
        )
        return any(re.search(pattern, normalized) for pattern in general_patterns)

    @staticmethod
    def _valid_distance(distance: object) -> float:
        if (isinstance(distance, (int, float)) and not isinstance(distance, bool)
                and math.isfinite(distance)):
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
    def _merge_documents(documents_by_id: dict[str, dict], new_documents: list[dict]) -> None:
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
    def _build_filter(source_id: int | None, madhhab: str | None, general_question: bool) -> dict | None:
        conditions = []
        if source_id is not None:
            conditions.append({"source_id": source_id})
        if madhhab and madhhab.casefold() != "general" and not general_question:
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
        where_filter = self._build_filter(test_source_id, madhhab, general_question)
        print(f"[RAG] Test source ID: {test_source_id or 'All'}", flush=True)
        print(f"[RAG] Selected Madhhab: {madhhab or 'General'}", flush=True)
        print(f"[RAG] ChromaDB filter: {where_filter}", flush=True)
        documents_by_id: dict[str, dict] = {}
        for query_index, search_query in enumerate(queries):
            print(f"[RAG] Search query {query_index + 1}: {search_query}", flush=True)
            start = time.perf_counter()
            vectors = await embedding_service.embed([search_query])
            if len(vectors) != 1:
                raise ValueError("Query embedding count mismatch")
            print(f"[TIMING] Query {query_index + 1} embedding: {time.perf_counter() - start:.2f}s", flush=True)
            start = time.perf_counter()
            results = chroma_store.search(
                query_embedding=vectors[0], limit=limit, where=where_filter,
            )
            documents = self._extract_results(results, query_index)
            self._merge_documents(documents_by_id, documents)
            print(f"[TIMING] Query {query_index + 1} ChromaDB search: {time.perf_counter() - start:.2f}s", flush=True)
            print(f"[RAG] Query {query_index + 1} candidates: {len(documents)}", flush=True)
        documents = sorted(documents_by_id.values(), key=lambda item: (item["distance"], item["id"]))
        if test_source_id is not None:
            documents = [document for document in documents if str(document["metadata"].get("source_id")) == str(test_source_id)]
        print(f"[RAG] Unique retrieved documents: {len(documents)}", flush=True)
        for index, document in enumerate(documents[:5], start=1):
            metadata = document["metadata"]
            print(
                f"[RAG] Candidate {index}: source={metadata.get('source_name', 'Unknown')} | "
                f"source_id={metadata.get('source_id')} | madhhab={metadata.get('madhhab', 'N/A')} | "
                f"distance={document['distance']:.4f}", flush=True,
            )
        print(f"[TIMING] Retriever total: {time.perf_counter() - total_start:.2f}s", flush=True)
        return documents


rag_retriever = RAGRetriever()
