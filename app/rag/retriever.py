
"""Hybrid Chroma semantic + BM25 retrieval using RRF."""

import math
import os
import re
import time
import unicodedata

from app.embeddings.ollama_embeddings import embedding_service
from app.vector_store.chroma_store import chroma_store
from app.rag.bm25_index import bm25_index


class RAGRetriever:

    QUERY_FILLER_WORDS = {
        "give", "me", "please", "tell", "show", "share",
        "explain", "what", "which", "who", "how",
        "does", "do", "did", "the", "a", "an",
        "about", "of", "in", "is", "are",
        "was", "were", "to", "for", "on",
        "and", "according", "regarding",
        "teach", "teaches", "say", "says",
        "islam", "islamic",
    }

    RRF_K = 60
    SEMANTIC_POOL = 40
    KEYWORD_POOL = 40

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(
            unicodedata.normalize(
                "NFKC", str(text)
            ).strip().split()
        )

    def build_search_queries(
        self,
        query: str,
    ) -> list[str]:
        original = self._normalize(query)

        if not original:
            return []

        words = re.findall(
            r"[^\W_]+",
            original,
            flags=re.UNICODE,
        )

        core = [
            word for word in words
            if word.casefold()
            not in self.QUERY_FILLER_WORDS
        ]

        simplified = " ".join(core)

        if (
            len(core) >= 2
            and len(simplified) >= 8
            and simplified.casefold()
            != " ".join(words).casefold()
        ):
            return [original, simplified]

        return [original]

    @staticmethod
    def is_clearly_general_question(
        query: str,
    ) -> bool:
        normalized = " ".join(
            re.findall(
                r"[^\W_]+",
                unicodedata.normalize(
                    "NFKC", str(query)
                ).casefold(),
            )
        )

        if not normalized:
            return False

        fiqh_terms = {
            "permissible", "allowed", "forbidden",
            "haram", "halal", "obligatory",
            "mandatory", "required", "wajib",
            "fard", "farz", "sunnah",
            "mustahabb", "valid", "invalid",
            "ruling", "fatwa", "madhhab",
            "hanafi", "shafi", "maliki",
            "hanbali", "break", "invalidate",
            "conditions", "method",
        }

        if set(normalized.split()) & fiqh_terms:
            return False

        patterns = (
            r"\b(?:five|5) pillars of islam\b",
            r"\b(?:six|6) pillars of (?:iman|faith)\b",
            r"\b(?:who|what) (?:was|were|is|are) "
            r"(?:the )?prophet\b",
            r"\b(?:who|what) (?:was|were|is|are) "
            r"(?:the )?messenger\b",
        )

        return any(
            re.search(pattern, normalized)
            for pattern in patterns
        )

    @staticmethod
    def _valid_distance(
        distance: object,
    ) -> float:
        if (
            isinstance(distance, (int, float))
            and not isinstance(distance, bool)
            and math.isfinite(distance)
        ):
            return float(distance)

        return float("inf")

    @classmethod
    def _extract_results(
        cls,
        results: dict,
        query_index: int,
    ) -> list[dict]:
        if not isinstance(results, dict):
            raise ValueError(
                "Invalid ChromaDB search response"
            )

        ids = (results.get("ids") or [[]])[0] or []
        contents = (
            (results.get("documents") or [[]])[0] or []
        )
        metadatas = (
            (results.get("metadatas") or [[]])[0] or []
        )
        distances = (
            (results.get("distances") or [[]])[0] or []
        )

        documents = []

        for index, doc_id in enumerate(ids):
            content = (
                contents[index]
                if index < len(contents) else ""
            )
            metadata = (
                metadatas[index]
                if index < len(metadatas) else {}
            )
            distance = (
                distances[index]
                if index < len(distances) else None
            )

            if not str(content or "").strip():
                continue

            documents.append({
                "id": str(doc_id),
                "content": str(content),
                "metadata": (
                    metadata
                    if isinstance(metadata, dict) else {}
                ),
                "distance": cls._valid_distance(distance),
                "query_index": query_index,
            })

        return documents

    @staticmethod
    def _get_test_source_id() -> int | None:
        value = os.getenv(
            "RAG_TEST_SOURCE_ID", ""
        ).strip()

        if not value:
            return None

        source_id = int(value)

        if source_id <= 0:
            raise ValueError(
                "RAG_TEST_SOURCE_ID must be positive"
            )

        return source_id

    @staticmethod
    def _build_filter(
        source_id: int | None,
        madhhab: str | None,
        general_question: bool,
    ) -> dict | None:
        conditions = []

        if source_id is not None:
            conditions.append({
                "source_id": source_id
            })

        if (
            madhhab
            and madhhab.casefold() != "general"
            and not general_question
        ):
            conditions.append({
                "madhhab": madhhab
            })

        if not conditions:
            return None

        if len(conditions) == 1:
            return conditions[0]

        return {"$and": conditions}

    @staticmethod
    def _fuse(
        rankings: list[list[dict]],
        limit: int,
        k: int,
    ) -> list[dict]:
        scores = {}
        documents = {}

        for ranking in rankings:
            seen_in_ranking = set()

            for rank, document in enumerate(
                ranking, start=1
            ):
                doc_id = document["id"]

                if doc_id in seen_in_ranking:
                    continue

                seen_in_ranking.add(doc_id)

                scores[doc_id] = (
                    scores.get(doc_id, 0.0)
                    + 1.0 / (k + rank)
                )

                previous = documents.get(doc_id)

                if (
                    previous is None
                    or document.get(
                        "distance", float("inf")
                    )
                    < previous.get(
                        "distance", float("inf")
                    )
                ):
                    documents[doc_id] = document

        ordered = sorted(
            scores,
            key=lambda doc_id: (
                -scores[doc_id],
                doc_id,
            ),
        )[:limit]

        return [
            {
                **documents[doc_id],
                "retrieval_score": scores[doc_id],
            }
            for doc_id in ordered
        ]

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

        source_id = self._get_test_source_id()

        where_filter = self._build_filter(
            source_id,
            madhhab,
            general_question,
        )

        print(
            f"[RAG] Test source ID: "
            f"{source_id or 'All'}",
            flush=True,
        )
        print(
            f"[RAG] Selected Madhhab: "
            f"{madhhab or 'General'}",
            flush=True,
        )
        print(
            f"[RAG] ChromaDB filter: {where_filter}",
            flush=True,
        )

        rankings = []

        semantic_pool = max(
            limit, self.SEMANTIC_POOL
        )
        keyword_pool = max(
            limit, self.KEYWORD_POOL
        )

        for query_index, search_query in enumerate(
            queries
        ):
            print(
                f"[RAG] Search query "
                f"{query_index + 1}: {search_query}",
                flush=True,
            )

            start = time.perf_counter()

            vectors = await embedding_service.embed(
                [search_query]
            )

            if len(vectors) != 1:
                raise ValueError(
                    "Query embedding count mismatch"
                )

            print(
                f"[TIMING] Query {query_index + 1} "
                f"embedding: "
                f"{time.perf_counter() - start:.2f}s",
                flush=True,
            )

            start = time.perf_counter()

            results = chroma_store.search(
                query_embedding=vectors[0],
                limit=semantic_pool,
                where=where_filter,
            )

            semantic = self._extract_results(
                results,
                query_index,
            )

            rankings.append(
                sorted(
                    semantic,
                    key=lambda d: (
                        d["distance"],
                        d["id"],
                    ),
                )
            )

            print(
                f"[TIMING] Query {query_index + 1} "
                f"ChromaDB search: "
                f"{time.perf_counter() - start:.2f}s",
                flush=True,
            )

            print(
                f"[RAG] Query {query_index + 1} "
                f"semantic candidates: {len(semantic)}",
                flush=True,
            )

            start = time.perf_counter()

            try:
                lexical = bm25_index.search(
                    search_query,
                    limit=keyword_pool,
                    source_id=source_id,
                    madhhab=madhhab,
                    general_question=general_question,
                )

                rankings.append(lexical)

                print(
                    f"[RAG] Query {query_index + 1} "
                    f"BM25 candidates: {len(lexical)}",
                    flush=True,
                )

            except Exception as error:
                print(
                    "[RAG WARNING] BM25 unavailable; "
                    f"using semantic only: {error}",
                    flush=True,
                )

            print(
                f"[TIMING] Query {query_index + 1} "
                f"BM25 search: "
                f"{time.perf_counter() - start:.2f}s",
                flush=True,
            )

        documents = self._fuse(
            rankings,
            limit=max(
                limit,
                semantic_pool + keyword_pool,
            ),
            k=self.RRF_K,
        )

        print(
            f"[RAG] Unique retrieved documents: "
            f"{len(documents)}",
            flush=True,
        )

        for index, document in enumerate(
            documents[:5], start=1
        ):
            metadata = document["metadata"]

            print(
                f"[RAG] Candidate {index}: "
                f"source={metadata.get('source_name', 'Unknown')} | "
                f"source_id={metadata.get('source_id')} | "
                f"madhhab={metadata.get('madhhab', 'N/A')} | "
                f"distance={document['distance']:.4f} | "
                f"RRF={document['retrieval_score']:.6f}",
                flush=True,
            )

        print(
            f"[TIMING] Retriever total: "
            f"{time.perf_counter() - total_start:.2f}s",
            flush=True,
        )

        return documents


rag_retriever = RAGRetriever()
