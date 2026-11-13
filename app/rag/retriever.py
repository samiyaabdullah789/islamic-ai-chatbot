
"""Hybrid Chroma semantic + BM25 retrieval using weighted RRF."""

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

    # Search terminology, not religious rulings or fixed answers.
    # Each group contains closely related search expressions.
    SYNONYM_GROUPS = (
        ("namaz", "salah", "salat", "prayer"),
        ("roza", "sawm", "siyam", "fasting"),
        ("zakat", "zakah", "almsgiving"),
        ("wudu", "wudhu", "wuzu", "ablution"),
        ("ghusl", "ritual bath"),
        ("backbiting", "gheebah", "gheebat"),
        ("anger", "angry", "wrath"),
        ("hypocrite", "hypocrisy", "munafiq"),
    )

    RRF_K = 60
    SEMANTIC_POOL = 40
    KEYWORD_POOL = 40

    # Original question should remain the strongest signal.
    ORIGINAL_QUERY_WEIGHT = 1.0
    SIMPLIFIED_QUERY_WEIGHT = 0.75
    EXPANDED_QUERY_WEIGHT = 0.55

    # Prevent too many embedding and Chroma calls.
    MAX_SEARCH_QUERIES = 3

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(
            unicodedata.normalize(
                "NFKC", str(text or "")
            ).strip().split()
        )

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return re.findall(
            r"[^\W_]+",
            unicodedata.normalize(
                "NFKC", str(text or "")
            ).casefold(),
            flags=re.UNICODE,
        )

    @classmethod
    def _expand_terms(cls, query: str) -> str:
        """
        Expand recognized terminology without changing the
        original question or adding factual answers.
        """
        normalized = cls._normalize(query)
        words = set(cls._tokens(normalized))

        additions = []

        for group in cls.SYNONYM_GROUPS:
            matched = False

            for expression in group:
                expression_words = cls._tokens(expression)

                if len(expression_words) == 1:
                    if expression_words[0] in words:
                        matched = True
                        break
                else:
                    pattern = (
                        r"(?<!\w)"
                        + re.escape(expression.casefold())
                        + r"(?!\w)"
                    )
                    if re.search(
                        pattern,
                        normalized.casefold(),
                    ):
                        matched = True
                        break

            if not matched:
                continue

            for expression in group:
                if expression.casefold() not in {
                    item.casefold() for item in additions
                }:
                    additions.append(expression)

        if not additions:
            return ""

        # Focus on the user's original terms plus a small
        # number of related expressions.
        core = [
            word
            for word in cls._tokens(normalized)
            if word not in cls.QUERY_FILLER_WORDS
        ]

        expanded = " ".join(
            core + additions[:8]
        )

        return cls._normalize(expanded)

    def build_search_queries(
        self,
        query: str,
    ) -> list[str]:
        original = self._normalize(query)

        if not original:
            return []

        words = self._tokens(original)

        core = [
            word
            for word in words
            if word not in self.QUERY_FILLER_WORDS
        ]

        simplified = " ".join(core)
        expanded = self._expand_terms(original)

        candidates = [
            original,
            simplified
            if len(core) >= 2
            and len(simplified) >= 8
            else "",
            expanded,
        ]

        queries = []
        seen = set()

        for candidate in candidates:
            candidate = self._normalize(candidate)
            key = candidate.casefold()

            if not candidate or key in seen:
                continue

            seen.add(key)
            queries.append(candidate)

            if len(queries) >= self.MAX_SEARCH_QUERIES:
                break

        return queries

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
        weights: list[float] | None = None,
    ) -> list[dict]:

        scores = {}
        documents = {}

        if weights is None:
            weights = [1.0] * len(rankings)

        if len(weights) != len(rankings):
            raise ValueError(
                "Rankings and weights count mismatch"
            )

        for ranking, weight in zip(rankings, weights):
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
                    + weight / (k + rank)
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
        print(
            f"[RAG] Search queries: {queries}",
            flush=True,
        )

        rankings = []
        weights = []

        semantic_pool = max(
            limit, self.SEMANTIC_POOL
        )
        keyword_pool = max(
            limit, self.KEYWORD_POOL
        )

        original_tokens = self._tokens(query)

        simplified_tokens = [
            word
            for word in original_tokens
            if word not in self.QUERY_FILLER_WORDS
        ]

        simplified_text = " ".join(simplified_tokens)

        for query_index, search_query in enumerate(
            queries
        ):

            if query_index == 0:
                query_weight = self.ORIGINAL_QUERY_WEIGHT
                query_type = "original"
            elif (
                search_query.casefold()
                == simplified_text.casefold()
            ):
                query_weight = self.SIMPLIFIED_QUERY_WEIGHT
                query_type = "simplified"
            else:
                query_weight = self.EXPANDED_QUERY_WEIGHT
                query_type = "expanded"

            print(
                f"[RAG] Search query {query_index + 1} "
                f"({query_type}, weight={query_weight}): "
                f"{search_query}",
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
            weights.append(query_weight)

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
                weights.append(query_weight)

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
            weights=weights,
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

