import asyncio
import math
import os
import re
import time
import unicodedata
from collections import Counter

from app.embeddings.ollama_embeddings import embedding_service
from app.vector_store.chroma_store import chroma_store
from app.llm.client import ollama_client


class RAGRetriever:
    """Hybrid semantic + lexical retrieval with source-aware RRF fusion.

    No fixed questions, source IDs, hadith IDs, or answers are embedded here.
    """

    QUERY_FILLER_WORDS = {
        "give", "me", "one", "some", "please", "tell", "show",
        "share", "what", "the", "a", "an", "about", "islam", "islamic",
        "who", "is", "are", "was", "were", "of", "in", "to", "among",
        "which", "can", "you", "could", "would", "does", "do", "for",
    }
    STOP_WORDS = QUERY_FILLER_WORDS | {
        "and", "or", "by", "with", "that", "this", "it", "from", "at",
        "as", "be", "on", "a", "an", "the", "his", "her", "their",
    }
    FUSION_K = 60
    SEMANTIC_WEIGHT = 1.0
    LEXICAL_WEIGHT = 1.0
    INDEX_REFRESH_SECONDS = 600
    INDEX_BATCH_SIZE = 1000
    MAX_EXPANSIONS = 2

    def __init__(self):
        self._index = []
        self._document_frequency = Counter()
        self._average_length = 1.0
        self._index_loaded_at = 0.0
        self._index_count = -1
        self._index_lock = asyncio.Lock()

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(unicodedata.normalize("NFKC", str(text)).strip().split())

    @classmethod
    def _tokens(cls, text: str) -> list[str]:
        return re.findall(r"[^\W_]+", cls._normalize(text).casefold(), re.UNICODE)

    def build_search_queries(self, query: str) -> list[str]:
        original = self._normalize(query)
        if not original:
            return []
        simplified = " ".join(
            word for word in self._tokens(original)
            if word not in self.QUERY_FILLER_WORDS
        )
        queries = [original]
        if simplified and len(simplified) >= 5 and simplified.casefold() != original.casefold():
            queries.append(simplified)
        return queries

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
        patterns = (
            r"\b(?:five|5) pillars of islam\b",
            r"\b(?:six|6) pillars of (?:iman|faith)\b",
            r"\b(?:who|what) (?:was|were|is|are) (?:the )?prophet\b",
            r"\b(?:who|what) (?:was|were|is|are) (?:the )?messenger\b",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    @staticmethod
    def _valid_distance(distance: object) -> float:
        if isinstance(distance, (int, float)) and not isinstance(distance, bool) and math.isfinite(distance):
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
            if not str(content or "").strip():
                continue
            metadata = metadatas[index] if index < len(metadatas) else {}
            distance = distances[index] if index < len(distances) else None
            documents.append({
                "id": str(doc_id),
                "content": str(content),
                "metadata": metadata if isinstance(metadata, dict) else {},
                "distance": cls._valid_distance(distance),
                "query_index": query_index,
            })
        return documents

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
        return None if not conditions else conditions[0] if len(conditions) == 1 else {"$and": conditions}

    @staticmethod
    def _matches_filter(metadata: dict, where_filter: dict | None) -> bool:
        if not where_filter:
            return True
        if "$and" in where_filter:
            return all(RAGRetriever._matches_filter(metadata, part) for part in where_filter["$and"])
        return all(metadata.get(key) == value for key, value in where_filter.items())

    async def _expand_queries(self, query: str) -> list[str]:
        """Ask the existing rewrite model for general source-style paraphrases.

        Expansion is optional: errors fall back to the original query.
        The model must never return answers or document identifiers.
        """
        try:
            result = await ollama_client.generate(
                [
                    {"role": "system", "content": (
                        "Rewrite the user question as TWO alternative short English search queries "
                        "for finding relevant passages in a religious text collection. "
                        "Preserve the exact intent, subjects, negation, and legal degree. "
                        "Use natural alternative vocabulary that a source passage might use. "
                        "Do not answer the question, add religious facts, invent references, "
                        "or name a specific document. Output exactly two queries, one per line, "
                        "without numbering or explanations."
                    )},
                    {"role": "user", "content": query},
                ],
                task="rewrite",
            )
            result = re.sub(r"<think>.*?</think>", "", str(result), flags=re.I | re.S)
            expansions = []
            for line in result.splitlines():
                candidate = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip().strip('"\'')
                if (8 <= len(candidate) <= 180 and len(candidate.split()) <= 30
                        and candidate.casefold() != query.casefold()
                        and not any(candidate.casefold() == q.casefold() for q in expansions)):
                    expansions.append(candidate)
                if len(expansions) == self.MAX_EXPANSIONS:
                    break
            return expansions
        except Exception as exc:
            print(f"[RAG] Query expansion skipped: {exc}", flush=True)
            return []

    def _load_index_sync(self) -> tuple[list[dict], Counter, float, int]:
        collection = chroma_store.collection
        total = collection.count()
        records = []
        df = Counter()
        token_total = 0
        for offset in range(0, total, self.INDEX_BATCH_SIZE):
            batch = collection.get(
                limit=self.INDEX_BATCH_SIZE,
                offset=offset,
                include=["documents", "metadatas"],
            )
            ids = batch.get("ids") or []
            docs = batch.get("documents") or []
            metas = batch.get("metadatas") or []
            for i, doc_id in enumerate(ids):
                content = str(docs[i] or "") if i < len(docs) else ""
                if not content.strip():
                    continue
                metadata = metas[i] if i < len(metas) and isinstance(metas[i], dict) else {}
                tokens = [token for token in self._tokens(content) if token not in self.STOP_WORDS]
                counts = Counter(tokens)
                df.update(counts.keys())
                token_total += len(tokens)
                records.append({
                    "id": str(doc_id), "content": content, "metadata": metadata,
                    "counts": counts, "length": len(tokens),
                })
        avg = token_total / len(records) if records else 1.0
        return records, df, avg, total

    async def _ensure_index(self) -> None:
        now = time.monotonic()
        if self._index and now - self._index_loaded_at < self.INDEX_REFRESH_SECONDS:
            return
        async with self._index_lock:
            now = time.monotonic()
            if self._index and now - self._index_loaded_at < self.INDEX_REFRESH_SECONDS:
                return
            start = time.perf_counter()
            records, df, avg, count = await asyncio.to_thread(self._load_index_sync)
            self._index, self._document_frequency = records, df
            self._average_length, self._index_count = avg, count
            self._index_loaded_at = time.monotonic()
            print(f"[RAG] Lexical index: {len(records)} documents (collection={count})", flush=True)
            print(f"[TIMING] Lexical index refresh: {time.perf_counter() - start:.2f}s", flush=True)

    def _lexical_search(self, queries: list[str], limit: int, where_filter: dict | None) -> list[dict]:
        terms = set()
        for query in queries:
            terms.update(token for token in self._tokens(query) if token not in self.STOP_WORDS)
        if not terms or not self._index:
            return []
        n = len(self._index)
        k1, b = 1.5, 0.75
        idfs = {
            token: math.log(1 + (n - self._document_frequency[token] + 0.5)
                            / (self._document_frequency[token] + 0.5))
            for token in terms if self._document_frequency[token]
        }
        ranked = []
        for record in self._index:
            if not self._matches_filter(record["metadata"], where_filter):
                continue
            norm = k1 * (1 - b + b * record["length"] / self._average_length)
            score = sum(
                idf * (record["counts"].get(token, 0) * (k1 + 1))
                / (record["counts"].get(token, 0) + norm)
                for token, idf in idfs.items() if record["counts"].get(token, 0)
            )
            if score > 0:
                ranked.append((score, record))
        ranked.sort(key=lambda item: (-item[0], item[1]["id"]))
        return [{
            "id": record["id"], "content": record["content"],
            "metadata": record["metadata"], "distance": float("inf"),
            "lexical_score": score, "query_index": -1,
        } for score, record in ranked[:limit]]

    async def retrieve(
        self, query: str, limit: int = 20,
        madhhab: str | None = None, general_question: bool = False,
    ) -> list[dict]:
        started = time.perf_counter()
        if limit <= 0:
            return []
        queries = self.build_search_queries(query)
        if not queries:
            return []
        source_id = self._get_test_source_id()
        where_filter = self._build_filter(source_id, madhhab, general_question)
        print(f"[RAG] Test source ID: {source_id or 'All'}", flush=True)
        print(f"[RAG] Selected Madhhab: {madhhab or 'General'}", flush=True)
        print(f"[RAG] ChromaDB filter: {where_filter}", flush=True)

        expansions = await self._expand_queries(queries[0])
        for expansion in expansions:
            if expansion.casefold() not in {q.casefold() for q in queries}:
                queries.append(expansion)

        # Semantic retrieval: keep best distance and rank for each document.
        semantic = {}
        for index, search_query in enumerate(queries):
            print(f"[RAG] Search query {index + 1}: {search_query}", flush=True)
            start = time.perf_counter()
            vectors = await embedding_service.embed([search_query])
            if len(vectors) != 1:
                raise ValueError("Query embedding count mismatch")
            results = chroma_store.search(query_embedding=vectors[0], limit=limit, where=where_filter)
            documents = self._extract_results(results, index)
            print(f"[RAG] Query {index + 1} candidates: {len(documents)}", flush=True)
            print(f"[TIMING] Query {index + 1} retrieval: {time.perf_counter() - start:.2f}s", flush=True)
            for rank, doc in enumerate(documents, start=1):
                existing = semantic.get(doc["id"])
                contribution = 1 / (self.FUSION_K + rank)
                if existing is None:
                    semantic[doc["id"]] = {"document": doc, "score": contribution}
                else:
                    existing["score"] += contribution
                    if doc["distance"] < existing["document"]["distance"]:
                        existing["document"] = doc

        await self._ensure_index()
        lexical = self._lexical_search(queries, limit * 2, where_filter)
        print(f"[RAG] Lexical candidates: {len(lexical)}", flush=True)
        combined = {}
        for doc_id, entry in semantic.items():
            combined[doc_id] = {
                "document": entry["document"],
                "score": self.SEMANTIC_WEIGHT * entry["score"],
            }
        for rank, doc in enumerate(lexical, start=1):
            contribution = self.LEXICAL_WEIGHT / (self.FUSION_K + rank)
            if doc["id"] in combined:
                combined[doc["id"]]["score"] += contribution
            else:
                combined[doc["id"]] = {"document": doc, "score": contribution}

        ranked = sorted(combined.values(), key=lambda entry: (-entry["score"], entry["document"]["id"]))
        # Preserve semantic candidates while admitting high-quality lexical candidates.
        # Return a bounded set for the existing CrossEncoder reranker.
        result = []
        for entry in ranked[:max(limit * 2, limit)]:
            doc = dict(entry["document"])
            doc["retrieval_fusion_score"] = entry["score"]
            result.append(doc)
        print(f"[RAG] Unique retrieved documents: {len(combined)}", flush=True)
        for index, doc in enumerate(result[:5], 1):
            meta = doc["metadata"]
            print(
                f"[RAG] Candidate {index}: source={meta.get('source_name', 'Unknown')} | "
                f"source_id={meta.get('source_id')} | madhhab={meta.get('madhhab', 'N/A')} | "
                f"distance={doc['distance']:.4f}", flush=True,
            )
        print(f"[TIMING] Retriever total: {time.perf_counter() - started:.2f}s", flush=True)
        return result


rag_retriever = RAGRetriever()
