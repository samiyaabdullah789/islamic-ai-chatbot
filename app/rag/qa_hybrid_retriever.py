
import re
import time

from rank_bm25 import BM25Okapi

from app.embeddings.ollama_embeddings import embedding_service
from app.vector_store.chroma_client import chroma_client


class QAHybridRetriever:

    def __init__(self):
        self.collection = chroma_client.qa_collection

        self.rrf_k = 60

        # Retrieval weights (not hardcoded domain rules)
        self.dense_weight = 2.0
        self.bm25_weight = 1.0

        self._cached_ids = []
        self._cached_documents = []
        self._cached_metadatas = []
        self._record_lookup = {}

        self._bm25 = None
        self._cache_count = -1

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return re.findall(
            r"\w+",
            text.lower(),
            flags=re.UNICODE,
        )

    def _load_bm25_index(self):
        start = time.perf_counter()

        count = self.collection.count()

        if (
            count == self._cache_count
            and self._bm25 is not None
        ):
            return

        ids = []
        documents = []
        metadatas = []

        offset = 0
        batch_size = 500

        while offset < count:
            results = self.collection.get(
                limit=batch_size,
                offset=offset,
                include=["documents", "metadatas"],
            )

            batch_ids = results.get("ids", [])

            if not batch_ids:
                break

            ids.extend(batch_ids)
            documents.extend(
                results.get("documents", [])
            )
            metadatas.extend(
                results.get("metadatas", [])
            )

            offset += len(batch_ids)

        tokenized_corpus = [
            self.tokenize(document or "")
            for document in documents
        ]

        bm25 = (
            BM25Okapi(tokenized_corpus)
            if tokenized_corpus
            else None
        )

        self._cached_ids = ids
        self._cached_documents = documents
        self._cached_metadatas = metadatas
        self._bm25 = bm25

        self._record_lookup = {}

        for document_id, metadata in zip(
            ids,
            metadatas,
        ):
            metadata = metadata or {}

            self._record_lookup[document_id] = {
                "id": document_id,
                "question": metadata.get(
                    "question", ""
                ),
                "answer": metadata.get(
                    "answer", ""
                ),
                "source_id": metadata.get(
                    "source_id"
                ),
                "source_name": metadata.get(
                    "source_name", ""
                ),
                "original_record_id": metadata.get(
                    "original_record_id", ""
                ),
                "madhhab": metadata.get(
                    "madhhab", ""
                ),
                "category": metadata.get(
                    "category", ""
                ),
                "tags": metadata.get(
                    "tags", ""
                ),
            }

        self._cache_count = len(ids)

        print(
            f"[HYBRID] BM25 indexed records: "
            f"{self._cache_count}",
            flush=True,
        )

        print(
            "[TIMING] BM25 index build: "
            f"{time.perf_counter() - start:.2f}s",
            flush=True,
        )

    @staticmethod
    def _matches_madhhab(
        metadata: dict,
        madhhab: str | None,
    ) -> bool:

        if not madhhab:
            return True

        requested = madhhab.strip().lower()

        if requested == "general":
            return True

        record_madhhab = str(
            metadata.get("madhhab", "")
        ).strip().lower()

        return (
            not record_madhhab
            or record_madhhab == "general"
            or requested in record_madhhab
        )

    def _matches_filters(
        self,
        record: dict,
        madhhab: str | None,
        source_id: int | None,
    ) -> bool:

        if (
            source_id is not None
            and record["source_id"] != source_id
        ):
            return False

        return self._matches_madhhab(
            record,
            madhhab,
        )

    async def retrieve(
        self,
        query: str,
        limit: int = 5,
        dense_limit: int = 30,
        bm25_limit: int = 30,
        madhhab: str | None = None,
        source_id: int | None = None,
    ) -> list[dict]:

        if not query.strip() or limit <= 0:
            return []

        self._load_bm25_index()

        if not self._cached_ids:
            return []

        total_start = time.perf_counter()

        # -----------------------------------
        # 1. Dense retrieval
        # -----------------------------------
        start = time.perf_counter()

        query_embeddings = await embedding_service.embed(
            [query]
        )

        where_filter = None

        if source_id is not None:
            where_filter = {
                "source_id": source_id
            }

        dense_results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=min(
                max(dense_limit, limit),
                len(self._cached_ids),
            ),
            where=where_filter,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        dense_ids = dense_results["ids"][0]

        dense_ids = [
            document_id
            for document_id in dense_ids
            if self._matches_filters(
                self._record_lookup[document_id],
                madhhab,
                source_id,
            )
        ]

        print(
            "[TIMING] Hybrid dense search: "
            f"{time.perf_counter() - start:.2f}s",
            flush=True,
        )

        # -----------------------------------
        # 2. BM25 retrieval
        # -----------------------------------
        start = time.perf_counter()

        bm25_scores = self._bm25.get_scores(
            self.tokenize(query)
        )

        ranked_indices = sorted(
            range(len(bm25_scores)),
            key=lambda index: bm25_scores[index],
            reverse=True,
        )

        bm25_ids = []

        for index in ranked_indices:
            if bm25_scores[index] <= 0:
                break

            document_id = self._cached_ids[index]
            record = self._record_lookup[document_id]

            if not self._matches_filters(
                record,
                madhhab,
                source_id,
            ):
                continue

            bm25_ids.append(document_id)

            if len(bm25_ids) >= bm25_limit:
                break

        print(
            "[TIMING] Hybrid BM25 search: "
            f"{time.perf_counter() - start:.2f}s",
            flush=True,
        )

        # -----------------------------------
        # 3. Weighted RRF
        # -----------------------------------
        start = time.perf_counter()

        fused_scores = {}

        for rank, document_id in enumerate(
            dense_ids,
            start=1,
        ):
            fused_scores[document_id] = (
                fused_scores.get(document_id, 0.0)
                + self.dense_weight
                / (self.rrf_k + rank)
            )

        for rank, document_id in enumerate(
            bm25_ids,
            start=1,
        ):
            fused_scores[document_id] = (
                fused_scores.get(document_id, 0.0)
                + self.bm25_weight
                / (self.rrf_k + rank)
            )

        ranked_ids = sorted(
            fused_scores,
            key=fused_scores.get,
            reverse=True,
        )

        final_results = []

        for document_id in ranked_ids[:limit]:
            record = self._record_lookup.get(
                document_id
            )

            if record is None:
                continue

            final_results.append({
                **record,
                "rrf_score": fused_scores[document_id],
            })

        print(
            "[TIMING] Hybrid fusion: "
            f"{time.perf_counter() - start:.2f}s",
            flush=True,
        )

        print(
            "[TIMING] Hybrid retrieval total: "
            f"{time.perf_counter() - total_start:.2f}s",
            flush=True,
        )

        return final_results


qa_hybrid_retriever = QAHybridRetriever()
