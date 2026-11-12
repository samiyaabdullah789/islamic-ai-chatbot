
"""Read-only BM25 index over existing Chroma documents. No extra dependency."""
import math
import re
import threading
import time
import unicodedata
from collections import Counter, defaultdict

from app.vector_store.chroma_store import chroma_store


class BM25Index:
    K1 = 1.5
    B = 0.75
    PAGE_SIZE = 500

    def __init__(self):
        self._lock = threading.RLock()
        self._ready = False
        self._documents = {}
        self._postings = {}
        self._lengths = {}
        self._average_length = 0.0
        self._count = 0

    @staticmethod
    def tokenize(text):
        text = unicodedata.normalize(
            "NFKC", str(text or "")
        ).casefold()
        return re.findall(
            r"[^\W_]+", text, flags=re.UNICODE
        )

    def invalidate(self):
        """Call after ingesting or deleting source documents."""
        with self._lock:
            self._ready = False

    def refresh(self):
        start = time.perf_counter()
        documents = {}
        postings = defaultdict(dict)
        lengths = {}
        offset = 0

        while True:
            batch = chroma_store.get_documents_page(
                offset=offset,
                limit=self.PAGE_SIZE,
            )
            ids = batch.get("ids") or []
            contents = batch.get("documents") or []
            metadatas = batch.get("metadatas") or []

            for index, doc_id in enumerate(ids):
                content = (
                    str(contents[index] or "")
                    if index < len(contents) else ""
                )
                if not content.strip():
                    continue

                metadata = (
                    metadatas[index]
                    if index < len(metadatas) else {}
                )
                metadata = (
                    metadata
                    if isinstance(metadata, dict) else {}
                )
                doc_id = str(doc_id)

                searchable = " ".join((
                    content,
                    str(metadata.get("category") or ""),
                    str(metadata.get("tags") or ""),
                ))
                frequencies = Counter(
                    self.tokenize(searchable)
                )
                if not frequencies:
                    continue

                documents[doc_id] = {
                    "id": doc_id,
                    "content": content,
                    "metadata": metadata,
                    "distance": float("inf"),
                    "query_index": -1,
                }
                lengths[doc_id] = sum(
                    frequencies.values()
                )

                for token, frequency in frequencies.items():
                    postings[token][doc_id] = frequency

            offset += len(ids)
            if len(ids) < self.PAGE_SIZE:
                break

        with self._lock:
            self._documents = documents
            self._postings = dict(postings)
            self._lengths = lengths
            self._count = len(documents)
            self._average_length = (
                sum(lengths.values())
                / max(1, len(lengths))
            )
            self._ready = True

        print(
            f"[BM25] Indexed {len(documents)} documents "
            f"in {time.perf_counter() - start:.2f}s",
            flush=True,
        )

    @staticmethod
    def _matches(
        metadata,
        source_id,
        madhhab,
        general_question,
    ):
        if (
            source_id is not None
            and str(metadata.get("source_id"))
            != str(source_id)
        ):
            return False

        if (
            madhhab
            and madhhab.casefold() != "general"
            and not general_question
        ):
            return (
                str(metadata.get("madhhab", "")).casefold()
                == madhhab.casefold()
            )

        return True

    def search(
        self,
        query,
        limit=30,
        source_id=None,
        madhhab=None,
        general_question=False,
    ):
        if limit <= 0:
            return []

        if not self._ready:
            self.refresh()

        tokens = set(self.tokenize(query))

        with self._lock:
            scores = defaultdict(float)
            n = self._count
            avg = max(self._average_length, 1)

            for token in tokens:
                posting = self._postings.get(token, {})
                df = len(posting)
                if not df:
                    continue

                idf = math.log(
                    1 + (n - df + 0.5) / (df + 0.5)
                )

                for doc_id, frequency in posting.items():
                    metadata = self._documents[
                        doc_id
                    ]["metadata"]

                    if not self._matches(
                        metadata,
                        source_id,
                        madhhab,
                        general_question,
                    ):
                        continue

                    denominator = (
                        frequency
                        + self.K1
                        * (
                            1 - self.B
                            + self.B
                            * self._lengths[doc_id]
                            / avg
                        )
                    )
                    scores[doc_id] += (
                        idf
                        * frequency
                        * (self.K1 + 1)
                        / denominator
                    )

            ranked = sorted(
                scores.items(),
                key=lambda pair: (-pair[1], pair[0]),
            )[:limit]

            return [
                {
                    **self._documents[doc_id],
                    "bm25_score": score,
                }
                for doc_id, score in ranked
            ]


bm25_index = BM25Index()
