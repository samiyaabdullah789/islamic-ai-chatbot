
import math
import os
import re
import time
import unicodedata

from app.embeddings.ollama_embeddings import embedding_service
from app.vector_store.chroma_store import chroma_store


class RAGRetriever:
    MIN_CANDIDATES_BEFORE_FALLBACK = 5

    QUERY_FILLER_WORDS = {
        "give", "me", "one", "some", "please",
        "tell", "show", "share", "what", "the",
        "a", "an", "about", "islam", "islamic",
    }

    GENERAL_TOPICS = (
        r"\bfive pillars\b",
        r"\bpillars of islam\b",
        r"\bpillars of faith\b",
        r"\barticles of faith\b",
        r"\bnames of allah\b",
        r"\b99 names\b",
        r"\bsurah al.fatiha\b",
        r"\bsurah al.ikhlas\b",
    )

    GENERAL_FACT_PATTERNS = (
        r"\bwho was prophet\b",
        r"\bwho is prophet\b",
        r"\bwhat is the quran\b",
        r"\bwhat is the qur.an\b",
        r"\bhow many surahs\b",
        r"\bhow many chapters\b",
    )

    @staticmethod
    def _normalize(text: str) -> str:
        text = unicodedata.normalize("NFKC", str(text))
        return " ".join(text.strip().split())

    @classmethod
    def is_clearly_general_question(cls, question: str) -> bool:
        normalized = cls._normalize(question).casefold()

        return any(
            re.search(pattern, normalized)
            for pattern in (
                cls.GENERAL_TOPICS + cls.GENERAL_FACT_PATTERNS
            )
        )

    @staticmethod
    def _focused_query(query: str) -> str:
        focused = re.split(
            r"\b(?:answer according to|"
            r"according to the available|"
            r"provide (?:the )?(?:citations?|references?|sources?)|"
            r"cite (?:the )?sources?)\b",
            query,
            maxsplit=1,
            flags=re.I,
        )[0].strip(" \t\n.!?")

        return focused or query.strip()

    def build_search_queries(self, query: str) -> list[str]:
        original = self._normalize(query)

        if not original:
            return []

        focused = self._focused_query(original)

        words = re.findall(r"\b\w+\b", focused)

        filler = self.QUERY_FILLER_WORDS | {
            "answer", "according", "available",
            "hadith", "hadiths", "sahih", "bukhari",
            "listening", "listen", "is", "are",
            "in", "to", "of", "whether",
            "can", "do", "does",
        }

        topic = " ".join(
            word for word in words
            if word.casefold() not in filler
        )

        queries = [focused]

        if (
            topic
            and len(topic) >= 4
            and topic.casefold() != focused.casefold()
        ):
            queries.append(topic)

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

    @staticmethod
    def _extract_results(
        results: dict,
        query_index: int,
    ) -> list[dict]:
        ids = results.get("ids", [[]])[0]
        contents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        documents = []

        for doc_id, content, metadata, distance in zip(
            ids, contents, metadatas, distances
        ):
            if not str(content or "").strip():
                continue

            documents.append({
                "id": str(doc_id),
                "content": str(content),
                "metadata": metadata or {},
                "distance": self_distance(distance),
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

            if (
                previous is None
                or document["distance"] < previous["distance"]
            ):
                documents_by_id[doc_id] = document

    @staticmethod
    def _get_test_source_id() -> int | None:
        value = os.getenv("RAG_TEST_SOURCE_ID", "").strip()

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
            conditions.append({"source_id": source_id})

        if madhhab is not None and not general_question:
            conditions.append({"madhhab": madhhab})

        if not conditions:
            return None

        if len(conditions) == 1:
            return conditions[0]

        return {"$and": conditions}

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
            test_source_id,
            madhhab,
            general_question,
        )

        print(
            f"[RAG] Test source ID: "
            f"{test_source_id if test_source_id is not None else 'All'}",
            flush=True,
        )

        print(
            f"[RAG] Selected Madhhab: {madhhab or 'General'}",
            flush=True,
        )

        print(
            f"[RAG] ChromaDB filter: {where_filter}",
            flush=True,
        )

        documents_by_id = {}

        for query_index, search_query in enumerate(queries):
            print(
                f"[RAG] Search query {query_index + 1}: "
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

            documents = self._extract_results(
                results,
                query_index,
            )

            self._merge_documents(
                documents_by_id,
                documents,
            )

            print(
                f"[TIMING] Query {query_index + 1} ChromaDB search: "
                f"{time.perf_counter() - start:.2f}s",
                flush=True,
            )

            print(
                f"[RAG] Query {query_index + 1} candidates: "
                f"{len(documents)}",
                flush=True,
            )

        documents = sorted(
            documents_by_id.values(),
            key=lambda document: document["distance"],
        )

        if test_source_id is not None:
            documents = [
                document
                for document in documents
                if str(
                    document["metadata"].get("source_id")
                ) == str(test_source_id)
            ]

        print(
            f"[RAG] Unique retrieved documents: "
            f"{len(documents)}",
            flush=True,
        )

        for index, document in enumerate(
            documents[:5],
            start=1,
        ):
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
            f"[TIMING] Retriever total: "
            f"{time.perf_counter() - total_start:.2f}s",
            flush=True,
        )

        return documents


def self_distance(distance: object) -> float:
    return RAGRetriever._valid_distance(distance)


rag_retriever = RAGRetriever()
