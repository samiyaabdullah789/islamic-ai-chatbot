
import asyncio
import math
import re
import threading
import time
import unicodedata

from app.core.config import settings


class RAGReranker:
    STOP_WORDS = {
        "give", "me", "one", "some", "about",
        "the", "a", "an", "of", "in", "is",
        "are", "what", "how", "does", "do",
        "tell", "please", "show", "can", "you",
        "islam", "islamic", "for", "and", "to",
        "on", "regarding", "teach", "teaches",
        "says", "that", "which", "who", "with",
        "from", "this", "these", "those", "it",
        "according", "school", "schools",
        "i", "whether",
    }

    SYNONYMS = (
        {
            "prayer", "prayers", "pray",
            "praying", "namaz", "salah", "salat",
        },
        {
            "fast", "fasting", "fasts",
            "sawm", "siyam", "roza", "rozay",
        },
        {"zakat", "zakah", "alms"},
        {"wudu", "wudhu", "wuzu", "ablution"},
        {"ghusl", "bath"},
        {
            "obligatory", "obligation",
            "mandatory", "required", "requirement",
            "compulsory", "wajib", "fard", "farz",
        },
        {
            "recommended", "recommendation",
            "sunnah", "mustahabb",
        },
        {
            "invalidate", "invalidates",
            "invalidated", "nullify",
            "nullifies", "nullified",
        },
        {"permissible", "allowed", "halal"},
        {"forbidden", "prohibited", "haram"},
        {"marriage", "nikah", "nikkah"},
    )

    WINDOW_SIZE = 45

    DEFINITION_PATTERNS = (
        r"^what\s+(?:is|are|does)\b",
        r"^define\b",
        r"^what\s+does\s+.+\s+mean\b",
        r"^explain\s+(?:the\s+)?(?:meaning|definition)\b",
    )

    DEFINITION_TERMS = {
        "definition", "define", "meaning",
        "means", "refers", "term",
    }

    PROCEDURE_TERMS = {
        "calculate", "calculated", "calculation",
        "date", "amount", "rate", "percentage",
        "how", "when",
    }

    TOPIC_STOP_WORDS = STOP_WORDS | {
        "haram", "halal", "allowed", "permissible",
        "forbidden", "prohibited", "obligatory",
        "mandatory", "required", "sunnah",
        "listening", "listen", "heard", "hear",
        "answer", "available", "sahih", "bukhari",
        "hadith", "hadiths", "reference",
        "source", "sources", "should", "could",
        "would", "may", "must", "according",
        "provide", "explain", "please",
        "if", "he", "she", "they", "them",
        "cannot", "something", "someone",
        "anything", "anyone",
    }

    SUBJECT_GROUPS = (
        {
            "music", "musical", "instrument",
            "instruments", "sing", "singing",
            "singer", "singers", "song", "songs",
        },
        {"zakat", "zakah", "alms"},
        {
            "prayer", "prayers", "pray",
            "praying", "salah", "salat", "namaz",
        },
        {
            "fast", "fasts", "fasting",
            "sawm", "siyam", "roza", "rozay",
        },
        {"wudu", "wudhu", "wuzu", "ablution"},
        {"ghusl", "ritual", "bath"},
        {"marriage", "nikah", "nikkah"},
    )

    # Heuristic-only ranking uses its original score scale.
    TOPIC_WEIGHT = 3.0
    NO_TOPIC_PENALTY = 1.0

    @staticmethod
    def tokenize(text: str) -> set[str]:
        normalized = unicodedata.normalize(
            "NFKC", str(text)
        ).casefold().replace("’", "'")

        return set(
            re.findall(r"[^\W_]+", normalized)
        )

    @staticmethod
    def _token_list(text: str) -> list[str]:
        normalized = unicodedata.normalize(
            "NFKC", str(text)
        ).casefold().replace("’", "'")

        return re.findall(
            r"[^\W_]+", normalized
        )

    @staticmethod
    def _focused_question(question: str) -> str:
        focused = str(question).strip()

        focused = re.split(
            r"\b(?:answer according to|"
            r"according to the available|"
            r"provide (?:the )?"
            r"(?:citations?|references?|sources?)|"
            r"cite (?:the )?sources?)\b",
            focused,
            maxsplit=1,
            flags=re.I,
        )[0]

        return (
            focused.strip(" \t\n.!?")
            or str(question).strip()
        )

    def get_concepts(
        self,
        question: str,
    ) -> list[set[str]]:
        words = (
            self.tokenize(question)
            - self.STOP_WORDS
        )

        concepts = []
        seen = set()

        for word in sorted(words):
            group = next(
                (
                    synonyms
                    for synonyms in self.SYNONYMS
                    if word in synonyms
                ),
                None,
            )

            concept = (
                set(group)
                if group is not None
                else {word}
            )

            key = frozenset(concept)

            if key not in seen:
                seen.add(key)
                concepts.append(concept)

        return concepts

    def _topic_concepts(
        self,
        question: str,
    ) -> list[set[str]]:
        tokens = self.tokenize(
            self._focused_question(question)
        )

        concepts = []
        used = set()

        for group in self.SUBJECT_GROUPS:
            if tokens & group:
                concepts.append(set(group))
                used.update(tokens & group)

        remaining = (
            tokens
            - used
            - self.TOPIC_STOP_WORDS
        )

        for token in sorted(remaining):
            if len(token) >= 3:
                concepts.append({token})

        return concepts

    def _topic_coverage(
        self,
        question: str,
        document: dict,
    ) -> float:
        concepts = self._topic_concepts(question)

        if not concepts:
            return 0.0

        metadata = document.get("metadata") or {}

        content = str(
            document.get("content", "")
        )

        category = str(
            metadata.get("category", "")
        )

        tags = str(
            metadata.get("tags", "")
        )

        tokens = self.tokenize(
            content + " " + category + " " + tags
        )

        matched = sum(
            bool(tokens & concept)
            for concept in concepts
        )

        return matched / len(concepts)

    @staticmethod
    def _split_sections(
        content: str,
    ) -> tuple[str, str]:
        match = re.search(
            r"Question:\s*(.*?)\s*Answer:\s*(.*)",
            content,
            flags=re.I | re.S,
        )

        if match:
            return (
                match.group(1).strip(),
                match.group(2).strip(),
            )

        return "", content.strip()

    @staticmethod
    def _coverage(
        concepts: list[set[str]],
        tokens: set[str],
    ) -> float:
        if not concepts:
            return 0.0

        matched = sum(
            bool(tokens & concept)
            for concept in concepts
        )

        return matched / len(concepts)

    def _best_window_coverage(
        self,
        concepts: list[set[str]],
        text: str,
    ) -> float:
        words = self._token_list(text)

        if not words or not concepts:
            return 0.0

        if len(words) <= self.WINDOW_SIZE:
            return self._coverage(
                concepts,
                set(words),
            )

        best = 0.0
        step = max(
            1,
            self.WINDOW_SIZE // 3,
        )

        for start in range(
            0,
            len(words),
            step,
        ):
            window = set(
                words[
                    start:start + self.WINDOW_SIZE
                ]
            )

            best = max(
                best,
                self._coverage(
                    concepts,
                    window,
                ),
            )

            if best >= 1.0:
                break

        return best

    @staticmethod
    def _duplicate_key(
        document: dict,
    ) -> str:
        return str(
            document.get("id")
            or document.get("content", "")
        )

    @classmethod
    def _is_definition_question(
        cls,
        question: str,
    ) -> bool:
        normalized = " ".join(
            question.casefold().split()
        )

        return any(
            re.search(pattern, normalized)
            for pattern in cls.DEFINITION_PATTERNS
        )

    def _score(
        self,
        document: dict,
        concepts: list[set[str]],
        definition_question: bool,
    ) -> float:
        content = str(
            document.get("content", "")
        )

        question_section, answer_section = (
            self._split_sections(content)
        )

        metadata = (
            document.get("metadata") or {}
        )

        category = str(
            metadata.get("category", "")
        )

        tags = str(
            metadata.get("tags", "")
        )

        question_coverage = self._coverage(
            concepts,
            self.tokenize(question_section),
        )

        answer_coverage = (
            self._best_window_coverage(
                concepts,
                answer_section,
            )
        )

        metadata_coverage = self._coverage(
            concepts,
            self.tokenize(
                category + " " + tags
            ),
        )

        distance = document.get("distance")

        if (
            isinstance(distance, (int, float))
            and not isinstance(distance, bool)
            and math.isfinite(distance)
        ):
            semantic = 1.0 / (
                1.0 + max(
                    float(distance),
                    0.0,
                )
            )
        else:
            semantic = 0.0

        score = (
            2.0 * semantic
            + 2.5 * question_coverage
            + 1.5 * answer_coverage
            + 0.25 * metadata_coverage
        )

        if question_section:
            if question_coverage == 0:
                score -= 1.0

            if (
                question_coverage < 0.5
                and answer_coverage < 0.5
            ):
                score -= 0.5

        if definition_question:
            source_tokens = self.tokenize(
                question_section
            )

            answer_tokens = self.tokenize(
                answer_section
            )

            if (
                source_tokens
                & self.DEFINITION_TERMS
            ):
                score += 2.0

            elif (
                source_tokens
                & self.PROCEDURE_TERMS
            ):
                score -= 0.7

            if (
                answer_tokens
                & self.DEFINITION_TERMS
            ):
                score += 0.35

        doc_id = str(
            document.get("id", "")
        )

        match = re.search(
            r"-chunk-\d+-(\d+)$",
            doc_id,
        )

        if (
            match
            and int(match.group(1)) == 0
        ):
            score += 0.15

        return score

    def _rank_documents(
        self,
        question: str,
        documents: list[dict],
    ) -> list[dict]:
        focused_question = (
            self._focused_question(question)
        )

        concepts = self.get_concepts(
            focused_question
        )

        definition_question = (
            self._is_definition_question(
                focused_question
            )
        )

        topic_scores = [
            self._topic_coverage(
                focused_question,
                document,
            )
            for document in documents
        ]

        has_topic_matches = any(
            score > 0
            for score in topic_scores
        )

        scored = []

        for document, topic_score in zip(
            documents,
            topic_scores,
        ):
            base_score = self._score(
                document,
                concepts,
                definition_question,
            )

            final_score = base_score

            if has_topic_matches:
                if topic_score > 0:
                    final_score += (
                        self.TOPIC_WEIGHT
                        * topic_score
                    )
                else:
                    final_score -= (
                        self.NO_TOPIC_PENALTY
                    )

            scored.append({
                **document,
                "heuristic_score": base_score,
                "topic_score": topic_score,
                "rerank_score": final_score,
            })

        scored.sort(
            key=lambda document: (
                -document["rerank_score"],
                str(document.get("id", "")),
            )
        )

        return scored

    async def rerank(
        self,
        question: str,
        documents: list[dict],
        limit: int = 5,
    ) -> list[dict]:
        start = time.perf_counter()

        if not documents or limit <= 0:
            return []

        scored = self._rank_documents(
            question,
            documents,
        )

        selected = []
        selected_ids = set()

        for document in scored:
            doc_id = self._duplicate_key(
                document
            )

            if doc_id in selected_ids:
                continue

            selected.append(document)
            selected_ids.add(doc_id)

            if len(selected) >= limit:
                break

        for index, document in enumerate(
            selected,
            start=1,
        ):
            metadata = (
                document.get("metadata") or {}
            )

            print(
                f"[RERANK] Selected {index}: "
                f"source={metadata.get('source_name', 'Unknown')} | "
                f"madhhab={metadata.get('madhhab', 'N/A')} | "
                f"score={document['rerank_score']:.4f} | "
                f"topic={document['topic_score']:.3f} | "
                f"id={document.get('id')}",
                flush=True,
            )

        print(
            f"[TIMING] Reranker total: "
            f"{time.perf_counter() - start:.2f}s",
            flush=True,
        )

        return selected


class CrossEncoderReranker:
    """
    Hybrid reranker using:
    1. CrossEncoder semantic relevance
    2. Keyword heuristic relevance
    3. Subject/topic relevance
    4. Reciprocal rank fusion

    Evidence validation remains in RAGPipeline.
    """

    FUSION_K = 60

    CROSS_ENCODER_WEIGHT = 1.0
    HEURISTIC_WEIGHT = 1.0

    # Small topic adjustments relative to fusion scores.
    # These are tuning values, not evaluation-specific rules.
    TOPIC_WEIGHT = 0.0
    NO_TOPIC_PENALTY = 0.0

    def __init__(self):
        self._model = None
        self._lock = threading.Lock()
        self._heuristic = RAGReranker()

    def _load_model(self):
        with self._lock:
            if self._model is None:
                load_start = time.perf_counter()

                from sentence_transformers import (
                    CrossEncoder,
                )

                self._model = CrossEncoder(
                    settings.RERANKER_MODEL,
                    max_length=512,
                    device=settings.RERANKER_DEVICE,
                )

                print(
                    f"[TIMING] CrossEncoder model load: "
                    f"{time.perf_counter() - load_start:.2f}s",
                    flush=True,
                )

        return self._model

    @staticmethod
    def _focused_question(
        question: str,
    ) -> str:
        return RAGReranker._focused_question(
            question
        )

    def _topic_coverage(
        self,
        question: str,
        document: dict,
    ) -> float:
        return self._heuristic._topic_coverage(
            question,
            document,
        )

    def _score(
        self,
        question: str,
        documents: list[dict],
    ) -> list[float]:
        model = self._load_model()

        pairs = [
            (
                question,
                str(
                    document.get(
                        "content",
                        "",
                    )
                ),
            )
            for document in documents
        ]

        inference_start = time.perf_counter()

        predictions = model.predict(
            pairs
        )

        print(
            f"[TIMING] CrossEncoder inference: "
            f"{time.perf_counter() - inference_start:.2f}s",
            flush=True,
        )

        return [
            float(score)
            for score in predictions
        ]

    @staticmethod
    def _ranks(
        scores: list[float],
    ) -> dict[int, int]:
        ordering = sorted(
            range(len(scores)),
            key=lambda index: (
                -scores[index],
                index,
            ),
        )

        return {
            document_index: rank
            for rank, document_index in enumerate(
                ordering,
                start=1,
            )
        }

    async def rerank(
        self,
        question: str,
        documents: list[dict],
        limit: int = 5,
    ) -> list[dict]:
        start = time.perf_counter()

        if not documents or limit <= 0:
            return []

        focused_question = (
            self._focused_question(question)
        )

        print(
            f"[RERANK] Original query: "
            f"{question}",
            flush=True,
        )

        print(
            f"[RERANK] Focused query: "
            f"{focused_question}",
            flush=True,
        )

        cross_scores = await asyncio.to_thread(
            self._score,
            focused_question,
            documents,
        )

        concepts = (
            self._heuristic.get_concepts(
                focused_question
            )
        )

        definition_question = (
            self._heuristic._is_definition_question(
                focused_question
            )
        )

        heuristic_scores = [
            self._heuristic._score(
                document,
                concepts,
                definition_question,
            )
            for document in documents
        ]

        cross_ranks = self._ranks(
            cross_scores
        )

        heuristic_ranks = self._ranks(
            heuristic_scores
        )

        topic_scores = [
            self._topic_coverage(
                focused_question,
                document,
            )
            for document in documents
        ]

        has_topic_matches = any(
            score > 0
            for score in topic_scores
        )

        scored = []

        for index, document in enumerate(
            documents
        ):
            fusion_score = (
                self.CROSS_ENCODER_WEIGHT
                / (
                    self.FUSION_K
                    + cross_ranks[index]
                )
                + self.HEURISTIC_WEIGHT
                / (
                    self.FUSION_K
                    + heuristic_ranks[index]
                )
            )

            topic_score = topic_scores[index]
            final_score = fusion_score

            if has_topic_matches:
                if topic_score > 0:
                    final_score += (
                        self.TOPIC_WEIGHT
                        * topic_score
                    )
                else:
                    final_score -= (
                        self.NO_TOPIC_PENALTY
                    )

            scored.append({
                **document,
                "rerank_score": final_score,
                "fusion_score": fusion_score,
                "cross_encoder_score": (
                    cross_scores[index]
                ),
                "heuristic_score": (
                    heuristic_scores[index]
                ),
                "topic_score": topic_score,
                "cross_encoder_rank": (
                    cross_ranks[index]
                ),
                "heuristic_rank": (
                    heuristic_ranks[index]
                ),
            })

        scored.sort(
            key=lambda document: (
                -document["rerank_score"],
                document["cross_encoder_rank"],
            )
        )

        selected = []
        selected_ids = set()

        for document in scored:
            doc_id = (
                self._heuristic._duplicate_key(
                    document
                )
            )

            if doc_id in selected_ids:
                continue

            selected.append(document)
            selected_ids.add(doc_id)

            if len(selected) >= limit:
                break

        print(
            "[RERANK DEBUG] Hybrid candidate ranking:",
            flush=True,
        )

        for document in scored:
            print(
                f"[RERANK DEBUG] "
                f"id={document.get('id')} | "
                f"final={document['rerank_score']:.6f} | "
                f"fusion={document['fusion_score']:.6f} | "
                f"cross={document['cross_encoder_score']:.6f} | "
                f"heuristic={document['heuristic_score']:.4f} | "
                f"topic={document['topic_score']:.3f} | "
                f"cross_rank={document['cross_encoder_rank']} | "
                f"heuristic_rank={document['heuristic_rank']}",
                flush=True,
            )

        for index, document in enumerate(
            selected,
            start=1,
        ):
            metadata = (
                document.get("metadata") or {}
            )

            print(
                f"[RERANK] Selected {index}: "
                f"source={metadata.get('source_name', 'Unknown')} | "
                f"madhhab={metadata.get('madhhab', 'N/A')} | "
                f"score={document['rerank_score']:.6f} | "
                f"topic={document['topic_score']:.3f} | "
                f"id={document.get('id')}",
                flush=True,
            )

        print(
            f"[TIMING] Hybrid reranker total: "
            f"{time.perf_counter() - start:.2f}s",
            flush=True,
        )

        return selected


rag_reranker = (
    CrossEncoderReranker()
    if settings.RERANKER == "cross_encoder"
    else RAGReranker()
)
