
import asyncio
import math
import re
import threading
import time
import unicodedata

from app.core.config import settings


class RAGReranker:
    """
    Generic heuristic reranker.

    Uses:
    - Semantic retrieval distance
    - Question and answer keyword coverage
    - Metadata relevance
    - Topic coverage
    - Near-duplicate filtering
    """

    STOP_WORDS = {
        "a", "an", "the", "is", "are", "was", "were",
        "what", "which", "who", "how", "why", "when",
        "where", "do", "does", "did", "can", "could",
        "would", "should", "will", "may", "might",
        "of", "in", "on", "at", "to", "for", "from",
        "with", "and", "or", "but", "about", "that",
        "this", "these", "those", "it", "its",
        "me", "my", "you", "your", "we", "our",
        "give", "tell", "show", "share", "please",
        "explain", "describe", "provide", "some",
        "one", "any", "according", "regarding",
        "islam", "islamic",
    }

    SYNONYMS = (
        {"prayer", "prayers", "pray", "praying",
         "namaz", "salah", "salat"},
        {"fast", "fasting", "fasts", "sawm",
         "siyam", "roza", "rozay"},
        {"zakat", "zakah", "alms"},
        {"wudu", "wudhu", "wuzu", "ablution"},
        {"ghusl", "ritual", "bath"},
        {"obligatory", "obligation", "mandatory",
         "required", "requirement", "compulsory",
         "wajib", "fard", "farz"},
        {"recommended", "recommendation",
         "sunnah", "mustahabb"},
        {"invalidate", "invalidates", "invalidated",
         "nullify", "nullifies", "nullified"},
        {"permissible", "allowed", "halal"},
        {"forbidden", "prohibited", "haram"},
        {"marriage", "nikah", "nikkah"},
    )

    WINDOW_SIZE = 60
    TOPIC_WEIGHT = 0.75
    NO_TOPIC_PENALTY = 0.35
    DUPLICATE_THRESHOLD = 0.88

    @staticmethod
    def _token_list(text: str) -> list[str]:
        text = unicodedata.normalize(
            "NFKC", str(text or "")
        ).casefold()

        text = text.replace("’", "'")

        return re.findall(r"[^\W_]+", text)

    @classmethod
    def tokenize(cls, text: str) -> set[str]:
        return set(cls._token_list(text))

    @staticmethod
    def _focused_question(question: str) -> str:
        question = str(question or "").strip()

        focused = re.split(
            r"\b(?:answer according to|"
            r"according to the available|"
            r"provide (?:the )?"
            r"(?:citations?|references?|sources?)|"
            r"cite (?:the )?sources?)\b",
            question,
            maxsplit=1,
            flags=re.I,
        )[0]

        return focused.strip(" \t\n.!?") or question

    def get_concepts(
        self,
        question: str,
    ) -> list[set[str]]:
        words = self.tokenize(question) - self.STOP_WORDS

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

            concept = set(group) if group else {word}
            key = frozenset(concept)

            if key not in seen:
                seen.add(key)
                concepts.append(concept)

        return concepts

    def _topic_concepts(
        self,
        question: str,
    ) -> list[set[str]]:
        return self.get_concepts(
            self._focused_question(question)
        )

    @staticmethod
    def _coverage(
        concepts: list[set[str]],
        tokens: set[str],
    ) -> float:
        if not concepts:
            return 0.0

        matched = sum(
            bool(tokens.intersection(concept))
            for concept in concepts
        )

        return matched / len(concepts)

    def _topic_coverage(
        self,
        question: str,
        document: dict,
    ) -> float:
        concepts = self._topic_concepts(question)

        metadata = document.get("metadata") or {}

        text = " ".join([
            str(document.get("content") or ""),
            str(metadata.get("category") or ""),
            str(metadata.get("tags") or ""),
        ])

        return self._coverage(
            concepts,
            self.tokenize(text),
        )

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

        return "", str(content).strip()

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
        step = max(1, self.WINDOW_SIZE // 3)

        for start in range(0, len(words), step):
            window = set(
                words[start:start + self.WINDOW_SIZE]
            )

            best = max(
                best,
                self._coverage(concepts, window),
            )

            if best >= 1.0:
                break

        return best

    @staticmethod
    def _valid_distance(distance) -> float | None:
        if (
            isinstance(distance, (int, float))
            and not isinstance(distance, bool)
            and math.isfinite(distance)
        ):
            return max(0.0, float(distance))

        return None

    @classmethod
    def _is_definition_question(
        cls,
        question: str,
    ) -> bool:
        normalized = " ".join(
            cls._token_list(question)
        )

        patterns = (
            r"^what\s+(?:is|are)\b",
            r"^define\b",
            r"^what\s+does\s+.+\s+mean\b",
            r"^explain\s+(?:the\s+)?"
            r"(?:meaning|definition)\b",
        )

        return any(
            re.search(pattern, normalized)
            for pattern in patterns
        )

    def _score(
        self,
        document: dict,
        concepts: list[set[str]],
        definition_question: bool,
    ) -> float:
        content = str(document.get("content") or "")
        metadata = document.get("metadata") or {}

        question_text, answer_text = (
            self._split_sections(content)
        )

        question_coverage = self._coverage(
            concepts,
            self.tokenize(question_text),
        )

        answer_coverage = self._best_window_coverage(
            concepts,
            answer_text,
        )

        metadata_text = " ".join([
            str(metadata.get("category") or ""),
            str(metadata.get("tags") or ""),
        ])

        metadata_coverage = self._coverage(
            concepts,
            self.tokenize(metadata_text),
        )

        distance = self._valid_distance(
            document.get("distance")
        )

        semantic_score = (
            1.0 / (1.0 + distance)
            if distance is not None
            else 0.0
        )

        score = (
            2.0 * semantic_score
            + 2.5 * question_coverage
            + 1.5 * answer_coverage
            + 0.25 * metadata_coverage
        )

        if question_text:
            if question_coverage == 0:
                score -= 0.5

            if (
                question_coverage < 0.5
                and answer_coverage < 0.5
            ):
                score -= 0.35

        if definition_question:
            definition_terms = {
                "definition", "define",
                "meaning", "means", "refers",
            }

            if (
                self.tokenize(answer_text)
                & definition_terms
            ):
                score += 0.25

        return score

    @staticmethod
    def _duplicate_key(document: dict) -> str:
        return str(
            document.get("id")
            or document.get("content")
            or ""
        )

    def _is_near_duplicate(
        self,
        candidate: dict,
        selected: dict,
    ) -> bool:
        candidate_text = str(
            candidate.get("content") or ""
        )

        selected_text = str(
            selected.get("content") or ""
        )

        if not candidate_text or not selected_text:
            return False

        candidate_tokens = self.tokenize(
            candidate_text
        )

        selected_tokens = self.tokenize(
            selected_text
        )

        if not candidate_tokens or not selected_tokens:
            return False

        intersection = len(
            candidate_tokens & selected_tokens
        )

        union = len(
            candidate_tokens | selected_tokens
        )

        similarity = intersection / max(1, union)

        return similarity >= self.DUPLICATE_THRESHOLD

    def _select_diverse(
        self,
        ranked: list[dict],
        limit: int,
    ) -> list[dict]:
        selected = []
        selected_ids = set()

        for document in ranked:
            doc_id = self._duplicate_key(document)

            if doc_id in selected_ids:
                continue

            if any(
                self._is_near_duplicate(document, previous)
                for previous in selected
            ):
                continue

            selected.append(document)
            selected_ids.add(doc_id)

            if len(selected) >= limit:
                return selected

        # Fill remaining slots if diversity filtering
        # leaves fewer passages than requested.
        for document in ranked:
            doc_id = self._duplicate_key(document)

            if doc_id in selected_ids:
                continue

            selected.append(document)
            selected_ids.add(doc_id)

            if len(selected) >= limit:
                break

        return selected

    async def rerank(
        self,
        question: str,
        documents: list[dict],
        limit: int = 5,
    ) -> list[dict]:
        start = time.perf_counter()

        if not documents or limit <= 0:
            return []

        focused = self._focused_question(question)
        concepts = self.get_concepts(focused)

        definition_question = (
            self._is_definition_question(focused)
        )

        topic_scores = [
            self._topic_coverage(focused, document)
            for document in documents
        ]

        has_topic_matches = any(
            score > 0 for score in topic_scores
        )

        scored = []

        for document, topic_score in zip(
            documents, topic_scores
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
                        self.TOPIC_WEIGHT * topic_score
                    )
                else:
                    final_score -= self.NO_TOPIC_PENALTY

            scored.append({
                **document,
                "heuristic_score": base_score,
                "topic_score": topic_score,
                "rerank_score": final_score,
            })

        scored.sort(
            key=lambda item: (
                -item["rerank_score"],
                str(item.get("id") or ""),
            )
        )

        selected = self._select_diverse(
            scored, limit
        )

        for index, document in enumerate(
            selected, start=1
        ):
            metadata = document.get("metadata") or {}

            print(
                f"[RERANK] Selected {index}: "
                f"source={metadata.get('source_name', 'Unknown')} | "
                f"score={document['rerank_score']:.6f} | "
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
    Hybrid reranker.

    Combines:
    1. CrossEncoder relevance
    2. Heuristic relevance
    3. Topic coverage
    4. Reciprocal rank fusion
    5. Near-duplicate filtering

    Final evidence verification remains in RAGPipeline.
    """

    FUSION_K = 60

    CROSS_ENCODER_WEIGHT = 1.5
    HEURISTIC_WEIGHT = 1.0

    # Modest adjustments relative to fusion scores.
    TOPIC_WEIGHT = 0.004
    NO_TOPIC_PENALTY = 0.002

    def __init__(self):
        self._model = None
        self._lock = threading.Lock()
        self._heuristic = RAGReranker()

    def _load_model(self):
        with self._lock:
            if self._model is None:
                start = time.perf_counter()

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
                    f"{time.perf_counter() - start:.2f}s",
                    flush=True,
                )

        return self._model

    @staticmethod
    def _focused_question(question: str) -> str:
        return RAGReranker._focused_question(
            question
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
                str(document.get("content") or ""),
            )
            for document in documents
        ]

        start = time.perf_counter()

        predictions = model.predict(pairs)

        print(
            f"[TIMING] CrossEncoder inference: "
            f"{time.perf_counter() - start:.2f}s",
            flush=True,
        )

        scores = []

        for prediction in predictions:
            score = float(prediction)

            if not math.isfinite(score):
                score = float("-inf")

            scores.append(score)

        return scores

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
        total_start = time.perf_counter()

        if not documents or limit <= 0:
            return []

        focused_question = (
            self._focused_question(question)
        )

        print(
            f"[RERANK] Original query: {question}",
            flush=True,
        )

        print(
            f"[RERANK] Focused query: {focused_question}",
            flush=True,
        )

        cross_scores = await asyncio.to_thread(
            self._score,
            focused_question,
            documents,
        )

        concepts = self._heuristic.get_concepts(
            focused_question
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

        topic_scores = [
            self._heuristic._topic_coverage(
                focused_question,
                document,
            )
            for document in documents
        ]

        cross_ranks = self._ranks(cross_scores)
        heuristic_ranks = self._ranks(
            heuristic_scores
        )

        has_topic_matches = any(
            score > 0 for score in topic_scores
        )

        scored = []

        for index, document in enumerate(documents):
            fusion_score = (
                self.CROSS_ENCODER_WEIGHT
                / (self.FUSION_K + cross_ranks[index])
                + self.HEURISTIC_WEIGHT
                / (self.FUSION_K + heuristic_ranks[index])
            )

            topic_score = topic_scores[index]
            final_score = fusion_score

            if has_topic_matches:
                if topic_score > 0:
                    final_score += (
                        self.TOPIC_WEIGHT * topic_score
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
                str(document.get("id") or ""),
            )
        )

        selected = self._heuristic._select_diverse(
            scored,
            limit,
        )

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
            metadata = document.get("metadata") or {}

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
            f"{time.perf_counter() - total_start:.2f}s",
            flush=True,
        )

        return selected


rag_reranker = (
    CrossEncoderReranker()
    if settings.RERANKER == "cross_encoder"
    else RAGReranker()
)
