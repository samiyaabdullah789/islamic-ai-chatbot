
import asyncio
import math
import re
import threading
import time
import unicodedata

from app.core.config import settings


class RAGReranker:
    """
    Generic intent-aware heuristic reranker.

    Scores:
    - Semantic retrieval similarity
    - Main subject coverage
    - Question intent relevance
    - Question/answer text coverage
    - Metadata relevance
    - Passage diversity
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
        {"backbiting", "gheebah", "gheebat"},
        {"anger", "angry", "wrath"},
        {"hypocrite", "hypocrisy", "munafiq"},
    )

    # Intent signals are generic, not tied to a
    # particular religious topic or answer.
    INTENT_GROUPS = {
        "importance": {
            "importance", "important", "significance",
            "benefit", "benefits", "virtue", "virtues",
            "value", "reward", "rewards", "merit",
            "merits", "status", "rank", "essential",
        },
        "definition": {
            "define", "definition", "meaning",
            "means", "mean", "refers", "called",
            "known", "described",
        },
        "obligation": {
            "obligatory", "obligation", "mandatory",
            "compulsory", "required", "requirement",
            "must", "duty", "duties", "fard",
            "farz", "wajib", "enjoined", "commanded",
        },
        "method": {
            "perform", "performed", "procedure",
            "steps", "step", "method", "manner",
            "way", "practice", "practiced",
        },
        "prohibition": {
            "forbidden", "prohibited", "haram",
            "avoid", "avoiding", "sin", "sins",
            "sinful", "unlawful",
        },
        "permission": {
            "permissible", "allowed", "halal",
            "permitted", "lawful",
        },
        "consequence": {
            "punishment", "consequence", "consequences",
            "warning", "warned", "danger", "harm",
            "accountability",
        },
    }

    WINDOW_SIZE = 60
    DUPLICATE_THRESHOLD = 0.88

    TOPIC_WEIGHT = 0.75
    NO_TOPIC_PENALTY = 0.35

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

    @classmethod
    def _detect_intent(cls, question: str) -> str:
        words = cls.tokenize(question)

        patterns = (
            (
                "importance",
                r"\b(?:importance|important|"
                r"significance|benefits?|virtues?|"
                r"rewards?|merits?)\b",
            ),
            (
                "obligation",
                r"\b(?:obligatory|obligation|"
                r"mandatory|compulsory|required|"
                r"must|fard|farz|wajib)\b",
            ),
            (
                "prohibition",
                r"\b(?:forbidden|prohibited|"
                r"haram|sinful|unlawful)\b",
            ),
            (
                "permission",
                r"\b(?:permissible|allowed|"
                r"halal|permitted|lawful)\b",
            ),
            (
                "method",
                r"\b(?:how to|how do|how did|"
                r"steps|procedure|method|"
                r"perform|performed)\b",
            ),
            (
                "definition",
                r"\b(?:what is|what are|define|"
                r"definition|meaning|what does)\b",
            ),
            (
                "consequence",
                r"\b(?:punishment|consequences?|"
                r"warnings?|harm)\b",
            ),
        )

        normalized = " ".join(
            cls._token_list(question)
        )

        for intent, pattern in patterns:
            if re.search(pattern, normalized):
                return intent

        if words & {"why"}:
            return "importance"

        return "general"

    @classmethod
    def _intent_terms(cls, intent: str) -> set[str]:
        return set(cls.INTENT_GROUPS.get(intent, set()))

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

    def _subject_concepts(
        self,
        question: str,
    ) -> list[set[str]]:
        """
        Remove intent terms so topic matching
        is not confused with answer intent.
        """
        intent = self._detect_intent(question)
        words = self.tokenize(question) - self.STOP_WORDS

        intent_vocabulary = set()
        for group in self.INTENT_GROUPS.values():
            intent_vocabulary.update(group)

        # Remove common intent words, but keep
        # any remaining subject-specific concepts.
        words -= intent_vocabulary

        if not words:
            return self.get_concepts(question)

        return self.get_concepts(" ".join(sorted(words)))

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

        return "", str(content or "").strip()

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

    def _topic_coverage(
        self,
        question: str,
        document: dict,
    ) -> float:
        concepts = self._subject_concepts(question)
        metadata = document.get("metadata") or {}

        content = " ".join([
            str(document.get("content") or ""),
            str(metadata.get("category") or ""),
            str(metadata.get("tags") or ""),
        ])

        return self._coverage(
            concepts,
            self.tokenize(content),
        )

    def _intent_relevance(
        self,
        question: str,
        document: dict,
    ) -> float:
        """
        Estimate whether the passage addresses
        the question's requested intent.

        This is a ranking signal, not a
        substitute for evidence verification.
        """
        intent = self._detect_intent(question)

        if intent == "general":
            return 0.0

        content = str(document.get("content") or "")
        metadata = document.get("metadata") or {}

        text = " ".join([
            content,
            str(metadata.get("category") or ""),
            str(metadata.get("tags") or ""),
        ])

        tokens = self.tokenize(text)
        terms = self._intent_terms(intent)

        if not terms:
            return 0.0

        matched = len(tokens & terms)
        if matched == 0:
            return 0.0

        # Saturate after two distinct intent terms.
        return min(1.0, matched / 2.0)

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

            if self.tokenize(answer_text) & definition_terms:
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

        candidate_tokens = self.tokenize(candidate_text)
        selected_tokens = self.tokenize(selected_text)

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

        # Do not reintroduce near-duplicates.
        return selected

    def _build_heuristic_scores(
        self,
        question: str,
        documents: list[dict],
    ) -> list[dict]:
        focused = self._focused_question(question)
        concepts = self.get_concepts(focused)
        definition_question = (
            self._is_definition_question(focused)
            and self._detect_intent(focused) == "definition"
        )

        topic_scores = [
            self._topic_coverage(focused, document)
            for document in documents
        ]

        intent_scores = [
            self._intent_relevance(focused, document)
            for document in documents
        ]

        has_topic_matches = any(
            score > 0 for score in topic_scores
        )

        scored = []

        for document, topic_score, intent_score in zip(
            documents,
            topic_scores,
            intent_scores,
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

            # Intent is a bounded supporting signal.
            # It cannot by itself establish relevance.
            final_score += 0.5 * intent_score * topic_score

            scored.append({
                **document,
                "heuristic_score": base_score,
                "topic_score": topic_score,
                "intent_score": intent_score,
                "rerank_score": final_score,
            })

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

        focused = self._focused_question(question)
        intent = self._detect_intent(focused)

        print(
            f"[RERANK] Detected intent: {intent}",
            flush=True,
        )

        scored = self._build_heuristic_scores(
            focused,
            documents,
        )

        scored.sort(
            key=lambda item: (
                -item["rerank_score"],
                str(item.get("id") or ""),
            )
        )

        selected = self._select_diverse(scored, limit)

        for index, document in enumerate(
            selected, start=1
        ):
            metadata = document.get("metadata") or {}

            print(
                f"[RERANK] Selected {index}: "
                f"source={metadata.get('source_name', 'Unknown')} | "
                f"score={document['rerank_score']:.6f} | "
                f"topic={document['topic_score']:.3f} | "
                f"intent={document['intent_score']:.3f} | "
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
    Hybrid reranker combining:
    - CrossEncoder
    - Heuristic scores
    - Topic coverage
    - Intent relevance
    - Reciprocal rank fusion
    - Near-duplicate filtering

    Final claim verification remains in RAGPipeline.
    """

    FUSION_K = 60

    CROSS_ENCODER_WEIGHT = 1.5
    HEURISTIC_WEIGHT = 1.0

    TOPIC_WEIGHT = 0.004
    NO_TOPIC_PENALTY = 0.002
    INTENT_WEIGHT = 0.006

    def __init__(self):
        self._model = None
        self._lock = threading.Lock()
        self._heuristic = RAGReranker()

    def _load_model(self):
        with self._lock:
            if self._model is None:
                start = time.perf_counter()

                from sentence_transformers import CrossEncoder

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
        return RAGReranker._focused_question(question)

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

        focused_question = self._focused_question(question)
        intent = self._heuristic._detect_intent(
            focused_question
        )

        print(
            f"[RERANK] Original query: {question}",
            flush=True,
        )
        print(
            f"[RERANK] Focused query: {focused_question}",
            flush=True,
        )
        print(
            f"[RERANK] Detected intent: {intent}",
            flush=True,
        )

        cross_scores = await asyncio.to_thread(
            self._score,
            focused_question,
            documents,
        )

        heuristic_documents = (
            self._heuristic._build_heuristic_scores(
                focused_question,
                documents,
            )
        )

        heuristic_scores = [
            document["rerank_score"]
            for document in heuristic_documents
        ]

        topic_scores = [
            document["topic_score"]
            for document in heuristic_documents
        ]

        intent_scores = [
            document["intent_score"]
            for document in heuristic_documents
        ]

        cross_ranks = self._ranks(cross_scores)
        heuristic_ranks = self._ranks(heuristic_scores)

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
            intent_score = intent_scores[index]

            final_score = fusion_score

            if has_topic_matches:
                if topic_score > 0:
                    final_score += (
                        self.TOPIC_WEIGHT * topic_score
                    )
                else:
                    final_score -= self.NO_TOPIC_PENALTY

            # Require some subject overlap before
            # rewarding an intent keyword.
            final_score += (
                self.INTENT_WEIGHT
                * intent_score
                * topic_score
            )

            scored.append({
                **document,
                "rerank_score": final_score,
                "fusion_score": fusion_score,
                "cross_encoder_score": cross_scores[index],
                "heuristic_score": heuristic_scores[index],
                "topic_score": topic_score,
                "intent_score": intent_score,
                "cross_encoder_rank": cross_ranks[index],
                "heuristic_rank": heuristic_ranks[index],
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

        # Limit debug output to avoid huge logs.
        for document in scored[:20]:
            print(
                f"[RERANK DEBUG] "
                f"id={document.get('id')} | "
                f"final={document['rerank_score']:.6f} | "
                f"fusion={document['fusion_score']:.6f} | "
                f"cross={document['cross_encoder_score']:.6f} | "
                f"heuristic={document['heuristic_score']:.4f} | "
                f"topic={document['topic_score']:.3f} | "
                f"intent={document['intent_score']:.3f} | "
                f"cross_rank={document['cross_encoder_rank']} | "
                f"heuristic_rank={document['heuristic_rank']}",
                flush=True,
            )

        for index, document in enumerate(
            selected, start=1
        ):
            metadata = document.get("metadata") or {}

            print(
                f"[RERANK] Selected {index}: "
                f"source={metadata.get('source_name', 'Unknown')} | "
                f"madhhab={metadata.get('madhhab', 'N/A')} | "
                f"score={document['rerank_score']:.6f} | "
                f"topic={document['topic_score']:.3f} | "
                f"intent={document['intent_score']:.3f} | "
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
