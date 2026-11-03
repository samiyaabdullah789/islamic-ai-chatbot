
import math
import re
import time
import unicodedata


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
        {
            "zakat", "zakah", "alms",
        },
        {
            "wudu", "wudhu", "wuzu",
            "ablution",
        },
        {
            "ghusl", "bath",
        },
        {
            "obligatory", "obligation",
            "mandatory", "required",
            "requirement", "compulsory",
            "wajib", "fard", "farz",
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
        {
            "permissible", "allowed", "halal",
        },
        {
            "forbidden", "prohibited", "haram",
        },
        {
            "marriage", "nikah", "nikkah",
        },
    )

    WINDOW_SIZE = 45

    DEFINITION_PATTERNS = (
        r"^what\s+(?:is|are|does)\b",
        r"^define\b",
        r"^what\s+does\s+.+\s+mean\b",
        r"^explain\s+(?:the\s+)?(?:meaning|definition)\b",
    )

    DEFINITION_TERMS = {
        "definition",
        "define",
        "meaning",
        "means",
        "refers",
        "term",
    }

    PROCEDURE_TERMS = {
        "calculate",
        "calculated",
        "calculation",
        "date",
        "amount",
        "rate",
        "percentage",
        "how",
        "when",
    }

    @staticmethod
    def tokenize(text: str) -> set[str]:

        normalized = unicodedata.normalize(
            "NFKC",
            str(text),
        ).casefold()

        normalized = normalized.replace(
            "’", "'"
        )

        return set(
            re.findall(
                r"[^\W_]+",
                normalized,
            )
        )

    @staticmethod
    def _token_list(text: str) -> list[str]:

        normalized = unicodedata.normalize(
            "NFKC",
            str(text),
        ).casefold()

        normalized = normalized.replace(
            "’", "'"
        )

        return re.findall(
            r"[^\W_]+",
            normalized,
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

        for start in range(
            0,
            len(words),
            max(1, self.WINDOW_SIZE // 3),
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

    @classmethod
    def _duplicate_key(
        cls,
        document: dict,
    ) -> str:

        # Keep distinct chunks even when
        # they share the same source question.
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
            re.search(
                pattern,
                normalized,
            )
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
            self.tokenize(
                question_section
            ),
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

        distance = document.get(
            "distance"
        )

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

        # Prefer initial chunk slightly.
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

    async def rerank(
        self,
        question: str,
        documents: list[dict],
        limit: int = 5,
    ) -> list[dict]:

        start = time.perf_counter()

        if not documents or limit <= 0:
            return []

        concepts = self.get_concepts(
            question
        )

        definition_question = (
            self._is_definition_question(
                question
            )
        )

        scored = [
            (
                self._score(
                    document,
                    concepts,
                    definition_question,
                ),
                document,
            )
            for document in documents
        ]

        scored.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        selected = []
        selected_ids = set()

        for score, document in scored:
            doc_id = self._duplicate_key(
                document
            )

            if doc_id in selected_ids:
                continue

            selected.append(document)
            selected_ids.add(
                doc_id
            )

            if len(selected) >= limit:
                break

        for index, document in enumerate(
            selected,
            start=1,
        ):
            metadata = (
                document.get("metadata") or {}
            )

            score = self._score(
                document,
                concepts,
                definition_question,
            )

            print(
                f"[RERANK] Selected {index}: "
                f"source={metadata.get('source_name', 'Unknown')} | "
                f"madhhab={metadata.get('madhhab', 'N/A')} | "
                f"score={score:.4f} | "
                f"id={document.get('id')}",
                flush=True,
            )

        print(
            "[TIMING] Reranker total: "
            f"{time.perf_counter() - start:.2f}s",
            flush=True,
        )

        return selected


rag_reranker = RAGReranker()
