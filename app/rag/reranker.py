
class CrossEncoderReranker:
    """
    Hybrid reranker using CrossEncoder and the existing
    lexical/semantic RAGReranker.

    Reciprocal Rank Fusion combines rankings without
    assuming that their raw scores share a scale.
    """

    FUSION_K = 60
    CROSS_ENCODER_WEIGHT = 1.0
    HEURISTIC_WEIGHT = 1.0

    def __init__(self):
        self._model = None
        self._lock = threading.Lock()
        self._heuristic = RAGReranker()

    def _load_model(self):
        with self._lock:
            if self._model is None:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(
                    settings.RERANKER_MODEL,
                    max_length=512,
                    device=settings.RERANKER_DEVICE,
                )

        return self._model

    @staticmethod
    def _focused_question(question: str) -> str:
        """
        Remove source/output instructions from the
        ranking query, not from the actual user question.
        """
        focused = str(question).strip()

        focused = re.sub(
            r"\banswer according to\b.*$",
            "",
            focused,
            flags=re.I | re.S,
        )

        focused = re.sub(
            r"\baccording to the available\b.*$",
            "",
            focused,
            flags=re.I | re.S,
        )

        focused = re.sub(
            r"\b(?:please )?(?:give|provide|show) "
            r"(?:me )?(?:the )?(?:source|reference|citation)s?"
            r"\b.*$",
            "",
            focused,
            flags=re.I | re.S,
        )

        focused = focused.strip(" \t\n.!?")

        return focused or question

    def _score(
        self,
        question: str,
        documents: list[dict],
    ) -> list[float]:
        model = self._load_model()

        pairs = [
            (question, str(document.get("content", "")))
            for document in documents
        ]

        return [
            float(score)
            for score in model.predict(pairs)
        ]

    @staticmethod
    def _ranks(scores: list[float]) -> dict[int, int]:
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

        focused_question = self._focused_question(
            question
        )

        print(
            f"[RERANK] Original query: {question}",
            flush=True,
        )
        print(
            f"[RERANK] Focused query: {focused_question}",
            flush=True,
        )

        # CrossEncoder scores the focused question.
        cross_scores = await asyncio.to_thread(
            self._score,
            focused_question,
            documents,
        )

        # Existing heuristic scoring, using the same query.
        concepts = self._heuristic.get_concepts(
            focused_question
        )
        is_definition = (
            self._heuristic._is_definition_question(
                focused_question
            )
        )

        heuristic_scores = [
            self._heuristic._score(
                document,
                concepts,
                is_definition,
            )
            for document in documents
        ]

        cross_ranks = self._ranks(cross_scores)
        heuristic_ranks = self._ranks(
            heuristic_scores
        )

        scored = []

        for index, document in enumerate(documents):
            cross_score = cross_scores[index]
            heuristic_score = heuristic_scores[index]

            fusion_score = (
                self.CROSS_ENCODER_WEIGHT
                / (self.FUSION_K + cross_ranks[index])
                + self.HEURISTIC_WEIGHT
                / (self.FUSION_K + heuristic_ranks[index])
            )

            scored.append({
                **document,
                "rerank_score": fusion_score,
                "cross_encoder_score": cross_score,
                "heuristic_score": heuristic_score,
                "cross_encoder_rank": cross_ranks[index],
                "heuristic_rank": heuristic_ranks[index],
            })

        scored.sort(
            key=lambda document: (
                -document["rerank_score"],
                document["cross_encoder_rank"],
            )
        )

        print(
            "[RERANK DEBUG] Hybrid candidate ranking:",
            flush=True,
        )

        for document in scored:
            print(
                f"[RERANK DEBUG] "
                f"id={document.get('id')} | "
                f"fusion={document['rerank_score']:.6f} | "
                f"cross={document['cross_encoder_score']:.6f} | "
                f"heuristic={document['heuristic_score']:.4f} | "
                f"cross_rank={document['cross_encoder_rank']} | "
                f"heuristic_rank={document['heuristic_rank']}",
                flush=True,
            )

        # Experimental behavior:
        # No fixed raw CrossEncoder threshold here.
        # The existing pipeline must still validate
        # exact quotes and verify the generated answer.
        selected = scored[:limit]

        for index, document in enumerate(
            selected,
            start=1,
        ):
            metadata = document.get("metadata") or {}

            print(
                f"[RERANK] Selected {index}: "
                f"source={metadata.get('source_name', 'Unknown')} | "
                f"score={document['rerank_score']:.6f} | "
                f"id={document.get('id')}",
                flush=True,
            )

        print(
            "[TIMING] Hybrid reranker total: "
            f"{time.perf_counter() - start:.2f}s",
            flush=True,
        )

        return selected


rag_reranker = (
    CrossEncoderReranker()
    if settings.RERANKER == "cross_encoder"
    else RAGReranker()
)
