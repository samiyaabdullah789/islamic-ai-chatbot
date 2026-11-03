
import asyncio
import time

from sentence_transformers import CrossEncoder


class QAReranker:

    def __init__(self):
        self.model_name = "cross-encoder/ms-marco-MiniLM-L6-v2"
        self.model = None

    def _load_model(self):
        if self.model is None:
            self.model = CrossEncoder(
                self.model_name,
                device="cpu",
            )

    def _predict(self, pairs):
        return self.model.predict(
            pairs,
            batch_size=16,
            show_progress_bar=False,
        )

    def _rerank_sync(
        self,
        question: str,
        documents: list[dict],
        limit: int,
    ) -> list[dict]:

        self._load_model()

        # Stage 1: Question-only relevance
        pairs = [
            (question, str(doc.get("question") or ""))
            for doc in documents
        ]

        question_scores = self._predict(pairs)

        stage1 = sorted(
            [
                {
                    **doc,
                    "question_score": float(score),
                    "hybrid_rank": index + 1,
                }
                for index, (doc, score) in enumerate(
                    zip(documents, question_scores)
                )
            ],
            key=lambda x: x["question_score"],
            reverse=True,
        )

        # Keep both model-ranked and hybrid-ranked records.
        # Avoid losing relevant records like SG-00062.
        shortlist = []
        seen = set()

        for doc in stage1[:40] + sorted(
            stage1,
            key=lambda x: x["hybrid_rank"],
        )[:40]:

            doc_id = doc.get("id")

            if doc_id in seen:
                continue

            seen.add(doc_id)
            shortlist.append(doc)

        # Stage 2: Evaluate question + supporting answer
        answer_pairs = []

        for doc in shortlist:
            source_text = (
                f"Question: {doc.get('question') or ''}\n"
                f"Answer: {str(doc.get('answer') or '')[:1000]}"
            )

            answer_pairs.append(
                (question, source_text)
            )

        answer_scores = self._predict(answer_pairs)

        ranked = sorted(
            [
                {
                    **doc,
                    "rerank_score": float(score),
                }
                for doc, score in zip(
                    shortlist,
                    answer_scores,
                )
            ],
            key=lambda x: x["rerank_score"],
            reverse=True,
        )

        results = []
        seen_questions = set()

        for doc in ranked:
            normalized = " ".join(
                str(doc.get("question") or "")
                .casefold()
                .split()
            )

            if normalized in seen_questions:
                continue

            seen_questions.add(normalized)
            results.append(doc)

            if len(results) >= limit:
                break

        return results

    async def rerank(
        self,
        question: str,
        documents: list[dict],
        limit: int = 5,
    ) -> list[dict]:

        if not documents or limit <= 0:
            return []

        start = time.perf_counter()

        results = await asyncio.to_thread(
            self._rerank_sync,
            question,
            documents,
            limit,
        )

        print(
            f"[TIMING] Two-stage QA reranking: "
            f"{time.perf_counter() - start:.2f}s",
            flush=True,
        )

        for rank, doc in enumerate(results, 1):
            print(
                f"[QA RERANK] {rank}. "
                f"{doc.get('id')} | "
                f"score={doc['rerank_score']:.4f}",
                flush=True,
            )

        return results


qa_reranker = QAReranker()
