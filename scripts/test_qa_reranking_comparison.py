
import asyncio
import time

from sentence_transformers import CrossEncoder

from app.rag.qa_hybrid_retriever import qa_hybrid_retriever


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"

TEST_CASES = [
    {
        "query": "Is Witr prayer wajib in the Hanafi school?",
        "expected_ids": ["qa-source-2-record-61"],
    },
    {
        "query": "What are the proofs for forgetful prostration in the Hanafi madhhab?",
        "expected_ids": [],
    },
    {
        "query": "When should sajda al-sahw be performed?",
        "expected_ids": [],
    },
]

CANDIDATE_LIMIT = 100
ANSWER_MAX_CHARS = 1000
TOP_K = 5


def rerank(model, query, records, include_answer):
    pairs = []

    for record in records:
        source_text = str(record.get("question", ""))

        if include_answer:
            answer = str(record.get("answer", ""))
            source_text += "\nAnswer: " + answer[:ANSWER_MAX_CHARS]

        pairs.append((query, source_text))

    scores = model.predict(
        pairs,
        batch_size=16,
        show_progress_bar=False,
    )

    ranked = sorted(
        zip(scores, records),
        key=lambda item: float(item[0]),
        reverse=True,
    )

    return [
        {
            **record,
            "rerank_score": float(score),
        }
        for score, record in ranked
    ]


def show_results(title, results, expected_ids):
    print(f"\n{title}")

    for rank, record in enumerate(results[:TOP_K], 1):
        print(
            f"{rank}. "
            f"{record.get('id')} | "
            f"score={record['rerank_score']:.4f} | "
            f"{record.get('question', '')[:110]}"
        )

    if expected_ids:
        matches = [
            rank
            for rank, record in enumerate(results, 1)
            if record.get("id") in expected_ids
        ]

        print(
            "Expected source rank:",
            min(matches) if matches else "NOT FOUND",
        )


async def main():
    print("Loading CrossEncoder model...")
    model = CrossEncoder(MODEL_NAME, device="cpu")

    for case in TEST_CASES:
        query = case["query"]
        expected_ids = case["expected_ids"]

        print("\n" + "=" * 65)
        print("QUESTION:", query)

        candidates = await qa_hybrid_retriever.retrieve(
            query=query,
            limit=CANDIDATE_LIMIT,
            dense_limit=CANDIDATE_LIMIT,
            bm25_limit=CANDIDATE_LIMIT,
        )

        print("Hybrid candidates:", len(candidates))

        start = time.perf_counter()

        question_only = await asyncio.to_thread(
            rerank,
            model,
            query,
            candidates,
            False,
        )

        print(
            f"[TIMING] Question-only: "
            f"{time.perf_counter() - start:.2f}s"
        )

        start = time.perf_counter()

        question_answer = await asyncio.to_thread(
            rerank,
            model,
            query,
            candidates,
            True,
        )

        print(
            f"[TIMING] Question+Answer: "
            f"{time.perf_counter() - start:.2f}s"
        )

        show_results(
            "QUESTION-ONLY RESULTS",
            question_only,
            expected_ids,
        )

        show_results(
            "QUESTION+ANSWER RESULTS",
            question_answer,
            expected_ids,
        )


if __name__ == "__main__":
    asyncio.run(main())
