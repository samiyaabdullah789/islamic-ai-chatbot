"""
Run the RAG regression test set and print a score.

Usage (from the project root, Ollama running, ChromaDB ingested):
    python -m scripts.run_eval
    python -m scripts.run_eval --only intentions-en --verbose

Calls the pipeline directly, so nothing is written to PostgreSQL.
"""

import argparse
import asyncio
import contextlib
import io
import json
import re
import time
from datetime import datetime
from pathlib import Path

from app.rag.pipeline import rag_pipeline
from app.rag.reranker import rag_reranker
from app.rag.retriever import rag_retriever


BASE_DIR = Path(__file__).resolve().parents[1]
EVAL_FILE = BASE_DIR / "evals" / "bukhari_eval.json"
RESULTS_DIR = BASE_DIR / "evals" / "results"

CITATION_PATTERN = re.compile(
    r"Volume (\d+), Book (\d+), Hadith (\d+)"
)

FALLBACKS = {
    rag_pipeline._fallback("missing"),
    rag_pipeline._fallback("invalid"),
}


def document_ref(document: dict) -> str:
    metadata = document.get("metadata") or {}

    return (
        f"{metadata.get('volume')}-"
        f"{metadata.get('book')}-"
        f"{metadata.get('hadith_number')}"
    )


async def check_retrieval(case: dict) -> dict:
    # Mirrors the pipeline's retrieval steps so a failure can be
    # traced to search, reranking, or the LLM answer step.
    question = case["question"]
    history = case.get("history", [])

    language = rag_pipeline._detect_language(question)

    query = await rag_pipeline._build_retrieval_query(
        question,
        history,
        language,
    )

    retrieved = await rag_retriever.retrieve(query, limit=10)
    reranked = await rag_reranker.rerank(query, retrieved, limit=5)

    retrieved_refs = [document_ref(d) for d in retrieved]
    top3_refs = [document_ref(d) for d in reranked[:3]]

    expected = set(case.get("expected_refs", []))

    rank = next(
        (
            index + 1
            for index, ref in enumerate(retrieved_refs)
            if ref in expected
        ),
        None,
    )

    return {
        "language": language,
        "query": query,
        "retrieved_refs": retrieved_refs,
        "top3_refs": top3_refs,
        "hit_at_10": rank is not None,
        "rank_at_10": rank,
        "hit_at_top3": bool(expected & set(top3_refs)),
    }


async def run_case(case: dict, verbose: bool) -> dict:
    output = io.StringIO()
    redirect = (
        contextlib.nullcontext()
        if verbose
        else contextlib.redirect_stdout(output)
    )

    start = time.perf_counter()

    try:
        with redirect:
            answer = await rag_pipeline.generate(
                question=case["question"],
                conversation_history=case.get("history", []),
            )

            retrieval = (
                await check_retrieval(case)
                if case["type"] == "answer"
                else None
            )

    except Exception as exc:
        return {
            "id": case["id"],
            "category": case["category"],
            "type": case["type"],
            "status": "ERROR",
            "error": f"{type(exc).__name__}: {exc}",
            "seconds": round(time.perf_counter() - start, 2),
        }

    seconds = round(time.perf_counter() - start, 2)

    cited_refs = [
        "-".join(match)
        for match in CITATION_PATTERN.findall(answer)
    ]

    refused = answer.strip() in FALLBACKS

    if case["type"] == "refuse":
        status = "PASS" if refused else "FAIL"

    elif set(cited_refs) & set(case["expected_refs"]):
        status = "PASS"

    elif cited_refs:
        status = "WRONG_REF"

    else:
        status = "REFUSED"

    return {
        "id": case["id"],
        "category": case["category"],
        "type": case["type"],
        "question": case["question"],
        "status": status,
        "answer": answer,
        "cited_refs": cited_refs,
        "expected_refs": case.get("expected_refs", []),
        "retrieval": retrieval,
        "seconds": seconds,
    }


def print_result(result: dict) -> None:
    line = (
        f"[{result['status']:<9}] "
        f"{result['id']:<18} "
        f"{result['seconds']:>6.1f}s"
    )

    if result["status"] == "ERROR":
        print(f"{line}  {result['error']}")
        return

    if result["type"] == "answer":
        retrieval = result["retrieval"]
        rank = retrieval["rank_at_10"]

        line += (
            f"  cited={','.join(result['cited_refs']) or '-'}"
            f"  top3={'yes' if retrieval['hit_at_top3'] else 'no'}"
            f"  rank@10={rank or '-'}"
        )

    print(line)

    if result["status"] not in {"PASS"}:
        print(f"             answer: {result['answer'][:160]!r}")

        if result["retrieval"]:
            print(f"             query:  {result['retrieval']['query']!r}")


def print_summary(results: list[dict]) -> None:
    answer_cases = [r for r in results if r["type"] == "answer"]
    refuse_cases = [r for r in results if r["type"] == "refuse"]

    def count(items, predicate):
        return sum(1 for item in items if predicate(item))

    def pct(part, total):
        return f"{part}/{total} ({100 * part / total:.0f}%)" if total else "-"

    retrieval_ok = [r for r in answer_cases if r.get("retrieval")]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(
        "Correct answer + citation: ",
        pct(count(answer_cases, lambda r: r["status"] == "PASS"), len(answer_cases)),
    )
    print(
        "Correct refusals:          ",
        pct(count(refuse_cases, lambda r: r["status"] == "PASS"), len(refuse_cases)),
    )
    print(
        "Expected hadith in top 3:  ",
        pct(count(retrieval_ok, lambda r: r["retrieval"]["hit_at_top3"]), len(retrieval_ok)),
    )
    print(
        "Expected hadith in top 10: ",
        pct(count(retrieval_ok, lambda r: r["retrieval"]["hit_at_10"]), len(retrieval_ok)),
    )

    categories = sorted({r["category"] for r in results})

    print("\nBy category:")

    for category in categories:
        items = [r for r in results if r["category"] == category]
        passed = count(items, lambda r: r["status"] == "PASS")
        print(f"  {category:<14} {pct(passed, len(items))}")

    errors = count(results, lambda r: r["status"] == "ERROR")

    if errors:
        print(f"\nErrors: {errors}")

    timed = [r["seconds"] for r in results]
    print(f"\nAverage time per case: {sum(timed) / len(timed):.1f}s")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", help="Run only these case IDs")
    parser.add_argument("--verbose", action="store_true", help="Show pipeline debug output")
    args = parser.parse_args()

    cases = json.loads(EVAL_FILE.read_text(encoding="utf-8"))["cases"]

    if args.only:
        cases = [case for case in cases if case["id"] in args.only]

    print(f"Running {len(cases)} cases...\n")

    results = []

    for case in cases:
        result = await run_case(case, args.verbose)
        print_result(result)
        results.append(result)

    print_summary(results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_file = RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    result_file.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Full results saved to: {result_file.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    asyncio.run(main())
