"""
Run a RAG regression test set and print a score.

Usage (from the project root, Ollama running, ChromaDB ingested):
    python -m scripts.run_eval
    python -m scripts.run_eval --only intentions-en --verbose
    python -m scripts.run_eval --eval-file evals/seekers_eval.json

Calls the pipeline directly, so nothing is written to PostgreSQL.
The only file written is the results JSON in evals/results/.

Eval file format
----------------
{
  "description": "...",
  "source_id": 1,              # optional; pins RAG_TEST_SOURCE_ID
  "ref_type": "bukhari",       # "bukhari" (default) or "doc_id"
  "cases": [...]
}

Case fields:
  id, category, type ("answer" | "refuse"), question
  history          optional conversation history
  expected_refs    bukhari: ["volume-book-number", ...]
  expected_doc_ids doc_id: exact Chroma IDs or ID prefixes,
                   e.g. "source-2-chunk-61-" matches every chunk
                   of record 61 in source 2
  language         optional: "english" | "roman_urdu" | "urdu"
                   (defaults to the category when it is one of these)
  madhhab          optional: "Hanafi", "Shafi'i", "Maliki", "Hanbali"

Scoring
-------
The PASS / FAIL / WRONG_REF / REFUSED rules for "bukhari" sets are
unchanged. The quality metrics (answer language, repetition, citation
presence, refusal stage, prompt tokens, LLM calls) are reported only;
they never change a case's status.

"doc_id" sets cannot check the cited record yet, because SeekersGuidance
citations do not carry record IDs. Until they do, an answer case passes
when the pipeline answers and an expected record was among the passages
sent to the LLM (top CONTEXT_SIZE after reranking).
"""

import argparse
import asyncio
import contextlib
import io
import json
import os
import re
import time
from collections import Counter
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

from app.rag.pipeline import rag_pipeline
from app.rag.reranker import rag_reranker
from app.rag.retriever import rag_retriever


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_FILE = BASE_DIR / "evals" / "bukhari_eval.json"
RESULTS_DIR = BASE_DIR / "evals" / "results"

CITATION_PATTERN = re.compile(
    r"Volume (\d+), Book (\d+), Hadith (\d+)"
)

FALLBACKS = {
    rag_pipeline.MISSING_MESSAGE,
    rag_pipeline.INVALID_MESSAGE,
}

# Number of reranked passages the pipeline sends to the LLM
CONTEXT_SIZE = rag_pipeline.MAX_PASSAGES

LANGUAGES = {"english", "roman_urdu", "urdu"}

# Log lines printed by the pipeline and the Ollama client.
REFUSAL_LINE = re.compile(r"\[RAG\] Refusal stage: (.+)")
RETRY_LINE = re.compile(r"\[RAG RETRY\]")
PROMPT_TOKENS_LINE = re.compile(r"\[OLLAMA\] Prompt tokens: (\d+)")
LLM_CALL_LINE = re.compile(r"\[OLLAMA\] Task: (\w+)")
TRUNCATION_LINE = re.compile(r"\[WARNING\] Prompt token count reached")

# Common Roman Urdu function words. Deliberately independent of the
# pipeline's own language detector, so the eval does not grade the
# detector with itself. A heuristic: report-only, never scored.
ROMAN_URDU_WORDS = {
    # Words that are also common English words or Islamic names
    # ("to", "main", "par", "ya" as in "Ya Allah") are left out.
    "hai", "hain", "hota", "hoti", "hote", "ka", "ki", "ke", "ko",
    "se", "mein", "aur", "nahi", "nahin", "kya", "kyun",
    "kaise", "kis", "kab", "agar", "bhi", "sakta", "sakti", "sakte",
    "chahiye", "karna", "karni", "karne", "karein", "wala", "wali",
    "liye", "jab", "yeh", "woh", "apne", "apni",
    "unhon", "gaya", "gayi", "diya", "kiya", "jata", "jati",
}
URDU_SCRIPT = re.compile(r"[؀-ۿ]")


class Tee(io.TextIOBase):
    """Capture output for log parsing while optionally echoing it."""

    def __init__(self, echo: bool):
        self.buffer_text = io.StringIO()
        self.echo = echo

    def write(self, text):
        self.buffer_text.write(text)
        if self.echo:
            import sys
            sys.__stdout__.write(text)
        return len(text)

    def getvalue(self):
        return self.buffer_text.getvalue()


def document_ref(document: dict, ref_type: str) -> str:
    if ref_type == "doc_id":
        return str(document.get("id") or "")

    metadata = document.get("metadata") or {}

    return (
        f"{metadata.get('volume')}-"
        f"{metadata.get('book')}-"
        f"{metadata.get('hadith_number')}"
    )


def matches_expected(ref: str, case: dict, ref_type: str) -> bool:
    if ref_type == "doc_id":
        return any(
            ref == expected or ref.startswith(expected)
            for expected in case.get("expected_doc_ids", [])
        )

    return ref in set(case.get("expected_refs", []))


def split_answer_and_citations(answer: str) -> tuple[str, list[str]]:
    """The pipeline appends citation lines of the form '[...]'."""
    body_lines = []
    citations = []

    for line in str(answer).splitlines():
        stripped = line.strip()

        if stripped.startswith("[") and stripped.endswith("]"):
            citations.append(stripped)
        else:
            body_lines.append(line)

    return "\n".join(body_lines).strip(), citations


def detect_answer_language(text: str) -> str | None:
    text = str(text or "").strip()

    if not text:
        return None

    letters = [ch for ch in text if ch.isalpha()]

    if letters:
        urdu_letters = sum(1 for ch in letters if URDU_SCRIPT.match(ch))

        if urdu_letters / len(letters) >= 0.3:
            return "urdu"

    # Keep apostrophes inside words so "Ka'bah" is not read as "ka".
    words = re.findall(r"[a-z]+(?:['\u2019][a-z]+)*", text.casefold())

    if not words:
        return None

    hits = sum(1 for word in words if word in ROMAN_URDU_WORDS)

    if hits >= 2 and hits / len(words) >= 0.08:
        return "roman_urdu"

    return "english"


def sentence_key(sentence: str) -> str:
    return " ".join(re.findall(r"[^\W_]+", sentence.casefold()))


def repetition_report(body: str) -> dict:
    sentences = [
        part.strip()
        for part in re.split(r"(?<=[.!?۔])\s+|\n+", body)
        if part.strip()
    ]
    keys = [sentence_key(s) for s in sentences]
    keys = [k for k in keys if len(k.split()) >= 3]

    exact_duplicates = len(keys) - len(set(keys))
    near_duplicates = 0
    max_similarity = 0.0

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if keys[i] == keys[j]:
                continue

            ratio = SequenceMatcher(None, keys[i], keys[j]).ratio()
            max_similarity = max(max_similarity, ratio)

            if ratio >= 0.85:
                near_duplicates += 1

    return {
        "sentences": len(sentences),
        "exact_duplicates": exact_duplicates,
        "near_duplicates": near_duplicates,
        "max_similarity": round(max_similarity, 3),
        "repetitive": bool(exact_duplicates or near_duplicates),
    }


def parse_pipeline_log(log: str) -> dict:
    refusals = REFUSAL_LINE.findall(log)
    prompt_tokens = [int(n) for n in PROMPT_TOKENS_LINE.findall(log)]

    return {
        "refusal_stage": refusals[-1].strip() if refusals else None,
        "retries": len(RETRY_LINE.findall(log)),
        "llm_calls": len(LLM_CALL_LINE.findall(log)),
        "llm_tasks": dict(Counter(LLM_CALL_LINE.findall(log))),
        "max_prompt_tokens": max(prompt_tokens) if prompt_tokens else None,
        "truncation_warnings": len(TRUNCATION_LINE.findall(log)),
    }


def expected_language(case: dict) -> str | None:
    language = case.get("language")

    if language in LANGUAGES:
        return language

    if case.get("category") in LANGUAGES:
        return case["category"]

    return None


def quality_metrics(case: dict, answer: str, log: str, refused: bool) -> dict:
    body, citations = split_answer_and_citations(answer)
    wanted = expected_language(case)
    got = detect_answer_language(body)

    return {
        "expected_language": wanted,
        "answer_language": got,
        "language_match": (
            None if wanted is None or got is None else wanted == got
        ),
        "has_citation": bool(citations),
        "citations": citations,
        "repetition": repetition_report(body),
        "refused": refused,
        **parse_pipeline_log(log),
    }


async def check_retrieval(case: dict, ref_type: str) -> dict:
    # Mirrors the pipeline's retrieval steps so a failure can be
    # traced to search, reranking, or the LLM answer step.
    question = case["question"]
    history = case.get("history", [])
    madhhab = case.get("madhhab")

    query = await rag_pipeline._build_retrieval_query(
        question,
        history,
    )

    retrieve_args = {
        "limit": rag_pipeline.RETRIEVAL_LIMIT,
        "general_question": rag_retriever.is_clearly_general_question(query),
    }

    if madhhab is not None:
        retrieve_args["madhhab"] = madhhab

    retrieved = await rag_retriever.retrieve(query, **retrieve_args)
    reranked = await rag_reranker.rerank(query, retrieved, limit=len(retrieved))

    retrieved_refs = [document_ref(d, ref_type) for d in retrieved]
    reranked_refs = [document_ref(d, ref_type) for d in reranked]
    context_refs = reranked_refs[:CONTEXT_SIZE]

    rank = next(
        (
            index + 1
            for index, ref in enumerate(retrieved_refs)
            if matches_expected(ref, case, ref_type)
        ),
        None,
    )

    rerank_scores = {
        document_ref(d, ref_type): round(d["rerank_score"], 3)
        for d in reranked
        if "rerank_score" in d
    }

    return {
        "query": query,
        "retrieved_refs": retrieved_refs,
        "context_refs": context_refs,
        "context_scores": [rerank_scores.get(ref) for ref in context_refs],
        "expected_scores": {
            ref: rerank_scores.get(ref)
            for ref in retrieved_refs
            if matches_expected(ref, case, ref_type)
        },
        "retrieved": rank is not None,
        "retrieved_rank": rank,
        "in_context": any(
            matches_expected(ref, case, ref_type) for ref in context_refs
        ),
    }


def score_case(case: dict, answer: str, retrieval: dict, ref_type: str) -> tuple[str, list[str], bool]:
    refused = answer.strip() in FALLBACKS

    if ref_type == "doc_id":
        # Record IDs are not in SeekersGuidance citations yet.
        if case["type"] == "refuse":
            return ("PASS" if refused else "FAIL"), [], refused

        if refused:
            return "REFUSED", [], refused

        if retrieval["in_context"]:
            return "PASS", [], refused

        return "NO_EVIDENCE", [], refused

    # Bukhari scoring: unchanged from the original runner.
    cited_refs = [
        "-".join(match)
        for match in CITATION_PATTERN.findall(answer)
    ]

    if case["type"] == "refuse":
        status = "PASS" if refused else "FAIL"

    elif set(cited_refs) & set(case["expected_refs"]):
        status = "PASS"

    elif cited_refs:
        status = "WRONG_REF"

    else:
        status = "REFUSED"

    return status, cited_refs, refused


async def run_case(case: dict, verbose: bool, ref_type: str) -> dict:
    answer_log = Tee(echo=verbose)
    retrieval_log = Tee(echo=verbose)

    generate_args = {
        "question": case["question"],
        "conversation_history": case.get("history", []),
    }

    if case.get("madhhab") is not None:
        generate_args["madhhab"] = case["madhhab"]

    start = time.perf_counter()

    try:
        with contextlib.redirect_stdout(answer_log):
            answer = await rag_pipeline.generate(**generate_args)

        answer_seconds = round(time.perf_counter() - start, 2)

        with contextlib.redirect_stdout(retrieval_log):
            retrieval = await check_retrieval(case, ref_type)

    except Exception as exc:
        return {
            "id": case["id"],
            "category": case["category"],
            "type": case["type"],
            "status": "ERROR",
            "error": f"{type(exc).__name__}: {exc}",
            "seconds": round(time.perf_counter() - start, 2),
            "metrics": parse_pipeline_log(answer_log.getvalue()),
        }

    seconds = round(time.perf_counter() - start, 2)

    status, cited_refs, refused = score_case(case, answer, retrieval, ref_type)

    return {
        "id": case["id"],
        "category": case["category"],
        "type": case["type"],
        "question": case["question"],
        "status": status,
        "answer": answer,
        "cited_refs": cited_refs,
        "expected_refs": case.get(
            "expected_doc_ids" if ref_type == "doc_id" else "expected_refs",
            [],
        ),
        "retrieval": retrieval,
        "metrics": quality_metrics(case, answer, answer_log.getvalue(), refused),
        "answer_seconds": answer_seconds,
        "seconds": seconds,
    }


def print_result(result: dict) -> None:
    line = (
        f"[{result['status']:<11}] "
        f"{result['id']:<18} "
        f"{result['seconds']:>6.1f}s"
    )

    if result["status"] == "ERROR":
        print(f"{line}  {result['error']}")
        return

    metrics = result["metrics"]

    if result["type"] == "answer":
        retrieval = result["retrieval"]
        rank = retrieval["retrieved_rank"]

        line += (
            f"  cited={','.join(result['cited_refs']) or '-'}"
            f"  in_context={'yes' if retrieval['in_context'] else 'no'}"
            f"  retrieved_rank={rank or '-'}"
        )

    else:
        best = result["retrieval"]["context_scores"][:1]

        if best and best[0] is not None:
            line += f"  best_rerank_score={best[0]}"

    flags = []

    if metrics["language_match"] is False:
        flags.append(
            f"lang={metrics['answer_language']}!={metrics['expected_language']}"
        )

    if metrics["repetition"]["repetitive"]:
        flags.append("repetitive")

    if not metrics["refused"] and not metrics["has_citation"]:
        flags.append("no_citation")

    if metrics["refusal_stage"]:
        flags.append(f"refusal='{metrics['refusal_stage']}'")

    if metrics["truncation_warnings"]:
        flags.append("prompt_at_ctx_limit")

    if flags:
        line += "  [" + " | ".join(flags) + "]"

    print(line)

    if result["status"] not in {"PASS"}:
        print(f"             answer: {result['answer'][:160]!r}")

        if result["retrieval"]:
            retrieval = result["retrieval"]
            print(f"             query:  {retrieval['query']!r}")

            if any(score is not None for score in retrieval["context_scores"]):
                print(
                    f"             context: {list(zip(retrieval['context_refs'], retrieval['context_scores']))}"
                    f"  expected: {retrieval['expected_scores']}"
                )


def pct(part, total):
    return f"{part}/{total} ({100 * part / total:.0f}%)" if total else "-"


def print_summary(results: list[dict], ref_type: str) -> None:
    answer_cases = [r for r in results if r["type"] == "answer"]
    refuse_cases = [r for r in results if r["type"] == "refuse"]

    def count(items, predicate):
        return sum(1 for item in items if predicate(item))

    retrieval_ok = [r for r in answer_cases if r.get("retrieval")]
    unit = "record" if ref_type == "doc_id" else "hadith"

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if ref_type == "doc_id":
        print(
            "Answered with expected record in context:",
            pct(count(answer_cases, lambda r: r["status"] == "PASS"), len(answer_cases)),
        )
    else:
        print(
            "Correct answer + citation: ",
            pct(count(answer_cases, lambda r: r["status"] == "PASS"), len(answer_cases)),
        )

    print(
        "Correct refusals:          ",
        pct(count(refuse_cases, lambda r: r["status"] == "PASS"), len(refuse_cases)),
    )
    print(
        f"Expected {unit} sent to LLM (top {CONTEXT_SIZE}):",
        pct(count(retrieval_ok, lambda r: r["retrieval"]["in_context"]), len(retrieval_ok)),
    )
    print(
        f"Expected {unit} retrieved:          ",
        pct(count(retrieval_ok, lambda r: r["retrieval"]["retrieved"]), len(retrieval_ok)),
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

    if timed:
        print(f"\nAverage time per case: {sum(timed) / len(timed):.1f}s")
        print(f"Slowest case:          {max(timed):.1f}s")

    print_quality_summary(results)


def print_quality_summary(results: list[dict]) -> None:
    scored = [r for r in results if r["status"] != "ERROR"]

    if not scored:
        return

    def count(items, predicate):
        return sum(1 for item in items if predicate(item))

    answered = [r for r in scored if not r["metrics"]["refused"]]

    print("\n" + "-" * 60)
    print("QUALITY METRICS (report only, not part of the score)")
    print("-" * 60)

    print("Answer language matches question:")

    for language in sorted(LANGUAGES):
        items = [
            r for r in scored
            if r["metrics"]["expected_language"] == language
        ]

        if items:
            print(
                f"  {language:<12} "
                f"{pct(count(items, lambda r: r['metrics']['language_match'] is True), len(items))}"
            )

    print(
        "Answers with repetition:  ",
        pct(count(answered, lambda r: r["metrics"]["repetition"]["repetitive"]), len(answered)),
    )
    print(
        "Answers with a citation:  ",
        pct(count(answered, lambda r: r["metrics"]["has_citation"]), len(answered)),
    )

    stages = Counter(
        r["metrics"]["refusal_stage"]
        for r in scored
        if r["metrics"]["refusal_stage"]
    )

    if stages:
        print("Refusal stages:")

        for stage, n in stages.most_common():
            print(f"  {n:>3}  {stage}")

    tokens = [
        r["metrics"]["max_prompt_tokens"]
        for r in scored
        if r["metrics"]["max_prompt_tokens"] is not None
    ]

    if tokens:
        print(f"Max prompt tokens:         {max(tokens)} (median {sorted(tokens)[len(tokens) // 2]})")

    truncated = count(scored, lambda r: r["metrics"]["truncation_warnings"] > 0)
    print(f"Cases at context limit:    {truncated}")

    calls = [r["metrics"]["llm_calls"] for r in scored]
    print(f"LLM calls per case:        avg {sum(calls) / len(calls):.1f}, max {max(calls)}")

    retried = count(scored, lambda r: r["metrics"]["retries"] > 0)
    print(f"Cases that retried:        {retried}")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-file",
        type=Path,
        default=DEFAULT_EVAL_FILE,
        help="Eval set to run (default: evals/bukhari_eval.json)",
    )
    parser.add_argument("--only", nargs="+", help="Run only these case IDs")
    parser.add_argument("--verbose", action="store_true", help="Show pipeline debug output")
    args = parser.parse_args()

    eval_file = args.eval_file

    if not eval_file.is_absolute():
        eval_file = (Path.cwd() / eval_file).resolve()

    eval_set = json.loads(eval_file.read_text(encoding="utf-8"))
    cases = eval_set["cases"]
    ref_type = eval_set.get("ref_type", "bukhari")

    if ref_type not in {"bukhari", "doc_id"}:
        raise SystemExit(f"Unsupported ref_type: {ref_type}")

    # The retriever reads RAG_TEST_SOURCE_ID on every call; pin it to
    # the source this test set was written for.
    if eval_set.get("source_id") is not None:
        os.environ["RAG_TEST_SOURCE_ID"] = str(eval_set["source_id"])

    if args.only:
        cases = [case for case in cases if case["id"] in args.only]

    print(f"Eval set: {eval_file.name} ({ref_type})")
    print(f"RAG_TEST_SOURCE_ID: {os.getenv('RAG_TEST_SOURCE_ID') or 'All'}")
    print(f"Running {len(cases)} cases...\n")

    results = []

    for case in cases:
        result = await run_case(case, args.verbose, ref_type)
        print_result(result)
        results.append(result)

    print_summary(results, ref_type)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_file = RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{eval_file.stem}.json"
    result_file.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Full results saved to: {result_file.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    asyncio.run(main())
