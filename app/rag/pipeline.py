import json
import re
import time

from app.rag.retriever import rag_retriever
from app.rag.reranker import rag_reranker
from app.llm.client import ollama_client


class RAGPipeline:

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip().casefold()

    @staticmethod
    def _words(text: str) -> set[str]:
        return set(re.findall(r"\b\w+\b", text.lower()))

    @staticmethod
    def _is_hadith_request(question: str) -> bool:
        return bool(
            re.search(
                r"\b(hadith|hadees|hadeeth|hadis)\b",
                question,
                re.IGNORECASE,
            )
        )

    @staticmethod
    def _build_citation(document: dict) -> str:
        metadata = document.get("metadata") or {}

        parts = [
            str(metadata.get("source_name") or "Unknown source")
        ]

        for key, label in (
            ("volume", "Volume"),
            ("book", "Book"),
            ("hadith_number", "Hadith"),
        ):
            value = metadata.get(key)

            if value is not None:
                parts.append(f"{label} {value}")

        return "[" + ", ".join(parts) + "]"

    def _validated_source_indexes(
        self,
        evidence: object,
        documents: list[dict],
    ) -> list[int]:

        if not isinstance(evidence, list):
            return []

        valid = []

        for item in evidence:

            if not isinstance(item, dict):
                continue

            index = item.get("source_index")
            quote = item.get("quote")

            if type(index) is not int:
                continue

            if not 0 <= index < len(documents):
                continue

            if not isinstance(quote, str):
                continue

            normalized_quote = self._normalize_text(quote)

            normalized_source = self._normalize_text(
                documents[index].get("content", "")
            )

            if len(normalized_quote.split()) < 6:
                continue

            if normalized_quote not in normalized_source:
                continue

            if index not in valid:
                valid.append(index)

        return valid

    async def _build_retrieval_query(
        self,
        question: str,
        conversation_history: list[dict[str, str]],
    ) -> str:

        question = question.strip()

        if not conversation_history:
            return question

        words = self._words(question)

        references = {
            "it", "its", "this", "that",
            "these", "those", "they", "them",
            "their", "he", "she", "his",
            "her", "ones",
        }

        follow_up_starts = (
            "what about",
            "how about",
            "and ",
            "but ",
            "why ",
        )

        needs_context = (
            bool(words & references)
            or question.lower().startswith(follow_up_starts)
        )

        if not needs_context:
            return question

        history_text = "\n".join(
            f"{item.get('role', 'user')}: "
            f"{item.get('content', '')}"
            for item in conversation_history[-4:]
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "Rewrite the latest question as a standalone "
                    "Islamic search query. Use conversation history "
                    "only to resolve references. "
                    "Return only the rewritten query."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"History:\n{history_text}\n\n"
                    f"Question:\n{question}"
                ),
            },
        ]

        rewritten = await ollama_client.generate(messages)

        return rewritten.strip() or question

    async def generate(
        self,
        question: str,
        conversation_history: list[dict[str, str]],
    ) -> str:

        pipeline_start = time.perf_counter()

        # STEP 1: Build retrieval query

        start = time.perf_counter()

        retrieval_query = await self._build_retrieval_query(
            question,
            conversation_history,
        )

        print(
            f"[TIMING] Query rewriting: "
            f"{time.perf_counter() - start:.2f}s",
            flush=True,
        )

        # STEP 2: Retrieve source passages

        start = time.perf_counter()

        documents = await rag_retriever.retrieve(
            retrieval_query,
            limit=10,
        )

        print(
            f"[TIMING] ChromaDB retrieval: "
            f"{time.perf_counter() - start:.2f}s",
            flush=True,
        )

        if not documents:
            return (
                "I couldn't find relevant passages "
                "in the available Islamic sources."
            )

        # STEP 3: Rerank passages

        start = time.perf_counter()

        documents = await rag_reranker.rerank(
            question=retrieval_query,
            documents=documents,
            limit=5,
        )

        print(
            f"[TIMING] Reranking: "
            f"{time.perf_counter() - start:.2f}s",
            flush=True,
        )

        if not documents:
            return (
                "I couldn't find relevant passages "
                "in the available Islamic sources."
            )

        # STEP 4: Identify request type

        hadith_request = self._is_hadith_request(question)

        source_context = "\n\n".join(
            f"PASSAGE {index}\n"
            f"{document.get('content', '')}"
            for index, document in enumerate(documents)
        )

        # STEP 5: Prepare task instructions

        if hadith_request:

            task = """
The user is requesting a Hadith.

Your task:
1. Read the user's exact question.
2. Select ONE Hadith directly relevant to the topic.
3. Identify a short exact quote that proves relevance.
4. Do not rewrite or summarize the Hadith.
5. Set the answer field to "selected".
6. The application will return the original source text.
7. If no Hadith directly matches, return an empty answer
   and empty evidence.
"""

        else:

            task = """
The user is asking a general Islamic question.

Your task:
1. Understand exactly what the user is asking.
2. Read the available source passages carefully.
3. Identify the statement that DIRECTLY answers
   the user's question.
4. Use that direct statement as your primary evidence.
5. Generate a clear, natural answer in your own words.
6. Preserve the original meaning and context.
7. Do not add conditions, exceptions, restrictions,
   or conclusions that are not directly supported
   by the selected evidence.
8. If a passage contains multiple statements,
   use the statement that answers the question,
   not an unrelated statement from the same passage.
9. If a passage describes one specific incident,
   explain it as that incident. Do not invent a
   universal rule from it.
10. Do not combine unrelated statements into
    a new Islamic ruling.
11. If the sources only support a limited answer,
    give that limited answer rather than guessing.
12. If the sources do not answer the question,
    return an empty answer and empty evidence.

IMPORTANT:
The evidence quote must support the actual answer,
not merely be present somewhere in the same passage.

Before returning JSON, internally check:
- Does my answer directly address the question?
- Does my selected evidence support my answer?
- Have I added any unsupported condition?
- Have I changed the meaning of the source?

If any claim is unsupported, remove that claim.
"""

        # STEP 6: Build LLM prompt

        system_prompt = f"""
You are a careful Islamic knowledge assistant.

Answer questions using ONLY the supplied Islamic sources.

{task}

GENERAL RULES:

1. Never invent Islamic facts, Hadith text,
   Quranic verses, or references.

2. Never use outside knowledge to fill gaps
   in the supplied passages.

3. Never misrepresent the original meaning
   or context of a source.

4. Keep answers concise, clear, and natural.

5. For general questions, explain the source
   in your own words.

6. For Hadith requests, select the Hadith;
   the application returns the original text.

7. Evidence must directly support the answer.

8. Evidence quotes must be copied exactly
   from the supplied source passage.

9. Each evidence quote must contain
   6 to 15 continuous words.

10. Do not change punctuation, capitalization,
    spelling, or wording inside evidence quotes.

11. If the sources are insufficient,
    do not guess.

12. Return valid JSON only.

JSON FORMAT:

{{
    "answer": "Source-supported answer",
    "evidence": [
        {{
            "source_index": 0,
            "quote": "Exact continuous quote from passage"
        }}
    ]
}}

If no answer is supported:

{{
    "answer": "",
    "evidence": []
}}

SOURCE PASSAGES:

{source_context}
"""

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            *conversation_history[-4:],
            {
                "role": "user",
                "content": question,
            },
        ]

        # STEP 7: Generate answer using Ollama

        start = time.perf_counter()

        raw_response = await ollama_client.generate(
            messages,
            json_mode=True,
        )

        print(
            f"[TIMING] Final LLM generation: "
            f"{time.perf_counter() - start:.2f}s",
            flush=True,
        )

        # STEP 8: Debug logs

        print("\n===== RAG DEBUG =====", flush=True)
        print("Question:", question, flush=True)

        for index, document in enumerate(documents):

            print(
                f"PASSAGE {index}",
                flush=True,
            )

            print(
                "Content:",
                document.get("content"),
                flush=True,
            )

            print(
                "Metadata:",
                document.get("metadata"),
                flush=True,
            )

        print(
            "LLM raw response:",
            raw_response,
            flush=True,
        )

        print("=====================\n", flush=True)

        # STEP 9: Parse LLM JSON

        try:

            cleaned = raw_response.strip()

            if cleaned.startswith("```"):
                cleaned = re.sub(
                    r"^```(?:json)?\s*|\s*```$",
                    "",
                    cleaned,
                    flags=re.IGNORECASE,
                ).strip()

            result = json.loads(cleaned)

            if not isinstance(result, dict):
                raise ValueError("Invalid JSON structure")

            answer = result.get("answer", "")

            if not isinstance(answer, str):
                raise ValueError("Invalid answer")

            answer = answer.strip()

            evidence = result.get("evidence", [])

        except (ValueError, TypeError):

            return (
                "I couldn't generate a reliable "
                "source-based answer."
            )

        if not answer:
            return (
                "The available passages don't provide "
                "enough evidence to answer this question."
            )

        # STEP 10: Verify source evidence

        valid_indexes = self._validated_source_indexes(
            evidence,
            documents,
        )

        if not valid_indexes:
            return (
                "I couldn't verify supporting evidence "
                "for this answer."
            )

        # STEP 11: Original Hadith output

        if hadith_request:

            selected_index = valid_indexes[0]

            answer = documents[selected_index].get(
                "content", ""
            ).strip()

            valid_indexes = [selected_index]

        if not answer:
            return "No verified source text is available."

        # STEP 12: Attach citations

        citations = list(
            dict.fromkeys(
                self._build_citation(documents[index])
                for index in valid_indexes
            )
        )

        print(
            f"[TIMING] RAG pipeline total: "
            f"{time.perf_counter() - pipeline_start:.2f}s",
            flush=True,
        )

        return (
            f"{answer}\n\n"
            + "\n".join(citations)
        )


rag_pipeline = RAGPipeline()