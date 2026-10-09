
import json
import re
import time

from app.rag.retriever import rag_retriever
from app.rag.reranker import rag_reranker
from app.llm.client import ollama_client


class RAGPipeline:

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(
            r"\s+", " ", text
        ).strip().casefold()

    @staticmethod
    def _words(text: str) -> set[str]:
        return set(
            re.findall(r"\b\w+\b", text.lower())
        )

    @staticmethod
    def _is_hadith_request(question: str) -> bool:
        return bool(
            re.search(
                r"\b(hadith|hadees|hadeeth|hadis)\b",
                question,
                re.IGNORECASE,
            )
        )

    def _is_prayer_virtue_question(
        self,
        question: str,
    ) -> bool:

        words = self._words(question)

        return bool(
            words & {
                "namaz", "salah", "salat",
                "prayer", "prayers"
            }
        ) and bool(
            words & {
                "importance", "virtue", "virtues",
                "reward", "rewards", "benefit",
                "benefits"
            }
        )

    def _select_prayer_reward_hadith(
        self,
        documents: list[dict],
    ) -> dict | None:

        phrases = (
            "reward of the prayer",
            "reward of prayer",
            "reward of the prayers",
            "reward of congregational prayer",
            "reward of the noon prayer",
        )

        for document in documents:
            content = document.get("content", "")
            normalized = self._normalize_text(content)
            words = self._words(content)

            if len(words) < 35:
                continue

            if not (
                words & {
                    "prayer", "prayers",
                    "praying", "salah"
                }
            ):
                continue

            if not words & {"reward", "rewards"}:
                continue

            if any(
                phrase in normalized
                for phrase in phrases
            ):
                return document

        return None

    def _select_fast_hadith(
        self,
        question: str,
        documents: list[dict],
    ) -> dict | None:

        words = self._words(question)

        prayer_words = {
            "namaz", "salah", "salat",
            "prayer", "prayers"
        }

        if self._is_prayer_virtue_question(question):
            allowed_prayer_words = (
                prayer_words
                | {
                    "give", "me", "one", "a", "an",
                    "hadith", "hadees", "hadeeth",
                    "hadis", "about", "on",
                    "regarding", "the", "please",
                    "tell", "share", "of",
                    "importance", "virtue", "virtues",
                    "reward", "rewards", "benefit",
                    "benefits",
                }
            )

            if words <= allowed_prayer_words:
                return self._select_prayer_reward_hadith(
                    documents
                )

            return None

        fasting_words = {
            "fast", "fasting", "roza",
            "rozay", "sawm", "saum"
        }

        if not words & fasting_words:
            return None

        allowed_fasting_words = (
            fasting_words
            | {
                "give", "me", "one", "a", "an",
                "hadith", "hadees", "hadeeth",
                "hadis", "about", "on",
                "regarding", "the", "please",
                "tell", "share", "of",
                "importance", "virtue", "virtues",
                "reward", "rewards", "benefit",
                "benefits",
            }
        )

        if not words <= allowed_fasting_words:
            return None

        phrases = (
            "fasting is a shield",
            "fasting is a screen",
            "the fast is for me",
            "fasting which is for me",
        )

        for document in documents:
            content = document.get("content", "")
            normalized = self._normalize_text(content)
            metadata = document.get("metadata") or {}

            if len(self._words(content)) < 35:
                continue

            if not all(
                metadata.get(key) is not None
                for key in (
                    "source_name", "volume",
                    "book", "hadith_number"
                )
            ):
                continue

            if any(
                phrase in normalized
                for phrase in phrases
            ):
                return document

        return None

    @staticmethod
    def _build_citation(document: dict) -> str:
        metadata = document.get("metadata") or {}

        source = metadata.get(
            "source_name", "Unknown source"
        )

        parts = [str(source)]

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

        follow_up_words = {
            "it", "its", "this", "that",
            "these", "those", "they",
            "them", "their", "he", "she",
            "his", "her", "ones",
        }

        follow_up_starts = (
            "what about", "how about",
            "and ", "but ", "why ",
        )

        needs_context = (
            bool(words & follow_up_words)
            or question.lower().startswith(
                follow_up_starts
            )
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
                    "Rewrite the latest question as "
                    "a standalone Islamic search query. "
                    "Use conversation history only to "
                    "resolve references. Return only "
                    "the rewritten query."
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

        result = await ollama_client.generate(messages)

        return result.strip() or question

    async def generate(
        self,
        question: str,
        conversation_history: list[dict[str, str]],
    ) -> str:

        pipeline_start = time.perf_counter()

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
                "I couldn't find sufficiently relevant "
                "Islamic source passages."
            )

        # Extractive fast path for supported Hadith requests.
        if self._is_hadith_request(question):
            selected = self._select_fast_hadith(
                question,
                documents,
            )

            if selected is not None:
                print(
                    "[RAG] Fast extractive Hadith path",
                    flush=True,
                )

                print(
                    f"[TIMING] RAG pipeline total: "
                    f"{time.perf_counter() - pipeline_start:.2f}s",
                    flush=True,
                )

                return (
                    f"{selected['content'].strip()}\n\n"
                    f"{self._build_citation(selected)}"
                )

        # General questions: one LLM call.
        # Keep context compact.
        source_context = "\n\n".join(
            f"PASSAGE {index}\n"
            f"{document.get('content', '')}"
            for index, document in enumerate(documents)
        )

        system_prompt = f"""
You are an Islamic knowledge assistant.

Answer ONLY using the source passages below.

Rules:
1. Never invent facts, Hadith text, or references.
2. Use only information directly supported by
   the supplied passages.
3. If the passages do not support an answer,
   return an empty answer and empty evidence.
4. Keep the answer concise.
5. Evidence quotes must be copied EXACTLY.
6. For each quote, copy a SHORT continuous
   substring of 6 to 15 words from one passage.
7. Do not change punctuation, spelling,
   capitalization, or quotation marks.
8. Never paraphrase inside the evidence quote.
9. Return valid JSON only.

Output:
{{
  "answer": "Concise supported answer",
  "evidence": [
    {{
      "source_index": 0,
      "quote": "Exact continuous quote from passage"
    }}
  ]
}}

For Hadith requests, select a directly relevant
passage. The application will return its original text.

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

        print("\n===== RAG DEBUG =====", flush=True)
        print("Question:", question, flush=True)

        for index, document in enumerate(documents):
            print(f"PASSAGE {index}", flush=True)
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

        valid_indexes = self._validated_source_indexes(
            evidence,
            documents,
        )

        if not valid_indexes:
            return (
                "I couldn't verify supporting evidence "
                "for this answer."
            )

        if self._is_hadith_request(question):
            selected_index = valid_indexes[0]

            answer = documents[selected_index].get(
                "content", ""
            ).strip()

            valid_indexes = [selected_index]

        if not answer:
            return "No verified source text is available."

        citations = [
            self._build_citation(documents[index])
            for index in valid_indexes
        ]

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
