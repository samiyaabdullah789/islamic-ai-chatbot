
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

    def _is_prayer_virtue_question(self, question: str) -> bool:
        words = self._words(question)
        return (
            bool(words & {
                "namaz", "salah", "salat", "prayer", "prayers"
            })
            and bool(words & {
                "importance", "virtue", "virtues",
                "reward", "rewards", "benefit", "benefits"
            })
        )

    def _select_prayer_reward_hadith(
        self,
        documents: list[dict],
    ) -> dict | None:
        """
        Conservative fast path for prayer reward questions.

        This is a topic-specific rule, not a general
        semantic relevance verifier.
        """
        for document in documents:
            content = document.get("content", "")
            normalized = self._normalize_text(content)
            words = self._words(content)

            has_prayer = bool(
                words & {"prayer", "prayers", "praying", "salah"}
            )
            has_reward = bool(
                words & {"reward", "rewards"}
            )

            # Avoid incomplete or cross-boundary passages.
            if len(words) < 35:
                continue

            if not has_prayer or not has_reward:
                continue

            # Prefer an explicit reward for prayer itself,
            # rather than a passage about a different topic.
            phrases = (
                "reward of the prayer",
                "reward of prayer",
                "reward of the prayers",
                "reward of congregational prayer",
                "reward of the noon prayer",
            )

            if any(phrase in normalized for phrase in phrases):
                return document

        return None

    def _build_citation(self, document: dict) -> str:
        metadata = document.get("metadata") or {}

        return (
            f"[{metadata.get('source_name', 'Unknown')}, "
            f"Volume {metadata.get('volume', 'Unknown')}, "
            f"Book {metadata.get('book', 'Unknown')}, "
            f"Hadith {metadata.get('hadith_number', 'Unknown')}]"
        )

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
            "it", "its", "this", "that", "these",
            "those", "they", "them", "their",
            "he", "she", "his", "her", "ones",
        }

        follow_up_starts = (
            "what about", "how about",
            "and ", "but ", "why ",
        )

        needs_context = (
            bool(words & follow_up_words)
            or question.lower().startswith(follow_up_starts)
        )

        if not needs_context:
            return question

        recent_history = conversation_history[-4:]

        history_text = "\n".join(
            f"{item['role']}: {item['content']}"
            for item in recent_history
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "Rewrite the question as a standalone "
                    "Islamic search query using conversation "
                    "history. Preserve meaning. "
                    "Return only the query, not an answer."
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

        # Fast path: a narrowly defined Hadith question.
        # No final LLM call is required here.
        if (
            self._is_hadith_request(question)
            and self._is_prayer_virtue_question(question)
        ):
            selected = self._select_prayer_reward_hadith(
                documents
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

        # Other questions use one LLM generation call.
        source_parts = [
            f"PASSAGE {index}\n{document.get('content', '')}"
            for index, document in enumerate(documents)
        ]

        source_context = "\n\n".join(source_parts)

        system_prompt = f"""
You are an Islamic knowledge assistant.

Use ONLY the SOURCE PASSAGES below.

Rules:
- Never invent Hadith wording or references.
- Do not use external or memorized facts.
- Select passages that directly answer the question.
- A matching keyword alone is not sufficient.
- If no passage answers the question, return an
  empty answer and an empty evidence list.
- Copy supporting quotes exactly from the passages.
- Return only valid JSON.
- Do not put citations in the answer.

JSON format:
{{
    "answer": "Answer supported by the passages",
    "evidence": [
        {{
            "source_index": 0,
            "quote": "Exact quote from passage"
        }}
    ]
}}

For a Hadith request, select one relevant passage.
The application will return the original passage text.

SOURCE PASSAGES:
{source_context}
"""

        messages = [
            {"role": "system", "content": system_prompt},
            *conversation_history[-4:],
            {"role": "user", "content": question},
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
        print("Retrieval query:", retrieval_query, flush=True)

        for index, document in enumerate(documents):
            print(f"PASSAGE {index}", flush=True)
            print("Content:", document.get("content"), flush=True)
            print("Metadata:", document.get("metadata"), flush=True)

        print("LLM raw response:", raw_response, flush=True)
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
            evidence = result.get("evidence", [])

            if not isinstance(answer, str):
                raise ValueError("Invalid answer")

            answer = answer.strip()

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
            # Use original source text, not generated wording.
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

        return f"{answer}\n\n" + "\n".join(citations)


rag_pipeline = RAGPipeline()
