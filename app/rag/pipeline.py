
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

    async def _build_retrieval_query(
        self,
        question: str,
        conversation_history: list[dict[str, str]],
    ) -> str:

        question = question.strip()

        if not conversation_history:
            return question

        follow_up_words = {
            "it", "its", "this", "that", "these",
            "those", "they", "them", "their",
            "he", "she", "his", "her", "ones",
        }

        words = set(re.findall(r"\b\w+\b", question.lower()))

        follow_up_starts = (
            "what about",
            "how about",
            "and ",
            "but ",
            "why ",
        )

        needs_context = (
            bool(words.intersection(follow_up_words))
            or question.lower().startswith(follow_up_starts)
        )

        if not needs_context:
            return question

        recent_history = conversation_history[-4:]

        history_text = "\n".join(
            f"{message['role']}: {message['content']}"
            for message in recent_history
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "Rewrite the current Islamic question as a "
                    "standalone search query using conversation "
                    "history only when necessary. "
                    "Preserve the original meaning. "
                    "Do not answer the question. "
                    "Do not invent facts. "
                    "Return only the search query."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Conversation history:\n{history_text}\n\n"
                    f"Current question:\n{question}"
                ),
            },
        ]

        retrieval_query = await ollama_client.generate(messages)

        return retrieval_query.strip() or question

    def _build_citation(self, document: dict) -> str:

        metadata = document.get("metadata") or {}

        source_name = metadata.get("source_name", "Unknown")
        volume = metadata.get("volume", "Unknown")
        book = metadata.get("book", "Unknown")
        hadith_number = metadata.get("hadith_number", "Unknown")

        return (
            f"[{source_name}, Volume {volume}, "
            f"Book {book}, Hadith {hadith_number}]"
        )

    def _validate_evidence(
        self,
        evidence: object,
        documents: list[dict],
    ) -> list[int]:

        if not isinstance(evidence, list):
            return []

        valid_indexes = []

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

            # Reject empty or extremely short evidence.
            if len(normalized_quote.split()) < 6:
                continue

            source_text = self._normalize_text(
                documents[index].get("content", "")
            )

            if normalized_quote not in source_text:
                continue

            if index not in valid_indexes:
                valid_indexes.append(index)

        return valid_indexes

    async def generate(
        self,
        question: str,
        conversation_history: list[dict[str, str]],
    ) -> str:

        pipeline_start = time.perf_counter()

        # 1. Build retrieval query

        start = time.perf_counter()

        retrieval_query = await self._build_retrieval_query(
            question=question,
            conversation_history=conversation_history,
        )

        print(
            f"[TIMING] Query rewriting: "
            f"{time.perf_counter() - start:.2f}s",
            flush=True,
        )

        # 2. Retrieve source passages

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
                "I don't have verified Islamic sources "
                "available to answer this question yet."
            )

        # 3. Rerank retrieved passages

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
                "I don't have enough relevant Islamic "
                "source passages to answer this question."
            )

        # 4. Prepare source context

        start = time.perf_counter()

        source_parts = []

        for index, document in enumerate(documents):
            source_parts.append(
                f"PASSAGE {index}\n"
                f"{document.get('content', '')}"
            )

        source_context = "\n\n".join(source_parts)

        print(
            f"[TIMING] Context preparation: "
            f"{time.perf_counter() - start:.2f}s",
            flush=True,
        )

        # 5. Generate source-grounded answer

        system_prompt = f"""
You are an Islamic knowledge assistant.

Answer the user's question using ONLY the SOURCE PASSAGES.

IMPORTANT RULES:

1. Do not use pretrained Islamic knowledge to add facts.
2. Never invent Hadith wording, narrators, or references.
3. Answer the exact question, not merely a related topic.
4. Conversation history is for understanding follow-up
   questions, not for establishing Islamic facts.
5. If the passages do not directly support an answer,
   return an empty answer and empty evidence list.
6. Keep answers concise and natural.
7. For every factual claim, identify supporting evidence.
8. Every evidence quote must be copied exactly from
   the corresponding source passage.
9. Do not cite a passage merely because it discusses
   the same general topic.
10. Do not invent supporting quotes.
11. If a question asks for one Hadith, provide only one.
12. Do not include citations inside the answer text.

Return ONLY valid JSON:

{{
    "answer": "Source-supported answer",
    "evidence": [
        {{
            "source_index": 0,
            "quote": "Exact supporting text from the passage"
        }}
    ]
}}

If no passage directly supports the answer, return:

{{
    "answer": "",
    "evidence": []
}}

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

        # Temporary debug logs
        print("\n===== RAG DEBUG =====", flush=True)
        print("Original question:", question, flush=True)
        print("Retrieval query:", retrieval_query, flush=True)

        for index, document in enumerate(documents):
            print(f"\nPASSAGE {index}", flush=True)
            print("Content:", document.get("content"), flush=True)
            print("Metadata:", document.get("metadata"), flush=True)

        print("\nLLM raw response:", raw_response, flush=True)
        print("=====================\n", flush=True)

        # 6. Parse structured response

        start = time.perf_counter()

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
                raise ValueError("Expected JSON object")

            answer = result.get("answer", "")
            evidence = result.get("evidence", [])

            if not isinstance(answer, str):
                raise ValueError("Invalid answer")

            answer = answer.strip()

        except (json.JSONDecodeError, TypeError, ValueError):
            return (
                "I couldn't generate a reliable "
                "source-based answer for this question."
            )

        print(
            f"[TIMING] JSON parsing: "
            f"{time.perf_counter() - start:.2f}s",
            flush=True,
        )

        if not answer:
            return (
                "The retrieved Islamic sources do not "
                "provide enough information to answer "
                "this question."
            )

        # 7. Validate exact supporting quotes

        valid_source_indexes = self._validate_evidence(
            evidence=evidence,
            documents=documents,
        )

        if not valid_source_indexes:
            return (
                "I couldn't verify the supporting "
                "Hadith evidence for this answer."
            )

        # 8. Build citations from actual source metadata

        citations = [
            self._build_citation(documents[index])
            for index in valid_source_indexes
        ]

        citation_text = "\n".join(citations)

        print(
            f"[TIMING] RAG pipeline total: "
            f"{time.perf_counter() - pipeline_start:.2f}s",
            flush=True,
        )

        return f"{answer}\n\n{citation_text}"


rag_pipeline = RAGPipeline()
