
import json
import re
import time
import unicodedata

from app.rag.retriever import rag_retriever
from app.rag.reranker import rag_reranker
from app.llm.client import ollama_client


class RAGPipeline:

    @staticmethod
    def _normalize_text(text: str) -> str:
        if not isinstance(text, str):
            return ""

        text = unicodedata.normalize("NFKC", text)

        text = re.sub(
            r"(?<=\w)-[ \t]*\r?\n[ \t]*(?=\w)",
            "",
            text,
        )

        return " ".join(
            re.findall(
                r"[^\W_]+",
                text.casefold(),
                re.UNICODE,
            )
        )

    @staticmethod
    def _words(text: str) -> set[str]:
        return set(re.findall(r"\b\w+\b", text.casefold()))

    @staticmethod
    def _detect_language(question: str) -> str:
        if re.search(r"[\u0600-\u06FF]", question):
            return "urdu"

        words = set(re.findall(r"[a-z]+", question.casefold()))

        markers = {
            "kya", "kia", "kyun", "kion",
            "kaise", "kaisa", "kaisi",
            "kese", "kesy", "kis",
            "mein", "mai", "hain", "hai",
            "hota", "hoti", "hote",
            "chahiye", "chahye",
            "karna", "karni", "karo",
            "sath", "saath",
            "walidain", "waliden",
            "maa", "baap",
            "namaz", "roza",
            "gunah", "sawab",
            "batao", "btao",
            "mujhe", "mje",
            "apne", "apni",
            "unka", "unki",
            "unke", "hum",
            "aur", "se", "ke", "ko", "ki", "ka",
        }

        strong = {
            "kya", "kia", "kyun", "kion",
            "kaise", "kaisa", "kaisi",
            "kese", "kesy", "mein",
            "hain", "hota", "hoti",
            "chahiye", "chahye",
            "walidain", "waliden",
            "mujhe", "mje",
            "batao", "btao",
        }

        matches = words & markers

        if matches & strong or len(matches) >= 3:
            return "roman_urdu"

        return "english"

    @staticmethod
    def _is_hadith_request(question: str) -> bool:
        return bool(
            re.search(
                r"\b(?:show|give|quote|share|provide|tell|send)"
                r"\b.{0,55}\b(?:hadith|hadees|hadeeth|hadis)\b"
                r"|\b(?:hadith|hadees|hadeeth|hadis)"
                r"\b.{0,45}\b(?:text|number|reference)\b"
                r"|(?:حدیث|حدیثیں).{0,30}"
                r"(?:سنائیں|دکھائیں|بتائیں|متن)",
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

    @staticmethod
    def _parse_json(raw_response: str) -> dict:
        cleaned = raw_response.strip()

        if cleaned.startswith("```"):
            cleaned = re.sub(
                r"^```(?:json)?\s*|\s*```$",
                "",
                cleaned,
                flags=re.IGNORECASE,
            ).strip()

        parsed = json.loads(cleaned)

        if not isinstance(parsed, dict):
            raise ValueError("Expected a JSON object")

        return parsed

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

            if len(normalized_quote.split()) < 5:
                continue

            if normalized_quote not in normalized_source:
                print(
                    "[RAG EVIDENCE] Quote mismatch "
                    f"or too short: source_index={index}",
                    flush=True,
                )
                continue

            if index not in valid:
                valid.append(index)

        return valid

    async def _build_retrieval_query(
        self,
        question: str,
        conversation_history: list[dict[str, str]],
        language: str,
    ) -> str:

        question = question.strip()
        words = self._words(question)

        references = {
            "it", "its", "this", "that",
            "these", "those", "they", "them",
            "their", "he", "she", "his",
            "her", "ones",
            "iska", "iski", "iske",
            "uska", "uski", "uske",
            "unke", "unki", "woh", "wo",
            "yeh", "ye",
        }

        starts = (
            "what about",
            "how about",
            "and ",
            "but ",
            "why ",
            "aur ",
            "lekin ",
        )

        needs_context = bool(
            conversation_history
            and (
                bool(words & references)
                or question.casefold().startswith(starts)
            )
        )

        # Fast path: no additional Ollama call
        # for standalone English questions.
        if language == "english" and not needs_context:
            return question

        if needs_context:
            history_text = "\n".join(
                f"{item.get('role', 'user')}: "
                f"{item.get('content', '')[:300]}"
                for item in conversation_history[-4:]
            )
        else:
            history_text = "None"

        messages = [
            {
                "role": "system",
                "content": (
                    "Convert the user's question into ONE "
                    "short English search query for an "
                    "English Islamic source database. "
                    "Use conversation context only to "
                    "resolve unclear references. "
                    "Preserve meaning and Islamic terminology. "
                    "Do not answer or add facts. "
                    "Output only the English query."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Context: {history_text}\n"
                    f"Question: {question}"
                ),
            },
        ]

        start = time.perf_counter()

        rewritten = await ollama_client.generate(
            messages,
            task="translation",
        )

        print(
            "[TIMING] English query preparation: "
            f"{time.perf_counter() - start:.2f}s",
            flush=True,
        )

        rewritten = rewritten.strip().strip('"').strip()

        if (
            not rewritten
            or len(rewritten) > 350
            or "\n" in rewritten
        ):
            return question

        return rewritten

    @staticmethod
    def _fallback(reason: str) -> str:
        messages = {
            "missing": (
                "I couldn't find sufficient evidence "
                "in the available sources."
            ),
            "invalid": (
                "I couldn't verify a reliable "
                "source-based answer."
            ),
        }

        return messages[reason]

    async def generate(
        self,
        question: str,
        conversation_history: list[dict[str, str]],
    ) -> str:

        pipeline_start = time.perf_counter()

        language = self._detect_language(question)

        print(
            f"[LANGUAGE] Detected: {language}",
            flush=True,
        )

        retrieval_query = await self._build_retrieval_query(
            question,
            conversation_history,
            language,
        )

        print(
            f"[RAG] Retrieval query: {retrieval_query}",
            flush=True,
        )

        start = time.perf_counter()

        documents = await rag_retriever.retrieve(
            retrieval_query,
            limit=10,
        )

        print(
            "[TIMING] ChromaDB retrieval: "
            f"{time.perf_counter() - start:.2f}s",
            flush=True,
        )

        if not documents:
            return self._fallback("missing")

        start = time.perf_counter()

        documents = await rag_reranker.rerank(
            retrieval_query,
            documents,
            limit=5,
        )

        print(
            "[TIMING] Reranking: "
            f"{time.perf_counter() - start:.2f}s",
            flush=True,
        )

        if not documents:
            return self._fallback("missing")

        hadith_request = self._is_hadith_request(question)

        documents = documents[:3]

        source_context = "\n\n".join(
            f"PASSAGE {index}:\n"
            f"{document.get('content', '')}"
            for index, document in enumerate(documents)
        )

        if hadith_request:
            task = (
                "Select the ONE passage containing "
                "the requested Hadith. "
                "Set answer to 'selected'."
            )
        else:
            task = (
                "Answer in clear, simple English using "
                "at most two short sentences. "
                "Use only facts directly supported "
                "by the passages. "
                "Do not invent Islamic rulings, advice, "
                "quotations, or explanations. "
                "If the passages do not directly answer "
                "the question, return an empty answer "
                "and empty evidence."
            )

        system_prompt = (
            "You answer Islamic questions using ONLY "
            "the supplied passages. "
            + task
            + " "
            "Provide at most TWO evidence items. "
            "Each quote must be an exact English excerpt "
            "of 5 to 15 words from its referenced passage. "
            "Quotes must directly support the answer. "
            "Never invent quotes. "
            "If uncertain, return empty answer "
            "and empty evidence. "
            'Return JSON only: '
            '{"answer":"...","evidence":'
            '[{"source_index":0,"quote":"exact excerpt"}]}. '
            "Do not put citations in the answer; "
            "citations are added automatically.\n\n"
            + source_context
        )

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": (
                    f"Question: {retrieval_query}"
                ),
            },
        ]

        start = time.perf_counter()

        raw_response = await ollama_client.generate(
            messages,
            json_mode=True,
            task="answer",
        )

        print(
            "[TIMING] English answer generation: "
            f"{time.perf_counter() - start:.2f}s",
            flush=True,
        )

        print(
            "\n===== RAG DEBUG =====",
            flush=True,
        )

        print(
            "Original question:",
            question,
            flush=True,
        )

        print(
            "Search query:",
            retrieval_query,
            flush=True,
        )

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
            "LLM raw English response:",
            raw_response,
            flush=True,
        )

        print(
            "=====================\n",
            flush=True,
        )

        try:
            result = self._parse_json(raw_response)

            english_answer = result.get("answer")

            if not isinstance(english_answer, str):
                raise ValueError("Invalid answer")

            english_answer = english_answer.strip()
            evidence = result.get("evidence", [])

        except (ValueError, TypeError):
            return self._fallback("invalid")

        if not english_answer:
            return self._fallback("missing")

        valid_indexes = self._validated_source_indexes(
            evidence,
            documents,
        )

        if not valid_indexes:
            return self._fallback("invalid")

        if hadith_request:
            english_answer = documents[
                valid_indexes[0]
            ].get("content", "").strip()

            valid_indexes = valid_indexes[:1]

        if not english_answer:
            return self._fallback("invalid")

        # English-only output.
        # No extra answer translation call.
        final_answer = english_answer

        print(
            "[RAG] Verified English answer:",
            english_answer,
            flush=True,
        )

        print(
            "[RAG] Final answer:",
            final_answer,
            flush=True,
        )

        citations = list(
            dict.fromkeys(
                self._build_citation(documents[index])
                for index in valid_indexes
            )
        )

        print(
            "[TIMING] RAG pipeline total: "
            f"{time.perf_counter() - pipeline_start:.2f}s",
            flush=True,
        )

        return (
            final_answer
            + "\n\n"
            + "\n".join(citations)
        )


rag_pipeline = RAGPipeline()
