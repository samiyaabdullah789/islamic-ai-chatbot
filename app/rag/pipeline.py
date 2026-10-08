
import json
import time

from app.rag.retriever import rag_retriever
from app.rag.reranker import rag_reranker
from app.llm.client import ollama_client


class RAGPipeline:

    async def _build_retrieval_query(
        self,
        question: str,
        conversation_history: list[dict[str, str]],
    ) -> str:

        # If there is no history, use the original question
        if not conversation_history:
            return question

        # Use only recent conversation context
        recent_history = conversation_history[-4:]

        history_text = "\n".join(
            f"{message['role']}: {message['content']}"
            for message in recent_history
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "Your task is to convert the CURRENT QUESTION into a clear, "
                    "standalone search query for retrieving relevant Islamic "
                    "source passages.\n\n"

                    "Use the CONVERSATION HISTORY to understand what the user "
                    "is referring to in follow-up questions.\n\n"

                    "If the current question contains references such as "
                    "'it', 'this', 'that', 'these', 'those', 'they', 'them', "
                    "'ones', or similar wording, replace those references with "
                    "the actual subject from the conversation history.\n\n"

                    "The standalone search query must preserve the meaning of "
                    "the user's current question while including enough context "
                    "for semantic retrieval.\n\n"

                    "Example:\n"
                    "Conversation history:\n"
                    "user: What are the five pillars of Islam?\n"
                    "assistant: The five pillars of Islam are...\n\n"
                    "Current question:\n"
                    "Which one of these is related to Ramadan?\n\n"
                    "Standalone search query:\n"
                    "Which of the five pillars of Islam is related to Ramadan?\n\n"

                    "Do not answer the question.\n"
                    "Do not add facts that are not present in the conversation.\n"
                    "Do not use outside Islamic knowledge.\n"
                    "If the current question is already clear and standalone, "
                    "return it unchanged.\n\n"

                    "Return ONLY the standalone search query."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"CONVERSATION HISTORY:\n{history_text}\n\n"
                    f"CURRENT QUESTION:\n{question}\n\n"
                    "STANDALONE SEARCH QUERY:"
                ),
            },
        ]

        retrieval_query = await ollama_client.generate(messages)
        retrieval_query = retrieval_query.strip()

        if not retrieval_query:
            return question

        return retrieval_query

    def _build_citation(self, document: dict) -> str:

        metadata = document["metadata"]

        source_name = metadata.get("source_name", "Unknown")
        volume = metadata.get("volume", "Unknown")
        book = metadata.get("book", "Unknown")
        hadith_number = metadata.get("hadith_number", "Unknown")

        return (
            f"[{source_name}, Volume {volume}, "
            f"Book {book}, Hadith {hadith_number}]"
        )

    async def generate(
        self,
        question: str,
        conversation_history: list[dict[str, str]],
    ) -> str:

        pipeline_start = time.perf_counter()

        # -------------------------------------------------
        # 1. Build context-aware retrieval query
        # -------------------------------------------------

        start = time.perf_counter()

        retrieval_query = await self._build_retrieval_query(
            question=question,
            conversation_history=conversation_history,
        )

        print(
            f"[TIMING] Query rewriting: {time.perf_counter() - start:.2f}s",
            flush=True,
        )

        # -------------------------------------------------
        # 2. Retrieve relevant passages from ChromaDB
        # -------------------------------------------------

        start = time.perf_counter()

        documents = await rag_retriever.retrieve(
            retrieval_query,
            limit=10,
        )

        print(
            f"[TIMING] ChromaDB retrieval: {time.perf_counter() - start:.2f}s",
            flush=True,
        )

        if not documents:
            return (
                "I don't have verified Islamic sources available "
                "to answer this question yet."
            )

        # -------------------------------------------------
        # 3. Rerank and keep the most relevant passages
        # -------------------------------------------------

        start = time.perf_counter()

        documents = await rag_reranker.rerank(
            question=retrieval_query,
            documents=documents,
            limit=5,
        )

        print(
            f"[TIMING] Reranking: {time.perf_counter() - start:.2f}s",
            flush=True,
        )

        if not documents:
            return (
                "I don't have enough relevant verified Islamic sources "
                "to answer this question."
            )

        # -------------------------------------------------
        # 4. Prepare retrieved source passages
        # -------------------------------------------------

        start = time.perf_counter()

        source_parts = []

        for index, document in enumerate(documents):
            source_parts.append(
                f"""PASSAGE {index}

{document["content"]}"""
            )

        source_context = "\n\n".join(source_parts)

        print(
            f"[TIMING] Context preparation: {time.perf_counter() - start:.2f}s",
            flush=True,
        )

        # -------------------------------------------------
        # 5. Generate answer from retrieved knowledge
        # -------------------------------------------------

        system_prompt = f"""
You are an Islamic knowledge assistant.

Your job is to answer the user's question using the trusted Islamic
SOURCE PASSAGES retrieved for that question.

The SOURCE PASSAGES are the knowledge source.

Your role is to understand them and generate a clear, natural answer
to the user's specific question.

RULES:

1. Answer exactly what the user asked.

2. Islamic factual information in your answer must come from the
   SOURCE PASSAGES.

3. You may summarize, combine, simplify, and explain information from
   the passages in natural language.

4. You do not need to copy the wording of the passages.

5. Do not introduce Islamic facts, names, rulings, numbers, events,
   explanations, or details from your own pretrained knowledge when
   they are not supported by the SOURCE PASSAGES.

6. Conversation history may be used to understand what the user means,
   especially in follow-up questions, but conversation history is not
   a trusted Islamic source.

7. Keep the answer focused on the user's question.
   Do not include background information, related events, examples,
   historical details, or additional explanations unless they are
   necessary to answer the question.

8. For a simple factual or yes/no question, give a short and direct
   answer unless more explanation is necessary.

9. If multiple passages contain the same supporting information,
   you do not need to use all of them.

10. Use the minimum number of passages needed to fully support the
    answer. Prefer the passage or passages that most directly support
    the answer.

11. Do not select passages merely because they are related to the
    topic. Select them only when they directly support information
    actually included in your answer.

12. If the retrieved passages do not contain enough information to
    answer the question, clearly say that the available sources do not
    provide enough information.

13. Do not create citations, Hadith numbers, source names, or references.
    The application will handle citations separately.

14. In "used_sources", include only the passage numbers that directly
    support the final answer.

Return ONLY valid JSON in exactly this structure:

{{
    "answer": "A clear natural answer based on the retrieved passages.",
    "used_sources": [0]
}}

Do not use markdown.

Do not include any text before or after the JSON.

SOURCE PASSAGES:

{source_context}
"""

        messages = [
            {"role": "system", "content": system_prompt},
            *conversation_history,
            {"role": "user", "content": question},
        ]

        start = time.perf_counter()

        # JSON mode enabled only for final answer
        raw_response = await ollama_client.generate(
            messages,
            json_mode=True,
        )

        print(
            f"[TIMING] Final LLM generation: {time.perf_counter() - start:.2f}s",
            flush=True,
        )

        # -------------------------------------------------
        # TEMPORARY DEBUG
        # -------------------------------------------------

        print("\n===== RAG DEBUG =====")
        print("Original question:", question)
        print("Retrieval query:", retrieval_query)

        print("\nRetrieved / reranked passages:")

        for index, document in enumerate(documents):
            print(f"\nPASSAGE {index}")
            print("Content:", document["content"])
            print("Metadata:", document["metadata"])

        print("\nLLM raw response:")
        print(raw_response)
        print("=====================\n")

        # -------------------------------------------------
        # 6. Parse structured response
        # -------------------------------------------------

        start = time.perf_counter()

        try:
            cleaned_response = raw_response.strip()

            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response.strip("`")

                if cleaned_response.startswith("json"):
                    cleaned_response = cleaned_response[4:].strip()

            result = json.loads(cleaned_response)

            answer = str(
                result.get("answer", "")
            ).strip()

            used_sources = result.get(
                "used_sources",
                [],
            )

        except (json.JSONDecodeError, TypeError, ValueError):
            return (
                "I couldn't generate a reliable source-based answer "
                "for this question."
            )

        print(
            f"[TIMING] JSON parsing: {time.perf_counter() - start:.2f}s",
            flush=True,
        )

        if not answer:
            return (
                "The available sources do not provide enough "
                "information to answer this question."
            )

        # -------------------------------------------------
        # 7. Validate sources selected by the LLM
        # -------------------------------------------------

        valid_source_indexes = []

        if isinstance(used_sources, list):

            for source_index in used_sources:

                if (
                    isinstance(source_index, int)
                    and 0 <= source_index < len(documents)
                    and source_index not in valid_source_indexes
                ):
                    valid_source_indexes.append(source_index)

        if not valid_source_indexes:
            return (
                "I don't have enough verified source support "
                "to answer this question."
            )

        # -------------------------------------------------
        # 8. Build citations from actual stored metadata
        # -------------------------------------------------

        citations = [
            self._build_citation(
                documents[source_index]
            )
            for source_index in valid_source_indexes
        ]

        citation_text = "\n".join(citations)

        # -------------------------------------------------
        # 9. Final response
        # -------------------------------------------------

        print(
            f"[TIMING] RAG pipeline total: {time.perf_counter() - pipeline_start:.2f}s",
            flush=True,
        )

        return f"{answer}\n\n{citation_text}"


rag_pipeline = RAGPipeline()
