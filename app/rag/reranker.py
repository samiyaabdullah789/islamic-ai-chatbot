import re


class RAGReranker:

    SYNONYMS = {
        "namaz": {"namaz", "salah", "salat", "prayer", "pray"},
        "salah": {"namaz", "salah", "salat", "prayer", "pray"},
        "prayer": {"namaz", "salah", "salat", "prayer", "pray"},
        "zakat": {"zakat", "zakah", "charity", "alms"},
        "fasting": {"fasting", "fast", "sawm", "ramadan"},
    }

    STOP_WORDS = {
        "give", "me", "one", "about", "the", "a", "an",
        "of", "in", "is", "are", "what", "how", "tell",
        "hadith", "please", "some",
    }

    def tokenize(self, text: str) -> set[str]:
        return set(re.findall(r"\b\w+\b", text.lower()))

    def expand_query(self, question: str) -> set[str]:
        words = self.tokenize(question) - self.STOP_WORDS
        expanded = set(words)

        for word in words:
            expanded.update(self.SYNONYMS.get(word, set()))

        return expanded

    async def rerank(
        self,
        question: str,
        documents: list[dict],
        limit: int = 5,
    ) -> list[dict]:

        if not documents:
            return []

        query_terms = self.expand_query(question)

        def relevance_score(document: dict) -> float:
            content = document.get("content", "")
            document_terms = self.tokenize(content)

            keyword_score = len(
                query_terms.intersection(document_terms)
            )

            distance = document.get("distance")
            semantic_score = (
                1 / (1 + max(float(distance), 0))
                if isinstance(distance, (int, float))
                else 0
            )

            return keyword_score + semantic_score

        ranked_documents = sorted(
            documents,
            key=relevance_score,
            reverse=True,
        )

        return ranked_documents[:limit]


rag_reranker = RAGReranker()