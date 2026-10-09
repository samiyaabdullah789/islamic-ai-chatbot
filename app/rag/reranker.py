
import re


class RAGReranker:

    SYNONYMS = {
        "prayer": {
            "namaz", "salah", "salat", "prayer",
            "prayers", "pray", "praying",
        },
        "zakat": {
            "zakat", "zakah", "alms",
        },
        "fasting": {
            "fast", "fasts", "fasting", "sawm",
            "siyam",
        },
    }

    STOP_WORDS = {
        "give", "me", "one", "about", "the", "a",
        "an", "of", "in", "is", "are", "what",
        "how", "tell", "hadith", "please",
        "some", "show", "can", "you",
    }

    def tokenize(self, text: str) -> set[str]:
        return set(
            re.findall(r"\b\w+\b", text.lower())
        )

    def get_concepts(self, question: str) -> list[set[str]]:

        words = self.tokenize(question) - self.STOP_WORDS
        concepts = []
        handled = set()

        for word in words:

            if word in handled:
                continue

            synonym_group = next(
                (
                    group
                    for group in self.SYNONYMS.values()
                    if word in group
                ),
                None,
            )

            if synonym_group:
                concepts.append(synonym_group)
                handled.update(synonym_group)
            else:
                concepts.append({word})
                handled.add(word)

        return concepts

    async def rerank(
        self,
        question: str,
        documents: list[dict],
        limit: int = 5,
    ) -> list[dict]:

        if not documents:
            return []

        concepts = self.get_concepts(question)

        def relevance_score(document: dict) -> float:

            content = document.get("content", "")
            document_terms = self.tokenize(content)

            matched_concepts = sum(
                1
                for concept in concepts
                if concept.intersection(document_terms)
            )

            coverage = (
                matched_concepts / len(concepts)
                if concepts
                else 0.0
            )

            distance = document.get("distance")

            semantic_score = (
                1 / (1 + max(float(distance), 0))
                if isinstance(distance, (int, float))
                else 0.0
            )

            # Concept coverage is more important than
            # repeated keyword matches.
            return (coverage * 3.0) + semantic_score

        return sorted(
            documents,
            key=relevance_score,
            reverse=True,
        )[:limit]


rag_reranker = RAGReranker()
