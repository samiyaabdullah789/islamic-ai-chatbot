import re


class RAGReranker:
    STOP_WORDS = {
        "give", "me", "one", "some", "about", "the", "a", "an", "of", "in",
        "is", "are", "what", "how", "does", "do", "tell", "hadith", "hadees",
        "hadeeth", "hadis", "please", "show", "can", "you", "islam", "islamic",
        "for", "and", "to", "on", "regarding", "teach", "teaches", "says",
    }
    SYNONYMS = (
        {"prayer", "prayers", "pray", "praying", "namaz", "salah", "salat"},
        {"fast", "fasting", "fasts", "sawm", "siyam", "roza", "rozay"},
        {"zakat", "zakah", "alms", "charity"},
        {"mother", "mothers", "mom"},
        {"father", "fathers", "dad"},
        {"parent", "parents", "mother", "father", "mothers", "fathers"},
        {"kindness", "kind", "kindly", "goodness", "respect", "respectful", "treat", "treatment"},
        {"honesty", "honest", "truth", "truthful", "truthfulness"},
    )

    @staticmethod
    def tokenize(text: str) -> set[str]:
        return set(re.findall(r"\b\w+\b", text.casefold()))

    def get_concepts(self, question: str) -> list[set[str]]:
        words = self.tokenize(question) - self.STOP_WORDS
        concepts = []
        for word in sorted(words):
            group = next((synonyms for synonyms in self.SYNONYMS if word in synonyms), None)
            concept = group if group else {word}
            if not any(concept == existing for existing in concepts):
                concepts.append(concept)
        return concepts

    async def rerank(self, question: str, documents: list[dict], limit: int = 5) -> list[dict]:
        concepts = self.get_concepts(question)

        def score(document: dict) -> float:
            tokens = self.tokenize(document.get("content", ""))
            coverage = sum(bool(tokens & concept) for concept in concepts) / max(len(concepts), 1)
            distance = document.get("distance")
            semantic = (
                1 / (1 + max(distance, 0))
                if isinstance(distance, (int, float)) else 0.0
            )
            return 2 * coverage + semantic

        return sorted(documents, key=score, reverse=True)[:limit]


rag_reranker = RAGReranker()