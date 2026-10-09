
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
            "fast", "fasts", "fasting",
            "sawm", "siyam", "roza", "rozay",
        },
    }

    STOP_WORDS = {
        "give", "me", "one", "about", "the",
        "a", "an", "of", "in", "is", "are",
        "what", "how", "tell", "hadith",
        "hadees", "hadeeth", "hadis",
        "please", "some", "show", "can",
        "you", "islam", "islamic", "for",
        "and", "to", "on", "regarding",
    }

    def tokenize(self, text: str) -> set[str]:
        return set(
            re.findall(r"\b\w+\b", text.lower())
        )

    def get_concepts(
        self,
        question: str,
    ) -> list[set[str]]:

        words = self.tokenize(question) - self.STOP_WORDS
        concepts = []
        handled = set()

        for word in sorted(words):
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
        question_words = self.tokenize(question)

        asks_definition = bool(
            question_words & {
                "what", "meaning", "definition", "define"
            }
        )

        definition_terms = {
            "obligatory", "compulsory", "duty",
            "charity", "poor", "needy", "pillar",
            "pillars", "prescribed", "enjoined",
        }

        def relevance_score(document: dict) -> float:
            content = document.get("content", "")
            document_terms = self.tokenize(content)

            matched = sum(
                bool(concept & document_terms)
                for concept in concepts
            )

            coverage = (
                matched / len(concepts)
                if concepts
                else 0.0
            )

            distance = document.get("distance")

            semantic_score = (
                1.0 / (1.0 + max(float(distance), 0.0))
                if isinstance(distance, (int, float))
                else 0.0
            )

            intent_bonus = 0.0

            if asks_definition and (
                question_words & {"zakat", "zakah"}
            ):
                matches = len(
                    document_terms & definition_terms
                )

                # Small bonus; does not guarantee
                # that the passage defines Zakat.
                intent_bonus = min(matches, 3) * 0.15

            return (
                coverage * 3.0
                + semantic_score
                + intent_bonus
            )

        return sorted(
            documents,
            key=relevance_score,
            reverse=True,
        )[:limit]


rag_reranker = RAGReranker()
