from __future__ import annotations

import sqlite3
import string
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brain.language import LanguageEngine


class MemoryRetrieval:
    """Finds the most relevant past conversation instead of generating text."""

    def __init__(self, db_path: str | Path, language_engine: "LanguageEngine") -> None:
        self.db_path = Path(db_path)
        self.language = language_engine

    def _text_to_vector(self, text: str) -> dict[str, float]:
        words = str(text or "").lower().split()
        vectors = [
            self.language.associations[word]
            for word in words
            if word in self.language.associations
        ]
        if not vectors:
            return {}

        totals: dict[str, float] = {}
        for vector in vectors:
            for feature, value in vector.items():
                totals[feature] = totals.get(feature, 0.0) + value

        count = len(vectors)
        return {feature: value / count for feature, value in totals.items()}

    def _extract_keywords(self, text: str) -> list[str]:
        words = (word.strip(string.punctuation) for word in str(text or "").lower().split())
        return [word for word in words if len(word) > 3]

    def _cosine_similarity(self, vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
        if not vec_a or not vec_b:
            return 0.0

        shared_features = set(vec_a) | set(vec_b)
        dot_product = sum(vec_a.get(feature, 0.0) * vec_b.get(feature, 0.0) for feature in shared_features)
        magnitude_a = sum(value * value for value in vec_a.values()) ** 0.5
        magnitude_b = sum(value * value for value in vec_b.values()) ** 0.5

        if magnitude_a == 0.0 or magnitude_b == 0.0:
            return 0.0

        return dot_product / (magnitude_a * magnitude_b)

    def retrieve(self, human_message: str, limit: int = 5) -> str | None:
        try:
            connection = sqlite3.connect(self.db_path)
            try:
                keywords = self._extract_keywords(human_message)
                if keywords:
                    conditions = " OR ".join(["question LIKE ? OR answer LIKE ?"] * len(keywords))
                    params: list[str] = []
                    for keyword in keywords:
                        pattern = f"%{keyword}%"
                        params.extend([pattern, pattern])

                    keyword_rows = connection.execute(
                        f"""
                        SELECT question, answer FROM conversations
                        WHERE source IN ('human_taught', 'agent_generated')
                        AND ({conditions})
                        ORDER BY timestamp DESC LIMIT 20
                        """,
                        params,
                    ).fetchall()

                    keyword_set = set(keywords)
                    best_keyword_answer: str | None = None
                    best_overlap = 0

                    for question, answer in keyword_rows:
                        overlap = len(keyword_set & set(self._extract_keywords(question)))
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_keyword_answer = answer

                    if best_keyword_answer is not None and best_overlap >= 1:
                        return best_keyword_answer

                query_vector = self._text_to_vector(human_message)
                rows = connection.execute(
                    """
                    SELECT question, answer FROM conversations
                    WHERE source IN ('human_taught', 'agent_generated')
                    ORDER BY timestamp DESC LIMIT 500
                    """
                ).fetchall()
            finally:
                connection.close()

            best_answer: str | None = None
            best_similarity = 0.0

            for question, answer in rows:
                candidate_vector = self._text_to_vector(f"{question} {answer}")
                similarity = self._cosine_similarity(query_vector, candidate_vector)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_answer = answer

            if best_answer is None or best_similarity < 0.15:
                return None

            return best_answer
        except Exception:
            return None
