from __future__ import annotations

import sqlite3
import string
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brain.language import LanguageEngine

KEYWORD_STOPWORDS = {
    "quien", "quienes", "sobre", "porque", "puedes", "puede", "tienes",
    "tiene", "donde", "cuando", "como", "cual", "cuales", "eres", "estas",
    "sabes", "sabe", "conoces", "conoce", "dime", "cuentame", "explicame",
    "recuerdas", "podrias",
    "that", "what", "this", "with", "have", "does",
}


class MemoryRetrieval:
    """Finds the most relevant past conversation instead of generating text."""

    def __init__(self, db_path: str | Path, language_engine: "LanguageEngine") -> None:
        self.db_path = Path(db_path)
        self.language = language_engine

    def _extract_keywords(self, text: str) -> list[str]:
        words = (word.strip(string.punctuation) for word in str(text or "").lower().split())
        return [word for word in words if len(word) >= 4 and word not in KEYWORD_STOPWORDS]

    def retrieve(self, human_message: str, limit: int = 5) -> str | None:
        try:
            connection = sqlite3.connect(self.db_path)
            try:
                keywords = self._extract_keywords(human_message)
                if not keywords:
                    return None

                conditions = " OR ".join(["question LIKE ? OR answer LIKE ?"] * len(keywords))
                params: list[str] = []
                for keyword in keywords:
                    pattern = f"%{keyword}%"
                    params.extend([pattern, pattern])

                keyword_rows = connection.execute(
                    f"""
                    SELECT question, answer FROM conversations
                    WHERE source IN ('human_taught', 'taught_inferred')
                    AND ({conditions})
                    ORDER BY timestamp DESC LIMIT 100
                    """,
                    params,
                ).fetchall()
            finally:
                connection.close()

            keyword_set = set(keywords)
            best_keyword_answer: str | None = None
            best_overlap = 0

            for question, answer in keyword_rows:
                overlap = len(keyword_set & set(self._extract_keywords(question)))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_keyword_answer = answer

            # 2+ query keywords need 2 overlapping hits; with exactly
            # one keyword, that keyword must exactly match one of the
            # stored question's own keywords (overlap >= 1).
            required_overlap = 2 if len(keyword_set) >= 2 else 1
            if best_keyword_answer is not None and best_overlap >= required_overlap:
                return best_keyword_answer

            return None
        except Exception:
            return None
