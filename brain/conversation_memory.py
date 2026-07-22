import re
import sqlite3
import time
from pathlib import Path
from typing import Any


# Palabras funcionales que no tienen valor semántico para el matching
_STOPWORDS = {
    "what", "that", "this", "with", "have", "been",
    "will", "from", "they", "them", "their", "there", "here", "when",
    "where", "which", "would", "could", "should", "about", "into",
    "than", "then", "some", "your", "also", "just", "like", "make",
    "know", "feel", "think", "look", "come", "more", "very", "much",
    "happen", "does", "see",
    # español
    "hace", "para", "pero", "como", "esto", "esta", "este", "algo",
    "todo", "porque", "cuando", "donde", "tiene", "puedo", "puedes",
    "eres", "soy", "que", "quien", "cual", "cuanto", "cuanta",
}


class ConversationMemory:
    """SQLite-backed store for conversational exchanges with the user."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.connection = sqlite3.connect(self.database_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        columns = [
            row["name"]
            for row in self.connection.execute("PRAGMA table_info(conversations)")
        ]
        recreated = "answerTEXT" in columns and "answer" not in columns
        if recreated:
            self.connection.execute("DROP TABLE conversations")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'unknown',
                keywords TEXT NOT NULL,
                timestamp REAL NOT NULL
            )
            """
        )
        columns = [
            row["name"]
            for row in self.connection.execute("PRAGMA table_info(conversations)")
        ]
        if "source" not in columns:
            self.connection.execute(
                "ALTER TABLE conversations ADD COLUMN source TEXT NOT NULL DEFAULT 'unknown'"
            )
        self.connection.commit()
        if recreated:
            print("ConversationMemory: conversations table recreated with correct schema.")
        else:
            print("ConversationMemory: conversations table schema already correct.")

    def store(self, question: str, answer: str, source: str = "unknown") -> int:
        keywords = self._extract_keywords(question)
        cursor = self.connection.execute(
            """
            INSERT INTO conversations (question, answer, source, keywords, timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            (question, answer, source, ",".join(keywords), time.time()),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def recall(self, question: str, limit: int = 3) -> list[dict]:
        keywords = self._extract_keywords(question)
        if not keywords:
            return []
        clauses = " OR ".join("keywords LIKE ?" for _ in keywords)
        rows = self.connection.execute(
            f"""
            SELECT id, question, answer, source, keywords, timestamp
            FROM conversations
            WHERE {clauses}
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            [f"%{keyword}%" for keyword in keywords] + [int(limit)],
        )
        return [self._row_to_conversation(row) for row in rows]

    def has_answered(self, question: str) -> bool:
        """
        Devuelve True solo si la pregunta es semanticamente identica
        a una ya respondida. Requiere coincidencia del prefijo interrogativo
        Y al menos 2 keywords semanticas en comun (sin stopwords).
        """
        keywords = set(self._extract_keywords(question))
        if len(keywords) < 1:
            return False

        new_prefix = self._question_prefix(question)

        rows = self.connection.execute("SELECT question, keywords FROM conversations")
        for row in rows:
            stored_keywords = set(filter(None, row["keywords"].split(",")))
            overlap = keywords & stored_keywords
            stored_prefix = self._question_prefix(row["question"])
            question_words = {"happen", "does", "why", "what", "is", "are", "how"}
            semantic_overlap = overlap - question_words
            if stored_prefix == new_prefix and len(semantic_overlap) >= 1:
                return True

        return False

    def summarize(self) -> dict:
        total = self.connection.execute(
            "SELECT COUNT(*) FROM conversations"
        ).fetchone()[0]
        if total == 0:
            return {
                "total_conversations": 0,
                "last_question": "",
                "last_answer": "",
                "last_source": "",
            }

        last = self.connection.execute(
            """
            SELECT question, answer, source FROM conversations
            ORDER BY timestamp DESC LIMIT 1
            """
        ).fetchone()
        return {
            "total_conversations": total,
            "last_question": last["question"],
            "last_answer": last["answer"],
            "last_source": last["source"],
        }

    def _extract_keywords(self, text: str) -> list[str]:
        """Extrae keywords semanticas filtrando stopwords funcionales."""
        keywords = []
        seen = set()
        for word in text.lower().split():
            keyword = re.sub(r"^\W+|\W+$", "", word)
            if len(keyword) > 3 and keyword not in seen and keyword not in _STOPWORDS:
                keywords.append(keyword)
                seen.add(keyword)
        return keywords

    def _question_prefix(self, question: str) -> str:
        """Extrae el prefijo interrogativo para comparar estructura."""
        q = question.lower().strip()
        for prefix in ("why does", "why is", "what is", "what are", "is ", "are ", "how "):
            if q.startswith(prefix):
                return prefix.strip()
        return q.split()[0] if q.split() else ""

    def _row_to_conversation(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "question": row["question"],
            "answer": row["answer"],
            "source": row["source"],
            "keywords": row["keywords"],
            "timestamp": row["timestamp"],
        }
