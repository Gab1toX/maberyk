from __future__ import annotations

import atexit
import json
import math
import queue
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


class EpisodicMemory:
    """SQLite-backed store for concrete agent experiences."""

    MAX_CACHE_ENTRIES = 10_000
    WRITE_BATCH_SIZE = 32
    WRITE_FLUSH_INTERVAL = 0.25

    def __init__(self, database_path: str | Path = "episodic_memory.sqlite3") -> None:
        self.database_path = Path(database_path)
        self.connection = sqlite3.connect(self.database_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self._summary_cache: dict[str, Any] = {}
        self._episodes_by_recency: list[dict[str, Any]] = []
        self._rolling_surprise_sum = 0.0
        self._rolling_count = 0
        self._next_episode_id = 1
        self._write_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._writer_stop = threading.Event()
        self._closed = False
        self._create_tables()
        self._load_episode_cache()
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name="episodic-memory-writer",
            daemon=True,
        )
        self._writer_thread.start()
        atexit.register(self.close)

    def store(self, episode: dict[str, Any], thought: str = "") -> int:
        timestamp = episode.get("timestamp", time.time())
        episode_id = self._next_episode_id
        self._next_episode_id += 1
        cached_episode = {
            "id": episode_id,
            "observation": episode["observation"],
            "action": episode["action"],
            "outcome": episode["outcome"],
            "surprise_level": float(episode["surprise_level"]),
            "timestamp": float(timestamp),
            "thought": str(thought),
            "_observation_vector": self._to_vector(episode["observation"]),
        }
        self._episodes_by_recency.insert(0, cached_episode)
        self._rolling_surprise_sum += cached_episode["surprise_level"]
        self._rolling_count += 1
        if len(self._episodes_by_recency) > self.MAX_CACHE_ENTRIES:
            evicted = self._episodes_by_recency.pop()
            self._rolling_surprise_sum -= evicted["surprise_level"]
            self._rolling_count -= 1
        self._refresh_summary_cache()
        self._write_queue.put(cached_episode)
        return episode_id

    def recall_similar(
        self, observation: Any, limit: int = 5, max_candidates: int = 200
    ) -> list[dict[str, Any]]:
        target_vector = self._to_vector(observation)
        scored_episodes = []

        for cached_episode in self._episodes_by_recency[: int(max_candidates)]:
            episode = self._episode_for_public_use(cached_episode)
            episode_vector = cached_episode["_observation_vector"]
            similarity = self._similarity(target_vector, episode_vector)
            episode["similarity"] = similarity
            scored_episodes.append(episode)

        scored_episodes.sort(key=lambda episode: episode["similarity"], reverse=True)
        return scored_episodes[:limit]

    def recent_episodes(self, limit: int = 20) -> list[dict[str, Any]]:
        """Most recently stored episodes, newest first, straight from the
        in-memory cache (no SQLite read) -- unlike recall_by_surprise, which
        scans the full episodes table, this stays cheap enough to call on
        every request."""
        return [
            self._episode_for_public_use(episode)
            for episode in self._episodes_by_recency[: int(limit)]
        ]

    def recall_by_surprise(self, threshold: float) -> list[dict[str, Any]]:
        self.flush()
        rows = self.connection.execute(
            """
            SELECT * FROM episodes
            WHERE surprise_level >= ?
            ORDER BY surprise_level DESC, timestamp DESC
            """,
            (float(threshold),),
        )
        return [self._row_to_episode(row) for row in rows]

    def summarize(self) -> dict[str, Any]:
        return self._summary_cache

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.flush()
        self._writer_stop.set()
        self._writer_thread.join()
        self.connection.close()

    def flush(self) -> None:
        self._write_queue.join()

    def memory_feature_snapshot(self, observation: Any) -> dict[str, float]:
        features = {
            "memory.familiar": 0.0,
            "memory.strange": 1.0,
            "memory.surprising": 0.0,
            "memory.repeated": 0.0,
        }
        similar = self.recall_similar(observation, limit=3)
        if similar:
            best_similarity = similar[0].get("similarity", 0.0)
            features["memory.familiar"] = self._clamp(best_similarity)
            features["memory.strange"] = self._clamp(1.0 - best_similarity)

        average_surprise = self._average_surprise()
        total = self._rolling_count
        features["memory.surprising"] = self._clamp(average_surprise)
        features["memory.repeated"] = self._clamp(total / 100.0)
        return features

    def _create_tables(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                observation TEXT NOT NULL,
                action TEXT NOT NULL,
                outcome TEXT NOT NULL,
                surprise_level REAL NOT NULL,
                timestamp REAL NOT NULL
            )
            """
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_episodes_surprise ON episodes (surprise_level)"
        )
        existing_columns = {
            row["name"] for row in self.connection.execute("PRAGMA table_info(episodes)")
        }
        if "thought" not in existing_columns:
            self.connection.execute("ALTER TABLE episodes ADD COLUMN thought TEXT DEFAULT ''")
        self.connection.commit()

    def _load_episode_cache(self) -> None:
        rows = list(
            self.connection.execute(
                """
                SELECT * FROM episodes
                ORDER BY id DESC
                LIMIT ?
                """,
                (self.MAX_CACHE_ENTRIES,),
            )
        )
        self._episodes_by_recency = []
        self._rolling_surprise_sum = 0.0
        self._rolling_count = 0
        for row in rows:
            episode = self._row_to_episode(row)
            episode["_observation_vector"] = self._to_vector(episode["observation"])
            self._episodes_by_recency.append(episode)
            self._rolling_surprise_sum += float(episode["surprise_level"])
            self._rolling_count += 1

        db_count = int(self._single_value("SELECT COUNT(*) FROM episodes", 0))
        expected_cache_count = min(db_count, self.MAX_CACHE_ENTRIES)
        cache_count = len(self._episodes_by_recency)
        if expected_cache_count != cache_count:
            raise RuntimeError(
                "EpisodicMemory cache load mismatch: "
                f"database window expected {expected_cache_count} episodes "
                f"but cache has {cache_count}."
            )

        max_id = self._single_value("SELECT MAX(id) FROM episodes", 0)
        self._next_episode_id = int(max_id or 0) + 1
        self._refresh_summary_cache()
        print(
            "EpisodicMemory: episode cache loaded "
            f"({cache_count} rows, count verified)."
        )

    def _writer_loop(self) -> None:
        pending: list[dict[str, Any]] = []
        connection = sqlite3.connect(self.database_path)
        try:
            while not self._writer_stop.is_set() or not self._write_queue.empty():
                timed_out = False
                try:
                    pending.append(self._write_queue.get(timeout=self.WRITE_FLUSH_INTERVAL))
                except queue.Empty:
                    timed_out = True

                if pending and (
                    len(pending) >= self.WRITE_BATCH_SIZE
                    or timed_out
                    or self._writer_stop.is_set()
                ):
                    self._persist_batch(connection, pending)
                    for _ in pending:
                        self._write_queue.task_done()
                    pending = []

            if pending:
                self._persist_batch(connection, pending)
                for _ in pending:
                    self._write_queue.task_done()
        finally:
            connection.close()

    def _persist_batch(
        self, connection: sqlite3.Connection, episodes: list[dict[str, Any]]
    ) -> None:
        connection.executemany(
            """
            INSERT INTO episodes (observation, action, outcome, surprise_level, timestamp, thought)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    self._to_json(episode["observation"]),
                    self._to_json(episode["action"]),
                    self._to_json(episode["outcome"]),
                    float(episode["surprise_level"]),
                    float(episode["timestamp"]),
                    str(episode.get("thought", "")),
                )
                for episode in episodes
            ],
        )
        connection.commit()

    def _refresh_summary_cache(self) -> None:
        total = self._rolling_count
        average_surprise = self._average_surprise()
        most_common_actions = self._action_counts_from_memory()
        most_surprising_actions = self._surprise_by_action_from_memory()
        patterns = self._patterns_from_memory(
            total,
            average_surprise,
            most_common_actions,
            most_surprising_actions,
        )
        self._summary_cache = {
            "total_episodes": total,
            "average_surprise": average_surprise,
            "most_common_actions": most_common_actions,
            "most_surprising_actions": most_surprising_actions,
            "patterns": patterns,
        }

    def _average_surprise(self) -> float:
        if self._rolling_count == 0:
            return 0.0
        return self._rolling_surprise_sum / self._rolling_count

    def _row_to_episode(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "observation": json.loads(row["observation"]),
            "action": json.loads(row["action"]),
            "outcome": json.loads(row["outcome"]),
            "surprise_level": row["surprise_level"],
            "timestamp": row["timestamp"],
        }

    def _episode_for_public_use(self, episode: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": episode["id"],
            "observation": episode["observation"],
            "action": episode["action"],
            "outcome": episode["outcome"],
            "surprise_level": episode["surprise_level"],
            "timestamp": episode["timestamp"],
        }

    def _to_json(self, value: Any) -> str:
        return json.dumps(value, sort_keys=True)

    def _to_vector(self, value: Any) -> list[float]:
        vector: list[float] = []
        self._flatten_numbers(value, vector)
        return vector

    def _flatten_numbers(self, value: Any, vector: list[float]) -> None:
        if isinstance(value, bool):
            vector.append(float(value))
        elif isinstance(value, int | float):
            vector.append(float(value))
        elif isinstance(value, dict):
            for key in sorted(value):
                self._flatten_numbers(value[key], vector)
        elif isinstance(value, list | tuple):
            for item in value:
                self._flatten_numbers(item, vector)

    def _similarity(self, first: list[float], second: list[float]) -> float:
        if not first or not second:
            return 0.0

        size = min(len(first), len(second))
        distance = math.sqrt(
            sum((first[index] - second[index]) ** 2 for index in range(size))
        )
        length_penalty = abs(len(first) - len(second))
        return 1.0 / (1.0 + distance + length_penalty)

    def _single_value(self, query: str, default: Any) -> Any:
        value = self.connection.execute(query).fetchone()[0]
        return default if value is None else value

    def _action_counts(self) -> list[dict[str, Any]]:
        counts: dict[str, int] = {}
        for row in self.connection.execute("SELECT action FROM episodes"):
            action = row["action"]
            counts[action] = counts.get(action, 0) + 1

        return [
            {"action": json.loads(action), "count": count}
            for action, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)
        ]

    def _surprise_by_action(self) -> list[dict[str, Any]]:
        action_surprises: dict[str, list[float]] = {}
        for row in self.connection.execute("SELECT action, surprise_level FROM episodes ORDER BY id DESC LIMIT 500"):
            action_surprises.setdefault(row["action"], []).append(row["surprise_level"])

        summaries = []
        for action, surprises in action_surprises.items():
            summaries.append(
                {
                    "action": json.loads(action),
                    "average_surprise": sum(surprises) / len(surprises),
                    "count": len(surprises),
                }
            )

        summaries.sort(key=lambda summary: summary["average_surprise"], reverse=True)
        return summaries

    def _patterns(
        self,
        total: int,
        average_surprise: float,
        most_common_actions: list[dict[str, Any]],
        most_surprising_actions: list[dict[str, Any]],
    ) -> list[str]:
        patterns = [f"Stored {total} concrete episodes."]

        if most_common_actions:
            action = most_common_actions[0]["action"]
            count = most_common_actions[0]["count"]
            patterns.append(f"Most repeated action is {action}, seen {count} times.")

        if most_surprising_actions:
            action = most_surprising_actions[0]["action"]
            surprise = most_surprising_actions[0]["average_surprise"]
            patterns.append(f"Action {action} has the highest average surprise: {surprise:.6f}.")

        high_surprise_count = self._single_value(
            "SELECT COUNT(*) FROM episodes WHERE surprise_level > "
            "(SELECT AVG(surprise_level) FROM episodes)",
            0,
        )
        patterns.append(
            f"{high_surprise_count} episodes were above the average surprise level of {average_surprise:.6f}."
        )
        return patterns

    def _action_counts_from_memory(self) -> list[dict[str, Any]]:
        counts: dict[str, int] = {}
        for episode in self._episodes_by_recency[:500]:
            action = self._to_json(episode["action"])
            counts[action] = counts.get(action, 0) + 1

        return [
            {"action": json.loads(action), "count": count}
            for action, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)
        ]

    def _surprise_by_action_from_memory(self) -> list[dict[str, Any]]:
        action_surprises: dict[str, list[float]] = {}
        for episode in self._episodes_by_recency[:500]:
            action = self._to_json(episode["action"])
            action_surprises.setdefault(action, []).append(episode["surprise_level"])

        summaries = []
        for action, surprises in action_surprises.items():
            summaries.append(
                {
                    "action": json.loads(action),
                    "average_surprise": sum(surprises) / len(surprises),
                    "count": len(surprises),
                }
            )

        summaries.sort(key=lambda summary: summary["average_surprise"], reverse=True)
        return summaries

    def _patterns_from_memory(
        self,
        total: int,
        average_surprise: float,
        most_common_actions: list[dict[str, Any]],
        most_surprising_actions: list[dict[str, Any]],
    ) -> list[str]:
        patterns = [f"Stored {total} concrete episodes."]

        if most_common_actions:
            action = most_common_actions[0]["action"]
            count = most_common_actions[0]["count"]
            patterns.append(f"Most repeated action is {action}, seen {count} times.")

        if most_surprising_actions:
            action = most_surprising_actions[0]["action"]
            surprise = most_surprising_actions[0]["average_surprise"]
            patterns.append(f"Action {action} has the highest average surprise: {surprise:.6f}.")

        high_surprise_count = sum(
            1
            for episode in self._episodes_by_recency
            if episode["surprise_level"] > average_surprise
        )
        patterns.append(
            f"{high_surprise_count} episodes were above the average surprise level of {average_surprise:.6f}."
        )
        return patterns

    def _clamp(self, value: float) -> float:
        return max(0.0, min(float(value), 1.0))
