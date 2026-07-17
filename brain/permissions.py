from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class PermissionLevel(Enum):
    ALLOWED = "allowed"
    DENIED = "denied"
    PENDING = "pending"


# Acciones que jamas pueden ser otorgadas, ni permanentemente ni por
# aprobacion puntual. Este conjunto es inmutable en tiempo de ejecucion:
# ninguna llamada publica de PermissionManager puede alterarlo.
PROHIBITED_ACTIONS: frozenset[str] = frozenset(
    {
        "file.delete_permanent",
        "system.modify_settings",
        "financial.any",
        "communication.send_external",
        "credentials.enter",
        "software.install",
        "system.shutdown",
        "system.restart",
    }
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class PermissionManager:
    """Gatekeeper for actions Maberyk wants to take outside its sandboxed world.

    Every decision is durable and auditable: permanent grants, pending
    requests awaiting a human answer, and a full audit trail all live in
    `permissions.sqlite3`, next to `agent_state.pt`.
    """

    def __init__(self, db_path: str | Path = "permissions.sqlite3") -> None:
        self.database_path = Path(db_path)
        self._lock = threading.Lock()
        self.connection = sqlite3.connect(self.database_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self._create_tables()
        self._permanent_grants: set[str] = set()
        self._load_permanent_grants()

    def _create_tables(self) -> None:
        with self._lock:
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS permission_grants (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action_name TEXT NOT NULL UNIQUE,
                    category TEXT NOT NULL,
                    granted_at TEXT NOT NULL
                )
                """
            )
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS pending_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action_name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    context TEXT NOT NULL DEFAULT '',
                    requested_at TEXT NOT NULL,
                    resolved_at TEXT,
                    approved INTEGER
                )
                """
            )
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action_name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    level TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    context TEXT NOT NULL DEFAULT '',
                    timestamp TEXT NOT NULL
                )
                """
            )
            self.connection.commit()

    def _load_permanent_grants(self) -> None:
        with self._lock:
            rows = self.connection.execute(
                "SELECT action_name FROM permission_grants"
            ).fetchall()
            self._permanent_grants = {row["action_name"] for row in rows}

    def is_prohibited(self, action_name: str) -> bool:
        return action_name in PROHIBITED_ACTIONS

    def is_granted(self, action_name: str) -> bool:
        return action_name in self._permanent_grants

    def request(
        self, action_name: str, category: str, context: str = ""
    ) -> tuple[PermissionLevel, str]:
        if self.is_prohibited(action_name):
            level = PermissionLevel.DENIED
            reason = f"'{action_name}' is a permanently prohibited action."
            self._audit(action_name, category, level, reason, context)
            return level, reason

        with self._lock:
            has_permanent_grant = action_name in self._permanent_grants

        if has_permanent_grant:
            level = PermissionLevel.ALLOWED
            reason = f"'{action_name}' has a permanent grant."
            self._audit(action_name, category, level, reason, context)
            return level, reason

        level = PermissionLevel.PENDING
        reason = f"'{action_name}' requires human approval."
        with self._lock:
            self.connection.execute(
                """
                INSERT INTO pending_requests
                    (action_name, category, context, requested_at, resolved_at, approved)
                VALUES (?, ?, ?, ?, NULL, NULL)
                """,
                (action_name, category, context, _utc_now_iso()),
            )
            self.connection.commit()
        self._audit(action_name, category, level, reason, context)
        return level, reason

    def grant_permanent(self, action_name: str, category: str) -> None:
        if self.is_prohibited(action_name):
            raise PermissionError(
                f"'{action_name}' is permanently prohibited and cannot be granted."
            )
        with self._lock:
            self.connection.execute(
                """
                INSERT INTO permission_grants (action_name, category, granted_at)
                VALUES (?, ?, ?)
                ON CONFLICT(action_name) DO UPDATE SET
                    category = excluded.category,
                    granted_at = excluded.granted_at
                """,
                (action_name, category, _utc_now_iso()),
            )
            self.connection.commit()
            self._permanent_grants.add(action_name)
            self._audit_unsafe(
                action_name,
                category,
                PermissionLevel.ALLOWED,
                f"'{action_name}' granted permanently.",
                "",
            )

    def revoke_permanent(self, action_name: str) -> bool:
        with self._lock:
            cursor = self.connection.execute(
                "DELETE FROM permission_grants WHERE action_name = ?",
                (action_name,),
            )
            self.connection.commit()
            revoked = cursor.rowcount > 0
            if revoked:
                self._permanent_grants.discard(action_name)
                self._audit_unsafe(
                    action_name,
                    "",
                    PermissionLevel.DENIED,
                    f"'{action_name}' permanent grant revoked.",
                    "",
                )
        return revoked

    def answer_pending(self, request_id: int, approved: bool) -> bool:
        with self._lock:
            row = self.connection.execute(
                "SELECT action_name, category, context FROM pending_requests WHERE id = ? AND resolved_at IS NULL",
                (request_id,),
            ).fetchone()
            if row is None:
                return False
            self.connection.execute(
                """
                UPDATE pending_requests
                SET resolved_at = ?, approved = ?
                WHERE id = ?
                """,
                (_utc_now_iso(), int(approved), request_id),
            )
            self.connection.commit()

            level = PermissionLevel.ALLOWED if approved else PermissionLevel.DENIED
            reason = (
                f"Pending request #{request_id} approved by human."
                if approved
                else f"Pending request #{request_id} denied by human."
            )
            self._audit_unsafe(row["action_name"], row["category"], level, reason, row["context"])
        return True

    def get_pending(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.connection.execute(
                """
                SELECT id, action_name, category, context, requested_at
                FROM pending_requests
                WHERE resolved_at IS NULL
                ORDER BY requested_at ASC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get_audit_log(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.connection.execute(
                """
                SELECT id, action_name, category, level, reason, context, timestamp
                FROM audit_log
                ORDER BY id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        return [dict(row) for row in rows]

    def _audit(
        self,
        action_name: str,
        category: str,
        level: PermissionLevel,
        reason: str,
        context: str,
    ) -> None:
        with self._lock:
            self._audit_unsafe(action_name, category, level, reason, context)

    def _audit_unsafe(
        self,
        action_name: str,
        category: str,
        level: PermissionLevel,
        reason: str,
        context: str,
    ) -> None:
        """Insert an audit row. Caller must already hold self._lock."""
        self.connection.execute(
            """
            INSERT INTO audit_log (action_name, category, level, reason, context, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (action_name, category, level.value, reason, context, _utc_now_iso()),
        )
        self.connection.commit()

    def close(self) -> None:
        with self._lock:
            self.connection.close()
