"""SQLite log of agent tasks and their events."""

from dataclasses import dataclass
import json
import sqlite3
import threading
import time


SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    started_at REAL NOT NULL,
    finished_at REAL,
    history BLOB
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id INTEGER NOT NULL REFERENCES tasks(id),
    kind TEXT NOT NULL,
    payload TEXT NOT NULL,
    at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS events_task ON events(task_id, id);
"""

RUNNING = "running"
DONE = "done"
FAILED = "failed"


@dataclass(frozen=True)
class Task:
    id: int
    message: str
    status: str
    history: bytes | None


class EventStore:
    def __init__(self, path: str, clock=time.time):
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.executescript(SCHEMA)
        self._lock = threading.Lock()
        self._clock = clock

    def close(self) -> None:
        self._db.close()

    def start_task(self, message: str) -> int:
        with self._lock, self._db:
            cursor = self._db.execute(
                "INSERT INTO tasks (message, status, started_at) VALUES (?, ?, ?)",
                (message, RUNNING, self._clock()),
            )
            return int(cursor.lastrowid)

    def save_history(self, task_id: int, history: bytes) -> None:
        with self._lock, self._db:
            self._db.execute("UPDATE tasks SET history = ? WHERE id = ?", (history, task_id))

    def finish_task(self, task_id: int, status: str = DONE) -> None:
        with self._lock, self._db:
            self._db.execute(
                "UPDATE tasks SET status = ?, finished_at = ? WHERE id = ?",
                (status, self._clock(), task_id),
            )

    def interrupted_task(self) -> Task | None:
        """Returns the newest task still marked running."""
        with self._lock:
            row = self._db.execute(
                "SELECT id, message, status, history FROM tasks WHERE status = ? "
                "ORDER BY id DESC LIMIT 1",
                (RUNNING,),
            ).fetchone()
        return Task(*row) if row else None

    def task(self, task_id: int) -> Task | None:
        with self._lock:
            row = self._db.execute(
                "SELECT id, message, status, history FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
        return Task(*row) if row else None

    def record(self, task_id: int, kind: str, **payload) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO events (task_id, kind, payload, at) VALUES (?, ?, ?, ?)",
                (task_id, kind, json.dumps(payload, ensure_ascii=False, default=str), self._clock()),
            )

    def events(self, task_id: int, kind: str | None = None) -> list[dict]:
        query = "SELECT kind, payload, at FROM events WHERE task_id = ?"
        params: tuple = (task_id,)
        if kind is not None:
            query += " AND kind = ?"
            params += (kind,)
        with self._lock:
            rows = self._db.execute(query + " ORDER BY id", params).fetchall()
        return [{"kind": k, "at": at, **json.loads(payload)} for k, payload, at in rows]
