"""Long-term memory: SQLite for the records, a NumPy matrix for similarity search."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np

from .config import contained
from .neural import NEURONS, Bits, encode, pack, shared

SCHEMA = """
CREATE TABLE IF NOT EXISTS memory (
    id INTEGER PRIMARY KEY,
    ts REAL NOT NULL,
    region TEXT NOT NULL,
    kind TEXT NOT NULL,
    content TEXT NOT NULL,
    parent INTEGER,
    value REAL NOT NULL DEFAULT 0,
    rewards INTEGER NOT NULL DEFAULT 0,
    heuristics TEXT NOT NULL DEFAULT '[]',
    lesson TEXT,
    vec BLOB
);
CREATE TABLE IF NOT EXISTS heuristic (
    id INTEGER PRIMARY KEY,
    region TEXT NOT NULL,
    text TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 0.5,
    uses INTEGER NOT NULL DEFAULT 0,
    reward_sum REAL NOT NULL DEFAULT 0,
    active INTEGER NOT NULL DEFAULT 1,
    origin TEXT NOT NULL DEFAULT 'cerebellum',
    created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS reward (
    id INTEGER PRIMARY KEY,
    ts REAL NOT NULL,
    memory_id INTEGER NOT NULL,
    region TEXT NOT NULL,
    reward REAL NOT NULL,
    rpe REAL NOT NULL,
    source TEXT NOT NULL,
    reason TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


class Memory:
    def __init__(self, path: Path | str) -> None:
        self.db = sqlite3.connect(path, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        if contained():  # a folder shared into a container offers no shared memory for the WAL index
            self.db.execute("PRAGMA locking_mode=EXCLUSIVE")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(SCHEMA)
        self._ids: list[int] = []
        self._kinds: list[str] = []
        self._mat: np.ndarray | None = None
        self._codes: np.ndarray | None = None  # every memory as a pattern of neurons, 64 bytes each
        self._load_index()

    def close(self) -> None:
        self.db.close()

    # -- vector index ---------------------------------------------------------

    def _load_index(self) -> None:
        """Index the vectors written by the embedding model that was used most recently."""
        self._ids, self._kinds, self._mat, self._codes = [], [], None, None
        last = self.db.execute(
            "SELECT LENGTH(vec) FROM memory WHERE vec IS NOT NULL ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if last is None:
            return
        rows = self.db.execute(
            "SELECT id, kind, vec FROM memory WHERE LENGTH(vec)=? ORDER BY id", (last[0],)
        )
        for row in rows:
            self._index(row["id"], row["kind"], np.frombuffer(row["vec"], dtype=np.float32))

    def _index(self, memory_id: int, kind: str, vec: np.ndarray) -> None:
        if self._mat is None or self._mat.shape[1] != len(vec):
            # First vector, or the embedding model changed: older vectors stay on disk only.
            self._ids, self._kinds = [], []
            self._mat = np.zeros((1024, len(vec)), dtype=np.float32)
            self._codes = np.zeros((1024, len(pack(encode(vec)))), dtype=np.uint8)
        n = len(self._ids)
        if n == len(self._mat):
            self._mat = np.concatenate([self._mat, np.zeros_like(self._mat)])
            self._codes = np.concatenate([self._codes, np.zeros_like(self._codes)])
        self._mat[n] = vec
        self._codes[n] = pack(encode(vec))
        self._ids.append(memory_id)
        self._kinds.append(kind)

    def bits(self, memory_id: int | None) -> Bits | None:
        """A memory as its pattern of neurons."""
        return self.pattern(self.get(memory_id) if memory_id else None)

    @staticmethod
    def pattern(row: sqlite3.Row | None) -> Bits | None:
        """The pattern of a memory that has already been fetched."""
        if row is None or row["vec"] is None:
            return None
        return encode(np.frombuffer(row["vec"], dtype=np.float32))

    def resonate(
        self, bits: Bits, exclude: set[int] | None = None, kinds: tuple[str, ...] = (), last: int = 0
    ) -> tuple[int, float] | None:
        """The memory whose neurons a pattern lights most, and how much of it they share. No
        model and no embedding: bits against bits, a few microseconds for every memory there is."""
        n = len(self._ids)
        if self._codes is None or n == 0:
            return None
        start = max(0, n - last) if last else 0
        share = shared(self._codes[start:n], bits)
        for index in np.argsort(share)[::-1][:64]:
            memory_id = self._ids[start + index]
            if (not exclude or memory_id not in exclude) and (not kinds or self._kinds[start + index] in kinds):
                return memory_id, float(share[index])
        return None

    def dwelt(self, bits: Bits, kinds: tuple[str, ...], floor: float) -> int:
        """How many memories of these kinds share at least `floor` of a pattern: how much the mind
        has been on that theme."""
        n = len(self._ids)
        if self._codes is None or n == 0:
            return 0
        of_kind = np.fromiter((kind in kinds for kind in self._kinds), dtype=bool, count=n)
        return int((shared(self._codes[:n], bits)[of_kind] >= floor).sum())

    def novelty(self, vec: np.ndarray | None, window: int = 200) -> float:
        """1 minus the highest similarity to anything remembered recently."""
        n = len(self._ids)
        if vec is None or self._mat is None or n == 0 or self._mat.shape[1] != len(vec):
            return 1.0
        return float(1.0 - (self._mat[max(0, n - window) : n] @ vec).max())

    def search(self, vec: np.ndarray, k: int = 4, exclude: set[int] | None = None) -> list[sqlite3.Row]:
        """Nearest memories, with strongly rewarded or punished ones weighted up."""
        n = len(self._ids)
        if self._mat is None or n == 0 or self._mat.shape[1] != len(vec):
            return []
        sims = self._mat[:n] @ vec
        top = np.argsort(sims)[::-1][: k * 4]
        ids = [self._ids[i] for i in top if not exclude or self._ids[i] not in exclude]
        if not ids:
            return []
        sim_by_id = {self._ids[i]: float(sims[i]) for i in top}
        marks = ",".join("?" * len(ids))
        rows = self.db.execute(f"SELECT * FROM memory WHERE id IN ({marks})", ids).fetchall()
        rows.sort(key=lambda r: sim_by_id[r["id"]] * (1 + 0.5 * abs(r["value"])), reverse=True)
        return rows[:k]

    # -- records ----------------------------------------------------------------

    def add(
        self,
        region: str,
        kind: str,
        content: str,
        parent: int | None = None,
        heuristics: list[int] | None = None,
        vec: np.ndarray | None = None,
        value: float = 0.0,
    ) -> int:
        cur = self.db.execute(
            "INSERT INTO memory (ts, region, kind, content, parent, heuristics, vec, value) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (
                time.time(), region, kind, content, parent, json.dumps(heuristics or []),
                None if vec is None else vec.astype(np.float32).tobytes(), value,
            ),
        )
        if vec is not None:
            self._index(cur.lastrowid, kind, vec)
        return cur.lastrowid

    def get(self, memory_id: int) -> sqlite3.Row | None:
        return self.db.execute("SELECT * FROM memory WHERE id=?", (memory_id,)).fetchone()

    def count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM memory").fetchone()[0]

    def last_id(self) -> int:
        return self.db.execute("SELECT COALESCE(MAX(id), 0) FROM memory").fetchone()[0]

    def recent(self, region: str, kind: str, limit: int) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT * FROM memory WHERE region=? AND kind=? ORDER BY id DESC LIMIT ?",
            (region, kind, limit),
        ).fetchall()

    def since(self, memory_id: int, kinds: tuple[str, ...], limit: int) -> list[sqlite3.Row]:
        marks = ",".join("?" * len(kinds))
        rows = self.db.execute(
            f"SELECT * FROM memory WHERE id>? AND kind IN ({marks}) ORDER BY id DESC LIMIT ?",
            (memory_id, *kinds, limit),
        ).fetchall()
        return rows[::-1]

    def episodes(self, since_id: int, limit: int = 4000) -> list[tuple[int, Bits | None, Bits, float, tuple[str, str]]]:
        """What happened since then, as patterns: each thought, the one it came from, what it
        turned out to be worth, and the tract it travelled: from the region of the one to the
        region of the other. This is what sleep replays."""
        rows = self.db.execute(
            "SELECT m.id, m.vec, m.value, m.region, p.vec AS before, p.region AS source FROM memory m "
            "LEFT JOIN memory p ON p.id=m.parent WHERE m.id>? AND m.vec IS NOT NULL ORDER BY m.id DESC LIMIT ?",
            (since_id, limit),
        ).fetchall()

        def pattern(blob: bytes | None) -> Bits | None:
            return None if blob is None else encode(np.frombuffer(blob, dtype=np.float32))

        return [
            (r["id"], pattern(r["before"]), pattern(r["vec"]), r["value"], (r["source"] or r["region"], r["region"]))
            for r in rows[::-1]
        ]

    def codes(self) -> np.ndarray:
        """Every memory as its pattern, packed: what the neurons' places on their sheet are laid out from."""
        if self._codes is None:
            return np.zeros((0, NEURONS // 8), dtype=np.uint8)
        return self._codes[: len(self._ids)]

    def extremes(self, since_id: int, n: int, best: bool) -> list[sqlite3.Row]:
        order = "DESC" if best else "ASC"
        sign = ">" if best else "<"
        return self.db.execute(
            f"SELECT * FROM memory WHERE id>? AND rewards>0 AND value{sign}0 "
            f"ORDER BY value {order} LIMIT ?",
            (since_id, n),
        ).fetchall()

    def adjust_value(self, memory_id: int, delta: float, lesson: str | None = None) -> None:
        self.db.execute(
            "UPDATE memory SET value=MAX(-1, MIN(1, value+?)), rewards=rewards+1, "
            "lesson=COALESCE(?, lesson) WHERE id=?",
            (delta, lesson, memory_id),
        )

    def forget_neutral(self, cap: int) -> int:
        """Above the cap, drop the oldest memories that were never strongly rewarded or punished.
        What the human said to it is kept whatever it was worth: nobody judges an exchange."""
        excess = self.count() - cap
        if excess <= 0:
            return 0
        cur = self.db.execute(
            "DELETE FROM memory WHERE id IN (SELECT id FROM memory WHERE ABS(value)<0.15 "
            "AND kind NOT IN ('insight','lesson','day','exchange') ORDER BY id LIMIT ?)",
            (excess,),
        )
        if cur.rowcount:
            self._load_index()
        return cur.rowcount

    # -- rewards ------------------------------------------------------------------

    def log_reward(self, memory_id: int, region: str, reward: float, rpe: float, source: str, reason: str) -> None:
        self.db.execute(
            "INSERT INTO reward (ts, memory_id, region, reward, rpe, source, reason) VALUES (?,?,?,?,?,?,?)",
            (time.time(), memory_id, region, reward, rpe, source, reason),
        )

    def reward_history(self, limit: int = 60) -> list[float]:
        rows = self.db.execute("SELECT reward FROM reward ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [r["reward"] for r in rows][::-1]

    def reward_stats(self) -> tuple[int, float]:
        row = self.db.execute("SELECT COUNT(*), COALESCE(AVG(reward), 0) FROM reward").fetchone()
        return row[0], row[1]

    # -- key/value state ----------------------------------------------------------

    def get_state(self, key: str, default: Any = None) -> Any:
        row = self.db.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        return default if row is None else json.loads(row["value"])

    def set_state(self, key: str, value: Any) -> None:
        self.db.execute(
            "INSERT INTO state (key, value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value)),
        )
