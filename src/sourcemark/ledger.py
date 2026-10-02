"""Append-only, hash-chained local ledger (SQLite).

Every event row stores ``hash = sha256(prev_hash + canonical_json(payload))``.
Rewriting or deleting an old row breaks the chain, which ``verify()`` reports.
Marks are also indexed by id and by short token for lookup.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from typing import Any, Iterator

from .anchor import Mark

GENESIS = "0" * 64

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
  seq       INTEGER PRIMARY KEY AUTOINCREMENT,
  at        REAL NOT NULL,
  kind      TEXT NOT NULL,
  session   TEXT,
  payload   TEXT NOT NULL,
  prev_hash TEXT NOT NULL,
  hash      TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS marks (
  id         TEXT PRIMARY KEY,
  token      TEXT NOT NULL,
  kind       TEXT NOT NULL,
  body       TEXT NOT NULL,
  created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS marks_token ON marks(token);
CREATE INDEX IF NOT EXISTS events_session ON events(session);
"""


def default_path() -> str:
    home = os.environ.get("SOURCEMARK_HOME") or os.path.join(os.path.expanduser("~"), ".sourcemark")
    return os.path.join(home, "ledger.db")


def _canon(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _chain(prev: str, kind: str, session: str | None, payload_json: str) -> str:
    return hashlib.sha256(f"{prev}|{kind}|{session or ''}|{payload_json}".encode()).hexdigest()


class Ledger:
    def __init__(self, path: str | None = None):
        self.path = path or default_path()
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA busy_timeout=30000")
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def append(self, kind: str, payload: dict[str, Any], session: str | None = None) -> str:
        """Append an event inside an IMMEDIATE transaction so concurrent writers keep one chain."""
        body = _canon(payload)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute("SELECT hash FROM events ORDER BY seq DESC LIMIT 1").fetchone()
            prev = row[0] if row else GENESIS
            h = _chain(prev, kind, session, body)
            self.db.execute(
                "INSERT INTO events(at, kind, session, payload, prev_hash, hash) VALUES (?,?,?,?,?,?)",
                (time.time(), kind, session, body, prev, h),
            )
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return h

    def put_mark(self, mark: Mark, session: str | None = None) -> str:
        body = mark.to_dict()
        self.db.execute(
            "INSERT OR IGNORE INTO marks(id, token, kind, body, created_at) VALUES (?,?,?,?,?)",
            (mark.id, mark.id[4:14], mark.kind, _canon(body), mark.observed_at),
        )
        self.append("mark", {"id": mark.id, "fingerprints": mark.fingerprints, "source": mark.source}, session)
        return mark.id

    def get_mark(self, ref: str) -> Mark | None:
        ref = ref.strip()
        if ref.startswith("[sm:") and ref.endswith("]"):
            ref = ref[4:-1]
        row = self.db.execute("SELECT body FROM marks WHERE id = ? OR token = ?", (ref, ref)).fetchone()
        if row is None and len(ref) >= 6:
            row = self.db.execute("SELECT body FROM marks WHERE token LIKE ?", (ref + "%",)).fetchone()
        return Mark.from_dict(json.loads(row[0])) if row else None

    def marks(self) -> Iterator[Mark]:
        for (body,) in self.db.execute("SELECT body FROM marks ORDER BY created_at"):
            yield Mark.from_dict(json.loads(body))

    def events(self, session: str | None = None, kind: str | None = None) -> Iterator[dict[str, Any]]:
        q, args = "SELECT seq, at, kind, session, payload, hash FROM events WHERE 1=1", []
        if session is not None:
            q += " AND session = ?"
            args.append(session)
        if kind is not None:
            q += " AND kind = ?"
            args.append(kind)
        for seq, at, k, s, payload, h in self.db.execute(q + " ORDER BY seq", args):
            yield {"seq": seq, "at": at, "kind": k, "session": s, "payload": json.loads(payload), "hash": h}

    def verify(self) -> dict[str, Any]:
        """Recompute the chain. Returns {"ok": bool, "events": n, "broken_at": seq|None}."""
        prev, n = GENESIS, 0
        for seq, kind, session, payload, stored_prev, h in self.db.execute(
            "SELECT seq, kind, session, payload, prev_hash, hash FROM events ORDER BY seq"
        ):
            n += 1
            if stored_prev != prev or _chain(prev, kind, session, payload) != h:
                return {"ok": False, "events": n, "broken_at": seq}
            prev = h
        return {"ok": True, "events": n, "broken_at": None, "head": prev}
