"""Append-only, hash-chained local ledger (SQLite).

Every event row stores ``hash = sha256(prev_hash + canonical_json(payload))``.
Rewriting or deleting an old row breaks the chain, which ``verify()`` reports.
Marks are also indexed by id and by short token for lookup.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
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
CREATE TABLE IF NOT EXISTS meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
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
            n = self.db.execute("SELECT count(*) FROM events").fetchone()[0]
            self.db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('head', ?), ('count', ?)", (h, str(n)))
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return h

    def put_mark(self, mark: Mark, session: str | None = None) -> str:
        body = _canon(mark.to_dict())
        cur = self.db.execute(
            "INSERT OR IGNORE INTO marks(id, token, kind, body, created_at) VALUES (?,?,?,?,?)",
            (mark.id, mark.id[4:14], mark.kind, body, mark.observed_at),
        )
        if cur.rowcount:
            # The chain covers the mark's full stored body, so editing it later is detectable.
            body_sha = hashlib.sha256(body.encode()).hexdigest()
            self.append("mark", {"id": mark.id, "body_sha256": body_sha, "source": mark.source}, session)
        return mark.id

    def get_mark(self, ref: str) -> Mark | None:
        ref = ref.strip()
        if ref.startswith("[sm:") and ref.endswith("]"):
            ref = ref[4:-1]
        row = self.db.execute("SELECT body FROM marks WHERE id = ? OR token = ?", (ref, ref)).fetchone()
        if row is None and len(ref) >= 6 and re.fullmatch(r"[a-z2-7]+", ref):
            rows = self.db.execute("SELECT body FROM marks WHERE substr(token, 1, ?) = ?", (len(ref), ref)).fetchall()
            if len(rows) > 1:
                raise LookupError(f"{len(rows)} marks start with {ref!r}; use more characters")
            row = rows[0] if rows else None
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

    def verify(self, anchor: dict[str, Any] | None = None) -> dict[str, Any]:
        """Recompute the chain and cross-check marks.

        Detects: edited or reordered events, deleted events (against the stored head and count),
        mark rows whose body no longer matches the chained fingerprint, and mark rows that were
        inserted without an event. A database rewritten wholesale (events, marks and meta) can
        only be caught against an ``anchor`` saved elsewhere: ``{"count": n, "head": hash}``.
        """
        problems: list[str] = []
        prev, n, broken_at = GENESIS, 0, None
        heads: list[str] = []
        mark_events: dict[str, str] = {}
        for seq, kind, session, payload, stored_prev, h in self.db.execute(
            "SELECT seq, kind, session, payload, prev_hash, hash FROM events ORDER BY seq"
        ):
            n += 1
            if broken_at is None and (stored_prev != prev or _chain(prev, kind, session, payload) != h):
                broken_at = seq
                problems.append(f"chain broken at event {seq}")
            prev = h
            heads.append(h)
            if kind == "mark":
                p = json.loads(payload)
                if "body_sha256" in p:
                    mark_events[p["id"]] = p["body_sha256"]
        meta = dict(self.db.execute("SELECT key, value FROM meta").fetchall())
        if meta:
            if meta.get("head") != (heads[-1] if heads else None) or meta.get("count") != str(n):
                problems.append(f"events missing or added: head/count say {meta.get('count')}, found {n}")
        for mid, body in self.db.execute("SELECT id, body FROM marks"):
            want = mark_events.get(mid)
            got = hashlib.sha256(body.encode()).hexdigest()
            if want is None:
                problems.append(f"mark {mid[:14]} has no ledger event (inserted outside the ledger)")
            elif want != got:
                problems.append(f"mark {mid[:14]} body differs from its chained fingerprint")
        if anchor:
            k = int(anchor.get("count", 0))
            if k > n or (k and heads[k - 1] != anchor.get("head")):
                problems.append(f"anchor mismatch: the first {k} events are not the anchored history")
        return {
            "ok": not problems,
            "events": n,
            "broken_at": broken_at,
            "head": prev,
            "problems": problems,
        }

    def anchor(self) -> dict[str, Any]:
        """A small record to keep OUTSIDE this database (another disk, a commit, a message)."""
        v = self.verify()
        return {"count": v["events"], "head": v["head"], "at": time.time()}
