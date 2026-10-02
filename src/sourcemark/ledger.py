"""Append-only, hash-chained local ledger (SQLite).

Every event row stores ``hash = sha256(prev_hash | kind | session | canonical_json(payload) | at)``.
Rewriting, reordering, re-timing or deleting a row breaks the chain, which ``verify()`` reports.
Marks are also indexed by id and by short token for lookup; ``verify()`` checks those indexes
against the chain too.
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


CHAIN_VERSION = "2"  # 2: the timestamp is chained too; 1 (older ledgers): it was not


def _chain(prev: str, kind: str, session: str | None, payload_json: str, at: float | None = None) -> str:
    tail = "" if at is None else f"|{at!r}"
    return hashlib.sha256(f"{prev}|{kind}|{session or ''}|{payload_json}{tail}".encode()).hexdigest()


class Ledger:
    def __init__(self, path: str | None = None):
        self.path = path or default_path()
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA busy_timeout=30000")
        self.db.executescript(SCHEMA)
        if self.db.execute("SELECT count(*) FROM events").fetchone()[0] == 0:
            self.db.execute("INSERT OR IGNORE INTO meta(key, value) VALUES ('chain', ?)", (CHAIN_VERSION,))
        row = self.db.execute("SELECT value FROM meta WHERE key = 'chain'").fetchone()
        self.chain_version = row[0] if row else "1"

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def append(self, kind: str, payload: dict[str, Any], session: str | None = None) -> str:
        """Append an event. ``mark`` events are written only by :meth:`put_mark`, together with
        the mark row they vouch for."""
        if kind == "mark":
            raise ValueError("mark events are written by put_mark(), not append()")
        return self._append(kind, payload, session)

    def _append(self, kind: str, payload: dict[str, Any], session: str | None = None) -> str:
        """Append inside an IMMEDIATE transaction so concurrent writers keep one chain."""
        body = _canon(payload)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute("SELECT hash FROM events ORDER BY seq DESC LIMIT 1").fetchone()
            prev = row[0] if row else GENESIS
            at = time.time()
            h = _chain(prev, kind, session, body, at if self.chain_version == "2" else None)
            self.db.execute(
                "INSERT INTO events(at, kind, session, payload, prev_hash, hash) VALUES (?,?,?,?,?,?)",
                (at, kind, session, body, prev, h),
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
            self._append("mark", {"id": mark.id, "body_sha256": body_sha, "source": mark.source}, session)
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
        timed = self.chain_version == "2"
        for seq, at, kind, session, payload, stored_prev, h in self.db.execute(
            "SELECT seq, at, kind, session, payload, prev_hash, hash FROM events ORDER BY seq"
        ):
            n += 1
            if broken_at is None and (stored_prev != prev or _chain(prev, kind, session, payload, at if timed else None) != h):
                broken_at = seq
                problems.append(f"chain broken at event {seq}")
            prev = h
            heads.append(h)
            if kind == "mark":
                p = json.loads(payload)
                if "body_sha256" in p:
                    if p["id"] in mark_events:
                        problems.append(f"mark {p['id'][:14]} has more than one mark event (re-vouched after an edit?)")
                    else:
                        mark_events[p["id"]] = p["body_sha256"]
        meta = dict(self.db.execute("SELECT key, value FROM meta").fetchall())
        if "head" in meta or "count" in meta:
            if meta.get("head") != (heads[-1] if heads else None) or meta.get("count") != str(n):
                problems.append(f"events missing or added: head/count say {meta.get('count')}, found {n}")
        seqs = [r[0] for r in self.db.execute("SELECT seq FROM events ORDER BY seq")]
        if seqs != list(range(1, len(seqs) + 1)):
            problems.append("event sequence numbers have gaps (events were deleted)")
        row = self.db.execute("SELECT seq FROM sqlite_sequence WHERE name = 'events'").fetchone()
        if row and row[0] != n:
            problems.append(f"{row[0]} events were ever written but {n} remain")
        if n and timed and not ("head" in meta or "count" in meta):
            # A v2 ledger always records head/count with its first event: their absence means removal.
            problems.append("head/count records are missing (deleted to hide removed events?)")
        rows = set()
        for mid, token, body in self.db.execute("SELECT id, token, body FROM marks"):
            rows.add(mid)
            want = mark_events.get(mid)
            got = hashlib.sha256(body.encode()).hexdigest()
            if want is None:
                problems.append(f"mark {mid[:14]} has no ledger event (inserted outside the ledger)")
            elif want != got:
                problems.append(f"mark {mid[:14]} body differs from its chained fingerprint")
            if token != mid[4:14]:
                problems.append(f"mark {mid[:14]} token index was changed (lookups would be redirected)")
            try:
                if json.loads(body).get("id") != mid:
                    problems.append(f"mark {mid[:14]} body carries a different id")
            except ValueError:
                problems.append(f"mark {mid[:14]} body is not JSON")
        for mid in sorted(set(mark_events) - rows):
            problems.append(f"mark {mid[:14]} was deleted (its event remains)")
        if not timed:
            problems_note = "chain v1: event timestamps are not covered by the hash"
        else:
            problems_note = None
        if anchor:
            k = int(anchor.get("count", 0))
            if k > n or (k and heads[k - 1] != anchor.get("head")):
                problems.append(f"anchor mismatch: the first {k} events are not the anchored history")
        out = {
            "ok": not problems,
            "events": n,
            "broken_at": broken_at,
            "head": prev,
            "problems": problems,
            "chain_version": self.chain_version,
        }
        if problems_note:
            out["note"] = problems_note
        return out

    def anchor(self) -> dict[str, Any]:
        """A small record to keep OUTSIDE this database (another disk, a commit, a message)."""
        v = self.verify()
        return {"count": v["events"], "head": v["head"], "at": time.time()}
