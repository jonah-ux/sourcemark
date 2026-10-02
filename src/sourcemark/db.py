"""Citations for database values (PostgreSQL).

A db mark pins *which* value was read: database label, schema, table, primary
key, column(s), a fingerprint of each value, and the database clock when it was
read. Resolving re-reads the same row by primary key and reports:

* ``intact``  every cited column has the same fingerprint
* ``drifted`` the row exists but at least one cited value changed (columns listed)
* ``deleted`` no row with that primary key exists any more

Queries go through a *runner*: any callable that takes SQL and returns a list
of dict rows. :class:`PsqlRunner` shells out to ``psql`` (no Python driver
needed) and forces every statement to run read-only. Callers can plug in their
own runner (a connection pool, an HTTP SQL endpoint, ...).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .anchor import Mark, compute_id
from .redact import find_secrets
from .textnorm import sha256_hex

Runner = Callable[[str], list[dict[str, Any]]]

NULL_SENTINEL = "∅null"
SENSITIVE_COLUMN = re.compile(
    r"(?i)(pass(word)?|secret|token|api[_-]?key|private|ssn|social|dob|birth|email|phone|address|card|iban|salary)"
)
MAX_EXCERPT = 200


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def quote_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    return "'" + str(value).replace("'", "''") + "'"


def canonical(value: Any) -> str:
    """Stable text form of a JSON value; NULL gets an explicit sentinel."""
    if value is None:
        return NULL_SENTINEL
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def value_fingerprint(value: Any) -> str:
    return sha256_hex(canonical(value))


class PsqlRunner:
    """Run SQL with ``psql`` and parse a JSON result. Every query is read-only."""

    def __init__(self, dsn: str | None = None, psql: str = "psql", timeout: float = 30.0):
        self.dsn = dsn or os.environ.get("SOURCEMARK_DSN") or os.environ.get("DATABASE_URL")
        self.psql = psql
        self.timeout = timeout

    def __call__(self, sql: str) -> list[dict[str, Any]]:
        wrapped = (
            "SET default_transaction_read_only = on; "
            f"SELECT coalesce(json_agg(q), '[]'::json) FROM ({sql.rstrip().rstrip(';')}) q;"
        )
        cmd = [self.psql, "-X", "-q", "-A", "-t", "-v", "ON_ERROR_STOP=1"]
        if self.dsn:
            cmd += ["-d", self.dsn]
        cmd += ["-c", wrapped]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
        if out.returncode != 0:
            raise RuntimeError(out.stderr.strip() or f"psql exited {out.returncode}")
        text = out.stdout.strip()
        return json.loads(text) if text else []


def primary_key(run: Runner, schema: str, table: str) -> list[str]:
    reg = quote_literal(f"{quote_ident(schema)}.{quote_ident(table)}")
    rows = run(
        "SELECT a.attname AS col FROM pg_index i "
        "JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey) "
        f"WHERE i.indrelid = {reg}::regclass AND i.indisprimary "
        "ORDER BY array_position(i.indkey, a.attnum)"
    )
    return [r["col"] for r in rows]


def _row_sql(schema: str, table: str, pk: dict[str, Any], columns: Iterable[str]) -> str:
    cols = ", ".join(f"to_jsonb(t) -> {quote_literal(c)} AS {quote_ident(c)}" for c in columns)
    where = " AND ".join(f"t.{quote_ident(k)}::text = {quote_literal(v)}" for k, v in pk.items())
    return f"SELECT {cols}, now() AS sourcemark_now FROM {quote_ident(schema)}.{quote_ident(table)} t WHERE {where}"


@dataclass
class DbResolution:
    mark_id: str
    status: str  # intact | drifted | deleted | error
    changed: list[str] = field(default_factory=list)
    checked_at: str | None = None
    elapsed_ms: float = 0.0
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def mark_row(
    run: Runner,
    table: str,
    pk: dict[str, Any] | Any,
    columns: list[str],
    *,
    schema: str = "public",
    database: str = "default",
    observer: dict[str, Any] | None = None,
) -> Mark:
    """Cite ``columns`` of one row. ``pk`` may be a dict or a single value for a 1-column key."""
    if not isinstance(pk, dict):
        keys = primary_key(run, schema, table)
        if len(keys) != 1:
            raise ValueError(f"{schema}.{table} has a {len(keys)}-column primary key; pass pk as a dict")
        pk = {keys[0]: pk}
    rows = run(_row_sql(schema, table, pk, columns))
    if not rows:
        raise LookupError(f"no row in {schema}.{table} with {pk}")
    if len(rows) > 1:
        raise LookupError(f"{len(rows)} rows matched {pk}; not a primary key")
    row = rows[0]
    values: dict[str, dict[str, Any]] = {}
    for c in columns:
        v = row.get(c)
        excerpt: str | None = None
        text = canonical(v)
        if not SENSITIVE_COLUMN.search(c) and len(text) <= MAX_EXCERPT and not find_secrets(text):
            excerpt = text
        values[c] = {"fingerprint": value_fingerprint(v), "excerpt": excerpt}
    source = {"database": database, "schema": schema, "table": table, "pk": {k: str(v) for k, v in pk.items()}}
    identity = {"kind": "db", **source, "columns": sorted(columns),
                "values": {c: values[c]["fingerprint"] for c in sorted(columns)}}
    m = Mark(
        kind="db",
        source=source,
        quote={"columns": values},
        position={"read_at": row.get("sourcemark_now")},
        fingerprints={"row": sha256_hex(canonical({c: values[c]["fingerprint"] for c in sorted(columns)}))},
        observed_at=time.time(),
        observer=observer or {},
        redacted=[c for c in columns if values[c]["excerpt"] is None],
    )
    m.id = compute_id(identity)
    return m


def resolve_row(mark: Mark, run: Runner) -> DbResolution:
    t0 = time.perf_counter()
    src = mark.source
    cols = list(mark.quote["columns"].keys())
    try:
        rows = run(_row_sql(src["schema"], src["table"], src["pk"], cols))
    except Exception as e:  # unreachable DB, dropped table, permissions...
        return DbResolution(mark.id, "error", detail=str(e)[:300], elapsed_ms=(time.perf_counter() - t0) * 1000)
    res = DbResolution(mark.id, "deleted", elapsed_ms=0.0)
    if rows:
        row = rows[0]
        res.checked_at = row.get("sourcemark_now")
        res.changed = [c for c in cols if value_fingerprint(row.get(c)) != mark.quote["columns"][c]["fingerprint"]]
        res.status = "drifted" if res.changed else "intact"
    res.elapsed_ms = (time.perf_counter() - t0) * 1000
    return res
