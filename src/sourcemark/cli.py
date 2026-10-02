"""Command-line entry point.

All commands print human-readable text by default and a single JSON document
with ``--json``. Exit codes: 0 ok, 1 a check or verification failed, 2 usage error.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any

from . import __version__
from .anchor import mark_lines, mark_text
from .check import check_text
from .db import PsqlRunner, mark_row, resolve_row
from .gitinfo import source_for
from .hooks import failing, run_stop
from .ledger import Ledger
from .observe import read_claude_transcript
from .resolve import resolve
from .textnorm import normalize_newlines

_TARGET = re.compile(r"^(?P<path>.+?)(?::(?P<l1>\d+)(?:-(?P<l2>\d+))?)?$")


def _out(args: argparse.Namespace, data: Any, text: str) -> None:
    print(json.dumps(data, indent=2, default=str) if args.json else text)


def cmd_mark(args: argparse.Namespace) -> int:
    m = _TARGET.match(args.target)
    if not m or not os.path.isfile(m.group("path")):
        print(f"sourcemark: no such file: {args.target}", file=sys.stderr)
        return 2
    path = m.group("path")
    with open(path, encoding="utf-8", errors="replace") as fh:
        doc = normalize_newlines(fh.read())
    src = source_for(path)
    if args.quote:
        start = doc.find(args.quote)
        if start < 0:
            print("sourcemark: quote not found in file", file=sys.stderr)
            return 1
        mark = mark_text(doc, start, start + len(args.quote), src)
    else:
        l1 = int(m.group("l1") or 1)
        l2 = int(m.group("l2") or l1)
        mark = mark_lines(doc, l1, l2, src)
    with Ledger(args.ledger) as led:
        led.put_mark(mark)
    _out(args, mark.to_dict(), f"{mark.token}  {path}:{mark.position['line_start']}-{mark.position['line_end']}")
    return 0


def cmd_mark_row(args: argparse.Namespace) -> int:
    run = PsqlRunner(args.dsn)
    schema, _, table = args.table.rpartition(".")
    pk: object = args.pk
    if "=" in args.pk:
        pk = dict(kv.split("=", 1) for kv in args.pk.split(","))
    mark = mark_row(run, table, pk, args.columns, schema=schema or "public", database=args.database)
    with Ledger(args.ledger) as led:
        led.put_mark(mark)
    _out(args, mark.to_dict(), f"{mark.token}  {args.database}:{schema or 'public'}.{table} {mark.source['pk']} {','.join(args.columns)}")
    return 0


def cmd_resolve(args: argparse.Namespace) -> int:
    with Ledger(args.ledger) as led:
        mark = led.get_mark(args.ref)
        if mark is None:
            print(f"sourcemark: unknown mark {args.ref}", file=sys.stderr)
            return 2
        if mark.kind == "db":
            dres = resolve_row(mark, PsqlRunner(args.dsn))
            led.append("resolve", dres.to_dict())
            extra = f" changed: {', '.join(dres.changed)}" if dres.changed else ""
            _out(args, dres.to_dict(), f"{dres.status:10}{extra}  ({dres.elapsed_ms:.1f} ms)")
            return 0 if dres.status == "intact" else 1
        res = resolve(mark, roots=args.root or [])
        led.append("resolve", res.to_dict())
    where = f"{res.path}:{res.line_start}-{res.line_end}" if res.path else "-"
    _out(args, res.to_dict(), f"{res.status:10} {where}  ({res.elapsed_ms:.1f} ms)")
    return 0 if res.status in ("intact", "shifted", "moved") else 1


def cmd_show(args: argparse.Namespace) -> int:
    with Ledger(args.ledger) as led:
        mark = led.get_mark(args.ref)
    if mark is None:
        print(f"sourcemark: unknown mark {args.ref}", file=sys.stderr)
        return 2
    if mark.kind == "db":
        cols = ", ".join(f"{c}={v['excerpt'] if v['excerpt'] is not None else '[fingerprint only]'}" for c, v in mark.quote["columns"].items())
        _out(args, mark.to_dict(), f"{mark.token} {mark.source['database']}:{mark.source['schema']}.{mark.source['table']} {mark.source['pk']}\n{cols}")
        return 0
    q = mark.quote.get("exact")
    _out(args, mark.to_dict(), f"{mark.token} {mark.source.get('path')}:{mark.position['line_start']}\n{q if q is not None else '[redacted]'}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    sess, texts = read_claude_transcript(args.transcript)
    if args.text:
        body = [open(args.text, encoding="utf-8").read()]
    elif args.all:
        body = [t for _, t in texts]
    else:
        last = sess.text_turns[-1] if sess.text_turns else None
        body = [t for (_, t), n in zip(texts, sess.text_turns) if n == last] or [t for _, t in texts[-1:]]
    from .check import Report

    from .check import UnavailableLedger
    from .ledger import default_path

    lpath = args.ledger or default_path()
    led: Any = Ledger(lpath) if os.path.isfile(lpath) else UnavailableLedger(f"no ledger at {lpath}")
    rep = Report()
    try:
        for t in body:
            rep.checks.extend(check_text(t, sess, now=args.now, ledger=led).checks)
    finally:
        if isinstance(led, Ledger):
            led.close()
    lines = [f"{c.verdict:15} {c.raw[:100]}" + (f"  ({c.detail})" if c.detail else "") for c in rep.checks]
    lines.append(f"-- {rep.passing}/{rep.total} backed by this session")
    _out(args, rep.to_dict(), "\n".join(lines))
    return 1 if any(c.verdict in failing() for c in rep.checks) else 0


def cmd_verify(args: argparse.Namespace) -> int:
    anchor = None
    if args.anchor:
        with open(args.anchor, encoding="utf-8") as fh:
            anchor = json.load(fh)
    with Ledger(args.ledger) as led:
        v = led.verify(anchor)
        if v["ok"] and args.write_anchor:
            with open(args.write_anchor, "w", encoding="utf-8") as fh:
                json.dump(led.anchor(), fh)
    text = f"ledger {'OK' if v['ok'] else 'TAMPERED'} ({v['events']} events)"
    if v["problems"]:
        text += "\n" + "\n".join(f"  - {p}" for p in v["problems"])
    _out(args, v, text)
    return 0 if v["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sourcemark",
        description="Anchor citations to quotes, file lines, and database values; re-resolve them after moves and edits.",
    )
    parser.add_argument("--version", action="version", version=f"sourcemark {__version__}")
    parser.add_argument("--ledger", help="ledger path (default: $SOURCEMARK_HOME/ledger.db)")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("mark", help="cite FILE[:LINE[-LINE]] or a --quote inside FILE")
    p.add_argument("target")
    p.add_argument("--quote", help="exact text to cite inside the file")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_mark)

    p = sub.add_parser("resolve", help="find a mark again and report what happened to it")
    p.add_argument("ref", help="mark id or [sm:token]")
    p.add_argument("--root", action="append", help="directory to search when the file moved (repeatable)")
    p.add_argument("--dsn", help="connection string for database marks")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_resolve)

    p = sub.add_parser("mark-row", help="cite columns of one database row (PostgreSQL, read-only)")
    p.add_argument("table", help="[schema.]table")
    p.add_argument("pk", help="primary key value, or col=val,col=val for composite keys")
    p.add_argument("columns", nargs="+")
    p.add_argument("--dsn", help="connection string (default: $SOURCEMARK_DSN or $DATABASE_URL)")
    p.add_argument("--database", default="default", help="label stored in the citation")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_mark_row)

    p = sub.add_parser("show", help="print a stored mark")
    p.add_argument("ref")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("check", help="check citations in a transcript's last turn against what the session read")
    p.add_argument("transcript", help="Claude Code JSONL transcript")
    p.add_argument("--text", help="check this file's text instead of the transcript's last turn")
    p.add_argument("--all", action="store_true", help="check every assistant message, not just the last turn")
    p.add_argument("--now", action="store_true", help="also report whether cited lines changed since they were read")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("verify-ledger", help="recompute the ledger hash chain and cross-check marks")
    p.add_argument("--anchor", help="JSON anchor saved earlier (detects a wholesale rewrite)")
    p.add_argument("--write-anchor", help="after a clean verify, write the current anchor here (keep it elsewhere)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("hook", help="agent runtime hooks (read the event JSON on stdin)")
    p.add_argument("event", choices=["stop"])
    p.set_defaults(func=lambda a: run_stop(ledger_path=a.ledger))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        return int(args.func(args) or 0)
    except (ValueError, LookupError, FileNotFoundError, IsADirectoryError, PermissionError) as e:
        # Bad input is a usage error (exit 2), not a crash.
        print(f"sourcemark: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
