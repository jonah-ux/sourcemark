"""Agent-runtime hooks.

``stop`` runs when an agent finishes a turn. Claude Code passes the transcript
path, which already records every read and write, so one hook is enough: parse
the session, check the citations in the final assistant turn, record the
report in the ledger, and (by mode) stay silent, warn, or ask the agent to fix
its citations before it stops.

Modes (``SOURCEMARK_MODE``): ``off`` | ``shadow`` (record only, default) |
``warn`` (show a one-line summary) | ``enforce`` (block the stop with the reasons
so the agent corrects or removes unsupported citations).
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, TextIO

from .check import PASSING, Report, check_text
from .ledger import Ledger
from .observe import read_claude_transcript

MODES = ("off", "shadow", "warn", "enforce")
# Verdicts that mean "this citation is not backed by the session's evidence".
FAILING = {"unread_lines", "out_of_range", "unread_file", "nonexistent", "quote_mismatch", "url_unsourced", "partial"}


def _last_turn_texts(texts: list[tuple[str, str]], turns: list[int]) -> list[str]:
    """All assistant text written since the most recent human prompt."""
    if not texts:
        return []
    if len(turns) != len(texts):
        return [texts[-1][1]]
    last = turns[-1]
    return [t for (_, t), n in zip(texts, turns) if n == last]


def summarize(rep: Report) -> str:
    bad = [c for c in rep.checks if c.verdict in FAILING]
    head = f"sourcemark: {rep.total} citation(s), {rep.passing} backed by this session"
    if not bad:
        return head
    items = "; ".join(f"{c.raw[:80]} → {c.verdict}" + (f" ({c.detail})" if c.detail else "") for c in bad[:6])
    return f"{head}, {len(bad)} not backed: {items}"


def stop(payload: dict[str, Any], mode: str | None = None, ledger_path: str | None = None) -> dict[str, Any] | None:
    mode = (mode or os.environ.get("SOURCEMARK_MODE") or "shadow").lower()
    if mode == "off" or mode not in MODES:
        return None
    if payload.get("stop_hook_active") and mode == "enforce":
        mode = "warn"  # never loop: a second stop after a block only warns
    tpath = payload.get("transcript_path")
    if not tpath or not os.path.isfile(tpath):
        return None
    t0 = time.perf_counter()
    sess, texts = read_claude_transcript(tpath)
    rep = Report()
    for t in _last_turn_texts(texts, sess.text_turns):
        rep.checks.extend(check_text(t, sess).checks)
    elapsed = (time.perf_counter() - t0) * 1000
    try:
        with Ledger(ledger_path) as led:
            led.append(
                "check",
                {"mode": mode, "elapsed_ms": round(elapsed, 1), **rep.to_dict()},
                session=payload.get("session_id"),
            )
    except Exception:  # the ledger must never break the agent
        pass
    bad = [c for c in rep.checks if c.verdict in FAILING]
    if mode == "shadow" or not rep.checks:
        return None
    if mode == "warn":
        return {"systemMessage": summarize(rep)}
    if not bad:
        return None  # enforce stays silent when every citation is backed
    reasons = "\n".join(
        f"- {c.raw[:120]}: {c.verdict}" + (f" — {c.detail}" if c.detail else "") for c in bad[:10]
    )
    return {
        "decision": "block",
        "reason": (
            "Some citations in your last reply are not backed by anything you read or wrote in this "
            "session. Open the cited source and correct each one, or remove it:\n" + reasons
        ),
    }


def run_stop(stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> int:
    try:
        payload = json.load(stdin)
    except (json.JSONDecodeError, ValueError):
        return 0
    try:
        out = stop(payload)
    except Exception as e:  # fail open: a citation checker must never wedge an agent
        print(f"sourcemark hook error (ignored): {e}", file=sys.stderr)
        return 0
    if out:
        stdout.write(json.dumps(out))
    return 0


__all__ = ["stop", "run_stop", "summarize", "FAILING", "PASSING", "MODES"]
