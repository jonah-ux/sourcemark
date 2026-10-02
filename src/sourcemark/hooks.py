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
from .redact import redact

MODES = ("off", "shadow", "warn", "enforce")
# Verdicts that mean "this citation is not backed by the session's evidence".
FAILING = {
    "unread_lines", "out_of_range", "unread_file", "nonexistent", "unresolved",
    "quote_mismatch", "partial", "unknown_token", "token_stale",
}
# Reported but not blocking by default: agents often give the user a link to click
# (a console, a login page) rather than citing a source. SOURCEMARK_STRICT_URLS=1 makes it block.
WARNING = {"url_unsourced"}


def failing() -> set[str]:
    if os.environ.get("SOURCEMARK_STRICT_URLS", "").lower() in ("1", "true", "yes"):
        return FAILING | WARNING
    return FAILING


def _last_turn_texts(texts: list[tuple[str, str]], turns: list[int]) -> list[str]:
    """All assistant text written since the most recent human prompt."""
    if not texts:
        return []
    if len(turns) != len(texts):
        return [texts[-1][1]]
    last = turns[-1]
    return [t for (_, t), n in zip(texts, turns) if n == last]


def _read_settled(tpath: str, tries: int = 5, wait: float = 0.2):
    """Read the transcript; if the turn's final assistant text may not be flushed yet, re-read.

    The runtime can invoke Stop hooks a moment before the last message reaches the
    transcript file. Re-read briefly while the file is still growing.
    """
    size = -1
    for i in range(tries):
        try:
            cur = os.path.getsize(tpath)
        except OSError:
            cur = -1
        sess, texts = read_claude_transcript(tpath)
        turn_texts = _last_turn_texts(texts, sess.text_turns)
        if turn_texts and cur == size:
            break
        size = cur
        if i < tries - 1:
            time.sleep(wait)
    return sess, texts, turn_texts


def summarize(rep: Report) -> str:
    bad = [c for c in rep.checks if c.verdict in FAILING | WARNING]
    head = f"sourcemark: {rep.total} citation(s), {rep.passing} backed by this session"
    if not bad:
        return head
    items = "; ".join(f"{c.raw[:80]} → {c.verdict}" + (f" ({c.detail})" if c.detail else "") for c in bad[:6])
    return f"{head}, {len(bad)} not backed: {items}"


def _redacted(value: Any) -> Any:
    """Cited text (URLs, paths, quotes) is stored in the ledger: never store a secret in it."""
    if isinstance(value, str):
        return redact(value)[0]
    if isinstance(value, dict):
        return {k: _redacted(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redacted(v) for v in value]
    return value


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
    sess, texts, turn_texts = _read_settled(tpath)
    rep = Report()
    try:
        led_for_tokens: Any = Ledger(ledger_path)
    except Exception:
        led_for_tokens = None
    try:
        for t in turn_texts:
            rep.checks.extend(check_text(t, sess, ledger=led_for_tokens).checks)
    finally:
        if led_for_tokens is not None:
            led_for_tokens.close()
    elapsed = (time.perf_counter() - t0) * 1000
    try:
        with Ledger(ledger_path) as led:
            led.append(
                "check",
                _redacted({"mode": mode, "elapsed_ms": round(elapsed, 1), "payload_keys": sorted(payload),
                           "turn_texts": len(turn_texts), **rep.to_dict()}),
                session=payload.get("session_id"),
            )
    except Exception:  # the ledger must never break the agent
        pass
    bad = [c for c in rep.checks if c.verdict in failing()]
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


def run_stop(stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout, ledger_path: str | None = None) -> int:
    try:
        payload = json.load(stdin)
    except (json.JSONDecodeError, ValueError):
        return 0
    try:
        out = stop(payload, ledger_path=ledger_path)
    except Exception as e:  # fail open: a citation checker must never wedge an agent
        print(f"sourcemark hook error (ignored): {e}", file=sys.stderr)
        return 0
    if out:
        stdout.write(json.dumps(out))
    return 0


__all__ = ["stop", "run_stop", "summarize", "failing", "FAILING", "WARNING", "PASSING", "MODES"]
