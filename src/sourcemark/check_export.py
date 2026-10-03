"""Sanitized, versioned exports of citation/read-evidence checks.

The regular :mod:`sourcemark.check` report is intentionally detailed for a
local operator.  ``sourcemark/check/v1`` is the small boundary projection
that another tool may consume: it contains only a bounded state, fixed scalar
counts, and identities represented by hashes.  It never carries the report's
paths, quotes, URLs, raw tokens, or ledger data.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping

from .check import PASSING, Report
from .observe import Session

SCHEMA = "sourcemark/check/v1"
STATES = ("ok", "observed", "partial", "timed_out")
COUNT_FIELDS = ("total", "passing", "failing", "unknown", "observations", "timed_out")

# These verdicts mean that the checker could not look up the cited token.  The
# other non-passing verdicts are still represented by the bounded ``failing``
# count, without exposing their local details across the protocol boundary.
UNKNOWN_VERDICTS = frozenset({"unknown_token", "token_unchecked"})

_HASH = re.compile(r"^[0-9a-f]{64}$")
_TOP_LEVEL = frozenset({"schema", "state", "counts", "policy_sha256", "session_sha256"})
_COUNT_KEYS = frozenset(COUNT_FIELDS)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


_POLICY_SPEC = {
    "name": "sourcemark-check-export",
    "schema": SCHEMA,
    "states": STATES,
    "count_fields": COUNT_FIELDS,
    "passing_verdicts": tuple(sorted(PASSING)),
    "unknown_verdicts": tuple(sorted(UNKNOWN_VERDICTS)),
    "identity": "bounded-counts-and-hashed-identities-v1",
}
POLICY_SHA256 = _digest(_POLICY_SPEC)


def _session_identity(session: Session) -> str:
    """Hash session shape and evidence without returning any source material."""

    observations = []
    for observation in sorted(
        session.observations,
        key=lambda item: (
            str(item.path),
            int(item.line_start),
            tuple(item.line_numbers or ()),
            str(item.tool),
            bool(item.delegated),
        ),
    ):
        observations.append(
            {
                # Paths and line text are inputs to the identity digest only;
                # neither value crosses the export boundary.
                "path": _digest(str(observation.path)),
                "line_start": int(observation.line_start),
                "line_numbers": None if observation.line_numbers is None else list(observation.line_numbers),
                "line_count": len(observation.lines),
                "line_digest": _digest(observation.lines),
                "tool": str(observation.tool),
                "total_lines": observation.total_lines,
                "delegated": bool(observation.delegated),
            }
        )
    identity = {
        "cwd": _digest(session.cwd) if session.cwd is not None else None,
        "last_cwd": _digest(session.last_cwd) if session.last_cwd is not None else None,
        "cwds": sorted(_digest(value) for value in session.cwds),
        "urls": sorted(_digest(value) for value in session.urls),
        "delegated_urls": sorted(_digest(value) for value in session.delegated_urls),
        "text_turns": list(session.text_turns),
        "observations": observations,
    }
    return _digest(identity)


def _counts(report: Report, session: Session, *, timed_out: bool) -> dict[str, int]:
    total = report.total
    passing = report.passing
    failing = total - passing
    unknown = sum(check.verdict in UNKNOWN_VERDICTS for check in report.checks)
    return {
        "total": total,
        "passing": passing,
        "failing": failing,
        "unknown": unknown,
        "observations": len(session.observations),
        "timed_out": int(timed_out),
    }


def _state(counts: Mapping[str, int], *, timed_out: bool) -> str:
    if timed_out:
        return "timed_out"
    if counts["total"] == 0:
        return "observed"
    if counts["failing"] == 0:
        return "ok"
    return "partial"


def validate_check_export_v1(value: Any) -> dict[str, Any]:
    """Validate and return a ``sourcemark/check/v1`` envelope.

    This is deliberately strict so a future consumer can fail closed on a
    malformed or expanded payload rather than silently accepting a new field
    or an unbounded verdict.  Error messages describe the shape only and do
    not echo source paths, quotes, URLs, tokens, or ledger values.
    """

    if not isinstance(value, dict):
        raise ValueError("check export must be an object")
    if set(value) != _TOP_LEVEL:
        raise ValueError("check export has an unexpected field set")
    if value.get("schema") != SCHEMA:
        raise ValueError("check export has an unsupported schema")
    state = value.get("state")
    if state not in STATES:
        raise ValueError("check export has an unsupported state")
    if not isinstance(value.get("counts"), dict) or set(value["counts"]) != _COUNT_KEYS:
        raise ValueError("check export has an invalid count set")
    counts = value["counts"]
    for name in COUNT_FIELDS:
        count = counts[name]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("check export counts must be non-negative integers")
    if counts["timed_out"] not in (0, 1):
        raise ValueError("check export timeout count must be zero or one")
    if counts["passing"] + counts["failing"] != counts["total"]:
        raise ValueError("check export counts do not reconcile")
    if counts["unknown"] > counts["failing"]:
        raise ValueError("check export unknown count exceeds failing count")
    if state == "ok" and (counts["total"] == 0 or counts["failing"] != 0 or counts["timed_out"] != 0):
        raise ValueError("ok export state does not match counts")
    if state == "observed" and (counts["total"] != 0 or counts["timed_out"] != 0):
        raise ValueError("observed export state does not match counts")
    if state == "partial" and (counts["total"] == 0 or counts["failing"] == 0 or counts["timed_out"] != 0):
        raise ValueError("partial export state does not match counts")
    if state == "timed_out" and counts["timed_out"] != 1:
        raise ValueError("timed_out export state does not match counts")
    for name in ("policy_sha256", "session_sha256"):
        if not isinstance(value.get(name), str) or not _HASH.fullmatch(value[name]):
            raise ValueError("check export identity must be a sha256 digest")
    return value


def export_check_v1(report: Report, session: Session, *, timed_out: bool = False) -> dict[str, Any]:
    """Build a sanitized ``sourcemark/check/v1`` envelope from a local report."""

    if not isinstance(report, Report) or not isinstance(session, Session):
        raise TypeError("check export requires a Report and Session")
    if not isinstance(timed_out, bool):
        raise TypeError("timed_out must be a boolean")
    counts = _counts(report, session, timed_out=timed_out)
    value = {
        "schema": SCHEMA,
        "state": _state(counts, timed_out=timed_out),
        "counts": counts,
        "policy_sha256": POLICY_SHA256,
        "session_sha256": _session_identity(session),
    }
    return validate_check_export_v1(value)


__all__ = [
    "COUNT_FIELDS",
    "POLICY_SHA256",
    "SCHEMA",
    "STATES",
    "UNKNOWN_VERDICTS",
    "export_check_v1",
    "validate_check_export_v1",
]
