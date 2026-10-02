"""The citation record ("mark") for text sources.

A mark stores several independent ways to find the cited text again:

* quote selector  - the exact text plus a little context before and after it
* position        - character offsets and 1-based line numbers when it was cited
* fingerprints    - sha256 of the normalized quote, a whitespace-blind variant,
                    and the whole document at citation time
* source          - path, plus git repository/commit/blob when available

Any one of them can be stale later; together they let the resolver say exactly
what happened to the citation.
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from .redact import find_secret_spans
from .textnorm import (
    fingerprint,
    line_offsets,
    loose_fingerprint,
    normalize_newlines,
    offset_to_line,
    sha256_hex,
)

SCHEMA = "sourcemark/mark/v1"
DEFAULT_CONTEXT = 32


@dataclass
class TextSource:
    path: str
    repo_root: str | None = None
    repo_remote: str | None = None
    repo_path: str | None = None  # path relative to repo_root
    git_commit: str | None = None
    git_blob: str | None = None
    machine: str | None = None


@dataclass
class Mark:
    kind: str
    source: dict[str, Any]
    quote: dict[str, Any]
    position: dict[str, Any]
    fingerprints: dict[str, str]
    observed_at: float
    observer: dict[str, Any] = field(default_factory=dict)
    redacted: list[str] = field(default_factory=list)
    schema: str = SCHEMA
    id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Mark":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})

    @property
    def token(self) -> str:
        """Short inline citation token, e.g. ``[sm:7f3a9c2b1d]``."""
        return f"[sm:{self.id[4:14]}]"


def compute_id(identity: dict[str, Any]) -> str:
    """Content-addressed id: same quote at same source + revision => same id."""
    blob = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(blob).digest()
    return "sm1_" + base64.b32encode(digest).decode().lower().rstrip("=")[:26]


def mark_text(
    doc: str,
    start: int,
    end: int,
    source: TextSource,
    *,
    context: int = DEFAULT_CONTEXT,
    observer: dict[str, Any] | None = None,
    observed_at: float | None = None,
) -> Mark:
    """Cite ``doc[start:end]``. Offsets are into the LF-normalized document."""
    doc = normalize_newlines(doc)
    if not (0 <= start < end <= len(doc)):
        raise ValueError(f"bad span {start}:{end} for document of length {len(doc)}")
    exact = doc[start:end]
    if not exact.strip():
        raise ValueError("cannot cite whitespace only: it matches everywhere")
    prefix = doc[max(0, start - context) : start]
    suffix = doc[end : end + context]
    offsets = line_offsets(doc)
    line_start = offset_to_line(offsets, start)
    line_end = offset_to_line(offsets, max(start, end - 1))

    # Scan a WIDER window than we store: a secret cut off at the context edge would no
    # longer match its own pattern, yet its stored fragment would still leak.
    lo, hi = max(0, start - context), min(len(doc), end + context)
    wide_lo, wide_hi = max(0, lo - 200), min(len(doc), hi + 200)
    secrets = sorted(
        {name for name, a, b in find_secret_spans(doc[wide_lo:wide_hi]) if a + wide_lo < hi and b + wide_lo > lo}
    )
    quote = {"exact": exact, "prefix": prefix, "suffix": suffix}
    if secrets:
        # Never persist secret-shaped text; keep only what is needed to verify.
        quote = {"exact": None, "prefix": None, "suffix": None, "length": len(exact)}

    fps = {
        "quote": fingerprint(exact),
        "quote_loose": loose_fingerprint(exact),
        "document": sha256_hex(doc),
    }
    position = {"start": start, "end": end, "line_start": line_start, "line_end": line_end}
    src = {k: v for k, v in asdict(source).items() if v is not None}
    identity = {
        "kind": "text",
        "path": src.get("repo_path") or src.get("path"),
        "remote": src.get("repo_remote"),
        "commit": src.get("git_commit"),
        "quote": fps["quote"],
        "line_start": line_start,
    }
    m = Mark(
        kind="text",
        source=src,
        quote=quote,
        position=position,
        fingerprints=fps,
        observed_at=observed_at if observed_at is not None else time.time(),
        observer=observer or {},
        redacted=secrets,
    )
    m.id = compute_id(identity)
    return m


def mark_lines(
    doc: str,
    line_start: int,
    line_end: int,
    source: TextSource,
    **kw: Any,
) -> Mark:
    """Cite whole lines ``line_start..line_end`` (1-based, inclusive)."""
    doc = normalize_newlines(doc)
    offsets = line_offsets(doc)
    if not (1 <= line_start <= line_end <= len(offsets)):
        raise ValueError(f"lines {line_start}-{line_end} outside 1-{len(offsets)}")
    start = offsets[line_start - 1]
    end = offsets[line_end] - 1 if line_end < len(offsets) else len(doc)
    if end <= start:  # empty line(s): include the newline so the span is non-empty
        end = min(len(doc), start + 1)
    return mark_text(doc, start, end, source, **kw)
