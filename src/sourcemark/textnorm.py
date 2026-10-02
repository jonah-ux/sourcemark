"""Text normalization and fingerprint helpers.

Every fingerprint in Sourcemark is computed over a *normalized* form so that
cosmetic churn (CRLF vs LF, trailing spaces, Unicode composition) does not turn
an intact citation into an "edited" one. The raw text is always kept alongside.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

_WS_RUN = re.compile(r"[ \t\f\v]+")
_ANY_WS = re.compile(r"\s+")


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def normalize(text: str) -> str:
    """Canonical form used for exact fingerprints.

    NFC composition, LF newlines, trailing whitespace stripped per line,
    runs of horizontal whitespace collapsed to one space.
    """
    text = unicodedata.normalize("NFC", normalize_newlines(text))
    lines = [_WS_RUN.sub(" ", line).rstrip() for line in text.split("\n")]
    return "\n".join(lines)


def squash(text: str) -> str:
    """Whitespace-insensitive form used for loose comparison (reflowed text)."""
    return _ANY_WS.sub(" ", unicodedata.normalize("NFC", text)).strip()


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint(text: str) -> str:
    """Stable content fingerprint of a quote: sha256 over the normalized text."""
    return sha256_hex(normalize(text))


def loose_fingerprint(text: str) -> str:
    """Fingerprint that ignores all whitespace layout (catches reflow/reindent)."""
    return sha256_hex(squash(text))


def line_offsets(text: str) -> list[int]:
    """Character offset of the start of each line (0-based list, line N is index N-1)."""
    offsets = [0]
    for i, ch in enumerate(text):
        if ch == "\n":
            offsets.append(i + 1)
    return offsets


def offset_to_line(offsets: list[int], offset: int) -> int:
    """1-based line number containing character ``offset``."""
    lo, hi = 0, len(offsets) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if offsets[mid] <= offset:
            lo = mid
        else:
            hi = mid - 1
    return lo + 1
