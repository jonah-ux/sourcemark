"""Detect secret-shaped text before an excerpt is persisted.

Deliberately conservative: a false positive only costs readability (the quote
is stored as a fingerprint instead of text); a false negative leaks a secret.
"""

from __future__ import annotations

import re

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}\b|\bgithub_pat_[A-Za-z0-9_]{40,}\b")),
    ("slack_token", re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}\b")),
    ("openai_like_key", re.compile(r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_\-]{20,}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("stripe_key", re.compile(r"\b(?:sk|rk)_(?:live|test)_[0-9A-Za-z]{16,}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")),
    ("url_credentials", re.compile(r"\b[a-z][a-z0-9+.\-]*://[^/\s:@]+:[^/\s@]{3,}@")),
    (
        "assigned_secret",
        re.compile(
            r"(?i)\b(?:password|passwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token|private[_-]?key)"
            r"\s*[:=]\s*['\"]?[^\s'\"]{8,}"
        ),
    ),
]


def find_secrets(text: str) -> list[str]:
    """Return the names of secret patterns present in ``text`` (empty == clean)."""
    return [name for name, pat in _PATTERNS if pat.search(text)]


def redact(text: str, placeholder: str = "[REDACTED]") -> tuple[str, list[str]]:
    """Replace secret-shaped spans; return (redacted_text, pattern_names)."""
    found: list[str] = []
    for name, pat in _PATTERNS:
        if pat.search(text):
            found.append(name)
            text = pat.sub(placeholder, text)
    return text, found
