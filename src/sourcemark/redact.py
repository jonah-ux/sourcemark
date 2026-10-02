"""Detect secret-shaped text before an excerpt is persisted.

Deliberately conservative: a false positive only costs readability (the quote
is stored as a fingerprint instead of text); a false negative leaks a secret.
"""

from __future__ import annotations

import re

# Name fragments that mark an assignment's value as secret. Matched without \b so that
# DB_PASSWORD, AWS_SECRET_ACCESS_KEY, client_secret, x-api-key all qualify. Bare "auth"
# is excluded on purpose (author, authority); auth_token / auth_key are included.
_SECRET_NAME = (
    r"(?:pass(?:word|wd|phrase)|passwd|pwd|secret|token|api[_\-]?key|apikey|access[_\-]?key|"
    r"private[_\-]?key|credentials?|auth[_\-]?(?:token|key)|session[_\-]?key|signing[_\-]?key)"
)
_NOT_NUMBER = r"(?![0-9.]+(?:\s|$|[,;\]}]))"

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    # PEM/base64 key body: a long run mixing upper case, lower case and digits (hex hashes are lower-case only).
    ("key_material", re.compile(r"(?=[A-Za-z0-9+/]{60,})(?=[^\s]*[A-Z])(?=[^\s]*[a-z])(?=[^\s]*\d)[A-Za-z0-9+/]{60,}={0,2}")),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}\b|\bgithub_pat_[A-Za-z0-9_]{40,}\b")),
    ("gitlab_token", re.compile(r"\bglpat-[A-Za-z0-9_\-]{20,}\b")),
    ("npm_token", re.compile(r"\bnpm_[A-Za-z0-9]{36}\b")),
    ("huggingface_token", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")),
    ("supabase_token", re.compile(r"\bsbp_[a-f0-9]{40}\b")),
    ("sendgrid_key", re.compile(r"\bSG\.[A-Za-z0-9_\-]{16,}\.[A-Za-z0-9_\-]{16,}\b")),
    ("slack_token", re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}\b")),
    ("webhook_url", re.compile(r"https://(?:hooks\.slack\.com/services|discord(?:app)?\.com/api/webhooks)/[A-Za-z0-9/_\-]{10,}")),
    ("openai_like_key", re.compile(r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_\-]{20,}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("stripe_key", re.compile(r"\b(?:sk|rk)_(?:live|test)_[0-9A-Za-z]{16,}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")),
    ("url_credentials", re.compile(r"\b[a-z][a-z0-9+.\-]*://[^/\s:@]+:[^/\s@]{3,}@")),
    ("auth_header", re.compile(r"(?i)\b(?:authorization|proxy-authorization)\s*[:=]\s*[\"']?(?:bearer|basic|token|bot)\s+[A-Za-z0-9._~+/=\-]{8,}")),
    ("bearer_token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=\-]{20,}")),
    # Env files and shells: DB_PASSWORD=..., export GITHUB_TOKEN=..., --password=...
    ("env_secret", re.compile(
        rf"(?m)^\s*(?:export\s+)?[A-Z0-9_]*(?:PASSWORD|PASSWD|PWD|SECRET|TOKEN|API_?KEY|ACCESS_?KEY|PRIVATE_?KEY|CREDENTIALS?|AUTH_?TOKEN)[A-Z0-9_]*\s*=\s*{_NOT_NUMBER}[\"']?(?!\$)[^\s\"']{{4,}}"
    )),
    ("cli_secret", re.compile(r"(?i)--(?:password|passwd|token|api-key|secret)(?:=|\s+)(?![-$])[^\s\"']{4,}")),
    # name = "literal" in code / config (quoted value only, so expressions are not flagged).
    ("assigned_secret", re.compile(
        rf"(?i)(?<![A-Za-z0-9])[A-Za-z0-9_\-]*{_SECRET_NAME}[A-Za-z0-9_\-]*\s*[:=]\s*[\"'](?![\"'])[^\"'\s]{{4,}}[\"']"
    )),
    # YAML: "password: hunter22" (unquoted scalar alone on the line, not a number, not a reference).
    ("yaml_secret", re.compile(
        rf"(?im)^\s*[A-Za-z0-9_\-]*{_SECRET_NAME}[A-Za-z0-9_\-]*\s*:\s+{_NOT_NUMBER}(?![\"'#&*{{\[|>$])[^\s#]{{4,}}\s*$"
    )),
    # "password": "value" in JSON / dict literals.
    ("quoted_secret", re.compile(rf"(?i)[\"'][A-Za-z0-9_\-]*{_SECRET_NAME}[A-Za-z0-9_\-]*[\"']\s*[:=]\s*[\"'][^\"'\s]{{4,}}[\"']")),
    # host:port:database:user:password (.pgpass)
    ("pgpass_line", re.compile(r"(?m)^[^\s:#]+:\d+:[^\s:]+:[^\s:]+:\S{4,}$")),
]


def find_secret_spans(text: str) -> list[tuple[str, int, int]]:
    """(pattern name, start, end) for every secret-shaped span in ``text``."""
    spans = []
    for name, pat in _PATTERNS:
        for m in pat.finditer(text):
            spans.append((name, m.start(), m.end()))
    return spans


def find_secrets(text: str) -> list[str]:
    """Return the names of secret patterns present in ``text`` (empty == clean)."""
    return sorted({name for name, _s, _e in find_secret_spans(text)})


def redact(text: str, placeholder: str = "[REDACTED]") -> tuple[str, list[str]]:
    """Replace secret-shaped spans; return (redacted_text, pattern_names)."""
    found: list[str] = []
    for name, pat in _PATTERNS:
        if pat.search(text):
            found.append(name)
            text = pat.sub(placeholder, text)
    return text, found
