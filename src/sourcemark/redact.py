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
    # key_material is checked separately (linear): see _key_material_spans.
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
    ("url_credentials", re.compile(r"\b[a-z][a-z0-9+.\-]*://[^/\s:@]*:[^/\s@]{3,}@")),
    ("auth_header", re.compile(r"(?i)\b(?:authorization|proxy-authorization)\s*[:=]\s*[\"']?(?:bearer|basic|token|bot)\s+[A-Za-z0-9._~+/=\-]{8,}")),
    ("bearer_token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=\-]{20,}")),
    # Env files and shells: DB_PASSWORD=..., export GITHUB_TOKEN=..., "- POSTGRES_PASSWORD=..." in
    # compose lists, and inline "PGPASSWORD=... psql" after && or ;.
    ("env_secret", re.compile(
        rf"(?m)(?:^\s*(?:-\s*)?[\"']?|[;&|]\s*|\s)(?:export\s+)?(?<![A-Z0-9_])(?=[A-Z0-9_]{{0,60}}?(?:PASSWORD|PASSWD|PWD|SECRET|TOKEN|API_?KEY|ACCESS_?KEY|PRIVATE_?KEY|CREDENTIALS?|AUTH_?TOKEN))[A-Z0-9_]{{1,120}}+\s*=\s*{_NOT_NUMBER}[\"']?(?!\$)[^\s\"']{{4,}}"
    )),
    # .properties / .ini / my.cnf: "spring.datasource.password=hunter22" (whole-line value, no call or index).
    ("kv_secret", re.compile(
        rf"(?im)^\s*(?=[\w.\-]{{0,80}}?{_SECRET_NAME})[\w.\-]{{1,160}}+\s*[=:]\s*{_NOT_NUMBER}(?![\"'$({{\[<#&*|>])[^\s()\[\]{{}};,#]{{4,}}\s*(?:#.*)?$"
    )),
    # Connection strings: "Server=db;User Id=app;Password=hunter22;" (a ; on one side, so
    # "token = tokens[0]" in code is not one)
    ("connstring_secret", re.compile(r"(?i)(?:;\s*(?:password|pwd|api[_-]?key|account[_-]?key|shared[_-]?access[_-]?key|client[_-]?secret|secret|token)\s*=\s*[^;\s]{3,}"
        r"|(?:^|\s)(?:password|pwd|api[_-]?key|account[_-]?key|shared[_-]?access[_-]?key|client[_-]?secret|secret|token)\s*=\s*[^;\s]{3,};)")),
    ("basic_auth_cli", re.compile(r"(?:^|\s)(?:-u|--user)\s+[^\s:@]+:[^\s@]{3,}")),
    ("api_key_header", re.compile(r"(?i)\b(?:x-api-key|api-key|x-auth-token|x-access-token|private-token|x-api-token|apikey)\s*:\s*[A-Za-z0-9._~+/=\-]{6,}")),
    ("mysql_password_flag", re.compile(r"(?i)\bmysql(?:dump|admin)?\b[^\n]*?\s-p(?!assword)[^\s\-$][^\s]{3,}")),
    # Secrets in URL query strings: ?api_key=..., &token=..., &sig=...
    ("url_secret_param", re.compile(
        r"(?i)[?&](?:api[_-]?key|apikey|key|token|access[_-]?token|refresh[_-]?token|id[_-]?token|client[_-]?secret|"
        r"auth|sig|signature|secret|password|pwd|x-amz-security-token|x-amz-signature|x-amz-credential|"
        r"x-goog-signature|x-goog-credential)=[^&\s#]{6,}"
    )),
    ("telegram_bot_token", re.compile(r"\bbot\d{6,}:[A-Za-z0-9_\-]{30,}")),
    ("netrc_password", re.compile(r"(?im)\b(?:machine|login)\s+\S+\s+(?:login\s+\S+\s+)?password\s+\S{4,}|^\s*password\s+(?=\S*[^A-Za-z\s])\S{6,}\s*$")),
    ("npmrc_token", re.compile(r"(?:_authToken|_auth|_password)\s*=\s*\S{6,}")),
    ("docker_auth", re.compile(r"\"auth\"\s*:\s*\"[A-Za-z0-9+/=]{12,}\"")),
    ("cli_secret", re.compile(r"(?i)--(?:password|passwd|token|api-key|secret)(?:=|\s+)(?![-$])[^\s\"']{4,}")),
    # name = "literal" in code / config (quoted value only, so expressions are not flagged).
    ("assigned_secret", re.compile(
        rf"(?i)(?<![A-Za-z0-9_\-])(?=[A-Za-z0-9_\-]{{0,60}}?{_SECRET_NAME})[A-Za-z0-9_\-]{{1,120}}+\s*[:=]\s*[\"'](?![\"'])[^\"'\s]{{4,}}[\"']"
    )),
    # YAML: "password: hunter22" (unquoted scalar alone on the line, not a number, not a reference).
    ("yaml_secret", re.compile(
        rf"(?im)^\s*(?=[A-Za-z0-9_\-]{{0,60}}?{_SECRET_NAME})[A-Za-z0-9_\-]{{1,120}}+\s*:\s+{_NOT_NUMBER}(?![\"'#&*{{\[|>$])[^\s#]{{4,}}\s*$"
    )),
    # "password": "value" in JSON / dict literals.
    ("quoted_secret", re.compile(rf"(?i)[\"'](?=[A-Za-z0-9_\-]{{0,60}}?{_SECRET_NAME})[A-Za-z0-9_\-]{{1,120}}+[\"']\s*[:=]\s*[\"'][^\"'\s]{{4,}}[\"']")),
    # host:port:database:user:password (.pgpass)
    ("pgpass_line", re.compile(r"(?m)^[^\s:#]+:\d+:[^\s:]+:[^\s:]+:\S{4,}$")),
]


# Patterns keyed on a secret-ish NAME are also run only in windows around a name hit, and their
# identifier part is a bounded lookahead plus a possessive match: an unbounded [A-Za-z0-9_]* prefix
# rescans from every start position (seconds on a 20 KB line of identifiers).
_NAMED = {"env_secret", "kv_secret", "assigned_secret", "yaml_secret", "quoted_secret"}
_NAME_HIT = re.compile(_SECRET_NAME + r"|PASSWORD|PASSWD|PWD|SECRET|TOKEN|API_?KEY|ACCESS_?KEY|CREDENTIAL", re.I)
_WINDOW = 400
_KEY_RUN = re.compile(r"[A-Za-z0-9+/]{60,}={0,2}")


def _windows(text: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for m in _NAME_HIT.finditer(text):
        lo = max(text.rfind("\n", 0, m.start()) + 1, m.start() - _WINDOW)
        nl = text.find("\n", m.end())
        hi = min(len(text) if nl < 0 else nl, m.end() + _WINDOW)
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def _key_material_spans(text: str) -> list[tuple[str, int, int]]:
    """PEM/base64 key bodies: a long run mixing upper case, lower case and digits (hex hashes are
    lower-case only). Found as plain runs, then tested, so the scan stays linear."""
    out = []
    for m in _KEY_RUN.finditer(text):
        run = m.group(0)
        if any(c.isupper() for c in run) and any(c.islower() for c in run) and any(c.isdigit() for c in run):
            out.append(("key_material", m.start(), m.end()))
    return out


def find_secret_spans(text: str) -> list[tuple[str, int, int]]:
    """(pattern name, start, end) for every secret-shaped span in ``text``."""
    spans = _key_material_spans(text)
    windows = None
    for name, pat in _PATTERNS:
        if name in _NAMED:
            if windows is None:
                windows = _windows(text)
            seen: set[tuple[int, int]] = set()
            for lo, hi in windows:
                for m in pat.finditer(text, lo, hi):
                    if (m.start(), m.end()) not in seen:
                        seen.add((m.start(), m.end()))
                        spans.append((name, m.start(), m.end()))
        else:
            for m in pat.finditer(text):
                spans.append((name, m.start(), m.end()))
    return spans


def find_secrets(text: str) -> list[str]:
    """Return the names of secret patterns present in ``text`` (empty == clean)."""
    return sorted({name for name, _s, _e in find_secret_spans(text)})


def redact(text: str, placeholder: str = "[REDACTED]") -> tuple[str, list[str]]:
    """Replace secret-shaped spans; return (redacted_text, pattern_names)."""
    spans = sorted(find_secret_spans(text), key=lambda x: (x[1], -x[2]))
    if not spans:
        return text, []
    out, pos = [], 0
    for _name, a, b in spans:
        if b <= pos:
            continue
        out.append(text[pos:max(pos, a)])
        out.append(placeholder)
        pos = b
    out.append(text[pos:])
    return "".join(out), sorted({n for n, _a, _b in spans})


def redact_obj(value):
    """Redact every string inside a JSON-like value (dicts, lists) before it is stored."""
    if isinstance(value, str):
        return redact(value)[0]
    if isinstance(value, dict):
        return {k: redact_obj(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_obj(v) for v in value]
    return value
