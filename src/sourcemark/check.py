"""Check an agent's citations against what the agent actually read.

Verdicts (per citation):

* ``verified``       every cited line was read in this session
* ``partial``        some cited lines were read
* ``unread_lines``   the file was read, but not the cited lines
* ``out_of_range``   the cited line is past the end of the file
* ``unread_file``    the file exists but was never read in this session
* ``nonexistent``    an absolute path that does not exist and was never read
* ``unresolved``     a relative path that matches nothing read and no file near the session
* ``file_only``      a path citation without line numbers, and the file was read
* ``quote_mismatch`` the lines were read, but the quoted code is not in them
* ``unknown_token``  an ``[sm:…]`` token that is not in the provided marks
* ``url_verified``   a link the session fetched or got back from a search
* ``url_unsourced``  a link that never appeared in any fetch/search this session
* ``delegated``      only a subagent read it; the citation is relayed, not first-hand
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from .cite import Citation, extract
from .observe import Session, normalize_url
from .textnorm import squash

PASSING = {"verified", "file_only", "url_verified"}
# Only code EXPRESSIONS are checked as quotes. A bare identifier or dotted name
# (`content_hash`, `c.text`) is usually a reference to a concept, not a quotation.
_CODEISH = re.compile(r"\s|[()=\[\]{}<>;,'\"+*]|->|=>")


@dataclass
class CitationCheck:
    raw: str
    verdict: str
    path: str | None = None
    resolved_path: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    lines_read: int = 0
    lines_cited: int = 0
    quotes_checked: int = 0
    quotes_found: int = 0
    changed_since_read: bool | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Report:
    checks: list[CitationCheck] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.checks)

    @property
    def passing(self) -> int:
        return sum(c.verdict in PASSING for c in self.checks)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self.checks:
            out[c.verdict] = out.get(c.verdict, 0) + 1
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "passing": self.passing,
            "counts": self.counts(),
            "checks": [c.to_dict() for c in self.checks],
        }


def _match_path(cited: str, session: Session, lines: set[int] | None = None) -> str | None:
    """Map a cited (possibly relative) path to a path the session observed."""
    cited_x = os.path.expanduser(cited)
    observed = session.paths()
    if os.path.isabs(cited_x):
        norm = os.path.normpath(cited_x)
        if norm in observed:
            return norm
        real = os.path.realpath(norm)
        for p in observed:
            if os.path.realpath(p) == real:
                return p
        return None
    rel = cited_x
    while rel.startswith("./"):
        rel = rel[2:]  # drop a leading "./" prefix (lstrip would also eat ".claude" -> "claude")
    rel = os.path.normpath(rel)
    if session.cwd:
        joined = os.path.normpath(os.path.join(session.cwd, cited_x))
        if joined in observed:
            return joined
    hits = [p for p in observed if p == rel or p.endswith(os.sep + rel)]
    if len(hits) == 1:
        return hits[0]
    if hits:
        # Ambiguous (e.g. a repo and its worktree copy): prefer the copy whose reads cover
        # the cited lines, then the shortest path.
        def coverage(p: str) -> int:
            if not lines:
                return 0
            cov: set[int] = set()
            for o in session.for_path(p):
                cov |= o.covered()
            return len(cov & lines)

        return sorted(hits, key=lambda p: (-coverage(p), len(p)))[0]
    return None


def _bases(session: Session) -> list[str]:
    """Directories a relative citation may be relative to: cwds, then ancestors of touched files."""
    bases: list[str] = []
    for d in [session.cwd, *sorted(session.cwds)]:
        if d and d not in bases:
            bases.append(d)
    for p in sorted(session.paths()):
        d = os.path.dirname(p)
        for _ in range(4):
            if d and d not in bases:
                bases.append(d)
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent
    return bases[:64]


def _on_disk(cited: str, session: Session) -> str | None:
    p = os.path.expanduser(cited)
    cands = [p] if os.path.isabs(p) else [os.path.join(b, p) for b in _bases(session)]
    for c in cands:
        if os.path.isfile(c):
            return os.path.normpath(c)
    return None


def _file_lines(path: str) -> list[str] | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read().replace("\r\n", "\n").split("\n")
    except OSError:
        return None


def _judge_lines(c: Citation, obs: list, out: CitationCheck) -> str:
    """Line-coverage verdict for a citation against a set of observations of one file."""
    covered: set[int] = set()
    total = None
    for o in obs:
        covered |= o.covered()
        total = o.total_lines or total
    cited = set(range(c.line_start, (c.line_end or c.line_start) + 1))
    out.lines_cited = len(cited)
    out.lines_read = len(cited & covered)
    if total is not None and c.line_start > total:
        out.detail = f"file had {total} lines when read"
        return "out_of_range"
    if out.lines_read == len(cited):
        return "verified"
    if out.lines_read:
        return "partial"
    near = sorted(covered, key=lambda n: abs(n - c.line_start))[:1]
    if near:
        out.detail = f"nearest line read: {near[0]}"
    return "unread_lines"


def check_citation(c: Citation, session: Session, *, now: bool = False) -> CitationCheck:
    out = CitationCheck(raw=c.raw, verdict="nonexistent", path=c.path, line_start=c.line_start, line_end=c.line_end)
    if c.token:
        out.verdict, out.detail = "unknown_token", "token lookup requires a mark ledger"
        return out
    if c.url:
        u = normalize_url(c.url)
        out.path = c.url
        def seen(urls: set[str]) -> bool:
            return u in urls or any(s.startswith(u + "/") or u.startswith(s + "/") for s in urls)

        if seen(session.urls):
            out.verdict = "url_verified"
        elif seen(session.delegated_urls):
            out.verdict, out.detail = "delegated", "only a subagent fetched this link"
        else:
            out.verdict = "url_unsourced"
            out.detail = "link never fetched or returned by a search in this session"
        return out
    assert c.path is not None
    want = set(range(c.line_start, (c.line_end or c.line_start) + 1)) if c.line_start else None
    hit = _match_path(c.path, session, want)
    if hit is None:
        disk = _on_disk(c.path, session)
        if not disk and not os.path.isabs(os.path.expanduser(c.path)):
            out.verdict = "unresolved"
            return out
        if disk:
            out.verdict, out.resolved_path = "unread_file", disk
            if c.line_start is not None:
                lines = _file_lines(disk)
                if lines is not None and c.line_start > len(lines):
                    out.verdict = "out_of_range"
                    out.detail = f"file has {len(lines)} lines and was never read"
        return out
    out.resolved_path = hit
    all_obs = session.for_path(hit)
    obs = [o for o in all_obs if not o.delegated]
    if not obs:
        # Only subagents touched this file: judge the citation on their evidence, then label it.
        sub = _judge_lines(c, all_obs, out) if c.line_start is not None else "file_only"
        if sub in ("verified", "partial", "file_only"):
            out.verdict, out.detail = "delegated", "only a subagent read this; the orchestrator relayed it"
        else:
            out.verdict = sub
        return out
    if c.line_start is None:
        out.verdict = "file_only"
        return out

    out.verdict = _judge_lines(c, obs, out)
    if out.verdict == "out_of_range":
        return out
    if out.verdict in ("unread_lines", "partial"):
        # The agent's own reads fall short; did a subagent read exactly these lines?
        deleg = [o for o in all_obs if o.delegated]
        if deleg:
            probe = CitationCheck(raw=c.raw, verdict="")
            if _judge_lines(c, obs + deleg, probe) == "verified":
                out.verdict, out.detail = "delegated", "only a subagent read these lines; the orchestrator relayed them"
                return out

    # Quote check: code-like quotes next to the citation must appear in the lines read.
    if out.verdict in ("verified", "partial") and c.claimed_quotes:
        window: list[str] = []
        for o in obs:
            for n in range(c.line_start - 2, (c.line_end or c.line_start) + 3):
                t = o.text_at(n)
                if t is not None:
                    window.append(t)
        hay = squash("\n".join(window))
        for q in c.claimed_quotes:
            if not _CODEISH.search(q):
                continue
            out.quotes_checked += 1
            # "foo(...)" / "a … b": the agent elided text; every remaining fragment must be present.
            parts = [squash(p) for p in re.split(r"\.\.\.|…", q)]
            parts = [p for p in parts if len(p) >= 3]
            if parts and all(p in hay for p in parts):
                out.quotes_found += 1
        if out.quotes_checked and not out.quotes_found:
            out.verdict = "quote_mismatch"
            out.detail = "quoted code not found in the cited lines as read"

    if now:
        lines = _file_lines(hit)
        if lines is not None:
            changed = False
            for o in obs:
                for n in cited:
                    seen = o.text_at(n)
                    if seen is not None and (n > len(lines) or lines[n - 1] != seen):
                        changed = True
            out.changed_since_read = changed
    return out


def check_text(text: str, session: Session, *, now: bool = False) -> Report:
    rep = Report()
    for c in extract(text, known_names=session.basenames()):
        rep.checks.append(check_citation(c, session, now=now))
    return rep


def check_many(texts: Iterable[str], session: Session, **kw: Any) -> Report:
    rep = Report()
    for t in texts:
        rep.checks.extend(check_text(t, session, **kw).checks)
    return rep
