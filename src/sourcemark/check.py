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
* ``endpoint``       a local/private service address (localhost, LAN, CGNAT): not a source citation
* ``token_ok`` / ``token_stale`` / ``unknown_token`` / ``token_unchecked``  an ``[sm:…]`` token that
                     resolves / no longer resolves / is not in the ledger / could not be looked up
"""

from __future__ import annotations

import os
import difflib
import re
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Iterable

from .cite import Citation, extract
from .observe import Session, normalize_url
from .textnorm import squash

PASSING = {"verified", "file_only", "url_verified", "token_ok"}
# Only code EXPRESSIONS are checked as quotes. A bare identifier or dotted name
# (`content_hash`, `c.text`) is usually a reference to a concept, not a quotation.
_CODEISH = re.compile(r"\s|[()=\[\]{}<>;,'\"+*]|->|=>")
# A bare identifier is still checked, softly: it must appear SOMEWHERE in what was read of the
# file. A name that occurs nowhere in the file as read was not taken from it.
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")


def _near_miss(tok: str, text: str) -> str | None:
    """A token in ``text`` that ``tok`` misquotes: one or two characters added, dropped or
    changed at the end or inside a long name (``forbidden_patternsX`` for ``forbidden_patterns``)."""
    if len(tok) < 8:
        return None
    for cand in set(re.findall(r"[A-Za-z_][\w\-]{6,}", text)):
        if cand == tok or abs(len(cand) - len(tok)) > 2:
            continue
        flat = lambda x: re.sub(r"[-_.]", "", x).lower()  # noqa: E731
        if flat(cand).rstrip("s") == flat(tok).rstrip("s"):
            continue  # `pattern` / `patterns`, `tag_install` / `tag-install`: prose, not a misquote
        if difflib.SequenceMatcher(None, cand, tok, autojunk=False).ratio() >= 1 - 2.5 / max(len(cand), len(tok)) and cand[:4] == tok[:4]:
            return cand
    return None


def _near_miss_on_line(tok: str, window: list[str]) -> str | None:
    """A token on the cited lines that ``tok`` misquotes by a character or two, whatever its
    shape (``@classmethodX``, ``!MASTER-INDEXX.md``, ``p.settlement_revisionX``, a URL)."""
    if len(tok) < 6:
        return None
    flat = lambda x: re.sub(r"[-_.]", "", x).lower()  # noqa: E731
    for line in window:
        for cand in re.split(r"[\s`'\"(),;=<>\[\]{}]+", line):
            cand = cand.strip(":.*")
            if not cand or cand == tok or abs(len(cand) - len(tok)) > 2 or cand[:3] != tok[:3]:
                continue
            if flat(cand).rstrip("s") == flat(tok).rstrip("s"):
                continue  # plural or separator variant: prose, not a misquote
            if difflib.SequenceMatcher(None, cand, tok, autojunk=False).ratio() >= 1 - 2.5 / max(len(cand), len(tok)):
                return cand
    return None


def _soft_checkable(tok: str) -> bool:
    """A single name worth the soft check: letters, 4+ chars, and not a file name or path (a
    sentence may name another file that the cited one never spells out)."""
    if len(tok) < 4 or not re.search(r"[A-Za-z]", tok) or re.search(r"\s", tok):
        return False
    if "/" in tok or re.fullmatch(r"[\w.\-]+\.[A-Za-z][A-Za-z0-9]{0,7}", tok) and not _IDENT.fullmatch(tok):
        return False
    if "." in tok and _IDENT.fullmatch(tok) and re.search(r"\.[A-Za-z]{1,4}$", tok) and tok.count(".") == 1:
        return False  # `setup.py`-style file name
    return bool(re.fullmatch(r"[\w.\-]+", tok))


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
    def coverage(p: str) -> int:
        if not lines:
            return 0
        cov: set[int] = set()
        for o in session.for_path(p):
            cov |= o.covered()
        return len(cov & lines)

    # Where the session ended up comes first: that is where its answer was written.
    dirs = list(dict.fromkeys(d for d in [session.last_cwd, session.cwd, *sorted(session.cwds)] if d))
    joins = [os.path.normpath(os.path.join(d, cited_x)) for d in dirs]
    seen_joins = [j for j in joins if j in observed]
    if seen_joins:
        # Several working directories hold a read copy: prefer the one whose reads cover the lines.
        return max(seen_joins, key=lambda j: (coverage(j), -seen_joins.index(j)))
    home = [os.path.normpath(os.path.join(d, cited_x)) for d in {session.cwd, session.last_cwd} if d]
    if any(os.path.isfile(j) for j in home):
        # The path names a real file in the agent's own working tree (where it started or
        # ended up) that it never read: do not bind it to a same-named file somewhere else.
        # A directory merely visited along the way is not "its" tree.
        return None
    hits = [p for p in observed if p == rel or p.endswith(os.sep + rel)]
    if len(hits) == 1:
        return hits[0]
    if hits:
        # Ambiguous (e.g. a repo and its worktree copy): prefer the copy whose reads cover
        # the cited lines, then the shortest path.
        return sorted(hits, key=lambda p: (-coverage(p), len(p)))[0]
    return None


def _other_copies(cited: str, session: Session, chosen: str) -> list[str]:
    """Other read files a RELATIVE citation could name (``runtime/lib/x.py`` in two worktrees)."""
    cited_x = os.path.expanduser(cited)
    if os.path.isabs(cited_x):
        return []
    rel = os.path.normpath(cited_x[2:] if cited_x.startswith("./") else cited_x)
    return sorted(p for p in session.paths() if p != chosen and (p == rel or p.endswith(os.sep + rel)))


def _is_local_endpoint(url: str) -> bool:
    """A local/private SERVICE address. A document path on a private host is still a citation."""
    import ipaddress
    from urllib.parse import urlsplit

    try:
        parts = urlsplit(url)
        host = (parts.hostname or "").lower()
    except ValueError:
        return False
    last = parts.path.rstrip("/").rsplit("/", 1)[-1]
    if re.search(r"\.[A-Za-z0-9]{1,8}$", last):
        return False  # ".../q3/report.pdf" names a document, not an endpoint
    if host in ("localhost", "0.0.0.0") or host.endswith((".local", ".localhost", ".internal", ".lan")):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    cgnat = ipaddress.ip_network((0x64400000, 10))  # RFC 6598 shared address space (CGNAT / overlay VPNs)
    return ip.is_loopback or ip.is_private or ip.is_link_local or (ip.version == 4 and ip in cgnat)


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


def _in_order(hay: str, parts: list[str]) -> bool:
    """Every fragment present, in the order written."""
    pos = 0
    for p in parts:
        i = hay.find(p, pos)
        if i < 0:
            return False
        pos = i + len(p)
    return True


def _quotes_missing(c: Citation, obs: list, out: CitationCheck) -> bool:
    """True when a code-like quote attributed to ``c`` is not in the cited lines as read."""
    if not c.claimed_quotes or c.line_start is None:
        return False
    window: list[str] = []
    for o in obs:
        for n in range(c.line_start - 1, (c.line_end or c.line_start) + 2):
            t = o.text_at(n)
            if t is not None:
                window.append(t)
    if not window:
        return False  # the lines were seen but their text was not attributable (Bash-range)
    hay = squash("\n".join(window))
    read_all: str | None = None
    for q in c.claimed_quotes:
        if not _CODEISH.search(q):
            tok = q.strip().strip("'\"").rstrip(":")
            if squash(tok) not in hay and _near_miss_on_line(tok, window):
                if read_all is None:
                    texts = [t for o in obs for t in (o.lines or [])]
                    read_all = "" if any(t is None for t in texts) else "\n".join(texts)
                # Absent from the cited lines while they hold it with a character or two
                # changed: a misquote of those lines, whatever the token's shape. Unless the
                # file has it elsewhere: then it is a real name (`closedate` beside `closedDate`).
                if tok not in read_all:
                    out.quotes_checked += 1
                    continue
            if not _soft_checkable(tok):
                continue
            else:
                if read_all is None:
                    texts = [t for o in obs for t in (o.lines or [])]
                    # Only judge against complete text: lines known by number alone could hold it.
                    read_all = "" if any(t is None for t in texts) else "\n".join(texts)
                if not read_all or re.search(rf"(?<![A-Za-z0-9_]){re.escape(tok)}(?![A-Za-z0-9_])", read_all):
                    continue  # present, or not judgeable: a bare name is usually a mention
                near = _near_miss(tok, read_all)
                if near is None:
                    continue  # absent but nothing like it was read: a mention of something else
                out.quotes_checked += 1  # absent, while a near-identical name WAS read: a misquote
                continue
        out.quotes_checked += 1
        # "foo(...)" / "a … b": the agent elided text; every remaining fragment must be present.
        # "foo()" names the function, not an empty call; "**Gate 2**" bolds a prefix of the line.
        q = re.sub(r"(?<=\w)\(\)", "(...)", q)
        # "{id}" in "/crm/v6/Shop/{id}/Leads" stands for whatever placeholder the template has: a
        # bare name, not code (`{dest.relative_to(ROOT)}`), and not a near-miss of a brace
        # expression the lines do have (`{shop_idX}` for `{shop_id}` is a misquote).
        braces = re.findall(r"\{([^{}]*)\}", hay)
        holes = [h for h in re.findall(r"\{(\w{1,40})\}", q)
                 if h not in braces and not any(difflib.SequenceMatcher(None, h, b, autojunk=False).ratio() >= 0.8 for b in braces)]
        hole_rx = "|".join(re.escape("{" + h + "}") for h in holes)
        parts = [squash(p) for p in re.split(r"\.\.\.|…|\*\*" + ("|" + hole_rx if hole_rx else ""), q)]
        parts = [p for p in parts if len(p) >= 3]
        if parts and _in_order(hay, parts):
            out.quotes_found += 1
    if out.quotes_checked and out.quotes_found < out.quotes_checked:
        missing = out.quotes_checked - out.quotes_found
        out.detail = f"{missing} quoted expression(s) not found in the cited lines as read"
        return True
    return False


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
    if out.lines_read == len(cited):
        return "verified"  # seen; a later, shorter read of an edited file does not unsee it
    if total is not None and c.line_start > total:
        out.detail = f"file had {total} lines when read"
        return "out_of_range"
    if out.lines_read:
        return "partial"
    near = sorted(covered, key=lambda n: abs(n - c.line_start))[:1]
    if near:
        out.detail = f"nearest line read: {near[0]}"
    return "unread_lines"


def check_citation(c: Citation, session: Session, *, now: bool = False, ledger: Any = None) -> CitationCheck:
    if c.root and c.path:
        rooted = os.path.join(c.root, c.path[2:] if c.path.startswith("./") else c.path)
        if _match_path(rooted, session) or os.path.isfile(os.path.expanduser(rooted)):
            c = replace(c, path=rooted)
    out = CitationCheck(raw=c.raw, verdict="nonexistent", path=c.path, line_start=c.line_start, line_end=c.line_end)
    if c.token:
        if ledger is None:
            out.verdict, out.detail = "token_unchecked", "no ledger supplied to look the token up"
            return out
        try:
            mark = ledger.get_mark(c.token)
        except LookupError as e:  # ambiguous prefix: unknown, never an exception that skips the turn
            out.verdict, out.detail = "unknown_token", str(e)
            return out
        if mark is None:
            out.verdict, out.detail = "unknown_token", "no mark with this token in the ledger"
            return out
        out.path = mark.source.get("path") or mark.source.get("table")
        if mark.kind == "text":
            from .resolve import resolve

            res = resolve(mark, search=False)
            out.resolved_path, out.line_start, out.line_end = res.path, res.line_start, res.line_end
            out.verdict = "token_ok" if res.status in ("intact", "shifted", "moved") else "token_stale"
            out.detail = f"mark resolves {res.status}"
        else:
            out.verdict, out.detail = "token_ok", "database mark exists (resolve with a DSN to check drift)"
        return out
    if c.url:
        if _is_local_endpoint(c.url):
            out.path, out.verdict = c.url, "endpoint"
            out.detail = "local or private address; an endpoint, not a cited source"
            return out
        u = normalize_url(c.url)
        out.path = c.url
        def seen(urls: set[str]) -> bool:
            return u in urls  # exact (normalized) match only; a parent or child page is a different source

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
        if sub == "verified" and _quotes_missing(c, all_obs, out):
            out.verdict = "quote_mismatch"  # a relayed claim is checked as strictly as one's own
        elif sub in ("verified", "file_only"):
            out.verdict, out.detail = "delegated", "only a subagent read this; the orchestrator relayed it"
        else:
            out.verdict = sub  # partial / unread lines stay failing even when relayed
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
                if _quotes_missing(c, obs + deleg, out):
                    out.verdict = "quote_mismatch"
                else:
                    out.verdict, out.detail = "delegated", "only a subagent read these lines; the orchestrator relayed them"
                return out
        # A relative path can name several read copies (one repo in several worktrees), and the
        # working directory picked one whose reads miss these lines; the claim stands if another
        # copy has them. Only for a path with a directory part: a bare README.md or notes.md
        # names unrelated files, and without a quote nothing shows which one was meant.
        rel_dir = "/" in os.path.normpath(os.path.expanduser(c.path)).lstrip("./")
        for alt in _other_copies(c.path, session, hit) if rel_dir else []:
            alt_obs = [o for o in session.for_path(alt) if not o.delegated]
            probe = CitationCheck(raw=c.raw, verdict="")
            if alt_obs and _judge_lines(c, alt_obs, probe) == "verified":
                hit, obs, out.resolved_path, out.verdict = alt, alt_obs, alt, "verified"
                out.lines_read, out.detail = probe.lines_read, "matched another read copy of this path"
                break

    # Quote check: code-like quotes next to the citation must appear in the lines read.
    if out.verdict in ("verified", "partial") and _quotes_missing(c, obs, out):
        out.verdict = "quote_mismatch"
        # A relative path can name several copies the session read (a repo and its worktrees);
        # the claim stands if one of them has these lines with this text.
        for alt in _other_copies(c.path, session, hit):
            alt_obs = [o for o in session.for_path(alt) if not o.delegated]
            probe = CitationCheck(raw=c.raw, verdict="")
            if alt_obs and _judge_lines(c, alt_obs, probe) == "verified" and not _quotes_missing(c, alt_obs, probe):
                out.verdict, out.resolved_path = "verified", alt
                out.lines_read, out.quotes_checked, out.quotes_found = probe.lines_read, probe.quotes_checked, probe.quotes_found
                out.detail = "matched another read copy of this path"
                break

    if now:
        cited = set(range(c.line_start, (c.line_end or c.line_start) + 1))
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


class UnavailableLedger:
    """Stands in for a ledger that could not be opened: every token is unknown (failing),
    never silently unchecked."""

    def __init__(self, why: str):
        self.why = why

    def get_mark(self, ref: str) -> None:
        raise LookupError(f"ledger unavailable: {self.why}")


def check_text(text: str, session: Session, *, now: bool = False, ledger: Any = None) -> Report:
    rep = Report()
    for c in extract(text, known_names=session.basenames()):
        try:
            rep.checks.append(check_citation(c, session, now=now, ledger=ledger))
        except Exception as e:  # one bad citation must not drop the others from the check
            rep.checks.append(CitationCheck(raw=c.raw, verdict="unresolved", path=c.path, detail=f"check failed: {e}"))
    return rep


def check_many(texts: Iterable[str], session: Session, **kw: Any) -> Report:
    rep = Report()
    for t in texts:
        rep.checks.extend(check_text(t, session, **kw).checks)
    return rep
