"""Extract citations from agent-written text.

Recognized forms (paths may be relative or absolute):

* ``path/to/file.py:42`` and ``path/to/file.py:42-50``
* ``path/to/file.py#L42`` and ``path/to/file.py#L42-L50``
* markdown links ``[label](path/to/file.py#L42-L50)``
* inline tokens ``[sm:7f3a9c2b1d]`` that refer to a stored mark
* Codex ``<oai-mem-citation>`` entries (``MEMORY.md:72-103|note=[...]``), rooted at its memories dir

A nearby inline-code span or quoted string is captured as the citation's
claimed quote, so the checker can confirm the quoted text really is there.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

_EXT = r"[A-Za-z][A-Za-z0-9]{0,7}"  # an extension starts with a letter: "3.12" or "1.26" are versions
# Directory names may carry framework route syntax: app/(admin)/users/[id]/page.tsx
_SEG = r"(?:[\w.@+\-]|\[[\w.\-]+\]|\([\w.\-]+\))+"
# The file is name.ext, or a dotfile (.gitignore, .env, .zshrc) that has no other extension.
# ...plus a timestamped backup suffix: config.yaml.bak-20260902-154450, app.py-2026-01-01.
_PATH = rf"(?:~?/|\.{{1,2}}/)?(?:{_SEG}/)*(?:[\w.@+\-]+\.{_EXT}(?:-\d[\w\-]*)?|\.[A-Za-z][\w\-]{{1,30}})"
_LINES = r"(?P<l1>\d{1,6})(?:\s*[-–]\s*L?(?P<l2>\d{1,6}))?"

_MD_LINK = re.compile(rf"\[(?P<label>[^\]\n]{{0,200}})\]\((?P<path>{_PATH})(?:#L{_LINES})?\)")
_HASH_L = re.compile(rf"(?<![\w/.\-])(?P<path>{_PATH})#L{_LINES}")
_COLON = re.compile(rf"(?<![\w/.\-])(?P<path>{_PATH}):{_LINES}(?![\d])")
# Inside backticks a path may contain spaces ("Application Support"); outside it cannot be told
# apart from the prose around it.
_TICK = re.compile(rf"`(?P<path>(?:~?/|\.{{1,2}}/)?(?:[^`\n/:]+/)+[^`\n/:]*\.{_EXT}):{_LINES}`")
def _elided(url: str) -> bool:
    """`https://x/ab…`, `https://x/ab...`, `https://x/.../b` name no page; GitHub `a...b` does."""
    u = url.rstrip(")]},;:!?*'\"")
    return "\u2026" in u or u.endswith("...") or "/.../" in u


def valid_url(url: str) -> bool:
    """A real host, not a template: ``https://<node>``, ``https://{host}``, ``https://$``."""
    m = re.match(r"https?://([^/?#]*)", url)
    host = m.group(1).rsplit("@", 1)[-1] if m else ""
    # Flat character classes only: nested quantifiers here backtrack exponentially on long junk.
    return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.\-]*(?::\d{1,5})?|\[[0-9A-Fa-f:.]+\](?::\d{1,5})?", host))


_TOKEN = re.compile(r"\[sm:(?P<tok>[a-z2-7]{6,26})\]")
_MD_URL = re.compile(r"\[(?P<label>[^\]\n]{0,200})\]\((?P<url>https?://(?:[^()\s]|\([^()\s]*\))+)\)")
_BARE_URL = re.compile(r"(?<![\w])https?://(?:[^\s()\]>`\"']|\([^\s()\]>`\"']*\))+")
# Every inline code span, paired left to right. The length limits are applied afterwards:
# a length-limited pattern re-syncs on the wrong backtick after a span it skips (a 1-char
# `x`, a 300-char path) and reads the prose between two spans as code.
_CODE = re.compile(r"`([^`\n]+)`")
_FENCE = re.compile(r"^(```|~~~)[^\n]*\n.*?^\1[ \t]*$", re.S | re.M)
_TLDS = {"com", "org", "net", "io", "dev", "ai", "co", "app", "edu", "gov", "us", "uk", "de", "info", "biz", "me", "xyz"}
_SENTENCE_BREAK = re.compile(r"(?:[.!?](?:\s|$))|\n")


@dataclass
class Citation:
    raw: str
    start: int  # offset in the source text
    end: int
    path: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    token: str | None = None
    url: str | None = None
    claimed_quotes: list[str] = field(default_factory=list)
    form: str = ""
    root: str | None = None  # a directory the relative path is first looked up in


def _lines(m: re.Match[str]) -> tuple[int | None, int | None]:
    l1 = m.group("l1")
    if l1 is None:
        return None, None
    l2 = m.group("l2")
    a, b = int(l1), int(l2) if l2 else int(l1)
    return (a, b) if b >= a else (b, a)


def extract(
    text: str,
    quote_window: int = 48,
    known_names: set[str] | None = None,
    urls: bool = True,
) -> list[Citation]:
    """Return citations in order of appearance, without overlapping duplicates.

    ``known_names`` are basenames of files the session touched; extensionless
    names (``bin/deploy:12``) are only recognized when they appear there, which
    keeps things like ``localhost:8080`` from being mistaken for citations.
    """
    found: list[Citation] = []
    # Fenced code blocks are examples, not claims: mask them out (offsets preserved).
    taken: list[tuple[int, int]] = [(m.start(), m.end()) for m in _FENCE.finditer(text)]

    def free(s: int, e: int) -> bool:
        return all(e <= a or s >= b for a, b in taken)

    def plausible(path: str, start: int) -> bool:
        if "/" not in path and path.rsplit(".", 1)[-1].lower() in _TLDS:
            return False  # "api.example.com:443" is a host and port
        before = text[max(0, start - 3) : start]
        if "\\" in before or re.search(r"[A-Za-z]:\\?$", before):
            return False  # tail of a Windows path like C:\x\y.py
        return True

    if urls:
        for rx in (_MD_URL, _BARE_URL):
            for m in rx.finditer(text):
                if free(m.start(), m.end()):
                    u = m.group("url") if rx is _MD_URL else m.group(0).rstrip(".,;:!?*")
                    end = m.end() if rx is _MD_URL else m.start() + len(u)
                    # In [svc.py:4](https://…#L4) the label is a citation too: leave it free.
                    taken.append((m.start("url") - 1 if rx is _MD_URL else m.start(), end))
                    if _elided(m.group(0)):
                        continue  # an elided link ("https://github.com/…") names no page
                    if not valid_url(u):
                        continue  # a placeholder ("https://<node>:8080") names no page
                    found.append(Citation(text[m.start() : end], m.start(), end, url=u, form="url"))

    for rx, form in ((_TICK, "tick"), (_MD_LINK, "markdown"), (_HASH_L, "hash"), (_COLON, "colon")):
        for m in rx.finditer(text):
            if not free(m.start(), m.end()):
                continue
            if form == "tick" and " " not in m.group("path"):
                continue  # no space: the colon form reads it the same way
            ls, le = _lines(m)
            if form == "colon" and ls is None:
                continue
            if not plausible(m.group("path"), m.start("path")):
                continue
            found.append(Citation(m.group(0), m.start(), m.end(), m.group("path"), ls, le, form=form))
            taken.append((m.start(), m.end()))
    names = {n for n in (known_names or set()) if "." not in n and len(n) >= 3}
    if names:
        alt = "|".join(sorted((re.escape(n) for n in names), key=len, reverse=True))
        rx = re.compile(rf"(?<![\w.\-])(?P<path>(?:~?/)?(?:[\w.@+\-]+/)*(?:{alt}))(?::|#L){_LINES}(?![\d])")
        for m in rx.finditer(text):
            if free(m.start(), m.end()):
                ls, le = _lines(m)
                found.append(Citation(m.group(0), m.start(), m.end(), m.group("path"), ls, le, form="known"))
                taken.append((m.start(), m.end()))
    for m in _TOKEN.finditer(text):
        found.append(Citation(m.group(0), m.start(), m.end(), token=m.group("tok"), form="token"))

    found.sort(key=lambda c: c.start)
    _root_memory_entries(text, found)
    _attach_quotes(text, found, quote_window)
    return found


# Codex appends the memories it used as entries ("MEMORY.md:72-103|note=[...]") inside an
# <oai-mem-citation> block; their paths are relative to the Codex memories directory, not to
# the working directory (which often holds an unrelated MEMORY.md). Some entries are relative to
# home instead ("notes/memory/x.md"), so the root is a first guess, not a rewrite.
_MEM_BLOCK = re.compile(r"<oai-mem-citation>.*?(?:</oai-mem-citation>|\Z)", re.S)


def _root_memory_entries(text: str, cites: list[Citation]) -> None:
    blocks = [(m.start(), m.end()) for m in _MEM_BLOCK.finditer(text)]
    if not blocks:
        return
    root = os.path.join(os.environ.get("CODEX_HOME") or "~/.codex", "memories")
    for c in cites:
        if (c.path and not c.path.startswith(("/", "~")) and text.startswith("|note=", c.end)
                and any(a <= c.start < b for a, b in blocks)):
            c.root = root

def _attach_quotes(text: str, cites: list[Citation], window: int) -> None:
    """Give each inline-code quote to exactly ONE citation: the nearest one in the same sentence,
    and only when it sits right next to it (within ``window`` characters).

    Dense answers put several citations in one paragraph; a fixed window around each
    citation would hand one claim's quote to its neighbours.
    """
    path_cites = [c for c in cites if c.path]
    if not path_cites:
        return
    for m in _CODE.finditer(text):
        q = m.group(1)
        if not 4 <= len(q) <= 200:
            continue
        owner = next(
            (c for c in path_cites if c.form == "markdown" and c.start < m.start() and m.end() <= c.start + c.raw.find("](")),
            None,
        )
        if owner is not None:
            owner.claimed_quotes.append(q)  # [`code`](file#L10): the label IS the claim about the lines
            continue
        if re.fullmatch(rf"{_PATH}(?::\d+(?:[-–]\d+)?)?", q.strip()) or any(q in c.raw for c in path_cites):
            continue  # the code span IS a citation, not a quote
        inner = sorted((c for c in path_cites if m.start(1) <= c.start and c.end <= m.end(1)), key=lambda c: -c.start)
        if inner:
            rest = q
            for c in inner:
                rest = rest[: c.start - m.start(1)] + rest[c.end - m.start(1) :]
            # `[a.py:3](/abs/a.py:3)`: a link written as code; `a.ts:282,411`: more line numbers
            # for the same file. Still only citations.
            if not re.sub(r"[\s\[\]()]|,\s*\d+(?:[-–]\d+)?", "", rest):
                continue
        best: Citation | None = None
        best_d = window + 1
        for c in path_cites:
            if c.start >= m.end():
                lo, hi = m.end(), c.start
            elif c.end <= m.start():
                lo, hi = c.end, m.start()
            else:
                continue
            if hi - lo > window or _SENTENCE_BREAK.search(text, lo, hi):
                continue
            # prefer a citation that FOLLOWS the quote ("`code` (path:line)") on ties
            d = (hi - lo) + (0 if c.start >= m.end() else 1)
            if d < best_d:
                best, best_d = c, d
        if best is not None:
            best.claimed_quotes.append(q)
