"""Extract citations from agent-written text.

Recognized forms (paths may be relative or absolute):

* ``path/to/file.py:42`` and ``path/to/file.py:42-50``
* ``path/to/file.py#L42`` and ``path/to/file.py#L42-L50``
* markdown links ``[label](path/to/file.py#L42-L50)``
* inline tokens ``[sm:7f3a9c2b1d]`` that refer to a stored mark

A nearby inline-code span or quoted string is captured as the citation's
claimed quote, so the checker can confirm the quoted text really is there.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_EXT = r"[A-Za-z0-9]{1,8}"
_PATH = rf"(?:~?/|\.{{1,2}}/)?(?:[\w.@+\-]+/)*[\w.@+\-]+\.{_EXT}"
_LINES = r"(?P<l1>\d{1,6})(?:\s*[-–]\s*L?(?P<l2>\d{1,6}))?"

_MD_LINK = re.compile(rf"\[(?P<label>[^\]\n]{{0,200}})\]\((?P<path>{_PATH})(?:#L{_LINES})?\)")
_HASH_L = re.compile(rf"(?<![\w/.\-])(?P<path>{_PATH})#L{_LINES}")
_COLON = re.compile(rf"(?<![\w/.\-])(?P<path>{_PATH}):{_LINES}(?![\d])")
_TOKEN = re.compile(r"\[sm:(?P<tok>[a-z2-7]{6,26})\]")
_MD_URL = re.compile(r"\[(?P<label>[^\]\n]{0,200})\]\((?P<url>https?://[^)\s]+)\)")
_BARE_URL = re.compile(r"(?<![\(\[<\w])https?://[^\s)\]>`\"']+")
_CODE = re.compile(r"`([^`\n]{4,200})`")
_QUOTED = re.compile(r"[\"“]([^\"”\n]{8,200})[\"”]")


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


def _lines(m: re.Match[str]) -> tuple[int | None, int | None]:
    l1 = m.group("l1")
    if l1 is None:
        return None, None
    l2 = m.group("l2")
    a, b = int(l1), int(l2) if l2 else int(l1)
    return (a, b) if b >= a else (b, a)


def extract(
    text: str,
    quote_window: int = 160,
    known_names: set[str] | None = None,
    urls: bool = True,
) -> list[Citation]:
    """Return citations in order of appearance, without overlapping duplicates.

    ``known_names`` are basenames of files the session touched; extensionless
    names (``bin/deploy:12``) are only recognized when they appear there, which
    keeps things like ``localhost:8080`` from being mistaken for citations.
    """
    found: list[Citation] = []
    taken: list[tuple[int, int]] = []

    def free(s: int, e: int) -> bool:
        return all(e <= a or s >= b for a, b in taken)

    if urls:
        for rx in (_MD_URL, _BARE_URL):
            for m in rx.finditer(text):
                if free(m.start(), m.end()):
                    u = m.group("url") if rx is _MD_URL else m.group(0).rstrip(".,;:!?*_")
                    end = m.end() if rx is _MD_URL else m.start() + len(u)
                    found.append(Citation(text[m.start() : end], m.start(), end, url=u, form="url"))
                    taken.append((m.start(), end))

    for rx, form in ((_MD_LINK, "markdown"), (_HASH_L, "hash"), (_COLON, "colon")):
        for m in rx.finditer(text):
            if not free(m.start(), m.end()):
                continue
            ls, le = _lines(m)
            if form == "colon" and ls is None:
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
    for c in found:
        lo, hi = max(0, c.start - quote_window), min(len(text), c.end + quote_window)
        window = text[lo:hi]
        for rx in (_CODE, _QUOTED):
            for q in rx.findall(window):
                if c.path and (c.path in q or q in c.raw):
                    continue  # the code span is the citation itself
                if re.fullmatch(rf"{_PATH}(?::\d+(?:-\d+)?)?", q.strip()):
                    continue  # a bare path, not a quote
                c.claimed_quotes.append(q)
    return found
