"""Re-resolve a text mark against the current state of the world.

Search order (cheapest and most trustworthy first):

1. the original path
2. paths git says the file was renamed to since the cited commit
3. files under the search roots that contain a distinctive line of the quote

Each candidate is searched with :func:`sourcemark.locate.locate`. The result
says what happened: intact, shifted, moved, edited, or orphaned.
"""

from __future__ import annotations

import difflib
import os
import re
import shutil
import subprocess
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from .anchor import Mark
from .locate import DEFAULT_MIN_SIMILARITY, DISTINCTIVE_CHARS, EXACT_CONTEXT_MIN, Match, locate
from .textnorm import fingerprint, line_offsets, normalize_newlines, offset_to_line

STATUSES = ("intact", "shifted", "moved", "edited", "orphaned", "unverifiable")

# Below this length an exact hit in an unrelated file is too likely to be a coincidence
# unless the surrounding context also agrees.
SHORT_QUOTE = 24
MIN_CONTEXT_FOR_SHORT = 0.5
# Lines that recur across files (imports, shebangs, lone closers): a unique copy of one
# under the search roots still says nothing about where the citation went.
_BOILERPLATE = re.compile(
    r"^\s*(?:import\s|from\s+\S+\s+import\s|#include\b|#!|use\s|package\s|require\(|"
    r"export\s+\*|[}\])]+[;,]?\s*$|(?:end|pass|else:?|return;?)\s*$)"
)
# Comment lines: license headers and banners are copied into every file of a project.
_COMMENT = re.compile(r"^\s*(?:#|//|/\*|\*|--|<!--|;|%)")
OTHER_FILE_DISTINCTIVE = 48
MAX_SEARCH_FILES = 50
FUZZY_SEARCH_FILES = 5
IN_PLACE_CONTEXT = 0.8


@dataclass
class Resolution:
    mark_id: str
    status: str
    path: str | None = None
    start: int | None = None
    end: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    similarity: float = 0.0
    context_score: float = 0.0
    method: str = ""
    moved: bool = False
    candidates_checked: int = 0
    elapsed_ms: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Above this many lines (old + new) the line alignment is skipped: too slow for a tie-break.
HISTORY_ALIGN_MAX_LINES = 20000
# Uneven hunks are paired line by line only up to this many (old x new) comparisons.
HUNK_PAIR_MAX = 2500
HUNK_PAIR_MIN = 0.6


def _align(old: str, new: str) -> tuple[dict[int, int], dict[int, int], set[int]] | None:
    """Line alignment of two versions, the way git follows lines through a diff (0-based).

    Returns (kept: old->new for unchanged lines, edited: old->new for lines replaced one-for-one,
    descended: new lines that are an unchanged old line)."""
    a, b = old.split("\n"), new.split("\n")
    if len(a) + len(b) > HISTORY_ALIGN_MAX_LINES:
        return None
    kept: dict[int, int] = {}
    edited: dict[int, int] = {}
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            kept.update((i1 + k, j1 + k) for k in range(i2 - i1))
        elif tag == "replace" and i2 - i1 == j2 - j1:
            edited.update((i1 + k, j1 + k) for k in range(i2 - i1))
        elif tag == "replace" and (i2 - i1) * (j2 - j1) <= HUNK_PAIR_MAX:
            # Uneven hunk: pair each old line with its most similar later new line, in order.
            nxt = j1
            for i in range(i1, i2):
                best, best_j = HUNK_PAIR_MIN, None
                for j in range(nxt, j2):
                    r = difflib.SequenceMatcher(None, a[i], b[j], autojunk=False).ratio()
                    if r > best:
                        best, best_j = r, j
                if best_j is not None:
                    edited[i] = best_j
                    nxt = best_j + 1
    return kept, edited, set(kept.values())


def _follow_history(mark: Mark, doc: str, m: Match | None, old: str | None) -> tuple[Match | None, bool]:
    """Re-locate a duplicated quote through the diff from the marked version (``git_blob``).

    Identical copies cannot be told apart by their text, and often not by their context; the
    diff can. Returns (match, decided): decided=False leaves ``m`` to the text search."""
    exact = mark.quote.get("exact")
    line = mark.position.get("line_start")
    start = mark.position.get("start")
    if not exact or not line or old is None or start is None:
        return m, False
    old = normalize_newlines(old)
    if old[start : start + len(exact)] != exact:
        return m, False  # the blob is not the text that was marked
    aligned = _align(old, doc)
    if aligned is None:
        return m, False
    kept, edited, descended = aligned
    o_offs, offs = line_offsets(old), line_offsets(doc)
    first, last = line - 1, line - 1 + exact.count("\n")
    col = start - o_offs[first]
    rows = [kept.get(i, edited.get(i)) for i in range(first, last + 1)]
    if all(r is not None for r in rows) and rows == list(range(rows[0], rows[0] + len(rows))):
        at = offs[rows[0]] + col if rows[0] < len(offs) else None
        unchanged = all(i in kept for i in range(first, last + 1))  # an edit can keep a prefix
        if unchanged and at is not None and doc[at : at + len(exact)] == exact:
            if m is not None and at == m.start:
                return m, True
            return Match(at, at + len(exact), 1.0, "history", m.context_score if m is not None else 0.0), True
        if any(i in edited for i in range(first, last + 1)):
            # Edited in place: the new text of those lines, scored like a fuzzy hit.
            s0 = offs[rows[0]]
            s1 = offs[rows[-1] + 1] - 1 if rows[-1] + 1 < len(offs) else len(doc)
            text = doc[s0:s1]
            sim = difflib.SequenceMatcher(None, exact, text, autojunk=False).ratio()
            if sim >= DEFAULT_MIN_SIMILARITY:
                return Match(s0, s1, round(sim, 4), "history-edit", 0.0), True
    # Our lines did not survive. An exact hit that descends from another, unchanged old line is
    # that line's copy, not ours; anything else (a block moved within the file) stays as found.
    if m is not None and m.similarity >= 0.999:
        hit_line = offset_to_line(offs, m.start) - 1
        if hit_line in descended:
            return None, True
    return m, False


def _read(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return unicodedata.normalize("NFC", normalize_newlines(fh.read()))
    except (OSError, ValueError):
        return None


GIT = os.environ.get("SOURCEMARK_GIT", "git")


def _git(root: str, *args: str) -> str | None:
    try:
        out = subprocess.run(
            [GIT, "-C", root, *args], capture_output=True, text=True, timeout=20
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout if out.returncode == 0 else None


def git_renames(root: str, commit: str, repo_path: str) -> list[str]:
    """Paths (relative to root) that ``repo_path`` was renamed/copied to since ``commit``."""
    out = _git(root, "diff", "-M30%", "-C", "--name-status", commit, "--")
    if out is None:
        return []
    found: list[str] = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) == 3 and parts[0][:1] in ("R", "C") and parts[1] == repo_path:
            found.append(parts[2])
    # Uncommitted renames show up in the working tree status.
    status = _git(root, "status", "--porcelain=v1", "-M")
    for line in (status or "").splitlines():
        if " -> " in line:
            old, new = line[3:].split(" -> ", 1)
            if old.strip('"') == repo_path:
                found.append(new.strip('"'))
    return found


def _distinctive_lines(exact: str, n: int = 3) -> list[str]:
    lines = [ln.strip() for ln in exact.split("\n")]
    lines = [ln for ln in lines if len(ln) >= 12]
    lines.sort(key=len, reverse=True)
    return lines[:n]


def _boilerplate(exact: str) -> bool:
    lines = [ln for ln in exact.split("\n") if ln.strip()]
    if not lines:
        return False
    if all(_BOILERPLATE.match(ln) for ln in lines):
        return True
    return len(lines) >= 2 and all(_BOILERPLATE.match(ln) or _COMMENT.match(ln) for ln in lines)


def search_roots(roots: Iterable[str], exact: str, exclude: set[str]) -> list[str]:
    """Files under ``roots`` containing distinctive lines of ``exact``, best first."""
    needles = _distinctive_lines(exact)
    if not needles:
        return []
    roots = [r for r in roots if os.path.isdir(r)]
    if not roots:
        return []
    hits: dict[str, int] = {}
    rg = shutil.which("rg")
    for needle in needles:
        if rg:
            cmd = [rg, "-l", "-F", "--no-messages", "--max-filesize", "5M", "-e", needle, *roots]
        else:
            cmd = ["grep", "-rlF", "--", needle, *roots]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
        except (OSError, subprocess.TimeoutExpired):
            continue
        for p in out.splitlines():
            p = os.path.abspath(p)
            if p not in exclude:
                hits[p] = hits.get(p, 0) + 1
    ranked = sorted(hits, key=lambda p: -hits[p])
    return ranked[:MAX_SEARCH_FILES]


def _ctx_worst(doc: str, m: Match, mark: Mark) -> float:
    from .locate import _context_score

    return _context_score(doc, m.start, m.end, mark.quote.get("prefix") or "", mark.quote.get("suffix") or "", worst=True)


def _classify(mark: Mark, path: str, m: Match, original_path: str, line_start: int) -> tuple[str, bool]:
    """Status is line-based: humans and git both count lines, not characters."""
    moved = os.path.abspath(path) != os.path.abspath(original_path)
    if m.similarity < 0.999:
        return "edited", moved
    if moved:
        return "moved", True
    if line_start == mark.position.get("line_start"):
        return "intact", False
    return "shifted", False


def _verify_redacted(mark: Mark, path: str, doc: str, old: str | None = None) -> Match | None:
    """A redacted mark stores no text, only its hash, length and column. Verify it in place, or
    find the one line where the same-length span at the same column has the same hash. With the
    marked version (``old``), identical copies are told apart through the diff."""
    s, e = mark.position["start"], mark.position["end"]
    fp = mark.fingerprints["quote"]
    in_place = fingerprint(doc[s:e]) == fp
    col, n = mark.position.get("column"), e - s
    if col is None:
        return Match(s, e, 1.0, "position", 1.0) if in_place else None  # marked before columns were recorded
    offs = line_offsets(doc)
    hits = [o + col for o in offs if o + col + n <= len(doc) and fingerprint(doc[o + col : o + col + n]) == fp]
    if len(hits) > 1 and old is not None and mark.position.get("line_start"):
        aligned = _align(normalize_newlines(old), doc)
        target = aligned[0].get(mark.position["line_start"] - 1) if aligned else None
        if target is not None and target < len(offs) and offs[target] + col in hits:
            hits = [offs[target] + col]
    if in_place and (len(hits) != 1 or hits[0] == s):
        return Match(s, e, 1.0, "position", 1.0)
    if len(hits) == 1:
        return Match(hits[0], hits[0] + n, 1.0, "redacted-hash", 0.0)
    return None


def resolve(
    mark: Mark,
    roots: Iterable[str] = (),
    *,
    search: bool = True,
    path_map: dict[str, str] | None = None,
) -> Resolution:
    """Resolve ``mark``. ``path_map`` lets callers relocate a whole tree (e.g. a clone)."""
    t0 = time.perf_counter()
    src = mark.source
    original = src.get("path", "")
    if path_map:
        for old, new in path_map.items():
            if original.startswith(old):
                original = new + original[len(old) :]
    res = Resolution(mark_id=mark.id, status="orphaned")
    exact = mark.quote.get("exact")

    repo_root = src.get("repo_root")
    if path_map and repo_root:
        for old, new in path_map.items():
            if repo_root.startswith(old):
                repo_root = new + repo_root[len(old) :]

    def attempt(path: str, allow_fuzzy: bool = True, *, searched: bool = False, unique: bool = False, strict: bool = False) -> bool:
        doc = _read(path)
        res.candidates_checked += 1
        if doc is None:
            return False
        if exact is None:
            blob = src.get("git_blob") if path == original and repo_root else None
            m = _verify_redacted(mark, path, doc, _git(repo_root, "cat-file", "-p", blob) if blob else None)
        else:
            m = locate(
                doc,
                exact,
                mark.quote.get("prefix") or "",
                mark.quote.get("suffix") or "",
                hint_start=mark.position.get("start") if path == original else None,
                fuzzy=allow_fuzzy,
                unique_when_marked=mark.position.get("occurrences", 1) <= 1,
            )
        if (
            path == original
            and exact is not None
            and src.get("git_blob")
            and repo_root
            and (m is None or mark.position.get("occurrences", 1) > 1 or doc.count(exact) > 1)
        ):
            # Several identical copies: context may not tell them apart, but the diff from the
            # marked version does. Nothing found: the diff may show the line edited in place
            # while its surroundings changed too (which defeats the fuzzy search's context check).
            followed, decided = _follow_history(mark, doc, m, _git(repo_root, "cat-file", "-p", src["git_blob"]))
            if decided:
                if followed is not m:
                    res.notes.append("duplicated quote: followed its line through the diff from the marked blob")
                m = followed
        if m is None:
            return False
        if searched and exact is not None:
            # A file found by searching (not a git rename) holds the same citation only when
            # the quote is distinctive, the only copy under the roots, or its surroundings
            # moved with it. Boilerplate appears in many files.
            boiler = _boilerplate(exact)
            # strict: the original file still holds an edited version in place, so a copy
            # elsewhere must ALSO have moved with its surroundings; length alone is not enough.
            distinctive = not boiler and not strict and (
                len(exact.strip()) >= OTHER_FILE_DISTINCTIVE
                or exact.count("\n") >= 2
                or (unique and len(exact.strip()) >= DISTINCTIVE_CHARS)
            )
            worst = _ctx_worst(doc, m, mark)
            # Boilerplate needs its real surroundings to have moved with it, not a look-alike.
            if not distinctive and worst < (EXACT_CONTEXT_MIN if boiler else MIN_CONTEXT_FOR_SHORT):
                res.notes.append(f"rejected non-distinctive hit in {path}")
                return False
        offs = line_offsets(doc)
        status, moved = _classify(mark, path, m, original, offset_to_line(offs, m.start))
        res.status, res.moved, res.path = status, moved, path
        res.start, res.end = m.start, m.end
        res.line_start = offset_to_line(offs, m.start)
        res.line_end = offset_to_line(offs, max(m.start, m.end - 1))
        res.similarity, res.context_score, res.method = m.similarity, m.context_score, m.method
        return True

    def pick_best(cands: list[str], allow_fuzzy: bool, unique: bool, strict: bool = False) -> bool:
        """Try every candidate and keep the one whose surroundings agree most, not the first
        hit: an identical decoy elsewhere must not win over the real moved file."""
        nonlocal res
        best_r: Resolution | None = None
        for c in cands:
            if attempt(c, allow_fuzzy=allow_fuzzy, searched=True, unique=unique, strict=strict):
                if best_r is None or (res.similarity, res.context_score) > (best_r.similarity, best_r.context_score):
                    best_r = Resolution(**res.to_dict())
        if best_r is None:
            return False
        best_r.candidates_checked, best_r.notes = res.candidates_checked, res.notes
        res = best_r
        return True

    def done() -> Resolution:
        res.elapsed_ms = (time.perf_counter() - t0) * 1000
        return res

    seen: set[str] = {original}
    # The original path is checked first: an intact or shifted citation never pays for git.
    if attempt(original) and res.status in ("intact", "shifted"):
        return done()
    best = Resolution(**res.to_dict()) if res.path else None

    if repo_root and src.get("git_commit") and src.get("repo_path"):
        for rel in git_renames(repo_root, src["git_commit"], src["repo_path"]):
            cand = os.path.join(repo_root, rel)
            if cand in seen:
                continue
            seen.add(cand)
            res.notes.append(f"git rename -> {rel}")
            if attempt(cand) and (best is None or res.similarity > best.similarity):
                best = Resolution(**res.to_dict())
            if best is not None and best.similarity >= 0.999:
                break
    if best is not None and best.similarity >= 0.999:
        best.candidates_checked = res.candidates_checked
        best.notes = res.notes
        res = best
        return done()
    # Only an edited (fuzzy) match so far: an exact copy elsewhere is stronger evidence.
    if best is not None and exact is not None and search:
        excl = {os.path.abspath(p) for p in seen}
        found = search_roots(roots, exact, excl)
        # Strict only when the original's fuzzy hit looks like an in-place edit (its own
        # surroundings agree); a look-alike elsewhere in the original must not block a real move.
        orig_doc = _read(best.path) if best.path else None
        in_place = (
            orig_doc is not None
            and abs((best.line_start or 0) - (mark.position.get("line_start") or 0)) <= 1
            and _ctx_worst(orig_doc, Match(best.start, best.end, best.similarity, best.method, best.context_score), mark)
            >= IN_PLACE_CONTEXT
        )
        if pick_best(found, allow_fuzzy=False, unique=len(found) == 1, strict=in_place):
            res.notes.append("exact copy elsewhere preferred over fuzzy match in original")
            return done()
    if best is not None:
        best.candidates_checked = res.candidates_checked
        best.notes = res.notes
        res = best
        return done()
    res.status, res.path = "orphaned", None

    if exact is None:
        res.status = "unverifiable"
        res.notes.append("quote was redacted; only the original position can be verified")
    elif search:
        found = search_roots(roots, exact, {os.path.abspath(p) for p in seen})
        if not pick_best(found, allow_fuzzy=False, unique=len(found) == 1):
            # Fuzzy only on the best-ranked few: each fuzzy pass reads and scores a whole file.
            pick_best(found[:FUZZY_SEARCH_FILES], allow_fuzzy=True, unique=len(found) == 1)
    return done()
