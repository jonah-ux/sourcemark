"""Re-resolve a text mark against the current state of the world.

Search order (cheapest and most trustworthy first):

1. the original path
2. paths git says the file was renamed to since the cited commit
3. files under the search roots that contain a distinctive line of the quote

Each candidate is searched with :func:`sourcemark.locate.locate`. The result
says what happened: intact, shifted, moved, edited, or orphaned.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from .anchor import Mark
from .locate import Match, locate
from .textnorm import fingerprint, line_offsets, normalize_newlines, offset_to_line

STATUSES = ("intact", "shifted", "moved", "edited", "orphaned", "unverifiable")

# Below this length an exact hit in an unrelated file is too likely to be a coincidence
# unless the surrounding context also agrees.
SHORT_QUOTE = 24
MIN_CONTEXT_FOR_SHORT = 0.6
MAX_SEARCH_FILES = 50


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


def _read(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return normalize_newlines(fh.read())
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


def _verify_redacted(mark: Mark, path: str, doc: str) -> Match | None:
    s, e = mark.position["start"], mark.position["end"]
    if fingerprint(doc[s:e]) == mark.fingerprints["quote"]:
        return Match(s, e, 1.0, "position", 1.0)
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

    def attempt(path: str, allow_fuzzy: bool = True) -> bool:
        doc = _read(path)
        res.candidates_checked += 1
        if doc is None:
            return False
        if exact is None:
            m = _verify_redacted(mark, path, doc)
        else:
            m = locate(
                doc,
                exact,
                mark.quote.get("prefix") or "",
                mark.quote.get("suffix") or "",
                hint_start=mark.position.get("start") if path == original else None,
                fuzzy=allow_fuzzy,
            )
        if m is None:
            return False
        if path != original and exact is not None and len(exact) < SHORT_QUOTE:
            if m.context_score < MIN_CONTEXT_FOR_SHORT:
                res.notes.append(f"rejected weak short-quote hit in {path}")
                return False
        offs = line_offsets(doc)
        status, moved = _classify(mark, path, m, original, offset_to_line(offs, m.start))
        res.status, res.moved, res.path = status, moved, path
        res.start, res.end = m.start, m.end
        res.line_start = offset_to_line(offs, m.start)
        res.line_end = offset_to_line(offs, max(m.start, m.end - 1))
        res.similarity, res.context_score, res.method = m.similarity, m.context_score, m.method
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
        for c in search_roots(roots, exact, excl):
            if attempt(c, allow_fuzzy=False):
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
        for c in search_roots(roots, exact, {os.path.abspath(p) for p in seen}):
            if attempt(c):
                break
    return done()
