"""Collect git provenance for a file without ever recording credentials."""

from __future__ import annotations

import os
import re
import subprocess
from functools import lru_cache

from .anchor import TextSource

_CRED_IN_URL = re.compile(r"(?P<scheme>[a-z][a-z0-9+.\-]*://)[^/@\s]+@", re.I)


def clean_remote(url: str | None) -> str | None:
    """Drop any user:token@ part and normalize to host/owner/repo."""
    if not url:
        return None
    url = _CRED_IN_URL.sub(r"\g<scheme>", url.strip())
    m = re.match(r"^git@([^:]+):(.+?)(?:\.git)?$", url)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    m = re.match(r"^[a-z][a-z0-9+.\-]*://([^/]+)/(.+?)(?:\.git)?/?$", url, re.I)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    return url


GIT = os.environ.get("SOURCEMARK_GIT", "git")


def _git(cwd: str, *args: str) -> str | None:
    try:
        out = subprocess.run([GIT, "-C", cwd, *args], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


@lru_cache(maxsize=1024)
def _toplevel(directory: str) -> str | None:
    return _git(directory, "rev-parse", "--show-toplevel")


@lru_cache(maxsize=256)
def _repo_facts(root: str) -> tuple[str | None, str | None]:
    """(clean remote, HEAD) per repository, cached for the life of the process."""
    return clean_remote(_git(root, "remote", "get-url", "origin")), _git(root, "rev-parse", "HEAD")


def clear_cache() -> None:
    _toplevel.cache_clear()
    _repo_facts.cache_clear()


def source_for(path: str, machine: str | None = None, *, blob: bool = True) -> TextSource:
    """Build a :class:`TextSource` for ``path``, filling git fields when it is tracked."""
    path = os.path.abspath(path)
    src = TextSource(path=path, machine=machine)
    root = _toplevel(os.path.dirname(path) or ".")
    if not root:
        return src
    rel = os.path.relpath(os.path.realpath(path), os.path.realpath(root))
    if _git(root, "ls-files", "--error-unmatch", "--", rel) is None:
        return src  # inside a repo but untracked: path-only provenance
    src.repo_root = root
    src.repo_path = rel
    src.repo_remote, src.git_commit = _repo_facts(root)
    if blob and _git(root, "status", "--porcelain", "--", rel) == "":
        # clean: the committed blob is exactly what was read
        src.git_blob = _git(root, "rev-parse", f"HEAD:{rel}")
    return src
