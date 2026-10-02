"""Record what an agent actually read during a session.

Observations come from tool calls: a Read result carries the file path, the
first line number and the raw text; Grep output carries ``path:line:text``;
shell commands such as ``cat``, ``sed -n 'A,Bp'``, ``head`` and ``tail``
print a known slice of a known file. Each observation keeps the exact lines
seen, so a citation can be checked against the text as it was when read,
even if the file changes later.
"""

from __future__ import annotations

import functools
import json
import os
import re
import shlex
import warnings
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator


@dataclass
class Observation:
    path: str
    line_start: int  # 1-based, first line seen
    lines: list[str]  # text of each line seen, in order (may be sparse via `line_numbers`)
    tool: str
    at: str | None = None
    line_numbers: list[int] | None = None  # set for sparse observations (grep hits)
    total_lines: int | None = None
    delegated: bool = False  # seen by a subagent, not by the agent itself

    def covered(self) -> set[int]:
        if self.line_numbers is not None:
            return set(self.line_numbers)
        return set(range(self.line_start, self.line_start + len(self.lines)))

    def text_at(self, line: int) -> str | None:
        if self.line_numbers is not None:
            try:
                return self.lines[self.line_numbers.index(line)]
            except ValueError:
                return None
        i = line - self.line_start
        return self.lines[i] if 0 <= i < len(self.lines) else None


_FILE_LEVEL_TOOLS = ("Bash-touch", "Bash-redirect", "Bash-output", "Bash-copy", "Bash-append")


@dataclass
class Session:
    observations: list[Observation] = field(default_factory=list)
    cwd: str | None = None
    urls: set[str] = field(default_factory=set)  # normalized URLs seen in any tool output or user message
    cwds: set[str] = field(default_factory=set)  # every working directory the session used
    last_cwd: str | None = None  # where the session ended up (where its final answer was written)
    text_turns: list[int] = field(default_factory=list)  # turn number of each assistant text, in order
    delegated_urls: set[str] = field(default_factory=set)  # URLs only a subagent saw

    def add(self, obs: Observation | None) -> None:
        if obs is not None and (obs.lines or obs.line_numbers is not None):
            # One canonical spelling per file: "~/x", "a/../x" and "/home/u/x" must match.
            p = os.path.expanduser(obs.path)
            if not os.path.isabs(p) and self.cwd:
                p = os.path.join(self.cwd, p)
            obs.path = os.path.normpath(p)
            self.observations.append(obs)

    def line_evidence(self, path: str) -> list[Observation]:
        """Observations that carry actual line text (not file-level touches)."""
        return [o for o in self.for_path(path) if o.lines and o.tool not in ("Bash-touch", "Bash-redirect")]

    def paths(self) -> set[str]:
        return {o.path for o in self.observations}

    def basenames(self) -> set[str]:
        # Only files actually read or written: a `which node` hit must not make "node:20" a citation.
        return {os.path.basename(o.path) for o in self.observations if o.lines and o.tool not in _FILE_LEVEL_TOOLS}

    def for_path(self, path: str) -> list[Observation]:
        return [o for o in self.observations if o.path == path]


# --- tool-result parsers -------------------------------------------------------

_NUMBERED = re.compile(r"^\s*(\d+)[\t→](.*)$")


def _strip_numbered(content: str, start: int) -> list[str]:
    """Read results may arrive as ``N\\ttext`` lines; return just the text."""
    lines = content.split("\n")
    if lines and all(_NUMBERED.match(x) for x in lines[: min(5, len(lines))] if x):
        out = []
        for x in lines:
            m = _NUMBERED.match(x)
            out.append(m.group(2) if m else x)
        return out
    return lines


def from_read(tool_input: dict[str, Any], result: Any, at: str | None = None) -> Observation | None:
    """Claude Code ``Read``: structured result has file.filePath/startLine/content."""
    f = result.get("file") if isinstance(result, dict) else None
    if isinstance(f, dict) and "content" in f:
        path = f.get("filePath") or tool_input.get("file_path")
        try:
            start = int(f.get("startLine") or 1)
        except (TypeError, ValueError):
            start = 1
        if not f["content"]:
            return None  # offset past the end: nothing was shown
        lines = f["content"].split("\n")
        if lines and lines[-1] == "" and len(lines) > 1:
            lines = lines[:-1]
        return Observation(path, start, lines, "Read", at, total_lines=f.get("totalLines"))
    if isinstance(result, str) and tool_input.get("file_path"):
        sample = [x for x in result.split("\n")[:5] if x]
        if not sample or not all(_NUMBERED.match(x) for x in sample):
            return None  # an error message or anything else that is not file content
        try:
            start = int(str(tool_input.get("offset") or 1).split(",")[0])
        except ValueError:
            start = 1
        return Observation(tool_input["file_path"], start, _strip_numbered(result, start), "Read", at)
    return None


_GREP_LINE = re.compile(r"^(?P<path>[^:\n]+?):(?P<line>\d+)[:\-](?P<text>.*)$")


def _pathlike(p: str) -> bool:
    p = p.strip()
    if not p or p.isdigit() or "  " in p or any(ch in p for ch in "<>|*?\t"):
        return False
    return _isfile(os.path.expanduser(p)) or bool(re.search(r"\.[A-Za-z0-9]{1,8}$", p)) or "/" in p


_GREP_BARE = re.compile(r"^(?P<line>\d+)[:\-](?P<text>.*)$")
_GREP_CONTEXT = re.compile(r"^(?P<path>[^\n]+?)-(?P<line>\d+)-(?P<text>.*)$")


def from_grep_text(
    stdout: str,
    cwd: str | None,
    tool: str,
    at: str | None = None,
    single_file: str | None = None,
    alt_base: str | None = None,
    roots: list[str] | None = None,
) -> list[Observation]:
    """``path:line:text`` hit lines (ripgrep/grep -n). Context lines use ``-`` and count too.

    When the search targeted exactly one file, grep and ripgrep omit the file name and print
    ``line:text``; those lines belong to ``single_file``.
    """
    by_path: dict[str, Observation] = {}

    def add(p: str, n: int, text: str) -> None:
        obs = by_path.setdefault(p, Observation(p, 0, [], tool, at, line_numbers=[]))
        if n not in obs.line_numbers:  # type: ignore[operator]
            obs.line_numbers.append(n)  # type: ignore[union-attr]
            obs.lines.append(text)

    lines = stdout.split("\n")
    if single_file:
        # One file searched: hits are "N:text" / context "N-text". Anything shaped like
        # "path:N:..." is CONTENT of that file (e.g. a lint log), never a read of another file.
        for raw in lines:
            b = _GREP_BARE.match(raw)
            if b:
                add(single_file, int(b.group("line")), b.group("text"))
        return list(by_path.values())

    def absolute(p: str) -> str:
        p = p.strip()
        if os.path.isabs(p):
            return os.path.normpath(p)
        # Output paths may be relative to the search directory OR the session directory.
        cands = [os.path.normpath(os.path.join(b, p)) for b in (cwd, alt_base) if b]
        return next((c for c in cands if _isfile(c)), cands[0] if cands else p)

    def in_scope(p: str) -> bool:
        if not roots:
            return True
        rp = _realpath(p)
        return any(rp == r or rp.startswith(r.rstrip(os.sep) + os.sep) for r in roots)

    match_paths: set[str] = set()
    for raw in lines:
        m = _GREP_LINE.match(raw)
        if m and _pathlike(m.group("path")):
            match_paths.add(m.group("path").strip())
    for raw in lines:
        m = _GREP_LINE.match(raw)
        if m and _pathlike(m.group("path")):
            rel, n, text = m.group("path"), int(m.group("line")), m.group("text")
        else:
            # Context line (grep -C/-A/-B): "path-N-text", only for paths that also had a hit.
            c = _GREP_CONTEXT.match(raw)
            if not c or c.group("path").strip() not in match_paths:
                continue
            rel, n, text = c.group("path"), int(c.group("line")), c.group("text")
        p = absolute(rel)
        if in_scope(p):
            add(p, n, text)
    return list(by_path.values())


_SED = re.compile(r"^sed$")


_HEREDOC = re.compile(
    r"(?P<cmd>cat|tee)\s+(?P<append>-a\s+|>>\s*)?>?\s*(?P<path>[^\s<>|;&]+)\s*<<-?\s*['\"]?(?P<tag>\w+)['\"]?[^\n]*\n(?P<body>.*?)\n\s*(?P=tag)\b",
    re.S,
)
# The other common order: cat <<'EOF' > path
_HEREDOC_FIRST = re.compile(
    r"cat\s+<<-?\s*['\"]?(?P<tag>\w+)['\"]?\s*(?P<op>>>?)\s*(?P<path>[^\s<>|;&]+)[^\n]*\n(?P<body>.*?)\n\s*(?P=tag)\b",
    re.S,
)
_REDIRECT = re.compile(r"(?:^|[\s;&|])(?:>|>>|tee\s+(?:-a\s+)?)\s*(?P<path>[~/.\w][^\s<>|;&]*)")


_CD = re.compile(r"^\s*cd\s+(?P<dir>[^\s;&|]+)(?:\s+2>\s*/dev/null)?[ \t]*(?:&&|;|\n)")
_PATHISH = re.compile(r"(?<![\w@:/])(?:~?/|\.{1,2}/)?(?:[\w.\-]+/)*[\w.\-]+\.[A-Za-z][A-Za-z0-9]{0,7}(?![\w/])")


_ASSIGN = re.compile(r"(?:^|[;&|\n(]\s*)(?:export\s+)?(?P<name>[A-Za-z_]\w*)=(?P<val>\"[^\"$`]*\"|'[^']*'|[^\s;&|$`\"']+)(?=[\s;&|)]|$)")


def expand_assignments(command: str) -> str:
    """Substitute ``$NAME`` / ``${NAME}`` for plain ``NAME=value`` assignments made in the command."""
    vals: dict[str, str] = {}
    seen: dict[str, set[str]] = {}
    for m in _ASSIGN.finditer(command):
        raw = m.group("val")
        # The shell expands an unquoted leading ~ in an assignment (W=~/x), not a quoted one.
        val = os.path.expanduser(raw) if raw.startswith("~") else raw.strip("'\"")
        seen.setdefault(m.group("name"), set()).add(val)
    # A variable set to two values (F=a.py; ...; F=b.py) cannot be substituted by position here.
    vals = {k: next(iter(v)) for k, v in seen.items() if len(v) == 1}
    if not vals:
        return command
    return re.sub(
        r"\$(?:\{(\w+)\}|(\w+))",
        lambda m: vals.get(m.group(1) or m.group(2), m.group(0)),
        command,
    )


def effective_cwd(command: str, cwd: str | None) -> str | None:
    """Follow a leading ``cd DIR &&`` so relative paths in the command resolve correctly."""
    m = _CD.match(command)
    if not m:
        return cwd
    d = os.path.expanduser(m.group("dir").strip("'\""))
    if re.search(r"[$`]", d):
        return None  # `cd "$W"`: the agent's shell variable, unknown here
    return os.path.normpath(d if os.path.isabs(d) or not cwd else os.path.join(cwd, d))


def from_shell_touches(command: str, cwd: str | None, exclude: set[str], at: str | None = None) -> list[Observation]:
    """Existing files named anywhere in a command (incl. heredoc scripts): file-level evidence only."""
    out: list[Observation] = []
    for tok in set(_PATHISH.findall(command)):
        p = os.path.expanduser(tok)
        p = os.path.normpath(p if os.path.isabs(p) or not cwd else os.path.join(cwd, p))
        if p in exclude or not _isfile(p):
            continue
        exclude.add(p)
        out.append(Observation(p, 0, [], "Bash-touch", at, line_numbers=[]))
    return out


@functools.lru_cache(maxsize=1024)
def _root_present(root: str) -> bool:
    return os.path.isdir(root)


@functools.lru_cache(maxsize=65536)
def _realpath(p: str) -> str:
    parts = p.split(os.sep)
    if os.path.isabs(p) and len(parts) > 3 and not _root_present(os.sep.join(parts[:3])):
        return os.path.normpath(p)  # nothing here to resolve symlinks through
    return os.path.realpath(p)


@functools.lru_cache(maxsize=65536)
def _isfile(p: str) -> bool:
    """``os.path.isfile`` that skips paths whose top two directories are absent here.

    Transcripts from other machines are full of ``/home/<user>/...``; on macOS each stat of
    such a path goes through the automounter and costs milliseconds. One cached miss on
    ``/home/<user>`` answers them all.
    """
    p = os.path.expanduser(p)
    if os.path.isabs(p):
        parts = p.split(os.sep)
        if len(parts) > 3 and not _root_present(os.sep.join(parts[:3])):
            return False
    return os.path.isfile(p)


_ABS_IN_OUTPUT = re.compile(r"(?<![\w./~-])(?:~|/)[\w.~+@-]*(?:/[\w.~+@-]+)+")


def from_output_paths(stdout: str, exclude: set[str], at: str | None = None, limit: int = 400) -> list[Observation]:
    """Existing files a command printed by absolute path (shasum, ls -l, find): file-level only."""
    out: list[Observation] = []
    for tok in dict.fromkeys(_ABS_IN_OUTPUT.findall(stdout[:50_000])):
        if len(out) >= limit:
            break
        p = os.path.normpath(os.path.expanduser(tok))
        if p in exclude or not _isfile(p):
            continue
        exclude.add(p)
        out.append(Observation(p, 0, [], "Bash-output", at, line_numbers=[]))
    return out


def from_shell_writes(command: str, cwd: str | None, at: str | None = None) -> list[Observation]:
    """Files a shell command authored: heredoc bodies are known exactly; other redirects file-level."""
    out: list[Observation] = []
    seen: set[str] = set()

    def absolute(p: str) -> str:
        p = os.path.expanduser(p.strip("'\""))
        return os.path.normpath(p if os.path.isabs(p) or not cwd else os.path.join(cwd, p))

    heredocs = [(m.group("path"), bool(m.group("append")), m.group("body")) for m in _HEREDOC.finditer(command)]
    heredocs += [(m.group("path"), m.group("op") == ">>", m.group("body")) for m in _HEREDOC_FIRST.finditer(command)]
    for raw, append, body in heredocs:
        p = absolute(raw)
        if p in seen:
            continue
        seen.add(p)
        if append:
            # Appended after an unknown number of existing lines: the body is not lines 1..N.
            out.append(Observation(p, 0, [""], "Bash-append", at, line_numbers=[]))
        else:
            out.append(Observation(p, 1, body.split("\n"), "Bash-write", at))
    for m in _REDIRECT.finditer(command):
        raw = m.group("path")
        if raw.startswith(("/dev/", "&")) or raw in ("-",):
            continue
        p = absolute(raw)
        if p not in seen:
            seen.add(p)
            # Content unknown: record a file-level observation with no lines.
            out.append(Observation(p, 0, [""], "Bash-redirect", at, line_numbers=[]))
    for seg in _split_unquoted(command, newlines=True):
        if not _COPY.match(seg):
            continue  # cheap prefix test first: shlex on every segment dominates large sessions
        try:
            argv = shlex.split(seg)
        except ValueError:
            continue
        if not argv or os.path.basename(argv[0]) not in ("cp", "mv", "install", "ln"):
            continue
        ops = [a for a in argv[1:] if not a.startswith("-")]
        if len(ops) < 2 or "$" in ops[-1]:
            continue
        dest = absolute(ops[-1])
        for src in (x for o in ops[:-1] for x in _braces(o)):
            p = os.path.join(dest, os.path.basename(src.rstrip("/"))) if os.path.isdir(dest) else dest
            if p not in seen and _isfile(p):
                seen.add(p)
                out.append(Observation(p, 0, [""], "Bash-copy", at, line_numbers=[]))  # file-level only
    return out


_COPY = re.compile(r"\s*(?:\S*/)?(?:cp|mv|install|ln)\s")


def _braces(word: str) -> list[str]:
    """One level of shell brace expansion: ``d/{a,b}.txt`` -> ``d/a.txt d/b.txt``."""
    m = re.search(r"\{([^{}]*,[^{}]*)\}", word)
    if not m:
        return [word]
    return [word[: m.start()] + part + word[m.end() :] for part in m.group(1).split(",")]


_GH_REF = re.compile(r"\bgh\s+(?P<kind>pr|issue)\s+(?:view|merge|checks|diff|comment|close|edit|ready|review|reopen)\s+(?P<num>\d+)\b")
_GH_REPO = re.compile(r"(?:-R|--repo)[=\s]+(?:https://github\.com/)?(?P<repo>[\w.\-]+/[\w.\-]+)")


def gh_refs(command: str, stdout: str = "") -> set[str]:
    """``gh pr view 12 --repo o/r`` that ran and printed something: the session looked at
    o/r#12. Not when the text is an argument (``echo "gh pr view 12 ..."``) or the failure is
    masked (``|| true``, ``2>/dev/null``) with nothing printed."""
    urls: set[str] = set()
    if not stdout.strip() or re.search(r"\|\|\s*(?:true|:)\b", command):
        return urls
    listed: set[str] = set()  # repos whose PR/issue LIST ran in this command
    for seg in _split_unquoted(command, pipes=True, newlines=True):
        # gh must be the command run, possibly inside `X=$(...)`; not text in an echo or string.
        if not re.match(r"\s*(?:[A-Za-z_]\w*=)?(?:\$\(\s*)?(?:(?:timeout|env)\s+\S+\s+)*gh\s", seg):
            continue
        ref, repo = _GH_REF.search(seg), _GH_REPO.search(seg)
        if ref and repo:
            for kind in ("pull", "issues"):  # GitHub serves a PR under both
                urls.add(normalize_url(f"https://github.com/{repo.group('repo')}/{kind}/{ref.group('num')}"))
        if repo and _GH_LIST.search(seg):
            listed.add(repo.group("repo"))
    if len(listed) == 1:
        # `gh pr list --repo o/r`: each row the session saw ("10545  MERGED  2026-…  title") names
        # o/r#10545. A row counts only with a PR/issue state on it, so other numbers do not.
        repo = next(iter(listed))
        for m in _GH_ROW.finditer(stdout):
            for kind in ("pull", "issues"):
                urls.add(normalize_url(f"https://github.com/{repo}/{kind}/{m.group(1)}"))
    return urls


_GH_LIST = re.compile(r"\bgh\s+(?:(?:pr|issue)\s+list|search\s+(?:prs|issues))\b")
_GH_ROW = re.compile(r"(?m)^#?(\d{1,7})\b[^\n]*\b(?:OPEN|MERGED|CLOSED|open|merged|closed)\b")


# Options whose value is the next argument (so it is neither the pattern nor a path).
_GREP_VALUE_SHORT = {"grep": set("ABCmefdD"), "rg": set("ABCmefgtTMjEd")}
_GREP_VALUE_LONG = {
    "--glob", "--iglob", "--type", "--type-not", "--max-count", "--context", "--after-context",
    "--before-context", "--regexp", "--file", "--max-depth", "--max-columns", "--threads",
    "--include", "--exclude", "--exclude-dir", "--sort", "--sortr",
    "--encoding", "--type-add", "--replace", "--pre", "--pre-glob",
}


def _grep_operands(args: list[str], tool: str = "grep") -> tuple[bool, list[str]]:
    """(pattern given via -e/-f, positional operands), skipping option values."""
    short = _GREP_VALUE_SHORT.get(tool, _GREP_VALUE_SHORT["grep"])
    via_e, ops, skip = False, [], False
    for a in args:
        if skip:
            skip = False
            continue
        if a == "--":
            continue
        if a.startswith("--"):
            name = a.split("=", 1)[0]
            if name in ("--regexp", "--file"):
                via_e = True
            if name in _GREP_VALUE_LONG and "=" not in a:
                skip = True
            continue
        if a.startswith("-") and len(a) > 1:
            letters = a[1:]
            if "e" in letters or "f" in letters:
                via_e = True
            # "-C" / "-nC" take the next argument; "-C2" carries its value.
            if letters[-1] in short and letters.isalpha():
                skip = True
            continue
        ops.append(a)
    return via_e, ops


def _grep_targets(args: list[str], cwd: str | None, tool: str = "grep") -> list[str] | None:
    via_e, operands = _grep_operands(args, tool)
    paths = operands if via_e else operands[1:]
    out = []
    for t in paths or ([cwd] if cwd else []):
        t = os.path.expanduser(t)
        t = t if os.path.isabs(t) or not cwd else os.path.join(cwd, t)
        out.append(_realpath(t))
    return out or None


def _single_target(args: list[str], cwd: str | None, tool: str = "grep") -> str | None:
    """The one regular file a grep/rg invocation searched, if it searched exactly one."""
    if any(a in ("-r", "-R", "--recursive") or (a.startswith("-") and not a.startswith("--") and "r" in a[1:] and a[1:].isalpha()) for a in args):
        return None
    via_e, operands = _grep_operands(args, tool)
    files = operands if via_e else operands[1:]  # without -e, the first operand is the pattern
    if len(files) != 1:
        return None
    p = os.path.expanduser(files[0])
    p = os.path.normpath(p if os.path.isabs(p) or not cwd else os.path.join(cwd, p))
    return p if _isfile(p) else None


def _numbered(args: list[str]) -> bool:
    """grep/rg print line numbers only when asked (rg numbers by default only on a terminal)."""
    if "--no-line-number" in args or any(a.startswith("-") and not a.startswith("--") and "N" in a[1:] and a[1:].isalpha() for a in args):
        return False
    return "--line-number" in args or "--vimgrep" in args or any(
        a.startswith("-") and not a.startswith("--") and a[1:].isalpha() and "n" in a[1:] for a in args
    )


def _grep_evidence(argv: list[str], stdout: str, cwd: str | None, at: str | None, *, alone: bool, other_printers: bool = False) -> list[Observation] | None:
    """Line evidence from a grep/rg run, or None when its output cannot be read as numbered hits:
    no -n, or other commands printed into the same stdout (a linter's ``f.py:97:5:`` looks alike)."""
    if not argv:
        return None
    tool, args = os.path.basename(argv[0]), argv[1:]
    if not _numbered(args):
        return None
    single = None if tool == "git-grep" else _single_target(args, cwd, tool)
    if single and other_printers:
        # A single-file grep prints bare "N:text"; another file printed into the same stdout
        # (a second grep, a cat) would have its lines credited to this file.
        return [Observation(single, 0, [], "Bash-touch", at, line_numbers=[])]
    obs = from_grep_text(stdout, cwd, "Bash", at, single_file=single, roots=_grep_targets(args, cwd, tool))
    if alone:
        return obs
    # Other commands printed into the same stdout: keep only lines that match the pattern.
    match = _grep_matcher(args, tool)
    if match is None:
        return None
    kept = []
    for o in obs:
        if o.line_numbers is None:
            continue
        pairs = [(n, t) for n, t in zip(o.line_numbers, o.lines) if match(t)]
        if pairs:
            kept.append(Observation(o.path, pairs[0][0], [t for _, t in pairs], o.tool, o.at, line_numbers=[n for n, _ in pairs]))
    return kept


def _grep_matcher(args: list[str], tool: str):
    """A predicate for 'this text is a hit of this grep', or None if the pattern is unknown."""
    via_e, ops = _grep_operands(args, tool)
    pat = None
    for i, a in enumerate(args):
        if a in ("-e", "--regexp") and i + 1 < len(args):
            pat = args[i + 1]
            break
        if a.startswith("--regexp="):
            pat = a.split("=", 1)[1]
            break
    if pat is None and not via_e and ops:
        pat = ops[0]
    if not pat:
        return None
    flags = " ".join(a for a in args if a.startswith("-") and not a.startswith("--"))
    ci = "i" in flags or "--ignore-case" in args
    if "F" in flags or "--fixed-strings" in args:
        needle = pat.lower() if ci else pat
        return lambda t: needle in (t.lower() if ci else t)
    try:
        with warnings.catch_warnings():  # the agent's own pattern; its style is not our concern
            warnings.simplefilter("ignore")
            rx = re.compile(pat.replace("\\|", "|"), re.I if ci else 0)
    except re.error:
        return lambda t: pat in t
    return lambda t: rx.search(t) is not None


# Segments that print nothing: they do not share the stdout with the command that read a file.
_SILENT = re.compile(r"^(?:(?:export\s+)?[A-Za-z_]\w*=(?:\"[^\"]*\"|'[^']*'|\S*)|cd(?:\s+\S+)?|set\s+[-+]\w+)$")


_PRINTERS = {"cat", "sed", "head", "tail", "nl", "rg", "grep", "awk", "less", "bat", "more"}


def _drop_redirects(argv: list[str]) -> list[str]:
    """``cat a.py 2>/dev/null`` / ``sed ... 2>&1``: redirections are not operands."""
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
            continue
        if re.fullmatch(r"\d*(?:>>?|<)", a) or a == "&>":
            skip = True  # the target follows as its own token
            continue
        if re.fullmatch(r"\d*(?:>>?|<|&>)&?\S+", a):
            continue
        out.append(a)
    return out


# Filters that only select whole lines: a numbered line passes through them unchanged.
_LINE_FILTER = re.compile(
    r"^(?:sed\s+-n\s+['\"]?\d+(?:,\d+)?p(?:;\d+(?:,\d+)?p)*['\"]?|head(?:\s+-n)?(?:\s+-?\d+)?|tail(?:\s+-n)?(?:\s+-?\+?\d+)?|"
    r"(?:grep|rg)(?:\s+-(?![\w-]*[onbcABC])[\w-]+)*\s+(?:'[^']*'|\"[^\"]*\"|\S+))$"
)
_NUMBERED_LINE = re.compile(r"^\s*(\d+)\t(.*)$")


def _self_numbered(seg: str, cwd: str | None) -> str | None:
    """The file of ``nl -ba F | sed -n 'A,Bp'`` / ``cat -n F | head``: commands whose output lines
    carry their own line numbers, so where the slice starts does not matter."""
    parts = [x.strip() for x in _split_unquoted(seg, pipes=True)]
    try:
        argv = _drop_redirects(shlex.split(parts[0]))
    except ValueError:
        return None
    if not argv:
        return None
    cmd, args = os.path.basename(argv[0]), argv[1:]
    files = [a for a in args if not a.startswith("-")]
    if cmd == "nl":
        flags = [a for a in args if a.startswith("-")]
        ok = all(re.fullmatch(r"-b(?:a)?|-w\d*|-n(?:ln|rn|rz)?", f) for f in flags)
        files = [a for a in files if a not in ("a", "ln", "rn", "rz") and not a.isdigit()]
    elif cmd == "cat":
        ok = args[:1] == ["-n"] or args[:1] == ["-bn"]
        ok = ok and all(a in ("-n",) for a in args if a.startswith("-"))
    else:
        return None
    if not ok or len(files) != 1 or not all(_LINE_FILTER.match(x) for x in parts[1:]):
        return None
    path = os.path.expanduser(files[0])
    return os.path.normpath(path if os.path.isabs(path) or not cwd else os.path.join(cwd, path))


def _numbered_lines(text: str) -> list[tuple[int, str]]:
    return [(int(m.group(1)), m.group(2)) for m in map(_NUMBERED_LINE.match, text.split("\n")) if m]


def from_shell(command: str, stdout: str, cwd: str | None, at: str | None = None) -> list[Observation]:
    """Recognize simple file-printing commands whose stdout is a known file slice."""
    out: list[Observation] = []
    owner: dict[int, int] = {}  # index in ``out`` -> index of the segment that printed it
    segments = [_unwrap(x) for x in _split_unquoted(command, newlines=True)]
    loud = [x for x in segments if x and not _SILENT.match(x)]
    printers = [x for x in loud if (x.split() or [""])[0].rsplit("/", 1)[-1] in _PRINTERS]
    if any(re.match(r"cd\b", x) for x in segments[1:]):
        # The command changed directory part-way: the same relative name can mean two files.
        return [o for o in _file_level_mentions(segments, cwd, at)]
    extra: list[Observation] = []  # grep evidence; other segments keep contributing
    fenced: list[tuple[list[int], list[str]]] | None | bool = False  # echo-marker chunks, lazily

    def grep_for(k: int, gargv: list[str]) -> list[Observation] | None:
        nonlocal fenced
        if len(loud) > 1:
            # Several commands share stdout. If echo markers fence this grep's output off on its
            # own, read that chunk as if the grep ran alone.
            if fenced is False:
                fenced = _echo_chunks(segments, stdout)
            for segs, chunk in fenced or []:
                if segs == [k]:
                    return _grep_evidence(gargv, "\n".join(chunk) + "\n", cwd, at, alone=True)
        return _grep_evidence(gargv, stdout, cwd, at, alone=len(loud) == 1, other_printers=len(printers) > 1)

    numbered: dict[int, str] = {}  # segment -> file, for self-numbered printers (nl -ba, cat -n)
    for k, seg in enumerate(segments):
        nfile = _self_numbered(seg, cwd)
        if nfile is not None:
            numbered[k] = nfile
            continue
        pipe = _unquoted_pipe(seg)
        if pipe is not None:
            first = seg[:pipe].strip()
            if first.startswith(("rg ", "grep ", "git grep ")):
                try:
                    fargv = _drop_redirects(shlex.split(first))
                except ValueError:
                    fargv = []
                if fargv[:2] == ["git", "grep"]:
                    fargv = ["git-grep", *fargv[2:]]
                grep = grep_for(k, fargv)
                if grep is not None:
                    extra.extend(grep)
            continue
        try:
            argv = _drop_redirects(shlex.split(seg))
        except ValueError:
            continue
        if not argv:
            continue
        cmd, args = os.path.basename(argv[0]), argv[1:]
        if cmd == "head" and "-n" in args:
            i = args.index("-n")
            args = args[:i] + args[i + 2 :]  # drop "-n N"
        files = [a for a in args if not a.startswith("-") and not re.fullmatch(r"'?\d*,?\d*p'?", a)]
        if cmd == "cat" and any(a.startswith("-") for a in args):
            continue  # cat -s/-n/-v... changes the lines; do not attribute line numbers
        if cmd == "git" and args[:1] == ["grep"]:
            # `git grep -n` always prefixes the path, even for one file: never the bare N:text form.
            grep = grep_for(k, ["git-grep", *args[1:]])
            if grep is not None:
                extra.extend(grep)
            continue
        if cmd in ("rg", "grep"):
            grep = grep_for(k, argv)
            if grep is not None:
                extra.extend(grep)
            continue
        if cmd not in ("cat", "sed", "head", "tail", "nl") or len(files) != 1:
            continue
        path = os.path.expanduser(files[0])  # before the join: `~/x` is absolute, not relative
        path = path if os.path.isabs(path) or not cwd else os.path.join(cwd, path)
        lines = stdout.split("\n")
        if lines and lines[-1] == "":
            lines = lines[:-1]
        start = 1
        if cmd == "sed":
            # Exactly one "N,Mp" / "Np" script: several ranges print back to back into one
            # stdout, so the second range's text would land on the first range's line numbers.
            scripts = [a for a in args if re.fullmatch(r"\d+(?:,\d+)?p", a)]
            if "-n" not in args or len(scripts) != 1 or args.count("-e") > 1:
                continue
            start = int(scripts[0].split(",")[0].rstrip("p"))
        elif cmd == "tail":
            continue  # start line unknown without the file length; skip rather than guess
        elif cmd == "nl":
            lines = [re.sub(r"^\s*\d+\t", "", x) for x in lines]
        out.append(Observation(os.path.normpath(path), start, lines, "Bash", at))
        owner[len(out) - 1] = k
    if numbered and stdout.strip():
        extra.extend(_numbered_evidence(numbered, segments, loud, stdout, at))
    if len(out) > 1 or (out and len(loud) > 1):
        split = _split_by_echo_markers(segments, stdout, {owner[i]: o for i, o in enumerate(out)})
        if split is not None:
            return extra + split
        # Several things printed into one stdout: the TEXT cannot be attributed to lines. File-level
        # evidence, plus, for an unpiped `sed -n A,Bp f`, the line NUMBERS it printed (text unknown,
        # so quotes on those lines are not judged from it).
        res = extra + [Observation(o.path, 0, [], "Bash-touch", at, line_numbers=[]) for o in out]
        if stdout.strip():
            for i, o in enumerate(out):
                seg = segments[owner[i]]
                rng = re.fullmatch(r"sed\s+-n\s+'?(\d+),(\d+)p'?\s+\S+", seg.strip())
                if rng and "|" not in seg:
                    a, b = int(rng.group(1)), int(rng.group(2))
                    n_file = _line_count(o.path)
                    if n_file is not None:
                        b = min(b, n_file)
                    if a <= b and b - a < 5000:
                        res.append(Observation(o.path, 0, [None] * (b - a + 1), "Bash-range", at, line_numbers=list(range(a, b + 1))))
        return res
    return extra + out


# Commands that run another command unchanged: `timeout 60 rg -n ...` is the rg.
_WRAPPER = re.compile(
    r"^(?:(?:timeout(?:\s+-[sk]\s*\S+|\s+--\S+)*\s+\d+(?:\.\d+)?[smhd]?|nice(?:\s+-n\s*-?\d+|\s+-\d+)?"
    r"|command(?=\s+[^-\s])|time|stdbuf(?:\s+-[ioe]\S+)+|env(?:\s+[A-Za-z_]\w*=\S*)+)\s+)+"
)


def _unwrap(seg: str) -> str:
    return _WRAPPER.sub("", seg, count=1)


_HEREDOC_START = re.compile(r"<<(?P<dash>-?)\s*(['\"]?)(?P<word>[A-Za-z_]\w*)\2")


def _split_unquoted(command: str, *, pipes: bool = False, newlines: bool = False) -> list[str]:
    """Split a shell command on ``&&``, ``;``, ``||`` (and ``|``/newlines if asked) outside quotes.

    A grep pattern like ``"a && b"`` is one argument, not two commands. If the quotes never
    balance (an apostrophe in a heredoc or comment), fall back to splitting on every separator."""
    command = command.replace("\\\n", " ")  # line continuations
    seps = ["&&", "||", ";"] + (["|"] if pipes else []) + (["\n"] if newlines else [])
    out, cur, q, i, n = [], [], None, 0, len(command)
    heredocs: list[tuple[str, bool]] = []  # (terminator, tabs stripped) opened on this line
    while i < n:
        ch = command[i]
        if not q and ch == "<" and command.startswith("<<", i) and not command.startswith("<<<", i):
            m = _HEREDOC_START.match(command, i)
            if m:
                heredocs.append((m.group("word"), bool(m.group("dash"))))
        if not q and ch == "\n" and heredocs:
            # The body is data, not commands: skip to each terminator line in turn.
            j = i + 1
            for word, dash in heredocs:
                while j < n:
                    end = command.find("\n", j)
                    line = command[j : end if end != -1 else n]
                    j = end + 1 if end != -1 else n
                    if (line.lstrip("\t") if dash else line) == word:
                        break
            heredocs = []
            if newlines:
                out.append("".join(cur).strip())
                cur = []
            i = j
            continue
        if q:
            if ch == "\\" and q == '"' and i + 1 < n:
                cur.append(command[i : i + 2])
                i += 2
                continue
            if ch == q:
                q = None
            cur.append(ch)
            i += 1
            continue
        if ch == "\\" and i + 1 < n:
            cur.append(command[i : i + 2])
            i += 2
            continue
        if ch in "'\"":
            q = ch
            cur.append(ch)
            i += 1
            continue
        sep = next((x for x in seps if command.startswith(x, i)), None)
        if sep:
            out.append("".join(cur).strip())
            cur = []
            i += len(sep)
            continue
        cur.append(ch)
        i += 1
    if q:
        alt = "|".join(re.escape(x) for x in seps)
        return [x.strip() for x in re.split(rf"\s*(?:{alt})\s*", command)]
    out.append("".join(cur).strip())
    return out


def _unquoted_pipe(seg: str) -> int | None:
    """Index of the first `|` outside quotes (a pattern like "a|b" is not a pipeline)."""
    q = None
    for i, ch in enumerate(seg):
        if q:
            if ch == q:
                q = None
        elif ch in "'\"":
            q = ch
        elif ch == "|":
            return i
    return None


def _line_count(path: str) -> int | None:
    if not _isfile(path):
        return None
    try:
        with open(path, "rb") as fh:
            return sum(1 for _ in fh)
    except OSError:
        return None


def _file_level_mentions(segments: list[str], cwd: str | None, at: str | None) -> list[Observation]:
    """Absolute paths a command printed from: file-level evidence only (relative ones are ambiguous)."""
    out = []
    for seg in segments:
        try:
            argv = _drop_redirects(shlex.split(seg))
        except ValueError:
            continue
        if argv and os.path.basename(argv[0]) in _PRINTERS:
            for a in argv[1:]:
                p = os.path.expanduser(a)
                if os.path.isabs(p) and _isfile(p):
                    out.append(Observation(os.path.normpath(p), 0, [], "Bash-touch", at, line_numbers=[]))
    return out


def _numbered_evidence(numbered: dict[int, str], segments: list[str], loud: list[str], stdout: str, at: str | None) -> list[Observation]:
    """Attribute self-numbered lines to their files. Alone in stdout: all of them. Fenced by echo
    markers: the fence's lines. Several in one stdout: split where the numbering restarts, but only
    if the restarts match the commands one for one (and nothing else prints numbered lines).
    Anything else: file-level."""
    def obs(path: str, pairs: list[tuple[int, str]]) -> Observation:
        return Observation(path, 0, [t for _, t in pairs], "Bash", at, line_numbers=[n for n, _ in pairs])

    touch = [Observation(p, 0, [], "Bash-touch", at, line_numbers=[]) for p in numbered.values()]
    if len(loud) == 1 and len(numbered) == 1:
        (path,) = numbered.values()
        pairs = _numbered_lines(stdout)
        return [obs(path, pairs)] if pairs else touch
    chunks = _echo_chunks(segments, stdout)
    if chunks is not None:
        res = []
        for segs, chunk in chunks:
            if len(segs) == 1 and segs[0] in numbered:
                pairs = _numbered_lines("\n".join(chunk))
                if pairs:
                    res.append(obs(numbered[segs[0]], pairs))
        return res + touch
    others = [x for x in loud if not re.match(r"(?:echo|printf)\b", x) and x not in {segments[k] for k in numbered}]
    pairs = _numbered_lines(stdout)
    if others or not pairs:
        return touch
    if len(set(numbered.values())) == 1:
        # Several slices of ONE file: every numbered line is that file's, wherever the runs break.
        return [obs(next(iter(numbered.values())), pairs)] + touch
    runs: list[list[tuple[int, str]]] = [[pairs[0]]]
    for prev, cur in zip(pairs, pairs[1:]):
        if cur[0] <= prev[0]:
            runs.append([])
        runs[-1].append(cur)
    order = [numbered[k] for k in sorted(numbered)]
    if len(runs) != len(order):
        return touch  # a range that continues past the previous one hides a boundary
    return [obs(p, r) for p, r in zip(order, runs)] + touch


def _echo_chunks(segments: list[str], stdout: str) -> list[tuple[list[int], list[str]]] | None:
    """``echo "--- a"; sed -n 1,9p a; echo "--- b"; cat b``: each literal echo line in stdout
    fences off the output of the commands after it. Returns (segment indexes, output lines) per
    fenced group. Anything ambiguous returns None: a marker that is blank or too short, or that
    occurs in stdout more often than it was echoed (a markdown ``---`` inside a printed file
    would split it in the wrong place). A bare ``echo`` prints a blank line, not a marker."""
    lines = stdout.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    groups: list[list[int]] = [[]]
    blanks: list[int] = [0]  # bare `echo`s per group: blank lines that are not command output
    markers: list[str] = []
    for k, seg in enumerate(segments):
        if not seg or _SILENT.match(seg):
            continue
        if re.fullmatch(r"echo(?:\s+(?:\"\"|''))?", seg):
            blanks[-1] += 1
            continue
        m = re.fullmatch(r"echo\s+(?:\"([^\"$`\\]*)\"|'([^']*)'|((?!-[A-Za-z]+(?:\s|$))[^\s\"'$`\\;&|<>]+(?:\s+[^\s\"'$`\\;&|<>]+)*))", seg)
        if m:
            if m.group(3) is not None:
                mk = " ".join(m.group(3).split())  # unquoted: `echo -----`, `echo ... next`
            else:
                mk = m.group(1) if m.group(1) is not None else m.group(2)
            if len(mk.strip()) < 3:
                return None
            markers.append(mk)
            groups.append([])
            blanks.append(0)
        else:
            groups[-1].append(k)
    if not markers:
        return None
    for mk in set(markers):
        if lines.count(mk) != markers.count(mk):
            return None
    pos, bounds = 0, []
    for mk in markers:
        try:
            i = lines.index(mk, pos)
        except ValueError:
            return None
        bounds.append(i)
        pos = i + 1
    edges = [-1, *bounds, len(lines)]
    out = []
    for g, segs in enumerate(groups):
        chunk = lines[edges[g] + 1 : edges[g + 1]]
        for _ in range(blanks[g]):
            if chunk and chunk[-1] == "":
                chunk = chunk[:-1]
        out.append((segs, chunk))
    return out


def _split_by_echo_markers(segments: list[str], stdout: str, printers: dict[int, Observation]) -> list[Observation] | None:
    """Attribute each echo-fenced chunk to the single file printer in it (see ``_echo_chunks``)."""
    chunks = _echo_chunks(segments, stdout)
    if chunks is None:
        return None
    result: list[Observation] = []
    for segs, chunk in chunks:
        found = [printers[k] for k in segs if k in printers]
        if not found:
            continue
        o = found[0]
        if len(segs) == 1 and len(found) == 1:
            rng = re.search(r"\b(\d+),(\d+)p\b", segments[segs[0]])
            if rng and len(chunk) > int(rng.group(2)) - int(rng.group(1)) + 1:
                return None  # more lines than `sed -n A,Bp` can print: the split is wrong somewhere
            result.append(Observation(o.path, o.line_start, chunk, "Bash", o.at))
        else:
            result.extend(Observation(f.path, 0, [], "Bash-touch", f.at, line_numbers=[]) for f in found)
    return result


# --- transcript readers ---------------------------------------------------------


_RUNTIME_NOTICES = ("<task-notification>", "<system-reminder>", "<command-name>", "<local-command")


def read_claude_transcript(path: str, subagents: bool = True) -> tuple[Session, list[tuple[str, str]]]:
    """Parse a Claude Code JSONL transcript.

    Returns the session observations and the assistant text messages as
    ``(timestamp, text)`` in order. Subagent transcripts stored next to the
    session (``<session>/subagents/*.jsonl``) are loaded as *delegated*
    observations: the orchestrator did not read those lines itself.
    """
    sess = Session()
    texts: list[tuple[str, str]] = []
    pending: dict[str, tuple[str, dict[str, Any]]] = {}
    turn = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(e, dict):
                continue  # a stray array or scalar line must not abort the whole check
            if e.get("type") == "pr-link" and isinstance(e.get("prUrl"), str):
                # The runtime records PRs the session opened or linked: the session produced that URL.
                sess.urls.add(normalize_url(e["prUrl"]))
            if e.get("cwd"):
                sess.cwd = sess.cwd or e["cwd"]
                sess.cwds.add(e["cwd"])
                sess.last_cwd = e["cwd"]
            msg = e.get("message") if isinstance(e.get("message"), dict) else {}
            content = msg.get("content")
            at = e.get("timestamp")
            if e.get("type") == "user" and not e.get("isMeta"):
                prompt = isinstance(content, str) or (
                    isinstance(content, list)
                    and any(isinstance(c, dict) and c.get("type") == "text" for c in content)
                    and not any(isinstance(c, dict) and c.get("type") == "tool_result" for c in content)
                )
                origin = e.get("origin") if isinstance(e.get("origin"), dict) else {}
                text0 = content if isinstance(content, str) else ""
                if origin.get("kind") not in (None, "human", "user") or text0.lstrip().startswith(_RUNTIME_NOTICES):
                    prompt = False  # a background-task notice is not the human starting a new turn
                if prompt:
                    turn += 1  # a new human prompt starts a new turn
            if e.get("type") == "user" and isinstance(content, str):
                sess.urls |= urls_in(content)  # a link the user gave is sourced
            if not isinstance(content, list):
                continue
            for c in content:
                if not isinstance(c, dict):
                    continue
                if c.get("type") == "tool_use":
                    pending[c.get("id", "")] = (c.get("name", ""), c.get("input") or {})
                elif c.get("type") == "text" and e.get("type") == "assistant":
                    texts.append((at or "", c.get("text", "")))
                    sess.text_turns.append(turn)
                elif c.get("type") == "text" and e.get("type") == "user":
                    sess.urls |= urls_in(c.get("text", ""))
                elif c.get("type") == "tool_result":
                    name, tin = pending.get(c.get("tool_use_id", ""), ("", {}))
                    tur = e.get("toolUseResult")
                    # Any URL that came back from ANY tool (gh pr create, curl, WebFetch, MCP...) is sourced;
                    # inputs count only for web tools (a fetched URL), not for arbitrary commands.
                    if name in _AUTHORING_TOOLS:
                        pass  # the agent's own writing echoed back is not a source for its links
                    elif name in _DELEGATING_TOOLS:
                        # A subagent's report: the orchestrator did not fetch these itself.
                        sess.delegated_urls |= urls_in(tur if tur is not None else c.get("content"))
                    elif not is_error_result(tur, c.get("content"), bool(c.get("is_error"))):
                        # A URL inside an error ("fetch failed: https://...") was not obtained.
                        got = urls_in(tur if tur is not None else c.get("content"))
                        if name in ("Bash", "Shell") and isinstance(tin, dict):
                            cmd = str(tin.get("command", ""))
                            got -= urls_in(cmd)  # `echo https://x` returns what the agent typed
                            out_text = tur.get("stdout") if isinstance(tur, dict) else _text_of(c.get("content"))
                            sess.urls |= gh_refs(cmd, out_text if isinstance(out_text, str) else "")
                        sess.urls |= got
                        if _is_web_tool(name):
                            sess.urls |= urls_in(tin)
                    for obs in observe_tool(
                        name, tin, tur, c.get("content"), e.get("cwd") or sess.cwd, at, bool(c.get("is_error"))
                    ):
                        sess.add(obs)
    if subagents:
        sub_dir = os.path.join(path[: -len(".jsonl")] if path.endswith(".jsonl") else path, "subagents")
        if os.path.isdir(sub_dir):
            for name in sorted(os.listdir(sub_dir)):
                if not name.endswith(".jsonl"):
                    continue
                try:
                    sub, _ = read_claude_transcript(os.path.join(sub_dir, name), subagents=False)
                except OSError:
                    continue
                for o in sub.observations:
                    o.delegated = True
                    sess.observations.append(o)
                sess.delegated_urls |= sub.urls - sess.urls
    return sess, texts


_AUTHORING_TOOLS = {"Write", "Edit", "MultiEdit", "NotebookEdit", "TodoWrite"}
_DELEGATING_TOOLS = {"Task", "Agent"}


def _is_web_tool(name: str) -> bool:
    n = name.lower()
    return n in ("webfetch", "websearch") or ("fetch" in n or "search" in n or "browse" in n) and n.startswith("mcp__")


def _persisted_preview(text: str) -> str | None:
    """``<persisted-output>`` results show the model a ~2 KB preview; the full output went to a
    file. Return the preview's complete lines (the last one may be cut), or None."""
    if "<persisted-output>" not in text:
        return None
    m = re.search(r"Preview \(first[^)]*\):\n(.*?)(?:\n\.\.\.)?\n</persisted-output>", text, re.S)
    if not m:
        return ""
    lines = m.group(1).split("\n")
    return "\n".join(lines[:-1]) + "\n" if len(lines) > 1 else ""


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(x.get("text", "") for x in content if isinstance(x, dict))
    return ""



# --- Codex rollouts -----------------------------------------------------------
#
# Codex runs tools from small JavaScript cells: ``const r = await tools.exec_command({cmd: "...",
# workdir: "..."}); text(r.output);``. The cell's output is whatever the JS printed, not the
# command's stdout. Only cells that call exec_command once, with literal arguments, and print its
# output unchanged, are read as "this command printed this"; any other cell still sources URLs.

_JS_LIT = r"\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'|`(?:[^`\\$]|\\.)*`"
_JS_PAIR = re.compile(rf"\s*(?:\"(\w+)\"|'(\w+)'|(\w+))\s*:\s*({_JS_LIT}|-?\d+(?:\.\d+)?|true|false|null)\s*,?")
_JS_CALL = re.compile(r"tools\.exec_command\(\s*\{(?P<obj>(?:[^{}`\"']|" + _JS_LIT + r")*)\}\s*\)", re.S)
_CELL_FORMS = {
    re.compile(r"(?:const|let|var) (\w+) ?= ?await CALL;? ?text\(\1\.output\);?"): "raw",
    re.compile(r"text\(\(await CALL\)\.output\);?"): "raw",
    re.compile(r"(?:const|let|var) (\w+) ?= ?await CALL;? ?text\((?:JSON\.stringify\()?\1\)?\);?"): "json",
    re.compile(r"text\((?:JSON\.stringify\()?await CALL\)?\);?"): "json",
}
_CODEX_DELEGATING = ("wait_agent", "read_thread", "wait_threads")
_CODEX_AUTHORING = ("send_message", "spawn_agent", "followup_task")


def _js_string(lit: str) -> str | None:
    """Decode a JS string literal; None for a template with ``${...}`` (the value is unknown)."""
    q, body = lit[0], lit[1:-1]
    if q == "`" and "${" in body:
        return None
    out, i = [], 0
    simple = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f", "v": "\v", "0": "\0"}
    while i < len(body):
        ch = body[i]
        if ch != "\\" or i + 1 == len(body):
            out.append(ch)
            i += 1
            continue
        nxt = body[i + 1]
        if nxt == "u" and re.fullmatch(r"[0-9a-fA-F]{4}", body[i + 2 : i + 6]):
            out.append(chr(int(body[i + 2 : i + 6], 16)))
            i += 6
        elif nxt == "x" and re.fullmatch(r"[0-9a-fA-F]{2}", body[i + 2 : i + 4]):
            out.append(chr(int(body[i + 2 : i + 4], 16)))
            i += 4
        elif nxt == "\n":
            i += 2  # line continuation
        else:
            out.append(simple.get(nxt, nxt))
            i += 2
    return "".join(out)


def codex_exec_cell(code: str) -> tuple[dict[str, Any], str] | None:
    """(arguments, "raw"|"json") for a cell that runs one literal exec_command and prints its
    output as-is; None for anything else."""
    code = re.sub(r"^\s*//[^\n]*\n", "", code)
    if code.count("tools.") != 1:
        return None
    m = _JS_CALL.search(code)
    if not m:
        return None
    args: dict[str, Any] = {}
    pos, obj = 0, m.group("obj")
    while pos < len(obj.rstrip()):
        pm = _JS_PAIR.match(obj, pos)
        if not pm or pm.end() == pos:
            return None  # a computed value: not a literal call
        key, val = pm.group(1) or pm.group(2) or pm.group(3), pm.group(4)
        args[key] = _js_string(val) if val[0] in "\"'`" else val
        pos = pm.end()
    if not isinstance(args.get("cmd"), str):
        return None
    rest = " ".join((code[: m.start()] + "CALL" + code[m.end() :]).split())
    for rx, form in _CELL_FORMS.items():
        if rx.fullmatch(rest):
            return args, form
    return None


_JS_CONST = re.compile(r"\b(?:const|let|var)\s+(\w+)\s*=\s*(" + _JS_LIT + r")\s*[;\n]")
_JS_OBJ = re.compile(r"\{((?:[^{}`\"']|" + _JS_LIT + r")*\bcmd\s*:(?:[^{}`\"']|" + _JS_LIT + r")*)\}")
_JS_PAIR_ID = re.compile(rf"\s*(?:\"(\w+)\"|'(\w+)'|(\w+))\s*:\s*({_JS_LIT}|-?\d+(?:\.\d+)?|true|false|null|[A-Za-z_]\w*)\s*,?")


def codex_batch_commands(code: str) -> list[dict[str, Any]]:
    """Literal exec_command argument objects in a cell that runs several commands (a list mapped
    over, Promise.all). Values may be literals or simple ``const NAME = "..."`` constants."""
    consts = {m.group(1): _js_string(m.group(2)) for m in _JS_CONST.finditer(code)}
    found = []
    for m in _JS_OBJ.finditer(code):
        obj, pos, args = m.group(1), 0, {}
        while pos < len(obj.rstrip()):
            pm = _JS_PAIR_ID.match(obj, pos)
            if not pm or pm.end() == pos:
                args = {}
                break
            key, val = pm.group(1) or pm.group(2) or pm.group(3), pm.group(4)
            if val[0] in "\"'`":
                args[key] = _js_string(val)
            elif re.fullmatch(r"[A-Za-z_]\w*", val) and val not in ("true", "false", "null"):
                args[key] = consts.get(val)  # None when not a simple constant
            else:
                args[key] = val
            pos = pm.end()
        if isinstance(args.get("cmd"), str):
            found.append(args)
    return found


def _strings_of(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings_of(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings_of(v)]
    return []


def _batch_grep_evidence(code: str, out_text: str, cwd: str | None, at: str | None) -> list[Observation]:
    """A cell that ran several commands printed their outputs together in its own format. Only
    self-describing lines are attributed: ``path:N:text`` hits of a multi-file grep in the cell,
    kept when the path is under that grep's roots and the text matches its pattern."""
    try:
        parsed = json.loads(out_text)
        text = "\n".join(_strings_of(parsed))
    except ValueError:
        text = out_text
    res: list[Observation] = []
    for args in codex_batch_commands(code):
        wd = args.get("workdir") if isinstance(args.get("workdir"), str) else None
        ecwd = os.path.join(cwd, wd) if wd and cwd and not os.path.isabs(wd) else (wd or cwd)
        cmd = expand_assignments(args["cmd"])
        ecwd = effective_cwd(cmd, ecwd)
        for seg in _split_unquoted(_CD.sub("", cmd, count=1), newlines=True):
            seg = _unwrap(seg)
            pipe = _unquoted_pipe(seg)
            first = seg[:pipe].strip() if pipe is not None else seg
            if not first.startswith(("rg ", "grep ", "git grep ")):
                continue
            try:
                argv = _drop_redirects(shlex.split(first))
            except ValueError:
                continue
            if argv[:2] == ["git", "grep"]:
                argv = ["git-grep", *argv[2:]]
            elif _single_target(argv[1:], ecwd, os.path.basename(argv[0])) and "-H" not in argv and "--with-filename" not in argv:
                continue  # a one-file grep prints bare "N:text": not self-describing
            got = _grep_evidence(argv, text, ecwd, at, alone=False)
            res.extend(o for o in got or [] if o.line_numbers)
    return res


_CUT_HEADER = re.compile(r"^Warning: truncated output[^\n]*\n(?:Total output lines:[^\n]*\n)?\n?")
_CUT_MARK = re.compile(r"…\d+ tokens truncated…")


def _codex_uncut(stdout: str) -> tuple[bool, str]:
    """(was the output cut, the output without Codex's cut banner and the one line that holds
    the cut mark, which joins a head fragment to a tail fragment)."""
    cut = False
    while True:
        m = _CUT_HEADER.match(stdout)
        if not m:
            break
        stdout, cut = stdout[m.end() :], True
    if _CUT_MARK.search(stdout):
        cut = True
        stdout = "\n".join(x for x in stdout.split("\n") if not _CUT_MARK.search(x))
    return cut, stdout


_JS_LABELLED = re.compile(r"\[\s*(" + _JS_LIT + r")\s*,\s*(" + _JS_LIT + r")\s*(?:,[^\]\[]*)?\]")


def _labelled_batch_evidence(code: str, text: str, cwd: str | None, at: str | None) -> list[Observation]:
    """``const commands = [["label", "nl -ba f | sed -n '1,9p'"], ...]`` run in a loop that prints
    each label before its output. When every label is distinctive and occurs on exactly one
    output line, in order, the lines between two labels are that command's output. Only lines
    that carry their own numbers are kept: the cell may print extra text around each output."""
    pairs = [(_js_string(a), _js_string(b)) for a, b in _JS_LABELLED.findall(code)]
    pairs = [(a, b) for a, b in pairs if a and b]
    if len(pairs) < 2 or any(len(a.strip()) < 4 for a, _ in pairs):
        return []
    consts = {m.group(1): _js_string(m.group(2)) for m in _JS_CONST.finditer(code)}
    wd_m = re.search(r"workdir\s*:\s*(" + _JS_LIT + r"|[A-Za-z_]\w*)", code)
    wd = None
    if wd_m:
        wd = _js_string(wd_m.group(1)) if wd_m.group(1)[0] in "\"'`" else consts.get(wd_m.group(1))
    ecwd = os.path.join(cwd, wd) if wd and cwd and not os.path.isabs(wd) else (wd or cwd)
    lines = text.split("\n")
    bounds, pos = [], 0
    for label, _ in pairs:
        hits = [i for i, x in enumerate(lines) if label in x]
        if len(hits) != 1 or hits[0] < pos:
            return []  # a label that is missing, repeated, or out of order: no split
        bounds.append(hits[0])
        pos = hits[0] + 1
    res: list[Observation] = []
    for (label, cmd), start, end in zip(pairs, bounds, [*bounds[1:], len(lines)]):
        chunk = "\n".join(lines[start + 1 : end]) + "\n"
        cmd = expand_assignments(cmd)
        for o in from_shell(_CD.sub("", cmd, count=1), chunk, effective_cwd(cmd, ecwd), at):
            if o.line_numbers:
                res.append(o)
    return res


def _codex_text(out: Any) -> str:
    """Codex splits one output into parts ("...Output:\\n", then the stdout): concatenate them as
    they are. Joining with newlines would insert a line and shift every line number by one."""
    if isinstance(out, list):
        return "".join(x.get("text", "") for x in out if isinstance(x, dict))
    return out if isinstance(out, str) else ""


def _codex_output(out: Any) -> str | None:
    text = _codex_text(out)
    m = re.match(r"Script completed\n(?:Wall time[^\n]*\n)?Output:\n", text)
    return text[m.end() :] if m else None


def read_codex_transcript(path: str) -> tuple[Session, list[tuple[str, str]]]:
    """Parse a Codex rollout (``~/.codex/sessions/.../rollout-*.jsonl``). Same contract as
    :func:`read_claude_transcript`. Forked subagent threads live in their own rollouts and are
    not loaded; their reports reach this session as delegated URLs."""
    sess = Session()
    texts: list[tuple[str, str]] = []
    calls: dict[str, tuple[str, str]] = {}  # call_id -> (name, input)
    turn = 0
    cwd: str | None = None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(e, dict) or not isinstance(e.get("payload"), dict):
                continue
            p, at = e["payload"], e.get("timestamp")
            if e.get("type") == "turn_context" and isinstance(p.get("cwd"), str):
                cwd = p["cwd"]
                sess.cwd = sess.cwd or cwd
                sess.cwds.add(cwd)
                sess.last_cwd = cwd
                continue
            if e.get("type") != "response_item":
                continue
            kind = p.get("type")
            if kind == "message":
                body = "\n".join(c.get("text", "") for c in p.get("content") or [] if isinstance(c, dict))
                if p.get("role") == "assistant":
                    texts.append((at or "", body))
                    sess.text_turns.append(turn)
                elif p.get("role") == "user":
                    if not body.lstrip().startswith("<"):
                        turn += 1  # a human prompt, not an injected <environment_context> block
                    sess.urls |= urls_in(body)
            elif kind in ("custom_tool_call", "function_call"):
                calls[p.get("call_id", "")] = (p.get("name", ""), p.get("input") or p.get("arguments") or "")
            elif kind in ("custom_tool_call_output", "function_call_output"):
                name, code = calls.get(p.get("call_id", ""), ("", ""))
                out = p.get("output")
                text = _codex_text(out)
                if any(t in code or t == name for t in _CODEX_AUTHORING):
                    continue  # the agent's own words echoed back
                if any(t in code or t == name for t in _CODEX_DELEGATING):
                    sess.delegated_urls |= urls_in(text)
                    continue
                if name != "exec":
                    continue
                sess.urls |= urls_in(text) - urls_in(code)  # `echo https://x` returns what was typed
                cell = codex_exec_cell(code)
                stdout = _codex_output(out)
                if stdout is None:
                    continue
                if cell is None:
                    if "tools.exec_command" in code:
                        _cut, uncut = _codex_uncut(stdout)
                        for o in _batch_grep_evidence(code, uncut, cwd, at) + _labelled_batch_evidence(code, uncut, cwd, at):
                            sess.add(o)
                    continue
                args, form = cell
                exit_code = 0
                if form == "json":
                    try:
                        res = json.loads(stdout)
                    except ValueError:
                        continue
                    if not isinstance(res, dict) or not isinstance(res.get("output"), str):
                        continue
                    stdout, exit_code = res["output"], res.get("exit_code") or 0
                cmd = expand_assignments(args["cmd"])
                wd = args.get("workdir") if isinstance(args.get("workdir"), str) else None
                ecwd = os.path.join(cwd, wd) if wd and cwd and not os.path.isabs(wd) else (wd or cwd)
                sess.urls |= gh_refs(cmd, stdout)
                if exit_code:
                    # Failed: the printed lines cannot be trusted as the file. File-level only.
                    for o in from_shell_touches(cmd, effective_cwd(cmd, ecwd), set(), at):
                        sess.add(o)
                    continue
                cut, stdout = _codex_uncut(stdout)
                for o in observe_tool("Bash", {"command": cmd}, {"stdout": stdout}, "", ecwd, at):
                    if cut and o.line_numbers is None and o.lines:
                        # The middle was cut: lines placed by counting would be misplaced. Keep
                        # only lines that carry their own numbers (grep hits, nl / cat -n).
                        o = Observation(o.path, 0, [], "Bash-touch", o.at, line_numbers=[])
                    sess.add(o)
    return sess, texts


def read_transcript(path: str) -> tuple[Session, list[tuple[str, str]]]:
    """Read a Claude Code transcript or a Codex rollout, whichever ``path`` is."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(e, dict):
                if e.get("type") in ("session_meta", "response_item", "turn_context") and "payload" in e:
                    return read_codex_transcript(path)
                break
    return read_claude_transcript(path)

def from_write(tool_input: dict[str, Any], at: str | None = None) -> Observation | None:
    """The agent authored this content, so it knows every line of it."""
    path, content = tool_input.get("file_path"), tool_input.get("content")
    if not path or not isinstance(content, str):
        return None
    lines = content.replace("\r\n", "\n").split("\n")
    if len(lines) > 1 and lines[-1] == "":
        lines = lines[:-1]
    return Observation(path, 1, lines, "Write", at)


def from_edit(tool_input: dict[str, Any], structured: Any, at: str | None = None) -> Observation | None:
    """Edit results carry a structured patch; the new-side lines are known with numbers."""
    path = tool_input.get("file_path")
    patch = structured.get("structuredPatch") if isinstance(structured, dict) else None
    if not path or not isinstance(patch, list):
        return None
    nums: list[int] = []
    texts: list[str] = []
    for hunk in patch:
        try:
            n = int(hunk.get("newStart", 1))
        except (TypeError, ValueError):
            continue
        for ln in hunk.get("lines") or []:
            if ln.startswith("-"):
                continue
            if ln.startswith("+"):
                # Only the lines the agent wrote: the patch's context lines were never shown
                # to it (the tool result just says the file was updated).
                nums.append(n)
                texts.append(ln[1:])
            n += 1
    if not nums:
        return None
    return Observation(path, 0, texts, "Edit", at, line_numbers=nums)


# Balanced parentheses are part of a URL: /wiki/Python_(programming_language)
_URL = re.compile(r"https?://(?:[^\s\"'<>()\[\]{}|\\^`]|\([^\s\"'<>()]*\))+")


def normalize_url(url: str) -> str:
    url = url.strip().rstrip(".,;:!?]}>'\"*`")
    while url.endswith(")") and url.count(")") > url.count("("):
        url = url[:-1].rstrip(".,;:!?]}>'\"*`")  # "(see https://x/a)" but not ".../Python_(language)"
    url = url.split("#", 1)[0]
    if "?" in url:
        base, q = url.split("?", 1)
        keep = [kv for kv in q.split("&") if not kv.lower().startswith(("utm_", "fbclid=", "gclid="))]
        url = base + ("?" + "&".join(keep) if keep else "")
    url = re.sub(r"^http://", "https://", url, flags=re.I)
    m = re.match(r"^(https://)([^/?]+)(.*)$", url, flags=re.I)
    if m:  # scheme and host are case-insensitive; path and query are not
        host = m.group(2).lower()
        host = host[4:] if host.startswith("www.") else host
        url = "https://" + host + m.group(3)
    return url.rstrip("/")


def urls_in(value: Any) -> set[str]:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    from .cite import _elided, valid_url

    # An elided link ("https://loom.com/share/ad4…") names no page; it is not a source either.
    return {normalize_url(u) for u in _URL.findall(text) if valid_url(u) and not _elided(u)}


_ERROR_TEXT = re.compile(r"^\s*(?:Error\b|<tool_use_error>|Exit code [1-9])")


def is_error_result(structured: Any, content: Any, flagged: bool = False) -> bool:
    """A failed tool call is never evidence: not a read, not a write."""
    if flagged:
        return True
    if isinstance(structured, str) and _ERROR_TEXT.match(structured):
        return True
    if structured is None and _ERROR_TEXT.match(_text_of(content)):
        return True
    return False


def observe_tool(
    name: str,
    tool_input: dict[str, Any],
    structured: Any,
    content: Any,
    cwd: str | None,
    at: str | None = None,
    is_error: bool = False,
) -> Iterator[Observation]:
    """Turn one tool call + result into observations (hook payloads use the same shape)."""
    if is_error_result(structured, content, is_error):
        return
    if name == "Read":
        obs = from_read(tool_input, structured if structured is not None else _text_of(content), at)
        if obs:
            yield obs
    elif name == "Write":
        # Only a write the runtime confirms (create/update) is authorship; a rejected one is not.
        if isinstance(structured, dict) and structured.get("type") not in ("create", "update"):
            return
        if isinstance(structured, str):
            return
        obs = from_write(tool_input, at)
        if obs:
            yield obs
    elif name in ("Edit", "MultiEdit"):
        obs = from_edit(tool_input, structured, at)
        if obs:
            yield obs
    elif name == "Grep" and _persisted_preview(_text_of(content)) is not None:
        return  # a saved-to-disk grep result: only a cut preview was seen; no line evidence
    elif name == "Grep":
        unnumbered = tool_input.get("output_mode") == "content" and tool_input.get("-n") is False
        text = structured.get("content") if isinstance(structured, dict) else None
        text = text if isinstance(text, str) else _text_of(content)
        target = tool_input.get("path")
        single = None
        if target:
            tp = os.path.expanduser(target)
            tp = tp if os.path.isabs(tp) or not cwd else os.path.join(cwd, tp)
            single = os.path.normpath(tp) if _isfile(tp) else None
        base = os.path.dirname(single) if single else (target or cwd)
        root = os.path.expanduser(target) if target else cwd
        if root and not os.path.isabs(root) and cwd:
            root = os.path.join(cwd, root)
        roots = [_realpath(root)] if root else None
        if unnumbered:
            # Without line numbers a leading "2024-01-15 ..." is text, not line 2024.
            if single and text:
                yield Observation(single, 0, [], "Grep", at, line_numbers=[])
            return
        yield from from_grep_text(text, base, "Grep", at, single_file=single, alt_base=cwd, roots=roots)
    elif name in ("Bash", "Shell"):
        stdout = structured.get("stdout") if isinstance(structured, dict) else None
        stdout = stdout if isinstance(stdout, str) else _text_of(content)
        preview = _persisted_preview(_text_of(content))
        if preview is not None or (isinstance(structured, dict) and structured.get("persistedOutputPath")):
            stdout = preview or ""  # the model saw only the preview, not the saved full output
        cmd = expand_assignments(tool_input.get("command", ""))
        ecwd = effective_cwd(cmd, cwd)
        produced = list(from_shell_writes(cmd, ecwd, at)) + list(from_shell(_CD.sub("", cmd, count=1), stdout, ecwd, at))
        touched = list(from_shell_touches(cmd, ecwd, {o.path for o in produced}, at))
        touched += from_output_paths(stdout or "", {o.path for o in produced + touched}, at)
        if ecwd is None and _CD.match(cmd):
            # The command moved to a directory we cannot name: relative paths in it are not
            # evidence about any file we can name either.
            produced = [o for o in produced if os.path.isabs(o.path)]
            touched = [o for o in touched if os.path.isabs(o.path)]
        # A path still holding "$VAR" names a file only the agent's shell could resolve.
        yield from (o for o in produced if "$" not in o.path)
        yield from (o for o in touched if "$" not in o.path)


def iter_hook_events(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    for line in lines:
        line = line.strip()
        if line:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
