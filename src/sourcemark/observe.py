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
        return {os.path.basename(o.path) for o in self.observations}

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
    r"(?:cat|tee)\s+(?:-a\s+)?>?\s*(?P<path>[^\s<>|;&]+)\s*<<-?\s*['\"]?(?P<tag>\w+)['\"]?[^\n]*\n(?P<body>.*?)\n\s*(?P=tag)\b",
    re.S,
)
_REDIRECT = re.compile(r"(?:^|[\s;&|])(?:>|>>|tee\s+(?:-a\s+)?)\s*(?P<path>[~/.\w][^\s<>|;&]*)")


_CD = re.compile(r"^\s*cd\s+(?P<dir>[^\s;&|]+)\s*(?:&&|;)")
_PATHISH = re.compile(r"(?<![\w@:/])(?:~?/|\.{1,2}/)?(?:[\w.\-]+/)*[\w.\-]+\.[A-Za-z][A-Za-z0-9]{0,7}(?![\w/])")


_ASSIGN = re.compile(r"(?:^|[;&|\n(]\s*)(?:export\s+)?(?P<name>[A-Za-z_]\w*)=(?P<val>\"[^\"$`]*\"|'[^']*'|[^\s;&|$`\"']+)(?=[\s;&|)]|$)")


def expand_assignments(command: str) -> str:
    """Substitute ``$NAME`` / ``${NAME}`` for plain ``NAME=value`` assignments made in the command."""
    vals: dict[str, str] = {}
    for m in _ASSIGN.finditer(command):
        vals[m.group("name")] = m.group("val").strip("'\"")
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

    for m in _HEREDOC.finditer(command):
        p = absolute(m.group("path"))
        seen.add(p)
        out.append(Observation(p, 1, m.group("body").split("\n"), "Bash-write", at))
    for m in _REDIRECT.finditer(command):
        raw = m.group("path")
        if raw.startswith(("/dev/", "&")) or raw in ("-",):
            continue
        p = absolute(raw)
        if p not in seen:
            seen.add(p)
            # Content unknown: record a file-level observation with no lines.
            out.append(Observation(p, 0, [""], "Bash-redirect", at, line_numbers=[]))
    for seg in re.split(r"\s*(?:&&|;|\|\||\n)\s*", command):
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


def gh_refs(command: str) -> set[str]:
    """``gh pr view 12 --repo o/r`` that succeeded: the session looked at o/r#12 on GitHub."""
    urls: set[str] = set()
    for seg in re.split(r"\s*(?:&&|;|\|\||\||\n)\s*", command):
        ref, repo = _GH_REF.search(seg), _GH_REPO.search(seg)
        if ref and repo:
            for kind in ("pull", "issues"):  # GitHub serves a PR under both
                urls.add(normalize_url(f"https://github.com/{repo.group('repo')}/{kind}/{ref.group('num')}"))
    return urls


def _grep_targets(args: list[str], cwd: str | None) -> list[str] | None:
    operands = [a for a in args if not a.startswith("-")]
    paths = operands if ("-e" in args or "--regexp" in args) else operands[1:]
    out = []
    for t in paths or ([cwd] if cwd else []):
        t = os.path.expanduser(t)
        t = t if os.path.isabs(t) or not cwd else os.path.join(cwd, t)
        out.append(_realpath(t))
    return out or None


def _single_target(args: list[str], cwd: str | None) -> str | None:
    """The one regular file a grep/rg invocation searched, if it searched exactly one."""
    if any(a in ("-r", "-R", "--recursive") or (a.startswith("-") and not a.startswith("--") and "r" in a[1:] and a[1:].isalpha()) for a in args):
        return None
    operands = [a for a in args if not a.startswith("-")]
    if "-e" in args or "--regexp" in args:
        files = operands  # pattern supplied via -e; every operand is a path
    else:
        files = operands[1:]  # first operand is the pattern
    if len(files) != 1:
        return None
    p = os.path.expanduser(files[0])
    p = os.path.normpath(p if os.path.isabs(p) or not cwd else os.path.join(cwd, p))
    return p if _isfile(p) else None


# Segments that print nothing: they do not share the stdout with the command that read a file.
_SILENT = re.compile(r"^(?:(?:export\s+)?[A-Za-z_]\w*=(?:\"[^\"]*\"|'[^']*'|\S*)|cd(?:\s+\S+)?|set\s+[-+]\w+)$")


def from_shell(command: str, stdout: str, cwd: str | None, at: str | None = None) -> list[Observation]:
    """Recognize simple file-printing commands whose stdout is a known file slice."""
    out: list[Observation] = []
    owner: dict[int, str] = {}  # index in ``out`` -> the segment that printed it
    for segment in re.split(r"\s*(?:&&|;|\|\|)\s*", command):
        seg = segment.strip()
        if "|" in seg:
            first = seg.split("|", 1)[0].strip()
            if first.startswith(("rg ", "grep ")):
                try:
                    fargs = shlex.split(first)[1:]
                except ValueError:
                    fargs = []
                return from_grep_text(stdout, cwd, "Bash", at, single_file=_single_target(fargs, cwd), roots=_grep_targets(fargs, cwd))
            continue
        try:
            argv = shlex.split(seg)
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
        numbered = "--line-number" in args or any(
            a.startswith("-") and not a.startswith("--") and a[1:].isalpha() and "n" in a[1:] for a in args
        )
        if cmd in ("rg", "grep") and (numbered or cmd == "rg"):
            return from_grep_text(stdout, cwd, "Bash", at, single_file=_single_target(args, cwd), roots=_grep_targets(args, cwd))
        if cmd not in ("cat", "sed", "head", "tail", "nl") or len(files) != 1:
            continue
        path = files[0] if os.path.isabs(files[0]) or not cwd else os.path.join(cwd, files[0])
        path = os.path.expanduser(path)
        lines = stdout.split("\n")
        if lines and lines[-1] == "":
            lines = lines[:-1]
        start = 1
        if cmd == "sed":
            m = re.search(r"(\d+),(\d+)p", seg)
            if not m or "-n" not in args:
                continue
            start = int(m.group(1))
        elif cmd == "tail":
            continue  # start line unknown without the file length; skip rather than guess
        elif cmd == "nl":
            lines = [re.sub(r"^\s*\d+\t", "", x) for x in lines]
        out.append(Observation(os.path.normpath(path), start, lines, "Bash", at))
        owner[len(out) - 1] = segment
    segments = [x for x in re.split(r"\s*(?:&&|;|\|\|)\s*", command) if x.strip() and not _SILENT.match(x.strip())]
    if len(out) > 1 or (out and len(segments) > 1):
        split = _split_by_echo_markers(segments, stdout, {owner[i].strip(): o for i, o in enumerate(out)})
        if split is not None:
            return split
        # Several things printed into one stdout: lines cannot be attributed; keep file-level only.
        return [Observation(o.path, 0, [], "Bash-touch", at, line_numbers=[]) for o in out]
    return out


def _split_by_echo_markers(segments: list[str], stdout: str, printers: dict[str, Observation]) -> list[Observation] | None:
    """``echo "--- a"; sed -n 1,9p a; echo "--- b"; cat b``: each literal echo line in stdout
    fences off the output of the single command after it. Anything ambiguous returns None."""
    lines = stdout.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    groups: list[list[str]] = [[]]
    markers: list[str] = []
    for seg in (x.strip() for x in segments):
        m = re.fullmatch(r"echo\s+(?:\"([^\"$`\\]*)\"|'([^']*)')", seg)
        if m:
            markers.append(m.group(1) if m.group(1) is not None else m.group(2))
            groups.append([])
        else:
            groups[-1].append(seg)
    if not markers:
        return None
    # Locate each marker line in order.
    pos, bounds = 0, []
    for mk in markers:
        try:
            i = lines.index(mk, pos)
        except ValueError:
            return None
        bounds.append(i)
        pos = i + 1
    edges = [-1, *bounds, len(lines)]
    result: list[Observation] = []
    for g, segs in enumerate(groups):
        chunk = lines[edges[g] + 1 : edges[g + 1]]
        found = [printers[x] for x in segs if x in printers]
        if not found:
            continue
        o = found[0]
        if len(segs) == 1 and len(found) == 1:
            result.append(Observation(o.path, o.line_start, chunk, "Bash", o.at))
        else:
            result.extend(Observation(f.path, 0, [], "Bash-touch", f.at, line_numbers=[]) for f in found)
    return result


# --- transcript readers ---------------------------------------------------------


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
                    if not is_error_result(tur, c.get("content"), bool(c.get("is_error"))):
                        # A URL inside an error ("fetch failed: https://...") was not obtained.
                        sess.urls |= urls_in(tur if tur is not None else c.get("content"))
                        if _is_web_tool(name):
                            sess.urls |= urls_in(tin)
                        if name in ("Bash", "Shell") and isinstance(tin, dict):
                            sess.urls |= gh_refs(str(tin.get("command", "")))
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


def _is_web_tool(name: str) -> bool:
    n = name.lower()
    return n in ("webfetch", "websearch") or ("fetch" in n or "search" in n or "browse" in n) and n.startswith("mcp__")


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(x.get("text", "") for x in content if isinstance(x, dict))
    return ""


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
            nums.append(n)
            texts.append(ln[1:] if ln[:1] in ("+", " ") else ln)
            n += 1
    if not nums:
        return None
    return Observation(path, 0, texts, "Edit", at, line_numbers=nums)


_URL = re.compile(r"https?://[^\s\"'<>()\[\]{}|\\^`]+")


def normalize_url(url: str) -> str:
    url = url.strip().rstrip(".,;:!?)]}>'\"*_`")
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
    from .cite import valid_url

    return {normalize_url(u) for u in _URL.findall(text) if valid_url(u)}


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
    elif name == "Grep":
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
        yield from from_grep_text(text, base, "Grep", at, single_file=single, alt_base=cwd, roots=roots)
    elif name in ("Bash", "Shell"):
        stdout = structured.get("stdout") if isinstance(structured, dict) else None
        stdout = stdout if isinstance(stdout, str) else _text_of(content)
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
