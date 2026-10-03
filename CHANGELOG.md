# Changelog

## 0.5.1 — 2026-10-03

- Verify CI/release demos against the installed package with `--installed`, without the
  source-checkout import fallback.
- Validate export records during the actual transcript read instead of a separate preflight.
  Reject malformed, non-object, invalid UTF-8, or unreadable delegated evidence without an
  export, while preserving the ordinary transcript readers' handling of incomplete logs.
- Capture the root export input in a private temporary snapshot so an in-place rewrite
  cannot change the format decision or omit citations. In-memory buffering uses a 1 MiB
  threshold; temporary files may hold the full transcript and are cleaned on success or failure.

## 0.5.0 — 2026-10-02

Learned from the real Codex citations the independent oracle saw but the checker flagged. Every fix has a regression test built from the real cell shape.

Measured against v0.4.0 on the same real data:

- **Stored oracle-seen Codex citations:** 8,471, of which 3,018 now verify (995 before).
- **Codex hard bench, 600 rollouts:** agreement with the oracle 72.87% → 77.69%, recall 99.61% → 99.83%, 0 false flags.
- **Real-verdict A/B, 16,203 Codex and 1,092 Claude Code citations:** 968 Codex verdicts improved and 0 became newly failing. Each verdict change was inspected; every new verification was checked against the file on disk.
- **Claude Code hard bench:** recall 99.67% → 99.89%, 0 false flags.
- **Resolver history:** no changes.

### Codex cells that run several commands

- **Lists of commands.** Three kinds of cell are now read command by command:
  - a tuple list mapped over: `cmds.map(([name, cmd]) => tools.exec_command({cmd}))`;
  - a list of paths put into a command template: `` `nl -ba ${JSON.stringify(path)} | sed -n '1,120p'` ``;
  - literal calls written out in `Promise.all([...])`.
- **Finding each command's output.** Three ways, in order:
  - JSON results are paired with their command;
  - otherwise the output is split at a header line per command (`=== name ===`, `--- path ---`, `---RESULT 2---`). The header shape is learned from the output: it must match exactly one whole line per command, in order;
  - a cell that only prints each output, one after another, is read as the commands joined in sequence.
- **What counts.** Lines that carry their own numbers always count. A plain slice counts only when the output is known to be exactly the command's, and uncut.

### Shell evidence (Claude Code and Codex)

- **`nl -ba` / `cat -n` slices split by what each can print.** Several slices printed one after another are now told apart even when the numbers keep rising: each starts at its `sed` range, runs on line by line, and jumps only between its own ranges. Beside other commands (a grep, a jq), only lines in the exact six-column `nl` shape count, and only when no other command could print numbered lines.
- **A one-file grep printed first or last** owns the ascending run of `N:hit` lines at that end of the output, when nothing else in the command prints bare numbered lines.
- **`bash -lc '...'`, `sh -c "..."` and `worker-lifecycle run ... --`** are read as the script they run.
- **Literal `for` loops are unrolled**: `for f in a b; do echo "== $f"; nl -ba "$f"; done` is read as the commands it ran. Loops over globs, command output or `${f%...}` expansions are not.

### Checker

- **A relative path read in several copies** (a repo and its worktrees): when the chosen copy's text does not match a quote, the citation is judged against the other read copies that cover the cited lines. An absolute path still names one file.
- **Line numbers survive echo fences.** A `sed -n A,Bp f` that shares a fenced chunk with another command keeps the line numbers it printed. Before, the fence split dropped them, which the old fallback kept.

### Faster

- The command splitter skips plain runs in one step and caches its results: 0 differences on 32,915 real commands.
- Line counts are cached per file version.
- Reading 300 recent Codex rollouts took 117 s instead of 165 s; the slowest dropped from 17 s to 10 s.

### Evidence export

- Add the explicit `check --export v1` boundary projection for sanitized citation/read-evidence
  results. The versioned envelope carries only bounded state, fixed scalar counts, and hashed
  policy/session identities; malformed input is refused without emitting report details.

## 0.4.0 — 2026-10-02

Learned from the real Claude Code citations the independent oracle saw but the checker flagged `unread_lines`: 55 open cases are down to 26. Every fix has a regression test built from the real failing shape.

Measured against v0.3.1 on the newest real sessions:

- **Codex:** 108 false `quote_mismatch` flags removed across 15,866 citations.
- **Claude Code:** 0 newly failing verdicts across 1,086 citations.
- **Hard bench, 600 Claude Code sessions:** recall 99.77% → 99.89%, changed-word quotes caught 99.66% → 100%, agreement with the oracle 91.61% → 92.45%, 0 false flags.
- **Codex hard bench:** 0 false flags, recall unchanged.
- **Resolver history:** no changes.

One citation became newly failing: a runtime value written next to a citation ("leaving `serving_pid=0`") is now taken as a quote of the cited line. v0.3.1 avoided that only because its broken backtick pairing dropped the quote.

- **`git show REV:F` and `cat F` count as file sources.** `git show origin/main:F | grep -n`, `| sed -n 'A,Bp'` and `| head` now give line evidence for `F`. `REV:F` resolves from the top of the work tree, `REV:./F` from the current directory. Hundreds of these pipelines appear in recent sessions.
- **`cd` part-way through a command is followed.** A plain `cd DIR` segment moves later relative paths. A `cd` to somewhere unknown (`cd "$d"`, `cd -`, inside a subshell, loop or `if`, `pushd`) still drops the command to file-level evidence.
- **A one-file `grep` keeps its hits after the file is gone** (a removed worktree, a deleted temp file). Only plain `grep` on one non-glob operand is trusted: it never descends into a directory without `-r`.
- **Option clusters with a value:** `-n` is recognised in `git grep -nA5` and `grep -rnC2`. Before, the output of these greps was ignored.
- **Glob search roots:** hits of `grep -rn PAT ~/x/*/lib/f.py` are no longer dropped as out of scope.
- **Code spans pair correctly.** A backticked span shorter than 4 or longer than 200 characters made the extractor pair the wrong backticks. It read the prose between two spans as code and lost the quote that followed, so a misquote after a long path went unchecked.
- **Links written as code are not quotes.** Codex writes citations as `` `[a.py:24](/abs/a.py:24)` ``. Each such span was taken as a quote for the citation next to it, so correct citations were flagged `quote_mismatch`.
- **Shorthand quotes:** `` `fn()` `` naming a function, and `` `Line: **Gate 2**` `` bolding a prefix of the line, are no longer flagged `quote_mismatch`.

## 0.3.1 — 2026-10-02

- **Misquoted names are caught (#69, #70).** A quote that is a single name is flagged only when it occurs nowhere in what was read of the file and a near-identical name was read (`forbidden_patternsX` for `forbidden_patterns`). Plural, separator and case variants, file names and paths, and text known only by line number are never flagged. 0 verdict changes over 17,759 real citations.
- **Timestamped backup paths are extracted:** `config.yaml.bak-20260902-154450:80`.
- **Lines inserted inside a cited block (#71, #72)** now resolve as `edited` instead of `orphaned`, when every quoted line survives in order within a modest spread.

## 0.3.0 — 2026-10-02

### Codex rollouts (#65, #66)

`sourcemark check` reads Codex rollouts (`~/.codex/sessions/.../rollout-*.jsonl`) as well as Claude Code transcripts; the format is detected. Codex tools run from JavaScript cells, so cells are read conservatively:

- A literal one-call cell is read like a shell command.
- Multi-command cells credit only self-describing lines.
- Cut outputs keep only lines that carry their own numbers.
- Failed or computed commands give file-level evidence only.

On 2,973 real rollouts with 47,394 real citations: 0.00% false flags, every injected near-miss caught.

### Self-numbered evidence

`nl -ba F | sed -n 'A,Bp'` and `cat -n F | head` lines carry their own line numbers, so they now count as line evidence in Claude Code sessions too. Several slices are split only where the numbering restarts, one restart per command.

## 0.2.0 — 2026-10-02

Learned from failures on much larger real benchmarks: 4,691 real git-history edits from 4 repositories, and all 3,398 local agent sessions with 8,000+ injected near-miss citations. Every fix has a regression test built from the real failing shape.

### Resolver: follow lines through the diff (#61, #62)

When the file was clean at mark time, the mark's `git_blob` is aligned against the current file, the way git follows a line. This is used only where the text search is unsure:

- **Duplicated quotes** resolve to the copy the marked line became, not the copy nearest the old line number.
- **Edits whose surroundings also changed** resolve as `edited`; previously they were `orphaned`.
- **An identical copy descended from a different line** is no longer taken for the cited one.
- **Redacted marks** record their column and are found again after a move by hash, so no text is stored.
- **Unique short lines** whose neighbour was edited stay found.

Real-history accuracy went from 98.52% to 99.53%, with unchanged lines at 100%. The mutation benchmark rose from 96.7% to 97.3% (seed 1) and from 96.6% to 97.4% (seed 2).

### Checker: evidence in real shell commands (#59, #60)

- Commands are split on quote-aware separators, so a grep pattern containing `&&` stays one argument.
- Each line of a multi-line script is a command, and heredoc bodies are skipped.
- Greps fenced by `echo` markers are attributed.
- `timeout`, `env`, `nice`, `time` and `stdbuf` wrappers are unwrapped.
- A leading `cd DIR` followed by a newline is followed.
- `~` is expanded before the cwd join, which fixes `<cwd>/~/…` paths.
- Dotfile citations (`.gitignore:7`) are extracted.

On real sessions, false flags stayed at 0.00% and every near-miss injection kind is still caught at 100%.

## 0.1.0 — 2026-10-02

First release.

- **Text anchoring**: marks with quote + context selectors, positions, fingerprints, and git provenance;
  re-resolution through the original position, git renames, and search roots; statuses `intact`,
  `shifted`, `moved`, `edited`, `orphaned`, `unverifiable`. Secret-shaped quotes are never stored.
- **Citation checking**: observations from Read / Write / Edit / Grep / shell slices / heredoc writes /
  URLs in any tool output; citations in `path:line`, `path#L…`, markdown, bare and markdown URLs, and
  `[sm:…]` tokens; verdicts from `verified` to `url_unsourced`.
- **Delegated evidence**: subagent transcripts are loaded; relayed citations are `delegated`.
- **Database citations** (PostgreSQL, read-only): `intact`, `drifted` (with changed columns), `deleted`.
- **CLI**: `mark`, `mark-row`, `resolve`, `show`, `check`, `verify-ledger`, `hook stop`.
- **Ledger**: append-only, hash-chained SQLite with tamper detection.
- **Claude Code Stop hook** with `shadow`, `warn`, `enforce` modes; fails open; never loops.

### Hardening before release

Two independent adversarial reviews and a loop over 120 real agent sessions found and fixed:

- **Checker false passes**:
  - linter output read as grep hits
  - unnumbered `rg` output read as line numbers
  - multi-range `sed`
  - `tee -a`/`>>` appends recorded as lines 1..N
  - quotes in markdown link labels never checked
  - grep option values taken as paths
  - URLs from the agent's own writes and from subagent reports
  - URLs with parentheses
  - an ambiguous `[sm:…]` token that turned the whole check off
- **Checker false flags on real sessions**:
  - copies of a file in visited directories
  - PRs opened (`pr-link`) or checked (`gh pr view N --repo`)
  - shell variables (`W=…; sed … $W/f`)
  - several reads in one command split by `echo` markers
  - `cp` destinations
  - lines seen before a file shrank
  - paths with spaces or route folders (`[id]`, `(group)`)
  - placeholder links
- **Resolver confident-wrong answers**:
  - loose matches inside other lines
  - duplicates of the cited line
  - decoy copies under the search roots
  - license headers "moving"
  - a deleted line's look-alike neighbour sliding into place
- **Redaction**:
  - compose/properties/ini/`my.cnf` values
  - `curl -u` and API-key headers
  - `mysql -p`
  - inline `PGPASSWORD=`
  - connection strings
  - secrets in URL query strings
  - Stop-hook check events are now redacted before storage
- **Ledger**:
  - event timestamps are chained
  - one `mark` event per mark (`append()` refuses `mark`)
  - deleted mark rows and rewritten token indexes are detected
- **Performance**:
  - long fuzzy matches go line by line (21 s → under 2 s for a 12k-character block)
  - the orphan search is bounded
  - a catastrophic URL-host regex is gone (82 s → 0.8 s on a real 8 MB session)
  - paths from other machines skip the macOS automounter
- **CLI**:
  - usage errors exit 2
  - `hook stop` honours `--ledger`
  - `python -m sourcemark` works

A third review found and fixed:

- **Only what the model saw counts as evidence**:
  - the `<persisted-output>` preview, not the saved full output
  - an Edit's written lines, not its patch context
  - no line credit from a Read past the end
- **Shell parsing**:
  - chained single-file greps
  - echo markers that are blank or occur inside printed files
  - a `cd` part-way through a command
  - reassigned variables
  - redirection tokens
  - `--color`
- **Citations**:
  - `path:line` inside a link label is checked
  - relayed (subagent) citations get quote and coverage checks
  - background-task notices no longer start a new turn
- **Self-sourced links**: `TodoWrite`, `echo`, and `gh` text that was never run
- **Tokens**: tokens are looked up by `check`; an unusable ledger fails tokens instead of skipping them
- **Resolver**:
  - a line extended in place is `edited`, not `intact`
  - an in-place edit beats an old copy elsewhere
  - swapped duplicates follow their context
- **Secrets**:
  - psql errors no longer carry the DSN
  - more URL params, Telegram, netrc, npmrc, docker and connection-string shapes
  - redaction is linear on long lines
- **Ledger**: deleting the head/count rows with trailing events is caught through the event sequence
