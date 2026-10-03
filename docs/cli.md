# CLI reference

Every command accepts `--json` (one JSON document on stdout) and the global `--ledger PATH`
(default `$SOURCEMARK_HOME/ledger.db`, `SOURCEMARK_HOME` defaults to `~/.sourcemark`).

Exit codes: `0` ok · `1` a check, resolve, or verification found a problem · `2` usage error.

## `sourcemark mark FILE[:LINE[-LINE]] [--quote TEXT]`

Cite whole lines, or an exact quote inside the file. Prints the short token (`[sm:xxxxxxxxxx]`)
and stores the full mark in the ledger. Git provenance (repository, commit, path, blob) is
recorded when the file is tracked; credentials are stripped from remote URLs.

## `sourcemark mark-row [SCHEMA.]TABLE PK COLUMN... [--dsn DSN] [--database LABEL]`

Cite columns of one PostgreSQL row. `PK` is a value for single-column keys, or `col=val,col=val`
for composite keys. Runs read-only through `psql`; DSN from `--dsn`, `$SOURCEMARK_DSN`, or
`$DATABASE_URL`. Sensitive-looking columns are stored as fingerprints only.

## `sourcemark resolve REF [--root DIR]... [--dsn DSN]`

Find a mark again. `REF` is a mark id, a token (`[sm:…]` or the bare 10 characters), or a unique
prefix. For files, `--root` adds directories to search when the file moved outside git.
Exit 0 for `intact`/`shifted`/`moved`, 1 otherwise.

## `sourcemark show REF`

Print a stored mark (quote text, or `[redacted]` / `[fingerprint only]`).

## `sourcemark check TRANSCRIPT [--all] [--now] [--text FILE] [--export v1]`

Check citations against what a Claude Code session or a Codex rollout read and wrote (the format
is detected; see the README's Codex section for how Codex cells are read). By default checks the last
turn (all assistant text since the latest human prompt). `--all` checks every assistant message;
`--now` also reports whether cited lines changed since they were read; `--text` checks the text
in FILE against the session instead of the transcript's own reply. Exit 1 when any citation is
unsupported (`delegated` is reported but is not a failure).

### `--export v1`

The explicit export flag emits one sanitized `sourcemark/check/v1` JSON envelope, even without
`--json`. Its fields are `schema`, a bounded `state` (`ok`, `observed`, `partial`, or
`timed_out`), fixed scalar counts (`total`, `passing`, `failing`, `unknown`, `observations`, and
`timed_out`), and `sha256:` policy/session identities. It omits paths, quotes, transcript text,
URLs, raw mark tokens, and ledger contents. Export checks the records it actually parses,
using one open transcript stream. Malformed or non-object JSONL records and invalid UTF-8
are refused with exit 2 and no export. A malformed or unreadable delegated Claude transcript
also refuses the export. The ordinary check keeps its existing tolerant JSONL handling.
The ordinary `check --json` output remains the detailed local report.

## `sourcemark verify-ledger`

Recompute the ledger's hash chain. Exit 1 and the first broken sequence number if any stored
event was altered or removed.

## `sourcemark hook stop`

Read a Stop-hook event (JSON) on stdin. See [hooks.md](hooks.md).
