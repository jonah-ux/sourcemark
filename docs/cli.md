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

## `sourcemark check TRANSCRIPT [--all] [--now] [--text FILE]`

Check citations against what a Claude Code session read and wrote. By default checks the last
turn (all assistant text since the latest human prompt). `--all` checks every assistant message;
`--now` also reports whether cited lines changed since they were read; `--text` checks the text
in FILE against the session instead of the transcript's own reply. Exit 1 when any citation is
unsupported (`delegated` is reported but is not a failure).

## `sourcemark verify-ledger`

Recompute the ledger's hash chain. Exit 1 and the first broken sequence number if any stored
event was altered or removed.

## `sourcemark hook stop`

Read a Stop-hook event (JSON) on stdin. See [hooks.md](hooks.md).
