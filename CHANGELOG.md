# Changelog

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
