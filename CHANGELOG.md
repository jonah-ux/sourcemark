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
