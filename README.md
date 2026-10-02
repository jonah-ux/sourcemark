# Sourcemark

**Citations that survive moves and edits — for agents and the humans who check them.**

Sourcemark anchors a citation to a quote, a file line, or a database value, then re-resolves it later: after the file moved, the line shifted, the text was edited, or the row changed. It also checks an agent's answer against what the agent actually read, so a citation to something it never opened is caught.

Python 3.11+ · Zero runtime dependencies · MIT · Local only · Early development

> Status: under active construction. The commands below land one pull request at a time; see [CHANGELOG.md](CHANGELOG.md).

## Why

Agents cite `file.py:42`. Two commits later line 42 is something else, the file was renamed, or the agent never read that file at all. Sourcemark stores enough about each citation — the exact quote, the text around it, its position, content fingerprints, and the git revision — to find it again or to say precisely what happened to it:

| Status | Meaning |
|---|---|
| `intact` | Found exactly where it was cited |
| `shifted` | Same file, different position |
| `moved` | Found in a different file |
| `edited` | Close match found; the text changed |
| `orphaned` | Not found anywhere it was looked for |
| `drifted` | A cited database value has changed |
| `deleted` | A cited database row no longer exists |

## License

MIT
