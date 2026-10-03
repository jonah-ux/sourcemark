# AGENTS.md

Instructions for AI agents using or modifying Sourcemark.

## Using it

- Cite as `path:line` or `path:start-end` relative to the repository root, and **read before you cite**.
  Quote code in backticks next to the citation it supports.
- Check your own answer before sending it: `sourcemark check <transcript.jsonl> --json` (exit 1 means
  at least one citation is not backed by what you read). Fix or remove each flagged citation.
- To cite something durably: `sourcemark mark path:12-14 --json` returns a token you can paste as
  `[sm:xxxxxxxxxx]`; `sourcemark resolve [sm:xxxxxxxxxx] --json` finds it later.
- `delegated` means a subagent read it, not you. Open the source yourself if the claim matters.

## Changing it

- Python 3.11+, standard library only in `src/`.
- Run `PYTHONPATH=src python3 -m unittest discover -s tests -v` before every commit.
- Fixtures are synthetic. Never commit real paths, hostnames, transcripts, credentials, or database identifiers.
- Accuracy changes need evidence: say which behaviour moved and by how much.
