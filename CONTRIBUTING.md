# Contributing

1. Python 3.11+, no runtime dependencies in the core package.
2. Run the tests from a checkout: `PYTHONPATH=src python3 -m unittest discover -s tests -v`.
3. Keep fixtures synthetic. Never commit real paths, hostnames, credentials, or database identifiers.
4. One focused change per pull request, with tests.

## `.data-truth/`

The maintainers' own push tooling requires a small provenance manifest whenever the ledger, hook or
database code changes: which files changed (by hash), why, and which test proves it. Those manifests
live in `.data-truth/`. They are not part of the package and contributors do not need to touch them;
a maintainer refreshes them when landing your change.
