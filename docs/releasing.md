# Releasing

Semantic versions. Until 1.0, minor versions may change interfaces; patch versions keep documented
commands and JSON shapes. Each release has an annotated tag, a changelog entry, a wheel, a source
distribution, and a `SHA256SUMS` file attached to the GitHub release.

1. Update `pyproject.toml`, `src/sourcemark/__init__.py`, and `CHANGELOG.md` together.
2. CI must be green on Linux and macOS for every supported Python version.
3. Build locally with `python3 -m build --sdist --wheel`; install each artifact into a fresh
   virtual environment and run `sourcemark --version`, `sourcemark --help`, and `python3 demos/demo.py`.
4. Scan the tree and full git history for secrets; review the distributed file list.
5. Tag the reviewed commit: `git tag -a vX.Y.Z -m 'Sourcemark X.Y.Z'` and push the tag.
6. The `release` workflow verifies the tag matches the package version, builds, checksums, installs
   both artifacts in clean environments, runs the demo, and publishes the release.
7. Download an artifact from the published release, verify it against `SHA256SUMS`, install it in a
   fresh environment, and run the CLI. A local build is not proof that the published release works.

No PyPI publication is implied by a GitHub release; the README installs from the versioned tag.
