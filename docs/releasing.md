# Releasing

Semantic versions. Until 1.0, minor versions may change interfaces; patch versions keep documented
commands and JSON shapes. Each release has an annotated tag, a changelog entry, a wheel, a source
distribution, and a `SHA256SUMS` file attached to the GitHub release.

1. Update `pyproject.toml`, `src/sourcemark/__init__.py`, and `CHANGELOG.md` together.
2. CI must be green on Linux and macOS for every supported Python version.
3. Build locally with `python3 -m build --sdist --wheel`; install each artifact into a fresh
   virtual environment and run `sourcemark --version`, `sourcemark --help`, and
   `env -u PYTHONPATH <venv>/bin/python demos/demo.py --installed`. The installed flag prevents
   the demo from adding the source checkout to its import path; missing installations must fail.
4. Scan the tree and full git history for secrets; review the distributed file list.
5. Tag the reviewed commit: `git tag -a vX.Y.Z -m 'Sourcemark X.Y.Z'` and push the tag.
6. The `release` workflow verifies the tag matches the package version, builds, checksums, installs
   both artifacts in clean environments, runs the installed-package demo, and publishes the release.
   It also requires an annotated tag identifying the checked-out commit, runs the source suite,
   checks each consumer's installed package/version identity outside the checkout, and runs the
   export regression suite and exact demo-result assertions before publication. Manual dispatch
   checks out the requested tag, rather than building an unrelated default-branch commit.
7. Download an artifact from the published release, verify it against `SHA256SUMS`, install it in a
   fresh environment, and run the CLI. A local build is not proof that the published release works.

No PyPI publication is implied by a GitHub release; the README installs from the versioned tag.

The synthetic demo deliberately contains three unsupported citations. Its expected export state
is `partial`, with two passing and three failing citations; publication checks require those
refusals rather than treating the entire answer as verified.
