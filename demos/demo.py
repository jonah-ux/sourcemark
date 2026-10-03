#!/usr/bin/env python3
"""Sourcemark in 30 seconds — fully synthetic, runs anywhere git is installed.

1. Cite two lines of a file in a fresh git repo.
2. Rename the file with `git mv`, add lines above, and edit a word inside the citation.
3. Resolve: Sourcemark follows the rename, finds the shifted lines, reports the edit.
4. Check an agent's answer against what that agent actually read: one real
   citation, one line it never read, one made-up file, one made-up link.

Prints a one-line JSON verdict at the end.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from sourcemark.anchor import mark_lines  # noqa: E402
from sourcemark.check import check_text  # noqa: E402
from sourcemark.check_export import export_check_v1  # noqa: E402
from sourcemark.gitinfo import source_for  # noqa: E402
from sourcemark.observe import Observation, Session  # noqa: E402
from sourcemark.resolve import resolve  # noqa: E402

LIGHTHOUSE = """# Harbor lighthouse operations

The lamp turns on thirty minutes before sunset.
The rotation period is ten seconds per full turn.
Keepers log fog-horn use in the brass ledger by the stairs.
Spare bulbs are stored in the cabinet marked B.
"""


def git(cwd: str, *args: str) -> None:
    subprocess.run(["git", "-C", cwd, *args], check=True, capture_output=True)


def main() -> int:
    box = tempfile.mkdtemp(prefix="sourcemark-demo-")
    try:
        git(box, "init", "-q", "-b", "main")
        git(box, "config", "user.email", "demo@example.com")
        git(box, "config", "user.name", "demo")
        path = os.path.join(box, "docs", "lighthouse.md")
        os.makedirs(os.path.dirname(path))
        with open(path, "w") as fh:
            fh.write(LIGHTHOUSE)
        git(box, "add", "-A")
        git(box, "commit", "-q", "-m", "add notes")

        mark = mark_lines(LIGHTHOUSE, 4, 5, source_for(path))
        print(f"1. cited {mark.token}  docs/lighthouse.md:4-5")

        os.makedirs(os.path.join(box, "ops"))
        git(box, "mv", "docs/lighthouse.md", "ops/harbor-lighthouse.md")
        moved = os.path.join(box, "ops", "harbor-lighthouse.md")
        with open(moved) as fh:
            text = fh.read()
        text = "<!-- reviewed -->\n\n" + text.replace("ten seconds", "twelve seconds")
        with open(moved, "w") as fh:
            fh.write(text)
        git(box, "add", "-A")
        git(box, "commit", "-q", "-m", "rename and update")
        print("2. renamed to ops/harbor-lighthouse.md, added 2 lines above, edited a word inside")

        res = resolve(mark, roots=[box])
        rel = os.path.relpath(os.path.realpath(res.path), os.path.realpath(box)) if res.path else None
        print(f"3. resolve -> {res.status} at {rel}:{res.line_start}-{res.line_end} (similarity {res.similarity:.2f})")

        sess = Session(cwd=box)
        sess.add(Observation(moved, 1, text.split("\n"), "Read"))
        sess.urls.add("https://example.com/harbor-guide")
        answer = (
            "The lamp rotates every twelve seconds (ops/harbor-lighthouse.md:6). "
            "Bulbs are replaced weekly (ops/harbor-lighthouse.md:40). "
            "Keepers sign in at ops/keepers-roster.md:3. "
            "See [the guide](https://example.com/harbor-guide) and [the spec](https://example.com/made-up-spec)."
        )
        rep = check_text(answer, sess)
        print("4. checking an agent's answer against what it read:")
        for c in rep.checks:
            print(f"   {c.verdict:15} {c.raw}")

        export = export_check_v1(rep, sess)
        print(f"5. sanitized check export -> {export['schema']} ({export['state']})")

        verdict = {
            "resolve": res.status,
            "moved": res.moved,
            "line": res.line_start,
            "citations": rep.counts(),
            "check_export": {
                "schema": export["schema"],
                "state": export["state"],
                "counts": export["counts"],
            },
            "ok": res.status == "edited" and res.moved and rep.counts().get("verified") == 1,
        }
        print(json.dumps(verdict, sort_keys=True))
        return 0 if verdict["ok"] else 1
    finally:
        shutil.rmtree(box, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
