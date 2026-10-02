import os
import shutil
import subprocess
import tempfile
import unittest

from sourcemark.anchor import mark_lines
from sourcemark.gitinfo import clean_remote, source_for
from sourcemark.redact import find_secrets
from sourcemark.resolve import resolve

BODY = "".join(f"line {i}: the quick brown fox number {i} jumps over the lazy dog\n" for i in range(1, 60))


def git(cwd, *args):
    subprocess.run(["git", "-C", cwd, *args], check=True, capture_output=True)


class ResolveTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="sm-test-")
        git(self.dir, "init", "-q", "-b", "main")
        git(self.dir, "config", "user.email", "t@example.com")
        git(self.dir, "config", "user.name", "t")
        self.path = os.path.join(self.dir, "docs", "notes.txt")
        os.makedirs(os.path.dirname(self.path))
        with open(self.path, "w") as fh:
            fh.write(BODY)
        git(self.dir, "add", "-A")
        git(self.dir, "commit", "-q", "-m", "init")
        self.mark = mark_lines(BODY, 20, 21, source_for(self.path))

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def write(self, text, path=None):
        with open(path or self.path, "w") as fh:
            fh.write(text)

    def test_intact(self):
        r = resolve(self.mark)
        self.assertEqual((r.status, r.line_start, r.line_end), ("intact", 20, 21))

    def test_shifted(self):
        self.write("inserted\n" * 7 + BODY)
        r = resolve(self.mark)
        self.assertEqual((r.status, r.line_start), ("shifted", 27))

    def test_edited(self):
        self.write(BODY.replace("number 20 jumps", "number 20 leaps"))
        r = resolve(self.mark)
        self.assertEqual(r.status, "edited")
        self.assertEqual(r.line_start, 20)

    def test_git_rename(self):
        git(self.dir, "mv", "docs/notes.txt", "renamed.txt")
        git(self.dir, "commit", "-q", "-m", "mv")
        r = resolve(self.mark)
        self.assertEqual(r.status, "moved")
        self.assertTrue(r.path.endswith("renamed.txt"))

    def test_untracked_move_found_by_search(self):
        dest = os.path.join(self.dir, "elsewhere", "copy.md")
        os.makedirs(os.path.dirname(dest))
        shutil.move(self.path, dest)
        r = resolve(self.mark, roots=[self.dir])
        self.assertEqual(r.status, "moved")
        self.assertEqual(os.path.realpath(r.path), os.path.realpath(dest))

    def test_orphaned(self):
        self.write("nothing like the original\n")
        r = resolve(self.mark, roots=[self.dir])
        self.assertEqual(r.status, "orphaned")

    def test_secret_quote_not_stored(self):
        # Assembled at runtime so no secret-shaped literal ever sits in the repository.
        secret = "api" + "_key = '" + "abcd1234" + "efgh5678" + "ijkl'\n"
        doc = BODY + secret
        m = mark_lines(doc, 60, 60, source_for(self.path))
        self.assertIsNone(m.quote["exact"])
        self.assertTrue(m.redacted)
        self.write(doc)
        self.assertEqual(resolve(m).status, "intact")
        self.write(BODY + "api" + "_key = '" + "rotated0" * 2 + "'\n")
        self.assertEqual(resolve(m).status, "unverifiable")


class HelpersTest(unittest.TestCase):
    def test_clean_remote_strips_credentials(self):
        self.assertEqual(clean_remote("https://user:tok@github.com/o/r.git"), "github.com/o/r")
        self.assertEqual(clean_remote("git@github.com:o/r.git"), "github.com/o/r")

    def test_find_secrets(self):
        self.assertTrue(find_secrets("token ghp_" + "a" * 36))
        self.assertFalse(find_secrets("ordinary prose about tokens"))


if __name__ == "__main__":
    unittest.main()
