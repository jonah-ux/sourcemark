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


class AnchorPrecisionTest(unittest.TestCase):
    """Regression tests for adversarial anchoring findings."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="sm-anchor-")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def write(self, name, text):
        p = os.path.join(self.dir, name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(text)
        return p

    def test_deleted_short_line_does_not_jump_to_duplicate(self):
        body = "def alpha(x):\n    if x is None:\n        return None\n    return x+1\n\ndef beta(y):\n    if y<0:\n        return None\n    return y*2\n"
        p = self.write("dup.py", body)
        from sourcemark.anchor import TextSource
        m = mark_lines(body, 8, 8, TextSource(path=p))
        lines = body.split("\n")
        del lines[7]
        self.write("dup.py", "\n".join(lines))
        self.assertEqual(resolve(m, roots=[self.dir]).status, "orphaned")

    def test_blank_lines_cannot_be_cited(self):
        from sourcemark.anchor import TextSource
        with self.assertRaises(ValueError):
            mark_lines("first\n\nsecond\n", 2, 2, TextSource(path="/x"))

    def test_boilerplate_does_not_move_to_another_file(self):
        body = "import os\nfrom __future__ import annotations\nimport sys\n"
        p = self.write("a/c.py", body)
        self.write("b/other.py", "# other\nfrom __future__ import annotations\n# unrelated\n")
        from sourcemark.anchor import TextSource
        m = mark_lines(body, 2, 2, TextSource(path=p))
        self.write("a/c.py", "import os\nimport sys\n")
        self.assertEqual(resolve(m, roots=[self.dir]).status, "orphaned")

    def test_unicode_normalization_is_not_an_edit(self):
        import unicodedata
        body = "intro\nRésumé café naïve façade — the cited line\noutro\n"
        p = self.write("u.txt", unicodedata.normalize("NFC", body))
        from sourcemark.anchor import TextSource
        m = mark_lines(unicodedata.normalize("NFC", body), 2, 2, TextSource(path=p))
        self.write("u.txt", unicodedata.normalize("NFD", body))
        self.assertIn(resolve(m).status, ("intact", "shifted"))


class RedactionTest(unittest.TestCase):
    def test_formats_that_used_to_leak(self):
        cases = [
            "DB_PASS" + "WORD=Sup3rS3cretValue99",
            '{"pass' + 'word": "hunter2xx"}',
            "Authorization: Bearer " + "abcdefghijklmnopqrstuvwxyz012345",
            "glpat-" + "a" * 20,
            "host.example:5432:app:admin:" + "Pa55word!",
        ]
        for c in cases:
            self.assertTrue(find_secrets(c), c)

    def test_ordinary_code_is_not_redacted(self):
        for c in ['author = "Jonah"', "max_tokens: 4096", "password_hash = hash(password)", "export API_KEY=$FROM_VAULT"]:
            self.assertFalse(find_secrets(c), c)

    def test_secret_cut_at_context_edge_is_not_stored(self):
        from sourcemark.anchor import TextSource, mark_text
        tok = "ghp_" + "A" * 36
        doc = "x = 1\nprint('hello world, this is the cited line')\ntoken = '" + tok + "'\n"
        s = doc.index("print")
        e = doc.index("\n", s)
        m = mark_text(doc, s, e, TextSource(path="/x"), context=20)
        self.assertIsNone(m.quote["exact"])  # the 20-char suffix would hold a fragment of the token
