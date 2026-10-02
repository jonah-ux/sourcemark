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

    def test_unique_distinctive_line_moves_to_another_file(self):
        body = "head\nretry_budget = compute_backoff(attempts, ceiling)\ntail\n"
        p = self.write("a/src.py", body)
        from sourcemark.anchor import TextSource
        m = mark_lines(body, 2, 2, TextSource(path=p))
        self.write("a/src.py", "head\ntail\n")
        dest = self.write("b/pasted.md", "notes\nretry_budget = compute_backoff(attempts, ceiling)\nmore\n")
        r = resolve(m, roots=[self.dir])
        self.assertEqual((r.status, r.path, r.line_start), ("moved", dest, 2))

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

    def test_remaining_patterns(self):
        import hashlib
        cases = {
            "cli_secret": "mysql --pass" + "word=Hunter2Hunter2 -h db",
            "yaml_secret": "  pass" + "word: Hunter2Hunter2\n",
            "npm_token": "npm_" + "a1B2" * 9,
            "huggingface_token": "hf_" + "a1B2c3" * 6,
            "sendgrid_key": "SG." + "a1B2c3d4" * 3 + "." + "e5F6g7h8" * 3,
            "key_material": "Ab1" * 30,
        }
        for name, text in cases.items():
            self.assertIn(name, find_secrets(text), text)
        self.assertFalse(find_secrets(hashlib.sha256(b"x").hexdigest() * 2))

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


class AdversarialRound2ResolveTest(unittest.TestCase):
    """Confident wrong answers and slow paths found by the second adversarial review."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="sm-res2-")
        self.addCleanup(shutil.rmtree, self.dir, True)

    def write(self, name, text):
        p = os.path.join(self.dir, name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as fh:
            fh.write(text)
        return p

    def mark(self, p, a, b):
        from sourcemark.anchor import TextSource
        with open(p) as fh:
            return mark_lines(fh.read(), a, b, TextSource(path=p))

    def test_loose_match_does_not_land_inside_another_line(self):
        p = self.write("calc.py", "def a(x, y):\n    total = compute(x)   \n    return total\n\ndef b(x, y):\n    total = compute(x) + offset(y)\n    return total\n")
        m = self.mark(p, 2, 2)
        self.write("calc.py", "def a(x, y):\n    total = compute(y)\n    return total\n\ndef b(x, y):\n    total = compute(x) + offset(y)\n    return total\n")
        r = resolve(m, search=False)
        self.assertNotEqual(r.line_start, 6)
        self.assertIn(r.status, ("edited", "orphaned"))

    def test_long_duplicate_left_alone_is_not_the_cited_copy(self):
        line = '    raise ValueError("invalid configuration value")'
        p = self.write("cfg.py", f"def load():\n    cfg = read()\n{line}\n\ndef save():\n    cfg = write()\n{line}\n")
        m = self.mark(p, 7, 7)
        self.assertEqual(m.position["occurrences"], 2)
        self.write("cfg.py", f"def load():\n    cfg = read()\n{line}\n\ndef save():\n    cfg = write()\n")
        self.assertNotEqual(resolve(m, search=False).status, "shifted")

    def test_real_moved_file_beats_an_identical_decoy(self):
        body = '    if not user.is_active:\n        raise PermissionError("inactive account cannot log in")\n    return issue_session(user)\n'
        p = self.write("proj/auth.py", f"# auth module\nimport os\n\ndef login(user):\n{body}\ndef logout(user):\n    drop(user)\n")
        self.write("proj/vendor/aaa_legacy.py", f"# legacy copy, never called\nfrom old import *\n\ndef legacy_login_v1(user, ctx, flags):\n{body}\n# end legacy\n")
        m = self.mark(p, 5, 7)
        os.makedirs(os.path.join(self.dir, "proj/newpkg"))
        os.rename(p, os.path.join(self.dir, "proj/newpkg/auth.py"))
        r = resolve(m, roots=[os.path.join(self.dir, "proj")])
        self.assertEqual((r.status, os.path.basename(os.path.dirname(r.path))), ("moved", "newpkg"))

    def test_license_header_does_not_move_to_another_file(self):
        h = "# Copyright (c) 2024 Example Corp. All rights reserved.\n# Licensed under the MIT License.\n# See LICENSE in the project root for details.\n"
        p = self.write("proj/billing.py", f"{h}\ndef billing():\n    charge()\n")
        self.write("proj/report.py", f"{h}\ndef unrelated_report():\n    render()\n")
        m = self.mark(p, 1, 3)
        os.remove(p)
        self.assertEqual(resolve(m, roots=[os.path.join(self.dir, "proj")]).status, "orphaned")

    def _big(self, n=600, seed=1):
        import random
        r = random.Random(seed)
        words = ["self", "value", "result", "items", "config", "data", "index", "count", "total", "name"]
        out = []
        for i in range(n):
            a, b, c = r.sample(words, 3)
            out.append(f"    {a}_{i} = compute_{b}({c}, {i}) + helper({a}, {b})")
        return out

    def test_long_edited_block_resolves_fast(self):
        import time
        lines = self._big()
        p = self.write("big.py", "\n".join(lines) + "\n")
        m = self.mark(p, 1, 200)
        for i in range(0, 200, 10):
            lines[i] = lines[i].replace("helper", "assist")
        self.write("big.py", "\n".join(lines) + "\n")
        t = time.perf_counter()
        r = resolve(m, search=False)
        self.assertLess(time.perf_counter() - t, 2.0)
        self.assertEqual((r.status, r.line_start), ("edited", 1))

    def test_orphan_search_over_siblings_is_bounded(self):
        import time
        header = "# Copyright (c) 2024 Example Corp. Licensed under the MIT License."
        p = self.write("root/cited.py", "\n".join([header, *self._big(300, 0)]) + "\n")
        m = self.mark(p, 1, 100)
        os.remove(p)
        for k in range(10):
            self.write(f"root/other_{k}.py", "\n".join([header, *self._big(300, k + 1)]) + "\n")
        t = time.perf_counter()
        r = resolve(m, roots=[os.path.join(self.dir, "root")])
        self.assertLess(time.perf_counter() - t, 5.0)
        self.assertEqual(r.status, "orphaned")


class NeighbourSlideTest(unittest.TestCase):
    def test_deleted_line_is_not_matched_to_its_lookalike_neighbour(self):
        from sourcemark.anchor import TextSource
        from sourcemark.locate import locate

        doc = "def load(blob, data):\n    try: return json.loads(blob)\n    try: return json.loads(data)\n    finally: close()\n"
        m = mark_lines(doc, 2, 2, TextSource(path="/x"))
        after = doc.replace("    try: return json.loads(blob)\n", "")
        q = m.quote
        self.assertIsNone(locate(after, q["exact"], q["prefix"], q["suffix"], hint_start=m.position["start"]))
        edited = doc.replace("json.loads(blob)", "json.loads(blob2)")  # in place: neighbours stay
        hit = locate(edited, q["exact"], q["prefix"], q["suffix"], hint_start=m.position["start"])
        self.assertIsNotNone(hit)
        self.assertIn("blob2", edited[hit.start : hit.end])


class RedactionRound2Test(unittest.TestCase):
    # Fake values are assembled at runtime so secret scanners do not flag the test file.
    PW = "Hunter" + "2Hunter2xyz"

    def test_shapes_that_reached_the_ledger(self):
        cases = {
            "compose list": f"    - POSTGRES_PASS" + f"WORD={self.PW}",
            "properties": f"spring.datasource.pass" + f"word={self.PW}",
            "my.cnf": f"pass" + f"word={self.PW}",
            "curl basic": " ".join(["curl", "-u", "admin" + ":" + self.PW, "https://api.example.com"]),
            "curl header": f'curl -H "X-API-Key: {self.PW}" https://api.example.com',
            "mysql": f"mysql -uroot -p{self.PW} app",
            "inline env": f"cd /srv && PGPASS" + f"WORD={self.PW} psql",
            "connstring": f"Server=db;User Id=app;Pass" + f"word={self.PW};",
            "url param": f"https://api.example.com/v1/x?api_key={self.PW}&page=2",
        }
        for name, text in cases.items():
            self.assertTrue(find_secrets(text), name)

    def test_ordinary_code_still_readable(self):
        for c in [
            "password_hash = hash(password)",
            "token = tokens[0]",
            "max_tokens: 4096",
            "if (token) {",
            "export API_KEY=$FROM_VAULT",
            "https://example.com/search?q=hello&page=2",
            "def check_password(user, password):",
        ]:
            self.assertFalse(find_secrets(c), c)

    def test_hook_check_event_is_redacted(self):
        import json
        import sqlite3
        from sourcemark.hooks import stop

        d = tempfile.mkdtemp(prefix="sm-redhook-")
        self.addCleanup(shutil.rmtree, d, True)
        url = "https://hooks.slack.com/services/" + "T000/B000/" + "abcdefghijklmnop"
        t = os.path.join(d, "s.jsonl")
        with open(t, "w") as fh:
            fh.write(json.dumps({"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": f"posted to {url} and https://api.example.com/x?api_key={self.PW}"}]}}) + "\n")
        db = os.path.join(d, "l.db")
        stop({"transcript_path": t}, mode="shadow", ledger_path=db)
        rows = sqlite3.connect(db).execute("select payload from events").fetchall()
        blob = " ".join(r[0] for r in rows)
        self.assertNotIn("abcdefghijklmnop", blob)
        self.assertNotIn(self.PW, blob)
