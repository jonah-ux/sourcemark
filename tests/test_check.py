import json
import os
import shutil
import tempfile
import unittest

from sourcemark.check import check_text
from sourcemark.cite import extract
from sourcemark.observe import Session, from_shell, from_shell_writes, normalize_url, read_claude_transcript

LINES = [f"def handler_{i}(event):  # step {i} of the pipeline" for i in range(1, 41)]


def transcript(dirpath: str, cwd: str, events: list[dict]) -> str:
    path = os.path.join(dirpath, "session.jsonl")
    with open(path, "w") as fh:
        for e in events:
            e.setdefault("cwd", cwd)
            fh.write(json.dumps(e) + "\n")
    return path


def tool_use(i, name, inp):
    return {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": f"t{i}", "name": name, "input": inp}]}}


def tool_result(i, structured, text=""):
    return {"type": "user", "toolUseResult": structured,
            "message": {"content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": text}]}}


def say(text):
    return {"type": "assistant", "message": {"content": [{"type": "text", "text": text}]}}


class ExtractTest(unittest.TestCase):
    def test_forms(self):
        text = (
            "See src/app.py:12 and [the view](web/view.tsx#L4-L9), plus lib/x.go#L7. "
            "Docs: [spec](https://example.com/spec) and https://example.org/a. Token [sm:abcdefgh]."
        )
        got = [(c.form, c.path or c.url or c.token, c.line_start, c.line_end) for c in extract(text)]
        self.assertIn(("colon", "src/app.py", 12, 12), got)
        self.assertIn(("markdown", "web/view.tsx", 4, 9), got)
        self.assertIn(("hash", "lib/x.go", 7, 7), got)
        self.assertIn(("url", "https://example.com/spec", None, None), got)
        self.assertIn(("url", "https://example.org/a", None, None), got)
        self.assertIn(("token", "abcdefgh", None, None), got)

    def test_not_citations(self):
        self.assertEqual(extract("listening on localhost:8080 at 12:30, ratio 3:1", urls=False), [])

    def test_extensionless_only_when_known(self):
        self.assertEqual(extract("bin/deploy:12"), [])
        got = extract("bin/deploy:12", known_names={"deploy"})
        self.assertEqual((got[0].path, got[0].line_start), ("bin/deploy", 12))

    def test_claimed_quote_captured(self):
        c = extract("In src/app.py:3 the call `handler_3(event)` runs first.")[0]
        self.assertIn("handler_3(event)", c.claimed_quotes)


class CheckTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="sm-check-")
        self.repo = os.path.join(self.dir, "repo")
        os.makedirs(os.path.join(self.repo, "src"))
        self.file = os.path.join(self.repo, "src", "pipeline.py")
        with open(self.file, "w") as fh:
            fh.write("\n".join(LINES) + "\n")
        with open(os.path.join(self.repo, "src", "unread.py"), "w") as fh:
            fh.write("x = 1\n" * 10)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def session(self, answer: str, extra=()):
        content = "\n".join(LINES[9:20])
        events = [
            tool_use(1, "Read", {"file_path": self.file, "offset": 10, "limit": 11}),
            tool_result(1, {"type": "text", "file": {"filePath": self.file, "content": content,
                                                     "startLine": 10, "numLines": 11, "totalLines": 40}}),
            tool_use(2, "WebFetch", {"url": "https://docs.example.com/guide?utm_source=x"}),
            tool_result(2, {"result": "ok"}, "fetched"),
            *extra,
            say(answer),
        ]
        sess, texts = read_claude_transcript(transcript(self.dir, self.repo, events))
        return check_text(texts[-1][1], sess)

    def verdict(self, answer, extra=()):
        rep = self.session(answer, extra)
        self.assertEqual(rep.total, 1, rep.to_dict())
        return rep.checks[0]

    def test_verified_relative_and_absolute(self):
        self.assertEqual(self.verdict("see src/pipeline.py:12").verdict, "verified")
        self.assertEqual(self.verdict(f"see {self.file}:10-20").verdict, "verified")

    def test_partial_and_unread_lines(self):
        self.assertEqual(self.verdict("see src/pipeline.py:18-25").verdict, "partial")
        self.assertEqual(self.verdict("see src/pipeline.py:33").verdict, "unread_lines")

    def test_out_of_range(self):
        self.assertEqual(self.verdict("see src/pipeline.py:400").verdict, "out_of_range")

    def test_unread_and_nonexistent(self):
        self.assertEqual(self.verdict("see src/unread.py:2").verdict, "unread_file")
        self.assertEqual(self.verdict(f"see {self.repo}/src/ghost_impl.py:2").verdict, "nonexistent")
        self.assertEqual(self.verdict("see lib/ghost_impl.py:2").verdict, "unresolved")

    def test_quote(self):
        self.assertEqual(self.verdict("src/pipeline.py:12 calls `handler_12(event)`").verdict, "verified")
        self.assertEqual(self.verdict("src/pipeline.py:12 calls `wrong_name(x)`").verdict, "quote_mismatch")

    def test_urls(self):
        self.assertEqual(self.verdict("per [guide](https://www.docs.example.com/guide/)").verdict, "url_verified")
        self.assertEqual(self.verdict("per [made up](https://docs.example.com/nope)").verdict, "url_unsourced")

    def test_url_from_any_tool_output_is_sourced(self):
        extra = [tool_use(3, "Bash", {"command": "gh pr create"}),
                 tool_result(3, {"stdout": "https://github.com/o/r/pull/7\n", "stderr": ""})]
        self.assertEqual(self.verdict("opened https://github.com/o/r/pull/7", extra).verdict, "url_verified")

    def test_written_file_counts(self):
        new = os.path.join(self.repo, "notes.md")
        extra = [tool_use(3, "Write", {"file_path": new, "content": "alpha\nbeta\ngamma\n"}),
                 tool_result(3, {"type": "create"})]
        self.assertEqual(self.verdict("wrote notes.md:2", extra).verdict, "verified")


class ShellTest(unittest.TestCase):
    def test_sed_slice(self):
        obs = from_shell("sed -n '5,7p' a.txt", "five\nsix\nseven\n", "/w")
        self.assertEqual((obs[0].path, obs[0].line_start, obs[0].lines), ("/w/a.txt", 5, ["five", "six", "seven"]))

    def test_multi_print_is_file_level_only(self):
        obs = from_shell("cat a.txt; echo ===; cat b.txt", "x\n===\ny\n", "/w")
        self.assertTrue(all(o.tool == "Bash-touch" for o in obs))

    def test_heredoc_write(self):
        obs = from_shell_writes("cat > out.md <<'EOF'\nhello\nworld\nEOF", "/w")
        self.assertEqual((obs[0].path, obs[0].lines), ("/w/out.md", ["hello", "world"]))

    def test_normalize_url(self):
        self.assertEqual(normalize_url("http://www.Example.com/a/?utm_source=x#frag"), "https://example.com/a")


class SessionTest(unittest.TestCase):
    def test_empty_session(self):
        rep = check_text("see a/b.py:1", Session())
        self.assertEqual(rep.checks[0].verdict, "unresolved")


if __name__ == "__main__":
    unittest.main()


class PathNormalizationTest(unittest.TestCase):
    def test_dot_directory_and_tilde(self):
        from sourcemark.observe import Observation
        home = os.path.expanduser("~")
        sess = Session(cwd=home)
        sess.add(Observation("~/.config/tool/settings.toml", 1, ["a", "b", "c"], "Read"))
        self.assertEqual(check_text("see ~/.config/tool/settings.toml:2", sess).checks[0].verdict, "verified")
        self.assertEqual(check_text("see .config/tool/settings.toml:2", sess).checks[0].verdict, "verified")
        self.assertEqual(check_text(f"see {home}/.config/tool/settings.toml:3", sess).checks[0].verdict, "verified")


class DelegatedEvidenceTest(unittest.TestCase):
    def test_subagent_reads_are_delegated(self):
        d = tempfile.mkdtemp(prefix="sm-deleg-")
        try:
            target = os.path.join(d, "schema.sql")
            with open(target, "w") as fh:
                fh.write("\n".join(f"col_{i} text," for i in range(1, 60)) + "\n")
            main = os.path.join(d, "sess.jsonl")
            with open(main, "w") as fh:
                fh.write(json.dumps({"type": "user", "cwd": d, "message": {"content": "what is in the schema?"}}) + "\n")
                fh.write(json.dumps(say(f"The column is defined at {target}:49.")) + "\n")
            os.makedirs(os.path.join(d, "sess", "subagents"))
            lines = "\n".join(f"col_{i} text," for i in range(40, 60))
            with open(os.path.join(d, "sess", "subagents", "agent-x.jsonl"), "w") as fh:
                fh.write(json.dumps(tool_use(1, "Read", {"file_path": target})) + "\n")
                fh.write(json.dumps(tool_result(1, {"type": "text", "file": {"filePath": target, "content": lines, "startLine": 40, "numLines": 20, "totalLines": 59}})) + "\n")
            sess, texts = read_claude_transcript(main)
            rep = check_text(texts[-1][1], sess)
            self.assertEqual(rep.checks[0].verdict, "delegated")
            sess2, texts2 = read_claude_transcript(main, subagents=False)
            self.assertEqual(check_text(texts2[-1][1], sess2).checks[0].verdict, "unread_file")
        finally:
            shutil.rmtree(d, ignore_errors=True)
