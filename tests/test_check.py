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


class SingleFileGrepTest(unittest.TestCase):
    """grep/rg on ONE file omit the file name: `N:text`, not `path:N:text`."""

    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="sm-grep-")
        self.f = os.path.join(self.d, "inject.py")
        with open(self.f, "w") as fh:
            fh.write("\n".join(f"line {i}" for i in range(1, 2001)) + "\n")

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_bash_grep_single_file(self):
        obs = from_shell(f"wc -l inject.py && grep -n 'budget' inject.py", "2000 inject.py\n1079:    budget = 1\n1335:    render()\n", self.d)
        self.assertEqual(obs[0].path, self.f)
        self.assertEqual(obs[0].line_numbers, [1079, 1335])

    def test_grep_tool_single_file(self):
        from sourcemark.observe import observe_tool
        res = {"mode": "content", "content": "1079:    budget = 1\n1335:    render()", "numLines": 2}
        sess = Session(cwd=self.d)
        for o in observe_tool("Grep", {"pattern": "budget", "path": self.f, "-n": True}, res, "", self.d):
            sess.add(o)
        self.assertEqual(check_text("see inject.py:1335", sess).checks[0].verdict, "verified")
        self.assertEqual(check_text("see inject.py:1336", sess).checks[0].verdict, "unread_lines")

    def test_recursive_grep_is_not_single(self):
        obs = from_shell("grep -rn 'x' .", "a.py:3:x\n", self.d)
        self.assertEqual(obs[0].path, os.path.join(self.d, "a.py"))


class EndpointTest(unittest.TestCase):
    def test_local_addresses_are_endpoints(self):
        sess = Session()
        import ipaddress
        shared = str(ipaddress.ip_address(0x64500102))  # an address inside RFC 6598 shared space
        for u in ("http://127.0.0.1:8080/api", "http://localhost:3000", "http://10.0.0.5/x", f"http://{shared}:11434", "http://printer.local/"):
            self.assertEqual(check_text(f"server at {u}", sess).checks[0].verdict, "endpoint", u)
        self.assertEqual(check_text("see https://example.com/docs", sess).checks[0].verdict, "url_unsourced")


class EvidenceIntegrityTest(unittest.TestCase):
    """Regression tests for adversarial findings: failed tool calls are never evidence."""

    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="sm-integ-")
        os.makedirs(os.path.join(self.d, "src"))
        self.real = os.path.join(self.d, "src", "a.py")
        with open(self.real, "w") as fh:
            fh.write("".join(f"line {i}\n" for i in range(1, 21)))

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def run_calls(self, calls, answer):
        events = []
        for i, (name, inp, structured, content, err) in enumerate(calls):
            events.append(tool_use(i, name, inp))
            r = tool_result(i, structured, content)
            if err:
                r["message"]["content"][0]["is_error"] = True
            events.append(r)
        events.append(say(answer))
        sess, texts = read_claude_transcript(transcript(self.d, self.d, events))
        return [c.verdict for c in check_text(texts[-1][1], sess).checks]

    def test_failed_read_is_not_a_read(self):
        msg = "Error: File does not exist. Note: your current working directory is /p."
        v = self.run_calls([("Read", {"file_path": os.path.join(self.d, "src/ghost.py")}, msg, msg, True)], "Defined at src/ghost.py:1.")
        self.assertNotIn("verified", v)
        v = self.run_calls([("Read", {"file_path": os.path.join(self.d, "src/ghost.py")}, msg, msg, False)], "Defined at src/ghost.py:1.")
        self.assertNotIn("verified", v)

    def test_failed_cat_is_not_a_read(self):
        out = "Error: Exit code 1\ncat: src/ghost2.py: No such file or directory"
        v = self.run_calls([("Bash", {"command": "cat src/ghost2.py"}, out, out, True)], "See src/ghost2.py:1.")
        self.assertNotIn("verified", v)

    def test_rejected_write_is_not_authorship(self):
        target = os.path.join(self.d, "src", "never.py")
        msg = "Error: The user doesn't want to proceed with this tool use."
        v = self.run_calls([("Write", {"file_path": target, "content": "def h():\n    return 42\n"}, msg, "rejected", True)], "I added it in src/never.py:2.")
        self.assertNotIn("verified", v)
        ok = self.run_calls([("Write", {"file_path": target, "content": "a\nb\n"}, {"type": "create", "filePath": target}, "", False)], "I added it in src/never.py:2.")
        self.assertEqual(ok, ["verified"])

    def test_failed_heredoc_is_not_authorship(self):
        cmd = "cat > src/denied.py <<'EOF'\nx = 1\nEOF"
        v = self.run_calls([("Bash", {"command": cmd}, "Error: Exit code 1\npermission denied", "", True)], "Wrote src/denied.py:1.")
        self.assertNotIn("verified", v)

    def test_grep_over_a_log_is_not_a_read_of_mentioned_files(self):
        log = os.path.join(self.d, "lint.txt")
        with open(log, "w") as fh:
            fh.write("src/a.py:12:5: E501 line too long\n")
        v = self.run_calls([("Bash", {"command": "rg E501 lint.txt"}, {"stdout": "src/a.py:12:5: E501 line too long\n", "stderr": ""}, "", False)], "See src/a.py:12.")
        self.assertNotIn("verified", v)

    def test_grep_context_lines_count(self):
        out = "src/a.py-1-line 1\nsrc/a.py:2:line 2\nsrc/a.py-3-line 3\n"
        v = self.run_calls([("Bash", {"command": "grep -rn -C1 'line 2' src"}, {"stdout": out, "stderr": ""}, "", False)], "See src/a.py:1 and src/a.py:3.")
        self.assertEqual(v, ["verified", "verified"])

    def test_head_n_and_flagged_cat(self):
        v = self.run_calls([("Bash", {"command": "head -n 2 src/a.py"}, {"stdout": "line 1\nline 2\n", "stderr": ""}, "", False)], "See src/a.py:2.")
        self.assertEqual(v, ["verified"])
        v = self.run_calls([("Bash", {"command": "cat -s src/a.py"}, {"stdout": "line 1\n", "stderr": ""}, "", False)], "See src/a.py:1.")
        self.assertNotIn("verified", v)


class StrictnessTest(unittest.TestCase):
    """Regression tests for adversarial findings in the checker."""

    def sess_with(self, path, start, lines, cwd):
        from sourcemark.observe import Observation
        s = Session(cwd=cwd)
        s.add(Observation(path, start, lines, "Read"))
        return s

    def test_exact_url_matching(self):
        s = Session()
        s.urls.update({"https://github.com", "https://docs.example.com/guide/install/linux", "https://x.com/r?ref=v1", "https://x.com/p?id=ABC"})
        v = lambda t: check_text(t, s).checks[0].verdict
        self.assertEqual(v("see https://github.com/torvalds/linux/blob/master/kernel/fabricated.c"), "url_unsourced")
        self.assertEqual(v("see https://docs.example.com/guide"), "url_unsourced")
        self.assertEqual(v("see https://x.com/r?ref=v2"), "url_unsourced")
        self.assertEqual(v("see https://x.com/p?id=abc"), "url_unsourced")
        self.assertEqual(v("see https://x.com/p?id=ABC"), "url_verified")

    def test_urls_in_parens_and_angles_are_extracted(self):
        got = [c.url for c in extract("(https://fabricated.example/x) and <https://other.example/y>")]
        self.assertEqual(got, ["https://fabricated.example/x", "https://other.example/y"])

    def test_private_document_is_not_an_endpoint(self):
        s = Session()
        self.assertEqual(check_text("see https://reports.internal/q3/fabricated.pdf", s).checks[0].verdict, "url_unsourced")

    def test_no_cross_tree_binding(self):
        d = tempfile.mkdtemp(prefix="sm-tree-")
        try:
            for root in ("proj", "other"):
                os.makedirs(os.path.join(d, root, "src"))
                with open(os.path.join(d, root, "src", "a.py"), "w") as fh:
                    fh.write("x = 1\ny = 2\n")
            s = self.sess_with(os.path.join(d, "other", "src", "a.py"), 1, ["x = 1", "y = 2"], os.path.join(d, "proj"))
            self.assertEqual(check_text("see src/a.py:2", s).checks[0].verdict, "unread_file")
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_every_quote_must_match_in_order_nearby(self):
        lines = ["def f(a, b):", "    return compute(a, b)", "", "", "", "    cleanup(a)"]
        s = self.sess_with("/w/src/b.py", 1, lines, "/w")
        v = lambda t: check_text(t, s).checks[0].verdict
        self.assertEqual(v("`compute(a, b)` at /w/src/b.py:2"), "verified")
        self.assertEqual(v("`compute(a, b)` and `delete_everything(db)` at /w/src/b.py:2"), "quote_mismatch")
        self.assertEqual(v("`compute(a, ... def f(` at /w/src/b.py:2"), "quote_mismatch")  # fragments out of order
        self.assertEqual(v("`cleanup(a)` at /w/src/b.py:4"), "quote_mismatch")

    def test_extraction_hygiene(self):
        text = "Python 3.12:5, numpy==1.26:2, api.example.com:443, C:\\x\\y.py:3\n```\nsee example/x.py:9\n```\nreal: src/z.py:4"
        self.assertEqual([c.raw for c in extract(text)], ["src/z.py:4"])

    def test_tokens_use_the_ledger(self):
        from sourcemark.anchor import TextSource, mark_lines
        from sourcemark.ledger import Ledger
        d = tempfile.mkdtemp(prefix="sm-tok-")
        try:
            f = os.path.join(d, "n.txt")
            body = "".join(f"note {i} about the harbor\n" for i in range(1, 11))
            with open(f, "w") as fh:
                fh.write(body)
            with Ledger(os.path.join(d, "l.db")) as led:
                m = mark_lines(body, 3, 3, TextSource(path=f))
                led.put_mark(m)
                v = lambda t: check_text(t, Session(), ledger=led).checks[0].verdict
                self.assertEqual(v(f"per {m.token}"), "token_ok")
                self.assertEqual(v("per [sm:abcdefghij]"), "unknown_token")
            self.assertEqual(check_text("per [sm:abcdefghij]", Session()).checks[0].verdict, "token_unchecked")
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_unresolved_fails_enforcement(self):
        from sourcemark.hooks import FAILING
        self.assertIn("unresolved", FAILING)
        self.assertIn("unknown_token", FAILING)
