import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from sourcemark.check import check_text
from sourcemark.cite import extract
from sourcemark.observe import Observation, Session, from_shell, from_shell_writes, normalize_url, read_claude_transcript

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
        obs = from_shell("cat a.txt; cat b.txt", "x\ny\n", "/w")
        self.assertTrue(all(o.tool == "Bash-touch" for o in obs))
        # An unquoted marker printed exactly once splits the output; one printed twice does not.
        obs = from_shell("cat a.txt; echo ===; cat b.txt", "x\n===\ny\n", "/w")
        self.assertEqual(sorted((o.path, tuple(o.lines)) for o in obs), [("/w/a.txt", ("x",)), ("/w/b.txt", ("y",))])
        obs = from_shell("cat a.txt; echo ===; cat b.txt", "x\n===\n===\ny\n", "/w")
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
        obs = from_shell(f"wc -l inject.py && grep -n 'budget' inject.py", "2000 inject.py\n1079:    budget = 1\n1335:    render(budget)\n", self.d)
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

    def test_backticked_path_may_contain_spaces(self):
        c = extract("see `/Users/x/Library/Application Support/a/b.md:45-47` now")
        self.assertEqual([(x.path, x.line_start, x.line_end) for x in c], [("/Users/x/Library/Application Support/a/b.md", 45, 47)])

    def test_placeholder_url_is_not_a_citation(self):
        self.assertEqual(extract("open https://<node>:8080/ui or https://{host}/x"), [])
        self.assertEqual([c.url for c in extract("open http://10.0.0.5:8080/ui")], ["http://10.0.0.5:8080/ui"])

    def _transcript(self, entries):
        d = tempfile.mkdtemp(prefix="sm-tx-")
        self.addCleanup(shutil.rmtree, d, True)
        t = os.path.join(d, "s.jsonl")
        with open(t, "w") as fh:
            fh.write("".join(json.dumps(e) + "\n" for e in entries))
        return read_claude_transcript(t)[0]

    def _bash(self, cmd, stdout, is_error=False):
        return [
            {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": cmd}}]}},
            {"type": "user", "toolUseResult": {"stdout": stdout}, "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": stdout, "is_error": is_error}]}},
        ]

    def test_gh_view_with_repo_sources_the_pr_url(self):
        s = self._transcript(self._bash("gh pr view 42 --repo o/r --json state", '{"state":"MERGED"}'))
        self.assertEqual(check_text("[#42](https://github.com/o/r/pull/42)", s).checks[0].verdict, "url_verified")
        self.assertEqual(check_text("[#43](https://github.com/o/r/pull/43)", s).checks[0].verdict, "url_unsourced")

    def test_a_fetcher_that_ran_sources_its_url(self):
        # From a real session: yt-dlp downloaded the clip; the agent linked the source video.
        s = self._transcript(self._bash("yt-dlp -f mp4 https://www.youtube.com/watch?v=03HKX86LAuc -o clip.mp4",
                                        "[youtube] 03HKX86LAuc: Downloading webpage\n[download] 100%\n"))
        self.assertEqual(check_text("[clip](https://www.youtube.com/watch?v=03HKX86LAuc)", s).checks[0].verdict, "url_verified")
        s = self._transcript(self._bash("curl -s https://api.example.com/v1/status | jq .state", '"ok"\n'))
        self.assertEqual(check_text("per https://api.example.com/v1/status", s).checks[0].verdict, "url_verified")
        # An echo returns what was typed; a fallback prints whatever happened.
        for cmd in ("echo https://api.example.com/v1/status", "curl -s https://api.example.com/v1/status || echo down"):
            s = self._transcript(self._bash(cmd, "https://api.example.com/v1/status\n" if cmd.startswith("echo") else "down\n"))
            self.assertEqual(check_text("per https://api.example.com/v1/status", s).checks[0].verdict, "url_unsourced", cmd)

    def test_share_parameters_name_the_same_page(self):
        # From real sessions: Drive returned .../edit?usp=drivesdk; the agent linked .../edit.
        s = self._transcript(self._bash("gws drive files get 1SL",
                                        '{"webViewLink": "https://docs.google.com/spreadsheets/d/1SL/edit?usp=drivesdk"}'))
        self.assertEqual(check_text("[sheet](https://docs.google.com/spreadsheets/d/1SL/edit)", s).checks[0].verdict, "url_verified")
        self.assertEqual(normalize_url("https://youtu.be/abc?si=Zq9"), "https://youtu.be/abc")
        self.assertEqual(normalize_url("https://example.com/a?usp=1"), "https://example.com/a?usp=1")

    def test_url_in_a_failed_result_is_not_sourced(self):
        s = self._transcript(self._bash("curl -f https://example.com/doc", "Exit code 22\ncurl: (22) https://example.com/doc 404", True))
        self.assertEqual(check_text("per https://example.com/doc", s).checks[0].verdict, "url_unsourced")

    def test_copy_destination_is_file_level_evidence(self):
        d = tempfile.mkdtemp(prefix="sm-cp-")
        self.addCleanup(shutil.rmtree, d, True)
        for n in ("a.json", "b.json"):
            with open(os.path.join(d, n), "w") as fh:
                fh.write("{}\n")
        s = self._transcript(self._bash(f"cp /tmp/src/{{a.json,b.json}} {d}/", ""))
        self.assertEqual(check_text(f"[b]({d}/b.json)", s).checks[0].verdict, "file_only")

    def test_route_segments_in_paths(self):
        c = extract("see /w/app/(admin)/applicants/[id]/Thread.tsx:505 now")
        self.assertEqual([(x.path, x.line_start) for x in c], [("/w/app/(admin)/applicants/[id]/Thread.tsx", 505)])

    def test_line_seen_then_file_shrank_is_still_verified(self):
        s = self.sess_with("/w/a.py", 170, [f"l{i}" for i in range(170, 190)], "/w")
        s.add(Observation(path="/w/a.py", line_start=1, lines=["x"] * 50, tool="Read", total_lines=50))
        self.assertEqual(check_text("see /w/a.py:180", s).checks[0].verdict, "verified")
        self.assertEqual(check_text("see /w/a.py:400", s).checks[0].verdict, "out_of_range")

    def test_url_validation_is_linear_on_junk_hosts(self):
        import time
        from sourcemark.cite import valid_url
        t = time.perf_counter()
        self.assertFalse(valid_url("https://" + "a-" * 5000 + "!"))
        self.assertLess(time.perf_counter() - t, 0.5)

    def test_elided_url_is_not_a_citation(self):
        self.assertEqual(extract("pushed to https://github.com/\u2026 and https://example.com/a/.../b"), [])

    def test_pr_link_entry_sources_its_url(self):
        d = tempfile.mkdtemp(prefix="sm-pr-")
        try:
            t = os.path.join(d, "s.jsonl")
            with open(t, "w") as fh:
                fh.write(json.dumps({"type": "pr-link", "prNumber": 7, "prUrl": "https://github.com/o/r/pull/7"}) + "\n")
            s, _ = read_claude_transcript(t)
            self.assertEqual(check_text("opened [PR 7](https://github.com/o/r/pull/7)", s).checks[0].verdict, "url_verified")
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_absolute_path_printed_by_a_command_is_file_level_evidence(self):
        from sourcemark.observe import observe_tool
        d = tempfile.mkdtemp(prefix="sm-out-")
        try:
            f = os.path.join(d, "out.sql")
            with open(f, "w") as fh:
                fh.write("select 1;\n")
            s = Session(cwd="/elsewhere")
            for o in observe_tool("Bash", {"command": 'cp x "$OUT/" && shasum "$OUT/out.sql"'}, {"stdout": f"abc123  {f}\n"}, "", "/elsewhere", None):
                s.add(o)
            self.assertEqual(check_text(f"wrote [the file]({f})", s).checks[0].verdict, "file_only")
            self.assertEqual(check_text(f"see {f}:1", s).checks[0].verdict, "unread_lines")
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_variable_assigned_in_the_command_is_expanded(self):
        from sourcemark.observe import expand_assignments, observe_tool
        self.assertEqual(expand_assignments('W=/r/x; sed -n 5,7p $W/a.ts ${W}/b.ts $OTHER/c'), "W=/r/x; sed -n 5,7p /r/x/a.ts /r/x/b.ts $OTHER/c")
        obs = list(observe_tool("Bash", {"command": "W=/r/x; sed -n 5,7p $W/a.ts"}, {"stdout": "e\nf\ng\n"}, "", "/w", None))
        s = Session(cwd="/w")
        for o in obs:
            s.add(o)
        self.assertEqual(check_text("see a.ts:6", s).checks[0].verdict, "verified")

    def test_echo_markers_split_one_stdout_between_reads(self):
        out = from_shell('echo "--- a"; sed -n 5,6p /r/a.ts; echo "--- b"; sed -n 10,11p /r/b.ts', "--- a\nA5\nA6\n--- b\nB10\nB11\n", "/w")
        got = sorted((o.path, o.line_start, tuple(o.lines)) for o in out)
        self.assertEqual(got, [("/r/a.ts", 5, ("A5", "A6")), ("/r/b.ts", 10, ("B10", "B11"))])
        # A command between a marker and the read pollutes the chunk: the text is not attributed,
        # only the line numbers `sed -n 5,6p` printed (as without markers).
        out = from_shell('echo "--- a"; git status; sed -n 5,6p /r/a.ts', "--- a\nM x\nA5\nA6\n", "/w")
        self.assertEqual(sorted((o.path, o.tool, tuple(o.line_numbers or [])) for o in out),
                         [("/r/a.ts", "Bash-range", (5, 6)), ("/r/a.ts", "Bash-touch", ())])
        self.assertTrue(all(t is None for o in out for t in o.lines))
        # From a real session: a loop that unrolls into echo markers keeps the later read's numbers.
        out = from_shell('for n in m1 vps; do echo "== $n"; ssh $n hostname; done; sed -n 10240,10242p /r/p',
                         "== m1\nm1\n== vps\nvps\nl1\nl2\nl3\n", "/w")
        self.assertIn(("/r/p", (10240, 10241, 10242)), [(o.path, tuple(o.line_numbers or [])) for o in out])
        # A marker that never appears in stdout: no attribution.
        out = from_shell('echo "--- a"; sed -n 5,6p /r/a.ts; sed -n 1,2p /r/b.ts', "A5\nA6\nB1\nB2\n", "/w")
        self.assertTrue(all(o.line_start == 0 for o in out))

    def test_unknown_cd_target_yields_no_relative_evidence(self):
        from sourcemark.observe import observe_tool
        obs = list(observe_tool("Bash", {"command": 'cd "$W" && sed -n 1,3p src/a.py'}, {"stdout": "a\nb\nc\n"}, "", "/w", None))
        self.assertEqual([o for o in obs if not os.path.isabs(o.path) or "$W" in o.path], [])

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

    def test_visited_directory_does_not_block_the_read_copy(self):
        d = tempfile.mkdtemp(prefix="sm-tree-")
        try:
            for root in ("home", "mirror", "read"):
                os.makedirs(os.path.join(d, root))
            for root in ("mirror", "read"):
                with open(os.path.join(d, root, "notes.md"), "w") as fh:
                    fh.write("a\nb\n")
            read = os.path.join(d, "read", "notes.md")
            s = self.sess_with(read, 1, ["a", "b"], os.path.join(d, "home"))
            s.cwds.add(os.path.join(d, "mirror"))  # passed through, never read there
            s.last_cwd = os.path.join(d, "home")
            c = check_text("see notes.md:2", s).checks[0]
            self.assertEqual((c.verdict, c.resolved_path), ("verified", read))
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_two_read_copies_prefer_the_one_covering_the_lines(self):
        d = tempfile.mkdtemp(prefix="sm-tree-")
        try:
            for root in ("start", "end"):
                os.makedirs(os.path.join(d, root))
                with open(os.path.join(d, root, "AGENTS.md"), "w") as fh:
                    fh.write("".join(f"line {i}\n" for i in range(1, 101)))
            start, end = (os.path.join(d, r, "AGENTS.md") for r in ("start", "end"))
            s = self.sess_with(start, 1, ["line 1", "line 2"], os.path.join(d, "start"))
            s.add(Observation(path=end, line_start=80, lines=[f"line {i}" for i in range(80, 96)], tool="Read"))
            s.cwds.add(os.path.join(d, "end"))
            s.last_cwd = os.path.join(d, "end")
            c = check_text("see AGENTS.md:88", s).checks[0]
            self.assertEqual((c.verdict, c.resolved_path), ("verified", end))
            c = check_text("see AGENTS.md:2", s).checks[0]
            self.assertEqual((c.verdict, c.resolved_path), ("verified", start))
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


class AdversarialRound2CheckTest(unittest.TestCase):
    """Second independent adversarial review: false passes in the session evidence."""

    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="sm-adv2-")
        self.addCleanup(shutil.rmtree, self.d, True)

    def file(self, name, lines):
        p = os.path.join(self.d, name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        return p

    def shell(self, cmd, stdout):
        from sourcemark.observe import observe_tool

        s = Session(cwd=self.d)
        for o in observe_tool("Bash", {"command": cmd}, {"stdout": stdout}, "", self.d, None):
            s.add(o)
        return s

    def verdict(self, s, text):
        return check_text(text, s).checks[0].verdict

    def test_linter_output_beside_a_grep_is_not_grep_hits(self):
        self.file("src/pipeline.py", [f"x{i}" for i in range(120)])
        s = self.shell("ruff check src; rg -n 'import json' src", "src/pipeline.py:97:5: F821 undefined name\nsrc/a.py:3:import json\n")
        self.assertNotEqual(self.verdict(s, "see src/pipeline.py:97"), "verified")

    def test_rg_without_line_numbers_gives_no_lines(self):
        self.file("server.log", ["a", "b", "c", "2024-01-15 ERROR db timeout"])
        s = self.shell("rg ERROR server.log", "2024-01-15 ERROR db timeout\n")
        self.assertNotEqual(self.verdict(s, "see server.log:2024"), "verified")
        s = self.shell("rg -n ERROR server.log", "4:2024-01-15 ERROR db timeout\n")
        self.assertEqual(self.verdict(s, "see server.log:4"), "verified")

    def test_grep_tool_with_n_false_gives_no_lines(self):
        from sourcemark.observe import observe_tool

        p = self.file("server.log", ["a", "2024-01-15 ERROR db timeout"])
        s = Session(cwd=self.d)
        for o in observe_tool("Grep", {"pattern": "ERROR", "path": p, "output_mode": "content", "-n": False}, {"content": "2024-01-15 ERROR db timeout"}, "", self.d, None):
            s.add(o)
        self.assertNotEqual(self.verdict(s, f"see {p}:2024"), "verified")

    def test_sed_with_two_ranges_is_not_line_evidence(self):
        self.file("app.py", [f"line_{i} = compute_{i}(x)" for i in range(1, 120)])
        out = "\n".join([f"line_{i} = compute_{i}(x)" for i in (*range(1, 6), *range(100, 106))]) + "\n"
        s = self.shell("sed -n -e '1,5p' -e '100,105p' app.py", out)
        self.assertNotEqual(self.verdict(s, "app.py:7 has `line_101 = compute_101(x)`"), "verified")

    def test_appending_heredoc_is_not_lines_one_to_n(self):
        self.file("notes.py", [f"old{i}" for i in range(100)])
        s = self.shell("tee -a notes.py <<'EOF'\nappended = 1\nEOF", "appended = 1\n")
        self.assertNotEqual(self.verdict(s, "notes.py:1 has `appended = 1`"), "verified")

    def test_heredoc_first_form_is_a_write(self):
        self.file("gen.py", ["def gen():", "    return 42"])
        s = self.shell("cat <<'EOF' > gen.py\ndef gen():\n    return 42\nEOF", "")
        self.assertEqual(self.verdict(s, "gen.py:2 has `return 42`"), "verified")

    def test_quote_in_a_markdown_link_label_is_checked(self):
        s = Session(cwd=self.d)
        p = self.file("calc.py", [f"v{i} = {i}" for i in range(1, 20)])
        s.add(Observation(path=p, line_start=1, lines=[f"v{i} = {i}" for i in range(1, 20)], tool="Read"))
        self.assertEqual(self.verdict(s, "[`drop_all_tables(db)`](calc.py#L10)"), "quote_mismatch")
        self.assertEqual(self.verdict(s, "[`v10 = 10`](calc.py#L10)"), "verified")

    def test_option_values_are_not_pattern_or_path(self):
        p = self.file("src/routes.py", [f"r{i}" for i in range(40)])
        for cmd, out in (
            ("rg -n -C 2 r30 src/routes.py", "29-r28\n30:r29\n31-r30\n"),
            ("rg -n -g '*.py' r29 src/routes.py", "30:r29\n"),
            ("grep -n -A 2 r29 src/routes.py", "30:r29\n31-r30\n32-r31\n"),
        ):
            s = self.shell(cmd, out)
            self.assertEqual(self.verdict(s, "see src/routes.py:30"), "verified", cmd)

    def test_ambiguous_token_is_unknown_not_a_crash(self):
        class Amb:
            def get_mark(self, ref):
                raise LookupError("2 marks start with 'abcdef'")

        r = check_text("see [sm:abcdef] and /etc/never_read_file.conf:12", Session(cwd=self.d), ledger=Amb())
        self.assertEqual([c.verdict for c in r.checks][0], "unknown_token")


class AdversarialRound2UrlTest(unittest.TestCase):
    def tx(self, entries):
        d = tempfile.mkdtemp(prefix="sm-url2-")
        self.addCleanup(shutil.rmtree, d, True)
        t = os.path.join(d, "s.jsonl")
        with open(t, "w") as fh:
            fh.write("".join(json.dumps(e) + "\n" for e in entries))
        return read_claude_transcript(t)[0]

    def tool(self, name, tin, result, tid="t1"):
        return [
            {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "tool_use", "id": tid, "name": name, "input": tin}]}},
            {"type": "user", "toolUseResult": result, "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tid, "content": json.dumps(result)}]}},
        ]

    def v(self, s, text):
        return check_text(text, s).checks[0].verdict

    def test_link_in_the_agents_own_write_is_not_sourced(self):
        s = self.tx(self.tool("Write", {"file_path": "/tmp/x.md", "content": "see https://made.example/up"}, {"type": "create", "filePath": "/tmp/x.md", "content": "see https://made.example/up"}))
        self.assertEqual(self.v(s, "per https://made.example/up"), "url_unsourced")

    def test_link_from_a_subagent_report_is_delegated(self):
        s = self.tx(self.tool("Task", {"prompt": "research"}, {"content": [{"type": "text", "text": "found https://docs.example/a"}]}))
        self.assertEqual(self.v(s, "per https://docs.example/a"), "delegated")

    def test_parenthesised_url_round_trips_and_prefix_does_not_pass(self):
        u = "https://en.wikipedia.org/wiki/Python_(programming_language)"
        s = self.tx(self.tool("WebFetch", {"url": u}, {"result": "Python is ..."}))
        self.assertEqual(self.v(s, f"see [Python]({u})"), "url_verified")
        self.assertEqual(self.v(s, f"see {u}."), "url_verified")
        self.assertEqual(self.v(s, "see https://en.wikipedia.org/wiki/Python"), "url_unsourced")
        self.assertEqual([c.url for c in extract(f"(see {u})")], [u])


class AdversarialRound3CheckTest(unittest.TestCase):
    """Third independent adversarial review: evidence the model never actually saw."""

    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="sm-adv3-")
        self.addCleanup(shutil.rmtree, self.d, True)
        self.n = 0
        self.entries = []

    def file(self, name, lines):
        p = os.path.join(self.d, name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        return p

    def tool(self, name, tin, tur, content):
        self.n += 1
        tid = f"t{self.n}"
        self.entries += [
            {"type": "assistant", "cwd": self.d, "message": {"role": "assistant", "content": [{"type": "tool_use", "id": tid, "name": name, "input": tin}]}},
            {"type": "user", "cwd": self.d, "toolUseResult": tur, "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tid, "content": content}]}},
        ]

    def bash(self, cmd, stdout, content=None, **extra):
        self.tool("Bash", {"command": cmd}, {"stdout": stdout, "stderr": "", **extra}, stdout if content is None else content)

    def session(self):
        t = os.path.join(self.d, "s.jsonl")
        with open(t, "w") as fh:
            fh.write("".join(json.dumps(e) + "\n" for e in self.entries))
        return read_claude_transcript(t)

    def v(self, text):
        return [c.verdict for c in check_text(text, self.session()[0]).checks]

    def test_persisted_output_credits_only_the_preview(self):
        lines = [f"value_{i} = {i}" for i in range(1, 2001)]
        self.file("big.py", lines)
        full = "\n".join(lines)
        preview = "\n".join(lines[:60]) + "\nvalue_61 = 6"
        seen = f"<persisted-output>\nOutput too large (40KB). Full output saved to: /x/o.txt\n\nPreview (first 2KB):\n{preview}\n...\n</persisted-output>"
        self.bash("cat big.py", full[:30000], content=seen, persistedOutputPath="/x/o.txt")
        self.assertEqual(self.v("see big.py:700 and big.py:30"), ["unread_lines", "verified"])

    def test_two_single_file_greps_are_not_attributed(self):
        self.file("m1.py", ["import os", "x = 1", "def run():", "    pass"])
        self.file("m2.py", ["a", "b", "c", "d", "e", "def run_fast():"])
        self.bash('grep -n "def run" m1.py && grep -n "def run" m2.py', "3:def run():\n6:def run_fast():\n")
        self.assertNotEqual(self.v("m1.py:6 defines `def run_fast():`"), ["verified"])

    def test_path_in_a_link_label_is_checked(self):
        self.assertIn("unresolved", self.v("[nothere.py:88](https://github.com/acme/app/blob/main/nothere.py#L88)"))

    def test_echo_marker_ambiguities_give_no_lines(self):
        self.file("a.py", ["def alpha():", "    pass", "", "def secret_beta():", "    pass"])
        self.file("b.py", ["x = 1", "y = 2"])
        self.bash('cat a.py; echo ""; cat b.py', "def alpha():\n    pass\n\ndef secret_beta():\n    pass\n\nx = 1\ny = 2\n")
        self.assertNotEqual(self.v("b.py:4 has `def secret_beta():`"), ["verified"])

    def test_reassigned_variable_is_not_substituted(self):
        from sourcemark.observe import expand_assignments

        self.assertIn("$F", expand_assignments('F=a.py; cat $F; F=b.py; cat $F'))

    def test_mid_command_cd_gives_no_lines(self):
        for sub, val in (("api", "False"), ("web", "True")):
            self.file(f"{sub}/config.py", ["x = 1", f"DEBUG = {val}"])
        self.bash('cd api && echo "== api" && cat config.py; cd ../web && echo "== web" && cat config.py',
                  "== api\nx = 1\nDEBUG = False\n== web\nx = 1\nDEBUG = True\n")
        self.assertNotEqual(self.v(f"`DEBUG = True` at {self.d}/api/config.py:2"), ["verified"])

    def test_task_notification_does_not_start_a_new_turn(self):
        self.entries.append({"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "See nothere.py:412."}]}})
        self.entries.append({"type": "user", "origin": {"kind": "task-notification"}, "message": {"role": "user", "content": "<task-notification>done</task-notification>"}})
        self.entries.append({"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "The background run also passed."}]}})
        sess, texts = self.session()
        from sourcemark.hooks import _last_turn_texts

        self.assertEqual(len(_last_turn_texts(texts, sess.text_turns)), 2)

    def test_edit_context_lines_are_not_seen(self):
        p = self.file("svc.py", [f"line{i}" for i in range(1, 12)])
        patch = [{"oldStart": 4, "newStart": 4, "lines": [" line4", " line5", " line6", "-line7", "+line7b", " line8", " line9"]}]
        self.tool("Edit", {"file_path": p, "old_string": "line7", "new_string": "line7b"}, {"filePath": p, "structuredPatch": patch}, "updated successfully")
        self.assertEqual(self.v(f"{p}:4 and {p}:7"), ["unread_lines", "verified"])

    def test_self_sourced_urls(self):
        self.tool("TodoWrite", {"todos": [{"content": "read https://docs.invented.dev/v9/limits"}]}, {"newTodos": [{"content": "read https://docs.invented.dev/v9/limits"}]}, "ok")
        self.bash('echo "see https://docs.invented.dev/v9/quotas"', "see https://docs.invented.dev/v9/quotas\n")
        self.bash('echo "TODO: gh pr view 4242 --repo acme/app"', "TODO: gh pr view 4242 --repo acme/app\n")
        self.bash("gh pr view 999 --repo acme/app 2>/dev/null || true", "")
        self.bash('M=$(timeout 30 gh pr view 77 --repo acme/app --json state -q .state); echo "77=$M"', "77=MERGED\n")
        got = self.v("https://docs.invented.dev/v9/limits https://docs.invented.dev/v9/quotas https://github.com/acme/app/pull/4242 https://github.com/acme/app/pull/999 https://github.com/acme/app/pull/77")
        self.assertEqual(got, ["url_unsourced"] * 4 + ["url_verified"])

    def test_read_past_the_end_is_not_a_read(self):
        p = self.file("b.py", ["a", "b", "c"])
        self.tool("Read", {"file_path": p, "offset": 50}, {"type": "text", "file": {"filePath": p, "content": "", "startLine": 50, "numLines": 0, "totalLines": 3}}, "")
        self.assertNotEqual(self.v(f"{p}:50"), ["verified"])

    def test_realistic_commands_are_not_false_fails(self):
        self.file("a.py", [f"l{i}" for i in range(1, 6)] + ["def secret_beta():", "    pass"])
        for cmd, out in (("grep -n --color secret_beta a.py", "6:def secret_beta():\n"),
                         ("sed -n 1,7p a.py 2>/dev/null", "l1\nl2\nl3\nl4\nl5\ndef secret_beta():\n    pass\n")):
            self.entries = []
            self.bash(cmd, out)
            self.assertEqual(self.v("a.py:6 has `def secret_beta():`"), ["verified"], cmd)


class AdversarialRound3DelegatedTokenTest(unittest.TestCase):
    def test_relayed_citation_gets_quote_and_coverage_checks(self):
        s = Session(cwd="/w")
        s.add(Observation(path="/w/svc.py", line_start=1, lines=["A = 1", "B = 2", "TIMEOUT = 30"], tool="Read", delegated=True))
        v = lambda t: check_text(t, s).checks[0].verdict
        self.assertEqual(v("`TIMEOUT = 999` (/w/svc.py:3)"), "quote_mismatch")
        self.assertEqual(v("/w/svc.py:1-10"), "partial")
        self.assertEqual(v("`TIMEOUT = 30` (/w/svc.py:3)"), "delegated")

    def test_check_command_fails_an_unknown_token_without_a_ledger(self):
        import contextlib
        import io

        from sourcemark.cli import main

        d = tempfile.mkdtemp(prefix="sm-tok-")
        self.addCleanup(shutil.rmtree, d, True)
        t = os.path.join(d, "s.jsonl")
        with open(t, "w") as fh:
            fh.write(json.dumps({"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "per [sm:zzzzzzzzzz]"}]}}) + "\n")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--ledger", os.path.join(d, "none.db"), "check", t]), 1)


class GhListRefsTest(unittest.TestCase):
    def test_rows_of_a_pr_list_are_sourced(self):
        from sourcemark.observe import gh_refs

        cmd = "gh pr list --repo acme/app --state all --json number,state,title | python3 -c 'print_rows()'"
        out = "10543 MERGED 2026-09-25T00:59:11Z feat: hooks\n10545 MERGED 2026-09-25T00:45:36Z fix: gate\n2026 rows total\n"
        got = gh_refs(cmd, out)
        self.assertIn("https://github.com/acme/app/pull/10545", got)
        self.assertNotIn("https://github.com/acme/app/pull/2026", got)  # no state on that line

    def test_two_listed_repos_are_ambiguous(self):
        from sourcemark.observe import gh_refs

        cmd = "gh pr list --repo acme/app; gh pr list --repo acme/web"
        self.assertEqual(gh_refs(cmd, "12 OPEN title\n"), set())


class ElidedSourceUrlTest(unittest.TestCase):
    def test_elided_url_in_tool_output_is_not_a_source(self):
        from sourcemark.observe import urls_in

        self.assertEqual(urls_in("shared https://loom.com/share/ad4\u2026 and https://loom.com/share/full123"), {"https://loom.com/share/full123"})


class RealSessionLearningTest(unittest.TestCase):
    """Failures learned from the full-scale run over every local session."""

    def test_unquoted_echo_markers_split_sed_ranges(self):
        out = from_shell("sed -n 1,3p /r/z.ts && echo ... && sed -n 200,202p /r/z.ts",
                         "a\nb\nc\n...\nx200\nx201\nx202\n", "/w")
        got = sorted((o.line_start, tuple(o.lines)) for o in out)
        self.assertEqual(got, [(1, ("a", "b", "c")), (200, ("x200", "x201", "x202"))])
        out = from_shell("sed -n 320,321p /r/z.ts && echo ------- && sed -n 520,521p /r/z.ts", "p\nq\n-------\nr\ns\n", "/w")
        self.assertEqual(sorted(o.line_start for o in out), [320, 520])

    def test_split_that_overfills_a_sed_range_is_rejected(self):
        out = from_shell("sed -n 1,2p /r/z.ts && echo --- && sed -n 9,9p /r/z.ts", "a\nb\nextra\n---\nz\n", "/w")
        self.assertTrue(all(o.line_start == 0 for o in out))

    def test_github_compare_urls_are_not_elided(self):
        u = "https://github.com/acme/app/compare/main...feature-x"
        self.assertEqual([c.url for c in extract(f"see [diff]({u}) and {u}")], [u, u])
        from sourcemark.observe import urls_in

        self.assertEqual(urls_in(f"opened {u}"), {u})
        self.assertEqual(extract("cut https://github.com/acme/... here"), [])


class RealSessionLearning2Test(unittest.TestCase):
    def test_git_grep_lines_are_attributed(self):
        d = tempfile.mkdtemp(prefix="sm-gg-")
        self.addCleanup(shutil.rmtree, d, True)
        os.makedirs(os.path.join(d, "lib"))
        with open(os.path.join(d, "lib", "z.ts"), "w") as fh:
            fh.write("\n".join(f"l{i}" for i in range(1, 1000)) + "\n")
        obs = from_shell('git grep -n -E "l932|l933" -- lib/z.ts', "lib/z.ts:932:l932\nlib/z.ts:933:l933\n", d)
        self.assertEqual(sorted(n for o in obs for n in (o.line_numbers or [])), [932, 933])

    def test_sed_range_mixed_with_other_output_credits_numbers_not_text(self):
        d = tempfile.mkdtemp(prefix="sm-sr-")
        self.addCleanup(shutil.rmtree, d, True)
        f = os.path.join(d, "r.ts")
        with open(f, "w") as fh:
            fh.write("\n".join(f"line {i}" for i in range(1, 400)) + "\n")
        s = Session(cwd=d)
        for o in from_shell("sed -n 270,330p r.ts; grep -n zzz other.ts", "line 270\n...\nother.ts:5:zzz\n", d):
            s.add(o)
        self.assertEqual(check_text("see r.ts:313-325", s).checks[0].verdict, "verified")
        self.assertEqual(check_text("see r.ts:331", s).checks[0].verdict, "unread_lines")

    def test_separator_inside_a_quoted_pattern_does_not_split_the_command(self):
        # Real session: `git grep -n -E "batch.length === 0|stopOnShortPage &&|..." -- f | cut ...; git grep ...`
        d = tempfile.mkdtemp(prefix="sm-qs-")
        self.addCleanup(shutil.rmtree, d, True)
        os.makedirs(os.path.join(d, "lib"))
        for name in ("z.ts", "c.ts"):
            with open(os.path.join(d, "lib", name), "w") as fh:
                fh.write("\n".join(f"l{i}" for i in range(1, 1000)) + "\n")
        cmd = 'git grep -n -E "l932 && x|l933" -- lib/z.ts | cut -c1-150; git grep -n "l4" -- lib/c.ts'
        obs = from_shell(cmd, "lib/z.ts:932:l932 && x\nlib/z.ts:933:l933\nlib/c.ts:4:l4\n", d)
        got = {(os.path.basename(o.path), n) for o in obs for n in (o.line_numbers or [])}
        self.assertEqual(got, {("z.ts", 932), ("z.ts", 933), ("c.ts", 4)})

    def test_multi_line_script_lines_are_commands(self):
        from sourcemark.observe import observe_tool

        # Real sessions: `cd D` / `echo "=== x ==="` / `grep -n ... f` on separate lines, not joined by &&.
        d = tempfile.mkdtemp(prefix="sm-ml-")
        self.addCleanup(shutil.rmtree, d, True)
        with open(os.path.join(d, "f.py"), "w") as fh:
            fh.write("\n".join(f"v{i} = {i}" for i in range(1, 50)) + "\n")
        cmd = f'cd {d}\necho "=== where v12 ==="\ngrep -n "v12 =" f.py'
        s = Session(cwd="/")
        for o in observe_tool("Bash", {"command": cmd}, {"stdout": "=== where v12 ===\n12:v12 = 12\n"}, "", "/"):
            s.add(o)
        self.assertEqual(check_text("f.py:12", s).checks[0].verdict, "verified")
        self.assertEqual(check_text("f.py:13", s).checks[0].verdict, "unread_lines")

    def test_echo_fenced_single_file_greps_are_attributed(self):
        d = tempfile.mkdtemp(prefix="sm-ef-")
        self.addCleanup(shutil.rmtree, d, True)
        for name in ("a.py", "b.py"):
            with open(os.path.join(d, name), "w") as fh:
                fh.write("\n".join(f"{name[0]}{i} = {i}" for i in range(1, 200)) + "\n")
        cmd = 'echo "=== in a ==="\ngrep -n "a12 " a.py | head\necho\necho "=== in b ==="\ngrep -n "b40 " b.py'
        obs = from_shell(cmd, "=== in a ===\n12:a12 = 12\n\n=== in b ===\n40:b40 = 40\n", d)
        got = {(os.path.basename(o.path), n) for o in obs for n in (o.line_numbers or [])}
        self.assertEqual(got, {("a.py", 12), ("b.py", 40)})
        # Two single-file greps inside one fence: the bare "N:" lines cannot be told apart.
        both = 'echo "=== both ==="\ngrep -n "a12 " a.py\ngrep -n "b40 " b.py'
        obs = from_shell(both, "=== both ===\n12:a12 = 12\n40:b40 = 40\n", d)
        self.assertEqual([n for o in obs for n in (o.line_numbers or [])], [])

    def test_tilde_path_is_home_not_relative_to_cwd(self):
        from unittest import mock
        from sourcemark.observe import expand_assignments
        home = tempfile.mkdtemp(prefix="sm-home-")
        self.addCleanup(shutil.rmtree, home, True)
        with mock.patch.dict(os.environ, {"HOME": home}):
            obs = from_shell("sed -n '1,2p' ~/bin/tool", "a\nb\n", "/some/repo")
            self.assertEqual([o.path for o in obs], [os.path.join(home, "bin", "tool")])
            self.assertIn(os.path.join(home, "bin", "x"), expand_assignments('W=~/bin/x; cat "$W"'))

    def test_wrapped_grep_is_still_a_grep(self):
        from sourcemark.observe import _unwrap
        self.assertEqual(_unwrap("command -v rg"), "command -v rg")
        d = tempfile.mkdtemp(prefix="sm-wr-")
        self.addCleanup(shutil.rmtree, d, True)
        os.makedirs(os.path.join(d, "fn"))
        with open(os.path.join(d, "fn", "index.ts"), "w") as fh:
            fh.write("\n".join(f"x{i}" for i in range(1, 500)) + "\n")
        obs = from_shell('timeout 150 rg -n --no-heading "x458$" 2>/dev/null | rg -i x | head -20', "fn/index.ts:458:x458\n", d)
        self.assertEqual([(os.path.basename(o.path), o.line_numbers) for o in obs], [("index.ts", [458])])

    def test_dotfile_citations_are_extracted(self):
        got = [(c.path, c.line_start) for c in extract("See /r/.gitignore:7 and ~/.zshrc:12; python 3.12:5 is a version")]
        self.assertEqual(got, [("/r/.gitignore", 7), ("~/.zshrc", 12)])

    def test_heredoc_body_is_not_a_command(self):
        from sourcemark.observe import _split_unquoted
        cmd = "python3 - <<'EOF'\ncat notes.txt\nEOF\nsed -n 1,3p g.py"
        self.assertEqual(_split_unquoted(cmd, newlines=True), ["python3 - <<'EOF'", "sed -n 1,3p g.py"])

    def test_unbalanced_quote_falls_back_to_plain_split(self):
        from sourcemark.observe import _split_unquoted
        self.assertEqual(_split_unquoted("echo don't && cat f"), ["echo don't", "cat f"])
        self.assertEqual(_split_unquoted('grep -n "a && b" f && cat g'), ['grep -n "a && b" f', "cat g"])


class CodexRolloutTest(unittest.TestCase):
    """Codex runs tools from JS cells; only literal one-call cells are read as command output."""

    def rollout(self, d, items):
        p = os.path.join(d, "rollout-test.jsonl")
        rows = [{"type": "session_meta", "payload": {"id": "t"}}, {"type": "turn_context", "payload": {"cwd": d}}]
        for i, (code, out) in enumerate(items):
            cid = f"call_{i}"
            rows.append({"type": "response_item", "payload": {"type": "custom_tool_call", "name": "exec", "call_id": cid, "input": code}})
            rows.append({"type": "response_item", "payload": {"type": "custom_tool_call_output", "call_id": cid,
                         "output": [{"type": "input_text", "text": "Script completed\nWall time 0.1 seconds\nOutput:\n"}, {"type": "input_text", "text": out}]}})
        rows.append({"type": "response_item", "payload": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "done"}]}})
        with open(p, "w") as fh:
            fh.write("\n".join(json.dumps(r) for r in rows) + "\n")
        return p

    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="sm-codex-")
        self.addCleanup(shutil.rmtree, self.d, True)
        with open(os.path.join(self.d, "a.py"), "w") as fh:
            fh.write("\n".join(f"x{i} = {i}" for i in range(1, 100)) + "\n")

    def test_literal_cell_is_read_and_checked(self):
        from sourcemark.observe import read_transcript

        body = "\n".join(f"x{i} = {i}" for i in range(10, 21)) + "\n"
        p = self.rollout(self.d, [('const r = await tools.exec_command({cmd: "sed -n \'10,20p\' a.py", workdir: "%s"}); text(r.output);' % self.d, body)])
        sess, texts = read_transcript(p)
        self.assertEqual(texts[-1][1], "done")
        self.assertEqual(check_text("a.py:15", sess).checks[0].verdict, "verified")
        self.assertEqual(check_text("a.py:21", sess).checks[0].verdict, "unread_lines")

    def test_batch_cells_from_lists_templates_and_json(self):
        """From real rollouts: commands mapped over a list, printed under headers or as JSON."""
        from sourcemark.observe import read_transcript

        d = self.d
        tuples = ('const cmds = [\n  ["env", "uname -a", 30000],\n  ["memory", "grep -n x5 a.py", 30000],\n];\n'
                  'const results = await Promise.all(cmds.map(async ([name, cmd, y]) => {\n'
                  '  const r = await tools.exec_command({cmd, workdir: "%s", yield_time_ms: y});\n  return {name, ...r};\n}));\n'
                  'for (const r of results) text(`=== ${r.name} ===\\n${r.output}\\n[exit ${r.exit_code}]`);' % d)
        tuples_out = "=== env ===\nDarwin\n[exit 0]\n=== memory ===\n5:x5 = 5\n50:x50 = 50\n[exit 0]\n"
        tpl = ('const paths = ["%s/a.py"];\nconst more = ["b"];\n'
               'const rs = await Promise.all(paths.map(p => tools.exec_command({cmd: `sed -n \'60,62p\' \'${p}\'`, workdir: "%s"})));\n'
               'rs.forEach((r,i)=>{ text(`--- ${paths[i]} ---\\n${r.output}`); });' % (d, d))
        tpl_out = "--- %s/a.py ---\nx60 = 60\nx61 = 61\nx62 = 62\n" % d
        js = ('const cmds = [["sed -n \'70,71p\' a.py", "why"], ["grep -n x9 a.py", "why"]];\n'
              'const results = await Promise.allSettled(cmds.map(async ([cmd, why]) => {\n'
              '  const r = await tools.exec_command({cmd, workdir:"%s"});\n  return {cmd, output:r.output, exit_code:r.exit_code};\n}));\n'
              'for (const r of results) text(JSON.stringify(r.value));' % d)
        js_out = json.dumps({"cmd": "sed -n '70,71p' a.py", "output": "x70 = 70\nx71 = 71\n", "exit_code": 0}) + json.dumps({"cmd": "grep -n x9 a.py", "output": "9:x9 = 9\n90:x90 = 90\n", "exit_code": 0})
        sess, _ = read_transcript(self.rollout(d, [(tuples, tuples_out), (tpl, tpl_out), (js, js_out)]))
        for cite in ("a.py:5", "a.py:50", "a.py:61", "a.py:70-71", "a.py:90"):
            self.assertEqual(check_text(cite, sess).checks[0].verdict, "verified", cite)
        self.assertEqual(check_text("a.py:63", sess).checks[0].verdict, "unread_lines")

    def test_concatenated_outputs_read_as_one_sequential_command(self):
        """From real rollouts: Promise.allSettled over literal calls, each output printed bare."""
        from sourcemark.observe import read_transcript

        d = self.d
        code = ('const results = await Promise.allSettled([\n'
                '  tools.exec_command({cmd:"uname -a",workdir:"%s"}),\n'
                '  tools.exec_command({cmd:"rg -n \'x4[0-2] \' a.py | head -80",workdir:"%s"})\n]);\n'
                'for (const r of results) text(r.status==="fulfilled" ? r.value.output : `ERROR: ${r.reason}`);' % (d, d))
        out = "Darwin studio 25.3.0\n40:x40 = 40\n41:x41 = 41\n42:x42 = 42\n"
        res_hdr = ('const results = await Promise.all([\n  tools.exec_command({cmd:"uname",workdir:"%s"}),\n'
                   '  tools.exec_command({cmd:"grep -n \'x7\' a.py",workdir:"%s"})\n]);\n'
                   'for (const [i, r] of results.entries()) {\n  text(`---RESULT ${i + 1}---\\n${r.output}`);\n}' % (d, d))
        hdr_out = ("---RESULT 1---\nDarwin\n---RESULT 2---\nWarning: truncated output (original token count: 900)\n"
                   "Total output lines: 40\n\n7:x7 = 7\n70:x70 = 70\n")
        sess, _ = read_transcript(self.rollout(d, [(code, out), (res_hdr, hdr_out)]))
        for cite in ("a.py:40-42", "a.py:7", "a.py:70"):
            self.assertEqual(check_text(cite, sess).checks[0].verdict, "verified", cite)

    def test_template_cmd_with_line_continuations_is_read(self):
        """From a real rollout: a multi-line `cmd` template with `\\` + newline continuations."""
        from sourcemark.observe import read_transcript

        d = self.d
        code = ('const r = await tools.exec_command({\n  cmd: `printf \'%%s\\\\n\' \'--- hits ---\'\n'
                'git grep -n -E \'x4[12] \' -- \\\n  a.py`,\n  workdir: "%s",\n  yield_time_ms: 10000\n});\ntext(r.output);' % d)
        sess, _ = read_transcript(self.rollout(d, [(code, "--- hits ---\na.py:41:x41 = 41\na.py:42:x42 = 42\n")]))
        self.assertEqual(check_text("a.py:41-42", sess).checks[0].verdict, "verified")

    def test_footer_frames_and_calls_in_different_directories(self):
        """From real rollouts: `text(r.output); text(`--- exit=${r.exit_code} ---`)`, and literal
        calls whose outputs are printed bare but ran in different working directories."""
        from sourcemark.observe import read_transcript

        d = self.d
        os.makedirs(os.path.join(d, "sub"))
        with open(os.path.join(d, "sub", "b.md"), "w") as fh:
            fh.write("y\n" * 400)
        footer = ('const results = await Promise.all([\n'
                  '  tools.exec_command({cmd:"grep -n \'x3[34] \' a.py",workdir:"%s"}),\n'
                  '  tools.exec_command({cmd:"git status --short",workdir:"%s"})\n]);\n'
                  'for (const r of results) { text(r.output); text(`\\n--- exit=${r.exit_code} session=${r.session_id ?? ""} ---\\n`); }' % (d, d))
        footer_out = "33:x33 = 33\n34:x34 = 34\n\n--- exit=0 session= ---\n M a.py\n\n--- exit=0 session= ---\n"
        dirs = ('const results = await Promise.all([\n'
                '  tools.exec_command({cmd:"rg -n \'^y\' b.md | head -2",workdir:"%s/sub"}),\n'
                '  tools.exec_command({cmd:"uname",workdir:"%s"})\n]);\n'
                'for (const r of results) text(r.output);' % (d, d))
        dirs_out = "1:y\n2:y\nDarwin\n"
        sess, _ = read_transcript(self.rollout(d, [(footer, footer_out), (dirs, dirs_out)]))
        for cite in ("a.py:33-34", "sub/b.md:2"):
            self.assertEqual(check_text(cite, sess).checks[0].verdict, "verified", cite)

    def test_batch_cells_with_unknown_split_give_no_lines(self):
        from sourcemark.observe import read_transcript

        d = self.d
        # A header that does not frame each command exactly once, and a cell whose output may carry
        # text after each output: no plain slices are credited.
        dup = ('const paths = ["%s/a.py", "%s/a.py"];\n'
               'const rs = await Promise.all(paths.map(p => tools.exec_command({cmd: `sed -n \'10,11p\' \'${p}\'`})));\n'
               'rs.forEach((r,i)=>{ text(`--- ${paths[i]} ---\\n${r.output}`); });' % (d, d))
        dup_out = "--- {0}/a.py ---\nx10 = 10\nx11 = 11\nx10 = 10\nx11 = 11\n".format(d)  # a header went missing
        tail = ('const cmds = [["one", "sed -n \'30,31p\' a.py"], ["two", "uname"]];\n'
                'const rs = await Promise.all(cmds.map(async ([name, cmd]) => ({name, ...(await tools.exec_command({cmd, workdir: "%s"}))})));\n'
                'for (const r of rs) text(`## ${r.name}\\n${r.output}\\n(exit ${r.exit_code})`);' % d)
        tail_out = "## one\nx30 = 30\nx31 = 31\n(exit 0)\n## two\nDarwin\n(exit 0)\n"
        sess, _ = read_transcript(self.rollout(d, [(dup, dup_out), (tail, tail_out)]))
        for cite in ("a.py:10", "a.py:31"):
            self.assertNotEqual(check_text(cite, sess).checks[0].verdict, "verified", cite)

    def test_json_form_and_failures(self):
        from sourcemark.observe import read_transcript

        ok = json.dumps({"exit_code": 0, "output": "3:x3 = 3\n"})
        bad = json.dumps({"exit_code": 2, "output": "40:x40 = 40\n"})
        cut = "Warning: truncated output (original token count: 9000)\nTotal output lines: 99\n\n50:x50 = 50\n"
        p = self.rollout(self.d, [
            ('text(await tools.exec_command({cmd:"grep -n \'x3 \' a.py"}));', ok),
            ('text(await tools.exec_command({cmd:"grep -n \'x40 \' a.py"}));', bad),
            ('const r=await tools.exec_command({cmd:"grep -n \'x50 \' a.py"});text(r.output);', cut),
            ('const r=await tools.exec_command({cmd:"grep -n x60 " + f});text(r.output);', "60:x60 = 60\n"),
        ])
        sess, _ = read_transcript(p)
        self.assertEqual(check_text("a.py:3", sess).checks[0].verdict, "verified")
        # A cut output keeps the hits that survived: each carries its own line number.
        self.assertEqual(check_text("a.py:50", sess).checks[0].verdict, "verified")
        for n in (40, 60):  # failed, computed command: no line credit
            self.assertNotEqual(check_text(f"a.py:{n}", sess).checks[0].verdict, "verified")

    def test_cut_output_places_no_counted_lines(self):
        from sourcemark.observe import read_transcript

        body = "Warning: truncated output (original token count: 9000)\nTotal output lines: 90\n\n" + "\n".join(f"x{i} = {i}" for i in range(10, 15)) + "…99 tokens truncated…x80 = 80\n"
        p = self.rollout(self.d, [('const r = await tools.exec_command({cmd: "sed -n \'10,80p\' a.py"}); text(r.output);', body)])
        sess, _ = read_transcript(p)
        self.assertNotEqual(check_text("a.py:11", sess).checks[0].verdict, "verified")

    def test_urls_typed_into_a_cell_are_not_sourced(self):
        from sourcemark.observe import read_transcript

        typed = "https://example.org/" + "typed"
        got = "https://example.org/" + "fetched"
        p = self.rollout(self.d, [(f'const r=await tools.exec_command({{cmd:"echo {typed}; curl -s x"}});text(r.output);', f"{typed}\n{got}\n")])
        sess, _ = read_transcript(p)
        self.assertIn(got, " ".join(sess.urls))
        self.assertNotIn(typed, " ".join(sess.urls))


class VanishedGrepFileTest(unittest.TestCase):
    """From real sessions: a one-file grep run in a worktree that has since been removed."""

    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="sm-gone-")
        self.addCleanup(shutil.rmtree, self.d, True)
        os.makedirs(os.path.join(self.d, "lib"))
        self.gone = os.path.join(self.d, "removed-worktree")
        self.out = "217:  typescript: { ignoreBuildErrors: true },\n218-  eslint: {\n"

    def nums(self, obs):
        return {(os.path.relpath(o.path, self.d), n) for o in obs for n in (o.line_numbers or [])}

    def test_one_file_grep_keeps_its_hits_after_the_file_is_gone(self):
        obs = from_shell('grep -n -A1 "typescript" apps/web/next.config.js | head -20', self.out, self.gone)
        self.assertEqual(self.nums(obs), {("removed-worktree/apps/web/next.config.js", 217), ("removed-worktree/apps/web/next.config.js", 218)})

    def test_escaped_brackets_and_stderr_redirect(self):
        cmd = r'grep -n "kind" apps/api/applicant/\[id\]/book/route.ts 2>/dev/null | head -20'
        obs = from_shell(cmd, "476:  interview_kind,\n", self.gone)
        self.assertEqual(self.nums(obs), {("removed-worktree/apps/api/applicant/[id]/book/route.ts", 476)})

    def test_not_trusted_when_the_target_could_be_a_directory_or_glob(self):
        for cmd in ('grep -n x lib', 'grep -n x lib/*.py', 'rg -n x lib/a.py', 'grep -rn x lib/a.py', 'grep -n -d recurse x lib/a.py'):
            with self.subTest(cmd=cmd):
                self.assertEqual(self.nums(from_shell(cmd, "12:x = 1\n", self.gone)), set())

    def test_an_existing_directory_is_not_a_vanished_file(self):
        os.makedirs(os.path.join(self.d, "lib", "pkg.d"))
        self.assertEqual(self.nums(from_shell("grep -n x lib/pkg.d", "12:x = 1\n", self.d)), set())


class PipedSourceTest(unittest.TestCase):
    """From real sessions: `git show origin/main:F | grep -n`, `cat F | sed -n 'A,Bp'`."""

    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="sm-src-")
        self.addCleanup(shutil.rmtree, self.d, True)
        os.makedirs(os.path.join(self.d, ".git"))
        os.makedirs(os.path.join(self.d, "apps", "web"))
        with open(os.path.join(self.d, "apps", "web", "page.tsx"), "w") as fh:
            fh.write("\n".join(f"line {i}" for i in range(1, 200)) + "\n")

    def nums(self, obs):
        return {(os.path.relpath(o.path, self.d), n) for o in obs for n in (o.line_numbers or [])}

    def lines(self, obs):
        return {(os.path.relpath(o.path, self.d), o.line_start, len(o.lines)) for o in obs if o.line_numbers is None and o.lines}

    def test_git_show_into_grep_numbers_the_shown_file(self):
        cmd = 'git show origin/main:apps/web/page.tsx | grep -nE "requireTier|tier" | head -20'
        obs = from_shell(cmd, "114:  requireTier('admin'),\n120:  tier,\n", self.d)
        self.assertEqual(self.nums(obs), {("apps/web/page.tsx", 114), ("apps/web/page.tsx", 120)})

    def test_git_show_path_is_from_the_top_of_the_work_tree(self):
        sub = os.path.join(self.d, "apps")
        obs = from_shell("git show HEAD:apps/web/page.tsx | grep -n x", "7:x\n", sub)
        self.assertEqual(self.nums(obs), {("apps/web/page.tsx", 7)})
        obs = from_shell("git -C apps show HEAD:./web/page.tsx | grep -n x", "7:x\n", self.d)
        self.assertEqual(self.nums(obs), {("apps/web/page.tsx", 7)})

    def test_slices_of_a_piped_file(self):
        obs = from_shell("git show origin/main:apps/web/page.tsx | sed -n '760,762p'", "a\nb\nc\n", self.d)
        self.assertEqual(self.lines(obs), {("apps/web/page.tsx", 760, 3)})
        obs = from_shell("cat apps/web/page.tsx | head -40", "x\n" * 40, self.d)
        self.assertEqual(self.lines(obs), {("apps/web/page.tsx", 1, 40)})
        obs = from_shell("git show HEAD:apps/web/page.tsx", "x\n" * 199, self.d)
        self.assertEqual(self.lines(obs), {("apps/web/page.tsx", 1, 199)})

    def test_transforms_and_other_inputs_are_not_slices(self):
        for cmd in (
            "git show HEAD:apps/web/page.tsx | sed -n '/start/,/end/p'",
            "git show HEAD:apps/web/page.tsx | tail -5",
            "git show HEAD:apps/web/page.tsx | sed -n '5,9p' | grep x",
            "git show HEAD:apps/web/page.tsx | awk 'NR>3'",
            "git show --stat HEAD:apps/web/page.tsx | head -3",
            "cat -v apps/web/page.tsx | head -3",
            "cat apps/web/page.tsx | grep -n x apps/other.ts",
        ):
            with self.subTest(cmd=cmd):
                obs = from_shell(cmd, "5:x\n", self.d)
                self.assertEqual(self.nums(obs) | self.lines(obs), set())


class GlobRootGrepTest(unittest.TestCase):
    """From a real session: `grep -rn PAT ~/runtime/*/lib/hooks.py`, echo-fenced among other commands."""

    def test_hits_under_an_unexpanded_glob_root_are_in_scope(self):
        d = tempfile.mkdtemp(prefix="sm-glob-")
        self.addCleanup(shutil.rmtree, d, True)
        for kit in ("hook-surface", "other"):
            os.makedirs(os.path.join(d, kit, "lib"))
            with open(os.path.join(d, kit, "lib", "hooks.py"), "w") as fh:
                fh.write("x\n" * 600)
        hit = os.path.join(d, "hook-surface", "lib", "hooks.py")
        cmd = f'echo "=== a ==="\ngrep -c x {d}/s.json | sed "s/^/  n: /"\necho "=== b ==="\ngrep -rn adapter {d}/*/lib/hooks.py | head -3 | cut -c1-160\necho "=== c ==="\nls -1 {d}'
        out = f"=== a ===\n  n: 2\n=== b ===\n{hit}:575:  adapter = 1\n=== c ===\nhook-surface\n"
        nums = {(o.path, n) for o in from_shell(cmd, out, d) for n in (o.line_numbers or [])}
        self.assertEqual(nums, {(hit, 575)})

    def test_a_hit_outside_the_glob_is_still_out_of_scope(self):
        d = tempfile.mkdtemp(prefix="sm-glob-")
        self.addCleanup(shutil.rmtree, d, True)
        os.makedirs(os.path.join(d, "a", "lib"))
        os.makedirs(os.path.join(d, "b"))
        for f in ("a/lib/hooks.py", "b/hooks.py"):
            with open(os.path.join(d, f), "w") as fh:
                fh.write("x\n" * 20)
        out = f"{d}/b/hooks.py:5:x\n"
        self.assertEqual([o for o in from_shell(f"grep -rn x {d}/*/lib/hooks.py", out, d) if o.line_numbers], [])


class MidCommandCdTest(unittest.TestCase):
    """From a real session: `cd "$W"; git switch ...; cd apps/web/src; echo ===; grep -rn ...`."""

    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="sm-cd-")
        self.addCleanup(shutil.rmtree, self.d, True)
        for rel in ("apps/web/src/app/Ads.tsx", "app/Ads.tsx"):
            os.makedirs(os.path.dirname(os.path.join(self.d, rel)), exist_ok=True)
            with open(os.path.join(self.d, rel), "w") as fh:
                fh.write("x\n" * 700)

    def nums(self, obs):
        return {(os.path.relpath(o.path, self.d), n) for o in obs for n in (o.line_numbers or [])}

    def test_relative_hits_resolve_against_the_directory_cd_moved_to(self):
        cmd = f'cd {self.d}\ngit switch -q b && echo on\ncd apps/web/src\necho "=== sites ==="\ngrep -rn "acqHref(" app | head -20'
        out = "on\n=== sites ===\napp/Ads.tsx:659:  href={acqHref(x)}\n"
        self.assertEqual(self.nums(from_shell(cmd, out, "/elsewhere")), {("apps/web/src/app/Ads.tsx", 659)})

    def test_cd_to_an_unknown_or_conditional_place_keeps_file_level_only(self):
        out = "app/Ads.tsx:659:x\n"
        for cmd in (
            'cd apps/web/src && cd "$SUB" && grep -rn x app',
            "cd - && grep -rn x app",
            "(cd apps/web/src && true); grep -rn x app",
            "for d in */; do cd $d; done; grep -rn x app",
            "pushd apps/web/src; grep -rn x app",
        ):
            with self.subTest(cmd=cmd):
                self.assertEqual(self.nums(from_shell(cmd, out, self.d)), set())

    def test_segments_before_the_cd_keep_the_old_directory(self):
        cmd = 'grep -n x app/Ads.tsx; cd apps/web/src; echo "=== b ==="; grep -n y app/Ads.tsx'
        out = "3:x\n=== b ===\n9:y\n"
        got = self.nums(from_shell(cmd, out, self.d))
        self.assertNotIn(("app/Ads.tsx", 9), got)
        self.assertNotIn(("apps/web/src/app/Ads.tsx", 3), got)


class AttachedOptionValueTest(unittest.TestCase):
    """From a real session: `git grep -nA5 PAT -- dir`, where -n hides in a cluster with a value."""

    def test_line_numbers_are_seen_in_a_cluster_with_a_value(self):
        d = tempfile.mkdtemp(prefix="sm-opt-")
        self.addCleanup(shutil.rmtree, d, True)
        os.makedirs(os.path.join(d, "app"))
        with open(os.path.join(d, "app", "page.tsx"), "w") as fh:
            fh.write("x\n" * 80)
        out = "app/page.tsx:65:const OPTS = {\napp/page.tsx-66-  ...BASE,\napp/page.tsx-67-  maxStaleMs: 1,\n"
        for cmd in ("git grep -nA5 'OPTS\\s*=' -- app | head -10", "grep -rnA2 OPTS app"):
            with self.subTest(cmd=cmd):
                nums = {n for o in from_shell(cmd, out, d) for n in (o.line_numbers or [])}
                self.assertEqual(nums, {65, 66, 67})
        self.assertEqual([o for o in from_shell("grep -rNA2 OPTS app", out, d) if o.line_numbers], [])


class CodexCellShapesTest(unittest.TestCase):
    """From real Codex cells: one-file greps beside other printers, `for` loops, `bash -lc`."""

    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="sm-cx-")
        self.addCleanup(shutil.rmtree, self.d, True)
        for f in ("a.md", "b.md", "M.md"):
            with open(os.path.join(self.d, f), "w") as fh:
                fh.write("x\n" * 400)

    def nums(self, obs):
        return {(os.path.basename(o.path), n) for o in obs for n in (o.line_numbers or [])}

    def test_a_one_file_grep_printed_last_owns_the_ascending_tail(self):
        cmd = "sed -n '1,3p' a.md; sed -n '1,2p' b.md; rg -n 'Spark|Codex Desktop' M.md | head -35"
        out = "---\nname: x\n12: Spark in a.md's text\nfoo\nbar\n249:Spark node\n265:Codex Desktop app\n"
        self.assertEqual({x for x in self.nums(from_shell(cmd, out, self.d)) if x[0] == "M.md"}, {("M.md", 249), ("M.md", 265)})

    def test_a_one_file_grep_printed_first_owns_the_ascending_head(self):
        cmd = "grep -n 'def ' M.md; sed -n '1,2p' a.md"
        out = "3:def a():\n9:def b():\nheader\nbody\n"
        self.assertEqual({x for x in self.nums(from_shell(cmd, out, self.d)) if x[0] == "M.md"}, {("M.md", 3), ("M.md", 9)})

    def test_no_edge_block_when_another_command_prints_numbered_lines(self):
        for cmd in ("grep -n x a.md; rg -n Spark M.md", "nl -ba a.md | sed -n '1,2p'; rg -n Spark M.md"):
            with self.subTest(cmd=cmd):
                obs = from_shell(cmd, "1:x\n249:Spark\n", self.d)
                self.assertNotIn(("M.md", 249), self.nums(obs))

    def test_a_literal_for_loop_is_read_as_the_commands_it_ran(self):
        cmd = 'for f in a.md b.md; do echo "===== $f ====="; nl -ba "$f" | sed -n \'7,8p\'; done'
        out = "===== a.md =====\n     7\ta\n     8\ta\n===== b.md =====\n     7\tb\n     8\tb\n"
        self.assertEqual(self.nums(from_shell(cmd, out, self.d)), {("a.md", 7), ("a.md", 8), ("b.md", 7), ("b.md", 8)})

    def test_loops_that_compute_their_words_or_values_are_not_unrolled(self):
        for cmd in (
            'for f in *.md; do echo "== $f"; nl -ba "$f"; done',
            'for s in 1,2 3,4; do sed -n "${s}p" a.md | nl -ba -v ${s%,*}; done',
            'for f in $(ls); do nl -ba "$f"; done',
        ):
            with self.subTest(cmd=cmd):
                self.assertEqual(self.nums(from_shell(cmd, "== a.md\n     1\ta\n", self.d)), set())

    def test_a_shell_wrapper_is_read_as_its_script(self):
        for cmd in ("bash -lc 'nl -ba a.md | sed -n \"5,6p\"'", 'sh -c "nl -ba a.md | sed -n 5,6p"',
                    "worker-lifecycle run --kind helper --operation-id x -- sh -lc 'nl -ba a.md | sed -n 5,6p'"):
            with self.subTest(cmd=cmd):
                self.assertEqual(self.nums(from_shell(cmd, "     5\ta\n     6\ta\n", self.d)), {("a.md", 5), ("a.md", 6)})
        self.assertEqual(self.nums(from_shell('bash -c "nl -ba $F | sed -n 5,6p"', "     5\ta\n", self.d)), set())


class SelfNumberedAndBatchTest(unittest.TestCase):
    """From real Codex rollouts: `nl -ba F | sed -n 'A,Bp'` and multi-command cells."""

    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="sm-nl-")
        self.addCleanup(shutil.rmtree, self.d, True)
        os.makedirs(os.path.join(self.d, "lib"))
        for n in ("a.py", "b.py"):
            with open(os.path.join(self.d, "lib", n), "w") as fh:
                fh.write("\n".join(f"{n[0]}{i} = {i}" for i in range(1, 300)) + "\n")

    def nums(self, obs):
        return {(os.path.basename(o.path), n) for o in obs for n in (o.line_numbers or [])}

    def test_nl_slice_carries_its_own_numbers(self):
        obs = from_shell("nl -ba lib/a.py | sed -n '10,12p'", "    10\ta10 = 10\n    11\ta11 = 11\n    12\ta12 = 12\n", self.d)
        self.assertEqual(self.nums(obs), {("a.py", 10), ("a.py", 11), ("a.py", 12)})

    def test_several_nl_slices_split_where_numbering_restarts(self):
        out = "    40\ta40 = 40\n    41\ta41 = 41\n     3\tb3 = 3\n"
        obs = from_shell("nl -ba lib/a.py | sed -n '40,41p'; nl -ba lib/b.py | sed -n '3,3p'", out, self.d)
        self.assertEqual(self.nums(obs), {("a.py", 40), ("a.py", 41), ("b.py", 3)})

    def test_continuing_numbers_split_by_what_each_slice_can_print(self):
        out = "    40\ta40 = 40\n    41\ta41 = 41\n    90\tb90 = 90\n"
        obs = from_shell("nl -ba lib/a.py | sed -n '40,41p'; nl -ba lib/b.py | sed -n '90,90p'", out, self.d)
        self.assertEqual(self.nums(obs), {("a.py", 40), ("a.py", 41), ("b.py", 90)})
        # From a real Codex cell: three slices, two of one file, all ascending, in `bash -lc`.
        cmd = "bash -lc 'nl -ba lib/a.py | sed -n \"18,19p\"; nl -ba lib/b.py | sed -n \"200,201p;250,250p\"; nl -ba lib/b.py | sed -n \"280,281p\"'"
        out = "    18\ta\n    19\ta\n   200\tb\n   201\tb\n   250\tb\n   280\tb\n   281\tb\n"
        self.assertEqual(self.nums(from_shell(cmd, out, self.d)), {("a.py", 18), ("a.py", 19), ("b.py", 200), ("b.py", 201), ("b.py", 250), ("b.py", 280), ("b.py", 281)})

    def test_slices_of_files_that_changed_since_still_split(self):
        # From a real rollout: a.py had more lines then than now, so its current length would cap
        # the slice that printed 400-401 and wrongly leave those lines to no one.
        out = "    10\ta\n    11\ta\n   400\ta\n   401\ta\n    20\tb\n"
        cmd = "nl -ba lib/a.py | sed -n '10,11p'; nl -ba lib/a.py | sed -n '400,401p'; nl -ba lib/b.py | sed -n '20p'"
        got = self.nums(from_shell(cmd, out, self.d))
        self.assertTrue({("a.py", 400), ("a.py", 401), ("b.py", 20)} <= got, got)

    def test_a_boundary_no_range_can_place_stays_file_level(self):
        # gone.py's length is unknown: it may have ended at 60, so 61-62 could be b.py's.
        out = "    40\tx\n" + "".join(f"    {n}\tx\n" for n in range(41, 63))
        obs = from_shell("nl -ba lib/gone.py | sed -n '40,80p'; nl -ba lib/b.py | sed -n '61,62p'", out, self.d)
        self.assertEqual(self.nums(obs), set())
        out = "    40\ta\n    42\ta\n    90\tb\n"  # a gap inside a range: not an `nl` slice
        self.assertEqual(self.nums(from_shell("nl -ba lib/a.py | sed -n '40,45p'; nl -ba lib/b.py | sed -n '90p'", out, self.d)), set())

    def test_nl_slices_beside_other_printers_keep_their_exact_shape(self):
        cmd = "nl -ba lib/a.py | sed -n '30,31p'; rg -n -i 'redeem' lib other 2>/dev/null; wc -l lib/a.py"
        out = "    30\ta30 = 30\n    31\ta31 = 31\nother/x.md:12:redeem\n299 lib/a.py\n"
        self.assertTrue({("a.py", 30), ("a.py", 31)} <= self.nums(from_shell(cmd, out, self.d)))
        for other in ("python3 -c 'print(1)'", "awk '{print NR}' lib/b.py", "cat -n lib/b.py", "for f in x; do nl $f; done", "echo $(nl lib/b.py)"):
            with self.subTest(other=other):
                obs = from_shell(f"nl -ba lib/a.py | sed -n '30,31p'; {other}", "    30\ta\n    31\ta\n", self.d)
                self.assertNotIn(("a.py", 30), self.nums(obs))

    def test_renumbering_pipelines_are_not_self_numbered(self):
        from sourcemark.observe import _self_numbered

        for seg in ("sed -n '5,9p' lib/a.py | nl -ba", "nl -ba -v5 lib/a.py", "nl -ba lib/a.py | cut -c1-20", "git show x:lib/a.py | nl -ba"):
            self.assertIsNone(_self_numbered(seg, self.d), seg)

    def test_batch_cell_credits_only_self_describing_grep_hits(self):
        from sourcemark.observe import _batch_grep_evidence

        code = (
            'const w = "%s";\nconst cmds = [\n ["a", {cmd: "rg -n \'a1[0-2] \' lib", workdir: w}],\n'
            ' ["b", {cmd: "rg -n \'a5 \' lib/a.py", workdir: w}],\n ["c", {cmd: "echo lib/b.py:77:b77", workdir: w}],\n];\n'
            "for (const [n, o] of cmds) { const r = await tools.exec_command(o); text(`== ${n}\\n` + r.output); }"
        ) % self.d
        out = "== a\nlib/a.py:10:a10 = 10\nlib/b.py:11:b11 = 11\n== b\n5:a5 = 5\n== c\nlib/b.py:77:b77\n"
        self.assertEqual(self.nums(_batch_grep_evidence(code, out, "/elsewhere", None)), {("a.py", 10)})

    def test_slices_of_one_file_need_no_split(self):
        out = "     5\ta5 = 5\n    90\ta90 = 90\n"
        obs = from_shell("nl -ba lib/a.py | sed -n '5,5p'; nl -ba lib/a.py | sed -n '90,90p'", out, self.d)
        self.assertEqual(self.nums(obs), {("a.py", 5), ("a.py", 90)})

    def test_labelled_batch_splits_only_on_unique_labels(self):
        from sourcemark.observe import _labelled_batch_evidence

        code = (
            'const commands = [\n ["release gates", "nl -ba lib/a.py | sed -n \'10,11p\'"],\n'
            ' ["plan checks", "nl -ba lib/b.py | sed -n \'20,20p\'"]\n];\n'
            'for (const [l, c] of commands) { const r = await tools.exec_command({cmd: c, workdir: "%s"}); text(`--- ${l}\\n` + r.output); }'
        ) % self.d
        out = "--- release gates\n    10\ta10 = 10\n    11\ta11 = 11\nexit 0\n--- plan checks\n    20\tb20 = 20\n"
        self.assertEqual(self.nums(_labelled_batch_evidence(code, out, "/x", None)), {("a.py", 10), ("a.py", 11), ("b.py", 20)})
        repeated = out + "see release gates above\n"
        self.assertEqual(_labelled_batch_evidence(code, repeated, "/x", None), [])

    def test_codex_cut_keeps_only_self_numbered_lines(self):
        from sourcemark.observe import _codex_uncut

        cut, text = _codex_uncut("Warning: truncated output (original token count: 99)\nTotal output lines: 9\n\n  10\ta\n  11\tb…20 tokens truncated…c\n  90\tz\n")
        self.assertTrue(cut)
        self.assertEqual(text, "  10\ta\n  90\tz\n")


class QuoteShorthandTest(unittest.TestCase):
    """From real sessions: `fn()` naming a function, `**Gate 2**` bolding a prefix of the line."""

    def setUp(self):
        self.s = Session(cwd="/r")
        self.s.add(Observation("/r/v.py", 1522, ['    errors.extend(gate3_errors(documents["gate3"]))'], "Read", None))
        self.s.add(Observation("/r/S.md", 5, ["Current source gate: **Gate 2 - Freeze core contracts**"], "Read", None))

    def v(self, text):
        return check_text(text, self.s).checks[0].verdict

    def test_shorthand_quotes_match_the_line_they_abbreviate(self):
        self.assertEqual(self.v("the gate is `gate3_errors()` (wired at `v.py:1522`)"), "verified")
        self.assertEqual(self.v("`S.md:5` — `Current source gate: **Gate 2**` is stale"), "verified")

    def test_a_wrong_name_or_wrong_text_is_still_caught(self):
        self.assertEqual(self.v("the gate is `gate4_errors()` (wired at `v.py:1522`)"), "quote_mismatch")
        self.assertEqual(self.v("`S.md:5` — `Current source gate: **Gate 3**` is stale"), "quote_mismatch")


class CodeSpanPairingTest(unittest.TestCase):
    """From a real session: a 205-char backticked path made the quote after it vanish."""

    def test_a_quote_after_a_long_or_short_code_span_is_still_attached(self):
        long_path = "/Users/j/Library/Application Support/" + "x" * 170 + "/plan.md"
        cs = extract(f"`{long_path}:37` has `- versioned directories plus an atomicX link switch`.")
        self.assertEqual(cs[0].claimed_quotes, ["- versioned directories plus an atomicX link switch"])
        cs = extract("`x` is set at `a.py:3` to `retry_limit = 5`.")
        self.assertEqual([c.claimed_quotes for c in cs], [["retry_limit = 5"]])

    def test_prose_between_two_spans_is_never_a_quote(self):
        cs = extract("`a.py:3` and `b` both have it")
        self.assertEqual(cs[0].claimed_quotes, [])


class LinkAsCodeTest(unittest.TestCase):
    """From real Codex rollouts: citations written as `[a.py:24](/abs/a.py:24)` in backticks."""

    def test_a_neighbouring_link_in_backticks_is_not_a_quote(self):
        t = ("Reserved at `[life.py:24](/w/agent/life.py:24)` and invoked from "
             "`[turn.py:1453](/w/agent/turn.py:1453)`, with `_start_work(x)` there.")
        got = {c.raw: c.claimed_quotes for c in extract(t)}
        self.assertTrue(all(not q.startswith("[") for qs in got.values() for q in qs), got)
        self.assertIn(["_start_work(x)"], list(got.values()))


class OtherCopyTest(unittest.TestCase):
    """From a real Codex rollout: one relative path read in two worktrees with different text."""

    def test_a_quote_is_judged_against_any_read_copy_of_a_relative_path(self):
        s = Session(cwd="/elsewhere")
        s.add(Observation("/w/a/runtime/lib/sync.py", 110, ["x", "def plan(rows):", "y"], "Read", None))
        s.add(Observation("/w/longer-name/runtime/lib/sync.py", 110, ["x", "def plan(rows, *, strict):", "y"], "Read", None))
        r = check_text("`runtime/lib/sync.py:111` has `def plan(rows, *, strict):`", s).checks[0]
        self.assertEqual((r.verdict, r.resolved_path), ("verified", "/w/longer-name/runtime/lib/sync.py"))
        self.assertEqual(check_text("`runtime/lib/sync.py:111` has `def plan(cols):`", s).checks[0].verdict, "quote_mismatch")
        # An absolute path names one file: no other copy is consulted.
        self.assertEqual(check_text("`/w/a/runtime/lib/sync.py:111` has `def plan(rows, *, strict):`", s).checks[0].verdict, "quote_mismatch")

    def test_lines_are_judged_against_any_read_copy_of_a_relative_path(self):
        # From real Codex rollouts: a repo read in several worktrees; the working directory
        # bound the path to a copy whose reads miss the cited line.
        s = Session(cwd="/w/a")
        s.add(Observation("/w/a/runtime/lib/sync.py", 1, ["x"] * 5, "Read", None))
        s.add(Observation("/w/b/runtime/lib/sync.py", 100, [f"line {n}" for n in range(100, 131)], "Read", None))
        r = check_text("`runtime/lib/sync.py:111` runs it", s).checks[0]
        self.assertEqual((r.verdict, r.resolved_path), ("verified", "/w/b/runtime/lib/sync.py"))
        self.assertEqual(check_text("`runtime/lib/sync.py:150` runs it", s).checks[0].verdict, "unread_lines")
        self.assertEqual(check_text("`runtime/lib/sync.py:111` has `line 99`", s).checks[0].verdict, "quote_mismatch")
        self.assertEqual(check_text("`/w/a/runtime/lib/sync.py:111`", s).checks[0].verdict, "unread_lines")

    def test_a_bare_name_is_not_moved_to_an_unrelated_copy(self):
        # From a real rollout: `README.md:17` "installs @v0.2.0"; the only README whose line 17
        # was read belongs to another repo and installs @v0.3.0.
        s = Session(cwd="/job")
        s.add(Observation("/job/README.md", 1, ["# Job"], "Read", None))
        s.add(Observation("/other/README.md", 1, [f"line {n}" for n in range(1, 31)], "Read", None))
        self.assertEqual(check_text("`README.md:17` installs `@v0.2.0`.", s).checks[0].verdict, "unread_lines")


class CodexMemoryCitationTest(unittest.TestCase):
    """Codex lists the memories it used in <oai-mem-citation>; the paths are under its memories dir."""

    def test_entries_resolve_under_the_codex_memories_directory(self):
        with tempfile.TemporaryDirectory() as home:
            mem = os.path.join(home, "memories", "MEMORY.md")
            s = Session(cwd="/proj")
            s.add(Observation("/proj/MEMORY.md", 1, ["# project"] * 3, "Read", None))
            s.add(Observation(mem, 1, [f"m{n}" for n in range(1, 2001)], "Bash", None))
            text = ("Done.\n<oai-mem-citation>\n<citation_entries>\n"
                    "MEMORY.md:1725-1727|note=[prior guidance]\n</citation_entries>\n</oai-mem-citation>")
            with mock.patch.dict(os.environ, {"CODEX_HOME": home}):
                r = check_text(text, s).checks[0]
                self.assertEqual((r.verdict, r.resolved_path), ("verified", mem))
                # Outside the block the same name is still the working directory's file.
                r = check_text("See MEMORY.md:1725.", s).checks[0]
                self.assertEqual((r.verdict, r.resolved_path), ("unread_lines", "/proj/MEMORY.md"))

    def test_an_entry_missing_from_the_memories_directory_keeps_its_own_path(self):
        # From a real rollout: an entry relative to the working directory, not the memories dir.
        with tempfile.TemporaryDirectory() as home:
            s = Session(cwd="/u")
            s.add(Observation("/u/notes/memory/pool.md", 1, [f"p{n}" for n in range(1, 41)], "Read", None))
            text = "<oai-mem-citation>\nnotes/memory/pool.md:12-26|note=[pool doctrine]\n</oai-mem-citation>"
            with mock.patch.dict(os.environ, {"CODEX_HOME": home}):
                r = check_text(text, s).checks[0]
            self.assertEqual((r.verdict, r.resolved_path), ("verified", "/u/notes/memory/pool.md"))


class MisquoteOnCitedLineTest(unittest.TestCase):
    """From the hard bench: one character changed in a quoted path, URL or decorator passed,
    because only bare names were soft-checked."""

    def setUp(self):
        self.s = Session(cwd="/r")
        self.s.add(Observation("/r/.gitignore", 10, ["!COMMAND-STATUS.md", "!apps/senate-dashboard/api/", "!data/.gitkeep"], "Read", None))
        self.s.add(Observation("/r/t.py", 45, ["class T:", "    @classmethod", "    def make(cls):",
                                               "        closedDate = row.get('closedate')"], "Read", None))

    def v(self, text):
        return check_text(text, self.s).checks[0].verdict

    def test_a_character_changed_on_the_cited_line_is_a_misquote(self):
        for text in (".gitignore:10 has `!COMMAND-STATUSX.md`", ".gitignore:11 has `!apps/senate-dashboardX/api/`",
                     ".gitignore:12 has `!dataX/.gitkeep`", "t.py:46 has `@classmethodX`"):
            self.assertEqual(self.v(text), "quote_mismatch", text)

    def test_exact_text_and_real_names_pass(self):
        for text in (".gitignore:10 has `!COMMAND-STATUS.md`", "t.py:46 has `@classmethod`",
                     # A name the file has elsewhere is a real name, not a misquote of `closedDate`.
                     "t.py:48 prefers `closedate`",
                     # A plural or separator variant is prose.
                     ".gitignore:11 keeps `apps/senate-dashboards`"):
            self.assertEqual(self.v(text), "verified", text)


class MisquotedNameTest(unittest.TestCase):
    """A bare name is a mention, not a quote, unless it misquotes a name that WAS read."""

    def setUp(self):
        self.s = Session(cwd="/r")
        self.s.add(Observation("/r/a.yaml", 20, ["rules:", "  forbidden_patterns:", "    - x", "  self_correct: true", "G2_STATUS_STALE = 1", "run hermes-tag-install now"], "Read", None))

    def v(self, text):
        return check_text(text, self.s).checks[0].verdict

    def test_a_misquoted_name_is_caught(self):
        self.assertEqual(self.v("a.yaml:21 has `forbidden_patternsX:`"), "quote_mismatch")
        self.assertEqual(self.v("a.yaml:24 sets `G2_STATUS_STALEX`"), "quote_mismatch")

    def test_mentions_and_prose_variants_are_not_flagged(self):
        for text in (
            "a.yaml:21 has `forbidden_patterns:`",  # exact
            "a.yaml:21 lists each `forbidden_pattern`",  # singular in prose
            "a.yaml:25 hands off to `hermes_tag_install`",  # separator variant
            "a.yaml:21 is parsed by `undici`",  # a mention of something else
            "a.yaml:21 relates to `setup.py`",  # another file
        ):
            self.assertEqual(self.v(text), "verified", text)

    def test_names_are_not_judged_against_text_known_only_by_line_number(self):
        s = Session(cwd="/r")
        s.add(Observation("/r/b.py", 0, [None] * 5, "Bash-range", None, line_numbers=list(range(10, 15))))
        s.add(Observation("/r/b.py", 30, ["def fetch_all_rows(): pass"], "Read", None))
        self.assertEqual(check_text("b.py:12 defines `fetch_all_rowz`", s).checks[0].verdict, "verified")

    def test_timestamped_backup_paths_are_extracted(self):
        got = [(c.path, c.line_start) for c in extract("See /h/.cfg/config.yaml.bak-20260902-154450:80.")]
        self.assertEqual(got, [("/h/.cfg/config.yaml.bak-20260902-154450", 80)])
