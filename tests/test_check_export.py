import contextlib
import io
import json
import os
import shutil
import stat
import tempfile
import unittest
from unittest import mock

from sourcemark.check import CitationCheck, Report
from sourcemark.check_export import export_check_v1, validate_check_export_v1
from sourcemark.cli import main
from sourcemark.observe import Observation, Session


def _tool_use(name, inp, ident="read-1"):
    return {
        "type": "assistant",
        "message": {"content": [{"type": "tool_use", "id": ident, "name": name, "input": inp}]},
    }


def _tool_result(structured, ident="read-1"):
    return {
        "type": "user",
        "toolUseResult": structured,
        "message": {"content": [{"type": "tool_result", "tool_use_id": ident, "content": ""}]},
    }


def _say(text):
    return {"type": "assistant", "message": {"content": [{"type": "text", "text": text}]}}


class CheckExportTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp(prefix="sourcemark-export-")
        self.addCleanup(shutil.rmtree, self.directory, True)
        self.source = os.path.join(self.directory, "fixture.py")
        with open(self.source, "w", encoding="utf-8") as fh:
            fh.write("alpha = 1\nbeta = 2\n")

    def transcript(self, events):
        path = os.path.join(self.directory, "session.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for event in events:
                event.setdefault("cwd", self.directory)
                fh.write(json.dumps(event) + "\n")
        return path

    def run_export(self, transcript):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(["check", transcript, "--export", "v1"])
        return code, json.loads(out.getvalue()), err.getvalue()

    def reader_events(self, source):
        if source == "codex":
            return [
                {"type": "session_meta", "payload": {"id": "synthetic-codex", "cwd": self.directory}},
                {"type": "response_item", "payload": {"type": "message", "role": "assistant",
                    "content": [{"type": "output_text", "text": "No citations in this synthetic note."}]}},
            ]
        return [_say("No citations in this synthetic note.")]

    def assert_export_refused(self, transcript):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(["check", transcript, "--export", "v1"])
        self.assertEqual(code, 2)
        self.assertEqual(out.getvalue(), "")
        self.assertEqual(err.getvalue().strip(), "sourcemark: check export refused: malformed transcript")
        self.assertNotIn("synthetic-private-marker", err.getvalue())

    def test_export_refuses_changes_before_the_actual_read(self):
        from sourcemark import cli

        original_read = cli.read_transcript
        for source in ("claude", "codex"):
            for changed_line in ('synthetic-private-marker\n', '["synthetic-private-marker"]\n'):
                with self.subTest(source=source, changed_line=changed_line):
                    transcript = self.transcript(self.reader_events(source))

                    def mutate_then_read(path, **kwargs):
                        with open(path, "a", encoding="utf-8") as fh:
                            fh.write(changed_line)
                        return original_read(path, **kwargs)

                    with mock.patch("sourcemark.cli.read_transcript", side_effect=mutate_then_read):
                        self.assert_export_refused(transcript)

    def test_export_uses_one_root_transcript_read(self):
        original_open = open
        for source in ("claude", "codex"):
            with self.subTest(source=source):
                transcript = self.transcript(self.reader_events(source))
                reads = []

                def record_open(path, *args, **kwargs):
                    if os.fspath(path) == transcript:
                        reads.append(path)
                    return original_open(path, *args, **kwargs)

                with mock.patch("builtins.open", side_effect=record_open):
                    code, payload, err = self.run_export(transcript)
                self.assertEqual((code, err), (0, ""))
                self.assertEqual(payload["state"], "observed")
                self.assertEqual(len(reads), 1)

    def test_export_format_and_records_use_the_same_snapshot(self):
        from sourcemark import observe

        original_detect = observe._detect_transcript_kind
        for source in ("claude", "codex"):
            with self.subTest(source=source):
                events = self.reader_events(source)
                content = (events[-1]["payload"]["content"] if source == "codex"
                           else events[-1]["message"]["content"])
                content[0]["text"] = "The mark is [sm:zzzzzzzzzz]."
                transcript = self.transcript(events)
                replacement = self.reader_events("codex" if source == "claude" else "claude")

                def replace_after_detection(lines, **kwargs):
                    kind = original_detect(lines, **kwargs)
                    with open(transcript, "w", encoding="utf-8") as fh:
                        for event in replacement:
                            fh.write(json.dumps(event) + "\n")
                    return kind

                with mock.patch("sourcemark.observe._detect_transcript_kind", side_effect=replace_after_detection):
                    code, payload, err = self.run_export(transcript)
                self.assertEqual((code, err), (1, ""))
                self.assertEqual(payload["state"], "partial")
                self.assertEqual(payload["counts"]["total"], 1)
                self.assertEqual(payload["counts"]["unknown"], 1)
                self.assertNotIn("zzzzzzzzzz", json.dumps(payload))

    def test_large_snapshot_is_private_and_closed_after_return_or_error(self):
        from sourcemark import observe

        original_spool = observe.tempfile.SpooledTemporaryFile
        for malformed in (False, True):
            with self.subTest(malformed=malformed):
                events = self.reader_events("claude")
                events[0]["synthetic_padding"] = "x" * (2 << 20)
                transcript = self.transcript(events)
                if malformed:
                    with open(transcript, "a", encoding="utf-8") as fh:
                        fh.write("synthetic-private-marker\n")
                snapshots, modes = [], []

                def tracked_spool(*args, **kwargs):
                    snapshot = original_spool(*args, **kwargs)
                    snapshots.append(snapshot)
                    original_rollover = snapshot.rollover

                    def record_rollover():
                        original_rollover()
                        if os.name == "posix":
                            modes.append(stat.S_IMODE(os.fstat(snapshot._file.fileno()).st_mode))

                    snapshot.rollover = record_rollover
                    return snapshot

                with mock.patch("sourcemark.observe.tempfile.SpooledTemporaryFile", side_effect=tracked_spool):
                    if malformed:
                        self.assert_export_refused(transcript)
                    else:
                        code, payload, err = self.run_export(transcript)
                        self.assertEqual((code, err, payload["state"]), (0, "", "observed"))
                self.assertTrue(snapshots)
                self.assertTrue(all(snapshot._rolled and snapshot.closed for snapshot in snapshots))
                if os.name == "posix":
                    self.assertTrue(modes)
                    self.assertEqual(set(modes), {0o600})

    def test_regular_check_keeps_tolerant_jsonl_behavior(self):
        for source in ("claude", "codex"):
            with self.subTest(source=source):
                transcript = self.transcript(self.reader_events(source))
                with open(transcript, "a", encoding="utf-8") as fh:
                    fh.write('synthetic-private-marker\n["synthetic-private-marker"]\n')
                out, err = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    code = main(["check", transcript, "--json"])
                self.assertEqual(code, 0)
                self.assertEqual(err.getvalue(), "")
                self.assertIn("checks", json.loads(out.getvalue()))

    def test_export_refuses_non_object_jsonl_records(self):
        for source in ("claude", "codex"):
            with self.subTest(source=source):
                transcript = self.transcript(self.reader_events(source))
                with open(transcript, "a", encoding="utf-8") as fh:
                    fh.write('["synthetic-private-marker"]\n')
                self.assert_export_refused(transcript)

    def test_export_refuses_invalid_utf8(self):
        for source in ("claude", "codex"):
            with self.subTest(source=source):
                transcript = self.transcript(self.reader_events(source))
                with open(transcript, "ab") as fh:
                    fh.write(b'{"synthetic-private-marker":"\xff"}\n')
                self.assert_export_refused(transcript)

    def test_export_refuses_malformed_delegated_transcript(self):
        transcript = self.transcript(self.reader_events("claude"))
        children = os.path.join(transcript.removesuffix(".jsonl"), "subagents")
        os.makedirs(children)
        with open(os.path.join(children, "synthetic-child.jsonl"), "w", encoding="utf-8") as fh:
            fh.write("synthetic-private-marker\n")
        self.assert_export_refused(transcript)

    def test_regular_check_preserves_reader_decoding_policy(self):
        for source in ("claude", "codex"):
            with self.subTest(source=source):
                transcript = self.transcript(self.reader_events(source))
                with open(transcript, "ab") as fh:
                    fh.write(b'{"synthetic-private-marker":"\xff"}\n')
                out = io.StringIO()
                err = io.StringIO()
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    if source == "claude":
                        self.assertEqual(main(["check", transcript, "--json"]), 2)
                        self.assertEqual(out.getvalue(), "")
                        self.assertTrue(err.getvalue())
                    else:
                        self.assertEqual(main(["check", transcript, "--json"]), 0)
                        self.assertIn("checks", json.loads(out.getvalue()))
                        self.assertEqual(err.getvalue(), "")

    def test_export_refuses_unreadable_delegated_transcript(self):
        transcript = self.transcript(self.reader_events("claude"))
        children = os.path.join(transcript.removesuffix(".jsonl"), "subagents")
        os.makedirs(children)
        child = os.path.join(children, "synthetic-child.jsonl")
        with open(child, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(_say("No citations.")) + "\n")
        original_open = open

        def unreadable_child(path, *args, **kwargs):
            if os.fspath(path) == child:
                raise PermissionError("synthetic-private-marker")
            return original_open(path, *args, **kwargs)

        out, err = io.StringIO(), io.StringIO()
        with mock.patch("builtins.open", side_effect=unreadable_child):
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = main(["check", transcript, "--export", "v1"])
        self.assertEqual(code, 2)
        self.assertEqual(out.getvalue(), "")
        self.assertEqual(err.getvalue().strip(), "sourcemark: check export refused: unreadable transcript")
        self.assertNotIn("synthetic-private-marker", err.getvalue())

    def test_export_refuses_malformed_read_payload_without_details(self):
        transcript = self.transcript([
            _tool_use("Read", {"file_path": self.source}),
            _tool_result({"type": "text", "file": {
                "filePath": self.source,
                "content": {"synthetic-private-marker": "not line text"},
            }}),
            _say("See fixture.py:1."),
        ])
        self.assert_export_refused(transcript)

    def test_valid_export_is_deterministic_and_sanitized(self):
        transcript = self.transcript(
            [
                _tool_use("Read", {"file_path": self.source}),
                _tool_result(
                    {
                        "type": "text",
                        "file": {
                            "filePath": self.source,
                            "content": "alpha = 1\nbeta = 2",
                            "startLine": 1,
                            "numLines": 2,
                            "totalLines": 2,
                        },
                    }
                ),
                _say("The value is in fixture.py:2."),
            ]
        )
        code1, payload1, err1 = self.run_export(transcript)
        code2, payload2, err2 = self.run_export(transcript)
        self.assertEqual(code1, 0)
        self.assertEqual(code2, 0)
        self.assertEqual(err1, "")
        self.assertEqual(err2, "")
        self.assertEqual(payload1, payload2)
        self.assertEqual(payload1["schema"], "sourcemark/check/v1")
        self.assertEqual(payload1["state"], "ok")
        self.assertEqual(
            set(payload1), {"schema", "state", "counts", "policy_sha256", "session_sha256"}
        )
        self.assertEqual(
            payload1["counts"],
            {"total": 1, "passing": 1, "failing": 0, "unknown": 0, "observations": 1, "timed_out": 0},
        )
        self.assertRegex(payload1["policy_sha256"], r"^sha256:[0-9a-f]{64}$")
        self.assertRegex(payload1["session_sha256"], r"^sha256:[0-9a-f]{64}$")
        serialized = json.dumps(payload1)
        for secret in (self.source, "fixture.py:2", "alpha = 1", "read-1"):
            self.assertNotIn(secret, serialized)

    def test_refusal_is_bounded_and_keeps_unknown_count(self):
        transcript = self.transcript([_say("The mark is [sm:zzzzzzzzzz].")])
        code, payload, err = self.run_export(transcript)
        self.assertEqual(code, 1)
        self.assertEqual(err, "")
        self.assertEqual(payload["state"], "partial")
        self.assertEqual(payload["counts"]["total"], 1)
        self.assertEqual(payload["counts"]["passing"], 0)
        self.assertEqual(payload["counts"]["failing"], 1)
        self.assertEqual(payload["counts"]["unknown"], 1)
        self.assertNotIn("unknown_token", json.dumps(payload))
        self.assertNotIn("zzzzzzzzzz", json.dumps(payload))

    def test_empty_check_is_observed(self):
        transcript = self.transcript([_say("No citations in this note.")])
        code, payload, err = self.run_export(transcript)
        self.assertEqual((code, err), (0, ""))
        self.assertEqual(payload["state"], "observed")
        self.assertEqual(payload["counts"]["total"], 0)

    def test_malformed_transcript_refuses_without_exporting_input(self):
        transcript = os.path.join(self.directory, "malformed.jsonl")
        with open(transcript, "w", encoding="utf-8") as fh:
            fh.write('{"type":"assistant"}\n')
            fh.write("raw path and quote should never be echoed\n")
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(["check", transcript, "--export", "v1"])
        self.assertEqual(code, 2)
        self.assertEqual(out.getvalue(), "")
        self.assertEqual(err.getvalue().strip(), "sourcemark: check export refused: malformed transcript")
        self.assertNotIn("raw path", err.getvalue())

    def test_validator_rejects_unknown_count_field(self):
        value = export_check_v1(Report(), Session())
        value["counts"]["unknown_verdict"] = 1
        with self.assertRaisesRegex(ValueError, "invalid count set"):
            validate_check_export_v1(value)

    def test_validator_rejects_unreconciled_counts(self):
        value = export_check_v1(Report(), Session())
        value["counts"]["unknown"] = 1
        with self.assertRaisesRegex(ValueError, "unknown count exceeds"):
            validate_check_export_v1(value)

    def test_timed_out_state_is_bounded(self):
        report = Report([CitationCheck(raw="redacted", verdict="unresolved")])
        value = export_check_v1(report, Session(), timed_out=True)
        self.assertEqual(value["state"], "timed_out")
        self.assertEqual(value["counts"]["timed_out"], 1)
        self.assertEqual(value["counts"]["failing"], 1)

    def test_regular_json_report_keeps_detail_fields(self):
        transcript = self.transcript([_say("See fixture.py:2.")])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(["check", transcript, "--json"])
        self.assertEqual(code, 1)
        report = json.loads(out.getvalue())
        self.assertIn("checks", report)
        self.assertIn("fixture.py", report["checks"][0]["raw"])


if __name__ == "__main__":
    unittest.main()
