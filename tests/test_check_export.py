import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest

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
