import contextlib
import io
import json
import os
import shutil
import sqlite3
import tempfile
import unittest

from sourcemark.cli import main
from sourcemark.hooks import stop
from sourcemark.ledger import Ledger


def write_transcript(path, file_path, answer):
    lines = "\n".join(f"line {i} of the config" for i in range(1, 21))
    events = [
        {"type": "user", "message": {"content": "please check the config"}},
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": file_path}}]}},
        {"type": "user", "toolUseResult": {"type": "text", "file": {"filePath": file_path, "content": lines, "startLine": 1, "numLines": 20, "totalLines": 20}},
         "message": {"content": [{"type": "tool_result", "tool_use_id": "t1", "content": ""}]}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": answer}]}},
    ]
    with open(path, "w") as fh:
        for e in events:
            fh.write(json.dumps(e) + "\n")


class LedgerTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="sm-ledger-")
        self.db = os.path.join(self.dir, "ledger.db")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_chain_verifies_and_detects_tampering(self):
        with Ledger(self.db) as led:
            for i in range(5):
                led.append("note", {"i": i}, session="s1")
            self.assertTrue(led.verify()["ok"])
        con = sqlite3.connect(self.db)
        con.execute("UPDATE events SET payload = ? WHERE seq = 3", (json.dumps({"i": 99}),))
        con.commit()
        con.close()
        with Ledger(self.db) as led:
            v = led.verify()
        self.assertFalse(v["ok"])
        self.assertEqual(v["broken_at"], 3)


class HookTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="sm-hook-")
        self.cfg = os.path.join(self.dir, "app.cfg")
        with open(self.cfg, "w") as fh:
            fh.write("\n".join(f"line {i} of the config" for i in range(1, 21)) + "\n")
        self.tr = os.path.join(self.dir, "t.jsonl")
        self.db = os.path.join(self.dir, "ledger.db")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def run_stop(self, answer, mode):
        write_transcript(self.tr, self.cfg, answer)
        return stop({"transcript_path": self.tr, "session_id": "s"}, mode=mode, ledger_path=self.db)

    def test_shadow_is_silent_but_recorded(self):
        self.assertIsNone(self.run_stop(f"see {self.cfg}:99", "shadow"))
        with Ledger(self.db) as led:
            ev = list(led.events(kind="check"))
        self.assertEqual(ev[-1]["payload"]["counts"], {"out_of_range": 1})

    def test_warn_and_enforce(self):
        self.assertIn("1 backed", self.run_stop(f"see {self.cfg}:3", "warn")["systemMessage"])
        out = self.run_stop(f"see {self.cfg}:99", "enforce")
        self.assertEqual(out["decision"], "block")
        self.assertIn("out_of_range", out["reason"])
        self.assertIsNone(self.run_stop(f"see {self.cfg}:3", "enforce"))

    def test_enforce_never_loops(self):
        write_transcript(self.tr, self.cfg, f"see {self.cfg}:99")
        out = stop({"transcript_path": self.tr, "stop_hook_active": True}, mode="enforce", ledger_path=self.db)
        self.assertNotIn("decision", out)


class CliTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="sm-cli-")
        self.db = os.path.join(self.dir, "ledger.db")
        self.f = os.path.join(self.dir, "notes.txt")
        with open(self.f, "w") as fh:
            fh.write("".join(f"note {i}: the harbor light blinks every {i} seconds\n" for i in range(1, 30)))

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def cli(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(["--ledger", self.db, *argv])
        return code, out.getvalue()

    def test_mark_resolve_show_verify(self):
        code, out = self.cli("mark", f"{self.f}:10-11", "--json")
        self.assertEqual(code, 0)
        mark_id = json.loads(out)["id"]
        with open(self.f, "r+") as fh:
            body = fh.read()
            fh.seek(0)
            fh.write("inserted\n" * 4 + body)
        code, out = self.cli("resolve", mark_id, "--json")
        res = json.loads(out)
        self.assertEqual((res["status"], res["line_start"]), ("shifted", 14))
        code, out = self.cli("show", "[sm:" + mark_id[4:14] + "]")
        self.assertIn("harbor light blinks every 10", out)
        self.assertEqual(self.cli("verify-ledger")[0], 0)

    def test_check_exit_codes(self):
        tr = os.path.join(self.dir, "t.jsonl")
        write_transcript(tr, self.f, f"see {self.f}:2")
        self.assertEqual(self.cli("check", tr)[0], 0)
        write_transcript(tr, self.f, f"see {self.f}:200")
        self.assertEqual(self.cli("check", tr)[0], 1)


if __name__ == "__main__":
    unittest.main()
