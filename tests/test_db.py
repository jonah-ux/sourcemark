import os
import shutil
import socket
import subprocess
import tempfile
import time
import unittest

from sourcemark.db import (
    NULL_SENTINEL,
    PsqlRunner,
    canonical,
    mark_row,
    quote_ident,
    quote_literal,
    resolve_row,
    value_fingerprint,
)


class ScriptedRunner:
    """Returns pre-programmed rows in order; records SQL for inspection."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.sql: list[str] = []

    def __call__(self, sql):
        self.sql.append(sql)
        return self.responses.pop(0)


class UnitTest(unittest.TestCase):
    def test_quoting(self):
        self.assertEqual(quote_ident('we"ird'), '"we""ird"')
        self.assertEqual(quote_literal("o'brien"), "'o''brien'")
        self.assertEqual(quote_literal(None), "NULL")

    def test_canonical(self):
        self.assertEqual(canonical(None), NULL_SENTINEL)
        self.assertEqual(canonical({"b": 1, "a": [1, 2]}), '{"a":[1,2],"b":1}')
        self.assertNotEqual(value_fingerprint(None), value_fingerprint("null"))

    def test_mark_and_resolve_states(self):
        row = {"status": "active", "score": 7, "email": "a@example.com", "sourcemark_now": "2026-01-01T00:00:00Z"}
        run = ScriptedRunner([{"col": "id"}], [row])
        m = mark_row(run, "accounts", 42, ["status", "score", "email"])
        self.assertEqual(m.source["pk"], {"id": "42"})
        self.assertEqual(m.quote["columns"]["status"]["excerpt"], '"active"')
        self.assertIsNone(m.quote["columns"]["email"]["excerpt"])  # sensitive: fingerprint only
        self.assertIn("email", m.redacted)
        self.assertIn('"id"::text = \'42\'', run.sql[1])

        self.assertEqual(resolve_row(m, ScriptedRunner([row])).status, "intact")
        r = resolve_row(m, ScriptedRunner([{**row, "score": 8}]))
        self.assertEqual((r.status, r.changed), ("drifted", ["score"]))
        self.assertEqual(resolve_row(m, ScriptedRunner([])).status, "deleted")

        def boom(_sql):
            raise RuntimeError("connection refused")

        self.assertEqual(resolve_row(m, boom).status, "error")

    def test_composite_key_requires_dict(self):
        run = ScriptedRunner([{"col": "a"}, {"col": "b"}])
        with self.assertRaises(ValueError):
            mark_row(run, "pairs", 1, ["v"])


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@unittest.skipUnless(shutil.which("initdb") and shutil.which("pg_ctl") and shutil.which("psql"), "PostgreSQL not installed")
class PostgresIntegrationTest(unittest.TestCase):
    """A throwaway cluster in a temp dir: real SQL, real catalog lookups, real read-only enforcement."""

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(prefix="sm-pg-")
        data = os.path.join(cls.dir, "data")
        subprocess.run(["initdb", "-D", data, "-U", "sm", "--auth=trust", "-E", "UTF8"], check=True, capture_output=True)
        cls.port = _free_port()
        subprocess.run(
            ["pg_ctl", "-D", data, "-l", os.path.join(cls.dir, "log"), "-w", "-o", f"-p {cls.port} -k {cls.dir} -c listen_addresses=''", "start"],
            check=True, capture_output=True,
        )
        cls.dsn = f"postgresql://sm@/postgres?host={cls.dir}&port={cls.port}"
        dsn = cls.dsn
        cls.admin = staticmethod(lambda sql: subprocess.run(["psql", "-X", "-q", "-d", dsn, "-c", sql], check=True, capture_output=True))
        cls.admin("CREATE TABLE shops (id uuid PRIMARY KEY, name text, plan text, mrr numeric, meta jsonb, updated_at timestamptz DEFAULT now());")
        cls.admin("CREATE TABLE seats (shop_id uuid, seat int, holder text, PRIMARY KEY (shop_id, seat));")
        cls.admin("INSERT INTO shops VALUES ('11111111-1111-1111-1111-111111111111','Harbor Tire','pro',499.50,'{\"tier\":2}'), ('22222222-2222-2222-2222-222222222222','Dockside Auto',NULL,0,NULL);")
        cls.admin("INSERT INTO seats VALUES ('11111111-1111-1111-1111-111111111111', 1, 'ana'), ('11111111-1111-1111-1111-111111111111', 2, 'bo');")
        cls.runner = PsqlRunner(cls.dsn)

    @classmethod
    def tearDownClass(cls):
        subprocess.run(["pg_ctl", "-D", os.path.join(cls.dir, "data"), "-m", "immediate", "stop"], capture_output=True)
        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_round_trip_drift_delete(self):
        sid = "11111111-1111-1111-1111-111111111111"
        m = mark_row(self.runner, "shops", sid, ["name", "plan", "mrr", "meta"], database="test")
        self.assertEqual(resolve_row(m, self.runner).status, "intact")
        self.admin(f"UPDATE shops SET mrr = 520 WHERE id = '{sid}'")
        r = resolve_row(m, self.runner)
        self.assertEqual((r.status, r.changed), ("drifted", ["mrr"]))
        self.admin(f"UPDATE shops SET updated_at = now() + interval '1 day' WHERE id = '{sid}'")
        self.assertEqual(resolve_row(m, self.runner).changed, ["mrr"])  # uncited columns are ignored
        self.admin(f"DELETE FROM shops WHERE id = '{sid}'")
        self.assertEqual(resolve_row(m, self.runner).status, "deleted")

    def test_nulls_and_composite_keys(self):
        m = mark_row(self.runner, "shops", "22222222-2222-2222-2222-222222222222", ["plan", "meta"])
        self.assertEqual(m.quote["columns"]["plan"]["excerpt"], NULL_SENTINEL)
        self.assertEqual(resolve_row(m, self.runner).status, "intact")
        s = mark_row(self.runner, "seats", {"shop_id": "11111111-1111-1111-1111-111111111111", "seat": 2}, ["holder"])
        self.assertEqual(resolve_row(s, self.runner).status, "intact")

    def test_runner_is_read_only(self):
        with self.assertRaises(RuntimeError):
            self.runner("SELECT 1 FROM (VALUES (1)) v; INSERT INTO seats VALUES ('22222222-2222-2222-2222-222222222222', 9, 'x') RETURNING 1")
        with self.assertRaises(RuntimeError):
            self.runner("WITH d AS (DELETE FROM seats RETURNING 1) SELECT * FROM d")


if __name__ == "__main__":
    unittest.main()
