import json
from pathlib import Path
import unittest

from sourcemark.check_export import validate_check_export_v1


ROOT = Path(__file__).parents[1]
MANIFEST = ROOT / "conformance" / "agent-systems-lab.json"


def export(state="ok", *, total=1, passing=1, failing=0, unknown=0, observations=1, timed_out=0):
    return {
        "schema": "sourcemark/check/v1",
        "state": state,
        "counts": {
            "total": total,
            "passing": passing,
            "failing": failing,
            "unknown": unknown,
            "observations": observations,
            "timed_out": timed_out,
        },
        "policy_sha256": "sha256:" + "a" * 64,
        "session_sha256": "sha256:" + "b" * 64,
    }


class AgentSystemsLabConformanceTests(unittest.TestCase):
    def test_owner_manifest_declares_native_contract_and_downstream_provenance(self):
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema"], "agent-systems-lab-sourcemark-conformance/v1")
        self.assertEqual(manifest["native_schema"], "sourcemark/check/v1")
        self.assertEqual(manifest["shared_adapter"]["owner"], "agent-proof")
        self.assertRegex(manifest["shared_adapter"]["revision"], r"^[0-9a-f]{40}$")
        self.assertRegex(manifest["shared_adapter"]["manifest_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(manifest["states"], ["ok", "observed", "partial", "timed_out"])
        self.assertEqual(manifest["counts"], ["total", "passing", "failing", "unknown", "observations", "timed_out"])
        self.assertEqual(manifest["identities"], ["policy_sha256", "session_sha256"])
        for key in ("raw_paths_exported", "quotes_exported", "transcript_text_exported", "urls_exported", "ledger_values_exported"):
            self.assertIs(manifest["privacy"][key], False, key)

    def test_native_states_and_counts_are_validated(self):
        for state, values in (
            ("ok", dict(total=1, passing=1)),
            ("observed", dict(total=0, passing=0, observations=0)),
            ("partial", dict(total=1, passing=0, failing=1, unknown=1)),
            ("timed_out", dict(total=1, passing=0, failing=1, timed_out=1)),
        ):
            with self.subTest(state=state):
                self.assertEqual(validate_check_export_v1(export(state, **values))["state"], state)

    def test_native_refusals_fail_closed(self):
        mutations = []
        bad_state = export()
        bad_state["state"] = "unsupported"
        mutations.append(bad_state)
        bad_field = export()
        bad_field["private"] = "must not cross"
        mutations.append(bad_field)
        bad_counts = export()
        bad_counts["counts"]["passing"] = 2
        mutations.append(bad_counts)
        bad_identity = export()
        bad_identity["policy_sha256"] = "private"
        mutations.append(bad_identity)
        for payload in mutations:
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    validate_check_export_v1(payload)


if __name__ == "__main__":
    unittest.main()
