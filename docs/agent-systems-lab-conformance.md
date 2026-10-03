# Agent Systems Lab conformance

Sourcemark owns the native `sourcemark/check/v1` export. The export is a deliberately small
boundary for citation/read-evidence checks: four bounded states, six non-negative counts, and two
`sha256:` identities. It does not expose paths, quotes, transcript text, URLs, raw tokens, or
ledger values.

The owner manifest is [`conformance/agent-systems-lab.json`](../conformance/agent-systems-lab.json).
It pins the Agent Proof `agent-proof/interop/v1` adapter by immutable revision and manifest digest.
Agent Proof may normalize this export without importing Sourcemark at runtime; Sourcemark remains
the authority for the native schema, state/count reconciliation, and export privacy boundary.

The owner tests cover valid `ok`, `observed`, `partial`, and `timed_out` shapes plus refusal for
unsupported states, unexpected fields, malformed identities, and unreconciled counts. These tests
prove the native export contract only; they do not claim deployment, adoption, or citation truth
outside the supplied synthetic session.
