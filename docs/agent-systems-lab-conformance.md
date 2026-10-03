# Agent Systems Lab conformance

Sourcemark owns the native `sourcemark/check/v1` export. The export is a deliberately small
boundary for citation/read-evidence checks: four bounded states, six non-negative counts, and two
`sha256:` identities. It does not expose paths, quotes, transcript text, URLs, raw tokens, or
ledger values.

The owner manifest is [`conformance/agent-systems-lab.json`](../conformance/agent-systems-lab.json).
Its `shared_adapter` fields identify the downstream Agent Proof owner and generic
`agent-proof/interop/v1` contract by immutable revision and manifest digest. The pinned manifest
declares seven legacy adapters and does not declare a `sourcemark/check/v1` adapter. These fields
are provenance metadata; they do not prove normalization or installed consumer acceptance of
Sourcemark exports at that revision. Sourcemark remains the authority for the native schema,
state/count reconciliation, and export privacy boundary, without a runtime dependency on Agent Proof.

The owner tests cover valid `ok`, `observed`, `partial`, and `timed_out` shapes plus refusal for
unsupported states, unexpected fields, malformed identities, and unreconciled counts. These tests
prove the native export contract only; they do not claim deployment, adoption, or citation truth
outside the supplied synthetic session.

The source distribution includes the owner manifest beside these tests. CI checks archive
membership and runs the native conformance cases with installed wheel and source consumers;
the downstream provenance qualification still applies.
