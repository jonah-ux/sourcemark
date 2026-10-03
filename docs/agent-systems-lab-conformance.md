# Agent Systems Lab conformance

Sourcemark owns the native `sourcemark/check/v1` export. The export is a deliberately small
boundary for citation/read-evidence checks: four bounded states, six non-negative counts, and two
`sha256:` identities. It does not expose paths, quotes, transcript text, URLs, raw tokens, or
ledger values.

The owner manifest is [`conformance/agent-systems-lab.json`](../conformance/agent-systems-lab.json).
Its `shared_adapter` fields identify the downstream Agent Proof owner and
`agent-proof/interop/v1` contract by immutable revision and manifest digest. At that pinned
source revision, Agent Proof declares and implements the `sourcemark/check/v1` adapter with
the six counts and two identities listed here. The published Agent Proof v0.4.1 release
predates that adapter. The pin establishes source provenance; Sourcemark's tests do not
prove that an installed or published Agent Proof package contains or adopts the adapter.
Sourcemark remains the authority for the native schema,
state/count reconciliation, and export privacy boundary, without a runtime dependency on Agent Proof.

The owner tests cover valid `ok`, `observed`, `partial`, and `timed_out` shapes plus refusal for
unsupported states, unexpected fields, malformed identities, and unreconciled counts. These tests
prove the native export contract only; they do not claim deployment, adoption, or citation truth
outside the supplied synthetic session.

The source distribution includes the owner manifest beside these tests. CI checks archive
membership and runs the native conformance cases with installed wheel and source consumers;
the downstream provenance qualification still applies.
