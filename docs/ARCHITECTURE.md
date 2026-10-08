# Architecture

## Trust boundary

The trusted caller owns object identity, version, semantic epoch, invariant and
evidence set identities, the unresolved-unknown mask, and the exact tool that
may resolve the current unknown.

The untrusted model owns only its proposed class, decision, tool, and hypothesis
set identifier. The assembler copies trusted identities into the proposal and
rejects a decision frame bound to another observation.

`ToolAdmission64V1` is minted inside the crate from the trusted observation. It
is not a capability for executing a tool. It only authorizes one proposal to
cross the read-only admission boundary when the proposed tool is exactly equal
to the trusted allowed tool.

## Frames

All runtime frames are one 64-byte cache-line-sized block or an exact multiple:

- `CognitiveFrame64V0`: trusted observation or assembled proposal;
- `DecisionFrame64V1`: untrusted model-owned delta;
- `ToolAdmission64V1`: trusted exact-tool admission;
- `GateReceipt64V0`: deterministic verdict and reason mask;
- `LakeSpanRef64V1`: private reference to borrowed Lake bytes.

CRC32 detects accidental or untrusted frame mutation. It is not a cryptographic
signature and does not establish source authority.

## Failure policy

Integrity, schema, identity, version, unknown-mask, policy, or exact-tool
mismatch results in `QUARANTINE`. There is no automatic promotion from one
read-only tool to another.
