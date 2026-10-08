# ZLP Guarded Agent

`zlp-guarded-agent` is an experimental Rust boundary for local LLM agents.
The model may propose a read-only tool, but a deterministic gate decides
whether that exact tool was authorized for the trusted observation.

The model is a proposal source, not an authority.

## The problem

Treating every read-only tool as equivalent is unsafe. Reading one known file,
searching a project, and asking the user are different operations. A model that
chooses `READ_FILE` when only `SEARCH_FILES` or `REQUEST_USER_INPUT` is valid
must not be admitted merely because all three operations are read-only.

## The boundary

```text
trusted observation
        |
        v
untrusted local LLM -> DecisionFrame64V1
        |
        v
ToolAdmission64V1 + deterministic Gate
        |
        +-> exact authorized tool: ADMIT_READ_ONLY
        |
        +-> any other tool: QUARANTINE
```

Runtime transport uses fixed 64-byte frames. Large invariant and evidence
bodies remain in a read-only Lake view and are passed as borrowed byte slices,
not copied into the decision frames. JSON is not part of the runtime boundary.

## Current verified result

A frozen 20-case local-model run produced the correct class and decision in all
20 cases, but selected the exact tool in only 14 cases. Before tool admission,
all 20 proposals passed the broad read-only gate. After `ToolAdmission64V1`:

- the 14 exact tool choices were admitted;
- all 6 wrong read-only tool choices were quarantined;
- two ordered proposal runs were byte-identical;
- both proposal streams had SHA-256
  `679a816489e86a38da90dbbc5953bd2cf2f6018da630241f89a5781e4514e2c5`.

This is a bounded regression result, not a general model-quality benchmark.

## What the code guarantees

- `ToolAdmission64V1` is exactly 64 bytes and has private fields.
- Calling code cannot construct an admission with safe Rust public APIs.
- An admission is bound to task, object, physical version, semantic epoch,
  observation integrity, trusted unknown mask, policy ID/version, and one exact
  allowed tool.
- Corrupt, stale, cross-object, reissued, or tool-mismatched admissions fail
  closed.
- The returned receipt binds the tool-admission integrity and policy identity.
- Publication and execution flags remain forbidden.

## What the code does not guarantee

- semantic correctness of the trusted observation;
- source authority or evidence sufficiency;
- general reasoning quality of an LLM;
- safe write, shell, network, publication, or external actions;
- production readiness, evaluation validity, profitability, or trading safety.

## Test

Run the deterministic demonstration (no model download required):

```bash
cargo run --example tool_admission
```

Expected output:

```text
allowed=SEARCH_FILES proposed=READ_FILE verdict=QUARANTINE
allowed=SEARCH_FILES proposed=SEARCH_FILES verdict=ADMIT_READ_ONLY
No tools executed. Both expected verdicts verified.
```

The example uses a deterministic model stand-in to demonstrate the admission
boundary. It does not measure LLM quality or execute either proposed tool.

Run the checks:

```bash
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt -- --check
```

The core crate has no runtime dependencies and does not include model weights.

## Status

Research preview. The current surface is deliberately read-only and
fail-closed.

## License

MIT. See [LICENSE](LICENSE).
