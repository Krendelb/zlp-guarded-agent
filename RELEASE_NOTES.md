# v0.1.0 — Research preview

Initial public Rust core with fixed 64-byte frames, borrowed Lake ranges,
identity-bound model proposals, exact read-only tool admission, and a
deterministic gate.

Run `cargo run --example tool_admission` to reproduce a wrong READ_FILE
proposal being quarantined and the authorized SEARCH_FILES proposal being
admitted. The example requires no model weights and executes no tools.

Validation: 36 core tests, the executable demonstration, rustfmt, and strict
Clippy. The earlier bounded local-model experiment remains 14/20 for exact
tool selection; all six mismatches were quarantined by the new gate.

Limitations: trusted observation correctness remains the caller's
responsibility. CRC32 is not authentication. This release grants no execution
or publication authority and makes no production-readiness claim.
