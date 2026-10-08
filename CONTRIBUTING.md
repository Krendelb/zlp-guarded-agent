# Contributing

Changes to the trust boundary must include negative regression tests. A change
is not accepted solely because an LLM output looks reasonable.

Before submitting a change, run:

```bash
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt -- --check
```

Keep model weights, private data, network credentials, generated build output,
and unrelated ZLP research out of this repository.
