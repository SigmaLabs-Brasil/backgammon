# PGO/BOLT release build notes

The workspace is configured for the always-on low-risk optimizations requested
for SIG-291: release LTO=fat, codegen-units=1, symbols stripped, mimalloc in the
CLI, and x86-64-v3 target CPU flags in `.cargo/config.toml`.

Profile-guided build flow on a native x86_64 Linux host:

```bash
cargo install cargo-pgo
cargo pgo build --release --bin gnubg
./target/x86_64-unknown-linux-gnu/release/gnubg bench --positions 100000 --candidates 8
cargo pgo optimize --release --bin gnubg
```

If `cargo-pgo` is unavailable, collect LLVM profiles manually with
`RUSTFLAGS="-Cprofile-generate=/tmp/gnubg-pgo"`, run the `bench` command, merge
profiles with `llvm-profdata`, then rebuild with
`RUSTFLAGS="-Cprofile-use=/tmp/gnubg.profdata"`.

Optional BOLT pass, when `cargo-bolt` and Linux perf are available:

```bash
cargo install cargo-bolt
cargo bolt build --release --bin gnubg -- -reorder-blocks=ext-tsp -reorder-functions=hfsort
```
