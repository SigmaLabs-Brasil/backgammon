# GNU Backgammon C reference harness

This directory contains optional tooling for cross-validating the Rust neural-network forward pass against a small standalone C implementation of GNU Backgammon's `Evaluate()` logic.

The Rust integration test in `../cross_validation.rs` is the CI smoke test. Building this C harness is optional and intended for local investigation or regenerating expected values.

## Build

```bash
cd gnubg-eval/tests/c-ref
make
```

The build requires `gcc` and links against `libm`.

## Run

```bash
make run
```

The harness loads weights from:

```text
../../../gnubg-sys/vendor/gnubg.weights
```

By default it evaluates an all-zero 250-float contact-network input and prints the hidden-layer stats plus the 5 output values.

## Regenerating expected values

1. Build the Rust code and generate a 250-float little-endian input vector at `/tmp/rust_inputs.bin` for the position you want to inspect.
2. Run `make run` from this directory.
3. Copy the printed `With Rust inputs output: [...]` values into the Rust smoke test only after verifying they are stable and within the intended epsilon.

If `/tmp/rust_inputs.bin` is absent, the harness still prints the zero-input reference output and skips the arbitrary-input section.

## Clean

```bash
make clean
```
