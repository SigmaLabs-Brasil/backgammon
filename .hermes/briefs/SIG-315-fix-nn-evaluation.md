# Brief: SIG-315 — Fix Neural Network Evaluation (Saturated Outputs)

## Context

The backgammon engine's neural network evaluation is broken — every position evaluates to `win=100%` for the player to move (`[1.0, 0.0, 0.0, 0.0, 0.0]` after SanityCheck). This affects ALL commands: `evaluate`, `best-move`, `analyze`, `play`.

### What was already fixed (SIG-315a)
Two bugs in `input encoding` were corrected by Neo:

1. **Missing bar encoding**: `base_inputs()` in `gnubg-eval/src/inputs.rs` iterated `1..=24` instead of `0..=24` — skipped the bar (point 0). Fixed: loop now covers all 25 points, `BASE_INPUTS` constant changed from 96 to 100.

2. **Missing opponent side**: `calculate_contact_inputs()` / `calculate_crashed_inputs()` only encoded ONE side's base inputs (`BASE_INPUTS` = 96 slots), leaving the 104 slots reserved for the opponent's base and bar always zero. Fixed: both sides are now encoded (`BASE_INPUTS_FULL` = 100 per side, 200 total base inputs).

**All 93 tests pass** and release build compiles cleanly.

### The remaining problem
Even with correct inputs, the neural network still produces saturated outputs for every position tested:

| Position | Raw NN | Hidden mean | Classification |
|----------|--------|-------------|----------------|
| Opening (`4HPwATDgc/ABMA`) | [1.0, 0.0, 1.0, 0.0, 0.0] | 0.979 | Contact |
| After 31 played (8/5 6/5) | [1.0, 0.0, 1.0, 0.0, 0.0] | 0.979 | Contact |
| Pure race | [1.0, 1.0, 0.0, 0.0, 1.0] | — | Race |

Root cause: the hidden thresholds are extremely negative (min=-898, mean=-55 for contact network) and the output weights are very large (range [-214, +523]). Combined with 128 hidden neurons, the weighted sums at the output layer always push the sigmoid to saturation regardless of input signal.

## Key suspicion

The weights file `gnubg-sys/vendor/gnubg.weights` has `n_trained=0` for ALL 6 networks — suspicious because properly trained gnubg weights should have non-zero training counts. The file is ~1.1MB ASCII with 101,973 lines. The header matches standard gnubg format ("GNU Backgammon 1.01"), architecture lines parse correctly, and the total float counts match expected dimensions. But the weight magnitudes suggest untrained or partially trained weights.

The C reference test (`contact_network_zero_input_matches_c_reference`) uses a standalone C harness (`gnubg-eval/tests/c-ref/harness.c`) that reads the same weights file and produces the same saturated output for zero input — so the parsing and NN evaluation logic is correct.

## Objective

Fix the NN evaluation so it produces reasonable win probabilities for backgammon positions (e.g., opening position should show ~50-52% win for the player to move).

## Investigation approach (suggested)

### Option A: Download official trained weights
The standard gnubg distribution ships trained weights as `gnubg.wd` (binary) or `gnubg.weights` (ASCII). A standard trained file should produce sensible evaluations. Sources:
- Official gnubg release tarballs: https://www.gnu.org/software/gnubg/
- Debian/Ubuntu package: `gnubg-data` contains trained weights
- The standard trained file is typically ~50MB (ours is 1.1MB — very suspicious)

The Coder should attempt to:
1. Download or locate the official gnubg.weights (trained, not template)
2. Compare float counts and headers with our current file
3. If a proper file is found, replace `gnubg-sys/vendor/gnubg.weights` and re-run tests

### Option B: Verify against gnubg C source
If Option A doesn't yield a better file, verify that the weight layout matches gnubg's `neuralnet.c` exactly:
1. Read the gnubg source for `NeuralNetLoad()` to confirm weight order in file
2. Cross-check our `feed_forward_scalar_impl()` against the C `Evaluate()` function
3. Verify input encoding matches `BaseInputs()` in gnubg's C source for contact/race/crashed

### Option C: Diagnostic — generate random weights
Create a synthetic test: generate a small random NN (e.g., 10→5→5) with controlled biases to verify that the forward pass can produce non-saturated outputs with appropriate weights. This helps isolate whether the issue is the weights file vs. the computation.

## Files to investigate

| File | Role |
|------|------|
| `gnubg-sys/vendor/gnubg.weights` | The weights file (1.1MB, n_trained=0) |
| `gnubg-eval/src/weights.rs` | Weight parser — verify layout matches C |
| `gnubg-eval/src/neuralnet.rs` | Forward pass — verify against C Evaluate() |
| `gnubg-eval/src/inputs.rs` | Input encoding — base_inputs was already fixed |
| `gnubg-eval/src/contact.rs` | Contact input encoder — already fixed |
| `gnubg-eval/src/crashed.rs` | Crashed input encoder — already fixed |
| `gnubg-eval/src/race.rs` | Race input encoder — verify correctness |
| `gnubg-eval/src/sanity.rs` | SanityCheck — verify normalization (may mask issues) |

## Out of scope

- Search depth / alpha-beta correctness
- Cubeful equity / cube decisions
- WASM or GUI integration
- Performance tuning
- Any file outside the `gnubg-eval` crate

## Acceptance criteria

1. `gnubg evaluate 4HPwATDgc/ABMA` shows win < 60% (reasonable for opening position)
2. `gnubg evaluate` on 3+ different positions produces different win probabilities (not all 100%)
3. `gnubg best-move 4HPwATDgc/ABMA 31 --depth 2` selects a reasonable opening move (e.g., 8/5 6/5)
4. All 93 existing tests still pass
5. `cargo build --release` has 0 warnings
6. The C reference harness (`gnubg-eval/tests/c-ref/harness.c`) should produce the SAME raw NN output as the Rust implementation for the opening position inputs

## Branch

`fix/SIG-315-nn-evaluation` from `main`

## Additional context from investigation

The cross-validation test `opening_position_matches_post_sanity_check_smoke_values` currently expects `[1.0, 0.0, 0.0, 0.0, 0.0]` — this expectation was written based on the broken weights. **It MUST be updated** once a correct weights file is obtained. The test should use known-good reference values from the C gnubg.
