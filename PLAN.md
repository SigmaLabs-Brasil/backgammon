# PLAN: SIG-291 — Pure Rust Neural Network Evaluation

**Version:** 1.0.0
**Estimate:** ~16-20 hours (single developer, familiar with Rust and gnubg internals)
**Milestones:** 9 sequential phases, each gated by compile + tests

---

## Dependency Graph

```
Phase 1: Scaffold + weights.rs
    │
Phase 2: neuralnet.rs (scalar)
    │
Phase 3: classify.rs
    │
Phase 4: inputs.rs
    │
Phase 5: race.rs, contact.rs, crashed.rs
    │
Phase 6: lib.rs (public API)
    │
Phase 7: neuralnet.rs (AVX2)
    │
Phase 8: Integration (gnubg-search)
    │
Phase 9: Cleanup (gnubg-sys)
```

Each phase can be implemented **independently** within itself, but must be **sequential** across phases.

---

## Phase 1: Scaffold + Weight Parser

**Files:** `gnubg-eval/Cargo.toml`, `gnubg-eval/src/lib.rs` (stub), `gnubg-eval/src/weights.rs`
**Deps:** None (greenfield crate)
**Estimate:** 1.5 hours

### Tasks

| # | Task | File | Est. |
|---|---|---|---|
| 1.1 | Create `gnubg-eval/Cargo.toml` with workspace inheritance + rayon dep | `Cargo.toml` | 15 min |
| 1.2 | Create stub `lib.rs` with `WeightFile` struct | `lib.rs` | 10 min |
| 1.3 | Implement `parse_weights()` — header parser, dimension validation, float parsing | `weights.rs` | 45 min |
| 1.4 | Write tests: parse real file, validate float count, error on truncated input | `weights.rs` (tests) | 20 min |

### Validation Gate

```bash
cargo test -p gnubg-eval -- weights
# Expected: 4 tests pass
# - parses real gnubg.weights header
# - verifies 32000 hidden weights
# - verifies 128 hidden thresholds
# - verifies 640 output weights + 5 output thresholds
```

---

## Phase 2: Scalar Neural Net

**Files:** `gnubg-eval/src/neuralnet.rs`
**Deps:** Phase 1
**Estimate:** 2.5 hours

### Tasks

| # | Task | File | Est. |
|---|---|---|---|
| 2.1 | Implement `NeuralNet::new(&WeightFile, c_input)` — copy weight slices | `neuralnet.rs` | 30 min |
| 2.2 | Implement `feed_forward_scalar()` — hidden layer tanh, output layer sigmoid | `neuralnet.rs` | 45 min |
| 2.3 | Implement helper: `sigmoid(x)` and test range | `neuralnet.rs` | 15 min |
| 2.4 | Write tests: identity weights, zero weights, output range [0,1] | `neuralnet.rs` (tests) | 30 min |
| 2.5 | Build small synthetic network (2×2×1) and verify forward pass manually | `neuralnet.rs` (tests) | 30 min |

### Validation Gate

```bash
cargo test -p gnubg-eval -- neuralnet
# Expected: 5+ tests pass
# - test_scalar_forward_output_range: all outputs in [0,1]
# - test_small_network_manual: 2×2×1 network matches hand calculation
# - test_zero_weights: all outputs ≈ 0.5 (sigmoid(0) = 0.5)
# - test_nnrace_dimensions: c_input=92, c_hidden=128, c_output=5
# - test_nncontact_dimensions: c_input=250, c_hidden=128, c_output=5
```

---

## Phase 3: Position Classification

**Files:** `gnubg-eval/src/classify.rs`
**Deps:** Phase 1 (uses Board type from gnubg-types)
**Estimate:** 1.5 hours

### Tasks

| # | Task | File | Est. |
|---|---|---|---|
| 3.1 | Port `ClassifyPosition` from eval.c lines ~5910-5968 | `classify.rs` | 30 min |
| 3.2 | Implement `Classification` enum with Display | `classify.rs` | 15 min |
| 3.3 | Port helpers: sum back checkers, count total checkers per side | `classify.rs` | 20 min |
| 3.4 | Write tests: opening position, race position (no contact), crashed position, bearoff | `classify.rs` (tests) | 25 min |

### Validation Gate

```bash
cargo test -p gnubg-eval -- classify
# Expected: 4+ tests pass
# - test_opening_position_is_contact: 4HPwATDgc/ABMA → Contact
# - test_race_position: both sides have broken contact → Race
# - test_crashed_position: one side has ≤6 checkers → Crashed
# - test_bearoff_position: all checkers in home board → Race (bearoff sub-class)
```

---

## Phase 4: Input Encoding Primitives

**Files:** `gnubg-eval/src/inputs.rs`
**Deps:** Phase 3 (uses Board)
**Estimate:** 3.0 hours (highest complexity)

### Tasks

| # | Task | File | Est. |
|---|---|---|---|
| 4.1 | Implement `base_inputs(board, side) → [f32; 96]` — MINPPERPOINT(4) encoding | `inputs.rs` | 45 min |
| 4.2 | Write base_inputs tests: empty board, one checker, full point, 5+ checkers | `inputs.rs` (tests) | 20 min |
| 4.3 | Copy `aanCombination` and `aIntermediate` tables from eval.c as `const` arrays | `inputs.rs` | 15 min |
| 4.4 | Implement `calculate_half_inputs(board, side) → [f32; 25]` | `inputs.rs` | 90 min |
| 4.5 | Write half_inputs tests: opening position, key features verified | `inputs.rs` (tests) | 20 min |

### Validation Gate

```bash
cargo test -p gnubg-eval -- inputs
# Expected: 8+ tests pass
# - test_base_inputs_empty: all 96 values are 0.0
# - test_base_inputs_one_checker: slot 0 = 1.0, rest = 0.0
# - test_base_inputs_full_point: slots 0-2 = 1.0, slot 3 = scaled
# - test_base_inputs_side_independent: player vs opponent different slices
# - test_half_inputs_off_checkers: borne-off counts correct
# - test_half_inputs_break_contact: 0.0 or 1.0
# - test_half_inputs_back_chequer: within [0, 1]
# - test_hit_tables_present: arrays are non-empty
```

---

## Phase 5: Per-Classification Encoders

**Files:** `gnubg-eval/src/race.rs`, `gnubg-eval/src/contact.rs`, `gnubg-eval/src/crashed.rs`
**Deps:** Phase 4
**Estimate:** 2.0 hours

### Tasks

| # | Task | File | Est. |
|---|---|---|---|
| 5.1 | Implement `calculate_race_inputs() → [f32; 92]` — base_inputs truncated to 46 per side | `race.rs` | 30 min |
| 5.2 | Write race test: verify output length, values in [0,1] | `race.rs` (tests) | 10 min |
| 5.3 | Implement `calculate_contact_inputs() → [f32; 250]` — base + half + extra | `contact.rs` | 45 min |
| 5.4 | Write contact test: verify length, known features at expected indices | `contact.rs` (tests) | 15 min |
| 5.5 | Implement `calculate_crashed_inputs() → [f32; 250]` — same layout, crashed-specific encoding | `crashed.rs` | 30 min |
| 5.6 | Write crashed test: verify length, values in [0,1] | `crashed.rs` (tests) | 10 min |

### Validation Gate

```bash
cargo test -p gnubg-eval -- race contact crashed
# Expected: 6+ tests pass
# - test_race_inputs_length: outputs.len() == 92
# - test_race_inputs_opening: all values in [0,1]
# - test_contact_inputs_length: outputs.len() == 250
# - test_contact_inputs_opening: I_BREAK_CONTACT at expected index
# - test_crashed_inputs_length: outputs.len() == 250
# - test_crashed_inputs_range: all values in [0,1]
```

---

## Phase 6: Public API (`lib.rs`)

**Files:** `gnubg-eval/src/lib.rs` (replace stub)
**Deps:** Phase 2, 3, 5
**Estimate:** 1.5 hours

### Tasks

| # | Task | File | Est. |
|---|---|---|---|
| 6.1 | Implement `init_weights()` with `Once` — parse, build 3 networks | `lib.rs` | 30 min |
| 6.2 | Implement `evaluate(board) → EvalOutput` — classify → encode → feed_forward | `lib.rs` | 30 min |
| 6.3 | Implement `simd_supported() → bool` | `lib.rs` | 10 min |
| 6.4 | Write integration tests: evaluate opening position, output range, win > 0.5 | `lib.rs` (tests) | 20 min |

### Validation Gate

```bash
cargo test -p gnubg-eval
# Expected: ALL tests pass (~25+ tests across all modules)
# Key integration tests:
# - test_evaluate_opening_position: win > 0.50
# - test_evaluate_outputs_in_range: all 5 outputs in [0,1]
# - test_evaluate_deterministic: same position → same outputs
```

---

## Phase 7: AVX2 Forward Pass

**Files:** `gnubg-eval/src/neuralnet.rs` (extend)
**Deps:** Phase 2, 6
**Estimate:** 2.0 hours

### Tasks

| # | Task | File | Est. |
|---|---|---|---|
| 7.1 | Implement `feed_forward_avx2()` using `std::arch::x86_64` intrinsics | `neuralnet.rs` | 60 min |
| 7.2 | Implement fast sigmoid via tanh polynomial approximation for AVX2 | `neuralnet.rs` | 20 min |
| 7.3 | Add runtime dispatch in `NeuralNet::new()` via `is_x86_feature_detected!` | `neuralnet.rs` | 20 min |
| 7.4 | Write tests: AVX2 output matches scalar output within ε=1e-4 | `neuralnet.rs` (tests) | 15 min |
| 7.5 | Verify no `unsafe` leaks — only AVX2 intrinsics are unsafe | `neuralnet.rs` | 5 min |

### Validation Gate

```bash
# On AVX2-capable CPU:
cargo test -p gnubg-eval -- neuralnet
# Expected: test_avx2_matches_scalar passes (within 1e-4 per channel)

# On non-AVX2 CPU:
cargo test -p gnubg-eval -- neuralnet
# Expected: AVX2 test skipped, scalar path used, all scalar tests still pass

# Verify runtime detection:
cargo run --release -p gnubg-cli -- evaluate 4HPwATDgc/ABMA
# Output includes simd_supported: true (on AVX2 CPU)
```

---

## Phase 8: Integration

**Files:** `gnubg-search/src/lib.rs`, `gnubg-search/Cargo.toml`, workspace `Cargo.toml`
**Deps:** Phase 6
**Estimate:** 2.0 hours

### Tasks

| # | Task | Est. |
|---|---|---|
| 8.1 | Add `gnubg-eval` member to workspace `Cargo.toml` | 5 min |
| 8.2 | Add `gnubg-eval` dependency to `gnubg-search/Cargo.toml` | 5 min |
| 8.3 | Modify `evaluate_key_with_thread_cache()` to call `gnubg_eval::evaluate()` | 30 min |
| 8.4 | Add `Eval(String)` variant to `SearchError` + conversion impl | 10 min |
| 8.5 | Run full test suite: `cargo test --workspace` | 20 min |
| 8.6 | Fix any test failures (expected: existing tests now use real eval → different equity values, but structure is preserved) | 40 min |
| 8.7 | Run CLI smoke tests: `evaluate`, `best-move`, `analyze`, `bench` | 10 min |

### Validation Gate

```bash
# All workspace tests pass:
cargo test --workspace
# Expected: 0 failures

# CLI evaluation:
cargo run --release -p gnubg-cli -- evaluate 4HPwATDgc/ABMA
# Expected: win ~0.52-0.56, all values in [0,1], output is NOT the hash stub

# CLI best-move:
cargo run --release -p gnubg-cli -- best-move 4HPwATDgc/ABMA 31
# Expected: returns a legal move, equity is a real number

# Benchmark:
cargo run --release -p gnubg-cli -- bench --positions 1000
# Expected: positions_per_second > 1000 (scalar), no errors

# No C compilation:
cargo clean && cargo build --release 2>&1 | grep -c 'cc'
# Expected: 0 (no C compiler invoked)
```

---

## Phase 9: Cleanup

**Files:** `gnubg-sys/Cargo.toml`, `gnubg-sys/src/lib.rs`, delete vendor C files
**Deps:** Phase 8
**Estimate:** 1.0 hour

### Tasks

| # | Task | Est. |
|---|---|---|
| 9.1 | Remove unused C files: `gnubg_bridge.c`, `cache.c`, `eval.c`, `neuralnet.c`, `neuralnetsse.c` from `gnubg-sys/vendor/` | 10 min |
| 9.2 | DO NOT remove `gnubg-sys/vendor/gnubg.weights` (still embedded by gnubg-eval) | — |
| 9.3 | Remove `gnubg-sys/build.rs` | 5 min |
| 9.4 | Strip `gnubg-sys/Cargo.toml`: remove `links`, `build`, `cc` dependency | 10 min |
| 9.5 | Strip `gnubg-sys/src/lib.rs`: remove FFI declarations, `evaluate_position_key`, `neuralnet_evaluate`, `simd_supported`, `embedded_weights_len`, `Once` init | 15 min |
| 9.6 | Keep `PositionKey`, `RawEval`, `decode_position_id`, `GnuBgError` — still used by search code | — |
| 9.7 | Run full test suite again: `cargo test --workspace` | 10 min |
| 9.8 | Verify `gnubg-sys` still compiles as a pure Rust crate: `cargo build -p gnubg-sys` | 5 min |

### Validation Gate

```bash
# gnubg-sys has zero C dependencies:
cargo tree -p gnubg-sys --depth 1
# Expected: no cc, no libc, no C crates

# Full build is pure Rust:
cargo clean && cargo build --release
# Expected: no cc invocation in build output

# All tests pass:
cargo test --workspace
# Expected: 0 failures
```

---

## Summary: Time & Risk

| Phase | Description | Est. (hrs) | Risk | Blocks |
|---|---|---|---|---|
| 1 | Scaffold + weights | 1.5 | Low | 2 |
| 2 | Scalar neural net | 2.5 | Low | 3, 6 |
| 3 | Classification | 1.5 | Low | 4, 5 |
| 4 | Input encoding | 3.0 | HIGH | 5 |
| 5 | Per-class encoders | 2.0 | Medium | 6 |
| 6 | Public API | 1.5 | Low | 7, 8 |
| 7 | AVX2 forward pass | 2.0 | Medium | — |
| 8 | Integration | 2.0 | Medium | 9 |
| 9 | Cleanup | 1.0 | Low | — |
| **Total** | | **17.0** | | |

### Risk Heat Map

- **Phase 4 (HIGH):** Input encoding is the most complex port. `CalculateHalfInputs` has 39 features with non-trivial array indexing. Getting this wrong produces quietly wrong evaluations. Mitigation: cross-validate against gnubg C output for 50 positions.
- **Phases 5, 7, 8 (MEDIUM):** These depend on Phase 4 being correct. Integration risk is test breakage from changed equity values.

### Parallelization Potential

Phases 3 and 2 are independent of each other (both depend on Phase 1). But Phase 3 is small (1.5 hrs) and Phase 2 is rote math — parallelizing them saves at most 1.5 hours and adds coordination overhead. Sequential is simpler.

---

## Milestone Checkpoints

| Checkpoint | After Phase | What Must Work |
|---|---|---|
| M1: Weights Loaded | 1 | `cargo test -p gnubg-eval` parses real file |
| M2: NN Computes | 2 | Scalar forward pass produces [0,1] outputs |
| M3: Positions Classified | 3 | Opening position → Contact |
| M4: Inputs Encoded | 5 | Opening position → 250 floats for contact |
| M5: Eval Works End-to-End | 6 | `evaluate(opening_board)` returns plausible equity |
| M6: Fast Path Ready | 7 | AVX2 matches scalar output |
| M7: Integrated | 8 | CLI `evaluate` uses real NN, not hash stub |
| M8: Ship It | 9 | Zero C dependencies in the build |
