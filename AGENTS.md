# AGENTS: SIG-291 — Pure Rust Neural Network Evaluation

**For:** Coder (Codex / Cursor / Claude Code)
**Repo:** SigmaLabs-Brasil/backgammon
**Branch:** spec/SIG-291 (this branch)

---

## 1. Stack & Target

| Layer | Constraint |
|---|---|
| Language | Rust 2021 edition, MSRV 1.78 |
| Target | `x86_64-unknown-linux-gnu` only |
| SIMD | `std::arch::x86_64` intrinsics (AVX2 + SSE2) |
| Dependencies | `rayon` only (already in workspace) |
| Allocator | `mimalloc` (via gnubg-cli, not in gnubg-eval) |
| No FFI | Zero `extern "C"`, zero `cc` crate, zero `links` |
| Profile | `target-cpu=x86-64-v3` (`.cargo/config.toml`) |

---

## 2. Crate Structure

Create the crate at `gnubg-eval/` in the workspace root.

### 2.1 Cargo.toml

```toml
[package]
name = "gnubg-eval"
version = "0.1.0"
edition.workspace = true
license.workspace = true
repository.workspace = true
rust-version.workspace = true

[dependencies]
rayon = "1.10"
```

No `[features]`, no `[build-dependencies]`, no `[dev-dependencies]` beyond what's implied.

### 2.2 File listing (order of creation)

```
gnubg-eval/
├── Cargo.toml
└── src/
    ├── lib.rs
    ├── weights.rs
    ├── neuralnet.rs
    ├── classify.rs
    ├── inputs.rs
    ├── race.rs
    ├── contact.rs
    └── crashed.rs
```

---

## 3. Module Implementation Order

Respect the dependency graph. Each module must compile before the next one starts.

### Phase 1: Weight Parser (`weights.rs`)
- Input: `&str` (the entire ASCII weights file)
- Output: `WeightFile` struct
- **START HERE.** Everything else depends on parsed weights.
- Must handle exactly 32,773 float values after the header
- No `unsafe` needed — pure string parsing

### Phase 2: Neural Net Scalar (`neuralnet.rs`)
- `NeuralNet::new(&WeightFile, c_input) -> Self`
- `feed_forward_scalar(&self, inputs: &[f32]) -> [f32; 5]`
- Two loops: hidden layer (c_input × c_hidden) + output layer (c_hidden × c_output)
- Use `f32::tanh()` and `1.0 / (1.0 + (-x).exp())` for sigmoid
- **NO SIMD yet.** Get scalar correct first.

### Phase 3: Position Classification (`classify.rs`)
- `classify_position(board: &Board) -> Classification`
- Port from `eval.c ClassifyPosition` (lines ~5910-5968)
- Use `gnubg_types::Board` = `[[u32; 25]; 2]`

### Phase 4: Input Encoding (`inputs.rs`)
- `base_inputs(board, side) -> [f32; 96]`
- `calculate_half_inputs(board, side) -> [f32; 25]`
- (Optional) hit probability lookup tables as `const` arrays

### Phase 5: Per-Classification Encoders
- `race.rs`: `calculate_race_inputs(board) -> [f32; 92]`
- `contact.rs`: `calculate_contact_inputs(board) -> [f32; 250]`
- `crashed.rs`: `calculate_crashed_inputs(board) -> [f32; 250]`

### Phase 6: Public API (`lib.rs`)
- `init_weights()` — called once, parses weights, builds 3 networks
- `evaluate(board: &Board) -> EvalOutput` — full pipeline
- `simd_supported() -> bool`

### Phase 7: AVX2 Forward Pass
- Add `feed_forward_avx2` in `neuralnet.rs`
- `#[cfg(target_arch = "x86_64")]` guarded
- Runtime dispatch in `NeuralNet::new()` via `is_x86_feature_detected!("avx2")`
- Function pointer `forward_fn` stores the chosen implementation

### Phase 8: Integration
- Modify `gnubg-search/src/lib.rs` — replace `gnubg_sys::evaluate_position_key` call
- Modify `gnubg-search/Cargo.toml` — swap dependency
- Modify workspace `Cargo.toml` — add member
- Modify `gnubg-sys` — remove FFI, keep types

### Phase 9: Cleanup (LAST)
- Remove C vendor files from `gnubg-sys/vendor/`
- Remove `gnubg-sys/build.rs`
- Update `gnubg-sys/Cargo.toml` (drop `links`, `build`, `cc`)

**Do not remove `gnubg-sys/vendor/gnubg.weights`** — gnubg-eval embeds it via `include_str!`.

---

## 4. Conventions

### 4.1 Code Style

- Follow `rustfmt` defaults (no custom rustfmt.toml needed)
- No `unsafe` except in AVX2 intrinsics (which are inherently unsafe)
- Every `unsafe` block must have a `// SAFETY:` comment
- Use `#![forbid(unsafe_code)]` in modules without SIMD
- Prefer `&[f32]` over `&Vec<f32>` in function signatures
- Use `[f32; N]` fixed arrays for input/output vectors where N is known at compile time

### 4.2 Error Handling

```rust
#[derive(Debug)]
pub enum EvalError {
    WeightsNotInitialized,
    WeightsParseError(String),
    InvalidBoard,
    InvalidInputLength { expected: usize, got: usize },
}

impl std::fmt::Display for EvalError { ... }
impl std::error::Error for EvalError {}
```

- `init_weights()` panics on parse failure (weights file is embedded — can't recover)
- `evaluate()` returns `Result<EvalOutput, EvalError>` for runtime errors

### 4.3 Testing Standards

Every module must have `#[cfg(test)] mod tests` at the bottom.

**Minimum tests per module:**
| Module | Tests |
|---|---|
| `weights.rs` | Parse real file, parse header, parse trailing newline, error on truncated, verify float count |
| `neuralnet.rs` | Forward pass on identity weights, forward pass on zeros, tanh/sigmoid range check, output in [0,1] |
| `classify.rs` | Opening position → Contact, both-borne-off → Crashed, race position → Race |
| `inputs.rs` | Empty board → all zeros, one checker → correct slot, 5 checkers → scaled slot 3 |
| `race.rs` | Known position → expected input length 92 |
| `contact.rs` | Known position → expected input length 250 |
| `integration` (in lib.rs) | `evaluate()` on opening position → outputs in [0,1], win > 0.5 |

### 4.4 Performance Rules

- Hot loops must NOT allocate. All input/output vectors are stack-allocated `[f32; N]`.
- The `feed_forward` path must be zero-allocation after init.
- Cache the `forward_fn` pointer — do NOT check `is_x86_feature_detected` on every call.
- Do NOT use `Vec::push` inside `feed_forward`. Pre-allocate `[f32; 128]` for hidden layer.
- The `base_inputs` function returns a fixed array, not a Vec.

### 4.5 No-Go Rules

- Do NOT add `libc`, `cc`, `cmake`, `bindgen`, or any C compilation step
- Do NOT call `extern "C"` functions
- Do NOT use `include_bytes!` on `.c` files — embed only `.weights`
- Do NOT change `gnubg-types`, `gnubg-moves`, or `gnubg-search/src/search.rs` (only `lib.rs` integration)
- Do NOT remove `gnubg-sys/vendor/gnubg.weights` (gnubg-eval embeds it)
- Do NOT change the `EvalResult` struct in `gnubg-search` — its shape is correct
- Do NOT use `println!` in library code — only in tests

---

## 5. Key Implementation Details

### 5.1 Weight File Parsing

The `gnubg.weights` file is accessible from `gnubg-eval/src/` via:

```rust
const WEIGHTS_DATA: &str = include_str!("../../gnubg-sys/vendor/gnubg.weights");
```

Parse it line by line:
1. Skip empty lines
2. First non-empty line: `GNU Backgammon 1.01` (verify magic)
3. Second non-empty line: `250 128 5 0 0.1000000 1.0000000`
4. Parse: `nInput nHidden nOutput nTrained betaHidden betaOutput`
5. Remaining lines: parse as `f32`, expect exactly `nInput * nHidden + nHidden + nHidden * nOutput + nOutput` floats

The file has exactly 101,973 lines (verified). Lines 3–32002 are hidden weights, 32003–32130 are hidden thresholds, 32131–32770 are output weights, 32771–32775 are output thresholds.

### 5.2 Neural Net Weight Layout

The hidden weights are stored row-major: `ar_hidden_weight[input_idx * c_hidden + hidden_idx]`.

```
// Hidden layer: for each of c_hidden neurons:
//   sum = threshold[j] + Σ(input[i] * weight[i * c_hidden + j])  for i in 0..c_input
//   hidden[j] = tanh(beta_hidden * sum)

// Output layer: for each of c_output outputs:
//   sum = output_threshold[k] + Σ(hidden[j] * weight[j * c_output + k])  for j in 0..c_hidden
//   output[k] = sigmoid(beta_output * sum)
```

### 5.3 AVX2 Implementation

The hidden layer is the bottleneck (250×128 = 32K multiply-adds). AVX2 processes 8 f32 values at once:

```rust
#[target_feature(enable = "avx2")]
unsafe fn feed_forward_avx2(net: &NeuralNet, inputs: &[f32]) -> [f32; 5] {
    for j in 0..c_hidden {
        let mut sum = _mm256_setzero_ps();  // 8-wide accumulator
        let threshold = _mm256_set1_ps(ar_hidden_threshold[j]);  // broadcast
        sum = _mm256_add_ps(sum, threshold);

        for i in (0..c_input).step_by(8) {
            let input_chunk = _mm256_loadu_ps(&inputs[i]);      // load 8 inputs
            let weight_chunk = _mm256_loadu_ps(&ar_hidden_weight[i * c_hidden + j]); // 8 weights
            sum = _mm256_fmadd_ps(input_chunk, weight_chunk, sum); // fused multiply-add
        }
        // Horizontal sum: sum[0]+sum[1]+...+sum[7]
        hidden[j] = tanh_approx(horizontal_sum(sum) * beta_hidden);
    }
    // ... output layer (scalar is fine for 128×5 = 640 ops)
}
```

Wait — this layout is wrong. `ar_hidden_weight[i * c_hidden + j]` means weights for *different inputs to the SAME hidden neuron* are contiguous. But AVX2 wants 8 different hidden neurons at once.

**Correct AVX2 approach:** Process 8 hidden neurons in parallel. Each inner iteration loads 1 input value, broadcasts it, and does 8 FMA operations (one per neuron):

```rust
for i in 0..c_input {
    let input = _mm256_set1_ps(inputs[i]);  // broadcast
    for j in (0..c_hidden).step_by(8) {
        let weight_chunk = _mm256_loadu_ps(&ar_hidden_weight[i * c_hidden + j]);
        accum[j..j+7] = _mm256_fmadd_ps(input, weight_chunk, accum[j..j+7]);
    }
}
```

This requires `ar_hidden_weight` to be contiguous for 8 adjacent hidden neurons for the same input — the `[i * c_hidden + j]` layout already gives this.

### 5.4 Sigmoid Approximation for AVX2

For AVX2, a polynomial approximation is faster than `1.0 / (1.0 + exp(-x))`:

```rust
// Fast sigmoid: 0.5 + 0.5 * tanh(x/2)
// Then use minmax polynomial for tanh
fn fast_sigmoid(x: f32) -> f32 {
    let x2 = x * x;
    let a = x + 0.16489087 * x * x2 + 0.00985468 * x2 * x2 * x;
    0.5 + 0.5 * a / (1.0 + a.abs())
}
```

Max error: ~0.00015 across [-6, 6]. Good enough for ±0.05 output tolerance.

### 5.5 Boards to Input Conversion Gotcha

The `Board` type in `gnubg-types` uses C indexing conventions:
- `board[0]` = opponent (top of screen)
- `board[1]` = player (bottom of screen)
- Point 0 = bear-off tray (checkers already removed)
- Points 1..24 = board points
- Point 24 = bar

When encoding `base_inputs` for the player:
- Iterate points 1..24
- `board[1][p]` = checkers at point p for the player
- Encode into slots 0..95 of the output

When encoding for the opponent, use `board[0][p]`.

### 5.6 HalfInputs Pitfalls

The `CalculateHalfInputs` function in `eval.c` is ~200 lines with many array indices. Key things to get right:

1. **I_OFF[1], I_OFF[2], I_OFF[3]:** Borne-off checkers. `I_OFF[1] = board[side][0] / 15.0`. `I_OFF[2]` and `I_OFF[3]` are progress indicators: how many checkers are past the 6-point, 9-point, etc.

2. **I_BREAK_CONTACT:** `board[side][opponent_home_points_19_24].iter().sum() == 0` → 1.0, else 0.0. If the player has no checkers behind the 18-point, contact is broken.

3. **I_BACK_CHEQUER:** Farthest-back checker position (max point index with checkers), normalized to [0, 1] as `(back_point - 1) / 23.0`.

4. **Hit probability tables:** Two const arrays from `eval.c`:
   - `aanCombination[6][5]` — combinatory counts for hit probabilities
   - `aIntermediate[39]` — pre-computed intermediate values
   - Copy these as `const` arrays in Rust

5. **aHit[39]:** 39 intermediate features. These use the pre-computed tables with board-specific indices. The formulas are:
   - `I_BREAK_CONTACT * interaction_probability(board)` for various checker positions
   - Each `aHit[k]` encodes the probability that a specific set of points will be hit

### 5.7 Race Input Dimensions

`nnRace.c_input = 92`. The race network uses exactly 92 inputs, but `baseInputs` produces 192 (96 per side). The C code only copies the first 46 per side:

```c
// eval.c, race encoding:
for (i = 0; i < 46; ++i) {
    afInput[i] = arBaseInput[0][i];       // opponent's first 46
    afInput[46 + i] = arBaseInput[1][i];  // player's first 46
}
```

Why 46? Because nnRace.c_input = 92 / 2 = 46 per side. The higher-indexed baseInputs encode denser checker stacks which are only relevant in contact positions.

Internally, `nnRace.c_input` is set as `((25+2+1+6+72) * 2) / 2 * 2 = 92` where:
- 25 = board points with ≥1 checker per side
- 2 = extra features
- 1 = I_BREAK_CONTACT
- 6 = acey-deucy features
- 72 = more features
- `* 2` = both sides
- `/ 2 * 2` = round down to even (SIMD alignment)

For the Rust port, we only need to produce 92 floats. The exact mapping matters.

---

## 6. Integration Checklist

When integrating `gnubg-eval` into `gnubg-search`, follow this exact diff:

### `gnubg-search/src/lib.rs`

1. Add `use gnubg_eval;` to imports
2. Add `use gnubg_types::board_from_old_key;` to imports
3. Change `evaluate_key_with_thread_cache`:

```rust
pub fn evaluate_key_with_thread_cache(
    key: PositionKey,
    depth: u8,
) -> Result<EvalResult, SearchError> {
    EVAL_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if let Some(hit) = cache.lookup(&key, depth) {
            return Ok(hit);
        }
        // NEW: decode key → Board → gnubg_eval
        let gt_key = gnubg_types::PositionKey::from_raw(key.0);
        let board = gnubg_types::board_from_old_key(&gt_key);
        let eval_output = gnubg_eval::evaluate(&board).map_err(|e| {
            SearchError::Eval(e.to_string().into())
        })?;
        let raw = gnubg_sys::RawEval {
            outputs: [
                eval_output.win,
                eval_output.win_gammon,
                eval_output.win_backgammon,
                eval_output.lose_gammon,
                eval_output.lose_backgammon,
            ],
        };
        let eval = EvalResult::from_raw(raw, depth, false);
        cache.insert(key, depth, eval);
        Ok(eval)
    })
}
```

4. Add a new `SearchError` variant or convert from `EvalError`:

```rust
#[derive(Debug)]
pub enum SearchError {
    Ffi(GnuBgError),
    Eval(String),  // NEW
    EmptyMoveList,
}
```

### `gnubg-search/Cargo.toml`

```diff
 [dependencies]
-gnubg-sys = { path = "../gnubg-sys" }
+gnubg-sys = { path = "../gnubg-sys" }  # still needed for PositionKey, RawEval
+gnubg-eval = { path = "../gnubg-eval" }
 gnubg-types = { path = "../gnubg-types" }
 gnubg-moves = { path = "../gnubg-moves" }
 rayon = "1.10"
```

### Workspace `Cargo.toml`

```diff
-members = ["gnubg-sys", "gnubg-types", "gnubg-moves", "gnubg-search", "gnubg-cli"]
+members = ["gnubg-sys", "gnubg-types", "gnubg-moves", "gnubg-search", "gnubg-cli", "gnubg-eval"]
```

---

## 7. Verification Commands

Run these commands to validate each phase:

```bash
# Phase 1 — weights parser
cargo test -p gnubg-eval -- weights

# Phase 2 — scalar neural net
cargo test -p gnubg-eval -- neuralnet

# Phase 3 — classification
cargo test -p gnubg-eval -- classify

# Phase 4-5 — input encoding
cargo test -p gnubg-eval -- inputs
cargo test -p gnubg-eval -- race

# Phase 6 — full integration
cargo test -p gnubg-eval

# Phase 8 — workspace integration
cargo test -p gnubg-search

# Final validation
cargo test --workspace
cargo clippy -p gnubg-eval -- -D warnings
cargo build --release
./target/release/gnubg evaluate 4HPwATDgc/ABMA
```

---

## 8. Common Pitfalls

1. **Weights array indexing:** `ar_hidden_weight` is 250×128 = 32,000 floats. `ar_hidden_threshold` is 128. The formula for index `k` in the flat array: `k = i * c_hidden + j` where `i` is input index (0..c_input) and `j` is hidden neuron index (0..c_hidden).

2. **Float parsing locale:** Use `f32::from_str()`. Do NOT use any locale-dependent parsing. The weights file uses `.` as decimal separator.

3. **SIMD alignment:** `_mm256_loadu_ps` (unaligned) is fine for the general case. The weights `Vec<f32>` may not be 32-byte aligned. Use `loadu`, not `load`.

4. **Race condition on `init_weights()`:** Use `std::sync::Once` (same pattern as current `gnubg-sys`). The init function is called lazily on first `evaluate()`.

5. **Board reconstruction cost:** `board_from_old_key()` is called once per cache miss. It's O(80) bit operations — negligible next to the 32K multiply-adds of the forward pass.

6. **Hidden layer pre-allocation:** Allocate `[0.0f32; 128]` on the stack for the hidden layer. Do NOT use `Vec` in the hot path.

7. **Tanh range:** `f32::tanh()` returns [-1, 1]. The hidden layer outputs feed into sigmoid in the output layer, which maps to [0, 1]. The final output is guaranteed to be in [0, 1] because sigmoid(any_real) ∈ (0, 1).

8. **Clippy lint `excessive_precision`:** The weights file has 7 decimal places. `f32` can represent ~6-7 significant digits. Clippy may warn about excessive precision when you write `0.1000000f32`. Use `0.1_f32` instead — the parsed values from the file will have the same precision.

---

## 9. Deliverable Checklist

- [ ] `gnubg-eval/Cargo.toml` created
- [ ] All 7 `.rs` files implemented and tested
- [ ] `gnubg-search/src/lib.rs` integration done
- [ ] `gnubg-search/Cargo.toml` updated
- [ ] Workspace `Cargo.toml` updated
- [ ] `gnubg-sys` cleaned up (C files removed, build.rs removed, Cargo.toml stripped)
- [ ] `cargo test --workspace` passes
- [ ] `cargo clippy --workspace -- -D warnings` passes
- [ ] CLI `evaluate 4HPwATDgc/ABMA` outputs plausible values
- [ ] `bench --positions 1000` runs without errors
- [ ] No C compilation step (`cargo clean && cargo build 2>&1 | grep cc` returns empty)
