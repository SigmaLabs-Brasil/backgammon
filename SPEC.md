# SPEC: SIG-291 — Pure Rust Neural Network Evaluation

**Version:** 1.0.0
**Status:** Pending Review
**Author:** Max (Especificador Tecnico, Sigma Labs)
**Date:** 2026-06-15

---

## 1. Overview

Replace the hash-stub evaluation in `gnubg-sys/vendor/gnubg_bridge.c` with a **pure Rust** port of the gnubg neural network evaluation engine. The new crate `gnubg-eval` loads the real `gnubg.weights` (250×128×5 architecture), encodes board positions into neural net inputs, and runs the forward pass — producing actual backgammon equity evaluations instead of deterministic pseudo-random noise.

### Why This Exists

The current evaluator (`gnubg_bridge.c`) hashes the position key mixed with the weights file to produce deterministic floats in [0,1]. It does **zero backgammon reasoning**. The alpha-beta search (SIG-293) works, but every evaluation is garbage — the engine plays random move sequences that happen to be deterministic. Without real evaluation, there is no search and no engine.

---

## 2. Architecture

### 2.1 Crate Dependency Graph (New)

```
gnubg-cli
  ├── gnubg-search
  │     ├── gnubg-eval          ← NEW (replaces gnubg-sys for eval)
  │     ├── gnubg-types
  │     └── gnubg-moves
  ├── gnubg-sys                 ← SHRUNK (PositionKey only, eventually removed)
  ├── rayon
  └── mimalloc

gnubg-eval                       ← NEW crate
  ├── rayon (already in workspace)
  └── (no other deps — pure Rust, no cc, no FFI)
```

### 2.2 Module Map: `gnubg-eval`

```
gnubg-eval/
├── Cargo.toml
└── src/
    ├── lib.rs              — Public API: evaluate(), EvalOutput, init_weights()
    ├── weights.rs          — Parse gnubg.weights ASCII format → NeuralNet
    ├── neuralnet.rs        — NeuralNet struct + forward pass (scalar + AVX2)
    ├── classify.rs         — ClassifyPosition (race/contact/crashed/bearoff)
    ├── inputs.rs           — input encoding: baseInputs, CalculateHalfInputs
    ├── race.rs             — CalculateRaceInputs (92 → 5)
    ├── contact.rs          — CalculateContactInputs (250 → 5)
    └── crashed.rs          — CalculateCrashedInputs (250 → 5)
```

### 2.3 Data Flow

```
Board (PositionKey, 10 bytes)
    │
    ▼
gnubg_types::board_from_old_key()
    │
    ▼
Board ([[u32; 25]; 2])
    │
    ▼
classify::ClassifyPosition(board) → Classification
    │
    ├── Race ──────► race::CalculateRaceInputs(board) → [f32; 92]
    ├── Contact ───► contact::CalculateContactInputs(board) → [f32; 250]
    └── Crashed ───► crashed::CalculateCrashedInputs(board) → [f32; 250]
    │
    ▼
neuralnet::feed_forward(&net, &inputs) → [f32; 5]
    │
    ▼
EvalOutput { win, win_gammon, win_backgammon, lose_gammon, lose_backgammon }
    │
    ▼
EvalResult { ..., equity = (2*win - 1) + win_gammon + win_backgammon - lose_gammon - lose_backgammon }
```

### 2.4 Integration Point

The change is **surgical**: exactly one function changes its implementation.

```rust
// gnubg-search/src/lib.rs — evaluate_key_with_thread_cache()

// BEFORE (SIG-293):
let raw = gnubg_sys::evaluate_position_key(&key)?;   // hash stub

// AFTER (SIG-291):
let raw = gnubg_eval::evaluate_position_key(&key)?;   // real NN eval
```

The rest of `gnubg-search` (cache, search, TT, parallel eval) and `gnubg-cli` (CLI commands, board rendering) are **untouched**. The `EvalResult::from_raw()` helper already converts `RawEval` (5-floats) to equity — the output shape is identical.

---

## 3. Key Decisions

### D1: AVX2 via `std::arch`, scalar fallback

**Choice:** Use `#[cfg(target_arch = "x86_64")]` with `std::arch::x86_64` intrinsics, plus a **runtime** CPUID check via `is_x86_feature_detected!("avx2")`.

**Why:** The Brief requires AVX2 but also fallback. The workspace already compiles with `target-cpu=x86-64-v3` (`.cargo/config.toml`), which implies SSE4.2/AVX but not necessarily AVX2. Runtime detection is required because the binary must run on CPUs without AVX2.

**Trade-off:** Runtime detection adds a branch on every `feed_forward()` call. We pay this cost once per eval (not per inner loop) — amortized across thousands of multiply-adds. The branch is a single function-pointer assignment at init time, not an if-check inside the hot loop.

**Rejected:** Compile-time-only AVX2 (`#[target_feature(enable = "avx2")]`) — would crash on CPUs without AVX2. Separate `avx2` feature flag — overcomplicates the Cargo.toml for a single function.

### D2: Weights embedded at compile time, not loaded at runtime

**Choice:** `include_bytes!("../gnubg-sys/vendor/gnubg.weights")` embedded into the binary. The weights file is 1.1 MB. The binary will be ~1.1 MB larger.

**Why:** The current code uses `include_bytes!` already (`gnubg-sys/src/lib.rs` line 16). The `gnubg.weights` file lives in the repo under `gnubg-sys/vendor/`. Loading at runtime would require a file path argument; the Brief says the evaluation should "just work" after `cargo build`.

**Trade-off:** Binary size increases from ~3 MB to ~4 MB (release, stripped). This is acceptable for a native CLI — no WASM or embedded target.

**Alternative considered:** Load from `gnubg-sys/vendor/gnubg.weights` at runtime via `include_str!` then parse. Same effect, different compile-time mechanism. We choose `include_bytes!` because we parse floats from bytes, not a string — the weights file has one float per line in ASCII.

**Correction:** The weights file is ASCII text (one float per line), not binary. We use `include_str!` to embed the text, then parse lines into `Vec<f32>` at init time.

### D3: Three network instances from one weights file

**Choice:** Parse the weights file once. Build three `NeuralNet` structs: `nnRace` (92×128×5), `nnContact` (250×128×5), `nnCrashed` (250×128×5). The weights file contains all parameters (input-to-hidden weights, hidden thresholds, hidden-to-output weights, output thresholds) for all three networks in a single flat array. The gnubg C code indexes into this array differently for each network.

**How gnubg.weights is structured:**
- Line 1: `GNU Backgammon 1.01` (magic)
- Line 2: `250 128 5 0 0.1000000 1.0000000` (nInput nHidden nOutput nTrained betaHidden betaOutput)
- Lines 3–32002: `arHiddenWeight[250×128 = 32000]` — first 92×128 = 11776 for nnRace, all 32000 for nnContact/nnCrashed
- Lines 32003–32130: `arHiddenThreshold[128]`
- Lines 32131–32770: `arOutputWeight[128×5 = 640]`
- Lines 32771–32775: `arOutputThreshold[5]`

**Important:** The Brief says nnRace has 92 inputs. In gnubg C, `nnRace.c_input` is set to `((25+2+1+6+72)*2) / 2 * 2 = 92` (the final `* 2` is for even alignment for SIMD). The first 92×128 = 11776 floats of the hidden weights correspond to nnRace input weights. nnContact and nnCrashed use all 250 inputs. The same hidden-threshold, output-weight, and output-threshold arrays are shared across all three networks.

**Validation:** After `nnRace.c_input = 92`, the `arHiddenWeight` slice for nnRace is `&weights.arHiddenWeight[0..92*128]`. For nnContact and nnCrashed, it's `&weights.arHiddenWeight[0..250*128]`. The hidden thresholds are always the same 128 floats. The output weights are always the same 640 floats. All three networks differ only in which sub-slice of `arHiddenWeight` they use and how many inputs they receive.

### D4: Input encoding from scratch (not delegating to C)

**Choice:** Pure Rust port of the input encoding functions from `eval.c`. All encoding functions (`baseInputs`, `CalculateHalfInputs`, `CalculateRaceInputs`, `CalculateContactInputs`, `CalculateCrashedInputs`, `ClassifyPosition`) must be reproduced.

**Why:** The Brief demands zero C/FFI. Input encoding is the most complex part — it encodes board state (25 points × 4 slots per side = 200 floats, plus ~25 aggregate features per side, plus hit-probability tables). Getting this right is the primary risk.

**Approach:**
1. Port `ClassifyPosition` first — it's ~60 lines of heuristic logic and gates which encoding function runs.
2. Port `baseInputs` — the core checker-position encoding (96 floats per side).
3. Port `CalculateHalfInputs` — the aggregate features (25 per side).
4. Assemble `CalculateContactInputs`, `CalculateRaceInputs`, `CalculateCrashedInputs` from baseInputs + HalfInputs + classification-specific features.

### D5: Sigmoid/tanh approximations for performance

**Choice:** Use the standard `f32::tanh()` from `std` for the hidden layer, and a rational approximation for sigmoid in the output layer when AVX2 is available. Scalar fallback uses `libm` or the standard library.

**Why:** The gnubg C code uses `tanh` for hidden layer and a custom sigmoid (`1.0 / (1.0 + exp(-x))`) for output. Rust's `f32::tanh()` is well-optimized. For AVX2 sigmoid, we use the `_mm256_fmadd_ps` for dot products and a polynomial approximation for sigmoid: `1.0 / (1.0 + exp(-x)) ≈ 0.5 + 0.5 * tanh(x/2)` — then use a minmax polynomial for tanh.

**Precision target:** ±0.001 per output channel (since the acceptance criteria allows ±0.05 tolerance overall).

---

## 4. Module Specifications

### 4.1 `weights.rs` — Parser

```rust
pub struct WeightFile {
    pub n_input: u32,          // 250
    pub n_hidden: u32,         // 128
    pub n_output: u32,         // 5
    pub n_trained: u32,        // 0 (placeholder)
    pub beta_hidden: f32,      // 0.1
    pub beta_output: f32,      // 1.0
    pub ar_hidden_weight: Vec<f32>,  // 250 × 128 = 32000
    pub ar_hidden_threshold: Vec<f32>, // 128
    pub ar_output_weight: Vec<f32>,   // 128 × 5 = 640
    pub ar_output_threshold: Vec<f32>, // 5
}

pub fn parse_weights(data: &str) -> Result<WeightFile, WeightError>;
```

**Error cases:**
- Missing magic header
- Wrong dimensions (must be 250 128 5)
- Float parsing failure
- Wrong number of floats (must be exactly 32000 + 128 + 640 + 5 = 32773 floats after header)

### 4.2 `neuralnet.rs` — NeuralNet + Forward Pass

```rust
pub struct NeuralNet {
    pub c_input: u32,
    pub c_hidden: u32,
    pub c_output: u32,
    pub ar_hidden_weight: Vec<f32>,
    pub ar_output_weight: Vec<f32>,
    pub ar_hidden_threshold: Vec<f32>,
    pub ar_output_threshold: Vec<f32>,
    pub beta_hidden: f32,
    pub beta_output: f32,
    // AVX2 dispatch
    forward_fn: fn(&NeuralNet, &[f32]) -> [f32; 5],
}

impl NeuralNet {
    pub fn new(weights: &WeightFile, c_input: u32) -> Self;

    /// Choose the right slice of ar_hidden_weight based on c_input.
    /// nnRace: c_input=92, uses ar_hidden_weight[0..92*128]
    /// nnContact: c_input=250, uses ar_hidden_weight[0..250*128]
    /// nnCrashed: c_input=250, same as nnContact

    pub fn feed_forward(&self, inputs: &[f32]) -> [f32; 5] {
        (self.forward_fn)(self, inputs)
    }
}
```

**Forward pass algorithm (scalar):**
```
for j in 0..c_hidden:
    sum = ar_hidden_threshold[j]
    for i in 0..c_input:
        sum += inputs[i] * ar_hidden_weight[i * c_hidden + j]
    hidden[j] = tanh(beta_hidden * sum)

for k in 0..c_output:
    sum = ar_output_threshold[k]
    for j in 0..c_hidden:
        sum += hidden[j] * ar_output_weight[j * c_output + k]
    output[k] = sigmoid(beta_output * sum)

return output  // [win, win_gammon, win_backgammon, lose_gammon, lose_backgammon]
```

**AVX2 forward pass (key optimization):**
- Hidden layer: process 8 output neurons in parallel. For each neuron `j`, accumulate `sum` from 8 packed `inputs[i] * weights[i][j]` using `_mm256_fmadd_ps`.
- Output layer: 5 outputs × 128 hidden = only 640 multiply-adds. AVX2 or scalar — both are fast here.
- Sigmoid approximation: `_mm256_sigmoid_ps` using minmax polynomial.

**Weight layout:** `ar_hidden_weight` is stored as `[input][hidden]` (row-major from the weights file). The inner loop over `j` (hidden neurons) iterates over columns, which is cache-friendly for the forward pass (all weights for one hidden neuron are contiguous).

### 4.3 `classify.rs` — Position Classification

```rust
pub enum Classification {
    Race,       // Both players have broken contact
    Contact,    // Pieces still interacting
    Crashed,    // One side has ≤ 6 checkers total
    Bearoff1,   // 1-checker bearoff (hypergammon-like)
    Bearoff2,   // 2-checker bearoff
    Bearoff3,   // 3-checker bearoff
    BearoffOS,  // One-sided bearoff database
    BearoffTS,  // Two-sided bearoff database
}

pub fn classify_position(board: &Board) -> Classification;
```

**Heuristic (ported from eval.c `ClassifyPosition`):**
1. Count total checkers for each player
2. If both players' back-checker sum > 22 → `Contact`
3. If either side has ≤ 6 checkers total → `Crashed`
4. Otherwise → `Race` (with bearoff sub-classes)

**Scope note:** Bearoff database classes (pbc1, pbc2, pbcOS, pbcTS) are **out of scope** for SIG-291. We handle them as `Race` with the neural net — no database lookup. The Brief explicitly excludes bearoff databases.

### 4.4 `inputs.rs` — Shared Input Primitives

```rust
/// Encode checker positions: for each point 1..24, encode up to 4 checkers.
/// Returns [f32; 96] per side (24 points × MINPPERPOINT(4) slots).
/// MINPPERPOINT = 4 means:
///   n≥1 → slot0=1.0
///   n≥2 → slot1=1.0
///   n≥3 → slot2=1.0
///   n≥4 → slot3=(n-3)/4.0  (linearly scaled)
pub fn base_inputs(board: &Board, side: usize) -> [f32; 96];

/// Calculate aggregate features for one side. Returns ~25 floats.
/// Includes: I_OFF1/2/3 (borne off), I_BREAK_CONTACT, I_BACK_CHEQUER, I_BACK_ANCHOR, hit probabilities.
pub fn calculate_half_inputs(board: &Board, side: usize) -> [f32; 25];
```

**baseInputs encoding detail:**
- `side = 0` → opponent's checkers (board[0])
- `side = 1` → player's checkers (board[1])
- For each point `p` in 1..24, let `n = board[side][p]`
- Index `(p-1) * 4 + 0` = 1.0 if n ≥ 1, else 0.0
- Index `(p-1) * 4 + 1` = 1.0 if n ≥ 2, else 0.0
- Index `(p-1) * 4 + 2` = 1.0 if n ≥ 3, else 0.0
- Index `(p-1) * 4 + 3` = if n ≥ 4, (n-3) as f32 / 4.0, else 0.0

**CalculateHalfInputs features (ported from eval.c):**
- `I_OFF[1..3]` — borne-off checkers (rescaled)
- `I_BREAK_CONTACT` — does this side have no checkers in opponent's home? (0.0 or 1.0)
- `I_BACK_CHEQUER` — most backward checker, normalized to [0,1]
- `I_BACK_ANCHOR` — farthest-back anchor (point with ≥2 checkers in opponent's home)
- `I_FORWARD_ANCHOR` — farthest-forward anchor
- `I_PIP_COUNT` — total pip count, normalized
- `aHit[39]` — hit probability intermediate features (from pre-computed lookup tables `aanCombination` and `aIntermediate`)

### 4.5 `race.rs`, `contact.rs`, `crashed.rs` — Per-Classification Encoders

```rust
// race.rs
pub fn calculate_race_inputs(board: &Board) -> [f32; 92];
// = base_inputs(side=0) ++ base_inputs(side=1) ++ [additional race-specific features]
// 92 = 96 + 96 - 100 (gnubg truncates baseInputs to match nnRace.c_input)

// contact.rs
pub fn calculate_contact_inputs(board: &Board) -> [f32; 250];
// = base_inputs(side=0) ++ base_inputs(side=1) ++ half_inputs(side=0) ++ half_inputs(side=1)
// 250 = 96 + 96 + 25 + 25 + 8 (contact-specific)

// crashed.rs
pub fn calculate_crashed_inputs(board: &Board) -> [f32; 250];
// Same layout as contact but with crashed-specific feature encoding
```

**Key detail:** The baseInputs for race is truncated differently from contact. In gnubg C, the first 92 inputs for nnRace are `baseInputs(opponent)[0..46] + baseInputs(player)[0..46]` (i.e., 46 per side, not the full 96). This is because nnRace.c_input = 92 = 46×2. The truncation drops the higher-indexed baseInputs which encode points with dense checker stacks (less relevant to pure races).

### 4.6 `lib.rs` — Public API

```rust
pub struct EvalOutput {
    pub win: f32,
    pub win_gammon: f32,
    pub win_backgammon: f32,
    pub lose_gammon: f32,
    pub lose_backgammon: f32,
}

// One-time init (load + parse + build networks)
pub fn init_weights() -> Result<(), EvalError>;

// Main entry point — equivalent to gnubg_sys::evaluate_position_key()
pub fn evaluate_position_key(key: &[u8; 10], board: &Board) -> Result<EvalOutput, EvalError>;

// Low-level
pub fn evaluate_board(board: &Board) -> Result<EvalOutput, EvalError>;
pub fn simd_supported() -> bool;
```

---

## 5. Integration Changes

### 5.1 `gnubg-search/src/lib.rs` changes

Replace exactly one function:

```rust
// OLD: uses gnubg_sys::evaluate_position_key
let raw = gnubg_sys::evaluate_position_key(&key)?;

// NEW: uses gnubg_eval::evaluate
let board = /* decode key → Board via gnubg_types */;
let eval_output = gnubg_eval::evaluate_position_key(&key, &board)?;
let raw = RawEval { outputs: [eval_output.win, eval_output.win_gammon, eval_output.win_backgammon, eval_output.lose_gammon, eval_output.lose_backgammon] };
```

**Important:** The `evaluate_key_with_thread_cache` function currently only has a `PositionKey`. To call `gnubg_eval::evaluate()`, we also need the `Board`. The caller (`evaluate_board`, `leaf_eval`, etc.) has the board. We change the signature to accept both, or decode the key inside the function using `gnubg_types::board_from_old_key()`.

**Decision:** Decode inside `evaluate_key_with_thread_cache`. The cost is one `board_from_old_key()` call per cache miss — trivial compared to the forward pass.

### 5.2 `gnubg-search/Cargo.toml` changes

```diff
 [dependencies]
-gnubg-sys = { path = "../gnubg-sys" }
+gnubg-eval = { path = "../gnubg-eval" }
 gnubg-types = { path = "../gnubg-types" }
 gnubg-moves = { path = "../gnubg-moves" }
 rayon = "1.10"
```

**Note:** `gnubg-sys` is still needed for `PositionKey` type. We keep the dependency but stop calling eval functions on it.

### 5.3 `Cargo.toml` (workspace root)

```diff
 [workspace]
-members = ["gnubg-sys", "gnubg-types", "gnubg-moves", "gnubg-search", "gnubg-cli"]
+members = ["gnubg-sys", "gnubg-types", "gnubg-moves", "gnubg-search", "gnubg-cli", "gnubg-eval"]
```

### 5.4 Files to Remove

- `gnubg-sys/vendor/gnubg_bridge.c` — hash stub, no longer compiled
- `gnubg-sys/build.rs` — no longer compiles C code
- `gnubg-sys/vendor/cache.c` — unused (C cache wrapper)
- `gnubg-sys/vendor/eval.c` — unused (C eval logic)
- `gnubg-sys/vendor/neuralnet.c` — unused (C NN logic)
- `gnubg-sys/vendor/neuralnetsse.c` — unused (C SSE NN logic)

### 5.5 Files to Keep (Modified)

- `gnubg-sys/src/lib.rs` — Keep `PositionKey`, `RawEval` types, `decode_position_id`. Remove `evaluate_position_key`, `neuralnet_evaluate`, `simd_supported`, `embedded_weights_len`, FFI declarations.
- `gnubg-sys/Cargo.toml` — Remove `links`, `build`, `cc` dependency. Keep the crate for types.

---

## 6. Acceptance Criteria Mapping

| Criteria | How Verified |
|---|---|
| `cargo test` passes in all workspace crates | `cargo test --workspace` green |
| `cargo clippy` passes with no warnings in new code | `cargo clippy -p gnubg-eval -- -D warnings` |
| CLI `evaluate 4HPwATDgc/ABMA` outputs plausible probs | win > 0.5, all in [0,1] |
| Forward pass matches gnubg C within ±0.05 | Manual cross-validation script |
| `bench --positions 1000` runs without errors | Output shows throughput |
| `best-move 4HPwATDgc/ABMA 31` returns sensible move | Move is legal, equity within reason |
| No C compilation step in build | `cargo clean && cargo build 2>&1 | grep -c cc` == 0 |

---

## 7. Risks & Mitigations

### R1: Input encoding bugs (HIGH)
**Risk:** `CalculateHalfInputs` has 39 intermediate features with complex array indexing from the hit-probability tables `aanCombination` and `aIntermediate`. An off-by-one error anywhere produces wrong evaluations that are hard to detect (outputs still in [0,1], just wrong).

**Mitigation:**
- Cross-validate 50 random positions against gnubg C output (accept ±0.05 tolerance)
- The hit-probability tables are **literal lookup tables** — copy the float arrays verbatim from eval.c
- Use `assert_eq!` on array dimensions to catch shape mismatches at compile/init time

### R2: Weight file parsing mismatch (MEDIUM)
**Risk:** The weights are in ASCII, not binary. Parsing 32,773 float strings is slower than a binary load but only happens once at startup. More critical: float parsing precision differences between C (`atof`) and Rust (`f32::from_str`) may cause tiny weight value differences that accumulate across 128 hidden neurons.

**Mitigation:**
- Test that `beta_hidden * weight_sum` differences are below 1e-5 for any input vector
- The ±0.05 output tolerance accounts for this

### R3: AVX2 not available (LOW)
**Risk:** Deployed CPU might not support AVX2.

**Mitigation:** Runtime dispatch via `is_x86_feature_detected!("avx2")` at `init_weights()` time. Scalar fallback is always available. The binary still works; just 2-4× slower for the forward pass.

### R4: gnubg-sys removal breaks existing code (LOW)
**Risk:** Removing eval functions from `gnubg-sys` might break code that imports them.

**Mitigation:** `gnubg-sys` pub API is only used by `gnubg-search` and `gnubg-cli`. Audit all imports before removing. The `PositionKey` type stays.

---

## 8. Out of Scope

- Match equity tables / cube decisions (money game only)
- Bearoff databases (pbc1, pbc2, pbcOS, pbcTS)
- Rollout / variance reduction
- Full gnubg parity beyond neural net output
- WebAssembly target
- GUI rendering
- NNUE-style incremental updates (re-evaluate from scratch each time)

---

## 9. Performance Targets

| Metric | Target | Measurement |
|---|---|---|
| Eval throughput (scalar) | > 10,000 evals/sec | `bench --positions 10000` on i7-class CPU |
| Eval throughput (AVX2) | > 40,000 evals/sec | Same benchmark, AVX2 CPU |
| Init time (parse weights) | < 500 ms | Startup time before first eval |
| Memory (per thread cache) | < 16 MB | RSS after 10K evals |
| Output precision vs gnubg C | ±0.05 per channel | Cross-validation on 50 positions |

---

## 10. Testing Strategy

1. **Unit tests per module:**
   - `weights.rs`: parse header, parse float count, error on malformed
   - `neuralnet.rs`: scalar forward pass on known input → known output (hardcoded small network)
   - `classify.rs`: opening position → Contact, endgame → Race/Crashed
   - `inputs.rs`: baseInputs on empty board → all zeros; on full point → correct encoding

2. **Integration tests:**
   - `cargo test -p gnubg-eval` with the real weights file embedded
   - Opening position (4HPwATDgc/ABMA): classify → Contact, win > 0.50
   - Crashed position: classify → Crashed, outputs in [0,1]

3. **Cross-validation:**
   - Script that encodes 50 random positions, evaluates with both C stub (via hash) and Rust NN, validates output differences
   - Current hash stub can be used as a sanity check (different path, same output range)

4. **Existing tests must pass:**
   - `gnubg-search`: cache_hits_on_second_eval, root_eval_returns_all_candidates, best_move_selects_max_equity, alpha_beta_depth_one
   - These tests exercise the FULL eval path — they will use the new evaluator

---

## 11. Diagram: Evaluation Pipeline

```mermaid
flowchart TD
    A[PositionID string<br/>eg: 4HPwATDgc/ABMA] --> B[gnubg_sys::decode_position_id]
    B --> C[PositionKey<br/>10 bytes]
    C --> D[gnubg_types::board_from_old_key]
    D --> E[Board<br/>[[u32; 25]; 2]]
    E --> F[gnubg_eval::classify_position]
    F --> G{Classification?}
    G -- Race --> H[race::calculate_race_inputs]
    G -- Contact --> I[contact::calculate_contact_inputs]
    G -- Crashed --> J[crashed::calculate_crashed_inputs]
    H --> K[Input vector<br/>[f32; 92]]
    I --> L[Input vector<br/>[f32; 250]]
    J --> M[Input vector<br/>[f32; 250]]
    K --> N[neuralnet::feed_forward<br/>nnRace]
    L --> O[neuralnet::feed_forward<br/>nnContact]
    M --> P[neuralnet::feed_forward<br/>nnCrashed]
    N --> Q[Output [f32; 5]<br/>win, win_gammon, win_backgammon, lose_gammon, lose_backgammon]
    O --> Q
    P --> Q
    Q --> R[EvalResult::from_raw → equity]
    R --> S[Alpha-beta search]
```
