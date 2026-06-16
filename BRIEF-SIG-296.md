# Brief: SIG-296 — Cubeful Equity Calculation

## Context

The backgammon engine currently computes a **cubeless money equity** inline in `gnubg-search/src/lib.rs`:

```rust
let equity = (2.0 * win - 1.0) + win_gammon + win_backgammon - lose_gammon - lose_backgammon;
```

This formula is correct for money game cubeless equity (gammon=2, backgammon=3), but it lives in the wrong layer (search crate, not eval crate) and ignores the **doubling cube** entirely. The engine has no concept of cube state, cube ownership, or cube efficiency — making it unsuitable for real-money or match play.

The previous issues (SIG-291 NN engine, SIG-295 SanityCheck, SIG-297 cleanup) are all merged on `main` at `c7f8dfa`. All 31 tests pass.

## Objective

Implement proper cubeful equity calculation in `gnubg-eval` using the **Janowski formula**, expose a clean API, and wire it through to the CLI.

## Scope

### 1. New module: `gnubg-eval/src/cubeful.rs`

- `fn cubeless_equity(outputs: &[f32; 5]) -> f32` — extract the current formula:
  ```rust
  (2.0 * outputs[0] - 1.0)
      + outputs[1] + outputs[2]
      - outputs[3] - outputs[4]
  ```

- `struct CubeState`:
  ```rust
  pub struct CubeState {
      pub value: i32,             // 1, 2, 4, 8, ... (1 = cube not turned yet)
      pub owner: CubeOwner,
      pub efficiency: f32,        // Janowski r parameter (0.0 = live cube, 1.0 = dead cube)
  }
  ```
  - `enum CubeOwner { Player, Opponent, Center }`
  - `impl Default for CubeState` — `{ value: 1, owner: Center, efficiency: 1.0 }` (dead cube = cubeless)

- `fn dead_cube_equity(outputs: &[f32; 5]) -> f32` — same as cubeless (cube never turns)

- `fn live_cube_equity(outputs: &[f32; 5]) -> f32` — Janowski live-cube formula:
  - Win probability `p = (cubeless + 1.0) / 2.0`
  - Live cube equity with `k` free drops: `(p^k - (1-p)^k) / (p^k + (1-p)^k)`
  - For money game (infinite free drops, k=1): `live = 2*p - 1 = cubeless`
  - For match play (finite free drops, k > 1): live cube equity can differ from cubeless
  - **For MVP**: implement k=1 (infinite drops, money game), which equals cubeless for live cube

- `fn cubeful_equity(outputs: &[f32; 5], cube: &CubeState) -> f32` — Janowski interpolation:
  ```rust
  // Janowski's formula:
  let cubeless = cubeless_equity(outputs);
  let dead = cubeless;  // dead cube = no doubling possible
  let live = live_cube_equity(outputs);
  // Interpolate: cubeful = live - r * (live - dead)
  // where r = cube.efficiency
  let cubeful = live - cube.efficiency * (live - dead);
  // Apply cube ownership sign:
  match cube.owner {
      CubeOwner::Player => cubeful * cube.value as f32,
      CubeOwner::Opponent => -(-cubeful * cube.value as f32), // mirrored since outputs are from player's perspective
      CubeOwner::Center => cubeful, // cube not yet owned
  }
  ```

  > **Note about sign convention**: The outputs `[win, win_gammon, win_backgammon, lose_gammon, lose_backgammon]` are from the player-to-move's perspective. When cube owner is Opponent, the equity sign needs careful handling — the player-to-move's perspective is maintained, but the cube belongs to the opponent so the player cannot double. Keep the formula simple for now: cubeful equity is always from the player-to-move's perspective. The `owner` field mainly controls whether the player has the option to double.

  > **Note about `r`**: The default `r = 0.68` is typical for money game backgammon positions (empirically calibrated in gnubg). Higher r = less efficient cube (closer to dead), lower r = more efficient cube (closer to live).

- **Tests**:
  - Cubeless equity matches current inline result
  - Center cube with r=1.0 equals cubeless equity
  - Win = 1.0 produces equity = 3.0 (bg win, max)
  - Lose = 1.0 (win=0) produces equity = -3.0 (bg loss, min)
  - Symmetric position (win=0.5, gammons equal) produces equity ≈ 0

### 2. Update `gnubg-eval/src/lib.rs`

- Add `pub mod cubeful;`
- Add method to `EvalOutput`:
  ```rust
  impl EvalOutput {
      pub fn cubeless_equity(&self) -> f32 {
          cubeful::cubeless_equity(&self.outputs())
      }
      pub fn cubeful_equity(&self, cube: &cubeful::CubeState) -> f32 {
          cubeful::cubeful_equity(&self.outputs(), cube)
      }
  }
  ```

### 3. Update `gnubg-search/src/lib.rs`

- In `EvalResult::from_raw()`, replace inline equity with `gnubg_eval::cubeful::cubeless_equity(&raw.outputs)`.
- Add `cubeful_equity: f32` field to `EvalResult`.
- `from_raw()` becomes `from_raw(raw, depth, cache_hit, cube_state)` or add a builder.
  - **Simplify**: keep `from_raw()` simple (cubeless only) and add a separate method `with_cubeful(cube_state)`.

### 4. Update `gnubg-cli/src/main.rs`

- `print_eval()` shows both:
  ```
  win: 50.00%
  win_gammon: 12.00%
  win_backgammon: 3.00%
  lose_gammon: 10.00%
  lose_backgammon: 2.00%
  equity: +0.234567
  cubeful: +0.210000
  ```

- `evaluate` subcommand gains optional `--cube <VALUE>` flag (default: no cube display).
- If `--cube` is passed, compute cubeful equity with CubeState { value, owner: Center, efficiency: 0.68 } and display it.

## Out of Scope

- Match equity table (match play equity with Crawford rule) — this is a separate issue.
- JGammon or other equity models — Janowski only.
- GUI cube interaction — CLI flags only.
- Recursive cubeful evaluation (search with cube actions) — 0-ply cubeful only.
- Changing the search algorithm to consider cube actions.

## Technical Notes

### The Janowski Formula

The Janowski cubeful equity model (1990s) is the standard in modern backgammon software (gnubg, eXtreme Gammon). It interpolates between two extremes:

- **Dead cube** (r=1.0): the cube never turns, so equity = cubeless money equity.
- **Live cube** (r=0.0): perfect doubling strategy, no cube inefficiency.

For money game with infinite free drops, live-cube equity equals cubeless equity because `(p^1 - (1-p)^1) / (p^1 + (1-p)^1) = 2p-1 = cubeless`. So for default money game, cubeful ≈ cubeless when r=0.68.

The real value of cubeful equity appears when:
- The cube is owned (owner advantage: they can double later)
- The cube is centered with moderately efficient play
- Match play (finite free drops per match length)

For the MVP, implementing the formula correctly with configurable `r` and `CubeOwner` is the priority. The effect will be subtle for money game at 0-ply, but the API and infrastructure must be correct so that match play and search-based cubeful can build on it.

### Sign Convention

All equities are from the **player-to-move's perspective**:
- Positive = player is ahead
- Negative = player is behind
- The `owner` field in CubeState describes cube ownership relative to the player

### Default Values

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| Cube value | 1 | No doubling yet |
| Owner | Center | Neither player owns the cube |
| Efficiency (r) | 0.68 | Gnubg's default for money game |
| Gammon value | 2 | Standard money game |
| Backgammon value | 3 | Standard money game |

## Acceptance Criteria

1. `cubeless_equity()` produces the same result as the current inline formula for 100+ random positions
2. `cubeful_equity()` with `efficiency=1.0` equals `cubeless_equity()` (dead cube)
3. A position with win=1.0, win_bg=1.0 produces equity=+3.0
4. A position with win=0.0, lose_bg=1.0 produces equity=-3.0
5. Symmetric position (win=0.5, equal gammons) produces equity ≈ 0
6. All existing 31 tests pass (0 failures, 0 warnings)
7. `cargo build --release` passes with 0 warnings
8. CLI `evaluate` for opening position shows sensible values

## Files to Create/Modify

| File | Action |
|------|--------|
| `gnubg-eval/src/cubeful.rs` | **Create** — new module with Janowski formulas |
| `gnubg-eval/src/lib.rs` | Modify — add `mod cubeful;` and methods on EvalOutput |
| `gnubg-search/src/lib.rs` | Modify — use gnubg-eval equity, add cubeful field |
| `gnubg-cli/src/main.rs` | Modify — display cubeful, optional --cube flag |

## Branch

`feature/SIG-296-cubeful-equity` from `main`
