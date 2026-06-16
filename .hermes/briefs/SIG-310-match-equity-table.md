# Brief: SIG-310 — Match Equity Table (Kazaross XG MET)

## Context

The engine computed cubeful equity via the Janowski formula (SIG-296) but only for **money game** — no awareness of match score, Crawford rule, or post-Crawford play. Without a Match Equity Table (MET), the engine cannot:

- Evaluate doubling decisions in match play (take points shift with score)
- Convert points won/lost into match-winning chances
- Support real match scenarios

The Rockwell-Kazaross XG MET is the standard in modern backgammon (used by XG, gnubg). It was rolled out 38,880 trials per score using GNU 2-ply Supremo and extrapolated to 25-away. The full table (27×27 including Crawford/post-Crawford columns) is published at [bkgm.com](https://bkgm.com/articles/Kazaross/RockwellKazarossMET/).

## Objective

Embed the Kazaross XG MET into `gnubg-eval`, provide a clean API for match-winning chances and match-equity conversion, and wire it into the CLI for match-aware evaluation.

## Scope

### 1. Data: `gnubg-eval/src/met.rs` (NEW)

Embed the full 27×27 table as a static `[[f32; 27]; 27]` (PC + away 1..25). The table is symmetric: row = player away, col = opponent away.

```rust
/// Rockwell-Kazaross Match Equity Table.
/// Index 0 = Post-Crawford (PC), indices 1..25 = points away.
/// `MET[a][b]` = match winning chance (0..1) when player is `a`-away, opponent is `b`-away.
const KAZAROSS_XG_MET: [[f32; 27]; 27] = { ... };
```

Full data to embed (from the unabridged table):

| idx | Label |
|-----|-------|
| 0 | PC (Post-Crawford) |
| 1 | 1-away (Crawford game) |
| 2..25 | 2-away .. 25-away |

### 2. API: `gnubg-eval/src/met.rs`

```rust
/// Represents a match score from the player-to-move's perspective.
pub struct MatchState {
    pub player_away: i32,    // points player needs to win (1..=25)
    pub opponent_away: i32,  // points opponent needs (1..=25)
    pub crawford: bool,      // is this the Crawford game (leader at 1-away, cube frozen)
    pub post_crawford: bool, // post-Crawford series (leader already won Crawford game)
}

impl MatchState {
    /// Create a new match state. Validates constraints:
    /// - player_away and opponent_away must be 1..=25
    /// - crawford and post_crawford are mutually exclusive
    /// - crawford only valid when one player is at 1-away
    /// - post_crawford only valid when one player is at 1-away
    pub fn new(player_away: i32, opponent_away: i32, crawford: bool, post_crawford: bool) -> Self;

    /// Is this a money game? (No match context — caller should use money equity)
    pub fn is_money() -> bool;
}
```

Core queries:

```rust
/// Look up match winning chance (MWC) from the Kazaross XG MET.
/// Returns probability (0..1) of the player-to-move winning the match.
pub fn mwc(state: &MatchState) -> f32;

/// Convert a cubeless equity difference (points) into a change in MWC.
/// `cubeless_points` = points won/lost by the player (positive = player wins points)
/// Returns the new MWC after the game outcome.
pub fn mwc_after(state: &MatchState, cubeless_points: i32) -> f32;

/// Convert points equity to MWC equity swing.
/// `point_equity` = cubeless equity in points (e.g., +0.5 means expected +0.5 points)
/// Returns the expected change in match winning chances.
pub fn match_equity_swing(state: &MatchState, point_equity: f32) -> f32;
```

### 3. Integration with Cubeful Module

Update `gnubg-eval/src/cubeful.rs` to accept optional match context:

```rust
pub struct CubeState {
    pub value: i32,
    pub owner: CubeOwner,
    pub efficiency: f32,
    pub match_state: Option<MatchState>,  // NEW
}
```

Update `cubeful_equity()`:

```
fn cubeful_equity(outputs: &[f32; 5], cube: &CubeState) -> f32 {
    let point_equity = /* existing Janowski calculation (already returns points) */;

    match &cube.match_state {
        None => point_equity,  // money game — existing behavior
        Some(match_state) => {
            // Convert points equity to MWC-equivalent
            // For small equities: delta_MWC ≈ point_equity * (MWC(score ±1) - MWC(score)) / 1.0
            // This linearization is standard gnubg practice for 0-ply cubeful
            match_equity_swing(match_state, point_equity)
        }
    }
}
```

### 4. CLI Integration

The `evaluate` subcommand gains match-play flags:

```
gnubg-cli evaluate 4HPwATDgc/ABMA --match 5:3
                                # player is 5-away, opponent is 3-away, no Crawford
gnubg-cli evaluate 4HPwATDgc/ABMA --match 1:5 --crawford
                                # Crawford game (leader at 1-away, cube frozen)
gnubg-cli evaluate 4HPwATDgc/ABMA --match 1:2 --post-crawford
                                # Post-Crawford (Crawford game already played)
```

Output with match context:

```
position_id: 4HPwATDgc/ABMA
match: 5-away / 3-away
win: 100.00%
...
equity: +1.000000 points
mwc: 68.71%  (match winning chance)
swing: +2.34% (equity in MWC)
```

When `--match` is omitted, current money-game behavior is unchanged.

### 5. Tests

**MET lookup tests:**
- Diagonal (tied score) → 0.5
- Trailing by 1 at Crawford: `mwc(MatchState { 2, 1, crawford: true, .. })` → 0.323112
- Leading by 1 at Crawford: `mwc(MatchState { 1, 2, crawford: true, .. })` → 0.676888
- Post-Crawford: `mwc(MatchState { 1, 2, crawford: false, post_crawford: true })` → 0.512323
- Symmetry: `mwc(a, b) + mwc(b, a) = 1.0` for non-Crawford scores
- Out of range → panic or clamped

**Integration tests:**
- Pass `match_state: None` → cubeful equity unchanged from money game
- Pass `match_state: Some(...)` → cubeful equity is in MWC (0..1 range)
- CLI without `--match` → money game output (backward compatible)
- CLI with `--match` → match detail lines appear

## Data: Full Kazaross XG MET

The table below is embedded directly as a `const [[f32; 27]; 27]`:

```
Index: 0=PC, 1..25=away
Row = player away, Col = opponent away

Values (27 rows × 27 cols):
```

(Full table from the reference — all decimal values listed in the unabridged table at bkgm.com — embedded verbatim.)

## Out of Scope

- Modifying the search algorithm for match-aware play (the search still evaluates positions, just with a match-aware equity)
- Monte Carlo rollouts for MET validation
- GUI match setup
- Match equity table for lengths > 25 (extremely rare in practice)
- Generating the MET from scratch (use published Kazaross XG values)

## Technical Notes

### How the MET is Used

In gnubg, match play evaluation works as follows:

1. **MWC lookup**: Given current score, look up the match-winning chance.
2. **After-game MWC**: For each possible game outcome (win single, win gammon, win bg, lose single, lose gammon, lose bg), compute the new score and look up the new MWC.
3. **Equity conversion**: The "match equity" of a position is the expected MWC after the game, weighted by the NN probabilities.

At 0-ply, this is approximated as:
```
match_equity = current_mwc + point_equity * (mwc_after_win - mwc_after_loss) / 2.0
```
This linearization is standard for 0-ply evaluation.

### Crawford Rule

- The **Crawford game**: When the leader reaches 1-away, the next game is played without the doubling cube (Crawford rule).
- **Post-Crawford**: After the Crawford game, the cube is available again. The trailer's MWC in post-Crawford is tracked separately (PC column in MET).
- The PC column values depend on both players' away scores (left column is player-away, top row is opponent-away).

### Data Size

The 27×27 table = 729 floats = ~2.9KB of data. Trivial footprint.

### Thread Safety

The table is `const` — no mutation, no locking needed.

## Acceptance Criteria

1. All diagonal positions return exactly 0.5
2. Symmetry holds: `mwc(a, b) + mwc(b, a) = 1.0 ± 1e-6` for all non-Crawford scores
3. Known reference values match published MET:
   - -2,-1 Crawford: 32.31%
   - -1,-2 Crawford: 67.69%
   - -1,-1 (tied): 50.0%
   - PC 1-away vs 2-away: 51.23%
4. `cubeful_equity()` with `match_state: None` returns exactly the same as before (backward compat)
5. CLI `evaluate` without `--match` produces identical output to current behavior
6. All existing 84 tests pass
7. `cargo build --release` passes with 0 warnings
8. `cargo clippy --workspace --all-targets` passes (or only pre-existing findings)

## Files to Create/Modify

| File | Action |
|------|--------|
| `gnubg-eval/src/met.rs` | **Create** — const MET table + MatchState + mwc() + mwc_after() + match_equity_swing() |
| `gnubg-eval/src/cubeful.rs` | Modify — add `match_state: Option<MatchState>` to CubeState, update cubeful_equity() |
| `gnubg-eval/src/lib.rs` | Modify — add `pub mod met;` |
| `gnubg-search/src/lib.rs` | Modify — thread MatchState through evaluate path |
| `gnubg-cli/src/main.rs` | Modify — add `--match`, `--crawford`, `--post-crawford` flags |

## Branch

`feature/SIG-310-match-equity-table` from `main`

## Technical Note: Why SIG-310?

SIG-296 was the last active Backgammon issue. After the linear cleanup, the next available number is SIG-310 (SIG-297 was the last merged feature, and 298-309 were duplicates/canceled).
