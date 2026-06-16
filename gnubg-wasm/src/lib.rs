use wasm_bindgen::prelude::*;
use gnubg_eval::cubeful::{CubeOwner, CubeState};
use gnubg_eval::met::MatchState;
use gnubg_search::{Board, CubeAction, SearchConfig, SearchError};
use serde::Serialize;

/// Initialize the neural-network weights. Must be called once before any other
/// engine function. Returns `true` on success.
#[wasm_bindgen]
pub fn init_engine() -> bool {
    gnubg_eval::init_weights().is_ok()
}

/// Return a short version string for the engine.
#[wasm_bindgen]
pub fn engine_info() -> String {
    format!("backgammon-rust v{}", env!("CARGO_PKG_VERSION"))
}

/// Evaluate a position and return JSON with cubeless and cubeful equity.
///
/// `position_id` — a standard backgammon position ID (e.g. "4HPwATDgc/ABMA").
/// `match_score` — optional, colon-separated match score e.g. "3:5" (player:opponent away).
/// `cube_value` — optional cube value (defaults to 1).
#[wasm_bindgen]
pub fn evaluate_position(position_id: &str, match_score: Option<String>, cube_value: Option<i32>) -> JsValue {
    let result = evaluate_position_inner(position_id, match_score, cube_value);
    match result {
        Ok(val) => serde_wasm_bindgen::to_value(&val).unwrap_or(JsValue::UNDEFINED),
        Err(e) => {
            let err = serde_json::json!({ "error": e.to_string() });
            serde_wasm_bindgen::to_value(&err).unwrap_or(JsValue::UNDEFINED)
        }
    }
}

/// Find the best move for a given position and dice roll.
///
/// `position_id` — standard backgammon position ID.
/// `dice` — two digits e.g. "31" or "64" (order doesn't matter).
/// `depth` — search depth (0 = static evaluation only).
#[wasm_bindgen]
pub fn best_move(position_id: &str, dice: &str, depth: u8) -> JsValue {
    let result = best_move_inner(position_id, dice, depth);
    match result {
        Ok(val) => serde_wasm_bindgen::to_value(&val).unwrap_or(JsValue::UNDEFINED),
        Err(e) => {
            let err = serde_json::json!({ "error": e.to_string() });
            serde_wasm_bindgen::to_value(&err).unwrap_or(JsValue::UNDEFINED)
        }
    }
}

/// Analyze all 21 possible dice rolls for a position.
///
/// `position_id` — standard backgammon position ID.
/// `depth` — search depth (0 = static).
#[wasm_bindgen]
pub fn analyze_position(position_id: &str, depth: u8) -> JsValue {
    let result = analyze_position_inner(position_id, depth);
    match result {
        Ok(val) => serde_wasm_bindgen::to_value(&val).unwrap_or(JsValue::UNDEFINED),
        Err(e) => {
            let err = serde_json::json!({ "error": e.to_string() });
            serde_wasm_bindgen::to_value(&err).unwrap_or(JsValue::UNDEFINED)
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct EvalOutput {
    win: f32,
    win_gammon: f32,
    win_backgammon: f32,
    lose_gammon: f32,
    lose_backgammon: f32,
    equity: f32,
    cubeful_equity: f32,
    cube_decision: Option<CubeDecisionJson>,
}

#[derive(Serialize)]
struct CubeDecisionJson {
    action: String,         // "NoDouble" | "DoubleTake" | "DoubleDrop"
    no_double_equity: f32,
    double_take_equity: f32,
    double_drop_equity: f32,
    gain: f32,
}

#[derive(Serialize)]
struct BestMoveOutput {
    move_id: usize,
    from: u8,
    to: u8,
    notation: String,
    equity: f32,
    cubeful_equity: f32,
    win: f32,
    win_gammon: f32,
    win_backgammon: f32,
    lose_gammon: f32,
    lose_backgammon: f32,
}

#[derive(Serialize)]
struct AnalyzeRollOutput {
    dice: String,
    best_move_id: usize,
    equity: f32,
}

#[derive(Serialize)]
struct AnalyzeOutput {
    rolls: Vec<AnalyzeRollOutput>,
    num_rolls: usize,
}

fn evaluate_position_inner(
    position_id: &str,
    match_score: Option<String>,
    cube_value: Option<i32>,
) -> Result<EvalOutput, SearchError> {
    let board = Board::from_position_id(position_id)?;
    let eval = gnubg_search::evaluate_board(&board, 0)?;

    let cube_state = make_cube_state(match_score, cube_value);

    let outputs = [eval.win, eval.win_gammon, eval.win_backgammon,
                   eval.lose_gammon, eval.lose_backgammon];
    let cubeful_equity = gnubg_eval::cubeful::cubeful_equity(&outputs, &cube_state);

    let cube_decision = if cube_state.owner == CubeOwner::Center {
        match gnubg_search::compute_cube_decision(&board, &cube_state) {
            Ok(decision) => Some(CubeDecisionJson {
                action: match decision.action {
                    CubeAction::NoDouble => "NoDouble".into(),
                    CubeAction::DoubleTake => "DoubleTake".into(),
                    CubeAction::DoubleDrop => "DoubleDrop".into(),
                },
                no_double_equity: decision.no_double_equity,
                double_take_equity: decision.double_take_equity,
                double_drop_equity: decision.double_drop_equity,
                gain: decision.gain,
            }),
            Err(_) => None,
        }
    } else {
        None
    };

    Ok(EvalOutput {
        win: eval.win,
        win_gammon: eval.win_gammon,
        win_backgammon: eval.win_backgammon,
        lose_gammon: eval.lose_gammon,
        lose_backgammon: eval.lose_backgammon,
        equity: eval.equity,
        cubeful_equity,
        cube_decision,
    })
}

fn best_move_inner(position_id: &str, dice: &str, depth: u8) -> Result<BestMoveOutput, SearchError> {
    let board = Board::from_position_id(position_id)?;
    let parsed_dice = parse_dice(dice)?;
    let (mv, eval) = gnubg_search::best_move(&board, parsed_dice, depth)?;

    Ok(BestMoveOutput {
        move_id: mv.id,
        from: mv.from,
        to: mv.to,
        notation: mv.to_string(),
        equity: eval.equity,
        cubeful_equity: eval.cubeful_equity,
        win: eval.win,
        win_gammon: eval.win_gammon,
        win_backgammon: eval.win_backgammon,
        lose_gammon: eval.lose_gammon,
        lose_backgammon: eval.lose_backgammon,
    })
}

fn analyze_position_inner(position_id: &str, depth: u8) -> Result<AnalyzeOutput, SearchError> {
    let board = Board::from_position_id(position_id)?;
    let config = SearchConfig {
        max_depth: depth,
        ..SearchConfig::default()
    };
    let result = gnubg_search::analyze_position(&board, &config)?;

    let rolls: Vec<AnalyzeRollOutput> = result
        .rolls
        .iter()
        .map(|roll| AnalyzeRollOutput {
            dice: format!("{}{}", roll.dice.0, roll.dice.1),
            best_move_id: roll.best_move.id,
            equity: roll.equity,
        })
        .collect();

    let num_rolls = rolls.len();
    Ok(AnalyzeOutput { rolls, num_rolls })
}

fn parse_dice(s: &str) -> Result<(u8, u8), SearchError> {
    let chars: Vec<char> = s.trim().chars().collect();
    if chars.len() != 2 {
        return Err(SearchError::EmptyMoveList); // hack: reuse as invalid input
    }
    let a = chars[0].to_digit(10).unwrap_or(0) as u8;
    let b = chars[1].to_digit(10).unwrap_or(0) as u8;
    if !(1..=6).contains(&a) || !(1..=6).contains(&b) {
        return Err(SearchError::EmptyMoveList);
    }
    Ok((a, b))
}

fn make_cube_state(match_score: Option<String>, cube_value: Option<i32>) -> CubeState {
    let match_state = match_score.and_then(|s| parse_match_score(&s));
    CubeState {
        value: cube_value.unwrap_or(1),
        owner: CubeOwner::Center,
        efficiency: 1.0,
        match_state,
    }
}

fn parse_match_score(s: &str) -> Option<MatchState> {
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() != 2 {
        return None;
    }
    let player_away = parts[0].parse::<i32>().ok()?;
    let opponent_away = parts[1].parse::<i32>().ok()?;
    if player_away <= 0 || opponent_away <= 0 {
        return None;
    }
    Some(MatchState::new(player_away, opponent_away, false, false))
}
