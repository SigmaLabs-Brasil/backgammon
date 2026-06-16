use gnubg_eval::cubeful::{cubeful_equity, CubeOwner, CubeState};
use gnubg_eval::met::{match_equity_swing, mwc, mwc_after, MatchState};

fn approx_eq(actual: f32, expected: f32, tolerance: f32) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual} expected={expected} tolerance={tolerance}"
    );
}

#[test]
fn met_reference_values_match_rockwell_kazaross() {
    approx_eq(mwc(&MatchState::new(2, 1, true, false)), 0.323112, 1.0e-6);
    approx_eq(mwc(&MatchState::new(1, 2, true, false)), 0.676888, 1.0e-6);
    approx_eq(mwc(&MatchState::new(1, 1, true, false)), 0.5, 1.0e-6);
    approx_eq(mwc(&MatchState::new(1, 2, false, true)), 0.512323, 1.0e-6);
}

#[test]
fn met_diagonal_and_symmetry_hold_for_non_crawford_scores() {
    for away in 1..=25 {
        approx_eq(mwc(&MatchState::new(away, away, false, false)), 0.5, 1.0e-6);
    }

    for player_away in 2..=25 {
        for opponent_away in 2..=25 {
            let a = mwc(&MatchState::new(player_away, opponent_away, false, false));
            let b = mwc(&MatchState::new(opponent_away, player_away, false, false));
            approx_eq(a + b, 1.0, 1.0e-6);
        }
    }
}

#[test]
#[should_panic(expected = "player_away must be in 1..=25")]
fn match_state_rejects_out_of_range_scores() {
    MatchState::new(26, 2, false, false);
}

#[test]
#[should_panic(expected = "crawford and post_crawford are mutually exclusive")]
fn match_state_rejects_conflicting_crawford_flags() {
    MatchState::new(1, 2, true, true);
}

#[test]
fn mwc_after_applies_game_points_and_terminal_results() {
    let state = MatchState::new(3, 3, false, false);
    approx_eq(
        mwc_after(&state, 1),
        mwc(&MatchState::new(2, 3, false, false)),
        1.0e-6,
    );
    approx_eq(
        mwc_after(&state, -1),
        mwc(&MatchState::new(3, 2, false, false)),
        1.0e-6,
    );
    approx_eq(mwc_after(&state, 3), 1.0, 1.0e-6);
    approx_eq(mwc_after(&state, -3), 0.0, 1.0e-6);
}

#[test]
fn match_equity_swing_is_linearized_around_current_mwc() {
    let state = MatchState::new(5, 3, false, false);
    let expected = 0.5 * (mwc_after(&state, 1) - mwc_after(&state, -1));
    approx_eq(match_equity_swing(&state, 1.0), expected, 1.0e-6);
    approx_eq(match_equity_swing(&state, 0.0), 0.0, 1.0e-6);
}

#[test]
fn cubeful_money_game_behavior_is_unchanged_without_match_state() {
    let outputs = [0.55, 0.12, 0.03, 0.10, 0.02];
    let cube = CubeState {
        value: 2,
        owner: CubeOwner::Player,
        efficiency: 1.0,
        match_state: None,
    };

    approx_eq(cubeful_equity(&outputs, &cube), 0.26, 1.0e-6);
}

#[test]
fn cubeful_match_context_returns_match_winning_chance() {
    let outputs = [0.55, 0.12, 0.03, 0.10, 0.02];
    let state = MatchState::new(5, 3, false, false);
    let cube = CubeState {
        value: 1,
        owner: CubeOwner::Center,
        efficiency: 1.0,
        match_state: Some(state),
    };

    let match_equity = cubeful_equity(&outputs, &cube);
    assert!((0.0..=1.0).contains(&match_equity));
    approx_eq(
        match_equity,
        mwc(&state) + match_equity_swing(&state, gnubg_eval::cubeful::cubeless_equity(&outputs)),
        1.0e-6,
    );
}
