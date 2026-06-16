use gnubg_eval::{
    evaluate_board,
    neuralnet::NeuralNet,
    weights::{parse_weights, CONTACT_INPUTS},
};

const REAL_WEIGHTS: &str = include_str!("../../gnubg-sys/vendor/gnubg.weights");

fn assert_outputs_close(actual: [f32; 5], expected: [f32; 5], epsilon: f32) {
    for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (*actual - *expected).abs() <= epsilon,
            "output[{idx}] expected {expected}, got {actual}"
        );
    }
}

fn evaluate_position_id(position_id: &str) -> [f32; 5] {
    let board = gnubg_types::board_from_position_id(position_id).expect("position decodes");
    evaluate_board(&board)
        .expect("position evaluates")
        .outputs()
}

#[test]
fn contact_network_zero_input_matches_c_reference() {
    let weights = parse_weights(REAL_WEIGHTS).expect("weights parse");
    let net = NeuralNet::new(&weights.contact);
    let inputs = [0.0_f32; CONTACT_INPUTS];

    let output = net
        .feed_forward_scalar(&inputs)
        .expect("zero input evaluates");

    assert_outputs_close(output, [1.0, 0.0, 1.0, 0.0, 0.0], 1e-6);
}

#[test]
fn opening_position_matches_post_sanity_check_smoke_values() {
    let output = evaluate_position_id("4HPwATDgc/ABMA");

    assert_outputs_close(output, [1.0, 0.0, 0.0, 0.0, 0.0], 1e-4);
}

#[test]
fn race_position_outputs_are_valid_and_plausible() {
    let [win, win_gammon, win_backgammon, lose_gammon, lose_backgammon] =
        evaluate_position_id("2sAQAQAPAQAAAA");

    for value in [
        win,
        win_gammon,
        win_backgammon,
        lose_gammon,
        lose_backgammon,
    ] {
        assert!((0.0..=1.0).contains(&value), "{value}");
    }
    assert!(win_gammon <= win, "win gammon cannot exceed win");
    assert!(
        win_backgammon <= win_gammon,
        "win backgammon cannot exceed win gammon"
    );
    assert!(
        lose_backgammon <= lose_gammon,
        "lose backgammon cannot exceed lose gammon"
    );
}

#[test]
fn evaluation_is_deterministic_for_opening_position() {
    let first = evaluate_position_id("4HPwATDgc/ABMA");
    let second = evaluate_position_id("4HPwATDgc/ABMA");

    assert_eq!(first, second);
}
