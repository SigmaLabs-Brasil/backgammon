//! Pure Rust GNU Backgammon neural-network evaluation.

pub mod classify;
pub mod contact;
pub mod crashed;
pub mod cubeful;
pub mod inputs;
pub mod met;
pub mod neuralnet;
pub mod race;
pub mod sanity;
pub mod weights;

use classify::{classify_position, Classification};
use gnubg_sys::PositionKey;
use gnubg_types::{board_from_old_key, Board};
use neuralnet::{NeuralNet, NeuralNetError};
use sanity::sanity_check;
use std::error::Error;
use std::fmt;
use std::sync::OnceLock;
use weights::{parse_weights, WeightError};

const WEIGHTS_DATA: &str = include_str!("../../gnubg-sys/vendor/gnubg.weights");
static EVALUATOR: OnceLock<Result<Evaluator, EvalError>> = OnceLock::new();

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EvalOutput {
    pub win: f32,
    pub win_gammon: f32,
    pub win_backgammon: f32,
    pub lose_gammon: f32,
    pub lose_backgammon: f32,
}

impl EvalOutput {
    pub const fn outputs(self) -> [f32; 5] {
        [
            self.win,
            self.win_gammon,
            self.win_backgammon,
            self.lose_gammon,
            self.lose_backgammon,
        ]
    }

    pub fn cubeless_equity(self) -> f32 {
        cubeful::cubeless_equity(&self.outputs())
    }

    pub fn cubeful_equity(self, cube: &cubeful::CubeState) -> f32 {
        cubeful::cubeful_equity(&self.outputs(), cube)
    }

    fn from_outputs(outputs: [f32; 5]) -> Self {
        Self {
            win: outputs[0],
            win_gammon: outputs[1],
            win_backgammon: outputs[2],
            lose_gammon: outputs[3],
            lose_backgammon: outputs[4],
        }
    }
}

#[derive(Clone, Debug)]
pub enum EvalError {
    WeightsParseError(String),
    InvalidBoard,
    InvalidInputLength { expected: usize, got: usize },
}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WeightsParseError(err) => write!(f, "weights parse error: {err}"),
            Self::InvalidBoard => f.write_str("invalid board"),
            Self::InvalidInputLength { expected, got } => {
                write!(f, "invalid input length: expected {expected}, got {got}")
            }
        }
    }
}

impl Error for EvalError {}

impl From<WeightError> for EvalError {
    fn from(value: WeightError) -> Self {
        Self::WeightsParseError(value.to_string())
    }
}

impl From<NeuralNetError> for EvalError {
    fn from(value: NeuralNetError) -> Self {
        match value {
            NeuralNetError::InvalidInputLength { expected, got } => {
                Self::InvalidInputLength { expected, got }
            }
        }
    }
}

struct Evaluator {
    contact: NeuralNet,
    race: NeuralNet,
    crashed: NeuralNet,
}

pub fn init_weights() -> Result<(), EvalError> {
    evaluator().map(|_| ())
}
pub fn simd_supported() -> bool {
    neuralnet::simd_supported()
}
pub fn embedded_weights_len() -> usize {
    WEIGHTS_DATA.len()
}

pub fn evaluate_position_key(key: &PositionKey) -> Result<EvalOutput, EvalError> {
    let gt_key = gnubg_types::PositionKey::from_raw(key.0);
    let board = board_from_old_key(&gt_key);
    evaluate_board(&board)
}

pub fn evaluate_board(board: &Board) -> Result<EvalOutput, EvalError> {
    validate_board(board)?;
    let evaluator = evaluator()?;
    let mut outputs = match classify_position(board) {
        Classification::Race => evaluator
            .race
            .feed_forward(&race::calculate_race_inputs(board))?,
        Classification::Contact => evaluator
            .contact
            .feed_forward(&contact::calculate_contact_inputs(board))?,
        Classification::Crashed => evaluator
            .crashed
            .feed_forward(&crashed::calculate_crashed_inputs(board))?,
    };
    sanity_check(&mut outputs);
    Ok(EvalOutput::from_outputs(outputs))
}

fn evaluator() -> Result<&'static Evaluator, EvalError> {
    EVALUATOR
        .get_or_init(|| {
            let weights = parse_weights(WEIGHTS_DATA)?;
            Ok(Evaluator {
                contact: NeuralNet::new(&weights.contact),
                race: NeuralNet::new(&weights.race),
                crashed: NeuralNet::new(&weights.crashed),
            })
        })
        .as_ref()
        .map_err(Clone::clone)
}

fn validate_board(board: &Board) -> Result<(), EvalError> {
    for side in board {
        if side.iter().sum::<u32>() > 15 {
            return Err(EvalError::InvalidBoard);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opening_board() -> Board {
        gnubg_types::board_from_position_id("4HPwATDgc/ABMA").expect("opening decodes")
    }

    #[test]
    fn init_parses_embedded_weights() {
        init_weights().expect("weights initialize");
        assert!(embedded_weights_len() > 1_000_000);
    }

    #[test]
    fn evaluates_opening_position_in_range() {
        let output = evaluate_board(&opening_board()).expect("opening evaluates");
        for value in output.outputs() {
            assert!((0.0..=1.0).contains(&value), "{value}");
        }
    }

    #[test]
    fn evaluation_is_deterministic() {
        let board = opening_board();
        let a = evaluate_board(&board).expect("first eval");
        let b = evaluate_board(&board).expect("second eval");
        assert_eq!(a, b);
    }

    #[test]
    fn evaluates_position_key() {
        let key = gnubg_sys::decode_position_id("4HPwATDgc/ABMA").expect("decode");
        let output = evaluate_position_key(&key).expect("eval key");
        assert!((0.0..=1.0).contains(&output.win));
    }
}
