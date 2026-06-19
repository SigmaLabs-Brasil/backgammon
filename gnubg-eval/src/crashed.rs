#![forbid(unsafe_code)]

use crate::inputs::{
    base_inputs, calculate_half_inputs_for, men_off_all, BASE_INPUTS_FULL, I_OFF1,
    MORE_INPUTS,
};
use gnubg_types::Board;

pub const CRASHED_INPUTS: usize = (BASE_INPUTS_FULL + MORE_INPUTS) * 2;

pub fn calculate_crashed_inputs(board: &Board) -> [f32; CRASHED_INPUTS] {
    let mut inputs = [0.0_f32; CRASHED_INPUTS];

    // Encode base inputs for both sides (25 points × 4 slots each = 100 per side)
    let base0 = base_inputs(board, 0);
    inputs[..BASE_INPUTS_FULL].copy_from_slice(&base0);
    let base1 = base_inputs(board, 1);
    inputs[BASE_INPUTS_FULL..2 * BASE_INPUTS_FULL].copy_from_slice(&base1);

    // Half inputs (25 features per side)
    let first = BASE_INPUTS_FULL * 2;
    let second = first + MORE_INPUTS;
    let mut half_first = calculate_half_inputs_for(&board[1], &board[0]);
    half_first[I_OFF1..I_OFF1 + 3].copy_from_slice(&men_off_all(&board[1]));
    let mut half_second = calculate_half_inputs_for(&board[0], &board[1]);
    half_second[I_OFF1..I_OFF1 + 3].copy_from_slice(&men_off_all(&board[0]));
    inputs[first..first + MORE_INPUTS].copy_from_slice(&half_first);
    inputs[second..second + MORE_INPUTS].copy_from_slice(&half_second);
    inputs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crashed_inputs_have_expected_length_and_are_finite() {
        let mut b = [[0_u32; 25]; 2];
        b[0][24] = 6;
        b[1][24] = 6;
        let inputs = calculate_crashed_inputs(&b);
        assert_eq!(inputs.len(), CRASHED_INPUTS);
        assert!(inputs.iter().all(|v| v.is_finite()));
    }
}
