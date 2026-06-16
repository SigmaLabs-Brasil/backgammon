#![forbid(unsafe_code)]

use gnubg_types::Board;

pub const HALF_RACE_INPUTS: usize = 107;
pub const RACE_INPUTS: usize = HALF_RACE_INPUTS * 2;
const RI_OFF: usize = 92;
const RI_NCROSS: usize = 106;

pub fn calculate_race_inputs(board: &Board) -> [f32; RACE_INPUTS] {
    let mut inputs = [0.0_f32; RACE_INPUTS];
    for side in 0..2 {
        let side_offset = side * HALF_RACE_INPUTS;
        let side_board = &board[side];
        let mut men_off = 15_i32;
        for i in 0..23 {
            let nc = side_board[i];
            men_off -= nc as i32;
            let k = side_offset + i * 4;
            inputs[k] = (nc == 1) as u8 as f32;
            inputs[k + 1] = (nc == 2) as u8 as f32;
            inputs[k + 2] = (nc >= 3) as u8 as f32;
            inputs[k + 3] = if nc > 3 { (nc - 3) as f32 / 2.0 } else { 0.0 };
        }
        if (1..=14).contains(&men_off) {
            inputs[side_offset + RI_OFF + men_off as usize - 1] = 1.0;
        }
        let mut n_cross = 0_u32;
        for k in 1..4 {
            for i in 6 * k..6 * k + 6 {
                n_cross += side_board[i] * k as u32;
            }
        }
        inputs[side_offset + RI_NCROSS] = n_cross as f32 / 10.0;
    }
    inputs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn race_inputs_have_expected_length_and_are_finite() {
        let mut b = [[0_u32; 25]; 2];
        b[0][1] = 15;
        b[1][5] = 15;
        let inputs = calculate_race_inputs(&b);
        assert_eq!(inputs.len(), RACE_INPUTS);
        assert!(inputs.iter().all(|v| v.is_finite()));
    }
}
