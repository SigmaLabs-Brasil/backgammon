#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CubeOwner {
    Player,
    Opponent,
    Center,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CubeState {
    pub value: i32,
    pub owner: CubeOwner,
    pub efficiency: f32,
}

impl Default for CubeState {
    fn default() -> Self {
        Self {
            value: 1,
            owner: CubeOwner::Center,
            efficiency: 1.0,
        }
    }
}

pub fn cubeless_equity(outputs: &[f32; 5]) -> f32 {
    (2.0 * outputs[0] - 1.0) + outputs[1] + outputs[2] - outputs[3] - outputs[4]
}

pub fn dead_cube_equity(outputs: &[f32; 5]) -> f32 {
    cubeless_equity(outputs)
}

pub fn live_cube_equity(outputs: &[f32; 5]) -> f32 {
    let cubeless = cubeless_equity(outputs);
    let p = (cubeless + 1.0) / 2.0;
    2.0 * p - 1.0
}

pub fn cubeful_equity(outputs: &[f32; 5], cube: &CubeState) -> f32 {
    let dead = dead_cube_equity(outputs);
    let live = live_cube_equity(outputs);
    let cubeful = live - cube.efficiency * (live - dead);

    match cube.owner {
        CubeOwner::Player => cubeful * cube.value as f32,
        CubeOwner::Opponent => -(-cubeful * cube.value as f32),
        CubeOwner::Center => cubeful,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn current_inline_equity(outputs: &[f32; 5]) -> f32 {
        (2.0 * outputs[0] - 1.0) + outputs[1] + outputs[2] - outputs[3] - outputs[4]
    }

    fn approx_eq(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 1.0e-6,
            "actual={actual} expected={expected}"
        );
    }

    #[test]
    fn cubeless_equity_matches_current_inline_formula_for_many_outputs() {
        let mut state = 0x296_u64;
        for _ in 0..128 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let win = ((state >> 32) as u32) as f32 / u32::MAX as f32;
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let win_gammon = win * (((state >> 32) as u32) as f32 / u32::MAX as f32);
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let win_backgammon = win_gammon * (((state >> 32) as u32) as f32 / u32::MAX as f32);
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let lose_gammon = (1.0 - win) * (((state >> 32) as u32) as f32 / u32::MAX as f32);
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let lose_backgammon = lose_gammon * (((state >> 32) as u32) as f32 / u32::MAX as f32);
            let outputs = [
                win,
                win_gammon,
                win_backgammon,
                lose_gammon,
                lose_backgammon,
            ];

            approx_eq(cubeless_equity(&outputs), current_inline_equity(&outputs));
        }
    }

    #[test]
    fn centered_dead_cube_equals_cubeless_equity() {
        let outputs = [0.55, 0.12, 0.03, 0.10, 0.02];
        let cube = CubeState {
            value: 1,
            owner: CubeOwner::Center,
            efficiency: 1.0,
        };

        approx_eq(cubeful_equity(&outputs, &cube), cubeless_equity(&outputs));
    }

    #[test]
    fn maximum_backgammon_win_is_three_points() {
        let outputs = [1.0, 1.0, 1.0, 0.0, 0.0];

        approx_eq(cubeless_equity(&outputs), 3.0);
    }

    #[test]
    fn maximum_backgammon_loss_is_minus_three_points() {
        let outputs = [0.0, 0.0, 0.0, 1.0, 1.0];

        approx_eq(cubeless_equity(&outputs), -3.0);
    }

    #[test]
    fn symmetric_position_has_zero_equity() {
        let outputs = [0.5, 0.12, 0.03, 0.12, 0.03];

        approx_eq(cubeless_equity(&outputs), 0.0);
        approx_eq(cubeful_equity(&outputs, &CubeState::default()), 0.0);
    }
}
