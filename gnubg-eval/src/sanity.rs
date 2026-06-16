/// Post-process raw NN outputs to enforce probabilistic consistency.
///
/// Matches GNU Backgammon's `SanityCheck()` intent for evaluator outputs:
/// `win >= win_gammon >= win_backgammon`,
/// `lose >= lose_gammon >= lose_backgammon`, and total probability <= 1.0.
pub fn sanity_check(outputs: &mut [f32; 5]) {
    // outputs = [win, win_gammon, win_backgammon, lose_gammon, lose_backgammon]

    // Win-side: win >= win_gammon >= win_backgammon.
    outputs[1] = outputs[1].min(outputs[0]);
    outputs[2] = outputs[2].min(outputs[1]);

    // Lose-side: lose >= lose_gammon >= lose_backgammon.
    let lose = (1.0 - outputs[0]).max(0.0);
    outputs[3] = outputs[3].min(lose);
    outputs[4] = outputs[4].min(outputs[3]);

    // Normalize: sum of all 5 outputs must not exceed 1.0.
    let sum: f32 = outputs.iter().sum();
    if sum > 1.0 {
        let non_win_sum: f32 = outputs[1..].iter().sum();
        let remaining = (1.0 - outputs[0]).max(0.0);

        if non_win_sum > 0.0 {
            let scale = remaining / non_win_sum;
            for output in &mut outputs[1..] {
                *output *= scale;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-6;

    #[test]
    fn already_consistent_is_unchanged() {
        let mut out = [0.5, 0.2, 0.05, 0.15, 0.03];
        let original = out;

        sanity_check(&mut out);

        assert_eq!(out, original, "consistent probs should be unchanged");
    }

    #[test]
    fn clamps_win_gammon_below_win() {
        let mut out = [0.4, 0.5, 0.2, 0.0, 0.0];

        sanity_check(&mut out);

        assert!(out[0] >= out[1], "win >= win_gammon");
        assert!(out[1] >= out[2], "win_gammon >= win_backgammon");
        assert_eq!(out[1], 0.4);
    }

    #[test]
    fn clamps_lose_gammon_below_lose() {
        let mut out = [0.8, 0.0, 0.0, 0.3, 0.0];

        sanity_check(&mut out);

        let lose = 1.0 - out[0];
        assert!(lose >= out[3], "lose >= lose_gammon");
        assert!(out[3] >= out[4], "lose_gammon >= lose_backgammon");
        assert!((out[3] - lose).abs() < EPSILON);
    }

    #[test]
    fn normalizes_excessive_sum() {
        let mut out = [0.6, 0.3, 0.2, 0.2, 0.1];

        sanity_check(&mut out);

        let sum: f32 = out.iter().sum();
        assert!(sum <= 1.0 + EPSILON, "sum = {sum}");
        assert_eq!(out[0], 0.6, "win is unchanged during normalization");
    }

    #[test]
    fn running_total_is_one() {
        let mut out = [0.6, 0.3, 0.2, 0.2, 0.1];

        sanity_check(&mut out);

        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < EPSILON, "sum = {sum}");
    }
}
