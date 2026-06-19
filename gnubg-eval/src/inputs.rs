#![forbid(unsafe_code)]

use gnubg_types::Board;

pub const MINPPERPOINT: usize = 4;
pub const BOARD_POINTS: usize = 25;
pub const BASE_INPUTS: usize = 100;
pub const BASE_INPUTS_FULL: usize = BOARD_POINTS * MINPPERPOINT;
pub const MORE_INPUTS: usize = 25;
pub const I_OFF1: usize = 0;
pub const I_BREAK_CONTACT: usize = 3;
pub const I_BACK_CHEQUER: usize = 4;
pub const I_BACK_ANCHOR: usize = 5;
pub const I_FORWARD_ANCHOR: usize = 6;
pub const I_PIPLOSS: usize = 7;
pub const I_P1: usize = 8;
pub const I_P2: usize = 9;
pub const I_BACKESCAPES: usize = 10;
pub const I_ACONTAIN: usize = 11;
pub const I_ACONTAIN2: usize = 12;
pub const I_CONTAIN: usize = 13;
pub const I_CONTAIN2: usize = 14;
pub const I_MOBILITY: usize = 15;
pub const I_MOMENT2: usize = 16;
pub const I_ENTER: usize = 17;
pub const I_ENTER2: usize = 18;
pub const I_TIMING: usize = 19;
pub const I_BACKBONE: usize = 20;
pub const I_BACKG: usize = 21;
pub const I_BACKG1: usize = 22;
pub const I_FREEPIP: usize = 23;
pub const I_BACKRESCAPES: usize = 24;

pub fn base_inputs(board: &Board, side: usize) -> [f32; BASE_INPUTS] {
    let mut out = [0.0; BASE_INPUTS];
    for point in 0..24 {
        encode_board_point(
            board[side][point],
            &mut out[point * MINPPERPOINT..(point + 1) * MINPPERPOINT],
        );
    }
    encode_bar(
        board[side][24],
        &mut out[24 * MINPPERPOINT..25 * MINPPERPOINT],
    );
    out
}

#[inline]
fn encode_board_point(nc: u32, slots: &mut [f32]) {
    // GNU Backgammon's baseInputs() uses one-hot slots for exactly one/two
    // checkers, then a scaled overflow slot for stacks above three.
    slots[0] = (nc == 1) as u8 as f32;
    slots[1] = (nc == 2) as u8 as f32;
    slots[2] = (nc >= 3) as u8 as f32;
    slots[3] = if nc > 3 { (nc - 3) as f32 / 2.0 } else { 0.0 };
}

#[inline]
fn encode_bar(nc: u32, slots: &mut [f32]) {
    // The bar is encoded cumulatively in gnubg C: >=1, >=2, >=3.
    slots[0] = (nc >= 1) as u8 as f32;
    slots[1] = (nc >= 2) as u8 as f32;
    slots[2] = (nc >= 3) as u8 as f32;
    slots[3] = if nc > 3 { (nc - 3) as f32 / 2.0 } else { 0.0 };
}

pub fn calculate_half_inputs(board: &Board, side: usize) -> [f32; MORE_INPUTS] {
    let opp = 1 - side;
    calculate_half_inputs_for(&board[side], &board[opp])
}

pub(crate) fn calculate_half_inputs_for(
    an_board: &[u32; 25],
    an_board_opp: &[u32; 25],
) -> [f32; MORE_INPUTS] {
    let mut out = [0.0_f32; MORE_INPUTS];
    let n_opp_back = back_checker(an_board_opp);
    let n_opp_back_norm = 23 - n_opp_back;

    let mut break_contact = 0_i32;
    for i in (n_opp_back_norm + 1).max(0) as usize..25 {
        if an_board[i] > 0 {
            break_contact += (i as i32 + 1 - n_opp_back_norm).max(0) * an_board[i] as i32;
        }
    }
    out[I_BREAK_CONTACT] = break_contact as f32 / 167.0;

    let mut free_pips = 0_u32;
    for i in 0..n_opp_back_norm.max(0) as usize {
        free_pips += (i as u32 + 1) * an_board[i];
    }
    out[I_FREEPIP] = free_pips as f32 / 100.0;

    let n_back = back_checker(an_board).max(0) as usize;
    out[I_BACK_CHEQUER] = n_back as f32 / 24.0;

    let mut anchor = 0_usize;
    let start = if n_back == 24 { 23 } else { n_back };
    for i in (0..=start).rev() {
        if an_board[i] >= 2 {
            anchor = i;
            break;
        }
    }
    out[I_BACK_ANCHOR] = anchor as f32 / 24.0;

    let mut forward_anchor = 0_u32;
    for j in 18..=anchor {
        if an_board[j] >= 2 {
            forward_anchor = 24 - j as u32;
            break;
        }
    }
    if forward_anchor == 0 {
        for j in (12..=17).rev() {
            if an_board[j] >= 2 {
                forward_anchor = 24 - j as u32;
                break;
            }
        }
    }
    out[I_FORWARD_ANCHOR] = if forward_anchor == 0 {
        2.0
    } else {
        forward_anchor as f32 / 6.0
    };

    let inner_points = (0..6).filter(|&i| an_board[i] >= 2).count();
    let mut hit_rolls = 0_u32;
    let mut multi_hit_rolls = 0_u32;
    let mut pip_loss = 0_u32;
    for d0 in 1..=6 {
        for d1 in 1..=6 {
            let mut hits = 0_u32;
            let mut best_pip = 0_u32;
            for blot in 0..24 {
                if an_board_opp[blot] != 1 {
                    continue;
                }
                for dist in [d0, d1, d0 + d1] {
                    let hitter = blot + dist;
                    if hitter < 25
                        && an_board[hitter] > 0
                        && !(hitter < 6 && an_board[hitter] == 2 && inner_points < 3)
                    {
                        hits += 1;
                        best_pip = best_pip.max((hitter - dist + 1) as u32);
                        break;
                    }
                }
            }
            if hits > 0 {
                hit_rolls += 1;
                pip_loss += best_pip;
            }
            if hits > 1 {
                multi_hit_rolls += 1;
            }
        }
    }
    out[I_PIPLOSS] = pip_loss as f32 / (12.0 * 36.0);
    out[I_P1] = hit_rolls as f32 / 36.0;
    out[I_P2] = multi_hit_rolls as f32 / 36.0;

    out[I_BACKESCAPES] = escapes(an_board, 23 - n_opp_back_norm) as f32 / 36.0;
    out[I_BACKRESCAPES] = escapes1(an_board, 23 - n_opp_back_norm) as f32 / 36.0;

    let mut contain_min = 36;
    for i in 15..(24 - n_opp_back_norm).max(15) as usize {
        contain_min = contain_min.min(escapes(an_board, i as i32));
    }
    out[I_ACONTAIN] = (36 - contain_min) as f32 / 36.0;
    out[I_ACONTAIN2] = out[I_ACONTAIN] * out[I_ACONTAIN];

    let mut contain = 36;
    for i in 15..24 {
        contain = contain.min(escapes(an_board, i));
    }
    out[I_CONTAIN] = (36 - contain) as f32 / 36.0;
    out[I_CONTAIN2] = out[I_CONTAIN] * out[I_CONTAIN];

    let mut mobility = 0_u32;
    for i in 6..25 {
        if an_board[i] > 0 {
            mobility += (i as u32 - 5) * an_board[i] * escapes(an_board_opp, i as i32) as u32;
        }
    }
    out[I_MOBILITY] = mobility as f32 / 3600.0;

    let mut total_checkers = 0_u32;
    let mut weighted = 0_u32;
    for (i, &n) in an_board.iter().enumerate() {
        total_checkers += n;
        weighted += i as u32 * n;
    }
    if total_checkers > 0 {
        let avg = weighted.div_ceil(total_checkers);
        let mut count = 0_u32;
        let mut moment = 0_u32;
        for (i, &n) in an_board.iter().enumerate().skip(avg as usize + 1) {
            count += n;
            moment += n * (i as u32 - avg).pow(2);
        }
        if count > 0 {
            out[I_MOMENT2] = moment.div_ceil(count) as f32 / 400.0;
        }
    }

    out[I_ENTER] = enter_loss(an_board, an_board_opp);
    let closed = (0..6).filter(|&i| an_board_opp[i] > 1).count() as i32;
    out[I_ENTER2] = (36 - (closed - 6).pow(2)) as f32 / 36.0;

    out[I_TIMING] = timing(an_board, n_opp_back_norm.max(0) as usize);
    out[I_BACKBONE] = backbone(an_board);
    let anchors = (18..24).filter(|&i| an_board[i] > 1).count();
    if anchors >= 1 {
        let total: u32 = an_board[18..25].iter().sum();
        if anchors > 1 {
            out[I_BACKG] = total.saturating_sub(3) as f32 / 4.0;
        } else {
            out[I_BACKG1] = total as f32 / 8.0;
        }
    }
    for value in &mut out {
        if value.is_nan() || *value < 0.0 {
            *value = 0.0;
        }
    }
    out
}

pub(crate) fn men_off_all(an_board: &[u32; 25]) -> [f32; 3] {
    encode_men_off(15_i32 - an_board.iter().sum::<u32>() as i32, 5, 10)
}

pub(crate) fn men_off_non_crashed(an_board: &[u32; 25]) -> [f32; 3] {
    let men_off = 15_i32 - an_board.iter().sum::<u32>() as i32;
    if men_off <= 2 {
        [
            if men_off > 0 {
                men_off as f32 / 3.0
            } else {
                0.0
            },
            0.0,
            0.0,
        ]
    } else if men_off <= 5 {
        [1.0, (men_off - 3) as f32 / 3.0, 0.0]
    } else {
        [1.0, 1.0, (men_off - 6) as f32 / 3.0]
    }
}

fn encode_men_off(men_off: i32, first: i32, second: i32) -> [f32; 3] {
    if men_off <= first {
        [
            if men_off > 0 {
                men_off as f32 / first as f32
            } else {
                0.0
            },
            0.0,
            0.0,
        ]
    } else if men_off <= second {
        [1.0, (men_off - first) as f32 / first as f32, 0.0]
    } else {
        [1.0, 1.0, (men_off - second) as f32 / first as f32]
    }
}

fn back_checker(points: &[u32; 25]) -> i32 {
    for idx in (0..25).rev() {
        if points[idx] > 0 {
            return idx as i32;
        }
    }
    -1
}

fn escapes(board: &[u32; 25], n: i32) -> i32 {
    let af = escape_mask(board, n);
    let mut count = 0;
    for n0 in 0..=5 {
        for n1 in 0..=n0 {
            if (af & (1 << (n0 + n1 + 1))) == 0 && !((af & (1 << n0)) != 0 && (af & (1 << n1)) != 0)
            {
                count += if n0 == n1 { 1 } else { 2 };
            }
        }
    }
    count
}

fn escapes1(board: &[u32; 25], n: i32) -> i32 {
    let af = escape_mask(board, n);
    if af == 0 {
        return 0;
    }
    let low = af.trailing_zeros() as i32;
    let mut count = 0;
    for n0 in 0..=5 {
        for n1 in 0..=n0 {
            if n0 + n1 + 1 > low
                && (af & (1 << (n0 + n1 + 1))) == 0
                && !((af & (1 << n0)) != 0 && (af & (1 << n1)) != 0)
            {
                count += if n0 == n1 { 1 } else { 2 };
            }
        }
    }
    count
}

fn escape_mask(board: &[u32; 25], n: i32) -> i32 {
    let m = n.min(12).max(0);
    let mut af = 0_i32;
    for i in 0..m {
        let idx = 24 + i - n;
        if (0..25).contains(&idx) && board[idx as usize] >= 2 {
            af |= 1 << i;
        }
    }
    af
}

fn enter_loss(an_board: &[u32; 25], an_board_opp: &[u32; 25]) -> f32 {
    if an_board[24] == 0 {
        return 0.0;
    }
    let two = an_board[24] > 1;
    let mut loss = 0_u32;
    for i in 0..6 {
        if an_board_opp[i] > 1 {
            loss += 4 * (i as u32 + 1);
            for j in i + 1..6 {
                if an_board_opp[j] > 1 {
                    loss += 2 * (i as u32 + j as u32 + 2);
                } else if two {
                    loss += 2 * (i as u32 + 1);
                }
            }
        } else if two {
            for j in i + 1..6 {
                if an_board_opp[j] > 1 {
                    loss += 2 * (j as u32 + 1);
                }
            }
        }
    }
    loss as f32 / (36.0 * (49.0 / 6.0))
}

fn timing(an_board: &[u32; 25], n_opp_back: usize) -> f32 {
    let mut t = 24_i32 * an_board[24] as i32;
    let mut no = an_board[24] as i32;
    let m = n_opp_back.max(11);
    let mut i = 23_i32;
    while i > m as i32 {
        let n = an_board[i as usize];
        if n > 0 && n != 2 {
            let ns = if n > 2 { n - 2 } else { 1 } as i32;
            no += ns;
            t += i * ns;
        }
        i -= 1;
    }
    while i >= 6 {
        let n = an_board[i as usize] as i32;
        no += n;
        t += i * n;
        i -= 1;
    }
    while i >= 0 {
        let n = an_board[i as usize] as i32;
        if n > 2 {
            t += i * (n - 2);
            no += n - 2;
        } else if n < 2 {
            let nm = 2 - n;
            if no >= nm {
                t -= i * nm;
                no -= nm;
            }
        }
        i -= 1;
    }
    t as f32 / 100.0
}

fn backbone(an_board: &[u32; 25]) -> f32 {
    let ac = [
        11, 11, 11, 11, 11, 11, 11, 6, 5, 4, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ];
    let mut pa = None;
    let mut w = 0_u32;
    let mut total = 0_u32;
    for np in (1..24).rev() {
        if an_board[np] >= 2 {
            if let Some(prev) = pa {
                let d = prev - np;
                w += ac[d] * an_board[prev];
                total += an_board[prev];
            } else {
                pa = Some(np);
            }
        }
    }
    if total > 0 {
        1.0 - (w as f32 / (total as f32 * 11.0))
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_board_base_inputs_are_zero() {
        let b = [[0_u32; 25]; 2];
        assert!(base_inputs(&b, 0).iter().all(|v| *v == 0.0));
    }

    #[test]
    fn one_checker_on_bar_sets_first_slot() {
        let mut b = [[0_u32; 25]; 2];
        b[0][24] = 1; // bar
        let inputs = base_inputs(&b, 0);
        assert_eq!(&inputs[96..100], &[1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn full_point_sets_scaled_slot() {
        let mut b = [[0_u32; 25]; 2];
        b[0][6] = 5;
        let inputs = base_inputs(&b, 0);
        let offset = 6 * 4; // point 6 starts at slot 24
        assert_eq!(&inputs[offset..offset + 4], &[0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn half_inputs_are_finite() {
        let mut b = [[0_u32; 25]; 2];
        b[1][24] = 2;
        b[1][13] = 5;
        b[1][8] = 3;
        b[1][6] = 5;
        b[0][1] = 2;
        b[0][12] = 5;
        b[0][7] = 3;
        b[0][6] = 5;
        let inputs = calculate_half_inputs(&b, 1);
        assert_eq!(inputs.len(), MORE_INPUTS);
        assert!(inputs.iter().all(|v| v.is_finite()));
        assert!(inputs[I_BACK_CHEQUER] > 0.0);
    }
}
