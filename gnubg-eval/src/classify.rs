#![forbid(unsafe_code)]

use gnubg_types::Board;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Classification {
    Race,
    Contact,
    Crashed,
}

pub fn classify_position(board: &Board) -> Classification {
    let n_opp_back = back_checker(&board[0]);
    let n_back = back_checker(&board[1]);
    if n_back + n_opp_back > 22 {
        for side_board in board.iter().take(2) {
            let total: u32 = side_board.iter().sum();
            if total <= 6 {
                return Classification::Crashed;
            }
            if side_board[0] > 1 {
                if total <= 6 + side_board[0] {
                    return Classification::Crashed;
                }
                if 1 + total.saturating_sub(side_board[0] + side_board[1]) <= 6 && side_board[1] > 1
                {
                    return Classification::Crashed;
                }
            } else if total <= 6 + side_board[1].saturating_sub(1) {
                return Classification::Crashed;
            }
        }
        Classification::Contact
    } else {
        Classification::Race
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

#[cfg(test)]
mod tests {
    use super::*;

    fn opening_board() -> Board {
        let mut b = [[0u32; 25]; 2];
        b[1][24] = 2;
        b[1][13] = 5;
        b[1][8] = 3;
        b[1][6] = 5;
        b[0][1] = 2;
        b[0][12] = 5;
        b[0][7] = 3;
        b[0][6] = 5;
        b
    }

    #[test]
    fn opening_position_is_contact() {
        assert_eq!(classify_position(&opening_board()), Classification::Contact);
    }

    #[test]
    fn race_position_is_race() {
        let mut b = [[0u32; 25]; 2];
        b[0][1] = 15;
        b[1][5] = 15;
        assert_eq!(classify_position(&b), Classification::Race);
    }

    #[test]
    fn low_checker_contact_is_crashed() {
        let mut b = opening_board();
        b[0] = [0; 25];
        b[0][24] = 6;
        assert_eq!(classify_position(&b), Classification::Crashed);
    }
}
