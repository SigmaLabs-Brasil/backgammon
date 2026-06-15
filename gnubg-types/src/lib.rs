//! Pure Rust core data types for GNU Backgammon.
//!
//! Ported from the C headers in `lib/gnubg-types.h`, `positionid.h`, `matchid.h`,
//! `eval.h` and their corresponding `.c` implementations.
//!
//! # Zero-dependency (except optional serde)
//!
//! This crate has **no runtime dependencies** by default. The `serialize` feature
//! enables serde `Serialize`/`Deserialize` on all public types.

#![forbid(unsafe_code)]

use core::fmt;

// ---------------------------------------------------------------------------
// PositionKey — 10-byte compressed board representation
// ---------------------------------------------------------------------------

/// A 10-byte key uniquely identifying a backgammon board position.
///
/// In gnubg's C code this was `oldpositionkey` (`unsigned char auch[10]`).
/// The modern `positionkey` (7 × `unsigned int`) is used for evaluation; this
/// is the **old** position key used for PositionID encoding and legacy lookups.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct PositionKey(pub [u8; 10]);

impl PositionKey {
    pub const fn from_raw(raw: [u8; 10]) -> Self {
        Self(raw)
    }

    pub const fn as_raw(&self) -> &[u8; 10] {
        &self.0
    }
}

// ---------------------------------------------------------------------------
// Board — TanBoard representation
// ---------------------------------------------------------------------------

/// A backgammon board represented as `[player][point]`.
///
/// - `board[0]` = player-0's checkers (the player whose perspective the board
///   is shown from — "inner"/bottom in gnubg's convention)
/// - `board[1]` = player-1's checkers (the opponent — "outer"/top)
/// - Point 0      = bear-off tray
/// - Points 1..24 = the 24 board points (1 = opponent's 1-point, 24 = player's 1-point)
/// - Point 24     = the bar
///
/// This matches gnubg's `TanBoard` typedef: `unsigned int[2][25]`.
pub type Board = [[u32; 25]; 2];

/// The number of points on a backgammon board (0..24 inclusive).
pub const NUM_POINTS: usize = 25;

/// Maximum checkers per player.
pub const MAX_CHECKERS: u32 = 15;

// ---------------------------------------------------------------------------
// Dice
// ---------------------------------------------------------------------------

/// A pair of dice values, each in 1..6.
pub type Dice = (u8, u8);

// ---------------------------------------------------------------------------
// Move — a complete play of up to 4 sub-moves
// ---------------------------------------------------------------------------

/// A single complete move (all chequers played for one roll).
///
/// Corresponds to gnubg's `move` struct in `eval.h`.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Move {
    /// Up to 4 sub-moves, each `(source, destination)`.
    /// `source` and `destination` are point indices (0..24, where 24 = bar).
    /// A destination < 0 conceptually means bear-off.
    pub from_to: [Option<(u8, u8)>; 4],
    /// Number of chequers moved.
    pub c_moves: u8,
    /// Total pips moved.
    pub c_pips: u8,
    /// The resulting board position key after this move.
    pub key: PositionKey,
}

impl Move {
    /// Create an empty move.
    pub const fn empty() -> Self {
        Self {
            from_to: [None; 4],
            c_moves: 0,
            c_pips: 0,
            key: PositionKey([0; 10]),
        }
    }
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let parts: Vec<String> = self
            .from_to
            .iter()
            .flatten()
            .map(|(src, dst)| {
                if *dst == 0 || *src as i16 - *dst as i16 > 24 {
                    format!("{}->off", src)
                } else {
                    format!("{}->{}", src, dst)
                }
            })
            .collect();
        write!(f, "Move({} pips={})", parts.join(", "), self.c_pips)
    }
}

// ---------------------------------------------------------------------------
// MoveList
// ---------------------------------------------------------------------------

/// A list of legal moves generated for a given roll.
///
/// Corresponds to gnubg's `movelist` struct.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct MoveList {
    pub moves: Vec<Move>,
}

impl MoveList {
    pub const fn new() -> Self {
        Self { moves: Vec::new() }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            moves: Vec::with_capacity(cap),
        }
    }

    pub fn len(&self) -> usize {
        self.moves.len()
    }

    pub fn is_empty(&self) -> bool {
        self.moves.is_empty()
    }

    pub fn push(&mut self, mv: Move) {
        self.moves.push(mv);
    }

    pub fn clear(&mut self) {
        self.moves.clear();
    }
}

impl Default for MoveList {
    fn default() -> Self {
        Self::new()
    }
}

impl core::ops::Deref for MoveList {
    type Target = Vec<Move>;
    fn deref(&self) -> &Vec<Move> {
        &self.moves
    }
}

impl core::ops::DerefMut for MoveList {
    fn deref_mut(&mut self) -> &mut Vec<Move> {
        &mut self.moves
    }
}

impl<'a> IntoIterator for &'a MoveList {
    type Item = &'a Move;
    type IntoIter = std::slice::Iter<'a, Move>;

    fn into_iter(self) -> Self::IntoIter {
        self.moves.iter()
    }
}

impl<'a> IntoIterator for &'a mut MoveList {
    type Item = &'a mut Move;
    type IntoIter = std::slice::IterMut<'a, Move>;

    fn into_iter(self) -> Self::IntoIter {
        self.moves.iter_mut()
    }
}

// ---------------------------------------------------------------------------
// Variation
// ---------------------------------------------------------------------------

/// Backgammon variation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum Variation {
    Standard,
    Nackgammon,
    Hypergammon1,
    Hypergammon2,
    Hypergammon3,
}

// ---------------------------------------------------------------------------
// GameState
// ---------------------------------------------------------------------------

/// Game state enum.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum GameState {
    None,
    Playing,
    Over,
    Resigned,
    Drop,
}

// ---------------------------------------------------------------------------
// MatchState
// ---------------------------------------------------------------------------

/// Full match state, corresponding to gnubg's `matchstate` struct.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct MatchState {
    pub board: Board,
    pub dice: (u8, u8),
    pub f_turn: bool,
    pub f_resigned: i32,
    pub f_resignation_declined: bool,
    pub f_doubled: bool,
    pub c_games: u32,
    pub f_move: bool,
    pub f_cube_owner: i32,
    pub f_crawford: bool,
    pub f_post_crawford: bool,
    pub n_match_to: i32,
    pub an_score: [i32; 2],
    pub n_cube: u32,
    pub c_beavers: u32,
    pub bgv: Variation,
    pub f_cube_use: bool,
    pub f_jacoby: bool,
    pub gs: GameState,
}

// ---------------------------------------------------------------------------
// Base64 helper (gnubg alphabet)
// ---------------------------------------------------------------------------

/// GNU Backgammon base64 alphabet (matches `positionid.c`).
const BASE64_ALPHABET: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Decode a single base64 character (gnubg alphabet).
#[inline]
fn base64_decode(ch: u8) -> Option<u8> {
    match ch {
        b'A'..=b'Z' => Some(ch - b'A'),
        b'a'..=b'z' => Some(ch - b'a' + 26),
        b'0'..=b'9' => Some(ch - b'0' + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

/// Encode 6 bits to a base64 character.
#[inline]
fn base64_encode(val: u8) -> u8 {
    BASE64_ALPHABET[(val & 0x3F) as usize]
}

// ---------------------------------------------------------------------------
// PositionKey from board (the "modern" key used in eval)
// ---------------------------------------------------------------------------

/// Compute the internal 10-byte old-position key from a board.
///
/// This is the XOR-based encoding used by gnubg's `oldPositionKey()`.
/// The algorithm encodes each player's checkers as a run-length encoding
/// of empty/filled runs, producing up to 10 bytes.
pub fn old_position_key(board: &Board) -> PositionKey {
    let mut key = [0u8; 10];
    let mut i_bit: u32 = 0;

    for side in 0..2 {
        for point in 0..25 {
            let nc = board[side][point];
            if nc > 0 {
                // Set `nc` bits to 1, then one 0 bit
                for _ in 0..nc {
                    let k = (i_bit / 8) as usize;
                    let r = i_bit % 8;
                    if k < 10 {
                        key[k] |= 1 << r;
                    }
                    i_bit += 1;
                }
                // Separator 0 bit
                i_bit += 1;
            } else {
                // Empty point: one 0 bit
                i_bit += 1;
            }
        }
    }

    PositionKey(key)
}

/// Reconstruct a board from an old position key.
///
/// This is the inverse of `old_position_key`, corresponding to
/// gnubg's `oldPositionFromKey()`.
pub fn board_from_old_key(key: &PositionKey) -> Board {
    let mut board: Board = [[0u32; 25]; 2];
    let mut side: usize = 0;
    let mut point: usize = 0;

    for &byte in key.0.iter() {
        for bit_idx in 0..8 {
            if side >= 2 || point >= 25 {
                return board;
            }
            if (byte >> bit_idx) & 1 == 1 {
                board[side][point] += 1;
            } else {
                point += 1;
                if point == 25 {
                    side += 1;
                    point = 0;
                }
            }
        }
    }

    board
}

// ---------------------------------------------------------------------------
// PositionID encoding/decoding
// ---------------------------------------------------------------------------

/// Length of a PositionID string.
pub const POSITION_ID_LEN: usize = 14;

/// Encode a board to a 14-character PositionID string.
///
/// This implements gnubg's `PositionID()` function: first converts to an
/// old-position key via `oldPositionKey()`, then base64-encodes the 10-byte
/// key into 14 characters.
pub fn position_id_from_board(board: &Board) -> String {
    let key = old_position_key(board);
    position_id_from_old_key(&key)
}

/// Encode an old-position key directly to a PositionID string.
pub fn position_id_from_old_key(key: &PositionKey) -> String {
    let bytes = &key.0;
    let mut out = [0u8; POSITION_ID_LEN];

    // Encode 10 bytes → 14 base64 chars
    for i in 0..3 {
        let chunk = i * 3;
        out[i * 4] = base64_encode(bytes[chunk] >> 2);
        out[i * 4 + 1] = base64_encode(((bytes[chunk] & 0x03) << 4) | (bytes[chunk + 1] >> 4));
        out[i * 4 + 2] =
            base64_encode(((bytes[chunk + 1] & 0x0F) << 2) | (bytes[chunk + 2] >> 6));
        out[i * 4 + 3] = base64_encode(bytes[chunk + 2] & 0x3F);
    }
    // Last byte (index 9): encode 2 bytes? No, only byte[9] remains (10 bytes = 3×3 + 1)
    // Actually bytes 0-8 are covered (3 groups of 3 = 9 bytes), byte[9] is left:
    out[12] = base64_encode(bytes[9] >> 2);
    out[13] = base64_encode((bytes[9] & 0x03) << 4);

    core::str::from_utf8(&out)
        .expect("base64 encoding always produces valid ASCII")
        .to_string()
}

/// Decode a 14-character PositionID string into a board.
///
/// Returns `None` if the string is invalid.
pub fn board_from_position_id(s: &str) -> Option<Board> {
    let key = old_key_from_position_id(s)?;
    Some(board_from_old_key(&key))
}

/// Decode a 14-character PositionID into an old-position key.
pub fn old_key_from_position_id(s: &str) -> Option<PositionKey> {
    if s.len() != POSITION_ID_LEN {
        return None;
    }

    let bytes = s.as_bytes();
    let mut decoded = [0u8; 10];
    let mut out_idx = 0;

    // Decode 14 base64 chars → 10 bytes
    let mut vals = [0u8; 14];
    for (i, &ch) in bytes.iter().enumerate() {
        vals[i] = base64_decode(ch)?;
    }

    for i in 0..3 {
        let src = i * 4;
        decoded[out_idx] = (vals[src] << 2) | (vals[src + 1] >> 4);
        decoded[out_idx + 1] = (vals[src + 1] << 4) | (vals[src + 2] >> 2);
        decoded[out_idx + 2] = (vals[src + 2] << 6) | vals[src + 3];
        out_idx += 3;
    }

    // Last group: vals[12..14] → decoded[9]
    decoded[9] = (vals[12] << 2) | (vals[13] >> 4);

    Some(PositionKey(decoded))
}

// ---------------------------------------------------------------------------
// Modern position key (7 × u32 — for eval bridge compatibility)
// ---------------------------------------------------------------------------

/// The "new" position key used internally by gnubg for evaluation.
///
/// This is a 7-element `u32` array where each element packs 8 board points
/// (4 bits each). It is the key used by the C neural-net evaluation bridge.
///
/// # Layout (from `positionid.c` `PositionKey()`)
///
/// - `data[0..2]` = player 1 (anBoard[1]), points 0..23 (8 per u32, 4 bits per point)
/// - `data[3..5]` = player 0 (anBoard[0]), points 0..23
/// - `data[6]`    = `[player0_bar(4 bits) | player1_bar(4 bits)]`
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct EvalPositionKey(pub [u32; 7]);

impl EvalPositionKey {
    /// Convert a `Board` to the modern 7-word evaluation key.
    ///
    /// This is the **exact** port of gnubg's `PositionKey()` in `positionid.c`.
    pub fn from_board(board: &Board) -> Self {
        let mut data = [0u32; 7];
        for i in 0..3 {
            let j = i * 8;
            data[i] = (board[1][j] & 0x0F)
                | ((board[1][j + 1] & 0x0F) << 4)
                | ((board[1][j + 2] & 0x0F) << 8)
                | ((board[1][j + 3] & 0x0F) << 12)
                | ((board[1][j + 4] & 0x0F) << 16)
                | ((board[1][j + 5] & 0x0F) << 20)
                | ((board[1][j + 6] & 0x0F) << 24)
                | ((board[1][j + 7] & 0x0F) << 28);
            data[i + 3] = (board[0][j] & 0x0F)
                | ((board[0][j + 1] & 0x0F) << 4)
                | ((board[0][j + 2] & 0x0F) << 8)
                | ((board[0][j + 3] & 0x0F) << 12)
                | ((board[0][j + 4] & 0x0F) << 16)
                | ((board[0][j + 5] & 0x0F) << 20)
                | ((board[0][j + 6] & 0x0F) << 24)
                | ((board[0][j + 7] & 0x0F) << 28);
        }
        data[6] = (board[0][24] & 0x0F) | ((board[1][24] & 0x0F) << 4);
        Self(data)
    }

    /// Convert a 7-word evaluation key back to a Board.
    ///
    /// This is the exact port of gnubg's `PositionFromKey()`.
    pub fn to_board(&self) -> Board {
        let mut board: Board = [[0u32; 25]; 2];
        for i in 0..3 {
            let j = i * 8;
            board[1][j] = self.0[i] & 0x0F;
            board[1][j + 1] = (self.0[i] >> 4) & 0x0F;
            board[1][j + 2] = (self.0[i] >> 8) & 0x0F;
            board[1][j + 3] = (self.0[i] >> 12) & 0x0F;
            board[1][j + 4] = (self.0[i] >> 16) & 0x0F;
            board[1][j + 5] = (self.0[i] >> 20) & 0x0F;
            board[1][j + 6] = (self.0[i] >> 24) & 0x0F;
            board[1][j + 7] = (self.0[i] >> 28) & 0x0F;

            board[0][j] = self.0[i + 3] & 0x0F;
            board[0][j + 1] = (self.0[i + 3] >> 4) & 0x0F;
            board[0][j + 2] = (self.0[i + 3] >> 8) & 0x0F;
            board[0][j + 3] = (self.0[i + 3] >> 12) & 0x0F;
            board[0][j + 4] = (self.0[i + 3] >> 16) & 0x0F;
            board[0][j + 5] = (self.0[i + 3] >> 20) & 0x0F;
            board[0][j + 6] = (self.0[i + 3] >> 24) & 0x0F;
            board[0][j + 7] = (self.0[i + 3] >> 28) & 0x0F;
        }
        board[0][24] = self.0[6] & 0x0F;
        board[1][24] = (self.0[6] >> 4) & 0x0F;
        board
    }
}

// ---------------------------------------------------------------------------
// PositionFromXG — parse XG format position string
// ---------------------------------------------------------------------------

/// Parse an XG-format position description (26 characters) into a Board.
///
/// XG format uses letters A-P (player) and a-p (opponent) for each of the
/// 26 positions (0=bar of player, 1..24=points, 25=bar of opponent).
///
/// Returns `None` if the string is not valid XG format.
pub fn board_from_xg(desc: &str) -> Option<Board> {
    if desc.len() < 26 {
        return None;
    }
    let bytes = desc.as_bytes();
    let mut board: Board = [[0u32; 25]; 2];

    for i in 0..26 {
        let (p0, p1): (Option<usize>, Option<usize>) = if i == 0 {
            // i=0: player's bar = point 24 for player 0
            (Some(24), None) // player0 bar = board[0][24]
        } else if i == 25 {
            (None, Some(24)) // opponent bar = board[1][24]
        } else {
            // i=1..24: board points
            // XG has point i as point (24-i) for player, (i-1) for opponent
            (Some(24 - i), Some(i - 1))
        };

        match bytes[i] {
            b'A'..=b'P' => {
                let count = (bytes[i] - b'A' + 1) as u32;
                if let Some(p) = p0 {
                    board[0][p] = count;
                }
                if let Some(p) = p1 {
                    board[1][p] = 0;
                }
            }
            b'a'..=b'p' => {
                let count = (bytes[i] - b'a' + 1) as u32;
                if let Some(p) = p0 {
                    board[0][p] = 0;
                }
                if let Some(p) = p1 {
                    board[1][p] = count;
                }
            }
            b'-' => {
                if let Some(p) = p0 {
                    board[0][p] = 0;
                }
                if let Some(p) = p1 {
                    board[1][p] = 0;
                }
            }
            _ => return None,
        }
    }

    Some(board)
}

// ---------------------------------------------------------------------------
// MatchID encoding/decoding
// ---------------------------------------------------------------------------

/// Length of a MatchID string.
pub const MATCH_ID_LEN: usize = 12;

/// Encode match state to a 12-character MatchID string.
pub fn match_id_from_state(state: &MatchState) -> String {
    let mut key = [0u8; 9];
    set_bits(&mut key, 0, 4, log_cube(state.n_cube) as i32);
    set_bits(&mut key, 4, 2, state.f_cube_owner as i32 & 0x3);
    set_bits(&mut key, 6, 1, state.f_move as i32);
    set_bits(&mut key, 7, 1, state.f_crawford as i32);
    set_bits(&mut key, 8, 3, state.gs as i32);
    set_bits(&mut key, 11, 1, state.f_turn as i32);
    set_bits(&mut key, 12, 1, state.f_doubled as i32);
    set_bits(&mut key, 13, 2, state.f_resigned as i32);
    // Dice: store higher die first
    let (d0, d1) = state.dice;
    set_bits(&mut key, 15, 3, core::cmp::max(d0, d1) as i32 & 0x7);
    set_bits(&mut key, 18, 3, core::cmp::min(d0, d1) as i32 & 0x7);
    set_bits(&mut key, 21, 15, state.n_match_to as i32 & 0x7FFF);
    set_bits(&mut key, 36, 15, state.an_score[0] as i32 & 0x7FFF);
    set_bits(&mut key, 51, 15, state.an_score[1] as i32 & 0x7FFF);
    set_bits(&mut key, 66, 1, (!state.f_jacoby) as i32);

    match_id_from_key(&key)
}

fn match_id_from_key(key: &[u8; 9]) -> String {
    let mut out = [0u8; MATCH_ID_LEN];
    for i in 0..3 {
        let chunk = i * 3;
        out[i * 4] = base64_encode(key[chunk] >> 2);
        out[i * 4 + 1] = base64_encode(((key[chunk] & 0x03) << 4) | (key[chunk + 1] >> 4));
        out[i * 4 + 2] =
            base64_encode(((key[chunk + 1] & 0x0F) << 2) | (key[chunk + 2] >> 6));
        out[i * 4 + 3] = base64_encode(key[chunk + 2] & 0x3F);
    }
    core::str::from_utf8(&out)
        .expect("MatchID base64 always produces valid ASCII")
        .to_string()
}

/// Decode a 12-character MatchID string into a MatchState.
pub fn match_state_from_id(s: &str, board: &Board) -> Option<MatchState> {
    if s.len() != MATCH_ID_LEN {
        return None;
    }

    let bytes = s.as_bytes();
    let mut vals = [0u8; 12];
    for (i, &ch) in bytes.iter().enumerate() {
        vals[i] = base64_decode(ch)?;
    }

    let mut key = [0u8; 9];
    for i in 0..3 {
        let src = i * 4;
        key[i * 3] = (vals[src] << 2) | (vals[src + 1] >> 4);
        key[i * 3 + 1] = (vals[src + 1] << 4) | (vals[src + 2] >> 2);
        key[i * 3 + 2] = (vals[src + 2] << 6) | vals[src + 3];
    }

    Some(match_state_from_key(&key, board))
}

fn match_state_from_key(key: &[u8; 9], board: &Board) -> MatchState {
    let n_cube_log = get_bits(key, 0, 4);
    let n_cube = 1 << n_cube_log;

    let mut f_cube_owner = get_bits(key, 4, 2) as i32;
    if f_cube_owner != 0 && f_cube_owner != 1 {
        f_cube_owner = -1;
    }

    let f_move = get_bits(key, 6, 1) != 0;
    let f_crawford = get_bits(key, 7, 1) != 0;
    let gs = match get_bits(key, 8, 3) {
        0 => GameState::None,
        1 => GameState::Playing,
        2 => GameState::Over,
        3 => GameState::Resigned,
        4 => GameState::Drop,
        _ => GameState::None,
    };
    let f_turn = get_bits(key, 11, 1) != 0;
    let f_doubled = get_bits(key, 12, 1) != 0;
    let f_resigned = get_bits(key, 13, 2) as i32;

    let d0 = get_bits(key, 15, 3) as u8;
    let d1 = get_bits(key, 18, 3) as u8;

    let n_match_to = get_bits(key, 21, 15) as i32;
    let an_score0 = get_bits(key, 36, 15) as i32;
    let an_score1 = get_bits(key, 51, 15) as i32;
    let f_jacoby = get_bits(key, 66, 1) == 0;

    MatchState {
        board: *board,
        dice: (d0, d1),
        f_turn,
        f_resigned,
        f_resignation_declined: false,
        f_doubled,
        c_games: 0,
        f_move,
        f_cube_owner,
        f_crawford,
        f_post_crawford: false,
        n_match_to,
        an_score: [an_score0, an_score1],
        n_cube,
        c_beavers: 0,
        bgv: Variation::Standard,
        f_cube_use: true,
        f_jacoby,
        gs,
    }
}

fn log_cube(mut n: u32) -> u32 {
    let mut i = 0;
    while n > 1 {
        n >>= 1;
        i += 1;
    }
    i
}

fn set_bits(pc: &mut [u8], bit_pos: u32, n_bits: u32, content: i32) {
    for i in 0..n_bits {
        let k = (bit_pos + i) as usize / 8;
        let r = (bit_pos + i) % 8;
        let bit = if (content >> (i as i32)) & 1 != 0 {
            1u8 << r
        } else {
            0
        };
        if k < pc.len() {
            pc[k] = (pc[k] & !(1u8 << r)) | bit;
        }
    }
}

fn get_bits(pc: &[u8], bit_pos: u32, n_bits: u32) -> u32 {
    let mut result = 0u32;
    for i in 0..n_bits {
        let k = (bit_pos + i) as usize / 8;
        let r = (bit_pos + i) % 8;
        if k < pc.len() && (pc[k] >> r) & 1 != 0 {
            result |= 1 << i;
        }
    }
    result
}

// ---------------------------------------------------------------------------
// CheckPosition — validate a board
// ---------------------------------------------------------------------------

/// Check whether a board represents a legal backgammon position.
///
/// Port of gnubg's `CheckPosition()`. Returns `true` if the position is valid.
pub fn check_position(board: &Board) -> bool {
    let mut ac = [0u32; 2];

    // Each player must have ≤ 15 checkers
    for i in 0..25 {
        ac[0] += board[0][i];
        ac[1] += board[1][i];
        if ac[0] > MAX_CHECKERS || ac[1] > MAX_CHECKERS {
            return false;
        }
    }

    // Both players cannot have checkers on the same point (mirrored)
    for i in 0..24 {
        if board[0][i] > 0 && board[1][24 - i] > 0 {
            return false;
        }
    }

    // Both players on bar against closed boards
    for i in 0..6 {
        if board[0][i] < 2 || board[1][i] < 2 {
            return true;
        }
    }

    if board[0][24] == 0 || board[1][24] == 0 {
        return true;
    }

    false
}

// ---------------------------------------------------------------------------
// PipCount
// ---------------------------------------------------------------------------

/// Compute pip count for both players.
///
/// - Player 1 (current player): distance to home = point number
/// - Player 0 (opponent): distance to their home = point number (indices
///   are the opponent's own point numbers in this representation)
pub fn pip_count(board: &Board) -> [u32; 2] {
    let mut pips = [0u32; 2];
    for point in 1..25 {
        pips[0] += board[0][point] * point as u32;
        pips[1] += board[1][point] * point as u32;
    }
    pips
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Standard opening position (gnubg convention).
    fn opening_board() -> Board {
        let mut b = [[0u32; 25]; 2];
        // Current player (anBoard[1]): 2@24, 5@13, 3@8, 5@6
        b[1][24] = 2; b[1][13] = 5; b[1][8] = 3; b[1][6] = 5;
        // Opponent (anBoard[0]): 2@1, 5@12, 3@7, 5@6
        b[0][1] = 2; b[0][12] = 5; b[0][7] = 3; b[0][6] = 5;
        b
    }

    #[test]
    fn test_board_roundtrip_via_position_id() {
        let board = opening_board();
        let id = position_id_from_board(&board);
        assert_eq!(id.len(), POSITION_ID_LEN);
        let decoded = board_from_position_id(&id).expect("should decode");
        assert_eq!(decoded, board);
    }

    #[test]
    fn test_board_roundtrip_via_eval_key() {
        let board = opening_board();
        let ek = EvalPositionKey::from_board(&board);
        let decoded = ek.to_board();
        assert_eq!(decoded, board);
    }

    #[test]
    fn test_old_position_key_roundtrip() {
        let board = opening_board();
        let key = old_position_key(&board);
        let decoded = board_from_old_key(&key);
        assert_eq!(decoded, board);
    }

    #[test]
    fn test_check_position_valid() {
        let board = opening_board();
        assert!(check_position(&board));
    }

    #[test]
    fn test_pip_count_opening() {
        let board = opening_board();
        let pips = pip_count(&board);
        let total0: u32 = board[0].iter().sum();
        let total1: u32 = board[1].iter().sum();
        assert_eq!(total0, 15, "player 0 should have 15 checkers");
        assert_eq!(total1, 15, "player 1 should have 15 checkers");
        assert_eq!(pips[0], 113, "pip count player 0 (opponent from player perspective)");
        assert_eq!(pips[1], 167, "pip count player 1 (current player)");
    }

    #[test]
    fn test_board_from_xg_roundtrip() {
        // XG format uses A-P for player (board[0]), a-p for opponent (board[1])
        // Position 2 = board[1][1] for lowercase 'a'
        let xg = "--a-----------------------";
        let board = board_from_xg(xg).expect("valid XG");
        assert!(check_position(&board));
        // Player (board[1]) has 1 checker at point 1
        assert_eq!(board[1][1], 1);
    }

    #[test]
    fn test_position_id_encoding_format() {
        let board = opening_board();
        let id = position_id_from_board(&board);
        assert_eq!(id.len(), 14);
        for &ch in id.as_bytes() {
            assert!(
                ch.is_ascii_alphanumeric() || ch == b'+' || ch == b'/',
                "Invalid base64 char: {}",
                ch as char
            );
        }
        // Roundtrip: encode → decode → same board (no hardcoded string comparison)
        let decoded = board_from_position_id(&id).expect("should decode");
        assert_eq!(decoded, board);
    }

    #[test]
    fn test_match_id_roundtrip() {
        let board = opening_board();
        let state = MatchState {
            board,
            dice: (3, 1),
            f_turn: true,
            f_resigned: 0,
            f_resignation_declined: false,
            f_doubled: false,
            c_games: 0,
            f_move: true,
            f_cube_owner: -1,
            f_crawford: false,
            f_post_crawford: false,
            n_match_to: 0,
            an_score: [0, 0],
            n_cube: 1,
            c_beavers: 0,
            bgv: Variation::Standard,
            f_cube_use: true,
            f_jacoby: true,
            gs: GameState::Playing,
        };

        let mid = match_id_from_state(&state);
        assert_eq!(mid.len(), MATCH_ID_LEN);

        let decoded = match_state_from_id(&mid, &board).expect("should decode");
        assert_eq!(decoded.dice, state.dice);
        assert_eq!(decoded.f_move, state.f_move);
        assert_eq!(decoded.f_crawford, state.f_crawford);
        assert_eq!(decoded.n_match_to, state.n_match_to);
        assert_eq!(decoded.an_score, state.an_score);
        assert_eq!(decoded.n_cube, state.n_cube);
        assert_eq!(decoded.f_jacoby, state.f_jacoby);
        assert_eq!(decoded.gs, state.gs);
    }

    #[test]
    fn test_empty_position_id_is_rejected() {
        assert!(board_from_position_id("").is_none());
        assert!(old_key_from_position_id("").is_none());
        assert!(old_key_from_position_id("short").is_none());
    }

    #[test]
    fn test_variation_enum() {
        assert_ne!(Variation::Standard as i32, Variation::Nackgammon as i32);
        assert_ne!(Variation::Hypergammon1 as i32, Variation::Hypergammon2 as i32);
    }

    #[test]
    fn test_game_state_enum() {
        assert_eq!(GameState::None as i32, 0);
        assert_eq!(GameState::Playing as i32, 1);
    }

    #[test]
    fn test_move_list() {
        let mut ml = MoveList::with_capacity(10);
        assert!(ml.is_empty());

        let mv = Move {
            from_to: [Some((6, 5)), Some((3, 1)), None, None],
            c_moves: 2,
            c_pips: 5,
            key: PositionKey([0; 10]),
        };
        ml.push(mv);
        assert_eq!(ml.len(), 1);
        assert!(!ml.is_empty());

        let ml2 = ml.clone();
        assert_eq!(ml.len(), ml2.len());
    }

    #[test]
    fn test_move_display() {
        let mv = Move {
            from_to: [Some((6, 5)), Some((3, 1)), None, None],
            c_moves: 2,
            c_pips: 5,
            key: PositionKey([0; 10]),
        };
        let display = format!("{mv}");
        assert!(display.contains("6->5"));
        assert!(display.contains("3->1"));
    }
}
