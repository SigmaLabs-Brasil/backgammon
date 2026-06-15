//! Parallel root evaluation, alpha-beta search, and lock-free per-thread eval cache.

pub mod search;
pub mod transposition;

pub use search::*;
pub use transposition::*;

use gnubg_sys::{decode_position_id, evaluate_position_key, GnuBgError, PositionKey, RawEval};
use rayon::prelude::*;
use std::cell::RefCell;
use std::fmt;

pub const DEFAULT_CACHE_ENTRIES: usize = 1 << 19;
pub const ROOT_PARALLEL_THRESHOLD: usize = 4;

thread_local! {
    static EVAL_CACHE: RefCell<EvalCache> = RefCell::new(EvalCache::new(DEFAULT_CACHE_ENTRIES));
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Board {
    key: PositionKey,
}

impl Board {
    pub fn from_position_id(position_id: &str) -> Result<Self, SearchError> {
        Ok(Self {
            key: decode_position_id(position_id)?,
        })
    }

    pub const fn from_key(key: PositionKey) -> Self {
        Self { key }
    }

    pub const fn key(&self) -> PositionKey {
        self.key
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Move {
    pub id: usize,
    pub dice: (u8, u8),
    pub from: u8,
    pub to: u8,
    pub steps: [Option<(u8, u8)>; 4],
    pub resulting_position: PositionKey,
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let steps: Vec<String> = self
            .steps
            .iter()
            .flatten()
            .map(|(from, to)| {
                if *to == 0 {
                    format!("{from}/off")
                } else {
                    format!("{from}/{to}")
                }
            })
            .collect();
        write!(
            f,
            "#{} {}{} {}",
            self.id,
            self.dice.0,
            self.dice.1,
            steps.join(" ")
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EvalResult {
    pub win: f32,
    pub win_gammon: f32,
    pub win_backgammon: f32,
    pub lose_gammon: f32,
    pub lose_backgammon: f32,
    pub equity: f32,
    pub depth: u8,
    pub cache_hit: bool,
}

impl EvalResult {
    fn from_raw(raw: RawEval, depth: u8, cache_hit: bool) -> Self {
        let [win, win_gammon, win_backgammon, lose_gammon, lose_backgammon] = raw.outputs;
        let equity =
            (2.0 * win - 1.0) + win_gammon + win_backgammon - lose_gammon - lose_backgammon;
        Self {
            win,
            win_gammon,
            win_backgammon,
            lose_gammon,
            lose_backgammon,
            equity,
            depth,
            cache_hit,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CacheStats {
    pub entries: usize,
    pub lookups: u64,
    pub hits: u64,
    pub inserts: u64,
}

#[derive(Clone, Copy, Debug)]
struct EvalCacheEntry {
    key: PositionKey,
    eval: EvalResult,
    depth: u8,
}

#[derive(Debug)]
pub struct EvalCache {
    entries: Vec<Option<EvalCacheEntry>>,
    mask: usize,
    lookups: u64,
    hits: u64,
    inserts: u64,
}

impl EvalCache {
    pub fn new(capacity: usize) -> Self {
        let entries = capacity.next_power_of_two().max(2);
        let mut slots = Vec::with_capacity(entries);
        slots.resize_with(entries, || None);
        Self {
            entries: slots,
            mask: entries - 1,
            lookups: 0,
            hits: 0,
            inserts: 0,
        }
    }

    pub fn lookup(&mut self, key: &PositionKey, depth: u8) -> Option<EvalResult> {
        self.lookups += 1;
        let index = self.index(key);
        let entry = self.entries[index]?;
        if entry.key == *key && entry.depth >= depth {
            self.hits += 1;
            Some(EvalResult {
                cache_hit: true,
                ..entry.eval
            })
        } else {
            None
        }
    }

    pub fn insert(&mut self, key: PositionKey, depth: u8, mut eval: EvalResult) {
        let index = self.index(&key);
        let replace = self.entries[index].map_or(true, |entry| depth >= entry.depth);
        if replace {
            eval.cache_hit = false;
            self.entries[index] = Some(EvalCacheEntry { key, eval, depth });
            self.inserts += 1;
        }
    }

    pub fn stats(&self) -> CacheStats {
        CacheStats {
            entries: self.entries.len(),
            lookups: self.lookups,
            hits: self.hits,
            inserts: self.inserts,
        }
    }

    fn index(&self, key: &PositionKey) -> usize {
        (hash_position_key(key) as usize) & self.mask
    }
}

#[derive(Debug)]
pub enum SearchError {
    Ffi(GnuBgError),
    EmptyMoveList,
}

impl fmt::Display for SearchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ffi(err) => write!(f, "{err}"),
            Self::EmptyMoveList => f.write_str("no candidate moves to evaluate"),
        }
    }
}

impl std::error::Error for SearchError {}

impl From<GnuBgError> for SearchError {
    fn from(value: GnuBgError) -> Self {
        Self::Ffi(value)
    }
}

pub fn evaluate_board(board: &Board, depth: u8) -> Result<EvalResult, SearchError> {
    evaluate_key_with_thread_cache(board.key(), depth)
}

pub fn evaluate_key_with_thread_cache(
    key: PositionKey,
    depth: u8,
) -> Result<EvalResult, SearchError> {
    EVAL_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if let Some(hit) = cache.lookup(&key, depth) {
            return Ok(hit);
        }
        let raw = evaluate_position_key(&key)?;
        let eval = EvalResult::from_raw(raw, depth, false);
        cache.insert(key, depth, eval);
        Ok(eval)
    })
}

pub fn thread_cache_stats() -> CacheStats {
    EVAL_CACHE.with(|cache| cache.borrow().stats())
}

pub fn generate_candidate_moves(board: &Board, dice: (u8, u8)) -> Vec<Move> {
    // Convert gnubg_sys::PositionKey -> gnubg_types::PositionKey -> raw Board
    let gt_key = gnubg_types::PositionKey::from_raw(board.key().0);
    let raw_board = gnubg_types::board_from_old_key(&gt_key);

    let move_list = gnubg_moves::generate_moves(&raw_board, dice);

    move_list
        .moves
        .iter()
        .enumerate()
        .map(|(idx, mv)| {
            let (from, to) = mv.from_to.iter().find_map(|&st| st).unwrap_or((0, 0));
            Move {
                id: idx,
                dice,
                from,
                to,
                steps: mv.from_to,
                resulting_position: PositionKey(mv.key.0),
            }
        })
        .collect()
}

pub fn raw_board(board: &Board) -> gnubg_types::Board {
    let gt_key = gnubg_types::PositionKey::from_raw(board.key().0);
    gnubg_types::board_from_old_key(&gt_key)
}

pub fn parallel_eval_root(
    _board: &Board,
    moves: &[Move],
    depth: u8,
) -> Result<Vec<(Move, EvalResult)>, SearchError> {
    if moves.is_empty() {
        return Err(SearchError::EmptyMoveList);
    }

    if moves.len() < ROOT_PARALLEL_THRESHOLD {
        moves
            .iter()
            .map(|mv| {
                evaluate_key_with_thread_cache(mv.resulting_position, depth)
                    .map(|eval| (mv.clone(), eval))
            })
            .collect()
    } else {
        moves
            .par_iter()
            .cloned()
            .map(|mv| {
                evaluate_key_with_thread_cache(mv.resulting_position, depth).map(|eval| (mv, eval))
            })
            .collect()
    }
}

pub fn best_move(
    board: &Board,
    dice: (u8, u8),
    depth: u8,
) -> Result<(Move, EvalResult), SearchError> {
    let moves = generate_candidate_moves(board, dice);
    parallel_eval_root(board, &moves, depth)?
        .into_iter()
        .max_by(|(_, a), (_, b)| a.equity.total_cmp(&b.equity))
        .ok_or(SearchError::EmptyMoveList)
}

fn hash_position_key(key: &PositionKey) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in key.0 {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xff51afd7ed558ccd);
    hash ^ (hash >> 33)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_hits_on_second_eval() {
        let board = Board::from_position_id("4HPwATDgc/ABMA").expect("valid board");
        let first = evaluate_board(&board, 0).expect("first eval");
        let second = evaluate_board(&board, 0).expect("second eval");
        assert!(!first.cache_hit);
        assert!(second.cache_hit);
    }

    #[test]
    fn root_eval_returns_all_candidates() {
        let board = Board::from_position_id("4HPwATDgc/ABMA").expect("valid board");
        let moves = generate_candidate_moves(&board, (3, 1));
        let evaluated = parallel_eval_root(&board, &moves, 0).expect("root eval");
        assert_eq!(evaluated.len(), moves.len());
    }

    #[test]
    fn best_move_selects_max_equity() {
        let board = Board::from_position_id("4HPwATDgc/ABMA").expect("valid board");
        let (best, best_eval) = best_move(&board, (6, 6), 0).expect("best move");
        let moves = generate_candidate_moves(&board, (6, 6));
        let evaluated = parallel_eval_root(&board, &moves, 0).expect("root eval");
        let max_equity = evaluated
            .iter()
            .map(|(_, eval)| eval.equity)
            .fold(f32::NEG_INFINITY, f32::max);
        assert_eq!(best.dice, (6, 6));
        assert!((best_eval.equity - max_equity).abs() < f32::EPSILON);
        assert!(best.id < moves.len());
    }
}
