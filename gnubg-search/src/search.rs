use crate::transposition::{zobrist_hash, TTFlag, TranspositionTable};
use crate::{
    best_move, evaluate_key_with_thread_cache, generate_candidate_moves, Board, Move, SearchError,
};
use gnubg_eval::cubeful::{cubeful_equity, CubeOwner, CubeState};
use gnubg_eval::met::mwc_after;
use gnubg_types::Dice;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SearchConfig {
    pub max_depth: u8,
    pub time_limit_ms: u64,
    pub tt_size: usize,
    pub randomize: bool,
    pub cube_state: Option<CubeState>,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            max_depth: 4,
            time_limit_ms: 0,
            tt_size: 1 << 20,
            randomize: false,
            cube_state: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SearchStats {
    pub nodes_searched: u64,
    pub tt_hits: u64,
    pub tt_lookups: u64,
    pub eval_calls: u64,
    pub time_ms: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CubeAction { NoDouble, DoubleTake, DoubleDrop }

#[derive(Clone, Debug, PartialEq)]
pub struct CubeDecision {
    pub action: CubeAction,
    pub no_double_equity: f32,
    pub double_take_equity: f32,
    pub double_drop_equity: f32,
    pub gain: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MoveEvaluation {
    pub mv: Move,
    pub equity: f32,
    pub depth_searched: u8,
    pub pv: Vec<Move>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SearchResult {
    pub evaluations: Vec<MoveEvaluation>,
    pub best_move: Move,
    pub best_equity: f32,
    pub stats: SearchStats,
    pub config: SearchConfig,
    pub cube_decision: Option<CubeDecision>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AnalyzeRoll {
    pub dice: Dice,
    pub best_move: Move,
    pub equity: f32,
    pub pv: Vec<Move>,
    pub depth_searched: u8,
    pub cube_decision: Option<CubeDecision>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AnalyzeResult {
    pub rolls: Vec<AnalyzeRoll>,
    pub stats: SearchStats,
    pub config: SearchConfig,
}

struct SearchContext {
    config: SearchConfig,
    start: Instant,
    limit: Option<Duration>,
    tt: TranspositionTable,
    stats: SearchStats,
    stopped: bool,
}

impl SearchContext {
    fn new(config: SearchConfig) -> Self {
        let limit = (config.time_limit_ms > 0).then(|| Duration::from_millis(config.time_limit_ms));
        Self {
            config,
            start: Instant::now(),
            limit,
            tt: TranspositionTable::new(config.tt_size),
            stats: SearchStats::default(),
            stopped: false,
        }
    }

    fn should_stop(&mut self) -> bool {
        if self.stopped {
            return true;
        }
        if self.stats.nodes_searched & 0x3ff == 0 {
            if let Some(limit) = self.limit {
                self.stopped = self.start.elapsed() >= limit;
            }
        }
        self.stopped
    }

    fn finish_stats(&mut self) -> SearchStats {
        let (tt_hits, tt_lookups) = self.tt.stats();
        self.stats.tt_hits = tt_hits;
        self.stats.tt_lookups = tt_lookups;
        self.stats.time_ms = self.start.elapsed().as_millis() as u64;
        self.stats
    }
}

pub fn search_position(
    board: &Board,
    dice: Dice,
    config: &SearchConfig,
) -> Result<SearchResult, SearchError> {
    let normalized = normalize_config(*config);
    if normalized.max_depth == 0 {
        return depth_zero_search(board, dice, normalized);
    }

    let mut ctx = SearchContext::new(normalized);
    let mut best_completed = depth_zero_search(board, dice, normalized)?;
    best_completed.config = normalized;

    for depth in 1..=normalized.max_depth {
        ctx.tt.new_search();
        let evaluations = search_root(board, dice, depth, &mut ctx)?;
        if evaluations.is_empty() {
            break;
        }
        let best = evaluations[0].clone();
        best_completed = SearchResult {
            evaluations,
            best_move: best.mv,
            best_equity: best.equity,
            stats: ctx.finish_stats(),
            config: normalized,
            cube_decision: None,
        };
        if ctx.stopped {
            break;
        }
    }

    best_completed.stats = ctx.finish_stats();

    // Compute cube decision if we have a centered cube
    if let Some(cube) = &config.cube_state {
        if cube.owner == CubeOwner::Center {
            best_completed.cube_decision = Some(compute_cube_decision(board, cube)?);
        }
    }

    Ok(best_completed)
}

pub fn analyze_position(
    board: &Board,
    config: &SearchConfig,
) -> Result<AnalyzeResult, SearchError> {
    let normalized = normalize_config(*config);
    let mut rolls = Vec::with_capacity(21);
    let mut totals = SearchStats::default();

    for dice in all_rolls() {
        let result = search_position(board, dice, &normalized)?;
        totals.nodes_searched += result.stats.nodes_searched;
        totals.tt_hits += result.stats.tt_hits;
        totals.tt_lookups += result.stats.tt_lookups;
        totals.eval_calls += result.stats.eval_calls;
        totals.time_ms += result.stats.time_ms;

        // Per-roll cube decision: evaluate resulting position with the cube state
        let cube_decision = if let Some(cube) = &normalized.cube_state {
            if cube.owner == CubeOwner::Center {
                let resulting_board = Board::from_key(result.best_move.resulting_position);
                Some(compute_cube_decision(&resulting_board, cube)?)
            } else {
                None
            }
        } else {
            None
        };

        rolls.push(AnalyzeRoll {
            dice,
            best_move: result.best_move,
            equity: result.best_equity,
            pv: result
                .evaluations
                .first()
                .map(|evaluation| evaluation.pv.clone())
                .unwrap_or_default(),
            depth_searched: result
                .evaluations
                .first()
                .map_or(normalized.max_depth, |evaluation| evaluation.depth_searched),
            cube_decision,
        });
    }

    Ok(AnalyzeResult {
        rolls,
        stats: totals,
        config: normalized,
    })
}

fn depth_zero_search(
    board: &Board,
    dice: Dice,
    config: SearchConfig,
) -> Result<SearchResult, SearchError> {
    let moves = generate_candidate_moves(board, dice);
    if moves.is_empty() {
        return Err(SearchError::EmptyMoveList);
    }
    let (best, best_eval) = best_move(board, dice, 0)?;
    let best_equity = match &config.cube_state {
        Some(cube) => {
            let outputs = [best_eval.win, best_eval.win_gammon, best_eval.win_backgammon,
                           best_eval.lose_gammon, best_eval.lose_backgammon];
            cubeful_equity(&outputs, cube)
        }
        None => best_eval.equity,
    };
    let mut evaluations = Vec::with_capacity(moves.len());
    for mv in moves {
        let eval = evaluate_key_with_thread_cache(mv.resulting_position, 0)?;
        let equity = match &config.cube_state {
            Some(cube) => {
                let outputs = [eval.win, eval.win_gammon, eval.win_backgammon,
                               eval.lose_gammon, eval.lose_backgammon];
                cubeful_equity(&outputs, cube)
            }
            None => eval.equity,
        };
        evaluations.push(MoveEvaluation {
            mv: mv.clone(),
            equity,
            depth_searched: 0,
            pv: vec![mv],
        });
    }
    evaluations.sort_by(|a, b| {
        b.equity
            .total_cmp(&a.equity)
            .then_with(|| a.mv.id.cmp(&b.mv.id))
    });
    Ok(SearchResult {
        evaluations,
        best_move: best,
        best_equity,
        stats: SearchStats {
            nodes_searched: 0,
            tt_hits: 0,
            tt_lookups: 0,
            eval_calls: 0,
            time_ms: 0,
        },
        config,
        cube_decision: None,
    })
}

fn search_root(
    board: &Board,
    dice: Dice,
    depth: u8,
    ctx: &mut SearchContext,
) -> Result<Vec<MoveEvaluation>, SearchError> {
    let moves = generate_candidate_moves(board, dice);
    if moves.is_empty() {
        return Err(SearchError::EmptyMoveList);
    }

    let mut previous: Vec<(usize, f32)> = Vec::with_capacity(moves.len());
    let mut evaluations = Vec::with_capacity(moves.len());
    let mut alpha = f32::NEG_INFINITY;
    let beta = f32::INFINITY;

    for mv in order_moves(moves, None, &previous) {
        if ctx.should_stop() {
            break;
        }
        let mut pv = vec![mv.clone()];
        let equity = if depth == 0 {
            leaf_eval(mv.resulting_position, ctx)?
        } else {
            alpha_beta(
                mv.resulting_position,
                depth.saturating_sub(1),
                1,
                alpha,
                beta,
                ctx,
                &mut pv,
            )?
        };
        alpha = alpha.max(equity);
        previous.push((mv.id, equity));
        evaluations.push(MoveEvaluation {
            mv,
            equity,
            depth_searched: depth,
            pv,
        });
    }

    evaluations.sort_by(|a, b| {
        b.equity
            .total_cmp(&a.equity)
            .then_with(|| a.mv.id.cmp(&b.mv.id))
    });
    Ok(evaluations)
}

fn alpha_beta(
    key: gnubg_sys::PositionKey,
    depth: u8,
    ply: u8,
    mut alpha: f32,
    beta: f32,
    ctx: &mut SearchContext,
    pv: &mut Vec<Move>,
) -> Result<f32, SearchError> {
    ctx.stats.nodes_searched += 1;
    if depth == 0 || ctx.should_stop() || is_game_over(key) {
        return leaf_eval(key, ctx);
    }

    let hash = zobrist_hash(&key);
    if let Some((score, flag, best_idx)) = ctx.tt.lookup(hash, depth, ply) {
        match flag {
            TTFlag::Exact => return Ok(score),
            TTFlag::LowerBound if score >= beta => return Ok(score),
            TTFlag::UpperBound if score <= alpha => return Ok(score),
            _ => {
                let _ = best_idx;
            }
        }
    }

    let board = Board::from_key(key);
    let original_alpha = alpha;
    let mut best_score = f32::NEG_INFINITY;
    let mut best_idx = 0_u16;
    let mut best_line = Vec::new();
    let mut found_child = false;

    for dice in all_rolls() {
        let moves = generate_candidate_moves(&board, dice);
        if moves.is_empty() {
            continue;
        }
        found_child = true;
        for mv in order_moves(moves, None, &[]) {
            if ctx.should_stop() {
                break;
            }
            let mut child_line = vec![mv.clone()];
            let score = -alpha_beta(
                mv.resulting_position,
                depth.saturating_sub(1),
                ply.saturating_add(1),
                -beta,
                -alpha,
                ctx,
                &mut child_line,
            )?;
            if score > best_score {
                best_score = score;
                best_idx = mv.id.min(u16::MAX as usize) as u16;
                best_line = child_line;
            }
            alpha = alpha.max(score);
            if alpha >= beta {
                break;
            }
        }
        if alpha >= beta || ctx.stopped {
            break;
        }
    }

    if !found_child {
        return leaf_eval(key, ctx);
    }

    if !best_line.is_empty() {
        pv.extend(best_line);
    }
    let flag = if best_score <= original_alpha {
        TTFlag::UpperBound
    } else if best_score >= beta {
        TTFlag::LowerBound
    } else {
        TTFlag::Exact
    };
    ctx.tt.store(hash, depth, flag, best_score, best_idx, ply);
    Ok(best_score)
}

fn leaf_eval(key: gnubg_sys::PositionKey, ctx: &mut SearchContext) -> Result<f32, SearchError> {
    ctx.stats.eval_calls += 1;
    let eval = evaluate_key_with_thread_cache(key, ctx.config.max_depth)?;
    let score = match &ctx.config.cube_state {
        Some(cube) => {
            let outputs = [eval.win, eval.win_gammon, eval.win_backgammon,
                           eval.lose_gammon, eval.lose_backgammon];
            cubeful_equity(&outputs, cube)
        }
        None => eval.equity,
    };
    if ctx.config.randomize {
        Ok(score + deterministic_jitter(key))
    } else {
        Ok(score)
    }
}

fn order_moves(
    mut moves: Vec<Move>,
    tt_best_idx: Option<u16>,
    previous: &[(usize, f32)],
) -> Vec<Move> {
    moves.sort_by(|a, b| {
        let a_tt = tt_best_idx.is_some_and(|idx| usize::from(idx) == a.id);
        let b_tt = tt_best_idx.is_some_and(|idx| usize::from(idx) == b.id);
        b_tt.cmp(&a_tt)
            .then_with(|| hit_score(b).cmp(&hit_score(a)))
            .then_with(|| previous_score(previous, b.id).total_cmp(&previous_score(previous, a.id)))
            .then_with(|| a.id.cmp(&b.id))
    });
    moves
}

fn hit_score(mv: &Move) -> u8 {
    mv.steps
        .iter()
        .flatten()
        .filter(|(from, to)| *to > 0 && *from > *to)
        .count()
        .min(u8::MAX as usize) as u8
}

fn previous_score(previous: &[(usize, f32)], id: usize) -> f32 {
    previous
        .iter()
        .find_map(|(move_id, score)| (*move_id == id).then_some(*score))
        .unwrap_or(f32::NEG_INFINITY)
}

fn is_game_over(key: gnubg_sys::PositionKey) -> bool {
    let gt_key = gnubg_types::PositionKey::from_raw(key.0);
    let board = gnubg_types::board_from_old_key(&gt_key);
    board[0][0] >= 15 || board[1][0] >= 15
}

fn all_rolls() -> impl Iterator<Item = Dice> {
    (1_u8..=6).flat_map(|first| (first..=6).map(move |second| (first, second)))
}

fn normalize_config(mut config: SearchConfig) -> SearchConfig {
    config.tt_size = config.tt_size.max(2).next_power_of_two();
    config
}

fn deterministic_jitter(key: gnubg_sys::PositionKey) -> f32 {
    let noise = (zobrist_hash(&key) & 0xffff) as f32 / 65_535.0;
    (noise - 0.5) * 0.002
}

pub fn compute_cube_decision(
    board: &Board,
    cube: &CubeState,
) -> Result<CubeDecision, SearchError> {
    let eval = evaluate_key_with_thread_cache(board.key(), 0)?;
    let outputs = [eval.win, eval.win_gammon, eval.win_backgammon,
                   eval.lose_gammon, eval.lose_backgammon];

    // No double: current cube state
    let no_double_equity = cubeful_equity(&outputs, cube);

    // Double/Take: opponent owns cube at double value
    let double_take_cube = CubeState {
        value: cube.value * 2,
        owner: CubeOwner::Opponent,
        ..*cube
    };
    let double_take_equity = cubeful_equity(&outputs, &double_take_cube);

    // Double/Drop: player wins cube.value points
    let double_drop_equity = match &cube.match_state {
        Some(ms) => mwc_after(ms, cube.value),
        None => cube.value as f32,
    };

    let double_equity = double_take_equity.min(double_drop_equity);
    let is_double = double_equity > no_double_equity;

    let action = if is_double {
        if (double_equity - double_take_equity).abs() < 1e-6 {
            CubeAction::DoubleTake
        } else {
            CubeAction::DoubleDrop
        }
    } else {
        CubeAction::NoDouble
    };

    let gain = double_equity - no_double_equity;

    Ok(CubeDecision {
        action,
        no_double_equity,
        double_take_equity,
        double_drop_equity,
        gain,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depth_zero_matches_best_move() {
        let board = Board::from_position_id("4HPwATDgc/ABMA").expect("valid board");
        let config = SearchConfig {
            max_depth: 0,
            ..SearchConfig::default()
        };
        let searched = search_position(&board, (3, 1), &config).expect("search");
        let (best, eval) = best_move(&board, (3, 1), 0).expect("best move");
        assert_eq!(searched.best_move, best);
        assert!((searched.best_equity - eval.equity).abs() < f32::EPSILON);
    }

    #[test]
    fn search_returns_sorted_pv_results() {
        let board = Board::from_position_id("4HPwATDgc/ABMA").expect("valid board");
        let config = SearchConfig {
            max_depth: 1,
            tt_size: 1 << 12,
            ..SearchConfig::default()
        };
        let result = search_position(&board, (6, 6), &config).expect("search");
        assert_eq!(result.best_move, result.evaluations[0].mv);
        assert!(!result.evaluations[0].pv.is_empty());
        assert!(result
            .evaluations
            .windows(2)
            .all(|window| window[0].equity >= window[1].equity));
    }

    #[test]
    fn analyze_returns_all_twenty_one_rolls() {
        let board = Board::from_position_id("4HPwATDgc/ABMA").expect("valid board");
        let config = SearchConfig {
            max_depth: 0,
            tt_size: 1 << 10,
            ..SearchConfig::default()
        };
        let result = analyze_position(&board, &config).expect("analyze");
        assert_eq!(result.rolls.len(), 21);
        assert_eq!(result.rolls[0].dice, (1, 1));
        assert_eq!(result.rolls[20].dice, (6, 6));
    }

    #[test]
    fn shallow_search_records_tt_activity() {
        let board = Board::from_position_id("4HPwATDgc/ABMA").expect("valid board");
        let config = SearchConfig {
            max_depth: 2,
            tt_size: 64,
            ..SearchConfig::default()
        };
        let result = search_position(&board, (3, 1), &config).expect("search");
        assert!(result.stats.tt_lookups > 0);
    }

    #[test]
    fn alpha_beta_depth_one_keeps_a_valid_best_move() {
        let board = Board::from_position_id("4HPwATDgc/ABMA").expect("valid board");
        let config = SearchConfig {
            max_depth: 1,
            ..SearchConfig::default()
        };
        let result = search_position(&board, (2, 1), &config).expect("search");
        let legal_moves = generate_candidate_moves(&board, (2, 1));
        assert!(legal_moves.iter().any(|mv| *mv == result.best_move));
    }
}
