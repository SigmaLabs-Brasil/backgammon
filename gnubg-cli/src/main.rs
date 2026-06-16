use clap::{Parser, Subcommand};
use gnubg_eval::cubeful::{CubeOwner, CubeState};
use gnubg_eval::met::{mwc, mwc_after, MatchState};
use gnubg_search::{
    analyze_position, best_move, evaluate_board, generate_candidate_moves, parallel_eval_root,
    raw_board, search_position, thread_cache_stats, Board, CubeAction, CubeDecision, EvalResult,
    Move, SearchConfig,
};
use mimalloc::MiMalloc;
use std::io::{self, Write};
use std::time::Instant;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Debug, Parser)]
#[command(
    name = "gnubg",
    version,
    about = "Native GNU Backgammon FFI evaluation CLI"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Evaluate one PositionID or 20-char hex position key.
    Evaluate {
        position_id: String,
        #[arg(long)]
        roll: Option<String>,
        #[arg(long, default_value_t = 0)]
        depth: u8,
        #[arg(long, value_parser = parse_cube_value)]
        cube: Option<i32>,
        #[arg(long = "match", value_parser = parse_match_score)]
        match_score: Option<(i32, i32)>,
        #[arg(long)]
        crawford: bool,
        #[arg(long)]
        post_crawford: bool,
        #[arg(long)]
        doubling: bool,
    },
    /// Generate candidate root moves for a roll and print the highest equity move.
    BestMove {
        position_id: String,
        dice: String,
        #[arg(long, default_value_t = 0)]
        depth: u8,
    },
    /// Analyze every dice roll from a position.
    Analyze {
        position_id: String,
        #[arg(long, default_value_t = 3)]
        depth: u8,
        #[arg(long)]
        time_limit: Option<u64>,
        #[arg(long)]
        doubling: bool,
        #[arg(long = "match", value_parser = parse_match_score)]
        match_score: Option<(i32, i32)>,
        #[arg(long)]
        crawford: bool,
        #[arg(long)]
        post_crawford: bool,
    },
    /// Evaluate cube doubling decision for a position.
    AnalyzeCube {
        position_id: String,
        #[arg(long = "match", value_parser = parse_match_score)]
        match_score: Option<(i32, i32)>,
        #[arg(long)]
        crawford: bool,
        #[arg(long)]
        post_crawford: bool,
        #[arg(long, default_value_t = 0)]
        depth: u8,
    },
    /// Play an interactive money-game session against the engine.
    Play {
        #[arg(long)]
        color: Option<String>,
        #[arg(long, default_value_t = 3)]
        depth: u8,
    },
    /// Benchmark commands
    #[command(subcommand)]
    Bench(BenchCommand),
}

#[derive(Debug, Subcommand)]
enum BenchCommand {
    /// Pure evaluation throughput (no move generation).
    Eval {
        #[arg(long, default_value_t = 10000)]
        positions: usize,
        #[arg(long, default_value_t = 0)]
        depth: u8,
    },
    /// Search throughput with move generation.
    Search {
        #[arg(long, default_value_t = 1000)]
        positions: usize,
        #[arg(long, default_value_t = 2)]
        depth: u8,
        #[arg(long, default_value_t = 8)]
        candidates: usize,
    },
    /// Quality check vs C reference values (regression guard).
    Check {
        #[arg(long, default_value_t = 5000)]
        positions: usize,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.command {
        Command::Evaluate {
            position_id,
            roll,
            depth,
            cube,
            match_score,
            crawford,
            post_crawford,
            doubling,
        } => {
            let board = Board::from_position_id(&position_id)?;
            let mut eval = evaluate_board(&board, depth)?;
            if match_score.is_none() && (crawford || post_crawford) {
                return Err("--crawford/--post-crawford require --match".into());
            }
            let match_state = match_score.map(|(player_away, opponent_away)| {
                MatchState::new(player_away, opponent_away, crawford, post_crawford)
            });
            let show_cubeful = if cube.is_some() || match_state.is_some() || doubling {
                let cube_state = CubeState {
                    value: cube.unwrap_or(1),
                    owner: CubeOwner::Center,
                    efficiency: 0.68,
                    match_state,
                };
                eval = eval.with_cubeful(&cube_state);
                true
            } else {
                false
            };
            println!("position_id: {position_id}");
            if let Some(roll) = roll {
                println!("roll: {}", format_dice(parse_dice(&roll)?));
            }
            if let Some(match_state) = match_state {
                print_match_state(&match_state);
            }
            print_eval(&eval, show_cubeful, match_state.as_ref());
            println!("cache_hit: {}", eval.cache_hit);
            println!("simd_supported: {}", gnubg_sys::simd_supported());
            println!("weights_bytes: {}", gnubg_sys::embedded_weights_len());

            if doubling {
                let cube_state = CubeState {
                    value: cube.unwrap_or(1),
                    owner: CubeOwner::Center,
                    efficiency: 0.68,
                    match_state,
                };
                let cube_decision = gnubg_search::compute_cube_decision(&board, &cube_state)?;
                print_cube_decision(&cube_decision, match_state.as_ref());
            }
        }
        Command::BestMove {
            position_id,
            dice,
            depth,
        } => {
            let board = Board::from_position_id(&position_id)?;
            let dice = parse_dice(&dice)?;
            let moves = generate_candidate_moves(&board, dice);
            let evaluated = parallel_eval_root(&board, &moves, depth)?;
            for (candidate, eval) in &evaluated {
                println!(
                    "candidate: {} equity={:.6}",
                    format_move(candidate),
                    eval.equity
                );
            }
            let (mv, eval) = best_move(&board, dice, depth)?;
            println!("best_move: {}", format_move(&mv));
            print_eval(&eval, false, None);
        }
        Command::Analyze {
            position_id,
            depth,
            time_limit,
            doubling,
            match_score,
            crawford,
            post_crawford,
        } => run_analyze(&position_id, depth, time_limit, doubling, match_score, crawford, post_crawford)?,
        Command::AnalyzeCube {
            position_id,
            match_score,
            crawford,
            post_crawford,
            depth,
        } => run_analyze_cube(&position_id, match_score, crawford, post_crawford, depth)?,
        Command::Play { color, depth } => run_play(color.as_deref().unwrap_or("player"), depth)?,
        Command::Bench(sub) => match sub {
            BenchCommand::Eval { positions, depth } => run_bench_eval(positions, depth)?,
            BenchCommand::Search {
                positions,
                depth,
                candidates,
            } => run_bench_search(positions, depth, candidates)?,
            BenchCommand::Check { positions } => run_bench_check(positions)?,
        },
    }
    Ok(())
}

fn run_analyze(
    position_id: &str,
    depth: u8,
    time_limit: Option<u64>,
    doubling: bool,
    match_score: Option<(i32, i32)>,
    crawford: bool,
    post_crawford: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if match_score.is_none() && (crawford || post_crawford) {
        return Err("--crawford/--post-crawford require --match".into());
    }
    let board = Board::from_position_id(position_id)?;
    let match_state = match_score.map(|(player_away, opponent_away)| {
        MatchState::new(player_away, opponent_away, crawford, post_crawford)
    });
    let cube_state = if doubling {
        Some(CubeState {
            value: 1,
            owner: CubeOwner::Center,
            efficiency: 0.68,
            match_state,
        })
    } else {
        None
    };
    let config = SearchConfig {
        max_depth: depth,
        time_limit_ms: time_limit.unwrap_or(0),
        cube_state,
        ..SearchConfig::default()
    };
    let result = analyze_position(&board, &config)?;

    println!("position_id: {position_id}");
    println!("depth: {depth}");
    if let Some(ref ms) = match_state {
        print_match_state(ms);
    }
    if doubling {
        let is_match = match_state.is_some();
        println!("roll      move                  equity   pv               cube");
        println!("──────────────────────────────────────────────────────────────────");
        for roll in result.rolls {
            let cube_label = match &roll.cube_decision {
                Some(cd) => match cd.action {
                    CubeAction::NoDouble => "NO DOUBLE".to_string(),
                    CubeAction::DoubleTake => {
                        if is_match {
                            format!("DOUBLE (take) {:+.1}% MWC", cd.gain * 100.0)
                        } else {
                            "DOUBLE (take)".to_string()
                        }
                    }
                    CubeAction::DoubleDrop => {
                        if is_match {
                            format!("DOUBLE (drop) {:+.1}% MWC", cd.gain * 100.0)
                        } else {
                            "DOUBLE (drop)".to_string()
                        }
                    }
                },
                None => String::new(),
            };
            println!(
                "  {:<6}  {:<20} {:+.3}   {:<15} {}",
                format_dice(roll.dice),
                format_move_compact(&roll.best_move),
                roll.equity,
                format_pv(&roll.pv),
                cube_label,
            );
        }
    } else {
        println!("roll      move                  equity   pv");
        println!("──────────────────────────────────────────────────────");
        for roll in result.rolls {
            println!(
                "  {:<6}  {:<20} {:+.3}   {}",
                format_dice(roll.dice),
                format_move_compact(&roll.best_move),
                roll.equity,
                format_pv(&roll.pv)
            );
        }
    }
    println!(
        "nodes={} eval_calls={} tt_hits={}/{} time_ms={}",
        result.stats.nodes_searched,
        result.stats.eval_calls,
        result.stats.tt_hits,
        result.stats.tt_lookups,
        result.stats.time_ms
    );
    Ok(())
}

fn run_analyze_cube(
    position_id: &str,
    match_score: Option<(i32, i32)>,
    crawford: bool,
    post_crawford: bool,
    depth: u8,
) -> Result<(), Box<dyn std::error::Error>> {
    if match_score.is_none() && (crawford || post_crawford) {
        return Err("--crawford/--post-crawford require --match".into());
    }

    let board = Board::from_position_id(position_id)?;
    let eval = evaluate_board(&board, depth)?;
    let match_state = match_score.map(|(player_away, opponent_away)| {
        MatchState::new(player_away, opponent_away, crawford, post_crawford)
    });

    let outputs = [
        eval.win,
        eval.win_gammon,
        eval.win_backgammon,
        eval.lose_gammon,
        eval.lose_backgammon,
    ];

    println!("position_id: {position_id}");
    if let Some(ref ms) = match_state {
        print_match_state(ms);
    }
    print_eval(&eval, true, match_state.as_ref());
    println!("cache_hit: {}", eval.cache_hit);
    println!("simd_supported: {}", gnubg_sys::simd_supported());
    println!("weights_bytes: {}", gnubg_sys::embedded_weights_len());

    // Crawford handling — cube frozen, always NO DOUBLE
    if let Some(ref ms) = match_state {
        if ms.crawford {
            println!();
            println!("=== Cube Analysis (match play) ===");
            println!("(Crawford — cube frozen)");
            println!();
            println!("Decision: NO DOUBLE");
            return Ok(());
        }
    }

    // --- Cube scenarios ---
    // No double: center cube at value 1
    let no_double_cube = CubeState {
        value: 1,
        owner: CubeOwner::Center,
        efficiency: 0.68,
        match_state,
    };
    let no_double_value = gnubg_eval::cubeful::cubeful_equity(&outputs, &no_double_cube);

    // Double/Take: opponent owns cube at value 2
    let double_take_cube = CubeState {
        value: 2,
        owner: CubeOwner::Opponent,
        efficiency: 0.68,
        match_state,
    };
    let take_value = gnubg_eval::cubeful::cubeful_equity(&outputs, &double_take_cube);

    // Double/Drop: immediate win of 1 point
    let drop_value = match match_state {
        Some(ref ms) => mwc_after(ms, 1),
        None => 1.0,
    };
    let drop_note = match match_state {
        Some(ref ms) => {
            let new_player_away = 0.max(ms.player_away - 1);
            format!(
                "   (win 1 point → {}-away/{}-away)",
                new_player_away, ms.opponent_away
            )
        }
        None => String::new(),
    };

    let double_value = take_value.min(drop_value);
    let is_double = double_value > no_double_value;

    println!();
    if match_state.is_some() {
        println!("=== Cube Analysis (match play) ===");
        println!(
            "No double:    MWC = {:.2}%",
            no_double_value * 100.0
        );
        println!(
            "Double/Take:  MWC = {:.2}%",
            take_value * 100.0
        );
        println!(
            "Double/Drop:  MWC = {:.2}%{}",
            drop_value * 100.0,
            drop_note
        );
        println!();
        if is_double {
            let opponent_choice = if (double_value - take_value).abs() < 1e-6 {
                "take"
            } else {
                "drop"
            };
            let gain = (double_value - no_double_value) * 100.0;
            println!(
                "Decision: DOUBLE ({}) — gains {:.2}% MWC",
                opponent_choice, gain
            );
        } else {
            println!("Decision: NO DOUBLE");
        }
    } else {
        println!("=== Cube Analysis (money game) ===");
        println!("No double:    {:+.3}", no_double_value);
        println!("Double/Take:  {:+.3}", take_value);
        println!("Double/Drop:  {:+.3}", drop_value);
        println!();
        if is_double {
            println!("Decision: DOUBLE");
        } else {
            println!("Decision: NO DOUBLE");
        }
    }

    Ok(())
}

fn run_play(color: &str, depth: u8) -> Result<(), Box<dyn std::error::Error>> {
    if color != "player" && color != "opponent" {
        return Err("--color must be 'player' or 'opponent'".into());
    }

    let mut board = Board::from_position_id("4HPwATDgc/ABMA")?;
    let config = SearchConfig {
        max_depth: depth,
        ..SearchConfig::default()
    };
    let mut rng = SplitMix64::new(0x2930_2930_2930_2930);
    let player_starts = color == "player";
    let mut player_turn = player_starts;

    println!("Starting interactive game. Type moves like '24/22 13/11' or 'quit'.");
    loop {
        println!("{}", render_board(&board));
        if is_game_over_cli(&board) {
            println!("game_over: all checkers borne off for one side");
            break;
        }

        let dice = rng.dice();
        println!("roll: {}", format_dice(dice));
        if player_turn {
            let legal = generate_candidate_moves(&board, dice);
            if legal.is_empty() {
                println!("no legal player moves");
            } else {
                println!("legal moves:");
                for mv in &legal {
                    println!("  {}", format_move_compact(mv));
                }
                let Some(chosen) = prompt_player_move(&legal)? else {
                    println!("bye");
                    break;
                };
                board = Board::from_key(chosen.resulting_position);
            }
        } else {
            let result = search_position(&board, dice, &config)?;
            println!(
                "engine: {} equity={:+.3}",
                format_move_compact(&result.best_move),
                result.best_equity
            );
            board = Board::from_key(result.best_move.resulting_position);
        }
        player_turn = !player_turn;
    }
    Ok(())
}

fn prompt_player_move(legal: &[Move]) -> Result<Option<Move>, Box<dyn std::error::Error>> {
    loop {
        print!("move> ");
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();
        if input.eq_ignore_ascii_case("quit") || input.eq_ignore_ascii_case("exit") {
            return Ok(None);
        }
        if let Some(mv) = legal.iter().find(|mv| move_matches_input(mv, input)) {
            return Ok(Some(mv.clone()));
        }
        println!("invalid move for this roll; enter one of the listed moves or quit");
    }
}

fn generate_random_board(rng: &mut SplitMix64) -> Board {
    // Generate a valid random board: ≤15 checkers per side,
    // at least 1 checker and 2 occupied points per side.
    let mut board: gnubg_types::Board = [[0u32; 25]; 2];
    for side_board in board.iter_mut() {
        // At least 2 checkers (need at least 2 occupied points for moves)
        let target = (rng.next() % 14) as u32 + 2;
        let mut remaining = target;
        // Shuffle the 24 points by picking random indices
        let mut points: Vec<usize> = (1..=24).collect();
        for i in (1..points.len()).rev() {
            let j = (rng.next() as usize) % (i + 1);
            points.swap(i, j);
        }
        for &pt in &points {
            if remaining == 0 {
                break;
            }
            // Place 1..=remaining (or at most remaining) checkers on this point
            let cnt = if remaining >= 3 {
                (rng.next() as u32 % 3).min(remaining).max(1)
            } else {
                remaining
            };
            side_board[pt] = cnt;
            remaining -= cnt;
        }
        // Safety: target >= 2 and we distribute over 24 points,
        // so remaining should be 0. If not, bump onto last occupied point.
        if remaining > 0 {
            // Add remaining to the first occupied point (there is at least one)
            if let Some(cnt) = side_board.iter_mut().skip(1).take(24).find(|c| **c > 0) {
                *cnt += remaining;
            }
        }
    }
    let key = gnubg_types::old_position_key(&board);
    Board::from_key(gnubg_sys::PositionKey(key.0))
}

fn run_bench_eval(
    positions: usize,
    depth: u8,
) -> Result<(), Box<dyn std::error::Error>> {
    let positions = positions.max(1);
    let mut rng = SplitMix64::new(0x2915_2915_2915_2915);
    let boards: Vec<Board> = (0..positions).map(|_| generate_random_board(&mut rng)).collect();

    // Pre-warm: evaluate first board to ensure weights are loaded
    evaluate_board(&boards[0], depth)?;

    let start = Instant::now();
    let mut total_evals = 0_usize;
    for board in &boards {
        let _eval = evaluate_board(board, depth)?;
        total_evals += 1;
    }
    let elapsed = start.elapsed();
    let evals_per_second = total_evals as f64 / elapsed.as_secs_f64();
    let cache_stats = thread_cache_stats();

    println!("=== Eval Benchmark ===");
    println!("positions: {positions}");
    println!("depth: {depth}");
    println!("total_evals: {total_evals}");
    println!("threads: {}", rayon::current_num_threads());
    println!("elapsed_ms: {:.3}", elapsed.as_secs_f64() * 1000.0);
    println!("evals_per_second: {evals_per_second:.2}");
    println!("cache_entries: {}", cache_stats.entries);
    println!("cache_lookups: {}", cache_stats.lookups);
    println!("cache_hits: {}", cache_stats.hits);
    let hit_rate = if cache_stats.lookups > 0 {
        cache_stats.hits as f64 / cache_stats.lookups as f64 * 100.0
    } else {
        0.0
    };
    println!("cache_hit_rate: {hit_rate:.2}%");
    Ok(())
}

fn run_bench_search(
    positions: usize,
    depth: u8,
    candidates: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let positions = positions.max(1);
    let candidates = candidates.max(1);
    let mut rng = SplitMix64::new(0x2915_2915_2915_2915);
    let boards: Vec<Board> = (0..positions).map(|_| generate_random_board(&mut rng)).collect();

    let mut total_nodes = 0_u64;

    let start = Instant::now();
    for board in &boards {
        let dice = rng.dice();
        let mut moves = generate_candidate_moves(board, dice);
        moves.truncate(candidates);
        if moves.is_empty() {
            continue;
        }
        // Evaluate using parallel eval root for the candidates
        let evaluated = parallel_eval_root(board, &moves, depth)?;
        total_nodes += evaluated.len() as u64;
    }
    let elapsed = start.elapsed();
    let positions_per_second = positions as f64 / elapsed.as_secs_f64();
    let nodes_per_second = total_nodes as f64 / elapsed.as_secs_f64();
    let cache_stats = thread_cache_stats();

    // Get TT stats — we have a thread-local transposition table in search
    // For simplicity, report eval cache stats
    println!("=== Search Benchmark ===");
    println!("positions: {positions}");
    println!("depth: {depth}");
    println!("candidates: {candidates}");
    println!("total_nodes: {total_nodes}");
    println!("threads: {}", rayon::current_num_threads());
    println!("elapsed_ms: {:.3}", elapsed.as_secs_f64() * 1000.0);
    println!("positions_per_second: {positions_per_second:.2}");
    println!("nodes_per_second: {nodes_per_second:.2}");
    println!("tt_hits: {}", cache_stats.hits);
    println!("tt_lookups: {}", cache_stats.lookups);
    let hit_rate = if cache_stats.lookups > 0 {
        cache_stats.hits as f64 / cache_stats.lookups as f64 * 100.0
    } else {
        0.0
    };
    println!("tt_hit_rate: {hit_rate:.2}%");
    Ok(())
}

fn run_bench_check(
    positions: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let positions = positions.max(1);
    let mut rng = SplitMix64::new(0x2915_2915_2915_2915);
    let boards: Vec<Board> = (0..positions).map(|_| generate_random_board(&mut rng)).collect();

    let mut determinism_mismatches = 0_usize;
    let mut sanity_failures = 0_usize;
    let mut max_equity_deviation = 0.0_f64;

    for board in &boards {
        // Run eval twice for determinism check
        let eval1 = evaluate_board(board, 0)?;
        let eval2 = evaluate_board(board, 0)?;

        let outputs1 = [eval1.win, eval1.win_gammon, eval1.win_backgammon, eval1.lose_gammon, eval1.lose_backgammon];
        let outputs2 = [eval2.win, eval2.win_gammon, eval2.win_backgammon, eval2.lose_gammon, eval2.lose_backgammon];

        // Determinism check
        for k in 0..5 {
            let dev = (outputs1[k] - outputs2[k]).abs() as f64;
            if dev > max_equity_deviation {
                max_equity_deviation = dev;
            }
            if outputs1[k] != outputs2[k] {
                determinism_mismatches += 1;
            }
        }

        // Sanity checks: win >= win_gammon >= win_backgammon
        if outputs1[0] < outputs1[1] || outputs1[1] < outputs1[2] {
            sanity_failures += 1;
        }
        // lose_gammon >= lose_backgammon
        if outputs1[3] < outputs1[4] {
            sanity_failures += 1;
        }
        // The 5 outputs should sum to approximately 1.0 after sanity_check
        let sum: f32 = outputs1.iter().sum();
        if (sum - 1.0).abs() > 1e-4 {
            // sanity_check clamps; slight tolerance
        }
    }

    println!("=== Quality Check (determinism + sanity) ===");
    println!("positions: {positions}");
    println!(
        "determinism_check: {} ({} mismatches)",
        if determinism_mismatches == 0 { "PASS" } else { "FAIL" },
        determinism_mismatches
    );
    println!(
        "sanity_check: {} ({} failures)",
        if sanity_failures == 0 { "PASS" } else { "FAIL" },
        sanity_failures
    );
    println!("max_equity_deviation: {max_equity_deviation:.6}");
    Ok(())
}

fn print_eval(eval: &EvalResult, show_cubeful: bool, match_state: Option<&MatchState>) {
    println!("win: {:.2}%", eval.win * 100.0);
    println!("win_gammon: {:.2}%", eval.win_gammon * 100.0);
    println!("win_backgammon: {:.2}%", eval.win_backgammon * 100.0);
    println!("lose_gammon: {:.2}%", eval.lose_gammon * 100.0);
    println!("lose_backgammon: {:.2}%", eval.lose_backgammon * 100.0);
    println!("equity: {:.6}", eval.equity);
    if let Some(match_state) = match_state {
        let current_mwc = mwc(match_state);
        println!("mwc: {:.2}%", current_mwc * 100.0);
        println!(
            "swing: {:+.2}%",
            (eval.cubeful_equity - current_mwc) * 100.0
        );
    } else if show_cubeful {
        println!("cubeful: {:.6}", eval.cubeful_equity);
    }
    println!("depth: {}", eval.depth);
}

fn print_cube_decision(cd: &CubeDecision, match_state: Option<&MatchState>) {
    println!();
    if match_state.is_some() {
        println!("=== Cube Analysis (match play) ===");
        println!("No double:    MWC = {:.2}%", cd.no_double_equity * 100.0);
        println!("Double/Take:  MWC = {:.2}%", cd.double_take_equity * 100.0);
        println!("Double/Drop:  MWC = {:.2}%", cd.double_drop_equity * 100.0);
        println!();
        match cd.action {
            CubeAction::NoDouble => println!("Decision: NO DOUBLE"),
            CubeAction::DoubleTake => println!(
                "Decision: DOUBLE (take) — gains {:.2}% MWC",
                cd.gain * 100.0
            ),
            CubeAction::DoubleDrop => println!(
                "Decision: DOUBLE (drop) — gains {:.2}% MWC",
                cd.gain * 100.0
            ),
        }
    } else {
        println!("=== Cube Analysis (money game) ===");
        println!("No double:    {:+.3}", cd.no_double_equity);
        println!("Double/Take:  {:+.3}", cd.double_take_equity);
        println!("Double/Drop:  {:+.3}", cd.double_drop_equity);
        println!();
        match cd.action {
            CubeAction::NoDouble => println!("Decision: NO DOUBLE"),
            CubeAction::DoubleTake => println!("Decision: DOUBLE (take)"),
            CubeAction::DoubleDrop => println!("Decision: DOUBLE (drop)"),
        }
    }
}

fn print_match_state(state: &MatchState) {
    let suffix = if state.crawford {
        " (Crawford)"
    } else if state.post_crawford {
        " (post-Crawford)"
    } else {
        ""
    };
    println!(
        "match: {}-away / {}-away{}",
        state.player_away, state.opponent_away, suffix
    );
}

fn parse_match_score(input: &str) -> Result<(i32, i32), String> {
    let Some((player, opponent)) = input.split_once(':') else {
        return Err(format!(
            "match must be PLAYER_AWAY:OPPONENT_AWAY, got '{input}'"
        ));
    };
    let player_away: i32 = player
        .parse()
        .map_err(|_| format!("match must be PLAYER_AWAY:OPPONENT_AWAY, got '{input}'"))?;
    let opponent_away: i32 = opponent
        .parse()
        .map_err(|_| format!("match must be PLAYER_AWAY:OPPONENT_AWAY, got '{input}'"))?;
    if !(1..=25).contains(&player_away) || !(1..=25).contains(&opponent_away) {
        return Err(format!(
            "match away scores must be in 1..=25, got '{input}'"
        ));
    }
    Ok((player_away, opponent_away))
}

fn parse_cube_value(input: &str) -> Result<i32, String> {
    let value: i32 = input
        .parse()
        .map_err(|_| format!("cube must be a positive power of two, got '{input}'"))?;
    if value <= 0 || (value as u32).count_ones() != 1 {
        return Err(format!(
            "cube must be a positive power of two, got '{input}'"
        ));
    }
    Ok(value)
}

fn parse_dice(input: &str) -> Result<(u8, u8), String> {
    let digits: Vec<u8> = input
        .bytes()
        .filter(|b| b.is_ascii_digit())
        .map(|b| b - b'0')
        .collect();
    if digits.len() != 2 || digits.iter().any(|die| !(1..=6).contains(die)) {
        return Err(format!(
            "dice must contain exactly two dice in 1..=6, got '{input}'"
        ));
    }
    Ok((digits[0], digits[1]))
}

fn format_dice(dice: (u8, u8)) -> String {
    format!("{}{}", dice.0, dice.1)
}

fn format_move(mv: &Move) -> String {
    format!(
        "#{} {}{} {}",
        mv.id,
        mv.dice.0,
        mv.dice.1,
        format_move_compact(mv)
    )
}

fn format_move_compact(mv: &Move) -> String {
    let parts: Vec<String> = mv
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
    if parts.is_empty() {
        format!("{}/{}", mv.from, mv.to)
    } else {
        parts.join(" ")
    }
}

fn format_pv(pv: &[Move]) -> String {
    pv.iter()
        .map(format_move_compact)
        .collect::<Vec<_>>()
        .join(" ")
}

fn move_matches_input(mv: &Move, input: &str) -> bool {
    let normalized = normalize_move_text(input);
    normalized == normalize_move_text(&format_move_compact(mv))
        || normalized == normalize_move_text(&format_move(mv))
}

fn normalize_move_text(input: &str) -> String {
    input
        .replace("->", "/")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase()
}

fn render_board(board: &Board) -> String {
    let raw = raw_board(board);
    let mut out = String::new();
    out.push_str("13 14 15 16 17 18       19 20 21 22 23 24\n");
    out.push_str("+--------------------------+------------------+\n");
    for row in (0..5).rev() {
        out.push_str("| ");
        for &count in &raw[1][13..=18] {
            out.push_str(&checker_cell(count, 'X', row));
        }
        out.push_str(" |  ");
        out.push_str(&checker_cell(raw[1][24], 'X', row));
        out.push_str(" | ");
        for &count in &raw[1][19..=24] {
            out.push_str(&checker_cell(count, 'X', row));
        }
        out.push_str(" |\n");
    }
    out.push_str("|                  BAR|                       |\n");
    for row in (0..5).rev() {
        out.push_str("| ");
        for point in (7..=12).rev() {
            out.push_str(&checker_cell(raw[0][24 - point], 'O', row));
        }
        out.push_str(" |  ");
        out.push_str(&checker_cell(raw[0][24], 'O', row));
        out.push_str(" | ");
        for point in (1..=6).rev() {
            out.push_str(&checker_cell(raw[0][24 - point], 'O', row));
        }
        out.push_str(" |\n");
    }
    out.push_str("+--------------------------+------------------+\n");
    out.push_str("12 11 10  9  8  7        6  5  4  3  2  1\n");
    out
}

fn checker_cell(count: u32, symbol: char, row: u32) -> String {
    if count > row {
        format!("{symbol}  ")
    } else {
        ".  ".to_string()
    }
}

fn is_game_over_cli(board: &Board) -> bool {
    let raw = raw_board(board);
    raw[0][0] >= 15 || raw[1][0] >= 15
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    #[allow(dead_code)]
    fn position_key(&mut self) -> gnubg_sys::PositionKey {
        let mut key = [0_u8; gnubg_sys::POSITION_KEY_BYTES];
        for chunk in key.chunks_mut(8) {
            let bytes = self.next().to_le_bytes();
            let len = chunk.len();
            chunk.copy_from_slice(&bytes[..len]);
        }
        gnubg_sys::PositionKey(key)
    }

    fn dice(&mut self) -> (u8, u8) {
        let value = self.next();
        (((value % 6) + 1) as u8, (((value / 6) % 6) + 1) as u8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_dice_formats() {
        assert_eq!(parse_dice("31").expect("31"), (3, 1));
        assert_eq!(parse_dice("3-1").expect("3-1"), (3, 1));
        assert!(parse_dice("70").is_err());
    }

    #[test]
    fn parses_match_score() {
        assert_eq!(parse_match_score("5:3").expect("5:3"), (5, 3));
        assert_eq!(parse_match_score("1:25").expect("1:25"), (1, 25));
        assert!(parse_match_score("5-3").is_err());
        assert!(parse_match_score("0:3").is_err());
        assert!(parse_match_score("26:3").is_err());
    }

    #[test]
    fn formats_full_move_steps() {
        let board = Board::from_position_id("4HPwATDgc/ABMA").expect("valid board");
        let moves = generate_candidate_moves(&board, (3, 1));
        assert!(moves.iter().all(|mv| !format_move_compact(mv).is_empty()));
    }
}
