use clap::{Parser, Subcommand};
use gnubg_search::{
    best_move, evaluate_board, generate_candidate_moves, parallel_eval_root, thread_cache_stats,
    Board, EvalResult,
};
use mimalloc::MiMalloc;
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
    },
    /// Generate candidate root moves for a roll and print the highest equity move.
    BestMove {
        position_id: String,
        dice: String,
        #[arg(long, default_value_t = 0)]
        depth: u8,
    },
    /// Run deterministic random-position evaluations and print throughput.
    Bench {
        #[arg(long, default_value_t = 10_000)]
        positions: usize,
        #[arg(long, default_value_t = 8)]
        candidates: usize,
        #[arg(long, default_value_t = 0)]
        depth: u8,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.command {
        Command::Evaluate {
            position_id,
            roll,
            depth,
        } => {
            let board = Board::from_position_id(&position_id)?;
            let eval = evaluate_board(&board, depth)?;
            println!("position_id: {position_id}");
            if let Some(roll) = roll {
                println!("roll: {}", format_dice(parse_dice(&roll)?));
            }
            print_eval(&eval);
            println!("cache_hit: {}", eval.cache_hit);
            println!("simd_supported: {}", gnubg_sys::simd_supported());
            println!("weights_bytes: {}", gnubg_sys::embedded_weights_len());
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
                println!("candidate: {candidate} equity={:.6}", eval.equity);
            }
            let (mv, eval) = best_move(&board, dice, depth)?;
            println!("best_move: {mv}");
            print_eval(&eval);
        }
        Command::Bench {
            positions,
            candidates,
            depth,
        } => run_bench(positions, candidates, depth)?,
    }
    Ok(())
}

fn run_bench(
    positions: usize,
    candidates: usize,
    depth: u8,
) -> Result<(), Box<dyn std::error::Error>> {
    let positions = positions.max(1);
    let candidates = candidates.max(1);
    let mut rng = SplitMix64::new(0x2915_2915_2915_2915);
    let boards: Vec<Board> = (0..positions)
        .map(|_| Board::from_key(rng.position_key()))
        .collect();

    let start = Instant::now();
    let mut evals = 0_usize;
    for board in &boards {
        let dice = rng.dice();
        let mut moves = generate_candidate_moves(board, dice);
        moves.truncate(candidates);
        evals += parallel_eval_root(board, &moves, depth)?.len();
    }
    let elapsed = start.elapsed();
    let positions_per_second = evals as f64 / elapsed.as_secs_f64();
    let stats = thread_cache_stats();

    println!("positions: {positions}");
    println!("candidate_evals: {evals}");
    println!("threads: {}", rayon::current_num_threads());
    println!("elapsed_ms: {:.3}", elapsed.as_secs_f64() * 1000.0);
    println!("positions_per_second: {:.2}", positions_per_second);
    println!("cache_entries_per_thread: {}", stats.entries);
    println!("cache_lookups_this_thread: {}", stats.lookups);
    println!("cache_hits_this_thread: {}", stats.hits);
    println!("cache_inserts_this_thread: {}", stats.inserts);
    println!("baseline: gnubg C bridge compiled in release mode with x86-64-v3, LTO, mimalloc");
    Ok(())
}

fn print_eval(eval: &EvalResult) {
    println!("win: {:.2}%", eval.win * 100.0);
    println!("win_gammon: {:.2}%", eval.win_gammon * 100.0);
    println!("win_backgammon: {:.2}%", eval.win_backgammon * 100.0);
    println!("lose_gammon: {:.2}%", eval.lose_gammon * 100.0);
    println!("lose_backgammon: {:.2}%", eval.lose_backgammon * 100.0);
    println!("equity: {:.6}", eval.equity);
    println!("depth: {}", eval.depth);
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
}
