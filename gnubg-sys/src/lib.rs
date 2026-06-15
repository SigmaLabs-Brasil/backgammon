//! Minimal safe FFI surface for the GNU Backgammon evaluation lane.
//!
//! Rust owns threading and caching while C owns the hot numeric evaluation ABI.
//! The embedded gnubg weights are registered once so release binaries are
//! self-contained.

use std::error::Error;
use std::ffi::CString;
use std::fmt;
use std::sync::Once;

pub const POSITION_KEY_BYTES: usize = 10;
pub const NUM_OUTPUTS: usize = 5;

static INIT: Once = Once::new();
static GNUBG_WEIGHTS: &[u8] = include_bytes!("../vendor/gnubg.weights");

unsafe extern "C" {
    fn gnubg_init_embedded_weights(ptr: *const u8, len: usize);
    fn gnubg_position_id_decode(id: *const std::ffi::c_char, out_key: *mut u8) -> i32;
    fn gnubg_evaluate_position(position_key: *const u8, out: *mut f32) -> i32;
    fn gnubg_neuralnet_evaluate(input: *const f32, len: usize, out: *mut f32) -> i32;
    fn gnubg_simd_supported() -> i32;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct PositionKey(pub [u8; POSITION_KEY_BYTES]);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RawEval {
    pub outputs: [f32; NUM_OUTPUTS],
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum GnuBgError {
    InvalidPositionId,
    InteriorNul,
    EvaluationFailed,
    NeuralNetFailed,
}

impl fmt::Display for GnuBgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPositionId => f.write_str("invalid GNU Backgammon PositionID"),
            Self::InteriorNul => f.write_str("position id contains an interior NUL byte"),
            Self::EvaluationFailed => f.write_str("C EvaluatePosition bridge returned an error"),
            Self::NeuralNetFailed => f.write_str("C neural net bridge returned an error"),
        }
    }
}

impl Error for GnuBgError {}

pub type Result<T> = std::result::Result<T, GnuBgError>;

fn ensure_initialized() {
    INIT.call_once(|| unsafe {
        gnubg_init_embedded_weights(GNUBG_WEIGHTS.as_ptr(), GNUBG_WEIGHTS.len());
    });
}

pub fn embedded_weights_len() -> usize {
    GNUBG_WEIGHTS.len()
}

pub fn simd_supported() -> bool {
    unsafe { gnubg_simd_supported() != 0 }
}

pub fn decode_position_id(position_id: &str) -> Result<PositionKey> {
    ensure_initialized();
    let c_id = CString::new(position_id).map_err(|_| GnuBgError::InteriorNul)?;
    let mut key = [0_u8; POSITION_KEY_BYTES];
    let rc = unsafe { gnubg_position_id_decode(c_id.as_ptr(), key.as_mut_ptr()) };
    if rc == 0 {
        Ok(PositionKey(key))
    } else {
        Err(GnuBgError::InvalidPositionId)
    }
}

pub fn evaluate_position_key(position: &PositionKey) -> Result<RawEval> {
    ensure_initialized();
    let mut outputs = [0.0_f32; NUM_OUTPUTS];
    let rc = unsafe { gnubg_evaluate_position(position.0.as_ptr(), outputs.as_mut_ptr()) };
    if rc == 0 {
        Ok(RawEval { outputs })
    } else {
        Err(GnuBgError::EvaluationFailed)
    }
}

pub fn evaluate_position_id(position_id: &str) -> Result<(PositionKey, RawEval)> {
    let key = decode_position_id(position_id)?;
    let eval = evaluate_position_key(&key)?;
    Ok((key, eval))
}

pub fn neuralnet_evaluate(inputs: &[f32]) -> Result<RawEval> {
    ensure_initialized();
    let mut outputs = [0.0_f32; NUM_OUTPUTS];
    let rc =
        unsafe { gnubg_neuralnet_evaluate(inputs.as_ptr(), inputs.len(), outputs.as_mut_ptr()) };
    if rc == 0 {
        Ok(RawEval { outputs })
    } else {
        Err(GnuBgError::NeuralNetFailed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_base64_position_id() {
        let key = decode_position_id("4HPwATDgc/ABMA").expect("start position id decodes");
        assert_eq!(key.0.len(), POSITION_KEY_BYTES);
    }

    #[test]
    fn decodes_hex_position_key() {
        let key = decode_position_id("00112233445566778899").expect("hex position key decodes");
        assert_eq!(
            key.0,
            [0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99]
        );
    }

    #[test]
    fn evaluates_to_probability_vector() {
        let (_, eval) = evaluate_position_id("4HPwATDgc/ABMA").expect("position evaluates");
        for value in eval.outputs {
            assert!((0.0..=1.0).contains(&value));
        }
    }

    #[test]
    fn embeds_weights() {
        assert!(embedded_weights_len() > 1_000_000);
    }
}
