//! Pure Rust GNU Backgammon PositionID compatibility surface.

#![forbid(unsafe_code)]

use std::error::Error;
use std::fmt;

pub const POSITION_KEY_BYTES: usize = 10;
pub const NUM_OUTPUTS: usize = 5;

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
            Self::EvaluationFailed => f.write_str("evaluation is provided by gnubg-eval"),
            Self::NeuralNetFailed => f.write_str("neural net evaluation is provided by gnubg-eval"),
        }
    }
}

impl Error for GnuBgError {}

pub type Result<T> = std::result::Result<T, GnuBgError>;

pub fn decode_position_id(position_id: &str) -> Result<PositionKey> {
    if position_id.as_bytes().contains(&0) {
        return Err(GnuBgError::InteriorNul);
    }
    if position_id.len() == POSITION_KEY_BYTES * 2 {
        return decode_hex_key(position_id).map(PositionKey);
    }
    gnubg_types::old_key_from_position_id(position_id)
        .map(|key| PositionKey(key.0))
        .ok_or(GnuBgError::InvalidPositionId)
}

pub fn simd_supported() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

pub fn embedded_weights_len() -> usize {
    include_bytes!("../vendor/gnubg.weights").len()
}

fn decode_hex_key(input: &str) -> Result<[u8; POSITION_KEY_BYTES]> {
    let mut key = [0_u8; POSITION_KEY_BYTES];
    let bytes = input.as_bytes();
    for i in 0..POSITION_KEY_BYTES {
        let hi = hex_value(bytes[i * 2]).ok_or(GnuBgError::InvalidPositionId)?;
        let lo = hex_value(bytes[i * 2 + 1]).ok_or(GnuBgError::InvalidPositionId)?;
        key[i] = (hi << 4) | lo;
    }
    Ok(key)
}

fn hex_value(ch: u8) -> Option<u8> {
    match ch {
        b'0'..=b'9' => Some(ch - b'0'),
        b'a'..=b'f' => Some(ch - b'a' + 10),
        b'A'..=b'F' => Some(ch - b'A' + 10),
        _ => None,
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
    fn rejects_invalid_position_id() {
        assert_eq!(
            decode_position_id("short"),
            Err(GnuBgError::InvalidPositionId)
        );
    }
}
