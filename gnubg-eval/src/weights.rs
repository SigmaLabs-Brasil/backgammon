#![forbid(unsafe_code)]

use std::error::Error;
use std::fmt;

pub const CONTACT_INPUTS: usize = 250;
pub const RACE_INPUTS: usize = 214;
pub const CRASHED_INPUTS: usize = 250;
pub const OUTPUTS: usize = 5;

#[derive(Clone, Debug)]
pub struct NetworkWeights {
    pub c_input: usize,
    pub c_hidden: usize,
    pub c_output: usize,
    pub n_trained: usize,
    pub beta_hidden: f32,
    pub beta_output: f32,
    pub hidden_weights: Vec<f32>,
    pub hidden_thresholds: Vec<f32>,
    pub output_weights: Vec<f32>,
    pub output_thresholds: Vec<f32>,
}

impl NetworkWeights {
    pub fn expected_float_count(c_input: usize, c_hidden: usize, c_output: usize) -> usize {
        c_input * c_hidden + c_hidden + c_hidden * c_output + c_output
    }

    pub fn total_float_count(&self) -> usize {
        self.hidden_weights.len()
            + self.hidden_thresholds.len()
            + self.output_weights.len()
            + self.output_thresholds.len()
    }
}

#[derive(Clone, Debug)]
pub struct WeightFile {
    pub contact: NetworkWeights,
    pub race: NetworkWeights,
    pub crashed: NetworkWeights,
    pub networks: Vec<NetworkWeights>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WeightError {
    Empty,
    InvalidMagic(String),
    MissingHeader {
        network: usize,
    },
    InvalidHeader {
        network: usize,
        line: String,
    },
    InvalidDimension {
        network: usize,
        got: (usize, usize, usize),
    },
    InvalidFloat {
        network: usize,
        line: String,
    },
    Truncated {
        network: usize,
        expected: usize,
        got: usize,
    },
    NotEnoughNetworks {
        got: usize,
    },
}

impl fmt::Display for WeightError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => f.write_str("empty weights file"),
            Self::InvalidMagic(line) => write!(f, "invalid weights magic header: {line}"),
            Self::MissingHeader { network } => write!(f, "missing header for network {network}"),
            Self::InvalidHeader { network, line } => {
                write!(f, "invalid header for network {network}: {line}")
            }
            Self::InvalidDimension { network, got } => write!(
                f,
                "unexpected dimensions for network {network}: {} {} {}",
                got.0, got.1, got.2
            ),
            Self::InvalidFloat { network, line } => {
                write!(f, "invalid float in network {network}: {line}")
            }
            Self::Truncated {
                network,
                expected,
                got,
            } => write!(
                f,
                "truncated network {network}: expected {expected} floats, got {got}"
            ),
            Self::NotEnoughNetworks { got } => write!(f, "expected at least 3 networks, got {got}"),
        }
    }
}

impl Error for WeightError {}

pub fn parse_weights(data: &str) -> Result<WeightFile, WeightError> {
    let mut lines = data
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .peekable();
    let magic = lines.next().ok_or(WeightError::Empty)?;
    if magic != "GNU Backgammon 1.01" {
        return Err(WeightError::InvalidMagic(magic.to_string()));
    }

    let expected_dims = [(250, 128, 5), (214, 128, 5), (250, 128, 5)];
    let mut networks = Vec::new();
    while lines.peek().is_some() {
        let network = networks.len();
        let header = lines.next().ok_or(WeightError::MissingHeader { network })?;
        let parts: Vec<_> = header.split_whitespace().collect();
        if parts.len() != 6 {
            return Err(WeightError::InvalidHeader {
                network,
                line: header.to_string(),
            });
        }
        let parse_usize = |s: &str| {
            s.parse::<usize>().map_err(|_| WeightError::InvalidHeader {
                network,
                line: header.to_string(),
            })
        };
        let parse_f32 = |s: &str| {
            s.parse::<f32>().map_err(|_| WeightError::InvalidHeader {
                network,
                line: header.to_string(),
            })
        };
        let c_input = parse_usize(parts[0])?;
        let c_hidden = parse_usize(parts[1])?;
        let c_output = parse_usize(parts[2])?;
        let n_trained = parse_usize(parts[3])?;
        let beta_hidden = parse_f32(parts[4])?;
        let beta_output = parse_f32(parts[5])?;
        if network < expected_dims.len() && (c_input, c_hidden, c_output) != expected_dims[network]
        {
            return Err(WeightError::InvalidDimension {
                network,
                got: (c_input, c_hidden, c_output),
            });
        }
        let expected = NetworkWeights::expected_float_count(c_input, c_hidden, c_output);
        let mut floats = Vec::with_capacity(expected);
        for _ in 0..expected {
            let line = lines.next().ok_or(WeightError::Truncated {
                network,
                expected,
                got: floats.len(),
            })?;
            floats.push(line.parse::<f32>().map_err(|_| WeightError::InvalidFloat {
                network,
                line: line.to_string(),
            })?);
        }
        let hidden_end = c_input * c_hidden;
        let output_end = hidden_end + c_hidden * c_output;
        let hidden_threshold_end = output_end + c_hidden;
        networks.push(NetworkWeights {
            c_input,
            c_hidden,
            c_output,
            n_trained,
            beta_hidden,
            beta_output,
            hidden_weights: floats[..hidden_end].to_vec(),
            hidden_thresholds: floats[output_end..hidden_threshold_end].to_vec(),
            output_weights: floats[hidden_end..output_end].to_vec(),
            output_thresholds: floats[hidden_threshold_end..].to_vec(),
        });
    }
    if networks.len() < 3 {
        return Err(WeightError::NotEnoughNetworks {
            got: networks.len(),
        });
    }
    Ok(WeightFile {
        contact: networks[0].clone(),
        race: networks[1].clone(),
        crashed: networks[2].clone(),
        networks,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    const REAL_WEIGHTS: &str = include_str!("../../gnubg-sys/vendor/gnubg.weights");

    #[test]
    fn parses_real_file() {
        let weights = parse_weights(REAL_WEIGHTS).expect("real weights parse");
        assert_eq!(weights.contact.c_input, CONTACT_INPUTS);
        assert_eq!(weights.race.c_input, RACE_INPUTS);
        assert_eq!(weights.crashed.c_input, CRASHED_INPUTS);
        assert_eq!(weights.contact.c_hidden, 128);
        assert_eq!(weights.contact.c_output, OUTPUTS);
        assert_eq!(weights.networks.len(), 6);
    }

    #[test]
    fn verifies_float_counts() {
        let weights = parse_weights(REAL_WEIGHTS).expect("real weights parse");
        assert_eq!(weights.contact.total_float_count(), 32_773);
        assert_eq!(weights.race.total_float_count(), 28_165);
        assert_eq!(weights.crashed.total_float_count(), 32_773);
    }

    #[test]
    fn rejects_bad_magic() {
        let err = parse_weights("not gnubg\n250 128 5 0 0.1 1.0\n").unwrap_err();
        assert!(matches!(err, WeightError::InvalidMagic(_)));
    }

    #[test]
    fn rejects_truncated_input() {
        let data = "GNU Backgammon 1.01\n250 128 5 0 0.1 1.0\n1.0\n";
        let err = parse_weights(data).unwrap_err();
        assert!(matches!(err, WeightError::Truncated { .. }));
    }
}
