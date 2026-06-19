use crate::weights::NetworkWeights;
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum NeuralNetError {
    InvalidInputLength { expected: usize, got: usize },
}

impl fmt::Display for NeuralNetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInputLength { expected, got } => {
                write!(
                    f,
                    "invalid neural net input length: expected {expected}, got {got}"
                )
            }
        }
    }
}

impl Error for NeuralNetError {}

type ForwardFn = fn(&NeuralNet, &[f32]) -> Result<[f32; 5], NeuralNetError>;

#[derive(Clone)]
pub struct NeuralNet {
    pub c_input: usize,
    pub c_hidden: usize,
    pub c_output: usize,
    beta_hidden: f32,
    beta_output: f32,
    hidden_weights: Vec<f32>,
    hidden_thresholds: Vec<f32>,
    output_weights: Vec<f32>,
    output_thresholds: Vec<f32>,
    forward_fn: ForwardFn,
}

impl fmt::Debug for NeuralNet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NeuralNet")
            .field("c_input", &self.c_input)
            .field("c_hidden", &self.c_hidden)
            .field("c_output", &self.c_output)
            .field("beta_hidden", &self.beta_hidden)
            .field("beta_output", &self.beta_output)
            .finish_non_exhaustive()
    }
}

impl NeuralNet {
    pub fn new(weights: &NetworkWeights) -> Self {
        let forward_fn = if simd_supported() {
            feed_forward_avx2_dispatch as ForwardFn
        } else {
            feed_forward_scalar_dispatch as ForwardFn
        };
        Self {
            c_input: weights.c_input,
            c_hidden: weights.c_hidden,
            c_output: weights.c_output,
            beta_hidden: weights.beta_hidden,
            beta_output: weights.beta_output,
            hidden_weights: weights.hidden_weights.clone(),
            hidden_thresholds: weights.hidden_thresholds.clone(),
            output_weights: weights.output_weights.clone(),
            output_thresholds: weights.output_thresholds.clone(),
            forward_fn,
        }
    }

    pub fn feed_forward(&self, inputs: &[f32]) -> Result<[f32; 5], NeuralNetError> {
        (self.forward_fn)(self, inputs)
    }

    pub fn feed_forward_scalar(&self, inputs: &[f32]) -> Result<[f32; 5], NeuralNetError> {
        feed_forward_scalar_impl(self, inputs)
    }
}

pub fn simd_supported() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

fn feed_forward_scalar_dispatch(
    net: &NeuralNet,
    inputs: &[f32],
) -> Result<[f32; 5], NeuralNetError> {
    feed_forward_scalar_impl(net, inputs)
}

fn feed_forward_avx2_dispatch(net: &NeuralNet, inputs: &[f32]) -> Result<[f32; 5], NeuralNetError> {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: this dispatch function is installed only when runtime AVX2 detection succeeds.
        unsafe { feed_forward_avx2_impl(net, inputs) }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        feed_forward_scalar_impl(net, inputs)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn feed_forward_avx2_impl(
    net: &NeuralNet,
    inputs: &[f32],
) -> Result<[f32; 5], NeuralNetError> {
    use std::arch::x86_64::{
        _mm256_add_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_setzero_ps,
        _mm256_storeu_ps,
    };

    if inputs.len() != net.c_input {
        return Err(NeuralNetError::InvalidInputLength {
            expected: net.c_input,
            got: inputs.len(),
        });
    }

    // AVX2 hidden layer: weight layout = hidden_weights[input_idx * cHidden + hidden_idx]
    // C Evaluate computes: hidden[j] += input[i] * weight[i][j]
    // AVX2: process 8 hidden neurons at once, accumulating each input's contribution
    let mut hidden = [0.0_f32; 128];

    // Initialize hidden with thresholds
    for (j, h) in hidden.iter_mut().enumerate().take(net.c_hidden) {
        *h = net.hidden_thresholds[j];
    }

    // AVX2 inner loop: for each input, broadcast to 8 lanes and FMA into hidden[j..j+8]
    for (i, input) in inputs.iter().enumerate().take(net.c_input) {
        let ari = *input;
        if ari == 0.0 {
            continue;
        }
        let base = i * net.c_hidden; // weight row offset for this input
        let mut j = 0;
        while j + 8 <= net.c_hidden {
            let mut h_vec = unsafe { _mm256_loadu_ps(hidden.as_ptr().add(j)) };
            let w_vec = unsafe { _mm256_loadu_ps(net.hidden_weights.as_ptr().add(base + j)) };
            if ari == 1.0 {
                h_vec = _mm256_add_ps(h_vec, w_vec);
            } else {
                let ari_vec = _mm256_set1_ps(ari);
                h_vec = _mm256_fmadd_ps(ari_vec, w_vec, h_vec);
            }
            unsafe { _mm256_storeu_ps(hidden.as_mut_ptr().add(j), h_vec) };
            j += 8;
        }
        // Scalar remainder
        for (j, h) in hidden.iter_mut().enumerate().take(net.c_hidden).skip(j) {
            *h += net.hidden_weights[base + j] * ari;
        }
    }

    // sigmoid(-beta_hidden * sum) — matches C Evaluate() line 153
    for (_j, h) in hidden.iter_mut().enumerate().take(net.c_hidden) {
        *h = sigmoid(-net.beta_hidden * *h);
    }

    // Output layer: weight layout = output_weights[output_idx * cHidden + hidden_idx]
    let mut output = [0.0_f32; 5];
    for (k, out) in output.iter_mut().enumerate().take(net.c_output) {
        let mut acc = _mm256_setzero_ps();
        let base = k * net.c_hidden;
        let mut j = 0;
        while j + 8 <= net.c_hidden {
            let h_vec = unsafe { _mm256_loadu_ps(hidden.as_ptr().add(j)) };
            let w_vec = unsafe { _mm256_loadu_ps(net.output_weights.as_ptr().add(base + j)) };
            acc = _mm256_fmadd_ps(h_vec, w_vec, acc);
            j += 8;
        }
        let mut sum = net.output_thresholds[k] + reduce_m256(acc);
        // Scalar remainder
        for (j, h) in hidden.iter().enumerate().take(net.c_hidden).skip(j) {
            sum += *h * net.output_weights[base + j];
        }
        // sigmoid(-beta_output * sum) — matches C Evaluate() line 164
        *out = sigmoid(-net.beta_output * sum).clamp(0.0, 1.0);
    }
    Ok(output)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn reduce_m256(value: std::arch::x86_64::__m256) -> f32 {
    let mut lanes = [0.0_f32; 8];
    unsafe { std::arch::x86_64::_mm256_storeu_ps(lanes.as_mut_ptr(), value) };
    lanes.iter().sum()
}

/// C reference: neuralnet.c Evaluate() lines 120-166
///
/// Verified against C source:
///   hidden[j] = sigmoid(-beta_hidden * (threshold[j] + Σ(input[i] * weight[i][j])))
///   output[k] = sigmoid(-beta_output * (output_threshold[k] + Σ(hidden[j] * output_weight[k][j])))
///
/// Weight layout: hidden_weights[input_idx * cHidden + hidden_idx]  (row-major in input)
///                output_weights[output_idx * cHidden + hidden_idx] (row-major in output)
fn feed_forward_scalar_impl(net: &NeuralNet, inputs: &[f32]) -> Result<[f32; 5], NeuralNetError> {
    if inputs.len() != net.c_input {
        return Err(NeuralNetError::InvalidInputLength {
            expected: net.c_input,
            got: inputs.len(),
        });
    }

    let mut hidden = [0.0_f32; 128];

    // Initialize hidden with thresholds (C: ar[i] = pnn->arHiddenThreshold[i])
    for (j, h) in hidden.iter_mut().enumerate().take(net.c_hidden) {
        *h = net.hidden_thresholds[j];
    }

    // Add input contributions (C: for each input i, add weight[i][j] to hidden[j])
    // Weight layout: hidden_weights[input * cHidden + hidden]
    for (i, input) in inputs.iter().enumerate().take(net.c_input) {
        let ari = *input;
        if ari == 0.0 {
            continue; // C: prWeight += cHidden (skip this input's weights)
        }
        let base = i * net.c_hidden;
        if ari == 1.0 {
            for (j, h) in hidden.iter_mut().enumerate().take(net.c_hidden) {
                *h += net.hidden_weights[base + j];
            }
        } else {
            for (j, h) in hidden.iter_mut().enumerate().take(net.c_hidden) {
                *h += net.hidden_weights[base + j] * ari;
            }
        }
    }

    // sigmoid(-beta_hidden * sum) — C: sigmoid(-rBetaHidden * ar[i])
    for (_j, h) in hidden.iter_mut().enumerate().take(net.c_hidden) {
        *h = sigmoid(-net.beta_hidden * *h);
    }

    // Output layer: weight layout = output_weights[output * cHidden + hidden]
    let mut output = [0.0_f32; 5];
    for (k, out) in output.iter_mut().enumerate().take(net.c_output) {
        let mut sum = net.output_thresholds[k];
        let base = k * net.c_hidden;
        for (j, h) in hidden.iter().enumerate().take(net.c_hidden) {
            sum += *h * net.output_weights[base + j];
        }
        // sigmoid(-beta_output * sum) — C: sigmoid(-rBetaOutput * r)
        *out = sigmoid(-net.beta_output * sum).clamp(0.0, 1.0);
    }
    Ok(output)
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    // GNU Backgammon's neuralnet.c sigmoid is the decreasing logistic
    // 1 / (1 + exp(x)). NeuralNetEvaluate calls it with -beta * sum.
    // Using the conventional increasing logistic here flips every neuron and
    // saturates real gnubg weights to [1, 0, 1, 0, 0].
    if x <= -40.0 {
        return 1.0;
    }
    if x >= 40.0 {
        return 0.0;
    }
    (1.0 / (1.0 + x.exp())).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_network() -> NetworkWeights {
        NetworkWeights {
            c_input: 2,
            c_hidden: 2,
            c_output: 5,
            n_trained: 0,
            beta_hidden: 1.0,
            beta_output: 1.0,
            hidden_weights: vec![1.0, 0.0, 0.0, 1.0],
            hidden_thresholds: vec![0.0, 0.0],
            output_weights: vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            output_thresholds: vec![0.0; 5],
        }
    }

    fn simd_synthetic_network() -> NetworkWeights {
        let c_input = 3;
        let c_hidden = 16;
        let c_output = 5;
        let hidden_weights = (0..c_input * c_hidden)
            .map(|i| (i as f32 % 11.0 - 5.0) * 0.025)
            .collect();
        let hidden_thresholds = (0..c_hidden)
            .map(|i| (i as f32 % 7.0 - 3.0) * 0.03)
            .collect();
        let output_weights = (0..c_hidden * c_output)
            .map(|i| (i as f32 % 13.0 - 6.0) * 0.02)
            .collect();
        let output_thresholds = (0..c_output).map(|i| (i as f32 - 2.0) * 0.01).collect();

        NetworkWeights {
            c_input,
            c_hidden,
            c_output,
            n_trained: 0,
            beta_hidden: 0.7,
            beta_output: 1.1,
            hidden_weights,
            hidden_thresholds,
            output_weights,
            output_thresholds,
        }
    }

    #[test]
    fn zero_weights_produce_half_outputs() {
        let mut w = synthetic_network();
        w.hidden_weights.fill(0.0);
        w.output_weights.fill(0.0);
        let net = NeuralNet::new(&w);
        let out = net.feed_forward_scalar(&[1.0, 1.0]).unwrap();
        for value in out {
            assert!((value - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn synthetic_network_is_in_range() {
        let net = NeuralNet::new(&synthetic_network());
        let out = net.feed_forward_scalar(&[0.25, -0.5]).unwrap();
        for value in out {
            assert!((0.0..=1.0).contains(&value));
        }
        assert_ne!(out[0], out[1]);
    }

    #[test]
    fn rejects_wrong_input_length() {
        let net = NeuralNet::new(&synthetic_network());
        let err = net.feed_forward_scalar(&[1.0]).unwrap_err();
        assert_eq!(
            err,
            NeuralNetError::InvalidInputLength {
                expected: 2,
                got: 1
            }
        );
    }

    #[test]
    fn avx2_matches_scalar_when_available() {
        if !simd_supported() {
            return;
        }
        let net = NeuralNet::new(&simd_synthetic_network());
        let scalar = net.feed_forward_scalar(&[0.3, 0.7, -0.2]).unwrap();
        let dispatched = net.feed_forward(&[0.3, 0.7, -0.2]).unwrap();
        for (a, b) in scalar.iter().zip(dispatched.iter()) {
            assert!((a - b).abs() < 1e-4);
        }
    }
}
