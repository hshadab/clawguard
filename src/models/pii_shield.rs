//! PII Shield model: 2-layer MLP [1,8] -> [1,4] -> [1,2] with ReLU.
//! Detects personally identifiable information patterns in text.

use onnx_tracer::builder::ModelBuilder;
use onnx_tracer::graph::model::Model;
use onnx_tracer::ops::poly::PolyOp;
use onnx_tracer::tensor::Tensor;

// ---------------------------------------------------------------------------
// Weight constants — documented rationale
// ---------------------------------------------------------------------------

// Fixed-point scale=7 (2^7 = 128). See action_gatekeeper.rs for full explanation.

/// Layer 1 weights: [4 hidden neurons, 8 input features]
///
/// Neuron 0 (structured PII): fires on SSN[0]=90 and CC[3]=80 patterns.
///   These are high-value identifiers that should always trigger detection.
///   30 on digit_density[6]: structured PII tends to be digit-heavy.
///
/// Neuron 1 (contact PII): fires on email[1]=70 and phone[2]=80.
///   Contact info is less critical than SSN/CC but still PII.
///   20 on digit_density[6]: phone numbers contribute digits.
///
/// Neuron 2 (secret keywords): fires on password[4]=90 and secret/token[5]=80.
///   Keyword-based detection for credentials that lack regex patterns.
///
/// Neuron 3 (clean signal): inhibited by all PII signals (-30 to -60).
///   20 on text_length[7]: longer texts without PII signals are likely clean.
///   30 bias: starts positive so clean text passes by default.
const W1: &[i32] = &[
    90, 0, 0, 80, 0, 0, 30, 0,    // structured PII
    0, 70, 80, 0, 0, 0, 20, 0,    // contact PII
    0, 0, 0, 0, 90, 80, 0, 0,     // secret keywords
    -30, -30, -30, -30, -40, -40, -60, 20, // clean signal
];

const B1: &[i32] = &[0, 0, 0, 30]; // clean signal starts positive

/// Layer 2 weights: [2 output neurons, 4 hidden neurons]
///
/// Output 0 (PII_DETECTED): 70 from structured + 60 from contact + 50 from secrets.
///   -40 from clean: clean signal inhibits false positives.
///
/// Output 1 (CLEAN): 80 from clean signal, inhibited by all PII neurons.
///   20 bias: slight default toward clean for ambiguous text.
const W2: &[i32] = &[
    70, 60, 50, -40,  // pii_detected
    -40, -30, -30, 80, // clean
];

const B2: &[i32] = &[0, 20]; // slight bias toward clean

/// Build the pii-shield model.
///
/// Input [1,8]: regex match counts (scaled 0-128) for SSN, email, phone, CC
/// patterns + keyword flags (password, secret/token) + digit density + text
/// length bucket.
///
/// Output [1,2]: [pii_detected, clean].
pub fn pii_shield_model() -> Model {
    const SCALE: i32 = 7;
    let mut b = ModelBuilder::new(SCALE);

    let input = b.input(vec![1, 8], 1);

    // Layer 1: [1,8] x [4,8] -> [1,4]
    let mut w1 = Tensor::new(Some(W1), &[4, 8]).unwrap();
    w1.set_scale(SCALE);
    let w1_const = b.const_tensor(w1, vec![4, 8], 1);

    let mm1 = b.matmult(input, w1_const, vec![1, 4], 1);
    let mm1_rescaled = b.div(128, mm1, vec![1, 4], 1);

    let mut b1 = Tensor::new(Some(B1), &[1, 4]).unwrap();
    b1.set_scale(SCALE);
    let b1_const = b.const_tensor(b1, vec![1, 4], 1);
    let biased1 = b.poly(PolyOp::Add, mm1_rescaled, b1_const, vec![1, 4], 1);

    let relu1 = b.relu(biased1, vec![1, 4], 1);

    // Layer 2: [1,4] x [2,4] -> [1,2]
    let mut w2 = Tensor::new(Some(W2), &[2, 4]).unwrap();
    w2.set_scale(SCALE);
    let w2_const = b.const_tensor(w2, vec![2, 4], 1);

    let mm2 = b.matmult(relu1, w2_const, vec![1, 2], 1);
    let mm2_rescaled = b.div(128, mm2, vec![1, 2], 1);

    let mut b2 = Tensor::new(Some(B2), &[1, 2]).unwrap();
    b2.set_scale(SCALE);
    let b2_const = b.const_tensor(b2, vec![1, 2], 1);
    let output = b.poly(PolyOp::Add, mm2_rescaled, b2_const, vec![1, 2], 1);

    b.take(vec![input.0], vec![output])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blocks_ssn_pattern() {
        let model = pii_shield_model();
        // SSN match=128, rest zeros
        let input = Tensor::new(
            Some(&[128, 0, 0, 0, 0, 0, 0, 0]),
            &[1, 8],
        )
        .unwrap();
        let result = model.forward(&[input]).unwrap();
        let out = result.outputs[0].clone();
        let data = out.inner;
        assert!(data[0] > data[1], "Expected PII_DETECTED, got pii={} clean={}", data[0], data[1]);
    }

    #[test]
    fn test_allows_clean_text() {
        let model = pii_shield_model();
        // All zeros = no PII signals, some text length
        let input = Tensor::new(
            Some(&[0, 0, 0, 0, 0, 0, 0, 64]),
            &[1, 8],
        )
        .unwrap();
        let result = model.forward(&[input]).unwrap();
        let out = result.outputs[0].clone();
        let data = out.inner;
        assert!(data[1] > data[0], "Expected CLEAN, got pii={} clean={}", data[0], data[1]);
    }
}
