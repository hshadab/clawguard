//! PII Shield model: 2-layer MLP [1,8] -> [1,4] -> [1,2] with ReLU.
//! Detects personally identifiable information patterns in text.

use onnx_tracer::builder::ModelBuilder;
use onnx_tracer::graph::model::Model;
use onnx_tracer::ops::poly::PolyOp;
use onnx_tracer::tensor::Tensor;

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
    // Neuron 0 (structured PII): SSN + CC patterns
    // Neuron 1 (contact PII): email + phone
    // Neuron 2 (secret keywords): password + secret/token flags
    // Neuron 3 (clean signal): low digit density, short text
    let mut w1 = Tensor::new(
        Some(&[
            90, 0, 0, 80, 0, 0, 30, 0,    // structured PII
            0, 70, 80, 0, 0, 0, 20, 0,    // contact PII
            0, 0, 0, 0, 90, 80, 0, 0,     // secret keywords
            -30, -30, -30, -30, -40, -40, -60, 20, // clean signal
        ]),
        &[4, 8],
    )
    .unwrap();
    w1.set_scale(SCALE);
    let w1_const = b.const_tensor(w1, vec![4, 8], 1);

    let mm1 = b.matmult(input, w1_const, vec![1, 4], 1);
    let mm1_rescaled = b.div(128, mm1, vec![1, 4], 1);

    let mut b1 = Tensor::new(Some(&[0, 0, 0, 30]), &[1, 4]).unwrap();
    b1.set_scale(SCALE);
    let b1_const = b.const_tensor(b1, vec![1, 4], 1);
    let biased1 = b.poly(PolyOp::Add, mm1_rescaled, b1_const, vec![1, 4], 1);

    let relu1 = b.relu(biased1, vec![1, 4], 1);

    // Layer 2: [1,4] x [2,4] -> [1,2]
    let mut w2 = Tensor::new(
        Some(&[
            70, 60, 50, -40,  // pii_detected
            -40, -30, -30, 80, // clean
        ]),
        &[2, 4],
    )
    .unwrap();
    w2.set_scale(SCALE);
    let w2_const = b.const_tensor(w2, vec![2, 4], 1);

    let mm2 = b.matmult(relu1, w2_const, vec![1, 2], 1);
    let mm2_rescaled = b.div(128, mm2, vec![1, 2], 1);

    let mut b2 = Tensor::new(Some(&[0, 20]), &[1, 2]).unwrap();
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
