//! Action Gatekeeper model: 2-layer MLP [1,8] -> [1,4] -> [1,2] with ReLU.
//! Blocks dangerous command patterns (sudo + pipe, etc), allows safe reads.

use onnx_tracer::builder::ModelBuilder;
use onnx_tracer::graph::model::Model;
use onnx_tracer::ops::poly::PolyOp;
use onnx_tracer::tensor::Tensor;

/// Build the action-gatekeeper model.
///
/// Input [1,8]: one-hot action type (5 slots: run_command, send_email, read_file,
/// write_file, network_request) + 3 binary features (has_sudo, targets_dotfile,
/// has_pipe_redirect).
///
/// Output [1,2]: [deny_score, allow_score].
pub fn action_gatekeeper_model() -> Model {
    const SCALE: i32 = 7;
    let mut b = ModelBuilder::new(SCALE);

    // Input: [1, 8]
    let input = b.input(vec![1, 8], 1);

    // Layer 1: [1,8] x [4,8] -> [1,4]
    // Weights: rows are hidden neurons
    // Neuron 0 (danger): fires on run_command(0) + has_sudo(5) + has_pipe(7)
    // Neuron 1 (email risk): fires on send_email(1) + has_sudo(5)
    // Neuron 2 (write risk): fires on write_file(3) + targets_dotfile(6)
    // Neuron 3 (safety): fires on read_file(2), negative on danger signals
    let mut w1 = Tensor::new(
        Some(&[
            80, 0, 0, 0, 0, 90, 0, 70,   // danger neuron
            0, 60, 0, 0, 0, 50, 0, 0,     // email risk
            0, 0, 0, 70, 0, 0, 80, 0,     // write risk
            -40, 0, 90, -20, 0, -60, 0, -40, // safety neuron
        ]),
        &[4, 8],
    )
    .unwrap();
    w1.set_scale(SCALE);
    let w1_const = b.const_tensor(w1, vec![4, 8], 1);

    let mm1 = b.matmult(input, w1_const, vec![1, 4], 1);
    let mm1_rescaled = b.div(128, mm1, vec![1, 4], 1);

    let mut b1 = Tensor::new(Some(&[0, 0, 0, 40]), &[1, 4]).unwrap();
    b1.set_scale(SCALE);
    let b1_const = b.const_tensor(b1, vec![1, 4], 1);
    let biased1 = b.poly(PolyOp::Add, mm1_rescaled, b1_const, vec![1, 4], 1);

    let relu1 = b.relu(biased1, vec![1, 4], 1);

    // Layer 2: [1,4] x [2,4] -> [1,2]
    // Output neuron 0 (deny): fires on danger, email risk, write risk
    // Output neuron 1 (allow): fires on safety neuron
    let mut w2 = Tensor::new(
        Some(&[
            80, 60, 50, -30,  // deny
            -50, -30, -20, 80, // allow
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
    fn test_blocks_sudo_pipe() {
        let model = action_gatekeeper_model();
        // run_command=128, has_sudo=128, has_pipe_redirect=128
        let input = Tensor::new(
            Some(&[128, 0, 0, 0, 0, 128, 0, 128]),
            &[1, 8],
        )
        .unwrap();
        let result = model.forward(&[input]).unwrap();
        let out = result.outputs[0].clone();
        let data = out.inner;
        // deny_score should be > allow_score
        assert!(data[0] > data[1], "Expected DENIED, got deny={} allow={}", data[0], data[1]);
    }

    #[test]
    fn test_allows_read_file() {
        let model = action_gatekeeper_model();
        // read_file=128, all other features=0
        let input = Tensor::new(
            Some(&[0, 0, 128, 0, 0, 0, 0, 0]),
            &[1, 8],
        )
        .unwrap();
        let result = model.forward(&[input]).unwrap();
        let out = result.outputs[0].clone();
        let data = out.inner;
        // allow_score should be > deny_score
        assert!(data[1] > data[0], "Expected ALLOWED, got deny={} allow={}", data[0], data[1]);
    }
}
