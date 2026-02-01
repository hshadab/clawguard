//! Action Gatekeeper model: 2-layer MLP [1,8] -> [1,4] -> [1,2] with ReLU.
//! Blocks dangerous command patterns (sudo + pipe, etc), allows safe reads.

use onnx_tracer::builder::ModelBuilder;
use onnx_tracer::graph::model::Model;
use onnx_tracer::ops::poly::PolyOp;
use onnx_tracer::tensor::Tensor;

// ---------------------------------------------------------------------------
// Weight constants — documented rationale for each value
// ---------------------------------------------------------------------------

// All weights use fixed-point arithmetic at scale=7 (multiplied by 2^7 = 128).
// Input features are binary (0 or 128). After matmul, results are divided by 128
// to normalize back. Weights in the range 50-90 represent "strong signal" while
// negative weights represent "inhibition".

/// Layer 1 weights: [4 hidden neurons, 8 input features]
///
/// Neuron 0 (danger detector):
///   - 80 on run_command[0]: running commands is the primary danger vector
///   - 90 on has_sudo[5]: sudo escalation is a strong danger signal
///   - 70 on has_pipe[7]: pipe/redirect enables chaining dangerous operations
///
/// Neuron 1 (email risk):
///   - 60 on send_email[1]: email is a data exfiltration vector
///   - 50 on has_sudo[5]: sudo + email is suspicious
///
/// Neuron 2 (write risk):
///   - 70 on write_file[3]: writing files can modify system state
///   - 80 on targets_dotfile[6]: dotfiles control shell/app behavior
///
/// Neuron 3 (safety signal):
///   - 90 on read_file[2]: reading is inherently safe
///   - -40 on run_command[0], -60 on has_sudo[5]: inhibited by danger signals
///   - 40 bias: starts positive so safe actions pass by default
const W1: &[i32] = &[
    80, 0, 0, 0, 0, 90, 0, 70,   // danger neuron
    0, 60, 0, 0, 0, 50, 0, 0,     // email risk
    0, 0, 0, 70, 0, 0, 80, 0,     // write risk
    -40, 0, 90, -20, 0, -60, 0, -40, // safety neuron
];

const B1: &[i32] = &[0, 0, 0, 40]; // safety neuron starts with positive bias

/// Layer 2 weights: [2 output neurons, 4 hidden neurons]
///
/// Output 0 (DENIED):
///   - 80 from danger, 60 from email risk, 50 from write risk: all risks contribute to denial
///   - -30 from safety: safety signal inhibits denial
///
/// Output 1 (APPROVED):
///   - 80 from safety: strong positive from safety neuron
///   - -50 from danger, -30 from email, -20 from write: risks inhibit approval
///   - 20 bias: slight default toward approval for ambiguous cases
const W2: &[i32] = &[
    80, 60, 50, -30,  // deny
    -50, -30, -20, 80, // allow
];

const B2: &[i32] = &[0, 20]; // slight bias toward allow for ambiguous inputs

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
        assert!(data[1] > data[0], "Expected ALLOWED, got deny={} allow={}", data[0], data[1]);
    }
}
