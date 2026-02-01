//! Scope Guard model: 2-layer MLP [1,8] -> [1,4] -> [1,2] with ReLU.
//! Blocks file access outside the workspace scope.

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
/// Neuron 0 (safe path): 90 on in_workspace[0], inhibited by danger signals.
///   -40 on has_dotdot[1], -30 on system_dir[3], -40 on home_escape[4].
///   30 bias: paths that are clearly in-workspace start with a positive signal.
///
/// Neuron 1 (traversal danger): 90 on has_dotdot[1], 30 on path_depth[2].
///   Path traversal (../) is the most common escape technique.
///   20 on is_absolute[6]: absolute paths combined with traversal are extra risky.
///
/// Neuron 2 (system target): 90 on targets_system_dir[3], 70 on is_absolute[6].
///   Accessing /etc, /sys, /proc etc. is almost always out of scope.
///
/// Neuron 3 (home escape): 90 on home_outside_workspace[4], 80 on sensitive_dotfile[5].
///   Accessing ~/.ssh, ~/.aws etc. outside workspace is a credential theft vector.
const W1: &[i32] = &[
    90, -40, -20, -30, -40, -30, -20, 0,  // safe path
    -30, 90, 30, 0, 0, 0, 20, 0,          // traversal danger
    -20, 0, 0, 90, 0, 0, 70, 0,           // system target
    -20, 0, 0, 0, 90, 80, 0, 0,           // home escape
];

const B1: &[i32] = &[30, 0, 0, 0]; // safe path starts with positive bias

/// Layer 2 weights: [2 output neurons, 4 hidden neurons]
///
/// Output 0 (OUT_OF_SCOPE): -40 from safe + 70 from traversal + 60 from system + 60 from home.
///   All danger neurons contribute to out-of-scope classification.
///
/// Output 1 (IN_SCOPE): 80 from safe, inhibited by all danger neurons.
///   10 bias: slight default toward in-scope for paths that don't trigger any danger.
const W2: &[i32] = &[
    -40, 70, 60, 60,   // out_of_scope
    80, -50, -40, -40,  // in_scope
];

const B2: &[i32] = &[0, 10]; // slight bias toward in_scope

/// Build the scope-guard model.
///
/// Input [1,8]: path_in_workspace, has_dotdot, path_depth, targets_system_dir,
/// targets_home_outside_workspace, targets_sensitive_dotfile, is_absolute,
/// path_length_bucket.
///
/// Output [1,2]: [out_of_scope, in_scope].
pub fn scope_guard_model() -> Model {
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
    fn test_blocks_etc_passwd() {
        let model = scope_guard_model();
        // not in workspace, targets_system_dir=128, is_absolute=128
        let input = Tensor::new(
            Some(&[0, 0, 30, 128, 0, 0, 128, 30]),
            &[1, 8],
        )
        .unwrap();
        let result = model.forward(&[input]).unwrap();
        let out = result.outputs[0].clone();
        let data = out.inner;
        assert!(data[0] > data[1], "Expected OUT_OF_SCOPE, got out={} in={}", data[0], data[1]);
    }

    #[test]
    fn test_allows_workspace_path() {
        let model = scope_guard_model();
        // in_workspace=128, relative path, low depth
        let input = Tensor::new(
            Some(&[128, 0, 20, 0, 0, 0, 0, 20]),
            &[1, 8],
        )
        .unwrap();
        let result = model.forward(&[input]).unwrap();
        let out = result.outputs[0].clone();
        let data = out.inner;
        assert!(data[1] > data[0], "Expected IN_SCOPE, got out={} in={}", data[0], data[1]);
    }
}
