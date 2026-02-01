//! Scope Guard model: 2-layer MLP [1,8] -> [1,4] -> [1,2] with ReLU.
//! Blocks file access outside the workspace scope.

use onnx_tracer::builder::ModelBuilder;
use onnx_tracer::graph::model::Model;
use onnx_tracer::ops::poly::PolyOp;
use onnx_tracer::tensor::Tensor;

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
    // Neuron 0 (safe path): in_workspace, low depth
    // Neuron 1 (traversal danger): has_dotdot, deep path
    // Neuron 2 (system target): targets_system_dir, is_absolute
    // Neuron 3 (home escape): targets_home_outside, sensitive_dotfile
    let mut w1 = Tensor::new(
        Some(&[
            90, -40, -20, -30, -40, -30, -20, 0,  // safe path
            -30, 90, 30, 0, 0, 0, 20, 0,          // traversal danger
            -20, 0, 0, 90, 0, 0, 70, 0,           // system target
            -20, 0, 0, 0, 90, 80, 0, 0,           // home escape
        ]),
        &[4, 8],
    )
    .unwrap();
    w1.set_scale(SCALE);
    let w1_const = b.const_tensor(w1, vec![4, 8], 1);

    let mm1 = b.matmult(input, w1_const, vec![1, 4], 1);
    let mm1_rescaled = b.div(128, mm1, vec![1, 4], 1);

    let mut b1 = Tensor::new(Some(&[30, 0, 0, 0]), &[1, 4]).unwrap();
    b1.set_scale(SCALE);
    let b1_const = b.const_tensor(b1, vec![1, 4], 1);
    let biased1 = b.poly(PolyOp::Add, mm1_rescaled, b1_const, vec![1, 4], 1);

    let relu1 = b.relu(biased1, vec![1, 4], 1);

    // Layer 2: [1,4] x [2,4] -> [1,2]
    // Swapped rows: output[0] = out_of_scope (deny), output[1] = in_scope (allow)
    let mut w2 = Tensor::new(
        Some(&[
            -40, 70, 60, 60,   // out_of_scope
            80, -50, -40, -40,  // in_scope
        ]),
        &[2, 4],
    )
    .unwrap();
    w2.set_scale(SCALE);
    let w2_const = b.const_tensor(w2, vec![2, 4], 1);

    let mm2 = b.matmult(relu1, w2_const, vec![1, 2], 1);
    let mm2_rescaled = b.div(128, mm2, vec![1, 2], 1);

    let mut b2 = Tensor::new(Some(&[0, 10]), &[1, 2]).unwrap();
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
        // output[0] = out_of_scope, output[1] = in_scope
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
        // output[0] = out_of_scope, output[1] = in_scope
        assert!(data[1] > data[0], "Expected IN_SCOPE, got out={} in={}", data[0], data[1]);
    }
}
