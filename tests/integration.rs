//! Integration test: full prove-verify round trip for the action-gatekeeper model.
//!
//! Run with: cargo test -p clawguard --test integration --release

use ark_bn254::Fr;
use jolt_core::poly::commitment::dory::DoryCommitmentScheme;
use jolt_core::transcripts::KeccakTranscript;
use onnx_tracer::tensor::Tensor;
use zkml_jolt_core::jolt::JoltSNARK;

use clawguard::models::action_gatekeeper::action_gatekeeper_model;

type PCS = DoryCommitmentScheme;
type Snark = JoltSNARK<Fr, PCS, KeccakTranscript>;

#[test]
fn test_action_gatekeeper_inference() {
    let model = action_gatekeeper_model();
    // Dangerous: run_command + sudo + pipe
    let input = Tensor::new(Some(&[128, 0, 0, 0, 0, 128, 0, 128]), &[1, 8]).unwrap();
    let result = model.forward(&[input]).unwrap();
    let out = &result.outputs[0].inner;
    assert!(out[0] > out[1], "Expected deny > allow");
}

#[test]
fn test_action_gatekeeper_prove_verify() {
    let max_trace_length = 1 << 14;

    let preprocessing = Snark::prover_preprocess(action_gatekeeper_model, max_trace_length);

    // Dangerous input
    let input = Tensor::new(Some(&[128, 0, 0, 0, 0, 128, 0, 128]), &[1, 8]).unwrap();

    let (snark, program_io, _debug_info) =
        Snark::prove(&preprocessing, action_gatekeeper_model, &input);

    let verifier_preprocessing = (&preprocessing).into();
    snark
        .verify(&verifier_preprocessing, program_io, None)
        .expect("proof should verify");
}

#[test]
fn test_enforcement_library() {
    use clawguard::enforcement::{Decision, EnforcementLevel, Guardrail};
    use clawguard::{GuardModel, GuardsConfig};

    let config = GuardsConfig::default();
    let guardrail = Guardrail::new(config, EnforcementLevel::Hard);

    let model = GuardModel::ActionGatekeeper;
    let context = serde_json::json!({
        "command": "sudo rm -rf /"
    });

    let decision = guardrail.check(&model, "run_command", &context).unwrap();
    assert!(decision.is_deny(), "Expected deny for sudo rm -rf /");

    match &decision {
        Decision::Deny { overridable, .. } => {
            assert!(!overridable, "Hard enforcement should not be overridable");
        }
        _ => panic!("Expected Deny"),
    }
}
