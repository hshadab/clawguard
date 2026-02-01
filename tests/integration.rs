//! Integration tests for ClawGuard guardrail models and library.
//!
//! Run with: cargo test -p clawguard --test integration --release

use ark_bn254::Fr;
use jolt_core::poly::commitment::dory::DoryCommitmentScheme;
use jolt_core::transcripts::KeccakTranscript;
use onnx_tracer::tensor::Tensor;
use zkml_jolt_core::jolt::JoltSNARK;

use clawguard::models::action_gatekeeper::action_gatekeeper_model;
use clawguard::models::pii_shield::pii_shield_model;
use clawguard::models::scope_guard::scope_guard_model;

type PCS = DoryCommitmentScheme;
type Snark = JoltSNARK<Fr, PCS, KeccakTranscript>;

// ---------------------------------------------------------------------------
// Action Gatekeeper tests
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// PII Shield tests
// ---------------------------------------------------------------------------

#[test]
fn test_pii_shield_detects_ssn() {
    let model = pii_shield_model();
    // SSN match=128
    let input = Tensor::new(Some(&[128, 0, 0, 0, 0, 0, 0, 0]), &[1, 8]).unwrap();
    let result = model.forward(&[input]).unwrap();
    let out = &result.outputs[0].inner;
    assert!(out[0] > out[1], "Expected PII_DETECTED for SSN, got pii={} clean={}", out[0], out[1]);
}

#[test]
fn test_pii_shield_detects_cc() {
    let model = pii_shield_model();
    // CC match=128
    let input = Tensor::new(Some(&[0, 0, 0, 128, 0, 0, 0, 0]), &[1, 8]).unwrap();
    let result = model.forward(&[input]).unwrap();
    let out = &result.outputs[0].inner;
    assert!(out[0] > out[1], "Expected PII_DETECTED for CC, got pii={} clean={}", out[0], out[1]);
}

#[test]
fn test_pii_shield_detects_password_keyword() {
    let model = pii_shield_model();
    // password keyword=128
    let input = Tensor::new(Some(&[0, 0, 0, 0, 128, 0, 0, 64]), &[1, 8]).unwrap();
    let result = model.forward(&[input]).unwrap();
    let out = &result.outputs[0].inner;
    assert!(out[0] > out[1], "Expected PII_DETECTED for password keyword, got pii={} clean={}", out[0], out[1]);
}

#[test]
fn test_pii_shield_allows_clean() {
    let model = pii_shield_model();
    // No PII signals, some text length
    let input = Tensor::new(Some(&[0, 0, 0, 0, 0, 0, 0, 64]), &[1, 8]).unwrap();
    let result = model.forward(&[input]).unwrap();
    let out = &result.outputs[0].inner;
    assert!(out[1] > out[0], "Expected CLEAN, got pii={} clean={}", out[0], out[1]);
}

// ---------------------------------------------------------------------------
// Scope Guard tests
// ---------------------------------------------------------------------------

#[test]
fn test_scope_guard_blocks_system_dir() {
    let model = scope_guard_model();
    // targets_system_dir=128, is_absolute=128
    let input = Tensor::new(Some(&[0, 0, 30, 128, 0, 0, 128, 30]), &[1, 8]).unwrap();
    let result = model.forward(&[input]).unwrap();
    let out = &result.outputs[0].inner;
    assert!(out[0] > out[1], "Expected OUT_OF_SCOPE for system dir, got out={} in={}", out[0], out[1]);
}

#[test]
fn test_scope_guard_blocks_traversal() {
    let model = scope_guard_model();
    // has_dotdot=128, path_depth=80
    let input = Tensor::new(Some(&[0, 128, 80, 0, 0, 0, 0, 64]), &[1, 8]).unwrap();
    let result = model.forward(&[input]).unwrap();
    let out = &result.outputs[0].inner;
    assert!(out[0] > out[1], "Expected OUT_OF_SCOPE for traversal, got out={} in={}", out[0], out[1]);
}

#[test]
fn test_scope_guard_blocks_sensitive_dotfile() {
    let model = scope_guard_model();
    // home_outside_workspace=128, sensitive_dotfile=128
    let input = Tensor::new(Some(&[0, 0, 30, 0, 128, 128, 0, 30]), &[1, 8]).unwrap();
    let result = model.forward(&[input]).unwrap();
    let out = &result.outputs[0].inner;
    assert!(out[0] > out[1], "Expected OUT_OF_SCOPE for sensitive dotfile, got out={} in={}", out[0], out[1]);
}

#[test]
fn test_scope_guard_allows_workspace() {
    let model = scope_guard_model();
    // in_workspace=128, relative, low depth
    let input = Tensor::new(Some(&[128, 0, 20, 0, 0, 0, 0, 20]), &[1, 8]).unwrap();
    let result = model.forward(&[input]).unwrap();
    let out = &result.outputs[0].inner;
    assert!(out[1] > out[0], "Expected IN_SCOPE for workspace path, got out={} in={}", out[0], out[1]);
}

// ---------------------------------------------------------------------------
// Encoding tests
// ---------------------------------------------------------------------------

#[test]
fn test_encode_action_sudo_pipe() {
    let features = clawguard::encoding::encode_action(
        "run_command",
        r#"{"command": "sudo cat /etc/passwd | grep root"}"#,
    );
    assert_eq!(features[0], 128, "run_command should be set");
    assert_eq!(features[5], 128, "has_sudo should be set");
    assert_eq!(features[7], 128, "has_pipe should be set");
}

#[test]
fn test_encode_action_safe_read() {
    let features = clawguard::encoding::encode_action(
        "read_file",
        r#"{"path": "README.md"}"#,
    );
    assert_eq!(features[2], 128, "read_file should be set");
    assert_eq!(features[5], 0, "has_sudo should not be set");
}

#[test]
fn test_encode_action_unknown_type() {
    let features = clawguard::encoding::encode_action(
        "unknown_action",
        r#"{"command": "ls"}"#,
    );
    // All action one-hot slots should be 0
    for i in 0..5 {
        assert_eq!(features[i], 0, "slot {} should be 0 for unknown action", i);
    }
}

#[test]
fn test_encode_pii_with_ssn() {
    let features = clawguard::encoding::encode_pii("My SSN is 123-45-6789");
    assert_eq!(features[0], 128, "SSN pattern should be detected");
}

#[test]
fn test_encode_pii_with_email() {
    let features = clawguard::encoding::encode_pii("Contact me at user@example.com");
    assert_eq!(features[1], 128, "Email pattern should be detected");
}

#[test]
fn test_encode_pii_clean_text() {
    let features = clawguard::encoding::encode_pii("Hello world, this is a normal message.");
    assert_eq!(features[0], 0, "No SSN");
    assert_eq!(features[1], 0, "No email");
    assert_eq!(features[2], 0, "No phone");
    assert_eq!(features[3], 0, "No CC");
    assert_eq!(features[4], 0, "No password keyword");
}

#[test]
fn test_encode_scope_system_dir() {
    let features = clawguard::encoding::encode_scope(
        "read_file",
        r#"{"path": "/etc/passwd"}"#,
    );
    assert_eq!(features[3], 128, "targets_system_dir should be set");
    assert_eq!(features[6], 128, "is_absolute should be set");
}

#[test]
fn test_encode_scope_traversal() {
    let features = clawguard::encoding::encode_scope(
        "read_file",
        r#"{"path": "../../../etc/passwd"}"#,
    );
    assert_eq!(features[1], 128, "has_dotdot should be set");
}

#[test]
fn test_encode_scope_workspace_relative() {
    let features = clawguard::encoding::encode_scope(
        "read_file",
        r#"{"path": "src/main.rs"}"#,
    );
    assert_eq!(features[0], 128, "path_in_workspace should be set");
    assert_eq!(features[1], 0, "no dotdot");
    assert_eq!(features[3], 0, "not system dir");
}

// ---------------------------------------------------------------------------
// ActionType enum tests
// ---------------------------------------------------------------------------

#[test]
fn test_action_type_round_trip() {
    use clawguard::action::ActionType;
    for at in ActionType::ALL {
        let s = at.as_str();
        let parsed = ActionType::from_str_opt(s).unwrap();
        assert_eq!(*at, parsed);
    }
}

#[test]
fn test_action_type_unknown() {
    use clawguard::action::ActionType;
    assert!(ActionType::from_str_opt("unknown_action").is_none());
}

// ---------------------------------------------------------------------------
// Enforcement library tests
// ---------------------------------------------------------------------------

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

#[test]
fn test_enforcement_soft_is_overridable() {
    use clawguard::enforcement::{Decision, EnforcementLevel, Guardrail};
    use clawguard::{GuardModel, GuardsConfig};

    let config = GuardsConfig::default();
    let guardrail = Guardrail::new(config, EnforcementLevel::Soft);

    let model = GuardModel::ActionGatekeeper;
    let context = serde_json::json!({
        "command": "sudo rm -rf /"
    });

    let decision = guardrail.check(&model, "run_command", &context).unwrap();
    match &decision {
        Decision::Deny { overridable, .. } => {
            assert!(*overridable, "Soft enforcement should be overridable");
        }
        _ => panic!("Expected Deny"),
    }
}

#[test]
fn test_enforcement_allows_safe_read() {
    use clawguard::enforcement::{EnforcementLevel, Guardrail};
    use clawguard::{GuardModel, GuardsConfig};

    let config = GuardsConfig::default();
    let guardrail = Guardrail::new(config, EnforcementLevel::Hard);

    let model = GuardModel::ActionGatekeeper;
    let context = serde_json::json!({
        "command": "cat README.md"
    });

    let decision = guardrail.check(&model, "read_file", &context).unwrap();
    assert!(decision.is_allow(), "Expected allow for safe read");
}

// ---------------------------------------------------------------------------
// Config validation tests
// ---------------------------------------------------------------------------

#[test]
fn test_validate_config_valid() {
    let config = clawguard::GuardsConfig {
        settings: Some(clawguard::SettingsConfig {
            enforcement: Some("hard".into()),
            ..Default::default()
        }),
        ..Default::default()
    };
    let issues = clawguard::validate_config(&config);
    assert!(issues.is_empty(), "Valid config should have no issues: {:?}", issues);
}

#[test]
fn test_validate_config_bad_enforcement() {
    let config = clawguard::GuardsConfig {
        settings: Some(clawguard::SettingsConfig {
            enforcement: Some("invalid_level".into()),
            ..Default::default()
        }),
        ..Default::default()
    };
    let issues = clawguard::validate_config(&config);
    assert!(!issues.is_empty(), "Bad enforcement should produce issues");
    assert!(issues[0].contains("unknown enforcement level"));
}

// ---------------------------------------------------------------------------
// Deny-decision helper tests
// ---------------------------------------------------------------------------

#[test]
fn test_is_deny_decision() {
    assert!(clawguard::is_deny_decision("DENIED"));
    assert!(clawguard::is_deny_decision("PII_DETECTED"));
    assert!(clawguard::is_deny_decision("OUT_OF_SCOPE"));
    assert!(!clawguard::is_deny_decision("APPROVED"));
    assert!(!clawguard::is_deny_decision("CLEAN"));
    assert!(!clawguard::is_deny_decision("IN_SCOPE"));
}

// ---------------------------------------------------------------------------
// History rotation test
// ---------------------------------------------------------------------------

#[test]
fn test_history_rotation() {
    use std::fs;
    use std::io::Write;

    let tmp = std::env::temp_dir().join("clawguard_test_history");
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();
    let hist_file = tmp.join("history.jsonl");

    // Write 100 lines
    {
        let mut f = fs::File::create(&hist_file).unwrap();
        for i in 0..100 {
            writeln!(f, r#"{{"entry": {}}}"#, i).unwrap();
        }
    }

    let config = clawguard::GuardsConfig {
        settings: Some(clawguard::SettingsConfig {
            history_dir: Some(tmp.to_string_lossy().to_string()),
            max_history_bytes: Some(100), // very small to trigger rotation
            ..Default::default()
        }),
        ..Default::default()
    };

    clawguard::rotate_history_if_needed(Some(&config));

    let content = fs::read_to_string(&hist_file).unwrap();
    let lines: Vec<&str> = content.lines().filter(|l| !l.is_empty()).collect();
    assert!(lines.len() < 100, "History should have been rotated, got {} lines", lines.len());
    assert_eq!(lines.len(), 50, "Should keep half the entries");

    let _ = fs::remove_dir_all(&tmp);
}
