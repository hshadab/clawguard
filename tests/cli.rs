//! CLI integration tests for ClawGuard.
//!
//! These tests verify the CLI commands work correctly by running the binary.
//! Run with: cargo test --test cli

use std::process::Command;

fn clawguard_cmd() -> Command {
    let mut cmd = Command::new("cargo");
    cmd.args(["run", "--", "--quiet"]);
    // Alternatively, build first and use the binary directly:
    // Command::new("./target/debug/clawguard")
    cmd
}

// ---------------------------------------------------------------------------
// Basic CLI tests
// ---------------------------------------------------------------------------

#[test]
fn test_cli_help() {
    let output = Command::new("cargo")
        .args(["run", "--", "--help"])
        .output()
        .expect("Failed to run clawguard");

    assert!(output.status.success(), "Help command should succeed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("clawguard") || stdout.contains("ClawGuard"));
}

#[test]
fn test_cli_models_list() {
    let output = Command::new("cargo")
        .args(["run", "--", "models"])
        .output()
        .expect("Failed to run clawguard models");

    assert!(output.status.success(), "Models command should succeed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("action-gatekeeper"), "Should list action-gatekeeper model");
    assert!(stdout.contains("pii-shield"), "Should list pii-shield model");
    assert!(stdout.contains("scope-guard"), "Should list scope-guard model");
    assert!(stdout.contains("skill-safety"), "Should list skill-safety model");
}

#[test]
fn test_cli_config_check_no_config() {
    let output = Command::new("cargo")
        .args(["run", "--", "config-check"])
        .output()
        .expect("Failed to run clawguard config-check");

    // Should succeed even without a config file
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Either "No config file found" or "All checks passed"
    assert!(
        stdout.contains("No config file") || stdout.contains("All checks passed") || stdout.contains("Config path"),
        "Config check should report status: stdout={}, stderr={}",
        stdout,
        stderr
    );
}

// ---------------------------------------------------------------------------
// Check command tests
// ---------------------------------------------------------------------------

#[test]
fn test_cli_check_action_gatekeeper_safe() {
    let output = Command::new("cargo")
        .args([
            "run", "--",
            "check",
            "--model", "action-gatekeeper",
            "--action", "read_file",
            "--context", r#"{"path": "README.md"}"#,
        ])
        .output()
        .expect("Failed to run check command");

    assert!(output.status.success(), "Safe read should be approved");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("APPROVED"), "Safe action should be APPROVED");
}

#[test]
fn test_cli_check_action_gatekeeper_denies_dangerous() {
    let output = Command::new("cargo")
        .args([
            "run", "--",
            "check",
            "--model", "action-gatekeeper",
            "--action", "run_command",
            "--context", r#"{"command": "sudo rm -rf /"}"#,
        ])
        .output()
        .expect("Failed to run check command");

    // With hard enforcement, dangerous commands should exit with code 1
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("DENIED"), "Dangerous command should be DENIED");
}

#[test]
fn test_cli_check_pii_shield_detects_ssn() {
    let output = Command::new("cargo")
        .args([
            "run", "--",
            "check",
            "--model", "pii-shield",
            "--action", "send_email",
            "--context", "My SSN is 123-45-6789",
        ])
        .output()
        .expect("Failed to run check command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("PII_DETECTED"), "SSN should trigger PII detection");
}

#[test]
fn test_cli_check_scope_guard_blocks_system() {
    let output = Command::new("cargo")
        .args([
            "run", "--",
            "check",
            "--model", "scope-guard",
            "--action", "read_file",
            "--context", r#"{"path": "/etc/passwd"}"#,
        ])
        .output()
        .expect("Failed to run check command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("OUT_OF_SCOPE"), "System file should be out of scope");
}

// ---------------------------------------------------------------------------
// Model name matching tests
// ---------------------------------------------------------------------------

#[test]
fn test_cli_model_name_exact_match() {
    // Test that exact model names work
    for model in &["action-gatekeeper", "pii-shield", "scope-guard", "skill-safety"] {
        let output = Command::new("cargo")
            .args([
                "run", "--",
                "check",
                "--model", model,
                "--action", "read_file",
                "--context", r#"{"path": "test.txt"}"#,
            ])
            .output()
            .expect("Failed to run check command");

        assert!(output.status.success() || output.status.code() == Some(1),
            "Model {} should be recognized", model);
    }
}

#[test]
fn test_cli_model_name_unknown_fails() {
    let output = Command::new("cargo")
        .args([
            "run", "--",
            "check",
            "--model", "unknown-model-name",
            "--action", "read_file",
            "--context", r#"{"path": "test.txt"}"#,
        ])
        .output()
        .expect("Failed to run check command");

    assert!(!output.status.success(), "Unknown model should fail");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Unknown model"), "Should report unknown model error");
}

// ---------------------------------------------------------------------------
// Config validation tests
// ---------------------------------------------------------------------------

#[test]
fn test_cli_config_check_with_ignore_flag() {
    let output = Command::new("cargo")
        .args([
            "run", "--",
            "config-check",
            "--ignore-config-errors",
        ])
        .output()
        .expect("Failed to run config-check");

    // With --ignore-config-errors, should always exit 0
    assert!(output.status.success(), "Config check with ignore flag should always succeed");
}

// ---------------------------------------------------------------------------
// History command tests
// ---------------------------------------------------------------------------

#[test]
fn test_cli_history_empty() {
    let output = Command::new("cargo")
        .args([
            "run", "--",
            "history",
            "--limit", "5",
        ])
        .output()
        .expect("Failed to run history command");

    assert!(output.status.success(), "History command should succeed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should output valid JSON (either empty array or entries)
    assert!(stdout.contains("["), "History should output JSON array");
}

// ---------------------------------------------------------------------------
// Migration command tests
// ---------------------------------------------------------------------------

#[test]
fn test_cli_migrate_proofs_dry_run() {
    let output = Command::new("cargo")
        .args([
            "run", "--",
            "migrate-proofs",
            "--dry-run",
        ])
        .output()
        .expect("Failed to run migrate-proofs command");

    assert!(output.status.success(), "Migrate proofs dry run should succeed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Proof directory") || stdout.contains("model hash"),
        "Should show proof directory info");
}
