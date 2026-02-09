//! Feature encoding for guardrail model inputs.
//! Converts action/context strings into fixed-size i32 vectors scaled for the MLP.

use crate::action::ActionType;
use crate::patterns::{CC_RE, EMAIL_RE, PHONE_RE, SSN_RE};
use crate::rules::CompiledPolicy;
use eyre::{Result, WrapErr};

const SCALE_MULTIPLIER: i32 = 128; // 2^7, matching scale=7

// ---------------------------------------------------------------------------
// Error-handling encoding functions (return Result)
// ---------------------------------------------------------------------------

/// Encode an action + context into an [1,8] feature vector for action-gatekeeper.
///
/// Features:
/// 0-4: one-hot action type (run_command, send_email, read_file, write_file, network_request)
/// 5: has_sudo
/// 6: targets_dotfile
/// 7: has_pipe_redirect
///
/// Returns an error if context JSON parsing fails.
pub fn encode_action_checked(action: &str, context: &str) -> Result<Vec<i32>> {
    let ctx: serde_json::Value = serde_json::from_str(context)
        .wrap_err_with(|| format!("encode_action: failed to parse context JSON"))?;

    let command = ctx
        .get("command")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let path = ctx.get("path").and_then(|v| v.as_str()).unwrap_or("");
    let full_context = format!("{} {} {}", command, path, context);

    let mut features = vec![0i32; 8];

    // One-hot action type via ActionType enum
    if let Some(at) = ActionType::from_str_opt(action) {
        features[at.one_hot_index()] = SCALE_MULTIPLIER;
    }

    // Binary features from context
    if full_context.contains("sudo") || full_context.contains("su -c") {
        features[5] = SCALE_MULTIPLIER;
    }
    if full_context.contains("/.") || full_context.contains("dotfile") {
        features[6] = SCALE_MULTIPLIER;
    }
    if full_context.contains('|') || full_context.contains('>') || full_context.contains('<') {
        features[7] = SCALE_MULTIPLIER;
    }

    Ok(features)
}

/// Encode text into an [1,8] feature vector for pii-shield.
///
/// Features:
/// 0: SSN pattern count (scaled)
/// 1: email pattern count (scaled)
/// 2: phone pattern count (scaled)
/// 3: credit card pattern count (scaled)
/// 4: password keyword flag
/// 5: secret/token keyword flag
/// 6: digit density (scaled)
/// 7: text length bucket
///
/// This function always succeeds for text input.
pub fn encode_pii_checked(text: &str) -> Result<Vec<i32>> {
    let mut features = vec![0i32; 8];

    features[0] = (SSN_RE.find_iter(text).count() as i32).min(1) * SCALE_MULTIPLIER;
    features[1] = (EMAIL_RE.find_iter(text).count() as i32).min(1) * SCALE_MULTIPLIER;
    features[2] = (PHONE_RE.find_iter(text).count() as i32).min(1) * SCALE_MULTIPLIER;
    features[3] = (CC_RE.find_iter(text).count() as i32).min(1) * SCALE_MULTIPLIER;

    // Keyword flags
    let lower = text.to_lowercase();
    if lower.contains("password") || lower.contains("passwd") {
        features[4] = SCALE_MULTIPLIER;
    }
    if lower.contains("secret") || lower.contains("token") || lower.contains("api_key") || lower.contains("apikey") {
        features[5] = SCALE_MULTIPLIER;
    }

    // Digit density
    let digit_count = text.chars().filter(|c| c.is_ascii_digit()).count();
    let density = if text.is_empty() {
        0.0
    } else {
        digit_count as f64 / text.len() as f64
    };
    features[6] = (density * SCALE_MULTIPLIER as f64) as i32;

    // Text length bucket (0=short, 64=medium, 128=long)
    features[7] = match text.len() {
        0..=50 => 20,
        51..=200 => 64,
        _ => SCALE_MULTIPLIER,
    };

    Ok(features)
}

/// Encode action + context into an [1,8] feature vector for scope-guard.
///
/// Features:
/// 0: path_in_workspace
/// 1: has_dotdot
/// 2: path_depth (scaled)
/// 3: targets_system_dir
/// 4: targets_home_outside_workspace
/// 5: targets_sensitive_dotfile
/// 6: is_absolute
/// 7: path_length_bucket
///
/// Returns an error if context JSON parsing fails.
pub fn encode_scope_checked(_action: &str, context: &str) -> Result<Vec<i32>> {
    let ctx: serde_json::Value = serde_json::from_str(context)
        .wrap_err_with(|| format!("encode_scope: failed to parse context JSON"))?;

    let path = ctx
        .get("path")
        .or_else(|| ctx.get("command"))
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let workspace = ctx
        .get("workspace")
        .and_then(|v| v.as_str())
        .unwrap_or(".");

    let mut features = vec![0i32; 8];

    // path_in_workspace
    if !path.is_empty() && (path.starts_with(workspace) || (!path.starts_with('/') && !path.contains(".."))) {
        features[0] = SCALE_MULTIPLIER;
    }

    // has_dotdot
    if path.contains("..") {
        features[1] = SCALE_MULTIPLIER;
    }

    // path_depth
    let depth = path.split('/').filter(|s| !s.is_empty()).count();
    features[2] = ((depth as i32) * 10).min(SCALE_MULTIPLIER);

    // targets_system_dir
    let system_dirs = ["/etc", "/usr", "/bin", "/sbin", "/var", "/sys", "/proc", "/dev", "/boot"];
    if system_dirs.iter().any(|d| path.starts_with(d)) {
        features[3] = SCALE_MULTIPLIER;
    }

    // targets_home_outside_workspace
    if path.starts_with("/home") || path.starts_with('~') {
        if !path.starts_with(workspace) {
            features[4] = SCALE_MULTIPLIER;
        }
    }

    // targets_sensitive_dotfile
    let sensitive = [".ssh", ".gnupg", ".aws", ".env", ".git/config", ".npmrc", ".pypirc"];
    if sensitive.iter().any(|s| path.contains(s)) {
        features[5] = SCALE_MULTIPLIER;
    }

    // is_absolute
    if path.starts_with('/') {
        features[6] = SCALE_MULTIPLIER;
    }

    // path_length_bucket
    features[7] = match path.len() {
        0..=20 => 20,
        21..=80 => 64,
        _ => SCALE_MULTIPLIER,
    };

    Ok(features)
}

// ---------------------------------------------------------------------------
// Fallback wrappers (legacy API - use deny-safe defaults on error)
// ---------------------------------------------------------------------------

/// Default feature vector for action encoding (deny-safe: sudo + pipe flags set)
fn deny_safe_action_features() -> Vec<i32> {
    let mut features = vec![0i32; 8];
    features[5] = SCALE_MULTIPLIER; // has_sudo
    features[7] = SCALE_MULTIPLIER; // has_pipe_redirect
    features
}

/// Default feature vector for PII encoding (deny-safe: flag all PII types)
fn deny_safe_pii_features() -> Vec<i32> {
    let mut features = vec![0i32; 8];
    features[0] = SCALE_MULTIPLIER; // SSN
    features[4] = SCALE_MULTIPLIER; // password keyword
    features[5] = SCALE_MULTIPLIER; // secret/token
    features
}

/// Default feature vector for scope encoding (deny-safe: system dir + absolute)
fn deny_safe_scope_features() -> Vec<i32> {
    let mut features = vec![0i32; 8];
    features[1] = SCALE_MULTIPLIER; // has_dotdot
    features[3] = SCALE_MULTIPLIER; // targets_system_dir
    features[6] = SCALE_MULTIPLIER; // is_absolute
    features
}

/// Encode action with fallback on error. If `deny_on_error` is true, returns
/// deny-safe defaults; otherwise returns all-zeros.
pub fn encode_action_or_default(action: &str, context: &str, deny_on_error: bool) -> Vec<i32> {
    match encode_action_checked(action, context) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("WARNING: encode_action: {}", e);
            if deny_on_error {
                deny_safe_action_features()
            } else {
                vec![0i32; 8]
            }
        }
    }
}

/// Encode PII with fallback on error.
pub fn encode_pii_or_default(text: &str, deny_on_error: bool) -> Vec<i32> {
    match encode_pii_checked(text) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("WARNING: encode_pii: {}", e);
            if deny_on_error {
                deny_safe_pii_features()
            } else {
                vec![0i32; 8]
            }
        }
    }
}

/// Encode scope with fallback on error.
pub fn encode_scope_or_default(action: &str, context: &str, deny_on_error: bool) -> Vec<i32> {
    match encode_scope_checked(action, context) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("WARNING: encode_scope: {}", e);
            if deny_on_error {
                deny_safe_scope_features()
            } else {
                vec![0i32; 8]
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Legacy API (backward compatible - always succeeds with warnings)
// ---------------------------------------------------------------------------

/// Legacy encode_action that never fails (logs warnings on parse errors).
/// Use `encode_action_checked` for explicit error handling.
pub fn encode_action(action: &str, context: &str) -> Vec<i32> {
    encode_action_or_default(action, context, false)
}

/// Legacy encode_pii that never fails.
pub fn encode_pii(text: &str) -> Vec<i32> {
    encode_pii_or_default(text, false)
}

/// Legacy encode_scope that never fails (logs warnings on parse errors).
/// Use `encode_scope_checked` for explicit error handling.
pub fn encode_scope(action: &str, context: &str) -> Vec<i32> {
    encode_scope_or_default(action, context, false)
}

/// Encode action + context into a feature vector for a compiled policy model.
///
/// Features:
/// 0-4: one-hot action type (same as action-gatekeeper)
/// 5+: one per rule condition (binary: does context match this domain/path/keyword?)
pub fn encode_policy(
    action: &str,
    context: &str,
    policy: Option<&CompiledPolicy>,
    input_width: usize,
) -> Vec<i32> {
    encode_policy_or_default(action, context, policy, input_width, false)
}

/// Encode policy with explicit error handling.
pub fn encode_policy_checked(
    action: &str,
    context: &str,
    policy: Option<&CompiledPolicy>,
    input_width: usize,
) -> Result<Vec<i32>> {
    let mut features = vec![0i32; input_width];

    // One-hot action type (slots 0-4) via ActionType enum
    if let Some(at) = ActionType::from_str_opt(action) {
        features[at.one_hot_index()] = SCALE_MULTIPLIER;
    }

    // Condition slots
    if let Some(policy) = policy {
        let ctx: serde_json::Value = serde_json::from_str(context)
            .wrap_err_with(|| format!("encode_policy: failed to parse context JSON"))?;

        let url = ctx.get("url").and_then(|v| v.as_str()).unwrap_or("");
        let path = ctx.get("path").and_then(|v| v.as_str()).unwrap_or("");
        let command = ctx.get("command").and_then(|v| v.as_str()).unwrap_or("");
        let full_text = format!("{} {} {} {}", url, path, command, context);
        let lower = full_text.to_lowercase();

        for (i, cond) in policy.conditions.iter().enumerate() {
            let slot = 5 + i;
            if slot >= input_width {
                break;
            }
            let matched = match &cond.kind {
                crate::rules::ConditionKind::Domain => {
                    lower.contains(&cond.pattern.to_lowercase())
                }
                crate::rules::ConditionKind::Path => {
                    let pat = cond.pattern.to_lowercase();
                    // Support basic glob: *.ext matches any path ending with .ext
                    if pat.starts_with('*') {
                        let suffix = pat.trim_start_matches('*');
                        lower.contains(&suffix)
                    } else {
                        let cleaned = pat.trim_start_matches('~');
                        lower.contains(cleaned)
                    }
                }
                crate::rules::ConditionKind::Keyword => {
                    lower.contains(&cond.pattern.to_lowercase())
                }
            };
            if matched {
                features[slot] = SCALE_MULTIPLIER;
            }
        }
    }

    Ok(features)
}

/// Encode policy with fallback on error.
pub fn encode_policy_or_default(
    action: &str,
    context: &str,
    policy: Option<&CompiledPolicy>,
    input_width: usize,
    deny_on_error: bool,
) -> Vec<i32> {
    match encode_policy_checked(action, context, policy, input_width) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("WARNING: encode_policy: {}", e);
            if deny_on_error {
                // Deny-safe: set all condition slots to trigger block rules
                let mut features = vec![0i32; input_width];
                for i in 5..input_width {
                    features[i] = SCALE_MULTIPLIER;
                }
                features
            } else {
                vec![0i32; input_width]
            }
        }
    }
}
