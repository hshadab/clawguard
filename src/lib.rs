//! ClawGuard library — enforcement, models, encoding, proving, and policy rules
//! for gating agent actions with zero-knowledge proofs.

pub mod encoding;
pub mod enforcement;
pub mod models;
pub mod onnx_support;
pub mod proving;
pub mod rules;

use eyre::{bail, Result};
use onnx_tracer::graph::model::Model;
use onnx_tracer::tensor::Tensor;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::PathBuf;
use std::fs;

// ---------------------------------------------------------------------------
// Config types
// ---------------------------------------------------------------------------

#[derive(Deserialize, Clone, Debug, Default)]
pub struct GuardsConfig {
    pub models: Option<HashMap<String, ModelConfig>>,
    pub settings: Option<SettingsConfig>,
    pub rules: Option<Vec<rules::PolicyRuleConfig>>,
}

#[derive(Deserialize, Clone, Debug)]
pub struct ModelConfig {
    pub path: Option<String>,
    pub meta: Option<String>,
    pub actions: Vec<String>,
}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct SettingsConfig {
    pub require_proof: Option<bool>,
    pub proof_dir: Option<String>,
    pub history_dir: Option<String>,
    pub deny_on_error: Option<bool>,
    pub enforcement: Option<String>,
    pub max_trace_length: Option<usize>,
}

// ---------------------------------------------------------------------------
// Directory helpers
// ---------------------------------------------------------------------------

pub fn config_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".openclaw")
        .join("clawguard")
}

pub fn proof_dir(config: Option<&GuardsConfig>) -> PathBuf {
    config
        .and_then(|c| c.settings.as_ref())
        .and_then(|s| s.proof_dir.as_ref())
        .map(|p| {
            let expanded =
                p.replace('~', &dirs::home_dir().unwrap_or_default().to_string_lossy());
            PathBuf::from(expanded)
        })
        .unwrap_or_else(|| config_dir().join("proofs"))
}

pub fn history_path(config: Option<&GuardsConfig>) -> PathBuf {
    config
        .and_then(|c| c.settings.as_ref())
        .and_then(|s| s.history_dir.as_ref())
        .map(|p| {
            let expanded =
                p.replace('~', &dirs::home_dir().unwrap_or_default().to_string_lossy());
            PathBuf::from(expanded).join("history.jsonl")
        })
        .unwrap_or_else(|| config_dir().join("history.jsonl"))
}

pub fn load_config() -> Option<GuardsConfig> {
    let config_path = config_dir().join("config.toml");
    let content = fs::read_to_string(config_path).ok()?;
    toml::from_str(&content).ok()
}

// ---------------------------------------------------------------------------
// Deny-decision check (Issue 2: centralized, not scattered magic literals)
// ---------------------------------------------------------------------------

pub fn is_deny_decision(label: &str) -> bool {
    matches!(label, "DENIED" | "PII_DETECTED" | "OUT_OF_SCOPE")
}

// ---------------------------------------------------------------------------
// Model hash
// ---------------------------------------------------------------------------

pub fn hash_model_fn(model_fn: fn() -> Model) -> String {
    let model = model_fn();
    let bytecode = onnx_tracer::decode_model(model);
    // Use serde_json for deterministic serialization instead of Debug formatting
    let serialized = serde_json::to_vec(&bytecode).unwrap_or_else(|_| format!("{:?}", bytecode).into_bytes());
    let hash = Sha256::digest(&serialized);
    format!("sha256:{}", hex::encode(hash))
}

pub fn hash_bytes(data: &[u8]) -> String {
    let hash = Sha256::digest(data);
    format!("sha256:{}", hex::encode(hash))
}

// ---------------------------------------------------------------------------
// GuardModel
// ---------------------------------------------------------------------------

pub enum GuardModel {
    ActionGatekeeper,
    PiiShield,
    ScopeGuard,
    PolicyRules,
    Onnx {
        path: PathBuf,
        meta: onnx_support::OnnxModelMeta,
    },
}

impl GuardModel {
    /// Resolve a CLI argument to a GuardModel. Tries known names first, then
    /// falls back to loading as an ONNX file path.
    pub fn from_cli_arg(arg: &str) -> Result<Self> {
        // Try known model names first
        if let Ok(m) = Self::from_name(arg) {
            return Ok(m);
        }
        // Try as a file path
        let path = PathBuf::from(arg);
        if path.is_file() {
            let meta_path = path.with_extension("meta.toml");
            let meta = if meta_path.is_file() {
                onnx_support::OnnxModelMeta::load(&meta_path)?
            } else {
                // Try .onnx -> .meta.toml (strip .onnx, add .meta.toml)
                let alt = path.with_file_name(
                    format!(
                        "{}.meta.toml",
                        path.file_stem().and_then(|s| s.to_str()).unwrap_or("model")
                    )
                );
                if alt.is_file() {
                    onnx_support::OnnxModelMeta::load(&alt)?
                } else {
                    bail!(
                        "ONNX file found at '{}' but no .meta.toml sidecar. Create {} or {}",
                        arg, meta_path.display(), alt.display()
                    );
                }
            };
            Ok(Self::Onnx { path, meta })
        } else {
            bail!(
                "Unknown model '{}'. Use a built-in name (action-gatekeeper, pii-shield, \
                 scope-guard, policy-rules) or a path to an ONNX file.",
                arg
            );
        }
    }

    pub fn from_name(name: &str) -> Result<Self> {
        if name.contains("action") || name.contains("gatekeeper") {
            Ok(Self::ActionGatekeeper)
        } else if name.contains("pii") || name.contains("shield") {
            Ok(Self::PiiShield)
        } else if name.contains("scope") {
            Ok(Self::ScopeGuard)
        } else if name.contains("policy") || name.contains("rule") {
            Ok(Self::PolicyRules)
        } else {
            bail!(
                "Unknown model name '{}'. Use: action-gatekeeper, pii-shield, scope-guard, or policy-rules",
                name
            );
        }
    }

    pub fn model_fn(&self) -> fn() -> Model {
        match self {
            Self::ActionGatekeeper => models::action_gatekeeper::action_gatekeeper_model,
            Self::PiiShield => models::pii_shield::pii_shield_model,
            Self::ScopeGuard => models::scope_guard::scope_guard_model,
            Self::PolicyRules => rules::policy_model,
            Self::Onnx { path, .. } => {
                onnx_support::set_onnx_path(path.clone());
                onnx_support::load_onnx_model
            }
        }
    }

    pub fn encode(&self, action: &str, context: &str) -> Vec<i32> {
        match self {
            Self::ActionGatekeeper => encoding::encode_action(action, context),
            Self::PiiShield => encoding::encode_pii(context),
            Self::ScopeGuard => encoding::encode_scope(action, context),
            Self::PolicyRules => {
                let compiled = rules::COMPILED_POLICY.get();
                let input_width = compiled.map(|c| c.input_width).unwrap_or(8);
                encoding::encode_policy(action, context, compiled, input_width)
            }
            Self::Onnx { meta, .. } => {
                let width = meta.input_shape.last().copied().unwrap_or(8);
                match meta.encoding.as_str() {
                    "pii" => {
                        let mut v = encoding::encode_pii(context);
                        v.resize(width, 0);
                        v
                    }
                    "scope" => {
                        let mut v = encoding::encode_scope(action, context);
                        v.resize(width, 0);
                        v
                    }
                    "raw" => {
                        serde_json::from_str::<Vec<i32>>(context).unwrap_or_else(|_| vec![0; width])
                    }
                    _ => {
                        let mut v = encoding::encode_action(action, context);
                        v.resize(width, 0);
                        v
                    }
                }
            }
        }
    }

    pub fn labels(&self) -> Vec<String> {
        match self {
            Self::ActionGatekeeper => vec!["DENIED".into(), "APPROVED".into()],
            Self::PiiShield => vec!["PII_DETECTED".into(), "CLEAN".into()],
            Self::ScopeGuard => vec!["OUT_OF_SCOPE".into(), "IN_SCOPE".into()],
            Self::PolicyRules => vec!["DENIED".into(), "APPROVED".into()],
            Self::Onnx { meta, .. } => meta.labels.clone(),
        }
    }

    pub fn model_hash(&self) -> String {
        match self {
            Self::Onnx { path, .. } => {
                if let Ok(bytes) = fs::read(path) {
                    hash_bytes(&bytes)
                } else {
                    "sha256:unknown".to_string()
                }
            }
            _ => hash_model_fn(self.model_fn()),
        }
    }

    pub fn input_width(&self) -> usize {
        match self {
            Self::Onnx { meta, .. } => meta.input_shape.last().copied().unwrap_or(8),
            Self::PolicyRules => {
                rules::COMPILED_POLICY.get().map(|c| c.input_width).unwrap_or(8)
            }
            _ => 8,
        }
    }

    pub fn max_trace_length(&self) -> usize {
        match self {
            Self::Onnx { meta, .. } => meta.max_trace_length.unwrap_or(1 << 14),
            _ => 1 << 14,
        }
    }

    /// Returns the action types this model is relevant for (Issue 9).
    pub fn applicable_actions(&self) -> &[&str] {
        match self {
            Self::ActionGatekeeper => &["run_command", "send_email", "write_file", "network_request"],
            Self::PiiShield => &["send_email", "run_command"],
            Self::ScopeGuard => &["read_file", "write_file"],
            Self::PolicyRules => &["run_command", "send_email", "read_file", "write_file", "network_request"],
            Self::Onnx { .. } => &["run_command", "send_email", "read_file", "write_file", "network_request"],
        }
    }
}

// ---------------------------------------------------------------------------
// Core guardrail runner
// ---------------------------------------------------------------------------

pub fn run_guardrail(
    guard: &GuardModel,
    action: &str,
    context: &str,
    generate_proof: bool,
    config: Option<&GuardsConfig>,
) -> Result<(String, f64, String, Option<PathBuf>)> {
    let model_fn = guard.model_fn();
    let model_hash = guard.model_hash();
    let labels = guard.labels();

    let input_vec = guard.encode(action, context);
    let width = input_vec.len();
    let input = Tensor::new(Some(&input_vec), &[1, width])
        .map_err(|e| eyre::eyre!("tensor error: {:?}", e))?;

    let model = model_fn();
    let result = model
        .forward(&[input.clone()])
        .map_err(|e| eyre::eyre!("forward error: {}", e))?;
    let output = result.outputs[0].clone();
    let data = &output.inner;

    let (decision, confidence) = if data.len() >= 2 {
        let margin = (data[0] - data[1]).abs();
        let conf = (margin as f64 / 128.0).min(1.0);
        if data[0] > data[1] {
            (labels.first().cloned().unwrap_or_default(), conf)
        } else {
            (labels.get(1).cloned().unwrap_or_default(), conf)
        }
    } else {
        ("UNKNOWN".to_string(), 0.0)
    };

    let proof_path = if generate_proof {
        let dir = proof_dir(config);
        let trace_len = guard.max_trace_length();
        let (path, _program_io) =
            proving::prove_and_save(model_fn, &input, &dir, &model_hash, trace_len)?;
        Some(path)
    } else {
        None
    };

    Ok((decision, confidence, model_hash, proof_path))
}
