//! Enforcement types and the `Guardrail` struct for programmatic usage.

use eyre::Result;
use serde_json::Value;

use crate::{is_deny_decision, load_config, run_guardrail, GuardModel, GuardsConfig};

// ---------------------------------------------------------------------------
// Enforcement level
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnforcementLevel {
    Log,
    Soft,
    Hard,
}

impl std::str::FromStr for EnforcementLevel {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "hard" => Ok(Self::Hard),
            "soft" => Ok(Self::Soft),
            "log" => Ok(Self::Log),
            other => Err(format!("unknown enforcement level '{other}', expected log/soft/hard")),
        }
    }
}

// ---------------------------------------------------------------------------
// Decision
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum Decision {
    Allow {
        confidence: f64,
        model_hash: String,
    },
    Deny {
        reason: String,
        confidence: f64,
        model_hash: String,
        overridable: bool,
    },
}

impl Decision {
    pub fn is_deny(&self) -> bool {
        matches!(self, Decision::Deny { .. })
    }

    pub fn is_allow(&self) -> bool {
        matches!(self, Decision::Allow { .. })
    }
}

// ---------------------------------------------------------------------------
// ActionGuard trait
// ---------------------------------------------------------------------------

pub trait ActionGuard {
    fn before_action(&self, action: &str, context: &Value) -> Result<Decision>;
}

// ---------------------------------------------------------------------------
// Guardrail struct
// ---------------------------------------------------------------------------

pub struct Guardrail {
    config: GuardsConfig,
    enforcement: EnforcementLevel,
}

impl Guardrail {
    pub fn new(config: GuardsConfig, enforcement: EnforcementLevel) -> Self {
        Self {
            config,
            enforcement,
        }
    }

    pub fn from_config() -> Self {
        let config = load_config().unwrap_or_default();
        let enforcement = config
            .settings
            .as_ref()
            .and_then(|s| s.enforcement.as_deref())
            .and_then(|s| match s.parse::<EnforcementLevel>() {
                Ok(level) => Some(level),
                Err(e) => {
                    eprintln!("WARNING: {e}, defaulting to Log");
                    None
                }
            })
            .unwrap_or(EnforcementLevel::Log);
        Self {
            config,
            enforcement,
        }
    }

    pub fn enforcement(&self) -> EnforcementLevel {
        self.enforcement
    }

    pub fn check(
        &self,
        model: &GuardModel,
        action: &str,
        context: &Value,
    ) -> Result<Decision> {
        let deny_on_error = self
            .config
            .settings
            .as_ref()
            .and_then(|s| s.deny_on_error)
            .unwrap_or(false);

        let require_proof = self
            .config
            .settings
            .as_ref()
            .and_then(|s| s.require_proof)
            .unwrap_or(false);

        let context_str = serde_json::to_string(context).unwrap_or_default();

        let result = run_guardrail(model, action, &context_str, require_proof, Some(&self.config));

        match result {
            Ok((decision, confidence, model_hash, _proof_path)) => {
                if is_deny_decision(&decision) {
                    let overridable = self.enforcement != EnforcementLevel::Hard;

                    Ok(Decision::Deny {
                        reason: decision,
                        confidence,
                        model_hash,
                        overridable,
                    })
                } else {
                    Ok(Decision::Allow {
                        confidence,
                        model_hash,
                    })
                }
            }
            Err(e) => {
                if deny_on_error {
                    Ok(Decision::Deny {
                        reason: format!("inference error: {e}"),
                        confidence: 0.0,
                        model_hash: String::new(),
                        overridable: self.enforcement != EnforcementLevel::Hard,
                    })
                } else {
                    Err(e)
                }
            }
        }
    }
}

impl ActionGuard for Guardrail {
    fn before_action(&self, action: &str, context: &Value) -> Result<Decision> {
        let mut most_restrictive: Option<Decision> = None;

        let builtin_models = [
            GuardModel::ActionGatekeeper,
            GuardModel::PiiShield,
            GuardModel::ScopeGuard,
        ];

        // Filter models to only those whose applicable actions include this action type
        let models_to_check: Vec<&GuardModel> = builtin_models
            .iter()
            .filter(|m| m.applicable_actions().contains(&action))
            .collect();

        for model in models_to_check {
            let decision = self.check(model, action, context)?;
            match (&most_restrictive, &decision) {
                (None, _) => most_restrictive = Some(decision),
                (Some(Decision::Allow { .. }), Decision::Deny { .. }) => {
                    most_restrictive = Some(decision);
                }
                _ => {}
            }
        }

        Ok(most_restrictive.unwrap_or(Decision::Allow {
            confidence: 0.0,
            model_hash: String::new(),
        }))
    }
}
