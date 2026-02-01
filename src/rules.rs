//! User-defined policy rules compiled into model weights for ZK proving.

use eyre::Result;
use onnx_tracer::builder::ModelBuilder;
use onnx_tracer::graph::model::Model;
use onnx_tracer::ops::poly::PolyOp;
use onnx_tracer::tensor::Tensor;
use serde::Deserialize;
use std::sync::OnceLock;

// Max total input width (5 action slots + conditions)
const MAX_INPUT_WIDTH: usize = 32;
const SCALE: i32 = 7;

// ---------------------------------------------------------------------------
// Config parsing (Step 2.1)
// ---------------------------------------------------------------------------

#[derive(Deserialize, Clone, Debug)]
pub struct PolicyRuleConfig {
    pub name: String,
    pub block_domains: Option<Vec<String>>,
    pub block_paths: Option<Vec<String>>,
    pub block_keywords: Option<Vec<String>>,
    pub actions: Option<Vec<String>>,
}

#[derive(Debug, Clone)]
pub struct PolicyRule {
    pub name: String,
    pub block_domains: Vec<String>,
    pub block_paths: Vec<String>,
    pub block_keywords: Vec<String>,
    pub actions: Vec<String>,
}

impl PolicyRule {
    pub fn from_config(cfg: &PolicyRuleConfig) -> Result<Self> {
        Ok(Self {
            name: cfg.name.clone(),
            block_domains: cfg.block_domains.clone().unwrap_or_default(),
            block_paths: cfg.block_paths.clone().unwrap_or_default(),
            block_keywords: cfg.block_keywords.clone().unwrap_or_default(),
            actions: cfg.actions.clone().unwrap_or_else(|| vec!["*".into()]),
        })
    }

    pub fn condition_count(&self) -> usize {
        self.block_domains.len() + self.block_paths.len() + self.block_keywords.len()
    }
}

// ---------------------------------------------------------------------------
// Compiled policy (Step 2.2 + 2.3)
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct CompiledPolicy {
    pub input_width: usize,
    pub num_rules: usize,
    /// Flat condition list: (kind, pattern) for encoding
    pub conditions: Vec<ConditionSlot>,
    /// Layer 1 weights: [num_rules, input_width]
    pub w1: Vec<i32>,
    /// Layer 1 biases: [num_rules]
    pub b1: Vec<i32>,
    /// Layer 2 weights: [2, num_rules]
    pub w2: Vec<i32>,
    /// Layer 2 biases: [2]
    pub b2: Vec<i32>,
}

#[derive(Debug, Clone)]
pub enum ConditionKind {
    Domain,
    Path,
    Keyword,
}

#[derive(Debug, Clone)]
pub struct ConditionSlot {
    pub kind: ConditionKind,
    pub pattern: String,
}

pub static COMPILED_POLICY: OnceLock<CompiledPolicy> = OnceLock::new();

pub fn init_policy(rules: &[PolicyRule]) -> bool {
    let compiled = compile_rules(rules);
    COMPILED_POLICY.set(compiled).is_ok()
}

pub fn compile_rules(rules: &[PolicyRule]) -> CompiledPolicy {
    // Collect all conditions across all rules
    let mut conditions = Vec::new();
    let mut rule_condition_ranges: Vec<(usize, usize)> = Vec::new();

    for rule in rules {
        let start = conditions.len();
        for d in &rule.block_domains {
            conditions.push(ConditionSlot {
                kind: ConditionKind::Domain,
                pattern: d.clone(),
            });
        }
        for p in &rule.block_paths {
            conditions.push(ConditionSlot {
                kind: ConditionKind::Path,
                pattern: p.clone(),
            });
        }
        for k in &rule.block_keywords {
            conditions.push(ConditionSlot {
                kind: ConditionKind::Keyword,
                pattern: k.clone(),
            });
        }
        let end = conditions.len();
        rule_condition_ranges.push((start, end));
    }

    // Input width = 5 (action one-hot) + num_conditions, capped at MAX_INPUT_WIDTH
    let total_conditions = conditions.len();
    let num_conditions = total_conditions.min(MAX_INPUT_WIDTH - 5);
    if total_conditions > num_conditions {
        eprintln!(
            "WARNING: policy rules have {} conditions but max is {}. \
             {} conditions will be ignored. Consider consolidating rules.",
            total_conditions,
            MAX_INPUT_WIDTH - 5,
            total_conditions - num_conditions
        );
    }
    let input_width = 5 + num_conditions;
    let num_rules = rules.len();

    // Layer 1: each rule → one hidden neuron
    // Weights: large positive on its conditions, zero elsewhere
    let mut w1 = vec![0i32; num_rules * input_width];
    let mut b1 = vec![0i32; num_rules];

    for (rule_idx, (start, end)) in rule_condition_ranges.iter().enumerate() {
        // For each condition in this rule, set a large positive weight
        for cond_idx in *start..*end {
            if cond_idx < num_conditions {
                let slot = 5 + cond_idx; // offset past action one-hot
                w1[rule_idx * input_width + slot] = 90;
            }
        }
        // Bias: negative threshold so neuron only fires when at least one condition matches
        // A single match at 128*90/128 = 90 exceeds threshold of -10 → fires
        b1[rule_idx] = -10;
    }

    // Layer 2: [2, num_rules]
    // Output 0 = DENIED: positive weights from all rule neurons
    // Output 1 = APPROVED: negative weights from all rule neurons + positive bias
    let mut w2 = vec![0i32; 2 * num_rules];
    for i in 0..num_rules {
        w2[i] = 80; // deny neuron
        w2[num_rules + i] = -80; // allow neuron (negative from rules)
    }
    let b2 = vec![-20, 60]; // bias: deny needs rule to fire; allow starts high

    CompiledPolicy {
        input_width,
        num_rules,
        conditions,
        w1,
        b1,
        w2,
        b2,
    }
}

// ---------------------------------------------------------------------------
// Policy model builder (Step 2.3)
// ---------------------------------------------------------------------------

pub fn policy_model() -> Model {
    let policy = COMPILED_POLICY
        .get()
        .expect("policy not initialized — call init_policy() first");

    let mut b = ModelBuilder::new(SCALE);
    let nr = policy.num_rules;
    let iw = policy.input_width;

    let input = b.input(vec![1, iw], 1);

    // Layer 1: [1, input_width] x [num_rules, input_width]^T -> [1, num_rules]
    let mut w1_tensor = Tensor::new(Some(&policy.w1), &[nr, iw]).unwrap();
    w1_tensor.set_scale(SCALE);
    let w1_const = b.const_tensor(w1_tensor, vec![nr, iw], 1);

    let mm1 = b.matmult(input, w1_const, vec![1, nr], 1);
    let mm1_rescaled = b.div(128, mm1, vec![1, nr], 1);

    let mut b1_tensor = Tensor::new(Some(&policy.b1), &[1, nr]).unwrap();
    b1_tensor.set_scale(SCALE);
    let b1_const = b.const_tensor(b1_tensor, vec![1, nr], 1);
    let biased1 = b.poly(PolyOp::Add, mm1_rescaled, b1_const, vec![1, nr], 1);
    let relu1 = b.relu(biased1, vec![1, nr], 1);

    // Layer 2: [1, num_rules] x [2, num_rules]^T -> [1, 2]
    let mut w2_tensor = Tensor::new(Some(&policy.w2), &[2, nr]).unwrap();
    w2_tensor.set_scale(SCALE);
    let w2_const = b.const_tensor(w2_tensor, vec![2, nr], 1);

    let mm2 = b.matmult(relu1, w2_const, vec![1, 2], 1);
    let mm2_rescaled = b.div(128, mm2, vec![1, 2], 1);

    let mut b2_tensor = Tensor::new(Some(&policy.b2), &[1, 2]).unwrap();
    b2_tensor.set_scale(SCALE);
    let b2_const = b.const_tensor(b2_tensor, vec![1, 2], 1);
    let output = b.poly(PolyOp::Add, mm2_rescaled, b2_const, vec![1, 2], 1);

    b.take(vec![input.0], vec![output])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compile_and_run_policy() {
        let rules = vec![PolicyRule {
            name: "block-chase".into(),
            block_domains: vec!["chase.com".into()],
            block_paths: vec![],
            block_keywords: vec![],
            actions: vec!["*".into()],
        }];

        let compiled = compile_rules(&rules);
        assert_eq!(compiled.input_width, 6); // 5 action + 1 condition
        assert_eq!(compiled.num_rules, 1);

        // Init and build model
        let _ = COMPILED_POLICY.set(compiled);
        let model = policy_model();

        // Input: network_request + domain matches
        let input = Tensor::new(Some(&[0, 0, 0, 0, 128, 128]), &[1, 6]).unwrap();
        let result = model.forward(&[input]).unwrap();
        let out = &result.outputs[0].inner;
        // Should deny
        assert!(out[0] > out[1], "Expected DENIED, got deny={} allow={}", out[0], out[1]);
    }
}
