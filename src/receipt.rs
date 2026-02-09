//! Guardrail Receipt schema for verifiable safety evaluations.
//!
//! Receipts provide cryptographic proof that a skill was evaluated by a specific
//! model, with the evaluation result binding to the input features and payment.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::skill::{SafetyClassification, SafetyDecision, SkillFeatures};

/// Version of the receipt schema
pub const RECEIPT_VERSION: &str = "1.0.0";

/// A complete guardrail receipt for skill safety evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardrailReceipt {
    /// Schema version
    pub version: String,

    /// Unique receipt identifier
    pub receipt_id: String,

    /// Timestamp of evaluation
    pub timestamp: DateTime<Utc>,

    /// Guardrail metadata
    pub guardrail: GuardrailInfo,

    /// Subject being evaluated
    pub subject: Subject,

    /// Evaluation results
    pub evaluation: Evaluation,

    /// Cryptographic proof
    pub proof: ProofInfo,

    /// Optional payment information (for x402 integration)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payment: Option<PaymentInfo>,

    /// Nonce for uniqueness
    pub nonce: String,

    /// Additional metadata
    #[serde(default)]
    pub metadata: ReceiptMetadata,
}

/// Information about the guardrail that produced this receipt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardrailInfo {
    /// Domain of the guardrail (e.g., "safety", "spending")
    pub domain: String,

    /// Type of action being gated
    pub action_type: String,

    /// Policy identifier
    pub policy_id: String,

    /// Hash of the model used
    pub model_hash: String,
}

/// Subject being evaluated
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subject {
    /// SHA-256 commitment to the input features
    pub commitment: String,

    /// Human-readable description
    pub description: String,

    /// URI identifying the subject
    pub uri: String,
}

/// Evaluation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evaluation {
    /// Decision: allow, deny, or flag
    pub decision: String,

    /// Classification label
    pub classification: String,

    /// Confidence score (0.0 - 1.0)
    pub confidence: f64,

    /// Scores for each class
    pub scores: ClassScores,

    /// Human-readable reasoning
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
}

/// Scores for each safety class
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassScores {
    #[serde(rename = "SAFE")]
    pub safe: f64,
    #[serde(rename = "CAUTION")]
    pub caution: f64,
    #[serde(rename = "DANGEROUS")]
    pub dangerous: f64,
    #[serde(rename = "MALICIOUS")]
    pub malicious: f64,
}

impl ClassScores {
    pub fn from_raw_scores(raw: &[i32; 4]) -> Self {
        // Convert raw i32 scores to normalized probabilities using softmax
        // Scale down to avoid overflow: divide by 128 (the scale factor)
        let scaled: Vec<f64> = raw.iter().map(|&x| (x as f64) / 128.0).collect();
        let max_val = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_vals: Vec<f64> = scaled.iter().map(|&x| (x - max_val).exp()).collect();
        let total: f64 = exp_vals.iter().sum();

        if total == 0.0 || !total.is_finite() {
            // Fallback to uniform distribution
            return Self {
                safe: 0.25,
                caution: 0.25,
                dangerous: 0.25,
                malicious: 0.25,
            };
        }

        Self {
            safe: exp_vals[0] / total,
            caution: exp_vals[1] / total,
            dangerous: exp_vals[2] / total,
            malicious: exp_vals[3] / total,
        }
    }

    pub fn to_array(&self) -> [f64; 4] {
        [self.safe, self.caution, self.dangerous, self.malicious]
    }
}

/// Proof information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofInfo {
    /// Proof system used
    pub system: String,

    /// Base64-encoded proof bytes
    pub proof_bytes: String,

    /// Hash of the verification key
    pub verification_key_hash: String,

    /// Time to generate proof in milliseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prove_time_ms: Option<u64>,

    /// Serialized program IO for proof verification
    #[serde(skip_serializing_if = "Option::is_none")]
    pub program_io: Option<String>,
}

/// Payment information (for x402 integration)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaymentInfo {
    /// Network identifier (e.g., "eip155:8453" for Base)
    pub network: String,

    /// Asset contract address (e.g., USDC on Base)
    pub asset: String,

    /// Amount in smallest unit (e.g., "5000" for 0.005 USDC)
    pub amount: String,

    /// Payer address
    pub payer: String,

    /// Payee address
    pub payee: String,

    /// Transaction hash
    pub tx_hash: String,

    /// Payment scheme
    pub scheme: String,
}

/// Additional receipt metadata
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ReceiptMetadata {
    /// Version of the prover service
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prover_version: Option<String>,

    /// Version of JOLT Atlas
    #[serde(skip_serializing_if = "Option::is_none")]
    pub jolt_atlas_version: Option<String>,

    /// Number of input features
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_features_count: Option<usize>,

    /// Number of model parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_params_count: Option<usize>,
}

impl GuardrailReceipt {
    /// Create a new receipt for a skill safety evaluation
    pub fn new_safety_receipt(
        skill_name: &str,
        skill_version: &str,
        features: &SkillFeatures,
        classification: SafetyClassification,
        decision: SafetyDecision,
        reasoning: &str,
        scores: ClassScores,
        confidence: f64,
        model_hash: String,
        proof_bytes: String,
        vk_hash: String,
        prove_time_ms: Option<u64>,
        program_io: Option<String>,
        nonce: [u8; 32],
    ) -> Self {
        let receipt_id = generate_receipt_id();
        let feature_vec = features.to_normalized_vec();
        let commitment = compute_commitment(&feature_vec);

        Self {
            version: RECEIPT_VERSION.to_string(),
            receipt_id,
            timestamp: Utc::now(),
            guardrail: GuardrailInfo {
                domain: "safety".to_string(),
                action_type: "install_skill".to_string(),
                policy_id: "icme:skill-safety-v1".to_string(),
                model_hash,
            },
            subject: Subject {
                commitment,
                description: format!("OpenClaw skill: {} v{}", skill_name, skill_version),
                uri: format!("clawhub://{}/{}", skill_name, skill_version),
            },
            evaluation: Evaluation {
                decision: decision.as_str().to_string(),
                classification: classification.as_str().to_string(),
                confidence,
                scores,
                reasoning: Some(reasoning.to_string()),
            },
            proof: ProofInfo {
                system: "jolt-atlas".to_string(),
                proof_bytes,
                verification_key_hash: vk_hash,
                prove_time_ms,
                program_io,
            },
            payment: None,
            nonce: hex::encode(nonce),
            metadata: ReceiptMetadata {
                prover_version: Some(env!("CARGO_PKG_VERSION").to_string()),
                jolt_atlas_version: Some("0.5.0".to_string()),
                input_features_count: Some(22),
                model_params_count: Some(1924),
            },
        }
    }

    /// Add payment information to the receipt
    pub fn with_payment(mut self, payment: PaymentInfo) -> Self {
        self.payment = Some(payment);
        self
    }

    /// Verify the receipt's input commitment matches the provided features
    pub fn verify_commitment(&self, features: &SkillFeatures) -> bool {
        let feature_vec = features.to_normalized_vec();
        let expected = compute_commitment(&feature_vec);
        self.subject.commitment == expected
    }

    /// Check if this receipt indicates the skill should be blocked
    pub fn is_blocked(&self) -> bool {
        self.evaluation.decision == "deny"
    }

    /// Check if this receipt indicates the skill needs review
    pub fn is_flagged(&self) -> bool {
        self.evaluation.decision == "flag"
    }
}

/// Generate a unique receipt ID
fn generate_receipt_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();

    let mut hasher = Sha256::new();
    hasher.update(timestamp.to_le_bytes());
    hasher.update(&rand_bytes());
    let hash = hasher.finalize();

    format!("gr_safety_{}", &hex::encode(&hash[..6]))
}

/// Compute SHA-256 commitment to feature vector
fn compute_commitment(features: &[i32]) -> String {
    // Canonicalize: convert to JSON (sorted keys not needed for array)
    let canonical = serde_json::to_string(features).unwrap_or_default();

    let mut hasher = Sha256::new();
    hasher.update(canonical.as_bytes());
    let hash = hasher.finalize();

    format!("sha256:{}", hex::encode(hash))
}

/// Generate random bytes for nonce
fn rand_bytes() -> [u8; 16] {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};

    let mut bytes = [0u8; 16];
    let state = RandomState::new();
    let mut hasher = state.build_hasher();
    hasher.write_u64(std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64);
    let hash1 = hasher.finish();

    let state2 = RandomState::new();
    let mut hasher2 = state2.build_hasher();
    hasher2.write_u64(hash1);
    let hash2 = hasher2.finish();

    bytes[..8].copy_from_slice(&hash1.to_le_bytes());
    bytes[8..].copy_from_slice(&hash2.to_le_bytes());
    bytes
}

/// Generate a random 32-byte nonce
pub fn generate_nonce() -> [u8; 32] {
    let mut nonce = [0u8; 32];
    let r1 = rand_bytes();
    let r2 = rand_bytes();
    nonce[..16].copy_from_slice(&r1);
    nonce[16..].copy_from_slice(&r2);
    nonce
}

// ---------------------------------------------------------------------------
// Scan result for batch processing
// ---------------------------------------------------------------------------

/// Result of scanning a single skill
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanResult {
    pub skill_name: String,
    pub skill_version: String,
    pub skill_uri: String,
    pub scanned_at: DateTime<Utc>,
    pub features: SkillFeatures,
    pub classification: String,
    pub confidence: f64,
    pub decision: String,
    pub receipt_id: String,
    pub receipt_file: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payment_tx: Option<String>,
    pub model_hash: String,
}

/// Flagged skill details for dangerous/malicious classifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlaggedSkill {
    pub skill_name: String,
    pub skill_version: String,
    pub classification: String,
    pub confidence: f64,
    pub primary_risk_factors: Vec<String>,
    pub cross_reference: CrossReference,
    pub receipt_id: String,
    pub verify_url: String,
}

/// Cross-reference with other security reports
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CrossReference {
    pub koi_flagged: bool,
    pub snyk_flagged: bool,
    pub vt_flagged: bool,
}

impl FlaggedSkill {
    /// Create a flagged skill record from features and classification
    pub fn from_scan(
        skill_name: &str,
        skill_version: &str,
        features: &SkillFeatures,
        classification: &str,
        confidence: f64,
        receipt_id: &str,
    ) -> Self {
        let mut risk_factors = Vec::new();

        // Identify primary risk factors
        if features.llm_secret_exposure {
            risk_factors.push(format!(
                "llm_secret_exposure: true (SKILL.md instructs passing secrets through context)"
            ));
        }
        if features.credential_patterns > 0 {
            risk_factors.push(format!(
                "credential_patterns: {} (API key, password, token references)",
                features.credential_patterns
            ));
        }
        if features.reverse_shell_patterns > 0 {
            risk_factors.push(format!(
                "reverse_shell_patterns: {} (nc -e, /dev/tcp, bash -i)",
                features.reverse_shell_patterns
            ));
        }
        if features.obfuscation_score > 0.0 {
            risk_factors.push(format!(
                "obfuscation_score: {:.1} (eval, base64, dynamic imports)",
                features.obfuscation_score
            ));
        }
        if features.persistence_mechanisms > 0 {
            risk_factors.push(format!(
                "persistence_mechanisms: {} (cron, systemd, autostart)",
                features.persistence_mechanisms
            ));
        }
        if features.data_exfiltration_patterns > 0 {
            risk_factors.push(format!(
                "data_exfiltration_patterns: {} (POST to external, webhooks)",
                features.data_exfiltration_patterns
            ));
        }
        if features.privilege_escalation {
            risk_factors.push("privilege_escalation: true (sudo, chmod)".to_string());
        }
        if features.password_protected_archives {
            risk_factors.push(
                "password_protected_archives: true (scanner evasion)".to_string()
            );
        }
        if features.env_access_count > 5 {
            risk_factors.push(format!(
                "env_access_count: {} (heavy .env usage)",
                features.env_access_count
            ));
        }
        if features.vt_malicious_flags > 0 {
            risk_factors.push(format!(
                "vt_malicious_flags: {} (VirusTotal detections)",
                features.vt_malicious_flags
            ));
        }

        Self {
            skill_name: skill_name.to_string(),
            skill_version: skill_version.to_string(),
            classification: classification.to_string(),
            confidence,
            primary_risk_factors: risk_factors,
            cross_reference: CrossReference::default(),
            receipt_id: receipt_id.to_string(),
            verify_url: format!("https://audit.icme.io/receipt/{}", receipt_id),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_receipt_serialization() {
        let scores = ClassScores {
            safe: 0.1,
            caution: 0.15,
            dangerous: 0.7,
            malicious: 0.05,
        };

        let features = SkillFeatures {
            shell_exec_count: 0,
            network_call_count: 5,
            fs_write_count: 0,
            env_access_count: 3,
            credential_patterns: 2,
            external_download: false,
            obfuscation_score: 0.0,
            privilege_escalation: false,
            persistence_mechanisms: 0,
            data_exfiltration_patterns: 0,
            skill_md_line_count: 50,
            script_file_count: 1,
            dependency_count: 3,
            author_account_age_days: 30,
            author_skill_count: 5,
            stars: 10,
            downloads: 100,
            has_virustotal_report: false,
            vt_malicious_flags: 0,
            password_protected_archives: false,
            reverse_shell_patterns: 0,
            llm_secret_exposure: true,
        };

        let receipt = GuardrailReceipt::new_safety_receipt(
            "test-skill",
            "1.0.0",
            &features,
            SafetyClassification::Dangerous,
            SafetyDecision::Deny,
            "Credential exposure detected",
            scores,
            0.7,
            "sha256:abc123".to_string(),
            "base64proof".to_string(),
            "sha256:vk123".to_string(),
            Some(1500),
            Some("{\"inputs\":[],\"outputs\":[]}".to_string()),
            [0u8; 32],
        );

        let json = serde_json::to_string_pretty(&receipt).unwrap();
        assert!(json.contains("gr_safety_"));
        assert!(json.contains("DANGEROUS"));
        assert!(json.contains("deny"));

        // Verify it can be deserialized back
        let parsed: GuardrailReceipt = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.evaluation.classification, "DANGEROUS");
    }

    #[test]
    fn test_commitment_verification() {
        let features = SkillFeatures {
            shell_exec_count: 5,
            network_call_count: 10,
            fs_write_count: 2,
            env_access_count: 3,
            credential_patterns: 1,
            external_download: true,
            obfuscation_score: 2.0,
            privilege_escalation: false,
            persistence_mechanisms: 0,
            data_exfiltration_patterns: 1,
            skill_md_line_count: 100,
            script_file_count: 3,
            dependency_count: 5,
            author_account_age_days: 60,
            author_skill_count: 10,
            stars: 50,
            downloads: 500,
            has_virustotal_report: true,
            vt_malicious_flags: 0,
            password_protected_archives: false,
            reverse_shell_patterns: 0,
            llm_secret_exposure: false,
        };

        let scores = ClassScores {
            safe: 0.6,
            caution: 0.3,
            dangerous: 0.08,
            malicious: 0.02,
        };

        let receipt = GuardrailReceipt::new_safety_receipt(
            "my-skill",
            "2.0.0",
            &features,
            SafetyClassification::Safe,
            SafetyDecision::Allow,
            "No issues detected",
            scores,
            0.6,
            "sha256:def456".to_string(),
            "proof".to_string(),
            "sha256:vk456".to_string(),
            None,
            None,
            generate_nonce(),
        );

        // Verify with same features should pass
        assert!(receipt.verify_commitment(&features));

        // Verify with different features should fail
        let mut modified = features.clone();
        modified.shell_exec_count = 100;
        assert!(!receipt.verify_commitment(&modified));
    }

    #[test]
    fn test_flagged_skill_generation() {
        let features = SkillFeatures {
            shell_exec_count: 10,
            network_call_count: 15,
            fs_write_count: 5,
            env_access_count: 8,
            credential_patterns: 5,
            external_download: true,
            obfuscation_score: 8.0,
            privilege_escalation: true,
            persistence_mechanisms: 3,
            data_exfiltration_patterns: 4,
            skill_md_line_count: 200,
            script_file_count: 5,
            dependency_count: 10,
            author_account_age_days: 5,
            author_skill_count: 100,
            stars: 0,
            downloads: 10,
            has_virustotal_report: true,
            vt_malicious_flags: 5,
            password_protected_archives: true,
            reverse_shell_patterns: 2,
            llm_secret_exposure: true,
        };

        let flagged = FlaggedSkill::from_scan(
            "malicious-skill",
            "1.0.0",
            &features,
            "MALICIOUS",
            0.92,
            "gr_safety_abc123",
        );

        assert!(flagged.primary_risk_factors.len() > 5);
        assert!(flagged.primary_risk_factors.iter().any(|f| f.contains("reverse_shell")));
        assert!(flagged.primary_risk_factors.iter().any(|f| f.contains("llm_secret_exposure")));
    }
}
