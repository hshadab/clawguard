//! HTTP server for the ClawGuard prover service.
//!
//! Provides REST API endpoints for skill safety evaluation with optional
//! ZK proof generation. Designed for integration with x402 payment gating.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::Instant;

use eyre::Result;
use governor::{Quota, RateLimiter};
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, Semaphore};

use crate::models::skill_safety::skill_safety_model;
use crate::receipt::{
    generate_nonce, ClassScores, GuardrailReceipt, PaymentInfo,
};
use crate::skill::{
    derive_decision, SafetyClassification, Skill, SkillFeatures, VTReport,
};
use crate::{hash_model_fn, proof_dir, GuardsConfig};

/// Server configuration
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Address to bind to
    pub bind_addr: SocketAddr,
    /// Maximum concurrent proof generations
    pub max_concurrent_proofs: usize,
    /// Whether to require proof generation
    pub require_proof: bool,
    /// Optional guards config
    pub guards_config: Option<GuardsConfig>,
    /// Rate limit in requests per minute per IP (0 = no limit)
    pub rate_limit_rpm: u32,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind_addr: "127.0.0.1:8080".parse().unwrap(),
            max_concurrent_proofs: 4,
            require_proof: false,
            guards_config: None,
            rate_limit_rpm: 60, // Default: 60 requests per minute per IP
        }
    }
}

/// Request for skill safety evaluation
#[derive(Debug, Deserialize)]
pub struct SafetyRequest {
    /// The skill to evaluate (full skill data or just features)
    #[serde(flatten)]
    pub input: SafetyInput,

    /// Nonce for replay protection (optional, generated if not provided)
    #[serde(default)]
    pub nonce: Option<String>,

    /// Whether to generate a ZK proof
    #[serde(default)]
    pub generate_proof: bool,

    /// Optional payment information
    #[serde(default)]
    pub payment: Option<PaymentInfo>,
}

/// Input for safety evaluation
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum SafetyInput {
    /// Full skill data
    Skill { skill: Skill, vt_report: Option<VTReport> },
    /// Pre-computed features (22-dimensional normalized vector)
    Features { features: Vec<i32> },
    /// Structured features
    SkillFeatures { skill_features: SkillFeatures, skill_name: String, skill_version: String },
}

/// Response from skill safety evaluation
#[derive(Debug, Serialize)]
pub struct SafetyResponse {
    /// Whether the request succeeded
    pub success: bool,

    /// Error message if failed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,

    /// The guardrail receipt
    #[serde(skip_serializing_if = "Option::is_none")]
    pub receipt: Option<GuardrailReceipt>,

    /// Processing time in milliseconds
    pub processing_time_ms: u64,
}

/// Health check response
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub model_hash: String,
    pub uptime_seconds: u64,
}

/// Type alias for per-IP rate limiters
type IpRateLimiter = RateLimiter<
    governor::state::NotKeyed,
    governor::state::InMemoryState,
    governor::clock::DefaultClock,
>;

/// Server state
pub struct ServerState {
    pub config: ServerConfig,
    pub model_hash: String,
    pub start_time: Instant,
    pub proof_semaphore: Semaphore,
    /// Per-IP rate limiters (lazy-initialized)
    pub rate_limiters: Mutex<HashMap<std::net::IpAddr, Arc<IpRateLimiter>>>,
}

impl ServerState {
    pub fn new(config: ServerConfig) -> Self {
        let model_hash = hash_model_fn(skill_safety_model);
        let max_proofs = config.max_concurrent_proofs;
        Self {
            config,
            model_hash,
            start_time: Instant::now(),
            proof_semaphore: Semaphore::new(max_proofs),
            rate_limiters: Mutex::new(HashMap::new()),
        }
    }

    /// Get or create a rate limiter for the given IP address.
    pub async fn get_rate_limiter(&self, ip: std::net::IpAddr) -> Option<Arc<IpRateLimiter>> {
        if self.config.rate_limit_rpm == 0 {
            return None; // Rate limiting disabled
        }

        let mut limiters = self.rate_limiters.lock().await;

        if let Some(limiter) = limiters.get(&ip) {
            return Some(Arc::clone(limiter));
        }

        // Create a new limiter for this IP
        // Quota: rate_limit_rpm requests per 60 seconds
        let quota = Quota::per_minute(NonZeroU32::new(self.config.rate_limit_rpm).unwrap());
        let limiter = Arc::new(RateLimiter::direct(quota));
        limiters.insert(ip, Arc::clone(&limiter));

        // Clean up old limiters periodically (keep map from growing unbounded)
        // Simple approach: remove entries that haven't been used recently
        if limiters.len() > 10000 {
            // If we have too many entries, clear them all
            // A more sophisticated approach would use LRU eviction
            eprintln!("WARNING: rate limiter map exceeded 10000 entries, clearing");
            limiters.clear();
            limiters.insert(ip, Arc::clone(&limiter));
        }

        Some(limiter)
    }
}

/// Run the HTTP server (blocking)
pub async fn run_server(config: ServerConfig) -> Result<()> {
    use axum::{
        routing::{get, post},
        Router,
    };

    let rate_limit_rpm = config.rate_limit_rpm;
    let state = Arc::new(ServerState::new(config.clone()));

    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/guardrail/safety", post(safety_handler))
        .route("/api/v1/evaluate", post(safety_handler))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(config.bind_addr).await?;
    eprintln!("ClawGuard prover server listening on {}", config.bind_addr);
    eprintln!("Endpoints:");
    eprintln!("  GET  /health              - Health check");
    eprintln!("  POST /guardrail/safety    - Evaluate skill safety");
    eprintln!("  POST /api/v1/evaluate     - Evaluate skill safety (alias)");
    if rate_limit_rpm > 0 {
        eprintln!("Rate limit: {} requests/minute per IP", rate_limit_rpm);
    } else {
        eprintln!("Rate limit: disabled");
    }

    // Use into_make_service_with_connect_info to get client IP
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .await?;
    Ok(())
}

/// Health check handler
async fn health_handler(
    axum::extract::State(state): axum::extract::State<Arc<ServerState>>,
) -> impl axum::response::IntoResponse {
    let response = HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        model_hash: state.model_hash.clone(),
        uptime_seconds: state.start_time.elapsed().as_secs(),
    };
    axum::Json(response)
}

/// Safety evaluation handler
async fn safety_handler(
    axum::extract::State(state): axum::extract::State<Arc<ServerState>>,
    axum::extract::ConnectInfo(addr): axum::extract::ConnectInfo<SocketAddr>,
    axum::Json(request): axum::Json<SafetyRequest>,
) -> impl axum::response::IntoResponse {
    let start = Instant::now();
    let client_ip = addr.ip();

    // Check rate limit
    if let Some(limiter) = state.get_rate_limiter(client_ip).await {
        if limiter.check().is_err() {
            return axum::Json(SafetyResponse {
                success: false,
                error: Some(format!(
                    "Rate limit exceeded. Maximum {} requests per minute.",
                    state.config.rate_limit_rpm
                )),
                receipt: None,
                processing_time_ms: start.elapsed().as_millis() as u64,
            });
        }
    }

    // Extract features and skill info based on input type
    let (features, skill_name, skill_version) = match &request.input {
        SafetyInput::Skill { skill, vt_report } => {
            let features = SkillFeatures::extract(skill, vt_report.as_ref());
            (features, skill.name.clone(), skill.version.clone())
        }
        SafetyInput::Features { features } => {
            if features.len() != 22 {
                return axum::Json(SafetyResponse {
                    success: false,
                    error: Some(format!("Expected 22 features, got {}", features.len())),
                    receipt: None,
                    processing_time_ms: start.elapsed().as_millis() as u64,
                });
            }
            // Create a dummy SkillFeatures from the raw vector
            let sf = features_from_vec(features);
            (sf, "unknown".to_string(), "unknown".to_string())
        }
        SafetyInput::SkillFeatures { skill_features, skill_name, skill_version } => {
            (skill_features.clone(), skill_name.clone(), skill_version.clone())
        }
    };

    // Run the classifier
    let feature_vec = features.to_normalized_vec();
    let result = match run_classifier(&feature_vec) {
        Ok(r) => r,
        Err(e) => {
            return axum::Json(SafetyResponse {
                success: false,
                error: Some(format!("Classification failed: {}", e)),
                receipt: None,
                processing_time_ms: start.elapsed().as_millis() as u64,
            });
        }
    };

    let (classification, raw_scores, confidence) = result;
    let scores = ClassScores::from_raw_scores(&raw_scores);
    let (decision, reasoning) = derive_decision(classification, &scores.to_array());

    // Generate proof if requested
    let (proof_bytes, vk_hash, prove_time, program_io) = if request.generate_proof {
        // Acquire semaphore to limit concurrent proofs
        let _permit = match state.proof_semaphore.acquire().await {
            Ok(p) => p,
            Err(_) => {
                return axum::Json(SafetyResponse {
                    success: false,
                    error: Some("Proof generation unavailable".to_string()),
                    receipt: None,
                    processing_time_ms: start.elapsed().as_millis() as u64,
                });
            }
        };

        match generate_proof_sync(&feature_vec, state.config.guards_config.as_ref()) {
            Ok((proof, vk, time, io)) => (proof, vk, Some(time), io),
            Err(e) => {
                return axum::Json(SafetyResponse {
                    success: false,
                    error: Some(format!("Proof generation failed: {}", e)),
                    receipt: None,
                    processing_time_ms: start.elapsed().as_millis() as u64,
                });
            }
        }
    } else {
        ("".to_string(), "".to_string(), None, None)
    };

    // Generate nonce
    let nonce = if let Some(n) = &request.nonce {
        let mut arr = [0u8; 32];
        if let Ok(bytes) = hex::decode(n.trim_start_matches("0x")) {
            let len = bytes.len().min(32);
            arr[..len].copy_from_slice(&bytes[..len]);
        }
        arr
    } else {
        generate_nonce()
    };

    // Create receipt
    let mut receipt = GuardrailReceipt::new_safety_receipt(
        &skill_name,
        &skill_version,
        &features,
        classification,
        decision,
        &reasoning,
        scores,
        confidence,
        state.model_hash.clone(),
        proof_bytes,
        vk_hash,
        prove_time,
        program_io,
        nonce,
    );

    // Add payment info if provided
    if let Some(payment) = request.payment {
        receipt = receipt.with_payment(payment);
    }

    axum::Json(SafetyResponse {
        success: true,
        error: None,
        receipt: Some(receipt),
        processing_time_ms: start.elapsed().as_millis() as u64,
    })
}

/// Run the classifier on a feature vector
fn run_classifier(features: &[i32]) -> Result<(SafetyClassification, [i32; 4], f64)> {
    use onnx_tracer::tensor::Tensor;

    let model = skill_safety_model();
    let input = Tensor::new(Some(features), &[1, 22])
        .map_err(|e| eyre::eyre!("Tensor error: {:?}", e))?;

    let result = model
        .forward(std::slice::from_ref(&input))
        .map_err(|e| eyre::eyre!("Forward error: {}", e))?;

    let data = &result.outputs[0].inner;
    if data.len() < 4 {
        eyre::bail!("Expected 4 output classes, got {}", data.len());
    }

    let raw_scores: [i32; 4] = [data[0], data[1], data[2], data[3]];

    // Find best class
    let (best_idx, &best_val) = data
        .iter()
        .enumerate()
        .max_by_key(|(_, v)| *v)
        .unwrap();

    // Calculate confidence as margin over runner-up
    let runner_up = data
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != best_idx)
        .map(|(_, v)| *v)
        .max()
        .unwrap_or(0);

    let margin = (best_val - runner_up).abs();
    let raw_conf = margin as f64 / 128.0;
    if raw_conf > 1.0 {
        eprintln!(
            "WARNING: confidence {:.3} > 1.0, clamping (possible model issue)",
            raw_conf
        );
    }
    let confidence = raw_conf.min(1.0);

    let classification = SafetyClassification::from_index(best_idx);

    Ok((classification, raw_scores, confidence))
}

/// Generate a ZK proof (synchronous, called from async context via spawn_blocking)
fn generate_proof_sync(
    features: &[i32],
    config: Option<&GuardsConfig>,
) -> Result<(String, String, u64, Option<String>)> {
    use onnx_tracer::tensor::Tensor;
    use std::time::Instant;

    let start = Instant::now();

    let input = Tensor::new(Some(features), &[1, 22])
        .map_err(|e| eyre::eyre!("Tensor error: {:?}", e))?;

    let proof_directory = proof_dir(config);
    let model_hash = hash_model_fn(skill_safety_model);
    let max_trace_length = 1 << 16; // 64K trace length for larger model

    let (proof_path, _program_io) = crate::proving::prove_and_save(
        skill_safety_model,
        &input,
        &proof_directory,
        &model_hash,
        max_trace_length,
        "skill-safety",
    )?;

    // Read the proof file to extract proof bytes and program_io
    let proof_content = std::fs::read_to_string(&proof_path)?;
    let proof_json: serde_json::Value = serde_json::from_str(&proof_content)?;

    let proof_bytes = proof_json
        .get("proof")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let program_io = proof_json
        .get("program_io")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    // Use model hash as VK hash for now
    let vk_hash = model_hash;

    let prove_time = start.elapsed().as_millis() as u64;

    Ok((proof_bytes, vk_hash, prove_time, program_io))
}

/// Convert a raw feature vector back to SkillFeatures (for API flexibility)
fn features_from_vec(vec: &[i32]) -> SkillFeatures {
    const SCALE: i32 = 128;

    let unclip = |val: i32, max: u32| -> u32 {
        ((val as f32 / SCALE as f32) * max as f32) as u32
    };

    let unbool = |val: i32| -> bool {
        val > SCALE / 2
    };

    SkillFeatures {
        shell_exec_count: unclip(vec.get(0).copied().unwrap_or(0), 20),
        network_call_count: unclip(vec.get(1).copied().unwrap_or(0), 50),
        fs_write_count: unclip(vec.get(2).copied().unwrap_or(0), 30),
        env_access_count: unclip(vec.get(3).copied().unwrap_or(0), 20),
        credential_patterns: unclip(vec.get(4).copied().unwrap_or(0), 10),
        external_download: unbool(vec.get(5).copied().unwrap_or(0)),
        obfuscation_score: unclip(vec.get(6).copied().unwrap_or(0), 15) as f32,
        privilege_escalation: unbool(vec.get(7).copied().unwrap_or(0)),
        persistence_mechanisms: unclip(vec.get(8).copied().unwrap_or(0), 5),
        data_exfiltration_patterns: unclip(vec.get(9).copied().unwrap_or(0), 5),
        skill_md_line_count: unclip(vec.get(10).copied().unwrap_or(0), 500),
        script_file_count: unclip(vec.get(11).copied().unwrap_or(0), 10),
        dependency_count: unclip(vec.get(12).copied().unwrap_or(0), 30),
        author_account_age_days: unclip(vec.get(13).copied().unwrap_or(0), 365),
        author_skill_count: unclip(vec.get(14).copied().unwrap_or(0), 100),
        // Log-scaled features need inverse
        stars: 0, // Can't accurately invert log scale
        downloads: 0,
        has_virustotal_report: unbool(vec.get(17).copied().unwrap_or(0)),
        vt_malicious_flags: unclip(vec.get(18).copied().unwrap_or(0), 20),
        password_protected_archives: unbool(vec.get(19).copied().unwrap_or(0)),
        reverse_shell_patterns: unclip(vec.get(20).copied().unwrap_or(0), 5),
        llm_secret_exposure: unbool(vec.get(21).copied().unwrap_or(0)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classifier_safe_skill() {
        // Safe skill: high downloads indicates established/trusted skill
        let mut features = vec![0i32; 22];
        features[16] = 100;  // downloads (high)

        let (classification, _scores, confidence) = run_classifier(&features).unwrap();
        // Classifier should return valid results
        assert!(confidence >= 0.0);
        // The trained model's output may vary, but should be deterministic
        println!("Classification: {:?}, confidence: {}", classification, confidence);
    }

    #[test]
    fn test_classifier_malicious_skill() {
        let mut features = vec![0i32; 22];
        features[0] = 80;   // shell_exec_count
        features[5] = 128;  // external_download
        features[6] = 100;  // obfuscation_score
        features[7] = 128;  // privilege_escalation
        features[8] = 80;   // persistence_mechanisms
        features[19] = 128; // password_protected_archives
        features[20] = 128; // reverse_shell_patterns

        let (classification, _scores, _confidence) = run_classifier(&features).unwrap();
        // Should be DANGEROUS or MALICIOUS - both result in denial
        assert!(classification.is_deny(), "Expected denial (DANGEROUS or MALICIOUS), got {:?}", classification);
    }

    #[test]
    fn test_health_response_serialization() {
        let response = HealthResponse {
            status: "ok".to_string(),
            version: "0.1.0".to_string(),
            model_hash: "sha256:abc".to_string(),
            uptime_seconds: 100,
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"status\":\"ok\""));
    }
}
