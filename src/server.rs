//! HTTP server for the ClawGuard prover service.
//!
//! Provides REST API endpoints for skill safety evaluation with optional
//! ZK proof generation. Includes a free public evaluate-by-name endpoint
//! that fetches skill data from ClawHub before classification.

use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::net::SocketAddr;
use std::num::NonZeroU32;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use eyre::Result;
use governor::{Quota, RateLimiter};
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, Semaphore};

use crate::clawhub::ClawHubClient;
use crate::models::skill_safety::skill_safety_model;
use crate::receipt::{
    generate_nonce, ClassScores, GuardrailReceipt, PaymentInfo,
};
use crate::skill::{
    derive_decision, SafetyClassification, SafetyDecision, Skill, SkillFeatures, VTReport,
};
use crate::{hash_model_fn, proof_dir, GuardsConfig};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

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
    /// Path for JSONL access log
    pub access_log_path: String,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind_addr: "127.0.0.1:8080".parse().unwrap(),
            max_concurrent_proofs: 4,
            require_proof: false,
            guards_config: None,
            rate_limit_rpm: 60,
            access_log_path: "clawguard-access.jsonl".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Request / Response types
// ---------------------------------------------------------------------------

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

/// Request for evaluate-by-name endpoint
#[derive(Debug, Deserialize)]
pub struct EvaluateByNameRequest {
    /// Skill name (slug) on ClawHub
    pub skill: String,

    /// Optional version (defaults to latest)
    #[serde(default)]
    pub version: Option<String>,

    /// Whether to generate a ZK proof
    #[serde(default)]
    pub generate_proof: bool,
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

/// Stats response
#[derive(Debug, Serialize)]
pub struct StatsResponse {
    pub uptime_seconds: u64,
    pub model_hash: String,
    pub requests: RequestStats,
    pub classifications: ClassificationStats,
    pub decisions: DecisionStats,
    pub endpoints: EndpointStats,
}

#[derive(Debug, Serialize)]
pub struct RequestStats {
    pub total: u64,
    pub errors: u64,
}

#[derive(Debug, Serialize)]
pub struct ClassificationStats {
    pub safe: u64,
    pub caution: u64,
    pub dangerous: u64,
    pub malicious: u64,
}

#[derive(Debug, Serialize)]
pub struct DecisionStats {
    pub allow: u64,
    pub deny: u64,
    pub flag: u64,
}

#[derive(Debug, Serialize)]
pub struct EndpointStats {
    pub safety: u64,
    pub evaluate_by_name: u64,
    pub stats: u64,
}

// ---------------------------------------------------------------------------
// Usage metrics
// ---------------------------------------------------------------------------

/// Atomic usage counters for the server.
pub struct UsageMetrics {
    // Request totals
    pub total_requests: AtomicU64,
    pub total_errors: AtomicU64,

    // Per-classification
    pub safe: AtomicU64,
    pub caution: AtomicU64,
    pub dangerous: AtomicU64,
    pub malicious: AtomicU64,

    // Per-decision
    pub allow: AtomicU64,
    pub deny: AtomicU64,
    pub flag: AtomicU64,

    // Per-endpoint
    pub ep_safety: AtomicU64,
    pub ep_evaluate_by_name: AtomicU64,
    pub ep_stats: AtomicU64,

    // JSONL access log (append-only)
    pub access_log: std::sync::Mutex<Option<File>>,
}

impl UsageMetrics {
    fn new(access_log_path: &str) -> Self {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(access_log_path)
            .ok();
        if file.is_none() {
            eprintln!("WARNING: could not open access log: {}", access_log_path);
        }
        Self {
            total_requests: AtomicU64::new(0),
            total_errors: AtomicU64::new(0),
            safe: AtomicU64::new(0),
            caution: AtomicU64::new(0),
            dangerous: AtomicU64::new(0),
            malicious: AtomicU64::new(0),
            allow: AtomicU64::new(0),
            deny: AtomicU64::new(0),
            flag: AtomicU64::new(0),
            ep_safety: AtomicU64::new(0),
            ep_evaluate_by_name: AtomicU64::new(0),
            ep_stats: AtomicU64::new(0),
            access_log: std::sync::Mutex::new(file),
        }
    }

    /// Record a successful classification result.
    fn record(
        &self,
        endpoint: &str,
        skill_name: &str,
        classification: SafetyClassification,
        decision: SafetyDecision,
        confidence: f64,
        processing_time_ms: u64,
        proof_requested: bool,
    ) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);

        match classification {
            SafetyClassification::Safe => { self.safe.fetch_add(1, Ordering::Relaxed); }
            SafetyClassification::Caution => { self.caution.fetch_add(1, Ordering::Relaxed); }
            SafetyClassification::Dangerous => { self.dangerous.fetch_add(1, Ordering::Relaxed); }
            SafetyClassification::Malicious => { self.malicious.fetch_add(1, Ordering::Relaxed); }
        }

        match decision {
            SafetyDecision::Allow => { self.allow.fetch_add(1, Ordering::Relaxed); }
            SafetyDecision::Deny => { self.deny.fetch_add(1, Ordering::Relaxed); }
            SafetyDecision::Flag => { self.flag.fetch_add(1, Ordering::Relaxed); }
        }

        // Append to access log (non-blocking: use try_lock)
        if let Ok(mut guard) = self.access_log.try_lock() {
            if let Some(ref mut file) = *guard {
                let entry = serde_json::json!({
                    "timestamp": chrono::Utc::now().to_rfc3339(),
                    "endpoint": endpoint,
                    "skill_name": skill_name,
                    "classification": classification.as_str(),
                    "decision": decision.as_str(),
                    "confidence": confidence,
                    "processing_time_ms": processing_time_ms,
                    "proof_requested": proof_requested,
                });
                let mut line = entry.to_string();
                line.push('\n');
                let _ = file.write_all(line.as_bytes());
            }
        }
    }

    fn record_error(&self) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.total_errors.fetch_add(1, Ordering::Relaxed);
    }
}

// ---------------------------------------------------------------------------
// Server state
// ---------------------------------------------------------------------------

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
    /// ClawHub API client for fetching skill data
    pub clawhub_client: ClawHubClient,
    /// Usage metrics
    pub usage: UsageMetrics,
}

impl ServerState {
    pub fn new(config: ServerConfig) -> Self {
        let model_hash = hash_model_fn(skill_safety_model);
        let max_proofs = config.max_concurrent_proofs;
        let usage = UsageMetrics::new(&config.access_log_path);
        Self {
            config,
            model_hash,
            start_time: Instant::now(),
            proof_semaphore: Semaphore::new(max_proofs),
            rate_limiters: Mutex::new(HashMap::new()),
            clawhub_client: ClawHubClient::new(),
            usage,
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
        let quota = Quota::per_minute(NonZeroU32::new(self.config.rate_limit_rpm).unwrap());
        let limiter = Arc::new(RateLimiter::direct(quota));
        limiters.insert(ip, Arc::clone(&limiter));

        // Clean up old limiters periodically (keep map from growing unbounded)
        if limiters.len() > 10000 {
            eprintln!("WARNING: rate limiter map exceeded 10000 entries, clearing");
            limiters.clear();
            limiters.insert(ip, Arc::clone(&limiter));
        }

        Some(limiter)
    }
}

// ---------------------------------------------------------------------------
// Shared classification logic
// ---------------------------------------------------------------------------

/// Core classification + receipt generation used by all evaluation endpoints.
fn classify_and_respond(
    state: &Arc<ServerState>,
    features: SkillFeatures,
    skill_name: String,
    skill_version: String,
    generate_proof: bool,
    nonce: Option<String>,
    payment: Option<PaymentInfo>,
    start: Instant,
    endpoint: &str,
) -> SafetyResponse {
    let feature_vec = features.to_normalized_vec();

    // Run the classifier
    let result = match run_classifier(&feature_vec) {
        Ok(r) => r,
        Err(e) => {
            state.usage.record_error();
            return SafetyResponse {
                success: false,
                error: Some(format!("Classification failed: {}", e)),
                receipt: None,
                processing_time_ms: start.elapsed().as_millis() as u64,
            };
        }
    };

    let (classification, raw_scores, confidence) = result;
    let scores = ClassScores::from_raw_scores(&raw_scores);
    let (decision, reasoning) = derive_decision(classification, &scores.to_array());

    // Generate proof if requested
    let (proof_bytes, vk_hash, prove_time, program_io) = if generate_proof {
        match generate_proof_sync(&feature_vec, state.config.guards_config.as_ref()) {
            Ok((proof, vk, time, io)) => (proof, vk, Some(time), io),
            Err(e) => {
                state.usage.record_error();
                return SafetyResponse {
                    success: false,
                    error: Some(format!("Proof generation failed: {}", e)),
                    receipt: None,
                    processing_time_ms: start.elapsed().as_millis() as u64,
                };
            }
        }
    } else {
        ("".to_string(), "".to_string(), None, None)
    };

    // Generate nonce
    let nonce_bytes = if let Some(n) = &nonce {
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
        nonce_bytes,
    );

    // Add payment info if provided
    if let Some(pay) = payment {
        receipt = receipt.with_payment(pay);
    }

    let processing_time_ms = start.elapsed().as_millis() as u64;

    // Record metrics
    state.usage.record(
        endpoint,
        &skill_name,
        classification,
        decision,
        confidence,
        processing_time_ms,
        generate_proof,
    );

    SafetyResponse {
        success: true,
        error: None,
        receipt: Some(receipt),
        processing_time_ms,
    }
}

// ---------------------------------------------------------------------------
// HTTP server
// ---------------------------------------------------------------------------

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
        .route("/api/v1/evaluate/name", post(evaluate_by_name_handler))
        .route("/stats", get(stats_handler))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(config.bind_addr).await?;
    eprintln!("ClawGuard prover server listening on {}", config.bind_addr);
    eprintln!("Endpoints:");
    eprintln!("  GET  /health                  - Health check");
    eprintln!("  POST /guardrail/safety        - Evaluate skill safety");
    eprintln!("  POST /api/v1/evaluate         - Evaluate skill safety (alias)");
    eprintln!("  POST /api/v1/evaluate/name    - Evaluate skill by ClawHub name");
    eprintln!("  GET  /stats                   - Usage statistics");
    if rate_limit_rpm > 0 {
        eprintln!("Rate limit: {} requests/minute per IP", rate_limit_rpm);
    } else {
        eprintln!("Rate limit: disabled");
    }
    eprintln!("Access log: {}", config.access_log_path);

    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

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

/// Safety evaluation handler (existing endpoint, refactored to use shared logic)
async fn safety_handler(
    axum::extract::State(state): axum::extract::State<Arc<ServerState>>,
    axum::extract::ConnectInfo(addr): axum::extract::ConnectInfo<SocketAddr>,
    axum::Json(request): axum::Json<SafetyRequest>,
) -> impl axum::response::IntoResponse {
    let start = Instant::now();
    let client_ip = addr.ip();

    state.usage.ep_safety.fetch_add(1, Ordering::Relaxed);

    // Check rate limit
    if let Some(limiter) = state.get_rate_limiter(client_ip).await {
        if limiter.check().is_err() {
            state.usage.record_error();
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
                state.usage.record_error();
                return axum::Json(SafetyResponse {
                    success: false,
                    error: Some(format!("Expected 22 features, got {}", features.len())),
                    receipt: None,
                    processing_time_ms: start.elapsed().as_millis() as u64,
                });
            }
            let sf = features_from_vec(features);
            (sf, "unknown".to_string(), "unknown".to_string())
        }
        SafetyInput::SkillFeatures { skill_features, skill_name, skill_version } => {
            (skill_features.clone(), skill_name.clone(), skill_version.clone())
        }
    };

    // If proof is requested, acquire semaphore first
    let generate_proof = request.generate_proof;
    if generate_proof {
        match state.proof_semaphore.acquire().await {
            Ok(_permit) => {
                // permit held for duration of classify_and_respond
            }
            Err(_) => {
                state.usage.record_error();
                return axum::Json(SafetyResponse {
                    success: false,
                    error: Some("Proof generation unavailable".to_string()),
                    receipt: None,
                    processing_time_ms: start.elapsed().as_millis() as u64,
                });
            }
        }
    }

    let response = classify_and_respond(
        &state,
        features,
        skill_name,
        skill_version,
        generate_proof,
        request.nonce,
        request.payment,
        start,
        "safety",
    );

    axum::Json(response)
}

/// Evaluate a skill by name, fetching it from ClawHub first.
async fn evaluate_by_name_handler(
    axum::extract::State(state): axum::extract::State<Arc<ServerState>>,
    axum::extract::ConnectInfo(addr): axum::extract::ConnectInfo<SocketAddr>,
    axum::Json(request): axum::Json<EvaluateByNameRequest>,
) -> impl axum::response::IntoResponse {
    let start = Instant::now();
    let client_ip = addr.ip();

    state.usage.ep_evaluate_by_name.fetch_add(1, Ordering::Relaxed);

    // Check rate limit
    if let Some(limiter) = state.get_rate_limiter(client_ip).await {
        if limiter.check().is_err() {
            state.usage.record_error();
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

    // Fetch skill from ClawHub
    let skill = match state
        .clawhub_client
        .fetch_skill(&request.skill, request.version.as_deref())
        .await
    {
        Ok(s) => s,
        Err(e) => {
            state.usage.record_error();
            return axum::Json(SafetyResponse {
                success: false,
                error: Some(format!("Failed to fetch skill '{}': {}", request.skill, e)),
                receipt: None,
                processing_time_ms: start.elapsed().as_millis() as u64,
            });
        }
    };

    // Extract features
    let features = SkillFeatures::extract(&skill, None);
    let skill_name = skill.name;
    let skill_version = skill.version;

    let response = classify_and_respond(
        &state,
        features,
        skill_name,
        skill_version,
        request.generate_proof,
        None,
        None,
        start,
        "evaluate_by_name",
    );

    axum::Json(response)
}

/// Stats endpoint
async fn stats_handler(
    axum::extract::State(state): axum::extract::State<Arc<ServerState>>,
) -> impl axum::response::IntoResponse {
    state.usage.ep_stats.fetch_add(1, Ordering::Relaxed);

    let response = StatsResponse {
        uptime_seconds: state.start_time.elapsed().as_secs(),
        model_hash: state.model_hash.clone(),
        requests: RequestStats {
            total: state.usage.total_requests.load(Ordering::Relaxed),
            errors: state.usage.total_errors.load(Ordering::Relaxed),
        },
        classifications: ClassificationStats {
            safe: state.usage.safe.load(Ordering::Relaxed),
            caution: state.usage.caution.load(Ordering::Relaxed),
            dangerous: state.usage.dangerous.load(Ordering::Relaxed),
            malicious: state.usage.malicious.load(Ordering::Relaxed),
        },
        decisions: DecisionStats {
            allow: state.usage.allow.load(Ordering::Relaxed),
            deny: state.usage.deny.load(Ordering::Relaxed),
            flag: state.usage.flag.load(Ordering::Relaxed),
        },
        endpoints: EndpointStats {
            safety: state.usage.ep_safety.load(Ordering::Relaxed),
            evaluate_by_name: state.usage.ep_evaluate_by_name.load(Ordering::Relaxed),
            stats: state.usage.ep_stats.load(Ordering::Relaxed),
        },
    };
    axum::Json(response)
}

// ---------------------------------------------------------------------------
// Classifier helpers
// ---------------------------------------------------------------------------

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
    let max_trace_length = 1 << 16;

    let (proof_path, _program_io) = crate::proving::prove_and_save(
        skill_safety_model,
        &input,
        &proof_directory,
        &model_hash,
        max_trace_length,
        "skill-safety",
    )?;

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
        stars: 0,
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
        let mut features = vec![0i32; 22];
        features[16] = 100;

        let (classification, _scores, confidence) = run_classifier(&features).unwrap();
        assert!(confidence >= 0.0);
        println!("Classification: {:?}, confidence: {}", classification, confidence);
    }

    #[test]
    fn test_classifier_malicious_skill() {
        let mut features = vec![0i32; 22];
        features[0] = 80;
        features[5] = 128;
        features[6] = 100;
        features[7] = 128;
        features[8] = 80;
        features[19] = 128;
        features[20] = 128;

        let (classification, _scores, _confidence) = run_classifier(&features).unwrap();
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

    #[test]
    fn test_stats_response_serialization() {
        let response = StatsResponse {
            uptime_seconds: 3600,
            model_hash: "sha256:abc".to_string(),
            requests: RequestStats { total: 100, errors: 2 },
            classifications: ClassificationStats {
                safe: 80,
                caution: 10,
                dangerous: 7,
                malicious: 3,
            },
            decisions: DecisionStats {
                allow: 90,
                deny: 8,
                flag: 2,
            },
            endpoints: EndpointStats {
                safety: 60,
                evaluate_by_name: 35,
                stats: 5,
            },
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"total\":100"));
        assert!(json.contains("\"evaluate_by_name\":35"));
    }

    #[test]
    fn test_evaluate_by_name_request_deserialization() {
        let json = r#"{"skill": "weather-helper", "version": "1.0.0", "generate_proof": false}"#;
        let req: EvaluateByNameRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.skill, "weather-helper");
        assert_eq!(req.version, Some("1.0.0".to_string()));
        assert!(!req.generate_proof);
    }

    #[test]
    fn test_evaluate_by_name_request_minimal() {
        let json = r#"{"skill": "my-skill"}"#;
        let req: EvaluateByNameRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.skill, "my-skill");
        assert!(req.version.is_none());
        assert!(!req.generate_proof);
    }

    #[test]
    fn test_usage_metrics_counters() {
        let metrics = UsageMetrics::new("/dev/null");
        metrics.record(
            "safety",
            "test-skill",
            SafetyClassification::Safe,
            SafetyDecision::Allow,
            0.9,
            42,
            false,
        );
        assert_eq!(metrics.total_requests.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.safe.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.allow.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.total_errors.load(Ordering::Relaxed), 0);

        metrics.record_error();
        assert_eq!(metrics.total_requests.load(Ordering::Relaxed), 2);
        assert_eq!(metrics.total_errors.load(Ordering::Relaxed), 1);
    }
}
