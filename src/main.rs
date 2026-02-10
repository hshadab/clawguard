use chrono::Utc;
use clap::{Parser, Subcommand};
use eyre::{Result, WrapErr};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::path::PathBuf;

use clawguard::{
    enforcement::EnforcementLevel,
    history_path, is_deny_decision, load_config, rotate_history_if_needed, run_guardrail,
    run_skill_safety, validate_config, config_has_errors, GuardModel,
    receipt::{
        CrossReferenceData, FlaggedSkill, GuardrailReceipt, ClassScores, PaymentInfo,
        ScanResult, generate_nonce,
    },
    skill::{Skill, SkillFeatures, VTReport, derive_decision, skill_from_skill_md},
};

#[derive(Parser)]
#[command(
    name = "clawguard",
    about = "Gate agent actions through ONNX guardrail models with zero-knowledge proof verification."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Check an action against a guardrail model
    Check {
        /// Model name (e.g. "action-gatekeeper") or path to an ONNX file
        #[arg(long)]
        model: String,

        /// Action type (e.g. "send_email", "run_command", "read_file")
        #[arg(long)]
        action: String,

        /// JSON context for the action
        #[arg(long)]
        context: String,

        /// Generate a zero-knowledge proof of correct evaluation
        #[arg(long, default_value_t = false)]
        prove: bool,

        /// Dry run — classify only, no proof generation
        #[arg(long, default_value_t = false)]
        dry_run: bool,
    },

    /// Verify an existing proof file
    Verify {
        /// Path to the proof file
        #[arg(long)]
        proof: PathBuf,

        /// Expected model hash (sha256) — verification fails if this doesn't match
        #[arg(long)]
        model_hash: String,

        /// Which guardrail model to use for verification
        #[arg(long)]
        model_name: String,
    },

    /// List recent proof history
    History {
        /// Number of entries to show
        #[arg(long, default_value_t = 10)]
        limit: usize,
    },

    /// Show loaded guardrail models from config
    Models,

    /// Validate the config file and report any issues
    ConfigCheck {
        /// Ignore config errors (exit 0 even if errors found)
        #[arg(long, default_value_t = false)]
        ignore_config_errors: bool,
    },

    /// Scan a skill for safety issues
    ScanSkill {
        /// Path to a SKILL.md file or JSON skill definition
        #[arg(long)]
        input: PathBuf,

        /// Path to optional VirusTotal report JSON
        #[arg(long)]
        vt_report: Option<PathBuf>,

        /// Generate a zero-knowledge proof
        #[arg(long, default_value_t = false)]
        prove: bool,

        /// Output format: json, summary, or receipt
        #[arg(long, default_value = "summary")]
        format: String,

        /// Save receipt to file
        #[arg(long)]
        output: Option<PathBuf>,

        /// Optional x402 payment JSON
        #[arg(long)]
        payment_json: Option<String>,
    },

    /// Batch-scan multiple skills for safety issues
    ScanBatch {
        /// Directory of *.json skill files
        #[arg(long)]
        input_dir: Option<PathBuf>,

        /// JSON array of file paths to scan
        #[arg(long)]
        manifest: Option<PathBuf>,

        /// Output directory for receipts and flagged files
        #[arg(long, default_value = ".")]
        output_dir: PathBuf,

        /// Delay between scans in milliseconds
        #[arg(long, default_value_t = 0)]
        delay_ms: u64,

        /// Generate zero-knowledge proofs
        #[arg(long, default_value_t = false)]
        prove: bool,

        /// Log progress every N skills
        #[arg(long, default_value_t = 10)]
        progress_every: usize,

        /// Optional x402 payment JSON
        #[arg(long)]
        payment_json: Option<String>,
    },

    /// Start the HTTP prover service
    Serve {
        /// Address to bind to
        #[arg(long, default_value = "127.0.0.1:8080")]
        bind: String,

        /// Maximum concurrent proof generations
        #[arg(long, default_value_t = 4)]
        max_proofs: usize,

        /// Require proof generation for all requests
        #[arg(long, default_value_t = false)]
        require_proof: bool,

        /// Rate limit in requests per minute per IP (0 = no limit)
        #[arg(long, default_value_t = 60)]
        rate_limit: u32,

        /// Path for JSONL access log
        #[arg(long, default_value = "clawguard-access.jsonl")]
        access_log: String,
    },

    /// Verify a guardrail receipt
    VerifyReceipt {
        /// Path to the receipt JSON file
        #[arg(long)]
        input: PathBuf,

        /// Optional path to skill JSON to verify input commitment
        #[arg(long)]
        skill: Option<PathBuf>,

        /// Verify the ZK proof (requires model to be available)
        #[arg(long, default_value_t = false)]
        verify_proof: bool,
    },

    /// Manage proof migrations after model updates
    MigrateProofs {
        /// Directory containing proof files (defaults to config proof_dir)
        #[arg(long)]
        proof_dir: Option<PathBuf>,

        /// Only show what would be done, don't actually archive
        #[arg(long, default_value_t = false)]
        dry_run: bool,

        /// Archive old proofs instead of just scanning
        #[arg(long, default_value_t = false)]
        archive: bool,
    },
}

#[derive(Serialize, Deserialize)]
struct CheckResult {
    decision: String,
    confidence: f64,
    model_hash: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    proof_file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    proof_generated: Option<bool>,
    action: String,
    timestamp: String,
}

#[derive(Serialize, Deserialize)]
struct VerifyResult {
    valid: bool,
    model_hash_matches: bool,
    proof_file: String,
}

/// Result of receipt verification
#[derive(Serialize)]
struct ReceiptVerifyResult {
    /// Overall verification status
    valid: bool,
    /// Receipt ID being verified
    receipt_id: String,
    /// Individual check results
    checks: ReceiptChecks,
    /// Warnings (non-fatal issues)
    #[serde(skip_serializing_if = "Vec::is_empty")]
    warnings: Vec<String>,
    /// Errors (fatal issues)
    #[serde(skip_serializing_if = "Vec::is_empty")]
    errors: Vec<String>,
}

#[derive(Serialize)]
struct ReceiptChecks {
    /// Schema version is valid
    schema_valid: bool,
    /// Nonce format is valid
    nonce_valid: bool,
    /// Model hash matches a known model
    model_known: bool,
    /// Model hash from receipt
    model_hash: String,
    /// Input commitment format is valid
    commitment_valid: bool,
    /// Input commitment matches provided skill (if skill provided)
    #[serde(skip_serializing_if = "Option::is_none")]
    commitment_matches: Option<bool>,
    /// Decision is consistent with classification
    decision_consistent: bool,
    /// ZK proof verified (if verify_proof enabled)
    #[serde(skip_serializing_if = "Option::is_none")]
    proof_valid: Option<bool>,
    /// Classification from receipt
    classification: String,
    /// Decision from receipt
    decision: String,
    /// Confidence score
    confidence: f64,
}

/// Returns an exit code: 0 for success/soft, 1 for hard denial.
fn cmd_check(
    model: String,
    action: String,
    context: String,
    prove: bool,
    dry_run: bool,
) -> Result<i32> {
    let config = load_config();
    let generate_proof = prove && !dry_run;

    // Rotate history if needed before appending
    rotate_history_if_needed(config.as_ref());

    // Determine enforcement level from config
    let enforcement = config
        .as_ref()
        .and_then(|c| c.settings.as_ref())
        .and_then(|s| s.enforcement.as_deref())
        .unwrap_or("log");
    let level = match enforcement.parse::<EnforcementLevel>() {
        Ok(l) => l,
        Err(e) => {
            eprintln!("WARNING: {e}, defaulting to Log");
            EnforcementLevel::Log
        }
    };

    let guard = GuardModel::from_cli_arg(&model)?;
    let (decision, confidence, model_hash, proof_path) =
        run_guardrail(&guard, &action, &context, generate_proof, config.as_ref())?;

    let is_deny = is_deny_decision(&decision);

    let result = CheckResult {
        decision: decision.clone(),
        confidence,
        model_hash,
        proof_file: proof_path.as_ref().map(|p| p.to_string_lossy().to_string()),
        proof_generated: if proof_path.is_some() {
            Some(true)
        } else {
            None
        },
        action,
        timestamp: Utc::now().to_rfc3339(),
    };

    println!("{}", serde_json::to_string_pretty(&result)?);

    // Append to history log
    let hist = history_path(config.as_ref());
    if let Some(parent) = hist.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut line = serde_json::to_string(&result)?;
    line.push('\n');
    fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&hist)?
        .write_all(line.as_bytes())
        .wrap_err("failed to write history")?;

    // Exit code based on enforcement
    if is_deny {
        match level {
            EnforcementLevel::Hard => {
                return Ok(1);
            }
            EnforcementLevel::Soft => {
                eprintln!("WARNING: action denied (soft enforcement)");
            }
            EnforcementLevel::Log => {
                // Already printed, no special action
            }
        }
    }

    Ok(0)
}

fn cmd_verify(proof: PathBuf, model_hash: String, model_name: String) -> Result<()> {
    let guard = GuardModel::from_name(&model_name)?;
    let trace_len = guard.max_trace_length();

    // Fail-closed: pass expected hash so verification rejects mismatches
    let valid = clawguard::proving::verify_proof_file(
        &proof,
        guard.model_fn(),
        trace_len,
        Some(&model_hash),
    )?;

    let result = VerifyResult {
        valid,
        model_hash_matches: true, // If we got here, hash matched (otherwise bail above)
        proof_file: proof.to_string_lossy().to_string(),
    };

    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

fn cmd_history(limit: usize) -> Result<()> {
    let config = load_config();
    let hist = history_path(config.as_ref());

    if !hist.exists() {
        println!("[]");
        return Ok(());
    }

    let content = fs::read_to_string(&hist)?;
    let entries: Vec<serde_json::Value> = content
        .lines()
        .filter(|l| !l.is_empty())
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect();

    let start = entries.len().saturating_sub(limit);
    let recent = &entries[start..];
    println!("{}", serde_json::to_string_pretty(recent)?);
    Ok(())
}

fn cmd_models() -> Result<()> {
    println!("Built-in guardrail models:");
    println!("  action-gatekeeper  — blocks dangerous command patterns (sudo, pipes)");
    println!("  pii-shield         — detects PII (SSN, email, phone, CC, passwords)");
    println!("  scope-guard        — blocks file access outside workspace");
    println!("  skill-safety       — classifies OpenClaw skills (SAFE/CAUTION/DANGEROUS/MALICIOUS)");
    println!();

    let config = load_config();
    match config.as_ref().and_then(|c| c.models.as_ref()) {
        Some(models) => {
            println!("Configured models:");
            for (name, model) in models {
                let path_str = model.path.as_deref().unwrap_or("(built-in)");
                let exists = if model.path.is_some() {
                    PathBuf::from(path_str).exists()
                } else {
                    true
                };
                println!(
                    "  {name}: {} (actions: {}) [{}]",
                    path_str,
                    model.actions.join(", "),
                    if exists { "found" } else { "missing" }
                );
            }
        }
        None => {
            println!("No additional models configured.");
        }
    }
    Ok(())
}

fn cmd_config_check(ignore_errors: bool) -> Result<i32> {
    let config_path = clawguard::config_dir().join("config.toml");
    println!("Config path: {}", config_path.display());

    if !config_path.exists() {
        println!("No config file found. Using defaults.");
        return Ok(0);
    }

    let content = fs::read_to_string(&config_path)
        .wrap_err("failed to read config file")?;

    let config: clawguard::GuardsConfig = match toml::from_str(&content) {
        Ok(c) => {
            println!("Config file parsed successfully.");
            c
        }
        Err(e) => {
            eprintln!("ERROR: failed to parse config: {}", e);
            return Ok(if ignore_errors { 0 } else { 1 });
        }
    };

    let issues = validate_config(&config);
    let has_errors = config_has_errors(&issues);

    if issues.is_empty() {
        println!("All checks passed.");
    } else {
        let error_count = issues.iter().filter(|i| i.is_error()).count();
        let warning_count = issues.len() - error_count;

        for issue in &issues {
            println!("  {}", issue);
        }

        println!();
        if error_count > 0 {
            println!("{} error(s), {} warning(s) found.", error_count, warning_count);
        } else {
            println!("{} warning(s) found.", warning_count);
        }
    }

    if has_errors && !ignore_errors {
        Ok(1)
    } else {
        Ok(0)
    }
}

/// Result for scan-skill command
#[derive(Serialize)]
struct ScanSkillResult {
    skill_name: String,
    skill_version: String,
    classification: String,
    decision: String,
    confidence: f64,
    reasoning: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    receipt_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    proof_generated: Option<bool>,
    features: SkillFeatures,
    model_hash: String,
    timestamp: String,
}

fn cmd_scan_skill(
    input: PathBuf,
    vt_report_path: Option<PathBuf>,
    prove: bool,
    format: String,
    output: Option<PathBuf>,
    payment_json: Option<String>,
) -> Result<i32> {
    let config = load_config();

    // Load skill from input
    let skill: Skill = if input.extension().map(|e| e == "json").unwrap_or(false) {
        let content = fs::read_to_string(&input)?;
        serde_json::from_str(&content)?
    } else {
        // Assume SKILL.md format
        skill_from_skill_md(&input)?
    };

    // Load optional VT report
    let vt_report: Option<VTReport> = if let Some(vt_path) = vt_report_path {
        let content = fs::read_to_string(&vt_path)?;
        Some(serde_json::from_str(&content)?)
    } else {
        None
    };

    // Extract features
    let features = SkillFeatures::extract(&skill, vt_report.as_ref());

    // Run safety classification
    let (classification, confidence, model_hash, proof_path) =
        run_skill_safety(&features, prove, config.as_ref())?;

    // Derive decision
    let feature_vec = features.to_normalized_vec();
    let scores = compute_scores(&feature_vec)?;
    let scores_array = [scores.safe, scores.caution, scores.dangerous, scores.malicious];
    let (decision, reasoning) = derive_decision(classification, &scores_array);

    // Create receipt if proof was generated
    let receipt = if prove {
        let (proof_bytes, program_io) = if let Some(ref path) = proof_path {
            let content = fs::read_to_string(path)?;
            let json: serde_json::Value = serde_json::from_str(&content)?;
            let bytes = json.get("proof").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let io = json.get("program_io").and_then(|v| v.as_str()).map(|s| s.to_string());
            (bytes, io)
        } else {
            (String::new(), None)
        };

        let mut r = GuardrailReceipt::new_safety_receipt(
            &skill.name,
            &skill.version,
            &features,
            classification,
            decision,
            &reasoning,
            scores.clone(),
            confidence,
            model_hash.clone(),
            proof_bytes,
            model_hash.clone(),
            None,
            program_io,
            generate_nonce(),
        );

        // Attach payment if provided
        if let Some(ref pj) = payment_json {
            let pay: PaymentInfo = serde_json::from_str(pj)
                .wrap_err("Failed to parse payment JSON")?;
            r = r.with_payment(pay);
        }

        Some(r)
    } else {
        None
    };

    // Format output
    match format.as_str() {
        "json" => {
            let result = ScanSkillResult {
                skill_name: skill.name.clone(),
                skill_version: skill.version.clone(),
                classification: classification.as_str().to_string(),
                decision: decision.as_str().to_string(),
                confidence,
                reasoning: reasoning.clone(),
                receipt_id: receipt.as_ref().map(|r| r.receipt_id.clone()),
                proof_generated: if prove { Some(true) } else { None },
                features: features.clone(),
                model_hash: model_hash.clone(),
                timestamp: Utc::now().to_rfc3339(),
            };
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        "receipt" => {
            if let Some(ref r) = receipt {
                println!("{}", serde_json::to_string_pretty(r)?);
            } else {
                eprintln!("No receipt generated. Use --prove to generate a receipt.");
            }
        }
        _ => {
            // Summary format
            println!("Skill Safety Scan Results");
            println!("========================");
            println!("Skill: {} v{}", skill.name, skill.version);
            println!();
            println!("Classification: {}", classification.as_str());
            println!("Decision:       {}", decision.as_str());
            println!("Confidence:     {:.1}%", confidence * 100.0);
            println!("Reasoning:      {}", reasoning);
            println!();
            println!("Scores:");
            println!("  SAFE:       {:.1}%", scores.safe * 100.0);
            println!("  CAUTION:    {:.1}%", scores.caution * 100.0);
            println!("  DANGEROUS:  {:.1}%", scores.dangerous * 100.0);
            println!("  MALICIOUS:  {:.1}%", scores.malicious * 100.0);
            println!();

            if classification.is_deny() {
                let flagged = FlaggedSkill::from_scan(
                    &skill.name,
                    &skill.version,
                    &features,
                    classification.as_str(),
                    confidence,
                    receipt.as_ref().map(|r| r.receipt_id.as_str()).unwrap_or("n/a"),
                    None,
                );
                println!("Risk Factors:");
                for factor in &flagged.primary_risk_factors {
                    println!("  - {}", factor);
                }
                println!();
            }

            if let Some(ref r) = receipt {
                println!("Receipt ID: {}", r.receipt_id);
            }
            println!("Model Hash: {}", model_hash);
        }
    }

    // Save receipt if output specified
    if let (Some(output_path), Some(ref r)) = (output, &receipt) {
        fs::write(&output_path, serde_json::to_string_pretty(r)?)?;
        eprintln!("Receipt saved to: {}", output_path.display());
    }

    // Exit code: 1 for DANGEROUS/MALICIOUS, 0 otherwise
    if classification.is_deny() {
        Ok(1)
    } else {
        Ok(0)
    }
}

fn cmd_scan_batch(
    input_dir: Option<PathBuf>,
    manifest: Option<PathBuf>,
    output_dir: PathBuf,
    delay_ms: u64,
    prove: bool,
    progress_every: usize,
    payment_json: Option<String>,
) -> Result<i32> {
    let config = load_config();

    // Collect skill file paths
    let skill_paths: Vec<PathBuf> = if let Some(ref dir) = input_dir {
        let mut paths: Vec<PathBuf> = fs::read_dir(dir)
            .wrap_err_with(|| format!("Cannot read input directory: {}", dir.display()))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map(|e| e == "json").unwrap_or(false))
            .collect();
        paths.sort();
        paths
    } else if let Some(ref manifest_path) = manifest {
        let content = fs::read_to_string(manifest_path)
            .wrap_err("Failed to read manifest file")?;
        let paths: Vec<String> = serde_json::from_str(&content)
            .wrap_err("Manifest must be a JSON array of file paths")?;
        paths.into_iter().map(PathBuf::from).collect()
    } else {
        eyre::bail!("Either --input-dir or --manifest must be specified");
    };

    if skill_paths.is_empty() {
        eprintln!("No skill files found.");
        return Ok(0);
    }

    eprintln!("Found {} skill files to scan", skill_paths.len());

    // Create output directories
    let receipts_dir = output_dir.join("receipts").join("safety");
    let flagged_dir = output_dir.join("flagged").join("safety");
    fs::create_dir_all(&receipts_dir)?;
    fs::create_dir_all(&flagged_dir)?;

    // Load cross-reference data if data/ directory exists
    let data_dir = std::path::Path::new("data");
    let cross_ref = if data_dir.is_dir() {
        Some(CrossReferenceData::load(data_dir))
    } else {
        None
    };

    // Parse payment info once if provided
    let payment: Option<PaymentInfo> = if let Some(ref pj) = payment_json {
        Some(serde_json::from_str(pj).wrap_err("Failed to parse payment JSON")?)
    } else {
        None
    };

    // Open JSONL output
    let jsonl_path = output_dir.join("safety-evaluations.jsonl");
    let mut jsonl_file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&jsonl_path)
        .wrap_err("Failed to open JSONL output file")?;

    let mut flagged_count = 0;
    let total = skill_paths.len();

    for (i, path) in skill_paths.iter().enumerate() {
        // Load skill
        let content = match fs::read_to_string(path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("WARNING: skipping {}: {}", path.display(), e);
                continue;
            }
        };
        let skill: Skill = match serde_json::from_str(&content) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("WARNING: skipping {} (parse error): {}", path.display(), e);
                continue;
            }
        };

        // Extract features and classify
        let features = SkillFeatures::extract(&skill, None);
        let (classification, confidence, model_hash, proof_path) =
            run_skill_safety(&features, prove, config.as_ref())?;

        let feature_vec = features.to_normalized_vec();
        let scores = compute_scores(&feature_vec)?;
        let scores_array = [scores.safe, scores.caution, scores.dangerous, scores.malicious];
        let (decision, reasoning) = derive_decision(classification, &scores_array);

        // Build receipt if proving
        let receipt = if prove {
            let (proof_bytes, program_io) = if let Some(ref pp) = proof_path {
                let pc = fs::read_to_string(pp)?;
                let json: serde_json::Value = serde_json::from_str(&pc)?;
                let bytes = json.get("proof").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let io = json.get("program_io").and_then(|v| v.as_str()).map(|s| s.to_string());
                (bytes, io)
            } else {
                (String::new(), None)
            };

            let mut r = GuardrailReceipt::new_safety_receipt(
                &skill.name,
                &skill.version,
                &features,
                classification,
                decision,
                &reasoning,
                scores.clone(),
                confidence,
                model_hash.clone(),
                proof_bytes,
                model_hash.clone(),
                None,
                program_io,
                generate_nonce(),
            );

            // Attach payment if provided
            if let Some(ref pay) = payment {
                r = r.with_payment(pay.clone());
            }

            Some(r)
        } else {
            None
        };

        let receipt_id = receipt
            .as_ref()
            .map(|r| r.receipt_id.clone())
            .unwrap_or_else(|| "n/a".to_string());

        // Build ScanResult and write JSONL line
        let scan_result = ScanResult {
            skill_name: skill.name.clone(),
            skill_version: skill.version.clone(),
            skill_uri: format!("clawhub://{}/{}", skill.name, skill.version),
            scanned_at: chrono::Utc::now(),
            features: features.clone(),
            classification: classification.as_str().to_string(),
            confidence,
            decision: decision.as_str().to_string(),
            receipt_id: receipt_id.clone(),
            receipt_file: if prove {
                format!("receipts/safety/{}.json", receipt_id)
            } else {
                String::new()
            },
            payment_tx: payment.as_ref().map(|p| p.tx_hash.clone()),
            model_hash: model_hash.clone(),
        };

        let mut line = serde_json::to_string(&scan_result)?;
        line.push('\n');
        jsonl_file.write_all(line.as_bytes())?;

        // If DANGEROUS/MALICIOUS: write flagged file
        if classification.is_deny() {
            flagged_count += 1;
            let flagged = FlaggedSkill::from_scan(
                &skill.name,
                &skill.version,
                &features,
                classification.as_str(),
                confidence,
                &receipt_id,
                cross_ref.as_ref(),
            );
            let flagged_path = flagged_dir.join(format!("{}.json", skill.name));
            fs::write(&flagged_path, serde_json::to_string_pretty(&flagged)?)?;
        }

        // Write receipt if proving
        if let Some(ref r) = receipt {
            let receipt_path = receipts_dir.join(format!("{}.json", r.receipt_id));
            fs::write(&receipt_path, serde_json::to_string_pretty(r)?)?;
        }

        // Delay between scans
        if delay_ms > 0 {
            std::thread::sleep(std::time::Duration::from_millis(delay_ms));
        }

        // Progress logging
        if progress_every > 0 && (i + 1) % progress_every == 0 {
            eprintln!("[{}/{}] scanned {} — {}", i + 1, total, skill.name, classification.as_str());
        }
    }

    // Print summary
    eprintln!();
    eprintln!("Batch scan complete:");
    eprintln!("  Total skills: {}", total);
    eprintln!("  Flagged: {}", flagged_count);
    eprintln!("  JSONL output: {}", jsonl_path.display());

    if flagged_count > 0 {
        Ok(1)
    } else {
        Ok(0)
    }
}

/// Compute softmax scores from raw classifier output
fn compute_scores(feature_vec: &[i32]) -> Result<ClassScores> {
    use onnx_tracer::tensor::Tensor;

    let model = clawguard::models::skill_safety::skill_safety_model();
    let input = Tensor::new(Some(feature_vec), &[1, 22])
        .map_err(|e| eyre::eyre!("Tensor error: {:?}", e))?;

    let result = model
        .forward(std::slice::from_ref(&input))
        .map_err(|e| eyre::eyre!("Forward error: {}", e))?;

    let data = &result.outputs[0].inner;
    let raw_scores: [i32; 4] = [
        data.get(0).copied().unwrap_or(0),
        data.get(1).copied().unwrap_or(0),
        data.get(2).copied().unwrap_or(0),
        data.get(3).copied().unwrap_or(0),
    ];

    Ok(ClassScores::from_raw_scores(&raw_scores))
}

fn cmd_serve(bind: String, max_proofs: usize, require_proof: bool, rate_limit: u32, access_log: String) -> Result<()> {
    use clawguard::server::{run_server, ServerConfig};

    let bind_addr = bind.parse()
        .wrap_err_with(|| format!("Invalid bind address: {}", bind))?;

    let config = ServerConfig {
        bind_addr,
        max_concurrent_proofs: max_proofs,
        require_proof,
        guards_config: load_config(),
        rate_limit_rpm: rate_limit,
        access_log_path: access_log,
    };

    eprintln!("Starting ClawGuard prover service...");
    eprintln!("Model: skill-safety (1,924 params)");
    eprintln!("Max concurrent proofs: {}", max_proofs);

    // Run the async server
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(run_server(config))?;

    Ok(())
}

fn cmd_verify_receipt(
    input: PathBuf,
    skill_path: Option<PathBuf>,
    verify_proof: bool,
) -> Result<()> {
    use clawguard::receipt::RECEIPT_VERSION;

    // Load the receipt
    let content = fs::read_to_string(&input)
        .wrap_err_with(|| format!("Failed to read receipt: {}", input.display()))?;
    let receipt: GuardrailReceipt = serde_json::from_str(&content)
        .wrap_err("Failed to parse receipt JSON")?;

    let mut warnings = Vec::new();
    let mut errors = Vec::new();

    // Check 1: Schema version
    let schema_valid = receipt.version == RECEIPT_VERSION;
    if !schema_valid {
        warnings.push(format!(
            "Schema version mismatch: got {}, expected {}",
            receipt.version, RECEIPT_VERSION
        ));
    }

    // Check 2: Nonce format (should be 64 hex chars = 32 bytes)
    let nonce_valid = receipt.nonce.len() == 64
        && receipt.nonce.chars().all(|c| c.is_ascii_hexdigit());
    if !nonce_valid {
        errors.push(format!(
            "Invalid nonce format: expected 64 hex chars, got {} chars",
            receipt.nonce.len()
        ));
    }

    // Check 3: Model hash matches known models
    let known_model_hashes = get_known_model_hashes();
    let model_known = known_model_hashes.contains(&receipt.guardrail.model_hash);
    if !model_known {
        warnings.push(format!(
            "Model hash not in known registry: {}",
            receipt.guardrail.model_hash
        ));
    }

    // Check 4: Commitment format
    let commitment_valid = receipt.subject.commitment.starts_with("sha256:")
        && receipt.subject.commitment.len() == 71; // "sha256:" + 64 hex
    if !commitment_valid {
        errors.push("Invalid commitment format".to_string());
    }

    // Check 5: Commitment matches skill (if provided)
    let commitment_matches = if let Some(skill_path) = skill_path {
        let skill: Skill = if skill_path.extension().map(|e| e == "json").unwrap_or(false) {
            let content = fs::read_to_string(&skill_path)?;
            serde_json::from_str(&content)?
        } else {
            skill_from_skill_md(&skill_path)?
        };
        let features = SkillFeatures::extract(&skill, None);
        let matches = receipt.verify_commitment(&features);
        if !matches {
            errors.push("Input commitment does not match provided skill".to_string());
        }
        Some(matches)
    } else {
        None
    };

    // Check 6: Decision consistency
    let decision_consistent = match receipt.evaluation.classification.as_str() {
        "SAFE" | "CAUTION" => receipt.evaluation.decision == "allow",
        "DANGEROUS" | "MALICIOUS" => receipt.evaluation.decision == "deny",
        _ => {
            warnings.push(format!(
                "Unknown classification: {}",
                receipt.evaluation.classification
            ));
            true // Can't verify unknown classifications
        }
    };
    if !decision_consistent {
        errors.push(format!(
            "Decision '{}' inconsistent with classification '{}'",
            receipt.evaluation.decision, receipt.evaluation.classification
        ));
    }

    // Check 7: ZK proof verification (if requested and proof exists)
    let proof_valid = if verify_proof && !receipt.proof.proof_bytes.is_empty() {
        // Try to verify the proof
        match verify_receipt_proof(&receipt) {
            Ok(valid) => {
                if !valid {
                    errors.push("ZK proof verification failed".to_string());
                }
                Some(valid)
            }
            Err(e) => {
                errors.push(format!("ZK proof verification error: {}", e));
                Some(false)
            }
        }
    } else if verify_proof && receipt.proof.proof_bytes.is_empty() {
        warnings.push("No proof bytes in receipt to verify".to_string());
        None
    } else {
        None
    };

    // Overall validity
    let valid = errors.is_empty() && schema_valid && nonce_valid && commitment_valid && decision_consistent;

    let result = ReceiptVerifyResult {
        valid,
        receipt_id: receipt.receipt_id.clone(),
        checks: ReceiptChecks {
            schema_valid,
            nonce_valid,
            model_known,
            model_hash: receipt.guardrail.model_hash.clone(),
            commitment_valid,
            commitment_matches,
            decision_consistent,
            proof_valid,
            classification: receipt.evaluation.classification.clone(),
            decision: receipt.evaluation.decision.clone(),
            confidence: receipt.evaluation.confidence,
        },
        warnings,
        errors,
    };

    println!("{}", serde_json::to_string_pretty(&result)?);

    // Also print human-readable summary
    eprintln!();
    if result.valid {
        eprintln!("✅ Receipt verification PASSED");
        eprintln!("   Receipt ID: {}", receipt.receipt_id);
        eprintln!("   Subject: {}", receipt.subject.description);
        eprintln!("   Classification: {} ({})",
            receipt.evaluation.classification,
            receipt.evaluation.decision
        );
        eprintln!("   Confidence: {:.1}%", receipt.evaluation.confidence * 100.0);
    } else {
        eprintln!("❌ Receipt verification FAILED");
        for err in &result.errors {
            eprintln!("   Error: {}", err);
        }
    }
    if !result.warnings.is_empty() {
        eprintln!();
        for warn in &result.warnings {
            eprintln!("   ⚠️  {}", warn);
        }
    }

    Ok(())
}

#[derive(Deserialize)]
struct ModelRegistry {
    models: Vec<ModelRegistryEntry>,
}

#[derive(Deserialize)]
struct ModelRegistryEntry {
    #[allow(dead_code)]
    name: String,
    hash: String,
    #[allow(dead_code)]
    version: Option<String>,
    #[allow(dead_code)]
    params: Option<u32>,
    #[allow(dead_code)]
    architecture: Option<String>,
    #[allow(dead_code)]
    trained_at: Option<String>,
}

/// Get known model hashes for verification.
///
/// Tries loading from `data/registry.json` first, then falls back to
/// computing hashes from model functions.
fn get_known_model_hashes() -> Vec<String> {
    // Try loading from data/registry.json
    let registry_path = std::path::Path::new("data/registry.json");
    if let Ok(content) = fs::read_to_string(registry_path) {
        if let Ok(registry) = serde_json::from_str::<ModelRegistry>(&content) {
            let registry_hashes: Vec<String> = registry
                .models
                .iter()
                .filter(|m| m.hash.starts_with("sha256:"))
                .map(|m| m.hash.clone())
                .collect();
            if !registry_hashes.is_empty() {
                // Merge with computed hashes to cover all models
                let mut hashes = registry_hashes;
                hashes.extend(get_computed_model_hashes());
                hashes.sort();
                hashes.dedup();
                return hashes;
            }
        }
    }

    // Fall back to computing hashes from model functions
    get_computed_model_hashes()
}

fn get_computed_model_hashes() -> Vec<String> {
    use clawguard::hash_model_fn;
    use clawguard::models::skill_safety::skill_safety_model;
    use clawguard::models::action_gatekeeper::action_gatekeeper_model;
    use clawguard::models::pii_shield::pii_shield_model;
    use clawguard::models::scope_guard::scope_guard_model;

    vec![
        hash_model_fn(skill_safety_model),
        hash_model_fn(action_gatekeeper_model),
        hash_model_fn(pii_shield_model),
        hash_model_fn(scope_guard_model),
    ]
}

/// Verify the ZK proof in a receipt
fn verify_receipt_proof(receipt: &GuardrailReceipt) -> Result<bool> {
    use base64::{engine::general_purpose::STANDARD as B64, Engine};
    use clawguard::models::skill_safety::skill_safety_model;
    use onnx_tracer::ProgramIO;

    // For now, we only support verifying skill-safety proofs
    if receipt.guardrail.domain != "safety" {
        eyre::bail!("Only safety domain proofs are supported for verification");
    }

    // Verify model hash matches
    let expected_hash = clawguard::hash_model_fn(skill_safety_model);
    if receipt.guardrail.model_hash != expected_hash {
        return Ok(false);
    }

    // Check that proof bytes exist
    if receipt.proof.proof_bytes.is_empty() {
        return Ok(false);
    }

    // Check if we have program_io for full verification
    let program_io_str = match &receipt.proof.program_io {
        Some(s) => s,
        None => {
            // Without program_io, we can only verify model hash and proof existence
            // This is a partial verification
            eprintln!("WARNING: Receipt missing program_io, performing partial verification");
            return Ok(true);
        }
    };

    // Decode proof bytes from base64
    let proof_bytes = B64.decode(&receipt.proof.proof_bytes)
        .wrap_err("Failed to decode proof bytes from base64")?;

    // Parse program IO
    let program_io: ProgramIO = serde_json::from_str(program_io_str)
        .wrap_err("Failed to parse program_io")?;

    // Perform full proof verification
    let max_trace_length = 1 << 16; // Same as skill-safety model
    clawguard::proving::verify_proof_from_bytes(
        &proof_bytes,
        skill_safety_model,
        program_io,
        max_trace_length,
    )
}

fn cmd_migrate_proofs(
    proof_dir_override: Option<PathBuf>,
    dry_run: bool,
    archive: bool,
) -> Result<()> {
    use clawguard::models::skill_safety::skill_safety_model;

    let config = load_config();
    let proof_directory = proof_dir_override.unwrap_or_else(|| clawguard::proof_dir(config.as_ref()));
    let current_model_hash = clawguard::hash_model_fn(skill_safety_model);

    println!("Proof directory: {}", proof_directory.display());
    println!("Current model hash: {}", current_model_hash);
    println!();

    if !proof_directory.exists() {
        println!("Proof directory does not exist. Nothing to migrate.");
        return Ok(());
    }

    if archive {
        println!("{}archiving old proofs...", if dry_run { "[DRY RUN] " } else { "" });
        let results = clawguard::migration::archive_old_proofs(
            &proof_directory,
            &current_model_hash,
            dry_run,
        )?;

        if results.is_empty() {
            println!("No proofs need archiving.");
        } else {
            for result in &results {
                println!(
                    "  {} -> {}",
                    result.proof_path,
                    result.new_path.as_deref().unwrap_or("(error)")
                );
                if let Some(ref err) = result.error {
                    println!("    Error: {}", err);
                }
            }
            let archived_count = results.iter().filter(|r| r.migrated).count();
            let would_archive = results.iter().filter(|r| r.status == "would_archive").count();
            if dry_run {
                println!("\n{} proofs would be archived.", would_archive);
            } else {
                println!("\n{} proofs archived.", archived_count);
            }
        }
    } else {
        println!("Scanning proofs for migration status...");
        let results = clawguard::migration::scan_proofs_for_migration(
            &proof_directory,
            &current_model_hash,
        )?;

        if results.is_empty() {
            println!("No proof files found.");
        } else {
            let mut needs_migration = 0;
            let mut current = 0;
            let mut other = 0;

            for result in &results {
                let status_display = if result.status.starts_with("needs_migration") {
                    needs_migration += 1;
                    "⚠️  NEEDS MIGRATION"
                } else if result.status == "current" {
                    current += 1;
                    "✅ current"
                } else {
                    other += 1;
                    &result.status
                };
                println!("  {} - {}", result.proof_path, status_display);
            }

            println!();
            println!("Summary: {} current, {} need migration, {} other", current, needs_migration, other);

            if needs_migration > 0 {
                println!();
                println!("To archive old proofs, run: clawguard migrate-proofs --archive");
                println!("To preview without changes: clawguard migrate-proofs --archive --dry-run");
            }
        }
    }

    Ok(())
}

fn main() {
    // Initialize policy rules from config if present
    if let Some(config) = load_config() {
        if let Some(rule_configs) = &config.rules {
            let policy_rules: Vec<clawguard::rules::PolicyRule> = rule_configs
                .iter()
                .filter_map(|rc| match clawguard::rules::PolicyRule::from_config(rc) {
                    Ok(rule) => Some(rule),
                    Err(e) => {
                        eprintln!("WARNING: failed to load policy rule '{}': {}", rc.name, e);
                        None
                    }
                })
                .collect();
            if !policy_rules.is_empty() {
                if !clawguard::rules::init_policy(&policy_rules) {
                    eprintln!("WARNING: policy rules already initialized, skipping re-init");
                }
            }
        }
    }

    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Check {
            model,
            action,
            context,
            prove,
            dry_run,
        } => {
            match cmd_check(model, action, context, prove, dry_run) {
                Ok(code) => {
                    if code != 0 {
                        std::process::exit(code);
                    }
                    Ok(())
                }
                Err(e) => Err(e),
            }
        }
        Commands::Verify {
            proof,
            model_hash,
            model_name,
        } => cmd_verify(proof, model_hash, model_name),
        Commands::History { limit } => cmd_history(limit),
        Commands::Models => cmd_models(),
        Commands::ConfigCheck { ignore_config_errors } => {
            match cmd_config_check(ignore_config_errors) {
                Ok(code) => {
                    if code != 0 {
                        std::process::exit(code);
                    }
                    Ok(())
                }
                Err(e) => Err(e),
            }
        }
        Commands::ScanSkill {
            input,
            vt_report,
            prove,
            format,
            output,
            payment_json,
        } => {
            match cmd_scan_skill(input, vt_report, prove, format, output, payment_json) {
                Ok(code) => {
                    if code != 0 {
                        std::process::exit(code);
                    }
                    Ok(())
                }
                Err(e) => Err(e),
            }
        }
        Commands::ScanBatch {
            input_dir,
            manifest,
            output_dir,
            delay_ms,
            prove,
            progress_every,
            payment_json,
        } => {
            match cmd_scan_batch(input_dir, manifest, output_dir, delay_ms, prove, progress_every, payment_json) {
                Ok(code) => {
                    if code != 0 {
                        std::process::exit(code);
                    }
                    Ok(())
                }
                Err(e) => Err(e),
            }
        }
        Commands::Serve {
            bind,
            max_proofs,
            require_proof,
            rate_limit,
            access_log,
        } => cmd_serve(bind, max_proofs, require_proof, rate_limit, access_log),
        Commands::VerifyReceipt {
            input,
            skill,
            verify_proof,
        } => cmd_verify_receipt(input, skill, verify_proof),
        Commands::MigrateProofs {
            proof_dir,
            dry_run,
            archive,
        } => cmd_migrate_proofs(proof_dir, dry_run, archive),
    };

    if let Err(e) = result {
        eprintln!("Error: {e:?}");
        std::process::exit(1);
    }
}
