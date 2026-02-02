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
    validate_config, GuardModel,
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
    ConfigCheck,
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

fn cmd_config_check() -> Result<()> {
    let config_path = clawguard::config_dir().join("config.toml");
    println!("Config path: {}", config_path.display());

    if !config_path.exists() {
        println!("No config file found. Using defaults.");
        return Ok(());
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
            return Ok(());
        }
    };

    let issues = validate_config(&config);
    if issues.is_empty() {
        println!("All checks passed.");
    } else {
        for issue in &issues {
            println!("  {}", issue);
        }
        println!("\n{} issue(s) found.", issues.len());
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
        Commands::ConfigCheck => cmd_config_check(),
    };

    if let Err(e) = result {
        eprintln!("Error: {e:?}");
        std::process::exit(1);
    }
}
