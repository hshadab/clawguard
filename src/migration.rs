//! Model version migration utilities for ClawGuard.
//!
//! This module provides tools for managing proof compatibility across model version changes.
//! When model weights or architecture change, existing proofs become invalid. This module
//! helps detect version mismatches and provides guidance for re-proving.
//!
//! # Version Bumping Policy
//!
//! The model hash version prefix (e.g., "v1") should be bumped when:
//! - Model weights change
//! - Model architecture changes
//! - Serialization format changes
//! - Any change that would invalidate existing proofs
//!
//! Minor changes that don't affect proof validity (e.g., documentation) don't require bumps.

use eyre::{Result, WrapErr};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Current proof format version.
/// Bump this when proof structure changes in incompatible ways.
pub const PROOF_FORMAT_VERSION: &str = "0.1.0";

/// Supported proof format versions (current + older compatible versions).
pub const SUPPORTED_VERSIONS: &[&str] = &["0.1.0"];

/// Proof metadata for migration checking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofMetadata {
    pub version: String,
    pub backend: String,
    pub model_hash: String,
    pub timestamp: String,
}

/// Result of checking if a proof needs migration.
#[derive(Debug, Clone)]
pub enum MigrationStatus {
    /// Proof is current and valid
    Current,
    /// Proof is from an older but supported version
    Supported { from_version: String },
    /// Proof needs re-generation (model or format changed)
    NeedsMigration { reason: String },
    /// Cannot determine status (missing metadata)
    Unknown { reason: String },
}

impl MigrationStatus {
    pub fn needs_migration(&self) -> bool {
        matches!(self, MigrationStatus::NeedsMigration { .. })
    }

    pub fn is_current(&self) -> bool {
        matches!(self, MigrationStatus::Current)
    }
}

/// Check if a proof file needs migration.
pub fn check_proof_status(proof_path: &Path, current_model_hash: &str) -> Result<MigrationStatus> {
    let content = fs::read_to_string(proof_path)
        .wrap_err_with(|| format!("Failed to read proof file: {}", proof_path.display()))?;

    let json: serde_json::Value = serde_json::from_str(&content)
        .wrap_err("Failed to parse proof JSON")?;

    // Extract version
    let version = json
        .get("version")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    // Extract model hash
    let model_hash = json
        .get("model_hash")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    // Check model hash first (most common reason for migration)
    if model_hash != current_model_hash && model_hash != "unknown" {
        return Ok(MigrationStatus::NeedsMigration {
            reason: format!(
                "Model hash mismatch: proof has '{}', current is '{}'",
                model_hash, current_model_hash
            ),
        });
    }

    // Check version compatibility
    if version == "unknown" {
        return Ok(MigrationStatus::Unknown {
            reason: "Proof file missing version field".to_string(),
        });
    }

    if version == PROOF_FORMAT_VERSION {
        return Ok(MigrationStatus::Current);
    }

    if SUPPORTED_VERSIONS.contains(&version) {
        return Ok(MigrationStatus::Supported {
            from_version: version.to_string(),
        });
    }

    Ok(MigrationStatus::NeedsMigration {
        reason: format!(
            "Unsupported proof version '{}', current is '{}'",
            version, PROOF_FORMAT_VERSION
        ),
    })
}

/// Result of a migration operation.
#[derive(Debug, Clone, Serialize)]
pub struct MigrationResult {
    pub proof_path: String,
    pub status: String,
    pub migrated: bool,
    pub new_path: Option<String>,
    pub error: Option<String>,
}

/// Scan a directory for proofs that need migration.
pub fn scan_proofs_for_migration(
    proof_dir: &Path,
    current_model_hash: &str,
) -> Result<Vec<MigrationResult>> {
    let mut results = Vec::new();

    if !proof_dir.exists() {
        return Ok(results);
    }

    for entry in fs::read_dir(proof_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.extension().map(|e| e == "json").unwrap_or(false) {
            let status = match check_proof_status(&path, current_model_hash) {
                Ok(s) => s,
                Err(e) => {
                    results.push(MigrationResult {
                        proof_path: path.display().to_string(),
                        status: "error".to_string(),
                        migrated: false,
                        new_path: None,
                        error: Some(e.to_string()),
                    });
                    continue;
                }
            };

            let (status_str, needs_migration) = match &status {
                MigrationStatus::Current => ("current".to_string(), false),
                MigrationStatus::Supported { from_version } => {
                    (format!("supported (v{})", from_version), false)
                }
                MigrationStatus::NeedsMigration { reason } => {
                    (format!("needs_migration: {}", reason), true)
                }
                MigrationStatus::Unknown { reason } => {
                    (format!("unknown: {}", reason), false)
                }
            };

            results.push(MigrationResult {
                proof_path: path.display().to_string(),
                status: status_str.to_string(),
                migrated: false,
                new_path: None,
                error: if needs_migration {
                    Some("Re-run with --prove to generate a new proof".to_string())
                } else {
                    None
                },
            });
        }
    }

    Ok(results)
}

/// Archive old proofs to a backup directory.
pub fn archive_old_proofs(
    proof_dir: &Path,
    current_model_hash: &str,
    dry_run: bool,
) -> Result<Vec<MigrationResult>> {
    let mut results = Vec::new();
    let archive_dir = proof_dir.join("archived");

    if !dry_run && !archive_dir.exists() {
        fs::create_dir_all(&archive_dir)?;
    }

    if !proof_dir.exists() {
        return Ok(results);
    }

    for entry in fs::read_dir(proof_dir)? {
        let entry = entry?;
        let path = entry.path();

        // Skip directories and non-json files
        if path.is_dir() || path.extension().map(|e| e != "json").unwrap_or(true) {
            continue;
        }

        let status = match check_proof_status(&path, current_model_hash) {
            Ok(s) => s,
            Err(e) => {
                results.push(MigrationResult {
                    proof_path: path.display().to_string(),
                    status: "error".to_string(),
                    migrated: false,
                    new_path: None,
                    error: Some(e.to_string()),
                });
                continue;
            }
        };

        if status.needs_migration() {
            let filename = path.file_name().unwrap();
            let archive_path = archive_dir.join(filename);

            if dry_run {
                results.push(MigrationResult {
                    proof_path: path.display().to_string(),
                    status: "would_archive".to_string(),
                    migrated: false,
                    new_path: Some(archive_path.display().to_string()),
                    error: None,
                });
            } else {
                match fs::rename(&path, &archive_path) {
                    Ok(()) => {
                        results.push(MigrationResult {
                            proof_path: path.display().to_string(),
                            status: "archived".to_string(),
                            migrated: true,
                            new_path: Some(archive_path.display().to_string()),
                            error: None,
                        });
                    }
                    Err(e) => {
                        results.push(MigrationResult {
                            proof_path: path.display().to_string(),
                            status: "archive_failed".to_string(),
                            migrated: false,
                            new_path: None,
                            error: Some(e.to_string()),
                        });
                    }
                }
            }
        }
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_proof_status_current() {
        let tmp = std::env::temp_dir().join("clawguard_migration_test");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();

        let proof_path = tmp.join("test.proof.json");
        let current_hash = "sha256:abc123";

        let proof_content = serde_json::json!({
            "version": PROOF_FORMAT_VERSION,
            "backend": "jolt-atlas",
            "model_hash": current_hash,
            "timestamp": "2024-01-01T00:00:00Z",
        });

        fs::write(&proof_path, serde_json::to_string_pretty(&proof_content).unwrap()).unwrap();

        let status = check_proof_status(&proof_path, current_hash).unwrap();
        assert!(status.is_current());

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_check_proof_status_needs_migration() {
        let tmp = std::env::temp_dir().join("clawguard_migration_test2");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();

        let proof_path = tmp.join("old.proof.json");

        let proof_content = serde_json::json!({
            "version": PROOF_FORMAT_VERSION,
            "backend": "jolt-atlas",
            "model_hash": "sha256:old_hash",
            "timestamp": "2024-01-01T00:00:00Z",
        });

        fs::write(&proof_path, serde_json::to_string_pretty(&proof_content).unwrap()).unwrap();

        let status = check_proof_status(&proof_path, "sha256:new_hash").unwrap();
        assert!(status.needs_migration());

        let _ = fs::remove_dir_all(&tmp);
    }
}
