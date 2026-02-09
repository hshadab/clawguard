//! ZK proof generation and verification using jolt-atlas.

use ark_bn254::Fr;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use base64::{engine::general_purpose::STANDARD as B64, Engine};
use eyre::{bail, Result, WrapErr};
use jolt_core::poly::commitment::dory::DoryCommitmentScheme;
use jolt_core::transcripts::KeccakTranscript;
use onnx_tracer::graph::model::Model;
use onnx_tracer::tensor::Tensor;
use onnx_tracer::ProgramIO;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use zkml_jolt_core::jolt::JoltSNARK;

type PCS = DoryCommitmentScheme;
type Snark = JoltSNARK<Fr, PCS, KeccakTranscript>;

/// Per-model mutexes to allow concurrent proving across different models
/// while still serializing proofs for the same model (OnceLock state limitation).
static MODEL_LOCKS: std::sync::LazyLock<Mutex<HashMap<String, std::sync::Arc<Mutex<()>>>>> =
    std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));

fn get_model_lock(model_name: &str) -> std::sync::Arc<Mutex<()>> {
    let mut locks = MODEL_LOCKS.lock().unwrap();
    locks
        .entry(model_name.to_string())
        .or_insert_with(|| std::sync::Arc::new(Mutex::new(())))
        .clone()
}

/// Generate a proof and save it to a JSON file.
///
/// Returns the path to the proof file and the program IO.
pub fn prove_and_save(
    model_fn: fn() -> Model,
    input: &Tensor<i32>,
    proof_dir: &Path,
    model_hash: &str,
    max_trace_length: usize,
    model_name: &str,
) -> Result<(PathBuf, ProgramIO)> {
    let lock = get_model_lock(model_name);
    let _guard = lock.lock().unwrap();

    let preprocessing = Snark::prover_preprocess(model_fn, max_trace_length);

    let (snark, program_io, _debug_info) = Snark::prove(&preprocessing, model_fn, input);

    // Verify locally before saving
    let verifier_preprocessing = (&preprocessing).into();
    snark
        .clone()
        .verify(&verifier_preprocessing, program_io.clone(), None)
        .wrap_err("local verification failed")?;

    // Serialize proof
    let mut proof_bytes = Vec::new();
    snark
        .serialize_compressed(&mut proof_bytes)
        .wrap_err("failed to serialize proof")?;

    let program_io_json =
        serde_json::to_string(&program_io).wrap_err("failed to serialize program_io")?;

    // Write proof file
    fs::create_dir_all(proof_dir)?;
    let filename = format!(
        "{}.proof.json",
        chrono::Utc::now().format("%Y-%m-%dT%H-%M-%SZ")
    );
    let path = proof_dir.join(&filename);

    let proof_data = serde_json::json!({
        "version": "0.1.0",
        "backend": "jolt-atlas",
        "model_hash": model_hash,
        "proof": B64.encode(&proof_bytes),
        "program_io": program_io_json,
        "timestamp": chrono::Utc::now().to_rfc3339(),
    });
    fs::write(&path, serde_json::to_string_pretty(&proof_data)?)
        .wrap_err("failed to write proof file")?;

    Ok((path, program_io))
}

/// Verify a proof from raw bytes and program IO.
///
/// This is used for verifying proofs embedded in receipts.
pub fn verify_proof_from_bytes(
    proof_bytes: &[u8],
    model_fn: fn() -> Model,
    program_io: ProgramIO,
    max_trace_length: usize,
) -> Result<bool> {
    let snark = Snark::deserialize_compressed(proof_bytes)
        .map_err(|e| eyre::eyre!("failed to deserialize proof: {}", e))?;

    let preprocessing = Snark::prover_preprocess(model_fn, max_trace_length);
    let verifier_preprocessing = (&preprocessing).into();

    match snark.verify(&verifier_preprocessing, program_io, None) {
        Ok(()) => Ok(true),
        Err(_) => Ok(false),
    }
}

/// Verify a proof from a saved JSON file.
///
/// Fails if `expected_model_hash` does not match the hash stored in the proof file
/// (fail-closed verification).
pub fn verify_proof_file(
    proof_path: &Path,
    model_fn: fn() -> Model,
    max_trace_length: usize,
    expected_model_hash: Option<&str>,
) -> Result<bool> {
    let content = fs::read_to_string(proof_path).wrap_err("failed to read proof file")?;
    let data: serde_json::Value = serde_json::from_str(&content)?;

    // Fail-closed: if an expected hash is provided, it must match the proof's stored hash
    let stored_hash = data
        .get("model_hash")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    if let Some(expected) = expected_model_hash {
        if stored_hash != expected {
            bail!(
                "model hash mismatch: proof contains '{}' but expected '{}'. \
                 The model may have been swapped since this proof was generated.",
                stored_hash,
                expected
            );
        }
    }

    let proof_b64 = data
        .get("proof")
        .and_then(|v| v.as_str())
        .ok_or_else(|| eyre::eyre!("missing proof field"))?;
    let proof_bytes = B64.decode(proof_b64).wrap_err("invalid base64 in proof")?;

    let program_io_str = data
        .get("program_io")
        .and_then(|v| v.as_str())
        .ok_or_else(|| eyre::eyre!("missing program_io field"))?;
    let program_io: ProgramIO =
        serde_json::from_str(program_io_str).wrap_err("invalid program_io")?;

    let snark = Snark::deserialize_compressed(proof_bytes.as_slice())
        .map_err(|e| eyre::eyre!("failed to deserialize proof: {}", e))?;

    let preprocessing = Snark::prover_preprocess(model_fn, max_trace_length);
    let verifier_preprocessing = (&preprocessing).into();

    match snark.verify(&verifier_preprocessing, program_io, None) {
        Ok(()) => Ok(true),
        Err(_) => Ok(false),
    }
}
