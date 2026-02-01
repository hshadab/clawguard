//! ONNX model support — load trained ONNX classifiers with metadata.

use eyre::Result;
use onnx_tracer::graph::model::Model;
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

// ---------------------------------------------------------------------------
// OnceLock + Mutex for ONNX model path (Step 3.1)
// ---------------------------------------------------------------------------

static ONNX_MODEL_PATH: OnceLock<Mutex<PathBuf>> = OnceLock::new();

pub fn set_onnx_path(path: PathBuf) {
    match ONNX_MODEL_PATH.get() {
        Some(mutex) => {
            *mutex.lock().unwrap() = path;
        }
        None => {
            let _ = ONNX_MODEL_PATH.set(Mutex::new(path));
        }
    }
}

pub fn load_onnx_model() -> Model {
    let path = ONNX_MODEL_PATH
        .get()
        .expect("ONNX path not set — call set_onnx_path() first")
        .lock()
        .unwrap();
    onnx_tracer::model(&*path)
}

// ---------------------------------------------------------------------------
// Model metadata (Step 3.2)
// ---------------------------------------------------------------------------

#[derive(Deserialize, Debug, Clone)]
pub struct OnnxModelMeta {
    pub input_shape: Vec<usize>,
    #[serde(default = "default_encoding")]
    pub encoding: String,
    #[serde(default = "default_labels")]
    pub labels: Vec<String>,
    pub scale: Option<i32>,
    pub max_trace_length: Option<usize>,
}

fn default_encoding() -> String {
    "action".into()
}

fn default_labels() -> Vec<String> {
    vec!["DENIED".into(), "APPROVED".into()]
}

impl OnnxModelMeta {
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| eyre::eyre!("failed to read meta file {}: {}", path.display(), e))?;
        let meta: Self = toml::from_str(&content)
            .map_err(|e| eyre::eyre!("failed to parse meta file: {}", e))?;
        Ok(meta)
    }
}
