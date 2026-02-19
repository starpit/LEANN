use anyhow::Result;
use ndarray::Array2;
use std::path::PathBuf;

use super::EmbeddingProvider;

/// ONNX Runtime embedding provider for local sentence-transformer models.
///
/// This provider loads an ONNX-exported model and runs inference locally
/// without needing Python, torch, or sentence-transformers.
pub struct OnnxEmbedding {
    model_path: PathBuf,
    dimensions: usize,
    max_seq_length: usize,
    model_name: String,
}

impl OnnxEmbedding {
    /// Create a new ONNX embedding provider.
    ///
    /// `model_path` should point to a directory containing:
    /// - `model.onnx` or `model_optimized.onnx`
    /// - `tokenizer.json` (HuggingFace tokenizer)
    pub fn new(model_path: &str, dimensions: Option<usize>) -> Result<Self> {
        let path = PathBuf::from(model_path);

        if !path.exists() {
            anyhow::bail!("ONNX model path does not exist: {}", model_path);
        }

        // Check for model file
        let model_file = if path.join("model_optimized.onnx").exists() {
            path.join("model_optimized.onnx")
        } else if path.join("model.onnx").exists() {
            path.join("model.onnx")
        } else {
            anyhow::bail!("No ONNX model file found in {}", model_path);
        };

        let _tokenizer_file = path.join("tokenizer.json");

        // Default dimensions for common models
        let dimensions = dimensions.unwrap_or(768);

        Ok(Self {
            model_path: path,
            dimensions,
            max_seq_length: 512,
            model_name: model_path.to_string(),
        })
    }

    /// Get the path to the ONNX model file.
    pub fn model_file(&self) -> PathBuf {
        if self.model_path.join("model_optimized.onnx").exists() {
            self.model_path.join("model_optimized.onnx")
        } else {
            self.model_path.join("model.onnx")
        }
    }

    /// Get the path to the tokenizer file.
    pub fn tokenizer_file(&self) -> PathBuf {
        self.model_path.join("tokenizer.json")
    }
}

impl EmbeddingProvider for OnnxEmbedding {
    fn compute_embeddings(&self, chunks: &[String]) -> Result<Array2<f32>> {
        if chunks.is_empty() {
            return Ok(Array2::zeros((0, self.dimensions)));
        }

        // Note: Full ONNX Runtime integration requires the `ort` crate and
        // a compiled ONNX Runtime library. This is a placeholder that shows
        // the intended API. To enable, add `ort = "2"` to dependencies and
        // uncomment the implementation below.
        //
        // The full implementation would:
        // 1. Load tokenizer from tokenizer.json
        // 2. Tokenize input texts (input_ids, attention_mask, token_type_ids)
        // 3. Run ONNX session inference
        // 4. Mean-pool the token embeddings using attention_mask
        // 5. Optionally normalize to unit length

        anyhow::bail!(
            "ONNX Runtime inference not yet enabled. \
             Install the ort crate and ONNX Runtime library, \
             or use --embedding-mode openai/ollama instead. \
             Model path: {}",
            self.model_path.display()
        )
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn name(&self) -> &str {
        &self.model_name
    }
}
