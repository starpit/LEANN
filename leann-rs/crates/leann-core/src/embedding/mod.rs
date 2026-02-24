pub mod manager;
pub mod onnx;

#[cfg(feature = "embedding-remote")]
pub mod gemini;
#[cfg(feature = "embedding-remote")]
pub mod ollama;
#[cfg(feature = "embedding-remote")]
pub mod openai;

use anyhow::Result;
use ndarray::Array2;
use std::collections::HashMap;

/// Trait for embedding computation backends.
pub trait EmbeddingProvider: Send + Sync {
    /// Compute embeddings for a batch of text chunks.
    fn compute_embeddings(&self, chunks: &[String]) -> Result<Array2<f32>>;

    /// Get the dimensionality of the embeddings.
    fn dimensions(&self) -> usize;

    /// Get the provider name.
    fn name(&self) -> &str;
}

/// Embedding mode enum matching Python's embedding modes.
#[derive(Debug, Clone, PartialEq)]
pub enum EmbeddingMode {
    SentenceTransformers,
    OpenAI,
    Ollama,
    Gemini,
    Mlx,
}

impl EmbeddingMode {
    pub fn from_str_lossy(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "openai" => EmbeddingMode::OpenAI,
            "ollama" => EmbeddingMode::Ollama,
            "gemini" => EmbeddingMode::Gemini,
            "mlx" => EmbeddingMode::Mlx,
            _ => EmbeddingMode::SentenceTransformers,
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            EmbeddingMode::SentenceTransformers => "sentence-transformers",
            EmbeddingMode::OpenAI => "openai",
            EmbeddingMode::Ollama => "ollama",
            EmbeddingMode::Gemini => "gemini",
            EmbeddingMode::Mlx => "mlx",
        }
    }
}

/// Create an embedding provider from mode, model name, and options map.
///
/// The `options` map may contain provider-specific keys:
/// - `"host"` — Ollama host override
/// - `"api_key"` — API key for OpenAI/Gemini
/// - `"base_url"` — Base URL for OpenAI-compatible services
#[cfg(feature = "embedding-remote")]
pub fn create_embedding_provider(
    mode: &EmbeddingMode,
    model: &str,
    options: &HashMap<String, serde_json::Value>,
) -> Result<Box<dyn EmbeddingProvider>> {
    match mode {
        EmbeddingMode::OpenAI => {
            let api_key = options.get("api_key").and_then(|v| v.as_str());
            let base_url = options.get("base_url").and_then(|v| v.as_str());
            let provider = openai::OpenAiEmbedding::new(model, api_key, base_url, None)?;
            Ok(Box::new(provider))
        }
        EmbeddingMode::Ollama => {
            let host = options.get("host").and_then(|v| v.as_str());
            let provider = ollama::OllamaEmbedding::new(model, host);
            Ok(Box::new(provider))
        }
        EmbeddingMode::Gemini => {
            let api_key = options.get("api_key").and_then(|v| v.as_str());
            let provider = gemini::GeminiEmbedding::new(model, api_key)?;
            Ok(Box::new(provider))
        }
        _ => {
            // sentence-transformers / mlx: try OpenAI, fall back to Ollama
            if let Ok(provider) =
                openai::OpenAiEmbedding::new("text-embedding-3-small", None, None, None)
            {
                Ok(Box::new(provider))
            } else {
                let provider = ollama::OllamaEmbedding::new("nomic-embed-text", None);
                Ok(Box::new(provider))
            }
        }
    }
}
