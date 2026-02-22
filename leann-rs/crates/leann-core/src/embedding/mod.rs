pub mod manager;
pub mod onnx;

#[cfg(feature = "embedding-zmq")]
pub mod client;
#[cfg(feature = "embedding-remote")]
pub mod gemini;
#[cfg(feature = "embedding-remote")]
pub mod ollama;
#[cfg(feature = "embedding-remote")]
pub mod openai;
#[cfg(feature = "embedding-zmq")]
pub mod server;

use anyhow::Result;
use ndarray::Array2;

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
