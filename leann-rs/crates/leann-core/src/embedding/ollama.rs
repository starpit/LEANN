use anyhow::{Context, Result};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use super::EmbeddingProvider;
use crate::settings::resolve_ollama_host;

/// Ollama embedding API client.
pub struct OllamaEmbedding {
    model: String,
    host: String,
    dimensions: usize,
    client: reqwest::blocking::Client,
}

#[derive(Serialize)]
struct OllamaEmbeddingRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Deserialize)]
struct OllamaEmbeddingResponse {
    embeddings: Vec<Vec<f32>>,
}

impl OllamaEmbedding {
    pub fn new(model: &str, host: Option<&str>) -> Self {
        Self {
            model: model.to_string(),
            host: resolve_ollama_host(host),
            dimensions: 768, // Default, will be updated on first compute
            client: reqwest::blocking::Client::new(),
        }
    }

    /// Send a single batch to the Ollama /api/embed endpoint.
    fn embed_batch(&self, batch: &[String]) -> Result<Vec<Vec<f32>>, OllamaBatchError> {
        let request = OllamaEmbeddingRequest {
            model: self.model.clone(),
            input: batch.to_vec(),
        };

        let response = self
            .client
            .post(format!("{}/api/embed", self.host))
            .json(&request)
            .send()
            .map_err(|e| {
                OllamaBatchError::Other(
                    anyhow::anyhow!(e).context("sending embedding request to Ollama"),
                )
            })?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().unwrap_or_default();
            if status.as_u16() == 400 && body.contains("context length") {
                return Err(OllamaBatchError::ContextLength);
            }
            return Err(OllamaBatchError::Other(anyhow::anyhow!(
                "Ollama API error ({}): {}",
                status,
                body
            )));
        }

        let resp: OllamaEmbeddingResponse = response.json().map_err(|e| {
            OllamaBatchError::Other(anyhow::anyhow!(e).context("parsing Ollama embedding response"))
        })?;

        Ok(resp.embeddings)
    }

    /// Embed a slice of chunks, halving the sub-batch size on context-length errors.
    /// Returns embeddings in order.
    fn embed_with_backoff(&self, chunks: &[String], batch_size: usize) -> Result<Vec<Vec<f32>>> {
        match self.embed_batch(chunks) {
            Ok(embeddings) => Ok(embeddings),
            Err(OllamaBatchError::ContextLength) => {
                if chunks.len() == 1 {
                    anyhow::bail!(
                        "Single chunk exceeds Ollama context length ({} chars). \
                         Reduce chunk size or use a model with a larger context window.",
                        chunks[0].len()
                    );
                }
                let smaller = batch_size / 2;
                warn!(
                    "Batch of {} chunks exceeded context length, retrying with batch size {}",
                    chunks.len(),
                    smaller
                );
                let mut results = Vec::with_capacity(chunks.len());
                for sub_batch in chunks.chunks(smaller.max(1)) {
                    results.extend(self.embed_with_backoff(sub_batch, smaller)?);
                }
                Ok(results)
            }
            Err(OllamaBatchError::Other(e)) => Err(e),
        }
    }
}

enum OllamaBatchError {
    ContextLength,
    Other(anyhow::Error),
}

impl EmbeddingProvider for OllamaEmbedding {
    fn compute_embeddings(&self, chunks: &[String]) -> Result<Array2<f32>> {
        if chunks.is_empty() {
            return Ok(Array2::zeros((0, self.dimensions)));
        }

        // Start with a large batch size to keep GPU saturated; automatically
        // halve on context-length errors so we converge to the largest safe size.
        let batch_size: usize = 128;
        let mut all_embeddings: Vec<Vec<f32>> = Vec::with_capacity(chunks.len());
        let num_batches = (chunks.len() + batch_size - 1) / batch_size;

        for (i, batch) in chunks.chunks(batch_size).enumerate() {
            info!(
                "Ollama embedding batch {}/{} ({} chunks)",
                i + 1,
                num_batches,
                batch.len()
            );

            let embeddings = self.embed_with_backoff(batch, batch_size)?;
            all_embeddings.extend(embeddings);
        }

        if all_embeddings.is_empty() {
            return Ok(Array2::zeros((0, self.dimensions)));
        }

        let n = all_embeddings.len();
        let d = all_embeddings[0].len();
        let flat: Vec<f32> = all_embeddings.into_iter().flatten().collect();

        Array2::from_shape_vec((n, d), flat).context("reshaping Ollama embeddings")
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn name(&self) -> &str {
        "ollama"
    }
}
