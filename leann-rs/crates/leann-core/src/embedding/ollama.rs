use anyhow::{Context, Result};
use ndarray::Array2;
use serde::{Deserialize, Serialize};

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
}

impl EmbeddingProvider for OllamaEmbedding {
    fn compute_embeddings(&self, chunks: &[String]) -> Result<Array2<f32>> {
        if chunks.is_empty() {
            return Ok(Array2::zeros((0, self.dimensions)));
        }

        let request = OllamaEmbeddingRequest {
            model: self.model.clone(),
            input: chunks.to_vec(),
        };

        let response = self
            .client
            .post(format!("{}/api/embed", self.host))
            .json(&request)
            .send()
            .context("sending embedding request to Ollama")?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().unwrap_or_default();
            anyhow::bail!("Ollama API error ({}): {}", status, body);
        }

        let resp: OllamaEmbeddingResponse = response
            .json()
            .context("parsing Ollama embedding response")?;

        let n = resp.embeddings.len();
        if n == 0 {
            return Ok(Array2::zeros((0, self.dimensions)));
        }
        let d = resp.embeddings[0].len();
        let flat: Vec<f32> = resp.embeddings.into_iter().flatten().collect();

        Array2::from_shape_vec((n, d), flat).context("reshaping Ollama embeddings")
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn name(&self) -> &str {
        "ollama"
    }
}
