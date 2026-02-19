use anyhow::{Context, Result};
use ndarray::Array2;
use serde::{Deserialize, Serialize};

use super::EmbeddingProvider;
use crate::settings::resolve_openai_api_key;

/// OpenAI embedding API client.
pub struct OpenAiEmbedding {
    model: String,
    api_key: String,
    base_url: String,
    dimensions: usize,
    client: reqwest::blocking::Client,
}

#[derive(Serialize)]
struct EmbeddingRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

impl OpenAiEmbedding {
    pub fn new(
        model: &str,
        api_key: Option<&str>,
        base_url: Option<&str>,
        dimensions: Option<usize>,
    ) -> Result<Self> {
        let api_key = resolve_openai_api_key(api_key)
            .ok_or_else(|| anyhow::anyhow!("OpenAI API key required (set OPENAI_API_KEY)"))?;

        let base_url = base_url
            .unwrap_or("https://api.openai.com/v1")
            .trim_end_matches('/')
            .to_string();

        let dimensions = dimensions.unwrap_or(1536);

        Ok(Self {
            model: model.to_string(),
            api_key,
            base_url,
            dimensions,
            client: reqwest::blocking::Client::new(),
        })
    }
}

impl EmbeddingProvider for OpenAiEmbedding {
    fn compute_embeddings(&self, chunks: &[String]) -> Result<Array2<f32>> {
        if chunks.is_empty() {
            return Ok(Array2::zeros((0, self.dimensions)));
        }

        let request = EmbeddingRequest {
            model: self.model.clone(),
            input: chunks.to_vec(),
        };

        let response = self
            .client
            .post(format!("{}/embeddings", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .context("sending embedding request to OpenAI")?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().unwrap_or_default();
            anyhow::bail!("OpenAI API error ({}): {}", status, body);
        }

        let resp: EmbeddingResponse = response
            .json()
            .context("parsing OpenAI embedding response")?;

        let n = resp.data.len();
        if n == 0 {
            return Ok(Array2::zeros((0, self.dimensions)));
        }
        let d = resp.data[0].embedding.len();
        let flat: Vec<f32> = resp.data.into_iter().flat_map(|e| e.embedding).collect();

        Array2::from_shape_vec((n, d), flat).context("reshaping OpenAI embeddings")
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn name(&self) -> &str {
        "openai"
    }
}
