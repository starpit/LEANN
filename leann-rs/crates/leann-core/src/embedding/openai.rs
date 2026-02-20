use anyhow::{Context, Result};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use tracing::info;

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

        // Batch to avoid overwhelming the API (matches Python's max_batch_size=800)
        let max_batch_size = if self.base_url.contains("generativelanguage.googleapis.com") {
            100 // Gemini OpenAI-compatible endpoint limits to 100
        } else {
            800
        };

        let mut all_embeddings: Vec<Vec<f32>> = Vec::with_capacity(chunks.len());
        let num_batches = chunks.len().div_ceil(max_batch_size);

        for (i, batch) in chunks.chunks(max_batch_size).enumerate() {
            info!(
                "OpenAI embedding batch {}/{} ({} chunks)",
                i + 1,
                num_batches,
                batch.len()
            );
            let request = EmbeddingRequest {
                model: self.model.clone(),
                input: batch.to_vec(),
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

            for item in resp.data {
                all_embeddings.push(item.embedding);
            }
        }

        if all_embeddings.is_empty() {
            return Ok(Array2::zeros((0, self.dimensions)));
        }

        let n = all_embeddings.len();
        let d = all_embeddings[0].len();
        let flat: Vec<f32> = all_embeddings.into_iter().flatten().collect();

        Array2::from_shape_vec((n, d), flat).context("reshaping OpenAI embeddings")
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn name(&self) -> &str {
        "openai"
    }
}
