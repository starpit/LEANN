use anyhow::Result;
use ndarray::Array2;
use tracing::info;

use super::EmbeddingProvider;
use crate::settings;

/// Gemini API embedding provider.
pub struct GeminiEmbedding {
    model: String,
    api_key: String,
    client: reqwest::blocking::Client,
    dimensions: usize,
}

impl GeminiEmbedding {
    pub fn new(model: &str, api_key: Option<&str>) -> Result<Self> {
        let api_key = settings::resolve_gemini_api_key(api_key).ok_or_else(|| {
            anyhow::anyhow!("Gemini API key required (set GOOGLE_API_KEY or GEMINI_API_KEY)")
        })?;

        Ok(Self {
            model: model.to_string(),
            api_key,
            client: reqwest::blocking::Client::new(),
            dimensions: 768, // Default, will be updated on first call
        })
    }
}

impl EmbeddingProvider for GeminiEmbedding {
    fn compute_embeddings(&self, chunks: &[String]) -> Result<Array2<f32>> {
        if chunks.is_empty() {
            return Ok(Array2::zeros((0, self.dimensions)));
        }

        // Gemini limits to 100 requests per batch call (matches Python's max_batch_size=100)
        let max_batch_size = 100;
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:batchEmbedContents?key={}",
            self.model, self.api_key
        );

        let mut all_data: Vec<f32> = Vec::new();
        let mut dim: Option<usize> = None;
        let num_batches = chunks.len().div_ceil(max_batch_size);

        for (i, batch) in chunks.chunks(max_batch_size).enumerate() {
            info!(
                "Gemini embedding batch {}/{} ({} chunks)",
                i + 1,
                num_batches,
                batch.len()
            );
            let requests: Vec<serde_json::Value> = batch
                .iter()
                .map(|text| {
                    serde_json::json!({
                        "model": format!("models/{}", self.model),
                        "content": {
                            "parts": [{"text": text}]
                        }
                    })
                })
                .collect();

            let payload = serde_json::json!({
                "requests": requests,
            });

            let response = self.client.post(&url).json(&payload).send()?;

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().unwrap_or_default();
                anyhow::bail!("Gemini API error ({}): {}", status, body);
            }

            let body: serde_json::Value = response.json()?;

            let embeddings_array = body["embeddings"]
                .as_array()
                .ok_or_else(|| anyhow::anyhow!("Missing 'embeddings' in Gemini response"))?;

            if embeddings_array.is_empty() {
                anyhow::bail!("Empty embeddings response from Gemini");
            }

            if dim.is_none() {
                let first_values = embeddings_array[0]["values"]
                    .as_array()
                    .ok_or_else(|| anyhow::anyhow!("Missing 'values' in embedding"))?;
                dim = Some(first_values.len());
            }

            for emb in embeddings_array {
                let values = emb["values"]
                    .as_array()
                    .ok_or_else(|| anyhow::anyhow!("Missing 'values' in embedding"))?;
                for v in values {
                    all_data.push(v.as_f64().unwrap_or(0.0) as f32);
                }
            }
        }

        let d = dim.ok_or_else(|| anyhow::anyhow!("No embeddings returned from Gemini"))?;
        Ok(Array2::from_shape_vec((chunks.len(), d), all_data)?)
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn name(&self) -> &str {
        "gemini"
    }
}
