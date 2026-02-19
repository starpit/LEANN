use anyhow::Result;
use ndarray::Array2;

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
        let api_key = settings::resolve_gemini_api_key(api_key)
            .ok_or_else(|| anyhow::anyhow!("Gemini API key required (set GOOGLE_API_KEY or GEMINI_API_KEY)"))?;

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

        // Gemini batch embedding API
        let requests: Vec<serde_json::Value> = chunks
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

        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:batchEmbedContents?key={}",
            self.model, self.api_key
        );

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

        let first_values = embeddings_array[0]["values"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("Missing 'values' in embedding"))?;
        let dim = first_values.len();

        let mut data = Vec::with_capacity(chunks.len() * dim);
        for emb in embeddings_array {
            let values = emb["values"]
                .as_array()
                .ok_or_else(|| anyhow::anyhow!("Missing 'values' in embedding"))?;
            for v in values {
                data.push(v.as_f64().unwrap_or(0.0) as f32);
            }
        }

        Ok(Array2::from_shape_vec((chunks.len(), dim), data)?)
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn name(&self) -> &str {
        "gemini"
    }
}
