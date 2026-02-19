use anyhow::{Context, Result};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{info, warn};

use super::EmbeddingProvider;
use crate::settings::resolve_ollama_host;

/// Maximum number of concurrent in-flight requests to Ollama.
/// Keeps the GPU pipeline fed without overwhelming the server.
const MAX_CONCURRENT: usize = 3;

/// Shared state cloned into each spawned batch task.
#[derive(Clone)]
struct OllamaHandle {
    model: String,
    host: String,
    client: reqwest::Client, // internally Arc'd, cheap to clone
}

/// Ollama embedding API client with pipelined batch dispatch.
pub struct OllamaEmbedding {
    handle: OllamaHandle,
    dimensions: usize,
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

impl OllamaHandle {
    /// Send a single batch to the Ollama /api/embed endpoint.
    async fn embed_batch(&self, batch: &[String]) -> Result<Vec<Vec<f32>>, OllamaBatchError> {
        let request = OllamaEmbeddingRequest {
            model: self.model.clone(),
            input: batch.to_vec(),
        };

        let response = self
            .client
            .post(format!("{}/api/embed", self.host))
            .json(&request)
            .send()
            .await
            .map_err(|e| {
                OllamaBatchError::Other(
                    anyhow::anyhow!(e).context("sending embedding request to Ollama"),
                )
            })?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            if status.as_u16() == 400 && body.contains("context length") {
                return Err(OllamaBatchError::ContextLength);
            }
            return Err(OllamaBatchError::Other(anyhow::anyhow!(
                "Ollama API error ({}): {}",
                status,
                body
            )));
        }

        let resp: OllamaEmbeddingResponse = response.json().await.map_err(|e| {
            OllamaBatchError::Other(anyhow::anyhow!(e).context("parsing Ollama embedding response"))
        })?;

        Ok(resp.embeddings)
    }

    /// Embed a slice of chunks, halving the sub-batch size on context-length errors.
    fn embed_with_backoff<'a>(
        &'a self,
        chunks: &'a [String],
        batch_size: usize,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<Vec<f32>>>> + Send + 'a>>
    {
        Box::pin(async move {
            match self.embed_batch(chunks).await {
                Ok(embeddings) => Ok(embeddings),
                Err(OllamaBatchError::ContextLength) => {
                    if chunks.len() == 1 {
                        return self.truncate_single_chunk(&chunks[0]).await;
                    }
                    let smaller = batch_size / 2;
                    warn!(
                        "Batch of {} chunks exceeded context length, retrying with batch size {}",
                        chunks.len(),
                        smaller
                    );
                    let mut results = Vec::with_capacity(chunks.len());
                    for sub_batch in chunks.chunks(smaller.max(1)) {
                        results.extend(self.embed_with_backoff(sub_batch, smaller).await?);
                    }
                    Ok(results)
                }
                Err(OllamaBatchError::Other(e)) => Err(e),
            }
        })
    }

    /// Binary-search for the largest prefix of an oversized chunk that fits.
    async fn truncate_single_chunk(&self, chunk: &str) -> Result<Vec<Vec<f32>>> {
        let original_len = chunk.len();
        let mut lo = 0usize;
        let mut hi = original_len;
        let mut last_good = None;

        while lo < hi {
            let mid = (lo + hi) / 2;
            let truncated = vec![chunk[..mid].to_string()];
            match self.embed_batch(&truncated).await {
                Ok(emb) => {
                    last_good = Some(emb);
                    lo = mid + 1;
                }
                Err(OllamaBatchError::ContextLength) => {
                    hi = mid;
                }
                Err(OllamaBatchError::Other(e)) => return Err(e),
            }
        }

        if let Some(embeddings) = last_good {
            warn!(
                "Truncated oversized chunk from {} to ~{} chars to fit Ollama context",
                original_len,
                lo.saturating_sub(1)
            );
            return Ok(embeddings);
        }

        anyhow::bail!(
            "Single chunk exceeds Ollama context length ({} chars) \
             and could not be truncated to fit.",
            original_len
        );
    }
}

impl OllamaEmbedding {
    pub fn new(model: &str, host: Option<&str>) -> Self {
        Self {
            handle: OllamaHandle {
                model: model.to_string(),
                host: resolve_ollama_host(host),
                client: reqwest::Client::new(),
            },
            dimensions: 768, // Default, will be updated on first compute
        }
    }

    /// Dispatch all batches concurrently (up to MAX_CONCURRENT in-flight) so the
    /// GPU never idles waiting for the next request.
    async fn compute_embeddings_async(&self, chunks: &[String]) -> Result<Array2<f32>> {
        if chunks.is_empty() {
            return Ok(Array2::zeros((0, self.dimensions)));
        }

        let batch_size: usize = 128;
        let num_batches = (chunks.len() + batch_size - 1) / batch_size;

        info!(
            "Ollama embedding: {} chunks in {} batches (concurrency={})",
            chunks.len(),
            num_batches,
            MAX_CONCURRENT
        );

        let semaphore = Arc::new(tokio::sync::Semaphore::new(MAX_CONCURRENT));
        let mut handles = Vec::with_capacity(num_batches);

        for (i, batch) in chunks.chunks(batch_size).enumerate() {
            let sem = semaphore.clone();
            let handle = self.handle.clone();
            let batch_owned: Vec<String> = batch.to_vec();
            let batch_idx = i;
            let total = num_batches;

            handles.push(tokio::spawn(async move {
                let _permit = sem.acquire().await.expect("semaphore closed");
                info!(
                    "Ollama embedding batch {}/{} ({} chunks)",
                    batch_idx + 1,
                    total,
                    batch_owned.len()
                );
                let result = handle.embed_with_backoff(&batch_owned, batch_size).await;
                (batch_idx, result)
            }));
        }

        // Collect results in order.
        let mut indexed_results: Vec<(usize, Vec<Vec<f32>>)> = Vec::with_capacity(num_batches);
        for h in handles {
            let (idx, result) = h.await.context("embedding task panicked")?;
            indexed_results.push((idx, result?));
        }
        indexed_results.sort_by_key(|(idx, _)| *idx);

        let all_embeddings: Vec<Vec<f32>> = indexed_results
            .into_iter()
            .flat_map(|(_, embs)| embs)
            .collect();

        if all_embeddings.is_empty() {
            return Ok(Array2::zeros((0, self.dimensions)));
        }

        let n = all_embeddings.len();
        let d = all_embeddings[0].len();
        let flat: Vec<f32> = all_embeddings.into_iter().flatten().collect();

        Array2::from_shape_vec((n, d), flat).context("reshaping Ollama embeddings")
    }
}

enum OllamaBatchError {
    ContextLength,
    Other(anyhow::Error),
}

impl EmbeddingProvider for OllamaEmbedding {
    fn compute_embeddings(&self, chunks: &[String]) -> Result<Array2<f32>> {
        // Enter async context - reuse existing runtime if available,
        // otherwise create one.
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => {
                // We're inside a tokio runtime already but in a sync call.
                // Spawn a thread to avoid blocking the runtime with block_on.
                std::thread::scope(|s| {
                    s.spawn(|| handle.block_on(self.compute_embeddings_async(chunks)))
                        .join()
                        .expect("embedding thread panicked")
                })
            }
            Err(_) => {
                let rt = tokio::runtime::Runtime::new()?;
                rt.block_on(self.compute_embeddings_async(chunks))
            }
        }
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn name(&self) -> &str {
        "ollama"
    }
}
