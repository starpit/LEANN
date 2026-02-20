use anyhow::Result;
use std::sync::Arc;
use tracing::{error, info};

use super::EmbeddingProvider;
use crate::index::DistanceMetric;
use crate::passages::PassageManager;

/// ZMQ-based embedding server that handles three request types:
/// 1. Text embedding: list of strings -> embeddings
/// 2. Distance calculation: [[node_ids], [query_vector]] -> distances
/// 3. Embedding by ID: [[node_ids]] -> embeddings
pub struct EmbeddingServer {
    port: u16,
    provider: Arc<dyn EmbeddingProvider>,
    passages: Arc<PassageManager>,
    distance_metric: DistanceMetric,
    id_map: Vec<String>,
    dimensions: usize,
}

impl EmbeddingServer {
    pub fn new(
        port: u16,
        provider: Arc<dyn EmbeddingProvider>,
        passages: Arc<PassageManager>,
        distance_metric: DistanceMetric,
        id_map: Vec<String>,
        dimensions: usize,
    ) -> Self {
        Self {
            port,
            provider,
            passages,
            distance_metric,
            id_map,
            dimensions,
        }
    }

    /// Run the ZMQ REP server. This blocks until shutdown is signaled.
    pub async fn run(&self, shutdown: tokio::sync::watch::Receiver<bool>) -> Result<()> {
        use zeromq::{Socket, SocketRecv, SocketSend, ZmqMessage};

        let mut socket = zeromq::RepSocket::new();
        socket
            .bind(&format!("tcp://*:{}", self.port))
            .await
            .map_err(|e| anyhow::anyhow!("binding ZMQ socket: {}", e))?;

        info!("HNSW ZMQ REP server listening on port {}", self.port);

        loop {
            if *shutdown.borrow() {
                info!("Shutdown signal received, stopping server");
                break;
            }

            // Use a timeout-based approach for checking shutdown
            let recv_result =
                tokio::time::timeout(std::time::Duration::from_secs(1), socket.recv()).await;

            let msg = match recv_result {
                Ok(Ok(msg)) => msg,
                Ok(Err(e)) => {
                    error!("Error receiving ZMQ message: {}", e);
                    continue;
                }
                Err(_) => {
                    // Timeout - check shutdown and continue
                    continue;
                }
            };

            let request_bytes = msg.get(0).map(|f| f.as_ref().to_vec()).unwrap_or_default();

            let response = match self.handle_request(&request_bytes) {
                Ok(resp) => resp,
                Err(e) => {
                    error!("Error handling request: {}", e);
                    // Send empty response to maintain REQ/REP pattern
                    rmp_serde::to_vec(&Vec::<Vec<f32>>::new()).unwrap_or_default()
                }
            };

            let resp_msg = ZmqMessage::from(response);
            if let Err(e) = socket.send(resp_msg).await {
                error!("Error sending ZMQ response: {}", e);
            }
        }

        Ok(())
    }

    fn handle_request(&self, request_bytes: &[u8]) -> Result<Vec<u8>> {
        // Try as list of strings (text embedding)
        if let Ok(texts) = rmp_serde::from_slice::<Vec<String>>(request_bytes)
            && !texts.is_empty()
            && texts.iter().all(|t| !t.is_empty())
        {
            return self.handle_text_embedding(&texts);
        }

        // Try as [[ids], [query_vec]] (distance calculation)
        if let Ok(parts) = rmp_serde::from_slice::<Vec<Vec<serde_json::Value>>>(request_bytes)
            && parts.len() == 2
        {
            return self.handle_distance_request(&parts);
        }

        // Fall back to embedding by ID
        if let Ok(ids) = rmp_serde::from_slice::<Vec<Vec<i64>>>(request_bytes) {
            let flat_ids: Vec<usize> = ids.into_iter().flatten().map(|id| id as usize).collect();
            return self.handle_embedding_by_id(&flat_ids);
        }

        anyhow::bail!("Unknown request format")
    }

    fn handle_text_embedding(&self, texts: &[String]) -> Result<Vec<u8>> {
        let texts_owned: Vec<String> = texts.to_vec();
        let embeddings = self.provider.compute_embeddings(&texts_owned)?;
        let result: Vec<Vec<f32>> = embeddings
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect();
        Ok(rmp_serde::to_vec(&result)?)
    }

    fn handle_distance_request(&self, parts: &[Vec<serde_json::Value>]) -> Result<Vec<u8>> {
        let node_ids: Vec<usize> = parts[0]
            .iter()
            .filter_map(|v| v.as_u64().map(|n| n as usize))
            .collect();
        let query_vector: Vec<f32> = parts[1]
            .iter()
            .filter_map(|v| v.as_f64().map(|f| f as f32))
            .collect();

        let large_distance: f32 = 1e9;
        let mut distances = vec![large_distance; node_ids.len()];

        // Look up texts for each node ID
        let mut texts = Vec::new();
        let mut found_indices = Vec::new();

        for (idx, &nid) in node_ids.iter().enumerate() {
            let passage_id = self.map_node_id(nid);
            if let Ok(passage) = self.passages.get_passage(&passage_id)
                && !passage.text.is_empty()
            {
                texts.push(passage.text);
                found_indices.push(idx);
            }
        }

        if !texts.is_empty()
            && let Ok(embeddings) = self.provider.compute_embeddings(&texts)
        {
            for (i, &original_idx) in found_indices.iter().enumerate() {
                let emb = embeddings.row(i);
                let dist = match self.distance_metric {
                    DistanceMetric::L2 => emb
                        .iter()
                        .zip(query_vector.iter())
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum(),
                    _ => -emb
                        .iter()
                        .zip(query_vector.iter())
                        .map(|(a, b)| a * b)
                        .sum::<f32>(),
                };
                distances[original_idx] = dist;
            }
        }

        Ok(rmp_serde::to_vec(&vec![distances])?)
    }

    fn handle_embedding_by_id(&self, node_ids: &[usize]) -> Result<Vec<u8>> {
        let n = node_ids.len();
        let d = self.dimensions;

        let mut texts = Vec::new();
        let mut found_indices = Vec::new();

        for (idx, &nid) in node_ids.iter().enumerate() {
            let passage_id = self.map_node_id(nid);
            if let Ok(passage) = self.passages.get_passage(&passage_id)
                && !passage.text.is_empty()
            {
                texts.push(passage.text);
                found_indices.push(idx);
            }
        }

        let mut flat_data = vec![0.0f32; n * d];

        if !texts.is_empty()
            && let Ok(embeddings) = self.provider.compute_embeddings(&texts)
        {
            for (j, &pos) in found_indices.iter().enumerate() {
                let emb = embeddings.row(j);
                let start = pos * d;
                for (k, &val) in emb.iter().enumerate() {
                    if start + k < flat_data.len() {
                        flat_data[start + k] = val;
                    }
                }
            }
        }

        let response: Vec<Vec<f32>> = vec![vec![n as f32, d as f32], flat_data];
        Ok(rmp_serde::to_vec(&response)?)
    }

    fn map_node_id(&self, nid: usize) -> String {
        if !self.id_map.is_empty() && nid < self.id_map.len() {
            self.id_map[nid].clone()
        } else {
            nid.to_string()
        }
    }
}
