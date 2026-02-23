use anyhow::{Context, Result};
use ndarray::Array2;

use super::EmbeddingProvider;

/// ZMQ client for communicating with the embedding server.
///
/// Supports three request types matching the Python msgpack protocol:
/// 1. Text embedding: list of strings -> list of embedding vectors
/// 2. Distance calculation: [[node_ids], [query_vector]] -> [[distances]]
/// 3. Embedding by ID: [[node_ids]] -> [[dims], [flat_data]]
pub struct EmbeddingClient {
    port: u16,
}

impl EmbeddingClient {
    pub fn new(port: u16) -> Self {
        Self { port }
    }

    /// Compute embeddings for text chunks via the ZMQ server.
    /// Chunks are sent in batches to avoid overwhelming the server.
    pub fn compute_text_embeddings(&self, chunks: &[String]) -> Result<Array2<f32>> {
        let rt = tokio::runtime::Handle::try_current()
            .unwrap_or_else(|_| tokio::runtime::Runtime::new().unwrap().handle().clone());

        rt.block_on(async { self.compute_text_embeddings_async(chunks).await })
    }

    async fn compute_text_embeddings_async(&self, chunks: &[String]) -> Result<Array2<f32>> {
        use zeromq::{Socket, SocketRecv, SocketSend, ZmqMessage};

        if chunks.is_empty() {
            anyhow::bail!("Empty input to embedding server");
        }

        // Batch to avoid overwhelming the embedding server (matches Python's batch_size=32)
        let batch_size = 128;
        let mut all_embeddings: Vec<Vec<f32>> = Vec::with_capacity(chunks.len());

        for batch in chunks.chunks(batch_size) {
            let mut socket = zeromq::ReqSocket::new();
            socket
                .connect(&format!("tcp://localhost:{}", self.port))
                .await
                .context("connecting to embedding server")?;

            // Pack request as msgpack: list of strings
            let request = rmp_serde::to_vec(batch)?;
            let msg = ZmqMessage::from(request);
            socket
                .send(msg)
                .await
                .context("sending embedding request")?;

            // Receive response
            let response = socket
                .recv()
                .await
                .context("receiving embedding response")?;
            let response_bytes = response.get(0).map(|f| f.as_ref()).unwrap_or(&[]);

            // Decode msgpack response: list of lists of floats
            let embeddings: Vec<Vec<f32>> =
                rmp_serde::from_slice(response_bytes).context("decoding embedding response")?;

            if embeddings.is_empty() {
                anyhow::bail!("Empty response from embedding server");
            }

            all_embeddings.extend(embeddings);
        }

        let n = all_embeddings.len();
        let d = all_embeddings[0].len();
        let flat: Vec<f32> = all_embeddings.into_iter().flatten().collect();

        Array2::from_shape_vec((n, d), flat).context("reshaping embeddings")
    }

    /// Compute distances between node embeddings and a query vector.
    /// Returns distances for each node ID.
    pub fn compute_distances(&self, node_ids: &[usize], query: &[f32]) -> Result<Vec<f32>> {
        let rt = tokio::runtime::Handle::try_current()
            .unwrap_or_else(|_| tokio::runtime::Runtime::new().unwrap().handle().clone());

        rt.block_on(async { self.compute_distances_async(node_ids, query).await })
    }

    async fn compute_distances_async(&self, node_ids: &[usize], query: &[f32]) -> Result<Vec<f32>> {
        use zeromq::{Socket, SocketRecv, SocketSend, ZmqMessage};

        let mut socket = zeromq::ReqSocket::new();
        socket
            .connect(&format!("tcp://localhost:{}", self.port))
            .await
            .context("connecting to embedding server")?;

        // Pack request as msgpack: [[node_ids], [query_vector]]
        let request: Vec<serde_json::Value> =
            vec![serde_json::json!(node_ids), serde_json::json!(query)];
        let request_bytes = rmp_serde::to_vec(&request)?;
        let msg = ZmqMessage::from(request_bytes);
        socket.send(msg).await?;

        let response = socket.recv().await?;
        let response_bytes = response.get(0).map(|f| f.as_ref()).unwrap_or(&[]);

        // Response: [[distances]]
        let result: Vec<Vec<f32>> = rmp_serde::from_slice(response_bytes)?;
        Ok(result.into_iter().next().unwrap_or_default())
    }

    /// Fetch embeddings by node ID.
    pub fn get_embeddings_by_id(&self, node_ids: &[usize]) -> Result<Array2<f32>> {
        let rt = tokio::runtime::Handle::try_current()
            .unwrap_or_else(|_| tokio::runtime::Runtime::new().unwrap().handle().clone());

        rt.block_on(async { self.get_embeddings_by_id_async(node_ids).await })
    }

    async fn get_embeddings_by_id_async(&self, node_ids: &[usize]) -> Result<Array2<f32>> {
        use zeromq::{Socket, SocketRecv, SocketSend, ZmqMessage};

        let mut socket = zeromq::ReqSocket::new();
        socket
            .connect(&format!("tcp://localhost:{}", self.port))
            .await?;

        // Pack request: [[node_ids]]
        let request = vec![node_ids.to_vec()];
        let request_bytes = rmp_serde::to_vec(&request)?;
        let msg = ZmqMessage::from(request_bytes);
        socket.send(msg).await?;

        let response = socket.recv().await?;
        let response_bytes = response.get(0).map(|f| f.as_ref()).unwrap_or(&[]);

        // Response: [[n, d], [flat_data]]
        let result: Vec<Vec<f32>> = rmp_serde::from_slice(response_bytes)?;
        if result.len() < 2 {
            anyhow::bail!("Invalid embedding-by-id response");
        }

        let dims = &result[0];
        if dims.len() < 2 {
            anyhow::bail!("Invalid dimensions in response");
        }
        let n = dims[0] as usize;
        let d = dims[1] as usize;
        let flat_data = &result[1];

        if flat_data.len() != n * d {
            anyhow::bail!(
                "Data length mismatch: expected {}, got {}",
                n * d,
                flat_data.len()
            );
        }

        Array2::from_shape_vec((n, d), flat_data.clone()).context("reshaping embeddings")
    }
}

impl EmbeddingProvider for EmbeddingClient {
    fn compute_embeddings(&self, chunks: &[String]) -> Result<Array2<f32>> {
        self.compute_text_embeddings(chunks)
    }

    fn dimensions(&self) -> usize {
        // Dimensions are detected by the builder via a probe call.
        0
    }

    fn name(&self) -> &str {
        "zmq"
    }
}
