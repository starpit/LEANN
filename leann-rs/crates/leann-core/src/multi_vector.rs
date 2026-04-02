//! Multi-vector (ColBERT) index support.
//!
//! Builds on the existing HNSW backend by flattening all token vectors from all
//! documents into a single index (one HNSW node per token), then aggregating
//! results per-document at query time using the ColBERT MaxSim formula:
//!
//! ```text
//! score(Q, D) = Σ_i max_j (q_i · d_j)
//! ```

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use ndarray::{Array2, ArrayView1};
use serde::{Deserialize, Serialize};

use crate::backend::{self, BackendConfig, BackendIndex};
use crate::hnsw::search::SearchParams;
use crate::index::DistanceMetric;

// ---------------------------------------------------------------------------
// Token label — one per HNSW node
// ---------------------------------------------------------------------------

/// Metadata for a single token vector in the flattened index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenLabel {
    /// Document this token belongs to.
    pub doc_id: u32,
    /// Token position within the document.
    pub seq_id: u32,
    /// Arbitrary per-document metadata (filepath, image_path, etc.).
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Pending document: token embeddings + metadata, buffered before index build.
struct PendingDoc {
    doc_id: u32,
    embeddings: Array2<f32>,
    metadata: HashMap<String, serde_json::Value>,
}

/// Builds a multi-vector index from per-document token embeddings.
pub struct MultiVectorBuilder {
    dim: usize,
    pending: Vec<PendingDoc>,
    backend_config: BackendConfig,
}

impl MultiVectorBuilder {
    /// Create a builder for the given embedding dimension.
    pub fn new(dim: usize) -> Self {
        let mut config = BackendConfig::hnsw_default();
        // Multi-vector indexes use MIPS (inner product) for ColBERT scoring.
        config.set_distance_metric(DistanceMetric::Mips);
        // Store vectors so we can do exact MaxSim reranking.
        config.set_recompute(false);
        config.set_compact(false);
        Self {
            dim,
            pending: Vec::new(),
            backend_config: config,
        }
    }

    /// Set the HNSW M parameter.
    pub fn set_m(&mut self, m: usize) -> &mut Self {
        self.backend_config.set_m(m);
        self
    }

    /// Set the HNSW efConstruction parameter.
    pub fn set_ef_construction(&mut self, ef: usize) -> &mut Self {
        self.backend_config.set_ef_construction(ef);
        self
    }

    /// Insert a document's token embeddings.
    ///
    /// `embeddings` has shape `[num_tokens, dim]`.
    pub fn insert(
        &mut self,
        doc_id: u32,
        embeddings: Array2<f32>,
        metadata: HashMap<String, serde_json::Value>,
    ) -> &mut Self {
        assert_eq!(
            embeddings.ncols(),
            self.dim,
            "embedding dim {} != expected {}",
            embeddings.ncols(),
            self.dim
        );
        self.pending.push(PendingDoc {
            doc_id,
            embeddings,
            metadata,
        });
        self
    }

    /// Build the index and write it to `index_path`.
    ///
    /// Produces:
    /// - `<index_path>.index` — HNSW binary index
    /// - `<index_path>.labels.json` — per-node token labels
    /// - `<index_path>.emb.npy` — raw embedding matrix for exact reranking
    pub fn build(&self, index_path: &Path) -> Result<()> {
        anyhow::ensure!(!self.pending.is_empty(), "no documents inserted");

        // Flatten all token vectors + build labels.
        let total_tokens: usize = self.pending.iter().map(|d| d.embeddings.nrows()).sum();
        let mut flat = Array2::<f32>::zeros((total_tokens, self.dim));
        let mut labels = Vec::with_capacity(total_tokens);

        let mut row = 0;
        for doc in &self.pending {
            for seq_id in 0..doc.embeddings.nrows() {
                flat.row_mut(row).assign(&doc.embeddings.row(seq_id));
                labels.push(TokenLabel {
                    doc_id: doc.doc_id,
                    seq_id: seq_id as u32,
                    metadata: doc.metadata.clone(),
                });
                row += 1;
            }
        }

        // Build HNSW index.
        let index_file = with_ext(index_path, "index");
        backend::build_backend(&self.backend_config, &flat, &index_file)?;

        // Write labels sidecar.
        let labels_file = with_ext(index_path, "labels.json");
        let labels_json = serde_json::to_string(&labels)?;
        fs::write(&labels_file, labels_json)
            .with_context(|| format!("writing {}", labels_file.display()))?;

        // Write .emb.npy for exact reranking.
        let npy_file = with_ext(index_path, "emb.npy");
        write_npy(&flat, &npy_file)?;

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Searcher
// ---------------------------------------------------------------------------

/// A loaded multi-vector index, ready for MaxSim search.
pub struct MultiVectorSearcher {
    index: BackendIndex,
    labels: Vec<TokenLabel>,
    /// doc_id → list of flat row indices into the embedding matrix.
    doc_to_rows: HashMap<u32, Vec<usize>>,
    /// Memory-mapped embedding matrix for exact reranking.
    #[cfg(feature = "multi-vector")]
    emb_mmap: memmap2::Mmap,
    #[cfg(not(feature = "multi-vector"))]
    emb_data: Vec<u8>,
    dim: usize,
    total_tokens: usize,
}

impl MultiVectorSearcher {
    /// Open a multi-vector index from disk.
    pub fn open(index_path: &Path) -> Result<Self> {
        // Read HNSW index.
        let index_file = with_ext(index_path, "index");
        let index = backend::read_backend_index("hnsw", &index_file)?;

        // Read labels.
        let labels_file = with_ext(index_path, "labels.json");
        let labels_data = fs::read_to_string(&labels_file)
            .with_context(|| format!("reading {}", labels_file.display()))?;
        let labels: Vec<TokenLabel> = serde_json::from_str(&labels_data)?;

        // Build doc_id → rows mapping.
        let mut doc_to_rows: HashMap<u32, Vec<usize>> = HashMap::new();
        for (i, label) in labels.iter().enumerate() {
            doc_to_rows.entry(label.doc_id).or_default().push(i);
        }

        let dim = index.dimensions();
        let total_tokens = labels.len();

        // Mmap the .emb.npy file.
        let npy_file = with_ext(index_path, "emb.npy");

        #[cfg(feature = "multi-vector")]
        let emb_mmap = {
            let file = fs::File::open(&npy_file)
                .with_context(|| format!("opening {}", npy_file.display()))?;
            unsafe { memmap2::Mmap::map(&file)? }
        };

        Ok(Self {
            index,
            labels,
            doc_to_rows,
            #[cfg(feature = "multi-vector")]
            emb_mmap,
            #[cfg(not(feature = "multi-vector"))]
            emb_data: fs::read(&npy_file)?,
            dim,
            total_tokens,
        })
    }

    /// Number of documents in the index.
    pub fn num_docs(&self) -> usize {
        self.doc_to_rows.len()
    }

    /// Total number of token vectors in the index.
    pub fn num_tokens(&self) -> usize {
        self.total_tokens
    }

    /// Approximate MaxSim search.
    ///
    /// For each query token, runs HNSW ANN search, then aggregates per-document
    /// using the MaxSim formula.
    ///
    /// `query_tokens` has shape `[num_query_tokens, dim]`.
    pub fn search(
        &self,
        query_tokens: &Array2<f32>,
        top_k: usize,
    ) -> Result<Vec<MultiVectorResult>> {
        self.search_with_params(query_tokens, top_k, 50)
    }

    /// Approximate MaxSim search with configurable per-token k.
    pub fn search_with_params(
        &self,
        query_tokens: &Array2<f32>,
        top_k: usize,
        per_token_k: usize,
    ) -> Result<Vec<MultiVectorResult>> {
        let params = SearchParams::default();

        // For each query token, find nearest neighbors in the HNSW index.
        // Accumulate MaxSim: for each doc, sum of max scores across query tokens.
        let mut doc_scores: HashMap<u32, f32> = HashMap::new();

        for qi in 0..query_tokens.nrows() {
            let query_vec = query_tokens.row(qi);
            let query_slice = query_vec.as_slice().unwrap();

            let (labels_idx, distances) =
                backend::search_backend(&self.index, query_slice, per_token_k, &params);

            // For this query token, find best score per doc.
            // HNSW inner_product_distance returns -dot(a,b), so negate to get similarity.
            let mut best_per_doc: HashMap<u32, f32> = HashMap::new();
            for (idx, dist) in labels_idx.into_iter().zip(distances) {
                if idx >= self.labels.len() {
                    continue;
                }
                let doc_id = self.labels[idx].doc_id;
                let sim = -dist; // negate HNSW's negated inner product
                let entry = best_per_doc.entry(doc_id).or_insert(f32::NEG_INFINITY);
                if sim > *entry {
                    *entry = sim;
                }
            }

            // Accumulate into global scores.
            for (doc_id, score) in best_per_doc {
                *doc_scores.entry(doc_id).or_insert(0.0) += score;
            }
        }

        Ok(top_k_results(
            &doc_scores,
            top_k,
            &self.doc_to_rows,
            &self.labels,
        ))
    }

    /// Two-stage exact MaxSim search.
    ///
    /// Stage 1: approximate HNSW search to find candidate doc_ids.
    /// Stage 2: exact MaxSim reranking using mmap'd embeddings.
    pub fn search_exact(
        &self,
        query_tokens: &Array2<f32>,
        top_k: usize,
        first_stage_k: usize,
    ) -> Result<Vec<MultiVectorResult>> {
        // Stage 1: collect candidate doc_ids via approximate search.
        let approx = self.search_with_params(query_tokens, first_stage_k, 50)?;
        let candidate_docs: Vec<u32> = approx.iter().map(|r| r.doc_id).collect();

        if candidate_docs.is_empty() {
            return Ok(Vec::new());
        }

        // Parse the npy data to get the embedding slice.
        let emb_bytes = self.emb_bytes();
        let (header_len, _rows, _cols) = parse_npy_header(emb_bytes)?;
        let data_start = header_len;
        let float_data = &emb_bytes[data_start..];

        // Stage 2: exact MaxSim for each candidate.
        let mut doc_scores: HashMap<u32, f32> = HashMap::new();
        for &doc_id in &candidate_docs {
            if let Some(row_indices) = self.doc_to_rows.get(&doc_id) {
                let score = exact_max_sim(query_tokens, float_data, row_indices, self.dim);
                doc_scores.insert(doc_id, score);
            }
        }

        Ok(top_k_results(
            &doc_scores,
            top_k,
            &self.doc_to_rows,
            &self.labels,
        ))
    }

    fn emb_bytes(&self) -> &[u8] {
        #[cfg(feature = "multi-vector")]
        {
            &self.emb_mmap
        }
        #[cfg(not(feature = "multi-vector"))]
        {
            &self.emb_data
        }
    }
}

// ---------------------------------------------------------------------------
// Search result
// ---------------------------------------------------------------------------

/// A multi-vector search result (one per document).
#[derive(Debug, Clone)]
pub struct MultiVectorResult {
    pub doc_id: u32,
    pub score: f32,
    /// Per-document metadata from the first token label.
    pub metadata: HashMap<String, serde_json::Value>,
}

// ---------------------------------------------------------------------------
// MaxSim helpers
// ---------------------------------------------------------------------------

/// Compute exact MaxSim: Σ_i max_j (q_i · d_j).
fn exact_max_sim(
    query_tokens: &Array2<f32>,
    float_data: &[u8],
    doc_row_indices: &[usize],
    dim: usize,
) -> f32 {
    let mut total = 0.0f32;
    for qi in 0..query_tokens.nrows() {
        let q = query_tokens.row(qi);
        let mut best = f32::NEG_INFINITY;
        for &row_idx in doc_row_indices {
            let offset = row_idx * dim * 4;
            let end = offset + dim * 4;
            if end > float_data.len() {
                continue;
            }
            let dot = dot_product_bytes(q, &float_data[offset..end]);
            if dot > best {
                best = dot;
            }
        }
        if best > f32::NEG_INFINITY {
            total += best;
        }
    }
    total
}

/// Dot product between an ndarray row and raw LE f32 bytes.
#[inline]
fn dot_product_bytes(a: ArrayView1<f32>, b_bytes: &[u8]) -> f32 {
    let mut sum = 0.0f32;
    for (i, &ai) in a.iter().enumerate() {
        let offset = i * 4;
        let bi = f32::from_le_bytes(b_bytes[offset..offset + 4].try_into().unwrap());
        sum += ai * bi;
    }
    sum
}

/// Extract top-k docs from score map, sorted descending.
fn top_k_results(
    doc_scores: &HashMap<u32, f32>,
    top_k: usize,
    doc_to_rows: &HashMap<u32, Vec<usize>>,
    labels: &[TokenLabel],
) -> Vec<MultiVectorResult> {
    let mut entries: Vec<(u32, f32)> = doc_scores.iter().map(|(&d, &s)| (d, s)).collect();
    entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    entries.truncate(top_k);

    entries
        .into_iter()
        .map(|(doc_id, score)| {
            let metadata = doc_to_rows
                .get(&doc_id)
                .and_then(|rows| rows.first())
                .map(|&idx| labels[idx].metadata.clone())
                .unwrap_or_default();
            MultiVectorResult {
                doc_id,
                score,
                metadata,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Minimal NPY v1.0 read/write
// ---------------------------------------------------------------------------

/// Write an Array2<f32> as a NumPy .npy v1.0 file.
fn write_npy(arr: &Array2<f32>, path: &Path) -> Result<()> {
    let (rows, cols) = arr.dim();
    let header = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, {}), }}",
        rows, cols
    );
    // Pad header to 64-byte alignment (magic(6) + version(2) + header_len(2) + header + \n).
    let prefix_len = 10; // magic + version + header_len
    let total_unpadded = prefix_len + header.len() + 1; // +1 for trailing \n
    let padding = (64 - (total_unpadded % 64)) % 64;
    let header_content_len = header.len() + padding + 1; // header + spaces + \n

    let mut file = fs::File::create(path)?;
    // Magic
    file.write_all(&[0x93, b'N', b'U', b'M', b'P', b'Y'])?;
    // Version 1.0
    file.write_all(&[1, 0])?;
    // Header length (little-endian u16)
    file.write_all(&(header_content_len as u16).to_le_bytes())?;
    // Header string + padding + newline
    file.write_all(header.as_bytes())?;
    for _ in 0..padding {
        file.write_all(b" ")?;
    }
    file.write_all(b"\n")?;

    // Data: row-major f32 LE
    for val in arr.iter() {
        file.write_all(&val.to_le_bytes())?;
    }

    Ok(())
}

/// Parse a .npy v1.0 header, returning (data_offset, rows, cols).
fn parse_npy_header(data: &[u8]) -> Result<(usize, usize, usize)> {
    anyhow::ensure!(data.len() >= 10, "npy file too small");
    anyhow::ensure!(&data[0..6] == b"\x93NUMPY", "invalid npy magic");

    let header_len = u16::from_le_bytes([data[8], data[9]]) as usize;
    let header_end = 10 + header_len;
    anyhow::ensure!(data.len() >= header_end, "npy header truncated");

    let header_str = std::str::from_utf8(&data[10..header_end])?;
    // Parse shape tuple from the header string.
    let shape_start = header_str
        .find("'shape': (")
        .context("no shape in npy header")?
        + "'shape': (".len();
    let shape_end = header_str[shape_start..]
        .find(')')
        .context("unclosed shape tuple")?
        + shape_start;
    let shape_str = &header_str[shape_start..shape_end];
    let dims: Vec<usize> = shape_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    anyhow::ensure!(dims.len() == 2, "expected 2D shape, got {:?}", dims);

    Ok((header_end, dims[0], dims[1]))
}

// ---------------------------------------------------------------------------
// Path helpers
// ---------------------------------------------------------------------------

fn with_ext(base: &Path, ext: &str) -> PathBuf {
    let mut p = base.to_path_buf();
    let name = p
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    p.set_file_name(format!("{}.{}", name, ext));
    p
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn make_test_data() -> (Array2<f32>, Array2<f32>, Array2<f32>) {
        // Doc 0: 3 tokens, dim=4. Tokens point roughly in the +x direction.
        let doc0 = array![
            [1.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.8, 0.2, 0.0, 0.0],
        ];
        // Doc 1: 2 tokens, dim=4. Tokens point roughly in the +y direction.
        let doc1 = array![[0.0, 1.0, 0.0, 0.0], [0.1, 0.9, 0.0, 0.0],];
        // Query: 2 tokens — one in +x, one in +y. Should score both docs.
        let query = array![[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],];
        (doc0, doc1, query)
    }

    #[test]
    fn test_build_and_search() {
        let dir = tempfile::tempdir().unwrap();
        let index_path = dir.path().join("test_mv");

        let (doc0, doc1, query) = make_test_data();

        let mut builder = MultiVectorBuilder::new(4);
        builder.insert(0, doc0, HashMap::new());
        builder.insert(1, doc1, HashMap::new());
        builder.build(&index_path).unwrap();

        // Verify files exist.
        assert!(with_ext(&index_path, "index").exists());
        assert!(with_ext(&index_path, "labels.json").exists());
        assert!(with_ext(&index_path, "emb.npy").exists());

        let searcher = MultiVectorSearcher::open(&index_path).unwrap();
        assert_eq!(searcher.num_docs(), 2);
        assert_eq!(searcher.num_tokens(), 5);

        // Approximate search.
        let results = searcher.search(&query, 2).unwrap();
        assert_eq!(results.len(), 2);
        // Both docs should appear — doc0 best for +x query token, doc1 for +y.

        // Exact search.
        let exact_results = searcher.search_exact(&query, 2, 10).unwrap();
        assert_eq!(exact_results.len(), 2);
    }

    #[test]
    fn test_max_sim_scoring() {
        let dir = tempfile::tempdir().unwrap();
        let index_path = dir.path().join("test_scoring");

        // Doc 0: perfect match in +x
        let doc0 = array![[1.0, 0.0, 0.0, 0.0]];
        // Doc 1: perfect match in +y
        let doc1 = array![[0.0, 1.0, 0.0, 0.0]];
        // Query: just +x — should prefer doc0.
        let query = array![[1.0, 0.0, 0.0, 0.0]];

        let mut builder = MultiVectorBuilder::new(4);
        builder.insert(0, doc0, HashMap::new());
        builder.insert(1, doc1, HashMap::new());
        builder.build(&index_path).unwrap();

        let searcher = MultiVectorSearcher::open(&index_path).unwrap();
        let results = searcher.search_exact(&query, 2, 10).unwrap();

        assert_eq!(results[0].doc_id, 0);
        assert!(results[0].score > results[1].score);
        assert!((results[0].score - 1.0).abs() < 1e-5);
        assert!((results[1].score - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_npy_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.npy");

        let arr = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
        write_npy(&arr, &path).unwrap();

        let data = fs::read(&path).unwrap();
        let (header_len, rows, cols) = parse_npy_header(&data).unwrap();
        assert_eq!(rows, 2);
        assert_eq!(cols, 3);

        let float_data = &data[header_len..];
        assert_eq!(float_data.len(), 2 * 3 * 4);
        let first = f32::from_le_bytes(float_data[0..4].try_into().unwrap());
        assert!((first - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_metadata_propagation() {
        let dir = tempfile::tempdir().unwrap();
        let index_path = dir.path().join("test_meta");

        let doc0 = array![[1.0, 0.0]];
        let mut meta = HashMap::new();
        meta.insert("filepath".to_string(), serde_json::json!("/tmp/page1.png"));

        let mut builder = MultiVectorBuilder::new(2);
        builder.insert(42, doc0, meta);
        builder.build(&index_path).unwrap();

        let searcher = MultiVectorSearcher::open(&index_path).unwrap();
        let query = array![[1.0, 0.0]];
        let results = searcher.search(&query, 1).unwrap();

        assert_eq!(results[0].doc_id, 42);
        assert_eq!(results[0].metadata["filepath"], "/tmp/page1.png");
    }

    #[test]
    fn test_many_docs_ranking() {
        // 10 docs, each with tokens along a different basis direction.
        // Query for a specific direction should rank that doc first.
        let dir = tempfile::tempdir().unwrap();
        let index_path = dir.path().join("test_many");
        let dim = 16;

        let mut builder = MultiVectorBuilder::new(dim);
        for doc_id in 0..10u32 {
            let mut tokens = Array2::<f32>::zeros((3, dim));
            // Each doc's tokens have energy in dimension doc_id.
            for t in 0..3 {
                tokens[[t, doc_id as usize]] = 1.0;
                // Add some noise in other dims.
                tokens[[t, (doc_id as usize + 1) % dim]] = 0.1 * (t as f32);
            }
            builder.insert(doc_id, tokens, HashMap::new());
        }
        builder.build(&index_path).unwrap();

        let searcher = MultiVectorSearcher::open(&index_path).unwrap();
        assert_eq!(searcher.num_docs(), 10);

        // Query for doc 5's direction.
        let mut query = Array2::<f32>::zeros((1, dim));
        query[[0, 5]] = 1.0;

        let results = searcher.search_exact(&query, 3, 30).unwrap();
        assert_eq!(results[0].doc_id, 5);
    }

    #[test]
    fn test_multi_token_query_aggregation() {
        // Verify MaxSim aggregates across query tokens correctly.
        let dir = tempfile::tempdir().unwrap();
        let index_path = dir.path().join("test_agg");

        // Doc 0 has tokens in +x and +y.
        let doc0 = array![[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],];
        // Doc 1 has tokens only in +z.
        let doc1 = array![[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.9, 0.1],];

        let mut builder = MultiVectorBuilder::new(4);
        builder.insert(0, doc0, HashMap::new());
        builder.insert(1, doc1, HashMap::new());
        builder.build(&index_path).unwrap();

        let searcher = MultiVectorSearcher::open(&index_path).unwrap();

        // Query with +x and +y tokens — should strongly prefer doc0
        // since it matches both query tokens perfectly.
        let query = array![[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],];
        let results = searcher.search_exact(&query, 2, 10).unwrap();
        assert_eq!(results[0].doc_id, 0);
        // Doc 0 score: max(1,0) for q0=+x + max(0,1) for q1=+y = 2.0
        assert!((results[0].score - 2.0).abs() < 1e-5);
        // Doc 1 score: max(0,0) for q0=+x + max(0,0) for q1=+y ≈ 0.0 + 0.0
        assert!(results[1].score < 0.2);
    }

    #[test]
    fn test_single_doc_single_token() {
        let dir = tempfile::tempdir().unwrap();
        let index_path = dir.path().join("test_single");

        let doc = array![[0.6, 0.8]];
        let mut builder = MultiVectorBuilder::new(2);
        builder.insert(0, doc, HashMap::new());
        builder.build(&index_path).unwrap();

        let searcher = MultiVectorSearcher::open(&index_path).unwrap();
        assert_eq!(searcher.num_docs(), 1);
        assert_eq!(searcher.num_tokens(), 1);

        let query = array![[0.6, 0.8]];
        let results = searcher.search(&query, 1).unwrap();
        assert_eq!(results.len(), 1);
        // dot(q, d) = 0.36 + 0.64 = 1.0
        assert!((results[0].score - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_top_k_limits_results() {
        let dir = tempfile::tempdir().unwrap();
        let index_path = dir.path().join("test_topk");

        let mut builder = MultiVectorBuilder::new(4);
        for i in 0..5u32 {
            let doc = array![[1.0, 0.0, 0.0, 0.0]];
            builder.insert(i, doc, HashMap::new());
        }
        builder.build(&index_path).unwrap();

        let searcher = MultiVectorSearcher::open(&index_path).unwrap();
        let query = array![[1.0, 0.0, 0.0, 0.0]];

        let results = searcher.search(&query, 3).unwrap();
        assert_eq!(results.len(), 3);

        let results_all = searcher.search(&query, 10).unwrap();
        assert_eq!(results_all.len(), 5);
    }

    #[test]
    fn test_variable_token_counts() {
        // Documents with different numbers of tokens.
        let dir = tempfile::tempdir().unwrap();
        let index_path = dir.path().join("test_vartok");

        let doc0 = array![[1.0, 0.0]]; // 1 token
        let doc1 = array![[0.0, 1.0], [0.5, 0.5], [0.3, 0.7]]; // 3 tokens
        let doc2 = array![[0.7, 0.7], [0.8, 0.6]]; // 2 tokens

        let mut builder = MultiVectorBuilder::new(2);
        builder.insert(0, doc0, HashMap::new());
        builder.insert(1, doc1, HashMap::new());
        builder.insert(2, doc2, HashMap::new());
        builder.build(&index_path).unwrap();

        let searcher = MultiVectorSearcher::open(&index_path).unwrap();
        assert_eq!(searcher.num_docs(), 3);
        assert_eq!(searcher.num_tokens(), 6); // 1 + 3 + 2

        let query = array![[0.0, 1.0]];
        let results = searcher.search_exact(&query, 3, 10).unwrap();
        assert_eq!(results.len(), 3);
        // Doc 1 has a token [0, 1] — perfect match.
        assert_eq!(results[0].doc_id, 1);
    }

    #[test]
    fn test_labels_sidecar_format() {
        let dir = tempfile::tempdir().unwrap();
        let index_path = dir.path().join("test_labels");

        let doc0 = array![[1.0, 0.0], [0.0, 1.0]];
        let doc1 = array![[0.5, 0.5]];

        let mut meta0 = HashMap::new();
        meta0.insert("page".to_string(), serde_json::json!(1));

        let mut builder = MultiVectorBuilder::new(2);
        builder.insert(10, doc0, meta0);
        builder.insert(20, doc1, HashMap::new());
        builder.build(&index_path).unwrap();

        // Read and verify labels.json directly.
        let labels_path = with_ext(&index_path, "labels.json");
        let data = fs::read_to_string(&labels_path).unwrap();
        let labels: Vec<TokenLabel> = serde_json::from_str(&data).unwrap();

        assert_eq!(labels.len(), 3);
        assert_eq!(labels[0].doc_id, 10);
        assert_eq!(labels[0].seq_id, 0);
        assert_eq!(labels[0].metadata["page"], 1);
        assert_eq!(labels[1].doc_id, 10);
        assert_eq!(labels[1].seq_id, 1);
        assert_eq!(labels[2].doc_id, 20);
        assert_eq!(labels[2].seq_id, 0);
        assert!(labels[2].metadata.is_empty());
    }

    #[test]
    fn test_exact_vs_approximate_consistency() {
        // Exact and approximate search should agree on the top-1 result
        // for a well-separated dataset.
        let dir = tempfile::tempdir().unwrap();
        let index_path = dir.path().join("test_consistency");

        // 8 docs in distinct directions (need enough nodes for HNSW to work well).
        let dim = 8;
        let mut builder = MultiVectorBuilder::new(dim);
        for i in 0..8u32 {
            let mut emb = Array2::<f32>::zeros((1, dim));
            emb[[0, i as usize]] = 1.0;
            builder.insert(i, emb, HashMap::new());
        }
        builder.build(&index_path).unwrap();

        let searcher = MultiVectorSearcher::open(&index_path).unwrap();
        let mut query = Array2::<f32>::zeros((1, dim));
        query[[0, 2]] = 1.0;

        let exact = searcher.search_exact(&query, 1, 10).unwrap();
        assert_eq!(exact[0].doc_id, 2);
        assert!((exact[0].score - 1.0).abs() < 1e-5);

        // Approximate should also find doc 2 (well-separated).
        let approx = searcher.search(&query, 1).unwrap();
        assert_eq!(approx[0].doc_id, 2);
    }

    #[test]
    #[should_panic(expected = "no documents inserted")]
    fn test_build_empty_panics() {
        let dir = tempfile::tempdir().unwrap();
        let index_path = dir.path().join("test_empty");
        let builder = MultiVectorBuilder::new(4);
        builder.build(&index_path).unwrap();
    }

    #[test]
    #[should_panic(expected = "embedding dim 3 != expected 4")]
    fn test_dimension_mismatch_panics() {
        let mut builder = MultiVectorBuilder::new(4);
        builder.insert(0, array![[1.0, 2.0, 3.0]], HashMap::new());
    }
}
