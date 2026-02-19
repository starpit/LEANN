use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Represents the metadata stored in `<name>.meta.json` for a LEANN index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMeta {
    pub version: String,
    pub backend_name: String,
    pub embedding_model: String,
    pub dimensions: usize,
    #[serde(default)]
    pub backend_kwargs: HashMap<String, serde_json::Value>,
    #[serde(default = "default_embedding_mode")]
    pub embedding_mode: String,
    #[serde(default)]
    pub passage_sources: Vec<PassageSource>,
    #[serde(default)]
    pub embedding_options: HashMap<String, serde_json::Value>,
    /// Whether the HNSW index uses compact CSR storage.
    #[serde(default)]
    pub is_compact: Option<bool>,
    /// Whether embeddings have been pruned (for recompute mode).
    #[serde(default)]
    pub is_pruned: Option<bool>,
    /// Total passages in the index (updated on append).
    #[serde(default)]
    pub total_passages: Option<usize>,
    /// Set if built from pre-computed embeddings.
    #[serde(default)]
    pub built_from_precomputed_embeddings: Option<bool>,
    #[serde(default)]
    pub embeddings_source: Option<String>,
}

fn default_embedding_mode() -> String {
    "sentence-transformers".to_string()
}

/// Describes a passage source (JSONL file + offset index).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PassageSource {
    #[serde(rename = "type")]
    pub source_type: String,
    #[serde(default)]
    pub path: String,
    #[serde(default)]
    pub index_path: String,
    #[serde(default)]
    pub path_relative: Option<String>,
    #[serde(default)]
    pub index_path_relative: Option<String>,
}

/// The distance metric used for vector similarity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DistanceMetric {
    Mips,
    L2,
    Cosine,
}

impl DistanceMetric {
    pub fn from_str_lossy(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "l2" => DistanceMetric::L2,
            "cosine" => DistanceMetric::Cosine,
            _ => DistanceMetric::Mips,
        }
    }
}

impl Default for DistanceMetric {
    fn default() -> Self {
        DistanceMetric::Mips
    }
}

impl IndexMeta {
    /// Load metadata from a `.meta.json` file.
    pub fn load(path: &Path) -> Result<Self> {
        let content =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let meta: IndexMeta = serde_json::from_str(&content)
            .with_context(|| format!("parsing meta.json at {}", path.display()))?;
        Ok(meta)
    }

    /// Save metadata to a `.meta.json` file.
    pub fn save(&self, path: &Path) -> Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Get the distance metric from backend kwargs.
    pub fn distance_metric(&self) -> DistanceMetric {
        self.backend_kwargs
            .get("distance_metric")
            .and_then(|v| v.as_str())
            .map(DistanceMetric::from_str_lossy)
            .unwrap_or_default()
    }

    /// Whether this index requires embedding recomputation at search time.
    pub fn requires_recompute(&self) -> bool {
        if let Some(pruned) = self.is_pruned {
            return pruned;
        }
        self.backend_kwargs
            .get("is_recompute")
            .and_then(|v| v.as_bool())
            .unwrap_or(true)
    }
}

/// Resolve file paths associated with an index.
pub struct IndexPaths {
    pub base_dir: PathBuf,
    pub index_name: String,
}

impl IndexPaths {
    pub fn new(index_path: &Path) -> Self {
        let base_dir = index_path.parent().unwrap_or(Path::new(".")).to_path_buf();
        let index_name = index_path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        Self {
            base_dir,
            index_name,
        }
    }

    pub fn meta_path(&self) -> PathBuf {
        self.base_dir.join(format!("{}.meta.json", self.index_name))
    }

    pub fn passages_path(&self) -> PathBuf {
        self.base_dir
            .join(format!("{}.passages.jsonl", self.index_name))
    }

    pub fn offset_path(&self) -> PathBuf {
        self.base_dir
            .join(format!("{}.passages.idx", self.index_name))
    }

    pub fn index_file_path(&self) -> PathBuf {
        // The index file uses the stem (without .leann extension)
        let stem = self
            .index_name
            .strip_suffix(".leann")
            .unwrap_or(&self.index_name);
        self.base_dir.join(format!("{}.index", stem))
    }

    pub fn id_map_path(&self) -> PathBuf {
        let stem = self
            .index_name
            .strip_suffix(".leann")
            .unwrap_or(&self.index_name);
        self.base_dir.join(format!("{}.ids.txt", stem))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_index_meta_roundtrip() {
        let meta = IndexMeta {
            version: "1.0".to_string(),
            backend_name: "hnsw".to_string(),
            embedding_model: "facebook/contriever".to_string(),
            dimensions: 768,
            backend_kwargs: HashMap::new(),
            embedding_mode: "sentence-transformers".to_string(),
            passage_sources: vec![],
            embedding_options: HashMap::new(),
            is_compact: Some(true),
            is_pruned: Some(true),
            total_passages: None,
            built_from_precomputed_embeddings: None,
            embeddings_source: None,
        };

        let json = serde_json::to_string_pretty(&meta).unwrap();
        let deserialized: IndexMeta = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.backend_name, "hnsw");
        assert_eq!(deserialized.dimensions, 768);
    }

    #[test]
    fn test_index_meta_load() {
        let mut file = NamedTempFile::new().unwrap();
        write!(
            file,
            r#"{{
            "version": "1.0",
            "backend_name": "hnsw",
            "embedding_model": "test-model",
            "dimensions": 384,
            "passage_sources": []
        }}"#
        )
        .unwrap();

        let meta = IndexMeta::load(file.path()).unwrap();
        assert_eq!(meta.embedding_model, "test-model");
        assert_eq!(meta.dimensions, 384);
        assert_eq!(meta.embedding_mode, "sentence-transformers");
    }

    #[test]
    fn test_distance_metric_parsing() {
        assert_eq!(DistanceMetric::from_str_lossy("mips"), DistanceMetric::Mips);
        assert_eq!(DistanceMetric::from_str_lossy("l2"), DistanceMetric::L2);
        assert_eq!(
            DistanceMetric::from_str_lossy("cosine"),
            DistanceMetric::Cosine
        );
        assert_eq!(
            DistanceMetric::from_str_lossy("unknown"),
            DistanceMetric::Mips
        );
    }

    #[test]
    fn test_index_paths() {
        let paths = IndexPaths::new(Path::new("/data/my_index.leann"));
        assert_eq!(paths.base_dir, Path::new("/data"));
        assert_eq!(paths.index_name, "my_index.leann");
        assert_eq!(
            paths.meta_path(),
            Path::new("/data/my_index.leann.meta.json")
        );
        assert_eq!(
            paths.passages_path(),
            Path::new("/data/my_index.leann.passages.jsonl")
        );
        assert_eq!(paths.index_file_path(), Path::new("/data/my_index.index"));
        assert_eq!(paths.id_map_path(), Path::new("/data/my_index.ids.txt"));
    }
}
