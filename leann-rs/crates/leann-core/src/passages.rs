use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::index::PassageSource;
use crate::metadata_filter::{MetadataFilterEngine, MetadataFilters};
use crate::search_result::SearchResult;

/// A single passage as stored in the JSONL file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Passage {
    pub id: String,
    pub text: String,
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Manages passage storage and retrieval using JSONL files with offset-based random access.
pub struct PassageManager {
    /// Per-shard offset maps: passage_file_path -> (passage_id -> byte_offset)
    offset_maps: HashMap<PathBuf, HashMap<String, u64>>,
    /// Total number of passages across all shards.
    total_count: usize,
    /// Filter engine for metadata filtering.
    filter_engine: MetadataFilterEngine,
}

impl PassageManager {
    /// Create a PassageManager from passage sources defined in metadata.
    pub fn load(
        passage_sources: &[PassageSource],
        metadata_file_path: Option<&Path>,
    ) -> Result<Self> {
        let mut offset_maps: HashMap<PathBuf, HashMap<String, u64>> = HashMap::new();
        let mut total_count = 0;

        // Derive index base name for standard sibling fallbacks
        let index_name_base = metadata_file_path.and_then(|p| {
            let name = p.file_name()?.to_str()?;
            name.strip_suffix(".meta.json").map(String::from)
        });

        for source in passage_sources {
            anyhow::ensure!(
                source.source_type == "jsonl",
                "Only jsonl passage sources are supported, got: {}",
                source.source_type
            );

            let idx_default = index_name_base
                .as_ref()
                .map(|base| format!("{}.passages.idx", base));
            let pas_default = index_name_base
                .as_ref()
                .map(|base| format!("{}.passages.jsonl", base));

            let idx_candidates = resolve_candidates(
                &source.index_path,
                source.index_path_relative.as_deref(),
                idx_default.as_deref(),
                metadata_file_path,
            );
            let pas_candidates = resolve_candidates(
                &source.path,
                source.path_relative.as_deref(),
                pas_default.as_deref(),
                metadata_file_path,
            );

            let index_file = pick_existing(&idx_candidates)?;
            let passage_file = pick_existing(&pas_candidates)?;

            anyhow::ensure!(
                index_file.exists(),
                "Passage index file not found: {}",
                index_file.display()
            );

            // Load the offset map (bincode format for Rust, or migration from pickle)
            let offset_map = load_offset_map(&index_file)?;
            total_count += offset_map.len();
            offset_maps.insert(passage_file, offset_map);
        }

        Ok(Self {
            offset_maps,
            total_count,
            filter_engine: MetadataFilterEngine::new(),
        })
    }

    /// Get a single passage by ID.
    pub fn get_passage(&self, passage_id: &str) -> Result<Passage> {
        for (passage_file, offset_map) in &self.offset_maps {
            if let Some(&offset) = offset_map.get(passage_id) {
                let mut file = File::open(passage_file)
                    .with_context(|| format!("opening {}", passage_file.display()))?;
                file.seek(SeekFrom::Start(offset))?;
                let mut reader = BufReader::new(file);
                let mut line = String::new();
                reader.read_line(&mut line)?;
                let passage: Passage = serde_json::from_str(&line).with_context(|| {
                    format!("parsing passage {} at offset {}", passage_id, offset)
                })?;
                return Ok(passage);
            }
        }
        anyhow::bail!("Passage ID not found: {}", passage_id)
    }

    /// Filter search results by metadata.
    pub fn filter_search_results(
        &self,
        results: &[SearchResult],
        filters: &MetadataFilters,
    ) -> Vec<SearchResult> {
        if filters.is_empty() {
            return results.to_vec();
        }

        // Convert to dict format for the filter engine
        let result_dicts: Vec<HashMap<String, serde_json::Value>> = results
            .iter()
            .map(|r| {
                let mut map = HashMap::new();
                map.insert("id".to_string(), serde_json::Value::String(r.id.clone()));
                map.insert("score".to_string(), serde_json::json!(r.score));
                map.insert(
                    "text".to_string(),
                    serde_json::Value::String(r.text.clone()),
                );
                map.insert(
                    "metadata".to_string(),
                    serde_json::to_value(&r.metadata).unwrap_or_default(),
                );
                map
            })
            .collect();

        let filtered = self.filter_engine.apply_filters(&result_dicts, filters);

        filtered
            .into_iter()
            .filter_map(|d| {
                Some(SearchResult {
                    id: d.get("id")?.as_str()?.to_string(),
                    score: d.get("score")?.as_f64()?,
                    text: d.get("text")?.as_str()?.to_string(),
                    metadata: d
                        .get("metadata")
                        .and_then(|v| serde_json::from_value(v.clone()).ok())
                        .unwrap_or_default(),
                })
            })
            .collect()
    }

    /// Total passage count across all shards.
    pub fn len(&self) -> usize {
        self.total_count
    }

    pub fn is_empty(&self) -> bool {
        self.total_count == 0
    }

    /// Iterator over all passage file paths.
    pub fn passage_files(&self) -> impl Iterator<Item = &Path> {
        self.offset_maps.keys().map(|p| p.as_path())
    }
}

/// Write passages to JSONL and create an offset map.
pub fn write_passages(
    chunks: &[Passage],
    passages_path: &Path,
    offset_path: &Path,
) -> Result<HashMap<String, u64>> {
    let mut file = File::create(passages_path)?;
    let mut offset_map = HashMap::new();

    for chunk in chunks {
        let offset = file.stream_position()?;
        serde_json::to_writer(&mut file, chunk)?;
        file.write_all(b"\n")?;
        offset_map.insert(chunk.id.clone(), offset);
    }

    // Write offset map using bincode
    let offset_file = File::create(offset_path)?;
    bincode::serialize_into(offset_file, &offset_map)?;

    Ok(offset_map)
}

/// Write the ID map file (one ID per line, in order).
pub fn write_id_map(ids: &[String], path: &Path) -> Result<()> {
    let mut file = File::create(path)?;
    for id in ids {
        writeln!(file, "{}", id)?;
    }
    Ok(())
}

/// Load an ID map file.
pub fn load_id_map(path: &Path) -> Result<Vec<String>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut ids = Vec::new();
    for line in reader.lines() {
        ids.push(line?.trim_end().to_string());
    }
    Ok(ids)
}

// --- Private helpers ---

fn resolve_candidates(
    primary: &str,
    relative_key: Option<&str>,
    default_name: Option<&str>,
    metadata_file_path: Option<&Path>,
) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    let meta_dir = metadata_file_path.and_then(|p| p.parent());

    // 1) Primary path
    if !primary.is_empty() {
        let p = PathBuf::from(primary);
        if p.is_absolute() {
            candidates.push(p);
        } else {
            if let Some(dir) = meta_dir {
                candidates.push(dir.join(&p));
            }
            candidates.push(std::env::current_dir().unwrap_or_default().join(&p));
        }
    }

    // 2) Relative key
    if let (Some(dir), Some(rel)) = (meta_dir, relative_key) {
        if !rel.is_empty() {
            candidates.push(dir.join(rel));
        }
    }

    // 3) Standard sibling default
    if let (Some(dir), Some(name)) = (meta_dir, default_name) {
        candidates.push(dir.join(name));
    }

    candidates
}

fn pick_existing(candidates: &[PathBuf]) -> Result<PathBuf> {
    for c in candidates {
        if c.exists() {
            return Ok(c.canonicalize().unwrap_or_else(|_| c.clone()));
        }
    }
    // Return last candidate as best guess
    candidates
        .last()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("No path candidates provided"))
}

/// Load offset map. Try bincode first (Rust-native), then fall back to a simple
/// JSON format for migration from Python pickle.
fn load_offset_map(path: &Path) -> Result<HashMap<String, u64>> {
    let mut file = File::open(path)?;

    // Try bincode first
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;

    if let Ok(map) = bincode::deserialize::<HashMap<String, u64>>(&buf) {
        return Ok(map);
    }

    // Try JSON fallback (for migration)
    if let Ok(map) = serde_json::from_slice::<HashMap<String, u64>>(&buf) {
        return Ok(map);
    }

    // Try Python pickle offset format: these are Python pickle protocol 2+ files.
    // We implement a minimal parser for the common case: dict of str -> int.
    parse_python_pickle_offset_map(&buf)
        .with_context(|| format!("Failed to parse offset map at {}", path.display()))
}

/// Minimal parser for Python pickle offset maps (dict[str, int]).
/// Supports the common pickle protocol used by Python's pickle.dump().
fn parse_python_pickle_offset_map(data: &[u8]) -> Result<HashMap<String, u64>> {
    // Python pickle protocol 2+ starts with \x80\x02 (or higher protocol number).
    // We use a simple state machine approach to extract string keys and integer values.
    let mut map = HashMap::new();
    let mut pos = 0;

    if data.len() < 2 {
        anyhow::bail!("Data too short for pickle format");
    }

    // Check pickle protocol header
    if data[0] == 0x80 {
        // Protocol version byte
        pos = 2; // Skip \x80 + version byte
    }

    // Opcodes we care about:
    // } = EMPTY_DICT (0x7d)
    // X = SHORT_BINUNICODE (0x8c) - protocol 4
    // q = BINPUT (0x71)
    // r = LONG_BINPUT (0x72)
    // J = BININT (0x4a)
    // K = BININT1 (0x4b)
    // M = BININT2 (0x4d)
    // s = SETITEM (0x73)
    // u = SETITEMS (0x75)
    // . = STOP (0x2e)
    // 0 = POP (0x30)
    // 8c = SHORT_BINUNICODE

    let mut stack: Vec<PickleValue> = Vec::new();
    let mut mark_positions: Vec<usize> = Vec::new();

    while pos < data.len() {
        let opcode = data[pos];
        pos += 1;

        match opcode {
            0x80 => {
                // PROTO - skip version byte
                pos += 1;
            }
            0x7d => {
                // EMPTY_DICT
                stack.push(PickleValue::Dict(HashMap::new()));
            }
            0x28 => {
                // MARK
                mark_positions.push(stack.len());
            }
            0x8c => {
                // SHORT_BINUNICODE
                if pos >= data.len() {
                    break;
                }
                let len = data[pos] as usize;
                pos += 1;
                if pos + len > data.len() {
                    break;
                }
                let s = String::from_utf8_lossy(&data[pos..pos + len]).to_string();
                pos += len;
                stack.push(PickleValue::Str(s));
            }
            0x8d => {
                // BINUNICODE8
                if pos + 8 > data.len() {
                    break;
                }
                let len = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap()) as usize;
                pos += 8;
                if pos + len > data.len() {
                    break;
                }
                let s = String::from_utf8_lossy(&data[pos..pos + len]).to_string();
                pos += len;
                stack.push(PickleValue::Str(s));
            }
            0x58 => {
                // BINUNICODE (4 byte length)
                if pos + 4 > data.len() {
                    break;
                }
                let len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
                pos += 4;
                if pos + len > data.len() {
                    break;
                }
                let s = String::from_utf8_lossy(&data[pos..pos + len]).to_string();
                pos += len;
                stack.push(PickleValue::Str(s));
            }
            0x4a => {
                // BININT (4 byte signed int)
                if pos + 4 > data.len() {
                    break;
                }
                let v = i32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as i64;
                pos += 4;
                stack.push(PickleValue::Int(v));
            }
            0x4b => {
                // BININT1 (1 byte unsigned int)
                if pos >= data.len() {
                    break;
                }
                let v = data[pos] as i64;
                pos += 1;
                stack.push(PickleValue::Int(v));
            }
            0x4d => {
                // BININT2 (2 byte unsigned int)
                if pos + 2 > data.len() {
                    break;
                }
                let v = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()) as i64;
                pos += 2;
                stack.push(PickleValue::Int(v));
            }
            0x8a => {
                // LONG1 (1-byte length followed by little-endian twos-complement)
                if pos >= data.len() {
                    break;
                }
                let nbytes = data[pos] as usize;
                pos += 1;
                if pos + nbytes > data.len() {
                    break;
                }
                let val = read_long_bytes(&data[pos..pos + nbytes]);
                pos += nbytes;
                stack.push(PickleValue::Int(val));
            }
            0x8b => {
                // LONG4 (4-byte length)
                if pos + 4 > data.len() {
                    break;
                }
                let nbytes = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
                pos += 4;
                if pos + nbytes > data.len() {
                    break;
                }
                let val = read_long_bytes(&data[pos..pos + nbytes]);
                pos += nbytes;
                stack.push(PickleValue::Int(val));
            }
            0x73 => {
                // SETITEM: pop value, pop key, peek dict, insert
                if stack.len() >= 3 {
                    let value = stack.pop().unwrap();
                    let key = stack.pop().unwrap();
                    if let Some(PickleValue::Dict(d)) = stack.last_mut() {
                        if let (PickleValue::Str(k), PickleValue::Int(v)) = (key, value) {
                            d.insert(k, v as u64);
                        }
                    }
                }
            }
            0x75 => {
                // SETITEMS: pop pairs from mark to top, insert into dict below mark
                if let Some(mark_pos) = mark_positions.pop() {
                    let items: Vec<PickleValue> = stack.drain(mark_pos..).collect();
                    if let Some(PickleValue::Dict(d)) = stack.last_mut() {
                        for chunk in items.chunks(2) {
                            if let [PickleValue::Str(k), PickleValue::Int(v)] = chunk {
                                d.insert(k.clone(), *v as u64);
                            }
                        }
                    }
                }
            }
            0x71 => {
                // BINPUT (memo) - skip 1 byte
                pos += 1;
            }
            0x72 => {
                // LONG_BINPUT (memo) - skip 4 bytes
                pos += 4;
            }
            0x94 => {
                // MEMOIZE (protocol 4) - no argument
            }
            0x95 => {
                // FRAME (protocol 4) - skip 8 bytes
                pos += 8;
            }
            0x30 => {
                // POP
                stack.pop();
            }
            0x2e => {
                // STOP
                break;
            }
            _ => {
                // Unknown opcode - try to continue
            }
        }
    }

    // The final value on the stack should be our dict
    if let Some(PickleValue::Dict(d)) = stack.pop() {
        map = d;
    }

    if map.is_empty() {
        anyhow::bail!("Failed to parse any entries from pickle offset map");
    }

    Ok(map)
}

fn read_long_bytes(bytes: &[u8]) -> i64 {
    if bytes.is_empty() {
        return 0;
    }
    let mut val: i64 = 0;
    for (i, &b) in bytes.iter().enumerate() {
        val |= (b as i64) << (8 * i);
    }
    // Sign extend if the high bit of the last byte is set
    if bytes.last().map_or(false, |&b| b & 0x80 != 0) {
        let bits = bytes.len() * 8;
        if bits < 64 {
            val |= !0i64 << bits;
        }
    }
    val
}

#[derive(Debug, Clone)]
enum PickleValue {
    Str(String),
    Int(i64),
    Dict(HashMap<String, u64>),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_and_read_passages() {
        let dir = tempfile::tempdir().unwrap();
        let passages_path = dir.path().join("test.passages.jsonl");
        let offset_path = dir.path().join("test.passages.idx");

        let passages = vec![
            Passage {
                id: "0".to_string(),
                text: "Hello world".to_string(),
                metadata: HashMap::new(),
            },
            Passage {
                id: "1".to_string(),
                text: "Rust is great".to_string(),
                metadata: {
                    let mut m = HashMap::new();
                    m.insert("source".to_string(), serde_json::json!("test.rs"));
                    m
                },
            },
        ];

        let offset_map = write_passages(&passages, &passages_path, &offset_path).unwrap();
        assert_eq!(offset_map.len(), 2);

        // Now load and verify
        let sources = vec![PassageSource {
            source_type: "jsonl".to_string(),
            path: passages_path.to_string_lossy().to_string(),
            index_path: offset_path.to_string_lossy().to_string(),
            path_relative: None,
            index_path_relative: None,
        }];

        let manager = PassageManager::load(&sources, None).unwrap();
        assert_eq!(manager.len(), 2);

        let p0 = manager.get_passage("0").unwrap();
        assert_eq!(p0.text, "Hello world");

        let p1 = manager.get_passage("1").unwrap();
        assert_eq!(p1.text, "Rust is great");
        assert_eq!(
            p1.metadata.get("source"),
            Some(&serde_json::json!("test.rs"))
        );
    }

    #[test]
    fn test_passage_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let passages_path = dir.path().join("test.passages.jsonl");
        let offset_path = dir.path().join("test.passages.idx");

        let passages = vec![Passage {
            id: "0".to_string(),
            text: "Hello".to_string(),
            metadata: HashMap::new(),
        }];

        write_passages(&passages, &passages_path, &offset_path).unwrap();

        let sources = vec![PassageSource {
            source_type: "jsonl".to_string(),
            path: passages_path.to_string_lossy().to_string(),
            index_path: offset_path.to_string_lossy().to_string(),
            path_relative: None,
            index_path_relative: None,
        }];

        let manager = PassageManager::load(&sources, None).unwrap();
        assert!(manager.get_passage("999").is_err());
    }

    #[test]
    fn test_write_and_load_id_map() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.ids.txt");

        let ids: Vec<String> = vec!["doc_0".into(), "doc_1".into(), "doc_2".into()];
        write_id_map(&ids, &path).unwrap();

        let loaded = load_id_map(&path).unwrap();
        assert_eq!(loaded, ids);
    }

    // --- Tests moved from test_index_format.rs ---

    /// Helper: build a test index using FakeEmbeddingProvider (inlined from tests/common).
    fn build_test_index(
        n_docs: usize,
        dir: &std::path::Path,
        compact: bool,
        recompute: bool,
    ) -> Result<std::path::PathBuf> {
        use crate::embedding::EmbeddingProvider;
        use crate::index::DistanceMetric;

        struct FakeEmbeddingProvider {
            dims: usize,
        }
        impl FakeEmbeddingProvider {
            fn new(dims: usize) -> Self {
                Self { dims }
            }
            fn text_to_vector(&self, text: &str) -> Vec<f32> {
                let bytes = text.as_bytes();
                let mut vec = vec![0.0f32; self.dims];
                for (i, &b) in bytes.iter().enumerate() {
                    vec[i % self.dims] += b as f32 / 255.0;
                }
                let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for v in &mut vec {
                        *v /= norm;
                    }
                }
                vec
            }
        }
        impl EmbeddingProvider for FakeEmbeddingProvider {
            fn compute_embeddings(&self, chunks: &[String]) -> Result<ndarray::Array2<f32>> {
                let mut data = Vec::with_capacity(chunks.len() * self.dims);
                for chunk in chunks {
                    data.extend(self.text_to_vector(chunk));
                }
                Ok(ndarray::Array2::from_shape_vec(
                    (chunks.len(), self.dims),
                    data,
                )?)
            }
            fn dimensions(&self) -> usize {
                self.dims
            }
            fn name(&self) -> &str {
                "fake-test-provider"
            }
        }

        let provider = FakeEmbeddingProvider::new(64);
        let mut builder = crate::builder::LeannBuilder::new("fake-test-model", Some(64), "test");
        builder = builder
            .with_m(16)
            .with_ef_construction(40)
            .with_compact(compact)
            .with_recompute(recompute)
            .with_distance_metric(DistanceMetric::L2);

        for i in 0..n_docs {
            let topic = format!("topic_{}", i % 5);
            let text = format!("This is document {} about {}", i, topic);
            let mut meta = HashMap::new();
            meta.insert("id".to_string(), serde_json::json!(i.to_string()));
            meta.insert("doc_num".to_string(), serde_json::json!(i));
            meta.insert("topic".to_string(), serde_json::json!(topic));
            builder.add_text(&text, meta);
        }

        let index_path = dir.join("test_index");
        builder.build_index(&index_path, &provider)?;
        Ok(index_path)
    }

    #[test]
    fn test_passages_jsonl_format() {
        use crate::index::IndexPaths;
        use std::io::BufRead;

        let dir = tempfile::tempdir().unwrap();
        let index_path = build_test_index(15, dir.path(), true, true).unwrap();
        let paths = IndexPaths::new(&index_path);

        let file = std::fs::File::open(paths.passages_path()).unwrap();
        let reader = std::io::BufReader::new(file);
        let mut count = 0;

        for line in reader.lines() {
            let line = line.unwrap();
            let parsed: serde_json::Value = serde_json::from_str(&line)
                .unwrap_or_else(|e| panic!("Invalid JSON on line {}: {}", count + 1, e));

            assert!(
                parsed.get("id").is_some(),
                "Missing 'id' field on line {}",
                count + 1
            );
            assert!(
                parsed.get("text").is_some(),
                "Missing 'text' field on line {}",
                count + 1
            );

            let text = parsed["text"].as_str().unwrap();
            assert!(!text.is_empty(), "Empty text on line {}", count + 1);

            count += 1;
        }

        assert_eq!(count, 15, "Expected 15 passages, got {}", count);
    }

    #[test]
    fn test_id_map_roundtrip_50() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.ids.txt");

        let ids: Vec<String> = (0..50).map(|i| format!("doc_{}", i)).collect();
        write_id_map(&ids, &path).unwrap();

        let loaded = load_id_map(&path).unwrap();
        assert_eq!(loaded.len(), ids.len());
        for (a, b) in ids.iter().zip(loaded.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn test_passages_offset_random_access() {
        let dir = tempfile::tempdir().unwrap();
        let passages_path = dir.path().join("test.passages.jsonl");
        let offset_path = dir.path().join("test.passages.idx");

        let passages: Vec<Passage> = (0..20)
            .map(|i| Passage {
                id: format!("p_{}", i),
                text: format!(
                    "Passage number {} with some content about topic {}",
                    i,
                    i % 3
                ),
                metadata: {
                    let mut m = HashMap::new();
                    m.insert("index".to_string(), serde_json::json!(i));
                    m
                },
            })
            .collect();

        let offset_map = write_passages(&passages, &passages_path, &offset_path).unwrap();
        assert_eq!(offset_map.len(), 20);

        let sources = vec![PassageSource {
            source_type: "jsonl".to_string(),
            path: passages_path.to_string_lossy().to_string(),
            index_path: offset_path.to_string_lossy().to_string(),
            path_relative: None,
            index_path_relative: None,
        }];

        let manager = PassageManager::load(&sources, None).unwrap();

        for i in [15, 3, 0, 19, 7, 12] {
            let p = manager.get_passage(&format!("p_{}", i)).unwrap();
            assert!(
                p.text.contains(&format!("Passage number {}", i)),
                "Wrong passage for p_{}: '{}'",
                i,
                p.text
            );
        }
    }

    #[test]
    fn test_passage_sources_reference_valid_files() {
        use crate::index::{IndexMeta, IndexPaths};

        let dir = tempfile::tempdir().unwrap();
        let index_path = build_test_index(10, dir.path(), true, true).unwrap();
        let paths = IndexPaths::new(&index_path);

        let meta = IndexMeta::load(&paths.meta_path()).unwrap();

        for source in &meta.passage_sources {
            assert_eq!(source.source_type, "jsonl");
            assert!(
                !source.path.is_empty(),
                "Passage source path should not be empty"
            );
        }

        let manager =
            PassageManager::load(&meta.passage_sources, Some(&paths.meta_path())).unwrap();
        assert_eq!(manager.len(), 10);
    }

    // --- Tests moved from test_build_search.rs ---

    #[test]
    fn test_id_map_roundtrip_after_build() {
        use crate::index::IndexPaths;

        let dir = tempfile::tempdir().unwrap();
        let index_path = build_test_index(25, dir.path(), true, true).unwrap();
        let paths = IndexPaths::new(&index_path);

        let ids = load_id_map(&paths.id_map_path()).unwrap();
        assert_eq!(ids.len(), 25);
        for i in 0..25 {
            assert_eq!(ids[i], i.to_string());
        }
    }

    #[test]
    fn test_passage_random_access_after_build() {
        use crate::index::{IndexMeta, IndexPaths};

        let dir = tempfile::tempdir().unwrap();
        let index_path = build_test_index(30, dir.path(), true, true).unwrap();
        let paths = IndexPaths::new(&index_path);

        let meta = IndexMeta::load(&paths.meta_path()).unwrap();
        let manager =
            PassageManager::load(&meta.passage_sources, Some(&paths.meta_path())).unwrap();
        assert_eq!(manager.len(), 30);

        let p0 = manager.get_passage("0").unwrap();
        assert!(p0.text.contains("document 0"));

        let p15 = manager.get_passage("15").unwrap();
        assert!(p15.text.contains("document 15"));

        assert!(manager.get_passage("999").is_err());
    }
}
