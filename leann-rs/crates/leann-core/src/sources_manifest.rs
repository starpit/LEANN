//! Sources manifest for incremental build support.
//!
//! Tracks `{file_path: mtime}` in `documents.leann.sources.json` so that
//! subsequent builds can detect new, modified, or removed files without
//! re-scanning content.

use anyhow::Result;
use std::collections::HashMap;
use std::path::Path;

/// Filename for the sources manifest, matching the Python convention.
pub const SOURCES_MANIFEST_FILENAME: &str = "documents.leann.sources.json";

/// Serialisation wrapper: `{ "sources": { path: mtime, ... } }`.
#[derive(serde::Serialize, serde::Deserialize)]
struct Manifest {
    sources: HashMap<String, f64>,
}

/// Load the sources manifest from `<index_dir>/documents.leann.sources.json`.
///
/// Returns an empty map if the file is missing or contains invalid JSON.
pub fn load_sources_manifest(index_dir: &Path) -> Result<HashMap<String, f64>> {
    let path = index_dir.join(SOURCES_MANIFEST_FILENAME);
    if !path.exists() {
        return Ok(HashMap::new());
    }
    let data = std::fs::read_to_string(&path)?;
    match serde_json::from_str::<Manifest>(&data) {
        Ok(m) => Ok(m.sources),
        Err(_) => Ok(HashMap::new()),
    }
}

/// Save the sources manifest to `<index_dir>/documents.leann.sources.json`.
pub fn save_sources_manifest(index_dir: &Path, sources: &HashMap<String, f64>) -> Result<()> {
    let path = index_dir.join(SOURCES_MANIFEST_FILENAME);
    let manifest = Manifest {
        sources: sources.clone(),
    };
    let json = serde_json::to_string_pretty(&manifest)?;
    std::fs::write(&path, json)?;
    Ok(())
}

/// Collect `{normalized_path: mtime}` for a set of loaded documents.
///
/// `documents` is `&[(path_string, _content)]` — the same structure produced
/// by the CLI document-loading loop.
pub fn collect_sources(documents: &[(String, String)]) -> HashMap<String, f64> {
    let mut sources = HashMap::new();
    for (path, _) in documents {
        let mtime = std::fs::metadata(path)
            .and_then(|m| m.modified())
            .map(|t| {
                t.duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs_f64()
            })
            .unwrap_or(0.0);
        // Normalize to absolute path when possible
        let normalized = std::fs::canonicalize(path)
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|_| path.clone());
        sources.insert(normalized, mtime);
    }
    sources
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_save_and_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let mut sources = HashMap::new();
        sources.insert("/tmp/a.txt".to_string(), 1700000000.123);
        sources.insert("/tmp/b.md".to_string(), 1700000001.456);

        save_sources_manifest(dir.path(), &sources).unwrap();
        let loaded = load_sources_manifest(dir.path()).unwrap();

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded["/tmp/a.txt"], 1700000000.123);
        assert_eq!(loaded["/tmp/b.md"], 1700000001.456);
    }

    #[test]
    fn test_load_missing_returns_empty() {
        let dir = tempfile::tempdir().unwrap();
        let loaded = load_sources_manifest(dir.path()).unwrap();
        assert!(loaded.is_empty());
    }

    #[test]
    fn test_load_invalid_json_returns_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(SOURCES_MANIFEST_FILENAME);
        std::fs::write(&path, "not valid json {{{").unwrap();
        let loaded = load_sources_manifest(dir.path()).unwrap();
        assert!(loaded.is_empty());
    }

    #[test]
    fn test_collect_sources_from_real_files() {
        let dir = tempfile::tempdir().unwrap();
        let file_a = dir.path().join("a.txt");
        let file_b = dir.path().join("b.txt");
        std::fs::write(&file_a, "content a").unwrap();
        std::fs::write(&file_b, "content b").unwrap();

        let documents = vec![
            (
                file_a.to_string_lossy().to_string(),
                "content a".to_string(),
            ),
            (
                file_b.to_string_lossy().to_string(),
                "content b".to_string(),
            ),
        ];

        let sources = collect_sources(&documents);
        assert_eq!(sources.len(), 2);

        // Keys should be canonical paths
        let canonical_a = std::fs::canonicalize(&file_a)
            .unwrap()
            .to_string_lossy()
            .to_string();
        let canonical_b = std::fs::canonicalize(&file_b)
            .unwrap()
            .to_string_lossy()
            .to_string();
        assert!(sources.contains_key(&canonical_a), "Should contain a.txt");
        assert!(sources.contains_key(&canonical_b), "Should contain b.txt");

        // Mtimes should be recent (within last minute)
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();
        for mtime in sources.values() {
            assert!(
                *mtime > now - 60.0 && *mtime <= now + 1.0,
                "mtime {mtime} should be recent (now={now})"
            );
        }
    }

    #[test]
    fn test_collect_sources_nonexistent_file() {
        let documents = vec![(
            "/nonexistent/path/file.txt".to_string(),
            "content".to_string(),
        )];
        let sources = collect_sources(&documents);
        assert_eq!(sources.len(), 1);
        // Non-existent file: can't canonicalize, falls back to original path
        assert!(sources.contains_key("/nonexistent/path/file.txt"));
        // mtime defaults to 0.0 for non-existent files
        assert_eq!(sources["/nonexistent/path/file.txt"], 0.0);
    }
}
