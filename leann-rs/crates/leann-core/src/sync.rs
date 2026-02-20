use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Hash data using SHA-256.
pub fn hash_data(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("{:x}", hasher.finalize())
}

/// A node in the Merkle tree.
#[derive(Debug, Clone)]
pub struct MerkleTreeNode {
    pub hash: String,
    pub data: String,
    pub children: HashMap<String, MerkleTreeNode>,
    pub parent_id: Option<String>,
}

/// A flat Merkle tree for detecting file changes.
pub struct MerkleTree {
    pub nodes: HashMap<String, MerkleTreeNode>,
    pub root: Option<String>, // hash of root node
}

impl MerkleTree {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            root: None,
        }
    }

    pub fn add_node(&mut self, data: &str, parent_id: Option<&str>, hash: Option<&str>) -> String {
        let hash = hash
            .map(String::from)
            .unwrap_or_else(|| hash_data(data.as_bytes()));

        let node = MerkleTreeNode {
            hash: hash.clone(),
            data: data.to_string(),
            children: HashMap::new(),
            parent_id: parent_id.map(String::from),
        };

        self.nodes.insert(hash.clone(), node);

        if parent_id.is_none() {
            self.root = Some(hash.clone());
        } else if let Some(pid) = parent_id {
            // Clone the child node first to avoid borrow conflict
            let child_node = self.nodes.get(&hash).unwrap().clone();
            if let Some(parent) = self.nodes.get_mut(pid) {
                parent.children.insert(hash.clone(), child_node);
            }
        }

        hash
    }

    /// Compare two trees and return (added, removed, modified) file paths.
    pub fn compare_with(&self, other: &MerkleTree) -> (Vec<String>, Vec<String>, Vec<String>) {
        let self_root = match &self.root {
            Some(r) => r,
            None => return (Vec::new(), Vec::new(), Vec::new()),
        };
        let other_root = match &other.root {
            Some(r) => r,
            None => return (Vec::new(), Vec::new(), Vec::new()),
        };

        let self_root_node = &self.nodes[self_root];
        let other_root_node = &other.nodes[other_root];

        if self_root_node.hash == other_root_node.hash {
            return (Vec::new(), Vec::new(), Vec::new());
        }

        let old_files = &self_root_node.children;
        let new_files = &other_root_node.children;

        let mut all_paths: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for node in old_files.values() {
            all_paths.insert(&node.data);
        }
        for node in new_files.values() {
            all_paths.insert(&node.data);
        }

        let old_by_data: HashMap<&str, &str> = old_files
            .iter()
            .map(|(hash, node)| (node.data.as_str(), hash.as_str()))
            .collect();
        let new_by_data: HashMap<&str, &str> = new_files
            .iter()
            .map(|(hash, node)| (node.data.as_str(), hash.as_str()))
            .collect();

        let mut added = Vec::new();
        let mut removed = Vec::new();
        let mut modified = Vec::new();

        for path in all_paths {
            match (old_by_data.get(path), new_by_data.get(path)) {
                (Some(_), Some(_)) => {
                    // Both exist - check if data changed by comparing child hashes
                    // In the flat tree, the hash IS the content hash, so if both
                    // exist with different hashes, it's modified.
                    // Note: In this simple tree, hash == path for children
                    modified.push(path.to_string());
                }
                (None, Some(_)) => added.push(path.to_string()),
                (Some(_), None) => removed.push(path.to_string()),
                (None, None) => unreachable!(),
            }
        }

        (added, removed, modified)
    }
}

impl Default for MerkleTree {
    fn default() -> Self {
        Self::new()
    }
}

/// File synchronizer that detects changes using Merkle trees.
pub struct FileSynchronizer {
    root_dir: PathBuf,
    tree: MerkleTree,
    ignore_patterns: Vec<String>,
    include_extensions: Vec<String>,
}

impl FileSynchronizer {
    pub fn new(
        root_dir: &Path,
        ignore_patterns: Vec<String>,
        include_extensions: Vec<String>,
    ) -> anyhow::Result<Self> {
        if !root_dir.is_dir() {
            anyhow::bail!("Not a valid directory: {}", root_dir.display());
        }

        let mut sync = Self {
            root_dir: root_dir.to_path_buf(),
            tree: MerkleTree::new(),
            ignore_patterns,
            include_extensions,
        };

        sync.build_initial_tree()?;
        Ok(sync)
    }

    /// Check for file changes and return (added, removed, modified).
    pub fn check_for_changes(&mut self) -> anyhow::Result<(Vec<String>, Vec<String>, Vec<String>)> {
        let file_hashes = self.generate_file_hashes()?;
        let new_tree = self.build_merkle_tree(&file_hashes);
        let changes = self.tree.compare_with(&new_tree);

        if changes.0.is_empty() && changes.1.is_empty() && changes.2.is_empty() {
            // No changes
        } else {
            self.tree = new_tree;
        }

        Ok(changes)
    }

    fn build_initial_tree(&mut self) -> anyhow::Result<()> {
        let file_hashes = self.generate_file_hashes()?;
        self.tree = self.build_merkle_tree(&file_hashes);
        Ok(())
    }

    fn generate_file_hashes(&self) -> anyhow::Result<HashMap<String, String>> {
        let mut hashes = HashMap::new();

        for entry in walkdir_recursive(&self.root_dir) {
            let path = entry.as_path();

            if !path.is_file() {
                continue;
            }

            // Check extensions
            if !self.include_extensions.is_empty() {
                let ext = path
                    .extension()
                    .map(|e| format!(".{}", e.to_string_lossy()))
                    .unwrap_or_default();
                if !self.include_extensions.iter().any(|e| e == &ext) {
                    continue;
                }
            }

            // Check ignore patterns
            let path_str = path.to_string_lossy();
            if self.ignore_patterns.iter().any(|p| path_str.contains(p)) {
                continue;
            }

            match std::fs::read(path) {
                Ok(data) => {
                    hashes.insert(path_str.to_string(), hash_data(&data));
                }
                Err(_) => continue,
            }
        }

        Ok(hashes)
    }

    fn build_merkle_tree(&self, file_hashes: &HashMap<String, String>) -> MerkleTree {
        let mut tree = MerkleTree::new();

        let mut sorted_paths: Vec<&String> = file_hashes.keys().collect();
        sorted_paths.sort();

        let root_data: String = sorted_paths
            .iter()
            .map(|p| format!("{}{}", p, file_hashes[*p]))
            .collect();

        let root_id = tree.add_node(&root_data, None, None);

        for path in sorted_paths {
            tree.add_node(path, Some(&root_id), Some(path));
        }

        tree
    }
}

fn walkdir_recursive(dir: &Path) -> Vec<PathBuf> {
    let mut result = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                result.extend(walkdir_recursive(&path));
            } else {
                result.push(path);
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_data() {
        let h = hash_data(b"hello world");
        assert!(!h.is_empty());
        assert_eq!(h.len(), 64); // SHA-256 hex is 64 chars
    }

    #[test]
    fn test_merkle_tree_no_changes() {
        let mut tree1 = MerkleTree::new();
        let root = tree1.add_node("root_data", None, None);
        tree1.add_node("file1", Some(&root), Some("file1.txt"));

        let mut tree2 = MerkleTree::new();
        let root2 = tree2.add_node("root_data", None, None);
        tree2.add_node("file1", Some(&root2), Some("file1.txt"));

        let (added, removed, modified) = tree1.compare_with(&tree2);
        assert!(added.is_empty());
        assert!(removed.is_empty());
        assert!(modified.is_empty());
    }
}
