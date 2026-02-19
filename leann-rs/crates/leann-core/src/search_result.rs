use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single search result with its associated metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: String,
    pub score: f64,
    pub text: String,
    pub metadata: HashMap<String, serde_json::Value>,
}

impl SearchResult {
    pub fn new(id: String, score: f64, text: String) -> Self {
        Self {
            id,
            score,
            text,
            metadata: HashMap::new(),
        }
    }

    pub fn with_metadata(
        id: String,
        score: f64,
        text: String,
        metadata: HashMap<String, serde_json::Value>,
    ) -> Self {
        Self {
            id,
            score,
            text,
            metadata,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_result_creation() {
        let result = SearchResult::new("doc1".into(), 0.95, "hello world".into());
        assert_eq!(result.id, "doc1");
        assert!((result.score - 0.95).abs() < f64::EPSILON);
        assert_eq!(result.text, "hello world");
        assert!(result.metadata.is_empty());
    }

    #[test]
    fn test_search_result_with_metadata() {
        let mut meta = HashMap::new();
        meta.insert(
            "source".to_string(),
            serde_json::Value::String("test.txt".to_string()),
        );
        let result = SearchResult::with_metadata("doc2".into(), 0.8, "test".into(), meta);
        assert_eq!(
            result.metadata.get("source"),
            Some(&serde_json::Value::String("test.txt".to_string()))
        );
    }

    #[test]
    fn test_search_result_serialization() {
        let result = SearchResult::new("doc1".into(), 0.95, "hello".into());
        let json = serde_json::to_string(&result).unwrap();
        let deserialized: SearchResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.id, result.id);
    }
}
