use serde_json::Value;
use std::collections::HashMap;
use tracing::warn;

/// A filter specification: maps operator names to expected values.
/// Example: `{"<=": 5}` or `{"in": ["fiction", "drama"]}`
pub type FilterSpec = HashMap<String, Value>;

/// Metadata filters: maps field names to their filter specifications.
/// Example: `{"chapter": {"<=": 5}, "tags": {"in": ["fiction", "drama"]}}`
pub type MetadataFilters = HashMap<String, FilterSpec>;

/// Engine for evaluating metadata filters against search results.
///
/// Supports various operators for filtering based on metadata fields:
/// - Comparison: `==`, `!=`, `<`, `<=`, `>`, `>=`
/// - Membership: `in`, `not_in`
/// - String operations: `contains`, `starts_with`, `ends_with`
/// - Boolean operations: `is_true`, `is_false`
pub struct MetadataFilterEngine;

impl MetadataFilterEngine {
    pub fn new() -> Self {
        Self
    }

    /// Apply metadata filters to a list of items. Each item is a map that should
    /// contain a `"metadata"` key with nested fields.
    pub fn apply_filters(
        &self,
        results: &[HashMap<String, Value>],
        filters: &MetadataFilters,
    ) -> Vec<HashMap<String, Value>> {
        if filters.is_empty() {
            return results.to_vec();
        }

        results
            .iter()
            .filter(|result| self.evaluate_filters(result, filters))
            .cloned()
            .collect()
    }

    /// Evaluate all filters against a single result (AND logic).
    fn evaluate_filters(&self, result: &HashMap<String, Value>, filters: &MetadataFilters) -> bool {
        for (field_name, filter_spec) in filters {
            if !self.evaluate_field_filter(result, field_name, filter_spec) {
                return false;
            }
        }
        true
    }

    /// Evaluate a single field filter against a result.
    fn evaluate_field_filter(
        &self,
        result: &HashMap<String, Value>,
        field_name: &str,
        filter_spec: &FilterSpec,
    ) -> bool {
        // First check top-level fields, then check metadata
        let field_value = result.get(field_name).or_else(|| {
            result
                .get("metadata")
                .and_then(|m| m.as_object())
                .and_then(|m| m.get(field_name))
        });

        let field_value = match field_value {
            Some(v) if !v.is_null() => v,
            _ => return false,
        };

        for (operator, expected_value) in filter_spec {
            let passes = match operator.as_str() {
                "==" => self.op_equals(field_value, expected_value),
                "!=" => self.op_not_equals(field_value, expected_value),
                "<" => self.op_less_than(field_value, expected_value),
                "<=" => self.op_less_than_or_equal(field_value, expected_value),
                ">" => self.op_greater_than(field_value, expected_value),
                ">=" => self.op_greater_than_or_equal(field_value, expected_value),
                "in" => self.op_in(field_value, expected_value),
                "not_in" => self.op_not_in(field_value, expected_value),
                "contains" => self.op_contains(field_value, expected_value),
                "starts_with" => self.op_starts_with(field_value, expected_value),
                "ends_with" => self.op_ends_with(field_value, expected_value),
                "is_true" => self.op_is_true(field_value),
                "is_false" => self.op_is_false(field_value),
                unknown => {
                    warn!("Unsupported filter operator: {}", unknown);
                    false
                }
            };

            if !passes {
                return false;
            }
        }

        true
    }

    // --- Comparison operators ---

    fn op_equals(&self, field: &Value, expected: &Value) -> bool {
        field == expected
    }

    fn op_not_equals(&self, field: &Value, expected: &Value) -> bool {
        field != expected
    }

    fn op_less_than(&self, field: &Value, expected: &Value) -> bool {
        self.numeric_compare(field, expected, |a, b| a < b)
    }

    fn op_less_than_or_equal(&self, field: &Value, expected: &Value) -> bool {
        self.numeric_compare(field, expected, |a, b| a <= b)
    }

    fn op_greater_than(&self, field: &Value, expected: &Value) -> bool {
        self.numeric_compare(field, expected, |a, b| a > b)
    }

    fn op_greater_than_or_equal(&self, field: &Value, expected: &Value) -> bool {
        self.numeric_compare(field, expected, |a, b| a >= b)
    }

    // --- Membership operators ---

    fn op_in(&self, field: &Value, expected: &Value) -> bool {
        match expected.as_array() {
            Some(arr) => arr.contains(field),
            None => false,
        }
    }

    fn op_not_in(&self, field: &Value, expected: &Value) -> bool {
        match expected.as_array() {
            Some(arr) => !arr.contains(field),
            None => true,
        }
    }

    // --- String operators ---

    fn op_contains(&self, field: &Value, expected: &Value) -> bool {
        let field_str = value_to_string(field);
        let expected_str = value_to_string(expected);
        field_str.contains(&expected_str)
    }

    fn op_starts_with(&self, field: &Value, expected: &Value) -> bool {
        let field_str = value_to_string(field);
        let expected_str = value_to_string(expected);
        field_str.starts_with(&expected_str)
    }

    fn op_ends_with(&self, field: &Value, expected: &Value) -> bool {
        let field_str = value_to_string(field);
        let expected_str = value_to_string(expected);
        field_str.ends_with(&expected_str)
    }

    // --- Boolean operators ---

    fn op_is_true(&self, field: &Value) -> bool {
        value_is_truthy(field)
    }

    fn op_is_false(&self, field: &Value) -> bool {
        !value_is_truthy(field)
    }

    // --- Helpers ---

    fn numeric_compare(&self, field: &Value, expected: &Value, cmp: fn(f64, f64) -> bool) -> bool {
        // If both are strings, do string comparison
        if let (Some(a), Some(b)) = (field.as_str(), expected.as_str()) {
            return cmp(
                a.parse::<f64>().unwrap_or(f64::NAN),
                b.parse::<f64>().unwrap_or(f64::NAN),
            );
        }

        // Try numeric comparison
        let field_num = value_to_f64(field);
        let expected_num = value_to_f64(expected);

        match (field_num, expected_num) {
            (Some(a), Some(b)) => cmp(a, b),
            _ => {
                // Fall back to string comparison
                let a = value_to_string(field);
                let b = value_to_string(expected);
                cmp(
                    a.parse::<f64>().unwrap_or(f64::NAN),
                    b.parse::<f64>().unwrap_or(f64::NAN),
                )
            }
        }
    }
}

impl Default for MetadataFilterEngine {
    fn default() -> Self {
        Self::new()
    }
}

fn value_to_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.parse().ok(),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        _ => None,
    }
}

fn value_to_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Null => "null".to_string(),
        other => other.to_string(),
    }
}

fn value_is_truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map_or(false, |f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_result(metadata: Value) -> HashMap<String, Value> {
        let mut result = HashMap::new();
        result.insert("id".to_string(), json!("doc1"));
        result.insert("score".to_string(), json!(0.95));
        result.insert("text".to_string(), json!("sample text"));
        result.insert("metadata".to_string(), metadata);
        result
    }

    fn make_filter(field: &str, op: &str, value: Value) -> MetadataFilters {
        let mut filters = MetadataFilters::new();
        let mut spec = FilterSpec::new();
        spec.insert(op.to_string(), value);
        filters.insert(field.to_string(), spec);
        filters
    }

    fn sample_results() -> Vec<HashMap<String, Value>> {
        vec![
            {
                let mut m = HashMap::new();
                m.insert("id".to_string(), json!("doc0"));
                m.insert("score".to_string(), json!(0.95));
                m.insert("text".to_string(), json!("A tale of two cities"));
                m.insert(
                    "metadata".to_string(),
                    json!({
                        "chapter": 1, "genre": "fiction", "word_count": 1500,
                        "is_published": true, "tags": ["classic", "drama"],
                        "topic": "topic_0", "doc_num": 0
                    }),
                );
                m
            },
            {
                let mut m = HashMap::new();
                m.insert("id".to_string(), json!("doc1"));
                m.insert("score".to_string(), json!(0.85));
                m.insert("text".to_string(), json!("Data structures and algorithms"));
                m.insert(
                    "metadata".to_string(),
                    json!({
                        "chapter": 5, "genre": "science", "word_count": 3000,
                        "is_published": true, "tags": ["textbook"],
                        "topic": "topic_1", "doc_num": 5
                    }),
                );
                m
            },
            {
                let mut m = HashMap::new();
                m.insert("id".to_string(), json!("doc2"));
                m.insert("score".to_string(), json!(0.75));
                m.insert("text".to_string(), json!("Pride and Prejudice"));
                m.insert(
                    "metadata".to_string(),
                    json!({
                        "chapter": 3, "genre": "fiction", "word_count": 2000,
                        "is_published": false, "tags": ["classic", "romance"],
                        "topic": "topic_0", "doc_num": 12
                    }),
                );
                m
            },
            {
                let mut m = HashMap::new();
                m.insert("id".to_string(), json!("doc3"));
                m.insert("score".to_string(), json!(0.65));
                m.insert(
                    "text".to_string(),
                    json!("Introduction to machine learning"),
                );
                m.insert(
                    "metadata".to_string(),
                    json!({
                        "chapter": 10, "genre": "science", "word_count": 5000,
                        "is_published": true, "tags": ["textbook", "ai"],
                        "topic": "topic_2", "doc_num": 25
                    }),
                );
                m
            },
        ]
    }

    #[test]
    fn test_equals_filter() {
        let engine = MetadataFilterEngine::new();
        let results = vec![
            make_result(json!({"chapter": 3})),
            make_result(json!({"chapter": 5})),
        ];

        let mut filters = MetadataFilters::new();
        let mut spec = FilterSpec::new();
        spec.insert("==".to_string(), json!(5));
        filters.insert("chapter".to_string(), spec);

        let filtered = engine.apply_filters(&results, &filters);
        assert_eq!(filtered.len(), 1);
    }

    #[test]
    fn test_less_than_filter() {
        let engine = MetadataFilterEngine::new();
        let results = vec![
            make_result(json!({"chapter": 3})),
            make_result(json!({"chapter": 5})),
            make_result(json!({"chapter": 7})),
        ];

        let mut filters = MetadataFilters::new();
        let mut spec = FilterSpec::new();
        spec.insert("<=".to_string(), json!(5));
        filters.insert("chapter".to_string(), spec);

        let filtered = engine.apply_filters(&results, &filters);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_in_filter() {
        let engine = MetadataFilterEngine::new();
        let results = vec![
            make_result(json!({"genre": "fiction"})),
            make_result(json!({"genre": "science"})),
            make_result(json!({"genre": "drama"})),
        ];

        let mut filters = MetadataFilters::new();
        let mut spec = FilterSpec::new();
        spec.insert("in".to_string(), json!(["fiction", "drama"]));
        filters.insert("genre".to_string(), spec);

        let filtered = engine.apply_filters(&results, &filters);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_contains_filter() {
        let engine = MetadataFilterEngine::new();
        let results = vec![
            make_result(json!({"source": "chapter1.txt"})),
            make_result(json!({"source": "appendix.txt"})),
        ];

        let mut filters = MetadataFilters::new();
        let mut spec = FilterSpec::new();
        spec.insert("contains".to_string(), json!("chapter"));
        filters.insert("source".to_string(), spec);

        let filtered = engine.apply_filters(&results, &filters);
        assert_eq!(filtered.len(), 1);
    }

    #[test]
    fn test_missing_field_fails_filter() {
        let engine = MetadataFilterEngine::new();
        let results = vec![make_result(json!({"other_field": "value"}))];

        let mut filters = MetadataFilters::new();
        let mut spec = FilterSpec::new();
        spec.insert("==".to_string(), json!(5));
        filters.insert("chapter".to_string(), spec);

        let filtered = engine.apply_filters(&results, &filters);
        assert_eq!(filtered.len(), 0);
    }

    #[test]
    fn test_empty_filters_returns_all() {
        let engine = MetadataFilterEngine::new();
        let results = vec![make_result(json!({"chapter": 3}))];
        let filters = MetadataFilters::new();
        let filtered = engine.apply_filters(&results, &filters);
        assert_eq!(filtered.len(), 1);
    }

    #[test]
    fn test_boolean_operators() {
        let engine = MetadataFilterEngine::new();
        let results = vec![
            make_result(json!({"active": true})),
            make_result(json!({"active": false})),
        ];

        let mut filters = MetadataFilters::new();
        let mut spec = FilterSpec::new();
        spec.insert("is_true".to_string(), json!(null));
        filters.insert("active".to_string(), spec);

        let filtered = engine.apply_filters(&results, &filters);
        assert_eq!(filtered.len(), 1);
    }

    #[test]
    fn test_multiple_filters_and_logic() {
        let engine = MetadataFilterEngine::new();
        let results = vec![
            make_result(json!({"chapter": 3, "genre": "fiction"})),
            make_result(json!({"chapter": 5, "genre": "fiction"})),
            make_result(json!({"chapter": 3, "genre": "science"})),
        ];

        let mut filters = MetadataFilters::new();
        let mut chapter_spec = FilterSpec::new();
        chapter_spec.insert("<=".to_string(), json!(3));
        filters.insert("chapter".to_string(), chapter_spec);

        let mut genre_spec = FilterSpec::new();
        genre_spec.insert("==".to_string(), json!("fiction"));
        filters.insert("genre".to_string(), genre_spec);

        let filtered = engine.apply_filters(&results, &filters);
        assert_eq!(filtered.len(), 1);
    }

    // --- Tests from test_metadata_filtering.rs (E2E-4) ---

    #[test]
    fn test_filter_equals() {
        let engine = MetadataFilterEngine::new();
        let dicts = sample_results();
        let filters = make_filter("genre", "==", json!("fiction"));
        let filtered = engine.apply_filters(&dicts, &filters);
        assert_eq!(filtered.len(), 2);
        for r in &filtered {
            let genre = r
                .get("metadata")
                .unwrap()
                .get("genre")
                .unwrap()
                .as_str()
                .unwrap();
            assert_eq!(genre, "fiction");
        }
    }

    #[test]
    fn test_filter_not_equals() {
        let engine = MetadataFilterEngine::new();
        let dicts = sample_results();
        let filters = make_filter("genre", "!=", json!("fiction"));
        let filtered = engine.apply_filters(&dicts, &filters);
        assert_eq!(filtered.len(), 2);
        for r in &filtered {
            let genre = r
                .get("metadata")
                .unwrap()
                .get("genre")
                .unwrap()
                .as_str()
                .unwrap();
            assert_ne!(genre, "fiction");
        }
    }

    #[test]
    fn test_filter_less_than() {
        let engine = MetadataFilterEngine::new();
        let dicts = sample_results();
        let filters = make_filter("chapter", "<", json!(5));
        let filtered = engine.apply_filters(&dicts, &filters);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_filter_less_than_or_equal() {
        let engine = MetadataFilterEngine::new();
        let dicts = sample_results();
        let filters = make_filter("chapter", "<=", json!(5));
        let filtered = engine.apply_filters(&dicts, &filters);
        assert_eq!(filtered.len(), 3);
    }

    #[test]
    fn test_filter_greater_than() {
        let engine = MetadataFilterEngine::new();
        let dicts = sample_results();
        let filters = make_filter("chapter", ">", json!(5));
        let filtered = engine.apply_filters(&dicts, &filters);
        assert_eq!(filtered.len(), 1);
    }

    #[test]
    fn test_filter_greater_equal() {
        let engine = MetadataFilterEngine::new();
        let dicts = sample_results();
        let filters = make_filter("doc_num", ">=", json!(12));
        let filtered = engine.apply_filters(&dicts, &filters);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_filter_in() {
        let engine = MetadataFilterEngine::new();
        let dicts = sample_results();
        let filters = make_filter("topic", "in", json!(["topic_0", "topic_1"]));
        let filtered = engine.apply_filters(&dicts, &filters);
        assert_eq!(filtered.len(), 3);
    }

    #[test]
    fn test_filter_not_in() {
        let engine = MetadataFilterEngine::new();
        let dicts = sample_results();
        let filters = make_filter("topic", "not_in", json!(["topic_0"]));
        let filtered = engine.apply_filters(&dicts, &filters);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_filter_contains() {
        let engine = MetadataFilterEngine::new();
        let dicts = sample_results();
        let filters = make_filter("topic", "contains", json!("_0"));
        let filtered = engine.apply_filters(&dicts, &filters);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_filter_starts_with() {
        let engine = MetadataFilterEngine::new();
        let dicts = sample_results();
        let filters = make_filter("topic", "starts_with", json!("topic_"));
        let filtered = engine.apply_filters(&dicts, &filters);
        assert_eq!(filtered.len(), 4);
    }

    #[test]
    fn test_filter_ends_with() {
        let engine = MetadataFilterEngine::new();
        let dicts = sample_results();
        let filters = make_filter("topic", "ends_with", json!("_0"));
        let filtered = engine.apply_filters(&dicts, &filters);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_filter_is_true() {
        let engine = MetadataFilterEngine::new();
        let dicts = sample_results();
        let filters = make_filter("is_published", "is_true", json!(null));
        let filtered = engine.apply_filters(&dicts, &filters);
        assert_eq!(filtered.len(), 3);
    }

    #[test]
    fn test_filter_is_false() {
        let engine = MetadataFilterEngine::new();
        let dicts = sample_results();
        let filters = make_filter("is_published", "is_false", json!(null));
        let filtered = engine.apply_filters(&dicts, &filters);
        assert_eq!(filtered.len(), 1);
    }

    #[test]
    fn test_filter_compound_and() {
        let engine = MetadataFilterEngine::new();
        let dicts = sample_results();

        let mut filters = MetadataFilters::new();
        let mut genre_spec = FilterSpec::new();
        genre_spec.insert("==".to_string(), json!("fiction"));
        filters.insert("genre".to_string(), genre_spec);

        let mut chapter_spec = FilterSpec::new();
        chapter_spec.insert("<=".to_string(), json!(3));
        filters.insert("chapter".to_string(), chapter_spec);

        let filtered = engine.apply_filters(&dicts, &filters);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_filter_range() {
        let engine = MetadataFilterEngine::new();
        let dicts = sample_results();

        let mut filters = MetadataFilters::new();
        let mut spec = FilterSpec::new();
        spec.insert(">=".to_string(), json!(2000));
        spec.insert("<".to_string(), json!(4000));
        filters.insert("word_count".to_string(), spec);

        let filtered = engine.apply_filters(&dicts, &filters);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_filter_no_matches() {
        let engine = MetadataFilterEngine::new();
        let dicts = sample_results();
        let filters = make_filter("chapter", "==", json!(999));
        let filtered = engine.apply_filters(&dicts, &filters);
        assert!(filtered.is_empty());
    }

    #[test]
    fn test_filter_none_passthrough() {
        let engine = MetadataFilterEngine::new();
        let dicts = sample_results();
        let filters = MetadataFilters::new();
        let filtered = engine.apply_filters(&dicts, &filters);
        assert_eq!(filtered.len(), 4);
    }

    #[test]
    fn test_filter_missing_field() {
        let engine = MetadataFilterEngine::new();
        let dicts = sample_results();
        let filters = make_filter("nonexistent_field", "==", json!("value"));
        let filtered = engine.apply_filters(&dicts, &filters);
        assert!(filtered.is_empty(), "Missing field should exclude all");
    }

    #[test]
    fn test_filter_invalid_operator() {
        let engine = MetadataFilterEngine::new();
        let dicts = sample_results();
        let filters = make_filter("genre", "regex", json!("fic.*"));
        let filtered = engine.apply_filters(&dicts, &filters);
        assert!(
            filtered.is_empty(),
            "Unknown operator should filter out all items, got {} results",
            filtered.len()
        );
    }

    #[test]
    fn test_filter_type_coercion() {
        let engine = MetadataFilterEngine::new();

        let mut results = Vec::new();
        for (i, val_str) in ["1", "3", "7", "10"].iter().enumerate() {
            let mut map = HashMap::new();
            map.insert("id".to_string(), json!(format!("doc{}", i)));
            map.insert("score".to_string(), json!(0.9));
            map.insert("text".to_string(), json!("text"));
            map.insert(
                "metadata".to_string(),
                json!({ "priority": val_str.to_string() }),
            );
            results.push(map);
        }

        let filters = make_filter("priority", "<", json!(5));
        let filtered = engine.apply_filters(&results, &filters);
        assert_eq!(
            filtered.len(),
            2,
            "String '1' and '3' should pass < 5 via coercion, got {:?}",
            filtered.iter().map(|r| r.get("id")).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_filter_empty_results_list() {
        let engine = MetadataFilterEngine::new();
        let empty: Vec<HashMap<String, Value>> = vec![];
        let filters = make_filter("chapter", "==", json!(1));
        let filtered = engine.apply_filters(&empty, &filters);
        assert!(
            filtered.is_empty(),
            "Filtering empty list should return empty"
        );
    }
}
