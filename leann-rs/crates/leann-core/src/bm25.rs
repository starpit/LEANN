use std::collections::{HashMap, HashSet};

use crate::search_result::SearchResult;

/// BM25 scoring for keyword-based search.
pub struct BM25Scorer {
    k1: f64,
    b: f64,
    /// Document frequency: how many docs contain each term.
    doc_freqs: HashMap<String, usize>,
    /// Length (in tokens) of each document.
    doc_lengths: HashMap<String, usize>,
    /// Term frequency: word counts per document.
    word_counts: HashMap<String, HashMap<String, usize>>,
    avg_doc_length: f64,
    corpus_size: usize,
    /// Set of all document IDs.
    id_set: HashSet<String>,
}

impl BM25Scorer {
    pub fn new(k1: f64, b: f64) -> Self {
        Self {
            k1,
            b,
            doc_freqs: HashMap::new(),
            doc_lengths: HashMap::new(),
            word_counts: HashMap::new(),
            avg_doc_length: 0.0,
            corpus_size: 0,
            id_set: HashSet::new(),
        }
    }

    /// Tokenize text: strip punctuation, lowercase, split on whitespace.
    /// Single-pass char scanner — no regex, no intermediate String allocations.
    fn tokenize(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut buf = String::new();
        for ch in text.chars() {
            if ch.is_alphanumeric() || ch == '_' {
                for lc in ch.to_lowercase() {
                    buf.push(lc);
                }
            } else if ch.is_whitespace() && !buf.is_empty() {
                tokens.push(std::mem::take(&mut buf));
            }
            // else: punctuation — strip (equivalent to regex [^\w\s] → "")
        }
        if !buf.is_empty() {
            tokens.push(buf);
        }
        tokens
    }

    /// Emit a token into `counts`, reusing `buf` when the token already exists.
    #[inline]
    fn emit_token(buf: &mut String, counts: &mut HashMap<String, usize>) {
        if buf.is_empty() {
            return;
        }
        // Fast path: token already seen → increment count, reuse buf allocation.
        if let Some(c) = counts.get_mut(buf.as_str()) {
            *c += 1;
            buf.clear();
        } else {
            // First occurrence → move buf into the map (zero-copy).
            counts.insert(std::mem::take(buf), 1);
        }
    }

    /// Build BM25 statistics from a document corpus.
    /// Each document should have "id" and "text" fields.
    pub fn fit(&mut self, documents: &[(String, String)]) {
        self.corpus_size = documents.len();
        self.doc_lengths.clear();
        self.word_counts.clear();
        self.id_set.clear();
        let mut doc_freqs: HashMap<String, usize> = HashMap::new();
        let mut total_length: usize = 0;

        for (doc_id, text) in documents {
            // Single-pass: tokenize + count directly, no intermediate Vec.
            let mut counts: HashMap<String, usize> = HashMap::new();
            let mut buf = String::new();
            let mut doc_length: usize = 0;

            for ch in text.chars() {
                if ch.is_alphanumeric() || ch == '_' {
                    for lc in ch.to_lowercase() {
                        buf.push(lc);
                    }
                } else if ch.is_whitespace() && !buf.is_empty() {
                    doc_length += 1;
                    Self::emit_token(&mut buf, &mut counts);
                }
            }
            if !buf.is_empty() {
                doc_length += 1;
                Self::emit_token(&mut buf, &mut counts);
            }

            // Unique words = counts.keys() — no separate HashSet needed.
            for word in counts.keys() {
                *doc_freqs.entry(word.clone()).or_insert(0) += 1;
            }

            self.doc_lengths.insert(doc_id.clone(), doc_length);
            total_length += doc_length;
            self.word_counts.insert(doc_id.clone(), counts);
            self.id_set.insert(doc_id.clone());
        }

        self.doc_freqs = doc_freqs;
        self.avg_doc_length = if self.corpus_size > 0 {
            total_length as f64 / self.corpus_size as f64
        } else {
            0.0
        };
    }

    /// Score a single document against a query.
    pub fn score(&self, query_words: &[String], document_id: &str) -> f64 {
        let passage_words = match self.word_counts.get(document_id) {
            Some(w) => w,
            None => return 0.0,
        };

        let passage_length: usize = passage_words.values().sum();
        let mut score = 0.0;

        for word in query_words {
            let df = match self.doc_freqs.get(word) {
                Some(&f) => f,
                None => continue,
            };

            let word_freq = *passage_words.get(word).unwrap_or(&0) as f64;

            let idf = ((self.corpus_size as f64 - df as f64 + 0.5) / (df as f64 + 0.5) + 1.0).ln();

            let tf = (word_freq * (self.k1 + 1.0))
                / (word_freq
                    + self.k1
                        * (1.0 - self.b + self.b * (passage_length as f64 / self.avg_doc_length)));

            score += idf * tf;
        }

        score
    }

    /// Search all documents and return top-k results.
    pub fn search(&self, query: &str, top_k: usize) -> Vec<SearchResult> {
        let query_words = self.tokenize(query);

        let mut scores: Vec<(String, f64)> = self
            .id_set
            .iter()
            .map(|doc_id| {
                let s = self.score(&query_words, doc_id);
                (doc_id.clone(), s)
            })
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);

        scores
            .into_iter()
            .map(|(id, score)| SearchResult::new(id, score, String::new()))
            .collect()
    }

    #[allow(dead_code)]
    pub fn is_fitted(&self) -> bool {
        self.corpus_size > 0
    }
}

impl Default for BM25Scorer {
    fn default() -> Self {
        Self::new(1.2, 0.75)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_docs() -> Vec<(String, String)> {
        vec![
            ("0".into(), "the cat sat on the mat".into()),
            ("1".into(), "the dog sat on the log".into()),
            ("2".into(), "the cat and the dog are friends".into()),
            ("3".into(), "birds fly in the sky".into()),
        ]
    }

    #[test]
    fn test_bm25_fit_and_search() {
        let mut scorer = BM25Scorer::default();
        scorer.fit(&sample_docs());

        assert!(scorer.is_fitted());
        assert_eq!(scorer.corpus_size, 4);

        let results = scorer.search("cat", 2);
        assert_eq!(results.len(), 2);
        // Documents mentioning "cat" should score highest
        assert!(results[0].id == "0" || results[0].id == "2");
    }

    #[test]
    fn test_bm25_score_nonexistent_term() {
        let mut scorer = BM25Scorer::default();
        scorer.fit(&sample_docs());

        let query_words = vec!["xyz123nonexistent".to_string()];
        let score = scorer.score(&query_words, "0");
        assert!((score - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_bm25_tokenizer() {
        let scorer = BM25Scorer::default();
        let tokens = scorer.tokenize("Hello, World! This is a TEST.");
        assert_eq!(tokens, vec!["hello", "world", "this", "is", "a", "test"]);
    }

    #[test]
    fn test_bm25_empty_corpus() {
        let mut scorer = BM25Scorer::default();
        scorer.fit(&[]);
        assert!(!scorer.is_fitted());
        let results = scorer.search("query", 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_bm25_single_document_corpus() {
        let mut scorer = BM25Scorer::default();
        scorer.fit(&[("0".into(), "the cat sat on the mat".into())]);

        assert!(scorer.is_fitted());
        assert_eq!(scorer.corpus_size, 1);

        let results = scorer.search("cat", 5);
        assert_eq!(results.len(), 1);
        assert!(
            results[0].score > 0.0,
            "Single doc matching query should have positive score, got {}",
            results[0].score
        );

        // Non-matching query should still return the doc but with zero score
        let results = scorer.search("xyz", 5);
        assert_eq!(results.len(), 1);
        assert!(
            results[0].score.abs() < f64::EPSILON,
            "Non-matching query on single doc should score 0"
        );
    }

    #[test]
    fn test_bm25_repeated_query_terms() {
        let mut scorer = BM25Scorer::default();
        scorer.fit(&sample_docs());

        // "cat cat cat" should not crash and should boost cat-containing docs
        let results_repeated = scorer.search("cat cat cat", 4);
        let results_single = scorer.search("cat", 4);

        assert_eq!(results_repeated.len(), results_single.len());

        // Same ranking order expected
        assert_eq!(
            results_repeated[0].id, results_single[0].id,
            "Repeated terms should maintain same top result"
        );

        // Repeated term should produce higher score (term counted multiple times)
        assert!(
            results_repeated[0].score >= results_single[0].score,
            "Repeated term score ({}) should be >= single term score ({})",
            results_repeated[0].score,
            results_single[0].score
        );
    }

    // --- Tests from test_bm25_search.rs (E2E) ---

    fn large_corpus() -> Vec<(String, String)> {
        vec![
            ("0".into(), "Python is a versatile programming language used for web development, data science, and machine learning".into()),
            ("1".into(), "JavaScript runs in web browsers and is essential for front-end web development and user interfaces".into()),
            ("2".into(), "Rust provides memory safety without garbage collection, making it ideal for systems programming".into()),
            ("3".into(), "Machine learning algorithms can identify patterns in large datasets and make predictions".into()),
            ("4".into(), "Database systems store and retrieve data efficiently using indexing and query optimization".into()),
            ("5".into(), "Cloud computing offers scalable infrastructure and services like storage and computation".into()),
            ("6".into(), "Neural networks are a subset of machine learning inspired by the human brain".into()),
            ("7".into(), "The weather forecast uses atmospheric models to predict temperature and rainfall".into()),
            ("8".into(), "Cooking Italian cuisine requires fresh ingredients like olive oil, tomatoes, and basil".into()),
            ("9".into(), "Ancient Egyptian pyramids were engineering marvels built over four thousand years ago".into()),
            ("10".into(), "Vector databases enable similarity search across high-dimensional embedding spaces".into()),
            ("11".into(), "Deep learning has revolutionized computer vision, natural language processing, and speech recognition".into()),
            ("12".into(), "Graph databases model relationships between entities using nodes and edges".into()),
            ("13".into(), "Functional programming emphasizes immutability and pure functions without side effects".into()),
            ("14".into(), "The human genome contains approximately three billion base pairs of DNA".into()),
        ]
    }

    #[test]
    fn test_bm25_programming_query() {
        let mut scorer = BM25Scorer::default();
        scorer.fit(&large_corpus());

        let results = scorer.search("programming language", 5);
        assert_eq!(results.len(), 5);

        let top_ids: Vec<&str> = results.iter().take(3).map(|r| r.id.as_str()).collect();
        assert!(
            top_ids.contains(&"0") || top_ids.contains(&"2") || top_ids.contains(&"13"),
            "Top results for 'programming language' should include Python/Rust/Functional: got {:?}",
            top_ids
        );
    }

    #[test]
    fn test_bm25_machine_learning_query() {
        let mut scorer = BM25Scorer::default();
        scorer.fit(&large_corpus());

        let results = scorer.search("machine learning neural networks", 5);
        assert_eq!(results.len(), 5);

        let top_ids: Vec<&str> = results.iter().take(3).map(|r| r.id.as_str()).collect();
        let has_ml = top_ids.contains(&"3") || top_ids.contains(&"6") || top_ids.contains(&"11");
        assert!(
            has_ml,
            "Top results for 'machine learning neural networks' should include ML docs: got {:?}",
            top_ids
        );
    }

    #[test]
    fn test_bm25_database_query() {
        let mut scorer = BM25Scorer::default();
        scorer.fit(&large_corpus());

        let results = scorer.search("database indexing query", 3);
        assert!(!results.is_empty());

        let top_id = &results[0].id;
        assert!(
            top_id == "4" || top_id == "12" || top_id == "10",
            "Top result for 'database' should be doc 4, 10, or 12: got {}",
            top_id
        );
    }

    #[test]
    fn test_bm25_scores_are_positive_for_matches() {
        let mut scorer = BM25Scorer::default();
        scorer.fit(&large_corpus());

        let results = scorer.search("Python programming", 5);
        assert!(
            results[0].score > 0.0,
            "Top BM25 score should be positive for matching query"
        );
    }

    #[test]
    fn test_bm25_no_match_zero_scores() {
        let mut scorer = BM25Scorer::default();
        scorer.fit(&large_corpus());

        let results = scorer.search("xyznonexistentterm123", 3);
        for r in &results {
            assert!(
                r.score.abs() < 1e-10,
                "Score should be ~0 for nonexistent term, got {}",
                r.score
            );
        }
    }

    #[test]
    fn test_bm25_scores_descending() {
        let mut scorer = BM25Scorer::default();
        scorer.fit(&large_corpus());

        let results = scorer.search("web development JavaScript", 10);
        for i in 1..results.len() {
            assert!(
                results[i].score <= results[i - 1].score + 1e-10,
                "BM25 scores not descending at pos {}: {} > {}",
                i,
                results[i].score,
                results[i - 1].score
            );
        }
    }

    #[test]
    fn test_bm25_top_k_larger_than_corpus() {
        let mut scorer = BM25Scorer::default();
        scorer.fit(&large_corpus());

        let results = scorer.search("Python", 100);
        assert_eq!(
            results.len(),
            15,
            "Should return all 15 docs when top_k > corpus size"
        );
    }

    #[test]
    fn test_bm25_custom_k1_b() {
        let mut scorer = BM25Scorer::new(2.0, 0.5);
        scorer.fit(&large_corpus());

        let results = scorer.search("machine learning", 3);
        assert!(!results.is_empty());
        assert!(results[0].score > 0.0);
    }
}
