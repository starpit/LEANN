use regex::Regex;
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
    /// Compiled regex for tokenization.
    tokenizer_re: Regex,
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
            tokenizer_re: Regex::new(r"[^\w\s]").unwrap(),
        }
    }

    /// Tokenize text by removing punctuation and lowercasing.
    fn tokenize(&self, text: &str) -> Vec<String> {
        let cleaned = self.tokenizer_re.replace_all(text, "");
        cleaned
            .to_lowercase()
            .split_whitespace()
            .map(String::from)
            .collect()
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
            let words = self.tokenize(text);
            let doc_length = words.len();
            self.doc_lengths.insert(doc_id.clone(), doc_length);
            total_length += doc_length;

            let unique_words: HashSet<&String> = words.iter().collect();
            for word in &unique_words {
                *doc_freqs.entry((*word).clone()).or_insert(0) += 1;
            }

            let mut counts: HashMap<String, usize> = HashMap::new();
            for word in &words {
                *counts.entry(word.clone()).or_insert(0) += 1;
            }
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
}
