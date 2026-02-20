//! LEANN — Lightweight Embedding-Augmented Nearest Neighbors.
//!
//! Build, search, and manage vector indexes with optional graph-based
//! embedding recomputation for 97% storage reduction.
//!
//! # Quick start
//!
//! ```ignore
//! use leann_core::{LeannBuilder, LeannSearcher};
//!
//! // Build an index
//! let mut builder = LeannBuilder::new("model-name", Some(384), "sentence-transformers");
//! builder.add_text("Hello world", Default::default());
//! builder.build_index(&path, &provider)?;
//!
//! // Search
//! let searcher = LeannSearcher::open(&path)?;
//! let results = searcher.search("hello", 5)?;
//! ```

pub(crate) mod bm25;
pub mod builder;
pub mod chat;
pub mod chunking;
pub mod document_loaders;
pub mod embedding;
pub mod hnsw;
pub mod index;
pub(crate) mod metadata_filter;
pub mod passages;
pub mod react_agent;
pub mod search_result;
pub mod searcher;
pub(crate) mod settings;
pub mod sync;

pub use builder::LeannBuilder;
pub use chat::LeannChat;
pub use index::IndexMeta;
pub use passages::{Passage, PassageManager};
pub use search_result::SearchResult;
pub use searcher::LeannSearcher;
