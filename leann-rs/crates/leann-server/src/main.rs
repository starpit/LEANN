#![allow(unused_imports)]

use anyhow::Result;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::Json,
    routing::{delete, get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Clone)]
struct AppState {
    index_dir: PathBuf,
}

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    version: String,
}

#[derive(Serialize)]
struct IndexInfo {
    name: String,
    embedding_model: String,
    dimensions: usize,
    backend: String,
    total_passages: Option<usize>,
}

#[derive(Serialize)]
struct IndexListResponse {
    indexes: Vec<IndexInfo>,
}

#[derive(Deserialize)]
struct SearchRequest {
    query: String,
    #[serde(default = "default_top_k")]
    top_k: usize,
    #[serde(default)]
    complexity: Option<usize>,
    #[serde(default)]
    use_grep: Option<bool>,
}

fn default_top_k() -> usize {
    5
}

#[derive(Serialize)]
struct SearchResultResponse {
    id: String,
    score: f64,
    text: String,
    metadata: HashMap<String, serde_json::Value>,
}

#[derive(Serialize)]
struct SearchResponse {
    results: Vec<SearchResultResponse>,
    query: String,
    total: usize,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

async fn list_indexes(State(state): State<AppState>) -> Json<IndexListResponse> {
    use leann_core::index::IndexMeta;

    let mut indexes = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&state.index_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            let name = path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            if name.ends_with(".meta.json") {
                if let Ok(meta) = IndexMeta::load(&path) {
                    let index_name = name.strip_suffix(".meta.json").unwrap_or(&name).to_string();
                    indexes.push(IndexInfo {
                        name: index_name,
                        embedding_model: meta.embedding_model,
                        dimensions: meta.dimensions,
                        backend: meta.backend_name,
                        total_passages: meta.total_passages,
                    });
                }
            }
        }
    }

    Json(IndexListResponse { indexes })
}

async fn search_index(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(request): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, (StatusCode, Json<ErrorResponse>)> {
    use leann_core::searcher::{LeannSearcher, SearchConfig};

    let index_path = state.index_dir.join(&name);
    let searcher = LeannSearcher::open(&index_path).map_err(|e| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("Index '{}' not found: {}", name, e),
            }),
        )
    })?;

    let config = SearchConfig {
        complexity: request.complexity.unwrap_or(64),
        use_grep: request.use_grep.unwrap_or(false),
        ..Default::default()
    };

    let results = searcher
        .search_with_params(&request.query, request.top_k, &config)
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("Search failed: {}", e),
                }),
            )
        })?;

    let total = results.len();
    let response_results = results
        .into_iter()
        .map(|r| SearchResultResponse {
            id: r.id,
            score: r.score,
            text: r.text,
            metadata: r.metadata,
        })
        .collect();

    Ok(Json(SearchResponse {
        results: response_results,
        query: request.query,
        total,
    }))
}

async fn get_index_info(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<IndexInfo>, (StatusCode, Json<ErrorResponse>)> {
    use leann_core::index::IndexMeta;

    let meta_path = state.index_dir.join(format!("{}.meta.json", name));
    let meta = IndexMeta::load(&meta_path).map_err(|e| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("Index '{}' not found: {}", name, e),
            }),
        )
    })?;

    Ok(Json(IndexInfo {
        name,
        embedding_model: meta.embedding_model,
        dimensions: meta.dimensions,
        backend: meta.backend_name,
        total_passages: meta.total_passages,
    }))
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let index_dir = std::env::var("LEANN_INDEX_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."));

    let state = AppState { index_dir };

    let app = Router::new()
        .route("/health", get(health))
        .route("/indexes", get(list_indexes))
        .route("/indexes/{name}", get(get_index_info))
        .route("/indexes/{name}/search", post(search_index))
        .with_state(state);

    let port = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080u16);

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", port)).await?;
    tracing::info!("LEANN HTTP server listening on port {}", port);

    axum::serve(listener, app).await?;
    Ok(())
}
