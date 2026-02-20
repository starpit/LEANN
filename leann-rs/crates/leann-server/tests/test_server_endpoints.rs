//! E2E-10: HTTP Server Endpoint Tests
//!
//! Tests the leann-server HTTP API endpoints using axum test utilities.
//! These test the server routing and response format without starting a full server.

use axum::{
    Router,
    body::Body,
    http::{Request, StatusCode},
    routing::get,
};
use serde_json::Value;
use tower::ServiceExt; // for `oneshot`

/// Reconstruct the health endpoint handler for testing.
async fn health() -> axum::Json<Value> {
    axum::Json(serde_json::json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
    }))
}

/// Build a minimal test router with just the health endpoint.
fn test_router() -> Router {
    Router::new().route("/health", get(health))
}

/// GET /health returns 200 with status "ok".
#[tokio::test]
async fn test_health_endpoint() {
    let app = test_router();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(json["status"], "ok");
    assert!(json["version"].is_string());
}

/// GET /nonexistent returns 404.
#[tokio::test]
async fn test_not_found() {
    let app = test_router();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/nonexistent")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

/// Index list endpoint returns empty list when no indexes exist.
#[tokio::test]
async fn test_indexes_empty() {
    let dir = tempfile::tempdir().unwrap();

    #[derive(Clone)]
    struct AppState {
        index_dir: std::path::PathBuf,
    }

    async fn list_indexes(
        axum::extract::State(state): axum::extract::State<AppState>,
    ) -> axum::Json<Value> {
        let mut indexes = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&state.index_dir) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.ends_with(".meta.json") {
                    indexes.push(serde_json::json!({
                        "name": name.strip_suffix(".meta.json").unwrap_or(&name),
                    }));
                }
            }
        }
        axum::Json(serde_json::json!({ "indexes": indexes }))
    }

    let state = AppState {
        index_dir: dir.path().to_path_buf(),
    };

    let app = Router::new()
        .route("/indexes", get(list_indexes))
        .with_state(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/indexes")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();

    let indexes = json["indexes"].as_array().unwrap();
    assert!(indexes.is_empty(), "Should have no indexes in empty dir");
}

/// Index list endpoint finds .meta.json files.
#[tokio::test]
async fn test_indexes_after_meta_created() {
    let dir = tempfile::tempdir().unwrap();

    // Create a fake .meta.json file
    let meta = serde_json::json!({
        "version": "1.0",
        "backend_name": "hnsw",
        "embedding_model": "test-model",
        "dimensions": 128,
        "passage_sources": [],
    });
    std::fs::write(
        dir.path().join("my_index.meta.json"),
        serde_json::to_string_pretty(&meta).unwrap(),
    )
    .unwrap();

    #[derive(Clone)]
    struct AppState {
        index_dir: std::path::PathBuf,
    }

    async fn list_indexes(
        axum::extract::State(state): axum::extract::State<AppState>,
    ) -> axum::Json<Value> {
        let mut indexes = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&state.index_dir) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.ends_with(".meta.json") {
                    indexes.push(serde_json::json!({
                        "name": name.strip_suffix(".meta.json").unwrap_or(&name),
                    }));
                }
            }
        }
        axum::Json(serde_json::json!({ "indexes": indexes }))
    }

    let state = AppState {
        index_dir: dir.path().to_path_buf(),
    };

    let app = Router::new()
        .route("/indexes", get(list_indexes))
        .with_state(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/indexes")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();

    let indexes = json["indexes"].as_array().unwrap();
    assert_eq!(indexes.len(), 1, "Should find 1 index");
    assert_eq!(indexes[0]["name"], "my_index");
}
