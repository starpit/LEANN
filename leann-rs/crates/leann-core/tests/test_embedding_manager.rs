//! E2E-7: Embedding Server Manager Lifecycle Tests
//!
//! Tests the EmbeddingServerManager without requiring a running embedding server.
//! Validates manager construction, port allocation, lifecycle state, and graceful
//! stop behavior. Tests that require an actual server binary are marked #[ignore].

use leann_core::embedding::manager::EmbeddingServerManager;

/// New manager starts with no process and no port.
#[test]
fn test_manager_new_state() {
    let manager = EmbeddingServerManager::new();
    assert_eq!(manager.port(), None, "Fresh manager should have no port");
}

/// is_alive() returns false for a fresh manager with no server.
#[test]
fn test_manager_is_alive_fresh() {
    let mut manager = EmbeddingServerManager::new();
    assert!(!manager.is_alive(), "Fresh manager should not be alive");
}

/// stop_server on a fresh manager is a no-op (doesn't panic).
#[test]
fn test_manager_stop_noop() {
    let mut manager = EmbeddingServerManager::new();
    manager.stop_server();
    assert_eq!(manager.port(), None);
    assert!(!manager.is_alive());
}

/// Double stop is safe.
#[test]
fn test_manager_double_stop() {
    let mut manager = EmbeddingServerManager::new();
    manager.stop_server();
    manager.stop_server();
    assert_eq!(manager.port(), None);
}

/// Default trait works identically to new().
#[test]
fn test_manager_default() {
    let manager = EmbeddingServerManager::default();
    assert_eq!(manager.port(), None);
}
