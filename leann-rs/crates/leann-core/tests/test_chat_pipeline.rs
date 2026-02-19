//! E2E-2: Full RAG Pipeline — Build → Search → Chat
//!
//! Tests the LeannChat interface with SimulatedChat LLM.
//! Mirrors Python test_readme_examples.py.
//!
//! Note: Full vector search through LeannChat requires a ZMQ embedding server.
//! These tests verify the chat construction and SimulatedChat LLM independently.

use leann_core::chat::{get_llm, LlmConfig, LlmParams, LlmProvider, SimulatedChat};

/// SimulatedChat returns a fixed response.
#[test]
fn test_simulated_chat_response() {
    let chat = SimulatedChat;
    let response = chat.ask("What is LEANN?", &LlmParams::default()).unwrap();
    assert!(
        !response.is_empty(),
        "SimulatedChat should return non-empty response"
    );
    assert!(
        response.contains("simulated"),
        "Response should indicate simulation: '{}'",
        response
    );
}

/// get_llm factory creates SimulatedChat from config.
#[test]
fn test_get_llm_simulated() {
    let config = LlmConfig {
        llm_type: "simulated".to_string(),
        model: None,
        api_key: None,
        base_url: None,
        host: None,
    };
    let llm = get_llm(&config).unwrap();
    let response = llm.ask("test", &LlmParams::default()).unwrap();
    assert!(!response.is_empty());
}

/// get_llm factory rejects unknown types.
#[test]
fn test_get_llm_unknown_type() {
    let config = LlmConfig {
        llm_type: "unknown_provider".to_string(),
        model: None,
        api_key: None,
        base_url: None,
        host: None,
    };
    assert!(
        get_llm(&config).is_err(),
        "Should fail for unknown LLM type"
    );
}

/// LlmConfig default is OpenAI/gpt-4o.
#[test]
fn test_llm_config_default() {
    let config = LlmConfig::default();
    assert_eq!(config.llm_type, "openai");
    assert_eq!(config.model, Some("gpt-4o".to_string()));
}

/// LlmConfig serialization roundtrip.
#[test]
fn test_llm_config_serde() {
    let config = LlmConfig {
        llm_type: "simulated".to_string(),
        model: Some("test-model".to_string()),
        api_key: None,
        base_url: None,
        host: None,
    };
    let json = serde_json::to_string(&config).unwrap();
    let deserialized: LlmConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.llm_type, "simulated");
    assert_eq!(deserialized.model, Some("test-model".to_string()));
}
