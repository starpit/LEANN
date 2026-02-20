use std::env;

const DEFAULT_OLLAMA_HOST: &str = "http://localhost:11434";
const DEFAULT_OPENAI_BASE_URL: &str = "https://api.openai.com/v1";
const DEFAULT_ANTHROPIC_BASE_URL: &str = "https://api.anthropic.com";

fn clean_url(value: &str) -> String {
    value.trim_end_matches('/').to_string()
}

/// Resolve the Ollama-compatible endpoint to use.
pub fn resolve_ollama_host(explicit: Option<&str>) -> String {
    let candidates = [
        explicit.map(String::from),
        env::var("LEANN_LOCAL_LLM_HOST").ok(),
        env::var("LEANN_OLLAMA_HOST").ok(),
        env::var("OLLAMA_HOST").ok(),
        env::var("LOCAL_LLM_ENDPOINT").ok(),
    ];

    for candidate in &candidates {
        if let Some(val) = candidate {
            if !val.is_empty() {
                return clean_url(val);
            }
        }
    }

    clean_url(DEFAULT_OLLAMA_HOST)
}

/// Resolve the base URL for OpenAI-compatible services.
pub fn resolve_openai_base_url(explicit: Option<&str>) -> String {
    let candidates = [
        explicit.map(String::from),
        env::var("LEANN_OPENAI_BASE_URL").ok(),
        env::var("OPENAI_BASE_URL").ok(),
        env::var("LOCAL_OPENAI_BASE_URL").ok(),
    ];

    for candidate in &candidates {
        if let Some(val) = candidate {
            if !val.is_empty() {
                return clean_url(val);
            }
        }
    }

    clean_url(DEFAULT_OPENAI_BASE_URL)
}

/// Resolve the base URL for Anthropic-compatible services.
pub fn resolve_anthropic_base_url(explicit: Option<&str>) -> String {
    let candidates = [
        explicit.map(String::from),
        env::var("LEANN_ANTHROPIC_BASE_URL").ok(),
        env::var("ANTHROPIC_BASE_URL").ok(),
        env::var("LOCAL_ANTHROPIC_BASE_URL").ok(),
    ];

    for candidate in &candidates {
        if let Some(val) = candidate {
            if !val.is_empty() {
                return clean_url(val);
            }
        }
    }

    clean_url(DEFAULT_ANTHROPIC_BASE_URL)
}

/// Resolve the API key for OpenAI-compatible services.
pub fn resolve_openai_api_key(explicit: Option<&str>) -> Option<String> {
    if let Some(key) = explicit {
        if !key.is_empty() {
            return Some(key.to_string());
        }
    }
    env::var("OPENAI_API_KEY").ok()
}

/// Resolve the API key for Anthropic services.
pub fn resolve_anthropic_api_key(explicit: Option<&str>) -> Option<String> {
    if let Some(key) = explicit {
        if !key.is_empty() {
            return Some(key.to_string());
        }
    }
    env::var("ANTHROPIC_API_KEY").ok()
}

/// Resolve the API key for Gemini services.
pub fn resolve_gemini_api_key(explicit: Option<&str>) -> Option<String> {
    if let Some(key) = explicit {
        if !key.is_empty() {
            return Some(key.to_string());
        }
    }
    env::var("GEMINI_API_KEY").ok()
}

/// Serialize provider options for child processes.
pub fn encode_provider_options(
    options: Option<&std::collections::HashMap<String, serde_json::Value>>,
) -> Option<String> {
    options.and_then(|opts| {
        if opts.is_empty() {
            None
        } else {
            serde_json::to_string(opts).ok()
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_ollama_host_explicit() {
        assert_eq!(
            resolve_ollama_host(Some("http://my-host:1234/")),
            "http://my-host:1234"
        );
    }

    #[test]
    fn test_resolve_ollama_host_default() {
        // Clear env vars to ensure default
        // SAFETY: This test is not run concurrently with other env-modifying tests
        unsafe {
            env::remove_var("LEANN_LOCAL_LLM_HOST");
            env::remove_var("LEANN_OLLAMA_HOST");
            env::remove_var("OLLAMA_HOST");
            env::remove_var("LOCAL_LLM_ENDPOINT");
        }
        assert_eq!(resolve_ollama_host(None), "http://localhost:11434");
    }

    #[test]
    fn test_resolve_openai_api_key_explicit() {
        assert_eq!(
            resolve_openai_api_key(Some("sk-test123")),
            Some("sk-test123".to_string())
        );
    }

    #[test]
    fn test_clean_url_strips_trailing_slash() {
        assert_eq!(clean_url("http://example.com/"), "http://example.com");
        assert_eq!(clean_url("http://example.com"), "http://example.com");
    }
}
