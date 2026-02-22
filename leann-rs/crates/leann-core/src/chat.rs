use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::info;

use crate::searcher::{LeannSearcher, SearcherOptions};
use crate::settings;

/// Trait for LLM chat backends.
pub trait LlmProvider: Send + Sync {
    /// Send a prompt and get a response.
    fn ask(&self, prompt: &str, params: &LlmParams) -> Result<String>;
}

/// Parameters for LLM generation.
#[derive(Debug, Clone, Default)]
pub struct LlmParams {
    pub temperature: Option<f64>,
    pub max_tokens: Option<usize>,
    pub top_p: Option<f64>,
    pub extra: HashMap<String, serde_json::Value>,
}

/// Configuration for creating an LLM provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    #[serde(rename = "type")]
    pub llm_type: String,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(default)]
    pub host: Option<String>,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            llm_type: "openai".to_string(),
            model: Some("gpt-4o".to_string()),
            api_key: None,
            base_url: None,
            host: None,
        }
    }
}

/// Ollama LLM chat backend.
pub struct OllamaChat {
    model: String,
    host: String,
    client: reqwest::blocking::Client,
}

impl OllamaChat {
    pub fn new(model: &str, host: Option<&str>) -> Self {
        Self {
            model: model.to_string(),
            host: settings::resolve_ollama_host(host),
            client: reqwest::blocking::Client::new(),
        }
    }
}

impl LlmProvider for OllamaChat {
    fn ask(&self, prompt: &str, _params: &LlmParams) -> Result<String> {
        let payload = serde_json::json!({
            "model": self.model,
            "prompt": prompt,
            "stream": false,
        });

        let response = self
            .client
            .post(format!("{}/api/generate", self.host))
            .json(&payload)
            .send()?;

        let body: serde_json::Value = response.json()?;
        Ok(body["response"].as_str().unwrap_or("").to_string())
    }
}

/// OpenAI LLM chat backend.
pub struct OpenAiChat {
    model: String,
    api_key: String,
    base_url: String,
    client: reqwest::blocking::Client,
}

impl OpenAiChat {
    pub fn new(model: &str, api_key: Option<&str>, base_url: Option<&str>) -> Result<Self> {
        let api_key = settings::resolve_openai_api_key(api_key)
            .ok_or_else(|| anyhow::anyhow!("OpenAI API key required"))?;
        let base_url = settings::resolve_openai_base_url(base_url);

        Ok(Self {
            model: model.to_string(),
            api_key,
            base_url,
            client: reqwest::blocking::Client::new(),
        })
    }
}

impl LlmProvider for OpenAiChat {
    fn ask(&self, prompt: &str, params: &LlmParams) -> Result<String> {
        let mut payload = serde_json::json!({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": params.temperature.unwrap_or(0.7),
        });

        if let Some(max_tokens) = params.max_tokens {
            payload["max_tokens"] = serde_json::json!(max_tokens);
        }

        let response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&payload)
            .send()?;

        let body: serde_json::Value = response.json()?;
        Ok(body["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .trim()
            .to_string())
    }
}

/// Anthropic LLM chat backend.
pub struct AnthropicChat {
    model: String,
    api_key: String,
    base_url: String,
    client: reqwest::blocking::Client,
}

impl AnthropicChat {
    pub fn new(model: &str, api_key: Option<&str>, base_url: Option<&str>) -> Result<Self> {
        let api_key = settings::resolve_anthropic_api_key(api_key)
            .ok_or_else(|| anyhow::anyhow!("Anthropic API key required"))?;
        let base_url = settings::resolve_anthropic_base_url(base_url);

        Ok(Self {
            model: model.to_string(),
            api_key,
            base_url,
            client: reqwest::blocking::Client::new(),
        })
    }
}

impl LlmProvider for AnthropicChat {
    fn ask(&self, prompt: &str, params: &LlmParams) -> Result<String> {
        let mut payload = serde_json::json!({
            "model": self.model,
            "max_tokens": params.max_tokens.unwrap_or(1000),
            "messages": [{"role": "user", "content": prompt}],
        });

        if let Some(temp) = params.temperature {
            payload["temperature"] = serde_json::json!(temp);
        }

        let response = self
            .client
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&payload)
            .send()?;

        let body: serde_json::Value = response.json()?;
        Ok(body["content"][0]["text"]
            .as_str()
            .unwrap_or("")
            .trim()
            .to_string())
    }
}

/// Google Gemini LLM chat backend.
pub struct GeminiChat {
    model: String,
    api_key: String,
    client: reqwest::blocking::Client,
}

impl GeminiChat {
    pub fn new(model: &str, api_key: Option<&str>) -> Result<Self> {
        let api_key = settings::resolve_gemini_api_key(api_key)
            .ok_or_else(|| anyhow::anyhow!("Gemini API key required. Set GEMINI_API_KEY environment variable or pass api_key parameter."))?;

        Ok(Self {
            model: model.to_string(),
            api_key,
            client: reqwest::blocking::Client::new(),
        })
    }
}

impl LlmProvider for GeminiChat {
    fn ask(&self, prompt: &str, params: &LlmParams) -> Result<String> {
        let mut generation_config = serde_json::json!({
            "temperature": params.temperature.unwrap_or(0.7),
            "maxOutputTokens": params.max_tokens.unwrap_or(1000),
        });

        if let Some(top_p) = params.top_p {
            generation_config["topP"] = serde_json::json!(top_p);
        }

        let payload = serde_json::json!({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": generation_config,
        });

        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
            self.model, self.api_key
        );

        let response = self.client.post(&url).json(&payload).send()?;

        let body: serde_json::Value = response.json()?;
        Ok(body["candidates"][0]["content"]["parts"][0]["text"]
            .as_str()
            .unwrap_or("")
            .trim()
            .to_string())
    }
}

/// Simulated LLM for testing.
pub struct SimulatedChat;

impl LlmProvider for SimulatedChat {
    fn ask(&self, _prompt: &str, _params: &LlmParams) -> Result<String> {
        Ok("This is a simulated answer from the LLM based on the retrieved context.".to_string())
    }
}

/// Factory function to create an LLM provider from configuration.
pub fn get_llm(config: &LlmConfig) -> Result<Box<dyn LlmProvider>> {
    match config.llm_type.as_str() {
        "ollama" => Ok(Box::new(OllamaChat::new(
            config.model.as_deref().unwrap_or("llama3:8b"),
            config.host.as_deref(),
        ))),
        "openai" => Ok(Box::new(OpenAiChat::new(
            config.model.as_deref().unwrap_or("gpt-4o"),
            config.api_key.as_deref(),
            config.base_url.as_deref(),
        )?)),
        "anthropic" => Ok(Box::new(AnthropicChat::new(
            config
                .model
                .as_deref()
                .unwrap_or("claude-3-5-sonnet-20241022"),
            config.api_key.as_deref(),
            config.base_url.as_deref(),
        )?)),
        "gemini" => Ok(Box::new(GeminiChat::new(
            config.model.as_deref().unwrap_or("gemini-2.5-flash"),
            config.api_key.as_deref(),
        )?)),
        "simulated" => Ok(Box::new(SimulatedChat)),
        other => anyhow::bail!("Unknown LLM type: {}", other),
    }
}

/// High-level RAG chat interface combining search + LLM.
pub struct LeannChat {
    searcher: LeannSearcher,
    llm: Box<dyn LlmProvider>,
    #[allow(dead_code)]
    owns_searcher: bool,
}

impl LeannChat {
    pub fn new(searcher: LeannSearcher, llm_config: Option<&LlmConfig>) -> Result<Self> {
        let config = llm_config.cloned().unwrap_or_default();
        let llm = get_llm(&config)?;

        Ok(Self {
            searcher,
            llm,
            owns_searcher: true,
        })
    }

    /// Create a new chat from an index path with custom searcher options.
    pub fn new_with_options(
        index_path: &std::path::Path,
        llm_config: Option<&LlmConfig>,
        searcher_options: &SearcherOptions,
    ) -> Result<Self> {
        let searcher = LeannSearcher::open_with_options(index_path, searcher_options)?;
        Self::new(searcher, llm_config)
    }

    /// Ask a question using RAG (retrieve context, then generate answer).
    pub fn ask(&self, question: &str, top_k: usize) -> Result<String> {
        self.ask_with_params(question, top_k, &crate::searcher::SearchConfig::default())
    }

    /// Ask a question using RAG with full search configuration.
    pub fn ask_with_params(
        &self,
        question: &str,
        top_k: usize,
        config: &crate::searcher::SearchConfig,
    ) -> Result<String> {
        let results = self.searcher.search_with_params(question, top_k, config)?;

        let context: String = results
            .iter()
            .map(|r| r.text.as_str())
            .collect::<Vec<_>>()
            .join("\n\n");

        let prompt = format!(
            "Here is some retrieved context that might help answer your question:\n\n\
             {}\n\n\
             Question: {}\n\n\
             Please provide the best answer you can based on this context and your knowledge.",
            context, question
        );

        info!(
            "Sending RAG prompt to LLM ({} context results)",
            results.len()
        );
        let answer = self.llm.ask(&prompt, &LlmParams::default())?;
        Ok(answer)
    }

    pub fn cleanup(&mut self) {
        // Cleanup will be handled by Drop
    }
}
