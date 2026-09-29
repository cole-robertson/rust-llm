//! Port of `lib/ruby_llm/configuration.rb`.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, RwLock};
use std::time::Duration;

/// `RubyLLM::Configuration`. Provider keys live in `values` under RubyLLM's option names
/// (`openai_api_key`, `anthropic_api_base`, ...) so every provider reads its settings the same way.
#[derive(Debug, Clone)]
pub struct Config {
    pub default_model: String,
    pub default_embedding_model: String,
    /// `default_image_model`: the model `rust_llm::paint` uses when none is given.
    pub default_image_model: String,
    /// `default_judgment_model`: what `judge` uses; judges never fall back to the chat model.
    pub default_judgment_model: String,
    pub request_timeout: Duration,
    pub max_retries: u32,
    pub retry_interval: f64,
    pub retry_backoff_factor: f64,
    pub retry_interval_randomness: f64,
    pub retry_max_interval: f64,
    pub tool_concurrency: bool,
    /// `auto_upload_large_files`: upload oversized local attachments to the provider's Files API.
    pub auto_upload_large_files: bool,
    values: HashMap<String, String>,
}

/// Provider options RubyLLM reads, and the environment variable each falls back to.
const PROVIDER_OPTIONS: &[&str] = &[
    "openai_api_key", "openai_api_base", "openai_organization_id", "openai_project_id", "openai_use_system_role",
    "anthropic_api_key", "anthropic_api_base",
    "gemini_api_key", "gemini_api_base",
    "deepseek_api_key", "deepseek_api_base",
    "mistral_api_key", "mistral_api_base",
    "openrouter_api_key", "openrouter_api_base", "openrouter_app_url", "openrouter_app_name",
    "xai_api_key", "xai_api_base",
    "perplexity_api_key", "perplexity_api_base",
    "ollama_api_base", "ollama_api_key",
    "ollama_cloud_api_key", "ollama_cloud_api_base",
    "gpustack_api_base", "gpustack_api_key",
    "hetzner_api_key", "hetzner_api_base",
    "typesafe_api_key", "typesafe_api_base",
];

impl Default for Config {
    fn default() -> Self {
        Config {
            default_model: "gpt-5.6".into(),
            default_embedding_model: "text-embedding-3-small".into(),
            default_image_model: "gpt-image-2".into(),
            default_judgment_model: "jev-latest".into(),
            request_timeout: Duration::from_secs(300),
            max_retries: 3,
            retry_interval: 0.1,
            retry_backoff_factor: 2.0,
            retry_interval_randomness: 0.5,
            retry_max_interval: 30.0,
            tool_concurrency: false,
            auto_upload_large_files: true,
            values: HashMap::new(),
        }
    }
}

impl Config {
    /// Defaults plus any `OPENAI_API_KEY`-style variables present in the environment.
    pub fn from_env() -> Config {
        let mut config = Config::default();
        for option in PROVIDER_OPTIONS {
            if let Ok(value) = std::env::var(option.to_uppercase())
                && !value.is_empty() {
                    config.values.insert(option.to_string(), value);
                }
        }
        if let Ok(model) = std::env::var("RUST_LLM_DEFAULT_MODEL") {
            config.default_model = model;
        }
        config
    }

    pub fn get(&self, option: &str) -> Option<&str> {
        self.values.get(option).map(String::as_str).filter(|v| !v.is_empty())
    }

    pub fn set(&mut self, option: impl Into<String>, value: impl Into<String>) -> &mut Self {
        self.values.insert(option.into(), value.into());
        self
    }

    pub fn openai_api_key(&mut self, v: impl Into<String>) -> &mut Self {
        self.set("openai_api_key", v)
    }
    pub fn anthropic_api_key(&mut self, v: impl Into<String>) -> &mut Self {
        self.set("anthropic_api_key", v)
    }
    pub fn gemini_api_key(&mut self, v: impl Into<String>) -> &mut Self {
        self.set("gemini_api_key", v)
    }
    pub fn deepseek_api_key(&mut self, v: impl Into<String>) -> &mut Self {
        self.set("deepseek_api_key", v)
    }
    pub fn openrouter_api_key(&mut self, v: impl Into<String>) -> &mut Self {
        self.set("openrouter_api_key", v)
    }
    pub fn ollama_api_base(&mut self, v: impl Into<String>) -> &mut Self {
        self.set("ollama_api_base", v)
    }
}

static CONFIG: LazyLock<RwLock<Arc<Config>>> = LazyLock::new(|| RwLock::new(Arc::new(Config::from_env())));

/// `RubyLLM.configure { |config| ... }`.
pub fn configure(f: impl FnOnce(&mut Config)) {
    let mut guard = CONFIG.write().unwrap();
    let mut config = (**guard).clone();
    f(&mut config);
    *guard = Arc::new(config);
}

/// `RubyLLM.config`.
pub fn config() -> Arc<Config> {
    CONFIG.read().unwrap().clone()
}
