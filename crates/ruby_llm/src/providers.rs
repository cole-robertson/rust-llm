//! Port of `lib/ruby_llm/provider.rb` and `lib/ruby_llm/providers/*.rb`: where each service
//! lives, how it authenticates, and which wire protocol it speaks for a model.

use crate::config::Config;
use crate::error::{Error, Result};
use crate::model::Model;

/// The providers this port supports, by RubyLLM slug.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Provider {
    OpenAI,
    Anthropic,
    Gemini,
    DeepSeek,
    Mistral,
    OpenRouter,
    XAI,
    Perplexity,
    Ollama,
    OllamaCloud,
    GPUStack,
    Hetzner,
}

/// The wire formats under `RubyLLM::Protocols`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProtocolName {
    ChatCompletions,
    Responses,
    Anthropic,
    Gemini,
}

impl ProtocolName {
    pub fn parse(name: &str) -> Result<ProtocolName> {
        match name {
            "chat_completions" => Ok(ProtocolName::ChatCompletions),
            "responses" => Ok(ProtocolName::Responses),
            "anthropic" => Ok(ProtocolName::Anthropic),
            "gemini" => Ok(ProtocolName::Gemini),
            other => Err(Error::Argument(format!("unknown protocol: {other}"))),
        }
    }
}

pub const ALL: &[Provider] = &[
    Provider::OpenAI,
    Provider::Anthropic,
    Provider::Gemini,
    Provider::DeepSeek,
    Provider::Mistral,
    Provider::OpenRouter,
    Provider::XAI,
    Provider::Perplexity,
    Provider::Ollama,
    Provider::OllamaCloud,
    Provider::GPUStack,
    Provider::Hetzner,
];

/// `Providers::OpenAI::Capabilities::SEARCH_MODELS`: these only work over Chat Completions.
const OPENAI_SEARCH_MODELS: &[&str] = &[
    "gpt-4o-mini-search-preview",
    "gpt-4o-mini-search-preview-2025-03-11",
    "gpt-4o-search-preview",
    "gpt-4o-search-preview-2025-03-11",
];

pub fn display_name(slug: &str) -> &str {
    Provider::resolve(slug).map(|p| p.display()).unwrap_or(slug)
}

impl Provider {
    pub fn resolve(slug: &str) -> Option<Provider> {
        ALL.iter().copied().find(|p| p.slug() == slug)
    }

    /// `Provider.resolve!`.
    pub fn resolve_or_err(slug: &str) -> Result<Provider> {
        Provider::resolve(slug).ok_or_else(|| {
            let known: Vec<&str> = ALL.iter().map(|p| p.slug()).collect();
            Error::Api(format!("Unknown provider: {slug:?}. Available providers: {}", known.join(", ")), None)
        })
    }

    pub fn slug(&self) -> &'static str {
        match self {
            Provider::OpenAI => "openai",
            Provider::Anthropic => "anthropic",
            Provider::Gemini => "gemini",
            Provider::DeepSeek => "deepseek",
            Provider::Mistral => "mistral",
            Provider::OpenRouter => "openrouter",
            Provider::XAI => "xai",
            Provider::Perplexity => "perplexity",
            Provider::Ollama => "ollama",
            Provider::OllamaCloud => "ollama_cloud",
            Provider::GPUStack => "gpustack",
            Provider::Hetzner => "hetzner",
        }
    }

    pub fn display(&self) -> &'static str {
        match self {
            Provider::OpenAI => "OpenAI",
            Provider::Anthropic => "Anthropic",
            Provider::Gemini => "Gemini",
            Provider::DeepSeek => "DeepSeek",
            Provider::Mistral => "Mistral",
            Provider::OpenRouter => "OpenRouter",
            Provider::XAI => "XAI",
            Provider::Perplexity => "Perplexity",
            Provider::Ollama => "Ollama",
            Provider::OllamaCloud => "OllamaCloud",
            Provider::GPUStack => "GPUStack",
            Provider::Hetzner => "Hetzner",
        }
    }

    /// `Provider.local?`: local providers skip the registry (models are assumed to exist).
    pub fn is_local(&self) -> bool {
        matches!(self, Provider::Ollama | Provider::GPUStack)
    }

    /// `Provider.assume_models_exist?`.
    pub fn assume_models_exist(&self) -> bool {
        self.is_local() || matches!(self, Provider::OllamaCloud | Provider::Hetzner)
    }

    /// `Provider.configuration_requirements`.
    pub fn configuration_requirements(&self) -> &'static [&'static str] {
        match self {
            Provider::OpenAI => &["openai_api_key"],
            Provider::Anthropic => &["anthropic_api_key"],
            Provider::Gemini => &["gemini_api_key"],
            Provider::DeepSeek => &["deepseek_api_key"],
            Provider::Mistral => &["mistral_api_key"],
            Provider::OpenRouter => &["openrouter_api_key"],
            Provider::XAI => &["xai_api_key"],
            Provider::Perplexity => &["perplexity_api_key"],
            Provider::Ollama => &["ollama_api_base"],
            Provider::OllamaCloud => &["ollama_cloud_api_key"],
            Provider::GPUStack => &["gpustack_api_base"],
            Provider::Hetzner => &["hetzner_api_key"],
        }
    }

    pub fn is_configured(&self, config: &Config) -> bool {
        self.configuration_requirements().iter().all(|r| config.get(r).is_some())
    }

    /// `Provider#ensure_configured!`.
    pub fn ensure_configured(&self, config: &Config) -> Result<()> {
        if self.is_configured(config) {
            return Ok(());
        }
        let lines: Vec<String> = self
            .configuration_requirements()
            .iter()
            .filter(|r| config.get(r).is_none())
            .map(|r| format!("    config.set(\"{r}\", std::env::var(\"{}\")?);", r.to_uppercase()))
            .collect();
        Err(Error::Configuration(format!(
            "{} provider is not configured. Add this to your initialization:\n\nruby_llm::configure(|config| {{\n{}\n}});",
            self.display(),
            lines.join("\n")
        )))
    }

    pub fn api_base(&self, config: &Config) -> Result<String> {
        let (key, default) = match self {
            Provider::OpenAI => ("openai_api_base", Some("https://api.openai.com/v1")),
            Provider::Anthropic => ("anthropic_api_base", Some("https://api.anthropic.com")),
            Provider::Gemini => ("gemini_api_base", Some("https://generativelanguage.googleapis.com/v1beta")),
            Provider::DeepSeek => ("deepseek_api_base", Some("https://api.deepseek.com")),
            Provider::Mistral => ("mistral_api_base", Some("https://api.mistral.ai/v1")),
            Provider::OpenRouter => ("openrouter_api_base", Some("https://openrouter.ai/api/v1")),
            Provider::XAI => ("xai_api_base", Some("https://api.x.ai/v1")),
            Provider::Perplexity => ("perplexity_api_base", Some("https://api.perplexity.ai")),
            Provider::Ollama => ("ollama_api_base", None),
            Provider::OllamaCloud => ("ollama_cloud_api_base", Some("https://ollama.com/v1")),
            Provider::GPUStack => ("gpustack_api_base", None),
            Provider::Hetzner => ("hetzner_api_base", Some("https://inference.hetzner.com/api/v1")),
        };
        config
            .get(key)
            .or(default)
            .map(str::to_string)
            .ok_or_else(|| Error::Configuration(format!("{key} is not set")))
    }

    /// `Provider#headers`.
    pub fn headers(&self, config: &Config) -> Vec<(String, String)> {
        let bearer = |key: &str| {
            config.get(key).map(|v| vec![("Authorization".to_string(), format!("Bearer {v}"))]).unwrap_or_default()
        };
        match self {
            Provider::OpenAI => {
                let mut h = bearer("openai_api_key");
                if let Some(org) = config.get("openai_organization_id") {
                    h.push(("OpenAI-Organization".into(), org.into()));
                }
                if let Some(project) = config.get("openai_project_id") {
                    h.push(("OpenAI-Project".into(), project.into()));
                }
                h
            }
            Provider::Anthropic => vec![
                ("x-api-key".into(), config.get("anthropic_api_key").unwrap_or_default().into()),
                ("anthropic-version".into(), "2023-06-01".into()),
            ],
            Provider::Gemini => {
                vec![("x-goog-api-key".into(), config.get("gemini_api_key").unwrap_or_default().into())]
            }
            Provider::DeepSeek => bearer("deepseek_api_key"),
            Provider::Mistral => bearer("mistral_api_key"),
            Provider::OpenRouter => {
                let mut h = bearer("openrouter_api_key");
                h.push(("HTTP-Referer".into(), config.get("openrouter_app_url").unwrap_or("https://rubyllm.com").into()));
                h.push(("X-OpenRouter-Title".into(), config.get("openrouter_app_name").unwrap_or("RubyLLM").into()));
                h
            }
            Provider::XAI => bearer("xai_api_key"),
            Provider::Perplexity => bearer("perplexity_api_key"),
            Provider::Ollama => bearer("ollama_api_key"),
            Provider::OllamaCloud => bearer("ollama_cloud_api_key"),
            Provider::GPUStack => bearer("gpustack_api_key"),
            Provider::Hetzner => bearer("hetzner_api_key"),
        }
    }

    /// The first `protocol` a provider registers is its default.
    pub fn default_protocol(&self) -> ProtocolName {
        match self {
            Provider::OpenAI | Provider::XAI => ProtocolName::Responses,
            Provider::Anthropic => ProtocolName::Anthropic,
            Provider::Gemini => ProtocolName::Gemini,
            Provider::Perplexity => ProtocolName::Responses,
            _ => ProtocolName::ChatCompletions,
        }
    }

    pub fn supports_protocol(&self, protocol: ProtocolName) -> bool {
        match self {
            Provider::OpenAI | Provider::XAI | Provider::DeepSeek | Provider::OpenRouter | Provider::GPUStack => {
                matches!(protocol, ProtocolName::ChatCompletions | ProtocolName::Responses)
            }
            Provider::Anthropic => protocol == ProtocolName::Anthropic,
            Provider::Gemini => protocol == ProtocolName::Gemini,
            Provider::Perplexity => matches!(protocol, ProtocolName::ChatCompletions | ProtocolName::Responses),
            _ => protocol == ProtocolName::ChatCompletions,
        }
    }

    /// `Provider#protocol_for` plus `resolve_protocol`: an explicit protocol wins, then the
    /// `<slug>_protocol` config option, then the provider's per-model rule.
    pub fn resolve_protocol(&self, explicit: Option<ProtocolName>, model: &Model, config: &Config) -> Result<ProtocolName> {
        let configured = config.get(&format!("{}_protocol", self.slug())).map(ProtocolName::parse).transpose()?;
        let protocol = match explicit.or(configured) {
            Some(p) => p,
            None => match self {
                Provider::OpenAI
                    if OPENAI_SEARCH_MODELS.contains(&model.id.as_str())
                        || model.id.contains("audio")
                        || model.id.contains("realtime") =>
                {
                    ProtocolName::ChatCompletions
                }
                _ => self.default_protocol(),
            },
        };
        if !self.supports_protocol(protocol) {
            return Err(Error::Api(format!("{protocol:?} is not a protocol of {}", self.display()), None));
        }
        Ok(protocol)
    }

    /// `Provider#parse_error` overrides are folded into `error::parse_error_message`, except
    /// Perplexity's HTML error pages.
    pub(crate) fn strip_html_error(&self, body: &str) -> Option<String> {
        if *self != Provider::Perplexity || !body.contains("<html>") {
            return None;
        }
        let start = body.find("<title>")? + 7;
        let end = body[start..].find("</title>")? + start;
        let title = &body[start..end];
        Some(title.trim_start_matches(|c: char| c.is_ascii_digit()).trim_start().to_string())
    }
}
