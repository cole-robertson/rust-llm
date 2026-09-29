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
    /// TypeSafe's System One API (Jev judgment models).
    TypeSafe,
}

/// The wire formats under `RubyLLM::Protocols`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProtocolName {
    ChatCompletions,
    Responses,
    Anthropic,
    Gemini,
    /// `Protocols::Interactions`: Gemini's Interactions API, opted into with `protocol: :interactions`.
    Interactions,
    /// `Providers::Mistral::Conversations`: Mistral's Conversations API (`protocol: :conversations`).
    Conversations,
    /// `Protocols::Perplexity::Router`: Perplexity Router (`protocol: :router_chat_completions`).
    RouterChatCompletions,
}

impl ProtocolName {
    pub fn parse(name: &str) -> Result<ProtocolName> {
        match name {
            "chat_completions" => Ok(ProtocolName::ChatCompletions),
            "responses" => Ok(ProtocolName::Responses),
            "anthropic" => Ok(ProtocolName::Anthropic),
            "gemini" => Ok(ProtocolName::Gemini),
            "interactions" => Ok(ProtocolName::Interactions),
            "conversations" => Ok(ProtocolName::Conversations),
            "router_chat_completions" => Ok(ProtocolName::RouterChatCompletions),
            other => Err(Error::Argument(format!("unknown protocol: {other}"))),
        }
    }

    /// The name a provider registers the protocol under (`protocol :chat_completions, ...`).
    pub fn name(&self) -> &'static str {
        match self {
            ProtocolName::ChatCompletions => "chat_completions",
            ProtocolName::Responses => "responses",
            ProtocolName::Anthropic => "anthropic",
            ProtocolName::Gemini => "gemini",
            ProtocolName::Interactions => "interactions",
            ProtocolName::Conversations => "conversations",
            ProtocolName::RouterChatCompletions => "router_chat_completions",
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
    Provider::TypeSafe,
];

/// `Providers::OpenAI::Capabilities::SEARCH_MODELS`: these only work over Chat Completions.
const OPENAI_SEARCH_MODELS: &[&str] = &[
    "gpt-4o-mini-search-preview",
    "gpt-4o-mini-search-preview-2025-03-11",
    "gpt-4o-search-preview",
    "gpt-4o-search-preview-2025-03-11",
    "gpt-5-search-api",
    "gpt-5-search-api-2025-10-14",
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
            Error::Api(
                format!(
                    "Unknown provider: {slug:?}. Available providers: {}",
                    known.join(", ")
                ),
                None,
            )
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
            Provider::TypeSafe => "typesafe",
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
            Provider::TypeSafe => "TypeSafe",
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
            Provider::TypeSafe => &["typesafe_api_key"],
        }
    }

    pub fn is_configured(&self, config: &Config) -> bool {
        self.configuration_requirements()
            .iter()
            .all(|r| config.get(r).is_some())
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
            .map(|r| {
                format!(
                    "    config.set(\"{r}\", std::env::var(\"{}\")?);",
                    r.to_uppercase()
                )
            })
            .collect();
        Err(Error::Configuration(format!(
            "{} provider is not configured. Add this to your initialization:\n\nrust_llm::configure(|config| {{\n{}\n}});",
            self.display(),
            lines.join("\n")
        )))
    }

    pub fn api_base(&self, config: &Config) -> Result<String> {
        let (key, default) = match self {
            Provider::OpenAI => ("openai_api_base", Some("https://api.openai.com/v1")),
            Provider::Anthropic => ("anthropic_api_base", Some("https://api.anthropic.com")),
            Provider::Gemini => (
                "gemini_api_base",
                Some("https://generativelanguage.googleapis.com/v1beta"),
            ),
            Provider::DeepSeek => ("deepseek_api_base", Some("https://api.deepseek.com")),
            Provider::Mistral => ("mistral_api_base", Some("https://api.mistral.ai/v1")),
            Provider::OpenRouter => ("openrouter_api_base", Some("https://openrouter.ai/api/v1")),
            Provider::XAI => ("xai_api_base", Some("https://api.x.ai/v1")),
            Provider::Perplexity => ("perplexity_api_base", Some("https://api.perplexity.ai")),
            Provider::Ollama => ("ollama_api_base", None),
            Provider::OllamaCloud => ("ollama_cloud_api_base", Some("https://ollama.com/v1")),
            Provider::GPUStack => ("gpustack_api_base", None),
            Provider::Hetzner => (
                "hetzner_api_base",
                Some("https://inference.hetzner.com/api/v1"),
            ),
            Provider::TypeSafe => ("typesafe_api_base", Some("https://api.typesafe.ai")),
        };
        config
            .get(key)
            .or(default)
            .map(str::to_string)
            .ok_or_else(|| Error::Configuration(format!("{key} is not set")))
    }

    /// `Providers::Perplexity#agent_url`: the Agent endpoint under the configured base, which may
    /// be a gateway path with or without its own `/v1`.
    pub fn agent_url(&self, config: &Config) -> Result<String> {
        let base = self.api_base(config)?;
        let base = base.strip_suffix('/').unwrap_or(&base);
        Ok(format!(
            "{}/v1/agent",
            base.strip_suffix("/v1").unwrap_or(base)
        ))
    }

    /// `Provider#headers`.
    pub fn headers(&self, config: &Config) -> Vec<(String, String)> {
        let bearer = |key: &str| {
            config
                .get(key)
                .map(|v| vec![("Authorization".to_string(), format!("Bearer {v}"))])
                .unwrap_or_default()
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
                (
                    "x-api-key".into(),
                    config.get("anthropic_api_key").unwrap_or_default().into(),
                ),
                ("anthropic-version".into(), "2023-06-01".into()),
            ],
            Provider::Gemini => {
                vec![(
                    "x-goog-api-key".into(),
                    config.get("gemini_api_key").unwrap_or_default().into(),
                )]
            }
            Provider::DeepSeek => bearer("deepseek_api_key"),
            Provider::Mistral => bearer("mistral_api_key"),
            Provider::OpenRouter => {
                let mut h = bearer("openrouter_api_key");
                h.push((
                    "HTTP-Referer".into(),
                    config
                        .get("openrouter_app_url")
                        .unwrap_or("https://github.com/cole-robertson/rust_llm")
                        .into(),
                ));
                h.push((
                    "X-OpenRouter-Title".into(),
                    config
                        .get("openrouter_app_name")
                        .unwrap_or("RustLLM")
                        .into(),
                ));
                h
            }
            Provider::XAI => bearer("xai_api_key"),
            Provider::Perplexity => bearer("perplexity_api_key"),
            Provider::Ollama => bearer("ollama_api_key"),
            Provider::OllamaCloud => bearer("ollama_cloud_api_key"),
            Provider::GPUStack => bearer("gpustack_api_key"),
            Provider::Hetzner => bearer("hetzner_api_key"),
            Provider::TypeSafe => bearer("typesafe_api_key"),
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
            Provider::OpenAI
            | Provider::XAI
            | Provider::DeepSeek
            | Provider::OpenRouter
            | Provider::GPUStack => {
                matches!(
                    protocol,
                    ProtocolName::ChatCompletions | ProtocolName::Responses
                )
            }
            Provider::Anthropic => protocol == ProtocolName::Anthropic,
            Provider::Gemini => {
                matches!(protocol, ProtocolName::Gemini | ProtocolName::Interactions)
            }
            Provider::Mistral => matches!(
                protocol,
                ProtocolName::ChatCompletions | ProtocolName::Conversations
            ),
            Provider::Perplexity => {
                matches!(
                    protocol,
                    ProtocolName::ChatCompletions
                        | ProtocolName::Responses
                        | ProtocolName::RouterChatCompletions
                )
            }
            // System One answers judgments only (`rust_llm::judge`); it has no chat protocol.
            Provider::TypeSafe => false,
            _ => protocol == ProtocolName::ChatCompletions,
        }
    }

    /// `Provider#protocol_for` plus `resolve_protocol`: an explicit protocol wins, then the
    /// `<slug>_protocol` config option, then the provider's per-model rule.
    pub fn resolve_protocol(
        &self,
        explicit: Option<ProtocolName>,
        model: &Model,
        config: &Config,
    ) -> Result<ProtocolName> {
        let configured = match config.get(&format!("{}_protocol", self.slug())) {
            Some(name) => Some(ProtocolName::parse(name).map_err(|_| self.not_a_protocol(name))?),
            None => None,
        };
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
        if *self == Provider::TypeSafe {
            return Err(Error::Api(
                format!("{} doesn't support chat", self.display()),
                None,
            ));
        }
        if !self.supports_protocol(protocol) {
            return Err(self.not_a_protocol(protocol.name()));
        }
        Ok(protocol)
    }

    /// `Provider.protocols.keys`: every protocol the Ruby provider registers, in order.
    pub fn protocol_names(&self) -> &'static [&'static str] {
        match self {
            Provider::OpenAI => &["responses", "chat_completions", "embeddings", "files"],
            Provider::Anthropic => &["anthropic", "files"],
            Provider::Gemini => &["gemini", "interactions", "live_transcription", "files"],
            Provider::DeepSeek => &["chat_completions", "responses", "files"],
            Provider::Mistral => &["chat_completions", "conversations", "files"],
            Provider::OpenRouter => &["chat_completions", "responses", "files"],
            Provider::XAI => &["responses", "chat_completions", "files"],
            Provider::Perplexity => &[
                "chat_completions",
                "router_chat_completions",
                "files",
                "agent_responses",
            ],
            Provider::GPUStack => &["chat_completions", "responses"],
            Provider::Ollama | Provider::OllamaCloud | Provider::Hetzner => &["chat_completions"],
            Provider::TypeSafe => &["system_one"],
        }
    }

    /// `Provider#fetch_protocol`'s error for a protocol the provider does not register.
    fn not_a_protocol(&self, name: &str) -> Error {
        Error::Api(
            format!(
                "{name} is not a protocol of {}. Available: {}",
                self.display(),
                self.protocol_names().join(", ")
            ),
            None,
        )
    }

    /// `Provider#configuration_options`: every option the provider reads.
    pub fn configuration_options(&self) -> &'static [&'static str] {
        match self {
            Provider::OpenAI => &[
                "openai_api_key",
                "openai_api_base",
                "openai_organization_id",
                "openai_project_id",
                "openai_use_system_role",
            ],
            Provider::Anthropic => &["anthropic_api_key", "anthropic_api_base"],
            Provider::Gemini => &["gemini_api_key", "gemini_api_base"],
            Provider::DeepSeek => &["deepseek_api_key", "deepseek_api_base"],
            Provider::Mistral => &["mistral_api_key", "mistral_api_base"],
            Provider::OpenRouter => &[
                "openrouter_api_key",
                "openrouter_api_base",
                "openrouter_app_url",
                "openrouter_app_name",
            ],
            Provider::XAI => &["xai_api_key", "xai_api_base"],
            Provider::Perplexity => &["perplexity_api_key", "perplexity_api_base"],
            Provider::Ollama => &["ollama_api_base", "ollama_api_key"],
            Provider::OllamaCloud => &["ollama_cloud_api_key", "ollama_cloud_api_base"],
            Provider::GPUStack => &["gpustack_api_base", "gpustack_api_key"],
            Provider::Hetzner => &["hetzner_api_key", "hetzner_api_base"],
            Provider::TypeSafe => &["typesafe_api_key", "typesafe_api_base"],
        }
    }

    /// `Provider.remote?`.
    pub fn is_remote(&self) -> bool {
        !self.is_local()
    }

    /// `Provider#retry_delay`: seconds a rate-limited response asked us to wait, read from
    /// provider-specific headers. Only OpenAI overrides the base `nil`: it reports each limit's
    /// reset as a duration like `"6m0s"`, `"7.66s"`, or `"76ms"` (`providers/openai.rb`), and the
    /// longer of the two waits wins.
    pub fn retry_delay(&self, headers: &[(String, String)]) -> Option<f64> {
        if *self != Provider::OpenAI {
            return None;
        }
        ["x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"]
            .iter()
            .filter_map(|name| headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)))
            .filter_map(|(_, v)| parse_reset_duration(v))
            .reduce(f64::max)
    }

    /// `Provider#parse_error`, with the overrides: Perplexity's HTML error pages and
    /// OpenRouter's `error.metadata.raw` upstream message. `None` for an empty body.
    pub fn parse_error(&self, body: &str) -> Option<String> {
        match self {
            Provider::Perplexity => self
                .strip_html_error(body)
                .or_else(|| crate::error::parse_error_message(body)),
            Provider::OpenRouter => openrouter_parse_error(body),
            _ => crate::error::parse_error_message(body),
        }
    }

    /// `Perplexity#router_url(operation)`: the Router lives under `/router/v1` of the API base,
    /// which may already end in it.
    pub(crate) fn router_url(&self, config: &Config, operation: &str) -> Result<String> {
        let base = self.api_base(config)?;
        let base = base.strip_suffix('/').unwrap_or(&base);
        let base = base.strip_suffix("/router/v1").unwrap_or(base);
        Ok(format!("{base}/router/v1/{operation}"))
    }

    /// `Perplexity#parse_error`: the `<title>` of the HTML page Perplexity returns for auth
    /// failures, minus the leading status code (`/<title>(.+?)<\/title>/`, `sub(/^\d+\s+/, '')`).
    pub(crate) fn strip_html_error(&self, body: &str) -> Option<String> {
        if *self != Provider::Perplexity || !body.contains("<html>") || !body.contains("<title>") {
            return None;
        }
        let start = body.find("<title>")? + 7;
        let end = body[start..].find("</title>")? + start;
        let title = &body[start..end];
        if title.is_empty() || title.contains('\n') {
            return None;
        }
        let digits = title.len() - title.trim_start_matches(|c: char| c.is_ascii_digit()).len();
        let rest = &title[digits..];
        let trimmed = rest.trim_start();
        Some(
            if digits > 0 && trimmed.len() < rest.len() {
                trimmed
            } else {
                title
            }
            .to_string(),
        )
    }
}

/// `Providers::OpenAI#parse_reset_duration`: `"1h2m3s"` style durations; `None` for anything
/// that is not entirely made of `<number><unit>` parts.
fn parse_reset_duration(value: &str) -> Option<f64> {
    static PART: std::sync::LazyLock<regex::Regex> =
        std::sync::LazyLock::new(|| regex::Regex::new(r"(\d+(?:\.\d+)?)(ms|h|m|s)").unwrap()); // constant pattern
    let parts: Vec<(&str, &str)> = PART
        .captures_iter(value)
        .map(|c| {
            let (_, [amount, unit]) = c.extract();
            (amount, unit)
        })
        .collect();
    if parts.is_empty()
        || parts
            .iter()
            .map(|(a, u)| format!("{a}{u}"))
            .collect::<String>()
            != value
    {
        return None;
    }
    let seconds = |unit: &str| match unit {
        "h" => 3600.0,
        "m" => 60.0,
        "s" => 1.0,
        _ => 0.001,
    };
    Some(
        parts
            .iter()
            .map(|(a, u)| a.parse::<f64>().unwrap_or(0.0) * seconds(u))
            .sum(),
    )
}

/// `Providers::OpenRouter#parse_error`: the shared body shapes, plus the upstream provider's own
/// message from `error.metadata.raw` appended as `"<message> - <raw message>"`.
fn openrouter_parse_error(body: &str) -> Option<String> {
    use serde_json::Value;
    if body.is_empty() {
        return None;
    }
    let try_parse =
        |s: &str| serde_json::from_str::<Value>(s).unwrap_or_else(|_| Value::String(s.to_string()));
    // `error_message`: a hash's `message`, a list's messages joined, a scalar as text.
    fn error_message(value: Option<&Value>) -> Option<String> {
        match value? {
            Value::Object(o) => o.get("message").and_then(Value::as_str).map(str::to_string),
            Value::Array(parts) => {
                let messages: Vec<String> = parts
                    .iter()
                    .filter_map(|p| error_message(Some(p)))
                    .collect();
                (!messages.is_empty()).then(|| messages.join(". "))
            }
            Value::Null => None,
            Value::String(s) => Some(s.clone()),
            other => Some(other.to_string()),
        }
    }
    let part_message = |part: &Value| -> Option<String> {
        let Value::Object(part) = part else {
            return error_message(Some(part));
        };
        let error = part.get("error");
        let message = error_message(error);
        let Some(Value::Object(metadata)) = error.and_then(|e| e.get("metadata")) else {
            return message;
        };
        let raw = match metadata.get("raw") {
            Some(Value::String(s)) => try_parse(s),
            Some(other) => other.clone(),
            None => Value::Null,
        };
        let Value::Object(raw) = raw else {
            return message;
        };
        match error_message(raw.get("error")) {
            Some(raw_message) => Some(
                message
                    .into_iter()
                    .chain([raw_message])
                    .collect::<Vec<_>>()
                    .join(" - "),
            ),
            None => message,
        }
    };
    match try_parse(body) {
        Value::Object(o) => part_message(&Value::Object(o)),
        Value::Array(parts) => {
            let messages: Vec<String> = parts
                .iter()
                .filter_map(part_message)
                .filter(|m| !m.is_empty())
                .collect();
            (!messages.is_empty()).then(|| messages.join(". "))
        }
        Value::String(s) => Some(s),
        other => Some(other.to_string()),
    }
}

/// `Provider.local_providers`.
pub fn local_providers() -> Vec<Provider> {
    ALL.iter().copied().filter(Provider::is_local).collect()
}

/// `Provider.remote_providers`.
pub fn remote_providers() -> Vec<Provider> {
    ALL.iter().copied().filter(Provider::is_remote).collect()
}

/// `Provider.configured_providers(config)`: the providers whose requirements `config` meets.
pub fn configured_providers(config: &Config) -> Vec<Provider> {
    ALL.iter()
        .copied()
        .filter(|p| p.is_configured(config))
        .collect()
}

/// `Provider.configured_remote_providers(config)`.
pub fn configured_remote_providers(config: &Config) -> Vec<Provider> {
    ALL.iter()
        .copied()
        .filter(|p| p.is_remote() && p.is_configured(config))
        .collect()
}
