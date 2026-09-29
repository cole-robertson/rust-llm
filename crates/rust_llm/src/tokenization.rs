//! Port of `lib/ruby_llm/tokenization.rb` (`RubyLLM.tokenize`) with `protocols/xai/tokenization.rb`
//! and `protocols/gpustack/tokenization.rb`, plus the input-token counting seams behind
//! `Chat#count_tokens`: `render_count_tokens_payload`/`count_tokens_url`/`parse_count_tokens_response`
//! in `protocols/anthropic/chat.rb`, `protocols/gemini/chat.rb`, and `protocols/responses/token_counting.rb`.
//!
//! ```ruby
//! RubyLLM.tokenize("Hello Ruby", model: "grok-4.3", provider: :xai).count
//! chat.count_tokens("What's the weather in Berlin?")
//! ```

use std::sync::Arc;

use serde_json::{Map, Value, json};

use crate::chat::resolve_model;
use crate::config::Config;
use crate::error::{Error, Result};
use crate::model::Model;
use crate::providers::{ProtocolName, Provider};
use crate::transport::Connection;

/// The token ids a model's tokenizer produced for plain text (`RubyLLM::Tokenization`). They
/// exclude chat formatting, tools, and media, and are not billable usage.
#[derive(Debug, Clone, PartialEq)]
pub struct Tokenization {
    /// The token ids in text order.
    pub ids: Vec<i64>,
    /// The id of the model whose tokenizer was used.
    pub model: String,
    /// The provider's unmodified response, including token strings when available.
    pub raw: Value,
}

impl Tokenization {
    /// The number of tokens in the text.
    pub fn count(&self) -> usize {
        self.ids.len()
    }
}

/// Options for [`tokenize`], the keyword arguments of `Tokenization.tokenize`.
#[derive(Default)]
pub struct TokenizeOptions<'a> {
    /// `model:`, defaulting to `config.default_model`.
    pub model: Option<&'a str>,
    pub provider: Option<&'a str>,
    pub assume_model_exists: bool,
    /// `context:`: use this configuration instead of the global one.
    pub config: Option<Arc<Config>>,
}

/// `RubyLLM.tokenize(text, model:, provider:)`. Records no usage: tokenizing is not generation.
pub async fn tokenize(text: &str, options: TokenizeOptions<'_>) -> Result<Tokenization> {
    let TokenizeOptions {
        model,
        provider,
        assume_model_exists,
        config,
    } = options;
    let config = config.unwrap_or_else(crate::config);
    let model_id = model.unwrap_or(&config.default_model).to_string();
    let (model, provider) = resolve_model(&model_id, provider, assume_model_exists)?;
    provider.ensure_configured(&config)?;
    let (path, payload) = match provider {
        Provider::XAI => (
            "tokenize-text".to_string(),
            json!({ "model": model.id, "text": text }),
        ),
        Provider::GPUStack => (
            format!(
                "{}/tokenize",
                gpustack_backend_base(&provider.api_base(&config)?)?
            ),
            json!({ "model": model.id, "prompt": text }),
        ),
        other => {
            return Err(Error::Api(
                format!("{} doesn't support text tokenization", other.display()),
                None,
            ));
        }
    };
    let connection = Connection::new(provider, config)?;
    let body = connection
        .post(&path, &payload, &[], &mut |_| {})
        .await?
        .body;
    let ids = match provider {
        Provider::XAI => body.get("token_ids").and_then(Value::as_array).map(|ids| {
            ids.iter()
                .map(|t| t.get("token_id").and_then(Value::as_i64))
                .collect::<Option<Vec<_>>>()
        }),
        _ => body
            .get("tokens")
            .and_then(Value::as_array)
            .map(|ids| ids.iter().map(Value::as_i64).collect::<Option<Vec<_>>>()),
    }
    .flatten()
    .ok_or_else(|| {
        Error::Api(
            "The provider returned an invalid tokenization response".into(),
            None,
        )
    })?;
    Ok(Tokenization {
        ids,
        model: model.id,
        raw: body,
    })
}

/// `GPUStack#backend_api_base`: the model proxy root, without its `/v1`.
fn gpustack_backend_base(api_base: &str) -> Result<String> {
    let trimmed = api_base.trim_end_matches('/');
    let proxy = trimmed.strip_suffix("/v1").filter(|base| {
        let mut parts = base.rsplit('/');
        parts
            .next()
            .is_some_and(|id| !id.is_empty() && id.bytes().all(|b| b.is_ascii_digit()))
            && parts.next() == Some("proxy")
            && parts.next() == Some("model")
    });
    proxy.map(str::to_string).ok_or_else(|| {
        Error::Api(
            "This GPUStack operation requires gpustack_api_base to end in /model/proxy/ROUTE_ID/v1"
                .into(),
            None,
        )
    })
}

/// Which protocols can count input tokens: Anthropic, Gemini, and OpenAI's Responses API
/// (`OpenAI::Responses` includes `Responses::TokenCounting`; other Responses dialects do not).
pub(crate) fn count_tokens_endpoint(
    protocol: ProtocolName,
    provider: Provider,
    model: &Model,
) -> Result<String> {
    match (protocol, provider) {
        (ProtocolName::Anthropic, _) => Ok("v1/messages/count_tokens".into()),
        (ProtocolName::Gemini, _) => Ok(format!("models/{}:countTokens", model.id)),
        (ProtocolName::Responses, Provider::OpenAI) => Ok("responses/input_tokens".into()),
        _ => Err(Error::Api(
            format!("{} doesn't support token counting", provider.display()),
            None,
        )),
    }
}

const ANTHROPIC_COUNT_TOKENS_KEYS: &[&str] = &[
    "model",
    "messages",
    "system",
    "tools",
    "tool_choice",
    "thinking",
];
const RESPONSES_COUNT_TOKENS_KEYS: &[&str] = &[
    "model",
    "input",
    "instructions",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "reasoning",
    "text",
];
const GEMINI_COUNT_TOKENS_KEYS: &[&str] = &["contents", "systemInstruction", "tools"];

/// `render_count_tokens_payload`: the chat payload, sliced to what the counting endpoint takes.
pub(crate) fn count_tokens_payload(
    protocol: ProtocolName,
    model: &Model,
    rendered: Value,
) -> Value {
    let slice = |keys: &[&str]| -> Map<String, Value> {
        let mut object = rendered.as_object().cloned().unwrap_or_default();
        object.retain(|k, _| keys.contains(&k.as_str()));
        object
    };
    match protocol {
        ProtocolName::Anthropic => Value::Object(slice(ANTHROPIC_COUNT_TOKENS_KEYS)),
        ProtocolName::Gemini => {
            let mut request = slice(GEMINI_COUNT_TOKENS_KEYS);
            request.insert("model".into(), format!("models/{}", model.id).into());
            json!({ "generateContentRequest": request })
        }
        _ => Value::Object(slice(RESPONSES_COUNT_TOKENS_KEYS)),
    }
}

/// `parse_count_tokens_response`.
pub(crate) fn parse_count_tokens(protocol: ProtocolName, body: &Value) -> Result<i64> {
    let key = if protocol == ProtocolName::Gemini {
        "totalTokens"
    } else {
        "input_tokens"
    };
    body.get(key).and_then(Value::as_i64).ok_or_else(|| {
        Error::Api(
            format!("The provider returned no {key} in its token count"),
            None,
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpustack_needs_a_model_proxy_base() {
        assert_eq!(
            gpustack_backend_base("http://gpu.local/model/proxy/12/v1/").unwrap(),
            "http://gpu.local/model/proxy/12"
        );
        assert!(gpustack_backend_base("http://gpu.local/v1").is_err());
    }

    #[test]
    fn gemini_wraps_the_sliced_request_with_the_model_name() {
        let model = Model::default_for("gemini-3.5-flash", "gemini");
        let rendered =
            json!({ "contents": [], "generationConfig": {}, "systemInstruction": { "parts": [] } });
        assert_eq!(
            count_tokens_payload(ProtocolName::Gemini, &model, rendered),
            json!({ "generateContentRequest": { "contents": [], "systemInstruction": { "parts": [] }, "model": "models/gemini-3.5-flash" } })
        );
    }
}
