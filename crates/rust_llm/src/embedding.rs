//! Port of `lib/ruby_llm/embedding.rb` with the OpenAI-compatible and Gemini embedding protocols.

use serde_json::{Value, json};

use crate::attachment::{Attachment, AttachmentType};
use crate::chat::resolve_model;
use crate::cost::{Cost, Tier};
use crate::error::{Error, Result};
use crate::model::Model;
use crate::providers::Provider;
use crate::tokens::Tokens;
use crate::transport::Connection;

/// Vectors for one text (`Single`) or a batch (`Batch`), matching RubyLLM returning a flat array
/// for a single string and an array of arrays for an array input.
#[derive(Debug, Clone, PartialEq)]
pub enum Vectors {
    Single(Vec<f64>),
    Batch(Vec<Vec<f64>>),
}

#[derive(Debug, Clone)]
pub struct Embedding {
    pub vectors: Vectors,
    pub model: String,
    pub input_tokens: Option<i64>,
    /// What the provider billed, when it says (OpenRouter's `usage.cost`).
    pub reported_cost: Option<f64>,
    /// `ruby_llm_usage_entries`: set when a batch prices the embedding at batch rates.
    pub usage_entries: Vec<crate::message::UsageEntry>,
    model_info: Option<Model>,
}

impl Embedding {
    pub fn tokens(&self) -> Tokens {
        if !self.usage_entries.is_empty() {
            return Tokens::aggregate(self.usage_entries.iter().map(|e| &e.tokens));
        }
        Tokens { input: self.input_tokens, reported_cost: self.reported_cost, ..Default::default() }
    }

    pub fn cost(&self) -> Cost {
        if !self.usage_entries.is_empty() {
            let complete = self.usage_entries.iter().all(crate::message::UsageEntry::cost_available);
            return Cost::aggregate(self.usage_entries.iter().map(|e| &e.cost), complete);
        }
        Cost::new(&self.tokens(), self.model_info.as_ref(), Tier::Standard)
    }

    /// `parse_embedding_response` for an OpenAI-compatible body, e.g. one line of a batch result.
    pub(crate) fn from_openai_body(body: &Value, single: bool) -> Embedding {
        let rows: Vec<Vec<f64>> =
            body.get("data").and_then(Value::as_array).map(|d| d.iter().map(|x| floats(&x["embedding"])).collect()).unwrap_or_default();
        let model = body.get("model").and_then(Value::as_str).unwrap_or_default().to_string();
        Embedding {
            vectors: vectors_from(rows, single),
            model_info: crate::models::models().find(&model, None).ok(),
            model,
            input_tokens: body.pointer("/usage/prompt_tokens").and_then(Value::as_i64),
            reported_cost: None,
            usage_entries: Vec::new(),
        }
    }
}

/// Input for `embed`: one string or several, or `nil` when only `with:` attachments are embedded.
pub enum EmbedInput {
    One(String),
    Many(Vec<String>),
    Nil,
}

impl From<Option<String>> for EmbedInput {
    fn from(s: Option<String>) -> Self {
        s.map_or(EmbedInput::Nil, EmbedInput::One)
    }
}

impl From<&str> for EmbedInput {
    fn from(s: &str) -> Self {
        EmbedInput::One(s.into())
    }
}

impl From<String> for EmbedInput {
    fn from(s: String) -> Self {
        EmbedInput::One(s)
    }
}

impl From<Vec<String>> for EmbedInput {
    fn from(v: Vec<String>) -> Self {
        EmbedInput::Many(v)
    }
}

#[derive(Default)]
pub struct EmbedOptions<'a> {
    pub model: Option<&'a str>,
    pub provider: Option<&'a str>,
    pub dimensions: Option<i64>,
    pub assume_model_exists: bool,
    /// `context:`: use this configuration instead of the global one.
    pub config: Option<std::sync::Arc<crate::Config>>,
    /// `with:`: images, audio, video, or PDFs embedded together with the text, on providers
    /// whose embeddings accept media (Gemini, OpenRouter, GPUStack).
    pub with: Vec<Attachment>,
}

fn vectors_from(rows: Vec<Vec<f64>>, single: bool) -> Vectors {
    if single && rows.len() == 1 {
        Vectors::Single(rows.into_iter().next().unwrap()) // len() == 1 checked above
    } else {
        Vectors::Batch(rows)
    }
}

fn int8s(v: &Value) -> Vec<f64> {
    use base64::Engine;
    match v.as_str() {
        Some(encoded) => base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .map(|bytes| bytes.into_iter().map(|b| b as i8 as f64).collect())
            .unwrap_or_default(),
        None => floats(v),
    }
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().map(|a| a.iter().filter_map(Value::as_f64).collect()).unwrap_or_default()
}

/// `RubyLLM.embed(text, model:, provider:, dimensions:)`.
pub async fn embed(input: impl Into<EmbedInput>, options: EmbedOptions<'_>) -> Result<Embedding> {
    let config = options.config.clone().unwrap_or_else(crate::config);
    let model_id = options.model.unwrap_or(&config.default_embedding_model).to_string();
    let (model, provider) = resolve_model(&model_id, options.provider, options.assume_model_exists)?;
    provider.ensure_configured(&config)?;
    let connection = Connection::new(provider, config.clone())?;
    let input = input.into();
    let single = !matches!(input, EmbedInput::Many(_));
    let text = match &input {
        EmbedInput::One(s) => Some(s.as_str()),
        _ => None,
    };
    let texts: Vec<String> = match &input {
        EmbedInput::One(s) => vec![s.clone()],
        EmbedInput::Many(v) => v.clone(),
        EmbedInput::Nil => vec![String::new()],
    };

    // `Protocol#embed`: media needs `supports_embedding_media?`, and one text at a time.
    let mut attachments = options.with;
    if let Some(first) = attachments.first() {
        if !matches!(provider, Provider::Gemini | Provider::OpenRouter | Provider::GPUStack) {
            return Err(Error::UnsupportedAttachment(crate::protocols::anthropic::unsupported(&first.mime_type)));
        }
        if !single {
            return Err(Error::Argument("embed one text at a time when embedding attachments".into()));
        }
        for a in &mut attachments {
            a.load(connection.client()).await?;
        }
    }

    let (path, payload) = match provider {
        // `Gemini::Embeddings#media_embedding_payload`.
        Provider::Gemini if !attachments.is_empty() => {
            let mut r = json!({
                "model": format!("models/{}", model.id),
                "content": { "parts": crate::protocols::gemini::format_content(text, &attachments)? },
            });
            if let Some(d) = options.dimensions {
                r["outputDimensionality"] = d.into();
            }
            (format!("models/{}:batchEmbedContents", model.id), json!({ "requests": [r] }))
        }
        Provider::Gemini => {
            let requests: Vec<Value> = texts
                .iter()
                .map(|t| {
                    let mut r = json!({ "model": format!("models/{}", model.id), "content": { "parts": [{ "text": t }] } });
                    if let Some(d) = options.dimensions {
                        r["outputDimensionality"] = d.into();
                    }
                    r
                })
                .collect();
            (format!("models/{}:batchEmbedContents", model.id), json!({ "requests": requests }))
        }
        Provider::Anthropic => return Err(Error::Api("Anthropic doesn't support embeddings".into(), None)),
        _ => {
            let input = match &input {
                _ if !attachments.is_empty() && provider == Provider::OpenRouter => {
                    json!([{ "content": openrouter_embedding_content(text, &attachments)? }])
                }
                EmbedInput::One(s) => json!(s),
                EmbedInput::Many(v) => json!(v),
                EmbedInput::Nil => Value::Null,
            };
            let mut payload = json!({ "model": model.id });
            if !input.is_null() {
                payload["input"] = input;
            }
            if let Some(d) = options.dimensions {
                payload["dimensions"] = d.into();
            }
            // `GPUStack::Embeddings`: media goes as a chat-style user message instead of `input`.
            if provider == Provider::GPUStack && !attachments.is_empty() {
                payload.as_object_mut().map(|p| p.remove("input"));
                let content = crate::protocols::chat_completions::format_content(provider, text, &attachments)?;
                payload["messages"] = json!([{ "role": "user", "content": content }]);
            }
            // `Perplexity::Embeddings#embedding_url`: embeddings stay off the Agent API.
            let path = if provider == Provider::Perplexity { "v1/embeddings" } else { "embeddings" };
            (path.to_string(), payload)
        }
    };

    let raw = connection.post(&path, &payload, &[], &mut |_| {}).await?;
    let body = raw.body;
    let (rows, input_tokens) = match provider {
        Provider::Gemini => (
            body.get("embeddings").and_then(Value::as_array).map(|e| e.iter().map(|x| floats(&x["values"])).collect()).unwrap_or_default(),
            None,
        ),
        // `Perplexity::Embeddings#decode_embedding`: base64-encoded signed int8 vectors.
        Provider::Perplexity => (
            body.get("data").and_then(Value::as_array).map(|d| d.iter().map(|x| int8s(&x["embedding"])).collect()).unwrap_or_default(),
            body.pointer("/usage/prompt_tokens").and_then(Value::as_i64),
        ),
        _ => (
            body.get("data").and_then(Value::as_array).map(|d| d.iter().map(|x| floats(&x["embedding"])).collect()).unwrap_or_default(),
            body.pointer("/usage/prompt_tokens").and_then(Value::as_i64),
        ),
    };
    let reported_cost = crate::protocols::chat_completions::reported_cost(provider, body.get("usage").unwrap_or(&Value::Null));
    Ok(Embedding {
        vectors: vectors_from(rows, single),
        model: model.id.clone(),
        input_tokens,
        reported_cost,
        usage_entries: Vec::new(),
        model_info: Some(model),
    })
}

/// `OpenRouter::Embeddings#format_embedding_content`: audio, video, and PDFs as `input_*` parts
/// carrying a data URI and short format, images as `image_url`, text files inline.
fn openrouter_embedding_content(text: Option<&str>, attachments: &[Attachment]) -> Result<Vec<Value>> {
    let mut parts = Vec::new();
    if let Some(t) = text {
        parts.push(json!({ "type": "text", "text": t }));
    }
    for a in attachments {
        let kind = match a.kind() {
            _ if a.is_provider_file() => None,
            AttachmentType::Audio => Some("input_audio"),
            AttachmentType::Video => Some("input_video"),
            AttachmentType::Pdf => Some("input_file"),
            AttachmentType::Image => {
                parts.push(json!({ "type": "image_url", "image_url": { "url": a.url_or_data_uri()? } }));
                continue;
            }
            AttachmentType::Text => {
                parts.push(json!({ "type": "text", "text": a.for_llm()? }));
                continue;
            }
            _ => None,
        };
        let Some(kind) = kind else {
            return Err(Error::UnsupportedAttachment(crate::protocols::anthropic::unsupported(&a.mime_type)));
        };
        parts.push(json!({ "type": kind, kind: { "data": a.for_llm()?, "format": a.format() } }));
    }
    Ok(parts)
}
