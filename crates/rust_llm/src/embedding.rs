//! Port of `lib/ruby_llm/embedding.rb` with the OpenAI-compatible and Gemini embedding protocols.

use std::collections::BTreeMap;

use serde_json::{Value, json};

use crate::attachment::{Attachment, AttachmentType};
use crate::chat::resolve_model;
use crate::cost::{Cost, Tier};
use crate::error::{Error, Result};
use crate::message::{Operation, UsageEntry, UsageStatus};
use crate::model::Model;
use crate::protocols::deep_merge;
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

/// Sparse vectors for one text (`Single`) or a batch (`Batch`), shaped like [`Vectors`]: each maps
/// a token id to its weight. A batch row is `None` where the server sent no sparse vector.
#[derive(Debug, Clone, PartialEq)]
pub enum SparseVectors {
    Single(BTreeMap<i64, f64>),
    Batch(Vec<Option<BTreeMap<i64, f64>>>),
}

#[derive(Debug, Clone)]
pub struct Embedding {
    pub vectors: Vectors,
    /// `sparse_vectors`: the token-to-weight map sparse-capable models return beside the dense
    /// vector; `None` on models that return only dense vectors.
    pub sparse_vectors: Option<SparseVectors>,
    pub model: String,
    pub input_tokens: Option<i64>,
    /// What the provider billed, when it says (OpenRouter's `usage.cost`).
    pub reported_cost: Option<f64>,
    /// `ruby_llm_usage_entries`: set when a batch prices the embedding at batch rates.
    pub usage_entries: Vec<crate::message::UsageEntry>,
    /// `attr_writer :model_info`: set by `Accounting::Usage::Tracker#succeed`.
    pub(crate) model_info: Option<Model>,
}

impl Embedding {
    pub fn tokens(&self) -> Tokens {
        if !self.usage_entries.is_empty() {
            return Tokens::aggregate(self.usage_entries.iter().map(|e| &e.tokens));
        }
        Tokens {
            input: self.input_tokens,
            reported_cost: self.reported_cost,
            ..Default::default()
        }
    }

    pub fn cost(&self) -> Cost {
        if !self.usage_entries.is_empty() {
            let complete = self
                .usage_entries
                .iter()
                .all(crate::message::UsageEntry::cost_available);
            return Cost::aggregate(self.usage_entries.iter().map(|e| &e.cost), complete);
        }
        Cost::new(&self.tokens(), self.model_info.as_ref(), Tier::Standard)
    }

    /// `model_info`: the model `model_info=` set, else the registry model for `model`.
    pub fn model_info(&self) -> Option<Model> {
        self.model_info.clone()
    }

    /// `Embedding.new(vectors:, model:, input_tokens:)`, e.g. one Gemini batch embedding result.
    pub fn new(vectors: Vectors, model: String, input_tokens: Option<i64>) -> Embedding {
        Embedding {
            model_info: crate::models::models().find(&model, None).ok(),
            vectors,
            sparse_vectors: None,
            model,
            input_tokens,
            reported_cost: None,
            usage_entries: Vec::new(),
        }
    }

    /// `parse_embedding_response` for an OpenAI-compatible body, e.g. one line of a batch result.
    pub(crate) fn from_openai_body(body: &Value, single: bool) -> Result<Embedding> {
        let data = body.get("data").and_then(Value::as_array);
        let rows: Vec<Vec<f64>> = data
            .map(|d| d.iter().map(|x| floats(&x["embedding"])).collect())
            .unwrap_or_default();
        let model = body
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        Ok(Embedding {
            sparse_vectors: parse_sparse_vectors(
                data.map_or(&[][..], Vec::as_slice),
                single && rows.len() == 1,
            )?,
            vectors: vectors_from(rows, single),
            model_info: crate::models::models().find(&model, None).ok(),
            model,
            input_tokens: body.pointer("/usage/prompt_tokens").and_then(Value::as_i64),
            reported_cost: None,
            usage_entries: Vec::new(),
        })
    }
}

/// Input for `embed`: one string or several, or `nil` when only `with:` attachments are embedded.
#[derive(Debug, Clone, PartialEq)]
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
    /// `task_type:`: the embedding task in the provider's vocabulary (Gemini's `taskType`,
    /// OpenRouter's `input_type`); ignored by providers without one.
    pub task_type: Option<&'a str>,
    /// `title:`: labels the document on Gemini retrieval tasks; ignored by other providers.
    pub title: Option<&'a str>,
    /// `provider_options:`: merged into the request as-is, overriding rendered fields.
    pub provider_options: Value,
    /// `metadata:`: added to the `embedding.rust_llm` event payload, never sent to the provider.
    pub metadata: Option<Value>,
    /// `owner:`: who the usage is attributed to, such as a user; wins over
    /// [`crate::accounting::with_usage_owner`].
    pub owner: Option<crate::accounting::UsageOwner>,
}

fn vectors_from(rows: Vec<Vec<f64>>, single: bool) -> Vectors {
    if single && rows.len() == 1 {
        Vectors::Single(rows.into_iter().next().unwrap()) // len() == 1 checked above
    } else {
        Vectors::Batch(rows)
    }
}

/// `ChatCompletions::Embeddings#parse_sparse_vectors`: sparse-capable models return a
/// token-to-weight map beside the dense vector, under `lexical_weights` on BGE-M3 and
/// `sparse_embedding` elsewhere. `None` when no row carries one.
fn parse_sparse_vectors(rows: &[Value], single: bool) -> Result<Option<SparseVectors>> {
    let sparse = rows
        .iter()
        .map(|row| {
            let weights = row
                .get("sparse_embedding")
                .filter(|w| !w.is_null())
                .or_else(|| row.get("lexical_weights"));
            weights.map_or(Ok(None), normalize_sparse_vector)
        })
        .collect::<Result<Vec<_>>>()?;
    if sparse.iter().all(Option::is_none) {
        return Ok(None);
    }
    Ok(Some(if single {
        SparseVectors::Single(sparse.into_iter().next().flatten().unwrap_or_default())
    } else {
        SparseVectors::Batch(sparse)
    }))
}

/// `normalize_sparse_vector`: `{ Integer(token) => Float(weight) }`; anything but a Hash is `nil`.
/// Like Ruby's `Integer()`/`Float()`, a token or weight that is not a number is an error.
fn normalize_sparse_vector(weights: &Value) -> Result<Option<BTreeMap<i64, f64>>> {
    let Some(weights) = weights.as_object() else {
        return Ok(None);
    };
    weights
        .iter()
        .map(|(token, weight)| {
            let token = token
                .trim()
                .parse()
                .map_err(|_| Error::Argument(format!("invalid value for Integer(): {token:?}")))?;
            let weight = weight
                .as_f64()
                .or_else(|| weight.as_str()?.trim().parse().ok())
                .ok_or_else(|| Error::Argument(format!("invalid value for Float(): {weight}")))?;
            Ok((token, weight))
        })
        .collect::<Result<_>>()
        .map(Some)
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
    v.as_array()
        .map(|a| a.iter().filter_map(Value::as_f64).collect())
        .unwrap_or_default()
}

/// `RubyLLM.embed(text, model:, provider:, dimensions:)`, inside an `embedding.rust_llm` event.
pub async fn embed(input: impl Into<EmbedInput>, options: EmbedOptions<'_>) -> Result<Embedding> {
    let input = input.into();
    let config = options.config.clone().unwrap_or_else(crate::config);
    let model_id = options
        .model
        .unwrap_or(&config.default_embedding_model)
        .to_string();
    let (model, provider) =
        resolve_model(&model_id, options.provider, options.assume_model_exists)?;
    let owner = options.owner.clone();
    let mut event = crate::instrumentation::Event::start(&config, "embedding.rust_llm", || {
        let empty = Tokens::default();
        crate::instrumentation::payload([
            ("provider", provider.slug().into()),
            ("provider_class", provider.display().into()),
            ("model", model.id.clone().into()),
            (
                "input",
                match &input {
                    EmbedInput::One(s) => s.clone().into(),
                    EmbedInput::Many(v) => serde_json::json!(v),
                    EmbedInput::Nil => Value::Null,
                },
            ),
            ("dimensions", options.dimensions.into()),
            ("task_type", options.task_type.into()),
            ("title", options.title.into()),
            ("attachment_count", options.with.len().into()),
            ("provider_options", options.provider_options.clone()),
            (
                "metadata",
                crate::instrumentation::metadata(&options.metadata),
            ),
            ("tokens", crate::instrumentation::tokens_h(&empty)),
            (
                "cost",
                crate::instrumentation::cost_h(&Cost::new(&empty, Some(&model), Tier::Standard)),
            ),
        ])
    });
    let result = tracing::Instrument::instrument(
        crate::accounting::owned_by(owner, embed_inner(input, options)),
        event.span(),
    )
    .await;
    if let Ok(e) = &result {
        crate::accounting::report(&config, &e.usage_entries).await;
        event.set("result", || {
            serde_json::json!({ "model": e.model, "vectors": match &e.vectors {
            Vectors::Single(v) => serde_json::json!(v),
            Vectors::Batch(v) => serde_json::json!(v),
        } })
        });
        event.set("response_model", || e.model.clone().into());
        event.set("tokens", || crate::instrumentation::tokens_h(&e.tokens()));
        event.set("cost", || crate::instrumentation::cost_h(&e.cost()));
        let (dimensions, count) = match &e.vectors {
            Vectors::Single(v) => (v.len(), 1),
            Vectors::Batch(v) => (v.first().map_or(0, Vec::len), v.len()),
        };
        event.set("embedding_dimensions", || dimensions.into());
        event.set("embedding_count", || count.into());
    }
    event.finish(result.as_ref().err());
    result
}

async fn embed_inner(input: EmbedInput, options: EmbedOptions<'_>) -> Result<Embedding> {
    let config = options.config.clone().unwrap_or_else(crate::config);
    let model_id = options
        .model
        .unwrap_or(&config.default_embedding_model)
        .to_string();
    let (model, provider) =
        resolve_model(&model_id, options.provider, options.assume_model_exists)?;
    provider.ensure_configured(&config)?;
    let connection = Connection::new(provider, config.clone())?;
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
        if !matches!(
            provider,
            Provider::Gemini | Provider::OpenRouter | Provider::GPUStack
        ) {
            return Err(Error::UnsupportedAttachment(
                crate::protocols::anthropic::unsupported(&first.mime_type),
            ));
        }
        if !single {
            return Err(Error::Argument(
                "embed one text at a time when embedding attachments".into(),
            ));
        }
        for a in &mut attachments {
            // `OpenRouter::Embeddings` passes image URLs through untouched; everything else is inlined.
            if !(provider == Provider::OpenRouter
                && a.is_url()
                && a.kind() == AttachmentType::Image)
            {
                a.load(connection.client()).await?;
            }
        }
    }
    let provider_options = options
        .provider_options
        .as_object()
        .cloned()
        .unwrap_or_default();

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
            if let Some(t) = options.task_type {
                r["taskType"] = t.into();
            }
            if let Some(t) = options.title {
                r["title"] = t.into();
            }
            let mut payload = json!({ "requests": [r] });
            deep_merge(&mut payload, &Value::Object(provider_options));
            (format!("models/{}:batchEmbedContents", model.id), payload)
        }
        Provider::Gemini => {
            let requests: Vec<Value> = texts
                .iter()
                .map(|t| {
                    let mut r = json!({ "model": format!("models/{}", model.id), "content": { "parts": [{ "text": t }] } });
                    if let Some(d) = options.dimensions {
                        r["outputDimensionality"] = d.into();
                    }
                    if let Some(t) = options.task_type {
                        r["taskType"] = t.into();
                    }
                    if let Some(t) = options.title {
                        r["title"] = t.into();
                    }
                    r
                })
                .collect();
            let mut payload = json!({ "requests": requests });
            deep_merge(&mut payload, &Value::Object(provider_options));
            (format!("models/{}:batchEmbedContents", model.id), payload)
        }
        Provider::Anthropic => {
            return Err(Error::Api(
                "Anthropic doesn't support embeddings".into(),
                None,
            ));
        }
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
                // `Mistral::Embeddings#render_embedding_payload` names it `output_dimension`.
                let key = if provider == Provider::Mistral {
                    "output_dimension"
                } else {
                    "dimensions"
                };
                payload[key] = d.into();
            }
            // `OpenRouter::Embeddings`: the task type is OpenRouter's `input_type`.
            if provider == Provider::OpenRouter
                && let Some(t) = options.task_type
            {
                payload["input_type"] = t.into();
            }
            // `GPUStack::Embeddings`: media goes as a chat-style user message instead of `input`.
            if provider == Provider::GPUStack && !attachments.is_empty() {
                payload.as_object_mut().map(|p| p.remove("input"));
                let content = crate::protocols::chat_completions::format_content(
                    provider,
                    text,
                    &attachments,
                )?;
                payload["messages"] = json!([{ "role": "user", "content": content }]);
            }
            // `.merge(provider_options)`: provider options replace top-level keys.
            if let Some(p) = payload.as_object_mut() {
                p.extend(provider_options);
            }
            // `Perplexity::Embeddings#embedding_url`: embeddings stay off the Agent API.
            let path = if provider == Provider::Perplexity {
                "v1/embeddings"
            } else {
                "embeddings"
            };
            (path.to_string(), payload)
        }
    };

    // `track_usage(:embedding)`: one entry per HTTP attempt.
    let mut retried: Vec<Tokens> = Vec::new();
    let mut on_attempt = |previous: Option<&Error>| {
        if let Some(e) = previous {
            retried.push(crate::chat::failure_tokens(e, None));
        }
    };
    let raw = match connection.post(&path, &payload, &[], &mut on_attempt).await {
        Ok(raw) => raw,
        // `track_usage`'s rescue (`fail_pending`): the failed attempts have no result to attach
        // to, so they are only reported.
        Err(e) => {
            let entries: Vec<UsageEntry> = retried
                .iter()
                .cloned()
                .chain([crate::chat::failure_tokens(&e, None)])
                .map(|tokens| {
                    let mut entry =
                        UsageEntry::new(Operation::Embedding, provider.slug(), Some(&model.id));
                    entry.status = UsageStatus::Failed;
                    entry.cost = crate::rerank::embeddings_cost(&tokens, Some(&model));
                    entry.tokens = tokens;
                    entry
                })
                .collect();
            crate::accounting::report(&config, &entries).await;
            return Err(e);
        }
    };
    let body = raw.body;
    // `ChatCompletions::Embeddings#parse_sparse_vectors`; Gemini, Mistral, and Perplexity override
    // `parse_embedding_response` without it.
    let sparse_vectors = match provider {
        Provider::Gemini | Provider::Mistral | Provider::Perplexity => None,
        _ => parse_sparse_vectors(
            body.get("data")
                .and_then(Value::as_array)
                .map_or(&[][..], Vec::as_slice),
            single && body.get("data").and_then(Value::as_array).map(Vec::len) == Some(1),
        )?,
    };
    let (rows, input_tokens) = match provider {
        Provider::Gemini => (
            body.get("embeddings")
                .and_then(Value::as_array)
                .map(|e| e.iter().map(|x| floats(&x["values"])).collect())
                .unwrap_or_default(),
            None,
        ),
        // `Perplexity::Embeddings#decode_embedding`: base64-encoded signed int8 vectors.
        Provider::Perplexity => (
            body.get("data")
                .and_then(Value::as_array)
                .map(|d| d.iter().map(|x| int8s(&x["embedding"])).collect())
                .unwrap_or_default(),
            body.pointer("/usage/prompt_tokens").and_then(Value::as_i64),
        ),
        _ => (
            body.get("data")
                .and_then(Value::as_array)
                .map(|d| d.iter().map(|x| floats(&x["embedding"])).collect())
                .unwrap_or_default(),
            body.pointer("/usage/prompt_tokens").and_then(Value::as_i64),
        ),
    };
    let reported_cost = crate::protocols::chat_completions::reported_cost(
        provider,
        body.get("usage").unwrap_or(&Value::Null),
    );
    let mut embedding = Embedding {
        vectors: vectors_from(rows, single),
        sparse_vectors,
        model: model.id.clone(),
        input_tokens,
        reported_cost,
        usage_entries: Vec::new(),
        model_info: Some(model.clone()),
    };
    let entry = |status, tokens: Tokens, cost: Option<Cost>| UsageEntry {
        id: UsageEntry::next_id(),
        owner: crate::accounting::usage_owner(),
        operation: Operation::Embedding,
        provider: provider.slug().into(),
        model: model.id.clone(),
        status,
        cost: cost
            .filter(|c| c.total().is_some())
            .unwrap_or_else(|| crate::rerank::embeddings_cost(&tokens, Some(&model))),
        tokens,
    };
    let mut entries: Vec<UsageEntry> = retried
        .into_iter()
        .map(|t| entry(UsageStatus::Failed, t, None))
        .collect();
    entries.push(entry(
        UsageStatus::Succeeded,
        embedding.tokens(),
        Some(embedding.cost()),
    ));
    embedding.usage_entries = entries;
    Ok(embedding)
}

/// `OpenRouter::Embeddings#format_embedding_content`: audio, video, and PDFs as `input_*` parts
/// carrying a data URI and short format, images as `image_url`, text files inline.
fn openrouter_embedding_content(
    text: Option<&str>,
    attachments: &[Attachment],
) -> Result<Vec<Value>> {
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
                parts.push(
                    json!({ "type": "image_url", "image_url": { "url": a.url_or_data_uri()? } }),
                );
                continue;
            }
            AttachmentType::Text => {
                parts.push(json!({ "type": "text", "text": a.for_llm()? }));
                continue;
            }
            _ => None,
        };
        let Some(kind) = kind else {
            return Err(Error::UnsupportedAttachment(
                crate::protocols::anthropic::unsupported(&a.mime_type),
            ));
        };
        parts.push(json!({ "type": kind, kind: { "data": a.for_llm()?, "format": a.format() } }));
    }
    Ok(parts)
}
