//! Port of `lib/ruby_llm/embedding.rb` with the OpenAI-compatible and Gemini embedding protocols.

use serde_json::{Value, json};

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
    model_info: Option<Model>,
}

impl Embedding {
    pub fn tokens(&self) -> Tokens {
        Tokens { input: self.input_tokens, ..Default::default() }
    }

    pub fn cost(&self) -> Cost {
        Cost::new(&self.tokens(), self.model_info.as_ref(), Tier::Standard)
    }
}

/// Input for `embed`: one string or several.
pub enum EmbedInput {
    One(String),
    Many(Vec<String>),
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
}

fn vectors_from(rows: Vec<Vec<f64>>, single: bool) -> Vectors {
    if single && rows.len() == 1 {
        Vectors::Single(rows.into_iter().next().unwrap())
    } else {
        Vectors::Batch(rows)
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
    let single = matches!(input, EmbedInput::One(_));
    let texts: Vec<String> = match input {
        EmbedInput::One(s) => vec![s],
        EmbedInput::Many(v) => v,
    };

    let (path, payload) = match provider {
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
            let input = if single { json!(texts[0]) } else { json!(texts) };
            let mut payload = json!({ "model": model.id, "input": input });
            if let Some(d) = options.dimensions {
                payload["dimensions"] = d.into();
            }
            ("embeddings".to_string(), payload)
        }
    };

    let raw = connection.post(&path, &payload, &[], &mut || {}).await?;
    let body = raw.body;
    let (rows, input_tokens) = match provider {
        Provider::Gemini => (
            body.get("embeddings").and_then(Value::as_array).map(|e| e.iter().map(|x| floats(&x["values"])).collect()).unwrap_or_default(),
            None,
        ),
        _ => (
            body.get("data").and_then(Value::as_array).map(|d| d.iter().map(|x| floats(&x["embedding"])).collect()).unwrap_or_default(),
            body.pointer("/usage/prompt_tokens").and_then(Value::as_i64),
        ),
    };
    Ok(Embedding { vectors: vectors_from(rows, single), model: model.id.clone(), input_tokens, model_info: Some(model) })
}
