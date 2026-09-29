//! Port of `lib/ruby_llm/rerank.rb` (`RubyLLM.rerank`), `Protocol#rerank`, and
//! `lib/ruby_llm/protocols/chat_completions/rerank.rb`, the Jina-style endpoint OpenRouter and
//! GPUStack serve. Cohere's and Bedrock's rerank protocols belong to providers RustLLM does not port.
//!
//! ```ruby
//! rerank = RubyLLM.rerank("what is ruby", docs, model: "voyageai/rerank-2.5-lite", provider: :openrouter)
//! rerank.results.first.document # => "Ruby is a programming language"
//! ```

use std::sync::Arc;

use serde_json::{Value, json};

use crate::chat::{failure_tokens, resolve_model};
use crate::config::Config;
use crate::cost::{Cost, Tier};
use crate::error::{Error, Result};
use crate::message::{Operation, UsageEntry, UsageStatus};
use crate::model::Model;
use crate::models;
use crate::protocols::int;
use crate::providers::Provider;
use crate::tokens::Tokens;
use crate::transport::Connection;

/// One reranked document (`RubyLLM::Rerank::Result`).
#[derive(Debug, Clone, PartialEq)]
pub struct RerankResult {
    /// The document's position in the input array.
    pub index: usize,
    pub document: String,
    /// The provider's relevance score.
    pub score: Option<f64>,
}

/// Documents ordered by relevance to a query (`RubyLLM::Rerank`).
#[derive(Debug, Clone)]
pub struct Rerank {
    /// Most relevant first.
    pub results: Vec<RerankResult>,
    /// The id of the model that ranked the documents.
    pub model: String,
    /// The raw provider response body.
    pub raw: Value,
    /// One entry per provider attempt.
    pub usage_entries: Vec<UsageEntry>,
    input_tokens: Option<i64>,
    reported_cost: Option<f64>,
}

impl Rerank {
    /// Usage across every attempt, or what the response reported.
    pub fn tokens(&self) -> Tokens {
        if !self.usage_entries.is_empty() {
            return Tokens::aggregate(self.usage_entries.iter().map(|e| &e.tokens));
        }
        Tokens { input: self.input_tokens, reported_cost: self.reported_cost, ..Default::default() }
    }

    /// The rerank cost across every attempt, priced as embeddings.
    pub fn cost(&self) -> Cost {
        if !self.usage_entries.is_empty() {
            let complete = self.usage_entries.iter().all(UsageEntry::cost_available);
            return Cost::aggregate(self.usage_entries.iter().map(|e| &e.cost), complete);
        }
        embeddings_cost(&self.tokens(), self.model_info().as_ref())
    }

    /// The registry entry for `model`, or `None`.
    pub fn model_info(&self) -> Option<Model> {
        models::models().find(&self.model, None).ok()
    }
}

/// `Cost.new(category: :embeddings)`: input and output use the embeddings prices, each falling
/// back to the text price.
fn embeddings_cost(tokens: &Tokens, model: Option<&Model>) -> Cost {
    let Some(model) = model else { return Cost::new(tokens, None, Tier::Standard) };
    let Some(embeddings) = model.pricing.embeddings.as_ref() else { return Cost::new(tokens, Some(model), Tier::Standard) };
    let mut priced = model.clone();
    let mut text = priced.pricing.text_tokens.clone().unwrap_or_default();
    let mut standard = text.standard.clone().unwrap_or_default();
    standard.input_per_million = embeddings.input().or(standard.input_per_million);
    standard.output_per_million = embeddings.output().or(standard.output_per_million);
    text.standard = Some(standard);
    priced.pricing.text_tokens = Some(text);
    Cost::new(tokens, Some(&priced), Tier::Standard)
}

/// Options for [`rerank`], the keyword arguments of `Rerank.rerank`.
#[derive(Default)]
pub struct RerankOptions<'a> {
    pub provider: Option<&'a str>,
    pub assume_model_exists: bool,
    /// `top_n:`: how many results come back.
    pub top_n: Option<i64>,
    /// `provider_options:`: merged into the request as-is.
    pub provider_options: Value,
    /// `context:`: use this configuration instead of the global one.
    pub config: Option<Arc<Config>>,
}

/// `RubyLLM.rerank(query, documents, model:, provider:, top_n:, provider_options:)`. `model` is
/// required: rerank catalogs are provider-specific.
pub async fn rerank(query: &str, documents: &[&str], model: &str, options: RerankOptions<'_>) -> Result<Rerank> {
    let RerankOptions { provider, assume_model_exists, top_n, provider_options, config } = options;
    let config = config.unwrap_or_else(crate::config);
    let (model, provider) = resolve_model(model, provider, assume_model_exists)?;
    provider.ensure_configured(&config)?;
    if !matches!(provider, Provider::OpenRouter | Provider::GPUStack) {
        return Err(Error::Api(format!("{} doesn't support reranking", provider.display()), None));
    }
    let connection = Connection::new(provider, config.clone())?;
    let payload = render_payload(query, documents, &model.id, top_n, &provider_options);

    // `track_usage(:rerank)`: one entry per HTTP attempt.
    let mut retried: Vec<Tokens> = Vec::new();
    let mut on_attempt = |previous: Option<&Error>| {
        if let Some(e) = previous {
            retried.push(failure_tokens(e, None));
        }
    };
    let raw = connection.post("rerank", &payload, &[], &mut on_attempt).await?;
    let mut result = parse_response(raw.body, &model.id, documents)?;
    let entry = |status, tokens: Tokens, cost: Option<Cost>| UsageEntry {
        id: UsageEntry::next_id(),
        operation: Operation::Rerank,
        provider: provider.slug().into(),
        model: model.id.clone(),
        status,
        cost: cost.filter(|c| c.total().is_some()).unwrap_or_else(|| embeddings_cost(&tokens, Some(&model))),
        tokens,
    };
    let mut entries: Vec<UsageEntry> = retried.into_iter().map(|t| entry(UsageStatus::Failed, t, None)).collect();
    entries.push(entry(UsageStatus::Succeeded, result.tokens(), Some(result.cost())));
    result.usage_entries = entries;
    Ok(result)
}

/// `render_rerank_payload`: `top_n` only when given, then provider options.
fn render_payload(query: &str, documents: &[&str], model: &str, top_n: Option<i64>, provider_options: &Value) -> Value {
    let mut payload = json!({ "model": model, "query": query, "documents": documents });
    if let Some(top_n) = top_n {
        payload["top_n"] = top_n.into();
    }
    if let (Some(p), Some(options)) = (payload.as_object_mut(), provider_options.as_object()) {
        p.extend(options.clone());
    }
    payload
}

/// `parse_rerank_response`: each result names a document by index; the document text comes from
/// the response when present, otherwise from the request.
fn parse_response(data: Value, model: &str, documents: &[&str]) -> Result<Rerank> {
    let mut results = Vec::new();
    for result in data.get("results").and_then(Value::as_array).into_iter().flatten() {
        let index = result
            .get("index")
            .and_then(Value::as_u64)
            .map(|i| i as usize)
            .filter(|i| *i < documents.len())
            .ok_or_else(|| Error::Api("Rerank endpoint returned an invalid document index".into(), None))?;
        let document = match result.get("document") {
            Some(Value::Object(d)) => d.get("text").and_then(Value::as_str).map(str::to_string),
            Some(Value::String(s)) => Some(s.clone()),
            _ => None,
        };
        results.push(RerankResult {
            index,
            document: document.unwrap_or_else(|| documents[index].to_string()),
            score: result.get("relevance_score").and_then(Value::as_f64),
        });
    }
    let usage = data.get("usage").cloned().unwrap_or_else(|| json!({}));
    Ok(Rerank {
        results,
        model: data.get("model").and_then(Value::as_str).unwrap_or(model).to_string(),
        input_tokens: int(usage.get("total_tokens")),
        reported_cost: usage.get("cost").and_then(Value::as_f64),
        raw: data,
        usage_entries: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // rerank_spec.rb: "renders the Jina-style rerank request".
    #[test]
    fn renders_the_jina_style_rerank_request() {
        let payload = render_payload("what is ruby", &["a", "b"], "voyageai/rerank-2.5-lite", Some(2), &Value::Null);
        assert_eq!(
            payload,
            json!({ "model": "voyageai/rerank-2.5-lite", "query": "what is ruby", "documents": ["a", "b"], "top_n": 2 })
        );
    }

    #[test]
    fn an_out_of_range_index_is_an_error() {
        let err = parse_response(json!({ "results": [{ "index": 2, "relevance_score": 0.1 }] }), "m", &["a", "b"]).unwrap_err();
        assert!(err.to_string().contains("invalid document index"));
    }
}
