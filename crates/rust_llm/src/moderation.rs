//! Port of `lib/ruby_llm/moderation.rb` (`RubyLLM.moderate`), `Protocol#moderate`, and
//! `lib/ruby_llm/protocols/chat_completions/moderation.rb`, which every Chat Completions and
//! Responses provider inherits.
//!
//! ```ruby
//! result = RubyLLM.moderate("This is a safe message about Ruby programming")
//! result.flagged?  # => false
//! ```

use std::sync::Arc;

use serde_json::{Map, Value, json};

use crate::attachment::{Attachment, AttachmentType};
use crate::chat::{failure_tokens, resolve_model};
use crate::config::Config;
use crate::cost::{Cost, Tier};
use crate::error::{Error, Result};
use crate::message::{Operation, UsageEntry, UsageStatus};
use crate::protocols::anthropic::unsupported;
use crate::providers::ProtocolName;
use crate::tokens::Tokens;
use crate::transport::Connection;

/// The verdict for one moderated input (`RubyLLM::Moderation::Result`).
#[derive(Debug, Clone, PartialEq)]
pub struct ModerationResult {
    flagged: bool,
    /// The names of the categories flagged for this input, empty when nothing was flagged.
    pub categories: Vec<String>,
    /// Category name to a score between 0.0 and 1.0, in the provider's order.
    pub category_scores: Map<String, Value>,
}

impl ModerationResult {
    /// `Moderation::Result.new(flagged:, categories:, category_scores:)`.
    pub fn new(
        flagged: bool,
        categories: Vec<String>,
        category_scores: Map<String, Value>,
    ) -> ModerationResult {
        ModerationResult {
            flagged,
            categories,
            category_scores,
        }
    }

    /// `Result.from_h`.
    fn from_h(data: &Value) -> ModerationResult {
        let categories: Vec<String> = data
            .get("categories")
            .and_then(Value::as_object)
            .map(|c| {
                c.iter()
                    .filter(|(_, flagged)| truthy(flagged))
                    .map(|(name, _)| name.clone())
                    .collect()
            })
            .unwrap_or_default();
        ModerationResult {
            flagged: data
                .get("flagged")
                .map(truthy)
                .unwrap_or(!categories.is_empty()),
            categories,
            category_scores: data
                .get("category_scores")
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default(),
        }
    }

    /// `flagged?`.
    pub fn is_flagged(&self) -> bool {
        self.flagged
    }
}

fn truthy(v: &Value) -> bool {
    !matches!(v, Value::Null | Value::Bool(false))
}

/// The result of screening text or images (`RubyLLM::Moderation`).
#[derive(Debug, Clone)]
pub struct Moderation {
    /// The provider-assigned identifier of the moderation request.
    pub id: Option<String>,
    /// The id of the model that performed the moderation.
    pub model: String,
    /// One verdict per moderated input.
    pub results: Vec<ModerationResult>,
    /// The provider's response body.
    pub raw: Value,
    /// One entry per provider attempt.
    pub usage_entries: Vec<UsageEntry>,
}

impl Moderation {
    /// `flagged?`: whether any input was flagged.
    pub fn is_flagged(&self) -> bool {
        self.results.iter().any(ModerationResult::is_flagged)
    }

    /// The unique names of the categories flagged across all results.
    pub fn flagged_categories(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for c in self.results.iter().flat_map(|r| &r.categories) {
            if !out.contains(c) {
                out.push(c.clone());
            }
        }
        out
    }

    /// Scores across all results, keeping the highest score per category.
    pub fn category_scores(&self) -> Map<String, Value> {
        let mut merged = Map::new();
        for (category, score) in self.results.iter().flat_map(|r| &r.category_scores) {
            let keep = match (merged.get(category).and_then(Value::as_f64), score.as_f64()) {
                (Some(left), Some(right)) => right > left,
                _ => true,
            };
            if keep {
                merged.insert(category.clone(), score.clone());
            }
        }
        merged
    }

    /// Provider-reported usage across every attempt.
    pub fn tokens(&self) -> Tokens {
        Tokens::aggregate(self.usage_entries.iter().map(|e| &e.tokens))
    }

    /// The moderation cost across every attempt.
    pub fn cost(&self) -> Cost {
        let complete = self.usage_entries.iter().all(UsageEntry::cost_available);
        Cost::aggregate(self.usage_entries.iter().map(|e| &e.cost), complete)
    }
}

/// What to moderate: one text, several, or none when only attachments are screened.
#[derive(Debug, Clone, Default)]
pub enum ModerationInput {
    #[default]
    None,
    One(String),
    Many(Vec<String>),
}

impl From<&str> for ModerationInput {
    fn from(s: &str) -> Self {
        ModerationInput::One(s.into())
    }
}

impl From<String> for ModerationInput {
    fn from(s: String) -> Self {
        ModerationInput::One(s)
    }
}

impl From<Vec<String>> for ModerationInput {
    fn from(v: Vec<String>) -> Self {
        ModerationInput::Many(v)
    }
}

impl ModerationInput {
    fn to_value(&self) -> Option<Value> {
        match self {
            ModerationInput::None => None,
            ModerationInput::One(s) => Some(json!(s)),
            ModerationInput::Many(v) => Some(json!(v)),
        }
    }
}

/// Options for [`moderate`], the keyword arguments of `Moderation.moderate`.
#[derive(Default)]
pub struct ModerateOptions<'a> {
    /// `model:`, defaulting to `config.default_moderation_model`.
    pub model: Option<&'a str>,
    pub provider: Option<&'a str>,
    pub assume_model_exists: bool,
    /// `with:`: images to screen alongside (or instead of) the text.
    pub with: Vec<Attachment>,
    /// `provider_options:`: merged into the request as-is.
    pub provider_options: Value,
    /// `context:`: use this configuration instead of the global one.
    pub config: Option<Arc<Config>>,
    /// `metadata:`: added to the `moderation.rust_llm` event payload, never sent to the provider.
    pub metadata: Option<Value>,
    /// `owner:`: who the usage is attributed to, such as a user; wins over
    /// [`crate::accounting::with_usage_owner`].
    pub owner: Option<crate::accounting::UsageOwner>,
}

/// `RubyLLM.moderate(input, model:, with:, provider:, assume_model_exists:, provider_options:,
/// metadata:)`, inside a `moderation.rust_llm` event.
pub async fn moderate(
    input: impl Into<ModerationInput>,
    options: ModerateOptions<'_>,
) -> Result<Moderation> {
    let input = input.into();
    if matches!(input, ModerationInput::None) && options.with.is_empty() {
        return Err(Error::Argument(
            "must provide input text, image attachment, or both".into(),
        ));
    }
    let config = options.config.clone().unwrap_or_else(crate::config);
    let model_id = options
        .model
        .unwrap_or(&config.default_moderation_model)
        .to_string();
    let (model, provider) =
        resolve_model(&model_id, options.provider, options.assume_model_exists)?;
    let owner = options.owner.clone();
    let mut event = crate::instrumentation::Event::start(&config, "moderation.rust_llm", || {
        let empty = Tokens::default();
        crate::instrumentation::payload([
            ("provider", provider.slug().into()),
            ("provider_class", provider.display().into()),
            ("model", model.id.clone().into()),
            ("input", input.to_value().unwrap_or(Value::Null)),
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
    let result = event
        .instrument(crate::accounting::owned_by(
            owner,
            moderate_inner(input, options, config.clone(), model, provider),
        ))
        .await;
    if let Ok(m) = &result {
        crate::accounting::report(&config, &m.usage_entries).await;
        event.set(
            "result",
            || json!({ "id": m.id, "model": m.model, "results": m.raw.get("results") }),
        );
        event.set("flagged", || m.is_flagged().into());
        event.set("tokens", || crate::instrumentation::tokens_h(&m.tokens()));
        event.set("cost", || crate::instrumentation::cost_h(&m.cost()));
    }
    event.finish(result.as_ref().err());
    result
}

async fn moderate_inner(
    input: ModerationInput,
    options: ModerateOptions<'_>,
    config: Arc<Config>,
    model: crate::model::Model,
    provider: crate::providers::Provider,
) -> Result<Moderation> {
    let ModerateOptions {
        mut with,
        provider_options,
        ..
    } = options;
    provider.ensure_configured(&config)?;
    // `ChatCompletions::Moderation` reaches every protocol built on Chat Completions.
    if !matches!(
        provider.resolve_protocol(None, &model, &config),
        Ok(ProtocolName::ChatCompletions | ProtocolName::Responses)
    ) {
        return Err(Error::Api(
            format!("{} doesn't support moderation", provider.display()),
            None,
        ));
    }
    let connection = Connection::new(provider, config.clone())?;
    for a in with.iter_mut().filter(|a| !a.is_url()) {
        a.load(connection.client()).await?;
    }
    let payload = render_payload(&input, &model.id, &with, &provider_options)?;

    // `track_usage(:moderation)`: one entry per HTTP attempt.
    let mut retried: Vec<Tokens> = Vec::new();
    let mut on_attempt = |previous: Option<&Error>| {
        if let Some(e) = previous {
            retried.push(failure_tokens(e, None));
        }
    };
    let raw = connection
        .post("moderations", &payload, &[], &mut on_attempt)
        .await?;
    let mut moderation = parse_response(&raw.body, &model.id)?;
    let entry = |status, tokens: Tokens| UsageEntry {
        id: UsageEntry::next_id(),
        owner: crate::accounting::usage_owner(),
        operation: Operation::Moderation,
        provider: provider.slug().into(),
        model: model.id.clone(),
        status,
        cost: Cost::new(&tokens, Some(&model), Tier::Standard),
        tokens,
    };
    let mut entries: Vec<UsageEntry> = retried
        .into_iter()
        .map(|t| entry(UsageStatus::Failed, t))
        .collect();
    // The response reports no usage, so the billed attempt carries empty tokens, as in Ruby.
    entries.push(entry(UsageStatus::Succeeded, Tokens::default()));
    moderation.usage_entries = entries;
    Ok(moderation)
}

/// `render_moderation_payload`.
fn render_payload(
    input: &ModerationInput,
    model: &str,
    with: &[Attachment],
    provider_options: &Value,
) -> Result<Value> {
    let input = if with.is_empty() {
        input.to_value().unwrap_or(Value::Null)
    } else {
        let mut parts = Vec::new();
        if let Some(text) = input.to_value() {
            parts.push(json!({ "type": "text", "text": text }));
        }
        for a in with {
            if a.kind() != AttachmentType::Image {
                return Err(Error::UnsupportedAttachment(unsupported(&a.mime_type)));
            }
            parts
                .push(json!({ "type": "image_url", "image_url": { "url": a.url_or_data_uri()? } }));
        }
        Value::Array(parts)
    };
    let mut payload = json!({ "model": model, "input": input });
    if let (Some(p), Some(options)) = (payload.as_object_mut(), provider_options.as_object()) {
        p.extend(options.clone());
    }
    Ok(payload)
}

/// `parse_moderation_response`.
fn parse_response(data: &Value, model: &str) -> Result<Moderation> {
    if let Some(message) = data.pointer("/error/message").and_then(Value::as_str) {
        return Err(Error::Api(message.into(), None));
    }
    let results = match data.get("results") {
        Some(Value::Array(results)) => results.iter().map(ModerationResult::from_h).collect(),
        Some(Value::Null) | None => Vec::new(),
        Some(one) => vec![ModerationResult::from_h(one)],
    };
    Ok(Moderation {
        id: data.get("id").and_then(Value::as_str).map(str::to_string),
        model: model.to_string(),
        results,
        raw: data.clone(),
        usage_entries: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flagged_categories_come_from_true_entries_and_flagged_defaults_to_any() {
        let result =
            ModerationResult::from_h(&json!({ "categories": { "hate": false, "violence": true } }));
        assert_eq!(result.categories, vec!["violence"]);
        assert!(result.is_flagged());
    }

    #[test]
    fn category_scores_keep_the_highest_score_across_results() {
        let moderation = parse_response(
            &json!({ "id": "m", "results": [
                { "flagged": false, "categories": {}, "category_scores": { "hate": 0.1, "violence": 0.5 } },
                { "flagged": true, "categories": { "hate": true }, "category_scores": { "hate": 0.9, "violence": 0.2 } }
            ]}),
            "omni-moderation-latest",
        )
        .unwrap();
        assert_eq!(
            moderation.category_scores(),
            json!({ "hate": 0.9, "violence": 0.5 })
                .as_object()
                .unwrap()
                .clone()
        );
        assert!(moderation.is_flagged());
        assert_eq!(moderation.flagged_categories(), vec!["hate"]);
    }

    // spec: protocols/chat_completions/moderation_spec.rb:6 preserves the raw response alongside normalized verdicts
    #[test]
    fn preserves_the_raw_response_alongside_normalized_verdicts() {
        let body = json!({ "id": "moderation-request", "results": [{ "flagged": false, "categories": {},
                                                                    "category_scores": { "violence": 0.02 } }] });
        let result =
            parse_response(&body, &crate::Config::default().default_moderation_model).unwrap();
        assert_eq!(result.raw, body);
        assert!(!result.is_flagged());
        assert_eq!(
            result.category_scores(),
            json!({ "violence": 0.02 }).as_object().unwrap().clone()
        );
    }

    // spec: protocols/chat_completions/moderation_spec.rb:74 .render_moderation_payload > rejects non-image attachments
    #[test]
    fn rejects_non_image_attachments() {
        let attachment = Attachment::from_bytes(b"hello".to_vec(), "note.txt", None);
        let err = render_payload(
            &ModerationInput::None,
            "omni-moderation-latest",
            &[attachment],
            &Value::Null,
        )
        .unwrap_err();
        assert!(
            matches!(&err, Error::UnsupportedAttachment(m) if m.contains("text/plain")),
            "{err:?}"
        );
    }
}
