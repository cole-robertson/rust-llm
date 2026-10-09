//! Port of `lib/ruby_llm/ocr.rb` (`RubyLLM.ocr`), `Protocol#ocr`, and
//! `lib/ruby_llm/providers/mistral/ocr.rb`, the only OCR seam among the providers RustLLM supports.
//!
//! ```ruby
//! ocr = RubyLLM.ocr("contract.pdf")
//! ocr.markdown             # => "# Contract\n\n..."
//! ```

use std::sync::Arc;

use serde_json::{Value, json};

use crate::attachment::{Attachment, AttachmentType};
use crate::chat::{failure_tokens, resolve_model};
use crate::config::Config;
use crate::cost::{Cost, Tier};
use crate::error::{Error, Result};
use crate::message::{Operation, UsageEntry, UsageStatus};
use crate::providers::Provider;
use crate::tokens::Tokens;
use crate::transport::Connection;

/// One page of an OCR result (`RubyLLM::OCR::Page`).
#[derive(Debug, Clone, PartialEq)]
pub struct Page {
    /// The zero-based page index.
    pub index: i64,
    pub markdown: Option<String>,
    /// The images the provider reports for the page, or `None`.
    pub images: Option<Value>,
    /// The tables the provider reports for the page, or `None`.
    pub tables: Option<Value>,
    /// The provider's unmodified page hash.
    pub raw: Value,
}

/// The text a document AI model extracted from a file (`RubyLLM::OCR`).
#[derive(Debug, Clone)]
pub struct Ocr {
    pub pages: Vec<Page>,
    /// The id of the model that performed the OCR.
    pub model: String,
    /// The provider's usage block, such as `pages_processed`, or `None`.
    pub usage: Option<Value>,
    /// The provider's raw response.
    pub raw: Value,
    /// One entry per provider attempt.
    pub usage_entries: Vec<UsageEntry>,
}

impl Ocr {
    /// `OCR.new(pages:, model:, usage:, raw:)`: page hashes become [`Page`]s, indexed by position
    /// when the provider omits `index`.
    pub fn new(pages: &[Value], model: impl Into<String>, usage: Option<Value>, raw: Value) -> Ocr {
        let pages = pages
            .iter()
            .enumerate()
            .map(|(position, page)| Page {
                index: page
                    .get("index")
                    .and_then(Value::as_i64)
                    .unwrap_or(position as i64),
                markdown: page
                    .get("markdown")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                images: page.get("images").filter(|v| !v.is_null()).cloned(),
                tables: page.get("tables").filter(|v| !v.is_null()).cloned(),
                raw: page.clone(),
            })
            .collect();
        Ocr {
            pages,
            model: model.into(),
            usage,
            raw,
            usage_entries: Vec::new(),
        }
    }

    /// The markdown of every page, joined with blank lines.
    pub fn markdown(&self) -> String {
        self.pages
            .iter()
            .filter_map(|p| p.markdown.as_deref())
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// Provider-reported usage across every attempt.
    pub fn tokens(&self) -> Tokens {
        Tokens::aggregate(self.usage_entries.iter().map(|e| &e.tokens))
    }

    /// The OCR cost across every attempt.
    pub fn cost(&self) -> Cost {
        let complete = self.usage_entries.iter().all(UsageEntry::cost_available);
        Cost::aggregate(self.usage_entries.iter().map(|e| &e.cost), complete)
    }
}

/// Options for [`ocr`], the keyword arguments of `OCR.ocr`.
#[derive(Default)]
pub struct OcrOptions<'a> {
    /// `model:`, defaulting to `config.default_ocr_model`.
    pub model: Option<&'a str>,
    pub provider: Option<&'a str>,
    pub assume_model_exists: bool,
    /// `pages:`: zero-based page indexes to read.
    pub pages: Option<Vec<i64>>,
    /// `provider_options:`: merged into the request in the provider's vocabulary.
    pub provider_options: Value,
    /// `context:`: use this configuration instead of the global one.
    pub config: Option<Arc<Config>>,
    /// `metadata:`: added to the `ocr.rust_llm` event payload, never sent to the provider.
    pub metadata: Option<Value>,
    /// `owner:`: who the usage is attributed to, such as a user; wins over
    /// [`crate::accounting::with_usage_owner`].
    pub owner: Option<crate::accounting::UsageOwner>,
}

/// `RubyLLM.ocr(file, model:, provider:, pages:, provider_options:, metadata:)`, inside an
/// `ocr.rust_llm` event. `file` is a path, URL, or [`Attachment`].
pub async fn ocr(file: impl Into<Attachment>, options: OcrOptions<'_>) -> Result<Ocr> {
    let config = options.config.clone().unwrap_or_else(crate::config);
    let model_id = options
        .model
        .unwrap_or(&config.default_ocr_model)
        .to_string();
    let (model, provider) =
        resolve_model(&model_id, options.provider, options.assume_model_exists)?;
    let owner = options.owner.clone();
    let mut event = crate::instrumentation::Event::start(&config, "ocr.rust_llm", || {
        crate::instrumentation::payload([
            ("provider", provider.slug().into()),
            ("provider_class", provider.display().into()),
            ("model", model.id.clone().into()),
            ("pages", json!(options.pages)),
            ("provider_options", options.provider_options.clone()),
            (
                "metadata",
                crate::instrumentation::metadata(&options.metadata),
            ),
        ])
    });
    let result = tracing::Instrument::instrument(
        crate::accounting::owned_by(
            owner,
            ocr_inner(file.into(), options, config.clone(), model, provider),
        ),
        event.span(),
    )
    .await;
    if let Ok(r) = &result {
        crate::accounting::report(&config, &r.usage_entries).await;
        event.set(
            "result",
            || json!({ "model": r.model, "pages": r.pages.len() }),
        );
        event.set("response_model", || r.model.clone().into());
    }
    event.finish(result.as_ref().err());
    result
}

async fn ocr_inner(
    file: Attachment,
    options: OcrOptions<'_>,
    config: Arc<Config>,
    model: crate::model::Model,
    provider: Provider,
) -> Result<Ocr> {
    let OcrOptions {
        pages,
        provider_options,
        ..
    } = options;
    provider.ensure_configured(&config)?;
    if provider != Provider::Mistral {
        return Err(Error::Api(
            format!("{} doesn't support OCR", provider.display()),
            None,
        ));
    }
    let connection = Connection::new(provider, config.clone())?;
    let mut attachment = file;
    if !attachment.is_url() {
        attachment.load(connection.client()).await?;
    }
    let payload = render_payload(&attachment, &model.id, pages.as_deref(), &provider_options)?;

    // `track_usage(:ocr)`: one entry per HTTP attempt.
    let mut retried: Vec<Tokens> = Vec::new();
    let mut on_attempt = |previous: Option<&Error>| {
        if let Some(e) = previous {
            retried.push(failure_tokens(e, None));
        }
    };
    let raw = connection
        .post("ocr", &payload, &[], &mut on_attempt)
        .await?;
    let data = raw.body;
    let pages: Vec<Value> = data
        .get("pages")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let response_model = data
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or(&model.id)
        .to_string();
    let usage = data.get("usage_info").filter(|v| !v.is_null()).cloned();
    let mut result = Ocr::new(&pages, response_model, usage, data);
    let entry = |status, tokens: Tokens| UsageEntry {
        id: UsageEntry::next_id(),
        owner: crate::accounting::usage_owner(),
        operation: Operation::Ocr,
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
    // OCR responses report pages, not tokens, so the billed attempt carries empty tokens as in Ruby.
    entries.push(entry(UsageStatus::Succeeded, Tokens::default()));
    result.usage_entries = entries;
    Ok(result)
}

/// `Mistral::OCR#render_ocr_payload`: remote files go as URLs, local ones as data URIs; images
/// use the `image_url` document variant.
fn render_payload(
    attachment: &Attachment,
    model: &str,
    pages: Option<&[i64]>,
    provider_options: &Value,
) -> Result<Value> {
    let reference = attachment.url_or_data_uri()?;
    let document = if attachment.kind() == AttachmentType::Image {
        json!({ "type": "image_url", "image_url": reference })
    } else {
        json!({ "type": "document_url", "document_url": reference })
    };
    let mut payload = json!({ "model": model, "document": document });
    if let Some(pages) = pages {
        payload["pages"] = json!(pages);
    }
    if let (Some(p), Some(options)) = (payload.as_object_mut(), provider_options.as_object()) {
        p.extend(options.clone());
    }
    Ok(payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ocr_spec.rb: "continues converting hash pages with their original metadata".
    #[test]
    fn pages_without_an_index_take_their_position_and_keep_raw() {
        let raw = json!({ "markdown": "# Ruby", "images": [], "tables": [] });
        let ocr = Ocr::new(
            std::slice::from_ref(&raw),
            "mistral-ocr-latest",
            None,
            Value::Null,
        );
        assert_eq!(ocr.pages[0].index, 0);
        assert_eq!(ocr.pages[0].markdown.as_deref(), Some("# Ruby"));
        assert_eq!(ocr.pages[0].images, Some(json!([])));
        assert_eq!(ocr.pages[0].raw, raw);
        assert_eq!(ocr.markdown(), "# Ruby");
    }
}
