//! Port of `lib/ruby_llm/image.rb` (`RubyLLM.paint`), `Protocol#paint`/`post_image`/`parse_image_responses`
//! in `lib/ruby_llm/protocol.rb`, and the image seams in `protocols/chat_completions/images.rb`
//! (OpenAI Images API), `protocols/gemini/images.rb`, `providers/xai/images.rb`, and
//! `providers/openrouter/images.rb`.
//!
//! ```ruby
//! image = RubyLLM.paint("a sunset over mountains in watercolor style")
//! image.save("sunset.png")
//! ```

use std::sync::{Arc, LazyLock};

use base64::Engine;
use regex::Regex;
use serde_json::{Map, Value, json};

use crate::attachment::{Attachment, AttachmentType};
use crate::chat::{failure_tokens, resolve_model};
use crate::config::Config;
use crate::cost::Cost;
use crate::error::{Error, Result, error_for_status};
use crate::message::{Operation, UsageEntry, UsageStatus};
use crate::model::Model;
use crate::models;
use crate::protocols::{anthropic::unsupported, chat_completions, deep_merge, gemini, int};
use crate::providers::Provider;
use crate::tokens::Tokens;
use crate::transport::Connection;

/// A generated or edited image (`RubyLLM::Image`). Save it with [`Image::save`] or read its bytes
/// with [`Image::to_blob`]; both handle hosted URLs and inline data.
#[derive(Debug, Clone)]
pub struct Image {
    /// The URL of the hosted image, for providers that return one.
    pub url: Option<String>,
    /// The Base64-encoded image data, for providers that return the image inline.
    pub data: Option<String>,
    /// The MIME type of the image data, such as `"image/png"`.
    pub mime_type: Option<String>,
    /// The provider's rewritten version of the prompt, when reported.
    pub revised_prompt: Option<String>,
    /// The id of the model that generated the image.
    pub model: String,
    /// One entry per provider attempt. Only the first image of a request carries them, so the
    /// call is billed once (`Accounting::Usage::Tracker#succeed`).
    pub usage_entries: Vec<UsageEntry>,
    raw_usage: Value,
    config: Option<Arc<Config>>,
}

/// `paint` returns one `Image`, or several when a provider generated more than one
/// (`images.size <= 1 ? images.first : images`).
#[derive(Debug, Clone)]
pub enum Images {
    One(Image),
    Many(Vec<Image>),
}

impl Images {
    /// The single image, or the first of several.
    pub fn into_image(self) -> Image {
        match self {
            Images::One(image) => image,
            Images::Many(mut images) => images.remove(0),
        }
    }

    pub fn into_vec(self) -> Vec<Image> {
        match self {
            Images::One(image) => vec![image],
            Images::Many(images) => images,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Images::One(_) => 1,
            Images::Many(images) => images.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Options for [`paint`], the keyword arguments of `Image.paint`.
#[derive(Default)]
pub struct PaintOptions<'a> {
    /// `model:`, defaulting to `config.default_image_model`.
    pub model: Option<&'a str>,
    pub provider: Option<&'a str>,
    pub assume_model_exists: bool,
    pub size: Option<&'a str>,
    /// `count:`: several images in one request, on providers that support it.
    pub count: Option<i64>,
    /// `with:`: source images to edit.
    pub with: Vec<Attachment>,
    /// `mask:`: which parts of the image may change.
    pub mask: Option<Attachment>,
    /// `provider_options:`: merged into the request as-is.
    pub provider_options: Value,
    /// `context:`: use this configuration instead of the global one.
    pub config: Option<Arc<Config>>,
    /// `metadata:`: added to the `image.rust_llm` event payload, never sent to the provider.
    pub metadata: Option<Value>,
}

impl Image {
    /// `Image.new(model:, usage:)`: an image with only accounting data, as protocols build it
    /// before filling in the payload fields.
    pub fn new(model: impl Into<String>, usage: Value) -> Image {
        Image {
            url: None,
            data: None,
            mime_type: None,
            revised_prompt: None,
            model: model.into(),
            usage_entries: Vec::new(),
            raw_usage: usage,
            config: None,
        }
    }

    /// `base64?`: whether the image holds inline Base64 data.
    pub fn is_base64(&self) -> bool {
        self.data.is_some()
    }

    /// `Image#config`: the configuration of the context that generated the image, whose
    /// connection settings (`http_proxy`, `request_timeout`) the download uses; the global one otherwise.
    pub fn config(&self) -> Arc<Config> {
        self.config.clone().unwrap_or_else(crate::config)
    }

    /// The raw image bytes, decoding `data` when present or downloading `url` otherwise.
    pub async fn to_blob(&self) -> Result<Vec<u8>> {
        if let Some(data) = &self.data {
            return base64::engine::general_purpose::STANDARD
                .decode(data.trim())
                .map_err(|e| Error::Argument(format!("image data is not valid Base64: {e}")));
        }
        let url = self
            .url
            .as_deref()
            .ok_or_else(|| Error::Argument("image has neither data nor a url".into()))?;
        let client = crate::transport::basic(&self.config())?;
        let response = client
            .get(url)
            .send()
            .await
            .map_err(|e| Error::ConnectionFailed(e.to_string()))?;
        let status = response.status();
        let bytes = response
            .bytes()
            .await
            .map_err(|e| Error::ConnectionFailed(e.to_string()))?;
        if !status.is_success() {
            return Err(error_for_status(
                status.as_u16(),
                &String::from_utf8_lossy(&bytes),
            ));
        }
        Ok(bytes.to_vec())
    }

    /// Writes the image to `path` and returns `path` as given.
    pub async fn save<P: AsRef<std::path::Path>>(&self, path: P) -> Result<P> {
        tokio::fs::write(path.as_ref(), self.to_blob().await?).await?;
        Ok(path)
    }

    /// Usage across every provider attempt, or the usage this image reported.
    pub fn tokens(&self) -> Tokens {
        if !self.usage_entries.is_empty() {
            return Tokens::aggregate(self.usage_entries.iter().map(|e| &e.tokens));
        }
        let usage = &self.raw_usage;
        Tokens {
            input: int(usage.get("input_tokens")),
            output: int(usage.get("output_tokens")),
            reported_cost: usage.get("cost").and_then(Value::as_f64),
            ..Default::default()
        }
    }

    /// Cost across every provider attempt, using reported prices when available and registry
    /// image pricing otherwise.
    pub fn cost(&self) -> Cost {
        if !self.usage_entries.is_empty() {
            let complete = self.usage_entries.iter().all(UsageEntry::cost_available);
            return Cost::aggregate(self.usage_entries.iter().map(|e| &e.cost), complete);
        }
        Cost::images(
            &self.tokens(),
            self.model_info().as_ref(),
            self.raw_usage.get("input_tokens_details"),
        )
    }

    /// The registry model for `model`, or `None` when it is not in the registry.
    pub fn model_info(&self) -> Option<Model> {
        models::models().find(&self.model, None).ok()
    }
}

/// `RubyLLM.paint(prompt, model:, provider:, size:, count:, with:, mask:, provider_options:)`,
/// inside an `image.rust_llm` event.
pub async fn paint(prompt: &str, options: PaintOptions<'_>) -> Result<Images> {
    let config = options.config.clone().unwrap_or_else(crate::config);
    let model_id = options
        .model
        .unwrap_or(&config.default_image_model)
        .to_string();
    let (model, provider) =
        resolve_model(&model_id, options.provider, options.assume_model_exists)?;
    let mut event = crate::instrumentation::Event::start(&config, "image.rust_llm", || {
        let empty = Tokens::default();
        crate::instrumentation::payload([
            ("provider", provider.slug().into()),
            ("provider_class", provider.display().into()),
            ("model", model.id.clone().into()),
            ("prompt", prompt.into()),
            ("size", options.size.into()),
            ("count", options.count.into()),
            ("provider_options", options.provider_options.clone()),
            (
                "metadata",
                crate::instrumentation::metadata(&options.metadata),
            ),
            ("tokens", crate::instrumentation::tokens_h(&empty)),
            (
                "cost",
                crate::instrumentation::cost_h(&Cost::images(&empty, Some(&model), None)),
            ),
        ])
    });
    let result = tracing::Instrument::instrument(paint_inner(prompt, options), event.span()).await;
    if let Ok(images) = &result {
        let all: Vec<&Image> = match images {
            Images::One(i) => vec![i],
            Images::Many(v) => v.iter().collect(),
        };
        if let Some(billed) = all.first() {
            crate::instrumentation::usages(&config, &billed.usage_entries);
        }
        event.set("result", || serde_json::json!(all.iter().map(|i| serde_json::json!({ "url": i.url, "mime_type": i.mime_type, "model": i.model, "revised_prompt": i.revised_prompt })).collect::<Vec<_>>()));
        event.set("response_model", || {
            all.first().map(|i| i.model.clone()).into()
        });
        let tokens: Vec<Tokens> = all.iter().map(|i| i.tokens()).collect();
        event.set("tokens", || {
            crate::instrumentation::tokens_h(&Tokens::aggregate(tokens.iter()))
        });
        let costs: Vec<Cost> = all.iter().map(|i| i.cost()).collect();
        let complete = costs.iter().all(|c| c.total().is_some());
        event.set("cost", || {
            crate::instrumentation::cost_h(&Cost::aggregate(costs.iter(), complete))
        });
    }
    event.finish(result.as_ref().err());
    result
}

async fn paint_inner(prompt: &str, options: PaintOptions<'_>) -> Result<Images> {
    let config = options.config.clone().unwrap_or_else(crate::config);
    let model_id = options
        .model
        .unwrap_or(&config.default_image_model)
        .to_string();
    let (model, provider) =
        resolve_model(&model_id, options.provider, options.assume_model_exists)?;
    provider.ensure_configured(&config)?;
    let connection = Connection::new(provider, config.clone())?;
    let family = Family::for_provider(provider)?;

    let PaintOptions {
        size,
        count,
        mut with,
        mut mask,
        provider_options,
        ..
    } = options;
    family.validate(&model.id, &with, mask.as_ref())?;
    for a in with.iter_mut().chain(mask.iter_mut()) {
        if family.loads(a) {
            a.load(connection.client()).await?;
        }
    }
    let (path, payload) = family.render(
        prompt,
        &model.id,
        size,
        count,
        &with,
        mask.as_ref(),
        &provider_options,
    )?;

    // `track_usage(:image)`: one entry per HTTP attempt.
    let mut retried: Vec<Tokens> = Vec::new();
    let mut on_attempt = |previous: Option<&Error>| {
        if let Some(e) = previous {
            retried.push(failure_tokens(e, None));
        }
    };
    let result = connection.post(&path, &payload, &[], &mut on_attempt).await;
    let entry = |status, tokens: Tokens, cost: Option<Cost>| UsageEntry {
        id: UsageEntry::next_id(),
        operation: Operation::Image,
        provider: provider.slug().into(),
        model: model.id.clone(),
        status,
        cost: cost
            .filter(|c| c.total().is_some())
            .unwrap_or_else(|| Cost::images(&tokens, Some(&model), None)),
        tokens,
    };
    let mut entries: Vec<UsageEntry> = retried
        .into_iter()
        .map(|t| entry(UsageStatus::Failed, t, None))
        .collect();
    // A failed paint has no result to attach its entries to; Ruby only reports them to
    // instrumentation, which this port does not have.
    let mut images = match family {
        Family::Mistral => mistral_images(&connection, &result?.body, &model.id).await?,
        _ => result.and_then(|raw| family.parse(&raw.body, &model.id))?,
    };
    let billed = &images[0];
    entries.push(entry(
        UsageStatus::Succeeded,
        billed.tokens(),
        Some(billed.cost()),
    ));
    images[0].usage_entries = entries;
    for image in &mut images {
        image.config = Some(config.clone());
    }
    Ok(if images.len() == 1 {
        Images::One(images.remove(0))
    } else {
        Images::Many(images)
    })
}

/// `Conversations::Images#parse_image_responses`: each generated file downloaded from Mistral's
/// Files API, typed from its bytes; only the first image carries the usage.
async fn mistral_images(connection: &Connection, data: &Value, model: &str) -> Result<Vec<Image>> {
    let (files, usage) = crate::protocols::mistral::parse_image_files(data)?;
    let mut images = Vec::new();
    for (index, id) in files.iter().enumerate() {
        let bytes = crate::files::download_file(connection, Provider::Mistral, id).await?;
        let mut image = Image::new(model, if index == 0 { usage.clone() } else { json!({}) });
        image.mime_type = Some(crate::attachment::mime_type_for_bytes(&bytes));
        image.data = Some(base64::engine::general_purpose::STANDARD.encode(&bytes));
        images.push(image);
    }
    Ok(images)
}

/// Which image seams a provider's protocol includes.
#[derive(Clone, Copy)]
enum Family {
    /// `ChatCompletions::Images` (OpenAI and every OpenAI-compatible provider).
    OpenAI,
    XAI,
    OpenRouter,
    Gemini,
    /// `Mistral::Conversations::Images`: `Mistral#protocol_for(operation: :paint)` is Conversations.
    Mistral,
}

impl Family {
    fn for_provider(provider: Provider) -> Result<Family> {
        match provider {
            Provider::XAI => Ok(Family::XAI),
            Provider::OpenRouter => Ok(Family::OpenRouter),
            Provider::Gemini => Ok(Family::Gemini),
            Provider::Anthropic => Err(Error::Api(
                "Anthropic doesn't support image generation".into(),
                None,
            )),
            Provider::Mistral => Ok(Family::Mistral),
            _ => Ok(Family::OpenAI),
        }
    }

    /// Whether the attachment's bytes are needed to render it. URLs pass through as references
    /// except on Gemini, which inlines every image.
    fn loads(self, attachment: &Attachment) -> bool {
        matches!(self, Family::Gemini) || !attachment.is_url()
    }

    /// `validate_paint_inputs!`.
    fn validate(self, model: &str, with: &[Attachment], mask: Option<&Attachment>) -> Result<()> {
        match self {
            Family::OpenAI if mask.is_some() && with.is_empty() => Err(Error::Argument(
                "with: is required when mask: is provided".into(),
            )),
            Family::XAI if mask.is_some() => Err(Error::Api(
                "xAI image editing does not support a mask parameter".into(),
                None,
            )),
            Family::OpenRouter if mask.is_some() => {
                Err(Error::UnsupportedAttachment(unsupported("image mask")))
            }
            Family::Gemini if gemini_image_model(model) && mask.is_some() => {
                Err(Error::UnsupportedAttachment(unsupported("image mask")))
            }
            Family::Gemini
                if !gemini_image_model(model) && (!with.is_empty() || mask.is_some()) =>
            {
                Err(Error::UnsupportedAttachment(unsupported("image reference")))
            }
            _ => Ok(()),
        }
    }

    /// `images_url` and `render_image_payload`.
    #[allow(clippy::too_many_arguments)]
    fn render(
        self,
        prompt: &str,
        model: &str,
        size: Option<&str>,
        count: Option<i64>,
        with: &[Attachment],
        mask: Option<&Attachment>,
        provider_options: &Value,
    ) -> Result<(String, Value)> {
        let editing = !with.is_empty() || mask.is_some();
        let options = provider_options.as_object().cloned().unwrap_or_default();
        match self {
            Family::Mistral => Ok((
                "conversations".into(),
                crate::protocols::mistral::render_image_payload(
                    prompt,
                    model,
                    size,
                    count,
                    editing,
                    provider_options,
                )?,
            )),
            Family::OpenAI if editing => {
                if !(model.starts_with("gpt-image") || model.starts_with("chatgpt-image")) {
                    return Err(Error::Argument(format!(
                        "Editing with {model} needs a multipart upload, which rust_llm has not ported yet; \
                         gpt-image and chatgpt-image models take JSON image references"
                    )));
                }
                let mut payload =
                    json!({ "model": model, "prompt": prompt, "n": count.unwrap_or(1) });
                payload["images"] = with
                    .iter()
                    .map(openai_reference)
                    .collect::<Result<Vec<_>>>()?
                    .into();
                if let Some(mask) = mask {
                    payload["mask"] = openai_reference(mask)?;
                }
                if let Some(size) = size {
                    payload["size"] = size.into();
                }
                Ok(("images/edits".into(), merge(payload, options)))
            }
            Family::OpenAI => {
                let payload = json!({ "model": model, "prompt": prompt, "n": count.unwrap_or(1), "size": size });
                Ok(("images/generations".into(), merge(payload, options)))
            }
            Family::XAI => {
                if !editing && let Some(size) = size {
                    tracing::debug!(
                        "Ignoring size {size}. xAI image generation does not support a size parameter."
                    );
                }
                let mut payload = json!({ "model": model, "prompt": prompt });
                if editing {
                    let images = with
                        .iter()
                        .map(|a| Ok(json!({ "type": "image_url", "url": reference_url(a)? })))
                        .collect::<Result<Vec<_>>>()?;
                    payload["images"] = images.into();
                }
                if let Some(count) = count {
                    payload["n"] = count.into();
                }
                let path = if editing {
                    "images/edits"
                } else {
                    "images/generations"
                };
                Ok((path.into(), merge(payload, options)))
            }
            Family::OpenRouter => {
                if let Some(size) = size {
                    tracing::debug!(
                        "Ignoring size {size}. Use aspect_ratio/resolution provider options instead."
                    );
                }
                if count.is_some_and(|c| c > 1) {
                    tracing::debug!(
                        "Ignoring count {count:?}. OpenRouter generates one image per request."
                    );
                }
                let mut payload = json!({ "model": model, "prompt": prompt });
                let references = with
                    .iter()
                    .map(|a| {
                        require_image(a)?;
                        Ok(json!({ "type": "image_url", "image_url": { "url": a.url_or_data_uri()? } }))
                    })
                    .collect::<Result<Vec<_>>>()?;
                if !references.is_empty() {
                    payload["input_references"] = references.into();
                }
                Ok(("images".into(), merge(payload, options)))
            }
            Family::Gemini => {
                let image_model = gemini_image_model(model);
                let mut payload = if image_model {
                    for a in with {
                        require_image(a)?;
                    }
                    let mut generation_config = json!({ "responseModalities": ["TEXT", "IMAGE"] });
                    if let Some(count) = count.filter(|c| *c > 1) {
                        generation_config["candidateCount"] = count.into();
                    }
                    if let Some(image_config) = gemini_image_config(size)? {
                        generation_config["imageConfig"] = image_config;
                    }
                    json!({
                        "contents": [{ "role": "user", "parts": gemini::format_content(Some(prompt), with)? }],
                        "generationConfig": generation_config,
                    })
                } else {
                    if let Some(size) = size {
                        tracing::debug!("Ignoring size {size}. Imagen sizing is not supported.");
                    }
                    json!({ "instances": [{ "prompt": prompt }], "parameters": { "sampleCount": count.unwrap_or(1) } })
                };
                deep_merge(&mut payload, &Value::Object(options));
                let action = if image_model {
                    "generateContent"
                } else {
                    "predict"
                };
                Ok((format!("models/{model}:{action}"), payload))
            }
        }
    }

    /// `parse_image_responses`.
    fn parse(self, data: &Value, model: &str) -> Result<Vec<Image>> {
        let entries = || {
            data.get("data")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default()
        };
        let usage = || data.get("usage").cloned().unwrap_or_else(|| json!({}));
        match self {
            Family::OpenAI | Family::XAI => {
                let entries = entries();
                if entries.is_empty() {
                    let name = if matches!(self, Family::XAI) {
                        "xAI"
                    } else {
                        "OpenAI"
                    };
                    return Err(Error::Api(
                        format!("Unexpected response format from {name} image API"),
                        None,
                    ));
                }
                Ok(entries
                    .iter()
                    .enumerate()
                    .map(|(i, entry)| {
                        let mut image = Image::new(model, if i == 0 { usage() } else { json!({}) });
                        image.url = str_at(entry, "url");
                        image.data = str_at(entry, "b64_json");
                        image.mime_type = Some(match self {
                            Family::XAI => {
                                str_at(entry, "mime_type").unwrap_or_else(|| "image/png".into())
                            }
                            _ => "image/png".into(),
                        });
                        if matches!(self, Family::OpenAI) {
                            image.revised_prompt = str_at(entry, "revised_prompt");
                        }
                        image
                    })
                    .collect())
            }
            Family::OpenRouter => {
                let entry = entries().into_iter().next().ok_or_else(|| {
                    Error::Api(
                        "Unexpected response format from OpenRouter image API".into(),
                        None,
                    )
                })?;
                let raw = usage();
                let mut usage = Map::new();
                for (key, value) in [
                    ("input_tokens", raw.get("prompt_tokens").cloned()),
                    ("output_tokens", raw.get("completion_tokens").cloned()),
                    (
                        "cost",
                        chat_completions::reported_cost(Provider::OpenRouter, &raw)
                            .map(Value::from),
                    ),
                ] {
                    if let Some(value) = value.filter(|v| !v.is_null()) {
                        usage.insert(key.into(), value);
                    }
                }
                let mut image = Image::new(model, Value::Object(usage));
                image.data = str_at(&entry, "b64_json");
                image.mime_type =
                    Some(str_at(&entry, "media_type").unwrap_or_else(|| "image/png".into()));
                Ok(vec![image])
            }
            Family::Gemini if gemini_image_model(model) => {
                let parts: Vec<Value> = data
                    .get("candidates")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(|candidate| {
                        candidate
                            .pointer("/content/parts")?
                            .as_array()?
                            .iter()
                            .filter_map(|p| p.get("inlineData"))
                            .find(|inline| {
                                let mime = inline.get("mimeType").and_then(Value::as_str);
                                mime.is_none_or(|m| m.starts_with("image/"))
                                    && inline.get("data").is_some_and(|d| !d.is_null())
                            })
                            .cloned()
                    })
                    .collect();
                if parts.is_empty() {
                    return Err(Error::Api(
                        "Unexpected response format from Gemini image generation API".into(),
                        None,
                    ));
                }
                let response_model =
                    str_at(data, "modelVersion").unwrap_or_else(|| model.to_string());
                Ok(parts
                    .iter()
                    .enumerate()
                    .map(|(i, part)| {
                        let mut image = Image::new(
                            &response_model,
                            if i == 0 {
                                gemini_usage(data)
                            } else {
                                json!({})
                            },
                        );
                        image.data = str_at(part, "data");
                        image.mime_type =
                            Some(str_at(part, "mimeType").unwrap_or_else(|| "image/png".into()));
                        image
                    })
                    .collect())
            }
            Family::Gemini => {
                let predictions: Vec<&Value> = data
                    .get("predictions")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter(|p| p.get("bytesBase64Encoded").is_some_and(|b| !b.is_null()))
                    .collect();
                if predictions.is_empty() {
                    return Err(Error::Api(
                        "Unexpected response format from Gemini image generation API".into(),
                        None,
                    ));
                }
                Ok(predictions
                    .into_iter()
                    .map(|p| {
                        let mut image = Image::new(model, json!({}));
                        image.data = str_at(p, "bytesBase64Encoded");
                        image.mime_type =
                            Some(str_at(p, "mimeType").unwrap_or_else(|| "image/png".into()));
                        image
                    })
                    .collect())
            }
            // Mistral's images need a download per file, so `paint` parses them (`mistral_images`).
            Family::Mistral => Err(Error::Api(
                "Mistral images are parsed with their downloads".into(),
                None,
            )),
        }
    }
}

fn str_at(v: &Value, key: &str) -> Option<String> {
    v.get(key).and_then(Value::as_str).map(str::to_string)
}

/// Ruby's `Hash#merge`: provider options replace top-level keys.
fn merge(mut payload: Value, options: Map<String, Value>) -> Value {
    if let Some(object) = payload.as_object_mut() {
        object.extend(options);
    }
    payload
}

fn require_image(a: &Attachment) -> Result<()> {
    if a.kind() == AttachmentType::Image {
        Ok(())
    } else {
        Err(Error::UnsupportedAttachment(unsupported(&a.mime_type)))
    }
}

/// `ChatCompletions::Images#build_image_reference` / `XAI::Images#image_reference_url`: URLs
/// pass through for the provider to fetch; local images become data URIs.
fn reference_url(a: &Attachment) -> Result<String> {
    if let Some(url) = a.url() {
        return Ok(url.to_string());
    }
    require_image(a)?;
    a.for_llm()
}

fn openai_reference(a: &Attachment) -> Result<Value> {
    Ok(json!({ "image_url": reference_url(a)? }))
}

/// `Gemini::Images#gemini_image_model?`: Gemini image models answer on `generateContent`;
/// everything else is Imagen on `predict`.
fn gemini_image_model(model: &str) -> bool {
    let id = model.to_lowercase();
    id.starts_with("nano-banana")
        || id.starts_with("nanobanana")
        || (id.starts_with("gemini-") && id.contains("-image"))
}

const GEMINI_IMAGE_SIZES: &[&str] = &["512", "512P", "512PX", "1K", "2K", "4K"];
static ASPECT_RATIO: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\A\d+:\d+\z").unwrap()); // constant regex
static PIXEL_DIMENSIONS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\A(\d+)\s*[x×]\s*(\d+)\z").unwrap()); // constant regex

/// `Gemini::Images#build_image_config`: Gemini sizes by aspect ratio and resolution tier, so
/// `WxH` becomes the ratio it reduces to.
fn gemini_image_config(size: Option<&str>) -> Result<Option<Value>> {
    let value = size.unwrap_or("").trim();
    if value.is_empty() {
        return Ok(None);
    }
    if GEMINI_IMAGE_SIZES.contains(&value.to_uppercase().as_str()) {
        return Ok(Some(json!({ "imageSize": value.to_uppercase() })));
    }
    if ASPECT_RATIO.is_match(value) {
        return Ok(Some(json!({ "aspectRatio": value })));
    }
    let unsupported = |s: &str| {
        Error::Argument(format!(
            "Gemini cannot generate an image of size {s:?}. Give pixel dimensions such as \"1024x1024\", an aspect \
             ratio such as \"16:9\", or a resolution such as {}.",
            GEMINI_IMAGE_SIZES.join(", ")
        ))
    };
    let captures = PIXEL_DIMENSIONS
        .captures(value)
        .ok_or_else(|| unsupported(size.unwrap_or("")))?;
    let (width, height): (u64, u64) = match (captures[1].parse(), captures[2].parse()) {
        (Ok(w), Ok(h)) if w > 0 && h > 0 => (w, h),
        _ => return Err(unsupported(&format!("{}x{}", &captures[1], &captures[2]))),
    };
    let divisor = gcd(width, height);
    Ok(Some(
        json!({ "aspectRatio": format!("{}:{}", width / divisor, height / divisor) }),
    ))
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// `Gemini::Images#gemini_image_usage`.
fn gemini_usage(data: &Value) -> Value {
    let meta = data.get("usageMetadata");
    let mut usage = Map::new();
    if let Some(prompt) = int(meta.and_then(|m| m.get("promptTokenCount"))) {
        let cached = int(meta.and_then(|m| m.get("cachedContentTokenCount"))).unwrap_or(0);
        usage.insert("input_tokens".into(), (prompt - cached).max(0).into());
    }
    let candidates = int(meta.and_then(|m| m.get("candidatesTokenCount"))).unwrap_or(0);
    let thoughts = int(meta.and_then(|m| m.get("thoughtsTokenCount"))).unwrap_or(0);
    usage.insert("output_tokens".into(), (candidates + thoughts).into());
    Value::Object(usage)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemini_reduces_pixel_sizes_to_aspect_ratios() {
        assert_eq!(
            gemini_image_config(Some("1792x1024")).unwrap(),
            Some(json!({ "aspectRatio": "7:4" }))
        );
        assert_eq!(
            gemini_image_config(Some("2k")).unwrap(),
            Some(json!({ "imageSize": "2K" }))
        );
        assert_eq!(
            gemini_image_config(Some("16:9")).unwrap(),
            Some(json!({ "aspectRatio": "16:9" }))
        );
        assert_eq!(gemini_image_config(None).unwrap(), None);
        assert!(matches!(
            gemini_image_config(Some("huge")),
            Err(Error::Argument(_))
        ));
    }

    #[test]
    fn gemini_image_models_are_told_apart_from_imagen() {
        assert!(gemini_image_model("gemini-3.1-flash-lite-image"));
        assert!(gemini_image_model("nano-banana-pro"));
        assert!(!gemini_image_model("imagen-4.0-generate-001"));
    }
}
