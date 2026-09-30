//! Port of `lib/ruby_llm/transcription.rb` (`RubyLLM.transcribe`), `transcription_chunk.rb`,
//! `Provider#transcribe` and `Protocol#transcribe` / `build_audio_file_part` / `audio_file_name`
//! in `lib/ruby_llm/protocol.rb`, and the transcription seams in
//! `protocols/chat_completions/transcription.rb` (OpenAI and every OpenAI-compatible provider),
//! `protocols/gemini/transcription.rb` (generateContent), `protocols/gemini/file_transcription.rb`
//! with `protocols/interactions/transcription.rb` (Gemini's dedicated transcription models),
//! `protocols/openrouter/transcription.rb`, `providers/mistral/transcription.rb`,
//! `providers/xai/transcription.rb`, and `providers/gpustack/transcription.rb`. WebSocket
//! transcription lives in `gemini_live` (`protocols/gemini/live_transcription.rb`) and
//! `xai_streaming` (`protocols/xai/streaming_transcription.rb`), over
//! [`WebsocketConnection`](crate::transport::WebsocketConnection), with `wav_audio`
//! (`transcription/wav_audio.rb`) reading the PCM format.
//!
//! ```ruby
//! transcription = RubyLLM.transcribe("meeting.wav")
//! transcription.text   # => "Welcome to today's meeting..."
//! ```

mod gemini_live;
#[cfg(test)]
mod spec_tests;
mod wav_audio;
mod xai_streaming;

use std::sync::Arc;

use serde_json::{Map, Value, json};

use crate::attachment::{Attachment, AttachmentType};
use crate::chat::resolve_model;
use crate::config::Config;
use crate::cost::Cost;
use crate::error::{Error, Result};
use crate::message::{Operation, UsageEntry};
use crate::model::Model;
use crate::models;
use crate::protocols::{anthropic::unsupported, chat_completions, deep_merge, int, str_of};
use crate::providers::{ProtocolName, Provider};
use crate::speech::Tracker;
use crate::tokens::Tokens;
use crate::transport::{Connection, SseEvent, json_response};

/// Text produced from spoken audio (`RubyLLM::Transcription`). `transcribe` returns one.
#[derive(Debug, Clone)]
pub struct Transcription {
    /// The transcribed text, or `None` when the provider returned none.
    pub text: Option<String>,
    /// The id of the model that produced the transcription.
    pub model: String,
    /// The language of the audio, when the provider reports it.
    pub language: Option<String>,
    /// The audio duration in seconds, when the provider reports it.
    pub duration: Option<f64>,
    /// The timed segments of the transcript, when the provider returns them. Diarization models
    /// add a speaker label to each.
    pub segments: Option<Vec<Value>>,
    /// Word timing and speaker labels, when the provider returns them (request timing with
    /// `timestamps`).
    pub words: Option<Vec<Value>>,
    /// One entry per provider attempt (`ruby_llm_usage_entries`).
    pub usage_entries: Vec<UsageEntry>,
    input_tokens: Option<i64>,
    output_tokens: Option<i64>,
    reported_cost: Option<f64>,
}

/// One event from a streaming transcription (`RubyLLM::TranscriptionChunk`).
#[derive(Debug, Clone)]
pub struct TranscriptionChunk {
    /// The normalized event type, such as [`TranscriptionChunk::DELTA`].
    pub kind: String,
    /// The text added by this event.
    pub delta: Option<String>,
    /// The transcript for a partial event, or the complete transcript on the final event.
    pub text: Option<String>,
    /// The segment this event completed. Diarization models label each with a speaker.
    pub segment: Option<Value>,
    /// The parsed provider event, for fields RubyLLM does not normalize.
    pub raw: Value,
}

impl TranscriptionChunk {
    /// Text deltas, arriving as the model transcribes.
    pub const DELTA: &'static str = "transcript.text.delta";
    /// A tentative transcript that may change before its segment completes.
    pub const PARTIAL: &'static str = "transcript.text.partial";
    /// A completed segment, on models that return timed or diarized segments.
    pub const SEGMENT: &'static str = "transcript.text.segment";
    /// The final event, carrying the complete transcript.
    pub const DONE: &'static str = "transcript.text.done";

    /// `delta?`: whether this event carries a text delta.
    pub fn is_delta(&self) -> bool {
        self.delta.is_some()
    }

    /// `partial?`: whether this is a tentative transcript, replacing the previous partial.
    pub fn is_partial(&self) -> bool {
        self.kind == Self::PARTIAL
    }

    /// `segment?`: whether this event completed a segment.
    pub fn is_segment(&self) -> bool {
        self.segment.is_some()
    }

    /// `done?`: whether this is the final event of the transcription.
    pub fn is_done(&self) -> bool {
        self.kind == Self::DONE
    }
}

/// Options for [`transcribe`], the keyword arguments of `Transcription.transcribe`.
#[derive(Default)]
pub struct TranscribeOptions<'a> {
    /// `model:`, defaulting to `config.default_transcription_model`.
    pub model: Option<&'a str>,
    pub provider: Option<&'a str>,
    pub assume_model_exists: bool,
    /// `language:`: an ISO 639-1 or BCP-47 hint for the spoken language.
    pub language: Option<&'a str>,
    /// `prompt:`: vocabulary or formatting guidance.
    pub prompt: Option<&'a str>,
    pub temperature: Option<f64>,
    /// `format:`: the transcript format in the provider's vocabulary (`"verbose_json"`,
    /// `"diarized_json"`, or a MIME type on Gemini).
    pub format: Option<&'a str>,
    /// `timestamps:`: `["word"]`, or `["segment"]` on providers that accept it.
    pub timestamps: Option<Vec<&'a str>>,
    /// `speaker_names:`: turns on diarization on models that support it. `Some(vec![])` asks for
    /// diarization without naming the speakers.
    pub speaker_names: Option<Vec<String>>,
    /// `speaker_references:`: reference audio for each named speaker.
    pub speaker_references: Option<Vec<Attachment>>,
    /// `provider_options:`: merged into the rendered request as-is.
    pub provider_options: Value,
    /// `context:`: use this configuration instead of the global one.
    pub config: Option<Arc<Config>>,
    /// `metadata:`: added to the `transcription.rust_llm` event payload, never sent to the
    /// provider.
    pub metadata: Option<Value>,
}

impl Transcription {
    fn new(text: Option<String>, model: &str) -> Transcription {
        Transcription {
            text,
            model: model.to_string(),
            language: None,
            duration: None,
            segments: None,
            words: None,
            usage_entries: Vec::new(),
            input_tokens: None,
            output_tokens: None,
            reported_cost: None,
        }
    }

    /// Usage across every provider attempt, or the usage this transcription reported.
    pub fn tokens(&self) -> Tokens {
        if !self.usage_entries.is_empty() {
            return Tokens::aggregate(self.usage_entries.iter().map(|e| &e.tokens));
        }
        Tokens {
            input: self.input_tokens,
            output: self.output_tokens,
            reported_cost: self.reported_cost,
            ..Default::default()
        }
    }

    /// Cost across every provider attempt, priced as audio tokens.
    pub fn cost(&self) -> Cost {
        if !self.usage_entries.is_empty() {
            let complete = self.usage_entries.iter().all(UsageEntry::cost_available);
            return Cost::aggregate(self.usage_entries.iter().map(|e| &e.cost), complete);
        }
        Cost::audio(&self.tokens(), self.model_info().as_ref())
    }

    /// The registry model for `model`, or `None` when it is not in the registry.
    pub fn model_info(&self) -> Option<Model> {
        models::models().find(&self.model, None).ok()
    }
}

/// `RubyLLM.transcribe(audio_file, model:, language:, provider:, prompt:, temperature:, format:,
/// timestamps:, speaker_names:, speaker_references:, provider_options:)`. `audio` is a path, URL,
/// or [`Attachment`].
pub async fn transcribe(
    audio: impl Into<Attachment>,
    options: TranscribeOptions<'_>,
) -> Result<Transcription> {
    run(audio.into(), options, None).await
}

/// `RubyLLM.transcribe(audio_file, ...) { |chunk| ... }`: `on_chunk` receives each
/// [`TranscriptionChunk`] as it arrives, and the completed [`Transcription`] is still returned.
pub async fn transcribe_stream(
    audio: impl Into<Attachment>,
    options: TranscribeOptions<'_>,
    mut on_chunk: impl FnMut(&TranscriptionChunk) + Send,
) -> Result<Transcription> {
    run(
        audio.into(),
        options,
        Some(&mut on_chunk as &mut (dyn FnMut(&TranscriptionChunk) + Send)),
    )
    .await
}

/// Which transcription seams a provider's protocol includes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Family {
    /// `ChatCompletions::Transcription` (OpenAI and every OpenAI-compatible provider).
    OpenAI,
    GPUStack,
    Mistral,
    XAI,
    OpenRouter,
    /// `Gemini::Transcription` over generateContent.
    Gemini,
    /// `Interactions::Transcription`: Gemini's dedicated transcription model.
    Interactions,
    /// `Gemini::LiveTranscription`, over the Live API WebSocket.
    GeminiLive,
}

impl Family {
    /// `resolve_protocol(nil, model, operation: :transcribe)`.
    fn for_model(provider: Provider, model: &str) -> Result<Family> {
        match provider {
            Provider::Gemini if model == "gemini-3.5-transcribe" => Ok(Family::Interactions),
            Provider::Gemini if model == "gemini-3.5-transcribe-live" => Ok(Family::GeminiLive),
            Provider::Gemini => Ok(Family::Gemini),
            Provider::Mistral => Ok(Family::Mistral),
            Provider::XAI => Ok(Family::XAI),
            Provider::OpenRouter => Ok(Family::OpenRouter),
            Provider::GPUStack => Ok(Family::GPUStack),
            Provider::Anthropic | Provider::TypeSafe => Err(Error::Api(
                format!("{} doesn't support transcription", provider.display()),
                None,
            )),
            _ => Ok(Family::OpenAI),
        }
    }

    /// Whether the family streams: over SSE, or over a WebSocket for xAI and Gemini Live.
    fn streams(self) -> bool {
        matches!(
            self,
            Family::OpenAI | Family::GPUStack | Family::Mistral | Family::XAI | Family::GeminiLive
        )
    }

    /// `render_transcription_options(timestamps:, format:, streaming:)`.
    fn render_options(
        self,
        timestamps: Option<&[&str]>,
        format: Option<&str>,
        streaming: bool,
    ) -> Result<Value> {
        let Some(values) = timestamps else {
            return Ok(json!({}));
        };
        let single = |name: &str| values == [name];
        match self {
            Family::OpenAI | Family::GPUStack | Family::OpenRouter => {
                if values.is_empty() || values.iter().any(|v| !["word", "segment"].contains(v)) {
                    return Err(Error::Argument(
                        "Transcription timestamps must be word or segment".into(),
                    ));
                }
                if streaming || format.is_some_and(|f| f != "verbose_json") {
                    return Err(Error::Argument(
                        "Transcription timestamps require a non-streaming verbose_json response"
                            .into(),
                    ));
                }
                Ok(json!({ "response_format": "verbose_json", "timestamp_granularities": values }))
            }
            Family::Mistral if single("segment") => {
                Ok(json!({ "timestamp_granularities": ["segment"] }))
            }
            Family::Mistral => Err(Error::Argument(
                "Mistral transcription timestamps must be segment".into(),
            )),
            Family::XAI if single("word") => Ok(json!({})),
            Family::XAI => Err(Error::Argument(
                "xAI transcription timestamps must be word".into(),
            )),
            Family::Interactions if single("word") => Ok(json!({
                "generation_config": { "transcription_config": { "mode": { "type": "verbatim", "timestamp_granularities": ["word"] } } }
            })),
            Family::Interactions => Err(Error::Argument(
                "Gemini transcription timestamps must be word".into(),
            )),
            Family::Gemini | Family::GeminiLive => Err(Error::Argument(
                "This transcription protocol does not support timestamps".into(),
            )),
        }
    }
}

/// `Transcription.transcribe`: the provider call inside a `transcription.rust_llm` event.
async fn run(
    audio: Attachment,
    options: TranscribeOptions<'_>,
    on_chunk: Option<&mut (dyn FnMut(&TranscriptionChunk) + Send)>,
) -> Result<Transcription> {
    let config = options.config.clone().unwrap_or_else(crate::config);
    let model_id = options
        .model
        .unwrap_or(&config.default_transcription_model)
        .to_string();
    let (model, provider) =
        resolve_model(&model_id, options.provider, options.assume_model_exists)?;
    let mut event = crate::instrumentation::Event::start(&config, "transcription.rust_llm", || {
        let empty = Tokens::default();
        crate::instrumentation::payload([
            ("provider", provider.slug().into()),
            ("provider_class", provider.display().into()),
            ("model", model.id.clone().into()),
            ("language", options.language.into()),
            ("provider_options", options.provider_options.clone()),
            (
                "metadata",
                crate::instrumentation::metadata(&options.metadata),
            ),
            ("tokens", crate::instrumentation::tokens_h(&empty)),
            (
                "cost",
                crate::instrumentation::cost_h(&Cost::audio(&empty, Some(&model))),
            ),
        ])
    });
    let result = tracing::Instrument::instrument(
        transcribe_inner(audio, options, on_chunk, config.clone(), model, provider),
        event.span(),
    )
    .await;
    if let Ok(t) = &result {
        crate::instrumentation::usages(&config, &t.usage_entries);
        event.set("result", || {
            json!({ "text": t.text, "model": t.model, "language": t.language, "duration": t.duration })
        });
        event.set("response_model", || t.model.clone().into());
        event.set("tokens", || crate::instrumentation::tokens_h(&t.tokens()));
        event.set("cost", || crate::instrumentation::cost_h(&t.cost()));
    }
    event.finish(result.as_ref().err());
    result
}

async fn transcribe_inner(
    mut audio: Attachment,
    options: TranscribeOptions<'_>,
    on_chunk: Option<&mut (dyn FnMut(&TranscriptionChunk) + Send)>,
    config: Arc<Config>,
    model: Model,
    provider: Provider,
) -> Result<Transcription> {
    provider.ensure_configured(&config)?;
    let connection = Connection::new(provider, config.clone())?;
    let family = Family::for_model(provider, &model.id)?;

    // `Provider#transcribe`: timestamp options are rendered first and sit under provider_options.
    let mut provider_options = family.render_options(
        options.timestamps.as_deref(),
        options.format,
        on_chunk.is_some(),
    )?;
    // `provider_options: {}` by default; `Null` would replace the options in a deep merge.
    if !options.provider_options.is_null() {
        deep_merge(&mut provider_options, &options.provider_options);
    }
    if on_chunk.is_some() && !family.streams() {
        return Err(Error::Api(
            format!(
                "{} doesn't support streaming transcription",
                provider.display()
            ),
            None,
        ));
    }

    audio.load(connection.client()).await?;
    let mut references = options.speaker_references.clone();
    for reference in references.iter_mut().flatten() {
        reference.load(connection.client()).await?;
    }
    let request = Request {
        model: &model.id,
        language: options.language,
        prompt: options.prompt,
        temperature: options.temperature,
        format: options.format,
        speaker_names: options.speaker_names.as_deref(),
        speaker_references: references.as_deref(),
        provider_options: &provider_options,
    };

    let mut tracker = Tracker::default();
    let mut transcription = match family {
        Family::Gemini => {
            let payload = gemini_payload(&audio, &request)?;
            let raw = connection
                .post(
                    &format!("models/{}:generateContent", model.id),
                    &payload,
                    &[],
                    &mut tracker.on_attempt(),
                )
                .await?;
            parse_gemini(&raw.body, &model.id)
        }
        Family::Interactions => {
            let payload = interactions_payload(&audio, &request)?;
            let raw = connection
                .post("interactions", &payload, &[], &mut tracker.on_attempt())
                .await?;
            parse_interactions(&raw.body, &model.id)?
        }
        Family::OpenRouter => {
            let payload = openrouter_payload(&audio, &request)?;
            let raw = connection
                .post(
                    "audio/transcriptions",
                    &payload,
                    &[],
                    &mut tracker.on_attempt(),
                )
                .await?;
            parse_json(provider, &raw.body, &model.id)
        }
        Family::GeminiLive => {
            gemini_live::transcribe(
                audio.bytes()?,
                &request,
                &provider.api_base(&config)?,
                &provider.headers(&config),
                &config,
                on_chunk,
            )
            .await?
        }
        _ => {
            let file = FilePart::new(&audio)?;
            let mut payload = multipart_payload(family, &request)?;
            let path = if family == Family::XAI {
                "stt"
            } else {
                "audio/transcriptions"
            };
            match on_chunk {
                // `XAI::StreamingTranscription#stream_transcription`: the payload goes out as
                // WebSocket query parameters instead of a multipart form.
                Some(on_chunk) if family == Family::XAI => {
                    xai_streaming::stream_transcription(
                        audio.bytes()?,
                        &payload,
                        &model.id,
                        &provider.api_base(&config)?,
                        &provider.headers(&config),
                        &config,
                        on_chunk,
                    )
                    .await?
                }
                Some(on_chunk) => {
                    if family == Family::GPUStack {
                        // `{ stream_include_usage: 'true' }.merge(payload)`: the flag goes first.
                        let mut merged = Map::from_iter([(
                            "stream_include_usage".to_string(),
                            Value::from("true"),
                        )]);
                        merged.extend(payload);
                        payload = merged;
                    }
                    payload.insert("stream".into(), "true".into());
                    stream(
                        family,
                        provider,
                        &connection,
                        path,
                        &payload,
                        &file,
                        &model.id,
                        &mut tracker,
                        on_chunk,
                    )
                    .await?
                }
                None => {
                    let resp = connection
                        .send_tracked(
                            reqwest::Method::POST,
                            path,
                            &[],
                            true,
                            &|req| req.multipart(file.form(&payload)),
                            &mut tracker.on_attempt(),
                        )
                        .await?;
                    let raw = json_response(resp, Value::Null).await?;
                    if family == Family::XAI {
                        parse_xai(&raw.body, &model.id)
                    } else {
                        parse_json(provider, &raw.body, &model.id)
                    }
                }
            }
        }
    };
    transcription.usage_entries = tracker.entries(
        Operation::Transcription,
        provider,
        &model,
        transcription.tokens(),
        transcription.cost(),
    );
    Ok(transcription)
}

/// The arguments every `render_transcription_payload` receives.
struct Request<'a> {
    model: &'a str,
    language: Option<&'a str>,
    prompt: Option<&'a str>,
    temperature: Option<f64>,
    format: Option<&'a str>,
    speaker_names: Option<&'a [String]>,
    speaker_references: Option<&'a [Attachment]>,
    provider_options: &'a Value,
}

/// `build_audio_file_part`: the audio as a multipart file with an extension providers accept.
struct FilePart {
    bytes: Vec<u8>,
    filename: String,
    mime_type: String,
}

impl FilePart {
    fn new(audio: &Attachment) -> Result<FilePart> {
        // `audio_file_name`: providers reject audio whose filename carries no extension.
        let name = audio
            .filename
            .clone()
            .filter(|n| !n.is_empty())
            .unwrap_or_else(|| "audio".into());
        let has_extension = std::path::Path::new(&name)
            .extension()
            .is_some_and(|e| !e.is_empty());
        let filename = if has_extension {
            name
        } else {
            format!("{name}.{}", audio.format())
        };
        reqwest::multipart::Part::bytes(Vec::new())
            .mime_str(&audio.mime_type)
            .map_err(|e| {
                Error::Argument(format!("invalid content type {:?}: {e}", audio.mime_type))
            })?;
        Ok(FilePart {
            bytes: audio.bytes()?.to_vec(),
            filename,
            mime_type: audio.mime_type.clone(),
        })
    }

    /// Faraday's multipart encoding of `payload`, where the `file` key (a `Null` placeholder)
    /// is this file. Arrays become repeated `key[]` fields and hashes `key[sub]` fields.
    fn form(&self, payload: &Map<String, Value>) -> reqwest::multipart::Form {
        let mut form = reqwest::multipart::Form::new();
        for (key, value) in payload {
            if key == "file" && value.is_null() {
                let part = reqwest::multipart::Part::bytes(self.bytes.clone())
                    .file_name(self.filename.clone())
                    .mime_str(&self.mime_type)
                    .expect("content type validated in FilePart::new");
                form = form.part("file", part);
            } else {
                for (name, text) in flatten_field(key, value) {
                    form = form.text(name, text);
                }
            }
        }
        form
    }
}

fn flatten_field(key: &str, value: &Value) -> Vec<(String, String)> {
    match value {
        Value::Null => Vec::new(),
        Value::Array(items) => items
            .iter()
            .flat_map(|v| flatten_field(&format!("{key}[]"), v))
            .collect(),
        Value::Object(map) => map
            .iter()
            .flat_map(|(k, v)| flatten_field(&format!("{key}[{k}]"), v))
            .collect(),
        Value::String(s) => vec![(key.to_string(), s.clone())],
        other => vec![(key.to_string(), other.to_string())],
    }
}

/// `render_transcription_payload` for the multipart families, with `file` as a `Null`
/// placeholder in Ruby's key position.
fn multipart_payload(family: Family, r: &Request) -> Result<Map<String, Value>> {
    let mut payload = Map::new();
    let put = |payload: &mut Map<String, Value>, key: &str, value: Option<Value>| {
        if let Some(value) = value {
            payload.insert(key.to_string(), value);
        }
    };
    match family {
        Family::Mistral => {
            put(&mut payload, "model", Some(r.model.into()));
            payload.insert("file".into(), Value::Null);
            put(&mut payload, "language", r.language.map(Into::into));
            put(&mut payload, "temperature", r.temperature.map(Into::into));
            put(&mut payload, "context_bias", r.prompt.map(|p| json!([p])));
            if r.speaker_names.is_some() {
                payload.insert("diarize".into(), true.into());
                payload.insert("timestamp_granularities".into(), json!(["segment"]));
            }
            merge(&mut payload, r.provider_options);
        }
        Family::XAI => {
            put(&mut payload, "language", r.language.map(Into::into));
            if r.speaker_names.is_some() {
                payload.insert("diarize".into(), true.into());
            }
            merge(&mut payload, r.provider_options);
            // `.merge(file: file_part)`: the audio goes last in the form.
            payload.remove("file");
            payload.insert("file".into(), Value::Null);
        }
        _ => {
            let references = r
                .speaker_references
                .map(|refs| {
                    refs.iter()
                        .map(|a| a.for_llm().map(Value::from))
                        .collect::<Result<Vec<_>>>()
                })
                .transpose()?;
            let default_format = r.model.contains("diarize").then_some("diarized_json");
            put(&mut payload, "model", Some(r.model.into()));
            payload.insert("file".into(), Value::Null);
            put(&mut payload, "language", r.language.map(Into::into));
            put(
                &mut payload,
                "response_format",
                r.format.or(default_format).map(Into::into),
            );
            put(&mut payload, "prompt", r.prompt.map(Into::into));
            put(&mut payload, "temperature", r.temperature.map(Into::into));
            put(
                &mut payload,
                "known_speaker_names",
                r.speaker_names.map(|n| json!(n)),
            );
            put(
                &mut payload,
                "known_speaker_references",
                references.map(Value::Array),
            );
            merge(&mut payload, r.provider_options);
        }
    }
    Ok(payload)
}

/// Ruby's `Hash#merge`: provider options replace top-level keys in place.
fn merge(payload: &mut Map<String, Value>, options: &Value) {
    if let Some(options) = options.as_object() {
        for (k, v) in options {
            payload.insert(k.clone(), v.clone());
        }
    }
}

/// `ChatCompletions::Transcription#parse_transcription_response` (OpenAI, Mistral, OpenRouter,
/// GPUStack). A non-JSON body is the transcript itself (`response_format: "text"`).
fn parse_json(provider: Provider, data: &Value, model: &str) -> Transcription {
    if let Value::String(text) = data {
        return Transcription::new(Some(text.clone()), model);
    }
    let usage = data.get("usage").cloned().unwrap_or_else(|| json!({}));
    let mut t = Transcription::new(str_of(data.get("text")), model);
    t.language = str_of(data.get("language"));
    t.duration = data
        .get("duration")
        .and_then(Value::as_f64)
        .or_else(|| duration(provider, &usage));
    t.segments = array(data.get("segments"));
    t.words = array(data.get("words"));
    fill_tokens(&mut t, provider, &usage);
    t
}

fn fill_tokens(t: &mut Transcription, provider: Provider, usage: &Value) {
    t.input_tokens = int(usage.get("input_tokens")).or_else(|| int(usage.get("prompt_tokens")));
    t.output_tokens =
        int(usage.get("output_tokens")).or_else(|| int(usage.get("completion_tokens")));
    t.reported_cost = chat_completions::reported_cost(provider, usage);
}

/// `transcription_duration`: `usage.seconds`, or Mistral's `prompt_audio_seconds`.
fn duration(provider: Provider, usage: &Value) -> Option<f64> {
    let mistral = (provider == Provider::Mistral)
        .then(|| usage.get("prompt_audio_seconds").and_then(Value::as_f64))
        .flatten();
    mistral.or_else(|| usage.get("seconds").and_then(Value::as_f64))
}

fn array(v: Option<&Value>) -> Option<Vec<Value>> {
    v.and_then(Value::as_array).cloned()
}

/// `XAI::Transcription#parse_transcription_response`: channels stand in for segments.
fn parse_xai(data: &Value, model: &str) -> Transcription {
    let mut t = Transcription::new(str_of(data.get("text")), model);
    t.language = str_of(data.get("language"));
    t.duration = data.get("duration").and_then(Value::as_f64);
    t.segments = array(data.get("channels"));
    t.words = array(data.get("words"));
    t
}

/// `ChatCompletions::Transcription#stream_transcription`: a multipart POST answered with
/// server-sent events. A failure after the first chunk is final, like every stream.
#[allow(clippy::too_many_arguments)]
async fn stream(
    family: Family,
    provider: Provider,
    connection: &Connection,
    path: &str,
    payload: &Map<String, Value>,
    file: &FilePart,
    model: &str,
    tracker: &mut Tracker,
    on_chunk: &mut (dyn FnMut(&TranscriptionChunk) + Send),
) -> Result<Transcription> {
    let resp = connection
        .send_tracked(
            reqwest::Method::POST,
            path,
            &[],
            true,
            &|req| req.multipart(file.form(payload)),
            &mut tracker.on_attempt(),
        )
        .await?;
    let mut chunks: Vec<TranscriptionChunk> = Vec::new();
    let mut on_event = |_event: SseEvent, data: Value| -> Result<()> {
        let chunk = build_chunk(family, data);
        on_chunk(&chunk);
        chunks.push(chunk);
        Ok(())
    };
    let mut delivered = false;
    connection
        .read_stream(
            resp,
            &mut on_event,
            crate::protocols::streaming_error_status(ProtocolName::ChatCompletions),
            &mut delivered,
        )
        .await?;
    Ok(streamed_transcription(provider, &chunks, model))
}

/// `build_transcription_chunk`, with the Mistral and GPUStack event shapes.
fn build_chunk(family: Family, data: Value) -> TranscriptionChunk {
    let kind = data
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let segment_of = |data: &Value| {
        let mut segment = data.as_object().cloned().unwrap_or_default();
        segment.remove("type");
        Value::Object(segment)
    };
    let mistral = match kind.as_str() {
        "transcription.text.delta" => Some(TranscriptionChunk::DELTA),
        "transcription.segment" => Some(TranscriptionChunk::SEGMENT),
        "transcription.done" => Some(TranscriptionChunk::DONE),
        _ => None,
    };
    if family == Family::Mistral
        && let Some(kind) = mistral
    {
        return TranscriptionChunk {
            kind: kind.into(),
            delta: (kind == TranscriptionChunk::DELTA)
                .then(|| str_of(data.get("text")))
                .flatten(),
            text: (kind == TranscriptionChunk::DONE)
                .then(|| str_of(data.get("text")))
                .flatten(),
            segment: (kind == TranscriptionChunk::SEGMENT).then(|| segment_of(&data)),
            raw: data,
        };
    }
    if family == Family::GPUStack
        && let Some(choices) = data.get("choices").and_then(Value::as_array)
    {
        let choice = choices.first().cloned().unwrap_or_else(|| json!({}));
        let finished =
            choice.get("finish_reason").is_some_and(|r| !r.is_null()) || choices.is_empty();
        return TranscriptionChunk {
            kind: if finished {
                TranscriptionChunk::DONE
            } else {
                TranscriptionChunk::DELTA
            }
            .into(),
            delta: str_of(choice.pointer("/delta/content")),
            text: None,
            segment: None,
            raw: data,
        };
    }
    TranscriptionChunk {
        delta: str_of(data.get("delta")),
        text: (kind == TranscriptionChunk::DONE)
            .then(|| str_of(data.get("text")))
            .flatten(),
        segment: (kind == TranscriptionChunk::SEGMENT).then(|| segment_of(&data)),
        kind,
        raw: data,
    }
}

/// `build_streamed_transcription`: the final event's transcript and usage, or the transcript
/// rebuilt from deltas (or segments, on diarization models that stream no deltas).
fn streamed_transcription(
    provider: Provider,
    chunks: &[TranscriptionChunk],
    model: &str,
) -> Transcription {
    let last = chunks.iter().rev().find(|c| c.is_done());
    let data = last.map(|c| c.raw.clone()).unwrap_or_else(|| json!({}));
    let usage = data.get("usage").cloned().unwrap_or_else(|| json!({}));
    let text = last.and_then(|c| c.text.clone()).unwrap_or_else(|| {
        let deltas: Vec<&str> = chunks.iter().filter_map(|c| c.delta.as_deref()).collect();
        if !deltas.is_empty() {
            return deltas.concat();
        }
        chunks
            .iter()
            .filter_map(|c| c.segment.as_ref()?.get("text")?.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    });
    let mut t = Transcription::new(Some(text), model);
    t.language = str_of(data.get("language"));
    t.duration = duration(provider, &usage);
    let segments = array(data.get("segments"))
        .unwrap_or_else(|| chunks.iter().filter_map(|c| c.segment.clone()).collect());
    t.segments = (!segments.is_empty()).then_some(segments);
    fill_tokens(&mut t, provider, &usage);
    t
}

const GEMINI_PROMPT: &str =
    "Transcribe the provided audio and respond with only the transcript text.";

/// `Gemini::Transcription#render_transcription_payload` (generateContent).
fn gemini_payload(audio: &Attachment, r: &Request) -> Result<Value> {
    let mut prompt = GEMINI_PROMPT.to_string();
    if let Some(language) = r.language {
        prompt.push_str(&format!(" Respond in the {language} language."));
    }
    if let Some(custom) = r.prompt {
        prompt.push_str(&format!(" {custom}"));
    }
    let audio_part =
        json!({ "inline_data": { "mime_type": audio.mime_type, "data": audio.encoded()? } });
    if audio.kind() != AttachmentType::Audio {
        return Err(Error::UnsupportedAttachment(unsupported(&audio.mime_type)));
    }
    let mut generation_config = json!({ "responseMimeType": r.format.unwrap_or("text/plain") });
    if let Some(temperature) = r.temperature {
        generation_config["temperature"] = temperature.into();
    }
    let mut payload = json!({
        "contents": [{ "role": "user", "parts": [{ "text": prompt }, audio_part] }],
        "generationConfig": generation_config,
    });
    deep_merge(&mut payload, r.provider_options);
    Ok(payload)
}

/// `Gemini::Transcription#parse_transcription_response`.
fn parse_gemini(data: &Value, model: &str) -> Transcription {
    let texts: Vec<&str> = data
        .pointer("/candidates/0/content/parts")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|part| part.get("text").and_then(Value::as_str))
        .collect();
    let mut t = Transcription::new((!texts.is_empty()).then(|| texts.concat()), model);
    if let Some(meta) = data.get("usageMetadata").filter(|m| m.is_object()) {
        t.input_tokens = int(meta.get("promptTokenCount"));
        t.output_tokens = Some(
            int(meta.get("candidatesTokenCount")).unwrap_or(0)
                + int(meta.get("thoughtsTokenCount")).unwrap_or(0),
        );
    }
    t
}

/// `FileTranscription#transcribe` validation plus `Interactions::Transcription#render_transcription_payload`.
fn interactions_payload(audio: &Attachment, r: &Request) -> Result<Value> {
    if r.format.is_some() || r.speaker_references.is_some() || r.temperature.is_some() {
        return Err(Error::Argument(
            "Dedicated transcription does not accept format, speaker references, or temperature"
                .into(),
        ));
    }
    if audio.kind() != AttachmentType::Audio {
        return Err(Error::Argument(
            "Dedicated transcription requires exactly one audio file".into(),
        ));
    }
    let mut config = Map::new();
    if let Some(language) = r.language {
        config.insert("language_codes".into(), json!([language]));
    }
    if let Some(prompt) = r.prompt {
        config.insert("custom_vocabulary".into(), json!([prompt]));
    }
    if r.speaker_names.is_some() {
        config.insert(
            "mode".into(),
            json!({ "type": "verbatim", "diarization_mode": "speaker" }),
        );
    }
    let mut payload = json!({
        "model": r.model,
        "store": false,
        "input": [{ "type": "audio", "mime_type": audio.mime_type, "data": audio.encoded()? }],
        "generation_config": { "transcription_config": config },
    });
    deep_merge(&mut payload, r.provider_options);
    // `validate_transcription_config`.
    let config = &payload["generation_config"]["transcription_config"];
    let mode = config.get("mode").filter(|m| m.is_object());
    if config
        .get("custom_vocabulary")
        .is_some_and(|v| !v.is_null())
        && mode.is_some_and(|m| {
            m.get("diarization_mode").is_some() || m.get("timestamp_granularities").is_some()
        })
    {
        return Err(Error::Argument(
            "Gemini custom vocabulary cannot be combined with diarization or word timestamps"
                .into(),
        ));
    }
    Ok(payload)
}

/// `Interactions::Transcription#parse_transcription_response`: the model output text, word
/// timings from `word_info` annotations, and `parse_interaction_usage`'s input/output tokens.
fn parse_interactions(data: &Value, model: &str) -> Result<Transcription> {
    let status = data.get("status").and_then(Value::as_str).unwrap_or("");
    if !["completed", "requires_action", "incomplete"].contains(&status) {
        let messages: Vec<&str> = data
            .get("errors")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|e| e.get("message").and_then(Value::as_str))
            .collect();
        let message = if messages.is_empty() {
            format!("Gemini interaction ended with status {status}")
        } else {
            messages.join("; ")
        };
        return Err(Error::Api(message, None));
    }
    let steps: Vec<&Value> = data
        .get("steps")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .collect();
    let parts = || {
        steps.iter().flat_map(|s| {
            s.get("content")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
        })
    };
    let text: String = steps
        .iter()
        .filter(|s| s.get("type").and_then(Value::as_str) == Some("model_output"))
        .flat_map(|s| {
            s.get("content")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
        })
        .filter(|p| p.get("type").and_then(Value::as_str) == Some("text"))
        .filter_map(|p| p.get("text").and_then(Value::as_str))
        .collect();
    let words: Vec<Value> = parts()
        .flat_map(|p| {
            p.get("annotations")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
        })
        .filter(|a| a.get("type").and_then(Value::as_str) == Some("word_info"))
        .map(|a| {
            let offset = |key: &str| {
                a.get(key)
                    .and_then(Value::as_str)
                    .and_then(|s| s.trim_end_matches('s').parse::<f64>().ok())
            };
            let mut word = Map::new();
            for (key, value) in [
                ("word", a.get("text").cloned()),
                ("speaker", a.get("speaker").cloned()),
                ("start", offset("start_offset").map(Value::from)),
                ("end", offset("end_offset").map(Value::from)),
            ] {
                if let Some(value) = value.filter(|v| !v.is_null()) {
                    word.insert(key.into(), value);
                }
            }
            Value::Object(word)
        })
        .collect();
    let mut t = Transcription::new(Some(text), model);
    t.words = (!words.is_empty()).then_some(words);
    let usage = data.get("usage").cloned().unwrap_or_else(|| json!({}));
    let cached = int(usage.get("total_cached_tokens")).unwrap_or(0);
    t.input_tokens = int(usage.get("total_input_tokens")).map(|input| {
        (input + int(usage.get("total_tool_use_tokens")).unwrap_or(0) - cached).max(0)
    });
    t.output_tokens = int(usage.get("total_output_tokens"))
        .map(|out| out + int(usage.get("total_thought_tokens")).unwrap_or(0));
    Ok(t)
}

/// `OpenRouter::Transcription#transcribe`: JSON with Base64 audio, validated first.
fn openrouter_payload(audio: &Attachment, r: &Request) -> Result<Value> {
    if r.speaker_references.is_some() || r.speaker_names.is_some_and(|n| !n.is_empty()) {
        return Err(Error::Argument(
            "OpenRouter accepts speaker_names: [] for diarization, but not speaker identities or reference audio".into(),
        ));
    }
    if r.prompt.is_some() {
        return Err(Error::Argument(
            "OpenRouter transcription ignores prompt; use provider_options for backend hints"
                .into(),
        ));
    }
    let diarization = r.speaker_names.is_some();
    let formats: &[&str] = if diarization {
        &["verbose_json"]
    } else {
        &["json", "verbose_json"]
    };
    if r.format.is_some_and(|f| !formats.contains(&f)) {
        return Err(Error::Argument(
            "OpenRouter transcription accepts json or verbose_json; diarization requires verbose_json".into(),
        ));
    }
    let mut payload = Map::new();
    payload.insert("model".into(), r.model.into());
    payload.insert(
        "input_audio".into(),
        json!({ "data": audio.encoded()?, "format": audio.format() }),
    );
    if let Some(language) = r.language {
        payload.insert("language".into(), language.into());
    }
    if let Some(temperature) = r.temperature {
        payload.insert("temperature".into(), temperature.into());
    }
    let default_format = if diarization { "verbose_json" } else { "json" };
    payload.insert(
        "response_format".into(),
        r.format.unwrap_or(default_format).into(),
    );
    if diarization {
        payload.insert(
            "provider".into(),
            json!({ "options": { "azure": { "diarization": { "enabled": true } }, "deepgram": { "diarize": true } } }),
        );
    }
    let mut payload = Value::Object(payload);
    deep_merge(&mut payload, r.provider_options);
    Ok(payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multipart_arrays_and_hashes_flatten_like_faraday() {
        assert_eq!(
            flatten_field("timestamp_granularities", &json!(["word", "segment"])),
            vec![
                ("timestamp_granularities[]".into(), "word".into()),
                ("timestamp_granularities[]".into(), "segment".into())
            ]
        );
        assert_eq!(
            flatten_field("diarize", &json!(true)),
            vec![("diarize".to_string(), "true".to_string())]
        );
        assert_eq!(
            flatten_field("chunking", &json!({ "type": "auto" })),
            vec![("chunking[type]".to_string(), "auto".to_string())]
        );
    }

    #[test]
    fn timestamps_follow_each_providers_rules() {
        let word = Some(&["word"][..]);
        assert_eq!(
            Family::OpenAI.render_options(word, None, false).unwrap(),
            json!({ "response_format": "verbose_json", "timestamp_granularities": ["word"] })
        );
        assert!(matches!(
            Family::OpenAI.render_options(word, None, true),
            Err(Error::Argument(_))
        ));
        assert!(matches!(
            Family::OpenAI.render_options(word, Some("json"), false),
            Err(Error::Argument(_))
        ));
        assert!(matches!(
            Family::Mistral.render_options(word, None, false),
            Err(Error::Argument(_))
        ));
        assert_eq!(
            Family::XAI.render_options(word, None, false).unwrap(),
            json!({})
        );
        assert!(matches!(
            Family::Gemini.render_options(word, None, false),
            Err(Error::Argument(_))
        ));
    }
}
