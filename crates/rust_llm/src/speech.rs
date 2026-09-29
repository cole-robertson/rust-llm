//! Port of `lib/ruby_llm/speech.rb` (`RubyLLM.speak`), `speech_chunk.rb`, `Protocol#speak` /
//! `stream_speech` / `stream_speech_response` in `lib/ruby_llm/protocol.rb`,
//! `protocol/binary_streaming.rb`, and the speech seams in `protocols/chat_completions/speech.rb`
//! (OpenAI and every OpenAI-compatible provider), `protocols/gemini/speech.rb`,
//! `providers/mistral/speech.rb`, `providers/xai/speech.rb`, `providers/openrouter/speech.rb`, and
//! `providers/gpustack/speech.rb`.
//!
//! ```ruby
//! speech = RubyLLM.speak "Hello, welcome to RubyLLM!"
//! speech.save "welcome.mp3"
//! ```

use std::path::Path;
use std::sync::Arc;

use base64::Engine;
use futures::StreamExt;
use serde_json::{Map, Value, json};

use crate::chat::{failure_tokens, resolve_model};
use crate::config::Config;
use crate::cost::Cost;
use crate::error::{Error, ErrorResponse, Result, error_for_status};
use crate::message::{Operation, UsageEntry, UsageStatus};
use crate::model::Model;
use crate::models;
use crate::protocols::{deep_merge, int};
use crate::providers::Provider;
use crate::tokens::Tokens;
use crate::transport::Connection;

/// `Speech::MIME_TYPES`: audio format names and their MIME types.
pub const MIME_TYPES: &[(&str, &str)] = &[
    ("aac", "audio/aac"),
    ("flac", "audio/flac"),
    ("mp3", "audio/mpeg"),
    ("opus", "audio/opus"),
    ("pcm", "audio/pcm"),
    ("wav", "audio/wav"),
];

/// `MIME_TYPES.fetch(format, "audio/#{format}")`.
fn mime_type_for(format: &str) -> String {
    MIME_TYPES
        .iter()
        .find(|(f, _)| *f == format)
        .map(|(_, m)| m.to_string())
        .unwrap_or_else(|| format!("audio/{format}"))
}

/// Audio generated from text (`RubyLLM::Speech`). `speak` returns one.
#[derive(Debug, Clone)]
pub struct Speech {
    /// The raw audio bytes returned by the provider.
    pub data: Vec<u8>,
    /// The id of the model that generated the audio.
    pub model: String,
    /// The voice used for synthesis; the provider default when none was given.
    pub voice: Option<String>,
    /// The audio format name, such as `"mp3"` or `"pcm"`.
    pub format: String,
    /// The MIME type of the audio, such as `"audio/mpeg"`.
    pub mime_type: String,
    /// One entry per provider attempt (`ruby_llm_usage_entries`).
    pub usage_entries: Vec<UsageEntry>,
    input_tokens: Option<i64>,
    output_tokens: Option<i64>,
}

/// A piece of generated audio, passed to `speak_stream`'s callback as it arrives
/// (`RubyLLM::SpeechChunk`). Chunks are consecutive bytes of one recording, not separate files.
#[derive(Debug, Clone)]
pub struct SpeechChunk {
    pub data: Vec<u8>,
    pub format: String,
    pub mime_type: String,
}

impl SpeechChunk {
    pub fn new(data: Vec<u8>, format: &str, mime_type: Option<&str>) -> SpeechChunk {
        SpeechChunk {
            data,
            format: format.to_string(),
            mime_type: mime_type.map_or_else(|| mime_type_for(format), str::to_string),
        }
    }

    /// The raw audio bytes. Alias for `data`.
    pub fn to_blob(&self) -> &[u8] {
        &self.data
    }
}

/// Options for [`speak`], the keyword arguments of `Speech.speak`.
#[derive(Default)]
pub struct SpeakOptions<'a> {
    /// `model:`, defaulting to `config.default_speech_model`.
    pub model: Option<&'a str>,
    pub provider: Option<&'a str>,
    pub assume_model_exists: bool,
    /// `voice:`: the provider's default voice when `None`.
    pub voice: Option<&'a str>,
    /// `format:`: an audio format name such as `"mp3"` or `"wav"`.
    pub format: Option<&'a str>,
    /// `provider_options:`: merged into the request as-is, in the provider's vocabulary.
    pub provider_options: Value,
    /// `context:`: use this configuration instead of the global one.
    pub config: Option<Arc<Config>>,
}

impl Speech {
    /// `Speech.new(data:, model:, voice:, format:, mime_type:)`. `format` defaults to `"mp3"` and
    /// `mime_type` to the one `format` implies.
    pub fn new(
        data: Vec<u8>,
        model: impl Into<String>,
        voice: Option<&str>,
        format: Option<&str>,
        mime_type: Option<&str>,
    ) -> Speech {
        let format = format.unwrap_or("mp3").to_string();
        Speech {
            data,
            model: model.into(),
            voice: voice.map(str::to_string),
            mime_type: mime_type.map_or_else(|| mime_type_for(&format), str::to_string),
            format,
            usage_entries: Vec::new(),
            input_tokens: None,
            output_tokens: None,
        }
    }

    fn with_tokens(mut self, usage: &Value) -> Speech {
        self.input_tokens = int(usage.get("prompt_tokens"));
        self.output_tokens = int(usage.get("completion_tokens"));
        self
    }

    /// The raw audio bytes. Alias for `data`, mirroring `Image#to_blob`.
    pub fn to_blob(&self) -> &[u8] {
        &self.data
    }

    /// Writes the audio to `path` and returns `path` as given.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<P> {
        std::fs::write(path.as_ref(), &self.data)?;
        Ok(path)
    }

    /// Usage across every provider attempt, or the usage this speech reported. Fields are `None`
    /// when the provider did not report them.
    pub fn tokens(&self) -> Tokens {
        if !self.usage_entries.is_empty() {
            return Tokens::aggregate(self.usage_entries.iter().map(|e| &e.tokens));
        }
        Tokens {
            input: self.input_tokens,
            output: self.output_tokens,
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

/// `RubyLLM.speak(input, model:, provider:, voice:, format:, provider_options:)`.
pub async fn speak(input: &str, options: SpeakOptions<'_>) -> Result<Speech> {
    run(input, options, None).await
}

/// `RubyLLM.speak(input, ...) { |chunk| ... }`: `on_chunk` receives each [`SpeechChunk`] as audio
/// arrives, and the complete [`Speech`] is still returned.
pub async fn speak_stream(
    input: &str,
    options: SpeakOptions<'_>,
    mut on_chunk: impl FnMut(&SpeechChunk) + Send,
) -> Result<Speech> {
    run(
        input,
        options,
        Some(&mut on_chunk as &mut (dyn FnMut(&SpeechChunk) + Send)),
    )
    .await
}

async fn run(
    input: &str,
    options: SpeakOptions<'_>,
    on_chunk: Option<&mut (dyn FnMut(&SpeechChunk) + Send)>,
) -> Result<Speech> {
    let config = options.config.clone().unwrap_or_else(crate::config);
    let model_id = options
        .model
        .unwrap_or(&config.default_speech_model)
        .to_string();
    let (model, provider) =
        resolve_model(&model_id, options.provider, options.assume_model_exists)?;
    provider.ensure_configured(&config)?;
    let connection = Connection::new(provider, config.clone())?;
    let family = Family::for_provider(provider)?;
    let (voice, format) = (options.voice, options.format);
    // `provider_options: {}` by default; `Null` would replace the payload in a deep merge.
    let provider_options = if options.provider_options.is_null() {
        json!({})
    } else {
        options.provider_options.clone()
    };
    let payload = family.render(input, &model.id, voice, format, &provider_options);

    let mut tracker = Tracker::default();
    let result = match on_chunk {
        None => post(&connection, family.url(&model.id), &payload, &mut tracker)
            .await
            .and_then(|body| family.parse(&body, &model.id, voice, format)),
        Some(on_chunk) => {
            family
                .stream(
                    provider,
                    &connection,
                    &model.id,
                    voice,
                    format,
                    payload,
                    &mut tracker,
                    on_chunk,
                )
                .await
        }
    };
    let mut speech = result?;
    speech.usage_entries = tracker.entries(
        Operation::Speech,
        provider,
        &model,
        speech.tokens(),
        speech.cost(),
    );
    Ok(speech)
}

/// `track_usage`: the failed attempts before the one that succeeded.
#[derive(Default)]
pub(crate) struct Tracker {
    retried: Vec<Tokens>,
}

impl Tracker {
    pub(crate) fn on_attempt(&mut self) -> impl FnMut(Option<&Error>) + Send + '_ {
        |previous: Option<&Error>| {
            if let Some(e) = previous {
                self.retried.push(failure_tokens(e, None));
            }
        }
    }

    /// `Tracker#succeed`: one entry per attempt, the last billed with the result's usage. The
    /// result's cost is kept when it has a total; otherwise the tokens are priced as audio.
    pub(crate) fn entries(
        self,
        operation: Operation,
        provider: Provider,
        model: &Model,
        tokens: Tokens,
        cost: Cost,
    ) -> Vec<UsageEntry> {
        let entry = |status, tokens: Tokens, cost: Option<Cost>| UsageEntry {
            id: UsageEntry::next_id(),
            operation,
            provider: provider.slug().into(),
            model: model.id.clone(),
            status,
            cost: cost
                .filter(|c| c.total().is_some())
                .unwrap_or_else(|| Cost::audio(&tokens, Some(model))),
            tokens,
        };
        let mut entries: Vec<UsageEntry> = self
            .retried
            .into_iter()
            .map(|t| entry(UsageStatus::Failed, t, None))
            .collect();
        entries.push(entry(UsageStatus::Succeeded, tokens, Some(cost)));
        entries
    }
}

/// POSTs JSON and returns the raw response bytes: speech endpoints answer with audio, not JSON.
async fn post(
    connection: &Connection,
    path: String,
    payload: &Value,
    tracker: &mut Tracker,
) -> Result<Vec<u8>> {
    let resp = connection
        .send_tracked(
            reqwest::Method::POST,
            &path,
            &[],
            true,
            &|req| req.json(payload),
            &mut tracker.on_attempt(),
        )
        .await?;
    Ok(resp
        .bytes()
        .await
        .map_err(|e| Error::ConnectionFailed(e.to_string()))?
        .to_vec())
}

/// Which speech seams a provider's protocol includes.
#[derive(Clone, Copy)]
enum Family {
    /// `ChatCompletions::Speech` (OpenAI and every OpenAI-compatible provider).
    OpenAI,
    /// `ChatCompletions::Speech` plus `GPUStack::Speech`'s stream flag.
    GPUStack,
    Mistral,
    XAI,
    OpenRouter,
    Gemini,
}

impl Family {
    fn for_provider(provider: Provider) -> Result<Family> {
        match provider {
            Provider::Mistral => Ok(Family::Mistral),
            Provider::XAI => Ok(Family::XAI),
            Provider::OpenRouter => Ok(Family::OpenRouter),
            Provider::Gemini => Ok(Family::Gemini),
            Provider::GPUStack => Ok(Family::GPUStack),
            Provider::Anthropic | Provider::TypeSafe => Err(Error::Api(
                format!("{} doesn't support speech generation", provider.display()),
                None,
            )),
            _ => Ok(Family::OpenAI),
        }
    }

    /// `speech_url`.
    fn url(self, model: &str) -> String {
        match self {
            Family::XAI => "tts".into(),
            Family::Gemini => format!("models/{model}:generateContent"),
            _ => "audio/speech".into(),
        }
    }

    /// `render_speech_payload`.
    fn render(
        self,
        input: &str,
        model: &str,
        voice: Option<&str>,
        format: Option<&str>,
        provider_options: &Value,
    ) -> Value {
        let mut payload = Map::new();
        let mut put = |key: &str, value: Option<Value>| {
            if let Some(value) = value {
                payload.insert(key.into(), value);
            }
        };
        match self {
            Family::OpenAI | Family::GPUStack => {
                put("model", Some(model.into()));
                put("input", Some(input.into()));
                put("voice", Some(voice.unwrap_or("alloy").into()));
                put("response_format", format.map(Into::into));
            }
            Family::Mistral => {
                put("model", Some(model.into()));
                put("input", Some(input.into()));
                put("voice_id", voice.map(Into::into));
                put("response_format", format.map(Into::into));
            }
            Family::OpenRouter => {
                put("model", Some(model.into()));
                put("input", Some(input.into()));
                put("voice", voice.map(Into::into));
                put("response_format", Some(format.unwrap_or("mp3").into()));
            }
            Family::XAI => {
                put("text", Some(input.into()));
                put("voice_id", voice.map(Into::into));
                put("language", Some("auto".into()));
                put("output_format", format.map(|f| json!({ "codec": f })));
                let mut payload = Value::Object(payload);
                deep_merge(&mut payload, provider_options);
                return payload;
            }
            Family::Gemini => {
                if let Some(format) = format {
                    tracing::debug!("Ignoring speech format {format}. Gemini returns PCM audio.");
                }
                let mut payload = json!({
                    "contents": [{ "role": "user", "parts": [{ "text": input }] }],
                    "generationConfig": {
                        "responseModalities": ["AUDIO"],
                        "speechConfig": { "voiceConfig": { "prebuiltVoiceConfig": { "voiceName": voice.unwrap_or("Kore") } } },
                    },
                    "model": model,
                });
                deep_merge(&mut payload, provider_options);
                return payload;
            }
        }
        // `.compact.merge(provider_options)`: provider options replace top-level keys.
        if let Some(options) = provider_options.as_object() {
            payload.extend(options.clone());
        }
        Value::Object(payload)
    }

    /// `parse_speech_response`.
    fn parse(
        self,
        body: &[u8],
        model: &str,
        voice: Option<&str>,
        format: Option<&str>,
    ) -> Result<Speech> {
        let format = format.unwrap_or("mp3");
        match self {
            Family::OpenAI | Family::GPUStack => Ok(Speech::new(
                body.to_vec(),
                model,
                Some(voice.unwrap_or("alloy")),
                Some(format),
                None,
            )),
            Family::OpenRouter => Ok(Speech::new(body.to_vec(), model, voice, Some(format), None)),
            Family::XAI => Ok(Speech::new(
                body.to_vec(),
                model,
                Some(voice.unwrap_or("eve")),
                Some(format),
                None,
            )),
            Family::Mistral => {
                let data: Value = serde_json::from_slice(body).unwrap_or(Value::Null);
                let audio = data.get("audio_data").and_then(Value::as_str).unwrap_or("");
                Ok(Speech::new(
                    decode(audio)?,
                    model,
                    voice,
                    Some(format),
                    None,
                ))
            }
            Family::Gemini => {
                let data: Value = serde_json::from_slice(body).unwrap_or(Value::Null);
                let audio = data
                    .pointer("/candidates/0/content/parts/0/inlineData/data")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        Error::Api(
                            "Unexpected response format from Gemini speech generation API".into(),
                            None,
                        )
                    })?;
                Ok(Speech::new(
                    decode(audio)?,
                    model,
                    Some(voice.unwrap_or("Kore")),
                    Some("pcm"),
                    None,
                ))
            }
        }
    }

    /// `stream_speech`.
    #[allow(clippy::too_many_arguments)]
    async fn stream(
        self,
        provider: Provider,
        connection: &Connection,
        model: &str,
        voice: Option<&str>,
        format: Option<&str>,
        mut payload: Value,
        tracker: &mut Tracker,
        on_chunk: &mut (dyn FnMut(&SpeechChunk) + Send),
    ) -> Result<Speech> {
        match self {
            Family::Gemini => Err(Error::Api(
                format!(
                    "{} doesn't support streaming speech with this protocol",
                    provider.display()
                ),
                None,
            )),
            Family::Mistral => {
                stream_mistral(connection, model, voice, format, payload, tracker, on_chunk).await
            }
            Family::XAI
                if payload
                    .get("with_timestamps")
                    .is_some_and(|v| !v.is_null() && v != &Value::Bool(false)) =>
            {
                Err(Error::Argument(
                    "xAI streaming speech does not accept with_timestamps".into(),
                ))
            }
            Family::GPUStack => {
                let format = format.unwrap_or("pcm");
                if let Some(object) = payload.as_object_mut() {
                    object.insert("stream".into(), true.into());
                    object.insert("response_format".into(), format.into());
                }
                self.stream_binary(
                    connection,
                    model,
                    voice,
                    Some(format),
                    &payload,
                    tracker,
                    on_chunk,
                )
                .await
            }
            _ => {
                self.stream_binary(
                    connection, model, voice, format, &payload, tracker, on_chunk,
                )
                .await
            }
        }
    }

    /// `stream_speech_response` over `BinaryStreaming#stream_binary`: audio bytes reach
    /// `on_chunk` as they arrive once the response is known to be audio. A successful response
    /// that is not audio is an error, raised from its JSON body when it has one.
    #[allow(clippy::too_many_arguments)]
    async fn stream_binary(
        self,
        connection: &Connection,
        model: &str,
        voice: Option<&str>,
        format: Option<&str>,
        payload: &Value,
        tracker: &mut Tracker,
        on_chunk: &mut (dyn FnMut(&SpeechChunk) + Send),
    ) -> Result<Speech> {
        let empty = self.parse(&[], model, voice, format)?;
        let resp = connection
            .send_tracked(
                reqwest::Method::POST,
                &self.url(model),
                &[],
                true,
                &|req| req.json(payload),
                &mut tracker.on_attempt(),
            )
            .await?;
        let status = resp.status().as_u16();
        let content_type = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.split(';').next())
            .unwrap_or("")
            .trim()
            .to_string();
        let audio =
            content_type.starts_with("audio/") || content_type == "application/octet-stream";
        let mut buffer = Vec::new();
        let mut stream = resp.bytes_stream();
        while let Some(bytes) = stream.next().await {
            let bytes = bytes.map_err(|e| Error::ConnectionFailed(e.to_string()))?;
            if bytes.is_empty() {
                continue;
            }
            buffer.extend_from_slice(&bytes);
            if audio {
                on_chunk(&SpeechChunk::new(
                    bytes.to_vec(),
                    &empty.format,
                    Some(&empty.mime_type),
                ));
            }
        }
        if !audio {
            let text = String::from_utf8_lossy(&buffer).into_owned();
            if let Ok(data) = serde_json::from_str::<Value>(&text) {
                return Err(error_for_status(streaming_error_status(&data), &text));
            }
            return Err(Error::Api(
                "Expected an audio response from the speech endpoint".into(),
                Some(ErrorResponse { status, body: text }),
            ));
        }
        self.parse(&buffer, model, voice, format)
    }
}

/// `ChatCompletions::Streaming#parse_streaming_error` status, falling back to 500 when the body
/// carries no typed `error` object.
fn streaming_error_status(data: &Value) -> u16 {
    match data.pointer("/error/type").and_then(Value::as_str) {
        Some("server_error") => 500,
        Some("rate_limit_exceeded" | "insufficient_quota") => 429,
        Some(_) => 400,
        None if data.get("error").is_some_and(Value::is_object) => 400,
        None => 500,
    }
}

/// `Mistral::Speech#stream_speech`: server-sent events carrying Base64 audio deltas and a final
/// event with usage.
async fn stream_mistral(
    connection: &Connection,
    model: &str,
    voice: Option<&str>,
    format: Option<&str>,
    mut payload: Value,
    tracker: &mut Tracker,
    on_chunk: &mut (dyn FnMut(&SpeechChunk) + Send),
) -> Result<Speech> {
    if let Some(object) = payload.as_object_mut() {
        object.insert("stream".into(), true.into());
    }
    let format = format.unwrap_or("mp3");
    let mut audio = Vec::new();
    let mut last: Option<Value> = None;
    let mut on_event = |_event: crate::transport::SseEvent, data: Value| -> Result<()> {
        match data.get("type").and_then(Value::as_str) {
            Some("speech.audio.delta") => {
                let encoded = data
                    .get("audio_data")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        Error::Api("Mistral speech delta carries no audio_data".into(), None)
                    })?;
                let bytes = decode(encoded)?;
                audio.extend_from_slice(&bytes);
                if !bytes.is_empty() {
                    on_chunk(&SpeechChunk::new(bytes, format, None));
                }
            }
            Some("speech.audio.done") => last = Some(data),
            _ => {}
        }
        Ok(())
    };
    connection
        .stream(
            "audio/speech",
            &payload,
            &[],
            &mut tracker.on_attempt(),
            &mut on_event,
            crate::protocols::streaming_error_status(
                crate::providers::ProtocolName::ChatCompletions,
            ),
        )
        .await?;
    let last = last.ok_or_else(|| {
        Error::Api(
            "Mistral speech stream ended before its completion event".into(),
            None,
        )
    })?;
    let usage = last.get("usage").cloned().unwrap_or_else(|| json!({}));
    Ok(Speech::new(audio, model, voice, Some(format), None).with_tokens(&usage))
}

fn decode(encoded: &str) -> Result<Vec<u8>> {
    base64::engine::general_purpose::STANDARD
        .decode(encoded.trim())
        .map_err(|e| Error::Api(format!("speech audio is not valid Base64: {e}"), None))
}

#[cfg(test)]
mod tests {
    use super::*;

    // speech_spec.rb "#mime_type uses the format when no explicit MIME type is provided".
    #[test]
    fn mime_type_follows_the_format() {
        assert_eq!(
            Speech::new(b"audio bytes".to_vec(), "tts-1", None, Some("wav"), None).mime_type,
            "audio/wav"
        );
        assert_eq!(
            Speech::new(Vec::new(), "tts-1", None, None, None).mime_type,
            "audio/mpeg"
        );
        assert_eq!(
            Speech::new(Vec::new(), "tts-1", None, Some("ogg"), None).mime_type,
            "audio/ogg"
        );
    }

    #[test]
    fn openai_payload_defaults_the_voice_and_lets_provider_options_win() {
        let payload = Family::OpenAI.render(
            "Hi",
            "tts-1",
            None,
            None,
            &json!({ "speed": 1.5, "voice": "nova" }),
        );
        assert_eq!(
            payload,
            json!({ "model": "tts-1", "input": "Hi", "voice": "nova", "speed": 1.5 })
        );
    }

    #[test]
    fn xai_nests_the_format_as_a_codec() {
        let payload = Family::XAI.render("Hi", "grok-tts", Some("ara"), Some("wav"), &json!({}));
        assert_eq!(
            payload,
            json!({ "text": "Hi", "voice_id": "ara", "language": "auto", "output_format": { "codec": "wav" } })
        );
    }
}
