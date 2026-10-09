//! `RubyLLM.speak`, replayed from RubyLLM's `speech_*` cassettes. Assertions follow
//! `spec/ruby_llm/speech_spec.rb` and `spec/ruby_llm/speech_streaming_spec.rb`.
//!
//! The cassettes store binary audio as scrubbed UTF-8, so replayed audio is not byte-identical to
//! what the provider sent; the tests assert what the specs assert (size, format, MIME type, and
//! that streamed chunks add up to the returned speech).

mod support;

use std::sync::Arc;

use rust_llm::message::Operation;
use rust_llm::{Config, Error, SpeakOptions, Speech, UsageStatus, speak, speak_stream};
use serde_json::{Value, json};
use support::{Cassette, config_for};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const TEXT: &str = "Ruby is a programming language designed for developer happiness.";
const STREAM_TEXT: &str = "Ruby makes it easy to build applications with streaming audio. \
                           You can start listening while the rest of this sentence is generated.";

async fn start(name: &str) -> Cassette {
    Cassette::start(name)
        .await
        .unwrap_or_else(|| panic!("missing cassette {name}; run bin/convert-cassettes 'speech_*'"))
}

fn options<'a>(
    model: &'a str,
    provider: &'a str,
    voice: Option<&'a str>,
    config: Arc<Config>,
) -> SpeakOptions<'a> {
    SpeakOptions {
        model: Some(model),
        provider: Some(provider),
        voice,
        config: Some(config),
        ..Default::default()
    }
}

/// The ledger records one succeeded `speech` operation for the call.
fn billed_once_as_speech(speech: &Speech, provider: &str, model: &str) {
    assert_eq!(speech.usage_entries.len(), 1);
    let entry = &speech.usage_entries[0];
    assert_eq!(entry.operation, Operation::Speech);
    assert_eq!(entry.operation.as_str(), "speech");
    assert_eq!(entry.status, UsageStatus::Succeeded);
    assert_eq!(
        (entry.provider.as_str(), entry.model.as_str()),
        (provider, model)
    );
}

// ---- basic functionality: `it "#{provider}/#{model} can speak"` ------------------------------

async fn can_speak(provider: &str, model: &str, voice: Option<&str>, slug: &str) -> Speech {
    let cassette = start(&format!(
        "speech_basic_functionality_{provider}_{slug}_can_speak"
    ))
    .await;
    let speech = speak(
        TEXT,
        options(model, provider, voice, config_for(&cassette, provider)),
    )
    .await
    .expect("speak");
    assert!(
        speech.data.len() > 1000,
        "a real recording is larger than 1KB, got {}",
        speech.data.len()
    );
    assert_eq!(speech.model, model);
    assert!(
        speech.mime_type.starts_with("audio/"),
        "{}",
        speech.mime_type
    );
    billed_once_as_speech(&speech, provider, model);
    cassette.assert_all_matched().await;
    speech
}

#[tokio::test]
async fn openai_gpt_4o_mini_tts_can_speak() {
    let speech = can_speak("openai", "gpt-4o-mini-tts", None, "gpt-4o-mini-tts").await;
    assert_eq!(
        speech.voice.as_deref(),
        Some("alloy"),
        "OpenAI's default voice"
    );
    assert_eq!(
        (speech.format.as_str(), speech.mime_type.as_str()),
        ("mp3", "audio/mpeg")
    );
}

#[tokio::test]
async fn gemini_2_5_flash_preview_tts_can_speak() {
    let speech = can_speak(
        "gemini",
        "gemini-2.5-flash-preview-tts",
        None,
        "gemini-2_5-flash-preview-tts",
    )
    .await;
    assert_eq!(
        speech.voice.as_deref(),
        Some("Kore"),
        "Gemini's default voice"
    );
    assert_eq!(
        (speech.format.as_str(), speech.mime_type.as_str()),
        ("pcm", "audio/pcm"),
        "Gemini returns PCM"
    );
    // The inline Base64 decodes exactly (it is ASCII, so the cassette kept it intact).
    let body: Value = serde_json::from_str(
        &support::load("speech_basic_functionality_gemini_gemini-2_5-flash-preview-tts_can_speak")
            .unwrap()[0]
            .response_body,
    )
    .unwrap();
    let encoded = body
        .pointer("/candidates/0/content/parts/0/inlineData/data")
        .and_then(Value::as_str)
        .unwrap();
    use base64::Engine;
    assert_eq!(
        speech.data,
        base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .unwrap()
    );
}

#[tokio::test]
async fn mistral_voxtral_mini_tts_latest_can_speak() {
    let speech = can_speak(
        "mistral",
        "voxtral-mini-tts-latest",
        Some("en_paul_neutral"),
        "voxtral-mini-tts-latest",
    )
    .await;
    assert_eq!(speech.voice.as_deref(), Some("en_paul_neutral"));
    assert!(
        speech.data.starts_with(b"ID3"),
        "Mistral's Base64 audio_data decodes to an MP3"
    );
}

#[tokio::test]
async fn openrouter_kokoro_can_speak() {
    let speech = can_speak(
        "openrouter",
        "hexgrad/kokoro-82m",
        Some("af_bella"),
        "hexgrad_kokoro-82m",
    )
    .await;
    assert_eq!(speech.voice.as_deref(), Some("af_bella"));
    assert_eq!(speech.format, "mp3");
}

#[tokio::test]
async fn xai_grok_tts_can_speak() {
    let speech = can_speak("xai", "grok-tts", None, "grok-tts").await;
    assert_eq!(speech.voice.as_deref(), Some("eve"), "xAI's default voice");
}

// ---- streaming: `it "streams speech from #{provider}"` ------------------------------------

async fn streams_speech(
    provider: &str,
    model: &str,
    voice: Option<&str>,
) -> (Speech, Vec<Vec<u8>>) {
    let cassette = start(&format!("speech_streams_speech_from_{provider}")).await;
    let mut chunks: Vec<Vec<u8>> = Vec::new();
    let mut formats = Vec::new();
    let speech = speak_stream(
        STREAM_TEXT,
        options(model, provider, voice, config_for(&cassette, provider)),
        |chunk| {
            chunks.push(chunk.to_blob().to_vec());
            formats.push((chunk.format.clone(), chunk.mime_type.clone()));
        },
    )
    .await
    .expect("speak_stream");
    assert!(speech.to_blob().len() > 1000);
    assert!(!chunks.is_empty());
    assert_eq!(
        chunks.concat(),
        speech.to_blob(),
        "chunks are consecutive bytes of the returned speech"
    );
    assert!(
        formats
            .iter()
            .all(|f| *f == (speech.format.clone(), speech.mime_type.clone()))
    );
    billed_once_as_speech(&speech, provider, model);
    cassette.assert_all_matched().await;
    (speech, chunks)
}

#[tokio::test]
async fn streams_speech_from_openai() {
    streams_speech("openai", "gpt-4o-mini-tts", None).await;
}

#[tokio::test]
async fn streams_speech_from_openrouter() {
    streams_speech("openrouter", "hexgrad/kokoro-82m", Some("af_bella")).await;
}

#[tokio::test]
async fn streams_speech_from_mistral() {
    let (speech, chunks) = streams_speech(
        "mistral",
        "voxtral-mini-tts-latest",
        Some("en_paul_neutral"),
    )
    .await;
    assert!(
        chunks.len() > 1,
        "each speech.audio.delta event is its own chunk"
    );
    // The speech.audio.done event reports usage.
    assert_eq!(speech.tokens().input, Some(134));
    assert_eq!(speech.tokens().output, Some(153600));
}

// ---- `.speak`, `#save`, `#mime_type` ------------------------------------------------------

async fn audio_server(content_type: &str, body: &[u8]) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/speech"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(body.to_vec(), content_type))
        .mount(&server)
        .await;
    server
}

fn openai_config(server: &MockServer) -> Config {
    let mut config = Config::default();
    config
        .set("openai_api_base", format!("{}/v1", server.uri()))
        .set("openai_api_key", "test-key");
    config.max_retries = 0;
    config
}

fn sent_json(request: &wiremock::Request) -> Value {
    serde_json::from_slice(&request.body).expect("json body")
}

#[tokio::test]
async fn uses_the_configured_default_speech_model() {
    assert_eq!(
        Config::default().default_speech_model,
        "gpt-4o-mini-tts-2025-12-15"
    );
    let server = audio_server("audio/mpeg", b"audio bytes").await;
    let config = Arc::new(openai_config(&server));
    let speech = speak(
        "Hello",
        SpeakOptions {
            config: Some(config),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(speech.model, "gpt-4o-mini-tts-2025-12-15");
    let requests = server.received_requests().await.unwrap();
    assert_eq!(
        sent_json(&requests[0]),
        json!({ "model": "gpt-4o-mini-tts-2025-12-15", "input": "Hello", "voice": "alloy" })
    );
}

#[tokio::test]
async fn works_from_a_context_with_its_own_default_speech_model() {
    let server = audio_server("audio/mpeg", b"audio bytes").await;
    let mut config = openai_config(&server);
    config.default_speech_model = "tts-1".into();
    let speech = speak(
        "Hello",
        SpeakOptions {
            config: Some(Arc::new(config)),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(speech.model, "tts-1");
    assert_eq!(
        sent_json(&server.received_requests().await.unwrap()[0])["model"],
        "tts-1"
    );
}

#[tokio::test]
async fn save_writes_the_audio_bytes_and_returns_the_path() {
    let speech = Speech::new(b"audio bytes".to_vec(), "tts-1", None, None, None);
    let path = std::env::temp_dir().join(format!("rust_llm_speech_{}.mp3", uuid::Uuid::new_v4()));
    assert_eq!(speech.save(path.clone()).unwrap(), path);
    assert_eq!(std::fs::read(&path).unwrap(), b"audio bytes");
    std::fs::remove_file(&path).ok();
}

#[test]
fn mime_type_uses_the_format_when_none_is_given() {
    assert_eq!(
        Speech::new(b"audio bytes".to_vec(), "tts-1", None, Some("wav"), None).mime_type,
        "audio/wav"
    );
}

// ---- speech_streaming_spec.rb binary-stream rules -----------------------------------------

#[tokio::test]
async fn rejects_a_block_for_a_speech_protocol_without_streaming_support() {
    let mut config = Config::default();
    config
        .set("gemini_api_key", "test-key")
        .set("gemini_api_base", "http://127.0.0.1:9");
    let opts = SpeakOptions {
        model: Some("gemini-2.5-flash-preview-tts"),
        provider: Some("gemini"),
        config: Some(Arc::new(config)),
        ..Default::default()
    };
    let error = speak_stream("Hello Ruby.", opts, |_| {}).await.unwrap_err();
    assert!(
        error
            .to_string()
            .contains("doesn't support streaming speech"),
        "{error}"
    );
}

#[tokio::test]
async fn never_yields_a_json_error_response_as_audio() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(429).set_body_raw(
            r#"{"error":{"message":"Too many requests"}}"#,
            "application/json",
        ))
        .mount(&server)
        .await;
    let config = Arc::new(openai_config(&server));
    let mut chunks = 0;
    let error = speak_stream(
        "Hello",
        SpeakOptions {
            model: Some("gpt-4o-mini-tts"),
            config: Some(config),
            ..Default::default()
        },
        |_| chunks += 1,
    )
    .await
    .unwrap_err();
    assert!(
        matches!(error, Error::RateLimit(ref m, _) if m == "Too many requests"),
        "{error:?}"
    );
    assert_eq!(chunks, 0);
}

#[tokio::test]
async fn does_not_treat_successful_json_or_html_as_audio() {
    let server = audio_server("text/html", b"<html>Service unavailable</html>").await;
    let config = Arc::new(openai_config(&server));
    let opts = SpeakOptions {
        model: Some("gpt-4o-mini-tts"),
        config: Some(config),
        ..Default::default()
    };
    let error = speak_stream("Hello", opts, |_| panic!("audio must not be delivered"))
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("Expected an audio response"),
        "{error}"
    );

    let server = audio_server(
        "application/json",
        br#"{"error":{"type":"server_error","message":"Synthesis failed"}}"#,
    )
    .await;
    let config = Arc::new(openai_config(&server));
    let opts = SpeakOptions {
        model: Some("gpt-4o-mini-tts"),
        config: Some(config),
        ..Default::default()
    };
    let error = speak_stream("Hello", opts, |_| panic!("audio must not be delivered"))
        .await
        .unwrap_err();
    assert!(
        matches!(error, Error::Server(ref m, _) if m == "Synthesis failed"),
        "{error:?}"
    );
}

#[tokio::test]
async fn retries_a_failure_before_audio_delivery_and_records_both_attempts() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(429)
                .set_body_raw(r#"{"error":{"message":"Try again"}}"#, "application/json"),
        )
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(b"ID3audio".to_vec(), "audio/mpeg"))
        .mount(&server)
        .await;
    let mut config = openai_config(&server);
    config.max_retries = 1;
    config.retry_interval = 0.0;
    let mut chunks = Vec::new();
    let opts = SpeakOptions {
        model: Some("gpt-4o-mini-tts"),
        config: Some(Arc::new(config)),
        ..Default::default()
    };
    let speech = speak_stream("Hello", opts, |c| chunks.extend_from_slice(&c.data))
        .await
        .unwrap();
    assert_eq!(server.received_requests().await.unwrap().len(), 2);
    assert_eq!(chunks, b"ID3audio");
    assert_eq!(speech.data, b"ID3audio");
    let statuses: Vec<UsageStatus> = speech.usage_entries.iter().map(|e| e.status).collect();
    assert_eq!(statuses, [UsageStatus::Failed, UsageStatus::Succeeded]);
}
