//! RubyLLM 2.1's blocked Gemini transcriptions through the public API (upstream commits 6e006cc5
//! "Raise when Gemini blocks a transcription" and 1c0e06a4 "Keep the tokens of blocked Gemini
//! attempts"). Ruby stubs the generateContent endpoint with webmock; here a wiremock server
//! answers, so the port's render, parse, and usage ledger run in between. The private-seam
//! examples of `protocols/gemini/transcription_spec.rb`, `file_transcription_spec.rb`, and
//! `videos_spec.rb` live next to their functions (`src/transcription/spec_tests.rs`,
//! `src/video.rs`).

use std::sync::{Arc, Mutex};
use std::time::Duration;

use rust_llm::{Config, Error, TranscribeOptions, transcribe};
use serde_json::{Map, Value, json};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

type Events = Arc<Mutex<Vec<(String, Map<String, Value>)>>>;

const MODEL: &str = "gemini-2.5-flash";

fn ruby_wav() -> String {
    format!("{}/tests/fixtures/ruby.wav", env!("CARGO_MANIFEST_DIR"))
}

/// A Gemini server whose generateContent answers 200 with `body`.
async fn gemini_answering(body: Value) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(format!("/v1beta/models/{MODEL}:generateContent")))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&server)
        .await;
    server
}

/// A context pointing Gemini at `server`, with `CaptureInstrumenter`.
fn context(server: &MockServer) -> (Arc<Config>, Events) {
    let events: Events = Default::default();
    let sink = events.clone();
    let mut config = Config::default();
    config
        .set("gemini_api_key", "test")
        .set("gemini_api_base", format!("{}/v1beta", server.uri()));
    config.max_retries = 0;
    config.instrumenter = Some(Arc::new(
        move |name: &str, payload: &Map<String, Value>, _: Option<Duration>| {
            sink.lock()
                .unwrap()
                .push((name.to_string(), payload.clone()));
        },
    ));
    (Arc::new(config), events)
}

fn options(config: Arc<Config>) -> TranscribeOptions<'static> {
    TranscribeOptions {
        model: Some(MODEL),
        provider: Some("gemini"),
        config: Some(config),
        ..Default::default()
    }
}

// spec: transcription_spec.rb:42 blocked transcriptions > raises instead of returning an empty transcript when Gemini blocks the audio
#[tokio::test]
async fn raises_instead_of_returning_an_empty_transcript_when_gemini_blocks_the_audio() {
    let server = gemini_answering(json!({
        "promptFeedback": { "blockReason": "SAFETY" },
        "usageMetadata": { "promptTokenCount": 133, "totalTokenCount": 133 },
        "modelVersion": MODEL
    }))
    .await;
    let (config, _) = context(&server);

    let result = transcribe(ruby_wav().as_str(), options(config)).await;

    match result {
        Err(Error::ContentFilter(message, _)) => {
            assert_eq!(message, "Gemini blocked the transcription: SAFETY")
        }
        other => panic!("expected ContentFilterError, got {other:?}"),
    }
}

// spec: transcription_spec.rb:56 blocked transcriptions > records the tokens Gemini billed for the blocked attempt
#[tokio::test]
async fn records_the_tokens_gemini_billed_for_the_blocked_attempt() {
    let server = gemini_answering(json!({
        "candidates": [{ "finishReason": "SAFETY" }],
        "usageMetadata": { "promptTokenCount": 133, "candidatesTokenCount": 0, "totalTokenCount": 133 }
    }))
    .await;
    let (config, events) = context(&server);

    let result = transcribe(ruby_wav().as_str(), options(config)).await;

    assert!(
        matches!(result, Err(Error::ContentFilter(..))),
        "{result:?}"
    );
    let events = events.lock().unwrap();
    let usage = &events
        .iter()
        .find(|(name, _)| name == "usage.rust_llm")
        .expect("a usage.rust_llm event")
        .1;
    assert_eq!(usage["operation"], "transcription");
    assert_eq!(usage["status"], "failed");
    assert_eq!(
        usage["tokens"],
        json!({ "input_tokens": 133, "output_tokens": 0 })
    );
    assert!(!usage["cost"]["total"].is_null(), "{usage:?}");
}
