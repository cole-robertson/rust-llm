//! WebSocket transcription through the public API: `protocols/gemini/live_transcription_spec.rb`,
//! `protocols/xai/streaming_transcription_spec.rb`, and `transcription_google_spec.rb` replayed
//! from RubyLLM's WebSocket cassettes (`spec/fixtures/websocket_cassettes`) over a local
//! tokio-tungstenite server that asserts every sent frame equals the recording. Also the
//! OpenRouter and Gemini dedicated-transcription examples that go through `RubyLLM.transcribe`.

mod support;
#[path = "support/websocket.rs"]
mod websocket_support;

use std::sync::Arc;
use std::time::Duration;

use rust_llm::{
    Config, Error, TranscribeOptions, Transcription, TranscriptionChunk, UsageStatus, transcribe,
    transcribe_stream,
};
use serde_json::{Value, json};
use websocket_support::{Replay, WebsocketCassette, recorded_path};

fn ruby_wav() -> String {
    format!("{}/tests/fixtures/ruby.wav", env!("CARGO_MANIFEST_DIR"))
}

fn config_at(provider: &str, base: String) -> Arc<Config> {
    let mut config = Config::default();
    config
        .set(format!("{provider}_api_key"), "test-key")
        .set(format!("{provider}_api_base"), base);
    config.max_retries = 0;
    config.request_timeout = Duration::from_secs(10);
    Arc::new(config)
}

fn options<'a>(model: &'a str, provider: &'a str, config: Arc<Config>) -> TranscribeOptions<'a> {
    TranscribeOptions {
        model: Some(model),
        provider: Some(provider),
        config: Some(config),
        ..Default::default()
    }
}

async fn streamed(
    options: TranscribeOptions<'_>,
) -> (rust_llm::Result<Transcription>, Vec<TranscriptionChunk>) {
    let mut chunks = Vec::new();
    let result = transcribe_stream(ruby_wav().as_str(), options, |c| chunks.push(c.clone())).await;
    (result, chunks)
}

fn kinds(chunks: &[TranscriptionChunk]) -> Vec<&str> {
    chunks.iter().map(|c| c.kind.as_str()).collect()
}

fn deltas(chunks: &[TranscriptionChunk]) -> String {
    chunks.iter().filter_map(|c| c.delta.clone()).collect()
}

fn succeeded_once(t: &Transcription) {
    let statuses: Vec<UsageStatus> = t.usage_entries.iter().map(|e| e.status).collect();
    assert_eq!(statuses, [UsageStatus::Succeeded]);
}

// ---- Gemini Live ---------------------------------------------------------------------------

const LIVE: &str = "gemini-3.5-transcribe-live";

/// `stub_socket(incoming)`: a server that sends these events and records the client's frames.
fn scripted(incoming: Vec<Value>) -> WebsocketCassette {
    WebsocketCassette {
        url: String::new(),
        incoming,
        outgoing: Vec::new(),
    }
}

fn live_events() -> Vec<Value> {
    vec![
        json!({ "setupComplete": {} }),
        json!({ "serverContent": { "interimInputTranscription": { "text": "Hel" } } }),
        json!({ "serverContent": { "inputTranscription": { "text": "Hello." } },
                "usageMetadata": { "promptTokenCount": 12, "candidatesTokenCount": 2 } }),
        json!({ "serverContent": { "generationComplete": true } }),
    ]
}

async fn live_server(incoming: Vec<Value>) -> Replay {
    Replay::start(&scripted(incoming)).await
}

fn gemini_config(replay: &Replay) -> Arc<Config> {
    config_at("gemini", format!("{}/v1beta", replay.base()))
}

// spec: protocols/gemini/live_transcription_spec.rb:29 keeps partial text separate and waits for generation completion before returning final text and usage
#[tokio::test]
async fn live_keeps_partial_text_separate_and_waits_for_generation_completion() {
    let replay = live_server(live_events()).await;
    let (result, chunks) = streamed(options(LIVE, "gemini", gemini_config(&replay))).await;
    let t = result.expect("transcribe_stream");
    let session = replay.finish().await;

    assert_eq!(t.text.as_deref(), Some("Hello."));
    assert!((t.duration.unwrap() - 3.7).abs() < 1e-9);
    assert_eq!(t.words, None);
    assert_eq!((t.tokens().input, t.tokens().output), (Some(12), Some(2)));
    assert_eq!(
        kinds(&chunks),
        [
            TranscriptionChunk::PARTIAL,
            TranscriptionChunk::DELTA,
            TranscriptionChunk::DONE
        ]
    );
    assert_eq!(deltas(&chunks), "Hello.");
    // `have_received(:close).once`: one connection, closed with a close frame.
    assert_eq!(session.connections, 1);
    assert!(session.closed_cleanly);
    assert_eq!(
        session.path,
        "/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
    );
    assert!(
        session
            .headers
            .contains(&("x-goog-api-key".into(), "test-key".into()))
    );
}

// spec: protocols/gemini/live_transcription_spec.rb:43 does not mark an input transcript complete when its generation boundary never arrives
#[tokio::test]
async fn live_does_not_complete_without_its_generation_boundary() {
    let mut events = live_events();
    events.pop();
    let replay = live_server(events).await;
    let (result, chunks) = streamed(options(LIVE, "gemini", gemini_config(&replay))).await;

    let error = result.unwrap_err();
    assert!(
        error.to_string().contains("before generation completed"),
        "{error}"
    );
    assert!(!chunks.iter().any(TranscriptionChunk::is_done));
}

// spec: protocols/gemini/live_transcription_spec.rb:52 supports returning a completed Live transcript without a consumer block
#[tokio::test]
async fn live_returns_a_completed_transcript_without_a_consumer() {
    let replay = live_server(live_events()).await;
    let t = transcribe(
        ruby_wav().as_str(),
        options(LIVE, "gemini", gemini_config(&replay)),
    )
    .await
    .expect("transcribe");
    assert_eq!(t.text.as_deref(), Some("Hello."));
}

// spec: protocols/gemini/live_transcription_spec.rb:58 preserves provider errors and consumer exceptions without starting a second connection
// (Only the provider-error half: `transcribe_stream`'s callback returns `()`, so a consumer
// cannot raise into the transcription.)
#[tokio::test]
async fn live_preserves_provider_errors_without_starting_a_second_connection() {
    let replay = live_server(vec![
        json!({ "error": { "message": "Unsupported configuration" } }),
    ])
    .await;
    let (result, _) = streamed(options(LIVE, "gemini", gemini_config(&replay))).await;
    let error = result.unwrap_err();
    assert!(
        error.to_string().contains("Unsupported configuration"),
        "{error}"
    );
    assert_eq!(replay.finish().await.connections, 1);
}

// spec: protocols/gemini/live_transcription_spec.rb:83 waits for setupComplete before sending any audio
#[tokio::test]
async fn live_waits_for_setup_complete_before_sending_any_audio() {
    let replay = Replay::start_holding(&scripted(live_events()), Duration::from_millis(300)).await;
    let (result, _) = streamed(options(LIVE, "gemini", gemini_config(&replay))).await;
    result.expect("transcribe_stream");
    let session = replay.finish().await;

    assert_eq!(session.before_first_event, 1, "only the setup went out");
    assert!(session.outgoing[0].get("setup").is_some());
    assert_eq!(
        session.outgoing[1],
        json!({ "realtimeInput": { "activityStart": {} } })
    );
}

// spec: protocols/gemini/live_transcription_spec.rb:104 rejects unsupported metadata requests and overrides before connecting
#[tokio::test]
async fn live_rejects_unsupported_metadata_requests_before_connecting() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}/v1beta", listener.local_addr().unwrap());
    let config = config_at("gemini", base);
    let argument =
        |result: (rust_llm::Result<Transcription>, Vec<TranscriptionChunk>)| match result.0 {
            Err(Error::Argument(message)) => message,
            other => panic!("expected an argument error, got {other:?}"),
        };

    let timestamps = TranscribeOptions {
        timestamps: Some(vec!["word"]),
        ..options(LIVE, "gemini", config.clone())
    };
    assert!(argument(streamed(timestamps).await).contains("timestamps"));
    let speakers = TranscribeOptions {
        speaker_names: Some(vec!["Speaker".into()]),
        ..options(LIVE, "gemini", config.clone())
    };
    assert!(argument(streamed(speakers).await).contains("diarization"));
    let word_timestamps = TranscribeOptions {
        provider_options: json!({ "inputAudioTranscription": { "wordTimestamp": true } }),
        ..options(LIVE, "gemini", config.clone())
    };
    assert!(argument(streamed(word_timestamps).await).contains("word timestamps"));

    let accepted = tokio::time::timeout(Duration::from_millis(100), listener.accept()).await;
    assert!(accepted.is_err(), "no socket was opened");
}

// spec: transcription_google_spec.rb:38 streams partial and final transcription through #{provider} Live
#[tokio::test]
async fn streams_partial_and_final_transcription_through_gemini_live() {
    let cassette = WebsocketCassette::load("transcription_google_gemini");
    let replay = Replay::start(&cassette).await;
    let opts = TranscribeOptions {
        language: Some("en-US"),
        ..options(LIVE, "gemini", gemini_config(&replay))
    };
    let (result, chunks) = streamed(opts).await;
    let t = result.expect("transcribe_stream");
    let session = replay.finish().await;

    // `Recorded WebSocket request does not match` / `expect(uri.to_s).to eq(fixture.fetch('url'))`.
    assert_eq!(session.path, recorded_path(&cassette));
    assert_eq!(session.outgoing, cassette.outgoing);
    assert!(session.closed_cleanly);

    let text = t.text.clone().unwrap_or_default();
    assert!(
        text.contains("Ruby") && text.contains("developer happiness"),
        "{text}"
    );
    assert!(t.duration.unwrap() > 3.0);
    assert!(chunks.iter().any(TranscriptionChunk::is_partial));
    assert_eq!(deltas(&chunks), text);
    let last = chunks.last().unwrap();
    assert!(last.is_done());
    assert_eq!(last.text.as_deref(), Some(text.as_str()));
    assert_eq!(t.words, None);
    succeeded_once(&t);
}

// ---- xAI -----------------------------------------------------------------------------------

// spec: protocols/xai/streaming_transcription_spec.rb:99 streams transcription through the public API with typed chunks and word timing
#[tokio::test]
async fn xai_streams_transcription_with_typed_chunks_and_word_timing() {
    let cassette = WebsocketCassette::load("transcription_xai");
    let replay = Replay::start(&cassette).await;
    let opts = TranscribeOptions {
        language: Some("en"),
        speaker_names: Some(vec!["Speaker".into()]),
        ..options(
            "grok-stt",
            "xai",
            config_at("xai", format!("{}/v1", replay.base())),
        )
    };
    let (result, chunks) = streamed(opts).await;
    let t = result.expect("transcribe_stream");
    let session = replay.finish().await;

    assert_eq!(session.path, recorded_path(&cassette));
    assert_eq!(session.outgoing, cassette.outgoing);
    assert!(session.closed_cleanly);
    assert!(
        session
            .headers
            .contains(&("authorization".into(), "Bearer test-key".into()))
    );

    let text = t.text.clone().unwrap_or_default();
    assert!(
        text.contains("Ruby") && text.contains("developer happiness"),
        "{text}"
    );
    assert!(t.duration.unwrap() > 3.0);
    let words = t.words.clone().unwrap_or_default();
    assert!(!words.is_empty());
    assert!(
        words
            .iter()
            .all(|w| w.get("start").is_some() && w.get("end").is_some())
    );
    assert!(chunks.iter().any(TranscriptionChunk::is_partial));
    assert_eq!(deltas(&chunks), text);
    let last = chunks.last().unwrap();
    assert!(last.is_done());
    assert_eq!(last.text.as_deref(), Some(text.as_str()));
    succeeded_once(&t);
}

// ---- OpenRouter ----------------------------------------------------------------------------

const DIARIZATION: &str = "microsoft/mai-transcribe-2";

// spec: protocols/openrouter/transcription_spec.rb:10 sends JSON diarization options and preserves segment and word speakers with actual cost
#[tokio::test]
async fn openrouter_sends_json_diarization_options_and_preserves_speakers_with_cost() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/v1/audio/transcriptions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "text": "Ruby.", "duration": 4, "segments": [{ "text": "Ruby.", "speaker": 0 }],
            "words": [{ "word": "Ruby.", "speaker": 0 }], "usage": { "seconds": 4, "cost": 0.0001 }
        })))
        .expect(1)
        .mount(&server)
        .await;
    let opts = TranscribeOptions {
        speaker_names: Some(vec![]),
        provider_options: json!({ "provider": { "options": { "azure": { "custom": "value" } } } }),
        ..options(
            DIARIZATION,
            "openrouter",
            config_at("openrouter", format!("{}/api/v1", server.uri())),
        )
    };
    let t = transcribe(ruby_wav().as_str(), opts)
        .await
        .expect("transcribe");

    let requests = server.received_requests().await.unwrap();
    let payload: Value = serde_json::from_slice(&requests[0].body).unwrap();
    use base64::Engine;
    let audio =
        base64::engine::general_purpose::STANDARD.encode(std::fs::read(ruby_wav()).unwrap());
    assert_eq!(payload["input_audio"]["data"], json!(audio));
    assert_eq!(payload["response_format"], json!("verbose_json"));
    assert_eq!(
        payload["provider"]["options"]["azure"]["diarization"],
        json!({ "enabled": true })
    );
    assert_eq!(
        payload["provider"]["options"]["deepgram"],
        json!({ "diarize": true })
    );
    assert_eq!(
        payload["provider"]["options"]["azure"]["custom"],
        json!("value")
    );

    assert_eq!(t.text.as_deref(), Some("Ruby."));
    assert_eq!(t.segments.as_ref().unwrap()[0]["speaker"], json!(0));
    assert_eq!(t.words.as_ref().unwrap()[0]["speaker"], json!(0));
    assert_eq!(t.cost().total(), Some(0.0001));
}

// spec: protocols/openrouter/transcription_spec.rb:29 rejects unsupported speaker identity, prompt and output options before sending audio
#[tokio::test]
async fn openrouter_rejects_unsupported_options_before_sending_audio() {
    let server = wiremock::MockServer::start().await;
    let config = config_at("openrouter", format!("{}/api/v1", server.uri()));
    let base = || options(DIARIZATION, "openrouter", config.clone());
    let cases = [
        TranscribeOptions {
            speaker_names: Some(vec!["Alice".into()]),
            ..base()
        },
        TranscribeOptions {
            speaker_references: Some(vec![rust_llm::Attachment::new(ruby_wav())]),
            ..base()
        },
        TranscribeOptions {
            prompt: Some("Ruby"),
            ..base()
        },
        TranscribeOptions {
            format: Some("srt"),
            ..base()
        },
        TranscribeOptions {
            speaker_names: Some(vec![]),
            format: Some("json"),
            ..base()
        },
    ];
    for opts in cases {
        let result = transcribe(ruby_wav().as_str(), opts).await;
        assert!(matches!(result, Err(Error::Argument(_))), "{result:?}");
    }
    assert!(server.received_requests().await.unwrap().is_empty());
}

// spec: protocols/openrouter/transcription_spec.rb:37 transcribes a real recording with speaker labels and reported duration and cost
#[tokio::test]
async fn openrouter_transcribes_a_real_recording_with_speakers_duration_and_cost() {
    let name = "protocols_openrouter_transcription_transcribes_a_real_recording_with_speaker_labels_and_reported_duration_and_cost";
    let cassette = support::Cassette::start(name)
        .await
        .expect("run bin/convert-cassettes 'protocols_openrouter_transcription_*'");
    let opts = TranscribeOptions {
        speaker_names: Some(vec![]),
        provider_options: json!({ "timestamp_granularities": ["segment", "word"] }),
        ..options(
            DIARIZATION,
            "openrouter",
            support::config_for(&cassette, "openrouter"),
        )
    };
    let t = transcribe(ruby_wav().as_str(), opts)
        .await
        .expect("transcribe");
    cassette.assert_all_matched().await;

    assert!(t.text.as_deref().unwrap_or_default().contains("Ruby"));
    let speakers = |items: &Option<Vec<Value>>| -> Vec<Value> {
        items
            .iter()
            .flatten()
            .map(|i| i["speaker"].clone())
            .collect()
    };
    assert!(speakers(&t.segments).contains(&json!(0)));
    assert!(speakers(&t.words).contains(&json!(0)));
    assert!(t.duration.unwrap() > 0.0);
    assert!(t.cost().total().unwrap() > 0.0);
}

// ---- Gemini dedicated transcription --------------------------------------------------------

// UPSTREAM-REMOVED in 2.1 (was spec: protocols/gemini/file_transcription_spec.rb:95) rejects incompatible custom vocabulary and timestamp options before requesting an interaction
#[tokio::test]
async fn gemini_rejects_custom_vocabulary_with_timestamps_before_requesting_an_interaction() {
    let server = wiremock::MockServer::start().await;
    let opts = TranscribeOptions {
        timestamps: Some(vec!["word"]),
        prompt: Some("RubyLLM"),
        ..options(
            "gemini-3.5-transcribe",
            "gemini",
            config_at("gemini", format!("{}/v1beta", server.uri())),
        )
    };
    let result = transcribe(ruby_wav().as_str(), opts).await;
    assert!(
        matches!(&result, Err(Error::Argument(m)) if m.contains("custom vocabulary cannot be combined")),
        "{result:?}"
    );
    assert!(server.received_requests().await.unwrap().is_empty());
}
