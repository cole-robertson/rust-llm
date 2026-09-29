//! `RubyLLM.transcribe`, replayed from RubyLLM's `transcription_*` cassettes. Assertions follow
//! `spec/ruby_llm/transcription_spec.rb`, `transcription_timestamps_spec.rb`, and
//! `transcription_google_spec.rb`. Multipart uploads can't be JSON-compared, so each one asserts
//! the method, path, and every form field (name, order, value, filename, content type, and the
//! audio bytes) against the recording.

mod support;

use std::sync::Arc;

use rust_llm::message::Operation;
use rust_llm::{
    Config, Error, TranscribeOptions, Transcription, TranscriptionChunk, UsageStatus, transcribe,
    transcribe_stream,
};
use serde_json::Value;
use support::{Cassette, Interaction, config_for};

fn fixture(name: &str) -> String {
    format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
}

fn ruby_wav() -> String {
    fixture("ruby.wav")
}

async fn start(name: &str) -> Cassette {
    Cassette::start(name).await.unwrap_or_else(|| {
        panic!("missing cassette {name}; run bin/convert-cassettes 'transcription_*'")
    })
}

fn options<'a>(model: &'a str, provider: &'a str, config: Arc<Config>) -> TranscribeOptions<'a> {
    TranscribeOptions {
        model: Some(model),
        provider: Some(provider),
        config: Some(config),
        ..Default::default()
    }
}

fn says_ruby(t: &Transcription) {
    let text = t.text.as_deref().unwrap_or_default();
    assert!(
        text.to_lowercase().contains("ruby"),
        "transcript mentions Ruby: {text:?}"
    );
}

/// The ledger records one succeeded `transcription` operation for the call.
fn billed_once_as_transcription(t: &Transcription, provider: &str, model: &str) {
    let statuses: Vec<UsageStatus> = t.usage_entries.iter().map(|e| e.status).collect();
    assert_eq!(statuses, [UsageStatus::Succeeded]);
    let entry = &t.usage_entries[0];
    assert_eq!(entry.operation, Operation::Transcription);
    assert_eq!(entry.operation.as_str(), "transcription");
    assert_eq!(
        (entry.provider.as_str(), entry.model.as_str()),
        (provider, model)
    );
}

// ---- multipart form comparison ------------------------------------------------------------

#[derive(Debug)]
struct Part {
    name: String,
    filename: Option<String>,
    content_type: Option<String>,
    body: Vec<u8>,
}

fn find(haystack: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    haystack
        .get(from..)?
        .windows(needle.len())
        .position(|w| w == needle)
        .map(|p| p + from)
}

fn parse_multipart(body: &[u8], boundary: &str) -> Vec<Part> {
    let delimiter = format!("--{boundary}");
    let mut parts = Vec::new();
    let mut at = find(body, delimiter.as_bytes(), 0).expect("first boundary") + delimiter.len();
    while body.get(at..at + 2) == Some(b"\r\n") {
        let head_end = find(body, b"\r\n\r\n", at).expect("part headers");
        let head = String::from_utf8_lossy(&body[at + 2..head_end]).to_string();
        let next =
            find(body, format!("\r\n{delimiter}").as_bytes(), head_end).expect("next boundary");
        let header = |key: &str| {
            head.lines().find_map(|l| {
                let (k, v) = l.split_once(':')?;
                k.trim()
                    .eq_ignore_ascii_case(key)
                    .then(|| v.trim().to_string())
            })
        };
        let disposition = header("content-disposition").unwrap_or_default();
        let attr = |key: &str| {
            let marker = format!("{key}=\"");
            let start = disposition.find(&marker)? + marker.len();
            Some(disposition[start..start + disposition[start..].find('"')?].to_string())
        };
        parts.push(Part {
            name: attr("name").unwrap_or_default(),
            filename: attr("filename"),
            content_type: header("content-type"),
            body: body[head_end + 4..next].to_vec(),
        });
        at = next + 2 + delimiter.len();
    }
    parts
}

/// The sent form matches RubyLLM's: same fields in the same order with the same values, and a
/// file part named like Ruby's, typed like Ruby's, carrying exactly the fixture's bytes (the
/// cassette's copy is lossy UTF-8, so the bytes are checked against the fixture and Ruby's
/// recorded `Content-Length`).
fn assert_same_form(recorded: &Interaction, sent: &wiremock::Request, audio: &[u8]) {
    let first = recorded
        .request_body
        .lines()
        .next()
        .expect("recorded multipart body");
    let expected = parse_multipart(
        recorded.request_body.as_bytes(),
        first.trim().strip_prefix("--").expect("boundary line"),
    );
    let content_type = sent
        .headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    assert!(
        content_type.starts_with("multipart/form-data"),
        "transcription upload must be multipart, sent {content_type}"
    );
    let boundary = content_type
        .split("boundary=")
        .nth(1)
        .expect("multipart boundary");
    let actual = parse_multipart(&sent.body, boundary);
    let names = |parts: &[Part]| parts.iter().map(|p| p.name.clone()).collect::<Vec<_>>();
    assert_eq!(
        names(&actual),
        names(&expected),
        "multipart field names and order"
    );
    for (e, a) in expected.iter().zip(&actual) {
        assert_eq!(a.filename, e.filename, "filename of {}", e.name);
        assert_eq!(a.content_type, e.content_type, "content type of {}", e.name);
        if e.filename.is_some() {
            assert_eq!(a.body, audio, "the file part is the audio's bytes");
            let recorded_len = recorded
                .request_body
                .lines()
                .find_map(|l| l.strip_prefix("Content-Length: "))
                .expect("recorded length");
            assert_eq!(a.body.len().to_string(), recorded_len.trim());
        } else {
            assert_eq!(
                String::from_utf8_lossy(&a.body),
                String::from_utf8_lossy(&e.body),
                "value of {}",
                e.name
            );
        }
    }
}

/// Replays a multipart cassette: the transcription, plus the form checked against Ruby's.
async fn multipart(
    name: &str,
    provider: &str,
    opts: impl FnOnce(Arc<Config>) -> TranscribeOptions<'static>,
) -> Transcription {
    let cassette = start(name).await;
    let t = transcribe(ruby_wav().as_str(), opts(config_for(&cassette, provider)))
        .await
        .expect("transcribe");
    cassette.assert_all_matched().await;
    let requests = cassette
        .server
        .received_requests()
        .await
        .unwrap_or_default();
    assert_eq!(requests[0].method.as_str(), "POST");
    assert_same_form(
        &support::load(name).unwrap()[0],
        &requests[0],
        &std::fs::read(ruby_wav()).unwrap(),
    );
    t
}

fn with<'a>(
    model: &'a str,
    provider: &'a str,
) -> impl FnOnce(Arc<Config>) -> TranscribeOptions<'a> {
    move |config| options(model, provider, config)
}

fn with_language<'a>(
    model: &'a str,
    provider: &'a str,
) -> impl FnOnce(Arc<Config>) -> TranscribeOptions<'a> {
    move |config| TranscribeOptions {
        language: Some("en"),
        ..options(model, provider, config)
    }
}

fn with_speakers<'a>(
    model: &'a str,
    provider: &'a str,
) -> impl FnOnce(Arc<Config>) -> TranscribeOptions<'a> {
    move |config| TranscribeOptions {
        speaker_names: Some(vec!["Speaker".into()]),
        ..options(model, provider, config)
    }
}

// ---- basic functionality: every provider `can transcribe audio` / `with language hint` -------

const BASIC: &str = "transcription_basic_functionality";

#[tokio::test]
async fn openai_whisper_1_can_transcribe_audio() {
    let t = multipart(
        &format!("{BASIC}_openai_whisper-1_can_transcribe_audio"),
        "openai",
        with("whisper-1", "openai"),
    )
    .await;
    says_ruby(&t);
    assert_eq!(t.model, "whisper-1");
    assert_eq!(t.duration, Some(4.0), "usage.seconds is the duration");
    billed_once_as_transcription(&t, "openai", "whisper-1");
}

#[tokio::test]
async fn openai_whisper_1_can_transcribe_with_language_hint() {
    let name = format!("{BASIC}_openai_whisper-1_can_transcribe_with_language_hint");
    let t = multipart(&name, "openai", with_language("whisper-1", "openai")).await;
    says_ruby(&t);
    assert_eq!(t.model, "whisper-1");
}

#[tokio::test]
async fn openai_gpt_4o_transcribe_diarize_can_transcribe_audio() {
    let model = "gpt-4o-transcribe-diarize";
    let t = multipart(
        &format!("{BASIC}_openai_{model}_can_transcribe_audio"),
        "openai",
        with(model, "openai"),
    )
    .await;
    says_ruby(&t);
    assert_eq!(t.model, model);
    assert_eq!(t.duration, Some(3.7));
    let segments = t.segments.as_ref().expect("diarized_json returns segments");
    assert_eq!(segments[0]["speaker"], "A");
    assert_eq!((t.tokens().input, t.tokens().output), (Some(37), Some(144)));
    billed_once_as_transcription(&t, "openai", model);
}

#[tokio::test]
async fn openai_gpt_4o_transcribe_diarize_can_transcribe_with_language_hint() {
    let model = "gpt-4o-transcribe-diarize";
    let t = multipart(
        &format!("{BASIC}_openai_{model}_can_transcribe_with_language_hint"),
        "openai",
        with_language(model, "openai"),
    )
    .await;
    says_ruby(&t);
    assert_eq!(t.model, model);
}

#[tokio::test]
async fn mistral_voxtral_mini_latest_can_transcribe_audio() {
    let model = "voxtral-mini-latest";
    let t = multipart(
        &format!("{BASIC}_mistral_{model}_can_transcribe_audio"),
        "mistral",
        with(model, "mistral"),
    )
    .await;
    says_ruby(&t);
    assert_eq!(t.model, model);
    assert_eq!(
        t.duration,
        Some(3.0),
        "Mistral reports prompt_audio_seconds"
    );
    assert_eq!((t.tokens().input, t.tokens().output), (Some(3), Some(11)));
    billed_once_as_transcription(&t, "mistral", model);
}

#[tokio::test]
async fn mistral_voxtral_mini_latest_can_transcribe_with_language_hint() {
    let model = "voxtral-mini-latest";
    let t = multipart(
        &format!("{BASIC}_mistral_{model}_can_transcribe_with_language_hint"),
        "mistral",
        with_language(model, "mistral"),
    )
    .await;
    says_ruby(&t);
    assert_eq!(t.model, model);
}

#[tokio::test]
async fn mistral_labels_segments_with_speakers_when_speaker_names_are_given() {
    let model = "voxtral-mini-latest";
    let name = format!(
        "{BASIC}_mistral_{model}_labels_segments_with_speakers_when_speaker_names_are_given"
    );
    let t = multipart(&name, "mistral", with_speakers(model, "mistral")).await;
    says_ruby(&t);
    let segments = t.segments.as_ref().expect("segments");
    assert!(segments[0].get("speaker_id").is_some(), "{segments:?}");
}

#[tokio::test]
async fn xai_grok_stt_can_transcribe_audio() {
    let t = multipart(
        &format!("{BASIC}_xai_grok-stt_can_transcribe_audio"),
        "xai",
        with("grok-stt", "xai"),
    )
    .await;
    says_ruby(&t);
    assert_eq!(t.model, "grok-stt");
    assert_eq!((t.language.as_deref(), t.duration), (Some("en"), Some(3.7)));
    assert!(t.words.as_ref().is_some_and(|w| !w.is_empty()));
    billed_once_as_transcription(&t, "xai", "grok-stt");
}

#[tokio::test]
async fn xai_grok_stt_can_transcribe_with_language_hint() {
    let t = multipart(
        &format!("{BASIC}_xai_grok-stt_can_transcribe_with_language_hint"),
        "xai",
        with_language("grok-stt", "xai"),
    )
    .await;
    says_ruby(&t);
    assert_eq!(t.model, "grok-stt");
}

#[tokio::test]
async fn xai_labels_words_with_speakers_when_speaker_names_are_given() {
    let name =
        format!("{BASIC}_xai_grok-stt_labels_words_with_speakers_when_speaker_names_are_given");
    let t = multipart(&name, "xai", with_speakers("grok-stt", "xai")).await;
    says_ruby(&t);
    let words = t.words.as_ref().expect("words");
    assert!(words[0].get("speaker").is_some(), "{words:?}");
}

/// JSON-bodied providers: the replay server compares the body against Ruby's exactly.
async fn json_bodied(
    name: &str,
    provider: &str,
    model: &str,
    language: Option<&'static str>,
) -> Transcription {
    let cassette = start(name).await;
    let opts = TranscribeOptions {
        language,
        ..options(model, provider, config_for(&cassette, provider))
    };
    let t = transcribe(ruby_wav().as_str(), opts)
        .await
        .expect("transcribe");
    cassette.assert_all_matched().await;
    says_ruby(&t);
    assert_eq!(t.model, model);
    billed_once_as_transcription(&t, provider, model);
    t
}

#[tokio::test]
async fn gemini_2_5_flash_can_transcribe_audio() {
    let t = json_bodied(
        &format!("{BASIC}_gemini_gemini-2_5-flash_can_transcribe_audio"),
        "gemini",
        "gemini-2.5-flash",
        None,
    )
    .await;
    assert_eq!(t.tokens().input, Some(133));
    assert_eq!(t.tokens().output, Some(30), "candidates plus thoughts");
    // gemini-2.5-flash prices audio input at $1/M and text output at $2.50/M.
    let cost = t.cost();
    assert!(
        (cost.input.unwrap() - 133.0 / 1e6).abs() < 1e-12,
        "{cost:?}"
    );
    assert!(
        (cost.output.unwrap() - 30.0 * 2.5 / 1e6).abs() < 1e-12,
        "{cost:?}"
    );
}

#[tokio::test]
async fn gemini_2_5_flash_can_transcribe_with_language_hint() {
    let name = format!("{BASIC}_gemini_gemini-2_5-flash_can_transcribe_with_language_hint");
    json_bodied(&name, "gemini", "gemini-2.5-flash", Some("en")).await;
}

#[tokio::test]
async fn openrouter_gpt_4o_mini_transcribe_can_transcribe_audio() {
    let name = format!("{BASIC}_openrouter_openai_gpt-4o-mini-transcribe_can_transcribe_audio");
    let t = json_bodied(&name, "openrouter", "openai/gpt-4o-mini-transcribe", None).await;
    assert_eq!((t.tokens().input, t.tokens().output), (Some(37), Some(12)));
    assert_eq!(t.tokens().reported_cost, Some(0.00010625));
    assert_eq!(
        t.cost().total(),
        Some(0.00010625),
        "OpenRouter's reported cost wins"
    );
}

#[tokio::test]
async fn openrouter_gpt_4o_mini_transcribe_can_transcribe_with_language_hint() {
    let name = format!(
        "{BASIC}_openrouter_openai_gpt-4o-mini-transcribe_can_transcribe_with_language_hint"
    );
    json_bodied(
        &name,
        "openrouter",
        "openai/gpt-4o-mini-transcribe",
        Some("en"),
    )
    .await;
}

// ---- streaming ----------------------------------------------------------------------------

async fn streamed(
    name: &str,
    provider: &str,
    opts: impl FnOnce(Arc<Config>) -> TranscribeOptions<'static>,
) -> (Transcription, Vec<TranscriptionChunk>) {
    let cassette = start(name).await;
    let mut chunks = Vec::new();
    let t = transcribe_stream(
        ruby_wav().as_str(),
        opts(config_for(&cassette, provider)),
        |c| chunks.push(c.clone()),
    )
    .await
    .expect("transcribe_stream");
    cassette.assert_all_matched().await;
    let requests = cassette
        .server
        .received_requests()
        .await
        .unwrap_or_default();
    assert_same_form(
        &support::load(name).unwrap()[0],
        &requests[0],
        &std::fs::read(ruby_wav()).unwrap(),
    );
    (t, chunks)
}

#[tokio::test]
async fn openai_gpt_4o_transcribe_streams_text_deltas_and_returns_the_final_transcription() {
    let model = "gpt-4o-transcribe";
    let name =
        format!("{BASIC}_openai_{model}_streams_text_deltas_and_returns_the_final_transcription");
    let (t, chunks) = streamed(&name, "openai", with(model, "openai")).await;
    assert!(!chunks.is_empty());
    let deltas: String = chunks.iter().filter_map(|c| c.delta.as_deref()).collect();
    assert!(deltas.to_lowercase().contains("ruby"), "{deltas}");
    assert!(chunks.last().unwrap().is_done());
    says_ruby(&t);
    assert_eq!(t.model, model);
    assert_eq!((t.tokens().input, t.tokens().output), (Some(37), Some(12)));
    billed_once_as_transcription(&t, "openai", model);
}

#[tokio::test]
async fn openai_gpt_4o_transcribe_diarize_streams_segments_labelled_with_speakers() {
    let model = "gpt-4o-transcribe-diarize";
    let name = format!("{BASIC}_openai_{model}_streams_segments_labelled_with_speakers");
    let (t, chunks) = streamed(&name, "openai", with(model, "openai")).await;
    let segments: Vec<Value> = chunks
        .iter()
        .filter(|c| c.is_segment())
        .filter_map(|c| c.segment.clone())
        .collect();
    assert!(!segments.is_empty());
    assert!(segments[0].get("speaker").is_some());
    assert!(
        segments[0].get("type").is_none(),
        "the segment is the event without its type"
    );
    says_ruby(&t);
    assert_eq!(t.segments.as_deref(), Some(&segments[..]));
}

#[tokio::test]
async fn streams_mistral_transcriptions_with_speaker_segments_and_usage() {
    let name = format!("{BASIC}_streams_mistral_transcriptions_with_speaker_segments_and_usage");
    let (t, chunks) = streamed(
        &name,
        "mistral",
        with_speakers("voxtral-mini-latest", "mistral"),
    )
    .await;
    assert!(chunks.last().unwrap().is_done());
    assert_eq!(chunks[0].kind, TranscriptionChunk::SEGMENT);
    says_ruby(&t);
    assert!(t.segments.as_ref().unwrap()[0].get("speaker_id").is_some());
    assert!(t.tokens().input.unwrap() > 0);
    assert!(t.duration.unwrap() > 0.0);
}

#[tokio::test]
async fn raises_for_providers_that_do_not_stream_transcriptions() {
    let mut config = Config::default();
    config
        .set("gemini_api_key", "test-key")
        .set("gemini_api_base", "http://127.0.0.1:9");
    let opts = options("gemini-2.5-flash", "gemini", Arc::new(config));
    let error = transcribe_stream(ruby_wav().as_str(), opts, |_| {})
        .await
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("doesn't support streaming transcription"),
        "{error}"
    );
}

// ---- timestamps and Gemini dedicated transcription ----------------------------------------

fn timed_words(t: &Transcription) -> &[Value] {
    let words = t.words.as_deref().expect("words");
    assert!(!words.is_empty());
    assert!(
        words
            .iter()
            .all(|w| w["start"].is_number() && w["end"].is_number()),
        "{words:?}"
    );
    words
}

#[tokio::test]
async fn requests_word_timestamps_through_openai_with_the_shared_keyword() {
    let name = "transcription_requests_word_timestamps_through_openai_with_the_shared_keyword";
    let t = multipart(name, "openai", |config| TranscribeOptions {
        timestamps: Some(vec!["word"]),
        ..options("whisper-1", "openai", config)
    })
    .await;
    says_ruby(&t);
    timed_words(&t);
    billed_once_as_transcription(&t, "openai", "whisper-1");
}

#[tokio::test]
async fn transcribes_audio_with_speaker_labels_and_word_timestamps_through_gemini() {
    let name =
        "transcription_transcribes_audio_with_speaker_labels_and_word_timestamps_through_gemini";
    let cassette = start(name).await;
    let opts = TranscribeOptions {
        language: Some("en-US"),
        timestamps: Some(vec!["word"]),
        speaker_names: Some(vec!["Speaker".into()]),
        ..options(
            "gemini-3.5-transcribe",
            "gemini",
            config_for(&cassette, "gemini"),
        )
    };
    let t = transcribe(ruby_wav().as_str(), opts)
        .await
        .expect("transcribe");
    cassette.assert_all_matched().await;
    let text = t.text.clone().unwrap();
    assert!(
        text.contains("Ruby") && text.contains("developer happiness"),
        "{text}"
    );
    let words = timed_words(&t);
    assert!(words.iter().any(|w| w.get("speaker").is_some()));
    assert_eq!(
        words[7],
        serde_json::json!({ "word": "developer", "speaker": "spk:0", "start": 2.4, "end": 3.0 })
    );
    billed_once_as_transcription(&t, "gemini", "gemini-3.5-transcribe");
}

#[tokio::test]
async fn distinguishes_two_speakers_through_gemini_dedicated_transcription() {
    let name = "transcription_distinguishes_two_speakers_through_gemini_dedicated_transcription";
    let cassette = start(name).await;
    let opts = TranscribeOptions {
        timestamps: Some(vec!["word"]),
        speaker_names: Some(vec![]),
        ..options(
            "gemini-3.5-transcribe",
            "gemini",
            config_for(&cassette, "gemini"),
        )
    };
    let t = transcribe(fixture("google-speakers.wav").as_str(), opts)
        .await
        .expect("transcribe");
    cassette.assert_all_matched().await;
    let text = t.text.clone().unwrap().to_lowercase();
    let workshop = text.find("workshop").expect("workshop");
    assert!(text[workshop..].contains("thank you"), "{text}");
    let mut speakers: Vec<&str> = timed_words(&t)
        .iter()
        .filter_map(|w| w["speaker"].as_str())
        .collect();
    speakers.dedup();
    speakers.sort();
    speakers.dedup();
    assert_eq!(speakers.len(), 2, "{speakers:?}");
}

// ---- `.transcribe` ------------------------------------------------------------------------

#[tokio::test]
async fn validates_model_existence() {
    let opts = TranscribeOptions {
        model: Some("invalid-transcription-model"),
        ..Default::default()
    };
    assert!(matches!(
        transcribe(ruby_wav().as_str(), opts).await,
        Err(Error::ModelNotFound(_))
    ));
}

#[test]
fn defaults_to_gpt_transcribe() {
    assert_eq!(
        Config::default().default_transcription_model,
        "gpt-transcribe"
    );
}

#[tokio::test]
async fn websocket_streaming_is_not_ported_and_says_so() {
    let mut config = Config::default();
    config
        .set("xai_api_key", "test-key")
        .set("xai_api_base", "http://127.0.0.1:9");
    let error = transcribe_stream(
        ruby_wav().as_str(),
        options("grok-stt", "xai", Arc::new(config)),
        |_| {},
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("WebSocket"), "{error}");
}
