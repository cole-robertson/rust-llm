//! RubyLLM 2.0 media specs: `protocols/gemini/speech_spec.rb`, `protocols/gemini/videos_spec.rb`,
//! `protocols/gpustack/tokenization_spec.rb`, `video_extension_spec.rb`, and the context example
//! of `speech_streaming_spec.rb`. Ruby calls the protocol seams on an instance double; these go
//! through the public API against a wiremock server, which runs the same seams, or replay
//! RubyLLM's cassette.

mod support;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use rust_llm::{
    AnimateOptions, Attachment, Config, Context, Error, SpeakOptions, TokenizeOptions, UsageStatus,
    Video, VideoJob, VideoStatus,
};
use serde_json::{Map, Value, json};
use wiremock::matchers::{body_json, header, method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

type Events = Arc<Mutex<Vec<(String, Map<String, Value>)>>>;

/// `CaptureInstrumenter`.
fn capture(config: &mut Config) -> Events {
    let events: Events = Default::default();
    let sink = events.clone();
    config.instrumenter = Some(Arc::new(
        move |name: &str, payload: &Map<String, Value>, _: Option<Duration>| {
            sink.lock()
                .unwrap()
                .push((name.to_string(), payload.clone()));
        },
    ));
    events
}

fn gemini(server: &MockServer) -> Config {
    let mut config = Config::default();
    config.set("gemini_api_key", "test");
    config.set("gemini_api_base", format!("{}/v1beta", server.uri()));
    config.max_retries = 0;
    config.video_generation_poll_interval = Duration::ZERO;
    config
}

fn sent(request: &Request) -> Value {
    serde_json::from_slice(&request.body).expect("json body")
}

async fn requests(server: &MockServer) -> Vec<Request> {
    server.received_requests().await.unwrap_or_default()
}

// ---- protocols/gemini/speech_spec.rb ---------------------------------------------------------

// spec: protocols/gemini/speech_spec.rb:72 raises a clear error for unexpected responses
#[tokio::test]
async fn gemini_speech_raises_a_clear_error_for_unexpected_responses() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(
            "/v1beta/models/gemini-2.5-flash-preview-tts:generateContent",
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({})))
        .mount(&server)
        .await;
    let err = Context::new(gemini(&server))
        .speak(
            "Hello",
            SpeakOptions {
                model: Some("gemini-2.5-flash-preview-tts"),
                provider: Some("gemini"),
                ..Default::default()
            },
        )
        .await
        .unwrap_err();
    assert!(
        matches!(err, Error::Api(ref m, _) if m == "Unexpected response format from Gemini speech generation API"),
        "{err:?}"
    );
}

// ---- protocols/gemini/videos_spec.rb ---------------------------------------------------------

const VEO: &str = "veo-3.1-fast-generate-preview";

fn veo(extra: impl FnOnce(&mut AnimateOptions<'static>)) -> AnimateOptions<'static> {
    let mut options = AnimateOptions {
        model: Some(VEO),
        provider: Some("gemini"),
        ..Default::default()
    };
    extra(&mut options);
    options
}

/// A Veo server that accepts any submission and answers polls of its operation with `status`.
async fn veo_server(status: Value) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(format!("/v1beta/models/{VEO}:predictLongRunning")))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({ "name": format!("models/{VEO}/operations/abc123") })),
        )
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path(format!("/v1beta/models/{VEO}/operations/abc123")))
        .respond_with(ResponseTemplate::new(200).set_body_json(status))
        .mount(&server)
        .await;
    server
}

// spec: protocols/gemini/videos_spec.rb:29 inlines a reference image for image-to-video
#[tokio::test]
async fn gemini_inlines_a_reference_image_for_image_to_video() {
    let server = veo_server(json!({})).await;
    let image_path = format!("{}/tests/fixtures/ruby.png", env!("CARGO_MANIFEST_DIR"));
    Context::new(gemini(&server))
        .animate_later(
            Some("bring the logo to life"),
            veo(|o| o.with = vec![Attachment::new(&image_path)]),
        )
        .await
        .unwrap();

    let payload = sent(&requests(&server).await[0]);
    let image = &payload["instances"][0]["image"]["inlineData"];
    assert_eq!(image["mimeType"], "image/png");
    let data = base64::Engine::decode(
        &base64::engine::general_purpose::STANDARD,
        image["data"].as_str().unwrap(),
    )
    .unwrap();
    assert_eq!(data, std::fs::read(&image_path).unwrap());
}

// spec: protocols/gemini/videos_spec.rb:71 rejects local videos without a generated Veo URI
#[tokio::test]
async fn gemini_rejects_local_videos_without_a_generated_veo_uri() {
    let server = veo_server(json!({})).await;
    let mut source = Video::new(None, Some("video/mp4".into()), Value::Null);
    source.data = Some(b"mp4 bytes".to_vec());
    let err = Context::new(gemini(&server))
        .animate_later(
            Some("Continue the scene"),
            veo(|o| o.extend = Some(source.into())),
        )
        .await
        .unwrap_err();
    assert!(
        matches!(err, Error::Argument(ref m) if m.contains("generated Veo videos")),
        "{err:?}"
    );
    assert!(requests(&server).await.is_empty());
}

// spec: protocols/gemini/videos_spec.rb:77 preserves the original Veo video URI instead of downloading and reuploading it
#[tokio::test]
async fn gemini_preserves_the_original_veo_video_uri() {
    let server = veo_server(json!({})).await;
    let uri = "https://generativelanguage.googleapis.com/v1beta/files/clip:download?alt=media";
    let source = Video::new(
        None,
        None,
        json!({ "response": { "generateVideoResponse": { "generatedSamples": [{ "video": { "uri": uri } }] } } }),
    );
    Context::new(gemini(&server))
        .animate_later(Some("Continue"), veo(|o| o.extend = Some(source.into())))
        .await
        .unwrap();

    // Only the submission: nothing was downloaded or uploaded.
    let requests = requests(&server).await;
    assert_eq!(requests.len(), 1);
    assert_eq!(
        sent(&requests[0])["instances"][0]["video"],
        json!({ "uri": uri })
    );
}

/// Points the job's generated file URI at the replay server, which stands in for the real host.
fn download_from(job: &mut VideoJob, base: &str) {
    let pointer = "/response/generateVideoResponse/generatedSamples/0/video/uri";
    let uri = job.raw.pointer(pointer).and_then(Value::as_str).unwrap();
    let rewritten = uri.replace("https://generativelanguage.googleapis.com", base);
    *job.raw.pointer_mut(pointer).unwrap() = rewritten.into();
}

// spec: protocols/gemini/videos_spec.rb:89 extends a freshly generated Veo video through the public API
#[tokio::test]
async fn gemini_extends_a_freshly_generated_veo_video_through_the_public_api() {
    // The extension request carries the generated file URI, which the recording names on the
    // real host; the download is pointed here, so the recorded body is rewritten to match.
    let cassette = support::Cassette::start_serving(
        "protocols_gemini_videos_extends_a_freshly_generated_veo_video_through_the_public_api",
        &["https://generativelanguage.googleapis.com"],
    )
    .await
    .expect("run bin/convert-cassettes 'protocols_gemini_videos_*'");
    let base = cassette.server.uri();
    let mut config = (*support::config_for(&cassette, "gemini")).clone();
    config.video_generation_poll_interval = Duration::ZERO;
    let context = Context::new(config);

    // `context.animate(...)`, split so the download can be pointed at the replay server.
    let mut original = context
        .animate_later(
            Some("A calm ocean wave at sunset"),
            veo(|o| o.provider_options = json!({ "parameters": { "durationSeconds": 4 } })),
        )
        .await
        .unwrap();
    original.wait(None, None).await.unwrap();
    download_from(&mut original, &base);
    let original = original.video().await.unwrap().unwrap();

    let mut job = context
        .animate_later(
            Some("The wave gently reaches the sandy shore"),
            veo(|o| o.extend = Some(original.into())),
        )
        .await
        .unwrap();
    assert!(
        job.wait(Some(Duration::from_secs(240)), None)
            .await
            .unwrap()
            .is_completed()
    );
    download_from(&mut job, &base);
    let video = job.video().await.unwrap().unwrap();
    assert_eq!(video.mime_type.as_deref(), Some("video/mp4"));
    assert!(video.to_blob().await.unwrap().len() > 1000);
    assert_eq!(video.raw["done"], true);
    cassette.assert_all_matched().await;
}

// spec: protocols/gemini/videos_spec.rb:133 fails with the operation error message
#[tokio::test]
async fn gemini_fails_with_the_operation_error_message() {
    let server = veo_server(
        json!({ "done": true, "error": { "code": 400, "message": "unsupported duration" } }),
    )
    .await;
    let mut job = Context::new(gemini(&server))
        .animate_later(Some("a cat"), veo(|_| {}))
        .await
        .unwrap();
    job.refresh().await.unwrap();
    assert_eq!(job.status, VideoStatus::Failed);
    assert_eq!(job.error.as_deref(), Some("unsupported duration"));
}

// spec: protocols/gemini/videos_spec.rb:189 follows the redirect to the download host with the API key
#[tokio::test]
async fn gemini_follows_the_download_redirect_with_the_api_key() {
    let server = MockServer::start().await;
    let file = format!("{}/v1beta/files/xyz:download?alt=media", server.uri());
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({ "name": "models/veo/operations/abc123" })),
        )
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/v1beta/models/veo/operations/abc123"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "done": true,
            "response": { "generateVideoResponse": { "generatedSamples": [{ "video": { "uri": file } }] } }
        })))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/v1beta/files/xyz:download"))
        .and(header("x-goog-api-key", "test"))
        .respond_with(ResponseTemplate::new(302).insert_header(
            "location",
            format!("{}/download/v1beta/files/xyz:download", server.uri()).as_str(),
        ))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/download/v1beta/files/xyz:download"))
        .and(header("x-goog-api-key", "test"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(b"mp4 bytes".to_vec(), "video/mp4"))
        .expect(1)
        .mount(&server)
        .await;

    let mut job = Context::new(gemini(&server))
        .animate_later(Some("a cat"), veo(|_| {}))
        .await
        .unwrap();
    job.wait(None, None).await.unwrap();
    let video = job.video().await.unwrap().unwrap();
    assert_eq!(video.data.as_deref(), Some(&b"mp4 bytes"[..]));
    server.verify().await;
}

// ---- protocols/gpustack/tokenization_spec.rb -------------------------------------------------

fn gpustack(server: &MockServer) -> Config {
    let mut config = Config::default();
    config.set(
        "gpustack_api_base",
        format!("{}/cluster/model/proxy/42/v1/", server.uri()),
    );
    config.set("gpustack_api_key", "isolated-key");
    config
}

fn qwen3() -> TokenizeOptions<'static> {
    TokenizeOptions {
        model: Some("qwen3"),
        provider: Some("gpustack"),
        ..Default::default()
    }
}

// spec: protocols/gpustack/tokenization_spec.rb:14 uses the model proxy tokenizer and preserves IDs and raw backend metadata without charging usage
#[tokio::test]
async fn gpustack_tokenizes_through_the_model_proxy_without_charging_usage() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/cluster/model/proxy/42/tokenize"))
        .and(header("Authorization", "Bearer isolated-key"))
        .and(body_json(
            json!({ "model": "qwen3", "prompt": "Hello Ruby" }),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(
            json!({ "tokens": [151_643, 123, 456], "count": 3, "max_model_len": 40_960 }),
        ))
        .expect(1)
        .mount(&server)
        .await;
    let mut config = gpustack(&server);
    let events = capture(&mut config);

    let result = Context::new(config)
        .tokenize("Hello Ruby", qwen3())
        .await
        .unwrap();

    assert_eq!(result.ids, vec![151_643, 123, 456]);
    assert_eq!(result.count(), 3);
    assert_eq!(result.model, "qwen3");
    assert_eq!(result.raw["max_model_len"], 40_960);
    // `Accounting::Usage.instrument` never ran: tokenizing is not billable usage.
    assert!(
        !events
            .lock()
            .unwrap()
            .iter()
            .any(|(n, _)| n == "usage.rust_llm")
    );
    server.verify().await;
}

// spec: protocols/gpustack/tokenization_spec.rb:37 keeps tokenization on its dialect when Responses is the configured chat protocol
#[tokio::test]
async fn gpustack_tokenization_ignores_the_configured_responses_protocol() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/cluster/model/proxy/42/tokenize"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(json!({ "tokens": [123], "count": 1 })),
        )
        .mount(&server)
        .await;
    let mut config = gpustack(&server);
    config.set("gpustack_protocol", "responses");

    let result = Context::new(config)
        .tokenize("Ruby", qwen3())
        .await
        .unwrap();
    assert_eq!(result.ids, vec![123]);
}

// ---- video_extension_spec.rb -----------------------------------------------------------------

// spec: video_extension_spec.rb:11 returns a completed video through the blocking public API
#[tokio::test]
async fn returns_a_completed_extension_through_the_blocking_public_api() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/videos/extensions"))
        .and(|req: &Request| {
            sent(req)["video"] == json!({ "url": "data:video/mp4;base64,bXA0IGJ5dGVz" })
        })
        .respond_with(
            ResponseTemplate::new(200).set_body_json(json!({ "request_id": "video_job" })),
        )
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/v1/videos/video_job"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "status": "done",
            "video": { "url": "https://example.com/extended.mp4", "duration": 7 }
        })))
        .mount(&server)
        .await;
    let mut config = Config::default();
    config.set("xai_api_key", "test");
    config.set("xai_api_base", format!("{}/v1", server.uri()));
    config.video_generation_poll_interval = Duration::ZERO;
    let context = Context::new(config);
    let mut source = Video::new(None, Some("video/mp4".into()), Value::Null);
    source.data = Some(b"mp4 bytes".to_vec());

    let result = context
        .animate(
            Some("Continue"),
            AnimateOptions {
                model: Some("grok-imagine-video"),
                provider: Some("xai"),
                extend: Some(source.into()),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    server.verify().await;
    assert_eq!(
        result.url.as_deref(),
        Some("https://example.com/extended.mp4")
    );
    assert_eq!(result.duration, Some(7.0));
    // `result.config` is the context's configuration object itself.
    assert!(Arc::ptr_eq(&result.config(), context.config()));
}

// spec: video_extension_spec.rb:28 rejects extension on unsupported providers before issuing a request
#[tokio::test]
async fn rejects_extension_on_unsupported_providers_before_a_request() {
    let server = MockServer::start().await;
    let mut config = Config::default();
    config.set("anthropic_api_key", "test");
    config.set("anthropic_api_base", server.uri());
    let err = Context::new(config)
        .animate_later(
            Some("Continue"),
            AnimateOptions {
                model: Some("claude-haiku-4-5"),
                provider: Some("anthropic"),
                extend: Some("https://example.com/clip.mp4".into()),
                ..Default::default()
            },
        )
        .await
        .unwrap_err();
    assert!(
        matches!(err, Error::Api(ref m, _) if m.contains("Anthropic doesn't support video extension")),
        "{err:?}"
    );
    assert!(requests(&server).await.is_empty());
}

// ---- speech_streaming_spec.rb ----------------------------------------------------------------

// spec: speech_streaming_spec.rb:134 forwards the block through a context and reports the complete result in instrumentation
#[tokio::test]
async fn forwards_the_speech_block_through_a_context_and_instruments_the_result() {
    // Ruby streams from Deepgram, which RustLLM does not port; OpenAI streams through the same
    // binary path (`BinaryStreaming#stream_binary`).
    let audio = b"ID3\xFF\x00audio".to_vec();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/speech"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(audio.clone(), "audio/mpeg"))
        .mount(&server)
        .await;
    let mut config = Config::default();
    config.set("openai_api_key", "test");
    config.set("openai_api_base", format!("{}/v1", server.uri()));
    let events = capture(&mut config);
    let context = Context::new(config);
    let mut chunks = Vec::new();

    let speech = context
        .speak_stream(
            "Hello Ruby.",
            SpeakOptions {
                model: Some("gpt-4o-mini-tts"),
                provider: Some("openai"),
                ..Default::default()
            },
            |chunk| chunks.extend_from_slice(&chunk.data),
        )
        .await
        .unwrap();

    assert_eq!(chunks, speech.data);
    let event = events
        .lock()
        .unwrap()
        .iter()
        .find(|(n, _)| n == "speech.rust_llm")
        .map(|(_, p)| p.clone())
        .expect("a speech.rust_llm event");
    assert_eq!(event["streaming"], true);
    assert_eq!(event["result"]["model"], speech.model.as_str());
    assert_eq!(event["audio_bytes"], audio.len());
    let statuses: Vec<UsageStatus> = speech.usage_entries.iter().map(|e| e.status).collect();
    assert_eq!(statuses, [UsageStatus::Succeeded]);
}
