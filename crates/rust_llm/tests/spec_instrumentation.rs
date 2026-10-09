//! RubyLLM 2.0's `support/instrumentation_spec.rb` "one-shot provider options" examples and the
//! speech event, plus the non-chat events RubyLLM fires (`moderation`, `ocr`, `rerank`,
//! `transcription`, `tokenization`, `video_job`, `video`) with their `usage.rust_llm` attempts and
//! per-call `metadata:`. Ruby stubs `provider.embed` etc.; here a wiremock server answers in the
//! provider's wire format, so the port's render and parse run in between.
//!
//! Also the leftover single examples of `tool_spec.rb`, `chat_options_spec.rb`, and
//! `chat_cache_until_here_spec.rb` in this lane.

mod support;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use rust_llm::{
    Attachment, Config, EmbedOptions, ModerateOptions, OcrOptions, PaintOptions, RerankOptions,
    SpeakOptions, ToolResult, TranscribeOptions,
};
use serde_json::{Map, Value, json};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

type Events = Arc<Mutex<Vec<(String, Map<String, Value>)>>>;

/// `CaptureInstrumenter` in a context whose OpenAI-compatible providers point at `server`.
fn api_context(server: &MockServer) -> (Arc<Config>, Events) {
    let events: Events = Default::default();
    let sink = events.clone();
    let mut config = Config::default();
    for (provider, prefix) in [
        ("openai", "/v1"),
        ("mistral", "/v1"),
        ("openrouter", "/api/v1"),
    ] {
        config
            .set(
                format!("{provider}_api_base"),
                format!("{}{prefix}", server.uri()),
            )
            .set(format!("{provider}_api_key"), "test");
    }
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

fn last(events: &Events) -> (String, Map<String, Value>) {
    events.lock().unwrap().last().cloned().expect("an event")
}

fn usage_events(events: &Events) -> Vec<Map<String, Value>> {
    events
        .lock()
        .unwrap()
        .iter()
        .filter(|(n, _)| n == "usage.rust_llm")
        .map(|(_, p)| p.clone())
        .collect()
}

async fn respond(server: &MockServer, route: &str, template: ResponseTemplate) {
    Mock::given(method("POST"))
        .and(path(route))
        .respond_with(template)
        .mount(server)
        .await;
}

async fn body(server: &MockServer) -> Value {
    let requests: Vec<Request> = server.received_requests().await.unwrap();
    serde_json::from_slice(&requests.last().expect("a request").body).unwrap()
}

fn provider_options() -> Value {
    json!({ "custom": "value" })
}

fn metadata() -> Option<Value> {
    Some(json!({ "academy_id": 42, "feature": "search" }))
}

// spec: support/instrumentation_spec.rb:266 one-shot provider options > forwards embedding provider options and includes metadata in the event payload
#[tokio::test]
async fn embedding_forwards_provider_options_and_reports_metadata() {
    let server = MockServer::start().await;
    respond(
        &server,
        "/v1/embeddings",
        ResponseTemplate::new(200).set_body_json(json!({
            "model": "text-embedding-3-small",
            "data": [{ "embedding": [0.1, 0.2] }],
            "usage": { "prompt_tokens": 3 }
        })),
    )
    .await;
    let (config, events) = api_context(&server);
    let embedding = rust_llm::embed(
        "hello",
        EmbedOptions {
            model: Some("text-embedding-3-small"),
            provider_options: provider_options(),
            metadata: metadata(),
            config: Some(config),
            ..Default::default()
        },
    )
    .await
    .unwrap();

    assert_eq!(body(&server).await["custom"], "value");
    let (name, payload) = last(&events);
    assert_eq!(name, "embedding.rust_llm");
    assert_eq!(payload["provider_options"], provider_options());
    assert_eq!(payload["metadata"], metadata().unwrap());
    // `track_usage(:embedding)`: the attempt is on the result and reported as it finishes.
    assert_eq!(embedding.usage_entries.len(), 1);
    let usage = usage_events(&events);
    assert_eq!(usage.len(), 1);
    assert_eq!(usage[0]["operation"], "embedding");
    assert_eq!(usage[0]["status"], "succeeded");
    assert_eq!(usage[0]["tokens"], json!({ "input_tokens": 3 }));
    assert_eq!(payload["tokens"], json!({ "input_tokens": 3 }));
}

// spec: support/instrumentation_spec.rb:281 one-shot provider options > forwards image provider options and includes metadata in the event payload
#[tokio::test]
async fn image_forwards_provider_options_and_reports_metadata() {
    let server = MockServer::start().await;
    respond(
        &server,
        "/v1/images/generations",
        ResponseTemplate::new(200).set_body_json(json!({
            "data": [{ "url": "https://example.com/image.png" }]
        })),
    )
    .await;
    let (config, events) = api_context(&server);
    rust_llm::paint(
        "draw this",
        PaintOptions {
            model: Some("gpt-image-1"),
            provider_options: provider_options(),
            metadata: metadata(),
            config: Some(config),
            ..Default::default()
        },
    )
    .await
    .unwrap();

    assert_eq!(body(&server).await["custom"], "value");
    let (name, payload) = last(&events);
    assert_eq!(name, "image.rust_llm");
    assert_eq!(payload["prompt"], "draw this");
    assert_eq!(payload["provider_options"], provider_options());
    assert_eq!(payload["metadata"], metadata().unwrap());
    let usage = usage_events(&events);
    assert_eq!(usage.len(), 1);
    assert_eq!(usage[0]["operation"], "image");
}

fn moderation_response() -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(json!({
        "id": "mod_123",
        "model": "omni-moderation-latest",
        "results": [{ "flagged": false, "categories": {}, "category_scores": {} }]
    }))
}

// spec: support/instrumentation_spec.rb:303 one-shot provider options > forwards moderation provider options and includes metadata in the event payload
#[tokio::test]
async fn moderation_forwards_provider_options_and_reports_metadata() {
    let server = MockServer::start().await;
    respond(&server, "/v1/moderations", moderation_response()).await;
    let (config, events) = api_context(&server);
    let moderation = rust_llm::moderate(
        "check this",
        ModerateOptions {
            model: Some("omni-moderation-latest"),
            provider_options: provider_options(),
            metadata: metadata(),
            config: Some(config),
            ..Default::default()
        },
    )
    .await
    .unwrap();

    let sent = body(&server).await;
    assert_eq!(sent["input"], "check this");
    assert_eq!(sent["custom"], "value");
    assert_eq!(moderation.id.as_deref(), Some("mod_123"));
    let (name, payload) = last(&events);
    assert_eq!(name, "moderation.rust_llm");
    assert_eq!(payload["provider"], "openai");
    assert_eq!(payload["provider_class"], "OpenAI");
    assert_eq!(payload["model"], "omni-moderation-latest");
    assert_eq!(payload["input"], "check this");
    assert_eq!(payload["attachment_count"], 0);
    assert_eq!(payload["provider_options"], provider_options());
    assert_eq!(payload["metadata"], metadata().unwrap());
    assert_eq!(payload["flagged"], false);
    assert!(!payload.contains_key("operation"));
    let usage = usage_events(&events);
    assert_eq!(usage.len(), 1);
    assert_eq!(usage[0]["operation"], "moderation");
    assert_eq!(usage[0]["status"], "succeeded");
}

// spec: support/instrumentation_spec.rb:318 one-shot provider options > forwards moderation attachments
#[tokio::test]
async fn moderation_forwards_attachments() {
    let server = MockServer::start().await;
    respond(&server, "/v1/moderations", moderation_response()).await;
    let (config, events) = api_context(&server);
    rust_llm::moderate(
        "check this",
        ModerateOptions {
            model: Some("omni-moderation-latest"),
            with: vec![
                Attachment::new("https://example.com/safe.png"),
                Attachment::new("https://example.com/also-safe.png"),
            ],
            config: Some(config),
            ..Default::default()
        },
    )
    .await
    .unwrap();

    let input = body(&server).await["input"].clone();
    assert_eq!(
        input,
        json!([
            { "type": "text", "text": "check this" },
            { "type": "image_url", "image_url": { "url": "https://example.com/safe.png" } },
            { "type": "image_url", "image_url": { "url": "https://example.com/also-safe.png" } }
        ])
    );
    let (_, payload) = last(&events);
    assert_eq!(payload["attachment_count"], 2);
    assert_eq!(payload["metadata"], Value::Null);
}

// spec: support/instrumentation_spec.rb:221 emits speech events with output metadata
#[tokio::test]
async fn speech_events_carry_output_metadata() {
    let server = MockServer::start().await;
    respond(
        &server,
        "/v1/audio/speech",
        ResponseTemplate::new(200).set_body_bytes(b"audio bytes".to_vec()),
    )
    .await;
    let (config, events) = api_context(&server);
    let speech = rust_llm::speak(
        "hello",
        SpeakOptions {
            model: Some("gpt-4o-mini-tts"),
            config: Some(config),
            ..Default::default()
        },
    )
    .await
    .unwrap();

    let (name, payload) = last(&events);
    assert_eq!(name, "speech.rust_llm");
    assert_eq!(payload["provider"], "openai");
    assert_eq!(payload["provider_class"], "OpenAI");
    assert_eq!(payload["model"], "gpt-4o-mini-tts");
    assert_eq!(payload["input"], "hello");
    assert_eq!(payload["response_model"], json!(speech.model));
    assert_eq!(payload["voice"], "alloy");
    assert_eq!(payload["format"], "mp3");
    assert_eq!(payload["audio_bytes"], 11);
    assert!(payload["tokens"].is_object() && payload["cost"].is_object());
    assert!(!payload.contains_key("operation"));
    assert_eq!(usage_events(&events)[0]["operation"], "speech");
}

// spec: support/instrumentation_spec.rb:337 one-shot provider options > forwards speech provider options and includes metadata in the event payload
#[tokio::test]
async fn speech_forwards_provider_options_and_reports_metadata() {
    let server = MockServer::start().await;
    respond(
        &server,
        "/v1/audio/speech",
        ResponseTemplate::new(200).set_body_bytes(b"audio bytes".to_vec()),
    )
    .await;
    let (config, events) = api_context(&server);
    rust_llm::speak(
        "say this",
        SpeakOptions {
            model: Some("gpt-4o-mini-tts"),
            provider_options: provider_options(),
            metadata: metadata(),
            config: Some(config),
            ..Default::default()
        },
    )
    .await
    .unwrap();

    let sent = body(&server).await;
    assert_eq!(sent["input"], "say this");
    assert_eq!(sent["custom"], "value");
    let (_, payload) = last(&events);
    assert_eq!(payload["provider_options"], provider_options());
    assert_eq!(payload["metadata"], metadata().unwrap());
}

// spec: support/instrumentation_spec.rb:357 one-shot provider options > forwards transcription provider options and includes metadata in the event payload
#[tokio::test]
async fn transcription_forwards_provider_options_and_reports_metadata() {
    let server = MockServer::start().await;
    respond(
        &server,
        "/v1/audio/transcriptions",
        ResponseTemplate::new(200).set_body_json(json!({ "text": "hello" })),
    )
    .await;
    let (config, events) = api_context(&server);
    let audio = Attachment::from_bytes(b"RIFF....".to_vec(), "audio.wav", Some("audio/wav"));
    let transcription = rust_llm::transcribe(
        audio,
        TranscribeOptions {
            model: Some("whisper-1"),
            timestamps: Some(vec!["word"]),
            provider_options: provider_options(),
            metadata: metadata(),
            config: Some(config),
            ..Default::default()
        },
    )
    .await
    .unwrap();

    assert_eq!(transcription.text.as_deref(), Some("hello"));
    // `timestamps: :word` and the provider options reach the multipart form.
    let requests: Vec<Request> = server.received_requests().await.unwrap();
    let form = String::from_utf8_lossy(&requests[0].body).to_string();
    for field in ["custom", "timestamp_granularities", "verbose_json"] {
        assert!(form.contains(field), "{field} missing from {form}");
    }
    let (name, payload) = last(&events);
    assert_eq!(name, "transcription.rust_llm");
    assert_eq!(payload["provider_options"], provider_options());
    assert_eq!(payload["metadata"], metadata().unwrap());
    assert_eq!(payload["response_model"], "whisper-1");
    assert_eq!(usage_events(&events)[0]["operation"], "transcription");
}

/// `ocr.rust_llm` with `pages`, per-call metadata, and one usage attempt.
#[tokio::test]
async fn ocr_events_carry_pages_and_metadata() {
    let server = MockServer::start().await;
    respond(
        &server,
        "/v1/ocr",
        ResponseTemplate::new(200).set_body_json(json!({
            "model": "mistral-ocr-latest",
            "pages": [{ "index": 0, "markdown": "Hello" }],
            "usage_info": { "pages_processed": 1 }
        })),
    )
    .await;
    let (config, events) = api_context(&server);
    rust_llm::ocr(
        Attachment::new("https://example.com/doc.pdf"),
        OcrOptions {
            model: Some("mistral-ocr-latest"),
            pages: Some(vec![0]),
            metadata: metadata(),
            config: Some(config),
            ..Default::default()
        },
    )
    .await
    .unwrap();

    let (name, payload) = last(&events);
    assert_eq!(name, "ocr.rust_llm");
    assert_eq!(payload["provider"], "mistral");
    assert_eq!(payload["pages"], json!([0]));
    assert_eq!(payload["metadata"], metadata().unwrap());
    assert_eq!(payload["response_model"], "mistral-ocr-latest");
    assert_eq!(usage_events(&events)[0]["operation"], "ocr");
}

/// `rerank.rust_llm` with the query, document count, `top_n`, metadata, and usage.
#[tokio::test]
async fn rerank_events_carry_the_query_and_metadata() {
    let server = MockServer::start().await;
    respond(
        &server,
        "/api/v1/rerank",
        ResponseTemplate::new(200).set_body_json(json!({
            "model": "cohere/rerank-v3.5",
            "results": [{ "index": 1, "relevance_score": 0.9 }],
            "usage": { "total_tokens": 7 }
        })),
    )
    .await;
    let (config, events) = api_context(&server);
    rust_llm::rerank(
        "ruby",
        &["python", "ruby"],
        "cohere/rerank-v3.5",
        RerankOptions {
            provider: Some("openrouter"),
            assume_model_exists: true,
            top_n: Some(1),
            metadata: metadata(),
            config: Some(config),
            ..Default::default()
        },
    )
    .await
    .unwrap();

    let (name, payload) = last(&events);
    assert_eq!(name, "rerank.rust_llm");
    assert_eq!(payload["query"], "ruby");
    assert_eq!(payload["document_count"], 2);
    assert_eq!(payload["top_n"], 1);
    assert_eq!(payload["metadata"], metadata().unwrap());
    assert_eq!(payload["tokens"]["input_tokens"], 7);
    let usage = usage_events(&events);
    assert_eq!(usage.len(), 1);
    assert_eq!(usage[0]["operation"], "rerank");
}

/// A failed attempt before the success: one `usage.rust_llm` per attempt, as for chats.
#[tokio::test]
async fn one_shot_operations_report_every_attempt() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/moderations"))
        .respond_with(
            ResponseTemplate::new(500).set_body_json(json!({ "error": { "message": "retry" } })),
        )
        .up_to_n_times(1)
        .mount(&server)
        .await;
    respond(&server, "/v1/moderations", moderation_response()).await;
    let (config, events) = api_context(&server);
    let mut retrying = (*config).clone();
    retrying.max_retries = 1;
    retrying.retry_interval = 0.0;
    let moderation = rust_llm::moderate(
        "check this",
        ModerateOptions {
            model: Some("omni-moderation-latest"),
            config: Some(Arc::new(retrying)),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(moderation.usage_entries.len(), 2);
    let statuses: Vec<Value> = usage_events(&events)
        .iter()
        .map(|p| p["status"].clone())
        .collect();
    assert_eq!(statuses, [json!("failed"), json!("succeeded")]);
}

/// A failed operation still ends its event, with the exception, and reports no result.
#[tokio::test]
async fn failed_operations_record_the_exception() {
    let server = MockServer::start().await;
    respond(
        &server,
        "/v1/moderations",
        ResponseTemplate::new(400).set_body_json(json!({ "error": { "message": "bad input" } })),
    )
    .await;
    let (config, events) = api_context(&server);
    let result = rust_llm::moderate(
        "check this",
        ModerateOptions {
            model: Some("omni-moderation-latest"),
            config: Some(config),
            ..Default::default()
        },
    )
    .await;
    assert!(result.is_err());
    let (name, payload) = last(&events);
    assert_eq!(name, "moderation.rust_llm");
    assert!(
        payload["exception"][1]
            .as_str()
            .unwrap()
            .contains("bad input")
    );
    assert!(!payload.contains_key("result"));
}

/// Adds `CaptureInstrumenter` to a replay configuration.
fn capturing(config: &Config) -> (Arc<Config>, Events) {
    let events: Events = Default::default();
    let sink = events.clone();
    let mut config = config.clone();
    config.instrumenter = Some(Arc::new(
        move |name: &str, payload: &Map<String, Value>, _: Option<Duration>| {
            sink.lock()
                .unwrap()
                .push((name.to_string(), payload.clone()));
        },
    ));
    (Arc::new(config), events)
}

/// `tokenization.rust_llm`, replayed from RubyLLM's xAI tokenization cassette.
#[tokio::test]
async fn tokenization_events_carry_the_model_and_result() {
    let cassette =
        support::Cassette::start("tokenization_tokenizes_text_with_xai_through_the_public_api")
            .await
            .expect("cassette");
    let (config, events) = capturing(&support::config_for(&cassette, "xai"));
    let result = rust_llm::tokenize(
        "Ruby makes AI useful.",
        rust_llm::TokenizeOptions {
            model: Some("grok-4.3"),
            provider: Some("xai"),
            config: Some(config),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    cassette.assert_all_matched().await;
    let (name, payload) = last(&events);
    assert_eq!(name, "tokenization.rust_llm");
    assert_eq!(payload["model"], "grok-4.3");
    assert_eq!(payload["provider"], "xai");
    assert_eq!(payload["result"]["ids"], json!(result.ids));
    // Tokenizing is not generation: no usage.
    assert!(usage_events(&events).is_empty());
}

/// `video.rust_llm` around `video_job.rust_llm`, replayed from RubyLLM's xAI video cassette.
#[tokio::test]
async fn video_events_wrap_the_job_event() {
    let cassette = support::Cassette::start(
        "video_basic_functionality_xai_grok-imagine-video_can_animate_videos",
    )
    .await
    .expect("cassette");
    let mut config = (*support::config_for(&cassette, "xai")).clone();
    config.video_generation_poll_interval = Duration::ZERO;
    let (config, events) = capturing(&config);
    let video = rust_llm::animate(
        Some("a calm ocean wave at sunset"),
        rust_llm::AnimateOptions {
            model: Some("grok-imagine-video"),
            provider: Some("xai"),
            provider_options: json!({ "duration": 1, "resolution": "480p" }),
            metadata: metadata(),
            config: Some(config),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    // The cassette's last request is the clip download, which `animate` does not make.

    let names: Vec<String> = events
        .lock()
        .unwrap()
        .iter()
        .map(|(n, _)| n.clone())
        .filter(|n| n.starts_with("video"))
        .collect();
    assert_eq!(names, ["video_job.rust_llm", "video.rust_llm"]);
    let job = events
        .lock()
        .unwrap()
        .iter()
        .find(|(n, _)| n == "video_job.rust_llm")
        .map(|(_, p)| p.clone())
        .unwrap();
    assert_eq!(job["provider"], "xai");
    assert_eq!(job["model"], "grok-imagine-video");
    assert_eq!(job["metadata"], metadata().unwrap());
    let (_, payload) = last(&events);
    assert_eq!(payload["model"], "grok-imagine-video");
    assert_eq!(payload["prompt"], "a calm ocean wave at sunset");
    assert_eq!(payload["metadata"], metadata().unwrap());
    assert_eq!(payload["job_id"], job["job_id"]);
    assert_eq!(payload["response_model"], json!(video.model));
}

// ---- tool_spec.rb ------------------------------------------------------------------------------

// spec: tool_spec.rb:345 .split_result > returns a lone attachment with empty text
#[test]
fn a_lone_attachment_result_has_empty_text() {
    let attachment = Attachment::from_bytes(b"bytes".to_vec(), "a.txt", Some("text/plain"));
    let result = ToolResult::from(attachment.clone());
    assert_eq!(result.content, "");
    assert_eq!(result.attachments, vec![attachment]);
}

/// `GuardedTool`: `execute(query:, tool_call: nil)`; the `ToolCall` comes from RubyLLM, never the
/// model.
struct Guarded;

#[async_trait::async_trait]
impl rust_llm::Tool for Guarded {
    fn description(&self) -> String {
        "Guarded".into()
    }
    fn parameters(&self) -> Vec<rust_llm::Parameter> {
        vec![rust_llm::Parameter::new("query")]
    }
    async fn execute(
        &self,
        args: Map<String, Value>,
        call: &rust_llm::ToolCall,
    ) -> Result<ToolResult, rust_llm::ToolError> {
        Ok(json!([args["query"], call.id]).into())
    }
}

// spec: tool_spec.rb:225 #call > with a tool_call keyword > rejects tool_call as a model-provided argument
#[tokio::test]
async fn a_model_provided_tool_call_argument_is_rejected() {
    let server = MockServer::start().await;
    let mut config = Config::default();
    config
        .set("anthropic_api_key", "test")
        .set("anthropic_api_base", server.uri());
    let mut chat = rust_llm::Chat::with_config(
        Arc::new(config),
        Some("claude-haiku-4-5"),
        Some("anthropic"),
        false,
    )
    .unwrap()
    .with_tool(Guarded);
    chat.ask_later("go").unwrap();
    let mut m = rust_llm::Message::new(rust_llm::Role::Assistant, Some(String::new()));
    let arguments = json!({ "query": "ruby", "tool_call": "spoofed" });
    m.tool_calls = Some(
        [(
            "call_1".to_string(),
            rust_llm::ToolCall::new("call_1", "guarded", arguments.as_object().unwrap().clone()),
        )]
        .into_iter()
        .collect(),
    );
    chat.add_message(m);
    chat.run_tools().await.unwrap();
    assert_eq!(
        chat.messages().last().unwrap().content(),
        json!({ "error": "Invalid tool arguments: unknown keyword: tool_call" }).to_string()
    );
}

// ---- chat_options_spec.rb ------------------------------------------------------------------------

fn openai_chat() -> rust_llm::Chat {
    let mut config = Config::default();
    config.set("openai_api_key", "test");
    rust_llm::Chat::with_config(
        Arc::new(config),
        Some("gpt-4.1-nano"),
        Some("openai"),
        false,
    )
    .unwrap()
}

// spec: chat_options_spec.rb:195 #thinking > returns nil when thinking is not configured
#[test]
fn thinking_is_none_when_not_configured() {
    assert!(openai_chat().thinking().is_none());
}

// spec: chat_options_spec.rb:199 #thinking > returns the options configured for the current model
#[test]
fn thinking_returns_the_configured_options() {
    let mut options = rust_llm::ThinkingConfig::effort("high");
    options.budget = Some(2_000);
    let chat = openai_chat().with_thinking(options);
    let thinking = chat.thinking().expect("thinking");
    assert_eq!(thinking.effort.as_deref(), Some("high"));
    assert_eq!(thinking.budget, Some(2_000));
    assert_eq!(thinking.display, None);
    assert_eq!(thinking.enabled, None);
}

// ---- chat_cache_until_here_spec.rb ---------------------------------------------------------------

// spec: chat_cache_until_here_spec.rb:158 Gemini explicit caching > accepts a CachedContent as the id
#[test]
fn a_cached_content_is_accepted_as_the_cache_id() {
    let cache = rust_llm::CachedContent::new("cachedContents/abc123");
    let mut config = Config::default();
    config.set("gemini_api_key", "test");
    let mut chat = rust_llm::Chat::with_config(
        Arc::new(config),
        Some("gemini-2.5-flash"),
        Some("gemini"),
        false,
    )
    .unwrap()
    .with_caching(json!({ "id": cache.as_ref() }))
    .unwrap();
    chat.ask_later("Hello").unwrap();
    assert_eq!(
        chat.render().unwrap()["cachedContent"],
        "cachedContents/abc123"
    );
}
