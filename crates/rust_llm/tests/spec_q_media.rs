//! RubyLLM 2.1 parity (lane Q, media): Mistral hosted images and dialect fixes, video job cost and
//! request settings, and the tokens of blocked Gemini image and transcription attempts.
//! Stubbed examples run the real paint/animate/transcribe paths against a wiremock server.

mod support;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use rust_llm::message::{Operation, Thinking, UsageEntry, UsageStatus};
use rust_llm::protocols::mistral;
use rust_llm::{
    AnimateOptions, Chat, Config, Context, Error, Message, PaintOptions, Role, ToolCall,
    TranscribeOptions, VideoStatus,
};
use serde_json::{Map, Value, json};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

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

fn usages(events: &Events) -> Vec<Map<String, Value>> {
    events
        .lock()
        .unwrap()
        .iter()
        .filter(|(name, _)| name == "usage.rust_llm")
        .map(|(_, payload)| payload.clone())
        .collect()
}

fn config_for(server: &MockServer, provider: &str, prefix: &str) -> Config {
    let mut config = Config::default();
    config.set(format!("{provider}_api_key"), "test");
    config.set(
        format!("{provider}_api_base"),
        format!("{}{prefix}", server.uri()),
    );
    config.max_retries = 0;
    config.video_generation_poll_interval = Duration::ZERO;
    config
}

async fn json_server(route: &str, body: Value) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(route))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&server)
        .await;
    server
}

fn fixture(name: &str) -> String {
    format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
}

// ---- protocols/mistral/conversations/images_spec.rb -------------------------------------------

const MISTRAL: &str = "mistral-small-latest";
const JPEG: &[u8] = b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00";

// spec: protocols/mistral/conversations/images_spec.rb:34 downloads generated image URLs from hosted tool results
#[tokio::test]
async fn downloads_generated_image_urls_from_hosted_tool_results() {
    let server = MockServer::start().await;
    let image_url = format!("{}/generated.jpg", server.uri());
    let data = json!({ "outputs": [
        { "type": "tool.execution", "name": "image_generation", "function": "generate_image",
          "info": { "result": json!({ "url": image_url }).to_string() } },
        { "type": "message.output", "content": "Here is your image." }
    ] });
    Mock::given(method("POST"))
        .and(path("/v1/conversations"))
        .respond_with(ResponseTemplate::new(200).set_body_json(data))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/generated.jpg"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(JPEG.to_vec(), "image/jpeg"))
        .mount(&server)
        .await;
    let image = rust_llm::paint(
        "Red circle",
        PaintOptions {
            model: Some(MISTRAL),
            provider: Some("mistral"),
            config: Some(Arc::new(config_for(&server, "mistral", "/v1"))),
            ..Default::default()
        },
    )
    .await
    .unwrap()
    .into_image();
    assert_eq!(image.to_blob().await.unwrap(), JPEG);
    assert_eq!(image.mime_type.as_deref(), Some("image/jpeg"));
}

// ---- protocols/mistral/multi_completion_spec.rb -----------------------------------------------

fn image_call() -> Value {
    json!({ "id": "imagecall", "type": "function", "function": { "name": "generate_image", "arguments": "{}" },
            "metadata": { "tool_type": "image" } })
}

fn multi_messages(call: Value, last_content: Value) -> Vec<Value> {
    vec![
        json!({ "role": "assistant", "content": "", "tool_calls": [call], "index": 0 }),
        json!({ "role": "tool", "tool_call_id": "imagecall", "content": "{\"url\":\"https://example.com/image.jpg\"}", "index": 1 }),
        json!({ "role": "assistant", "content": last_content, "index": 2 }),
    ]
}

fn multi_body(messages: &[Value]) -> Value {
    json!({ "model": MISTRAL, "choices": [{ "messages": messages, "finish_reason": "stop" }], "usage": {} })
}

// spec: protocols/mistral/multi_completion_spec.rb:49 returns generated images when the assistant only links to the hosted tool result
#[test]
fn returns_generated_images_when_the_assistant_only_links_to_the_hosted_tool_result() {
    let messages = multi_messages(
        image_call(),
        json!("Your image: https://example.com/image.jpg"),
    );
    let message = mistral::parse_multi_message(&multi_body(&messages), None)
        .unwrap()
        .unwrap();
    let sources: Vec<Option<&str>> = message.attachments.iter().map(|a| a.url()).collect();
    assert_eq!(sources, [Some("https://example.com/image.jpg")]);
    assert_eq!(
        message.content(),
        "Your image: https://example.com/image.jpg"
    );
}

/// The tool result and the assistant's `image_url` part name the same image: one attachment.
// spec: protocols/mistral/multi_completion_spec.rb:30 parses completed hosted tools and images without scheduling a local tool
#[test]
fn keeps_one_attachment_per_hosted_image_source() {
    let messages = multi_messages(
        image_call(),
        json!([{ "type": "text", "text": "Your image." },
               { "type": "image_url", "image_url": "https://example.com/image.jpg" }]),
    );
    let message = mistral::parse_multi_message(&multi_body(&messages), None)
        .unwrap()
        .unwrap();
    assert_eq!(message.attachments.len(), 1);
}

// spec: protocols/mistral/multi_completion_spec.rb:58 does not treat URLs from local tools as generated images
#[test]
fn does_not_treat_urls_from_local_tools_as_generated_images() {
    let mut call = image_call();
    call.as_object_mut().unwrap().remove("metadata");
    let messages = multi_messages(call, json!("Here is a link."));
    // Without hosted metadata the call has a result, so it is a completed step, not pending.
    let message = mistral::parse_multi_message(&multi_body(&messages), None)
        .unwrap()
        .unwrap();
    assert!(message.attachments.is_empty());
}

// ---- Mistral Conversations connector usage (a5fe12ca) -----------------------------------------

// spec: protocols/mistral/conversations_spec.rb:85 parses hosted steps, citation markers, and connector input tokens without inflating output
#[test]
fn names_mistral_connector_counts_like_other_providers() {
    let data = json!({ "outputs": [{ "type": "message.output", "content": "Ruby docs" }],
        "usage": { "prompt_tokens": 20, "completion_tokens": 5, "connector_tokens": 100,
                   "connectors": { "web_search": 1, "code_interpreter": 2, "document_library": 1, "image_generation": 0 } } });
    let message = mistral::parse_completion_body(MISTRAL, &data, None).unwrap();
    assert_eq!(
        message.tokens.server_tool_use,
        json!({ "web_search_requests": 1, "code_execution_requests": 2, "file_search_requests": 1 })
            .as_object()
            .cloned()
    );
}

// spec: protocols/mistral/conversations_spec.rb:141 accumulates tool arguments, text, and final usage from Conversations events
#[test]
fn streamed_conversation_usage_names_code_execution_requests() {
    let mut state = mistral::ConversationStream::default();
    let done = json!({ "type": "conversation.response.done",
        "usage": { "prompt_tokens": 10, "completion_tokens": 2, "connector_tokens": 1,
                   "connectors": { "code_interpreter": 1 } } });
    let chunk = mistral::build_conversation_chunk(MISTRAL, &mut state, &done).unwrap();
    assert_eq!(
        (chunk.tokens.input, chunk.tokens.output),
        (Some(11), Some(2))
    );
    assert_eq!(
        chunk.tokens.server_tool_use,
        json!({ "code_execution_requests": 1 }).as_object().cloned()
    );
}

// ---- providers/mistral/chat_spec.rb -----------------------------------------------------------

fn mistral_render(message: Message) -> Value {
    let mut c = Config::default();
    c.set("mistral_api_base", "http://127.0.0.1:9");
    c.set("mistral_api_key", "test");
    let mut chat = Chat::with_config(
        Arc::new(c),
        Some("magistral-small-latest"),
        Some("mistral"),
        true,
    )
    .unwrap();
    chat.add_message(message);
    chat.render().unwrap()
}

// spec: providers/mistral/chat_spec.rb:156 #format_tool_calls leaves out the thought signature Gemini puts on a call
#[test]
fn mistral_leaves_out_the_thought_signature_gemini_puts_on_a_call() {
    let mut arguments = Map::new();
    arguments.insert("city".into(), "Paris".into());
    let mut call = ToolCall::new("call_1", "weather", arguments);
    call.thought_signature = Some("gemini-signature".into());
    let mut message = Message::new(Role::Assistant, None);
    message.tool_calls = Some([("call_1".to_string(), call)].into_iter().collect());
    let payload = mistral_render(message);
    assert_eq!(
        payload["messages"][0]["tool_calls"],
        json!([{ "id": "call_1", "type": "function", "function": { "name": "weather", "arguments": "{\"city\":\"Paris\"}" } }])
    );
}

// spec: providers/mistral/chat_spec.rb:217 #build_thinking_blocks sends no signature Mistral did not produce
#[test]
fn mistral_sends_no_signature_it_did_not_produce() {
    let answer = |text: Option<&str>, signature: &str| {
        let mut m = Message::new(Role::Assistant, Some("Done".into()));
        m.thinking = Some(Thinking {
            text: text.map(str::to_string),
            signature: Some(signature.into()),
        });
        m
    };
    let content =
        mistral_render(answer(Some("why"), "anthropic-signature"))["messages"][0]["content"]
            .clone();
    assert_eq!(
        content[0],
        json!({ "type": "thinking", "thinking": [{ "type": "text", "text": "why" }] })
    );
    let content = mistral_render(answer(None, "sig"))["messages"][0]["content"].clone();
    assert_eq!(content, json!([{ "type": "text", "text": "Done" }]));

    // A Gemini answer restored without a producer keeps its text but not a signature; Mistral's
    // own answer keeps it.
    let mut own = answer(Some("why"), "sig");
    let mut entry = UsageEntry::new(Operation::Chat, "mistral", Some("magistral-small-latest"));
    entry.status = UsageStatus::Succeeded;
    own.usage_entries = vec![entry];
    assert_eq!(
        mistral_render(own)["messages"][0]["content"][0]["signature"],
        "sig"
    );
}

// ---- Video jobs: request settings, reported cost, usage ---------------------------------------

fn gpustack_context(server: &MockServer) -> Context {
    let mut config = Config::default();
    config.set(
        "gpustack_api_base",
        format!("{}/cluster/model/proxy/42/v1/", server.uri()),
    );
    config.set("gpustack_api_key", "isolated-key");
    config.video_generation_poll_interval = Duration::ZERO;
    Context::new(config)
}

// spec: protocols/gpustack/videos_spec.rb:57 reads the requested seconds and size
#[tokio::test]
async fn gpustack_reads_the_requested_seconds_and_size() {
    let server = json_server(
        "/cluster/model/proxy/42/v1/videos",
        json!({ "id": "video_gen_1", "model": "qwen3", "status": "queued" }),
    )
    .await;
    let job = gpustack_context(&server)
        .animate_later(
            Some("A street"),
            AnimateOptions {
                model: Some("qwen3"),
                provider: Some("gpustack"),
                provider_options: json!({ "seconds": "4", "size": "1280x720" }),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(
        (job.duration, job.resolution.as_deref()),
        (Some(4.0), Some("1280x720"))
    );
}

fn openrouter_options(provider_options: Value) -> AnimateOptions<'static> {
    AnimateOptions {
        model: Some("x-ai/grok-imagine-video"),
        provider: Some("openrouter"),
        // Ruby builds the protocol for this model directly, without a registry lookup.
        assume_model_exists: true,
        provider_options,
        ..Default::default()
    }
}

// spec: providers/openrouter/videos_spec.rb:47 #parse_video_request reads the requested duration and resolution, or the size in pixels
#[tokio::test]
async fn openrouter_reads_the_requested_duration_and_resolution_or_the_size() {
    let server = json_server(
        "/api/v1/videos",
        json!({ "id": "abc123", "status": "pending" }),
    )
    .await;
    let context = Context::new(config_for(&server, "openrouter", "/api/v1"));
    let job = context
        .animate_later(
            Some("a wave"),
            openrouter_options(json!({ "duration": 1, "resolution": "480p" })),
        )
        .await
        .unwrap();
    assert_eq!(
        (job.duration, job.resolution.as_deref()),
        (Some(1.0), Some("480p"))
    );
    let sized = context
        .animate_later(
            Some("a wave"),
            openrouter_options(json!({ "size": "1280x720" })),
        )
        .await
        .unwrap();
    assert_eq!(
        (sized.duration, sized.resolution.as_deref()),
        (None, Some("1280x720"))
    );
}

/// A server that accepts a job and answers every poll with `status`.
async fn video_server(route: &str, accepted: Value, poll: &str, status: Value) -> MockServer {
    let server = json_server(route, accepted).await;
    Mock::given(method("GET"))
        .and(path(poll))
        .respond_with(ResponseTemplate::new(200).set_body_json(status))
        .mount(&server)
        .await;
    server
}

// spec: providers/openrouter/videos_spec.rb:98 #parse_video_job_status reports the cost OpenRouter billed as the job and usage cost
#[tokio::test]
async fn openrouter_reports_the_billed_cost_as_the_job_and_usage_cost() {
    let body = json!({ "id": "abc123", "generation_id": "gen-1234567890-abcdef", "status": "completed",
        "unsigned_urls": ["https://openrouter.ai/api/v1/videos/abc123/content?index=0"],
        "usage": { "cost": 0.25, "is_byok": false } });
    let server = video_server(
        "/api/v1/videos",
        json!({ "id": "abc123", "status": "pending" }),
        "/api/v1/videos/abc123",
        body,
    )
    .await;
    let mut config = config_for(&server, "openrouter", "/api/v1");
    let events = capture(&mut config);
    let mut job = Context::new(config)
        .animate_later(Some("a wave"), openrouter_options(json!({})))
        .await
        .unwrap();
    job.refresh().await.unwrap();
    assert_eq!(job.cost().total(), Some(0.25));
    let usage = usages(&events);
    assert_eq!(usage.len(), 1);
    assert_eq!(usage[0]["operation"], "video");
    assert_eq!(usage[0]["cost"]["total"], json!(0.25));
}

fn xai_options(provider_options: Value) -> AnimateOptions<'static> {
    AnimateOptions {
        model: Some("grok-imagine-video"),
        provider: Some("xai"),
        provider_options,
        ..Default::default()
    }
}

// spec: providers/xai/videos_spec.rb:54 #parse_video_job keeps the requested duration and resolution on the job
#[tokio::test]
async fn xai_keeps_the_requested_duration_and_resolution_on_the_job() {
    let server = json_server(
        "/v1/videos/generations",
        json!({ "request_id": "request-1" }),
    )
    .await;
    let job = Context::new(config_for(&server, "xai", "/v1"))
        .animate_later(
            Some("a calm ocean wave"),
            xai_options(json!({ "duration": 5, "resolution": "720p" })),
        )
        .await
        .unwrap();
    assert_eq!(
        (job.id.as_str(), job.duration, job.resolution.as_deref()),
        ("request-1", Some(5.0), Some("720p"))
    );
}

// spec: providers/xai/videos_spec.rb:171 #parse_video_job_status reports the cost xAI billed in USD ticks as the job and usage cost
#[tokio::test]
async fn xai_reports_the_cost_billed_in_usd_ticks_as_the_job_and_usage_cost() {
    let body = json!({ "status": "done", "video": { "url": "https://vidgen.x.ai/clip.mp4", "duration": 1 },
        "model": "grok-imagine-video", "usage": { "cost_in_usd_ticks": 500_000_000 }, "progress": 100 });
    let server = video_server(
        "/v1/videos/generations",
        json!({ "request_id": "4482fadb" }),
        "/v1/videos/4482fadb",
        body,
    )
    .await;
    let mut config = config_for(&server, "xai", "/v1");
    let events = capture(&mut config);
    let mut job = Context::new(config)
        .animate_later(Some("a wave"), xai_options(json!({})))
        .await
        .unwrap();
    job.refresh().await.unwrap();
    let total = job.cost().total().unwrap();
    assert!((total - 0.05).abs() < 1e-12, "{total}");
    let usage = usages(&events);
    assert_eq!(usage.len(), 1);
    assert_eq!(usage[0]["operation"], "video");
    assert_eq!(usage[0]["cost"]["total"], json!(total));
}

// spec: video_spec.rb:99 VideoJob records its usage once when it finishes, attributed to the owner at submission
#[tokio::test]
async fn video_job_records_its_usage_once_when_it_finishes() {
    let model = "veo-3.1-fast-generate-preview";
    let server = json_server(
        &format!("/v1beta/models/{model}:predictLongRunning"),
        json!({ "name": "operations/op-1" }),
    )
    .await;
    Mock::given(method("GET"))
        .and(path("/v1beta/operations/op-1"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(json!({ "name": "operations/op-1" })),
        )
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/v1beta/operations/op-1"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "name": "operations/op-1", "done": true,
            "response": { "generateVideoResponse": { "generatedSamples": [{ "video": { "uri": "https://example.com/v.mp4" } }] } } })))
        .mount(&server)
        .await;
    let mut config = config_for(&server, "gemini", "/v1beta");
    let events = capture(&mut config);
    let mut job = Context::new(config)
        .animate_later(
            Some("a hummingbird"),
            AnimateOptions {
                model: Some(model),
                provider: Some("gemini"),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    job.refresh().await.unwrap();
    assert!(usages(&events).is_empty());
    job.refresh().await.unwrap();
    job.refresh().await.unwrap();
    assert_eq!(job.status, VideoStatus::Completed);
    let usage = usages(&events);
    assert_eq!(usage.len(), 1);
    assert_eq!(
        (
            &usage[0]["operation"],
            &usage[0]["provider"],
            &usage[0]["model"],
            &usage[0]["status"]
        ),
        (
            &json!("video"),
            &json!("gemini"),
            &json!(model),
            &json!("succeeded")
        )
    );
    assert!(usage[0]["cost"].get("total").is_none());
    // Owner attribution (`RubyLLM.with_usage_owner`) is checked where the owner API lives.
}

// ---- image_spec.rb / transcription_spec.rb: blocked Gemini attempts ---------------------------

// spec: image_spec.rb:242 blocked generations records the tokens Gemini billed when it returns no image
#[tokio::test]
async fn records_the_tokens_gemini_billed_when_it_returns_no_image() {
    let model = "gemini-3.1-flash-lite-image";
    let server = json_server(
        &format!("/v1beta/models/{model}:generateContent"),
        json!({ "candidates": [{ "finishReason": "IMAGE_SAFETY", "content": { "parts": [{ "text": "I cannot draw that." }] } }],
                "usageMetadata": { "promptTokenCount": 12, "candidatesTokenCount": 6 } }),
    )
    .await;
    let mut config = config_for(&server, "gemini", "/v1beta");
    let events = capture(&mut config);
    let result = Context::new(config)
        .paint(
            "A paper boat",
            PaintOptions {
                model: Some(model),
                provider: Some("gemini"),
                ..Default::default()
            },
        )
        .await;
    assert!(
        matches!(&result, Err(Error::Api(m, _)) if m == "Unexpected response format from Gemini image generation API"),
        "{result:?}"
    );
    let usage = usages(&events);
    assert_eq!(
        (&usage[0]["operation"], &usage[0]["status"]),
        (&json!("image"), &json!("failed"))
    );
    assert_eq!(
        usage[0]["tokens"],
        json!({ "input_tokens": 12, "output_tokens": 6 })
    );
}

const TRANSCRIPTION: &str = "gemini-2.5-flash";

async fn gemini_transcription(body: Value) -> (rust_llm::Result<rust_llm::Transcription>, Events) {
    let server = json_server(
        &format!("/v1beta/models/{TRANSCRIPTION}:generateContent"),
        body,
    )
    .await;
    let mut config = config_for(&server, "gemini", "/v1beta");
    let events = capture(&mut config);
    let result = rust_llm::transcribe(
        fixture("ruby.wav").as_str(),
        TranscribeOptions {
            model: Some(TRANSCRIPTION),
            provider: Some("gemini"),
            config: Some(Arc::new(config)),
            ..Default::default()
        },
    )
    .await;
    (result, events)
}

// spec: transcription_spec.rb:42 blocked transcriptions raises instead of returning an empty transcript when Gemini blocks the audio
#[tokio::test]
async fn raises_instead_of_returning_an_empty_transcript_when_gemini_blocks_the_audio() {
    let (result, _) = gemini_transcription(json!({
        "promptFeedback": { "blockReason": "SAFETY" },
        "usageMetadata": { "promptTokenCount": 133, "totalTokenCount": 133 },
        "modelVersion": TRANSCRIPTION
    }))
    .await;
    assert!(
        matches!(&result, Err(Error::ContentFilter(m, _)) if m == "Gemini blocked the transcription: SAFETY"),
        "{result:?}"
    );
}

// spec: transcription_spec.rb:56 blocked transcriptions records the tokens Gemini billed for the blocked attempt
#[tokio::test]
async fn records_the_tokens_gemini_billed_for_the_blocked_attempt() {
    let (result, events) = gemini_transcription(json!({
        "candidates": [{ "finishReason": "SAFETY" }],
        "usageMetadata": { "promptTokenCount": 133, "candidatesTokenCount": 0, "totalTokenCount": 133 }
    }))
    .await;
    assert!(
        matches!(&result, Err(Error::ContentFilter(..))),
        "{result:?}"
    );
    let usage = usages(&events);
    assert_eq!(
        (&usage[0]["operation"], &usage[0]["status"]),
        (&json!("transcription"), &json!("failed"))
    );
    assert_eq!(
        usage[0]["tokens"],
        json!({ "input_tokens": 133, "output_tokens": 0 })
    );
    assert!(usage[0]["cost"].get("total").is_some());
}
