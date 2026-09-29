//! Chat Completions dialect specs ported from RubyLLM 2.0: `providers/{openrouter,mistral,deepseek,
//! ollama}/chat_spec.rb`, `providers/{ollama,gpustack,openrouter}/media_spec.rb`, and
//! `providers/gpustack_spec.rb`. Ruby calls the dialect's private `render_payload`,
//! `format_messages`, `parse_completion_response`, and `build_chunk`; these go through
//! `Chat#render`, `ask`, and `ask_stream` against a mock server, which run the same code.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use base64::Engine;
use rust_llm::message::indexmap_lite::IndexMap;
use rust_llm::protocols::chat_completions::inject_cache_control;
use rust_llm::{
    Attachment, Chat, Config, Error, FinishReason, Message, Role, ThinkingConfig, ThinkingDisplay, Tool, ToolCall, ToolChoice, ToolError,
    ToolResult,
};
use rust_llm::message::Thinking;
use serde_json::{Map, Value, json};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Every provider in this file pointed at `server`.
fn config(server: &MockServer) -> Arc<Config> {
    let mut c = Config::default();
    for (provider, prefix) in
        [("openrouter", "/api/v1"), ("mistral", "/v1"), ("deepseek", ""), ("ollama", "/v1"), ("gpustack", "/v1")]
    {
        c.set(format!("{provider}_api_base"), format!("{}{prefix}", server.uri()));
        c.set(format!("{provider}_api_key"), "test");
    }
    c.max_retries = 0;
    Arc::new(c)
}

fn chat(server: &MockServer, provider: &str, model: &str) -> Chat {
    Chat::with_config(config(server), Some(model), Some(provider), true).expect("chat")
}

/// A chat holding `messages`, rendered.
fn render(chat: Chat, messages: Vec<Message>) -> rust_llm::Result<Value> {
    let mut chat = chat;
    for m in messages {
        chat.add_message(m);
    }
    chat.render()
}

fn docx() -> Attachment {
    Attachment::from_bytes(b"docx bytes".to_vec(), "proposal.docx", None)
}

fn b64(bytes: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

fn assistant_thinking(content: Option<&str>, text: Option<&str>, signature: Option<&str>) -> Message {
    let mut m = Message::new(Role::Assistant, content.map(str::to_string));
    m.thinking = Some(Thinking { text: text.map(str::to_string), signature: signature.map(str::to_string) });
    m
}

fn assert_unsupported_docx(result: rust_llm::Result<Value>) {
    match result {
        Err(Error::UnsupportedAttachment(msg)) => assert!(
            msg.contains("Unsupported attachment type: application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
            "{msg}"
        ),
        other => panic!("expected UnsupportedAttachment, got {other:?}"),
    }
}

async fn json_server(body: Value) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(wiremock::matchers::any()).respond_with(ResponseTemplate::new(200).set_body_json(body)).mount(&server).await;
    server
}

async fn sse_server(events: &[Value]) -> MockServer {
    let mut body: String = events.iter().map(|e| format!("data: {e}\n\n")).collect();
    body.push_str("data: [DONE]\n\n");
    let server = MockServer::start().await;
    Mock::given(wiremock::matchers::any())
        .respond_with(ResponseTemplate::new(200).set_body_raw(body.into_bytes(), "text/event-stream"))
        .mount(&server)
        .await;
    server
}

/// Collects `tracing` events on this thread, standing in for `RubyLLM.logger`.
struct LogCollector(Arc<Mutex<Vec<(tracing::Level, String)>>>);

impl tracing::Subscriber for LogCollector {
    // Tests run in parallel: a callsite first hit with no collector set is cached as "never",
    // so ask on every event instead of caching the interest.
    fn register_callsite(&self, _: &'static tracing::Metadata<'static>) -> tracing::subscriber::Interest {
        tracing::subscriber::Interest::sometimes()
    }
    // Without this the global max level is recomputed from other threads' (absent) collectors and
    // can drop WARN events before they reach this one.
    fn max_level_hint(&self) -> Option<tracing::level_filters::LevelFilter> {
        Some(tracing::level_filters::LevelFilter::TRACE)
    }
    fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        struct Text<'a>(&'a mut String);
        impl tracing::field::Visit for Text<'_> {
            fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                if field.name() == "message" {
                    self.0.push_str(&format!("{value:?}"));
                }
            }
        }
        let mut text = String::new();
        event.record(&mut Text(&mut text));
        self.0.lock().unwrap().push((*event.metadata().level(), text));
    }
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}

fn logs_of<T>(f: impl FnOnce() -> T) -> (T, Vec<(tracing::Level, String)>) {
    let logs = Arc::new(Mutex::new(Vec::new()));
    let out = {
        let _guard = tracing::dispatcher::set_default(&tracing::Dispatch::new(LogCollector(logs.clone())));
        f()
    };
    let logs = logs.lock().unwrap().clone();
    (out, logs)
}

// ================================================================================================
// providers/openrouter/chat_spec.rb
// ================================================================================================

const OPENROUTER_MODEL: &str = "anthropic/claude-haiku-4.5";

fn openrouter(server: &MockServer) -> Chat {
    chat(server, "openrouter", OPENROUTER_MODEL)
}

/// `parse_completion_response` on a canned OpenRouter body, through `ask`.
async fn openrouter_parse(body: Value) -> rust_llm::Result<Message> {
    let server = json_server(body).await;
    openrouter(&server).ask("hi").await
}

fn openrouter_message(message: Value, usage: Value) -> Value {
    json!({ "model": "openai/gpt-4.1-nano", "choices": [{ "message": message }], "usage": usage })
}

// spec: providers/openrouter/chat_spec.rb:9 #parse_completion_response raises RubyLLM::Error for a nil response body
#[tokio::test]
async fn openrouter_nil_body_raises_empty_body_error() {
    let server = MockServer::start().await;
    Mock::given(wiremock::matchers::any())
        .respond_with(ResponseTemplate::new(200).set_body_raw(b"null".to_vec(), "application/json"))
        .mount(&server)
        .await;
    let err = openrouter(&server).ask("hi").await.unwrap_err();
    assert!(matches!(err, Error::Api(..)), "{err:?}");
    assert_eq!(err.to_string(), "Provider returned an empty response body");
}

// spec: providers/openrouter/chat_spec.rb:17 #parse_completion_response normalizes cached prompt tokens out of input tokens
#[tokio::test]
async fn openrouter_normalizes_cached_prompt_tokens() {
    let message = openrouter_parse(openrouter_message(
        json!({ "role": "assistant", "content": "Hello!" }),
        json!({ "prompt_tokens": 12, "completion_tokens": 4, "prompt_tokens_details": { "cached_tokens": 6, "cache_write_tokens": 4 } }),
    ))
    .await
    .unwrap();
    assert_eq!(message.tokens.input, Some(2));
    assert_eq!(message.tokens.cache_read, Some(6));
    assert_eq!(message.tokens.cache_write, Some(4));
    assert_eq!(message.tokens.output, Some(4));
}

// spec: providers/openrouter/chat_spec.rb:44 #parse_completion_response normalizes OpenAI-compatible reasoning tokens that are reported outside completion tokens
#[tokio::test]
async fn openrouter_counts_reasoning_tokens_reported_outside_completion_tokens() {
    let message = openrouter_parse(json!({
        "model": "x-ai/grok-4-fast-reasoning",
        "choices": [{ "message": { "role": "assistant", "content": "Hello!" } }],
        "usage": { "prompt_tokens": 50, "completion_tokens": 208, "total_tokens": 2443, "completion_tokens_details": { "reasoning_tokens": 2185 } }
    }))
    .await
    .unwrap();
    assert_eq!(message.tokens.output, Some(2393));
    assert_eq!(message.tokens.thinking, Some(2185));
}

// spec: providers/openrouter/chat_spec.rb:70 #parse_completion_response captures the reported cost from usage
#[tokio::test]
async fn openrouter_captures_reported_cost() {
    let message = openrouter_parse(json!({
        "model": "meta-llama/llama-3.2-1b-instruct",
        "choices": [{ "message": { "role": "assistant", "content": "Hi!" } }],
        "usage": {
            "prompt_tokens": 13, "completion_tokens": 3, "total_tokens": 16, "cost": 9.54e-07, "is_byok": false,
            "cost_details": { "upstream_inference_cost": 9.54e-07 }
        }
    }))
    .await
    .unwrap();
    assert_eq!(message.tokens.reported_cost, Some(9.54e-07));
    assert_eq!(message.cost(None).total(), Some(9.54e-07));
}

// spec: providers/openrouter/chat_spec.rb:93 #parse_completion_response adds the upstream inference cost to the OpenRouter fee on BYOK requests
#[tokio::test]
async fn openrouter_byok_cost_adds_upstream_inference_cost() {
    let message = openrouter_parse(openrouter_message(
        json!({ "role": "assistant", "content": "Hi!" }),
        json!({ "cost": 0.0001, "is_byok": true, "cost_details": { "upstream_inference_cost": 0.002 } }),
    ))
    .await
    .unwrap();
    let cost = message.tokens.reported_cost.expect("reported cost");
    assert!((cost - 0.0021).abs() <= 1e-12, "{cost}");
}

// spec: providers/openrouter/chat_spec.rb:103 #parse_completion_response reports no cost when usage carries none
#[tokio::test]
async fn openrouter_reports_no_cost_without_one() {
    let message = openrouter_parse(openrouter_message(json!({ "role": "assistant", "content": "Hi!" }), json!({}))).await.unwrap();
    assert_eq!(message.tokens.reported_cost, None);
}

async fn openrouter_chunks(events: &[Value]) -> Vec<Message> {
    let server = sse_server(events).await;
    let mut chunks = Vec::new();
    openrouter(&server).ask_stream("hi", |c| chunks.push(c.clone())).await.unwrap();
    chunks
}

// spec: providers/openrouter/chat_spec.rb:109 #build_chunk preserves raw finish reasons on streaming chunks
#[tokio::test]
async fn openrouter_chunk_keeps_finish_reason() {
    let chunks = openrouter_chunks(&[json!({
        "model": "openai/gpt-4.1-nano",
        "choices": [{ "delta": { "content": "" }, "finish_reason": "tool_calls" }]
    })])
    .await;
    assert_eq!(chunks[0].finish_reason, Some(FinishReason::ToolCalls));
}

// spec: providers/openrouter/chat_spec.rb:123 #build_chunk captures the reported cost from the final usage chunk
#[tokio::test]
async fn openrouter_final_usage_chunk_carries_reported_cost() {
    let chunks = openrouter_chunks(&[
        json!({ "model": "openai/gpt-4.1-nano", "choices": [{ "delta": { "content": "Hi" } }] }),
        json!({
            "model": "openai/gpt-4.1-nano", "choices": [],
            "usage": { "prompt_tokens": 13, "completion_tokens": 3, "cost": 9.54e-07, "is_byok": false }
        }),
    ])
    .await;
    assert_eq!(chunks.last().unwrap().tokens.reported_cost, Some(9.54e-07));
}

// spec: providers/openrouter/chat_spec.rb:154 #format_messages keeps non-PDF documents disabled for OpenRouter chat completions
#[tokio::test]
async fn openrouter_rejects_non_pdf_documents() {
    let server = MockServer::start().await;
    let message = Message::user("Summarize this file").with_attachments(vec![docx()]);
    assert_unsupported_docx(render(openrouter(&server), vec![message]));
}

fn cache_boundary(content: &str) -> Message {
    let mut m = Message::user(content);
    m.cache_until_here = true;
    m
}

// spec: providers/openrouter/chat_spec.rb:166 #format_messages adds cache_control to a message marked as a cache boundary
#[tokio::test]
async fn openrouter_cache_boundary_gets_cache_control() {
    let server = MockServer::start().await;
    let payload = render(openrouter(&server), vec![cache_boundary("Long context")]).unwrap();
    let last = payload["messages"][0]["content"].as_array().and_then(|c| c.last()).cloned().unwrap();
    assert_eq!(last["cache_control"], json!({ "type": "ephemeral" }));
}

// spec: providers/openrouter/chat_spec.rb:174 #format_messages uses configured cache_control for a cache boundary
#[tokio::test]
async fn openrouter_cache_boundary_uses_configured_ttl() {
    let server = MockServer::start().await;
    let chat = openrouter(&server).with_caching(json!({ "ttl": "1h" })).unwrap();
    let payload = render(chat, vec![cache_boundary("Long context")]).unwrap();
    let last = payload["messages"][0]["content"].as_array().and_then(|c| c.last()).cloned().unwrap();
    assert_eq!(last["cache_control"], json!({ "type": "ephemeral", "ttl": "1h" }));
}

// spec: providers/openrouter/chat_spec.rb:237 #render_payload uses wrapper schema name and inner schema
#[tokio::test]
async fn openrouter_uses_wrapper_schema_name_and_inner_schema() {
    let server = MockServer::start().await;
    let inner = json!({ "type": "object", "properties": { "name": { "type": "string" } } });
    let chat = openrouter(&server).with_schema(json!({ "name": "PersonSchema", "schema": inner, "strict": false }));
    let payload = render(chat, vec![Message::user("Hello")]).unwrap();
    assert_eq!(payload["response_format"]["json_schema"]["name"], "PersonSchema");
    assert_eq!(payload["response_format"]["json_schema"]["schema"], inner);
    assert_eq!(payload["response_format"]["json_schema"]["strict"], false);
}

// spec: providers/openrouter/chat_spec.rb:264 #render_payload adds top-level automatic cache_control when caching is enabled without explicit boundaries
#[tokio::test]
async fn openrouter_adds_top_level_cache_control() {
    let server = MockServer::start().await;
    let chat = openrouter(&server).with_caching(json!({ "ttl": "1h" })).unwrap();
    let payload = render(chat, vec![Message::user("Hello")]).unwrap();
    assert_eq!(payload["cache_control"], json!({ "type": "ephemeral", "ttl": "1h" }));
}

// spec: providers/openrouter/chat_spec.rb:278 #render_payload adds top-level cache_control alongside an explicit boundary
#[tokio::test]
async fn openrouter_adds_top_level_cache_control_alongside_a_boundary() {
    let server = MockServer::start().await;
    let chat = openrouter(&server).with_caching(json!({ "ttl": "1h" })).unwrap();
    let payload = render(chat, vec![cache_boundary("Long context"), Message::user("Latest question")]).unwrap();
    assert_eq!(payload["cache_control"], json!({ "type": "ephemeral", "ttl": "1h" }));
    let last = payload["messages"][0]["content"].as_array().and_then(|c| c.last()).cloned().unwrap();
    assert_eq!(last["cache_control"], json!({ "type": "ephemeral", "ttl": "1h" }));
}

// spec: providers/openrouter/chat_spec.rb:299 #render_payload rejects caching options it cannot render
#[tokio::test]
async fn openrouter_rejects_unsupported_caching_options() {
    let server = MockServer::start().await;
    let chat = openrouter(&server).with_caching(json!({ "retention": "24h" })).unwrap();
    match render(chat, vec![Message::user("Hello")]) {
        Err(Error::Argument(msg)) => assert!(msg.contains("OpenRouter prompt caching accepts :ttl"), "{msg}"),
        other => panic!("expected ArgumentError, got {other:?}"),
    }
}

/// Ruby passes a bare `Struct.new(:enabled?)`; the Rust counterpart of "enabled, with no effort,
/// budget, or explicit toggle" is a config that only sets the display.
// spec: providers/openrouter/chat_spec.rb:328 #build_reasoning falls back to just enabling reasoning
#[tokio::test]
async fn openrouter_reasoning_falls_back_to_enabled() {
    let server = MockServer::start().await;
    let chat = openrouter(&server).with_thinking(ThinkingConfig::default().with_display(ThinkingDisplay::Summarized));
    let payload = render(chat, vec![Message::user("Hello")]).unwrap();
    assert_eq!(payload["reasoning"], json!({ "enabled": true }));
}

// spec: providers/openrouter/chat_spec.rb:336 #format_thinking ignores native replay data from another protocol
#[tokio::test]
async fn openrouter_ignores_other_protocols_replay_data() {
    let server = MockServer::start().await;
    let mut message = Message::assistant("done");
    message.raw_reasoning = Some(json!({ "anthropic": [{ "type": "redacted_thinking", "data": "encrypted" }] }));
    let payload = render(openrouter(&server), vec![message]).unwrap();
    assert_eq!(payload["messages"][0], json!({ "role": "assistant", "content": "done" }));
}

// spec: providers/openrouter/chat_spec.rb:350 #format_thinking sends reasoning text with its signature
#[tokio::test]
async fn openrouter_sends_reasoning_text_with_signature() {
    let server = MockServer::start().await;
    let payload = render(openrouter(&server), vec![assistant_thinking(Some("done"), Some("why"), Some("sig"))]).unwrap();
    assert_eq!(payload["messages"][0]["reasoning_details"], json!([{ "type": "reasoning.text", "text": "why", "signature": "sig" }]));
}

/// `extract_thinking_text` / `extract_thinking_signature` on a response message, through `ask`.
async fn openrouter_thinking(message: Value) -> Option<Thinking> {
    let mut message = message;
    message["role"] = "assistant".into();
    message["content"] = "ok".into();
    openrouter_parse(openrouter_message(message, json!({}))).await.unwrap().thinking
}

// spec: providers/openrouter/chat_spec.rb:372 reasoning details on the way back joins reasoning text and summary details
#[tokio::test]
async fn openrouter_joins_reasoning_text_and_summary_details() {
    let thinking = openrouter_thinking(json!({ "reasoning_details": [
        { "type": "reasoning.text", "text": "first " },
        { "type": "reasoning.summary", "summary": "second" },
        { "type": "reasoning.encrypted", "data": "blob" }
    ]}))
    .await;
    assert_eq!(thinking.and_then(|t| t.text).as_deref(), Some("first second"));
}

// spec: providers/openrouter/chat_spec.rb:384 reasoning details on the way back is nil when the response carries no reasoning details
#[tokio::test]
async fn openrouter_thinking_is_nil_without_reasoning_details() {
    assert_eq!(openrouter_thinking(json!({})).await, None);
}

// spec: providers/openrouter/chat_spec.rb:389 reasoning details on the way back is nil when the details carry no text
#[tokio::test]
async fn openrouter_thinking_text_is_nil_when_details_carry_no_text() {
    assert_eq!(openrouter_thinking(json!({ "reasoning_details": [] })).await.and_then(|t| t.text), None);
}

// spec: providers/openrouter/chat_spec.rb:393 reasoning details on the way back prefers an explicit signature over encrypted data
#[tokio::test]
async fn openrouter_prefers_explicit_signature_over_encrypted_data() {
    let thinking = openrouter_thinking(json!({ "reasoning_details": [
        { "type": "reasoning.encrypted", "data": "blob" },
        { "type": "reasoning.text", "signature": "sig" }
    ]}))
    .await;
    assert_eq!(thinking.and_then(|t| t.signature).as_deref(), Some("sig"));
}

// spec: providers/openrouter/chat_spec.rb:404 reasoning details on the way back falls back to encrypted data
#[tokio::test]
async fn openrouter_signature_falls_back_to_encrypted_data() {
    let thinking = openrouter_thinking(json!({ "reasoning_details": [{ "type": "reasoning.encrypted", "data": "blob" }] })).await;
    assert_eq!(thinking.and_then(|t| t.signature).as_deref(), Some("blob"));
}

// spec: providers/openrouter/chat_spec.rb:412 #inject_cache_control wraps plain text content in a cacheable block
#[test]
fn inject_cache_control_wraps_plain_text() {
    assert_eq!(
        inject_cache_control(json!("hello"), None).unwrap(),
        json!([{ "type": "text", "text": "hello", "cache_control": { "type": "ephemeral" } }])
    );
}

// spec: providers/openrouter/chat_spec.rb:418 #inject_cache_control leaves empty content alone
#[test]
fn inject_cache_control_leaves_empty_content_alone() {
    assert_eq!(inject_cache_control(json!([]), None).unwrap(), json!([]));
}

// spec: providers/openrouter/chat_spec.rb:422 #inject_cache_control leaves a block that already carries cache_control alone
#[test]
fn inject_cache_control_keeps_an_existing_cache_control() {
    let blocks = json!([{ "type": "text", "text": "hello", "cache_control": { "type": "ephemeral" } }]);
    let caching = rust_llm::Caching::On(json!({ "ttl": "1h" }).as_object().cloned().unwrap());
    assert_eq!(inject_cache_control(blocks.clone(), Some(&caching)).unwrap(), blocks);
}

// spec: providers/openrouter/chat_spec.rb:428 #inject_cache_control leaves a trailing block it cannot annotate alone
#[test]
fn inject_cache_control_leaves_a_trailing_non_block_alone() {
    assert_eq!(inject_cache_control(json!(["plain"]), None).unwrap(), json!(["plain"]));
}

// ================================================================================================
// providers/openrouter/media_spec.rb
// ================================================================================================

// spec: providers/openrouter/media_spec.rb:7 formats video attachments as video_url parts
#[tokio::test]
async fn openrouter_formats_video_url_parts() {
    let server = MockServer::start().await;
    let chat = chat(&server, "openrouter", "openai/gpt-5.2");
    let message = Message::user("what happens here?").with_attachments(vec![Attachment::new("https://example.com/clip.mp4")]);
    let payload = render(chat, vec![message]).unwrap();
    let part = payload["messages"][0]["content"].as_array().unwrap().iter().find(|p| p["type"] == "video_url").cloned();
    assert_eq!(part, Some(json!({ "type": "video_url", "video_url": { "url": "https://example.com/clip.mp4" } })));
}

// ================================================================================================
// providers/mistral/chat_spec.rb
// ================================================================================================

fn mistral(server: &MockServer, model: &str) -> Chat {
    chat(server, "mistral", model)
}

fn mistral_payload(model: &str, thinking: Option<ThinkingConfig>) -> Value {
    let server_uri = "http://127.0.0.1:9";
    let mut c = Config::default();
    c.set("mistral_api_base", server_uri);
    c.set("mistral_api_key", "test");
    let mut chat = Chat::with_config(Arc::new(c), Some(model), Some("mistral"), true).unwrap();
    if let Some(t) = thinking {
        chat = chat.with_thinking(t);
    }
    render(chat, vec![Message::user("Hello")]).unwrap()
}

// spec: providers/mistral/chat_spec.rb:26 #render_payload renders system messages before conversation messages for Mistral
#[tokio::test]
async fn mistral_renders_system_messages_first() {
    let server = MockServer::start().await;
    let payload = render(mistral(&server, "mistral-small-latest"), vec![Message::user("Hello"), Message::system("Be terse.")]).unwrap();
    let roles: Vec<&str> = payload["messages"].as_array().unwrap().iter().filter_map(|m| m["role"].as_str()).collect();
    assert_eq!(roles, ["system", "user"]);
}

// spec: providers/mistral/chat_spec.rb:38 #render_payload renders Mistral prompt cache key
#[tokio::test]
async fn mistral_renders_prompt_cache_key() {
    let server = MockServer::start().await;
    let chat = mistral(&server, "mistral-large-latest").with_caching(json!({ "key": "support-session-42" })).unwrap();
    assert_eq!(render(chat, vec![Message::user("Hello")]).unwrap()["prompt_cache_key"], "support-session-42");
}

// spec: providers/mistral/chat_spec.rb:44 #render_payload rejects caching options Mistral cannot render
#[tokio::test]
async fn mistral_rejects_unsupported_caching_options() {
    let server = MockServer::start().await;
    let chat = mistral(&server, "mistral-large-latest").with_caching(json!({ "retention": "24h" })).unwrap();
    match render(chat, vec![Message::user("Hello")]) {
        Err(Error::Argument(msg)) => assert!(msg.contains("Mistral prompt caching accepts :key"), "{msg}"),
        other => panic!("expected ArgumentError, got {other:?}"),
    }
}

// spec: providers/mistral/chat_spec.rb:60 #render_payload sends the effort the caller asked for rather than a supported one
#[test]
fn mistral_sends_the_requested_effort() {
    assert_eq!(mistral_payload("mistral-medium-latest", Some(ThinkingConfig::effort("medium")))["reasoning_effort"], "medium");
}

// spec: providers/mistral/chat_spec.rb:69 #render_payload keeps explicit none effort
#[test]
fn mistral_keeps_explicit_none_effort() {
    assert_eq!(mistral_payload("mistral-small-latest", Some(ThinkingConfig::effort("none")))["reasoning_effort"], "none");
}

// spec: providers/mistral/chat_spec.rb:78 #render_payload sends reasoning_effort without checking the model id
#[test]
fn mistral_sends_effort_for_any_model_id() {
    let payload = mistral_payload("pixtral-12b", Some(ThinkingConfig::effort("medium")));
    assert_eq!(payload["reasoning_effort"], "medium");
    assert!(payload.get("prompt_mode").is_none());
}

// spec: providers/mistral/chat_spec.rb:222 reasoning effort leaves every effort the caller picks untouched
#[test]
fn mistral_leaves_every_effort_untouched() {
    for effort in ["high", "none", "low", "medium", "xhigh"] {
        assert_eq!(mistral_payload("magistral-medium-latest", Some(ThinkingConfig::effort(effort)))["reasoning_effort"], effort);
    }
}

// spec: providers/mistral/chat_spec.rb:97 #format_messages keeps parallel tool results consecutive and moves attachment carriers after the run
#[tokio::test]
async fn mistral_keeps_parallel_tool_results_consecutive() {
    let server = MockServer::start().await;
    let png = Attachment::from_bytes(b"png bytes".to_vec(), "chart.png", None);
    let mut call = Message::new(Role::Assistant, None);
    let calls: IndexMap<ToolCall> =
        ["call_1", "call_2"].into_iter().map(|id| (id.to_string(), ToolCall::new(id, "chart", Map::new()))).collect();
    call.tool_calls = Some(calls);
    let payload = render(
        mistral(&server, "mistral-small-latest"),
        vec![
            Message::user("Chart it"),
            call,
            Message::tool_result("call_1", "first").with_attachments(vec![png]),
            Message::tool_result("call_2", "second"),
        ],
    )
    .unwrap();
    let messages = payload["messages"].as_array().unwrap();
    let roles: Vec<&str> = messages.iter().filter_map(|m| m["role"].as_str()).collect();
    assert_eq!(roles, ["user", "assistant", "tool", "tool", "user"]);
    assert_eq!(messages.last().unwrap()["content"].as_array().unwrap().last().unwrap()["type"], "image_url");
}

struct Named(&'static str, Map<String, Value>);

#[async_trait]
impl Tool for Named {
    fn name(&self) -> String {
        self.0.into()
    }
    fn description(&self) -> String {
        String::new()
    }
    fn provider_options(&self) -> Map<String, Value> {
        self.1.clone()
    }
    async fn execute(&self, _a: Map<String, Value>, _c: &ToolCall) -> Result<ToolResult, ToolError> {
        Ok("".into())
    }
}

// spec: providers/mistral/chat_spec.rb:152 #build_tool_choice maps required tool choice to the Mistral any mode
// spec: providers/mistral/chat_spec.rb:247 #normalize_required_tool_choice leaves a multi-tool request on the any mode
#[tokio::test]
async fn mistral_required_tool_choice_is_any_for_several_tools() {
    let server = MockServer::start().await;
    let chat = mistral(&server, "mistral-small-latest")
        .with_tool(Named("weather", Map::new()))
        .with_tool(Named("time", Map::new()))
        .with_tool_choice(ToolChoice::Required)
        .unwrap();
    assert_eq!(render(chat, vec![Message::user("Hello")]).unwrap()["tool_choice"], "any");
}

/// Ruby's payload has `function: {}`; a Rust tool always has a name, so its provider options null
/// the name out the same way a deep merge would.
// spec: providers/mistral/chat_spec.rb:261 #normalize_required_tool_choice leaves the payload alone when the single tool has no name
#[tokio::test]
async fn mistral_leaves_any_for_a_nameless_single_tool() {
    let server = MockServer::start().await;
    let nameless = json!({ "function": { "name": null } }).as_object().cloned().unwrap();
    let chat = mistral(&server, "mistral-small-latest").with_tool(Named("weather", nameless)).with_tool_choice(ToolChoice::Required).unwrap();
    assert_eq!(render(chat, vec![Message::user("Hello")]).unwrap()["tool_choice"], "any");
}

fn mistral_content(message: Message) -> Value {
    let mut c = Config::default();
    c.set("mistral_api_base", "http://127.0.0.1:9");
    c.set("mistral_api_key", "test");
    let chat = Chat::with_config(Arc::new(c), Some("magistral-small-latest"), Some("mistral"), true).unwrap();
    render(chat, vec![message]).unwrap()["messages"][0]["content"].clone()
}

// spec: providers/mistral/chat_spec.rb:182 #build_thinking_blocks wraps thinking text in a text block
#[test]
fn mistral_wraps_thinking_text_with_its_signature() {
    let content = mistral_content(assistant_thinking(Some("Done"), Some("why"), Some("sig")));
    assert_eq!(content[0], json!({ "type": "thinking", "thinking": [{ "type": "text", "text": "why" }], "signature": "sig" }));
}

// spec: providers/mistral/chat_spec.rb:190 #build_thinking_blocks sends a signature-only block
#[test]
fn mistral_sends_a_signature_only_thinking_block() {
    let content = mistral_content(assistant_thinking(Some("Done"), None, Some("sig")));
    assert_eq!(content[0], json!({ "type": "thinking", "signature": "sig" }));
}

/// The formatted content is a list of parts when the message carries attachments; the parts follow
/// the thinking block flat rather than nested.
// spec: providers/mistral/chat_spec.rb:198 #append_formatted_content concatenates a list of parts
#[test]
fn mistral_concatenates_a_list_of_parts_after_the_thinking_block() {
    let png = Attachment::from_bytes(b"png bytes".to_vec(), "chart.png", None);
    let message = assistant_thinking(Some("hi"), None, Some("sig")).with_attachments(vec![png]);
    let content = mistral_content(message);
    assert_eq!(
        content,
        json!([
            { "type": "thinking", "signature": "sig" },
            { "type": "text", "text": "hi" },
            { "type": "image_url", "image_url": format!("data:image/png;base64,{}", b64(b"png bytes")) }
        ])
    );
}

// spec: providers/mistral/chat_spec.rb:212 #append_formatted_content leaves the blocks alone for empty content
#[test]
fn mistral_leaves_thinking_blocks_alone_for_empty_content() {
    for content in [None, Some("")] {
        assert_eq!(mistral_content(assistant_thinking(content, None, Some("sig"))), json!([{ "type": "thinking", "signature": "sig" }]));
    }
}

// spec: providers/mistral/chat_spec.rb:235 #prompt_cache_params renders only the cache key
#[tokio::test]
async fn mistral_prompt_cache_params_render_only_the_key() {
    let server = MockServer::start().await;
    let chat = mistral(&server, "mistral-small-latest").with_caching(json!({ "key": "abc" })).unwrap();
    let payload = render(chat, vec![Message::user("Hello")]).unwrap();
    assert_eq!(payload["prompt_cache_key"], "abc");
    for key in ["prompt_cache_options", "prompt_cache_retention", "cache_control"] {
        assert!(payload.get(key).is_none(), "{key}");
    }
}

// spec: providers/mistral/chat_spec.rb:239 #prompt_cache_params rejects options Mistral cannot render
#[tokio::test]
async fn mistral_prompt_cache_params_reject_ttl() {
    let server = MockServer::start().await;
    let chat = mistral(&server, "mistral-small-latest").with_caching(json!({ "ttl": "1h" })).unwrap();
    match render(chat, vec![Message::user("Hello")]) {
        Err(Error::Argument(msg)) => assert_eq!(msg, "Mistral prompt caching accepts :key, got :ttl"),
        other => panic!("expected ArgumentError, got {other:?}"),
    }
}

// ================================================================================================
// providers/deepseek/chat_spec.rb
// ================================================================================================

fn deepseek_render(thinking: Option<ThinkingConfig>, schema: Option<Value>, messages: Vec<Message>) -> Value {
    let mut c = Config::default();
    c.set("deepseek_api_base", "http://127.0.0.1:9");
    c.set("deepseek_api_key", "test");
    let mut chat = Chat::with_config(Arc::new(c), Some("deepseek-v4-flash"), Some("deepseek"), true).unwrap();
    if let Some(t) = thinking {
        chat = chat.with_thinking(t);
    }
    if let Some(s) = schema {
        chat = chat.with_schema(s);
    }
    render(chat, messages).unwrap()
}

// spec: providers/deepseek/chat_spec.rb:39 .render_payload sends efforts outside the common tiers unchanged
#[test]
fn deepseek_sends_efforts_unchanged() {
    let medium = deepseek_render(Some(ThinkingConfig::effort("medium")), None, vec![Message::user("Hello")]);
    assert_eq!(medium["thinking"], json!({ "type": "enabled" }));
    assert_eq!(medium["reasoning_effort"], "medium");
    for effort in ["minimal", "xhigh"] {
        assert_eq!(deepseek_render(Some(ThinkingConfig::effort(effort)), None, vec![Message::user("Hello")])["reasoning_effort"], effort);
    }
}

// spec: providers/deepseek/chat_spec.rb:51 .render_payload ignores thinking budgets with a debug note
#[test]
fn deepseek_ignores_thinking_budgets_with_a_debug_note() {
    let (payload, logs) = logs_of(|| deepseek_render(Some(ThinkingConfig::budget(2048)), None, vec![Message::user("Hello")]));
    assert_eq!(payload["thinking"], json!({ "type": "enabled" }));
    assert!(payload.get("reasoning_effort").is_none());
    assert!(logs.iter().any(|(level, msg)| *level == tracing::Level::DEBUG && msg.contains("DeepSeek has no thinking budgets")), "{logs:?}");
}

// spec: providers/deepseek/chat_spec.rb:68 .render_payload degrades json_schema response formats to json_object with a warning
#[test]
fn deepseek_degrades_json_schema_to_json_object() {
    let schema = json!({ "name": "person", "schema": { "type": "object" }, "strict": true });
    let (payload, logs) = logs_of(|| deepseek_render(None, Some(schema), vec![Message::user("Hello")]));
    assert_eq!(payload["response_format"], json!({ "type": "json_object" }));
    assert!(logs.iter().any(|(level, msg)| *level == tracing::Level::WARN && msg.contains("does not support json_schema")), "{logs:?}");
}

// spec: providers/deepseek/chat_spec.rb:80 .format_thinking with an assistant message emits reasoning_content (and reasoning) when thinking text is present
#[test]
fn deepseek_replays_reasoning_with_its_signature() {
    let payload = deepseek_render(None, None, vec![assistant_thinking(Some("Hi"), Some("pondering"), Some("sig"))]);
    let message = &payload["messages"][0];
    assert_eq!(message["reasoning_content"], "pondering");
    assert_eq!(message["reasoning"], "pondering");
    assert_eq!(message["reasoning_signature"], "sig");
}

// ================================================================================================
// providers/ollama/{chat,media}_spec.rb
// ================================================================================================

fn local_chat(provider: &str, model: &str) -> Chat {
    let mut c = Config::default();
    c.set(format!("{provider}_api_base"), "http://127.0.0.1:9/v1");
    Chat::with_config(Arc::new(c), Some(model), Some(provider), true).unwrap()
}

// spec: providers/ollama/chat_spec.rb:25 .render_payload passes #{effort} effort through as reasoning_effort
#[test]
fn ollama_passes_effort_through() {
    for effort in ["low", "medium", "high", "max", "none"] {
        let chat = local_chat("ollama", "qwen3").with_thinking(ThinkingConfig::effort(effort));
        assert_eq!(render(chat, vec![Message::user("Hello")]).unwrap()["reasoning_effort"], effort);
    }
}

// spec: providers/ollama/chat_spec.rb:32 .render_payload ignores thinking budgets with a debug note
#[test]
fn ollama_ignores_thinking_budgets_with_a_debug_note() {
    let chat = local_chat("ollama", "qwen3").with_thinking(ThinkingConfig::budget(4096));
    let (payload, logs) = logs_of(|| render(chat, vec![Message::user("Hello")]).unwrap());
    assert!(payload.get("reasoning_effort").is_none());
    assert!(logs.iter().any(|(level, msg)| *level == tracing::Level::DEBUG && msg.contains("Ollama has no thinking budgets")), "{logs:?}");
}

fn thinking_only_replay(provider: &str) {
    let payload = render(local_chat(provider, "qwen3"), vec![assistant_thinking(None, Some("I should reason first"), None)]).unwrap();
    assert_eq!(payload["messages"][0]["content"], "");
    assert_eq!(payload["messages"][0]["reasoning_content"], "I should reason first");
}

// spec: providers/ollama/chat_spec.rb:43 .format_messages includes empty content when replaying a thinking-only assistant message
#[test]
fn ollama_replays_thinking_only_messages_with_empty_content() {
    thinking_only_replay("ollama");
}

// spec: providers/ollama/media_spec.rb:7 .format_content encodes audio attachments in the released Ollama input_audio format
#[test]
fn ollama_encodes_audio_as_input_audio() {
    let wav = Attachment::from_bytes(b"wav bytes".to_vec(), "meeting.wav", None);
    let message = Message::user("Summarize this recording").with_attachments(vec![wav]);
    let payload = render(local_chat("ollama", "qwen3"), vec![message]).unwrap();
    assert_eq!(
        payload["messages"][0]["content"],
        json!([
            { "type": "text", "text": "Summarize this recording" },
            { "type": "input_audio", "input_audio": { "data": b64(b"wav bytes"), "format": "wav" } }
        ])
    );
}

// spec: providers/ollama/media_spec.rb:20 .format_content raises an actionable error for unsupported document attachments
#[test]
fn ollama_rejects_documents() {
    let message = Message::user("Summarize this file").with_attachments(vec![docx()]);
    assert_unsupported_docx(render(local_chat("ollama", "qwen3"), vec![message]));
}

// ================================================================================================
// providers/gpustack_spec.rb, providers/gpustack/media_spec.rb
// ================================================================================================

// spec: providers/gpustack_spec.rb:9 #format_messages includes empty content when replaying a thinking-only assistant message
#[test]
fn gpustack_replays_thinking_only_messages_with_empty_content() {
    thinking_only_replay("gpustack");
}

// spec: providers/gpustack_spec.rb:21 #format_content raises an actionable error for unsupported document attachments
#[test]
fn gpustack_rejects_documents() {
    let message = Message::user("Summarize this file").with_attachments(vec![docx()]);
    assert_unsupported_docx(render(local_chat("gpustack", "qwen3"), vec![message]));
}

/// A server that serves `bytes` as `content_type` at `file_path` and a chat completion elsewhere.
async fn file_server(file_path: &str, bytes: Vec<u8>, content_type: &str) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path(file_path))
        .respond_with(ResponseTemplate::new(200).set_body_raw(bytes, content_type))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({ "choices": [{ "message": { "role": "assistant", "content": "A Ruby tutorial." } }] })),
        )
        .mount(&server)
        .await;
    server
}

/// `RubyLLM::Attachment.new(url)` whose bytes were fetched, as `format_content` reads them.
async fn fetched(url: String) -> Attachment {
    let mut a = Attachment::new(url);
    a.content().await.unwrap();
    a
}

// spec: providers/gpustack_spec.rb:32 #format_content sends remote images inline, because clusters often cannot reach the internet
#[tokio::test]
async fn gpustack_inlines_remote_images() {
    let png = std::fs::read(format!("{}/tests/fixtures/ruby.png", env!("CARGO_MANIFEST_DIR"))).unwrap();
    let server = file_server("/photo.png", png.clone(), "image/png").await;
    let image = fetched(format!("{}/photo.png", server.uri())).await;
    let payload = render(chat(&server, "gpustack", "qwen3"), vec![Message::user("Describe this image").with_attachments(vec![image])]).unwrap();
    let last = payload["messages"][0]["content"].as_array().unwrap().last().cloned();
    assert_eq!(last, Some(json!({ "type": "image_url", "image_url": { "url": format!("data:image/png;base64,{}", b64(&png)), "detail": "auto" } })));
}

// spec: providers/gpustack/media_spec.rb:8 sends remote videos inline, because clusters often cannot reach the internet
#[tokio::test]
async fn gpustack_inlines_remote_videos() {
    let server = file_server("/clip.mp4", b"video bytes".to_vec(), "video/mp4").await;
    let video = fetched(format!("{}/clip.mp4", server.uri())).await;
    let payload = render(chat(&server, "gpustack", "qwen3"), vec![Message::user("Describe this clip").with_attachments(vec![video])]).unwrap();
    assert_eq!(
        payload["messages"][0]["content"],
        json!([
            { "type": "text", "text": "Describe this clip" },
            { "type": "video_url", "video_url": { "url": format!("data:video/mp4;base64,{}", b64(b"video bytes")) } }
        ])
    );
}

// spec: providers/gpustack/media_spec.rb:24 encodes local video attachments as data URLs
#[test]
fn gpustack_encodes_local_videos_as_data_urls() {
    let video = Attachment::from_bytes(b"video bytes".to_vec(), "clip.mp4", None);
    let payload = render(local_chat("gpustack", "qwen3"), vec![Message::new(Role::User, None).with_attachments(vec![video])]).unwrap();
    assert_eq!(
        payload["messages"][0]["content"],
        json!([{ "type": "video_url", "video_url": { "url": format!("data:video/mp4;base64,{}", b64(b"video bytes")) } }])
    );
}

// spec: providers/gpustack/media_spec.rb:34 accepts video attachments through the public chat API
#[tokio::test]
async fn gpustack_accepts_videos_through_ask() {
    let server = file_server("/clip.mp4", b"video bytes".to_vec(), "video/mp4").await;
    let mut chat = chat(&server, "gpustack", "qwen3");
    let response = chat.ask_with("Describe this clip", vec![Attachment::new(format!("{}/clip.mp4", server.uri()))]).await.unwrap();

    let posts: Vec<_> =
        server.received_requests().await.unwrap_or_default().into_iter().filter(|r| r.method.as_str() == "POST").collect();
    assert_eq!(posts.len(), 1);
    let body: Value = serde_json::from_slice(&posts[0].body).unwrap();
    assert_eq!(
        body["messages"],
        json!([{ "role": "user", "content": [
            { "type": "text", "text": "Describe this clip" },
            { "type": "video_url", "video_url": { "url": format!("data:video/mp4;base64,{}", b64(b"video bytes")) } }
        ]}])
    );
    assert_eq!(response.content(), "A Ruby tutorial.");
}
