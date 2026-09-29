//! Shared setup for the `spec_*.rs` ports of RubyLLM's non-cassette unit specs. The Ruby specs
//! stub `provider.complete` to return canned messages; here a mock server answers with canned
//! provider responses in order, so the port's real render/parse path runs in between.

#![allow(dead_code)]

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use rust_llm::message::indexmap_lite::IndexMap;
use rust_llm::{Chat, Config, Message, Role, ToolCall};
use serde_json::{Map, Value, json};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

/// `model_for(:anthropic)` in `spec/support/models_to_test.rb`.
pub const MODEL: &str = "claude-haiku-4-5";

/// An Anthropic Messages response carrying `text` (the specs' `answer_message`).
pub fn text_response(text: &str) -> Value {
    json!({
        "id": "msg_1", "type": "message", "role": "assistant", "model": MODEL,
        "content": [{ "type": "text", "text": text }], "stop_reason": "end_turn",
        "usage": { "input_tokens": 1, "output_tokens": 1 }
    })
}

/// An Anthropic Messages response asking for `calls` (`[(id, name, arguments)]`).
pub fn tool_use_response(calls: &[(&str, &str, Value)]) -> Value {
    let content: Vec<Value> =
        calls.iter().map(|(id, name, input)| json!({ "type": "tool_use", "id": id, "name": name, "input": input })).collect();
    json!({
        "id": "msg_1", "type": "message", "role": "assistant", "model": MODEL,
        "content": content, "stop_reason": "tool_use",
        "usage": { "input_tokens": 1, "output_tokens": 1 }
    })
}

/// Serves `responses` in order; anything past the end is answered with 599 and counted.
pub struct Sequence(pub Mutex<VecDeque<ResponseTemplate>>);

impl Respond for Sequence {
    fn respond(&self, _request: &Request) -> ResponseTemplate {
        self.0
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| ResponseTemplate::new(599).set_body_string("no stub left"))
    }
}

/// A server that answers every request with the next of `responses` (JSON bodies, status 200).
pub async fn serve(responses: Vec<Value>) -> MockServer {
    serve_templates(
        responses
            .into_iter()
            .map(|r| ResponseTemplate::new(200).set_body_json(r))
            .collect(),
    )
    .await
}

pub async fn serve_templates(responses: Vec<ResponseTemplate>) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(wiremock::matchers::any())
        .respond_with(Sequence(Mutex::new(responses.into())))
        .mount(&server)
        .await;
    server
}

pub async fn requests(server: &MockServer) -> usize {
    server.received_requests().await.unwrap_or_default().len()
}

/// `include_context 'with configured RubyLLM'`: every provider pointed at `server`, no retries.
pub fn config(server: &MockServer) -> Arc<Config> {
    let mut c = Config::default();
    for (provider, path) in [
        ("anthropic", ""),
        ("openai", "/v1"),
        ("deepseek", ""),
        ("gemini", "/v1beta"),
        ("openrouter", "/api/v1"),
    ] {
        c.set(
            format!("{provider}_api_base"),
            format!("{}{path}", server.uri()),
        );
        c.set(format!("{provider}_api_key"), "test");
    }
    c.max_retries = 0;
    Arc::new(c)
}

/// `RubyLLM::Chat.new(model: model_for(:anthropic))` against `server`.
pub fn chat(server: &MockServer) -> Chat {
    Chat::with_config(config(server), Some(MODEL), Some("anthropic"), false).expect("chat")
}

pub fn args(value: Value) -> Map<String, Value> {
    value.as_object().cloned().unwrap_or_default()
}

/// An assistant message calling `calls`, like the specs' `tool_call_message`.
pub fn tool_call_message(calls: &[(&str, &str, Value)]) -> Message {
    let mut m = Message::new(Role::Assistant, Some(String::new()));
    let calls: IndexMap<ToolCall> = calls
        .iter()
        .map(|(id, name, a)| (id.to_string(), ToolCall::new(*id, *name, args(a.clone()))))
        .collect();
    m.tool_calls = Some(calls);
    m
}

/// `RubyLLM::Message.new(role: :assistant, content:, model:, input_tokens: 1, output_tokens: 1)`.
pub fn answer_message(text: &str) -> Message {
    let mut m = Message::assistant(text);
    m.model = Some(MODEL.into());
    m.tokens.input = Some(1);
    m.tokens.output = Some(1);
    m
}

/// Server-sent events for an Anthropic stream that yields `pieces` as text deltas.
pub fn text_stream(pieces: &[&str]) -> String {
    let mut body = String::from(
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"model\":\"claude-haiku-4-5\",\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n",
    );
    for piece in pieces {
        body.push_str(&format!(
            "event: content_block_delta\ndata: {}\n\n",
            json!({ "type": "content_block_delta", "index": 0, "delta": { "type": "text_delta", "text": piece } })
        ));
    }
    body.push_str("event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n\n");
    body.push_str("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n");
    body
}

pub fn sse(body: String) -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_raw(body.into_bytes(), "text/event-stream")
}

/// Shared call log for tools and callbacks.
pub type Log<T> = Arc<Mutex<Vec<T>>>;

pub fn log<T>() -> Log<T> {
    Arc::new(Mutex::new(Vec::new()))
}
