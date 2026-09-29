//! `spec/ruby_llm/protocols/perplexity/router_spec.rb`: Perplexity Router's Chat Completions
//! dialect (`protocol: :router_chat_completions`). The live example replays its cassette; the rest
//! stub the Router URL with wiremock or only render.

mod support;

use std::sync::Arc;

use async_trait::async_trait;
use rust_llm::{Attachment, Chat, Config, Error, Parameter, ProtocolName, Provider, Tool, ToolCall, ToolCalls, ToolChoice, ToolError, ToolResult};
use serde_json::{Map, Value, json};
use support::{Cassette, config_for};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// `model_for(:perplexity, :router)`.
const MODEL: &str = "perplexity/kimi-k3";

struct Add;

#[async_trait]
impl Tool for Add {
    fn name(&self) -> String {
        "add".into()
    }
    fn description(&self) -> String {
        "Add two integers.".into()
    }
    fn parameters(&self) -> Vec<Parameter> {
        vec![Parameter::new("left").kind("integer"), Parameter::new("right").kind("integer")]
    }
    async fn execute(&self, args: Map<String, Value>, _: &ToolCall) -> Result<ToolResult, ToolError> {
        Ok(json!(args["left"].as_i64().unwrap_or(0) + args["right"].as_i64().unwrap_or(0)).into())
    }
}

/// A tool declared without a description (`Class.new(RubyLLM::Tool) { ... }`).
struct Undescribed;

#[async_trait]
impl Tool for Undescribed {
    fn name(&self) -> String {
        "undescribed".into()
    }
    fn description(&self) -> String {
        String::new()
    }
    async fn execute(&self, _: Map<String, Value>, _: &ToolCall) -> Result<ToolResult, ToolError> {
        Ok(json!(null).into())
    }
}

fn config(base: Option<&str>) -> Arc<Config> {
    let mut c = Config::default();
    c.set("perplexity_api_key", "test");
    if let Some(base) = base {
        c.set("perplexity_api_base", base);
    }
    c.max_retries = 0;
    Arc::new(c)
}

fn router(config: Arc<Config>) -> Chat {
    Chat::with_config(config, Some(MODEL), Some("perplexity"), false).unwrap().with_protocol(ProtocolName::RouterChatCompletions)
}

fn completion(message: Value, finish_reason: &str) -> Value {
    json!({ "id": "reply", "model": MODEL, "choices": [{ "index": 0, "message": message, "finish_reason": finish_reason }],
            "usage": { "prompt_tokens": 100, "completion_tokens": 8,
                       "prompt_tokens_details": { "cached_tokens": 30, "cache_write_tokens": 20 } } })
}

fn answer() -> Value {
    completion(json!({ "role": "assistant", "content": "4" }), "stop")
}

// spec: protocols/perplexity/router_spec.rb:27
#[tokio::test]
async fn selects_router_explicitly_while_preserving_sonar_and_configured_gateway_base_paths() {
    // Chat runs on the Agent API by default (`protocol_for`); the Router is only explicit.
    assert_ne!(Provider::Perplexity.default_protocol(), ProtocolName::RouterChatCompletions);
    for (base, expected) in [
        (None, "/router/v1/chat/completions"),
        (Some("/perplexity"), "/perplexity/router/v1/chat/completions"),
        (Some("/perplexity/router/v1/"), "/perplexity/router/v1/chat/completions"),
    ] {
        let server = MockServer::start().await;
        Mock::given(method("POST")).and(path(expected)).respond_with(ResponseTemplate::new(200).set_body_json(answer())).expect(1).mount(&server).await;
        let base = format!("{}{}", server.uri(), base.unwrap_or(""));
        router(config(Some(&base))).ask("Hi").await.unwrap();
    }
}

// spec: protocols/perplexity/router_spec.rb:37
#[test]
fn renders_required_and_named_tool_choices_parallel_controls_schema_and_cache_boundaries_through_chat() {
    let mut chat = router(config(None))
        .with_tool(Add)
        .with_tool_choice(ToolChoice::Required)
        .unwrap()
        .with_tool_calls(ToolCalls::One)
        .with_caching(json!(true))
        .unwrap()
        .with_instructions("Use the documented arithmetic tools.");
    chat.cache_until_here().unwrap();
    chat.ask_later("Add two and two.").unwrap();
    let payload = chat.render().unwrap();
    assert_eq!(payload["tool_choice"], json!("required"));
    assert_eq!(payload["parallel_tool_calls"], json!(false));
    assert!(payload.get("prompt_cache_options").is_none());
    let first = payload["messages"][0]["content"].as_array().unwrap().last().unwrap().clone();
    assert_eq!(first["prompt_cache_breakpoint"], json!({ "mode": "explicit" }));
    let chat = chat.with_tool_choice(ToolChoice::Tool("add".into())).unwrap().with_tool_calls(ToolCalls::Many);
    let payload = chat.render().unwrap();
    assert_eq!(payload["tool_choice"], json!({ "type": "function", "function": { "name": "add" } }));
    assert_eq!(payload["parallel_tool_calls"], json!(true));
    let schema = json!({ "type": "object", "properties": { "answer": { "type": "integer" } }, "required": ["answer"] });
    let chat = chat.with_schema(schema);
    assert_eq!(chat.render().unwrap()["response_format"]["json_schema"]["strict"], json!(true));
    let chat = chat.with_caching(json!(false)).unwrap();
    let payload = chat.render().unwrap();
    assert!(payload.get("prompt_cache_options").is_none());
    assert_eq!(payload["messages"][0]["content"], json!("Use the documented arithmetic tools."));
}

// spec: protocols/perplexity/router_spec.rb:53
#[tokio::test]
async fn executes_and_replays_local_tool_results_with_reasoning_context_and_normalized_cache_usage() {
    let call = json!({ "role": "assistant", "content": null, "reasoning_content": "Add the values.",
                       "tool_calls": [{ "id": "call_add", "type": "function",
                                        "function": { "name": "add", "arguments": "{\"left\":2,\"right\":2}" } }] });
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/router/v1/chat/completions"))
        .respond_with(support_sequence(vec![completion(call, "tool_calls"), answer()]))
        .mount(&server)
        .await;
    let mut chat = router(config(Some(&server.uri()))).with_tool(Add);
    let result = chat.ask("Add two and two.").await.unwrap();
    assert_eq!(result.content(), "4");
    let t = &result.tokens;
    assert_eq!((t.input, t.output, t.cache_read, t.cache_write), (Some(50), Some(8), Some(30), Some(20)));
    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 2);
    let last: Value = serde_json::from_slice(&requests[1].body).unwrap();
    let messages = last["messages"].as_array().unwrap();
    assert!(messages.iter().any(|m| m["role"] == "tool" && m["tool_call_id"] == "call_add" && m["content"] == "4"));
    assert!(messages.iter().any(|m| m["role"] == "assistant" && m["reasoning_content"] == "Add the values."));
}

struct Sequence(std::sync::Mutex<std::collections::VecDeque<Value>>);

impl wiremock::Respond for Sequence {
    fn respond(&self, _: &wiremock::Request) -> ResponseTemplate {
        match self.0.lock().unwrap().pop_front() {
            Some(body) => ResponseTemplate::new(200).set_body_json(body),
            None => ResponseTemplate::new(599),
        }
    }
}

fn support_sequence(bodies: Vec<Value>) -> Sequence {
    Sequence(std::sync::Mutex::new(bodies.into()))
}

// spec: protocols/perplexity/router_spec.rb:73
#[tokio::test]
async fn streams_actual_router_shaped_deltas_and_final_usage_through_the_public_chat_api() {
    let mut usage_only = answer();
    usage_only["choices"] = json!([]);
    let events = [
        json!({ "choices": [{ "index": 0, "delta": { "role": "assistant", "content": "Hello" } }] }),
        json!({ "choices": [{ "index": 0, "delta": {}, "finish_reason": "stop" }] }),
        usage_only,
    ];
    let body: String = events.iter().map(|e| format!("data: {e}\n\n")).chain(["data: [DONE]\n\n".to_string()]).collect();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/router/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(body, "text/event-stream"))
        .expect(1)
        .mount(&server)
        .await;
    let mut chat = router(config(Some(&server.uri())));
    let mut text = String::new();
    let result = chat.ask_stream("Say hello.", |c| text.push_str(c.content())).await.unwrap();
    assert_eq!(text, "Hello");
    assert_eq!(result.content(), "Hello");
    let t = result.tokens();
    assert_eq!((t.input, t.output, t.cache_read, t.cache_write), (Some(50), Some(8), Some(30), Some(20)));
    let sent: Value = serde_json::from_slice(&server.received_requests().await.unwrap()[0].body).unwrap();
    assert_eq!(sent["stream_options"]["include_usage"], json!(true));
}

// spec: protocols/perplexity/router_spec.rb:92
#[tokio::test]
async fn serializes_documented_wav_audio_input_while_keeping_sonar_audio_unsupported() {
    let wav = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/ruby.wav");
    let mut audio = Attachment::new(wav);
    audio.content().await.unwrap();
    let mut chat = router(config(None));
    chat.ask_later_with("Transcribe this audio.", vec![audio.clone()]).unwrap();
    let payload = chat.render().unwrap();
    let part = payload["messages"].as_array().unwrap().last().unwrap()["content"].as_array().unwrap().last().unwrap().clone();
    assert_eq!(part["type"], json!("input_audio"));
    assert_eq!(part["input_audio"]["format"], json!("wav"));
    let decoded = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, part["input_audio"]["data"].as_str().unwrap()).unwrap();
    assert_eq!(decoded, std::fs::read(wav).unwrap());
    let mut sonar = Chat::with_config(config(None), Some("sonar"), Some("perplexity"), false).unwrap().with_protocol(ProtocolName::ChatCompletions);
    sonar.ask_later_with("Transcribe this audio.", vec![audio]).unwrap();
    assert!(matches!(sonar.render(), Err(Error::UnsupportedAttachment(_))));
}

// spec: protocols/perplexity/router_spec.rb:104
#[tokio::test]
async fn rejects_explicitly_unsupported_request_controls_before_making_a_request() {
    let server = MockServer::start().await;
    for options in [
        json!({ "seed": 1 }),
        json!({ "modalities": ["audio"] }),
        json!({ "n": 2 }),
        json!({ "presence_penalty": 1 }),
        json!({ "stream_options": { "include_obfuscation": true } }),
    ] {
        let mut chat = router(config(Some(&server.uri()))).with_provider_options(options.clone());
        chat.ask_later("Hello").unwrap();
        let err = chat.render().unwrap_err();
        assert!(matches!(&err, Error::Argument(m) if m.contains("Perplexity Router does not support")), "{options}: {err:?}");
    }
    assert!(server.received_requests().await.unwrap().is_empty());
}

// spec: protocols/perplexity/router_spec.rb:113
#[test]
fn requires_tool_descriptions_and_strict_schemas_without_changing_the_default_protocol() {
    let mut chat = router(config(None)).with_tool(Undescribed);
    chat.ask_later("Hello").unwrap();
    assert!(matches!(chat.render(), Err(Error::Argument(m)) if m.contains("require a description")));
    let schema = json!({ "name": "answer", "schema": { "type": "object", "properties": {} }, "strict": false });
    let err = router(config(None)).with_schema(schema).render().unwrap_err();
    assert!(matches!(&err, Error::Argument(m) if m.contains("strict structured output")), "{err:?}");
}

// spec: protocols/perplexity/router_spec.rb:122
#[tokio::test]
async fn requests_a_named_tool_with_a_cache_boundary_from_the_configured_router_account() {
    let name = "protocols_perplexity_router_requests_a_named_tool_with_a_cache_boundary_from_the_configured_router_account";
    let cassette = Cassette::start(name).await.unwrap();
    let mut chat = router(config_for(&cassette, "perplexity"))
        .with_tool(Add)
        .with_tool_choice(ToolChoice::Tool("add".into()))
        .unwrap()
        .with_tool_calls(ToolCalls::One)
        .with_caching(json!(true))
        .unwrap()
        .with_instructions("Use arithmetic tools for arithmetic.");
    chat.cache_until_here().unwrap();
    chat.ask_later("Add two and two.").unwrap();
    // The recorded account had no Router preview access: Ruby skips on this ForbiddenError, and the
    // request it sent is what this replay checks.
    let err = chat.generate().await.unwrap_err();
    assert!(matches!(&err, Error::Forbidden(m, _) if m.contains("Router API is currently in limited preview")), "{err:?}");
    cassette.assert_all_matched().await;
}
