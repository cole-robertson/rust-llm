//! `spec/ruby_llm/protocols/interactions_spec.rb` and `protocols/interactions/tools_spec.rb`:
//! Gemini's Interactions protocol, its live examples replayed from RubyLLM's cassettes and its
//! stubbed examples run through the real render/parse path. `// spec:` lines tie each test to Ruby.

mod support;

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use rust_llm::message::indexmap_lite::IndexMap;
use rust_llm::protocols::interactions;
use rust_llm::{
    Attachment, Chat, Config, Error, Message, Model, Parameter, ProtocolName, ProviderTool, Role, ThinkingConfig,
    ThinkingDisplay, Tool, ToolCall, ToolChoice, ToolError, ToolResult,
};
use serde_json::{Map, Value, json};
use support::{Cassette, config_for};

const MODEL: &str = "gemini-3.8-flash";

/// The spec's anonymous `multiply` tool.
struct Multiply;

#[async_trait]
impl Tool for Multiply {
    fn name(&self) -> String {
        "multiply".into()
    }
    fn description(&self) -> String {
        "Multiply two numbers".into()
    }
    fn parameters(&self) -> Vec<Parameter> {
        vec![Parameter::new("left").kind("integer"), Parameter::new("right").kind("integer")]
    }
    async fn execute(&self, args: Map<String, Value>, _: &ToolCall) -> Result<ToolResult, ToolError> {
        Ok(json!(args["left"].as_i64().unwrap_or(0) * args["right"].as_i64().unwrap_or(0)).into())
    }
}

fn offline_config() -> Arc<Config> {
    let mut c = Config::default();
    c.set("gemini_api_key", "test");
    Arc::new(c)
}

/// `RubyLLM.chat(model: model_for(:gemini, :mcp), provider: :gemini, protocol: :interactions)`.
fn chat_with(config: Arc<Config>) -> Chat {
    Chat::with_config(config, Some(MODEL), Some("gemini"), false).unwrap().with_protocol(ProtocolName::Interactions)
}

fn chat() -> Chat {
    chat_with(offline_config())
}

fn model() -> Model {
    chat().model().clone()
}

fn steps() -> Value {
    json!([
        { "type": "mcp_server_tool_call", "id": "remote1", "name": "docs:search",
          "server_name": "docs", "arguments": { "query": "Ruby" }, "signature": "call-signature" },
        { "type": "mcp_server_tool_result", "call_id": "remote1", "name": "docs:search",
          "server_name": "docs", "result": "Ruby documentation", "signature": "result-signature" },
        { "type": "thought", "signature": "thought-signature" },
        { "type": "model_output", "content": [{ "type": "text", "text": "Ruby docs." }] }
    ])
}

fn body() -> Value {
    json!({
        "object": "interaction", "id": "interaction1", "status": "completed", "model": MODEL,
        "steps": steps(), "usage": { "total_input_tokens": 20, "total_tool_use_tokens": 30,
                                     "total_output_tokens": 5, "total_thought_tokens": 10,
                                     "total_cached_tokens": 4 }
    })
}

fn parse(data: &Value) -> rust_llm::Result<Message> {
    interactions::parse_completion_body(&model(), data, None)
}

fn without_signature(step: &Value) -> Value {
    let mut step = step.clone();
    step.as_object_mut().unwrap().remove("signature");
    step
}

/// `JSON.parse(JSON.generate(message.to_h))`: a serialized round trip. The port's `to_h` omits
/// `raw_content`, so the restored message keeps the fields that replay needs, as Ruby's does.
fn round_trip(message: &Message) -> Message {
    let mut restored = Message::new(message.role, message.content.clone());
    restored.tool_calls = message.tool_calls.clone();
    restored.tool_call_id = message.tool_call_id.clone();
    restored.thinking = message.thinking.clone();
    restored.raw_content = message.raw_content.as_ref().map(|r| serde_json::from_str(&r.to_string()).unwrap());
    restored
}

// spec: protocols/interactions_spec.rb:36
#[test]
fn keeps_generate_content_as_the_default_and_renders_the_opt_in_protocol_through_chat() {
    assert_eq!(rust_llm::Provider::Gemini.default_protocol(), ProtocolName::Gemini);
    let mut chat = chat()
        .with_instructions("Be concise")
        .with_tool(Multiply)
        .with_tool_choice(ToolChoice::Required)
        .unwrap()
        .with_provider_tools([ProviderTool::with_options("mcp", json!({ "name": "docs", "url": "https://example.com/mcp" }))])
        .with_max_output_tokens(50);
    chat.ask_later("Multiply").unwrap();
    let payload = chat.render().unwrap();
    assert_eq!(payload["store"], json!(false));
    assert_eq!(payload["system_instruction"], json!("Be concise"));
    assert_eq!(payload["input"], json!([{ "type": "user_input", "content": [{ "type": "text", "text": "Multiply" }] }]));
    assert_eq!(payload["generation_config"]["max_output_tokens"], json!(50));
    assert_eq!(payload["generation_config"]["tool_choice"], json!("any"));
    let tools = payload["tools"].as_array().unwrap();
    assert_eq!((tools[0]["type"].as_str(), tools[0]["name"].as_str()), (Some("function"), Some("multiply")));
    let last = tools.last().unwrap();
    assert_eq!(
        (last["type"].as_str(), last["name"].as_str(), last["url"].as_str()),
        (Some("mcp_server"), Some("docs"), Some("https://example.com/mcp"))
    );
}

// spec: protocols/interactions_spec.rb:48
#[test]
fn normalizes_complete_mcp_output_and_preserves_every_signed_step_for_stateless_replay() {
    let message = parse(&body()).unwrap();
    assert_eq!(message.content(), "Ruby docs.");
    assert!(!message.is_tool_call());
    let kinds: Vec<&str> = message.server_tool_calls.iter().map(|c| c.kind.as_str()).collect();
    assert_eq!(kinds, ["mcp_server_tool_call", "mcp_server_tool_result"]);
    assert_eq!(message.server_tool_calls.last().unwrap().result, Some(json!("Ruby documentation")));
    let t = &message.tokens;
    assert_eq!((t.input, t.output, t.thinking, t.cache_read), (Some(46), Some(15), Some(10), Some(4)));
    let mut chat = chat();
    chat.set_messages(vec![round_trip(&message)]);
    chat.ask_later("Continue").unwrap();
    let input = chat.render().unwrap()["input"].as_array().unwrap().clone();
    let steps = steps();
    let steps = steps.as_array().unwrap();
    assert_eq!(input[..2], [without_signature(&steps[0]), without_signature(&steps[1])]);
    assert_eq!(input[2]["signature"], json!("thought-signature"));
    assert_eq!(message.raw_content.as_ref().unwrap().pointer("/response/steps"), Some(&Value::Array(steps.clone())));
}

// spec: protocols/interactions_spec.rb:63
#[test]
fn keeps_local_function_calls_separate_from_remote_executions() {
    let mut data = body();
    data["status"] = json!("requires_action");
    data["steps"].as_array_mut().unwrap().push(json!({
        "type": "function_call", "id": "local1", "name": "multiply",
        "arguments": { "left": 2, "right": 3 }, "signature": "local-signature"
    }));
    let message = parse(&data).unwrap();
    let calls = message.tool_calls.as_ref().unwrap();
    assert_eq!(calls.keys().collect::<Vec<_>>(), ["local1"]);
    let call = calls.get("local1").unwrap();
    assert!(!call.remote);
    assert_eq!(call.thought_signature.as_deref(), Some("local-signature"));
}

// spec: protocols/interactions_spec.rb:73
#[test]
fn includes_the_function_name_with_a_local_result() {
    let mut chat = chat().with_tool(Multiply);
    let mut calls = IndexMap::new();
    calls.insert("local1".into(), ToolCall::new("local1", "multiply", Map::new()));
    let mut call = Message::new(Role::Assistant, None);
    call.tool_calls = Some(calls);
    chat.add_message(call);
    chat.add_message(Message::tool_result("local1", "91"));
    let payload = chat.render().unwrap();
    let last = payload["input"].as_array().unwrap().last().unwrap().clone();
    assert_eq!((last["type"].as_str(), last["call_id"].as_str(), last["name"].as_str()), (Some("function_result"), Some("local1"), Some("multiply")));
}

// spec: protocols/interactions_spec.rb:82
#[test]
fn replays_edited_history_and_signed_results_after_serialization_without_a_remote_cursor() {
    let mut chat = chat();
    chat.ask_later("Remember violet").unwrap();
    chat.add_message(parse(&body()).unwrap());
    let mut restored = chat_with(offline_config());
    let mut messages: Vec<Message> = chat.messages().iter().map(round_trip).collect();
    messages[0].content = Some("Remember orange".into());
    restored.set_messages(messages);
    restored.ask_later("What color?").unwrap();
    let payload = restored.render().unwrap();
    assert_eq!(payload["store"], json!(false));
    assert!(payload.get("previous_interaction_id").is_none());
    let input = payload["input"].as_array().unwrap();
    assert_eq!(input[0], json!({ "type": "user_input", "content": [{ "type": "text", "text": "Remember orange" }] }));
    let steps = steps();
    assert!(input.contains(&without_signature(&steps[0])));
    assert!(input.contains(&steps[2]));
    assert_eq!(input.last().unwrap(), &json!({ "type": "user_input", "content": [{ "type": "text", "text": "What color?" }] }));
}

// spec: protocols/interactions_spec.rb:98
#[tokio::test]
async fn replays_a_local_function_call_and_its_named_result_without_provider_storage() {
    let mut data = body();
    data["status"] = json!("requires_action");
    data["steps"] = json!([{ "type": "function_call", "id": "local1", "name": "multiply", "arguments": { "left": 13, "right": 7 } }]);
    let mut chat = chat().with_tool(Multiply);
    chat.ask_later("Multiply").unwrap();
    chat.add_message(parse(&data).unwrap());
    chat.run_tools().await.unwrap();
    let payload = chat.render().unwrap();
    assert_eq!(payload["store"], json!(false));
    assert!(payload.get("previous_interaction_id").is_none());
    let input = payload["input"].as_array().unwrap();
    assert!(input.contains(&data["steps"][0]));
    assert_eq!(
        input.last().unwrap(),
        &json!({ "type": "function_result", "call_id": "local1", "name": "multiply", "result": [{ "type": "text", "text": "91" }] })
    );
}

// spec: protocols/interactions_spec.rb:114
#[test]
fn renders_json_schema_and_specific_tool_choice_in_the_documented_fields() {
    let schema = json!({ "type": "object", "properties": { "answer": { "type": "integer" } }, "required": ["answer"] });
    let mut chat = chat().with_schema(schema).with_tool(Multiply).with_tool_choice(ToolChoice::Tool("multiply".into())).unwrap();
    chat.ask_later("Multiply").unwrap();
    let payload = chat.render().unwrap();
    assert_eq!(payload["response_format"]["type"], json!("text"));
    assert_eq!(payload["response_format"]["mime_type"], json!("application/json"));
    assert_eq!(payload["response_format"]["schema"]["type"], json!("object"));
    assert_eq!(payload["generation_config"]["tool_choice"], json!({ "allowed_tools": { "mode": "any", "tools": ["multiply"] } }));
}

// spec: protocols/interactions_spec.rb:123
#[test]
fn converts_citation_byte_offsets_to_characters_across_multiple_output_parts() {
    let mut data = body();
    data["steps"] = json!([{ "type": "model_output", "content": [
        { "type": "text", "text": "First. " },
        { "type": "text", "text": "Café Ruby", "annotations": [
            { "type": "url_citation", "url": "https://www.ruby-lang.org", "start_index": 6, "end_index": 10 },
            { "type": "file_citation", "file_name": "Manual", "document_uri": "gs://docs/manual.pdf",
              "page_number": 2, "start_index": 0, "end_index": 5 }
        ] }
    ] }]);
    let message = parse(&data).unwrap();
    let first = &message.citations[0];
    assert_eq!((first.start_index, first.end_index, first.text.as_deref()), (Some(12), Some(16), Some("Ruby")));
    let last = message.citations.last().unwrap();
    assert_eq!((last.title.as_deref(), last.start_page, last.end_page, last.text.as_deref()), (Some("Manual"), Some(2), Some(2), Some("Café")));
}

// spec: protocols/interactions_spec.rb:139
#[test]
fn separates_thinking_effort_from_summary_display_and_rejects_an_unsupported_off_control() {
    for effort in ["minimal", "low", "medium", "high"] {
        let mut chat = chat().with_thinking(ThinkingConfig::effort(effort).with_display(ThinkingDisplay::Summarized));
        chat.ask_later("Think").unwrap();
        let config = chat.render().unwrap()["generation_config"].clone();
        assert_eq!((config["thinking_level"].as_str(), config["thinking_summaries"].as_str()), (Some(effort), Some("auto")));
    }
    let config = chat().with_thinking(ThinkingConfig::effort("low").with_display(ThinkingDisplay::Omitted)).render().unwrap()["generation_config"].clone();
    assert_eq!((config["thinking_level"].as_str(), config["thinking_summaries"].as_str()), (Some("low"), Some("none")));
    let mut off = ThinkingConfig::default();
    off.enabled = Some(false);
    for config in [off, ThinkingConfig::effort("none"), ThinkingConfig::budget(0)] {
        let err = interactions::render_interaction_thinking(&config).unwrap_err();
        assert!(matches!(&err, Error::Argument(m) if m.contains("thinking-off")), "{err:?}");
    }
    let message = |t: ThinkingConfig| match chat().with_thinking(t).render() {
        Err(Error::Argument(m)) => m,
        other => panic!("expected an ArgumentError, got {other:?}"),
    };
    assert!(message(ThinkingConfig::effort("xhigh")).contains("effort must be"));
    assert!(message(ThinkingConfig::budget(1024)).contains("not a token budget"));
    assert!(message(ThinkingConfig::default().with_display(ThinkingDisplay::Full)).contains("display must be"));
}

// spec: protocols/interactions_spec.rb:155
#[tokio::test]
async fn renders_image_and_pdf_attachments_as_content() {
    let fixtures = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
    let mut chat = chat();
    let mut attachments = vec![Attachment::new(format!("{fixtures}/ruby.png")), Attachment::new(format!("{fixtures}/sample.pdf"))];
    for a in &mut attachments {
        a.content().await.unwrap();
    }
    chat.ask_later_with("Read", attachments).unwrap();
    let payload = chat.render().unwrap();
    let parts = payload["input"][0]["content"].as_array().unwrap();
    let types: Vec<&str> = parts.iter().map(|p| p["type"].as_str().unwrap()).collect();
    assert_eq!(types, ["text", "image", "document"]);
    assert_eq!(parts[2]["mime_type"], json!("application/pdf"));
    assert!(parts[2]["data"].as_str().unwrap().starts_with("JVBER"));
}

/// Serves `events` as one Interactions stream and runs a streamed chat against it.
async fn stream(events: Vec<Value>) -> (rust_llm::Result<Message>, Vec<Message>) {
    let body: String = events.iter().map(|e| format!("data: {e}\n\n")).collect();
    let server = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::any())
        .respond_with(wiremock::ResponseTemplate::new(200).set_body_raw(body, "text/event-stream"))
        .mount(&server)
        .await;
    let mut config = Config::default();
    config.set("gemini_api_key", "test");
    config.set("gemini_api_base", server.uri());
    config.max_retries = 0;
    let mut chat = chat_with(Arc::new(config));
    let chunks = Arc::new(Mutex::new(Vec::new()));
    let sink = chunks.clone();
    // One request, like the spec's direct `stream_response` call: a tool call is not run.
    chat.ask_later("Go").unwrap();
    let result = chat.step_stream(move |c| sink.lock().unwrap().push(c.clone())).await.map(|m| m.expect("a response"));
    let chunks = chunks.lock().unwrap().clone();
    (result, chunks)
}

// spec: protocols/interactions_spec.rb:162
#[tokio::test]
async fn accumulates_streamed_mcp_steps_and_never_yields_remote_result_text_as_assistant_text() {
    let body = body();
    let mut created = body.clone();
    created.as_object_mut().unwrap().remove("steps");
    created.as_object_mut().unwrap().remove("usage");
    let mut events = vec![json!({ "event_type": "interaction.created", "interaction": created })];
    for (index, step) in steps().as_array().unwrap().iter().enumerate() {
        let mut start = step.clone();
        start.as_object_mut().unwrap().remove("content");
        start.as_object_mut().unwrap().remove("result");
        events.push(json!({ "event_type": "step.start", "index": index, "step": start }));
        let delta = if step["type"] == "model_output" { json!({ "type": "text", "text": "Ruby docs." }) } else { step.clone() };
        events.push(json!({ "event_type": "step.delta", "index": index, "delta": delta }));
    }
    let mut completed = body.clone();
    completed.as_object_mut().unwrap().remove("steps");
    events.push(json!({ "event_type": "interaction.completed", "interaction": completed }));
    let (message, chunks) = stream(events).await;
    let message = message.unwrap();
    assert_eq!(chunks.iter().filter_map(|c| c.content.clone()).collect::<String>(), "Ruby docs.");
    assert_eq!(message.server_tool_calls.last().unwrap().result, Some(json!("Ruby documentation")));
    assert_eq!(message.raw_content.as_ref().unwrap().pointer("/response/steps"), Some(&steps()));
    let last = &chunks.last().unwrap().tokens;
    assert_eq!((last.input, last.output), (Some(46), Some(15)));
}

// spec: protocols/interactions_spec.rb:182
#[tokio::test]
async fn rejects_failed_or_truncated_streams_and_unsupported_required_actions() {
    let (truncated, _) = stream(vec![json!({ "event_type": "interaction.created", "interaction": { "status": "in_progress" } })]).await;
    assert!(matches!(&truncated, Err(Error::Api(m, _)) if m.contains("ended before")), "{truncated:?}");
    let mut state = interactions::StreamState::default();
    let failed = interactions::build_chunk(&model(), &mut state, &json!({ "event_type": "error", "error": { "message": "Failed" } }));
    assert!(matches!(&failed, Err(Error::Api(m, _)) if m == "Failed"), "{failed:?}");
    let mut data = body();
    data["status"] = json!("requires_action");
    assert!(matches!(parse(&data), Err(Error::Api(m, _)) if m.contains("unsupported action")));
}

// spec: protocols/interactions_spec.rb:191
#[tokio::test]
async fn accumulates_local_function_argument_deltas_after_an_empty_object_in_the_initial_step() {
    let mut completed = body();
    completed.as_object_mut().unwrap().remove("steps");
    completed["status"] = json!("requires_action");
    let events = vec![
        json!({ "event_type": "step.start", "index": 0,
                "step": { "type": "function_call", "id": "local1", "name": "multiply", "arguments": {} } }),
        json!({ "event_type": "step.delta", "index": 0, "delta": { "type": "arguments_delta", "arguments": "{\"left\":17," } }),
        json!({ "event_type": "step.delta", "index": 0, "delta": { "type": "arguments_delta", "arguments": "\"right\":19}" } }),
        json!({ "event_type": "interaction.completed", "interaction": completed }),
    ];
    let message = stream(events).await.0.unwrap();
    assert_eq!(Value::Object(message.tool_calls.as_ref().unwrap().get("local1").unwrap().arguments()), json!({ "left": 17, "right": 19 }));
    let mut chat = chat();
    chat.add_message(message);
    assert_eq!(chat.render().unwrap()["input"].as_array().unwrap().last().unwrap()["arguments"], json!({ "left": 17, "right": 19 }));
}

// ---- live examples, replayed --------------------------------------------------------------------

async fn replay(name: &str) -> (Cassette, Chat) {
    let cassette = Cassette::start(&format!("protocols_interactions_{name}")).await.unwrap_or_else(|| panic!("missing cassette {name}"));
    let chat = chat_with(config_for(&cassette, "gemini"));
    (cassette, chat)
}

const MCP_PROMPT: &str = "Use the Microsoft Learn MCP search tool to find the Azure Functions overview. Reply briefly.";

fn microsoft_learn() -> ProviderTool {
    ProviderTool::with_options("mcp", json!({ "name": "microsoft_learn", "url": "https://learn.microsoft.com/api/mcp" }))
}

// spec: protocols/interactions_spec.rb:222
#[tokio::test]
async fn executes_a_remote_mcp_tool_and_replays_its_signed_results_through_stateless_chat() {
    let (cassette, chat) = replay("executes_a_remote_mcp_tool_and_replays_its_signed_results_through_stateless_chat").await;
    let mut chat = chat.with_provider_tools([microsoft_learn()]);
    let message = chat.ask(MCP_PROMPT).await.unwrap();
    assert!(message.server_tool_calls.iter().any(|c| c.kind == "mcp_server_tool_call"));
    assert!(message.server_tool_calls.iter().any(|c| c.kind == "mcp_server_tool_result"));
    assert!(!message.is_tool_call());
    assert!(message.content().to_lowercase().contains("functions"), "{:?}", message.content);
    let followup = chat.ask("What Microsoft service did you look up? Answer using the previous results.").await.unwrap();
    assert!(followup.content().to_lowercase().contains("azure functions"), "{:?}", followup.content);
    cassette.assert_all_matched().await;
}

// spec: protocols/interactions_spec.rb:233
#[tokio::test]
async fn streams_remote_mcp_results_and_preserves_the_complete_signed_history() {
    let (cassette, chat) = replay("streams_remote_mcp_results_and_preserves_the_complete_signed_history").await;
    let mut chat = chat.with_provider_tools([microsoft_learn()]);
    let mut text = String::new();
    let message = chat.ask_stream(MCP_PROMPT, |c| text.push_str(c.content())).await.unwrap();
    assert_eq!(text, message.content());
    assert!(message.server_tool_calls.iter().any(|c| c.kind == "mcp_server_tool_result"));
    assert!(message.tokens().input.is_some_and(|i| i > 0));
    let steps = message.raw_content.as_ref().unwrap().pointer("/response/steps").unwrap().as_array().unwrap().clone();
    assert!(steps.iter().any(|s| s.get("signature").is_some()));
    cassette.assert_all_matched().await;
}

// spec: protocols/interactions_spec.rb:246
#[tokio::test]
async fn executes_local_tools_and_returns_json_schema_output_through_interactions() {
    let (cassette, chat) = replay("executes_local_tools_and_returns_json_schema_output_through_interactions").await;
    let schema = json!({ "type": "object", "properties": { "answer": { "type": "integer" } }, "required": ["answer"] });
    let mut chat = chat.with_tool(Multiply).with_schema(schema);
    let message = chat.ask("Use multiply to calculate 13 times 7 and return the answer.").await.unwrap();
    assert_eq!(message.parsed().unwrap(), Some(json!({ "answer": 91 })));
    assert!(chat.messages().iter().filter(|m| m.is_tool_result()).any(|m| m.content() == "91"));
    cassette.assert_all_matched().await;
}

// spec: protocols/interactions_spec.rb:254
#[tokio::test]
async fn streams_local_function_arguments_and_continues_with_the_actual_tool_result() {
    let (cassette, chat) = replay("streams_local_function_arguments_and_continues_with_the_actual_tool_result").await;
    let mut chat = chat.with_tool(Multiply);
    let response = chat.ask_stream("Use multiply to calculate 17 times 19. State the result.", |_| {}).await.unwrap();
    assert!(response.content().contains("323"), "{:?}", response.content);
    assert!(chat.messages().iter().filter(|m| m.is_tool_result()).any(|m| m.content() == "323"));
    cassette.assert_all_matched().await;
}

// ---- protocols/interactions/tools_spec.rb -------------------------------------------------------

fn call(arguments: Option<Value>) -> rust_llm::Result<ToolCall> {
    let mut step = json!({ "type": "function_call", "id": "c1", "name": "now" });
    if let Some(arguments) = arguments {
        step["arguments"] = arguments;
    }
    Ok(interactions::parse_interaction_calls(&[step])?.remove(0))
}

// spec: protocols/interactions/tools_spec.rb:13
#[test]
fn parses_a_json_object_string() {
    assert_eq!(Value::Object(call(Some(json!("{\"tz\":\"UTC\"}"))).unwrap().arguments()), json!({ "tz": "UTC" }));
}

// spec: protocols/interactions/tools_spec.rb:17
#[test]
fn returns_empty_for_empty_string_arguments() {
    assert!(call(Some(json!(""))).unwrap().arguments().is_empty());
}

// spec: protocols/interactions/tools_spec.rb:21
#[test]
fn returns_empty_for_a_missing_arguments_key() {
    assert!(call(None).unwrap().arguments().is_empty());
}

// spec: protocols/interactions/tools_spec.rb:25
#[test]
fn passes_an_object_through_unchanged() {
    assert_eq!(Value::Object(call(Some(json!({ "tz": "UTC" }))).unwrap().arguments()), json!({ "tz": "UTC" }));
}

// spec: protocols/interactions/tools_spec.rb:29
#[test]
fn wraps_malformed_json_in_a_tool_call_parse_error() {
    let err = call(Some(json!("{\"tz\":"))).unwrap_err();
    assert!(matches!(err, Error::ToolCallParse { .. }), "{err:?}");
}
