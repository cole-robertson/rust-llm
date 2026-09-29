//! The remaining RubyLLM specs with recorded cassettes, replayed against the port:
//! `chat_thinking_spec.rb` (Mistral hybrid reasoning, DeepSeek thinking control, OpenRouter
//! reasoning_details round-trip, Gemini token accounting), `chat_streaming_spec.rb` (Gemini token
//! accounting in streaming), `chat_tools_spec.rb` (tool call callbacks), `chat_content_spec.rb`
//! (XLSX spreadsheets), `chat_assume_model_exists_spec.rb`, `agent_spec.rb`,
//! `error_handling_spec.rb`, and `providers/xai/speech_spec.rb`.

mod support;

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use rust_llm::attachment::AttachmentType;
use rust_llm::speech::{SpeakOptions, SpeechChunk};
use rust_llm::{Agent, Attachment, Chat, ErrorKind, Model, Parameter, ThinkingConfig, Tool, ToolCall, ToolError, ToolResult};
use serde_json::{Map, Value, json};
use support::{Cassette, config_for};

/// Tests that swap the process-wide registry or configuration hold this, so they can't see each
/// other's changes (Ruby's specs run one at a time).
static GLOBAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

fn fixture(name: &str) -> String {
    format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
}

async fn start(name: &str) -> Cassette {
    Cassette::start(name).await.unwrap_or_else(|| panic!("missing cassette {name}"))
}

fn chat_for(cassette: &Cassette, provider: &str, model: &str) -> Chat {
    Chat::with_config(config_for(cassette, provider), Some(model), Some(provider), false).expect("chat")
}

/// `RubyLLM.chat` with no model: `config.default_model` on OpenAI.
fn default_chat(cassette: &Cassette) -> Chat {
    Chat::with_config(config_for(cassette, "openai"), None, None, false).expect("chat")
}

fn arg(args: &Map<String, Value>, key: &str) -> String {
    match args.get(key) {
        Some(Value::String(s)) => s.clone(),
        Some(v) => v.to_string(),
        None => String::new(),
    }
}

// ---- tools, ported exactly: their schemas are part of the recorded request bodies ------------

struct Weather;

#[async_trait]
impl Tool for Weather {
    fn description(&self) -> String {
        "Gets current weather for a location".into()
    }
    fn parameters(&self) -> Vec<Parameter> {
        vec![
            Parameter::new("latitude").description("Latitude (e.g., 52.5200)"),
            Parameter::new("longitude").description("Longitude (e.g., 13.4050)"),
        ]
    }
    async fn execute(&self, args: Map<String, Value>, _: &ToolCall) -> Result<ToolResult, ToolError> {
        Ok(format!("Current weather at {}, {}: 15°C, Wind: 10 km/h", arg(&args, "latitude"), arg(&args, "longitude")).into())
    }
}

/// `{ roll: rand(1..6) }`.
struct DiceRoll;

#[async_trait]
impl Tool for DiceRoll {
    fn description(&self) -> String {
        "Rolls a single six-sided die and returns the result".into()
    }
    async fn execute(&self, _: Map<String, Value>, _: &ToolCall) -> Result<ToolResult, ToolError> {
        Ok(json!({ "roll": rand::random::<u8>() % 6 + 1 }).into())
    }
}

struct ReasoningWeather;

#[async_trait]
impl Tool for ReasoningWeather {
    fn description(&self) -> String {
        "Gets current weather for a city".into()
    }
    fn parameters(&self) -> Vec<Parameter> {
        vec![Parameter::new("city").description("City name (e.g., Berlin)")]
    }
    async fn execute(&self, args: Map<String, Value>, _: &ToolCall) -> Result<ToolResult, ToolError> {
        Ok(format!("Current weather in {}: 15°C, cloudy", arg(&args, "city")).into())
    }
}

// ---- chat_thinking_spec.rb: Mistral hybrid reasoning ------------------------------------------

fn mistral_chat(cassette: &Cassette) -> Chat {
    chat_for(cassette, "mistral", "mistral-small-latest").with_thinking(ThinkingConfig::effort("high"))
}

#[tokio::test]
async fn mistral_hybrid_reasoning_separates_thinking_from_final_content() {
    let cassette = start("chat_mistral_hybrid_reasoning_separates_thinking_from_final_content").await;
    let mut chat = mistral_chat(&cassette);
    let response = chat.ask("What is 12 * 12? Answer with just the number.").await.unwrap();

    let thinking = response.thinking.as_ref().and_then(|t| t.text.clone()).unwrap_or_default();
    assert!(!thinking.trim().is_empty(), "thinking {:?}", response.thinking);
    assert!(!response.content().trim().is_empty());
    assert!(!response.content().contains(&thinking));
    cassette.assert_all_matched().await;
}

#[tokio::test]
async fn mistral_hybrid_reasoning_replays_thinking_chunks_across_turns() {
    let cassette = start("chat_mistral_hybrid_reasoning_replays_thinking_chunks_across_turns").await;
    let mut chat = mistral_chat(&cassette);
    chat.ask("What is 12 * 12? Answer with just the number.").await.unwrap();

    let response = chat.ask("Now add 10 to that. Answer with just the number.").await.unwrap();

    assert!(!response.content().trim().is_empty());
    assert!(response.thinking.as_ref().and_then(|t| t.text.as_deref()).is_some_and(|t| !t.trim().is_empty()));
    cassette.assert_all_matched().await;
}

#[tokio::test]
async fn mistral_hybrid_reasoning_streams_thinking_separately_from_content() {
    let cassette = start("chat_mistral_hybrid_reasoning_streams_thinking_separately_from_content").await;
    let mut chat = mistral_chat(&cassette);
    let mut thinking_parts = Vec::new();
    let mut content_parts = Vec::new();

    let response = chat
        .ask_stream("What is 12 * 12? Answer with just the number.", |chunk| {
            if let Some(text) = chunk.thinking.as_ref().and_then(|t| t.text.clone()) {
                thinking_parts.push(text);
            }
            if !chunk.content().trim().is_empty() {
                content_parts.push(chunk.content().to_string());
            }
        })
        .await
        .unwrap();

    assert_eq!(Some(thinking_parts.join("")), response.thinking.as_ref().and_then(|t| t.text.clone()));
    assert_eq!(content_parts.join(""), response.content());
    assert!(!response.content().trim().is_empty());
    cassette.assert_all_matched().await;
}

// ---- chat_thinking_spec.rb: DeepSeek thinking control -----------------------------------------

#[tokio::test]
async fn deepseek_thinking_control_disables_thinking_for_effort_none() {
    let cassette = start("chat_deepseek_thinking_control_disables_thinking_for_effort_none").await;
    let mut chat = chat_for(&cassette, "deepseek", "deepseek-v4-flash").with_thinking(ThinkingConfig::effort("none"));

    let response = chat.ask("What is 2 + 2? Answer with just the number.").await.unwrap();

    assert!(!response.content().trim().is_empty());
    assert!(response.thinking.is_none(), "thinking {:?}", response.thinking);
    cassette.assert_all_matched().await;
}

#[tokio::test]
async fn deepseek_thinking_control_returns_reasoning_content_for_effort_high() {
    let cassette = start("chat_deepseek_thinking_control_returns_reasoning_content_for_effort_high").await;
    let mut chat = chat_for(&cassette, "deepseek", "deepseek-v4-flash").with_thinking(ThinkingConfig::effort("high"));

    let response = chat.ask("What is 2 + 2? Answer with just the number.").await.unwrap();

    assert!(!response.content().trim().is_empty());
    assert!(response.thinking.as_ref().and_then(|t| t.text.as_deref()).is_some_and(|t| !t.trim().is_empty()));
    cassette.assert_all_matched().await;
}

// ---- chat_thinking_spec.rb: OpenRouter reasoning_details round-trip ---------------------------

fn openrouter_chat(cassette: &Cassette) -> Chat {
    chat_for(cassette, "openrouter", "claude-haiku-4-5").with_thinking(ThinkingConfig::budget(2000)).with_tool(ReasoningWeather)
}

fn tool_call_raw_reasoning(chat: &Chat) -> Vec<Value> {
    let message = chat.messages().iter().find(|m| m.is_tool_call()).expect("a tool call message");
    message.raw_reasoning.as_ref().and_then(Value::as_array).cloned().unwrap_or_default()
}

#[tokio::test]
async fn openrouter_replays_reasoning_details_across_tool_calls_and_turns() {
    let cassette = start("chat_openrouter_reasoning_details_round-trip_replays_reasoning_details_across_tool_calls_and_turns").await;
    let mut chat = openrouter_chat(&cassette);

    let response = chat.ask("What is the weather in Berlin? Use the reasoning_weather tool.").await.unwrap();

    assert!(!response.content().trim().is_empty());
    let details = tool_call_raw_reasoning(&chat);
    assert!(!details.is_empty());
    assert!(details[0].get("format").is_some() && details[0].get("signature").is_some(), "{}", details[0]);

    let second = chat.ask("Should I bring an umbrella? Answer briefly.").await.unwrap();
    assert!(!second.content().trim().is_empty());
    cassette.assert_all_matched().await;
}

#[tokio::test]
async fn openrouter_accumulates_reasoning_details_while_streaming_tool_calls() {
    let cassette = start("chat_openrouter_reasoning_details_round-trip_accumulates_reasoning_details_while_streaming_tool_calls").await;
    let mut chat = openrouter_chat(&cassette);

    let response = chat.ask_stream("What is the weather in Berlin? Use the reasoning_weather tool.", |_| {}).await.unwrap();

    assert!(!response.content().trim().is_empty());
    let details = tool_call_raw_reasoning(&chat);
    assert!(!details.is_empty());
    assert!(details.iter().any(|d| d.get("signature").is_some_and(|s| !s.is_null())));
    cassette.assert_all_matched().await;
}

// ---- chat_thinking_spec.rb / chat_streaming_spec.rb: Gemini token accounting ------------------

#[tokio::test]
async fn gemini_token_accounting_sums_candidates_and_thoughts_token_count() {
    let cassette = start("chat_gemini_token_accounting_correctly_sums_candidatestokencount_and_thoughtstokencount").await;
    let mut chat = chat_for(&cassette, "gemini", "gemini-2.5-flash");
    let response = chat.ask("What is 2+2? Think step by step.").await.unwrap();

    let body = &response.raw.as_ref().expect("raw response").body;
    let candidates = body.pointer("/usageMetadata/candidatesTokenCount").and_then(Value::as_i64).unwrap_or(0);
    let thoughts = body.pointer("/usageMetadata/thoughtsTokenCount").and_then(Value::as_i64).unwrap_or(0);

    assert_eq!(response.tokens.output, Some(candidates + thoughts));
    cassette.assert_all_matched().await;
}

#[tokio::test]
async fn gemini_token_accounting_sums_candidates_and_thoughts_token_count_in_streaming() {
    let cassette = start("chat_gemini_token_accounting_correctly_sums_candidatestokencount_and_thoughtstokencount_in_streaming").await;
    let mut chat = chat_for(&cassette, "gemini", "gemini-2.5-flash");
    let mut chunks = Vec::new();

    let response = chat.ask_stream("What is 2+2? Think step by step.", |chunk| chunks.push(chunk.clone())).await.unwrap();

    let final_chunk = chunks.last().expect("chunks");
    if let Some(output) = final_chunk.tokens.output {
        assert_eq!(response.tokens.output, Some(output));
    }
    cassette.assert_all_matched().await;
}

// ---- chat_tools_spec.rb: tool call callbacks --------------------------------------------------

#[tokio::test]
async fn tool_call_callbacks_calls_before_tool_call_when_tools_are_used() {
    let cassette = start("chat_tool_call_callbacks_calls_before_tool_call_callback_when_tools_are_used").await;
    let received: Arc<Mutex<Vec<ToolCall>>> = Arc::default();
    let sink = received.clone();
    let mut chat = default_chat(&cassette).with_tool(Weather).before_tool_call(move |call| sink.lock().unwrap().push(call.clone()));

    let response = chat.ask("What's the weather in Berlin? (52.5200, 13.4050)").await.unwrap();

    let received = received.lock().unwrap();
    assert!(!received.is_empty());
    assert_eq!(received[0].name, "weather");
    assert!(!received[0].arguments().is_empty());
    assert!(response.content().contains("15"));
    assert!(response.content().contains("10"));
    cassette.assert_all_matched().await;
}

#[tokio::test]
async fn tool_call_callbacks_calls_after_tool_result_when_tools_return_results() {
    let cassette = start("chat_tool_call_callbacks_calls_after_tool_result_callback_when_tools_return_results").await;
    let received: Arc<Mutex<Vec<String>>> = Arc::default();
    let sink = received.clone();
    let mut chat = default_chat(&cassette).with_tool(Weather).after_tool_result(move |r| sink.lock().unwrap().push(r.content.clone()));

    let response = chat.ask("What's the weather in Berlin? (52.5200, 13.4050)").await.unwrap();

    let received = received.lock().unwrap();
    assert!(!received.is_empty());
    assert!(received[0].contains("15°C"));
    assert!(received[0].contains("10 km/h"));
    assert!(response.content().contains("15"));
    assert!(response.content().contains("10"));
    cassette.assert_all_matched().await;
}

#[tokio::test]
async fn tool_call_callbacks_calls_both_callbacks_in_order() {
    let cassette = start("chat_tool_call_callbacks_calls_both_before_tool_call_and_after_tool_result_callbacks_in_order").await;
    let order: Arc<Mutex<Vec<&str>>> = Arc::default();
    let (a, b) = (order.clone(), order.clone());
    let mut chat = default_chat(&cassette)
        .with_tool(DiceRoll)
        .before_tool_call(move |_| a.lock().unwrap().push("tool_call"))
        .after_tool_result(move |_| b.lock().unwrap().push("tool_result"));

    chat.ask("Roll a die for me").await.unwrap();

    assert_eq!(*order.lock().unwrap(), vec!["tool_call", "tool_result"]);
    // `rand(1..6)`: the roll the recording sent back is the one random field.
    let mismatches = cassette.mismatches.lock().unwrap().clone();
    let real: Vec<_> = mismatches.iter().filter(|m| !m.starts_with("request 1: /input/2/output:")).collect();
    assert!(real.is_empty(), "request bodies differ from RubyLLM's:\n  {real:?}");
    assert_eq!(cassette.server.received_requests().await.unwrap_or_default().len(), cassette.count);
}

// ---- chat_content_spec.rb: spreadsheet models -------------------------------------------------

#[tokio::test]
async fn openai_gpt_5_nano_understands_xlsx_spreadsheets() {
    let cassette = start("chat_spreadsheet_models_openai_gpt-5-nano_understands_xlsx_spreadsheets").await;
    let mut chat = chat_for(&cassette, "openai", "gpt-5-nano");

    let response = chat
        .ask_with(
            "What is the spreadsheet_code value in this spreadsheet? Answer with only the code.",
            vec![Attachment::new(fixture("sample.xlsx"))],
        )
        .await
        .unwrap();

    let content = response.content().to_uppercase();
    assert!(content.contains("ORCHID-97") || content.contains("ORCHID 97") || content.contains("ORCHID97"), "{content}");
    let attachment = &chat.messages()[0].attachments[0];
    assert_eq!(attachment.filename.as_deref(), Some("sample.xlsx"));
    assert_eq!(attachment.kind(), AttachmentType::Document);
    cassette.assert_all_matched().await;
}

// ---- chat_assume_model_exists_spec.rb ---------------------------------------------------------

#[tokio::test]
async fn assume_model_exists_works_with_models_not_in_registry_but_available_in_api() {
    let _guard = GLOBAL.lock().await;
    let cassette = start("chat_assume_model_exists_works_with_models_not_in_registry_but_available_in_api").await;
    let config = config_for(&cassette, "openai");
    let real_model = "gpt-4.1-nano";
    let original: Vec<Model> = rust_llm::models().all().into_iter().cloned().collect();
    rust_llm::models::Models::install(original.iter().filter(|m| m.id != real_model).cloned().collect());

    let missing = Chat::with_config(config.clone(), Some(real_model), None, false);
    let assumed = Chat::with_config(config, Some(real_model), Some("openai"), true);
    let response = match assumed {
        Ok(mut chat) => chat.ask("What is 2 + 2?").await,
        Err(e) => Err(e),
    };
    rust_llm::models::Models::install(original);

    assert!(matches!(missing.err(), Some(rust_llm::Error::ModelNotFound(_))));
    assert!(response.unwrap().content().contains('4'));
    cassette.assert_all_matched().await;
}

// ---- agent_spec.rb ----------------------------------------------------------------------------

struct SpecChatAgent;

impl Agent for SpecChatAgent {
    fn model(&self) -> Option<&str> {
        Some(support::CHAT_MODELS[0].1)
    }
    fn provider(&self) -> Option<&str> {
        Some(support::CHAT_MODELS[0].0)
    }
    fn instructions(&self) -> Option<String> {
        Some("Answer questions clearly.".into())
    }
}

#[tokio::test]
async fn agent_can_ask_using_the_first_configured_chat_model() {
    let _guard = GLOBAL.lock().await;
    let cassette = start("agent_can_ask_using_the_first_configured_chat_model").await;
    let replay = config_for(&cassette, "anthropic");
    let previous = rust_llm::config();
    rust_llm::configure(|c| *c = (*replay).clone());

    let response = match SpecChatAgent.chat() {
        Ok(mut chat) => chat.ask("What's 2 + 2?").await,
        Err(e) => Err(e),
    };
    rust_llm::configure(|c| *c = (*previous).clone());

    let response = response.unwrap();
    assert!(response.content().contains('4'));
    assert_eq!(response.role, rust_llm::Role::Assistant);
    cassette.assert_all_matched().await;
}

// ---- error_handling_spec.rb -------------------------------------------------------------------

#[tokio::test]
async fn handles_invalid_api_keys_gracefully() {
    let _guard = GLOBAL.lock().await;
    let cassette = start("error_handles_invalid_api_keys_gracefully").await;
    let mut config = (*config_for(&cassette, "openai")).clone();
    config.openai_api_key("invalid-key");
    let mut chat = Chat::with_config(Arc::new(config), Some("gpt-4.1-nano"), None, false).unwrap();

    let err = chat.ask("Hello").await.unwrap_err();

    assert_eq!(err.kind(), ErrorKind::Unauthorized, "{err}");
    cassette.assert_all_matched().await;
}

// ---- providers/xai/speech_spec.rb -------------------------------------------------------------

#[tokio::test]
async fn xai_streams_speech_and_retains_the_complete_audio_through_the_public_api() {
    let cassette = start("providers_xai_speech_streams_speech_and_retains_the_complete_audio_through_the_public_api").await;
    let mut chunks: Vec<SpeechChunk> = Vec::new();

    let speech = rust_llm::speech::speak_stream(
        "Ruby makes it easy to build useful AI applications. Stream each piece of audio as it arrives, \
         while keeping the complete recording for later.",
        SpeakOptions {
            model: Some("grok-tts"),
            provider: Some("xai"),
            voice: Some("eve"),
            format: Some("mp3"),
            config: Some(config_for(&cassette, "xai")),
            ..Default::default()
        },
        |chunk| chunks.push(chunk.clone()),
    )
    .await
    .unwrap();

    assert!(!chunks.is_empty());
    let joined: Vec<u8> = chunks.iter().flat_map(|c| c.data.iter().copied()).collect();
    assert_eq!(joined, speech.to_blob());
    assert!(speech.to_blob().len() > 1000);
    assert_eq!(speech.mime_type, "audio/mpeg");
    cassette.assert_all_matched().await;
}

// ---- chat_tools_spec.rb: concurrent tool execution --------------------------------------------

/// `ConcurrentProbeTool`: records how many calls run at once, sleeps `delay` seconds. The state
/// is `(running, max_running)`, shared with the test.
#[derive(Clone, Default)]
struct ConcurrentProbe {
    state: Arc<Mutex<(usize, usize)>>,
}

#[async_trait]
impl Tool for ConcurrentProbe {
    fn name(&self) -> String {
        "concurrent_probe".into()
    }
    fn description(&self) -> String {
        "Records concurrent execution".into()
    }
    fn parameters(&self) -> Vec<Parameter> {
        vec![Parameter::new("label"), Parameter::new("delay").kind("number").optional()]
    }
    async fn execute(&self, args: Map<String, Value>, _: &ToolCall) -> Result<ToolResult, ToolError> {
        {
            let mut s = self.state.lock().unwrap();
            s.0 += 1;
            s.1 = s.1.max(s.0);
        }
        let delay = args.get("delay").and_then(Value::as_f64).unwrap_or(0.01);
        tokio::time::sleep(std::time::Duration::from_secs_f64(delay)).await;
        self.state.lock().unwrap().0 -= 1;
        Ok(format!("finished {}", arg(&args, "label")).into())
    }
}

/// `stub_tool_response`: the model asks for the two probe calls, then answers "done".
async fn probe_server() -> wiremock::MockServer {
    use wiremock::{Mock, MockServer, ResponseTemplate, matchers};
    let tool_calls = json!({
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-haiku-4-5",
        "content": [
            { "type": "tool_use", "id": "call_1", "name": "concurrent_probe", "input": { "label": "slow", "delay": 0.05 } },
            { "type": "tool_use", "id": "call_2", "name": "concurrent_probe", "input": { "label": "fast", "delay": 0.01 } }
        ],
        "stop_reason": "tool_use", "usage": { "input_tokens": 1, "output_tokens": 1 }
    });
    let done = json!({
        "id": "msg_2", "type": "message", "role": "assistant", "model": "claude-haiku-4-5",
        "content": [{ "type": "text", "text": "done" }], "stop_reason": "end_turn", "usage": { "input_tokens": 1, "output_tokens": 1 }
    });
    let server = MockServer::start().await;
    Mock::given(matchers::method("POST")).respond_with(ResponseTemplate::new(200).set_body_json(tool_calls)).up_to_n_times(1).with_priority(1).mount(&server).await;
    Mock::given(matchers::method("POST")).respond_with(ResponseTemplate::new(200).set_body_json(done)).with_priority(2).mount(&server).await;
    server
}

fn probe_chat(server: &wiremock::MockServer, tool_concurrency: bool) -> Chat {
    let mut config = rust_llm::Config::default();
    config.set("anthropic_api_base", server.uri());
    config.set("anthropic_api_key", "test-key");
    config.max_retries = 0;
    config.tool_concurrency = tool_concurrency;
    Chat::with_config(Arc::new(config), Some("claude-haiku-4-5"), Some("anthropic"), false).expect("chat")
}

fn tool_messages(chat: &Chat) -> (Vec<String>, Vec<String>) {
    let tools: Vec<_> = chat.messages().iter().filter(|m| m.role == rust_llm::Role::Tool).collect();
    (tools.iter().filter_map(|m| m.tool_call_id.clone()).collect(), tools.iter().map(|m| m.content().to_string()).collect())
}

#[tokio::test]
async fn executes_multiple_tool_calls_concurrently() {
    let server = probe_server().await;
    let probe = ConcurrentProbe::default();
    let mut chat = probe_chat(&server, false).with_tool(probe.clone()).with_tool_concurrency(true);
    chat.ask("Run the tools").await.unwrap();

    assert_eq!(probe.state.lock().unwrap().1, 2, "both calls ran at once");
    assert!(chat.concurrency());
    let (ids, contents) = tool_messages(&chat);
    assert_eq!(ids, ["call_2", "call_1"], "results land as each call finishes");
    assert_eq!(contents, ["finished fast", "finished slow"]);
}

#[tokio::test]
async fn config_tool_concurrency_runs_calls_concurrently() {
    let server = probe_server().await;
    let probe = ConcurrentProbe::default();
    let mut chat = probe_chat(&server, true).with_tool(probe.clone());
    assert!(chat.concurrency(), "the chat starts from config.tool_concurrency");
    chat.ask("Run the tools").await.unwrap();
    assert_eq!(probe.state.lock().unwrap().1, 2);
    assert_eq!(tool_messages(&chat).0, ["call_2", "call_1"]);
}

#[tokio::test]
async fn sequential_tool_calls_run_one_at_a_time_in_call_order() {
    let server = probe_server().await;
    let probe = ConcurrentProbe::default();
    let mut chat = probe_chat(&server, false).with_tool(probe.clone());
    chat.ask("Run the tools").await.unwrap();
    assert_eq!(probe.state.lock().unwrap().1, 1);
    assert_eq!(tool_messages(&chat).0, ["call_1", "call_2"]);
}

#[tokio::test]
async fn adds_concurrent_tool_result_messages_as_each_call_finishes_before_resuming_the_model() {
    let server = probe_server().await;
    let events: Arc<Mutex<Vec<String>>> = Arc::default();
    let sink = events.clone();
    let mut chat = probe_chat(&server, false)
        .with_tool(ConcurrentProbe::default())
        .with_tool_concurrency(true)
        .after_message(move |m| {
            if let Some(id) = &m.tool_call_id {
                sink.lock().unwrap().push(format!("tool_message {id}"));
            }
        });
    chat.ask("Run the tools").await.unwrap();

    // The follow-up request carries both results, in the order they were added.
    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 2);
    let body: Value = serde_json::from_slice(&requests[1].body).unwrap();
    let results: Vec<&str> = body["messages"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|m| m["content"].as_array().into_iter().flatten())
        .filter(|b| b["type"] == "tool_result")
        .filter_map(|b| b["tool_use_id"].as_str())
        .collect();
    assert_eq!(*events.lock().unwrap(), ["tool_message call_2", "tool_message call_1"]);
    assert_eq!(results, ["call_2", "call_1"]);
}

// ---- chat.rb#render: before_request hooks apply ----------------------------------------------

#[tokio::test]
async fn render_applies_before_request_hooks() {
    let server = probe_server().await;
    let mut chat = probe_chat(&server, false).before_request(|payload| payload["metadata"] = json!({ "user_id": "u-1" }));
    chat.ask_later("Hello").unwrap();
    assert_eq!(chat.render().unwrap()["metadata"], json!({ "user_id": "u-1" }));
    // A request applies them once, as before.
    chat.complete().await.unwrap();
    let requests = server.received_requests().await.unwrap();
    let body: Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(body["metadata"], json!({ "user_id": "u-1" }));
}
