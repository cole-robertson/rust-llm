//! RubyLLM 2.1's cross-model replay rules (`spec/ruby_llm/chat_thinking_replay_spec.rb`,
//! `chat_model_switch_replay_spec.rb`), the request history a crash leaves
//! (`chat_unfinished_history_spec.rb`), and the Sonnet 5.5 thinking-off control
//! (`thinking/controls_spec.rb`). `// spec:` lines tie each test to its Ruby example.

mod spec_helpers;

use std::sync::{Arc, Mutex};

use rust_llm::message::{Operation, indexmap_lite::IndexMap};
use rust_llm::{
    Chat, Config, Cost, FinishReason, FnTool, Message, Model, ProtocolName, Role, Thinking,
    ThinkingConfig, Tokens, ToolCall, ToolResult, UsageEntry, UsageStatus,
};
use serde_json::{Map, Value, json};
use spec_helpers::{config, serve};

fn configured() -> Arc<Config> {
    let mut config = Config::default();
    for provider in ["openai", "anthropic", "gemini", "deepseek"] {
        config.set(format!("{provider}_api_key"), "test-key");
    }
    Arc::new(config)
}

fn chat(model: &str, provider: &str) -> Chat {
    Chat::with_config(configured(), Some(model), Some(provider), false).expect("chat")
}

fn entry(provider: &str, model: &str) -> UsageEntry {
    UsageEntry {
        id: UsageEntry::next_id(),
        owner: None,
        operation: Operation::Chat,
        provider: provider.into(),
        model: model.into(),
        status: UsageStatus::Succeeded,
        tokens: Tokens::default(),
        cost: Cost::default(),
    }
}

/// `produced_by(provider, model, thinking, **attributes)`.
fn produced_by(provider: &str, model: &str, thinking: Option<Thinking>) -> Message {
    let mut message = Message::assistant("Done.");
    message.model = Some(model.into());
    message.thinking = thinking;
    message.usage_entries = vec![entry(provider, model)];
    message
}

/// `replay(chat, message)`: Hi, the message, "And now?", rendered.
fn replay(chat: &mut Chat, message: Message) -> Value {
    chat.add_message(Message::user("Hi"));
    chat.add_message(message);
    chat.add_message(Message::user("And now?"));
    chat.render().expect("render")
}

fn signature(sig: &str) -> Option<Thinking> {
    Thinking::build(None, Some(sig.into()))
}

fn calls(list: &[ToolCall]) -> Option<IndexMap<ToolCall>> {
    let mut map = IndexMap::new();
    for call in list {
        map.insert(call.id.clone(), call.clone());
    }
    Some(map)
}

// ---- chat_thinking_replay_spec.rb ----------------------------------------------------------------

// spec: chat_thinking_replay_spec.rb:59 drops the interaction a Gemini answer carries when the chat moves to Anthropic
#[test]
fn drops_the_interaction_a_gemini_answer_carries_when_the_chat_moves_to_anthropic() {
    let state = json!({ "response": { "object": "interaction", "status": "completed",
        "steps": [{ "type": "model_output", "content": [{ "type": "text", "text": "Done." }] }] } });
    let mut message = produced_by("gemini", "gemini-3.8-flash", None);
    message.raw_content = Some(state);
    let mut chat = chat("gemini-3.8-flash", "gemini")
        .with_protocol(ProtocolName::Interactions)
        .with_model("claude-haiku-4-5", Some("anthropic"))
        .unwrap();
    let payload = replay(&mut chat, message);
    assert_eq!(
        payload["messages"][1],
        json!({ "role": "assistant", "content": [{ "type": "text", "text": "Done." }] })
    );
}

// spec: chat_thinking_replay_spec.rb:71 sends generateContent the answer of an Interactions turn, not the interaction
#[test]
fn sends_generate_content_the_answer_of_an_interactions_turn_not_the_interaction() {
    let state =
        json!({ "response": { "object": "interaction", "status": "completed", "steps": [] } });
    let mut message = produced_by("gemini", "gemini-3.8-flash", None);
    message.raw_content = Some(state);
    let mut chat = chat("gemini-3.8-flash", "gemini")
        .with_protocol(ProtocolName::Interactions)
        .with_model("gemini-3.8-flash", Some("gemini"))
        .unwrap();
    let payload = replay(&mut chat, message);
    assert_eq!(
        payload["contents"][1],
        json!({ "role": "model", "parts": [{ "text": "Done." }] })
    );
}

// spec: chat_thinking_replay_spec.rb:81 drops Anthropic server tool blocks when the chat moves to OpenAI
#[test]
fn drops_anthropic_server_tool_blocks_when_the_chat_moves_to_openai() {
    let blocks = json!([
        { "type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": { "query": "Ruby" } },
        { "type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": [] },
        { "type": "text", "text": "Done." }
    ]);
    let mut message = produced_by("anthropic", "claude-haiku-4-5", None);
    message.raw_content = Some(blocks);
    let mut chat = chat("claude-haiku-4-5", "anthropic")
        .with_model("gpt-5-nano", Some("openai"))
        .unwrap();
    let payload = replay(&mut chat, message);
    assert_eq!(
        payload["input"][1],
        json!({ "role": "assistant", "content": [{ "type": "output_text", "text": "Done." }] })
    );
}

// spec: chat_thinking_replay_spec.rb:95 keeps the server tool blocks of the provider that produced them
#[test]
fn keeps_the_server_tool_blocks_of_the_provider_that_produced_them() {
    let blocks = json!([
        { "type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {} },
        { "type": "text", "text": "Done." }
    ]);
    let mut message = produced_by("anthropic", "claude-haiku-4-5", None);
    message.raw_content = Some(blocks.clone());
    let payload = replay(&mut chat("claude-haiku-4-5", "anthropic"), message);
    assert_eq!(
        payload["messages"][1],
        json!({ "role": "assistant", "content": blocks })
    );
}

// spec: chat_thinking_replay_spec.rb:121 signs a call another provider made with the placeholder Gemini documents
#[test]
fn signs_a_call_another_provider_made_with_the_placeholder_gemini_documents() {
    let mut message = produced_by(
        "openai",
        "gpt-5-nano",
        signature("openai-encrypted-content"),
    );
    message.tool_calls = calls(&[ToolCall::new("call-1", "lookup", Map::new())]);
    let mut chat = chat("gpt-5-nano", "openai");
    chat.add_message(Message::user("Hi"));
    chat.add_message(message);
    chat.add_message(Message::tool_result("call-1", "Found it."));
    let payload = chat
        .with_model("gemini-2.5-flash", Some("gemini"))
        .unwrap()
        .render()
        .unwrap();
    assert_eq!(
        payload["contents"][1]["parts"],
        json!([{ "text": "Done." },
               { "functionCall": { "name": "lookup", "args": {} },
                 "thoughtSignature": "skip_thought_signature_validator" }])
    );
}

// spec: chat_thinking_replay_spec.rb:138 sends Gemini no answer signature whose producer is unknown
#[test]
fn sends_gemini_no_answer_signature_whose_producer_is_unknown() {
    let mut message = Message::assistant("Done.");
    message.model = Some("gpt-5-nano".into());
    message.thinking = signature("openai-encrypted-content");
    let payload = replay(&mut chat("gemini-2.5-flash", "gemini"), message);
    assert_eq!(
        payload["contents"][1]["parts"],
        json!([{ "text": "Done." }])
    );
}

// spec: chat_thinking_replay_spec.rb:148 sends Gemini back the answer signature it produced
#[test]
fn sends_gemini_back_the_answer_signature_it_produced() {
    let message = produced_by("gemini", "gemini-2.5-flash", signature("gemini-signature"));
    let payload = replay(&mut chat("gemini-2.5-flash", "gemini"), message);
    assert_eq!(
        payload["contents"][1]["parts"],
        json!([{ "text": "Done.", "thoughtSignature": "gemini-signature" }])
    );
}

// spec: chat_thinking_replay_spec.rb:157 sends the Responses API no reasoning whose producer is unknown
#[test]
fn sends_the_responses_api_no_reasoning_whose_producer_is_unknown() {
    let mut message = Message::assistant("Done.");
    message.model = Some("claude-haiku-4-5".into());
    message.thinking = Thinking::build(Some("Adding.".into()), Some("anthropic-signature".into()));
    let payload = replay(&mut chat("gpt-5-nano", "openai"), message);
    assert_eq!(
        payload["input"][1],
        json!({ "role": "assistant", "content": [{ "type": "output_text", "text": "Done." }] })
    );
}

// spec: chat_thinking_replay_spec.rb:167 sends the Responses API back the reasoning it produced
#[test]
fn sends_the_responses_api_back_the_reasoning_it_produced() {
    let message = produced_by(
        "openai",
        "gpt-5-nano",
        signature("openai-encrypted-content"),
    );
    let payload = replay(&mut chat("gpt-5-nano", "openai"), message);
    assert_eq!(
        payload["input"][1],
        json!({ "type": "reasoning", "summary": [], "encrypted_content": "openai-encrypted-content" })
    );
}

// spec: chat_thinking_replay_spec.rb:188 sends Claude no thinking whose producer is unknown
#[test]
fn sends_claude_no_thinking_whose_producer_is_unknown() {
    let mut message = Message::assistant("Done.");
    message.thinking = Thinking::build(Some("Adding.".into()), Some("gemini-signature".into()));
    let payload = replay(&mut chat("claude-haiku-4-5", "anthropic"), message);
    assert_eq!(
        payload["messages"][1],
        json!({ "role": "assistant", "content": [{ "type": "text", "text": "Done." }] })
    );
}

// spec: chat_thinking_replay_spec.rb:198 keeps the thinking blocks of a Claude answer whose producer is unknown
#[test]
fn keeps_the_thinking_blocks_of_a_claude_answer_whose_producer_is_unknown() {
    let block = json!({ "type": "thinking", "thinking": "Let me think.", "signature": "anthropic-signature" });
    let mut message = Message::assistant("Done.");
    message.raw_reasoning = Some(json!({ "anthropic": [block.clone()] }));
    let payload = replay(&mut chat("claude-haiku-4-5", "anthropic"), message);
    assert_eq!(payload["messages"][1]["content"][0], block);
}

// ---- chat_model_switch_replay_spec.rb ------------------------------------------------------------
// The Ruby spec switches between two Bedrock models (Nova and Claude), and Bedrock is not ported.
// The rule it checks, native content replaying only to the model that produced it, is the same on
// every provider; two Anthropic models stand in here, with raw Anthropic thinking blocks.

fn produced_by_model(model: &str) -> Message {
    let mut message = produced_by(
        "anthropic",
        model,
        Thinking::build(Some("Checking.".into()), None),
    );
    message.raw_reasoning = Some(json!({ "anthropic": [
        { "type": "thinking", "thinking": "Checking.", "signature": "sig" }
    ] }));
    message
}

// spec: chat_model_switch_replay_spec.rb:26 replays the reasoning of another model on the same provider without it
#[test]
fn replays_the_reasoning_of_another_model_on_the_same_provider_without_it() {
    let payload = replay(
        &mut chat("claude-sonnet-4-5", "anthropic"),
        produced_by_model("claude-haiku-4-5"),
    );
    assert_eq!(
        payload["messages"][1]["content"],
        json!([{ "type": "text", "text": "Done." }])
    );
}

// spec: chat_model_switch_replay_spec.rb:32 replays reasoning to the model that produced it
#[test]
fn replays_reasoning_to_the_model_that_produced_it() {
    let payload = replay(
        &mut chat("claude-haiku-4-5", "anthropic"),
        produced_by_model("claude-haiku-4-5"),
    );
    assert_eq!(
        payload["messages"][1]["content"],
        json!([{ "type": "thinking", "thinking": "Checking.", "signature": "sig" },
               { "type": "text", "text": "Done." }])
    );
}

// ---- chat_unfinished_history_spec.rb -------------------------------------------------------------

const UNFINISHED: &str = r#"{"error":"The tool call did not finish."}"#;

type Runs = Arc<Mutex<Vec<&'static str>>>;

fn lookup_tool(runs: Runs) -> FnTool {
    FnTool::new("lookup", "Look something up", move |_| {
        let runs = runs.clone();
        async move {
            runs.lock().unwrap().push("lookup");
            Ok(ToolResult::from(json!("found")))
        }
    })
}

fn approval_tool() -> FnTool {
    FnTool::new("dangerous", "Do something dangerous", |_| async {
        Ok(ToolResult::from(json!("done")))
    })
    .requires_approval()
}

/// `round(*calls)`: an assistant message with empty content calling `ids`.
fn round(ids: &[(&str, &str)]) -> Message {
    let mut m = Message::new(Role::Assistant, Some(String::new()));
    let list: Vec<ToolCall> = ids
        .iter()
        .map(|(id, name)| ToolCall::new(*id, *name, Map::new()))
        .collect();
    m.tool_calls = calls(&list);
    m
}

fn result(id: &str) -> Message {
    Message::tool_result(id, "found")
}

fn placeholder() -> Message {
    Message::new(Role::Assistant, Some(String::new()))
}

struct Harness {
    chat: Chat,
    server: wiremock::MockServer,
    runs: Runs,
}

/// `chat_with(*history)`: the lookup and approval tools and `history`, with the provider answering
/// "Answered". The Ruby spec stubs `provider.complete` on an OpenAI chat; here a mock Anthropic
/// server answers, and `sent` reads the messages back from the payload it received.
async fn chat_with(history: Vec<Message>) -> Harness {
    let answered = spec_helpers::text_response("Answered");
    let server = serve(vec![answered.clone(), answered]).await;
    let runs: Runs = Arc::new(Mutex::new(Vec::new()));
    let mut chat = Chat::with_config(
        config(&server),
        Some(spec_helpers::MODEL),
        Some("anthropic"),
        false,
    )
    .unwrap()
    .with_tool(lookup_tool(runs.clone()))
    .with_tool(approval_tool());
    for m in history {
        chat.add_message(m);
    }
    Harness { chat, server, runs }
}

/// `sent(requests.last)`: `[role, content, tool_call_id]` of each message of the last request.
async fn sent(server: &wiremock::MockServer) -> Vec<Vec<String>> {
    let requests = server.received_requests().await.unwrap();
    let body: Value = serde_json::from_slice(&requests.last().unwrap().body).unwrap();
    let mut out = Vec::new();
    for message in body["messages"].as_array().unwrap() {
        let role = message["role"].as_str().unwrap();
        let blocks: Vec<Value> = match &message["content"] {
            Value::String(s) => vec![json!({ "type": "text", "text": s })],
            Value::Array(b) => b.clone(),
            _ => Vec::new(),
        };
        if role == "assistant" {
            let text: String = blocks.iter().filter_map(|b| b["text"].as_str()).collect();
            out.push(vec!["assistant".into(), text]);
            continue;
        }
        for block in blocks {
            if block["type"] == "tool_result" {
                out.push(vec![
                    "tool".into(),
                    block["content"][0]["text"].as_str().unwrap_or("").into(),
                    block["tool_use_id"].as_str().unwrap_or("").into(),
                ]);
            } else {
                out.push(vec![
                    "user".into(),
                    block["text"].as_str().unwrap_or("").into(),
                ]);
            }
        }
    }
    out
}

fn row(parts: &[&str]) -> Vec<String> {
    parts.iter().map(|p| p.to_string()).collect()
}

// spec: chat_unfinished_history_spec.rb:65 leaves blank assistant messages out of a request
#[tokio::test]
async fn leaves_blank_assistant_messages_out_of_a_request() {
    let mut h = chat_with(vec![Message::user("Hi"), placeholder()]).await;
    h.chat.ask("Still there?").await.unwrap();
    assert_eq!(
        sent(&h.server).await,
        [row(&["user", "Hi"]), row(&["user", "Still there?"])]
    );
    let roles: Vec<Role> = h.chat.messages().iter().map(|m| m.role).collect();
    assert_eq!(
        roles,
        [Role::User, Role::Assistant, Role::User, Role::Assistant]
    );
}

// spec: chat_unfinished_history_spec.rb:74 answers a call the conversation moved past as unfinished
#[tokio::test]
async fn answers_a_call_the_conversation_moved_past_as_unfinished() {
    let mut h = chat_with(vec![
        Message::user("Look twice"),
        round(&[("a", "lookup"), ("b", "lookup")]),
        result("a"),
        placeholder(),
        Message::user("Hello?"),
    ])
    .await;
    h.chat.complete().await.unwrap();
    assert_eq!(
        sent(&h.server).await,
        [
            row(&["user", "Look twice"]),
            row(&["assistant", ""]),
            row(&["tool", "found", "a"]),
            row(&["tool", UNFINISHED, "b"]),
            row(&["user", "Hello?"]),
        ]
    );
    let results = h.chat.messages().iter().filter(|m| m.is_tool_result());
    assert_eq!(results.count(), 1);
    assert!(h.runs.lock().unwrap().is_empty());
}

// spec: chat_unfinished_history_spec.rb:86 runs the calls a crash left unfinished in the last round when the chat resumes
#[tokio::test]
async fn runs_the_calls_a_crash_left_unfinished_in_the_last_round_when_the_chat_resumes() {
    let mut h = chat_with(vec![
        Message::user("Look twice"),
        round(&[("a", "lookup"), ("b", "lookup")]),
        result("a"),
        placeholder(),
    ])
    .await;
    assert!(!h.chat.is_complete());
    let Err(err) = h.chat.ask_later("Hello?") else {
        panic!("pending calls")
    };
    assert!(
        matches!(&err, rust_llm::Error::PendingToolCalls(m) if m.contains("lookup")),
        "{err:?}"
    );
    assert_eq!(h.chat.complete().await.unwrap().content(), "Answered");
    assert_eq!(*h.runs.lock().unwrap(), ["lookup"]);
    assert_eq!(
        sent(&h.server).await,
        [
            row(&["user", "Look twice"]),
            row(&["assistant", ""]),
            row(&["tool", "found", "a"]),
            row(&["tool", "found", "b"]),
        ]
    );
}

// spec: chat_unfinished_history_spec.rb:98 leaves a call that waits on approval unanswered
#[tokio::test]
async fn leaves_a_call_that_waits_on_approval_unanswered() {
    let mut h = chat_with(vec![Message::user("Do it"), round(&[("a", "dangerous")])]).await;
    let last = h.chat.complete().await.unwrap();
    assert_eq!(&last, h.chat.messages().last().unwrap());
    assert!(h.chat.is_awaiting_approval());
    h.chat.generate().await.unwrap();
    assert_eq!(
        sent(&h.server).await,
        [row(&["user", "Do it"]), row(&["assistant", ""])]
    );
}

// spec: chat_unfinished_history_spec.rb:109 leaves a call that waits on input or a task unanswered
#[tokio::test]
async fn leaves_a_call_that_waits_on_input_or_a_task_unanswered() {
    let mut h = chat_with(vec![
        Message::user("Deploy"),
        round(&[("a", "lookup"), ("b", "lookup")]),
    ])
    .await;
    h.chat.set_tool_call_inputs([
        (
            "a".to_string(),
            json!({ "requests": [{ "key": "environment", "response": null }] }),
        ),
        ("b".to_string(), json!({ "task": { "taskId": "task-1" } })),
    ]);
    assert!(h.chat.is_waiting());
    h.chat.generate().await.unwrap();
    assert_eq!(
        sent(&h.server).await,
        [row(&["user", "Deploy"]), row(&["assistant", ""])]
    );
}

// spec: chat_unfinished_history_spec.rb:122 answers a call that waited on approval as unfinished once the conversation moved past it
#[tokio::test]
async fn answers_a_call_that_waited_on_approval_as_unfinished_once_the_conversation_moved_past_it()
{
    let mut h = chat_with(vec![
        Message::user("Do it"),
        round(&[("a", "dangerous")]),
        placeholder(),
        Message::user("Never mind"),
    ])
    .await;
    h.chat.complete().await.unwrap();
    assert_eq!(
        sent(&h.server).await,
        [
            row(&["user", "Do it"]),
            row(&["assistant", ""]),
            row(&["tool", UNFINISHED, "a"]),
            row(&["user", "Never mind"]),
        ]
    );
}

// spec: chat_unfinished_history_spec.rb:131 leaves the calls of the latest round for the loop to run
#[tokio::test]
async fn leaves_the_calls_of_the_latest_round_for_the_loop_to_run() {
    let mut h = chat_with(vec![Message::user("Look"), round(&[("a", "lookup")])]).await;
    h.chat.generate().await.unwrap();
    assert_eq!(
        sent(&h.server).await,
        [row(&["user", "Look"]), row(&["assistant", ""])]
    );
    assert!(h.runs.lock().unwrap().is_empty());
}

/// `describe 'a provider-executed call the conversation moved past'`.
fn remote_round_chat() -> (Chat, Value) {
    let request_item = json!({ "type": "mcp_approval_request", "id": "mcpr_1", "name": "search",
                               "arguments": "{}", "server_label": "docs" });
    let mut chat = chat("gpt-5-nano", "openai").with_protocol(ProtocolName::Responses);
    let mut remote = ToolCall::new("mcpr_1", "search", Map::new());
    remote.remote = true;
    let mut answer = Message::new(Role::Assistant, Some(String::new()));
    answer.tool_calls = calls(&[remote]);
    answer.raw_content = Some(json!([request_item.clone()]));
    chat.add_message(Message::user("Search the docs"));
    chat.add_message(answer);
    chat.add_message(Message::user("Never mind"));
    (chat, request_item)
}

// spec: chat_unfinished_history_spec.rb:155 a provider-executed call the conversation moved past > is refused the way the provider expects
#[test]
fn a_remote_call_the_conversation_moved_past_is_refused_the_way_the_provider_expects() {
    let (chat, request_item) = remote_round_chat();
    assert_eq!(
        chat.render().unwrap()["input"],
        json!([{ "role": "user", "content": "Search the docs" },
               request_item,
               { "type": "mcp_approval_response", "approval_request_id": "mcpr_1", "approve": false },
               { "role": "user", "content": "Never mind" }])
    );
}

// spec: chat_unfinished_history_spec.rb:164 a provider-executed call the conversation moved past > is answered as unfinished after a move to another provider
#[test]
fn a_remote_call_the_conversation_moved_past_is_answered_as_unfinished_after_a_move() {
    let (chat, _) = remote_round_chat();
    let payload = chat
        .with_model("claude-haiku-4-5", Some("anthropic"))
        .unwrap()
        .render()
        .unwrap();
    assert_eq!(
        payload["messages"][2],
        json!({ "role": "user", "content": [{ "type": "tool_result", "tool_use_id": "mcpr_1",
                                               "content": [{ "type": "text", "text": UNFINISHED }] }] })
    );
}

// spec: chat_unfinished_history_spec.rb:174 keeps an empty answer from the model as the end of the conversation
#[tokio::test]
async fn keeps_an_empty_answer_from_the_model_as_the_end_of_the_conversation() {
    let mut empty = Message::new(Role::Assistant, Some(String::new()));
    empty.finish_reason = Some(FinishReason::Stop);
    empty.tokens.output = Some(0);
    let mut h = chat_with(vec![Message::user("Hi"), empty]).await;
    assert!(h.chat.is_complete());
    h.chat.complete().await.unwrap();
    assert_eq!(spec_helpers::requests(&h.server).await, 0);
}

// ---- thinking/controls_spec.rb -------------------------------------------------------------------

/// `model_for(id, provider:, reasoning_options:)`.
fn model_with(id: &str, provider: &str, reasoning_options: Value) -> Model {
    let mut model = Model::default_for(id, provider);
    model
        .metadata
        .insert("reasoning_options".into(), reasoning_options);
    model
}

fn sonnet_efforts() -> Value {
    json!([{ "type": "effort", "values": ["low", "medium", "high", "xhigh", "max"] }])
}

// spec: thinking/controls_spec.rb:15 #disable > uses the Anthropic between_tools off control for Sonnet 5.5
#[test]
fn disable_uses_the_anthropic_between_tools_off_control_for_sonnet_5_5() {
    let model = model_with("claude-sonnet-5-5", "anthropic", sonnet_efforts());
    let resolved = ThinkingConfig::off().resolve(&model).unwrap().unwrap();
    assert_eq!(
        (resolved.enabled, resolved.effort, resolved.budget),
        (Some(false), None, None)
    );
}

// spec: thinking/controls_spec.rb:56 #disable > still raises when neither the registry nor the provider exposes an off control
#[test]
fn disable_still_raises_without_an_off_control() {
    let model = model_with("magistral-medium-latest", "mistral", json!([]));
    let err = ThinkingConfig::off().resolve(&model).unwrap_err();
    assert!(
        err.to_string()
            .contains("does not know how to disable thinking"),
        "{err}"
    );
}
