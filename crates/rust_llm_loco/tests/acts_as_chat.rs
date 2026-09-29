//! `acts_as_chat` behavior on SQLite: a tool-calling conversation replayed from RubyLLM's
//! Anthropic cassette is written row-for-row, and reloads into an equivalent chat.

use std::sync::Arc;

use async_trait::async_trait;
use rust_llm::{Parameter, Role, Tool, ToolCall, ToolError, ToolResult};
use rust_llm_loco::entities::{rust_llm_tool_calls, rust_llm_usages};
use rust_llm_loco::{ChatRecord, migrations};
use sea_orm::{Database, DatabaseConnection, EntityTrait};
use sea_orm_migration::SchemaManager;
use serde_json::{Map, Value};
use wiremock::{Mock, MockServer, ResponseTemplate, matchers};

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
        Ok(format!(
            "Current weather at {}, {}: 15°C, Wind: 10 km/h",
            args["latitude"].as_str().unwrap_or_default(),
            args["longitude"].as_str().unwrap_or_default()
        )
        .into())
    }
}

struct DeleteEverything;

#[async_trait]
impl Tool for DeleteEverything {
    fn description(&self) -> String {
        "Deletes everything".into()
    }
    fn requires_approval(&self) -> bool {
        true
    }
    async fn execute(&self, _: Map<String, Value>, _: &ToolCall) -> Result<ToolResult, ToolError> {
        Ok("deleted".into())
    }
}

async fn db() -> DatabaseConnection {
    let db = Database::connect("sqlite::memory:").await.unwrap();
    let manager = SchemaManager::new(&db);
    for m in migrations() {
        m.up(&manager).await.unwrap();
    }
    db
}

#[derive(serde::Deserialize)]
struct Interaction {
    response_body: String,
}

/// Serves the recorded Anthropic responses in order.
async fn anthropic_replay(cassette: &str) -> MockServer {
    let path = format!("{}/../rust_llm/tests/cassettes/{cassette}.json", env!("CARGO_MANIFEST_DIR"));
    let interactions: Vec<Interaction> = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let server = MockServer::start().await;
    for (i, interaction) in interactions.iter().enumerate() {
        Mock::given(matchers::method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(interaction.response_body.clone().into_bytes(), "application/json"))
            .up_to_n_times(1)
            .with_priority((i + 1) as u8)
            .mount(&server)
            .await;
    }
    server
}

fn config(server: &MockServer) -> Arc<rust_llm::Config> {
    let mut c = rust_llm::Config::default();
    c.set("anthropic_api_base", server.uri());
    c.set("anthropic_api_key", "test");
    c.max_retries = 0;
    Arc::new(c)
}

#[tokio::test]
async fn persists_a_tool_calling_conversation_and_reloads_it() {
    let db = db().await;
    let server = anthropic_replay("chat_function_calling_anthropic_claude-haiku-4-5_can_use_tools").await;
    let record = ChatRecord::create(&db, "claude-haiku-4-5", Some("anthropic")).await.unwrap();
    let mut chat = record.to_llm_with(&db, config(&server)).await.unwrap().with_tool(Weather);

    let answer = record.ask(&db, &mut chat, "What's the weather in Berlin? (52.5200, 13.4050)").await.unwrap();
    assert!(answer.content().contains("15"), "{:?}", answer.content);

    // user, assistant tool call, tool result, assistant answer: the same four rows RubyLLM writes.
    let rows = record.messages(&db).await.unwrap();
    let roles: Vec<&str> = rows.iter().map(|m| m.role.as_str()).collect();
    assert_eq!(roles, ["user", "assistant", "tool", "assistant"]);

    let calls = rust_llm_tool_calls::Entity::find().all(&db).await.unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "weather");
    assert_eq!(calls[0].message_id, rows[1].id as i64, "tool call belongs to the assistant message");
    assert_eq!(calls[0].result_id, Some(rows[2].id as i64), "tool call links to its result message");
    assert_eq!(calls[0].arguments.as_ref().unwrap()["latitude"], "52.5200");

    // Two billed requests, each linked to the assistant message it produced, priced from the registry.
    let usages = rust_llm_usages::Entity::find().all(&db).await.unwrap();
    assert_eq!(usages.len(), 2);
    assert!(usages.iter().all(|u| u.status == "succeeded" && u.provider == "anthropic"));
    assert_eq!(usages[0].message_id, Some(rows[1].id as i64));
    assert_eq!(usages[1].message_id, Some(rows[3].id as i64));
    assert_eq!(usages[0].input_tokens, Some(633), "matches the cassette's usage block");
    assert!(record.total_cost(&db).await.unwrap().unwrap() > 0.0);

    // Reloading rebuilds the conversation: tool result linked back to its call, history intact.
    let reloaded = record.to_llm_with(&db, config(&server)).await.unwrap();
    let msgs = reloaded.messages();
    assert_eq!(msgs.len(), 4);
    assert_eq!(msgs[2].role, Role::Tool);
    assert_eq!(msgs[2].tool_call_id.as_deref(), Some(calls[0].tool_call_id.as_str()));
    assert!(reloaded.is_complete());
    assert_eq!(reloaded.tokens().input, Some(633 + usages[1].input_tokens.unwrap() as i64));
}

#[tokio::test]
async fn a_chat_parked_on_approval_resumes_from_the_database() {
    let db = db().await;
    let server = MockServer::start().await;
    let tool_use = serde_json::json!({
        "model": "claude-haiku-4-5-20251001", "id": "msg_1", "type": "message", "role": "assistant",
        "content": [{ "type": "tool_use", "id": "toolu_1", "name": "delete_everything", "input": {} }],
        "stop_reason": "tool_use", "usage": { "input_tokens": 10, "output_tokens": 5 }
    });
    let done = serde_json::json!({
        "model": "claude-haiku-4-5-20251001", "id": "msg_2", "type": "message", "role": "assistant",
        "content": [{ "type": "text", "text": "Done." }],
        "stop_reason": "end_turn", "usage": { "input_tokens": 20, "output_tokens": 2 }
    });
    Mock::given(matchers::method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(tool_use))
        .up_to_n_times(1)
        .with_priority(1)
        .mount(&server)
        .await;
    Mock::given(matchers::method("POST")).respond_with(ResponseTemplate::new(200).set_body_json(done)).with_priority(2).mount(&server).await;

    let record = ChatRecord::create(&db, "claude-haiku-4-5", Some("anthropic")).await.unwrap();
    let mut chat = record.to_llm_with(&db, config(&server)).await.unwrap().with_tool(DeleteEverything);
    record.ask(&db, &mut chat, "Delete everything").await.unwrap();
    assert!(chat.is_awaiting_approval());
    assert_eq!(chat.pending_approvals()[0].id, "toolu_1");
    drop(chat);

    // A different request (another process, a job) picks the chat up from rows alone.
    let mut resumed = record.to_llm_with(&db, config(&server)).await.unwrap().with_tool(DeleteEverything);
    assert!(resumed.is_awaiting_approval(), "the pending call survives the reload");
    record.approve(&db, &mut resumed, "toolu_1").await.unwrap();
    let answer = record.complete(&db, &mut resumed).await.unwrap();
    assert_eq!(answer.content(), "Done.");

    let calls = rust_llm_tool_calls::Entity::find().all(&db).await.unwrap();
    assert_eq!(calls[0].approval.as_deref(), Some("approved"));
    let roles: Vec<String> = record.messages(&db).await.unwrap().into_iter().map(|m| m.role).collect();
    assert_eq!(roles, ["user", "assistant", "tool", "assistant"]);
}

fn anthropic_text(text: &str) -> serde_json::Value {
    serde_json::json!({
        "model": "claude-haiku-4-5-20251001", "id": "m", "type": "message", "role": "assistant",
        "content": [{ "type": "text", "text": text }], "stop_reason": "end_turn",
        "usage": { "input_tokens": 3, "output_tokens": 1 }
    })
}

// Review finding 2: history is keyed by row identity, not row counts, so runtime instruction
// changes on a reloaded chat never skip or misplace messages.
#[tokio::test]
async fn instructions_on_a_reloaded_chat_keep_history_in_order() {
    let db = db().await;
    let server = MockServer::start().await;
    Mock::given(matchers::method("POST")).respond_with(ResponseTemplate::new(200).set_body_json(anthropic_text("ok"))).mount(&server).await;
    let record = ChatRecord::create(&db, "claude-haiku-4-5", Some("anthropic")).await.unwrap();

    let mut chat = record.to_llm_with(&db, config(&server)).await.unwrap();
    record.with_instructions(&db, &mut chat, "Be brief.").await.unwrap();
    record.ask(&db, &mut chat, "first").await.unwrap();

    // Next request: reload, drop the instructions in memory only, ask again.
    let mut chat = record.to_llm_with(&db, config(&server)).await.unwrap();
    chat.set_instructions(None, false, false);
    record.ask(&db, &mut chat, "second").await.unwrap();

    let rows: Vec<(String, Option<String>)> =
        record.messages(&db).await.unwrap().into_iter().map(|m| (m.role, m.content)).collect();
    let expected = [("system", "Be brief."), ("user", "first"), ("assistant", "ok"), ("user", "second"), ("assistant", "ok")];
    assert_eq!(rows.len(), expected.len(), "{rows:?}");
    for ((role, content), (er, ec)) in rows.iter().zip(expected) {
        assert_eq!((role.as_str(), content.as_deref()), (er, Some(ec)));
    }
}

// Review finding 6: decisions are scoped to the chat's own tool calls.
#[tokio::test]
async fn approving_another_chats_tool_call_is_rejected() {
    let db = db().await;
    let server = MockServer::start().await;
    let tool_use = serde_json::json!({
        "model": "claude-haiku-4-5-20251001", "id": "msg_1", "type": "message", "role": "assistant",
        "content": [{ "type": "tool_use", "id": "toolu_other", "name": "delete_everything", "input": {} }],
        "stop_reason": "tool_use", "usage": { "input_tokens": 10, "output_tokens": 5 }
    });
    Mock::given(matchers::method("POST")).respond_with(ResponseTemplate::new(200).set_body_json(tool_use)).mount(&server).await;
    let victim = ChatRecord::create(&db, "claude-haiku-4-5", Some("anthropic")).await.unwrap();
    let mut victim_chat = victim.to_llm_with(&db, config(&server)).await.unwrap().with_tool(DeleteEverything);
    victim.ask(&db, &mut victim_chat, "Delete everything").await.unwrap();

    let attacker = ChatRecord::create(&db, "claude-haiku-4-5", Some("anthropic")).await.unwrap();
    let mut attacker_chat = attacker.to_llm_with(&db, config(&server)).await.unwrap();
    assert!(attacker.approve(&db, &mut attacker_chat, "toolu_other").await.is_err());
    let calls = rust_llm_tool_calls::Entity::find().all(&db).await.unwrap();
    assert_eq!(calls[0].approval, None);
}

struct Broken;

#[async_trait]
impl Tool for Broken {
    fn description(&self) -> String {
        "Always fails".into()
    }
    async fn execute(&self, _: Map<String, Value>, _: &ToolCall) -> Result<ToolResult, ToolError> {
        Err("This tool is broken".into())
    }
}

// Review findings 1 and 6: rows are written as the loop goes, and a round that fails mid-way is
// rolled back (cleanup_after_failure) so the next ask starts clean.
#[tokio::test]
async fn a_failed_tool_round_is_rolled_back_and_the_chat_stays_usable() {
    let db = db().await;
    let server = MockServer::start().await;
    let tool_use = serde_json::json!({
        "model": "claude-haiku-4-5-20251001", "id": "msg_1", "type": "message", "role": "assistant",
        "content": [{ "type": "tool_use", "id": "toolu_b", "name": "broken", "input": {} }],
        "stop_reason": "tool_use", "usage": { "input_tokens": 10, "output_tokens": 5 }
    });
    Mock::given(matchers::method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(tool_use))
        .up_to_n_times(1)
        .with_priority(1)
        .mount(&server)
        .await;
    Mock::given(matchers::method("POST")).respond_with(ResponseTemplate::new(200).set_body_json(anthropic_text("fine"))).with_priority(2).mount(&server).await;
    let record = ChatRecord::create(&db, "claude-haiku-4-5", Some("anthropic")).await.unwrap();
    let mut chat = record.to_llm_with(&db, config(&server)).await.unwrap().with_tool(Broken);

    let err = record.ask(&db, &mut chat, "use the tool").await.unwrap_err();
    assert!(err.to_string().contains("This tool is broken"));
    let roles: Vec<String> = record.messages(&db).await.unwrap().into_iter().map(|m| m.role).collect();
    assert_eq!(roles, ["user"], "the dangling tool-call message was destroyed");
    assert!(rust_llm_tool_calls::Entity::find().all(&db).await.unwrap().is_empty());
    // The billed attempt survives, unlinked, like RubyLLM's ledger.
    let usages = rust_llm_usages::Entity::find().all(&db).await.unwrap();
    assert_eq!(usages.len(), 1);
    assert_eq!(usages[0].message_id, None);

    let answer = record.ask(&db, &mut chat, "try again without it").await.unwrap();
    assert_eq!(answer.content(), "fine");
}

/// A chat can be completed inside a Loco worker or an axum handler, which both require `Send`
/// futures (the generated chat_ui worker awaits `ChatRecord::complete` directly).
#[allow(dead_code)]
fn complete_future_is_send(db: &'static DatabaseConnection, record: &'static ChatRecord, chat: &'static mut rust_llm::Chat) {
    fn assert_send<T: Send>(_: T) {}
    assert_send(record.complete(db, chat));
}
