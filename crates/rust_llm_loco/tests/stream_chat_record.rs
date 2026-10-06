//! `ChatRecord::complete_stream` / `ask_stream`: `acts_as_chat` with a streaming block. RubyLLM's
//! persistence callbacks create the assistant row before the first chunk (`persist_new_message`)
//! and fill it in when the message completes (`persist_message_completion`); the chat_ui job
//! broadcasts each chunk into that row (`chat.messages.last.broadcast_append_chunk`).
//!
//! Replays RubyLLM's recorded Anthropic streaming cassette and requires every request body to be
//! JSON-equal to the recording.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use rust_llm::{Parameter, Tool, ToolCall, ToolError, ToolResult};
use rust_llm_loco::entities::{messages, rust_llm_tool_calls, rust_llm_usages};
use rust_llm_loco::{ChatRecord, StreamEvent, migrations};
use sea_orm::{Database, DatabaseConnection, EntityTrait};
use sea_orm_migration::SchemaManager;
use serde_json::{Map, Value};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate, matchers};

const CASSETTE: &str = "chat_function_calling_anthropic_claude-haiku-4-5_can_use_tools_with_multi-turn_streaming_conversations";

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
    async fn execute(
        &self,
        args: Map<String, Value>,
        _: &ToolCall,
    ) -> Result<ToolResult, ToolError> {
        Ok(format!(
            "Current weather at {}, {}: 15°C, Wind: 10 km/h",
            args["latitude"].as_str().unwrap_or_default(),
            args["longitude"].as_str().unwrap_or_default()
        )
        .into())
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
    request_body: String,
    response_body: String,
}

/// Serves the recorded responses in order, noting any request body that differs.
struct Replay {
    interactions: Vec<Interaction>,
    next: Mutex<usize>,
    mismatches: Arc<Mutex<Vec<String>>>,
}

impl Respond for Replay {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let mut next = self.next.lock().unwrap();
        let Some(i) = self.interactions.get(*next) else {
            self.mismatches.lock().unwrap().push("extra request".into());
            return ResponseTemplate::new(599);
        };
        *next += 1;
        let expected: Value = serde_json::from_str(&i.request_body).unwrap();
        let sent: Value = serde_json::from_slice(&request.body).unwrap_or(Value::Null);
        if expected != sent {
            self.mismatches
                .lock()
                .unwrap()
                .push(format!("expected {expected}\n  sent {sent}"));
        }
        ResponseTemplate::new(200)
            .set_body_raw(i.response_body.clone().into_bytes(), "text/event-stream")
    }
}

/// The first `take` interactions of the cassette.
async fn replay(take: usize) -> (MockServer, Arc<Mutex<Vec<String>>>) {
    let path = format!(
        "{}/../rust_llm/tests/cassettes/{CASSETTE}.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let mut interactions: Vec<Interaction> =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    interactions.truncate(take);
    let server = MockServer::start().await;
    let mismatches = Arc::new(Mutex::new(Vec::new()));
    Mock::given(matchers::method("POST"))
        .respond_with(Replay {
            interactions,
            next: Mutex::new(0),
            mismatches: mismatches.clone(),
        })
        .mount(&server)
        .await;
    (server, mismatches)
}

fn config(server: &MockServer) -> Arc<rust_llm::Config> {
    let mut c = rust_llm::Config::default();
    c.set("anthropic_api_base", server.uri());
    c.set("anthropic_api_key", "test");
    c.max_retries = 0;
    Arc::new(c)
}

/// What the callback saw, owned: `("new" | "chunk" | "end", message id, content)`.
type Seen = Vec<(&'static str, i32, String)>;

fn note(seen: &mut Seen, event: StreamEvent<'_>) {
    match event {
        StreamEvent::NewMessage(row) => {
            seen.push(("new", row.id, row.content.clone().unwrap_or_default()))
        }
        StreamEvent::Chunk { message_id, chunk } => {
            seen.push(("chunk", message_id, chunk.content().to_string()))
        }
        StreamEvent::EndMessage(row) => {
            seen.push(("end", row.id, row.content.clone().unwrap_or_default()))
        }
    }
}

#[tokio::test]
async fn streams_a_tool_round_into_rows_created_before_their_chunks() {
    let db = db().await;
    let (server, mismatches) = replay(2).await;
    let record = ChatRecord::create(&db, "claude-haiku-4-5", Some("anthropic"))
        .await
        .unwrap();
    let mut chat = record
        .to_llm_with(&db, config(&server))
        .await
        .unwrap()
        .with_tool(Weather);

    // The row a chunk names must already exist and still be blank while streaming, as
    // `chat.messages.last` is in RubyLLM's job.
    let rows_during_chunks: Arc<Mutex<Vec<i32>>> = Arc::default();
    let mut seen: Seen = Vec::new();
    let answer = record
        .ask_stream(
            &db,
            &mut chat,
            "What's the weather in Berlin? (52.5200, 13.4050)",
            |event| {
                if let StreamEvent::Chunk { message_id, .. } = &event {
                    rows_during_chunks.lock().unwrap().push(*message_id);
                }
                note(&mut seen, event);
            },
        )
        .await
        .unwrap();
    assert!(
        mismatches.lock().unwrap().is_empty(),
        "{:?}",
        mismatches.lock().unwrap()
    );
    assert!(answer.content().contains("15°C"), "{:?}", answer.content);

    // user, assistant tool call, tool result, assistant answer: the rows `ask` writes.
    let rows = record.messages(&db).await.unwrap();
    let roles: Vec<&str> = rows.iter().map(|m| m.role.as_str()).collect();
    assert_eq!(roles, ["user", "assistant", "tool", "assistant"]);
    let (call_row, result_row, answer_row) = (rows[1].id, rows[2].id, rows[3].id);

    // Each response: NewMessage (blank) before its first chunk, chunks naming that row, then
    // EndMessage with the final content. The tool result arrives whole.
    let kinds: Vec<(&str, i32)> = seen.iter().map(|(k, id, _)| (*k, *id)).collect();
    let first_chunk = kinds.iter().position(|k| k.0 == "chunk").unwrap();
    assert_eq!(kinds[0], ("new", call_row));
    assert_eq!(seen[0].2, "", "the placeholder is blank");
    assert!(first_chunk > 0);
    let end_call = kinds.iter().position(|k| *k == ("end", call_row)).unwrap();
    assert!(kinds[1..end_call].iter().all(|k| *k == ("chunk", call_row)));
    assert_eq!(kinds[end_call + 1], ("new", result_row));
    assert_eq!(kinds[end_call + 2], ("end", result_row));
    assert!(seen[end_call + 2].2.contains("15°C"));
    assert_eq!(kinds[end_call + 3], ("new", answer_row));
    let answer_chunks: Vec<&String> = seen
        .iter()
        .filter(|(k, id, _)| *k == "chunk" && *id == answer_row)
        .map(|(_, _, text)| text)
        .collect();
    assert!(
        answer_chunks.len() > 3,
        "token by token, not one block: {answer_chunks:?}"
    );
    let (last_kind, last_id, last_text) = seen.last().unwrap();
    assert_eq!((*last_kind, *last_id), ("end", answer_row));
    assert_eq!(last_text, rows[3].content.as_deref().unwrap());
    // The streamed text is the persisted text (the protocol's final whole-message chunk, if any,
    // aside).
    let streamed: String = answer_chunks.iter().map(|s| s.as_str()).collect();
    assert!(
        streamed.starts_with(rows[3].content.as_deref().unwrap()),
        "{streamed:?}"
    );

    // The tool call belongs to the placeholder row it streamed into, and links to its result.
    let calls = rust_llm_tool_calls::Entity::find().all(&db).await.unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].message_id, i64::from(call_row));
    assert_eq!(calls[0].result_id, Some(i64::from(result_row)));
    // Each request's usage is linked to the row it produced.
    let usages = rust_llm_usages::Entity::find().all(&db).await.unwrap();
    assert_eq!(
        usages.iter().map(|u| u.message_id).collect::<Vec<_>>(),
        [Some(i64::from(call_row)), Some(i64::from(answer_row))]
    );
    assert_eq!(usages[0].input_tokens, Some(633));

    // Reloaded from the rows, the chat continues where the stream left off.
    let reloaded = record.to_llm_with(&db, config(&server)).await.unwrap();
    assert_eq!(reloaded.messages().len(), 4);
    assert!(reloaded.is_complete());
}

#[tokio::test]
async fn a_failed_stream_leaves_no_blank_row() {
    let db = db().await;
    let server = MockServer::start().await;
    Mock::given(matchers::method("POST"))
        .respond_with(ResponseTemplate::new(500).set_body_json(
            serde_json::json!({ "type": "error", "error": { "type": "api_error", "message": "boom" } }),
        ))
        .mount(&server)
        .await;
    let record = ChatRecord::create(&db, "claude-haiku-4-5", Some("anthropic"))
        .await
        .unwrap();
    let mut chat = record.to_llm_with(&db, config(&server)).await.unwrap();
    let mut seen: Seen = Vec::new();
    let err = record
        .ask_stream(&db, &mut chat, "Hello", |event| note(&mut seen, event))
        .await
        .unwrap_err();
    assert!(matches!(err, rust_llm_loco::Error::Llm(_)), "{err:?}");
    // The placeholder was announced, then destroyed (`cleanup_failed_messages`).
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0].0, "new");
    let rows = messages::Entity::find().all(&db).await.unwrap();
    assert_eq!(
        rows.iter().map(|m| m.role.as_str()).collect::<Vec<_>>(),
        ["user"]
    );
    // Usage for the failed attempt stays in the ledger, unlinked.
    let usages = rust_llm_usages::Entity::find().all(&db).await.unwrap();
    assert!(usages.iter().all(|u| u.message_id.is_none()), "{usages:?}");
}

/// The chat_ui worker streams inside a Loco worker, which needs a `Send` future.
#[allow(dead_code)]
fn complete_stream_future_is_send(
    db: &'static DatabaseConnection,
    record: &'static ChatRecord,
    chat: &'static mut rust_llm::Chat,
) {
    fn assert_send<T: Send>(_: T) {}
    assert_send(record.complete_stream(db, chat, |_| {}));
}
