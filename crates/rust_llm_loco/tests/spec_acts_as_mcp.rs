//! `spec/ruby_llm/active_record/acts_as_mcp_apps_spec.rb` and `acts_as_mcp_tasks_spec.rb` on
//! SQLite: the result an MCP App UI renders persists on its tool call, and a tool call that
//! became an MCP task persists the task so another process checks on it, resumes, or cancels it.

use std::sync::Arc;

use rust_llm::mcp::{Extension, Mcp, TaskStatus};
use rust_llm::{Message, Role, ToolCall};
use rust_llm_loco::entities::rust_llm_tool_calls;
use rust_llm_loco::{ChatRecord, migrations, tool_error_message};
use sea_orm::{ColumnTrait, Database, DatabaseConnection, EntityTrait, QueryFilter};
use sea_orm_migration::SchemaManager;
use serde_json::{Map, Value, json};
use wiremock::{Mock, MockServer, ResponseTemplate, matchers};

const MODEL: &str = "gpt-4.1-nano";

async fn db() -> DatabaseConnection {
    rust_llm::configure(|c| {
        c.set("openai_api_key", "test");
    });
    let db = Database::connect("sqlite::memory:").await.unwrap();
    let manager = SchemaManager::new(&db);
    for m in migrations() {
        m.up(&manager).await.unwrap();
    }
    db
}

/// The model answers "Done" to every request.
async fn done() -> MockServer {
    let server = MockServer::start().await;
    Mock::given(matchers::method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "resp_1", "object": "response", "status": "completed", "model": MODEL,
            "output": [{ "type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                         "content": [{ "type": "output_text", "text": "Done", "annotations": [] }] }],
            "usage": { "input_tokens": 1, "output_tokens": 1, "total_tokens": 2 }
        })))
        .mount(&server)
        .await;
    server
}

fn config(server: &MockServer) -> Arc<rust_llm::Config> {
    let mut c = rust_llm::Config::default();
    c.set("openai_api_base", format!("{}/v1", server.uri()));
    c.set("openai_api_key", "test");
    c.max_retries = 0;
    Arc::new(c)
}

fn spec_server(extension: Extension) -> Mcp {
    let server = format!(
        "{}/../rust_llm/tests/fixtures/mcp/server.rb",
        env!("CARGO_MANIFEST_DIR")
    );
    Mcp::command(["ruby".to_string(), server])
        .name("files")
        .with_extension(extension, json!({}))
        .build()
        .unwrap()
}

/// `Chat.find(id).with_mcp(server)`: the chat as another process loads it.
async fn load(
    db: &DatabaseConnection,
    id: i32,
    server: &MockServer,
    mcp: &Mcp,
) -> (ChatRecord, rust_llm::Chat) {
    let record = ChatRecord::find(db, id).await.unwrap();
    let chat = record
        .to_llm_with(db, config(server))
        .await
        .unwrap()
        .with_mcp(mcp.clone());
    (record, chat)
}

/// Adds an assistant message calling `tool` (`call_1`) and completes the chat.
async fn called(
    db: &DatabaseConnection,
    server: &MockServer,
    mcp: &Mcp,
    tool: &str,
    arguments: Value,
) -> ChatRecord {
    let record = ChatRecord::create(db, MODEL, None).await.unwrap();
    let mut chat = record
        .to_llm_with(db, config(server))
        .await
        .unwrap()
        .with_mcp(mcp.clone());
    let mut assistant = Message::new(Role::Assistant, Some(String::new()));
    let arguments: Map<String, Value> = arguments.as_object().cloned().unwrap_or_default();
    assistant.tool_calls = Some(
        [(
            "call_1".to_string(),
            ToolCall::new("call_1", tool, arguments),
        )]
        .into_iter()
        .collect(),
    );
    record.add_message(db, &mut chat, assistant).await.unwrap();
    record.complete(db, &mut chat).await.unwrap();
    record
}

async fn tool_call_row(db: &DatabaseConnection) -> rust_llm_tool_calls::Model {
    rust_llm_tool_calls::Entity::find()
        .filter(rust_llm_tool_calls::Column::ToolCallId.eq("call_1"))
        .one(db)
        .await
        .unwrap()
        .unwrap()
}

/// `Chat.find(id).messages_association.find_by(role: 'tool')`, as `to_llm` restores it.
async fn tool_message(db: &DatabaseConnection, id: i32, server: &MockServer, mcp: &Mcp) -> Message {
    let (_, chat) = load(db, id, server, mcp).await;
    chat.messages()
        .iter()
        .find(|m| m.role == Role::Tool)
        .cloned()
        .expect("a tool result")
}

// ---- acts_as_mcp_apps_spec ---------------------------------------------------------------------

// spec: active_record/acts_as_mcp_apps_spec.rb:32 keeps the result of a tool with a UI on its tool call to render it again after a reload
#[tokio::test]
async fn keeps_the_result_of_a_tool_with_a_ui_on_its_tool_call() {
    let db = db().await;
    let server = done().await;
    let weather = spec_server(Extension::Apps);
    let record = called(
        &db,
        &server,
        &weather,
        "forecast",
        json!({ "city": "Rome" }),
    )
    .await;

    assert_eq!(
        tool_call_row(&db).await.arguments,
        Some(json!({ "city": "Rome" }))
    );
    let result = tool_message(&db, record.id(), &server, &weather)
        .await
        .mcp_result
        .expect("the result the UI renders");
    assert_eq!(result.ui_uri.as_deref(), Some("ui://spec/forecast"));
    assert_eq!(result.text, "Sunny in Rome");
    assert_eq!(
        result.structured,
        Some(json!({ "city": "Rome", "temperature": 24 }))
    );
    assert_eq!(
        Value::Object(result.meta),
        json!({ "com.example/station": "spec" })
    );
    weather.close().await;
}

// spec: active_record/acts_as_mcp_apps_spec.rb:44 stores a failed call as 2.0 did
#[tokio::test]
async fn stores_a_failed_call_as_2_0_did() {
    let db = db().await;
    let server = done().await;
    let weather = spec_server(Extension::Apps);
    let record = called(&db, &server, &weather, "fail", json!({})).await;

    let message = tool_message(&db, record.id(), &server, &weather).await;
    assert_eq!(message.content(), r#"{"error":"Something broke"}"#);
    assert_eq!(
        tool_error_message(message.content.as_deref()).as_deref(),
        Some("Something broke")
    );
    assert!(message.mcp_result.is_none());
    weather.close().await;
}

// spec: active_record/acts_as_mcp_apps_spec.rb:53 keeps nothing for tools without a UI
#[tokio::test]
async fn keeps_nothing_for_tools_without_a_ui() {
    let db = db().await;
    let server = done().await;
    let weather = spec_server(Extension::Apps);
    let record = called(&db, &server, &weather, "add", json!({ "a": 2, "b": 3 })).await;

    assert_eq!(tool_call_row(&db).await.mcp_result, None);
    assert!(
        tool_message(&db, record.id(), &server, &weather)
            .await
            .mcp_result
            .is_none()
    );
    weather.close().await;
}

// ---- acts_as_mcp_tasks_spec --------------------------------------------------------------------

async fn server_tasks(mcp: &Mcp) -> Value {
    mcp.client()
        .request("spec/tasks", json!({}), &[], &mut |_| {})
        .await
        .unwrap()
}

// spec: active_record/acts_as_mcp_tasks_spec.rb:33 persists the task a tool call became
#[tokio::test]
async fn persists_the_task_a_tool_call_became() {
    let db = db().await;
    let server = done().await;
    let reports = spec_server(Extension::Tasks);
    let record = called(&db, &server, &reports, "report", json!({})).await;

    let (_, chat) = load(&db, record.id(), &server, &reports).await;
    assert!(chat.is_awaiting_tasks());
    let state = tool_call_row(&db).await.mcp_state.unwrap();
    assert_eq!(state["task"]["taskId"], "task-1");
    assert_eq!(state["task"]["pollIntervalMs"], 10);
    assert_eq!(state["task"]["ttlMs"], 60_000);
    let task = chat.pending_tasks().remove(0);
    assert_eq!(
        (task.id.as_str(), task.status()),
        ("task-1", TaskStatus::Working)
    );
    reports.close().await;
}

// spec: active_record/acts_as_mcp_tasks_spec.rb:42 checks on the task from another process and resumes once it finishes
#[tokio::test]
async fn checks_on_the_task_from_another_process_and_resumes_once_it_finishes() {
    let db = db().await;
    let server = done().await;
    let reports = spec_server(Extension::Tasks);
    let record = called(&db, &server, &reports, "report", json!({})).await;

    let (checking, mut chat) = load(&db, record.id(), &server, &reports).await;
    checking.complete(&db, &mut chat).await.unwrap();
    drop(chat);
    let (resumed, mut chat) = load(&db, record.id(), &server, &reports).await;
    assert_eq!(chat.pending_tasks()[0].status_message(), Some("Rendering"));

    resumed.complete(&db, &mut chat).await.unwrap();

    let tool = resumed
        .messages(&db)
        .await
        .unwrap()
        .into_iter()
        .find(|m| m.role == "tool")
        .unwrap();
    assert_eq!(tool.content.as_deref(), Some("Report ready"));
    assert_eq!(tool_call_row(&db).await.mcp_state, None);
    let (_, reloaded) = load(&db, record.id(), &server, &reports).await;
    assert!(!reloaded.is_awaiting_tasks());
    reports.close().await;
}

// spec: active_record/acts_as_mcp_tasks_spec.rb:59 cancels the task when another process cancels the chat
#[tokio::test]
async fn cancels_the_task_when_another_process_cancels_the_chat() {
    let db = db().await;
    let server = done().await;
    let reports = spec_server(Extension::Tasks);
    let record = called(&db, &server, &reports, "endless_report", json!({})).await;
    ChatRecord::find(&db, record.id())
        .await
        .unwrap()
        .cancel(&db)
        .await
        .unwrap();

    let (resuming, mut chat) = load(&db, record.id(), &server, &reports).await;
    let result = resuming.complete(&db, &mut chat).await;
    assert!(
        matches!(
            &result,
            Err(rust_llm_loco::Error::Llm(rust_llm::Error::Cancelled))
        ),
        "{result:?}"
    );
    assert_eq!(server_tasks(&reports).await["cancelled"], json!(["task-1"]));
    reports.close().await;
}
