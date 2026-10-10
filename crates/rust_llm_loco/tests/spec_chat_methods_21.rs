//! RubyLLM 2.1's new `ChatMethods` examples on `ChatRecord`: cache boundary lifetimes
//! (`chat_methods_spec.rb`), provider tool uses per usage row, unsupported attachments and deferred
//! tools through the record's chat, transcripts rebuilt from rows, the usage inverses
//! (`acts_as_spec.rb`), tool calls reloaded in the order they were made
//! (`acts_as_tool_approval_spec.rb`), and the persisted side of crash recovery
//! (`chat_methods_crash_recovery_spec.rb`).

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use rust_llm::message::Operation;
use rust_llm::message::indexmap_lite::IndexMap;
use rust_llm::{
    Attachment, Config, Message, Role, Tokens, Tool, ToolCall, ToolError, ToolResult, UsageEntry,
    UsageStatus,
};
use rust_llm_loco::entities::{
    messages, rust_llm_attachments, rust_llm_tool_calls, rust_llm_usages,
};
use rust_llm_loco::{ChatRecord, message_to_llm, migrations};
use sea_orm::{
    ActiveModelTrait, ActiveValue::Set, ColumnTrait, ConnectionTrait, Database, DatabaseConnection,
    EntityTrait, QueryFilter,
};
use sea_orm_migration::SchemaManager;
use serde_json::{Map, Value, json};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

/// `model_for(:openai, :temperature)`.
const MODEL: &str = "gpt-4.1-nano";

async fn db() -> DatabaseConnection {
    rust_llm::configure(|c| {
        c.set("openai_api_key", "test");
        c.set("anthropic_api_key", "test");
    });
    let db = Database::connect("sqlite::memory:").await.unwrap();
    let manager = SchemaManager::new(&db);
    for m in migrations() {
        m.up(&manager).await.unwrap();
    }
    db
}

fn now() -> sea_orm::prelude::DateTimeWithTimeZone {
    chrono::Utc::now().into()
}

/// `chat.messages.create!(role:, content:)`.
async fn create_message(
    db: &DatabaseConnection,
    chat_id: i32,
    role: &str,
    content: &str,
) -> messages::Model {
    messages::ActiveModel {
        chat_id: Set(chat_id),
        role: Set(role.into()),
        content: Set(Some(content.into())),
        cache_until_here: Set(false),
        created_at: Set(now()),
        updated_at: Set(now()),
        ..Default::default()
    }
    .insert(db)
    .await
    .unwrap()
}

// ---- chat_methods_spec.rb: cache boundary lifetimes -----------------------------------------------

// spec: active_record/chat_methods_spec.rb:331 #with_instructions with a cache boundary > persists the boundary lifetime and replays it after a reload
#[tokio::test]
async fn persists_the_boundary_lifetime_and_replays_it_after_a_reload() {
    let db = db().await;
    let mut record = ChatRecord::create(&db, MODEL, None).await.unwrap();
    let mut chat = record.to_llm(&db).await.unwrap();

    record
        .set_instructions_with(
            &db,
            &mut chat,
            Some("Stable policy"),
            false,
            true,
            &json!({ "ttl": "1h" }),
        )
        .await
        .unwrap();

    let system: Vec<_> = record
        .messages(&db)
        .await
        .unwrap()
        .into_iter()
        .filter(|m| m.role == "system")
        .collect();
    assert_eq!(system.len(), 1);
    assert_eq!(system[0].cache_ttl.as_deref(), Some("1h"));
    let reloaded = ChatRecord::find(&db, record.id()).await.unwrap();
    let messages = reloaded.to_llm(&db).await.unwrap().messages().to_vec();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].cache_ttl.as_deref(), Some("1h"));
}

// spec: active_record/chat_methods_spec.rb:340 #with_instructions with a cache boundary > clears the lifetime when the same instructions return without one
#[tokio::test]
async fn clears_the_lifetime_when_the_same_instructions_return_without_one() {
    let db = db().await;
    let mut record = ChatRecord::create(&db, MODEL, None).await.unwrap();
    let mut chat = record.to_llm(&db).await.unwrap();
    record
        .set_instructions_with(
            &db,
            &mut chat,
            Some("Stable policy"),
            false,
            true,
            &json!({ "ttl": "1h" }),
        )
        .await
        .unwrap();

    record
        .set_instructions(&db, &mut chat, Some("Stable policy"), false, true, true)
        .await
        .unwrap();

    let system: Vec<_> = record
        .messages(&db)
        .await
        .unwrap()
        .into_iter()
        .filter(|m| m.role == "system")
        .collect();
    assert_eq!(system.len(), 1);
    assert_eq!(system[0].cache_ttl, None);
    assert!(system[0].cache_until_here);
}

// spec: active_record/chat_methods_spec.rb:388 #cache_until_here > persists a boundary lifetime on the last persisted message
#[tokio::test]
async fn persists_a_boundary_lifetime_on_the_last_persisted_message() {
    let db = db().await;
    let record = ChatRecord::create(&db, MODEL, None).await.unwrap();
    let mut chat = record.to_llm(&db).await.unwrap();
    record
        .add_message(&db, &mut chat, Message::user("Reusable prompt"))
        .await
        .unwrap();

    record
        .cache_until_here_with(&db, &mut chat, Some("1h"))
        .await
        .unwrap();

    let last = record.messages(&db).await.unwrap().pop().unwrap();
    assert_eq!(last.cache_ttl.as_deref(), Some("1h"));
    let reloaded = ChatRecord::find(&db, record.id()).await.unwrap();
    let messages = reloaded.to_llm(&db).await.unwrap().messages().to_vec();
    assert_eq!(messages.last().unwrap().cache_ttl.as_deref(), Some("1h"));
}

// ---- accounting --------------------------------------------------------------------------------

// spec: active_record/chat_methods_spec.rb:427 accounting > keeps the provider tool uses of each attempt
#[tokio::test]
async fn keeps_the_provider_tool_uses_of_each_attempt() {
    let db = db().await;
    let record = ChatRecord::create(&db, MODEL, None).await.unwrap();
    let message = create_message(&db, record.id(), "assistant", "done").await;
    for searches in [2, 1] {
        let mut entry = UsageEntry::new(Operation::Chat, "openai", Some(MODEL));
        entry.status = UsageStatus::Succeeded;
        entry.tokens = Tokens {
            input: Some(10),
            output: Some(5),
            server_tool_use: json!({ "web_search_requests": searches })
                .as_object()
                .cloned(),
            ..Default::default()
        };
        record.persist_usage_entry(&db, &entry).await.unwrap();
    }
    rust_llm_usages::Entity::update_many()
        .col_expr(
            rust_llm_usages::Column::MessageId,
            sea_orm::sea_query::Expr::value(i64::from(message.id)),
        )
        .col_expr(
            rust_llm_usages::Column::MessageType,
            sea_orm::sea_query::Expr::value("Message"),
        )
        .exec(&db)
        .await
        .unwrap();

    let found = ChatRecord::find(&db, record.id()).await.unwrap();
    assert_eq!(
        found
            .tokens(&db)
            .await
            .unwrap()
            .server_tool_use
            .map(Value::Object),
        Some(json!({ "web_search_requests": 3 }))
    );
    let row = messages::Entity::find_by_id(message.id)
        .one(&db)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        message_to_llm(&db, &row)
            .await
            .unwrap()
            .tokens()
            .server_tool_use
            .map(Value::Object),
        Some(json!({ "web_search_requests": 3 }))
    );
}

// ---- delegation to the underlying chat -----------------------------------------------------------

// spec: active_record/chat_methods_spec.rb:466 delegation to the underlying chat > replaces unsupported attachments for rendering while persisting the original
#[tokio::test]
async fn replaces_unsupported_attachments_for_rendering_while_persisting_the_original() {
    let db = db().await;
    let record = ChatRecord::create(&db, "claude-haiku-4-5", Some("anthropic"))
        .await
        .unwrap();
    let document = Attachment::from_bytes(b"office document".to_vec(), "report.docx", None);
    let chat = record.to_llm(&db).await.unwrap();
    let mut chat = chat.convert_unsupported_attachments(|_| {
        Ok(Some(Attachment::from_bytes(
            b"Extracted report".to_vec(),
            "report.txt",
            None,
        )))
    });
    record
        .ask_later_with(&db, &mut chat, "Summarize this.", vec![document])
        .await
        .unwrap();

    assert!(
        chat.render()
            .unwrap()
            .to_string()
            .contains("Extracted report")
    );
    let found = ChatRecord::find(&db, record.id()).await.unwrap();
    let mut stored = found
        .to_llm(&db)
        .await
        .unwrap()
        .messages()
        .last()
        .unwrap()
        .attachments[0]
        .clone();
    assert_eq!(stored.filename.as_deref(), Some("report.docx"));
    assert_eq!(stored.content().await.unwrap(), b"office document");
}

struct Deferred;

#[async_trait]
impl Tool for Deferred {
    fn name(&self) -> String {
        "deferred".into()
    }
    fn description(&self) -> String {
        "A deferred tool".into()
    }
    async fn execute(&self, _: Map<String, Value>, _: &ToolCall) -> Result<ToolResult, ToolError> {
        Ok("done".into())
    }
}

// spec: active_record/chat_methods_spec.rb:553 delegation to the underlying chat > defers tools registered with defer: true
#[tokio::test]
async fn defers_tools_registered_with_defer_true() {
    let db = db().await;
    let record = ChatRecord::create(&db, MODEL, None).await.unwrap();
    // `Chat.create!(...).with_tools(DeferredTool, defer: true)`: the record hands back its chat,
    // whose tools stay deferred.
    let chat = record
        .to_llm(&db)
        .await
        .unwrap()
        .with_deferred_tools([Arc::new(Deferred) as rust_llm::SharedTool]);

    assert_eq!(
        chat.deferred_tools()
            .iter()
            .map(|t| t.name())
            .collect::<Vec<_>>(),
        ["deferred"]
    );
}

// ---- #eager_load_messages -----------------------------------------------------------------------

/// `chat_with_attachments(messages:, attachments:)`.
async fn chat_with_attachments(
    db: &DatabaseConnection,
    count: usize,
    attachments: usize,
) -> ChatRecord {
    let record = ChatRecord::create(db, MODEL, None).await.unwrap();
    for index in 0..count {
        let message = create_message(db, record.id(), "user", &format!("message {index}")).await;
        for slot in 0..attachments {
            let bytes = format!("content {index}-{slot}").into_bytes();
            rust_llm_attachments::ActiveModel {
                message_type: Set("Message".into()),
                message_id: Set(i64::from(message.id)),
                filename: Set(format!("file{index}-{slot}.txt")),
                content_type: Set("text/plain".into()),
                byte_size: Set(bytes.len() as i64),
                data: Set(bytes),
                created_at: Set(now()),
                ..Default::default()
            }
            .insert(db)
            .await
            .unwrap();
        }
    }
    record
}

// spec: active_record/chat_methods_spec.rb:846 #eager_load_messages > rebuilds and inspects a transcript without downloading its files
#[tokio::test]
async fn rebuilds_and_inspects_a_transcript_without_downloading_its_files() {
    let db = db().await;
    let record = chat_with_attachments(&db, 2, 2).await;

    // The bytes live in the rows (`rust_llm_attachments.data`), so a rebuild reads no storage
    // service; what Ruby counts as downloads cannot happen. The rebuild, inspection, and approval
    // checks all succeed from the rows alone.
    let loaded = ChatRecord::find(&db, record.id()).await.unwrap();
    let mut chat = loaded.to_llm(&db).await.unwrap();
    for m in chat.messages() {
        let _ = format!("{m:?}");
    }
    assert!(!loaded.is_awaiting_approval(&db, &mut chat).await.unwrap());
    assert!(
        loaded
            .pending_approvals(&db, &mut chat)
            .await
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        chat.messages()
            .iter()
            .map(|m| m.attachments.len())
            .collect::<Vec<_>>(),
        [2, 2]
    );
}

// spec: active_record/chat_methods_spec.rb:859 #eager_load_messages > downloads each file once, when a request needs its bytes
#[tokio::test]
async fn reads_each_file_when_a_request_needs_its_bytes() {
    let db = db().await;
    let record = chat_with_attachments(&db, 2, 2).await;
    let chat = ChatRecord::find(&db, record.id())
        .await
        .unwrap()
        .to_llm(&db)
        .await
        .unwrap();

    for _ in 0..2 {
        chat.render().unwrap();
    }

    let mut contents = Vec::new();
    for m in chat.messages() {
        for a in &m.attachments {
            contents.push(String::from_utf8(a.clone().content().await.unwrap()).unwrap());
        }
    }
    assert_eq!(
        contents,
        ["content 0-0", "content 0-1", "content 1-0", "content 1-1"]
    );
}

// spec: active_record/chat_methods_spec.rb:882 #eager_load_messages > reads preloaded rows without building a proxy per message
#[tokio::test]
async fn reads_preloaded_rows_for_the_whole_transcript() {
    let db = db().await;
    let record = ChatRecord::create(&db, MODEL, None).await.unwrap();
    let call_ids: Vec<String> = (0..4).map(|i| format!("call_{i:016x}")).collect();
    for (index, call_id) in call_ids.iter().enumerate() {
        create_message(&db, record.id(), "user", &format!("question {index}")).await;
        let answer =
            create_message(&db, record.id(), "assistant", &format!("answer {index}")).await;
        rust_llm_tool_calls::ActiveModel {
            message_type: Set("Message".into()),
            message_id: Set(i64::from(answer.id)),
            tool_call_id: Set(call_id.clone()),
            name: Set("lookup".into()),
            remote: Set(false),
            created_at: Set(now()),
            updated_at: Set(now()),
            ..Default::default()
        }
        .insert(&db)
        .await
        .unwrap();
        rust_llm_usages::ActiveModel {
            chat_type: Set(Some("Chat".into())),
            chat_id: Set(Some(i64::from(record.id()))),
            message_type: Set(Some("Message".into())),
            message_id: Set(Some(i64::from(answer.id))),
            operation: Set("chat".into()),
            provider: Set("openai".into()),
            model: Set(MODEL.into()),
            status: Set("succeeded".into()),
            input_tokens: Set(Some(3)),
            output_tokens: Set(Some(5)),
            created_at: Set(now()),
            updated_at: Set(now()),
            ..Default::default()
        }
        .insert(&db)
        .await
        .unwrap();
    }

    // Ruby counts `CollectionProxy` builds; the port reads each table once per rebuild
    // (`sync_messages`), so what this checks is what those rows rebuild into.
    let chat = ChatRecord::find(&db, record.id())
        .await
        .unwrap()
        .to_llm(&db)
        .await
        .unwrap();

    assert_eq!(
        chat.messages()
            .iter()
            .map(|m| m.tokens().output)
            .collect::<Vec<_>>(),
        [None, Some(5)].repeat(4)
    );
    assert_eq!(
        chat.messages()
            .last()
            .unwrap()
            .tool_calls
            .as_ref()
            .unwrap()
            .keys()
            .cloned()
            .collect::<Vec<_>>(),
        [call_ids.last().unwrap().clone()]
    );
    // `usage_entries.map { |entry| entry.message.content }`: each entry belongs to its answer.
    let owners: Vec<String> = chat
        .usage_entries()
        .iter()
        .map(|entry| {
            chat.messages()
                .iter()
                .find(|m| m.usage_entries.iter().any(|e| e.id == entry.id))
                .map(|m| m.content().to_string())
                .unwrap_or_default()
        })
        .collect();
    assert_eq!(
        owners,
        (0..4).map(|i| format!("answer {i}")).collect::<Vec<_>>()
    );
}

// ---- acts_as_spec.rb ----------------------------------------------------------------------------

// spec: active_record/acts_as_spec.rb:65 usage persistence > declares the usage inverses so preloaded usages do not query their owners
#[tokio::test]
async fn usage_entries_share_their_message_without_querying_it_again() {
    let db = db().await;
    let record = ChatRecord::create(&db, MODEL, None).await.unwrap();
    let answer = create_message(&db, record.id(), "assistant", "answer").await;
    let mut entry = UsageEntry::new(Operation::Chat, "openai", Some(MODEL));
    entry.status = UsageStatus::Succeeded;
    let id = record.persist_usage_entry(&db, &entry).await.unwrap();
    rust_llm_usages::Entity::update_many()
        .col_expr(
            rust_llm_usages::Column::MessageId,
            sea_orm::sea_query::Expr::value(i64::from(answer.id)),
        )
        .col_expr(
            rust_llm_usages::Column::MessageType,
            sea_orm::sea_query::Expr::value("Message"),
        )
        .filter(rust_llm_usages::Column::Id.eq(id))
        .exec(&db)
        .await
        .unwrap();

    // Ruby's `inverse_of:` makes a preloaded usage reuse its chat and message objects. The port
    // has no lazy association: one rebuild reads the usage rows once and hands the same entry to
    // the chat's ledger and to its message, which is what the inverse guarantees.
    let chat = record.to_llm(&db).await.unwrap();
    let ledger = chat.usage_entries().to_vec();
    let message = chat.messages().last().unwrap();
    assert_eq!(ledger.len(), 1);
    assert_eq!(message.usage_entries, ledger);
}

// ---- acts_as_tool_approval_spec.rb ----------------------------------------------------------------

struct Dangerous;

#[async_trait]
impl Tool for Dangerous {
    fn name(&self) -> String {
        "dangerous".into()
    }
    fn description(&self) -> String {
        "Dangerous".into()
    }
    fn requires_approval(&self) -> bool {
        true
    }
    async fn execute(&self, _: Map<String, Value>, _: &ToolCall) -> Result<ToolResult, ToolError> {
        Ok("done".into())
    }
}

// spec: active_record/acts_as_tool_approval_spec.rb:71 reloads parallel calls in the order the model made them
#[tokio::test]
async fn reloads_parallel_calls_in_the_order_the_model_made_them() {
    let db = db().await;
    let ids = ["call_b0".to_string(), "call_a1".to_string()];
    let record = ChatRecord::create(&db, MODEL, None).await.unwrap();
    let mut chat = record.to_llm(&db).await.unwrap().with_tool(Dangerous);
    let mut m = Message::new(Role::Assistant, Some(String::new()));
    let map: IndexMap<ToolCall> = ids
        .iter()
        .map(|id| {
            (
                id.clone(),
                ToolCall::new(id.clone(), "dangerous", Map::new()),
            )
        })
        .collect();
    m.tool_calls = Some(map);
    record.add_message(&db, &mut chat, m).await.unwrap();
    record.approve(&db, &mut chat, &ids[0]).await.unwrap();

    // `reading_rows_in_reverse`: SQLite returns unordered rows backwards under this pragma, as
    // PostgreSQL returns a row after an update moves it.
    db.execute_unprepared("PRAGMA reverse_unordered_selects = ON")
        .await
        .unwrap();
    let reloaded = ChatRecord::find(&db, record.id()).await.unwrap();
    let calls: Vec<String> = reloaded
        .to_llm(&db)
        .await
        .unwrap()
        .messages()
        .last()
        .unwrap()
        .tool_calls
        .as_ref()
        .unwrap()
        .keys()
        .cloned()
        .collect();
    db.execute_unprepared("PRAGMA reverse_unordered_selects = OFF")
        .await
        .unwrap();

    assert_eq!(calls, ids);
}

// ---- chat_methods_crash_recovery_spec.rb ------------------------------------------------------------

struct Lookup(Arc<Mutex<Vec<&'static str>>>);

#[async_trait]
impl Tool for Lookup {
    fn name(&self) -> String {
        "lookup".into()
    }
    fn description(&self) -> String {
        "Looks things up".into()
    }
    async fn execute(&self, _: Map<String, Value>, _: &ToolCall) -> Result<ToolResult, ToolError> {
        self.0.lock().unwrap().push("lookup");
        Ok("found".into())
    }
}

/// Answers every chat request with "Answered" and keeps the request bodies.
async fn answering_server() -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(|_: &Request| {
            ResponseTemplate::new(200).set_body_json(json!({
                "id": "resp_1", "object": "response", "status": "completed", "model": MODEL,
                "output": [{ "type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                             "content": [{ "type": "output_text", "text": "Answered", "annotations": [] }] }],
                "usage": { "input_tokens": 1, "output_tokens": 1 }
            }))
        })
        .mount(&server)
        .await;
    server
}

fn openai(server: &MockServer) -> Arc<Config> {
    let mut c = Config::default();
    c.set("openai_api_base", format!("{}/v1", server.uri()));
    c.set("openai_api_key", "test");
    c.max_retries = 0;
    Arc::new(c)
}

/// What a worker leaves when it dies after one of two tools finished and before the placeholder
/// for its result was filled: `(record, placeholder id, first call, second call)`.
async fn crashed_chat(
    db: &DatabaseConnection,
    later: &[&str],
) -> (ChatRecord, i32, String, String) {
    let record = ChatRecord::create(db, MODEL, None).await.unwrap();
    let mut chat = record.to_llm(db).await.unwrap();
    let (first, second) = ("call_first01".to_string(), "call_second2".to_string());
    record
        .add_message(db, &mut chat, Message::user("Look twice"))
        .await
        .unwrap();
    let mut calling = Message::new(Role::Assistant, Some(String::new()));
    calling.tool_calls = Some(
        [&first, &second]
            .iter()
            .map(|id| {
                (
                    (*id).clone(),
                    ToolCall::new((*id).clone(), "lookup", Map::new()),
                )
            })
            .collect(),
    );
    record.add_message(db, &mut chat, calling).await.unwrap();
    record
        .add_message(db, &mut chat, Message::tool_result(first.clone(), "found"))
        .await
        .unwrap();
    let placeholder = create_message(db, record.id(), "assistant", "").await;
    for question in later {
        create_message(db, record.id(), "user", question).await;
    }
    (record, placeholder.id, first, second)
}

/// `sent`: the last request's messages as `[role, content, tool_call_id]`. On the Responses wire
/// an assistant turn's calls are one `function_call` item each; they fold back into the one
/// assistant message Ruby's provider stub receives.
async fn sent(server: &MockServer) -> Vec<Vec<String>> {
    let requests = server.received_requests().await.unwrap();
    let body: Value = serde_json::from_slice(&requests.last().unwrap().body).unwrap();
    let mut out: Vec<Vec<String>> = Vec::new();
    let mut in_calls = false;
    for item in body["input"].as_array().unwrap() {
        let text = |v: &Value| match v {
            Value::String(s) => s.clone(),
            Value::Array(parts) => parts
                .iter()
                .filter_map(|p| p["text"].as_str())
                .collect::<String>(),
            _ => String::new(),
        };
        match item["type"].as_str() {
            Some("function_call") => {
                if !in_calls {
                    out.push(vec!["assistant".into(), String::new()]);
                }
                in_calls = true;
                continue;
            }
            Some("function_call_output") => out.push(vec![
                "tool".into(),
                item["output"].as_str().unwrap_or_default().into(),
                item["call_id"].as_str().unwrap_or_default().into(),
            ]),
            _ => out.push(vec![
                item["role"].as_str().unwrap_or_default().into(),
                text(&item["content"]),
            ]),
        }
        in_calls = false;
    }
    out
}

/// `ToolCall.find_by(tool_call_id:).result`.
async fn result_of(db: &DatabaseConnection, tool_call_id: &str) -> Option<messages::Model> {
    let call = rust_llm_tool_calls::Entity::find()
        .filter(rust_llm_tool_calls::Column::ToolCallId.eq(tool_call_id))
        .one(db)
        .await
        .unwrap()?;
    messages::Entity::find_by_id(call.result_id? as i32)
        .one(db)
        .await
        .unwrap()
}

// spec: active_record/chat_methods_crash_recovery_spec.rb:53 runs the unfinished call again when the chat resumes
#[tokio::test]
async fn runs_the_unfinished_call_again_when_the_chat_resumes() {
    let db = db().await;
    let server = answering_server().await;
    let (record, placeholder, first, second) = crashed_chat(&db, &[]).await;
    let executions: Arc<Mutex<Vec<&'static str>>> = Arc::default();
    let resumed = ChatRecord::find(&db, record.id()).await.unwrap();
    let mut chat = resumed
        .to_llm_with(&db, openai(&server))
        .await
        .unwrap()
        .with_tool(Lookup(executions.clone()));

    assert_eq!(
        resumed.complete(&db, &mut chat).await.unwrap().content(),
        "Answered"
    );

    assert_eq!(*executions.lock().unwrap(), ["lookup"]);
    assert_eq!(
        sent(&server).await,
        [
            vec!["user".to_string(), "Look twice".into()],
            vec!["assistant".into(), String::new()],
            vec!["tool".into(), "found".into(), first.clone()],
            vec!["tool".into(), "found".into(), second.clone()],
        ]
    );
    assert_eq!(
        result_of(&db, &second)
            .await
            .and_then(|m| m.content)
            .as_deref(),
        Some("found")
    );
    assert!(
        messages::Entity::find_by_id(placeholder)
            .one(&db)
            .await
            .unwrap()
            .is_some()
    );
}

// spec: active_record/chat_methods_crash_recovery_spec.rb:64 answers the unfinished call as unfinished once the user moved on
#[tokio::test]
async fn answers_the_unfinished_call_as_unfinished_once_the_user_moved_on() {
    let db = db().await;
    let server = answering_server().await;
    let (record, placeholder, first, second) = crashed_chat(&db, &["Hello?"]).await;
    let executions: Arc<Mutex<Vec<&'static str>>> = Arc::default();
    let resumed = ChatRecord::find(&db, record.id()).await.unwrap();
    let mut chat = resumed
        .to_llm_with(&db, openai(&server))
        .await
        .unwrap()
        .with_tool(Lookup(executions.clone()));

    assert_eq!(
        resumed.complete(&db, &mut chat).await.unwrap().content(),
        "Answered"
    );

    assert!(executions.lock().unwrap().is_empty());
    assert_eq!(
        sent(&server).await,
        [
            vec!["user".to_string(), "Look twice".into()],
            vec!["assistant".into(), String::new()],
            vec!["tool".into(), "found".into(), first.clone()],
            vec![
                "tool".into(),
                r#"{"error":"The tool call did not finish."}"#.into(),
                second.clone()
            ],
            vec!["user".into(), "Hello?".into()],
        ]
    );
    assert!(result_of(&db, &second).await.is_none());
    assert!(
        messages::Entity::find_by_id(placeholder)
            .one(&db)
            .await
            .unwrap()
            .is_some()
    );
}

// ---- chat/tool_concurrency_rails_spec.rb ----------------------------------------------------------

/// `CountChats`: reads the database from inside a concurrent tool call.
struct CountChats(DatabaseConnection);

#[async_trait]
impl Tool for CountChats {
    fn name(&self) -> String {
        "count_chats".into()
    }
    fn description(&self) -> String {
        "Counts chats".into()
    }
    async fn execute(&self, _: Map<String, Value>, _: &ToolCall) -> Result<ToolResult, ToolError> {
        use sea_orm::PaginatorTrait;
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        let count = rust_llm_loco::entities::chats::Entity::find()
            .count(&self.0)
            .await
            .map_err(|e| ToolError::from(e.to_string()))?;
        Ok(count.to_string().into())
    }
}

// spec: chat/tool_concurrency_rails_spec.rb:80 when Rails isolates execution state per fiber > returns the connections a fiber job leases while persisting results
#[tokio::test]
async fn returns_the_connections_concurrent_tools_lease_while_persisting_results() {
    // A small pool, so a connection a tool or a result write kept would starve the next one.
    let mut options = sea_orm::ConnectOptions::new("sqlite::memory:");
    options
        .max_connections(1)
        .acquire_timeout(std::time::Duration::from_secs(5));
    let db = Database::connect(options).await.unwrap();
    let manager = SchemaManager::new(&db);
    for m in migrations() {
        m.up(&manager).await.unwrap();
    }
    rust_llm::configure(|c| {
        c.set("openai_api_key", "test");
    });
    let server = MockServer::start().await;
    let round = Arc::new(Mutex::new(0usize));
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(move |_: &Request| {
            let mut n = round.lock().unwrap();
            *n += 1;
            let output = if *n <= 3 {
                (0..3)
                    .map(|i| json!({ "type": "function_call", "id": format!("fc_{n}_{i}"), "call_id": format!("call_{n}_{i}"),
                                     "name": "count_chats", "arguments": "{}", "status": "completed" }))
                    .collect::<Vec<_>>()
            } else {
                vec![json!({ "type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                             "content": [{ "type": "output_text", "text": "Counted", "annotations": [] }] })]
            };
            ResponseTemplate::new(200).set_body_json(json!({
                "id": format!("resp_{n}"), "object": "response", "status": "completed", "model": MODEL,
                "output": output, "usage": { "input_tokens": 1, "output_tokens": 1 }
            }))
        })
        .mount(&server)
        .await;
    let record = ChatRecord::create(&db, MODEL, None).await.unwrap();
    let mut chat = record
        .to_llm_with(&db, openai(&server))
        .await
        .unwrap()
        .with_tool(CountChats(db.clone()))
        .with_tool_concurrency(true);

    let reply = tokio::time::timeout(
        std::time::Duration::from_secs(30),
        record.ask(&db, &mut chat, "Count the chats three times"),
    )
    .await
    .expect("no connection was kept")
    .unwrap();

    assert_eq!(reply.content(), "Counted");
    let tool_rows = record
        .messages(&db)
        .await
        .unwrap()
        .into_iter()
        .filter(|m| m.role == "tool")
        .count();
    assert_eq!(tool_rows, 9);
}
