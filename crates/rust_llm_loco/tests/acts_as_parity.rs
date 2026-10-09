//! RubyLLM's Active Record specs (`spec/ruby_llm/active_record/*`) on SQLite: the
//! `activerecord_actsas_*` cassettes replayed with request bodies compared against RubyLLM's, and
//! the unit cases for instructions, attachments, cancellation, approvals, MCP input requests,
//! usage linking, cost, the model association, and agents with a persisted chat.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use rust_llm::{Agent, Attachment, Message, Role, Tool, ToolCall, ToolError, ToolResult};
use rust_llm_loco::entities::{
    chats, messages, rust_llm_attachments, rust_llm_models, rust_llm_tool_calls, rust_llm_usages,
};
use rust_llm_loco::{ChatRecord, migrations};
use sea_orm::{
    ActiveModelTrait, ActiveValue::Set, ColumnTrait, Database, DatabaseConnection, EntityTrait,
    PaginatorTrait, QueryFilter,
};
use sea_orm_migration::SchemaManager;
use serde_json::{Map, Value, json};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate, matchers};

const MODEL: &str = "gpt-4.1-nano";

/// `include_context 'with configured RubyLLM'`: a key, so chats that never reach the network
/// can be built from the global configuration.
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

fn fixture(name: &str) -> String {
    format!(
        "{}/../rust_llm/tests/fixtures/{name}",
        env!("CARGO_MANIFEST_DIR")
    )
}

// ---- cassette replay ----------------------------------------------------------------------

#[derive(serde::Deserialize)]
struct Interaction {
    uri: String,
    request_body: String,
    response_body: String,
}

/// Serves a converted RubyLLM cassette in order and records any request whose path or JSON body
/// differs from the recording.
struct Replay {
    interactions: Vec<Interaction>,
    next: Mutex<usize>,
    mismatches: Arc<Mutex<Vec<String>>>,
}

impl Respond for Replay {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let mut next = self.next.lock().unwrap();
        let Some(i) = self.interactions.get(*next) else {
            self.mismatches
                .lock()
                .unwrap()
                .push("unexpected extra request".into());
            return ResponseTemplate::new(599);
        };
        *next += 1;
        if !i.uri.ends_with(request.url.path()) {
            self.mismatches.lock().unwrap().push(format!(
                "path {} != {}",
                request.url.path(),
                i.uri
            ));
        }
        let expected: Value = serde_json::from_str(&i.request_body).unwrap_or(Value::Null);
        let actual: Value = serde_json::from_slice(&request.body).unwrap_or(Value::Null);
        if expected != actual {
            self.mismatches.lock().unwrap().push(format!(
                "body differs:\n  expected {expected}\n  sent     {actual}"
            ));
        }
        ResponseTemplate::new(200)
            .set_body_raw(i.response_body.clone().into_bytes(), "application/json")
    }
}

struct Cassette {
    server: MockServer,
    mismatches: Arc<Mutex<Vec<String>>>,
    count: usize,
}

impl Cassette {
    async fn start(name: &str) -> Cassette {
        let path = format!(
            "{}/../rust_llm/tests/cassettes/{name}.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let interactions: Vec<Interaction> =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let count = interactions.len();
        let server = MockServer::start().await;
        let mismatches = Arc::new(Mutex::new(Vec::new()));
        Mock::given(matchers::any())
            .respond_with(Replay {
                interactions,
                next: Mutex::new(0),
                mismatches: mismatches.clone(),
            })
            .mount(&server)
            .await;
        Cassette {
            server,
            mismatches,
            count,
        }
    }

    fn config(&self) -> Arc<rust_llm::Config> {
        openai_config(&self.server)
    }

    async fn assert_all_matched(&self) {
        let mismatches = self.mismatches.lock().unwrap().clone();
        assert!(
            mismatches.is_empty(),
            "requests differ from RubyLLM's:\n{}",
            mismatches.join("\n")
        );
        assert_eq!(
            self.server.received_requests().await.unwrap().len(),
            self.count
        );
    }
}

fn openai_config(server: &MockServer) -> Arc<rust_llm::Config> {
    let mut c = rust_llm::Config::default();
    c.set("openai_api_base", format!("{}/v1", server.uri()));
    c.set("openai_api_key", "test");
    c.max_retries = 0;
    Arc::new(c)
}

/// `uploaded_file(path, type)`: RubyLLM's spec copies the fixture into a Tempfile, so the upload's
/// name is the tempfile's. The recorded PDF name is used verbatim, since it is part of the body.
fn upload(path: &str, filename: &str, mime: &str) -> Attachment {
    Attachment::from_bytes(std::fs::read(fixture(path)).unwrap(), filename, Some(mime))
}

#[tokio::test]
async fn persists_chat_history() {
    let db = db().await;
    let cassette =
        Cassette::start("activerecord_actsas_basic_chat_functionality_persists_chat_history").await;
    let record = ChatRecord::create(&db, MODEL, None).await.unwrap();
    let mut chat = record.to_llm_with(&db, cassette.config()).await.unwrap();

    record
        .ask(&db, &mut chat, "What's your favorite Ruby feature?")
        .await
        .unwrap();

    let rows = record.messages(&db).await.unwrap();
    assert_eq!(
        rows.iter().map(|m| m.role.as_str()).collect::<Vec<_>>(),
        ["user", "assistant"]
    );
    assert!(!rows[1].content.clone().unwrap_or_default().is_empty());
    cassette.assert_all_matched().await;
}

#[tokio::test]
async fn handles_attachments_in_ask_method() {
    let db = db().await;
    let cassette = Cassette::start(
        "activerecord_actsas_attachment_handling_handles_attachments_in_ask_method",
    )
    .await;
    let record = ChatRecord::create(&db, MODEL, None).await.unwrap();
    let mut chat = record.to_llm_with(&db, cassette.config()).await.unwrap();

    let response = record
        .ask_with(
            &db,
            &mut chat,
            "What do you see?",
            vec![upload("ruby.png", "ruby.png", "image/png")],
        )
        .await
        .unwrap();

    let user = &record.messages(&db).await.unwrap()[0];
    assert_eq!(attachments_of(&db, user.id).await.len(), 1);
    assert!(!response.content().is_empty());
    cassette.assert_all_matched().await;
}

#[tokio::test]
async fn handles_multiple_attachments() {
    let db = db().await;
    let cassette =
        Cassette::start("activerecord_actsas_attachment_handling_handles_multiple_attachments")
            .await;
    let record = ChatRecord::create(&db, MODEL, None).await.unwrap();
    let mut chat = record.to_llm_with(&db, cassette.config()).await.unwrap();
    let files = vec![
        upload("ruby.png", "ruby.png", "image/png"),
        upload(
            "sample.pdf",
            "sample20261007-681054-dscbr5.pdf",
            "application/pdf",
        ),
    ];

    let response = record
        .ask_with(&db, &mut chat, "Analyze these", files)
        .await
        .unwrap();

    let user = &record.messages(&db).await.unwrap()[0];
    let stored = attachments_of(&db, user.id).await;
    assert_eq!(stored.len(), 2);
    assert!(!response.content().is_empty());
    cassette.assert_all_matched().await;

    // Reloaded, the attachments come back with their bytes and types (`message.to_llm`).
    let reloaded = record.to_llm_with(&db, cassette.config()).await.unwrap();
    let restored = &reloaded.messages()[0].attachments;
    assert_eq!(restored.len(), 2);
    assert_eq!(restored[0].mime_type, "image/png");
    assert_eq!(
        restored[0].kind(),
        rust_llm::attachment::AttachmentType::Image
    );
    assert_eq!(
        restored[1].kind(),
        rust_llm::attachment::AttachmentType::Pdf
    );
    assert_eq!(
        restored[1].filename.as_deref(),
        Some("sample20261007-681054-dscbr5.pdf")
    );
    let mut first = restored[0].clone();
    assert_eq!(
        first.content().await.unwrap(),
        std::fs::read(fixture("ruby.png")).unwrap()
    );
    assert_eq!(
        stored[0].byte_size,
        std::fs::metadata(fixture("ruby.png")).unwrap().len() as i64
    );
}

async fn attachments_of(
    db: &DatabaseConnection,
    message_id: i32,
) -> Vec<rust_llm_attachments::Model> {
    rust_llm_attachments::Entity::find()
        .filter(rust_llm_attachments::Column::MessageId.eq(message_id as i64))
        .all(db)
        .await
        .unwrap()
}

// ---- a stubbed model -----------------------------------------------------------------------

fn responses_text(text: &str) -> Value {
    json!({
        "id": "resp_1", "object": "response", "status": "completed", "model": MODEL,
        "output": [{ "type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                     "content": [{ "type": "output_text", "text": text, "annotations": [] }] }],
        "usage": { "input_tokens": 8, "output_tokens": 3, "total_tokens": 11 }
    })
}

/// Answers with `replies` in order, then repeats the last.
async fn stub(replies: Vec<(u16, Value)>) -> MockServer {
    let server = MockServer::start().await;
    let n = replies.len();
    for (i, (status, body)) in replies.into_iter().enumerate() {
        let mock = Mock::given(matchers::method("POST"))
            .respond_with(ResponseTemplate::new(status).set_body_json(body))
            .with_priority((i + 1) as u8);
        if i + 1 < n {
            mock.up_to_n_times(1).mount(&server).await
        } else {
            mock.mount(&server).await
        }
    }
    server
}

async fn roles(record: &ChatRecord, db: &DatabaseConnection) -> Vec<(String, Option<String>)> {
    record
        .messages(db)
        .await
        .unwrap()
        .into_iter()
        .map(|m| (m.role, m.content))
        .collect()
}

fn contents(chat: &rust_llm::Chat) -> Vec<String> {
    chat.messages()
        .iter()
        .map(|m| m.content().to_string())
        .collect()
}

// ---- chat_methods_spec: #with_instructions --------------------------------------------------

#[tokio::test]
async fn with_instructions_replaces_the_persisted_system_message_and_keeps_its_place() {
    let db = db().await;
    let mut record = ChatRecord::create(&db, MODEL, None).await.unwrap();
    let mut chat = record.to_llm(&db).await.unwrap();

    record
        .with_instructions(&db, &mut chat, "Be concise")
        .await
        .unwrap();
    let system_id = record.messages(&db).await.unwrap()[0].id;
    record
        .add_message(&db, &mut chat, Message::user("Hello"))
        .await
        .unwrap();
    record
        .with_instructions(&db, &mut chat, "Be concise")
        .await
        .unwrap();
    record
        .with_instructions(&db, &mut chat, "Be terse")
        .await
        .unwrap();

    let rows = record.messages(&db).await.unwrap();
    assert_eq!(
        rows[0].id, system_id,
        "the system row stays ahead of the conversation"
    );
    let expected = [("system", "Be terse"), ("user", "Hello")];
    assert_eq!(
        rows.iter()
            .map(|m| (m.role.as_str(), m.content.as_deref().unwrap()))
            .collect::<Vec<_>>(),
        expected
    );
    assert_eq!(contents(&chat), ["Be terse", "Hello"]);

    // `appends when asked`
    record
        .set_instructions(&db, &mut chat, Some("Cite sources"), true, true, false)
        .await
        .unwrap();
    let systems: Vec<String> = record
        .messages(&db)
        .await
        .unwrap()
        .into_iter()
        .filter(|m| m.role == "system")
        .filter_map(|m| m.content)
        .collect();
    assert_eq!(systems, ["Be terse", "Cite sources"]);

    // `clears the persisted system messages when given nil`
    record
        .set_instructions(&db, &mut chat, None, false, true, false)
        .await
        .unwrap();
    assert!(
        record
            .messages(&db)
            .await
            .unwrap()
            .iter()
            .all(|m| m.role != "system")
    );
    assert_eq!(contents(&chat), ["Hello"]);
}

#[tokio::test]
async fn an_appended_instruction_on_a_reloaded_chat_is_not_duplicated() {
    let db = db().await;
    let record = ChatRecord::create(&db, MODEL, None).await.unwrap();
    let mut chat = record.to_llm(&db).await.unwrap();
    record
        .with_instructions(&db, &mut chat, "Be concise")
        .await
        .unwrap();

    let mut reloaded = ChatRecord::find(&db, record.id()).await.unwrap();
    let mut chat = reloaded.to_llm(&db).await.unwrap();
    reloaded
        .set_instructions(&db, &mut chat, Some("Cite sources"), true, true, false)
        .await
        .unwrap();
    assert_eq!(contents(&chat), ["Be concise", "Cite sources"]);
}

#[tokio::test]
async fn runtime_instructions_apply_without_persisting_and_survive_a_reload() {
    let db = db().await;
    let mut record = ChatRecord::create(&db, MODEL, None).await.unwrap();
    let mut chat = record.to_llm(&db).await.unwrap();

    // `applies instructions without persisting them` / `appends runtime instructions`
    record
        .set_instructions(
            &db,
            &mut chat,
            Some("Answer in French"),
            false,
            false,
            false,
        )
        .await
        .unwrap();
    record
        .set_instructions(&db, &mut chat, Some("Be brief"), true, false, false)
        .await
        .unwrap();
    assert!(record.messages(&db).await.unwrap().is_empty());
    assert_eq!(contents(&chat), ["Answer in French", "Be brief"]);

    // `survives a reload`
    record.reload(&db, &mut chat).await.unwrap();
    assert_eq!(contents(&chat), ["Answer in French", "Be brief"]);

    // `drops them when given nil`
    record
        .set_instructions(&db, &mut chat, None, false, false, false)
        .await
        .unwrap();
    assert!(chat.messages().is_empty());
}

#[tokio::test]
async fn runtime_instructions_append_to_persisted_ones_and_keep_their_cache_boundary() {
    let db = db().await;
    let mut record = ChatRecord::create(&db, MODEL, None).await.unwrap();
    let mut chat = record.to_llm(&db).await.unwrap();
    record
        .with_instructions(&db, &mut chat, "Be concise")
        .await
        .unwrap();
    record
        .set_instructions(&db, &mut chat, Some("Answer in French"), true, false, false)
        .await
        .unwrap();
    assert_eq!(contents(&chat), ["Be concise", "Answer in French"]);
    record.reload(&db, &mut chat).await.unwrap();
    assert_eq!(contents(&chat), ["Be concise", "Answer in French"]);

    // `keeps a runtime cache boundary through a reload`: a runtime replace drops the persisted one
    // from the request only.
    record
        .set_instructions(&db, &mut chat, Some("Stable policy"), false, false, true)
        .await
        .unwrap();
    record
        .set_instructions(&db, &mut chat, Some("Current context"), true, false, false)
        .await
        .unwrap();
    assert_eq!(
        chat.messages()
            .iter()
            .map(|m| m.cache_until_here)
            .collect::<Vec<_>>(),
        [true, false]
    );
    assert!(
        record
            .messages(&db)
            .await
            .unwrap()
            .iter()
            .all(|m| !m.cache_until_here)
    );
    record.reload(&db, &mut chat).await.unwrap();
    assert_eq!(contents(&chat), ["Stable policy", "Current context"]);
    assert_eq!(
        chat.messages()
            .iter()
            .map(|m| m.cache_until_here)
            .collect::<Vec<_>>(),
        [true, false]
    );
}

#[tokio::test]
async fn persisted_instructions_keep_a_cache_boundary() {
    let db = db().await;
    let mut record = ChatRecord::create(&db, MODEL, None).await.unwrap();
    let mut chat = record.to_llm(&db).await.unwrap();
    record
        .set_instructions(&db, &mut chat, Some("Stable policy"), false, true, true)
        .await
        .unwrap();
    assert!(record.messages(&db).await.unwrap()[0].cache_until_here);
    assert!(chat.messages()[0].cache_until_here);
}

#[tokio::test]
async fn runtime_instructions_are_sent_but_never_written() {
    let db = db().await;
    let server = stub(vec![(200, responses_text("Bonjour"))]).await;
    let mut record = ChatRecord::create(&db, MODEL, None).await.unwrap();
    let mut chat = record
        .to_llm_with(&db, openai_config(&server))
        .await
        .unwrap();
    record
        .set_instructions(
            &db,
            &mut chat,
            Some("Answer in French"),
            false,
            false,
            false,
        )
        .await
        .unwrap();
    record.ask(&db, &mut chat, "Hello").await.unwrap();

    let sent: Value =
        serde_json::from_slice(&server.received_requests().await.unwrap()[0].body).unwrap();
    assert_eq!(
        sent["instructions"]
            .as_str()
            .or(sent["input"][0]["content"].as_str()),
        Some("Answer in French")
    );
    let rows = roles(&record, &db).await;
    assert_eq!(
        rows,
        [
            ("user".into(), Some("Hello".into())),
            ("assistant".into(), Some("Bonjour".into()))
        ]
    );
}

// ---- #add_message / #cache_until_here -------------------------------------------------------

#[tokio::test]
async fn add_message_links_a_tool_result_to_its_call() {
    let db = db().await;
    let record = ChatRecord::create(&db, MODEL, None).await.unwrap();
    let mut chat = record.to_llm(&db).await.unwrap();
    let call = ToolCall::new("call_1", "lookup", Map::new());
    let mut assistant = Message::new(Role::Assistant, Some(String::new()));
    assistant.tool_calls = Some([(call.id.clone(), call)].into_iter().collect());
    record.add_message(&db, &mut chat, assistant).await.unwrap();

    let result = record
        .add_message(&db, &mut chat, Message::tool_result("call_1", "done"))
        .await
        .unwrap();

    let row = rust_llm_tool_calls::Entity::find()
        .one(&db)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(row.result_id, Some(result.id as i64));
}

#[tokio::test]
async fn cache_until_here_marks_the_last_persisted_message_or_the_in_memory_one() {
    let db = db().await;
    let record = ChatRecord::create(&db, MODEL, None).await.unwrap();
    let mut chat = record.to_llm(&db).await.unwrap();
    let err = record.cache_until_here(&db, &mut chat).await.unwrap_err();
    assert_eq!(err.to_string(), "No messages to cache");

    chat.add_message(Message::user("Reusable prompt"));
    record.cache_until_here(&db, &mut chat).await.unwrap();
    assert!(chat.messages().last().unwrap().cache_until_here);

    let other = ChatRecord::create(&db, MODEL, None).await.unwrap();
    let mut chat = other.to_llm(&db).await.unwrap();
    other
        .add_message(&db, &mut chat, Message::user("Reusable prompt"))
        .await
        .unwrap();
    other.cache_until_here(&db, &mut chat).await.unwrap();
    assert!(other.messages(&db).await.unwrap()[0].cache_until_here);
}

// ---- accounting / usage persistence ---------------------------------------------------------

async fn usage_row(
    db: &DatabaseConnection,
    chat_id: i32,
    message_id: Option<i32>,
    status: &str,
    tokens: (Option<i32>, Option<i32>),
    total: Option<f64>,
) {
    rust_llm_usages::ActiveModel {
        chat_type: Set("Chat".into()),
        chat_id: Set(chat_id as i64),
        message_type: Set(message_id.map(|_| "Message".into())),
        message_id: Set(message_id.map(i64::from)),
        operation: Set("chat".into()),
        provider: Set("openai".into()),
        model: Set(MODEL.into()),
        status: Set(status.into()),
        input_tokens: Set(tokens.0),
        output_tokens: Set(tokens.1),
        total_cost: Set(total),
        created_at: Set(chrono::Utc::now().into()),
        updated_at: Set(chrono::Utc::now().into()),
        ..Default::default()
    }
    .insert(db)
    .await
    .unwrap();
}

#[tokio::test]
async fn aggregates_tokens_and_cost_across_persisted_usage_entries() {
    let db = db().await;
    let record = ChatRecord::create(&db, MODEL, None).await.unwrap();
    // `reports an empty cost for a chat that never ran`
    assert_eq!(record.total_cost(&db).await.unwrap(), None);
    usage_row(
        &db,
        record.id(),
        None,
        "succeeded",
        (Some(10), Some(20)),
        Some(0.3),
    )
    .await;
    let tokens = record.tokens(&db).await.unwrap();
    assert_eq!((tokens.input, tokens.output), (Some(10), Some(20)));
    assert!((record.total_cost(&db).await.unwrap().unwrap() - 0.3).abs() < 1e-4);
}

#[tokio::test]
async fn uses_a_stored_exact_cost_when_token_counts_were_unavailable() {
    let db = db().await;
    let record = ChatRecord::create(&db, MODEL, None).await.unwrap();
    let mut chat = record.to_llm(&db).await.unwrap();
    let message = record
        .add_message(&db, &mut chat, Message::assistant("done"))
        .await
        .unwrap();
    usage_row(
        &db,
        record.id(),
        Some(message.id),
        "succeeded",
        (None, None),
        Some(0.0042),
    )
    .await;
    assert_eq!(record.total_cost(&db).await.unwrap(), Some(0.0042));
    let reloaded = record.to_llm(&db).await.unwrap();
    assert_eq!(reloaded.messages()[0].cost(None).total(), Some(0.0042));
}

#[tokio::test]
async fn does_not_reprice_a_persisted_entry_whose_cost_was_unavailable() {
    let db = db().await;
    let record = ChatRecord::create(&db, MODEL, None).await.unwrap();
    usage_row(&db, record.id(), None, "succeeded", (Some(10), None), None).await;
    let cost = record.cost(&db).await.unwrap();
    assert_eq!(cost.input, None);
    assert_eq!(cost.total(), None);
}

#[tokio::test]
async fn reloads_linked_and_unlinked_entries_chronologically() {
    let db = db().await;
    let record = ChatRecord::create(&db, MODEL, None).await.unwrap();
    let mut chat = record.to_llm(&db).await.unwrap();
    let message = record
        .add_message(&db, &mut chat, Message::assistant("done"))
        .await
        .unwrap();
    usage_row(
        &db,
        record.id(),
        Some(message.id),
        "succeeded",
        (Some(4), Some(2)),
        None,
    )
    .await;
    usage_row(&db, record.id(), None, "cancelled", (None, None), None).await;

    let chat = record.to_llm(&db).await.unwrap();
    let statuses: Vec<_> = chat.usage_entries().iter().map(|e| e.status).collect();
    assert_eq!(
        statuses,
        [
            rust_llm::UsageStatus::Succeeded,
            rust_llm::UsageStatus::Cancelled
        ]
    );
    let linked = &chat.messages()[0].usage_entries;
    assert_eq!(linked.len(), 1);
    assert_eq!(
        linked[0].id,
        chat.usage_entries()[0].id,
        "the message and the ledger share one entry"
    );
}

#[tokio::test]
async fn persists_attempts_independently_and_links_them_to_the_resulting_message() {
    let db = db().await;
    let server = stub(vec![
        (500, json!({ "error": { "message": "retry" } })),
        (200, responses_text("Hello")),
    ])
    .await;
    let mut config = (*openai_config(&server)).clone();
    config.max_retries = 1;
    config.retry_interval = 0.0;
    config.retry_interval_randomness = 0.0;
    let record = ChatRecord::create(&db, MODEL, None).await.unwrap();
    let mut chat = record.to_llm_with(&db, Arc::new(config)).await.unwrap();

    record.ask(&db, &mut chat, "Hello").await.unwrap();

    let usages = record.usages(&db).await.unwrap();
    let rows = record.messages(&db).await.unwrap();
    assert_eq!(
        usages.iter().map(|u| u.status.as_str()).collect::<Vec<_>>(),
        ["failed", "succeeded"]
    );
    assert!(
        usages
            .iter()
            .all(|u| u.message_id == Some(rows[1].id as i64))
    );
    let tokens = record.tokens(&db).await.unwrap();
    assert_eq!((tokens.input, tokens.output), (Some(8), Some(3)));
}

// ---- model association ----------------------------------------------------------------------

#[tokio::test]
async fn belongs_to_rust_llms_internal_model_record() {
    let db = db().await;
    let record = ChatRecord::create(&db, MODEL, None).await.unwrap();
    let model = record.model(&db).await.unwrap();
    assert_eq!(
        (model.model_id.as_str(), model.provider.as_str()),
        (MODEL, "openai")
    );
    assert_eq!(model.name, "GPT-4.1 nano");
    assert_eq!(
        model.model_created_at.map(|t| t.to_rfc3339()),
        Some("2025-04-14T00:00:00+00:00".into())
    );

    // One row per provider and model, reused by every chat.
    ChatRecord::create(&db, MODEL, None).await.unwrap();
    let rows = rust_llm_models::Entity::find()
        .filter(rust_llm_models::Column::ModelId.eq(MODEL))
        .filter(rust_llm_models::Column::Provider.eq("openai"))
        .all(&db)
        .await
        .unwrap();
    assert_eq!(rows.len(), 1);
}

#[tokio::test]
async fn switches_models_with_with_model() {
    let db = db().await;
    let mut record = ChatRecord::create(&db, MODEL, None).await.unwrap();
    let chat = record.to_llm(&db).await.unwrap();
    let chat = record
        .with_model(&db, chat, "gpt-4o-mini", Some("openai"))
        .await
        .unwrap();
    assert_eq!(record.model(&db).await.unwrap().model_id, "gpt-4o-mini");
    assert_eq!(
        ChatRecord::find(&db, record.id())
            .await
            .unwrap()
            .model(&db)
            .await
            .unwrap()
            .model_id,
        "gpt-4o-mini"
    );
    assert_eq!(chat.model().id, "gpt-4o-mini");
}

#[tokio::test]
async fn accepts_an_unregistered_id_only_when_told_the_model_exists() {
    let db = db().await;
    let mut record = ChatRecord::create(&db, MODEL, None).await.unwrap();
    let chat = record.to_llm(&db).await.unwrap();
    assert!(
        record
            .with_model(&db, chat, "made-up-deployment", Some("openai"))
            .await
            .is_err()
    );

    record.assume_model_exists = true;
    let chat = record.to_llm(&db).await.unwrap();
    let chat = record
        .with_model(&db, chat, "made-up-deployment", Some("openai"))
        .await
        .unwrap();
    assert_eq!(
        record.model(&db).await.unwrap().model_id,
        "made-up-deployment"
    );
    assert_eq!(chat.model().id, "made-up-deployment");

    // `requires a provider when assuming the model exists`
    let err = ChatRecord::create_with(&db, "made-up-deployment", None, true)
        .await
        .unwrap_err();
    assert_eq!(
        err.to_string(),
        "Provider must be specified if assume_model_exists is true"
    );
}

// ---- cancellation ---------------------------------------------------------------------------

#[tokio::test]
async fn cancel_persists_the_request_on_the_row() {
    let db = db().await;
    let record = ChatRecord::create(&db, MODEL, None).await.unwrap();
    record.cancel(&db).await.unwrap();
    assert!(record.is_cancelled(&db).await.unwrap());
    assert!(
        chats::Entity::find_by_id(record.id())
            .one(&db)
            .await
            .unwrap()
            .unwrap()
            .cancelled
    );
}

/// A `complete` that finds a request another process wrote stops with `Cancelled`, clears the
/// column, and keeps the chat usable.
#[tokio::test]
async fn consumes_a_request_written_straight_to_the_row() {
    let db = db().await;
    let server = stub(vec![(200, responses_text("Hello"))]).await;
    let record = ChatRecord::create(&db, MODEL, None).await.unwrap();
    let mut chat = record
        .to_llm_with(&db, openai_config(&server))
        .await
        .unwrap();
    ChatRecord::find(&db, record.id())
        .await
        .unwrap()
        .cancel(&db)
        .await
        .unwrap();

    let err = record.ask(&db, &mut chat, "Hello").await.unwrap_err();
    assert!(
        matches!(err, rust_llm_loco::Error::Llm(rust_llm::Error::Cancelled)),
        "{err:?}"
    );
    assert!(
        !record.is_cancelled(&db).await.unwrap(),
        "the request is consumed"
    );
    assert!(server.received_requests().await.unwrap().is_empty());

    record.complete(&db, &mut chat).await.unwrap();
    assert_eq!(
        roles(&record, &db).await.last().unwrap().1.as_deref(),
        Some("Hello")
    );
}

/// `polls the row at most once per interval` / `notices a request another process wrote once
/// the interval elapses`: a request written while the model is answering stops the run.
#[tokio::test]
async fn notices_a_request_another_process_wrote_while_running() {
    let db = db().await;
    let server = MockServer::start().await;
    Mock::given(matchers::method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(responses_text("late"))
                .set_delay(std::time::Duration::from_millis(1500)),
        )
        .mount(&server)
        .await;
    let record = ChatRecord::create(&db, MODEL, None).await.unwrap();
    let mut chat = record
        .to_llm_with(&db, openai_config(&server))
        .await
        .unwrap();
    let other = ChatRecord::find(&db, record.id()).await.unwrap();
    let db2 = db.clone();
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        other.cancel(&db2).await.unwrap();
    });

    let err = record.ask(&db, &mut chat, "Hello").await.unwrap_err();
    assert!(
        matches!(err, rust_llm_loco::Error::Llm(rust_llm::Error::Cancelled)),
        "{err:?}"
    );
    // `keeps usage when a cancelled stream produces no message`: the attempt is billed, unlinked.
    let usages = record.usages(&db).await.unwrap();
    assert_eq!(usages.len(), 1);
    assert_eq!(usages[0].message_id, None);
    assert_eq!(
        roles(&record, &db)
            .await
            .iter()
            .map(|r| r.0.as_str())
            .collect::<Vec<_>>(),
        ["user"]
    );
    assert!(!record.is_cancelled(&db).await.unwrap());
}

// ---- approvals ------------------------------------------------------------------------------

struct Dangerous;

#[async_trait]
impl Tool for Dangerous {
    fn name(&self) -> String {
        "dangerous".into()
    }
    fn description(&self) -> String {
        "Does something dangerous".into()
    }
    fn requires_approval(&self) -> bool {
        true
    }
    async fn execute(&self, _: Map<String, Value>, _: &ToolCall) -> Result<ToolResult, ToolError> {
        Ok("done".into())
    }
}

/// `parked_chat`: an assistant message calling `dangerous`, persisted with `add_message`.
async fn parked_chat(
    db: &DatabaseConnection,
    server: Option<&MockServer>,
) -> (ChatRecord, rust_llm::Chat) {
    let record = ChatRecord::create(db, MODEL, None).await.unwrap();
    let chat = match server {
        Some(s) => record.to_llm_with(db, openai_config(s)).await.unwrap(),
        None => record.to_llm(db).await.unwrap(),
    };
    let mut chat = chat.with_tool(Dangerous);
    let mut assistant = Message::new(Role::Assistant, Some(String::new()));
    assistant.tool_calls = Some(
        [(
            "call_1".to_string(),
            ToolCall::new("call_1", "dangerous", Map::new()),
        )]
        .into_iter()
        .collect(),
    );
    record.add_message(db, &mut chat, assistant).await.unwrap();
    (record, chat)
}

#[tokio::test]
async fn returns_the_persisted_tool_call_rows_awaiting_a_decision() {
    let db = db().await;
    let (record, mut chat) = parked_chat(&db, None).await;
    let rows = record.pending_approvals(&db, &mut chat).await.unwrap();
    assert_eq!(
        rows.iter()
            .map(|r| r.tool_call_id.as_str())
            .collect::<Vec<_>>(),
        ["call_1"]
    );
    assert!(record.is_awaiting_approval(&db, &mut chat).await.unwrap());

    // `persists an approval recorded from a pending record`
    record
        .approve(&db, &mut chat, &rows[0].tool_call_id)
        .await
        .unwrap();
    assert!(
        record
            .pending_approvals(&db, &mut chat)
            .await
            .unwrap()
            .is_empty()
    );
    assert!(!record.is_awaiting_approval(&db, &mut chat).await.unwrap());
    assert_eq!(
        rust_llm_tool_calls::Entity::find()
            .one(&db)
            .await
            .unwrap()
            .unwrap()
            .approval
            .as_deref(),
        Some("approved")
    );
}

/// `reads a decision persisted by another process`, including into a chat already built:
/// RubyLLM's `approval_checker` rereads the row on every check.
#[tokio::test]
async fn reads_a_decision_another_process_persisted_into_a_built_chat() {
    let db = db().await;
    let server = stub(vec![(200, responses_text("ok"))]).await;
    let (record, mut chat) = parked_chat(&db, Some(&server)).await;
    assert!(chat.is_awaiting_approval());

    let row = rust_llm_tool_calls::Entity::find()
        .one(&db)
        .await
        .unwrap()
        .unwrap();
    let mut row: rust_llm_tool_calls::ActiveModel = row.into();
    row.approval = Set(Some("denied".into()));
    row.update(&db).await.unwrap();

    assert!(!record.is_awaiting_approval(&db, &mut chat).await.unwrap());
    assert!(
        record
            .pending_approvals(&db, &mut chat)
            .await
            .unwrap()
            .is_empty()
    );
    record.complete(&db, &mut chat).await.unwrap();
    let tool = record
        .messages(&db)
        .await
        .unwrap()
        .into_iter()
        .find(|m| m.role == "tool")
        .unwrap();
    assert!(
        tool.content
            .unwrap()
            .contains("The user denied the dangerous tool call.")
    );
}

#[tokio::test]
async fn refuses_to_persist_a_new_question_while_the_round_is_parked() {
    let db = db().await;
    let (record, mut chat) = parked_chat(&db, None).await;
    let err = record
        .ask_later(&db, &mut chat, "Write an essay instead")
        .await
        .unwrap_err();
    assert!(
        matches!(
            err,
            rust_llm_loco::Error::Llm(rust_llm::Error::PendingToolCalls(_))
        ),
        "{err:?}"
    );
    assert!(
        record
            .messages(&db)
            .await
            .unwrap()
            .iter()
            .all(|m| m.role != "user")
    );
}

#[tokio::test]
async fn records_a_denial_and_rejects_an_unknown_tool_call() {
    let db = db().await;
    let (record, mut chat) = parked_chat(&db, None).await;
    record.deny(&db, &mut chat, "call_1").await.unwrap();
    assert_eq!(
        rust_llm_tool_calls::Entity::find()
            .one(&db)
            .await
            .unwrap()
            .unwrap()
            .approval
            .as_deref(),
        Some("denied")
    );
    let err = record
        .approve(&db, &mut chat, "call_missing")
        .await
        .unwrap_err();
    assert_eq!(err.to_string(), "Unknown tool call: \"call_missing\"");
}

// ---- persisted MCP input requests (acts_as_mcp_input_spec) ----------------------------------

fn files() -> rust_llm::Mcp {
    let server = format!(
        "{}/../rust_llm/tests/fixtures/mcp/server.rb",
        env!("CARGO_MANIFEST_DIR")
    );
    rust_llm::Mcp::command(["ruby".to_string(), server])
        .name("files")
        .build()
        .unwrap()
}

async fn paused_chat(
    db: &DatabaseConnection,
    server: &MockServer,
    files: &rust_llm::Mcp,
) -> ChatRecord {
    let record = ChatRecord::create(db, MODEL, None).await.unwrap();
    let mut chat = record
        .to_llm_with(db, openai_config(server))
        .await
        .unwrap()
        .with_mcp(files.clone());
    let mut assistant = Message::new(Role::Assistant, Some(String::new()));
    assistant.tool_calls = Some(
        [(
            "call_1".to_string(),
            ToolCall::new("call_1", "deploy", Map::new()),
        )]
        .into_iter()
        .collect(),
    );
    record.add_message(db, &mut chat, assistant).await.unwrap();
    record.complete(db, &mut chat).await.unwrap();
    record
}

async fn deploy_call(db: &DatabaseConnection) -> rust_llm_tool_calls::Model {
    rust_llm_tool_calls::Entity::find()
        .filter(rust_llm_tool_calls::Column::ToolCallId.eq("call_1"))
        .one(db)
        .await
        .unwrap()
        .unwrap()
}

#[tokio::test]
async fn persists_the_input_a_paused_tool_call_waits_on() {
    let db = db().await;
    let server = stub(vec![(200, responses_text("Done"))]).await;
    let files = files();
    let record = paused_chat(&db, &server, &files).await;

    let chat = record
        .to_llm_with(&db, openai_config(&server))
        .await
        .unwrap()
        .with_mcp(files.clone());
    assert!(chat.is_awaiting_input());
    assert_eq!(
        deploy_call(&db).await.pending_input.unwrap()["request_state"],
        "environment-state"
    );
    assert_eq!(
        chat.pending_inputs()[0].message.as_deref(),
        Some("Which environment?")
    );
    files.close().await;
}

#[tokio::test]
async fn resumes_from_another_process_after_the_user_answers() {
    let db = db().await;
    let server = stub(vec![(200, responses_text("Done"))]).await;
    let files = files();
    let record = paused_chat(&db, &server, &files).await;

    let answering = ChatRecord::find(&db, record.id()).await.unwrap();
    let mut chat = answering
        .to_llm_with(&db, openai_config(&server))
        .await
        .unwrap()
        .with_mcp(files.clone());
    let request = chat.pending_inputs().remove(0);
    answering
        .answer(
            &db,
            &mut chat,
            &request,
            json!({ "environment": "staging" })
                .as_object()
                .unwrap()
                .clone(),
        )
        .await
        .unwrap();
    drop(chat);

    let resumed = ChatRecord::find(&db, record.id()).await.unwrap();
    let mut chat = resumed
        .to_llm_with(&db, openai_config(&server))
        .await
        .unwrap()
        .with_mcp(files.clone());
    resumed.complete(&db, &mut chat).await.unwrap();

    let tool = resumed
        .messages(&db)
        .await
        .unwrap()
        .into_iter()
        .find(|m| m.role == "tool")
        .unwrap();
    assert_eq!(tool.content.as_deref(), Some("Deployed to staging"));
    assert_eq!(deploy_call(&db).await.pending_input, None);
    assert!(!chat.is_awaiting_input());
    files.close().await;
}

// ---- round-trips through the rows ------------------------------------------------------------

#[tokio::test]
async fn round_trips_server_tool_calls_thinking_citations_and_raw_content() {
    let db = db().await;
    let record = ChatRecord::create(&db, MODEL, None).await.unwrap();
    let mut chat = record.to_llm(&db).await.unwrap();
    let raw_block = json!({ "type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": { "query": "ruby" } });
    let mut m = Message::assistant("Found it.");
    m.server_tool_calls = vec![serde_json::from_value(json!({
        "type": "server_tool_use", "name": "web_search", "id": "srvtoolu_1", "input": { "query": "ruby" }, "result": null, "raw": raw_block
    }))
    .unwrap()];
    m.raw_content = Some(json!([raw_block, { "type": "text", "text": "Found it." }]));
    m.thinking = rust_llm::Thinking::build(Some(String::new()), Some("sig".into()));
    m.citations = vec![
        serde_json::from_value(json!({ "url": "https://example.test", "source_id": "file_facts" }))
            .unwrap(),
    ];
    m.finish_reason = Some(rust_llm::FinishReason::Stop);
    record.add_message(&db, &mut chat, m).await.unwrap();

    let restored = &record.to_llm(&db).await.unwrap().messages()[0].clone();
    assert_eq!(restored.server_tool_calls[0].kind, "server_tool_use");
    assert_eq!(restored.server_tool_calls[0].raw, raw_block);
    assert_eq!(
        restored
            .raw_content
            .as_ref()
            .unwrap()
            .as_array()
            .unwrap()
            .len(),
        2
    );
    assert_eq!(
        restored.thinking.as_ref().unwrap().text.as_deref(),
        Some("")
    );
    assert_eq!(
        restored.thinking.as_ref().unwrap().signature.as_deref(),
        Some("sig")
    );
    assert_eq!(
        restored.citations[0].source_id.as_deref(),
        Some("file_facts")
    );
    assert_eq!(restored.finish_reason, Some(rust_llm::FinishReason::Stop));
}

#[tokio::test]
async fn a_generated_assistant_attachment_is_stored_and_reloaded() {
    let db = db().await;
    let record = ChatRecord::create(&db, MODEL, None).await.unwrap();
    let mut chat = record.to_llm(&db).await.unwrap();
    let bytes = std::fs::read(fixture("ruby.png")).unwrap();
    let m = Message::new(Role::Assistant, None).with_attachments(vec![Attachment::from_bytes(
        bytes.clone(),
        "gemini_attachment_1.png",
        Some("image/png"),
    )]);
    let row = record.add_message(&db, &mut chat, m).await.unwrap();

    assert_eq!(row.content, None);
    let stored = attachments_of(&db, row.id).await;
    assert_eq!(
        (stored[0].filename.as_str(), stored[0].content_type.as_str()),
        ("gemini_attachment_1.png", "image/png")
    );
    assert_eq!(stored[0].data, bytes);
}

// ---- agents with a persisted chat (agent_rails_spec) ----------------------------------------

struct Assistant {
    instructions: Option<String>,
}

impl Agent for Assistant {
    fn model(&self) -> Option<&str> {
        Some(MODEL)
    }
    fn instructions(&self) -> Option<String> {
        self.instructions.clone()
    }
}

#[tokio::test]
async fn an_agent_creates_a_chat_with_persisted_instructions_and_find_reapplies_them() {
    let db = db().await;
    let agent = Assistant {
        instructions: Some("chat-class: Chat".into()),
    };
    let (created, chat) = ChatRecord::create_for_agent(&db, &agent).await.unwrap();
    let systems: Vec<_> = created
        .messages(&db)
        .await
        .unwrap()
        .into_iter()
        .filter(|m| m.role == "system")
        .collect();
    assert_eq!(systems.len(), 1);
    assert_eq!(systems[0].content.as_deref(), Some("chat-class: Chat"));
    assert_eq!(contents(&chat), ["chat-class: Chat"]);

    let (found, chat) = ChatRecord::find_for_agent(&db, created.id(), &agent)
        .await
        .unwrap();
    assert_eq!(contents(&chat), ["chat-class: Chat"]);
    assert_eq!(
        found.messages(&db).await.unwrap().len(),
        1,
        "find does not rewrite history"
    );
}

#[tokio::test]
async fn an_agent_without_instructions_adds_no_system_message() {
    let db = db().await;
    for instructions in [None, Some("  \n".to_string())] {
        let (record, _) = ChatRecord::create_for_agent(&db, &Assistant { instructions })
            .await
            .unwrap();
        assert!(record.messages(&db).await.unwrap().is_empty());
    }
}

// ---- render with hooks through a record ------------------------------------------------------

#[tokio::test]
async fn renders_the_payload_with_before_request_hooks_applied() {
    let db = db().await;
    let record = ChatRecord::create(&db, MODEL, None).await.unwrap();
    let mut chat = record
        .to_llm(&db)
        .await
        .unwrap()
        .before_request(|payload| payload["metadata"] = json!({ "user_id": "u-1" }));
    record.ask_later(&db, &mut chat, "Hello").await.unwrap();
    assert_eq!(
        chat.render().unwrap()["metadata"],
        json!({ "user_id": "u-1" })
    );
    assert_eq!(
        messages::Entity::find().all(&db).await.unwrap().len(),
        1,
        "ask_later persists the user message"
    );
}

// spec: active_record/acts_as_spec.rb:103 persists each attempt before publishing its usage event
#[tokio::test]
async fn persists_each_attempt_before_publishing_its_usage_event() {
    // A file-backed database, so the probe below can read on its own connection while the chat
    // holds one (the shared `sqlite::memory:` pool has a single connection).
    let path = std::env::temp_dir().join(format!(
        "rust_llm_usage_order_{}.sqlite",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&path);
    let db = Database::connect(format!("sqlite://{}?mode=rwc", path.display()))
        .await
        .unwrap();
    let manager = SchemaManager::new(&db);
    for m in migrations() {
        m.up(&manager).await.unwrap();
    }
    let probe_url = format!("sqlite://{}?mode=ro", path.display());
    let server = MockServer::start().await;
    // One failed attempt (retried), then the answer: two usage events, like Ruby's tracker run.
    Mock::given(matchers::method("POST"))
        .respond_with(
            ResponseTemplate::new(500).set_body_json(json!({ "error": { "message": "retry" } })),
        )
        .up_to_n_times(1)
        .with_priority(1)
        .mount(&server)
        .await;
    Mock::given(matchers::method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "resp_1", "object": "response", "status": "completed", "model": MODEL,
            "output": [{ "type": "message", "role": "assistant",
                         "content": [{ "type": "output_text", "text": "Hello" }] }],
            "usage": { "input_tokens": 8, "output_tokens": 3 }
        })))
        .with_priority(2)
        .mount(&server)
        .await;

    let record = ChatRecord::create(&db, MODEL, None).await.unwrap();
    let observed: Arc<Mutex<Vec<u64>>> = Arc::default();
    let (sink, chat_id) = (observed.clone(), record.id());
    let mut config = (*openai_config(&server)).clone();
    config.max_retries = 1;
    config.retry_interval = 0.001;
    // The instrumenter counts the chat's persisted usage rows at the moment each event fires.
    config.instrumenter = Some(Arc::new(move |name: &str, _: &Map<String, Value>, _| {
        if name == "usage.rust_llm" {
            let (url, sink) = (probe_url.clone(), sink.clone());
            let count = std::thread::spawn(move || {
                tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .unwrap()
                    .block_on(async move {
                        let probe = Database::connect(url).await.unwrap();
                        rust_llm_usages::Entity::find()
                            .filter(rust_llm_usages::Column::ChatId.eq(chat_id as i64))
                            .count(&probe)
                            .await
                            .unwrap()
                    })
            })
            .join()
            .unwrap();
            sink.lock().unwrap().push(count);
        }
    }));
    let mut chat = record.to_llm_with(&db, Arc::new(config)).await.unwrap();

    record.ask(&db, &mut chat, "Hello").await.unwrap();

    assert_eq!(*observed.lock().unwrap(), vec![1, 2]);
    let _ = std::fs::remove_file(&path);
}
