//! Port of `spec/ruby_llm/active_record/usage_ledger_spec.rb`: `rust_llm_usages` as the usage
//! ledger for attempts outside a chat record ([`rust_llm_loco::UsageLedger`]), and the owner of
//! each persisted chat attempt.
//!
//! Ruby's `owner` is an Active Record (`Chat.create!`); here it is
//! [`UsageOwner::record`]`("Chat", id)`, the `owner_type`/`owner_id` pair Rails would store.

mod ledger_support;

use std::sync::{Arc, Mutex};

use rust_llm::accounting::{UsageOwner, with_usage_owner};
use rust_llm::message::Operation;
use rust_llm::{Chat, JudgeOptions, PaintOptions, UsageEntry, UsageStatus};
use rust_llm_loco::entities::rust_llm_usages;
use rust_llm_loco::{ChatRecord, UsageLedger, migrations};
use sea_orm::{Database, DatabaseConnection, EntityTrait, PaginatorTrait, TransactionTrait};
use sea_orm_migration::SchemaManager;
use serde_json::{Value, json};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use ledger_support::*;

// spec: active_record/usage_ledger_spec.rb:51 writes a row for each one-shot operation, attributed to its owner and to no chat
#[tokio::test]
async fn writes_a_row_for_each_one_shot_operation_attributed_to_its_owner() {
    let db = db().await;
    let server = MockServer::start().await;
    let config = config(&server, &db);
    let owner = ChatRecord::create(&db, ANTHROPIC, None).await.unwrap();
    stub_embedding(&server).await;
    json_response(
        &server,
        "/v1/images/generations",
        200,
        json!({ "data": [{ "b64_json": "aGk=" }], "usage": { "input_tokens": 5, "output_tokens": 7 } }),
    )
    .await;
    json_response(
        &server,
        &format!("/v1beta/models/{TRANSCRIPTION}:generateContent"),
        200,
        json!({ "candidates": [{ "content": { "parts": [{ "text": "Ruby" }] }, "finishReason": "STOP" }],
                "usageMetadata": { "promptTokenCount": 40, "candidatesTokenCount": 2 } }),
    )
    .await;
    json_response(
        &server,
        "/v1/systemone",
        200,
        json!({ "model": JUDGMENT, "answers": { "urgent": { "type": "noul", "noul": 0.9 } },
                "usage": { "input_tokens": 11, "output_tokens": 1 } }),
    )
    .await;

    embed(&config, Some(owner_of(&owner))).await.unwrap();
    rust_llm::paint(
        "A paper boat",
        PaintOptions {
            model: Some(IMAGE),
            provider: Some("openai"),
            config: Some(config.clone()),
            owner: Some(owner_of(&owner)),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    transcribe(&config, Some(owner_of(&owner))).await.unwrap();
    rust_llm::judge(
        "Help",
        json!({ "urgent": { "type": "probability", "instructions": "Is this urgent?" } }),
        JudgeOptions {
            model: Some(Some(JUDGMENT.into())),
            provider: Some("typesafe".into()),
            assume_model_exists: Some(true),
            config: Some(config.clone()),
            owner: Some(owner_of(&owner)),
            ..Default::default()
        },
    )
    .await
    .unwrap();

    let rows = rows_for(&db, &owner).await;
    assert_eq!(
        rows.iter()
            .map(|r| r.operation.as_str())
            .collect::<Vec<_>>(),
        ["embedding", "image", "transcription", "judgment"]
    );
    assert!(rows.iter().all(|r| r.status == "succeeded"));
    assert!(rows.iter().all(|r| r.chat_id.is_none()));
    assert_eq!(
        rows.iter()
            .map(|r| (r.input_tokens, r.output_tokens))
            .collect::<Vec<_>>(),
        [
            (Some(3), None),
            (Some(5), Some(7)),
            (Some(40), Some(2)),
            (Some(11), Some(1))
        ]
    );
    assert_eq!(
        (rows[0].provider.as_str(), rows[0].model.as_str()),
        ("openai", EMBEDDING)
    );
}

// spec: active_record/usage_ledger_spec.rb:68 attributes rows to the ambient owner, lets the keyword win, and writes unattributed rows
#[tokio::test]
async fn attributes_rows_to_the_ambient_owner_and_writes_unattributed_rows() {
    let db = db().await;
    let server = MockServer::start().await;
    let config = config(&server, &db);
    stub_embedding(&server).await;
    let owner = ChatRecord::create(&db, ANTHROPIC, None).await.unwrap();
    let other = ChatRecord::create(&db, ANTHROPIC, None).await.unwrap();

    with_usage_owner(owner_of(&owner), async {
        embed(&config, None).await.unwrap();
        embed(&config, Some(owner_of(&other))).await.unwrap();
    })
    .await;
    embed(&config, None).await.unwrap();

    assert_eq!(all_rows(&db).await.len(), 3);
    assert_eq!(rows_for(&db, &owner).await.len(), 1);
    assert_eq!(rows_for(&db, &other).await.len(), 1);
    let last = all_rows(&db).await.pop().unwrap();
    assert_eq!((owner_pair(&last), last.chat_id), ((None, None), None));
}

// spec: active_record/usage_ledger_spec.rb:85 records an attempt that failed after the provider may have billed it
#[tokio::test]
async fn records_an_attempt_that_failed_after_the_provider_may_have_billed_it() {
    let db = db().await;
    let server = MockServer::start().await;
    let config = config(&server, &db);
    let owner = ChatRecord::create(&db, ANTHROPIC, None).await.unwrap();
    json_response(
        &server,
        "/v1/embeddings",
        500,
        json!({ "error": { "message": "Boom" } }),
    )
    .await;

    let error = embed(&config, Some(owner_of(&owner))).await.unwrap_err();

    assert!(matches!(error, rust_llm::Error::Server(..)), "{error:?}");
    assert_eq!(
        rows_for(&db, &owner)
            .await
            .iter()
            .map(|r| r.status.as_str())
            .collect::<Vec<_>>(),
        ["failed"]
    );
}

// spec: active_record/usage_ledger_spec.rb:94 records a blocked transcription with the tokens the provider billed
#[tokio::test]
async fn records_a_blocked_transcription_with_the_tokens_the_provider_billed() {
    let db = db().await;
    let server = MockServer::start().await;
    let config = config(&server, &db);
    let owner = ChatRecord::create(&db, ANTHROPIC, None).await.unwrap();
    json_response(
        &server,
        &format!("/v1beta/models/{TRANSCRIPTION}:generateContent"),
        200,
        json!({ "promptFeedback": { "blockReason": "SAFETY" },
                "usageMetadata": { "promptTokenCount": 133, "totalTokenCount": 133 } }),
    )
    .await;

    let error = transcribe(&config, Some(owner_of(&owner)))
        .await
        .unwrap_err();

    assert!(
        matches!(error, rust_llm::Error::ContentFilter(..)),
        "{error:?}"
    );
    let rows = rows_for(&db, &owner).await;
    assert_eq!(rows.len(), 1);
    let row = &rows[0];
    assert_eq!(
        (
            row.operation.as_str(),
            row.status.as_str(),
            row.chat_id,
            row.input_tokens
        ),
        ("transcription", "failed", None, Some(133))
    );
}

// spec: active_record/usage_ledger_spec.rb:120 writes a row when a video or research job finishes, attributed to the owner at submission
#[tokio::test]
async fn writes_a_row_when_a_video_job_finishes_attributed_to_the_owner_at_submission() {
    let db = db().await;
    let server = MockServer::start().await;
    let config = config(&server, &db);
    let owner = ChatRecord::create(&db, ANTHROPIC, None).await.unwrap();
    let model = "veo-3.1-fast-generate-preview";
    json_response(
        &server,
        &format!("/v1beta/models/{model}:predictLongRunning"),
        200,
        json!({ "name": "operations/op-1" }),
    )
    .await;
    Mock::given(method("GET"))
        .and(path("/v1beta/operations/op-1"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "name": "operations/op-1", "done": true,
            "response": { "generateVideoResponse": { "generatedSamples": [{ "video": { "uri": "https://example.com/v.mp4" } }] } }
        })))
        .mount(&server)
        .await;

    // Submitted inside the owner's block, refreshed outside it.
    let mut video = with_usage_owner(owner_of(&owner), async {
        rust_llm::animate_later(
            Some("a wave"),
            rust_llm::AnimateOptions {
                model: Some(model),
                provider: Some("gemini"),
                config: Some(config.clone()),
                ..Default::default()
            },
        )
        .await
        .unwrap()
    })
    .await;
    assert!(
        rows_for(&db, &owner).await.is_empty(),
        "pending jobs record nothing"
    );
    video.refresh().await.unwrap();

    // Ruby's research job half needs `RubyLLM::ResearchJob`, which this port does not have.
    assert_eq!(
        rows_for(&db, &owner)
            .await
            .iter()
            .map(|r| (
                r.operation.clone(),
                r.model.clone(),
                r.status.clone(),
                r.total_cost
            ))
            .collect::<Vec<_>>(),
        [(
            "video".to_string(),
            model.to_string(),
            "succeeded".to_string(),
            None
        )]
    );
}

// spec: active_record/usage_ledger_spec.rb:141 writes a persisted chat attempt once, linked to both its chat and owner
#[tokio::test]
async fn writes_a_persisted_chat_attempt_once_linked_to_its_chat_and_owner() {
    let db = db().await;
    let server = MockServer::start().await;
    let config = config(&server, &db);
    stub_chat_reply(&server).await;
    let owner = ChatRecord::create(&db, ANTHROPIC, None).await.unwrap();
    let record = ChatRecord::create(&db, ANTHROPIC, None).await.unwrap();
    let mut chat = record.to_llm_with(&db, config).await.unwrap();

    with_usage_owner(owner_of(&owner), record.ask(&db, &mut chat, "Hello"))
        .await
        .unwrap();

    assert_eq!(all_rows(&db).await.len(), 1);
    let usages = record.usages(&db).await.unwrap();
    assert_eq!(usages.len(), 1);
    let last_message = record.messages(&db).await.unwrap().pop().unwrap();
    let row = &usages[0];
    assert_eq!(row.operation, "chat");
    assert_eq!(row.chat_id, Some(i64::from(record.id())));
    assert_eq!(
        owner_pair(row),
        (Some("Chat".into()), Some(i64::from(owner.id())))
    );
    assert_eq!(row.message_id, Some(i64::from(last_message.id)));
    assert_eq!(row.input_tokens, Some(9));
}

// spec: active_record/usage_ledger_spec.rb:151 keeps the owner of each turn when different people use the same chat
#[tokio::test]
async fn keeps_the_owner_of_each_turn() {
    let db = db().await;
    let server = MockServer::start().await;
    let config = config(&server, &db);
    stub_chat_reply(&server).await;
    let record = ChatRecord::create(&db, ANTHROPIC, None).await.unwrap();
    let owner = ChatRecord::create(&db, ANTHROPIC, None).await.unwrap();
    let other = ChatRecord::create(&db, ANTHROPIC, None).await.unwrap();

    let mut chat = record.to_llm_with(&db, config.clone()).await.unwrap();
    with_usage_owner(owner_of(&owner), record.ask(&db, &mut chat, "Hello"))
        .await
        .unwrap();
    let mut chat = record.to_llm_with(&db, config.clone()).await.unwrap();
    with_usage_owner(owner_of(&other), record.ask(&db, &mut chat, "Hello again"))
        .await
        .unwrap();
    let mut chat = record.to_llm_with(&db, config).await.unwrap();
    record.ask(&db, &mut chat, "Goodbye").await.unwrap();

    assert_eq!(
        record
            .usages(&db)
            .await
            .unwrap()
            .iter()
            .map(|u| u.owner_id)
            .collect::<Vec<_>>(),
        [
            Some(i64::from(owner.id())),
            Some(i64::from(other.id())),
            None
        ]
    );
}

// spec: active_record/usage_ledger_spec.rb:163 keeps the owner of a failed chat attempt that produced no message
#[tokio::test]
async fn keeps_the_owner_of_a_failed_chat_attempt() {
    let db = db().await;
    let server = MockServer::start().await;
    let config = config(&server, &db);
    json_response(
        &server,
        "/v1/messages",
        500,
        json!({ "error": { "type": "api_error", "message": "Boom" } }),
    )
    .await;
    let owner = ChatRecord::create(&db, ANTHROPIC, None).await.unwrap();
    let record = ChatRecord::create(&db, ANTHROPIC, None).await.unwrap();
    let mut chat = record.to_llm_with(&db, config).await.unwrap();

    let error = with_usage_owner(owner_of(&owner), record.ask(&db, &mut chat, "Hello"))
        .await
        .unwrap_err();

    assert!(
        matches!(
            error,
            rust_llm_loco::Error::Llm(rust_llm::Error::Server(..))
        ),
        "{error:?}"
    );
    let usages = record.usages(&db).await.unwrap();
    assert_eq!(usages.len(), 1);
    assert_eq!(
        (
            owner_pair(&usages[0]),
            usages[0].status.as_str(),
            usages[0].message_id
        ),
        (
            (Some("Chat".into()), Some(i64::from(owner.id()))),
            "failed",
            None
        )
    );
}

// spec: active_record/usage_ledger_spec.rb:173 keeps the owner of a cancelled chat attempt with partial usage
#[tokio::test]
async fn keeps_the_owner_of_a_cancelled_chat_attempt_with_partial_usage() {
    let db = db().await;
    let owner = ChatRecord::create(&db, ANTHROPIC, None).await.unwrap();
    let record = ChatRecord::create(&db, ANTHROPIC, None).await.unwrap();
    let mut chat = record.to_llm(&db).await.unwrap();
    record.ask_later(&db, &mut chat, "Hello").await.unwrap();

    // Ruby stubs `provider.complete` with a tracker that observes 9/1 tokens and then cancels.
    // Here the record's usage recorder sees the same entry: started inside the owner's block, so
    // it carries the owner, with the observed tokens and a cancelled status.
    let model = rust_llm::models().find(ANTHROPIC, None).unwrap();
    with_usage_owner(owner_of(&owner), async {
        let recorded: Arc<Mutex<Vec<UsageEntry>>> = Arc::default();
        let sink = recorded.clone();
        let mut tracker = rust_llm::accounting::Tracker::new(
            Operation::Chat,
            "anthropic",
            Some(model),
            rust_llm::config(),
            Some(Box::new(move |e: &UsageEntry| {
                sink.lock().unwrap().push(e.clone())
            })),
        );
        let entry = tracker.start();
        tracker.observe_tokens(&rust_llm::Tokens {
            input: Some(9),
            output: Some(1),
            ..Default::default()
        });
        tracker.fail_attempt(Some(entry), &rust_llm::Error::Cancelled);
        let entries = recorded.lock().unwrap().clone();
        for entry in entries {
            record.persist_usage_entry(&db, &entry).await;
        }
    })
    .await;

    let usages = record.usages(&db).await.unwrap();
    assert_eq!(usages.len(), 1);
    let row = &usages[0];
    assert_eq!(
        (
            owner_pair(row),
            row.status.as_str(),
            row.message_id,
            row.input_tokens,
            row.output_tokens
        ),
        (
            (Some("Chat".into()), Some(i64::from(owner.id()))),
            "cancelled",
            None,
            Some(9),
            Some(1)
        )
    );
}

// spec: active_record/usage_ledger_spec.rb:192 attributes the rows of a chat without a record to the owner
#[tokio::test]
async fn attributes_the_rows_of_a_chat_without_a_record_to_the_owner() {
    let db = db().await;
    let server = MockServer::start().await;
    let config = config(&server, &db);
    stub_chat_reply(&server).await;
    let owner = ChatRecord::create(&db, ANTHROPIC, None).await.unwrap();

    with_usage_owner(owner_of(&owner), async {
        Chat::with_config(config, Some(ANTHROPIC), None, false)
            .unwrap()
            .ask("Hello")
            .await
            .unwrap()
    })
    .await;

    let rows = rows_for(&db, &owner).await;
    assert_eq!(rows.len(), 1);
    assert_eq!(
        (
            rows[0].operation.as_str(),
            rows[0].chat_id,
            rows[0].output_tokens
        ),
        ("chat", None, Some(2))
    );
}

// spec: active_record/usage_ledger_spec.rb:200 writes unattributed rows for an owner that is not a record
#[tokio::test]
async fn writes_unattributed_rows_for_an_owner_that_is_not_a_record() {
    let db = db().await;
    let server = MockServer::start().await;
    let config = config(&server, &db);
    stub_chat_reply(&server).await;
    stub_embedding(&server).await;
    let record = ChatRecord::create(&db, ANTHROPIC, None).await.unwrap();
    let mut chat = record.to_llm_with(&db, config.clone()).await.unwrap();

    with_usage_owner(UsageOwner::from("user-42"), async {
        assert_eq!(
            record.ask(&db, &mut chat, "Hello").await.unwrap().content(),
            "Hi"
        );
        embed(&config, None).await.unwrap();
    })
    .await;

    let rows = all_rows(&db).await;
    assert_eq!(
        rows[rows.len() - 2..]
            .iter()
            .map(|r| (r.operation.as_str(), r.owner_id))
            .collect::<Vec<_>>(),
        [("chat", None), ("embedding", None)]
    );
}

// ---- when a chat row cannot be linked to its message ---------------------------------------------

// spec: active_record/usage_ledger_spec.rb:268 when a chat row cannot be linked to its message > keeps the row with its chat and counts it in the chat totals
#[tokio::test]
async fn keeps_the_unlinked_row_with_its_chat_and_counts_it() {
    let db = db().await;
    let server = MockServer::start().await;
    stub_chat_reply(&server).await;
    let (record, mut chat) = unlinkable_chat(&db, &server).await;

    record.ask(&db, &mut chat, "Hello").await.unwrap();

    let usages = record.usages(&db).await.unwrap();
    assert_eq!(usages.len(), 1);
    let row = &usages[0];
    assert_eq!(row.status, "succeeded");
    assert_eq!(row.message_id, None);
    assert_eq!((row.input_tokens, row.output_tokens), (Some(9), Some(2)));
    let tokens = ChatRecord::find(&db, record.id())
        .await
        .unwrap()
        .tokens(&db)
        .await
        .unwrap();
    assert_eq!((tokens.input, tokens.output), (Some(9), Some(2)));
}

/// Statements SQLite ran on a traced connection (`QueryHelpers.matching`).
static STATEMENTS: Mutex<Vec<String>> = Mutex::new(Vec::new());

unsafe extern "C" {
    fn sqlite3_trace_v2(
        db: *mut std::ffi::c_void,
        mask: std::ffi::c_uint,
        callback: Option<
            unsafe extern "C" fn(
                std::ffi::c_uint,
                *mut std::ffi::c_void,
                *mut std::ffi::c_void,
                *mut std::ffi::c_void,
            ) -> std::ffi::c_int,
        >,
        context: *mut std::ffi::c_void,
    ) -> std::ffi::c_int;
}

/// `SQLITE_TRACE_STMT`: `sql` is the statement's text as written.
unsafe extern "C" fn record_statement(
    _: std::ffi::c_uint,
    _: *mut std::ffi::c_void,
    _: *mut std::ffi::c_void,
    sql: *mut std::ffi::c_void,
) -> std::ffi::c_int {
    // SAFETY: SQLite passes the statement's NUL-terminated SQL text for SQLITE_TRACE_STMT.
    let sql = unsafe { std::ffi::CStr::from_ptr(sql.cast()) };
    STATEMENTS
        .lock()
        .unwrap()
        .push(sql.to_string_lossy().into_owned());
    0
}

/// Traces every statement on the in-memory database's one pooled connection.
async fn trace_statements(db: &DatabaseConnection) {
    let mut conn = db.get_sqlite_connection_pool().acquire().await.unwrap();
    let mut handle = conn.lock_handle().await.unwrap();
    // SAFETY: the handle is locked out of the worker, and the callback only reads its arguments.
    let status = unsafe {
        sqlite3_trace_v2(
            handle.as_raw_handle().as_ptr().cast(),
            1,
            Some(record_statement),
            std::ptr::null_mut(),
        )
    };
    assert_eq!(status, 0);
}

// spec: active_record/usage_ledger_spec.rb:279 when a chat row cannot be linked to its message > rolls the failed link back to a savepoint
#[tokio::test]
async fn rolls_the_failed_link_back_to_a_savepoint() {
    let db = db().await;
    let server = MockServer::start().await;
    stub_chat_reply(&server).await;
    let (record, mut chat) = unlinkable_chat(&db, &server).await;
    trace_statements(&db).await;

    record.ask(&db, &mut chat, "Hello").await.unwrap();

    let statements: Vec<String> = STATEMENTS
        .lock()
        .unwrap()
        .iter()
        .filter(|s| s.contains("SAVEPOINT"))
        .cloned()
        .collect();
    assert!(
        statements
            .iter()
            .any(|s| s.starts_with("ROLLBACK TO SAVEPOINT")),
        "{statements:?}"
    );
}

/// `ClockTool`: `def execute = 'noon'`.
struct Clock;

#[async_trait::async_trait]
impl rust_llm::Tool for Clock {
    fn name(&self) -> String {
        "clock".into()
    }
    fn description(&self) -> String {
        "Tells the time".into()
    }
    async fn execute(
        &self,
        _: serde_json::Map<String, Value>,
        _: &rust_llm::ToolCall,
    ) -> Result<rust_llm::ToolResult, rust_llm::ToolError> {
        Ok("noon".into())
    }
}

// spec: active_record/usage_ledger_spec.rb:288 when a chat row cannot be linked to its message > keeps the tool calls of the reply and finishes the tool loop
#[tokio::test]
async fn keeps_the_tool_calls_of_the_reply_and_finishes_the_tool_loop() {
    let db = db().await;
    let server = MockServer::start().await;
    let tool_use = json!({ "id": "msg_0", "type": "message", "role": "assistant", "model": ANTHROPIC,
        "content": [{ "type": "tool_use", "id": "toolu_1", "name": "clock", "input": {} }],
        "stop_reason": "tool_use", "usage": { "input_tokens": 7, "output_tokens": 3 } });
    let mut answer = tool_use.clone();
    answer["content"] = json!([{ "type": "text", "text": "Hi" }]);
    answer["stop_reason"] = json!("end_turn");
    let replies = Mutex::new(vec![answer, tool_use]);
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(move |_: &wiremock::Request| {
            let mut replies = replies.lock().unwrap();
            let body = if replies.len() > 1 {
                replies.pop().unwrap()
            } else {
                replies[0].clone()
            };
            ResponseTemplate::new(200).set_body_json(body)
        })
        .mount(&server)
        .await;
    let (record, chat) = unlinkable_chat(&db, &server).await;
    let mut chat = chat.with_tool(Clock);

    let reply = record
        .ask(&db, &mut chat, "What time is it?")
        .await
        .unwrap();

    assert_eq!(reply.content(), "Hi");
    let rows = record.messages(&db).await.unwrap();
    assert_eq!(
        rows.iter().map(|m| m.role.as_str()).collect::<Vec<_>>(),
        ["user", "assistant", "tool", "assistant"]
    );
    let second = rust_llm_loco::message_to_llm(&db, &rows[1]).await.unwrap();
    let ids: Vec<String> = second
        .tool_calls
        .unwrap()
        .values()
        .map(|c| c.id.clone())
        .collect();
    assert_eq!(ids, ["toolu_1"]);
    let message_ids: Vec<Option<i64>> = record
        .usages(&db)
        .await
        .unwrap()
        .iter()
        .map(|u| u.message_id)
        .collect();
    assert_eq!(message_ids, [None, None]);
}

// spec: active_record/usage_ledger_spec.rb:304 when a chat row cannot be linked to its message > keeps the caller transaction usable
#[tokio::test]
async fn keeps_the_caller_transaction_usable_when_a_link_fails() {
    // A file database, so the caller's transaction and the chat's writes use separate pooled
    // connections, as with Rails' pool.
    let dir = std::env::temp_dir().join(format!("ledger-link-{}.sqlite", std::process::id()));
    let _ = std::fs::remove_file(&dir);
    let mut options = sea_orm::ConnectOptions::new(format!("sqlite://{}?mode=rwc", dir.display()));
    options
        .max_connections(2)
        .acquire_timeout(std::time::Duration::from_secs(5));
    let db = Database::connect(options).await.unwrap();
    let manager = SchemaManager::new(&db);
    for m in migrations() {
        m.up(&manager).await.unwrap();
    }
    let server = MockServer::start().await;
    stub_chat_reply(&server).await;
    let (record, mut chat) = unlinkable_chat(&db, &server).await;

    let txn = db.begin().await.unwrap();
    record.ask(&db, &mut chat, "Hello").await.unwrap();
    let later = rust_llm_loco::entities::chats::ActiveModel {
        rust_llm_model_id: sea_orm::ActiveValue::Set(
            rust_llm_loco::entities::chats::Entity::find_by_id(record.id())
                .one(&txn)
                .await
                .unwrap()
                .unwrap()
                .rust_llm_model_id,
        ),
        cancelled: sea_orm::ActiveValue::Set(false),
        created_at: sea_orm::ActiveValue::Set(chrono::Utc::now().into()),
        updated_at: sea_orm::ActiveValue::Set(chrono::Utc::now().into()),
        ..Default::default()
    };
    let later = sea_orm::ActiveModelTrait::insert(later, &txn)
        .await
        .unwrap();
    txn.commit().await.unwrap();

    assert!(
        rust_llm_loco::entities::chats::Entity::find_by_id(later.id)
            .one(&db)
            .await
            .unwrap()
            .is_some()
    );
    assert_eq!(
        roles_and_contents(&db, &record).await.last().unwrap().1,
        "Hi"
    );
    let _ = std::fs::remove_file(&dir);
}

// spec: active_record/usage_ledger_spec.rb:318 returns the connection a thread borrowed to write its row
#[tokio::test]
async fn returns_the_connection_a_task_borrowed_to_write_its_row() {
    let db = db().await;
    let owner = ChatRecord::create(&db, ANTHROPIC, None).await.unwrap();
    let mut entry = UsageEntry::new(Operation::Embedding, "openai", Some(EMBEDDING));
    entry.status = UsageStatus::Succeeded;
    entry.owner = Some(owner_of(&owner));
    let ledger = UsageLedger::new(db.clone());

    tokio::spawn(async move { ledger.record(&entry).await })
        .await
        .unwrap()
        .unwrap();

    assert_eq!(rows_for(&db, &owner).await.len(), 1);
    // The pool's connection came back: an in-memory SQLite pool holds one connection, so this
    // query would wait forever if the task had kept it.
    tokio::time::timeout(
        std::time::Duration::from_secs(5),
        rust_llm_usages::Entity::find().count(&db),
    )
    .await
    .expect("the connection was returned")
    .unwrap();
}

// spec: active_record/usage_ledger_spec.rb:328 skips attempts without a model, which the ledger cannot store
#[tokio::test]
async fn skips_attempts_without_a_model() {
    let db = db().await;
    let owner = ChatRecord::create(&db, ANTHROPIC, None).await.unwrap();
    let mut entry = UsageEntry::new(Operation::Moderation, "bedrock", None);
    entry.status = UsageStatus::Succeeded;
    entry.owner = Some(owner_of(&owner));

    UsageLedger::new(db.clone()).record(&entry).await.unwrap();

    assert_eq!(all_rows(&db).await.len(), 0);
}
