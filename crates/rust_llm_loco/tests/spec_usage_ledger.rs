//! Port of `spec/ruby_llm/active_record/usage_ledger_spec.rb`: `rust_llm_usages` as the usage
//! ledger for attempts outside a chat record ([`rust_llm_loco::UsageLedger`]), and the owner of
//! each persisted chat attempt.
//!
//! Ruby's `owner` is an Active Record (`Chat.create!`); here it is
//! [`UsageOwner::record`]`("Chat", id)`, the `owner_type`/`owner_id` pair Rails would store.

use std::sync::{Arc, Mutex};

use rust_llm::accounting::{UsageOwner, with_usage_owner};
use rust_llm::message::Operation;
use rust_llm::{
    Chat, Config, EmbedOptions, JudgeOptions, PaintOptions, TranscribeOptions, UsageEntry,
    UsageStatus,
};
use rust_llm_loco::entities::rust_llm_usages;
use rust_llm_loco::{ChatRecord, UsageLedger, migrations};
use sea_orm::{
    ColumnTrait, ConnectionTrait, Database, DatabaseConnection, EntityTrait, PaginatorTrait,
    QueryFilter, QueryOrder, TransactionTrait,
};
use sea_orm_migration::SchemaManager;
use serde_json::{Value, json};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const ANTHROPIC: &str = "claude-haiku-4-5";
const EMBEDDING: &str = "text-embedding-3-small";
const IMAGE: &str = "gpt-image-1";
const TRANSCRIPTION: &str = "gemini-2.5-flash";
const JUDGMENT: &str = "jev-latest";

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

/// The configured RubyLLM, pointed at `server`, with the Rails ledger (`railtie.rb` sets
/// `Accounting::Usage.ledger ||= RubyLLM::ActiveRecord::Usage`).
fn config(server: &MockServer, db: &DatabaseConnection) -> Arc<Config> {
    let mut c = Config::default();
    for (provider, base) in [
        ("openai", "/v1"),
        ("anthropic", ""),
        ("gemini", "/v1beta"),
        ("typesafe", ""),
    ] {
        c.set(
            format!("{provider}_api_base"),
            format!("{}{base}", server.uri()),
        );
        c.set(format!("{provider}_api_key"), "test");
    }
    c.max_retries = 0;
    c.usage_ledger = Some(UsageLedger::shared(db.clone()));
    Arc::new(c)
}

async fn json_response(server: &MockServer, route: &str, status: u16, body: Value) {
    Mock::given(method("POST"))
        .and(path(route))
        .respond_with(ResponseTemplate::new(status).set_body_json(body))
        .mount(server)
        .await;
}

async fn stub_embedding(server: &MockServer) {
    json_response(
        server,
        "/v1/embeddings",
        200,
        json!({ "model": EMBEDDING, "data": [{ "embedding": [0.1, 0.2] }], "usage": { "prompt_tokens": 3, "total_tokens": 3 } }),
    )
    .await;
}

async fn stub_chat_reply(server: &MockServer) {
    json_response(
        server,
        "/v1/messages",
        200,
        json!({ "id": "msg_1", "type": "message", "role": "assistant", "model": ANTHROPIC,
                "content": [{ "type": "text", "text": "Hi" }], "stop_reason": "end_turn",
                "usage": { "input_tokens": 9, "output_tokens": 2 } }),
    )
    .await;
}

async fn embed(
    config: &Arc<Config>,
    owner: Option<UsageOwner>,
) -> rust_llm::Result<rust_llm::Embedding> {
    rust_llm::embed(
        "Ruby",
        EmbedOptions {
            model: Some(EMBEDDING),
            provider: Some("openai"),
            config: Some(config.clone()),
            owner,
            ..Default::default()
        },
    )
    .await
}

fn audio() -> rust_llm::Attachment {
    rust_llm::Attachment::new(format!(
        "{}/../rust_llm/tests/fixtures/ruby.wav",
        env!("CARGO_MANIFEST_DIR")
    ))
}

async fn transcribe(
    config: &Arc<Config>,
    owner: Option<UsageOwner>,
) -> rust_llm::Result<rust_llm::Transcription> {
    rust_llm::transcribe(
        audio(),
        TranscribeOptions {
            model: Some(TRANSCRIPTION),
            provider: Some("gemini"),
            config: Some(config.clone()),
            owner,
            ..Default::default()
        },
    )
    .await
}

fn owner_of(record: &ChatRecord) -> UsageOwner {
    UsageOwner::record("Chat", i64::from(record.id()))
}

/// `described_class.where(owner: record).order(:id)`.
async fn rows_for(db: &DatabaseConnection, owner: &ChatRecord) -> Vec<rust_llm_usages::Model> {
    rust_llm_usages::Entity::find()
        .filter(rust_llm_usages::Column::OwnerType.eq("Chat"))
        .filter(rust_llm_usages::Column::OwnerId.eq(i64::from(owner.id())))
        .order_by_asc(rust_llm_usages::Column::Id)
        .all(db)
        .await
        .unwrap()
}

async fn all_rows(db: &DatabaseConnection) -> Vec<rust_llm_usages::Model> {
    rust_llm_usages::Entity::find()
        .order_by_asc(rust_llm_usages::Column::Id)
        .all(db)
        .await
        .unwrap()
}

fn owner_pair(row: &rust_llm_usages::Model) -> (Option<String>, Option<i64>) {
    (row.owner_type.clone(), row.owner_id)
}

/// Collects the warnings RubyLLM's logger would receive.
#[derive(Clone, Default)]
struct Warnings(Arc<Mutex<Vec<String>>>);

impl tracing::Subscriber for Warnings {
    fn enabled(&self, metadata: &tracing::Metadata<'_>) -> bool {
        *metadata.level() <= tracing::Level::WARN
    }
    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        struct Message<'a>(&'a mut String);
        impl tracing::field::Visit for Message<'_> {
            fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                if field.name() == "message" {
                    self.0.push_str(&format!("{value:?}"));
                }
            }
        }
        let mut text = String::new();
        event.record(&mut Message(&mut text));
        self.0.lock().unwrap().push(text);
    }
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}

impl Warnings {
    fn lines(&self) -> Vec<String> {
        self.0.lock().unwrap().clone()
    }
}

/// `allow(described_class).to receive(:create!).and_raise(ActiveRecord::StatementInvalid, ...)`:
/// every insert into `rust_llm_usages` fails, through a trigger.
async fn break_usage_writes(db: &DatabaseConnection, message: &str) {
    db.execute_unprepared(&format!(
        "CREATE TRIGGER fail_usage BEFORE INSERT ON rust_llm_usages BEGIN SELECT RAISE(ABORT, '{message}'); END"
    ))
    .await
    .unwrap();
}

// spec: active_record/usage_ledger_spec.rb:50 writes a row for each one-shot operation, attributed to its owner and to no chat
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

// spec: active_record/usage_ledger_spec.rb:67 attributes rows to the ambient owner, lets the keyword win, and writes unattributed rows
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

// spec: active_record/usage_ledger_spec.rb:84 records an attempt that failed after the provider may have billed it
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

// spec: active_record/usage_ledger_spec.rb:93 records a blocked transcription with the tokens the provider billed
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

// spec: active_record/usage_ledger_spec.rb:119 writes a row when a video or research job finishes, attributed to the owner at submission
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

// spec: active_record/usage_ledger_spec.rb:140 writes a persisted chat attempt once, linked to both its chat and owner
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

// spec: active_record/usage_ledger_spec.rb:150 keeps the owner of each turn when different people use the same chat
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

// spec: active_record/usage_ledger_spec.rb:162 keeps the owner of a failed chat attempt that produced no message
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

// spec: active_record/usage_ledger_spec.rb:172 keeps the owner of a cancelled chat attempt with partial usage
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

// spec: active_record/usage_ledger_spec.rb:191 attributes the rows of a chat without a record to the owner
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

// spec: active_record/usage_ledger_spec.rb:199 writes unattributed rows for an owner that is not a record
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

// spec: active_record/usage_ledger_spec.rb:213 logs a failed chat row write and returns the reply
#[tokio::test]
async fn logs_a_failed_chat_row_write_and_returns_the_reply() {
    let db = db().await;
    let server = MockServer::start().await;
    let config = config(&server, &db);
    stub_chat_reply(&server).await;
    let record = ChatRecord::create(&db, ANTHROPIC, None).await.unwrap();
    let mut chat = record.to_llm_with(&db, config).await.unwrap();
    break_usage_writes(&db, "disk full").await;
    let warnings = Warnings::default();
    let _guard = tracing::subscriber::set_default(warnings.clone());

    let reply = record.ask(&db, &mut chat, "Hello").await.unwrap();

    assert_eq!(reply.content(), "Hi");
    assert_eq!(
        record
            .messages(&db)
            .await
            .unwrap()
            .iter()
            .map(|m| m.role.as_str())
            .collect::<Vec<_>>(),
        ["user", "assistant"]
    );
    assert!(
        warnings
            .lines()
            .iter()
            .any(|l| l.contains("could not record chat usage") && l.contains("disk full")),
        "{:?}",
        warnings.lines()
    );
}

// spec: active_record/usage_ledger_spec.rb:224 logs a failed write and returns the result of the operation
#[tokio::test]
async fn logs_a_failed_write_and_returns_the_result_of_the_operation() {
    let db = db().await;
    let server = MockServer::start().await;
    let config = config(&server, &db);
    stub_embedding(&server).await;
    let owner = ChatRecord::create(&db, ANTHROPIC, None).await.unwrap();
    break_usage_writes(&db, "disk full").await;
    let warnings = Warnings::default();
    let _guard = tracing::subscriber::set_default(warnings.clone());

    let embedding = embed(&config, Some(owner_of(&owner))).await.unwrap();

    assert_eq!(embedding.vectors, rust_llm::Vectors::Single(vec![0.1, 0.2]));
    assert!(
        warnings
            .lines()
            .iter()
            .any(|l| l.contains("could not record embedding usage") && l.contains("disk full")),
        "{:?}",
        warnings.lines()
    );
}

// spec: active_record/usage_ledger_spec.rb:233 keeps the caller transaction usable when a write fails
#[tokio::test]
async fn keeps_the_caller_transaction_usable_when_a_write_fails() {
    // A file database, so the ledger's own connection and the caller's transaction are separate
    // connections, as with Rails' pool.
    let dir = std::env::temp_dir().join(format!("ledger-{}.sqlite", std::process::id()));
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
    rust_llm::configure(|c| {
        c.set("anthropic_api_key", "test");
    });
    let server = MockServer::start().await;
    let config = config(&server, &db);
    stub_embedding(&server).await;
    let owner = ChatRecord::create(&db, ANTHROPIC, None).await.unwrap();
    break_usage_writes(&db, "constraint failed").await;

    let warnings = Warnings::default();
    let _guard = tracing::subscriber::set_default(warnings.clone());
    let txn = db.begin().await.unwrap();
    embed(&config, Some(owner_of(&owner))).await.unwrap();
    // The write reached the database and failed there, not on a pool timeout.
    assert!(
        warnings
            .lines()
            .iter()
            .any(|l| l.contains("constraint failed")),
        "{:?}",
        warnings.lines()
    );
    rust_llm_loco::entities::chats::Entity::find()
        .count(&txn)
        .await
        .unwrap();
    txn.commit().await.unwrap();

    assert!(rows_for(&db, &owner).await.is_empty());
    let _ = std::fs::remove_file(&dir);
}

// spec: active_record/usage_ledger_spec.rb:246 returns the connection a thread borrowed to write its row
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

// spec: active_record/usage_ledger_spec.rb:256 skips attempts without a model, which the ledger cannot store
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
