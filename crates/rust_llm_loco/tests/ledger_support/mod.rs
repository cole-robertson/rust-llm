//! Helpers shared by `spec_usage_ledger.rs` and `spec_usage_ledger_warnings.rs`.
#![allow(dead_code, unused_imports)]

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

pub const ANTHROPIC: &str = "claude-haiku-4-5";
pub const EMBEDDING: &str = "text-embedding-3-small";
pub const IMAGE: &str = "gpt-image-1";
pub const TRANSCRIPTION: &str = "gemini-2.5-flash";
pub const JUDGMENT: &str = "jev-latest";

pub async fn db() -> DatabaseConnection {
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
pub fn config(server: &MockServer, db: &DatabaseConnection) -> Arc<Config> {
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

pub async fn json_response(server: &MockServer, route: &str, status: u16, body: Value) {
    Mock::given(method("POST"))
        .and(path(route))
        .respond_with(ResponseTemplate::new(status).set_body_json(body))
        .mount(server)
        .await;
}

pub async fn stub_embedding(server: &MockServer) {
    json_response(
        server,
        "/v1/embeddings",
        200,
        json!({ "model": EMBEDDING, "data": [{ "embedding": [0.1, 0.2] }], "usage": { "prompt_tokens": 3, "total_tokens": 3 } }),
    )
    .await;
}

pub async fn stub_chat_reply(server: &MockServer) {
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

pub async fn embed(
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

pub fn audio() -> rust_llm::Attachment {
    rust_llm::Attachment::new(format!(
        "{}/../rust_llm/tests/fixtures/ruby.wav",
        env!("CARGO_MANIFEST_DIR")
    ))
}

pub async fn transcribe(
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

pub fn owner_of(record: &ChatRecord) -> UsageOwner {
    UsageOwner::record("Chat", i64::from(record.id()))
}

/// `described_class.where(owner: record).order(:id)`.
pub async fn rows_for(db: &DatabaseConnection, owner: &ChatRecord) -> Vec<rust_llm_usages::Model> {
    rust_llm_usages::Entity::find()
        .filter(rust_llm_usages::Column::OwnerType.eq("Chat"))
        .filter(rust_llm_usages::Column::OwnerId.eq(i64::from(owner.id())))
        .order_by_asc(rust_llm_usages::Column::Id)
        .all(db)
        .await
        .unwrap()
}

pub async fn all_rows(db: &DatabaseConnection) -> Vec<rust_llm_usages::Model> {
    rust_llm_usages::Entity::find()
        .order_by_asc(rust_llm_usages::Column::Id)
        .all(db)
        .await
        .unwrap()
}

pub fn owner_pair(row: &rust_llm_usages::Model) -> (Option<String>, Option<i64>) {
    (row.owner_type.clone(), row.owner_id)
}

/// Collects the warnings RubyLLM's logger would receive.
#[derive(Clone, Default)]
pub struct Warnings(pub Arc<Mutex<Vec<String>>>);

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
    pub fn lines(&self) -> Vec<String> {
        self.0.lock().unwrap().clone()
    }
}

/// `allow(described_class).to receive(:create!).and_raise(ActiveRecord::StatementInvalid, ...)`:
/// every insert into `rust_llm_usages` fails, through a trigger.
pub async fn break_usage_writes(db: &DatabaseConnection, message: &str) {
    db.execute_unprepared(&format!(
        "CREATE TRIGGER fail_usage BEFORE INSERT ON rust_llm_usages BEGIN SELECT RAISE(ABORT, '{message}'); END"
    ))
    .await
    .unwrap();
}

/// The context's `before`: a trigger rejects every link of a usage row to a message.
pub async fn reject_usage_links(db: &DatabaseConnection) {
    db.execute_unprepared(
        "CREATE TRIGGER reject_usage_link BEFORE UPDATE OF message_id ON rust_llm_usages \
         BEGIN SELECT RAISE(ABORT, 'usage link rejected'); END",
    )
    .await
    .unwrap();
}

/// A chat record on `db` whose replies come from `server`, with usage links rejected.
pub async fn unlinkable_chat(
    db: &DatabaseConnection,
    server: &MockServer,
) -> (ChatRecord, rust_llm::Chat) {
    let config = config(server, db);
    let record = ChatRecord::create(db, ANTHROPIC, None).await.unwrap();
    let chat = record.to_llm_with(db, config).await.unwrap();
    reject_usage_links(db).await;
    (record, chat)
}

pub async fn roles_and_contents(
    db: &DatabaseConnection,
    record: &ChatRecord,
) -> Vec<(String, String)> {
    record
        .messages(db)
        .await
        .unwrap()
        .into_iter()
        .map(|m| (m.role, m.content.unwrap_or_default()))
        .collect()
}
