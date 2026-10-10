//! `spec/ruby_llm/active_record/acts_as_model_spec.rb`: the `rust_llm_models` table as the model
//! registry store ([`rust_llm_loco::model_store`]), plus the registry fill and insert race from
//! `chat_methods_spec.rb`. Each test gets its own in-memory SQLite database, which stands in for
//! Ruby's per-example rollback.

use std::sync::{Arc, Mutex};

use rust_llm::Config;
use rust_llm::models::Models;
use rust_llm_loco::entities::rust_llm_models;
use rust_llm_loco::{ChatRecord, ModelStore, migrations, model_store};
use sea_orm::{
    ColumnTrait, ConnectionTrait, Database, DatabaseConnection, EntityTrait, PaginatorTrait,
    QueryFilter,
};
use sea_orm_migration::SchemaManager;
use serde_json::json;

/// Serializes the tests that change or read the process-wide registry and configuration.
static GLOBAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// Puts back the global registry and configuration a test changed.
struct Restore(Arc<Config>);

impl Restore {
    fn new() -> Restore {
        Restore(rust_llm::config())
    }
}

impl Drop for Restore {
    fn drop(&mut self) {
        let saved = (*self.0).clone();
        rust_llm::configure(|c| *c = saved);
        rust_llm::models::refresh::reset();
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

fn model(data: serde_json::Value) -> rust_llm::Model {
    serde_json::from_value(data).unwrap()
}

/// `let(:model_info)`.
fn model_info() -> rust_llm::Model {
    model(json!({
        "id": "test-model",
        "name": "Test Model",
        "provider": "openai",
        "family": "test",
        "context_window": 128_000,
        "modalities": { "input": ["text", "image"], "output": ["text"] },
        "capabilities": ["function_calling", "vision"],
        "pricing": { "text_tokens": { "standard": { "input_per_million": 1.0, "output_per_million": 2.0 } } }
    }))
}

/// `let(:dropped)`.
fn dropped() -> rust_llm::Model {
    model(json!({ "id": "dropped-model", "name": "Dropped Model", "provider": "openai" }))
}

async fn find_row(db: &DatabaseConnection, id: &str) -> Option<rust_llm_models::Model> {
    rust_llm_models::Entity::find()
        .filter(rust_llm_models::Column::ModelId.eq(id))
        .filter(rust_llm_models::Column::Provider.eq("openai"))
        .one(db)
        .await
        .unwrap()
}

async fn dropped_record(db: &DatabaseConnection) -> rust_llm_models::Model {
    find_row(db, "dropped-model").await.unwrap()
}

async fn model_ids(
    query: sea_orm::Select<rust_llm_models::Entity>,
    db: &DatabaseConnection,
) -> Vec<String> {
    query
        .all(db)
        .await
        .unwrap()
        .into_iter()
        .map(|r| r.model_id)
        .collect()
}

fn ids(models: &[rust_llm::Model]) -> Vec<&str> {
    models.iter().map(|m| m.id.as_str()).collect()
}

/// `Chat.create!(model: record)`: a chat pointing at an existing row.
async fn chat_on(db: &DatabaseConnection, model_id: &str) -> ChatRecord {
    ChatRecord::create_with(db, model_id, Some("openai"), true)
        .await
        .unwrap()
}

// ---- RubyLLM.logger ----------------------------------------------------------------------------

/// Collects `tracing` events on this thread as `(level, message)`.
struct Logger(Arc<Mutex<Vec<(tracing::Level, String)>>>);

impl tracing::Subscriber for Logger {
    fn register_callsite(
        &self,
        _: &'static tracing::Metadata<'static>,
    ) -> tracing::subscriber::Interest {
        tracing::subscriber::Interest::sometimes()
    }
    fn max_level_hint(&self) -> Option<tracing::level_filters::LevelFilter> {
        Some(tracing::level_filters::LevelFilter::TRACE)
    }
    fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
        true
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
        self.0
            .lock()
            .unwrap()
            .push((*event.metadata().level(), text));
    }
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}

/// A logger for the rest of the current scope; `.lines(level)` reads what it received.
struct Logged {
    lines: Arc<Mutex<Vec<(tracing::Level, String)>>>,
    _guard: tracing::subscriber::DefaultGuard,
}

impl Logged {
    fn start() -> Logged {
        let lines = Arc::new(Mutex::new(Vec::new()));
        let guard =
            tracing::dispatcher::set_default(&tracing::Dispatch::new(Logger(lines.clone())));
        Logged {
            lines,
            _guard: guard,
        }
    }

    fn lines(&self, level: tracing::Level) -> Vec<String> {
        self.lines
            .lock()
            .unwrap()
            .iter()
            .filter(|(l, _)| *l == level)
            .map(|(_, m)| m.clone())
            .collect()
    }

    fn clear(&self) {
        self.lines.lock().unwrap().clear();
    }
}

// ---- store basics ------------------------------------------------------------------------------

// spec: active_record/acts_as_model_spec.rb:33 stores the registry in RubyLLM-owned records
#[tokio::test]
async fn stores_the_registry_in_rust_llm_owned_records() {
    let db = db().await;

    model_store::write(&db, &Models::new(vec![model_info()]))
        .await
        .unwrap();

    let record = find_row(&db, "test-model").await.unwrap();
    let llm = record.to_llm().unwrap();
    assert_eq!(
        (llm.id.as_str(), llm.provider.as_str()),
        ("test-model", "openai")
    );
    assert_eq!(llm.context_window, Some(128_000));
    assert_eq!(llm.modalities.input, ["text", "image"]);
    assert_eq!(llm.pricing, model_info().pricing);
    assert!(record.supports("vision").unwrap());
}

// spec: active_record/acts_as_model_spec.rb:43 updates an existing provider and model pair
#[tokio::test]
async fn updates_an_existing_provider_and_model_pair() {
    let db = db().await;
    let mut old = model_info();
    old.name = "Old".into();
    model_store::write(&db, &Models::new(vec![old]))
        .await
        .unwrap();

    model_store::write(&db, &Models::new(vec![model_info()]))
        .await
        .unwrap();

    assert_eq!(
        find_row(&db, "test-model").await.unwrap().name,
        "Test Model"
    );
    assert_eq!(rust_llm_models::Entity::find().count(&db).await.unwrap(), 1);
}

// spec: active_record/acts_as_model_spec.rb:51 reads public RubyLLM::Model values
#[tokio::test]
async fn reads_public_rust_llm_model_values() {
    let db = db().await;
    model_store::save_to_database(&db, &Models::new(vec![model_info()]))
        .await
        .unwrap();

    let read = model_store::read(&db).await;
    let model: &rust_llm::Model = read.iter().find(|m| m.id == "test-model").unwrap();

    assert_eq!(model.provider, "openai");
}

// spec: active_record/acts_as_model_spec.rb:66 reports nothing when the table is missing
#[tokio::test]
async fn reports_nothing_when_the_table_is_missing() {
    let db = db().await;
    model_store::save_to_database(&db, &Models::new(vec![model_info()]))
        .await
        .unwrap();
    db.execute_unprepared("DROP TABLE rust_llm_models")
        .await
        .unwrap();

    assert!(model_store::read(&db).await.is_empty());
}

// spec: active_record/acts_as_model_spec.rb:72 falls back to an empty registry when reading blows up
#[tokio::test]
async fn falls_back_to_an_empty_registry_when_reading_blows_up() {
    let db = db().await;
    model_store::save_to_database(&db, &Models::new(vec![model_info()]))
        .await
        .unwrap();
    // The table exists but a column the entity selects is gone: "no such column".
    db.execute_unprepared("ALTER TABLE rust_llm_models DROP COLUMN metadata")
        .await
        .unwrap();
    let logged = Logged::start();

    assert!(model_store::read(&db).await.is_empty());
    let debug = logged.lines(tracing::Level::DEBUG);
    assert!(
        debug
            .iter()
            .any(|l| l.starts_with("Failed to load models from database: ")
                && l.contains("no such column")
                && l.ends_with(", falling back to JSON")),
        "{debug:?}"
    );
}

// spec: active_record/acts_as_model_spec.rb:79 describes itself by table name
#[test]
fn describes_itself_by_table_name() {
    assert_eq!(model_store::description(), "database:rust_llm_models");
    let store = ModelStore::new(sea_orm::DatabaseConnection::default());
    assert_eq!(
        rust_llm::models::registry::ModelRegistryStore::description(&store),
        "database:rust_llm_models"
    );
}

// spec: active_record/acts_as_model_spec.rb:83 refreshes through the public registry
#[tokio::test(flavor = "multi_thread")]
async fn refreshes_through_the_public_registry() {
    let _lock = GLOBAL.lock().await;
    let _restore = Restore::new();
    let db = db().await;
    let server = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("GET"))
        .and(wiremock::matchers::path("/models.json"))
        .respond_with(wiremock::ResponseTemplate::new(200).set_body_json(json!([
            { "id": "test-model", "name": "Test Model", "provider": "openai" }
        ])))
        .expect(1)
        .mount(&server)
        .await;
    let mut config = Config::default();
    config.set(
        "model_registry_url",
        format!("{}/models.json", server.uri()),
    );
    config.model_registry_store = Some(Arc::new(ModelStore::new(db.clone())));
    rust_llm::configure(|c| *c = config);
    // Refresh keeps the current registry's models for providers the catalog does not cover.
    Models::install(Vec::new());

    let registry = model_store::refresh().await.unwrap();

    // `RubyLLM.models.refresh` ran: it fetched the catalog, saved it into the table, and the
    // global registry adopted what the table reads back.
    assert_eq!(ids(&model_store::read(&db).await), ["test-model"]);
    assert_eq!(
        registry
            .all()
            .iter()
            .map(|m| m.id.as_str())
            .collect::<Vec<_>>(),
        ["test-model"]
    );
    assert_eq!(rust_llm::models().all().len(), 1);
}

// spec: active_record/acts_as_model_spec.rb:91 builds an unsaved record from a public model
#[test]
fn builds_an_unsaved_record_from_a_public_model() {
    let record = model_store::from_llm(&model_info());

    assert!(record.id.is_not_set());
    assert_eq!(record.model_id.clone().unwrap(), "test-model");
    assert_eq!(record.unlisted_at.clone().unwrap(), None);
}

// spec: active_record/acts_as_model_spec.rb:210 defaults the JSON columns when the row leaves them null
#[test]
fn defaults_the_json_columns_when_the_row_leaves_them_null() {
    let now = chrono::Utc::now().into();
    let record = rust_llm_models::Model {
        id: 1,
        model_id: "sparse-model".into(),
        name: "Sparse".into(),
        provider: "openai".into(),
        family: None,
        model_created_at: None,
        context_window: None,
        max_output_tokens: None,
        knowledge_cutoff: None,
        unlisted_at: None,
        modalities: None,
        capabilities: None,
        pricing: None,
        metadata: None,
        created_at: now,
        updated_at: now,
    };

    let model = record.to_llm().unwrap();

    assert!(model.modalities.input.is_empty());
    assert_eq!(model.pricing, rust_llm::model::Pricing::default());
    assert_eq!(serde_json::to_value(&model.pricing).unwrap(), json!({}));
    assert!(model.metadata.is_empty());
}

// ---- when a refresh replaces the registry --------------------------------------------------------

// spec: active_record/acts_as_model_spec.rb:115 drops the models the new registry no longer carries
#[tokio::test]
async fn drops_the_models_the_new_registry_no_longer_carries() {
    let db = db().await;
    model_store::write(&db, &Models::new(vec![model_info(), dropped()]))
        .await
        .unwrap();

    model_store::write(&db, &Models::new(vec![model_info()]))
        .await
        .unwrap();

    let read = model_store::read(&db).await;
    assert!(ids(&read).contains(&"test-model"));
    assert!(!ids(&read).contains(&"dropped-model"));
    assert!(find_row(&db, "dropped-model").await.is_none());
}

// spec: active_record/acts_as_model_spec.rb:125 drops everything an empty registry leaves behind
#[tokio::test]
async fn drops_everything_an_empty_registry_leaves_behind() {
    let db = db().await;
    model_store::write(&db, &Models::new(vec![model_info()]))
        .await
        .unwrap();

    model_store::write(&db, &Models::new(vec![])).await.unwrap();

    assert!(
        !model_ids(rust_llm_models::Entity::find(), &db)
            .await
            .contains(&"test-model".to_string())
    );
}

/// `context 'when an application record still points at a dropped model'`: writes both models,
/// points a chat at the dropped one, then writes a registry without it.
async fn with_a_chat_on_a_dropped_model() -> (DatabaseConnection, ChatRecord, Logged) {
    let db = db().await;
    let logged = Logged::start();
    model_store::write(&db, &Models::new(vec![model_info(), dropped()]))
        .await
        .unwrap();
    let chat = chat_on(&db, "dropped-model").await;

    model_store::write(&db, &Models::new(vec![model_info()]))
        .await
        .unwrap();
    (db, chat, logged)
}

// spec: active_record/acts_as_model_spec.rb:144 keeps the row and marks it unlisted
#[tokio::test]
async fn keeps_the_row_and_marks_it_unlisted() {
    let (db, chat, _logged) = with_a_chat_on_a_dropped_model().await;

    assert!(dropped_record(&db).await.unlisted_at.is_some());
    assert!(
        model_ids(model_store::unlisted(), &db)
            .await
            .contains(&"dropped-model".to_string())
    );
    assert!(
        !model_ids(model_store::listed(), &db)
            .await
            .contains(&"dropped-model".to_string())
    );
    let reloaded = ChatRecord::find(&db, chat.id()).await.unwrap();
    assert_eq!(reloaded.model(&db).await.unwrap().model_id, "dropped-model");
}

// spec: active_record/acts_as_model_spec.rb:151 warns once that the model is no longer listed and may no longer work
#[tokio::test]
async fn warns_once_that_the_model_is_no_longer_listed() {
    let (_db, _chat, logged) = with_a_chat_on_a_dropped_model().await;

    let warnings = logged.lines(tracing::Level::WARN);
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    for part in [
        "1 model is",
        "openai/dropped-model",
        "no longer listed by the provider",
        "may no longer work",
        "region",
    ] {
        assert!(warnings[0].contains(part), "{part:?} in {:?}", warnings[0]);
    }
    assert_eq!(
        warnings[0],
        "1 model is no longer listed by the provider and may no longer work: openai/dropped-model. \
         The rows stay because application records still reference them. The provider may have \
         dropped them, or your configured region may not offer them."
    );
}

// spec: active_record/acts_as_model_spec.rb:158 warns once for a refresh that leaves several models unlisted
#[tokio::test]
async fn warns_once_for_a_refresh_that_leaves_several_models_unlisted() {
    let (db, _chat, logged) = with_a_chat_on_a_dropped_model().await;
    let others: Vec<rust_llm::Model> = (1..=7)
        .map(|i| model(json!({ "id": format!("gone-{i}"), "name": format!("Gone {i}"), "provider": "openai" })))
        .collect();
    let mut all = vec![model_info(), dropped()];
    all.extend(others.iter().cloned());
    model_store::write(&db, &Models::new(all)).await.unwrap();
    for other in &others {
        chat_on(&db, &other.id).await;
    }
    logged.clear();

    model_store::write(&db, &Models::new(vec![model_info()]))
        .await
        .unwrap();

    let warnings = logged.lines(tracing::Level::WARN);
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(warnings[0].contains("8 models are"), "{}", warnings[0]);
    assert!(warnings[0].contains("and 3 more"), "{}", warnings[0]);
    assert!(
        warnings[0].contains(
            "may no longer work: openai/dropped-model, openai/gone-1, openai/gone-2, \
             openai/gone-3, openai/gone-4, and 3 more."
        ),
        "{}",
        warnings[0]
    );
}

// spec: active_record/acts_as_model_spec.rb:170 keeps reporting the unlisted model, flagged as unlisted
#[tokio::test]
async fn keeps_reporting_the_unlisted_model_flagged_as_unlisted() {
    let (db, _chat, _logged) = with_a_chat_on_a_dropped_model().await;

    let read = model_store::read(&db).await;
    let model = read.iter().find(|m| m.id == "dropped-model").unwrap();
    assert!(model.is_unlisted());

    let registry = Models::new(read.clone());
    assert!(!registry.all().iter().any(|m| m.id == "dropped-model"));
    assert!(registry.unlisted().iter().any(|m| m.id == "dropped-model"));
}

// spec: active_record/acts_as_model_spec.rb:178 still finds the unlisted model by id
#[tokio::test]
async fn still_finds_the_unlisted_model_by_id() {
    let (db, _chat, _logged) = with_a_chat_on_a_dropped_model().await;

    let registry = Models::new(model_store::read(&db).await);

    assert_eq!(
        registry.find("dropped-model", Some("openai")).unwrap().id,
        "dropped-model"
    );
}

// spec: active_record/acts_as_model_spec.rb:184 keeps the first unlisting time while the model stays unlisted
#[tokio::test]
async fn keeps_the_first_unlisting_time_while_the_model_stays_unlisted() {
    let (db, _chat, _logged) = with_a_chat_on_a_dropped_model().await;
    let unlisted_at = dropped_record(&db).await.unlisted_at;
    assert!(unlisted_at.is_some());
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;

    model_store::write(&db, &Models::new(vec![model_info()]))
        .await
        .unwrap();

    assert_eq!(dropped_record(&db).await.unlisted_at, unlisted_at);
}

// spec: active_record/acts_as_model_spec.rb:192 lists the model again when a later refresh carries it
#[tokio::test]
async fn lists_the_model_again_when_a_later_refresh_carries_it() {
    let (db, _chat, _logged) = with_a_chat_on_a_dropped_model().await;

    model_store::write(&db, &Models::new(vec![model_info(), dropped()]))
        .await
        .unwrap();

    let record = dropped_record(&db).await;
    assert_eq!(record.unlisted_at, None);
    assert!(!record.to_llm().unwrap().is_unlisted());
    assert!(
        model_ids(model_store::listed(), &db)
            .await
            .contains(&"dropped-model".to_string())
    );
}

// spec: active_record/acts_as_model_spec.rb:200 still resolves the chat that points at it
#[tokio::test(flavor = "multi_thread")]
async fn still_resolves_the_chat_that_points_at_it() {
    let _lock = GLOBAL.lock().await;
    let _restore = Restore::new();
    let (db, chat, _logged) = with_a_chat_on_a_dropped_model().await;
    rust_llm::configure(|c| {
        c.set("openai_api_key", "test");
        c.model_registry_store = Some(Arc::new(ModelStore::new(db.clone())));
    });

    // `RubyLLM.models.load_from_store`, on the process-wide registry.
    let mut registry = (*rust_llm::models()).clone();
    registry.load_from_store().unwrap();
    Models::install(registry.all_including_unlisted().to_vec());

    let reloaded = ChatRecord::find(&db, chat.id()).await.unwrap();
    assert_eq!(
        reloaded.to_llm(&db).await.unwrap().model().id,
        "dropped-model"
    );
}

// ---- chat_methods_spec.rb: model assignment ------------------------------------------------------

// spec: active_record/chat_methods_spec.rb:97 fills an empty model store with the registry before adding the first chat
#[tokio::test]
async fn fills_an_empty_model_store_with_the_registry_before_adding_the_first_chat() {
    let _lock = GLOBAL.lock().await;
    let db = db().await;
    assert_eq!(rust_llm_models::Entity::find().count(&db).await.unwrap(), 0);

    let started = std::time::Instant::now();
    let chat = ChatRecord::create(&db, "gpt-4.1-nano", None).await.unwrap();
    eprintln!("first chat with registry fill: {:?}", started.elapsed());

    let registry = rust_llm::models();
    let mut stored: Vec<(String, String)> = rust_llm_models::Entity::find()
        .all(&db)
        .await
        .unwrap()
        .into_iter()
        .map(|r| (r.provider, r.model_id))
        .collect();
    let mut expected: Vec<(String, String)> = registry
        .all()
        .iter()
        .map(|m| (m.provider.clone(), m.id.clone()))
        .collect();
    stored.sort();
    expected.sort();
    expected.dedup();
    assert!(expected.len() > 1);
    assert_eq!(stored, expected);
    assert_eq!(chat.model(&db).await.unwrap().model_id, "gpt-4.1-nano");

    // The table is no longer empty, so the next chat does not refill it.
    let started = std::time::Instant::now();
    ChatRecord::create(&db, "gpt-4.1-nano", None).await.unwrap();
    eprintln!("second chat: {:?}", started.elapsed());
    assert_eq!(
        rust_llm_models::Entity::find().count(&db).await.unwrap() as usize,
        expected.len()
    );
}

// spec: active_record/chat_methods_spec.rb:163 reuses a model row another process inserted after the lookup missed
#[tokio::test]
async fn reuses_a_model_row_another_process_inserted_after_the_lookup_missed() {
    let _lock = GLOBAL.lock().await;
    let db = db().await;
    let info = rust_llm::models().find("gpt-4.1-nano", None).unwrap();
    // Another process inserted the row after our lookup missed.
    let theirs = rust_llm_loco::insert_model(&db, &info).await.unwrap();

    // The insert half of `find_or_create_model` hits the unique index and reuses their row.
    let ours = rust_llm_loco::insert_model(&db, &info).await.unwrap();

    assert_eq!(ours.id, theirs.id);
    assert_eq!(ours.model_id, "gpt-4.1-nano");
    let count = rust_llm_models::Entity::find()
        .filter(rust_llm_models::Column::ModelId.eq("gpt-4.1-nano"))
        .filter(rust_llm_models::Column::Provider.eq("openai"))
        .count(&db)
        .await
        .unwrap();
    assert_eq!(count, 1);
}
