//! RubyLLM 2.1's `evaluation_persistence_spec.rb`: evaluations over application-owned chat
//! records, and the usage rows a run writes through the ledger (`rust_llm_usages`), attributed to
//! the `with_usage_owner` owner.

use std::sync::Arc;

use rust_llm::accounting::{UsageOwner, with_usage_owner};
use rust_llm::evaluation::{Case, Evaluation, Evaluator, Outcome, RunOptions};
use rust_llm::{Config, Cost, Message, Tokens};
use rust_llm_loco::entities::rust_llm_usages;
use rust_llm_loco::{ChatRecord, UsageLedger, migrations};
use sea_orm::{Database, DatabaseConnection, EntityTrait, PaginatorTrait, QueryOrder};
use sea_orm_migration::SchemaManager;
use serde_json::json;
use tokio::sync::Mutex;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const TASK_MODEL: &str = "claude-haiku-4-5";
const JUDGMENT_MODEL: &str = "jev-latest";

async fn db() -> DatabaseConnection {
    let db = Database::connect("sqlite::memory:").await.unwrap();
    let manager = SchemaManager::new(&db);
    for m in migrations() {
        m.up(&manager).await.unwrap();
    }
    db
}

fn config(server: &MockServer, db: &DatabaseConnection) -> Arc<Config> {
    let mut c = Config::default();
    for (provider, base) in [("anthropic", ""), ("typesafe", ""), ("openai", "/v1")] {
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

// spec: evaluation_persistence_spec.rb:8
#[tokio::test]
async fn evaluates_application_owned_chat_records_through_the_existing_conversion_boundary() {
    let db = Arc::new(db().await);
    let server = MockServer::start().await;
    let config = config(&server, &db);
    let record = ChatRecord::create(&db, "gpt-5-nano", None).await.unwrap();
    let mut chat = record.to_llm_with(&db, config.clone()).await.unwrap();
    record
        .add_message(&db, &mut chat, Message::user("Hello"))
        .await
        .unwrap();
    let mut welcome = Message::assistant("Welcome");
    welcome.finish_reason = Some(rust_llm::FinishReason::Stop);
    record.add_message(&db, &mut chat, welcome).await.unwrap();

    let mut e = Evaluation::new();
    e.without_evaluator();
    let (handle, cfg) = (db.clone(), config.clone());
    // `Chat.find(input)`, converted with `to_llm` (Ruby's `conversation.to_llm`).
    e.perform(move |i| {
        let (db, config) = (handle.clone(), cfg.clone());
        async move {
            let id = i.input().as_i64().unwrap_or_default() as i32;
            let record = ChatRecord::find(&db, id)
                .await
                .map_err(|e| rust_llm::evaluation::Failure::error(e.to_string()))?;
            let chat = record
                .to_llm_with(&db, config)
                .await
                .map_err(|e| rust_llm::evaluation::Failure::error(e.to_string()))?;
            Ok(Outcome::from(chat))
        }
    });
    e.assertions(|a| {
        a.assert_equal("Welcome", a.output().clone())?;
        let roles: Vec<&str> = a.messages().iter().map(|m| m.role.as_str()).collect();
        a.assert_equal(json!(["user", "assistant"]), json!(roles))?;
        Ok(())
    });
    let cases = vec![Case::new("persisted", record.id()).unwrap()];
    let report = e
        .run_with(RunOptions::default().dataset(cases))
        .await
        .unwrap();
    assert!(report.is_passed(), "{}", report.to_h());
    assert!(report.first().unwrap().result().unwrap().chat().is_some());
    assert_eq!(
        report.first().unwrap().evidence().unwrap()["messages"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
}

fn row_tokens(r: &rust_llm_usages::Model) -> Tokens {
    Tokens {
        input: r.input_tokens.map(i64::from),
        output: r.output_tokens.map(i64::from),
        cache_read: r.cache_read_tokens.map(i64::from),
        cache_write: r.cache_write_tokens.map(i64::from),
        thinking: r.thinking_tokens.map(i64::from),
        ..Default::default()
    }
}

fn tokens_h(t: &Tokens) -> serde_json::Value {
    let mut h = serde_json::Map::new();
    for (k, v) in [
        ("input_tokens", t.input),
        ("output_tokens", t.output),
        ("cache_read_tokens", t.cache_read),
        ("cache_write_tokens", t.cache_write),
        ("thinking_tokens", t.thinking),
    ] {
        if let Some(v) = v {
            h.insert(k.into(), v.into());
        }
    }
    h.into()
}

async fn records_task_and_native_evaluator_attempts_once(persisted: bool) {
    let db = Arc::new(db().await);
    let server = MockServer::start().await;
    let config = config(&server, &db);
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_1", "type": "message", "role": "assistant", "model": TASK_MODEL,
            "content": [{ "type": "text", "text": "Hello" }], "stop_reason": "end_turn",
            "usage": { "input_tokens": 9, "output_tokens": 2 }
        })))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model": JUDGMENT_MODEL, "answers": { "correct": { "type": "noul", "noul": 0.9 } },
            "usage": { "input_tokens": 11, "output_tokens": 1 }
        })))
        .mount(&server)
        .await;
    let owner = ChatRecord::create(&db, TASK_MODEL, None).await.unwrap();
    let record = if persisted {
        Some(ChatRecord::create(&db, TASK_MODEL, None).await.unwrap())
    } else {
        None
    };
    let chat = match &record {
        Some(r) => r.to_llm_with(&db, config.clone()).await.unwrap(),
        None => rust_llm::Chat::with_config(config.clone(), Some(TASK_MODEL), None, false).unwrap(),
    };
    let shared = Arc::new(Mutex::new(chat));
    let record = Arc::new(record);
    let mut e = Evaluation::new();
    e.with_config(config.clone());
    e.evaluator(Evaluator::model(JUDGMENT_MODEL).provider("typesafe"));
    e.evaluation_with(
        "correct",
        Some("The output greets the user"),
        Some(0.8),
        None,
    )
    .unwrap();
    let (chat, rec, handle) = (shared.clone(), record.clone(), db.clone());
    e.perform(move |i| {
        let (chat, record, db) = (chat.clone(), rec.clone(), handle.clone());
        async move {
            let mut chat = chat.lock().await;
            let input = i.input().as_str().unwrap_or_default().to_string();
            let answer = match &*record {
                Some(r) => r
                    .ask(&db, &mut chat, &input)
                    .await
                    .map_err(|e| rust_llm::evaluation::Failure::error(e.to_string()))?,
                None => chat.ask(input).await?,
            };
            Ok(Outcome::Value(answer.content().into()))
        }
    });
    let dataset = vec![Case::new("greeting", "Hi").unwrap()];
    let before = rust_llm_usages::Entity::find().count(&*db).await.unwrap();
    let owner_ref = UsageOwner::record("Chat", i64::from(owner.id()));
    let report = with_usage_owner(
        owner_ref.clone(),
        e.run_with(RunOptions::default().dataset(dataset)),
    )
    .await
    .unwrap();
    let rows = rust_llm_usages::Entity::find()
        .order_by_asc(rust_llm_usages::Column::Id)
        .all(&*db)
        .await
        .unwrap();
    assert_eq!(rows.len() as u64 - before, 2);
    let rows = &rows[rows.len() - 2..];
    assert!(report.is_passed(), "{}", report.to_h());
    let operations: Vec<&str> = rows.iter().map(|r| r.operation.as_str()).collect();
    assert_eq!(operations, ["chat", "judgment"]);
    let owner_pair = (Some("Chat".to_string()), Some(i64::from(owner.id())));
    assert_eq!((rows[1].owner_type.clone(), rows[1].owner_id), owner_pair);
    assert_eq!(rows[1].chat_id, None);
    assert_eq!(
        rows[0].chat_id,
        record.as_ref().as_ref().map(|r| i64::from(r.id()))
    );
    assert_eq!((rows[0].owner_type.clone(), rows[0].owner_id), owner_pair);
    assert_eq!(
        tokens_h(&report.tokens()),
        json!({ "input_tokens": 20, "output_tokens": 3 })
    );
    let row_tokens: Vec<Tokens> = rows.iter().map(row_tokens).collect();
    assert_eq!(
        tokens_h(&report.tokens()),
        tokens_h(&Tokens::aggregate(&row_tokens))
    );
    let task = report.first().unwrap().task_cost().total().unwrap();
    assert!((task - rows[0].total_cost.unwrap()).abs() < 1e-10);
    // `evaluator_cost.to_h == rows.last.cost.to_h`.
    let evaluator: Cost = report.first().unwrap().evaluator_cost();
    assert_eq!(evaluator.total(), rows[1].total_cost);
    assert_eq!(evaluator.input, rows[1].input_cost);
    assert_eq!(evaluator.output, rows[1].output_cost);
    // RubyLLM's registry leaves `jev-latest` unpriced, so its total is unknown; the port's
    // bundled registry prices it (see judge.rs), so the run's total is known here.
    assert!(report.cost().total().is_some());
}

// spec: evaluation_persistence_spec.rb:33
#[tokio::test]
async fn records_task_and_native_evaluator_attempts_once_with_a_persisted_chat() {
    records_task_and_native_evaluator_attempts_once(false).await;
    records_task_and_native_evaluator_attempts_once(true).await;
}
