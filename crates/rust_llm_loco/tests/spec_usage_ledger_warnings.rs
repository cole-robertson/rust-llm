//! The tests of `spec/ruby_llm/active_record/usage_ledger_spec.rb` that read `tracing` WARN
//! events, on their own: a callsite's interest is cached process-wide, and other tests
//! registering the same callsites concurrently made these fail now and then. The shared helpers
//! are in `ledger_support`.

mod ledger_support;

use ledger_support::*;
use rust_llm_loco::ChatRecord;
use rust_llm_loco::migrations;
use sea_orm::{Database, EntityTrait, PaginatorTrait, TransactionTrait};
use sea_orm_migration::SchemaManager;
use wiremock::MockServer;

// spec: active_record/usage_ledger_spec.rb:214 logs a failed chat row write and returns the reply
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

// spec: active_record/usage_ledger_spec.rb:225 logs a failed write and returns the result of the operation
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

// spec: active_record/usage_ledger_spec.rb:234 keeps the caller transaction usable when a write fails
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

// spec: active_record/usage_ledger_spec.rb:258 when a chat row cannot be linked to its message > logs the failed link and returns the reply
#[tokio::test]
async fn logs_the_failed_link_and_returns_the_reply() {
    let db = db().await;
    let server = MockServer::start().await;
    stub_chat_reply(&server).await;
    let (record, mut chat) = unlinkable_chat(&db, &server).await;
    let warnings = Warnings::default();
    let _guard = tracing::subscriber::set_default(warnings.clone());

    let reply = record.ask(&db, &mut chat, "Hello").await.unwrap();

    assert_eq!(reply.content(), "Hi");
    assert_eq!(
        roles_and_contents(&db, &record).await,
        [
            ("user".into(), "Hello".into()),
            ("assistant".into(), "Hi".into())
        ]
    );
    assert!(
        warnings
            .lines()
            .iter()
            .any(|l| l.contains("could not link usage to message")
                && l.contains("usage link rejected")),
        "{:?}",
        warnings.lines()
    );
}
