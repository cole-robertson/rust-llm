//! Ports of RubyLLM's generator specs that need the generated code to run, not just exist:
//! `install_generator_spec.rb` (schema checks and the chat round trip), `chat_ui_generator_spec.rb`
//! (the chat round trip), `upgrade_generator_spec.rb`, `provider/cli_spec.rb`,
//! `provider/scaffold_spec.rb`, and `active_record/message_methods_spec.rb`'s `tool_error_message`.
//!
//! The install and upgrade migrations and the chat/message models are templates the generators
//! copy verbatim into the app, so each test first checks the generated file equals its template,
//! then compiles that template here (`#[path]` modules below) and runs it against SQLite, the way
//! RubyLLM's specs run the generated Rails app.

use std::fs;
use std::path::Path;
use std::process::{Command, Output};

use rust_llm_cli::{Generator, chat_ui, install, provider, upgrade};
use sea_orm::{ConnectionTrait, Database, DatabaseConnection, DbBackend, QueryResult, Statement};
use sea_orm_migration::{MigrationTrait, SchemaManager};

mod support;
use support::app;

// The generated files, compiled as they are written into the app. `rustfmt::skip` keeps the
// workspace's `cargo fmt` from restyling the templates for this crate's edition.
#[rustfmt::skip]
#[path = "../templates/install/migration.rs"]
mod install_migration;
#[rustfmt::skip]
#[path = "../templates/upgrade/migration.rs"]
mod upgrade_migration;
#[rustfmt::skip]
#[allow(dead_code, unused_imports)] // Items the generated app's controllers use.
#[path = "../templates/install/chat_model.rs"]
mod chats;
#[rustfmt::skip]
#[allow(dead_code, unused_imports)] // Items the generated app's controllers use.
#[path = "../templates/install/message_model.rs"]
mod messages;
#[rustfmt::skip]
#[path = "../templates/chat_ui/migration.rs"]
mod chat_ui_migration;
/// `src/models/chats.rs` after `rust-llm generate chat_ui`: the install template with the
/// account functions appended (see `chat_ui_app_scopes_chats_to_their_account`).
/// The appended part sees the install template's items, as it does in the one file.
#[rustfmt::skip]
#[allow(dead_code, unused_imports)]
mod account_chats {
    pub use super::chats::*;
    use rust_llm_loco::entities::{messages, rust_llm_models};
    use sea_orm::sea_query::{Expr, ExprTrait};
    use sea_orm::{ColumnTrait, DatabaseConnection, DbErr, EntityTrait, QueryFilter, QueryOrder, QuerySelect};
    use serde_json::{json, Value};
    include!("../templates/chat_ui/chat_model_accounts.rs");
}

const MODEL: &str = "gpt-4.1-nano";

fn read(root: &Path, rel: &str) -> String {
    fs::read_to_string(root.join(rel)).unwrap_or_else(|e| panic!("{rel}: {e}"))
}

/// The one generated migration whose file name ends with `suffix`.
fn generated_migration(root: &Path, suffix: &str) -> String {
    let names: Vec<String> = fs::read_dir(root.join("migration/src"))
        .unwrap()
        .filter_map(|e| e.unwrap().file_name().into_string().ok())
        .filter(|n| n.ends_with(&format!("{suffix}.rs")))
        .collect();
    assert_eq!(names.len(), 1, "{names:?}");
    read(root, &format!("migration/src/{}", names[0]))
}

fn installed_app() -> tempfile::TempDir {
    let dir = app();
    let mut g = Generator::new(dir.path(), false);
    install::generate(&mut g, None).unwrap();
    assert!(g.failures.is_empty(), "{:?}", g.failures);
    let root = dir.path();
    assert_eq!(
        generated_migration(root, install::MIGRATION_SUFFIX),
        include_str!("../templates/install/migration.rs")
    );
    assert_eq!(
        read(root, "src/models/chats.rs"),
        include_str!("../templates/install/chat_model.rs")
    );
    assert_eq!(
        read(root, "src/models/messages.rs"),
        include_str!("../templates/install/message_model.rs")
    );
    dir
}

/// A database migrated by the generated install migration.
async fn installed_db() -> DatabaseConnection {
    let db = Database::connect("sqlite::memory:").await.unwrap();
    install_migration::Migration
        .up(&SchemaManager::new(&db))
        .await
        .unwrap();
    db
}

async fn query(db: &DatabaseConnection, sql: &str) -> Vec<QueryResult> {
    db.query_all_raw(Statement::from_string(DbBackend::Sqlite, sql))
        .await
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
}

async fn strings(db: &DatabaseConnection, sql: &str, column: &str) -> Vec<String> {
    query(db, sql)
        .await
        .iter()
        .map(|row| row.try_get::<String>("", column).unwrap())
        .collect()
}

async fn table_sql(db: &DatabaseConnection, table: &str) -> String {
    strings(
        db,
        &format!("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = '{table}'"),
        "sql",
    )
    .await
    .pop()
    .unwrap_or_else(|| panic!("no table {table}"))
}

/// `PRAGMA table_info` for one column: `(type, notnull, dflt_value)`.
async fn column(
    db: &DatabaseConnection,
    table: &str,
    name: &str,
) -> (String, bool, Option<String>) {
    let rows = query(db, &format!("PRAGMA table_info({table})")).await;
    let row = rows
        .iter()
        .find(|r| r.try_get::<String>("", "name").unwrap() == name)
        .unwrap_or_else(|| panic!("{table}.{name} missing"));
    (
        row.try_get("", "type").unwrap(),
        row.try_get::<i32>("", "notnull").unwrap() == 1,
        row.try_get("", "dflt_value").unwrap(),
    )
}

/// The column lists of a table's indexes, like `connection.indexes(table).map(&:columns)`.
async fn index_columns(db: &DatabaseConnection, table: &str) -> Vec<Vec<String>> {
    let mut out = Vec::new();
    for name in strings(db, &format!("PRAGMA index_list({table})"), "name").await {
        out.push(strings(db, &format!("PRAGMA index_info('{name}')"), "name").await);
    }
    out
}

async fn schema(db: &DatabaseConnection) -> Vec<String> {
    strings(
        db,
        "SELECT type || ' ' || name || ': ' || coalesce(sql, '') AS entry FROM sqlite_master ORDER BY name",
        "entry",
    )
    .await
}

// spec: generators/install_generator_spec.rb:73 creates internal records before the chat foreign key
#[tokio::test]
async fn install_creates_internal_records_before_the_chat_foreign_key() {
    // Ruby sorts the migration files; here the one generated migration runs
    // `rust_llm_loco::migrations()` in order, so the order tables are created in is the check:
    // `rust_llm_models` exists before `chats`, whose foreign key points at it.
    installed_app();
    let db = installed_db().await;
    let created = strings(
        &db,
        "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY rowid",
        "name",
    )
    .await;
    let at = |t: &str| created.iter().position(|n| n == t).unwrap();
    assert!(at("rust_llm_models") < at("chats"), "{created:?}");
    let targets = strings(&db, "PRAGMA foreign_key_list(chats)", "table").await;
    assert_eq!(targets, ["rust_llm_models"]);
}

// spec: generators/install_generator_spec.rb:83 stores tool call thought signatures and defaults calls to local execution
#[tokio::test]
async fn install_stores_thought_signatures_and_defaults_tool_calls_to_local() {
    let db = installed_db().await;
    let (kind, not_null, _) = column(&db, "rust_llm_tool_calls", "thought_signature").await;
    assert_eq!((kind.to_lowercase().as_str(), not_null), ("text", false));
    // `t.boolean :remote, default: false, null: false`
    let (kind, not_null, default) = column(&db, "rust_llm_tool_calls", "remote").await;
    assert_eq!(kind.to_lowercase(), "boolean");
    assert!(not_null);
    assert_eq!(
        default.as_deref().map(str::to_lowercase).as_deref(),
        Some("false")
    );
    query(
        &db,
        "INSERT INTO rust_llm_tool_calls (message_type, message_id, tool_call_id, name) VALUES ('Message', 1, 'call_1', 'lookup')",
    )
    .await;
    let remote = query(&db, "SELECT remote FROM rust_llm_tool_calls").await;
    assert!(!remote[0].try_get::<bool>("", "remote").unwrap());
    // Ruby's example also checks `t.json :reported_cost`, which lives on the batches table:
    // see `install_runs_the_batches_and_mcp_credentials_migrations`.
}

// spec: generators/install_generator_spec.rb:95 constrains the usage ledger to the operations and statuses RubyLLM records
#[tokio::test]
async fn install_constrains_usage_operations_and_statuses() {
    use rust_llm::message::{Operation, UsageStatus};
    let db = installed_db().await;
    let sql = table_sql(&db, "rust_llm_usages").await;
    assert!(
        column(&db, "rust_llm_usages", "model").await.1,
        "model NOT NULL"
    );
    // `Accounting::Usage::Entry::OPERATIONS` / `STATUSES`.
    let operations = [
        Operation::Chat,
        Operation::Embedding,
        Operation::Moderation,
        Operation::Image,
        Operation::Speech,
        Operation::Transcription,
        Operation::Ocr,
        Operation::Rerank,
        Operation::Judgment,
    ];
    let statuses = [
        UsageStatus::Pending,
        UsageStatus::Succeeded,
        UsageStatus::Failed,
        UsageStatus::Cancelled,
    ];
    for value in operations
        .iter()
        .map(Operation::as_str)
        .chain(statuses.iter().map(UsageStatus::as_str))
    {
        assert!(
            sql.contains(&format!("'{value}'")),
            "{value} missing from {sql}"
        );
    }

    let insert = |operation: &str, status: &str| {
        format!(
            "INSERT INTO rust_llm_usages (chat_type, chat_id, operation, provider, model, status) VALUES ('Chat', 1, '{operation}', 'openai', '{MODEL}', '{status}')"
        )
    };
    let stmt = |sql: String| Statement::from_string(DbBackend::Sqlite, sql);
    db.execute_raw(stmt(insert("judgment", "cancelled")))
        .await
        .unwrap();
    let bad_operation = db.execute_raw(stmt(insert("training", "pending"))).await;
    assert!(bad_operation.unwrap_err().to_string().contains("CHECK"));
    let bad_status = db.execute_raw(stmt(insert("chat", "done"))).await;
    assert!(bad_status.unwrap_err().to_string().contains("CHECK"));
}

// spec: generators/install_generator_spec.rb:116 indexes messages by chat without a standalone role index
#[tokio::test]
async fn install_indexes_messages_by_chat_without_a_role_index() {
    let db = installed_db().await;
    let indexes = index_columns(&db, "messages").await;
    assert!(
        indexes.contains(&vec!["chat_id".to_string()]),
        "{indexes:?}"
    );
    assert!(!indexes.contains(&vec!["role".to_string()]), "{indexes:?}");
}

// spec: generators/install_generator_spec.rb:83 (`t.json :reported_cost` on the batches table)
#[tokio::test]
async fn install_runs_the_batches_and_mcp_credentials_migrations() {
    let db = installed_db().await;
    let manager = SchemaManager::new(&db);
    assert!(manager.has_table("rust_llm_mcp_credentials").await.unwrap());
    assert!(
        index_columns(&db, "rust_llm_mcp_credentials")
            .await
            .contains(&vec!["key".to_string()])
    );
    assert!(
        manager.has_table("rust_llm_batches").await.unwrap(),
        "rust_llm_loco::migrations() should create rust_llm_batches"
    );
    assert_eq!(
        column(&db, "rust_llm_batches", "reported_cost").await.0,
        column(&db, "rust_llm_tool_calls", "pending_input").await.0,
        "reported_cost is a json column"
    );
}

/// `Chat.create!` then `chat.messages.create!(role: :user, content: 'Test')`, through the
/// generated model files, and `message.chat_id == chat.id`.
async fn chat_round_trip(db: &DatabaseConnection) -> (i32, messages::Model) {
    let chat = chats::ChatRecord::create(db, MODEL, None).await.unwrap();
    let message = messages::create_user(db, chat.id(), "Test").await.unwrap();
    assert_eq!(message.chat_id, chat.id());
    (chat.id(), message)
}

// spec: generators/install_generator_spec.rb:183 chat functionality works correctly
#[tokio::test]
async fn installed_chat_and_message_models_work() {
    installed_app();
    let db = installed_db().await;
    let (_, message) = chat_round_trip(&db).await;
    assert_eq!(message.role, "user");
    assert_eq!(message.content.as_deref(), Some("Test"));
}

/// The chat UI's account functions, compiled as `src/models/chats.rs` ends up after
/// `chat_ui`, against a schema with the chat UI's migration applied.
#[tokio::test]
async fn chat_ui_app_scopes_chats_to_their_account() {
    let db = installed_db().await;
    chat_ui_migration::Migration
        .up(&SchemaManager::new(&db))
        .await
        .unwrap();
    let (acme, globex) = (1_i64, 2_i64);
    let mine = account_chats::create_in_account(&db, acme, MODEL, None)
        .await
        .unwrap();
    let theirs = account_chats::create_in_account(&db, globex, MODEL, None)
        .await
        .unwrap();
    messages::create_user(&db, mine.id(), "Hi").await.unwrap();

    assert!(account_chats::find_in_account(&db, acme, mine.id()).await.unwrap().is_some());
    assert!(
        account_chats::find_in_account(&db, acme, theirs.id()).await.unwrap().is_none(),
        "another account's chat is not found"
    );
    let listed = account_chats::list_in_account(&db, acme).await.unwrap();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0]["id"], mine.id());
    assert_eq!(listed[0]["message_count"], 1);
    // An unknown model creates nothing.
    assert!(account_chats::create_in_account(&db, acme, "no-such-model", None).await.is_err());
    assert_eq!(account_chats::list_in_account(&db, acme).await.unwrap().len(), 1);

    chat_ui_migration::Migration
        .down(&SchemaManager::new(&db))
        .await
        .unwrap();
    assert!(!SchemaManager::new(&db).has_column("chats", "account_id").await.unwrap());
}

// spec: generators/chat_ui_generator_spec.rb:458 chat functionality works correctly
#[tokio::test]
async fn chat_ui_app_creates_a_chat_and_its_first_message() {
    // Ruby runs its functionality script (`Chat.create!`, `chat.messages.create!`) in the app the
    // chat UI was generated into. The chat UI's controllers and worker need loco-rs (they run
    // end to end in tests/e2e/chat_flow.sh); here the model functions they call run for real.
    let dir = installed_app();
    let mut g = Generator::new(dir.path(), false);
    chat_ui::generate(&mut g).unwrap();
    assert!(g.failures.is_empty(), "{:?}", g.failures);
    // The model functions below are the ones the generated controllers call.
    let controllers = read(dir.path(), "src/controllers/chats.rs");
    for call in [
        "chats::create_in_account(&ctx.db",
        "messages::create_user(&ctx.db",
        "chats::list_in_account(&ctx.db",
        "chats::find_props(&ctx.db",
        "messages::transcript(&ctx.db",
        "chats::destroy(&ctx.db",
    ] {
        assert!(controllers.contains(call), "{call}");
    }

    let db = installed_db().await;
    let (chat_id, _) = chat_round_trip(&db).await;
    let listed = chats::list(&db).await.unwrap();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0]["id"], chat_id);
    assert_eq!(listed[0]["message_count"], 1);
    assert_eq!(
        chats::find_props(&db, chat_id).await.unwrap().unwrap()["id"],
        chat_id
    );
    let transcript = messages::transcript(&db, chat_id).await.unwrap();
    assert_eq!(transcript.messages.len(), 1);
    assert_eq!(transcript.messages[0]["content"], "Test");
    assert!(
        transcript.awaiting_response,
        "the worker has not answered yet"
    );

    chats::destroy(&db, chat_id).await.unwrap();
    assert!(chats::find_props(&db, chat_id).await.unwrap().is_none());
    assert!(chats::list(&db).await.unwrap().is_empty());
}

// spec: active_record/message_methods_spec.rb:17 extracts error from hash content
#[test]
fn tool_error_message_reads_an_error_object() {
    // Ruby assigns a Hash (`{ error: 'tool failed' }`); the Rust column is text, so a JSON object
    // is how that content arrives.
    let content = serde_json::json!({ "error": "tool failed" }).to_string();
    assert_eq!(
        messages::tool_error_message(Some(&content)).as_deref(),
        Some("tool failed")
    );
}

// spec: active_record/message_methods_spec.rb:22 extracts error from JSON string content
#[test]
fn tool_error_message_reads_json_string_content() {
    assert_eq!(
        messages::tool_error_message(Some(r#"{"error":"tool failed"}"#)).as_deref(),
        Some("tool failed")
    );
}

// spec: active_record/message_methods_spec.rb:37 returns nil for invalid content
#[test]
fn tool_error_message_is_none_for_non_json_content() {
    assert_eq!(messages::tool_error_message(Some("not-json")), None);
}

/// A schema generated by 2.0's install, then brought to 2.1 by the generated upgrade migration.
fn upgraded_app() -> tempfile::TempDir {
    let dir = installed_app();
    let mut g = Generator::new(dir.path(), false);
    upgrade::generate(&mut g).unwrap();
    assert!(g.failures.is_empty(), "{:?}", g.failures);
    assert_eq!(
        generated_migration(dir.path(), upgrade::MIGRATION_SUFFIX),
        include_str!("../templates/upgrade/migration.rs")
    );
    dir
}

// spec: generators/upgrade_generator_spec.rb:20 adds what 2.1 needs to a 2.0 schema
#[tokio::test]
async fn upgrade_adds_what_2_1_needs_to_a_2_0_schema() {
    upgraded_app();
    let db = installed_db().await;
    query(&db, "DROP TABLE rust_llm_mcp_credentials").await;
    query(
        &db,
        "ALTER TABLE rust_llm_tool_calls DROP COLUMN pending_input",
    )
    .await;
    let manager = SchemaManager::new(&db);
    assert!(!manager.has_table("rust_llm_mcp_credentials").await.unwrap());

    upgrade_migration::Migration.up(&manager).await.unwrap();

    assert!(manager.has_table("rust_llm_mcp_credentials").await.unwrap());
    assert!(
        manager
            .has_column("rust_llm_tool_calls", "pending_input")
            .await
            .unwrap()
    );
    // The same shape install creates: a unique key and a polymorphic owner.
    let indexes = index_columns(&db, "rust_llm_mcp_credentials").await;
    assert!(indexes.contains(&vec!["key".to_string()]), "{indexes:?}");
    assert!(
        indexes.contains(&vec!["owner_type".to_string(), "owner_id".to_string()]),
        "{indexes:?}"
    );
    assert_eq!(
        column(&db, "rust_llm_tool_calls", "pending_input").await.0,
        column(
            &installed_db().await,
            "rust_llm_tool_calls",
            "pending_input"
        )
        .await
        .0
    );
}

// spec: generators/upgrade_generator_spec.rb:35 leaves an up-to-date schema alone
#[tokio::test]
async fn upgrade_leaves_an_up_to_date_schema_alone() {
    upgraded_app();
    let db = installed_db().await;
    let before = schema(&db).await;
    upgrade_migration::Migration
        .up(&SchemaManager::new(&db))
        .await
        .unwrap();
    assert_eq!(schema(&db).await, before);
}

fn rust_llm_bin(args: &[&str], cwd: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_rust-llm"))
        .args(args)
        .current_dir(cwd)
        .output()
        .unwrap()
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

// spec: generators/provider/cli_spec.rb:82 prints help and succeeds
#[test]
fn bare_invocation_prints_help_and_succeeds() {
    let dir = tempfile::tempdir().unwrap();
    let out = rust_llm_bin(&[], dir.path());
    assert_eq!(out.status.code(), Some(0));
    assert!(text(&out.stdout).contains("rust-llm generate provider NAME"));
}

// spec: generators/provider/cli_spec.rb:89 prints help for the help command
#[test]
fn help_command_prints_usage() {
    let dir = tempfile::tempdir().unwrap();
    let out = rust_llm_bin(&["help"], dir.path());
    assert_eq!(out.status.code(), Some(0));
    assert!(text(&out.stdout).contains("Usage:"));
}

// spec: generators/provider/cli_spec.rb:130 reports extra arguments
#[test]
fn extra_arguments_are_reported() {
    let dir = tempfile::tempdir().unwrap();
    let out = rust_llm_bin(&["generate", "provider", "acme", "extra"], dir.path());
    assert_eq!(out.status.code(), Some(1));
    assert!(
        text(&out.stderr).contains("Unexpected arguments: extra"),
        "{}",
        text(&out.stderr)
    );
    assert!(
        fs::read_dir(dir.path()).unwrap().next().is_none(),
        "nothing written"
    );

    let app = app();
    let out = rust_llm_bin(&["generate", "install", "chat", "message"], app.path());
    assert_eq!(out.status.code(), Some(1));
    assert!(text(&out.stderr).contains("Unexpected arguments: chat message"));
}

/// The core files the provider generator rewrites, copied from this checkout like Ruby's
/// `create_core_fixture` copies `lib/ruby_llm.rb` and `lib/ruby_llm/models.rb`.
fn core_fixture() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let checkout = Path::new(env!("CARGO_MANIFEST_DIR")).join("../rust_llm/src");
    for rel in ["providers.rs", "models/refresh.rs"] {
        let to = dir.path().join("crates/rust_llm/src").join(rel);
        fs::create_dir_all(to.parent().unwrap()).unwrap();
        fs::copy(checkout.join(rel), to).unwrap();
    }
    dir
}

const REFRESH: &str = "crates/rust_llm/src/models/refresh.rs";
const PROVIDERS: &str = "crates/rust_llm/src/providers.rs";

fn models_dev_map(root: &Path) -> Vec<String> {
    read(root, REFRESH)
        .lines()
        .skip_while(|l| !l.contains("const MODELS_DEV_PROVIDER_MAP"))
        .skip(1)
        .take_while(|l| l.starts_with("    (\""))
        .map(str::to_string)
        .collect()
}

// spec: generators/provider/cli_spec.rb:168 accepts the core-only models.dev option
#[test]
fn cli_accepts_models_dev_provider() {
    let dir = core_fixture();
    let out = rust_llm_bin(
        &[
            "generate",
            "provider",
            "acme",
            "--models-dev-provider",
            "acme",
        ],
        dir.path(),
    );
    assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
    assert!(
        models_dev_map(dir.path()).contains(&"    (\"acme\", \"acme\"),".to_string()),
        "{:?}",
        models_dev_map(dir.path())
    );
}

fn options<'a>(
    api_base: Option<&'a str>,
    models_dev_provider: Option<&'a str>,
) -> provider::Options<'a> {
    provider::Options {
        dialect: None,
        api_base,
        models_dev_provider,
        dynamic_models: false,
    }
}

// spec: generators/provider/scaffold_spec.rb:174 keeps core wiring valid and sorted
#[test]
fn provider_core_wiring_stays_valid_and_sorted() {
    let dir = core_fixture();
    let before = models_dev_map(dir.path());
    for name in ["Hetzner", "Zeta"] {
        let api_base = format!("https://{}.example/v1", name.to_lowercase());
        let key = name.to_lowercase();
        let mut g = Generator::new(dir.path(), false);
        provider::generate(&mut g, name, &options(Some(&api_base), Some(&key))).unwrap();
        assert!(g.failures.is_empty(), "{:?}", g.failures);
    }

    // `ruby -c` on each rewritten file.
    for rel in [PROVIDERS, REFRESH] {
        syn::parse_file(&read(dir.path(), rel))
            .unwrap_or_else(|e| panic!("{rel} does not parse: {e}"));
    }
    let mods: Vec<String> = read(dir.path(), PROVIDERS)
        .lines()
        .filter(|l| l.starts_with("pub mod "))
        .map(str::to_string)
        .collect();
    assert_eq!(mods, ["pub mod hetzner;", "pub mod zeta;"]);

    // Hetzner is already mapped, so only Zeta's entry is added, once, after the last smaller one.
    let after = models_dev_map(dir.path());
    assert_eq!(after.len(), before.len() + 1);
    assert_eq!(
        after.iter().filter(|l| l.contains("\"hetzner\"")).count(),
        1
    );
    assert_eq!(after.last().unwrap(), "    (\"zeta\", \"zeta\"),");
}

fn generated_provider(root: &Path, slug: &str) -> String {
    read(root, &format!("crates/rust_llm/src/providers/{slug}.rs"))
}

// spec: generators/provider/scaffold_spec.rb:300 keeps a name that is already a class name
#[test]
fn provider_keeps_a_class_cased_name() {
    let dir = tempfile::tempdir().unwrap();
    let mut g = Generator::new(dir.path(), false);
    provider::generate(&mut g, "OpenAI", &options(None, None)).unwrap();
    assert!(
        generated_provider(dir.path(), "open_ai").contains("pub const DISPLAY: &str = \"OpenAI\";")
    );
}

// spec: generators/provider/scaffold_spec.rb:304 fills in the defaults the CLI does not pass
#[test]
fn provider_fills_in_defaults() {
    let dir = core_fixture();
    let before = read(dir.path(), REFRESH);
    let mut g = Generator::new(dir.path(), false);
    provider::generate(&mut g, "Acme", &options(None, None)).unwrap();
    let module = generated_provider(dir.path(), "acme");
    assert!(module.contains("pub const DEFAULT_API_BASE: &str = \"https://api.example.com/v1\";"));
    assert!(module.contains("pub const ASSUME_MODELS_EXIST: bool = false;"));
    assert!(
        g.notes
            .iter()
            .any(|n| n.contains("ACME_API_KEY / ACME_API_BASE")),
        "{:?}",
        g.notes
    );
    // `models_dev_provider` is nil: the models.dev map is left alone. (`gem_name` and
    // `github_owner` belong to the provider-gem mode, which is not ported.)
    assert_eq!(read(dir.path(), REFRESH), before);
}

// spec: generators/provider/scaffold_spec.rb:325 treats a blank models.dev provider as none
#[test]
fn provider_treats_a_blank_models_dev_provider_as_none() {
    let dir = core_fixture();
    let before = read(dir.path(), REFRESH);
    let mut g = Generator::new(dir.path(), false);
    provider::generate(&mut g, "Acme", &options(None, Some("  "))).unwrap();
    assert!(g.failures.is_empty(), "{:?}", g.failures);
    assert_eq!(read(dir.path(), REFRESH), before);
}

// spec: generators/provider/scaffold_spec.rb:340 writes the provider without touching files that are not there
#[test]
fn provider_is_written_without_the_core_files() {
    let dir = tempfile::tempdir().unwrap();
    let mut g = Generator::new(dir.path(), false);
    provider::generate(&mut g, "Acme", &options(None, Some("acme"))).unwrap();
    assert!(g.failures.is_empty(), "{:?}", g.failures);
    assert_eq!(
        g.actions,
        [
            (
                "create".to_string(),
                "crates/rust_llm/src/providers/acme.rs".to_string()
            ),
            (
                "create".to_string(),
                "crates/rust_llm/tests/provider_acme.rs".to_string()
            ),
        ]
    );
    assert!(!dir.path().join(PROVIDERS).exists());
    assert!(!dir.path().join(REFRESH).exists());
}
