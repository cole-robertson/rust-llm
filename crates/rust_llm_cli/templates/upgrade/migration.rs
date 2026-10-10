//! Upgrades a RustLLM 2.0 schema to 2.1, from `rust-llm generate upgrade` (RubyLLM's
//! `upgrade_ruby_llm_to_2_1.rb.tt`). Each change applies only when it is missing, so the
//! migration also finishes an earlier 2.1 upgrade and leaves an up-to-date schema alone:
//!
//! - the MCP credentials table, and `rust_llm_tool_calls.mcp_state` (renaming the `pending_input`
//!   column an earlier 2.1 upgrade added) and `mcp_result`;
//! - `rust_llm_usages.server_tool_use`, a usage ledger free of chats (`chat_type`/`chat_id`
//!   nullable) with a polymorphic owner, and an operation constraint that accepts every operation;
//! - `messages.cache_ttl`, and the `rust_llm_provider_files` table, with `blob_key` on
//!   `rust_llm_attachments` for its rows to name.
//!
//! SQLite cannot change a column's nullability or a check constraint in place, so there the usage
//! ledger is rebuilt as the install creates it and its rows copied over. Like the Rails
//! migration's reversed `change` (whose guards see everything present), `down` leaves it all.

use sea_orm_migration::prelude::*;
use sea_orm_migration::schema::*;
use sea_orm_migration::sea_orm::{ConnectionTrait, DbBackend, Statement};

/// The message table this app maps `acts_as_message` to.
const MESSAGES: &str = "messages";

#[derive(DeriveMigrationName)]
pub struct Migration;

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, m: &SchemaManager) -> Result<(), DbErr> {
        if !m.has_table("rust_llm_mcp_credentials").await? {
            rust_llm_loco::migrations::CreateRustLlmMcpCredentials
                .up(m)
                .await?;
        }
        if !m.has_column("rust_llm_tool_calls", "mcp_state").await? {
            let table = Table::alter().table("rust_llm_tool_calls").to_owned();
            if m.has_column("rust_llm_tool_calls", "pending_input").await? {
                m.alter_table(
                    table
                        .clone()
                        .rename_column("pending_input", "mcp_state")
                        .to_owned(),
                )
                .await?;
            } else {
                m.alter_table(table.clone().add_column(json_null("mcp_state")).to_owned())
                    .await?;
            }
        }
        add_missing(m, "rust_llm_tool_calls", "mcp_result", json_null("mcp_result")).await?;
        upgrade_usages(m).await?;
        add_missing(m, MESSAGES, "cache_ttl", string_null("cache_ttl")).await?;
        add_missing(m, "rust_llm_attachments", "blob_key", string_null("blob_key")).await?;
        if !m.has_table("rust_llm_provider_files").await? {
            rust_llm_loco::migrations::CreateRustLlmProviderFiles
                .up(m)
                .await?;
        }
        Ok(())
    }

    async fn down(&self, _m: &SchemaManager) -> Result<(), DbErr> {
        Ok(())
    }
}

async fn add_missing(
    m: &SchemaManager<'_>,
    table: &str,
    name: &str,
    column: ColumnDef,
) -> Result<(), DbErr> {
    if !m.has_column(table, name).await? {
        let mut column = column;
        m.alter_table(
            Table::alter()
                .table(Alias::new(table))
                .add_column(&mut column)
                .to_owned(),
        )
        .await?;
    }
    Ok(())
}

/// The `rust_llm_usages` part of the upgrade: `server_tool_use`, nullable `chat_type`/`chat_id`,
/// the owner reference, and an operation constraint naming every operation.
async fn upgrade_usages(m: &SchemaManager<'_>) -> Result<(), DbErr> {
    let db = m.get_connection();
    if db.get_database_backend() == DbBackend::Sqlite {
        let sql = db
            .query_one_raw(Statement::from_string(
                DbBackend::Sqlite,
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'rust_llm_usages'",
            ))
            .await?
            .map(|row| row.try_get::<String>("", "sql"))
            .transpose()?
            .unwrap_or_default();
        let current = m.has_column("rust_llm_usages", "server_tool_use").await?
            && m.has_column("rust_llm_usages", "owner_id").await?
            && !chat_id_required(db).await?
            && operations_complete(&sql);
        if !current {
            rebuild_sqlite_usages(m).await?;
        }
        return Ok(());
    }

    add_missing(m, "rust_llm_usages", "server_tool_use", json_null("server_tool_use")).await?;
    if chat_id_required(db).await? {
        for column in ["chat_type", "chat_id"] {
            db.execute_unprepared(&format!(
                "ALTER TABLE rust_llm_usages ALTER COLUMN {column} DROP NOT NULL"
            ))
            .await?;
        }
    }
    if !m.has_column("rust_llm_usages", "owner_id").await? {
        m.alter_table(
            Table::alter()
                .table("rust_llm_usages")
                .add_column(string_null("owner_type"))
                .add_column(big_integer_null("owner_id"))
                .to_owned(),
        )
        .await?;
        m.create_index(
            Index::create()
                .name("idx-rust_llm_usages-owner")
                .table("rust_llm_usages")
                .col("owner_type")
                .col("owner_id")
                .to_owned(),
        )
        .await?;
    }
    // `check_constraints(:ruby_llm_usages).find { ... "operation" }`: replace one that leaves an
    // operation out.
    let rows = db
        .query_all_raw(Statement::from_string(
            DbBackend::Postgres,
            "SELECT conname, pg_get_constraintdef(oid) AS definition FROM pg_constraint \
             WHERE conrelid = 'rust_llm_usages'::regclass AND contype = 'c'",
        ))
        .await?;
    for row in rows {
        let name: String = row.try_get("", "conname")?;
        let definition: String = row.try_get("", "definition")?;
        if definition.contains("operation") && !operations_complete(&definition) {
            db.execute_unprepared(&format!(
                "ALTER TABLE rust_llm_usages DROP CONSTRAINT \"{name}\""
            ))
            .await?;
            db.execute_unprepared(&format!(
                "ALTER TABLE rust_llm_usages ADD CHECK ({})",
                rust_llm_loco::migrations::USAGE_OPERATIONS_CHECK
            ))
            .await?;
        }
    }
    Ok(())
}

/// Whether `chat_id` is still `NOT NULL`, as 2.0 created it.
async fn chat_id_required(db: &impl ConnectionTrait) -> Result<bool, DbErr> {
    let backend = db.get_database_backend();
    let sql = match backend {
        DbBackend::Sqlite => {
            "SELECT \"notnull\" = 1 AS required FROM pragma_table_info('rust_llm_usages') WHERE name = 'chat_id'"
        }
        _ => {
            "SELECT is_nullable = 'NO' AS required FROM information_schema.columns \
             WHERE table_name = 'rust_llm_usages' AND column_name = 'chat_id'"
        }
    };
    Ok(db
        .query_one_raw(Statement::from_string(backend, sql))
        .await?
        .map(|row| row.try_get::<bool>("", "required"))
        .transpose()?
        .unwrap_or(false))
}

/// Whether a constraint's text names every operation 2.1 records.
fn operations_complete(sql: &str) -> bool {
    [
        "chat",
        "embedding",
        "moderation",
        "image",
        "speech",
        "transcription",
        "ocr",
        "rerank",
        "judgment",
        "video",
        "research",
    ]
    .iter()
    .all(|operation| sql.contains(&format!("'{operation}'")))
}

/// SQLite: renames the ledger aside, creates it as the install does, copies the rows (columns the
/// old table lacks stay `NULL`), and drops the old one.
async fn rebuild_sqlite_usages(m: &SchemaManager<'_>) -> Result<(), DbErr> {
    let db = m.get_connection();
    let mut columns = Vec::new();
    for row in db
        .query_all_raw(Statement::from_string(
            DbBackend::Sqlite,
            "SELECT name FROM pragma_table_info('rust_llm_usages')",
        ))
        .await?
    {
        columns.push(row.try_get::<String>("", "name")?);
    }
    for index in [
        "idx-rust_llm_usages-chat",
        "idx-rust_llm_usages-message",
        "idx-rust_llm_usages-owner",
        "idx-rust_llm_usages-status",
    ] {
        db.execute_unprepared(&format!("DROP INDEX IF EXISTS \"{index}\""))
            .await?;
    }
    db.execute_unprepared("ALTER TABLE rust_llm_usages RENAME TO rust_llm_usages_2_0")
        .await?;
    rust_llm_loco::migrations::CreateRustLlmUsages.up(m).await?;
    let columns = columns
        .iter()
        .map(|c| format!("\"{c}\""))
        .collect::<Vec<_>>()
        .join(", ");
    db.execute_unprepared(&format!(
        "INSERT INTO rust_llm_usages ({columns}) SELECT {columns} FROM rust_llm_usages_2_0"
    ))
    .await?;
    db.execute_unprepared("DROP TABLE rust_llm_usages_2_0")
        .await?;
    Ok(())
}
