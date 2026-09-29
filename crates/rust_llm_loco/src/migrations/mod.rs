//! Port of `rails generate rust_llm:install`'s migrations: the same tables and columns, as
//! SeaORM migrations. Add `rust_llm_loco::migrations()` to your app's `Migrator`.

use sea_orm_migration::prelude::*;
use sea_orm_migration::schema::*;

/// All RubyLLM migrations, in order.
pub fn all() -> Vec<Box<dyn MigrationTrait>> {
    vec![
        Box::new(CreateRustLlmRecords),
        Box::new(CreateChats),
        Box::new(CreateMessages),
        Box::new(CreateRustLlmAttachments),
        Box::new(CreateRustLlmMcpCredentials),
    ]
}

/// `create_rust_llm_records_migration.rb.tt`: models, tool calls, usages.
#[derive(DeriveMigrationName)]
pub struct CreateRustLlmRecords;

#[async_trait::async_trait]
impl MigrationTrait for CreateRustLlmRecords {
    async fn up(&self, m: &SchemaManager) -> Result<(), DbErr> {
        m.create_table(
            Table::create()
                .table("rust_llm_models")
                .if_not_exists()
                .col(pk_auto("id"))
                .col(string("model_id"))
                .col(string("name"))
                .col(string("provider"))
                .col(string_null("family"))
                .col(timestamp_with_time_zone_null("model_created_at"))
                .col(integer_null("context_window"))
                .col(integer_null("max_output_tokens"))
                .col(date_null("knowledge_cutoff"))
                .col(timestamp_with_time_zone_null("unlisted_at"))
                .col(json_null("modalities"))
                .col(json_null("capabilities"))
                .col(json_null("pricing"))
                .col(json_null("metadata"))
                .col(timestamp_with_time_zone("created_at").default(Expr::current_timestamp()))
                .col(timestamp_with_time_zone("updated_at").default(Expr::current_timestamp()))
                .to_owned(),
        )
        .await?;
        m.create_index(
            Index::create()
                .name("idx-rust_llm_models-provider-model_id")
                .table("rust_llm_models")
                .col("provider")
                .col("model_id")
                .unique()
                .to_owned(),
        )
        .await?;
        m.create_index(
            Index::create()
                .name("idx-rust_llm_models-family")
                .table("rust_llm_models")
                .col("family")
                .to_owned(),
        )
        .await?;

        // Polymorphic message/result references, like the Rails table.
        m.create_table(
            Table::create()
                .table("rust_llm_tool_calls")
                .if_not_exists()
                .col(pk_auto("id"))
                .col(string("message_type"))
                .col(big_integer("message_id"))
                .col(string_null("result_type"))
                .col(big_integer_null("result_id"))
                .col(string("tool_call_id"))
                .col(string("name"))
                .col(text_null("thought_signature"))
                .col(string_null("approval"))
                .col(boolean("remote").default(false))
                .col(json_null("arguments"))
                .col(json_null("pending_input"))
                .col(timestamp_with_time_zone("created_at").default(Expr::current_timestamp()))
                .col(timestamp_with_time_zone("updated_at").default(Expr::current_timestamp()))
                .to_owned(),
        )
        .await?;
        m.create_index(
            Index::create()
                .name("idx-rust_llm_tool_calls-message")
                .table("rust_llm_tool_calls")
                .col("message_type")
                .col("message_id")
                .to_owned(),
        )
        .await?;
        m.create_index(
            Index::create()
                .name("idx-rust_llm_tool_calls-result")
                .table("rust_llm_tool_calls")
                .col("result_type")
                .col("result_id")
                .to_owned(),
        )
        .await?;
        m.create_index(
            Index::create()
                .name("idx-rust_llm_tool_calls-tool_call_id")
                .table("rust_llm_tool_calls")
                .col("tool_call_id")
                .unique()
                .to_owned(),
        )
        .await?;
        m.create_index(
            Index::create()
                .name("idx-rust_llm_tool_calls-name")
                .table("rust_llm_tool_calls")
                .col("name")
                .to_owned(),
        )
        .await?;

        m.create_table(
            Table::create()
                .table("rust_llm_usages")
                .if_not_exists()
                .col(pk_auto("id"))
                .col(string("chat_type"))
                .col(big_integer("chat_id"))
                .col(string_null("message_type"))
                .col(big_integer_null("message_id"))
                .col(string("operation"))
                .col(string("provider"))
                .col(string("model"))
                .col(string("status"))
                .col(integer_null("input_tokens"))
                .col(integer_null("output_tokens"))
                .col(integer_null("cache_read_tokens"))
                .col(integer_null("cache_write_tokens"))
                .col(integer_null("thinking_tokens"))
                .col(double_null("input_cost"))
                .col(double_null("output_cost"))
                .col(double_null("cache_read_cost"))
                .col(double_null("cache_write_cost"))
                .col(double_null("thinking_cost"))
                .col(double_null("total_cost"))
                .col(timestamp_with_time_zone("created_at").default(Expr::current_timestamp()))
                .col(timestamp_with_time_zone("updated_at").default(Expr::current_timestamp()))
                .check(Expr::cust(
                    "operation IN ('chat','embedding','moderation','image','speech','transcription','ocr','rerank','judgment')",
                ))
                .check(Expr::cust("status IN ('pending','succeeded','failed','cancelled')"))
                .to_owned(),
        )
        .await?;
        m.create_index(
            Index::create()
                .name("idx-rust_llm_usages-chat")
                .table("rust_llm_usages")
                .col("chat_type")
                .col("chat_id")
                .to_owned(),
        )
        .await?;
        m.create_index(
            Index::create()
                .name("idx-rust_llm_usages-message")
                .table("rust_llm_usages")
                .col("message_type")
                .col("message_id")
                .to_owned(),
        )
        .await?;
        m.create_index(
            Index::create()
                .name("idx-rust_llm_usages-status")
                .table("rust_llm_usages")
                .col("status")
                .to_owned(),
        )
        .await?;
        Ok(())
    }

    async fn down(&self, m: &SchemaManager) -> Result<(), DbErr> {
        for t in ["rust_llm_usages", "rust_llm_tool_calls", "rust_llm_models"] {
            m.drop_table(Table::drop().table(t).if_exists().to_owned())
                .await?;
        }
        Ok(())
    }
}

/// `create_chats_migration.rb.tt`.
#[derive(DeriveMigrationName)]
pub struct CreateChats;

#[async_trait::async_trait]
impl MigrationTrait for CreateChats {
    async fn up(&self, m: &SchemaManager) -> Result<(), DbErr> {
        m.create_table(
            Table::create()
                .table("chats")
                .if_not_exists()
                .col(pk_auto("id"))
                .col(integer("rust_llm_model_id"))
                .col(boolean("cancelled").default(false))
                .col(timestamp_with_time_zone("created_at").default(Expr::current_timestamp()))
                .col(timestamp_with_time_zone("updated_at").default(Expr::current_timestamp()))
                .foreign_key(
                    ForeignKey::create()
                        .name("fk-chats-rust_llm_model_id")
                        .from("chats", "rust_llm_model_id")
                        .to("rust_llm_models", "id"),
                )
                .to_owned(),
        )
        .await?;
        m.create_index(
            Index::create()
                .name("idx-chats-rust_llm_model_id")
                .table("chats")
                .col("rust_llm_model_id")
                .to_owned(),
        )
        .await
    }

    async fn down(&self, m: &SchemaManager) -> Result<(), DbErr> {
        m.drop_table(Table::drop().table("chats").if_exists().to_owned())
            .await
    }
}

/// `create_messages_migration.rb.tt`.
#[derive(DeriveMigrationName)]
pub struct CreateMessages;

#[async_trait::async_trait]
impl MigrationTrait for CreateMessages {
    async fn up(&self, m: &SchemaManager) -> Result<(), DbErr> {
        m.create_table(
            Table::create()
                .table("messages")
                .if_not_exists()
                .col(pk_auto("id"))
                .col(integer("chat_id"))
                .col(string("role"))
                .col(text_null("content"))
                .col(boolean("cache_until_here").default(false))
                .col(text_null("thinking_text"))
                .col(text_null("thinking_signature"))
                .col(json_null("citations"))
                .col(json_null("server_tool_calls"))
                .col(json_null("raw_content"))
                .col(json_null("raw_reasoning"))
                .col(string_null("finish_reason"))
                .col(timestamp_with_time_zone("created_at").default(Expr::current_timestamp()))
                .col(timestamp_with_time_zone("updated_at").default(Expr::current_timestamp()))
                .foreign_key(
                    ForeignKey::create()
                        .name("fk-messages-chat_id")
                        .from("messages", "chat_id")
                        .to("chats", "id")
                        .on_delete(ForeignKeyAction::Cascade),
                )
                .to_owned(),
        )
        .await?;
        m.create_index(
            Index::create()
                .name("idx-messages-chat_id")
                .table("messages")
                .col("chat_id")
                .to_owned(),
        )
        .await
    }

    async fn down(&self, m: &SchemaManager) -> Result<(), DbErr> {
        m.drop_table(Table::drop().table("messages").if_exists().to_owned())
            .await
    }
}

/// Attachment storage for messages. RubyLLM keeps message files in Active Storage
/// (`has_many_attached :attachments`, installed by `active_storage:install`); Loco has no
/// equivalent, so the bytes live in this table instead, one row per file, with the Active Storage
/// blob's filename, content type, byte size, and `metadata: { resolution: }`. Rows reference their
/// message polymorphically, like `rust_llm_tool_calls`.
#[derive(DeriveMigrationName)]
pub struct CreateRustLlmAttachments;

#[async_trait::async_trait]
impl MigrationTrait for CreateRustLlmAttachments {
    async fn up(&self, m: &SchemaManager) -> Result<(), DbErr> {
        m.create_table(
            Table::create()
                .table("rust_llm_attachments")
                .if_not_exists()
                .col(pk_auto("id"))
                .col(string("message_type"))
                .col(big_integer("message_id"))
                .col(string("filename"))
                .col(string("content_type"))
                .col(big_integer("byte_size"))
                .col(json_null("metadata"))
                .col(blob("data"))
                .col(timestamp_with_time_zone("created_at").default(Expr::current_timestamp()))
                .to_owned(),
        )
        .await?;
        m.create_index(
            Index::create()
                .name("idx-rust_llm_attachments-message")
                .table("rust_llm_attachments")
                .col("message_type")
                .col("message_id")
                .to_owned(),
        )
        .await
    }

    async fn down(&self, m: &SchemaManager) -> Result<(), DbErr> {
        m.drop_table(
            Table::drop()
                .table("rust_llm_attachments")
                .if_exists()
                .to_owned(),
        )
        .await
    }
}

/// The `ruby_llm_mcp_credentials` table of `create_ruby_llm_records_migration.rb.tt` (and of the
/// 2.1 upgrade): MCP OAuth credentials, keyed by owner and server, with a polymorphic owner.
/// `data` holds the encrypted JSON (see `McpCredentialStore`).
#[derive(DeriveMigrationName)]
pub struct CreateRustLlmMcpCredentials;

#[async_trait::async_trait]
impl MigrationTrait for CreateRustLlmMcpCredentials {
    async fn up(&self, m: &SchemaManager) -> Result<(), DbErr> {
        m.create_table(
            Table::create()
                .table("rust_llm_mcp_credentials")
                .if_not_exists()
                .col(pk_auto("id"))
                .col(string_null("owner_type"))
                .col(big_integer_null("owner_id"))
                .col(string("key"))
                .col(text_null("data"))
                .col(timestamp_with_time_zone("created_at").default(Expr::current_timestamp()))
                .col(timestamp_with_time_zone("updated_at").default(Expr::current_timestamp()))
                .to_owned(),
        )
        .await?;
        m.create_index(
            Index::create()
                .name("idx-rust_llm_mcp_credentials-owner")
                .table("rust_llm_mcp_credentials")
                .col("owner_type")
                .col("owner_id")
                .to_owned(),
        )
        .await?;
        m.create_index(
            Index::create()
                .name("idx-rust_llm_mcp_credentials-key")
                .table("rust_llm_mcp_credentials")
                .col("key")
                .unique()
                .to_owned(),
        )
        .await
    }

    async fn down(&self, m: &SchemaManager) -> Result<(), DbErr> {
        m.drop_table(
            Table::drop()
                .table("rust_llm_mcp_credentials")
                .if_exists()
                .to_owned(),
        )
        .await
    }
}
