//! Port of `rails generate ruby_llm:install`'s migrations: the same tables and columns, as
//! SeaORM migrations. Add `ruby_llm_loco::migrations()` to your app's `Migrator`.

use sea_orm_migration::prelude::*;
use sea_orm_migration::schema::*;

/// All RubyLLM migrations, in order.
pub fn all() -> Vec<Box<dyn MigrationTrait>> {
    vec![Box::new(CreateRubyLlmRecords), Box::new(CreateChats), Box::new(CreateMessages)]
}

/// `create_ruby_llm_records_migration.rb.tt`: models, tool calls, usages.
#[derive(DeriveMigrationName)]
pub struct CreateRubyLlmRecords;

#[async_trait::async_trait]
impl MigrationTrait for CreateRubyLlmRecords {
    async fn up(&self, m: &SchemaManager) -> Result<(), DbErr> {
        m.create_table(
            Table::create()
                .table("ruby_llm_models")
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
            Index::create().name("idx-ruby_llm_models-provider-model_id").table("ruby_llm_models").col("provider").col("model_id").unique().to_owned(),
        )
        .await?;
        m.create_index(Index::create().name("idx-ruby_llm_models-family").table("ruby_llm_models").col("family").to_owned()).await?;

        // Polymorphic message/result references, like the Rails table.
        m.create_table(
            Table::create()
                .table("ruby_llm_tool_calls")
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
        m.create_index(Index::create().name("idx-ruby_llm_tool_calls-message").table("ruby_llm_tool_calls").col("message_type").col("message_id").to_owned()).await?;
        m.create_index(Index::create().name("idx-ruby_llm_tool_calls-result").table("ruby_llm_tool_calls").col("result_type").col("result_id").to_owned()).await?;
        m.create_index(Index::create().name("idx-ruby_llm_tool_calls-tool_call_id").table("ruby_llm_tool_calls").col("tool_call_id").unique().to_owned()).await?;
        m.create_index(Index::create().name("idx-ruby_llm_tool_calls-name").table("ruby_llm_tool_calls").col("name").to_owned()).await?;

        m.create_table(
            Table::create()
                .table("ruby_llm_usages")
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
        m.create_index(Index::create().name("idx-ruby_llm_usages-chat").table("ruby_llm_usages").col("chat_type").col("chat_id").to_owned()).await?;
        m.create_index(Index::create().name("idx-ruby_llm_usages-message").table("ruby_llm_usages").col("message_type").col("message_id").to_owned()).await?;
        m.create_index(Index::create().name("idx-ruby_llm_usages-status").table("ruby_llm_usages").col("status").to_owned()).await?;
        Ok(())
    }

    async fn down(&self, m: &SchemaManager) -> Result<(), DbErr> {
        for t in ["ruby_llm_usages", "ruby_llm_tool_calls", "ruby_llm_models"] {
            m.drop_table(Table::drop().table(t).if_exists().to_owned()).await?;
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
                .col(integer("ruby_llm_model_id"))
                .col(boolean("cancelled").default(false))
                .col(timestamp_with_time_zone("created_at").default(Expr::current_timestamp()))
                .col(timestamp_with_time_zone("updated_at").default(Expr::current_timestamp()))
                .foreign_key(
                    ForeignKey::create()
                        .name("fk-chats-ruby_llm_model_id")
                        .from("chats", "ruby_llm_model_id")
                        .to("ruby_llm_models", "id"),
                )
                .to_owned(),
        )
        .await?;
        m.create_index(Index::create().name("idx-chats-ruby_llm_model_id").table("chats").col("ruby_llm_model_id").to_owned()).await
    }

    async fn down(&self, m: &SchemaManager) -> Result<(), DbErr> {
        m.drop_table(Table::drop().table("chats").if_exists().to_owned()).await
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
        m.create_index(Index::create().name("idx-messages-chat_id").table("messages").col("chat_id").to_owned()).await
    }

    async fn down(&self, m: &SchemaManager) -> Result<(), DbErr> {
        m.drop_table(Table::drop().table("messages").if_exists().to_owned()).await
    }
}
