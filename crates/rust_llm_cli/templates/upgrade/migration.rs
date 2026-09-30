//! Upgrades a RustLLM 2.0 schema to 2.1, from `rust-llm generate upgrade` (RubyLLM's
//! `upgrade_ruby_llm_to_2_1.rb.tt`): adds the MCP credentials table and
//! `rust_llm_tool_calls.pending_input`, each only when it is missing. Like the Rails migration's
//! reversed `change` (whose guards see both present), `down` leaves them in place.

use sea_orm_migration::prelude::*;
use sea_orm_migration::schema::*;

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
        if !m.has_column("rust_llm_tool_calls", "pending_input").await? {
            m.alter_table(
                Table::alter()
                    .table("rust_llm_tool_calls")
                    .add_column(json_null("pending_input"))
                    .to_owned(),
            )
            .await?;
        }
        Ok(())
    }

    async fn down(&self, _m: &SchemaManager) -> Result<(), DbErr> {
        Ok(())
    }
}
