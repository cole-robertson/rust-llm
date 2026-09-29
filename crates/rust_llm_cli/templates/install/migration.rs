//! RustLLM's tables (`rust_llm_models`, `rust_llm_tool_calls`, `rust_llm_usages`, `chats`,
//! `messages`), from `rust-llm generate install` (RubyLLM's `create_*_migration.rb.tt`). One
//! migration here runs `rust_llm_loco`'s in order, so they are recorded under this file's name.

use sea_orm_migration::prelude::*;

#[derive(DeriveMigrationName)]
pub struct Migration;

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, m: &SchemaManager) -> Result<(), DbErr> {
        for migration in rust_llm_loco::migrations() {
            migration.up(m).await?;
        }
        Ok(())
    }

    async fn down(&self, m: &SchemaManager) -> Result<(), DbErr> {
        for migration in rust_llm_loco::migrations().iter().rev() {
            migration.down(m).await?;
        }
        Ok(())
    }
}
