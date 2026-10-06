//! Chats belong to an account: `chats.account_id`, indexed (`rust-llm generate chat_ui`). The
//! column is the app's, not `rust_llm_loco`'s, the way a Rails app adds `belongs_to :account` to
//! its own `Chat` model; `src/models/chats.rs` scopes every chat UI query by it. SQLite cannot add
//! a foreign key to an existing table, so the column is a plain, nullable reference.

use sea_orm_migration::prelude::*;

#[derive(DeriveMigrationName)]
pub struct Migration;

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, m: &SchemaManager) -> Result<(), DbErr> {
        m.alter_table(
            Table::alter()
                .table(Alias::new("chats"))
                .add_column(ColumnDef::new(Alias::new("account_id")).big_integer().null())
                .to_owned(),
        )
        .await?;
        m.create_index(
            Index::create()
                .name("idx-chats-account_id")
                .table(Alias::new("chats"))
                .col(Alias::new("account_id"))
                .to_owned(),
        )
        .await
    }

    async fn down(&self, m: &SchemaManager) -> Result<(), DbErr> {
        m.drop_index(
            Index::drop()
                .name("idx-chats-account_id")
                .table(Alias::new("chats"))
                .to_owned(),
        )
        .await?;
        m.alter_table(
            Table::alter()
                .table(Alias::new("chats"))
                .drop_column(Alias::new("account_id"))
                .to_owned(),
        )
        .await
    }
}
