//! Port of `RubyLLM::ActiveRecord::Usage.record` (`lib/ruby_llm/active_record/usage.rb`): the
//! usage ledger for attempts no chat record writes, such as one-shot operations (`embed`, `paint`,
//! `transcribe`, `judge`, ...), finished video jobs, and chats without a record.
//!
//! ```no_run
//! # async fn run(db: sea_orm::DatabaseConnection) {
//! rust_llm::configure(|c| c.usage_ledger = Some(rust_llm_loco::UsageLedger::shared(db)));
//! # }
//! ```

use std::sync::Arc;

use rust_llm::UsageEntry;
use sea_orm::{ActiveModelTrait, DatabaseConnection, TransactionTrait};

/// `rust_llm_usages` as `Accounting::Usage.ledger` (Ruby's railtie sets
/// `RubyLLM::ActiveRecord::Usage` there). Rows belong to no chat and keep their owner when it is a
/// record ([`rust_llm::accounting::UsageOwner::Record`]).
pub struct UsageLedger {
    db: DatabaseConnection,
}

impl UsageLedger {
    pub fn new(db: DatabaseConnection) -> UsageLedger {
        UsageLedger { db }
    }

    /// The ledger, ready for `Config::usage_ledger`.
    pub fn shared(db: DatabaseConnection) -> Arc<dyn rust_llm::accounting::UsageLedger> {
        Arc::new(UsageLedger::new(db))
    }

    /// `Usage.record(entry)`: writes the row, or logs why it could not. An attempt without a model
    /// is skipped, since the `model` column is required. The write runs in its own transaction
    /// on a pooled connection, so a failure never touches a caller's transaction, and the
    /// connection goes back to the pool when it finishes.
    pub async fn record(&self, entry: &UsageEntry) -> crate::Result<()> {
        if entry.model.is_empty() {
            return Ok(());
        }
        let txn = self.db.begin().await?;
        crate::usage_attributes(entry).insert(&txn).await?;
        txn.commit().await?;
        Ok(())
    }
}

#[async_trait::async_trait]
impl rust_llm::accounting::UsageLedger for UsageLedger {
    async fn record(&self, entry: &UsageEntry) {
        if let Err(e) = UsageLedger::record(self, entry).await {
            tracing::warn!(
                "RustLLM could not record {} usage: {e}",
                entry.operation.as_str()
            );
        }
    }
}
