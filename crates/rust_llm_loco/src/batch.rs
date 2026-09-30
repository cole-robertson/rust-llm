//! Port of `lib/ruby_llm/active_record/batch.rb`: RustLLM's persistence for provider-side
//! batches, in `rust_llm_batches`. With a [`BatchStore`] set as `config.batch_store` (the Railtie
//! sets `RubyLLM::ActiveRecord::Batch` there), `rust_llm::batch(chats)` records the batch and its
//! chats when it is submitted, `Batch::find(id)` rebuilds it from the row in any process, with the
//! records' chats attached, and `refresh`/`cancel` write the new state back.
//!
//! ```no_run
//! # async fn run(db: sea_orm::DatabaseConnection) -> rust_llm_loco::Result<()> {
//! use std::sync::Arc;
//! use rust_llm_loco::{BatchStore, ChatRecord};
//!
//! rust_llm::configure(|c| c.batch_store = Some(Arc::new(BatchStore::new(db.clone()))));
//! let record = ChatRecord::create(&db, "claude-haiku-4-5", None).await?;
//! let mut chat = record.to_llm(&db).await?;
//! record.ask_later(&db, &mut chat, "What is 2 + 2?").await?;
//! let batch = rust_llm_loco::batch::submit(&db, vec![(record, chat)]).await?;
//! // later, in a job:
//! let mut batch = rust_llm::Batch::find(batch.id(), None).await?;
//! batch.refresh().await?;
//! rust_llm_loco::batch::collect(&db, &mut batch).await?; // answers land on the records
//! # Ok(()) }
//! ```
//!
//! Ruby's chats persist their answers through callbacks installed on the chat; a Rust `Chat` has
//! none, so collecting is [`collect`], which runs `batch.messages` and then writes each chat's new
//! messages through its record. Each record appends an answer once, however often it is collected.

use std::sync::Arc;

use async_trait::async_trait;
use rust_llm::batch::BatchAttributes;
use rust_llm::{Batch, BatchStatus, Chat, Config, Cost, Message, Tokens};
use sea_orm::{
    ActiveModelTrait, ActiveValue::Set, ColumnTrait, DatabaseConnection, EntityTrait, QueryFilter,
    QueryOrder,
};
use serde_json::Value;

use crate::entities::{chats, rust_llm_batches};
use crate::{CHAT_TYPE, ChatRecord, Error, Result, now};

/// `Batch::STATUSES`.
pub const STATUSES: [&str; 4] = ["pending", "succeeded", "failed", "cancelled"];

/// `RubyLLM::ActiveRecord::Batch` as `config.batch_store`: the record itself is the persistence
/// adapter, keyed by the provider's batch id.
#[derive(Debug, Clone)]
pub struct BatchStore {
    db: DatabaseConnection,
}

impl BatchStore {
    pub fn new(db: DatabaseConnection) -> BatchStore {
        BatchStore { db }
    }
}

fn status_name(status: BatchStatus) -> &'static str {
    match status {
        BatchStatus::Pending => "pending",
        BatchStatus::Succeeded => "succeeded",
        BatchStatus::Failed => "failed",
        BatchStatus::Cancelled => "cancelled",
    }
}

fn to_llm_error(e: Error) -> rust_llm::Error {
    match e {
        Error::Llm(e) => e,
        other => rust_llm::Error::Io(std::io::Error::other(other.to_string())),
    }
}

/// `find_record(id, provider:)`: the newest row for the provider's id.
pub async fn find_record(
    db: &DatabaseConnection,
    id: &str,
    provider: Option<&str>,
) -> Result<Option<rust_llm_batches::Model>> {
    let mut query =
        rust_llm_batches::Entity::find().filter(rust_llm_batches::Column::ProviderBatchId.eq(id));
    if let Some(provider) = provider {
        query = query.filter(rust_llm_batches::Column::Provider.eq(provider));
    }
    Ok(query
        .order_by_desc(rust_llm_batches::Column::Id)
        .one(db)
        .await?)
}

/// `Batch.persist(batch, chats)`: one row for a submitted chat batch, with its chats' ids in
/// submission order.
pub async fn persist(
    db: &DatabaseConnection,
    batch: &Batch,
    chat_ids: &[i32],
) -> Result<rust_llm_batches::Model> {
    Ok(rust_llm_batches::ActiveModel {
        provider_batch_id: Set(batch.id().to_string()),
        provider: Set(batch.provider().to_string()),
        status: Set(status_name(batch.status()).into()),
        raw_status: Set(batch.raw_status().map(str::to_string)),
        completed: Set(batch.is_complete()),
        request_counts: Set(batch.request_counts().cloned()),
        reported_cost: Set(batch.reported_cost().map(Cost::to_h)),
        batch_protocol: Set(batch.batch_protocol().map(str::to_string)),
        chat_type: Set(Some(CHAT_TYPE.into())),
        chat_ids: Set(Some(Value::from(chat_ids.to_vec()))),
        created_at: Set(now()),
        updated_at: Set(now()),
        ..Default::default()
    }
    .insert(db)
    .await?)
}

/// `Batch.sync(batch)` / `sync_from`: the batch's current state onto its row. A missing
/// reported cost keeps the stored one.
pub async fn sync(db: &DatabaseConnection, batch: &Batch) -> Result<()> {
    let Some(row) = find_record(db, batch.id(), Some(batch.provider())).await? else {
        return Ok(());
    };
    let mut row: rust_llm_batches::ActiveModel = row.into();
    row.status = Set(status_name(batch.status()).into());
    row.raw_status = Set(batch.raw_status().map(str::to_string));
    row.completed = Set(batch.is_complete());
    row.request_counts = Set(batch.request_counts().cloned());
    row.batch_protocol = Set(batch.batch_protocol().map(str::to_string));
    if let Some(cost) = batch.reported_cost() {
        row.reported_cost = Set(Some(cost.to_h()));
    }
    row.updated_at = Set(now());
    row.update(db).await?;
    Ok(())
}

/// The row's `chat_ids`, in submission order.
pub fn chat_ids(row: &rust_llm_batches::Model) -> Vec<i32> {
    row.chat_ids
        .as_ref()
        .and_then(Value::as_array)
        .map(|ids| {
            ids.iter()
                .filter_map(|id| id.as_i64().map(|id| id as i32))
                .collect()
        })
        .unwrap_or_default()
}

/// `chats`: the records in submission order, `None` for one deleted since.
pub async fn records(
    db: &DatabaseConnection,
    row: &rust_llm_batches::Model,
) -> Result<Vec<Option<ChatRecord>>> {
    let ids = chat_ids(row);
    let found = chats::Entity::find()
        .filter(chats::Column::Id.is_in(ids.clone()))
        .all(db)
        .await?;
    Ok(ids
        .iter()
        .map(|id| {
            found
                .iter()
                .find(|c| c.id == *id)
                .cloned()
                .map(ChatRecord::from_row)
        })
        .collect())
}

/// `to_llm`: the row as a `rust_llm::Batch` holding its records' chats (built with `config`), so
/// collecting appends each answer to the chat it belongs to. A deleted chat keeps its slot with
/// a placeholder chat nothing reads back, which keeps the other answers aligned.
pub async fn to_llm(
    db: &DatabaseConnection,
    row: &rust_llm_batches::Model,
    config: Arc<Config>,
) -> Result<Batch> {
    let reported_cost = row
        .reported_cost
        .as_ref()
        .map(|h| Cost::from_h(h, None::<&Tokens>));
    let attributes = BatchAttributes {
        id: row.provider_batch_id.clone(),
        raw_status: row.raw_status.clone(),
        completed: row.completed,
        request_counts: row.request_counts.clone(),
        request_count: None,
        reported_cost,
    };
    let batch = Batch::from_attributes(config.clone(), &row.provider, attributes)?
        .with_batch_protocol(row.batch_protocol.as_deref());
    let mut chats = Vec::new();
    for record in records(db, row).await? {
        chats.push(match record {
            Some(record) => record.to_llm_with(db, config.clone()).await?,
            None => placeholder_chat(config.clone(), &row.provider)?,
        });
    }
    Ok(batch.with_chats(chats))
}

/// The chat standing in for a record deleted before collection: its slot still receives an
/// answer, which goes nowhere.
fn placeholder_chat(config: Arc<Config>, provider: &str) -> Result<Chat> {
    let model = config.default_model.clone();
    Ok(Chat::with_config(
        config,
        Some(&model),
        Some(provider),
        true,
    )?)
}

#[async_trait]
impl rust_llm::batch::BatchStore for BatchStore {
    async fn fetch(
        &self,
        id: &str,
        provider: Option<&str>,
        config: Arc<Config>,
    ) -> rust_llm::Result<Option<Batch>> {
        let row = find_record(&self.db, id, provider)
            .await
            .map_err(to_llm_error)?;
        match row {
            Some(row) => Ok(Some(
                to_llm(&self.db, &row, config).await.map_err(to_llm_error)?,
            )),
            None => Ok(None),
        }
    }

    async fn sync(&self, batch: &Batch) -> rust_llm::Result<()> {
        sync(&self.db, batch).await.map_err(to_llm_error)
    }
}

/// `RubyLLM.batch(chats)` for records: submits their chats (each staged with `ask_later`) and,
/// like `store&.persist(batch, records)`, records the batch with the records' ids. The row is
/// written here rather than by the store's `persist`, which only sees the chats, not the records.
pub async fn submit(db: &DatabaseConnection, chats: Vec<(ChatRecord, Chat)>) -> Result<Batch> {
    let ids: Vec<i32> = chats.iter().map(|(record, _)| record.id()).collect();
    let batch =
        rust_llm::batch(chats.into_iter().map(|(_, chat)| chat).collect::<Vec<_>>()).await?;
    persist(db, &batch, &ids).await?;
    Ok(batch)
}

/// `batch.messages` on a batch built from a row: collects the answers and persists each new one
/// on its chat record with its usage, the way the record's persistence callbacks do in Ruby.
/// Records deleted since submission are skipped. Returns the answers in submission order.
pub async fn collect(db: &DatabaseConnection, batch: &mut Batch) -> Result<Vec<Option<Message>>> {
    let messages = batch.messages().await?;
    let Some(row) = find_record(db, batch.id(), Some(batch.provider())).await? else {
        return Ok(messages);
    };
    let records = records(db, &row).await?;
    if let Some(chats) = batch.chats_mut() {
        for (record, chat) in records.iter().zip(chats.iter_mut()) {
            if let Some(record) = record {
                record.persist_collected(db, chat).await?;
            }
        }
    }
    Ok(messages)
}
