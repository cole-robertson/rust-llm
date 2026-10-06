//! Port of `lib/ruby_llm/active_record/model.rb`: the `rust_llm_models` table as the model
//! registry store. Chats point at a row through `chats.rust_llm_model_id`; [`listed`] returns the
//! rows the provider still serves, [`unlisted`] the ones kept only because a chat references them.
//!
//! Configure it with `config.model_registry_store = Some(Arc::new(ModelStore::new(db)))`; the
//! registry then loads from the table and [`refresh`] saves into it.
//!
//! The store API is async (SeaORM). [`ModelRegistryStore`] is sync, so [`ModelStore`] runs the
//! async functions with `tokio::task::block_in_place` + `Handle::block_on`. That needs a
//! multi-threaded tokio runtime: on a current-thread runtime `read`/`write` return an error
//! instead of blocking it. Outside any runtime they run on a temporary current-thread runtime.

use std::collections::{HashMap, HashSet};

use rust_llm::models::Models;
use rust_llm::models::registry::ModelRegistryStore;
use sea_orm::{
    ActiveModelTrait, ActiveValue::Set, ColumnTrait, ConnectionTrait, DatabaseConnection,
    EntityTrait, QueryFilter, QueryOrder, Select, SqlErr, TransactionSession, TransactionTrait,
};
use serde_json::Value;

use crate::entities::rust_llm_models;
use crate::{Result, now};

/// `Model.table_name`.
pub const TABLE_NAME: &str = "rust_llm_models";

/// `UNLISTED_WARNING_LIMIT`: how many unlisted names the warning spells out.
pub const UNLISTED_WARNING_LIMIT: usize = 5;

/// `Model.read`: every row as a `rust_llm::Model`, including unlisted ones. Empty when the table
/// is missing, or when reading fails (logged at debug).
pub async fn read(db: &DatabaseConnection) -> Vec<rust_llm::Model> {
    let result: Result<Vec<rust_llm::Model>> = async {
        if !sea_orm_migration::SchemaManager::new(db)
            .has_table(TABLE_NAME)
            .await?
        {
            return Ok(Vec::new());
        }
        rust_llm_models::Entity::find()
            .order_by_asc(rust_llm_models::Column::Id)
            .all(db)
            .await?
            .iter()
            .map(rust_llm_models::Model::to_llm)
            .collect()
    }
    .await;
    result.unwrap_or_else(|e| {
        tracing::debug!("Failed to load models from database: {e}, falling back to JSON");
        Vec::new()
    })
}

/// `Model.write(registry)`.
pub async fn write<C>(db: &C, registry: &Models) -> Result<()>
where
    C: ConnectionTrait + TransactionTrait,
{
    save_to_database(db, registry).await
}

/// `Model.description`.
pub fn description() -> String {
    format!("database:{TABLE_NAME}")
}

/// `Model.refresh`: `RubyLLM.models.refresh`.
pub async fn refresh() -> rust_llm::Result<std::sync::Arc<Models>> {
    rust_llm::models::refresh(false).await
}

/// `Model.save_to_database(registry)`: in one transaction, upserts every listed model by
/// (provider, model_id) with all its attributes and `unlisted_at` cleared, then deletes the rows
/// the registry no longer carries. A row a chat still references cannot be deleted; it is stamped
/// `unlisted_at` instead (keeping the first stamp), and one warning names those rows.
pub async fn save_to_database<C>(db: &C, registry: &Models) -> Result<()>
where
    C: ConnectionTrait + TransactionTrait,
{
    let txn = crate::begin_write(db).await?;
    let existing: HashMap<(String, String), i32> = rust_llm_models::Entity::find()
        .all(&txn)
        .await?
        .into_iter()
        .map(|row| ((row.provider, row.model_id), row.id))
        .collect();

    let mut kept = HashSet::new();
    let mut inserts: Vec<rust_llm_models::ActiveModel> = Vec::new();
    let mut insert_index: HashMap<(String, String), usize> = HashMap::new();
    for info in registry.all() {
        let key = (info.provider.clone(), info.id.clone());
        let mut record = from_llm(info);
        record.updated_at = Set(now());
        if let Some(&id) = existing.get(&key) {
            record.id = Set(id);
            record.update(&txn).await?;
        } else {
            record.created_at = Set(now());
            // `find_or_initialize_by` finds the row an earlier entry created: the later one wins.
            match insert_index.get(&key) {
                Some(&i) => inserts[i] = record,
                None => {
                    insert_index.insert(key.clone(), inserts.len());
                    inserts.push(record);
                }
            }
        }
        kept.insert(key);
    }
    // Batched to stay under SQLite's bound-parameter limit.
    for chunk in inserts.chunks(500) {
        rust_llm_models::Entity::insert_many(chunk.to_vec())
            .exec_without_returning(&txn)
            .await?;
    }

    unlist(&txn, &kept).await?;
    txn.commit().await?;
    Ok(())
}

/// `unlist(kept_ids)`: a refresh replaces the registry, so models it no longer carries go away. A
/// row an application record points at cannot: deleting it would dangle the reference. That row
/// is stamped unlisted instead, keeping the chats that use it resolvable.
async fn unlist<C>(txn: &C, kept: &HashSet<(String, String)>) -> Result<()>
where
    C: ConnectionTrait + TransactionTrait,
{
    let rows = rust_llm_models::Entity::find()
        .order_by_asc(rust_llm_models::Column::Id)
        .all(txn)
        .await?;
    let mut stayed = Vec::new();
    for row in rows {
        if kept.contains(&(row.provider.clone(), row.model_id.clone())) {
            continue;
        }
        let savepoint = txn.begin().await?;
        match rust_llm_models::Entity::delete_by_id(row.id)
            .exec(&savepoint)
            .await
        {
            Ok(_) => savepoint.commit().await?,
            Err(e) if matches!(e.sql_err(), Some(SqlErr::ForeignKeyConstraintViolation(_))) => {
                savepoint.rollback().await?;
                let name = format!("{}/{}", row.provider, row.model_id);
                if row.unlisted_at.is_none() {
                    let mut record: rust_llm_models::ActiveModel = row.into();
                    record.unlisted_at = Set(Some(now()));
                    record.update(txn).await?;
                }
                stayed.push(name);
            }
            Err(e) => return Err(e.into()),
        }
    }
    warn_unlisted(&stayed);
    Ok(())
}

fn warn_unlisted(names: &[String]) {
    if names.is_empty() {
        return;
    }
    let subject = if names.len() == 1 {
        "1 model is".to_string()
    } else {
        format!("{} models are", names.len())
    };
    tracing::warn!(
        "{subject} no longer listed by the provider and may no longer work: {}. \
         The rows stay because application records still reference them. The provider may have \
         dropped them, or your configured region may not offer them.",
        unlisted_listing(names)
    );
}

fn unlisted_listing(names: &[String]) -> String {
    let listing = names
        .iter()
        .take(UNLISTED_WARNING_LIMIT)
        .cloned()
        .collect::<Vec<_>>()
        .join(", ");
    match names.len().checked_sub(UNLISTED_WARNING_LIMIT) {
        Some(extra) if extra > 0 => format!("{listing}, and {extra} more"),
        _ => listing,
    }
}

/// `Model.from_llm(model_info)`: an unsaved row with the model's attributes (`attributes_from_llm`).
pub fn from_llm(info: &rust_llm::Model) -> rust_llm_models::ActiveModel {
    let json = |v: serde_json::Result<Value>| Some(v.unwrap_or_default());
    rust_llm_models::ActiveModel {
        model_id: Set(info.id.clone()),
        name: Set(info.name.clone()),
        provider: Set(info.provider.clone()),
        family: Set(info.family.clone()),
        model_created_at: Set(info.created_at_time().map(Into::into)),
        context_window: Set(info.context_window.map(|v| v as i32)),
        max_output_tokens: Set(info.max_output_tokens.map(|v| v as i32)),
        knowledge_cutoff: Set(info.knowledge_cutoff_date()),
        modalities: Set(json(serde_json::to_value(&info.modalities))),
        capabilities: Set(json(serde_json::to_value(&info.capabilities))),
        pricing: Set(json(serde_json::to_value(&info.pricing))),
        metadata: Set(Some(Value::Object(info.metadata.clone()))),
        unlisted_at: Set(None),
        ..Default::default()
    }
}

/// `Model.listed`: the rows the provider still lists.
pub fn listed() -> Select<rust_llm_models::Entity> {
    rust_llm_models::Entity::find().filter(rust_llm_models::Column::UnlistedAt.is_null())
}

/// `Model.unlisted`: the rows kept only because an application record references them.
pub fn unlisted() -> Select<rust_llm_models::Entity> {
    rust_llm_models::Entity::find().filter(rust_llm_models::Column::UnlistedAt.is_not_null())
}

impl rust_llm_models::Model {
    /// `Model#to_llm`: this row as a `rust_llm::Model`. Null JSON columns become empty
    /// modalities, capabilities, pricing, and metadata.
    pub fn to_llm(&self) -> Result<rust_llm::Model> {
        fn column<T: serde::de::DeserializeOwned + Default>(v: &Option<Value>) -> Result<T> {
            match v {
                None | Some(Value::Null) => Ok(T::default()),
                Some(v) => Ok(serde_json::from_value(v.clone()).map_err(rust_llm::Error::from)?),
            }
        }
        Ok(rust_llm::model::ModelData {
            id: self.model_id.clone(),
            name: self.name.clone(),
            provider: self.provider.clone(),
            family: self.family.clone(),
            created_at: self.model_created_at.map(|t| t.to_rfc3339()),
            context_window: self.context_window.map(i64::from),
            max_output_tokens: self.max_output_tokens.map(i64::from),
            knowledge_cutoff: self
                .knowledge_cutoff
                .map(|d| d.format("%Y-%m-%d").to_string()),
            modalities: column(&self.modalities)?,
            capabilities: column(&self.capabilities)?,
            pricing: column(&self.pricing)?,
            metadata: column(&self.metadata)?,
            unlisted_at: self.unlisted_at.map(|t| t.to_rfc3339()),
            reasoning_options: None,
        }
        .into())
    }

    /// `supports?`, answered from [`to_llm`](Self::to_llm).
    pub fn supports(&self, capability: &str) -> Result<bool> {
        Ok(self.to_llm()?.supports(capability))
    }

    /// `unlisted?`.
    pub fn is_unlisted(&self) -> bool {
        self.unlisted_at.is_some()
    }
}

/// `config.model_registry_store = RubyLLM::ActiveRecord::Model`: the table as a sync
/// [`ModelRegistryStore`]. See the module docs for the runtime requirement.
#[derive(Debug, Clone)]
pub struct ModelStore {
    db: DatabaseConnection,
}

impl ModelStore {
    pub fn new(db: DatabaseConnection) -> ModelStore {
        ModelStore { db }
    }
}

/// Runs `f` to completion from sync code (see the module docs).
fn block_on<F: std::future::Future>(f: F) -> rust_llm::Result<F::Output> {
    use tokio::runtime::{Builder, Handle, RuntimeFlavor};
    match Handle::try_current() {
        Ok(handle) if handle.runtime_flavor() == RuntimeFlavor::CurrentThread => {
            Err(rust_llm::Error::ModelRegistry(format!(
                "The {} store needs a multi-threaded tokio runtime",
                description()
            )))
        }
        Ok(handle) => Ok(tokio::task::block_in_place(|| handle.block_on(f))),
        Err(_) => Ok(Builder::new_current_thread()
            .enable_all()
            .build()?
            .block_on(f)),
    }
}

impl ModelRegistryStore for ModelStore {
    fn read(&self) -> rust_llm::Result<Vec<rust_llm::Model>> {
        block_on(read(&self.db))
    }

    fn write(&self, models: &Models) -> rust_llm::Result<()> {
        block_on(save_to_database(&self.db, models))?.map_err(|e| match e {
            crate::Error::Llm(e) => e,
            other => rust_llm::Error::Io(std::io::Error::other(other.to_string())),
        })
    }

    fn description(&self) -> String {
        description()
    }
}
