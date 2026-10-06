//! # rust_llm_loco
//!
//! RubyLLM's Rails integration (`acts_as_chat`, `acts_as_message`, `acts_as_tool_call`, the
//! `rust_llm_models`/`rust_llm_usages` ledgers) for Loco's default ORM, SeaORM.
//!
//! ```ruby
//! class Chat < ApplicationRecord
//!   acts_as_chat
//! end
//! chat = Chat.create!(model: "claude-haiku-4-5")
//! chat.with_tools(Weather).ask("What's the weather in Berlin?")
//! chat.messages.count # => user, assistant tool call, tool result, assistant
//! ```
//!
//! ```no_run
//! # use rust_llm::{Tool, ToolCall, ToolError, ToolResult};
//! # struct Weather;
//! # #[async_trait::async_trait]
//! # impl Tool for Weather {
//! #     fn description(&self) -> String { "Gets the weather".into() }
//! #     async fn execute(&self, _: serde_json::Map<String, serde_json::Value>, _: &ToolCall) -> Result<ToolResult, ToolError> { Ok("Sunny".into()) }
//! # }
//! use rust_llm_loco::ChatRecord;
//!
//! # async fn run(db: &sea_orm::DatabaseConnection) -> rust_llm_loco::Result<()> {
//! let record = ChatRecord::create(db, "claude-haiku-4-5", None).await?;
//! let mut chat = record.to_llm(db).await?.with_tool(Weather);
//! record.ask(db, &mut chat, "What's the weather in Berlin?").await?;
//! let rows = record.messages(db).await?; // user, assistant tool call, tool result, assistant
//! # Ok(()) }
//! ```
//!
//! Like RubyLLM, every message, tool call, and billed attempt is written as it happens, so a
//! chat can be reloaded mid-round (parked on a tool approval or an MCP input request) and
//! continued later, and another process can cancel it through the `chats.cancelled` column. Add
//! [`migrations()`] to your Loco migrator for the tables.
//!
//! Message attachments live in `rust_llm_attachments` (bytes plus the filename, content type, and
//! resolution RubyLLM keeps on the Active Storage blob), since Loco has no Active Storage. See the
//! [persistence guide](https://github.com/cole-robertson/rust-llm/blob/main/docs/persistence-loco.md).

pub mod batch;
pub mod entities;
mod mcp_credential;
pub mod migrations;
pub mod model_store;

pub use batch::BatchStore;
pub use mcp_credential::McpCredentialStore;
pub use model_store::ModelStore;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use rust_llm::attachment::Resolution;
use rust_llm::message::indexmap_lite::IndexMap;
use rust_llm::{
    Agent, Attachment, Chat, Citation, FinishReason, Message, Role, Thinking, ToolCall, UsageEntry,
    UsageStatus,
};
use sea_orm::{
    ActiveModelTrait, ActiveValue::Set, ColumnTrait, ConnectionTrait, DatabaseConnection,
    EntityTrait, QueryFilter, QueryOrder, TransactionTrait,
};
use serde_json::{Map, Value};

/// Usage rows written as attempts finished: `(UsageEntry.id, rust_llm_usages.id)`, not yet linked.
type UsageRows = Arc<Mutex<Vec<(u64, i32)>>>;

/// Where a message's usage comes from when it is persisted.
enum Usages<'a> {
    /// Rows already written as the attempts finished; persisting only links them.
    Written(&'a UsageRows),
    /// Entries not written yet (batch results, out-of-band completions); persisting inserts them.
    Pending(&'a Mutex<Vec<UsageEntry>>),
}

use entities::{
    chats, messages, rust_llm_attachments, rust_llm_models, rust_llm_tool_calls, rust_llm_usages,
};

/// The polymorphic type names written into `message_type`/`chat_type`, like Rails' class names.
pub const CHAT_TYPE: &str = "Chat";
pub const MESSAGE_TYPE: &str = "Message";

/// `ChatMethods::CANCELLATION_POLL_INTERVAL`: how often a running `complete` reads the
/// `chats.cancelled` column another process may have set.
pub const CANCELLATION_POLL_INTERVAL: Duration = Duration::from_secs(1);

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Db(#[from] sea_orm::DbErr),
    #[error(transparent)]
    Llm(#[from] rust_llm::Error),
    #[error("{0}")]
    NotFound(String),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// The Rust migrations to append to an app's `Migrator::migrations()`.
pub fn migrations() -> Vec<Box<dyn sea_orm_migration::MigrationTrait>> {
    migrations::all()
}

pub(crate) fn now() -> sea_orm::prelude::DateTimeWithTimeZone {
    chrono::Utc::now().into()
}

/// The registry's `created_at` (`"2025-04-14 00:00:00 UTC"` or RFC 3339) as a timestamp.
fn parse_time(value: &str) -> Option<sea_orm::prelude::DateTimeWithTimeZone> {
    chrono::DateTime::parse_from_rfc3339(value)
        .ok()
        .or_else(|| {
            chrono::NaiveDateTime::parse_from_str(
                value.trim_end_matches(" UTC"),
                "%Y-%m-%d %H:%M:%S",
            )
            .ok()
            .map(|t| t.and_utc().into())
        })
}

/// `ChatMethods#find_or_create_model`: `Model.find_or_create_by!(provider:, model_id:)` with the
/// registry's metadata, reusing the row another process inserted between the lookup and the insert.
///
/// Like RubyLLM (`load_model_registry_into_store`), an empty `rust_llm_models` table is first filled
/// with the whole registry, so a [`model_store::ModelStore`] can load it on the next boot. A
/// unique violation from a concurrent fill is ignored, as Ruby rescues `RecordNotUnique`.
pub async fn find_or_create_model(
    db: &(impl ConnectionTrait + TransactionTrait),
    model: &rust_llm::Model,
) -> Result<rust_llm_models::Model> {
    if rust_llm_models::Entity::find().one(db).await?.is_none() {
        match model_store::save_to_database(db, &rust_llm::models()).await {
            Err(Error::Db(e))
                if matches!(
                    e.sql_err(),
                    Some(sea_orm::SqlErr::UniqueConstraintViolation(_))
                ) => {}
            other => other?,
        }
    }
    if let Some(found) = find_model_row(db, model).await? {
        return Ok(found);
    }
    insert_model(db, model).await
}

async fn find_model_row(
    db: &impl ConnectionTrait,
    model: &rust_llm::Model,
) -> Result<Option<rust_llm_models::Model>> {
    Ok(rust_llm_models::Entity::find()
        .filter(rust_llm_models::Column::Provider.eq(&model.provider))
        .filter(rust_llm_models::Column::ModelId.eq(&model.id))
        .one(db)
        .await?)
}

/// The insert half of [`find_or_create_model`], after its lookup missed: inserts the row, or
/// returns the one another process inserted in between (the unique index rejects ours).
pub async fn insert_model(
    db: &impl ConnectionTrait,
    model: &rust_llm::Model,
) -> Result<rust_llm_models::Model> {
    let record = rust_llm_models::ActiveModel {
        model_id: Set(model.id.clone()),
        name: Set(if model.name.is_empty() {
            model.id.clone()
        } else {
            model.name.clone()
        }),
        provider: Set(model.provider.clone()),
        family: Set(model.family.clone()),
        model_created_at: Set(model.created_at.as_deref().and_then(parse_time)),
        context_window: Set(model.context_window.map(|v| v as i32)),
        max_output_tokens: Set(model.max_output_tokens.map(|v| v as i32)),
        knowledge_cutoff: Set(model
            .knowledge_cutoff
            .as_deref()
            .and_then(|d| chrono::NaiveDate::parse_from_str(d, "%Y-%m-%d").ok())),
        modalities: Set(Some(
            serde_json::to_value(&model.modalities).unwrap_or_default(),
        )),
        capabilities: Set(Some(
            serde_json::to_value(&model.capabilities).unwrap_or_default(),
        )),
        pricing: Set(Some(
            serde_json::to_value(&model.pricing).unwrap_or_default(),
        )),
        metadata: Set(Some(Value::Object(model.metadata.clone()))),
        created_at: Set(now()),
        updated_at: Set(now()),
        ..Default::default()
    };
    match record.insert(db).await {
        Ok(row) => Ok(row),
        Err(e) => find_model_row(db, model).await?.ok_or(Error::Db(e)),
    }
}

/// `ChatMethods#resolve_model_info`.
fn resolve_model_info(
    model: &str,
    provider: Option<&str>,
    assume_model_exists: bool,
) -> Result<rust_llm::Model> {
    let assume = assume_model_exists
        || provider
            .and_then(rust_llm::Provider::resolve)
            .is_some_and(|p| p.assume_models_exist());
    if !assume {
        return Ok(rust_llm::models().find(model, provider)?);
    }
    let provider = provider.ok_or_else(|| {
        rust_llm::Error::Argument(
            "Provider must be specified if assume_model_exists is true".into(),
        )
    })?;
    Ok(rust_llm::models()
        .find(model, Some(provider))
        .unwrap_or_else(|_| rust_llm::Model::default_for(model, provider)))
}

/// An instruction applied with `persist: false`: `(text, append, cache_until_here)`.
type RuntimeInstruction = (String, bool, bool);

/// What [`ChatRecord::complete_stream`] reports while it runs: RubyLLM's `on_new_message`, each
/// streamed chunk, and `on_end_message`, each with the `messages` row it belongs to.
#[derive(Debug)]
pub enum StreamEvent<'a> {
    /// A message row exists. For a response this is the empty assistant row created before its
    /// first chunk (`persist_new_message`); for a tool result, the row just written.
    NewMessage(&'a messages::Model),
    /// A chunk of the response being written into row `message_id`.
    Chunk {
        message_id: i32,
        chunk: &'a Message,
    },
    /// The row is final: content, tool calls, and usage are written (`persist_message_completion`).
    EndMessage(&'a messages::Model),
}

/// A persisted chat: the `Chat` model that `acts_as_chat`.
#[derive(Debug, Clone)]
pub struct ChatRecord {
    pub record: chats::Model,
    /// `assume_model_exists`: accept a model id the registry does not know. Not persisted; set
    /// it again after loading the record, as in RubyLLM.
    pub assume_model_exists: bool,
    /// `protocol`: the wire protocol the chat is built with. Not persisted; set it again after
    /// loading the record, as in RubyLLM.
    pub protocol: Option<rust_llm::providers::ProtocolName>,
    /// `context`: the [`rust_llm::Context`] whose configuration `to_llm` builds the chat with.
    /// Runtime-only, like RubyLLM's.
    context: Option<rust_llm::Context>,
    /// `@unpersisted_instructions`, reapplied whenever the chat is rebuilt from rows.
    runtime_instructions: Vec<RuntimeInstruction>,
    /// `@last_cancellation_poll_at`.
    last_cancellation_poll: Arc<Mutex<Option<std::time::Instant>>>,
}

impl ChatRecord {
    pub(crate) fn from_row(record: chats::Model) -> ChatRecord {
        ChatRecord {
            record,
            assume_model_exists: false,
            protocol: None,
            context: None,
            runtime_instructions: Vec::new(),
            last_cancellation_poll: Arc::default(),
        }
    }

    /// `context`.
    pub fn context(&self) -> Option<&rust_llm::Context> {
        self.context.as_ref()
    }

    /// `chat.with_context(context)`: later `to_llm` calls build with the context's
    /// configuration, and `chat` (the chat already built, if any) is rebound to it. `None` goes
    /// back to the global configuration. Not persisted.
    pub fn with_context(
        &mut self,
        context: Option<rust_llm::Context>,
        chat: Option<Chat>,
    ) -> Result<Option<Chat>> {
        self.context = context;
        Ok(match chat {
            Some(chat) => Some(chat.with_context(self.context.as_ref())?),
            None => None,
        })
    }

    /// `Chat.create!(model:, provider:)`.
    pub async fn create(
        db: &DatabaseConnection,
        model: &str,
        provider: Option<&str>,
    ) -> Result<ChatRecord> {
        Self::create_with(db, model, provider, false).await
    }

    /// `Chat.create!(model:, provider:, assume_model_exists: true)`: a model id the registry does
    /// not know is stored with default metadata. Requires a provider.
    pub async fn create_with(
        db: &DatabaseConnection,
        model: &str,
        provider: Option<&str>,
        assume_model_exists: bool,
    ) -> Result<ChatRecord> {
        let info = resolve_model_info(model, provider, assume_model_exists)?;
        let model_row = find_or_create_model(db, &info).await?;
        let record = chats::ActiveModel {
            rust_llm_model_id: Set(model_row.id),
            cancelled: Set(false),
            created_at: Set(now()),
            updated_at: Set(now()),
            ..Default::default()
        }
        .insert(db)
        .await?;
        let mut chat = Self::from_row(record);
        chat.assume_model_exists = assume_model_exists;
        Ok(chat)
    }

    pub async fn find(db: &DatabaseConnection, id: i32) -> Result<ChatRecord> {
        let record = chats::Entity::find_by_id(id)
            .one(db)
            .await?
            .ok_or_else(|| Error::NotFound(format!("chat {id}")))?;
        Ok(Self::from_row(record))
    }

    pub fn id(&self) -> i32 {
        self.record.id
    }

    /// `reload`: rereads the row and refreshes `chat`'s history from the database, keeping its
    /// runtime configuration (tools, callbacks, runtime instructions).
    pub async fn reload(&mut self, db: &DatabaseConnection, chat: &mut Chat) -> Result<()> {
        self.record = chats::Entity::find_by_id(self.record.id)
            .one(db)
            .await?
            .ok_or_else(|| Error::NotFound(format!("chat {}", self.record.id)))?;
        self.sync_messages(db, chat).await
    }

    /// The chat's model row (`chat.model`).
    pub async fn model(&self, db: &impl ConnectionTrait) -> Result<rust_llm_models::Model> {
        rust_llm_models::Entity::find_by_id(self.record.rust_llm_model_id)
            .one(db)
            .await?
            .ok_or_else(|| Error::NotFound("rust_llm_model".into()))
    }

    /// `chat.with_model(model, provider:)`: stores the new model row on the chat and returns
    /// `chat` switched to it.
    pub async fn with_model(
        &mut self,
        db: &DatabaseConnection,
        chat: Chat,
        model: &str,
        provider: Option<&str>,
    ) -> Result<Chat> {
        let info = resolve_model_info(model, provider, self.assume_model_exists)?;
        let row = find_or_create_model(db, &info).await?;
        let mut record: chats::ActiveModel = self.record.clone().into();
        record.rust_llm_model_id = Set(row.id);
        record.updated_at = Set(now());
        self.record = record.update(db).await?;
        Ok(if self.assume_model_exists {
            chat.with_assumed_model(&row.model_id, &row.provider)?
        } else {
            chat.with_model(&row.model_id, Some(&row.provider))?
        })
    }

    /// `chat.with_model(nil)`: falls back to the configured `default_model` (the context's, when
    /// the record has one).
    pub async fn with_default_model(
        &mut self,
        db: &DatabaseConnection,
        chat: Chat,
    ) -> Result<Chat> {
        let default = chat.config().default_model.clone();
        self.with_model(db, chat, &default, None).await
    }

    /// `chat.messages`, oldest first.
    pub async fn messages(&self, db: &impl ConnectionTrait) -> Result<Vec<messages::Model>> {
        Ok(messages::Entity::find()
            .filter(messages::Column::ChatId.eq(self.record.id))
            .order_by_asc(messages::Column::Id)
            .all(db)
            .await?)
    }

    /// `chat.rust_llm_usages`.
    pub async fn usages(&self, db: &impl ConnectionTrait) -> Result<Vec<rust_llm_usages::Model>> {
        Ok(rust_llm_usages::Entity::find()
            .filter(rust_llm_usages::Column::ChatType.eq(CHAT_TYPE))
            .filter(rust_llm_usages::Column::ChatId.eq(self.record.id as i64))
            .order_by_asc(rust_llm_usages::Column::Id)
            .all(db)
            .await?)
    }

    /// This chat's tool-call rows, oldest first.
    async fn tool_calls(
        &self,
        db: &impl ConnectionTrait,
    ) -> Result<Vec<rust_llm_tool_calls::Model>> {
        let ids: Vec<i64> = self
            .messages(db)
            .await?
            .iter()
            .map(|m| m.id as i64)
            .collect();
        Ok(rust_llm_tool_calls::Entity::find()
            .filter(rust_llm_tool_calls::Column::MessageType.eq(MESSAGE_TYPE))
            .filter(rust_llm_tool_calls::Column::MessageId.is_in(ids))
            .order_by_asc(rust_llm_tool_calls::Column::Id)
            .all(db)
            .await?)
    }

    /// `chat.to_llm`: an in-memory `rust_llm::Chat` rebuilt from rows, including approval
    /// decisions, paused MCP input requests, and attachments.
    pub async fn to_llm(&self, db: &DatabaseConnection) -> Result<Chat> {
        let config = self
            .context
            .as_ref()
            .map(|c| c.config().clone())
            .unwrap_or_else(rust_llm::config);
        self.to_llm_with(db, config).await
    }

    /// `to_llm` with an explicit configuration (RubyLLM's `context:`).
    pub async fn to_llm_with(
        &self,
        db: &DatabaseConnection,
        config: Arc<rust_llm::Config>,
    ) -> Result<Chat> {
        let model = self.model(db).await?;
        let provider = rust_llm::Provider::resolve(&model.provider);
        let assume = self.assume_model_exists || provider.is_some_and(|p| p.assume_models_exist());
        let mut chat =
            Chat::with_config(config, Some(&model.model_id), Some(&model.provider), assume)?;
        if let Some(protocol) = self.protocol {
            chat = chat.with_protocol(protocol);
        }
        self.sync_messages(db, &mut chat).await?;
        Ok(chat)
    }

    /// `sync_messages`: replaces `chat`'s history with the rows, then reapplies the runtime
    /// instructions.
    async fn sync_messages(&self, db: &DatabaseConnection, chat: &mut Chat) -> Result<()> {
        let rows = self.messages(db).await?;
        let ids: Vec<i64> = rows.iter().map(|m| m.id as i64).collect();
        let calls = self.tool_calls(db).await?;
        let usages = self.usages(db).await?;
        let files = rust_llm_attachments::Entity::find()
            .filter(rust_llm_attachments::Column::MessageType.eq(MESSAGE_TYPE))
            .filter(rust_llm_attachments::Column::MessageId.is_in(ids))
            .order_by_asc(rust_llm_attachments::Column::Id)
            .all(db)
            .await?;
        // One entry per row, shared by the chat's ledger and the message it belongs to.
        let entries: Vec<(i32, Option<i64>, UsageEntry)> = usages
            .iter()
            .map(|u| (u.id, u.message_id, usage_entry(u)))
            .collect();

        let restored = rows
            .iter()
            .map(|row| restore_message(row, &calls, &files, &entries))
            .collect::<Result<Vec<_>>>()?;
        chat.set_messages(restored);
        chat.set_usage_entries(entries.into_iter().map(|(_, _, e)| e).collect());
        apply_tool_call_state(chat, &calls);
        for (text, append, cache_until_here) in &self.runtime_instructions {
            chat.set_instructions(Some(text.clone()), *append, *cache_until_here);
        }
        Ok(())
    }

    /// `approval_checker` / `input_checker`: decisions and paused inputs are reread from the rows
    /// before each move, so another process's approval or answer is seen by a running chat.
    async fn refresh_tool_call_state(
        &self,
        db: &DatabaseConnection,
        chat: &mut Chat,
    ) -> Result<()> {
        apply_tool_call_state(chat, &self.tool_calls(db).await?);
        Ok(())
    }

    /// `chat.with_instructions(text)`: persisted as a system message, replacing earlier ones.
    pub async fn with_instructions(
        &self,
        db: &DatabaseConnection,
        chat: &mut Chat,
        instructions: &str,
    ) -> Result<()> {
        self.persist_system_instruction(db, instructions, false, false)
            .await?;
        self.sync_messages(db, chat).await
    }

    /// `chat.with_instructions(text, append:, persist:, cache_until_here:)`. Persisted
    /// instructions replace the chat's system rows (one existing row is updated in place, so it
    /// keeps its position) or, with `append`, add another. With `persist: false` they apply to
    /// this record's chats only and survive reloads of it. `None` clears them.
    pub async fn set_instructions(
        &mut self,
        db: &DatabaseConnection,
        chat: &mut Chat,
        instructions: Option<&str>,
        append: bool,
        persist: bool,
        cache_until_here: bool,
    ) -> Result<()> {
        match (persist, instructions) {
            (true, None) => self.clear_persisted_system_instructions(db).await?,
            (true, Some(text)) => {
                self.persist_system_instruction(db, text, append, cache_until_here)
                    .await?
            }
            (false, None) => self.runtime_instructions.clear(),
            (false, Some(text)) => {
                if !append {
                    self.runtime_instructions.clear();
                }
                self.runtime_instructions
                    .push((text.to_string(), append, cache_until_here));
            }
        }
        self.sync_messages(db, chat).await
    }

    async fn clear_persisted_system_instructions(&self, db: &impl ConnectionTrait) -> Result<()> {
        messages::Entity::delete_many()
            .filter(messages::Column::ChatId.eq(self.record.id))
            .filter(messages::Column::Role.eq("system"))
            .exec(db)
            .await?;
        Ok(())
    }

    async fn persist_system_instruction(
        &self,
        db: &DatabaseConnection,
        text: &str,
        append: bool,
        cache_until_here: bool,
    ) -> Result<()> {
        let txn = db.begin().await?;
        let existing = messages::Entity::find()
            .filter(messages::Column::ChatId.eq(self.record.id))
            .filter(messages::Column::Role.eq("system"))
            .order_by_asc(messages::Column::Id)
            .all(&txn)
            .await?;
        if !append && existing.len() == 1 {
            // `update_persisted_system_instruction`: rewriting the same row keeps it ahead of the
            // conversation instead of moving it behind the user messages.
            let row = existing[0].clone();
            if row.content.as_deref() != Some(text) || row.cache_until_here != cache_until_here {
                let mut row: messages::ActiveModel = row.into();
                row.content = Set(Some(text.to_string()));
                row.cache_until_here = Set(cache_until_here);
                row.updated_at = Set(now());
                row.update(&txn).await?;
            }
        } else {
            if !append {
                self.clear_persisted_system_instructions(&txn).await?;
            }
            let mut m = Message::system(text);
            m.cache_until_here = cache_until_here;
            insert_message(&txn, self.record.id, &m).await?;
        }
        txn.commit().await?;
        Ok(())
    }

    /// `chat.add_message(message)`: persists the message with its tool calls and attachments,
    /// links a tool result to its call, and appends it to `chat`. Returns the row.
    pub async fn add_message(
        &self,
        db: &DatabaseConnection,
        chat: &mut Chat,
        mut message: Message,
    ) -> Result<messages::Model> {
        let id = self.persist(db, &message, &[], &[]).await?;
        message.record_id = Some(id);
        chat.add_message(message);
        messages::Entity::find_by_id(id as i32)
            .one(db)
            .await?
            .ok_or_else(|| Error::NotFound(format!("message {id}")))
    }

    /// `chat.add_message(message_record)`: copies an existing message row (from any chat) into
    /// this conversation as a new row, through its `to_llm`. The original stays where it is.
    pub async fn add_message_record(
        &self,
        db: &DatabaseConnection,
        chat: &mut Chat,
        row: &messages::Model,
    ) -> Result<messages::Model> {
        let mut message = message_to_llm(db, row).await?;
        message.record_id = None;
        self.add_message(db, chat, message).await
    }

    /// `chat.add_completion(response)`: appends an answer produced out of band (a batch) to
    /// `chat` and persists it with its usage, as RubyLLM's persistence callbacks do.
    pub async fn add_completion(
        &self,
        db: &DatabaseConnection,
        chat: &mut Chat,
        message: Message,
    ) -> Result<Message> {
        let pending: Arc<Mutex<Vec<UsageEntry>>> = Arc::default();
        let sink = pending.clone();
        chat.set_usage_recorder(Box::new(move |e| sink.lock().unwrap().push(e.clone())));
        chat.add_completion(message, false);
        self.persist_unsaved(db, chat, Usages::Pending(&pending))
            .await?;
        let orphans = std::mem::take(&mut *pending.lock().unwrap()); // poisoned lock only
        for entry in orphans {
            insert_usage(db, self.record.id, None, &entry).await?;
        }
        Ok(chat
            .messages()
            .last()
            .cloned()
            .unwrap_or_else(|| Message::new(Role::Assistant, None)))
    }

    /// Persists the messages a batch appended to `chat` (`add_completion(record_usage: true)`),
    /// with the usage entries they carry, as the record's persistence callbacks do in Ruby.
    pub async fn persist_collected(&self, db: &DatabaseConnection, chat: &mut Chat) -> Result<()> {
        let pending: Arc<Mutex<Vec<UsageEntry>>> = Arc::new(Mutex::new(
            chat.messages()
                .iter()
                .filter(|m| m.record_id.is_none())
                .flat_map(|m| m.usage_entries.clone())
                .collect(),
        ));
        self.persist_unsaved(db, chat, Usages::Pending(&pending))
            .await
    }

    /// `chat.destroy!`: deletes the chat with its messages, and the internal rows that belong to
    /// them (`dependent: :destroy` on messages, tool calls, usages, and attachments).
    pub async fn destroy(self, db: &DatabaseConnection) -> Result<()> {
        let ids: Vec<i64> = self
            .messages(db)
            .await?
            .iter()
            .map(|m| m.id as i64)
            .collect();
        let txn = db.begin().await?;
        rust_llm_tool_calls::Entity::delete_many()
            .filter(rust_llm_tool_calls::Column::MessageType.eq(MESSAGE_TYPE))
            .filter(rust_llm_tool_calls::Column::MessageId.is_in(ids.clone()))
            .exec(&txn)
            .await?;
        rust_llm_attachments::Entity::delete_many()
            .filter(rust_llm_attachments::Column::MessageType.eq(MESSAGE_TYPE))
            .filter(rust_llm_attachments::Column::MessageId.is_in(ids))
            .exec(&txn)
            .await?;
        rust_llm_usages::Entity::delete_many()
            .filter(rust_llm_usages::Column::ChatType.eq(CHAT_TYPE))
            .filter(rust_llm_usages::Column::ChatId.eq(self.record.id as i64))
            .exec(&txn)
            .await?;
        messages::Entity::delete_many()
            .filter(messages::Column::ChatId.eq(self.record.id))
            .exec(&txn)
            .await?;
        chats::Entity::delete_by_id(self.record.id)
            .exec(&txn)
            .await?;
        txn.commit().await?;
        Ok(())
    }

    /// `chat.cache_until_here`: marks the latest persisted message as a prompt cache boundary, or
    /// the latest in-memory one when nothing is persisted yet.
    pub async fn cache_until_here(&self, db: &DatabaseConnection, chat: &mut Chat) -> Result<()> {
        let last = messages::Entity::find()
            .filter(messages::Column::ChatId.eq(self.record.id))
            .order_by_desc(messages::Column::Id)
            .one(db)
            .await?;
        let Some(row) = last else {
            chat.cache_until_here()?;
            return Ok(());
        };
        let id = row.id as i64;
        let mut row: messages::ActiveModel = row.into();
        row.cache_until_here = Set(true);
        row.updated_at = Set(now());
        row.update(db).await?;
        if let Some(m) = chat
            .messages_mut()
            .iter_mut()
            .find(|m| m.record_id == Some(id))
        {
            m.cache_until_here = true;
        }
        Ok(())
    }

    /// `chat.ask(message)`: runs the loop and persists every message, tool call, and usage row.
    pub async fn ask(
        &self,
        db: &DatabaseConnection,
        chat: &mut Chat,
        message: &str,
    ) -> Result<Message> {
        self.ask_with(db, chat, message, Vec::new()).await
    }

    /// `chat.ask(message, with: [...])`: the attachments are stored with the user message.
    pub async fn ask_with(
        &self,
        db: &DatabaseConnection,
        chat: &mut Chat,
        message: &str,
        attachments: Vec<Attachment>,
    ) -> Result<Message> {
        self.ask_later_with(db, chat, message, attachments).await?;
        self.complete(db, chat).await
    }

    /// `chat.ask_later(message)`: persists the user message without calling the model.
    pub async fn ask_later(
        &self,
        db: &DatabaseConnection,
        chat: &mut Chat,
        message: &str,
    ) -> Result<()> {
        self.ask_later_with(db, chat, message, Vec::new()).await
    }

    /// `chat.ask_later(message, with: [...])`.
    pub async fn ask_later_with(
        &self,
        db: &DatabaseConnection,
        chat: &mut Chat,
        message: &str,
        attachments: Vec<Attachment>,
    ) -> Result<()> {
        chat.ask_later_with(message, attachments)?;
        self.persist_unsaved(db, chat, Usages::Written(&UsageRows::default()))
            .await
    }

    /// `chat.complete`: continue a staged or parked (awaiting approval or input) chat.
    ///
    /// Like RubyLLM's `install_persistence_callbacks`, each message is written the moment it is
    /// produced. The loop advances one `step` at a time and persists after every step, so a dropped
    /// request or a crash loses at most the step in flight, never earlier tool results or usage.
    /// While it runs, the `chats.cancelled` column is polled every [`CANCELLATION_POLL_INTERVAL`]
    /// (`consume_persisted_cancellation_request`), so [`ChatRecord::cancel`] from another process
    /// stops it with `rust_llm::Error::Cancelled`.
    pub async fn complete(&self, db: &DatabaseConnection, chat: &mut Chat) -> Result<Message> {
        let pending_usages = self.record_usages_as_they_finish(db, chat);

        // Anything not yet stored (ask_later on the plain chat) is written first, in order.
        self.persist_unsaved(db, chat, Usages::Written(&pending_usages))
            .await?;
        if self.consume_cancellation_request(db).await? {
            chat.cancel();
        }
        let poller = self.watch_cancellation(db, chat.cancel_handle());
        let outcome = self.run_loop(db, chat, &pending_usages).await;
        poller.abort();
        chat.clear_async_usage_recorder();
        // Attempts that produced no message stay in the ledger unlinked, as RubyLLM keeps them.
        if let Err(e) = outcome {
            if let Error::Llm(llm) = &e {
                self.cleanup_after_failure(db, chat, llm).await?;
            }
            return Err(e);
        }
        Ok(latest_message(chat))
    }

    /// `chat.ask(message) { |chunk| ... }`: [`ChatRecord::ask_later`], then
    /// [`ChatRecord::complete_stream`].
    pub async fn ask_stream(
        &self,
        db: &DatabaseConnection,
        chat: &mut Chat,
        message: &str,
        on_event: impl FnMut(StreamEvent<'_>) + Send,
    ) -> Result<Message> {
        self.ask_later(db, chat, message).await?;
        self.complete_stream(db, chat, on_event).await
    }

    /// `chat.complete { |chunk| ... }`: [`ChatRecord::complete`], streaming each response.
    ///
    /// As with RubyLLM's persistence callbacks, the assistant row is created (empty) before the
    /// first chunk arrives ([`StreamEvent::NewMessage`]), every chunk names that row
    /// ([`StreamEvent::Chunk`]), and the row is updated in place once the response is complete,
    /// with its tool calls and usage ([`StreamEvent::EndMessage`]). Tool results are written as
    /// they finish and reported as a `NewMessage` and an `EndMessage`. Chunk content is not
    /// written while streaming, so a failed or cancelled response leaves no row behind: the empty
    /// row is destroyed, as `cleanup_after_failure` does, along with an unfinished tool round.
    pub async fn complete_stream(
        &self,
        db: &DatabaseConnection,
        chat: &mut Chat,
        mut on_event: impl FnMut(StreamEvent<'_>) + Send,
    ) -> Result<Message> {
        let pending_usages = self.record_usages_as_they_finish(db, chat);
        self.persist_unsaved(db, chat, Usages::Written(&pending_usages))
            .await?;
        if self.consume_cancellation_request(db).await? {
            chat.cancel();
        }
        let poller = self.watch_cancellation(db, chat.cancel_handle());
        let outcome = self
            .run_stream_loop(db, chat, &pending_usages, &mut on_event)
            .await;
        poller.abort();
        chat.clear_async_usage_recorder();
        if let Err(e) = outcome {
            if let Error::Llm(llm) = &e {
                self.cleanup_after_failure(db, chat, llm).await?;
            }
            return Err(e);
        }
        Ok(latest_message(chat))
    }

    /// `chat.compact`: compacts the model context and persists the returned assistant message
    /// (carrying the compacted context) with its usage, without deleting earlier messages. A
    /// cancellation another process wrote stops it before the request.
    pub async fn compact(&self, db: &DatabaseConnection, chat: &mut Chat) -> Result<Message> {
        let pending_usages = self.record_usages_as_they_finish(db, chat);
        self.persist_unsaved(db, chat, Usages::Written(&pending_usages))
            .await?;
        if self.consume_cancellation_request(db).await? {
            chat.cancel();
        }
        let outcome = chat.compact().await;
        self.persist_unsaved(db, chat, Usages::Written(&pending_usages))
            .await?;
        chat.clear_async_usage_recorder();
        match outcome {
            Ok(message) => Ok(message),
            Err(e) => {
                self.cleanup_after_failure(db, chat, &e).await?;
                Err(e.into())
            }
        }
    }

    /// `chat.run_tools`: runs the pending tool calls (answering remote approvals with the
    /// recorded decisions) and persists their results, without calling the model.
    pub async fn run_tools(&self, db: &DatabaseConnection, chat: &mut Chat) -> Result<()> {
        let pending_usages: UsageRows = Arc::default();
        self.refresh_tool_call_state(db, chat).await?;
        let inputs_before = chat.tool_call_inputs().clone();
        let outcome = chat.run_tools().await.map(|_| ());
        self.persist_unsaved(db, chat, Usages::Written(&pending_usages))
            .await?;
        self.persist_tool_call_inputs(db, chat, &inputs_before)
            .await?;
        Ok(outcome?)
    }

    async fn run_loop(
        &self,
        db: &DatabaseConnection,
        chat: &mut Chat,
        pending_usages: &UsageRows,
    ) -> Result<()> {
        loop {
            self.refresh_tool_call_state(db, chat).await?;
            if chat.is_complete() || chat.is_awaiting_approval() || chat.is_awaiting_input() {
                return Ok(());
            }
            let inputs_before = chat.tool_call_inputs().clone();
            let step = chat.step().await;
            self.persist_unsaved(db, chat, Usages::Written(pending_usages))
                .await?;
            self.persist_tool_call_inputs(db, chat, &inputs_before)
                .await?;
            match step {
                Ok(Some(_)) => {}
                Ok(None) => return Ok(()),
                Err(e) => return Err(e.into()),
            }
        }
    }

    /// [`ChatRecord::run_loop`] for [`ChatRecord::complete_stream`]: a response gets its row
    /// before the request (`persist_new_message`) and is written into it afterwards; tool results
    /// are written as they finish.
    async fn run_stream_loop(
        &self,
        db: &DatabaseConnection,
        chat: &mut Chat,
        pending_usages: &UsageRows,
        on_event: &mut (dyn FnMut(StreamEvent<'_>) + Send),
    ) -> Result<()> {
        loop {
            self.refresh_tool_call_state(db, chat).await?;
            if chat.is_complete() || chat.is_awaiting_approval() || chat.is_awaiting_input() {
                return Ok(());
            }
            let inputs_before = chat.tool_call_inputs().clone();
            if has_unanswered_tool_calls(chat) {
                // `run_tools`: each result is written and reported once it has finished.
                let step = chat.step().await;
                let ids = self
                    .persist_unsaved_into(db, chat, Usages::Written(pending_usages), None)
                    .await?;
                self.persist_tool_call_inputs(db, chat, &inputs_before)
                    .await?;
                for id in ids {
                    let row = find_message(db, id).await?;
                    on_event(StreamEvent::NewMessage(&row));
                    on_event(StreamEvent::EndMessage(&row));
                }
                match step {
                    Ok(Some(_)) => continue,
                    Ok(None) => return Ok(()),
                    Err(e) => return Err(e.into()),
                }
            }
            // `persist_new_message`: the response's row exists before its first chunk.
            let blank = Message::new(Role::Assistant, Some(String::new()));
            let placeholder = insert_message(db, self.record.id, &blank).await?;
            on_event(StreamEvent::NewMessage(&placeholder));
            let message_id = placeholder.id;
            let step = chat
                .step_stream(|chunk| on_event(StreamEvent::Chunk { message_id, chunk }))
                .await;
            if let Err(e) = step {
                // `cleanup_failed_messages`: the blank row goes; usage stays in the ledger.
                messages::Entity::delete_by_id(message_id).exec(db).await?;
                return Err(e.into());
            }
            self.persist_unsaved_into(
                db,
                chat,
                Usages::Written(pending_usages),
                Some(&placeholder),
            )
            .await?;
            self.persist_tool_call_inputs(db, chat, &inputs_before)
                .await?;
            let row = find_message(db, i64::from(message_id)).await?;
            on_event(StreamEvent::EndMessage(&row));
        }
    }

    /// `persist_usage_entry`: installs a recorder that writes each finished attempt's
    /// `rust_llm_usages` row as it finishes, before `usage.rust_llm` is published (Ruby's tracker
    /// `on_finish`). Rows start unlinked; `persist` links them to the message they produced.
    fn record_usages_as_they_finish(&self, db: &DatabaseConnection, chat: &mut Chat) -> UsageRows {
        let rows: UsageRows = Arc::default();
        let (db, chat_id, sink) = (db.clone(), self.record.id, rows.clone());
        chat.set_async_usage_recorder(Arc::new(move |entry: UsageEntry| {
            let (db, sink) = (db.clone(), sink.clone());
            Box::pin(async move {
                match insert_usage(&db, chat_id, None, &entry).await {
                    Ok(id) => {
                        if let Ok(mut rows) = sink.lock() {
                            rows.push((entry.id, id));
                        }
                    }
                    Err(e) => tracing::error!("RustLLM: failed to persist usage row: {e}"),
                }
            })
        }));
        rows
    }

    /// Writes every non-system message without a `record_id`, in history order, and stamps the
    /// new ids. System messages reach the table only through `with_instructions`/`add_message`,
    /// as in RubyLLM, so runtime instructions stay runtime-only.
    async fn persist_unsaved(
        &self,
        db: &DatabaseConnection,
        chat: &mut Chat,
        usages: Usages<'_>,
    ) -> Result<()> {
        self.persist_unsaved_into(db, chat, usages, None)
            .await
            .map(|_| ())
    }

    /// [`ChatRecord::persist_unsaved`], writing the first unsaved message into `placeholder` (the
    /// row `persist_new_message` created for it) instead of a new row. Returns the ids written.
    async fn persist_unsaved_into(
        &self,
        db: &DatabaseConnection,
        chat: &mut Chat,
        usages: Usages<'_>,
        mut placeholder: Option<&messages::Model>,
    ) -> Result<Vec<i64>> {
        let mut written = Vec::new();
        let unsaved: Vec<usize> = chat
            .messages()
            .iter()
            .enumerate()
            .filter(|(_, m)| m.record_id.is_none() && m.role != Role::System)
            .map(|(i, _)| i)
            .collect();
        for i in unsaved {
            let message = chat.messages()[i].clone();
            let mut insert: Vec<UsageEntry> = Vec::new();
            let mut link: Vec<i32> = Vec::new();
            match &usages {
                Usages::Pending(pending) => {
                    let mut pending = pending.lock().unwrap(); // poisoned lock only
                    for entry in &message.usage_entries {
                        if let Some(p) = pending.iter().position(|u| u.id == entry.id) {
                            insert.push(pending.remove(p));
                        }
                    }
                }
                Usages::Written(rows) => {
                    let mut rows = rows.lock().unwrap(); // poisoned lock only
                    for entry in &message.usage_entries {
                        if let Some(p) = rows.iter().position(|(id, _)| *id == entry.id) {
                            link.push(rows.remove(p).1);
                        }
                    }
                }
            }
            let id = self
                .persist_into(db, &message, &insert, &link, placeholder.take())
                .await?;
            chat.messages_mut()[i].record_id = Some(id);
            written.push(id);
        }
        Ok(written)
    }

    async fn persist(
        &self,
        db: &DatabaseConnection,
        m: &Message,
        usages: &[UsageEntry],
        written_usages: &[i32],
    ) -> Result<i64> {
        self.persist_into(db, m, usages, written_usages, None).await
    }

    /// Writes `m` as a new row, or into `existing` (`persist_message_completion` assigning the
    /// placeholder's attributes), with its tool calls, attachments, and usage.
    async fn persist_into(
        &self,
        db: &DatabaseConnection,
        m: &Message,
        usages: &[UsageEntry],
        written_usages: &[i32],
        existing: Option<&messages::Model>,
    ) -> Result<i64> {
        // `persist_content`: read the bytes before the transaction, since a URL may be fetched.
        let mut files = Vec::new();
        for a in &m.attachments {
            let mut a = a.clone();
            match a.content().await {
                Ok(bytes) => files.push((a, bytes)),
                Err(e) => tracing::warn!(
                    "RustLLM: Failed to process attachment {:?}: {e}",
                    a.filename
                ),
            }
        }
        let txn = db.begin().await?;
        let row = match existing {
            Some(existing) => {
                let mut row = message_active_model(self.record.id, m);
                row.id = Set(existing.id);
                row.created_at = Set(existing.created_at);
                row.update(&txn).await?
            }
            None => insert_message(&txn, self.record.id, m).await?,
        };
        if let Some(calls) = &m.tool_calls {
            for call in calls.values() {
                rust_llm_tool_calls::ActiveModel {
                    message_type: Set(MESSAGE_TYPE.into()),
                    message_id: Set(row.id as i64),
                    tool_call_id: Set(call.id.clone()),
                    name: Set(call.name.clone()),
                    thought_signature: Set(call.thought_signature.clone()),
                    remote: Set(call.remote),
                    arguments: Set(Some(Value::Object(call.arguments()))),
                    created_at: Set(now()),
                    updated_at: Set(now()),
                    ..Default::default()
                }
                .insert(&txn)
                .await?;
            }
        }
        if let Some(id) = &m.tool_call_id
            && let Some(call) = self.find_tool_call(&txn, id).await?
        {
            let mut call: rust_llm_tool_calls::ActiveModel = call.into();
            call.result_type = Set(Some(MESSAGE_TYPE.into()));
            call.result_id = Set(Some(row.id as i64));
            call.updated_at = Set(now());
            call.update(&txn).await?;
        }
        for (a, bytes) in files {
            let metadata = a
                .resolution
                .map(|r| serde_json::json!({ "resolution": resolution_name(r) }));
            rust_llm_attachments::ActiveModel {
                message_type: Set(MESSAGE_TYPE.into()),
                message_id: Set(row.id as i64),
                filename: Set(a.filename.clone().unwrap_or_default()),
                content_type: Set(a.mime_type.clone()),
                byte_size: Set(bytes.len() as i64),
                metadata: Set(metadata),
                data: Set(bytes),
                created_at: Set(now()),
                ..Default::default()
            }
            .insert(&txn)
            .await?;
        }
        // `link_usage_entries`: every attempt behind this message, retries included.
        for entry in usages {
            insert_usage(&txn, self.record.id, Some(row.id), entry).await?;
        }
        if !written_usages.is_empty() {
            rust_llm_usages::Entity::update_many()
                .col_expr(
                    rust_llm_usages::Column::MessageId,
                    sea_orm::sea_query::Expr::value(row.id as i64),
                )
                .col_expr(
                    rust_llm_usages::Column::MessageType,
                    sea_orm::sea_query::Expr::value(MESSAGE_TYPE),
                )
                .col_expr(
                    rust_llm_usages::Column::UpdatedAt,
                    sea_orm::sea_query::Expr::value(now()),
                )
                .filter(rust_llm_usages::Column::Id.is_in(written_usages.to_vec()))
                .exec(&txn)
                .await?;
        }
        txn.commit().await?;
        Ok(row.id as i64)
    }

    /// `input_recorder`: writes the paused state of every tool call whose state changed during
    /// the step, clearing it (`NULL`) once the call resumed.
    async fn persist_tool_call_inputs(
        &self,
        db: &DatabaseConnection,
        chat: &Chat,
        before: &HashMap<String, Value>,
    ) -> Result<()> {
        let after = chat.tool_call_inputs();
        let mut changed: Vec<&String> = before
            .keys()
            .chain(after.keys())
            .filter(|id| before.get(*id) != after.get(*id))
            .collect();
        changed.sort();
        changed.dedup();
        for id in changed {
            self.write_tool_call_input(db, id, after.get(id).cloned())
                .await?;
        }
        Ok(())
    }

    async fn write_tool_call_input(
        &self,
        db: &DatabaseConnection,
        tool_call_id: &str,
        input: Option<Value>,
    ) -> Result<()> {
        if let Some(call) = self.find_tool_call(db, tool_call_id).await? {
            let mut call: rust_llm_tool_calls::ActiveModel = call.into();
            call.pending_input = Set(input);
            call.updated_at = Set(now());
            call.update(db).await?;
        }
        Ok(())
    }

    /// `find_tool_call`: only this chat's tool calls, never another chat's with the same id.
    async fn find_tool_call(
        &self,
        db: &impl ConnectionTrait,
        tool_call_id: &str,
    ) -> Result<Option<rust_llm_tool_calls::Model>> {
        let ids: Vec<i64> = self
            .messages(db)
            .await?
            .iter()
            .map(|m| m.id as i64)
            .collect();
        Ok(rust_llm_tool_calls::Entity::find()
            .filter(rust_llm_tool_calls::Column::ToolCallId.eq(tool_call_id))
            .filter(rust_llm_tool_calls::Column::MessageType.eq(MESSAGE_TYPE))
            .filter(rust_llm_tool_calls::Column::MessageId.is_in(ids))
            .one(db)
            .await?)
    }

    /// `cleanup_after_failure` / `cleanup_orphaned_tool_results`: a round that failed mid-way is
    /// rolled back, so the next `ask` starts clean instead of hitting `PendingToolCalls`.
    async fn cleanup_after_failure(
        &self,
        db: &DatabaseConnection,
        chat: &mut Chat,
        error: &rust_llm::Error,
    ) -> Result<()> {
        let reason = if matches!(error, rust_llm::Error::Cancelled) {
            "chat cancelled"
        } else {
            "API call failed"
        };
        self.cleanup_orphaned_tool_results_for(db, chat, reason)
            .await
    }

    /// `cleanup_orphaned_tool_results`: destroys a trailing tool-call message, or a whole round
    /// whose tool calls are not all answered. A completed round and a plain conversation stay.
    pub async fn cleanup_orphaned_tool_results(
        &self,
        db: &DatabaseConnection,
        chat: &mut Chat,
    ) -> Result<()> {
        self.cleanup_orphaned_tool_results_for(db, chat, "cleaning up orphaned tool results")
            .await
    }

    async fn cleanup_orphaned_tool_results_for(
        &self,
        db: &DatabaseConnection,
        chat: &mut Chat,
        reason: &str,
    ) -> Result<()> {
        let rows = self.messages(db).await?;
        let Some(last) = rows.last() else {
            return Ok(());
        };
        let own_calls = |message_id: i32| {
            rust_llm_tool_calls::Entity::find()
                .filter(rust_llm_tool_calls::Column::MessageType.eq(MESSAGE_TYPE))
                .filter(rust_llm_tool_calls::Column::MessageId.eq(message_id as i64))
                .all(db)
        };
        let mut doomed: Vec<i32> = Vec::new();
        let calls = own_calls(last.id).await?;
        if !calls.is_empty() {
            doomed.push(last.id);
        } else if let Some(parent) = rust_llm_tool_calls::Entity::find()
            .filter(rust_llm_tool_calls::Column::ResultType.eq(MESSAGE_TYPE))
            .filter(rust_llm_tool_calls::Column::ResultId.eq(last.id as i64))
            .one(db)
            .await?
        {
            let siblings = own_calls(parent.message_id as i32).await?;
            if siblings.iter().any(|c| c.result_id.is_none()) {
                doomed.extend(
                    siblings
                        .iter()
                        .filter_map(|c| c.result_id.map(|r| r as i32)),
                );
                doomed.push(parent.message_id as i32);
            }
        }
        if doomed.is_empty() {
            return Ok(());
        }
        let txn = db.begin().await?;
        for id in &doomed {
            tracing::warn!("RustLLM: {reason}, destroying message: {id}");
            rust_llm_tool_calls::Entity::delete_many()
                .filter(rust_llm_tool_calls::Column::MessageType.eq(MESSAGE_TYPE))
                .filter(rust_llm_tool_calls::Column::MessageId.eq(*id as i64))
                .exec(&txn)
                .await?;
            rust_llm_attachments::Entity::delete_many()
                .filter(rust_llm_attachments::Column::MessageType.eq(MESSAGE_TYPE))
                .filter(rust_llm_attachments::Column::MessageId.eq(*id as i64))
                .exec(&txn)
                .await?;
            rust_llm_usages::Entity::update_many()
                .col_expr(
                    rust_llm_usages::Column::MessageId,
                    sea_orm::sea_query::Expr::value(Option::<i64>::None),
                )
                .col_expr(
                    rust_llm_usages::Column::MessageType,
                    sea_orm::sea_query::Expr::value(Option::<String>::None),
                )
                .filter(rust_llm_usages::Column::MessageId.eq(*id as i64))
                .exec(&txn)
                .await?;
            messages::Entity::delete_by_id(*id).exec(&txn).await?;
        }
        txn.commit().await?;
        chat.messages_mut()
            .retain(|m| !m.record_id.is_some_and(|r| doomed.contains(&(r as i32))));
        Ok(())
    }

    // ---- cancellation ----------------------------------------------------------------------

    /// `chat.cancel`: records the request on the row, so a `complete` running in another process
    /// (a job) stops at its next checkpoint.
    pub async fn cancel(&self, db: &DatabaseConnection) -> Result<()> {
        chats::Entity::update_many()
            .col_expr(
                chats::Column::Cancelled,
                sea_orm::sea_query::Expr::value(true),
            )
            .filter(chats::Column::Id.eq(self.record.id))
            .exec(db)
            .await?;
        Ok(())
    }

    /// `chat.cancel` on a record whose chat is already built: persisted like [`ChatRecord::cancel`]
    /// and forwarded to `chat` (`@chat&.cancel`), so it stops at its next checkpoint without
    /// waiting for a poll.
    pub async fn cancel_chat(&self, db: &DatabaseConnection, chat: &Chat) -> Result<()> {
        self.cancel(db).await?;
        chat.cancel();
        Ok(())
    }

    /// `consume_persisted_cancellation_request`: reads the row at most once per
    /// [`CANCELLATION_POLL_INTERVAL`] (`cancellation_poll_due?`), and clears and reports a
    /// request it finds there.
    pub async fn consume_persisted_cancellation_request(
        &self,
        db: &DatabaseConnection,
    ) -> Result<bool> {
        {
            let mut last = self.last_cancellation_poll.lock().unwrap(); // poisoned lock only
            let now = std::time::Instant::now();
            if last.is_some_and(|at| now.duration_since(at) < CANCELLATION_POLL_INTERVAL) {
                return Ok(false);
            }
            *last = Some(now);
        }
        consume_cancellation(db, self.record.id).await
    }

    /// `chat.cancelled?`: whether a cancellation request is waiting on the row.
    pub async fn is_cancelled(&self, db: &DatabaseConnection) -> Result<bool> {
        Ok(chats::Entity::find_by_id(self.record.id)
            .one(db)
            .await?
            .is_some_and(|c| c.cancelled))
    }

    /// `consume_persisted_cancellation_request`: clears a waiting request and reports whether
    /// there was one. One statement, so two pollers never both consume it.
    async fn consume_cancellation_request(&self, db: &DatabaseConnection) -> Result<bool> {
        consume_cancellation(db, self.record.id).await
    }

    /// Polls the row while `complete` runs and cancels the chat when a request appears.
    fn watch_cancellation(
        &self,
        db: &DatabaseConnection,
        handle: rust_llm::CancelHandle,
    ) -> tokio::task::JoinHandle<()> {
        let db = db.clone();
        let id = self.record.id;
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(CANCELLATION_POLL_INTERVAL).await;
                match consume_cancellation(&db, id).await {
                    Ok(true) => {
                        handle.cancel();
                        return;
                    }
                    Ok(false) => {}
                    Err(e) => {
                        tracing::warn!("RustLLM: could not poll chat {id} for cancellation: {e}");
                        return;
                    }
                }
            }
        })
    }

    // ---- approvals and input requests ------------------------------------------------------

    /// `chat.approve(tool_call)`, persisted so another process can resume the chat.
    pub async fn approve(
        &self,
        db: &DatabaseConnection,
        chat: &mut Chat,
        tool_call_id: &str,
    ) -> Result<()> {
        self.record_decision(db, tool_call_id, "approved").await?;
        chat.approve(tool_call_id);
        Ok(())
    }

    /// `chat.deny(tool_call)`.
    pub async fn deny(
        &self,
        db: &DatabaseConnection,
        chat: &mut Chat,
        tool_call_id: &str,
    ) -> Result<()> {
        self.record_decision(db, tool_call_id, "denied").await?;
        chat.deny(tool_call_id);
        Ok(())
    }

    async fn record_decision(
        &self,
        db: &DatabaseConnection,
        tool_call_id: &str,
        decision: &str,
    ) -> Result<()> {
        let call = self
            .find_tool_call(db, tool_call_id)
            .await?
            .ok_or_else(|| Error::NotFound(format!("Unknown tool call: {tool_call_id:?}")))?;
        let mut call: rust_llm_tool_calls::ActiveModel = call.into();
        call.approval = Set(Some(decision.into()));
        call.updated_at = Set(now());
        call.update(db).await?;
        Ok(())
    }

    /// `chat.awaiting_approval?`, reading decisions other processes recorded.
    pub async fn is_awaiting_approval(
        &self,
        db: &DatabaseConnection,
        chat: &mut Chat,
    ) -> Result<bool> {
        self.refresh_tool_call_state(db, chat).await?;
        Ok(chat.is_awaiting_approval())
    }

    /// `chat.pending_approvals`: the tool-call rows that require approval and have no decision.
    pub async fn pending_approvals(
        &self,
        db: &DatabaseConnection,
        chat: &mut Chat,
    ) -> Result<Vec<rust_llm_tool_calls::Model>> {
        self.refresh_tool_call_state(db, chat).await?;
        let ids: Vec<String> = chat.pending_approvals().into_iter().map(|c| c.id).collect();
        Ok(self
            .tool_calls(db)
            .await?
            .into_iter()
            .filter(|c| ids.contains(&c.tool_call_id))
            .collect())
    }

    /// `chat.answer(request, **values)`: the answer is stored on the tool call, so any process
    /// can resume the call.
    pub async fn answer(
        &self,
        db: &DatabaseConnection,
        chat: &mut Chat,
        request: &rust_llm::InputRequest,
        values: Map<String, Value>,
    ) -> Result<()> {
        chat.answer(request, values)?;
        self.record_input(db, chat, request).await
    }

    /// `chat.decline(request)`.
    pub async fn decline(
        &self,
        db: &DatabaseConnection,
        chat: &mut Chat,
        request: &rust_llm::InputRequest,
    ) -> Result<()> {
        chat.decline(request)?;
        self.record_input(db, chat, request).await
    }

    async fn record_input(
        &self,
        db: &DatabaseConnection,
        chat: &Chat,
        request: &rust_llm::InputRequest,
    ) -> Result<()> {
        let Some(call) = &request.tool_call else {
            return Ok(());
        };
        self.write_tool_call_input(db, &call.id, chat.tool_call_inputs().get(&call.id).cloned())
            .await
    }

    // ---- accounting ------------------------------------------------------------------------

    /// `chat.tokens` from the persisted ledger.
    pub async fn tokens(&self, db: &DatabaseConnection) -> Result<rust_llm::Tokens> {
        let entries: Vec<UsageEntry> = self.usages(db).await?.iter().map(usage_entry).collect();
        Ok(rust_llm::Tokens::aggregate(
            entries.iter().map(|e| &e.tokens),
        ))
    }

    /// `chat.cost` from the persisted ledger, using the costs as recorded (never re-priced).
    pub async fn cost(&self, db: &DatabaseConnection) -> Result<rust_llm::Cost> {
        let entries: Vec<UsageEntry> = self.usages(db).await?.iter().map(usage_entry).collect();
        let complete = entries.iter().all(UsageEntry::cost_available);
        Ok(rust_llm::Cost::aggregate(
            entries.iter().map(|e| &e.cost),
            complete,
        ))
    }

    /// `chat.cost.total`: `None` when any attempt could not be priced, like the in-memory chat.
    pub async fn total_cost(&self, db: &DatabaseConnection) -> Result<Option<f64>> {
        Ok(self.cost(db).await?.total())
    }

    // ---- agents ----------------------------------------------------------------------------

    /// `Agent.create!` with `chat_model Chat` (`with_rails_chat_record`): a new record on the
    /// agent's model, built with its context, `assume_model_exists`, and protocol, with its
    /// configuration applied. Its instruction declarations are persisted unless declared
    /// `persist: false`.
    pub async fn create_for_agent<A: Agent + Sync + ?Sized>(
        db: &DatabaseConnection,
        agent: &A,
    ) -> Result<(ChatRecord, Chat)> {
        let context = agent.context();
        let default_model = context
            .as_ref()
            .map(|c| c.config().default_model.clone())
            .unwrap_or_else(|| rust_llm::config().default_model.clone());
        let record = Self::create_with(
            db,
            agent.model().unwrap_or(&default_model),
            agent.provider(),
            agent.assume_model_exists(),
        )
        .await?;
        record.configure_for_agent(db, agent, context, true).await
    }

    /// `Agent.find(id)`: the record with the agent's configuration applied at runtime. Its
    /// instructions apply without rewriting the persisted history.
    pub async fn find_for_agent<A: Agent + Sync + ?Sized>(
        db: &DatabaseConnection,
        id: i32,
        agent: &A,
    ) -> Result<(ChatRecord, Chat)> {
        let record = Self::find(db, id).await?;
        record.apply_agent(db, agent, false).await
    }

    /// `Agent.new(chat: record)`: this (reloaded) record with the agent's configuration applied,
    /// its instructions persisted like `create!`.
    pub async fn apply_agent<A: Agent + Sync + ?Sized>(
        self,
        db: &DatabaseConnection,
        agent: &A,
        persist_instructions: bool,
    ) -> Result<(ChatRecord, Chat)> {
        let context = agent.context();
        self.configure_for_agent(db, agent, context, persist_instructions)
            .await
    }

    /// `Agent.sync_instructions(record_or_id)`: re-renders the agent's instructions and persists
    /// the persistent declarations on the record, leaving `persist: false` ones out.
    pub async fn sync_instructions<A: Agent + Sync + ?Sized>(
        db: &DatabaseConnection,
        id: i32,
        agent: &A,
    ) -> Result<(ChatRecord, Chat)> {
        let mut record = Self::find(db, id).await?;
        record.apply_chat_options(agent);
        if let Some(context) = agent.context() {
            record.context = Some(context);
        }
        let mut chat = record.to_llm(db).await?;
        record
            .apply_agent_instructions(db, agent, &mut chat, true, true)
            .await?;
        Ok((record, chat))
    }

    /// `apply_chat_options`: the agent's `assume_model_exists` and `protocol` reach the record.
    fn apply_chat_options<A: Agent + Sync + ?Sized>(&mut self, agent: &A) {
        if agent.assume_model_exists() {
            self.assume_model_exists = true;
        }
        if let Some(protocol) = agent.protocol() {
            self.protocol = Some(protocol);
        }
    }

    /// `apply_configuration` on a record: chat options, then the context, then the
    /// instructions, then the rest of the agent's configuration.
    async fn configure_for_agent<A: Agent + Sync + ?Sized>(
        mut self,
        db: &DatabaseConnection,
        agent: &A,
        context: Option<rust_llm::Context>,
        persist_instructions: bool,
    ) -> Result<(ChatRecord, Chat)> {
        self.apply_chat_options(agent);
        if context.is_some() {
            self.context = context;
        }
        let mut chat = agent.apply_except_instructions(self.to_llm(db).await?)?;
        self.apply_agent_instructions(db, agent, &mut chat, persist_instructions, false)
            .await?;
        Ok((self, chat))
    }

    /// `apply_instructions`: each declaration, rendered with this record as `chat`, applied with
    /// `persist: persist && declaration.persist`. Blank ones are skipped (`blank_instruction?`).
    async fn apply_agent_instructions<A: Agent + Sync + ?Sized>(
        &mut self,
        db: &DatabaseConnection,
        agent: &A,
        chat: &mut Chat,
        persist: bool,
        persistent_only: bool,
    ) -> Result<()> {
        let runtime_chat = serde_json::json!({ "id": self.record.id });
        let config = chat.config().clone();
        for d in agent.instruction_declarations(&config, &runtime_chat)? {
            if (persistent_only && !d.persist) || d.text.trim().is_empty() {
                continue;
            }
            self.set_instructions(
                db,
                chat,
                Some(&d.text),
                d.append,
                persist && d.persist,
                d.cache_until_here,
            )
            .await?;
        }
        Ok(())
    }
}

/// `message.to_llm` for a `messages` row read from the database: its tool calls, the tool call it
/// answers, attachments, and usage entries are loaded with it, so `model`, `model_info`, and the
/// finish predicates answer as they do on the in-memory message.
pub async fn message_to_llm(db: &impl ConnectionTrait, row: &messages::Model) -> Result<Message> {
    let id = row.id as i64;
    let calls = rust_llm_tool_calls::Entity::find()
        .filter(rust_llm_tool_calls::Column::MessageType.eq(MESSAGE_TYPE))
        .filter(
            sea_orm::Condition::any()
                .add(rust_llm_tool_calls::Column::MessageId.eq(id))
                .add(rust_llm_tool_calls::Column::ResultId.eq(id)),
        )
        .order_by_asc(rust_llm_tool_calls::Column::Id)
        .all(db)
        .await?;
    let files = rust_llm_attachments::Entity::find()
        .filter(rust_llm_attachments::Column::MessageType.eq(MESSAGE_TYPE))
        .filter(rust_llm_attachments::Column::MessageId.eq(id))
        .order_by_asc(rust_llm_attachments::Column::Id)
        .all(db)
        .await?;
    let entries: Vec<(i32, Option<i64>, UsageEntry)> = rust_llm_usages::Entity::find()
        .filter(rust_llm_usages::Column::MessageType.eq(MESSAGE_TYPE))
        .filter(rust_llm_usages::Column::MessageId.eq(id))
        .order_by_asc(rust_llm_usages::Column::Id)
        .all(db)
        .await?
        .iter()
        .map(|u| (u.id, u.message_id, usage_entry(u)))
        .collect();
    restore_message(row, &calls, &files, &entries)
}

/// `MessageMethods#to_llm`: one `messages` row as a `rust_llm::Message`, with its tool calls, the
/// tool call it answers, its attachments, and its usage entries (`entries` are
/// `(usage id, message id, entry)`).
fn restore_message(
    row: &messages::Model,
    calls: &[rust_llm_tool_calls::Model],
    files: &[rust_llm_attachments::Model],
    entries: &[(i32, Option<i64>, UsageEntry)],
) -> Result<Message> {
    let own_calls: Vec<&rust_llm_tool_calls::Model> = calls
        .iter()
        .filter(|c| c.message_id == row.id as i64)
        .collect();
    let parent = calls.iter().find(|c| {
        c.result_id == Some(row.id as i64) && c.result_type.as_deref() == Some(MESSAGE_TYPE)
    });
    let mut m = Message::new(Role::parse(&row.role)?, row.content.clone());
    m.cache_until_here = row.cache_until_here;
    m.thinking = Thinking::build(row.thinking_text.clone(), row.thinking_signature.clone());
    m.citations = row
        .citations
        .clone()
        .and_then(|c| serde_json::from_value::<Vec<Citation>>(c).ok())
        .unwrap_or_default();
    m.server_tool_calls = row
        .server_tool_calls
        .clone()
        .and_then(|c| serde_json::from_value(c).ok())
        .unwrap_or_default();
    m.raw_content = row.raw_content.clone();
    m.raw_reasoning = row.raw_reasoning.clone();
    m.finish_reason = row.finish_reason.as_deref().map(FinishReason::from_symbol);
    m.tool_call_id = parent.map(|p| p.tool_call_id.clone());
    m.attachments = files
        .iter()
        .filter(|f| f.message_id == row.id as i64)
        .map(attachment)
        .collect();
    if !own_calls.is_empty() {
        let map: IndexMap<ToolCall> = own_calls
            .iter()
            .map(|c| {
                let mut call = ToolCall::new(
                    c.tool_call_id.clone(),
                    c.name.clone(),
                    c.arguments
                        .clone()
                        .and_then(|a| a.as_object().cloned())
                        .unwrap_or_default(),
                );
                call.thought_signature = c.thought_signature.clone();
                call.remote = c.remote;
                (c.tool_call_id.clone(), call)
            })
            .collect();
        m.tool_calls = Some(map);
    }
    m.usage_entries = entries
        .iter()
        .filter(|(_, message_id, _)| *message_id == Some(row.id as i64))
        .map(|(_, _, e)| e.clone())
        .collect();
    m.model = m
        .usage_entries
        .iter()
        .rev()
        .find(|e| e.status == UsageStatus::Succeeded)
        .map(|e| e.model.clone());
    m.record_id = Some(row.id as i64);
    Ok(m)
}

async fn consume_cancellation(db: &DatabaseConnection, id: i32) -> Result<bool> {
    let result = chats::Entity::update_many()
        .col_expr(
            chats::Column::Cancelled,
            sea_orm::sea_query::Expr::value(false),
        )
        .filter(chats::Column::Id.eq(id))
        .filter(chats::Column::Cancelled.eq(true))
        .exec(db)
        .await?;
    Ok(result.rows_affected > 0)
}

/// Decisions and paused inputs recorded on the rows, applied to `chat`.
fn apply_tool_call_state(chat: &mut Chat, calls: &[rust_llm_tool_calls::Model]) {
    chat.set_decisions(calls.iter().filter_map(|c| match c.approval.as_deref() {
        Some("approved") => Some((c.tool_call_id.clone(), true)),
        Some("denied") => Some((c.tool_call_id.clone(), false)),
        _ => None,
    }));
    chat.set_tool_call_inputs(calls.iter().filter_map(|c| {
        c.pending_input
            .clone()
            .filter(|i| !i.is_null())
            .map(|i| (c.tool_call_id.clone(), i))
    }));
}

fn resolution_name(resolution: Resolution) -> &'static str {
    match resolution {
        Resolution::Low => "low",
        Resolution::Medium => "medium",
        Resolution::High => "high",
        Resolution::UltraHigh => "ultra_high",
    }
}

/// An attachment row as the `RubyLLM::Attachment` `extract_attachments` builds from a blob.
fn attachment(row: &rust_llm_attachments::Model) -> Attachment {
    let a = Attachment::from_bytes(
        row.data.clone(),
        row.filename.clone(),
        Some(row.content_type.as_str()),
    );
    let resolution = row
        .metadata
        .as_ref()
        .and_then(|m| m.get("resolution"))
        .and_then(Value::as_str)
        .and_then(|r| match r {
            "low" => Some(Resolution::Low),
            "medium" => Some(Resolution::Medium),
            "high" => Some(Resolution::High),
            "ultra_high" => Some(Resolution::UltraHigh),
            _ => None,
        });
    match resolution {
        Some(r) => a.with_resolution(r),
        None => a,
    }
}

async fn insert_message(
    db: &impl ConnectionTrait,
    chat_id: i32,
    m: &Message,
) -> Result<messages::Model> {
    Ok(message_active_model(chat_id, m).insert(db).await?)
}

/// `message_attributes`: the row's columns for `m`.
fn message_active_model(chat_id: i32, m: &Message) -> messages::ActiveModel {
    let json_list = |v: Value| {
        if v.as_array().is_some_and(|a| a.is_empty()) {
            None
        } else {
            Some(v)
        }
    };
    messages::ActiveModel {
        chat_id: Set(chat_id),
        role: Set(m.role.as_str().into()),
        content: Set(m.content.clone()),
        cache_until_here: Set(m.cache_until_here),
        thinking_text: Set(m.thinking.as_ref().and_then(|t| t.text.clone())),
        thinking_signature: Set(m.thinking.as_ref().and_then(|t| t.signature.clone())),
        citations: Set(json_list(
            serde_json::to_value(&m.citations).unwrap_or_default(),
        )),
        server_tool_calls: Set(json_list(
            serde_json::to_value(&m.server_tool_calls).unwrap_or_default(),
        )),
        raw_content: Set(m.raw_content.clone()),
        raw_reasoning: Set(m.raw_reasoning.clone()),
        finish_reason: Set(m.finish_reason.as_ref().map(|f| f.as_str().to_string())),
        created_at: Set(now()),
        updated_at: Set(now()),
        ..Default::default()
    }
}

async fn find_message(db: &impl ConnectionTrait, id: i64) -> Result<messages::Model> {
    messages::Entity::find_by_id(id as i32)
        .one(db)
        .await?
        .ok_or_else(|| Error::NotFound(format!("message {id}")))
}

/// The message `complete` returns: the latest non-system one.
fn latest_message(chat: &Chat) -> Message {
    chat.messages()
        .iter()
        .rev()
        .find(|m| m.role != Role::System)
        .or(chat.messages().last())
        .cloned()
        .unwrap_or_else(|| Message::new(Role::Assistant, None))
}

/// `pending_tool_response`: the latest response asks for tools that have no result yet, so the
/// next `step` runs tools rather than calling the model.
fn has_unanswered_tool_calls(chat: &Chat) -> bool {
    let Some(response) = chat
        .messages()
        .iter()
        .rev()
        .find(|m| m.role != Role::System && !m.is_tool_result())
    else {
        return false;
    };
    let Some(calls) = response.tool_calls.as_ref().filter(|_| response.is_tool_call()) else {
        return false;
    };
    calls.values().any(|call| {
        !chat
            .messages()
            .iter()
            .any(|m| m.tool_call_id.as_deref() == Some(call.id.as_str()))
    })
}

async fn insert_usage(
    db: &impl ConnectionTrait,
    chat_id: i32,
    message_id: Option<i32>,
    e: &UsageEntry,
) -> Result<i32> {
    let i = |v: Option<i64>| v.map(|v| v as i32);
    Ok(rust_llm_usages::ActiveModel {
        chat_type: Set(CHAT_TYPE.into()),
        chat_id: Set(chat_id as i64),
        message_type: Set(message_id.map(|_| MESSAGE_TYPE.to_string())),
        message_id: Set(message_id.map(|id| id as i64)),
        operation: Set(e.operation.as_str().into()),
        provider: Set(e.provider.clone()),
        model: Set(e.model.clone()),
        status: Set(e.status.as_str().into()),
        input_tokens: Set(i(e.tokens.input)),
        output_tokens: Set(i(e.tokens.output)),
        cache_read_tokens: Set(i(e.tokens.cache_read)),
        cache_write_tokens: Set(i(e.tokens.cache_write)),
        thinking_tokens: Set(i(e.tokens.thinking)),
        input_cost: Set(e.cost.input),
        output_cost: Set(e.cost.output),
        cache_read_cost: Set(e.cost.cache_read),
        cache_write_cost: Set(e.cost.cache_write),
        thinking_cost: Set(e.cost.thinking),
        total_cost: Set(e.cost.total()),
        created_at: Set(now()),
        updated_at: Set(now()),
        ..Default::default()
    }
    .insert(db)
    .await?
    .id)
}

/// `rust_llm_usages` row -> `Accounting::Usage::Entry` (`Usage#to_entry`). The cost comes from
/// the stored columns (`Cost.from_h`), so provider-reported costs survive a reload.
fn usage_entry(u: &rust_llm_usages::Model) -> UsageEntry {
    let status = match u.status.as_str() {
        "succeeded" => UsageStatus::Succeeded,
        "failed" => UsageStatus::Failed,
        "cancelled" => UsageStatus::Cancelled,
        _ => UsageStatus::Pending,
    };
    let tokens = rust_llm::Tokens {
        input: u.input_tokens.map(i64::from),
        output: u.output_tokens.map(i64::from),
        cache_read: u.cache_read_tokens.map(i64::from),
        cache_write: u.cache_write_tokens.map(i64::from),
        thinking: u.thinking_tokens.map(i64::from),
        ..Default::default()
    };
    let cost = rust_llm::Cost::from_recorded(
        [
            u.input_cost,
            u.output_cost,
            u.cache_read_cost,
            u.cache_write_cost,
            u.thinking_cost,
        ],
        u.total_cost,
        &tokens,
    );
    UsageEntry {
        id: UsageEntry::next_id(),
        operation: rust_llm::message::Operation::Chat,
        provider: u.provider.clone(),
        model: u.model.clone(),
        status,
        cost,
        tokens,
    }
}
