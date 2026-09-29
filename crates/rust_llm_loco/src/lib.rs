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

pub mod entities;
mod mcp_credential;
pub mod migrations;

pub use mcp_credential::McpCredentialStore;

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

fn now() -> sea_orm::prelude::DateTimeWithTimeZone {
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
/// RubyLLM also copies the whole registry into an empty `ruby_llm_models` table first, because its
/// registry can read from that table on the next boot. RustLLM's registry never reads the table,
/// so only the rows chats use are written.
pub async fn find_or_create_model(
    db: &impl ConnectionTrait,
    model: &rust_llm::Model,
) -> Result<rust_llm_models::Model> {
    let existing = || {
        rust_llm_models::Entity::find()
            .filter(rust_llm_models::Column::Provider.eq(&model.provider))
            .filter(rust_llm_models::Column::ModelId.eq(&model.id))
            .one(db)
    };
    if let Some(found) = existing().await? {
        return Ok(found);
    }
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
        Err(e) => existing().await?.ok_or(Error::Db(e)),
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

/// A persisted chat: the `Chat` model that `acts_as_chat`.
#[derive(Debug, Clone)]
pub struct ChatRecord {
    pub record: chats::Model,
    /// `assume_model_exists`: accept a model id the registry does not know. Not persisted; set
    /// it again after loading the record, as in RubyLLM.
    pub assume_model_exists: bool,
    /// `@unpersisted_instructions`, reapplied whenever the chat is rebuilt from rows.
    runtime_instructions: Vec<RuntimeInstruction>,
}

impl ChatRecord {
    fn from_row(record: chats::Model) -> ChatRecord {
        ChatRecord {
            record,
            assume_model_exists: false,
            runtime_instructions: Vec::new(),
        }
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
        self.to_llm_with(db, rust_llm::config()).await
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

        let mut restored = Vec::new();
        for row in &rows {
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
            restored.push(m);
        }
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
        let id = self.persist(db, &message, &[]).await?;
        message.record_id = Some(id);
        chat.add_message(message);
        messages::Entity::find_by_id(id as i32)
            .one(db)
            .await?
            .ok_or_else(|| Error::NotFound(format!("message {id}")))
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
        let pending: Arc<Mutex<Vec<UsageEntry>>> = Arc::default();
        self.persist_unsaved(db, chat, &pending).await
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
        let pending_usages: Arc<Mutex<Vec<UsageEntry>>> = Arc::default();
        let sink = pending_usages.clone();
        chat.set_usage_recorder(Box::new(move |e| sink.lock().unwrap().push(e.clone())));

        // Anything not yet stored (ask_later on the plain chat) is written first, in order.
        self.persist_unsaved(db, chat, &pending_usages).await?;
        if self.consume_cancellation_request(db).await? {
            chat.cancel();
        }
        let poller = self.watch_cancellation(db, chat.cancel_handle());
        let outcome = self.run_loop(db, chat, &pending_usages).await;
        poller.abort();
        // Attempts that produced no message still land in the ledger, as RubyLLM keeps them.
        let orphans = std::mem::take(&mut *pending_usages.lock().unwrap());
        for entry in orphans {
            insert_usage(db, self.record.id, None, &entry).await?;
        }
        if let Err(e) = outcome {
            if let Error::Llm(llm) = &e {
                self.cleanup_after_failure(db, chat, llm).await?;
            }
            return Err(e);
        }
        Ok(chat
            .messages()
            .iter()
            .rev()
            .find(|m| m.role != Role::System)
            .or(chat.messages().last())
            .cloned()
            .unwrap_or_else(|| Message::new(Role::Assistant, None)))
    }

    async fn run_loop(
        &self,
        db: &DatabaseConnection,
        chat: &mut Chat,
        pending_usages: &Arc<Mutex<Vec<UsageEntry>>>,
    ) -> Result<()> {
        loop {
            self.refresh_tool_call_state(db, chat).await?;
            if chat.is_complete() || chat.is_awaiting_approval() || chat.is_awaiting_input() {
                return Ok(());
            }
            let inputs_before = chat.tool_call_inputs().clone();
            let step = chat.step().await;
            self.persist_unsaved(db, chat, pending_usages).await?;
            self.persist_tool_call_inputs(db, chat, &inputs_before)
                .await?;
            match step {
                Ok(Some(_)) => {}
                Ok(None) => return Ok(()),
                Err(e) => return Err(e.into()),
            }
        }
    }

    /// Writes every non-system message without a `record_id`, in history order, and stamps the
    /// new ids. System messages reach the table only through `with_instructions`/`add_message`,
    /// as in RubyLLM, so runtime instructions stay runtime-only.
    async fn persist_unsaved(
        &self,
        db: &DatabaseConnection,
        chat: &mut Chat,
        usages: &Arc<Mutex<Vec<UsageEntry>>>,
    ) -> Result<()> {
        let unsaved: Vec<usize> = chat
            .messages()
            .iter()
            .enumerate()
            .filter(|(_, m)| m.record_id.is_none() && m.role != Role::System)
            .map(|(i, _)| i)
            .collect();
        for i in unsaved {
            let message = chat.messages()[i].clone();
            let mut linked: Vec<UsageEntry> = Vec::new();
            {
                let mut pending = usages.lock().unwrap();
                for entry in &message.usage_entries {
                    if let Some(p) = pending.iter().position(|u| u.id == entry.id) {
                        linked.push(pending.remove(p));
                    }
                }
            }
            let id = self.persist(db, &message, &linked).await?;
            chat.messages_mut()[i].record_id = Some(id);
        }
        Ok(())
    }

    async fn persist(
        &self,
        db: &DatabaseConnection,
        m: &Message,
        usages: &[UsageEntry],
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
        let row = insert_message(&txn, self.record.id, m).await?;
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
    /// agent's model with its configuration applied. Its instructions are persisted.
    pub async fn create_for_agent<A: Agent + Sync + ?Sized>(
        db: &DatabaseConnection,
        agent: &A,
    ) -> Result<(ChatRecord, Chat)> {
        let default_model = rust_llm::config().default_model.clone();
        let mut record = Self::create(
            db,
            agent.model().unwrap_or(&default_model),
            agent.provider(),
        )
        .await?;
        let chat = agent.apply(record.to_llm(db).await?)?;
        record
            .apply_agent_instructions(db, agent, chat, true)
            .await
            .map(|chat| (record, chat))
    }

    /// `Agent.find(id)`: the record with the agent's configuration applied at runtime. Its
    /// instructions apply without rewriting the persisted history.
    pub async fn find_for_agent<A: Agent + Sync + ?Sized>(
        db: &DatabaseConnection,
        id: i32,
        agent: &A,
    ) -> Result<(ChatRecord, Chat)> {
        let mut record = Self::find(db, id).await?;
        let chat = agent.apply(record.to_llm(db).await?)?;
        record
            .apply_agent_instructions(db, agent, chat, false)
            .await
            .map(|chat| (record, chat))
    }

    /// `apply_instructions`: an empty prompt means no instructions (`blank_instruction?`).
    async fn apply_agent_instructions<A: Agent + Sync + ?Sized>(
        &mut self,
        db: &DatabaseConnection,
        agent: &A,
        mut chat: Chat,
        persist: bool,
    ) -> Result<Chat> {
        if let Some(text) = agent.instructions().filter(|t| !t.trim().is_empty()) {
            self.set_instructions(db, &mut chat, Some(&text), false, persist, false)
                .await?;
        }
        Ok(chat)
    }
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
    let json_list = |v: Value| {
        if v.as_array().is_some_and(|a| a.is_empty()) {
            None
        } else {
            Some(v)
        }
    };
    Ok(messages::ActiveModel {
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
    .insert(db)
    .await?)
}

async fn insert_usage(
    db: &impl ConnectionTrait,
    chat_id: i32,
    message_id: Option<i32>,
    e: &UsageEntry,
) -> Result<()> {
    let i = |v: Option<i64>| v.map(|v| v as i32);
    rust_llm_usages::ActiveModel {
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
    .await?;
    Ok(())
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
