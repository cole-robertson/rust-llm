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
//! ```ignore
//! let record = ChatRecord::create(&db, "claude-haiku-4-5", None).await?;
//! let mut chat = record.to_llm(&db).await?.with_tool(Weather);
//! record.ask(&db, &mut chat, "What's the weather in Berlin?").await?;
//! ```
//!
//! Like RubyLLM, every message, tool call, and billed attempt is written as it happens, so a
//! chat can be reloaded mid-round (e.g. parked on a tool approval) and continued later.

pub mod entities;
pub mod migrations;

use std::sync::{Arc, Mutex};

use rust_llm::message::indexmap_lite::IndexMap;
use rust_llm::{Chat, Citation, FinishReason, Message, Role, Thinking, ToolCall, UsageEntry, UsageStatus};
use sea_orm::{
    ActiveModelTrait, ActiveValue::Set, ColumnTrait, ConnectionTrait, DatabaseConnection, EntityTrait, QueryFilter, QueryOrder,
    TransactionTrait,
};
use serde_json::Value;

use entities::{chats, messages, rust_llm_models, rust_llm_tool_calls, rust_llm_usages};

/// The polymorphic type names written into `message_type`/`chat_type`, like Rails' class names.
pub const CHAT_TYPE: &str = "Chat";
pub const MESSAGE_TYPE: &str = "Message";

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

/// `RubyLLM::ActiveRecord::Model.find_or_create_by!(provider:, model_id:)` from the registry.
pub async fn find_or_create_model(db: &impl ConnectionTrait, model: &rust_llm::Model) -> Result<rust_llm_models::Model> {
    if let Some(existing) = rust_llm_models::Entity::find()
        .filter(rust_llm_models::Column::Provider.eq(&model.provider))
        .filter(rust_llm_models::Column::ModelId.eq(&model.id))
        .one(db)
        .await?
    {
        return Ok(existing);
    }
    let record = rust_llm_models::ActiveModel {
        model_id: Set(model.id.clone()),
        name: Set(model.name.clone()),
        provider: Set(model.provider.clone()),
        family: Set(model.family.clone()),
        context_window: Set(model.context_window.map(|v| v as i32)),
        max_output_tokens: Set(model.max_output_tokens.map(|v| v as i32)),
        modalities: Set(Some(serde_json::to_value(&model.modalities).unwrap_or_default())),
        capabilities: Set(Some(serde_json::to_value(&model.capabilities).unwrap_or_default())),
        pricing: Set(Some(serde_json::to_value(&model.pricing).unwrap_or_default())),
        metadata: Set(Some(Value::Object(model.metadata.clone()))),
        created_at: Set(now()),
        updated_at: Set(now()),
        ..Default::default()
    };
    Ok(record.insert(db).await?)
}

/// A persisted chat: the `Chat` model that `acts_as_chat`.
#[derive(Debug, Clone)]
pub struct ChatRecord {
    pub record: chats::Model,
}

impl ChatRecord {
    /// `Chat.create!(model:, provider:)`.
    pub async fn create(db: &DatabaseConnection, model: &str, provider: Option<&str>) -> Result<ChatRecord> {
        let assume = provider.and_then(rust_llm::Provider::resolve).is_some_and(|p| p.assume_models_exist());
        let info = if assume {
            rust_llm::models().find(model, provider).unwrap_or_else(|_| rust_llm::Model::default_for(model, provider.unwrap()))
        } else {
            rust_llm::models().find(model, provider)?
        };
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
        Ok(ChatRecord { record })
    }

    pub async fn find(db: &DatabaseConnection, id: i32) -> Result<ChatRecord> {
        let record = chats::Entity::find_by_id(id).one(db).await?.ok_or_else(|| Error::NotFound(format!("chat {id}")))?;
        Ok(ChatRecord { record })
    }

    pub fn id(&self) -> i32 {
        self.record.id
    }

    /// The chat's model row (`chat.model`).
    pub async fn model(&self, db: &impl ConnectionTrait) -> Result<rust_llm_models::Model> {
        rust_llm_models::Entity::find_by_id(self.record.rust_llm_model_id)
            .one(db)
            .await?
            .ok_or_else(|| Error::NotFound("rust_llm_model".into()))
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

    /// `chat.to_llm`: an in-memory `rust_llm::Chat` rebuilt from rows, including approval decisions.
    pub async fn to_llm(&self, db: &DatabaseConnection) -> Result<Chat> {
        self.to_llm_with(db, rust_llm::config()).await
    }

    /// `to_llm` with an explicit configuration (RubyLLM's `context:`).
    pub async fn to_llm_with(&self, db: &DatabaseConnection, config: Arc<rust_llm::Config>) -> Result<Chat> {
        let model = self.model(db).await?;
        let provider = rust_llm::Provider::resolve(&model.provider);
        let assume = provider.is_some_and(|p| p.assume_models_exist());
        let mut chat = Chat::with_config(config, Some(&model.model_id), Some(&model.provider), assume)?;
        let rows = self.messages(db).await?;
        let ids: Vec<i64> = rows.iter().map(|m| m.id as i64).collect();
        let calls = rust_llm_tool_calls::Entity::find()
            .filter(rust_llm_tool_calls::Column::MessageType.eq(MESSAGE_TYPE))
            .filter(rust_llm_tool_calls::Column::MessageId.is_in(ids))
            .order_by_asc(rust_llm_tool_calls::Column::Id)
            .all(db)
            .await?;
        let usages = self.usages(db).await?;

        let mut decisions = Vec::new();
        let mut restored = Vec::new();
        for row in &rows {
            let own_calls: Vec<&rust_llm_tool_calls::Model> = calls.iter().filter(|c| c.message_id == row.id as i64).collect();
            let parent = calls.iter().find(|c| c.result_id == Some(row.id as i64) && c.result_type.as_deref() == Some(MESSAGE_TYPE));
            let mut m = Message::new(Role::parse(&row.role)?, row.content.clone());
            m.cache_until_here = row.cache_until_here;
            m.thinking = Thinking::build(row.thinking_text.clone(), row.thinking_signature.clone());
            m.citations = row.citations.clone().and_then(|c| serde_json::from_value::<Vec<Citation>>(c).ok()).unwrap_or_default();
            m.raw_content = row.raw_content.clone();
            m.raw_reasoning = row.raw_reasoning.clone();
            m.finish_reason = row.finish_reason.as_deref().map(FinishReason::from_symbol);
            m.tool_call_id = parent.map(|p| p.tool_call_id.clone());
            if !own_calls.is_empty() {
                let map: IndexMap<ToolCall> = own_calls
                    .iter()
                    .map(|c| {
                        let mut call = ToolCall::new(
                            c.tool_call_id.clone(),
                            c.name.clone(),
                            c.arguments.clone().and_then(|a| a.as_object().cloned()).unwrap_or_default(),
                        );
                        call.thought_signature = c.thought_signature.clone();
                        call.remote = c.remote;
                        (c.tool_call_id.clone(), call)
                    })
                    .collect();
                m.tool_calls = Some(map);
            }
            for c in &own_calls {
                match c.approval.as_deref() {
                    Some("approved") => decisions.push((c.tool_call_id.clone(), true)),
                    Some("denied") => decisions.push((c.tool_call_id.clone(), false)),
                    _ => {}
                }
            }
            m.usage_entries = usages.iter().filter(|u| u.message_id == Some(row.id as i64)).map(usage_entry).collect();
            m.model = m.usage_entries.iter().rev().find(|e| e.status == UsageStatus::Succeeded).map(|e| e.model.clone());
            m.record_id = Some(row.id as i64);
            restored.push(m);
        }
        chat.set_messages(restored);
        chat.set_usage_entries(usages.iter().map(usage_entry).collect());
        chat.set_decisions(decisions);
        Ok(chat)
    }

    /// `chat.with_instructions(text)`: persisted as a system message, replacing earlier ones.
    pub async fn with_instructions(&self, db: &DatabaseConnection, chat: &mut Chat, instructions: &str) -> Result<()> {
        messages::Entity::delete_many()
            .filter(messages::Column::ChatId.eq(self.record.id))
            .filter(messages::Column::Role.eq("system"))
            .exec(db)
            .await?;
        let row = insert_message(db, self.record.id, &Message::system(instructions)).await?;
        chat.set_instructions(Some(instructions.into()), false, false);
        if let Some(m) = chat.messages_mut().iter_mut().rev().find(|m| m.role == Role::System) {
            m.record_id = Some(row.id as i64);
        }
        Ok(())
    }

    /// `chat.ask(message)`: runs the loop and persists every message, tool call, and usage row.
    pub async fn ask(&self, db: &DatabaseConnection, chat: &mut Chat, message: &str) -> Result<Message> {
        chat.ask_later(message)?;
        self.complete(db, chat).await
    }

    /// `chat.complete`: continue a staged or parked (awaiting approval) chat.
    ///
    /// Like RubyLLM's `install_persistence_callbacks`, each message is written the moment it is
    /// produced. The loop advances one `step` at a time and persists after every step, so a dropped
    /// request or a crash loses at most the step in flight, never earlier tool results or usage.
    pub async fn complete(&self, db: &DatabaseConnection, chat: &mut Chat) -> Result<Message> {
        let pending_usages: Arc<Mutex<Vec<UsageEntry>>> = Arc::default();
        let sink = pending_usages.clone();
        chat.set_usage_recorder(Box::new(move |e| sink.lock().unwrap().push(e.clone())));

        // Anything not yet stored (ask_later, runtime instructions) is written first, in order.
        self.persist_unsaved(db, chat, &pending_usages).await?;
        let outcome = loop {
            if chat.is_complete() || chat.is_awaiting_approval() {
                break Ok(());
            }
            let step = chat.step().await;
            self.persist_unsaved(db, chat, &pending_usages).await?;
            match step {
                Ok(Some(_)) => {}
                Ok(None) => break Ok(()),
                Err(e) => break Err(e),
            }
        };
        // Attempts that produced no message still land in the ledger, as RubyLLM keeps them.
        let orphans = std::mem::take(&mut *pending_usages.lock().unwrap());
        for entry in orphans {
            insert_usage(db, self.record.id, None, &entry).await?;
        }
        if let Err(e) = outcome {
            self.cleanup_after_failure(db, chat).await?;
            return Err(e.into());
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

    /// Writes every message without a `record_id`, in history order, and stamps the new ids.
    async fn persist_unsaved(&self, db: &DatabaseConnection, chat: &mut Chat, usages: &Arc<Mutex<Vec<UsageEntry>>>) -> Result<()> {
        let unsaved: Vec<usize> =
            chat.messages().iter().enumerate().filter(|(_, m)| m.record_id.is_none()).map(|(i, _)| i).collect();
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

    async fn persist(&self, db: &DatabaseConnection, m: &Message, usages: &[UsageEntry]) -> Result<i64> {
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
        // `link_usage_entries`: every attempt behind this message, retries included.
        for entry in usages {
            insert_usage(&txn, self.record.id, Some(row.id), entry).await?;
        }
        txn.commit().await?;
        Ok(row.id as i64)
    }

    /// `find_tool_call`: only this chat's tool calls, never another chat's with the same id.
    async fn find_tool_call(&self, db: &impl ConnectionTrait, tool_call_id: &str) -> Result<Option<rust_llm_tool_calls::Model>> {
        let ids: Vec<i64> = self.messages(db).await?.iter().map(|m| m.id as i64).collect();
        Ok(rust_llm_tool_calls::Entity::find()
            .filter(rust_llm_tool_calls::Column::ToolCallId.eq(tool_call_id))
            .filter(rust_llm_tool_calls::Column::MessageType.eq(MESSAGE_TYPE))
            .filter(rust_llm_tool_calls::Column::MessageId.is_in(ids))
            .one(db)
            .await?)
    }

    /// `cleanup_after_failure` / `cleanup_orphaned_tool_results`: a round that failed mid-way is
    /// rolled back, so the next `ask` starts clean instead of hitting `PendingToolCalls`.
    async fn cleanup_after_failure(&self, db: &DatabaseConnection, chat: &mut Chat) -> Result<()> {
        let rows = self.messages(db).await?;
        let Some(last) = rows.last() else { return Ok(()) };
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
                doomed.extend(siblings.iter().filter_map(|c| c.result_id.map(|r| r as i32)));
                doomed.push(parent.message_id as i32);
            }
        }
        if doomed.is_empty() {
            return Ok(());
        }
        let txn = db.begin().await?;
        for id in &doomed {
            tracing::warn!("RustLLM: API call failed, destroying message: {id}");
            rust_llm_tool_calls::Entity::delete_many()
                .filter(rust_llm_tool_calls::Column::MessageType.eq(MESSAGE_TYPE))
                .filter(rust_llm_tool_calls::Column::MessageId.eq(*id as i64))
                .exec(&txn)
                .await?;
            rust_llm_usages::Entity::update_many()
                .col_expr(rust_llm_usages::Column::MessageId, sea_orm::sea_query::Expr::value(Option::<i64>::None))
                .col_expr(rust_llm_usages::Column::MessageType, sea_orm::sea_query::Expr::value(Option::<String>::None))
                .filter(rust_llm_usages::Column::MessageId.eq(*id as i64))
                .exec(&txn)
                .await?;
            messages::Entity::delete_by_id(*id).exec(&txn).await?;
        }
        txn.commit().await?;
        chat.messages_mut().retain(|m| !m.record_id.is_some_and(|r| doomed.contains(&(r as i32))));
        Ok(())
    }

    /// `chat.approve(tool_call)`, persisted so another process can resume the chat.
    pub async fn approve(&self, db: &DatabaseConnection, chat: &mut Chat, tool_call_id: &str) -> Result<()> {
        self.record_decision(db, tool_call_id, "approved").await?;
        chat.approve(tool_call_id);
        Ok(())
    }

    /// `chat.deny(tool_call)`.
    pub async fn deny(&self, db: &DatabaseConnection, chat: &mut Chat, tool_call_id: &str) -> Result<()> {
        self.record_decision(db, tool_call_id, "denied").await?;
        chat.deny(tool_call_id);
        Ok(())
    }

    async fn record_decision(&self, db: &DatabaseConnection, tool_call_id: &str, decision: &str) -> Result<()> {
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

    /// `chat.tokens` from the persisted ledger.
    pub async fn tokens(&self, db: &DatabaseConnection) -> Result<rust_llm::Tokens> {
        let entries: Vec<UsageEntry> = self.usages(db).await?.iter().map(usage_entry).collect();
        Ok(rust_llm::Tokens::aggregate(entries.iter().map(|e| &e.tokens)))
    }

    /// `chat.cost` from the persisted ledger, using the costs as recorded (never re-priced).
    pub async fn cost(&self, db: &DatabaseConnection) -> Result<rust_llm::Cost> {
        let entries: Vec<UsageEntry> = self.usages(db).await?.iter().map(usage_entry).collect();
        let complete = entries.iter().all(UsageEntry::cost_available);
        Ok(rust_llm::Cost::aggregate(entries.iter().map(|e| &e.cost), complete))
    }

    /// `chat.cost.total`: `None` when any attempt could not be priced, like the in-memory chat.
    pub async fn total_cost(&self, db: &DatabaseConnection) -> Result<Option<f64>> {
        Ok(self.cost(db).await?.total())
    }
}

async fn insert_message(db: &impl ConnectionTrait, chat_id: i32, m: &Message) -> Result<messages::Model> {
    let json_list = |v: Value| if v.as_array().is_some_and(|a| a.is_empty()) { None } else { Some(v) };
    Ok(messages::ActiveModel {
        chat_id: Set(chat_id),
        role: Set(m.role.as_str().into()),
        content: Set(m.content.clone()),
        cache_until_here: Set(m.cache_until_here),
        thinking_text: Set(m.thinking.as_ref().and_then(|t| t.text.clone())),
        thinking_signature: Set(m.thinking.as_ref().and_then(|t| t.signature.clone())),
        citations: Set(json_list(serde_json::to_value(&m.citations).unwrap_or_default())),
        server_tool_calls: Set(json_list(serde_json::to_value(&m.server_tool_calls).unwrap_or_default())),
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

async fn insert_usage(db: &impl ConnectionTrait, chat_id: i32, message_id: Option<i32>, e: &UsageEntry) -> Result<()> {
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
        [u.input_cost, u.output_cost, u.cache_read_cost, u.cache_write_cost, u.thinking_cost],
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
