//! Port of `lib/ruby_llm/accounting/usage.rb`: the per-attempt usage `Tracker`, the `Entry`
//! readers it relies on (the `UsageEntry` struct itself lives in `message.rs`), and the
//! `Accounting::Usage::Result` storage shared by results that carry attempt accounting.
//!
//! Ruby shares entry objects between the tracker, the caller, and the result. Here the tracker
//! owns its entries: `start` returns the entry's `id`, [`Tracker::entry`] reads it back, and
//! `succeed` attaches a snapshot of the ledger to the result.

use std::sync::Arc;

use serde_json::{Map, Value};

use crate::chat::UsageRecorder;
use crate::config::Config;
use crate::cost::{Cost, Tier};
use crate::error::Error;
use crate::message::{Message, Operation, UsageEntry, UsageStatus};
use crate::model::Model;
use crate::moderation::Moderation;
use crate::tokens::Tokens;

/// `Accounting::Usage::Entry` readers (`accounting/usage.rb`).
impl UsageEntry {
    /// `Entry.new(operation:, provider:, model:)`: a pending attempt with unknown tokens and an
    /// unpriced cost. The port stores an absent model (`model: nil`) as `""`; `to_h` reports it
    /// as `null`.
    pub fn new(
        operation: Operation,
        provider: impl Into<String>,
        model: Option<&str>,
    ) -> UsageEntry {
        let tokens = Tokens::default();
        UsageEntry {
            id: UsageEntry::next_id(),
            operation,
            provider: provider.into(),
            model: model.unwrap_or_default().to_string(),
            status: UsageStatus::Pending,
            cost: Cost::new(&tokens, None, Tier::Standard),
            tokens,
        }
    }

    pub fn is_pending(&self) -> bool {
        self.status == UsageStatus::Pending
    }

    pub fn is_succeeded(&self) -> bool {
        self.status == UsageStatus::Succeeded
    }

    pub fn is_failed(&self) -> bool {
        self.status == UsageStatus::Failed
    }

    pub fn is_cancelled(&self) -> bool {
        self.status == UsageStatus::Cancelled
    }

    /// `usage_available?`: `tokens.to_h.any?`.
    pub fn usage_available(&self) -> bool {
        !self.tokens.is_empty()
    }

    /// `Entry#to_h`.
    pub fn to_h(&self) -> Value {
        let mut h = Map::new();
        h.insert("operation".into(), self.operation.as_str().into());
        h.insert("provider".into(), self.provider.clone().into());
        h.insert(
            "model".into(),
            if self.model.is_empty() {
                Value::Null
            } else {
                self.model.clone().into()
            },
        );
        h.insert("status".into(), self.status.as_str().into());
        h.insert(
            "tokens".into(),
            crate::instrumentation::tokens_h(&self.tokens),
        );
        h.insert("cost".into(), crate::instrumentation::cost_h(&self.cost));
        Value::Object(h)
    }
}

/// `Accounting::Usage::Result`: a result object that carries attempt accounting, plus what
/// `Tracker#succeed` reads from it (`billed.tokens`, `billed.cost`, and for a `Message` its
/// `model_info=`).
pub trait UsageResult {
    /// `ruby_llm_usage_entries=`.
    fn set_usage_entries(&mut self, entries: Vec<UsageEntry>);
    /// `billed.tokens`.
    fn usage_tokens(&self) -> Tokens;
    /// `billed.cost`.
    fn usage_cost(&self) -> Cost;
    /// `billed.is_a?(Message)`: messages get their `model_info` resolved before pricing.
    fn as_message_mut(&mut self) -> Option<&mut Message> {
        None
    }
}

impl UsageResult for Message {
    fn set_usage_entries(&mut self, entries: Vec<UsageEntry>) {
        self.usage_entries = entries;
    }
    fn usage_tokens(&self) -> Tokens {
        self.tokens()
    }
    fn usage_cost(&self) -> Cost {
        self.cost(None)
    }
    fn as_message_mut(&mut self) -> Option<&mut Message> {
        Some(self)
    }
}

impl UsageResult for Moderation {
    fn set_usage_entries(&mut self, entries: Vec<UsageEntry>) {
        self.usage_entries = entries;
    }
    fn usage_tokens(&self) -> Tokens {
        self.tokens()
    }
    fn usage_cost(&self) -> Cost {
        self.cost()
    }
}

/// `Accounting::Usage::Tracker`: tracks the physical transport attempts belonging to one
/// operation.
pub struct Tracker {
    operation: Operation,
    provider: String,
    model_info: Option<Model>,
    config: Arc<Config>,
    on_finish: Option<UsageRecorder>,
    entries: Vec<UsageEntry>,
    pending: Vec<u64>,
}

impl Tracker {
    /// `Tracker.new(operation:, provider:, model:, config:, on_finish:)`. `provider` is the
    /// provider's slug (Ruby reads `provider.slug`).
    pub fn new(
        operation: Operation,
        provider: impl Into<String>,
        model: Option<Model>,
        config: Arc<Config>,
        on_finish: Option<UsageRecorder>,
    ) -> Tracker {
        Tracker {
            operation,
            provider: provider.into(),
            model_info: model,
            config,
            on_finish,
            entries: Vec::new(),
            pending: Vec::new(),
        }
    }

    /// `attr_reader :entries`.
    pub fn entries(&self) -> &[UsageEntry] {
        &self.entries
    }

    /// The entry `start` returned the id of.
    pub fn entry(&self, id: u64) -> Option<&UsageEntry> {
        self.entries.iter().find(|e| e.id == id)
    }

    fn entry_mut(&mut self, id: u64) -> Option<&mut UsageEntry> {
        self.entries.iter_mut().find(|e| e.id == id)
    }

    /// `start`: records a new pending attempt and returns its id.
    pub fn start(&mut self) -> u64 {
        let entry = UsageEntry::new(
            self.operation,
            self.provider.clone(),
            self.model_info.as_ref().map(|m| m.id.as_str()),
        );
        let id = entry.id;
        self.entries.push(entry);
        self.pending.push(id);
        id
    }

    /// `observe(chunk)`: a streamed chunk's token counts update the attempt in flight; a count
    /// the chunk does not report keeps the earlier value (`merge_stream_tokens`).
    pub fn observe(&mut self, chunk: &Message) {
        let Some(&id) = self.pending.last() else {
            return;
        };
        let incoming = chunk.tokens();
        let Some(entry) = self.entry_mut(id) else {
            return;
        };
        entry.tokens.merge_latest(&incoming);
        entry.cost = Cost::new(&entry.tokens, None, Tier::Standard);
    }

    /// `fail_attempt(entry, error)`: ignored unless the attempt is still pending.
    pub fn fail_attempt(&mut self, entry: Option<u64>, error: &Error) {
        let Some(id) = entry else { return };
        let Some(entry) = self.entry(id).filter(|e| e.is_pending()) else {
            return;
        };
        let status = if matches!(error, Error::Cancelled) {
            UsageStatus::Cancelled
        } else {
            UsageStatus::Failed
        };
        let tokens = self.failure_tokens(entry, error);
        self.finish(id, status, tokens, None);
    }

    /// `fail_pending(error)`: fails every attempt still in flight.
    pub fn fail_pending(&mut self, error: &Error) {
        for id in self.pending.clone() {
            self.fail_attempt(Some(id), error);
        }
    }

    /// `succeed(result)`: the last pending attempt is credited with the result's tokens and cost;
    /// earlier pending attempts succeed with no tokens. The ledger is then attached to the result,
    /// even when no attempt was recorded.
    pub fn succeed<R: UsageResult>(&mut self, result: &mut R) {
        if let Some(message) = result.as_message_mut() {
            message.model_info = self.message_model(message);
        }
        let pending = self.pending.clone();
        let Some((&last, earlier)) = pending.split_last() else {
            result.set_usage_entries(self.entries.clone());
            return;
        };
        let tokens = result.usage_tokens();
        let cost = result.usage_cost();
        for &id in earlier {
            self.finish(id, UsageStatus::Succeeded, Tokens::default(), None);
        }
        self.finish(last, UsageStatus::Succeeded, tokens, Some(cost));
        result.set_usage_entries(self.entries.clone());
    }

    /// `message_model`: the requested model unless the response echoes a different id this
    /// provider's registry knows.
    fn message_model(&self, message: &Message) -> Option<Model> {
        let Some(id) = message.model.as_deref() else {
            return self.model_info.clone();
        };
        if self.model_info.as_ref().is_some_and(|m| m.id == id) {
            return self.model_info.clone();
        }
        crate::models::models()
            .find(id, Some(&self.provider))
            .ok()
            .or_else(|| self.model_info.clone())
    }

    /// `failure_tokens`: without a model the tokens stay as observed; otherwise the rule the chat
    /// loop uses (`chat::failure_tokens`): observed stream tokens are kept, a request that never
    /// reached the provider or that it refused (4xx) is zero, anything else stays unknown.
    fn failure_tokens(&self, entry: &UsageEntry, error: &Error) -> Tokens {
        if self.model_info.is_none() {
            return entry.tokens.clone();
        }
        crate::chat::failure_tokens(error, entry.usage_available().then(|| entry.tokens.clone()))
    }

    /// `finish`: a supplied cost wins when it has a total; otherwise the tokens are priced for
    /// the operation's category (`CATEGORY_BY_OPERATION`).
    fn finish(&mut self, id: u64, status: UsageStatus, tokens: Tokens, cost: Option<Cost>) {
        let cost = cost
            .filter(|c| c.total().is_some())
            .unwrap_or_else(|| self.price(&tokens));
        let Some(entry) = self.entries.iter_mut().find(|e| e.id == id) else {
            return;
        };
        entry.status = status;
        entry.tokens = tokens;
        entry.cost = cost;
        self.pending.retain(|&p| p != id);
        let entry = entry.clone();
        if let Some(on_finish) = &mut self.on_finish {
            on_finish(&entry);
        }
        crate::instrumentation::usage(&self.config, &entry);
    }

    fn price(&self, tokens: &Tokens) -> Cost {
        let model = self.model_info.as_ref();
        match self.operation {
            Operation::Chat | Operation::Moderation | Operation::Ocr | Operation::Judgment => {
                Cost::new(tokens, model, Tier::Standard)
            }
            Operation::Embedding | Operation::Rerank => {
                crate::rerank::embeddings_cost(tokens, model)
            }
            Operation::Image => Cost::images(tokens, model, None),
            Operation::Speech | Operation::Transcription => Cost::audio(tokens, model),
        }
    }
}
