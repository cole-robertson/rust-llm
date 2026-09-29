//! Port of `lib/ruby_llm/chat.rb`: the conversation loop.
//!
//! ```ruby
//! chat = RubyLLM.chat(model: "claude-haiku-4-5").with_tools(Weather)
//! chat.ask "What's the weather in Berlin?"
//! ```
//!
//! ```ignore
//! let mut chat = ruby_llm::chat().with_model("claude-haiku-4-5")?.with_tool(Weather);
//! chat.ask("What's the weather in Berlin?").await?;
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use serde_json::{Map, Value};

use crate::attachment::Attachment;
use crate::config::Config;
use crate::cost::{Cost, Tier};
use crate::error::{DEFAULT_FALLBACK_ERRORS, Error, ErrorKind, Result};
use crate::message::{Message, Operation, Role, ToolCall, UsageEntry, UsageStatus};
use crate::model::Model;
use crate::models;
use crate::protocols::{self, Request, Schema, StreamAccumulator, StreamState, ToolCalls, ToolChoice, ToolPrefs};
use crate::providers::{ProtocolName, Provider};
use crate::thinking::ThinkingConfig;
use crate::tokens::Tokens;
use crate::tool::{SharedTool, Tool, ToolResult, validate_arguments};
use crate::transport::Connection;

type MessageCallback = Box<dyn FnMut(&Message) + Send>;
type UnitCallback = Box<dyn FnMut() + Send>;
type ToolCallCallback = Box<dyn FnMut(&ToolCall) + Send>;
type ToolResultCallback = Box<dyn FnMut(&ToolResult) + Send>;
type FallbackCallback = Box<dyn FnMut(&FallbackAttempt) + Send>;
type RequestCallback = Box<dyn FnMut(&mut Value) + Send>;
/// Called with each persisted-state change; the Loco integration uses these to write rows.
pub type UsageRecorder = Box<dyn FnMut(&UsageEntry) + Send>;

/// A model to try when generation fails (`with_fallbacks`).
#[derive(Debug, Clone)]
pub struct Fallback {
    pub model: String,
    pub provider: Option<String>,
}

impl From<&str> for Fallback {
    fn from(model: &str) -> Self {
        Fallback { model: model.into(), provider: None }
    }
}

/// What `before_fallback`/`after_fallback` receive.
#[derive(Debug, Clone)]
pub struct FallbackAttempt {
    pub attempt: usize,
    pub error: String,
    pub from: String,
    pub to: String,
    pub streaming: bool,
    pub chunks_yielded: bool,
    pub succeeded: Option<bool>,
}

#[derive(Default)]
struct Callbacks {
    before_message: Vec<UnitCallback>,
    after_message: Vec<MessageCallback>,
    before_tool_call: Vec<ToolCallCallback>,
    after_tool_result: Vec<ToolResultCallback>,
    before_fallback: Vec<FallbackCallback>,
    after_fallback: Vec<FallbackCallback>,
    before_request: Vec<RequestCallback>,
}

/// `RubyLLM::Chat`.
pub struct Chat {
    config: Arc<Config>,
    model: Model,
    provider: Provider,
    protocol: Option<ProtocolName>,
    connection: Connection,
    messages: Vec<Message>,
    tools: Vec<SharedTool>,
    tool_prefs: ToolPrefs,
    temperature: Option<f64>,
    max_output_tokens: Option<i64>,
    schema: Option<Schema>,
    thinking: Option<ThinkingConfig>,
    provider_options: Value,
    headers: Vec<(String, String)>,
    fallbacks: Vec<Fallback>,
    fallback_errors: Vec<ErrorKind>,
    callbacks: Callbacks,
    usage_entries: Vec<UsageEntry>,
    usage_recorder: Option<UsageRecorder>,
    tool_call_decisions: HashMap<String, bool>,
    cancelled: Arc<AtomicBool>,
}

impl std::fmt::Debug for Chat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Chat")
            .field("model", &self.model.id)
            .field("provider", &self.provider.slug())
            .field("messages", &self.messages.len())
            .field("tools", &self.tools.iter().map(|t| t.name()).collect::<Vec<_>>())
            .finish()
    }
}

/// Handle to cancel a running chat from another task (`chat.cancel`).
#[derive(Clone)]
pub struct CancelHandle(Arc<AtomicBool>);

impl CancelHandle {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

/// `Models.resolve`: find the model and provider, honoring `assume_model_exists` and local providers.
pub(crate) fn resolve_model(
    model_id: &str,
    provider: Option<&str>,
    assume_model_exists: bool,
) -> Result<(Model, Provider)> {
    let provider_kind = provider.map(Provider::resolve_or_err).transpose()?;
    let assume = assume_model_exists || provider_kind.is_some_and(|p| p.assume_models_exist());
    if assume {
        let provider_kind = provider_kind
            .ok_or_else(|| Error::Argument("Provider must be specified if assume_model_exists is true".into()))?;
        let model = models::models()
            .find(model_id, Some(provider_kind.slug()))
            .unwrap_or_else(|_| Model::default_for(model_id, provider_kind.slug()));
        return Ok((model, provider_kind));
    }
    let model = models::models().find(model_id, provider)?;
    let provider = Provider::resolve_or_err(&model.provider)?;
    Ok((model, provider))
}

impl Chat {
    /// `RubyLLM.chat(model:, provider:)`. `None` uses `config.default_model`.
    pub fn new(model: Option<&str>, provider: Option<&str>) -> Result<Chat> {
        Chat::with_config(crate::config(), model, provider, false)
    }

    pub fn with_config(config: Arc<Config>, model: Option<&str>, provider: Option<&str>, assume_model_exists: bool) -> Result<Chat> {
        let model_id = model.unwrap_or(&config.default_model).to_string();
        let (model, provider) = resolve_model(&model_id, provider, assume_model_exists)?;
        provider.ensure_configured(&config)?;
        let connection = Connection::new(provider, config.clone())?;
        Ok(Chat {
            config,
            model,
            provider,
            protocol: None,
            connection,
            messages: Vec::new(),
            tools: Vec::new(),
            tool_prefs: ToolPrefs::default(),
            temperature: None,
            max_output_tokens: None,
            schema: None,
            thinking: None,
            provider_options: Value::Object(Map::new()),
            headers: Vec::new(),
            fallbacks: Vec::new(),
            fallback_errors: DEFAULT_FALLBACK_ERRORS.to_vec(),
            callbacks: Callbacks::default(),
            usage_entries: Vec::new(),
            usage_recorder: None,
            tool_call_decisions: HashMap::new(),
            cancelled: Arc::new(AtomicBool::new(false)),
        })
    }

    // ---- accessors -------------------------------------------------------------------------

    pub fn model(&self) -> &Model {
        &self.model
    }
    pub fn provider(&self) -> Provider {
        self.provider
    }
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }
    pub fn tools(&self) -> &[SharedTool] {
        &self.tools
    }
    pub fn temperature(&self) -> Option<f64> {
        self.temperature
    }
    pub fn usage_entries(&self) -> &[UsageEntry] {
        &self.usage_entries
    }
    pub fn config(&self) -> &Arc<Config> {
        &self.config
    }

    /// `chat.tokens`: everything billed across every attempt, retries and fallbacks included.
    pub fn tokens(&self) -> Tokens {
        Tokens::aggregate(self.usage_entries.iter().map(|e| &e.tokens))
    }

    /// `chat.cost`.
    pub fn cost(&self) -> Cost {
        let complete = self.usage_entries.iter().all(UsageEntry::cost_available);
        let cost = Cost::aggregate(self.usage_entries.iter().map(|e| &e.cost), complete);
        if complete { cost } else { cost.mark_incomplete() }
    }

    // ---- configuration (with_*) ------------------------------------------------------------

    /// `with_instructions(text)`: replaces earlier system messages.
    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.set_instructions(Some(instructions.into()), false, false);
        self
    }

    /// `with_instructions(text, append: true, cache_until_here:)`; `None` removes all instructions.
    pub fn set_instructions(&mut self, instructions: Option<String>, append: bool, cache_until_here: bool) -> &mut Self {
        if !append {
            self.messages.retain(|m| m.role != Role::System);
        }
        if let Some(text) = instructions {
            let mut m = Message::system(text);
            m.cache_until_here = cache_until_here;
            self.messages.push(m);
        }
        self
    }

    /// `with_tools(Weather)`.
    pub fn with_tool(mut self, tool: impl Tool + 'static) -> Self {
        self.add_tool(Arc::new(tool));
        self
    }

    pub fn with_tools(mut self, tools: impl IntoIterator<Item = SharedTool>) -> Self {
        for t in tools {
            self.add_tool(t);
        }
        self
    }

    pub fn add_tool(&mut self, tool: SharedTool) -> &mut Self {
        let name = tool.name();
        self.tools.retain(|t| t.name() != name);
        self.tools.push(tool);
        self
    }

    /// `with_tools(nil)`.
    pub fn clear_tools(&mut self) -> &mut Self {
        self.tools.clear();
        self
    }

    /// `with_tool_options(choice:)`.
    pub fn with_tool_choice(mut self, choice: ToolChoice) -> Result<Self> {
        if let ToolChoice::Tool(name) = &choice
            && !self.tools.iter().any(|t| &t.name() == name) {
                let mut valid = vec!["auto".to_string(), "none".into(), "required".into()];
                valid.extend(self.tools.iter().map(|t| t.name()));
                return Err(Error::InvalidToolChoice(format!(
                    "Invalid tool choice: {name}. Valid choices are: {}",
                    valid.join(", ")
                )));
            }
        self.tool_prefs.choice = Some(choice);
        Ok(self)
    }

    /// `with_tool_options(calls: :one | :many)`.
    pub fn with_tool_calls(mut self, calls: ToolCalls) -> Self {
        self.tool_prefs.calls = Some(calls);
        self
    }

    /// `with_model(id, provider:)`.
    pub fn with_model(mut self, model_id: &str, provider: Option<&str>) -> Result<Self> {
        self.switch_model(model_id, provider, false)?;
        self.protocol = None;
        Ok(self)
    }

    /// `with_model(id, provider:, assume_model_exists: true)`.
    pub fn with_assumed_model(mut self, model_id: &str, provider: &str) -> Result<Self> {
        self.switch_model(model_id, Some(provider), true)?;
        Ok(self)
    }

    fn switch_model(&mut self, model_id: &str, provider: Option<&str>, assume: bool) -> Result<()> {
        let (model, provider) = resolve_model(model_id, provider, assume)?;
        provider.ensure_configured(&self.config)?;
        self.connection = Connection::new(provider, self.config.clone())?;
        self.model = model;
        self.provider = provider;
        Ok(())
    }

    /// `with_model(..., protocol: :chat_completions)`.
    pub fn with_protocol(mut self, protocol: ProtocolName) -> Self {
        self.protocol = Some(protocol);
        self
    }

    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn with_max_output_tokens(mut self, max: i64) -> Self {
        self.max_output_tokens = Some(max);
        self
    }

    /// `with_thinking`, `with_thinking(false)`, `with_thinking(effort: :high)`, ...
    pub fn with_thinking(mut self, thinking: ThinkingConfig) -> Self {
        self.thinking = Some(thinking);
        self
    }

    /// `with_schema(schema)`. Accepts a raw JSON schema or `{ name:, schema:, strict: }`.
    pub fn with_schema(mut self, schema: Value) -> Self {
        self.schema = normalize_schema(schema);
        self
    }

    /// `with_schema(ProductSchema)` from a `schemars::JsonSchema` type.
    pub fn with_schema_for<T: schemars::JsonSchema>(self) -> Self {
        let name = T::schema_name().to_string();
        let mut schema = crate::tool::schema_for::<T>();
        if let Some(obj) = schema.as_object_mut() {
            obj.remove("$defs");
        }
        self.with_schema(serde_json::json!({ "name": name, "schema": schema }))
    }

    /// `with_provider_options(...)`: deep-merged into every rendered request.
    pub fn with_provider_options(mut self, options: Value) -> Self {
        protocols::deep_merge(&mut self.provider_options, &options);
        self
    }

    /// Deprecated RubyLLM name kept for 1.x muscle memory.
    pub fn with_params(self, params: Value) -> Self {
        self.with_provider_options(params)
    }

    pub fn with_headers(mut self, headers: impl IntoIterator<Item = (String, String)>) -> Self {
        self.headers.extend(headers);
        self
    }

    /// `with_fallbacks("gpt-4.1-mini", "claude-haiku-4-5")`.
    pub fn with_fallbacks(mut self, models: impl IntoIterator<Item = Fallback>) -> Self {
        self.fallbacks = models.into_iter().collect();
        self.fallback_errors = DEFAULT_FALLBACK_ERRORS.to_vec();
        self
    }

    /// `with_fallbacks(..., on: [...])`.
    pub fn with_fallback_errors(mut self, kinds: Vec<ErrorKind>) -> Self {
        self.fallback_errors = kinds;
        self
    }

    // ---- callbacks -------------------------------------------------------------------------

    pub fn before_message(mut self, f: impl FnMut() + Send + 'static) -> Self {
        self.callbacks.before_message.push(Box::new(f));
        self
    }
    pub fn after_message(mut self, f: impl FnMut(&Message) + Send + 'static) -> Self {
        self.callbacks.after_message.push(Box::new(f));
        self
    }
    pub fn before_tool_call(mut self, f: impl FnMut(&ToolCall) + Send + 'static) -> Self {
        self.callbacks.before_tool_call.push(Box::new(f));
        self
    }
    pub fn after_tool_result(mut self, f: impl FnMut(&ToolResult) + Send + 'static) -> Self {
        self.callbacks.after_tool_result.push(Box::new(f));
        self
    }
    pub fn before_fallback(mut self, f: impl FnMut(&FallbackAttempt) + Send + 'static) -> Self {
        self.callbacks.before_fallback.push(Box::new(f));
        self
    }
    pub fn after_fallback(mut self, f: impl FnMut(&FallbackAttempt) + Send + 'static) -> Self {
        self.callbacks.after_fallback.push(Box::new(f));
        self
    }
    /// `before_request { |payload| ... }`: last chance to edit the rendered payload.
    pub fn before_request(mut self, f: impl FnMut(&mut Value) + Send + 'static) -> Self {
        self.callbacks.before_request.push(Box::new(f));
        self
    }

    pub fn set_usage_recorder(&mut self, recorder: UsageRecorder) {
        self.usage_recorder = Some(recorder);
    }

    // ---- messages --------------------------------------------------------------------------

    /// `add_message`: append without calling the model.
    pub fn add_message(&mut self, message: Message) -> &Message {
        self.messages.push(message);
        self.messages.last().unwrap()
    }

    /// Replaces history, e.g. when reloading a persisted chat.
    pub fn set_messages(&mut self, messages: Vec<Message>) {
        self.messages = messages;
    }

    pub fn set_usage_entries(&mut self, entries: Vec<UsageEntry>) {
        self.usage_entries = entries;
    }

    /// `cache_until_here`: mark the last message as a prompt-cache boundary.
    pub fn cache_until_here(&mut self) -> Result<&mut Self> {
        let last = self.messages.last_mut().ok_or_else(|| Error::Argument("No messages to cache".into()))?;
        last.cache_until_here = true;
        Ok(self)
    }

    // ---- the loop --------------------------------------------------------------------------

    /// `ask(message)`: stage a user message, then run the loop until the model answers.
    pub async fn ask(&mut self, message: impl Into<String>) -> Result<Message> {
        self.ask_later(message)?;
        self.complete().await
    }

    /// `ask(message, with: [...])`.
    pub async fn ask_with(&mut self, message: impl Into<String>, attachments: Vec<Attachment>) -> Result<Message> {
        self.ask_later_with(message, attachments)?;
        self.complete().await
    }

    /// `ask(message) { |chunk| ... }`.
    pub async fn ask_stream(&mut self, message: impl Into<String>, on_chunk: impl FnMut(&Message) + Send) -> Result<Message> {
        self.ask_later(message)?;
        self.complete_stream(on_chunk).await
    }

    /// `say` is an alias of `ask`.
    pub async fn say(&mut self, message: impl Into<String>) -> Result<Message> {
        self.ask(message).await
    }

    /// `ask_later`: stage without requesting a completion.
    pub fn ask_later(&mut self, message: impl Into<String>) -> Result<&mut Self> {
        self.ask_later_with(message, Vec::new())
    }

    pub fn ask_later_with(&mut self, message: impl Into<String>, attachments: Vec<Attachment>) -> Result<&mut Self> {
        self.raise_if_pending_tool_calls()?;
        self.messages.push(Message::user(message).with_attachments(attachments));
        Ok(self)
    }

    /// `complete`: run until `complete?` or parked on an approval.
    pub async fn complete(&mut self) -> Result<Message> {
        while !self.is_complete() && !self.waiting() {
            self.step_inner(None).await?;
        }
        Ok(self.last_non_system_message().or_else(|| self.messages.last()).cloned().unwrap_or_else(Message::chunk))
    }

    pub async fn complete_stream(&mut self, mut on_chunk: impl FnMut(&Message) + Send) -> Result<Message> {
        while !self.is_complete() && !self.waiting() {
            self.step_inner(Some(&mut on_chunk)).await?;
        }
        Ok(self.last_non_system_message().or_else(|| self.messages.last()).cloned().unwrap_or_else(Message::chunk))
    }

    /// `step`: run pending tools, or generate the next response. `None` once there is nothing to do.
    pub async fn step(&mut self) -> Result<Option<Message>> {
        self.step_inner(None).await
    }

    async fn step_inner(&mut self, on_chunk: Option<&mut (dyn FnMut(&Message) + Send)>) -> Result<Option<Message>> {
        if self.is_complete() {
            return Ok(None);
        }
        self.raise_if_cancelled()?;
        if self.pending_tool_response().is_none() {
            return self.generate_inner(on_chunk).await.map(Some);
        }
        let before = self.messages.len();
        self.run_tools().await?;
        Ok((self.messages.len() > before).then(|| self.messages.last().cloned()).flatten())
    }

    /// `complete?`: nothing staged, or the model answered without requesting tools.
    pub fn is_complete(&self) -> bool {
        match self.last_non_system_message() {
            None => true,
            Some(m) => match m.role {
                Role::User | Role::Tool => false,
                _ => !m.is_tool_call(),
            },
        }
    }

    /// `generate`: one completion, honoring fallbacks. Tool calls are not executed.
    pub async fn generate(&mut self) -> Result<Message> {
        self.generate_inner(None).await
    }

    async fn generate_inner(&mut self, mut on_chunk: Option<&mut (dyn FnMut(&Message) + Send)>) -> Result<Message> {
        self.raise_if_cancelled()?;
        if self.fallbacks.is_empty() {
            return self.generate_once(on_chunk, &mut false).await;
        }
        let original = (self.model.clone(), self.provider, self.protocol, self.connection.clone());
        let mut queue: std::collections::VecDeque<Fallback> = self.fallbacks.iter().cloned().collect();
        let mut attempt = 0;
        let mut active: Option<FallbackAttempt> = None;
        let result = loop {
            let mut chunks_yielded = false;
            let streaming = on_chunk.is_some();
            let result = self.generate_once(on_chunk.as_mut().map(|f| &mut **f as &mut (dyn FnMut(&Message) + Send)), &mut chunks_yielded).await;
            if let Some(mut a) = active.take() {
                a.succeeded = Some(result.is_ok());
                for cb in &mut self.callbacks.after_fallback {
                    cb(&a);
                }
            }
            match result {
                Ok(message) => break Ok(message),
                Err(e) if self.fallback_errors.contains(&e.kind()) => {
                    let Some(next) = queue.pop_front() else { break Err(e) };
                    attempt += 1;
                    let from = self.model.id.clone();
                    let from_provider = self.provider;
                    if let Err(switch_err) = self.switch_model(&next.model, next.provider.as_deref(), false) {
                        break Err(switch_err);
                    }
                    if self.provider != from_provider {
                        self.protocol = None;
                    }
                    let a = FallbackAttempt {
                        attempt,
                        error: e.to_string(),
                        from,
                        to: self.model.id.clone(),
                        streaming,
                        chunks_yielded,
                        succeeded: None,
                    };
                    for cb in &mut self.callbacks.before_fallback {
                        cb(&a);
                    }
                    active = Some(a);
                }
                Err(e) => break Err(e),
            }
        };
        (self.model, self.provider, self.protocol, self.connection) = original;
        result
    }

    fn record_usage(&mut self, entry: UsageEntry) {
        if let Some(r) = &mut self.usage_recorder {
            r(&entry);
        }
        self.usage_entries.push(entry);
    }

    fn preprocessed_messages(&self) -> Vec<Message> {
        self.messages
            .iter()
            .map(|m| {
                // A thinking signature is opaque to every provider but the one that issued it.
                let carries = m.thinking.is_some()
                    || m.raw_reasoning.is_some()
                    || m.tool_calls.iter().flat_map(|c| c.values()).any(|c| c.thought_signature.is_some());
                let producer = m.usage_entries.iter().rev().find(|e| e.status == UsageStatus::Succeeded).map(|e| e.provider.as_str());
                if m.role == Role::Assistant && carries && producer.is_some_and(|p| p != self.provider.slug()) {
                    m.without_thinking()
                } else {
                    m.clone()
                }
            })
            .collect()
    }

    /// `Chat#render`: the payload that would be sent, without sending it.
    pub fn render(&self) -> Result<Value> {
        self.render_with(&self.preprocessed_messages(), false).map(|(p, _)| p)
    }

    fn render_with(&self, messages: &[Message], stream: bool) -> Result<(Value, ProtocolName)> {
        let protocol = self.provider.resolve_protocol(self.protocol, &self.model, &self.config)?;
        let thinking = match &self.thinking {
            Some(t) => t.resolve(&self.model)?,
            None => None,
        };
        let request = Request {
            provider: self.provider,
            config: &self.config,
            model: &self.model,
            messages,
            tools: &self.tools,
            tool_prefs: &self.tool_prefs,
            temperature: self.temperature,
            max_output_tokens: self.max_output_tokens,
            schema: self.schema.as_ref(),
            thinking: thinking.as_ref(),
            stream,
        };
        let mut payload = protocols::render(protocol, &request)?;
        protocols::deep_merge(&mut payload, &self.provider_options);
        Ok((payload, protocol))
    }

    async fn generate_once(
        &mut self,
        on_chunk: Option<&mut (dyn FnMut(&Message) + Send)>,
        chunks_yielded: &mut bool,
    ) -> Result<Message> {
        self.raise_if_cancelled()?;
        let mut messages = self.preprocessed_messages();
        for m in &mut messages {
            for a in &mut m.attachments {
                a.load(self.connection.client()).await?;
            }
        }
        let streaming = on_chunk.is_some();
        let (mut payload, protocol) = self.render_with(&messages, streaming)?;
        for cb in &mut self.callbacks.before_request {
            cb(&mut payload);
        }
        let endpoint = protocols::endpoint(protocol, self.provider, &self.model, streaming);
        let mut headers = endpoint.headers;
        headers.extend(self.headers.iter().cloned());

        let mut attempts = 0usize;
        let mut on_attempt = || attempts += 1;
        let result = if let Some(on_chunk) = on_chunk {
            for cb in &mut self.callbacks.before_message {
                cb();
            }
            let mut acc = StreamAccumulator::default();
            let mut state = StreamState::default();
            let provider = self.provider;
            let cancelled = self.cancelled.clone();
            let mut on_event = |_event: crate::transport::SseEvent, data: Value| -> Result<()> {
                let chunk = protocols::build_chunk(protocol, provider, &mut state, &data)?;
                acc.add(&chunk);
                *chunks_yielded = true;
                on_chunk(&chunk);
                if cancelled.load(Ordering::SeqCst) {
                    return Err(Error::Cancelled);
                }
                Ok(())
            };
            let status = protocols::streaming_error_status(protocol);
            match self.connection.stream(&endpoint.path, &payload, &headers, &mut on_attempt, &mut on_event, status).await {
                Ok(raw) => acc.into_message(raw),
                Err(e) => Err(e),
            }
        } else {
            match self.connection.post(&endpoint.path, &payload, &headers, &mut on_attempt).await {
                Ok(raw) => protocols::parse_completion(protocol, self.provider, &self.model, raw),
                Err(e) => Err(e),
            }
        };

        // Usage ledger: one entry per HTTP attempt, like Accounting::Usage::Tracker.
        let failed_attempts = attempts.saturating_sub(1);
        for _ in 0..failed_attempts {
            self.record_usage(self.entry(UsageStatus::Failed, Tokens::default(), None));
        }
        let mut message = match result {
            Ok(m) => m,
            Err(e) => {
                let status = if matches!(e, Error::Cancelled) { UsageStatus::Cancelled } else { UsageStatus::Failed };
                if attempts > 0 {
                    self.record_usage(self.entry(status, Tokens::default(), None));
                }
                return Err(e);
            }
        };
        self.raise_if_cancelled()?;
        let billed_model = message
            .model
            .as_deref()
            .and_then(|id| models::models().find(id, Some(self.provider.slug())).ok())
            .unwrap_or_else(|| self.model.clone());
        let entry = self.entry(UsageStatus::Succeeded, message.tokens.clone(), Some(&billed_model));
        self.record_usage(entry.clone());
        message.usage_entries = vec![entry];
        message.model_info = Some(billed_model);
        if !streaming {
            for cb in &mut self.callbacks.before_message {
                cb();
            }
        }
        self.messages.push(message.clone());
        for cb in &mut self.callbacks.after_message {
            cb(&message);
        }
        Ok(message)
    }

    fn entry(&self, status: UsageStatus, tokens: Tokens, model: Option<&Model>) -> UsageEntry {
        let model = model.unwrap_or(&self.model);
        let cost = Cost::new(&tokens, Some(model), Tier::Standard);
        UsageEntry { operation: Operation::Chat, provider: self.provider.slug().into(), model: self.model.id.clone(), status, tokens, cost }
    }

    // ---- tools -----------------------------------------------------------------------------

    fn pending_tool_response(&self) -> Option<&Message> {
        let response = self.messages.iter().rev().find(|m| m.role != Role::System && !m.is_tool_result())?;
        (response.is_tool_call() && !self.pending_tool_calls(response).is_empty()).then_some(response)
    }

    fn pending_tool_calls(&self, response: &Message) -> Vec<ToolCall> {
        let answered: Vec<&str> = self.messages.iter().filter_map(|m| m.tool_call_id.as_deref()).collect();
        response
            .tool_calls
            .iter()
            .flat_map(|c| c.values())
            .filter(|c| !answered.contains(&c.id.as_str()))
            .cloned()
            .collect()
    }

    fn find_tool(&self, name: &str) -> Option<SharedTool> {
        self.tools.iter().find(|t| t.name() == name).cloned()
    }

    fn approval_pending(&self, call: &ToolCall) -> bool {
        if call.remote {
            return !self.tool_call_decisions.contains_key(&call.id);
        }
        let Some(tool) = self.find_tool(&call.name) else { return false };
        tool.requires_approval() && !self.tool_call_decisions.contains_key(&call.id)
    }

    /// Execute the pending tool calls of the latest response (`run_tools`).
    pub async fn run_tools(&mut self) -> Result<&mut Self> {
        self.raise_if_cancelled()?;
        let Some(response) = self.pending_tool_response().cloned() else { return Ok(self) };
        for call in self.pending_tool_calls(&response) {
            self.raise_if_cancelled()?;
            if call.remote {
                // Remote (provider-executed) approvals are answered on the next request.
                if let Some(&approved) = self.tool_call_decisions.get(&call.id) {
                    let mut m = Message::tool_result(call.id.clone(), if approved { "Approved" } else { "Denied" });
                    m.raw_content = Some(serde_json::json!([{
                        "type": "mcp_approval_response", "approval_request_id": call.id, "approve": approved
                    }]));
                    self.push_tool_message(m);
                }
                continue;
            }
            let tool = self.find_tool(&call.name);
            let decision = match &tool {
                Some(t) if t.requires_approval() => self.tool_call_decisions.get(&call.id).copied(),
                _ => Some(self.tool_call_decisions.get(&call.id).copied() != Some(false)),
            };
            match decision {
                None => continue,
                Some(false) => {
                    let denied = ToolResult::error(format!("The user denied the {} tool call.", call.name));
                    self.push_tool_result(&call, denied);
                }
                Some(true) => {
                    for cb in &mut self.callbacks.before_tool_call {
                        cb(&call);
                    }
                    let result = self.execute_tool(tool.as_deref(), &call).await?;
                    self.raise_if_cancelled()?;
                    for cb in &mut self.callbacks.after_tool_result {
                        cb(&result);
                    }
                    self.push_tool_result(&call, result);
                }
            }
        }
        if matches!(self.tool_prefs.choice, Some(ToolChoice::Required | ToolChoice::Tool(_))) {
            self.tool_prefs.choice = None;
        }
        Ok(self)
    }

    async fn execute_tool(&self, tool: Option<&dyn Tool>, call: &ToolCall) -> Result<ToolResult> {
        let Some(tool) = tool else {
            let names: Vec<String> = self.tools.iter().map(|t| t.name()).collect();
            return Ok(ToolResult::error(format!(
                "Model tried to call unavailable tool `{}`. Available tools: {}.",
                call.name,
                serde_json::to_string(&names).unwrap()
            )));
        };
        let arguments = call.arguments();
        if let Some(problem) = validate_arguments(tool, &arguments) {
            return Ok(ToolResult::error(format!("Invalid tool arguments: {problem}")));
        }
        tool.execute(arguments, call).await.map_err(|e| Error::Tool(e.to_string()))
    }

    fn push_tool_result(&mut self, call: &ToolCall, result: ToolResult) {
        let mut m = Message::tool_result(call.id.clone(), result.content);
        m.attachments = result.attachments;
        self.push_tool_message(m);
    }

    fn push_tool_message(&mut self, m: Message) {
        for cb in &mut self.callbacks.before_message {
            cb();
        }
        self.messages.push(m.clone());
        for cb in &mut self.callbacks.after_message {
            cb(&m);
        }
    }

    /// `approve(tool_call)`: the next `complete` executes it.
    pub fn approve(&mut self, tool_call_id: &str) -> &mut Self {
        self.tool_call_decisions.insert(tool_call_id.into(), true);
        self
    }

    /// `deny(tool_call)`: the next `complete` answers it with a structured denial.
    pub fn deny(&mut self, tool_call_id: &str) -> &mut Self {
        self.tool_call_decisions.insert(tool_call_id.into(), false);
        self
    }

    /// Restores decisions recorded earlier, e.g. from persisted tool-call rows.
    pub fn set_decisions(&mut self, decisions: impl IntoIterator<Item = (String, bool)>) {
        self.tool_call_decisions.extend(decisions);
    }

    fn waiting(&self) -> bool {
        let Some(response) = self.pending_tool_response() else { return false };
        let pending = self.pending_tool_calls(response);
        !pending.is_empty() && pending.iter().all(|c| self.approval_pending(c))
    }

    /// `awaiting_approval?`.
    pub fn is_awaiting_approval(&self) -> bool {
        self.waiting() && !self.pending_approvals().is_empty()
    }

    /// `pending_approvals`.
    pub fn pending_approvals(&self) -> Vec<ToolCall> {
        let Some(response) = self.pending_tool_response() else { return Vec::new() };
        self.pending_tool_calls(response).into_iter().filter(|c| self.approval_pending(c)).collect()
    }

    fn raise_if_pending_tool_calls(&self) -> Result<()> {
        let Some(response) = self.pending_tool_response() else { return Ok(()) };
        let mut names: Vec<String> = self.pending_tool_calls(response).into_iter().map(|c| c.name).collect();
        names.dedup();
        Err(Error::PendingToolCalls(format!(
            "The last response has unanswered tool calls ({}). Run complete, recording approve or deny \
             decisions for calls that require approval and answering pending inputs, before asking again.",
            names.join(", ")
        )))
    }

    fn last_non_system_message(&self) -> Option<&Message> {
        self.messages.iter().rev().find(|m| m.role != Role::System)
    }

    // ---- cancellation ----------------------------------------------------------------------

    /// `cancel`: stop at the next checkpoint with `Error::Cancelled`.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    pub fn cancel_handle(&self) -> CancelHandle {
        CancelHandle(self.cancelled.clone())
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    fn raise_if_cancelled(&self) -> Result<()> {
        if self.cancelled.swap(false, Ordering::SeqCst) {
            return Err(Error::Cancelled);
        }
        Ok(())
    }
}

/// `Chat#normalize_schema_payload`.
fn normalize_schema(raw: Value) -> Option<Schema> {
    if raw.is_null() {
        return None;
    }
    let obj = raw.as_object().cloned().unwrap_or_default();
    let mut definition = obj.get("schema").cloned().unwrap_or_else(|| raw.clone());
    if let Some(d) = definition.as_object_mut() {
        d.remove("$schema");
        d.remove("title");
    }
    let strict = match obj.get("strict").and_then(Value::as_bool) {
        Some(s) => Some(s),
        None => definition.as_object_mut().and_then(|d| d.remove("strict")).and_then(|v| v.as_bool()),
    };
    let name = obj
        .get("name")
        .or_else(|| obj.get("title"))
        .and_then(Value::as_str)
        .unwrap_or("response")
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
        .collect::<String>();
    Some(Schema {
        name: if name.is_empty() { "response".into() } else { name },
        schema: definition,
        strict,
        description: obj.get("description").and_then(Value::as_str).map(str::to_string),
    })
}
