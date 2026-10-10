//! Port of `message.rb`, `chunk.rb`, `tool_call.rb`, `thinking.rb`, `citation.rb`,
//! `server_tool_call.rb`, and the usage `Entry` from `accounting/usage.rb`.

use indexmap_lite::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::attachment::Attachment;
use crate::cost::{Cost, Tier};
use crate::error::{Error, Result};
use crate::model::Model;
use crate::tokens::Tokens;

/// Insertion-ordered map keyed by tool call id, like the Ruby Hash RubyLLM uses.
pub mod indexmap_lite {
    #[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
    pub struct IndexMap<V>(pub Vec<(String, V)>);

    impl<V> Default for IndexMap<V> {
        fn default() -> Self {
            IndexMap(Vec::new())
        }
    }

    impl<V> IndexMap<V> {
        pub fn new() -> Self {
            IndexMap(Vec::new())
        }
        pub fn get(&self, key: &str) -> Option<&V> {
            self.0.iter().find(|(k, _)| k == key).map(|(_, v)| v)
        }
        pub fn get_mut(&mut self, key: &str) -> Option<&mut V> {
            self.0.iter_mut().find(|(k, _)| k == key).map(|(_, v)| v)
        }
        pub fn insert(&mut self, key: String, value: V) {
            match self.get_mut(&key) {
                Some(slot) => *slot = value,
                None => self.0.push((key, value)),
            }
        }
        pub fn contains_key(&self, key: &str) -> bool {
            self.get(key).is_some()
        }
        pub fn values(&self) -> impl Iterator<Item = &V> {
            self.0.iter().map(|(_, v)| v)
        }
        pub fn keys(&self) -> impl Iterator<Item = &String> {
            self.0.iter().map(|(k, _)| k)
        }
        pub fn iter(&self) -> impl Iterator<Item = (&String, &V)> {
            self.0.iter().map(|(k, v)| (k, v))
        }
        pub fn len(&self) -> usize {
            self.0.len()
        }
        pub fn is_empty(&self) -> bool {
            self.0.is_empty()
        }
    }

    impl<V> FromIterator<(String, V)> for IndexMap<V> {
        fn from_iter<I: IntoIterator<Item = (String, V)>>(iter: I) -> Self {
            let mut map = IndexMap::new();
            for (k, v) in iter {
                map.insert(k, v);
            }
            map
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

impl Role {
    pub fn as_str(&self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }

    pub fn parse(s: &str) -> Result<Role> {
        match s {
            "system" => Ok(Role::System),
            "user" => Ok(Role::User),
            "assistant" => Ok(Role::Assistant),
            "tool" => Ok(Role::Tool),
            _ => Err(Error::Argument(
                "Expected role to be one of: system, user, assistant, tool".into(),
            )),
        }
    }
}

/// Normalized finish reasons (`FINISH_REASONS` in each protocol). Unknown reasons are kept verbatim.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    MaxTokens,
    ToolCalls,
    ContentFilter,
    PauseTurn,
    Other(String),
}

impl FinishReason {
    pub fn from_symbol(s: &str) -> FinishReason {
        match s {
            "stop" => FinishReason::Stop,
            "max_tokens" => FinishReason::MaxTokens,
            "tool_calls" => FinishReason::ToolCalls,
            "content_filter" => FinishReason::ContentFilter,
            "pause_turn" => FinishReason::PauseTurn,
            other => FinishReason::Other(other.to_string()),
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::MaxTokens => "max_tokens",
            FinishReason::ToolCalls => "tool_calls",
            FinishReason::ContentFilter => "content_filter",
            FinishReason::PauseTurn => "pause_turn",
            FinishReason::Other(s) => s,
        }
    }
}

/// Model deliberation (`Thinking`). The signature is opaque provider state that must be replayed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Thinking {
    pub text: Option<String>,
    pub signature: Option<String>,
}

impl Thinking {
    /// `Thinking.build`: `None` when there is neither text nor a signature.
    pub fn build(text: Option<String>, signature: Option<String>) -> Option<Thinking> {
        let signature = signature.filter(|s| !s.is_empty());
        let text = text.filter(|t| !(t.is_empty() && signature.is_none()));
        if text.is_none() && signature.is_none() {
            return None;
        }
        Some(Thinking { text, signature })
    }
}

/// Arguments the model passed to a tool. While streaming they arrive as a JSON string in pieces.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolArguments {
    Parsed(Map<String, Value>),
    Partial(String),
}

impl Default for ToolArguments {
    fn default() -> Self {
        ToolArguments::Parsed(Map::new())
    }
}

impl ToolArguments {
    /// The arguments as a map, keys ordered as [`canonical_arguments`] does. A streamed string is
    /// parsed the way the finished call Ruby builds from it would be.
    pub fn as_map(&self) -> Map<String, Value> {
        let map = match self {
            ToolArguments::Parsed(m) => m.clone(),
            ToolArguments::Partial(s) => serde_json::from_str(s).unwrap_or_default(),
        };
        canonical_map(map)
    }
}

/// `ToolCall#canonical_arguments`: object keys ordered the way PostgreSQL `jsonb` and MySQL
/// `json` store them (shorter keys first, then bytewise), recursively, so a call reloaded from
/// the database renders the same request bytes and keeps the provider's prompt cache.
pub fn canonical_arguments(value: Value) -> Value {
    match value {
        Value::Object(map) => Value::Object(canonical_map(map)),
        Value::Array(items) => Value::Array(items.into_iter().map(canonical_arguments).collect()),
        other => other,
    }
}

fn canonical_map(map: Map<String, Value>) -> Map<String, Value> {
    let mut entries: Vec<(String, Value)> = map.into_iter().collect();
    entries.sort_by(|(a, _), (b, _)| (a.len(), a.as_bytes()).cmp(&(b.len(), b.as_bytes())));
    entries
        .into_iter()
        .map(|(k, v)| (k, canonical_arguments(v)))
        .collect()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: ToolArguments,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
    #[serde(default)]
    pub remote: bool,
    /// Streaming only: whether this piece carried an id field at all. RubyLLM starts a new call
    /// for any non-nil id (an empty one gets a UUID) and treats a nil id as an argument fragment.
    #[serde(skip)]
    pub(crate) starts: bool,
}

impl ToolCall {
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: Map<String, Value>,
    ) -> Self {
        ToolCall {
            id: id.into(),
            name: name.into(),
            arguments: ToolArguments::Parsed(canonical_map(arguments)),
            thought_signature: None,
            remote: false,
            starts: true,
        }
    }

    /// A streamed piece that opens a call (`id` present, possibly empty): Ruby's
    /// `ToolCall.new(id: '', name:, arguments: '')` as a stream chunk carries it.
    pub fn opening(id: String, name: String, arguments: String) -> ToolCall {
        ToolCall {
            id,
            name,
            arguments: ToolArguments::Partial(arguments),
            thought_signature: None,
            remote: false,
            starts: true,
        }
    }

    /// A streamed argument fragment for a call opened earlier: Ruby's
    /// `ToolCall.new(id: nil, name: nil, arguments: '...')`. A `nil` fragment is an empty one.
    pub fn fragment(arguments: String) -> ToolCall {
        ToolCall {
            id: String::new(),
            name: String::new(),
            arguments: ToolArguments::Partial(arguments),
            thought_signature: None,
            remote: false,
            starts: false,
        }
    }

    pub(crate) fn starts_call(&self) -> bool {
        self.starts
    }

    pub fn arguments(&self) -> Map<String, Value> {
        self.arguments.as_map()
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Citation {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cited_text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start_index: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end_index: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_index: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start_page: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end_page: Option<i64>,
}

/// A tool the provider ran itself (web search, code execution, ...).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ServerToolCall {
    #[serde(rename = "type")]
    pub kind: String,
    pub name: Option<String>,
    pub id: Option<String>,
    pub input: Option<Value>,
    pub result: Option<Value>,
    pub raw: Value,
    /// `#search_suggestions`: the HTML of the search suggestions the provider requires shown with
    /// a grounded answer (Google's for Google Search grounding). Only the live response carries
    /// them: `to_h` leaves them out, so a persisted chat never stores them.
    #[serde(skip)]
    pub search_suggestions: Option<String>,
}

/// The raw HTTP exchange behind a response (`message.raw`): status, headers, and body. Like
/// RubyLLM 2.1's `release_request`, provider calls leave `request_body` empty: each request of a
/// chat holds the whole conversation so far, so keeping one per reply grew a chat's memory
/// quadratically. Read what a chat sends with `render` or a `before_request` hook instead.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RawResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Value,
    pub request_body: std::sync::Arc<str>,
}

impl RawResponse {
    /// The request as JSON (`Null` when there was no JSON body, e.g. a GET or multipart upload).
    pub fn request_body_json(&self) -> Value {
        serde_json::from_str(&self.request_body).unwrap_or(Value::Null)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Operation {
    Chat,
    Embedding,
    Moderation,
    Image,
    Speech,
    Transcription,
    Ocr,
    Rerank,
    Judgment,
    /// A finished `VideoJob` (RubyLLM 2.1 `video_job.rb#record_usage`).
    Video,
    /// A finished research job (2.1 `Accounting::Usage::OPERATIONS`).
    Research,
}

impl Operation {
    pub fn as_str(&self) -> &'static str {
        match self {
            Operation::Chat => "chat",
            Operation::Embedding => "embedding",
            Operation::Moderation => "moderation",
            Operation::Image => "image",
            Operation::Speech => "speech",
            Operation::Transcription => "transcription",
            Operation::Ocr => "ocr",
            Operation::Rerank => "rerank",
            Operation::Judgment => "judgment",
            Operation::Video => "video",
            Operation::Research => "research",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UsageStatus {
    Pending,
    Succeeded,
    Failed,
    Cancelled,
}

impl UsageStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            UsageStatus::Pending => "pending",
            UsageStatus::Succeeded => "succeeded",
            UsageStatus::Failed => "failed",
            UsageStatus::Cancelled => "cancelled",
        }
    }
}

/// `Accounting::Usage::Entry`: one billed attempt. Retries and fallbacks each get their own.
/// `id` gives each attempt the identity Ruby gets from object identity, so persistence links the
/// exact entry even when two attempts carry identical numbers.
#[derive(Debug, Clone, PartialEq)]
pub struct UsageEntry {
    pub id: u64,
    pub operation: Operation,
    pub provider: String,
    pub model: String,
    pub status: UsageStatus,
    pub tokens: Tokens,
    pub cost: Cost,
    /// `owner`: who the attempt is attributed to, from the operation's `owner:` or the enclosing
    /// [`crate::accounting::with_usage_owner`] when the attempt started.
    pub owner: Option<crate::accounting::UsageOwner>,
}

static NEXT_USAGE_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

impl UsageEntry {
    /// A fresh process-unique id for a new entry.
    pub fn next_id() -> u64 {
        NEXT_USAGE_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }

    pub fn cost_available(&self) -> bool {
        self.cost.total().is_some()
    }
}

/// One message in a conversation (`RubyLLM::Message`). Streamed pieces use the same type
/// (`Chunk < Message`).
#[derive(Debug, Clone, PartialEq)]
pub struct Message {
    pub role: Role,
    pub content: Option<String>,
    pub attachments: Vec<Attachment>,
    pub model: Option<String>,
    pub tool_calls: Option<IndexMap<ToolCall>>,
    pub tool_call_id: Option<String>,
    pub tokens: Tokens,
    pub thinking: Option<Thinking>,
    pub citations: Vec<Citation>,
    pub server_tool_calls: Vec<ServerToolCall>,
    pub raw_content: Option<Value>,
    pub raw_reasoning: Option<Value>,
    pub finish_reason: Option<FinishReason>,
    pub raw: Option<RawResponse>,
    pub cache_until_here: bool,
    /// `#cache_ttl`: this boundary's own cache lifetime (`"1h"`), or `None` for the chat's.
    pub cache_ttl: Option<String>,
    pub usage_entries: Vec<UsageEntry>,
    /// `mcp_result`: the result of an MCP tool with a UI, kept on its tool result message so
    /// your app can render the UI again.
    pub mcp_result: Option<Box<crate::mcp::McpResult>>,
    /// Primary key of the row this message is stored as, set by a persistence layer
    /// (`rust_llm_loco`). `None` means not yet persisted.
    pub record_id: Option<i64>,
    pub(crate) model_info: Option<Model>,
    /// `@supplied_cost`: an explicit `cost:` the message was built with (see [`Message::with_cost`]).
    pub(crate) supplied_cost: Option<Cost>,
}

pub type Chunk = Message;

impl Message {
    pub fn new(role: Role, content: impl Into<Option<String>>) -> Message {
        Message {
            role,
            content: content.into(),
            attachments: Vec::new(),
            model: None,
            tool_calls: None,
            tool_call_id: None,
            tokens: Tokens::default(),
            thinking: None,
            citations: Vec::new(),
            server_tool_calls: Vec::new(),
            raw_content: None,
            raw_reasoning: None,
            finish_reason: None,
            raw: None,
            cache_until_here: false,
            cache_ttl: None,
            usage_entries: Vec::new(),
            mcp_result: None,
            record_id: None,
            model_info: None,
            supplied_cost: None,
        }
    }

    /// `Message#cache_until_here(ttl:)`: marks this message as a prompt cache boundary, with its
    /// own lifetime when `ttl` is given.
    pub fn with_cache_until_here(mut self, ttl: Option<&str>) -> Message {
        self.cache_until_here = true;
        self.cache_ttl = ttl.map(str::to_string);
        self
    }

    /// `Message.cache_boundary_options`: `true` (no lifetime) or `{ ttl: }`; `false`/`nil` is
    /// no boundary. Anything else is an `ArgumentError`.
    pub fn cache_boundary_options(value: &Value) -> Result<Option<Option<String>>> {
        match value {
            Value::Null | Value::Bool(false) => return Ok(None),
            Value::Bool(true) => return Ok(Some(None)),
            Value::Object(o) if o.keys().all(|k| k == "ttl") => {
                return Ok(Some(
                    o.get("ttl").and_then(Value::as_str).map(str::to_string),
                ));
            }
            _ => {}
        }
        Err(Error::Argument(format!(
            "cache_until_here accepts true, false, or ttl:, got {value}"
        )))
    }

    pub fn system(content: impl Into<String>) -> Message {
        Message::new(Role::System, Some(content.into()))
    }

    pub fn user(content: impl Into<String>) -> Message {
        Message::new(Role::User, Some(content.into()))
    }

    pub fn assistant(content: impl Into<String>) -> Message {
        Message::new(Role::Assistant, Some(content.into()))
    }

    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Message {
        let mut m = Message::new(Role::Tool, Some(content.into()));
        m.tool_call_id = Some(tool_call_id.into());
        m
    }

    pub(crate) fn chunk() -> Message {
        Message::new(Role::Assistant, None)
    }

    /// `normalize_content`: an assistant turn that only calls tools has content `""`.
    pub(crate) fn normalized(mut self) -> Message {
        if self.role == Role::Assistant && self.content.is_none() && self.is_tool_call() {
            self.content = Some(String::new());
        }
        self
    }

    pub fn with_attachments(mut self, attachments: Vec<Attachment>) -> Message {
        self.attachments = attachments;
        self
    }

    pub fn content(&self) -> &str {
        self.content.as_deref().unwrap_or("")
    }

    /// `Message#parsed`: the content as JSON, for structured output.
    pub fn parsed(&self) -> Result<Option<Value>> {
        match self.content.as_deref() {
            None | Some("") => Ok(None),
            Some(s) => Ok(Some(serde_json::from_str(s)?)),
        }
    }

    pub fn is_tool_call(&self) -> bool {
        self.tool_calls.as_ref().is_some_and(|t| !t.is_empty())
    }

    pub fn is_tool_result(&self) -> bool {
        self.tool_call_id.as_ref().is_some_and(|id| !id.is_empty())
    }

    pub fn is_stopped(&self) -> bool {
        self.finish_reason == Some(FinishReason::Stop) && !self.is_tool_call()
    }

    pub fn is_max_tokens(&self) -> bool {
        self.finish_reason == Some(FinishReason::MaxTokens)
    }

    pub fn is_tool_call_stop(&self) -> bool {
        self.finish_reason == Some(FinishReason::ToolCalls)
            || (self.is_tool_call() && self.finish_reason == Some(FinishReason::Stop))
    }

    pub fn is_content_filtered(&self) -> bool {
        self.finish_reason == Some(FinishReason::ContentFilter)
    }

    /// `Message#tokens`: the usage ledger when present, otherwise what the message was built with.
    pub fn tokens(&self) -> Tokens {
        if self.usage_entries.is_empty() {
            self.tokens.clone()
        } else {
            Tokens::aggregate(self.usage_entries.iter().map(|e| &e.tokens))
        }
    }

    /// `Message#cost`: the recorded attempt costs, else an explicitly supplied `cost:`, else
    /// pricing from `model_info`. Pass a model to price against it instead.
    pub fn cost(&self, model: Option<&Model>) -> Cost {
        if model.is_none() && !self.usage_entries.is_empty() {
            let complete = self.usage_entries.iter().all(UsageEntry::cost_available);
            return Cost::aggregate(self.usage_entries.iter().map(|e| &e.cost), complete);
        }
        if let (None, Some(supplied)) = (model, &self.supplied_cost) {
            return supplied.clone();
        }
        let info = model.cloned().or_else(|| self.model_info());
        Cost::new(&self.tokens(), info.as_ref(), Tier::Standard)
    }

    /// Registry entry of the model that produced this message, if known.
    pub fn model_info(&self) -> Option<Model> {
        if let Some(m) = &self.model_info {
            return Some(m.clone());
        }
        let registry = crate::models::models();
        let entry = self
            .usage_entries
            .iter()
            .rev()
            .find(|e| e.status == UsageStatus::Succeeded);
        match entry {
            Some(e) => registry.find(&e.model, Some(&e.provider)).ok(),
            None => self
                .model
                .as_deref()
                .and_then(|m| registry.find(m, None).ok()),
        }
    }

    /// The per-request view of a stored message: everything a protocol renders, without the
    /// bookkeeping it never reads (the HTTP exchange, registry entry). Cloning those for every
    /// message on every request made long chats quadratic. Of the usage ledger only the
    /// producing entry stays: it is what names the producer (`Protocol#producing_entry`).
    pub(crate) fn for_request(&self) -> Message {
        Message {
            role: self.role,
            content: self.content.clone(),
            attachments: self.attachments.clone(),
            model: self.model.clone(),
            tool_calls: self.tool_calls.clone(),
            tool_call_id: self.tool_call_id.clone(),
            tokens: Tokens::default(),
            thinking: self.thinking.clone(),
            citations: self.citations.clone(),
            server_tool_calls: self.server_tool_calls.clone(),
            raw_content: self.raw_content.clone(),
            raw_reasoning: self.raw_reasoning.clone(),
            finish_reason: self.finish_reason.clone(),
            raw: None,
            cache_until_here: self.cache_until_here,
            cache_ttl: self.cache_ttl.clone(),
            usage_entries: self.producing_entry().cloned().into_iter().collect(),
            mcp_result: None,
            record_id: self.record_id,
            model_info: None,
            supplied_cost: None,
        }
    }

    /// `Protocol#producing_entry`: the last succeeded usage entry, the one record of which
    /// provider and model produced this message.
    pub(crate) fn producing_entry(&self) -> Option<&UsageEntry> {
        self.usage_entries
            .iter()
            .rev()
            .find(|e| e.status == UsageStatus::Succeeded)
    }

    /// `Protocol#own_signature`: the thinking signature, only when `provider` produced it. A
    /// signature can be another provider's opaque blob, and a message without usage could come
    /// from anyone.
    pub(crate) fn own_signature(&self, provider: &str) -> Option<&str> {
        let signature = self.thinking.as_ref()?.signature.as_deref()?;
        (self.producing_entry()?.provider == provider).then_some(signature)
    }

    /// `Message#without_native_content(raw_content:)`: what gets replayed to a model that did
    /// not produce the thinking, raw reasoning, raw content and call signatures.
    pub(crate) fn without_native_content(&self, raw_content: Option<Value>) -> Message {
        let mut m = self.clone();
        m.thinking = None;
        m.raw_reasoning = None;
        m.raw_content = raw_content;
        if let Some(calls) = &mut m.tool_calls {
            for (_, call) in calls.0.iter_mut() {
                call.thought_signature = None;
            }
        }
        m
    }

    /// `Message.new(cost:)`: an explicitly supplied cost, kept through `to_h`/`from_h`.
    /// Recorded attempt costs still win over it, and an explicit model reprices.
    pub fn with_cost(mut self, cost: Cost) -> Message {
        self.supplied_cost = Some(cost);
        self
    }

    /// `Message#tool_results`: the tool result messages in `conversation` answering this
    /// message's tool calls, or none when it made no calls. Ruby reads the chat through the
    /// message's `conversation` back-link; pass the chat's messages (`chat.messages()`).
    pub fn tool_results<'a>(&self, conversation: &'a [Message]) -> Vec<&'a Message> {
        let Some(calls) = self.tool_calls.as_ref().filter(|c| !c.is_empty()) else {
            return Vec::new();
        };
        conversation
            .iter()
            .filter(|m| {
                m.is_tool_result()
                    && m.tool_call_id
                        .as_deref()
                        .is_some_and(|id| calls.contains_key(id))
            })
            .collect()
    }

    /// `Message.new(message.to_h)`: rebuilds a message from its `to_h` attributes (string keys,
    /// as after a JSON round-trip). Tool calls, thinking, citations, server tool calls, and the
    /// supplied cost come back as value objects. Attachments are rebuilt from their `source`.
    pub fn from_h(h: &Value) -> Result<Message> {
        let str_of = |key: &str| h.get(key).and_then(Value::as_str).map(str::to_string);
        let int_of = |key: &str| h.get(key).and_then(Value::as_i64);
        let role = Role::parse(h.get("role").and_then(Value::as_str).unwrap_or_default())?;
        let mut m = Message::new(role, str_of("content"));
        m.model = str_of("model");
        m.tool_call_id = str_of("tool_call_id");
        if let Some(calls) = h.get("tool_calls").and_then(Value::as_object) {
            m.tool_calls = Some(
                calls
                    .iter()
                    .map(|(id, call)| {
                        let mut tc = ToolCall::new(
                            call.get("id")
                                .and_then(Value::as_str)
                                .unwrap_or(id.as_str()),
                            call.get("name").and_then(Value::as_str).unwrap_or_default(),
                            call.get("arguments")
                                .and_then(Value::as_object)
                                .cloned()
                                .unwrap_or_default(),
                        );
                        tc.thought_signature = call
                            .get("thought_signature")
                            .and_then(Value::as_str)
                            .map(str::to_string);
                        tc.remote = call.get("remote").and_then(Value::as_bool).unwrap_or(false);
                        (id.clone(), tc)
                    })
                    .collect(),
            );
        }
        m.thinking = match h.get("thinking") {
            Some(Value::Object(t)) => Thinking::build(
                t.get("text").and_then(Value::as_str).map(str::to_string),
                t.get("signature")
                    .and_then(Value::as_str)
                    .map(str::to_string),
            ),
            Some(Value::String(text)) => {
                Thinking::build(Some(text.clone()), str_of("thinking_signature"))
            }
            // `coerce_thinking`: no thinking text still keeps a signature-only thinking.
            _ => Thinking::build(None, str_of("thinking_signature")),
        };
        if let Some(citations) = h.get("citations") {
            m.citations = serde_json::from_value(citations.clone())?;
        }
        for call in h
            .get("server_tool_calls")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            m.server_tool_calls.push(ServerToolCall {
                kind: call
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                name: call.get("name").and_then(Value::as_str).map(str::to_string),
                id: call.get("id").and_then(Value::as_str).map(str::to_string),
                input: call.get("input").cloned(),
                result: call.get("result").cloned(),
                raw: call.get("raw").cloned().unwrap_or(Value::Null),
                search_suggestions: None,
            });
        }
        for a in h
            .get("attachments")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            m.attachments.push(Attachment::from_h(a)?);
        }
        m.raw_content = h.get("raw_content").cloned();
        m.raw_reasoning = h.get("raw_reasoning").cloned();
        m.finish_reason = h
            .get("finish_reason")
            .and_then(Value::as_str)
            .map(FinishReason::from_symbol);
        m.cache_until_here = h
            .get("cache_until_here")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if m.cache_until_here {
            m.cache_ttl = h
                .get("cache_ttl")
                .and_then(Value::as_str)
                .map(str::to_string);
        }
        m.supplied_cost = h.get("cost").map(|c| Cost::from_h(c, None));
        // `coerce_mcp_result`: a dumped result comes back as an `MCP::Result`.
        m.mcp_result = h
            .get("mcp_result")
            .filter(|r| r.is_object())
            .map(|r| Box::new(crate::mcp::McpResult::load(r)));
        m.tokens = Tokens {
            input: int_of("input_tokens"),
            output: int_of("output_tokens"),
            cache_read: int_of("cache_read_tokens"),
            cache_write: int_of("cache_write_tokens"),
            thinking: int_of("thinking_tokens"),
            cache_write_by_ttl: h
                .get("cache_write_tokens_by_ttl")
                .and_then(Tokens::positive_counts),
            server_tool_use: h.get("server_tool_use").and_then(Tokens::positive_counts),
            reported_cost: h.get("reported_cost").and_then(Value::as_f64),
        };
        Ok(m.normalized())
    }

    /// `Message#to_h`: `nil` values and empty attachment, citation, and server tool call lists
    /// are omitted; `cost` appears only when one was supplied.
    pub fn to_h(&self) -> Value {
        let mut h = Map::new();
        h.insert("role".into(), self.role.as_str().into());
        if let Some(c) = &self.content {
            h.insert("content".into(), c.clone().into());
        }
        if !self.attachments.is_empty() {
            h.insert(
                "attachments".into(),
                self.attachments.iter().map(attachment_h).collect(),
            );
        }
        if let Some(m) = &self.model {
            h.insert("model".into(), m.clone().into());
        }
        if self.supplied_cost.is_some() {
            h.insert(
                "cost".into(),
                crate::instrumentation::cost_h(&self.cost(None)),
            );
        }
        // `reported_cost: tokens.reported_cost`: the provider's own price, summed over attempts.
        if let Some(reported) = self.tokens().reported_cost {
            h.insert("reported_cost".into(), reported.into());
        }
        if let Some(calls) = &self.tool_calls {
            let calls: Map<String, Value> = calls
                .iter()
                .map(|(k, v)| (k.clone(), tool_call_h(v)))
                .collect();
            h.insert("tool_calls".into(), calls.into());
        }
        if let Some(id) = &self.tool_call_id {
            h.insert("tool_call_id".into(), id.clone().into());
        }
        if let Some(t) = &self.thinking {
            if let Some(text) = &t.text {
                h.insert("thinking".into(), text.clone().into());
            }
            if let Some(sig) = &t.signature {
                h.insert("thinking_signature".into(), sig.clone().into());
            }
        }
        if !self.citations.is_empty() {
            h.insert(
                "citations".into(),
                serde_json::to_value(&self.citations).unwrap(),
            ); // Citation always serializes
        }
        if !self.server_tool_calls.is_empty() {
            h.insert(
                "server_tool_calls".into(),
                self.server_tool_calls
                    .iter()
                    .map(server_tool_call_h)
                    .collect(),
            );
        }
        if let Some(raw) = &self.raw_content {
            h.insert("raw_content".into(), raw.clone());
        }
        if let Some(raw) = &self.raw_reasoning {
            h.insert("raw_reasoning".into(), raw.clone());
        }
        if let Some(result) = &self.mcp_result {
            h.insert("mcp_result".into(), result.dump());
        }
        if let Some(r) = &self.finish_reason {
            h.insert("finish_reason".into(), r.as_str().into());
        }
        if self.cache_until_here {
            h.insert("cache_until_here".into(), true.into());
        }
        if let Some(ttl) = &self.cache_ttl {
            h.insert("cache_ttl".into(), ttl.clone().into());
        }
        let t = self.tokens();
        for (k, v) in [
            ("input_tokens", t.input),
            ("output_tokens", t.output),
            ("cache_read_tokens", t.cache_read),
            ("cache_write_tokens", t.cache_write),
            ("thinking_tokens", t.thinking),
        ] {
            if let Some(v) = v {
                h.insert(k.into(), v.into());
            }
        }
        if let Some(by_ttl) = &t.cache_write_by_ttl {
            h.insert(
                "cache_write_tokens_by_ttl".into(),
                Value::Object(by_ttl.clone()),
            );
        }
        if let Some(s) = &t.server_tool_use {
            h.insert("server_tool_use".into(), Value::Object(s.clone()));
        }
        Value::Object(h)
    }
}

/// `ToolCall#to_h`: `remote` only when true, `thought_signature` only when present.
fn tool_call_h(call: &ToolCall) -> Value {
    let mut h = Map::new();
    h.insert("id".into(), call.id.clone().into());
    h.insert("name".into(), call.name.clone().into());
    h.insert("arguments".into(), Value::Object(call.arguments()));
    if call.remote {
        h.insert("remote".into(), true.into());
    }
    if let Some(sig) = &call.thought_signature {
        h.insert("thought_signature".into(), sig.clone().into());
    }
    Value::Object(h)
}

/// `ServerToolCall#to_h`, omitting `nil` values.
fn server_tool_call_h(call: &ServerToolCall) -> Value {
    let mut h = Map::new();
    h.insert("type".into(), call.kind.clone().into());
    for (key, value) in [
        ("name", call.name.clone().map(Value::from)),
        ("id", call.id.clone().map(Value::from)),
        ("input", call.input.clone()),
        ("result", call.result.clone()),
        ("raw", Some(call.raw.clone()).filter(|r| !r.is_null())),
    ] {
        if let Some(v) = value {
            h.insert(key.into(), v);
        }
    }
    Value::Object(h)
}

/// `Attachment#to_h`: `{ type:, source: }`. In-memory bytes have no source to record (`null`).
fn attachment_h(a: &Attachment) -> Value {
    a.to_h()
}
