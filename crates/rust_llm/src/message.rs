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
            _ => Err(Error::Argument("Expected role to be one of: system, user, assistant, tool".into())),
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
    pub fn as_map(&self) -> Map<String, Value> {
        match self {
            ToolArguments::Parsed(m) => m.clone(),
            ToolArguments::Partial(s) => serde_json::from_str(s).unwrap_or_default(),
        }
    }
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
    pub fn new(id: impl Into<String>, name: impl Into<String>, arguments: Map<String, Value>) -> Self {
        ToolCall {
            id: id.into(),
            name: name.into(),
            arguments: ToolArguments::Parsed(arguments),
            thought_signature: None,
            remote: false,
            starts: true,
        }
    }

    /// A streamed piece that opens a call (`id` present, possibly empty).
    pub(crate) fn opening(id: String, name: String, arguments: String) -> ToolCall {
        ToolCall { id, name, arguments: ToolArguments::Partial(arguments), thought_signature: None, remote: false, starts: true }
    }

    /// A streamed argument fragment for a call opened earlier (`id: nil`).
    pub(crate) fn fragment(arguments: String) -> ToolCall {
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
}

/// The raw HTTP exchange behind a response (`message.raw`). Like Faraday's `env.request_body`,
/// the request is kept as the exact serialized text that was sent: every message of a long chat
/// keeps its request, and each request holds the whole conversation so far, so a JSON tree per
/// message cost ~20x the memory. Parse it on demand with `request_body_json`.
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
    pub usage_entries: Vec<UsageEntry>,
    /// Primary key of the row this message is stored as, set by a persistence layer
    /// (`rust_llm_loco`). `None` means not yet persisted.
    pub record_id: Option<i64>,
    pub(crate) model_info: Option<Model>,
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
            usage_entries: Vec::new(),
            record_id: None,
            model_info: None,
        }
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

    /// `Message#cost`. Pass a model to price against it instead of the one that answered.
    pub fn cost(&self, model: Option<&Model>) -> Cost {
        if model.is_none() && !self.usage_entries.is_empty() {
            let complete = self.usage_entries.iter().all(UsageEntry::cost_available);
            return Cost::aggregate(self.usage_entries.iter().map(|e| &e.cost), complete);
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
        let entry = self.usage_entries.iter().rev().find(|e| e.status == UsageStatus::Succeeded);
        match entry {
            Some(e) => registry.find(&e.model, Some(&e.provider)).ok(),
            None => self.model.as_deref().and_then(|m| registry.find(m, None).ok()),
        }
    }

    /// `Message#without_thinking`: what gets replayed to a provider that did not produce the thinking.
    /// The per-request view of a stored message: everything a protocol renders, without the
    /// bookkeeping it never reads (the HTTP exchange, usage ledger, registry entry). Cloning
    /// those for every message on every request made long chats quadratic.
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
            usage_entries: Vec::new(),
            record_id: self.record_id,
            model_info: None,
        }
    }

    pub(crate) fn without_thinking(&self) -> Message {
        let mut m = self.clone();
        m.thinking = None;
        m.raw_reasoning = None;
        if let Some(calls) = &mut m.tool_calls {
            for (_, call) in calls.0.iter_mut() {
                call.thought_signature = None;
            }
        }
        m
    }

    /// `Message#to_h`.
    pub fn to_h(&self) -> Value {
        let mut h = Map::new();
        h.insert("role".into(), self.role.as_str().into());
        if let Some(c) = &self.content {
            h.insert("content".into(), c.clone().into());
        }
        if let Some(m) = &self.model {
            h.insert("model".into(), m.clone().into());
        }
        if let Some(calls) = &self.tool_calls {
            let calls: Map<String, Value> =
                calls.iter().map(|(k, v)| (k.clone(), serde_json::to_value(v).unwrap())).collect(); // ToolCall always serializes
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
            h.insert("citations".into(), serde_json::to_value(&self.citations).unwrap()); // Citation always serializes
        }
        if let Some(r) = &self.finish_reason {
            h.insert("finish_reason".into(), r.as_str().into());
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
        Value::Object(h)
    }
}
