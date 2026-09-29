//! Port of `lib/ruby_llm/protocol.rb`, `protocol/streaming.rb`, and `protocol/stream_accumulator.rb`.
//!
//! RubyLLM builds a protocol class per provider by mixing overrides into a base wire format
//! (`class Mistral::ChatCompletions < Protocols::ChatCompletions; include Mistral::Chat`). Here each
//! wire format is a module of functions that consult the provider for its dialect differences.

pub mod anthropic;
pub mod chat_completions;
pub mod gemini;
pub mod responses;

use serde_json::{Map, Value};

use crate::error::{Error, Result};
use crate::message::{Citation, FinishReason, Message, RawResponse, ServerToolCall, Thinking, ToolArguments, ToolCall, indexmap_lite::IndexMap};
use crate::model::Model;
use crate::providers::{ProtocolName, Provider};
use crate::thinking::ThinkingConfig;
use crate::tokens::Tokens;
use crate::tool::SharedTool;

/// `tool_prefs`: `with_tool_options(choice:, calls:)`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ToolPrefs {
    pub choice: Option<ToolChoice>,
    pub calls: Option<ToolCalls>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolChoice {
    Auto,
    None,
    Required,
    Tool(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolCalls {
    One,
    Many,
}

/// The normalized `with_schema` payload: `{ name:, schema:, strict: }`.
#[derive(Debug, Clone, PartialEq)]
pub struct Schema {
    pub name: String,
    pub schema: Value,
    pub strict: Option<bool>,
    pub description: Option<String>,
}

/// Everything `Protocol#render_payload` receives.
pub struct Request<'a> {
    pub provider: Provider,
    pub config: &'a crate::Config,
    pub model: &'a Model,
    pub messages: &'a [Message],
    pub tools: &'a [SharedTool],
    pub tool_prefs: &'a ToolPrefs,
    pub temperature: Option<f64>,
    pub max_output_tokens: Option<i64>,
    pub schema: Option<&'a Schema>,
    pub thinking: Option<&'a ThinkingConfig>,
    /// `with_citations`.
    pub citations: bool,
    /// `with_caching`: `None` when not configured.
    pub caching: Option<&'a Caching>,
    pub stream: bool,
}

/// `with_caching(options)` / `with_caching(false)`, the chat's `@caching`.
#[derive(Debug, Clone, PartialEq)]
pub enum Caching {
    /// `with_caching(false)`: send no cache controls and render no cache boundaries.
    Off,
    /// `with_caching(key:, ttl:, mode:, id:, ...)`; empty for the provider's default behavior.
    On(Map<String, Value>),
}

impl Caching {
    /// `caching != false`: whether marked cache boundaries render.
    pub(crate) fn boundaries(caching: Option<&Caching>) -> bool {
        caching != Some(&Caching::Off)
    }

    /// The options of an enabled cache, or `None` when unset or off.
    pub(crate) fn options(caching: Option<&Caching>) -> Option<&Map<String, Value>> {
        match caching {
            Some(Caching::On(options)) => Some(options),
            _ => None,
        }
    }

    /// `prompt_cache_options`: rejects keys outside `allowed`, naming them as Ruby symbols.
    pub(crate) fn checked<'a>(caching: Option<&'a Caching>, allowed: &[&str], label: &str) -> Result<Option<&'a Map<String, Value>>> {
        let Some(options) = Caching::options(caching) else { return Ok(None) };
        let unsupported: Vec<String> = options.keys().filter(|k| !allowed.contains(&k.as_str())).map(|k| format!(":{k}")).collect();
        if unsupported.is_empty() {
            return Ok(Some(options));
        }
        Err(Error::Argument(format!("{label} prompt caching accepts {}, got {}", symbols(allowed), unsupported.join(", "))))
    }
}

/// `:key, :ttl, and :mode` for the Responses/Chat Completions messages, `:ttl` for one key.
fn symbols(keys: &[&str]) -> String {
    let visible: Vec<String> = keys.iter().filter(|k| **k != "retention").map(|k| format!(":{k}")).collect();
    match visible.len() {
        0 | 1 => visible.join(""),
        n => format!("{}, and {}", visible[..n - 1].join(", "), visible[n - 1]),
    }
}

/// `apply_end_user`: each protocol's safety identifier field; the rest drop it.
pub(crate) fn apply_end_user(protocol: ProtocolName, provider: Provider, payload: &mut Value, identifier: &str) {
    let field = match (protocol, provider) {
        (ProtocolName::Anthropic, _) => {
            deep_merge(payload, &serde_json::json!({ "metadata": { "user_id": identifier } }));
            return;
        }
        (ProtocolName::ChatCompletions | ProtocolName::Responses, Provider::OpenAI) => "safety_identifier",
        (ProtocolName::ChatCompletions | ProtocolName::Responses, Provider::DeepSeek) => "user_id",
        (ProtocolName::ChatCompletions, Provider::OpenRouter) => "user",
        _ => {
            tracing::debug!("{} has no safety identifier parameter, dropping {identifier}", provider.display());
            return;
        }
    };
    if let Some(p) = payload.as_object_mut() {
        p.insert(field.into(), identifier.into());
    }
}

/// `apply_compaction`: Anthropic's `context_management` edit, OpenAI's Responses
/// `compact_threshold` entry, or OpenRouter's context-compression plugin; dropped elsewhere.
pub(crate) fn apply_compaction(protocol: ProtocolName, provider: Provider, payload: &mut Value, compaction: &Map<String, Value>) {
    match (protocol, provider) {
        (ProtocolName::Anthropic, _) => anthropic::apply_compaction(payload, compaction),
        (ProtocolName::Responses, Provider::OpenAI) => {
            let mut entry = serde_json::json!({ "type": "compaction" });
            if let Some(at) = compaction.get("at").filter(|v| !v.is_null() && **v != Value::Bool(false)) {
                entry["compact_threshold"] = at.clone();
            }
            if let Some(p) = payload.as_object_mut() {
                p.insert("context_management".into(), serde_json::json!([entry]));
            }
        }
        (ProtocolName::ChatCompletions, Provider::OpenRouter) => {
            if !compaction.is_empty() {
                tracing::debug!("OpenRouter compresses context at the model's own limit, dropping {compaction:?}");
            }
            if let Some(p) = payload.as_object_mut() {
                let mut plugins = p.get("plugins").and_then(Value::as_array).cloned().unwrap_or_default();
                plugins.push(serde_json::json!({ "id": "context-compression" }));
                p.insert("plugins".into(), Value::Array(plugins));
            }
        }
        _ => tracing::debug!("{} has no context compaction parameter, dropping {compaction:?}", provider.display()),
    }
}

/// Where and how to send a rendered request.
pub struct Endpoint {
    pub path: String,
    pub headers: Vec<(String, String)>,
}

pub fn render(protocol: ProtocolName, req: &Request) -> Result<Value> {
    if req.citations && !req.model.supports("citations") {
        warn_unsupported_citations(protocol, req.model);
    }
    match protocol {
        ProtocolName::ChatCompletions => chat_completions::render_payload(req),
        ProtocolName::Responses => responses::render_payload(req),
        ProtocolName::Anthropic => anthropic::render_payload(req),
        ProtocolName::Gemini => gemini::render_payload(req),
    }
}

/// Each protocol's `warn_unsupported_citations`, called from `render_payload`.
fn warn_unsupported_citations(protocol: ProtocolName, model: &Model) {
    let hint = if protocol == ProtocolName::Gemini {
        "Gemini citations come from Google Search grounding: with_provider_options(tools: [{ google_search: {} }])."
    } else {
        "with_citations may have no effect."
    };
    tracing::warn!("{} does not support citations according to the model registry. {hint}", model.id);
}

pub fn endpoint(protocol: ProtocolName, provider: Provider, model: &Model, stream: bool) -> Endpoint {
    let path = match protocol {
        ProtocolName::ChatCompletions => "chat/completions".to_string(),
        ProtocolName::Responses if provider == Provider::Perplexity => "v1/agent".to_string(),
        ProtocolName::Responses => "responses".to_string(),
        ProtocolName::Anthropic => "v1/messages".to_string(),
        ProtocolName::Gemini if stream => format!("models/{}:streamGenerateContent?alt=sse", model.id),
        ProtocolName::Gemini => format!("models/{}:generateContent", model.id),
    };
    let headers = if protocol == ProtocolName::Anthropic && stream {
        vec![("Accept-Encoding".into(), "identity".into())]
    } else {
        Vec::new()
    };
    Endpoint { path, headers }
}

pub fn parse_completion(protocol: ProtocolName, provider: Provider, model: &Model, raw: RawResponse) -> Result<Message> {
    let body = raw.body.clone();
    if body.is_null() || body.as_object().is_some_and(Map::is_empty) {
        return Err(Error::Api("Provider returned an empty response body".into(), None));
    }
    match protocol {
        ProtocolName::ChatCompletions => chat_completions::parse_completion_body(provider, &body, raw),
        ProtocolName::Responses => responses::parse_completion_body(provider, &body, raw),
        ProtocolName::Anthropic => anthropic::parse_completion_body(&body, raw),
        ProtocolName::Gemini => gemini::parse_completion_body(model, &body, raw),
    }
}

/// Per-stream state a protocol's `build_chunk` needs (Anthropic block tracking, Responses citation offsets).
#[derive(Default)]
pub struct StreamState {
    pub anthropic: anthropic::StreamBlocks,
    pub gemini_parts: Vec<Value>,
    pub citation_lengths: Vec<((i64, i64), usize)>,
    /// `OpenRouter::Streaming`'s `@raw_reasoning`: the reasoning_details merged so far.
    pub openrouter_reasoning: Option<Vec<Value>>,
}

pub fn build_chunk(protocol: ProtocolName, provider: Provider, state: &mut StreamState, data: &Value) -> Result<Message> {
    match protocol {
        ProtocolName::ChatCompletions => {
            let mut chunk = chat_completions::build_chunk(provider, data);
            if provider == Provider::OpenRouter {
                let details = data.pointer("/choices/0/delta/reasoning_details");
                chunk.raw_reasoning = chat_completions::accumulate_raw_reasoning(&mut state.openrouter_reasoning, details);
            }
            Ok(chunk)
        }
        ProtocolName::Responses => responses::build_chunk(provider, state, data),
        ProtocolName::Anthropic => Ok(anthropic::build_chunk(&mut state.anthropic, data)),
        ProtocolName::Gemini => Ok(gemini::build_chunk(state, data)),
    }
}

/// Maps an in-stream error body to the HTTP status RubyLLM raises it as.
pub fn streaming_error_status(protocol: ProtocolName) -> fn(&str) -> Option<u16> {
    match protocol {
        ProtocolName::Anthropic => |data| {
            let v: Value = serde_json::from_str(data).ok()?;
            match v.pointer("/error/type").and_then(Value::as_str) {
                Some("overloaded_error") => Some(529),
                _ => Some(500),
            }
        },
        ProtocolName::Gemini => |data| {
            let v: Value = serde_json::from_str(data).ok()?;
            let err = v.get("error").unwrap_or(&v);
            err.get("code").and_then(Value::as_u64).map(|c| c as u16)
        },
        _ => |data| {
            let v: Value = serde_json::from_str(data).ok()?;
            let kind = v
                .pointer("/error/type")
                .or_else(|| v.get("code"))
                .and_then(Value::as_str)
                .unwrap_or("");
            Some(match kind {
                "server_error" => 500,
                "rate_limit_exceeded" | "insufficient_quota" => 429,
                _ => 400,
            })
        },
    }
}

/// `Protocol::StreamAccumulator`: folds chunks into the final message.
#[derive(Default)]
pub struct StreamAccumulator {
    content: String,
    model: Option<String>,
    citations: Vec<Citation>,
    thinking_text: Option<String>,
    thinking_signature: Option<String>,
    tool_calls: IndexMap<ToolCall>,
    tool_call_ids_by_index: Vec<(String, String)>,
    latest_tool_call_id: Option<String>,
    tokens: Tokens,
    server_tool_calls: Vec<ServerToolCall>,
    raw_content: Option<Value>,
    raw_reasoning: Option<Value>,
    finish_reason: Option<FinishReason>,
}

impl StreamAccumulator {
    /// Usage reported so far, for billing a stream that fails partway.
    pub fn tokens(&self) -> &Tokens {
        &self.tokens
    }

    pub fn add(&mut self, chunk: &Message) {
        if self.model.as_deref().unwrap_or("").is_empty() {
            self.model = chunk.model.clone();
        }
        if let Some(calls) = &chunk.tool_calls {
            for (stream_key, call) in calls.iter() {
                if call.starts_call() {
                    self.start_tool_call(stream_key, call);
                } else {
                    self.append_fragment(stream_key, call);
                }
            }
        }
        if let Some(text) = &chunk.content {
            self.content.push_str(text);
        }
        for c in &chunk.citations {
            if !self.citations.contains(c) {
                self.citations.push(c.clone());
            }
        }
        if let Some(t) = &chunk.thinking {
            if let Some(text) = &t.text {
                self.thinking_text.get_or_insert_with(String::new).push_str(text);
            }
            if self.thinking_signature.is_none() {
                self.thinking_signature = t.signature.clone();
            }
        }
        for call in &chunk.server_tool_calls {
            match self.server_tool_calls.iter_mut().find(|e| call.id.is_some() && e.id == call.id && e.kind == call.kind) {
                Some(existing) => *existing = call.clone(),
                None => self.server_tool_calls.push(call.clone()),
            }
        }
        if chunk.raw_content.is_some() {
            self.raw_content = chunk.raw_content.clone();
        }
        if chunk.raw_reasoning.is_some() {
            self.raw_reasoning = chunk.raw_reasoning.clone();
        }
        if chunk.finish_reason.is_some() {
            self.finish_reason = chunk.finish_reason.clone();
        }
        self.tokens.merge_latest(&chunk.tokens);
    }

    fn start_tool_call(&mut self, stream_key: &str, call: &ToolCall) {
        let mut call = call.clone();
        if let ToolArguments::Parsed(m) = &call.arguments
            && m.is_empty() {
                call.arguments = ToolArguments::Partial(String::new());
            }
        if call.id.is_empty() {
            call.id = uuid::Uuid::new_v4().to_string();
        }
        let id = call.id.clone();
        self.tool_calls.insert(id.clone(), call);
        self.tool_call_ids_by_index.retain(|(k, _)| k != stream_key);
        self.tool_call_ids_by_index.push((stream_key.to_string(), id.clone()));
        self.latest_tool_call_id = Some(id);
    }

    fn append_fragment(&mut self, stream_key: &str, call: &ToolCall) {
        let id = self
            .tool_call_ids_by_index
            .iter()
            .find(|(k, _)| k == stream_key)
            .map(|(_, id)| id.clone())
            .or_else(|| self.tool_calls.contains_key(stream_key).then(|| stream_key.to_string()))
            // A keyless fragment continues the latest call; an unknown key is dropped (find_tool_call).
            .or_else(|| stream_key.is_empty().then(|| self.latest_tool_call_id.clone()).flatten());
        let Some(existing) = id.and_then(|id| self.tool_calls.get_mut(&id)) else { return };
        let fragment = match &call.arguments {
            ToolArguments::Partial(s) => s.clone(),
            ToolArguments::Parsed(m) if m.is_empty() => String::new(),
            ToolArguments::Parsed(m) => Value::Object(m.clone()).to_string(),
        };
        match &mut existing.arguments {
            ToolArguments::Partial(s) => s.push_str(&fragment),
            ToolArguments::Parsed(_) => {}
        }
        if existing.thought_signature.is_none() {
            existing.thought_signature = call.thought_signature.clone();
        }
    }

    pub fn into_message(self, raw: RawResponse) -> Result<Message> {
        let finish = self.finish_reason.as_ref().map(|f| f.as_str().to_string());
        let mut tool_calls = IndexMap::new();
        for (id, call) in self.tool_calls.0 {
            let arguments = match call.arguments {
                ToolArguments::Partial(s) if s.is_empty() => Map::new(),
                ToolArguments::Partial(s) => serde_json::from_str::<Map<String, Value>>(&s)
                    .map_err(|_| Error::tool_call_parse(finish.as_deref()))?,
                ToolArguments::Parsed(m) => m,
            };
            tool_calls.insert(id, ToolCall { arguments: ToolArguments::Parsed(arguments), ..call });
        }
        let content = self.content;
        let citations = self
            .citations
            .into_iter()
            .map(|mut c| {
                if c.text.is_none()
                    && let (Some(s), Some(e)) = (c.start_index, c.end_index) {
                        c.text = char_slice(&content, s, e);
                    }
                c
            })
            .collect();
        let mut message = Message::chunk();
        message.content = (!content.is_empty()).then_some(content);
        message.citations = citations;
        message.thinking = Thinking::build(self.thinking_text, self.thinking_signature);
        message.tokens = self.tokens;
        message.server_tool_calls = self.server_tool_calls;
        message.raw_content = self.raw_content;
        message.raw_reasoning = self.raw_reasoning;
        message.finish_reason = self.finish_reason;
        message.model = self.model;
        message.tool_calls = (!tool_calls.is_empty()).then_some(tool_calls);
        message.raw = Some(raw);
        Ok(message.normalized())
    }
}

/// Ruby `content[start...end]` on characters.
pub(crate) fn char_slice(s: &str, start: i64, end: i64) -> Option<String> {
    if start < 0 || end < start {
        return None;
    }
    let chars: Vec<char> = s.chars().collect();
    if start as usize > chars.len() {
        return None;
    }
    Some(chars[start as usize..(end as usize).min(chars.len())].iter().collect())
}

pub(crate) fn normalize_finish_reason(reason: Option<&str>, table: &[(&str, &str)]) -> Option<FinishReason> {
    let reason = reason?;
    let symbol = table.iter().find(|(k, _)| *k == reason).map(|(_, v)| *v).unwrap_or(reason);
    Some(FinishReason::from_symbol(symbol))
}

pub(crate) fn int(v: Option<&Value>) -> Option<i64> {
    v.and_then(|v| v.as_i64().or_else(|| v.as_f64().map(|f| f as i64)))
}

pub(crate) fn str_of(v: Option<&Value>) -> Option<String> {
    v.and_then(Value::as_str).map(str::to_string)
}

/// Ruby's `deep_merge` for `provider_options` and tool `provider_options`.
pub(crate) fn deep_merge(base: &mut Value, overlay: &Value) {
    match (base, overlay) {
        (Value::Object(b), Value::Object(o)) => {
            for (k, v) in o {
                match b.get_mut(k) {
                    Some(existing) if existing.is_object() && v.is_object() => deep_merge(existing, v),
                    _ => {
                        b.insert(k.clone(), v.clone());
                    }
                }
            }
        }
        (b, o) => *b = o.clone(),
    }
}

pub(crate) fn tool_call_map(calls: Vec<ToolCall>) -> Option<IndexMap<ToolCall>> {
    if calls.is_empty() {
        return None;
    }
    Some(calls.into_iter().map(|c| (c.id.clone(), c)).collect())
}
