//! Port of `lib/ruby_llm/protocols/mistral/{content,conversations,multi_completion}.rb`,
//! `protocols/mistral/conversations/{chat,streaming,images}.rb`, and
//! `providers/mistral/conversations.rb`: Mistral's Conversations API (`protocol: :conversations`,
//! and the protocol `paint` goes through), plus the multi-completion responses Mistral's Chat
//! Completions returns when hosted tools run inside one request.

use std::collections::BTreeMap;

use serde_json::{Map, Value, json};

use super::{Request, ToolChoice, int, str_of};
use crate::attachment::Attachment;
use crate::error::{Error, Result};
use crate::files::UploadedFile;
use crate::message::{
    Citation, FinishReason, Message, RawResponse, Role, ServerToolCall, Thinking, ToolCall,
};
use crate::providers::{ProtocolName, Provider};

// ---- Content --------------------------------------------------------------------------------

/// `Mistral::Content#parse_conversation_content`'s accumulator.
#[derive(Default)]
pub(crate) struct Content {
    pub text: String,
    pub thinking: String,
    pub attachments: Vec<Attachment>,
    pub citations: Vec<Citation>,
}

impl Content {
    fn thinking(&self) -> Option<String> {
        (!self.thinking.is_empty()).then(|| self.thinking.clone())
    }
}

/// `parse_conversation_content`: the `message.output` entries' parts.
pub(crate) fn parse_conversation_content(output: &[Value]) -> Content {
    let mut result = Content::default();
    for entry in output
        .iter()
        .filter(|e| e.get("type").and_then(Value::as_str) == Some("message.output"))
    {
        parse_conversation_parts(entry.get("content"), &mut result);
    }
    result
}

/// `parse_conversation_parts`.
pub(crate) fn parse_conversation_parts(content: Option<&Value>, result: &mut Content) {
    let parts = match content {
        Some(Value::String(s)) => {
            result.text.push_str(s);
            return;
        }
        Some(Value::Array(parts)) => parts.clone(),
        Some(Value::Null) | None => Vec::new(),
        Some(other) => vec![other.clone()],
    };
    for part in parts.iter().filter(|p| !p.is_null()) {
        match part.get("type").and_then(Value::as_str) {
            Some("text") => result
                .text
                .push_str(part.get("text").and_then(Value::as_str).unwrap_or("")),
            Some("thinking") => {
                let text: String = match part.get("thinking") {
                    Some(Value::Array(items)) => items
                        .iter()
                        .filter_map(|i| i.get("text").and_then(Value::as_str))
                        .collect(),
                    Some(item @ Value::Object(_)) => item
                        .get("text")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string(),
                    _ => String::new(),
                };
                result.thinking.push_str(&text);
            }
            Some("tool_reference") => {
                let offset = result.text.chars().count() as i64;
                result.citations.push(Citation {
                    url: str_of(part.get("url")),
                    title: str_of(part.get("title")),
                    cited_text: str_of(part.get("description")),
                    start_index: Some(offset),
                    end_index: Some(offset),
                    ..Default::default()
                });
            }
            Some("tool_file") => result.attachments.push(parse_conversation_file(part)),
            Some("image_url") => {
                let url = match part.get("image_url") {
                    Some(Value::Object(o)) => o.get("url").and_then(Value::as_str).unwrap_or(""),
                    Some(Value::String(s)) => s,
                    _ => "",
                };
                result.attachments.push(Attachment::new(url));
            }
            _ => {}
        }
    }
}

/// `parse_conversation_file`: a generated file Mistral stores, as a downloadable provider file.
fn parse_conversation_file(part: &Value) -> Attachment {
    let format = part.get("file_type").and_then(Value::as_str).unwrap_or("");
    let mime_type = if format.contains('/') {
        format.to_string()
    } else {
        mime_guess::from_ext(format).first().map_or_else(
            || "application/octet-stream".to_string(),
            |m| m.essence_str().to_string(),
        )
    };
    Attachment::from_uploaded(UploadedFile {
        id: str_of(part.get("file_id")).unwrap_or_default(),
        provider: Provider::Mistral.slug().into(),
        filename: str_of(part.get("file_name")),
        byte_size: None,
        created_at: None,
        expires_at: None,
        status: None,
        mime_type: Some(mime_type),
        purpose: None,
        uri: None,
        downloadable: Some(true),
        metadata: part.clone(),
    })
}

// ---- Conversations::Chat --------------------------------------------------------------------

/// `Conversations::Chat#render_payload` (with `#render`'s tool checks applied by the caller via
/// `finish_render`): the Chat Completions options, moved into `completion_args`.
pub fn render_payload(req: &Request) -> Result<Value> {
    let chat_request = Request {
        provider: req.provider,
        config: req.config,
        model: req.model,
        messages: req.messages,
        tools: req.tools,
        tool_prefs: req.tool_prefs,
        temperature: req.temperature,
        max_output_tokens: req.max_output_tokens,
        schema: req.schema,
        thinking: req.thinking,
        citations: false,
        caching: None,
        stream: req.stream,
    };
    let mut options = super::chat_completions::render_payload(&chat_request)?;
    normalize_conversation_choice(&mut options, req)?;
    let system: Vec<String> = req
        .messages
        .iter()
        .filter(|m| m.role == Role::System)
        .map(|m| m.content().to_string())
        .collect();
    let mut completion_args = Map::new();
    for key in [
        "temperature",
        "max_tokens",
        "response_format",
        "reasoning_effort",
        "tool_choice",
    ] {
        if let Some(v) = options.get(key) {
            completion_args.insert(key.into(), v.clone());
        }
    }
    Ok(json!({
        "model": req.model.id,
        "inputs": format_entries(req.provider, req.messages)?,
        "instructions": system.join("\n\n"),
        "completion_args": completion_args,
        "tools": options.get("tools").cloned().unwrap_or_else(|| json!([])),
        "store": false,
        "stream": req.stream,
    }))
}

/// `normalize_conversation_choice`: only `:auto`, `:none`, or `:required` (sent as `any`).
/// Conversations subclasses the base Chat Completions protocol, so the choice is the base
/// rendering, not the Mistral Chat Completions dialect's single-tool rewrite.
fn normalize_conversation_choice(options: &mut Value, req: &Request) -> Result<()> {
    if options.get("tool_choice").is_none() {
        return Ok(());
    }
    options["tool_choice"] = match &req.tool_prefs.choice {
        Some(ToolChoice::Tool(_)) => {
            return Err(Error::Argument(
                "Mistral Conversations supports :auto, :none, or :required for tool choice".into(),
            ));
        }
        Some(ToolChoice::Required) => "any".into(),
        Some(ToolChoice::None) => "none".into(),
        Some(ToolChoice::Auto) | None => "auto".into(),
    };
    Ok(())
}

/// `Conversations::Chat#render`: tool entries deduplicated after provider tools apply, and hosted
/// confirmations refused (they need provider-side conversation storage).
pub(crate) fn finish_render(payload: &mut Value) -> Result<()> {
    let mut unique: Vec<Value> = Vec::new();
    for tool in payload
        .get("tools")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        if !unique.contains(tool) {
            unique.push(tool.clone());
        }
    }
    let confirming =
        unique.iter().any(
            |t| match t.pointer("/tool_configuration/requires_confirmation") {
                Some(Value::Array(a)) => !a.is_empty(),
                Some(Value::Null) | None => false,
                Some(_) => true,
            },
        );
    if confirming {
        return Err(Error::Argument(
            "Mistral hosted tool confirmations require provider conversation storage".into(),
        ));
    }
    if let Some(p) = payload.as_object_mut() {
        p.insert("tools".into(), Value::Array(unique));
    }
    Ok(())
}

/// `format_entries`: stored provider history replays verbatim (hosted executions as function
/// call/result pairs), everything else as Conversations entries.
pub fn format_entries(provider: Provider, messages: &[Message]) -> Result<Vec<Value>> {
    let mut entries = Vec::new();
    for message in messages.iter().filter(|m| m.role != Role::System) {
        if let Some(raw) = &message.raw_content {
            match raw {
                Value::Array(items) => entries.extend(items.iter().cloned()),
                other => entries.push(other.clone()),
            }
        } else if message.is_tool_result() {
            entries.push(json!({ "type": "function.result", "tool_call_id": message.tool_call_id, "result": message.content() }));
        } else {
            entries.extend(format_conversation_message(provider, message)?);
        }
    }
    Ok(entries
        .iter()
        .enumerate()
        .flat_map(|(index, entry)| replay_conversation_entry(entry, index))
        .collect())
}

fn replay_conversation_entry(entry: &Value, index: usize) -> Vec<Value> {
    if entry.get("type").and_then(Value::as_str) == Some("tool.execution") {
        return replay_conversation_execution(entry, index);
    }
    let mut result = entry.clone();
    if let Some(o) = result.as_object_mut() {
        for key in ["id", "object", "created_at", "completed_at"] {
            o.remove(key);
        }
    }
    if entry.get("type").and_then(Value::as_str) == Some("message.output")
        && let Some(Value::Array(parts)) = entry.get("content")
    {
        let parts: Vec<Value> = parts
            .iter()
            .map(|part| {
                if part.get("type").and_then(Value::as_str) != Some("tool_reference") {
                    return part.clone();
                }
                let title = part.get("title").and_then(Value::as_str).unwrap_or("");
                let url = part.get("url").and_then(Value::as_str).unwrap_or("");
                json!({ "type": "text", "text": format!("[{title}]({url})") })
            })
            .collect();
        result["content"] = Value::Array(parts);
    }
    vec![result]
}

/// `replay_conversation_execution`: a hosted execution replays as a call and its actual result,
/// under an id derived from the entry (`Digest::SHA256.hexdigest(identity)[0, 9]`).
fn replay_conversation_execution(entry: &Value, index: usize) -> Vec<Value> {
    let identity = match entry.get("id").and_then(Value::as_str) {
        Some(id) => id.to_string(),
        None => format!("{index}:{}", ruby_json(entry)),
    };
    let id = entry
        .get("tool_call_id")
        .and_then(Value::as_str)
        .map_or_else(
            || sha256_hex(identity.as_bytes())[..9].to_string(),
            str::to_string,
        );
    let info = entry.get("info").cloned().unwrap_or(Value::Null);
    let result = match &info {
        Value::Object(o) if o.contains_key("result") => o["result"].clone(),
        _ => info,
    };
    let name = entry
        .get("function")
        .filter(|v| !v.is_null())
        .or_else(|| entry.get("name"))
        .cloned()
        .unwrap_or(Value::Null);
    vec![
        json!({ "type": "function.call", "tool_call_id": id, "name": name, "arguments": entry.get("arguments").cloned().unwrap_or(Value::Null) }),
        json!({ "type": "function.result", "tool_call_id": id, "result": match result {
            Value::String(s) => s,
            other => ruby_json(&other),
        }}),
    ]
}

/// `JSON.generate`: compact, without escaping `/`.
fn ruby_json(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_default()
}

fn format_conversation_message(provider: Provider, message: &Message) -> Result<Vec<Value>> {
    let mut entries = Vec::new();
    if message.content.is_some() || !message.attachments.is_empty() {
        let content = super::chat_completions::format_content(
            provider,
            message.content.as_deref(),
            &message.attachments,
        )?;
        entries.push(
            json!({ "type": "message.input", "role": message.role.as_str(), "content": content }),
        );
    }
    for call in message.tool_calls.iter().flat_map(|c| c.values()) {
        entries.push(json!({
            "type": "function.call", "tool_call_id": call.id, "name": call.name,
            "arguments": ruby_json(&Value::Object(call.arguments())),
        }));
    }
    Ok(entries)
}

/// `Conversations::Chat#parse_completion_body`.
pub fn parse_completion_body(
    model_id: &str,
    data: &Value,
    raw: Option<RawResponse>,
) -> Result<Message> {
    let output: Vec<Value> = data
        .get("outputs")
        .and_then(Value::as_array)
        .cloned()
        .ok_or_else(|| Error::Api("key not found: \"outputs\"".into(), None))?;
    let content = parse_conversation_content(&output);
    let calls = parse_conversation_calls(&output)?;
    let mut m = Message::chunk();
    m.content = Some(content.text.clone());
    m.thinking = Thinking::build(content.thinking(), None);
    m.attachments = content.attachments;
    m.citations = content.citations;
    m.server_tool_calls = parse_conversation_steps(&output);
    m.raw_content = Some(Value::Array(output.clone()));
    m.model = output
        .iter()
        .filter_map(|e| str_of(e.get("model")))
        .next_back()
        .or_else(|| Some(model_id.to_string()));
    m.finish_reason = Some(if calls.is_empty() {
        FinishReason::Stop
    } else {
        FinishReason::ToolCalls
    });
    m.tool_calls = super::tool_call_map(calls);
    parse_conversation_usage(&mut m, data.get("usage").unwrap_or(&Value::Null));
    m.raw = raw;
    Ok(m.normalized())
}

fn is_pending_call(entry: &Value) -> bool {
    entry.get("type").and_then(Value::as_str) == Some("function.call")
        && entry.get("confirmation_status").is_none_or(Value::is_null)
}

fn parse_conversation_calls(output: &[Value]) -> Result<Vec<ToolCall>> {
    if output.iter().any(|e| {
        e.get("type").and_then(Value::as_str) == Some("function.call")
            && e.get("confirmation_status").and_then(Value::as_str) == Some("pending")
    }) {
        return Err(Error::Api("Mistral returned a hosted tool confirmation that requires provider conversation storage".into(), None));
    }
    output
        .iter()
        .filter(|e| is_pending_call(e))
        .map(|e| {
            let arguments = match e.get("arguments") {
                Some(Value::String(s)) if !s.is_empty() => serde_json::from_str(s)
                    .map_err(|_| Error::tool_call_parse(Some("tool_calls")))?,
                _ => Map::new(),
            };
            Ok(ToolCall::new(
                str_of(e.get("tool_call_id")).unwrap_or_default(),
                str_of(e.get("name")).unwrap_or_default(),
                arguments,
            ))
        })
        .collect()
}

fn parse_conversation_steps(output: &[Value]) -> Vec<ServerToolCall> {
    output
        .iter()
        .filter(|e| {
            let kind = e.get("type").and_then(Value::as_str);
            kind == Some("tool.execution") || (kind == Some("function.call") && !is_pending_call(e))
        })
        .map(|e| ServerToolCall {
            kind: str_of(e.get("type")).unwrap_or_default(),
            name: str_of(e.get("name")),
            id: str_of(e.get("id")),
            input: e.get("arguments").cloned(),
            result: e.get("info").cloned(),
            raw: e.clone(),
        })
        .collect()
}

/// `parse_conversation_usage`: connector tokens count as input.
fn parse_conversation_usage(m: &mut Message, usage: &Value) {
    m.tokens.input = int(usage.get("prompt_tokens"))
        .map(|p| p + int(usage.get("connector_tokens")).unwrap_or(0));
    m.tokens.output = int(usage.get("completion_tokens"));
    m.tokens.server_tool_use = usage.get("connectors").and_then(Value::as_object).cloned();
}

// ---- Conversations::Streaming -----------------------------------------------------------------

/// `@conversation_output`, `@conversation_response`, `@conversation_done`.
#[derive(Default)]
pub struct ConversationStream {
    output: BTreeMap<i64, Value>,
    usage: Option<Value>,
    pub(crate) done: bool,
    pub(crate) message: Option<Message>,
}

/// `Conversations::Streaming#build_chunk`.
pub fn build_conversation_chunk(
    model_id: &str,
    state: &mut ConversationStream,
    data: &Value,
) -> Result<Message> {
    match data.get("type").and_then(Value::as_str) {
        Some("conversation.response.done") => {
            state.done = true;
            state.usage = data.get("usage").cloned();
            let message =
                parse_completion_body(model_id, &streamed_conversation_response(state), None)?;
            let mut chunk = Message::chunk();
            chunk.model = message.model.clone();
            chunk.tokens = message.tokens.clone();
            chunk.citations = message.citations.clone();
            chunk.tool_calls = message.tool_calls.clone();
            chunk.server_tool_calls = message.server_tool_calls.clone();
            chunk.raw_content = message.raw_content.clone();
            chunk.attachments = message.attachments.clone();
            chunk.finish_reason = message.finish_reason.clone();
            state.message = Some(message);
            return Ok(chunk);
        }
        Some("conversation.response.error") => {
            let message = data
                .get("message")
                .and_then(Value::as_str)
                .or_else(|| data.pointer("/error/message").and_then(Value::as_str))
                .unwrap_or("Mistral conversation failed");
            return Err(Error::Api(message.into(), None));
        }
        Some("message.output.delta") => return Ok(conversation_text_chunk(state, data)),
        Some(
            kind @ ("function.call.delta"
            | "tool.execution.started"
            | "tool.execution.delta"
            | "tool.execution.done"),
        ) => {
            let kind = kind
                .trim_end_matches(".delta")
                .trim_end_matches(".started")
                .trim_end_matches(".done");
            let entry = state
                .output
                .entry(int(data.get("output_index")).unwrap_or(0))
                .or_insert_with(|| json!({ "type": kind, "arguments": "" }));
            for key in [
                "id",
                "model",
                "name",
                "tool_call_id",
                "confirmation_status",
                "function",
                "info",
            ] {
                if let Some(v) = data.get(key) {
                    entry[key] = v.clone();
                }
            }
            let arguments = format!(
                "{}{}",
                entry.get("arguments").and_then(Value::as_str).unwrap_or(""),
                data.get("arguments").and_then(Value::as_str).unwrap_or("")
            );
            entry["arguments"] = arguments.into();
        }
        _ => {}
    }
    Ok(Message::chunk())
}

fn streamed_conversation_response(state: &ConversationStream) -> Value {
    let mut response = json!({ "object": "conversation.response", "outputs": state.output.values().cloned().collect::<Vec<_>>() });
    if let Some(usage) = &state.usage {
        response["usage"] = usage.clone();
    } else {
        response["usage"] = Value::Null;
    }
    response
}

fn conversation_text_chunk(state: &mut ConversationStream, data: &Value) -> Message {
    let entry = state
        .output
        .entry(int(data.get("output_index")).unwrap_or(0))
        .or_insert_with(|| json!({ "type": "message.output", "content": [] }));
    for key in ["id", "model", "role"] {
        if let Some(v) = data.get(key) {
            entry[key] = v.clone();
        }
    }
    let part = match data.get("content") {
        Some(Value::String(s)) => json!({ "type": "text", "text": s }),
        Some(other) => other.clone(),
        None => Value::Null,
    };
    let index = int(data.get("content_index")).unwrap_or(0).max(0) as usize;
    if let Some(parts) = entry.get_mut("content").and_then(Value::as_array_mut) {
        merge_conversation_part(parts, index, &part);
    }
    let mut content = Content::default();
    parse_conversation_parts(Some(&Value::Array(vec![part])), &mut content);
    let mut chunk = Message::chunk();
    chunk.content = Some(content.text.clone());
    chunk.model = str_of(data.get("model"));
    chunk.thinking = Thinking::build(content.thinking(), None);
    chunk
}

fn merge_conversation_part(parts: &mut Vec<Value>, index: usize, part: &Value) {
    let kind = part.get("type").and_then(Value::as_str);
    match parts.get_mut(index) {
        Some(existing) if kind == Some("text") && existing.get("text").is_some() => {
            let joined = format!(
                "{}{}",
                existing.get("text").and_then(Value::as_str).unwrap_or(""),
                part.get("text").and_then(Value::as_str).unwrap_or("")
            );
            existing["text"] = joined.into();
        }
        Some(existing) if kind == Some("thinking") => {
            let mut items = existing
                .get("thinking")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            match part.get("thinking") {
                Some(Value::Array(more)) => items.extend(more.iter().cloned()),
                Some(Value::Null) | None => {}
                Some(one) => items.push(one.clone()),
            }
            existing["thinking"] = Value::Array(items);
        }
        _ => {
            while parts.len() < index {
                parts.push(Value::Null);
            }
            if parts.len() == index {
                parts.push(part.clone());
            } else {
                parts[index] = part.clone();
            }
        }
    }
}

/// `stream_response`'s ending.
pub(crate) fn finish_conversation_stream(
    state: &mut ConversationStream,
    raw: RawResponse,
) -> Result<Message> {
    let Some(mut message) = state.message.take().filter(|_| state.done) else {
        return Err(Error::Api(
            "Mistral conversation stream ended before completion".into(),
            None,
        ));
    };
    message.raw = Some(raw);
    Ok(message)
}

// ---- Conversations::Images --------------------------------------------------------------------

/// `Conversations::Images#render_image_payload`.
pub(crate) fn render_image_payload(
    prompt: &str,
    model: &str,
    size: Option<&str>,
    count: Option<i64>,
    editing: bool,
    provider_options: &Value,
) -> Result<Value> {
    if size.is_some() || count.is_some_and(|c| c != 1) {
        return Err(Error::Argument(
            "Mistral image generation does not accept size or count options".into(),
        ));
    }
    if editing {
        return Err(Error::UnsupportedAttachment(super::anthropic::unsupported(
            "image editing",
        )));
    }
    let mut payload = json!({ "model": model, "store": false, "inputs": prompt, "tools": [{ "type": "image_generation" }] });
    if provider_options.is_object() {
        super::deep_merge(&mut payload, provider_options);
    }
    Ok(payload)
}

/// `parse_image_responses` minus the download: the generated file ids and the usage the first
/// image carries (`{ "input_tokens", "output_tokens" }`).
pub(crate) fn parse_image_files(data: &Value) -> Result<(Vec<String>, Value)> {
    let output = data
        .get("outputs")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let files: Vec<String> = parse_conversation_content(&output)
        .attachments
        .iter()
        .filter_map(|a| a.provider_file_id().map(str::to_string))
        .collect();
    if files.is_empty() {
        return Err(Error::Api(
            "Mistral returned no generated image".into(),
            None,
        ));
    }
    let mut m = Message::chunk();
    parse_conversation_usage(&mut m, data.get("usage").unwrap_or(&Value::Null));
    let mut usage = Map::new();
    for (key, value) in [
        ("input_tokens", m.tokens.input),
        ("output_tokens", m.tokens.output),
    ] {
        if let Some(v) = value {
            usage.insert(key.into(), v.into());
        }
    }
    Ok((files, Value::Object(usage)))
}

// ---- MultiCompletion ------------------------------------------------------------------------

/// `MultiCompletion#format_message_group`: a multi-completion answer replays its own messages.
pub(crate) fn multi_messages_for_replay(message: &Message) -> Option<Vec<Value>> {
    let Some(Value::Array(content)) = &message.raw_content else {
        return None;
    };
    if content.is_empty()
        || !content
            .iter()
            .all(|e| e.get("role").is_some_and(|r| !r.is_null()))
    {
        return None;
    }
    Some(
        content
            .iter()
            .map(|e| {
                let mut e = e.clone();
                if let Some(o) = e.as_object_mut() {
                    o.remove("index");
                }
                e
            })
            .collect(),
    )
}

/// `MultiCompletion#parse_completion_body` when `choices[0].messages` is an array.
pub fn parse_multi_message(data: &Value, raw: Option<RawResponse>) -> Result<Option<Message>> {
    let Some(messages) = data
        .pointer("/choices/0/messages")
        .and_then(Value::as_array)
    else {
        return Ok(None);
    };
    let output: Vec<Value> = messages
        .iter()
        .filter(|m| m.get("role").and_then(Value::as_str) == Some("assistant"))
        .map(|m| {
            let mut m = m.clone();
            m["type"] = "message.output".into();
            m
        })
        .collect();
    let content = parse_conversation_content(&output);
    let results: BTreeMap<String, &Value> = messages
        .iter()
        .filter(|m| m.get("role").and_then(Value::as_str) == Some("tool"))
        .map(|m| (str_of(m.get("tool_call_id")).unwrap_or_default(), m))
        .collect();
    let calls: Vec<&Value> = messages
        .iter()
        .flat_map(|m| {
            m.get("tool_calls")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
        })
        .collect();
    let pending: Vec<&Value> = calls
        .iter()
        .copied()
        .filter(|c| !results.contains_key(c.get("id").and_then(Value::as_str).unwrap_or("")))
        .collect();
    if pending.iter().any(|c| {
        c.pointer("/metadata/tool_type")
            .is_some_and(|v| !v.is_null())
            || c.pointer("/metadata/integration_id")
                .is_some_and(|v| !v.is_null())
    }) {
        return Err(Error::Api(
            "Mistral returned an unfinished hosted tool call on Chat Completions".into(),
            None,
        ));
    }
    let mut tool_calls = Vec::new();
    for call in &pending {
        let arguments = match call.pointer("/function/arguments").and_then(Value::as_str) {
            Some(s) if !s.is_empty() => {
                serde_json::from_str(s).map_err(|_| Error::tool_call_parse(Some("tool_calls")))?
            }
            _ => Map::new(),
        };
        tool_calls.push(ToolCall::new(
            str_of(call.get("id")).unwrap_or_default(),
            str_of(call.pointer("/function/name")).unwrap_or_default(),
            arguments,
        ));
    }
    let steps: Vec<ServerToolCall> = calls
        .iter()
        .filter_map(|call| {
            let id = call.get("id").and_then(Value::as_str).unwrap_or("");
            let result = results.get(id)?;
            Some(ServerToolCall {
                kind: str_of(result.get("role")).unwrap_or_default(),
                id: Some(id.to_string()),
                name: str_of(call.pointer("/function/name")),
                input: call.pointer("/function/arguments").cloned(),
                result: result.get("content").cloned(),
                raw: (*result).clone(),
            })
        })
        .collect();
    // The usage and finish reason read like any Chat Completions response.
    let mut shell = data.clone();
    shell["choices"] = json!([{ "message": {}, "finish_reason": data.pointer("/choices/0/finish_reason").cloned().unwrap_or(Value::Null) }]);
    let base = super::chat_completions::parse_completion_body(
        Provider::Mistral,
        &shell,
        raw.clone().unwrap_or_else(empty_raw),
    )?;
    let mut m = Message::chunk();
    m.content = Some(content.text.clone());
    m.thinking = Thinking::build(content.thinking(), None);
    m.attachments = content.attachments;
    m.citations = content.citations;
    m.tool_calls = super::tool_call_map(tool_calls);
    m.server_tool_calls = steps;
    m.raw_content = Some(Value::Array(messages.clone()));
    m.tokens.input = base.tokens.input;
    m.tokens.output = base.tokens.output;
    m.tokens.cache_read = base.tokens.cache_read;
    m.model = str_of(data.get("model"));
    m.finish_reason = base.finish_reason;
    m.raw = raw;
    Ok(Some(m.normalized()))
}

fn empty_raw() -> RawResponse {
    RawResponse {
        status: 200,
        headers: Vec::new(),
        body: Value::Null,
        request_body: "".into(),
    }
}

/// `MultiCompletion#stream_response` applies when a tool other than a function is sent.
pub(crate) fn is_multi_stream(protocol: ProtocolName, provider: Provider, payload: &Value) -> bool {
    protocol == ProtocolName::ChatCompletions
        && provider == Provider::Mistral
        && payload
            .get("tools")
            .and_then(Value::as_array)
            .is_some_and(|tools| {
                tools
                    .iter()
                    .any(|t| t.get("type").and_then(Value::as_str) != Some("function"))
            })
}

/// `@multi_messages`, `@multi_usage`, `@multi_finish_reason`, `@multi_model`.
#[derive(Default)]
pub struct MultiStream {
    messages: BTreeMap<i64, Value>,
    usage: Vec<(String, Value)>,
    finish_reason: Option<Value>,
    model: Option<Value>,
}

/// `build_multi_chunk`.
pub fn build_multi_chunk(state: &mut MultiStream, data: &Value) -> Message {
    state.model = data.get("model").cloned();
    if let Some(usage) = data.get("usage").filter(|u| !u.is_null()) {
        let id = str_of(data.get("id")).unwrap_or_default();
        match state.usage.iter_mut().find(|(k, _)| *k == id) {
            Some((_, existing)) => *existing = usage.clone(),
            None => state.usage.push((id, usage.clone())),
        }
    }
    let empty = json!({});
    let choice = data.pointer("/choices/0").unwrap_or(&empty);
    let delta = choice.get("delta").unwrap_or(&empty);
    let index = int(delta.get("index")).unwrap_or(0);
    if !state.messages.contains_key(&index) {
        state.finish_reason = None;
    }
    let entry = state
        .messages
        .entry(index)
        .or_insert_with(|| json!({ "content": [], "tool_calls": [] }));
    for key in ["role", "tool_call_id", "metadata"] {
        if let Some(v) = delta.get(key) {
            entry[key] = v.clone();
        }
    }
    match delta.get("content") {
        None | Some(Value::Null) => {}
        Some(Value::String(s)) => push(entry, "content", json!({ "type": "text", "text": s })),
        Some(Value::Array(parts)) => {
            for part in parts {
                push(entry, "content", part.clone());
            }
        }
        Some(other) => push(entry, "content", other.clone()),
    }
    for call in delta
        .get("tool_calls")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let at = int(call.get("index")).unwrap_or(0).max(0) as usize;
        let calls = entry["tool_calls"]
            .as_array_mut()
            .expect("tool_calls is an array");
        while calls.len() <= at {
            calls.push(Value::Null);
        }
        if calls[at].is_null() {
            calls[at] = json!({ "function": { "arguments": "" } });
        }
        let target = &mut calls[at];
        for key in ["id", "type", "metadata"] {
            if let Some(v) = call.get(key) {
                target[key] = v.clone();
            }
        }
        if let Some(name) = call.pointer("/function/name").filter(|n| !n.is_null()) {
            target["function"]["name"] = name.clone();
        }
        let arguments = format!(
            "{}{}",
            target
                .pointer("/function/arguments")
                .and_then(Value::as_str)
                .unwrap_or(""),
            call.pointer("/function/arguments")
                .and_then(Value::as_str)
                .unwrap_or("")
        );
        target["function"]["arguments"] = arguments.into();
    }
    if let Some(reason) = choice.get("finish_reason").filter(|r| !r.is_null()) {
        state.finish_reason = Some(reason.clone());
    }
    let mut chunk = Message::chunk();
    chunk.model = str_of(data.get("model"));
    if entry.get("role").and_then(Value::as_str) == Some("assistant") {
        let (content, thinking) = assistant_delta(delta.get("content"));
        chunk.content = content;
        chunk.thinking = Thinking::build(thinking, None);
    }
    chunk
}

/// `extract_content_and_thinking` on an assistant delta.
fn assistant_delta(content: Option<&Value>) -> (Option<String>, Option<String>) {
    match content {
        Some(Value::String(s)) => (Some(s.clone()), None),
        Some(Value::Array(blocks)) => {
            let text: String = blocks
                .iter()
                .filter(|b| b.get("type").and_then(Value::as_str) == Some("text"))
                .filter_map(|b| b.get("text").and_then(Value::as_str))
                .collect();
            let mut result = Content::default();
            for block in blocks
                .iter()
                .filter(|b| b.get("type").and_then(Value::as_str) == Some("thinking"))
            {
                parse_conversation_parts(Some(&Value::Array(vec![block.clone()])), &mut result);
            }
            ((!text.is_empty()).then_some(text), result.thinking())
        }
        _ => (None, None),
    }
}

fn push(entry: &mut Value, key: &str, value: Value) {
    if let Some(list) = entry.get_mut(key).and_then(Value::as_array_mut) {
        list.push(value);
    }
}

/// `stream_response`'s ending: the accumulated response parsed, or an error when no completion
/// finished.
pub(crate) fn finish_multi_stream(state: &MultiStream, raw: RawResponse) -> Result<Message> {
    let Some(finish_reason) = &state.finish_reason else {
        return Err(Error::Api(
            "Mistral tool stream ended before completion".into(),
            None,
        ));
    };
    let sum = |key: &str| {
        state
            .usage
            .iter()
            .map(|(_, u)| int(u.get(key)).unwrap_or(0))
            .sum::<i64>()
    };
    let cached: i64 = state
        .usage
        .iter()
        .map(|(_, u)| int(u.pointer("/prompt_tokens_details/cached_tokens")).unwrap_or(0))
        .sum();
    let messages: Vec<Value> = state
        .messages
        .values()
        .map(|m| {
            let mut m = m.clone();
            if m.get("tool_calls")
                .and_then(Value::as_array)
                .is_some_and(Vec::is_empty)
                && let Some(o) = m.as_object_mut()
            {
                o.remove("tool_calls");
            }
            if m.get("role").and_then(Value::as_str) == Some("tool")
                && let Some(parts) = m.get("content").and_then(Value::as_array)
                && parts
                    .iter()
                    .all(|p| p.get("type").and_then(Value::as_str) == Some("text"))
            {
                let text: String = parts
                    .iter()
                    .filter_map(|p| p.get("text").and_then(Value::as_str))
                    .collect();
                m["content"] = text.into();
            }
            m
        })
        .collect();
    let response = json!({
        "model": state.model.clone().unwrap_or(Value::Null),
        "usage": {
            "prompt_tokens": sum("prompt_tokens"), "completion_tokens": sum("completion_tokens"),
            "total_tokens": sum("total_tokens"), "prompt_tokens_details": { "cached_tokens": cached },
        },
        "choices": [{ "messages": messages, "finish_reason": finish_reason }],
    });
    parse_multi_message(&response, Some(raw))?
        .ok_or_else(|| Error::Api("Mistral tool stream ended before completion".into(), None))
}

/// The final chunk `stream_response` yields after the stream: the whole message's fields.
pub(crate) fn final_chunk(message: &Message) -> Message {
    let mut chunk = Message::chunk();
    chunk.tokens = message.tokens.clone();
    chunk.model = message.model.clone();
    chunk.citations = message.citations.clone();
    chunk.server_tool_calls = message.server_tool_calls.clone();
    chunk.tool_calls = message.tool_calls.clone();
    chunk.raw_content = message.raw_content.clone();
    chunk.attachments = message.attachments.clone();
    chunk.finish_reason = message.finish_reason.clone();
    chunk
}

// ---- SHA-256 ------------------------------------------------------------------------------------

/// `Digest::SHA256.hexdigest`, for the replay ids of hosted executions (FIPS 180-4).
fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut message = data.to_vec();
    let bits = (data.len() as u64).wrapping_mul(8);
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&bits.to_be_bytes());
    for block in message.chunks(64) {
        let mut w = [0u32; 64];
        for (i, word) in block.chunks(4).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let mut v = h;
        for i in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
            let t1 = v[7]
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(maj);
            v = [
                t1.wrapping_add(t2),
                v[0],
                v[1],
                v[2],
                v[3].wrapping_add(t1),
                v[4],
                v[5],
                v[6],
            ];
        }
        for (a, b) in h.iter_mut().zip(v) {
            *a = a.wrapping_add(b);
        }
    }
    h.iter().map(|x| format!("{x:08x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_matches_ruby_digest() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        // Ruby: Digest::SHA256.hexdigest("tool_exec_01a0b16815fb72b69ceca436ae298639")[0, 9]
        assert_eq!(
            &sha256_hex(b"tool_exec_01a0b16815fb72b69ceca436ae298639")[..9],
            "2e5ec8371"
        );
    }
}
