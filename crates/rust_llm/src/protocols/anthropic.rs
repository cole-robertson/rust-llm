//! Port of `lib/ruby_llm/protocols/anthropic{,/chat,/tools,/media,/streaming,/models}.rb`.

use serde_json::{Map, Value, json};

use super::{Request, ToolCalls, ToolChoice, int, normalize_finish_reason, str_of, tool_call_map};
use crate::attachment::{Attachment, AttachmentType};
use crate::error::{Error, Result};
use crate::message::{Citation, Message, RawResponse, Role, ServerToolCall, Thinking, ToolArguments, ToolCall};
use crate::model::Model;
use crate::thinking::ThinkingConfig;
use crate::tool::{Tool, tool_schema};

pub const DEFAULT_MAX_OUTPUT_TOKENS: i64 = 4096;

const FINISH_REASONS: &[(&str, &str)] = &[
    ("end_turn", "stop"),
    ("stop_sequence", "stop"),
    ("max_tokens", "max_tokens"),
    ("model_context_window_exceeded", "max_tokens"),
    ("tool_use", "tool_calls"),
    ("refusal", "content_filter"),
];

const EFFORT_BUDGETS: &[(&str, i64)] = &[("low", 1024), ("medium", 40_000), ("high", 63_999)];

pub fn render_payload(req: &Request) -> Result<Value> {
    let (system, chat): (Vec<&Message>, Vec<&Message>) = req.messages.iter().partition(|m| m.role == Role::System);
    let mut system_content = Vec::new();
    for msg in &system {
        let mut blocks = format_content(msg.content.as_deref(), &msg.attachments)?;
        if msg.cache_until_here {
            inject_cache_control(&mut blocks);
        }
        system_content.extend(blocks);
    }

    let max_tokens = req.max_output_tokens.or(req.model.max_output_tokens).unwrap_or(DEFAULT_MAX_OUTPUT_TOKENS);
    let mut payload = Map::new();
    payload.insert("model".into(), req.model.id.clone().into());
    payload.insert("messages".into(), Value::Array(format_messages(&chat)?));
    payload.insert("stream".into(), req.stream.into());
    payload.insert("max_tokens".into(), max_tokens.into());
    add_thinking_fields(&mut payload, req.thinking, req.model, max_tokens);

    if !req.tools.is_empty() {
        let tools: Vec<Value> = req.tools.iter().map(|t| function_for(t.as_ref())).collect();
        payload.insert("tools".into(), Value::Array(tools));
        if req.tool_prefs.choice.is_some() || req.tool_prefs.calls.is_some() {
            payload.insert("tool_choice".into(), build_tool_choice(req.tool_prefs));
        }
    }
    if !system_content.is_empty() {
        payload.insert("system".into(), Value::Array(system_content));
    }
    if let Some(t) = req.temperature {
        payload.insert("temperature".into(), t.into());
    }
    if let Some(schema) = req.schema {
        let mut normalized = schema.schema.clone();
        if let Some(o) = normalized.as_object_mut() {
            o.remove("strict");
        }
        let output = payload.entry("output_config").or_insert_with(|| json!({}));
        output["format"] = json!({ "type": "json_schema", "schema": normalized });
    }
    Ok(Value::Object(payload))
}

fn format_messages(messages: &[&Message]) -> Result<Vec<Value>> {
    let mut rendered = Vec::new();
    let mut tool_results: Vec<Value> = Vec::new();
    for msg in messages {
        if msg.is_tool_result() {
            tool_results.push(format_tool_result_block(msg)?);
            if msg.cache_until_here {
                inject_cache_control(&mut tool_results);
            }
            continue;
        }
        if !tool_results.is_empty() {
            rendered.push(json!({ "role": "user", "content": std::mem::take(&mut tool_results) }));
        }
        let formatted = format_message(msg)?;
        if formatted["content"].as_array().is_some_and(|c| !c.is_empty()) {
            rendered.push(formatted);
        }
    }
    if !tool_results.is_empty() {
        rendered.push(json!({ "role": "user", "content": tool_results }));
    }
    Ok(rendered)
}

fn format_message(msg: &Message) -> Result<Value> {
    if msg.role == Role::Assistant
        && let Some(raw) = &msg.raw_content {
            let mut blocks = raw.as_array().cloned().unwrap_or_default();
            if msg.cache_until_here {
                inject_cache_control(&mut blocks);
            }
            return Ok(json!({ "role": "assistant", "content": blocks }));
        }
    if let Some(calls) = msg.tool_calls.as_ref().filter(|c| !c.is_empty()) {
        let mut blocks = format_thinking_blocks(msg);
        if !msg.content().is_empty() {
            blocks.extend(format_content(msg.content.as_deref(), &msg.attachments)?);
        }
        for call in calls.values() {
            blocks.push(json!({
                "type": "tool_use",
                "id": call.id,
                "name": call.name,
                "input": Value::Object(call.arguments()),
            }));
        }
        if msg.cache_until_here {
            inject_cache_control(&mut blocks);
        }
        return Ok(json!({ "role": "assistant", "content": blocks }));
    }
    let mut blocks = if msg.role == Role::Assistant { format_thinking_blocks(msg) } else { Vec::new() };
    blocks.extend(format_content(msg.content.as_deref(), &msg.attachments)?);
    if msg.cache_until_here {
        inject_cache_control(&mut blocks);
    }
    let role = match msg.role {
        Role::Tool | Role::User => "user",
        _ => "assistant",
    };
    Ok(json!({ "role": role, "content": blocks }))
}

fn format_thinking_blocks(msg: &Message) -> Vec<Value> {
    if let Some(blocks) = msg.raw_reasoning.as_ref().and_then(|r| r.get("anthropic")).and_then(Value::as_array) {
        return blocks.clone();
    }
    let Some(thinking) = &msg.thinking else { return Vec::new() };
    if let Some(text) = &thinking.text {
        let mut block = json!({ "type": "thinking", "thinking": text });
        if let Some(sig) = &thinking.signature {
            block["signature"] = sig.clone().into();
        }
        vec![block]
    } else if let Some(sig) = &thinking.signature {
        vec![json!({ "type": "redacted_thinking", "data": sig })]
    } else {
        Vec::new()
    }
}

fn inject_cache_control(blocks: &mut [Value]) {
    if let Some(Value::Object(last)) = blocks.last_mut() {
        last.entry("cache_control").or_insert_with(|| json!({ "type": "ephemeral" }));
    }
}

fn format_tool_result_block(msg: &Message) -> Result<Value> {
    let mut content = msg.content.clone().filter(|c| !c.is_empty());
    if content.is_none() && msg.attachments.is_empty() {
        content = Some("(no output)".into());
    }
    Ok(json!({
        "type": "tool_result",
        "tool_use_id": msg.tool_call_id,
        "content": format_content(content.as_deref(), &msg.attachments)?,
    }))
}

/// `Anthropic::Media.format_content`.
pub fn format_content(content: Option<&str>, attachments: &[Attachment]) -> Result<Vec<Value>> {
    let mut parts = Vec::new();
    if let Some(text) = content.filter(|t| !t.is_empty()) {
        parts.push(json!({ "type": "text", "text": text }));
    }
    for a in attachments {
        let part = match a.kind() {
            AttachmentType::Image => match a.url() {
                Some(url) => json!({ "type": "image", "source": { "type": "url", "url": url } }),
                None => json!({ "type": "image", "source": {
                    "type": "base64", "media_type": a.mime_type, "data": a.encoded()?
                }}),
            },
            AttachmentType::Pdf => match a.url() {
                Some(url) => json!({ "type": "document", "source": { "type": "url", "url": url } }),
                None => json!({ "type": "document", "source": {
                    "type": "base64", "media_type": a.mime_type, "data": a.encoded()?
                }}),
            },
            AttachmentType::Text => json!({ "type": "text", "text": a.for_llm()? }),
            _ => return Err(Error::UnsupportedAttachment(unsupported(&a.mime_type))),
        };
        parts.push(part);
    }
    Ok(parts)
}

pub(crate) fn unsupported(mime: &str) -> String {
    format!("Unsupported attachment type: {mime}. Consider using a model that supports this attachment type.")
}

fn function_for(tool: &dyn Tool) -> Value {
    let schema = tool_schema(tool).unwrap_or_else(|| {
        json!({ "type": "object", "properties": {}, "required": [], "additionalProperties": false, "strict": true })
    });
    let mut declaration = json!({
        "name": tool.name(),
        "description": tool.description(),
        "input_schema": schema,
    });
    let opts = tool.provider_options();
    if !opts.is_empty() {
        super::deep_merge(&mut declaration, &Value::Object(opts));
    }
    declaration
}

fn build_tool_choice(prefs: &super::ToolPrefs) -> Value {
    let choice = prefs.choice.clone().unwrap_or(ToolChoice::Auto);
    let mut tc = Map::new();
    let kind = match &choice {
        ToolChoice::Auto => "auto",
        ToolChoice::None => "none",
        ToolChoice::Required => "any",
        ToolChoice::Tool(_) => "tool",
    };
    tc.insert("type".into(), kind.into());
    if let ToolChoice::Tool(name) = &choice {
        tc.insert("name".into(), name.clone().into());
    }
    if kind != "none"
        && let Some(calls) = prefs.calls {
            tc.insert("disable_parallel_tool_use".into(), (calls == ToolCalls::One).into());
        }
    Value::Object(tc)
}

fn add_thinking_fields(payload: &mut Map<String, Value>, thinking: Option<&ThinkingConfig>, model: &Model, max_tokens: i64) {
    let Some(thinking) = thinking.filter(|t| t.is_enabled()) else { return };
    if thinking.enabled == Some(false) {
        payload.insert("thinking".into(), json!({ "type": "disabled" }));
        return;
    }
    let effort = thinking.effort.clone().filter(|e| !e.is_empty());
    if effort.as_deref() == Some("none") {
        return;
    }
    let mode = if thinking.enabled == Some(true) {
        Some(json!({ "type": "adaptive" }))
    } else {
        let budget = thinking.budget.or_else(|| effort_budget(effort.as_deref(), model, max_tokens));
        let mut mode = if let Some(budget) = budget {
            Some(json!({ "type": "enabled", "budget_tokens": budget }))
        } else if thinking.display.is_some()
            || (effort.is_some() && model.reasoning_option("effort").is_some() && model.reasoning_option("budget_tokens").is_none())
        {
            Some(json!({ "type": "adaptive" }))
        } else {
            None
        };
        if let (Some(m), Some(d)) = (mode.as_mut(), thinking.display) {
            m["display"] = d.as_str().into();
        }
        mode
    };
    if let Some(mode) = mode {
        payload.insert("thinking".into(), mode);
    }
    if let Some(effort) = effort {
        let output = payload.entry("output_config").or_insert_with(|| json!({}));
        output["effort"] = effort.into();
    }
}

fn effort_budget(effort: Option<&str>, model: &Model, max_tokens: i64) -> Option<i64> {
    let effort = effort?;
    let option = model.reasoning_option("budget_tokens")?;
    let budget = EFFORT_BUDGETS.iter().find(|(e, _)| *e == effort).map(|(_, b)| *b).unwrap_or(63_999);
    let minimum = option.get("min").and_then(Value::as_i64).unwrap_or(0).max(1);
    Some(budget.clamp(minimum, (max_tokens - 1).max(minimum)))
}

fn aggregate_usage(usage: Option<&Value>) -> Map<String, Value> {
    let Some(Value::Object(usage)) = usage else { return Map::new() };
    let mut usage = usage.clone();
    if let Some(iterations) = usage.get("iterations").and_then(Value::as_array).filter(|i| !i.is_empty()).cloned() {
        for key in ["input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"] {
            let sum: i64 = iterations.iter().map(|it| it.get(key).and_then(Value::as_i64).unwrap_or(0)).sum();
            usage.insert(key.into(), sum.into());
        }
    }
    usage
}

fn cache_write(usage: &Map<String, Value>) -> Option<i64> {
    if let Some(v) = int(usage.get("cache_creation_input_tokens")) {
        return Some(v);
    }
    let breakdown = usage.get("cache_creation")?.as_object()?;
    Some(breakdown.values().filter_map(Value::as_i64).sum())
}

fn parse_citation(data: &Value, text: Option<String>, start: Option<i64>, end: Option<i64>) -> Citation {
    let url = str_of(data.get("url")).or_else(|| str_of(data.get("source")));
    let url = url.filter(|u| u.to_lowercase().starts_with("http://") || u.to_lowercase().starts_with("https://"));
    Citation {
        url,
        title: str_of(data.get("document_title")).or_else(|| str_of(data.get("title"))),
        cited_text: str_of(data.get("cited_text")),
        text,
        start_index: start,
        end_index: end,
        source_index: int(data.get("document_index")).or_else(|| int(data.get("search_result_index"))),
        start_page: int(data.get("start_page_number")),
        end_page: int(data.get("end_page_number")).map(|p| p - 1),
        ..Default::default()
    }
}

fn is_server_tool_block(block: &Value) -> bool {
    let kind = block.get("type").and_then(Value::as_str).unwrap_or("");
    matches!(kind, "server_tool_use" | "mcp_tool_use" | "compaction") || kind.ends_with("_tool_result")
}

fn server_tool_calls(blocks: &[Value]) -> Vec<ServerToolCall> {
    blocks
        .iter()
        .filter(|b| is_server_tool_block(b))
        .map(|b| ServerToolCall {
            kind: str_of(b.get("type")).unwrap_or_default(),
            name: str_of(b.get("name")),
            id: str_of(b.get("id")).or_else(|| str_of(b.get("tool_use_id"))),
            input: b.get("input").cloned(),
            result: b.get("content").cloned(),
            raw: b.clone(),
        })
        .collect()
}

fn thinking_blocks(blocks: &[Value]) -> Option<Value> {
    let thinking: Vec<Value> = blocks
        .iter()
        .filter(|b| matches!(b.get("type").and_then(Value::as_str), Some("thinking" | "redacted_thinking")))
        .cloned()
        .collect();
    (!thinking.is_empty()).then(|| json!({ "anthropic": thinking }))
}

pub fn parse_completion_body(data: &Value, raw: RawResponse) -> Result<Message> {
    let blocks = data.get("content").and_then(Value::as_array).cloned().unwrap_or_default();
    let mut text = String::new();
    let mut citations = Vec::new();
    for block in blocks.iter().filter(|b| b.get("type").and_then(Value::as_str) == Some("text")) {
        let block_text = block.get("text").and_then(Value::as_str).unwrap_or("");
        let start = text.chars().count() as i64;
        let end = start + block_text.chars().count() as i64;
        for c in block.get("citations").and_then(Value::as_array).into_iter().flatten() {
            citations.push(parse_citation(c, Some(block_text.to_string()), Some(start), Some(end)));
        }
        text.push_str(block_text);
    }
    let thinking_text: Vec<String> = blocks
        .iter()
        .filter(|b| b.get("type").and_then(Value::as_str) == Some("thinking"))
        .map(|b| str_of(b.get("thinking")).or_else(|| str_of(b.get("text"))).unwrap_or_default())
        .collect();
    let signature_block = blocks
        .iter()
        .find(|b| b.get("type").and_then(Value::as_str) == Some("thinking"))
        .or_else(|| blocks.iter().find(|b| b.get("type").and_then(Value::as_str) == Some("redacted_thinking")));
    let signature = signature_block.and_then(|b| str_of(b.get("signature")).or_else(|| str_of(b.get("data"))));
    let tool_calls: Vec<ToolCall> = blocks
        .iter()
        .filter(|b| b.get("type").and_then(Value::as_str) == Some("tool_use"))
        .map(|b| {
            ToolCall::new(
                str_of(b.get("id")).unwrap_or_default(),
                str_of(b.get("name")).unwrap_or_default(),
                b.get("input").and_then(Value::as_object).cloned().unwrap_or_default(),
            )
        })
        .collect();
    let server_calls = server_tool_calls(&blocks);
    let usage = aggregate_usage(data.get("usage"));

    let mut m = Message::chunk();
    m.content = Some(text);
    m.citations = citations;
    m.thinking = Thinking::build((!thinking_text.is_empty()).then(|| thinking_text.join("")), signature);
    m.raw_reasoning = thinking_blocks(&blocks);
    m.tool_calls = tool_call_map(tool_calls);
    m.raw_content = (!server_calls.is_empty()).then(|| Value::Array(blocks.clone()));
    m.server_tool_calls = server_calls;
    m.tokens.input = int(usage.get("input_tokens"));
    m.tokens.output = int(usage.get("output_tokens"));
    m.tokens.cache_read = int(usage.get("cache_read_input_tokens"));
    m.tokens.cache_write = cache_write(&usage);
    m.tokens.thinking = int(usage.get("output_tokens_details").and_then(|d| d.get("thinking_tokens")))
        .or_else(|| int(usage.get("output_tokens_details").and_then(|d| d.get("reasoning_tokens"))))
        .or_else(|| int(usage.get("thinking_tokens")))
        .or_else(|| int(usage.get("reasoning_tokens")));
    m.tokens.server_tool_use = usage.get("server_tool_use").and_then(Value::as_object).cloned();
    m.finish_reason = normalize_finish_reason(data.get("stop_reason").and_then(Value::as_str), FINISH_REASONS);
    m.model = str_of(data.get("model"));
    m.raw = Some(raw);
    Ok(m.normalized())
}

/// `@stream_blocks` / `@stream_block_json` from `Anthropic::Streaming`.
#[derive(Default)]
pub struct StreamBlocks {
    blocks: Vec<(i64, Value)>,
    json: Vec<(i64, String)>,
    saw_server_block: bool,
}

impl StreamBlocks {
    fn block(&mut self, index: i64) -> Option<&mut Value> {
        self.blocks.iter_mut().find(|(i, _)| *i == index).map(|(_, b)| b)
    }
}

fn track(state: &mut StreamBlocks, data: &Value, delta_type: Option<&str>) {
    let index = data.get("index").and_then(Value::as_i64).unwrap_or(0);
    match data.get("type").and_then(Value::as_str) {
        Some("message_start") => *state = StreamBlocks::default(),
        Some("content_block_start") => {
            let block = data.get("content_block").cloned().unwrap_or_else(|| json!({}));
            if is_server_tool_block(&block) {
                state.saw_server_block = true;
            }
            if block.get("type").and_then(Value::as_str).is_some_and(|t| t.ends_with("tool_use")) {
                state.json.push((index, String::new()));
            }
            state.blocks.push((index, block));
        }
        Some("content_block_delta") => {
            let delta = &data["delta"];
            if delta_type == Some("input_json_delta") {
                if let Some((_, buf)) = state.json.iter_mut().find(|(i, _)| *i == index) {
                    buf.push_str(delta.get("partial_json").and_then(Value::as_str).unwrap_or(""));
                }
                return;
            }
            let Some(block) = state.block(index) else { return };
            let append = |block: &mut Value, key: &str, add: &Value| {
                let prev = block.get(key).and_then(Value::as_str).unwrap_or("").to_string();
                block[key] = format!("{prev}{}", add.as_str().unwrap_or("")).into();
            };
            match delta_type {
                Some("text_delta") => append(block, "text", &delta["text"]),
                Some("thinking_delta") => append(block, "thinking", &delta["thinking"]),
                Some("signature_delta") => block["signature"] = delta["signature"].clone(),
                Some("compaction_delta") => append(block, "content", &delta["content"]),
                Some("citations_delta") => {
                    if !block.get("citations").is_some_and(Value::is_array) {
                        block["citations"] = json!([]);
                    }
                    block["citations"].as_array_mut().unwrap().push(delta["citation"].clone());
                }
                _ => {}
            }
        }
        Some("content_block_stop") => {
            let json = state.json.iter().find(|(i, _)| *i == index).map(|(_, j)| j.clone());
            if let (Some(json), Some(block)) = (json, state.block(index)) {
                let input = if json.is_empty() {
                    block.get("input").cloned().unwrap_or_else(|| json!({}))
                } else {
                    serde_json::from_str(&json).unwrap_or_else(|_| block.get("input").cloned().unwrap_or_else(|| json!({})))
                };
                block["input"] = input;
            }
        }
        _ => {}
    }
}

pub fn build_chunk(state: &mut StreamBlocks, data: &Value) -> Message {
    let delta_type = data.pointer("/delta/type").and_then(Value::as_str);
    track(state, data, delta_type);
    let message_usage = aggregate_usage(data.pointer("/message/usage"));
    let delta_usage = aggregate_usage(data.get("usage"));
    let usage_int = |key: &str| int(message_usage.get(key)).or_else(|| int(delta_usage.get(key)));

    let mut chunk = Message::chunk();
    chunk.model = str_of(data.pointer("/message/model"));
    if delta_type == Some("text_delta") {
        chunk.content = str_of(data.pointer("/delta/text"));
    }
    if delta_type == Some("citations_delta")
        && let Some(c) = data.pointer("/delta/citation") {
            chunk.citations = vec![parse_citation(c, None, None, None)];
        }
    let thinking = (delta_type == Some("thinking_delta")).then(|| str_of(data.pointer("/delta/thinking"))).flatten();
    let signature = (delta_type == Some("signature_delta")).then(|| str_of(data.pointer("/delta/signature"))).flatten();
    chunk.thinking = Thinking::build(thinking, signature);
    chunk.tokens.input = int(message_usage.get("input_tokens"));
    chunk.tokens.output = usage_int("output_tokens");
    chunk.tokens.thinking = int(message_usage.get("output_tokens_details").and_then(|d| d.get("thinking_tokens")))
        .or_else(|| int(delta_usage.get("output_tokens_details").and_then(|d| d.get("thinking_tokens"))));
    chunk.tokens.cache_read = usage_int("cache_read_input_tokens");
    chunk.tokens.cache_write = cache_write(&message_usage).or_else(|| cache_write(&delta_usage));
    chunk.tokens.server_tool_use = data
        .pointer("/message/usage/server_tool_use")
        .or_else(|| data.pointer("/usage/server_tool_use"))
        .and_then(Value::as_object)
        .cloned();
    chunk.finish_reason = normalize_finish_reason(data.pointer("/delta/stop_reason").and_then(Value::as_str), FINISH_REASONS);

    // Tool calls: a start block opens a call keyed by the block index; json deltas append to it.
    let index = data.get("index").and_then(Value::as_i64).map(|i| i.to_string()).unwrap_or_default();
    if delta_type == Some("input_json_delta") {
        let fragment = str_of(data.pointer("/delta/partial_json")).unwrap_or_default();
        chunk.tool_calls = Some(
            [(index, ToolCall::fragment(fragment))]
                .into_iter()
                .collect(),
        );
    } else if let Some(block) = data.get("content_block").filter(|b| b.get("type").and_then(Value::as_str) == Some("tool_use")) {
        let mut call = ToolCall::opening(str_of(block.get("id")).unwrap_or_default(), str_of(block.get("name")).unwrap_or_default(), String::new());
        if let Some(m) = block.get("input").and_then(Value::as_object).filter(|m| !m.is_empty()) {
            call.arguments = ToolArguments::Parsed(m.clone());
        }
        chunk.tool_calls = Some([(index, call)].into_iter().collect());
    }

    if data.get("type").and_then(Value::as_str) == Some("message_stop") {
        let mut blocks = state.blocks.clone();
        blocks.sort_by_key(|(i, _)| *i);
        let blocks: Vec<Value> = blocks.into_iter().map(|(_, b)| b).collect();
        chunk.server_tool_calls = server_tool_calls(&blocks);
        chunk.raw_content = state.saw_server_block.then(|| Value::Array(blocks.clone()));
        chunk.raw_reasoning = thinking_blocks(&blocks);
    }
    chunk
}
