//! Port of `lib/ruby_llm/protocols/gemini/{chat,tools,media,streaming}.rb`.

use serde_json::{Map, Value, json};

use super::{Request, StreamState, ToolChoice, int, normalize_finish_reason, str_of, tool_call_map};
use crate::attachment::{Attachment, AttachmentType, Resolution};
use crate::error::{Error, Result};
use crate::message::{Citation, Message, RawResponse, Role, ServerToolCall, Thinking, ToolCall};
use crate::model::Model;
use crate::tool::{Tool, tool_schema};

const FINISH_REASONS: &[(&str, &str)] = &[
    ("STOP", "stop"),
    ("MAX_TOKENS", "max_tokens"),
    ("SAFETY", "content_filter"),
    ("RECITATION", "content_filter"),
    ("BLOCKLIST", "content_filter"),
    ("PROHIBITED_CONTENT", "content_filter"),
    ("SPII", "content_filter"),
    ("IMAGE_SAFETY", "content_filter"),
    ("IMAGE_RECITATION", "content_filter"),
    ("IMAGE_PROHIBITED_CONTENT", "content_filter"),
    ("MODEL_ARMOR", "content_filter"),
];

pub fn render_payload(req: &Request) -> Result<Value> {
    let chat: Vec<&Message> = req.messages.iter().filter(|m| m.role != Role::System).collect();
    let mut payload = Map::new();
    payload.insert("contents".into(), Value::Array(format_messages(req.model, &chat)?));
    let mut generation = Map::new();

    let mut system_parts = Vec::new();
    for m in req.messages.iter().filter(|m| m.role == Role::System) {
        let text = m.content().to_string();
        system_parts.extend(format_content((!text.is_empty()).then_some(text.as_str()), &m.attachments)?);
    }
    if !system_parts.is_empty() {
        payload.insert("systemInstruction".into(), json!({ "parts": system_parts }));
    }
    if let Some(t) = req.temperature {
        generation.insert("temperature".into(), t.into());
    }
    if let Some(max) = req.max_output_tokens {
        generation.insert("maxOutputTokens".into(), max.into());
    }
    if let Some(schema) = req.schema {
        let mut normalized = schema.schema.clone();
        if let Some(o) = normalized.as_object_mut() {
            o.remove("strict");
        }
        generation.insert("responseMimeType".into(), "application/json".into());
        generation.insert("responseJsonSchema".into(), normalized);
    }
    if let Some(thinking) = req.thinking.filter(|t| t.is_enabled()) {
        let config = if thinking.enabled == Some(false) {
            json!({ "includeThoughts": false, "thinkingBudget": 0 })
        } else {
            let mut c = json!({ "includeThoughts": true });
            if let Some(e) = &thinking.effort {
                c["thinkingLevel"] = e.clone().into();
            }
            if let Some(b) = thinking.budget {
                c["thinkingBudget"] = b.into();
            }
            if thinking.enabled == Some(true) {
                c["thinkingBudget"] = (-1).into();
            }
            c
        };
        generation.insert("thinkingConfig".into(), config);
    }
    payload.insert("generationConfig".into(), Value::Object(generation));
    if !req.tools.is_empty() {
        let declarations: Vec<Value> = req.tools.iter().map(|t| function_declaration(t.as_ref())).collect();
        payload.insert("tools".into(), json!([{ "functionDeclarations": declarations }]));
        if let Some(choice) = &req.tool_prefs.choice {
            let mode = match choice {
                ToolChoice::Auto => "auto",
                ToolChoice::None => "none",
                ToolChoice::Required | ToolChoice::Tool(_) => "any",
            };
            let mut config = json!({ "mode": mode });
            if let ToolChoice::Tool(name) = choice {
                config["allowedFunctionNames"] = json!([name]);
            }
            payload.insert("toolConfig".into(), json!({ "functionCallingConfig": config }));
        }
    }
    Ok(Value::Object(payload))
}

fn function_declaration(tool: &dyn Tool) -> Value {
    let mut d = json!({ "name": tool.name(), "description": tool.description() });
    if let Some(schema) = tool_schema(tool) {
        d["parametersJsonSchema"] = schema;
    }
    let opts = tool.provider_options();
    if !opts.is_empty() {
        super::deep_merge(&mut d, &Value::Object(opts));
    }
    d
}

/// `Gemini::Chat::MessageFormatter`: consecutive tool results collapse into one user turn,
/// ordered like the calls, and named after the function they answer.
fn format_messages(model: &Model, messages: &[&Message]) -> Result<Vec<Value>> {
    let mut out = Vec::new();
    let mut call_names: Vec<(String, String)> = Vec::new();
    let mut i = 0;
    while i < messages.len() {
        let msg = messages[i];
        if msg.role == Role::Tool {
            let mut results = Vec::new();
            while i < messages.len() && messages[i].role == Role::Tool {
                results.push(messages[i]);
                i += 1;
            }
            let order: Vec<String> = call_names.iter().map(|(id, _)| id.clone()).collect();
            let mut indexed: Vec<(usize, &Message)> = results.into_iter().enumerate().collect();
            indexed.sort_by_key(|(idx, m)| {
                let pos = order.iter().position(|id| Some(id.as_str()) == m.tool_call_id.as_deref()).unwrap_or(order.len());
                (pos, *idx)
            });
            let mut parts = Vec::new();
            for (_, m) in indexed {
                let name = m
                    .tool_call_id
                    .as_ref()
                    .and_then(|id| call_names.iter().position(|(cid, _)| cid == id).map(|p| call_names.remove(p).1));
                parts.extend(format_tool_result(model, m, name)?);
            }
            out.push(json!({ "role": "user", "parts": parts }));
            continue;
        }
        if let Some(calls) = &msg.tool_calls {
            for (id, call) in calls.iter() {
                call_names.push((id.clone(), call.name.clone()));
            }
        }
        let role = if msg.role == Role::Assistant { "model" } else { "user" };
        out.push(json!({ "role": role, "parts": format_parts(msg)? }));
        i += 1;
    }
    Ok(out)
}

fn format_parts(msg: &Message) -> Result<Value> {
    if msg.role == Role::Assistant
        && let Some(raw) = &msg.raw_content {
            return Ok(raw.clone());
        }
    if let Some(calls) = msg.tool_calls.as_ref().filter(|c| !c.is_empty()) {
        let mut parts = Vec::new();
        if !msg.content().is_empty() {
            parts.extend(format_content(msg.content.as_deref(), &msg.attachments)?);
        }
        let mut fallback = msg.thinking.as_ref().and_then(|t| t.signature.clone());
        for call in calls.values() {
            let mut part = json!({ "functionCall": { "name": call.name, "args": Value::Object(call.arguments()) } });
            let signature = call.thought_signature.clone().or_else(|| fallback.take());
            if let Some(sig) = signature {
                part["thoughtSignature"] = sig.into();
            }
            parts.push(part);
        }
        return Ok(Value::Array(parts));
    }
    let mut parts = Vec::new();
    if msg.role == Role::Assistant
        && let Some(t) = &msg.thinking {
            let mut part = json!({ "thought": true });
            if let Some(text) = &t.text {
                part["text"] = text.clone().into();
            }
            if let Some(sig) = &t.signature {
                part["thoughtSignature"] = sig.clone().into();
            }
            parts.push(part);
        }
    parts.extend(format_content(msg.content.as_deref(), &msg.attachments)?);
    Ok(Value::Array(parts))
}

fn supports_multimodal_function_responses(model: &Model) -> bool {
    let id = model.id.as_str();
    let Some(rest) = id.strip_prefix("gemini-") else { return false };
    if id.ends_with("-latest") {
        return true;
    }
    let version: String = rest.chars().take_while(|c| c.is_ascii_digit() || *c == '.').collect();
    version.split('.').next().and_then(|m| m.parse::<u32>().ok()).is_some_and(|major| major >= 3)
}

fn format_tool_result(model: &Model, msg: &Message, name: Option<String>) -> Result<Vec<Value>> {
    let name = name.or_else(|| msg.tool_call_id.clone()).unwrap_or_default();
    let mut content = msg.content.clone().filter(|c| !c.is_empty());
    if content.is_none() && msg.attachments.is_empty() {
        content = Some("(no output)".into());
    }
    let mut response = json!({
        "name": name,
        "response": { "name": name, "content": format_content(content.as_deref(), &[])? },
    });
    let parts: Vec<Value> = msg.attachments.iter().map(format_attachment).collect::<Result<_>>()?;
    let (media, siblings): (Vec<Value>, Vec<Value>) = if supports_multimodal_function_responses(model) {
        parts.into_iter().partition(|p| p.get("inline_data").is_some())
    } else {
        (Vec::new(), parts)
    };
    if !media.is_empty() {
        response["parts"] = Value::Array(media);
    }
    let mut out = vec![json!({ "functionResponse": response })];
    out.extend(siblings);
    Ok(out)
}

fn format_attachment(a: &Attachment) -> Result<Value> {
    match a.kind() {
        AttachmentType::Text => Ok(json!({ "text": a.for_llm()? })),
        AttachmentType::Document | AttachmentType::Unknown => {
            Err(Error::UnsupportedAttachment(super::anthropic::unsupported(&a.mime_type)))
        }
        _ => Ok(json!({ "inline_data": { "mime_type": a.mime_type, "data": a.encoded()? } })),
    }
}

/// `Gemini::Media.format_content`.
fn format_content(content: Option<&str>, attachments: &[Attachment]) -> Result<Vec<Value>> {
    let mut parts = Vec::new();
    if let Some(text) = content {
        parts.push(json!({ "text": text }));
    }
    for a in attachments {
        let mut part = format_attachment(a)?;
        if let Some(res) = a.resolution
            && matches!(a.kind(), AttachmentType::Image | AttachmentType::Video | AttachmentType::Pdf) {
                let level = match res {
                    Resolution::Low => "LOW",
                    Resolution::Medium => "MEDIUM",
                    Resolution::High => "HIGH",
                    Resolution::UltraHigh if a.kind() != AttachmentType::Image => "HIGH",
                    Resolution::UltraHigh => "ULTRA_HIGH",
                };
                part["media_resolution"] = json!({ "level": format!("MEDIA_RESOLUTION_{level}") });
            }
        parts.push(part);
    }
    Ok(parts)
}

fn is_server_tool_part(part: &Value) -> bool {
    part.get("executableCode").is_some() || part.get("codeExecutionResult").is_some()
}

fn part_server_calls(parts: &[Value]) -> Vec<ServerToolCall> {
    parts
        .iter()
        .filter(|p| is_server_tool_part(p))
        .map(|p| ServerToolCall {
            kind: if p.get("executableCode").is_some() { "executable_code" } else { "code_execution_result" }.into(),
            name: None,
            id: None,
            input: p.get("executableCode").cloned(),
            result: p.get("codeExecutionResult").cloned(),
            raw: p.clone(),
        })
        .collect()
}

fn metadata_server_calls(data: &Value) -> Vec<ServerToolCall> {
    let candidate = data.pointer("/candidates/0").cloned().unwrap_or_else(|| json!({}));
    let mut calls = Vec::new();
    if let Some(queries) = candidate.pointer("/groundingMetadata/webSearchQueries").filter(|q| q.as_array().is_some_and(|a| !a.is_empty())) {
        calls.push(ServerToolCall {
            kind: "google_search".into(),
            name: None,
            id: None,
            input: Some(json!({ "queries": queries })),
            result: None,
            raw: json!({ "webSearchQueries": queries }),
        });
    }
    if let Some(meta) = candidate.get("urlContextMetadata") {
        calls.push(ServerToolCall { kind: "url_context".into(), name: None, id: None, input: None, result: Some(meta.clone()), raw: meta.clone() });
    }
    calls
}

fn byte_to_char(content: Option<&str>, byte_index: Option<i64>) -> Option<i64> {
    let (content, idx) = (content?, byte_index? as usize);
    let prefix = content.as_bytes().get(..idx.min(content.len()))?;
    Some(String::from_utf8_lossy(prefix).chars().count() as i64)
}

fn extract_citations(data: &Value, content: Option<&str>) -> Vec<Citation> {
    let Some(meta) = data.pointer("/candidates/0/groundingMetadata") else { return Vec::new() };
    let chunks = meta.get("groundingChunks").and_then(Value::as_array).cloned().unwrap_or_default();
    let source = |i: usize| chunks.get(i).and_then(|c| c.get("web").or_else(|| c.get("retrievedContext"))).cloned();
    let supports = meta.get("groundingSupports").and_then(Value::as_array).cloned().unwrap_or_default();
    if supports.is_empty() {
        return (0..chunks.len())
            .filter_map(|i| {
                let s = source(i)?;
                Some(Citation { url: str_of(s.get("uri")), title: str_of(s.get("title")), source_index: Some(i as i64), ..Default::default() })
            })
            .collect();
    }
    let mut out = Vec::new();
    for support in supports {
        let segment = support.get("segment").cloned().unwrap_or_else(|| json!({}));
        let end = int(segment.get("endIndex"));
        let start = int(segment.get("startIndex")).or(end.map(|_| 0));
        for idx in support.get("groundingChunkIndices").and_then(Value::as_array).into_iter().flatten() {
            let Some(i) = idx.as_u64().map(|i| i as usize) else { continue };
            let Some(s) = source(i) else { continue };
            out.push(Citation {
                url: str_of(s.get("uri")),
                title: str_of(s.get("title")),
                text: str_of(segment.get("text")),
                start_index: byte_to_char(content, start),
                end_index: byte_to_char(content, end),
                source_index: Some(i as i64),
                ..Default::default()
            });
        }
    }
    out
}

fn thoughts(parts: &[Value]) -> (Option<String>, Option<String>) {
    let text: String = parts
        .iter()
        .filter(|p| p.get("thought").and_then(Value::as_bool) == Some(true))
        .filter_map(|p| p.get("text").and_then(Value::as_str))
        .collect();
    let signature = parts.iter().find_map(|p| {
        str_of(p.get("thoughtSignature"))
            .or_else(|| str_of(p.get("thought_signature")))
            .or_else(|| str_of(p.pointer("/functionCall/thoughtSignature")))
            .or_else(|| str_of(p.pointer("/functionCall/thought_signature")))
    });
    ((!text.is_empty()).then_some(text), signature)
}

fn extract_tool_calls(parts: &[Value]) -> Vec<ToolCall> {
    parts
        .iter()
        .filter_map(|p| {
            let f = p.get("functionCall")?;
            let mut call = ToolCall::new(
                uuid::Uuid::new_v4().to_string(),
                str_of(f.get("name")).unwrap_or_default(),
                f.get("args").and_then(Value::as_object).cloned().unwrap_or_default(),
            );
            call.thought_signature = str_of(p.get("thoughtSignature")).or_else(|| str_of(p.get("thought_signature")));
            Some(call)
        })
        .collect()
}

fn usage(message: &mut Message, data: &Value, streaming: bool) {
    let meta = data.get("usageMetadata");
    let cached = int(meta.and_then(|m| m.get("cachedContentTokenCount")));
    message.tokens.input = int(meta.and_then(|m| m.get("promptTokenCount"))).map(|p| (p - cached.unwrap_or(0)).max(0));
    let candidates = int(meta.and_then(|m| m.get("candidatesTokenCount"))).unwrap_or(0);
    let thought = int(meta.and_then(|m| m.get("thoughtsTokenCount")));
    let total = candidates + thought.unwrap_or(0);
    message.tokens.output = if streaming { (total > 0).then_some(total) } else { Some(total) };
    message.tokens.cache_read = cached;
    message.tokens.thinking = thought;
}

pub fn parse_completion_body(model: &Model, data: &Value, raw: RawResponse) -> Result<Message> {
    let parts = data.pointer("/candidates/0/content/parts").and_then(Value::as_array).cloned().unwrap_or_default();
    let mut text = String::new();
    let mut attachments = Vec::new();
    for (i, p) in parts.iter().filter(|p| p.get("thought").and_then(Value::as_bool) != Some(true)).enumerate() {
        if let Some(t) = p.get("text").and_then(Value::as_str) {
            text.push_str(t);
        } else if let Some(inline) = p.get("inlineData") {
            use base64::Engine;
            if let Some(bytes) = inline.get("data").and_then(Value::as_str).and_then(|d| base64::engine::general_purpose::STANDARD.decode(d).ok()) {
                let mime = inline.get("mimeType").and_then(Value::as_str).unwrap_or("application/octet-stream");
                let ext = mime.rsplit('/').next().unwrap_or("bin");
                attachments.push(Attachment::from_bytes(bytes, format!("gemini_output_{i}.{ext}"), Some(mime)));
            }
        }
    }
    let content = if parts.is_empty() || text.is_empty() { Some(String::new()) } else { Some(text) };
    let (thought_text, signature) = thoughts(&parts);
    let mut server_calls = part_server_calls(&parts);
    server_calls.extend(metadata_server_calls(data));

    let mut m = Message::chunk();
    m.citations = extract_citations(data, content.as_deref());
    m.content = content;
    m.attachments = attachments;
    m.thinking = Thinking::build(thought_text, signature);
    m.tool_calls = tool_call_map(extract_tool_calls(&parts));
    m.raw_content = parts.iter().any(is_server_tool_part).then(|| Value::Array(parts.clone()));
    m.server_tool_calls = server_calls;
    usage(&mut m, data, false);
    m.finish_reason = normalize_finish_reason(
        data.pointer("/candidates/0/finishReason").or_else(|| data.pointer("/promptFeedback/blockReason")).and_then(Value::as_str),
        FINISH_REASONS,
    );
    m.model = str_of(data.get("modelVersion")).or_else(|| Some(model.id.clone()));
    m.raw = Some(raw);
    Ok(m.normalized())
}

pub fn build_chunk(state: &mut StreamState, data: &Value) -> Message {
    let parts = data.pointer("/candidates/0/content/parts").and_then(Value::as_array).cloned().unwrap_or_default();
    state.gemini_parts.extend(parts.iter().cloned());
    let text: String = parts
        .iter()
        .filter(|p| p.get("thought").and_then(Value::as_bool) != Some(true))
        .filter_map(|p| p.get("text").and_then(Value::as_str))
        .collect();
    let (thought_text, signature) = thoughts(&parts);
    let mut m = Message::chunk();
    m.model = str_of(data.get("modelVersion"));
    m.content = (!text.is_empty()).then_some(text);
    m.citations = extract_citations(data, None);
    m.thinking = Thinking::build(thought_text, signature);
    usage(&mut m, data, true);
    m.finish_reason = normalize_finish_reason(
        data.pointer("/candidates/0/finishReason").or_else(|| data.pointer("/promptFeedback/blockReason")).and_then(Value::as_str),
        FINISH_REASONS,
    );
    let calls = extract_tool_calls(&parts);
    if !calls.is_empty() {
        m.tool_calls = Some(calls.into_iter().map(|c| (c.id.clone(), c)).collect());
    }
    if data.pointer("/candidates/0/finishReason").is_some() {
        let mut calls = part_server_calls(&state.gemini_parts);
        calls.extend(metadata_server_calls(data));
        if !calls.is_empty() {
            m.raw_content = state.gemini_parts.iter().any(is_server_tool_part).then(|| Value::Array(state.gemini_parts.clone()));
            m.server_tool_calls = calls;
        }
    }
    m
}
