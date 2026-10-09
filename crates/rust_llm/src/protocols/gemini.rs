//! Port of `lib/ruby_llm/protocols/gemini/{chat,tools,media,streaming}.rb`.

use serde_json::{Map, Value, json};

use super::{
    Caching, Request, StreamState, ToolChoice, int, normalize_finish_reason, str_of, tool_call_map,
};
use crate::attachment::{Attachment, AttachmentType, Resolution};
use crate::error::{Error, Result};
use crate::message::{Citation, Message, RawResponse, Role, ServerToolCall, Thinking, ToolCall};
use crate::model::Model;
use crate::tool::{Tool, tool_schema};

pub(crate) const FINISH_REASONS: &[(&str, &str)] = &[
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
    let chat: Vec<&Message> = req
        .messages
        .iter()
        .filter(|m| m.role != Role::System)
        .collect();
    let mut payload = Map::new();
    payload.insert(
        "contents".into(),
        Value::Array(format_messages(req.model, req.provider.slug(), &chat)?),
    );
    // `{ contents:, generationConfig: {} }` first, then `systemInstruction`: key order matches
    // the payload RubyLLM sends (and the `payload keys` a RequestShape shows).
    payload.insert("generationConfig".into(), json!({}));
    let mut generation = Map::new();

    let mut system_parts = Vec::new();
    for m in req.messages.iter().filter(|m| m.role == Role::System) {
        let text = m.content().to_string();
        system_parts.extend(format_content(
            (!text.is_empty()).then_some(text.as_str()),
            &m.attachments,
        )?);
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
        let declarations: Vec<Value> = req
            .tools
            .iter()
            .map(|t| function_declaration(t.as_ref()))
            .collect();
        payload.insert(
            "tools".into(),
            json!([{ "functionDeclarations": declarations }]),
        );
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
            payload.insert(
                "toolConfig".into(),
                json!({ "functionCallingConfig": config }),
            );
        }
    }
    // `with_caching(id:)` attaches an explicit cache; Gemini caches prefixes implicitly otherwise.
    if let Some(id) = Caching::options(req.caching)
        .and_then(|o| o.get("id"))
        .and_then(Value::as_str)
    {
        let name = if id.contains('/') {
            id.to_string()
        } else {
            format!("cachedContents/{id}")
        };
        payload.insert("cachedContent".into(), name.into());
    }
    // `maybe_log_implicit_caching_note` (`protocols/gemini/chat.rb`).
    let cache_without_id =
        Caching::options(req.caching).is_some_and(|o| o.get("id").is_none_or(Value::is_null));
    if Caching::boundaries(req.caching)
        && (cache_without_id || req.messages.iter().any(|m| m.cache_until_here))
    {
        tracing::debug!(
            "Gemini caches repeated prompt prefixes automatically (implicit caching). For explicit caching, create a cache with RubyLLM.cache and attach it with chat.with_caching(id: cache)."
        );
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
fn format_messages(model: &Model, provider: &str, messages: &[&Message]) -> Result<Vec<Value>> {
    let mut out = Vec::new();
    // Gemini checks signatures only in the current turn, which starts at the last user message.
    let turn_start = messages.iter().rposition(|m| m.role == Role::User);
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
                let pos = order
                    .iter()
                    .position(|id| Some(id.as_str()) == m.tool_call_id.as_deref())
                    .unwrap_or(order.len());
                (pos, *idx)
            });
            let mut parts = Vec::new();
            for (_, m) in indexed {
                let name = m.tool_call_id.as_ref().and_then(|id| {
                    call_names
                        .iter()
                        .position(|(cid, _)| cid == id)
                        .map(|p| call_names.remove(p).1)
                });
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
        let role = if msg.role == Role::Assistant {
            "model"
        } else {
            "user"
        };
        let mut parts = format_parts(msg, provider)?;
        if msg.is_tool_call() && turn_start.is_none_or(|start| i > start) {
            sign_step(&mut parts);
        }
        out.push(json!({ "role": role, "parts": parts }));
        i += 1;
    }
    Ok(out)
}

/// The signature Google documents for a function call Gemini did not make, such as one from
/// another provider (`Gemini::Tools::PLACEHOLDER_SIGNATURE`).
pub(crate) const PLACEHOLDER_SIGNATURE: &str = "skip_thought_signature_validator";

/// `sign_step`: Gemini 3 refuses a step of the current turn whose first function call carries no
/// signature.
fn sign_step(parts: &mut Value) {
    let first = parts
        .as_array_mut()
        .and_then(|p| p.iter_mut().find(|p| p.get("functionCall").is_some()));
    if let Some(first) = first
        && first.get("thoughtSignature").is_none()
    {
        first["thoughtSignature"] = PLACEHOLDER_SIGNATURE.into();
    }
}

/// `format_parts`: a recorded answer replays its parts as Gemini returned them; only an Array is
/// generateContent's own shape (an Interactions answer keeps a whole interaction).
fn format_parts(msg: &Message, provider: &str) -> Result<Value> {
    if msg.role == Role::Assistant
        && let Some(raw @ Value::Array(_)) = &msg.raw_content
    {
        return Ok(raw.clone());
    }
    if let Some(calls) = msg.tool_calls.as_ref().filter(|c| !c.is_empty()) {
        // `format_tool_call`: Gemini signs only the first of parallel calls, so each call goes
        // back with its own signature or none, never one another part carried.
        let mut parts = Vec::new();
        if !msg.content().is_empty() {
            parts.extend(format_content(msg.content.as_deref(), &msg.attachments)?);
        }
        for call in calls.values() {
            let mut part = json!({ "functionCall": { "name": call.name, "args": Value::Object(call.arguments()) } });
            if let Some(sig) = &call.thought_signature {
                part["thoughtSignature"] = sig.clone().into();
            }
            parts.push(part);
        }
        return Ok(Value::Array(parts));
    }
    // `format_message_parts`: the thought summary in an unsigned part of its own, the signature
    // on the answer's last part (an empty text part when the answer has none).
    let mut parts = format_content(msg.content.as_deref(), &msg.attachments)?;
    let Some(thinking) = msg
        .thinking
        .as_ref()
        .filter(|_| msg.role == Role::Assistant)
    else {
        return Ok(Value::Array(parts));
    };
    if let Some(signature) = msg.own_signature(provider) {
        if parts.is_empty() {
            parts.push(json!({ "text": "" }));
        }
        if let Some(last) = parts.last_mut() {
            last["thoughtSignature"] = signature.into();
        }
    }
    if let Some(text) = &thinking.text {
        parts.insert(0, json!({ "thought": true, "text": text }));
    }
    Ok(Value::Array(parts))
}

fn supports_multimodal_function_responses(model: &Model) -> bool {
    let id = model.id.as_str();
    let Some(rest) = id.strip_prefix("gemini-") else {
        return false;
    };
    if id.ends_with("-latest") {
        return true;
    }
    // `id[/\Agemini-(\d+(?:\.\d+)?)(?:-|\z)/, 1]`: a generation followed by `-` or the end.
    let generation = rest.split('-').next().unwrap_or("");
    let mut numbers = generation.split('.');
    let major = numbers
        .next()
        .filter(|m| !m.is_empty() && m.bytes().all(|b| b.is_ascii_digit()));
    let minor_ok = numbers
        .next()
        .is_none_or(|m| !m.is_empty() && m.bytes().all(|b| b.is_ascii_digit()));
    minor_ok
        && numbers.next().is_none()
        && major
            .and_then(|m| m.parse::<u32>().ok())
            .is_some_and(|major| major >= 3)
}

fn format_tool_result(model: &Model, msg: &Message, name: Option<String>) -> Result<Vec<Value>> {
    let name = name
        .or_else(|| msg.tool_call_id.clone())
        .unwrap_or_default();
    let mut content = msg.content.clone().filter(|c| !c.is_empty());
    if content.is_none() && msg.attachments.is_empty() {
        content = Some("(no output)".into());
    }
    let mut response = json!({
        "name": name,
        "response": { "name": name, "content": format_content(content.as_deref(), &[])? },
    });
    let parts: Vec<Value> = msg
        .attachments
        .iter()
        .map(format_attachment)
        .collect::<Result<_>>()?;
    let (media, siblings): (Vec<Value>, Vec<Value>) =
        if supports_multimodal_function_responses(model) {
            parts
                .into_iter()
                .partition(|p| p.get("inline_data").is_some())
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
        AttachmentType::Document | AttachmentType::Unknown => Err(Error::UnsupportedAttachment(
            super::anthropic::unsupported(&a.mime_type),
        )),
        // `format_file_data`: a stored file is referenced by URI.
        _ if a.is_provider_file() => Ok(json!({ "file_data": {
            "mime_type": a.mime_type,
            "file_uri": a.provider_file_uri().or(a.provider_file_id()),
        }})),
        _ => Ok(json!({ "inline_data": { "mime_type": a.mime_type, "data": a.encoded()? } })),
    }
}

/// `Gemini::Media.format_content`.
pub(crate) fn format_content(
    content: Option<&str>,
    attachments: &[Attachment],
) -> Result<Vec<Value>> {
    let mut parts = Vec::new();
    if let Some(text) = content {
        parts.push(json!({ "text": text }));
    }
    for a in attachments {
        let mut part = format_attachment(a)?;
        if let Some(res) = a.resolution
            && matches!(
                a.kind(),
                AttachmentType::Image | AttachmentType::Video | AttachmentType::Pdf
            )
        {
            let level = match res {
                Resolution::Low => "LOW",
                Resolution::Medium => "MEDIUM",
                Resolution::High => "HIGH",
                // `:original` asks for the highest level: ultra high on images, high otherwise.
                Resolution::UltraHigh | Resolution::Original
                    if a.kind() != AttachmentType::Image =>
                {
                    "HIGH"
                }
                Resolution::UltraHigh | Resolution::Original => "ULTRA_HIGH",
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
            kind: if p.get("executableCode").is_some() {
                "executable_code"
            } else {
                "code_execution_result"
            }
            .into(),
            name: None,
            id: None,
            input: p.get("executableCode").cloned(),
            result: p.get("codeExecutionResult").cloned(),
            raw: p.clone(),
            search_suggestions: None,
        })
        .collect()
}

fn metadata_server_calls(data: &Value) -> Vec<ServerToolCall> {
    let candidate = data
        .pointer("/candidates/0")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let mut calls = Vec::new();
    if let Some(queries) = candidate
        .pointer("/groundingMetadata/webSearchQueries")
        .filter(|q| q.as_array().is_some_and(|a| !a.is_empty()))
    {
        calls.push(ServerToolCall {
            kind: "google_search".into(),
            name: None,
            id: None,
            input: Some(json!({ "queries": queries })),
            result: None,
            raw: json!({ "webSearchQueries": queries }),
            search_suggestions: str_of(
                candidate.pointer("/groundingMetadata/searchEntryPoint/renderedContent"),
            ),
        });
    }
    if let Some(meta) = candidate.get("urlContextMetadata") {
        calls.push(ServerToolCall {
            kind: "url_context".into(),
            name: None,
            id: None,
            input: None,
            result: Some(meta.clone()),
            raw: meta.clone(),
            search_suggestions: None,
        });
    }
    calls
}

fn byte_to_char(content: Option<&str>, byte_index: Option<i64>) -> Option<i64> {
    let (content, idx) = (content?, byte_index? as usize);
    let prefix = content.as_bytes().get(..idx.min(content.len()))?;
    Some(String::from_utf8_lossy(prefix).chars().count() as i64)
}

fn extract_citations(data: &Value, content: Option<&str>) -> Vec<Citation> {
    let Some(meta) = data.pointer("/candidates/0/groundingMetadata") else {
        return Vec::new();
    };
    let chunks = meta
        .get("groundingChunks")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let source = |i: usize| {
        chunks
            .get(i)
            .and_then(|c| c.get("web").or_else(|| c.get("retrievedContext")))
            .cloned()
    };
    let supports = meta
        .get("groundingSupports")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if supports.is_empty() {
        return (0..chunks.len())
            .filter_map(|i| {
                let s = source(i)?;
                Some(Citation {
                    url: str_of(s.get("uri")),
                    title: str_of(s.get("title")),
                    source_index: Some(i as i64),
                    ..Default::default()
                })
            })
            .collect();
    }
    let mut out = Vec::new();
    for support in supports {
        let segment = support.get("segment").cloned().unwrap_or_else(|| json!({}));
        let end = int(segment.get("endIndex"));
        let start = int(segment.get("startIndex")).or(end.map(|_| 0));
        for idx in support
            .get("groundingChunkIndices")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let Some(i) = idx.as_u64().map(|i| i as usize) else {
                continue;
            };
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
                f.get("args")
                    .and_then(Value::as_object)
                    .cloned()
                    .unwrap_or_default(),
            );
            call.thought_signature =
                str_of(p.get("thoughtSignature")).or_else(|| str_of(p.get("thought_signature")));
            Some(call)
        })
        .collect()
}

fn usage(message: &mut Message, data: &Value, streaming: bool) {
    let meta = data.get("usageMetadata");
    let cached = int(meta.and_then(|m| m.get("cachedContentTokenCount")));
    message.tokens.input =
        int(meta.and_then(|m| m.get("promptTokenCount"))).map(|p| (p - cached.unwrap_or(0)).max(0));
    let candidates = int(meta.and_then(|m| m.get("candidatesTokenCount"))).unwrap_or(0);
    let thought = int(meta.and_then(|m| m.get("thoughtsTokenCount")));
    let total = candidates + thought.unwrap_or(0);
    message.tokens.output = if streaming {
        (total > 0).then_some(total)
    } else {
        Some(total)
    };
    message.tokens.cache_read = cached;
    message.tokens.thinking = thought;
    message.tokens.server_tool_use = parse_server_tool_use(data);
}

/// `parse_server_tool_use`: Gemini 3 bills each distinct, non-empty query a grounding ran.
fn parse_server_tool_use(data: &Value) -> Option<Map<String, Value>> {
    let searches: usize = data
        .get("candidates")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(|candidate| {
            let mut queries: Vec<&str> = candidate
                .pointer("/groundingMetadata/webSearchQueries")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .filter(|q| !q.is_empty())
                .collect();
            queries.sort_unstable();
            queries.dedup();
            queries.len()
        })
        .sum();
    crate::tokens::Tokens::positive_counts(&json!({ "web_search_requests": searches }))
}

/// `without_search_suggestions`: Google's terms forbid storing search suggestions, and Gemini
/// takes a replayed search result without them.
fn without_search_suggestions(parts: &[Value]) -> Value {
    Value::Array(
        parts
            .iter()
            .map(|part| {
                let mut part = part.clone();
                if let Some(response) = part
                    .pointer_mut("/toolResponse/response")
                    .and_then(Value::as_object_mut)
                {
                    response.remove("search_suggestions");
                }
                part
            })
            .collect(),
    )
}

/// `Gemini#build_response_content` (`protocols/gemini/media.rb`): joined text (`None` when there
/// is none) and the `inlineData`/`fileData` parts as attachments. Other parts are ignored.
fn build_response_content(parts: &[&Value]) -> (Option<String>, Vec<Attachment>) {
    use base64::Engine;
    let mut text = String::new();
    let mut attachments = Vec::new();
    for (index, part) in parts.iter().enumerate() {
        if let Some(t) = part.get("text").and_then(Value::as_str) {
            text.push_str(t);
        } else if let Some(inline) = part.get("inlineData") {
            // `build_inline_attachment`: skipped without data.
            let Some(bytes) = inline
                .get("data")
                .and_then(Value::as_str)
                .and_then(|d| base64::engine::general_purpose::STANDARD.decode(d).ok())
            else {
                continue;
            };
            let mime = inline.get("mimeType").and_then(Value::as_str);
            attachments.push(Attachment::from_bytes(
                bytes,
                attachment_filename(mime, index),
                mime,
            ));
        } else if let Some(file) = part.get("fileData") {
            // `build_file_attachment`: skipped without a URI; the response's filename wins.
            let Some(uri) = file.get("fileUri").and_then(Value::as_str) else {
                continue;
            };
            let filename = str_of(file.get("filename")).unwrap_or_else(|| {
                attachment_filename(file.get("mimeType").and_then(Value::as_str), index)
            });
            attachments.push(Attachment::new(uri).with_filename(&filename));
        }
    }
    ((!text.is_empty()).then_some(text), attachments)
}

/// `Gemini#attachment_filename`: `gemini_attachment_{n}` plus the MIME subtype, normalized.
fn attachment_filename(mime_type: Option<&str>, index: usize) -> String {
    let Some(mime) = mime_type else {
        return format!("gemini_attachment_{}", index + 1);
    };
    let extension = match mime.rsplit('/').next().unwrap_or("") {
        "jpeg" => "jpg".to_string(),
        "plain" => "txt".to_string(),
        other => other.replace('+', "."),
    };
    format!("gemini_attachment_{}.{extension}", index + 1)
}

pub fn parse_completion_body(model: &Model, data: &Value, raw: RawResponse) -> Result<Message> {
    let parts = data
        .pointer("/candidates/0/content/parts")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    // `parse_content`: `''` when there is nothing but thoughts, else `build_response_content`.
    let non_thought: Vec<&Value> = parts
        .iter()
        .filter(|p| p.get("thought").and_then(Value::as_bool) != Some(true))
        .collect();
    let (content, attachments) = if non_thought.is_empty() {
        (Some(String::new()), Vec::new())
    } else {
        build_response_content(&non_thought)
    };
    let (thought_text, signature) = thoughts(&parts);
    let mut server_calls = part_server_calls(&parts);
    server_calls.extend(metadata_server_calls(data));

    let mut m = Message::chunk();
    m.citations = extract_citations(data, content.as_deref());
    m.content = content;
    m.attachments = attachments;
    m.thinking = Thinking::build(thought_text, signature);
    m.tool_calls = tool_call_map(extract_tool_calls(&parts));
    m.raw_content = parts
        .iter()
        .any(is_server_tool_part)
        .then(|| without_search_suggestions(&parts));
    m.server_tool_calls = server_calls;
    usage(&mut m, data, false);
    m.finish_reason = normalize_finish_reason(
        data.pointer("/candidates/0/finishReason")
            .or_else(|| data.pointer("/promptFeedback/blockReason"))
            .and_then(Value::as_str),
        FINISH_REASONS,
    );
    m.model = str_of(data.get("modelVersion")).or_else(|| Some(model.id.clone()));
    m.raw = Some(raw);
    Ok(m.normalized())
}

pub fn build_chunk(state: &mut StreamState, data: &Value) -> Message {
    let parts = data
        .pointer("/candidates/0/content/parts")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
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
        data.pointer("/candidates/0/finishReason")
            .or_else(|| data.pointer("/promptFeedback/blockReason"))
            .and_then(Value::as_str),
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
            m.raw_content = state
                .gemini_parts
                .iter()
                .any(is_server_tool_part)
                .then(|| without_search_suggestions(&state.gemini_parts));
            m.server_tool_calls = calls;
        }
    }
    m
}
