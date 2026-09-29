//! Port of `lib/ruby_llm/protocols/chat_completions/{chat,tools,media,streaming}.rb` plus the
//! per-provider dialects that subclass it (`providers/{deepseek,mistral,openrouter,xai,ollama,
//! hetzner,gpustack}/chat.rb`).

use serde_json::{Map, Value, json};

use super::{Request, ToolCalls, ToolChoice, char_slice, deep_merge, int, normalize_finish_reason, str_of, tool_call_map};
use crate::attachment::{Attachment, AttachmentType};
use crate::error::{Error, Result};
use crate::message::{Citation, Message, RawResponse, Role, Thinking, ToolArguments, ToolCall};
use crate::providers::Provider;
use crate::tool::{Tool, tool_schema};

const FINISH_REASONS: &[(&str, &str)] = &[
    ("stop", "stop"),
    ("length", "max_tokens"),
    ("tool_calls", "tool_calls"),
    ("function_call", "tool_calls"),
    ("content_filter", "content_filter"),
];

pub(crate) fn empty_parameters_schema() -> Value {
    json!({ "type": "object", "properties": {}, "required": [], "additionalProperties": false, "strict": true })
}

/// OpenAI strict mode rejects objects whose properties are not all required.
pub(crate) fn schema_strict(schema: &super::Schema) -> bool {
    if let Some(strict) = schema.strict {
        return strict;
    }
    fn strict(node: &Value) -> bool {
        match node {
            Value::Object(map) => {
                if let Some(Value::Object(props)) = map.get("properties") {
                    let required: Vec<&str> =
                        map.get("required").and_then(Value::as_array).map(|r| r.iter().filter_map(Value::as_str).collect()).unwrap_or_default();
                    if props.keys().any(|k| !required.contains(&k.as_str())) {
                        return false;
                    }
                }
                map.values().all(strict)
            }
            Value::Array(items) => items.iter().all(strict),
            _ => true,
        }
    }
    strict(&schema.schema)
}

/// The base wire format sends `developer` for system messages; providers whose `Chat` module
/// overrides `format_role` send the plain role.
fn format_role(provider: Provider, role: Role, config: &crate::Config) -> &'static str {
    let plain_roles = matches!(
        provider,
        Provider::DeepSeek
            | Provider::Mistral
            | Provider::XAI
            | Provider::Ollama
            | Provider::OllamaCloud
            | Provider::GPUStack
            | Provider::Hetzner
    );
    match role {
        Role::System if !plain_roles && config.get("openai_use_system_role") != Some("true") => "developer",
        r => r.as_str(),
    }
}

pub fn render_payload(req: &Request) -> Result<Value> {
    let provider = req.provider;
    let mut payload = Map::new();
    payload.insert("model".into(), req.model.id.clone().into());
    payload.insert("messages".into(), Value::Array(format_messages(req)?));
    payload.insert("stream".into(), req.stream.into());
    if let Some(t) = req.temperature {
        payload.insert("temperature".into(), t.into());
    }
    if let Some(max) = req.max_output_tokens {
        let field = if provider == Provider::OpenAI { "max_completion_tokens" } else { "max_tokens" };
        payload.insert(field.into(), max.into());
    }
    if !req.tools.is_empty() {
        payload.insert("tools".into(), Value::Array(req.tools.iter().map(|t| tool_for(t.as_ref())).collect()));
        if let Some(choice) = &req.tool_prefs.choice {
            payload.insert("tool_choice".into(), build_tool_choice(provider, choice));
        }
        if let Some(calls) = req.tool_prefs.calls {
            payload.insert("parallel_tool_calls".into(), (calls == ToolCalls::Many).into());
        }
    }
    if let Some(schema) = req.schema {
        payload.insert(
            "response_format".into(),
            json!({ "type": "json_schema", "json_schema": {
                "name": schema.name, "schema": schema.schema, "strict": schema_strict(schema)
            }}),
        );
    }
    if let Some(effort) = req.thinking.and_then(|t| t.effort.clone()) {
        payload.insert("reasoning_effort".into(), effort.into());
    }
    if req.stream {
        payload.insert("stream_options".into(), json!({ "include_usage": true }));
    }

    match provider {
        Provider::DeepSeek => {
            if let Some(thinking) = req.thinking.filter(|t| t.is_enabled()) {
                if thinking.is_disabled() {
                    payload.remove("reasoning_effort");
                    payload.insert("thinking".into(), json!({ "type": "disabled" }));
                } else {
                    payload.insert("thinking".into(), json!({ "type": "enabled" }));
                }
            }
            if req.schema.is_some() {
                tracing::warn!(
                    "DeepSeek Chat Completions does not support json_schema response formats. \
                     Use protocol: :responses to enforce the schema. Falling back to json_object mode."
                );
                payload.insert("response_format".into(), json!({ "type": "json_object" }));
            }
        }
        Provider::Mistral => {
            payload.remove("stream_options");
            let single_tool = payload.get("tools").and_then(Value::as_array).filter(|t| t.len() == 1).cloned();
            if payload.get("tool_choice").and_then(Value::as_str) == Some("any")
                && let Some(name) = single_tool.as_ref().and_then(|t| t[0].pointer("/function/name")).cloned() {
                    payload.insert("tool_choice".into(), json!({ "type": "function", "function": { "name": name } }));
                }
        }
        Provider::OpenRouter => {
            payload.remove("reasoning_effort");
            if let Some(schema) = payload
                .get_mut("response_format")
                .and_then(|f| f.pointer_mut("/json_schema/schema"))
                .and_then(Value::as_object_mut)
            {
                schema.remove("strict");
            }
            if req.tool_prefs.choice == Some(ToolChoice::None) {
                payload.remove("tools");
                payload.remove("parallel_tool_calls");
            }
            if let Some(thinking) = req.thinking.filter(|t| t.is_enabled()) {
                let mut reasoning = Map::new();
                if let Some(e) = &thinking.effort {
                    reasoning.insert("effort".into(), e.clone().into());
                }
                if let Some(b) = thinking.budget {
                    reasoning.insert("max_tokens".into(), b.into());
                }
                if let Some(enabled) = thinking.enabled {
                    reasoning.insert("enabled".into(), enabled.into());
                }
                if reasoning.is_empty() {
                    reasoning.insert("enabled".into(), true.into());
                }
                payload.insert("reasoning".into(), Value::Object(reasoning));
            }
        }
        _ => {}
    }
    Ok(Value::Object(payload))
}

fn format_messages(req: &Request) -> Result<Vec<Value>> {
    let (system, other): (Vec<&Message>, Vec<&Message>) = req.messages.iter().partition(|m| m.role == Role::System);
    let ordered: Vec<&Message> = system.into_iter().chain(other).collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < ordered.len() {
        if ordered[i].is_tool_result() {
            // A parallel round's results stay consecutive; attachment carriers follow the whole run.
            let start = i;
            while i < ordered.len() && ordered[i].is_tool_result() {
                out.push(format_message(req, ordered[i])?);
                i += 1;
            }
            for msg in &ordered[start..i] {
                if !msg.attachments.is_empty() {
                    let mut parts =
                        vec![json!({ "type": "text", "text": format!("Attachments from tool call {}:", msg.tool_call_id.as_deref().unwrap_or("")) })];
                    if let Value::Array(more) = format_content(req.provider, None, &msg.attachments)? {
                        parts.extend(more);
                    }
                    out.push(json!({ "role": "user", "content": parts }));
                }
            }
        } else {
            out.push(format_message(req, ordered[i])?);
            i += 1;
        }
    }
    Ok(out)
}

fn format_message(req: &Request, msg: &Message) -> Result<Value> {
    let provider = req.provider;
    let attachments: &[Attachment] = if msg.is_tool_result() { &[] } else { &msg.attachments };
    let mut content = format_content(provider, msg.content.as_deref(), attachments)?;
    let thinking_only = msg.role == Role::Assistant && msg.thinking.is_some() && !msg.is_tool_call();
    if content.is_null() && thinking_only {
        content = Value::String(String::new());
    }
    if provider == Provider::Mistral && msg.role == Role::Assistant
        && let Some(thinking) = &msg.thinking {
            let mut blocks = Vec::new();
            if let Some(text) = &thinking.text {
                let mut block = json!({ "type": "thinking", "thinking": [{ "type": "text", "text": text }] });
                if let Some(sig) = &thinking.signature {
                    block["signature"] = sig.clone().into();
                }
                blocks.push(block);
            } else if let Some(sig) = &thinking.signature {
                blocks.push(json!({ "type": "thinking", "signature": sig }));
            }
            match content {
                Value::Array(parts) => blocks.extend(parts),
                Value::String(s) if !s.is_empty() => blocks.push(json!({ "type": "text", "text": s })),
                _ => {}
            }
            content = Value::Array(blocks);
        }
    if msg.cache_until_here && provider == Provider::OpenRouter {
        let mut blocks = match content {
            Value::Array(parts) => parts,
            Value::String(s) => vec![json!({ "type": "text", "text": s })],
            _ => Vec::new(),
        };
        if let Some(Value::Object(last)) = blocks.last_mut() {
            last.entry("cache_control").or_insert_with(|| json!({ "type": "ephemeral" }));
        }
        content = Value::Array(blocks);
    } else if msg.cache_until_here && matches!(provider, Provider::OpenAI) {
        let parts = match &content {
            Value::Array(parts) => Some(parts.clone()),
            Value::String(s) if !s.is_empty() => Some(vec![json!({ "type": "text", "text": s })]),
            _ => None,
        };
        if let Some(mut parts) = parts
            && let Some(Value::Object(last)) = parts.last_mut() {
                last.insert("prompt_cache_breakpoint".into(), json!({ "mode": "explicit" }));
                content = Value::Array(parts);
            }
    }

    let mut out = Map::new();
    out.insert("role".into(), format_role(provider, msg.role, req.config).into());
    if !content.is_null() {
        out.insert("content".into(), content);
    }
    if let Some(calls) = msg.tool_calls.as_ref().filter(|c| !c.is_empty()) {
        let calls: Vec<Value> = calls
            .values()
            .map(|tc| {
                let mut call = json!({
                    "id": tc.id,
                    "type": "function",
                    "function": { "name": tc.name, "arguments": Value::Object(tc.arguments()).to_string() },
                });
                if let Some(sig) = &tc.thought_signature {
                    call["extra_content"] = json!({ "google": { "thought_signature": sig } });
                }
                call
            })
            .collect();
        out.insert("tool_calls".into(), Value::Array(calls));
    }
    if let Some(id) = &msg.tool_call_id {
        out.insert("tool_call_id".into(), id.clone().into());
    }
    if msg.role == Role::Assistant {
        format_thinking(provider, msg, &mut out);
    }
    Ok(Value::Object(out))
}

fn format_thinking(provider: Provider, msg: &Message, out: &mut Map<String, Value>) {
    match provider {
        Provider::Mistral => {}
        Provider::DeepSeek => {
            let text = msg.thinking.as_ref().and_then(|t| t.text.clone()).unwrap_or_default();
            out.insert("reasoning_content".into(), text.clone().into());
            if !text.is_empty() {
                out.insert("reasoning".into(), text.into());
            }
            if let Some(sig) = msg.thinking.as_ref().and_then(|t| t.signature.clone()) {
                out.insert("reasoning_signature".into(), sig.into());
            }
        }
        Provider::OpenRouter => {
            if let Some(details) = msg.raw_reasoning.as_ref().filter(|r| r.is_array()) {
                out.insert("reasoning_details".into(), details.clone());
                return;
            }
            let Some(t) = &msg.thinking else { return };
            let detail = if let Some(text) = &t.text {
                let mut d = json!({ "type": "reasoning.text", "text": text });
                if let Some(sig) = &t.signature {
                    d["signature"] = sig.clone().into();
                }
                d
            } else if let Some(sig) = &t.signature {
                json!({ "type": "reasoning.encrypted", "data": sig })
            } else {
                return;
            };
            out.insert("reasoning_details".into(), json!([detail]));
        }
        _ => {
            let Some(t) = &msg.thinking else { return };
            if let Some(text) = &t.text {
                out.insert("reasoning".into(), text.clone().into());
                out.insert("reasoning_content".into(), text.clone().into());
            }
            if let Some(sig) = &t.signature {
                out.insert("reasoning_signature".into(), sig.clone().into());
            }
        }
    }
}

fn text_part(text: &str) -> Value {
    json!({ "type": "text", "text": text })
}

/// `ChatCompletions::Media.format_content` with each provider's attachment rules. Returns the
/// bare string when there are no attachments.
pub(crate) fn format_content(provider: Provider, content: Option<&str>, attachments: &[Attachment]) -> Result<Value> {
    if attachments.is_empty() {
        return Ok(content.map(|c| Value::String(c.to_string())).unwrap_or(Value::Null));
    }
    let mut parts = Vec::new();
    if let Some(c) = content {
        parts.push(text_part(c));
    }
    let unsupported = |a: &Attachment| Error::UnsupportedAttachment(super::anthropic::unsupported(&a.mime_type));
    for a in attachments {
        let kind = a.kind();
        // `Media.format_provider_file`, reached by the providers that use the shared
        // `format_attachment`; those with `document_attachments: :none` refuse it.
        let own_media = matches!(provider, Provider::Mistral | Provider::Ollama | Provider::OllamaCloud | Provider::GPUStack | Provider::Perplexity)
            || (provider == Provider::OpenRouter && kind == AttachmentType::Video);
        if let Some(file_id) = a.provider_file_id().filter(|_| !own_media) {
            if matches!(provider, Provider::DeepSeek | Provider::XAI | Provider::Hetzner) {
                return Err(unsupported(a));
            }
            parts.push(json!({ "type": "file", "file": { "file_id": file_id } }));
            continue;
        }
        let part = match (provider, kind) {
            (Provider::Mistral, AttachmentType::Image) => json!({ "type": "image_url", "image_url": a.url_or_data_uri()? }),
            (Provider::Mistral, AttachmentType::Pdf | AttachmentType::Document) => {
                json!({ "type": "document_url", "document_url": a.url_or_data_uri()? })
            }
            (Provider::Ollama | Provider::OllamaCloud | Provider::GPUStack, AttachmentType::Image) => {
                json!({ "type": "image_url", "image_url": { "url": a.for_llm()?, "detail": "auto" } })
            }
            (Provider::GPUStack, AttachmentType::Video) => json!({ "type": "video_url", "video_url": { "url": a.url_or_data_uri()? } }),
            (Provider::Ollama | Provider::OllamaCloud | Provider::GPUStack, AttachmentType::Pdf | AttachmentType::Document) => {
                return Err(unsupported(a));
            }
            (Provider::Hetzner, AttachmentType::Image) => json!({ "type": "image_url", "image_url": { "url": a.for_llm()? } }),
            (Provider::OpenRouter, AttachmentType::Video) => json!({ "type": "video_url", "video_url": { "url": a.url_or_data_uri()? } }),
            (Provider::DeepSeek | Provider::XAI | Provider::Hetzner, AttachmentType::Pdf | AttachmentType::Document) => {
                return Err(unsupported(a));
            }
            (Provider::DeepSeek | Provider::XAI | Provider::Hetzner, AttachmentType::Audio) => return Err(unsupported(a)),
            (_, AttachmentType::Image) => {
                let mut part = json!({ "type": "image_url", "image_url": { "url": a.url_or_data_uri()? } });
                if let Some(res) = a.resolution {
                    part["image_url"]["detail"] =
                        if res == crate::attachment::Resolution::Low { "low" } else { "high" }.into();
                }
                part
            }
            (_, AttachmentType::Audio) => json!({ "type": "input_audio", "input_audio": { "data": a.encoded()?, "format": a.format() } }),
            (_, AttachmentType::Pdf) => {
                json!({ "type": "file", "file": { "filename": a.filename, "file_data": a.for_llm()? } })
            }
            (_, AttachmentType::Text) => text_part(&a.for_llm()?),
            _ => return Err(unsupported(a)),
        };
        parts.push(part);
    }
    Ok(Value::Array(parts))
}

fn tool_for(tool: &dyn Tool) -> Value {
    let mut definition = json!({
        "type": "function",
        "function": {
            "name": tool.name(),
            "description": tool.description(),
            "parameters": tool_schema(tool).unwrap_or_else(empty_parameters_schema),
        }
    });
    let opts = tool.provider_options();
    if !opts.is_empty() {
        deep_merge(&mut definition, &Value::Object(opts));
    }
    definition
}

fn build_tool_choice(provider: Provider, choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => "auto".into(),
        ToolChoice::None => "none".into(),
        ToolChoice::Required if provider == Provider::Mistral => "any".into(),
        ToolChoice::Required => "required".into(),
        ToolChoice::Tool(name) => json!({ "type": "function", "function": { "name": name } }),
    }
}

fn input_tokens(usage: &Value) -> Option<i64> {
    if let Some(miss) = int(usage.get("prompt_cache_miss_tokens")) {
        return Some(miss);
    }
    let prompt = int(usage.get("prompt_tokens"))?;
    Some((prompt - cache_read_tokens(usage).unwrap_or(0) - cache_write_tokens(usage).unwrap_or(0)).max(0))
}

fn output_tokens(usage: &Value) -> Option<i64> {
    let completion = int(usage.get("completion_tokens"))?;
    let generated = match (int(usage.get("prompt_tokens")), int(usage.get("total_tokens"))) {
        (Some(p), Some(t)) => Some((t - p).max(0)),
        _ => None,
    };
    Some(match generated {
        Some(g) if g > completion => g,
        _ => completion,
    })
}

fn cache_read_tokens(usage: &Value) -> Option<i64> {
    int(usage.pointer("/prompt_tokens_details/cached_tokens")).or_else(|| int(usage.get("prompt_cache_hit_tokens")))
}

fn cache_write_tokens(usage: &Value) -> Option<i64> {
    int(usage.pointer("/prompt_tokens_details/cache_write_tokens"))
        .or_else(|| int(usage.pointer("/input_tokens_details/cache_write_tokens")))
        .or(Some(0))
}

fn thinking_tokens(usage: &Value) -> Option<i64> {
    int(usage.pointer("/completion_tokens_details/reasoning_tokens")).or_else(|| int(usage.get("reasoning_tokens")))
}

pub(crate) fn reported_cost(provider: Provider, usage: &Value) -> Option<f64> {
    match provider {
        Provider::OpenRouter => {
            let mut cost = usage.get("cost")?.as_f64()?;
            if usage.get("is_byok").and_then(Value::as_bool) == Some(true) {
                cost += usage.pointer("/cost_details/upstream_inference_cost").and_then(Value::as_f64).unwrap_or(0.0);
            }
            Some(cost)
        }
        Provider::XAI => usage.get("cost_in_usd_ticks").and_then(Value::as_f64).map(|t| t * 1e-10),
        _ => None,
    }
}

fn fill_usage(provider: Provider, message: &mut Message, usage: &Value) {
    message.tokens.input = input_tokens(usage);
    message.tokens.output = output_tokens(usage);
    message.tokens.cache_read = cache_read_tokens(usage);
    message.tokens.cache_write = if usage.is_object() && !usage.as_object().unwrap().is_empty() { cache_write_tokens(usage) } else { None };
    message.tokens.thinking = thinking_tokens(usage);
    message.tokens.server_tool_use = usage
        .get("server_tool_use")
        .or_else(|| usage.get("server_tool_use_details"))
        .and_then(Value::as_object)
        .cloned();
    message.tokens.reported_cost = reported_cost(provider, usage);
}

fn extract_content_and_thinking(content: Option<&Value>) -> (Option<String>, Option<String>) {
    match content {
        Some(Value::Array(blocks)) => {
            let text: String = blocks
                .iter()
                .filter(|b| b.get("type").and_then(Value::as_str) == Some("text"))
                .filter_map(|b| b.get("text").and_then(Value::as_str))
                .collect();
            let thinking: String = blocks
                .iter()
                .filter(|b| b.get("type").and_then(Value::as_str) == Some("thinking"))
                .filter_map(|b| match b.get("thinking") {
                    Some(Value::String(s)) => Some(s.clone()),
                    Some(Value::Array(items)) => Some(
                        items
                            .iter()
                            .filter(|i| i.get("type").and_then(Value::as_str) == Some("text"))
                            .filter_map(|i| i.get("text").and_then(Value::as_str))
                            .collect(),
                    ),
                    _ => str_of(b.get("text")),
                })
                .collect();
            ((!text.is_empty()).then_some(text), (!thinking.is_empty()).then_some(thinking))
        }
        Some(Value::String(s)) => (Some(s.clone()), None),
        _ => (None, None),
    }
}

fn parse_annotations(annotations: Option<&Value>, content: Option<&str>) -> Vec<Citation> {
    annotations
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|a| {
            let d = a.get("url_citation")?.as_object()?;
            let start = int(d.get("start_index"));
            let end = int(d.get("end_index"));
            Some(Citation {
                url: str_of(d.get("url")),
                title: str_of(d.get("title")),
                text: match (content, start, end) {
                    (Some(c), Some(s), Some(e)) => char_slice(c, s, e),
                    _ => None,
                },
                start_index: start,
                end_index: end,
                ..Default::default()
            })
        })
        .collect()
}

/// Perplexity and xAI return search citations at the root of the response.
pub(crate) fn parse_root_citations(data: &Value) -> Vec<Citation> {
    if let Some(results) = data.get("search_results").and_then(Value::as_array).filter(|r| !r.is_empty()) {
        return parse_search_results(results);
    }
    data.get("citations")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .enumerate()
        .filter_map(|(i, url)| url.as_str().map(|u| Citation { url: Some(u.into()), source_index: Some(i as i64), ..Default::default() }))
        .collect()
}

pub(crate) fn parse_search_results(results: &[Value]) -> Vec<Citation> {
    results
        .iter()
        .enumerate()
        .filter(|(_, r)| r.is_object())
        .map(|(i, r)| Citation {
            url: str_of(r.get("url")),
            title: str_of(r.get("title")),
            cited_text: str_of(r.get("snippet")),
            source_index: Some(i as i64),
            ..Default::default()
        })
        .collect()
}

fn parse_tool_calls(calls: Option<&Value>, parse_arguments: bool, stream_keys: bool, finish: Option<&str>) -> Result<Vec<(String, ToolCall)>> {
    let Some(calls) = calls.and_then(Value::as_array).filter(|c| !c.is_empty()) else { return Ok(Vec::new()) };
    calls
        .iter()
        .map(|tc| {
            let raw_args = tc.pointer("/function/arguments").and_then(Value::as_str).unwrap_or("");
            let arguments = if parse_arguments {
                if raw_args.is_empty() {
                    ToolArguments::Parsed(Map::new())
                } else {
                    ToolArguments::Parsed(serde_json::from_str(raw_args).map_err(|_| Error::tool_call_parse(finish))?)
                }
            } else {
                ToolArguments::Partial(raw_args.to_string())
            };
            let id = str_of(tc.get("id")).unwrap_or_default();
            let key = if stream_keys {
                tc.get("index").map(|i| i.to_string()).unwrap_or_else(|| id.clone())
            } else {
                id.clone()
            };
            Ok((
                key,
                ToolCall {
                    id,
                    name: str_of(tc.pointer("/function/name")).unwrap_or_default(),
                    arguments,
                    thought_signature: str_of(tc.pointer("/extra_content/google/thought_signature")),
                    remote: false,
                    starts: tc.get("id").is_some_and(|v| !v.is_null()),
                },
            ))
        })
        .collect()
}

pub fn parse_completion_body(provider: Provider, data: &Value, raw: RawResponse) -> Result<Message> {
    if let Some(msg) = data.pointer("/error/message").and_then(Value::as_str) {
        return Err(Error::Api(msg.into(), None));
    }
    let Some(message_data) = data.pointer("/choices/0/message") else {
        let mut message = "Provider returned no completion message".to_string();
        if let Some(r) = data.pointer("/choices/0/finish_reason").and_then(Value::as_str) {
            message = format!("{message} (finish_reason: {r})");
        }
        return Err(Error::Api(message, None));
    };
    let usage = data.get("usage").cloned().unwrap_or_else(|| json!({}));
    let finish_raw = data.pointer("/choices/0/finish_reason").and_then(Value::as_str);
    let finish = normalize_finish_reason(finish_raw, FINISH_REASONS);
    let (content, block_thinking) = extract_content_and_thinking(message_data.get("content"));
    let thinking_text = block_thinking.or_else(|| {
        ["reasoning_content", "reasoning", "thinking"].iter().find_map(|k| message_data.get(*k).and_then(Value::as_str).map(str::to_string))
    });
    let signature = ["reasoning_signature", "signature"].iter().find_map(|k| message_data.get(*k).and_then(Value::as_str).map(str::to_string));
    let calls = parse_tool_calls(message_data.get("tool_calls"), true, false, finish.as_ref().map(|f| f.as_str()))?;

    let mut m = Message::chunk();
    let mut citations = parse_annotations(message_data.get("annotations"), content.as_deref());
    if citations.is_empty() {
        citations = parse_root_citations(data);
    }
    m.citations = citations;
    m.content = content;
    m.thinking = Thinking::build(thinking_text, signature);
    if provider == Provider::OpenRouter {
        m.raw_reasoning = message_data.get("reasoning_details").filter(|d| d.as_array().is_some_and(|a| !a.is_empty())).cloned();
    }
    m.tool_calls = tool_call_map(calls.into_iter().map(|(_, c)| c).collect());
    fill_usage(provider, &mut m, &usage);
    m.finish_reason = finish;
    m.model = str_of(data.get("model"));
    m.raw = Some(raw);
    Ok(m.normalized())
}

pub fn build_chunk(provider: Provider, data: &Value) -> Message {
    let usage = data.get("usage").cloned().unwrap_or_else(|| json!({}));
    let delta = data.pointer("/choices/0/delta").cloned().unwrap_or_else(|| json!({}));
    let content_source = delta.get("content").or_else(|| data.pointer("/choices/0/message/content"));
    let (content, block_thinking) = extract_content_and_thinking(content_source);
    let mut m = Message::chunk();
    m.model = str_of(data.get("model"));
    m.content = content;
    let mut citations = parse_annotations(delta.get("annotations"), None);
    if citations.is_empty() {
        citations = parse_root_citations(data);
    }
    m.citations = citations;
    let text = block_thinking
        .or_else(|| str_of(delta.get("reasoning_content")))
        .or_else(|| str_of(delta.get("reasoning")));
    m.thinking = Thinking::build(text, str_of(delta.get("reasoning_signature")));
    if let Ok(calls) = parse_tool_calls(delta.get("tool_calls"), false, true, None)
        && !calls.is_empty() {
            m.tool_calls = Some(calls.into_iter().collect());
        }
    if usage.as_object().is_some_and(|u| !u.is_empty()) {
        fill_usage(provider, &mut m, &usage);
    }
    m.finish_reason = normalize_finish_reason(data.pointer("/choices/0/finish_reason").and_then(Value::as_str), FINISH_REASONS);
    m
}
