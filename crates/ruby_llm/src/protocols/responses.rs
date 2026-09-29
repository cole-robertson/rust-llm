//! Port of `lib/ruby_llm/protocols/responses/{chat,tools,media,streaming,approvals}.rb`, the
//! OpenAI Responses API that OpenAI and xAI default to, plus the Perplexity Agent API subclass.

use serde_json::{Map, Value, json};

use super::chat_completions::{empty_parameters_schema, parse_root_citations, parse_search_results, schema_strict};
use super::{Request, StreamState, ToolCalls, ToolChoice, char_slice, deep_merge, int, normalize_finish_reason, str_of, tool_call_map};
use crate::attachment::{Attachment, AttachmentType};
use crate::error::{Error, Result};
use crate::message::{Citation, Message, RawResponse, Role, ServerToolCall, Thinking, ToolArguments, ToolCall};
use crate::providers::Provider;
use crate::thinking::Display;
use crate::tool::{Tool, tool_schema};

const FINISH_REASONS: &[(&str, &str)] =
    &[("completed", "stop"), ("max_output_tokens", "max_tokens"), ("content_filter", "content_filter")];

const PERPLEXITY_PRESETS: &[&str] = &["fast", "low", "medium", "high", "xhigh", "wide-research"];
const SONAR_PRESETS: &[(&str, &str)] =
    &[("sonar", "fast"), ("sonar-pro", "low"), ("sonar-reasoning-pro", "medium"), ("sonar-deep-research", "high")];

pub fn render_payload(req: &Request) -> Result<Value> {
    let mut payload = Map::new();
    payload.insert("model".into(), req.model.id.clone().into());
    payload.insert("input".into(), Value::Array(format_input(req.messages)?));
    if let Some(instructions) = format_instructions(req.messages) {
        payload.insert("instructions".into(), instructions.into());
    }
    payload.insert("stream".into(), req.stream.into());
    payload.insert("store".into(), false.into());
    payload.insert("include".into(), json!(["reasoning.encrypted_content"]));
    if let Some(t) = req.temperature {
        payload.insert("temperature".into(), t.into());
    }
    let max_output = req.max_output_tokens.or_else(|| {
        (req.provider == Provider::Perplexity && req.model.id.starts_with("anthropic/"))
            .then(|| req.model.max_output_tokens.unwrap_or(super::anthropic::DEFAULT_MAX_OUTPUT_TOKENS))
    });
    if let Some(max) = max_output {
        payload.insert("max_output_tokens".into(), max.into());
    }
    if !req.tools.is_empty() {
        payload.insert("tools".into(), Value::Array(req.tools.iter().map(|t| tool_for(t.as_ref())).collect()));
        if let Some(choice) = &req.tool_prefs.choice {
            payload.insert(
                "tool_choice".into(),
                match choice {
                    ToolChoice::Auto => "auto".into(),
                    ToolChoice::None => "none".into(),
                    ToolChoice::Required => "required".into(),
                    ToolChoice::Tool(name) => json!({ "type": "function", "name": name }),
                },
            );
        }
        if let Some(calls) = req.tool_prefs.calls {
            payload.insert("parallel_tool_calls".into(), (calls == ToolCalls::Many).into());
        }
    }
    if let Some(schema) = req.schema {
        payload.insert(
            "text".into(),
            json!({ "format": {
                "type": "json_schema", "name": schema.name, "schema": schema.schema, "strict": schema_strict(schema)
            }}),
        );
    }
    if let Some(thinking) = req.thinking {
        if let Some(effort) = &thinking.effort {
            payload.insert("reasoning".into(), json!({ "effort": effort }));
        }
        if thinking.display == Some(Display::Summarized) {
            let reasoning = payload.entry("reasoning").or_insert_with(|| json!({}));
            reasoning["summary"] = "auto".into();
        }
    }
    if req.provider == Provider::Perplexity {
        let id = req.model.id.as_str();
        let preset = PERPLEXITY_PRESETS
            .iter()
            .find(|p| **p == id)
            .copied()
            .or_else(|| SONAR_PRESETS.iter().find(|(k, _)| *k == id).map(|(_, v)| *v));
        if let Some(preset) = preset {
            payload.remove("model");
            payload.insert("preset".into(), preset.into());
        }
    }
    Ok(Value::Object(payload))
}

fn system_input_item(msg: &Message) -> bool {
    msg.role == Role::System && (msg.cache_until_here || !msg.attachments.is_empty())
}

fn format_instructions(messages: &[Message]) -> Option<String> {
    let parts: Vec<String> = messages
        .iter()
        .filter(|m| m.role == Role::System && !system_input_item(m))
        .map(|m| m.content().to_string())
        .collect();
    (!parts.is_empty()).then(|| parts.join("\n\n"))
}

fn format_input(messages: &[Message]) -> Result<Vec<Value>> {
    let mut input = Vec::new();
    for msg in messages {
        if msg.role == Role::System && !system_input_item(msg) {
            continue;
        }
        input.extend(format_item(msg)?);
    }
    Ok(input)
}

fn with_cache_breakpoint(mut item: Value) -> Value {
    let parts = match item.get("content") {
        Some(Value::Array(parts)) => Some(parts.clone()),
        Some(Value::String(s)) if !s.is_empty() => Some(vec![json!({ "type": "input_text", "text": s })]),
        _ => None,
    };
    if let Some(mut parts) = parts
        && let Some(Value::Object(last)) = parts.last_mut() {
            last.insert("prompt_cache_breakpoint".into(), json!({ "mode": "explicit" }));
            item["content"] = Value::Array(parts);
        }
    item
}

fn format_item(msg: &Message) -> Result<Vec<Value>> {
    Ok(match msg.role {
        Role::System | Role::User => {
            let role = msg.role.as_str();
            let item = json!({ "role": role, "content": format_content(msg.content.as_deref(), &msg.attachments)? });
            vec![if msg.cache_until_here { with_cache_breakpoint(item) } else { item }]
        }
        Role::Tool => {
            if let Some(Value::Array(raw)) = &msg.raw_content {
                return Ok(raw.clone());
            }
            let mut items = vec![json!({
                "type": "function_call_output",
                "call_id": msg.tool_call_id,
                "output": msg.content.clone().unwrap_or_default(),
            })];
            if !msg.attachments.is_empty() {
                let mut parts = vec![json!({ "type": "input_text", "text": format!("Attachments from tool call {}:", msg.tool_call_id.as_deref().unwrap_or("")) })];
                if let Value::Array(more) = format_content(None, &msg.attachments)? {
                    parts.extend(more);
                }
                items.push(json!({ "role": "user", "content": parts }));
            }
            items
        }
        Role::Assistant => {
            if let Some(Value::Array(raw)) = &msg.raw_content {
                return Ok(raw.clone());
            }
            let mut items = Vec::new();
            if let Some(t) = msg.thinking.as_ref().filter(|t| t.signature.is_some()) {
                let summary = t.text.as_ref().map(|text| json!([{ "type": "summary_text", "text": text }])).unwrap_or_else(|| json!([]));
                items.push(json!({ "type": "reasoning", "summary": summary, "encrypted_content": t.signature }));
            }
            if !msg.content().trim().is_empty() {
                items.push(json!({ "role": "assistant", "content": [{ "type": "output_text", "text": msg.content() }] }));
            }
            for tc in msg.tool_calls.iter().flat_map(|c| c.values()) {
                items.push(json!({
                    "type": "function_call",
                    "call_id": tc.id,
                    "name": tc.name,
                    "arguments": Value::Object(tc.arguments()).to_string(),
                }));
            }
            items
        }
    })
}

/// `Responses::Media.format_content`.
fn format_content(content: Option<&str>, attachments: &[Attachment]) -> Result<Value> {
    if attachments.is_empty() {
        return Ok(content.map(|c| Value::String(c.into())).unwrap_or(Value::Null));
    }
    let mut parts = Vec::new();
    if let Some(c) = content {
        parts.push(json!({ "type": "input_text", "text": c }));
    }
    for a in attachments {
        parts.push(match a.kind() {
            AttachmentType::Image => {
                let mut part = json!({ "type": "input_image", "image_url": a.url_or_data_uri()? });
                if let Some(res) = a.resolution {
                    part["detail"] = if res == crate::attachment::Resolution::Low { "low" } else { "high" }.into();
                }
                part
            }
            AttachmentType::Pdf | AttachmentType::Document => {
                json!({ "type": "input_file", "filename": a.filename, "file_data": a.for_llm()? })
            }
            AttachmentType::Text => json!({ "type": "input_text", "text": a.for_llm()? }),
            _ => return Err(Error::UnsupportedAttachment(super::anthropic::unsupported(&a.mime_type))),
        });
    }
    Ok(Value::Array(parts))
}

fn tool_for(tool: &dyn Tool) -> Value {
    let mut definition = json!({
        "type": "function",
        "name": tool.name(),
        "description": tool.description(),
        "parameters": tool_schema(tool).unwrap_or_else(empty_parameters_schema),
        "strict": false,
    });
    let opts = tool.provider_options();
    if !opts.is_empty() {
        deep_merge(&mut definition, &Value::Object(opts));
    }
    definition
}

fn parse_usage(provider: Provider, message: &mut Message, usage: &Value) {
    let mut usage = usage.clone();
    if provider == Provider::Perplexity {
        let details = usage.get("input_tokens_details").cloned().unwrap_or_else(|| json!({}));
        let mut merged = json!({ "cache_write_tokens": details.get("cache_creation_input_tokens") });
        deep_merge(&mut merged, &details);
        usage["input_tokens_details"] = merged;
    }
    let details = usage.get("input_tokens_details").or_else(|| usage.get("prompt_tokens_details")).cloned().unwrap_or_else(|| json!({}));
    let cached = int(details.get("cached_tokens"));
    let writes = int(details.get("cache_write_tokens"));
    message.tokens.input = int(usage.get("input_tokens")).map(|i| (i - cached.unwrap_or(0) - writes.unwrap_or(0)).max(0));
    message.tokens.output = int(usage.get("output_tokens"));
    message.tokens.cache_read = cached;
    message.tokens.cache_write = writes;
    message.tokens.thinking = int(usage.pointer("/output_tokens_details/reasoning_tokens"));
    match provider {
        Provider::XAI => {
            let mut counters = Map::new();
            for key in ["num_sources_used", "num_server_side_tools_used"] {
                if let Some(v) = usage.get(key).filter(|v| v.as_i64().unwrap_or(0) != 0) {
                    counters.insert(key.into(), v.clone());
                }
            }
            message.tokens.server_tool_use = (!counters.is_empty()).then_some(counters);
            message.tokens.reported_cost = usage.get("cost_in_usd_ticks").and_then(Value::as_f64).map(|t| t * 1e-10);
        }
        Provider::Perplexity => message.tokens.reported_cost = usage.pointer("/cost/total_cost").and_then(Value::as_f64),
        _ => {}
    }
}

const CLIENT_OUTPUT_ITEM_TYPES: &[&str] = &["message", "reasoning", "function_call"];

fn server_tool_items(output: &[Value]) -> Vec<ServerToolCall> {
    output
        .iter()
        .filter(|i| !CLIENT_OUTPUT_ITEM_TYPES.contains(&i.get("type").and_then(Value::as_str).unwrap_or("")))
        .filter(|i| !(i.get("type").and_then(Value::as_str) == Some("search_results")))
        .map(|item| ServerToolCall {
            kind: str_of(item.get("type")).unwrap_or_default(),
            name: str_of(item.get("name")),
            id: str_of(item.get("id")),
            input: item.get("action").or_else(|| item.get("arguments")).or_else(|| item.get("code")).cloned(),
            result: ["result", "results", "outputs", "output", "encrypted_content"].iter().find_map(|k| item.get(*k).cloned()),
            raw: item.clone(),
        })
        .collect()
}

fn text_key(part_type: &str) -> Option<&'static str> {
    match part_type {
        "output_text" => Some("text"),
        "refusal" => Some("refusal"),
        _ => None,
    }
}

fn parse_output_text(output: &[Value]) -> Option<String> {
    let texts: Vec<String> = output
        .iter()
        .filter(|i| i.get("type").and_then(Value::as_str) == Some("message"))
        .flat_map(|m| m.get("content").and_then(Value::as_array).cloned().unwrap_or_default())
        .filter_map(|part| {
            let key = text_key(part.get("type").and_then(Value::as_str).unwrap_or(""))?;
            str_of(part.get(key))
        })
        .collect();
    (!texts.is_empty()).then(|| texts.join(""))
}

fn parse_annotation(a: &Value, content: Option<&str>) -> Option<Citation> {
    let kind = a.get("type").and_then(Value::as_str).unwrap_or("");
    let start = int(a.get("start_index"));
    let end = int(a.get("end_index"));
    let text = match (content, start, end) {
        (Some(c), Some(s), Some(e)) => char_slice(c, s, e),
        _ => None,
    };
    match kind {
        "file_citation" | "container_file_citation" => Some(Citation {
            source_id: str_of(a.get("file_id")),
            title: str_of(a.get("filename")),
            source_index: int(a.get("index")),
            text,
            start_index: start,
            end_index: end,
            ..Default::default()
        }),
        _ => {
            let d = a.get("url_citation").unwrap_or(a);
            if a.get("url_citation").is_none() && kind != "url_citation" {
                return None;
            }
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
        }
    }
}

fn offset(c: Citation, by: i64, content: Option<&str>) -> Citation {
    let start = c.start_index.map(|s| s + by);
    let end = c.end_index.map(|e| e + by);
    let text = match (content, start, end) {
        (Some(content), Some(s), Some(e)) => char_slice(content, s, e),
        _ => None,
    };
    Citation { start_index: start, end_index: end, text, ..c }
}

fn parse_citations(provider: Provider, data: &Value, output: &[Value], content: Option<&str>) -> Vec<Citation> {
    let mut offset_by = 0i64;
    let mut citations = Vec::new();
    for message in output.iter().filter(|i| i.get("type").and_then(Value::as_str) == Some("message")) {
        for part in message.get("content").and_then(Value::as_array).into_iter().flatten() {
            let Some(key) = text_key(part.get("type").and_then(Value::as_str).unwrap_or("")) else { continue };
            for a in part.get("annotations").and_then(Value::as_array).into_iter().flatten() {
                if let Some(c) = parse_annotation(a, None) {
                    citations.push(offset(c, offset_by, content));
                }
            }
            offset_by += part.get(key).and_then(Value::as_str).map(|s| s.chars().count() as i64).unwrap_or(0);
        }
    }
    if !citations.is_empty() {
        return citations;
    }
    let root = parse_root_citations(data);
    if !root.is_empty() || provider != Provider::Perplexity {
        return root;
    }
    let results: Vec<Value> = output
        .iter()
        .filter(|i| i.get("type").and_then(Value::as_str) == Some("search_results"))
        .flat_map(|i| i.get("results").and_then(Value::as_array).cloned().unwrap_or_default())
        .collect();
    parse_search_results(&results)
}

fn parse_finish_reason(data: &Value) -> Option<crate::message::FinishReason> {
    let reason = data
        .pointer("/incomplete_details/reason")
        .and_then(Value::as_str)
        .or_else(|| data.get("status").and_then(Value::as_str).filter(|s| *s == "completed"));
    normalize_finish_reason(reason, FINISH_REASONS)
}

fn parse_tool_calls(output: &[Value], finish: Option<&str>) -> Result<Vec<ToolCall>> {
    let mut calls = Vec::new();
    for item in output {
        let kind = item.get("type").and_then(Value::as_str);
        let parse = |args: Option<&Value>| -> Result<Map<String, Value>> {
            match args {
                Some(Value::Object(m)) => Ok(m.clone()),
                Some(Value::String(s)) if !s.is_empty() => serde_json::from_str(s).map_err(|_| Error::tool_call_parse(finish)),
                _ => Ok(Map::new()),
            }
        };
        match kind {
            Some("function_call") => calls.push(ToolCall::new(
                str_of(item.get("call_id")).unwrap_or_default(),
                str_of(item.get("name")).unwrap_or_default(),
                parse(item.get("arguments"))?,
            )),
            Some("mcp_approval_request") => {
                let mut call = ToolCall::new(
                    str_of(item.get("id")).unwrap_or_default(),
                    str_of(item.get("name")).unwrap_or_default(),
                    parse(item.get("arguments"))?,
                );
                call.remote = true;
                calls.push(call);
            }
            _ => {}
        }
    }
    Ok(calls)
}

fn reasoning(output: &[Value], provider: Provider) -> Option<Thinking> {
    let items: Vec<&Value> = output.iter().filter(|i| i.get("type").and_then(Value::as_str) == Some("reasoning")).collect();
    let mut texts: Vec<String> = Vec::new();
    if provider == Provider::DeepSeek {
        texts = items
            .iter()
            .flat_map(|i| i.get("content").and_then(Value::as_array).cloned().unwrap_or_default())
            .filter(|p| p.get("type").and_then(Value::as_str) == Some("reasoning_text"))
            .filter_map(|p| str_of(p.get("text")))
            .collect();
    }
    if texts.is_empty() {
        texts = items
            .iter()
            .flat_map(|i| i.get("summary").and_then(Value::as_array).cloned().unwrap_or_default())
            .filter_map(|p| str_of(p.get("text")))
            .collect();
    }
    let signature = items.first().and_then(|i| str_of(i.get("encrypted_content")));
    Thinking::build((!texts.is_empty()).then(|| texts.join("\n")), signature)
}

pub fn parse_completion_body(provider: Provider, data: &Value, raw: RawResponse) -> Result<Message> {
    if let Some(msg) = data.pointer("/error/message").and_then(Value::as_str) {
        return Err(Error::Api(msg.into(), None));
    }
    let output = data.get("output").and_then(Value::as_array).cloned().unwrap_or_default();
    let content = parse_output_text(&output);
    let finish = parse_finish_reason(data);
    let server_calls = server_tool_items(&output);
    let mut m = Message::chunk();
    m.citations = parse_citations(provider, data, &output, content.as_deref());
    m.content = content;
    m.thinking = reasoning(&output, provider);
    m.tool_calls = tool_call_map(parse_tool_calls(&output, finish.as_ref().map(|f| f.as_str()))?);
    m.raw_content = (!server_calls.is_empty()).then(|| Value::Array(output.clone()));
    m.server_tool_calls = server_calls;
    m.model = str_of(data.get("model"));
    m.finish_reason = finish;
    parse_usage(provider, &mut m, data.get("usage").unwrap_or(&json!({})));
    m.raw = Some(raw);
    Ok(m.normalized())
}

pub fn build_chunk(provider: Provider, state: &mut StreamState, data: &Value) -> Result<Message> {
    let mut chunk = Message::chunk();
    let kind = data.get("type").and_then(Value::as_str).unwrap_or("");
    let position = (int(data.get("output_index")).unwrap_or(0), int(data.get("content_index")).unwrap_or(0));
    match kind {
        "response.output_text.delta" | "response.refusal.delta" => {
            let delta = str_of(data.get("delta")).unwrap_or_default();
            let len = delta.chars().count();
            match state.citation_lengths.iter_mut().find(|(p, _)| *p == position) {
                Some((_, l)) => *l += len,
                None => state.citation_lengths.push((position, len)),
            }
            chunk.content = Some(delta);
        }
        "response.reasoning_summary_text.delta" => chunk.thinking = Thinking::build(str_of(data.get("delta")), None),
        "response.reasoning_text.delta" if provider == Provider::DeepSeek => {
            chunk.thinking = Thinking::build(str_of(data.get("delta")), None)
        }
        "response.reasoning_summary_part.added" => {
            if int(data.get("summary_index")).unwrap_or(0) > 0 {
                chunk.thinking = Thinking::build(Some("\n\n".into()), None);
            }
        }
        "response.output_text.annotation.added" => {
            let by: usize = state.citation_lengths.iter().filter(|(p, _)| *p < position).map(|(_, l)| *l).sum();
            if let Some(c) = data.get("annotation").and_then(|a| parse_annotation(a, None)) {
                chunk.citations = vec![offset(c, by as i64, None)];
            }
        }
        "response.output_item.added" => {
            let item = &data["item"];
            if item.get("type").and_then(Value::as_str) == Some("function_call") && provider != Provider::Perplexity {
                let key = position.0.to_string();
                chunk.tool_calls = Some(
                    [(key, ToolCall { id: str_of(item.get("call_id")).unwrap_or_default(), name: str_of(item.get("name")).unwrap_or_default(), arguments: ToolArguments::Partial(String::new()), thought_signature: None, remote: false })]
                        .into_iter()
                        .collect(),
                );
            }
        }
        "response.function_call_arguments.delta" => {
            let key = position.0.to_string();
            chunk.tool_calls = Some(
                [(key, ToolCall { id: String::new(), name: String::new(), arguments: ToolArguments::Partial(str_of(data.get("delta")).unwrap_or_default()), thought_signature: None, remote: false })]
                    .into_iter()
                    .collect(),
            );
        }
        "response.output_item.done" => {
            let item = &data["item"];
            match item.get("type").and_then(Value::as_str) {
                Some("reasoning") => {
                    if let Some(sig) = str_of(item.get("encrypted_content")) {
                        chunk.thinking = Thinking::build(None, Some(sig));
                    }
                }
                Some("function_call") if provider == Provider::Perplexity => {
                    let key = position.0.to_string();
                    chunk.tool_calls = Some(
                        [(key, ToolCall { id: str_of(item.get("call_id")).unwrap_or_default(), name: str_of(item.get("name")).unwrap_or_default(), arguments: ToolArguments::Partial(str_of(item.get("arguments")).unwrap_or_default()), thought_signature: None, remote: false })]
                            .into_iter()
                            .collect(),
                    );
                }
                _ => {}
            }
        }
        "response.completed" | "response.incomplete" => {
            let response = data.get("response").cloned().unwrap_or_else(|| json!({}));
            let output = response.get("output").and_then(Value::as_array).cloned().unwrap_or_default();
            let finish = parse_finish_reason(&response);
            let approvals: Vec<ToolCall> = parse_tool_calls(&output, finish.as_ref().map(|f| f.as_str()))?
                .into_iter()
                .filter(|c| c.remote)
                .collect();
            chunk.tool_calls = tool_call_map(approvals);
            chunk.model = str_of(response.get("model"));
            chunk.citations = parse_citations(provider, &response, &output, None);
            let server_calls = server_tool_items(&output);
            chunk.raw_content = (!server_calls.is_empty()).then(|| Value::Array(output.clone()));
            chunk.server_tool_calls = server_calls;
            chunk.finish_reason = finish;
            parse_usage(provider, &mut chunk, response.get("usage").unwrap_or(&json!({})));
        }
        "response.failed" => {
            return Err(Error::Api(
                str_of(data.pointer("/response/error/message")).unwrap_or_else(|| "response failed".into()),
                None,
            ));
        }
        _ => {}
    }
    Ok(chunk)
}
