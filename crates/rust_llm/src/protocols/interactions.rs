//! Port of `lib/ruby_llm/protocols/interactions.rb` and `protocols/interactions/{chat,content,tools,
//! streaming}.rb`: Gemini's Interactions API, opted into with `protocol: :interactions`. Requests are
//! stateless (`store: false`); an answer keeps the whole interaction in `raw_content` so its signed
//! steps replay verbatim on the next turn. (`interactions/transcription.rb` lives in `transcription.rs`.)

use std::collections::BTreeMap;

use base64::Engine;
use serde_json::{Map, Value, json};

use super::{Request, ToolChoice, int, str_of};
use crate::attachment::{Attachment, AttachmentType};
use crate::error::{Error, Result};
use crate::message::{
    Citation, FinishReason, Message, RawResponse, Role, ServerToolCall, Thinking, ToolCall,
};
use crate::model::Model;
use crate::thinking::{Display, ThinkingConfig};
use crate::tool::tool_schema;

const CITATION_TYPES: &[&str] = &["url_citation", "file_citation", "place_citation"];

/// `Interactions::Chat#render_payload`.
pub fn render_payload(req: &Request) -> Result<Value> {
    let mut config = Map::new();
    if let Some(t) = req.temperature {
        config.insert("temperature".into(), t.into());
    }
    if let Some(max) = req.max_output_tokens {
        config.insert("max_output_tokens".into(), max.into());
    }
    if let Some(thinking) = req.thinking {
        config.extend(render_interaction_thinking(thinking)?);
    }
    if let Some(choice) = &req.tool_prefs.choice {
        config.insert("tool_choice".into(), render_interaction_choice(choice));
    }
    let system: Vec<String> = req
        .messages
        .iter()
        .filter(|m| m.role == Role::System)
        .map(|m| m.content().to_string())
        .collect();
    let mut payload = json!({
        "model": req.model.id,
        "input": format_interaction_input(req.messages)?,
        "stream": req.stream,
        "store": false,
        "system_instruction": system.join("\n\n"),
        "generation_config": config,
        "tools": render_interaction_tools(req),
    });
    if let Some(schema) = req.schema {
        payload["response_format"] =
            json!({ "type": "text", "mime_type": "application/json", "schema": schema.schema });
    }
    Ok(payload)
}

/// `render_interaction_thinking`: effort maps to `thinking_level`, display to `thinking_summaries`.
pub fn render_interaction_thinking(thinking: &ThinkingConfig) -> Result<Map<String, Value>> {
    if thinking.is_disabled() || thinking.budget == Some(0) {
        return Err(Error::Argument(
            "Gemini Interactions does not expose a thinking-off control".into(),
        ));
    }
    if thinking.budget.is_some() {
        return Err(Error::Argument(
            "Gemini Interactions accepts thinking effort, not a token budget".into(),
        ));
    }
    if let Some(effort) = &thinking.effort
        && !["minimal", "low", "medium", "high"].contains(&effort.as_str())
    {
        return Err(Error::Argument(
            "Gemini Interactions thinking effort must be minimal, low, medium, or high".into(),
        ));
    }
    let summaries = match thinking.display {
        None | Some(Display::Summarized) => "auto",
        Some(Display::Omitted) => "none",
        Some(Display::Full) => {
            return Err(Error::Argument(
                "Gemini Interactions thinking display must be summarized or omitted".into(),
            ));
        }
    };
    let mut config = Map::new();
    if let Some(effort) = &thinking.effort {
        config.insert("thinking_level".into(), effort.clone().into());
    }
    config.insert("thinking_summaries".into(), summaries.into());
    Ok(config)
}

/// `Interactions::Tools#render_interaction_tools`.
fn render_interaction_tools(req: &Request) -> Value {
    let tools: Vec<Value> = req
        .tools
        .iter()
        .map(|tool| {
            let mut definition = json!({ "type": "function", "name": tool.name(), "description": tool.description() });
            if let Some(parameters) = tool_schema(tool.as_ref()) {
                definition["parameters"] = parameters;
            }
            let options = tool.provider_options();
            if !options.is_empty() {
                super::deep_merge(&mut definition, &Value::Object(options));
            }
            definition
        })
        .collect();
    Value::Array(tools)
}

/// `render_interaction_choice`.
fn render_interaction_choice(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Required => "any".into(),
        ToolChoice::Auto => "auto".into(),
        ToolChoice::None => "none".into(),
        ToolChoice::Tool(name) => json!({ "allowed_tools": { "mode": "any", "tools": [name] } }),
    }
}

/// `format_interaction_input`: every non-system turn as Interaction steps.
fn format_interaction_input(messages: &[Message]) -> Result<Vec<Value>> {
    let calls: BTreeMap<&str, &ToolCall> = messages
        .iter()
        .flat_map(|m| m.tool_calls.iter().flat_map(|c| c.values()))
        .map(|c| (c.id.as_str(), c))
        .collect();
    let mut input = Vec::new();
    // Gemini checks signatures only in the current turn, which starts at the last user message.
    let turn_start = messages.iter().rposition(|m| m.role == Role::User);
    for (index, message) in messages.iter().enumerate() {
        if message.role == Role::System {
            continue;
        }
        if let Some(raw) = message
            .raw_content
            .as_ref()
            .filter(|r| is_interaction_state(r))
        {
            let steps = raw
                .pointer("/response/steps")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            input.extend(render_interaction_history(steps));
        } else if message.is_tool_result() {
            let id = message.tool_call_id.clone().unwrap_or_default();
            let mut result = json!({ "type": "function_result", "call_id": id });
            if let Some(call) = calls.get(id.as_str()) {
                result["name"] = call.name.clone().into();
            }
            result["result"] = Value::Array(render_interaction_content(
                message.content.as_deref(),
                &message.attachments,
            )?);
            input.push(result);
        } else {
            let current = turn_start.is_none_or(|start| index > start);
            input.extend(render_interaction_message(message, current)?);
        }
    }
    Ok(input)
}

/// `render_interaction_history`: MCP steps, and function steps the provider already answered,
/// replay without their signatures.
fn render_interaction_history(steps: Vec<Value>) -> Vec<Value> {
    let answered = interaction_answered_calls(&steps);
    steps
        .into_iter()
        .map(|mut step| {
            let mcp = step
                .get("type")
                .and_then(Value::as_str)
                .is_some_and(|t| t.starts_with("mcp_server_"));
            let id = str_of(step.get("id")).or_else(|| str_of(step.get("call_id")));
            if (mcp || id.is_some_and(|id| answered.contains(&id)))
                && let Some(o) = step.as_object_mut()
            {
                o.remove("signature");
            }
            step
        })
        .collect()
}

/// `render_interaction_message`: the first call of a current-turn step without a signature gets
/// the placeholder Gemini documents.
fn render_interaction_message(message: &Message, current: bool) -> Result<Vec<Value>> {
    let mut steps = Vec::new();
    let content = render_interaction_content(message.content.as_deref(), &message.attachments)?;
    if !content.is_empty() {
        let kind = if message.role == Role::Assistant {
            "model_output"
        } else {
            "user_input"
        };
        steps.push(json!({ "type": kind, "content": content }));
    }
    for (position, call) in message
        .tool_calls
        .iter()
        .flat_map(|c| c.values())
        .enumerate()
    {
        let mut step = json!({ "type": "function_call", "id": call.id, "name": call.name, "arguments": Value::Object(call.arguments()) });
        let signature = call.thought_signature.clone().or_else(|| {
            (current && position == 0).then(|| super::gemini::PLACEHOLDER_SIGNATURE.to_string())
        });
        if let Some(signature) = signature {
            step["signature"] = signature.into();
        }
        steps.push(step);
    }
    Ok(steps)
}

fn is_interaction_state(content: &Value) -> bool {
    content.pointer("/response/object").and_then(Value::as_str) == Some("interaction")
}

/// `Interactions::Content#render_interaction_content`.
fn render_interaction_content(
    text: Option<&str>,
    attachments: &[Attachment],
) -> Result<Vec<Value>> {
    let mut parts: Vec<Value> = text
        .filter(|t| !t.is_empty())
        .map(|t| json!({ "type": "text", "text": t }))
        .into_iter()
        .collect();
    for attachment in attachments {
        parts.push(render_interaction_attachment(attachment)?);
    }
    Ok(parts)
}

fn render_interaction_attachment(attachment: &Attachment) -> Result<Value> {
    let kind = attachment.kind();
    if kind == AttachmentType::Text {
        return Ok(json!({ "type": "text", "text": attachment.content_text()? }));
    }
    let kind = match kind {
        AttachmentType::Image => "image",
        AttachmentType::Audio => "audio",
        AttachmentType::Video => "video",
        AttachmentType::Pdf => "document",
        _ => {
            return Err(Error::Argument(format!(
                "Gemini Interactions does not support {} input",
                attachment.mime_type
            )));
        }
    };
    let mut part = json!({ "type": kind, "mime_type": attachment.mime_type });
    if attachment.is_provider_file() {
        part["uri"] = attachment.provider_file_uri().into();
    } else if let Some(url) = attachment.url() {
        part["uri"] = url.into();
    } else {
        part["data"] = attachment.encoded()?.into();
    }
    Ok(part)
}

/// `Interactions::Chat#parse_completion_body`.
pub fn parse_completion_body(
    model: &Model,
    data: &Value,
    raw: Option<RawResponse>,
) -> Result<Message> {
    let status = data.get("status").and_then(Value::as_str).unwrap_or("");
    if !["completed", "requires_action", "incomplete"].contains(&status) {
        let messages: Vec<&str> = data
            .get("errors")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|e| e.get("message").and_then(Value::as_str))
            .collect();
        let message = if messages.is_empty() {
            format!("Gemini interaction ended with status {status}")
        } else {
            messages.join("; ")
        };
        return Err(Error::Api(message, None));
    }
    let steps: Vec<Value> = data
        .get("steps")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let (text, attachments, citations) = parse_interaction_content(&steps);
    let calls = parse_interaction_calls(&steps)?;
    let status = interaction_status(status, &calls, &steps);
    if status == "requires_action" && calls.is_empty() {
        return Err(Error::Api(
            "Gemini interaction requires an unsupported action".into(),
            None,
        ));
    }
    let mut m = Message::chunk();
    m.content = Some(text);
    m.attachments = attachments;
    m.citations = citations;
    m.thinking = parse_interaction_thinking(&steps);
    m.server_tool_calls = parse_interaction_server_calls(&steps);
    m.raw_content = Some(kept_interaction(data, &steps));
    m.model = str_of(data.get("model")).or_else(|| Some(model.id.clone()));
    m.finish_reason = Some(if !calls.is_empty() {
        FinishReason::ToolCalls
    } else {
        match status {
            "completed" => FinishReason::Stop,
            "requires_action" => FinishReason::ToolCalls,
            _ => FinishReason::MaxTokens,
        }
    });
    m.tool_calls = super::tool_call_map(calls);
    parse_interaction_usage(&mut m, data.get("usage").unwrap_or(&Value::Null));
    m.tokens.server_tool_use =
        parse_interaction_server_tool_use(data.get("usage").unwrap_or(&Value::Null));
    m.raw = raw;
    Ok(m.normalized())
}

/// `interaction_status`: a `requires_action` interaction whose function calls the provider
/// already answered, and which went on to answer, is complete.
fn interaction_status<'a>(status: &'a str, calls: &[ToolCall], steps: &[Value]) -> &'a str {
    if status != "requires_action"
        || !calls.is_empty()
        || interaction_answered_calls(steps).is_empty()
    {
        return status;
    }
    if steps
        .iter()
        .any(|s| s.get("type").and_then(Value::as_str) == Some("model_output"))
    {
        "completed"
    } else {
        status
    }
}

/// `parse_interaction_server_tool_use`: `{tool}_requests` per grounding tool, Google Search
/// counted as `web_search`.
fn parse_interaction_server_tool_use(usage: &Value) -> Option<Map<String, Value>> {
    let counts: Map<String, Value> = usage
        .get("grounding_tool_count")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(|g| {
            let tool = g.get("type").and_then(Value::as_str).unwrap_or("");
            let tool = if tool == "google_search" {
                "web_search"
            } else {
                tool
            };
            (
                format!("{tool}_requests"),
                g.get("count").cloned().unwrap_or(Value::Null),
            )
        })
        .collect();
    crate::tokens::Tokens::positive_counts(&Value::Object(counts))
}

/// `kept_interaction`: the interaction with its steps' search suggestions left out.
fn kept_interaction(data: &Value, steps: &[Value]) -> Value {
    let mut data = data.clone();
    data["steps"] = Value::Array(steps.iter().map(without_search_suggestions).collect());
    json!({ "response": data })
}

/// `without_search_suggestions`: Google's terms forbid storing them, and a replayed
/// `google_search_result` keeps the shape it needs without them.
fn without_search_suggestions(step: &Value) -> Value {
    if step.get("type").and_then(Value::as_str) != Some("google_search_result") {
        return step.clone();
    }
    let mut step = step.clone();
    let result: Vec<Value> = match step.get("result") {
        Some(Value::Array(items)) => items.clone(),
        Some(Value::Null) | None => Vec::new(),
        Some(other) => vec![other.clone()],
    };
    let result: Vec<Value> = result
        .into_iter()
        .map(|mut item| {
            if let Some(o) = item.as_object_mut() {
                o.remove("search_suggestions");
            }
            item
        })
        .collect();
    step["result"] = Value::Array(result);
    step
}

/// `search_suggestions(step)`: the suggestions a step's results carry, joined.
fn search_suggestions(step: &Value) -> Option<String> {
    let suggestions: Vec<&str> = step
        .get("result")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|i| i.get("search_suggestions").and_then(Value::as_str))
        .collect();
    (!suggestions.is_empty()).then(|| suggestions.concat())
}

/// `interaction_answered_calls`: function calls whose result the interaction already carries.
fn interaction_answered_calls(steps: &[Value]) -> Vec<String> {
    let of = |kind: &str, key: &str| -> Vec<String> {
        steps
            .iter()
            .filter(|s| s.get("type").and_then(Value::as_str) == Some(kind))
            .filter_map(|s| str_of(s.get(key)))
            .collect()
    };
    let results = of("function_result", "call_id");
    of("function_call", "id")
        .into_iter()
        .filter(|id| results.contains(id))
        .collect()
}

/// `parse_interaction_usage`: tool-use tokens count as input; thoughts count as output.
fn parse_interaction_usage(m: &mut Message, usage: &Value) {
    let cached = int(usage.get("total_cached_tokens"));
    let thoughts = int(usage.get("total_thought_tokens"));
    m.tokens.input = int(usage.get("total_input_tokens")).map(|prompt| {
        (prompt + int(usage.get("total_tool_use_tokens")).unwrap_or(0) - cached.unwrap_or(0)).max(0)
    });
    m.tokens.output = int(usage.get("total_output_tokens")).map(|out| out + thoughts.unwrap_or(0));
    m.tokens.cache_read = cached;
    m.tokens.thinking = thoughts;
}

fn parse_interaction_thinking(steps: &[Value]) -> Option<Thinking> {
    let thoughts: Vec<&Value> = steps
        .iter()
        .filter(|s| s.get("type").and_then(Value::as_str) == Some("thought"))
        .collect();
    let text: String = thoughts
        .iter()
        .flat_map(|s| match s.get("summary") {
            Some(Value::Array(parts)) => parts.clone(),
            Some(part @ Value::Object(_)) => vec![part.clone()],
            _ => Vec::new(),
        })
        .filter_map(|p| p.get("text").and_then(Value::as_str).map(str::to_string))
        .collect();
    let signature = thoughts.last().and_then(|s| str_of(s.get("signature")));
    Thinking::build((!text.is_empty()).then_some(text), signature)
}

/// `parse_interaction_content`: text, attachments, and citations of the `model_output` steps.
fn parse_interaction_content(steps: &[Value]) -> (String, Vec<Attachment>, Vec<Citation>) {
    let mut text = String::new();
    let mut attachments = Vec::new();
    let mut citations = Vec::new();
    for step in steps
        .iter()
        .filter(|s| s.get("type").and_then(Value::as_str) == Some("model_output"))
    {
        let parts = match step.get("content") {
            Some(Value::Array(parts)) => parts.clone(),
            Some(part @ Value::Object(_)) => vec![part.clone()],
            _ => Vec::new(),
        };
        for part in &parts {
            if part.get("type").and_then(Value::as_str) == Some("text") {
                let part_text = part.get("text").and_then(Value::as_str).unwrap_or("");
                citations.extend(parse_interaction_citations(
                    part,
                    part_text,
                    text.chars().count() as i64,
                ));
                text.push_str(part_text);
            } else if let Some(uri) = part.get("uri").and_then(Value::as_str) {
                attachments.push(Attachment::new(uri));
            } else if let Some(data) = part.get("data").and_then(Value::as_str) {
                let bytes = base64::engine::general_purpose::STANDARD
                    .decode(data)
                    .unwrap_or_default();
                let mime = part.get("mime_type").and_then(Value::as_str);
                attachments.push(Attachment::from_bytes(
                    bytes,
                    format!("interaction_output_{}", attachments.len() + 1),
                    mime,
                ));
            }
        }
    }
    (text, attachments, citations)
}

fn parse_interaction_citations(part: &Value, text: &str, offset: i64) -> Vec<Citation> {
    part.get("annotations")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|a| {
            a.get("type")
                .and_then(Value::as_str)
                .is_some_and(|t| CITATION_TYPES.contains(&t))
        })
        .map(|a| {
            let start = citation_index(text, int(a.get("start_index")));
            let end = citation_index(text, int(a.get("end_index")));
            let page = int(a.get("page_number"));
            Citation {
                url: str_of(a.get("url")).or_else(|| str_of(a.get("document_uri"))),
                title: str_of(a.get("title"))
                    .or_else(|| str_of(a.get("file_name")))
                    .or_else(|| str_of(a.get("name"))),
                source_id: str_of(a.get("media_id")).or_else(|| str_of(a.get("place_id"))),
                cited_text: str_of(a.get("source")),
                start_page: page,
                end_page: page,
                start_index: start.map(|s| offset + s),
                end_index: end.map(|e| offset + e),
                text: match (start, end) {
                    (Some(s), Some(e)) => super::char_slice(text, s, e),
                    _ => None,
                },
                ..Default::default()
            }
        })
        .collect()
}

/// `interaction_citation_index`: a byte offset as a character offset.
fn citation_index(text: &str, bytes: Option<i64>) -> Option<i64> {
    let bytes = usize::try_from(bytes?).ok()?;
    let prefix = text.as_bytes().get(..bytes.min(text.len()))?;
    Some(String::from_utf8_lossy(prefix).chars().count() as i64)
}

/// `Interactions::Tools#parse_interaction_calls`.
pub fn parse_interaction_calls(steps: &[Value]) -> Result<Vec<ToolCall>> {
    let answered = interaction_answered_calls(steps);
    steps
        .iter()
        .filter(|s| s.get("type").and_then(Value::as_str) == Some("function_call"))
        .filter(|s| str_of(s.get("id")).is_none_or(|id| !answered.contains(&id)))
        .map(|step| {
            let mut call = ToolCall::new(
                str_of(step.get("id")).unwrap_or_default(),
                str_of(step.get("name")).unwrap_or_default(),
                parse_interaction_arguments(step.get("arguments"))?,
            );
            call.thought_signature = str_of(step.get("signature"));
            Ok(call)
        })
        .collect()
}

/// `parse_interaction_arguments`: absent or `""` is `{}`, a string is JSON, an object passes through.
pub fn parse_interaction_arguments(arguments: Option<&Value>) -> Result<Map<String, Value>> {
    match arguments {
        None | Some(Value::Null) => Ok(Map::new()),
        Some(Value::String(s)) if s.is_empty() => Ok(Map::new()),
        Some(Value::String(s)) => {
            serde_json::from_str(s).map_err(|_| Error::tool_call_parse(Some("tool_calls")))
        }
        Some(Value::Object(m)) => Ok(m.clone()),
        Some(_) => Err(Error::tool_call_parse(Some("tool_calls"))),
    }
}

fn parse_interaction_server_calls(steps: &[Value]) -> Vec<ServerToolCall> {
    let answered = interaction_answered_calls(steps);
    steps
        .iter()
        .filter_map(|step| {
            let kind = step.get("type").and_then(Value::as_str).unwrap_or("");
            if !(kind.ends_with("_call") || kind.ends_with("_result")) {
                return None;
            }
            let id = str_of(step.get("id")).or_else(|| str_of(step.get("call_id")));
            if kind.starts_with("function_") && !id.as_ref().is_some_and(|id| answered.contains(id))
            {
                return None;
            }
            let kept = without_search_suggestions(step);
            Some(ServerToolCall {
                kind: kind.into(),
                id,
                name: str_of(kept.get("name")),
                input: kept.get("arguments").cloned(),
                result: kept.get("result").cloned(),
                raw: kept,
                search_suggestions: search_suggestions(step),
            })
        })
        .collect()
}

/// `Interactions::Streaming`'s `@interaction_steps`, `@interaction_response`, `@interaction_done`.
#[derive(Default)]
pub struct StreamState {
    steps: BTreeMap<i64, Value>,
    response: Map<String, Value>,
    pub(crate) done: bool,
    /// The message the completed interaction parses to (`stream_response`'s return value).
    pub(crate) message: Option<Message>,
}

/// `Interactions::Streaming#build_chunk`.
pub fn build_chunk(model: &Model, state: &mut StreamState, data: &Value) -> Result<Message> {
    match data.get("event_type").and_then(Value::as_str) {
        Some("interaction.created") => {
            if let Some(Value::Object(i)) = data.get("interaction") {
                state.response.extend(i.clone());
            }
        }
        Some("interaction.completed") => {
            state.done = true;
            if let Some(Value::Object(i)) = data.get("interaction") {
                state.response.extend(i.clone());
            }
            let message = parse_completion_body(model, &streamed_interaction(state)?, None)?;
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
        Some("error") => {
            let message = data
                .pointer("/error/message")
                .and_then(Value::as_str)
                .unwrap_or("Gemini interaction failed");
            return Err(Error::Api(message.into(), None));
        }
        Some("step.start") => {
            if let (Some(index), Some(step)) = (int(data.get("index")), data.get("step")) {
                state.steps.insert(index, step.clone());
            }
        }
        Some("step.delta") => {
            let index = int(data.get("index")).unwrap_or(0);
            let Some(step) = state.steps.get_mut(&index) else {
                return Err(Error::Api(
                    format!("Gemini interaction delta for unknown step {index}"),
                    None,
                ));
            };
            return Ok(append_interaction_delta(
                step,
                data.get("delta").unwrap_or(&Value::Null),
            ));
        }
        _ => {}
    }
    Ok(Message::chunk())
}

fn append_interaction_delta(step: &mut Value, delta: &Value) -> Message {
    let mut chunk = Message::chunk();
    match delta.get("type").and_then(Value::as_str) {
        Some("text") => {
            let text = delta.get("text").and_then(Value::as_str).unwrap_or("");
            append_interaction_text(step, text);
            chunk.content = Some(text.to_string());
        }
        Some("text_annotation") => {
            append_interaction_text(step, "");
            if let Some(last) = step
                .get_mut("content")
                .and_then(Value::as_array_mut)
                .and_then(|c| c.last_mut())
            {
                let annotations = last
                    .as_object_mut()
                    .map(|o| o.entry("annotations").or_insert_with(|| json!([])));
                if let Some(Value::Array(a)) = annotations {
                    a.push(delta.get("annotation").cloned().unwrap_or(Value::Null));
                }
            }
        }
        Some("thought_summary") => {
            push_to(
                step,
                "summary",
                delta.get("content").cloned().unwrap_or(Value::Null),
            );
            chunk.thinking = Thinking::build(
                delta
                    .pointer("/content/text")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                None,
            );
        }
        Some("thought_signature") => {
            step["signature"] = delta.get("signature").cloned().unwrap_or(Value::Null)
        }
        Some("arguments_delta") => {
            let mut arguments = step
                .get("arguments")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            arguments.push_str(delta.get("arguments").and_then(Value::as_str).unwrap_or(""));
            step["arguments"] = arguments.into();
        }
        Some("image" | "audio" | "video" | "document") => push_to(step, "content", delta.clone()),
        _ => {
            if let (Some(s), Some(d)) = (step.as_object_mut(), delta.as_object()) {
                s.extend(
                    d.iter()
                        .filter(|(k, _)| *k != "type")
                        .map(|(k, v)| (k.clone(), v.clone())),
                );
            }
        }
    }
    chunk
}

fn push_to(step: &mut Value, key: &str, value: Value) {
    let Some(o) = step.as_object_mut() else {
        return;
    };
    let list = o.entry(key).or_insert_with(|| json!([]));
    if !list.is_array() {
        *list = json!([]);
    }
    if let Some(a) = list.as_array_mut() {
        a.push(value);
    }
}

fn append_interaction_text(step: &mut Value, text: &str) {
    let Some(o) = step.as_object_mut() else {
        return;
    };
    let content = o.entry("content").or_insert_with(|| json!([]));
    let Some(parts) = content.as_array_mut() else {
        return;
    };
    if parts
        .last()
        .and_then(|p| p.get("type"))
        .and_then(Value::as_str)
        != Some("text")
    {
        parts.push(json!({ "type": "text", "text": "" }));
    }
    if let Some(last) = parts.last_mut() {
        let joined = format!(
            "{}{text}",
            last.get("text").and_then(Value::as_str).unwrap_or("")
        );
        last["text"] = joined.into();
    }
}

/// `streamed_interaction`: the steps in index order, function arguments parsed.
fn streamed_interaction(state: &StreamState) -> Result<Value> {
    let steps = state
        .steps
        .values()
        .map(|step| {
            if step.get("type").and_then(Value::as_str) != Some("function_call") {
                return Ok(step.clone());
            }
            let mut step = step.clone();
            step["arguments"] = Value::Object(parse_interaction_arguments(step.get("arguments"))?);
            Ok(step)
        })
        .collect::<Result<Vec<_>>>()?;
    let mut response = state.response.clone();
    response.insert("steps".into(), Value::Array(steps));
    Ok(Value::Object(response))
}

/// `stream_response`'s ending: the completed interaction, or an error for a truncated stream.
pub(crate) fn finish_stream(state: &mut StreamState, raw: RawResponse) -> Result<Message> {
    if !state.done {
        return Err(Error::Api(
            "Gemini interaction stream ended before completion".into(),
            None,
        ));
    }
    let mut message = state.message.take().ok_or_else(|| {
        Error::Api(
            "Gemini interaction stream ended before completion".into(),
            None,
        )
    })?;
    message.raw = Some(raw);
    Ok(message)
}
