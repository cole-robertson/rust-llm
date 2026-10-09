//! Port of `lib/ruby_llm/protocols/chat_completions/{chat,tools,media,streaming}.rb` plus the
//! per-provider dialects that subclass it (`providers/{deepseek,mistral,openrouter,xai,ollama,
//! hetzner,gpustack}/chat.rb`).

use serde_json::{Map, Value, json};

use super::{
    Caching, Request, ToolCalls, ToolChoice, char_slice, deep_merge, int, normalize_finish_reason,
    str_of, tool_call_map,
};
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
                    let required: Vec<&str> = map
                        .get("required")
                        .and_then(Value::as_array)
                        .map(|r| r.iter().filter_map(Value::as_str).collect())
                        .unwrap_or_default();
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
            | Provider::Perplexity
    );
    match role {
        Role::System if !plain_roles && config.get("openai_use_system_role") != Some("true") => {
            "developer"
        }
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
        let field = if provider == Provider::OpenAI {
            "max_completion_tokens"
        } else {
            "max_tokens"
        };
        payload.insert(field.into(), max.into());
    }
    if !req.tools.is_empty() {
        payload.insert(
            "tools".into(),
            Value::Array(req.tools.iter().map(|t| tool_for(t.as_ref())).collect()),
        );
        if let Some(choice) = &req.tool_prefs.choice {
            payload.insert("tool_choice".into(), build_tool_choice(provider, choice));
        }
        if let Some(calls) = req.tool_prefs.calls {
            payload.insert(
                "parallel_tool_calls".into(),
                (calls == ToolCalls::Many).into(),
            );
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
                if let Some(budget) = thinking.budget {
                    tracing::debug!("DeepSeek has no thinking budgets; ignoring budget {budget}");
                }
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
            let single_tool = payload
                .get("tools")
                .and_then(Value::as_array)
                .filter(|t| t.len() == 1)
                .cloned();
            if payload.get("tool_choice").and_then(Value::as_str) == Some("any")
                && let Some(name) = single_tool
                    .as_ref()
                    .and_then(|t| t[0].pointer("/function/name"))
                    .filter(|n| !n.is_null())
                    .cloned()
            {
                payload.insert(
                    "tool_choice".into(),
                    json!({ "type": "function", "function": { "name": name } }),
                );
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
        // `providers/ollama/chat.rb#render_payload`.
        Provider::Ollama | Provider::OllamaCloud => {
            if let Some(budget) = req.thinking.and_then(|t| t.budget) {
                tracing::debug!("Ollama has no thinking budgets; ignoring budget {budget}");
            }
        }
        _ => {}
    }
    apply_prompt_cache_params(req, &mut payload)?;
    Ok(Value::Object(payload))
}

/// `apply_prompt_cache_params` with the Mistral (`key:` only) and OpenRouter (top-level
/// `cache_control`) overrides.
fn apply_prompt_cache_params(req: &Request, payload: &mut Map<String, Value>) -> Result<()> {
    match req.provider {
        Provider::Mistral => {
            if let Some(key) = Caching::checked(req.caching, &["key"], "Mistral")?
                .and_then(|o| o.get("key"))
                .filter(|k| !k.is_null())
            {
                payload.insert("prompt_cache_key".into(), key.clone());
            }
        }
        Provider::OpenRouter => {
            if let Some(options) = Caching::checked(req.caching, &["ttl"], "OpenRouter")? {
                payload.insert(
                    "cache_control".into(),
                    openrouter_cache_control(Some(options)),
                );
            }
        }
        _ => {
            if let Some(options) = Caching::checked(
                req.caching,
                super::responses::PROMPT_CACHE_OPTIONS,
                "Chat Completions",
            )? {
                payload.extend(super::responses::prompt_cache_params(options));
            }
        }
    }
    Ok(())
}

/// OpenRouter's `prompt_cache_control(caching)`.
fn openrouter_cache_control(options: Option<&Map<String, Value>>) -> Value {
    let mut control = json!({ "type": "ephemeral" });
    if let Some(ttl) = options.and_then(|o| o.get("ttl")).filter(|t| !t.is_null()) {
        control["ttl"] = ttl.clone();
    }
    control
}

fn format_messages(req: &Request) -> Result<Vec<Value>> {
    let (system, other): (Vec<&Message>, Vec<&Message>) =
        req.messages.iter().partition(|m| m.role == Role::System);
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
                    let mut parts = vec![
                        json!({ "type": "text", "text": format!("Attachments from tool call {}:", msg.tool_call_id.as_deref().unwrap_or("")) }),
                    ];
                    if let Value::Array(more) =
                        format_content(req.provider, None, &msg.attachments)?
                    {
                        parts.extend(more);
                    }
                    out.push(json!({ "role": "user", "content": parts }));
                }
            }
        } else if let Some(replayed) = (req.provider == Provider::Mistral)
            .then(|| super::mistral::multi_messages_for_replay(ordered[i]))
            .flatten()
        {
            // `MultiCompletion#format_message_group`: a multi-completion answer replays its messages.
            out.extend(replayed);
            i += 1;
        } else {
            out.push(format_message(req, ordered[i])?);
            i += 1;
        }
    }
    Ok(out)
}

fn format_message(req: &Request, msg: &Message) -> Result<Value> {
    let provider = req.provider;
    let attachments: &[Attachment] = if msg.is_tool_result() {
        &[]
    } else {
        &msg.attachments
    };
    let mut content = format_content(provider, msg.content.as_deref(), attachments)?;
    let thinking_only =
        msg.role == Role::Assistant && msg.thinking.is_some() && !msg.is_tool_call();
    if content.is_null() && thinking_only {
        content = Value::String(String::new());
    }
    if provider == Provider::Mistral
        && msg.role == Role::Assistant
        && let Some(thinking) = &msg.thinking
    {
        // `build_thinking_blocks(msg)`: only a signature Mistral produced (`own_signature`).
        let signature = msg.own_signature(Provider::Mistral.slug());
        let mut blocks = Vec::new();
        if let Some(text) = &thinking.text {
            let mut block =
                json!({ "type": "thinking", "thinking": [{ "type": "text", "text": text }] });
            if let Some(sig) = signature {
                block["signature"] = sig.into();
            }
            blocks.push(block);
        } else if let Some(sig) = signature {
            blocks.push(json!({ "type": "thinking", "signature": sig }));
        }
        match content {
            Value::Array(parts) => blocks.extend(parts),
            Value::String(s) if !s.is_empty() => blocks.push(json!({ "type": "text", "text": s })),
            _ => {}
        }
        content = Value::Array(blocks);
    }
    let boundary = msg.cache_until_here && Caching::boundaries(req.caching);
    if boundary && provider == Provider::OpenRouter {
        content = inject_boundary_cache_control(content, req.caching, msg.cache_ttl.as_deref())?;
    } else if boundary && provider != Provider::Mistral {
        // `inject_cache_breakpoint` runs wherever `openai_prompt_caching?` holds: every Chat
        // Completions provider but Mistral and OpenRouter (`protocols/chat_completions/chat.rb`).
        let parts = match &content {
            Value::Array(parts) => Some(parts.clone()),
            Value::String(s) if !s.is_empty() => Some(vec![json!({ "type": "text", "text": s })]),
            _ => None,
        };
        if let Some(mut parts) = parts
            && let Some(Value::Object(last)) = parts.last_mut()
        {
            last.insert(
                "prompt_cache_breakpoint".into(),
                json!({ "mode": "explicit" }),
            );
            content = Value::Array(parts);
        }
    }

    let mut out = Map::new();
    out.insert(
        "role".into(),
        format_role(provider, msg.role, req.config).into(),
    );
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
                // The Mistral dialect's `format_tool_calls` drops `extra_content`.
                if let Some(sig) = tc.thought_signature.as_ref().filter(|_| provider != Provider::Mistral) {
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

/// `OpenRouter::Chat#inject_cache_control` (`providers/openrouter/chat.rb`): marks the last block
/// of a cache-boundary message with `cache_control`, wrapping non-array content in a text block.
/// An empty list, a trailing non-object, or a block already carrying `cache_control` is left alone.
#[doc(hidden)]
pub fn inject_cache_control(content: Value, caching: Option<&Caching>) -> Result<Value> {
    inject_boundary_cache_control(content, caching, None)
}

/// `inject_cache_control(content, caching:, ttl: msg.cache_ttl)`: the boundary's own lifetime
/// wins over the chat's.
fn inject_boundary_cache_control(
    content: Value,
    caching: Option<&Caching>,
    ttl: Option<&str>,
) -> Result<Value> {
    let mut blocks = match content {
        Value::Array(parts) => parts,
        other => vec![json!({ "type": "text", "text": other })],
    };
    if let Some(Value::Object(last)) = blocks.last_mut()
        && last.get("cache_control").is_none_or(Value::is_null)
    {
        let mut control =
            openrouter_cache_control(Caching::checked(caching, &["ttl"], "OpenRouter")?);
        if let Some(ttl) = ttl {
            control["ttl"] = ttl.into();
        }
        last.insert("cache_control".into(), control);
    }
    Ok(Value::Array(blocks))
}

fn format_thinking(provider: Provider, msg: &Message, out: &mut Map<String, Value>) {
    match provider {
        Provider::Mistral => {}
        Provider::DeepSeek => {
            let text = msg
                .thinking
                .as_ref()
                .and_then(|t| t.text.clone())
                .unwrap_or_default();
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
            // Rebuilt only for an answer OpenRouter produced (`producer_slug(msg) == @provider.slug`).
            let Some(t) = msg.thinking.as_ref().filter(|_| {
                msg.producing_entry()
                    .is_some_and(|e| e.provider == Provider::OpenRouter.slug())
            }) else {
                return;
            };
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
pub(crate) fn format_content(
    provider: Provider,
    content: Option<&str>,
    attachments: &[Attachment],
) -> Result<Value> {
    if attachments.is_empty() {
        return Ok(content
            .map(|c| Value::String(c.to_string()))
            .unwrap_or(Value::Null));
    }
    let mut parts = Vec::new();
    if let Some(c) = content {
        parts.push(text_part(c));
    }
    let unsupported =
        |a: &Attachment| Error::UnsupportedAttachment(super::anthropic::unsupported(&a.mime_type));
    for a in attachments {
        let kind = a.kind();
        // `Media.format_provider_file`, reached by the providers that use the shared
        // `format_attachment`; those with `document_attachments: :none` refuse it.
        let own_media = matches!(
            provider,
            Provider::Mistral
                | Provider::Ollama
                | Provider::OllamaCloud
                | Provider::GPUStack
                | Provider::Perplexity
        ) || (provider == Provider::OpenRouter && kind == AttachmentType::Video);
        if let Some(file_id) = a.provider_file_id().filter(|_| !own_media) {
            if matches!(
                provider,
                Provider::DeepSeek | Provider::XAI | Provider::Hetzner
            ) {
                return Err(unsupported(a));
            }
            parts.push(json!({ "type": "file", "file": { "file_id": file_id } }));
            continue;
        }
        let part = match (provider, kind) {
            (Provider::Mistral, AttachmentType::Image) => {
                json!({ "type": "image_url", "image_url": a.url_or_data_uri()? })
            }
            (Provider::Mistral, AttachmentType::Pdf | AttachmentType::Document) => {
                json!({ "type": "document_url", "document_url": a.url_or_data_uri()? })
            }
            (
                Provider::Ollama | Provider::OllamaCloud | Provider::GPUStack,
                AttachmentType::Image,
            ) => {
                json!({ "type": "image_url", "image_url": { "url": a.for_llm()?, "detail": "auto" } })
            }
            // `providers/gpustack/media.rb#format_video`: always inline, clusters often lack internet.
            (Provider::GPUStack, AttachmentType::Video) => {
                json!({ "type": "video_url", "video_url": { "url": a.for_llm()? } })
            }
            (
                Provider::Ollama | Provider::OllamaCloud | Provider::GPUStack,
                AttachmentType::Pdf | AttachmentType::Document,
            ) => {
                return Err(unsupported(a));
            }
            (Provider::Hetzner, AttachmentType::Image) => {
                json!({ "type": "image_url", "image_url": { "url": a.for_llm()? } })
            }
            // `providers/perplexity/media.rb`: images without a detail, documents as `file_url`
            // parts (the URL itself, or bare base64), text files as text, nothing else. 2.1 leaves
            // the document formats to Perplexity.
            (Provider::Perplexity, AttachmentType::Image) => {
                json!({ "type": "image_url", "image_url": { "url": a.url_or_data_uri()? } })
            }
            (Provider::Perplexity, AttachmentType::Pdf | AttachmentType::Document) => {
                let url = match a.url() {
                    Some(u) => u.to_string(),
                    None => a.encoded()?,
                };
                json!({ "type": "file_url", "file_url": { "url": url } })
            }
            (Provider::Perplexity, AttachmentType::Audio) => return Err(unsupported(a)),
            (Provider::OpenRouter, AttachmentType::Video) => {
                json!({ "type": "video_url", "video_url": { "url": a.url_or_data_uri()? } })
            }
            (
                Provider::DeepSeek | Provider::XAI | Provider::Hetzner,
                AttachmentType::Pdf | AttachmentType::Document,
            ) => {
                return Err(unsupported(a));
            }
            (Provider::DeepSeek | Provider::XAI | Provider::Hetzner, AttachmentType::Audio) => {
                return Err(unsupported(a));
            }
            (_, AttachmentType::Image) => {
                let mut part =
                    json!({ "type": "image_url", "image_url": { "url": a.url_or_data_uri()? } });
                if let Some(res) = a.resolution {
                    part["image_url"]["detail"] =
                        res.image_detail(provider.is_original_image_detail()).into();
                }
                part
            }
            (_, AttachmentType::Audio) => {
                json!({ "type": "input_audio", "input_audio": { "data": a.encoded()?, "format": a.format() } })
            }
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
    Some(
        (prompt - cache_read_tokens(usage).unwrap_or(0) - cache_write_tokens(usage).unwrap_or(0))
            .max(0),
    )
}

fn output_tokens(usage: &Value) -> Option<i64> {
    let completion = int(usage.get("completion_tokens"))?;
    let generated = match (
        int(usage.get("prompt_tokens")),
        int(usage.get("total_tokens")),
    ) {
        (Some(p), Some(t)) => Some((t - p).max(0)),
        _ => None,
    };
    Some(match generated {
        Some(g) if g > completion => g,
        _ => completion,
    })
}

fn cache_read_tokens(usage: &Value) -> Option<i64> {
    int(usage.pointer("/prompt_tokens_details/cached_tokens"))
        .or_else(|| int(usage.get("prompt_cache_hit_tokens")))
}

fn cache_write_tokens(usage: &Value) -> Option<i64> {
    int(usage.pointer("/prompt_tokens_details/cache_write_tokens"))
        .or_else(|| int(usage.pointer("/input_tokens_details/cache_write_tokens")))
        .or(Some(0))
}

fn thinking_tokens(usage: &Value) -> Option<i64> {
    int(usage.pointer("/completion_tokens_details/reasoning_tokens"))
        .or_else(|| int(usage.get("reasoning_tokens")))
}

pub(crate) fn reported_cost(provider: Provider, usage: &Value) -> Option<f64> {
    match provider {
        Provider::OpenRouter => {
            let mut cost = usage.get("cost")?.as_f64()?;
            if usage.get("is_byok").and_then(Value::as_bool) == Some(true) {
                cost += usage
                    .pointer("/cost_details/upstream_inference_cost")
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0);
            }
            Some(cost)
        }
        Provider::XAI => usage
            .get("cost_in_usd_ticks")
            .and_then(Value::as_f64)
            .map(|t| t * 1e-10),
        _ => None,
    }
}

fn fill_usage(provider: Provider, message: &mut Message, usage: &Value) {
    message.tokens.input = input_tokens(usage);
    message.tokens.output = output_tokens(usage);
    message.tokens.cache_read = cache_read_tokens(usage);
    message.tokens.cache_write = if usage.is_object() && !usage.as_object().unwrap().is_empty() {
        cache_write_tokens(usage)
    } else {
        None
    }; // is_object() checked first
    message.tokens.thinking = thinking_tokens(usage);
    message.tokens.server_tool_use =
        crate::tokens::Tokens::positive_counts(&Value::Object(server_tool_use(usage)));
    message.tokens.reported_cost = reported_cost(provider, usage);
}

/// `server_tool_use(usage)`: per-tool counters end in `_requests`; OpenRouter's
/// `tool_calls_requested` and `tool_calls_executed` count every tool at once and are left out.
pub(crate) fn server_tool_use(usage: &Value) -> Map<String, Value> {
    usage
        .get("server_tool_use")
        .or_else(|| usage.get("server_tool_use_details"))
        .and_then(Value::as_object)
        .into_iter()
        .flatten()
        .filter(|(counter, _)| counter.ends_with("_requests"))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect()
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
            (
                (!text.is_empty()).then_some(text),
                (!thinking.is_empty()).then_some(thinking),
            )
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
    if let Some(results) = data
        .get("search_results")
        .and_then(Value::as_array)
        .filter(|r| !r.is_empty())
    {
        return parse_search_results(results);
    }
    data.get("citations")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .enumerate()
        .filter_map(|(i, url)| {
            url.as_str().map(|u| Citation {
                url: Some(u.into()),
                source_index: Some(i as i64),
                ..Default::default()
            })
        })
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

fn parse_tool_calls(
    calls: Option<&Value>,
    parse_arguments: bool,
    stream_keys: bool,
    finish: Option<&str>,
) -> Result<Vec<(String, ToolCall)>> {
    let Some(calls) = calls.and_then(Value::as_array).filter(|c| !c.is_empty()) else {
        return Ok(Vec::new());
    };
    calls
        .iter()
        .map(|tc| {
            let raw_args = tc
                .pointer("/function/arguments")
                .and_then(Value::as_str)
                .unwrap_or("");
            let arguments = if parse_arguments {
                if raw_args.is_empty() {
                    ToolArguments::Parsed(Map::new())
                } else {
                    ToolArguments::Parsed(
                        serde_json::from_str(raw_args)
                            .map_err(|_| Error::tool_call_parse(finish))?,
                    )
                }
            } else {
                ToolArguments::Partial(raw_args.to_string())
            };
            let id = str_of(tc.get("id")).unwrap_or_default();
            let key = if stream_keys {
                tc.get("index")
                    .map(|i| i.to_string())
                    .unwrap_or_else(|| id.clone())
            } else {
                id.clone()
            };
            Ok((
                key,
                ToolCall {
                    id,
                    name: str_of(tc.pointer("/function/name")).unwrap_or_default(),
                    arguments,
                    thought_signature: str_of(
                        tc.pointer("/extra_content/google/thought_signature"),
                    ),
                    remote: false,
                    starts: tc.get("id").is_some_and(|v| !v.is_null()),
                },
            ))
        })
        .collect()
}

pub fn parse_completion_body(
    provider: Provider,
    data: &Value,
    raw: RawResponse,
) -> Result<Message> {
    // `raise_stream_error(JSON.generate(data), data, nil)`: a 200 body that only reports an error
    // is classified like a stream error event (OpenRouter puts the HTTP status in its code).
    if data.pointer("/error/message").is_some_and(|m| !m.is_null()) {
        let body = data.to_string();
        let status = super::streaming_error_status(super::ProtocolName::ChatCompletions)(&body)
            .unwrap_or(500);
        return Err(crate::error::error_for_status_message(
            status,
            &body,
            provider.parse_error(&body),
        ));
    }
    let Some(message_data) = data.pointer("/choices/0/message") else {
        let mut message = "Provider returned no completion message".to_string();
        if let Some(r) = data
            .pointer("/choices/0/finish_reason")
            .and_then(Value::as_str)
        {
            message = format!("{message} (finish_reason: {r})");
        }
        // `Error.new(message, response: raw)`.
        let response = crate::error::ErrorResponse {
            status: raw.status,
            body: raw.body.to_string(),
            ..Default::default()
        };
        return Err(Error::Api(message, Some(response)));
    };
    let usage = data.get("usage").cloned().unwrap_or_else(|| json!({}));
    let finish_raw = data
        .pointer("/choices/0/finish_reason")
        .and_then(Value::as_str);
    let finish = normalize_finish_reason(finish_raw, FINISH_REASONS);
    let (content, block_thinking) = extract_content_and_thinking(message_data.get("content"));
    let (thinking_text, signature) = if provider == Provider::OpenRouter {
        (
            block_thinking.or_else(|| openrouter_thinking_text(message_data)),
            openrouter_thinking_signature(message_data),
        )
    } else {
        // `extract_thinking_text`/`extract_thinking_signature`: the first truthy field wins
        // (`a || b || c`), and only counts if it is a string.
        let first_string = |keys: &[&str]| {
            keys.iter()
                .find_map(|k| {
                    message_data
                        .get(*k)
                        .filter(|v| !v.is_null() && **v != Value::Bool(false))
                })
                .and_then(Value::as_str)
                .map(str::to_string)
        };
        (
            block_thinking
                .or_else(|| first_string(&["reasoning_content", "reasoning", "thinking"])),
            first_string(&["reasoning_signature", "signature"]),
        )
    };
    let calls = parse_tool_calls(
        message_data.get("tool_calls"),
        true,
        false,
        finish.as_ref().map(|f| f.as_str()),
    )?;

    let mut m = Message::chunk();
    let mut citations = parse_annotations(message_data.get("annotations"), content.as_deref());
    if citations.is_empty() {
        citations = parse_root_citations(data);
    }
    m.citations = citations;
    m.content = content;
    m.thinking = Thinking::build(thinking_text, signature);
    if provider == Provider::OpenRouter {
        m.raw_reasoning = message_data
            .get("reasoning_details")
            .filter(|d| d.as_array().is_some_and(|a| !a.is_empty()))
            .cloned();
    }
    m.tool_calls = tool_call_map(calls.into_iter().map(|(_, c)| c).collect());
    fill_usage(provider, &mut m, &usage);
    m.finish_reason = finish;
    m.model = str_of(data.get("model"));
    m.raw = Some(raw);
    Ok(m.normalized())
}

/// `OpenRouter::Chat#extract_thinking_text` (`providers/openrouter/chat.rb`): the `reasoning`
/// string, else the joined `reasoning.text` and `reasoning.summary` details; `None` when empty.
fn openrouter_thinking_text(data: &Value) -> Option<String> {
    if let Some(reasoning) = data.get("reasoning").and_then(Value::as_str) {
        return Some(reasoning.to_string());
    }
    let details = data.get("reasoning_details")?.as_array()?;
    let text: String = details
        .iter()
        .filter_map(|d| match d.get("type").and_then(Value::as_str) {
            Some("reasoning.text") => d.get("text").and_then(Value::as_str),
            Some("reasoning.summary") => d.get("summary").and_then(Value::as_str),
            _ => None,
        })
        .collect();
    (!text.is_empty()).then_some(text)
}

/// `OpenRouter::Chat#extract_thinking_signature`: the first explicit detail signature, else the
/// first `reasoning.encrypted` detail's `data`.
fn openrouter_thinking_signature(data: &Value) -> Option<String> {
    let details = data.get("reasoning_details")?.as_array()?;
    details
        .iter()
        .find_map(|d| d.get("signature").and_then(Value::as_str))
        .or_else(|| {
            details
                .iter()
                .find(|d| {
                    d.get("type").and_then(Value::as_str) == Some("reasoning.encrypted")
                        && d.get("data").is_some_and(Value::is_string)
                })
                .and_then(|d| d.get("data").and_then(Value::as_str))
        })
        .map(str::to_string)
}

/// `OpenRouter::Streaming#accumulate_raw_reasoning` (`providers/openrouter/streaming.rb`): each
/// delta's reasoning_details merge into the stream's list, matching an entry by `index` and `type`.
/// `text`, `data`, and `summary` strings are appended; other keys are only filled when missing
/// (the closing signature arrives in its own fragment). Every chunk carries the list so far.
pub fn accumulate_raw_reasoning(
    acc: &mut Option<Vec<Value>>,
    details: Option<&Value>,
) -> Option<Value> {
    const ACCUMULATED_REASONING_KEYS: [&str; 3] = ["text", "data", "summary"];
    if let Some(details) = details.and_then(Value::as_array).filter(|d| !d.is_empty()) {
        let entries = acc.get_or_insert_with(Vec::new);
        for detail in details {
            let target = detail
                .get("index")
                .filter(|i| !i.is_null())
                .and_then(|index| {
                    entries.iter_mut().find(|e| {
                        e.get("index") == Some(index) && e.get("type") == detail.get("type")
                    })
                });
            let (Some(target), Some(fields)) = (target, detail.as_object()) else {
                entries.push(detail.clone());
                continue;
            };
            for (key, value) in fields {
                match (value, target.get_mut(key)) {
                    (Value::String(more), Some(Value::String(existing)))
                        if ACCUMULATED_REASONING_KEYS.contains(&key.as_str()) =>
                    {
                        existing.push_str(more)
                    }
                    (_, None | Some(Value::Null)) => {
                        target[key.as_str()] = value.clone();
                    }
                    _ => {}
                }
            }
        }
    }
    acc.clone().map(Value::Array)
}

pub fn build_chunk(provider: Provider, data: &Value) -> Message {
    let usage = data.get("usage").cloned().unwrap_or_else(|| json!({}));
    let delta = data
        .pointer("/choices/0/delta")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let content_source = delta
        .get("content")
        .or_else(|| data.pointer("/choices/0/message/content"));
    let (content, block_thinking) = extract_content_and_thinking(content_source);
    let mut m = Message::chunk();
    m.model = str_of(data.get("model"));
    m.content = content;
    let mut citations = parse_annotations(delta.get("annotations"), None);
    if citations.is_empty() {
        citations = parse_root_citations(data);
    }
    m.citations = citations;
    m.thinking = if provider == Provider::OpenRouter {
        // `OpenRouter::Streaming#build_chunk` reads thinking from the delta's reasoning fields only.
        Thinking::build(
            openrouter_thinking_text(&delta),
            openrouter_thinking_signature(&delta),
        )
    } else {
        let text = block_thinking
            .or_else(|| str_of(delta.get("reasoning_content")))
            .or_else(|| str_of(delta.get("reasoning")));
        Thinking::build(text, str_of(delta.get("reasoning_signature")))
    };
    if let Ok(calls) = parse_tool_calls(delta.get("tool_calls"), false, true, None)
        && !calls.is_empty()
    {
        m.tool_calls = Some(calls.into_iter().collect());
    }
    if usage.as_object().is_some_and(|u| !u.is_empty()) {
        fill_usage(provider, &mut m, &usage);
    }
    m.finish_reason = normalize_finish_reason(
        data.pointer("/choices/0/finish_reason")
            .and_then(Value::as_str),
        FINISH_REASONS,
    );
    m
}
