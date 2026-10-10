//! Port of `lib/ruby_llm/protocol/request_shapes.rb` (the `shape_*` helpers) and each protocol's
//! `parse_request_shape`: `protocols/{anthropic,chat_completions,gemini,interactions,responses}/
//! request_shapes.rb` and `protocols/mistral/conversations/request_shapes.rb`.
//!
//! A protocol reads its payload into turns of parts, keeping each piece's size and never its
//! content. Every reader takes any JSON: a payload a `before_request` hook reshaped describes what
//! it can and never fails.

use serde_json::{Map, Value};

use super::analysis::{Analysis, Pairing, PartSpec, Rule, Speaker, TurnSpec};
use super::{PartKind, RequestShape, Source, Unit};
use crate::providers::ProtocolName;

static EMPTY_MAP: std::sync::LazyLock<Map<String, Value>> = std::sync::LazyLock::new(Map::new);

/// `Protocol#request_shape` / `#parse_request_shape`.
pub(crate) fn describe(
    protocol: ProtocolName,
    provider: Option<&str>,
    model: Option<&str>,
    payload: &Value,
) -> Option<RequestShape> {
    let payload = payload.as_object()?;
    let ctx = Ctx {
        provider,
        model,
        payload,
    };
    match protocol {
        ProtocolName::Anthropic => anthropic(&ctx),
        ProtocolName::ChatCompletions | ProtocolName::RouterChatCompletions => {
            chat_completions(&ctx)
        }
        ProtocolName::Gemini => gemini(&ctx),
        ProtocolName::Interactions => interactions(&ctx),
        ProtocolName::Responses => responses(&ctx),
        ProtocolName::Conversations => conversations(&ctx),
    }
}

struct Ctx<'a> {
    provider: Option<&'a str>,
    model: Option<&'a str>,
    payload: &'a Map<String, Value>,
}

/// What `shape_request` takes beside the turns.
struct Request {
    instructions: Vec<PartSpec>,
    tool_names: Vec<Option<String>>,
    thinking_settings: Map<String, Value>,
    pairing: Pairing,
    rules: &'static [Rule],
}

impl Default for Request {
    fn default() -> Self {
        Request {
            instructions: Vec::new(),
            tool_names: Vec::new(),
            thinking_settings: Map::new(),
            pairing: Pairing::Id,
            rules: &[],
        }
    }
}

fn shape_request(ctx: &Ctx, turns: Vec<TurnSpec>, request: Request) -> RequestShape {
    let count = turns.len();
    let analysis = Analysis::new(turns, request.pairing, request.rules);
    let kept: Vec<_> = analysis
        .kept_turns()
        .into_iter()
        .map(TurnSpec::to_turn)
        .collect();
    RequestShape {
        provider: ctx.provider.map(str::to_string),
        model: ctx.model.map(str::to_string),
        step: analysis.step(),
        payload_keys: ctx.payload.keys().cloned().collect(),
        thinking_settings: request.thinking_settings,
        instructions: request.instructions.iter().map(PartSpec::to_part).collect(),
        omitted_turns: count - kept.len(),
        turns: kept,
        turn_count: count,
        tool_names: request.tool_names.into_iter().flatten().collect(),
        tool_rounds: analysis.tool_rounds(),
        problems: analysis.problems(),
    }
}

/// Reads one content block of a tool result in the protocol's format.
type BlockReader = dyn Fn(&Value) -> Vec<PartSpec>;

// ---- the shape_* helpers ------------------------------------------------------------------

fn hash(value: Option<&Value>) -> &Map<String, Value> {
    value.and_then(Value::as_object).unwrap_or(&EMPTY_MAP)
}

fn list(value: Option<&Value>) -> &[Value] {
    value.and_then(Value::as_array).map_or(&[], Vec::as_slice)
}

/// Ruby truthiness: `nil` and `false` are false.
fn truthy(value: Option<&Value>) -> Option<&Value> {
    value.filter(|v| !matches!(v, Value::Null | Value::Bool(false)))
}

/// Ruby `a || b` on payload values.
fn either<'a>(a: Option<&'a Value>, b: Option<&'a Value>) -> Option<&'a Value> {
    truthy(a).or(b)
}

/// `shape_name`: a non-empty String, or nothing.
fn name(value: Option<&Value>) -> Option<String> {
    value
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

/// `shape_length`: the characters of a String, or of the JSON of anything else.
fn length(value: Option<&Value>) -> Option<usize> {
    match value? {
        Value::Null => None,
        Value::String(s) => Some(s.chars().count()),
        other => Some(other.to_string().chars().count()),
    }
}

/// `shape_bytes`: base64 carries three bytes in every four characters, and its padding marks
/// the missing bytes of the last group.
fn bytes(data: Option<&Value>) -> Option<usize> {
    let data = data?.as_str()?;
    let len = data.chars().count() as i64;
    let breaks = data.chars().filter(|c| matches!(c, '\r' | '\n')).count() as i64;
    let padding = data.chars().filter(|&c| c == '=').count().min(2) as i64;
    Some((((len - breaks) * 3).div_euclid(4) - padding).max(0) as usize)
}

/// `shape_signature?`.
fn signature(value: Option<&Value>) -> bool {
    value.and_then(Value::as_str).is_some_and(|s| !s.is_empty())
}

/// `SETTING_VALUE = /\A[\w.-]{1,40}\z/`.
fn setting_word(s: &str) -> bool {
    (1..=40).contains(&s.chars().count())
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-'))
}

/// `shape_settings`: thinking settings are short words and numbers, so anything else is left out
/// rather than risk content.
fn settings(value: Option<&Value>, names: &[&str]) -> Map<String, Value> {
    let source = hash(value);
    names
        .iter()
        .filter_map(|n| source.get(*n).map(|v| (*n, v)))
        .filter(|(_, v)| match v {
            Value::Number(_) | Value::Bool(_) => true,
            Value::String(s) => setting_word(s),
            _ => false,
        })
        .map(|(n, v)| (n.to_string(), v.clone()))
        .collect()
}

fn merge(mut a: Map<String, Value>, b: Map<String, Value>) -> Map<String, Value> {
    a.extend(b);
    a
}

fn turn(index: usize, role: Option<&Value>, speaker: Speaker, parts: Vec<PartSpec>) -> TurnSpec {
    TurnSpec {
        index,
        role: name(role),
        speaker,
        parts,
    }
}

/// `shape_answer`: a turn on the user's side that carries tool results answers a model turn.
fn answer(parts: &[PartSpec]) -> Speaker {
    if parts.iter().any(PartSpec::is_result) {
        Speaker::Tool
    } else {
        Speaker::User
    }
}

fn instructions(text: Option<&Value>) -> Vec<PartSpec> {
    match text {
        None | Some(Value::Null) => Vec::new(),
        text => vec![shape_text(text, PartKind::Text, false)],
    }
}

fn shape_text(text: Option<&Value>, kind: PartKind, signed: bool) -> PartSpec {
    let measure = length(text);
    PartSpec {
        measure,
        signed,
        without_data: kind == PartKind::Text && measure.is_none(),
        ..PartSpec::new(kind)
    }
}

fn summary(value: Option<&Value>, signed: bool) -> PartSpec {
    let parts = list(value);
    let measure = (!parts.is_empty()).then(|| {
        parts
            .iter()
            .map(|p| length(hash(Some(p)).get("text")).unwrap_or(0))
            .sum()
    });
    PartSpec {
        measure,
        signed,
        ..PartSpec::new(PartKind::Thinking)
    }
}

fn tool_call(
    tool: Option<&Value>,
    arguments: Option<&Value>,
    id: Option<&Value>,
    signed: bool,
) -> PartSpec {
    PartSpec {
        name: name(tool),
        measure: length(arguments),
        call_id: name(id),
        signed,
        ..PartSpec::new(PartKind::ToolCall)
    }
}

/// `shape_tool_result`: a result made of content blocks counts the characters of its text, and
/// its other blocks follow it as parts of their own. `block` reads one block in the protocol's
/// format; without one the result measures its content whole.
fn tool_result(
    content: Option<&Value>,
    tool: Option<&Value>,
    id: Option<&Value>,
    block: Option<&BlockReader>,
) -> Vec<PartSpec> {
    let mut result = PartSpec {
        name: name(tool),
        result_id: name(id),
        ..PartSpec::new(PartKind::ToolResult)
    };
    let (Some(Value::Array(blocks)), Some(block)) = (content, block) else {
        result.measure = length(content);
        return vec![result];
    };
    let (texts, others): (Vec<PartSpec>, Vec<PartSpec>) = blocks
        .iter()
        .flat_map(|b| block(&Value::Object(hash(Some(b)).clone())))
        .partition(|p| p.kind == PartKind::Text);
    result.measure = Some(texts.iter().map(|p| p.measure.unwrap_or(0)).sum());
    std::iter::once(result).chain(others).collect()
}

fn other(kind: Option<&Value>, signed: bool) -> PartSpec {
    PartSpec {
        name: name(kind),
        signed,
        ..PartSpec::new(PartKind::Other)
    }
}

fn media_kind(mime: Option<&str>) -> PartKind {
    match mime.unwrap_or("") {
        m if m.starts_with("image/") => PartKind::Image,
        m if m.starts_with("audio/") => PartKind::Audio,
        m if m.starts_with("video/") => PartKind::Video,
        _ => PartKind::Document,
    }
}

fn media(
    kind: Option<PartKind>,
    source: Source,
    size: Option<usize>,
    unit: Option<Unit>,
    mime: Option<String>,
    signed: bool,
) -> PartSpec {
    let mime = mime.filter(|m| !m.is_empty());
    PartSpec {
        measure: size,
        unit,
        source: Some(source),
        signed,
        kind: kind.unwrap_or_else(|| media_kind(mime.as_deref())),
        mime_type: mime,
        ..PartSpec::new(PartKind::Other)
    }
}

fn inline(
    data: Option<&Value>,
    kind: Option<PartKind>,
    mime: Option<&Value>,
    signed: bool,
) -> PartSpec {
    media(
        kind,
        Source::Inline,
        bytes(data),
        Some(Unit::Bytes),
        name(mime),
        signed,
    )
}

fn file(kind: Option<PartKind>, mime: Option<&Value>) -> PartSpec {
    media(kind, Source::File, None, None, name(mime), false)
}

/// `shape_url`: a URL field can hold a data URI instead of a link.
fn url(value: Option<&Value>, kind: Option<PartKind>, mime: Option<&Value>) -> PartSpec {
    match value.and_then(Value::as_str) {
        Some(u) if u.starts_with("data:") => {
            let (header, data) = u.split_once(',').unwrap_or((u, ""));
            let mime = header
                .trim_start_matches("data:")
                .split(';')
                .next()
                .unwrap_or("");
            let data = Value::String(data.to_string());
            inline(Some(&data), kind, Some(&Value::String(mime.into())), false)
        }
        _ => media(kind, Source::Url, None, None, name(mime), false),
    }
}

/// `shape_tool_name`: `tool['name'] || tool['type']`.
fn tool_name(tool: &Value) -> Option<String> {
    let tool = hash(Some(tool));
    name(either(tool.get("name"), tool.get("type")))
}

fn str_of(value: Option<&Value>) -> Option<&str> {
    value.and_then(Value::as_str)
}

// ---- Anthropic ------------------------------------------------------------------------------

/// `Anthropic::RequestShapes`: Claude pairs tool results with calls by id and takes a thinking
/// block back only with its signature.
fn anthropic(ctx: &Ctx) -> Option<RequestShape> {
    let messages = ctx.payload.get("messages")?.as_array()?;
    let turns = messages
        .iter()
        .enumerate()
        .map(|(index, message)| {
            let message = hash(Some(message));
            let parts = anthropic_content(message.get("content"));
            let speaker = if str_of(message.get("role")) == Some("assistant") {
                Speaker::Model
            } else {
                answer(&parts)
            };
            turn(index, message.get("role"), speaker, parts)
        })
        .collect();
    let p = ctx.payload;
    let request = Request {
        rules: &[Rule::SignedThinking],
        instructions: anthropic_content(p.get("system")),
        tool_names: list(p.get("tools")).iter().map(tool_name).collect(),
        thinking_settings: merge(
            settings(p.get("thinking"), &["type", "budget_tokens", "display"]),
            settings(p.get("output_config"), &["effort"]),
        ),
        ..Request::default()
    };
    Some(shape_request(ctx, turns, request))
}

fn anthropic_content(content: Option<&Value>) -> Vec<PartSpec> {
    if let Some(Value::String(_)) = content {
        return vec![shape_text(content, PartKind::Text, false)];
    }
    list(content)
        .iter()
        .flat_map(|b| anthropic_block(hash(Some(b))))
        .collect()
}

fn anthropic_block(block: &Map<String, Value>) -> Vec<PartSpec> {
    let kind = str_of(block.get("type"));
    vec![match kind {
        Some("text") => shape_text(block.get("text"), PartKind::Text, false),
        Some("thinking") => shape_text(
            block.get("thinking"),
            PartKind::Thinking,
            signature(block.get("signature")),
        ),
        Some("redacted_thinking") => {
            shape_text(None, PartKind::Thinking, signature(block.get("data")))
        }
        Some("tool_use") => tool_call(
            block.get("name"),
            block.get("input"),
            block.get("id"),
            false,
        ),
        Some("tool_result") => {
            let read = |b: &Value| anthropic_block(hash(Some(b)));
            return tool_result(
                block.get("content"),
                None,
                block.get("tool_use_id"),
                Some(&read),
            );
        }
        Some(media @ ("image" | "document")) => {
            let kind = if media == "image" {
                PartKind::Image
            } else {
                PartKind::Document
            };
            anthropic_source(kind, hash(block.get("source")))
        }
        _ => other(block.get("type"), false),
    }]
}

fn anthropic_source(kind: PartKind, source: &Map<String, Value>) -> PartSpec {
    match str_of(source.get("type")) {
        Some("base64") => inline(
            source.get("data"),
            Some(kind),
            source.get("media_type"),
            false,
        ),
        Some("text") => media(
            Some(kind),
            Source::Inline,
            length(source.get("data")),
            None,
            name(source.get("media_type")),
            false,
        ),
        Some("url") => media(Some(kind), Source::Url, None, None, None, false),
        Some("file") => file(Some(kind), None),
        _ => media(
            Some(kind),
            Source::Inline,
            length(source.get("content")),
            None,
            None,
            false,
        ),
    }
}

// ---- Chat Completions -----------------------------------------------------------------------

/// `ChatCompletions::RequestShapes::SPEAKERS`.
fn cc_speaker(role: Option<&Value>) -> Speaker {
    match str_of(role) {
        Some("system" | "developer") => Speaker::System,
        Some("assistant") => Speaker::Model,
        Some("tool" | "function") => Speaker::Tool,
        _ => Speaker::User,
    }
}

/// `ChatCompletions::RequestShapes`: tool messages answer calls by id.
fn chat_completions(ctx: &Ctx) -> Option<RequestShape> {
    let messages = ctx.payload.get("messages")?.as_array()?;
    let turns = messages
        .iter()
        .enumerate()
        .map(|(index, message)| cc_message(hash(Some(message)), index))
        .collect();
    let p = ctx.payload;
    let request = Request {
        tool_names: list(p.get("tools")).iter().map(cc_function_name).collect(),
        thinking_settings: merge(
            merge(
                settings(Some(&Value::Object(p.clone())), &["reasoning_effort"]),
                settings(
                    p.get("reasoning"),
                    &["effort", "max_tokens", "enabled", "exclude"],
                ),
            ),
            settings(
                p.get("thinking"),
                &["type", "token_budget", "budget_tokens"],
            ),
        ),
        ..Request::default()
    };
    Some(shape_request(ctx, turns, request))
}

fn cc_message(message: &Map<String, Value>, index: usize) -> TurnSpec {
    let speaker = cc_speaker(message.get("role"));
    let content = message.get("content");
    let mut parts = cc_message_thinking(message);
    if speaker == Speaker::Tool {
        let read = |b: &Value| vec![cc_content_part(hash(Some(b)))];
        parts.extend(tool_result(
            content,
            message.get("name"),
            message.get("tool_call_id"),
            Some(&read),
        ));
    } else {
        parts.extend(cc_content(content));
    }
    parts.extend(
        list(message.get("tool_calls"))
            .iter()
            .map(|c| cc_function_call(hash(Some(c)))),
    );
    turn(index, message.get("role"), speaker, parts)
}

fn cc_message_thinking(message: &Map<String, Value>) -> Vec<PartSpec> {
    if let Some(Value::Array(details)) = message.get("reasoning_details") {
        return details
            .iter()
            .map(|d| cc_reasoning_detail(hash(Some(d))))
            .collect();
    }
    let text = either(message.get("reasoning_content"), message.get("reasoning"));
    let signed = signature(message.get("reasoning_signature"));
    if truthy(text).is_some() || signed {
        vec![shape_text(truthy(text), PartKind::Thinking, signed)]
    } else {
        Vec::new()
    }
}

fn cc_reasoning_detail(detail: &Map<String, Value>) -> PartSpec {
    match str_of(detail.get("type")) {
        Some("reasoning.encrypted") => {
            shape_text(None, PartKind::Thinking, signature(detail.get("data")))
        }
        Some("reasoning.summary") => shape_text(detail.get("summary"), PartKind::Thinking, false),
        _ => shape_text(
            detail.get("text"),
            PartKind::Thinking,
            signature(detail.get("signature")),
        ),
    }
}

fn cc_content(content: Option<&Value>) -> Vec<PartSpec> {
    if let Some(Value::String(_)) = content {
        return vec![shape_text(content, PartKind::Text, false)];
    }
    list(content)
        .iter()
        .map(|p| cc_content_part(hash(Some(p))))
        .collect()
}

fn cc_url_field(value: Option<&Value>) -> Option<&Value> {
    match value {
        Some(Value::Object(o)) => o.get("url"),
        other => other,
    }
}

fn cc_content_part(part: &Map<String, Value>) -> PartSpec {
    match str_of(part.get("type")) {
        Some("text") => shape_text(part.get("text"), PartKind::Text, false),
        Some("image_url") => url(
            cc_url_field(part.get("image_url")),
            Some(PartKind::Image),
            None,
        ),
        Some("video_url") => url(
            cc_url_field(part.get("video_url")),
            Some(PartKind::Video),
            None,
        ),
        Some("input_audio") => inline(
            hash(part.get("input_audio")).get("data"),
            Some(PartKind::Audio),
            None,
            false,
        ),
        Some("file") => {
            let f = hash(part.get("file"));
            if f.contains_key("file_id") {
                file(Some(PartKind::Document), None)
            } else {
                url(f.get("file_data"), Some(PartKind::Document), None)
            }
        }
        Some("document_url") => url(
            cc_url_field(part.get("document_url")),
            Some(PartKind::Document),
            None,
        ),
        // Perplexity sends a document's bytes as bare base64 where its link would go, and base64
        // never holds the colon that starts a scheme.
        Some("file_url") => match cc_url_field(part.get("file_url")) {
            Some(Value::String(u)) if !u.contains(':') => inline(
                Some(&Value::String(u.clone())),
                Some(PartKind::Document),
                None,
                false,
            ),
            u => url(u, Some(PartKind::Document), None),
        },
        Some("thinking") => {
            let thinking = part.get("thinking");
            let signed = signature(part.get("signature"));
            if let Some(Value::Array(_)) = thinking {
                summary(thinking, signed)
            } else {
                shape_text(thinking, PartKind::Thinking, signed)
            }
        }
        _ => other(part.get("type"), false),
    }
}

fn cc_function_call(call: &Map<String, Value>) -> PartSpec {
    let function = hash(call.get("function"));
    let google = hash(hash(call.get("extra_content")).get("google"));
    tool_call(
        function.get("name"),
        function.get("arguments"),
        call.get("id"),
        signature(google.get("thought_signature")),
    )
}

fn cc_function_name(tool: &Value) -> Option<String> {
    let tool = hash(Some(tool));
    name(either(
        hash(tool.get("function")).get("name"),
        tool.get("type"),
    ))
}

// ---- Gemini ---------------------------------------------------------------------------------

const GEMINI_PART_MODIFIERS: &[&str] = &[
    "thought",
    "thoughtSignature",
    "thought_signature",
    "mediaResolution",
    "media_resolution",
    "videoMetadata",
    "video_metadata",
    "partMetadata",
    "part_metadata",
];
const GEMINI_DATA_FIELDS: &[&str] = &[
    "text",
    "inlineData",
    "inline_data",
    "fileData",
    "file_data",
    "functionCall",
    "function_call",
    "functionResponse",
    "function_response",
    "executableCode",
    "executable_code",
    "codeExecutionResult",
    "code_execution_result",
    "toolCall",
    "tool_call",
    "toolResponse",
    "tool_response",
];
const GEMINI_THINKING_SETTINGS: &[&str] = &[
    "includeThoughts",
    "include_thoughts",
    "thinkingBudget",
    "thinking_budget",
    "thinkingLevel",
    "thinking_level",
];

/// `Gemini::RequestShapes`: function responses pair with calls by position and name, every part
/// needs data, and from Gemini 3 the first call of each step in the current turn needs a
/// signature. A countTokens request wraps the generateContent one.
fn gemini(ctx: &Ctx) -> Option<RequestShape> {
    let wrapped = hash(ctx.payload.get("generateContentRequest"));
    let request = if wrapped.is_empty() {
        ctx.payload
    } else {
        wrapped
    };
    let contents = request.get("contents")?.as_array()?;
    let turns = contents
        .iter()
        .enumerate()
        .map(|(index, content)| {
            let content = hash(Some(content));
            let parts = gemini_parts(Some(&Value::Object(content.clone())));
            let speaker = if str_of(content.get("role")) == Some("model") {
                Speaker::Model
            } else {
                answer(&parts)
            };
            turn(index, content.get("role"), speaker, parts)
        })
        .collect();
    let config = hash(either(
        request.get("generationConfig"),
        request.get("generation_config"),
    ));
    let shape = Request {
        pairing: Pairing::Position,
        rules: &[Rule::SignedSteps],
        instructions: gemini_parts(either(
            request.get("systemInstruction"),
            request.get("system_instruction"),
        )),
        tool_names: list(request.get("tools"))
            .iter()
            .flat_map(gemini_tools)
            .collect(),
        thinking_settings: settings(
            either(config.get("thinkingConfig"), config.get("thinking_config")),
            GEMINI_THINKING_SETTINGS,
        ),
    };
    Some(shape_request(ctx, turns, shape))
}

fn gemini_parts(content: Option<&Value>) -> Vec<PartSpec> {
    list(hash(content).get("parts"))
        .iter()
        .flat_map(|p| gemini_part(hash(Some(p))))
        .collect()
}

fn gemini_field<'a>(part: &'a Map<String, Value>, keys: &[&str]) -> Option<&'a Map<String, Value>> {
    keys.iter()
        .find_map(|k| part.get(*k).and_then(Value::as_object))
}

fn gemini_part(part: &Map<String, Value>) -> Vec<PartSpec> {
    let signed = signature(either(
        part.get("thoughtSignature"),
        part.get("thought_signature"),
    ));
    if let Some(call) = gemini_field(part, &["functionCall", "function_call"]) {
        return vec![tool_call(
            call.get("name"),
            call.get("args"),
            call.get("id"),
            signed,
        )];
    }
    if let Some(response) = gemini_field(part, &["functionResponse", "function_response"]) {
        return gemini_tool_result(response);
    }
    let thought = truthy(part.get("thought")).is_some();
    if part.contains_key("text") {
        let kind = if thought {
            PartKind::Thinking
        } else {
            PartKind::Text
        };
        return vec![shape_text(part.get("text"), kind, signed)];
    }
    vec![gemini_data(part, signed, thought)]
}

/// RubyLLM wraps a result's text in content parts, and Gemini 3 sends a result's media in parts
/// beside it.
fn gemini_tool_result(response: &Map<String, Value>) -> Vec<PartSpec> {
    let result = response.get("response");
    let content = match hash(result).get("content") {
        Some(c @ Value::Array(_)) => Some(c),
        _ => result,
    };
    let read = |b: &Value| gemini_part(hash(Some(b)));
    let mut parts = tool_result(content, response.get("name"), None, Some(&read));
    parts.extend(gemini_parts(Some(&Value::Object(response.clone()))));
    parts
}

/// A thought or a signature alone is a part with no data, which Gemini refuses.
fn gemini_data(part: &Map<String, Value>, signed: bool, thought: bool) -> PartSpec {
    if let Some(data) = gemini_field(part, &["inlineData", "inline_data"]) {
        return inline(
            data.get("data"),
            None,
            either(data.get("mimeType"), data.get("mime_type")),
            signed,
        );
    }
    if let Some(f) = gemini_field(part, &["fileData", "file_data"]) {
        return file(None, either(f.get("mimeType"), f.get("mime_type")));
    }
    let mut spec = if thought {
        shape_text(None, PartKind::Thinking, signed)
    } else {
        let key = part
            .keys()
            .find(|k| !GEMINI_PART_MODIFIERS.contains(&k.as_str()))
            .map(|k| Value::String(k.clone()));
        other(key.as_ref(), signed)
    };
    spec.without_data = !part
        .keys()
        .any(|k| GEMINI_DATA_FIELDS.contains(&k.as_str()));
    spec
}

fn gemini_tools(tool: &Value) -> Vec<Option<String>> {
    hash(Some(tool))
        .iter()
        .flat_map(|(key, value)| {
            if key == "functionDeclarations" || key == "function_declarations" {
                list(Some(value))
                    .iter()
                    .map(|d| name(hash(Some(d)).get("name")))
                    .collect()
            } else {
                vec![Some(key.clone()).filter(|k| !k.is_empty())]
            }
        })
        .collect()
}

// ---- Interactions ---------------------------------------------------------------------------

/// `Interactions::RequestShapes`: each input step is a turn; a function result answers its call
/// by id, and the first call of each step in the current turn needs a signature.
fn interactions(ctx: &Ctx) -> Option<RequestShape> {
    let input = ctx.payload.get("input")?.as_array()?;
    let turns = input
        .iter()
        .enumerate()
        .map(|(index, entry)| {
            let entry = hash(Some(entry));
            let speaker = match str_of(entry.get("type")) {
                Some("user_input") => Speaker::User,
                Some("function_result") => Speaker::Tool,
                _ => Speaker::Model,
            };
            turn(index, entry.get("type"), speaker, interaction_entry(entry))
        })
        .collect();
    let p = ctx.payload;
    let request = Request {
        rules: &[Rule::SignedSteps],
        instructions: instructions(p.get("system_instruction")),
        tool_names: list(p.get("tools")).iter().map(tool_name).collect(),
        thinking_settings: settings(
            p.get("generation_config"),
            &["thinking_level", "thinking_summaries"],
        ),
        ..Request::default()
    };
    Some(shape_request(ctx, turns, request))
}

fn interaction_entry(entry: &Map<String, Value>) -> Vec<PartSpec> {
    let signed = signature(entry.get("signature"));
    match str_of(entry.get("type")) {
        Some("user_input" | "model_output") => interaction_content(entry.get("content")),
        Some("function_call") => vec![tool_call(
            entry.get("name"),
            entry.get("arguments"),
            entry.get("id"),
            signed,
        )],
        Some("function_result") => {
            let read = |b: &Value| vec![interaction_part(hash(Some(b)))];
            tool_result(
                entry.get("result"),
                entry.get("name"),
                entry.get("call_id"),
                Some(&read),
            )
        }
        Some("thought") => vec![summary(entry.get("summary"), signed)],
        _ => vec![other(entry.get("type"), signed)],
    }
}

fn interaction_content(content: Option<&Value>) -> Vec<PartSpec> {
    if let Some(Value::String(_)) = content {
        return vec![shape_text(content, PartKind::Text, false)];
    }
    list(content)
        .iter()
        .map(|p| interaction_part(hash(Some(p))))
        .collect()
}

fn interaction_part(part: &Map<String, Value>) -> PartSpec {
    let kind = match str_of(part.get("type")) {
        Some("text") => return shape_text(part.get("text"), PartKind::Text, false),
        Some("image") => PartKind::Image,
        Some("audio") => PartKind::Audio,
        Some("video") => PartKind::Video,
        Some("document") => PartKind::Document,
        _ => return other(part.get("type"), false),
    };
    if part.contains_key("data") {
        inline(part.get("data"), Some(kind), part.get("mime_type"), false)
    } else {
        url(part.get("uri"), Some(kind), part.get("mime_type"))
    }
}

// ---- Responses ------------------------------------------------------------------------------

/// `Responses::RequestShapes`: each input item is a turn, and a call's output answers the call
/// by id.
fn responses(ctx: &Ctx) -> Option<RequestShape> {
    let input = ctx.payload.get("input")?.as_array()?;
    let turns = input
        .iter()
        .enumerate()
        .map(|(index, item)| {
            let item = hash(Some(item));
            turn(
                index,
                either(item.get("role"), item.get("type")),
                responses_speaker(item),
                responses_item(item),
            )
        })
        .collect();
    let p = ctx.payload;
    let request = Request {
        instructions: instructions(p.get("instructions")),
        tool_names: list(p.get("tools")).iter().map(tool_name).collect(),
        thinking_settings: settings(p.get("reasoning"), &["effort", "summary"]),
        ..Request::default()
    };
    Some(shape_request(ctx, turns, request))
}

fn responses_speaker(item: &Map<String, Value>) -> Speaker {
    match item.get("type") {
        None | Some(Value::Null) => cc_speaker(item.get("role")),
        Some(Value::String(t)) if t == "message" => cc_speaker(item.get("role")),
        Some(Value::String(t)) if t.ends_with("_output") || t.ends_with("_response") => {
            Speaker::Tool
        }
        _ => Speaker::Model,
    }
}

fn responses_item(item: &Map<String, Value>) -> Vec<PartSpec> {
    match item.get("type") {
        None | Some(Value::Null) => responses_content(item.get("content")),
        Some(Value::String(t)) => match t.as_str() {
            "message" => responses_content(item.get("content")),
            "function_call" => vec![tool_call(
                item.get("name"),
                item.get("arguments"),
                item.get("call_id"),
                false,
            )],
            "function_call_output" => {
                let read = |b: &Value| vec![responses_part(hash(Some(b)))];
                tool_result(item.get("output"), None, item.get("call_id"), Some(&read))
            }
            "reasoning" => vec![summary(
                item.get("summary"),
                signature(item.get("encrypted_content")),
            )],
            _ => vec![other(item.get("type"), false)],
        },
        _ => vec![other(item.get("type"), false)],
    }
}

fn responses_content(content: Option<&Value>) -> Vec<PartSpec> {
    if let Some(Value::String(_)) = content {
        return vec![shape_text(content, PartKind::Text, false)];
    }
    list(content)
        .iter()
        .map(|p| responses_part(hash(Some(p))))
        .collect()
}

fn responses_part(part: &Map<String, Value>) -> PartSpec {
    match str_of(part.get("type")) {
        Some("input_text" | "output_text") => shape_text(part.get("text"), PartKind::Text, false),
        Some("input_image") if part.contains_key("file_id") => file(Some(PartKind::Image), None),
        Some("input_image") => url(part.get("image_url"), Some(PartKind::Image), None),
        Some("input_file") if part.contains_key("file_id") => file(Some(PartKind::Document), None),
        Some("input_file") if part.contains_key("file_url") => {
            url(part.get("file_url"), Some(PartKind::Document), None)
        }
        Some("input_file") => url(part.get("file_data"), Some(PartKind::Document), None),
        Some("input_audio") => inline(
            hash(part.get("input_audio")).get("data"),
            Some(PartKind::Audio),
            None,
            false,
        ),
        _ => other(part.get("type"), false),
    }
}

// ---- Mistral Conversations ------------------------------------------------------------------

/// `Mistral::Conversations::RequestShapes`: each input entry is a turn, messages carry Chat
/// Completions content, and a function result answers its call by id.
fn conversations(ctx: &Ctx) -> Option<RequestShape> {
    let inputs = ctx.payload.get("inputs")?.as_array()?;
    let turns = inputs
        .iter()
        .enumerate()
        .map(|(index, entry)| {
            let entry = hash(Some(entry));
            let role = either(entry.get("role"), entry.get("type"));
            match str_of(entry.get("type")) {
                Some("message.input") => turn(
                    index,
                    role,
                    cc_speaker(entry.get("role")),
                    cc_content(entry.get("content")),
                ),
                Some("message.output") => turn(
                    index,
                    role,
                    Speaker::Model,
                    cc_content(entry.get("content")),
                ),
                Some("function.result") => turn(
                    index,
                    role,
                    Speaker::Tool,
                    tool_result(entry.get("result"), None, entry.get("tool_call_id"), None),
                ),
                Some("function.call") => turn(
                    index,
                    role,
                    Speaker::Model,
                    vec![tool_call(
                        entry.get("name"),
                        entry.get("arguments"),
                        entry.get("tool_call_id"),
                        false,
                    )],
                ),
                _ => turn(
                    index,
                    role,
                    Speaker::Model,
                    vec![other(entry.get("type"), false)],
                ),
            }
        })
        .collect();
    let p = ctx.payload;
    let request = Request {
        instructions: instructions(p.get("instructions")),
        tool_names: list(p.get("tools")).iter().map(cc_function_name).collect(),
        thinking_settings: settings(p.get("completion_args"), &["reasoning_effort"]),
        ..Request::default()
    };
    Some(shape_request(ctx, turns, request))
}
