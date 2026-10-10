//! Port of `lib/ruby_llm/request_shape.rb` and `lib/ruby_llm/request_shape/{part,turn,tool_round,problem}.rb`.
//!
//! A [`RequestShape`] describes a conversation request as the provider received it, without its
//! contents: each turn's role and parts with their sizes, the tools on offer, the thinking
//! settings, and the problems RustLLM finds in it, such as a part with no data or a tool call
//! with no result. Read it from [`crate::Error::request_shape`] when a provider rejects a request
//! without saying what is wrong with it.
//!
//! ```ignore
//! match chat.ask("What's due today?").await {
//!     Err(e) => tracing::error!("{e}\n{}", e.request_shape().map(ToString::to_string).unwrap_or_default()),
//!     Ok(_) => {}
//! }
//! ```
//!
//! A shape holds no text, tool arguments, tool results, signatures, URLs, file data, or setting
//! values beyond thinking settings, so it can be logged and sent to an error tracker. A long
//! conversation keeps its first 10 and last 50 turns.

#[doc(hidden)]
pub mod analysis;
pub(crate) mod protocols;

use std::fmt;
use std::sync::{Arc, OnceLock};

use serde_json::{Map, Value, json};

use crate::providers::ProtocolName;

/// `Part#kind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartKind {
    Text,
    Thinking,
    Image,
    Audio,
    Video,
    Document,
    ToolCall,
    ToolResult,
    /// A piece RustLLM does not classify, such as a provider tool step.
    Other,
}

impl PartKind {
    pub fn as_str(self) -> &'static str {
        match self {
            PartKind::Text => "text",
            PartKind::Thinking => "thinking",
            PartKind::Image => "image",
            PartKind::Audio => "audio",
            PartKind::Video => "video",
            PartKind::Document => "document",
            PartKind::ToolCall => "tool_call",
            PartKind::ToolResult => "tool_result",
            PartKind::Other => "other",
        }
    }
}

/// `Part#unit`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unit {
    Chars,
    Bytes,
}

impl Unit {
    pub fn as_str(self) -> &'static str {
        match self {
            Unit::Chars => "chars",
            Unit::Bytes => "bytes",
        }
    }
}

/// `Part#source`: where media comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// Bytes in the request.
    Inline,
    /// A link the provider fetches.
    Url,
    /// A file stored with the provider.
    File,
}

impl Source {
    pub fn as_str(self) -> &'static str {
        match self {
            Source::Inline => "inline",
            Source::Url => "url",
            Source::File => "file",
        }
    }
}

/// `Problem#kind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProblemKind {
    PartWithoutData,
    EmptyTurn,
    UnpairedRound,
    UnmatchedResult,
    UnsignedCall,
    UnsignedThinking,
    RoleOrder,
}

impl ProblemKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ProblemKind::PartWithoutData => "part_without_data",
            ProblemKind::EmptyTurn => "empty_turn",
            ProblemKind::UnpairedRound => "unpaired_round",
            ProblemKind::UnmatchedResult => "unmatched_result",
            ProblemKind::UnsignedCall => "unsigned_call",
            ProblemKind::UnsignedThinking => "unsigned_thinking",
            ProblemKind::RoleOrder => "role_order",
        }
    }
}

/// `Support::Inspectable#inspect`: `#<Class name: value, ...>`, omitting empty values.
enum Shown<'a> {
    Str(&'a str),
    Sym(&'a str),
    Int(usize),
    Bool(bool),
}

fn inspect(class: &str, attributes: &[(&str, Option<Shown>)]) -> String {
    let shown: Vec<String> = attributes
        .iter()
        .filter_map(|(name, value)| {
            let value = match value.as_ref()? {
                Shown::Str("") => return None,
                Shown::Str(s) if s.chars().count() > 64 => {
                    format!(
                        "{:?}",
                        format!("{}...", s.chars().take(64).collect::<String>())
                    )
                }
                Shown::Str(s) => format!("{s:?}"),
                Shown::Sym(s) => format!(":{s}"),
                Shown::Int(n) => n.to_string(),
                Shown::Bool(b) => b.to_string(),
            };
            Some(format!("{name}: {value}"))
        })
        .collect();
    if shown.is_empty() {
        format!("#<RustLLM::{class}>")
    } else {
        format!("#<RustLLM::{class} {}>", shown.join(", "))
    }
}

/// `RequestShape::Part`: one piece of a turn by its kind and size, never by its content.
#[derive(Clone, PartialEq)]
pub struct Part {
    pub kind: PartKind,
    /// The characters of text, thinking, tool call arguments, and a tool result's text (or of
    /// its JSON when it is structured data), and the bytes of inline media. `None` when there is
    /// nothing to measure, such as media sent by URL or thinking the provider encrypted.
    pub size: Option<usize>,
    /// The unit of `size`, or `None` without a size.
    pub unit: Option<Unit>,
    /// The tool's name for a call, and for a result the name of the call it answers. For an
    /// `Other` piece, the provider's name for it.
    pub name: Option<String>,
    pub source: Option<Source>,
    pub mime_type: Option<String>,
    pub signed: bool,
}

impl Part {
    /// `Part.new(kind:)`.
    pub fn new(kind: PartKind) -> Part {
        Part {
            kind,
            size: None,
            unit: None,
            name: None,
            source: None,
            mime_type: None,
            signed: false,
        }
    }

    /// `size:` with `unit:` (`unit = size && (unit || :chars)`).
    pub fn sized(mut self, size: usize, unit: Unit) -> Part {
        self.size = Some(size);
        self.unit = Some(unit);
        self
    }

    /// `size:` in characters.
    pub fn chars(self, size: usize) -> Part {
        self.sized(size, Unit::Chars)
    }

    /// `size:, unit: :bytes`.
    pub fn bytes(self, size: usize) -> Part {
        self.sized(size, Unit::Bytes)
    }

    pub fn name(mut self, name: impl Into<String>) -> Part {
        self.name = Some(name.into());
        self
    }

    pub fn source(mut self, source: Source) -> Part {
        self.source = Some(source);
        self
    }

    pub fn mime_type(mut self, mime_type: impl Into<String>) -> Part {
        self.mime_type = Some(mime_type.into());
        self
    }

    /// `signed: true`.
    pub fn signed(mut self) -> Part {
        self.signed = true;
        self
    }

    /// `signed?`: whether the piece carries a thinking signature.
    pub fn is_signed(&self) -> bool {
        self.signed
    }

    /// `to_h`: the size under its unit, omitting attributes the part does not have.
    pub fn to_h(&self) -> Value {
        let mut h = Map::new();
        h.insert("kind".into(), self.kind.as_str().into());
        if let Some(name) = &self.name {
            h.insert("name".into(), name.clone().into());
        }
        if let (Some(size), Some(unit)) = (self.size, self.unit) {
            h.insert(unit.as_str().into(), size.into());
        }
        if let Some(source) = self.source {
            h.insert("source".into(), source.as_str().into());
        }
        if let Some(mime) = &self.mime_type {
            h.insert("mime_type".into(), mime.clone().into());
        }
        if self.signed {
            h.insert("signed".into(), true.into());
        }
        Value::Object(h)
    }

    fn description(&self) -> String {
        match self.kind {
            PartKind::ToolCall => ["call".to_string()]
                .into_iter()
                .chain(self.name.clone())
                .chain(self.size.map(|_| format!("(args {})", self.measured())))
                .collect::<Vec<_>>()
                .join(" "),
            PartKind::ToolResult => ["result".to_string()]
                .into_iter()
                .chain(self.name.clone())
                .chain(self.size.map(|_| format!("({})", self.measured())))
                .collect::<Vec<_>>()
                .join(" "),
            PartKind::Other => self
                .name
                .clone()
                .unwrap_or_else(|| "part of no known kind".into()),
            kind => {
                let label = self.mime_type.as_deref().unwrap_or(kind.as_str());
                let size = if self.size.is_some() {
                    self.measured()
                } else {
                    self.missing()
                };
                format!("{label} ({size})")
            }
        }
    }

    fn measured(&self) -> String {
        let size = self.size.unwrap_or(0);
        let noun = match self.unit {
            Some(Unit::Bytes) => "byte",
            _ => "char",
        };
        format!("{size} {noun}{}", if size == 1 { "" } else { "s" })
    }

    fn missing(&self) -> String {
        match self.source {
            Some(source) if source != Source::Inline => source.as_str().into(),
            _ if matches!(self.kind, PartKind::Text | PartKind::Thinking) => "no text".into(),
            _ => "no data".into(),
        }
    }
}

/// `Part#to_s`: `text (13 chars)`, `image/png (34512 bytes)`, `thinking (no text), signed`.
impl fmt::Display for Part {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.signed {
            write!(f, "{}, signed", self.description())
        } else {
            f.write_str(&self.description())
        }
    }
}

/// `Part#inspect`: its `to_h`.
impl fmt::Debug for Part {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = self.name.as_deref().map(Shown::Str);
        let size = self.size.map(Shown::Int);
        let unit = self.unit.map_or("chars", Unit::as_str);
        f.write_str(&inspect(
            "RequestShape::Part",
            &[
                ("kind", Some(Shown::Sym(self.kind.as_str()))),
                ("name", name),
                (unit, size),
                ("source", self.source.map(|s| Shown::Sym(s.as_str()))),
                ("mime_type", self.mime_type.as_deref().map(Shown::Str)),
                ("signed", self.signed.then_some(Shown::Bool(true))),
            ],
        ))
    }
}

/// `RequestShape::Turn`: one entry of a request's conversation.
#[derive(Clone, PartialEq)]
pub struct Turn {
    /// The turn's 0-indexed position in the request.
    pub index: usize,
    /// The role as the payload names it, or for an entry that names no role, its type.
    pub role: Option<String>,
    pub parts: Vec<Part>,
}

impl Turn {
    pub fn new(index: usize, role: Option<&str>, parts: Vec<Part>) -> Turn {
        Turn {
            index,
            role: role.map(str::to_string),
            parts,
        }
    }

    pub fn to_h(&self) -> Value {
        json!({
            "index": self.index,
            "role": self.role,
            "parts": self.parts.iter().map(Part::to_h).collect::<Vec<_>>(),
        })
    }

    fn parts_line(&self) -> String {
        self.parts
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// `Turn#to_s`: `#4 user: text (31 chars)`.
impl fmt::Display for Turn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let head = match &self.role {
            Some(role) => format!("#{} {role}", self.index),
            None => format!("#{}", self.index),
        };
        let parts = if self.parts.is_empty() {
            "no parts".to_string()
        } else {
            self.parts_line()
        };
        write!(f, "{head}: {parts}")
    }
}

impl fmt::Debug for Turn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let parts = self.parts_line();
        f.write_str(&inspect(
            "RequestShape::Turn",
            &[
                ("index", Some(Shown::Int(self.index))),
                ("role", self.role.as_deref().map(Shown::Str)),
                ("parts", Some(Shown::Str(&parts))),
            ],
        ))
    }
}

fn counted(count: usize, noun: &str) -> String {
    format!("{count} {noun}{}", if count == 1 { "" } else { "s" })
}

/// `RequestShape::ToolRound`: a model turn that called tools and the results sent back for it.
#[derive(Clone, PartialEq)]
pub struct ToolRound {
    pub turn: usize,
    pub calls: usize,
    pub results: usize,
    pub paired: bool,
}

impl ToolRound {
    pub fn new(turn: usize, calls: usize, results: usize, paired: bool) -> ToolRound {
        ToolRound {
            turn,
            calls,
            results,
            paired,
        }
    }

    /// `paired?`: every call has its result, matched the way the provider matches them.
    pub fn is_paired(&self) -> bool {
        self.paired
    }

    pub fn to_h(&self) -> Value {
        json!({ "turn": self.turn, "calls": self.calls, "results": self.results, "paired": self.paired })
    }
}

impl fmt::Display for ToolRound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "tool round at #{}: {}, {}, {}",
            self.turn,
            counted(self.calls, "call"),
            counted(self.results, "result"),
            if self.paired { "paired" } else { "not paired" }
        )
    }
}

impl fmt::Debug for ToolRound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&inspect(
            "RequestShape::ToolRound",
            &[
                ("turn", Some(Shown::Int(self.turn))),
                ("calls", Some(Shown::Int(self.calls))),
                ("results", Some(Shown::Int(self.results))),
                ("paired", Some(Shown::Bool(self.paired))),
            ],
        ))
    }
}

/// `RequestShape::Problem`: something in a request providers are known to reject.
#[derive(Clone, PartialEq)]
pub struct Problem {
    pub kind: ProblemKind,
    pub turn: Option<usize>,
    /// The position of the part in the turn's parts, or `None` for the whole turn.
    pub part: Option<usize>,
    /// The id of the call an unmatched tool result names, the only id a shape shows.
    pub call_id: Option<String>,
    pub message: String,
}

impl Problem {
    pub fn new(kind: ProblemKind, message: impl Into<String>) -> Problem {
        Problem {
            kind,
            turn: None,
            part: None,
            call_id: None,
            message: message.into(),
        }
    }

    pub fn at(mut self, turn: usize, part: Option<usize>) -> Problem {
        self.turn = Some(turn);
        self.part = part;
        self
    }

    pub fn call_id(mut self, call_id: impl Into<String>) -> Problem {
        self.call_id = Some(call_id.into());
        self
    }

    /// `to_h`, omitting attributes the problem does not have.
    pub fn to_h(&self) -> Value {
        let mut h = Map::new();
        h.insert("kind".into(), self.kind.as_str().into());
        if let Some(turn) = self.turn {
            h.insert("turn".into(), turn.into());
        }
        if let Some(part) = self.part {
            h.insert("part".into(), part.into());
        }
        if let Some(id) = &self.call_id {
            h.insert("call_id".into(), id.clone().into());
        }
        h.insert("message".into(), self.message.clone().into());
        Value::Object(h)
    }
}

impl fmt::Display for Problem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let line = match self.turn {
            Some(turn) => {
                let location = match self.part {
                    Some(part) => format!("#{turn}, part {part}"),
                    None => format!("#{turn}"),
                };
                format!("problem at {location}: {}", self.message)
            }
            None => format!("problem: {}", self.message),
        };
        match &self.call_id {
            Some(id) => write!(f, "{line} (call id {id})"),
            None => f.write_str(&line),
        }
    }
}

impl fmt::Debug for Problem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&inspect(
            "RequestShape::Problem",
            &[
                ("kind", Some(Shown::Sym(self.kind.as_str()))),
                ("turn", self.turn.map(Shown::Int)),
                ("part", self.part.map(Shown::Int)),
                ("call_id", self.call_id.as_deref().map(Shown::Str)),
                ("message", Some(Shown::Str(&self.message))),
            ],
        ))
    }
}

/// `RubyLLM::RequestShape`.
#[derive(Clone, PartialEq)]
pub struct RequestShape {
    /// The provider's slug, such as `"gemini"`.
    pub provider: Option<String>,
    /// The id of the model the request went to.
    pub model: Option<String>,
    /// Which model call of the current turn the request makes, counting from 1, so `3` follows
    /// two tool rounds. `None` when the request holds no user turn to count from.
    pub step: Option<usize>,
    /// The names of the payload's top-level fields, without their values.
    pub payload_keys: Vec<String>,
    /// The thinking settings the request sends, by the provider's setting names.
    pub thinking_settings: Map<String, Value>,
    /// The instructions the request sends apart from its turns.
    pub instructions: Vec<Part>,
    /// The turns kept: the first 10 and last 50 of a long conversation.
    pub turns: Vec<Turn>,
    /// The number of turns the request sends, including omitted ones.
    pub turn_count: usize,
    pub omitted_turns: usize,
    /// The names of the tools the request offers, in order.
    pub tool_names: Vec<String>,
    pub tool_rounds: Vec<ToolRound>,
    pub problems: Vec<Problem>,
}

impl RequestShape {
    /// `RequestShape.new(turns:)`: every other attribute empty, `turn_count` the turns given.
    pub fn new(turns: Vec<Turn>) -> RequestShape {
        RequestShape {
            provider: None,
            model: None,
            step: None,
            payload_keys: Vec::new(),
            thinking_settings: Map::new(),
            instructions: Vec::new(),
            turn_count: turns.len(),
            turns,
            omitted_turns: 0,
            tool_names: Vec::new(),
            tool_rounds: Vec::new(),
            problems: Vec::new(),
        }
    }

    /// `to_h`: the same keys every time, for structured logs and error trackers.
    pub fn to_h(&self) -> Value {
        json!({
            "provider": self.provider,
            "model": self.model,
            "step": self.step,
            "payload_keys": self.payload_keys,
            "thinking_settings": self.thinking_settings,
            "instructions": self.instructions.iter().map(Part::to_h).collect::<Vec<_>>(),
            "turn_count": self.turn_count,
            "omitted_turns": self.omitted_turns,
            "turns": self.turns.iter().map(Turn::to_h).collect::<Vec<_>>(),
            "tool_names": self.tool_names,
            "tool_rounds": self.tool_rounds.iter().map(ToolRound::to_h).collect::<Vec<_>>(),
            "problems": self.problems.iter().map(Problem::to_h).collect::<Vec<_>>(),
        })
    }

    fn summary_line(&self) -> Option<String> {
        let step = self.step.map(|step| {
            let rounds = step.saturating_sub(1);
            let description = format!("model call {step} of the turn");
            if rounds == 0 {
                description
            } else {
                format!("{description}, after {}", counted(rounds, "tool round"))
            }
        });
        let parts: Vec<String> = [self.provider.clone(), self.model.clone(), step]
            .into_iter()
            .flatten()
            .collect();
        (!parts.is_empty()).then(|| parts.join(", "))
    }
}

fn list_line(label: &str, items: &[String]) -> Option<String> {
    (!items.is_empty()).then(|| format!("{label}: {}", items.join(", ")))
}

/// `RequestShape#to_s`: the request as text, one turn per line.
impl fmt::Display for RequestShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut lines: Vec<String> = Vec::new();
        lines.extend(self.summary_line());
        lines.extend(list_line("payload keys", &self.payload_keys));
        if !self.thinking_settings.is_empty() {
            let settings: Vec<String> = self
                .thinking_settings
                .iter()
                .map(|(name, value)| match value {
                    Value::String(s) => format!("{name} {s}"),
                    other => format!("{name} {other}"),
                })
                .collect();
            lines.push(format!("thinking: {}", settings.join(", ")));
        }
        let instructions: Vec<String> = self.instructions.iter().map(ToString::to_string).collect();
        lines.extend(list_line("instructions", &instructions));
        for (i, turn) in self.turns.iter().enumerate() {
            lines.push(turn.to_string());
            if let Some(following) = self.turns.get(i + 1) {
                let gap = following.index.saturating_sub(turn.index + 1);
                if gap > 0 {
                    lines.push(format!("({})", counted(gap, "turn") + " omitted"));
                }
            }
        }
        lines.extend(list_line("tools", &self.tool_names));
        lines.extend(self.tool_rounds.iter().map(ToString::to_string));
        if self.problems.is_empty() {
            lines.push("no problems found".into());
        } else {
            lines.extend(self.problems.iter().map(ToString::to_string));
        }
        f.write_str(&lines.join("\n"))
    }
}

/// `RequestShape#inspect`: `#<RustLLM::RequestShape provider: "gemini", model: "...", turns: 71, problems: 1>`.
impl fmt::Debug for RequestShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&inspect(
            "RequestShape",
            &[
                ("provider", self.provider.as_deref().map(Shown::Str)),
                ("model", self.model.as_deref().map(Shown::Str)),
                ("turns", Some(Shown::Int(self.turn_count))),
                ("problems", Some(Shown::Int(self.problems.len()))),
            ],
        ))
    }
}

/// `Error#request_protocol` and `#request_payload`: the request an error answered, read back
/// into a [`RequestShape`] once, when asked for (`Protocol#claim_error`).
#[derive(Clone)]
pub struct ClaimedRequest(Arc<Claimed>);

struct Claimed {
    protocol: ProtocolName,
    provider: String,
    model: String,
    payload: Value,
    shape: OnceLock<Option<RequestShape>>,
}

impl ClaimedRequest {
    pub(crate) fn new(
        protocol: ProtocolName,
        provider: &str,
        model: &str,
        payload: &Value,
    ) -> ClaimedRequest {
        ClaimedRequest(Arc::new(Claimed {
            protocol,
            provider: provider.to_string(),
            model: model.to_string(),
            payload: payload.clone(),
            shape: OnceLock::new(),
        }))
    }

    /// `Error#request_shape`: reads the payload the first time only.
    pub(crate) fn shape(&self) -> Option<&RequestShape> {
        let c = &self.0;
        c.shape
            .get_or_init(|| {
                protocols::describe(c.protocol, Some(&c.provider), Some(&c.model), &c.payload)
            })
            .as_ref()
    }

    /// Whether the payload has been read (for `reads the payload once`).
    #[doc(hidden)]
    pub fn is_read(&self) -> bool {
        self.0.shape.get().is_some()
    }
}

/// Never the payload: an error's `Debug` must not show the request's contents.
impl fmt::Debug for ClaimedRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ClaimedRequest")
            .field("protocol", &self.0.protocol)
            .finish()
    }
}

/// `Protocol#request_shape(payload)`: describes `payload`, a conversation request `protocol`
/// rendered for `provider`'s `model`. `None` for a payload that is not an object or holds no
/// conversation.
pub fn describe(
    protocol: ProtocolName,
    provider: Option<&str>,
    model: Option<&str>,
    payload: &Value,
) -> Option<RequestShape> {
    protocols::describe(protocol, provider, model, payload)
}
