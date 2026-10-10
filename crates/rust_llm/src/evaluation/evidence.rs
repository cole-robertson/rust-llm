//! Port of `lib/ruby_llm/evaluation/evidence.rb`: what `perform` returned, as portable data for
//! evaluators, with attachments kept apart from the text.
//!
//! Ruby inspects the returned object's class. Here `perform` returns an [`Outcome`], one variant
//! per kind of value Ruby recognizes; anything else goes through an adapter
//! ([`super::EvaluationDef::adapt`]) to JSON, as Ruby's `adapt` does.

use std::any::Any;
use std::sync::Arc;

use base64::Engine;
use serde_json::{Map, Value, json};

use crate::attachment::Attachment;
use crate::chat::Chat;
use crate::embedding::{Embedding, SparseVectors, Vectors};
use crate::error::{Error, Result};
use crate::image::Image;
use crate::judge::{Answer, Judgment};
use crate::message::{Message, Role, ServerToolCall, ToolCall};
use crate::moderation::{Moderation, ModerationResult};
use crate::ocr::Ocr;
use crate::rerank::Rerank;
use crate::speech::Speech;
use crate::tokenization::Tokenization;
use crate::transcription::Transcription;
use crate::video::Video;

/// What `perform` returned. Ruby returns any object; the port names the kinds it can evaluate.
/// [`Outcome::Agent`] is a chat an agent built (`Agent.new.chat`), kept apart only so evidence
/// can tell they were an agent's, which Ruby normalizes to the same data as the chat.
pub enum Outcome {
    /// JSON data: strings, numbers, booleans, `null`, arrays, objects.
    Value(Value),
    Chat(Box<Chat>),
    Agent(Box<Chat>),
    Message(Box<Message>),
    ToolCall(ToolCall),
    Answer(Answer),
    Judgment(Box<Judgment>),
    Embedding(Box<Embedding>),
    Transcription(Box<Transcription>),
    Ocr(Box<Ocr>),
    Rerank(Box<Rerank>),
    Moderation(Box<Moderation>),
    ModerationResult(ModerationResult),
    Tokenization(Tokenization),
    ServerToolCall(ServerToolCall),
    Image(Box<Image>),
    Video(Box<Video>),
    Speech(Box<Speech>),
    Attachment(Attachment),
    /// An array of outcomes (`[agent, { second: chat }]`).
    List(Vec<Outcome>),
    /// An object whose values are outcomes.
    Map(Vec<(String, Outcome)>),
    /// Any other type; it needs an adapter.
    Other(Box<dyn Any + Send + Sync>),
}

impl std::fmt::Debug for Outcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Outcome::Value(v) => write!(f, "Value({v})"),
            Outcome::Chat(_) => f.write_str("Chat"),
            Outcome::Agent(_) => f.write_str("Agent"),
            Outcome::Message(m) => write!(f, "Message({:?})", m.content),
            Outcome::List(items) => f.debug_list().entries(items).finish(),
            Outcome::Map(items) => f
                .debug_map()
                .entries(items.iter().map(|(k, v)| (k, v)))
                .finish(),
            Outcome::Other(_) => f.write_str("Other"),
            _ => f.write_str("Outcome"),
        }
    }
}

impl Outcome {
    /// Any other type, converted by an adapter declared for it.
    pub fn other<T: Any + Send + Sync>(value: T) -> Outcome {
        Outcome::Other(Box::new(value))
    }

    /// The returned chat (or the agent's chat), when `perform` returned one.
    pub fn chat(&self) -> Option<&Chat> {
        match self {
            Outcome::Chat(c) | Outcome::Agent(c) => Some(c),
            _ => None,
        }
    }

    /// The returned JSON value, when `perform` returned plain data.
    pub fn value(&self) -> Option<&Value> {
        match self {
            Outcome::Value(v) => Some(v),
            _ => None,
        }
    }

    /// `result.is_a?(RubyLLM::Agent)`.
    pub fn is_agent(&self) -> bool {
        matches!(self, Outcome::Agent(_))
    }
}

impl From<Value> for Outcome {
    fn from(v: Value) -> Outcome {
        Outcome::Value(v)
    }
}
impl From<&str> for Outcome {
    fn from(v: &str) -> Outcome {
        Outcome::Value(v.into())
    }
}
impl From<String> for Outcome {
    fn from(v: String) -> Outcome {
        Outcome::Value(v.into())
    }
}
impl From<Chat> for Outcome {
    fn from(v: Chat) -> Outcome {
        Outcome::Chat(Box::new(v))
    }
}
impl From<Message> for Outcome {
    fn from(v: Message) -> Outcome {
        Outcome::Message(Box::new(v))
    }
}
impl From<ToolCall> for Outcome {
    fn from(v: ToolCall) -> Outcome {
        Outcome::ToolCall(v)
    }
}
impl From<Answer> for Outcome {
    fn from(v: Answer) -> Outcome {
        Outcome::Answer(v)
    }
}
impl From<Judgment> for Outcome {
    fn from(v: Judgment) -> Outcome {
        Outcome::Judgment(Box::new(v))
    }
}
impl From<Embedding> for Outcome {
    fn from(v: Embedding) -> Outcome {
        Outcome::Embedding(Box::new(v))
    }
}
impl From<Transcription> for Outcome {
    fn from(v: Transcription) -> Outcome {
        Outcome::Transcription(Box::new(v))
    }
}
impl From<Ocr> for Outcome {
    fn from(v: Ocr) -> Outcome {
        Outcome::Ocr(Box::new(v))
    }
}
impl From<Rerank> for Outcome {
    fn from(v: Rerank) -> Outcome {
        Outcome::Rerank(Box::new(v))
    }
}
impl From<Moderation> for Outcome {
    fn from(v: Moderation) -> Outcome {
        Outcome::Moderation(Box::new(v))
    }
}
impl From<ModerationResult> for Outcome {
    fn from(v: ModerationResult) -> Outcome {
        Outcome::ModerationResult(v)
    }
}
impl From<Tokenization> for Outcome {
    fn from(v: Tokenization) -> Outcome {
        Outcome::Tokenization(v)
    }
}
impl From<Image> for Outcome {
    fn from(v: Image) -> Outcome {
        Outcome::Image(Box::new(v))
    }
}
impl From<Video> for Outcome {
    fn from(v: Video) -> Outcome {
        Outcome::Video(Box::new(v))
    }
}
impl From<Speech> for Outcome {
    fn from(v: Speech) -> Outcome {
        Outcome::Speech(Box::new(v))
    }
}
impl From<Attachment> for Outcome {
    fn from(v: Attachment) -> Outcome {
        Outcome::Attachment(v)
    }
}
impl From<Vec<Outcome>> for Outcome {
    fn from(v: Vec<Outcome>) -> Outcome {
        Outcome::List(v)
    }
}

/// An adapter: matches values of one type and converts them to evidence (`adapt Type { ... }`).
pub type Adapter = Arc<dyn Fn(&(dyn Any + Send + Sync)) -> Option<Value> + Send + Sync>;

/// An attachment the evaluator receives beside the text: a file, or a media `data:` URL / link.
#[derive(Debug, Clone)]
pub enum EvidenceAttachment {
    File(Attachment),
    Source(String),
}

impl PartialEq<Attachment> for EvidenceAttachment {
    fn eq(&self, other: &Attachment) -> bool {
        matches!(self, EvidenceAttachment::File(a) if a.to_h() == other.to_h())
    }
}

impl PartialEq<&str> for EvidenceAttachment {
    fn eq(&self, other: &&str) -> bool {
        matches!(self, EvidenceAttachment::Source(s) if s == other)
    }
}

impl EvidenceAttachment {
    pub(crate) fn to_attachment(&self) -> Attachment {
        match self {
            EvidenceAttachment::File(a) => a.clone(),
            EvidenceAttachment::Source(s) => Attachment::new(s),
        }
    }
}

/// `RubyLLM::Evaluation::Evidence`.
#[derive(Debug, Clone)]
pub struct Evidence {
    pub output: Value,
    pub messages: Vec<Message>,
    pub tool_calls: Vec<ToolCall>,
    pub attachments: Vec<EvidenceAttachment>,
    pub data: Value,
}

impl Evidence {
    /// `Evidence.new(result)`.
    pub fn new(result: &Outcome) -> Result<Evidence> {
        Evidence::with_adapters(result, &[])
    }

    /// `Evidence.new(result, adapters:)`. An adapter returns `None` for values it does not handle.
    pub fn with_adapters(result: &Outcome, adapters: &[Adapter]) -> Result<Evidence> {
        let messages: Vec<Message> = match result {
            Outcome::Chat(c) | Outcome::Agent(c) => c.messages().to_vec(),
            Outcome::Message(m) => vec![(**m).clone()],
            _ => Vec::new(),
        };
        let mut tool_calls: Vec<ToolCall> = messages
            .iter()
            .flat_map(|m| m.tool_calls.iter().flat_map(|c| c.values().cloned()))
            .collect();
        if let Outcome::ToolCall(call) = result {
            tool_calls.push(call.clone());
        }
        let output = match result {
            Outcome::Chat(_) | Outcome::Agent(_) => last_answer(&messages),
            Outcome::Message(m) => m.content.clone().map_or(Value::Null, Value::String),
            other => {
                let mut serializer = Serializer {
                    adapters,
                    attachments: Vec::new(),
                };
                serializer.serialize(other)?
            }
        };
        let mut serializer = Serializer {
            adapters,
            attachments: Vec::new(),
        };
        let data = serializer.serialize(result)?;
        Ok(Evidence {
            output,
            messages,
            tool_calls,
            attachments: serializer.attachments,
            data,
        })
    }
}

fn last_answer(messages: &[Message]) -> Value {
    messages
        .iter()
        .rev()
        .find(|m| m.role == Role::Assistant)
        .and_then(|m| m.content.clone())
        .map_or(Value::Null, Value::String)
}

struct Serializer<'a> {
    adapters: &'a [Adapter],
    attachments: Vec<EvidenceAttachment>,
}

/// `Judge::Data.copy` for numbers: non-finite floats are rejected. serde_json cannot hold them,
/// so an [`Outcome::Value`] always passes; adapters and typed results are checked as they convert.
fn finite(n: f64) -> Result<Value> {
    serde_json::Number::from_f64(n)
        .map(Value::Number)
        .ok_or_else(|| Error::Argument("Judgment data must contain finite numbers".into()))
}

impl Serializer<'_> {
    fn serialize(&mut self, value: &Outcome) -> Result<Value> {
        Ok(match value {
            Outcome::Value(v) => v.clone(),
            Outcome::Chat(c) | Outcome::Agent(c) => self.conversation(c)?,
            Outcome::Message(m) => self.message(m)?,
            Outcome::ToolCall(c) => tool_call_data(c),
            Outcome::Answer(a) => a.to_value(),
            Outcome::Judgment(j) => judgment_data(j),
            Outcome::Embedding(e) => json!({
                "model": e.model,
                "vectors": vectors(&e.vectors)?,
                "sparse_vectors": e.sparse_vectors.as_ref().map_or(Ok(Value::Null), sparse)?,
            }),
            Outcome::Transcription(t) => json!({
                "model": t.model, "text": t.text, "language": t.language,
                "duration": t.duration.map_or(Ok(Value::Null), finite)?,
                "segments": t.segments, "words": t.words,
            }),
            Outcome::Ocr(o) => json!({
                "model": o.model, "markdown": o.markdown(),
                "pages": o.pages.iter().map(|p| json!({
                    "index": p.index, "markdown": p.markdown, "images": p.images,
                    "tables": p.tables, "raw": p.raw,
                })).collect::<Vec<_>>(),
            }),
            Outcome::Rerank(r) => json!({
                "model": r.model,
                "results": r.results.iter().map(|x| Ok(json!({
                    "index": x.index, "document": x.document,
                    "score": x.score.map_or(Ok(Value::Null), finite)?,
                }))).collect::<Result<Vec<_>>>()?,
            }),
            Outcome::Moderation(m) => json!({
                "model": m.model,
                "results": m.results.iter().map(moderation_result).collect::<Vec<_>>(),
            }),
            Outcome::ModerationResult(r) => moderation_result(r),
            Outcome::Tokenization(t) => json!({ "model": t.model, "ids": t.ids }),
            Outcome::ServerToolCall(c) => server_tool_call(c),
            Outcome::Image(i) => {
                let data = i.data.clone();
                let source = match (&data, &i.url) {
                    (Some(d), _) => format!(
                        "data:{};base64,{d}",
                        i.mime_type.as_deref().unwrap_or_default()
                    ),
                    (None, Some(url)) => url.clone(),
                    (None, None) => {
                        return Err(Error::Argument("Media result has no content".into()));
                    }
                };
                let model = (!i.model.is_empty()).then(|| i.model.clone());
                self.media(source, i.mime_type.clone(), model)
            }
            Outcome::Speech(s) => {
                let encoded = base64::engine::general_purpose::STANDARD.encode(&s.data);
                let source = format!("data:{};base64,{encoded}", s.mime_type);
                self.media(source, Some(s.mime_type.clone()), Some(s.model.clone()))
            }
            Outcome::Video(v) => {
                let source = match (&v.data, &v.url) {
                    (Some(d), _) => format!(
                        "data:{};base64,{}",
                        v.mime_type.as_deref().unwrap_or_default(),
                        base64::engine::general_purpose::STANDARD.encode(d)
                    ),
                    (None, Some(url)) => url.clone(),
                    (None, None) => {
                        return Err(Error::Argument("Media result has no content".into()));
                    }
                };
                self.media(source, v.mime_type.clone(), v.model.clone())
            }
            Outcome::Attachment(a) => self.attachment(a),
            Outcome::List(items) => Value::Array(
                items
                    .iter()
                    .map(|i| self.serialize(i))
                    .collect::<Result<_>>()?,
            ),
            Outcome::Map(items) => {
                let mut out = Map::new();
                for (k, v) in items {
                    if out.contains_key(k) {
                        return Err(Error::Argument("Duplicate evaluation evidence keys".into()));
                    }
                    let v = self.serialize(v)?;
                    out.insert(k.clone(), v);
                }
                Value::Object(out)
            }
            Outcome::Other(any) => {
                let adapted = self.adapters.iter().find_map(|a| a(&**any));
                match adapted {
                    Some(v) => v,
                    None => {
                        return Err(Error::Argument(
                            "Cannot evaluate this value; declare an adapter with adapt".into(),
                        ));
                    }
                }
            }
        })
    }

    /// `conversation_data`.
    fn conversation(&mut self, chat: &Chat) -> Result<Value> {
        let messages = chat
            .messages()
            .iter()
            .map(|m| self.message(m))
            .collect::<Result<Vec<_>>>()?;
        Ok(json!({
            "output": last_answer(chat.messages()),
            "messages": messages,
            "complete": chat.is_complete(),
            "waiting": chat.is_waiting(),
            "pending_approvals": chat.pending_approvals().iter().map(|c| c.id.clone()).collect::<Vec<_>>(),
            "cancelled": chat.is_cancelled(),
            "model": chat.model().id,
        }))
    }

    /// `message_data`.
    fn message(&mut self, m: &Message) -> Result<Value> {
        let tool_calls = m.tool_calls.as_ref().map(|calls| {
            Value::Object(
                calls
                    .iter()
                    .map(|(k, c)| (k.clone(), tool_call_data(c)))
                    .collect(),
            )
        });
        let attachments: Vec<Value> = m.attachments.iter().map(|a| self.attachment(a)).collect();
        Ok(json!({
            "role": m.role.as_str(),
            "content": m.content,
            "model": m.model,
            "finish_reason": m.finish_reason.as_ref().map(|f| f.as_str().to_string()),
            "tool_calls": tool_calls,
            "tool_call_id": m.tool_call_id,
            "citations": serde_json::to_value(&m.citations)?,
            "server_tool_calls": m.server_tool_calls.iter().map(server_tool_call).collect::<Vec<_>>(),
            "attachments": attachments,
        }))
    }

    /// `attachment_data`.
    fn attachment(&mut self, a: &Attachment) -> Value {
        self.attachments.push(EvidenceAttachment::File(a.clone()));
        json!({
            "attachment": self.attachments.len(),
            "filename": a.filename,
            "content_type": a.mime_type,
        })
    }

    /// `media_data`.
    fn media(&mut self, source: String, mime_type: Option<String>, model: Option<String>) -> Value {
        self.attachments.push(EvidenceAttachment::Source(source));
        json!({
            "attachment": self.attachments.len(),
            "content_type": mime_type,
            "model": model,
        })
    }
}

/// `Judgment#to_h`: `{ model:, answers:, tokens:, cost: }` with the full token and cost hashes.
pub(crate) fn judgment_data(j: &Judgment) -> Value {
    json!({
        "model": j.model,
        "answers": j.answers.iter().map(|(k, a)| (k.clone(), a.to_value())).collect::<Map<_, _>>(),
        "tokens": crate::instrumentation::tokens_h(&j.tokens()),
        "cost": crate::instrumentation::cost_h(&j.cost()),
    })
}

/// `conversation_data` for a chat the caller still owns (the reviewer's conversation).
pub(crate) fn chat_data(chat: &Chat) -> Result<Value> {
    Serializer {
        adapters: &[],
        attachments: Vec::new(),
    }
    .conversation(chat)
}

/// `ToolCall#to_h.except(:thought_signature)`.
fn tool_call_data(call: &ToolCall) -> Value {
    let mut h = Map::new();
    h.insert("id".into(), call.id.clone().into());
    h.insert("name".into(), call.name.clone().into());
    h.insert("arguments".into(), Value::Object(call.arguments()));
    if call.remote {
        h.insert("remote".into(), true.into());
    }
    Value::Object(h)
}

/// `FIELDS[ServerToolCall]`: `type name id input result`, without the raw payload.
fn server_tool_call(c: &ServerToolCall) -> Value {
    json!({ "type": c.kind, "name": c.name, "id": c.id, "input": c.input, "result": c.result })
}

fn moderation_result(r: &ModerationResult) -> Value {
    json!({
        "flagged": r.is_flagged(),
        "categories": r.categories,
        "category_scores": r.category_scores,
    })
}

fn vectors(v: &Vectors) -> Result<Value> {
    let row = |r: &Vec<f64>| -> Result<Value> {
        Ok(Value::Array(
            r.iter().map(|x| finite(*x)).collect::<Result<_>>()?,
        ))
    };
    match v {
        Vectors::Single(r) => row(r),
        Vectors::Batch(rows) => Ok(Value::Array(rows.iter().map(row).collect::<Result<_>>()?)),
    }
}

fn sparse(v: &SparseVectors) -> Result<Value> {
    let one = |m: &std::collections::BTreeMap<i64, f64>| -> Result<Value> {
        let mut out = Map::new();
        for (k, x) in m {
            out.insert(k.to_string(), finite(*x)?);
        }
        Ok(Value::Object(out))
    };
    match v {
        SparseVectors::Single(m) => one(m),
        SparseVectors::Batch(items) => Ok(Value::Array(
            items
                .iter()
                .map(|m| m.as_ref().map_or(Ok(Value::Null), one))
                .collect::<Result<_>>()?,
        )),
    }
}
