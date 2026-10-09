//! OpenAI Decisions: typed questions answered with probabilities. Port of
//! `lib/ruby_llm/protocols/openai/decisions.rb`.

use serde_json::{Map, Value, json};

use super::{Answer, Judgment, Question, QuestionType, Resolved};
use crate::attachment::{Attachment, AttachmentType};
use crate::error::{Error, ErrorResponse, Result};
use crate::message::{Message, RawResponse, Role};
use crate::model::Model;
use crate::protocols::anthropic::unsupported;
use crate::providers::Provider;

/// `Decisions#judgment_url`.
pub(super) const JUDGMENT_URL: &str = "decisions";

/// `Decisions::TYPES`.
fn wire(kind: QuestionType) -> &'static str {
    match kind {
        QuestionType::Probability => "predicate",
        QuestionType::Choice => "choice",
        QuestionType::Score => "score",
    }
}

/// `describe`: text stays text, structured descriptions go out as JSON text.
fn describe(value: &Value) -> Option<String> {
    match value {
        Value::Null => None,
        Value::String(s) => Some(s.clone()),
        other => Some(other.to_string()),
    }
}

/// `Decisions#render_judgment_payload`.
pub(super) fn render_payload(
    input: &Value,
    questions: &[Resolved],
    model: &str,
    with: &[Attachment],
    provider_options: &Value,
) -> Result<Value> {
    let options = provider_options.as_object().cloned().unwrap_or_default();
    let reserved: Vec<&str> = options
        .keys()
        .map(String::as_str)
        .filter(|k| ["model", "input", "questions"].contains(k))
        .collect();
    if !reserved.is_empty() {
        return Err(Error::Argument(format!(
            "Use the judgment arguments instead of provider_options for {}",
            reserved.join(", ")
        )));
    }
    let mut payload = json!({
        "model": model,
        "input": render_input(input, with)?,
        "questions": questions.iter().map(render_question).collect::<Vec<_>>(),
    });
    for (k, v) in options {
        payload[k] = v;
    }
    Ok(payload)
}

/// `render_input`: plain text, or one user message carrying the text and the images.
fn render_input(input: &Value, with: &[Attachment]) -> Result<Value> {
    let text = describe(input);
    if with.is_empty() {
        return Ok(text.map(Value::String).unwrap_or(Value::Null));
    }
    let images = with.iter().map(render_image).collect::<Result<Vec<_>>>()?;
    let mut content: Vec<Value> = text
        .filter(|t| !t.is_empty())
        .map(|t| json!({ "type": "input_text", "text": t }))
        .into_iter()
        .collect();
    content.extend(images);
    Ok(json!([{ "type": "message", "role": "user", "content": content }]))
}

/// `render_image`: an uploaded image by file ID, any other image inline as a data URI
/// (`Responses::Media.format_image` with `image_url: image.for_llm`).
fn render_image(image: &Attachment) -> Result<Value> {
    if image.kind() != AttachmentType::Image {
        return Err(Error::UnsupportedAttachment(unsupported(&image.mime_type)));
    }
    if let Some(id) = image.provider_file_id() {
        return Ok(json!({ "type": "input_image", "file_id": id }));
    }
    let mut part = json!({ "type": "input_image", "image_url": image.for_llm()? });
    if let Some(resolution) = image.resolution {
        part["detail"] = resolution
            .image_detail(Provider::OpenAI.is_original_image_detail())
            .into();
    }
    Ok(part)
}

/// `render_question`.
fn render_question(q: &Resolved) -> Value {
    let mut out = Map::new();
    out.insert("type".into(), wire(q.kind).into());
    out.insert("name".into(), q.name.clone().into());
    if let Some(instructions) = render_instructions(q) {
        out.insert("instructions".into(), instructions.into());
    }
    match q.kind {
        // `render_choices`
        QuestionType::Choice => {
            let choices: Vec<Value> = q
                .criteria
                .as_object()
                .into_iter()
                .flatten()
                .map(|(value, description)| {
                    let mut choice = json!({ "value": value });
                    if let Some(d) = describe(description) {
                        choice["description"] = d.into();
                    }
                    choice
                })
                .collect();
            out.insert("choices".into(), choices.into());
        }
        // `render_levels`: every level carries its description, even a nil one.
        QuestionType::Score => {
            let levels: Vec<Value> = q
                .criteria
                .as_array()
                .into_iter()
                .flatten()
                .enumerate()
                .map(|(index, d)| json!({ "label": index.to_string(), "description": describe(d) }))
                .collect();
            out.insert("levels".into(), levels.into());
        }
        QuestionType::Probability => {}
    }
    Value::Object(out)
}

/// `render_instructions`: a predicate's yes and no descriptions follow its instructions
/// (`OUTCOMES`).
fn render_instructions(q: &Resolved) -> Option<String> {
    let mut lines: Vec<String> = describe(&q.instructions).into_iter().collect();
    if q.kind == QuestionType::Probability
        && let Some(criteria) = q.criteria.as_object()
    {
        for (outcome, description) in criteria {
            if let Some(d) = describe(description) {
                let label = if outcome == "yes" || outcome == "true" {
                    "Yes"
                } else {
                    "No"
                };
                lines.push(format!("{label}: {d}"));
            }
        }
    }
    let instructions = lines.join("\n");
    (!instructions.is_empty()).then_some(instructions)
}

/// `Decisions#parse_judgment_response`: answers arrive in question order. A body that doesn't
/// map onto the questions is one `Error` carrying the response.
pub(super) fn parse_response(
    raw: RawResponse,
    questions: &[Resolved],
    model: &Model,
) -> Result<Judgment> {
    let parsed = parse_body(&raw.body, questions);
    let (answers, reported_model, tokens) = parsed.map_err(|message| {
        Error::Api(
            format!("OpenAI Decisions returned an invalid judgment: {message}"),
            Some(ErrorResponse {
                status: raw.status,
                body: raw.body.to_string(),
                ..Default::default()
            }),
        )
    })?;
    Ok(Judgment {
        answers,
        model: reported_model,
        raw: Some(raw),
        usage_entries: Vec::new(),
        tokens,
        model_info: Some(model.clone()),
    })
}

type Parsed = (Vec<(String, Answer)>, String, crate::tokens::Tokens);

/// `Hash#fetch`, with Ruby's `KeyError`/`NoMethodError` messages.
fn fetch<'a>(value: &'a Value, key: &str) -> std::result::Result<&'a Value, String> {
    let Some(map) = value.as_object() else {
        return Err(format!("undefined method 'fetch' for {value}"));
    };
    map.get(key)
        .ok_or_else(|| format!("key not found: {key:?}"))
}

fn number(value: &Value, key: &str) -> std::result::Result<f64, String> {
    let v = fetch(value, key)?;
    v.as_f64()
        .ok_or_else(|| format!("{key} is not a number: {v}"))
}

fn parse_body(body: &Value, questions: &[Resolved]) -> std::result::Result<Parsed, String> {
    let answers = fetch(body, "answers")?;
    let answers = answers
        .as_array()
        .ok_or_else(|| format!("answers is not an Array: {answers}"))?;
    let mut parsed = Vec::with_capacity(questions.len());
    for (index, q) in questions.iter().enumerate() {
        // `questions.values.zip(answers)`: a missing answer is nil.
        let answer = answers.get(index).unwrap_or(&Value::Null);
        parsed.push((q.name.clone(), parse_answer(answer, q)?));
    }
    let model = fetch(body, "model")?
        .as_str()
        .unwrap_or_default()
        .to_string();
    Ok((parsed, model, parse_tokens(body.get("usage"))))
}

fn parse_answer(answer: &Value, q: &Resolved) -> std::result::Result<Answer, String> {
    match q.kind {
        QuestionType::Probability => Ok(Answer::Probability {
            probability: number(answer, "probability")?,
        }),
        // `parse_choice`
        QuestionType::Choice => {
            let options: Vec<&String> = q
                .criteria
                .as_object()
                .map(|m| m.keys().collect())
                .unwrap_or_default();
            let option = |value: &Value| {
                value
                    .as_str()
                    .filter(|v| options.iter().any(|o| o == v))
                    .map(str::to_string)
                    .ok_or_else(|| format!("key not found: {value}"))
            };
            Ok(Answer::Choice {
                choice: option(fetch(answer, "choice")?)?,
                probabilities: distribution(answer, "value", option)?,
                confidence: number(answer, "confidence")?,
            })
        }
        // `parse_score`
        QuestionType::Score => {
            let levels = q.criteria.as_array().cloned().unwrap_or_default();
            let labels: Vec<String> = (0..levels.len()).map(|i| i.to_string()).collect();
            let index = |label: &Value| {
                labels
                    .iter()
                    .position(|l| label.as_str() == Some(l))
                    .ok_or_else(|| format!("key not found: {label}"))
            };
            Ok(Answer::Score {
                score: number(answer, "score")?,
                probabilities: distribution(answer, "label", index)?,
                levels,
                confidence: number(answer, "confidence")?,
            })
        }
    }
}

/// `parse_distribution`: the entries in the order the API listed them.
fn distribution<K>(
    answer: &Value,
    key: &str,
    name: impl Fn(&Value) -> std::result::Result<K, String>,
) -> std::result::Result<Vec<(K, f64)>, String> {
    let entries = fetch(answer, "probabilities")?;
    entries
        .as_array()
        .ok_or_else(|| format!("probabilities is not an Array: {entries}"))?
        .iter()
        .map(|entry| Ok((name(fetch(entry, key)?)?, number(entry, "probability")?)))
        .collect()
}

/// `parse_tokens`: `Responses::Chat.parse_usage`, so cached input is split out of the input count.
fn parse_tokens(usage: Option<&Value>) -> crate::tokens::Tokens {
    let mut message = Message::new(Role::Assistant, None::<String>);
    let empty = json!({});
    crate::protocols::responses::parse_usage(
        Provider::OpenAI,
        &mut message,
        usage.filter(|u| u.is_object()).unwrap_or(&empty),
    );
    message.tokens
}

fn resolve_all(questions: &[Question]) -> Result<Vec<Resolved>> {
    questions.iter().map(|q| q.resolve(&Map::new())).collect()
}

/// `Decisions#render_judgment_payload` for already-built questions, reading the attachments
/// first (URL images are downloaded to be sent inline). For spec ports.
#[doc(hidden)]
pub async fn render_judgment_payload(
    input: Value,
    questions: &[Question],
    model: &str,
    mut with: Vec<Attachment>,
    provider_options: Value,
) -> Result<Value> {
    let resolved = resolve_all(questions)?;
    let client = crate::transport::basic(&crate::config())?;
    super::load_images(&mut with, &client).await?;
    render_payload(&input, &resolved, model, &with, &provider_options)
}

/// `Decisions#parse_judgment_response` for a response body. For spec ports.
#[doc(hidden)]
pub fn parse_judgment_response(
    body: Value,
    questions: &[Question],
    model: &Model,
) -> Result<Judgment> {
    let resolved = resolve_all(questions)?;
    let raw = RawResponse {
        status: 200,
        headers: Vec::new(),
        body,
        request_body: "".into(),
    };
    parse_response(raw, &resolved, model)
}
