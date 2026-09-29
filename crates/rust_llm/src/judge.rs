//! Judgments: typed questions about your data, answered by a System One model such as TypeSafe's
//! Jev. Port of `lib/ruby_llm/judge.rb`, `judge/question.rb`, `judgment.rb`, `probability.rb`,
//! `choice.rb`, `score.rb`, and `protocols/system_one/*.rb`.
//!
//! ```ruby
//! class Urgency < RubyLLM::Judge
//!   probability :urgent, "Does this need attention today?"
//! end
//! Urgency.judge("Please refund the duplicate charge today.").urgent.probability
//! ```
//!
//! ```no_run
//! # use rust_llm::Judge;
//! # async fn run() -> rust_llm::Result<()> {
//! let urgency = Judge::new().probability("urgent", "Does this need attention today?")?;
//! urgency.judge("Please refund the duplicate charge today.").await?.probability("urgent");
//! # Ok(()) }
//! ```
//!
//! A judgment is one request: every question is answered about the same input. Nothing is
//! generated and no conversation history is kept.

use std::sync::Arc;

use serde_json::{Map, Value, json};

use crate::chat::{failure_tokens, resolve_model};
use crate::config::Config;
use crate::cost::{Cost, Tier};
use crate::error::{Error, Result};
use crate::message::{Operation, RawResponse, UsageEntry, UsageStatus};
use crate::model::Model;
use crate::providers::Provider;
use crate::tokens::Tokens;
use crate::transport::Connection;

/// `System One choices support at most 255 options`.
const MAX_CHOICES: usize = 255;
/// `System One scores support at most 10 levels`.
const MAX_LEVELS: usize = 10;

/// The three question types (`probability`, `choice`, `score`), with their wire names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuestionType {
    Probability,
    Choice,
    Score,
}

impl QuestionType {
    /// `Judgments::TYPES`.
    fn wire(&self) -> &'static str {
        match self {
            QuestionType::Probability => "noul",
            QuestionType::Choice => "choice",
            QuestionType::Score => "score",
        }
    }

    /// `Question::CRITERIA_KEYS`: where `RubyLLM.judge(questions:)` data puts the criteria.
    fn criteria_key(&self) -> &'static str {
        match self {
            QuestionType::Probability => "criteria",
            QuestionType::Choice => "options",
            QuestionType::Score => "levels",
        }
    }

    /// `CRITERIA_KEYS.fetch(type) { raise ArgumentError, "Unknown judgment type: #{type.inspect}" }`
    /// (`judge/question.rb`); the message spells the value as Ruby's `inspect` would (`nil`,
    /// `:text`, `42`), since `from_h` turns a String type into a Symbol first.
    fn parse(value: Option<&Value>) -> Result<QuestionType> {
        match value.and_then(Value::as_str) {
            Some("probability") => Ok(QuestionType::Probability),
            Some("choice") => Ok(QuestionType::Choice),
            Some("score") => Ok(QuestionType::Score),
            _ => {
                let inspected = match value {
                    None | Some(Value::Null) => "nil".to_string(),
                    Some(Value::String(s)) => format!(":{s}"),
                    Some(other) => other.to_string(),
                };
                Err(Error::Argument(format!(
                    "Unknown judgment type: {inspected}"
                )))
            }
        }
    }
}

/// A value that may depend on the judge's runtime inputs, resolved once per judgment (a proc in
/// RubyLLM). Plain values convert into it.
/// Computes a value from the judge's declared inputs.
pub type Resolver = Arc<dyn Fn(&Map<String, Value>) -> Value + Send + Sync>;

#[derive(Clone)]
pub enum Dynamic {
    Value(Value),
    Resolve(Resolver),
}

impl Dynamic {
    fn resolve(&self, inputs: &Map<String, Value>) -> Value {
        match self {
            Dynamic::Value(v) => v.clone(),
            Dynamic::Resolve(f) => f(inputs),
        }
    }
}

impl std::fmt::Debug for Dynamic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Dynamic::Value(v) => write!(f, "{v}"),
            Dynamic::Resolve(_) => f.write_str("<resolved per judgment>"),
        }
    }
}

impl<T: Into<Value>> From<T> for Dynamic {
    fn from(value: T) -> Self {
        Dynamic::Value(value.into())
    }
}

/// `Dynamic::from_fn(|inputs| ...)`: a value computed from the judge's declared inputs.
impl Dynamic {
    pub fn from_fn(f: impl Fn(&Map<String, Value>) -> Value + Send + Sync + 'static) -> Dynamic {
        Dynamic::Resolve(Arc::new(f))
    }
}

/// `Judge::Question`.
#[derive(Debug, Clone)]
pub struct Question {
    pub name: String,
    pub kind: QuestionType,
    instructions: Option<Dynamic>,
    criteria: Option<Dynamic>,
}

/// A question after its dynamic parts were resolved and validated.
#[derive(Debug, Clone)]
struct Resolved {
    name: String,
    kind: QuestionType,
    instructions: Value,
    criteria: Value,
}

fn is_description(v: &Value) -> bool {
    matches!(
        v,
        Value::Null | Value::String(_) | Value::Object(_) | Value::Array(_)
    )
}

fn validate_descriptions<'a>(values: impl IntoIterator<Item = &'a Value>) -> Result<()> {
    if values.into_iter().all(is_description) {
        Ok(())
    } else {
        Err(Error::Argument(
            "Descriptions must be text, a Hash, an Array, or nil".into(),
        ))
    }
}

/// `Data.copy`: judgment data must be JSON with finite numbers (serde_json cannot hold others, so
/// this only needs to reject non-finite values arriving via `f64`).
fn validate_data(v: &Value) -> Result<()> {
    match v {
        Value::Number(n) if n.as_f64().is_some_and(|f| !f.is_finite()) => Err(Error::Argument(
            "Judgment data must contain finite numbers".into(),
        )),
        Value::Array(items) => items.iter().try_for_each(validate_data),
        Value::Object(map) => map.values().try_for_each(validate_data),
        _ => Ok(()),
    }
}

impl Question {
    fn new(
        name: impl Into<String>,
        kind: QuestionType,
        instructions: Option<Dynamic>,
        criteria: Option<Dynamic>,
    ) -> Result<Question> {
        let name = name.into();
        if name.is_empty() {
            return Err(Error::Argument("A question name cannot be empty".into()));
        }
        Ok(Question {
            name,
            kind,
            instructions,
            criteria,
        })
    }

    /// `Question.from_h`: one entry of `RubyLLM.judge(questions: { ... })`.
    pub fn from_value(name: impl Into<String>, definition: &Value) -> Result<Question> {
        let Some(def) = definition.as_object() else {
            return Err(Error::Argument("Each question must be a Hash".into()));
        };
        let kind = QuestionType::parse(def.get("type"))?;
        let key = kind.criteria_key();
        let extra: Vec<&str> = def
            .keys()
            .map(String::as_str)
            .filter(|k| !["type", "instructions", key].contains(k))
            .collect();
        if !extra.is_empty() {
            return Err(Error::Argument(format!(
                "Unknown question options: {}",
                extra.join(", ")
            )));
        }
        let instructions = def
            .get("instructions")
            .filter(|v| !v.is_null())
            .cloned()
            .map(Dynamic::Value);
        let criteria = def
            .get(key)
            .filter(|v| !v.is_null())
            .cloned()
            .map(Dynamic::Value);
        Question::new(name, kind, instructions, criteria)
    }

    /// `Question#resolve` + `#validate!`.
    fn resolve(&self, inputs: &Map<String, Value>) -> Result<Resolved> {
        let instructions = self
            .instructions
            .as_ref()
            .map(|d| d.resolve(inputs))
            .unwrap_or(Value::Null);
        let criteria = self
            .criteria
            .as_ref()
            .map(|d| d.resolve(inputs))
            .unwrap_or(Value::Null);
        validate_data(&instructions)?;
        validate_data(&criteria)?;
        if !is_description(&instructions) {
            return Err(Error::Argument(
                "Question instructions must be text, a Hash, an Array, or nil".into(),
            ));
        }
        match self.kind {
            QuestionType::Probability => {
                if !criteria.is_null() {
                    let Some(map) = criteria.as_object() else {
                        return Err(Error::Argument(
                            "Probability criteria must describe yes and no".into(),
                        ));
                    };
                    if map
                        .keys()
                        .any(|k| !["yes", "no", "true", "false"].contains(&k.as_str()))
                    {
                        return Err(Error::Argument(
                            "Probability criteria must describe yes and no".into(),
                        ));
                    }
                    let positives: Vec<bool> =
                        map.keys().map(|k| k == "yes" || k == "true").collect();
                    if positives.iter().filter(|p| **p).count() > 1
                        || positives.iter().filter(|p| !**p).count() > 1
                    {
                        return Err(Error::Argument(
                            "Probability criteria contain duplicate outcomes".into(),
                        ));
                    }
                    validate_descriptions(map.values())?;
                }
            }
            QuestionType::Choice => {
                let map = criteria
                    .as_object()
                    .filter(|m| !m.is_empty())
                    .ok_or_else(|| {
                        Error::Argument("A choice needs a nonempty Hash of options".into())
                    })?;
                if map.keys().any(String::is_empty) {
                    return Err(Error::Argument(
                        "Choice options must have nonempty String or Symbol names".into(),
                    ));
                }
                validate_descriptions(map.values())?;
            }
            QuestionType::Score => {
                let levels = criteria
                    .as_array()
                    .filter(|l| l.len() >= 2 && l.iter().all(|v| !v.is_null()));
                let Some(levels) = levels else {
                    return Err(Error::Argument(
                        "A score needs at least two non-nil levels".into(),
                    ));
                };
                validate_descriptions(levels)?;
            }
        }
        Ok(Resolved {
            name: self.name.clone(),
            kind: self.kind,
            instructions,
            criteria,
        })
    }
}

/// Where a judgment is sent: `model "jev-latest", provider: :typesafe, assume_model_exists: true`.
#[derive(Debug, Clone, Default)]
pub struct JudgeModel {
    pub model: Option<Dynamic>,
    pub provider: Option<String>,
    pub assume_model_exists: bool,
}

/// `RubyLLM::Judge`: reusable question definitions. Build once, judge many inputs.
#[derive(Debug, Clone, Default)]
pub struct Judge {
    model: JudgeModel,
    inputs: Vec<String>,
    questions: Vec<Question>,
    provider_options: Option<Dynamic>,
    config: Option<Arc<Config>>,
}

/// Per-call options for `judge_with` (`judge(input, model:, provider:, questions:, provider_options:)`).
#[derive(Debug, Clone, Default)]
pub struct JudgeOptions {
    /// `model:`. `Some(None)` is Ruby's explicit `model: nil`: use the configured default again.
    pub model: Option<Option<String>>,
    pub provider: Option<String>,
    pub assume_model_exists: Option<bool>,
    /// One-off questions added to the declared ones (`questions:`).
    pub questions: Map<String, Value>,
    pub provider_options: Option<Value>,
    /// Values for the judge's declared `inputs`.
    pub inputs: Map<String, Value>,
    /// `context:`: an isolated configuration for this call (its keys and default judgment model),
    /// taking precedence over `with_config`.
    pub config: Option<Arc<Config>>,
}

impl Judge {
    pub fn new() -> Judge {
        Judge::default()
    }

    /// `model "jev-latest"`.
    pub fn model(mut self, model: impl Into<Dynamic>) -> Judge {
        self.model.model = Some(model.into());
        self
    }

    /// `model id, provider: :typesafe`.
    pub fn provider(mut self, provider: impl Into<String>) -> Judge {
        self.model.provider = Some(provider.into());
        self
    }

    /// `model id, provider:, assume_model_exists: true`: for model ids outside the registry,
    /// such as a local Jev-compatible server's.
    pub fn assume_model_exists(mut self) -> Judge {
        self.model.assume_model_exists = true;
        self
    }

    /// `inputs :teams`: required runtime values, available to `Dynamic::from_fn` closures. They
    /// are not sent to the model unless you put them in the input or a question.
    pub fn inputs<S: Into<String>>(mut self, names: impl IntoIterator<Item = S>) -> Judge {
        self.inputs = names.into_iter().map(Into::into).collect();
        self
    }

    /// `provider_options { ... }`: merged into the request (reserved fields are rejected).
    pub fn provider_options(mut self, options: impl Into<Dynamic>) -> Judge {
        self.provider_options = Some(options.into());
        self
    }

    /// `RubyLLM.context { ... }.judge`: use this configuration instead of the global one.
    pub fn with_config(mut self, config: Arc<Config>) -> Judge {
        self.config = Some(config);
        self
    }

    fn declare(mut self, question: Result<Question>) -> Result<Judge> {
        let question = question?;
        if self.questions.iter().any(|q| q.name == question.name) {
            return Err(Error::Argument(format!(
                "Duplicate question: {}",
                question.name
            )));
        }
        self.questions.push(question);
        Ok(self)
    }

    /// Replaces an inherited question of the same name (a subclass redeclaring it).
    pub fn replacing(mut self, name: &str) -> Judge {
        self.questions.retain(|q| q.name != name);
        self
    }

    /// `probability :urgent, "Does this need attention today?"`.
    pub fn probability(
        self,
        name: impl Into<String>,
        instructions: impl Into<Dynamic>,
    ) -> Result<Judge> {
        self.declare(Question::new(
            name,
            QuestionType::Probability,
            Some(instructions.into()),
            None,
        ))
    }

    /// `probability :urgent, "..." do yes "..."; no "..." end`. Either description may be `null`.
    pub fn probability_with(
        self,
        name: impl Into<String>,
        instructions: Option<Dynamic>,
        yes: impl Into<Dynamic>,
        no: impl Into<Dynamic>,
    ) -> Result<Judge> {
        let (yes, no) = (yes.into(), no.into());
        let criteria = Dynamic::from_fn(
            move |inputs| json!({ "yes": yes.resolve(inputs), "no": no.resolve(inputs) }),
        );
        self.declare(Question::new(
            name,
            QuestionType::Probability,
            instructions,
            Some(criteria),
        ))
    }

    /// `choice :department, "Which team?" do billing "..."; other nil end`. `options` is a JSON
    /// object of option name to description (or `null`); insertion order is kept.
    pub fn choice(
        self,
        name: impl Into<String>,
        instructions: Option<Dynamic>,
        options: impl Into<Dynamic>,
    ) -> Result<Judge> {
        self.declare(Question::new(
            name,
            QuestionType::Choice,
            instructions,
            Some(options.into()),
        ))
    }

    /// `score :frustration, "How frustrated?", ["Calm", "Frustrated", "Angry"]`: ordered levels.
    pub fn score(
        self,
        name: impl Into<String>,
        instructions: Option<Dynamic>,
        levels: impl Into<Dynamic>,
    ) -> Result<Judge> {
        self.declare(Question::new(
            name,
            QuestionType::Score,
            instructions,
            Some(levels.into()),
        ))
    }

    /// `Judge.judge(input)`: text, a JSON object, or a JSON array.
    pub async fn judge(&self, input: impl Into<Value>) -> Result<Judgment> {
        self.judge_with(input, JudgeOptions::default()).await
    }

    /// `Judge.judge(input, model:, provider:, questions:, provider_options:, **inputs)`.
    pub async fn judge_with(
        &self,
        input: impl Into<Value>,
        options: JudgeOptions,
    ) -> Result<Judgment> {
        let missing: Vec<&str> = self
            .inputs
            .iter()
            .filter(|n| !options.inputs.contains_key(*n))
            .map(String::as_str)
            .collect();
        if !missing.is_empty() {
            return Err(Error::Argument(format!(
                "Missing judge inputs: {}",
                missing.join(", ")
            )));
        }
        let extra: Vec<&str> = options
            .inputs
            .keys()
            .filter(|k| !self.inputs.contains(k))
            .map(String::as_str)
            .collect();
        if !extra.is_empty() {
            return Err(Error::Argument(format!(
                "Unknown judge inputs: {}",
                extra.join(", ")
            )));
        }
        let input = input.into();
        if !matches!(input, Value::String(_) | Value::Object(_) | Value::Array(_)) {
            return Err(Error::Argument(
                "Judgment input must be text, a Hash, or an Array".into(),
            ));
        }
        validate_data(&input)?;

        let mut questions = self.questions.clone();
        for (name, definition) in &options.questions {
            if questions.iter().any(|q| &q.name == name) {
                return Err(Error::Argument(format!("Duplicate question: {name}")));
            }
            questions.push(Question::from_value(name.clone(), definition)?);
        }
        if questions.is_empty() {
            return Err(Error::Argument(
                "A judgment needs at least one question".into(),
            ));
        }
        let resolved: Vec<Resolved> = questions
            .iter()
            .map(|q| q.resolve(&options.inputs))
            .collect::<Result<_>>()?;

        let model = match options.model {
            Some(explicit) => explicit,
            None => self
                .model
                .model
                .as_ref()
                .map(|d| d.resolve(&options.inputs))
                .and_then(|v| v.as_str().map(str::to_string)),
        };
        let provider_options = match (&options.provider_options, &self.provider_options) {
            (Some(o), _) => o.clone(),
            (None, Some(d)) => d.resolve(&options.inputs),
            (None, None) => Value::Object(Map::new()),
        };
        judge_request(
            input,
            &resolved,
            model,
            options.provider.or_else(|| self.model.provider.clone()),
            options
                .assume_model_exists
                .unwrap_or(self.model.assume_model_exists),
            provider_options,
            options.config.or_else(|| self.config.clone()),
        )
        .await
    }
}

/// `RubyLLM.judge(input, questions: {...}, model:, provider:)`: questions from data.
pub async fn judge(
    input: impl Into<Value>,
    questions: Value,
    options: JudgeOptions,
) -> Result<Judgment> {
    let Value::Object(questions) = questions else {
        return Err(Error::Argument("Questions must be a Hash".into()));
    };
    let options = JudgeOptions {
        questions,
        ..options
    };
    Judge::new().judge_with(input, options).await
}

/// `Judgment.judge` + `Provider#judge` + the System One protocol, inside a `judgment.rust_llm`
/// event.
async fn judge_request(
    input: Value,
    questions: &[Resolved],
    model: Option<String>,
    provider: Option<String>,
    assume_model_exists: bool,
    provider_options: Value,
    config: Option<Arc<Config>>,
) -> Result<Judgment> {
    let config = config.unwrap_or_else(crate::config);
    let model_id = model
        .clone()
        .unwrap_or_else(|| config.default_judgment_model.clone());
    let resolved = resolve_model(&model_id, provider.as_deref(), assume_model_exists).ok();
    let mut event = crate::instrumentation::Event::start(&config, "judgment.rust_llm", || {
        let empty = Tokens::default();
        let (slug, display, id, cost) = match &resolved {
            Some((m, p)) => (
                Some(p.slug()),
                Some(p.display()),
                Some(m.id.clone()),
                Cost::new(&empty, Some(m), Tier::Standard),
            ),
            None => (None, None, None, Cost::new(&empty, None, Tier::Standard)),
        };
        crate::instrumentation::payload([
            ("provider", slug.into()),
            ("provider_class", display.into()),
            ("model", id.into()),
            ("question_count", questions.len().into()),
            ("provider_options", provider_options.clone()),
            ("tokens", crate::instrumentation::tokens_h(&empty)),
            ("cost", crate::instrumentation::cost_h(&cost)),
        ])
    });
    let request = judge_request_inner(
        input,
        questions,
        model,
        provider,
        assume_model_exists,
        provider_options,
        config.clone(),
    );
    let result = tracing::Instrument::instrument(request, event.span()).await;
    if let Ok(j) = &result {
        event.set("result", || {
            serde_json::json!(j.answers.iter().map(|(k, _)| k.clone()).collect::<Vec<_>>())
        });
        event.set("tokens", || crate::instrumentation::tokens_h(&j.tokens()));
        event.set("cost", || crate::instrumentation::cost_h(&j.cost()));
    }
    event.finish(result.as_ref().err());
    result
}

async fn judge_request_inner(
    input: Value,
    questions: &[Resolved],
    model: Option<String>,
    provider: Option<String>,
    assume_model_exists: bool,
    provider_options: Value,
    config: Arc<Config>,
) -> Result<Judgment> {
    let model_id = model.unwrap_or_else(|| config.default_judgment_model.clone());
    if model_id.is_empty() {
        return Err(Error::Argument("A judgment requires a model".into()));
    }
    let (model, provider) = resolve_model(&model_id, provider.as_deref(), assume_model_exists)?;
    if provider != Provider::TypeSafe {
        return Err(Error::Api(
            format!("{} doesn't support judgments", provider.display()),
            None,
        ));
    }
    provider.ensure_configured(&config)?;
    let payload = render_payload(&input, questions, &model.id, &provider_options)?;
    let connection = Connection::new(provider, config)?;

    // `track_usage(:judgment)`: one entry per HTTP attempt.
    let mut retried: Vec<Tokens> = Vec::new();
    let mut on_attempt = |previous: Option<&Error>| {
        if let Some(e) = previous {
            retried.push(failure_tokens(e, None));
        }
    };
    let raw = connection
        .post("v1/systemone", &payload, &[], &mut on_attempt)
        .await
        .map_err(system_one_error)?;
    let mut judgment = parse_response(raw, questions, &model)?;
    let entry = |status, tokens: Tokens| UsageEntry {
        id: UsageEntry::next_id(),
        operation: Operation::Judgment,
        provider: provider.slug().into(),
        model: model.id.clone(),
        status,
        cost: Cost::new(&tokens, Some(&model), Tier::Standard),
        tokens,
    };
    let mut entries: Vec<UsageEntry> = retried
        .into_iter()
        .map(|t| entry(UsageStatus::Failed, t))
        .collect();
    entries.push(entry(UsageStatus::Succeeded, judgment.tokens.clone()));
    judgment.usage_entries = entries;
    Ok(judgment)
}

/// `Judgments#render_judgment_payload`.
fn render_payload(
    input: &Value,
    questions: &[Resolved],
    model: &str,
    provider_options: &Value,
) -> Result<Value> {
    let options = provider_options.as_object().cloned().unwrap_or_default();
    let reserved: Vec<&str> = options
        .keys()
        .map(String::as_str)
        .filter(|k| ["model", "state", "questions"].contains(k))
        .collect();
    if !reserved.is_empty() {
        return Err(Error::Argument(format!(
            "Use the judgment arguments instead of provider_options for {}",
            reserved.join(", ")
        )));
    }
    let mut rendered = Map::new();
    for q in questions {
        rendered.insert(q.name.clone(), render_question(q)?);
    }
    let mut payload = json!({ "model": model, "state": input, "questions": rendered });
    for (k, v) in options {
        payload[k] = v;
    }
    Ok(payload)
}

/// `Judgments#render_question`.
fn render_question(q: &Resolved) -> Result<Value> {
    if q.kind == QuestionType::Choice
        && q.criteria
            .as_object()
            .is_some_and(|c| c.len() > MAX_CHOICES)
    {
        return Err(Error::Argument(
            "System One choices support at most 255 options".into(),
        ));
    }
    if q.kind == QuestionType::Score && q.criteria.as_array().is_some_and(|c| c.len() > MAX_LEVELS)
    {
        return Err(Error::Argument(
            "System One scores support at most 10 levels".into(),
        ));
    }
    let criteria = match (&q.kind, &q.criteria) {
        // BOOLEAN_KEYS: yes/no go out as "true"/"false".
        (QuestionType::Probability, Value::Object(map)) => Value::Object(
            map.iter()
                .map(|(k, v)| {
                    (
                        if k == "yes" || k == "true" {
                            "true"
                        } else {
                            "false"
                        }
                        .to_string(),
                        v.clone(),
                    )
                })
                .collect(),
        ),
        (_, c) => c.clone(),
    };
    let mut out = Map::new();
    out.insert("type".into(), q.kind.wire().into());
    if !q.instructions.is_null() {
        out.insert("instructions".into(), q.instructions.clone());
    }
    if !criteria.is_null() {
        out.insert("criteria".into(), criteria);
    }
    Ok(Value::Object(out))
}

/// `SystemOne#parse_error_response`: FastAPI-style `detail` as a string, `{message}`, or a list.
fn system_one_error(error: Error) -> Error {
    let Some(body) = error.response().map(|r| r.body.clone()) else {
        return error;
    };
    let Ok(parsed) = serde_json::from_str::<Value>(&body) else {
        return error;
    };
    let message = match parsed.get("detail") {
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Object(d)) => d.get("message").and_then(Value::as_str).map(str::to_string),
        Some(Value::Array(errors)) => Some(
            errors
                .iter()
                .map(|e| {
                    let loc: Vec<String> = e
                        .get("loc")
                        .and_then(Value::as_array)
                        .map(|l| {
                            l.iter()
                                .map(|p| {
                                    p.as_str()
                                        .map(str::to_string)
                                        .unwrap_or_else(|| p.to_string())
                                })
                                .collect()
                        })
                        .unwrap_or_default();
                    [
                        loc.join("."),
                        e.get("msg")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string(),
                    ]
                    .into_iter()
                    .filter(|s| !s.is_empty())
                    .collect::<Vec<_>>()
                    .join(": ")
                })
                .collect::<Vec<_>>()
                .join("; "),
        ),
        _ => None,
    };
    match message {
        Some(m) => error.with_message(m),
        None => error,
    }
}

/// One answer (`Probability`, `Choice`, or `Score`).
#[derive(Debug, Clone, PartialEq)]
pub enum Answer {
    /// `RubyLLM::Probability`: the probability of yes.
    Probability { probability: f64 },
    /// `RubyLLM::Choice`: the chosen option, a probability per option (declared order), and how
    /// concentrated that distribution is.
    Choice {
        choice: String,
        probabilities: Vec<(String, f64)>,
        confidence: f64,
    },
    /// `RubyLLM::Score`: a probability-weighted position on the zero-based scale.
    Score {
        score: f64,
        levels: Vec<Value>,
        probabilities: Vec<(usize, f64)>,
        confidence: f64,
    },
}

impl Answer {
    pub fn probability(&self) -> Option<f64> {
        match self {
            Answer::Probability { probability } => Some(*probability),
            _ => None,
        }
    }
    pub fn choice(&self) -> Option<&str> {
        match self {
            Answer::Choice { choice, .. } => Some(choice),
            _ => None,
        }
    }
    pub fn score(&self) -> Option<f64> {
        match self {
            Answer::Score { score, .. } => Some(*score),
            _ => None,
        }
    }
    /// Probability answers have no separate confidence.
    pub fn confidence(&self) -> Option<f64> {
        match self {
            Answer::Choice { confidence, .. } | Answer::Score { confidence, .. } => {
                Some(*confidence)
            }
            Answer::Probability { .. } => None,
        }
    }
    /// `Answer#to_h`.
    pub fn to_value(&self) -> Value {
        match self {
            Answer::Probability { probability } => {
                json!({ "type": "probability", "probability": probability })
            }
            Answer::Choice {
                choice,
                probabilities,
                confidence,
            } => json!({
                "type": "choice", "choice": choice, "confidence": confidence,
                "probabilities": probabilities.iter().map(|(k, v)| (k.clone(), json!(v))).collect::<Map<_, _>>(),
            }),
            Answer::Score {
                score,
                levels,
                probabilities,
                confidence,
            } => json!({
                "type": "score", "score": score, "levels": levels, "confidence": confidence,
                "probabilities": probabilities.iter().map(|(k, v)| (k.to_string(), json!(v))).collect::<Map<_, _>>(),
            }),
        }
    }
}

/// `RubyLLM::Judgment`: the answers, in declared order, with usage.
#[derive(Debug, Clone)]
pub struct Judgment {
    pub answers: Vec<(String, Answer)>,
    /// The model that answered, as the provider reported it (e.g. `jev-1.13.0`).
    pub model: String,
    pub raw: Option<RawResponse>,
    pub usage_entries: Vec<UsageEntry>,
    tokens: Tokens,
    model_info: Option<Model>,
}

impl Judgment {
    /// `Judgment.new(answers:, model:, tokens:)` (`judgment.rb`): a judgment built from answers
    /// you already have, with no usage entries or model info (so cost stays unknown).
    pub fn new(
        answers: Vec<(String, Answer)>,
        model: impl Into<String>,
        tokens: Tokens,
    ) -> Judgment {
        Judgment {
            answers,
            model: model.into(),
            raw: None,
            usage_entries: Vec::new(),
            tokens,
            model_info: None,
        }
    }

    /// `judgment[:urgent]`: `None` for an unknown name.
    pub fn get(&self, name: &str) -> Option<&Answer> {
        self.answers.iter().find(|(n, _)| n == name).map(|(_, a)| a)
    }

    /// `judgment.fetch(:urgent)`.
    pub fn fetch(&self, name: &str) -> Result<&Answer> {
        self.get(name)
            .ok_or_else(|| Error::Argument(format!("key not found: {name:?}")))
    }

    /// `judgment.urgent.probability`.
    pub fn probability(&self, name: &str) -> Option<f64> {
        self.get(name)?.probability()
    }

    /// `judgment.department.choice`.
    pub fn choice(&self, name: &str) -> Option<&str> {
        self.get(name)?.choice()
    }

    /// `judgment.frustration.score`.
    pub fn score(&self, name: &str) -> Option<f64> {
        self.get(name)?.score()
    }

    pub fn tokens(&self) -> Tokens {
        if self.usage_entries.is_empty() {
            self.tokens.clone()
        } else {
            Tokens::aggregate(
                self.usage_entries
                    .iter()
                    .filter(|e| e.status == UsageStatus::Succeeded)
                    .map(|e| &e.tokens),
            )
        }
    }

    /// Unknown prices stay `None`: the TypeSafe catalog carries no pricing.
    pub fn cost(&self) -> Cost {
        if self.usage_entries.is_empty() {
            return Cost::new(&self.tokens, self.model_info.as_ref(), Tier::Standard);
        }
        let complete = self.usage_entries.iter().all(UsageEntry::cost_available);
        Cost::aggregate(self.usage_entries.iter().map(|e| &e.cost), complete)
    }

    pub fn to_value(&self) -> Value {
        json!({
            "model": self.model,
            "answers": self.answers.iter().map(|(k, a)| (k.clone(), a.to_value())).collect::<Map<_, _>>(),
            "tokens": { "input": self.tokens().input, "output": self.tokens().output },
            "cost": { "total": self.cost().total() },
        })
    }
}

fn invalid(message: impl std::fmt::Display) -> Error {
    Error::Api(
        format!("System One returned an invalid judgment: {message}"),
        None,
    )
}

fn probability_value(v: Option<&Value>) -> Result<f64> {
    match v.and_then(Value::as_f64) {
        Some(p) if p.is_finite() && (0.0..=1.0).contains(&p) => Ok(p),
        _ => Err(invalid(
            "Probabilities and confidence must be numbers between 0 and 1",
        )),
    }
}

/// `Responses#parse_judgment_response`.
fn parse_response(raw: RawResponse, questions: &[Resolved], model: &Model) -> Result<Judgment> {
    let body = &raw.body;
    let valid = body
        .get("model")
        .and_then(Value::as_str)
        .is_some_and(|m| !m.is_empty())
        && body.get("answers").is_some_and(Value::is_object)
        && body.get("usage").is_some_and(Value::is_object);
    if !valid {
        return Err(Error::Api(
            "System One returned an invalid judgment response".into(),
            None,
        ));
    }
    let answers = body["answers"].as_object().expect("checked above");
    let mut got: Vec<&str> = answers.keys().map(String::as_str).collect();
    let mut asked: Vec<&str> = questions.iter().map(|q| q.name.as_str()).collect();
    got.sort_unstable();
    asked.sort_unstable();
    if got != asked {
        return Err(Error::Api(
            "System One returned different question IDs from the request".into(),
            None,
        ));
    }
    let usage = &body["usage"];
    for key in ["input_tokens", "output_tokens"] {
        match usage.get(key) {
            None | Some(Value::Null) => {}
            Some(v) if v.as_u64().is_some() => {}
            _ => {
                return Err(Error::Api(
                    format!("System One returned invalid {key}"),
                    None,
                ));
            }
        }
    }
    let parsed: Vec<(String, Answer)> = questions
        .iter()
        .map(|q| Ok((q.name.clone(), parse_answer(&answers[&q.name], q)?)))
        .collect::<Result<_>>()?;
    Ok(Judgment {
        answers: parsed,
        model: body["model"].as_str().unwrap_or_default().to_string(),
        tokens: Tokens {
            input: usage.get("input_tokens").and_then(Value::as_i64),
            output: usage.get("output_tokens").and_then(Value::as_i64),
            ..Default::default()
        },
        raw: Some(raw),
        usage_entries: Vec::new(),
        model_info: Some(model.clone()),
    })
}

fn parse_answer(answer: &Value, q: &Resolved) -> Result<Answer> {
    if answer.get("type").and_then(Value::as_str) != Some(q.kind.wire()) {
        return Err(invalid(format!("Unexpected answer type for {}", q.name)));
    }
    match q.kind {
        QuestionType::Probability => Ok(Answer::Probability {
            probability: probability_value(answer.get("noul"))?,
        }),
        QuestionType::Choice => {
            let options: Vec<&String> = q
                .criteria
                .as_object()
                .map(|m| m.keys().collect())
                .unwrap_or_default();
            let choice = answer
                .get("choice")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid("missing choice"))?;
            if !options.iter().any(|o| *o == choice) {
                return Err(invalid(format!("key not found: {choice:?}")));
            }
            Ok(Answer::Choice {
                choice: choice.to_string(),
                probabilities: distribution(
                    answer.get("probabilities"),
                    &options.iter().map(|o| o.to_string()).collect::<Vec<_>>(),
                )?
                .into_iter()
                .collect(),
                confidence: probability_value(answer.get("confidence"))?,
            })
        }
        QuestionType::Score => {
            let count = q.criteria.as_array().map(Vec::len).unwrap_or(0);
            let keys: Vec<String> = (0..count).map(|i| i.to_string()).collect();
            let legend = answer
                .get("legend")
                .and_then(Value::as_object)
                .ok_or_else(|| invalid("Unexpected score levels"))?;
            let mut legend_keys: Vec<&str> = legend.keys().map(String::as_str).collect();
            let mut expected: Vec<&str> = keys.iter().map(String::as_str).collect();
            legend_keys.sort_unstable();
            expected.sort_unstable();
            if legend_keys != expected {
                return Err(invalid("Unexpected score levels"));
            }
            let levels: Vec<Value> = keys.iter().map(|k| legend[k].clone()).collect();
            if levels.iter().any(|l| l.is_null() || !is_description(l)) {
                return Err(invalid("Invalid score descriptions"));
            }
            let score = answer.get("score").and_then(Value::as_f64);
            let score = match score {
                Some(s) if s.is_finite() && s >= 0.0 && s <= (count.saturating_sub(1)) as f64 => s,
                _ => return Err(invalid(format!("Invalid score for {}", q.name))),
            };
            Ok(Answer::Score {
                score,
                levels,
                probabilities: distribution(answer.get("probabilities"), &keys)?
                    .into_iter()
                    .map(|(k, v)| (k.parse().unwrap_or(0), v))
                    .collect(),
                confidence: probability_value(answer.get("confidence"))?,
            })
        }
    }
}

/// `parse_probabilities`: exactly the declared keys, returned in declared order.
fn distribution(values: Option<&Value>, keys: &[String]) -> Result<Vec<(String, f64)>> {
    let map = values
        .and_then(Value::as_object)
        .ok_or_else(|| invalid("Unexpected probability distribution keys"))?;
    let mut got: Vec<&str> = map.keys().map(String::as_str).collect();
    let mut want: Vec<&str> = keys.iter().map(String::as_str).collect();
    got.sort_unstable();
    want.sort_unstable();
    if got != want {
        return Err(invalid("Unexpected probability distribution keys"));
    }
    keys.iter()
        .map(|k| Ok((k.clone(), probability_value(map.get(k))?)))
        .collect()
}

/// `SystemOne::Models#parse_list_models_response`: the provider's judgment catalog.
pub async fn list_judgment_models(config: Option<Arc<Config>>) -> Result<Vec<Model>> {
    let config = config.unwrap_or_else(crate::config);
    Provider::TypeSafe.ensure_configured(&config)?;
    let connection = Connection::new(Provider::TypeSafe, config)?;
    let body = connection.get("v1/models", &[]).await?.body;
    let entries = body
        .get("models")
        .and_then(Value::as_array)
        .ok_or_else(|| Error::Api("System One returned an invalid model catalog".into(), None))?;
    entries
        .iter()
        .map(|e| {
            let name = e
                .get("name")
                .and_then(Value::as_str)
                .filter(|n| !n.is_empty())
                .ok_or_else(|| {
                    Error::Api("System One returned a model without a name".into(), None)
                })?;
            let mut model = Model::default_for(name, "typesafe");
            model.name = name.to_string();
            model.created_at = e
                .get("release_date")
                .and_then(Value::as_str)
                .map(str::to_string);
            model.modalities.input = vec!["text".into()];
            model.modalities.output = vec!["judgment".into()];
            model.capabilities = vec!["judgment".into()];
            model.metadata = Map::new();
            if let Some(d) = e.get("description") {
                model.metadata.insert("description".into(), d.clone());
            }
            Ok(model)
        })
        .collect()
}
