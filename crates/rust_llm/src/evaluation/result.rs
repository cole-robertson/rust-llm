//! Port of `lib/ruby_llm/evaluation/result.rb`: one criterion's verdict or native decision.

use serde_json::{Value, json};

use super::TrialError;
use crate::error::{Error, Result as CrateResult};
use crate::judge::Answer;

/// What a criterion measured: an LLM verdict (`true`/`false`) or a native judgment answer
/// (`Probability`, `Score`, or `Choice`) with its distribution.
#[derive(Debug, Clone, PartialEq)]
pub enum Measurement {
    Verdict(bool),
    Answer(Answer),
}

impl Measurement {
    fn to_h(&self) -> Value {
        match self {
            Measurement::Verdict(b) => Value::Bool(*b),
            Measurement::Answer(a) => a.to_value(),
        }
    }

    fn inspect(&self) -> String {
        match self {
            Measurement::Verdict(b) => b.to_string(),
            Measurement::Answer(a) => format!("{a:?}"),
        }
    }
}

/// `:passed`, `:failed`, `:measured`, `:unassessed`, or `:error`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Status {
    Passed,
    Failed,
    Measured,
    Unassessed,
    Error,
}

impl Status {
    /// Every status, in the order reports count them.
    pub const ALL: [Status; 5] = [
        Status::Passed,
        Status::Failed,
        Status::Measured,
        Status::Unassessed,
        Status::Error,
    ];

    pub fn as_str(&self) -> &'static str {
        match self {
            Status::Passed => "passed",
            Status::Failed => "failed",
            Status::Measured => "measured",
            Status::Unassessed => "unassessed",
            Status::Error => "error",
        }
    }
}

impl std::fmt::Display for Status {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// `RubyLLM::Evaluation::Result`: one criterion's verdict or native decision, with its
/// acceptance policy.
#[derive(Debug, Clone, PartialEq)]
pub struct Result {
    name: String,
    value: Option<Measurement>,
    reason: Option<String>,
    minimum: Option<f64>,
    error: Option<TrialError>,
    model: Option<String>,
    evidence: Option<Value>,
}

impl Result {
    /// `Result.new(name:, value:, minimum:)`. A minimum applies only to a probability or score.
    pub fn new(
        name: impl Into<String>,
        value: Option<Measurement>,
        minimum: Option<f64>,
    ) -> CrateResult<Result> {
        let numeric = matches!(
            value,
            Some(Measurement::Answer(
                Answer::Probability { .. } | Answer::Score { .. }
            ))
        );
        if minimum.is_some() && !numeric {
            return Err(Error::Argument(
                "Minimum applies only to a probability or score".into(),
            ));
        }
        Ok(Result {
            name: name.into(),
            value,
            reason: None,
            minimum,
            error: None,
            model: None,
            evidence: None,
        })
    }

    /// `Result.new(name:, error:)`: the criterion could not be evaluated.
    pub fn failed(name: impl Into<String>, error: TrialError) -> Result {
        Result {
            name: name.into(),
            value: None,
            reason: None,
            minimum: None,
            error: Some(error),
            model: None,
            evidence: None,
        }
    }

    pub(crate) fn with_details(
        mut self,
        reason: Option<String>,
        model: Option<String>,
        evidence: Option<Value>,
    ) -> Result {
        self.reason = reason;
        self.model = model;
        self.evidence = evidence;
        self
    }

    /// The criterion name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// A boolean verdict or a native answer, without discarding its distribution.
    pub fn value(&self) -> Option<&Measurement> {
        self.value.as_ref()
    }

    /// The evaluator's explanation, when available.
    pub fn reason(&self) -> Option<&str> {
        self.reason.as_deref()
    }

    /// The error when this criterion could not be evaluated.
    pub fn error(&self) -> Option<&TrialError> {
        self.error.as_ref()
    }

    /// The actual evaluator model identifier, when available.
    pub fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    /// The evaluator conversation or native judgment, as portable evidence.
    pub fn evidence(&self) -> Option<&Value> {
        self.evidence.as_ref()
    }

    /// `passed?`.
    pub fn is_passed(&self) -> bool {
        self.status() == Status::Passed
    }

    /// `decision`.
    fn decision(&self) -> Option<bool> {
        if self.error.is_some() {
            return None;
        }
        match &self.value {
            Some(Measurement::Verdict(b)) => return Some(*b),
            None => return None,
            _ => {}
        }
        let minimum = self.minimum?;
        let number = match &self.value {
            Some(Measurement::Answer(Answer::Probability { probability })) => *probability,
            Some(Measurement::Answer(Answer::Score { score, .. })) => *score,
            _ => return None,
        };
        Some(number >= minimum)
    }

    /// `status`.
    pub fn status(&self) -> Status {
        if self.error.is_some() {
            return Status::Error;
        }
        if self.value.is_none() {
            return Status::Unassessed;
        }
        match self.decision() {
            None => Status::Measured,
            Some(true) => Status::Passed,
            Some(false) => Status::Failed,
        }
    }

    /// `to_h`: the measurement, policy, and outcome.
    pub fn to_h(&self) -> Value {
        json!({
            "name": self.name,
            "status": self.status().as_str(),
            "value": self.value.as_ref().map_or(Value::Null, Measurement::to_h),
            "reason": self.reason,
            "minimum": self.minimum,
            "model": self.model,
            "evidence": self.evidence,
            "error": self.error.as_ref().map(TrialError::to_h),
        })
    }

    /// What a failure report says about this criterion (`error.message || reason || value.inspect`).
    pub(crate) fn failure_message(&self) -> String {
        if let Some(e) = &self.error {
            return e.message.clone();
        }
        if let Some(r) = &self.reason {
            return r.clone();
        }
        self.value
            .as_ref()
            .map_or_else(|| "nil".to_string(), Measurement::inspect)
    }
}
