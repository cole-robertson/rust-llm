//! Port of `lib/ruby_llm/evaluation/assertions.rb`: the assertions an evaluation runs after
//! `perform`. Ruby borrows `Minitest::Assertions`; here they are methods on [`Assertions`] that
//! return [`AssertionFailure`], so `?` stops at the first failed one, as a raised
//! `Minitest::Assertion` does. Counts follow Minitest's (`refute_empty` is two assertions).

use serde_json::Value;

use super::Outcome;
use crate::message::{Message, ToolCall};

/// `Minitest::Assertion`: a failed assertion, recorded as the trial's `assertion_failure`.
#[derive(Debug, Clone, PartialEq)]
pub struct AssertionFailure {
    message: String,
}

impl AssertionFailure {
    pub fn new(message: impl Into<String>) -> AssertionFailure {
        AssertionFailure {
            message: message.into(),
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl std::fmt::Display for AssertionFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

/// An error raised by application code (`StandardError`): its class and message, as reports keep
/// them (`{ class:, message: }`).
#[derive(Debug, Clone, PartialEq)]
pub struct TrialError {
    pub class: String,
    pub message: String,
}

impl TrialError {
    pub fn new(class: impl Into<String>, message: impl Into<String>) -> TrialError {
        TrialError {
            class: class.into(),
            message: message.into(),
        }
    }

    /// A failure with only a message (`raise "task failed"`, a `RuntimeError`).
    pub fn runtime(message: impl Into<String>) -> TrialError {
        TrialError::new("RuntimeError", message)
    }

    pub(crate) fn from_error(error: &crate::Error) -> TrialError {
        TrialError::new(error.class_name(), error.to_string())
    }

    pub fn to_h(&self) -> Value {
        serde_json::json!({ "class": self.class, "message": self.message })
    }
}

impl std::fmt::Display for TrialError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

/// What an evaluation hook can fail with. `Assertions::Failure` matches only assertion
/// failures; every other error is an execution error. Both convert with `?`.
#[derive(Debug, Clone, PartialEq)]
pub enum Failure {
    Assertion(AssertionFailure),
    Error(TrialError),
}

impl Failure {
    /// `raise "message"`.
    pub fn error(message: impl Into<String>) -> Failure {
        Failure::Error(TrialError::runtime(message))
    }
}

impl From<AssertionFailure> for Failure {
    fn from(failure: AssertionFailure) -> Failure {
        Failure::Assertion(failure)
    }
}

impl<E: std::error::Error + 'static> From<E> for Failure {
    fn from(error: E) -> Failure {
        let any: &dyn std::any::Any = &error;
        match any.downcast_ref::<crate::Error>() {
            Some(e) => Failure::Error(TrialError::from_error(e)),
            None => Failure::Error(TrialError::new(
                std::any::type_name::<E>(),
                error.to_string(),
            )),
        }
    }
}

/// The current case and returned evidence, with the Minitest assertion family. Every assertion
/// adds to the trial's `assertion_count`.
pub struct Assertions<'a> {
    pub(crate) input: &'a Value,
    pub(crate) expected_output: Option<&'a Value>,
    pub(crate) metadata: &'a serde_json::Map<String, Value>,
    pub(crate) result: &'a Outcome,
    pub(crate) output: &'a Value,
    pub(crate) messages: &'a [Message],
    pub(crate) tool_calls: &'a [ToolCall],
    pub(crate) count: usize,
}

type Check = std::result::Result<(), AssertionFailure>;

fn inspect(value: &Value) -> String {
    match value {
        Value::Null => "nil".into(),
        other => other.to_string(),
    }
}

impl<'a> Assertions<'a> {
    /// `input`: the case's inputs (a fresh copy).
    pub fn input(&self) -> &'a Value {
        self.input
    }

    /// `expected_output`: the reference answer, or `null` when none was supplied.
    pub fn expected_output(&self) -> &'a Value {
        self.expected_output.unwrap_or(&Value::Null)
    }

    pub fn metadata(&self) -> &'a serde_json::Map<String, Value> {
        self.metadata
    }

    /// `result`: the value `perform` returned.
    pub fn result(&self) -> &'a Outcome {
        self.result
    }

    /// `output`: the primary answer or value extracted from `result`.
    pub fn output(&self) -> &'a Value {
        self.output
    }

    /// `messages`: retained messages from a returned chat or message.
    pub fn messages(&self) -> &'a [Message] {
        self.messages
    }

    /// `tool_calls`: tool calls recorded in the returned conversation or message.
    pub fn tool_calls(&self) -> &'a [ToolCall] {
        self.tool_calls
    }

    /// `assertions`: how many have run.
    pub fn count(&self) -> usize {
        self.count
    }

    fn check(&mut self, ok: bool, message: impl FnOnce() -> String) -> Check {
        self.count += 1;
        if ok {
            Ok(())
        } else {
            Err(AssertionFailure::new(message()))
        }
    }

    /// `assert(condition, message)`.
    pub fn assert(&mut self, condition: bool, message: Option<&str>) -> Check {
        let message = message.unwrap_or("Expected false to be truthy.").to_string();
        self.check(condition, || message)
    }

    /// `refute(condition, message)`.
    pub fn refute(&mut self, condition: bool, message: Option<&str>) -> Check {
        let message = message.unwrap_or("Expected true to not be truthy.").to_string();
        self.check(!condition, || message)
    }

    /// `assert_equal(expected, actual)`.
    pub fn assert_equal(&mut self, expected: impl Into<Value>, actual: impl Into<Value>) -> Check {
        let (expected, actual) = (expected.into(), actual.into());
        self.check(expected == actual, || {
            format!(
                "Expected: {}\n  Actual: {}",
                inspect(&expected),
                inspect(&actual)
            )
        })
    }

    /// `assert_nil(value)`.
    pub fn assert_nil(&mut self, value: &Value) -> Check {
        self.check(value.is_null(), || {
            format!("Expected {} to be nil.", inspect(value))
        })
    }

    /// `refute_empty(collection)`: a string, array, or object with something in it.
    pub fn refute_empty(&mut self, value: &Value) -> Check {
        let (responds, empty) = match value {
            Value::String(s) => (true, s.is_empty()),
            Value::Array(a) => (true, a.is_empty()),
            Value::Object(o) => (true, o.is_empty()),
            _ => (false, false),
        };
        self.check(responds, || {
            format!("Expected {} to respond to #empty?.", inspect(value))
        })?;
        self.check(!empty, || format!("Expected {} to not be empty.", inspect(value)))
    }

    /// `assert_includes(collection, item)`.
    pub fn assert_includes<T: PartialEq + std::fmt::Debug>(
        &mut self,
        collection: &[T],
        item: &T,
    ) -> Check {
        self.count += 1;
        self.check(collection.contains(item), || {
            format!("Expected {collection:?} to include {item:?}.")
        })
    }

    /// `assert_kind_of(String, value)` for JSON: `"string"`, `"number"`, `"boolean"`, `"array"`,
    /// `"object"`, or `"null"`.
    pub fn assert_kind_of(&mut self, kind: &str, value: &Value) -> Check {
        let actual = match value {
            Value::Null => "null",
            Value::Bool(_) => "boolean",
            Value::Number(_) => "number",
            Value::String(_) => "string",
            Value::Array(_) => "array",
            Value::Object(_) => "object",
        };
        self.check(actual == kind, || {
            format!("Expected {} to be a kind of {kind}, not {actual}.", inspect(value))
        })
    }
}
