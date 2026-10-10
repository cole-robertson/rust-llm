//! Port of `lib/ruby_llm/evaluation/case.rb`: one dataset example.

use serde_json::{Map, Value};

use crate::error::{Error, Result};

/// `RubyLLM::Evaluation::Case`: one dataset example, with application inputs and optional
/// reference evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct Case {
    name: String,
    inputs: Value,
    /// `Some(Value::Null)` is an explicit `expected_output: nil`, which still counts as supplied.
    expected_output: Option<Value>,
    metadata: Map<String, Value>,
}

impl Case {
    /// `Case.new(name:, inputs:)`. Inputs must be JSON (finite numbers only).
    pub fn new(name: impl Into<String>, inputs: impl Into<Value>) -> Result<Case> {
        let name = name.into();
        if name.is_empty() {
            return Err(Error::Argument("A case name cannot be empty".into()));
        }
        Ok(Case {
            name,
            inputs: inputs.into(),
            expected_output: None,
            metadata: Map::new(),
        })
    }

    /// `expected_output:`: the reference answer. `Value::Null` is an explicit `nil` reference.
    pub fn with_expected_output(mut self, expected_output: impl Into<Value>) -> Case {
        self.expected_output = Some(expected_output.into());
        self
    }

    /// `metadata:`: application-defined reference evidence and category labels.
    pub fn with_metadata(mut self, metadata: Map<String, Value>) -> Case {
        self.metadata = metadata;
        self
    }

    /// A case from one dataset row (`Case.new(**row)`): `name`, `inputs`, and optionally
    /// `expected_output` and `metadata`.
    pub fn from_value(row: &Value) -> Result<Case> {
        let Some(row) = row.as_object() else {
            return Err(Error::Argument(format!(
                "A dataset case must be a Hash, not {row}"
            )));
        };
        let unknown: Vec<&str> = row
            .keys()
            .map(String::as_str)
            .filter(|k| !["name", "inputs", "metadata", "expected_output"].contains(k))
            .collect();
        if !unknown.is_empty() {
            return Err(Error::Argument(format!(
                "Unknown case fields: {}",
                unknown.join(", ")
            )));
        }
        let missing: Vec<&str> = ["name", "inputs"]
            .into_iter()
            .filter(|k| !row.contains_key(*k))
            .collect();
        if !missing.is_empty() {
            return Err(Error::Argument(format!(
                "missing keywords: {}",
                missing.join(", ")
            )));
        }
        let name = match &row["name"] {
            Value::String(s) => s.clone(),
            Value::Null => String::new(),
            other => other.to_string(),
        };
        let mut case = Case::new(name, row["inputs"].clone())?;
        match row.get("metadata") {
            None => {}
            Some(Value::Object(m)) => case.metadata = m.clone(),
            Some(_) => return Err(Error::Argument("Case metadata must be a Hash".into())),
        }
        if let Some(expected) = row.get("expected_output") {
            case.expected_output = Some(expected.clone());
        }
        Ok(case)
    }

    /// The stable case name used to compare runs.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The value passed to `perform`.
    pub fn inputs(&self) -> &Value {
        &self.inputs
    }

    /// The reference answer, or `None` when none was supplied (an explicit `nil` is `Null`).
    pub fn expected_output(&self) -> Option<&Value> {
        self.expected_output.as_ref()
    }

    /// `expected_output?`: whether a reference was supplied, including an explicit `nil`.
    pub fn is_expected_output(&self) -> bool {
        self.expected_output.is_some()
    }

    /// Application-defined reference evidence and category labels.
    pub fn metadata(&self) -> &Map<String, Value> {
        &self.metadata
    }

    /// `Case#to_h`: the portable dataset representation of this case.
    pub fn to_h(&self) -> Value {
        let mut h = Map::new();
        h.insert("name".into(), self.name.clone().into());
        h.insert("inputs".into(), self.inputs.clone());
        h.insert("metadata".into(), Value::Object(self.metadata.clone()));
        if let Some(e) = &self.expected_output {
            h.insert("expected_output".into(), e.clone());
        }
        Value::Object(h)
    }
}
