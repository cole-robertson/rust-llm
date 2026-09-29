//! Port of `lib/ruby_llm/mcp/input_request.rb` and `input_required_error.rb`.

use serde_json::{Map, Value, json};

use crate::message::ToolCall;

/// A value a form request asks for (`InputRequest::Field`).
#[derive(Debug, Clone, PartialEq)]
pub struct Field {
    pub name: String,
    pub kind: Option<String>,
    pub title: Option<String>,
    pub description: Option<String>,
    /// `required?`: whether the server needs this value.
    pub required: bool,
    pub choices: Option<Vec<Value>>,
    pub default: Option<Value>,
}

/// `RubyLLM::MCP::InputRequest`: a server's request for input from the user while it works on a
/// call. A form request asks for values; a URL request asks the user to visit a page.
#[derive(Debug, Clone, PartialEq)]
pub struct InputRequest {
    pub key: String,
    params: Value,
    /// The answer (`{ action: "accept", content: }`) or `{ action: "decline" }`, once settled.
    pub response: Option<Value>,
    /// The tool call paused on this request, when it came from a chat.
    pub tool_call: Option<ToolCall>,
    /// Why the server needs the input.
    pub message: Option<String>,
    /// The page a URL request asks the user to visit.
    pub url: Option<String>,
    pub fields: Vec<Field>,
}

impl InputRequest {
    pub fn new(key: impl Into<String>, params: Value) -> InputRequest {
        let s = |k: &str| params.get(k).and_then(Value::as_str).map(str::to_string);
        let url = if s("mode").as_deref() == Some("url") { s("url") } else { None };
        let fields = fields_from(params.get("requestedSchema").unwrap_or(&Value::Null));
        InputRequest { key: key.into(), message: s("message"), url, fields, params, response: None, tool_call: None }
    }

    /// `InputRequest.from_h`.
    pub fn from_h(data: &Value, tool_call: Option<ToolCall>) -> InputRequest {
        let key = data.get("key").and_then(Value::as_str).unwrap_or("");
        let mut request = InputRequest::new(key, data.get("params").cloned().unwrap_or_else(|| json!({})));
        request.response = data.get("response").filter(|r| !r.is_null()).cloned();
        request.tool_call = tool_call;
        request
    }

    /// `url?`.
    pub fn is_url(&self) -> bool {
        self.url.is_some()
    }

    /// `form?`.
    pub fn is_form(&self) -> bool {
        !self.is_url()
    }

    /// `answer(**values)`: a form request takes the values; a URL request takes none, meaning
    /// the user agreed to visit the page.
    pub fn answer(&mut self, values: Map<String, Value>) {
        self.response = Some(if self.is_url() {
            json!({ "action": "accept" })
        } else {
            json!({ "action": "accept", "content": values })
        });
    }

    /// `decline`.
    pub fn decline(&mut self) {
        self.response = Some(json!({ "action": "decline" }));
    }

    /// `answered?`: whether the request has been answered or declined.
    pub fn is_answered(&self) -> bool {
        self.response.is_some()
    }

    /// `to_h`.
    pub fn to_h(&self) -> Value {
        let mut h = json!({ "key": self.key, "params": self.params });
        if let Some(response) = &self.response {
            h["response"] = response.clone();
        }
        h
    }
}

fn fields_from(schema: &Value) -> Vec<Field> {
    let required: Vec<&str> = schema.get("required").and_then(Value::as_array).into_iter().flatten().filter_map(Value::as_str).collect();
    let Some(properties) = schema.get("properties").and_then(Value::as_object) else { return Vec::new() };
    properties
        .iter()
        .map(|(name, property)| {
            let s = |k: &str| property.get(k).and_then(Value::as_str).map(str::to_string);
            Field {
                name: name.clone(),
                kind: s("type"),
                title: s("title"),
                description: s("description"),
                required: required.contains(&name.as_str()),
                choices: choices(property),
                default: property.get("default").cloned(),
            }
        })
        .collect()
}

fn choices(property: &Value) -> Option<Vec<Value>> {
    let options: Vec<Value> = match property.get("enum").and_then(Value::as_array) {
        Some(values) => values.clone(),
        None => property
            .get("oneOf")
            .or_else(|| property.pointer("/items/anyOf"))
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .map(|option| option.get("const").cloned().unwrap_or(Value::Null))
            .collect(),
    };
    (!options.is_empty()).then_some(options)
}

/// The requests a paused call waits on, plus the server's opaque state, which is everything
/// needed to answer them later and resume the call.
#[derive(Debug, Clone, PartialEq)]
pub struct InputState {
    pub requests: Vec<InputRequest>,
    pub request_state: Option<Value>,
}

impl InputState {
    /// `InputRequiredError#to_h`: serializes to JSON for persistence.
    pub fn to_h(&self) -> Value {
        let mut h = json!({ "requests": self.requests.iter().map(InputRequest::to_h).collect::<Vec<_>>() });
        if let Some(state) = &self.request_state {
            h["request_state"] = state.clone();
        }
        h
    }

    pub fn from_h(data: &Value) -> InputState {
        let requests = data.get("requests").and_then(Value::as_array).into_iter().flatten().map(|r| InputRequest::from_h(r, None)).collect();
        InputState { requests, request_state: data.get("request_state").filter(|s| !s.is_null()).cloned() }
    }
}

/// `RubyLLM::MCP::InputRequiredError`: a server needs input from the user and no
/// `before_input_request` callback answered. In a chat the tool call pauses instead, and
/// `Chat::pending_inputs` returns the requests. Surfaces as `rust_llm::Error::McpInputRequired`.
#[derive(Debug, Clone)]
pub struct InputRequiredError {
    pub message: String,
    pub input: InputState,
}

impl InputRequiredError {
    pub(crate) fn new(server: &str, input: InputState) -> InputRequiredError {
        let asks: Vec<String> = input
            .requests
            .iter()
            .filter(|r| !r.is_answered())
            .map(|r| [r.message.as_deref(), r.url.as_deref()].into_iter().flatten().collect::<Vec<_>>().join(" "))
            .collect();
        InputRequiredError { message: format!("{server} needs input from the user: {}", asks.join("; ")), input }
    }

    /// `requests`: the unanswered requests.
    pub fn requests(&self) -> Vec<&InputRequest> {
        self.input.requests.iter().filter(|r| !r.is_answered()).collect()
    }
}

impl std::fmt::Display for InputRequiredError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for InputRequiredError {}

impl From<InputRequiredError> for crate::Error {
    fn from(error: InputRequiredError) -> Self {
        crate::Error::McpInputRequired(Box::new(error))
    }
}
