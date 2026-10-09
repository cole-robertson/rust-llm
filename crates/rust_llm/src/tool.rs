//! Port of `lib/ruby_llm/tool.rb`.
//!
//! RubyLLM declares tools as subclasses:
//!
//! ```ruby
//! class Weather < RubyLLM::Tool
//!   description "Gets current weather for a location"
//!   parameter :latitude, description: "Latitude (e.g., 52.5200)"
//!   def execute(latitude:, longitude:) = ...
//! end
//! ```
//!
//! Here a tool is a type implementing [`Tool`]. Arguments are either declared with
//! [`Parameter`]s (the `parameter` DSL) or derived from a `schemars::JsonSchema` struct via
//! [`Tool::parameters_schema`], the equivalent of `parameters do ... end`.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Map, Value, json};

use crate::attachment::Attachment;
use crate::message::ToolCall;

/// `RubyLLM::Parameter`.
#[derive(Debug, Clone)]
pub struct Parameter {
    pub name: String,
    pub kind: String,
    pub description: Option<String>,
    pub required: bool,
}

impl Parameter {
    /// A required string parameter, the Ruby default.
    pub fn new(name: impl Into<String>) -> Parameter {
        Parameter {
            name: name.into(),
            kind: "string".into(),
            description: None,
            required: true,
        }
    }

    pub fn description(mut self, text: impl Into<String>) -> Parameter {
        self.description = Some(text.into());
        self
    }

    pub fn kind(mut self, kind: impl Into<String>) -> Parameter {
        self.kind = kind.into();
        self
    }

    pub fn optional(mut self) -> Parameter {
        self.required = false;
        self
    }
}

/// What a tool returns: text/JSON content, optionally with attachments (`Tool.split_result`).
#[derive(Debug, Clone, PartialEq)]
pub struct ToolResult {
    pub content: String,
    pub attachments: Vec<Attachment>,
}

impl ToolResult {
    pub fn with_attachments(
        content: impl Into<String>,
        attachments: Vec<Attachment>,
    ) -> ToolResult {
        ToolResult {
            content: content.into(),
            attachments,
        }
    }

    /// A structured error the model can read and recover from, like `{ error: "..." }` in Ruby.
    pub fn error(message: impl Into<String>) -> ToolResult {
        ToolResult::from(json!({ "error": message.into() }))
    }
}

impl From<String> for ToolResult {
    fn from(content: String) -> Self {
        ToolResult {
            content,
            attachments: Vec::new(),
        }
    }
}

impl From<&str> for ToolResult {
    fn from(content: &str) -> Self {
        ToolResult::from(content.to_string())
    }
}

/// `split_result(attachment)`: a lone attachment goes back with empty text.
impl From<Attachment> for ToolResult {
    fn from(attachment: Attachment) -> Self {
        ToolResult::with_attachments("", vec![attachment])
    }
}

/// Hashes and arrays go to the model as JSON, like `result.to_json`.
impl From<Value> for ToolResult {
    fn from(value: Value) -> Self {
        match value {
            Value::String(s) => ToolResult::from(s),
            other => ToolResult::from(other.to_string()),
        }
    }
}

/// Error type tools return. It escapes `ask` like an exception raised in `execute`.
pub type ToolError = Box<dyn std::error::Error + Send + Sync>;

#[async_trait]
pub trait Tool: Send + Sync {
    /// `Tool.description`.
    fn description(&self) -> String;

    /// `Tool.tool_name`. Defaults to the type name, underscored, with a trailing `_tool` removed.
    fn name(&self) -> String {
        tool_name_from_type(std::any::type_name::<Self>())
    }

    /// `Tool.parameter` declarations.
    fn parameters(&self) -> Vec<Parameter> {
        Vec::new()
    }

    /// `Tool.parameters(schema)`: a full JSON schema, which wins over `parameters()`.
    fn parameters_schema(&self) -> Option<Value> {
        None
    }

    /// `Tool.requires_approval`: park the conversation until `approve`/`deny`.
    fn requires_approval(&self) -> bool {
        false
    }

    /// `requires_approval { |tool_call| ... }`: decide a call yourself instead of using the
    /// recorded `approve`/`deny`. `Some(true)` executes, `Some(false)` denies, `None` keeps it
    /// pending. Only consulted when `requires_approval()` is true, and possibly several times per
    /// call, so make it an idempotent read. The default defers to the recorded decision.
    fn approval(&self, _tool_call: &ToolCall) -> Option<Option<bool>> {
        None
    }

    /// `Tool.provider_options`: deep-merged into this tool's provider definition.
    fn provider_options(&self) -> Map<String, Value> {
        Map::new()
    }

    /// `Tool.defer` / `Tool.deferred?`: keeps this tool's definition out of the model's context
    /// until the provider's tool search loads it. Pass `Some(false)` to
    /// `Chat::add_tool_deferred` to offer it up front in one chat. See `Chat::deferred_tools`.
    fn is_deferred(&self) -> bool {
        false
    }

    /// Whether this is a [`Deferred`] registration, the tool set a chat renders for a request.
    /// Protocols with native tool search mark these `defer_loading`.
    #[doc(hidden)]
    fn is_deferred_registration(&self) -> bool {
        false
    }

    async fn execute(
        &self,
        arguments: Map<String, Value>,
        tool_call: &ToolCall,
    ) -> Result<ToolResult, ToolError>;

    /// `resume(input, arguments)`: continues a call that paused on input requests (MCP
    /// elicitation), now answered. `input` is the paused state `InputRequiredError#to_h` gave.
    /// Tools that never pause run again.
    async fn resume(
        &self,
        _input: &Value,
        arguments: Map<String, Value>,
        tool_call: &ToolCall,
    ) -> Result<ToolResult, ToolError> {
        self.execute(arguments, tool_call).await
    }
}

/// `Tool.tool_name`: `WeatherLookupTool` -> `weather_lookup`.
pub fn tool_name_from_type(type_name: &str) -> String {
    let base = type_name.rsplit("::").next().unwrap_or(type_name);
    let base = base.split('<').next().unwrap_or(base);
    // `unicode_normalize(:nfkd).encode('ASCII', replace: '')`: accents decompose and drop, other
    // non-ASCII characters drop, then anything outside `[a-zA-Z0-9_-]` becomes `-`.
    use unicode_normalization::UnicodeNormalization;
    let ascii: String = base
        .nfkd()
        .filter(char::is_ascii)
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '-'
            }
        })
        .collect();
    let name = underscore(&ascii);
    name.strip_suffix("_tool").unwrap_or(&name).to_string()
}

/// Acronym-aware underscoring: `HTTPProxyTool` -> `http_proxy_tool`.
pub fn underscore(name: &str) -> String {
    let chars: Vec<char> = name.chars().collect();
    let mut out = String::new();
    for (i, &c) in chars.iter().enumerate() {
        if i > 0 && c.is_ascii_uppercase() {
            let prev = chars[i - 1];
            let next_lower = chars.get(i + 1).is_some_and(|n| n.is_ascii_lowercase());
            if prev.is_ascii_lowercase()
                || prev.is_ascii_digit()
                || (prev.is_ascii_uppercase() && next_lower)
            {
                out.push('_');
            }
        }
        out.push(c.to_ascii_lowercase());
    }
    out
}

fn map_type(kind: &str) -> &'static str {
    match kind {
        "integer" | "int" => "integer",
        "number" | "float" | "double" => "number",
        "boolean" => "boolean",
        "array" => "array",
        "object" => "object",
        _ => "string",
    }
}

/// `SchemaDefinition.from_parameters`.
pub fn schema_from_parameters(parameters: &[Parameter]) -> Option<Value> {
    if parameters.is_empty() {
        return None;
    }
    let mut properties = Map::new();
    for p in parameters {
        let mut schema = Map::new();
        let kind = map_type(&p.kind);
        schema.insert("type".into(), kind.into());
        if let Some(d) = &p.description {
            schema.insert("description".into(), d.clone().into());
        }
        if kind == "array" {
            schema.insert("items".into(), json!({ "type": "string" }));
        }
        properties.insert(p.name.clone(), Value::Object(schema));
    }
    let required: Vec<Value> = parameters
        .iter()
        .filter(|p| p.required)
        .map(|p| Value::String(p.name.clone()))
        .collect();
    Some(json!({
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": false,
        "strict": true
    }))
}

/// `Tool#parameters_schema`: an explicit schema, else declared parameters, else the empty object
/// schema RubyLLM infers from an `execute` that takes no keywords.
pub(crate) fn tool_schema(tool: &dyn Tool) -> Option<Value> {
    if let Some(mut s) = tool.parameters_schema() {
        if let Some(obj) = s.as_object_mut() {
            obj.remove("$schema");
            obj.remove("title");
        }
        return Some(s);
    }
    let params = tool.parameters();
    if params.is_empty() {
        return Some(json!({
            "type": "object", "properties": {}, "required": [], "additionalProperties": false, "strict": true
        }));
    }
    schema_from_parameters(&params)
}

/// Builds a JSON schema from a `schemars` type, for `Tool::parameters_schema` and `with_schema`.
/// Like RubyLLM's schema DSL (Schematist), every object is closed with
/// `additionalProperties: false`, which strict structured-output modes require.
pub fn schema_for<T: schemars::JsonSchema>() -> Value {
    let mut schema = serde_json::to_value(schemars::schema_for!(T)).unwrap_or(Value::Null);
    if let Some(obj) = schema.as_object_mut() {
        obj.remove("$schema");
        obj.remove("title");
        // A type's doc comment documents Rust code, not the model's output.
        obj.remove("description");
    }
    close_objects(&mut schema);
    schema
}

/// Numeric `format`s schemars adds from Rust types; RubyLLM's DSL emits plain `number`/`integer`.
const RUST_NUMBER_FORMATS: &[&str] = &[
    "double", "float", "int8", "int16", "int32", "int64", "uint", "uint8", "uint16", "uint32",
    "uint64",
];

fn close_objects(node: &mut Value) {
    match node {
        Value::Object(map) => {
            if map.get("type").and_then(Value::as_str) == Some("object")
                && map.contains_key("properties")
            {
                map.entry("additionalProperties")
                    .or_insert(Value::Bool(false));
            }
            if map
                .get("format")
                .and_then(Value::as_str)
                .is_some_and(|f| RUST_NUMBER_FORMATS.contains(&f))
            {
                map.remove("format");
            }
            if map.get("minimum").and_then(Value::as_i64) == Some(0)
                && map.get("type").and_then(Value::as_str) == Some("integer")
            {
                map.remove("minimum");
            }
            map.values_mut().for_each(close_objects);
        }
        Value::Array(items) => items.iter_mut().for_each(close_objects),
        _ => {}
    }
}

/// Rejects arguments `execute(**kwargs)` would: missing required keys and unknown keys.
pub(crate) fn validate_arguments(
    tool: &dyn Tool,
    arguments: &Map<String, Value>,
) -> Option<String> {
    // A tool with no declared parameters renders the empty object schema, like Ruby's `execute`
    // without keywords, so any argument is unknown there too.
    let params = tool.parameters();
    if tool.parameters_schema().is_some() {
        return None;
    }
    if let Some(missing) = params
        .iter()
        .find(|p| p.required && !arguments.contains_key(&p.name))
    {
        return Some(format!("missing keyword: {}", missing.name));
    }
    if let Some(unknown) = arguments
        .keys()
        .find(|k| !params.iter().any(|p| &p.name == *k))
    {
        return Some(format!("unknown keyword: {unknown}"));
    }
    None
}

pub type SharedTool = Arc<dyn Tool>;

/// `Tool::Deferred`: a tool the chat registered as deferred, delegating everything to it. Only
/// protocols with tool search (Anthropic, OpenAI Responses) treat it differently.
pub struct Deferred(pub SharedTool);

#[async_trait]
impl Tool for Deferred {
    fn description(&self) -> String {
        self.0.description()
    }
    fn name(&self) -> String {
        self.0.name()
    }
    fn parameters(&self) -> Vec<Parameter> {
        self.0.parameters()
    }
    fn parameters_schema(&self) -> Option<Value> {
        self.0.parameters_schema()
    }
    fn requires_approval(&self) -> bool {
        self.0.requires_approval()
    }
    fn approval(&self, tool_call: &ToolCall) -> Option<Option<bool>> {
        self.0.approval(tool_call)
    }
    fn provider_options(&self) -> Map<String, Value> {
        self.0.provider_options()
    }
    fn is_deferred(&self) -> bool {
        self.0.is_deferred()
    }
    fn is_deferred_registration(&self) -> bool {
        true
    }
    async fn execute(
        &self,
        arguments: Map<String, Value>,
        tool_call: &ToolCall,
    ) -> Result<ToolResult, ToolError> {
        self.0.execute(arguments, tool_call).await
    }
    async fn resume(
        &self,
        input: &Value,
        arguments: Map<String, Value>,
        tool_call: &ToolCall,
    ) -> Result<ToolResult, ToolError> {
        self.0.resume(input, arguments, tool_call).await
    }
}

type ToolFnFuture =
    std::pin::Pin<Box<dyn std::future::Future<Output = Result<ToolResult, ToolError>> + Send>>;

/// A tool defined inline from a closure, for one-off tools that don't warrant a type.
pub struct FnTool {
    name: String,
    description: String,
    parameters: Vec<Parameter>,
    approval: bool,
    run: Box<dyn Fn(Map<String, Value>) -> ToolFnFuture + Send + Sync>,
}

impl FnTool {
    pub fn new<F, Fut>(name: impl Into<String>, description: impl Into<String>, run: F) -> FnTool
    where
        F: Fn(Map<String, Value>) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<ToolResult, ToolError>> + Send + 'static,
    {
        FnTool {
            name: name.into(),
            description: description.into(),
            parameters: Vec::new(),
            approval: false,
            run: Box::new(move |args| Box::pin(run(args))),
        }
    }

    pub fn parameter(mut self, parameter: Parameter) -> FnTool {
        self.parameters.push(parameter);
        self
    }

    pub fn requires_approval(mut self) -> FnTool {
        self.approval = true;
        self
    }
}

#[async_trait]
impl Tool for FnTool {
    fn name(&self) -> String {
        self.name.clone()
    }
    fn description(&self) -> String {
        self.description.clone()
    }
    fn parameters(&self) -> Vec<Parameter> {
        self.parameters.clone()
    }
    fn requires_approval(&self) -> bool {
        self.approval
    }
    async fn execute(
        &self,
        arguments: Map<String, Value>,
        _call: &ToolCall,
    ) -> Result<ToolResult, ToolError> {
        (self.run)(arguments).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_names_follow_rust_llm_underscoring() {
        assert_eq!(
            tool_name_from_type("app::tools::WeatherLookup"),
            "weather_lookup"
        );
        assert_eq!(tool_name_from_type("HTTPProxyTool"), "http_proxy");
        assert_eq!(
            tool_name_from_type("BestLanguageToLearn"),
            "best_language_to_learn"
        );
    }

    // Matches the "tools" payload recorded in chat_function_calling_*_can_use_tools cassettes.
    #[test]
    fn declared_parameters_render_the_upstream_schema() {
        let schema = schema_from_parameters(&[
            Parameter::new("latitude").description("Latitude (e.g., 52.5200)"),
            Parameter::new("longitude").description("Longitude (e.g., 13.4050)"),
        ])
        .unwrap();
        assert_eq!(
            schema.to_string(),
            r#"{"type":"object","properties":{"latitude":{"type":"string","description":"Latitude (e.g., 52.5200)"},"longitude":{"type":"string","description":"Longitude (e.g., 13.4050)"}},"required":["latitude","longitude"],"additionalProperties":false,"strict":true}"#
        );
    }
}
