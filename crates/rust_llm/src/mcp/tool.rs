//! Port of `lib/ruby_llm/mcp/tool.rb`: a tool offered by an MCP server. It works anywhere a
//! [`Tool`] does: a chat renders its name, description, and schema for the model, and calling it
//! calls the server.
//!
//! The behavior predicates read the server's annotations. They are hints from the server, so
//! trust them only as far as you trust the server.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Map, Value, json};

use super::{InputState, Mcp, McpResult, Task, apps};
use crate::error::{Error, Result};
use crate::message::ToolCall;
use crate::tool::{Tool, ToolError, ToolResult};

pub(crate) type FixedArgument = Arc<dyn Fn() -> Value + Send + Sync>;
pub(crate) type Wrap = Arc<dyn Fn(&McpResult, &Map<String, Value>) -> ToolResult + Send + Sync>;

/// How `MCP.tool :name, as:, description:, fixed_arguments:, wrap:` shapes a server tool.
#[derive(Clone, Default)]
pub struct ToolShape {
    pub(crate) as_name: Option<String>,
    pub(crate) description: Option<String>,
    pub(crate) fixed_arguments: Vec<(String, FixedArgument)>,
    pub(crate) wrap: Option<Wrap>,
}

impl ToolShape {
    pub fn new() -> ToolShape {
        ToolShape::default()
    }

    /// `as:`: the name the model calls the tool by. Renamed tools skip the MCP's `prefix`.
    pub fn as_name(mut self, name: impl Into<String>) -> ToolShape {
        self.as_name = Some(name.into());
        self
    }

    /// `description:`: rewrites what the model reads.
    pub fn description(mut self, text: impl Into<String>) -> ToolShape {
        self.description = Some(text.into());
        self
    }

    /// `fixed_arguments: { name => value }`: removes the argument from the model's view and
    /// always sends `value`.
    pub fn fixed_argument(self, name: impl Into<String>, value: Value) -> ToolShape {
        self.fixed_argument_with(name, move || value.clone())
    }

    /// `fixed_arguments: { name => -> { ... } }`: the value is computed on every call.
    pub fn fixed_argument_with(
        mut self,
        name: impl Into<String>,
        value: impl Fn() -> Value + Send + Sync + 'static,
    ) -> ToolShape {
        self.fixed_arguments.push((name.into(), Arc::new(value)));
        self
    }

    /// `wrap:`: receives the server's result and the model's arguments and returns what the
    /// model sees.
    pub fn wrap<R: Into<ToolResult>>(
        mut self,
        wrap: impl Fn(&McpResult, &Map<String, Value>) -> R + Send + Sync + 'static,
    ) -> ToolShape {
        self.wrap = Some(Arc::new(move |result, arguments| {
            wrap(result, arguments).into()
        }));
        self
    }

    pub(crate) fn merge(mut self, other: &ToolShape) -> ToolShape {
        if other.as_name.is_some() {
            self.as_name = other.as_name.clone();
        }
        if other.description.is_some() {
            self.description = other.description.clone();
        }
        self.fixed_arguments
            .extend(other.fixed_arguments.iter().cloned());
        if other.wrap.is_some() {
            self.wrap = other.wrap.clone();
        }
        self
    }
}

/// `RubyLLM::MCP::Tool`.
pub struct McpTool {
    mcp: Mcp,
    /// The name the model calls this tool by.
    pub name: String,
    /// The description the model sees.
    pub description: Option<String>,
    /// The tool's name on the server.
    pub server_name: String,
    /// The JSON Schema for the arguments the model provides.
    pub parameters_schema: Value,
    /// The tool's `_meta` as the server sent it, empty when there is none. Extensions keep
    /// their own vocabulary there.
    pub meta: Map<String, Value>,
    /// The URI of the tool's UI, the `ui://` resource an MCP App renders next to the tool's
    /// results, or `None` for a tool without one. Read it with `Mcp::resource`.
    pub ui_uri: Option<String>,
    /// Who may call the tool: `"model"` when chats offer it to the model, `"app"` when a UI
    /// from the same server may call it. Tools say nothing about it unless they belong to an MCP
    /// App, which makes them `["model", "app"]`.
    pub visibility: Vec<String>,
    pub(crate) fixed_arguments: Vec<(String, FixedArgument)>,
    pub(crate) wrap: Option<Wrap>,
    annotations: Value,
}

impl std::fmt::Debug for McpTool {
    /// `#<RubyLLM::MCP::Tool name: "repeat", from: "echo", read_only: true>`
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut d = f.debug_struct("McpTool");
        d.field("name", &self.name);
        if self.server_name != self.name {
            d.field("from", &self.server_name);
        }
        if self.is_read_only() {
            d.field("read_only", &true);
        }
        d.finish()
    }
}

impl McpTool {
    /// `MCP::Tool.new(mcp, definition, prefix:, **shape)`.
    pub fn new(mcp: Mcp, definition: &Value, prefix: Option<&str>, shape: ToolShape) -> McpTool {
        let server_name = definition
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let name = shape.as_name.clone().unwrap_or_else(|| match prefix {
            Some(prefix) => format!("{prefix}_{server_name}"),
            None => server_name.clone(),
        });
        let description = shape.description.clone().or_else(|| {
            definition
                .get("description")
                .and_then(Value::as_str)
                .map(str::to_string)
        });
        let schema = definition
            .get("inputSchema")
            .cloned()
            .unwrap_or_else(|| json!({}));
        let parameters_schema = model_schema(schema, &shape.fixed_arguments);
        let meta = definition
            .get("_meta")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        McpTool {
            mcp,
            name,
            description,
            server_name,
            parameters_schema,
            ui_uri: apps::uri(&meta),
            visibility: apps::visibility(&meta),
            meta,
            fixed_arguments: shape.fixed_arguments,
            wrap: shape.wrap,
            annotations: definition
                .get("annotations")
                .cloned()
                .unwrap_or_else(|| json!({})),
        }
    }

    fn hint(&self, key: &str) -> Option<bool> {
        self.annotations.get(key).and_then(Value::as_bool)
    }

    /// `read_only?`: whether the server says the tool only reads.
    pub fn is_read_only(&self) -> bool {
        self.hint("readOnlyHint") == Some(true)
    }

    /// `destructive?`: tools that are not read-only count as destructive unless the server says
    /// otherwise.
    pub fn is_destructive(&self) -> bool {
        !self.is_read_only() && self.hint("destructiveHint") != Some(false)
    }

    /// `idempotent?`.
    pub fn is_idempotent(&self) -> bool {
        self.hint("idempotentHint") == Some(true)
    }

    /// `open_world?`: whether the tool reaches beyond the server, such as the web.
    pub fn is_open_world(&self) -> bool {
        self.hint("openWorldHint") != Some(false)
    }

    /// `call(**arguments)`: calls the tool on the server and returns what the model sees: the
    /// result's content (with the result kept as `ToolResult::mcp_result`), what the `wrap`
    /// made of it, or `{ error: }` when the tool failed. When the server runs the call as a
    /// task, fails with `Error::McpTask` carrying the task, without waiting. Fails with
    /// `Error::McpInputRequired` when the server needs input no callback gave.
    pub async fn call(&self, arguments: Value) -> Result<ToolResult> {
        let arguments = match arguments {
            Value::Object(map) => map,
            _ => Map::new(),
        };
        self.mcp.run(self, arguments, None).await
    }

    /// `task(state, tool_call:)`: the task a call paused on, from its saved state.
    pub fn task(&self, state: &Value, tool_call: Option<ToolCall>) -> Task {
        Task::load(Some(self.mcp.clone()), state, tool_call)
    }
}

/// `model_schema`: fixed arguments disappear from what the model sees.
fn model_schema(mut schema: Value, fixed: &[(String, FixedArgument)]) -> Value {
    if let Some(obj) = schema.as_object_mut() {
        obj.remove("$schema");
        obj.remove("title");
    }
    if fixed.is_empty() || schema.get("properties").is_none() {
        return schema;
    }
    let hidden: Vec<&str> = fixed.iter().map(|(name, _)| name.as_str()).collect();
    if let Some(Value::Object(properties)) = schema.get_mut("properties") {
        properties.retain(|name, _| !hidden.contains(&name.as_str()));
    }
    let required: Vec<Value> = schema
        .get("required")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|name| !name.as_str().is_some_and(|n| hidden.contains(&n)))
        .cloned()
        .collect();
    schema["required"] = Value::Array(required);
    schema
}

fn boxed(error: Error) -> ToolError {
    Box::new(error)
}

#[async_trait]
impl Tool for McpTool {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn description(&self) -> String {
        self.description.clone().unwrap_or_default()
    }

    fn parameters_schema(&self) -> Option<Value> {
        Some(self.parameters_schema.clone())
    }

    /// `requires_approval?`, as declared with `McpBuilder::requires_approval`.
    fn requires_approval(&self) -> bool {
        self.mcp.requires_approval(self)
    }

    fn is_model_visible(&self) -> bool {
        self.visibility.iter().any(|v| v == "model")
    }

    fn mcp_task(&self, state: &Value, tool_call: &ToolCall) -> Option<Task> {
        Some(self.task(state, Some(tool_call.clone())))
    }

    async fn execute(
        &self,
        arguments: Map<String, Value>,
        _tool_call: &ToolCall,
    ) -> Result<ToolResult, ToolError> {
        self.mcp.run(self, arguments, None).await.map_err(boxed)
    }

    /// `resume(input, arguments)`: resumes a call that paused on input requests, now answered,
    /// or on a task, which it checks on once.
    async fn resume(
        &self,
        input: &Value,
        arguments: Map<String, Value>,
        _tool_call: &ToolCall,
    ) -> Result<ToolResult, ToolError> {
        self.mcp
            .run(self, arguments, Some(InputState::from_h(input)))
            .await
            .map_err(boxed)
    }
}
