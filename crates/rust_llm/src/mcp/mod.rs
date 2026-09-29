//! Port of `lib/ruby_llm/mcp.rb`: a client for a Model Context Protocol server.
//!
//! RubyLLM describes a server in a subclass and hands an instance to a chat:
//!
//! ```ruby
//! class Files < RubyLLM::MCP
//!   command "npx", "-y", "@modelcontextprotocol/server-filesystem", "."
//!   requires_approval if: :destructive?
//! end
//! chat.with_mcp(Files.new).ask "What's in the README?"
//! ```
//!
//! Here the class DSL is a builder, and settings Ruby evaluates on the instance (blocks reading
//! declared `inputs`) are closures that capture what they need:
//!
//! ```no_run
//! # use rust_llm::Mcp;
//! # async fn run() -> rust_llm::Result<()> {
//! # let chat = rust_llm::chat()?;
//! let files = Mcp::command(["npx", "-y", "@modelcontextprotocol/server-filesystem", "."])
//!     .requires_approval_if(&[] as &[&str], |tool| tool.is_destructive())
//!     .build()?;
//! chat.with_mcp(files).ask("What's in the README?").await?;
//! # Ok(()) }
//! ```
//!
//! A `url` connects over Streamable HTTP ([`Http`]), a `command` starts a local server that speaks
//! over stdio ([`Stdio`]), and a [`Transport`] carries the messages any other way.

mod client;
mod collection;
mod content;
mod error;
mod http;
mod input_request;
mod param_headers;
mod prompt;
mod resource;
mod resource_template;
mod result;
mod stdio;
mod tool;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{Map, Value, json};

pub use client::{Client, LEGACY_VERSION, VERSION, client_info};
pub use collection::Collection;
pub use error::McpError;
pub use http::{HeaderSource, Http};
pub use input_request::{Field, InputRequest, InputRequiredError, InputState};
pub use prompt::{Prompt, PromptArgument};
pub use resource::{Resource, ResourceContent};
pub use resource_template::ResourceTemplate;
pub use result::McpResult;
pub use stdio::Stdio;
pub use tool::{McpTool, ToolShape};

use crate::config::Config;
use crate::error::{Error, Result};
use crate::message::{Message, Role};
use crate::progress::{self, Progress};
use crate::tool::{SharedTool, ToolResult};

const INPUT_ROUNDS: usize = 10;

/// Receives the notifications a server sends while it works on a request. An alias so
/// `async_trait` keeps the `&Value` argument higher-ranked.
pub type OnNotification<'a> = dyn FnMut(&Value) + Send + 'a;

/// Carries an MCP server's JSON-RPC messages (`MCP.transport`). `request` sends a request and
/// returns the response, passing notifications the server sends meanwhile to `on_notification`;
/// `timeout` is `None` unless a shorter one than the transport's own is needed, and `headers`
/// holds the tool arguments the server asks to receive as `Mcp-Param-*` headers. `notify` and
/// `cancel` send a notification, and `close` releases the connection until the next request.
/// Fail with `Error::Mcp` when the server cannot be reached.
#[async_trait]
pub trait Transport: Send + Sync {
    async fn request(
        &self,
        message: &Value,
        version: Option<&str>,
        timeout: Option<Duration>,
        headers: &[(String, String)],
        on_notification: &mut OnNotification<'_>,
    ) -> Result<Value>;
    async fn notify(&self, message: &Value, version: Option<&str>) -> Result<()>;
    async fn cancel(&self, notification: &Value, version: Option<&str>) -> Result<()>;
    async fn close(&self);
}

type StringSource = Arc<dyn Fn() -> Option<String> + Send + Sync>;
type ApprovalCondition = Arc<dyn Fn(&McpTool) -> bool + Send + Sync>;
type ProgressCallback = Arc<dyn Fn(&Progress) + Send + Sync>;
type InputCallback = Arc<dyn Fn(&mut InputRequest) + Send + Sync>;
type AddedTool = Arc<dyn Fn(Mcp) -> SharedTool + Send + Sync>;

#[derive(Clone)]
enum Source {
    Url(String),
    Command(Vec<String>),
    Transport(Arc<dyn Transport>),
}

/// The class-level settings of a `RubyLLM::MCP` subclass (or the keywords of `RubyLLM.mcp`).
#[derive(Clone)]
pub struct McpBuilder {
    name: Option<String>,
    source: Source,
    directory: Option<PathBuf>,
    env: Vec<(String, String)>,
    headers: Vec<(String, StringSource)>,
    bearer_token: Option<StringSource>,
    timeout: Option<Duration>,
    only: Option<Vec<String>>,
    except: Vec<String>,
    prefix: Option<String>,
    shapes: Vec<(String, ToolShape)>,
    added_tools: Vec<AddedTool>,
    approvals: Vec<(Vec<String>, Option<ApprovalCondition>)>,
    after_progress: Vec<ProgressCallback>,
    before_input_request: Vec<InputCallback>,
    config: Option<Arc<Config>>,
}

fn strings<S: AsRef<str>>(names: &[S]) -> Vec<String> {
    names.iter().map(|n| n.as_ref().to_string()).collect()
}

impl McpBuilder {
    fn new(source: Source) -> McpBuilder {
        McpBuilder {
            name: None,
            source,
            directory: None,
            env: Vec::new(),
            headers: Vec::new(),
            bearer_token: None,
            timeout: None,
            only: None,
            except: Vec::new(),
            prefix: None,
            shapes: Vec::new(),
            added_tools: Vec::new(),
            approvals: Vec::new(),
            after_progress: Vec::new(),
            before_input_request: Vec::new(),
            config: None,
        }
    }

    /// `name:`: identifies this MCP in a chat. Defaults to the URL's host labels (minus `mcp`,
    /// `api`, `www`, and the TLD) or the command's basename.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// `directory`: the working directory for a stdio server's process.
    pub fn directory(mut self, directory: impl Into<PathBuf>) -> Self {
        self.directory = Some(directory.into());
        self
    }

    /// `env NAME: value`: an environment variable for a stdio server's process.
    pub fn env(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.push((name.into(), value.into()));
        self
    }

    /// `header "X-MCP-Toolsets", "issues"`: a header sent with every request.
    pub fn header(self, name: impl Into<String>, value: impl Into<String>) -> Self {
        let value = value.into();
        self.header_with(name, move || Some(value.clone()))
    }

    /// `header("X-Account") { user.account_id }`: evaluated on every request.
    pub fn header_with(mut self, name: impl Into<String>, value: impl Fn() -> Option<String> + Send + Sync + 'static) -> Self {
        let name = name.into();
        self.headers.retain(|(n, _)| *n != name);
        self.headers.push((name, Arc::new(value)));
        self
    }

    /// `bearer_token "..."`: sent as `Authorization: Bearer ...`.
    pub fn bearer_token(self, token: impl Into<String>) -> Self {
        let token = token.into();
        self.bearer_token_with(move || Some(token.clone()))
    }

    /// `bearer_token { user.linear_token }`: evaluated on every request.
    pub fn bearer_token_with(mut self, token: impl Fn() -> Option<String> + Send + Sync + 'static) -> Self {
        self.bearer_token = Some(Arc::new(token));
        self
    }

    /// `timeout`: how long a request to the server may take. Defaults to `request_timeout`.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// `only :search_issues, :get_issue`: limits the tools the model sees.
    pub fn only<S: AsRef<str>>(mut self, names: &[S]) -> Self {
        self.only = Some(strings(names));
        self
    }

    /// `except :delete_repository`: hides the named server tools from the model.
    pub fn except<S: AsRef<str>>(mut self, names: &[S]) -> Self {
        self.except = strings(names);
        self
    }

    /// `prefix :github`: `search_issues` becomes `github_search_issues`. Renamed tools keep the
    /// name you gave them.
    pub fn prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = Some(prefix.into());
        self
    }

    /// `tool :search_files, as:, description:, fixed_arguments:, wrap:`: shapes a server tool.
    pub fn tool(mut self, name: impl Into<String>, shape: ToolShape) -> Self {
        self.shapes.push((name.into(), shape));
        self
    }

    /// `tool SearchWithPreviews`: adds a tool of your own next to the server's, built with this
    /// MCP so it can call the server.
    pub fn add_tool(mut self, build: impl Fn(Mcp) -> SharedTool + Send + Sync + 'static) -> Self {
        self.added_tools.push(Arc::new(build));
        self
    }

    /// `requires_approval :create_issue`: pauses the named server tools for approval. No names
    /// means every tool.
    pub fn requires_approval<S: AsRef<str>>(mut self, names: &[S]) -> Self {
        self.approvals.push((strings(names), None));
        self
    }

    /// `requires_approval if: :destructive?` or `if: ->(tool) { ... }`, limited to `names` when
    /// any are given.
    pub fn requires_approval_if<S: AsRef<str>>(
        mut self,
        names: &[S],
        condition: impl Fn(&McpTool) -> bool + Send + Sync + 'static,
    ) -> Self {
        self.approvals.push((strings(names), Some(Arc::new(condition))));
        self
    }

    /// `after_progress { |progress| ... }`: the progress the server reports while it works. In a
    /// chat, a tool call's progress also reaches `Chat::after_tool_progress`.
    pub fn after_progress(mut self, callback: impl Fn(&Progress) + Send + Sync + 'static) -> Self {
        self.after_progress.push(Arc::new(callback));
        self
    }

    /// `before_input_request { |request| request.answer(...) }`: the server's requests for input
    /// from the user. In a chat, a request no callback answers pauses the tool call.
    pub fn before_input_request(mut self, callback: impl Fn(&mut InputRequest) + Send + Sync + 'static) -> Self {
        self.before_input_request.push(Arc::new(callback));
        self
    }

    /// `MCP.new(context:)`: connect with this configuration's `request_timeout`.
    pub fn config(mut self, config: Arc<Config>) -> Self {
        self.config = Some(config);
        self
    }

    /// `MCP.new`. Fails with `Error::Argument` for a URL that is neither HTTPS nor loopback HTTP,
    /// or that carries credentials, like `HTTP.new`. Nothing is contacted yet.
    pub fn build(self) -> Result<Mcp> {
        let config = self.config.clone().unwrap_or_else(crate::config);
        let timeout = self.timeout.unwrap_or(config.request_timeout);
        let transport: Arc<dyn Transport> = match &self.source {
            Source::Transport(t) => t.clone(),
            Source::Command(argv) => Arc::new(Stdio::new(argv.clone(), self.env.clone(), self.directory.clone(), timeout)),
            Source::Url(url) => {
                let headers = self.headers.clone();
                let token = self.bearer_token.clone();
                let source: HeaderSource = Arc::new(move || {
                    let mut out: Vec<(String, String)> =
                        headers.iter().filter_map(|(name, value)| value().map(|v| (name.clone(), v))).collect();
                    if let Some(token) = token.as_ref().and_then(|t| t()) {
                        out.push(("Authorization".into(), format!("Bearer {token}")));
                    }
                    out
                });
                Arc::new(Http::new(url, source, timeout)?)
            }
        };
        let name = self.name.clone().unwrap_or_else(|| self.default_name());
        let capabilities = json!({ "elicitation": { "form": {}, "url": {} } });
        let client = Client::new(transport, capabilities);
        Ok(Mcp(Arc::new(Inner { name, settings: self, client, server_tools: Mutex::new(None) })))
    }

    /// `default_name_for(url:, command:)`.
    fn default_name(&self) -> String {
        match &self.source {
            Source::Url(url) => {
                let host = reqwest::Url::parse(url).ok().and_then(|u| u.host_str().map(str::to_string)).unwrap_or_default();
                let labels: Vec<&str> = host.split('.').collect();
                let labels = &labels[..labels.len().saturating_sub(1)];
                labels.iter().filter(|l| !["mcp", "api", "www"].contains(l)).copied().collect::<Vec<_>>().join("_")
            }
            Source::Command(argv) => {
                let first = argv.first().map(String::as_str).unwrap_or("");
                first.rsplit('/').next().unwrap_or(first).to_string()
            }
            Source::Transport(_) => "mcp".into(),
        }
    }
}

struct Inner {
    name: String,
    settings: McpBuilder,
    client: Client,
    server_tools: Mutex<Option<Vec<Value>>>,
}

/// `RubyLLM::MCP`: an MCP server, connected on the first request. Cheap to clone; clones share
/// the connection.
#[derive(Clone)]
pub struct Mcp(Arc<Inner>);

impl std::fmt::Debug for Mcp {
    /// `#<RubyLLM::MCP name: "learn_microsoft", url: "https://learn.microsoft.com/api/mcp">`
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut d = f.debug_struct("Mcp");
        d.field("name", &self.0.name);
        match &self.0.settings.source {
            Source::Url(url) => d.field("url", url),
            Source::Command(argv) => d.field("command", &argv.join(" ")),
            Source::Transport(_) => d.field("transport", &true),
        };
        d.finish()
    }
}

impl Mcp {
    /// `url "https://mcp.linear.app/mcp"` / `RubyLLM.mcp(url:)`: Streamable HTTP.
    pub fn url(url: impl Into<String>) -> McpBuilder {
        McpBuilder::new(Source::Url(url.into()))
    }

    /// `command "npx", "-y", ...` / `RubyLLM.mcp(command:)`: a local server over stdio. The
    /// process starts on the first request.
    pub fn command<S: AsRef<str>>(argv: impl IntoIterator<Item = S>) -> McpBuilder {
        McpBuilder::new(Source::Command(argv.into_iter().map(|s| s.as_ref().to_string()).collect()))
    }

    /// `transport { Tunnel.new(device) }` / `RubyLLM.mcp(transport:, name:)`: an MCP with a
    /// transport needs a name.
    pub fn transport(name: impl Into<String>, transport: Arc<dyn Transport>) -> McpBuilder {
        McpBuilder::new(Source::Transport(transport)).name(name)
    }

    /// `name`: identifies this MCP in a chat.
    pub fn name(&self) -> String {
        self.0.name.clone()
    }

    /// The JSON-RPC client, for methods this API has no wrapper for.
    pub fn client(&self) -> &Client {
        &self.0.client
    }

    /// `tools`: the server's tools shaped by `only`, `except`, and `tool`, followed by the tools
    /// added with `add_tool`. The server's list is fetched once. Fails with
    /// `Error::Configuration` when a declaration names a tool the server does not offer.
    pub async fn tools(&self) -> Result<Vec<SharedTool>> {
        self.server_tools().await?;
        self.cached_tools().unwrap_or_else(|| Ok(Vec::new()))
    }

    /// The shaped server tools alone, as [`McpTool`]s with their annotation predicates.
    pub async fn mcp_tools(&self) -> Result<Vec<Arc<McpTool>>> {
        let definitions = self.server_tools().await?;
        self.check_declared_tools(&definitions)?;
        Ok(definitions.iter().filter_map(|d| self.shape(d)).collect())
    }

    /// `tools` once the server's list has been fetched, without contacting the server.
    pub(crate) fn cached_tools(&self) -> Option<Result<Vec<SharedTool>>> {
        let definitions = self.0.server_tools.lock().ok()?.clone()?;
        if let Err(e) = self.check_declared_tools(&definitions) {
            return Some(Err(e));
        }
        let mut tools: Vec<SharedTool> = definitions.iter().filter_map(|d| self.shape(d)).map(|t| t as SharedTool).collect();
        tools.extend(self.0.settings.added_tools.iter().map(|build| build(self.clone())));
        Some(Ok(tools))
    }

    /// `call(name, **arguments)` (and `mcp.<tool>(...)`): calls a server tool. A tool that fails
    /// returns a result whose `is_error` is true; a protocol error is `Error::Mcp`.
    pub async fn call(&self, name: &str, arguments: Value) -> Result<McpResult> {
        let arguments = if arguments.is_object() { arguments } else { json!({}) };
        let params = json!({ "name": name, "arguments": arguments });
        Ok(McpResult::new(self.request("tools/call", params, None).await?))
    }

    /// `resources`: read when their content is first needed.
    pub async fn resources(&self) -> Result<Vec<Resource>> {
        let items = self.0.client.list("resources/list", "resources").await?;
        Ok(items.into_iter().map(|data| Resource::new(self.clone(), data)).collect())
    }

    /// `resource(uri)`: reads the resource at `uri`.
    pub async fn resource(&self, uri: &str) -> Result<Resource> {
        let result = self.request("resources/read", json!({ "uri": uri }), None).await?;
        let contents = result.get("contents").and_then(Value::as_array).cloned().unwrap_or_default();
        let data = contents.iter().find(|c| c.get("uri").and_then(Value::as_str) == Some(uri)).or(contents.first());
        let data = data.cloned().ok_or_else(|| McpError::new(format!("{} returned no content for {uri}", self.name())))?;
        Ok(Resource::new(self.clone(), data))
    }

    /// `resource(template, **variables)`: fills in a template from `resource_templates`.
    pub async fn resource_from_template(&self, template: &str, variables: Value) -> Result<Resource> {
        self.resource(&ResourceTemplate::expand(template, &variables)).await
    }

    /// `resource_templates`.
    pub async fn resource_templates(&self) -> Result<Vec<ResourceTemplate>> {
        let items = self.0.client.list("resources/templates/list", "resourceTemplates").await?;
        Ok(items.iter().map(|data| ResourceTemplate::new(self.clone(), data)).collect())
    }

    /// `prompts`: the prompts the server offers, without messages.
    pub async fn prompts(&self) -> Result<Vec<Prompt>> {
        let items = self.0.client.list("prompts/list", "prompts").await?;
        Ok(items.iter().map(|data| Prompt::new(self.clone(), data, Vec::new())).collect())
    }

    /// `prompt(name, **arguments)`: fills in a prompt; pass it to `Chat::ask_prompt`.
    pub async fn prompt(&self, name: &str, arguments: &[(&str, &str)]) -> Result<Prompt> {
        let arguments: Map<String, Value> = arguments.iter().map(|(k, v)| (k.to_string(), Value::String(v.to_string()))).collect();
        let result = self.request("prompts/get", json!({ "name": name, "arguments": arguments }), None).await?;
        let mut messages = Vec::new();
        for message in result.get("messages").and_then(Value::as_array).into_iter().flatten() {
            let (text, attachments) = content::read(std::slice::from_ref(message.get("content").unwrap_or(&Value::Null)));
            let role = Role::parse(message.get("role").and_then(Value::as_str).unwrap_or(""))?;
            messages.push(Message::new(role, Some(text)).with_attachments(attachments));
        }
        let data = json!({ "name": name, "description": result.get("description") });
        Ok(Prompt::new(self.clone(), &data, messages))
    }

    /// `suggest(reference, values)`: completes the first value, with the rest as context.
    pub(crate) async fn suggest(&self, reference: Value, values: &[(&str, &str)]) -> Result<Vec<String>> {
        let Some(((argument, value), filled)) = values.split_first() else {
            return Err(Error::Argument("Pass the value to complete as a keyword".into()));
        };
        let mut params = json!({ "ref": reference, "argument": { "name": argument, "value": value } });
        if !filled.is_empty() {
            let arguments: Map<String, Value> = filled.iter().map(|(k, v)| (k.to_string(), Value::String(v.to_string()))).collect();
            params["context"] = json!({ "arguments": arguments });
        }
        let result = self.0.client.request("completion/complete", params, &[], &mut |_| {}).await?;
        Ok(result
            .pointer("/completion/values")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect())
    }

    /// `requires_approval?(tool)`, according to `requires_approval`.
    pub(crate) fn requires_approval(&self, tool: &McpTool) -> bool {
        self.0.settings.approvals.iter().any(|(names, condition)| {
            (names.is_empty() || names.contains(&tool.server_name)) && condition.as_ref().is_none_or(|c| c(tool))
        })
    }

    /// `run(tool, arguments, input:)`: runs one of this MCP's tools with the model's arguments.
    pub(crate) async fn run(&self, tool: &McpTool, arguments: Map<String, Value>, input: Option<InputState>) -> Result<ToolResult> {
        let mut sent = arguments.clone();
        for (name, value) in &tool.fixed_arguments {
            sent.insert(name.clone(), value());
        }
        let params = json!({ "name": tool.server_name, "arguments": sent });
        let result = McpResult::new(self.request("tools/call", params, input).await?);
        if result.is_error() {
            return Ok(ToolResult::error(result.text));
        }
        Ok(match &tool.wrap {
            Some(wrap) => wrap(&result, &arguments),
            None => result.content(),
        })
    }

    /// `instructions`: what the server says about using it.
    pub async fn instructions(&self) -> Result<Option<String>> {
        let server = self.0.client.server().await?;
        Ok(server.get("instructions").and_then(Value::as_str).map(str::to_string))
    }

    /// `version`: the version the server reports for itself.
    pub async fn version(&self) -> Result<Option<String>> {
        let server = self.0.client.server().await?;
        let info = server.get("serverInfo").or_else(|| server.pointer("/_meta/io.modelcontextprotocol~1serverInfo"));
        Ok(info.and_then(|i| i.get("version")).and_then(Value::as_str).map(str::to_string))
    }

    /// `close`: closes the connection, stopping a stdio server's process. The next request
    /// reconnects.
    pub async fn close(&self) {
        self.0.client.close().await;
    }

    /// `request(method, params, input:)`: answers `input_required` results with the
    /// `before_input_request` callbacks, up to `INPUT_ROUNDS` times.
    async fn request(&self, method: &str, params: Value, input: Option<InputState>) -> Result<Value> {
        let mut result = match &input {
            Some(input) => self.send_answers(method, &params, input).await?,
            None => self.send_request(method, params.clone()).await?,
        };
        for _ in 0..INPUT_ROUNDS {
            if result.get("resultType").and_then(Value::as_str) != Some("input_required") {
                return Ok(result);
            }
            let input = InputState { requests: self.input_requests(&result), request_state: result.get("requestState").cloned() };
            if !input.requests.iter().all(InputRequest::is_answered) {
                return Err(InputRequiredError::new(&self.name(), input).into());
            }
            result = self.send_answers(method, &params, &input).await?;
        }
        Err(McpError::new(format!("{} kept asking for input", self.name())).into())
    }

    async fn send_answers(&self, method: &str, params: &Value, input: &InputState) -> Result<Value> {
        let responses: Map<String, Value> =
            input.requests.iter().map(|r| (r.key.clone(), r.response.clone().unwrap_or(Value::Null))).collect();
        let mut params = params.clone();
        params["inputResponses"] = Value::Object(responses);
        if let Some(state) = &input.request_state {
            params["requestState"] = state.clone();
        }
        self.send_request(method, params).await
    }

    fn input_requests(&self, result: &Value) -> Vec<InputRequest> {
        let Some(requests) = result.get("inputRequests").and_then(Value::as_object) else { return Vec::new() };
        requests
            .iter()
            .filter(|(_, r)| r.get("method").and_then(Value::as_str) == Some("elicitation/create"))
            .map(|(key, r)| {
                let mut request = InputRequest::new(key.clone(), r.get("params").cloned().unwrap_or_else(|| json!({})));
                for callback in &self.0.settings.before_input_request {
                    if !request.is_answered() {
                        callback(&mut request);
                    }
                }
                request
            })
            .collect()
    }

    /// `send_request`: asks for progress (with a fresh `progressToken`) only when someone listens.
    async fn send_request(&self, method: &str, mut params: Value) -> Result<Value> {
        let headers = if method == "tools/call" { self.mirrored_headers(&params).await? } else { Vec::new() };
        let mut listeners: Vec<ProgressCallback> = self.0.settings.after_progress.clone();
        listeners.extend(progress::listener());
        if listeners.is_empty() {
            return self.0.client.request(method, params, &headers, &mut |_| {}).await;
        }
        let token = uuid::Uuid::new_v4().to_string();
        params["_meta"] = json!({ "progressToken": token });
        let mut on_notification = |notification: &Value| {
            let data = notification.get("params").cloned().unwrap_or_else(|| json!({}));
            let is_progress = notification.get("method").and_then(Value::as_str) == Some("notifications/progress");
            if !is_progress || data.get("progressToken").and_then(Value::as_str) != Some(token.as_str()) {
                return;
            }
            let progress = Progress {
                value: data.get("progress").and_then(Value::as_f64),
                total: data.get("total").and_then(Value::as_f64),
                message: data.get("message").and_then(Value::as_str).map(str::to_string),
            };
            for listener in &listeners {
                listener(&progress);
            }
        };
        self.0.client.request(method, params, &headers, &mut on_notification).await
    }

    async fn mirrored_headers(&self, params: &Value) -> Result<Vec<(String, String)>> {
        let definitions = self.server_tools().await?;
        let name = params.get("name").and_then(Value::as_str);
        let empty = Map::new();
        let arguments = params.get("arguments").and_then(Value::as_object).unwrap_or(&empty);
        Ok(definitions
            .iter()
            .find(|d| d.get("name").and_then(Value::as_str) == name)
            .map(|d| param_headers::headers_for(d, arguments))
            .unwrap_or_default())
    }

    fn shape(&self, definition: &Value) -> Option<Arc<McpTool>> {
        let settings = &self.0.settings;
        let name = definition.get("name").and_then(Value::as_str).unwrap_or("");
        if settings.only.as_ref().is_some_and(|only| !only.iter().any(|n| n == name)) || settings.except.iter().any(|n| n == name) {
            return None;
        }
        let shape = settings.shapes.iter().filter(|(n, _)| n == name).fold(ToolShape::default(), |acc, (_, s)| acc.merge(s));
        Some(Arc::new(McpTool::new(self.clone(), definition, settings.prefix.as_deref(), shape)))
    }

    fn check_declared_tools(&self, definitions: &[Value]) -> Result<()> {
        let settings = &self.0.settings;
        let mut declared: Vec<&String> = settings.only.iter().flatten().collect();
        declared.extend(settings.approvals.iter().flat_map(|(names, _)| names));
        declared.extend(settings.shapes.iter().map(|(name, _)| name));
        let offered: Vec<&str> = definitions.iter().filter_map(|d| d.get("name").and_then(Value::as_str)).collect();
        let mut missing: Vec<&str> = Vec::new();
        for name in declared {
            if !offered.contains(&name.as_str()) && !missing.contains(&name.as_str()) {
                missing.push(name);
            }
        }
        if missing.is_empty() {
            return Ok(());
        }
        Err(Error::Configuration(format!("{} declares {}, which the server does not offer", self.name(), missing.join(", "))))
    }

    async fn server_tools(&self) -> Result<Vec<Value>> {
        if let Some(cached) = self.0.server_tools.lock().ok().and_then(|c| c.clone()) {
            return Ok(cached);
        }
        let definitions: Vec<Value> =
            self.0.client.list("tools/list", "tools").await?.into_iter().filter(param_headers::is_valid).collect();
        if let Ok(mut cache) = self.0.server_tools.lock() {
            *cache = Some(definitions.clone());
        }
        Ok(definitions)
    }
}
