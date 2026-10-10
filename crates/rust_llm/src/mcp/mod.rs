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

mod apps;
mod client;
mod collection;
mod content;
mod error;
mod http;
mod input_request;
mod listener;
mod oauth;
mod param_headers;
mod prompt;
mod resource;
mod resource_template;
mod result;
mod stdio;
mod task;
mod tool;

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde_json::{Map, Value, json};

pub use client::{Client, LEGACY_VERSION, LEGACY_VERSIONS, VERSION, client_info};
pub use collection::Collection;
pub use error::McpError;
pub use http::{Authorization, HeaderSource, Http};
pub use input_request::{Field, InputRequest, InputRequiredError, InputState};
pub use listener::{Listener, ListenerCallback, Sleep, Subscribe, callback as listener_callback};
pub use oauth::{
    Challenge, CredentialStore, Grant, IdentityProvider, MemoryStore, OAuth, OAuthSettings,
    OwnerSource, Recovery, Synchronized, ValueSource,
};
pub use prompt::{Prompt, PromptArgument};
pub use resource::{Resource, ResourceContent};
pub use resource_template::ResourceTemplate;
pub use result::McpResult;
pub use stdio::Stdio;
pub use task::{Task, TaskStatus};
pub use tool::{McpTool, ToolShape};

use crate::config::Config;
use crate::error::{Error, Result};
use crate::message::{Message, Role};
use crate::progress::{self, Progress};
use crate::tool::{SharedTool, ToolResult};

const INPUT_ROUNDS: usize = 10;
const POLL_INTERVAL: Duration = Duration::from_secs(1);
const UNKNOWN_TOOL_ERRORS: [i64; 2] = [-32_601, -32_602];
const TASKS_EXTENSION: &str = "io.modelcontextprotocol/tasks";

/// The kinds of requests for input a server may send (`MCP.input_requests`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputKind {
    Form,
    Url,
}

impl InputKind {
    fn as_str(self) -> &'static str {
        match self {
            InputKind::Form => "form",
            InputKind::Url => "url",
        }
    }

    /// `input_requests :form, :url`: Fails with `Error::Argument` for a kind it does not know.
    pub fn parse(kind: &str) -> Result<InputKind> {
        match kind {
            "form" => Ok(InputKind::Form),
            "url" => Ok(InputKind::Url),
            other => Err(Error::Argument(format!("Unknown input requests: {other}"))),
        }
    }
}

/// The extensions to the protocol RustLLM implements, declared by name (`extension :apps`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Extension {
    /// MCP Apps: tools come with a UI your app renders next to their results, and tools that
    /// only a UI may call stay out of chats.
    Apps,
    /// Tasks: a server may run a long tool call in the background.
    Tasks,
}

impl Extension {
    /// `EXTENSIONS[name]`: the identifier and default settings.
    fn identifier(self) -> (&'static str, Map<String, Value>) {
        match self {
            Extension::Apps => (
                apps::EXTENSION,
                json!({ "mimeTypes": [apps::MIME_TYPE] })
                    .as_object()
                    .cloned()
                    .unwrap_or_default(),
            ),
            Extension::Tasks => (TASKS_EXTENSION, Map::new()),
        }
    }

    /// `extension :widgets`: Fails with `Error::Argument` for a name RustLLM does not know.
    pub fn parse(name: &str) -> Result<Extension> {
        match name {
            "apps" => Ok(Extension::Apps),
            "tasks" => Ok(Extension::Tasks),
            other => Err(Error::Argument(format!("Unknown MCP extension: {other}"))),
        }
    }
}

/// The protocol's log levels (`MCP.log_level`), from the least to the most severe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    Debug,
    Info,
    Notice,
    Warning,
    Error,
    Critical,
    Alert,
    Emergency,
}

impl LogLevel {
    const ALL: [LogLevel; 8] = [
        LogLevel::Debug,
        LogLevel::Info,
        LogLevel::Notice,
        LogLevel::Warning,
        LogLevel::Error,
        LogLevel::Critical,
        LogLevel::Alert,
        LogLevel::Emergency,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            LogLevel::Debug => "debug",
            LogLevel::Info => "info",
            LogLevel::Notice => "notice",
            LogLevel::Warning => "warning",
            LogLevel::Error => "error",
            LogLevel::Critical => "critical",
            LogLevel::Alert => "alert",
            LogLevel::Emergency => "emergency",
        }
    }

    /// `log_level :verbose`: Fails with `Error::Argument` for a level the protocol does not
    /// define.
    pub fn parse(level: &str) -> Result<LogLevel> {
        LogLevel::ALL
            .into_iter()
            .find(|l| l.as_str() == level)
            .ok_or_else(|| Error::Argument(format!("Unknown MCP log level: {level}")))
    }

    /// `LOG_LEVELS`: where each level lands in the logger.
    fn log(self, message: &str) {
        match self {
            LogLevel::Debug => tracing::debug!("{message}"),
            LogLevel::Info | LogLevel::Notice => tracing::info!("{message}"),
            LogLevel::Warning => tracing::warn!("{message}"),
            _ => tracing::error!("{message}"),
        }
    }
}

/// What a server announces it changed, as `after_change` receives it.
#[derive(Debug, Clone)]
pub enum Change {
    /// The server's list of tools changed. The MCP has already forgotten the old tools.
    Tools,
    /// The server's list of prompts changed.
    Prompts,
    /// The server's list of resources changed.
    Resources,
    /// The content of a resource it listens to changed.
    Resource(Resource),
    /// The status of a task it listens to changed, with the task as it stands.
    Task(Task),
}

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
    /// `listen(message, version:) { |notification| }`: with a `subscriptions/listen` message,
    /// keeps its stream open, passing the subscription's notifications to `on_notification`, and
    /// returns its answer (or the `notifications/cancelled` that ends it); without one, listens
    /// on the session's own stream of a server that predates 2026-07-28 until it ends. Transports
    /// that cannot listen fail like `respond_to?(:listen)` being false.
    async fn listen(
        &self,
        _message: Option<&Value>,
        _version: Option<&str>,
        _on_notification: &mut OnNotification<'_>,
    ) -> Result<Option<Value>> {
        Err(Error::Configuration(
            "The MCP transport does not respond to listen".into(),
        ))
    }
}

type StringSource = Arc<dyn Fn() -> Option<String> + Send + Sync>;
type ApprovalCondition = Arc<dyn Fn(&McpTool) -> bool + Send + Sync>;
type ProgressCallback = Arc<dyn Fn(&Progress) + Send + Sync>;
type InputCallback = Arc<dyn Fn(&mut InputRequest) + Send + Sync>;
type AddedTool = Arc<dyn Fn(Mcp) -> SharedTool + Send + Sync>;
type ChangeCallback = Arc<dyn Fn(Change) -> futures::future::BoxFuture<'static, ()> + Send + Sync>;

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
    after_change: Vec<ChangeCallback>,
    deferrals: Vec<Vec<String>>,
    input_requests: Vec<InputKind>,
    extensions: Map<String, Value>,
    log_level: Option<LogLevel>,
    oauth: Option<OAuthSettings>,
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
            after_change: Vec::new(),
            deferrals: Vec::new(),
            input_requests: vec![InputKind::Form, InputKind::Url],
            extensions: Map::new(),
            log_level: None,
            oauth: None,
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
    pub fn header_with(
        mut self,
        name: impl Into<String>,
        value: impl Fn() -> Option<String> + Send + Sync + 'static,
    ) -> Self {
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
    pub fn bearer_token_with(
        mut self,
        token: impl Fn() -> Option<String> + Send + Sync + 'static,
    ) -> Self {
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
        self.approvals
            .push((strings(names), Some(Arc::new(condition))));
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
    pub fn before_input_request(
        mut self,
        callback: impl Fn(&mut InputRequest) + Send + Sync + 'static,
    ) -> Self {
        self.before_input_request.push(Arc::new(callback));
        self
    }

    /// `after_change { |change| ... }`: the changes the server announces, `Change::Tools`,
    /// `Prompts`, or `Resources` when the server's list of them changes, the `Resource` whose
    /// content changed, or the `Task` whose status changed. The MCP has already forgotten the
    /// old tools when `Change::Tools` arrives. Async, so a callback can make requests.
    ///
    /// Servers that predate 2026-07-28 announce changes as they answer a request; the callback
    /// runs once the request is answered. Newer servers announce them only while the MCP listens;
    /// see [`Mcp::listen`]. Changes made while a listener reconnects are lost, so once it is
    /// back, the callback runs for everything it listens to.
    pub fn after_change<F, Fut>(mut self, callback: F) -> Self
    where
        F: Fn(Change) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        self.after_change
            .push(Arc::new(move |change| Box::pin(callback(change))));
        self
    }

    /// `defer :search_code, :list_workflows`: keeps the named server tools out of the model's
    /// context until the provider's tool search loads them. No names defers every tool.
    /// `Chat::with_mcp_deferred` overrides it per chat.
    pub fn defer<S: AsRef<str>>(mut self, names: &[S]) -> Self {
        self.deferrals.push(strings(names));
        self
    }

    /// `input_requests :url` / `input_requests false`: the kinds of requests for input the
    /// server may send. Both by default; an empty list when your app cannot show them to
    /// anyone, so a call never waits on an answer that will not come. RustLLM declines requests
    /// of other kinds.
    pub fn input_requests(mut self, kinds: &[InputKind]) -> Self {
        self.input_requests = kinds.to_vec();
        self
    }

    /// `extension "com.example/audit", level: "full"`: declares an extension to the protocol
    /// your app supports, with settings that go to the server as written. Fails with
    /// `Error::Argument` for a name without a vendor prefix.
    pub fn extension(mut self, name: &str, settings: Value) -> Result<Self> {
        if !name.contains('/') {
            return Err(Error::Argument(format!(
                "MCP extensions are named with a vendor prefix, such as com.example/{name}"
            )));
        }
        self.extensions.insert(
            name.to_string(),
            Value::Object(settings.as_object().cloned().unwrap_or_default()),
        );
        Ok(self)
    }

    /// `extension :apps` / `extension :tasks, **settings`: an extension RustLLM implements, its
    /// default settings merged with `settings`.
    pub fn with_extension(mut self, extension: Extension, settings: Value) -> Self {
        let (name, mut defaults) = extension.identifier();
        defaults.extend(settings.as_object().cloned().unwrap_or_default());
        self.extensions.insert(name.into(), Value::Object(defaults));
        self
    }

    /// `extensions`: what `extension` declared, by identifier.
    pub fn extensions(&self) -> &Map<String, Value> {
        &self.extensions
    }

    /// `input_requests`: the accepted kinds.
    pub fn accepted_input_requests(&self) -> &[InputKind] {
        &self.input_requests
    }

    /// `log_level :warning`: asks the server for the log messages it writes while it works on
    /// a request, at `level` and above, and writes them to the log. Without a level, servers
    /// send no log messages.
    pub fn log_level(mut self, level: LogLevel) -> Self {
        self.log_level = Some(level);
        self
    }

    /// `log_level`: the level asked for, if any.
    pub fn configured_log_level(&self) -> Option<LogLevel> {
        self.log_level
    }

    /// `oauth owner: :user, scopes:, client_id:, client_secret:` (or `RubyLLM.mcp(oauth:)`):
    /// authorizes requests with OAuth, as the MCP authorization spec describes. RustLLM discovers
    /// the server's authorization server and registers itself unless you pass the `client_id`
    /// and `client_secret` of an app you registered. Send the user to [`Mcp::authorization_url`],
    /// then pass the callback's parameters to [`Mcp::authorize`].
    pub fn oauth(mut self, settings: OAuthSettings) -> Self {
        self.oauth = Some(settings);
        self
    }

    /// `MCP.new(context:)`: connect with this configuration's `request_timeout`, and keep OAuth
    /// credentials in its `mcp_credential_store`.
    pub fn config(mut self, config: Arc<Config>) -> Self {
        self.config = Some(config);
        self
    }

    /// `MCP.new`. Fails with `Error::Argument` for a URL that is neither HTTPS nor loopback HTTP,
    /// or that carries credentials, like `HTTP.new`. Nothing is contacted yet.
    pub fn build(self) -> Result<Mcp> {
        let config = self.config.clone().unwrap_or_else(crate::config);
        let timeout = self.timeout.unwrap_or(config.request_timeout);
        let name = self.name.clone().unwrap_or_else(|| self.default_name());
        let authorizer = match (&self.source, &self.oauth) {
            (Source::Url(url), Some(settings)) => Some(Arc::new(oauth::Authorizer::new(
                name.clone(),
                url.clone(),
                settings.clone(),
                config.clone(),
            ))),
            _ => None,
        };
        let transport: Arc<dyn Transport> = match &self.source {
            Source::Transport(t) => t.clone(),
            Source::Command(argv) => Arc::new(Stdio::new(
                argv.clone(),
                self.env.clone(),
                self.directory.clone(),
                timeout,
            )),
            Source::Url(url) => {
                let headers = self.headers.clone();
                let token = if authorizer.is_some() {
                    None
                } else {
                    self.bearer_token.clone()
                };
                let source: HeaderSource = Arc::new(move |_verb| {
                    let mut out: Vec<(String, String)> = headers
                        .iter()
                        .filter_map(|(name, value)| value().map(|v| (name.clone(), v)))
                        .collect();
                    if let Some(token) = token.as_ref().and_then(|t| t()) {
                        out.push(("Authorization".into(), format!("Bearer {token}")));
                    }
                    out
                });
                let http = Http::new(url, source, timeout)?;
                match &authorizer {
                    Some(authorizer) => Arc::new(http.with_authorization(authorizer.clone())),
                    None => Arc::new(http),
                }
            }
        };
        // `capabilities`: the accepted input requests, and the extensions in use, the
        // authorization ones (`OAuth.extensions(oauth_settings)`) first.
        let mut capabilities = Map::new();
        if !self.input_requests.is_empty() {
            let kinds: Map<String, Value> = self
                .input_requests
                .iter()
                .map(|k| (k.as_str().to_string(), json!({})))
                .collect();
            capabilities.insert("elicitation".into(), Value::Object(kinds));
        }
        let mut extensions = self
            .oauth
            .as_ref()
            .map(OAuthSettings::extensions)
            .unwrap_or_default();
        extensions.extend(self.extensions.clone());
        if !extensions.is_empty() {
            capabilities.insert("extensions".into(), Value::Object(extensions));
        }
        let capabilities = Value::Object(capabilities);
        let client = Arc::new(Client::new(transport, capabilities));
        let mcp = Mcp(Arc::new(Inner {
            name,
            timeout,
            settings: self,
            client: client.clone(),
            server_tools: Mutex::new(None),
            tool_changes: AtomicU64::new(0),
            listener: std::sync::OnceLock::new(),
            authorizer,
        }));
        let weak = Arc::downgrade(&mcp.0);
        client.on_change(move |notification| {
            let mcp = weak.upgrade().map(Mcp);
            async move {
                if let Some(mcp) = mcp {
                    mcp.changed(&notification).await;
                }
            }
        });
        Ok(mcp)
    }

    /// `default_name_for(url:, command:)`.
    fn default_name(&self) -> String {
        match &self.source {
            Source::Url(url) => {
                let host = reqwest::Url::parse(url)
                    .ok()
                    .and_then(|u| u.host_str().map(str::to_string))
                    .unwrap_or_default();
                let labels: Vec<&str> = host.split('.').collect();
                let labels = &labels[..labels.len().saturating_sub(1)];
                labels
                    .iter()
                    .filter(|l| !["mcp", "api", "www"].contains(l))
                    .copied()
                    .collect::<Vec<_>>()
                    .join("_")
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
    timeout: Duration,
    settings: McpBuilder,
    client: Arc<Client>,
    server_tools: Mutex<Option<Vec<Value>>>,
    /// `@tool_changes`: bumped whenever the tools are forgotten, so a list fetched meanwhile is
    /// not kept.
    tool_changes: AtomicU64,
    listener: std::sync::OnceLock<Listener>,
    authorizer: Option<Arc<oauth::Authorizer>>,
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
        McpBuilder::new(Source::Command(
            argv.into_iter().map(|s| s.as_ref().to_string()).collect(),
        ))
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

    /// The class-level settings this MCP was built with.
    pub fn settings(&self) -> &McpBuilder {
        &self.0.settings
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
        let mut tools: Vec<SharedTool> = definitions
            .iter()
            .filter_map(|d| self.shape(d))
            .map(|t| t as SharedTool)
            .collect();
        tools.extend(
            self.0
                .settings
                .added_tools
                .iter()
                .map(|build| build(self.clone())),
        );
        Some(Ok(tools))
    }

    /// `call(name, **arguments)` (and `mcp.<tool>(...)`): calls a server tool. A tool that fails
    /// returns a result whose `is_error` is true; a protocol error is `Error::Mcp`. When the
    /// server runs the call as a task, waits for it the way [`Task::wait`] does, and cancels it
    /// if waiting fails.
    pub async fn call(&self, name: &str, arguments: Value) -> Result<McpResult> {
        let arguments = if arguments.is_object() {
            arguments
        } else {
            json!({})
        };
        let params = json!({ "name": name, "arguments": arguments });
        let data = match self.call_tool(params, None).await? {
            Outcome::Task(task) => self.finish(*task).await?,
            Outcome::Result(data) => data,
        };
        Ok(McpResult::new(data, self.ui_uri_of(name).await?))
    }

    /// `defers?(tool)`: whether one of this MCP's tools is deferred according to `defer`, or
    /// its own `Tool::is_deferred`.
    pub(crate) fn defers(&self, tool: &dyn crate::tool::Tool, server_name: Option<&str>) -> bool {
        if tool.is_deferred() {
            return true;
        }
        self.0.settings.deferrals.iter().any(|names| {
            names.is_empty() || server_name.is_some_and(|n| names.iter().any(|d| d == n))
        })
    }

    /// The server name of one of this MCP's shaped tools, by the name the model calls it.
    pub(crate) fn server_name_of(&self, tool_name: &str) -> Option<String> {
        let definitions = self.0.server_tools.lock().ok()?.clone()?;
        definitions
            .iter()
            .filter_map(|d| self.shape(d))
            .find(|t| t.name == tool_name)
            .map(|t| t.server_name.clone())
    }

    /// `poll_task(id)`: checks on the task once and returns its state, reporting what it is
    /// doing as progress.
    pub(crate) async fn poll_task(&self, id: &str) -> Result<Value> {
        let data = self
            .0
            .client
            .request("tasks/get", json!({ "taskId": id }), &[], &mut |_| {})
            .await?;
        let status = data.get("status").and_then(Value::as_str);
        if let Some(message) = data.get("statusMessage").and_then(Value::as_str)
            && matches!(status, Some("working" | "input_required"))
        {
            let progress = Progress {
                value: None,
                total: None,
                message: Some(message.to_string()),
            };
            for listener in self.progress_listeners() {
                listener(&progress);
            }
        }
        Ok(data)
    }

    /// `cancel_task(id)`: asks the server to cancel the task.
    pub(crate) async fn cancel_task(&self, id: &str) -> Result<()> {
        self.0
            .client
            .request("tasks/cancel", json!({ "taskId": id }), &[], &mut |_| {})
            .await
            .map(|_| ())
    }

    /// `await_task(task, timeout:, interval:)`: checks on the task until it is done.
    pub(crate) async fn await_task(
        &self,
        task: &mut Task,
        timeout: Option<Duration>,
        interval: Option<Duration>,
    ) -> Result<()> {
        let limit = timeout.unwrap_or(self.0.timeout);
        let deadline = Instant::now() + limit;
        while !task.is_done() {
            if task.status() == TaskStatus::InputRequired {
                self.ask_for_task_input(task).await?;
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(McpError::new(format!(
                    "Task {} did not finish in {} seconds",
                    task.id,
                    seconds(limit)
                ))
                .into());
            }
            let pause = interval
                .or(task.poll_interval())
                .unwrap_or(POLL_INTERVAL)
                .min(remaining);
            pause_cancellable(pause).await?;
            task.refresh().await?;
        }
        if !task.is_completed() {
            return Err(task.error().into());
        }
        Ok(())
    }

    /// `listen(resources:, tasks:)`: listens for the server's changes in a background task until
    /// [`Mcp::close`], so `after_change` callbacks run as changes happen and [`Mcp::tools`]
    /// follows the server's list. `resources` (URIs) hear when their content changes, `tasks`
    /// (ids) when their status changes; a later call replaces them. Returns once the server
    /// confirms. Without resources or tasks, does nothing for a server that announces no
    /// changes. Fails with `Error::Mcp` when the server cannot be reached or does not send
    /// updates for the resources or tasks.
    pub async fn listen(&self, resources: &[&str], tasks: &[&str]) -> Result<&Self> {
        let changes = self.listened_changes(resources, tasks).await?;
        let watched = self
            .listener()
            .start(changes)
            .await?
            .unwrap_or_else(|| json!({}));
        let listed = |key: &str| -> Vec<String> {
            watched
                .get(key)
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        };
        let (uris, ids) = (listed("resourceSubscriptions"), listed("taskIds"));
        let missing: Vec<&str> = resources
            .iter()
            .filter(|u| !uris.iter().any(|w| w == *u))
            .chain(tasks.iter().filter(|t| !ids.iter().any(|w| w == *t)))
            .copied()
            .collect();
        if missing.is_empty() {
            return Ok(self);
        }
        self.listener().stop().await;
        Err(McpError::new(format!(
            "{} does not send updates for {}",
            self.name(),
            missing.join(", ")
        ))
        .into())
    }

    /// `resources`: read when their content is first needed.
    pub async fn resources(&self) -> Result<Vec<Resource>> {
        let items = self.0.client.list("resources/list", "resources").await?;
        Ok(items
            .into_iter()
            .map(|data| Resource::new(self.clone(), data))
            .collect())
    }

    /// `resource(uri)`: reads the resource at `uri`.
    pub async fn resource(&self, uri: &str) -> Result<Resource> {
        let result = self
            .request("resources/read", json!({ "uri": uri }), None)
            .await?;
        let contents = result
            .get("contents")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let data = contents
            .iter()
            .find(|c| c.get("uri").and_then(Value::as_str) == Some(uri))
            .or(contents.first());
        let data = data.cloned().ok_or_else(|| {
            McpError::new(format!("{} returned no content for {uri}", self.name()))
        })?;
        Ok(Resource::new(self.clone(), data))
    }

    /// `resource(template, **variables)`: fills in a template from `resource_templates`.
    pub async fn resource_from_template(
        &self,
        template: &str,
        variables: Value,
    ) -> Result<Resource> {
        self.resource(&ResourceTemplate::expand(template, &variables))
            .await
    }

    /// `resource_templates`.
    pub async fn resource_templates(&self) -> Result<Vec<ResourceTemplate>> {
        let items = self
            .0
            .client
            .list("resources/templates/list", "resourceTemplates")
            .await?;
        Ok(items
            .iter()
            .map(|data| ResourceTemplate::new(self.clone(), data))
            .collect())
    }

    /// `prompts`: the prompts the server offers, without messages.
    pub async fn prompts(&self) -> Result<Vec<Prompt>> {
        let items = self.0.client.list("prompts/list", "prompts").await?;
        Ok(items
            .iter()
            .map(|data| Prompt::new(self.clone(), data, Vec::new()))
            .collect())
    }

    /// `prompt(name, **arguments)`: fills in a prompt; pass it to `Chat::ask_prompt`.
    pub async fn prompt(&self, name: &str, arguments: &[(&str, &str)]) -> Result<Prompt> {
        let arguments: Map<String, Value> = arguments
            .iter()
            .map(|(k, v)| (k.to_string(), Value::String(v.to_string())))
            .collect();
        let result = self
            .request(
                "prompts/get",
                json!({ "name": name, "arguments": arguments }),
                None,
            )
            .await?;
        let mut messages = Vec::new();
        for message in result
            .get("messages")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let (text, attachments) = content::read(std::slice::from_ref(
                message.get("content").unwrap_or(&Value::Null),
            ));
            let role = Role::parse(message.get("role").and_then(Value::as_str).unwrap_or(""))?;
            messages.push(Message::new(role, Some(text)).with_attachments(attachments));
        }
        let data = json!({ "name": name, "description": result.get("description") });
        Ok(Prompt::new(self.clone(), &data, messages))
    }

    /// `suggest(reference, values)`: completes the first value, with the rest as context.
    pub(crate) async fn suggest(
        &self,
        reference: Value,
        values: &[(&str, &str)],
    ) -> Result<Vec<String>> {
        let Some(((argument, value), filled)) = values.split_first() else {
            return Err(Error::Argument(
                "Pass the value to complete as a keyword".into(),
            ));
        };
        let mut params =
            json!({ "ref": reference, "argument": { "name": argument, "value": value } });
        if !filled.is_empty() {
            let arguments: Map<String, Value> = filled
                .iter()
                .map(|(k, v)| (k.to_string(), Value::String(v.to_string())))
                .collect();
            params["context"] = json!({ "arguments": arguments });
        }
        let result = self
            .0
            .client
            .request("completion/complete", params, &[], &mut |_| {})
            .await?;
        Ok(result
            .pointer("/completion/values")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect())
    }

    async fn ui_uri_of(&self, name: &str) -> Result<Option<String>> {
        Ok(self.server_tool(name).await?.and_then(|d| {
            d.get("_meta")
                .and_then(Value::as_object)
                .and_then(apps::uri)
        }))
    }

    async fn server_tool(&self, name: &str) -> Result<Option<Value>> {
        Ok(self
            .server_tools()
            .await?
            .into_iter()
            .find(|d| d.get("name").and_then(Value::as_str) == Some(name)))
    }

    /// `requires_approval?(tool)`, according to `requires_approval`.
    pub(crate) fn requires_approval(&self, tool: &McpTool) -> bool {
        self.0.settings.approvals.iter().any(|(names, condition)| {
            (names.is_empty() || names.contains(&tool.server_name))
                && condition.as_ref().is_none_or(|c| c(tool))
        })
    }

    /// `run(tool, arguments, input:)`: runs one of this MCP's tools with the model's arguments.
    pub(crate) async fn run(
        &self,
        tool: &McpTool,
        arguments: Map<String, Value>,
        input: Option<InputState>,
    ) -> Result<ToolResult> {
        let mut sent = arguments.clone();
        for (name, value) in &tool.fixed_arguments {
            sent.insert(name.clone(), value());
        }
        let params = json!({ "name": tool.server_name, "arguments": sent });
        let data = match self.call_tool(params, input).await? {
            Outcome::Task(task) => return Err(Error::McpTask(task)),
            Outcome::Result(data) => data,
        };
        let result = McpResult::new(data, tool.ui_uri.clone());
        if result.is_error() {
            return Ok(ToolResult::error(result.text));
        }
        Ok(match &tool.wrap {
            Some(wrap) => wrap(&result, &arguments),
            None => {
                let mut content = result.content();
                content.mcp_result = Some(Box::new(result));
                content
            }
        })
    }

    /// `call_tool(params, input:)`: a paused call resumes its task, if it waits on one.
    async fn call_tool(&self, params: Value, input: Option<InputState>) -> Result<Outcome> {
        if let Some(state) = input.as_ref().and_then(|i| i.task.clone()) {
            let mut task = Task::load(Some(self.clone()), &state, None);
            let answered = input.map(|i| i.requests).filter(|r| !r.is_empty());
            return self
                .check_task(&mut task, answered)
                .await
                .map(|done| match done {
                    Some(data) => Outcome::Result(data),
                    None => Outcome::Task(Box::new(task)),
                });
        }
        let data = match self.request("tools/call", params, input).await {
            Err(Error::Mcp(e)) if e.code.is_some_and(|c| UNKNOWN_TOOL_ERRORS.contains(&c)) => {
                self.forget_tools();
                return Err(Error::Mcp(e));
            }
            other => other?,
        };
        if data.get("resultType").and_then(Value::as_str) == Some("task") {
            return Ok(Outcome::Task(Box::new(Task::new(Some(self.clone()), data))));
        }
        Ok(Outcome::Result(data))
    }

    /// `check_task(task, answered_requests)`: the result once the task completed, `None` while
    /// it runs.
    async fn check_task(
        &self,
        task: &mut Task,
        answered: Option<Vec<InputRequest>>,
    ) -> Result<Option<Value>> {
        if let Some(requests) = answered {
            self.update_task(task, &requests).await?;
        }
        task.refresh().await?;
        if task.is_completed() {
            return Ok(Some(
                task.data()
                    .get("result")
                    .cloned()
                    .unwrap_or_else(|| json!({})),
            ));
        }
        if task.is_done() {
            return Err(task.error().into());
        }
        if task.status() == TaskStatus::InputRequired {
            self.ask_for_task_input(task).await?;
        }
        Ok(None)
    }

    async fn ask_for_task_input(&self, task: &mut Task) -> Result<()> {
        let mut pending = task
            .data()
            .get("inputRequests")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        for key in task.answered() {
            pending.remove(key);
        }
        let requests = self.input_requests(&json!({ "inputRequests": pending }));
        if requests.is_empty() {
            return Ok(());
        }
        if !requests.iter().all(InputRequest::is_answered) {
            let input = InputState {
                requests,
                request_state: None,
                task: Some(task.to_h()),
            };
            return Err(InputRequiredError::new(&self.name(), input).into());
        }
        self.update_task(task, &requests).await
    }

    async fn update_task(&self, task: &mut Task, requests: &[InputRequest]) -> Result<()> {
        let responses = responses(requests);
        let keys: Vec<String> = responses.keys().cloned().collect();
        self.0
            .client
            .request(
                "tasks/update",
                json!({ "taskId": task.id, "inputResponses": responses }),
                &[],
                &mut |_| {},
            )
            .await?;
        task.record_answers(keys);
        Ok(())
    }

    /// `finish(task)`: waits for the task of a direct call, cancelling it if waiting fails.
    async fn finish(&self, mut task: Task) -> Result<Value> {
        let flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let waited = match task.wait(None, None).await {
            Ok(task) => Ok(task
                .data()
                .get("result")
                .cloned()
                .unwrap_or_else(|| json!({}))),
            Err(e) => Err(e),
        };
        if waited.is_err() {
            let cancelled = progress::watch(flag, task.cancel()).await;
            if let Err(e) = cancelled {
                tracing::debug!("{} could not cancel task {}: {e}", self.name(), task.id);
            }
        }
        waited
    }

    /// `instructions`: what the server says about using it.
    pub async fn instructions(&self) -> Result<Option<String>> {
        let server = self.0.client.server().await?;
        Ok(server
            .get("instructions")
            .and_then(Value::as_str)
            .map(str::to_string))
    }

    /// `version`: the version the server reports for itself.
    pub async fn version(&self) -> Result<Option<String>> {
        let server = self.0.client.server().await?;
        let info = server
            .get("serverInfo")
            .or_else(|| server.pointer("/_meta/io.modelcontextprotocol~1serverInfo"));
        Ok(info
            .and_then(|i| i.get("version"))
            .and_then(Value::as_str)
            .map(str::to_string))
    }

    /// `prefix`: what `prefix` declared, or `None`.
    pub fn prefix(&self) -> Option<&str> {
        self.0.settings.prefix.as_deref()
    }

    /// `oauth_settings`: what `oauth` declared, or `None`.
    pub fn oauth_settings(&self) -> Option<&OAuthSettings> {
        self.0.settings.oauth.as_ref()
    }

    /// `MCP#oauth`: this server's OAuth for its owner. Fails with `Error::Configuration` for an
    /// MCP declared without `oauth`, and `Error::Argument` when a declared owner is `None`.
    pub fn oauth(&self) -> Result<Arc<OAuth>> {
        let authorizer =
            self.0.authorizer.as_ref().ok_or_else(|| {
                Error::Configuration(format!("{} does not use OAuth", self.name()))
            })?;
        authorizer.oauth()
    }

    /// `authorized?`: whether the owner has authorized this server.
    pub async fn is_authorized(&self) -> Result<bool> {
        self.oauth()?.is_authorized().await
    }

    /// `authorization_url(redirect_uri:)`: where to send the user so they can authorize this
    /// server. The authorization server redirects back to `redirect_uri`, whose parameters go to
    /// [`Mcp::authorize`]. Asks the server first, unless it already answered with a challenge,
    /// so the challenge's metadata URL and scopes are used.
    pub async fn authorization_url(&self, redirect_uri: &str) -> Result<String> {
        let oauth = self.oauth()?;
        let challenge = match self.0.authorizer.as_ref().and_then(|a| a.challenge()) {
            Some(challenge) => Some(challenge),
            None => self.challenge().await?,
        };
        oauth.authorization_url(redirect_uri, challenge).await
    }

    /// `authorize(params)`: completes an authorization with the parameters of the callback
    /// request. Fails with `Error::Mcp` when the callback does not match the authorization that
    /// `authorization_url` started.
    pub async fn authorize<K: Into<String>, V: Into<String>>(
        &self,
        params: impl IntoIterator<Item = (K, V)>,
    ) -> Result<&Self> {
        let params = params
            .into_iter()
            .map(|(k, v)| (k.into(), v.into()))
            .collect();
        self.oauth()?.authorize(&params).await?;
        Ok(self)
    }

    /// `deauthorize`: forgets the owner's credentials for this server.
    pub async fn deauthorize(&self) -> Result<&Self> {
        self.oauth()?.deauthorize().await?;
        Ok(self)
    }

    /// `challenge`: `nil` when the server answers, the challenge it sent when it wants
    /// credentials.
    async fn challenge(&self) -> Result<Option<Challenge>> {
        match self.0.client.server().await {
            Ok(_) => Ok(None),
            Err(Error::Unauthorized(..)) => {
                Ok(self.0.authorizer.as_ref().and_then(|a| a.challenge()))
            }
            Err(e) => Err(e),
        }
    }

    /// `close`: closes the connection, stopping a stdio server's process. The next request
    /// reconnects.
    pub async fn close(&self) {
        if let Some(listener) = self.0.listener.get() {
            listener.stop().await;
        }
        self.0.client.close().await;
    }

    /// `request(method, params, input:)`: answers `input_required` results with the
    /// `before_input_request` callbacks, up to `INPUT_ROUNDS` times.
    async fn request(
        &self,
        method: &str,
        params: Value,
        input: Option<InputState>,
    ) -> Result<Value> {
        let mut result = match &input {
            Some(input) => self.send_answers(method, &params, input).await?,
            None => self.send_request(method, params.clone()).await?,
        };
        for _ in 0..INPUT_ROUNDS {
            if result.get("resultType").and_then(Value::as_str) != Some("input_required") {
                return Ok(result);
            }
            let input = InputState {
                requests: self.input_requests(&result),
                request_state: result.get("requestState").cloned(),
                task: None,
            };
            if !input.requests.iter().all(InputRequest::is_answered) {
                return Err(InputRequiredError::new(&self.name(), input).into());
            }
            result = self.send_answers(method, &params, &input).await?;
        }
        Err(McpError::new(format!("{} kept asking for input", self.name())).into())
    }

    async fn send_answers(
        &self,
        method: &str,
        params: &Value,
        input: &InputState,
    ) -> Result<Value> {
        let mut params = params.clone();
        params["inputResponses"] = Value::Object(responses(&input.requests));
        if let Some(state) = &input.request_state {
            params["requestState"] = state.clone();
        }
        self.send_request(method, params).await
    }

    fn input_requests(&self, result: &Value) -> Vec<InputRequest> {
        let Some(requests) = result.get("inputRequests").and_then(Value::as_object) else {
            return Vec::new();
        };
        requests
            .iter()
            .filter(|(_, r)| r.get("method").and_then(Value::as_str) == Some("elicitation/create"))
            .map(|(key, r)| {
                let mut request = InputRequest::new(
                    key.clone(),
                    r.get("params").cloned().unwrap_or_else(|| json!({})),
                );
                let kind = if request.is_url() {
                    InputKind::Url
                } else {
                    InputKind::Form
                };
                if !self.0.settings.input_requests.contains(&kind) {
                    request.decline();
                    return request;
                }
                for callback in &self.0.settings.before_input_request {
                    if !request.is_answered() {
                        callback(&mut request);
                    }
                }
                request
            })
            .collect()
    }

    /// `send_request`: asks for progress (with a fresh `progressToken`) only when someone
    /// listens, and for log messages only when a level is set.
    async fn send_request(&self, method: &str, mut params: Value) -> Result<Value> {
        let headers = if method == "tools/call" {
            self.mirrored_headers(&params).await?
        } else {
            Vec::new()
        };
        let listeners = self.progress_listeners();
        let level = self.0.settings.log_level;
        if listeners.is_empty() && level.is_none() {
            return self
                .0
                .client
                .request(method, params, &headers, &mut |_| {})
                .await;
        }
        let token = (!listeners.is_empty()).then(|| uuid::Uuid::new_v4().to_string());
        let mut meta = Map::new();
        if let Some(token) = &token {
            meta.insert("progressToken".into(), token.clone().into());
        }
        if let Some(level) = level {
            meta.insert(
                "io.modelcontextprotocol/logLevel".into(),
                level.as_str().into(),
            );
        }
        params["_meta"] = Value::Object(meta);
        let name = self.name();
        let mut on_notification = |notification: &Value| {
            let data = notification
                .get("params")
                .cloned()
                .unwrap_or_else(|| json!({}));
            match notification.get("method").and_then(Value::as_str) {
                Some("notifications/progress") => {
                    report_progress(&data, token.as_deref(), &listeners)
                }
                Some("notifications/message") => log(&name, &data, level),
                _ => {}
            }
        };
        self.0
            .client
            .request(method, params, &headers, &mut on_notification)
            .await
    }

    fn progress_listeners(&self) -> Vec<ProgressCallback> {
        let mut listeners: Vec<ProgressCallback> = self.0.settings.after_progress.clone();
        listeners.extend(progress::listener());
        listeners
    }

    fn listener(&self) -> &Listener {
        self.0.listener.get_or_init(|| {
            let weak = Arc::downgrade(&self.0);
            let resumed_weak = weak.clone();
            Listener::new(
                self.0.client.clone(),
                self.name(),
                self.0.timeout,
                Some(listener::callback(move |listened| {
                    let mcp = resumed_weak.upgrade().map(Mcp);
                    async move {
                        if let Some(mcp) = mcp {
                            mcp.caught_up(&listened).await;
                        }
                    }
                })),
                listener::callback(move |notification| {
                    let mcp = weak.upgrade().map(Mcp);
                    async move {
                        if let Some(mcp) = mcp {
                            mcp.changed(&notification).await;
                        }
                    }
                }),
            )
        })
    }

    /// `changed(notification)`: forgets the tools a change makes stale and runs `after_change`.
    async fn changed(&self, notification: &Value) {
        let method = notification.get("method").and_then(Value::as_str);
        let change = match method {
            Some(client::ACKNOWLEDGED) => {
                self.forget_tools();
                return;
            }
            Some("notifications/tools/list_changed") => {
                self.forget_tools();
                Change::Tools
            }
            Some("notifications/prompts/list_changed") => Change::Prompts,
            Some("notifications/resources/list_changed") => Change::Resources,
            Some("notifications/resources/updated") => {
                let uri = notification.pointer("/params/uri").cloned();
                Change::Resource(Resource::new(self.clone(), json!({ "uri": uri })))
            }
            Some("notifications/tasks") => {
                let mut data = notification.get("params").cloned().unwrap_or_default();
                if let Some(map) = data.as_object_mut() {
                    map.remove("_meta");
                }
                Change::Task(Task::new(Some(self.clone()), data))
            }
            _ => return,
        };
        self.announce(change).await;
    }

    async fn announce(&self, change: Change) {
        for callback in &self.0.settings.after_change {
            callback(change.clone()).await;
        }
    }

    /// `caught_up(listened)`: after a listener resumes, everything it listens to may have
    /// changed.
    async fn caught_up(&self, listened: &Value) {
        for (key, change) in [
            ("toolsListChanged", Change::Tools),
            ("promptsListChanged", Change::Prompts),
            ("resourcesListChanged", Change::Resources),
        ] {
            if listened.get(key).and_then(Value::as_bool) == Some(true) {
                self.announce(change).await;
            }
        }
        for uri in listened
            .get("resourceSubscriptions")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let resource = Resource::new(self.clone(), json!({ "uri": uri }));
            self.announce(Change::Resource(resource)).await;
        }
        for id in listened
            .get("taskIds")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
        {
            match self.poll_task(id).await {
                Ok(data) => {
                    self.announce(Change::Task(Task::new(Some(self.clone()), data)))
                        .await
                }
                Err(e) => tracing::error!("{} could not check on task {id}: {e}", self.name()),
            }
        }
    }

    async fn listened_changes(&self, uris: &[&str], ids: &[&str]) -> Result<Value> {
        let server = self.0.client.server().await?;
        let mut changes = Map::new();
        for list in ["tools", "prompts", "resources"] {
            let listed = server
                .pointer(&format!("/capabilities/{list}/listChanged"))
                .and_then(Value::as_bool)
                == Some(true);
            if listed {
                changes.insert(format!("{list}ListChanged"), true.into());
            }
        }
        if !uris.is_empty() {
            changes.insert("resourceSubscriptions".into(), json!(uris));
        }
        if !ids.is_empty() {
            changes.insert("taskIds".into(), json!(ids));
        }
        Ok(Value::Object(changes))
    }

    /// `forget_tools`: the next `tools` lists them again.
    fn forget_tools(&self) {
        self.0.tool_changes.fetch_add(1, Ordering::SeqCst);
        if let Ok(mut cache) = self.0.server_tools.lock() {
            *cache = None;
        }
    }

    async fn mirrored_headers(&self, params: &Value) -> Result<Vec<(String, String)>> {
        let definitions = self.server_tools().await?;
        let name = params.get("name").and_then(Value::as_str);
        let empty = Map::new();
        let arguments = params
            .get("arguments")
            .and_then(Value::as_object)
            .unwrap_or(&empty);
        Ok(definitions
            .iter()
            .find(|d| d.get("name").and_then(Value::as_str) == name)
            .map(|d| param_headers::headers_for(d, arguments))
            .unwrap_or_default())
    }

    fn shape(&self, definition: &Value) -> Option<Arc<McpTool>> {
        let settings = &self.0.settings;
        let name = definition.get("name").and_then(Value::as_str).unwrap_or("");
        if settings
            .only
            .as_ref()
            .is_some_and(|only| !only.iter().any(|n| n == name))
            || settings.except.iter().any(|n| n == name)
        {
            return None;
        }
        let shape = settings
            .shapes
            .iter()
            .filter(|(n, _)| n == name)
            .fold(ToolShape::default(), |acc, (_, s)| acc.merge(s));
        Some(Arc::new(McpTool::new(
            self.clone(),
            definition,
            settings.prefix.as_deref(),
            shape,
        )))
    }

    fn check_declared_tools(&self, definitions: &[Value]) -> Result<()> {
        let settings = &self.0.settings;
        let mut declared: Vec<&String> = settings.only.iter().flatten().collect();
        declared.extend(settings.approvals.iter().flat_map(|(names, _)| names));
        declared.extend(settings.shapes.iter().map(|(name, _)| name));
        let offered: Vec<&str> = definitions
            .iter()
            .filter_map(|d| d.get("name").and_then(Value::as_str))
            .collect();
        let mut missing: Vec<&str> = Vec::new();
        for name in declared {
            if !offered.contains(&name.as_str()) && !missing.contains(&name.as_str()) {
                missing.push(name);
            }
        }
        if missing.is_empty() {
            return Ok(());
        }
        Err(Error::Configuration(format!(
            "{} declares {}, which the server does not offer",
            self.name(),
            missing.join(", ")
        )))
    }

    async fn server_tools(&self) -> Result<Vec<Value>> {
        if let Some(cached) = self.0.server_tools.lock().ok().and_then(|c| c.clone()) {
            return Ok(cached);
        }
        let changes = self.0.tool_changes.load(Ordering::SeqCst);
        let definitions: Vec<Value> = self
            .0
            .client
            .list("tools/list", "tools")
            .await?
            .into_iter()
            .filter(param_headers::is_valid)
            .collect();
        // `remember`: a list the server changed while sending it is not kept.
        if let Ok(mut cache) = self.0.server_tools.lock()
            && changes == self.0.tool_changes.load(Ordering::SeqCst)
        {
            *cache = Some(definitions.clone());
        }
        Ok(definitions)
    }
}

/// What a tool call came back with: the server's result, or the task it runs the call as.
enum Outcome {
    Result(Value),
    Task(Box<Task>),
}

/// `responses(requests)`: each request's answer, by key.
fn responses(requests: &[InputRequest]) -> Map<String, Value> {
    requests
        .iter()
        .map(|r| (r.key.clone(), r.response.clone().unwrap_or(Value::Null)))
        .collect()
}

fn report_progress(data: &Value, token: Option<&str>, listeners: &[ProgressCallback]) {
    if token.is_none() || data.get("progressToken").and_then(Value::as_str) != token {
        return;
    }
    let progress = Progress {
        value: data.get("progress").and_then(Value::as_f64),
        total: data.get("total").and_then(Value::as_f64),
        message: data
            .get("message")
            .and_then(Value::as_str)
            .map(str::to_string),
    };
    for listener in listeners {
        listener(&progress);
    }
}

/// `log(data, minimum)`: a server's log message at the level asked for or above.
fn log(name: &str, data: &Value, minimum: Option<LogLevel>) {
    let Some(minimum) = minimum else {
        return;
    };
    let Some(level) = data
        .get("level")
        .and_then(Value::as_str)
        .and_then(|l| LogLevel::parse(l).ok())
    else {
        return;
    };
    if level < minimum {
        return;
    }
    let text = match data.get("data") {
        Some(Value::String(text)) => text.clone(),
        Some(other) => other.to_string(),
        None => "null".into(),
    };
    let logger = data
        .get("logger")
        .and_then(Value::as_str)
        .map(|l| format!(" ({l})"))
        .unwrap_or_default();
    level.log(&format!("{name}{logger}: {text}"));
}

/// Seconds as Ruby prints a Float or an Integer: `0.05`, `30`.
fn seconds(duration: Duration) -> String {
    let secs = duration.as_secs_f64();
    if secs.fract() == 0.0 {
        format!("{}", secs as u64)
    } else {
        format!("{secs}")
    }
}

/// `Support::Cancellation.pause`: sleeps, stopping early with `Error::Cancelled` when the
/// surrounding chat is cancelled.
async fn pause_cancellable(duration: Duration) -> Result<()> {
    let deadline = Instant::now() + duration;
    loop {
        if progress::is_cancelled() {
            return Err(Error::Cancelled);
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Ok(());
        }
        tokio::time::sleep(remaining.min(Duration::from_millis(50))).await;
    }
}
