//! Port of `lib/ruby_llm/chat.rb`: the conversation loop.
//!
//! ```ruby
//! chat = RubyLLM.chat(model: "claude-haiku-4-5").with_tools(Weather)
//! chat.ask "What's the weather in Berlin?"
//! ```
//!
//! ```no_run
//! # use rust_llm::{Tool, ToolCall, ToolError, ToolResult};
//! # struct Weather;
//! # #[async_trait::async_trait]
//! # impl Tool for Weather {
//! #     fn description(&self) -> String { "Gets the weather".into() }
//! #     async fn execute(&self, _: serde_json::Map<String, serde_json::Value>, _: &ToolCall) -> Result<ToolResult, ToolError> { Ok("15C".into()) }
//! # }
//! # async fn run() -> rust_llm::Result<()> {
//! let mut chat = rust_llm::chat_with("claude-haiku-4-5")?.with_tool(Weather);
//! chat.ask("What's the weather in Berlin?").await?;
//! # Ok(()) }
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use serde_json::{Map, Value};

use crate::attachment::Attachment;
use crate::config::Config;
use crate::cost::{Cost, Tier};
use crate::error::{DEFAULT_FALLBACK_ERRORS, Error, ErrorKind, Result};
use crate::message::{FinishReason, Message, Operation, Role, ToolCall, UsageEntry, UsageStatus};
use crate::model::Model;
use crate::models;
use crate::protocols::{
    self, Caching, Request, Schema, StreamAccumulator, StreamState, ToolCalls, ToolChoice,
    ToolPrefs,
};
use crate::providers::{ProtocolName, Provider};
use crate::thinking::ThinkingConfig;
use crate::tokens::Tokens;
use crate::tool::{SharedTool, Tool, ToolResult, validate_arguments};
use crate::transport::Connection;

type MessageCallback = Box<dyn FnMut(&Message) + Send + Sync>;
type UnitCallback = Box<dyn FnMut() + Send + Sync>;
type ToolCallCallback = Box<dyn FnMut(&ToolCall) + Send + Sync>;
type ToolResultCallback = Box<dyn FnMut(&ToolResult) + Send + Sync>;
type FallbackCallback = Box<dyn FnMut(&FallbackAttempt) + Send + Sync>;
type RequestCallback = Box<dyn FnMut(&mut Value) + Send + Sync>;
/// A `convert_unsupported_attachments` block: a replacement, `Ok(None)` to keep the error, or
/// the application's own error.
type AttachmentConverter = Box<dyn Fn(&Attachment) -> Result<Option<Attachment>> + Send + Sync>;
/// Called with each persisted-state change; the Loco integration uses these to write rows.
pub type UsageRecorder = Box<dyn FnMut(&UsageEntry) + Send + Sync>;
/// An async `usage_recorder` (Ruby's tracker `on_finish`): awaited for each finished attempt
/// *before* its `usage.rust_llm` event is published, so persistence can write the row first.
pub type AsyncUsageRecorder = Arc<
    dyn Fn(UsageEntry) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>>
        + Send
        + Sync,
>;

/// A model to try when generation fails (`with_fallbacks`).
#[derive(Debug, Clone)]
pub struct Fallback {
    pub model: String,
    pub provider: Option<String>,
}

impl From<&str> for Fallback {
    fn from(model: &str) -> Self {
        Fallback {
            model: model.into(),
            provider: None,
        }
    }
}

/// What `before_fallback`/`after_fallback` receive.
#[derive(Debug, Clone)]
pub struct FallbackAttempt {
    pub attempt: usize,
    pub error: String,
    /// The class of `error` (`fallback.error.is_a?(ServerError)`).
    pub error_kind: ErrorKind,
    pub from: String,
    pub to: String,
    /// `fallback.to.provider`.
    pub to_provider: String,
    pub streaming: bool,
    pub chunks_yielded: bool,
    pub succeeded: Option<bool>,
    /// `fallback.response`: what the fallback model answered, once it did.
    pub response: Option<Message>,
    /// `fallback.fallback_error`: why the fallback attempt itself failed (`failed?`).
    pub fallback_error: Option<(ErrorKind, String)>,
}

#[derive(Default)]
struct Callbacks {
    before_message: Vec<UnitCallback>,
    after_message: Vec<MessageCallback>,
    before_tool_call: Vec<ToolCallCallback>,
    after_tool_result: Vec<ToolResultCallback>,
    before_fallback: Vec<FallbackCallback>,
    after_fallback: Vec<FallbackCallback>,
    /// Behind a lock so `render(&self)` can apply them, as `Chat#render` does.
    before_request: std::sync::Mutex<Vec<RequestCallback>>,
    /// `@callbacks[:unsupported_attachment]`.
    unsupported_attachment: Vec<AttachmentConverter>,
    /// `@attachment_replacements` (`compare_by_identity`): each original's replacement, so a
    /// converter runs once per attachment and its replacement keeps its upload. Behind a lock so
    /// `render(&self)` can fill it.
    attachment_replacements: std::sync::Mutex<Vec<(Attachment, Attachment)>>,
    /// Shared with the running tool's progress listener, which reports while `self` is borrowed.
    after_tool_progress: Arc<std::sync::Mutex<Vec<ToolProgressCallback>>>,
}

type ToolProgressCallback = Box<dyn FnMut(&ToolCall, &crate::progress::Progress) + Send + Sync>;

/// `cancellation_checker=`: returns true when the chat should stop.
pub type CancellationChecker = Arc<dyn Fn() -> bool + Send + Sync>;

/// `RubyLLM::Chat`.
pub struct Chat {
    config: Arc<Config>,
    model: Model,
    provider: Provider,
    protocol: Option<ProtocolName>,
    connection: Connection,
    messages: Vec<Message>,
    tools: Vec<SharedTool>,
    tool_prefs: ToolPrefs,
    temperature: Option<f64>,
    max_output_tokens: Option<i64>,
    schema: Option<Schema>,
    thinking: Option<ThinkingConfig>,
    citations: bool,
    /// `@caching`: `None` when not configured.
    caching: Option<Caching>,
    /// `@compaction`: `false`, the options object, or `None` when not configured.
    compaction: Option<Value>,
    /// `@end_user`.
    end_user: Option<String>,
    provider_options: Value,
    headers: Vec<(String, String)>,
    fallbacks: Vec<Fallback>,
    fallback_errors: Vec<ErrorKind>,
    callbacks: Callbacks,
    usage_entries: Vec<UsageEntry>,
    usage_recorder: Option<UsageRecorder>,
    async_usage_recorder: Option<AsyncUsageRecorder>,
    tool_call_decisions: HashMap<String, bool>,
    /// `@tool_deferrals`: per registered tool name, whether it is deferred (`with_tools(defer:)`).
    tool_deferrals: HashMap<String, bool>,
    cancelled: Arc<AtomicBool>,
    /// `@cancellation_checker`: an outside signal (e.g. a persisted record) consulted with the flag.
    cancellation_checker: Option<CancellationChecker>,
    mcp: crate::mcp::Collection,
    /// `@mcp_deferrals`: per server name, `with_mcp(defer:)`; `None` follows the server.
    mcp_deferrals: HashMap<String, Option<bool>>,
    /// `@tool_call_inputs`: per tool call id, the paused state (`InputRequiredError#to_h`).
    tool_call_inputs: HashMap<String, Value>,
    provider_tools: Vec<crate::provider_tools::ProviderTool>,
    /// `@concurrency`: run a response's tool calls at once (`config.tool_concurrency`).
    concurrency: bool,
}

impl std::fmt::Debug for Chat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut f = f.debug_struct("Chat");
        f.field("model", &self.model.id)
            .field("provider", &self.provider.slug())
            .field("messages", &self.messages.len())
            .field(
                "tools",
                &self.tools.iter().map(|t| t.name()).collect::<Vec<_>>(),
            );
        // `inspect_attributes` `awaiting_approval:`, omitted when empty like `Inspectable`.
        let mut awaiting: Vec<String> = self
            .pending_approvals()
            .into_iter()
            .map(|c| c.name)
            .collect();
        awaiting.dedup();
        if !awaiting.is_empty() {
            f.field("awaiting_approval", &awaiting);
        }
        f.finish()
    }
}

/// Handle to cancel a running chat from another task (`chat.cancel`).
#[derive(Clone)]
pub struct CancelHandle(Arc<AtomicBool>);

impl CancelHandle {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

/// `Models.resolve`: find the model and provider, honoring `assume_model_exists` and local providers.
pub(crate) fn resolve_model(
    model_id: &str,
    provider: Option<&str>,
    assume_model_exists: bool,
) -> Result<(Model, Provider)> {
    let provider_kind = provider.map(Provider::resolve_or_err).transpose()?;
    let assume = assume_model_exists || provider_kind.is_some_and(|p| p.assume_models_exist());
    if assume {
        let provider_kind = provider_kind.ok_or_else(|| {
            Error::Argument("Provider must be specified if assume_model_exists is true".into())
        })?;
        let model = models::models()
            .find(model_id, Some(provider_kind.slug()))
            .unwrap_or_else(|_| Model::default_for(model_id, provider_kind.slug()));
        return Ok((model, provider_kind));
    }
    let model = models::models().find(model_id, provider)?;
    let provider = Provider::resolve_or_err(&model.provider)?;
    Ok((model, provider))
}

impl Chat {
    /// `RubyLLM.chat(model:, provider:)`. `None` uses `config.default_model`.
    pub fn new(model: Option<&str>, provider: Option<&str>) -> Result<Chat> {
        Chat::with_config(crate::config(), model, provider, false)
    }

    pub fn with_config(
        config: Arc<Config>,
        model: Option<&str>,
        provider: Option<&str>,
        assume_model_exists: bool,
    ) -> Result<Chat> {
        let model_id = model.unwrap_or(&config.default_model).to_string();
        let (model, provider) = resolve_model(&model_id, provider, assume_model_exists)?;
        provider.ensure_configured(&config)?;
        let connection = Connection::new(provider, config.clone())?;
        let concurrency = config.tool_concurrency;
        Ok(Chat {
            config,
            model,
            provider,
            protocol: None,
            connection,
            messages: Vec::new(),
            tools: Vec::new(),
            tool_prefs: ToolPrefs::default(),
            temperature: None,
            max_output_tokens: None,
            schema: None,
            thinking: None,
            citations: false,
            caching: None,
            compaction: None,
            end_user: None,
            provider_options: Value::Object(Map::new()),
            headers: Vec::new(),
            fallbacks: Vec::new(),
            fallback_errors: DEFAULT_FALLBACK_ERRORS.to_vec(),
            callbacks: Callbacks::default(),
            usage_entries: Vec::new(),
            usage_recorder: None,
            async_usage_recorder: None,
            tool_call_decisions: HashMap::new(),
            tool_deferrals: HashMap::new(),
            cancelled: Arc::new(AtomicBool::new(false)),
            cancellation_checker: None,
            mcp: crate::mcp::Collection::default(),
            mcp_deferrals: HashMap::new(),
            tool_call_inputs: HashMap::new(),
            provider_tools: Vec::new(),
            concurrency,
        })
    }

    // ---- accessors -------------------------------------------------------------------------

    pub fn model(&self) -> &Model {
        &self.model
    }
    pub fn provider(&self) -> Provider {
        self.provider
    }
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }
    pub fn tools(&self) -> &[SharedTool] {
        &self.tools
    }
    pub fn temperature(&self) -> Option<f64> {
        self.temperature
    }
    pub fn usage_entries(&self) -> &[UsageEntry] {
        &self.usage_entries
    }
    pub fn config(&self) -> &Arc<Config> {
        &self.config
    }

    /// `chat.tokens`: everything billed across every attempt, retries and fallbacks included.
    pub fn tokens(&self) -> Tokens {
        Tokens::aggregate(self.usage_entries.iter().map(|e| &e.tokens))
    }

    /// `chat.cost`.
    pub fn cost(&self) -> Cost {
        let complete = self.usage_entries.iter().all(UsageEntry::cost_available);
        let cost = Cost::aggregate(self.usage_entries.iter().map(|e| &e.cost), complete);
        if complete {
            cost
        } else {
            cost.mark_incomplete()
        }
    }

    // ---- configuration (with_*) ------------------------------------------------------------

    /// `with_instructions(text)`: replaces earlier system messages.
    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.set_instructions(Some(instructions.into()), false, false);
        self
    }

    /// `with_instructions(text, append: true, cache_until_here:)`; `None` removes all instructions.
    pub fn set_instructions(
        &mut self,
        instructions: Option<String>,
        append: bool,
        cache_until_here: bool,
    ) -> &mut Self {
        self.set_instructions_with(instructions, append, &Value::Bool(cache_until_here))
            .expect("a Boolean is a valid cache boundary")
    }

    /// `with_instructions(text, append:, cache_until_here: true | false | { ttl: "1h" })`: the
    /// instruction becomes a cache boundary, with its own lifetime when `ttl` is given.
    pub fn set_instructions_with(
        &mut self,
        instructions: Option<String>,
        append: bool,
        cache_until_here: &Value,
    ) -> Result<&mut Self> {
        let boundary = Message::cache_boundary_options(cache_until_here)?;
        if !append {
            self.messages.retain(|m| m.role != Role::System);
        }
        if let Some(text) = instructions {
            let mut m = Message::system(text);
            if let Some(ttl) = &boundary {
                m = m.with_cache_until_here(ttl.as_deref());
            }
            self.messages.push(m);
        }
        Ok(self)
    }

    /// `with_tools(Weather)`.
    pub fn with_tool(mut self, tool: impl Tool + 'static) -> Self {
        self.add_tool(Arc::new(tool));
        self
    }

    pub fn with_tools(mut self, tools: impl IntoIterator<Item = SharedTool>) -> Self {
        for t in tools {
            self.add_tool(t);
        }
        self
    }

    pub fn add_tool(&mut self, tool: SharedTool) -> &mut Self {
        self.add_tool_deferred(tool, None)
    }

    /// `with_tools(tool, defer:)`: `Some(true)` keeps the tool's definition out of the model's
    /// context until the provider's tool search loads it, `Some(false)` offers a tool declared
    /// deferred (`Tool::is_deferred`) up front, `None` follows the tool. The latest registration
    /// of a name wins. See [`Chat::deferred_tools`].
    pub fn add_tool_deferred(&mut self, tool: SharedTool, defer: Option<bool>) -> &mut Self {
        let name = tool.name();
        self.tools.retain(|t| t.name() != name);
        self.tool_deferrals
            .insert(name, defer.unwrap_or_else(|| tool.is_deferred()));
        self.tools.push(tool);
        self
    }

    /// `with_tools(*tools, defer: true)`.
    pub fn with_deferred_tools(mut self, tools: impl IntoIterator<Item = SharedTool>) -> Self {
        for t in tools {
            self.add_tool_deferred(t, Some(true));
        }
        self
    }

    /// `with_tools(nil)`: also forgets which tools were deferred.
    pub fn clear_tools(&mut self) -> &mut Self {
        self.tools.clear();
        self.tool_deferrals.clear();
        self
    }

    /// `deferred_tools`: the tools whose definitions stay out of the model's context until the
    /// provider's tool search loads them, like [`Chat::tools`]. Providers without tool search
    /// receive them as ordinary tools.
    pub fn deferred_tools(&self) -> Vec<SharedTool> {
        let names = self.deferred_tool_names();
        self.combined_tools()
            .unwrap_or_else(|_| self.tools.clone())
            .into_iter()
            .filter(|t| names.contains(&t.name()))
            .collect()
    }

    /// `ToolSearch#deferred_tool_names`: registered tools deferred with `with_tools(defer:)`,
    /// then each server's tools, deferred by `with_mcp(defer:)` or else by the server's own
    /// `defer`.
    fn deferred_tool_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .tool_deferrals
            .iter()
            .filter(|(_, deferred)| **deferred)
            .map(|(name, _)| name.clone())
            .collect();
        for server in self.mcp.iter() {
            let explicit = self.mcp_deferrals.get(&server.name()).copied().flatten();
            let Some(Ok(tools)) = server.cached_tools() else {
                continue;
            };
            for tool in tools {
                let name = tool.name();
                let deferred = explicit.unwrap_or_else(|| {
                    server.defers(&*tool, server.server_name_of(&name).as_deref())
                });
                if deferred {
                    names.push(name);
                }
            }
        }
        names
    }

    /// `ToolSearch#request_tools`: the tool set rendered for a request, deferred tools wrapped
    /// so protocols with tool search mark them.
    fn request_tools(&self) -> Result<Vec<SharedTool>> {
        let deferred = self.deferred_tool_names();
        Ok(self
            .combined_tools()?
            .into_iter()
            .map(|t| {
                if deferred.contains(&t.name()) {
                    Arc::new(crate::tool::Deferred(t)) as SharedTool
                } else {
                    t
                }
            })
            .collect())
    }

    /// `with_mcp(server)`: gives the model the server's tools. The server is contacted when the
    /// chat first needs its tools.
    pub fn with_mcp(self, server: crate::mcp::Mcp) -> Self {
        self.with_mcp_deferred(server, None)
    }

    /// `with_mcp(server, defer:)`: `Some(true)` keeps every tool of the server out of the
    /// model's context until the provider's tool search loads it, `Some(false)` offers them up
    /// front despite the server's `defer`, `None` follows the server. See
    /// [`Chat::deferred_tools`].
    pub fn with_mcp_deferred(mut self, server: crate::mcp::Mcp, defer: Option<bool>) -> Self {
        self.mcp_deferrals.insert(server.name(), defer);
        self.mcp.push(server);
        self
    }

    /// `with_mcp(nil)`: disconnects every server, forgetting their deferrals.
    pub fn clear_mcp(&mut self) -> &mut Self {
        self.mcp = crate::mcp::Collection::default();
        self.mcp_deferrals.clear();
        self
    }

    /// `mcp`: the connected servers, readable by name (`chat.mcp().get("files")`).
    pub fn mcp(&self) -> &crate::mcp::Collection {
        &self.mcp
    }

    /// `tools` with the MCP servers' tools: registered tools followed by each server's, fetching
    /// server lists not yet fetched. Fails with `Error::Argument` when two tools share a name.
    pub async fn all_tools(&self) -> Result<Vec<SharedTool>> {
        self.load_mcp_tools().await?;
        self.combined_tools()
    }

    async fn load_mcp_tools(&self) -> Result<()> {
        for server in self.mcp.iter() {
            server.tools().await?;
        }
        Ok(())
    }

    /// `Chat#tools`, over the servers whose tool lists have been fetched. MCP tools that only
    /// the UI of an MCP App may call are left out (`hidden_from_model?`).
    fn combined_tools(&self) -> Result<Vec<SharedTool>> {
        let mut tools: Vec<SharedTool> = self
            .tools
            .iter()
            .filter(|t| t.is_model_visible())
            .cloned()
            .collect();
        for server in self.mcp.iter() {
            let Some(server_tools) = server.cached_tools() else {
                continue;
            };
            for tool in server_tools? {
                if !tool.is_model_visible() {
                    continue;
                }
                let name = tool.name();
                if tools.iter().any(|t| t.name() == name) {
                    return Err(Error::Argument(format!(
                        "Two tools are named {name}. Rename one with `tool :{name}, as:`"
                    )));
                }
                tools.push(tool);
            }
        }
        Ok(tools)
    }

    /// `with_provider_tools(:web_search, mcp: { ... })`: tools that run on the provider's
    /// servers. Entries add to any enabled earlier. Unknown aliases fail at request time with
    /// `Error::UnsupportedServerTool`.
    pub fn with_provider_tools(
        mut self,
        tools: impl IntoIterator<Item = crate::provider_tools::ProviderTool>,
    ) -> Self {
        self.provider_tools.extend(tools);
        self
    }

    /// `with_provider_tools(nil)`.
    pub fn clear_provider_tools(&mut self) -> &mut Self {
        self.provider_tools.clear();
        self
    }

    /// `provider_tools`.
    pub fn provider_tools(&self) -> &[crate::provider_tools::ProviderTool] {
        &self.provider_tools
    }

    /// `with_tool_options(choice:)`.
    pub fn with_tool_choice(mut self, choice: ToolChoice) -> Result<Self> {
        if let ToolChoice::Tool(name) = &choice
            && !self.tools.iter().any(|t| &t.name() == name)
        {
            let mut valid = vec!["auto".to_string(), "none".into(), "required".into()];
            valid.extend(self.tools.iter().map(|t| t.name()));
            return Err(Error::InvalidToolChoice(format!(
                "Invalid tool choice: {name}. Valid choices are: {}",
                valid.join(", ")
            )));
        }
        self.tool_prefs.choice = Some(choice);
        Ok(self)
    }

    /// `with_tool_options(choice: nil)`: forget the tool choice.
    pub fn clear_tool_choice(mut self) -> Self {
        self.tool_prefs.choice = None;
        self
    }

    /// `with_tool_options(calls: :one | :many | nil)`.
    pub fn with_tool_calls(mut self, calls: impl Into<Option<ToolCalls>>) -> Self {
        self.tool_prefs.calls = calls.into();
        self
    }

    /// `tool_prefs`: what `with_tool_options(choice:, calls:)` recorded.
    pub fn tool_prefs(&self) -> &protocols::ToolPrefs {
        &self.tool_prefs
    }

    /// `with_tool_options(concurrency: true | false | nil)`: run the tool calls of one response at
    /// once instead of one after another; `None` falls back to `config.tool_concurrency`. Their
    /// results are added as each call finishes. The calls share the chat's task, like RubyLLM's
    /// `:fibers` mode, so tools that block the thread should move that work to
    /// `tokio::task::spawn_blocking`.
    pub fn with_tool_concurrency(mut self, enabled: impl Into<Option<bool>>) -> Self {
        self.concurrency = enabled.into().unwrap_or(self.config.tool_concurrency);
        self
    }

    /// `concurrency`: whether tool calls run concurrently.
    pub fn concurrency(&self) -> bool {
        self.concurrency
    }

    /// `with_model(id, provider:)`.
    pub fn with_model(mut self, model_id: &str, provider: Option<&str>) -> Result<Self> {
        self.switch_model(model_id, provider, false)?;
        self.protocol = None;
        Ok(self)
    }

    /// `with_model(nil)`: back to `config.default_model`.
    pub fn with_default_model(self) -> Result<Self> {
        let default = self.config.default_model.clone();
        self.with_model(&default, None)
    }

    /// `with_model(id, provider:, assume_model_exists: true)`.
    pub fn with_assumed_model(mut self, model_id: &str, provider: &str) -> Result<Self> {
        self.switch_model(model_id, Some(provider), true)?;
        Ok(self)
    }

    fn switch_model(&mut self, model_id: &str, provider: Option<&str>, assume: bool) -> Result<()> {
        let (model, provider) = resolve_model(model_id, provider, assume)?;
        provider.ensure_configured(&self.config)?;
        self.connection = Connection::new(provider, self.config.clone())?;
        self.model = model;
        self.provider = provider;
        Ok(())
    }

    /// `with_model(..., protocol: :chat_completions)`.
    pub fn with_protocol(mut self, protocol: ProtocolName) -> Self {
        self.protocol = Some(protocol);
        self
    }

    /// `with_temperature(value)`; `None` clears it.
    pub fn with_temperature(mut self, temperature: impl Into<Option<f64>>) -> Self {
        self.temperature = temperature.into();
        self
    }

    /// `with_max_output_tokens(max)`; `None` removes the limit.
    pub fn with_max_output_tokens(mut self, max: impl Into<Option<i64>>) -> Self {
        self.max_output_tokens = max.into();
        self
    }

    /// `max_output_tokens`.
    pub fn max_output_tokens(&self) -> Option<i64> {
        self.max_output_tokens
    }

    /// `@protocol`: the protocol forced with `with_protocol`, if any.
    pub fn protocol(&self) -> Option<ProtocolName> {
        self.protocol
    }

    /// `@thinking`: what `with_thinking` set, before it is resolved against the model.
    pub fn thinking(&self) -> Option<&ThinkingConfig> {
        self.thinking.as_ref()
    }

    /// `with_thinking`, `with_thinking(false)`, `with_thinking(effort: :high)`, ...
    pub fn with_thinking(mut self, thinking: ThinkingConfig) -> Self {
        self.thinking = Some(thinking);
        self
    }

    /// `with_citations(enabled = true)`: asks the provider to back claims with quotes from
    /// attached documents (`response.citations`).
    pub fn with_citations(mut self, enabled: bool) -> Self {
        self.citations = enabled;
        self
    }

    /// `citations`: whether `with_citations` asked the provider for citations.
    pub fn citations(&self) -> bool {
        self.citations
    }

    /// `with_caching(options)`: `true` or `{}` for the provider's default prompt caching, an object
    /// such as `{ "ttl": "1h" }` or `{ "key": ..., "mode": ... }` for provider options, `{ "id": cache }`
    /// for a Gemini explicit cache, or `false` to send no cache controls and render no cache
    /// boundaries. Replaces earlier options; the provider rejects options it cannot render.
    pub fn with_caching(mut self, options: Value) -> Result<Self> {
        self.caching = Some(match options {
            Value::Bool(true) => Caching::On(Map::new()),
            Value::Bool(false) => Caching::Off,
            Value::Object(options) => Caching::On(options),
            _ => {
                return Err(Error::Argument(
                    "with_caching accepts true, false, or caching options".into(),
                ));
            }
        });
        Ok(self)
    }

    /// `caching`: the options set with `with_caching`, `Off` when disabled, `None` when unset.
    pub fn caching(&self) -> Option<&Caching> {
        self.caching.as_ref()
    }

    /// `with_compaction(options)`: provider-side context compaction. `true` or `{}` for the
    /// provider's defaults, `{ "at": tokens, "instructions": text, "pause_after": bool }` for the
    /// portable options, `false` to disable. Providers ignore the options they do not support.
    pub fn with_compaction(mut self, options: Value) -> Result<Self> {
        self.compaction = Some(match options {
            Value::Bool(true) => Value::Object(Map::new()),
            Value::Bool(false) => Value::Bool(false),
            Value::Object(options) => {
                let unsupported: Vec<String> = options
                    .keys()
                    .filter(|k| !COMPACTION_OPTIONS.contains(&k.as_str()))
                    .map(|k| format!(":{k}"))
                    .collect();
                if !unsupported.is_empty() {
                    let accepted: Vec<String> =
                        COMPACTION_OPTIONS.iter().map(|k| format!(":{k}")).collect();
                    return Err(Error::Argument(format!(
                        "with_compaction accepts {}, got {}. Provider-specific settings go through with_provider_options.",
                        accepted.join(", "),
                        unsupported.join(", ")
                    )));
                }
                Value::Object(options)
            }
            _ => {
                return Err(Error::Argument(
                    "with_compaction accepts true, false, or compaction options".into(),
                ));
            }
        });
        Ok(self)
    }

    /// `compaction`: the options set with `with_compaction`, `false` when disabled, `None` when unset.
    pub fn compaction(&self) -> Option<&Value> {
        self.compaction.as_ref()
    }

    /// `with_end_user(id)`: an opaque per-user identifier for the provider's abuse monitoring.
    /// Providers without an equivalent field omit it; `None` removes it.
    pub fn with_end_user(mut self, end_user: Option<&str>) -> Self {
        self.end_user = end_user.map(str::to_string);
        self
    }

    /// `end_user`.
    pub fn end_user(&self) -> Option<&str> {
        self.end_user.as_deref()
    }

    /// `with_context(context)`: send later requests with the context's configuration, or the
    /// global one for `None`. The model and provider stay as they are.
    pub fn with_context(mut self, context: Option<&crate::context::Context>) -> Result<Self> {
        let config = context
            .map(|c| c.config().clone())
            .unwrap_or_else(crate::config);
        self.provider.ensure_configured(&config)?;
        self.connection = Connection::new(self.provider, config.clone())?;
        self.config = config;
        Ok(self)
    }

    /// `with_schema(schema)`. Accepts a raw JSON schema or `{ name:, schema:, strict: }`.
    pub fn with_schema(mut self, schema: Value) -> Self {
        self.schema = normalize_schema(schema);
        self
    }

    /// `with_schema(ProductSchema)` from a `schemars::JsonSchema` type.
    pub fn with_schema_for<T: schemars::JsonSchema>(self) -> Self {
        let name = T::schema_name().to_string();
        let mut schema = crate::tool::schema_for::<T>();
        if let Some(obj) = schema.as_object_mut() {
            obj.remove("$defs");
        }
        self.with_schema(serde_json::json!({ "name": name, "schema": schema }))
    }

    /// `with_provider_options(...)`: deep-merged into every rendered request. Replaces options
    /// set earlier (`@provider_options = provider_options.to_h`); `Value::Null` clears them.
    pub fn with_provider_options(mut self, options: Value) -> Self {
        self.provider_options = match options {
            Value::Null => Value::Object(Map::new()),
            options => options,
        };
        self
    }

    /// `provider_options`.
    pub fn provider_options(&self) -> &Value {
        &self.provider_options
    }

    /// Deprecated RubyLLM name kept for 1.x muscle memory.
    pub fn with_params(self, params: Value) -> Self {
        self.with_provider_options(params)
    }

    /// `with_headers(headers)`: replaces headers set earlier (`@headers = headers.to_h`); an
    /// empty list clears them, like `with_headers(nil)`.
    pub fn with_headers(mut self, headers: impl IntoIterator<Item = (String, String)>) -> Self {
        self.headers = headers.into_iter().collect();
        self
    }

    /// `headers`.
    pub fn headers(&self) -> &[(String, String)] {
        &self.headers
    }

    /// `with_fallbacks("gpt-4.1-mini", "claude-haiku-4-5")`.
    pub fn with_fallbacks(mut self, models: impl IntoIterator<Item = Fallback>) -> Self {
        self.fallbacks = models.into_iter().collect();
        self.fallback_errors = DEFAULT_FALLBACK_ERRORS.to_vec();
        self
    }

    /// `with_fallbacks(..., on: [...])`.
    pub fn with_fallback_errors(mut self, kinds: Vec<ErrorKind>) -> Self {
        self.fallback_errors = kinds;
        self
    }

    /// `fallbacks`.
    pub fn fallbacks(&self) -> &[Fallback] {
        &self.fallbacks
    }

    /// `fallback_errors`.
    pub fn fallback_errors(&self) -> &[ErrorKind] {
        &self.fallback_errors
    }

    // ---- callbacks -------------------------------------------------------------------------

    pub fn before_message(mut self, f: impl FnMut() + Send + Sync + 'static) -> Self {
        self.callbacks.before_message.push(Box::new(f));
        self
    }
    pub fn after_message(mut self, f: impl FnMut(&Message) + Send + Sync + 'static) -> Self {
        self.callbacks.after_message.push(Box::new(f));
        self
    }
    pub fn before_tool_call(mut self, f: impl FnMut(&ToolCall) + Send + Sync + 'static) -> Self {
        self.callbacks.before_tool_call.push(Box::new(f));
        self
    }
    pub fn after_tool_result(mut self, f: impl FnMut(&ToolResult) + Send + Sync + 'static) -> Self {
        self.callbacks.after_tool_result.push(Box::new(f));
        self
    }
    /// `after_tool_progress { |tool_call, progress| ... }`: what a running tool reports, including
    /// an MCP server's progress notifications.
    pub fn after_tool_progress(
        self,
        f: impl FnMut(&ToolCall, &crate::progress::Progress) + Send + Sync + 'static,
    ) -> Self {
        if let Ok(mut callbacks) = self.callbacks.after_tool_progress.lock() {
            callbacks.push(Box::new(f));
        }
        self
    }
    pub fn before_fallback(
        mut self,
        f: impl FnMut(&FallbackAttempt) + Send + Sync + 'static,
    ) -> Self {
        self.callbacks.before_fallback.push(Box::new(f));
        self
    }
    pub fn after_fallback(
        mut self,
        f: impl FnMut(&FallbackAttempt) + Send + Sync + 'static,
    ) -> Self {
        self.callbacks.after_fallback.push(Box::new(f));
        self
    }
    /// `convert_unsupported_attachments { |attachment| ... }`: replaces an attachment the current
    /// protocol cannot render. The block returns a replacement, `Ok(None)` to keep the
    /// `UnsupportedAttachment` error, or an error of its own, which the request raises. Blocks
    /// run in registration order until one returns a replacement. The transcript keeps the
    /// original file, even when the model changes; each attachment is converted once and its
    /// replacement (and that replacement's upload) reused on later requests.
    pub fn convert_unsupported_attachments(
        mut self,
        f: impl Fn(&Attachment) -> Result<Option<Attachment>> + Send + Sync + 'static,
    ) -> Self {
        self.callbacks.unsupported_attachment.push(Box::new(f));
        self
    }

    /// `before_request { |payload| ... }`: last chance to edit the rendered payload.
    pub fn before_request(self, f: impl FnMut(&mut Value) + Send + Sync + 'static) -> Self {
        if let Ok(mut hooks) = self.callbacks.before_request.lock() {
            hooks.push(Box::new(f));
        }
        self
    }

    pub fn set_usage_recorder(&mut self, recorder: UsageRecorder) {
        self.usage_recorder = Some(recorder);
    }

    /// Registers an async recorder awaited for each finished attempt before its usage event is
    /// published (Ruby persists each attempt through the tracker's `on_finish` first).
    pub fn set_async_usage_recorder(&mut self, recorder: AsyncUsageRecorder) {
        self.async_usage_recorder = Some(recorder);
    }

    pub fn clear_async_usage_recorder(&mut self) {
        self.async_usage_recorder = None;
    }

    // ---- messages --------------------------------------------------------------------------

    /// `add_message`: append without calling the model.
    pub fn add_message(&mut self, message: Message) -> &Message {
        self.messages.push(message);
        self.messages.last().unwrap() // pushed just above
    }

    /// `add_completion(response, record_usage:)`: append an answer produced outside `complete`
    /// (e.g. by a batch), recording its usage and running the message callbacks.
    pub fn add_completion(&mut self, mut message: Message, record_usage: bool) -> &Message {
        if message.usage_entries.is_empty() {
            let entry = UsageEntry {
                id: UsageEntry::next_id(),
                owner: crate::accounting::usage_owner(),
                operation: Operation::Chat,
                provider: self.provider.slug().into(),
                model: message
                    .model
                    .clone()
                    .unwrap_or_else(|| self.model.id.clone()),
                status: UsageStatus::Succeeded,
                tokens: message.tokens.clone(),
                cost: message.cost(None),
            };
            message.usage_entries = vec![entry.clone()];
            self.record_usage_now(entry);
        } else if record_usage {
            for entry in message.usage_entries.clone() {
                self.record_usage_now(entry);
            }
        }
        for cb in &mut self.callbacks.before_message {
            cb();
        }
        self.messages.push(message);
        let message = self.messages.last().unwrap(); // pushed just above
        for cb in &mut self.callbacks.after_message {
            cb(message);
        }
        message
    }

    /// Replaces history, e.g. when reloading a persisted chat.
    pub fn set_messages(&mut self, messages: Vec<Message>) {
        self.messages = messages;
    }

    pub fn set_usage_entries(&mut self, entries: Vec<UsageEntry>) {
        self.usage_entries = entries;
    }

    /// `cache_until_here`: mark the last message as a prompt-cache boundary.
    pub fn cache_until_here(&mut self) -> Result<&mut Self> {
        self.cache_until_here_with(None)
    }

    /// `cache_until_here(ttl:)`: the boundary gets its own cache lifetime (`"1h"`), rendered ahead
    /// of the `with_caching(ttl:)` lifetime on providers that set one per boundary.
    pub fn cache_until_here_with(&mut self, ttl: Option<&str>) -> Result<&mut Self> {
        let last = self
            .messages
            .last_mut()
            .ok_or_else(|| Error::Argument("No messages to cache".into()))?;
        last.cache_until_here = true;
        last.cache_ttl = ttl.map(str::to_string);
        Ok(self)
    }

    // ---- the loop --------------------------------------------------------------------------

    /// `ask(message)`: stage a user message, then run the loop until the model answers.
    pub async fn ask(&mut self, message: impl Into<String>) -> Result<Message> {
        self.ask_later(message)?;
        self.complete().await
    }

    /// `ask(message, with: [...])`.
    pub async fn ask_with(
        &mut self,
        message: impl Into<String>,
        attachments: Vec<Attachment>,
    ) -> Result<Message> {
        self.ask_later_with(message, attachments)?;
        self.complete().await
    }

    /// `ask(message) { |chunk| ... }`.
    pub async fn ask_stream(
        &mut self,
        message: impl Into<String>,
        on_chunk: impl FnMut(&Message) + Send,
    ) -> Result<Message> {
        self.ask_later(message)?;
        self.complete_stream(on_chunk).await
    }

    /// `say` is an alias of `ask`.
    pub async fn say(&mut self, message: impl Into<String>) -> Result<Message> {
        self.ask(message).await
    }

    /// `ask_later`: stage without requesting a completion.
    pub fn ask_later(&mut self, message: impl Into<String>) -> Result<&mut Self> {
        self.ask_later_with(message, Vec::new())
    }

    pub fn ask_later_with(
        &mut self,
        message: impl Into<String>,
        attachments: Vec<Attachment>,
    ) -> Result<&mut Self> {
        self.raise_if_pending_tool_calls()?;
        self.messages
            .push(Message::user(message).with_attachments(attachments));
        Ok(self)
    }

    /// `ask(github.prompt(:code_review, code: diff))`: adds the prompt's messages, then completes.
    pub async fn ask_prompt(&mut self, prompt: &crate::mcp::Prompt) -> Result<Message> {
        self.ask_later_prompt(prompt)?;
        self.complete().await
    }

    /// `ask_later(prompt)`: stages an MCP prompt's messages.
    pub fn ask_later_prompt(&mut self, prompt: &crate::mcp::Prompt) -> Result<&mut Self> {
        self.raise_if_pending_tool_calls()?;
        self.messages.extend(prompt.messages.iter().cloned());
        Ok(self)
    }

    /// `complete`: run until `complete?` or parked on an approval.
    pub async fn complete(&mut self) -> Result<Message> {
        if self.is_awaiting_tasks() {
            self.run_tools().await?;
        }
        while !self.is_complete() && !self.waiting() {
            self.step_inner(None).await?;
        }
        Ok(self
            .last_non_system_message()
            .or_else(|| self.messages.last())
            .cloned()
            .unwrap_or_else(Message::chunk))
    }

    pub async fn complete_stream(
        &mut self,
        mut on_chunk: impl FnMut(&Message) + Send,
    ) -> Result<Message> {
        while !self.is_complete() && !self.waiting() {
            self.step_inner(Some(&mut on_chunk)).await?;
        }
        Ok(self
            .last_non_system_message()
            .or_else(|| self.messages.last())
            .cloned()
            .unwrap_or_else(Message::chunk))
    }

    /// `step`: run pending tools, or generate the next response. `None` once there is nothing to do.
    pub async fn step(&mut self) -> Result<Option<Message>> {
        self.step_inner(None).await
    }

    /// `step { |chunk| ... }`.
    pub async fn step_stream(
        &mut self,
        mut on_chunk: impl FnMut(&Message) + Send,
    ) -> Result<Option<Message>> {
        self.step_inner(Some(&mut on_chunk)).await
    }

    /// Mutable history, for persistence layers that stamp `record_id`s or drop rolled-back rows.
    pub fn messages_mut(&mut self) -> &mut Vec<Message> {
        &mut self.messages
    }

    async fn step_inner(
        &mut self,
        on_chunk: Option<&mut (dyn FnMut(&Message) + Send)>,
    ) -> Result<Option<Message>> {
        if self.is_complete() {
            return Ok(None);
        }
        if let Err(e) = self.raise_if_cancelled() {
            self.cancel_tasks().await;
            return Err(e);
        }
        if self.pending_tool_response().is_none() {
            return self.generate_inner(on_chunk).await.map(Some);
        }
        let before = self.messages.len();
        self.run_tools().await?;
        Ok((self.messages.len() > before)
            .then(|| self.messages.last().cloned())
            .flatten())
    }

    /// `complete?`: nothing staged, or the model answered without requesting tools.
    pub fn is_complete(&self) -> bool {
        if self.pending_tool_response().is_some() {
            return false;
        }
        match self.last_non_system_message() {
            None => true,
            Some(m) => match m.role {
                Role::User | Role::Tool => false,
                _ => !m.is_tool_call(),
            },
        }
    }

    /// `generate`: one completion, honoring fallbacks. Tool calls are not executed.
    pub async fn generate(&mut self) -> Result<Message> {
        self.generate_inner(None).await
    }

    async fn generate_inner(
        &mut self,
        mut on_chunk: Option<&mut (dyn FnMut(&Message) + Send)>,
    ) -> Result<Message> {
        self.raise_if_cancelled()?;
        if self.fallbacks.is_empty() {
            return self.generate_once(on_chunk, &mut false).await;
        }
        let original = (
            self.model.clone(),
            self.provider,
            self.protocol,
            self.connection.clone(),
        );
        let usage_start = self.usage_entries.len();
        let mut queue: std::collections::VecDeque<Fallback> =
            self.fallbacks.iter().cloned().collect();
        let mut attempt = 0;
        let mut active: Option<FallbackAttempt> = None;
        let result = loop {
            let mut chunks_yielded = false;
            let streaming = on_chunk.is_some();
            let result = self
                .generate_once(
                    on_chunk
                        .as_mut()
                        .map(|f| &mut **f as &mut (dyn FnMut(&Message) + Send)),
                    &mut chunks_yielded,
                )
                .await;
            if let Some(mut a) = active.take() {
                a.succeeded = Some(result.is_ok());
                match &result {
                    Ok(message) => a.response = Some(message.clone()),
                    Err(e) => a.fallback_error = Some((e.kind(), e.to_string())),
                }
                for cb in &mut self.callbacks.after_fallback {
                    cb(&a);
                }
            }
            match result {
                // `link_completion_usage(result, usage_start)`: attempts on models that failed
                // before the fallback answered belong to the answer too.
                Ok(mut message) => {
                    message.usage_entries = self.usage_entries[usage_start..].to_vec();
                    if let Some(last) = self.messages.last_mut() {
                        last.usage_entries = message.usage_entries.clone();
                    }
                    break Ok(message);
                }
                Err(e) if self.fallback_errors.contains(&e.kind()) => {
                    let Some(next) = queue.pop_front() else {
                        break Err(e);
                    };
                    attempt += 1;
                    let from = self.model.id.clone();
                    let from_provider = self.provider;
                    if let Err(switch_err) =
                        self.switch_model(&next.model, next.provider.as_deref(), false)
                    {
                        break Err(switch_err);
                    }
                    if self.provider != from_provider {
                        self.protocol = None;
                    }
                    let a = FallbackAttempt {
                        attempt,
                        error: e.to_string(),
                        error_kind: e.kind(),
                        from,
                        to: self.model.id.clone(),
                        to_provider: self.provider.slug().into(),
                        streaming,
                        chunks_yielded,
                        succeeded: None,
                        response: None,
                        fallback_error: None,
                    };
                    for cb in &mut self.callbacks.before_fallback {
                        cb(&a);
                    }
                    active = Some(a);
                }
                Err(e) => break Err(e),
            }
        };
        (self.model, self.provider, self.protocol, self.connection) = original;
        result
    }

    /// `Tracker#finish`: record the attempt (sync recorder, then any async recorder, which
    /// persistence uses to write the row), and only then publish `usage.rust_llm`.
    async fn record_usage(&mut self, entry: UsageEntry) {
        if let Some(r) = &mut self.usage_recorder {
            r(&entry);
        }
        // `@usage_recorder ? @usage_recorder.call(entry) : Accounting::Usage.record(entry)`: a
        // chat a record persists writes its own rows; any other chat goes to the ledger.
        match self.async_usage_recorder.clone() {
            Some(r) => r(entry.clone()).await,
            None => crate::accounting::record(&self.config, &entry).await,
        }
        crate::instrumentation::usage(&self.config, &entry);
        self.usage_entries.push(entry);
    }

    /// `record_usage` for the synchronous `add_completion` path: no async recorder can run here.
    fn record_usage_now(&mut self, entry: UsageEntry) {
        if let Some(r) = &mut self.usage_recorder {
            r(&entry);
        }
        crate::instrumentation::usage(&self.config, &entry);
        self.usage_entries.push(entry);
    }

    /// `preprocessed_messages`: the request history, each message as the current model may see
    /// it (`Protocol#preprocess_message`).
    fn preprocessed_messages(&self) -> Result<Vec<Message>> {
        let protocol = self
            .provider
            .resolve_protocol(self.protocol, &self.model, &self.config)
            .ok();
        self.request_history()
            .into_iter()
            .map(|m| {
                let mut m = if self.foreign_native_content(&m) {
                    // `without_foreign_native_content`: another model of this provider gets the
                    // raw content any of its models reads, such as a tool search.
                    let portable = protocol
                        .filter(|_| {
                            m.producing_entry()
                                .is_some_and(|e| e.provider == self.provider.slug())
                        })
                        .and_then(|p| crate::tool_search::portable_raw_content(p, &m));
                    m.for_request().without_native_content(portable)
                } else {
                    m.for_request()
                };
                self.replace_unsupported_attachments(&mut m)?;
                Ok(m)
            })
            .collect()
    }

    /// `Protocol#replace_unsupported_attachment` with `Chat#unsupported_attachment_handler`:
    /// swaps each attachment the protocol cannot render for the application's replacement.
    fn replace_unsupported_attachments(&self, message: &mut Message) -> Result<()> {
        if self.callbacks.unsupported_attachment.is_empty() || message.attachments.is_empty() {
            return Ok(());
        }
        let protocol = self
            .provider
            .resolve_protocol(self.protocol, &self.model, &self.config)?;
        let supported = |a: &Attachment| {
            protocols::supported_message_attachment(protocol, self.provider, message.role, a)
        };
        let mut prepared = Vec::with_capacity(message.attachments.len());
        for attachment in &message.attachments {
            if supported(attachment) {
                prepared.push(attachment.clone());
                continue;
            }
            let Some(replacement) = self.attachment_replacement(attachment)? else {
                prepared.push(attachment.clone());
                continue;
            };
            if !supported(&replacement) {
                return Err(Error::UnsupportedAttachment(
                    protocols::anthropic::unsupported(&replacement.mime_type),
                ));
            }
            prepared.push(replacement);
        }
        message.attachments = prepared;
        Ok(())
    }

    /// `@attachment_replacements[attachment] ||= callbacks.lazy.filter_map { ... }.first`.
    fn attachment_replacement(&self, attachment: &Attachment) -> Result<Option<Attachment>> {
        let memo = |replacements: &std::sync::Mutex<Vec<(Attachment, Attachment)>>| {
            replacements.lock().ok().and_then(|r| {
                r.iter()
                    .find(|(original, _)| original.same_attachment(attachment))
                    .map(|(_, replacement)| replacement.clone())
            })
        };
        if let Some(replacement) = memo(&self.callbacks.attachment_replacements) {
            return Ok(Some(replacement));
        }
        for convert in &self.callbacks.unsupported_attachment {
            if let Some(replacement) = convert(attachment)? {
                if let Ok(mut r) = self.callbacks.attachment_replacements.lock() {
                    r.push((attachment.clone(), replacement.clone()));
                }
                return Ok(Some(replacement));
            }
        }
        Ok(None)
    }

    /// `Protocol#foreign_native_content?`: a thinking signature or provider-shaped content belongs
    /// to the model that produced it; a message with no known producer replays as it is.
    fn foreign_native_content(&self, m: &Message) -> bool {
        let carries = m.thinking.is_some()
            || m.raw_reasoning.is_some()
            || m.raw_content.is_some()
            || m.tool_calls
                .iter()
                .flat_map(|c| c.values())
                .any(|c| c.thought_signature.is_some());
        if m.role != Role::Assistant || !carries {
            return false;
        }
        let Some(producer) = m.producing_entry() else {
            return false;
        };
        producer.provider != self.provider.slug() || !self.is_current_model(&producer.model)
    }

    /// `Protocol#current_model?`: usage records the resolved id, and a message built by hand may
    /// name the model by an alias of it.
    fn is_current_model(&self, model_id: &str) -> bool {
        model_id == self.model.id
            || models::models()
                .find(model_id, Some(self.provider.slug()))
                .is_ok_and(|m| m.id == self.model.id)
    }

    /// `request_history`: a process that dies mid-round leaves a blank assistant placeholder, or
    /// tool calls without results, and providers refuse both. A request leaves blank messages out
    /// and answers the calls of a round the conversation moved past as unfinished. The latest
    /// round stays as it is: the loop runs its calls or waits on them.
    fn request_history(&self) -> Vec<std::borrow::Cow<'_, Message>> {
        use std::borrow::Cow;
        let kept: Vec<&Message> = self
            .messages
            .iter()
            .filter(|m| !is_blank_response(m))
            .collect();
        let answered: Vec<&str> = kept
            .iter()
            .filter(|m| m.is_tool_result())
            .filter_map(|m| m.tool_call_id.as_deref())
            .collect();
        let current = latest_response(kept.iter().copied());
        let mut out = Vec::with_capacity(kept.len());
        let mut i = 0;
        while i < kept.len() {
            let response = kept[i];
            out.push(Cow::Borrowed(response));
            i += 1;
            while i < kept.len() && kept[i].is_tool_result() {
                out.push(Cow::Borrowed(kept[i]));
                i += 1;
            }
            if response.is_tool_call() && !current.is_some_and(|c| std::ptr::eq(c, response)) {
                for call in response.tool_calls.iter().flat_map(|c| c.values()) {
                    if !answered.contains(&call.id.as_str()) {
                        out.push(Cow::Owned(self.unfinished_result(call)));
                    }
                }
            }
        }
        out
    }

    /// `unfinished_result`: a provider-executed call gets the refusal its provider expects
    /// (`remote_refusal`); any other call the error shape tool results use.
    fn unfinished_result(&self, call: &ToolCall) -> Message {
        let responses = self
            .provider
            .resolve_protocol(self.protocol, &self.model, &self.config)
            .is_ok_and(|p| p == ProtocolName::Responses);
        if call.remote && responses {
            // `Responses::Approvals#render_tool_approval_response(approved: false)`.
            let mut m = Message::tool_result(call.id.clone(), "Denied");
            m.raw_content = Some(serde_json::json!([{
                "type": "mcp_approval_response", "approval_request_id": call.id, "approve": false
            }]));
            return m;
        }
        Message::tool_result(call.id.clone(), UNFINISHED_TOOL_RESULT)
    }

    /// `Chat#render`: the payload that would be sent, with `before_request` hooks applied,
    /// without sending it.
    pub fn render(&self) -> Result<Value> {
        self.render_with(&self.preprocessed_messages()?, false)
            .map(|(p, _)| p)
    }

    fn render_with(&self, messages: &[Message], stream: bool) -> Result<(Value, ProtocolName)> {
        let protocol = self
            .provider
            .resolve_protocol(self.protocol, &self.model, &self.config)?;
        let thinking = match &self.thinking {
            Some(t) => t.resolve(&self.model)?,
            None => None,
        };
        let tools = self.request_tools()?;
        let request = Request {
            provider: self.provider,
            config: &self.config,
            model: &self.model,
            messages,
            tools: &tools,
            tool_prefs: &self.tool_prefs,
            temperature: self.temperature,
            max_output_tokens: self.max_output_tokens,
            schema: self.schema.as_ref(),
            thinking: thinking.as_ref(),
            citations: self.citations,
            caching: self.caching.as_ref(),
            stream,
        };
        let mut payload = protocols::render(protocol, &request)?;
        if let Some(end_user) = &self.end_user {
            protocols::apply_end_user(protocol, self.provider, &mut payload, end_user);
        }
        if let Some(Value::Object(compaction)) = &self.compaction {
            protocols::apply_compaction(protocol, self.provider, &mut payload, compaction);
        }
        protocols::deep_merge(&mut payload, &self.provider_options);
        if let Some(resolution) =
            crate::provider_tools::resolve(protocol, self.provider, &self.provider_tools)?
        {
            crate::provider_tools::apply(&mut payload, &resolution);
        }
        crate::tool_search::apply(protocol, &mut payload);
        self.apply_before_request_hooks(&mut payload);
        protocols::finish_render(protocol, &mut payload)?;
        Ok((payload, protocol))
    }

    /// `Protocol#apply_before_request_hooks`.
    fn apply_before_request_hooks(&self, payload: &mut Value) {
        if let Ok(mut hooks) = self.callbacks.before_request.lock() {
            for hook in hooks.iter_mut() {
                hook(payload);
            }
        }
    }

    /// Reads what the request needs, as Ruby's lazy `Attachment#content` does: local files, a URL
    /// whose name leaves its type unknown (`Attachment.new` fetches it to detect one), and a URL
    /// only when the payload carries its bytes rather than the link. A trial render marks those.
    /// Clones share the bytes, so history keeps them (and the detected type) for later turns.
    async fn load_attachments(&mut self, messages: &mut [Message]) -> Result<()> {
        let client = self.connection.client().clone();
        for a in self
            .messages
            .iter_mut()
            .flat_map(|m| m.attachments.iter_mut())
        {
            a.prepare(&client).await?;
        }
        for a in messages.iter_mut().flat_map(|m| m.attachments.iter_mut()) {
            a.prepare(&client).await?;
        }
        loop {
            let _ = self.render_with(messages, false);
            let mut fetched = false;
            for a in messages
                .iter_mut()
                .flat_map(|m| m.attachments.iter_mut())
                .filter(|a| a.is_wanted())
            {
                a.load(&client).await?;
                fetched = true;
            }
            if !fetched {
                return Ok(());
            }
        }
    }

    /// `generate_once`: one completion inside a `chat.rust_llm` event.
    async fn generate_once(
        &mut self,
        on_chunk: Option<&mut (dyn FnMut(&Message) + Send)>,
        chunks_yielded: &mut bool,
    ) -> Result<Message> {
        self.raise_if_cancelled()?;
        let entries_before = self.usage_entries.len();
        let mut event = self.instrumentation_event("chat.rust_llm", on_chunk.is_some());
        let result = event
            .instrument(self.generate_once_inner(on_chunk, chunks_yielded))
            .await;
        // `event[:response_tokens]`: this attempt's tokens, or on failure what the attempts
        // recorded during this event were billed, so a fallback's usage stays on its own span.
        event.set("response_tokens", || {
            crate::instrumentation::tokens_h(&match &result {
                Ok(message) => message.tokens(),
                Err(_) => Tokens::aggregate(
                    self.usage_entries[entries_before.min(self.usage_entries.len())..]
                        .iter()
                        .map(|e| &e.tokens),
                ),
            })
        });
        self.finish_completion_event(event, &result);
        result
    }

    /// `instrumentation_payload(streaming:)`.
    fn instrumentation_event(&self, name: &str, streaming: bool) -> crate::instrumentation::Event {
        crate::instrumentation::Event::start(&self.config, name, || {
            let empty = Tokens::default();
            let choice = self.tool_prefs.choice.as_ref().map(|c| match c {
                ToolChoice::Auto => "auto".to_string(),
                ToolChoice::None => "none".into(),
                ToolChoice::Required => "required".into(),
                ToolChoice::Tool(name) => name.clone(),
            });
            let calls = self
                .tool_prefs
                .calls
                .map(|c| if c == ToolCalls::One { "one" } else { "many" });
            crate::instrumentation::payload([
                ("provider", self.provider.slug().into()),
                ("provider_class", self.provider.display().into()),
                ("model", self.model.id.clone().into()),
                (
                    "input_messages",
                    Value::Array(self.messages.iter().map(Message::to_h).collect()),
                ),
                ("message_count", self.messages.len().into()),
                (
                    "tools",
                    Value::Array(self.tools.iter().map(|t| t.name().into()).collect()),
                ),
                ("tool_choice", choice.into()),
                ("tool_call_limit", calls.into()),
                ("temperature", self.temperature.into()),
                ("max_output_tokens", self.max_output_tokens.into()),
                ("provider_options", self.provider_options.clone()),
                (
                    "schema",
                    self.schema.as_ref().map(|s| s.schema.clone()).into(),
                ),
                ("citations", self.citations.into()),
                ("streaming", streaming.into()),
                ("tokens", crate::instrumentation::tokens_h(&empty)),
                (
                    "cost",
                    crate::instrumentation::cost_h(&Cost::new(
                        &empty,
                        Some(&self.model),
                        Tier::Standard,
                    )),
                ),
            ])
        })
    }

    /// `record_completion_event`.
    fn finish_completion_event(
        &self,
        mut event: crate::instrumentation::Event,
        result: &Result<Message>,
    ) {
        if let Ok(message) = result {
            event.set("response", || message.to_h());
            event.set("messages_after", || {
                Value::Array(self.messages.iter().map(Message::to_h).collect())
            });
            event.set("response_role", || message.role.as_str().into());
            event.set("tokens", || {
                crate::instrumentation::tokens_h(&message.tokens())
            });
            event.set("cost", || {
                crate::instrumentation::cost_h(&message.cost(None))
            });
            event.set("response_model", || message.model.clone().into());
            event.set("tool_call", || message.is_tool_call().into());
            event.set("tool_calls", || {
                serde_json::to_value(&message.tool_calls).unwrap_or(Value::Null)
            });
        }
        event.finish(result.as_ref().err());
    }

    async fn generate_once_inner(
        &mut self,
        mut on_chunk: Option<&mut (dyn FnMut(&Message) + Send)>,
        chunks_yielded: &mut bool,
    ) -> Result<Message> {
        self.load_mcp_tools().await?;
        self.load_attachments(&mut []).await?;
        let mut messages = self.request_messages().await?;
        let streaming = on_chunk.is_some();
        if streaming {
            for cb in &mut self.callbacks.before_message {
                cb();
            }
        }
        // `Anthropic#complete`: a turn a server tool paused with `pause_turn` goes back verbatim to
        // continue, up to MAX_PAUSE_TURN_CONTINUATIONS requests, and the segments merge into one message.
        let mut segments = Vec::new();
        let mut call_entries = Vec::new();
        let mut billed_model = None;
        let mut reuploaded = false;
        let mut continuation = 0;
        while continuation < protocols::anthropic::MAX_PAUSE_TURN_CONTINUATIONS {
            let chunk_sink = on_chunk
                .as_mut()
                .map(|f| &mut **f as &mut (dyn FnMut(&Message) + Send));
            let (segment, entries, billed, protocol) = match self
                .request_once(&messages, chunk_sink, chunks_yielded)
                .await
            {
                Ok(answer) => answer,
                // `replacing_missing_uploads`: a request that failed over a file the provider
                // deleted, before streaming anything, uploads it again and runs once more.
                Err(e) => {
                    if reuploaded || *chunks_yielded || !self.discard_missing_uploads(&e).await {
                        return Err(e);
                    }
                    reuploaded = true;
                    segments.clear();
                    messages = self.request_messages().await?;
                    continuation = 0;
                    continue;
                }
            };
            continuation += 1;
            call_entries.extend(entries);
            billed_model = Some(billed);
            let paused = protocol == ProtocolName::Anthropic
                && segment.finish_reason == Some(FinishReason::PauseTurn);
            if paused {
                messages.push(segment.for_request());
            }
            segments.push(segment);
            if !paused {
                break;
            }
        }
        let mut message = protocols::anthropic::merge_turn_segments(segments);
        // `link_completion_usage`: every attempt of this call is linked to the message it produced.
        message.usage_entries = call_entries;
        message.model_info = billed_model;
        if !streaming {
            for cb in &mut self.callbacks.before_message {
                cb();
            }
        }
        self.messages.push(message.clone());
        for cb in &mut self.callbacks.after_message {
            cb(&message);
        }
        Ok(message)
    }

    /// `preprocessed_messages` plus the request-time uploads, with every attachment read.
    async fn request_messages(&mut self) -> Result<Vec<Message>> {
        let mut messages = self.preprocessed_messages()?;
        let upload_protocol =
            self.provider
                .resolve_protocol(self.protocol, &self.model, &self.config)?;
        crate::files::preprocess_messages(
            &mut messages,
            upload_protocol,
            self.provider,
            &self.config,
            &self.connection,
        )
        .await?;
        self.load_attachments(&mut messages).await?;
        Ok(messages)
    }

    /// `Chat#discard_missing_uploads`: whether the error named uploads the provider deleted
    /// (which the attachments now forget).
    async fn discard_missing_uploads(&self, error: &Error) -> bool {
        !crate::files::discard_missing_uploads(
            &self.messages,
            error,
            self.provider,
            &self.config,
            &self.connection,
        )
        .await
        .is_empty()
    }

    /// One completion request (`Protocol#complete`): renders `messages`, sends them, and records
    /// an entry per attempt. Returns the message, its attempts' entries, the billed model, and the
    /// protocol it spoke.
    async fn request_once(
        &mut self,
        messages: &[Message],
        on_chunk: Option<&mut (dyn FnMut(&Message) + Send)>,
        chunks_yielded: &mut bool,
    ) -> Result<(Message, Vec<UsageEntry>, Model, ProtocolName)> {
        let streaming = on_chunk.is_some();
        let (payload, protocol) = self.render_with(messages, streaming)?;
        let mut endpoint = protocols::endpoint(protocol, self.provider, &self.model, streaming);
        // `Perplexity::Agent#completion_url` is `@provider.agent_url`.
        if protocol == ProtocolName::Responses && self.provider == Provider::Perplexity {
            endpoint.path = self.provider.agent_url(&self.config)?;
        }
        // `Perplexity::Router#completion_url` is `@provider.router_url('chat/completions')`.
        if protocol == ProtocolName::RouterChatCompletions {
            endpoint.path = self.provider.router_url(&self.config, "chat/completions")?;
        }
        let mut headers = endpoint.headers;
        // `resolution.headers.merge(headers)`: the chat's own headers win.
        if let Some(resolution) =
            crate::provider_tools::resolve(protocol, self.provider, &self.provider_tools)?
        {
            headers.extend(
                resolution
                    .headers
                    .into_iter()
                    .filter(|(k, _)| !self.headers.iter().any(|(h, _)| h.eq_ignore_ascii_case(k))),
            );
        }
        headers.extend(self.headers.iter().cloned());
        if protocol == ProtocolName::Anthropic && matches!(self.compaction, Some(Value::Object(_)))
        {
            protocols::anthropic::apply_compaction_headers(&mut headers);
        }
        crate::files::apply_files_beta(protocol, &payload, &mut headers);

        // One entry per attempt: `retried` holds the tokens each retried attempt is billed.
        let mut attempts = 0usize;
        let mut retried: Vec<Tokens> = Vec::new();
        let mut on_attempt = |previous: Option<&Error>| {
            attempts += 1;
            if let Some(e) = previous {
                retried.push(failure_tokens(e, None));
            }
        };
        let mut observed = Tokens::default();
        let result = if let Some(on_chunk) = on_chunk {
            let mut acc = StreamAccumulator::default();
            let mut state =
                StreamState::for_payload(protocol, self.provider, &self.model, &payload);
            let provider = self.provider;
            let cancelled = self.cancelled.clone();
            let checker = self.cancellation_checker.clone();
            let mut on_event = |_event: crate::transport::SseEvent, data: Value| -> Result<()> {
                let chunk = protocols::build_chunk(protocol, provider, &mut state, &data)?;
                acc.add(&chunk);
                *chunks_yielded = true;
                on_chunk(&chunk);
                // `raise_if_cancelled!` inside the streaming block: one-shot, so it clears the flag.
                let external = checker.as_ref().is_some_and(|c| c());
                if cancelled.swap(false, Ordering::SeqCst) || external {
                    return Err(Error::Cancelled);
                }
                Ok(())
            };
            let status = protocols::streaming_error_status(protocol);
            let streamed = self
                .connection
                .stream(
                    &endpoint.path,
                    &payload,
                    &headers,
                    &mut on_attempt,
                    &mut on_event,
                    status,
                )
                .await;
            observed = acc.tokens().clone();
            match streamed.and_then(|raw| protocols::finish_stream(protocol, &mut state, acc, raw))
            {
                // `MultiCompletion#stream_response` yields one more chunk with the whole message.
                Ok((message, Some(last))) => {
                    on_chunk(&last);
                    Ok(message)
                }
                Ok((message, None)) => Ok(message),
                Err(e) => Err(e),
            }
        } else {
            match self
                .connection
                .post(&endpoint.path, &payload, &headers, &mut on_attempt)
                .await
            {
                Ok(raw) => protocols::parse_completion(protocol, self.provider, &self.model, raw),
                Err(e) => Err(e),
            }
        };

        // Usage ledger: one entry per HTTP attempt, like Accounting::Usage::Tracker. A failed
        // attempt keeps unknown tokens unless the provider refused it (4xx) or it was never sent,
        // in which case it is billed as zero, as `failure_tokens` does.
        let mut call_entries = Vec::new();
        for tokens in retried {
            let entry = self.entry(UsageStatus::Failed, tokens, None);
            self.record_usage(entry.clone()).await;
            call_entries.push(entry);
        }
        let mut message = match result {
            Ok(m) => m,
            Err(e) => {
                let status = if matches!(e, Error::Cancelled) {
                    UsageStatus::Cancelled
                } else {
                    UsageStatus::Failed
                };
                if attempts > 0 {
                    let observed = (!observed.is_empty()).then_some(observed);
                    self.record_usage(self.entry(status, failure_tokens(&e, observed), None))
                        .await;
                }
                return Err(e.claim(protocol, self.provider.slug(), &self.model.id, &payload));
            }
        };
        let billed_model = message
            .model
            .as_deref()
            .and_then(|id| models::models().find(id, Some(self.provider.slug())).ok())
            .unwrap_or_else(|| self.model.clone());
        let entry = self.entry(
            UsageStatus::Succeeded,
            message.tokens.clone(),
            Some(&billed_model),
        );
        self.record_usage(entry.clone()).await;
        // `record_generated_message`: the tracker has already billed the attempt, so a cancel
        // during the request keeps the usage but adds no message.
        self.raise_if_cancelled()?;
        call_entries.push(entry);
        message.usage_entries = call_entries.clone();
        message.model_info = Some(billed_model.clone());
        Ok((message, call_entries, billed_model, protocol))
    }

    /// `count_tokens(message)`: input tokens for the next request as configured (instructions,
    /// tools, schema, thinking, attachments), plus `message` as a staged user message when given.
    /// The chat is not changed. Provider tools, provider options, compaction, and before_request
    /// hooks are not included. Anthropic, Gemini, and OpenAI's Responses API count tokens.
    pub async fn count_tokens(&self, message: Option<&str>) -> Result<i64> {
        let protocol = self
            .provider
            .resolve_protocol(self.protocol, &self.model, &self.config)?;
        let path =
            crate::tokenization::count_tokens_endpoint(protocol, self.provider, &self.model)?;
        self.load_mcp_tools().await?;
        let mut messages = self.preprocessed_messages()?;
        if let Some(text) = message {
            messages.push(Message::user(text));
        }
        crate::files::preprocess_messages(
            &mut messages,
            protocol,
            self.provider,
            &self.config,
            &self.connection,
        )
        .await?;
        for m in &mut messages {
            for a in &mut m.attachments {
                a.load(self.connection.client()).await?;
            }
        }
        let thinking = match &self.thinking {
            Some(t) => t.resolve(&self.model)?,
            None => None,
        };
        let tools = self.request_tools()?;
        let request = Request {
            provider: self.provider,
            config: &self.config,
            model: &self.model,
            messages: &messages,
            tools: &tools,
            tool_prefs: &self.tool_prefs,
            temperature: None,
            max_output_tokens: None,
            schema: self.schema.as_ref(),
            thinking: thinking.as_ref(),
            citations: self.citations,
            caching: self.caching.as_ref(),
            stream: false,
        };
        let mut rendered = protocols::render(protocol, &request)?;
        // `Anthropic#render_count_tokens_payload`: `.then { Tools.apply_tool_search(payload) }`.
        if protocol == ProtocolName::Anthropic {
            crate::tool_search::apply(protocol, &mut rendered);
        }
        let payload = crate::tokenization::count_tokens_payload(protocol, &self.model, rendered);
        let raw = self
            .connection
            .post(&path, &payload, &[], &mut |_| {})
            .await
            .map_err(|e| e.claim(protocol, self.provider.slug(), &self.model.id, &payload))?;
        crate::tokenization::parse_count_tokens(protocol, &raw.body)
    }

    /// `compact`: condense the conversation's model context through the provider's manual
    /// compaction endpoint (OpenAI and xAI Responses). Returns an assistant message with empty
    /// text that carries the compacted context; later requests replay it in place of the turns it
    /// replaced, while `messages` keeps every earlier message. Records usage and runs the message
    /// callbacks. Uses the current instructions, headers, and before_request hooks.
    pub async fn compact(&mut self) -> Result<Message> {
        self.raise_if_cancelled()?;
        self.raise_if_pending_tool_calls()?;
        let event = self.instrumentation_event("compaction.rust_llm", false);
        let result = event.instrument(self.compact_inner()).await;
        self.finish_completion_event(event, &result);
        result
    }

    async fn compact_inner(&mut self) -> Result<Message> {
        let protocol = self
            .provider
            .resolve_protocol(self.protocol, &self.model, &self.config)?;
        if !(protocol == ProtocolName::Responses
            && matches!(self.provider, Provider::OpenAI | Provider::XAI))
        {
            return Err(Error::Api(
                format!(
                    "{} doesn't support manual compaction",
                    self.provider.display()
                ),
                None,
            ));
        }
        let mut reuploaded = false;
        loop {
            let mut entries = Vec::new();
            match self.compact_request(protocol, &mut entries).await {
                // `replacing_missing_uploads`: a file the provider deleted is uploaded again and
                // the compaction runs once more.
                Err(e) => {
                    if reuploaded || !self.discard_missing_uploads(&e).await {
                        return Err(e);
                    }
                    reuploaded = true;
                    continue;
                }
                Ok(mut message) => {
                    let entry = self.entry(UsageStatus::Succeeded, message.tokens.clone(), None);
                    self.record_usage(entry.clone()).await;
                    entries.push(entry);
                    message.usage_entries = entries;
                    // `record_generated_message`: a cancel during the request keeps the billed
                    // usage but adds no message.
                    self.raise_if_cancelled()?;
                    for cb in &mut self.callbacks.before_message {
                        cb();
                    }
                    self.messages.push(message.clone());
                    for cb in &mut self.callbacks.after_message {
                        cb(&message);
                    }
                    return Ok(message);
                }
            }
        }
    }

    /// One `responses/compact` request: prepares the history, sends it, and records the usage of
    /// each failed attempt in `entries`.
    async fn compact_request(
        &mut self,
        protocol: ProtocolName,
        entries: &mut Vec<UsageEntry>,
    ) -> Result<Message> {
        let mut messages = self.preprocessed_messages()?;
        crate::files::preprocess_messages(
            &mut messages,
            protocol,
            self.provider,
            &self.config,
            &self.connection,
        )
        .await?;
        for m in &mut messages {
            for a in &mut m.attachments {
                a.load(self.connection.client()).await?;
            }
        }
        let mut payload = protocols::responses::render_compaction_payload(
            self.provider,
            &self.model.id,
            &messages,
        )?;
        self.apply_before_request_hooks(&mut payload);
        let mut attempts = 0usize;
        let mut retried: Vec<Tokens> = Vec::new();
        let mut on_attempt = |previous: Option<&Error>| {
            attempts += 1;
            if let Some(e) = previous {
                retried.push(failure_tokens(e, None));
            }
        };
        let result = match self
            .connection
            .post(
                "responses/compact",
                &payload,
                &self.headers,
                &mut on_attempt,
            )
            .await
        {
            Ok(raw) => {
                protocols::responses::parse_compaction_response(self.provider, &self.model.id, raw)
            }
            Err(e) => Err(e.claim(protocol, self.provider.slug(), &self.model.id, &payload)),
        };
        for tokens in retried {
            let entry = self.entry(UsageStatus::Failed, tokens, None);
            self.record_usage(entry.clone()).await;
            entries.push(entry);
        }
        match result {
            Ok(m) => Ok(m),
            Err(e) => {
                if attempts > 0 {
                    self.record_usage(self.entry(
                        UsageStatus::Failed,
                        failure_tokens(&e, None),
                        None,
                    ))
                    .await;
                }
                Err(e)
            }
        }
    }

    fn entry(&self, status: UsageStatus, tokens: Tokens, model: Option<&Model>) -> UsageEntry {
        let model = model.unwrap_or(&self.model);
        let cost = Cost::new(&tokens, Some(model), Tier::Standard);
        UsageEntry {
            id: UsageEntry::next_id(),
            owner: crate::accounting::usage_owner(),
            operation: Operation::Chat,
            provider: self.provider.slug().into(),
            model: self.model.id.clone(),
            status,
            tokens,
            cost,
        }
    }

    // ---- tools -----------------------------------------------------------------------------

    fn pending_tool_response(&self) -> Option<&Message> {
        let response = latest_response(self.messages.iter())?;
        (response.is_tool_call() && !self.pending_tool_calls(response).is_empty())
            .then_some(response)
    }

    fn pending_tool_calls(&self, response: &Message) -> Vec<ToolCall> {
        let answered: Vec<&str> = self
            .messages
            .iter()
            .filter_map(|m| m.tool_call_id.as_deref())
            .collect();
        response
            .tool_calls
            .iter()
            .flat_map(|c| c.values())
            .filter(|c| !answered.contains(&c.id.as_str()))
            .cloned()
            .collect()
    }

    fn find_tool(&self, name: &str) -> Option<SharedTool> {
        let tools = self.combined_tools().unwrap_or_else(|_| self.tools.clone());
        tools.into_iter().find(|t| t.name() == name)
    }

    fn approval_pending(&self, call: &ToolCall) -> bool {
        if call.remote {
            return !self.tool_call_decisions.contains_key(&call.id);
        }
        let Some(tool) = self.find_tool(&call.name) else {
            return false;
        };
        tool.requires_approval() && self.tool_call_approval(&*tool, call).is_none()
    }

    /// `tool_call_approval`: the tool's own resolver, else the recorded decision.
    fn tool_call_approval(&self, tool: &dyn Tool, call: &ToolCall) -> Option<bool> {
        match tool.approval(call) {
            Some(decision) => decision,
            None => self.tool_call_decisions.get(&call.id).copied(),
        }
    }

    /// Execute the pending tool calls of the latest response (`run_tools`). Calls paused on an
    /// MCP task check on it once. A cancelled chat cancels the tasks its calls wait on.
    pub async fn run_tools(&mut self) -> Result<&mut Self> {
        match self.run_pending_tools().await {
            Err(Error::Cancelled) => {
                self.cancel_tasks().await;
                Err(Error::Cancelled)
            }
            Err(e) => Err(e),
            Ok(()) => Ok(self),
        }
    }

    async fn run_pending_tools(&mut self) -> Result<()> {
        self.raise_if_cancelled()?;
        let Some(response) = self.pending_tool_response().cloned() else {
            return Ok(());
        };
        self.load_mcp_tools().await?;
        let mut executable = Vec::new();
        for call in self.pending_tool_calls(&response) {
            self.raise_if_cancelled()?;
            if self.input_pending(&call) {
                continue;
            }
            if call.remote {
                // Remote (provider-executed) approvals are answered on the next request.
                if let Some(&approved) = self.tool_call_decisions.get(&call.id) {
                    // `Protocol#tool_approval_response`: only the Responses protocol renders one.
                    let responses = self
                        .provider
                        .resolve_protocol(self.protocol, &self.model, &self.config)
                        .is_ok_and(|p| p == ProtocolName::Responses);
                    if !responses {
                        return Err(Error::Api(
                            format!(
                                "{} doesn't support remote tool approvals",
                                self.provider.display()
                            ),
                            None,
                        ));
                    }
                    let mut m = Message::tool_result(
                        call.id.clone(),
                        if approved { "Approved" } else { "Denied" },
                    );
                    m.raw_content = Some(serde_json::json!([{
                        "type": "mcp_approval_response", "approval_request_id": call.id, "approve": approved
                    }]));
                    self.push_tool_message(m);
                }
                continue;
            }
            let tool = self.find_tool(&call.name);
            let decision = match &tool {
                Some(t) if t.requires_approval() => self.tool_call_approval(&**t, &call),
                _ => Some(self.tool_call_decisions.get(&call.id).copied() != Some(false)),
            };
            match decision {
                None => continue,
                Some(false) => {
                    let denied =
                        ToolResult::error(format!("The user denied the {} tool call.", call.name));
                    self.push_tool_result(&call, denied);
                }
                Some(true) if self.concurrency => executable.push((call, tool)),
                Some(true) => {
                    for cb in &mut self.callbacks.before_tool_call {
                        cb(&call);
                    }
                    // `invoke_tool`: a paused MCP call records its input requests and adds no
                    // result; a resumed one clears them.
                    let result = match self.tool_invocation(tool, &call).await {
                        Err(Error::McpInputRequired(paused)) => {
                            self.tool_call_inputs
                                .insert(call.id.clone(), paused.input.to_h());
                            continue;
                        }
                        Err(Error::McpTask(task)) => {
                            self.tool_call_inputs.insert(call.id.clone(), task.to_h());
                            continue;
                        }
                        other => other?,
                    };
                    self.tool_call_inputs.remove(&call.id);
                    self.raise_if_cancelled()?;
                    for cb in &mut self.callbacks.after_tool_result {
                        cb(&result);
                    }
                    self.push_tool_result(&call, result);
                }
            }
        }
        if !executable.is_empty() {
            self.run_tools_concurrently(executable).await?;
        }
        if matches!(
            self.tool_prefs.choice,
            Some(ToolChoice::Required | ToolChoice::Tool(_))
        ) {
            self.tool_prefs.choice = None;
        }
        Ok(())
    }

    /// `handle_concurrent_tool_calls` (`Chat::ToolConcurrency.run`): every call starts at once and
    /// each result is added as its call finishes, so results land in finishing order. The first
    /// failure is returned once every call has ended, like `collect_results`.
    async fn run_tools_concurrently(
        &mut self,
        calls: Vec<(ToolCall, Option<SharedTool>)>,
    ) -> Result<()> {
        use futures::stream::{FuturesUnordered, StreamExt};
        let mut running = FuturesUnordered::new();
        for (call, tool) in calls {
            self.raise_if_cancelled()?;
            for cb in &mut self.callbacks.before_tool_call {
                cb(&call);
            }
            let invocation = self.tool_invocation(tool, &call);
            running.push(async move { (call, invocation.await) });
        }
        let mut first_error = None;
        while let Some((call, result)) = running.next().await {
            let result = match result {
                Err(Error::McpInputRequired(paused)) => {
                    self.tool_call_inputs
                        .insert(call.id.clone(), paused.input.to_h());
                    continue;
                }
                Err(Error::McpTask(task)) => {
                    self.tool_call_inputs.insert(call.id.clone(), task.to_h());
                    continue;
                }
                Err(e) => {
                    first_error.get_or_insert(e);
                    continue;
                }
                Ok(result) => result,
            };
            self.tool_call_inputs.remove(&call.id);
            self.raise_if_cancelled()?;
            for cb in &mut self.callbacks.after_tool_result {
                cb(&result);
            }
            self.push_tool_result(&call, result);
        }
        first_error.map_or(Ok(()), Err)
    }

    /// `execute_tool`: the returned future owns everything the call needs, so several calls can
    /// run at once while the chat adds their results.
    fn tool_invocation(
        &self,
        tool: Option<SharedTool>,
        call: &ToolCall,
    ) -> impl Future<Output = Result<ToolResult>> + Send + 'static {
        let unavailable = tool.is_none().then(|| {
            let names: Vec<String> = self
                .combined_tools()
                .unwrap_or_else(|_| self.tools.clone())
                .iter()
                .map(|t| t.name())
                .collect();
            format!(
                "Model tried to call unavailable tool `{}`. Available tools: {}.",
                call.name,
                serde_json::to_string(&names).unwrap_or_default()
            )
        });
        let call = call.clone();
        let input = self.tool_call_inputs.get(&call.id).cloned();
        let cancelled = self.cancelled.clone();
        let listener = self.progress_listener(&call);
        // `tool_call.rust_llm` (Ruby skips it for an unavailable tool, as here).
        let event = tool.as_ref().map(|tool| {
            crate::instrumentation::Event::start(&self.config, "tool_call.rust_llm", || {
                crate::instrumentation::payload([
                    ("provider", self.provider.slug().into()),
                    ("provider_class", self.provider.display().into()),
                    ("model", self.model.id.clone().into()),
                    (
                        "tool_call",
                        serde_json::to_value(&call).unwrap_or(Value::Null),
                    ),
                    ("tool_name", tool.name().into()),
                    ("tool_arguments", Value::Object(call.arguments())),
                    ("tool_call_id", call.id.clone().into()),
                ])
            })
        });
        let run = self.run_tool(tool, call, input, cancelled, listener, unavailable);
        async move {
            let Some(mut event) = event else {
                return run.await;
            };
            let result = event.instrument(run).await;
            if let Ok(r) = &result {
                event.set("result", || r.content.clone().into());
                event.set("result_content", || r.content.clone().into());
                event.set("result_class", || "ToolResult".into());
            }
            event.finish(result.as_ref().err());
            result
        }
    }

    // Returns an explicit `Send + 'static` future (tool calls run concurrently), which an
    // `async fn` taking `&self` could not promise.
    #[allow(clippy::manual_async_fn)]
    fn run_tool(
        &self,
        tool: Option<SharedTool>,
        call: ToolCall,
        input: Option<Value>,
        cancelled: Arc<AtomicBool>,
        listener: Option<crate::progress::Listener>,
        unavailable: Option<String>,
    ) -> impl Future<Output = Result<ToolResult>> + Send + 'static {
        async move {
            let Some(tool) = tool else {
                return Ok(ToolResult::error(unavailable.unwrap_or_default()));
            };
            let arguments = call.arguments();
            if let Some(problem) = validate_arguments(&*tool, &arguments) {
                return Ok(ToolResult::error(format!(
                    "Invalid tool arguments: {problem}"
                )));
            }
            // `Cancellation.watch` + `ProgressReporter.listen(progress_listener(tool_call))`.
            let run = async {
                match &input {
                    Some(input) => tool.resume(input, arguments, &call).await,
                    None => tool.execute(arguments, &call).await,
                }
            };
            let result =
                crate::progress::watch(cancelled.clone(), crate::progress::listen(listener, run))
                    .await;
            result.map_err(|e| match e.downcast::<Error>() {
                // MCP pauses, protocol errors, and cancellation keep their meaning.
                Ok(e)
                    if matches!(
                        *e,
                        Error::Cancelled
                            | Error::Mcp(_)
                            | Error::McpInputRequired(_)
                            | Error::McpTask(_)
                    ) =>
                {
                    if matches!(*e, Error::Cancelled) {
                        cancelled.store(false, Ordering::SeqCst);
                    }
                    *e
                }
                Ok(e) => Error::Tool(e.to_string()),
                Err(e) => Error::Tool(e.to_string()),
            })
        }
    }

    /// `progress_listener(tool_call)`: `None` when no `after_tool_progress` callback listens.
    fn progress_listener(&self, call: &ToolCall) -> Option<crate::progress::Listener> {
        let callbacks = self.callbacks.after_tool_progress.clone();
        if callbacks.lock().map(|c| c.is_empty()).unwrap_or(true) {
            return None;
        }
        let call = call.clone();
        Some(Arc::new(move |progress: &crate::progress::Progress| {
            if let Ok(mut callbacks) = callbacks.lock() {
                for cb in callbacks.iter_mut() {
                    cb(&call, progress);
                }
            }
        }))
    }

    fn push_tool_result(&mut self, call: &ToolCall, result: ToolResult) {
        let mut m = Message::tool_result(call.id.clone(), result.content);
        m.attachments = result.attachments;
        // `mcp_result = result if result.is_a?(MCP::Result) && result.ui_uri`.
        m.mcp_result = result.mcp_result.filter(|r| r.ui_uri.is_some());
        self.push_tool_message(m);
    }

    fn push_tool_message(&mut self, m: Message) {
        for cb in &mut self.callbacks.before_message {
            cb();
        }
        self.messages.push(m.clone());
        for cb in &mut self.callbacks.after_message {
            cb(&m);
        }
    }

    /// `approve(tool_call)`: the next `complete` executes it.
    pub fn approve(&mut self, tool_call_id: &str) -> &mut Self {
        self.tool_call_decisions.insert(tool_call_id.into(), true);
        self
    }

    /// `deny(tool_call)`: the next `complete` answers it with a structured denial.
    pub fn deny(&mut self, tool_call_id: &str) -> &mut Self {
        self.tool_call_decisions.insert(tool_call_id.into(), false);
        self
    }

    /// Restores decisions recorded earlier, e.g. from persisted tool-call rows.
    pub fn set_decisions(&mut self, decisions: impl IntoIterator<Item = (String, bool)>) {
        self.tool_call_decisions.extend(decisions);
    }

    /// `waiting?`: every pending tool call waits on an approval decision, an input, or an MCP task.
    pub fn is_waiting(&self) -> bool {
        self.waiting()
    }

    fn waiting(&self) -> bool {
        let Some(response) = self.pending_tool_response() else {
            return false;
        };
        let pending = self.pending_tool_calls(response);
        !pending.is_empty()
            && pending
                .iter()
                .all(|c| self.approval_pending(c) || self.is_paused(c))
    }

    /// `paused?(tool_call)`: the call waits on an unsettled input request or an MCP task.
    fn is_paused(&self, call: &ToolCall) -> bool {
        self.input_pending(call)
            || self
                .tool_call_inputs
                .get(&call.id)
                .is_some_and(|state| state.get("task").is_some())
    }

    /// `input_pending?(tool_call)`: the call paused on a request nobody has settled.
    fn input_pending(&self, call: &ToolCall) -> bool {
        self.tool_call_inputs
            .get(&call.id)
            .and_then(|input| input.get("requests"))
            .and_then(Value::as_array)
            .is_some_and(|requests| {
                requests
                    .iter()
                    .any(|r| r.get("response").is_none_or(Value::is_null))
            })
    }

    /// `awaiting_input?`: every pending tool call waits on input or an approval decision, and at
    /// least one waits on input.
    pub fn is_awaiting_input(&self) -> bool {
        self.waiting() && !self.pending_inputs().is_empty()
    }

    /// `awaiting_tasks?`: every pending tool call waits on a task, input, or an approval
    /// decision, and at least one waits on a task. `complete` checks on the tasks again.
    pub fn is_awaiting_tasks(&self) -> bool {
        self.waiting() && !self.pending_tasks().is_empty()
    }

    /// `pending_tasks`: the MCP tasks that tool calls wait on, as they were when the chat last
    /// checked. `Task::refresh` checks on one without resuming the chat; `complete` checks on
    /// all of them and resumes their calls once they are done.
    pub fn pending_tasks(&self) -> Vec<crate::mcp::Task> {
        let Some(response) = self.pending_tool_response() else {
            return Vec::new();
        };
        self.pending_tool_calls(response)
            .into_iter()
            .filter_map(|call| {
                let state = self.tool_call_inputs.get(&call.id)?;
                (state.get("task").is_some() && !self.input_pending(&call))
                    .then(|| self.task_for(&call, state))
            })
            .collect()
    }

    /// `task_for(tool_call, state)`: through the call's tool when it is connected.
    fn task_for(&self, call: &ToolCall, state: &Value) -> crate::mcp::Task {
        self.find_tool(&call.name)
            .and_then(|tool| tool.mcp_task(state, call))
            .unwrap_or_else(|| crate::mcp::Task::load(None, state, Some(call.clone())))
    }

    /// `cancel_tasks`: cancels the tasks pending tool calls wait on and forgets them.
    async fn cancel_tasks(&mut self) {
        let Some(response) = self.pending_tool_response().cloned() else {
            return;
        };
        for call in self.pending_tool_calls(&response) {
            let Some(state) = self.tool_call_inputs.get(&call.id).cloned() else {
                continue;
            };
            if state.get("task").is_none() {
                continue;
            }
            let mut task = self.task_for(&call, &state);
            if let Err(e) = task.cancel().await {
                tracing::debug!("RustLLM could not cancel task {}: {e}", task.id);
            }
            self.tool_call_inputs.remove(&call.id);
        }
    }

    /// `pending_inputs`: the unanswered requests that paused MCP tool calls. Settle each with
    /// `answer` or `decline`, then call `complete` to resume the calls.
    pub fn pending_inputs(&self) -> Vec<crate::mcp::InputRequest> {
        let Some(response) = self.pending_tool_response() else {
            return Vec::new();
        };
        self.pending_tool_calls(response)
            .into_iter()
            .flat_map(|call| {
                let requests = self
                    .tool_call_inputs
                    .get(&call.id)
                    .and_then(|i| i.get("requests"))
                    .and_then(Value::as_array)
                    .cloned();
                requests
                    .unwrap_or_default()
                    .iter()
                    .map(|data| crate::mcp::InputRequest::from_h(data, Some(call.clone())))
                    .filter(|r| !r.is_answered())
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// `answer(request, **values)`: values for a form, or none to accept a URL request.
    pub fn answer(
        &mut self,
        request: &crate::mcp::InputRequest,
        values: Map<String, Value>,
    ) -> Result<&mut Self> {
        self.settle_input(request, |pending| pending.answer(values))
    }

    /// `decline(request)`.
    pub fn decline(&mut self, request: &crate::mcp::InputRequest) -> Result<&mut Self> {
        self.settle_input(request, crate::mcp::InputRequest::decline)
    }

    fn settle_input(
        &mut self,
        request: &crate::mcp::InputRequest,
        settle: impl FnOnce(&mut crate::mcp::InputRequest),
    ) -> Result<&mut Self> {
        let unknown = || Error::Argument("Unknown input request".into());
        let call_id = request
            .tool_call
            .as_ref()
            .map(|c| c.id.clone())
            .ok_or_else(unknown)?;
        let input = self
            .tool_call_inputs
            .get_mut(&call_id)
            .ok_or_else(unknown)?;
        let requests = input
            .get_mut("requests")
            .and_then(Value::as_array_mut)
            .ok_or_else(unknown)?;
        let data = requests
            .iter_mut()
            .find(|d| d.get("key").and_then(Value::as_str) == Some(request.key.as_str()))
            .ok_or_else(unknown)?;
        let mut pending = crate::mcp::InputRequest::from_h(data, None);
        settle(&mut pending);
        *data = pending.to_h();
        Ok(self)
    }

    /// Paused MCP calls' state per tool call id, for persistence (`input_recorder`).
    pub fn tool_call_inputs(&self) -> &HashMap<String, Value> {
        &self.tool_call_inputs
    }

    /// Restores paused MCP calls' state, e.g. from persisted tool-call rows (`input_checker`).
    pub fn set_tool_call_inputs(&mut self, inputs: impl IntoIterator<Item = (String, Value)>) {
        self.tool_call_inputs.extend(inputs);
    }

    /// `awaiting_approval?`.
    pub fn is_awaiting_approval(&self) -> bool {
        self.waiting() && !self.pending_approvals().is_empty()
    }

    /// `pending_approvals`.
    pub fn pending_approvals(&self) -> Vec<ToolCall> {
        let Some(response) = self.pending_tool_response() else {
            return Vec::new();
        };
        self.pending_tool_calls(response)
            .into_iter()
            .filter(|c| self.approval_pending(c))
            .collect()
    }

    fn raise_if_pending_tool_calls(&self) -> Result<()> {
        let Some(response) = self.pending_tool_response() else {
            return Ok(());
        };
        let mut names: Vec<String> = self
            .pending_tool_calls(response)
            .into_iter()
            .map(|c| c.name)
            .collect();
        names.dedup();
        Err(Error::PendingToolCalls(format!(
            "The last response has unanswered tool calls ({}). Run complete, recording approve or deny \
             decisions for calls that require approval, answering pending inputs, and waiting for tasks, before asking again.",
            names.join(", ")
        )))
    }

    fn last_non_system_message(&self) -> Option<&Message> {
        self.messages.iter().rev().find(|m| m.role != Role::System)
    }

    // ---- cancellation ----------------------------------------------------------------------

    /// `cancel`: stop at the next checkpoint with `Error::Cancelled`.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    pub fn cancel_handle(&self) -> CancelHandle {
        CancelHandle(self.cancelled.clone())
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    /// `cancellation_checker=`: consulted at every checkpoint along with `cancel`.
    pub fn set_cancellation_checker(&mut self, checker: Option<CancellationChecker>) {
        self.cancellation_checker = checker;
    }

    /// `raise_if_cancelled!`: the flag is one-shot; the external checker is asked every time.
    fn raise_if_cancelled(&self) -> Result<()> {
        let external = self.cancellation_checker.as_ref().is_some_and(|c| c());
        if self.cancelled.swap(false, Ordering::SeqCst) || external {
            return Err(Error::Cancelled);
        }
        Ok(())
    }
}

/// `Tracker#failure_tokens`: tokens a stream reported before failing are kept; a request the
/// provider refused (4xx) or that never reached it is billed as zero; anything else is unknown,
/// which keeps `cost.total` honest instead of silently low.
pub(crate) fn failure_tokens(error: &Error, observed: Option<Tokens>) -> Tokens {
    if let Some(tokens) = observed {
        return tokens;
    }
    let refused = error
        .response()
        .is_some_and(|r| (400..500).contains(&r.status));
    let never_sent = error.kind() == ErrorKind::ConnectionFailed;
    if refused || never_sent {
        Tokens {
            input: Some(0),
            output: Some(0),
            ..Default::default()
        }
    } else {
        Tokens::default()
    }
}

/// `Chat::COMPACTION_OPTIONS`: the provider-neutral options `with_compaction` accepts.
pub const COMPACTION_OPTIONS: &[&str] = &["at", "instructions", "pause_after"];

/// `Chat#normalize_schema_payload`.
fn normalize_schema(raw: Value) -> Option<Schema> {
    if raw.is_null() {
        return None;
    }
    let obj = raw.as_object().cloned().unwrap_or_default();
    let mut definition = obj.get("schema").cloned().unwrap_or_else(|| raw.clone());
    if let Some(d) = definition.as_object_mut() {
        d.remove("$schema");
        d.remove("title");
    }
    let strict = match obj.get("strict").and_then(Value::as_bool) {
        Some(s) => Some(s),
        None => definition
            .as_object_mut()
            .and_then(|d| d.remove("strict"))
            .and_then(|v| v.as_bool()),
    };
    let name = obj
        .get("name")
        .or_else(|| obj.get("title"))
        .and_then(Value::as_str)
        .unwrap_or("response")
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect::<String>();
    Some(Schema {
        name: if name.is_empty() {
            "response".into()
        } else {
            name
        },
        schema: definition,
        strict,
        description: obj
            .get("description")
            .and_then(Value::as_str)
            .map(str::to_string),
    })
}

/// `UNFINISHED_TOOL_RESULT`: what a request answers a call the conversation moved past with.
const UNFINISHED_TOOL_RESULT: &str = r#"{"error":"The tool call did not finish."}"#;

/// `latest_response`: a blank response answers nothing, so it never hides the round before it.
fn latest_response<'a>(
    messages: impl DoubleEndedIterator<Item = &'a Message>,
) -> Option<&'a Message> {
    messages
        .rev()
        .find(|m| m.role != Role::System && !m.is_tool_result() && !is_blank_response(m))
}

/// `blank_response?`: the placeholder a process saves before a response arrives.
fn is_blank_response(m: &Message) -> bool {
    m.role == Role::Assistant
        && !m.is_tool_call()
        && m.content().trim().is_empty()
        && m.attachments.is_empty()
        && m.thinking.is_none()
        && m.server_tool_calls.is_empty()
        && m.raw_content.is_none()
        && m.raw_reasoning.is_none()
}
