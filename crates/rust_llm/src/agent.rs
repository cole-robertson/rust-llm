//! Port of `lib/ruby_llm/agent.rb`: reusable assistants declared once.
//!
//! ```ruby
//! class WeatherAssistant < RubyLLM::Agent
//!   model "gpt-5.6-luna"
//!   instructions "Be concise and always use tools for weather."
//!   tools Weather
//! end
//! WeatherAssistant.new.ask "What's the weather in Berlin?"
//! ```
//!
//! ```no_run
//! # use std::sync::Arc;
//! # use rust_llm::{Agent, SharedTool, Tool, ToolCall, ToolError, ToolResult};
//! # struct Weather;
//! # #[async_trait::async_trait]
//! # impl Tool for Weather {
//! #     fn description(&self) -> String { "Gets the weather".into() }
//! #     async fn execute(&self, _: serde_json::Map<String, serde_json::Value>, _: &ToolCall) -> Result<ToolResult, ToolError> { Ok("15C".into()) }
//! # }
//! # async fn run() -> rust_llm::Result<()> {
//! struct WeatherAssistant;
//! impl Agent for WeatherAssistant {
//!     fn model(&self) -> Option<&str> { Some("gpt-5.6-luna") }
//!     fn instructions(&self) -> Option<String> { Some("Be concise and always use tools for weather.".into()) }
//!     fn tools(&self) -> Vec<SharedTool> { vec![Arc::new(Weather)] }
//! }
//! WeatherAssistant.chat()?.ask("What's the weather in Berlin?").await?;
//! # Ok(()) }
//! ```

use std::sync::Arc;

use serde_json::{Map, Value};

use crate::chat::{Chat, Fallback};
use crate::config::Config;
use crate::context::Context;
use crate::error::{ErrorKind, Result};
use crate::protocols::{ToolCalls, ToolChoice};
use crate::providers::ProtocolName;
use crate::thinking::ThinkingConfig;
use crate::tool::SharedTool;

/// Class-level declarations of `RubyLLM::Agent`. Every method has a default, so an agent only
/// overrides what it declares.
pub trait Agent {
    fn model(&self) -> Option<&str> {
        None
    }
    fn provider(&self) -> Option<&str> {
        None
    }
    fn protocol(&self) -> Option<ProtocolName> {
        None
    }
    /// `model "...", assume_model_exists: true`: build the chat without a registry lookup.
    fn assume_model_exists(&self) -> bool {
        false
    }
    /// `context SharedContext`: the [`Context`] whose configuration the chats this agent builds
    /// use, applied like `Chat#with_context`. Called once per [`Agent::chat`].
    fn context(&self) -> Option<Context> {
        None
    }
    /// `instructions "..."`. When `None`, the conventional prompt
    /// `app/prompts/<agent>/instructions.txt.jinja` is rendered with [`Agent::prompt_locals`] if
    /// it exists, like RubyLLM's `default_instructions_prompt`. An empty prompt file means no
    /// instructions.
    fn instructions(&self) -> Option<String> {
        None
    }
    /// The agent's class name, which names its prompt directory (`WorkAssistant` reads
    /// `app/prompts/work_assistant/`). Defaults to the type name.
    fn name(&self) -> String {
        let full = std::any::type_name::<Self>();
        let base = full.split('<').next().unwrap_or(full);
        base.rsplit("::").next().unwrap_or(base).to_string()
    }
    /// Locals for the agent's prompts (`instructions display_name: -> { ... }`).
    fn prompt_locals(&self) -> Value {
        Value::Object(Default::default())
    }
    /// `instructions_config` evaluated for one chat: every `instructions ...` declaration, in
    /// order. `chat` is what prompts see as `chat` (`Null` for a plain chat, the record in Rails
    /// mode); `config` supplies the prompt roots. The default is [`Agent::instructions`] when it
    /// is `Some`, else the conventional `instructions` prompt rendered with `chat` and
    /// [`Agent::prompt_locals`] when that file exists, else nothing. Blank texts are skipped when
    /// applied (`blank_instruction?`). Override to declare several, or to reference a prompt
    /// explicitly (`instructions { prompt('instructions') }`), whose absence is an error.
    fn instruction_declarations(
        &self,
        config: &Arc<Config>,
        chat: &Value,
    ) -> Result<Vec<InstructionDeclaration>> {
        if let Some(text) = self.instructions() {
            return Ok(vec![InstructionDeclaration::new(text)]);
        }
        let prompt = crate::prompt::Prompt::with_config(
            config.clone(),
            format!("{}/instructions", prompt_agent_path(&self.name())),
        );
        if !prompt.exists() {
            return Ok(Vec::new());
        }
        // `resolve_prompt_locals`: `{ chat: }` with the declared locals merged over it.
        let mut locals = Map::new();
        locals.insert("chat".into(), chat.clone());
        if let Value::Object(declared) = self.prompt_locals() {
            locals.extend(declared);
        }
        Ok(vec![InstructionDeclaration::new(
            prompt.render(Value::Object(locals))?,
        )])
    }
    /// `Agent.render_prompt(name)`: `app/prompts/<agent>/<name>.txt.jinja` with `locals` merged
    /// over [`Agent::prompt_locals`].
    fn render_prompt(&self, name: &str, locals: Value) -> Result<String> {
        let mut merged = match self.prompt_locals() {
            Value::Object(m) => m,
            _ => Default::default(),
        };
        if let Value::Object(extra) = locals {
            merged.extend(extra);
        }
        crate::prompt::render_prompt(
            &format!("{}/{name}", prompt_agent_path(&self.name())),
            Value::Object(merged),
        )
    }
    fn tools(&self) -> Vec<SharedTool> {
        Vec::new()
    }
    fn tool_choice(&self) -> Option<ToolChoice> {
        None
    }
    /// `tool_options calls: :one`.
    fn tool_calls(&self) -> Option<ToolCalls> {
        None
    }
    /// `tool_options concurrency: true`.
    fn tool_concurrency(&self) -> Option<bool> {
        None
    }
    fn temperature(&self) -> Option<f64> {
        None
    }
    fn max_output_tokens(&self) -> Option<i64> {
        None
    }
    fn thinking(&self) -> Option<ThinkingConfig> {
        None
    }
    fn schema(&self) -> Option<Value> {
        None
    }
    /// `citations` / `citations false`.
    fn citations(&self) -> Option<bool> {
        None
    }
    /// `caching ttl: '1h'` / `caching false`: what `with_caching` receives.
    fn caching(&self) -> Option<Value> {
        None
    }
    /// `compaction at: 50_000` / `compaction false`: what `with_compaction` receives.
    fn compaction(&self) -> Option<Value> {
        None
    }
    /// `end_user 'tenant-42'`.
    fn end_user(&self) -> Option<String> {
        None
    }
    fn provider_options(&self) -> Option<Value> {
        None
    }
    /// `headers 'X-Test' => '1'`.
    fn headers(&self) -> Vec<(String, String)> {
        Vec::new()
    }
    fn fallbacks(&self) -> Vec<Fallback> {
        Vec::new()
    }
    /// `fallbacks ..., on: RubyLLM::RateLimitError`; `None` keeps the default error classes.
    fn fallback_errors(&self) -> Option<Vec<ErrorKind>> {
        None
    }
    /// `mcp Files` / `mcp { [Linear.new(user: user)] }`: servers connected via `with_mcp`.
    fn mcp(&self) -> Vec<crate::mcp::Mcp> {
        Vec::new()
    }
    /// `mcp ..., defer:`: passed to `with_mcp(defer:)` for the servers `mcp` declares.
    fn mcp_defer(&self) -> Option<bool> {
        None
    }
    /// `provider_tools :web_search`: applied via `with_provider_tools`.
    fn provider_tools(&self) -> Vec<crate::provider_tools::ProviderTool> {
        Vec::new()
    }

    /// Applies this agent's configuration to an existing chat (`Agent.new(chat:)`), which is how
    /// a persisted chat record picks its agent back up. Follows `Agent.apply_configuration`
    /// (`lib/ruby_llm/agent.rb`): options left unset do not touch the chat.
    fn apply(&self, chat: Chat) -> Result<Chat> {
        let chat = self.apply_except_instructions(chat)?;
        apply_instructions(self, chat)
    }

    /// [`Agent::apply`] without the instruction declarations, for a caller that applies them
    /// itself (Rails mode persists each one per its `persist`).
    fn apply_except_instructions(&self, chat: Chat) -> Result<Chat> {
        apply_configuration(self, chat, self.context())
    }

    /// `Agent.chat` / `Agent.new.chat`: a fresh chat, built with [`Agent::context`]'s
    /// configuration when there is one, with everything declared applied.
    fn chat(&self) -> Result<Chat> {
        let context = self.context();
        let config = context
            .as_ref()
            .map(|c| c.config().clone())
            .unwrap_or_else(crate::config);
        let chat = Chat::with_config(
            config,
            self.model(),
            self.provider(),
            self.assume_model_exists(),
        )?;
        let chat = apply_configuration(self, chat, context)?;
        apply_instructions(self, chat)
    }
}

/// One `instructions text, append:, persist:, cache_until_here:` declaration, evaluated.
#[derive(Debug, Clone, PartialEq)]
pub struct InstructionDeclaration {
    pub text: String,
    pub append: bool,
    /// Rails mode: whether the record stores it (`persist: true`, the default).
    pub persist: bool,
    pub cache_until_here: bool,
}

impl InstructionDeclaration {
    /// `instructions text`: replaces earlier instructions, persisted, no cache boundary.
    pub fn new(text: impl Into<String>) -> Self {
        InstructionDeclaration {
            text: text.into(),
            append: false,
            persist: true,
            cache_until_here: false,
        }
    }
}

/// `apply_instructions`: each declaration in order, blank ones skipped. Ruby applies them right
/// after the context; applying them last is the same, since nothing else touches the messages.
fn apply_instructions<A: Agent + ?Sized>(agent: &A, mut chat: Chat) -> Result<Chat> {
    let config = chat.config().clone();
    for d in agent.instruction_declarations(&config, &Value::Null)? {
        if !d.text.trim().is_empty() {
            chat.set_instructions(Some(d.text), d.append, d.cache_until_here);
        }
    }
    Ok(chat)
}

/// `Agent.apply_configuration` (`lib/ruby_llm/agent.rb`) minus the instructions: options left
/// unset do not touch the chat. `context` is resolved once by the caller.
fn apply_configuration<A: Agent + ?Sized>(
    agent: &A,
    mut chat: Chat,
    context: Option<Context>,
) -> Result<Chat> {
    // `apply_context`: rebind unless the chat was already built with this context.
    if let Some(context) = context.filter(|c| !Arc::ptr_eq(c.config(), chat.config())) {
        chat = chat.with_context(Some(&context))?;
    }
    chat = chat.with_tools(agent.tools());
    for server in agent.mcp() {
        chat = chat.with_mcp_deferred(server, agent.mcp_defer());
    }
    if let Some(choice) = agent.tool_choice() {
        chat = chat.with_tool_choice(choice)?;
    }
    if let Some(calls) = agent.tool_calls() {
        chat = chat.with_tool_calls(calls);
    }
    if let Some(enabled) = agent.tool_concurrency() {
        chat = chat.with_tool_concurrency(enabled);
    }
    chat = chat.with_provider_tools(agent.provider_tools());
    if let Some(t) = agent.temperature() {
        chat = chat.with_temperature(t);
    }
    if let Some(m) = agent.max_output_tokens() {
        chat = chat.with_max_output_tokens(m);
    }
    if let Some(t) = agent.thinking() {
        chat = chat.with_thinking(t);
    }
    if let Some(c) = agent.citations() {
        chat = chat.with_citations(c);
    }
    if let Some(u) = agent.end_user() {
        chat = chat.with_end_user(Some(&u));
    }
    if let Some(c) = agent.caching() {
        chat = chat.with_caching(c)?;
    }
    if let Some(c) = agent.compaction() {
        chat = chat.with_compaction(c)?;
    }
    if let Some(o) = agent.provider_options() {
        chat = chat.with_provider_options(o);
    }
    let headers = agent.headers();
    if !headers.is_empty() {
        chat = chat.with_headers(headers);
    }
    if let Some(s) = agent.schema() {
        chat = chat.with_schema(s);
    }
    let fallbacks = agent.fallbacks();
    if !fallbacks.is_empty() {
        chat = chat.with_fallbacks(fallbacks);
        if let Some(kinds) = agent.fallback_errors() {
            chat = chat.with_fallback_errors(kinds);
        }
    }
    if let Some(p) = agent.protocol() {
        chat = chat.with_protocol(p);
    }
    Ok(chat)
}

/// `prompt_agent_path`: `Admin::WorkAssistant` is `admin/work_assistant`.
pub fn prompt_agent_path(name: &str) -> String {
    crate::tool::underscore(&name.replace("::", "/")).replace('-', "_")
}
