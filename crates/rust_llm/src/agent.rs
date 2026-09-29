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

use serde_json::Value;

use crate::chat::{Chat, Fallback};
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
        crate::prompt::render_prompt(&format!("{}/{name}", prompt_agent_path(&self.name())), Value::Object(merged))
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
    /// `provider_tools :web_search`: applied via `with_provider_tools`.
    fn provider_tools(&self) -> Vec<crate::provider_tools::ProviderTool> {
        Vec::new()
    }

    /// Applies this agent's configuration to an existing chat (`Agent.new(chat:)`), which is how
    /// a persisted chat record picks its agent back up. Follows `Agent.apply_configuration`
    /// (`lib/ruby_llm/agent.rb`): options left unset do not touch the chat.
    fn apply(&self, mut chat: Chat) -> Result<Chat> {
        let text = match self.instructions() {
            Some(text) => Some(text),
            // `instructions_config`: the conventional prompt when nothing is declared.
            None => {
                let prompt = crate::prompt::Prompt::with_config(chat.config().clone(), format!("{}/instructions", prompt_agent_path(&self.name())));
                if prompt.exists() { Some(prompt.render(self.prompt_locals())?).filter(|t| !t.trim().is_empty()) } else { None }
            }
        };
        if let Some(text) = text {
            chat.set_instructions(Some(text), false, false);
        }
        chat = chat.with_tools(self.tools());
        for server in self.mcp() {
            chat = chat.with_mcp(server);
        }
        if let Some(choice) = self.tool_choice() {
            chat = chat.with_tool_choice(choice)?;
        }
        if let Some(calls) = self.tool_calls() {
            chat = chat.with_tool_calls(calls);
        }
        if let Some(enabled) = self.tool_concurrency() {
            chat = chat.with_tool_concurrency(enabled);
        }
        chat = chat.with_provider_tools(self.provider_tools());
        if let Some(t) = self.temperature() {
            chat = chat.with_temperature(t);
        }
        if let Some(m) = self.max_output_tokens() {
            chat = chat.with_max_output_tokens(m);
        }
        if let Some(t) = self.thinking() {
            chat = chat.with_thinking(t);
        }
        if let Some(c) = self.citations() {
            chat = chat.with_citations(c);
        }
        if let Some(u) = self.end_user() {
            chat = chat.with_end_user(Some(&u));
        }
        if let Some(c) = self.caching() {
            chat = chat.with_caching(c)?;
        }
        if let Some(c) = self.compaction() {
            chat = chat.with_compaction(c)?;
        }
        if let Some(o) = self.provider_options() {
            chat = chat.with_provider_options(o);
        }
        let headers = self.headers();
        if !headers.is_empty() {
            chat = chat.with_headers(headers);
        }
        if let Some(s) = self.schema() {
            chat = chat.with_schema(s);
        }
        let fallbacks = self.fallbacks();
        if !fallbacks.is_empty() {
            chat = chat.with_fallbacks(fallbacks);
            if let Some(kinds) = self.fallback_errors() {
                chat = chat.with_fallback_errors(kinds);
            }
        }
        if let Some(p) = self.protocol() {
            chat = chat.with_protocol(p);
        }
        Ok(chat)
    }

    /// `Agent.chat` / `Agent.new.chat`: a fresh chat with everything declared applied.
    fn chat(&self) -> Result<Chat> {
        let chat = Chat::new(self.model(), self.provider())?;
        self.apply(chat)
    }
}

/// `prompt_agent_path`: `Admin::WorkAssistant` is `admin/work_assistant`.
pub fn prompt_agent_path(name: &str) -> String {
    crate::tool::underscore(&name.replace("::", "/")).replace('-', "_")
}
