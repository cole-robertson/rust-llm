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
use crate::error::Result;
use crate::protocols::ToolChoice;
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
    fn instructions(&self) -> Option<String> {
        None
    }
    fn tools(&self) -> Vec<SharedTool> {
        Vec::new()
    }
    fn tool_choice(&self) -> Option<ToolChoice> {
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
    fn provider_options(&self) -> Option<Value> {
        None
    }
    fn fallbacks(&self) -> Vec<Fallback> {
        Vec::new()
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
    /// a persisted chat record picks its agent back up.
    fn apply(&self, mut chat: Chat) -> Result<Chat> {
        if let Some(text) = self.instructions() {
            chat.set_instructions(Some(text), false, false);
        }
        chat = chat.with_tools(self.tools());
        for server in self.mcp() {
            chat = chat.with_mcp(server);
        }
        chat = chat.with_provider_tools(self.provider_tools());
        if let Some(choice) = self.tool_choice() {
            chat = chat.with_tool_choice(choice)?;
        }
        if let Some(t) = self.temperature() {
            chat = chat.with_temperature(t);
        }
        if let Some(m) = self.max_output_tokens() {
            chat = chat.with_max_output_tokens(m);
        }
        if let Some(t) = self.thinking() {
            chat = chat.with_thinking(t);
        }
        if let Some(s) = self.schema() {
            chat = chat.with_schema(s);
        }
        if let Some(o) = self.provider_options() {
            chat = chat.with_provider_options(o);
        }
        let fallbacks = self.fallbacks();
        if !fallbacks.is_empty() {
            chat = chat.with_fallbacks(fallbacks);
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
