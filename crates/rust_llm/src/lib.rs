//! # rust_llm
//!
//! A 1:1 Rust port of [RubyLLM](https://github.com/cole-robertson/rust_llm) 2.0: one API for chat, tools, agents,
//! structured output, streaming, and embeddings across providers.
//!
//! ```ruby
//! RubyLLM.chat.ask "What's the best way to learn Ruby?"
//! ```
//!
//! ```no_run
//! # async fn run() -> rust_llm::Result<()> {
//! rust_llm::chat()?.ask("What's the best way to learn Rust?").await?;
//! # Ok(()) }
//! ```
//!
//! The module layout mirrors `lib/ruby_llm`: `chat`, `message`, `tool`, `agent`, `models`,
//! `providers`, `protocols`, `transport`, `tokens`, `cost`. Names follow Ruby where Rust allows
//! (`ask`, `ask_later`, `complete`, `step`, `with_tools`, `before_tool_call`, ...), with `?`-suffixed
//! predicates spelled `is_*`.

pub mod agent;
pub mod attachment;
pub mod batch;
pub mod chat;
pub mod config;
pub mod cost;
pub mod embedding;
pub mod error;
pub mod files;
pub mod image;
pub mod judge;
pub mod mcp;
pub mod message;
pub mod model;
pub mod models;
pub mod progress;
pub mod protocols;
pub mod provider_tools;
pub mod providers;
pub mod thinking;
pub mod tokens;
pub mod tool;
pub mod transport;

pub use agent::Agent;
pub use attachment::{Attachment, Resolution};
pub use batch::{Batch, BatchResult, BatchStatus, EmbeddingRequest, batch, embed_later};
pub use chat::{CancelHandle, Chat, Fallback, FallbackAttempt};
pub use config::{Config, config, configure};
pub use cost::Cost;
pub use embedding::{EmbedOptions, Embedding, Vectors, embed};
pub use error::{Error, ErrorKind, Result};
pub use files::{DownloadedFile, FileOptions, UploadOptions, UploadedFile, download, upload};
pub use judge::{Answer, Dynamic, Judge, JudgeOptions, Judgment, QuestionType, judge, list_judgment_models};
pub use image::{Image, Images, PaintOptions, paint};
pub use mcp::{InputRequest, Mcp, McpResult, McpTool};
pub use progress::Progress;
pub use provider_tools::ProviderTool;
pub use message::{Chunk, Citation, FinishReason, Message, Role, Thinking, ToolCall, UsageEntry, UsageStatus};
pub use model::Model;
pub use models::models;
pub use protocols::{ToolCalls, ToolChoice};
pub use providers::{ProtocolName, Provider};
pub use thinking::{Display as ThinkingDisplay, ThinkingConfig};
pub use tokens::Tokens;
pub use tool::{FnTool, Parameter, SharedTool, Tool, ToolError, ToolResult, schema_for};

/// `RubyLLM.chat` with the configured default model.
pub fn chat() -> Result<Chat> {
    Chat::new(None, None)
}

/// `RubyLLM.chat(model:)`.
pub fn chat_with(model: &str) -> Result<Chat> {
    Chat::new(Some(model), None)
}

/// `RubyLLM::VERSION` of the port's source.
pub const VERSION: &str = "2.0.0";
