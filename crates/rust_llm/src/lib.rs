//! # rust_llm
//!
//! A 1:1 Rust port of [RubyLLM](https://github.com/crmne/ruby_llm) 2.0: one API for chat, tools,
//! agents, structured output, streaming, embeddings, images, batches, MCP, and judgments across
//! providers. Requests match RubyLLM's wire format exactly; the tests replay RubyLLM's own recorded
//! cassettes.
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
//! ## Features
//!
//! - [`Chat`]: conversations with instructions, model switching, temperature, provider options,
//!   headers, request hooks, callbacks, fallbacks, and cancellation. Start with [`chat()`] or
//!   [`chat_with`].
//! - [`Tool`]: tools the model can call, with [`Parameter`] declarations or a `schemars` schema
//!   ([`schema_for`]), approvals, and progress. [`ProviderTool`]s run on the provider.
//! - [`Agent`]: a chat configuration declared once and reused.
//! - Streaming ([`Chat::ask_stream`]), structured output ([`Chat::with_schema_for`]), and extended
//!   thinking ([`ThinkingConfig`]).
//! - Attachments ([`Attachment`]) and provider Files APIs ([`upload`], [`download`]).
//! - [`embed`], [`paint`], [`batch()`] and [`embed_later`], [`judge()`] and [`Judge`].
//! - MCP servers ([`mod@mcp`], [`Mcp`]).
//! - Per-attempt [`Tokens`] and [`Cost`] on every result.
//!
//! Providers: OpenAI, Anthropic, Gemini, DeepSeek, Mistral, OpenRouter, xAI, Perplexity, Ollama,
//! Ollama Cloud, GPUStack, Hetzner, and TypeSafe. Configure them with [`configure`]; keys such as
//! `OPENAI_API_KEY` are read from the environment automatically.
//!
//! ## Guides
//!
//! The [user guides](https://github.com/cole-robertson/rust-llm/tree/main/docs) follow RubyLLM's
//! documentation, with the Ruby next to the Rust. The
//! [migration guide](https://github.com/cole-robertson/rust-llm/blob/main/docs/migrating-from-rubyllm.md)
//! maps the whole Ruby API to this crate.
//!
//! The module layout mirrors `lib/ruby_llm`: `chat`, `message`, `tool`, `agent`, `models`,
//! `providers`, `protocols`, `transport`, `tokens`, `cost`. Names follow Ruby where Rust allows
//! (`ask`, `ask_later`, `complete`, `step`, `with_tools`, `before_tool_call`, ...), with `?`-suffixed
//! predicates spelled `is_*`.

pub mod accounting;
pub mod agent;
pub mod attachment;
pub mod batch;
pub mod cached_content;
pub mod chat;
pub mod config;
pub mod context;
pub mod cost;
pub mod embedding;
pub mod error;
pub mod files;
pub mod image;
pub mod instrumentation;
pub mod judge;
pub mod mcp;
pub mod message;
pub mod model;
pub mod models;
pub mod moderation;
pub mod ocr;
pub mod progress;
pub mod prompt;
pub mod protocols;
pub mod provider_tools;
pub mod search_results;
pub mod speech;
pub mod providers;
pub mod rerank;
pub mod thinking;
pub mod tokenization;
pub mod tokens;
pub mod tool;
pub mod transcription;
pub mod transport;
pub mod video;
pub mod workflow;

pub use agent::Agent;
pub use attachment::{Attachment, Resolution};
pub use cached_content::{CacheOptions, CachedContent, Ttl, cache};
pub use batch::{Batch, BatchResult, BatchStatus, EmbeddingRequest, batch, embed_later};
pub use chat::{CancelHandle, Chat, Fallback, FallbackAttempt};
pub use config::{Config, config, configure};
pub use context::{Context, context};
pub use cost::Cost;
pub use embedding::{EmbedOptions, Embedding, Vectors, embed};
pub use error::{Error, ErrorKind, Result};
pub use files::{DownloadedFile, FileOptions, UploadOptions, UploadedFile, download, upload};
pub use judge::{Answer, Dynamic, Judge, JudgeOptions, Judgment, QuestionType, judge, list_judgment_models};
pub use image::{Image, Images, PaintOptions, paint};
pub use mcp::{InputRequest, Mcp, McpResult, McpTool};
pub use progress::Progress;
pub use prompt::{Prompt, render_prompt};
pub use instrumentation::{Instrumenter, instrument};
pub use workflow::{Workflow, workflow};
pub use provider_tools::ProviderTool;
pub use search_results::SearchResults;
pub use speech::{SpeakOptions, Speech, SpeechChunk, speak, speak_stream};
pub use message::{Chunk, Citation, FinishReason, Message, Role, Thinking, ToolCall, UsageEntry, UsageStatus};
pub use model::Model;
pub use models::models;
pub use moderation::{ModerateOptions, Moderation, ModerationInput, ModerationResult, moderate};
pub use ocr::{Ocr, OcrOptions, Page as OcrPage, ocr};
pub use rerank::{Rerank, RerankOptions, RerankResult, rerank};
pub use tokenization::{TokenizeOptions, Tokenization, tokenize};
pub use video::{AnimateOptions, Video, VideoJob, VideoSource, VideoStatus, animate, animate_later};
pub use protocols::{Caching, ToolCalls, ToolChoice};
pub use providers::{ProtocolName, Provider};
pub use thinking::{Display as ThinkingDisplay, ThinkingConfig};
pub use tokens::Tokens;
pub use tool::{FnTool, Parameter, SharedTool, Tool, ToolError, ToolResult, schema_for};
pub use transcription::{TranscribeOptions, Transcription, TranscriptionChunk, transcribe, transcribe_stream};

/// `RubyLLM.chat` with the configured default model.
pub fn chat() -> Result<Chat> {
    Chat::new(None, None)
}

/// `RubyLLM.chat(model:)`.
pub fn chat_with(model: &str) -> Result<Chat> {
    Chat::new(Some(model), None)
}

/// `RubyLLM.count_tokens(text, model:, provider:)`: input tokens for one user message.
pub async fn count_tokens(text: &str, model: Option<&str>, provider: Option<&str>) -> Result<i64> {
    Chat::new(model, provider)?.count_tokens(Some(text)).await
}

/// `RubyLLM::VERSION` of the port's source.
pub const VERSION: &str = "2.0.0";
