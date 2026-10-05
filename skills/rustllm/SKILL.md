---
name: rustllm
description: Build and maintain Rust or Loco applications with the RustLLM AI framework (the rust_llm, rust_llm_loco, and rust_llm_cli crates). Use for chats, agents, tools, structured output, media generation, transcription, OCR, moderation, embeddings, reranking, batches, MCP, judgments, Loco persistence, and the rust-llm generators; not for contributing to the framework itself.
license: MIT
metadata:
  api_version: "2.0"
---

# Build with RustLLM

Use the application's installed RustLLM version and existing conventions. This skill describes the
2.0 API. Check the version before choosing an API:

```bash
cargo tree -p rust_llm --depth 0
cargo tree -i rust_llm_loco --depth 0   # present in Loco apps that persist chats
```

## Choose the API for the task

RustLLM has two public API families. Conversations use `Chat`, `Message`, tools, agents,
structured output, streaming, and loop control. Individual operations (`paint`, `animate`,
`speak`, `transcribe`, `ocr`, `moderate`, `embed`, `rerank`, `judge`) return typed results without
a chat. Providers and protocols translate requests and responses; build against the public API and
let RustLLM handle each provider's format.

`rust_llm_loco` adds SeaORM persistence for Loco apps. `rust_llm_cli` (the `rust-llm` binary)
generates the Loco + Inertia + React integration.

## Find the relevant API

The installed crate's rustdoc (`cargo doc -p rust_llm --open`, or <https://docs.rs/rust_llm>) is
the source of truth for signatures. Read only the guide you need from
<https://github.com/cole-robertson/rust-llm/tree/main/docs>:

| Task | Guide |
| --- | --- |
| Configure a provider or select a model | `configuration.md` |
| Chat, stream, or attach files | `chat.md`, `streaming.md`, `attachments-and-files.md` |
| Declare tools or require approval | `tools.md` |
| Parse structured output | `structured-output.md` |
| Build reusable agents or prompt templates | `agents.md`, `prompts.md` |
| Generate images or video | `images.md`, `video.md` |
| Generate speech or transcribe audio | `audio.md` |
| OCR, moderation, reranking, tokenization | `moderation-ocr-rerank.md` |
| Embeddings | `embeddings.md` |
| MCP servers | `mcp.md` |
| Judgments (probabilities, choices, scores) | `judgments.md` |
| Batches | `batches.md` |
| Thinking | `thinking.md` |
| Track spend or handle failures | `cost-and-usage.md`, `errors-and-retries.md` |
| Events and workflows | `instrumentation.md` |
| Persist chats in Loco | `persistence-loco.md` |
| Generators | `generators.md` |
| Coming from RubyLLM | `rubyllm.md` |

## Use the 2.0 API consistently

- Everything that talks to a provider is `async` and returns `rust_llm::Result`. Run it on tokio.
- Configure credentials with `rust_llm::configure(|c| { ... })` or environment variables
  (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, ...), following the application's secret storage. Verify
  explicit model ids with `rust_llm::models().find(id, Some(provider))`; do not invent ids or use
  `assume_model_exists` to hide a typo.
- Start a conversation with `rust_llm::chat()?` or `rust_llm::chat_with(model)?`. Builders take
  `self` and return the chat (`.with_instructions(..)`, `.with_tool(..)`, `.with_temperature(..)`),
  so chain or rebind them; methods that run the conversation take `&mut self`.
- Implement the `Agent` trait when instructions, tools, or options should be reused, and start it
  with `MyAgent.chat()?`.
- `ask` runs the conversation and returns a `Message`. `ask_later` stages input. For explicit loop
  control use `generate`, `run_tools`, `step`, and `is_complete`. A tool whose
  `requires_approval()` is true parks the loop: check `is_awaiting_approval()`, call `approve` or
  `deny` with the tool call id, then `complete().await`.
- Read text with `message.content()` and structured output with `message.parsed()?`
  (`Option<serde_json::Value>`). Declare schemas by deriving `schemars::JsonSchema` and calling
  `with_schema_for::<T>()`.
- Implement `rust_llm::Tool` with `#[async_trait::async_trait]`: `description`, `parameters`
  (`Parameter::new(..)`) or `parameters_schema`, and `execute`. Return
  `Ok(ToolResult::error(..))` for problems the model can recover from; `Err` stops the
  conversation. Keep application authorization inside the tool.
- Use shared options (`with_thinking`, `with_caching`, `with_citations`, `with_fallbacks`) before
  `with_provider_options`, which is for provider-specific request fields. Do not add a second
  provider SDK for something RustLLM already does.
- Read tokens through `message.tokens()` and cost through `message.cost(None).total()` or
  `chat.cost().total()`. Unknown cost is `None`, not zero. The ledger records every attempt,
  including retries and fallbacks.
- Ruby predicates are `is_*` methods; Ruby symbols are enums (`FinishReason::Stop`,
  `ToolChoice::Required`).

## Integrate with Loco

For a new integration run `rust-llm generate install` (dependencies, migration, initializer,
`Chat`/`Message` models) and optionally `rust-llm generate chat_ui`. Review the migration and
follow the application's migration workflow.

Persisted chats use `rust_llm_loco::ChatRecord`: `ChatRecord::create(&db, model, None)`, then
`record.to_llm(&db).await?` to rebuild the `Chat`, and `record.ask(&db, &mut chat, text)` to run it
while writing every message, tool call, and usage row. Runtime settings (tools, callbacks,
temperature) are not stored; reapply them after `to_llm`, or keep them in an `Agent`. A chat parked
on an approval resumes from rows alone in another request or worker: `ChatRecord::find`,
`to_llm`, `record.approve(..)`, `record.complete(..)`.

## Verify the application change

Run the application's tests. Test tool behavior, structured parsing, persisted conversation
reconstruction, and job resumption where the change depends on them. Use recorded responses or a
mock provider rather than live calls in tests, and report which provider behavior was actually
exercised. Keep the change within the requested feature and the existing architecture.
