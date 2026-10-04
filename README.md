# RustLLM

A 1:1 Rust port of [RubyLLM](https://github.com/crmne/ruby_llm) **2.0.0** (upstream `1e91b30`),
plus `rust_llm_loco`, the port of its Rails `acts_as_chat` layer for Loco's default ORM, SeaORM.

The API keeps RubyLLM's shape, and the wire format matches it exactly. The tests replay RubyLLM's
own recorded VCR cassettes, and every request this crate sends must equal the body RubyLLM recorded.

```rust
// RubyLLM.chat.ask "What's the best way to learn Ruby?"
rust_llm::chat()?.ask("What's the best way to learn Rust?").await?;

// chat.with_tools(Weather).ask "What's the weather in Berlin?"
let mut chat = rust_llm::chat_with("claude-haiku-4-5")?.with_tool(Weather);
chat.ask("What's the weather in Berlin?").await?;

// chat.ask("Tell me a story") { |chunk| print chunk.content }
chat.ask_stream("Tell me a story", |chunk| print!("{}", chunk.content())).await?;

// chat.with_schema(ProductSchema).ask(...).parsed
chat.with_schema_for::<Product>().ask("Analyze this product").await?.parsed()?;

// WeatherAssistant.new.ask "..."   (impl Agent for WeatherAssistant { fn model, fn instructions, fn tools })
WeatherAssistant.chat()?.ask("What's the weather in Berlin?").await?;

// RubyLLM.embed "Ruby is elegant and expressive"
rust_llm::embed("Rust is fast and safe", Default::default()).await?.vectors;

// RubyLLM.paint "a sunset over mountains in watercolor style"
rust_llm::paint("a sunset over mountains in watercolor style", Default::default()).await?;

// chats = tickets.map { |t| RubyLLM.chat.ask_later(t.body) }; RubyLLM.batch(chats)
let batch = rust_llm::batch(chats).await?;   // later: batch.refresh().await?; batch.messages()

// file = RubyLLM.upload("contract.pdf"); chat.ask "Summarize", with: file
let file = rust_llm::upload("contract.pdf", Default::default()).await?;
chat.ask_with("Summarize this", vec![file.into()]).await?;

// class Urgency < RubyLLM::Judge; probability :urgent, "Does this need attention today?"; end
let urgency = rust_llm::Judge::new().probability("urgent", "Does this need attention today?")?;
urgency.judge("Please refund the duplicate charge today.").await?.probability("urgent"); // => Some(0.91)

// chat.with_mcp(RubyLLM.mcp(command: ["npx", "-y", "@modelcontextprotocol/server-github"]))
let github = rust_llm::mcp::Mcp::command(["npx", "-y", "@modelcontextprotocol/server-github"]).build()?;
chat.with_mcp(github).ask("List my open PRs").await?;
```

`cargo run -p rust_llm --example readme` runs the whole RubyLLM README against a live provider.

## Mapping

| RubyLLM | rust_llm |
|---|---|
| `RubyLLM.configure { \|c\| c.openai_api_key = ... }` | `rust_llm::configure(\|c\| { c.openai_api_key(...); })`. Env vars like `OPENAI_API_KEY` are read automatically |
| `RubyLLM.chat(model:, provider:)` | `Chat::new(Some(model), Some(provider))`, `rust_llm::chat()` |
| `ask`, `say`, `ask_later`, `complete`, `step`, `generate`, `run_tools`, `complete?` | same names; `is_complete()` |
| `ask(msg, with: files)` | `ask_with(msg, vec![Attachment::new("ruby.png")])` |
| `ask(msg) { \|chunk\| }` | `ask_stream(msg, \|chunk\| ..)` |
| `with_instructions`, `with_tools`, `with_tool_options(choice:, calls:)`, `with_model`, `with_temperature`, `with_max_output_tokens`, `with_thinking`, `with_schema`, `with_provider_options`, `with_headers`, `with_fallbacks` | same names, builder style |
| `before_message`, `after_message`, `before_tool_call`, `after_tool_result`, `before_fallback`, `after_fallback`, `before_request` | same names, closures |
| `approve`, `deny`, `awaiting_approval?`, `pending_approvals`, `cancel` | same; `CancelHandle` for other tasks |
| `class Weather < RubyLLM::Tool` (`description`, `parameter`, `execute`, `requires_approval`) | `impl Tool for Weather`, `Parameter::new(..)`, or a `schemars` schema |
| `class X < RubyLLM::Agent` | `impl Agent for X` |
| `Message`, `Chunk`, `ToolCall`, `Tokens`, `Cost`, `Thinking`, `Citation`, `Attachment` | same types |
| `chat.tokens`, `chat.cost` (per-attempt usage ledger) | same; retries and fallbacks each get a `UsageEntry` |
| `RubyLLM.models.find`, `aliases.json`, provider preference | `rust_llm::models().find(id, provider)`, same bundled `models.json`/`aliases.json` |
| `Provider` + `Protocols::{ChatCompletions, Responses, Anthropic, Gemini}` | `Provider` enum + `protocols::{chat_completions, responses, anthropic, gemini}` |
| Error classes + `ErrorMiddleware` status/pattern mapping, Faraday retry | `Error` enum, same mapping; same retry rules (never retries a stream that already delivered) |
| `acts_as_chat` / `acts_as_message` / `acts_as_tool_call`, `rust_llm_models`, `rust_llm_usages` | `rust_llm_loco::ChatRecord`; same tables and columns via SeaORM migrations |
| `RubyLLM.paint` (generate, edit with reference images) | `rust_llm::paint` → `Image` (`save`, `to_blob`, `cost`) |
| `RubyLLM.batch`, `Batch.find`, `embed_later` | `rust_llm::batch`, `Batch::find`, `embed_later`; batch-tier pricing |
| `RubyLLM.upload` / `download`, `UploadedFile`, auto-upload of large attachments | `rust_llm::upload` / `download`, `UploadedFile` |
| `RubyLLM.mcp`, `chat.with_mcp`, MCP prompts/resources, input requests | `rust_llm::mcp` (stdio + streamable HTTP), `with_mcp`, `answer`/`decline` |
| `with_provider_tools` (web search, code execution, remote MCP) | `with_provider_tools` |
| `RubyLLM::Judge` (`probability`/`choice`/`score`, `inputs`), `RubyLLM.judge(questions:)`, TypeSafe/Jev and Jev-compatible servers | `rust_llm::Judge`, `rust_llm::judge`, `Provider::TypeSafe` (`typesafe_api_key`/`typesafe_api_base`, `default_judgment_model = "jev-latest"`) |
| `rails g ruby_llm:install / tool / agent / schema / chat_ui / provider / upgrade` | `rust-llm generate install / tool / agent / schema / chat_ui / provider / upgrade` (Loco + Inertia + React + shadcn) |

Providers: OpenAI (Responses by default, Chat Completions for audio/search models), Anthropic,
Gemini, DeepSeek, Mistral, OpenRouter, xAI, Perplexity (Agent API), Ollama, Ollama Cloud,
GPUStack, Hetzner, and TypeSafe (Jev judgments over the System One API).

## Loco / SeaORM

```rust
// migration/src/lib.rs
migrations.extend(rust_llm_loco::migrations());

// a controller or job
let record = ChatRecord::create(&ctx.db, "claude-haiku-4-5", None).await?;
let mut chat = record.to_llm(&ctx.db).await?.with_tool(Weather);
record.ask(&ctx.db, &mut chat, "What's the weather in Berlin?").await?;
// rows: messages (user, assistant+tool call, tool, assistant), rust_llm_tool_calls, rust_llm_usages

// a chat parked on requires_approval resumes from rows alone, in another request or job
let mut chat = ChatRecord::find(&ctx.db, id).await?.to_llm(&ctx.db).await?.with_tool(DeleteEverything);
record.approve(&ctx.db, &mut chat, &tool_call_id).await?;
record.complete(&ctx.db, &mut chat).await?;
```

## Verification

- **Cassette replay:** tests replay RubyLLM's own recorded VCR cassettes (HTTP and WebSocket), and
  every request RustLLM sends must be JSON-equal to the one RubyLLM recorded.
- **Spec parity:** every applicable RubyLLM spec example is ported as a Rust test that cites it
  (`// spec: file:line`); see the parity section below.
- **Live:** `examples/readme.rs` (chat, streaming, tools, agents, structured output, cost) and
  `examples/judge.rs` (TypeSafe Jev) run against real providers.
- **Benchmarks:** [`docs/BENCHMARK.md`](docs/BENCHMARK.md) compares RustLLM with RubyLLM + YJIT.

Over 2,000 tests; CI runs them with rustfmt, `clippy -D warnings`, docs, MSRV builds, and a
package check. `bin/fw cargo test --workspace` runs them on the `framework` build box;
`bin/convert-cassettes` turns more upstream cassettes into fixtures.

## Generators

`cargo install rust_llm_cli`, then from a Loco app:

```
rust-llm generate install        # deps, migration, initializer, Chat/Message models
rust-llm generate chat_ui        # Inertia + React chat pages, controllers, a Loco worker
rust-llm generate tool Weather   # src/tools/weather_tool.rs + React tool call/result components
rust-llm generate agent Support  # src/agents/support.rs + src/prompts/support/instructions.txt
rust-llm generate schema Product
```

Verified against a copy of the Loco + Inertia starter kit: it builds with no warnings, migrations
run up and down, the frontend passes check/lint/build, and a chat round-trips through the worker.
The chat UI polls for new messages while a reply is pending; it doesn't stream tokens.

## Parity and what is left out

[`docs/PARITY.md`](docs/PARITY.md) classifies all 3,747 examples in RubyLLM 2.0's spec suite:
2,157 ported as Rust tests, 177 replayed from their own recorded cassettes, 1,413 not applicable
(each with its reason), and **0 missing**. `bin/parity` regenerates it.

Deliberately left out:
- Providers: Bedrock, Vertex AI, Azure, Cohere, ElevenLabs, Deepgram (and so Vertex's
  `research` / `research_later`).
- Ruby-only mechanics with no Rust counterpart (ActiveRecord internals, Faraday adapters,
  metaprogrammed DSLs). The persistence layer covers the same behavior with SeaORM.

Differences worth knowing:
- Prompt templates are Jinja (minijinja) rather than ERB; see `docs/prompts.md`.
- Instrumentation events end in `.rust_llm` instead of `.ruby_llm`; payload keys match.
- MCP OAuth credentials stored by `rust_llm_loco` are encrypted with AES-256-GCM, but not in a
  format Rails can read (and vice versa).
