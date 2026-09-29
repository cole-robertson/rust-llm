# ruby_llm-rs

A 1:1 Rust port of [RubyLLM](https://github.com/crmne/ruby_llm) **2.0.0** (upstream `1e91b30`),
plus `ruby_llm_loco`, the port of its Rails `acts_as_chat` layer for Loco's default ORM, SeaORM.

The API keeps RubyLLM's shape, and the wire format matches it exactly. The tests replay RubyLLM's
own recorded VCR cassettes, and every request this crate sends must equal the body RubyLLM recorded.

```rust
// RubyLLM.chat.ask "What's the best way to learn Ruby?"
ruby_llm::chat()?.ask("What's the best way to learn Rust?").await?;

// chat.with_tools(Weather).ask "What's the weather in Berlin?"
let mut chat = ruby_llm::chat_with("claude-haiku-4-5")?.with_tool(Weather);
chat.ask("What's the weather in Berlin?").await?;

// chat.ask("Tell me a story") { |chunk| print chunk.content }
chat.ask_stream("Tell me a story", |chunk| print!("{}", chunk.content())).await?;

// chat.with_schema(ProductSchema).ask(...).parsed
chat.with_schema_for::<Product>().ask("Analyze this product").await?.parsed()?;

// WeatherAssistant.new.ask "..."   (impl Agent for WeatherAssistant { fn model, fn instructions, fn tools })
WeatherAssistant.chat()?.ask("What's the weather in Berlin?").await?;

// RubyLLM.embed "Ruby is elegant and expressive"
ruby_llm::embed("Rust is fast and safe", Default::default()).await?.vectors;
```

`cargo run -p ruby_llm --example readme` runs the whole RubyLLM README against a live provider.

## Mapping

| RubyLLM | ruby_llm |
|---|---|
| `RubyLLM.configure { \|c\| c.openai_api_key = ... }` | `ruby_llm::configure(\|c\| { c.openai_api_key(...); })`. Env vars like `OPENAI_API_KEY` are read automatically |
| `RubyLLM.chat(model:, provider:)` | `Chat::new(Some(model), Some(provider))`, `ruby_llm::chat()` |
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
| `RubyLLM.models.find`, `aliases.json`, provider preference | `ruby_llm::models().find(id, provider)`, same bundled `models.json`/`aliases.json` |
| `Provider` + `Protocols::{ChatCompletions, Responses, Anthropic, Gemini}` | `Provider` enum + `protocols::{chat_completions, responses, anthropic, gemini}` |
| Error classes + `ErrorMiddleware` status/pattern mapping, Faraday retry | `Error` enum, same mapping; same retry rules (never retries a stream that already delivered) |
| `acts_as_chat` / `acts_as_message` / `acts_as_tool_call`, `ruby_llm_models`, `ruby_llm_usages` | `ruby_llm_loco::ChatRecord`; same tables and columns via SeaORM migrations |

Providers: OpenAI (Responses by default, Chat Completions for audio/search models), Anthropic,
Gemini, DeepSeek, Mistral, OpenRouter, xAI, Perplexity (Agent API), Ollama, Ollama Cloud,
GPUStack, Hetzner.

## Loco / SeaORM

```rust
// migration/src/lib.rs
migrations.extend(ruby_llm_loco::migrations());

// a controller or job
let record = ChatRecord::create(&ctx.db, "claude-haiku-4-5", None).await?;
let mut chat = record.to_llm(&ctx.db).await?.with_tool(Weather);
record.ask(&ctx.db, &mut chat, "What's the weather in Berlin?").await?;
// rows: messages (user, assistant+tool call, tool, assistant), ruby_llm_tool_calls, ruby_llm_usages

// a chat parked on requires_approval resumes from rows alone, in another request or job
let mut chat = ChatRecord::find(&ctx.db, id).await?.to_llm(&ctx.db).await?.with_tool(DeleteEverything);
record.approve(&ctx.db, &mut chat, &tool_call_id).await?;
record.complete(&ctx.db, &mut chat).await?;
```

## Verification

| Check | What it proves |
|---|---|
| `tests/cassette_replay.rs`: 11 specs × 12 providers | basic, multi-turn, system prompt, replace instructions, raw responses, streaming, tools, multi-turn tools, parameterless tools, streaming tools, parallel tool calls: request bodies are JSON-equal to RubyLLM's recordings, and responses satisfy the Ruby spec's assertions |
| `tests/cassette_replay_more.rs` | JSON schema, typed schema (vs Schematist), removing a schema mid-chat, embeddings (single, batch, dimensions), human-readable auth errors |
| `ruby_llm_loco/tests/acts_as_chat.rs` | a replayed tool-calling chat writes the same rows as RubyLLM, reloads intact, and an approval-parked chat resumes from the DB |
| unit tests | error mapping, cost/tier pricing, token aggregation, aliases and provider preference, tool naming, SSE framing |
| `examples/readme.rs` | the README run live against Anthropic |

`bin/fw cargo test --workspace` builds and tests on the `framework` box; `bin/convert-cassettes`
turns more upstream cassettes into test fixtures.

## Not ported yet

These are listed so nothing is silently missing:
- Bedrock, Vertex AI, and Azure (cloud auth); Cohere; ElevenLabs; Deepgram; TypeSafe/`Judge`.
- `paint`, `animate`, `speak`, `transcribe`, `ocr`, `rerank`, `moderate`, and `batch`.
- MCP; provider tools (`with_provider_tools`); `with_compaction`; `count_tokens`.
- Provider file uploads, and prompt-caching options beyond `cache_until_here`.
- Generators; Active Storage attachments on persisted messages.
- `with_citations` as a request option. Citations in responses are parsed.
