# Changelog

RustLLM's version follows the RubyLLM release it ports: 2.1.1 ports RubyLLM 2.1.1 (upstream
`crmne/ruby_llm` at tag `v2.1.1`). Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [2.1.1] - 2026-10-13

First release: `rust_llm`, `rust_llm_loco`, and `rust_llm_cli`, a full port of RubyLLM 2.1.1.
`docs/PARITY.md` classifies all 4,684 examples in RubyLLM's spec suite: 2,991 ported as Rust tests,
176 replayed from their own recorded cassettes, 1,517 not applicable (each with a reason), 0 missing.

A 2.0.0 preview was published on crates.io before this release. Nothing in production used it, so
2.1.1 is the starting point: start new apps with `rust-llm generate install`. There is no upgrade
path from 2.0.0.

### Added
- `rust_llm`
  - Chat: `ask`, `ask_later`, `complete`, `step`, `generate`, `run_tools`, streaming, tools with
    approvals, concurrency, and cancellation, agents, structured output (`with_schema`,
    `with_schema_for::<T>()`), thinking, fallbacks, callbacks, citations, prompt caching
    (`with_caching`, `cache_until_here`, and `cache_until_here_with` for a per-boundary
    lifetime), compaction, `with_end_user`, provider tools, and isolated configuration
    (`context`).
  - Tool search: deferred tools (`add_tool_deferred`, `with_deferred_tools`) use native tool search
    on Anthropic and OpenAI Responses.
  - Operations: embeddings (text and multimodal), images (`paint`, edits incl. multipart),
    video (`animate`, `VideoJob`), speech (`speak`, streaming), transcription (incl. WebSocket
    live transcription for Gemini Live and xAI), moderation, OCR, rerank, token counting and
    tokenization, provider batches, file uploads and downloads (reusing a provider's stored copy
    of an attachment, `files::ProviderFileStore`), Gemini cached content, and judgments (`Judge`,
    `judge`) through TypeSafe Jev, OpenAI Decisions, and local Ollama decision models.
  - Attachments: `convert_unsupported_attachments`, and `Resolution::Original` for images.
  - MCP: stdio and streamable HTTP clients, tools, resources, prompts, input requests during a
    call, listening for server changes (`Mcp::listen`, `after_change`), tasks, MCP Apps, `_meta`,
    extensions, log levels, deferred tools, older protocol versions, session renewal, and
    resumable streams. OAuth with client credentials and JWT bearer grants, enterprise-managed
    authorization, and DPoP-bound tokens.
  - `rust_llm::evaluation`: run an agent over a dataset, grade correctness or your own criteria
    with a Judge, add Rust assertions, and report verdicts, progress, tokens, and cost; usable
    from `cargo test`.
  - OpenTelemetry tracing behind the `opentelemetry` feature (`rust_llm::open_telemetry::enable`):
    GenAI semantic-convention spans for model calls, tools, and workflows, without prompts or
    responses.
  - `Error::request_shape()`: a content-free description of a rejected request.
  - Prompt templates (Jinja), workflows, instrumentation events for every operation, and the
    model registry with refresh from providers and models.dev.
  - The per-attempt usage and cost ledger, with retries and fallbacks billed separately, a usage
    owner (`with_usage_owner`, per-call `owner`), provider tool counts (`web_search_requests`,
    ...), and a ledger for usage outside chats (`Config::usage_ledger`).
- Providers: OpenAI (Responses and Chat Completions), Anthropic, Gemini (incl. Interactions),
  DeepSeek, Mistral (incl. Conversations), OpenRouter, xAI, Perplexity (Agent API and Router),
  Ollama, Ollama Cloud, GPUStack, Hetzner, TypeSafe.
- `rust_llm_loco`: RubyLLM's Rails persistence for Loco/SeaORM with the same tables and columns
  (chats, messages, tool calls, usages, models, attachments, batches, MCP credentials, provider
  files), including resuming chats parked on approvals or MCP input from another process,
  cancellation, streaming persisted replies (`ChatRecord::ask_stream`, `complete_stream`), and a
  usage ledger for operations outside chats (`UsageLedger`). Write transactions use
  `BEGIN IMMEDIATE` on SQLite.
- `rust_llm_cli`: `rust-llm generate install / chat_ui / public_chat / tool / agent / schema /
  provider / upgrade` for Loco + Inertia + React apps. The chat UI streams replies over the
  starter kit's live updates and scopes chats to accounts; `public_chat` adds a rate-limited
  no-sign-in chat page.
- Verification against RubyLLM's recorded VCR cassettes (HTTP and WebSocket), with JSON-equal
  request bodies; benchmarks against RubyLLM + YJIT (`bench/`, `docs/BENCHMARK.md`).

### Differences from RubyLLM
- The bundled registry prices `jev-latest` and `jev-preview` at TypeSafe's published rate ($0.042
  per million input tokens, output free); RubyLLM's registry has no price for them.

### Not included
- Providers Bedrock, Vertex AI, Azure, Cohere, ElevenLabs, and Deepgram (and so Vertex's
  `research` jobs).
