# Changelog

RustLLM's version follows the RubyLLM release it ports: 2.0.x ports RubyLLM 2.0 (upstream
`crmne/ruby_llm` at `1e91b30`); 2.1.x ports RubyLLM 2.1 (tag `v2.1.0`). Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

Ports RubyLLM 2.1.0. `docs/PARITY.md` classifies all 4,658 examples in its spec suite: 2,966 ported
as Rust tests, 176 replayed from their own recorded cassettes, 1,516 not applicable (each with a
reason), 0 missing. Existing apps: run `rust-llm generate upgrade` and `cargo loco db migrate`.

### Added (RubyLLM 2.1)
- MCP: listening for server changes (`Mcp::listen`, `after_change`) with backoff and catch-up;
  tasks; MCP Apps and `_meta`; client extensions; answering a server's input requests during a
  call, with form defaults and declared accepted requests; per-request log levels; deferred MCP
  tools; protocol version negotiation across older servers; session expiry and renewal; resuming
  or resending broken HTTP streams; nested parameter headers.
- MCP OAuth: client credentials and JWT bearer grants, enterprise-managed authorization, DPoP-bound
  tokens with a proof on every request, scope-aware registration, re-registration after
  `invalid_client`, and one refresh at a time per grant (across processes with
  `rust_llm_loco::McpCredentialStore`).
- `rust_llm::evaluation`: run an agent over a dataset, grade correctness or your own criteria with a
  Judge, add Rust assertions, and report verdicts, progress, tokens, and cost; usable from
  `cargo test`.
- OpenTelemetry tracing behind the `opentelemetry` feature (`rust_llm::open_telemetry::enable`):
  GenAI semantic-convention spans for model calls, tools, and workflows, without prompts or
  responses.
- `Error::request_shape()`: a content-free description of a rejected request.
- Tool search: deferred tools (`add_tool_deferred`, `with_deferred_tools`) use native tool search
  on Anthropic and OpenAI Responses.
- Judgments through OpenAI Decisions and local Ollama decision models.
- `convert_unsupported_attachments`, `Resolution::Original` for images, and reusing (or uploading
  again) a provider's stored copy of an attachment (`files::ProviderFileStore`).
- `cache_until_here_with` sets a cache lifetime per boundary; one-hour cache writes are priced at
  their own rate (`Tokens::cache_write_by_ttl`).
- Usage outside chats: a usage owner (`with_usage_owner`, per-call `owner`) and a usage ledger
  (`Config::usage_ledger`, `rust_llm_loco::UsageLedger`) for one-shot operations and video jobs.
- Provider tool use counts in one shape (`web_search_requests`, ...) persisted with each usage row.
- `rust_llm_loco`: persisted provider files, `cache_ttl`, `mcp_state`/`mcp_result`; the upgrade
  generator adds all of them.

### Changed (RubyLLM 2.1)
- Native reasoning, signatures, and raw content go back only to the model that produced them, so
  switching providers mid-chat or resuming an interrupted job sends valid requests.
- Gemini sends each tool call with its own thought signature; Interactions can continue another
  protocol's tool round.
- One HTTP client is shared per provider, base URL, timeout, and proxy, with credentials sent per
  request; streams are parsed by a new event-stream parser; buffered error bodies are capped.
- Errors are classified more precisely: rejected keys, exhausted credit, quota 429s, oversized
  requests, and stream error codes.
- OpenAI Responses keeps system messages first and in order; Mistral returns images from its hosted
  tools; video jobs report the provider's billed cost once; blocked Gemini attempts keep their
  billed tokens.
- Local limits RubyLLM 2.1 removed are gone (left to the providers).

### Added
- `rust_llm_loco`: `ChatRecord::ask_stream` and `complete_stream`, RubyLLM's streaming block on a
  persisted chat. They persist like `ask`/`complete` and report `StreamEvent::NewMessage`,
  `Chunk`, and `EndMessage` with the row each belongs to; a failed reply leaves no empty row.
- `rust-llm generate chat_ui` streams: the worker runs `complete_stream` and broadcasts on a
  generated `ChatChannel` (the starter kit's live updates), and the page shows the reply as it
  arrives instead of polling. Chats belong to an account (`chats.account_id`, under
  `/{account_slug}/chats`); another account's chat is a 404 and its stream refuses subscription.
- `rust-llm generate public_chat`: a no-sign-in chat page at `/chat` that streams over Server-Sent
  Events, with per-IP and per-conversation rate limits, input and output caps, and a turn limit.

### Changed
- `rust_llm_loco` opens write transactions with `BEGIN IMMEDIATE` on SQLite, so a worker streaming a
  reply while the app serves requests waits for the lock instead of failing with
  `SQLITE_BUSY_SNAPSHOT`.
- The bundled registry prices `jev-latest` and `jev-preview` at TypeSafe's published rate ($0.042
  per million input tokens, output free), so `judgment.cost().total()` is known. RubyLLM's
  registry has no price for them; `rust_llm::models::refresh` still replaces the bundled copy
  with the published catalog's.

### Docs
- Judgments: bounded concurrency with `buffer_unordered` and a cloned `Judge`, and how retries
  add up for a batch.
- Configuration: a `Config` built from `Config::default()` for tests, with no keys from the
  environment. Chat: `Context::new(config).chat(..)` for your own configuration.

## [2.0.0] - unreleased

First release: `rust_llm`, `rust_llm_loco`, and `rust_llm_cli`, a full port of RubyLLM 2.0.
`docs/PARITY.md` classifies all 3,747 examples in RubyLLM's spec suite: 2,157 ported as Rust tests,
177 replayed from their own recorded cassettes, 1,413 not applicable (each with a reason), 0 missing.

### Added
- `rust_llm`
  - Chat: `ask`, `ask_later`, `complete`, `step`, `generate`, `run_tools`, streaming, tools with
    approvals, concurrency, and cancellation, agents, structured output (`with_schema`,
    `with_schema_for::<T>()`), thinking, fallbacks, callbacks, citations, prompt caching
    (`with_caching`, `cache_until_here`), compaction, `with_end_user`, provider tools, and
    isolated configuration (`context`).
  - Operations: embeddings (text and multimodal), images (`paint`, edits incl. multipart),
    video (`animate`, `VideoJob`), speech (`speak`, streaming), transcription (incl. WebSocket
    live transcription for Gemini Live and xAI), moderation, OCR, rerank, token counting and
    tokenization, provider batches, file uploads and downloads, Gemini cached content, and
    judgments (`Judge`, `judge`, TypeSafe/Jev).
  - MCP: stdio and streamable HTTP clients, tools, resources, prompts, input requests, and OAuth.
  - Prompt templates (Jinja), workflows, instrumentation events for every operation, and the
    model registry with refresh from providers and models.dev.
  - The per-attempt usage and cost ledger, with retries and fallbacks billed separately.
- Providers: OpenAI (Responses and Chat Completions), Anthropic, Gemini (incl. Interactions),
  DeepSeek, Mistral (incl. Conversations), OpenRouter, xAI, Perplexity (Agent API and Router),
  Ollama, Ollama Cloud, GPUStack, Hetzner, TypeSafe.
- `rust_llm_loco`: RubyLLM's Rails persistence for Loco/SeaORM with the same tables and columns
  (chats, messages, tool calls, usages, models, attachments, batches, MCP credentials), including
  resuming chats parked on approvals or MCP input from another process, and cancellation.
- `rust_llm_cli`: `rust-llm generate install / chat_ui / tool / agent / schema / provider /
  upgrade` for Loco + Inertia + React apps.
- Verification against RubyLLM's recorded VCR cassettes (HTTP and WebSocket), with JSON-equal
  request bodies; benchmarks against RubyLLM + YJIT (`bench/`, `docs/BENCHMARK.md`).

### Fixed
- Long conversations no longer grow quadratically in memory and time. Each response kept a deep
  copy of the request it answered, so the whole history was copied again at every turn, and every
  request cloned all of them. Found by the benchmark: at 400 asks in one chat, 16.4 s / 392 MiB
  before, 0.5 s / 40 MiB after.

### Not included
- Providers Bedrock, Vertex AI, Azure, Cohere, ElevenLabs, and Deepgram (and so Vertex's
  `research` jobs).
