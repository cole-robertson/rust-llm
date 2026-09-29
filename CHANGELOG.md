# Changelog

RustLLM's version follows the RubyLLM release it ports: 2.0.x ports RubyLLM 2.0 (upstream
`crmne/ruby_llm` at `1e91b30`). Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [2.0.0] - unreleased

First release: `rust_llm`, `rust_llm_loco`, and `rust_llm_cli`.

### Added
- `rust_llm`: chat (`ask`, `ask_later`, `complete`, `step`, `generate`, `run_tools`), streaming,
  tools with approvals and cancellation, agents, structured output (`with_schema`,
  `with_schema_for::<T>()`), thinking, fallbacks, callbacks, the per-attempt usage and cost ledger,
  the bundled model registry and aliases, embeddings, images (`paint`), batches, provider file
  uploads and downloads, MCP (stdio and streamable HTTP), provider tools, and judgments (`Judge`,
  `judge`, TypeSafe/Jev).
- Providers: OpenAI (Responses and Chat Completions), Anthropic, Gemini, DeepSeek, Mistral,
  OpenRouter, xAI, Perplexity, Ollama, Ollama Cloud, GPUStack, Hetzner, TypeSafe.
- `rust_llm_loco`: `acts_as_chat` for Loco/SeaORM, with the same tables and columns as
  RubyLLM's Rails integration, plus migrations.
- `rust_llm_cli`: the `rust-llm generate` CLI (`install`, `chat_ui`, `tool`, `agent`, `schema`,
  `provider`, `upgrade`) for Loco + Inertia + React apps.
- Verification: RubyLLM's recorded VCR cassettes, replayed with JSON-equal request bodies.
- Benchmarks against RubyLLM 2.0 (`bench/`, `docs/BENCHMARK.md`).

### Fixed
- Long conversations no longer grow quadratically in memory and time. Each response kept a deep
  copy of the request it answered, so the whole history was copied again at every turn, and every
  request cloned all of them. Found by the benchmark: at 400 asks in one chat, 16.4 s / 392 MiB
  before, 0.5 s / 40 MiB after.

### Not ported yet
- Bedrock, Vertex AI, Azure; Cohere; ElevenLabs; Deepgram.
- `animate`, `speak`, `transcribe`, `ocr`, `rerank`, `moderate`.
- MCP OAuth; Gemini embedding batches; multipart image edits for non-gpt-image models.
- `with_compaction`, `count_tokens`, `with_citations` as a request option; instrumentation events.
