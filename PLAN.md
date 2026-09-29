# RustLLM — a 1:1 Rust port of RubyLLM 2.0

Upstream: crmne/ruby_llm @ 1e91b30 (2.0.0), cloned to `upstream/` (MIT, see UPSTREAM_LICENSE).
Builds and tests run on the `framework` tailnet box via `bin/fw <cmd>`.

## Mapping (Ruby → Rust)

| RubyLLM | rust_llm (crate) |
|---|---|
| `RubyLLM.configure { }`, `Configuration` | `rust_llm::configure(|c| ..)`, `Config` (env fallbacks `OPENAI_API_KEY`, ...) |
| `RubyLLM.chat(model:, provider:)` | `rust_llm::chat()` / `Chat::new(model, provider)` |
| `chat.ask / ask_later / generate / step / complete / run_tools` | same names, `async` |
| `chat.ask(..) { |chunk| }` | `chat.ask_stream(msg, |chunk| ..)` |
| `with_instructions / with_tools / with_temperature / with_schema / with_thinking / with_params(provider_options) / with_headers / with_model / with_fallbacks / with_max_output_tokens` | same builder names |
| `before_message / after_message / before_tool_call / after_tool_result / before_fallback / after_fallback` | same, closures |
| `approve / deny / awaiting_approval? / pending_approvals / cancel` | same |
| `Message`, `Chunk`, `ToolCall`, `Tokens`, `Cost`, `Thinking`, `Citation`, `Attachment` | same types |
| `RubyLLM::Tool` (`description`, `param`, `execute`) | `Tool` trait + `#[derive(JsonSchema)]` args, `tool_fn` helper |
| `Tool.requires_approval` | `Tool::requires_approval()` |
| `RubyLLM::Agent` class DSL | `Agent` trait (model/instructions/tools/...) |
| `RubyLLM.embed` | `rust_llm::embed()` |
| `RubyLLM.models` registry (models.json, aliases.json) | `Models` (bundled JSON, same lookup + provider preference) |
| `Provider` / `Protocol` split | `Provider` struct (slug, api_base, headers, protocol routing) + `Protocol` trait |
| Protocols: chat_completions, anthropic, responses, gemini | same four modules |
| Error hierarchy + ErrorMiddleware status mapping | `Error` enum, same status/pattern rules |
| Faraday retry (429/5xx/529, backoff, no retry after stream delivered) | `transport` with the same rules |
| VCR cassettes | tests replay upstream cassettes byte-for-byte |
| `acts_as_chat / acts_as_message / acts_as_tool_call`, `rust_llm_models`, `rust_llm_usages` | `rust_llm_loco` crate: SeaORM entities + migrations with the same tables/columns, `ChatRecord::ask` persists messages, tool calls, usages |

## Scope for this build

In: chat loop, tools, approvals, streaming, structured output, thinking, fallbacks,
callbacks, cancellation, usage/cost ledger, model registry + aliases, embeddings,
providers openai, anthropic, gemini, deepseek, mistral, openrouter, xai, perplexity,
ollama, ollama_cloud, hetzner, gpustack (4 wire protocols), SeaORM persistence.

Out (later, listed so nothing is silently dropped): bedrock/vertexai/azure auth, cohere,
elevenlabs/deepgram, typesafe/judge, paint/animate/speak/transcribe/ocr/rerank/moderate,
batches, MCP, compaction, provider file uploads, prompt caching controls, generators.

## Verification

1. Unit tests for Tokens/Cost/Thinking/aliases/error mapping ported from specs.
2. Cassette replay: each ported protocol replays upstream VCR cassettes, asserting the
   rendered request body equals the recorded one (JSON-equal) and the parsed Message
   matches the spec's expectations.
3. `rust_llm_loco`: SQLite integration test that runs a tool-calling chat and checks rows.
4. Live smoke against Anthropic via the local proxy if keys are present.
