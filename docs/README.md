# RustLLM Guides

User guides for RustLLM, a 1:1 Rust port of [RubyLLM](https://github.com/crmne/ruby_llm) 2.0. They
follow the structure of RubyLLM's own guides, show the Ruby original next to the Rust, and cover
only what the port implements. Each guide ends with a "Not ported" list.

Every Rust sample compiles: `docs/check` includes each guide as a doctest
(`cargo test -p rust_llm_docs --doc`). Samples are `no_run`, so nothing calls a provider.

## Getting Started

- [Getting Started](getting-started.md): install, configure, and try each feature once.
- [Configuration](configuration.md): API keys, default models, timeouts and retries, isolated
  configurations.
- [Migrating from RubyLLM](migrating-from-rubyllm.md): naming conventions and a Ruby-to-Rust cheat
  sheet for the whole API.
- [Benchmark](BENCHMARK.md): RustLLM vs RubyLLM 2.0 against a mock provider, including the cases
  where Ruby is close or wins.

## Core Features

- [Chat](chat.md): conversations, instructions, models, request options, callbacks, finish
  reasons.
- [Tools](tools.md): defining tools, parameters, tool choice, approvals, progress, provider tools.
- [Streaming](streaming.md): chunks, streaming with tools, cancellation.
- [Structured Output](structured-output.md): `schemars` types and JSON schemas.
- [Extended Thinking](thinking.md): effort, budget, display, reading thinking.
- [Attachments and Files](attachments-and-files.md): images, PDFs, audio, video, and the Files API.
- [Embeddings](embeddings.md): vectors for one or many texts.
- [Images](images.md): generating and editing images with `paint`.
- [MCP](mcp.md): Model Context Protocol servers, their tools, resources, and prompts.
- [Judgments](judgments.md): probabilities, choices, and scores from Jev.
- [Cost and Usage](cost-and-usage.md): tokens, costs, and the per-attempt usage ledger.

## Advanced

- [Agents](agents.md): reusable chat configurations.
- [Batches](batches.md): provider batch APIs for chats and embeddings.
- [Errors and Retries](errors-and-retries.md): error variants, fallbacks, automatic retries.
- [Instrumentation and Workflows](instrumentation.md): `*.rust_llm` events and workflow context.
- [Prompt Templates](prompts.md): prompts on disk, partials, and the ERB to Jinja mapping.

## Loco

- [Persistence with Loco](persistence-loco.md): `rust_llm_loco`, RubyLLM's `acts_as_chat` for
  SeaORM.
- [Generators](generators.md): the `rust-llm` CLI (install, chat_ui, tool, agent, schema, provider,
  upgrade).
