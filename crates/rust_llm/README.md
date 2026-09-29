# rust_llm

A 1:1 Rust port of [RubyLLM](https://github.com/crmne/ruby_llm) 2.0: one API for chat, tools,
agents, structured output, streaming, embeddings, images, batches, files, MCP, and judgments across
OpenAI, Anthropic, Gemini, DeepSeek, Mistral, OpenRouter, xAI, Perplexity, Ollama, GPUStack,
Hetzner, and TypeSafe. Request bodies match the ones RubyLLM recorded, byte for byte as JSON.

```rust,no_run
# async fn run() -> rust_llm::Result<()> {
// RubyLLM.chat.ask "What's the best way to learn Ruby?"
let answer = rust_llm::chat()?.ask("What's the best way to learn Rust?").await?;
println!("{}", answer.content());
# Ok(()) }
```

- Guides: <https://github.com/cole-robertson/rust-llm/tree/main/docs>
- API docs: <https://docs.rs/rust_llm>
- Persistence for Loco/SeaORM: [`rust_llm_loco`](https://crates.io/crates/rust_llm_loco)
- Generators: [`rust_llm_cli`](https://crates.io/crates/rust_llm_cli)

MIT licensed. RustLLM is a port of Carmine Paolino's RubyLLM; see LICENSE.
