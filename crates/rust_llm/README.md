# rust_llm

One API for chat, tools, agents, structured output, streaming, embeddings, images, audio, video,
moderation, batches, files, MCP, and judgments across OpenAI, Anthropic, Gemini, DeepSeek, Mistral,
OpenRouter, xAI, Perplexity, Ollama, Ollama Cloud, GPUStack, Hetzner, and TypeSafe.

```rust,no_run
# async fn run() -> rust_llm::Result<()> {
let mut chat = rust_llm::chat_with("claude-opus-5-5")?;

let answer = chat.ask("What's the best way to learn Rust?").await?;
println!("{}", answer.content());
# Ok(()) }
```

- Examples and overview: <https://github.com/cole-robertson/rust-llm>
- Guides: <https://github.com/cole-robertson/rust-llm/tree/main/docs>
- API docs: <https://docs.rs/rust_llm>
- Persistence for Loco/SeaORM: [`rust_llm_loco`](https://crates.io/crates/rust_llm_loco)
- Generators: [`rust_llm_cli`](https://crates.io/crates/rust_llm_cli)

MIT licensed. RustLLM is a port of Carmine Paolino's [RubyLLM](https://github.com/crmne/ruby_llm);
see LICENSE.
