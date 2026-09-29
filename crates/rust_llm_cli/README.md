# rust_llm_cli

RubyLLM's Rails generators as a `rust-llm` CLI for Loco + Inertia + React apps.

```text
cargo install rust_llm_cli
rust-llm generate install        # deps, migration, initializer, Chat/Message models
rust-llm generate chat_ui        # Inertia + React chat pages, controllers, a Loco worker
rust-llm generate tool Weather   # src/tools/weather_tool.rs + React tool call/result components
rust-llm generate agent Support  # src/agents/support.rs + src/prompts/support/instructions.txt
rust-llm generate schema Product
rust-llm generate provider NAME
rust-llm generate upgrade
```

- Guide: <https://github.com/cole-robertson/rust-llm/blob/main/docs/generators.md>

MIT licensed. RustLLM is a port of Carmine Paolino's RubyLLM; see LICENSE.
