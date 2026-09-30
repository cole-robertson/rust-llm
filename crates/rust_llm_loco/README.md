# rust_llm_loco

RubyLLM's `acts_as_chat` / `acts_as_message` / `acts_as_tool_call` for
[Loco](https://loco.rs) and SeaORM: persist chats, messages, tool calls, attachments, and
per-attempt usage in the same tables and columns RubyLLM's Rails integration uses, and resume a
chat (for example one parked on a tool approval or an MCP input request) from the database alone.
It also stores the model registry, provider batches, and encrypted MCP OAuth credentials.

- Guide: <https://github.com/cole-robertson/rust-llm/blob/main/docs/persistence-loco.md>
- API docs: <https://docs.rs/rust_llm_loco>
- Built on [`rust_llm`](https://crates.io/crates/rust_llm)

MIT licensed. RustLLM is a port of Carmine Paolino's RubyLLM; see LICENSE.
