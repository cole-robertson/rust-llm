# rust_llm_loco

Persistence for [`rust_llm`](https://crates.io/crates/rust_llm) in [Loco](https://loco.rs) apps,
on SeaORM. Every message, tool call, attachment, and billed request is written as it happens, so a
chat can be resumed from the database alone, for example in another request or a background job
after a tool approval. It also stores the model registry, provider batches, and encrypted MCP OAuth
credentials.

```rust,ignore
let record = ChatRecord::create(&ctx.db, "claude-opus-5-5", None).await?;
let mut chat = record.to_llm(&ctx.db).await?.with_tool(Weather);
record.ask(&ctx.db, &mut chat, "What's the weather in Berlin?").await?;
```

- Guide: <https://github.com/cole-robertson/rust-llm/blob/main/docs/persistence-loco.md>
- API docs: <https://docs.rs/rust_llm_loco>

MIT licensed. Ports the Rails integration of Carmine Paolino's
[RubyLLM](https://github.com/crmne/ruby_llm); see LICENSE.
