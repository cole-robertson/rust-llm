# Persistence with Loco (`rust_llm_loco`)

`rust_llm_loco` ports RubyLLM's `acts_as_chat` / `acts_as_message` / `acts_as_tool_call` and its
usage ledger to SeaORM, Loco's default ORM. Every message, tool call, and billed attempt is written
as it happens, so a chat can be reloaded mid-round (for example, parked on a tool approval) and
continued in another request or job.

The fastest setup is `rust-llm generate install` (see [Generators](generators.md)). This page
describes what it wires up.

In short:

- `ChatRecord` persists chats, messages, tool calls, attachments, and billed attempts, and resumes
  a chat parked on a tool approval or an MCP input request.
- `ModelStore` keeps the model registry in `rust_llm_models`.
- `BatchStore` keeps provider batches in `rust_llm_batches` (see [Batches](batches.md#persisted-batches)).
- `McpCredentialStore` keeps MCP OAuth credentials, encrypted, in `rust_llm_mcp_credentials` (see [MCP](mcp.md#oauth)).

## Tables

The same tables and columns RubyLLM creates, with the `rust_llm_` prefix:

| Table | Holds |
|---|---|
| `chats` | one row per conversation, pointing at its model row |
| `messages` | role, content, thinking text and signature, citations, provider tool calls, raw content, finish reason |
| `rust_llm_models` | the model registry; filled from the registry when a chat first needs a model |
| `rust_llm_tool_calls` | each tool call, its arguments, approval decision, and the tool-result message it links to |
| `rust_llm_usages` | one row per provider attempt: operation, provider, model, status, token buckets, and cost components |
| `rust_llm_attachments` | message files: bytes, filename, content type, byte size, and `{ "resolution": ... }` metadata |
| `rust_llm_batches` | submitted provider batches: provider id, status, protocol, and the chat ids in submission order |
| `rust_llm_mcp_credentials` | MCP OAuth credentials per owner and server, encrypted |

RubyLLM stores message files with Active Storage (`has_many_attached :attachments`). Loco has no
Active Storage, so `rust_llm_attachments` holds the bytes in the database, one row per file,
pointing at its message the same polymorphic way `rust_llm_tool_calls` does. `ask_with` stores the
files with the user message, and `to_llm` gives them back as `Attachment`s with the same bytes,
name, and type. Keep large files in object storage and attach them by URL if the database should
stay small: a URL attachment is fetched once when it is stored.

Add the migrations to your Loco migrator:

```rust,no_run
use sea_orm_migration::prelude::*;

pub struct Migrator;

#[async_trait::async_trait]
impl MigratorTrait for Migrator {
    fn migrations() -> Vec<Box<dyn MigrationTrait>> {
        let mut migrations: Vec<Box<dyn MigrationTrait>> = vec![/* your app's migrations */];
        migrations.extend(rust_llm_loco::migrations());
        migrations
    }
}
```

The SeaORM entities are in `rust_llm_loco::entities` (`chats`, `messages`, `rust_llm_models`,
`rust_llm_tool_calls`, `rust_llm_usages`, `rust_llm_attachments`, `rust_llm_batches`,
`rust_llm_mcp_credentials`).

## Working with Persisted Chats

RubyLLM's record *is* the chat. In Rust the row (`ChatRecord`) and the in-memory `rust_llm::Chat`
are separate: `to_llm` rebuilds the `Chat` from rows, and the record's `ask`/`complete` drive it
while writing each step.

```ruby
chat = Chat.create!(model: 'claude-haiku-4-5')
chat.with_tools(Weather).ask("What's the weather in Berlin?")
chat.messages.count # => user, assistant tool call, tool result, assistant
```

```rust,no_run
# use rust_llm::{Tool, ToolCall, ToolError, ToolResult};
# struct Weather;
# #[async_trait::async_trait]
# impl Tool for Weather {
#     fn description(&self) -> String { String::new() }
#     async fn execute(&self, _: serde_json::Map<String, serde_json::Value>, _: &ToolCall) -> Result<ToolResult, ToolError> { Ok("".into()) }
# }
use rust_llm_loco::ChatRecord;
use sea_orm::DatabaseConnection;

# async fn run(db: &DatabaseConnection) -> rust_llm_loco::Result<()> {
let record = ChatRecord::create(db, "claude-haiku-4-5", None).await?;
let mut chat = record.to_llm(db).await?.with_tool(Weather);
let answer = record.ask(db, &mut chat, "What's the weather in Berlin?").await?;

let rows = record.messages(db).await?; // user, assistant (tool call), tool, assistant
# Ok(()) }
```

Runtime-only settings (tools, temperature, fallbacks, callbacks) live on the `Chat`, not in the
database. Reapply them after `to_llm`, or keep them in an [Agent](agents.md) and use
`Agent::apply`.

`ChatRecord`:

| Method | RubyLLM |
|---|---|
| `ChatRecord::create(db, model, provider)` | `Chat.create!(model:, provider:)` |
| `ChatRecord::find(db, id)` | `Chat.find(id)` |
| `to_llm(db)` / `to_llm_with(db, config)` | `chat.to_llm` (with `context:`) |
| `create_with(db, model, provider, true)`, `assume_model_exists` | `Chat.create!(..., assume_model_exists: true)` |
| `reload(db, &mut chat)` | `chat.reload` (keeps tools, callbacks, runtime instructions) |
| `with_model(db, chat, model, provider)` | `chat.with_model(model, provider:)` |
| `ask(db, &mut chat, text)` / `ask_with(..., attachments)` | `chat.ask(text)` / `chat.ask(text, with: [...])` |
| `ask_later(db, &mut chat, text)` / `ask_later_with` | `chat.ask_later(text)` (persisted) |
| `complete(db, &mut chat)` | `chat.complete` |
| `compact(db, &mut chat)` | `chat.compact` (persists the compaction message and its usage) |
| `run_tools(db, &mut chat)` | `chat.run_tools` |
| `add_message(db, &mut chat, message)` | `chat.add_message(message)` |
| `with_instructions(db, &mut chat, text)` | `chat.with_instructions(text)` |
| `set_instructions(db, &mut chat, text, append, persist, cache_until_here)` | `chat.with_instructions(text, append:, persist:, cache_until_here:)` |
| `cache_until_here(db, &mut chat)` | `chat.cache_until_here` |
| `approve` / `deny(db, &mut chat, tool_call_id)` | `chat.approve` / `chat.deny` (persisted) |
| `is_awaiting_approval(db, &mut chat)`, `pending_approvals(db, &mut chat)` | `chat.awaiting_approval?`, `chat.pending_approvals` (tool-call rows) |
| `answer` / `decline(db, &mut chat, request, ...)` | `chat.answer` / `chat.decline` (persisted) |
| `cancel(db)` / `cancel_chat(db, &chat)`, `is_cancelled(db)` | `chat.cancel`, `chat.cancelled?` |
| `create_for_agent(db, &agent)` / `find_for_agent(db, id, &agent)` | `Agent.create!` / `Agent.find(id)` with `chat_model Chat` |
| `persist_collected(db, &mut chat)` | the answers a batch appended to the chat |
| `destroy(db)` | `chat.destroy!` |
| `messages(db)`, `usages(db)`, `model(db)` | `chat.messages`, the usage rows, `chat.model` |
| `tokens(db)`, `cost(db)`, `total_cost(db)` | `chat.tokens`, `chat.cost`, `chat.cost.total` |

Instructions work as in RubyLLM. Persisted instructions replace the chat's system rows; a single
existing row is updated in place so it stays ahead of the conversation. `append: true` adds another.
With `persist: false` they apply only to chats this `ChatRecord` builds or reloads and are never
written. `create_for_agent` persists the agent's instructions; `find_for_agent` applies them
without rewriting history.

`complete` advances one `step` at a time and writes after every step, so a crash loses at most the
step in flight. If a round fails, the incomplete tool round is rolled back (usage rows stay, linked
only to the chat) so the next `ask` does not hit `Error::PendingToolCalls`, and the error is
returned.

## Approvals Across Requests

A chat parked on a `requires_approval` tool resumes from rows alone, in another request or worker:

```rust,no_run
# use rust_llm::{Tool, ToolCall, ToolError, ToolResult};
# struct DeleteEverything;
# #[async_trait::async_trait]
# impl Tool for DeleteEverything {
#     fn description(&self) -> String { String::new() }
#     fn requires_approval(&self) -> bool { true }
#     async fn execute(&self, _: serde_json::Map<String, serde_json::Value>, _: &ToolCall) -> Result<ToolResult, ToolError> { Ok("".into()) }
# }
use rust_llm_loco::ChatRecord;

# async fn run(db: &sea_orm::DatabaseConnection, id: i32, tool_call_id: &str) -> rust_llm_loco::Result<()> {
let record = ChatRecord::find(db, id).await?;
let mut chat = record.to_llm(db).await?.with_tool(DeleteEverything);

for call in chat.pending_approvals() {
    println!("{} wants to run with {:?}", call.name, call.arguments());
}

record.approve(db, &mut chat, tool_call_id).await?; // writes `approved` on the tool-call row
record.complete(db, &mut chat).await?;
# Ok(()) }
```

`complete` rereads the decisions and paused MCP input requests on the tool-call rows before each
step (RubyLLM's `approval_checker` and `input_checker`), so an approval or answer recorded by
another process reaches a chat that is already running. A tool call paused on an MCP input request
stores its state in `pending_input` and resumes from rows after `answer` or `decline`.

Make approval-gated tools safe to run twice: a worker can die after the side effect but before the
result row is written.

## Usage and Cost

Each provider attempt, including retries and failed attempts, gets a `rust_llm_usages` row linked
to the message it produced (or only to the chat when there is none). Costs are stored as numeric
columns when the attempt finishes and are never re-priced:

```rust,no_run
# async fn run(db: &sea_orm::DatabaseConnection, record: rust_llm_loco::ChatRecord) -> rust_llm_loco::Result<()> {
let tokens = record.tokens(db).await?;
let total = record.total_cost(db).await?; // None when any attempt could not be priced
# Ok(()) }
```

## Streaming

`record.ask_stream` and `record.complete_stream` are RubyLLM's `chat.ask(msg) { |chunk| }` and
`chat.complete { |chunk| }` on a persisted chat. They persist exactly as `ask` and `complete` do,
and report each step with the row it belongs to:

- `StreamEvent::NewMessage(row)`: the assistant row exists, empty, before its first chunk
  (`persist_new_message`), or a tool result was written.
- `StreamEvent::Chunk { message_id, chunk }`: text for that row.
- `StreamEvent::EndMessage(row)`: the row is final, with its tool calls and usage
  (`persist_message_completion`).

```rust,no_run
use rust_llm_loco::{ChatRecord, StreamEvent};
use sea_orm::DatabaseConnection;

# async fn run(db: &DatabaseConnection, record: ChatRecord) -> rust_llm_loco::Result<()> {
let mut chat = record.to_llm(db).await?;
record
    .ask_stream(db, &mut chat, "Tell me a story", |event| match event {
        StreamEvent::NewMessage(row) => println!("message {} started", row.id),
        StreamEvent::Chunk { chunk, .. } => print!("{}", chunk.content()),
        StreamEvent::EndMessage(row) => println!("\nmessage {} saved", row.id),
    })
    .await?;
# Ok(()) }
```

Chunks are not written while they stream, so a failed or cancelled reply leaves no row behind: the
empty row is removed, as RubyLLM's `cleanup_after_failure` does. Broadcast the events however your
app pushes updates; the generated chat UI sends them over a live channel (see
[Generators](generators.md#chat_ui)).

## Cancellation

`record.cancel(db)` sets `chats.cancelled`. A `complete` or `compact` running anywhere, in a job for
example, polls the column every second (`CANCELLATION_POLL_INTERVAL`), clears it, and stops with
`rust_llm::Error::Cancelled`. `cancel_chat(db, &chat)` also cancels a chat you hold, so it stops
without waiting for a poll. The billed attempt stays in the ledger, unlinked, and the chat stays
usable.

## The Model Registry in the Database

```ruby
RubyLLM.config.model_registry_store = RubyLLM::ActiveRecord::Model # set by the Railtie
RubyLLM.models.refresh
```

```rust,no_run
use std::sync::Arc;
use rust_llm_loco::ModelStore;

# async fn run(db: sea_orm::DatabaseConnection) -> rust_llm::Result<()> {
rust_llm::configure(|c| c.model_registry_store = Some(Arc::new(ModelStore::new(db.clone()))));
rust_llm::models::refresh(false).await?; // saves into rust_llm_models
# Ok(()) }
```

With the store set, the registry loads from `rust_llm_models` when it has rows. `refresh` writes the
new registry in one transaction; a model that dropped out of the catalog but is still referenced by
a chat is kept and stamped `unlisted_at`, and `model_store::listed()` / `unlisted()` query each
set. `ModelStore` bridges SeaORM's async API to the registry's sync one, so it needs a
multi-threaded tokio runtime.

## Errors

`rust_llm_loco::Error` wraps `Db(sea_orm::DbErr)`, `Llm(rust_llm::Error)`, and `NotFound`.

## Differences from RubyLLM

- The record and the chat are separate values: `ChatRecord` methods take `&mut Chat`, and chat
  methods the record does not persist (`generate`, `step`, `count_tokens`, callbacks) are called on
  the `Chat` directly.
- A streaming block on a record is `ask_stream`/`complete_stream` with a closure that receives
  `StreamEvent`s; Turbo broadcasting is up to the app (the generated chat UI uses a live channel).
- Custom or namespaced chat/message classes (`acts_as_chat messages:`, `message_class:`) and Action
  Text content are Rails mechanisms. The tables keep RubyLLM's names.
- Attachments live in `rust_llm_attachments` instead of Active Storage.
