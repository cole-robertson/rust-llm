# Persistence with Loco (`rust_llm_loco`)

`rust_llm_loco` ports RubyLLM's `acts_as_chat` / `acts_as_message` / `acts_as_tool_call` and its
usage ledger to SeaORM, Loco's default ORM. Every message, tool call, and billed attempt is written
as it happens, so a chat can be reloaded mid-round (for example, parked on a tool approval) and
continued in another request or job.

The fastest setup is `rust-llm generate install` (see [Generators](generators.md)). This page
describes what it wires up.

## Tables

The same tables and columns RubyLLM creates, with the `rust_llm_` prefix:

| Table | Holds |
|---|---|
| `chats` | one row per conversation, pointing at its model row |
| `messages` | role, content, thinking text and signature, citations, provider tool calls, raw content, finish reason |
| `rust_llm_models` | registry entries used by chats, created on first use |
| `rust_llm_tool_calls` | each tool call, its arguments, approval decision, and the tool-result message it links to |
| `rust_llm_usages` | one row per provider attempt: operation, provider, model, status, token buckets, and cost components |
| `rust_llm_attachments` | message files: bytes, filename, content type, byte size, and `{ "resolution": ... }` metadata |

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
`rust_llm_tool_calls`, `rust_llm_usages`).

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
| `add_message(db, &mut chat, message)` | `chat.add_message(message)` |
| `with_instructions(db, &mut chat, text)` | `chat.with_instructions(text)` |
| `set_instructions(db, &mut chat, text, append, persist, cache_until_here)` | `chat.with_instructions(text, append:, persist:, cache_until_here:)` |
| `cache_until_here(db, &mut chat)` | `chat.cache_until_here` |
| `approve` / `deny(db, &mut chat, tool_call_id)` | `chat.approve` / `chat.deny` (persisted) |
| `is_awaiting_approval(db, &mut chat)`, `pending_approvals(db, &mut chat)` | `chat.awaiting_approval?`, `chat.pending_approvals` (tool-call rows) |
| `answer` / `decline(db, &mut chat, request, ...)` | `chat.answer` / `chat.decline` (persisted) |
| `cancel(db)`, `is_cancelled(db)` | `chat.cancel`, `chat.cancelled?` |
| `create_for_agent(db, &agent)` / `find_for_agent(db, id, &agent)` | `Agent.create!` / `Agent.find(id)` with `chat_model Chat` |
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

## Cancellation

`record.cancel(db)` sets `chats.cancelled`. A `complete` running anywhere, in a job for example,
polls the column every second (`CANCELLATION_POLL_INTERVAL`), clears it, and stops with
`rust_llm::Error::Cancelled`. The billed attempt stays in the ledger, unlinked, and the chat stays
usable.

## Errors

`rust_llm_loco::Error` wraps `Db(sea_orm::DbErr)`, `Llm(rust_llm::Error)`, and `NotFound`.

## Not ported

- Streaming through a persisted chat (`chat.ask { |chunk| }` on a record) and Turbo broadcasting.
- Custom or namespaced chat/message classes, a separate user-visible transcript, Action Text
  content, the `ruby_llm_batches` and `ruby_llm_mcp_credentials` tables, and `compact`/`generate`
  on a record.
- Copying the whole registry into an empty `rust_llm_models` table on first use: RustLLM's
  registry never reads that table, so only the rows chats use are written.
- Registry refresh into `rust_llm_models` (`RubyLLM.models.refresh`, `ruby_llm:load_models`).
