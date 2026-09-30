# Agents

An agent defines a chat setup once (model, instructions, tools, options) and reuses it everywhere.
RubyLLM's `class X < RubyLLM::Agent` with class macros becomes a type implementing
`rust_llm::Agent`. Every trait method has a default, so an agent overrides only what it declares.

## Defining an Agent

```ruby
class WorkAssistant < RubyLLM::Agent
  model "gpt-5.6"
  instructions "You are a helpful assistant."
  tools SearchDocs
  temperature 0.2
  max_output_tokens 256
end

WorkAssistant.new.ask "How do I reset my API key?"
```

```rust,no_run
# use rust_llm::{Tool, ToolCall, ToolError, ToolResult};
# struct SearchDocs;
# #[async_trait::async_trait]
# impl Tool for SearchDocs {
#     fn description(&self) -> String { String::new() }
#     async fn execute(&self, _: serde_json::Map<String, serde_json::Value>, _: &ToolCall) -> Result<ToolResult, ToolError> { Ok("".into()) }
# }
use std::sync::Arc;
use rust_llm::{Agent, SharedTool};

struct WorkAssistant;

impl Agent for WorkAssistant {
    fn model(&self) -> Option<&str> {
        Some("gpt-5.6")
    }
    fn instructions(&self) -> Option<String> {
        Some("You are a helpful assistant.".into())
    }
    fn tools(&self) -> Vec<SharedTool> {
        vec![Arc::new(SearchDocs)]
    }
    fn temperature(&self) -> Option<f64> {
        Some(0.2)
    }
    fn max_output_tokens(&self) -> Option<i64> {
        Some(256)
    }
}

# async fn run() -> rust_llm::Result<()> {
let mut chat = WorkAssistant.chat()?; // a rust_llm::Chat with everything applied
chat.ask("How do I reset my API key?").await?;
# Ok(()) }
```

`chat()` returns a plain `Chat`, so everything in [Chat](chat.md) works on it.

The declarations map to the chat builders:

| RubyLLM macro | `Agent` method | Applied with |
|---|---|---|
| `model id, provider:` | `model`, `provider` | `Chat::new` |
| `model ..., protocol:` | `protocol` | `with_protocol` |
| `instructions` | `instructions` | `with_instructions` |
| `tools` | `tools` | `with_tools` |
| `tool_options choice:` | `tool_choice` | `with_tool_choice` |
| `temperature` | `temperature` | `with_temperature` |
| `max_output_tokens` | `max_output_tokens` | `with_max_output_tokens` |
| `thinking` | `thinking` | `with_thinking` |
| `schema` | `schema` (a JSON `Value`) | `with_schema` |
| `provider_options` | `provider_options` | `with_provider_options` |
| `fallbacks` | `fallbacks` | `with_fallbacks` |
| `mcp` | `mcp` | `with_mcp` |
| `provider_tools` | `provider_tools` | `with_provider_tools` |
| `model ..., assume_model_exists: true` | `assume_model_exists` | `Chat::with_config` |
| `context` | `context` (a `rust_llm::Context`) | `with_context` |
| `tool_options calls:`, `concurrency:` | `tool_calls`, `tool_concurrency` | `with_tool_calls`, `with_tool_concurrency` |
| `fallbacks ..., on:` | `fallback_errors` | `with_fallback_errors` |
| `citations` | `citations` | `with_citations` |
| `caching` | `caching` (what `with_caching` takes) | `with_caching` |
| `compaction` | `compaction` (what `with_compaction` takes) | `with_compaction` |
| `end_user` | `end_user` | `with_end_user` |
| `headers` | `headers` | `with_headers` |

## Runtime Values

Ruby agents read runtime `inputs` inside blocks. A Rust agent is a struct, so runtime values are
its fields and the trait methods read them:

```ruby
class WorkspaceAssistant < RubyLLM::Agent
  inputs :workspace
  instructions { "You are helping #{workspace.name}" }
end
WorkspaceAssistant.chat(workspace: current_workspace)
```

```rust,no_run
use rust_llm::Agent;

struct WorkspaceAssistant {
    workspace_name: String,
}

impl Agent for WorkspaceAssistant {
    fn instructions(&self) -> Option<String> {
        Some(format!("You are helping {}", self.workspace_name))
    }
}

# async fn run() -> rust_llm::Result<()> {
let agent = WorkspaceAssistant { workspace_name: "Acme".into() };
agent.chat()?.ask("What can you do?").await?;
# Ok(()) }
```

## Fallbacks and Structured Output

```ruby
class Critic < RubyLLM::Agent
  model "gpt-4.1"
  fallbacks "gpt-4.1-mini", "claude-haiku-4-5"
  schema VerdictSchema
end
```

```rust,no_run
use rust_llm::{Agent, Fallback};
use serde_json::Value;

#[derive(schemars::JsonSchema)]
struct Verdict {
    verdict: String,
    feedback: String,
}

struct Critic;

impl Agent for Critic {
    fn model(&self) -> Option<&str> {
        Some("gpt-4.1")
    }
    fn fallbacks(&self) -> Vec<Fallback> {
        vec!["gpt-4.1-mini".into(), "claude-haiku-4-5".into()]
    }
    fn schema(&self) -> Option<Value> {
        Some(serde_json::json!({ "name": "Verdict", "schema": rust_llm::schema_for::<Verdict>() }))
    }
}
```

## Prompts on Disk

RubyLLM loads `app/prompts/<agent>/instructions.txt.erb`. An agent that declares no
`instructions` does the same with `app/prompts/<agent>/instructions.txt.jinja`, rendered with its
`prompt_locals`; `name` (the type name by default) picks the directory, so `WorkAssistant` reads
`app/prompts/work_assistant/`. An empty file means no instructions. `render_prompt` renders the
agent's other prompts. See [Prompt Templates](prompts.md) for the template syntax.

`rust-llm generate agent Support` still writes an empty `src/prompts/support_agent/instructions.txt`
that the generated agent embeds with `include_str!`.

## Request Options

```ruby
class SupportAgent < RubyLLM::Agent
  model "claude-haiku-4-5"
  caching ttl: "1h"
  compaction at: 50_000
  citations
  headers "X-Team" => "support"
  context TenantContext
end
```

```rust,no_run
use rust_llm::{Agent, Context};
use serde_json::{Value, json};

struct SupportAgent {
    tenant: Context,
    account_id: String,
}

impl Agent for SupportAgent {
    fn model(&self) -> Option<&str> {
        Some("claude-haiku-4-5")
    }
    fn caching(&self) -> Option<Value> {
        Some(json!({ "ttl": "1h" }))
    }
    fn compaction(&self) -> Option<Value> {
        Some(json!({ "at": 50_000 }))
    }
    fn citations(&self) -> Option<bool> {
        Some(true)
    }
    fn end_user(&self) -> Option<String> {
        Some(self.account_id.clone())
    }
    fn headers(&self) -> Vec<(String, String)> {
        vec![("X-Team".into(), "support".into())]
    }
    fn context(&self) -> Option<Context> {
        Some(self.tenant.clone())
    }
}
```

`caching`, `compaction`, and `citations` accept `false` (`Some(json!(false))`, `Some(false)`) to
switch the feature off; `None` leaves the chat's setting alone.

## Applying an Agent to an Existing Chat

`apply` configures a chat you already have (RubyLLM's `Agent.new(chat:)`):

```rust,no_run
# use rust_llm::Agent;
# struct SupportAgent;
# impl Agent for SupportAgent {}
# async fn run(db: &sea_orm::DatabaseConnection, id: i32) -> rust_llm_loco::Result<()> {
use rust_llm_loco::ChatRecord;

let record = ChatRecord::find(db, id).await?;
let mut chat = SupportAgent.apply(record.to_llm(db).await?)?;
record.ask(db, &mut chat, "Any update on my ticket?").await?;
# Ok(()) }
```

## Persisted Agents

RubyLLM's Rails mode (`chat_model Chat`, `Agent.create!`, `Agent.find`) is on `ChatRecord`:

```rust,no_run
# use rust_llm::Agent;
# struct SupportAgent;
# impl Agent for SupportAgent {}
# async fn run(db: &sea_orm::DatabaseConnection, id: i32) -> rust_llm_loco::Result<()> {
use rust_llm_loco::ChatRecord;

let (record, mut chat) = ChatRecord::create_for_agent(db, &SupportAgent).await?; // Agent.create!
record.ask(db, &mut chat, "Hello").await?;

let (record, mut chat) = ChatRecord::find_for_agent(db, id, &SupportAgent).await?; // Agent.find(id)
record.ask(db, &mut chat, "Any update on my ticket?").await?;
# Ok(()) }
```

`create_for_agent` persists the agent's instructions; `find_for_agent` applies them without
rewriting history, and `sync_instructions` rewrites the persisted ones. See
[Persistence with Loco](persistence-loco.md).

## Differences from RubyLLM

- Class macros and their blocks and lambdas become trait methods, and `inputs` become struct
  fields, so there is no inheritance of declarations and nothing is evaluated lazily from inputs.
- `rescue_from` is a Ruby exception-class DSL; match on `rust_llm::Error` where you call the agent.
- Prompt templates are Jinja, not ERB (see [Prompt Templates](prompts.md)).
