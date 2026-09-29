# Tools

A tool lets the model call your Rust code to look up a record, fetch data, or carry out an action.
You write `execute`; RustLLM passes the model's arguments in and returns the result to the model.

## Creating a Tool

A RubyLLM `Tool` subclass becomes a type that implements `rust_llm::Tool`. `execute` is async (the
trait uses `async_trait`) and receives the arguments as a JSON object plus the `ToolCall` that
triggered it.

```ruby
class Weather < RubyLLM::Tool
  description "Gets current weather for a location"
  parameter :latitude, description: "Latitude (e.g., 52.5200)"
  parameter :longitude, description: "Longitude (e.g., 13.4050)"

  def execute(latitude:, longitude:)
    { latitude:, longitude:, temperature_2m: 14.2 }
  rescue Faraday::ConnectionFailed
    { error: "The weather service is unavailable. Try again later." }
  end
end
```

```rust,no_run
use rust_llm::{Parameter, Tool, ToolCall, ToolError, ToolResult};
use serde_json::{Map, Value, json};

struct Weather;

#[async_trait::async_trait]
impl Tool for Weather {
    fn description(&self) -> String {
        "Gets current weather for a location".into()
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![
            Parameter::new("latitude").description("Latitude (e.g., 52.5200)"),
            Parameter::new("longitude").description("Longitude (e.g., 13.4050)"),
        ]
    }

    async fn execute(&self, args: Map<String, Value>, _call: &ToolCall) -> Result<ToolResult, ToolError> {
        if args["latitude"].as_str().is_none_or(str::is_empty) {
            // A recoverable problem: tell the model, the conversation continues.
            return Ok(ToolResult::error("latitude is required"));
        }
        Ok(json!({ "latitude": args["latitude"], "longitude": args["longitude"], "temperature_2m": 14.2 }).into())
    }
}
```

The tool's name comes from the type name, underscored, with a trailing `Tool` dropped:
`WeatherLookupTool` is called `weather_lookup`. Override `fn name(&self) -> String` to choose
another.

What `execute` returns:

- `Ok(ToolResult)`: built from a `String`, `&str`, or `serde_json::Value` (objects and arrays go to
  the model as JSON) with `.into()`.
- `Ok(ToolResult::error(msg))`: `{"error": msg}`, the Rust spelling of returning
  `{ error: "..." }`. Use it for problems the model can recover from.
- `Err(e)`: stops the conversation. `ask` fails with `Error::Tool(message)`, like an exception
  escaping `execute`.

When the model calls a tool that does not exist, or passes a missing or unknown argument to a tool
declared with `parameters()`, the model gets an error result and the conversation continues.

## Declaring Parameters

`Parameter::new(name)` is a required string. Chain `.description(..)`, `.kind("integer")`
(`string`, `integer`, `number`, `boolean`, `array`, `object`), and `.optional()`:

```ruby
parameter :units, type: :string, description: "metric or imperial", required: false
```

```rust,no_run
# use rust_llm::Parameter;
let units = Parameter::new("units").kind("string").description("metric or imperial").optional();
```

A tool without parameters gets the empty object schema, like a Ruby `def execute` with no keywords.

For nested objects, arrays, and enums, return a full JSON Schema from `parameters_schema`. The
easiest way is to derive it with `schemars` through `rust_llm::schema_for`, which closes every
object with `additionalProperties: false` the way RubyLLM's schema DSL does:

```ruby
class Scheduler < RubyLLM::Tool
  description "Books a meeting"
  parameters do
    object :window do
      string :start, description: "ISO8601 start time"
      string :finish, description: "ISO8601 end time"
    end
    array :participants, of: :string
  end
end
```

```rust,no_run
use rust_llm::{Tool, ToolCall, ToolError, ToolResult};
use serde_json::{Map, Value};

#[derive(schemars::JsonSchema, serde::Deserialize)]
struct Window {
    /// ISO8601 start time
    start: String,
    /// ISO8601 end time
    finish: String,
}

#[derive(schemars::JsonSchema, serde::Deserialize)]
struct BookMeeting {
    window: Window,
    participants: Vec<String>,
}

struct Scheduler;

#[async_trait::async_trait]
impl Tool for Scheduler {
    fn description(&self) -> String {
        "Books a meeting".into()
    }

    fn parameters_schema(&self) -> Option<Value> {
        Some(rust_llm::schema_for::<BookMeeting>())
    }

    async fn execute(&self, args: Map<String, Value>, _call: &ToolCall) -> Result<ToolResult, ToolError> {
        let request: BookMeeting = serde_json::from_value(Value::Object(args))?;
        Ok(format!("Booked {} people from {}", request.participants.len(), request.window.start).into())
    }
}
```

`parameters_schema` can also return a hand-written `json!({...})` schema. It wins over
`parameters()`.

## Using Tools in Chat

```ruby
chat = RubyLLM.chat.with_tools(Weather)
chat.ask "What's the weather in Berlin? Latitude 52.52, longitude 13.40."
chat.with_tools(nil)
```

```rust,no_run
# use rust_llm::{Tool, ToolCall, ToolError, ToolResult};
# struct Weather;
# #[async_trait::async_trait]
# impl Tool for Weather {
#     fn description(&self) -> String { String::new() }
#     async fn execute(&self, _: serde_json::Map<String, serde_json::Value>, _: &ToolCall) -> Result<ToolResult, ToolError> { Ok("".into()) }
# }
# async fn run() -> rust_llm::Result<()> {
use std::sync::Arc;
use rust_llm::SharedTool;

let mut chat = rust_llm::chat()?.with_tool(Weather);
chat.ask("What's the weather in Berlin? Latitude 52.52, longitude 13.40.").await?;

// Several at once, e.g. from a list built at runtime.
let tools: Vec<SharedTool> = vec![Arc::new(Weather)];
let mut chat = rust_llm::chat()?.with_tools(tools);

chat.clear_tools(); // with_tools(nil)
# Ok(()) }
```

Adding a tool with the name of one already on the chat replaces it. A tool needing application
state (a database handle, the current user) is just a struct with fields.

For a one-off tool, `FnTool` builds one from a closure:

```rust,no_run
use rust_llm::{FnTool, Parameter, ToolResult};

let lookup = FnTool::new("lookup_order", "Finds an order by id", |args| async move {
    let id = args.get("id").and_then(|v| v.as_str()).unwrap_or_default().to_string();
    Ok::<ToolResult, rust_llm::ToolError>(format!("Order {id}: shipped").into())
})
.parameter(Parameter::new("id").description("Order id"));
```

## Tool Choice and Calls per Response

```ruby
chat.with_tool_options(choice: :required, calls: :one)
chat.with_tool_options(choice: :weather)
```

```rust,no_run
use rust_llm::{ToolCalls, ToolChoice};

# fn run(chat: rust_llm::Chat) -> rust_llm::Result<()> {
let chat = chat
    .with_tool_choice(ToolChoice::Required)? // Auto, None, Required, or Tool("weather")
    .with_tool_calls(ToolCalls::One);        // One or Many
# Ok(()) }
```

`ToolChoice::Tool(name)` must name a tool on the chat, or `with_tool_choice` fails with
`Error::InvalidToolChoice`. After a required or specific tool runs, the choice is cleared so the
model can answer.

## Concurrent Tool Calls

```ruby
chat.with_tool_options(concurrency: true)
RubyLLM.configure { |config| config.tool_concurrency = true }
```

```rust,no_run
# fn run(chat: rust_llm::Chat) -> rust_llm::Result<()> {
let chat = chat.with_tool_concurrency(true); // or `config.tool_concurrency = true`
# Ok(()) }
```

When a response asks for several tools, they run at the same time and each result is added as its
call finishes. The calls share the chat's task, like RubyLLM's `:fibers` mode, so move blocking
work in a tool to `tokio::task::spawn_blocking`.

## Requiring Approval

```ruby
class IssueRefund < RubyLLM::Tool
  description "Issues a refund for an order"
  requires_approval
  def execute(order_id:) = Refunds.issue!(order_id)
end

response = chat.ask "Refund order 42"
chat.awaiting_approval? # => true
chat.approve(chat.pending_approvals.first)
chat.complete
```

```rust,no_run
use rust_llm::{Parameter, Tool, ToolCall, ToolError, ToolResult};
use serde_json::{Map, Value};

struct IssueRefund;

#[async_trait::async_trait]
impl Tool for IssueRefund {
    fn description(&self) -> String {
        "Issues a refund for an order".into()
    }
    fn parameters(&self) -> Vec<Parameter> {
        vec![Parameter::new("order_id")]
    }
    fn requires_approval(&self) -> bool {
        true
    }
    async fn execute(&self, args: Map<String, Value>, _call: &ToolCall) -> Result<ToolResult, ToolError> {
        Ok(format!("Refunded {}", args["order_id"]).into())
    }
}

# async fn run() -> rust_llm::Result<()> {
let mut chat = rust_llm::chat()?.with_tool(IssueRefund);
chat.ask("Refund order 42").await?; // returns without running the tool

if chat.is_awaiting_approval() {
    let call = chat.pending_approvals().remove(0);
    chat.approve(&call.id); // or chat.deny(&call.id)
    chat.complete().await?;
}
# Ok(()) }
```

A denied call never runs; the model receives a structured denial. Calls that need no approval
still run while protected ones wait. Asking a new question while calls are pending fails with
`Error::PendingToolCalls`. `FnTool::requires_approval()` does the same for closure tools. To
persist decisions across requests, see [Persistence with Loco](persistence-loco.md).

## Callbacks and Progress

```ruby
chat.before_tool_call { |call| puts "Calling #{call.name} with #{call.arguments}" }
    .after_tool_result { |result| puts "Tool returned: #{result}" }
    .after_tool_progress { |call, progress| puts "#{call.name}: #{progress.message}" }
```

```rust,no_run
# fn run(chat: rust_llm::Chat) {
let chat = chat
    .before_tool_call(|call| println!("Calling {} with {:?}", call.name, call.arguments()))
    .after_tool_result(|result| println!("Tool returned: {}", result.content))
    .after_tool_progress(|call, progress| println!("{}: {:?}", call.name, progress.message));
# }
```

A tool reports progress from inside `execute` (RubyLLM's `progress "..."`):

```rust,no_run
use rust_llm::Progress;

rust_llm::progress::report(Progress { value: Some(1.0), total: Some(3.0), message: Some("Reading page 1 of 3".into()) });
```

## Returning Attachments

```ruby
return "Found: #{doc.name}", [RubyLLM::Attachment.new(doc.download_path)]
```

```rust,no_run
# use rust_llm::{Attachment, ToolResult};
let result = ToolResult::with_attachments("Found: report.pdf", vec![Attachment::new("/tmp/report.pdf")]);
```

## Provider-Specific Tool Options

```ruby
provider_options cache_control: { type: "ephemeral" }
```

```rust,no_run
use rust_llm::{Tool, ToolCall, ToolError, ToolResult};
use serde_json::{Map, Value, json};

struct TodoTool;

#[async_trait::async_trait]
impl Tool for TodoTool {
    fn description(&self) -> String {
        "Adds a task to the shared TODO list".into()
    }
    fn provider_options(&self) -> Map<String, Value> {
        let mut options = Map::new();
        options.insert("cache_control".into(), json!({ "type": "ephemeral" }));
        options
    }
    async fn execute(&self, _args: Map<String, Value>, _call: &ToolCall) -> Result<ToolResult, ToolError> {
        Ok("Added".into())
    }
}
```

The options are deep-merged into the tool's definition in the request.

## Provider Tools

Tools the provider runs on its own servers (web search, code execution, a remote MCP server):

```ruby
chat.with_provider_tools(:web_search)
chat.with_provider_tools(mcp: { name: "docs", url: "https://learn.microsoft.com/api/mcp" })
```

```rust,no_run
use rust_llm::ProviderTool;
use serde_json::json;

# fn run(chat: rust_llm::Chat) {
let chat = chat.with_provider_tools([
    ProviderTool::alias("web_search"),
    ProviderTool::with_options("mcp", json!({ "name": "docs", "url": "https://learn.microsoft.com/api/mcp" })),
    ProviderTool::raw(json!({ "type": "web_search_20260318", "name": "web_search" })),
]);
# }
```

An alias the protocol does not define fails at request time with `Error::UnsupportedServerTool`.

## MCP

`with_mcp` gives the model an MCP server's tools. See [MCP](mcp.md).

## Not ported

- Signature inference: Rust cannot read `execute`'s parameters, so declare them with
  `parameters()` or `parameters_schema()`.
- `requires_approval { |tool_call| ... }` resolver blocks: `requires_approval` is a `bool`.
- `concurrency: :threads` vs `:fibers`: concurrency is a `bool`; calls run as futures on the chat's task.
- `RubyLLM::SearchResults` (citable tool results).
- `RUBYLLM_DEBUG`: enable `tracing` at `debug` level instead.
