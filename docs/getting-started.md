# Getting Started

Install RustLLM, configure a provider, and try chats, streaming, files, structured output, tools,
agents, images, embeddings, and cost tracking. Each section shows one feature and links to its
guide. The API follows [RubyLLM](https://github.com/crmne/ruby_llm) 2.0, so the Ruby original sits
next to each Rust sample.

## Installation

`rust_llm` is not on crates.io yet. Depend on the repository:

```toml
[dependencies]
rust_llm = { git = "https://github.com/cole-robertson/rust-llm" }
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

Every call that talks to a provider is `async` and returns `rust_llm::Result`.

## Minimal Configuration

Provider keys are read from the environment on first use (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
...), so `export OPENAI_API_KEY=...` is enough. To set them in code:

```ruby
RubyLLM.configure do |config|
  config.openai_api_key = ENV.fetch('OPENAI_API_KEY')
end
```

```rust,no_run
rust_llm::configure(|config| {
    config.openai_api_key(std::env::var("OPENAI_API_KEY").unwrap_or_default());
});
```

See [Configuration](configuration.md) for other providers and default models.

## Your First Chat

```ruby
chat = RubyLLM.chat
response = chat.ask "What is Ruby on Rails?"
puts response.content
```

```rust,no_run
#[tokio::main]
async fn main() -> rust_llm::Result<()> {
    let mut chat = rust_llm::chat()?;
    let response = chat.ask("What is Loco?").await?;
    println!("{}", response.content());

    // The chat keeps the history, so follow-ups have context.
    let response = chat.ask("How do I create my first Loco app?").await?;
    println!("{}", response.content());
    Ok(())
}
```

See [Chat](chat.md) for models, instructions, and request options.

## Streaming a Response

```ruby
chat.ask "Tell me a story about a Ruby programmer" do |chunk|
  print chunk.content
end
```

```rust,no_run
# async fn run(mut chat: rust_llm::Chat) -> rust_llm::Result<()> {
chat.ask_stream("Tell me a story about a Rust programmer", |chunk| print!("{}", chunk.content())).await?;
# Ok(()) }
```

See [Streaming](streaming.md).

## Asking About Files

```ruby
response = RubyLLM.chat.ask "Summarize this document", with: "report.pdf"
```

```rust,no_run
# async fn run() -> rust_llm::Result<()> {
let response = rust_llm::chat()?.ask_with("Summarize this document", vec!["report.pdf".into()]).await?;
# Ok(()) }
```

See [Attachments and Files](attachments-and-files.md).

## Getting Structured Output

```ruby
class PersonSchema < Schematist::Schema
  string :name
  integer :age
end
RubyLLM.chat.with_schema(PersonSchema).ask("Alice is 30 years old.").parsed
```

```rust,no_run
#[derive(schemars::JsonSchema, serde::Deserialize)]
struct Person {
    name: String,
    age: i64,
}

# async fn run() -> rust_llm::Result<()> {
let response = rust_llm::chat()?.with_schema_for::<Person>().ask("Alice is 30 years old.").await?;
let json = response.parsed()?; // Some({"name": "Alice", "age": 30})
# Ok(()) }
```

See [Structured Output](structured-output.md).

## Giving the Model Tools

```ruby
class CurrentTime < RubyLLM::Tool
  description "Returns the current date, time, and time zone"
  def execute = Time.now.to_s
end
RubyLLM.chat.with_tools(CurrentTime).ask "What day is it?"
```

```rust,no_run
use rust_llm::{Tool, ToolCall, ToolError, ToolResult};
use serde_json::{Map, Value};

struct CurrentTime;

#[async_trait::async_trait]
impl Tool for CurrentTime {
    fn description(&self) -> String {
        "Returns the current date, time, and time zone".into()
    }

    async fn execute(&self, _args: Map<String, Value>, _call: &ToolCall) -> Result<ToolResult, ToolError> {
        Ok(format!("{:?}", std::time::SystemTime::now()).into())
    }
}

# async fn run() -> rust_llm::Result<()> {
let response = rust_llm::chat()?.with_tool(CurrentTime).ask("What day is it?").await?;
# Ok(()) }
```

The model calls it as `current_time`. See [Tools](tools.md).

## Defining an Agent

```ruby
class PlanningAssistant < RubyLLM::Agent
  model "gpt-5.6"
  instructions "Help plan the week. Check the current date before suggesting dates."
  tools CurrentTime
end
PlanningAssistant.new.ask "Help me plan a three-day Ruby study schedule."
```

```rust,no_run
# use rust_llm::{Tool, ToolCall, ToolError, ToolResult};
# struct CurrentTime;
# #[async_trait::async_trait]
# impl Tool for CurrentTime {
#     fn description(&self) -> String { String::new() }
#     async fn execute(&self, _: serde_json::Map<String, serde_json::Value>, _: &ToolCall) -> Result<ToolResult, ToolError> { Ok("".into()) }
# }
use std::sync::Arc;
use rust_llm::{Agent, SharedTool};

struct PlanningAssistant;

impl Agent for PlanningAssistant {
    fn model(&self) -> Option<&str> {
        Some("gpt-5.6")
    }
    fn instructions(&self) -> Option<String> {
        Some("Help plan the week. Check the current date before suggesting dates.".into())
    }
    fn tools(&self) -> Vec<SharedTool> {
        vec![Arc::new(CurrentTime)]
    }
}

# async fn run() -> rust_llm::Result<()> {
PlanningAssistant.chat()?.ask("Help me plan a three-day Rust study schedule.").await?;
# Ok(()) }
```

See [Agents](agents.md).

## Generating an Image

```ruby
RubyLLM.paint("A photorealistic red panda coding Ruby").save "red_panda.png"
```

```rust,no_run
# async fn run() -> rust_llm::Result<()> {
let image = rust_llm::paint("A photorealistic red panda coding Rust", Default::default()).await?.into_image();
image.save("red_panda.png").await?;
# Ok(()) }
```

See [Images](images.md).

## Creating an Embedding

```ruby
RubyLLM.embed("Ruby is optimized for programmer happiness.").vectors
```

```rust,no_run
# async fn run() -> rust_llm::Result<()> {
let embedding = rust_llm::embed("Rust is optimized for fearless concurrency.", Default::default()).await?;
let vectors = embedding.vectors; // Vectors::Single(Vec<f64>)
# Ok(()) }
```

See [Embeddings](embeddings.md).

## Tracking Usage and Costs

```ruby
response = RubyLLM.chat.ask "Explain Ruby blocks in one paragraph."
response.tokens.input
response.cost.total
```

```rust,no_run
# async fn run() -> rust_llm::Result<()> {
let response = rust_llm::chat()?.ask("Explain Rust closures in one paragraph.").await?;
let input = response.tokens().input; // Option<i64>
let total = response.cost(None).total(); // Option<f64>, None when pricing is unknown
# Ok(()) }
```

See [Cost and Usage](cost-and-usage.md).

## Using It in Loco

`rust-llm generate install` adds the dependencies, a migration, an initializer, and the
`Chat`/`Message` models to a Loco app; `rust-llm generate chat_ui` adds Inertia + React chat pages.
See [Generators](generators.md) and [Persistence with Loco](persistence-loco.md).

## Not ported

`animate` (video), `speak`, `transcribe`, `ocr`, `moderate`, and `rerank` have no Rust
equivalent yet. Neither do Active Storage attachments or `ruby_llm:load_models`.
