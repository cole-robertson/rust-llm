# RustLLM

One Rust API for chat, tools, agents, structured output, streaming, embeddings, images, audio,
batches, MCP, and judgments across OpenAI, Anthropic, Gemini, and ten more providers. Includes
persistence for [Loco](https://loco.rs) apps and generators for a full chat UI.

The API follows [RubyLLM](https://github.com/crmne/ruby_llm) 2.0: if you know RubyLLM, you already
know RustLLM. Same method names, same behavior, Rust types.
[RubyLLM vs RustLLM](docs/rubyllm.md) maps one to the other.

[![crates.io](https://img.shields.io/crates/v/rust_llm.svg)](https://crates.io/crates/rust_llm)
[![docs.rs](https://img.shields.io/docsrs/rust_llm)](https://docs.rs/rust_llm)
[![CI](https://github.com/cole-robertson/rust-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/cole-robertson/rust-llm/actions/workflows/ci.yml)

```toml
[dependencies]
rust_llm = "2.0"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

```rust,no_run
#[tokio::main]
async fn main() -> rust_llm::Result<()> {
    let mut chat = rust_llm::chat()?;
    let answer = chat.ask("What's the best way to learn Rust?").await?;
    println!("{}", answer.content());
    Ok(())
}
```

API keys come from the environment (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, ...),
or set them in code:

```rust,no_run
rust_llm::configure(|c| {
    c.anthropic_api_key(std::env::var("ANTHROPIC_API_KEY").unwrap_or_default());
    c.default_model = "claude-haiku-4-5".into();
});
```

## Chat

```rust,no_run
# async fn run() -> Result<(), Box<dyn std::error::Error>> {
let mut chat = rust_llm::chat_with("claude-haiku-4-5")?
    .with_instructions("You are a concise Rust mentor.")
    .with_temperature(0.2);

chat.ask("What is ownership?").await?;
let reply = chat.ask("Show me an example.").await?; // the chat keeps its history
println!("{}", reply.content());

// Files: images, PDFs, audio, video, by path or URL
chat.ask_with("What's in this picture?", vec!["photo.jpg".into()]).await?;

// Streaming
chat.ask_stream("Tell me a story", |chunk| print!("{}", chunk.content())).await?;
# Ok(()) }
```

## Tools

```rust,no_run
use rust_llm::{Parameter, Tool, ToolCall, ToolError, ToolResult};
use serde_json::{json, Map, Value};

struct Weather;

#[async_trait::async_trait]
impl Tool for Weather {
    fn description(&self) -> String {
        "Get the current weather for a location".into()
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![Parameter::new("latitude"), Parameter::new("longitude")]
    }

    async fn execute(&self, args: Map<String, Value>, _: &ToolCall) -> Result<ToolResult, ToolError> {
        Ok(json!({ "temperature": 14.2, "lat": args["latitude"], "lon": args["longitude"] }).into())
    }
}

# async fn run() -> Result<(), Box<dyn std::error::Error>> {
let mut chat = rust_llm::chat()?.with_tool(Weather);
chat.ask("What's the weather in Berlin?").await?;
# Ok(()) }
```

Tools can require approval before they run (`requires_approval`), report progress, or come from an
MCP server.

## Agents

```rust,no_run
# use rust_llm::{Tool, ToolCall, ToolError, ToolResult};
# struct Weather;
# #[async_trait::async_trait]
# impl Tool for Weather {
#     fn description(&self) -> String { String::new() }
#     async fn execute(&self, _: serde_json::Map<String, serde_json::Value>, _: &ToolCall) -> Result<ToolResult, ToolError> { Ok("".into()) }
# }
use std::sync::Arc;
use rust_llm::{Agent, SharedTool};

struct WeatherAssistant;

impl Agent for WeatherAssistant {
    fn model(&self) -> Option<&str> {
        Some("claude-haiku-4-5")
    }
    fn instructions(&self) -> Option<String> {
        Some("Be concise and always use tools for weather.".into())
    }
    fn tools(&self) -> Vec<SharedTool> {
        vec![Arc::new(Weather)]
    }
}

# async fn run() -> Result<(), Box<dyn std::error::Error>> {
WeatherAssistant.chat()?.ask("What's the weather in Paris?").await?;
# Ok(()) }
```

## Structured output

```rust,no_run
#[derive(schemars::JsonSchema)]
struct Product {
    name: String,
    price: f64,
    features: Vec<String>,
}

# async fn run() -> Result<(), Box<dyn std::error::Error>> {
let reply = rust_llm::chat()?
    .with_schema_for::<Product>()
    .ask("Invent a mechanical keyboard for Rust programmers.")
    .await?;
let product = reply.parsed()?; // Some({"name": "...", "price": 149.0, "features": [...]})
# Ok(()) }
```

## Embeddings, images, audio

```rust,no_run
# async fn run() -> Result<(), Box<dyn std::error::Error>> {
let embedding = rust_llm::embed("Rust is fast and safe", Default::default()).await?;

let image = rust_llm::paint("a sunset over mountains in watercolor", Default::default()).await?;
image.into_image().save("sunset.png").await?;

rust_llm::speak("Welcome aboard!", Default::default()).await?.save("welcome.mp3")?;
let text = rust_llm::transcribe("meeting.wav", Default::default()).await?.text;
# Ok(()) }
```

## MCP

```rust,no_run
use rust_llm::Mcp;

# async fn run() -> Result<(), Box<dyn std::error::Error>> {
let github = Mcp::command(["npx", "-y", "@modelcontextprotocol/server-github"]).build()?;
let mut chat = rust_llm::chat()?.with_mcp(github);
chat.ask("List my open pull requests.").await?;
# Ok(()) }
```

Stdio and Streamable HTTP servers, their resources and prompts, input requests, and OAuth.

## Batches

Send many chats at the provider's batch discount and collect the answers later:

```rust,no_run
# async fn run(tickets: Vec<String>) -> Result<(), Box<dyn std::error::Error>> {
let mut chats = Vec::new();
for ticket in &tickets {
    let mut chat = rust_llm::chat()?;
    chat.ask_later(ticket.as_str())?;
    chats.push(chat);
}
let mut batch = rust_llm::batch(chats).await?;

// later
if batch.refresh().await?.is_complete() {
    for message in batch.messages().await?.into_iter().flatten() {
        println!("{}", message.content());
    }
}
# Ok(()) }
```

## Judgments

Calibrated probabilities, choices, and scores from a judgment model such as TypeSafe's Jev:

```rust,no_run
# async fn run() -> Result<(), Box<dyn std::error::Error>> {
let urgency = rust_llm::Judge::new().probability("urgent", "Does this need attention today?")?;
let judgment = urgency.judge("Please refund the duplicate charge today.").await?;
judgment.probability("urgent"); // Some(0.91)
# Ok(()) }
```

## Cost and usage

```rust,no_run
# async fn run(mut chat: rust_llm::Chat) -> Result<(), Box<dyn std::error::Error>> {
let reply = chat.ask("Summarize our conversation.").await?;
reply.tokens().input;   // Some(812)
chat.cost().total();    // Some(0.0042), across every request including retries and fallbacks
# Ok(()) }
```

## Loco

`rust_llm_loco` persists chats, messages, tool calls, and usage with SeaORM, so a conversation can
continue in another request or background job:

```rust,no_run
# use rust_llm::{Tool, ToolCall, ToolError, ToolResult};
# struct Weather;
# #[async_trait::async_trait]
# impl Tool for Weather {
#     fn description(&self) -> String { String::new() }
#     async fn execute(&self, _: serde_json::Map<String, serde_json::Value>, _: &ToolCall) -> Result<ToolResult, ToolError> { Ok("".into()) }
# }
use rust_llm_loco::ChatRecord;

# struct Ctx { db: sea_orm::DatabaseConnection }
# async fn run(ctx: Ctx) -> Result<(), Box<dyn std::error::Error>> {
let record = ChatRecord::create(&ctx.db, "claude-haiku-4-5", None).await?;
let mut chat = record.to_llm(&ctx.db).await?.with_tool(Weather);
record.ask(&ctx.db, &mut chat, "What's the weather in Berlin?").await?;
# Ok(()) }
```

The `rust-llm` CLI sets up a Loco app and generates an Inertia + React chat UI:

```sh
cargo install rust_llm_cli
rust-llm generate install        # dependencies, migration, initializer, Chat/Message models
rust-llm generate chat_ui        # chat pages, controllers, and a background worker
rust-llm generate tool Weather
rust-llm generate agent Support
```

## Providers

OpenAI, Anthropic, Gemini, DeepSeek, Mistral, OpenRouter, xAI, Perplexity, Ollama, Ollama Cloud,
GPUStack, Hetzner, and TypeSafe.

## Docs

- [Guides](docs/README.md): every feature in depth, with samples that are compiled in CI
- [API reference](https://docs.rs/rust_llm)
- [RubyLLM vs RustLLM](docs/rubyllm.md): RustLLM is a port of
  [RubyLLM](https://github.com/crmne/ruby_llm) 2.0; this page maps one to the other and explains
  how the port is verified
- [Benchmarks](docs/BENCHMARK.md)
- [Contributing](CONTRIBUTING.md)

## License

MIT. RustLLM is a port of Carmine Paolino's RubyLLM; see [LICENSE](https://github.com/cole-robertson/rust-llm/blob/main/LICENSE).
