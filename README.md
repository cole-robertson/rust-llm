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

```rust
#[tokio::main]
async fn main() -> rust_llm::Result<()> {
    let mut chat = rust_llm::chat_with("claude-opus-5-5")?;

    let answer = chat.ask("What's the best way to learn Rust?").await?;
    println!("{}", answer.content());

    Ok(())
}
```

API keys are read from the environment (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`,
...), or you can set them in code:

```rust
rust_llm::configure(|config| {
    config.anthropic_api_key(std::env::var("ANTHROPIC_API_KEY").unwrap_or_default());
    config.default_model = "claude-opus-5-5".into();
});
```

## Chat

```rust
let mut chat = rust_llm::chat_with("claude-opus-5-5")?
    .with_instructions("You are a concise Rust mentor.")
    .with_temperature(0.2);

chat.ask("What is ownership?").await?;

// The chat keeps its history, so follow-ups have context.
let reply = chat.ask("Show me an example.").await?;
println!("{}", reply.content());
```

Attach images, PDFs, audio, or video by path or URL:

```rust
let mut chat = rust_llm::chat_with("claude-opus-5-5")?;

chat.ask_with("What's in this picture?", vec!["photo.jpg".into()])
    .await?;
```

Stream the reply as it arrives:

```rust
let mut chat = rust_llm::chat_with("claude-opus-5-5")?;

chat.ask_stream("Tell me a story", |chunk| {
    print!("{}", chunk.content());
})
.await?;
```

## Tools

```rust
use rust_llm::{Parameter, Tool, ToolCall, ToolError, ToolResult};
use serde_json::{json, Map, Value};

struct Weather;

#[async_trait::async_trait]
impl Tool for Weather {
    fn description(&self) -> String {
        "Get the current weather for a location".into()
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![
            Parameter::new("latitude"),
            Parameter::new("longitude"),
        ]
    }

    async fn execute(
        &self,
        args: Map<String, Value>,
        _call: &ToolCall,
    ) -> Result<ToolResult, ToolError> {
        let report = json!({
            "latitude": args["latitude"],
            "longitude": args["longitude"],
            "temperature": 14.2,
        });
        Ok(report.into())
    }
}

let mut chat = rust_llm::chat_with("claude-opus-5-5")?.with_tool(Weather);
chat.ask("What's the weather in Berlin?").await?;
```

Tools can require approval before they run (`requires_approval`), report progress, or come from an
MCP server.

## Agents

```rust
use std::sync::Arc;

use rust_llm::{Agent, SharedTool};

struct WeatherAssistant;

impl Agent for WeatherAssistant {
    fn model(&self) -> Option<&str> {
        Some("claude-opus-5-5")
    }

    fn instructions(&self) -> Option<String> {
        Some("Be concise and always use tools for weather.".into())
    }

    fn tools(&self) -> Vec<SharedTool> {
        vec![Arc::new(Weather)]
    }
}

WeatherAssistant
    .chat()?
    .ask("What's the weather in Paris?")
    .await?;
```

## Structured output

```rust
#[derive(schemars::JsonSchema)]
struct Product {
    name: String,
    price: f64,
    features: Vec<String>,
}

let reply = rust_llm::chat_with("claude-opus-5-5")?
    .with_schema_for::<Product>()
    .ask("Invent a mechanical keyboard for Rust programmers.")
    .await?;

// Some({"name": "...", "price": 149.0, "features": [...]})
let product = reply.parsed()?;
```

## Embeddings, images, audio

```rust
let embedding = rust_llm::embed("Rust is fast and safe", Default::default()).await?;

let images = rust_llm::paint("A sunset over mountains in watercolor", Default::default()).await?;
images.into_image().save("sunset.png").await?;

let speech = rust_llm::speak("Welcome aboard!", Default::default()).await?;
speech.save("welcome.mp3")?;

let transcript = rust_llm::transcribe("meeting.wav", Default::default()).await?;
println!("{}", transcript.text.unwrap_or_default());
```

Each call uses the configured default model for that task; pass options to choose another.

## MCP

```rust
use rust_llm::Mcp;

let github = Mcp::command(["npx", "-y", "@modelcontextprotocol/server-github"]).build()?;

let mut chat = rust_llm::chat_with("claude-opus-5-5")?.with_mcp(github);
chat.ask("List my open pull requests.").await?;
```

Stdio and Streamable HTTP servers, their resources and prompts, input requests, and OAuth.

## Batches

Send many chats at the provider's batch discount and collect the answers later:

```rust
let tickets = ["Refund my order", "The app crashes on login"];

let mut chats = Vec::new();
for ticket in tickets {
    let mut chat = rust_llm::chat_with("claude-opus-5-5")?;
    chat.ask_later(ticket)?;
    chats.push(chat);
}

let mut batch = rust_llm::batch(chats).await?;

// Later, for example from a scheduled job:
if batch.refresh().await?.is_complete() {
    for message in batch.messages().await?.into_iter().flatten() {
        println!("{}", message.content());
    }
}
```

## Judgments

Calibrated probabilities, choices, and scores from a judgment model such as TypeSafe's Jev:

```rust
let urgency = rust_llm::Judge::new()
    .probability("urgent", "Does this need attention today?")?;

let judgment = urgency
    .judge("Please refund the duplicate charge today.")
    .await?;

println!("{:?}", judgment.probability("urgent")); // Some(0.91)
```

## Cost and usage

```rust
let mut chat = rust_llm::chat_with("claude-opus-5-5")?;
let reply = chat.ask("Explain lifetimes in one paragraph.").await?;

println!("{:?}", reply.tokens().input); // Some(14)

// Every billed request, including retries and fallbacks
println!("{:?}", chat.cost().total()); // Some(0.0042)
```

## Loco

`rust_llm_loco` persists chats, messages, tool calls, and usage with SeaORM, so a conversation can
continue in another request or background job:

```rust
use rust_llm_loco::ChatRecord;

let record = ChatRecord::create(&ctx.db, "claude-opus-5-5", None).await?;
let mut chat = record.to_llm(&ctx.db).await?.with_tool(Weather);

record
    .ask(&ctx.db, &mut chat, "What's the weather in Berlin?")
    .await?;
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
