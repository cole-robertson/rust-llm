//! The RubyLLM README, in Rust, against a live provider.
//!
//!   ANTHROPIC_API_KEY=... cargo run -p ruby_llm --example readme
//!   (ANTHROPIC_API_BASE points it at a proxy.)

use std::sync::Arc;

use async_trait::async_trait;
use ruby_llm::{Agent, Parameter, SharedTool, Tool, ToolCall, ToolError, ToolResult};
use serde_json::{Map, Value, json};

/// ```ruby
/// class Weather < RubyLLM::Tool
///   description "Get current weather"
///   def execute(latitude:, longitude:) = ...
/// end
/// ```
struct Weather;

#[async_trait]
impl Tool for Weather {
    fn description(&self) -> String {
        "Get current weather".into()
    }
    fn parameters(&self) -> Vec<Parameter> {
        vec![Parameter::new("latitude"), Parameter::new("longitude")]
    }
    async fn execute(&self, args: Map<String, Value>, _: &ToolCall) -> Result<ToolResult, ToolError> {
        Ok(json!({ "latitude": args["latitude"], "longitude": args["longitude"], "temperature_2m": 14.2, "wind_speed_10m": 11.0 }).into())
    }
}

/// ```ruby
/// class WeatherAssistant < RubyLLM::Agent
///   model "claude-haiku-4-5"
///   instructions "Be concise and always use tools for weather."
///   tools Weather
/// end
/// ```
struct WeatherAssistant;

impl Agent for WeatherAssistant {
    fn model(&self) -> Option<&str> {
        Some("claude-haiku-4-5")
    }
    fn provider(&self) -> Option<&str> {
        Some("anthropic")
    }
    fn instructions(&self) -> Option<String> {
        Some("Be concise and always use tools for weather.".into())
    }
    fn tools(&self) -> Vec<SharedTool> {
        vec![Arc::new(Weather)]
    }
}

#[derive(schemars::JsonSchema)]
#[allow(dead_code)]
struct Product {
    name: String,
    price: f64,
    features: Vec<String>,
}

/// `RubyLLM.chat(model: "claude-haiku-4-5", provider: :anthropic)`: with a provider the alias
/// resolves to Anthropic's dated id, which the local proxy requires.
fn new_chat() -> ruby_llm::Result<ruby_llm::Chat> {
    ruby_llm::Chat::new(Some("claude-haiku-4-5"), Some("anthropic"))
}

#[tokio::main]
async fn main() -> ruby_llm::Result<()> {
    ruby_llm::configure(|c| {
        c.default_model = "claude-haiku-4-5".into();
        if let Ok(base) = std::env::var("ANTHROPIC_BASE_URL") {
            c.set("anthropic_api_base", base);
        }
        if let Ok(key) = std::env::var("ANTHROPIC_AUTH_TOKEN") {
            c.set("anthropic_api_key", key);
        }
    });

    // RubyLLM.chat.ask "What's the best way to learn Ruby?"
    let answer = new_chat()?.with_max_output_tokens(200).ask("In one sentence: what's the best way to learn Rust?").await?;
    println!("ask       -> {}", answer.content());

    // chat.ask("Tell me a story") { |chunk| print chunk.content }
    print!("stream    -> ");
    let mut chat = new_chat()?;
    let streamed = chat
        .ask_stream("Count from 1 to 5, comma separated, nothing else.", |chunk| {
            if let Some(text) = &chunk.content {
                print!("{text}");
            }
        })
        .await?;
    println!("   [{} in / {} out tokens]", streamed.tokens().input.unwrap_or(0), streamed.tokens().output.unwrap_or(0));

    // chat.with_tools(Weather).ask "What's the weather in Berlin?"
    let mut chat = new_chat()?.with_tool(Weather).before_tool_call(|call| println!("tool      -> {}({})", call.name, Value::Object(call.arguments())));
    let weather = chat.ask("What's the weather in Berlin (52.52, 13.405)? One sentence.").await?;
    println!("tools     -> {}", weather.content());

    // WeatherAssistant.new.ask "What's the weather in Berlin?"
    let reply = WeatherAssistant.chat()?.ask("Weather in Paris (48.8575, 2.3514)?").await?;
    println!("agent     -> {}", reply.content());

    // chat.with_schema(ProductSchema).ask "Analyze this product"
    let mut chat = new_chat()?.with_schema_for::<Product>();
    let product = chat.ask("Invent a product: a mechanical keyboard for Rust programmers.").await?;
    println!("schema    -> {}", product.parsed()?.unwrap_or_default());

    println!("cost      -> ${:.6} across {} billed requests", chat.cost().total().unwrap_or_default(), chat.usage_entries().len());
    Ok(())
}
