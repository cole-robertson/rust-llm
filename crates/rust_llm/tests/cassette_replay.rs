//! RubyLLM's live specs, replayed from its own recorded cassettes. Every request the port sends
//! must match the body RubyLLM 2.0 recorded, and the parsed responses must satisfy the same
//! expectations as the Ruby spec.

mod support;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use rust_llm::{Chat, Parameter, Role, Tool, ToolCall, ToolError, ToolResult};
use serde_json::{Map, Value, json};
use support::{CHAT_MODELS, Cassette, cassette_name, config_for};

fn chat_for(cassette: &Cassette, provider: &str, model: &str) -> Chat {
    let assume = matches!(provider, "ollama" | "gpustack" | "ollama_cloud" | "hetzner");
    Chat::with_config(config_for(cassette, provider), Some(model), Some(provider), assume).expect("chat")
}

struct Weather;

#[async_trait]
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
    async fn execute(&self, args: Map<String, Value>, _: &ToolCall) -> Result<ToolResult, ToolError> {
        let lat = args["latitude"].as_str().map(str::to_string).unwrap_or_else(|| args["latitude"].to_string());
        let lon = args["longitude"].as_str().map(str::to_string).unwrap_or_else(|| args["longitude"].to_string());
        Ok(format!("Current weather at {lat}, {lon}: 15°C, Wind: 10 km/h").into())
    }
}

struct BestLanguageToLearn;

#[async_trait]
impl Tool for BestLanguageToLearn {
    fn description(&self) -> String {
        "Gets the best language to learn".into()
    }
    async fn execute(&self, _: Map<String, Value>, _: &ToolCall) -> Result<ToolResult, ToolError> {
        Ok("Ruby".into())
    }
}

struct DiceRoll(Arc<AtomicUsize>);

#[async_trait]
impl Tool for DiceRoll {
    fn description(&self) -> String {
        "Rolls a single six-sided die and returns the result".into()
    }
    async fn execute(&self, _: Map<String, Value>, _: &ToolCall) -> Result<ToolResult, ToolError> {
        let n = self.0.fetch_add(1, Ordering::SeqCst) + 1;
        Ok(json!({ "roll": n }).into())
    }
}

fn total_input(m: &rust_llm::Message) -> i64 {
    let t = m.tokens();
    t.input.unwrap_or(0) + t.cache_read.unwrap_or(0) + t.cache_write.unwrap_or(0)
}

/// Runs `body` for each chat model that has a recorded cassette, and reports every failure
/// together instead of stopping at the first.
macro_rules! each_model {
    ($describe:expr, $it:expr, |$chat:ident, $provider:ident, $model:ident| $body:block) => {{
        let mut failures = Vec::new();
        let mut ran = 0;
        for &($provider, $model) in CHAT_MODELS {
            let name = cassette_name($describe, $provider, $model, $it);
            let Some(cassette) = Cassette::start(&name).await else { continue };
            ran += 1;
            #[allow(unused_mut)]
            let mut $chat = chat_for(&cassette, $provider, $model);
            let outcome: Result<(), String> = async { $body }.await;
            let replay = std::panic::AssertUnwindSafe(cassette.assert_all_matched());
            let replay = futures::FutureExt::catch_unwind(replay).await;
            if let Err(e) = outcome {
                failures.push(format!("{} {}: {e}", $provider, $model));
            } else if let Err(p) = replay {
                let msg = p.downcast_ref::<String>().cloned().unwrap_or_default();
                failures.push(format!("{} {}: {msg}", $provider, $model));
            }
        }
        assert!(ran > 0, "no cassettes found for {}", $it);
        assert!(failures.is_empty(), "{} of {ran} providers failed:\n{}", failures.len(), failures.join("\n\n"));
        eprintln!("{}: {ran} providers replayed", $it);
    }};
}

fn check(cond: bool, what: impl Into<String>) -> Result<(), String> {
    if cond { Ok(()) } else { Err(what.into()) }
}

#[tokio::test]
async fn basic_conversation() {
    each_model!("chat basic chat functionality", "can have a basic conversation", |chat, provider, model| {
        let response = chat.ask("What's 2 + 2?").await.map_err(|e| e.to_string())?;
        check(response.content().contains('4'), format!("content {:?}", response.content))?;
        check(response.role == Role::Assistant, "role")?;
        check(total_input(&response) > 0, "input tokens")?;
        check(response.tokens().output.unwrap_or(0) > 0, "output tokens")
    });
}

#[tokio::test]
async fn multi_turn_conversation() {
    each_model!("chat basic chat functionality", "can handle multi-turn conversations", |chat, provider, model| {
        let first = chat.ask("Who is the creator of the programming language Ruby?").await.map_err(|e| e.to_string())?;
        check(first.content().contains("Matz"), format!("first {:?}", first.content))?;
        let followup = chat.ask("What year did he create Ruby?").await.map_err(|e| e.to_string())?;
        check(followup.content().contains("199"), format!("followup {:?}", followup.content))
    });
}

#[tokio::test]
async fn system_prompt() {
    each_model!("chat basic chat functionality", "successfully uses the system prompt", |chat, provider, model| {
        chat = chat.with_instructions(r#"You must include the exact phrase "XKCD7392" somewhere in your response."#);
        let response = chat.ask("Tell me about the weather.").await.map_err(|e| e.to_string())?;
        check(response.content().to_lowercase().contains("xkcd7392"), "marker")
    });
}

#[tokio::test]
async fn replaces_previous_system_messages() {
    each_model!("chat basic chat functionality", "replaces previous system messages by default", |chat, provider, model| {
        chat = chat.with_instructions(r#"You must include the exact phrase "XKCD7392" somewhere in your response."#);
        let r1 = chat.ask("Tell me about the weather.").await.map_err(|e| e.to_string())?;
        check(r1.content().to_lowercase().contains("xkcd7392"), "first marker")?;
        chat = chat.with_instructions(r#"You must include the exact phrase "PURPLE-ELEPHANT-42" somewhere in your response."#);
        let r2 = chat.ask("What are some good books?").await.map_err(|e| e.to_string())?;
        check(!r2.content().to_lowercase().contains("xkcd7392"), "old marker gone")?;
        check(r2.content().to_lowercase().contains("purple"), "new marker")
    });
}

#[tokio::test]
async fn raw_responses() {
    each_model!("chat basic chat functionality", "returns raw responses", |chat, provider, model| {
        let response = chat.ask("What is the capital of France?").await.map_err(|e| e.to_string())?;
        let raw = response.raw.as_ref().ok_or("raw")?;
        check(raw.status == 200, "status")?;
        check(!raw.headers.is_empty(), "headers")?;
        check(!raw.request_body.is_null(), "request body")
    });
}

#[tokio::test]
async fn streaming() {
    each_model!("chat streaming responses", "supports streaming responses", |chat, provider, model| {
        let mut chunks = 0;
        let response = chat.ask_stream("Count from 1 to 3", |_| chunks += 1).await.map_err(|e| e.to_string())?;
        check(chunks > 0, "no chunks")?;
        check(response.raw.as_ref().is_some_and(|r| r.status == 200), "raw status")?;
        check(!response.content().is_empty() || response.thinking.is_some(), "content")
    });
}

#[tokio::test]
async fn tools() {
    each_model!("chat function calling", "can use tools", |chat, provider, model| {
        chat = chat.with_tool(Weather);
        let response = chat.ask("What's the weather in Berlin? (52.5200, 13.4050)").await.map_err(|e| e.to_string())?;
        check(response.content().contains("15"), format!("content {:?}", response.content))?;
        check(response.content().contains("10"), "wind")
    });
}

#[tokio::test]
async fn tools_multi_turn() {
    each_model!("chat function calling", "can use tools in multi-turn conversations", |chat, provider, model| {
        chat = chat.with_tool(Weather);
        let r1 = chat.ask("What's the weather in Berlin? (52.5200, 13.4050)").await.map_err(|e| e.to_string())?;
        check(r1.content().contains("15"), "berlin")?;
        let r2 = chat.ask("What's the weather in Paris? (48.8575, 2.3514)").await.map_err(|e| e.to_string())?;
        check(r2.content().contains("15"), "paris")
    });
}

#[tokio::test]
async fn tools_without_parameters() {
    each_model!("chat function calling", "can use tools without parameters", |chat, provider, model| {
        chat = chat.with_tool(BestLanguageToLearn);
        let response = chat
            .ask("Use the best_language_to_learn tool to tell me which language to learn.")
            .await
            .map_err(|e| e.to_string())?;
        check(chat.messages().iter().any(|m| m.role == Role::Assistant && m.is_tool_call()), "tool call made")?;
        check(response.content().contains("Ruby"), format!("content {:?}", response.content))
    });
}

#[tokio::test]
async fn tools_streaming_multi_turn() {
    each_model!("chat function calling", "can use tools with multi-turn streaming conversations", |chat, provider, model| {
        chat = chat.with_tool(Weather);
        let mut chunks = 0;
        let r1 = chat
            .ask_stream("What's the weather in Berlin? (52.5200, 13.4050)", |_| chunks += 1)
            .await
            .map_err(|e| e.to_string())?;
        check(chunks > 0, "chunks")?;
        check(r1.content().contains("15"), format!("berlin {:?}", r1.content))?;
        let r2 = chat
            .ask_stream("What's the weather in Paris? (48.8575, 2.3514)", |_| {})
            .await
            .map_err(|e| e.to_string())?;
        check(r2.content().contains("15"), format!("paris {:?}", r2.content))
    });
}

#[tokio::test]
async fn multiple_tool_calls_in_one_response() {
    each_model!("chat function calling", "can handle multiple tool calls in a single response", |chat, provider, model| {
        let count = Arc::new(AtomicUsize::new(0));
        chat = chat
            .with_tool(DiceRoll(count.clone()))
            .with_instructions("You must call the dice_roll tool exactly 3 times when asked to roll dice 3 times.");
        let response = chat.ask("Roll the dice 3 times").await.map_err(|e| e.to_string())?;
        check(count.load(Ordering::SeqCst) == 3, format!("rolled {} times", count.load(Ordering::SeqCst)))?;
        check(response.content().chars().any(|c| c.is_ascii_digit()), "numbers")
    });
}
