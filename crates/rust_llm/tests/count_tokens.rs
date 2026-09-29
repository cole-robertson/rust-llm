//! `chat.count_tokens(message)` and `RubyLLM.count_tokens`, replayed from RubyLLM's
//! `chat_count_tokens_*` and `chat_rubyllm_count_tokens_*` cassettes. Assertions follow
//! `spec/ruby_llm/chat_count_tokens_spec.rb`. The Vertex AI cassette needs a provider RustLLM
//! does not port.

mod support;

use rust_llm::{Attachment, Chat, FnTool, Parameter, ThinkingConfig, ToolResult};
use serde_json::json;
use support::{Cassette, config_for};

/// `CountingWeather < RubyLLM::Tool`.
fn counting_weather() -> FnTool {
    FnTool::new("counting_weather", "Gets current weather for a city", |args| async move {
        Ok(ToolResult::from(format!("It's sunny in {}.", args.get("city").and_then(|c| c.as_str()).unwrap_or(""))))
    })
    .parameter(Parameter::new("city").description("City name"))
}

async fn start(name: &str) -> Cassette {
    Cassette::start(name).await.unwrap_or_else(|| panic!("missing cassette {name}; run bin/convert-cassettes 'chat_count_tokens_*'"))
}

fn chat(cassette: &Cassette, provider: &str, model: &str) -> Chat {
    Chat::with_config(config_for(cassette, provider), Some(model), Some(provider), false).unwrap()
}

fn named(provider: &str, model: &str, it: &str) -> String {
    format!("chat_count_tokens_with_{provider}_{}_{it}", model.replace('.', "_"))
}

const QUESTION: &str = "What is the capital of France?";

/// "counts a staged message without mutating the chat", for each provider with counting.
#[tokio::test]
async fn counts_a_staged_message_without_mutating_the_chat() {
    for (provider, model, expected) in
        [("openai", "gpt-5-nano", 13), ("anthropic", "claude-haiku-4-5", 14), ("gemini", "gemini-3.5-flash", 8)]
    {
        let cassette = start(&named(provider, model, "counts_a_staged_message_without_mutating_the_chat")).await;
        let chat = chat(&cassette, provider, model);
        let count = chat.count_tokens(Some(QUESTION)).await.unwrap();
        assert_eq!(count, expected, "{provider}");
        assert!(chat.messages().is_empty(), "{provider}: counting stages nothing");
        assert!(chat.usage_entries().is_empty(), "{provider}: counting is not billed");
        cassette.assert_all_matched().await;
    }
}

/// OpenAI "counts instructions, tools and structured output".
#[tokio::test]
async fn openai_counts_instructions_tools_and_structured_output() {
    let cassette = start(&named("openai", "gpt-5-nano", "counts_instructions_tools_and_structured_output")).await;
    let mut chat = chat(&cassette, "openai", "gpt-5-nano");
    chat.ask_later("What is the weather in Berlin?").unwrap();
    let base = chat.count_tokens(None).await.unwrap();
    let schema = json!({
        "type": "object", "properties": { "weather": { "type": "string" } },
        "required": ["weather"], "additionalProperties": false
    });
    let chat = chat.with_instructions("Be terse.").with_tool(counting_weather()).with_schema(schema);
    let configured = chat.count_tokens(None).await.unwrap();
    assert!(configured > base, "{configured} > {base}");
    assert_eq!((base, configured), (13, 82));
    cassette.assert_all_matched().await;
}

/// OpenAI "counts image attachments before generation".
#[tokio::test]
async fn openai_counts_image_attachments_before_generation() {
    let cassette = start(&named("openai", "gpt-5-nano", "counts_image_attachments_before_generation")).await;
    let mut chat = chat(&cassette, "openai", "gpt-5-nano");
    let base = chat.count_tokens(Some("Describe this image.")).await.unwrap();
    let image = Attachment::new(format!("{}/tests/fixtures/ruby.png", env!("CARGO_MANIFEST_DIR")));
    chat.ask_later_with("Describe this image.", vec![image]).unwrap();
    let with_image = chat.count_tokens(None).await.unwrap();
    assert!(with_image > base, "{with_image} > {base}");
    cassette.assert_all_matched().await;
}

/// "counts the next request as configured" on Anthropic and Gemini.
#[tokio::test]
async fn counts_the_next_request_as_configured() {
    for (provider, model) in [("anthropic", "claude-haiku-4-5"), ("gemini", "gemini-3.5-flash")] {
        let cassette = start(&named(provider, model, "counts_the_next_request_as_configured")).await;
        let mut chat = chat(&cassette, provider, model);
        chat.ask_later(QUESTION).unwrap();
        let base = chat.count_tokens(None).await.unwrap();
        let chat = chat.with_instructions("Be terse.").with_tool(counting_weather());
        let configured = chat.count_tokens(None).await.unwrap();
        assert!(configured > base, "{provider}: {configured} > {base}");
        cassette.assert_all_matched().await;
    }
}

/// Anthropic "counts requests with thinking enabled".
#[tokio::test]
async fn anthropic_counts_requests_with_thinking_enabled() {
    let cassette = start(&named("anthropic", "claude-haiku-4-5", "counts_requests_with_thinking_enabled")).await;
    let chat = chat(&cassette, "anthropic", "claude-haiku-4-5").with_thinking(ThinkingConfig::budget(2048));
    assert_eq!(chat.count_tokens(Some(QUESTION)).await.unwrap(), 43);
    cassette.assert_all_matched().await;
}

/// `RubyLLM.count_tokens(text, model:)` "counts one user message": with no provider the registry
/// keeps the bare `claude-haiku-4-5` id, as recorded. Replayed through `Context#count_tokens`,
/// which is the same `chat(model:, provider:).count_tokens(text)` against an isolated config.
#[tokio::test]
async fn rust_llm_count_tokens_counts_one_user_message() {
    let cassette = start("chat_rubyllm_count_tokens_counts_one_user_message").await;
    let context = rust_llm::Context::new((*config_for(&cassette, "anthropic")).clone());
    assert_eq!(context.count_tokens(QUESTION, Some("claude-haiku-4-5"), None).await.unwrap(), 14);
    cassette.assert_all_matched().await;
}

/// "with a provider without token counting raises a clear error".
#[tokio::test]
async fn providers_without_token_counting_fail_clearly() {
    let mut config = rust_llm::Config::default();
    config.set("deepseek_api_key", "test");
    let chat = Chat::with_config(config.into(), Some("deepseek-v4-flash"), Some("deepseek"), false).unwrap();
    let err = chat.count_tokens(Some("Hello")).await.unwrap_err();
    assert!(err.to_string().contains("doesn't support token counting"), "{err}");
}
