//! Extended thinking, replayed from RubyLLM's `chat_with_extended_thinking_*` and
//! `chat_thinking_display_*` cassettes (`spec/ruby_llm/chat_thinking_spec.rb`), the
//! `with_thinking` payload examples from the same spec, and the cross-provider replay rules of
//! `spec/ruby_llm/chat_thinking_replay_spec.rb`.

mod support;

use rust_llm::message::{Operation, indexmap_lite::IndexMap};
use rust_llm::{Chat, Cost, Message, Role, Thinking, ThinkingConfig, ThinkingDisplay, Tokens, ToolCall, UsageEntry, UsageStatus};
use serde_json::{Map, Value, json};
use support::{Cassette, cassette_name, chat_for};

/// `THINKING_MODELS` for the providers this port implements.
const THINKING_MODELS: &[(&str, &str)] = &[
    ("anthropic", "claude-haiku-4-5"),
    ("gemini", "gemini-3-flash-preview"),
    ("gpustack", "qwen3"),
    ("mistral", "mistral-small-latest"),
    ("ollama", "qwen3"),
    ("openai", "gpt-5.4"),
    ("openrouter", "claude-haiku-4-5"),
    ("xai", "grok-3-mini"),
];

const QUESTION: &str = "If a magic mirror shows your future self, but only if you ask a question it cannot answer truthfully, what question do you ask to see your future, and what would the mirror reveal about the answer it gives?";

/// `thinking_config_for(provider)`.
fn thinking_config_for(provider: &str) -> Option<ThinkingConfig> {
    match provider {
        "anthropic" => Some(ThinkingConfig::budget(1024)),
        "gemini" => Some(ThinkingConfig::effort("low")),
        "mistral" => Some(ThinkingConfig::effort("high")),
        "gpustack" | "ollama" => None,
        _ => Some(ThinkingConfig::effort("medium")),
    }
}

fn chat_with_thinking(cassette: &Cassette, provider: &str, model: &str) -> Chat {
    let chat = chat_for(cassette, provider, model);
    match thinking_config_for(provider) {
        Some(config) => chat.with_thinking(config),
        None => chat,
    }
}

fn prompt_for(provider: &str) -> &'static str {
    if provider == "gpustack" { "What is 5 + 3? Think briefly before answering." } else { QUESTION }
}

fn check(cond: bool, what: impl Into<String>) -> Result<(), String> {
    if cond { Ok(()) } else { Err(what.into()) }
}

/// `expect_response_payload`: content, or thinking text when the model only thought.
fn expect_response_payload(response: &Message) -> Result<(), String> {
    let thinking = response.thinking.as_ref().and_then(|t| t.text.as_deref()).unwrap_or("");
    check(!response.content().trim().is_empty() || !thinking.trim().is_empty(), "no content or thinking")
}

/// Runs `body` for each thinking model with a cassette for `it`; returns how many replayed.
async fn each<F, Fut>(it: &str, body: F) -> usize
where
    F: Fn(Cassette, &'static str, &'static str) -> Fut,
    Fut: std::future::Future<Output = Result<Cassette, String>>,
{
    let mut failures = Vec::new();
    let mut ran = 0;
    for &(provider, model) in THINKING_MODELS {
        let name = cassette_name("chat with extended thinking", provider, model, it);
        let Some(cassette) = Cassette::start(&name).await else { continue };
        ran += 1;
        match body(cassette, provider, model).await {
            Ok(cassette) => {
                let r = futures::FutureExt::catch_unwind(std::panic::AssertUnwindSafe(cassette.assert_all_matched())).await;
                if let Err(p) = r {
                    failures.push(format!("{provider} {model}: {}", p.downcast_ref::<String>().cloned().unwrap_or_default()));
                }
            }
            Err(e) => failures.push(format!("{provider} {model}: {e}")),
        }
    }
    assert!(failures.is_empty(), "{} of {ran} failed:\n{}", failures.len(), failures.join("\n\n"));
    eprintln!("{it}: {ran} replayed");
    ran
}

#[tokio::test]
async fn returns_thinking_when_available() {
    let ran = each("returns thinking when available", |cassette, provider, model| async move {
        let mut chat = chat_with_thinking(&cassette, provider, model);
        let response = chat.ask(prompt_for(provider)).await.map_err(|e| e.to_string())?;
        expect_response_payload(&response)?;
        if provider == "openai" {
            check(response.tokens().thinking.unwrap_or(0) > 0, format!("thinking tokens {:?}", response.tokens()))?;
        } else {
            check(response.thinking.is_some(), "no thinking")?;
        }
        Ok(cassette)
    })
    .await;
    assert_eq!(ran, THINKING_MODELS.len());
}

#[tokio::test]
async fn streams_thinking_content_when_available() {
    let ran = each("streams thinking content when available", |cassette, provider, model| async move {
        let mut chat = chat_with_thinking(&cassette, provider, model);
        let mut chunks = 0;
        let mut thinking_chunks = 0;
        let response = chat
            .ask_stream(prompt_for(provider), |chunk| {
                chunks += 1;
                if chunk.thinking.is_some() {
                    thinking_chunks += 1;
                }
            })
            .await
            .map_err(|e| e.to_string())?;
        expect_response_payload(&response)?;
        check(chunks > 0, "no chunks")?;
        if response.thinking.is_some() {
            check(thinking_chunks > 0, "thinking arrived without a thinking chunk")?;
        }
        Ok(cassette)
    })
    .await;
    assert_eq!(ran, THINKING_MODELS.len());
}

#[tokio::test]
async fn preserves_thinking_signatures_between_turns_when_provided() {
    let ran = each("preserves thinking signatures between turns when provided", |cassette, provider, model| async move {
        let mut chat = chat_with_thinking(&cassette, provider, model);
        let first = chat.ask("What is 5 + 3?").await.map_err(|e| e.to_string())?;
        let signature = first.thinking.as_ref().and_then(|t| t.signature.clone());
        let second = chat.ask("Now multiply that by 2").await.map_err(|e| e.to_string())?;
        expect_response_payload(&second)?;
        if let Some(signature) = signature {
            check(second.thinking.as_ref().is_some_and(|t| t.signature.is_some()), "second turn has no signature")?;
            if matches!(provider, "anthropic" | "gemini") {
                let stored: Vec<&str> =
                    chat.messages().iter().filter_map(|m| m.thinking.as_ref()?.signature.as_deref()).collect();
                check(stored.contains(&signature.as_str()), "first signature not stored")?;
            }
        }
        Ok(cassette)
    })
    .await;
    assert_eq!(ran, THINKING_MODELS.len());
}

/// `thinking display with anthropic/claude-sonnet-5 returns readable thinking with display summarized`.
#[tokio::test]
async fn returns_readable_thinking_with_display_summarized() {
    let name = "chat_thinking_display_with_anthropic_claude-sonnet-5_returns_readable_thinking_with_display_summarized";
    let cassette = Cassette::start(name).await.expect("cassette");
    let mut chat = chat_for(&cassette, "anthropic", "claude-sonnet-5")
        .with_thinking(ThinkingConfig::effort("xhigh").with_display(ThinkingDisplay::Summarized));
    let response = chat
        .ask("A farmer has chickens and rabbits, 35 heads and 94 legs. How many of each? Reason step by step.")
        .await
        .unwrap();
    assert!(response.content().contains("23") && response.content().contains("12"), "{}", response.content());
    let thinking = response.thinking.expect("thinking");
    assert!(thinking.text.is_some_and(|t| !t.trim().is_empty()));
    assert!(thinking.signature.is_some_and(|s| !s.is_empty()));
    cassette.assert_all_matched().await;
}

// ---- #with_thinking payloads (chat_thinking_spec.rb) -------------------------------------------

/// `include_context 'with configured RubyLLM'`: every provider has a key, nothing is sent.
fn configured() -> std::sync::Arc<rust_llm::Config> {
    let mut config = rust_llm::Config::default();
    for provider in ["openai", "anthropic", "gemini", "mistral", "openrouter", "deepseek"] {
        config.set(format!("{provider}_api_key"), "test-key");
    }
    std::sync::Arc::new(config)
}

fn chat(model: &str, provider: &str) -> Chat {
    Chat::with_config(configured(), Some(model), Some(provider), false).expect("chat")
}

fn render(chat: &mut Chat) -> Value {
    chat.ask_later("Hello").expect("stage");
    chat.render().expect("render")
}

#[test]
fn uses_the_registered_model_controls_without_options() {
    let payload = render(&mut chat("gpt-5.2", "openai").with_thinking(ThinkingConfig::on()));
    assert_eq!(payload["reasoning"]["effort"], "medium");
}

#[test]
fn uses_the_new_model_controls_after_switching_models() {
    let mut chat = chat("gpt-5.2", "openai").with_thinking(ThinkingConfig::on()).with_model("claude-haiku-4-5", Some("anthropic")).unwrap();
    assert_eq!(render(&mut chat)["thinking"], json!({ "type": "enabled", "budget_tokens": 1024 }));
}

#[test]
fn uses_a_provider_toggle_before_inventing_a_token_budget() {
    let payload = render(&mut chat("gemini-2.5-flash", "gemini").with_thinking(ThinkingConfig::on()));
    assert_eq!(payload["generationConfig"]["thinkingConfig"]["thinkingBudget"], -1);
}

#[test]
fn uses_a_provider_toggle_before_choosing_an_effort() {
    let payload = render(&mut chat("claude-sonnet-5", "anthropic").with_thinking(ThinkingConfig::on()));
    assert_eq!(payload["thinking"], json!({ "type": "adaptive" }));
}

#[test]
fn sends_the_registered_off_control_with_false() {
    let payload = render(&mut chat("gpt-5.2", "openai").with_thinking(ThinkingConfig::off()));
    assert_eq!(payload["reasoning"]["effort"], "none");
}

#[test]
fn maps_false_to_a_zero_budget_when_the_model_uses_one_as_its_off_control() {
    let payload = render(&mut chat("gemini-2.5-flash", "gemini").with_thinking(ThinkingConfig::off()));
    assert_eq!(payload["generationConfig"]["thinkingConfig"]["thinkingBudget"], 0);
}

#[test]
fn does_not_add_controls_for_an_always_thinking_model() {
    let payload = render(&mut chat("magistral-small", "mistral").with_thinking(ThinkingConfig::on()));
    assert!(payload.get("thinking").is_none());
    assert!(payload.get("reasoning_effort").is_none());
}

#[test]
fn raises_when_the_registry_has_no_controls_for_the_model() {
    let mut chat = Chat::with_config(configured(), Some("private-reasoner"), Some("openai"), true)
        .unwrap()
        .with_thinking(ThinkingConfig::on());
    chat.ask_later("Hello").unwrap();
    let err = chat.render().unwrap_err();
    assert!(err.to_string().contains("does not know how to enable thinking"), "{err}");
}

#[test]
fn raises_when_the_registry_has_no_off_control_for_the_model() {
    let mut chat = chat("magistral-small", "mistral").with_thinking(ThinkingConfig::off());
    chat.ask_later("Hello").unwrap();
    let err = chat.render().unwrap_err();
    assert!(err.to_string().contains("does not know how to disable thinking"), "{err}");
}

#[test]
fn keeps_explicit_options_independent_of_registry_defaults() {
    let mut chat = Chat::with_config(configured(), Some("private-reasoner"), Some("openai"), true)
        .unwrap()
        .with_thinking(ThinkingConfig::effort("high"));
    assert_eq!(render(&mut chat)["reasoning"]["effort"], "high");
}

#[test]
fn renders_the_display_option_inside_the_anthropic_thinking_config() {
    let config = ThinkingConfig::effort("high").with_display(ThinkingDisplay::Summarized);
    let payload = render(&mut chat("claude-sonnet-5", "anthropic").with_thinking(config));
    assert_eq!(payload["thinking"], json!({ "type": "adaptive", "display": "summarized" }));
    assert_eq!(payload["output_config"]["effort"], "high");
}

#[test]
fn renders_display_alone_as_adaptive_thinking_with_the_default_effort() {
    let config = ThinkingConfig::default().with_display(ThinkingDisplay::Summarized);
    let payload = render(&mut chat("claude-sonnet-5", "anthropic").with_thinking(config));
    assert_eq!(payload["thinking"], json!({ "type": "adaptive", "display": "summarized" }));
    assert!(payload.get("output_config").is_none());
}

#[test]
fn passes_provider_specific_effort_tiers_through_untouched() {
    let payload = render(&mut chat("gpt-5.2", "openai").with_thinking(ThinkingConfig::effort("xhigh")));
    assert_eq!(payload["reasoning"]["effort"], "xhigh");
}

// ---- chat_thinking_replay_spec.rb ----------------------------------------------------------------

/// `produced_by(provider, model, thinking)`: an assistant message with a succeeded usage entry.
fn produced_by(provider: &str, model: &str, thinking: Option<Thinking>) -> Message {
    let mut message = Message::assistant("Done.");
    message.model = Some(model.into());
    message.thinking = thinking;
    message.usage_entries = vec![UsageEntry {
        id: UsageEntry::next_id(),
        operation: Operation::Chat,
        provider: provider.into(),
        model: model.into(),
        status: UsageStatus::Succeeded,
        tokens: Tokens::default(),
        cost: Cost::default(),
    }];
    message
}

/// `replay(chat, message)`: Hi, the message, "And now?", rendered.
fn replay(chat: &mut Chat, message: Message) -> Value {
    chat.add_message(Message::user("Hi"));
    chat.add_message(message);
    chat.add_message(Message::user("And now?"));
    chat.render().expect("render")
}

#[test]
fn drops_a_gemini_thought_signature_when_the_chat_moves_to_anthropic() {
    let message = produced_by("gemini", "gemini-2.5-flash", Thinking::build(None, Some("gemini-signature".into())));
    let mut chat = chat("gemini-2.5-flash", "gemini").with_model("claude-haiku-4-5", Some("anthropic")).unwrap();
    let payload = replay(&mut chat, message);
    assert_eq!(payload["messages"][1], json!({ "role": "assistant", "content": [{ "type": "text", "text": "Done." }] }));
}

#[test]
fn drops_anthropic_thinking_when_the_chat_moves_to_openai() {
    let thinking = Thinking::build(Some("Let me think.".into()), Some("anthropic-signature".into()));
    let message = produced_by("anthropic", "claude-haiku-4-5", thinking);
    let mut chat = chat("claude-haiku-4-5", "anthropic").with_model("gpt-5-nano", Some("openai")).unwrap();
    let payload = replay(&mut chat, message);
    let input = payload["input"].as_array().unwrap();
    assert!(!input.iter().any(|i| i["type"] == "reasoning"), "{payload}");
    assert_eq!(input[1]["role"], "assistant");
}

#[test]
fn drops_the_signature_of_a_model_the_registry_does_not_list() {
    let message = produced_by("gemini", "private-gemini", Thinking::build(None, Some("gemini-signature".into())));
    let mut chat = chat("claude-haiku-4-5", "anthropic");
    let payload = replay(&mut chat, message);
    assert_eq!(payload["messages"][1], json!({ "role": "assistant", "content": [{ "type": "text", "text": "Done." }] }));
}

#[test]
fn drops_gemini_tool_call_thought_signatures_when_the_chat_moves_to_a_chat_completions_provider() {
    let mut call = ToolCall::new("call-1", "lookup", Map::new());
    call.thought_signature = Some("gemini-signature".into());
    let mut message = produced_by("gemini", "gemini-2.5-flash", None);
    let mut calls = IndexMap::new();
    calls.insert("call-1".into(), call);
    message.tool_calls = Some(calls);
    let mut chat = chat("gemini-2.5-flash", "gemini");
    chat.add_message(Message::user("Hi"));
    chat.add_message(message);
    chat.add_message(Message::tool_result("call-1", "Found it."));

    let parts = chat.render().unwrap()["contents"][1]["parts"].clone();
    assert!(parts.as_array().unwrap().iter().any(|p| p["thoughtSignature"] == "gemini-signature"), "{parts}");
    let chat = chat.with_model("deepseek-v4-flash", Some("deepseek")).unwrap();
    let payload = chat.render().unwrap();
    assert!(payload["messages"][1]["tool_calls"][0].get("extra_content").is_none(), "{payload}");
    let kept = chat.messages()[1].tool_calls.as_ref().unwrap().get("call-1").unwrap();
    assert_eq!(kept.thought_signature.as_deref(), Some("gemini-signature"));
}

#[test]
fn keeps_thinking_for_the_provider_that_produced_it() {
    let thinking = Thinking::build(Some("Let me think.".into()), Some("anthropic-signature".into()));
    let message = produced_by("anthropic", "claude-haiku-4-5", thinking);
    let payload = replay(&mut chat("claude-haiku-4-5", "anthropic"), message);
    assert_eq!(
        payload["messages"][1]["content"][0],
        json!({ "type": "thinking", "thinking": "Let me think.", "signature": "anthropic-signature" })
    );
}

#[test]
fn keeps_thinking_whose_producer_is_unknown() {
    let mut message = Message::assistant("Done.");
    message.thinking = Thinking::build(None, Some("signature".into()));
    let payload = replay(&mut chat("claude-haiku-4-5", "anthropic"), message);
    assert_eq!(payload["messages"][1]["content"][0], json!({ "type": "redacted_thinking", "data": "signature" }));
}

#[test]
fn does_not_guess_the_producer_from_a_model_id_several_providers_serve() {
    let mut message = Message::assistant("Done.");
    message.model = Some("claude-haiku-4-5".into());
    message.thinking = Thinking::build(None, Some("signature".into()));
    let payload = replay(&mut chat("claude-haiku-4-5", "openrouter"), message);
    assert_eq!(payload["messages"][1]["reasoning_details"], json!([{ "type": "reasoning.encrypted", "data": "signature" }]));
}

#[test]
fn leaves_the_transcript_untouched() {
    let message = produced_by("gemini", "gemini-2.5-flash", Thinking::build(None, Some("gemini-signature".into())));
    let mut chat = chat("gemini-2.5-flash", "gemini").with_model("claude-haiku-4-5", Some("anthropic")).unwrap();
    replay(&mut chat, message);
    assert_eq!(chat.messages()[1].thinking.as_ref().unwrap().signature.as_deref(), Some("gemini-signature"));
    assert_eq!(chat.messages()[1].role, Role::Assistant);
}
