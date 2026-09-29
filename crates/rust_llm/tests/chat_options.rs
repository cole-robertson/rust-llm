//! Chat options ported from RubyLLM 2.0: `with_caching`, `with_compaction` + `compact`,
//! `with_end_user`, `with_context` / `RubyLLM.context`, and Anthropic `pause_turn` continuation.
//! Render-only examples mirror `chat_cache_until_here_spec.rb` (options part),
//! `chat_compaction_spec.rb`, `chat_compact_spec.rb`, `chat_end_user_spec.rb`, `context_spec.rb`,
//! and `context_entrypoints_spec.rb`; live examples replay `chat_anthropic_*compacts*`,
//! `chat_compacts_and_continues_*`, and `context_*` cassettes.

mod support;

use std::sync::Arc;

use rust_llm::{Caching, Chat, Config, EmbedOptions, Error, Message, ProtocolName, Vectors};
use serde_json::{Value, json};
use support::{Cassette, config_for};

/// A config where every provider these examples render for is configured; nothing is sent.
fn offline() -> Arc<Config> {
    let mut config = Config::default();
    for key in [
        "openai",
        "anthropic",
        "gemini",
        "deepseek",
        "openrouter",
        "xai",
        "mistral",
    ] {
        config.set(format!("{key}_api_key"), "test");
    }
    Arc::new(config)
}

fn chat(model: &str, provider: &str) -> Chat {
    Chat::with_config(offline(), Some(model), Some(provider), false).unwrap()
}

fn staged(mut chat: Chat) -> Chat {
    chat.ask_later("Hello").unwrap();
    chat
}

// ---- with_caching ----------------------------------------------------------------------------

#[test]
fn with_caching_stores_replaces_and_disables_options() {
    let chat = chat("gpt-4.1-nano", "openai")
        .with_caching(json!({ "key": "repo:ruby_llm", "retention": "24h" }))
        .unwrap();
    assert_eq!(
        chat.caching(),
        Some(&Caching::On(
            json!({ "key": "repo:ruby_llm", "retention": "24h" })
                .as_object()
                .unwrap()
                .clone()
        ))
    );
    let chat = chat.with_caching(json!({ "ttl": "1h" })).unwrap();
    assert_eq!(
        chat.caching(),
        Some(&Caching::On(
            json!({ "ttl": "1h" }).as_object().unwrap().clone()
        ))
    );
    assert_eq!(
        chat.with_caching(json!(true)).unwrap().caching(),
        Some(&Caching::On(Default::default()))
    );
    assert_eq!(
        self::chat("gpt-4.1-nano", "openai")
            .with_caching(json!(false))
            .unwrap()
            .caching(),
        Some(&Caching::Off)
    );
    let err = self::chat("gpt-4.1-nano", "openai")
        .with_caching(Value::Null)
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("accepts true, false, or caching options"),
        "{err}"
    );
}

#[test]
fn caching_options_survive_a_provider_switch_and_the_new_provider_rejects_what_it_cannot_render() {
    let chat = chat("gpt-4.1-nano", "openai")
        .with_caching(json!({ "retention": "24h" }))
        .unwrap();
    let chat = staged(chat.with_model("claude-haiku-4-5", None).unwrap());
    assert!(chat.caching().is_some());
    let err = chat.render().unwrap_err();
    assert!(
        matches!(&err, Error::Argument(m) if m.contains("Anthropic prompt caching accepts :ttl")),
        "{err}"
    );
}

#[test]
fn caching_false_omits_cache_controls_and_marked_boundaries() {
    let mut chat = chat("gpt-4.1-nano", "openai");
    chat.set_instructions(Some("Stable instructions".into()), false, false);
    chat.cache_until_here().unwrap();
    chat.ask_later("Hello").unwrap();
    chat.cache_until_here().unwrap();
    let payload = chat.with_caching(json!(false)).unwrap().render().unwrap();
    assert_eq!(payload["instructions"], "Stable instructions");
    assert_eq!(
        payload["input"].as_array().unwrap().last().unwrap()["content"],
        "Hello"
    );
    assert!(payload.get("prompt_cache_options").is_none());
}

#[test]
fn openai_renders_prompt_cache_params() {
    let chat = chat("gpt-4.1-nano", "openai")
        .with_caching(json!({ "key": "repo:ruby_llm", "ttl": "30m", "mode": "implicit" }))
        .unwrap();
    let payload = staged(chat).render().unwrap();
    assert_eq!(payload["prompt_cache_key"], "repo:ruby_llm");
    assert_eq!(
        payload["prompt_cache_options"],
        json!({ "mode": "implicit", "ttl": "30m" })
    );

    let chat = self::chat("gpt-4.1-nano", "openai")
        .with_caching(json!({ "key": "repo:ruby_llm", "retention": "24h" }))
        .unwrap();
    assert_eq!(
        staged(chat).render().unwrap()["prompt_cache_options"],
        json!({ "ttl": "24h" })
    );

    let chat = self::chat("gpt-4.1-nano", "openai")
        .with_caching(json!({ "scope": "user" }))
        .unwrap();
    let err = staged(chat).render().unwrap_err();
    assert!(
        err.to_string()
            .contains("Responses prompt caching accepts :key, :ttl, and :mode"),
        "{err}"
    );
    let chat = self::chat("gpt-4.1-nano", "openai")
        .with_caching(json!({ "id": "cachedContents/abc123" }))
        .unwrap();
    let err = staged(chat).render().unwrap_err();
    assert!(
        err.to_string()
            .contains("prompt caching accepts :key, :ttl, and :mode, got :id"),
        "{err}"
    );
}

#[test]
fn gemini_attaches_an_explicit_cache_by_id() {
    for (id, expected) in [
        ("cachedContents/abc123", "cachedContents/abc123"),
        ("abc123", "cachedContents/abc123"),
    ] {
        let chat = chat("gemini-2.5-flash", "gemini")
            .with_caching(json!({ "id": id }))
            .unwrap();
        assert_eq!(staged(chat).render().unwrap()["cachedContent"], expected);
    }
    let chat = chat("gemini-2.5-flash", "gemini")
        .with_caching(json!({ "ttl": "1h" }))
        .unwrap();
    assert!(
        staged(chat)
            .render()
            .unwrap()
            .get("cachedContent")
            .is_none()
    );
}

#[test]
fn anthropic_rejects_the_id_option_and_renders_ttl() {
    let chat = chat("claude-haiku-4-5", "anthropic")
        .with_caching(json!({ "id": "cachedContents/abc123" }))
        .unwrap();
    let err = staged(chat).render().unwrap_err();
    assert!(
        err.to_string()
            .contains("Anthropic prompt caching accepts :ttl, got :id"),
        "{err}"
    );

    let mut chat = self::chat("claude-haiku-4-5", "anthropic")
        .with_caching(json!({ "ttl": "1h" }))
        .unwrap();
    chat.set_instructions(Some("Stable".into()), false, true);
    let payload = staged(chat).render().unwrap();
    assert_eq!(
        payload["cache_control"],
        json!({ "type": "ephemeral", "ttl": "1h" })
    );
    assert_eq!(
        payload["system"][0]["cache_control"],
        json!({ "type": "ephemeral", "ttl": "1h" })
    );
}

/// The request half of `chat_prompt_cache_round-trip_*` (agent B replays the cassette): Anthropic
/// sends a top-level `cache_control` beside the marked system block, and OpenAI a shared key.
#[test]
fn prompt_cache_round_trip_requests_match_the_recording() {
    let mut chat = chat("claude-haiku-4-5", "anthropic")
        .with_caching(json!(true))
        .unwrap();
    chat.set_instructions(Some("Stable".into()), false, false);
    chat.cache_until_here().unwrap();
    let payload = staged(chat).render().unwrap();
    assert_eq!(payload["cache_control"], json!({ "type": "ephemeral" }));
    assert_eq!(
        payload["system"][0]["cache_control"],
        json!({ "type": "ephemeral" })
    );

    let chat = self::chat("gpt-5.2", "openai")
        .with_caching(json!({ "key": "rubyllm-test" }))
        .unwrap();
    let payload = staged(chat).render().unwrap();
    assert_eq!(payload["prompt_cache_key"], "rubyllm-test");
    assert!(payload.get("prompt_cache_options").is_none());
}

// ---- with_compaction -------------------------------------------------------------------------

fn compaction_payload(
    model: &str,
    provider: &str,
    protocol: Option<ProtocolName>,
    options: Value,
) -> Value {
    let mut chat = chat(model, provider).with_compaction(options).unwrap();
    if let Some(p) = protocol {
        chat = chat.with_protocol(p);
    }
    staged(chat).render().unwrap()
}

#[test]
fn with_compaction_remembers_normalizes_and_validates_options() {
    let chat = chat("claude-haiku-4-5", "anthropic")
        .with_compaction(json!({ "at": 50_000 }))
        .unwrap();
    assert_eq!(chat.compaction(), Some(&json!({ "at": 50_000 })));
    assert_eq!(
        chat.with_compaction(json!(false)).unwrap().compaction(),
        Some(&json!(false))
    );
    assert_eq!(
        self::chat("claude-haiku-4-5", "anthropic")
            .with_compaction(json!(true))
            .unwrap()
            .compaction(),
        Some(&json!({}))
    );
    let err = self::chat("claude-haiku-4-5", "anthropic")
        .with_compaction(Value::Null)
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("accepts true, false, or compaction options")
    );
    let err = self::chat("claude-haiku-4-5", "anthropic")
        .with_compaction(json!({ "compact_threshold": 50_000 }))
        .unwrap_err();
    assert!(
        err.to_string()
            .contains(":at, :instructions, :pause_after, got :compact_threshold"),
        "{err}"
    );
    assert!(
        staged(self::chat("claude-haiku-4-5", "anthropic"))
            .render()
            .unwrap()
            .get("context_management")
            .is_none()
    );
}

#[test]
fn compaction_maps_to_each_providers_parameter() {
    let anthropic = |options| compaction_payload("claude-haiku-4-5", "anthropic", None, options);
    assert_eq!(
        anthropic(json!({ "at": 50_000 }))["context_management"],
        json!({ "edits": [{ "type": "compact_20260112", "trigger": { "type": "input_tokens", "value": 50_000 } }] })
    );
    assert_eq!(
        anthropic(json!({}))["context_management"]["edits"],
        json!([{ "type": "compact_20260112" }])
    );
    let edit = &anthropic(
        json!({ "at": 50_000, "instructions": "Keep every decision.", "pause_after": true }),
    )["context_management"]["edits"][0];
    assert_eq!(edit["instructions"], "Keep every decision.");
    assert_eq!(edit["pause_after_compaction"], true);

    let openai = |options| {
        compaction_payload(
            "gpt-5-nano",
            "openai",
            Some(ProtocolName::Responses),
            options,
        )
    };
    assert_eq!(
        openai(json!({ "at": 200_000 }))["context_management"],
        json!([{ "type": "compaction", "compact_threshold": 200_000 }])
    );
    assert_eq!(
        openai(json!({}))["context_management"],
        json!([{ "type": "compaction" }])
    );

    assert_eq!(
        compaction_payload("claude-haiku-4-5", "openrouter", None, json!({}))["plugins"],
        json!([{ "id": "context-compression" }])
    );
    assert!(
        !compaction_payload("gemini-2.5-flash", "gemini", None, json!({ "at": 50_000 }))
            .to_string()
            .contains("compact")
    );
    let xai = compaction_payload(
        "grok-4-1-fast-non-reasoning",
        "xai",
        Some(ProtocolName::Responses),
        json!({ "at": 50_000 }),
    );
    assert!(xai.get("context_management").is_none());
    let cc = compaction_payload(
        "gpt-5-nano",
        "openai",
        Some(ProtocolName::ChatCompletions),
        json!({ "at": 50_000 }),
    );
    assert!(cc.get("context_management").is_none());

    let chat = chat("claude-haiku-4-5", "anthropic")
        .with_compaction(json!({ "at": 50_000 }))
        .unwrap()
        .with_provider_options(json!({ "context_management": { "edits": [] } }));
    assert_eq!(
        staged(chat).render().unwrap()["context_management"]["edits"],
        json!([])
    );
}

/// `chat_anthropic_claude-sonnet-4-6_compacts_a_long_conversation_and_bills_the_summarization_pass`:
/// the beta header goes out, and the summarization pass is billed through `usage.iterations`.
#[tokio::test]
async fn anthropic_compacts_a_long_conversation_and_bills_the_summarization_pass() {
    let name = "chat_anthropic_claude-sonnet-4-6_compacts_a_long_conversation_and_bills_the_summarization_pass";
    let cassette = Cassette::start(name)
        .await
        .expect("run bin/convert-cassettes 'chat_anthropic_*compacts*'");
    let config = config_for(&cassette, "anthropic");
    let mut chat = Chat::with_config(config, Some("claude-sonnet-4-6"), Some("anthropic"), false)
        .unwrap()
        .with_compaction(json!({ "at": 50_000 }))
        .unwrap();
    let notes = "The quick brown fox jumps over the lazy dog. ".repeat(9000);
    let response = chat
        .ask(format!(
            "Here are my notes:\n{notes}\nIn one sentence, which animal jumps in my notes?"
        ))
        .await
        .unwrap();

    let compaction = response
        .server_tool_calls
        .iter()
        .find(|c| c.kind == "compaction")
        .expect("a compaction block");
    assert!(
        compaction
            .result
            .as_ref()
            .and_then(Value::as_str)
            .is_some_and(|r| !r.is_empty())
    );
    assert!(response.content().contains("fox"));
    assert_eq!(
        response.raw_content.as_ref().unwrap()[0]["type"],
        "compaction"
    );
    assert!(response.tokens().input.unwrap() > 50_000);
    let reported = response
        .raw
        .as_ref()
        .unwrap()
        .body
        .pointer("/usage/input_tokens")
        .and_then(Value::as_i64)
        .unwrap();
    assert!(reported < response.tokens().input.unwrap());
    let requests = cassette.server.received_requests().await.unwrap();
    let beta = requests[0]
        .headers
        .get("anthropic-beta")
        .and_then(|v| v.to_str().ok());
    assert_eq!(beta, Some("compact-2026-01-12"));
    cassette.assert_all_matched().await;
}

// ---- compact ---------------------------------------------------------------------------------

/// `chat_compacts_and_continues_a_conversation_with_{openai,xai}`: the compact request carries
/// the history and instructions; the next request replays the compacted output in its place.
#[tokio::test]
async fn compacts_and_continues_a_conversation() {
    for (provider, model) in [("openai", "gpt-5-nano"), ("xai", "grok-4.3")] {
        let name = format!("chat_compacts_and_continues_a_conversation_with_{provider}");
        let cassette = Cassette::start(&name)
            .await
            .expect("run bin/convert-cassettes 'chat_compacts_*'");
        let mut chat = Chat::with_config(
            config_for(&cassette, provider),
            Some(model),
            Some(provider),
            false,
        )
        .unwrap()
        .with_protocol(ProtocolName::Responses)
        .with_instructions("Answer in one short sentence.");
        chat.ask_later("The project codename is Thimble. We write it in Ruby.")
            .unwrap();
        chat.add_message(Message::assistant(
            "I will remember the Thimble project and Ruby.",
        ));
        let history = chat.messages().to_vec();

        let result = chat.compact().await.unwrap();
        assert_eq!(
            result.raw_content.as_ref().unwrap()["object"],
            "response.compaction",
            "{provider}"
        );
        assert!(result.tokens().input.unwrap() > 0, "{provider}");
        assert_eq!(chat.messages().len(), history.len() + 1);
        assert_eq!(chat.messages()[..history.len()], history[..]);

        let answer = chat.ask("What is the project codename?").await.unwrap();
        assert!(
            answer.content().to_lowercase().contains("thimble"),
            "{provider}: {}",
            answer.content()
        );
        assert!(answer.tokens().input.unwrap() > 0);
        cassette.assert_all_matched().await;
    }
}

/// chat_compact_spec.rb: history is kept, the wire context is replaced, usage and callbacks are
/// recorded, and a malformed response adds nothing.
#[tokio::test]
async fn compact_replaces_the_wire_context_and_records_usage_and_callbacks() {
    use std::sync::Mutex;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let output =
        json!([{ "type": "compaction", "id": "cmp_1", "encrypted_content": "opaque context" }]);
    let body = json!({ "id": "cmp_1", "object": "response.compaction", "output": output, "usage": { "input_tokens": 23, "output_tokens": 7 } });
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses/compact"))
        .and(header("X-Trace", "trace"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body.clone()))
        .expect(1)
        .mount(&server)
        .await;
    let context = rust_llm::context(|c| {
        c.set("xai_api_key", "test");
        c.set("xai_api_base", format!("{}/v1", server.uri()));
    });
    let events = Arc::new(Mutex::new(Vec::<String>::new()));
    let (before, after) = (events.clone(), events.clone());
    let mut chat = context
        .chat(Some("grok-4.3"), Some("xai"))
        .unwrap()
        .with_instructions("Remember the project.")
        .with_headers([("X-Trace".to_string(), "trace".to_string())])
        .with_temperature(0.3)
        .before_message(move || before.lock().unwrap().push("before".into()))
        .after_message(move |m| after.lock().unwrap().push(format!("after:{}", m.content())))
        .before_request(|payload| payload["instructions"] = "Keep the names.".into());
    chat.add_message(Message::user("The project is Thimble."));
    chat.add_message(Message::assistant("I will remember Thimble."));
    let history = chat.messages().to_vec();

    let result = chat.compact().await.unwrap();
    assert_eq!(result.content(), "");
    assert_eq!(result.finish_reason, Some(rust_llm::FinishReason::Stop));
    assert_eq!(result.raw_content, Some(body));
    assert_eq!(chat.messages().len(), history.len() + 1);
    assert_eq!(
        *events.lock().unwrap(),
        vec!["before".to_string(), "after:".to_string()]
    );
    assert_eq!(
        (result.tokens().input, result.tokens().output),
        (Some(23), Some(7))
    );
    assert_eq!(
        (chat.tokens().input, chat.tokens().output),
        (Some(23), Some(7))
    );
    let sent: Value =
        serde_json::from_slice(&server.received_requests().await.unwrap()[0].body).unwrap();
    assert_eq!(sent["instructions"], "Keep the names.");
    assert!(sent.get("temperature").is_none() && sent.get("tools").is_none());

    // `render` runs the before_request hooks too, as `Chat#render` does.
    let rendered = chat.render().unwrap();
    assert_eq!(rendered["input"], output);
    assert_eq!(rendered["instructions"], "Keep the names.");
}

#[tokio::test]
async fn compact_refuses_pending_tool_calls_and_unsupported_protocols() {
    let mut chat = chat("grok-4.3", "xai");
    let call = rust_llm::ToolCall::new("call_1", "search", Default::default());
    let mut pending = Message::assistant("");
    pending.tool_calls = Some([("call_1".to_string(), call)].into_iter().collect());
    chat.set_messages(vec![pending]);
    assert!(matches!(
        chat.compact().await.unwrap_err(),
        Error::PendingToolCalls(_)
    ));

    let mut chat =
        self::chat("deepseek-v4-flash", "deepseek").with_protocol(ProtocolName::Responses);
    let err = chat.compact().await.unwrap_err();
    assert!(
        err.to_string()
            .contains("doesn't support manual compaction"),
        "{err}"
    );
}

#[tokio::test]
async fn compact_rejects_a_malformed_response_without_adding_a_message() {
    use wiremock::matchers::method;
    use wiremock::{Mock, MockServer, ResponseTemplate};
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(json!({ "object": "response", "output": [] })),
        )
        .mount(&server)
        .await;
    let context = rust_llm::context(|c| {
        c.set("xai_api_key", "test");
        c.set("xai_api_base", format!("{}/v1", server.uri()));
    });
    let mut chat = context.chat(Some("grok-4.3"), Some("xai")).unwrap();
    let err = chat.compact().await.unwrap_err();
    assert!(
        err.to_string().contains("invalid compaction response"),
        "{err}"
    );
    assert!(chat.messages().is_empty());
}

// ---- with_end_user ---------------------------------------------------------------------------

fn end_user_payload(model: &str, provider: &str, protocol: Option<ProtocolName>) -> Value {
    let mut chat = chat(model, provider).with_end_user(Some("user-123"));
    if let Some(p) = protocol {
        chat = chat.with_protocol(p);
    }
    staged(chat).render().unwrap()
}

#[test]
fn with_end_user_remembers_and_clears_the_identifier() {
    let chat = chat("gpt-4.1-nano", "openai").with_end_user(Some("user-123"));
    assert_eq!(chat.end_user(), Some("user-123"));
    assert_eq!(chat.with_end_user(None).end_user(), None);
    assert!(
        staged(self::chat("gpt-4.1-nano", "openai"))
            .render()
            .unwrap()
            .get("safety_identifier")
            .is_none()
    );
}

#[test]
fn end_user_maps_to_each_providers_field() {
    assert_eq!(
        end_user_payload("gpt-4.1-nano", "openai", Some(ProtocolName::Responses))["safety_identifier"],
        "user-123"
    );
    assert_eq!(
        end_user_payload(
            "gpt-4.1-nano",
            "openai",
            Some(ProtocolName::ChatCompletions)
        )["safety_identifier"],
        "user-123"
    );
    assert_eq!(
        end_user_payload("claude-haiku-4-5", "anthropic", None)["metadata"]["user_id"],
        "user-123"
    );
    assert_eq!(
        end_user_payload("deepseek-v4-flash", "deepseek", None)["user_id"],
        "user-123"
    );
    assert_eq!(
        end_user_payload("claude-haiku-4-5", "openrouter", None)["user"],
        "user-123"
    );
    assert!(
        !end_user_payload("gemini-2.5-flash", "gemini", None)
            .to_string()
            .contains("user-123")
    );

    let chat = chat("gpt-4.1-nano", "openai")
        .with_end_user(Some("user-123"))
        .with_provider_options(json!({ "safety_identifier": "override" }));
    assert_eq!(
        staged(chat).render().unwrap()["safety_identifier"],
        "override"
    );
}

// ---- pause_turn ------------------------------------------------------------------------------

/// A server tool pauses the turn with `pause_turn`; the conversation goes back verbatim and the
/// segments come back as one message (chat_provider_tools_spec.rb "pause_turn merging").
#[tokio::test]
async fn anthropic_continues_a_paused_turn_and_merges_the_segments() {
    use wiremock::matchers::method;
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let paused = json!({
        "model": "claude-haiku-4-5-20251001", "role": "assistant", "stop_reason": "pause_turn",
        "content": [
            { "type": "text", "text": "Searching. " },
            { "type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": { "query": "ruby" } }
        ],
        "usage": { "input_tokens": 10, "output_tokens": 5 }
    });
    let done = json!({
        "model": "claude-haiku-4-5-20251001", "role": "assistant", "stop_reason": "end_turn",
        "content": [{ "type": "text", "text": "Done." }],
        "usage": { "input_tokens": 20, "output_tokens": 7 }
    });
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(paused))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(done))
        .mount(&server)
        .await;
    let context = rust_llm::context(|c| {
        c.set("anthropic_api_key", "test");
        c.set("anthropic_api_base", server.uri());
    });
    let mut chat = context
        .chat(Some("claude-haiku-4-5"), Some("anthropic"))
        .unwrap();
    let merged = chat.ask("Search the web for ruby").await.unwrap();

    assert_eq!(merged.content(), "Searching. Done.");
    assert_eq!(merged.finish_reason, Some(rust_llm::FinishReason::Stop));
    assert_eq!(
        merged
            .raw_content
            .as_ref()
            .unwrap()
            .as_array()
            .unwrap()
            .len(),
        3
    );
    assert_eq!(merged.server_tool_calls.len(), 1);
    assert_eq!(
        (merged.tokens().input, merged.tokens().output),
        (Some(30), Some(12))
    );
    assert_eq!(merged.usage_entries.len(), 2, "each request is billed");
    assert_eq!(
        chat.messages().len(),
        2,
        "one merged answer joins the history"
    );

    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 2);
    let second: Value = serde_json::from_slice(&requests[1].body).unwrap();
    let messages = second["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[1]["role"], "assistant");
    assert_eq!(messages[1]["content"][1]["type"], "server_tool_use");
}

// ---- contexts --------------------------------------------------------------------------------

#[test]
fn a_context_copies_the_global_configuration_and_leaves_it_alone() {
    let original = rust_llm::config();
    let context = rust_llm::context(|c| {
        c.default_model = "modified-model".into();
        c.set("openai_api_key", "modified-key");
    });
    assert_eq!(rust_llm::config().default_model, original.default_model);
    assert_eq!(
        rust_llm::config().get("openai_api_key"),
        original.get("openai_api_key")
    );
    assert_eq!(context.config().default_model, "modified-model");
    assert_eq!(context.config().get("openai_api_key"), Some("modified-key"));
}

#[test]
fn context_chats_use_the_context_default_model_and_can_return_to_the_global_config() {
    let context = rust_llm::context(|c| {
        c.default_model = "claude-haiku-4-5".into();
        c.set("anthropic_api_key", "context-key");
    });
    let chat = context.chat(None, None).unwrap();
    assert_eq!(chat.model().provider, "anthropic");
    assert!(Arc::ptr_eq(chat.config(), context.config()));
    // "allows specifying a model when creating the chat".
    assert_eq!(
        context
            .chat(Some("claude-sonnet-4-6"), None)
            .unwrap()
            .model()
            .id,
        "claude-sonnet-4-6"
    );

    // "returns a chat to the global configuration with with_context(nil)"; the global config must
    // be able to serve the chat's provider, which the environment may not provide here.
    let global = rust_llm::config();
    match context.chat(None, None).unwrap().with_context(None) {
        Ok(chat) => assert!(Arc::ptr_eq(chat.config(), &global)),
        Err(e) => assert!(
            matches!(e, Error::Configuration(_)) && global.get("anthropic_api_key").is_none(),
            "{e}"
        ),
    }
    let other = rust_llm::context(|c| {
        c.set("anthropic_api_key", "other-key");
    });
    let chat = context
        .chat(None, None)
        .unwrap()
        .with_context(Some(&other))
        .unwrap();
    assert!(Arc::ptr_eq(chat.config(), other.config()));
}

#[test]
fn contexts_are_independent() {
    let one = rust_llm::context(|c| {
        c.set("openai_api_key", "key1");
        c.default_model = "model1".into();
    });
    let two = rust_llm::context(|c| {
        c.set("openai_api_key", "key2");
        c.default_model = "model2".into();
    });
    assert_eq!(one.config().get("openai_api_key"), Some("key1"));
    assert_eq!(two.config().get("openai_api_key"), Some("key2"));
    assert_eq!(two.config().default_model, "model2");
}

/// `context_context_chat_operations_uses_context-specific_api_keys`: the context's key goes out
/// and the provider's 401 comes back as Unauthorized.
#[tokio::test]
async fn context_chats_use_context_specific_api_keys() {
    let cassette =
        Cassette::start("context_context_chat_operations_uses_context-specific_api_keys")
            .await
            .expect("run bin/convert-cassettes 'context_*'");
    let context = rust_llm::Context::new((*config_for(&cassette, "openai")).clone());
    let mut chat = context.chat(Some("gpt-4.1-nano"), None).unwrap();
    let err = chat.ask("Hello").await.unwrap_err();
    assert!(matches!(err, Error::Unauthorized(..)), "{err}");
    cassette.assert_all_matched().await;
}

/// `context_context_embed_operations_*`: the context's default embedding model, and a model given
/// at embed time.
#[tokio::test]
async fn context_embeddings_use_the_context_default_model() {
    let cassette = Cassette::start(
        "context_context_embed_operations_respects_context-specific_embedding_model",
    )
    .await
    .unwrap();
    let mut config = (*config_for(&cassette, "openai")).clone();
    config.default_embedding_model = "text-embedding-3-large".into();
    let context = rust_llm::Context::new(config.clone());
    let embedding = context
        .embed("Test embedding", EmbedOptions::default())
        .await
        .unwrap();
    assert_eq!(embedding.model, "text-embedding-3-large");
    assert!(matches!(embedding.vectors, Vectors::Single(_)));
    // The Ruby spec then embeds with the global default (the second recorded request); here the
    // replay server's config with the stock default stands in for the global one.
    let mut stock = config;
    stock.default_embedding_model = Config::default().default_embedding_model;
    let global = rust_llm::embed(
        "Test embedding",
        EmbedOptions {
            config: Some(Arc::new(stock)),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(global.model, "text-embedding-3-small");
    cassette.assert_all_matched().await;

    let cassette =
        Cassette::start("context_context_embed_operations_allows_specifying_a_model_at_embed_time")
            .await
            .unwrap();
    let mut config = (*config_for(&cassette, "openai")).clone();
    config.default_embedding_model = "text-embedding-3-large".into();
    let context = rust_llm::Context::new(config);
    let embedding = context
        .embed(
            "Test embedding",
            EmbedOptions {
                model: Some("text-embedding-3-small"),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(embedding.model, "text-embedding-3-small");
    cassette.assert_all_matched().await;
}

/// context_entrypoints_spec.rb: OCR and rerank run with the context's configuration.
#[tokio::test]
async fn context_entry_points_use_the_context_configuration() {
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/ocr"))
        .and(header("Authorization", "Bearer tenant-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(
            json!({ "pages": [{ "index": 0, "markdown": "# Hi" }], "model": "mistral-ocr-latest" }),
        ))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/v1/rerank"))
        .and(header("Authorization", "Bearer tenant-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "results": [{ "index": 1, "relevance_score": 0.9 }, { "index": 0, "relevance_score": 0.1 }] })))
        .expect(1)
        .mount(&server)
        .await;
    let context = rust_llm::context(|c| {
        c.set("mistral_api_key", "tenant-key");
        c.set("mistral_api_base", format!("{}/v1", server.uri()));
        c.set("openrouter_api_key", "tenant-key");
        c.set("openrouter_api_base", format!("{}/api/v1", server.uri()));
    });
    let ocr = context
        .ocr(
            rust_llm::Attachment::new("https://example.com/report.pdf"),
            Default::default(),
        )
        .await
        .unwrap();
    assert_eq!(ocr.markdown(), "# Hi");
    let options = rust_llm::RerankOptions {
        provider: Some("openrouter"),
        assume_model_exists: true,
        ..Default::default()
    };
    let reranked = context
        .rerank(
            "query",
            &["first", "second"],
            "voyageai/rerank-2.5-lite",
            options,
        )
        .await
        .unwrap();
    assert_eq!(reranked.results[0].document, "second");
}
