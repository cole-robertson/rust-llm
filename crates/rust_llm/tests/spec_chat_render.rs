//! Chat feature specs ported from RubyLLM 2.0, mostly render-only: `chat_provider_tools_spec.rb`,
//! `chat_cache_until_here_spec.rb`, `chat_compaction_spec.rb` (request headers),
//! `chat_compact_spec.rb`, `chat_thinking_spec.rb`, `chat_tool_attachments_spec.rb`,
//! `chat_schema_spec.rb`, `chat_error_spec.rb`, `chat_tools_spec.rb` (stubbed-provider examples),
//! `chat_streaming_spec.rb` (error chunks), `chat_content_spec.rb`, and `chat_pricing_spec.rb`.
//! `// spec:` lines tie each test to its Ruby example.

mod spec_helpers;

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use rust_llm::message::indexmap_lite::IndexMap;
use rust_llm::{
    Attachment, Caching, CancelHandle, Chat, Config, Error, ErrorKind, Message, Parameter,
    ProtocolName, ProviderTool, Role, Tool, ToolCall, ToolError, ToolResult,
};
use serde_json::{Map, Value, json};
use spec_helpers::{Log, log, requests, serve, text_response, tool_use_response};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

/// Every provider this file renders for, pointed at `server` (or nowhere when rendering only).
fn config_at(base: &str) -> Arc<Config> {
    let mut c = Config::default();
    for (provider, path) in [
        ("anthropic", ""),
        ("openai", "/v1"),
        ("deepseek", ""),
        ("gemini", "/v1beta"),
        ("openrouter", "/api/v1"),
        ("xai", "/v1"),
        ("mistral", "/v1"),
        ("perplexity", ""),
        ("ollama", "/v1"),
        ("ollama_cloud", "/v1"),
        ("gpustack", "/v1"),
        ("hetzner", "/api/v1"),
    ] {
        c.set(format!("{provider}_api_base"), format!("{base}{path}"));
        c.set(format!("{provider}_api_key"), "test");
    }
    c.max_retries = 0;
    Arc::new(c)
}

fn offline() -> Arc<Config> {
    config_at("http://127.0.0.1:9")
}

/// `RubyLLM.chat(model:, provider:)`; local and self-hosted providers assume the model exists.
fn chat_with(config: Arc<Config>, model: &str, provider: &str) -> Chat {
    let assume = matches!(provider, "ollama" | "gpustack" | "ollama_cloud" | "hetzner");
    Chat::with_config(config, Some(model), Some(provider), assume).unwrap()
}

fn chat(model: &str, provider: &str) -> Chat {
    chat_with(offline(), model, provider)
}

fn fixture(name: &str) -> String {
    format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
}

/// A fixture attachment with its bytes read, as `Attachment.new(path)` reads them lazily.
async fn loaded(name: &str) -> Attachment {
    let mut a = Attachment::new(fixture(name));
    a.content().await.unwrap();
    a
}

fn tools_of(payload: &Value) -> Vec<Value> {
    payload["tools"].as_array().cloned().unwrap_or_default()
}

/// Collects `tracing` DEBUG events on this thread, standing in for `RubyLLM.logger.debug`.
struct DebugCollector(Arc<Mutex<Vec<String>>>);

impl tracing::Subscriber for DebugCollector {
    // Tests run in parallel: a callsite first hit with no collector set is cached as "never",
    // so ask on every event instead of caching the interest.
    fn register_callsite(
        &self,
        _: &'static tracing::Metadata<'static>,
    ) -> tracing::subscriber::Interest {
        tracing::subscriber::Interest::sometimes()
    }
    // Without this the global max level is recomputed from other threads' (absent) collectors and
    // can drop WARN events before they reach this one.
    fn max_level_hint(&self) -> Option<tracing::level_filters::LevelFilter> {
        Some(tracing::level_filters::LevelFilter::TRACE)
    }
    fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        struct Text<'a>(&'a mut String);
        impl tracing::field::Visit for Text<'_> {
            fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                if field.name() == "message" {
                    self.0.push_str(&format!("{value:?}"));
                }
            }
        }
        if *event.metadata().level() == tracing::Level::DEBUG {
            let mut text = String::new();
            event.record(&mut Text(&mut text));
            self.0.lock().unwrap().push(text);
        }
    }
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}

fn debug_logs_of(f: impl FnOnce()) -> Vec<String> {
    let logs = Arc::new(Mutex::new(Vec::new()));
    {
        let _guard =
            tracing::dispatcher::set_default(&tracing::Dispatch::new(DebugCollector(logs.clone())));
        f();
    }
    logs.lock().unwrap().clone()
}

/// `/implicit caching.*RubyLLM\.cache/m`.
fn implicit_caching_notes(logs: &[String]) -> usize {
    logs.iter()
        .filter(|l| {
            l.find("implicit caching")
                .is_some_and(|i| l[i..].contains("RubyLLM.cache"))
        })
        .count()
}

// ---- chat_provider_tools_spec.rb ---------------------------------------------------------------

// spec: chat_provider_tools_spec.rb:19 accumulates across calls and clears with nil
#[test]
fn provider_tools_accumulate_across_calls_and_clear() {
    let mut chat = chat("claude-haiku-4-5", "anthropic")
        .with_provider_tools(["web_search".into()])
        .with_provider_tools(["code_execution".into()]);
    assert_eq!(chat.provider_tools().len(), 2);
    chat.clear_provider_tools();
    assert!(chat.provider_tools().is_empty());
}

// spec: chat_provider_tools_spec.rb:35 renders Anthropic aliases into versioned tool entries
#[test]
fn renders_anthropic_aliases_into_versioned_tool_entries() {
    let payload = chat("claude-haiku-4-5", "anthropic")
        .with_provider_tools([
            ProviderTool::alias("web_search"),
            ProviderTool::with_options("web_fetch", json!({ "max_uses": 2 })),
        ])
        .render()
        .unwrap();
    let tools = tools_of(&payload);
    assert!(tools.contains(&json!({ "type": "web_search_20260318", "name": "web_search", "allowed_callers": ["direct"] })), "{tools:?}");
    assert!(
        tools.contains(&json!({ "type": "web_fetch_20260318", "name": "web_fetch", "allowed_callers": ["direct"], "max_uses": 2 })),
        "{tools:?}"
    );
}

// spec: chat_provider_tools_spec.rb:130 renders OpenAI Responses aliases
#[test]
fn renders_openai_responses_aliases() {
    let payload = chat("gpt-5.2", "openai")
        .with_provider_tools(["web_search".into(), "code_execution".into()])
        .render()
        .unwrap();
    let tools = tools_of(&payload);
    assert!(
        tools.contains(&json!({ "type": "web_search" })),
        "{tools:?}"
    );
    assert!(
        tools.contains(&json!({ "type": "code_interpreter", "container": { "type": "auto" } })),
        "{tools:?}"
    );
}

// spec: chat_provider_tools_spec.rb:139 renders Gemini aliases with options nested inside the tool key
#[test]
fn renders_gemini_aliases_with_options_nested_inside_the_tool_key() {
    let payload = chat("gemini-3.5-flash", "gemini")
        .with_provider_tools([
            ProviderTool::alias("web_search"),
            ProviderTool::with_options(
                "file_search",
                json!({ "file_search_store_names": ["store"] }),
            ),
        ])
        .render()
        .unwrap();
    let tools = tools_of(&payload);
    assert!(tools.contains(&json!({ "google_search": {} })), "{tools:?}");
    assert!(
        tools.contains(&json!({ "file_search": { "file_search_store_names": ["store"] } })),
        "{tools:?}"
    );
}

// spec: chat_provider_tools_spec.rb:148 renders xAI Responses aliases with passthrough options
#[test]
fn renders_xai_responses_aliases_with_passthrough_options() {
    let payload = chat("grok-4.3", "xai")
        .with_provider_tools([
            ProviderTool::alias("x_search"),
            ProviderTool::alias("code_execution"),
            ProviderTool::with_options(
                "web_search",
                json!({ "filters": { "allowed_domains": ["ruby-lang.org"] } }),
            ),
        ])
        .render()
        .unwrap();
    assert!(payload["input"].is_array());
    let tools = tools_of(&payload);
    assert!(tools.contains(&json!({ "type": "x_search" })), "{tools:?}");
    assert!(
        tools.contains(&json!({ "type": "code_execution" })),
        "{tools:?}"
    );
    assert!(
        tools.contains(
            &json!({ "type": "web_search", "filters": { "allowed_domains": ["ruby-lang.org"] } })
        ),
        "{tools:?}"
    );
}

// spec: chat_provider_tools_spec.rb:160 renders the xAI MCP alias with server options
#[test]
fn renders_the_xai_mcp_alias_with_server_options() {
    let payload = chat("grok-4.3", "xai")
        .with_provider_tools([ProviderTool::with_options(
            "mcp",
            json!({ "server_url": "https://mcp.example.com/mcp", "server_label": "example" }),
        )])
        .render()
        .unwrap();
    let tools = tools_of(&payload);
    assert!(
        tools.contains(&json!({ "type": "mcp", "server_url": "https://mcp.example.com/mcp", "server_label": "example" })),
        "{tools:?}"
    );
}

// ---- chat_cache_until_here_spec.rb -------------------------------------------------------------

// spec: chat_cache_until_here_spec.rb:27 keeps caching options when switching models on the same provider
#[test]
fn keeps_caching_options_when_switching_models_on_the_same_provider() {
    let chat = chat("gpt-4.1-nano", "openai")
        .with_caching(json!({ "retention": "24h" }))
        .unwrap();
    let chat = chat.with_model("gpt-5-nano", None).unwrap();
    assert_eq!(chat.provider().slug(), "openai");
    assert_eq!(
        chat.caching(),
        Some(&Caching::On(
            json!({ "retention": "24h" }).as_object().unwrap().clone()
        ))
    );
}

// spec: chat_cache_until_here_spec.rb:114 renders explicit breakpoints for OpenAI cache boundaries
#[test]
fn renders_explicit_breakpoints_for_openai_cache_boundaries() {
    let mut chat = chat("gpt-4.1-nano", "openai");
    chat.ask_later("Long context")
        .unwrap()
        .cache_until_here()
        .unwrap();
    let payload = chat.render().unwrap();
    assert_eq!(
        payload["input"].as_array().unwrap().last().unwrap()["content"],
        json!([{ "type": "input_text", "text": "Long context", "prompt_cache_breakpoint": { "mode": "explicit" } }])
    );
    assert!(payload.get("prompt_cache_options").is_none());
}

// spec: chat_cache_until_here_spec.rb:126 sends cache-bounded instructions as input items on Responses
#[test]
fn sends_cache_bounded_instructions_as_input_items_on_responses() {
    let mut chat = chat("gpt-4.1-nano", "openai");
    chat.set_instructions(Some("Stable instructions".into()), false, false)
        .cache_until_here()
        .unwrap();
    chat.ask_later("Hello").unwrap();
    let payload = chat.render().unwrap();
    assert!(payload.get("instructions").is_none_or(Value::is_null));
    assert_eq!(
        payload["input"][0],
        json!({
            "role": "system",
            "content": [{ "type": "input_text", "text": "Stable instructions", "prompt_cache_breakpoint": { "mode": "explicit" } }]
        })
    );
}

// spec: chat_cache_until_here_spec.rb:167 notes that Gemini caching is implicit when with_caching has no id
#[test]
fn notes_that_gemini_caching_is_implicit_when_with_caching_has_no_id() {
    let mut chat = chat("gemini-2.5-flash", "gemini")
        .with_caching(json!({ "ttl": "1h" }))
        .unwrap();
    chat.ask_later("Hello").unwrap();
    let mut payload = Value::Null;
    let logs = debug_logs_of(|| payload = chat.render().unwrap());
    assert!(payload.get("cachedContent").is_none());
    assert_eq!(implicit_caching_notes(&logs), 1, "{logs:?}");
}

// spec: chat_cache_until_here_spec.rb:179 notes that Gemini ignores explicit cache boundaries
#[test]
fn notes_that_gemini_ignores_explicit_cache_boundaries() {
    let mut chat = chat("gemini-2.5-flash", "gemini");
    chat.ask_later("Long context")
        .unwrap()
        .cache_until_here()
        .unwrap();
    let logs = debug_logs_of(|| {
        chat.render().unwrap();
    });
    assert_eq!(implicit_caching_notes(&logs), 1, "{logs:?}");
}

/// The note stays quiet for an explicit cache id and without caching or boundaries
/// (`maybe_log_implicit_caching_note`'s guards).
#[test]
fn gemini_implicit_caching_note_is_quiet_otherwise() {
    let mut explicit = chat("gemini-2.5-flash", "gemini")
        .with_caching(json!({ "id": "abc123" }))
        .unwrap();
    explicit.ask_later("Hello").unwrap();
    let mut plain = chat("gemini-2.5-flash", "gemini");
    plain.ask_later("Hello").unwrap();
    let mut off = chat("gemini-2.5-flash", "gemini");
    off.ask_later("Hello").unwrap().cache_until_here().unwrap();
    let off = off.with_caching(json!(false)).unwrap();
    for chat in [explicit, plain, off] {
        let logs = debug_logs_of(|| {
            chat.render().unwrap();
        });
        assert_eq!(implicit_caching_notes(&logs), 0, "{logs:?}");
    }
}

// spec: chat_cache_until_here_spec.rb:216 marks the staged user message from ask_later
#[test]
fn cache_until_here_marks_the_staged_user_message_from_ask_later() {
    let mut chat = chat("claude-haiku-4-5", "anthropic");
    chat.ask_later("Long context")
        .unwrap()
        .cache_until_here()
        .unwrap();
    assert!(chat.messages().last().unwrap().cache_until_here);
}

// spec: chat_cache_until_here_spec.rb:222 marks the instruction added by with_instructions
#[test]
fn cache_until_here_marks_the_instruction_added_by_with_instructions() {
    let mut chat = chat("claude-haiku-4-5", "anthropic");
    chat.add_message(Message::user("Existing message"));
    chat.set_instructions(Some("Stable instructions".into()), false, false)
        .cache_until_here()
        .unwrap();
    let system = chat
        .messages()
        .iter()
        .find(|m| m.role == Role::System)
        .unwrap();
    let user = chat
        .messages()
        .iter()
        .find(|m| m.role == Role::User)
        .unwrap();
    assert!(system.cache_until_here);
    assert!(!user.cache_until_here);
}

// spec: chat_cache_until_here_spec.rb:232 raises when the chat has no messages
#[test]
fn cache_until_here_raises_when_the_chat_has_no_messages() {
    let mut chat = chat("claude-haiku-4-5", "anthropic");
    let err = chat.cache_until_here().err().unwrap();
    assert!(
        matches!(&err, Error::Argument(m) if m == "No messages to cache"),
        "{err}"
    );
}

// ---- chat_compaction_spec.rb: request headers --------------------------------------------------

/// The headers of the one request the chat `build` makes (against a server answering
/// `response`) sent for `ask("Hello")`.
async fn sent_headers(
    build: impl FnOnce(Arc<Config>) -> Chat,
    response: Value,
) -> Vec<(String, String)> {
    let server = serve(vec![response]).await;
    let mut chat = build(config_at(&server.uri()));
    chat.ask("Hello").await.unwrap();
    let request = &server.received_requests().await.unwrap()[0];
    request
        .headers
        .iter()
        .map(|(k, v)| (k.as_str().to_string(), v.to_str().unwrap_or("").to_string()))
        .collect()
}

fn header<'a>(headers: &'a [(String, String)], name: &str) -> Vec<&'a str> {
    headers
        .iter()
        .filter(|(k, _)| k.eq_ignore_ascii_case(name))
        .map(|(_, v)| v.as_str())
        .collect()
}

fn anthropic_compacting(config: Arc<Config>) -> Chat {
    chat_with(config, "claude-sonnet-4-6", "anthropic")
        .with_compaction(json!({}))
        .unwrap()
}

// spec: chat_compaction_spec.rb:169 keeps betas another feature already asked for
#[tokio::test]
async fn compaction_beta_keeps_betas_another_feature_already_asked_for() {
    let build = |c| {
        anthropic_compacting(c).with_headers([(
            "anthropic-beta".to_string(),
            "mcp-client-2025-11-20".to_string(),
        )])
    };
    let headers = sent_headers(build, text_response("Hi")).await;
    assert_eq!(
        header(&headers, "anthropic-beta"),
        ["mcp-client-2025-11-20,compact-2026-01-12"]
    );
}

// spec: chat_compaction_spec.rb:175 asks for the beta once when it is already there
#[tokio::test]
async fn compaction_beta_is_asked_for_once_when_already_there() {
    let build = |c| {
        anthropic_compacting(c).with_headers([(
            "anthropic-beta".to_string(),
            "compact-2026-01-12".to_string(),
        )])
    };
    let headers = sent_headers(build, text_response("Hi")).await;
    assert_eq!(header(&headers, "anthropic-beta"), ["compact-2026-01-12"]);
}

// spec: chat_compaction_spec.rb:181 leaves headers alone for protocols with no compaction beta
#[tokio::test]
async fn compaction_leaves_headers_alone_for_protocols_with_no_compaction_beta() {
    let build = |c| {
        chat_with(c, "gpt-5-nano", "openai")
            .with_compaction(json!({}))
            .unwrap()
            .with_headers([("x-test".to_string(), "1".to_string())])
    };
    let headers = sent_headers(build, responses_text("Hi")).await;
    assert_eq!(header(&headers, "x-test"), ["1"]);
    assert!(header(&headers, "anthropic-beta").is_empty(), "{headers:?}");
}

// ---- chat_compact_spec.rb ----------------------------------------------------------------------

fn compaction_body(output: Value) -> Value {
    json!({ "id": "cmp_1", "object": "response.compaction", "output": output, "usage": { "input_tokens": 23, "output_tokens": 7 } })
}

fn first_output() -> Value {
    json!([{ "type": "compaction", "id": "cmp_1", "encrypted_content": "opaque context" }])
}

/// `context.chat(model: model_for(:xai, :provider_tools), provider: :xai)` against `server`.
fn xai_chat(server: &MockServer) -> Chat {
    chat_with(config_at(&server.uri()), "grok-4.3", "xai")
}

// spec: chat_compact_spec.rb:42 uses the latest instructions and last compacted context across multiple rounds
#[tokio::test]
async fn compact_uses_the_latest_instructions_and_last_compacted_context_across_rounds() {
    let second_output =
        json!([{ "type": "compaction", "id": "cmp_2", "encrypted_content": "second context" }]);
    let server = serve(vec![
        compaction_body(first_output()),
        compaction_body(second_output.clone()),
    ])
    .await;
    let mut chat = xai_chat(&server);
    chat.set_instructions(Some("Old instructions".into()), false, false)
        .ask_later("First request")
        .unwrap();
    chat.compact().await.unwrap();
    chat.set_instructions(Some("Answer briefly.".into()), false, false)
        .ask_later("Second request")
        .unwrap();
    chat.compact().await.unwrap();
    chat.ask_later("Third request").unwrap();

    let sent = server.received_requests().await.unwrap();
    assert_eq!(sent.len(), 2);
    let second: Value = serde_json::from_slice(&sent[1].body).unwrap();
    let mut expected = first_output().as_array().unwrap().clone();
    expected.push(json!({ "role": "user", "content": "Second request" }));
    assert_eq!(second["input"], Value::Array(expected));

    let payload = chat.render().unwrap();
    let mut expected = second_output.as_array().unwrap().clone();
    expected.push(json!({ "role": "user", "content": "Third request" }));
    assert_eq!(payload["input"], Value::Array(expected));
    assert_eq!(payload["instructions"], "Answer briefly.");
    assert_eq!(chat.messages().len(), 6);
}

// spec: chat_compact_spec.rb:62 keeps current system attachments and cache boundaries in the rendered input
#[tokio::test]
async fn compact_keeps_current_system_cache_boundaries_in_the_rendered_input() {
    let server = serve(vec![compaction_body(first_output())]).await;
    let mut chat = xai_chat(&server);
    chat.set_instructions(Some("Current instructions".into()), false, true)
        .ask_later("Summarize this")
        .unwrap();
    chat.compact().await.unwrap();

    let input = chat.render().unwrap()["input"].as_array().unwrap().clone();
    assert_eq!(input.first().unwrap()["role"], "system");
    assert_eq!(
        input.first().unwrap()["content"][0]["text"],
        "Current instructions"
    );
    assert_eq!(
        input.last().unwrap(),
        first_output().as_array().unwrap().last().unwrap()
    );
}

/// Answers with `body` after cancelling the chat, like the spec's `to_return { chat.cancel; ... }`.
struct CancelThenAnswer(CancelHandle, Value);

impl Respond for CancelThenAnswer {
    fn respond(&self, _: &Request) -> ResponseTemplate {
        self.0.cancel();
        ResponseTemplate::new(200).set_body_json(self.1.clone())
    }
}

// spec: chat_compact_spec.rb:85 leaves history unchanged after cancellation during the request while retaining billed usage
#[tokio::test]
async fn compact_cancelled_during_the_request_leaves_history_and_keeps_billed_usage() {
    let server = MockServer::start().await;
    let mut chat = xai_chat(&server);
    Mock::given(wiremock::matchers::any())
        .respond_with(CancelThenAnswer(
            chat.cancel_handle(),
            compaction_body(first_output()),
        ))
        .mount(&server)
        .await;
    chat.ask_later("Hello").unwrap();

    assert!(matches!(chat.compact().await, Err(Error::Cancelled)));
    assert_eq!(chat.messages().len(), 1);
    assert_eq!(chat.tokens().input, Some(23));
}

// ---- chat_thinking_spec.rb ---------------------------------------------------------------------

// spec: chat_thinking_spec.rb:332 renders stored reasoning_details verbatim
#[test]
fn openrouter_renders_stored_reasoning_details_verbatim() {
    let details = json!([{
        "type": "reasoning.text", "text": "Thinking about it.", "signature": "sig",
        "format": "anthropic-claude-v1", "index": 0
    }]);
    let mut chat = chat("claude-haiku-4-5", "openrouter");
    chat.add_message(Message::user("Hi"));
    let mut answer = Message::assistant("Hello!");
    answer.raw_reasoning = Some(details.clone());
    chat.add_message(answer);

    let payload = chat.render().unwrap();
    assert_eq!(
        payload["messages"].as_array().unwrap().last().unwrap()["reasoning_details"],
        details
    );
}

// ---- chat_tool_attachments_spec.rb: wire formatting --------------------------------------------

fn drive_search(id: &str) -> (String, ToolCall) {
    (
        id.to_string(),
        ToolCall::new(id, "drive_search", Map::new()),
    )
}

/// `messages_with_tool_attachment(path)`.
fn messages_with_tool_attachment(attachment: Attachment) -> Vec<Message> {
    let mut call = Message::new(Role::Assistant, None);
    call.tool_calls = Some(
        [drive_search("call_1")]
            .into_iter()
            .collect::<IndexMap<ToolCall>>(),
    );
    vec![
        Message::user("Find the ruby logo"),
        call,
        Message::tool_result("call_1", "Found it").with_attachments(vec![attachment]),
    ]
}

fn chat_with_tool_attachment(
    model: &str,
    provider: &str,
    protocol: Option<ProtocolName>,
    attachment: Attachment,
) -> Chat {
    let mut chat = chat(model, provider);
    if let Some(p) = protocol {
        chat = chat.with_protocol(p);
    }
    chat.set_messages(messages_with_tool_attachment(attachment));
    chat
}

// spec: chat_tool_attachments_spec.rb:123 keeps parallel Chat Completions tool results consecutive
#[tokio::test]
async fn keeps_parallel_chat_completions_tool_results_consecutive() {
    let mut chat = chat_with_tool_attachment(
        "gpt-5-nano",
        "openai",
        Some(ProtocolName::ChatCompletions),
        loaded("ruby.png").await,
    );
    let mut call = Message::new(Role::Assistant, None);
    call.tool_calls = Some(
        [drive_search("call_1"), drive_search("call_2")]
            .into_iter()
            .collect::<IndexMap<ToolCall>>(),
    );
    chat.messages_mut()[1] = call;
    chat.add_message(Message::tool_result("call_2", "Found it too"));

    let payload = chat.render().unwrap();
    let roles: Vec<&str> = payload["messages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["role"].as_str().unwrap())
        .collect();
    assert_eq!(roles, ["user", "assistant", "tool", "tool", "user"]);
}

// spec: chat_tool_attachments_spec.rb:134 raises for tool audio on providers without audio support
#[tokio::test]
async fn raises_for_tool_audio_on_providers_without_audio_support() {
    let chat = chat_with_tool_attachment(
        "deepseek-v4-flash",
        "deepseek",
        None,
        loaded("ruby.wav").await,
    );
    assert!(matches!(
        chat.render(),
        Err(Error::UnsupportedAttachment(_))
    ));
}

// spec: chat_tool_attachments_spec.rb:141 raises for tool PDFs on providers without document support
#[tokio::test]
async fn raises_for_tool_pdfs_on_providers_without_document_support() {
    let chat = chat_with_tool_attachment(
        "grok-4-1-fast-non-reasoning",
        "xai",
        Some(ProtocolName::ChatCompletions),
        loaded("sample.pdf").await,
    );
    assert!(matches!(
        chat.render(),
        Err(Error::UnsupportedAttachment(_))
    ));
}

// ---- chat_schema_spec.rb: schema name sanitization ---------------------------------------------

/// The name `with_schema` settled on, read from the rendered Responses `text.format`.
fn schema_name(schema: Value) -> Value {
    let mut chat = chat("gpt-4.1-nano", "openai").with_schema(schema);
    chat.ask_later("hi").unwrap();
    chat.render().unwrap()["text"]["format"]["name"].clone()
}

// spec: chat_schema_spec.rb:120 falls back to title for a bare JSON Schema document
#[test]
fn schema_name_falls_back_to_title_for_a_bare_json_schema_document() {
    let name = schema_name(json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "PersonSchema",
        "type": "object",
        "properties": {}
    }));
    assert_eq!(name, "PersonSchema");
}

// spec: chat_schema_spec.rb:132 prefers name over title when both are present
#[test]
fn schema_name_prefers_name_over_title() {
    let name = schema_name(
        json!({ "name": "EnvelopeName", "title": "DocumentTitle", "schema": { "type": "object", "properties": {} } }),
    );
    assert_eq!(name, "EnvelopeName");
}

// ---- chat_error_spec.rb: responses without a completion message --------------------------------

async fn deepseek_answering(body: Value) -> (Chat, MockServer) {
    let server = serve(vec![body]).await;
    (
        chat_with(config_at(&server.uri()), "deepseek-v4-flash", "deepseek"),
        server,
    )
}

// spec: chat_error_spec.rb:105 raises a RubyLLM::Error instead of an obscure NoMethodError
#[tokio::test]
async fn no_completion_message_raises_an_api_error_with_the_response() {
    let (mut chat, _server) = deepseek_answering(json!({ "choices": [] })).await;
    let err = chat.ask("Hello").await.unwrap_err();
    assert!(
        matches!(&err, Error::Api(m, _) if m == "Provider returned no completion message"),
        "{err:?}"
    );
    assert!(err.response().is_some());
}

// spec: chat_error_spec.rb:117 surfaces the finish_reason when the provider gives one
#[tokio::test]
async fn no_completion_message_surfaces_the_finish_reason() {
    let (mut chat, _server) =
        deepseek_answering(json!({ "choices": [{ "finish_reason": "content_filter" }] })).await;
    let err = chat.ask("Hello").await.unwrap_err();
    assert!(
        matches!(&err, Error::Api(m, _) if m == "Provider returned no completion message (finish_reason: content_filter)"),
        "{err:?}"
    );
}

// ---- chat_tools_spec.rb: stubbed-provider examples, per supported provider ---------------------

/// The chat models `spec/support/models_to_test.rb` covers that this port implements.
const CHAT_MODELS: &[(&str, &str)] = &[
    ("anthropic", "claude-haiku-4-5"),
    ("deepseek", "deepseek-v4-flash"),
    ("gemini", "gemini-2.5-flash"),
    ("gpustack", "qwen3"),
    ("hetzner", "Qwen3.8-27B"),
    ("mistral", "mistral-small-latest"),
    ("ollama", "qwen3"),
    ("ollama_cloud", "gpt-oss:120b"),
    ("openai", "gpt-5-nano"),
    ("openrouter", "claude-haiku-4-5"),
    ("perplexity", "openai/gpt-5-mini"),
    ("xai", "grok-4-1-fast-non-reasoning"),
];

fn responses_text(text: &str) -> Value {
    json!({ "id": "resp_2", "object": "response", "status": "completed", "model": "m",
            "output": [{ "type": "message", "role": "assistant", "content": [{ "type": "output_text", "text": text }] }],
            "usage": { "input_tokens": 1, "output_tokens": 1 } })
}

/// A provider response in `chat`'s wire format carrying `text`.
fn text_answer(chat: &Chat, text: &str) -> Value {
    match chat
        .provider()
        .resolve_protocol(chat.protocol(), chat.model(), chat.config())
        .unwrap()
    {
        ProtocolName::Anthropic => text_response(text),
        ProtocolName::Responses => responses_text(text),
        ProtocolName::Gemini => json!({
            "candidates": [{ "content": { "role": "model", "parts": [{ "text": text }] }, "finishReason": "STOP" }],
            "usageMetadata": { "promptTokenCount": 1, "candidatesTokenCount": 1 }
        }),
        ProtocolName::ChatCompletions => json!({
            "id": "c1", "model": "m",
            "choices": [{ "index": 0, "message": { "role": "assistant", "content": text }, "finish_reason": "stop" }],
            "usage": { "prompt_tokens": 1, "completion_tokens": 1 }
        }),
        #[allow(unreachable_patterns)]
        other => panic!("no canned answer for {other:?}"),
    }
}

/// A provider response in `chat`'s wire format calling `name` with no arguments as `call_1`.
fn tool_call_answer(chat: &Chat, name: &str) -> Value {
    match chat
        .provider()
        .resolve_protocol(chat.protocol(), chat.model(), chat.config())
        .unwrap()
    {
        ProtocolName::Anthropic => tool_use_response(&[("call_1", name, json!({}))]),
        ProtocolName::Responses => {
            json!({ "id": "resp_1", "object": "response", "status": "completed", "model": "m",
            "output": [{ "type": "function_call", "call_id": "call_1", "name": name, "arguments": "{}" }],
            "usage": { "input_tokens": 1, "output_tokens": 1 } })
        }
        ProtocolName::Gemini => json!({
            "candidates": [{ "content": { "role": "model", "parts": [{ "functionCall": { "name": name, "args": {} } }] }, "finishReason": "STOP" }],
            "usageMetadata": { "promptTokenCount": 1, "candidatesTokenCount": 1 }
        }),
        ProtocolName::ChatCompletions => json!({
            "id": "c1", "model": "m",
            "choices": [{ "index": 0, "finish_reason": "tool_calls", "message": { "role": "assistant", "content": null,
                "tool_calls": [{ "id": "call_1", "type": "function", "function": { "name": name, "arguments": "{}" } }] } }],
            "usage": { "prompt_tokens": 1, "completion_tokens": 1 }
        }),
        #[allow(unreachable_patterns)]
        other => panic!("no canned answer for {other:?}"),
    }
}

/// `Weather`.
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
    async fn execute(
        &self,
        args: Map<String, Value>,
        _: &ToolCall,
    ) -> Result<ToolResult, ToolError> {
        Ok(format!(
            "Current weather at {}, {}: 15°C, Wind: 10 km/h",
            args["latitude"], args["longitude"]
        )
        .into())
    }
}

/// `ParamsTool`: `provider_options cache_control: { type: 'ephemeral' }`.
struct ParamsTool;

#[async_trait]
impl Tool for ParamsTool {
    fn name(&self) -> String {
        "params".into()
    }
    fn description(&self) -> String {
        "Has provider-specific params".into()
    }
    fn provider_options(&self) -> Map<String, Value> {
        json!({ "cache_control": { "type": "ephemeral" } })
            .as_object()
            .unwrap()
            .clone()
    }
    async fn execute(&self, _: Map<String, Value>, _: &ToolCall) -> Result<ToolResult, ToolError> {
        Ok("ok".to_string().into())
    }
}

// spec: chat_tools_spec.rb:240 #{provider}/#{model} deals with non-existent tool calls
#[tokio::test]
async fn deals_with_non_existent_tool_calls() {
    let final_answer =
        "The `list_tools` tool is not supported, but I see you have the `weather` tool.";
    for &(provider, model) in CHAT_MODELS {
        let probe = chat(model, provider);
        let server = serve(vec![
            tool_call_answer(&probe, "list_tools"),
            text_answer(&probe, final_answer),
        ])
        .await;
        let received: Log<ToolResult> = log();
        let sink = received.clone();
        let mut chat = chat_with(config_at(&server.uri()), model, provider)
            .with_tool(Weather)
            .after_tool_result(move |r| sink.lock().unwrap().push(r.clone()));

        let response = chat
            .ask("What tools do you support?")
            .await
            .unwrap_or_else(|e| panic!("{provider}: {e}"));
        assert_eq!(response.content(), final_answer, "{provider}");
        assert_eq!(
            *received.lock().unwrap(),
            vec![ToolResult::error(
                "Model tried to call unavailable tool `list_tools`. Available tools: [\"weather\"]."
            )],
            "{provider}"
        );
        assert_eq!(requests(&server).await, 2, "{provider}");
    }
}

// spec: chat_tools_spec.rb:424 #{provider}/#{model} can handle tool provider_options
#[tokio::test]
async fn can_handle_tool_provider_options() {
    for &(provider, model) in CHAT_MODELS {
        let probe = chat(model, provider);
        // `skip_unless_supports_functions`: local providers always run.
        let local = matches!(provider, "ollama" | "gpustack");
        if !local && !probe.model().supports("function_calling") {
            eprintln!("{provider}/{model}: skipped, no function calling");
            continue;
        }
        let server = serve(vec![text_answer(&probe, "ok")]).await;
        let mut chat = chat_with(config_at(&server.uri()), model, provider)
            .with_tool(ParamsTool)
            .with_instructions("You must call the params tool.");
        chat.ask("Call the params tool for me")
            .await
            .unwrap_or_else(|e| panic!("{provider}: {e}"));

        let sent = server.received_requests().await.unwrap();
        let payload: Value = serde_json::from_slice(&sent[0].body).unwrap();
        let extracted = match provider {
            "gemini" => payload.pointer("/tools/0/functionDeclarations/0/cache_control"),
            _ => payload.pointer("/tools/0/cache_control"),
        };
        assert_eq!(
            extracted,
            Some(&json!({ "type": "ephemeral" })),
            "{provider}: {payload}"
        );
    }
}

// ---- chat_streaming_spec.rb: Error handling ----------------------------------------------------

/// `StreamingErrorHelpers::ERROR_HANDLING_CONFIGS` for the providers this port supports:
/// the error body, the status it streams with, and the error class RubyLLM raises.
fn error_chunk_config(provider: &str) -> (Value, u16, ErrorKind) {
    match provider {
        "anthropic" => (
            json!({ "type": "error", "error": { "type": "overloaded_error", "message": "Overloaded" } }),
            529,
            ErrorKind::Overloaded,
        ),
        "openai" => (
            json!({ "error": { "message": "The server is temporarily overloaded. Please try again later.", "type": "server_error", "param": null, "code": null } }),
            500,
            ErrorKind::Server,
        ),
        "gemini" => (
            json!({ "error": { "code": 529, "message": "Service overloaded - please try again later", "status": "RESOURCE_EXHAUSTED" } }),
            529,
            ErrorKind::Overloaded,
        ),
        _ => (
            json!({ "error": { "message": "Service overloaded - please try again later", "type": "server_error", "param": null, "code": null } }),
            500,
            ErrorKind::Server,
        ),
    }
}

// spec: chat_streaming_spec.rb:89 #{provider}/#{model} supports handling streaming error chunks
// (The Faraday 1/2 variants exercise Ruby's two streaming adapters; the port has one transport.)
#[tokio::test]
async fn supports_handling_streaming_error_chunks() {
    for &(provider, model) in CHAT_MODELS {
        let (body, status, expected) = error_chunk_config(provider);
        let server = MockServer::start().await;
        Mock::given(wiremock::matchers::any())
            .respond_with(
                ResponseTemplate::new(status)
                    .set_body_raw(format!("{body}\n\n").into_bytes(), "text/event-stream"),
            )
            .mount(&server)
            .await;
        let mut chat = chat_with(config_at(&server.uri()), model, provider);
        let mut chunks = Vec::new();
        let err = chat
            .ask_stream("Count from 1 to 3", |c| chunks.push(c.clone()))
            .await
            .unwrap_err();
        assert_eq!(err.kind(), expected, "{provider}: {err:?}");
    }
}

// ---- chat_content_spec.rb ----------------------------------------------------------------------

// spec: chat_content_spec.rb:233 handles URL MIME type detection without ArgumentError
#[test]
fn url_attachment_detects_its_mime_type() {
    let attachment =
        Attachment::new("https://upload.wikimedia.org/wikipedia/commons/f/f1/Ruby_logo.png");
    assert!(!attachment.mime_type.is_empty());
    assert_eq!(attachment.mime_type, "image/png");
}

// ---- chat_pricing_spec.rb ----------------------------------------------------------------------

/// A Chat Completions answer from `model` using 19 prompt and 17 completion tokens.
fn priced_answer(model: &str, streaming: bool) -> ResponseTemplate {
    let key = if streaming { "delta" } else { "message" };
    let body = json!({
        "model": model,
        "choices": [{ "index": 0, key: { "role": "assistant", "content": "ok" }, "finish_reason": "stop" }],
        "usage": { "prompt_tokens": 19, "completion_tokens": 17, "total_tokens": 36 }
    });
    if streaming {
        ResponseTemplate::new(200).set_body_raw(
            format!("data: {body}\n\ndata: [DONE]\n\n").into_bytes(),
            "text/event-stream",
        )
    } else {
        ResponseTemplate::new(200).set_body_json(body)
    }
}

// spec: chat_pricing_spec.rb:35 uses the custom provider's prices with streaming #{streaming}
// Ruby registers a custom provider whose model id another provider also lists. Providers are a
// closed enum here, so the same fact is checked with a shipped pair: `deepseek-v4-flash` is
// listed by DeepSeek, Ollama Cloud, and Azure at different prices, and an Ollama Cloud chat
// must bill at Ollama Cloud's ($0.22 in, $0.66 out per million).
#[tokio::test]
async fn uses_the_chat_providers_own_prices_streaming_and_not() {
    for streaming in [false, true] {
        let server = MockServer::start().await;
        Mock::given(wiremock::matchers::any())
            .respond_with(priced_answer("deepseek-v4-flash", streaming))
            .mount(&server)
            .await;
        let mut chat = chat_with(
            config_at(&server.uri()),
            "deepseek-v4-flash",
            "ollama_cloud",
        )
        .with_protocol(ProtocolName::ChatCompletions);
        let response = if streaming {
            chat.ask_stream("Reply with ok", |c| {
                if !c.content().is_empty() {
                    assert_eq!(c.content(), "ok");
                }
            })
            .await
            .unwrap()
        } else {
            chat.ask("Reply with ok").await.unwrap()
        };

        assert_eq!(response.content(), "ok");
        let info = response.model_info().unwrap();
        assert_eq!(
            (info.id.as_str(), info.provider.as_str()),
            ("deepseek-v4-flash", "ollama_cloud")
        );
        let cost = response.cost(None);
        let near = |a: Option<f64>, b: f64| a.is_some_and(|a| (a - b).abs() < 1e-12);
        assert!(near(cost.input, 19.0 * 0.22 / 1e6), "{streaming}: {cost:?}");
        assert!(
            near(cost.output, 17.0 * 0.66 / 1e6),
            "{streaming}: {cost:?}"
        );
        assert!(
            near(cost.total(), 19.0 * 0.22 / 1e6 + 17.0 * 0.66 / 1e6),
            "{streaming}: {cost:?}"
        );
        assert_eq!(chat.cost().total(), cost.total());
        let entries = chat.usage_entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(
            (entries[0].provider.as_str(), entries[0].model.as_str()),
            ("ollama_cloud", "deepseek-v4-flash")
        );
        assert_eq!(entries[0].cost.total(), cost.total());
    }
}
