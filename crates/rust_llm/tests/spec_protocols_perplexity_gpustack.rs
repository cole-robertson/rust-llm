//! Responses dialect specs ported from RubyLLM 2.0: `protocols/perplexity/agent_spec.rb` (the
//! Perplexity Agent API) and `protocols/gpustack/responses_spec.rb` (vLLM Responses with MCP
//! servers configured on the GPUStack deployment). Ruby calls `render`, `ask`, and the protocol's
//! `format_assistant_items`/`parse_completion_body`; these go through `Chat#render`, `ask`, and
//! `generate` against a mock server, which run the same code.

mod spec_helpers;

use std::sync::{Arc, Mutex};

use rust_llm::{
    Attachment, Chat, Config, Error, Message, ProtocolName, Provider, ProviderTool, Role,
    TranscribeOptions,
};
use serde_json::{Value, json};
use spec_helpers::*;
use wiremock::MockServer;

fn fixture(name: &str) -> String {
    format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
}

/// A fixture loaded into memory, as `Attachment.new(path)` reads it before rendering.
fn inline(name: &str) -> Attachment {
    Attachment::from_bytes(std::fs::read(fixture(name)).unwrap(), name, None)
}

/// Collects `tracing` WARN events on this thread, standing in for `RubyLLM.deprecator.warn`.
struct WarnCollector(Arc<Mutex<Vec<String>>>);

impl tracing::Subscriber for WarnCollector {
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
        struct Visitor<'a>(&'a mut String);
        impl tracing::field::Visit for Visitor<'_> {
            fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                if field.name() == "message" {
                    self.0.push_str(&format!("{value:?}"));
                }
            }
        }
        if *event.metadata().level() == tracing::Level::WARN {
            let mut text = String::new();
            event.record(&mut Visitor(&mut text));
            self.0.lock().unwrap().push(text);
        }
    }
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}

fn warnings_of<T>(f: impl FnOnce() -> T) -> (T, Vec<String>) {
    let warnings = Arc::new(Mutex::new(Vec::new()));
    let result = {
        let _guard = tracing::dispatcher::set_default(&tracing::Dispatch::new(WarnCollector(
            warnings.clone(),
        )));
        f()
    };
    let collected = warnings.lock().unwrap().clone();
    (result, collected)
}

async fn bodies(server: &MockServer) -> Vec<Value> {
    server
        .received_requests()
        .await
        .unwrap_or_default()
        .iter()
        .map(|r| serde_json::from_slice(&r.body).unwrap_or(Value::Null))
        .collect()
}

// ---- perplexity/agent_spec.rb ------------------------------------------------------------------

fn perplexity_config(base: &str) -> Arc<Config> {
    let mut c = Config::default();
    c.set("perplexity_api_base", base);
    c.set("perplexity_api_key", "test");
    c.max_retries = 0;
    Arc::new(c)
}

/// `RubyLLM.chat(model:, provider: :perplexity)` with a user message, like `preset_chat`.
fn perplexity(server: &MockServer, model: &str) -> Chat {
    let mut chat = Chat::with_config(
        perplexity_config(&server.uri()),
        Some(model),
        Some("perplexity"),
        false,
    )
    .unwrap();
    chat.add_message(Message::user("Hello"));
    chat
}

/// `model_for(:perplexity, :agent)`.
const AGENT_MODEL: &str = "openai/gpt-5-mini";

fn agent_response(usage: Value) -> Value {
    json!({
        "id": "resp_1", "object": "response", "status": "completed", "model": AGENT_MODEL,
        "output": [
            { "type": "search_results", "queries": ["rails creator"], "results": [
                { "id": 1, "title": "Ruby on Rails", "url": "https://rubyonrails.org", "snippet": "Rails is a web framework." },
                { "id": 2, "title": "DHH", "url": "https://dhh.dk" }
            ] },
            { "type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
              "content": [{ "type": "output_text", "text": "Rails was created by David Heinemeier Hansson.[1]", "annotations": [] }] }
        ],
        "usage": usage
    })
}

// spec: protocols/perplexity/agent_spec.rb:61 keeps Sonar Chat Completions available as an explicit protocol
#[tokio::test]
async fn keeps_sonar_chat_completions_available_as_an_explicit_protocol() {
    let server = serve(vec![]).await;
    let sonar = perplexity(&server, "sonar").with_protocol(ProtocolName::ChatCompletions);
    let payload = sonar.render().unwrap();
    assert_eq!(payload["model"], json!("sonar"));
    assert_eq!(
        payload["messages"],
        json!([{ "role": "user", "content": "Hello" }])
    );
}

// spec: protocols/perplexity/agent_spec.rb:82 runs a retired Sonar model id as its recommended preset and warns
#[tokio::test]
async fn runs_a_retired_sonar_model_id_as_its_recommended_preset_and_warns() {
    let server = serve(vec![]).await;
    let chat = perplexity(&server, "sonar-pro");
    let (payload, warnings) = warnings_of(|| chat.render().unwrap());
    assert_eq!(payload["preset"], json!("low"));
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("sonar-pro now runs the low Agent API preset")),
        "{warnings:?}"
    );
}

// spec: protocols/perplexity/agent_spec.rb:89 caps the output of Anthropic models, which Perplexity requires
// spec: protocols/perplexity/agent_spec.rb:93 caps Anthropic models the registry has no output limit for
// spec: protocols/perplexity/agent_spec.rb:97 keeps an explicit output cap for Anthropic models
#[tokio::test]
async fn caps_the_output_of_anthropic_models() {
    let server = serve(vec![]).await;
    assert_eq!(
        perplexity(&server, "anthropic/claude-haiku-4-5")
            .render()
            .unwrap()["max_output_tokens"],
        json!(64_000)
    );
    assert_eq!(
        perplexity(&server, "anthropic/claude-sonnet-5")
            .render()
            .unwrap()["max_output_tokens"],
        json!(4096)
    );
    let claude = perplexity(&server, "anthropic/claude-haiku-4-5").with_max_output_tokens(200);
    assert_eq!(claude.render().unwrap()["max_output_tokens"], json!(200));
}

// spec: protocols/perplexity/agent_spec.rb:109 posts to the Agent endpoint, keeping configured gateway base paths
#[tokio::test]
async fn posts_to_the_agent_endpoint_keeping_configured_gateway_base_paths() {
    let default = perplexity_config("https://api.perplexity.ai");
    let mut unset = Config::default();
    unset.set("perplexity_api_key", "test");
    assert_eq!(
        Provider::Perplexity.agent_url(&unset).unwrap(),
        "https://api.perplexity.ai/v1/agent"
    );
    assert_eq!(
        Provider::Perplexity.agent_url(&default).unwrap(),
        "https://api.perplexity.ai/v1/agent"
    );
    for base in [
        "https://gateway.test/perplexity",
        "https://gateway.test/perplexity/v1/",
    ] {
        assert_eq!(
            Provider::Perplexity
                .agent_url(&perplexity_config(base))
                .unwrap(),
            "https://gateway.test/perplexity/v1/agent"
        );
    }
    // And the chat really posts there through a gateway base that ends in /v1/.
    let server = serve(vec![agent_response(
        json!({ "input_tokens": 12, "output_tokens": 7 }),
    )])
    .await;
    let config = perplexity_config(&format!("{}/perplexity/v1/", server.uri()));
    let mut chat = Chat::with_config(config, Some(AGENT_MODEL), Some("perplexity"), false).unwrap();
    chat.ask("Who created Rails?").await.unwrap();
    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests[0].url.path(), "/perplexity/v1/agent");
}

// spec: protocols/perplexity/agent_spec.rb:150 counts cache writes apart from fresh input
#[tokio::test]
async fn counts_cache_writes_apart_from_fresh_input() {
    let usage = json!({ "input_tokens": 1494, "output_tokens": 7,
                        "input_tokens_details": { "cache_creation_input_tokens": 32, "cached_tokens": 1459 } });
    let server = serve(vec![agent_response(usage)]).await;
    let mut chat = Chat::with_config(
        perplexity_config(&server.uri()),
        Some(AGENT_MODEL),
        Some("perplexity"),
        false,
    )
    .unwrap();
    let tokens = chat.ask("Who created Rails?").await.unwrap().tokens;
    assert_eq!(
        (tokens.input, tokens.cache_read, tokens.cache_write),
        (Some(3), Some(1459), Some(32))
    );
}

// spec: protocols/perplexity/agent_spec.rb:209 rejects documents, which the Agent API does not accept
#[tokio::test]
async fn rejects_documents_which_the_agent_api_does_not_accept() {
    let server = serve(vec![]).await;
    let mut chat = Chat::with_config(
        perplexity_config(&server.uri()),
        Some(AGENT_MODEL),
        Some("perplexity"),
        false,
    )
    .unwrap();
    chat.add_message(Message::user("Summarize this.").with_attachments(vec![inline("sample.pdf")]));
    match chat.render().unwrap_err() {
        Error::UnsupportedAttachment(m) => assert!(m.contains("application/pdf"), "{m}"),
        other => panic!("expected UnsupportedAttachment, got {other:?}"),
    }
}

// ---- gpustack/responses_spec.rb ----------------------------------------------------------------

/// `model_for(:gpustack)`.
const GPUSTACK_MODEL: &str = "qwen3";
const PROXY: &str = "/cluster/model/proxy/42/v1";

fn gpustack_config(server: &MockServer, protocol: Option<&str>) -> Arc<Config> {
    let mut c = Config::default();
    c.set("gpustack_api_base", format!("{}{PROXY}", server.uri()));
    c.set("gpustack_api_key", "isolated-key");
    if let Some(p) = protocol {
        c.set("gpustack_protocol", p);
    }
    c.max_retries = 0;
    Arc::new(c)
}

/// `context.chat(model:, provider: :gpustack, protocol: :responses)`.
fn gpustack(server: &MockServer) -> Chat {
    Chat::with_config(
        gpustack_config(server, None),
        Some(GPUSTACK_MODEL),
        Some("gpustack"),
        false,
    )
    .unwrap()
    .with_protocol(ProtocolName::Responses)
}

fn mcp_call() -> Value {
    json!({ "type": "mcp_call", "id": "mcp_1", "name": "python", "server_label": "python",
            "status": "completed", "arguments": "{\"code\":\"print(2+2)\"}", "output": "4" })
}

fn completion() -> Value {
    json!({ "model": GPUSTACK_MODEL, "status": "completed",
            "output": [mcp_call(), { "type": "message", "role": "assistant", "content": [{ "type": "output_text", "text": "4" }] }],
            "usage": { "input_tokens": 20, "output_tokens": 5, "input_tokens_details": { "cached_tokens": 4 } } })
}

fn mcp(options: Value) -> ProviderTool {
    ProviderTool::with_options("mcp", options)
}

fn never() -> Value {
    json!({ "require_approval": "never" })
}

// spec: protocols/gpustack/responses_spec.rb:26 sends configured MCP labels and preserves complete calls, usage, and multi-turn replay
#[tokio::test]
async fn sends_configured_mcp_labels_and_preserves_complete_calls_usage_and_replay() {
    let server = serve(vec![completion(), completion()]).await;
    let mut chat = gpustack(&server).with_provider_tools([mcp(
        json!({ "name": "code_interpreter", "require_approval": "never" }),
    )]);
    let result = chat.ask("Calculate 2+2.").await.unwrap();

    let call = &result.server_tool_calls[0];
    assert_eq!(
        (
            call.name.as_deref(),
            call.result.clone(),
            call.id.as_deref()
        ),
        (Some("python"), Some(json!("4")), Some("mcp_1"))
    );
    assert_eq!(
        (
            result.tokens.input,
            result.tokens.cache_read,
            result.tokens.output
        ),
        (Some(16), Some(4), Some(5))
    );
    assert_eq!(result.content(), "4");
    assert_eq!(result.finish_reason, Some(rust_llm::FinishReason::Stop));

    chat.ask("Repeat the answer.").await.unwrap();
    assert_eq!(result.raw_content.as_ref().unwrap()[0], mcp_call());

    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 2);
    for request in &requests {
        assert_eq!(request.url.path(), format!("{PROXY}/responses"));
        assert_eq!(
            request.headers.get("authorization").unwrap(),
            "Bearer isolated-key"
        );
        let body: Value = serde_json::from_slice(&request.body).unwrap();
        assert_eq!(
            body["tools"],
            json!([{ "type": "mcp", "server_label": "code_interpreter", "require_approval": "never" }])
        );
    }
    let second: Value = serde_json::from_slice(&requests[1].body).unwrap();
    let input = second["input"].as_array().unwrap();
    assert!(input.contains(&json!({ "type": "function_call", "call_id": "mcp_1", "name": "python", "arguments": "{\"code\":\"print(2+2)\"}" })));
    assert!(
        input.contains(
            &json!({ "type": "function_call_output", "call_id": "mcp_1", "output": "4" })
        )
    );
    assert!(input.iter().all(|i| i["type"] != json!("mcp_call")));
}

// spec: protocols/gpustack/responses_spec.rb:51 retains incomplete Harmony tool records without inventing results or replaying unsupported item types
#[tokio::test]
async fn retains_incomplete_harmony_tool_records_without_inventing_results() {
    let raw = json!([
        { "type": "web_search_call", "id": "ws_1", "status": "completed", "action": { "type": "search", "query": "cursor:Ruby" } },
        { "type": "mcp_call", "name": "exec", "id": "mcp_2", "arguments": "{}", "output": null }
    ]);
    let server = serve(vec![json!({ "output": raw })]).await;
    let mut chat = gpustack(&server);
    let mut assistant = Message::new(Role::Assistant, Some("Ruby".to_string()));
    assistant.raw_content = Some(raw.clone());
    chat.add_message(assistant);

    let rendered = chat.render().unwrap()["input"].as_array().cloned().unwrap();
    assert_eq!(
        rendered
            .iter()
            .map(|i| i["role"].clone())
            .collect::<Vec<_>>(),
        [json!("assistant"), json!("assistant")]
    );
    let replayed: Vec<Value> = rendered
        .iter()
        .map(|i| serde_json::from_str(i["content"][0]["text"].as_str().unwrap()).unwrap())
        .collect();
    assert_eq!(Value::Array(replayed), raw);
    assert_eq!(chat.messages()[0].raw_content.as_ref(), Some(&raw));

    chat.add_message(Message::user("hi"));
    let parsed = chat.generate().await.unwrap();
    assert_eq!(
        parsed
            .server_tool_calls
            .iter()
            .map(|c| c.result.clone())
            .collect::<Vec<_>>(),
        [None, None]
    );
}

// spec: protocols/gpustack/responses_spec.rb:67 reads actual Harmony reasoning content when the provider has no summary
#[tokio::test]
async fn reads_actual_harmony_reasoning_content_when_there_is_no_summary() {
    let body = json!({ "output": [{ "type": "reasoning", "summary": [],
                                    "content": [{ "type": "reasoning_text", "text": "I can calculate this." }] }] });
    let server = serve(vec![body]).await;
    let mut chat = gpustack(&server);
    chat.add_message(Message::user("hi"));
    let message = chat.generate().await.unwrap();
    assert_eq!(
        message.thinking.and_then(|t| t.text).as_deref(),
        Some("I can calculate this.")
    );
}

// spec: protocols/gpustack/responses_spec.rb:75 passes documented browser subtool filters to the configured server
#[tokio::test]
async fn passes_documented_browser_subtool_filters_to_the_configured_server() {
    let server = serve(vec![completion()]).await;
    gpustack(&server)
        .with_provider_tools([mcp(json!({ "name": "web_search_preview", "allowed_tools": ["search"], "require_approval": "never" }))])
        .ask("Search for Ruby.")
        .await
        .unwrap();
    let bodies = bodies(&server).await;
    assert_eq!(bodies.len(), 1);
    assert_eq!(
        bodies[0]["tools"][0],
        json!({ "type": "mcp", "server_label": "web_search_preview", "allowed_tools": ["search"], "require_approval": "never" })
    );
}

// spec: protocols/gpustack/responses_spec.rb:87 maps portable server tools to only their configured vLLM namespace and subtools
#[tokio::test]
async fn maps_portable_server_tools_to_their_vllm_namespace_and_subtools() {
    let expected = [
        (
            "web_search",
            json!({ "server_label": "web_search_preview", "allowed_tools": ["search"] }),
        ),
        (
            "web_fetch",
            json!({ "server_label": "web_search_preview", "allowed_tools": ["open"] }),
        ),
        (
            "code_execution",
            json!({ "server_label": "code_interpreter" }),
        ),
    ];
    for (name, settings) in expected {
        let server = serve(vec![completion()]).await;
        let response = gpustack(&server)
            .with_provider_tools([ProviderTool::with_options(name, never())])
            .ask("Use the enabled tool.")
            .await
            .unwrap();
        assert_eq!(response.server_tool_calls[0].result, Some(json!("4")));
        let mut tool = json!({ "type": "mcp", "require_approval": "never" });
        tool.as_object_mut()
            .unwrap()
            .extend(settings.as_object().unwrap().clone());
        let bodies = bodies(&server).await;
        assert_eq!(bodies.len(), 1, "{name}");
        assert_eq!(bodies[0]["tools"], json!([tool]), "{name}");
    }
}

// spec: protocols/gpustack/responses_spec.rb:106 combines search and fetch filters so vLLM cannot overwrite one with the other
#[tokio::test]
async fn combines_search_and_fetch_filters() {
    let server = serve(vec![completion()]).await;
    gpustack(&server)
        .with_provider_tools([
            ProviderTool::with_options("web_search", never()),
            ProviderTool::with_options("web_fetch", never()),
        ])
        .ask("Search for the Ruby documentation and read the result.")
        .await
        .unwrap();
    let bodies = bodies(&server).await;
    assert_eq!(bodies.len(), 1);
    assert_eq!(
        bodies[0]["tools"],
        json!([{ "type": "mcp", "server_label": "web_search_preview", "allowed_tools": ["search", "open"], "require_approval": "never" }])
    );
}

async fn argument_error(server: &MockServer, tools: Vec<ProviderTool>) -> String {
    match gpustack(server)
        .with_provider_tools(tools)
        .ask("Use the tool.")
        .await
        .unwrap_err()
    {
        Error::Argument(m) => m,
        other => panic!("expected an ArgumentError, got {other:?}"),
    }
}

// spec: protocols/gpustack/responses_spec.rb:118 rejects alias options that would broaden or change the requested operation
#[tokio::test]
async fn rejects_alias_options_that_would_broaden_or_change_the_operation() {
    let server = serve(vec![]).await;
    for name in ["web_search", "web_fetch", "code_execution"] {
        for options in [
            json!({}),
            json!({ "require_approval": "always" }),
            json!({ "require_approval": "never", "allowed_tools": ["*"] }),
            json!({ "require_approval": "never", "url": "https://example.test/mcp" }),
            json!({ "require_approval": "never", "name": "container" }),
        ] {
            let message = argument_error(
                &server,
                vec![ProviderTool::with_options(name, options.clone())],
            )
            .await;
            assert!(message.contains("GPUStack"), "{name} {options}: {message}");
        }
    }
    assert_eq!(requests(&server).await, 0);
}

// spec: protocols/gpustack/responses_spec.rb:130 rejects duplicate server settings that cannot preserve the explicit filters
#[tokio::test]
async fn rejects_duplicate_server_settings_that_cannot_preserve_the_filters() {
    let server = serve(vec![]).await;
    for options in [
        json!({ "allowed_tools": null }),
        json!({ "allowed_tools": ["*"] }),
        json!({ "allowed_tools": { "tool_names": ["open"] } }),
        json!({ "allowed_tools": ["open"], "server_description": "A different browser" }),
    ] {
        let mut settings = json!({ "name": "web_search_preview", "require_approval": "never" });
        settings
            .as_object_mut()
            .unwrap()
            .extend(options.as_object().unwrap().clone());
        let message = argument_error(
            &server,
            vec![
                ProviderTool::with_options("web_search", never()),
                mcp(settings),
            ],
        )
        .await;
        assert!(
            message.contains("one entry with explicit tool names"),
            "{options}: {message}"
        );
    }
    assert_eq!(requests(&server).await, 0);
}

// spec: protocols/gpustack/responses_spec.rb:143 rejects approval modes, per-request servers, and ignored read-only filters before HTTP
#[tokio::test]
async fn rejects_approval_modes_per_request_servers_and_read_only_filters() {
    let server = serve(vec![]).await;
    let tool = || json!({ "name": "code_interpreter", "require_approval": "never" });
    let with = |key: &str, value: Value| {
        let mut t = tool();
        t[key] = value;
        t
    };
    let mut without_approval = tool();
    without_approval
        .as_object_mut()
        .unwrap()
        .remove("require_approval");
    for options in [
        without_approval,
        with("require_approval", json!("always")),
        with("url", json!("https://example.test/mcp")),
        with("connector_id", json!("connector")),
        with("allowed_tools", json!({ "read_only": true })),
        with("name", json!("unknown")),
    ] {
        let message = argument_error(&server, vec![mcp(options.clone())]).await;
        assert!(
            message.contains("GPUStack") || message.contains("vLLM"),
            "{options}: {message}"
        );
    }
    assert_eq!(requests(&server).await, 0);
}

// spec: protocols/gpustack/responses_spec.rb:154 requires explicit execution consent and keeps ordinary chat on Chat Completions
#[tokio::test]
async fn requires_explicit_execution_consent_and_keeps_chat_on_chat_completions() {
    let server = serve(vec![json!({ "text": "hello" })]).await;
    let message = argument_error(&server, vec!["web_search".into()]).await;
    assert!(message.contains("explicit require_approval"), "{message}");

    let chat = gpustack(&server);
    let model = chat.model().clone();
    let resolve = |config: &Config| {
        Provider::GPUStack
            .resolve_protocol(None, &model, config)
            .unwrap()
    };
    assert_eq!(
        resolve(&gpustack_config(&server, None)),
        ProtocolName::ChatCompletions
    );
    let configured = gpustack_config(&server, Some("responses"));
    assert_eq!(resolve(&configured), ProtocolName::Responses);

    // `operation: :transcribe` stays on the Chat Completions endpoints despite the configured protocol.
    let options = TranscribeOptions {
        model: Some("whisper-large-v3"),
        provider: Some("gpustack"),
        assume_model_exists: true,
        config: Some(configured),
        ..Default::default()
    };
    rust_llm::transcribe(Attachment::new(fixture("ruby.wav")), options)
        .await
        .unwrap();
    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(
        requests[0].url.path(),
        format!("{PROXY}/audio/transcriptions")
    );
}
