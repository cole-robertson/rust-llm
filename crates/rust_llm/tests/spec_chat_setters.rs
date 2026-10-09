//! Setter, agent, and context leftovers ported from RubyLLM 2.0: the `nil`-clearing forms of the
//! `with_*` setters (`chat_functions_spec.rb`, `chat_options_spec.rb`, `chat_request_options_spec.rb`,
//! `chat_headers_spec.rb`), `Agent.apply_configuration` (`agent_spec.rb`, `agent_dsl_spec.rb`),
//! `Attachment#extension`, and `Context#embed_later`/`#mcp`. Ruby reads instance variables; these
//! read the matching accessor or the rendered payload. `// spec:` lines tie each test to its
//! Ruby example.

mod spec_helpers;

use std::sync::{Arc, Once};

use rust_llm::{
    Agent, Attachment, Chat, Config, EmbedOptions, ErrorKind, Fallback, FnTool, Mcp, Message,
    ProtocolName, SharedTool, ThinkingConfig, ToolCalls, ToolChoice, UsageEntry, UsageStatus,
};
use serde_json::{Value, json};
use spec_helpers::*;

/// A configuration with every provider keyed and nothing reachable: these tests only render.
fn offline() -> Arc<Config> {
    let mut c = Config::default();
    for provider in ["openai", "anthropic", "gemini"] {
        c.set(format!("{provider}_api_key"), "test");
    }
    Arc::new(c)
}

fn chat(model: &str, provider: &str) -> Chat {
    Chat::with_config(offline(), Some(model), Some(provider), false).unwrap()
}

/// `RubyLLM.chat(model: model_for(:openai, :temperature))`.
fn openai() -> Chat {
    chat("gpt-4.1-nano", "openai")
}

fn render(mut chat: Chat) -> Value {
    chat.ask_later("Hello").unwrap();
    chat.render().unwrap()
}

/// `Agent.chat` reads the global configuration, as `RubyLLM.chat` does.
fn global_keys() {
    static KEYS: Once = Once::new();
    KEYS.call_once(|| {
        rust_llm::configure(|c| {
            c.set("openai_api_key", "test");
            c.set("anthropic_api_key", "test");
        })
    });
}

fn echo_tool() -> SharedTool {
    Arc::new(FnTool::new("echo_tool", "Echoes", |_args| async {
        Ok("ok".into())
    }))
}

// ---- with_tool_options -----------------------------------------------------------------------

// spec: chat_functions_spec.rb:101 stores calls preference as :many or :one
#[test]
fn stores_the_calls_preference() {
    let chat = openai().with_tool_calls(ToolCalls::Many);
    assert_eq!(chat.tool_prefs().calls, Some(ToolCalls::Many));
    let chat = chat.with_tool_calls(ToolCalls::One);
    assert_eq!(chat.tool_prefs().calls, Some(ToolCalls::One));
    // `calls: 1` has no Rust spelling: ToolCalls is the only accepted type.
}

// spec: chat_functions_spec.rb:123 stores tool concurrency preferences
#[test]
fn stores_the_tool_concurrency_preference() {
    assert!(openai().with_tool_concurrency(true).concurrency());
}

// spec: chat_functions_spec.rb:141 clears tool concurrency preferences
#[test]
fn clears_the_tool_concurrency_preference() {
    assert!(
        !openai()
            .with_tool_concurrency(true)
            .with_tool_concurrency(false)
            .concurrency()
    );
}

// spec: chat_functions_spec.rb:210 resets choice and calls to nil and concurrency to the configured default
#[test]
fn nil_tool_options_reset_to_the_configured_defaults() {
    let mut config = (*offline()).clone();
    config.tool_concurrency = true;
    let chat = Chat::with_config(
        Arc::new(config),
        Some("gpt-4.1-nano"),
        Some("openai"),
        false,
    )
    .unwrap()
    .with_tool_choice(ToolChoice::Required)
    .unwrap()
    .with_tool_calls(ToolCalls::One)
    .with_tool_concurrency(false);

    let chat = chat
        .clear_tool_choice()
        .with_tool_calls(None)
        .with_tool_concurrency(None);

    assert_eq!(chat.tool_prefs().choice, None);
    assert_eq!(chat.tool_prefs().calls, None);
    assert!(chat.concurrency(), "falls back to config.tool_concurrency");
}

// spec: chat_options_spec.rb:17 clears the recorded preferences when given nil
#[test]
fn clears_the_recorded_tool_preferences() {
    let chat = openai()
        .with_tool_choice(ToolChoice::Auto)
        .unwrap()
        .with_tool_calls(ToolCalls::One);

    let chat = chat.clear_tool_choice().with_tool_calls(None);

    assert_eq!(
        (chat.tool_prefs().choice.clone(), chat.tool_prefs().calls),
        (None, None)
    );
    let payload = render(
        chat.with_tool(FnTool::new("lookup", "Looks up", |_a| async {
            Ok("x".into())
        })),
    );
    assert!(payload.get("tool_choice").is_none(), "{payload}");
    assert!(payload.get("parallel_tool_calls").is_none(), "{payload}");
}

// spec: chat_options_spec.rb:25 falls back to the configured concurrency when given nil
#[test]
fn nil_concurrency_falls_back_to_the_configured_default() {
    // Config::default() leaves tool_concurrency off, Ruby's nil.
    assert!(
        !openai()
            .with_tool_concurrency(true)
            .with_tool_concurrency(None)
            .concurrency()
    );
}

// ---- with_model / with_temperature / with_max_output_tokens --------------------------------

// spec: chat_functions_spec.rb:235 resets to the configured default model with nil
#[test]
fn with_default_model_returns_to_the_configured_default() {
    let config = offline();
    let chat = Chat::with_config(config.clone(), Some(MODEL), Some("anthropic"), false).unwrap();

    let chat = chat.with_default_model().unwrap();

    assert_eq!(chat.model().id, config.default_model);
}

// spec: chat_functions_spec.rb:312 clears the temperature with with_temperature(nil)
#[test]
fn with_temperature_none_clears_it() {
    let chat = openai().with_temperature(0.8).with_temperature(None);
    assert_eq!(chat.temperature(), None);
    assert!(render(chat).get("temperature").is_none());
}

// spec: chat_functions_spec.rb:344 sends the temperature you set even when the registry marks the model as rejecting it
#[test]
fn sends_temperature_even_when_the_registry_says_the_model_rejects_it() {
    let chat = chat("claude-sonnet-5", "anthropic");
    assert_eq!(
        chat.model().metadata.get("temperature"),
        Some(&json!(false))
    );

    assert_eq!(
        render(chat.with_temperature(0.5))["temperature"],
        json!(0.5)
    );
}

// spec: chat_functions_spec.rb:355 sends the temperature you set to search models
#[test]
fn sends_temperature_to_search_models() {
    let chat = chat("gpt-5-search-api", "openai")
        .with_protocol(ProtocolName::ChatCompletions)
        .with_temperature(0.7);
    assert_eq!(render(chat)["temperature"], json!(0.7));
}

// spec: chat_request_options_spec.rb:25 clears the limit with with_max_output_tokens(nil)
#[test]
fn with_max_output_tokens_none_clears_the_limit() {
    let payload = render(
        openai()
            .with_max_output_tokens(1234)
            .with_max_output_tokens(None),
    );
    assert!(payload.get("max_output_tokens").is_none(), "{payload}");
}

// ---- with_provider_options / with_headers --------------------------------------------------

// spec: chat_request_options_spec.rb:34 clears provider options with with_provider_options(nil)
#[test]
fn with_provider_options_null_clears_them() {
    let chat = openai().with_provider_options(json!({ "max_tokens": 100 }));

    let chat = chat.with_provider_options(Value::Null);

    assert_eq!(chat.provider_options(), &json!({}));
}

// spec: chat_headers_spec.rb:18 clears headers with with_headers(nil)
#[test]
fn with_headers_empty_clears_them() {
    let chat = openai().with_headers([("X-Test".to_string(), "test".to_string())]);

    let chat = chat.with_headers([]);

    assert!(chat.headers().is_empty());
}

// ---- cancellation / add_completion ----------------------------------------------------------

// spec: chat_options_spec.rb:222 consults an external cancellation checker
#[tokio::test]
async fn consults_an_external_cancellation_checker() {
    let server = serve(vec![text_response("never sent")]).await;
    let mut chat = spec_helpers::chat(&server);
    chat.set_cancellation_checker(Some(Arc::new(|| true)));

    let err = chat.ask("Hello").await.unwrap_err();

    assert!(matches!(err, rust_llm::Error::Cancelled), "{err}");
    assert_eq!(requests(&server).await, 0);
}

// spec: chat_options_spec.rb:265 leaves the ledger alone
#[test]
fn add_completion_leaves_the_ledger_alone_when_usage_is_already_recorded() {
    let mut chat = chat(MODEL, "anthropic");
    let mut response = Message::assistant("from a batch");
    response.tokens.input = Some(5);
    response.tokens.output = Some(2);
    response.usage_entries = vec![UsageEntry {
        id: UsageEntry::next_id(),
        owner: None,
        operation: rust_llm::message::Operation::Chat,
        provider: "anthropic".into(),
        model: MODEL.into(),
        status: UsageStatus::Succeeded,
        tokens: response.tokens.clone(),
        cost: Default::default(),
    }];

    chat.add_completion(response, false);

    assert!(chat.usage_entries().is_empty());
}

// ---- agents -------------------------------------------------------------------------------

/// The agent in `agent_spec.rb:17`, with its `inputs :display_name`.
struct Greeter {
    display_name: String,
}

impl Agent for Greeter {
    fn model(&self) -> Option<&str> {
        Some("gpt-4.1-nano")
    }
    fn instructions(&self) -> Option<String> {
        Some(format!("Hello {}", self.display_name))
    }
    fn tools(&self) -> Vec<SharedTool> {
        vec![echo_tool()]
    }
    fn tool_choice(&self) -> Option<ToolChoice> {
        Some(ToolChoice::Required)
    }
    fn tool_calls(&self) -> Option<ToolCalls> {
        Some(ToolCalls::One)
    }
    fn caching(&self) -> Option<Value> {
        Some(json!({ "ttl": "1h" }))
    }
    fn provider_options(&self) -> Option<Value> {
        Some(json!({ "max_tokens": 12 }))
    }
}

// spec: agent_spec.rb:17 builds a configured plain chat via .chat with runtime inputs
#[test]
fn agent_chat_applies_instructions_tools_tool_options_caching_and_provider_options() {
    global_keys();
    let chat = Greeter {
        display_name: "Ava".into(),
    }
    .chat()
    .unwrap();

    let first = &chat.messages()[0];
    assert_eq!(first.role, rust_llm::Role::System);
    assert_eq!(first.content(), "Hello Ava");
    assert!(chat.tools().iter().any(|t| t.name() == "echo_tool"));
    assert_eq!(chat.tool_prefs().choice, Some(ToolChoice::Required));
    assert_eq!(chat.tool_prefs().calls, Some(ToolCalls::One));
    assert_eq!(
        chat.caching(),
        Some(&rust_llm::Caching::On(
            json!({ "ttl": "1h" }).as_object().unwrap().clone()
        ))
    );
    assert_eq!(chat.provider_options(), &json!({ "max_tokens": 12 }));
}

struct Capped;

impl Agent for Capped {
    fn model(&self) -> Option<&str> {
        Some("gpt-4.1-nano")
    }
    fn max_output_tokens(&self) -> Option<i64> {
        Some(1000)
    }
}

// spec: agent_spec.rb:43 applies max_output_tokens from the DSL macro
#[test]
fn agent_max_output_tokens_reaches_the_payload() {
    global_keys();
    assert_eq!(
        render(Capped.chat().unwrap())["max_output_tokens"],
        json!(1000)
    );
}

struct ToolOptions;

impl Agent for ToolOptions {
    fn model(&self) -> Option<&str> {
        Some("gpt-4.1-nano")
    }
    fn tools(&self) -> Vec<SharedTool> {
        vec![echo_tool()]
    }
    fn tool_choice(&self) -> Option<ToolChoice> {
        Some(ToolChoice::Required)
    }
    fn tool_calls(&self) -> Option<ToolCalls> {
        Some(ToolCalls::One)
    }
    fn tool_concurrency(&self) -> Option<bool> {
        Some(true)
    }
}

// spec: agent_spec.rb:52 applies tool_options separately from the declared tools
#[test]
fn agent_tool_options_apply_separately_from_the_tools() {
    global_keys();
    // `concurrency: :fibers` is `true` here: the port has one concurrent mode.
    assert_eq!(
        (
            ToolOptions.tool_choice(),
            ToolOptions.tool_calls(),
            ToolOptions.tool_concurrency()
        ),
        (Some(ToolChoice::Required), Some(ToolCalls::One), Some(true))
    );

    let chat = ToolOptions.chat().unwrap();

    assert!(chat.tools().iter().any(|t| t.name() == "echo_tool"));
    assert_eq!(chat.tool_prefs().choice, Some(ToolChoice::Required));
    assert_eq!(chat.tool_prefs().calls, Some(ToolCalls::One));
    assert!(chat.concurrency());
}

struct ChatCompletionsAgent;

impl Agent for ChatCompletionsAgent {
    fn model(&self) -> Option<&str> {
        Some("gpt-5-nano")
    }
    fn protocol(&self) -> Option<ProtocolName> {
        Some(ProtocolName::ChatCompletions)
    }
}

// spec: agent_spec.rb:71 forwards the protocol model option to new chats
#[test]
fn agent_protocol_is_forwarded_to_new_chats() {
    global_keys();
    let chat = ChatCompletionsAgent.chat().unwrap();
    assert_eq!(chat.protocol(), Some(ProtocolName::ChatCompletions));
    // And it is the protocol rendered: Chat Completions sends `messages`, Responses `input`.
    assert!(render(chat).get("messages").is_some());
}

struct Thinker;

impl Agent for Thinker {
    fn model(&self) -> Option<&str> {
        Some("gpt-4.1-nano")
    }
    fn thinking(&self) -> Option<ThinkingConfig> {
        Some(ThinkingConfig::effort("low"))
    }
}

// spec: agent_spec.rb:150 exposes resolved thinking on agent instances
#[test]
fn agent_exposes_its_thinking_and_applies_it() {
    global_keys();
    assert_eq!(Thinker.thinking(), Some(ThinkingConfig::effort("low")));
    assert_eq!(
        Thinker.chat().unwrap().thinking(),
        Some(&ThinkingConfig::effort("low"))
    );
}

struct WithFallbacks;

impl Agent for WithFallbacks {
    fn model(&self) -> Option<&str> {
        Some("gpt-4.1-nano")
    }
    fn fallbacks(&self) -> Vec<Fallback> {
        vec![
            Fallback::from("gpt-4.1-mini"),
            Fallback {
                model: "claude-haiku-4-5-20251001".into(),
                provider: Some("anthropic".into()),
            },
        ]
    }
    fn fallback_errors(&self) -> Option<Vec<ErrorKind>> {
        Some(vec![ErrorKind::RateLimit])
    }
}

// spec: agent_spec.rb:484 applies class-configured fallbacks to new chats
#[test]
fn agent_fallbacks_and_their_error_classes_reach_new_chats() {
    global_keys();
    let chat = WithFallbacks.chat().unwrap();

    let ids: Vec<&str> = chat.fallbacks().iter().map(|f| f.model.as_str()).collect();
    assert_eq!(ids, ["gpt-4.1-mini", "claude-haiku-4-5-20251001"]);
    assert_eq!(
        chat.fallbacks().last().unwrap().provider.as_deref(),
        Some("anthropic")
    );
    assert_eq!(chat.fallback_errors(), [ErrorKind::RateLimit]);
}

/// The `agent_class` of `agent_dsl_spec.rb`'s configuration readers.
struct Configured;

impl Agent for Configured {
    fn model(&self) -> Option<&str> {
        Some("gpt-4.1-nano")
    }
    fn provider(&self) -> Option<&str> {
        Some("openai")
    }
    fn temperature(&self) -> Option<f64> {
        Some(0.4)
    }
    fn max_output_tokens(&self) -> Option<i64> {
        Some(128)
    }
    fn thinking(&self) -> Option<ThinkingConfig> {
        Some(ThinkingConfig::effort("low"))
    }
    fn citations(&self) -> Option<bool> {
        Some(true)
    }
    fn caching(&self) -> Option<Value> {
        Some(json!({ "ttl": "1h" }))
    }
    fn provider_options(&self) -> Option<Value> {
        Some(json!({ "top_p": 0.9 }))
    }
    fn headers(&self) -> Vec<(String, String)> {
        vec![("X-Test".into(), "1".into())]
    }
    fn end_user(&self) -> Option<String> {
        Some("tenant-42".into())
    }
    fn compaction(&self) -> Option<Value> {
        Some(json!({ "at": 50_000 }))
    }
}

// spec: agent_dsl_spec.rb:53 applies the configured options to a new chat
#[test]
fn agent_applies_the_configured_options_to_a_new_chat() {
    global_keys();
    let chat = Configured.chat().unwrap();

    assert_eq!(chat.temperature(), Some(0.4));
    assert_eq!(chat.max_output_tokens(), Some(128));
    assert!(chat.citations());
    assert_eq!(
        chat.caching(),
        Some(&rust_llm::Caching::On(
            json!({ "ttl": "1h" }).as_object().unwrap().clone()
        ))
    );
    assert_eq!(chat.provider_options(), &json!({ "top_p": 0.9 }));
    assert_eq!(chat.headers(), [("X-Test".to_string(), "1".to_string())]);
    assert_eq!(
        chat.thinking().and_then(|t| t.effort.as_deref()),
        Some("low")
    );
    assert_eq!(chat.end_user(), Some("tenant-42"));
    assert_eq!(chat.compaction(), Some(&json!({ "at": 50_000 })));
}

// ---- Attachment#extension --------------------------------------------------------------------

// spec: attachment_spec.rb:154 is nil for a filename without one
#[test]
fn extension_is_none_without_one() {
    assert_eq!(
        Attachment::from_bytes(b"x".to_vec(), "README", None).extension(),
        None
    );
}

// spec: attachment_spec.rb:158 downcases the extension
#[test]
fn extension_is_downcased() {
    assert_eq!(
        Attachment::from_bytes(b"x".to_vec(), "REPORT.PDF", None)
            .extension()
            .as_deref(),
        Some("pdf")
    );
}

// ---- Context entry points ------------------------------------------------------------------

// spec: context_entrypoints_spec.rb:15 stages embedding requests with the context
#[test]
fn context_embed_later_stages_a_request_with_the_context() {
    let mut config = (*offline()).clone();
    config.default_embedding_model = "text-embedding-3-large".into();
    let context = rust_llm::Context::new(config);

    let request = context
        .embed_later(
            "Hello",
            EmbedOptions {
                model: Some("text-embedding-3-small"),
                provider: Some("openai"),
                dimensions: Some(256),
                ..Default::default()
            },
        )
        .unwrap();

    assert_eq!(
        (&request.text, request.dimensions),
        (&rust_llm::embedding::EmbedInput::from("Hello"), Some(256))
    );
    assert_eq!(request.model().id, "text-embedding-3-small");
    // Without a model it reads the context's default, not the global one.
    let request = context
        .embed_later("Hello", EmbedOptions::default())
        .unwrap();
    assert_eq!(request.model().id, "text-embedding-3-large");
}

// spec: context_entrypoints_spec.rb:24 connects to MCP servers with the context
#[tokio::test]
async fn context_mcp_connects_with_the_context() {
    // `server/discover` is refused at once (it has its own fixed 10s timeout, as in Ruby); the
    // `initialize` handshake is slower than the context's request_timeout (1s).
    let server = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::body_string_contains("server/discover"))
        .respond_with(wiremock::ResponseTemplate::new(200).set_body_json(
            json!({ "jsonrpc": "2.0", "id": 1, "error": { "code": -32601, "message": "Method not found" } }),
        ))
        .mount(&server)
        .await;
    wiremock::Mock::given(wiremock::matchers::any())
        .respond_with(
            wiremock::ResponseTemplate::new(200).set_delay(std::time::Duration::from_secs(30)),
        )
        .mount(&server)
        .await;
    let mut config = (*offline()).clone();
    config.request_timeout = std::time::Duration::from_secs(1);
    let context = rust_llm::Context::new(config);

    let mcp = context
        .mcp(Mcp::url(format!("{}/mcp", server.uri())).prefix("docs"))
        .build()
        .unwrap();

    let started = std::time::Instant::now();
    let err = mcp.tools().await.err().expect("timed out");
    assert!(matches!(err, rust_llm::Error::Timeout(_)), "{err}");
    assert!(
        started.elapsed() < std::time::Duration::from_secs(5),
        "used the context's timeout: {:?}",
        started.elapsed()
    );
}
