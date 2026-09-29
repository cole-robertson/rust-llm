//! Chat configuration specs ported from RubyLLM 2.0: `chat_options_spec.rb` (schema
//! normalization, tool choice), `chat_headers_spec.rb`, `chat_model_aliases_spec.rb`,
//! `chat_functions_spec.rb`, and the unit parts of `chat_spec.rb`. Ruby reads instance variables;
//! these read the rendered payload or the request actually sent, which is what the variables are
//! for. `// spec:` lines tie each test to its Ruby example.

mod spec_helpers;

use async_trait::async_trait;
use rust_llm::{Chat, Error, Message, Parameter, Role, Tool, ToolCall, ToolChoice, ToolError, ToolResult};
use serde_json::{Map, Value, json};
use spec_helpers::*;

/// `model_for(:openai, :temperature)`: an OpenAI chat on the Responses API.
fn openai(server: &wiremock::MockServer) -> Chat {
    Chat::with_config(config(server), Some("gpt-4.1-nano"), Some("openai"), false).unwrap()
}

fn schema_format(chat: Chat) -> Value {
    let mut chat = chat;
    chat.ask_later("hi").unwrap();
    chat.render().unwrap()["text"]["format"].clone()
}

// spec: chat_options_spec.rb:131 reads strict off the wrapper
#[tokio::test]
async fn with_schema_reads_strict_off_the_wrapper() {
    let server = serve(vec![]).await;
    let f = schema_format(openai(&server).with_schema(json!({ "name": "Person", "schema": { "type": "object" }, "strict": false })));
    assert_eq!(f["strict"], json!(false));
}

// spec: chat_options_spec.rb:137 reads strict out of the inner schema
#[tokio::test]
async fn with_schema_reads_strict_out_of_the_inner_schema() {
    let server = serve(vec![]).await;
    let f = schema_format(openai(&server).with_schema(json!({ "name": "Person", "schema": { "type": "object", "strict": true } })));
    assert_eq!(f["strict"], json!(true));
    assert!(f["schema"].get("strict").is_none());
}

// spec: chat_options_spec.rb:151 names an unnamed schema
#[tokio::test]
async fn with_schema_names_an_unnamed_schema() {
    let server = serve(vec![]).await;
    assert_eq!(schema_format(openai(&server).with_schema(json!({ "type": "object" })))["name"], json!("response"));
}

// spec: chat_options_spec.rb:157 sanitizes an unusable schema name
#[tokio::test]
async fn with_schema_sanitizes_the_name() {
    let server = serve(vec![]).await;
    let f = schema_format(openai(&server).with_schema(json!({ "name": "Person Schema!", "schema": { "type": "object" } })));
    assert_eq!(f["name"], json!("Person_Schema_"));
}

// spec: chat_options_spec.rb:163 falls back to a generic name when nothing survives sanitizing
#[tokio::test]
async fn with_schema_falls_back_to_a_generic_name() {
    let server = serve(vec![]).await;
    let f = schema_format(openai(&server).with_schema(json!({ "name": "", "schema": { "type": "object" } })));
    assert_eq!(f["name"], json!("response"));
}

// spec: chat_options_spec.rb:169 clears the schema when given nil
#[tokio::test]
async fn with_schema_null_clears_the_schema() {
    let server = serve(vec![]).await;
    let mut chat = openai(&server).with_schema(json!({ "type": "object" })).with_schema(Value::Null);
    chat.ask_later("hi").unwrap();
    assert!(chat.render().unwrap().get("text").is_none());
}

/// `LookupTool`.
struct Lookup;

#[async_trait]
impl Tool for Lookup {
    fn description(&self) -> String {
        "Looks things up".into()
    }
    fn parameters(&self) -> Vec<Parameter> {
        vec![Parameter::new("query")]
    }
    async fn execute(&self, args: Map<String, Value>, _call: &ToolCall) -> Result<ToolResult, ToolError> {
        Ok(args["query"].clone().into())
    }
}

// spec: chat_options_spec.rb:54 accepts a tool class
// spec: chat_options_spec.rb:62 accepts a tool instance (Rust names the tool by its name)
#[tokio::test]
async fn tool_choice_accepts_a_registered_tool() {
    let server = serve(vec![]).await;
    let mut chat = openai(&server).with_tool(Lookup).with_tool_choice(ToolChoice::Tool("lookup".into())).unwrap();
    chat.ask_later("hi").unwrap();
    assert_eq!(chat.render().unwrap()["tool_choice"], json!({ "type": "function", "name": "lookup" }));
}

// spec: chat_options_spec.rb:70 rejects a tool the chat does not carry
#[tokio::test]
async fn tool_choice_rejects_a_tool_the_chat_does_not_carry() {
    let server = serve(vec![]).await;
    let err = openai(&server).with_tool_choice(ToolChoice::Tool("missing".into())).unwrap_err();
    assert!(matches!(&err, Error::InvalidToolChoice(m) if m.contains("Invalid tool choice: missing")), "{err}");
}

// spec: chat_options_spec.rb:89 accepts an attributes hash, a Message, and anything convertible
// (Rust takes a `Message`; the other shapes are Ruby coercions.)
#[tokio::test]
async fn add_message_appends_messages() {
    let server = serve(vec![]).await;
    let mut chat = openai(&server);
    chat.add_message(Message::user("hash"));
    chat.add_message(Message::user("message"));
    assert_eq!(chat.messages().iter().map(|m| m.content()).collect::<Vec<_>>(), ["hash", "message"]);
}

// spec: chat_options_spec.rb:101 accepts nothing, one message, or a list
#[tokio::test]
async fn set_messages_replaces_the_transcript() {
    let server = serve(vec![]).await;
    let mut chat = openai(&server);
    chat.set_messages(vec![]);
    assert!(chat.messages().is_empty());
    chat.set_messages(vec![Message::user("a"), Message::assistant("b")]);
    assert_eq!(chat.messages().iter().map(|m| m.content()).collect::<Vec<_>>(), ["a", "b"]);
}

// ---- chat_headers_spec.rb ---------------------------------------------------------------------

// spec: chat_headers_spec.rb:32 passes headers to provider complete method
#[tokio::test]
async fn with_headers_sends_them_with_the_request() {
    let server = serve(vec![text_response("Test response")]).await;
    let mut chat = chat(&server).with_headers([("X-Custom".to_string(), "header".to_string())]);
    chat.ask("Test").await.unwrap();
    let sent = &server.received_requests().await.unwrap()[0];
    assert_eq!(sent.headers.get("x-custom").unwrap(), "header");
}

// spec: chat_headers_spec.rb:60 user headers do not override provider headers
#[tokio::test]
async fn user_headers_do_not_override_provider_headers() {
    let server = serve(vec![text_response("Test")]).await;
    let mut chat = chat(&server).with_headers([
        ("x-api-key".to_string(), "user-key".to_string()),
        ("X-Custom".to_string(), "user-value".to_string()),
    ]);
    chat.ask("Test").await.unwrap();
    let sent = &server.received_requests().await.unwrap()[0];
    let keys: Vec<_> = sent.headers.get_all("x-api-key").iter().map(|v| v.to_str().unwrap().to_string()).collect();
    assert_eq!(keys, ["test"], "the provider's key wins, and is sent once");
    assert_eq!(sent.headers.get("x-custom").unwrap(), "user-value");
}

// ---- chat_model_aliases_spec.rb ---------------------------------------------------------------

// spec: chat_model_aliases_spec.rb:8 finds models by alias name
// spec: chat_model_aliases_spec.rb:15 still supports exact model IDs
#[tokio::test]
async fn chats_find_models_by_alias_or_exact_id() {
    let server = serve(vec![]).await;
    for id in ["claude-haiku-4-5", "claude-haiku-4-5-20251001"] {
        let chat = Chat::with_config(config(&server), Some(id), None, false).unwrap();
        assert_eq!((chat.model().id.as_str(), chat.provider().slug()), (id, "anthropic"));
    }
}

// spec: chat_model_aliases_spec.rb:53 resolves xAI provider aliases
#[tokio::test]
async fn chats_resolve_provider_aliases() {
    let server = serve(vec![]).await;
    let mut c = (*config(&server)).clone();
    c.set("xai_api_key", "test");
    let chat = Chat::with_config(std::sync::Arc::new(c), Some("grok-4-1-fast-non-reasoning"), Some("xai"), false).unwrap();
    assert_eq!((chat.model().id.as_str(), chat.provider().slug()), ("grok-4.3", "xai"));
}

// ---- chat_spec.rb (unit parts) ----------------------------------------------------------------

// spec: chat_spec.rb:119 keeps manually added messages out of the conversation totals
#[tokio::test]
async fn manually_added_messages_stay_out_of_chat_totals() {
    let server = serve(vec![]).await;
    let mut chat = chat(&server);
    chat.add_message(Message::user("Hello"));
    let mut hi = Message::assistant("Hi");
    hi.tokens.input = Some(1_000);
    hi.tokens.output = Some(2_000);
    let response = chat.add_message(hi).clone();
    assert_eq!((response.tokens().input, response.tokens().output), (Some(1_000), Some(2_000)));
    assert!(chat.tokens().is_empty());
    assert_eq!(chat.cost().total(), None);
}

// spec: chat_spec.rb:134 returns empty value objects before the chat has usage
#[tokio::test]
async fn a_new_chat_has_empty_tokens_and_no_cost() {
    let server = serve(vec![]).await;
    let chat = chat(&server);
    assert!(chat.tokens().is_empty());
    assert_eq!(chat.cost().total(), None);
}

// spec: chat_spec.rb:143 prices a response against a given model when its own model id cannot be resolved
#[tokio::test]
async fn a_response_prices_against_a_given_model() {
    let server = serve(vec![]).await;
    let chat = chat(&server);
    let mut hi = Message::assistant("Hi");
    hi.tokens.input = Some(1_000_000);
    hi.model = Some("provider-backend-version".into());
    assert_eq!(hi.cost(None).total(), None, "unknown id prices nothing");
    assert!(hi.cost(Some(chat.model())).total().is_some_and(|t| t > 0.0));
}

// ---- chat_functions_spec.rb -------------------------------------------------------------------

// spec: chat_functions_spec.rb:9 adds a single tool regardless of model capabilities
// spec: chat_functions_spec.rb:21 adds multiple tools at once
#[tokio::test]
async fn with_tools_adds_tools() {
    let server = serve(vec![]).await;
    let chat = openai(&server).with_tool(Lookup);
    assert_eq!(chat.tools().iter().map(|t| t.name()).collect::<Vec<_>>(), ["lookup"]);
}

// spec: chat_functions_spec.rb:84 clears the tools while leaving the tool options unchanged
#[tokio::test]
async fn clear_tools_leaves_tool_options() {
    let server = serve(vec![]).await;
    let mut chat = openai(&server).with_tool(Lookup).with_tool_calls(rust_llm::ToolCalls::One);
    chat.clear_tools();
    assert!(chat.tools().is_empty());
    chat.ask_later("hi").unwrap();
    assert_eq!(chat.render().unwrap().get("parallel_tool_calls"), None, "no tools, no tool controls");
}

// spec: chat_functions_spec.rb:245 replaces existing system instructions by default
// spec: chat_functions_spec.rb:256 appends system instructions when append: true
// spec: chat_functions_spec.rb:293 clears system instructions with with_instructions(nil)
#[tokio::test]
async fn with_instructions_replaces_by_default_and_appends_on_request() {
    let server = serve(vec![]).await;
    let mut chat = openai(&server).with_instructions("one").with_instructions("two");
    assert_eq!(chat.messages().iter().filter(|m| m.role == Role::System).map(|m| m.content()).collect::<Vec<_>>(), ["two"]);
    chat.set_instructions(Some("three".into()), true, false);
    assert_eq!(chat.messages().iter().filter(|m| m.role == Role::System).map(|m| m.content()).collect::<Vec<_>>(), ["two", "three"]);
    chat.set_instructions(None, false, false);
    assert!(chat.messages().iter().all(|m| m.role != Role::System));
}

// spec: chat_functions_spec.rb:320 omits temperature when you never set one
// spec: chat_functions_spec.rb:335 sends the temperature you set to the Responses API untouched
#[tokio::test]
async fn temperature_is_sent_only_when_set() {
    let server = serve(vec![]).await;
    let mut chat = openai(&server);
    chat.ask_later("hi").unwrap();
    assert!(chat.render().unwrap().get("temperature").is_none());
    let chat = chat.with_temperature(0.7);
    assert_eq!(chat.render().unwrap()["temperature"], json!(0.7));
}

// ---- chat_request_options_spec.rb -------------------------------------------------------------

// spec: chat_request_options_spec.rb:12 maps to #{config[:key]} for #{provider}
// spec: chat_request_options_spec.rb:19 maps to generationConfig.maxOutputTokens for gemini
#[tokio::test]
async fn max_output_tokens_maps_to_each_providers_key() {
    let server = serve(vec![]).await;
    for (model, provider, pointer) in [
        ("gpt-4.1-nano", "openai", "/max_output_tokens"),
        ("claude-haiku-4-5", "anthropic", "/max_tokens"),
        ("deepseek-v4-flash", "deepseek", "/max_tokens"),
        ("gemini-2.5-flash", "gemini", "/generationConfig/maxOutputTokens"),
    ] {
        let mut chat = Chat::with_config(config(&server), Some(model), Some(provider), false).unwrap().with_max_output_tokens(1234);
        chat.ask_later("hi").unwrap();
        assert_eq!(chat.render().unwrap().pointer(pointer), Some(&json!(1234)), "{provider}");
    }
}

// spec: chat_functions_spec.rb:380 resets @protocol to nil when with_model is called without a protocol
#[tokio::test]
async fn with_model_drops_an_explicit_protocol() {
    let server = serve(vec![]).await;
    let chat = openai(&server).with_protocol(rust_llm::ProtocolName::ChatCompletions);
    let mut staged = openai(&server).with_protocol(rust_llm::ProtocolName::ChatCompletions);
    staged.ask_later("hi").unwrap();
    assert!(staged.render().unwrap().get("messages").is_some(), "Chat Completions while overridden");
    let mut chat = chat.with_model("gpt-4.1-nano", Some("openai")).unwrap();
    chat.ask_later("hi").unwrap();
    assert!(chat.render().unwrap().get("input").is_some(), "back to the default Responses protocol");
}

// spec: chat_functions_spec.rb:175 uses the configured tool concurrency by default
// spec: chat_functions_spec.rb:197 allows chats to override configured tool concurrency
#[tokio::test]
async fn tool_concurrency_defaults_to_the_config_and_chats_override_it() {
    let server = serve(vec![]).await;
    let mut c = (*config(&server)).clone();
    c.tool_concurrency = true;
    let chat = Chat::with_config(std::sync::Arc::new(c), Some("gpt-4.1-nano"), Some("openai"), false).unwrap();
    assert!(chat.concurrency());
    assert!(!chat.with_tool_concurrency(false).concurrency());
}

/// A named no-op tool for tool-set bookkeeping tests.
struct Named(&'static str);

#[async_trait]
impl Tool for Named {
    fn name(&self) -> String {
        self.0.into()
    }
    fn description(&self) -> String {
        String::new()
    }
    async fn execute(&self, _a: Map<String, Value>, _c: &ToolCall) -> Result<ToolResult, ToolError> {
        Ok("".into())
    }
}

// spec: chat_functions_spec.rb:60 replaces all tools when followed by with_tools
#[tokio::test]
async fn clearing_then_adding_tools_replaces_the_set() {
    let server = serve(vec![]).await;
    let mut chat = openai(&server).with_tool(Named("tool1")).with_tool(Named("tool2"));
    assert_eq!(chat.tools().len(), 2);
    chat.clear_tools();
    let chat = chat.with_tool(Named("tool3"));
    assert_eq!(chat.tools().iter().map(|t| t.name()).collect::<Vec<_>>(), ["tool3"]);
}

// spec: chat_functions_spec.rb:274 keeps system instructions in chronological message history
#[tokio::test]
async fn instructions_are_appended_in_chronological_order() {
    let server = serve(vec![]).await;
    let mut chat = openai(&server);
    chat.add_message(Message::user("Hi"));
    chat.add_message(Message::assistant("Hello"));
    let chat = chat.with_instructions("System");
    assert_eq!(chat.messages().iter().map(|m| m.role).collect::<Vec<_>>(), [Role::User, Role::Assistant, Role::System]);
}
