//! Tool search with deferred tools (RubyLLM 2.1): `chat_deferred_tools_spec.rb`,
//! `chat_tool_search_spec.rb` (replayed from its cassettes), `chat_tool_search_model_switch_spec.rb`,
//! `protocols/{anthropic,responses}/tools_tool_search_spec.rb`, and `tool_deferred_spec.rb`.

#[macro_use]
mod support;
mod spec_helpers;

use std::sync::Arc;

use async_trait::async_trait;
use rust_llm::message::indexmap_lite::IndexMap;
use rust_llm::protocols::{anthropic, responses};
use rust_llm::tool::Deferred;
use rust_llm::tool_search;
use rust_llm::{
    Chat, Message, Parameter, ProviderTool, Role, SharedTool, Tool, ToolCall, ToolError,
    ToolResult, UsageEntry,
};
use serde_json::{Map, Value, json};

// ---- tools ---------------------------------------------------------------------------------

/// `RegularTool`: `description 'plain tool'`.
struct RegularTool;

#[async_trait]
impl Tool for RegularTool {
    fn description(&self) -> String {
        "plain tool".into()
    }
    async fn execute(&self, _: Map<String, Value>, _: &ToolCall) -> Result<ToolResult, ToolError> {
        Ok("regular ran".into())
    }
}

/// `HeavyTool`: `description 'heavy deferred tool'; defer`.
struct HeavyTool;

#[async_trait]
impl Tool for HeavyTool {
    fn description(&self) -> String {
        "heavy deferred tool".into()
    }
    fn is_deferred(&self) -> bool {
        true
    }
    async fn execute(&self, _: Map<String, Value>, _: &ToolCall) -> Result<ToolResult, ToolError> {
        Ok("heavy ran".into())
    }
}

fn regular() -> SharedTool {
    Arc::new(RegularTool)
}

fn heavy() -> SharedTool {
    Arc::new(HeavyTool)
}

fn names(tools: &[SharedTool]) -> Vec<String> {
    tools.iter().map(|t| t.name()).collect()
}

/// `chat.render[:tools].map { |tool| tool.slice(:name, :type, :defer_loading) }`.
fn rendered_tools(chat: &Chat) -> Vec<Value> {
    let payload = chat.render().expect("render");
    payload["tools"]
        .as_array()
        .cloned()
        .unwrap_or_default()
        .iter()
        .map(|t| {
            let mut sliced = Map::new();
            for key in ["name", "type", "defer_loading"] {
                if let Some(v) = t.get(key) {
                    sliced.insert(key.into(), v.clone());
                }
            }
            Value::Object(sliced)
        })
        .collect()
}

async fn chat_on(provider: &str, model: &str) -> (wiremock::MockServer, Chat) {
    let server = spec_helpers::serve(vec![]).await;
    let config = spec_helpers::config(&server);
    let chat = Chat::with_config(config, Some(model), Some(provider), false).expect("chat");
    (server, chat)
}

// ---- tool_deferred_spec.rb -----------------------------------------------------------------

// spec: tool_deferred_spec.rb:7 .defer > is off by default
#[test]
fn defer_is_off_by_default() {
    assert!(!RegularTool.is_deferred());
}

/// A tool type that inherits `HeavyTool`'s declaration, as a Ruby subclass of a deferred class does.
struct HeavierTool(HeavyTool);

#[async_trait]
impl Tool for HeavierTool {
    fn description(&self) -> String {
        self.0.description()
    }
    fn is_deferred(&self) -> bool {
        self.0.is_deferred()
    }
    async fn execute(&self, a: Map<String, Value>, c: &ToolCall) -> Result<ToolResult, ToolError> {
        self.0.execute(a, c).await
    }
}

// spec: tool_deferred_spec.rb:11 .defer > defers the class and its subclasses only
#[test]
fn defer_marks_the_tool_and_what_inherits_it_only() {
    assert!(HeavyTool.is_deferred());
    assert!(HeavierTool(HeavyTool).is_deferred());
    assert!(!RegularTool.is_deferred());
}

// ---- chat_deferred_tools_spec.rb -----------------------------------------------------------

// spec: chat_deferred_tools_spec.rb:24 #deferred_tools > returns the tools declared with Tool.defer
#[tokio::test]
async fn deferred_tools_returns_the_tools_declared_deferred() {
    let (_s, chat) = chat_on("anthropic", spec_helpers::MODEL).await;
    let chat = chat.with_tools([regular(), heavy()]);

    assert_eq!(names(chat.tools()), ["regular", "heavy"]);
    let deferred = chat.deferred_tools();
    assert_eq!(names(&deferred), ["heavy"]);
    assert!(Arc::ptr_eq(&deferred[0], &chat.tools()[1]));
}

// spec: chat_deferred_tools_spec.rb:32 #deferred_tools > defers every tool registered with defer: true
#[tokio::test]
async fn defer_true_defers_every_tool() {
    let (_s, chat) = chat_on("anthropic", spec_helpers::MODEL).await;
    let chat = chat.with_deferred_tools([regular(), heavy()]);
    assert_eq!(names(&chat.deferred_tools()), ["regular", "heavy"]);
}

// spec: chat_deferred_tools_spec.rb:38 #deferred_tools > offers a deferred class up front with defer: false
#[tokio::test]
async fn defer_false_offers_a_deferred_tool_up_front() {
    let (_s, mut chat) = chat_on("anthropic", spec_helpers::MODEL).await;
    chat.add_tool_deferred(heavy(), Some(false));
    assert!(chat.deferred_tools().is_empty());
}

// spec: chat_deferred_tools_spec.rb:44 #deferred_tools > follows the latest registration of a name
#[tokio::test]
async fn the_latest_registration_of_a_name_wins() {
    let (_s, chat) = chat_on("anthropic", spec_helpers::MODEL).await;
    let chat = chat
        .with_deferred_tools([regular()])
        .with_tools([regular()]);
    assert!(chat.deferred_tools().is_empty());
}

// spec: chat_deferred_tools_spec.rb:50 #deferred_tools > forgets deferrals when the tools are cleared
#[tokio::test]
async fn clearing_tools_forgets_deferrals() {
    let (_s, mut chat) = chat_on("anthropic", spec_helpers::MODEL).await;
    chat.add_tool_deferred(regular(), Some(true));
    chat.clear_tools();
    chat.add_tool(regular());
    assert!(chat.deferred_tools().is_empty());
}

// spec: chat_deferred_tools_spec.rb:58 requests > marks deferred tools and adds the search tool
#[tokio::test]
async fn requests_mark_deferred_tools_and_add_the_search_tool() {
    let (_s, chat) = chat_on("anthropic", spec_helpers::MODEL).await;
    let chat = chat.with_tools([regular(), heavy()]);
    assert_eq!(
        rendered_tools(&chat),
        [
            json!({ "name": "regular" }),
            json!({ "name": "heavy", "defer_loading": true }),
            json!({ "name": "tool_search_tool_bm25", "type": "tool_search_tool_bm25_20251119" }),
        ]
    );
}

// spec: chat_deferred_tools_spec.rb:65 requests > does not gate deferral on the model catalog
#[tokio::test]
async fn deferral_does_not_read_model_capabilities() {
    // Ruby renders through Vertex AI (not ported) with `supports?` raising; any model the
    // protocol serves gets `defer_loading`, whatever the catalog says, as here with a model the
    // registry does not know.
    let server = spec_helpers::serve(vec![]).await;
    let chat = Chat::with_config(
        spec_helpers::config(&server),
        Some("claude-not-in-the-catalog"),
        Some("anthropic"),
        true,
    )
    .expect("chat")
    .with_tools([heavy()]);
    assert_eq!(rendered_tools(&chat)[0]["defer_loading"], json!(true));
}

// spec: chat_deferred_tools_spec.rb:72 requests > sends deferred tools as ordinary tools on a protocol without tool search
#[tokio::test]
async fn protocols_without_tool_search_send_ordinary_tools() {
    let (_s, chat) = chat_on("gemini", "gemini-2.5-flash").await;
    let chat = chat.with_tools([heavy()]);
    assert!(!chat.render().unwrap().to_string().contains("defer_loading"));
    assert_eq!(names(&chat.deferred_tools()), ["heavy"]);
}

// spec: chat_deferred_tools_spec.rb:79 requests > keeps a search tool configured through provider tools instead of adding another
#[tokio::test]
async fn a_configured_search_tool_is_kept() {
    let (_s, chat) = chat_on("anthropic", spec_helpers::MODEL).await;
    let regex =
        json!({ "type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex" });
    let chat = chat
        .with_tools([heavy()])
        .with_provider_tools([ProviderTool::Raw(regex.clone())]);
    let tools = rendered_tools(&chat);
    assert_eq!(tools.last(), Some(&regex));
    assert_eq!(
        tools
            .iter()
            .filter(|t| t["type"]
                .as_str()
                .is_some_and(|k| k.starts_with("tool_search")))
            .count(),
        1
    );
}

// spec: chat_deferred_tools_spec.rb:88 executes a deferred tool the model calls
#[tokio::test]
async fn a_deferred_tool_the_model_calls_executes() {
    let server = spec_helpers::serve(vec![
        spec_helpers::tool_use_response(&[("t1", "heavy", json!({}))]),
        spec_helpers::text_response("done"),
    ])
    .await;
    let mut chat = spec_helpers::chat(&server).with_tools([heavy()]);
    chat.ask("go").await.expect("ask");
    let result = chat
        .messages()
        .iter()
        .find(|m| m.role == Role::Tool)
        .expect("tool result");
    assert_eq!(result.content.as_deref(), Some("heavy ran"));
}

// ---- chat_tool_search_spec.rb (cassettes) ---------------------------------------------------

struct WeatherLookup;

#[async_trait]
impl Tool for WeatherLookup {
    fn description(&self) -> String {
        "Looks up the current weather for a city".into()
    }
    fn parameters(&self) -> Vec<Parameter> {
        vec![Parameter::new("city").description("City name")]
    }
    async fn execute(&self, a: Map<String, Value>, _: &ToolCall) -> Result<ToolResult, ToolError> {
        Ok(format!("Sunny and 22°C in {}", a["city"].as_str().unwrap_or("")).into())
    }
}

struct StockPrice;

#[async_trait]
impl Tool for StockPrice {
    fn description(&self) -> String {
        "Looks up the current price of a stock ticker".into()
    }
    fn parameters(&self) -> Vec<Parameter> {
        vec![Parameter::new("ticker").description("Ticker symbol")]
    }
    async fn execute(&self, a: Map<String, Value>, _: &ToolCall) -> Result<ToolResult, ToolError> {
        Ok(format!("{} is trading at 100", a["ticker"].as_str().unwrap_or("")).into())
    }
}

const TOOL_SEARCH_MODELS: &[(&str, &str)] =
    &[("anthropic", "claude-haiku-4-5"), ("openai", "gpt-5.4")];

fn deferred_weather_tools(chat: Chat) -> Chat {
    chat.with_deferred_tools([
        Arc::new(WeatherLookup) as SharedTool,
        Arc::new(StockPrice) as SharedTool,
    ])
}

/// `searched?`: a search result came back in some message's raw content.
fn searched(chat: &Chat) -> bool {
    chat.messages()
        .iter()
        .filter_map(|m| m.raw_content.as_ref()?.as_array())
        .flatten()
        .any(|i| {
            matches!(
                i["type"].as_str(),
                Some("tool_search_tool_result" | "tool_search_output")
            )
        })
}

fn called_tools(chat: &Chat) -> Vec<String> {
    chat.messages()
        .iter()
        .filter_map(|m| m.tool_calls.as_ref())
        .flat_map(|c| c.values().map(|c| c.name.clone()).collect::<Vec<_>>())
        .collect()
}

fn check(ok: bool, what: impl Into<String>) -> Result<(), String> {
    if ok { Ok(()) } else { Err(what.into()) }
}

// spec: chat_tool_search_spec.rb:40 deferred tools > loads a deferred tool through tool search and calls it
#[tokio::test]
async fn loads_a_deferred_tool_through_tool_search_and_calls_it() {
    each_model!(
        TOOL_SEARCH_MODELS,
        "chat deferred tools with",
        "loads a deferred tool through tool search and calls it",
        |chat, provider, model| {
            chat = deferred_weather_tools(chat);
            let response = chat
                .ask("What is the weather in Berlin right now? Use your tools.")
                .await
                .map_err(|e| e.to_string())?;
            check(
                response.content().contains("22"),
                format!("content {:?}", response.content),
            )?;
            check(searched(&chat), "no search in the history")?;
            check(
                called_tools(&chat).contains(&"weather_lookup".to_string()),
                "no call",
            )
        }
    );
}

// spec: chat_tool_search_spec.rb:48 deferred tools > keeps using a loaded tool on the next turn
#[tokio::test]
async fn keeps_using_a_loaded_tool_on_the_next_turn() {
    each_model!(
        TOOL_SEARCH_MODELS,
        "chat deferred tools with",
        "keeps using a loaded tool on the next turn",
        |chat, provider, model| {
            chat = deferred_weather_tools(chat);
            chat.ask("What is the weather in Berlin right now? Use your tools.")
                .await
                .map_err(|e| e.to_string())?;
            let response = chat.ask("And in Paris?").await.map_err(|e| e.to_string())?;
            check(
                response.content().contains("22"),
                format!("content {:?}", response.content),
            )?;
            let calls = called_tools(&chat)
                .iter()
                .filter(|n| *n == "weather_lookup")
                .count();
            check(calls == 2, format!("{calls} weather calls"))
        }
    );
}

// spec: chat_tool_search_spec.rb:57 deferred tools > loads a deferred tool while streaming
#[tokio::test]
async fn loads_a_deferred_tool_while_streaming() {
    each_model!(
        TOOL_SEARCH_MODELS,
        "chat deferred tools with",
        "loads a deferred tool while streaming",
        |chat, provider, model| {
            chat = deferred_weather_tools(chat);
            let mut chunks = 0;
            let response = chat
                .ask_stream(
                    "What is the weather in Berlin right now? Use your tools.",
                    |_| chunks += 1,
                )
                .await
                .map_err(|e| e.to_string())?;
            check(chunks > 0, "no chunks")?;
            check(
                response.content().contains("22"),
                format!("content {:?}", response.content),
            )?;
            check(searched(&chat), "no search in the history")?;
            check(
                called_tools(&chat).contains(&"weather_lookup".to_string()),
                "no call",
            )
        }
    );
}

// ---- chat_tool_search_model_switch_spec.rb --------------------------------------------------

fn produced_by(provider: &str, model: &str, raw_content: Value) -> Message {
    let call = ToolCall::new(
        "call_1",
        "weather_lookup",
        spec_helpers::args(json!({ "city": "Berlin" })),
    );
    let mut m = Message::new(Role::Assistant, None);
    m.raw_content = Some(raw_content);
    let calls: IndexMap<ToolCall> = [("call_1".to_string(), call)].into_iter().collect();
    m.tool_calls = Some(calls);
    m.usage_entries = vec![UsageEntry {
        id: UsageEntry::next_id(),
        owner: None,
        operation: rust_llm::message::Operation::Chat,
        provider: provider.into(),
        model: model.into(),
        status: rust_llm::UsageStatus::Succeeded,
        tokens: rust_llm::Tokens::default(),
        cost: rust_llm::Cost::default(),
    }];
    m
}

async fn render_switch(provider: &str, model: &str, message: Message) -> Value {
    let (_s, chat) = chat_on(provider, model).await;
    let mut chat = chat.with_deferred_tools([Arc::new(WeatherLookup) as SharedTool]);
    chat.add_message(Message::user("What is the weather in Berlin?"));
    chat.add_message(message);
    let mut result = Message::new(Role::Tool, Some("Sunny".into()));
    result.tool_call_id = Some("call_1".into());
    chat.add_message(result);
    chat.add_message(Message::user("And in Paris?"));
    chat.render().expect("render")
}

fn types(items: &Value) -> Vec<String> {
    items
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|i| i["type"].as_str().map(str::to_string))
        .collect()
}

fn responses_search() -> Vec<Value> {
    vec![
        json!({ "type": "tool_search_call", "id": "tsc_1", "arguments": { "query": "weather" } }),
        json!({ "type": "tool_search_output", "id": "tso_1", "tools": [{ "name": "weather_lookup" }] }),
    ]
}

fn function_call() -> Value {
    json!({ "type": "function_call", "call_id": "call_1", "name": "weather_lookup",
            "arguments": "{\"city\":\"Berlin\"}", "namespace": "weather_lookup" })
}

fn reasoning() -> Value {
    json!({ "type": "reasoning", "id": "rs_1", "encrypted_content": "opaque", "summary": [] })
}

async fn responses_input(model: &str, raw: Vec<Value>) -> Value {
    render_switch(
        "openai",
        model,
        produced_by("openai", "gpt-5.4", Value::Array(raw)),
    )
    .await["input"]
        .clone()
}

// spec: chat_tool_search_model_switch_spec.rb:52 with OpenAI Responses > replays the search and the namespaced call to another model, without the reasoning
#[tokio::test]
async fn responses_replays_the_search_to_another_model_without_reasoning() {
    let mut raw = vec![reasoning()];
    raw.extend(responses_search());
    raw.push(function_call());
    let replayed = responses_input("gpt-5.4-mini", raw).await;
    assert_eq!(
        types(&replayed),
        [
            "tool_search_call",
            "tool_search_output",
            "function_call",
            "function_call_output"
        ]
    );
    assert!(replayed.as_array().unwrap().contains(&function_call()));
}

// spec: chat_tool_search_model_switch_spec.rb:59 with OpenAI Responses > replays the reasoning too to the model that produced it
#[tokio::test]
async fn responses_replays_the_reasoning_to_its_own_model() {
    let mut raw = vec![reasoning()];
    raw.extend(responses_search());
    raw.push(function_call());
    assert_eq!(
        types(&responses_input("gpt-5.4", raw).await),
        [
            "reasoning",
            "tool_search_call",
            "tool_search_output",
            "function_call",
            "function_call_output"
        ]
    );
}

// spec: chat_tool_search_model_switch_spec.rb:64 with OpenAI Responses > replays a turn that also used another server tool without its native content
#[tokio::test]
async fn responses_drops_a_turn_with_another_server_tool_to_portable_items() {
    let mut raw = vec![json!({ "type": "web_search_call", "id": "ws_1" })];
    raw.extend(responses_search());
    raw.push(function_call());
    let replayed = responses_input("gpt-5.4-mini", raw).await;
    assert_eq!(types(&replayed), ["function_call", "function_call_output"]);
    let call = replayed
        .as_array()
        .unwrap()
        .iter()
        .find(|i| i["type"] == "function_call")
        .unwrap();
    assert!(call.get("namespace").is_none());
}

fn anthropic_search() -> Vec<Value> {
    vec![
        json!({ "type": "server_tool_use", "id": "srvtoolu_1", "name": "tool_search_tool_bm25",
                "input": { "query": "weather" } }),
        json!({ "type": "tool_search_tool_result", "tool_use_id": "srvtoolu_1",
                "content": { "type": "tool_search_tool_search_result",
                             "tool_references": [{ "type": "tool_reference", "tool_name": "weather_lookup" }] } }),
    ]
}

fn tool_use() -> Value {
    json!({ "type": "tool_use", "id": "call_1", "name": "weather_lookup", "input": { "city": "Berlin" } })
}

async fn replayed_turn(model: &str, raw: Vec<Value>) -> Value {
    let payload = render_switch(
        "anthropic",
        model,
        produced_by("anthropic", "claude-haiku-4-5", Value::Array(raw)),
    )
    .await;
    payload["messages"][1]["content"].clone()
}

// spec: chat_tool_search_model_switch_spec.rb:89 with Anthropic > replays the search and the call to another model, without the thinking
#[tokio::test]
async fn anthropic_replays_the_search_to_another_model_without_thinking() {
    let mut raw =
        vec![json!({ "type": "thinking", "thinking": "Checking.", "signature": "opaque" })];
    raw.extend(anthropic_search());
    raw.push(tool_use());
    assert_eq!(
        types(&replayed_turn("claude-sonnet-4-5", raw).await),
        ["server_tool_use", "tool_search_tool_result", "tool_use"]
    );
}

// spec: chat_tool_search_model_switch_spec.rb:94 with Anthropic > replays the thinking too to the model that produced it
#[tokio::test]
async fn anthropic_replays_the_thinking_to_its_own_model() {
    let mut raw =
        vec![json!({ "type": "thinking", "thinking": "Checking.", "signature": "opaque" })];
    raw.extend(anthropic_search());
    raw.push(tool_use());
    assert_eq!(
        types(&replayed_turn("claude-haiku-4-5", raw).await),
        [
            "thinking",
            "server_tool_use",
            "tool_search_tool_result",
            "tool_use"
        ]
    );
}

// spec: chat_tool_search_model_switch_spec.rb:99 with Anthropic > replays a turn that also used another server tool without its native content
#[tokio::test]
async fn anthropic_drops_a_turn_with_another_server_tool_to_portable_blocks() {
    let mut raw = vec![
        json!({ "type": "server_tool_use", "id": "srvtoolu_2", "name": "web_search", "input": {} }),
    ];
    raw.extend(anthropic_search());
    raw.push(tool_use());
    assert_eq!(
        types(&replayed_turn("claude-sonnet-4-5", raw).await),
        ["tool_use"]
    );
}

// ---- protocols/anthropic/tools_tool_search_spec.rb ------------------------------------------

/// The spec's `instance_double(RubyLLM::Tool, name:, description: "#{name} desc", parameters_schema: nil)`.
struct Named(String, Map<String, Value>);

#[async_trait]
impl Tool for Named {
    fn name(&self) -> String {
        self.0.clone()
    }
    fn description(&self) -> String {
        format!("{} desc", self.0)
    }
    fn provider_options(&self) -> Map<String, Value> {
        self.1.clone()
    }
    async fn execute(&self, _: Map<String, Value>, _: &ToolCall) -> Result<ToolResult, ToolError> {
        Ok("".into())
    }
}

fn tool(name: &str, deferred: bool) -> SharedTool {
    tool_with(name, deferred, Map::new())
}

fn tool_with(name: &str, deferred: bool, options: Map<String, Value>) -> SharedTool {
    let base: SharedTool = Arc::new(Named(name.into(), options));
    if deferred {
        Arc::new(Deferred(base))
    } else {
        base
    }
}

fn search_use() -> Value {
    json!({ "type": "server_tool_use", "id": "srv_1", "name": "tool_search_tool_bm25",
            "input": { "query": "weather" } })
}

fn search_result() -> Value {
    json!({ "type": "tool_search_tool_result", "tool_use_id": "srv_1",
            "content": { "type": "tool_search_tool_search_result",
                         "tool_references": [{ "type": "tool_reference", "tool_name": "weather_lookup" }] } })
}

// spec: protocols/anthropic/tools_tool_search_spec.rb:24 .function_for > omits defer_loading for a bare tool even if its class is deferred
#[test]
fn anthropic_function_for_omits_defer_loading_for_a_bare_tool() {
    assert!(
        anthropic::function_for(heavy().as_ref())
            .get("defer_loading")
            .is_none()
    );
    assert!(
        anthropic::function_for(tool("a", false).as_ref())
            .get("defer_loading")
            .is_none()
    );
}

// spec: protocols/anthropic/tools_tool_search_spec.rb:28 .function_for > emits defer_loading: true for a deferred tool
#[test]
fn anthropic_function_for_marks_a_deferred_tool() {
    assert_eq!(
        anthropic::function_for(tool("a", true).as_ref())["defer_loading"],
        json!(true)
    );
}

// spec: protocols/anthropic/tools_tool_search_spec.rb:32 .function_for > renders cache_control alongside defer_loading and leaves the combination to Anthropic
#[test]
fn anthropic_function_for_keeps_cache_control_with_defer_loading() {
    let options = spec_helpers::args(json!({ "cache_control": { "type": "ephemeral" } }));
    let declaration = anthropic::function_for(tool_with("a", true, options).as_ref());
    assert_eq!(declaration["defer_loading"], json!(true));
    assert_eq!(declaration["cache_control"], json!({ "type": "ephemeral" }));
}

fn anthropic_payload_tools(tools: &[SharedTool]) -> Vec<Value> {
    let mut payload = json!({
        "tools": tools.iter().map(|t| anthropic::function_for(t.as_ref())).collect::<Vec<_>>(),
        "messages": []
    });
    tool_search::apply_anthropic(&mut payload);
    payload["tools"].as_array().cloned().unwrap_or_default()
}

// spec: protocols/anthropic/tools_tool_search_spec.rb:45 .apply_tool_search > does not add the search tool when nothing is deferred
#[test]
fn anthropic_apply_tool_search_adds_nothing_without_deferred_tools() {
    let tools = anthropic_payload_tools(&[tool("a", false)]);
    assert_eq!(
        tools.iter().map(|t| t["name"].clone()).collect::<Vec<_>>(),
        [json!("a")]
    );
}

// spec: protocols/anthropic/tools_tool_search_spec.rb:49 .apply_tool_search > adds the BM25 search tool once when any tool is deferred
#[test]
fn anthropic_apply_tool_search_adds_bm25_once() {
    let tools = anthropic_payload_tools(&[tool("a", false), tool("b", true)]);
    assert_eq!(
        tools.last(),
        Some(&tool_search::anthropic_native_tool_search())
    );
    assert_eq!(
        tools
            .iter()
            .filter(|t| t["type"] == "tool_search_tool_bm25_20251119")
            .count(),
        1
    );
}

// spec: protocols/anthropic/tools_tool_search_spec.rb:58 .tool_search_block? > matches the BM25 server_tool_use and its result, but not other server tools
#[test]
fn anthropic_tool_search_block_matches_only_the_search() {
    assert!(tool_search::is_tool_search_block(&search_use()));
    assert!(tool_search::is_tool_search_block(&search_result()));
    let mut regex = search_use();
    regex["name"] = json!("tool_search_tool_regex");
    assert!(tool_search::is_tool_search_block(&regex));
    assert!(!tool_search::is_tool_search_block(
        &json!({ "type": "server_tool_use", "name": "web_search" })
    ));
    assert!(!tool_search::is_tool_search_block(
        &json!({ "type": "text", "text": "hi" })
    ));
}

fn anthropic_parse(blocks: Value) -> Message {
    let body = json!({ "model": "claude-haiku-4-5", "content": blocks,
                       "usage": { "input_tokens": 1, "output_tokens": 1 }, "stop_reason": "tool_use" });
    anthropic::parse_completion_body(&body, rust_llm::message::RawResponse::default()).unwrap()
}

// spec: protocols/anthropic/tools_tool_search_spec.rb:74 parse_completion_body with tool search > keeps the raw blocks for replay
#[test]
fn anthropic_parse_keeps_the_search_blocks() {
    let blocks = json!([{ "type": "text", "text": "searching" }, search_use(), search_result()]);
    assert_eq!(anthropic_parse(blocks.clone()).raw_content, Some(blocks));
}

// spec: protocols/anthropic/tools_tool_search_spec.rb:79 parse_completion_body with tool search > keeps no raw content for an ordinary answer
#[test]
fn anthropic_parse_keeps_no_raw_content_for_an_ordinary_answer() {
    assert_eq!(
        anthropic_parse(json!([{ "type": "text", "text": "hi" }])).raw_content,
        None
    );
}

fn anthropic_history_message() -> Message {
    let tool_use = json!({ "type": "tool_use", "id": "t1", "name": "weather_lookup", "input": { "city": "B" } });
    let web_search =
        json!({ "type": "server_tool_use", "id": "srv_2", "name": "web_search", "input": {} });
    let mut m = Message::assistant("searching");
    m.raw_content = Some(json!([
        { "type": "text", "text": "searching" }, search_use(), search_result(), web_search, tool_use
    ]));
    let call = ToolCall::new(
        "t1",
        "weather_lookup",
        spec_helpers::args(json!({ "city": "B" })),
    );
    m.tool_calls = Some([("t1".to_string(), call)].into_iter().collect());
    m
}

/// `protocol.format_message(message)` through a rendered request with `tools`, before
/// `apply_tool_search` when `apply` is false.
async fn anthropic_replayed(tools: Option<Value>) -> Value {
    // `format_message` alone: a deferred tool keeps the formatted history whole through render.
    let (_s, chat) = chat_on("anthropic", spec_helpers::MODEL).await;
    let mut chat = chat.with_tools([heavy()]);
    chat.add_message(Message::user("hi"));
    chat.add_message(anthropic_history_message());
    let mut payload = chat.render().expect("render");
    let message = payload["messages"][1].clone();
    match tools {
        None => message,
        Some(tools) => {
            payload = json!({ "tools": tools, "messages": [message] });
            tool_search::apply_anthropic(&mut payload);
            payload["messages"][0].clone()
        }
    }
}

// spec: protocols/anthropic/tools_tool_search_spec.rb:100 history replay of tool-search blocks through raw_content > replays the search pair verbatim while the request still declares deferred tools
#[tokio::test]
async fn anthropic_formats_the_search_pair_verbatim() {
    assert_eq!(
        types(&anthropic_replayed(None).await["content"]),
        [
            "text",
            "server_tool_use",
            "tool_search_tool_result",
            "server_tool_use",
            "tool_use"
        ]
    );
}

// spec: protocols/anthropic/tools_tool_search_spec.rb:110 history replay of tool-search blocks through raw_content > drops only the search pair when the request declares no search tool
#[tokio::test]
async fn anthropic_drops_only_the_search_pair_without_a_search_tool() {
    let formatted = anthropic_replayed(Some(json!([{ "name": "a" }]))).await;
    assert_eq!(
        types(&formatted["content"]),
        ["text", "server_tool_use", "tool_use"]
    );
    assert_eq!(formatted["content"][1]["name"], "web_search");
    assert_eq!(
        anthropic_history_message()
            .raw_content
            .unwrap()
            .as_array()
            .unwrap()
            .len(),
        5
    );
}

// spec: protocols/anthropic/tools_tool_search_spec.rb:118 history replay of tool-search blocks through raw_content > keeps the pair while a deferred tool or a configured search tool is declared
#[tokio::test]
async fn anthropic_keeps_the_pair_with_a_deferred_or_search_tool() {
    let regex =
        json!({ "type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex" });
    let deferred = anthropic_replayed(Some(json!([{ "name": "a", "defer_loading": true }]))).await;
    assert!(types(&deferred["content"]).contains(&"tool_search_tool_result".to_string()));
    let configured = anthropic_replayed(Some(json!([regex]))).await;
    assert!(types(&configured["content"]).contains(&"tool_search_tool_result".to_string()));
}

/// `WeatherLookupTool` (deferred) and `CurrentTimeTool` of the end-to-end examples.
struct WeatherLookupTool;

#[async_trait]
impl Tool for WeatherLookupTool {
    fn description(&self) -> String {
        "Looks up the current weather for a city.".into()
    }
    fn is_deferred(&self) -> bool {
        true
    }
    fn parameters(&self) -> Vec<Parameter> {
        vec![Parameter::new("city").description("City name")]
    }
    async fn execute(&self, a: Map<String, Value>, _: &ToolCall) -> Result<ToolResult, ToolError> {
        Ok(format!("weather in {}", a["city"].as_str().unwrap_or("")).into())
    }
}

struct CurrentTimeTool;

#[async_trait]
impl Tool for CurrentTimeTool {
    fn description(&self) -> String {
        "Returns the current time.".into()
    }
    async fn execute(&self, _: Map<String, Value>, _: &ToolCall) -> Result<ToolResult, ToolError> {
        Ok("now".into())
    }
}

async fn end_to_end(provider: &str, model: &str, search: Option<Value>) -> Vec<Value> {
    let (_s, chat) = chat_on(provider, model).await;
    let mut chat = chat.with_tool(WeatherLookupTool).with_tool(CurrentTimeTool);
    if let Some(search) = search {
        chat = chat.with_provider_tools([ProviderTool::Raw(search)]);
    }
    chat.ask_later("hi").unwrap();
    chat.render().unwrap()["tools"].as_array().cloned().unwrap()
}

fn named<'a>(tools: &'a [Value], name: &str) -> &'a Value {
    tools.iter().find(|t| t["name"] == name).expect(name)
}

// spec: protocols/anthropic/tools_tool_search_spec.rb:144 end-to-end request payload via Chat#render > sends defer_loading: true on deferred tools and appends the BM25 primitive
#[tokio::test]
async fn anthropic_end_to_end_marks_deferred_tools_and_adds_bm25() {
    let tools = end_to_end("anthropic", "claude-haiku-4-5", None).await;
    assert_eq!(
        named(&tools, "weather_lookup")["defer_loading"],
        json!(true)
    );
    assert!(named(&tools, "current_time").get("defer_loading").is_none());
    assert_eq!(
        tools
            .iter()
            .filter(|t| t["type"] == "tool_search_tool_bm25_20251119")
            .count(),
        1
    );
}

// spec: protocols/anthropic/tools_tool_search_spec.rb:158 end-to-end request payload via Chat#render > uses a configured search tool instead of adding the BM25 primitive
#[tokio::test]
async fn anthropic_end_to_end_uses_a_configured_search_tool() {
    let regex =
        json!({ "type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex" });
    let tools = end_to_end("anthropic", "claude-haiku-4-5", Some(regex)).await;
    let types: Vec<&str> = tools.iter().filter_map(|t| t["type"].as_str()).collect();
    assert_eq!(types, ["tool_search_tool_regex_20251119"]);
}

// ---- protocols/responses/tools_tool_search_spec.rb ------------------------------------------

// spec: protocols/responses/tools_tool_search_spec.rb:14 .tool_for > omits defer_loading for a bare tool
#[test]
fn responses_tool_for_omits_defer_loading_for_a_bare_tool() {
    assert!(
        responses::tool_for(tool("a", false).as_ref())
            .get("defer_loading")
            .is_none()
    );
}

// spec: protocols/responses/tools_tool_search_spec.rb:18 .tool_for > emits defer_loading: true for a deferred tool
#[test]
fn responses_tool_for_marks_a_deferred_tool() {
    assert_eq!(
        responses::tool_for(tool("a", true).as_ref())["defer_loading"],
        json!(true)
    );
}

fn responses_payload_tools(tools: &[SharedTool]) -> Vec<Value> {
    let mut payload = json!({
        "tools": tools.iter().map(|t| responses::tool_for(t.as_ref())).collect::<Vec<_>>(),
        "input": []
    });
    tool_search::apply_responses(&mut payload);
    payload["tools"].as_array().cloned().unwrap_or_default()
}

// spec: protocols/responses/tools_tool_search_spec.rb:28 .apply_tool_search > does not add the tool_search tool when nothing is deferred
#[test]
fn responses_apply_tool_search_adds_nothing_without_deferred_tools() {
    assert!(
        !responses_payload_tools(&[tool("a", false)])
            .iter()
            .any(|t| t["type"] == "tool_search")
    );
}

// spec: protocols/responses/tools_tool_search_spec.rb:32 .apply_tool_search > adds the tool_search tool once when any function is deferred
#[test]
fn responses_apply_tool_search_adds_tool_search_once() {
    let tools = responses_payload_tools(&[tool("a", false), tool("b", true)]);
    assert_eq!(tools.last(), Some(&json!({ "type": "tool_search" })));
    assert_eq!(
        tools.iter().filter(|t| t["type"] == "tool_search").count(),
        1
    );
}

fn responses_parse(output: Value) -> Message {
    let body = json!({ "output": output, "model": "gpt-5.4", "status": "completed", "usage": {} });
    responses::parse_completion_body(
        rust_llm::Provider::OpenAI,
        &body,
        rust_llm::message::RawResponse::default(),
    )
    .unwrap()
}

// spec: protocols/responses/tools_tool_search_spec.rb:49 Responses::Chat tool-search parsing > keeps the raw output on the parsed Message for replay
#[test]
fn responses_parse_keeps_the_search_output() {
    let output = json!([
        { "type": "tool_search_call", "id": "ts_1" },
        { "type": "tool_search_output", "tools": [{ "name": "weather_lookup" }, { "name": "stock_price" }] },
        { "type": "function_call", "call_id": "c1", "name": "weather_lookup", "arguments": "{}" }
    ]);
    assert_eq!(responses_parse(output.clone()).raw_content, Some(output));
}

fn namespaced_call() -> Value {
    json!({ "type": "function_call", "call_id": "c1", "name": "weather_lookup", "arguments": "{}",
            "namespace": "weather_lookup" })
}

/// `format_assistant_items(message)`, through a rendered request.
async fn responses_assistant_items(message: Message, apply_tools: Option<Value>) -> Value {
    // `format_assistant_items` alone: a deferred tool keeps the formatted items whole.
    let (_s, chat) = chat_on("openai", "gpt-5.4").await;
    let mut chat = chat.with_tools([heavy()]);
    chat.add_message(message);
    let payload = chat.render().expect("render");
    let items = payload["input"].clone();
    match apply_tools {
        None => items,
        Some(tools) => {
            let mut payload = json!({ "tools": tools, "input": items });
            tool_search::apply_responses(&mut payload);
            payload["input"].clone()
        }
    }
}

// spec: protocols/responses/tools_tool_search_spec.rb:68 a function call the model reached through tool search > keeps the raw output, so the namespace the API assigned is replayed with the call
#[tokio::test]
async fn responses_namespaced_call_replays_raw() {
    let message = responses_parse(json!([namespaced_call()]));
    assert_eq!(message.raw_content, Some(json!([namespaced_call()])));
    let call = message.tool_calls.as_ref().unwrap().get("c1").unwrap();
    assert!(
        serde_json::to_value(call)
            .unwrap()
            .get("namespace")
            .is_none()
    );
    assert_eq!(
        responses_assistant_items(message, None).await,
        json!([namespaced_call()])
    );
}

// spec: protocols/responses/tools_tool_search_spec.rb:76 a function call the model reached through tool search > leaves an ordinary function call to the plain replay
#[tokio::test]
async fn responses_ordinary_call_replays_plain() {
    let mut call = namespaced_call();
    call.as_object_mut().unwrap().remove("namespace");
    let message = responses_parse(json!([call]));
    assert_eq!(message.raw_content, None);
    let items = responses_assistant_items(message, None).await;
    assert!(items[0].get("namespace").is_none());
}

fn responses_history_message() -> Message {
    let mut m = Message::new(Role::Assistant, None);
    m.raw_content = Some(json!([
        { "type": "tool_search_call", "id": "ts_1" },
        { "type": "tool_search_output", "tools": [{ "name": "weather_lookup" }] },
        namespaced_call()
    ]));
    let call = ToolCall::new("c1", "weather_lookup", Map::new());
    m.tool_calls = Some([("c1".to_string(), call)].into_iter().collect());
    m
}

// spec: protocols/responses/tools_tool_search_spec.rb:99 history replay of tool-search items through raw_content > replays the items verbatim while the request still declares deferred tools
#[tokio::test]
async fn responses_formats_the_search_items_verbatim() {
    assert_eq!(
        types(&responses_assistant_items(responses_history_message(), None).await),
        ["tool_search_call", "tool_search_output", "function_call"]
    );
}

// spec: protocols/responses/tools_tool_search_spec.rb:109 history replay of tool-search items through raw_content > omits the search items when the request declares no search tool, keeping the namespaced call
#[tokio::test]
async fn responses_omits_the_search_items_without_a_search_tool() {
    let replayed = responses_assistant_items(
        responses_history_message(),
        Some(json!([{ "type": "function", "name": "a" }])),
    )
    .await;
    assert_eq!(replayed, json!([namespaced_call()]));
}

// spec: protocols/responses/tools_tool_search_spec.rb:113 history replay of tool-search items through raw_content > keeps the search items while a deferred tool or the tool_search tool is declared
#[tokio::test]
async fn responses_keeps_the_search_items_with_a_deferred_or_search_tool() {
    let deferred = responses_assistant_items(
        responses_history_message(),
        Some(json!([{ "type": "function", "name": "a", "defer_loading": true }])),
    )
    .await;
    assert_eq!(deferred.as_array().unwrap().len(), 3);
    let search = responses_assistant_items(
        responses_history_message(),
        Some(json!([{ "type": "tool_search" }])),
    )
    .await;
    assert_eq!(search.as_array().unwrap().len(), 3);
}

// spec: protocols/responses/tools_tool_search_spec.rb:137 end-to-end request payload via Chat#render > sends defer_loading on deferred tools and appends the tool_search tool
#[tokio::test]
async fn responses_end_to_end_marks_deferred_tools_and_adds_tool_search() {
    let tools = end_to_end("openai", "gpt-5.4", None).await;
    assert_eq!(
        named(&tools, "weather_lookup")["defer_loading"],
        json!(true)
    );
    assert!(named(&tools, "current_time").get("defer_loading").is_none());
    assert_eq!(
        tools.iter().filter(|t| t["type"] == "tool_search").count(),
        1
    );
}

// spec: protocols/responses/tools_tool_search_spec.rb:150 end-to-end request payload via Chat#render > does not add a second tool_search when one is configured
#[tokio::test]
async fn responses_end_to_end_keeps_a_configured_tool_search() {
    let tools = end_to_end("openai", "gpt-5.4", Some(json!({ "type": "tool_search" }))).await;
    assert_eq!(
        tools.iter().filter(|t| t["type"] == "tool_search").count(),
        1
    );
}
