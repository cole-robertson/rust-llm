//! RubyLLM 2.1's MCP additions, mirroring `spec/ruby_llm/mcp_spec.rb`, `mcp/client_spec.rb`,
//! `mcp/result_spec.rb`, and `mcp/input_request_spec.rb`: tool results with `_meta` and UIs,
//! refreshing tools the server changed, sessions of restarted servers, declared input requests,
//! listening for changes, extensions, MCP Apps, tasks, and log messages. Server-backed examples
//! run RubyLLM's own spec server (`tests/fixtures/mcp/server.rb`, copied verbatim).

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use rust_llm::mcp::{
    Change, Client, Extension, InputKind, InputRequest, LEGACY_VERSIONS, LogLevel, Mcp, McpBuilder,
    McpResult, McpTool, OnNotification, Stdio, Task, TaskStatus, Transport,
};
use rust_llm::{Error, Message, ToolResult};
use serde_json::{Map, Value, json};
use tokio::sync::mpsc;

fn server_path() -> String {
    format!(
        "{}/tests/fixtures/mcp/server.rb",
        env!("CARGO_MANIFEST_DIR")
    )
}

/// `Class.new(RubyLLM::MCP) { command(RbConfig.ruby, server) }`.
fn files() -> McpBuilder {
    Mcp::command(["ruby".to_string(), server_path()]).name("files")
}

fn legacy() -> McpBuilder {
    files().env("MCP_ERA", "legacy")
}

fn map(value: Value) -> Map<String, Value> {
    value.as_object().cloned().unwrap_or_default()
}

async fn ask(mcp: &Mcp, method: &str, params: Value) -> rust_llm::Result<Value> {
    mcp.client().request(method, params, &[], &mut |_| {}).await
}

async fn tool(mcp: &Mcp, name: &str) -> Arc<McpTool> {
    mcp.mcp_tools()
        .await
        .unwrap()
        .into_iter()
        .find(|t| t.name == name)
        .unwrap()
}

async fn names(mcp: &Mcp) -> Vec<String> {
    mcp.tools()
        .await
        .unwrap()
        .iter()
        .map(|t| t.name())
        .collect()
}

fn declared_capabilities(echoed: &Value) -> Value {
    echoed["meta"]["io.modelcontextprotocol/clientCapabilities"].clone()
}

/// A transport that answers each method with a canned reply, recording what was sent.
#[derive(Default)]
struct Scripted {
    sent: Mutex<Vec<(String, Value)>>,
    closed: AtomicBool,
    #[allow(clippy::type_complexity)]
    answer: Option<Box<dyn Fn(&str, usize, &mut OnNotification<'_>) -> Value + Send + Sync>>,
}

impl Scripted {
    fn new(
        answer: impl Fn(&str, usize, &mut OnNotification<'_>) -> Value + Send + Sync + 'static,
    ) -> Arc<Scripted> {
        Arc::new(Scripted {
            answer: Some(Box::new(answer)),
            ..Default::default()
        })
    }

    fn sent(&self) -> Vec<String> {
        self.sent
            .lock()
            .unwrap()
            .iter()
            .map(|(m, _)| m.clone())
            .collect()
    }

    fn params(&self, method: &str) -> Value {
        self.sent
            .lock()
            .unwrap()
            .iter()
            .find(|(m, _)| m == method)
            .map(|(_, p)| p.clone())
            .unwrap_or(Value::Null)
    }
}

#[async_trait]
impl Transport for Scripted {
    async fn request(
        &self,
        message: &Value,
        _: Option<&str>,
        _: Option<Duration>,
        _: &[(String, String)],
        on_notification: &mut OnNotification<'_>,
    ) -> rust_llm::Result<Value> {
        let method = message["method"].as_str().unwrap_or("").to_string();
        let attempt = {
            let mut sent = self.sent.lock().unwrap();
            sent.push((method.clone(), message["params"].clone()));
            sent.iter().filter(|(m, _)| *m == method).count()
        };
        let reply = (self.answer.as_ref().unwrap())(&method, attempt, on_notification);
        let mut response = json!({ "jsonrpc": "2.0", "id": message["id"] });
        for (k, v) in reply.as_object().cloned().unwrap_or_default() {
            response[k] = v;
        }
        Ok(response)
    }
    async fn notify(&self, message: &Value, _: Option<&str>) -> rust_llm::Result<()> {
        let method = message["method"].as_str().unwrap_or("").to_string();
        self.sent
            .lock()
            .unwrap()
            .push((method, message["params"].clone()));
        Ok(())
    }
    async fn cancel(&self, _: &Value, _: Option<&str>) -> rust_llm::Result<()> {
        Ok(())
    }
    async fn close(&self) {
        self.closed.store(true, Ordering::SeqCst);
    }
}

// ---- tools -------------------------------------------------------------------------------------

// spec: mcp_spec.rb:43 tools > calls a tool the way a chat does and returns the server result
#[tokio::test]
async fn a_tool_call_returns_the_server_result() {
    let mcp = files().build().unwrap();
    let result = tool(&mcp, "echo")
        .await
        .call(json!({ "text": "hello" }))
        .await
        .unwrap();
    // `RubyLLM::Tool.split_result(result)`.
    assert_eq!(
        (result.content.as_str(), result.attachments.len()),
        ("hello", 0)
    );
    let server_result = result.mcp_result.expect("the MCP::Result itself");
    assert_eq!(server_result.text, "hello");
    mcp.close().await;
}

// spec: mcp_spec.rb:50 tools > reports a failed tool as an error for the model, as 2.0 did
#[tokio::test]
async fn a_failed_tool_is_an_error_for_the_model() {
    let mcp = files().build().unwrap();
    let result = tool(&mcp, "fail").await.call(json!({})).await.unwrap();
    assert_eq!(result, ToolResult::error("Something broke"));
    assert_eq!(result.content, r#"{"error":"Something broke"}"#);
    assert!(result.attachments.is_empty());
    mcp.close().await;
}

// spec: mcp_spec.rb:57 tools > sends the model the attachments of a result
#[tokio::test]
async fn a_tool_sends_the_model_the_attachments_of_a_result() {
    let mcp = files().build().unwrap();
    let result = tool(&mcp, "picture").await.call(json!({})).await.unwrap();
    assert_eq!(result.content, "Here it is\n\npixel.png: file:///pixel.png");
    assert_eq!(result.attachments[0].mime_type, "image/png");
    mcp.close().await;
}

// spec: mcp_spec.rb:65 tools > lists them again after the server answers that a tool it called does not exist
#[tokio::test]
async fn lists_tools_again_after_the_server_says_a_called_tool_does_not_exist() {
    let mcp = files().build().unwrap();
    assert!(names(&mcp).await.contains(&"delete_everything".to_string()));
    ask(
        &mcp,
        "spec/remove_tool",
        json!({ "name": "delete_everything" }),
    )
    .await
    .unwrap();
    let Err(Error::Mcp(e)) = mcp.call("delete_everything", json!({})).await else {
        panic!("expected an MCP error");
    };
    assert_eq!(e.message, "Unknown tool: delete_everything");
    assert!(!names(&mcp).await.contains(&"delete_everything".to_string()));
    mcp.close().await;
}

// spec: mcp_spec.rb:73 tools > lists them again after the server answers a tool call with method not found
#[tokio::test]
async fn lists_tools_again_after_method_not_found() {
    let transport = Scripted::new(|method, _, _| match method {
        "server/discover" => json!({ "result": { "supportedVersions": ["2026-07-28"] } }),
        "tools/list" => json!({ "result": { "tools": [{ "name": "gone", "inputSchema": {} }] } }),
        _ => json!({ "error": { "code": -32_601, "message": "Method not found" } }),
    });
    let flaky = Mcp::transport("flaky", transport.clone()).build().unwrap();
    flaky.tools().await.unwrap();
    let Err(Error::Mcp(e)) = flaky.call("gone", json!({})).await else {
        panic!("expected an MCP error");
    };
    assert_eq!(e.message, "Method not found");
    flaky.tools().await.unwrap();
    assert_eq!(
        transport
            .sent()
            .iter()
            .filter(|m| *m == "tools/list")
            .count(),
        2
    );
}

// spec: mcp_spec.rb:109 tools > with a server that predates 2026-07-28 > lists them again after the server says they changed during a request
#[tokio::test]
async fn lists_tools_again_after_an_older_server_says_they_changed_during_a_request() {
    let mcp = legacy().build().unwrap();
    assert!(!names(&mcp).await.contains(&"extra_9".to_string()));
    ask(&mcp, "spec/change_tools", json!({})).await.unwrap();
    assert!(names(&mcp).await.contains(&"extra_9".to_string()));
    mcp.close().await;
}

// spec: mcp_spec.rb:117 tools > with a server that predates 2026-07-28 > starts a new session after the server process restarts
#[tokio::test]
async fn starts_a_new_session_after_an_older_server_restarts() {
    let mcp = legacy().build().unwrap();
    mcp.tools().await.unwrap();
    let Err(Error::Mcp(e)) = ask(&mcp, "spec/exit", json!({})).await else {
        panic!("expected the server to exit");
    };
    assert!(e.message.contains("exited"), "{}", e.message);
    let listed = ask(&mcp, "tools/list", json!({})).await.unwrap();
    assert!(!listed["tools"].as_array().unwrap().is_empty());
    mcp.close().await;
}

// ---- #call -------------------------------------------------------------------------------------

// spec: mcp_spec.rb:261 #call > returns a failed result whole, with its text as content
#[tokio::test]
async fn call_returns_a_failed_result_whole() {
    let mcp = files().build().unwrap();
    let result = mcp.call("fail", json!({})).await.unwrap();
    assert!(result.is_error());
    assert_eq!(result.text, "Something broke");
    assert_eq!(result.content().content, "Something broke");
    mcp.close().await;
}

// spec: mcp_spec.rb:268 #call > answers what the server asks mid-call, pings with a result and everything else with method not found
#[tokio::test]
async fn call_answers_what_the_server_asks_mid_call() {
    let mcp = files().build().unwrap();
    let answers: Value =
        serde_json::from_str(&mcp.call("ask_client", json!({})).await.unwrap().text).unwrap();
    let unsupported = json!({ "code": -32_601, "message": "Method not found" });
    assert_eq!(
        answers,
        json!({ "ping-1": {}, "roots-1": unsupported, "sample-1": unsupported, "elicit-1": unsupported })
    );
    mcp.close().await;
}

// ---- resources ---------------------------------------------------------------------------------

// spec: mcp_spec.rb:298 resources > reads the _meta of resources and of their contents
#[tokio::test]
async fn reads_the_meta_of_resources_and_of_their_contents() {
    let mcp = files().build().unwrap();
    let readme = mcp.resource("file:///project/README.md").await.unwrap();
    assert_eq!(readme.mime_type.as_deref(), Some("text/markdown"));
    assert_eq!(
        Value::Object(readme.meta),
        json!({ "com.example/etag": "v2" })
    );
    assert_eq!(
        Value::Object(mcp.resources().await.unwrap()[0].meta.clone()),
        json!({ "com.example/listed": true })
    );
    assert!(
        mcp.resource("file:///project/notes.txt")
            .await
            .unwrap()
            .meta
            .is_empty()
    );
    mcp.close().await;
}

// ---- input requests ----------------------------------------------------------------------------

// spec: mcp_spec.rb:449 input requests > declares only the input it accepts
#[tokio::test]
async fn declares_only_the_input_it_accepts() {
    let mcp = files().input_requests(&[InputKind::Url]).build().unwrap();
    let echoed = ask(&mcp, "meta/echo", json!({})).await.unwrap();
    assert_eq!(
        declared_capabilities(&echoed),
        json!({ "elicitation": { "url": {} } })
    );
    mcp.close().await;
}

// spec: mcp_spec.rb:458 input requests > declares no input when it accepts none
#[tokio::test]
async fn declares_no_input_when_it_accepts_none() {
    let builder = files().input_requests(&[]);
    assert!(builder.accepted_input_requests().is_empty());
    let mcp = builder.build().unwrap();
    let echoed = ask(&mcp, "meta/echo", json!({})).await.unwrap();
    assert_eq!(declared_capabilities(&echoed), json!({}));
    mcp.close().await;
}

// spec: mcp_spec.rb:468 input requests > declines requests it does not accept without asking its callbacks
#[tokio::test]
async fn declines_requests_it_does_not_accept_without_asking() {
    let asked = Arc::new(AtomicUsize::new(0));
    let counter = asked.clone();
    let mcp = files()
        .input_requests(&[InputKind::Url])
        .before_input_request(move |_| {
            counter.fetch_add(1, Ordering::SeqCst);
        })
        .build()
        .unwrap();
    assert_eq!(
        mcp.call("deploy", json!({})).await.unwrap().text,
        "Deploy cancelled"
    );
    assert_eq!(asked.load(Ordering::SeqCst), 0);
    mcp.close().await;
}

// spec: mcp_spec.rb:478 input requests > refuses kinds of input it does not know
#[test]
fn refuses_kinds_of_input_it_does_not_know() {
    let Err(Error::Argument(message)) = InputKind::parse("email") else {
        panic!("expected an argument error");
    };
    assert_eq!(message, "Unknown input requests: email");
}

// spec: mcp_spec.rb:1218 .mcp > accepts the input requests it takes
#[test]
fn accepts_the_input_requests_it_takes() {
    let none = Mcp::url("https://mcp.linear.app/mcp").input_requests(&[]);
    assert!(none.accepted_input_requests().is_empty());
    let form = Mcp::url("https://mcp.linear.app/mcp").input_requests(&[InputKind::Form]);
    assert_eq!(form.accepted_input_requests(), [InputKind::Form]);
}

// ---- listening ---------------------------------------------------------------------------------

/// What `after_change` saw: a list name, a resource's content, or a task.
#[derive(Debug, Clone, PartialEq)]
enum Seen {
    Tools,
    Prompts,
    Resources,
    Content(String),
    Task(String, TaskStatus, Option<String>, Option<String>),
}

/// `listening_to(server, **env)`: an MCP whose `after_change` records what changed, reading a
/// resource's content and a completed task's result text.
fn listening(builder: McpBuilder) -> (Mcp, mpsc::UnboundedReceiver<Seen>) {
    let (sender, receiver) = mpsc::unbounded_channel();
    let mcp = builder
        .after_change(move |change| {
            let sender = sender.clone();
            async move {
                let seen = match change {
                    Change::Tools => Seen::Tools,
                    Change::Prompts => Seen::Prompts,
                    Change::Resources => Seen::Resources,
                    Change::Resource(resource) => Seen::Content(
                        resource
                            .content()
                            .await
                            .ok()
                            .and_then(|c| c.as_text().map(str::to_string))
                            .unwrap_or_default(),
                    ),
                    Change::Task(task) => {
                        let text = task.result().ok().flatten().map(|r| r.text);
                        Seen::Task(
                            task.id.clone(),
                            task.status(),
                            task.status_message().map(str::to_string),
                            text,
                        )
                    }
                };
                let _ = sender.send(seen);
            }
        })
        .build()
        .unwrap();
    (mcp, receiver)
}

async fn next_change(changes: &mut mpsc::UnboundedReceiver<Seen>) -> Seen {
    tokio::time::timeout(Duration::from_secs(5), changes.recv())
        .await
        .expect("no change within 5 seconds")
        .expect("the change channel closed")
}

async fn subscriptions(mcp: &Mcp) -> Map<String, Value> {
    ask(mcp, "spec/subscriptions", json!({})).await.unwrap()["subscriptions"]
        .as_object()
        .cloned()
        .unwrap_or_default()
}

async fn eventually<T, F: Future<Output = Option<T>>>(mut check: impl FnMut() -> F) -> T {
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    loop {
        if let Some(value) = check().await {
            return value;
        }
        assert!(std::time::Instant::now() < deadline, "timed out");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

fn lists() -> Value {
    json!({ "toolsListChanged": true, "promptsListChanged": true, "resourcesListChanged": true })
}

// spec: mcp_spec.rb:530 listening > subscribes to the lists the server announces changes to
#[tokio::test(flavor = "multi_thread")]
async fn subscribes_to_the_lists_the_server_announces_changes_to() {
    let (mcp, _changes) = listening(files());
    mcp.listen(&[], &[]).await.unwrap();
    assert_eq!(
        subscriptions(&mcp)
            .await
            .values()
            .cloned()
            .collect::<Vec<_>>(),
        [lists()]
    );
    mcp.close().await;
}

// spec: mcp_spec.rb:537 listening > lists tools again and runs after_change when the server changes them
#[tokio::test(flavor = "multi_thread")]
async fn lists_tools_again_and_runs_after_change_when_they_change() {
    let (mcp, mut changes) = listening(files());
    mcp.listen(&[], &[]).await.unwrap().tools().await.unwrap();
    ask(&mcp, "spec/change_tools", json!({})).await.unwrap();
    assert_eq!(next_change(&mut changes).await, Seen::Tools);
    assert!(names(&mcp).await.contains(&"extra_9".to_string()));
    mcp.close().await;
}

// spec: mcp_spec.rb:546 listening > hears changes that arrive while no request is reading
#[tokio::test(flavor = "multi_thread")]
async fn hears_changes_that_arrive_while_no_request_is_reading() {
    let (mcp, mut changes) = listening(files());
    mcp.listen(&[], &[]).await.unwrap();
    ask(
        &mcp,
        "spec/announce_later",
        json!({ "method": "notifications/prompts/list_changed" }),
    )
    .await
    .unwrap();
    assert_eq!(next_change(&mut changes).await, Seen::Prompts);
    mcp.close().await;
}

// spec: mcp_spec.rb:554 listening > hears when a resource it listens to changes, and lets the callback read it
#[tokio::test(flavor = "multi_thread")]
async fn hears_when_a_resource_changes_and_lets_the_callback_read_it() {
    let (mcp, mut changes) = listening(files());
    let readme = mcp.resources().await.unwrap().remove(0);
    mcp.listen(&[readme.uri.as_str()], &[]).await.unwrap();
    ask(
        &mcp,
        "spec/announce",
        json!({ "method": "notifications/resources/updated", "params": { "uri": "file:///project/README.md" } }),
    )
    .await
    .unwrap();
    assert_eq!(
        next_change(&mut changes).await,
        Seen::Content("# Spec Project\n".into())
    );
    mcp.close().await;
}

// spec: mcp_spec.rb:563 listening > replaces the resources it listens to, cancelling the old subscription
#[tokio::test(flavor = "multi_thread")]
async fn replaces_the_resources_it_listens_to() {
    let (mcp, _changes) = listening(files());
    mcp.listen(&["file:///a"], &[]).await.unwrap();
    let first: Vec<String> = subscriptions(&mcp).await.keys().cloned().collect();
    mcp.listen(&["file:///b"], &[]).await.unwrap();
    let filters: Vec<Value> = subscriptions(&mcp)
        .await
        .values()
        .map(|f| f["resourceSubscriptions"].clone())
        .collect();
    assert_eq!(filters, [json!(["file:///b"])]);
    let cancelled = ask(&mcp, "spec/cancelled", json!({})).await.unwrap()["cancelled"].clone();
    assert_eq!(cancelled, json!(first));
    mcp.close().await;
}

// spec: mcp_spec.rb:573 listening > raises and stops listening when the server does not watch a resource
#[tokio::test(flavor = "multi_thread")]
async fn raises_and_stops_listening_when_the_server_does_not_watch_a_resource() {
    let (mcp, _changes) = listening(files());
    let Err(Error::Mcp(e)) = mcp.listen(&["unwatched:notes"], &[]).await else {
        panic!("expected an MCP error");
    };
    assert!(
        e.message
            .contains("does not send updates for unwatched:notes"),
        "{}",
        e.message
    );
    assert!(
        eventually(|| async { Some(subscriptions(&mcp).await) })
            .await
            .is_empty()
    );
    mcp.close().await;
}

// spec: mcp_spec.rb:579 listening > subscribes again when the server ends the subscription
#[tokio::test(flavor = "multi_thread")]
async fn subscribes_again_when_the_server_ends_the_subscription() {
    let (mcp, _changes) = listening(files());
    mcp.listen(&[], &[]).await.unwrap();
    let first = subscriptions(&mcp).await;
    ask(&mcp, "spec/end_subscriptions", json!({}))
        .await
        .unwrap();
    let renewed = eventually(|| async {
        let renewed: Vec<Value> = subscriptions(&mcp)
            .await
            .into_iter()
            .filter(|(id, _)| !first.contains_key(id))
            .map(|(_, f)| f)
            .collect();
        (!renewed.is_empty()).then_some(renewed)
    })
    .await;
    assert_eq!(renewed, first.values().cloned().collect::<Vec<_>>());
    mcp.close().await;
}

// spec: mcp_spec.rb:589 listening > runs after_change for everything it listens to once it subscribes again, since changes in between are lost
#[tokio::test(flavor = "multi_thread")]
async fn runs_after_change_for_everything_it_listens_to_once_it_subscribes_again() {
    let (mcp, mut changes) = listening(files());
    mcp.listen(&["file:///project/README.md"], &[])
        .await
        .unwrap();
    ask(&mcp, "spec/end_subscriptions", json!({}))
        .await
        .unwrap();
    let mut seen = Vec::new();
    for _ in 0..4 {
        seen.push(next_change(&mut changes).await);
    }
    assert_eq!(
        seen,
        [
            Seen::Tools,
            Seen::Prompts,
            Seen::Resources,
            Seen::Content("# Spec Project\n".into())
        ]
    );
    mcp.close().await;
}

// spec: mcp_spec.rb:597 listening > subscribes again when the server cancels the subscription
#[tokio::test(flavor = "multi_thread")]
async fn subscribes_again_when_the_server_cancels_the_subscription() {
    let (mcp, _changes) = listening(files());
    mcp.listen(&[], &[]).await.unwrap();
    let first = subscriptions(&mcp).await;
    ask(&mcp, "spec/cancel_subscriptions", json!({}))
        .await
        .unwrap();
    let renewed = eventually(|| async {
        let renewed: Vec<Value> = subscriptions(&mcp)
            .await
            .into_iter()
            .filter(|(id, _)| !first.contains_key(id))
            .map(|(_, f)| f)
            .collect();
        (!renewed.is_empty()).then_some(renewed)
    })
    .await;
    assert_eq!(renewed, first.values().cloned().collect::<Vec<_>>());
    mcp.close().await;
}

// spec: mcp_spec.rb:607 listening > subscribes again after the server exits
#[tokio::test(flavor = "multi_thread")]
async fn subscribes_again_after_the_server_exits() {
    let (mcp, _changes) = listening(files());
    mcp.listen(&[], &[]).await.unwrap();
    let Err(Error::Mcp(e)) = ask(&mcp, "spec/exit", json!({})).await else {
        panic!("expected the server to exit");
    };
    assert!(e.message.contains("exited"));
    let renewed = eventually(|| async {
        let current = subscriptions(&mcp).await;
        (!current.is_empty()).then_some(current)
    })
    .await;
    assert_eq!(renewed.len(), 1);
    mcp.close().await;
}

/// Collects `tracing` ERROR (and, for log messages, every) event, standing in for
/// `RubyLLM.logger`.
type Events = Arc<Mutex<Vec<(tracing::Level, String)>>>;

struct Collector(Events);

impl tracing::Subscriber for Collector {
    fn register_callsite(
        &self,
        _: &'static tracing::Metadata<'static>,
    ) -> tracing::subscriber::Interest {
        tracing::subscriber::Interest::sometimes()
    }
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
        let mut text = String::new();
        event.record(&mut Text(&mut text));
        self.0
            .lock()
            .unwrap()
            .push((*event.metadata().level(), text));
    }
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}

fn collector() -> (tracing::Dispatch, Events) {
    let events = Arc::new(Mutex::new(Vec::new()));
    (tracing::Dispatch::new(Collector(events.clone())), events)
}

// spec: mcp_spec.rb:615 listening > logs a callback that raises and keeps listening
#[test]
fn logs_a_callback_that_raises_and_keeps_listening() {
    let (dispatch, events) = collector();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let _guard = tracing::dispatcher::set_default(&dispatch);
    runtime.block_on(async {
        let calls = Arc::new(AtomicUsize::new(0));
        let counter = calls.clone();
        let (sender, mut changes) = mpsc::unbounded_channel();
        let mcp = files()
            .after_change(move |change| {
                let (sender, counter) = (sender.clone(), counter.clone());
                async move {
                    if matches!(change, Change::Prompts) {
                        let _ = sender.send(Seen::Prompts);
                        if counter.fetch_add(1, Ordering::SeqCst) == 0 {
                            panic!("Callback failed");
                        }
                    }
                }
            })
            .build()
            .unwrap();
        mcp.listen(&[], &[]).await.unwrap();
        for _ in 0..2 {
            ask(
                &mcp,
                "spec/announce",
                json!({ "method": "notifications/prompts/list_changed" }),
            )
            .await
            .unwrap();
        }
        assert_eq!(
            [
                next_change(&mut changes).await,
                next_change(&mut changes).await
            ],
            [Seen::Prompts, Seen::Prompts]
        );
        eventually(|| {
            let done = calls.load(Ordering::SeqCst) == 2;
            async move { done.then_some(()) }
        })
        .await;
        mcp.close().await;
    });
    let errors: Vec<String> = events
        .lock()
        .unwrap()
        .iter()
        .filter(|(level, text)| {
            *level == tracing::Level::ERROR && text.contains("after_change callback failed")
        })
        .map(|(_, text)| text.clone())
        .collect();
    assert_eq!(errors.len(), 1, "{errors:?}");
}

// spec: mcp_spec.rb:628 listening > stops its thread when the MCP closes
#[tokio::test(flavor = "multi_thread")]
async fn stops_listening_when_the_mcp_closes() {
    let (mcp, _changes) = listening(files());
    mcp.listen(&[], &[]).await.unwrap();
    assert_eq!(subscriptions(&mcp).await.len(), 1);
    mcp.close().await;
    // The listener task is gone: a fresh connection sees no subscription and no one resubscribes.
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(subscriptions(&mcp).await.is_empty());
    mcp.close().await;
}

// spec: mcp_spec.rb:638 listening > does nothing for a server that announces no changes
#[tokio::test(flavor = "multi_thread")]
async fn does_nothing_for_a_server_that_announces_no_changes() {
    let (quiet, _changes) = listening(files().env("MCP_CHANGES", "none"));
    assert!(quiet.listen(&[], &[]).await.is_ok());
    assert!(subscriptions(&quiet).await.is_empty());
    quiet.close().await;
}

async fn report(mcp: &Mcp) -> Task {
    let Err(Error::McpTask(task)) = tool(mcp, "report").await.call(json!({})).await else {
        panic!("expected a task");
    };
    *task
}

// spec: mcp_spec.rb:669 listening > with tasks > hears when the status of a task it listens to changes, with the task as it stands
#[tokio::test(flavor = "multi_thread")]
async fn hears_when_the_status_of_a_task_changes() {
    let (mcp, mut changes) = listening(files().with_extension(Extension::Tasks, json!({})));
    let task = report(&mcp).await;
    mcp.listen(&[], &[task.id.as_str()]).await.unwrap();
    ask(
        &mcp,
        "spec/announce",
        json!({ "method": "notifications/tasks", "params": { "taskId": task.id, "status": "completed",
                "result": { "content": [{ "type": "text", "text": "Report ready" }] } } }),
    )
    .await
    .unwrap();
    assert_eq!(
        next_change(&mut changes).await,
        Seen::Task(
            task.id.clone(),
            TaskStatus::Completed,
            None,
            Some("Report ready".into())
        )
    );
    let filters: Vec<Value> = subscriptions(&mcp)
        .await
        .values()
        .map(|f| f["taskIds"].clone())
        .collect();
    assert_eq!(filters, [json!([task.id])]);
    mcp.close().await;
}

// spec: mcp_spec.rb:683 listening > with tasks > checks on the tasks it listens to once it subscribes again
#[tokio::test(flavor = "multi_thread")]
async fn checks_on_the_tasks_it_listens_to_once_it_subscribes_again() {
    let (mcp, mut changes) = listening(files().with_extension(Extension::Tasks, json!({})));
    let task = report(&mcp).await;
    mcp.listen(&[], &[task.id.as_str()]).await.unwrap();
    ask(&mcp, "spec/end_subscriptions", json!({}))
        .await
        .unwrap();
    let mut seen = Vec::new();
    for _ in 0..4 {
        seen.push(next_change(&mut changes).await);
    }
    assert_eq!(&seen[..3], [Seen::Tools, Seen::Prompts, Seen::Resources]);
    assert_eq!(
        seen[3],
        Seen::Task(
            task.id.clone(),
            TaskStatus::Working,
            Some("Rendering".into()),
            None
        )
    );
    mcp.close().await;
}

// spec: mcp_spec.rb:694 listening > with tasks > raises and stops listening when the server does not send the status of a task
#[tokio::test(flavor = "multi_thread")]
async fn raises_when_the_server_does_not_send_the_status_of_a_task() {
    let (mcp, _changes) = listening(files().with_extension(Extension::Tasks, json!({})));
    let Err(Error::Mcp(e)) = mcp.listen(&[], &["task-404"]).await else {
        panic!("expected an MCP error");
    };
    assert!(e.message.contains("does not send updates for task-404"));
    assert!(
        eventually(|| async { Some(subscriptions(&mcp).await) })
            .await
            .is_empty()
    );
    mcp.close().await;
}

// spec: mcp_spec.rb:704 listening > with a server that predates subscriptions > subscribes to resources and hears the changes the server announces
#[tokio::test(flavor = "multi_thread")]
async fn an_older_server_subscribes_to_resources_and_announces_changes() {
    let (mcp, mut changes) = listening(legacy());
    mcp.listen(&["file:///project/README.md"], &[])
        .await
        .unwrap();
    ask(
        &mcp,
        "spec/announce_later",
        json!({ "method": "notifications/tools/list_changed" }),
    )
    .await
    .unwrap();
    assert_eq!(next_change(&mut changes).await, Seen::Tools);
    assert_eq!(
        ask(&mcp, "spec/subscriptions", json!({})).await.unwrap()["watched"],
        json!(["file:///project/README.md"])
    );
    mcp.close().await;
}

// spec: mcp_spec.rb:713 listening > with a server that predates subscriptions > starts a new session once the server process restarts, and catches up once
#[tokio::test(flavor = "multi_thread")]
async fn an_older_server_restart_starts_a_new_session_and_catches_up_once() {
    let (mcp, mut changes) = listening(legacy());
    mcp.listen(&[], &[]).await.unwrap();
    let Err(Error::Mcp(e)) = ask(&mcp, "spec/exit", json!({})).await else {
        panic!("expected the server to exit");
    };
    assert!(e.message.contains("exited"));
    let mut seen = Vec::new();
    for _ in 0..3 {
        seen.push(next_change(&mut changes).await);
    }
    assert_eq!(seen, [Seen::Tools, Seen::Prompts, Seen::Resources]);
    assert!(
        tokio::time::timeout(Duration::from_millis(500), changes.recv())
            .await
            .is_err()
    );
    ask(
        &mcp,
        "spec/announce_later",
        json!({ "method": "notifications/tools/list_changed" }),
    )
    .await
    .unwrap();
    assert_eq!(next_change(&mut changes).await, Seen::Tools);
    mcp.close().await;
}

// spec: mcp_spec.rb:724 listening > with a server that predates subscriptions > raises for tasks, whose status such a server never announces
#[tokio::test(flavor = "multi_thread")]
async fn an_older_server_raises_for_tasks() {
    let (mcp, _changes) = listening(legacy());
    let Err(Error::Mcp(e)) = mcp.listen(&[], &["task-1"]).await else {
        panic!("expected an MCP error");
    };
    assert!(e.message.contains("does not send updates for task-1"));
    mcp.close().await;
}

// spec: mcp_spec.rb:729 listening > with a server that predates subscriptions > unsubscribes from resources it no longer listens to
#[tokio::test(flavor = "multi_thread")]
async fn an_older_server_unsubscribes_from_resources_it_no_longer_listens_to() {
    let (mcp, _changes) = listening(legacy());
    mcp.listen(&["file:///a", "file:///b"], &[]).await.unwrap();
    mcp.listen(&["file:///b"], &[]).await.unwrap();
    assert_eq!(
        ask(&mcp, "spec/subscriptions", json!({})).await.unwrap()["watched"],
        json!(["file:///b"])
    );
    mcp.close().await;
}

// spec: mcp_spec.rb:737 listening > with a server that predates subscriptions > runs after_change for changes announced while it answers a request
#[tokio::test(flavor = "multi_thread")]
async fn an_older_server_runs_after_change_for_changes_announced_during_a_request() {
    let (mcp, mut changes) = listening(legacy());
    ask(&mcp, "spec/change_tools", json!({})).await.unwrap();
    assert_eq!(next_change(&mut changes).await, Seen::Tools);
    mcp.close().await;
}

// ---- extensions --------------------------------------------------------------------------------

async fn declared_extensions(mcp: &Mcp) -> Value {
    declared_capabilities(&ask(mcp, "meta/echo", json!({})).await.unwrap())["extensions"].clone()
}

// spec: mcp_spec.rb:758 extensions > declares extensions and their settings with every request
#[tokio::test]
async fn declares_extensions_and_their_settings_with_every_request() {
    let mcp = files()
        .extension("com.example/audit", json!({ "level": "full" }))
        .unwrap()
        .extension("com.example/replay", json!({}))
        .unwrap()
        .build()
        .unwrap();
    assert_eq!(
        declared_extensions(&mcp).await,
        json!({ "com.example/audit": { "level": "full" }, "com.example/replay": {} })
    );
    mcp.close().await;
}

// spec: mcp_spec.rb:769 extensions > declares none unless asked
#[tokio::test]
async fn declares_no_extensions_unless_asked() {
    let mcp = files().build().unwrap();
    assert_eq!(declared_extensions(&mcp).await, Value::Null);
    mcp.close().await;
}

// spec: mcp_spec.rb:773 extensions > refuses names without a vendor prefix
#[test]
fn refuses_extension_names_without_a_vendor_prefix() {
    let Err(Error::Argument(message)) = files().extension("audit", json!({})) else {
        panic!("expected an argument error");
    };
    assert!(message.contains("vendor prefix"), "{message}");
}

// spec: mcp_spec.rb:777 extensions > refuses names of extensions it does not know
#[test]
fn refuses_names_of_extensions_it_does_not_know() {
    let Err(Error::Argument(message)) = Extension::parse("widgets") else {
        panic!("expected an argument error");
    };
    assert_eq!(message, "Unknown MCP extension: widgets");
}

// spec: mcp_spec.rb:1224 .mcp > accepts extensions by name or with settings
#[test]
fn accepts_extensions_by_name_or_with_settings() {
    let url = "https://mcp.linear.app/mcp";
    let replay = Mcp::url(url)
        .extension("com.example/replay", json!({}))
        .unwrap();
    assert_eq!(
        Value::Object(replay.extensions().clone()),
        json!({ "com.example/replay": {} })
    );
    let audit = Mcp::url(url)
        .extension("com.example/audit", json!({ "level": "full" }))
        .unwrap();
    assert_eq!(
        Value::Object(audit.extensions().clone()),
        json!({ "com.example/audit": { "level": "full" } })
    );
}

// ---- MCP Apps ----------------------------------------------------------------------------------

fn weather() -> McpBuilder {
    files().with_extension(Extension::Apps, json!({}))
}

// spec: mcp_spec.rb:802 MCP Apps > declares UIs written in HTML
#[tokio::test]
async fn apps_declare_uis_written_in_html() {
    let mcp = weather().build().unwrap();
    assert_eq!(
        declared_extensions(&mcp).await,
        json!({ "io.modelcontextprotocol/ui": { "mimeTypes": ["text/html;profile=mcp-app"] } })
    );
    mcp.close().await;
}

// spec: mcp_spec.rb:809 MCP Apps > declares the settings you give it
#[test]
fn apps_declare_the_settings_you_give_it() {
    let builder = weather().with_extension(
        Extension::Apps,
        json!({ "mimeTypes": ["text/html;profile=mcp-app", "text/uri-list"] }),
    );
    assert_eq!(
        builder.extensions()["io.modelcontextprotocol/ui"],
        json!({ "mimeTypes": ["text/html;profile=mcp-app", "text/uri-list"] })
    );
}

// spec: mcp_spec.rb:816 MCP Apps > lists every tool with its UI and who may call it
#[tokio::test]
async fn apps_list_every_tool_with_its_ui_and_who_may_call_it() {
    let mcp = weather().build().unwrap();
    let forecast = tool(&mcp, "forecast").await;
    assert_eq!(
        (forecast.ui_uri.as_deref(), forecast.visibility.clone()),
        (
            Some("ui://spec/forecast"),
            vec!["model".to_string(), "app".to_string()]
        )
    );
    let refresh = tool(&mcp, "refresh_forecast").await;
    assert_eq!(
        (refresh.ui_uri.as_deref(), refresh.visibility.clone()),
        (Some("ui://spec/forecast"), vec!["app".to_string()])
    );
    let echo = tool(&mcp, "echo").await;
    assert_eq!(
        (echo.ui_uri.as_deref(), echo.visibility.clone()),
        (None, vec!["model".to_string(), "app".to_string()])
    );
    mcp.close().await;
}

// spec: mcp_spec.rb:822 MCP Apps > reads a UI and its content security policy through the resource API
#[tokio::test]
async fn apps_read_a_ui_and_its_content_security_policy() {
    let mcp = weather().build().unwrap();
    let uri = tool(&mcp, "forecast").await.ui_uri.clone().unwrap();
    let view = mcp.resource(&uri).await.unwrap();
    assert_eq!(view.mime_type.as_deref(), Some("text/html;profile=mcp-app"));
    let content = view.content().await.unwrap();
    assert!(content.as_text().unwrap().starts_with("<!DOCTYPE html>"));
    assert_eq!(
        view.meta["ui"],
        json!({ "csp": { "connectDomains": ["https://api.example.com"] }, "prefersBorder": true })
    );
    mcp.close().await;
}

// spec: mcp_spec.rb:830 MCP Apps > calls tools that only a UI may call
#[tokio::test]
async fn apps_call_tools_that_only_a_ui_may_call() {
    let mcp = weather().build().unwrap();
    let result = mcp.call("refresh_forecast", json!({})).await.unwrap();
    assert_eq!(
        (result.text.as_str(), result.structured),
        ("Refreshed", Some(json!({ "fresh": true })))
    );
    mcp.close().await;
}

// spec: mcp_spec.rb:835 MCP Apps > names the UI that renders a result
#[tokio::test]
async fn apps_name_the_ui_that_renders_a_result() {
    let mcp = weather().build().unwrap();
    let result = mcp
        .call("forecast", json!({ "city": "Rome" }))
        .await
        .unwrap();
    assert_eq!(result.ui_uri.as_deref(), Some("ui://spec/forecast"));
    assert_eq!(
        result.structured,
        Some(json!({ "city": "Rome", "temperature": 24 }))
    );
    assert_eq!(
        Value::Object(result.meta),
        json!({ "com.example/station": "spec" })
    );
    let through_tool = tool(&mcp, "forecast")
        .await
        .call(json!({ "city": "Rome" }))
        .await
        .unwrap();
    assert_eq!(
        through_tool.mcp_result.unwrap().ui_uri.as_deref(),
        Some("ui://spec/forecast")
    );
    assert_eq!(
        mcp.call("echo", json!({ "text": "hi" }))
            .await
            .unwrap()
            .ui_uri,
        None
    );
    mcp.close().await;
}

// spec: mcp_spec.rb:844 MCP Apps > keeps a result with a UI on a message across serialization
#[tokio::test]
async fn apps_keep_a_result_with_a_ui_on_a_message_across_serialization() {
    let mcp = weather().build().unwrap();
    let result = mcp
        .call("forecast", json!({ "city": "Rome" }))
        .await
        .unwrap();
    let mut message = Message::tool_result("call_1", result.text.clone());
    message.mcp_result = Some(Box::new(result.clone()));
    let copy = Message::from_h(&message.to_h()).unwrap();
    let kept = copy.mcp_result.unwrap();
    assert_eq!(kept.ui_uri.as_deref(), Some("ui://spec/forecast"));
    assert_eq!(kept.to_h(), result.to_h());
    mcp.close().await;
}

// spec: mcp_spec.rb:853 MCP Apps > reads the URI of a UI written the deprecated way
#[test]
fn apps_read_the_uri_of_a_ui_written_the_deprecated_way() {
    let mcp = weather().build().unwrap();
    let definition = json!({ "name": "chart", "_meta": { "ui/resourceUri": "ui://spec/chart" } });
    let chart = McpTool::new(mcp, &definition, None, Default::default());
    assert_eq!(chart.ui_uri.as_deref(), Some("ui://spec/chart"));
}

// spec: mcp_spec.rb:859 MCP Apps > lists no UI tools to a client that does not declare the extension
#[tokio::test]
async fn apps_list_no_ui_tools_without_the_extension() {
    let plain = files().build().unwrap();
    assert!(!names(&plain).await.contains(&"forecast".to_string()));
    plain.close().await;
}

// ---- tasks -------------------------------------------------------------------------------------

fn reports() -> McpBuilder {
    files().with_extension(Extension::Tasks, json!({}))
}

async fn task_of(mcp: &Mcp, name: &str) -> Task {
    let Err(Error::McpTask(task)) = tool(mcp, name).await.call(json!({})).await else {
        panic!("expected a task");
    };
    *task
}

async fn server_tasks(mcp: &Mcp) -> Value {
    ask(mcp, "spec/tasks", json!({})).await.unwrap()
}

// spec: mcp_spec.rb:881 tasks > declares the tasks extension
#[tokio::test]
async fn tasks_declare_the_extension() {
    let mcp = reports().build().unwrap();
    assert_eq!(
        declared_extensions(&mcp).await,
        json!({ "io.modelcontextprotocol/tasks": {} })
    );
    mcp.close().await;
}

// spec: mcp_spec.rb:887 tasks > hands a chat the task a tool call becomes
#[tokio::test]
async fn tasks_hand_a_chat_the_task_a_call_becomes() {
    let mcp = reports().build().unwrap();
    let task = task_of(&mcp, "report").await;
    assert_eq!(task.id, "task-1");
    assert_eq!(task.status(), TaskStatus::Working);
    assert_eq!(task.status_message(), Some("Queued"));
    assert_eq!(task.poll_interval(), Some(Duration::from_millis(10)));
    assert_eq!(
        task.expires_at().map(|t| t.to_rfc3339()),
        Some("2026-10-02T10:01:00+00:00".to_string())
    );
    assert!(task.result().unwrap().is_none());
    assert!(!task.is_done());
    mcp.close().await;
}

// spec: mcp_spec.rb:896 tasks > checks on a task once each time you refresh it
#[tokio::test]
async fn tasks_check_once_each_time_you_refresh() {
    let mcp = reports().build().unwrap();
    let mut task = task_of(&mcp, "report").await;
    task.refresh().await.unwrap();
    assert_eq!(
        (task.status(), task.status_message()),
        (TaskStatus::Working, Some("Rendering"))
    );
    task.refresh().await.unwrap();
    assert!(task.is_completed());
    let result = task.result().unwrap().unwrap();
    assert_eq!(
        (result.text.as_str(), result.structured),
        ("Report ready", Some(json!({ "pages": 2 })))
    );
    assert!(task.refresh().await.unwrap().is_done());
    assert_eq!(server_tasks(&mcp).await["polls"], json!({ "task-1": 2 }));
    mcp.close().await;
}

// spec: mcp_spec.rb:906 tasks > waits for a task
#[tokio::test]
async fn tasks_wait() {
    let mcp = reports().build().unwrap();
    let mut task = task_of(&mcp, "report").await;
    let done = task.wait(None, None).await.unwrap();
    assert_eq!(done.result().unwrap().unwrap().text, "Report ready");
    mcp.close().await;
}

// spec: mcp_spec.rb:910 tasks > waits for the task of a tool you call directly
#[tokio::test]
async fn tasks_wait_for_the_task_of_a_direct_call() {
    let mcp = reports().build().unwrap();
    let result = mcp.call("report", json!({})).await.unwrap();
    assert_eq!(
        (result.text.as_str(), result.structured),
        ("Report ready", Some(json!({ "pages": 2 })))
    );
    mcp.close().await;
}

// spec: mcp_spec.rb:914 tasks > reports what a task is doing as progress
#[tokio::test]
async fn tasks_report_what_they_do_as_progress() {
    let messages = Arc::new(Mutex::new(Vec::new()));
    let recorder = messages.clone();
    let mcp = reports()
        .after_progress(move |p| recorder.lock().unwrap().push(p.message.clone()))
        .build()
        .unwrap();
    mcp.call("report", json!({})).await.unwrap();
    assert_eq!(*messages.lock().unwrap(), [Some("Rendering".to_string())]);
    mcp.close().await;
}

// spec: mcp_spec.rb:923 tasks > raises the error a task failed with
#[tokio::test]
async fn tasks_raise_the_error_they_failed_with() {
    let mcp = reports().build().unwrap();
    let Err(Error::Mcp(e)) = mcp.call("broken_report", json!({})).await else {
        panic!("expected an MCP error");
    };
    assert_eq!(
        (e.message.as_str(), e.code),
        ("Renderer crashed", Some(-32_603))
    );
    let mut task = task_of(&mcp, "broken_report").await;
    assert!(task.refresh().await.unwrap().is_failed());
    mcp.close().await;
}

// spec: mcp_spec.rb:930 tasks > answers the input requests of a task with callbacks
#[tokio::test]
async fn tasks_answer_their_input_requests_with_callbacks() {
    let mcp = reports()
        .before_input_request(|r| r.answer(map(json!({ "approved": true }))))
        .build()
        .unwrap();
    assert_eq!(
        mcp.call("approve_report", json!({})).await.unwrap().text,
        "Approved: true"
    );
    mcp.close().await;
}

// spec: mcp_spec.rb:936 tasks > raises when no callback answers the input requests of a task
#[tokio::test]
async fn tasks_raise_when_no_callback_answers_their_input_requests() {
    let mcp = reports().build().unwrap();
    let Err(Error::McpInputRequired(e)) = mcp.call("approve_report", json!({})).await else {
        panic!("expected an input-required error");
    };
    assert!(e.message.contains("Publish the report"), "{}", e.message);
    mcp.close().await;
}

// spec: mcp_spec.rb:940 tasks > cancels a task
#[tokio::test]
async fn tasks_cancel() {
    let mcp = reports().build().unwrap();
    let mut task = task_of(&mcp, "endless_report").await;
    let id = task.id.clone();
    assert_eq!(task.cancel().await.unwrap().id, id);
    assert_eq!(server_tasks(&mcp).await["cancelled"], json!([id]));
    assert!(task.refresh().await.unwrap().is_cancelled());
    mcp.close().await;
}

// spec: mcp_spec.rb:948 tasks > stops waiting for a task after the timeout and leaves it to you
#[tokio::test]
async fn tasks_stop_waiting_after_the_timeout() {
    let mcp = reports().build().unwrap();
    let mut task = task_of(&mcp, "endless_report").await;
    let Err(Error::Mcp(e)) = task.wait(Some(Duration::from_millis(50)), None).await else {
        panic!("expected an MCP error");
    };
    assert_eq!(e.message, "Task task-1 did not finish in 0.05 seconds");
    assert_eq!(server_tasks(&mcp).await["cancelled"], json!([]));
    mcp.close().await;
}

// spec: mcp_spec.rb:957 tasks > cancels the task of a direct call when the chat is cancelled
#[tokio::test]
async fn tasks_cancel_the_task_of_a_direct_call_when_the_chat_is_cancelled() {
    let mcp = reports().build().unwrap();
    let flag = Arc::new(AtomicBool::new(false));
    let setter = flag.clone();
    // Ruby raises at the first checkpoint after the task exists; here the chat is cancelled
    // shortly after the call starts, while the direct call waits on the task.
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(300)).await;
        setter.store(true, Ordering::SeqCst);
    });
    let result = rust_llm::progress::watch(flag, mcp.call("endless_report", json!({}))).await;
    assert!(matches!(result, Err(Error::Cancelled)), "{result:?}");
    assert_eq!(server_tasks(&mcp).await["cancelled"], json!(["task-1"]));
    mcp.close().await;
}

// ---- log messages ------------------------------------------------------------------------------

fn logged(builder: McpBuilder) -> Vec<(tracing::Level, String)> {
    let (dispatch, events) = collector();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    {
        let _guard = tracing::dispatcher::set_default(&dispatch);
        runtime.block_on(async {
            let mcp = builder.build().unwrap();
            mcp.call("echo", json!({ "text": "hi" })).await.unwrap();
            mcp.close().await;
        });
    }
    let events = events.lock().unwrap().clone();
    events
        .into_iter()
        .filter(|(_, text)| text.starts_with("files (echo):"))
        .collect()
}

// spec: mcp_spec.rb:978 log messages > writes the messages at the level it asks for and above to the RubyLLM logger
#[test]
fn log_messages_at_the_level_asked_for_and_above_reach_the_log() {
    assert_eq!(
        logged(files().log_level(LogLevel::Warning)),
        [(
            tracing::Level::WARN,
            r#"files (echo): {"slow":true}"#.to_string()
        )]
    );
}

// spec: mcp_spec.rb:987 log messages > asks for messages down to debug
#[test]
fn log_messages_down_to_debug() {
    assert!(
        logged(files().log_level(LogLevel::Debug))
            .contains(&(tracing::Level::DEBUG, "files (echo): Echoing".to_string()))
    );
}

// spec: mcp_spec.rb:995 log messages > asks for no messages unless you set a level
#[test]
fn no_log_messages_unless_a_level_is_set() {
    assert!(logged(files()).is_empty());
}

// spec: mcp_spec.rb:1001 log messages > refuses levels the protocol does not define
#[test]
fn refuses_log_levels_the_protocol_does_not_define() {
    let Err(Error::Argument(message)) = LogLevel::parse("verbose") else {
        panic!("expected an argument error");
    };
    assert_eq!(message, "Unknown MCP log level: verbose");
}

// spec: mcp_spec.rb:1006 log messages > takes a level inline
#[test]
fn takes_a_log_level_inline() {
    let builder =
        Mcp::url("https://mcp.linear.app/mcp").log_level(LogLevel::parse("info").unwrap());
    assert_eq!(builder.configured_log_level(), Some(LogLevel::Info));
}

// ---- transport ---------------------------------------------------------------------------------

// spec: mcp_spec.rb:1179 transport > does not keep a tool list the server changed while sending it
#[tokio::test]
async fn does_not_keep_a_tool_list_the_server_changed_while_sending_it() {
    let transport = Scripted::new(|method, _, on_notification| match method {
        "server/discover" => json!({ "result": { "supportedVersions": ["2026-07-28"] } }),
        "tools/list" => {
            on_notification(
                &json!({ "jsonrpc": "2.0", "method": "notifications/tools/list_changed" }),
            );
            json!({ "result": { "tools": [{ "name": "echo" }] } })
        }
        _ => json!({ "result": {} }),
    });
    let mcp = Mcp::transport("tunnelled", transport.clone())
        .build()
        .unwrap();
    for _ in 0..2 {
        mcp.tools().await.unwrap();
    }
    assert_eq!(
        transport
            .sent()
            .iter()
            .filter(|m| *m == "tools/list")
            .count(),
        2
    );
}

// ---- mcp/result_spec ---------------------------------------------------------------------------

fn offline() -> Value {
    json!([{ "type": "text", "text": "The laptop is offline" }])
}

// spec: mcp/result_spec.rb:8 takes the data without braces
#[test]
fn a_result_takes_the_data_without_braces() {
    let result = McpResult::new(json!({ "content": offline(), "isError": true }), None);
    assert_eq!(
        (result.text.as_str(), result.ui_uri.as_deref()),
        ("The laptop is offline", None)
    );
    assert!(result.is_error());
}

// spec: mcp/result_spec.rb:15 takes the data in braces
#[test]
fn a_result_takes_the_data_in_braces() {
    // Ruby's two calls (`new('content' => ...)` and `new({ 'content' => ... })`) differ only
    // in braces; Rust has one way to pass the data.
    let data = json!({ "content": offline(), "isError": true });
    let result = McpResult::new(data, None);
    assert_eq!(result.text, "The laptop is offline");
    assert!(result.is_error());
}

// spec: mcp/result_spec.rb:22 takes the URI of the UI that renders it
#[test]
fn a_result_takes_the_uri_of_the_ui_that_renders_it() {
    let result = McpResult::new(
        json!({ "content": offline() }),
        Some("ui://laptop/files".into()),
    );
    assert_eq!(
        (result.text.as_str(), result.ui_uri.as_deref()),
        ("The laptop is offline", Some("ui://laptop/files"))
    );
}

// ---- mcp/input_request_spec --------------------------------------------------------------------

fn form(properties: Value) -> InputRequest {
    InputRequest::new(
        "profile",
        json!({ "message": "Your profile", "requestedSchema": { "properties": properties } }),
    )
}

// spec: mcp/input_request_spec.rb:10 fills in the defaults of the fields an answer leaves out
#[test]
fn an_answer_fills_in_the_defaults_of_the_fields_it_leaves_out() {
    let mut request = form(json!({
        "name": { "type": "string", "default": "John Doe" },
        "age": { "type": "integer", "default": 30 },
        "verified": { "type": "boolean", "default": false },
        "email": { "type": "string" }
    }));
    request.answer(map(json!({ "age": 31 })));
    assert_eq!(
        request.response,
        Some(
            json!({ "action": "accept", "content": { "name": "John Doe", "age": 31, "verified": false } })
        )
    );
}

// spec: mcp/input_request_spec.rb:22 keeps the values an answer gives
#[test]
fn an_answer_keeps_the_values_it_gives() {
    let mut request = form(
        json!({ "status": { "type": "string", "enum": ["active", "inactive"], "default": "active" } }),
    );
    request.answer(map(json!({ "status": "inactive" })));
    assert_eq!(
        request.response.unwrap()["content"],
        json!({ "status": "inactive" })
    );
}

// spec: mcp/input_request_spec.rb:30 accepts a URL request without content
#[test]
fn a_url_request_is_accepted_without_content() {
    let mut request = InputRequest::new(
        "connect",
        json!({ "mode": "url", "url": "https://example.com/connect" }),
    );
    request.answer(Map::new());
    assert_eq!(request.response, Some(json!({ "action": "accept" })));
}

// ---- mcp/client_spec ---------------------------------------------------------------------------

fn stdio_client(env: &[(&str, &str)]) -> Arc<Client> {
    let env = env
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    Arc::new(Client::new(
        Arc::new(Stdio::new(
            vec!["ruby".into(), server_path()],
            env,
            None,
            Duration::from_secs(30),
        )),
        json!({}),
    ))
}

// spec: mcp/client_spec.rb:68 with a server that predates 2026-07-28 > reports a change once the request that carried it is answered, so the report can make requests
#[tokio::test]
async fn a_change_is_reported_once_its_request_is_answered() {
    let changing = stdio_client(&[("MCP_ERA", "legacy")]);
    let sizes = Arc::new(Mutex::new(Vec::new()));
    let (weak, recorder) = (Arc::downgrade(&changing), sizes.clone());
    changing.on_change(move |_| {
        let (client, recorder) = (weak.upgrade(), recorder.clone());
        async move {
            if let Some(client) = client {
                let size = client.list("tools/list", "tools").await.unwrap().len();
                recorder.lock().unwrap().push(size);
            }
        }
    });
    changing
        .request("spec/change_tools", json!({}), &[], &mut |_| {})
        .await
        .unwrap();
    assert_eq!(*sizes.lock().unwrap(), [10]);
    changing.close().await;
}

fn unsupported(versions: &[&str]) -> Value {
    json!({ "error": { "code": -32_022, "message": "Unsupported protocol version",
                       "data": { "supported": versions, "requested": "2026-07-28" } } })
}

fn handshake(version: &str) -> Value {
    json!({ "result": { "protocolVersion": version, "capabilities": {} } })
}

fn legacy_server(version: &'static str) -> Arc<Scripted> {
    Scripted::new(move |method, _, _| {
        if method == "initialize" {
            handshake(version)
        } else {
            json!({ "error": { "code": -32_601, "message": "Not found" } })
        }
    })
}

// spec: mcp/client_spec.rb:122 protocol versions > sends a request again when the server rejects a version it lists
#[tokio::test]
async fn sends_a_request_again_when_the_server_rejects_a_version_it_lists() {
    let server = Scripted::new(|method, attempt, _| {
        if method == "server/discover" && attempt == 1 {
            return unsupported(&["2026-07-28"]);
        }
        json!({ "result": { "supportedVersions": ["2026-07-28"] } })
    });
    let client = Client::new(server.clone(), json!({}));
    assert_eq!(
        client.server().await.unwrap(),
        json!({ "supportedVersions": ["2026-07-28"] })
    );
    assert!(client.is_modern());
    assert_eq!(server.sent(), ["server/discover", "server/discover"]);
}

// spec: mcp/client_spec.rb:135 protocol versions > sends a request again only once
#[tokio::test]
async fn sends_a_request_again_only_once() {
    let server = Scripted::new(|_, _, _| unsupported(&["2026-07-28"]));
    let Err(Error::Mcp(e)) = Client::new(server.clone(), json!({})).server().await else {
        panic!("expected an MCP error");
    };
    assert_eq!(e.message, "Unsupported protocol version");
    assert_eq!(server.sent(), ["server/discover", "server/discover"]);
}

// spec: mcp/client_spec.rb:142 protocol versions > does not shake hands with a modern server that lists only older versions
#[tokio::test]
async fn does_not_shake_hands_with_a_modern_server_listing_only_older_versions() {
    let server = Scripted::new(|_, _, _| unsupported(&["2025-11-25"]));
    let Err(Error::Mcp(e)) = Client::new(server.clone(), json!({})).server().await else {
        panic!("expected an MCP error");
    };
    assert_eq!(e.message, "Unsupported protocol version");
    assert!(!server.sent().contains(&"initialize".to_string()));
}

// spec: mcp/client_spec.rb:149 protocol versions > speaks every version it knows from before 2026-07-28
#[tokio::test]
async fn speaks_every_version_it_knows_from_before_2026_07_28() {
    assert_eq!(
        LEGACY_VERSIONS,
        ["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"]
    );
    for version in LEGACY_VERSIONS {
        let client = Client::new(legacy_server(version), json!({}));
        client.server().await.unwrap();
        assert_eq!(client.version().as_deref(), Some(version));
    }
}

// spec: mcp/client_spec.rb:159 protocol versions > declares only its extensions in the handshake
#[tokio::test]
async fn declares_only_its_extensions_in_the_handshake() {
    let server = legacy_server("2025-06-18");
    let capabilities =
        json!({ "elicitation": { "form": {} }, "extensions": { "com.example/audit": {} } });
    Client::new(server.clone(), capabilities)
        .server()
        .await
        .unwrap();
    assert_eq!(
        server.params("initialize")["capabilities"],
        json!({ "extensions": { "com.example/audit": {} } })
    );
}

// spec: mcp/client_spec.rb:168 protocol versions > disconnects from a server that answers the handshake with a version it does not speak
#[tokio::test]
async fn disconnects_from_a_server_answering_the_handshake_with_an_unknown_version() {
    let server = legacy_server("2099-01-01");
    let client = Client::new(server.clone(), json!({}));
    let Err(Error::Mcp(e)) = client.server().await else {
        panic!("expected an MCP error");
    };
    assert!(
        e.message.contains("protocol version 2099-01-01"),
        "{}",
        e.message
    );
    assert_eq!(client.version(), None);
    assert!(server.closed.load(Ordering::SeqCst));
    assert!(
        !server
            .sent()
            .contains(&"notifications/initialized".to_string())
    );
}
