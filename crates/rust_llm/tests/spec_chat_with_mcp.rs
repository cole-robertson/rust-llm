//! RubyLLM 2.1's chat-side MCP additions from `spec/ruby_llm/chat_with_mcp_spec.rb`: tool
//! results as 2.0 stored them, form defaults and declared input requests, MCP Apps, tasks, and
//! deferred servers. The model is stubbed with canned Anthropic responses; the servers are
//! RubyLLM's spec server (`tests/fixtures/mcp/server.rb`) or a custom transport.

mod spec_helpers;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use rust_llm::mcp::{Extension, Mcp, McpBuilder, OnNotification, TaskStatus, Transport};
use rust_llm::{Agent, Chat, Error, Role, SharedTool, ToolResult};
use serde_json::{Map, Value, json};
use spec_helpers::{text_response, tool_use_response};

fn server_path() -> String {
    format!(
        "{}/tests/fixtures/mcp/server.rb",
        env!("CARGO_MANIFEST_DIR")
    )
}

/// `Class.new(RubyLLM::MCP) { command(RbConfig.ruby, server) }`, named `files`.
fn files() -> McpBuilder {
    Mcp::command(["ruby".to_string(), server_path()]).name("files")
}

/// A chat whose model calls `name` with `arguments`, then answers "Done".
async fn chat_calling(name: &str, arguments: Value) -> (Chat, wiremock::MockServer) {
    let server = spec_helpers::serve(vec![
        tool_use_response(&[("call_1", name, arguments)]),
        text_response("Done"),
    ])
    .await;
    (spec_helpers::chat(&server), server)
}

fn tool_message(chat: &Chat) -> rust_llm::Message {
    chat.messages()
        .iter()
        .find(|m| m.role == Role::Tool)
        .cloned()
        .expect("a tool result message")
}

async fn server_tasks(mcp: &Mcp) -> Value {
    mcp.client()
        .request("spec/tasks", json!({}), &[], &mut |_| {})
        .await
        .unwrap()
}

async fn names(tools: Vec<SharedTool>) -> Vec<String> {
    tools.iter().map(|t| t.name()).collect()
}

// ---- tool results ------------------------------------------------------------------------------

/// The `laptop_transport`: tool calls fail with `isError`.
struct Laptop;

#[async_trait]
impl Transport for Laptop {
    async fn request(
        &self,
        message: &Value,
        _: Option<&str>,
        _: Option<Duration>,
        _: &[(String, String)],
        _: &mut OnNotification<'_>,
    ) -> rust_llm::Result<Value> {
        let result = match message["method"].as_str() {
            Some("server/discover") => json!({ "supportedVersions": ["2026-07-28"] }),
            Some("tools/list") => json!({ "tools": [{ "name": "read_notes", "inputSchema": {} }] }),
            _ => {
                json!({ "isError": true, "content": [{ "type": "text", "text": "The laptop is offline" }] })
            }
        };
        Ok(json!({ "jsonrpc": "2.0", "id": message["id"], "result": result }))
    }
    async fn notify(&self, _: &Value, _: Option<&str>) -> rust_llm::Result<()> {
        Ok(())
    }
    async fn cancel(&self, _: &Value, _: Option<&str>) -> rust_llm::Result<()> {
        Ok(())
    }
    async fn close(&self) {}
}

// spec: chat_with_mcp_spec.rb:77 tool results > stores and reports a failed call as 2.0 did, through a custom transport
#[tokio::test]
async fn stores_and_reports_a_failed_call_as_2_0_did() {
    let laptop = Mcp::transport("laptop", Arc::new(Laptop)).build().unwrap();
    let (chat, _server) = chat_calling("read_notes", json!({})).await;
    let results = Arc::new(Mutex::new(Vec::new()));
    let recorder = results.clone();
    let mut chat = chat
        .with_mcp(laptop)
        .after_tool_result(move |r| recorder.lock().unwrap().push(r.clone()));
    chat.ask("Read my notes").await.unwrap();
    assert_eq!(
        tool_message(&chat).content(),
        r#"{"error":"The laptop is offline"}"#
    );
    assert_eq!(
        *results.lock().unwrap(),
        [ToolResult::error("The laptop is offline")]
    );
}

// spec: chat_with_mcp_spec.rb:88 tool results > stores a successful result as before
#[tokio::test]
async fn stores_a_successful_result_as_before() {
    let files = files().build().unwrap();
    let (chat, _server) = chat_calling("picture", json!({})).await;
    let mut chat = chat.with_mcp(files.clone());
    chat.ask("Show me").await.unwrap();
    let message = tool_message(&chat);
    assert_eq!(
        message.content(),
        "Here it is\n\npixel.png: file:///pixel.png"
    );
    assert_eq!(message.attachments[0].mime_type, "image/png");
    files.close().await;
}

// ---- input requests ----------------------------------------------------------------------------

// spec: chat_with_mcp_spec.rb:170 input requests > answers with the defaults the server gave
#[tokio::test]
async fn answers_with_the_defaults_the_server_gave() {
    let files = files().build().unwrap();
    let (chat, _server) = chat_calling("deploy", json!({})).await;
    let mut chat = chat.with_mcp(files.clone());
    chat.ask("Deploy").await.unwrap();
    let request = chat.pending_inputs().remove(0);
    chat.answer(&request, Map::new()).unwrap();
    chat.complete().await.unwrap();
    assert_eq!(tool_message(&chat).content(), "Deployed to staging");
    files.close().await;
}

// spec: chat_with_mcp_spec.rb:184 input requests > does not pause for input requests the MCP does not accept
#[tokio::test]
async fn does_not_pause_for_input_requests_the_mcp_does_not_accept() {
    let files = files().input_requests(&[]).build().unwrap();
    let (chat, _server) = chat_calling("deploy", json!({})).await;
    let mut chat = chat.with_mcp(files.clone());
    chat.ask("Deploy").await.unwrap();
    assert!(!chat.is_awaiting_input());
    assert_eq!(tool_message(&chat).content(), "Deploy cancelled");
    files.close().await;
}

// ---- MCP Apps ----------------------------------------------------------------------------------

fn weather() -> McpBuilder {
    files().with_extension(Extension::Apps, json!({}))
}

// spec: chat_with_mcp_spec.rb:223 MCP Apps > never offers the model tools that only a UI may call
#[tokio::test]
async fn never_offers_the_model_tools_that_only_a_ui_may_call() {
    let files = weather().build().unwrap();
    let (chat, _server) = chat_calling("forecast", json!({})).await;
    let chat = chat.with_mcp(files.clone());
    let offered = names(chat.all_tools().await.unwrap()).await;
    assert!(offered.contains(&"forecast".to_string()));
    assert!(!offered.contains(&"refresh_forecast".to_string()));
    assert!(
        names(files.tools().await.unwrap())
            .await
            .contains(&"refresh_forecast".to_string())
    );
    files.close().await;
}

// spec: chat_with_mcp_spec.rb:231 MCP Apps > keeps those tools from the model when given to the chat directly
#[tokio::test]
async fn keeps_app_only_tools_from_the_model_when_given_directly() {
    let files = weather().build().unwrap();
    let refresh = files
        .tools()
        .await
        .unwrap()
        .into_iter()
        .find(|t| t.name() == "refresh_forecast")
        .unwrap();
    let (chat, _server) = chat_calling("forecast", json!({})).await;
    let chat = chat.with_tools([refresh]);
    assert!(chat.all_tools().await.unwrap().is_empty());
    files.close().await;
}

// spec: chat_with_mcp_spec.rb:237 MCP Apps > keeps the result of a tool with a UI on its tool result message
#[tokio::test]
async fn keeps_the_result_of_a_tool_with_a_ui_on_its_tool_result_message() {
    let files = weather().build().unwrap();
    let (chat, _server) = chat_calling("forecast", json!({ "city": "Rome" })).await;
    let mut chat = chat.with_mcp(files.clone());
    chat.ask("How is the weather in Rome?").await.unwrap();
    let message = tool_message(&chat);
    assert_eq!(message.content(), "Sunny in Rome");
    let result = message.mcp_result.expect("the result the UI renders");
    assert_eq!(result.ui_uri.as_deref(), Some("ui://spec/forecast"));
    assert_eq!(
        result.structured,
        Some(json!({ "city": "Rome", "temperature": 24 }))
    );
    assert_eq!(
        Value::Object(result.meta),
        json!({ "com.example/station": "spec" })
    );
    files.close().await;
}

// spec: chat_with_mcp_spec.rb:250 MCP Apps > keeps no result for tools without a UI
#[tokio::test]
async fn keeps_no_result_for_tools_without_a_ui() {
    let files = weather().build().unwrap();
    let (chat, _server) = chat_calling("add", json!({ "a": 2, "b": 3 })).await;
    let mut chat = chat.with_mcp(files.clone());
    chat.ask("What is 2 + 3?").await.unwrap();
    assert!(tool_message(&chat).mcp_result.is_none());
    files.close().await;
}

// spec: chat_with_mcp_spec.rb:258 MCP Apps > hands the server result to after_tool_result
#[tokio::test]
async fn hands_the_server_result_to_after_tool_result() {
    let files = weather().build().unwrap();
    let (chat, _server) = chat_calling("forecast", json!({ "city": "Rome" })).await;
    let results = Arc::new(Mutex::new(Vec::new()));
    let recorder = results.clone();
    let mut chat = chat
        .with_mcp(files.clone())
        .after_tool_result(move |r| recorder.lock().unwrap().push(r.clone()));
    chat.ask("How is the weather in Rome?").await.unwrap();
    let first = results.lock().unwrap()[0].clone();
    assert_eq!(
        first.mcp_result.unwrap().structured,
        Some(json!({ "city": "Rome", "temperature": 24 }))
    );
    files.close().await;
}

// ---- tasks -------------------------------------------------------------------------------------

fn reports() -> McpBuilder {
    files().with_extension(Extension::Tasks, json!({}))
}

/// `run_task(name)`: the model calls the task tool, then answers "Done".
async fn run_task(name: &str) -> (Chat, Mcp, wiremock::MockServer) {
    let files = reports().build().unwrap();
    let (chat, server) = chat_calling(name, json!({})).await;
    let mut chat = chat.with_mcp(files.clone());
    chat.ask("Run it").await.unwrap();
    (chat, files, server)
}

// spec: chat_with_mcp_spec.rb:278 tasks > pauses a tool call that becomes a task, without waiting for it
#[tokio::test]
async fn pauses_a_tool_call_that_becomes_a_task() {
    let (chat, files, _server) = run_task("report").await;
    assert!(chat.is_awaiting_tasks());
    assert!(chat.is_waiting());
    assert!(!chat.is_complete());
    let task = chat.pending_tasks().remove(0);
    assert_eq!(task.id, "task-1");
    assert_eq!(task.status(), TaskStatus::Working);
    assert_eq!(task.poll_interval(), Some(Duration::from_millis(10)));
    assert_eq!(task.tool_call.map(|c| c.name).as_deref(), Some("report"));
    assert_eq!(server_tasks(&files).await["polls"], json!({ "task-1": 0 }));
    files.close().await;
}

// spec: chat_with_mcp_spec.rb:289 tasks > checks on each task once every time it completes, and resumes once they finish
#[tokio::test]
async fn checks_on_each_task_once_per_complete_and_resumes_once_they_finish() {
    let (mut chat, files, _server) = run_task("report").await;
    chat.complete().await.unwrap();
    assert!(chat.is_awaiting_tasks());
    let task = chat.pending_tasks().remove(0);
    assert_eq!(
        (task.status(), task.status_message()),
        (TaskStatus::Working, Some("Rendering"))
    );
    chat.complete().await.unwrap();
    assert!(chat.is_complete());
    assert!(!chat.is_awaiting_tasks());
    assert_eq!(tool_message(&chat).content(), "Report ready");
    assert_eq!(server_tasks(&files).await["polls"], json!({ "task-1": 2 }));
    files.close().await;
}

// spec: chat_with_mcp_spec.rb:303 tasks > checks on a task without resuming the chat
#[tokio::test]
async fn checks_on_a_task_without_resuming_the_chat() {
    let (mut chat, files, _server) = run_task("report").await;
    let mut task = chat.pending_tasks().remove(0);
    task.refresh().await.unwrap();
    task.refresh().await.unwrap();
    assert!(task.is_completed());
    assert!(chat.is_awaiting_tasks());
    assert_eq!(chat.complete().await.unwrap().content(), "Done");
    files.close().await;
}

// spec: chat_with_mcp_spec.rb:313 tasks > reports what a task is doing to after_tool_progress
#[tokio::test]
async fn reports_what_a_task_is_doing_to_after_tool_progress() {
    let (chat, files, _server) = run_task("report").await;
    let reports = Arc::new(Mutex::new(Vec::new()));
    let recorder = reports.clone();
    let mut chat = chat.after_tool_progress(move |call, progress| {
        recorder
            .lock()
            .unwrap()
            .push((call.name.clone(), progress.message.clone()))
    });
    chat.complete().await.unwrap();
    assert_eq!(
        *reports.lock().unwrap(),
        [("report".to_string(), Some("Rendering".to_string()))]
    );
    files.close().await;
}

// spec: chat_with_mcp_spec.rb:322 tasks > pauses on the input requests of a task until the user answers
#[tokio::test]
async fn pauses_on_the_input_requests_of_a_task_until_the_user_answers() {
    let (mut chat, files, _server) = run_task("approve_report").await;
    chat.complete().await.unwrap();
    assert!(chat.is_awaiting_input());
    assert!(!chat.is_awaiting_tasks());
    let request = chat.pending_inputs().remove(0);
    assert_eq!(request.message.as_deref(), Some("Publish the report?"));
    assert_eq!(
        request.tool_call.as_ref().map(|c| c.name.as_str()),
        Some("approve_report")
    );
    chat.answer(&request, spec_helpers::args(json!({ "approved": true })))
        .unwrap();
    chat.complete().await.unwrap();
    assert!(chat.is_complete());
    assert_eq!(tool_message(&chat).content(), "Approved: true");
    files.close().await;
}

// spec: chat_with_mcp_spec.rb:338 tasks > cancels its tasks when the chat is cancelled
#[tokio::test]
async fn cancels_its_tasks_when_the_chat_is_cancelled() {
    let (mut chat, files, _server) = run_task("endless_report").await;
    let task = chat.pending_tasks().remove(0);
    chat.cancel();
    let result = chat.complete().await;
    assert!(matches!(result, Err(Error::Cancelled)), "{result:?}");
    assert_eq!(server_tasks(&files).await["cancelled"], json!([task.id]));
    assert!(!chat.is_awaiting_tasks());
    assert!(!chat.is_waiting());
    files.close().await;
}

// spec: chat_with_mcp_spec.rb:350 tasks > raises the error a task failed with
#[tokio::test]
async fn raises_the_error_a_task_failed_with() {
    let (mut chat, files, _server) = run_task("broken_report").await;
    let Err(Error::Mcp(e)) = chat.complete().await else {
        panic!("expected an MCP error");
    };
    assert_eq!(e.message, "Renderer crashed");
    files.close().await;
}

// spec: chat_with_mcp_spec.rb:356 tasks > refuses a new question while a task runs
#[tokio::test]
async fn refuses_a_new_question_while_a_task_runs() {
    let (mut chat, files, _server) = run_task("report").await;
    let err = chat.ask("Anything else?").await.unwrap_err();
    assert!(
        matches!(&err, Error::PendingToolCalls(m) if m.contains("waiting for tasks")),
        "{err}"
    );
    files.close().await;
}

// ---- deferred tools ----------------------------------------------------------------------------

/// `chat.render[:tools].to_h { |tool| [tool[:name], tool[:defer_loading]] }`.
fn rendered(chat: &Chat) -> Map<String, Value> {
    chat.render().unwrap()["tools"]
        .as_array()
        .cloned()
        .unwrap_or_default()
        .iter()
        .map(|t| {
            (
                t["name"].as_str().unwrap_or_default().to_string(),
                t.get("defer_loading").cloned().unwrap_or(Value::Null),
            )
        })
        .collect()
}

/// Ruby's `chat.tools` contacts the servers; here `all_tools` does, before the sync readers.
async fn loaded(chat: &Chat) -> Vec<String> {
    names(chat.all_tools().await.unwrap()).await
}

fn deferred(chat: &Chat) -> Vec<String> {
    chat.deferred_tools().iter().map(|t| t.name()).collect()
}

// spec: chat_with_mcp_spec.rb:379 deferred tools > defers every tool of a server connected with defer: true
#[tokio::test]
async fn defers_every_tool_of_a_server_connected_with_defer_true() {
    let files = files().build().unwrap();
    let (chat, _server) = chat_calling("echo", json!({})).await;
    let chat = chat.with_mcp_deferred(files.clone(), Some(true));
    let tools = loaded(&chat).await;
    assert_eq!(deferred(&chat), tools);
    let rendered = rendered(&chat);
    assert_eq!(rendered["echo"], json!(true));
    assert_eq!(rendered["add"], json!(true));
    assert_eq!(rendered["tool_search_tool_bm25"], Value::Null);
    files.close().await;
}

// spec: chat_with_mcp_spec.rb:386 deferred tools > defers the tools a server class declares
#[tokio::test]
async fn defers_the_tools_a_server_declares() {
    let files = files().defer(&["echo"]).build().unwrap();
    let (chat, _server) = chat_calling("echo", json!({})).await;
    let chat = chat.with_mcp(files.clone());
    loaded(&chat).await;
    assert_eq!(deferred(&chat), ["echo"]);
    files.close().await;
}

// spec: chat_with_mcp_spec.rb:392 deferred tools > offers declared tools up front with defer: false
#[tokio::test]
async fn offers_declared_tools_up_front_with_defer_false() {
    let files = files().defer(&[] as &[&str]).build().unwrap();
    let (chat, _server) = chat_calling("echo", json!({})).await;
    let chat = chat.with_mcp_deferred(files.clone(), Some(false));
    loaded(&chat).await;
    assert!(deferred(&chat).is_empty());
    files.close().await;
}

// spec: chat_with_mcp_spec.rb:398 deferred tools > forgets deferrals when the servers are disconnected
#[tokio::test]
async fn forgets_deferrals_when_the_servers_are_disconnected() {
    let files = files().build().unwrap();
    let (chat, _server) = chat_calling("echo", json!({})).await;
    let mut chat = chat.with_mcp_deferred(files.clone(), Some(true));
    chat.clear_mcp();
    let chat = chat.with_mcp(files.clone());
    loaded(&chat).await;
    assert!(deferred(&chat).is_empty());
    files.close().await;
}

struct DeferredServer(Mcp);

impl Agent for DeferredServer {
    fn mcp(&self) -> Vec<Mcp> {
        vec![self.0.clone()]
    }
    fn mcp_defer(&self) -> Option<bool> {
        Some(true)
    }
}

// spec: chat_with_mcp_spec.rb:404 deferred tools > connects deferred servers declared on an agent
#[tokio::test]
async fn connects_deferred_servers_declared_on_an_agent() {
    let files = files().build().unwrap();
    let (chat, _server) = chat_calling("echo", json!({})).await;
    let chat = DeferredServer(files.clone()).apply(chat).unwrap();
    loaded(&chat).await;
    let deferred = deferred(&chat);
    assert!(deferred.contains(&"echo".to_string()) && deferred.contains(&"add".to_string()));
    files.close().await;
}
