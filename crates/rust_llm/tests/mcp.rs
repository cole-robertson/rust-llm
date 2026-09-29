//! MCP client, transports, and `chat.with_mcp`, mirroring RubyLLM's `spec/ruby_llm/mcp_spec.rb`,
//! `mcp/client_spec.rb`, `mcp/http_spec.rb`, and `chat_with_mcp_spec.rb`. The stdio tests run
//! RubyLLM's own spec server (`tests/fixtures/mcp/server.rb`, copied verbatim); the Streamable
//! HTTP tests use wiremock; the chat tests stub the model with canned Anthropic responses.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use base64::Engine;
use rust_llm::mcp::{Client, Http, InputRequest, Mcp, McpBuilder, OnNotification, Stdio, ToolShape, Transport};
use rust_llm::{Agent, Chat, Config, Error, Role, SharedTool, Tool, ToolCall, ToolError, ToolResult};
use serde_json::{Map, Value, json};
use wiremock::matchers::{body_partial_json, method};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

const TOOL_NAMES: [&str; 9] = ["echo", "add", "fail", "picture", "slow", "wait", "deploy", "connect", "delete_everything"];

fn server_path() -> String {
    format!("{}/tests/fixtures/mcp/server.rb", env!("CARGO_MANIFEST_DIR"))
}

fn files() -> McpBuilder {
    Mcp::command(["ruby".to_string(), server_path()]).name("files")
}

fn stdio(env: &[(&str, &str)], timeout: Duration) -> Client {
    let env = env.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
    Client::new(Arc::new(Stdio::new(vec!["ruby".into(), server_path()], env, None, timeout)), json!({}))
}

fn map(value: Value) -> Map<String, Value> {
    value.as_object().cloned().unwrap_or_default()
}

async fn tool_named(mcp: &Mcp, name: &str) -> SharedTool {
    mcp.tools().await.unwrap().into_iter().find(|t| t.name() == name).unwrap()
}

async fn run(tool: &SharedTool, arguments: Value) -> Result<ToolResult, ToolError> {
    tool.execute(map(arguments), &ToolCall::new("call_1", tool.name(), Map::new())).await
}

// ---- mcp_spec: tools -------------------------------------------------------------------------

#[tokio::test]
async fn lists_the_server_tools_across_pages() {
    let mcp = files().build().unwrap();
    let tools = mcp.tools().await.unwrap();
    assert_eq!(tools.iter().map(|t| t.name()).collect::<Vec<_>>(), TOOL_NAMES);
    assert_eq!(tools[0].description(), "Echoes the text back");
    assert_eq!(
        tools[0].parameters_schema(),
        Some(json!({ "type": "object", "properties": { "text": { "type": "string" } }, "required": ["text"] }))
    );
    mcp.close().await;
}

#[tokio::test]
async fn reads_the_server_annotations() {
    let mcp = files().build().unwrap();
    let tools = mcp.mcp_tools().await.unwrap();
    let (echo, add, delete) = (&tools[0], &tools[1], &tools[8]);
    assert!(echo.is_read_only() && !echo.is_destructive());
    assert!(add.is_destructive() && add.is_open_world());
    assert!(delete.is_destructive() && !delete.is_open_world());
    mcp.close().await;
}

#[tokio::test]
async fn calls_a_tool_the_way_a_chat_does_and_reports_failures_to_the_model() {
    let mcp = files().build().unwrap();
    assert_eq!(run(&tool_named(&mcp, "echo").await, json!({ "text": "hello" })).await.unwrap().content, "hello");
    assert_eq!(run(&tool_named(&mcp, "fail").await, json!({})).await.unwrap().content, r#"{"error":"Something broke"}"#);
    mcp.close().await;
}

// ---- mcp_spec: shaping tools -----------------------------------------------------------------

async fn names(builder: McpBuilder) -> Vec<String> {
    let mcp = builder.build().unwrap();
    let names = mcp.tools().await.unwrap().iter().map(|t| t.name()).collect();
    mcp.close().await;
    names
}

#[tokio::test]
async fn keeps_only_or_hides_the_named_tools() {
    assert_eq!(names(files().only(&["echo", "add"])).await, ["echo", "add"]);
    assert_eq!(
        names(files().except(&["delete_everything", "fail"])).await,
        ["echo", "add", "picture", "slow", "wait", "deploy", "connect"]
    );
}

#[tokio::test]
async fn prefixes_tool_names_except_for_renamed_tools() {
    let mcp = files().prefix("files").tool("echo", ToolShape::new().as_name("repeat")).build().unwrap();
    let tools = mcp.mcp_tools().await.unwrap();
    assert_eq!(
        tools.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(),
        ["repeat", "files_add", "files_fail", "files_picture", "files_slow", "files_wait", "files_deploy", "files_connect", "files_delete_everything"]
    );
    assert_eq!(tools.last().unwrap().server_name, "delete_everything");
    mcp.close().await;
}

#[tokio::test]
async fn renames_and_redescribes_a_tool() {
    let mcp = files().tool("echo", ToolShape::new().as_name("repeat").description("Repeats the text")).build().unwrap();
    let repeat = mcp.mcp_tools().await.unwrap().remove(0);
    assert_eq!((repeat.name.as_str(), repeat.server_name.as_str()), ("repeat", "echo"));
    assert_eq!(repeat.description.as_deref(), Some("Repeats the text"));
    assert_eq!(repeat.call(json!({ "text": "hi" })).await.unwrap().content, "hi");
    assert_eq!(format!("{repeat:?}"), r#"McpTool { name: "repeat", from: "echo", read_only: true }"#);
    mcp.close().await;
}

#[tokio::test]
async fn fixes_arguments_the_model_no_longer_sees() {
    let shape = ToolShape::new().fixed_argument("b", json!(10)).fixed_argument_with("a", || json!(5));
    let mcp = files().tool("add", shape).build().unwrap();
    let add = tool_named(&mcp, "add").await;
    assert_eq!(add.parameters_schema(), Some(json!({ "type": "object", "properties": {}, "required": [] })));
    assert_eq!(run(&add, json!({})).await.unwrap().content, "15");
    mcp.close().await;
}

#[tokio::test]
async fn wraps_results() {
    let shape = ToolShape::new().wrap(|result, terms| {
        let terms: Vec<String> = terms.values().map(|v| v.to_string()).collect();
        format!("{} = {}", terms.join(" + "), result.structured.as_ref().unwrap()["sum"])
    });
    let mcp = files().tool("add", shape).build().unwrap();
    assert_eq!(run(&tool_named(&mcp, "add").await, json!({ "a": 2, "b": 3 })).await.unwrap().content, "2 + 3 = 5");
    mcp.close().await;
}

struct Doubler(Mcp);

#[async_trait]
impl Tool for Doubler {
    fn name(&self) -> String {
        "double".into()
    }
    fn description(&self) -> String {
        "Doubles a number".into()
    }
    async fn execute(&self, args: Map<String, Value>, _: &ToolCall) -> Result<ToolResult, ToolError> {
        let n = args["number"].clone();
        Ok(self.0.call("add", json!({ "a": n, "b": n })).await?.text.into())
    }
}

#[tokio::test]
async fn adds_tools_that_receive_the_mcp() {
    let mcp = files().add_tool(|mcp| Arc::new(Doubler(mcp))).build().unwrap();
    let tools = mcp.tools().await.unwrap();
    assert_eq!(run(tools.last().unwrap(), json!({ "number": 21 })).await.unwrap().content, "42");
    mcp.close().await;
}

#[tokio::test]
async fn requires_approval_for_named_tools_by_annotation_and_by_closure() {
    let mcp = files().requires_approval(&["echo"]).requires_approval_if(&[] as &[&str], |t| t.is_destructive()).build().unwrap();
    assert!(mcp.tools().await.unwrap().iter().all(|t| t.requires_approval()));
    mcp.close().await;

    let mcp = files().requires_approval_if(&[] as &[&str], |t| t.name.starts_with("delete")).build().unwrap();
    let approvals: Vec<String> = mcp.tools().await.unwrap().iter().filter(|t| t.requires_approval()).map(|t| t.name()).collect();
    assert_eq!(approvals, ["delete_everything"]);
    mcp.close().await;
}

#[tokio::test]
async fn refuses_declarations_for_tools_the_server_does_not_offer() {
    let mcp = files().tool("read_file", ToolShape::new().as_name("drive_read")).build().unwrap();
    let err = mcp.tools().await.err().unwrap();
    assert!(matches!(&err, Error::Configuration(m) if m.contains("declares read_file")), "{err}");
    mcp.close().await;
}

// ---- mcp_spec: call, resources, prompts, server info -----------------------------------------

#[tokio::test]
async fn call_returns_text_structured_content_and_attachments() {
    let mcp = files().build().unwrap();
    let result = mcp.call("add", json!({ "a": 2, "b": 3 })).await.unwrap();
    assert_eq!((result.text.as_str(), result.structured.clone()), ("5", Some(json!({ "sum": 5 }))));
    assert!(!result.is_error());

    let picture = mcp.call("picture", json!({})).await.unwrap();
    assert_eq!(picture.text, "Here it is\n\npixel.png: file:///pixel.png");
    let image = &picture.attachments[0];
    assert_eq!((image.mime_type.as_str(), image.filename.as_deref()), ("image/png", Some("image.png")));
    let content = picture.content();
    assert_eq!((content.content.as_str(), content.attachments.len()), (picture.text.as_str(), 1));
    mcp.close().await;
}

#[tokio::test]
async fn lists_reads_saves_and_attaches_resources() {
    let mcp = files().build().unwrap();
    let resources = mcp.resources().await.unwrap();
    let (readme, pixel) = (&resources[0], &resources[1]);
    assert_eq!(
        (readme.uri.as_str(), readme.name.as_str(), readme.mime_type.as_deref()),
        ("file:///project/README.md", "README.md", Some("text/markdown"))
    );
    assert_eq!(readme.content().await.unwrap().as_text(), Some("# Spec Project\n"));
    assert_eq!(pixel.to_blob().await.unwrap().len(), 70);

    let notes = mcp.resource("file:///project/notes.txt").await.unwrap();
    assert_eq!(notes.content().await.unwrap().as_text(), Some("Contents of file:///project/notes.txt"));

    let path = std::env::temp_dir().join(format!("rust_llm_mcp_{}.md", std::process::id()));
    readme.save(&path).await.unwrap();
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "# Spec Project\n");
    let _ = std::fs::remove_file(&path);

    let attachment = pixel.to_attachment().await.unwrap();
    assert_eq!((attachment.filename.as_deref(), attachment.mime_type.as_str()), (Some("pixel.png"), "image/png"));
    mcp.close().await;
}

#[tokio::test]
async fn fills_in_resource_templates() {
    let mcp = files().build().unwrap();
    let template = mcp.resource_templates().await.unwrap().remove(0);
    assert_eq!((template.uri.as_str(), template.name.as_deref()), ("file:///project/{+path}", Some("Project files")));
    let resource = mcp.resource_from_template(&template.uri, json!({ "path": "app/models/user.rb" })).await.unwrap();
    assert_eq!(resource.uri, "file:///project/app/models/user.rb");
    mcp.close().await;
}

#[tokio::test]
async fn lists_fills_in_and_suggests_prompts() {
    let mcp = files().build().unwrap();
    let prompt = mcp.prompts().await.unwrap().remove(0);
    assert_eq!((prompt.name.as_str(), prompt.description.as_deref(), prompt.messages.len()), ("code_review", Some("Reviews code"), 0));
    assert_eq!(prompt.arguments.iter().map(|a| (a.name.as_str(), a.required)).collect::<Vec<_>>(), [("code", true), ("language", false)]);

    let filled = mcp.prompt("code_review", &[("code", "puts 1"), ("language", "Ruby")]).await.unwrap();
    assert_eq!(filled.messages.iter().map(|m| m.role).collect::<Vec<_>>(), [Role::User, Role::Assistant, Role::User]);
    assert_eq!(filled.messages[0].content(), "Review this Ruby code:\nputs 1");

    assert_eq!(prompt.suggest(&[("language", "r")]).await.unwrap(), ["ruby", "rust"]);
    assert_eq!(prompt.suggest(&[("language", "py"), ("code", "x = 1")]).await.unwrap(), ["python (x = 1)"]);
    let template = mcp.resource_templates().await.unwrap().remove(0);
    assert_eq!(template.suggest(&[("path", "ru")]).await.unwrap(), ["ruby", "rust"]);
    mcp.close().await;
}

#[tokio::test]
async fn reads_what_the_server_says_about_itself() {
    let mcp = files().build().unwrap();
    assert_eq!(mcp.instructions().await.unwrap().as_deref(), Some("A server for specs."));
    assert_eq!(mcp.version().await.unwrap().as_deref(), Some("1.0.0"));
    mcp.close().await;
}

// ---- mcp_spec: progress, input requests, cancellation ----------------------------------------

#[tokio::test]
async fn runs_progress_callbacks_as_the_server_reports() {
    let reports = Arc::new(Mutex::new(Vec::new()));
    let (a, b) = (reports.clone(), reports.clone());
    let mcp = files()
        .after_progress(move |p| a.lock().unwrap().push(json!(p.fraction())))
        .after_progress(move |p| b.lock().unwrap().push(json!(p.message)))
        .build()
        .unwrap();
    assert_eq!(mcp.call("slow", json!({})).await.unwrap().text, "Finished");
    assert_eq!(*reports.lock().unwrap(), [json!(0.5), Value::Null, json!(1.0), Value::Null]);
    mcp.close().await;
}

#[tokio::test]
async fn answers_form_requests_with_a_callback_and_retries_the_call() {
    let mcp = files()
        .before_input_request(|r| {
            let choice = r.fields[0].choices.as_ref().unwrap()[0].clone();
            r.answer(map(json!({ "environment": choice })));
        })
        .build()
        .unwrap();
    assert_eq!(mcp.call("deploy", json!({})).await.unwrap().text, "Deployed to staging");
    mcp.close().await;
}

#[tokio::test]
async fn describes_the_fields_a_form_asks_for() {
    let seen: Arc<Mutex<Vec<InputRequest>>> = Arc::default();
    let captured = seen.clone();
    let mcp = files()
        .before_input_request(move |r| {
            captured.lock().unwrap().push(r.clone());
            r.decline();
        })
        .build()
        .unwrap();
    assert_eq!(mcp.call("deploy", json!({})).await.unwrap().text, "Deploy cancelled");
    let request = seen.lock().unwrap()[0].clone();
    assert_eq!((request.message.as_deref(), request.url.as_deref()), (Some("Which environment?"), None));
    assert!(request.is_form());
    let field = &request.fields[0];
    assert_eq!((field.name.as_str(), field.title.as_deref()), ("environment", Some("Environment")));
    assert_eq!(field.choices, Some(vec![json!("staging"), json!("production")]));
    assert!(field.required);
    mcp.close().await;
}

#[tokio::test]
async fn accepts_url_requests() {
    let mcp = files()
        .before_input_request(|r| {
            if r.is_url() {
                r.answer(Map::new());
            }
        })
        .build()
        .unwrap();
    assert_eq!(mcp.call("connect", json!({})).await.unwrap().text, "Connected");
    mcp.close().await;
}

#[tokio::test]
async fn raises_when_no_callback_answers_from_a_call_or_a_tool() {
    let mcp = files().build().unwrap();
    let Err(Error::McpInputRequired(e)) = mcp.call("connect", json!({})).await else { panic!("expected McpInputRequired") };
    assert!(e.message.ends_with("needs input from the user: Connect your account https://example.com/connect"), "{}", e.message);
    assert!(e.requests()[0].is_url());

    let connect = mcp.mcp_tools().await.unwrap().into_iter().find(|t| t.name == "connect").unwrap();
    assert!(matches!(connect.call(json!({})).await, Err(Error::McpInputRequired(_))));
    mcp.close().await;
}

#[tokio::test]
async fn declares_form_and_url_input_to_the_server() {
    let mcp = files().build().unwrap();
    let echoed = mcp.client().request("meta/echo", json!({}), &[], &mut |_| {}).await.unwrap();
    assert_eq!(echoed["meta"]["io.modelcontextprotocol/clientCapabilities"], json!({ "elicitation": { "form": {}, "url": {} } }));
    mcp.close().await;
}

#[tokio::test]
async fn stops_waiting_and_tells_the_server_when_cancelled() {
    let mcp = files().build().unwrap();
    mcp.tools().await.unwrap();
    let flag = Arc::new(AtomicBool::new(false));
    let setter = flag.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(200)).await;
        setter.store(true, Ordering::SeqCst);
    });
    let started = std::time::Instant::now();
    let result = rust_llm::progress::watch(flag, mcp.call("wait", json!({}))).await;
    assert!(matches!(result, Err(Error::Cancelled)), "{result:?}");
    assert!(started.elapsed() < Duration::from_secs(5));
    let cancelled = mcp.client().request("spec/cancelled", json!({}), &[], &mut |_| {}).await.unwrap();
    assert_eq!(cancelled["cancelled"].as_array().map(Vec::len), Some(1));
    mcp.close().await;
}

// ---- mcp_spec: naming, transports ------------------------------------------------------------

#[test]
fn names_itself_after_its_server() {
    assert_eq!(Mcp::url("https://mcp.linear.app/mcp").build().unwrap().name(), "linear");
    let docs = Mcp::url("https://learn.microsoft.com/api/mcp").bearer_token("secret").build().unwrap();
    assert_eq!(docs.name(), "learn_microsoft");
    assert_eq!(format!("{docs:?}"), r#"Mcp { name: "learn_microsoft", url: "https://learn.microsoft.com/api/mcp" }"#);
    assert_eq!(Mcp::command(["npx", "server"]).name("files").build().unwrap().name(), "files");
    assert_eq!(Mcp::command(["/usr/bin/npx", "server"]).build().unwrap().name(), "npx");
}

#[test]
fn refuses_insecure_urls() {
    assert!(matches!(Mcp::url("http://mcp.example.com/mcp").build(), Err(Error::Argument(m)) if m.contains("HTTPS")));
    assert!(matches!(Mcp::url("https://mcp.example.com@attacker.io/mcp").build(), Err(Error::Argument(m)) if m.contains("credentials")));
    assert!(Mcp::url("http://localhost:3000/mcp").build().is_ok());
}

#[derive(Default)]
struct Tunnel {
    label: String,
    sent: Mutex<Vec<String>>,
    closed: AtomicBool,
}

#[async_trait]
impl Transport for Tunnel {
    async fn request(
        &self,
        message: &Value,
        _: Option<&str>,
        _: Option<Duration>,
        _: &[(String, String)],
        _: &mut OnNotification<'_>,
    ) -> rust_llm::Result<Value> {
        let method = message["method"].as_str().unwrap_or("").to_string();
        self.sent.lock().unwrap().push(method.clone());
        let result = match method.as_str() {
            "server/discover" => json!({ "supportedVersions": ["2026-07-28"], "serverInfo": { "version": self.label } }),
            "tools/list" => json!({ "tools": [{ "name": "echo", "annotations": { "readOnlyHint": true } }] }),
            "tools/call" => json!({ "content": [{ "type": "text", "text": message.pointer("/params/arguments/text") }] }),
            _ => Value::Null,
        };
        Ok(json!({ "jsonrpc": "2.0", "id": message["id"], "result": result }))
    }
    async fn notify(&self, _: &Value, _: Option<&str>) -> rust_llm::Result<()> {
        Ok(())
    }
    async fn cancel(&self, _: &Value, _: Option<&str>) -> rust_llm::Result<()> {
        Ok(())
    }
    async fn close(&self) {
        self.closed.store(true, Ordering::SeqCst);
    }
}

#[tokio::test]
async fn speaks_through_the_transport_it_is_given_and_closes_it() {
    let tunnel = Arc::new(Tunnel { label: "laptop".into(), ..Default::default() });
    let mcp = Mcp::transport("tunnelled", tunnel.clone()).prefix("remote").build().unwrap();
    let tools = mcp.tools().await.unwrap();
    assert_eq!(tools.iter().map(|t| t.name()).collect::<Vec<_>>(), ["remote_echo"]);
    assert_eq!(run(&tools[0], json!({ "text": "hi" })).await.unwrap().content, "hi");
    assert_eq!(*tunnel.sent.lock().unwrap(), ["server/discover", "tools/list", "tools/call"]);
    assert_eq!(mcp.version().await.unwrap().as_deref(), Some("laptop"));
    mcp.close().await;
    assert!(tunnel.closed.load(Ordering::SeqCst));
}

// ---- client_spec -----------------------------------------------------------------------------

#[tokio::test]
async fn a_modern_server_is_discovered_and_gets_meta_with_every_request() {
    let client = stdio(&[], Duration::from_secs(30));
    assert_eq!(client.server().await.unwrap()["instructions"], "A server for specs.");
    assert!(client.is_modern());
    let meta = client.request("meta/echo", json!({}), &[], &mut |_| {}).await.unwrap()["meta"].clone();
    assert_eq!(meta["io.modelcontextprotocol/protocolVersion"], "2026-07-28");
    // RubyLLM sends { name: "ruby_llm", version: RubyLLM::VERSION }; the port names itself.
    assert_eq!(meta["io.modelcontextprotocol/clientInfo"], json!({ "name": "rust_llm", "version": "2.0.0" }));
    assert_eq!(meta["io.modelcontextprotocol/clientCapabilities"], json!({}));
    let names: Vec<Value> = client.list("tools/list", "tools").await.unwrap().iter().map(|t| t["name"].clone()).collect();
    assert_eq!(names, TOOL_NAMES.map(Value::from));
    client.close().await;
}

#[tokio::test]
async fn raises_json_rpc_errors_with_their_code() {
    let client = stdio(&[], Duration::from_secs(30));
    let Err(Error::Mcp(e)) = client.request("unknown/method", json!({}), &[], &mut |_| {}).await else { panic!() };
    assert_eq!((e.message.as_str(), e.code), ("Method not found", Some(-32_601)));
    client.close().await;
}

#[tokio::test]
async fn times_out_on_a_server_that_stops_in_the_middle_of_a_line() {
    let client = stdio(&[], Duration::from_secs(1));
    let result = client.request("spec/stall", json!({}), &[], &mut |_| {}).await;
    assert!(matches!(&result, Err(Error::Mcp(e)) if e.message.contains("did not answer in time")), "{result:?}");
    client.close().await;
}

#[tokio::test]
async fn falls_back_to_the_initialize_handshake_and_shakes_hands_again_after_closing() {
    let client = stdio(&[("MCP_ERA", "legacy")], Duration::from_secs(30));
    assert_eq!(client.server().await.unwrap()["serverInfo"], json!({ "name": "spec-server", "version": "0.9.0" }));
    assert_eq!(client.version().as_deref(), Some("2025-06-18"));
    assert!(!client.is_modern());
    assert_eq!(client.list("tools/list", "tools").await.unwrap().len(), 9);
    client.close().await;
    assert_eq!(client.list("tools/list", "tools").await.unwrap().len(), 9);
    client.close().await;
}

#[tokio::test]
async fn falls_back_when_discovery_answers_without_2026_07_28() {
    let client = stdio(&[("MCP_ERA", "discover_without_modern")], Duration::from_secs(30));
    assert_eq!(client.list("tools/list", "tools").await.unwrap().len(), 9);
    assert_eq!(client.version().as_deref(), Some("2025-06-18"));
    client.close().await;
}

// ---- http_spec (wiremock) --------------------------------------------------------------------

/// Answers a JSON-RPC POST, echoing its id.
struct Rpc {
    status: u16,
    reply: Value,
    headers: Vec<(&'static str, &'static str)>,
}

impl Respond for Rpc {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let id = serde_json::from_slice::<Value>(&request.body).ok().and_then(|b| b.get("id").cloned());
        let mut body = json!({ "jsonrpc": "2.0", "id": id });
        for (k, v) in self.reply.as_object().into_iter().flatten() {
            body[k] = v.clone();
        }
        let mut response = ResponseTemplate::new(self.status).set_body_raw(body.to_string(), "application/json");
        for (k, v) in &self.headers {
            response = response.insert_header(*k, *v);
        }
        response
    }
}

async fn stub(server: &MockServer, rpc_method: &str, status: u16, reply: Value) {
    Mock::given(method("POST"))
        .and(body_partial_json(json!({ "method": rpc_method })))
        .respond_with(Rpc { status, reply, headers: vec![] })
        .mount(server)
        .await;
}

async fn stub_raw(server: &MockServer, rpc_method: &str, response: ResponseTemplate) {
    Mock::given(method("POST")).and(body_partial_json(json!({ "method": rpc_method }))).respond_with(response).mount(server).await;
}

fn discover_result() -> Value {
    json!({ "result": { "resultType": "complete", "supportedVersions": ["2026-07-28"], "capabilities": { "tools": {} } } })
}

fn http_client(server: &MockServer, headers: Vec<(&'static str, &'static str)>) -> Client {
    let headers: Vec<(String, String)> = headers.into_iter().map(|(k, v)| (k.into(), v.into())).collect();
    let http = Http::new(&format!("{}/mcp", server.uri()), Arc::new(move || headers.clone()), Duration::from_secs(10)).unwrap();
    Client::new(Arc::new(http), json!({}))
}

async fn requests_for(server: &MockServer, rpc_method: &str) -> Vec<Request> {
    let all = server.received_requests().await.unwrap_or_default();
    all.into_iter()
        .filter(|r| serde_json::from_slice::<Value>(&r.body).ok().and_then(|b| b["method"].as_str().map(str::to_string)).as_deref() == Some(rpc_method))
        .collect()
}

fn header<'a>(request: &'a Request, name: &str) -> Option<&'a str> {
    request.headers.get(name).and_then(|v| v.to_str().ok())
}

fn b64(value: &str) -> String {
    format!("=?base64?{}?=", base64::engine::general_purpose::STANDARD.encode(value))
}

#[tokio::test]
async fn http_sends_the_mcp_headers_mirrored_params_and_encoded_names() {
    let server = MockServer::start().await;
    stub(&server, "server/discover", 200, discover_result()).await;
    stub(&server, "tools/call", 200, json!({ "result": { "content": [] } })).await;
    stub(&server, "resources/read", 200, json!({ "result": { "contents": [] } })).await;
    let client = http_client(&server, vec![("Authorization", "Bearer secret")]);

    client.request("tools/call", json!({ "name": "search", "arguments": {} }), &[], &mut |_| {}).await.unwrap();
    let call = requests_for(&server, "tools/call").await.remove(0);
    assert_eq!(header(&call, "mcp-protocol-version"), Some("2026-07-28"));
    assert_eq!(header(&call, "mcp-method"), Some("tools/call"));
    assert_eq!(header(&call, "mcp-name"), Some("search"));
    assert_eq!(header(&call, "authorization"), Some("Bearer secret"));

    let params = [("Region".to_string(), " us-west1".to_string())];
    client.request("tools/call", json!({ "name": "query", "arguments": {} }), &params, &mut |_| {}).await.unwrap();
    let call = requests_for(&server, "tools/call").await.remove(1);
    assert_eq!(header(&call, "mcp-param-region"), Some(b64(" us-west1").as_str()));

    client.request("resources/read", json!({ "uri": "file:///Überblick.md" }), &[], &mut |_| {}).await.unwrap();
    let read = requests_for(&server, "resources/read").await.remove(0);
    assert_eq!(header(&read, "mcp-name"), Some(b64("file:///Überblick.md").as_str()));
}

struct Sse;

impl Respond for Sse {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let id = serde_json::from_slice::<Value>(&request.body).ok().and_then(|b| b.get("id").cloned());
        let progress = json!({ "jsonrpc": "2.0", "method": "notifications/progress", "params": { "progress": 1, "total": 2 } });
        let result = json!({ "jsonrpc": "2.0", "id": id, "result": { "content": [{ "type": "text", "text": "done" }] } });
        let body = format!("event: message\ndata: {progress}\n\nevent: message\ndata: {result}\n\n");
        ResponseTemplate::new(200).set_body_raw(body, "text/event-stream")
    }
}

#[tokio::test]
async fn http_reads_event_streams_and_yields_their_notifications() {
    let server = MockServer::start().await;
    stub(&server, "server/discover", 200, discover_result()).await;
    Mock::given(method("POST")).and(body_partial_json(json!({ "method": "tools/call" }))).respond_with(Sse).mount(&server).await;
    let client = http_client(&server, vec![]);
    let mut notifications = Vec::new();
    let result = client
        .request("tools/call", json!({ "name": "slow", "arguments": {} }), &[], &mut |n| notifications.push(n["method"].clone()))
        .await
        .unwrap();
    assert_eq!(result, json!({ "content": [{ "type": "text", "text": "done" }] }));
    assert_eq!(notifications, [json!("notifications/progress")]);
}

#[tokio::test]
async fn http_falls_back_to_initialize_and_keeps_the_session() {
    let server = MockServer::start().await;
    stub_raw(&server, "server/discover", ResponseTemplate::new(400).set_body_string("Bad Request: missing session")).await;
    Mock::given(method("POST"))
        .and(body_partial_json(json!({ "method": "initialize" })))
        .respond_with(Rpc {
            status: 200,
            reply: json!({ "result": { "protocolVersion": "2025-06-18", "capabilities": {} } }),
            headers: vec![("Mcp-Session-Id", "session-1")],
        })
        .mount(&server)
        .await;
    stub_raw(&server, "notifications/initialized", ResponseTemplate::new(202)).await;
    stub(&server, "tools/list", 200, json!({ "result": { "tools": [] } })).await;
    let client = http_client(&server, vec![]);

    client.request("tools/list", json!({}), &[], &mut |_| {}).await.unwrap();
    assert_eq!(client.version().as_deref(), Some("2025-06-18"));
    let all = server.received_requests().await.unwrap();
    let with_session = all
        .iter()
        .filter(|r| header(r, "mcp-session-id") == Some("session-1") && header(r, "mcp-protocol-version") == Some("2025-06-18"))
        .count();
    assert_eq!(with_session, 2);
}

#[tokio::test]
async fn http_does_not_fall_back_when_a_modern_server_rejects_the_version() {
    let server = MockServer::start().await;
    let error = json!({ "error": { "code": -32_022, "message": "Unsupported protocol version", "data": { "supported": ["2027-01-01"] } } });
    stub(&server, "server/discover", 400, error).await;
    let client = http_client(&server, vec![]);
    let Err(Error::Mcp(e)) = client.server().await else { panic!("expected an MCP error") };
    assert_eq!(e.message, "Unsupported protocol version");
    assert!(requests_for(&server, "initialize").await.is_empty());
}

#[tokio::test]
async fn http_closes_a_modern_stream_but_tells_an_older_server_it_was_cancelled() {
    for legacy in [false, true] {
        let server = MockServer::start().await;
        stub_raw(&server, "tools/call", ResponseTemplate::new(200).set_body_raw("event: message\ndata: {}\n\n", "text/event-stream")).await;
        stub_raw(&server, "notifications/cancelled", ResponseTemplate::new(202)).await;
        if legacy {
            stub_raw(&server, "server/discover", ResponseTemplate::new(404)).await;
            stub(&server, "initialize", 200, json!({ "result": { "protocolVersion": "2025-06-18", "capabilities": {} } })).await;
            stub_raw(&server, "notifications/initialized", ResponseTemplate::new(202)).await;
        } else {
            stub(&server, "server/discover", 200, discover_result()).await;
        }
        let client = http_client(&server, vec![]);
        client.server().await.unwrap();
        let cancelled = Arc::new(AtomicBool::new(true));
        let result = rust_llm::progress::watch(cancelled, client.request("tools/call", json!({ "name": "slow" }), &[], &mut |_| {})).await;
        assert!(matches!(result, Err(Error::Cancelled)), "{result:?}");
        assert_eq!(requests_for(&server, "notifications/cancelled").await.len(), usize::from(legacy), "legacy: {legacy}");
    }
}

#[tokio::test]
async fn http_uses_a_result_sent_with_an_error_status_and_maps_401() {
    let server = MockServer::start().await;
    stub(&server, "server/discover", 200, discover_result()).await;
    stub(&server, "tools/list", 403, json!({ "result": { "tools": [{ "name": "search" }] } })).await;
    let client = http_client(&server, vec![]);
    assert_eq!(client.request("tools/list", json!({}), &[], &mut |_| {}).await.unwrap(), json!({ "tools": [{ "name": "search" }] }));

    let server = MockServer::start().await;
    stub_raw(&server, "server/discover", ResponseTemplate::new(401)).await;
    let client = http_client(&server, vec![]);
    let err = client.server().await.err().unwrap();
    assert!(matches!(&err, Error::Unauthorized(m, _) if m == "127.0.0.1 requires authorization"), "{err:?}");
}

#[tokio::test]
async fn http_resolves_headers_on_every_request_and_mcp_inputs_reach_them() {
    let server = MockServer::start().await;
    stub(&server, "server/discover", 200, discover_result()).await;
    stub(&server, "tools/list", 200, json!({ "result": { "tools": [] } })).await;
    let tokens = Arc::new(Mutex::new(VecDeque::from(["first", "second"])));
    let http = Http::new(
        &format!("{}/mcp", server.uri()),
        Arc::new(move || vec![("Authorization".to_string(), tokens.lock().unwrap().pop_front().unwrap_or("").to_string())]),
        Duration::from_secs(10),
    )
    .unwrap();
    let client = Client::new(Arc::new(http), json!({}));
    client.request("tools/list", json!({}), &[], &mut |_| {}).await.unwrap();
    assert_eq!(header(&requests_for(&server, "tools/list").await[0], "authorization"), Some("second"));

    // mcp_spec "inputs": blocks read the instance's inputs (here, captured user data).
    let server = MockServer::start().await;
    stub(&server, "server/discover", 200, discover_result()).await;
    let user = json!({ "token": "secret", "account": "acme" });
    let (for_token, for_account) = (user.clone(), user.clone());
    let mcp = Mcp::url(format!("{}/mcp", server.uri()))
        .bearer_token_with(move || for_token["token"].as_str().map(str::to_string))
        .header_with("X-Account", move || for_account["account"].as_str().map(str::to_string))
        .build()
        .unwrap();
    mcp.instructions().await.unwrap();
    let discover = requests_for(&server, "server/discover").await.remove(0);
    assert_eq!((header(&discover, "authorization"), header(&discover, "x-account")), (Some("Bearer secret"), Some("acme")));
}

// ---- chat_with_mcp_spec ----------------------------------------------------------------------

/// Serves canned Anthropic responses in order.
struct Replies(Mutex<VecDeque<Value>>);

impl Respond for Replies {
    fn respond(&self, _: &Request) -> ResponseTemplate {
        let body = self.0.lock().unwrap().pop_front().unwrap_or_else(answer);
        ResponseTemplate::new(200).set_body_json(body)
    }
}

fn tool_call(calls: &[(&str, &str, Value)]) -> Value {
    let content: Vec<Value> =
        calls.iter().map(|(id, name, input)| json!({ "type": "tool_use", "id": id, "name": name, "input": input })).collect();
    json!({ "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-haiku-4-5", "content": content,
            "stop_reason": "tool_use", "usage": { "input_tokens": 1, "output_tokens": 1 } })
}

fn answer() -> Value {
    json!({ "id": "msg_2", "type": "message", "role": "assistant", "model": "claude-haiku-4-5",
            "content": [{ "type": "text", "text": "Done" }], "stop_reason": "end_turn", "usage": { "input_tokens": 1, "output_tokens": 1 } })
}

async fn chat_replying(replies: Vec<Value>) -> (Chat, MockServer) {
    let server = MockServer::start().await;
    Mock::given(method("POST")).respond_with(Replies(Mutex::new(replies.into()))).mount(&server).await;
    let mut config = Config::default();
    config.set("anthropic_api_base", server.uri());
    config.set("anthropic_api_key", "test-key");
    config.max_retries = 0;
    let chat = Chat::with_config(Arc::new(config), Some("claude-haiku-4-5"), Some("anthropic"), false).unwrap();
    (chat, server)
}

fn tool_message(chat: &Chat) -> String {
    chat.messages().iter().find(|m| m.role == Role::Tool).map(|m| m.content().to_string()).unwrap_or_default()
}

#[tokio::test]
async fn chat_reads_servers_by_name_and_waits_to_contact_them() {
    let tunnel = Arc::new(Tunnel::default());
    let tunnelled = Mcp::transport("tunnelled", tunnel.clone()).build().unwrap();
    let (chat, _server) = chat_replying(vec![]).await;
    let chat = chat.with_mcp(tunnelled);
    assert_eq!(chat.mcp().get("tunnelled").map(|m| m.name()).as_deref(), Some("tunnelled"));
    assert_eq!(chat.mcp().len(), 1);
    assert!(tunnel.sent.lock().unwrap().is_empty(), "with_mcp must not contact the server");
    let names: Vec<String> = chat.all_tools().await.unwrap().iter().map(|t| t.name()).collect();
    assert_eq!(names, ["echo"]);
}

#[tokio::test]
async fn chat_lets_the_model_call_server_tools() {
    let files = files().build().unwrap();
    let (chat, server) = chat_replying(vec![tool_call(&[("call_1", "add", json!({ "a": 2, "b": 3 }))])]).await;
    let mut chat = chat.with_mcp(files.clone());
    chat.ask("What is 2 + 3?").await.unwrap();
    assert_eq!(tool_message(&chat), "5");
    let first: Value = serde_json::from_slice(&server.received_requests().await.unwrap()[0].body).unwrap();
    let tool_names: Vec<&str> = first["tools"].as_array().unwrap().iter().filter_map(|t| t["name"].as_str()).collect();
    assert_eq!(tool_names, TOOL_NAMES, "the model sees the server's tools");
    files.close().await;
}

#[tokio::test]
async fn chat_pauses_server_tools_that_need_approval() {
    let files = files().requires_approval(&["add"]).build().unwrap();
    let (chat, _server) = chat_replying(vec![tool_call(&[("call_1", "add", json!({ "a": 2, "b": 3 }))])]).await;
    let mut chat = chat.with_mcp(files.clone());
    chat.ask("What is 2 + 3?").await.unwrap();
    assert!(chat.is_awaiting_approval());
    let id = chat.pending_approvals()[0].id.clone();
    chat.approve(&id).complete().await.unwrap();
    assert_eq!(tool_message(&chat), "5");
    files.close().await;
}

#[tokio::test]
async fn chat_asks_with_a_server_prompt_and_attaches_resources() {
    let files = files().build().unwrap();
    let (mut chat, _server) = chat_replying(vec![answer(), answer()]).await;
    chat.ask_prompt(&files.prompt("code_review", &[("code", "puts 1")]).await.unwrap()).await.unwrap();
    assert_eq!(chat.messages().iter().map(|m| m.role).collect::<Vec<_>>(), [Role::User, Role::Assistant, Role::User, Role::Assistant]);
    assert_eq!(chat.messages()[2].content(), "Security.");

    let (mut chat, _server) = chat_replying(vec![answer()]).await;
    let pixel = files.resources().await.unwrap().pop().unwrap();
    chat.ask_with("Describe this", vec![pixel.to_attachment().await.unwrap()]).await.unwrap();
    let attachment = &chat.messages()[0].attachments[0];
    assert_eq!((attachment.filename.as_deref(), attachment.mime_type.as_str()), (Some("pixel.png"), "image/png"));
    files.close().await;
}

#[tokio::test]
async fn chat_passes_server_progress_to_after_tool_progress() {
    let server_reports = Arc::new(Mutex::new(Vec::new()));
    let recorder = server_reports.clone();
    let files = files().after_progress(move |p| recorder.lock().unwrap().push(p.value)).build().unwrap();
    let (chat, _server) = chat_replying(vec![tool_call(&[("call_1", "slow", json!({}))])]).await;
    let reports = Arc::new(Mutex::new(Vec::new()));
    let recorder = reports.clone();
    let mut chat = chat
        .with_mcp(files.clone())
        .after_tool_progress(move |call, progress| recorder.lock().unwrap().push((call.name.clone(), progress.fraction())));
    chat.ask("Take your time").await.unwrap();
    assert_eq!(*reports.lock().unwrap(), [("slow".to_string(), Some(0.5)), ("slow".to_string(), Some(1.0))]);
    assert_eq!(*server_reports.lock().unwrap(), [Some(1.0), Some(2.0)]);
    assert_eq!(tool_message(&chat), "Finished");
    files.close().await;
}

#[tokio::test]
async fn chat_stops_a_server_tool_when_cancelled() {
    let files = files().build().unwrap();
    let (chat, _server) = chat_replying(vec![tool_call(&[("call_1", "wait", json!({}))])]).await;
    let handle = chat.cancel_handle();
    let mut chat = chat.with_mcp(files.clone()).before_tool_call(move |_| {
        let handle = handle.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(200)).await;
            handle.cancel();
        });
    });
    let result = tokio::time::timeout(Duration::from_secs(10), chat.ask("Wait for it")).await.expect("cancelled promptly");
    assert!(matches!(result, Err(Error::Cancelled)), "{result:?}");
    files.close().await;
}

#[tokio::test]
async fn chat_pauses_a_tool_call_until_the_user_answers_or_declines() {
    for (decline, expected) in [(false, "Deployed to production"), (true, "Deploy cancelled")] {
        let files = files().build().unwrap();
        let (chat, _server) = chat_replying(vec![tool_call(&[("call_1", "deploy", json!({}))])]).await;
        let mut chat = chat.with_mcp(files.clone());
        chat.ask("Deploy").await.unwrap();
        assert!(chat.is_awaiting_input());
        let request = chat.pending_inputs().remove(0);
        assert_eq!(request.message.as_deref(), Some("Which environment?"));
        assert_eq!(request.tool_call.as_ref().map(|c| c.name.as_str()), Some("deploy"));
        if decline {
            chat.decline(&request).unwrap();
        } else {
            chat.answer(&request, map(json!({ "environment": "production" }))).unwrap();
        }
        chat.complete().await.unwrap();
        assert_eq!(tool_message(&chat), expected);
        assert!(!chat.is_awaiting_input());
        files.close().await;
    }
}

#[tokio::test]
async fn chat_does_not_pause_when_a_callback_answers() {
    let files = files().before_input_request(|r| r.answer(map(json!({ "environment": "staging" })))).build().unwrap();
    let (chat, _server) = chat_replying(vec![tool_call(&[("call_1", "deploy", json!({}))])]).await;
    let mut chat = chat.with_mcp(files.clone());
    chat.ask("Deploy").await.unwrap();
    assert_eq!(tool_message(&chat), "Deployed to staging");
    files.close().await;
}

#[tokio::test]
async fn chat_waits_on_approvals_and_inputs_together() {
    let files = files().requires_approval(&["add"]).build().unwrap();
    let reply = tool_call(&[("call_1", "deploy", json!({})), ("call_2", "add", json!({ "a": 1, "b": 1 }))]);
    let (chat, _server) = chat_replying(vec![reply]).await;
    let mut chat = chat.with_mcp(files.clone());
    chat.ask("Deploy and add").await.unwrap();
    assert!(chat.is_awaiting_input());
    assert!(chat.is_awaiting_approval());
    let err = chat.ask_later("Next").err().unwrap();
    assert!(matches!(&err, Error::PendingToolCalls(m) if m.contains("answering pending inputs")), "{err}");
    files.close().await;
}

#[tokio::test]
async fn chat_refuses_two_tools_with_the_same_name_and_disconnects_with_clear() {
    let echo = rust_llm::FnTool::new("echo", "Echoes", |_| async { Ok(ToolResult::from("x")) });
    let files = files().build().unwrap();
    let (chat, _server) = chat_replying(vec![]).await;
    let mut chat = chat.with_tool(echo).with_mcp(files.clone());
    let err = chat.all_tools().await.err().unwrap();
    assert!(matches!(&err, Error::Argument(m) if m.contains("Two tools are named echo")), "{err}");

    chat.clear_tools().clear_mcp();
    assert!(chat.mcp().is_empty());
    assert!(chat.all_tools().await.unwrap().is_empty());
    files.close().await;
}

struct WithServer(Mcp);

impl Agent for WithServer {
    fn mcp(&self) -> Vec<Mcp> {
        vec![self.0.clone()]
    }
}

#[tokio::test]
async fn connects_servers_declared_on_an_agent() {
    let files = files().build().unwrap();
    let (chat, _server) = chat_replying(vec![]).await;
    let chat = WithServer(files).apply(chat).unwrap();
    assert!(chat.mcp().get("files").is_some());
}
