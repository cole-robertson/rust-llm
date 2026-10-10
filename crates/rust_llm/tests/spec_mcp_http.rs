//! Ports of RubyLLM 2.1's `spec/ruby_llm/mcp/http_spec.rb` examples added since 2.0: task names,
//! nested mirrored parameters, answers to the server's own requests, sessions of older servers,
//! streams that end before the answer, and real connections. The webmock examples use wiremock;
//! the `over a real connection` ones use a loopback server, as Ruby does with `TCPServer` and
//! `spec/support/mcp_stream_server.rb` (ported in `mcp_stream_server/`).

mod mcp_stream_server;

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use base64::Engine;
use rust_llm::Error;
use rust_llm::mcp::{Client, Http, Mcp, ToolShape};
use serde_json::{Value, json};
use wiremock::matchers::{body_partial_json, method};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

fn discover_result() -> Value {
    json!({ "resultType": "complete", "supportedVersions": ["2026-07-28"], "capabilities": { "tools": {} } })
}

fn body_of(request: &Request) -> Value {
    serde_json::from_slice(&request.body).unwrap_or(Value::Null)
}

fn rpc(request: &Request, fields: Value) -> String {
    let mut message = json!({ "jsonrpc": "2.0", "id": body_of(request)["id"] });
    if let (Some(m), Value::Object(f)) = (message.as_object_mut(), fields) {
        m.extend(f);
    }
    message.to_string()
}

/// `stub_method(method, result:)`.
async fn stub_result(server: &MockServer, rpc_method: &str, result: Value) {
    Mock::given(method("POST"))
        .and(body_partial_json(json!({ "method": rpc_method })))
        .respond_with(move |request: &Request| {
            ResponseTemplate::new(200).set_body_raw(
                rpc(request, json!({ "result": result.clone() })),
                "application/json",
            )
        })
        .mount(server)
        .await;
}

async fn stub_with(
    server: &MockServer,
    rpc_method: &str,
    respond: impl Fn(&Request) -> ResponseTemplate + Send + Sync + 'static,
) {
    Mock::given(method("POST"))
        .and(body_partial_json(json!({ "method": rpc_method })))
        .respond_with(respond)
        .mount(server)
        .await;
}

fn url(server: &MockServer) -> String {
    format!("{}/mcp", server.uri())
}

fn http(url: &str, timeout: Duration) -> Http {
    Http::new(
        url,
        Arc::new(|_| vec![("Authorization".to_string(), "Bearer secret".to_string())]),
        timeout,
    )
    .unwrap()
}

fn client(server: &MockServer) -> Client {
    Client::new(
        Arc::new(http(&url(server), Duration::from_secs(10))),
        json!({}),
    )
}

async fn requests(server: &MockServer) -> Vec<Request> {
    server.received_requests().await.unwrap_or_default()
}

async fn posts_for(server: &MockServer, rpc_method: &str) -> Vec<Request> {
    requests(server)
        .await
        .into_iter()
        .filter(|r| body_of(r)["method"] == rpc_method)
        .collect()
}

async fn gets(server: &MockServer) -> Vec<Request> {
    requests(server)
        .await
        .into_iter()
        .filter(|r| r.method.as_str() == "GET")
        .collect()
}

fn header<'a>(request: &'a Request, name: &str) -> Option<&'a str> {
    request.headers.get(name).and_then(|v| v.to_str().ok())
}

fn b64(value: &str) -> String {
    format!(
        "=?base64?{}?=",
        base64::engine::general_purpose::STANDARD.encode(value)
    )
}

fn mcp_error(result: Result<Value, Error>) -> String {
    match result {
        Err(Error::Mcp(e)) => e.message,
        other => panic!("expected an MCP error, got {other:?}"),
    }
}

// spec: mcp/http_spec.rb:48 names the task in the Mcp-Name header of task requests
#[tokio::test]
async fn names_the_task_in_the_mcp_name_header_of_task_requests() {
    let server = MockServer::start().await;
    stub_result(&server, "server/discover", discover_result()).await;
    stub_result(
        &server,
        "tasks/get",
        json!({ "resultType": "complete", "taskId": "task-1", "status": "working" }),
    )
    .await;

    client(&server)
        .request("tasks/get", json!({ "taskId": "task-1" }), &[], &mut |_| {})
        .await
        .unwrap();

    let get = posts_for(&server, "tasks/get").await.remove(0);
    assert_eq!(header(&get, "mcp-method"), Some("tasks/get"));
    assert_eq!(header(&get, "mcp-name"), Some("task-1"));
}

// ---- nested mirrored parameters through MCP ----------------------------------------------------

fn query_definition() -> Value {
    json!({ "name": "query", "inputSchema": { "type": "object", "properties": {
        "routing": { "type": "object", "properties": { "region": { "type": "string", "x-mcp-header": "Region" } } }
    } } })
}

async fn query_server(definitions: Value) -> MockServer {
    let server = MockServer::start().await;
    stub_result(&server, "server/discover", discover_result()).await;
    stub_result(
        &server,
        "tools/list",
        json!({ "resultType": "complete", "tools": definitions, "ttlMs": 0, "cacheScope": "public" }),
    )
    .await;
    stub_result(
        &server,
        "tools/call",
        json!({ "content": [{ "type": "text", "text": "ok" }] }),
    )
    .await;
    server
}

// spec: mcp/http_spec.rb:95 nested mirrored parameters through MCP > sends the nested argument header without changing the request body
#[tokio::test]
async fn sends_the_nested_argument_header_without_changing_the_request_body() {
    let server = query_server(json!([query_definition()])).await;
    let mcp = Mcp::url(url(&server)).name("query").build().unwrap();

    let result = mcp
        .call("query", json!({ "routing": { "region": "us-west1" } }))
        .await
        .unwrap();
    assert_eq!(result.text, "ok");

    let calls = posts_for(&server, "tools/call").await;
    assert_eq!(calls.len(), 1);
    assert_eq!(header(&calls[0], "mcp-param-region"), Some("us-west1"));
    assert_eq!(
        body_of(&calls[0])["params"]["arguments"],
        json!({ "routing": { "region": "us-west1" } })
    );
    mcp.close().await;
}

// spec: mcp/http_spec.rb:106 nested mirrored parameters through MCP > encodes the nested value #{value.inspect} with the existing HTTP sentinel
#[tokio::test]
async fn encodes_nested_values_with_the_existing_http_sentinel() {
    for value in [
        "Hello, 世界",
        " padded ",
        "line1\nline2",
        "=?base64?literal?=",
    ] {
        let server = query_server(json!([query_definition()])).await;
        let mcp = Mcp::url(url(&server)).name("query").build().unwrap();

        mcp.call("query", json!({ "routing": { "region": value } }))
            .await
            .unwrap();

        let calls = posts_for(&server, "tools/call").await;
        assert_eq!(calls.len(), 1);
        assert_eq!(
            header(&calls[0], "mcp-param-region"),
            Some(b64(value).as_str()),
            "{value:?}"
        );
        mcp.close().await;
    }
}

// spec: mcp/http_spec.rb:114 nested mirrored parameters through MCP > mirrors fixed nested arguments when a shaped tool is called
#[tokio::test]
async fn mirrors_fixed_nested_arguments_when_a_shaped_tool_is_called() {
    let server = query_server(json!([query_definition()])).await;
    let shaped = Mcp::url(url(&server))
        .name("query")
        .tool(
            "query",
            ToolShape::new().fixed_argument("routing", json!({ "region": "us-west1" })),
        )
        .build()
        .unwrap();

    let tool = shaped.tools().await.unwrap().remove(0);
    let result = tool
        .execute(
            serde_json::Map::new(),
            &rust_llm::ToolCall::new("call_1", tool.name(), serde_json::Map::new()),
        )
        .await
        .unwrap();
    assert_eq!(result.content, "ok");

    let calls = posts_for(&server, "tools/call").await;
    assert_eq!(calls.len(), 1);
    assert_eq!(header(&calls[0], "mcp-param-region"), Some("us-west1"));
    shaped.close().await;
}

// spec: mcp/http_spec.rb:151 nested mirrored parameters through MCP > with an annotation reached through an array > excludes the tool from the list offered to the model
#[tokio::test]
async fn excludes_a_tool_whose_annotation_is_reached_through_an_array() {
    let routing = query_definition()["inputSchema"]["properties"]["routing"].clone();
    let server = query_server(
        json!([{ "name": "invalid_query", "inputSchema": { "type": "object", "properties": {
        "routes": { "type": "array", "items": routing }
    } } }]),
    )
    .await;
    let mcp = Mcp::url(url(&server)).name("query").build().unwrap();

    assert!(mcp.tools().await.unwrap().is_empty());
    mcp.close().await;
}

// spec: mcp/http_spec.rb:173 goes on with a call when the server refuses the answer to its own request
#[tokio::test]
async fn goes_on_with_a_call_when_the_server_refuses_the_answer_to_its_own_request() {
    let server = MockServer::start().await;
    stub_result(&server, "server/discover", discover_result()).await;
    Mock::given(method("POST"))
        .and(body_partial_json(json!({ "id": "roots-1" })))
        .respond_with(ResponseTemplate::new(401))
        .mount(&server)
        .await;
    stub_with(&server, "tools/call", |request| {
        let ask = json!({ "jsonrpc": "2.0", "id": "roots-1", "method": "roots/list" });
        let result = rpc(request, json!({ "result": { "content": [] } }));
        ResponseTemplate::new(200).set_body_raw(
            format!("data: {ask}\n\ndata: {result}\n\n"),
            "text/event-stream",
        )
    })
    .await;

    let result = client(&server)
        .request("tools/call", json!({ "name": "deploy" }), &[], &mut |_| {})
        .await
        .unwrap();
    assert_eq!(result, json!({ "content": [] }));
}

// ---- sessions of older servers -----------------------------------------------------------------

/// An older server that keeps a session per `initialize` and ends the ones listed in `ended`.
async fn session_server(ended: Arc<Mutex<Vec<String>>>) -> MockServer {
    let server = MockServer::start().await;
    let sessions = Arc::new(AtomicUsize::new(0));
    Mock::given(method("POST"))
        .respond_with(move |request: &Request| {
            let body = body_of(request);
            match body["method"].as_str() {
                Some("server/discover") => ResponseTemplate::new(404),
                Some("initialize") => {
                    let number = sessions.fetch_add(1, Ordering::SeqCst) + 1;
                    ResponseTemplate::new(200)
                        .insert_header("Mcp-Session-Id", format!("session-{number}").as_str())
                        .set_body_raw(
                            rpc(
                                request,
                                json!({ "result": { "protocolVersion": "2025-06-18", "capabilities": {} } }),
                            ),
                            "application/json",
                        )
                }
                _ if body.get("id").is_none() => ResponseTemplate::new(202),
                _ if header(request, "mcp-session-id")
                    .is_some_and(|s| ended.lock().unwrap().iter().any(|e| e == s)) =>
                {
                    ResponseTemplate::new(404)
                }
                _ => ResponseTemplate::new(200).set_body_raw(
                    rpc(request, json!({ "result": { "tools": [] } })),
                    "application/json",
                ),
            }
        })
        .mount(&server)
        .await;
    server
}

async fn initializations(server: &MockServer) -> usize {
    posts_for(server, "initialize")
        .await
        .iter()
        .filter(|r| header(r, "mcp-session-id").is_none())
        .count()
}

// spec: mcp/http_spec.rb:268 sessions of older servers > starts a new session when the server ends the old one
#[tokio::test]
async fn starts_a_new_session_when_the_server_ends_the_old_one() {
    let ended = Arc::new(Mutex::new(Vec::new()));
    let server = session_server(ended.clone()).await;
    let client = client(&server);
    client
        .request("tools/list", json!({}), &[], &mut |_| {})
        .await
        .unwrap();
    ended.lock().unwrap().push("session-1".to_string());

    assert_eq!(
        client
            .request("tools/list", json!({}), &[], &mut |_| {})
            .await
            .unwrap(),
        json!({ "tools": [] })
    );
    assert_eq!(initializations(&server).await, 2);
    let in_session_2 = requests(&server)
        .await
        .iter()
        .filter(|r| header(r, "mcp-session-id") == Some("session-2"))
        .count();
    assert_eq!(in_session_2, 2);
}

// spec: mcp/http_spec.rb:277 sessions of older servers > gives up when the new session ends too
#[tokio::test]
async fn gives_up_when_the_new_session_ends_too() {
    let ended = Arc::new(Mutex::new(Vec::new()));
    let server = session_server(ended.clone()).await;
    let client = client(&server);
    client
        .request("tools/list", json!({}), &[], &mut |_| {})
        .await
        .unwrap();
    ended
        .lock()
        .unwrap()
        .extend(["session-1".to_string(), "session-2".to_string()]);

    assert_eq!(
        mcp_error(
            client
                .request("tools/list", json!({}), &[], &mut |_| {})
                .await
        ),
        "127.0.0.1 ended the session"
    );
    assert_eq!(initializations(&server).await, 2);
}

// spec: mcp/http_spec.rb:285 sessions of older servers > gives callable headers the HTTP method of each request
#[tokio::test]
async fn gives_callable_headers_the_http_method_of_each_request() {
    let server = session_server(Arc::new(Mutex::new(Vec::new()))).await;
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(405))
        .mount(&server)
        .await;
    Mock::given(method("DELETE"))
        .respond_with(ResponseTemplate::new(204))
        .mount(&server)
        .await;
    let verbs = Arc::new(Mutex::new(Vec::<String>::new()));
    let seen = verbs.clone();
    let http = Http::new(
        &url(&server),
        Arc::new(move |verb| {
            let mut verbs = seen.lock().unwrap();
            if !verbs.iter().any(|v| v == verb) {
                verbs.push(verb.to_string());
            }
            Vec::new()
        }),
        Duration::from_secs(10),
    )
    .unwrap();
    let transport: Arc<dyn rust_llm::mcp::Transport> = Arc::new(http);
    let client = Client::new(transport.clone(), json!({}));

    client
        .request("tools/list", json!({}), &[], &mut |_| {})
        .await
        .unwrap();
    // `client.listen({})` on an older server ends on the session stream, which this one refuses.
    let listened = transport
        .listen(None, client.version().as_deref(), &mut |_| {})
        .await;
    let Err(Error::Mcp(e)) = listened else {
        panic!("expected an MCP error, got {listened:?}")
    };
    assert_eq!(e.message, "127.0.0.1 sends no events");
    client.close().await;

    assert_eq!(*verbs.lock().unwrap(), ["POST", "GET", "DELETE"]);
}

// spec: mcp/http_spec.rb:302 sessions of older servers > ends the session when it closes
#[tokio::test]
async fn ends_the_session_when_it_closes() {
    let server = session_server(Arc::new(Mutex::new(Vec::new()))).await;
    Mock::given(method("DELETE"))
        .respond_with(ResponseTemplate::new(405))
        .mount(&server)
        .await;
    let client = client(&server);
    client
        .request("tools/list", json!({}), &[], &mut |_| {})
        .await
        .unwrap();

    client.close().await;

    let deletes: Vec<Request> = requests(&server)
        .await
        .into_iter()
        .filter(|r| r.method.as_str() == "DELETE")
        .collect();
    assert_eq!(deletes.len(), 1);
    assert_eq!(header(&deletes[0], "mcp-session-id"), Some("session-1"));
    assert_eq!(
        header(&deletes[0], "mcp-protocol-version"),
        Some("2025-06-18")
    );
    assert_eq!(header(&deletes[0], "authorization"), Some("Bearer secret"));
    client
        .request("tools/list", json!({}), &[], &mut |_| {})
        .await
        .unwrap();
    assert_eq!(initializations(&server).await, 2);
}

// spec: mcp/http_spec.rb:809 has no session to end with a 2026-07-28 server
#[tokio::test]
async fn has_no_session_to_end_with_a_modern_server() {
    let server = MockServer::start().await;
    stub_result(&server, "server/discover", discover_result()).await;
    let client = client(&server);
    client.server().await.unwrap();

    client.close().await;

    assert!(
        requests(&server)
            .await
            .iter()
            .all(|r| r.method.as_str() != "DELETE")
    );
}

// ---- streams that end before the answer --------------------------------------------------------

fn event(data: &str, id: Option<&str>, retry_after: Option<u64>) -> String {
    let mut fields = Vec::new();
    if let Some(id) = id {
        fields.push(format!("id: {id}"));
    }
    if let Some(retry) = retry_after {
        fields.push(format!("retry: {retry}"));
    }
    fields.push(format!("data: {data}"));
    format!("{}\n\n", fields.join("\n"))
}

fn progress() -> String {
    json!({ "jsonrpc": "2.0", "method": "notifications/progress", "params": { "progress": 1 } })
        .to_string()
}

fn answer(request: &Request) -> String {
    rpc(request, json!({ "result": { "content": [] } }))
}

fn events(body: String) -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_raw(body, "text/event-stream")
}

async fn modern_server() -> MockServer {
    let server = MockServer::start().await;
    stub_result(&server, "server/discover", discover_result()).await;
    server
}

async fn slow_call(client: &Client) -> Result<Value, Error> {
    client
        .request("tools/call", json!({ "name": "slow" }), &[], &mut |_| {})
        .await
}

// spec: mcp/http_spec.rb:354 streams that end before the answer > with a 2026-07-28 server > sends the request again with a new ID
#[tokio::test]
async fn sends_the_request_again_with_a_new_id() {
    let server = modern_server().await;
    let ids = Arc::new(Mutex::new(Vec::<Value>::new()));
    let seen = ids.clone();
    stub_with(&server, "tools/call", move |request| {
        let mut ids = seen.lock().unwrap();
        ids.push(body_of(request)["id"].clone());
        if ids.len() == 1 {
            events(event(&progress(), None, None))
        } else {
            events(event(&answer(request), None, None))
        }
    })
    .await;

    assert_eq!(
        slow_call(&client(&server)).await.unwrap(),
        json!({ "content": [] })
    );
    let mut ids = ids.lock().unwrap().clone();
    ids.dedup();
    assert_eq!(ids.len(), 2);
    assert!(gets(&server).await.is_empty());
}

// spec: mcp/http_spec.rb:366 streams that end before the answer > with a 2026-07-28 server > gives up after a few attempts
#[tokio::test]
async fn gives_up_after_a_few_attempts() {
    let server = modern_server().await;
    stub_with(&server, "tools/call", |_| {
        events(event(&progress(), None, None))
    })
    .await;

    assert_eq!(
        mcp_error(slow_call(&client(&server)).await),
        "127.0.0.1 did not answer tools/call"
    );
    assert_eq!(posts_for(&server, "tools/call").await.len(), 4);
}

// spec: mcp/http_spec.rb:374 streams that end before the answer > with a 2026-07-28 server > skips event data that is not a JSON-RPC message
#[tokio::test]
async fn skips_event_data_that_is_not_a_json_rpc_message() {
    let server = modern_server().await;
    stub_with(&server, "tools/call", |request| {
        events(event("\"keep-alive\"", None, None) + &event(&answer(request), None, None))
    })
    .await;

    assert_eq!(
        slow_call(&client(&server)).await.unwrap(),
        json!({ "content": [] })
    );
}

// spec: mcp/http_spec.rb:380 streams that end before the answer > with a 2026-07-28 server > does not send the request again after a JSON body without the answer
#[tokio::test]
async fn does_not_send_the_request_again_after_a_json_body_without_the_answer() {
    let server = modern_server().await;
    stub_with(&server, "tools/call", |_| {
        ResponseTemplate::new(200).set_body_raw(
            json!({ "jsonrpc": "2.0", "id": "another", "result": {} }).to_string(),
            "application/json",
        )
    })
    .await;

    assert!(mcp_error(slow_call(&client(&server)).await).contains("did not answer"));
    assert_eq!(posts_for(&server, "tools/call").await.len(), 1);
}

async fn older_server() -> MockServer {
    let server = MockServer::start().await;
    stub_with(&server, "server/discover", |_| ResponseTemplate::new(404)).await;
    stub_with(&server, "initialize", |request| {
        ResponseTemplate::new(200)
            .insert_header("Mcp-Session-Id", "session-1")
            .set_body_raw(
                rpc(
                    request,
                    json!({ "result": { "protocolVersion": "2025-11-25", "capabilities": {} } }),
                ),
                "application/json",
            )
    })
    .await;
    stub_with(&server, "notifications/initialized", |_| {
        ResponseTemplate::new(202)
    })
    .await;
    server
}

/// Answers `tools/call` with `first`, and the resuming GET with the answer to that call.
async fn resumable(server: &MockServer, first: impl Fn() -> String + Send + Sync + 'static) {
    let call = Arc::new(Mutex::new(None::<Request>));
    let kept = call.clone();
    stub_with(server, "tools/call", move |request| {
        *kept.lock().unwrap() = Some(request.clone());
        events(first())
    })
    .await;
    Mock::given(method("GET"))
        .respond_with(move |_: &Request| {
            let call = call.lock().unwrap().clone();
            events(event(
                &call.as_ref().map(answer).unwrap_or_default(),
                Some("event-3"),
                None,
            ))
        })
        .mount(server)
        .await;
}

// spec: mcp/http_spec.rb:396 streams that end before the answer > with an older server > resumes the stream from its last event after the wait the server asks for
#[tokio::test]
async fn resumes_the_stream_from_its_last_event_after_the_wait_the_server_asks_for() {
    let server = older_server().await;
    resumable(&server, || {
        event("", Some("event-1"), Some(200)) + &event(&progress(), Some("event-2"), None)
    })
    .await;
    let mut notifications = Vec::new();
    let started = Instant::now();

    let result = client(&server)
        .request("tools/call", json!({ "name": "slow" }), &[], &mut |n| {
            notifications.push(n["method"].clone())
        })
        .await
        .unwrap();

    assert_eq!(result, json!({ "content": [] }));
    assert!(started.elapsed() >= Duration::from_millis(200));
    assert_eq!(notifications, [json!("notifications/progress")]);
    let gets = gets(&server).await;
    assert_eq!(gets.len(), 1);
    assert_eq!(header(&gets[0], "accept"), Some("text/event-stream"));
    assert_eq!(header(&gets[0], "last-event-id"), Some("event-2"));
    assert_eq!(header(&gets[0], "mcp-session-id"), Some("session-1"));
    assert_eq!(header(&gets[0], "mcp-protocol-version"), Some("2025-11-25"));
    assert_eq!(header(&gets[0], "authorization"), Some("Bearer secret"));
    assert_eq!(posts_for(&server, "tools/call").await.len(), 1);
}

// spec: mcp/http_spec.rb:419 streams that end before the answer > with an older server > resumes from an event ID and wait the server sends without data
#[tokio::test]
async fn resumes_from_an_event_id_and_wait_the_server_sends_without_data() {
    let server = older_server().await;
    resumable(&server, || {
        format!(
            "{}id: event-2\nretry: 200\n\n",
            event(&progress(), None, None)
        )
    })
    .await;
    let started = Instant::now();

    assert_eq!(
        slow_call(&client(&server)).await.unwrap(),
        json!({ "content": [] })
    );
    assert!(started.elapsed() >= Duration::from_millis(200));
    let gets = gets(&server).await;
    assert_eq!(gets.len(), 1);
    assert_eq!(header(&gets[0], "last-event-id"), Some("event-2"));
}

// spec: mcp/http_spec.rb:433 streams that end before the answer > with an older server > gives up on a stream without event IDs
#[tokio::test]
async fn gives_up_on_a_stream_without_event_ids() {
    let server = older_server().await;
    stub_with(&server, "tools/call", |_| {
        events(event(&progress(), None, None))
    })
    .await;

    assert!(mcp_error(slow_call(&client(&server)).await).contains("did not answer"));
    assert!(gets(&server).await.is_empty());
}

// spec: mcp/http_spec.rb:440 streams that end before the answer > with an older server > gives up after a few resumptions
#[tokio::test]
async fn gives_up_after_a_few_resumptions() {
    let server = older_server().await;
    stub_with(&server, "tools/call", |_| {
        events(event("", Some("event-1"), Some(0)))
    })
    .await;
    Mock::given(method("GET"))
        .respond_with(events(event("", Some("event-2"), Some(0))))
        .mount(&server)
        .await;

    assert!(mcp_error(slow_call(&client(&server)).await).contains("did not answer"));
    assert_eq!(gets(&server).await.len(), 3);
}

// spec: mcp/http_spec.rb:448 streams that end before the answer > with an older server > gives up rather than wait longer than the timeout
#[tokio::test]
async fn gives_up_rather_than_wait_longer_than_the_timeout() {
    let server = older_server().await;
    stub_with(&server, "tools/call", |_| {
        events(event("", Some("event-1"), Some(3_600_000)))
    })
    .await;

    assert!(mcp_error(slow_call(&client(&server)).await).contains("did not answer"));
    assert!(gets(&server).await.is_empty());
}

// spec: mcp/http_spec.rb:455 streams that end before the answer > with an older server > stops waiting when the chat is cancelled
#[tokio::test]
async fn stops_waiting_when_the_chat_is_cancelled() {
    let server = older_server().await;
    stub_with(&server, "tools/call", |_| {
        events(event("", Some("event-1"), Some(60_000)))
    })
    .await;
    stub_with(&server, "notifications/cancelled", |_| {
        ResponseTemplate::new(202)
    })
    .await;
    let client = Client::new(
        Arc::new(http(&url(&server), Duration::from_secs(300))),
        json!({}),
    );
    client.server().await.unwrap();
    let started = Instant::now();
    let flag = Arc::new(AtomicBool::new(false));
    let trip = flag.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(200)).await;
        trip.store(true, Ordering::SeqCst);
    });

    let result = rust_llm::progress::watch(flag, slow_call(&client)).await;

    assert!(matches!(result, Err(Error::Cancelled)), "{result:?}");
    assert!(started.elapsed() < Duration::from_secs(5));
    assert!(gets(&server).await.is_empty());
}

// ---- over a real connection --------------------------------------------------------------------

type Respond = Box<dyn FnOnce(std::net::TcpStream, Value) + Send>;

/// `serve(responses)`: answers one connection with each of `responses`, in order.
fn serve(responses: Vec<Respond>) -> Client {
    let server = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = server.local_addr().unwrap().port();
    std::thread::spawn(move || {
        for respond in responses {
            let Ok((socket, _)) = server.accept() else {
                return;
            };
            let mut reader = BufReader::new(socket.try_clone().unwrap());
            let mut length = 0;
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                    break;
                }
                if let Some(value) = line.to_lowercase().strip_prefix("content-length:") {
                    length = value.trim().parse().unwrap_or(0);
                }
            }
            let mut body = vec![0; length];
            reader.read_exact(&mut body).unwrap();
            respond(socket, serde_json::from_slice(&body).unwrap());
        }
    });
    Client::new(
        Arc::new(http(
            &format!("http://127.0.0.1:{port}/mcp"),
            Duration::from_secs(3),
        )),
        json!({}),
    )
}

fn discovered() -> Respond {
    Box::new(|mut socket, request| {
        let body = json!({ "jsonrpc": "2.0", "id": request["id"], "result": discover_result() })
            .to_string();
        let _ = socket.write_all(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .as_bytes(),
        );
    })
}

fn open_stream(socket: &mut std::net::TcpStream, events: &[String]) {
    let _ = socket.write_all(
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n",
    );
    for data in events {
        let _ = socket.write_all(format!("{:x}\r\n{data}\r\n", data.len()).as_bytes());
    }
}

fn stream_answer(request: &Value) -> String {
    let answer = json!({ "jsonrpc": "2.0", "id": request["id"], "result": { "content": [] } });
    format!("data: {answer}\n\n")
}

/// Answers on a stream it keeps open until the client hangs up.
fn answers_and_stays_open() -> Respond {
    Box::new(|mut socket, request| {
        open_stream(&mut socket, &[stream_answer(&request)]);
        let mut sink = Vec::new();
        let _ = socket.read_to_end(&mut sink);
    })
}

// spec: mcp/http_spec.rb:512 over a real connection > returns the answer while the server keeps the stream open
#[tokio::test]
async fn returns_the_answer_while_the_server_keeps_the_stream_open() {
    let client = serve(vec![discovered(), answers_and_stays_open()]);
    assert_eq!(slow_call(&client).await.unwrap(), json!({ "content": [] }));
}

// spec: mcp/http_spec.rb:518 over a real connection > sends the request again when the connection breaks midway
#[tokio::test]
async fn sends_the_request_again_when_the_connection_breaks_midway() {
    let broken: Respond = Box::new(|mut socket, _| {
        let progress =
            json!({ "jsonrpc": "2.0", "method": "notifications/progress", "params": {} });
        open_stream(&mut socket, &[format!("data: {progress}\n\n")]);
        let _ = socket.shutdown(std::net::Shutdown::Both);
    });
    let client = serve(vec![discovered(), broken, answers_and_stays_open()]);
    assert_eq!(slow_call(&client).await.unwrap(), json!({ "content": [] }));
}

// ---- listening over a real connection ----------------------------------------------------------

use mcp_stream_server::{Reply, Server, ServerRequest};
use rust_llm::mcp::Change;
use tokio::sync::mpsc;

/// What a test sees from `after_change`: the change, or a number a callback recorded.
#[derive(Debug, PartialEq)]
enum Seen {
    Tools,
    Resource(String),
    Size(usize),
    Other,
}

fn seen(change: &Change) -> Seen {
    match change {
        Change::Tools => Seen::Tools,
        Change::Resource(resource) => Seen::Resource(resource.uri.clone()),
        _ => Seen::Other,
    }
}

/// `Class.new(RubyLLM::MCP) { url url; after_change { |change| changes << change } }.new`.
fn listening_mcp(server: &Server) -> (Mcp, mpsc::UnboundedReceiver<Seen>) {
    let (sender, receiver) = mpsc::unbounded_channel();
    let mcp = Mcp::url(server.url.clone())
        .after_change(move |change| {
            let sender = sender.clone();
            async move {
                let _ = sender.send(seen(&change));
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

async fn eventually<T>(mut check: impl FnMut() -> Option<T>) -> T {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if let Some(value) = check() {
            return value;
        }
        assert!(Instant::now() < deadline, "timed out");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

fn modern_listen_server(listen: impl Fn(&ServerRequest, Reply) + Send + Sync + 'static) -> Server {
    Server::new(move |request, reply| {
        match request.rpc_method() {
        Some("server/discover") => reply.json(
            200,
            &[],
            json!({ "result": { "supportedVersions": ["2026-07-28"],
                "capabilities": { "tools": { "listChanged": true }, "resources": { "subscribe": true } } } }),
        ),
        Some("subscriptions/listen") => listen(request, reply),
        Some("tools/list") => reply.json(200, &[], json!({ "result": { "tools": [] } })),
        _ => reply.status(404),
    }
    })
}

fn acknowledging_server() -> Server {
    modern_listen_server(|request, reply| {
        let body = request.body.clone().unwrap_or_default();
        reply.stream(&[
            json!({ "jsonrpc": "2.0", "method": "notifications/subscriptions/acknowledged",
            "params": { "_meta": { "io.modelcontextprotocol/subscriptionId": body["id"] },
                        "notifications": body["params"]["notifications"] } }),
        ]);
    })
}

fn legacy_listen_server(events: bool) -> Server {
    Server::new(move |request, reply| {
        let verb = if request.verb == "GET" {
            Some("GET")
        } else {
            request.rpc_method()
        };
        match verb {
            Some("server/discover") => reply.status(404),
            Some("initialize") => reply.json(
                200,
                &[("Mcp-Session-Id", "session-1")],
                json!({ "result": { "protocolVersion": "2025-06-18",
                    "capabilities": { "tools": { "listChanged": true }, "resources": { "subscribe": true } } } }),
            ),
            Some("GET") if events => reply.stream(&[]),
            Some("GET") => reply.status(405),
            Some("resources/subscribe" | "ping") => reply.json(200, &[], json!({ "result": {} })),
            _ => reply.status(202),
        }
    })
}

fn subscription_id(server: &Server) -> Value {
    server
        .requests_for("subscriptions/listen")
        .last()
        .and_then(|r| r.body.clone())
        .map(|b| b["id"].clone())
        .unwrap_or(Value::Null)
}

fn changed(server: &Server, rpc_method: &str, mut params: Value) {
    params["_meta"] = json!({ "io.modelcontextprotocol/subscriptionId": subscription_id(server) });
    server.push(json!({ "jsonrpc": "2.0", "method": rpc_method, "params": params }));
}

// spec: mcp/http_spec.rb:618 listening over a real connection > with a 2026-07-28 server > subscribes on a stream that stays open, with the standard headers
#[tokio::test(flavor = "multi_thread")]
async fn subscribes_on_a_stream_that_stays_open_with_the_standard_headers() {
    let server = acknowledging_server();
    let (mcp, mut changes) = listening_mcp(&server);
    mcp.listen(&["file:///notes.md"], &[]).await.unwrap();

    let listen = server.requests_for("subscriptions/listen").remove(0);
    assert_eq!(listen.header("mcp-method"), Some("subscriptions/listen"));
    assert_eq!(listen.header("mcp-protocol-version"), Some("2026-07-28"));
    assert_eq!(
        listen.body.unwrap()["params"]["notifications"],
        json!({ "toolsListChanged": true, "resourceSubscriptions": ["file:///notes.md"] })
    );

    changed(
        &server,
        "notifications/resources/updated",
        json!({ "uri": "file:///notes.md" }),
    );
    assert_eq!(
        next_change(&mut changes).await,
        Seen::Resource("file:///notes.md".into())
    );
    mcp.close().await;
}

// spec: mcp/http_spec.rb:631 listening over a real connection > with a 2026-07-28 server > runs callbacks that call the server while the stream stays open
#[tokio::test(flavor = "multi_thread")]
async fn runs_callbacks_that_call_the_server_while_the_stream_stays_open() {
    let server = acknowledging_server();
    let (sender, mut changes) = mpsc::unbounded_channel();
    let slot: Arc<Mutex<Option<Mcp>>> = Arc::new(Mutex::new(None));
    let inner = slot.clone();
    let mcp = Mcp::url(server.url.clone())
        .after_change(move |change| {
            let (sender, mcp) = (sender.clone(), inner.lock().unwrap().clone());
            async move {
                let _ = sender.send(seen(&change));
                if let (Change::Tools, Some(mcp)) = (&change, mcp) {
                    let size = mcp.tools().await.map(|t| t.len()).unwrap_or(usize::MAX);
                    let _ = sender.send(Seen::Size(size));
                }
            }
        })
        .build()
        .unwrap();
    *slot.lock().unwrap() = Some(mcp.clone());
    mcp.listen(&[], &[]).await.unwrap();

    changed(&server, "notifications/tools/list_changed", json!({}));

    assert_eq!(
        [
            next_change(&mut changes).await,
            next_change(&mut changes).await
        ],
        [Seen::Tools, Seen::Size(0)]
    );
    assert_eq!(server.open_streams(), 1);
    slot.lock().unwrap().take();
    mcp.close().await;
}

// spec: mcp/http_spec.rb:642 listening over a real connection > with a 2026-07-28 server > ignores the comments a server sends to keep the stream alive
#[tokio::test(flavor = "multi_thread")]
async fn ignores_the_comments_a_server_sends_to_keep_the_stream_alive() {
    let server = acknowledging_server();
    let (mcp, mut changes) = listening_mcp(&server);
    mcp.listen(&[], &[]).await.unwrap();

    server.write_event(": keepalive");
    changed(&server, "notifications/tools/list_changed", json!({}));

    assert_eq!(next_change(&mut changes).await, Seen::Tools);
    mcp.close().await;
}

// spec: mcp/http_spec.rb:651 listening over a real connection > with a 2026-07-28 server > closes the stream when the MCP closes
#[tokio::test(flavor = "multi_thread")]
async fn closes_the_stream_when_the_mcp_closes() {
    let server = acknowledging_server();
    let (mcp, _changes) = listening_mcp(&server);
    mcp.listen(&[], &[]).await.unwrap();

    mcp.close().await;

    eventually(|| (server.open_streams() == 0).then_some(())).await;
}

// spec: mcp/http_spec.rb:659 listening over a real connection > with a 2026-07-28 server > subscribes again for the same changes when the connection drops
#[tokio::test(flavor = "multi_thread")]
async fn subscribes_again_for_the_same_changes_when_the_connection_drops() {
    let server = acknowledging_server();
    let (mcp, mut changes) = listening_mcp(&server);
    mcp.listen(&[], &[]).await.unwrap();

    server.drop_streams();

    let renewed = eventually(|| server.requests_for("subscriptions/listen").get(1).cloned()).await;
    assert_eq!(
        renewed.body.unwrap()["params"]["notifications"],
        json!({ "toolsListChanged": true })
    );
    eventually(|| (server.open_streams() == 1).then_some(())).await;
    changed(&server, "notifications/tools/list_changed", json!({}));
    assert_eq!(next_change(&mut changes).await, Seen::Tools);
    mcp.close().await;
}

// spec: mcp/http_spec.rb:671 listening over a real connection > with a 2026-07-28 server > subscribes again for the same changes when the server ends the subscription
#[tokio::test(flavor = "multi_thread")]
async fn subscribes_again_for_the_same_changes_when_the_server_ends_the_subscription() {
    let server = acknowledging_server();
    let (mcp, _changes) = listening_mcp(&server);
    mcp.listen(&[], &[]).await.unwrap();

    server.finish();

    let renewed = eventually(|| server.requests_for("subscriptions/listen").get(1).cloned()).await;
    assert_eq!(
        renewed.body.unwrap()["params"]["notifications"],
        json!({ "toolsListChanged": true })
    );
    mcp.close().await;
}

// spec: mcp/http_spec.rb:688 listening over a real connection > with a 2026-07-28 server that does not know subscriptions/listen > raises
#[tokio::test(flavor = "multi_thread")]
async fn listening_raises_when_the_server_does_not_know_subscriptions_listen() {
    let server = modern_listen_server(|_, reply| {
        reply.json(
            404,
            &[],
            json!({ "error": { "code": -32_601, "message": "Method not found" } }),
        )
    });
    let (mcp, _changes) = listening_mcp(&server);

    match mcp.listen(&[], &[]).await {
        Err(Error::Mcp(e)) => assert_eq!(e.message, "Method not found"),
        Err(other) => panic!("expected an MCP error, got {other:?}"),
        Ok(_) => panic!("expected an MCP error"),
    }
    mcp.close().await;
}

// spec: mcp/http_spec.rb:696 listening over a real connection > with a server that predates 2026-07-28 > subscribes to resources, listens on the session stream, and answers pings
#[tokio::test(flavor = "multi_thread")]
async fn subscribes_to_resources_listens_on_the_session_stream_and_answers_pings() {
    let server = legacy_listen_server(true);
    let (mcp, mut changes) = listening_mcp(&server);
    mcp.listen(&["file:///notes.md"], &[]).await.unwrap();

    let uris: Vec<Value> = server
        .requests_for("resources/subscribe")
        .iter()
        .map(|r| r.body.clone().unwrap_or_default()["params"]["uri"].clone())
        .collect();
    assert_eq!(uris, [json!("file:///notes.md")]);
    eventually(|| (server.open_streams() == 1).then_some(())).await;
    let get = server
        .requests()
        .into_iter()
        .find(|r| r.verb == "GET")
        .unwrap();
    assert_eq!(get.header("mcp-session-id"), Some("session-1"));
    assert_eq!(get.header("accept"), Some("text/event-stream"));

    server.push(json!({ "jsonrpc": "2.0", "id": "ping-1", "method": "ping" }));
    server.push(json!({ "jsonrpc": "2.0", "method": "notifications/resources/updated", "params": { "uri": "file:///notes.md" } }));

    assert_eq!(
        next_change(&mut changes).await,
        Seen::Resource("file:///notes.md".into())
    );
    let pong = eventually(|| {
        server
            .requests()
            .into_iter()
            .find(|r| r.body.as_ref().is_some_and(|b| b["id"] == "ping-1"))
    })
    .await;
    assert_eq!(pong.body.as_ref().unwrap()["result"], json!({}));
    assert_eq!(pong.header("mcp-session-id"), Some("session-1"));
    mcp.close().await;
}

// spec: mcp/http_spec.rb:732 listening over a real connection > with a server that predates 2026-07-28 and ends the session > starts a new session and subscribes again
#[tokio::test(flavor = "multi_thread")]
async fn starts_a_new_session_and_subscribes_again_when_the_listening_session_ends() {
    let sessions = Arc::new(AtomicUsize::new(0));
    let server = Server::new(move |request, reply| {
        let verb = if request.verb == "GET" {
            Some("GET")
        } else {
            request.rpc_method()
        };
        match verb {
            Some("server/discover") => reply.status(404),
            Some("initialize") => {
                let number = sessions.fetch_add(1, Ordering::SeqCst) + 1;
                let session = format!("session-{number}");
                reply.json(
                    200,
                    &[("Mcp-Session-Id", session.as_str())],
                    json!({ "result": { "protocolVersion": "2025-06-18", "capabilities": { "resources": { "subscribe": true } } } }),
                )
            }
            Some("GET") if request.header("mcp-session-id") == Some("session-1") => {
                reply.status(404)
            }
            Some("GET") => reply.stream(&[]),
            Some("resources/subscribe" | "ping") => reply.json(200, &[], json!({ "result": {} })),
            _ => reply.status(202),
        }
    });
    let (mcp, _changes) = listening_mcp(&server);
    mcp.listen(&["file:///notes.md"], &[]).await.unwrap();

    eventually(|| (server.open_streams() == 1).then_some(())).await;

    let sessions: Vec<Option<String>> = server
        .requests_for("resources/subscribe")
        .iter()
        .map(|r| r.header("mcp-session-id").map(str::to_string))
        .collect();
    assert_eq!(
        sessions,
        [Some("session-1".to_string()), Some("session-2".to_string())]
    );
    mcp.close().await;
}

// spec: mcp/http_spec.rb:769 listening over a real connection > with a server that predates 2026-07-28 and asks the client something mid-call > answers right away, pings with a result and everything else with method not found
#[tokio::test(flavor = "multi_thread")]
async fn answers_what_an_older_server_asks_mid_call() {
    let asks = [
        ("ping-1", "ping"),
        ("roots-1", "roots/list"),
        ("sample-1", "sampling/createMessage"),
        ("elicit-1", "elicitation/create"),
    ];
    let server = Arc::new(Server::new(move |request, reply| {
        match request.rpc_method() {
        Some("server/discover") => reply.status(404),
        Some("initialize") => reply.json(
            200,
            &[("Mcp-Session-Id", "session-1")],
            json!({ "result": { "protocolVersion": "2025-06-18", "capabilities": { "tools": {} } } }),
        ),
        Some("tools/list") => reply.json(
            200,
            &[],
            json!({ "result": { "tools": [{ "name": "deploy", "inputSchema": { "type": "object" } }] } }),
        ),
        Some("tools/call") => {
            let messages: Vec<Value> = asks
                .iter()
                .map(|(id, method)| json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": {} }))
                .collect();
            reply.stream(&messages)
        }
        _ => reply.status(202),
    }
    }));
    let mcp = Mcp::url(server.url.clone())
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap();
    let ids: Vec<&str> = asks.iter().map(|(id, _)| *id).collect();
    let answers = {
        let server = server.clone();
        move || -> Vec<ServerRequest> {
            server
                .requests()
                .into_iter()
                .filter(|r| {
                    r.verb == "POST"
                        && r.body
                            .as_ref()
                            .and_then(|b| b["id"].as_str())
                            .is_some_and(|id| ids.contains(&id))
                })
                .collect()
        }
    };
    let call = tokio::spawn({
        let mcp = mcp.clone();
        async move { mcp.call("deploy", json!({})).await }
    });

    let replies = eventually(|| {
        let replies = answers();
        (replies.len() == asks.len()).then_some(replies)
    })
    .await;
    let call_id = server.requests_for("tools/call")[0].body.clone().unwrap()["id"].clone();
    server.push(json!({ "jsonrpc": "2.0", "id": call_id, "result": { "content": [{ "type": "text", "text": "Deployed" }] } }));

    assert_eq!(call.await.unwrap().unwrap().text, "Deployed");
    let not_found = json!({ "code": -32_601, "message": "Method not found" });
    let answered: std::collections::HashMap<String, Value> = replies
        .iter()
        .map(|r| {
            let body = r.body.clone().unwrap();
            let answer = body.get("result").or(body.get("error")).cloned().unwrap();
            (body["id"].as_str().unwrap().to_string(), answer)
        })
        .collect();
    assert_eq!(answered["ping-1"], json!({}));
    assert_eq!(answered["roots-1"], not_found);
    assert_eq!(answered["sample-1"], not_found);
    assert_eq!(answered["elicit-1"], not_found);
    assert!(
        replies
            .iter()
            .all(|r| r.header("mcp-session-id") == Some("session-1"))
    );
    mcp.close().await;
}

// spec: mcp/http_spec.rb:790 listening over a real connection > with a server that predates 2026-07-28 and offers no event stream > stops listening instead of asking again
#[tokio::test(flavor = "multi_thread")]
async fn stops_listening_instead_of_asking_again() {
    let server = legacy_listen_server(false);
    let (mcp, _changes) = listening_mcp(&server);
    mcp.listen(&[], &[]).await.unwrap();

    tokio::time::sleep(Duration::from_millis(200)).await;

    assert_eq!(
        server.requests().iter().filter(|r| r.verb == "GET").count(),
        1
    );
    mcp.close().await;
}
