//! `spec/ruby_llm/mcp/http_spec.rb:133` on its own: it reads `tracing` WARN events, whose
//! per-callsite interest cache races with other tests registering callsites concurrently, so it
//! runs alone in this binary.

use std::sync::{Arc, Mutex};

use rust_llm::mcp::Mcp;
use serde_json::{Value, json};
use wiremock::matchers::{body_partial_json, method};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

fn query_definition() -> Value {
    json!({ "name": "query", "inputSchema": { "type": "object", "properties": {
        "routing": { "type": "object", "properties": { "region": { "type": "string", "x-mcp-header": "Region" } } }
    } } })
}

async fn stub_result(server: &MockServer, rpc_method: &str, result: Value) {
    Mock::given(method("POST"))
        .and(body_partial_json(json!({ "method": rpc_method })))
        .respond_with(move |request: &Request| {
            let id =
                serde_json::from_slice::<Value>(&request.body).unwrap_or(Value::Null)["id"].clone();
            ResponseTemplate::new(200).set_body_raw(
                json!({ "jsonrpc": "2.0", "id": id, "result": result.clone() }).to_string(),
                "application/json",
            )
        })
        .mount(server)
        .await;
}

async fn query_server(definitions: Value) -> MockServer {
    let server = MockServer::start().await;
    stub_result(
        &server,
        "server/discover",
        json!({ "resultType": "complete", "supportedVersions": ["2026-07-28"], "capabilities": { "tools": {} } }),
    )
    .await;
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

fn url(server: &MockServer) -> String {
    format!("{}/mcp", server.uri())
}

/// Collects `tracing` WARN events on this thread, standing in for `RubyLLM.logger.warn`.
struct WarnCollector(Arc<Mutex<Vec<String>>>);

impl tracing::Subscriber for WarnCollector {
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
        struct Message<'a>(&'a mut String);
        impl tracing::field::Visit for Message<'_> {
            fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                if field.name() == "message" {
                    self.0.push_str(&format!("{value:?}"));
                }
            }
        }
        if *event.metadata().level() == tracing::Level::WARN {
            let mut text = String::new();
            event.record(&mut Message(&mut text));
            self.0.lock().unwrap().push(text);
        }
    }
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}

// spec: mcp/http_spec.rb:133 nested mirrored parameters through MCP > with an invalid nested declaration > excludes the invalid tool, logs its name, and leaves a valid tool callable
#[test]
fn excludes_an_invalid_nested_declaration_logs_its_name_and_keeps_a_valid_tool() {
    let mut invalid = query_definition();
    invalid["name"] = json!("invalid_query");
    invalid["inputSchema"]["properties"]["routing"]["properties"]["region"]["x-mcp-header"] =
        json!("Bad Name");
    let warnings = Arc::new(Mutex::new(Vec::new()));
    let _guard =
        tracing::dispatcher::set_default(&tracing::Dispatch::new(WarnCollector(warnings.clone())));
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let server = query_server(json!([query_definition(), invalid])).await;
        let mcp = Mcp::url(url(&server)).name("query").build().unwrap();

        let names: Vec<String> = mcp
            .mcp_tools()
            .await
            .unwrap()
            .iter()
            .map(|t| t.server_name.clone())
            .collect();
        assert_eq!(names, ["query"]);
        let logged = warnings.lock().unwrap().clone();
        assert!(
            logged
                .iter()
                .any(|w| w.contains("invalid_query") && w.contains("invalid x-mcp-header")),
            "{logged:?}"
        );
        let result = mcp
            .call("query", json!({ "routing": { "region": "us-west1" } }))
            .await
            .unwrap();
        assert_eq!(result.text, "ok");
        mcp.close().await;
    });
}
