//! RubyLLM 2.1's `open_telemetry_rails_spec.rb`: a persisted chat (`acts_as_chat`, here
//! `ChatRecord`) traced inside a workflow, with concurrent tools that read the database. Ruby runs
//! it under thread and fiber `IsolatedExecutionState`; Rust has neither (the chat's concurrent tools
//! are futures polled in one task), so both isolations are this one test. Ruby stubs
//! `provider.complete`; a wiremock server answers in the Responses wire format instead.

#![cfg(feature = "opentelemetry")]

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use opentelemetry::trace::TraceContextExt;
use opentelemetry::{Context, global};
use opentelemetry_sdk::trace::{InMemorySpanExporter, Sampler, SdkTracerProvider};
use rust_llm::{Config, Tool, ToolCall, ToolError, ToolResult};
use rust_llm_loco::{ChatRecord, migrations};
use sea_orm::{Database, DatabaseConnection, EntityTrait, PaginatorTrait};
use sea_orm_migration::SchemaManager;
use serde_json::{Map, Value, json};
use wiremock::matchers::path;
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

/// `model_for(:openai, :temperature)`.
const MODEL: &str = "gpt-4.1-nano";

/// `TelemetryCountChats`: counts chats from inside a concurrent tool call.
struct TelemetryCountChats(DatabaseConnection);

#[async_trait]
impl Tool for TelemetryCountChats {
    fn name(&self) -> String {
        "telemetry_count_chats".into()
    }
    fn description(&self) -> String {
        String::new()
    }
    async fn execute(&self, _: Map<String, Value>, _: &ToolCall) -> Result<ToolResult, ToolError> {
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        let count = rust_llm_loco::entities::chats::Entity::find()
            .count(&self.0)
            .await
            .map_err(|e| ToolError::from(e.to_string()))?;
        Ok(count.to_string().into())
    }
}

// spec: open_telemetry_rails_spec.rb:11 traces persisted chats and fiber tools under #{isolation} execution isolation
#[tokio::test]
async fn traces_persisted_chats_and_concurrent_tools() {
    let exporter = InMemorySpanExporter::default();
    let provider = SdkTracerProvider::builder()
        .with_sampler(Sampler::AlwaysOn)
        .with_simple_exporter(exporter.clone())
        .build();
    global::set_tracer_provider(provider.clone());
    rust_llm::open_telemetry::enable().unwrap();

    let db = Database::connect("sqlite::memory:").await.unwrap();
    let manager = SchemaManager::new(&db);
    for m in migrations() {
        m.up(&manager).await.unwrap();
    }
    rust_llm::configure(|c| {
        c.set("openai_api_key", "test");
    });
    let server = MockServer::start().await;
    let calls: Vec<String> = (0..2)
        .map(|_| uuid::Uuid::new_v4().simple().to_string()[..16].to_string())
        .collect();
    let round = Arc::new(Mutex::new(0usize));
    Mock::given(path("/v1/responses"))
        .respond_with(move |_: &Request| {
            let mut n = round.lock().unwrap();
            *n += 1;
            let (output, usage) = if *n == 1 {
                let output: Vec<Value> = calls
                    .iter()
                    .map(|id| json!({ "type": "function_call", "id": format!("fc_{id}"), "call_id": id,
                                      "name": "telemetry_count_chats", "arguments": "{}", "status": "completed" }))
                    .collect();
                (output, json!({}))
            } else {
                (
                    vec![json!({ "type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                                 "content": [{ "type": "output_text", "text": "done", "annotations": [] }] })],
                    json!({ "input_tokens": 5, "output_tokens": 1 }),
                )
            };
            ResponseTemplate::new(200).set_body_json(json!({
                "id": format!("resp_{n}"), "object": "response", "status": "completed", "model": MODEL,
                "output": output, "usage": usage
            }))
        })
        .mount(&server)
        .await;
    let mut config = Config::default();
    config.set("openai_api_base", format!("{}/v1", server.uri()));
    config.set("openai_api_key", "test");
    config.max_retries = 0;

    // `workflow` bodies return `rust_llm::Result`; persistence errors fail the test instead.
    let reply = rust_llm::workflow("persisted chat", None, None, |_| async {
        let record = ChatRecord::create(&db, MODEL, None).await.unwrap();
        let mut chat = record
            .to_llm_with(&db, Arc::new(config))
            .await
            .unwrap()
            .with_tool(TelemetryCountChats(db.clone()))
            .with_tool_concurrency(true);
        let reply = record
            .ask(&db, &mut chat, "Count chats twice")
            .await
            .unwrap();
        let tool_rows = record
            .messages(&db)
            .await
            .unwrap()
            .into_iter()
            .filter(|m| m.role == "tool")
            .count();
        assert_eq!(tool_rows, 2);
        Ok(reply)
    })
    .await
    .unwrap();
    rust_llm::open_telemetry::disable();

    assert_eq!(reply.content(), "done");
    let spans = exporter.get_finished_spans().unwrap();
    let workflow = spans
        .iter()
        .find(|s| s.name == "invoke_workflow persisted chat")
        .unwrap();
    let children: Vec<_> = spans
        .iter()
        .filter(|s| s.span_context != workflow.span_context)
        .collect();
    assert_eq!(
        children.len(),
        4,
        "{:?}",
        spans.iter().map(|s| &s.name).collect::<Vec<_>>()
    );
    assert!(
        children
            .iter()
            .all(|s| s.parent_span_id == workflow.span_context.span_id())
    );
    let mut operations: Vec<String> = children
        .iter()
        .map(|s| {
            s.attributes
                .iter()
                .find(|kv| kv.key.as_str() == "gen_ai.operation.name")
                .map(|kv| kv.value.as_str().into_owned())
                .unwrap_or_default()
        })
        .collect();
    operations.sort();
    assert_eq!(operations, ["chat", "chat", "execute_tool", "execute_tool"]);
    assert!(!Context::current().span().span_context().is_valid());
    let _ = provider.shutdown();
}
