//! A mock Anthropic Messages API that serves responses RubyLLM recorded (its VCR cassettes), so
//! the benchmark measures the client libraries rather than the network or the model.
//!
//!   mock <port>
//!
//! The URL path carries the knobs, so one server serves every case:
//!   POST /d/<delay_ms>/n/<chunks>/v1/messages
//! - waits `delay_ms` before answering (a stand-in for model latency);
//! - `"stream": true` gets an SSE stream with `chunks` text deltas;
//! - a request with tools gets a `tool_use` until it already carries 3 tool results, then text;
//! - anything else gets the recorded "2 + 2 = 4" message.

use std::time::Duration;

use axum::Router;
use axum::body::Bytes;
use axum::http::{HeaderValue, StatusCode, Uri, header};
use axum::response::{IntoResponse, Response};
use serde_json::Value;

const TEXT: &str = include_str!("../../fixtures/anthropic_text.json");
const TOOL_USE: &str = include_str!("../../fixtures/anthropic_tool_use.json");
const AFTER_TOOL: &str = include_str!("../../fixtures/anthropic_after_tool.json");
const TOOL_ROUNDS: usize = 3;

#[tokio::main]
async fn main() {
    let port: u16 = std::env::args().nth(1).and_then(|p| p.parse().ok()).unwrap_or(8765);
    let app = Router::new().fallback(handle);
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await.expect("bind");
    println!("mock listening on 127.0.0.1:{port}");
    axum::serve(listener, app).await.expect("serve");
}

fn knob(parts: &[&str], name: &str) -> u64 {
    parts.iter().position(|p| *p == name).and_then(|i| parts.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(0)
}

async fn handle(uri: Uri, body: Bytes) -> Response {
    let parts: Vec<&str> = uri.path().split('/').collect();
    let delay = knob(&parts, "d");
    let chunks = knob(&parts, "n").max(1) as usize;
    let Ok(request) = serde_json::from_slice::<Value>(&body) else {
        return (StatusCode::BAD_REQUEST, "bad json").into_response();
    };
    if delay > 0 {
        tokio::time::sleep(Duration::from_millis(delay)).await;
    }
    if request["stream"] == Value::Bool(true) {
        return with_type(sse(chunks), "text/event-stream; charset=utf-8");
    }
    let has_tools = request["tools"].as_array().is_some_and(|t| !t.is_empty());
    let results = tool_results(&request);
    let json = if has_tools && results < TOOL_ROUNDS {
        TOOL_USE.replace("toolu_01Go9uufLbuSYRKjVtPe9xmZ", &format!("toolu_bench_{results}"))
    } else if has_tools {
        AFTER_TOOL.to_string()
    } else {
        TEXT.to_string()
    };
    with_type(json, "application/json")
}

fn with_type(body: String, content_type: &'static str) -> Response {
    let mut response = body.into_response();
    response.headers_mut().insert(header::CONTENT_TYPE, HeaderValue::from_static(content_type));
    response
}

fn tool_results(request: &Value) -> usize {
    let Some(messages) = request["messages"].as_array() else { return 0 };
    messages
        .iter()
        .filter_map(|m| m["content"].as_array())
        .flatten()
        .filter(|block| block["type"] == "tool_result")
        .count()
}

/// The recorded "Count from 1 to 3" stream, with its one text delta repeated `chunks` times.
fn sse(chunks: usize) -> String {
    let mut out = String::with_capacity(200 * chunks + 1200);
    out.push_str("event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"model\":\"claude-haiku-4-5-20251001\",\"id\":\"msg_011Cf9mhiv32fvs7J6TDzAka\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"container\":null,\"stop_reason\":null,\"stop_sequence\":null,\"stop_details\":null,\"usage\":{\"input_tokens\":15,\"cache_creation_input_tokens\":0,\"cache_read_input_tokens\":0,\"cache_creation\":{\"ephemeral_5m_input_tokens\":0,\"ephemeral_1h_input_tokens\":0},\"output_tokens\":1,\"service_tier\":\"standard\",\"inference_geo\":\"not_available\"}}}\n\n");
    out.push_str("event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n");
    out.push_str("event: ping\ndata: {\"type\": \"ping\"}\n\n");
    for _ in 0..chunks {
        out.push_str("event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"1\\n2\\n3\"}}\n\n");
    }
    out.push_str("event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n");
    out.push_str("event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null,\"stop_details\":null,\"container\":null},\"usage\":{\"input_tokens\":15,\"cache_creation_input_tokens\":0,\"cache_read_input_tokens\":0,\"output_tokens\":9}}\n\n");
    out.push_str("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n");
    out
}
