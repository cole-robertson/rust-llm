//! Ports of RubyLLM 2.0's `protocol/streaming_spec.rb`. The spec feeds raw SSE text to the
//! `handle_stream` handler of a bare object extending `Protocol::Streaming`; here the same bytes
//! come from a local server into `Connection::stream`, the port of `stream_events` + the on_data
//! handler. The status mapping is `streaming_error_status(Interactions)`, which keeps the base
//! `Protocol::Streaming#parse_streaming_error` (every stream error is a 500), like the spec's
//! object. RubyLLM.logger spies have no Rust counterpart; the port logs through `tracing`.

use std::io::{Read, Write};
use std::sync::Arc;

use rust_llm::protocols::streaming_error_status;
use rust_llm::transport::{Connection, SseEvent};
use rust_llm::{Config, Error, ErrorKind, ProtocolName, Provider};
use serde_json::{Value, json};
use wiremock::{Mock, MockServer, ResponseTemplate, matchers};

fn connection(base: &str) -> Connection {
    let mut c = Config::default();
    c.set("gemini_api_base", base);
    c.set("gemini_api_key", "test");
    c.max_retries = 0;
    Connection::new(Provider::Gemini, Arc::new(c)).unwrap()
}

/// Streams from `base` and returns what reached the block, plus the outcome.
async fn run(base: &str) -> (Vec<Value>, rust_llm::Result<()>) {
    let mut yielded = Vec::new();
    let mut on_event = |_e: SseEvent, data: Value| {
        yielded.push(data);
        Ok(())
    };
    let result = connection(base)
        .stream(
            "stream",
            &json!({}),
            &[],
            &mut |_| {},
            &mut on_event,
            streaming_error_status(ProtocolName::Interactions),
        )
        .await
        .map(|_| ());
    (yielded, result)
}

/// A server answering with `status` and `body`.
async fn serve(status: u16, body: &str) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(matchers::method("POST"))
        .respond_with(
            ResponseTemplate::new(status).set_body_raw(body.to_string(), "text/event-stream"),
        )
        .mount(&server)
        .await;
    server
}

async fn handle(body: &str) -> (Vec<Value>, rust_llm::Result<()>) {
    let server = serve(200, body).await;
    run(&server.uri()).await
}

// spec: protocol/streaming_spec.rb:23 skips non-hash SSE payloads
#[tokio::test]
async fn skips_non_hash_sse_payloads() {
    let (yielded, result) = handle("data: true\n\n").await;
    result.unwrap();
    assert_eq!(yielded, Vec::<Value>::new());
}

// spec: protocol/streaming_spec.rb:40 prefers the failed HTTP response status over a generic parsed stream status
#[tokio::test]
async fn prefers_the_failed_http_response_status_over_a_generic_parsed_stream_status() {
    let parsed_error = json!({ "error": { "message": "Rate limit exceeded" } });
    // The stream mapping alone would call this a 500; the failed response said 429.
    assert_eq!(
        streaming_error_status(ProtocolName::Interactions)(&parsed_error.to_string()),
        Some(500)
    );
    let server = serve(429, &parsed_error.to_string()).await;
    let (_, result) = run(&server.uri()).await;
    let error = result.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::RateLimit);
    let response = error.response().unwrap();
    assert_eq!(response.status, 429);
    assert_eq!(
        serde_json::from_str::<Value>(&response.body).unwrap(),
        parsed_error
    );
}

// spec: protocol/streaming_spec.rb:67 raises the provider error when a failed response body parses to a bare JSON string
#[tokio::test]
async fn raises_the_provider_error_when_a_failed_response_body_parses_to_a_bare_json_string() {
    let server = serve(404, r#""model unavailable""#).await;
    let (_, result) = run(&server.uri()).await;
    let error = result.unwrap_err();
    assert!(matches!(error, Error::Api(..)), "{error:?}");
    assert_eq!(error.to_string(), "model unavailable");
    let response = error.response().unwrap();
    assert_eq!(response.status, 404);
    assert_eq!(
        serde_json::from_str::<Value>(&response.body).unwrap(),
        json!("model unavailable")
    );
}

/// A one-shot HTTP server that sends a chunked 200 event stream in `parts`, one network write
/// (and chunk) each, pausing between them so they reach the client as separate reads.
fn serve_in_reads(parts: &'static [&'static str]) -> String {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    std::thread::spawn(move || {
        let (mut socket, _) = listener.accept().unwrap();
        let mut request = Vec::new();
        let mut buf = [0u8; 4096];
        // Read the headers and the `{}` body.
        while !request.ends_with(b"\r\n\r\n{}") {
            let n = socket.read(&mut buf).unwrap();
            if n == 0 {
                return;
            }
            request.extend_from_slice(&buf[..n]);
        }
        socket
            .write_all(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n")
            .unwrap();
        for part in parts {
            socket
                .write_all(format!("{:x}\r\n{part}\r\n", part.len()).as_bytes())
                .unwrap();
            socket.flush().unwrap();
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        let _ = socket.write_all(b"0\r\n\r\n");
    });
    base
}

// spec: protocol/streaming_spec.rb:171 raises an error event that arrives split across reads
#[tokio::test]
async fn raises_an_error_event_that_arrives_split_across_reads() {
    let base = serve_in_reads(&[
        "event: error\ndata: {\"error\":{\"mes",
        "sage\":\"Rate limit exceeded\"}}\n\n",
    ]);
    let (yielded, result) = run(&base).await;
    let error = result.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Server);
    assert!(error.to_string().contains("Rate limit exceeded"), "{error}");
    assert!(yielded.is_empty());
}

// spec: protocol/streaming_spec.rb:188 ignores an error event that is not valid JSON
#[tokio::test]
async fn ignores_an_error_event_that_is_not_valid_json() {
    let (yielded, result) = handle("event: error\ndata: broken\n\n").await;
    result.unwrap();
    assert!(yielded.is_empty());
}

// spec: protocol/streaming_spec.rb:196 ignores a data chunk that is not valid JSON
#[tokio::test]
async fn ignores_a_data_chunk_that_is_not_valid_json() {
    let (yielded, result) = handle("data: {broken\n\n").await;
    result.unwrap();
    assert!(yielded.is_empty());
}

// spec: protocol/streaming_spec.rb:224 reports an unknown streaming error when the payload names none
#[tokio::test]
async fn reports_an_unknown_streaming_error_when_the_payload_names_none() {
    let (yielded, result) = handle("data: {\"error\":{}}\n\n").await;
    let error = result.unwrap_err();
    // `raise_stream_error` with status 500 and no provider message: a ServerError.
    assert_eq!(error.kind(), ErrorKind::Server);
    assert!(yielded.is_empty());
}
