//! Ports of RubyLLM 2.1's new `transport/error_middleware_spec.rb` classification examples and
//! `protocol/streaming_spec.rb` stream-handling examples.
//!
//! `ErrorMiddleware.parse_error` takes the message a provider's `parse_error` read from the body;
//! here a local server answers with `{"error":{"message": ...}}`, which every provider reads the
//! same way, and `Connection#post` raises. The streaming examples feed `handle_stream` raw reads;
//! here a local server sends the same bytes, one read per chunk, into `Connection::stream` (the
//! port of `stream_events` with the on_data handler), with the base `Protocol::Streaming` status
//! mapping (every stream error is a 500), like the spec's bare object.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::sync::Arc;

use rust_llm::protocols::streaming_error_status;
use rust_llm::transport::{Connection, SseEvent};
use rust_llm::{Config, Error, ErrorKind, ProtocolName, Provider};
use serde_json::{Value, json};
use wiremock::{Mock, MockServer, ResponseTemplate, matchers};

// ---- transport/error_middleware_spec.rb --------------------------------------------------------

/// `ErrorMiddleware.parse_error(provider:, response:)` for `status` with the provider's `message`.
async fn parse_error(status: u16, message: &str) -> Error {
    let server = MockServer::start().await;
    let body = json!({ "error": { "message": message } }).to_string();
    Mock::given(matchers::method("POST"))
        .respond_with(ResponseTemplate::new(status).set_body_string(body))
        .mount(&server)
        .await;
    let mut config = Config::default();
    config.set("openai_api_key", "test");
    config.set("openai_api_base", server.uri());
    config.max_retries = 0;
    Connection::new(Provider::OpenAI, Arc::new(config))
        .unwrap()
        .post("chat/completions", &json!({}), &[], &mut |_| {})
        .await
        .unwrap_err()
}

// spec: transport/error_middleware_spec.rb:70 .parse_error with a bad request > spots a rate limit reported as a bad request
#[tokio::test]
async fn spots_a_rate_limit_reported_as_a_bad_request() {
    let message = "Your requests to gpt-6-luna for gpt-6-luna in germanywestcentral have exceeded token rate limit.";
    let error = parse_error(400, message).await;
    assert_eq!(error.kind(), ErrorKind::RateLimit);
    assert_eq!(error.to_string(), message);
}

// spec: transport/error_middleware_spec.rb:201 .parse_error > raises UnauthorizedError for a rejected key whatever status reports it
#[tokio::test]
async fn raises_unauthorized_for_a_rejected_key_whatever_status_reports_it() {
    for (message, status) in [
        ("API key not valid. Please pass a valid API key.", 400),
        (
            "Incorrect API key provided. You can obtain an API key from https://console.x.ai.",
            400,
        ),
        (
            "The security token included in the request is invalid.",
            403,
        ),
    ] {
        let error = parse_error(status, message).await;
        assert_eq!(error.kind(), ErrorKind::Unauthorized, "{message}");
        assert_eq!(error.to_string(), message);
    }
}

// spec: transport/error_middleware_spec.rb:216 .parse_error > keeps 429 errors about token quotas as RateLimitError
#[tokio::test]
async fn keeps_429_errors_about_token_quotas_as_rate_limits() {
    let messages = [
        "You exceeded your current quota, please check your plan and billing details. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_input_token_count, limit: 250000, model: gemini-2.5-flash\nPlease retry in 39.844676573s.",
        "Quota exceeded for aiplatform.googleapis.com/online_prediction_input_tokens_per_minute_per_base_model with base model: anthropic-claude-haiku-4-5. Please submit a quota increase request.",
        "Too many tokens, please wait before trying again.",
    ];
    for message in messages {
        let error = parse_error(429, message).await;
        assert_eq!(error.kind(), ErrorKind::RateLimit, "{message}");
        assert_eq!(error.to_string(), message);
    }
}

// spec: transport/error_middleware_spec.rb:237 .parse_error > maps a 413 that calls the request too large to ContextLengthExceededError
#[tokio::test]
async fn maps_a_413_that_calls_the_request_too_large_to_context_length_exceeded() {
    let message = "failed to process LLM request: request was too large";
    let error = parse_error(413, message).await;
    assert_eq!(error.kind(), ErrorKind::ContextLengthExceeded);
    assert_eq!(error.to_string(), message);
}

// spec: transport/error_middleware_spec.rb:246 .parse_error > keeps any other 413 a plain error
#[tokio::test]
async fn keeps_any_other_413_a_plain_error() {
    let message = "File exceeds the maximum upload size.";
    let error = parse_error(413, message).await;
    assert!(matches!(error, Error::Api(..)), "{error:?}");
    assert_eq!(error.to_string(), message);
}

// spec: transport/error_middleware_spec.rb:325 .parse_error > maps Anthropic's credit balance 400 to PaymentRequiredError
#[tokio::test]
async fn maps_anthropics_credit_balance_400_to_payment_required() {
    let message = "Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing to upgrade or purchase credits.";
    let error = parse_error(400, message).await;
    assert_eq!(error.kind(), ErrorKind::PaymentRequired);
    assert_eq!(error.to_string(), message);
}

// spec: transport/error_middleware_spec.rb:355 .parse_error > leaves a successful response body to the JSON middleware
#[tokio::test]
async fn leaves_a_successful_response_body_to_the_json_reader() {
    let server = MockServer::start().await;
    // A 2xx body that would read as an error message if it were parsed as one.
    Mock::given(matchers::method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_string(r#"{"choices":[]}"#))
        .mount(&server)
        .await;
    let mut config = Config::default();
    config.set("openai_api_base", server.uri());
    let response = Connection::new(Provider::OpenAI, Arc::new(config))
        .unwrap()
        .post("chat/completions", &json!({}), &[], &mut |_| {})
        .await
        .unwrap();
    assert_eq!(response.status, 200);
    assert_eq!(response.body, json!({ "choices": [] }));
}

// ---- protocol/streaming_spec.rb ----------------------------------------------------------------

/// A one-shot HTTP server that answers with `status` and sends `reads` as separate chunks of a
/// chunked body, pausing between them so each reaches the client as its own read.
fn serve_in_reads(status: u16, reads: &'static [&'static str]) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    std::thread::spawn(move || {
        let (socket, _) = listener.accept().unwrap();
        let mut writer = socket.try_clone().unwrap();
        let mut reader = BufReader::new(socket);
        let mut length = 0;
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                break;
            }
            if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                length = v.trim().parse().unwrap_or(0);
            }
        }
        let mut body = vec![0; length];
        let _ = reader.read_exact(&mut body);
        let head = format!(
            "HTTP/1.1 {status} X\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n"
        );
        writer.write_all(head.as_bytes()).unwrap();
        for read in reads {
            let _ = writer.write_all(format!("{:x}\r\n{read}\r\n", read.len()).as_bytes());
            let _ = writer.flush();
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        let _ = writer.write_all(b"0\r\n\r\n");
    });
    base
}

/// Streams from `base` and returns what reached the block, plus the outcome.
async fn handle_stream(base: &str) -> (Vec<Value>, rust_llm::Result<()>) {
    let mut config = Config::default();
    config.set("gemini_api_base", base);
    config.set("gemini_api_key", "test");
    config.max_retries = 0;
    let mut yielded = Vec::new();
    let result = Connection::new(Provider::Gemini, Arc::new(config))
        .unwrap()
        .stream(
            "stream",
            &json!({}),
            &[],
            &mut |_| {},
            &mut |_e: SseEvent, data: Value| {
                yielded.push(data);
                Ok(())
            },
            streaming_error_status(ProtocolName::Interactions),
        )
        .await
        .map(|_| ());
    (yielded, result)
}

async fn reads(reads: &'static [&'static str]) -> (Vec<Value>, rust_llm::Result<()>) {
    handle_stream(&serve_in_reads(200, reads)).await
}

fn assert_server_error(result: rust_llm::Result<()>, message: &str) {
    let error = result.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Server, "{error:?}");
    assert!(error.to_string().contains(message), "{error}");
}

// spec: protocol/streaming_spec.rb:55 keeps the error body and the request when the adapter has set no status yet
// Ruby merges the status before the body so Faraday's env keeps both the error body and the
// request. The port's error carries its own response (status and body) and the chat attaches the
// request it answered (`Error#request_shape`), so a stream that fails with a 400 keeps all three.
#[tokio::test]
async fn keeps_the_error_body_and_the_request_when_a_stream_fails() {
    let server = MockServer::start().await;
    let error_body = json!({ "error": { "message": "Rate limit exceeded" } }).to_string();
    Mock::given(matchers::method("POST"))
        .respond_with(
            ResponseTemplate::new(400).set_body_raw(error_body.clone(), "application/json"),
        )
        .mount(&server)
        .await;
    let mut config = Config::default();
    config.set("openai_api_key", "test");
    config.set("openai_api_base", server.uri());
    config.max_retries = 0;
    let error = rust_llm::Context::new(config)
        .chat(Some("gpt-4.1-nano"), Some("openai"))
        .unwrap()
        .with_protocol(ProtocolName::ChatCompletions)
        .ask_stream("Hello", |_| {})
        .await
        .unwrap_err();
    let response = error.response().unwrap();
    assert_eq!(response.status, 400);
    assert_eq!(response.body, error_body);
    assert!(response.request.is_some(), "the request the error answered");
}

// spec: protocol/streaming_spec.rb:188 delivers an event when a read starts inside its JSON
#[tokio::test]
async fn delivers_an_event_when_a_read_starts_inside_its_json() {
    let (yielded, result) =
        reads(&["data: {\"x\":\"ok\",\"meta\":", "{\"error\":null}}\n\n"]).await;
    result.unwrap();
    assert_eq!(yielded, [json!({ "x": "ok", "meta": { "error": null } })]);
}

// spec: protocol/streaming_spec.rb:198 delivers an event when its data arrives in a read of its own
#[tokio::test]
async fn delivers_an_event_when_its_data_arrives_in_a_read_of_its_own() {
    let (yielded, result) =
        reads(&["data: ", "{\"x\":\"ok\",\"response\":{\"error\":null}}\n\n"]).await;
    result.unwrap();
    assert_eq!(
        yielded,
        [json!({ "x": "ok", "response": { "error": null } })]
    );
}

// spec: protocol/streaming_spec.rb:208 raises a bare JSON error body
#[tokio::test]
async fn raises_a_bare_json_error_body() {
    let (yielded, result) = reads(&["{\"error\":{\"message\":\"Rate limit exceeded\"}}\n\n"]).await;
    assert_server_error(result, "Rate limit exceeded");
    assert!(yielded.is_empty());
}

// spec: protocol/streaming_spec.rb:216 raises a bare JSON error body that arrives split across reads
#[tokio::test]
async fn raises_a_bare_json_error_body_that_arrives_split_across_reads() {
    let (_, result) = reads(&["{\"error\":{\"message\":", "\"Rate limit exceeded\"}}"]).await;
    assert_server_error(result, "Rate limit exceeded");
}

// spec: protocol/streaming_spec.rb:225 raises a bare JSON error body that follows a blank read
#[tokio::test]
async fn raises_a_bare_json_error_body_that_follows_a_blank_read() {
    let (_, result) = reads(&["\n", "{\"error\":{\"message\":\"Overloaded\"}}"]).await;
    assert_server_error(result, "Overloaded");
}

// spec: protocol/streaming_spec.rb:235 ignores a bare JSON body without an error
#[tokio::test]
async fn ignores_a_bare_json_body_without_an_error() {
    let (yielded, result) = reads(&["{\"detail\":\"Service busy\"}"]).await;
    result.unwrap();
    assert!(yielded.is_empty());
}

// spec: protocol/streaming_spec.rb:272 ignores a bare JSON body whose error is null
#[tokio::test]
async fn ignores_a_bare_json_body_whose_error_is_null() {
    let (yielded, result) = reads(&["{\"status\":\"completed\",\"error\":null}"]).await;
    result.unwrap();
    assert!(yielded.is_empty());
}

// spec: protocol/streaming_spec.rb:280 ignores a bare JSON body that only mentions an error
#[tokio::test]
async fn ignores_a_bare_json_body_that_only_mentions_an_error() {
    let (yielded, result) = reads(&["{\"detail\":{\"error\":\"Service busy\"}}"]).await;
    result.unwrap();
    assert!(yielded.is_empty());
}

/// A failed streaming response whose HTML body runs past `MAX_JSON_BODY_BYTES` (1 MiB): the body
/// is not retained, and the status alone raises (`6f16112c`). The unit test of the limit itself,
/// at Ruby's stubbed 32 bytes, is in `transport/event_stream_parser.rs`.
#[tokio::test]
async fn a_failed_stream_with_an_oversized_body_raises_from_its_status() {
    let server = MockServer::start().await;
    let page = format!("<html>{}</html>", "x".repeat(1024 * 1024 + 1));
    Mock::given(matchers::method("POST"))
        .respond_with(ResponseTemplate::new(502).set_body_raw(page, "text/html"))
        .mount(&server)
        .await;
    let (_, result) = handle_stream(&server.uri()).await;
    let error = result.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::ServiceUnavailable);
    assert_eq!(error.response().unwrap().body, "");
}
