//! Regressions for divergences from RubyLLM found in review: retry semantics, the usage ledger,
//! and streamed tool-call assembly.

use std::sync::Arc;

use ruby_llm::{Chat, UsageStatus};
use serde_json::json;
use wiremock::{Mock, MockServer, ResponseTemplate, matchers};

fn anthropic_ok() -> serde_json::Value {
    json!({
        "model": "claude-haiku-4-5-20251001", "id": "msg_1", "type": "message", "role": "assistant",
        "content": [{ "type": "text", "text": "4" }], "stop_reason": "end_turn",
        "usage": { "input_tokens": 10, "output_tokens": 1 }
    })
}

fn config(server: &MockServer, provider: &str, retries: u32) -> Arc<ruby_llm::Config> {
    let mut c = ruby_llm::Config::default();
    let base = if provider == "openai" { format!("{}/v1", server.uri()) } else { server.uri() };
    c.set(format!("{provider}_api_base"), base);
    c.set(format!("{provider}_api_key"), "k");
    c.max_retries = retries;
    c.retry_interval = 0.001;
    Arc::new(c)
}

// faraday-retry: a Retry-After longer than retry_max_interval is not waited out, it fails at once.
#[tokio::test]
async fn a_retry_after_beyond_the_max_interval_is_not_retried() {
    let server = MockServer::start().await;
    Mock::given(matchers::method("POST"))
        .respond_with(ResponseTemplate::new(429).insert_header("retry-after", "3600").set_body_string(r#"{"error":{"message":"rate limit"}}"#))
        .mount(&server)
        .await;
    let mut chat = Chat::with_config(config(&server, "anthropic", 3), Some("claude-haiku-4-5"), Some("anthropic"), false).unwrap();
    let err = chat.ask("hi").await.unwrap_err();
    assert_eq!(err.kind(), ruby_llm::ErrorKind::RateLimit);
    assert_eq!(server.received_requests().await.unwrap().len(), 1, "no pointless retries");
}

// Tracker#failure_tokens: a refused (4xx) attempt is billed as zero, so a retried 429 followed by
// success still has a known total cost, and the retry is linked to the answer.
#[tokio::test]
async fn a_retried_429_keeps_the_cost_known_and_links_to_the_answer() {
    let server = MockServer::start().await;
    Mock::given(matchers::method("POST"))
        .respond_with(ResponseTemplate::new(429).set_body_string(r#"{"error":{"message":"rate limit"}}"#))
        .up_to_n_times(1)
        .with_priority(1)
        .mount(&server)
        .await;
    Mock::given(matchers::method("POST")).respond_with(ResponseTemplate::new(200).set_body_json(anthropic_ok())).with_priority(2).mount(&server).await;
    let mut chat = Chat::with_config(config(&server, "anthropic", 3), Some("claude-haiku-4-5"), Some("anthropic"), false).unwrap();
    let answer = chat.ask("hi").await.unwrap();

    let statuses: Vec<UsageStatus> = chat.usage_entries().iter().map(|e| e.status).collect();
    assert_eq!(statuses, [UsageStatus::Failed, UsageStatus::Succeeded]);
    assert_eq!(chat.usage_entries()[0].tokens.input, Some(0));
    assert!(chat.cost().total().is_some(), "a refused retry must not make the total unknown");
    assert_eq!(answer.usage_entries.len(), 2, "link_completion_usage links the retry to the answer");
}

// protocol/streaming.rb: an error event before any chunk is delivered is retried.
#[tokio::test]
async fn an_error_event_before_any_chunk_is_retried() {
    let server = MockServer::start().await;
    let overloaded = "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n";
    let ok = concat!(
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"model\":\"claude-haiku-4-5-20251001\",\"usage\":{\"input_tokens\":5,\"output_tokens\":1}}}\n\n",
        "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hello\"}}\n\n",
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
    );
    Mock::given(matchers::method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(overloaded, "text/event-stream"))
        .up_to_n_times(1)
        .with_priority(1)
        .mount(&server)
        .await;
    Mock::given(matchers::method("POST")).respond_with(ResponseTemplate::new(200).set_body_raw(ok, "text/event-stream")).with_priority(2).mount(&server).await;
    let mut chat = Chat::with_config(config(&server, "anthropic", 3), Some("claude-haiku-4-5"), Some("anthropic"), false).unwrap();
    let answer = chat.ask_stream("hi", |_| {}).await.unwrap();
    assert_eq!(answer.content(), "hello");
    assert_eq!(server.received_requests().await.unwrap().len(), 2);
}

// ...but once a chunk reached the caller, the failure is final (never replayed into the block).
#[tokio::test]
async fn an_error_after_a_delivered_chunk_is_not_retried() {
    let server = MockServer::start().await;
    let partial = concat!(
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"model\":\"claude-haiku-4-5-20251001\",\"usage\":{\"input_tokens\":50000,\"output_tokens\":1}}}\n\n",
        "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n",
    );
    Mock::given(matchers::method("POST")).respond_with(ResponseTemplate::new(200).set_body_raw(partial, "text/event-stream")).mount(&server).await;
    let mut chat = Chat::with_config(config(&server, "anthropic", 3), Some("claude-haiku-4-5"), Some("anthropic"), false).unwrap();
    let mut chunks = 0;
    let err = chat.ask_stream("hi", |_| chunks += 1).await.unwrap_err();
    assert_eq!(err.kind(), ruby_llm::ErrorKind::Overloaded);
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
    // Tracker#observe: tokens reported before the failure are billed, not dropped.
    let failed = chat.usage_entries().last().unwrap();
    assert_eq!((failed.status, failed.tokens.input), (UsageStatus::Failed, Some(50000)));
}

// stream_accumulator.rb: an empty-string id still starts a call (with a generated id); a fragment
// for an unknown index is ignored rather than spliced into another call.
#[tokio::test]
async fn streamed_tool_calls_with_empty_ids_and_stray_fragments() {
    let server = MockServer::start().await;
    let events = [
        json!({"model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"","type":"function","function":{"name":"lookup","arguments":""}}]}}]}),
        json!({"model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"q\":"}}]}}]}),
        json!({"model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":7,"function":{"arguments":"garbage"}}]}}]}),
        json!({"model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"rust\"}"}}]}}]}),
        json!({"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}),
    ];
    let body: String = events.iter().map(|e| format!("data: {e}\n\n")).collect::<String>() + "data: [DONE]\n\n";
    Mock::given(matchers::method("POST")).respond_with(ResponseTemplate::new(200).set_body_raw(body, "text/event-stream")).mount(&server).await;
    let mut c = ruby_llm::Config::default();
    c.set("ollama_api_base", format!("{}/v1", server.uri()));
    c.max_retries = 0;
    let mut chat = Chat::with_config(Arc::new(c), Some("qwen3"), Some("ollama"), true).unwrap();
    chat.ask_later("look it up").unwrap();
    let message = chat.step_stream(|_| {}).await.unwrap().unwrap();
    let calls = message.tool_calls.expect("tool call kept");
    let call = calls.values().next().unwrap();
    assert!(!call.id.is_empty(), "empty id replaced with a generated one");
    assert_eq!(call.arguments()["q"], "rust");
}
