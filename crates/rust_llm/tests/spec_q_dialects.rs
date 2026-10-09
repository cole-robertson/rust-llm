//! RubyLLM 2.1 parity (lane Q): the Responses and Chat Completions dialects.
//! `protocols/responses/{chat,streaming}_spec.rb`, `protocols/openrouter/responses_spec.rb`,
//! `protocols/perplexity/agent_spec.rb`, `providers/xai/responses_spec.rb`,
//! `providers/openrouter/chat_spec.rb`, `protocols/chat_completions/{chat,streaming}_spec.rb`, and
//! the streamed web search examples of `chat_provider_tools_spec.rb`.
//!
//! Ruby calls the protocol's private `render_payload`, `format_messages`, `format_thinking`,
//! `parse_completion_response`, `build_chunk`, and `parse_streaming_error`; these go through
//! `Chat#render`, `generate`, `ask`, and `ask_stream` against a mock server (which run the same
//! code, the stream accumulator included), or replay the recorded cassette.

mod spec_helpers;
mod support;

use std::io::{Read, Write};
use std::sync::Arc;

use rust_llm::message::{Operation, Thinking};
use rust_llm::{
    Attachment, Chat, Config, Error, Message, ProtocolName, ProviderTool, Resolution, Role,
    UsageEntry, UsageStatus,
};
use serde_json::{Map, Value, json};
use spec_helpers::{serve, serve_templates, sse};
use support::{Cassette, chat_for, config_for};
use wiremock::{MockServer, ResponseTemplate};

fn fixture(name: &str) -> String {
    format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
}

/// Every provider in this file pointed at `server`, no retries.
fn config(server: &MockServer) -> Arc<Config> {
    let mut c = Config::default();
    for (provider, path) in [
        ("openai", "/v1"),
        ("xai", "/v1"),
        ("openrouter", "/api/v1"),
        ("perplexity", ""),
    ] {
        c.set(
            format!("{provider}_api_base"),
            format!("{}{path}", server.uri()),
        );
        c.set(format!("{provider}_api_key"), "test");
    }
    c.max_retries = 0;
    Arc::new(c)
}

fn responses_chat(server: &MockServer, provider: &str, model: &str) -> Chat {
    Chat::with_config(config(server), Some(model), Some(provider), true)
        .unwrap()
        .with_protocol(ProtocolName::Responses)
}

/// `RubyLLM.chat(model: 'gpt-5-nano', provider: :openai, protocol: :responses)`.
fn openai(server: &MockServer) -> Chat {
    responses_chat(server, "openai", "gpt-5-nano")
}

fn render(chat: Chat, messages: Vec<Message>) -> Value {
    let mut chat = chat;
    for m in messages {
        chat.add_message(m);
    }
    chat.render().unwrap()
}

fn system(content: &str) -> Message {
    Message::new(Role::System, Some(content.to_string()))
}

fn cached_system(content: &str) -> Message {
    system(content).with_cache_until_here(None)
}

/// An SSE body carrying each event as a `data:` frame.
fn events(events: &[Value]) -> String {
    events.iter().map(|e| format!("data: {e}\n\n")).collect()
}

/// Streams `frames` through `ask_stream` on `chat`'s mock server, returning the result and chunks.
async fn stream_on(
    chat: impl FnOnce(&MockServer) -> Chat,
    frames: &[Value],
) -> (rust_llm::Result<Message>, Vec<Message>) {
    let server = serve_templates(vec![sse(events(frames))]).await;
    let mut chat = chat(&server);
    let mut chunks = Vec::new();
    let result = chat.ask_stream("hi", |c| chunks.push(c.clone())).await;
    (result, chunks)
}

/// Parses `body` through a non-streaming `generate` on `chat`'s mock server.
async fn parse_on(chat: impl FnOnce(&MockServer) -> Chat, body: Value) -> Message {
    let server = serve(vec![body]).await;
    let mut chat = chat(&server);
    chat.add_message(Message::user("hi"));
    chat.generate().await.unwrap()
}

fn counts(pairs: &[(&str, i64)]) -> Option<Map<String, Value>> {
    Some(
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), json!(v)))
            .collect(),
    )
}

async fn start(name: &str) -> Cassette {
    Cassette::start(name)
        .await
        .unwrap_or_else(|| panic!("missing cassette {name}"))
}

// ================================================================================================
// protocols/responses/chat_spec.rb #render_payload
// ================================================================================================

/// A tool result carrying `ruby.png` at original resolution, as the tool returned it.
async fn tool_image() -> Message {
    let mut image = Attachment::new(fixture("ruby.png")).with_resolution(Resolution::Original);
    image.content().await.unwrap();
    Message::tool_result("call_1", "Page image").with_attachments(vec![image])
}

// spec: protocols/responses/chat_spec.rb:15 #render_payload > preserves #{detail} detail for #{provider_class} tool-returned images
#[tokio::test]
async fn preserves_detail_for_tool_returned_images() {
    let server = MockServer::start().await;
    for (provider, model, detail) in [
        ("openai", "gpt-5-nano", "original"),
        ("xai", "grok-4.3", "high"),
    ] {
        let payload = render(
            responses_chat(&server, provider, model),
            vec![tool_image().await],
        );
        assert_eq!(
            payload["input"][1]["content"][1]["detail"],
            json!(detail),
            "{provider}"
        );
    }
}

// spec: protocols/responses/chat_spec.rb:209 #render_payload > keeps unmarked system messages after the cache-bounded one they follow
#[tokio::test]
async fn keeps_unmarked_system_messages_after_the_cache_bounded_one_they_follow() {
    let server = MockServer::start().await;
    let payload = render(
        openai(&server),
        vec![
            cached_system("Shared policy"),
            system("Today is Monday."),
            Message::user("hi"),
        ],
    );
    assert!(payload.get("instructions").is_none());
    let roles: Vec<&str> = payload["input"]
        .as_array()
        .unwrap()
        .iter()
        .map(|i| i["role"].as_str().unwrap_or(""))
        .collect();
    assert_eq!(roles, ["system", "system", "user"]);
    assert_eq!(payload["input"][0]["content"][0]["text"], "Shared policy");
    assert_eq!(payload["input"][1]["content"], "Today is Monday.");
}

// spec: protocols/responses/chat_spec.rb:224 #render_payload > sends system messages ahead of the conversation they follow
#[tokio::test]
async fn sends_system_messages_ahead_of_the_conversation_they_follow() {
    let server = MockServer::start().await;
    let payload = render(
        openai(&server),
        vec![
            Message::user("hi"),
            Message::assistant("Hello!"),
            Message::user("again"),
            cached_system("Shared policy"),
        ],
    );
    let roles: Vec<&str> = payload["input"]
        .as_array()
        .unwrap()
        .iter()
        .map(|i| i["role"].as_str().unwrap_or(""))
        .collect();
    assert_eq!(roles, ["system", "user", "assistant", "user"]);
}

// spec: protocols/responses/chat_spec.rb:237 #render_payload > keeps cache-bounded system messages ahead of a compaction
#[tokio::test]
async fn keeps_cache_bounded_system_messages_ahead_of_a_compaction() {
    let server = MockServer::start().await;
    let mut compaction = Message::new(Role::Assistant, None);
    compaction.raw_content = Some(json!({
        "object": "response.compaction", "output": [{ "type": "compaction", "id": "cmp_1" }]
    }));
    let payload = render(
        openai(&server),
        vec![
            cached_system("Shared policy"),
            Message::user("old question"),
            compaction,
            Message::user("new question"),
        ],
    );
    let kinds: Vec<&str> = payload["input"]
        .as_array()
        .unwrap()
        .iter()
        .map(|i| i["role"].as_str().or(i["type"].as_str()).unwrap_or(""))
        .collect();
    assert_eq!(kinds, ["system", "compaction", "user"]);
}

// ================================================================================================
// protocols/responses/chat_spec.rb #parse_completion_response
// ================================================================================================

// spec: protocols/responses/chat_spec.rb:404 #parse_completion_response > counts the web searches the response reports in tool_usage
#[tokio::test]
async fn counts_the_web_searches_the_response_reports_in_tool_usage() {
    let message = parse_on(
        openai,
        json!({
            "model": "gpt-5.2", "status": "completed", "output": [],
            "usage": { "input_tokens": 8610, "output_tokens": 89 },
            "tool_usage": {
                "image_gen": { "input_tokens": 0, "output_tokens": 0, "total_tokens": 0 },
                "web_search": { "num_requests": 2 }
            }
        }),
    )
    .await;
    assert_eq!(
        message.tokens.server_tool_use,
        counts(&[("web_search_requests", 2)])
    );
}

// ================================================================================================
// protocols/responses/streaming_spec.rb
// ================================================================================================

// spec: protocols/responses/streaming_spec.rb:12 keeps the raw output when a streamed function_call carries a tool-search namespace
#[tokio::test]
async fn keeps_the_raw_output_when_a_streamed_function_call_carries_a_tool_search_namespace() {
    let item = json!({ "type": "function_call", "call_id": "c1", "name": "weather_lookup", "arguments": "{}",
                       "namespace": "weather_lookup" });
    let (_, chunks) = stream_on(
        openai,
        &[json!({ "type": "response.completed", "response": { "output": [item.clone()], "status": "completed" } })],
    )
    .await;
    assert_eq!(chunks[0].raw_content, Some(json!([item])));
}

// spec: protocols/responses/streaming_spec.rb:178 counts the web searches the completed event reports in tool_usage
#[tokio::test]
async fn counts_the_web_searches_the_completed_event_reports_in_tool_usage() {
    let (result, chunks) = stream_on(
        openai,
        &[json!({ "type": "response.completed", "response": {
            "model": "gpt-5.2", "status": "completed",
            "usage": { "input_tokens": 8610, "output_tokens": 89 },
            "tool_usage": { "web_search": { "num_requests": 1 } }
        } })],
    )
    .await;
    result.unwrap();
    assert_eq!(
        chunks[0].tokens.server_tool_use,
        counts(&[("web_search_requests", 1)])
    );
}

/// A one-shot HTTP server that answers with a chunked 200 event stream in `parts`, one network
/// write each, pausing between them so they reach the client as separate reads.
fn serve_in_reads(parts: Vec<String>) -> String {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    std::thread::spawn(move || {
        let (mut socket, _) = listener.accept().unwrap();
        let mut request = Vec::new();
        let mut buf = [0u8; 4096];
        // Read the headers, then the body their Content-Length announces.
        loop {
            let n = socket.read(&mut buf).unwrap();
            if n == 0 {
                return;
            }
            request.extend_from_slice(&buf[..n]);
            let text = String::from_utf8_lossy(&request).to_lowercase();
            if let Some(end) = text.find("\r\n\r\n") {
                let length = text
                    .lines()
                    .find_map(|l| l.strip_prefix("content-length:"))
                    .and_then(|l| l.trim().parse::<usize>().ok())
                    .unwrap_or(0);
                if request.len() >= end + 4 + length {
                    break;
                }
            }
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

// spec: protocols/responses/streaming_spec.rb:192 reads usage from a completed event that arrives split across reads
#[tokio::test]
async fn reads_usage_from_a_completed_event_that_arrives_split_across_reads() {
    let completed = json!({ "type": "response.completed",
                            "response": { "model": "gpt-5-nano", "status": "completed", "error": null,
                                          "usage": { "input_tokens": 10, "output_tokens": 7 } } });
    let event = format!("event: response.completed\ndata: {completed}\n\n");
    let split = event.find("{\"model\"").unwrap();
    let base = serve_in_reads(vec![event[..split].to_string(), event[split..].to_string()]);
    let mut c = Config::default();
    c.set("openai_api_base", format!("{base}/v1"));
    c.set("openai_api_key", "test");
    c.max_retries = 0;
    let mut chat = Chat::with_config(Arc::new(c), Some("gpt-5-nano"), Some("openai"), true)
        .unwrap()
        .with_protocol(ProtocolName::Responses);
    let mut outputs = Vec::new();
    chat.ask_stream("hi", |c| outputs.push(c.tokens.output))
        .await
        .unwrap();
    assert_eq!(outputs, [Some(7)]);
}

/// The status `Responses::Streaming#parse_streaming_error` reads, and the error a stream raises.
async fn stream_error(frame: Value) -> (Option<u16>, Error) {
    let status =
        rust_llm::protocols::streaming_error_status(ProtocolName::Responses)(&frame.to_string());
    let (result, _) = stream_on(openai, &[frame]).await;
    (status, result.unwrap_err())
}

// spec: protocols/responses/streaming_spec.rb:257 #parse_streaming_error > classifies an error event that nests its code under an error object
#[tokio::test]
async fn classifies_an_error_event_that_nests_its_code_under_an_error_object() {
    let (status, error) = stream_error(json!({ "type": "error", "error": {
        "type": "too_many_requests", "code": "rate_limit_exceeded", "message": "Slow down" } }))
    .await;
    assert_eq!(status, Some(429));
    assert!(
        matches!(&error, Error::RateLimit(m, Some(r)) if m == "Slow down" && r.status == 429),
        "{error:?}"
    );
}

// spec: protocols/responses/streaming_spec.rb:266 #parse_streaming_error > classifies a nested error by its type when it carries no code
#[tokio::test]
async fn classifies_a_nested_error_by_its_type_when_it_carries_no_code() {
    let (status, error) = stream_error(json!({ "type": "error", "error": {
        "type": "too_many_requests", "message": "Slow down" } }))
    .await;
    assert_eq!(status, Some(429));
    assert!(matches!(&error, Error::RateLimit(..)), "{error:?}");
}

const AZURE_ERROR_EVENT: &str = "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"too_many_requests\",\"code\":\"rate_limit_exceeded\",\"headers\":{\"x-ms-fe-error\":\"true\"},\"message\":\"Your requests to gpt-6-luna for gpt-6-luna in germanywestcentral have exceeded token rate limit.\",\"param\":null},\"sequence_number\":1}\n\n";
const FAILED_RESPONSE_EVENT: &str = "event: response.failed\ndata: {\"type\":\"response.failed\",\"response\":{\"object\":\"response\",\"status\":\"failed\",\"error\":{\"code\":\"rate_limit_exceeded\",\"message\":\"Your requests to gpt-6-luna for gpt-6-luna in germanywestcentral have exceeded token rate limit.\"},\"model\":\"gpt-6-luna\",\"output\":[]},\"sequence_number\":2}\n\n";

/// The error a Responses stream carrying `body` raises. Azure is a provider this port leaves out;
/// its events go through the same Responses stream parser, so an OpenAI Responses chat reads them.
async fn stream_events(body: String) -> Error {
    let server = serve_templates(vec![sse(body)]).await;
    let mut chat = openai(&server);
    chat.ask_stream("Hello", |_| {}).await.unwrap_err()
}

// spec: protocols/responses/streaming_spec.rb:302 stream errors > raises a rate limit that Azure reports in an error event
#[tokio::test]
async fn raises_a_rate_limit_that_azure_reports_in_an_error_event() {
    let error = stream_events(format!("{AZURE_ERROR_EVENT}{FAILED_RESPONSE_EVENT}")).await;
    assert!(
        matches!(&error, Error::RateLimit(m, _) if m.contains("exceeded token rate limit")),
        "{error:?}"
    );
}

// spec: protocols/responses/streaming_spec.rb:307 stream errors > raises a rate limit that a failed response reports
#[tokio::test]
async fn raises_a_rate_limit_that_a_failed_response_reports() {
    let error = stream_events(FAILED_RESPONSE_EVENT.to_string()).await;
    assert!(
        matches!(&error, Error::RateLimit(m, _) if m.contains("exceeded token rate limit")),
        "{error:?}"
    );
}

// spec: protocols/responses/streaming_spec.rb:312 stream errors > raises a rate limit that OpenAI reports in a flat error event
#[tokio::test]
async fn raises_a_rate_limit_that_openai_reports_in_a_flat_error_event() {
    let error = stream_events(
        "event: error\ndata: {\"type\":\"error\",\"code\":\"rate_limit_exceeded\",\"message\":\"Rate limit reached for requests\",\"param\":null,\"sequence_number\":1}\n\n".into(),
    )
    .await;
    assert!(
        matches!(&error, Error::RateLimit(m, _) if m == "Rate limit reached for requests"),
        "{error:?}"
    );
}

// ================================================================================================
// protocols/openrouter/responses_spec.rb
// ================================================================================================

// spec: protocols/openrouter/responses_spec.rb:77 counts the web searches it runs
#[tokio::test]
async fn openrouter_responses_counts_the_web_searches_it_runs() {
    let cassette = start("protocols_openrouter_responses_counts_the_web_searches_it_runs").await;
    let mut chat = Chat::with_config(
        config_for(&cassette, "openrouter"),
        Some("openai/gpt-5.2"),
        Some("openrouter"),
        false,
    )
    .unwrap()
    .with_protocol(ProtocolName::Responses)
    .with_max_output_tokens(700)
    .with_provider_tools([ProviderTool::alias("web_search")]);
    let message = chat
        .ask("Search the web: what is the latest stable Ruby version? Cite your source.")
        .await
        .unwrap();
    assert_eq!(
        message.tokens().server_tool_use,
        counts(&[("web_search_requests", 2)])
    );
    assert!(message.cost(None).total().is_some_and(|t| t > 0.0));
    cassette.assert_all_matched().await;
}

// ================================================================================================
// protocols/perplexity/agent_spec.rb
// ================================================================================================

/// `RubyLLM.chat(model: model_for(:perplexity, :agent), provider: :perplexity)`.
fn agent(server: &MockServer) -> Chat {
    Chat::with_config(
        config(server),
        Some("openai/gpt-5-mini"),
        Some("perplexity"),
        false,
    )
    .unwrap()
}

/// The spec's `response_body(text)` with `usage.tool_calls_details` for one billed web search.
fn searched_agent_body(text: &str) -> Value {
    json!({
        "id": "resp_1", "object": "response", "status": "completed", "model": "openai/gpt-5-mini",
        "output": [
            { "type": "search_results", "queries": ["rails creator"], "results": [
                { "id": 1, "title": "Ruby on Rails", "url": "https://rubyonrails.org", "snippet": "Rails is a web framework." },
                { "id": 2, "title": "DHH", "url": "https://dhh.dk" }
            ] },
            { "type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
              "content": [{ "type": "output_text", "text": text, "annotations": [] }] }
        ],
        "usage": { "input_tokens": 12, "output_tokens": 7,
                   "tool_calls_details": { "search_web": { "cost_usd": 0.0025, "invocation": 1 } } }
    })
}

// spec: protocols/perplexity/agent_spec.rb:150 counts the web searches Perplexity bills
#[tokio::test]
async fn counts_the_web_searches_perplexity_bills() {
    let server = serve(vec![searched_agent_body(
        "Rails was created by David Heinemeier Hansson.[1]",
    )])
    .await;
    let message = agent(&server).ask("Who created Rails?").await.unwrap();
    assert_eq!(
        message.tokens().server_tool_use,
        counts(&[("web_search_requests", 1)])
    );
}

// spec: protocols/perplexity/agent_spec.rb:158 counts the web searches Perplexity bills while streaming
#[tokio::test]
async fn counts_the_web_searches_perplexity_bills_while_streaming() {
    let (result, _) = stream_on(
        agent,
        &[
            json!({ "type": "response.output_text.delta", "delta": "Rails" }),
            json!({ "type": "response.completed", "response": searched_agent_body("Rails") }),
        ],
    )
    .await;
    assert_eq!(
        result.unwrap().tokens().server_tool_use,
        counts(&[("web_search_requests", 1)])
    );
}

// ================================================================================================
// providers/xai/responses_spec.rb
// ================================================================================================

fn xai(server: &MockServer) -> Chat {
    responses_chat(server, "xai", "grok-4.3")
}

fn xai_usage() -> Value {
    json!({
        "input_tokens": 12_540, "output_tokens": 486, "num_sources_used": 0, "num_server_side_tools_used": 3,
        "server_side_tool_usage_details": {
            "web_search_calls": 2, "x_search_calls": 0, "code_interpreter_calls": 1, "file_search_calls": 0,
            "mcp_calls": 0, "document_search_calls": 0, "image_generation_calls": 0
        }
    })
}

// spec: providers/xai/responses_spec.rb:52 server tool use > names each tool that ran the way other providers do
#[tokio::test]
async fn xai_names_each_tool_that_ran_the_way_other_providers_do() {
    let message = parse_on(
        xai,
        json!({ "status": "completed", "output": [], "usage": xai_usage() }),
    )
    .await;
    assert_eq!(
        message.tokens.server_tool_use,
        counts(&[("web_search_requests", 2), ("code_execution_requests", 1)])
    );
}

// spec: providers/xai/responses_spec.rb:59 server tool use > counts the tools a completed stream reports
#[tokio::test]
async fn xai_counts_the_tools_a_completed_stream_reports() {
    let (_, chunks) = stream_on(
        xai,
        &[json!({ "type": "response.completed",
                  "response": { "status": "completed", "output": [], "usage": xai_usage() } })],
    )
    .await;
    assert_eq!(
        chunks[0].tokens.server_tool_use,
        counts(&[("web_search_requests", 2), ("code_execution_requests", 1)])
    );
}

// ================================================================================================
// providers/openrouter/chat_spec.rb
// ================================================================================================

fn openrouter(server: &MockServer) -> Chat {
    Chat::with_config(
        config(server),
        Some("claude-haiku-4-5"),
        Some("openrouter"),
        true,
    )
    .unwrap()
}

/// `OpenRouter::ChatCompletions#build_chunk` on each event, through one stream.
async fn openrouter_chunks(frames: &[Value]) -> Vec<Message> {
    let (result, chunks) = stream_on(openrouter, frames).await;
    result.unwrap();
    chunks
}

// spec: providers/openrouter/chat_spec.rb:141 #build_chunk > counts the web searches the final usage chunk reports
#[tokio::test]
async fn openrouter_counts_the_web_searches_the_final_usage_chunk_reports() {
    let chunks = openrouter_chunks(&[json!({
        "model": "openai/gpt-5.2", "choices": [],
        "usage": { "prompt_tokens": 10_977, "completion_tokens": 367, "cost": 0.04031575,
                   "server_tool_use_details": { "web_search_requests": 2 } }
    })])
    .await;
    assert_eq!(
        chunks[0].tokens.server_tool_use,
        counts(&[("web_search_requests", 2)])
    );
}

// spec: providers/openrouter/chat_spec.rb:159 #build_chunk > keeps the url citations a streamed delta annotates
#[tokio::test]
async fn openrouter_keeps_the_url_citations_a_streamed_delta_annotates() {
    let chunks = openrouter_chunks(&[json!({
        "model": "openai/gpt-5.2",
        "choices": [{ "index": 0, "delta": {
            "content": "", "role": "assistant",
            "annotations": [{ "type": "url_citation", "url_citation": {
                "url": "https://www.ruby-lang.org/en/downloads/", "title": "Download Ruby | Ruby",
                "start_index": 54, "end_index": 112 } }]
        }, "finish_reason": null }]
    })])
    .await;
    let urls: Vec<_> = chunks[0]
        .citations
        .iter()
        .map(|c| c.url.as_deref())
        .collect();
    assert_eq!(urls, [Some("https://www.ruby-lang.org/en/downloads/")]);
    assert_eq!(
        chunks[0].citations[0].title.as_deref(),
        Some("Download Ruby | Ruby")
    );
}

// spec: providers/openrouter/chat_spec.rb:192 #build_chunk > leaves citations empty on a chunk that carries none
#[tokio::test]
async fn openrouter_leaves_citations_empty_on_a_chunk_that_carries_none() {
    let chunks = openrouter_chunks(&[json!({
        "model": "openai/gpt-5.2", "choices": [{ "index": 0, "delta": { "content": "Ruby" } }]
    })])
    .await;
    assert_eq!(chunks[0].content.as_deref(), Some("Ruby"));
    assert!(chunks[0].citations.is_empty());
}

/// Ruby checks that the third chunk's detail is the very Hash the second chunk handed out
/// (`be(text)`), appended in place. A Rust chunk owns its `raw_reasoning`, so object identity has
/// no counterpart; what the in-place append guarantees (one detail per index whose text grows
/// chunk by chunk, never a second entry) is asserted on every chunk.
// spec: providers/openrouter/chat_spec.rb:202 #build_chunk > appends streamed reasoning text to its detail in place
#[tokio::test]
async fn openrouter_appends_streamed_reasoning_text_to_its_detail_in_place() {
    let delta = |text: &str| {
        json!({ "choices": [{ "delta": { "reasoning_details": [
            { "type": "reasoning.text", "index": 0, "text": text }
        ] } }] })
    };
    let chunks = openrouter_chunks(&[delta("Let"), delta(" me"), delta(" think")]).await;
    let details: Vec<Value> = chunks
        .iter()
        .map(|c| c.raw_reasoning.clone().unwrap_or(Value::Null))
        .collect();
    assert_eq!(
        details,
        [
            json!([{ "type": "reasoning.text", "index": 0, "text": "Let" }]),
            json!([{ "type": "reasoning.text", "index": 0, "text": "Let me" }]),
            json!([{ "type": "reasoning.text", "index": 0, "text": "Let me think" }]),
        ]
    );
}

// spec: providers/openrouter/chat_spec.rb:256 #format_messages > uses a boundary lifetime ahead of the configured ttl
#[tokio::test]
async fn openrouter_uses_a_boundary_lifetime_ahead_of_the_configured_ttl() {
    let server = MockServer::start().await;
    let chat = openrouter(&server)
        .with_caching(json!({ "ttl": "5m" }))
        .unwrap();
    let payload = render(
        chat,
        vec![Message::user("Long context").with_cache_until_here(Some("1h"))],
    );
    let last = payload["messages"][0]["content"]
        .as_array()
        .and_then(|c| c.last())
        .cloned()
        .unwrap();
    assert_eq!(
        last["cache_control"],
        json!({ "type": "ephemeral", "ttl": "1h" })
    );
}

/// The spec's `answer(thinking, producer:)`: an assistant answer whose usage records `producer`.
fn answer(thinking: Option<Thinking>, producer: Option<&str>) -> Message {
    let mut m = Message::assistant("done");
    m.thinking = thinking;
    if let Some(producer) = producer {
        let mut entry = UsageEntry::new(Operation::Chat, producer, Some("claude-haiku-4-5"));
        entry.status = UsageStatus::Succeeded;
        m.usage_entries = vec![entry];
    }
    m
}

/// `format_thinking(message)`: what the rendered assistant message carries besides role and content.
async fn format_thinking(message: Message) -> Map<String, Value> {
    let server = MockServer::start().await;
    let mut out = render(openrouter(&server), vec![message])["messages"][0]
        .as_object()
        .cloned()
        .unwrap();
    out.remove("role");
    out.remove("content");
    out
}

// spec: providers/openrouter/chat_spec.rb:456 #format_thinking > sends no reasoning whose producer is unknown
#[tokio::test]
async fn openrouter_sends_no_reasoning_whose_producer_is_unknown() {
    let message = answer(
        Thinking::build(Some("why".into()), Some("gemini-signature".into())),
        None,
    );
    assert_eq!(format_thinking(message).await, Map::new());
}

// spec: providers/openrouter/chat_spec.rb:462 #format_thinking > sends the reasoning details OpenRouter returned whoever is known to have produced them
#[tokio::test]
async fn openrouter_sends_the_reasoning_details_it_returned_whoever_produced_them() {
    let details =
        json!([{ "type": "reasoning.text", "text": "why", "signature": "sig", "index": 0 }]);
    let mut message = Message::assistant("done");
    message.raw_reasoning = Some(details.clone());
    assert_eq!(
        Value::Object(format_thinking(message).await),
        json!({ "reasoning_details": details })
    );
}

// spec: providers/openrouter/chat_spec.rb:542 errors reported with a 200 status > raises the error an event reports in the middle of a stream by its code
#[tokio::test]
async fn openrouter_raises_the_error_an_event_reports_in_the_middle_of_a_stream_by_its_code() {
    let body = "data: {\"id\":\"gen-1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hel\"}}]}\n\n\
                data: {\"id\":\"gen-1\",\"object\":\"chat.completion.chunk\",\"error\":{\"code\":502,\"message\":\"Provider returned error\"},\"choices\":[{\"index\":0,\"delta\":{\"content\":\"\"},\"finish_reason\":\"error\"}]}\n\n";
    let server = serve_templates(vec![sse(body.to_string())]).await;
    let error = openrouter(&server)
        .ask_stream("Hello", |_| {})
        .await
        .unwrap_err();
    assert!(
        matches!(&error, Error::ServiceUnavailable(m, _) if m == "Provider returned error"),
        "{error:?}"
    );
}

// spec: providers/openrouter/chat_spec.rb:554 errors reported with a 200 status > raises the error a response body reports by its code
#[tokio::test]
async fn openrouter_raises_the_error_a_response_body_reports_by_its_code() {
    let server = serve_templates(vec![ResponseTemplate::new(200).set_body_json(
        json!({ "id": "gen-1", "error": { "code": 429, "message": "Provider returned error" } }),
    )])
    .await;
    let error = openrouter(&server).ask("Hello").await.unwrap_err();
    assert!(
        matches!(&error, Error::RateLimit(m, _) if m == "Provider returned error"),
        "{error:?}"
    );
}

// ================================================================================================
// protocols/chat_completions/{chat,streaming}_spec.rb
// ================================================================================================

fn openai_chat_completions(server: &MockServer) -> Chat {
    Chat::with_config(config(server), Some("gpt-4.1-nano"), Some("openai"), true)
        .unwrap()
        .with_protocol(ProtocolName::ChatCompletions)
}

// spec: protocols/chat_completions/chat_spec.rb:210 .format_messages > sends OpenAI tool-returned images at original detail
#[tokio::test]
async fn sends_openai_tool_returned_images_at_original_detail() {
    let server = MockServer::start().await;
    let payload = render(openai_chat_completions(&server), vec![tool_image().await]);
    assert_eq!(
        payload["messages"][1]["content"][1]["image_url"]["detail"],
        json!("original")
    );
}

// spec: protocols/chat_completions/streaming_spec.rb:79 #parse_streaming_error > reads the HTTP status from a numeric code
#[tokio::test]
async fn reads_the_http_status_from_a_numeric_code() {
    let data = r#"{"error":{"code":502,"message":"Provider returned error"}}"#;
    assert_eq!(
        rust_llm::protocols::streaming_error_status(ProtocolName::ChatCompletions)(data),
        Some(502)
    );
    let server = serve_templates(vec![sse(format!("data: {data}\n\n"))]).await;
    let error = openai_chat_completions(&server)
        .ask_stream("hi", |_| {})
        .await
        .unwrap_err();
    assert!(
        matches!(&error, Error::ServiceUnavailable(m, Some(r)) if m == "Provider returned error" && r.status == 502),
        "{error:?}"
    );
}

// ================================================================================================
// chat_provider_tools_spec.rb: streamed web search (recorded)
// ================================================================================================

const SEARCH: &str = "Search the web: what is the latest stable Ruby version?";
const SEARCH_AND_CITE: &str =
    "Search the web: what is the latest stable Ruby version? Cite your source.";

fn searching(cassette: &Cassette, provider: &str, model: &str) -> Chat {
    chat_for(cassette, provider, model).with_provider_tools([ProviderTool::alias("web_search")])
}

fn kinds(message: &Message) -> Vec<String> {
    message
        .server_tool_calls
        .iter()
        .map(|c| c.kind.clone())
        .collect()
}

fn web_searches(message: &Message) -> i64 {
    message
        .tokens()
        .server_tool_use
        .and_then(|u| u.get("web_search_requests").and_then(Value::as_i64))
        .unwrap_or(0)
}

// spec: chat_provider_tools_spec.rb:280 web search > with openai/#{model_for(:openai, :reasoning_effort)} > streams searches and counts the ones it bills
#[tokio::test]
async fn openai_streams_searches_and_counts_the_ones_it_bills() {
    let cassette =
        start("chat_web_search_with_openai_gpt-5_2_streams_searches_and_counts_the_ones_it_bills")
            .await;
    let mut chunks = 0;
    let response = searching(&cassette, "openai", "gpt-5.2")
        .ask_stream(SEARCH, |_| chunks += 1)
        .await
        .unwrap();
    assert!(chunks > 0);
    assert!(
        kinds(&response).contains(&"web_search_call".into()),
        "{:?}",
        kinds(&response)
    );
    assert!(web_searches(&response) > 0, "{:?}", response.tokens());
    cassette.assert_all_matched().await;
}

// spec: chat_provider_tools_spec.rb:318 web search > with gemini/#{model_for(:gemini, :provider_tools)} > streams the grounding and counts the searches it ran
#[tokio::test]
async fn gemini_streams_the_grounding_and_counts_the_searches_it_ran() {
    let cassette = start("chat_web_search_with_gemini_gemini-3_5-flash_streams_the_grounding_and_counts_the_searches_it_ran").await;
    let mut chunks = Vec::new();
    let response = searching(&cassette, "gemini", "gemini-3.5-flash")
        .ask_stream(SEARCH_AND_CITE, |c| chunks.push(c.clone()))
        .await
        .unwrap();
    assert!(!chunks.is_empty());
    assert!(
        kinds(&response).contains(&"google_search".into()),
        "{:?}",
        kinds(&response)
    );
    assert!(!response.citations.is_empty());
    assert!(web_searches(&response) > 0, "{:?}", response.tokens());
    let suggestions: String = chunks
        .iter()
        .flat_map(|c| c.server_tool_calls.iter())
        .filter_map(|c| c.search_suggestions.clone())
        .collect();
    assert!(suggestions.contains("<style>"), "{suggestions}");
    cassette.assert_all_matched().await;
}

// spec: chat_provider_tools_spec.rb:345 web search > with openrouter/#{model_for(:openrouter, :provider_tools)} > streams searches and counts them
#[tokio::test]
async fn openrouter_streams_searches_and_counts_them() {
    let cassette =
        start("chat_web_search_with_openrouter_openai_gpt-5_2_streams_searches_and_counts_them")
            .await;
    let mut chunks = 0;
    let response = searching(&cassette, "openrouter", "openai/gpt-5.2")
        .ask_stream(SEARCH_AND_CITE, |_| chunks += 1)
        .await
        .unwrap();
    assert!(chunks > 0);
    assert!(!response.content().is_empty());
    assert!(
        response
            .citations
            .iter()
            .any(|c| c.url.as_deref() == Some("https://www.ruby-lang.org/en/downloads/")),
        "{:?}",
        response.citations
    );
    assert!(web_searches(&response) > 0, "{:?}", response.tokens());
    cassette.assert_all_matched().await;
}

// spec: chat_provider_tools_spec.rb:372 web search > with xai/#{model_for(:xai, :provider_tools)} > streams searches and counts them
#[tokio::test]
async fn xai_streams_searches_and_counts_them() {
    let cassette =
        start("chat_web_search_with_xai_grok-4_3_streams_searches_and_counts_them").await;
    let response = searching(&cassette, "xai", "grok-4.3")
        .ask_stream(SEARCH, |_| {})
        .await
        .unwrap();
    assert!(!response.citations.is_empty());
    assert!(web_searches(&response) > 0, "{:?}", response.tokens());
    cassette.assert_all_matched().await;
}
