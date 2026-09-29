//! Responses protocol specs ported from RubyLLM 2.0: `protocols/responses/{approvals,chat,media,
//! streaming}_spec.rb`. Ruby calls the protocol's private `render_payload`,
//! `parse_completion_response`, `build_chunk`, and `parse_streaming_error`; these go through
//! `Chat#render`, `generate`, `ask`, and `ask_stream` against a mock OpenAI server, which run the
//! same code (the stream accumulator included).

mod spec_helpers;

use async_trait::async_trait;
use rust_llm::{
    Attachment, Chat, Error, Message, ProtocolName, Resolution, ThinkingConfig, ThinkingDisplay,
    Tool, ToolCall, ToolError, ToolResult,
};
use serde_json::{Map, Value, json};
use spec_helpers::*;
use wiremock::MockServer;

fn fixture(name: &str) -> String {
    format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
}

/// A fixture loaded into memory, as `Attachment.new(path)` reads it before rendering.
fn inline(name: &str) -> Attachment {
    Attachment::from_bytes(std::fs::read(fixture(name)).unwrap(), name, None)
}

/// `RubyLLM.chat(model: 'gpt-5-nano', provider: :openai, protocol: :responses)`.
fn openai(server: &MockServer) -> Chat {
    Chat::with_config(config(server), Some("gpt-5-nano"), Some("openai"), false)
        .unwrap()
        .with_protocol(ProtocolName::Responses)
}

fn user_chat(server: &MockServer) -> Chat {
    let mut chat = openai(server);
    chat.add_message(Message::user("hi"));
    chat
}

/// The spec's `response_with(output, usage:)` body.
fn response_with(output: Value, usage: Value) -> Value {
    json!({ "model": "gpt-5-nano", "output": output, "usage": usage, "status": "completed" })
}

/// Parses `body` through a non-streaming `generate` (tool calls are not executed).
async fn parse(body: Value) -> rust_llm::Result<Message> {
    let server = serve(vec![body]).await;
    user_chat(&server).generate().await
}

/// An SSE body carrying each event as a `data:` frame.
fn events(events: &[Value]) -> String {
    events.iter().map(|e| format!("data: {e}\n\n")).collect()
}

/// Streams `frames` through `ask_stream`, returning the final message and every chunk.
async fn stream(frames: &[Value]) -> (rust_llm::Result<Message>, Vec<Message>) {
    let server = serve_templates(vec![sse(events(frames))]).await;
    let mut chat = openai(&server);
    let mut chunks = Vec::new();
    let result = chat.ask_stream("hi", |c| chunks.push(c.clone())).await;
    (result, chunks)
}

// ---- responses/chat_spec.rb #render_payload ----------------------------------------------------

struct StrictWeather;

#[async_trait]
impl Tool for StrictWeather {
    fn name(&self) -> String {
        "weather".into()
    }
    fn description(&self) -> String {
        "Looks up weather".into()
    }
    fn parameters_schema(&self) -> Option<Value> {
        Some(json!({ "type": "object" }))
    }
    fn provider_options(&self) -> Map<String, Value> {
        json!({ "strict": true }).as_object().cloned().unwrap()
    }
    async fn execute(&self, _: Map<String, Value>, _: &ToolCall) -> Result<ToolResult, ToolError> {
        Ok("Sunny".into())
    }
}

// spec: protocols/responses/chat_spec.rb:90 #render_payload > lets tools opt into strict mode via provider_options
#[tokio::test]
async fn tools_opt_into_strict_mode_via_provider_options() {
    let server = serve(vec![]).await;
    let mut chat = openai(&server).with_tool(StrictWeather);
    chat.add_message(Message::user("hi"));
    assert_eq!(chat.render().unwrap()["tools"][0]["strict"], json!(true));
}

// spec: protocols/responses/chat_spec.rb:123 #render_payload > asks for a reasoning summary when display is summarized
#[tokio::test]
async fn asks_for_a_reasoning_summary_when_display_is_summarized() {
    let server = serve(vec![]).await;
    let mut chat = openai(&server)
        .with_thinking(ThinkingConfig::effort("low").with_display(ThinkingDisplay::Summarized));
    chat.add_message(Message::user("hi"));
    assert_eq!(
        chat.render().unwrap()["reasoning"],
        json!({ "effort": "low", "summary": "auto" })
    );
}

// spec: protocols/responses/chat_spec.rb:153 #render_payload > marks cache boundaries without disabling implicit caching
#[tokio::test]
async fn marks_cache_boundaries_without_disabling_implicit_caching() {
    let server = serve(vec![]).await;
    let mut chat = openai(&server);
    chat.add_message(Message::user("Long context"));
    chat.cache_until_here().unwrap();
    chat.add_message(Message::user("hi"));
    let payload = chat.render().unwrap();
    assert_eq!(
        payload["input"][0]["content"],
        json!([{ "type": "input_text", "text": "Long context", "prompt_cache_breakpoint": { "mode": "explicit" } }])
    );
    assert_eq!(
        payload["input"].as_array().unwrap().last().unwrap(),
        &json!({ "role": "user", "content": "hi" })
    );
    assert!(payload.get("prompt_cache_options").is_none());
}

// spec: protocols/responses/chat_spec.rb:168 #render_payload > preserves cache options alongside explicit boundaries
#[tokio::test]
async fn preserves_cache_options_alongside_explicit_boundaries() {
    let server = serve(vec![]).await;
    let mut chat = openai(&server)
        .with_caching(json!({ "ttl": "30m" }))
        .unwrap();
    chat.add_message(Message::user("Long context"));
    chat.cache_until_here().unwrap();
    let payload = chat.render().unwrap();
    assert_eq!(payload["prompt_cache_options"], json!({ "ttl": "30m" }));
    let parts = payload["input"][0]["content"].as_array().unwrap();
    assert_eq!(
        parts.last().unwrap()["prompt_cache_breakpoint"],
        json!({ "mode": "explicit" })
    );
}

// spec: protocols/responses/chat_spec.rb:177 #render_payload > sends cache-bounded system messages as input items
#[tokio::test]
async fn sends_cache_bounded_system_messages_as_input_items() {
    let server = serve(vec![]).await;
    let mut chat = openai(&server);
    chat.set_instructions(Some("Stable instructions".into()), false, true);
    chat.add_message(Message::user("hi"));
    let payload = chat.render().unwrap();
    assert!(payload.get("instructions").is_none());
    assert_eq!(
        payload["input"][0],
        json!({ "role": "system", "content": [{ "type": "input_text", "text": "Stable instructions", "prompt_cache_breakpoint": { "mode": "explicit" } }] })
    );
}

// ---- responses/chat_spec.rb #parse_completion_response -----------------------------------------

// spec: protocols/responses/chat_spec.rb:214 #parse_completion_response > preserves web, file-search and container-file citations
#[tokio::test]
async fn preserves_web_file_search_and_container_file_citations() {
    let message = parse(response_with(
        json!([{ "type": "message", "content": [{
            "type": "output_text", "text": "Ruby facts", "annotations": [
                { "type": "url_citation", "url": "https://ruby-lang.org", "title": "Ruby", "start_index": 0, "end_index": 4 },
                { "type": "file_citation", "file_id": "file_facts", "filename": "facts.pdf", "index": 0 },
                { "type": "container_file_citation", "container_id": "container_1", "file_id": "file_report",
                  "filename": "report.txt", "start_index": 5, "end_index": 10 },
                { "type": "file_path", "file_id": "file_download", "index": 1 }
            ]
        }] }]),
        json!({}),
    ))
    .await
    .unwrap();
    let c = &message.citations;
    assert_eq!(c.len(), 3);
    assert_eq!(
        (
            c[0].url.as_deref(),
            c[0].text.as_deref(),
            c[0].source_id.as_deref()
        ),
        (Some("https://ruby-lang.org"), Some("Ruby"), None)
    );
    assert_eq!(c[1].source_id.as_deref(), Some("file_facts"));
    assert_eq!(c[1].title.as_deref(), Some("facts.pdf"));
    assert_eq!(c[1].source_index, Some(0));
    assert_eq!(
        (c[1].start_index, c[1].end_index, c[1].url.as_deref()),
        (None, None, None)
    );
    assert_eq!(
        (
            c[2].source_id.as_deref(),
            c[2].title.as_deref(),
            c[2].text.as_deref()
        ),
        (Some("file_report"), Some("report.txt"), Some("facts"))
    );
}

// spec: protocols/responses/chat_spec.rb:240 #parse_completion_response > places citation spans against all preceding response text
#[tokio::test]
async fn places_citation_spans_against_all_preceding_response_text() {
    let message = parse(response_with(
        json!([
            { "type": "message", "content": [{ "type": "output_text", "text": "Café. " }] },
            { "type": "message", "content": [
                { "type": "output_text", "text": "Read " },
                { "type": "output_text", "text": "Ruby", "annotations": [
                    { "type": "url_citation", "url": "https://ruby-lang.org", "start_index": 0, "end_index": 4 }
                ] }
            ] }
        ]),
        json!({}),
    ))
    .await
    .unwrap();
    let citation = &message.citations[0];
    assert_eq!(
        (
            citation.start_index,
            citation.end_index,
            citation.text.as_deref()
        ),
        (Some(11), Some(15), Some("Ruby"))
    );
    let span: String = message.content().chars().skip(11).take(4).collect();
    assert_eq!(Some(span.as_str()), citation.text.as_deref());
}

// spec: protocols/responses/chat_spec.rb:260 #parse_completion_response > surfaces refusal parts as content
#[tokio::test]
async fn surfaces_refusal_parts_as_content() {
    let message = parse(response_with(
        json!([{ "type": "message", "content": [{ "type": "refusal", "refusal": "I cannot help with that." }] }]),
        json!({}),
    ))
    .await
    .unwrap();
    assert_eq!(message.content(), "I cannot help with that.");
}

/// Ruby also asserts `error.response == response` and a `JSON::ParserError` cause; the port's
/// `ToolCallParse` carries neither (no response field on the variant).
// spec: protocols/responses/chat_spec.rb:284 #parse_completion_response > wraps malformed function-call arguments in a RubyLLM error
#[tokio::test]
async fn wraps_malformed_function_call_arguments_in_a_rust_llm_error() {
    let body = json!({
        "model": "gpt-5-nano",
        "output": [{ "type": "function_call", "call_id": "call_1", "name": "weather", "arguments": "{\"city\":\"Berlin\"" }],
        "status": "incomplete",
        "incomplete_details": { "reason": "max_output_tokens" }
    });
    match parse(body).await.unwrap_err() {
        Error::ToolCallParse { finish_reason, .. } => {
            assert_eq!(finish_reason.as_deref(), Some("max_tokens"))
        }
        other => panic!("expected ToolCallParse, got {other:?}"),
    }
}

// spec: protocols/responses/chat_spec.rb:326 #parse_completion_response > maps usage with cached and reasoning tokens
#[tokio::test]
async fn maps_usage_with_cached_and_reasoning_tokens() {
    let message = parse(response_with(
        json!([]),
        json!({ "input_tokens": 10, "output_tokens": 7, "input_tokens_details": { "cached_tokens": 4 },
                "output_tokens_details": { "reasoning_tokens": 3 } }),
    ))
    .await
    .unwrap();
    assert_eq!(message.tokens.input, Some(6));
    assert_eq!(message.tokens.output, Some(7));
    assert_eq!(message.tokens.cache_read, Some(4));
    assert_eq!(message.tokens.thinking, Some(3));
}

// spec: protocols/responses/chat_spec.rb:342 #parse_completion_response > maps cache write tokens for models that bill cache writes
#[tokio::test]
async fn maps_cache_write_tokens_for_models_that_bill_cache_writes() {
    let message = parse(response_with(
        json!([]),
        json!({ "input_tokens": 2048, "output_tokens": 7, "input_tokens_details": { "cached_tokens": 1920, "cache_write_tokens": 100 } }),
    ))
    .await
    .unwrap();
    assert_eq!(message.tokens.input, Some(28));
    assert_eq!(message.tokens.cache_read, Some(1920));
    assert_eq!(message.tokens.cache_write, Some(100));
}

// spec: protocols/responses/chat_spec.rb:356 #parse_completion_response > reports the completed status as finish_reason for function calls
#[tokio::test]
async fn reports_the_completed_status_as_finish_reason_for_function_calls() {
    let message = parse(response_with(
        json!([{ "type": "function_call", "call_id": "call_1", "name": "weather", "arguments": "{}" }]),
        json!({}),
    ))
    .await
    .unwrap();
    assert_eq!(message.finish_reason, Some(rust_llm::FinishReason::Stop));
    assert!(message.is_tool_call_stop());
    assert!(!message.is_stopped());
}

// spec: protocols/responses/chat_spec.rb:369 #parse_completion_response > preserves incomplete_details reason as finish_reason when present
#[tokio::test]
async fn preserves_incomplete_details_reason_as_finish_reason() {
    let message = parse(json!({
        "model": "gpt-5-nano", "output": [], "status": "incomplete", "incomplete_details": { "reason": "max_output_tokens" }
    }))
    .await
    .unwrap();
    assert_eq!(
        message.finish_reason,
        Some(rust_llm::FinishReason::MaxTokens)
    );
}

// ---- responses/media_spec.rb -------------------------------------------------------------------

fn rendered_parts(chat: &Chat) -> rust_llm::Result<Vec<Value>> {
    chat.render().map(|p| {
        p["input"][0]["content"]
            .as_array()
            .cloned()
            .unwrap_or_default()
    })
}

// spec: protocols/responses/media_spec.rb:48 .format_content > still rejects attachments the API cannot take
#[tokio::test]
async fn still_rejects_attachments_the_api_cannot_take() {
    let server = serve(vec![]).await;
    let mut chat = openai(&server);
    chat.ask_later_with("Listen", vec![inline("ruby.wav")])
        .unwrap();
    match rendered_parts(&chat).unwrap_err() {
        Error::UnsupportedAttachment(m) => assert!(m.contains("audio/wav"), "{m}"),
        other => panic!("expected UnsupportedAttachment, got {other:?}"),
    }
}

// spec: protocols/responses/media_spec.rb:55 .format_content > maps low resolution to low image detail
// spec: protocols/responses/media_spec.rb:63 .format_content > maps higher resolutions to high image detail
#[tokio::test]
async fn maps_image_resolution_to_detail() {
    let server = serve(vec![]).await;
    for (resolution, detail) in [(Resolution::Low, "low"), (Resolution::UltraHigh, "high")] {
        let mut chat = openai(&server);
        chat.ask_later_with(
            "Describe this",
            vec![inline("ruby.png").with_resolution(resolution)],
        )
        .unwrap();
        assert_eq!(rendered_parts(&chat).unwrap()[1]["detail"], json!(detail));
    }
}

// ---- responses/approvals_spec.rb ---------------------------------------------------------------

fn approval() -> Value {
    json!({ "type": "mcp_approval_request", "id": "approval_1", "server_label": "docs", "name": "search",
            "arguments": "{\"query\":\"Ruby\"}" })
}

// spec: protocols/responses/approvals_spec.rb:28 accepts arguments already decoded by the provider
#[tokio::test]
async fn accepts_arguments_already_decoded_by_the_provider() {
    let mut item = approval();
    item["arguments"] = json!({ "query": "Ruby" });
    let message = parse(json!({ "status": "completed", "output": [item] }))
        .await
        .unwrap();
    assert_eq!(
        Value::Object(
            message
                .tool_calls
                .unwrap()
                .get("approval_1")
                .unwrap()
                .arguments()
        ),
        json!({ "query": "Ruby" })
    );
}

// spec: protocols/responses/approvals_spec.rb:34 identifies remote approvals independently of their provider label
#[tokio::test]
async fn identifies_remote_approvals_independently_of_their_provider_label() {
    let mut item = approval();
    item.as_object_mut().unwrap().remove("server_label");
    let message = parse(json!({ "status": "completed", "output": [item] }))
        .await
        .unwrap();
    assert!(
        message
            .tool_calls
            .unwrap()
            .get("approval_1")
            .unwrap()
            .remote
    );
}

// spec: protocols/responses/approvals_spec.rb:68 preserves a streamed approval exactly once across repeated final events
#[tokio::test]
async fn preserves_a_streamed_approval_exactly_once_across_repeated_final_events() {
    let event = json!({ "type": "response.completed", "response": { "status": "completed", "output": [approval()] } });
    let (result, _) = stream(&[event.clone(), event]).await;
    let message = result.unwrap();
    assert_eq!(message.server_tool_calls.len(), 1);
    let calls = message.tool_calls.unwrap();
    assert_eq!(calls.keys().cloned().collect::<Vec<_>>(), ["approval_1"]);
    let call = calls.get("approval_1").unwrap();
    assert!(call.remote);
    assert_eq!(Value::Object(call.arguments()), json!({ "query": "Ruby" }));
}

// ---- responses/streaming_spec.rb ---------------------------------------------------------------

// spec: protocols/responses/streaming_spec.rb:18 streams refusal deltas as content
#[tokio::test]
async fn streams_refusal_deltas_as_content() {
    let (result, chunks) =
        stream(&[json!({ "type": "response.refusal.delta", "delta": "I cannot help" })]).await;
    assert_eq!(chunks[0].content.as_deref(), Some("I cannot help"));
    assert_eq!(result.unwrap().content(), "I cannot help");
}

// spec: protocols/responses/streaming_spec.rb:24 streams file citations with their source identities
#[tokio::test]
async fn streams_file_citations_with_their_source_identities() {
    let (result, chunks) = stream(&[json!({
        "type": "response.output_text.annotation.added",
        "annotation": { "type": "file_citation", "file_id": "file_facts", "filename": "facts.pdf", "index": 0 }
    })])
    .await;
    result.unwrap();
    let c = &chunks[0].citations[0];
    assert_eq!(
        (c.source_id.as_deref(), c.title.as_deref(), c.source_index),
        (Some("file_facts"), Some("facts.pdf"), Some(0))
    );
}

// spec: protocols/responses/streaming_spec.rb:34 keeps streamed citation positions across output parts without duplicating final annotations
#[tokio::test]
async fn keeps_streamed_citation_positions_without_duplicating_final_annotations() {
    let annotation = json!({ "type": "container_file_citation", "container_id": "container_1", "file_id": "file_report",
                             "filename": "report.txt", "start_index": 0, "end_index": 4 });
    let (result, _) = stream(&[
        json!({ "type": "response.output_text.delta", "output_index": 0, "content_index": 0, "delta": "Café. " }),
        json!({ "type": "response.output_text.delta", "output_index": 1, "content_index": 0, "delta": "Read " }),
        json!({ "type": "response.output_text.delta", "output_index": 1, "content_index": 1, "delta": "Ruby" }),
        json!({ "type": "response.output_text.annotation.added", "output_index": 1, "content_index": 1, "annotation": annotation }),
        json!({ "type": "response.completed", "response": { "status": "completed", "output": [
            { "type": "message", "content": [{ "type": "output_text", "text": "Café. " }] },
            { "type": "message", "content": [
                { "type": "output_text", "text": "Read " },
                { "type": "output_text", "text": "Ruby", "annotations": [annotation] }
            ] }
        ] } }),
    ])
    .await;
    let citations = result.unwrap().citations;
    assert_eq!(citations.len(), 1);
    let c = &citations[0];
    assert_eq!(
        (
            c.source_id.as_deref(),
            c.start_index,
            c.end_index,
            c.text.as_deref()
        ),
        (Some("file_report"), Some(11), Some(15), Some("Ruby"))
    );
}

// spec: protocols/responses/streaming_spec.rb:74 resets citation positions when the protocol starts another stream
#[tokio::test]
async fn resets_citation_positions_when_another_stream_starts() {
    let body = events(&[
        json!({ "type": "response.output_text.delta", "output_index": 0, "delta": "Read " }),
        json!({ "type": "response.output_text.delta", "output_index": 1, "delta": "Ruby" }),
        json!({ "type": "response.output_text.annotation.added", "output_index": 1,
                "annotation": { "type": "url_citation", "url": "https://ruby-lang.org", "start_index": 0, "end_index": 4 } }),
    ]);
    let server = serve_templates(vec![sse(body.clone()), sse(body)]).await;
    let mut chat = openai(&server);
    for _ in 0..2 {
        let response = chat.ask_stream("hi", |_| {}).await.unwrap();
        let c = &response.citations[0];
        assert_eq!(
            (c.start_index, c.end_index, c.text.as_deref()),
            (Some(5), Some(9), Some("Ruby"))
        );
    }
}

// spec: protocols/responses/streaming_spec.rb:91 streams reasoning summary deltas as thinking
#[tokio::test]
async fn streams_reasoning_summary_deltas_as_thinking() {
    let (result, chunks) =
        stream(&[json!({ "type": "response.reasoning_summary_text.delta", "delta": "hmm" })]).await;
    result.unwrap();
    assert_eq!(
        chunks[0].thinking.as_ref().and_then(|t| t.text.as_deref()),
        Some("hmm")
    );
}

// spec: protocols/responses/streaming_spec.rb:97 separates reasoning summary parts
#[tokio::test]
async fn separates_reasoning_summary_parts() {
    let (result, _) = stream(&[
        json!({ "type": "response.reasoning_summary_part.added", "summary_index": 0 }),
        json!({ "type": "response.reasoning_summary_text.delta", "delta": "**First summary**" }),
        json!({ "type": "response.reasoning_summary_part.added", "summary_index": 1 }),
        json!({ "type": "response.reasoning_summary_text.delta", "delta": "**Second summary**" }),
    ])
    .await;
    let message = result.unwrap();
    assert_eq!(
        message.thinking.and_then(|t| t.text).as_deref(),
        Some("**First summary**\n\n**Second summary**")
    );
}

// spec: protocols/responses/streaming_spec.rb:146 reads usage and model from the completed event
#[tokio::test]
async fn reads_usage_and_model_from_the_completed_event() {
    let (result, chunks) = stream(&[json!({ "type": "response.completed", "response": {
        "model": "gpt-5-nano", "status": "completed",
        "usage": { "input_tokens": 10, "output_tokens": 7, "input_tokens_details": { "cached_tokens": 4 },
                   "output_tokens_details": { "reasoning_tokens": 3 } }
    } })])
    .await;
    result.unwrap();
    let chunk = &chunks[0];
    assert_eq!(chunk.model.as_deref(), Some("gpt-5-nano"));
    assert_eq!(chunk.tokens.input, Some(6));
    assert_eq!(chunk.tokens.output, Some(7));
    assert_eq!(chunk.tokens.cache_read, Some(4));
    assert_eq!(chunk.tokens.thinking, Some(3));
    assert_eq!(chunk.finish_reason, Some(rust_llm::FinishReason::Stop));
}

// spec: protocols/responses/streaming_spec.rb:169 reports the completed status as finish_reason for function-call responses
#[tokio::test]
async fn reports_the_completed_status_as_finish_reason_for_streamed_function_calls() {
    let (result, chunks) = stream(&[json!({ "type": "response.completed", "response": {
        "model": "gpt-5-nano", "status": "completed",
        "output": [{ "type": "function_call", "call_id": "call_1", "name": "weather", "arguments": "{}" }]
    } })])
    .await;
    result.unwrap();
    assert_eq!(chunks[0].finish_reason, Some(rust_llm::FinishReason::Stop));
}

// spec: protocols/responses/streaming_spec.rb:220 preserves incomplete_details reason on completed events
#[tokio::test]
async fn preserves_incomplete_details_reason_on_completed_events() {
    let (result, chunks) = stream(&[json!({ "type": "response.completed", "response": {
        "model": "gpt-5-nano", "status": "incomplete", "incomplete_details": { "reason": "max_output_tokens" }
    } })])
    .await;
    result.unwrap();
    assert_eq!(
        chunks[0].finish_reason,
        Some(rust_llm::FinishReason::MaxTokens)
    );
}

// ---- responses/streaming_spec.rb #parse_streaming_error ----------------------------------------

async fn stream_error(frame: Value) -> Error {
    let (result, _) = stream(&[frame]).await;
    result.unwrap_err()
}

// spec: protocols/responses/streaming_spec.rb:190 #parse_streaming_error > classifies a rate limit reported by a flat error event
#[tokio::test]
async fn classifies_a_rate_limit_reported_by_a_flat_error_event() {
    let error = stream_error(json!({ "type": "error", "code": "rate_limit_exceeded", "message": "Slow down", "param": null, "sequence_number": 3 })).await;
    assert!(
        matches!(&error, Error::RateLimit(m, Some(r)) if m == "Slow down" && r.status == 429),
        "{error:?}"
    );
}

// spec: protocols/responses/streaming_spec.rb:199 #parse_streaming_error > classifies a server error reported by a flat error event
#[tokio::test]
async fn classifies_a_server_error_reported_by_a_flat_error_event() {
    let error = stream_error(
        json!({ "type": "error", "code": "server_error", "message": "Internal error" }),
    )
    .await;
    assert!(
        matches!(&error, Error::Server(_, Some(r)) if r.status == 500),
        "{error:?}"
    );
}

// spec: protocols/responses/streaming_spec.rb:205 #parse_streaming_error > falls back to a 400 for other flat error codes
#[tokio::test]
async fn falls_back_to_a_400_for_other_flat_error_codes() {
    let error =
        stream_error(json!({ "type": "error", "code": "invalid_prompt", "message": "Bad prompt" }))
            .await;
    assert!(
        matches!(&error, Error::BadRequest(m, Some(r)) if m == "Bad prompt" && r.status == 400),
        "{error:?}"
    );
}

// spec: protocols/responses/streaming_spec.rb:212 #parse_streaming_error > still classifies nested error objects
#[tokio::test]
async fn still_classifies_nested_error_objects() {
    let error =
        stream_error(json!({ "error": { "type": "rate_limit_exceeded", "message": "Slow down" } }))
            .await;
    assert!(
        matches!(&error, Error::RateLimit(m, Some(r)) if m == "Slow down" && r.status == 429),
        "{error:?}"
    );
}
