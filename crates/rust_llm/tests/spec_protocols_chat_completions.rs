//! Chat Completions protocol specs ported from RubyLLM 2.0: `protocols/chat_completions/{chat,
//! media,streaming,tools}_spec.rb`. Ruby calls the protocol's private helpers (`format_messages`,
//! `parse_completion_body`, `extract_citations`, `parse_streaming_error`); these read the payload
//! `Chat#render` produces, the message the public `parse_completion_body`/`build_chunk` return, or
//! the error a stream raises, which is what those helpers feed. `// spec:` lines tie each test to
//! its Ruby example.

mod spec_helpers;

use std::sync::Arc;

use rust_llm::message::RawResponse;
use rust_llm::protocols::chat_completions;
use rust_llm::{
    Attachment, Chat, Config, Error, FinishReason, Message, ProtocolName, Provider, Resolution,
    Thinking, ToolCall,
};
use serde_json::{Value, json};
use spec_helpers::{serve, serve_templates, sse, tool_call_message};
use wiremock::{MockServer, ResponseTemplate};

/// `include_context 'with configured RubyLLM'`, plus the Chat Completions providers this file uses.
fn config(server: &MockServer) -> Arc<Config> {
    let mut c = Config::default();
    for (provider, path) in [
        ("openai", "/v1"),
        ("deepseek", ""),
        ("xai", "/v1"),
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

/// `<Provider>::ChatCompletions` for `model`.
fn cc_chat(server: &MockServer, provider: &str, model: &str) -> Chat {
    Chat::with_config(config(server), Some(model), Some(provider), true)
        .unwrap()
        .with_protocol(ProtocolName::ChatCompletions)
}

fn openai(server: &MockServer) -> Chat {
    cc_chat(server, "openai", "gpt-4.1-nano")
}

/// `parse_completion_body(response_body, raw:)`.
fn parse(provider: Provider, data: Value) -> Message {
    chat_completions::parse_completion_body(provider, &data, RawResponse::default()).unwrap()
}

fn parse_err(data: Value) -> Error {
    chat_completions::parse_completion_body(Provider::OpenAI, &data, RawResponse::default())
        .unwrap_err()
}

/// A completion body whose single message carries `message` and whose usage is `usage`.
fn body(model: &str, message: Value, usage: Value) -> Value {
    json!({ "model": model, "choices": [{ "message": message }], "usage": usage })
}

fn hello(model: &str, usage: Value) -> Message {
    let provider = if model.starts_with("deepseek") {
        Provider::DeepSeek
    } else {
        Provider::OpenAI
    };
    parse(
        provider,
        body(
            model,
            json!({ "role": "assistant", "content": "Hello!" }),
            usage,
        ),
    )
}

/// The first rendered message for a user turn carrying `attachments`.
fn render_attachments(
    mut chat: Chat,
    text: &str,
    attachments: Vec<Attachment>,
) -> rust_llm::Result<Value> {
    chat.ask_later_with(text, attachments).unwrap();
    chat.render().map(|p| p["messages"][0].clone())
}

fn docx() -> Attachment {
    Attachment::from_bytes(b"docx bytes".to_vec(), "proposal.docx", None)
}

const DOCX_UNSUPPORTED: &str = "Unsupported attachment type: application/vnd.openxmlformats-officedocument.wordprocessingml.document";

fn assert_unsupported(result: rust_llm::Result<Value>, needle: &str) {
    match result {
        Err(Error::UnsupportedAttachment(m)) => assert!(m.contains(needle), "{m}"),
        other => panic!("expected UnsupportedAttachment, got {other:?}"),
    }
}

// ---- chat_spec.rb: .parse_completion_body -------------------------------------------------------

// spec: protocols/chat_completions/chat_spec.rb:7 .parse_completion_body > captures cached token information when present
#[test]
fn cached_tokens_are_split_out_of_the_prompt() {
    let m = hello(
        "gpt-4.1-nano",
        json!({ "prompt_tokens": 8, "completion_tokens": 4, "prompt_tokens_details": { "cached_tokens": 6 } }),
    );
    assert_eq!(m.tokens.cache_read, Some(6));
    assert_eq!(m.tokens.input, Some(2));
    assert_eq!(m.tokens.output, Some(4));
    assert_eq!(m.tokens.cache_write, Some(0));
}

// spec: protocols/chat_completions/chat_spec.rb:36 .parse_completion_body > captures cache write tokens for models that bill cache writes
#[test]
fn cache_write_tokens_are_split_out_of_the_prompt() {
    let m = hello(
        "gpt-5.6",
        json!({ "prompt_tokens": 2048, "completion_tokens": 4, "prompt_tokens_details": { "cached_tokens": 1920, "cache_write_tokens": 100 } }),
    );
    assert_eq!(m.tokens.cache_read, Some(1920));
    assert_eq!(m.tokens.cache_write, Some(100));
    assert_eq!(m.tokens.input, Some(28));
}

// spec: protocols/chat_completions/chat_spec.rb:64 .parse_completion_body > normalizes finish reasons
#[test]
fn finish_reasons_are_normalized() {
    let data = json!({
        "model": "gpt-4.1-nano",
        "choices": [{
            "finish_reason": "tool_calls",
            "message": { "role": "assistant", "content": "", "tool_calls": [
                { "id": "call_1", "type": "function", "function": { "name": "weather", "arguments": "{}" } }
            ] }
        }]
    });
    assert_eq!(
        parse(Provider::OpenAI, data).finish_reason,
        Some(FinishReason::ToolCalls)
    );
}

// spec: protocols/chat_completions/chat_spec.rb:94 .parse_completion_body > normalizes DeepSeek cache hit and miss usage fields
#[test]
fn deepseek_cache_hit_and_miss_fields_are_normalized() {
    let m = hello(
        "deepseek-v4-flash",
        json!({ "prompt_tokens": 206, "completion_tokens": 4, "prompt_cache_hit_tokens": 192, "prompt_cache_miss_tokens": 14 }),
    );
    assert_eq!(m.tokens.input, Some(14));
    assert_eq!(m.tokens.cache_read, Some(192));
    assert_eq!(m.tokens.output, Some(4));
    assert_eq!(m.tokens.cache_write, Some(0));
}

// spec: protocols/chat_completions/chat_spec.rb:124 .parse_completion_body > keeps OpenAI reasoning tokens inside completion output tokens
#[test]
fn openai_reasoning_tokens_stay_inside_output() {
    let m = hello(
        "gpt-5.5",
        json!({ "prompt_tokens": 50, "completion_tokens": 1306, "total_tokens": 1356, "completion_tokens_details": { "reasoning_tokens": 1087 } }),
    );
    assert_eq!(m.tokens.output, Some(1306));
    assert_eq!(m.tokens.thinking, Some(1087));
}

// spec: protocols/chat_completions/chat_spec.rb:152 .parse_completion_body > adds reasoning tokens to output for OpenAI-compatible providers that report them separately
#[test]
fn separately_reported_reasoning_tokens_are_added_to_output() {
    let m = hello(
        "grok-4-fast-reasoning",
        json!({ "prompt_tokens": 43, "completion_tokens": 101, "total_tokens": 9971, "completion_tokens_details": { "reasoning_tokens": 9827 } }),
    );
    assert_eq!(m.tokens.output, Some(9928));
    assert_eq!(m.tokens.thinking, Some(9827));
}

// spec: protocols/chat_completions/chat_spec.rb:180 .parse_completion_body > captures top-level reasoning tokens when providers report them outside completion details
#[test]
fn top_level_reasoning_tokens_are_captured() {
    let m = hello(
        "sonar-deep-research",
        json!({ "prompt_tokens": 33, "completion_tokens": 11_395, "total_tokens": 11_428, "reasoning_tokens": 193_947 }),
    );
    assert_eq!(m.tokens.output, Some(11_395));
    assert_eq!(m.tokens.thinking, Some(193_947));
}

// ---- chat_spec.rb: .format_messages -------------------------------------------------------------

// spec: protocols/chat_completions/chat_spec.rb:253 .format_messages > keeps non-PDF documents disabled for OpenAI chat completions
// spec: protocols/chat_completions/media_spec.rb:49 .format_content > raises an actionable error for arbitrary files unless the provider opts in
#[tokio::test]
async fn openai_chat_completions_rejects_docx() {
    let server = serve(vec![]).await;
    assert_unsupported(
        render_attachments(openai(&server), "Summarize this file", vec![docx()]),
        DOCX_UNSUPPORTED,
    );
}

// spec: protocols/chat_completions/chat_spec.rb:262 .format_messages > keeps unsupported files disabled for DeepSeek
#[tokio::test]
async fn deepseek_rejects_docx() {
    let server = serve(vec![]).await;
    let chat = cc_chat(&server, "deepseek", "deepseek-v4-flash");
    assert_unsupported(
        render_attachments(chat, "Summarize this file", vec![docx()]),
        DOCX_UNSUPPORTED,
    );
}

// spec: protocols/chat_completions/chat_spec.rb:298 .format_messages > uses Perplexity file_url parts for supported file attachments
#[tokio::test]
async fn perplexity_sends_supported_files_as_file_url_parts() {
    let server = serve(vec![]).await;
    let chat = cc_chat(&server, "perplexity", "openai/gpt-5-mini");
    let message = render_attachments(chat, "Summarize this file", vec![docx()]).unwrap();
    assert_eq!(
        message["content"][1],
        json!({ "type": "file_url", "file_url": { "url": "ZG9jeCBieXRlcw==" } })
    );
}

// spec: protocols/chat_completions/chat_spec.rb:309 .format_messages > keeps Perplexity text file attachments as text parts
#[tokio::test]
async fn perplexity_keeps_text_files_as_text_parts() {
    let server = serve(vec![]).await;
    for extension in ["csv", "txt", "md", "html", "json"] {
        let attachment =
            Attachment::from_bytes(b"notes".to_vec(), format!("notes.{extension}"), None);
        let chat = cc_chat(&server, "perplexity", "openai/gpt-5-mini");
        let message =
            render_attachments(chat, "Summarize this file", vec![attachment.clone()]).unwrap();
        assert_eq!(
            message["content"][1],
            json!({ "type": "text", "text": attachment.for_llm().unwrap() }),
            "{extension}"
        );
    }
}

// spec: protocols/chat_completions/chat_spec.rb:325 .format_messages > keeps unsupported files disabled for xAI
#[tokio::test]
async fn xai_rejects_docx() {
    let server = serve(vec![]).await;
    let chat = cc_chat(&server, "xai", "grok-4-1-fast-non-reasoning");
    assert_unsupported(
        render_attachments(chat, "Summarize this file", vec![docx()]),
        DOCX_UNSUPPORTED,
    );
}

// spec: protocols/chat_completions/chat_spec.rb:336 .format_messages > keeps PDF file parts disabled for xAI chat completions
#[tokio::test]
async fn xai_chat_completions_rejects_pdf() {
    let server = serve(vec![]).await;
    let chat = cc_chat(&server, "xai", "grok-4-1-fast-non-reasoning");
    let pdf = Attachment::from_bytes(b"pdf bytes".to_vec(), "proposal.pdf", None);
    assert_unsupported(
        render_attachments(chat, "Summarize this file", vec![pdf]),
        "Unsupported attachment type: application/pdf",
    );
}

// ---- chat_spec.rb: .render_payload --------------------------------------------------------------

fn render_hello(chat: Chat) -> Value {
    let mut chat = chat;
    chat.ask_later("Hello").unwrap();
    chat.render().unwrap()
}

// spec: protocols/chat_completions/chat_spec.rb:360 .render_payload > renders prompt cache params for any Chat Completions-compatible provider
#[tokio::test]
async fn prompt_cache_params_render_for_any_chat_completions_provider() {
    let server = serve(vec![]).await;
    for chat in [
        cc_chat(&server, "openai", "gpt-4o"),
        cc_chat(&server, "deepseek", "deepseek-v4-flash"),
    ] {
        let payload = render_hello(
            chat.with_caching(json!({ "key": "repo:ruby_llm", "ttl": "30m" }))
                .unwrap(),
        );
        assert_eq!(payload["prompt_cache_key"], json!("repo:ruby_llm"));
        assert_eq!(payload["prompt_cache_options"], json!({ "ttl": "30m" }));
    }
}

fn person_schema(strict: bool) -> Value {
    json!({
        "name": "PersonSchema",
        "schema": { "type": "object", "properties": { "name": { "type": "string" }, "age": { "type": "integer" } } },
        "strict": strict
    })
}

// spec: protocols/chat_completions/chat_spec.rb:402 .render_payload > with schema > uses custom schema name when provided in full format
#[tokio::test]
async fn a_custom_schema_name_is_used() {
    let server = serve(vec![]).await;
    let schema = person_schema(true);
    let payload = render_hello(cc_chat(&server, "openai", "gpt-4o").with_schema(schema.clone()));
    assert_eq!(
        payload["response_format"]["json_schema"]["name"],
        json!("PersonSchema")
    );
    assert_eq!(
        payload["response_format"]["json_schema"]["schema"],
        schema["schema"]
    );
    assert_eq!(
        payload["response_format"]["json_schema"]["strict"],
        json!(true)
    );
}

// spec: protocols/chat_completions/chat_spec.rb:429 .render_payload > with schema > respects explicit strict: false
#[tokio::test]
async fn an_explicit_strict_false_is_respected() {
    let server = serve(vec![]).await;
    let payload =
        render_hello(cc_chat(&server, "openai", "gpt-4o").with_schema(person_schema(false)));
    assert_eq!(
        payload["response_format"]["json_schema"]["strict"],
        json!(false)
    );
}

fn strict_for(server: &MockServer, schema: Value) -> Value {
    render_hello(openai(server).with_schema(schema))["response_format"]["json_schema"]["strict"]
        .clone()
}

// spec: protocols/chat_completions/chat_spec.rb:740 .render_payload with a schema > sends non-strict when a property is optional, since strict mode would reject it
#[tokio::test]
async fn an_optional_property_sends_non_strict() {
    let server = serve(vec![]).await;
    let schema = json!({ "name": "person", "schema": {
        "type": "object", "properties": { "name": { "type": "string" }, "city": { "type": "string" } }, "required": ["name"]
    } });
    assert_eq!(strict_for(&server, schema), json!(false));
}

// spec: protocols/chat_completions/chat_spec.rb:753 .render_payload with a schema > sends non-strict when a nested object has optional properties
#[tokio::test]
async fn a_nested_optional_property_sends_non_strict() {
    let server = serve(vec![]).await;
    let schema = json!({ "name": "person", "schema": {
        "type": "object",
        "properties": { "address": {
            "type": "object", "properties": { "street": { "type": "string" }, "unit": { "type": "string" } }, "required": ["street"]
        } },
        "required": ["address"]
    } });
    assert_eq!(strict_for(&server, schema), json!(false));
}

// spec: protocols/chat_completions/chat_spec.rb:769 .render_payload with a schema > keeps an explicit strict choice
#[tokio::test]
async fn an_explicit_strict_choice_is_kept() {
    let server = serve(vec![]).await;
    let schema = json!({ "name": "person", "schema": { "type": "object", "properties": { "city": { "type": "string" } } }, "strict": true });
    assert_eq!(strict_for(&server, schema), json!(true));
}

// ---- chat_spec.rb: citations --------------------------------------------------------------------

// spec: protocols/chat_completions/chat_spec.rb:475 citations > leaves the cited text nil when the annotation carries no offsets
#[test]
fn a_url_citation_without_offsets_has_no_text() {
    let message = json!({ "role": "assistant", "content": "Hello", "annotations": [{ "url_citation": { "url": "https://a.example" } }] });
    let m = parse(Provider::OpenAI, body("gpt-4.1-nano", message, json!({})));
    assert_eq!(m.citations[0].url.as_deref(), Some("https://a.example"));
    assert_eq!(m.citations[0].text, None);
}

fn root_citations(root: Value) -> Vec<rust_llm::Citation> {
    let mut data = body(
        "sonar",
        json!({ "role": "assistant", "content": null }),
        json!({}),
    );
    for (k, v) in root.as_object().unwrap() {
        data[k] = v.clone();
    }
    parse(Provider::Perplexity, data).citations
}

// spec: protocols/chat_completions/chat_spec.rb:483 citations > falls back to root search results
#[test]
fn root_search_results_are_the_fallback() {
    let citations = root_citations(json!({ "search_results": [
        { "url": "https://a.example", "title": "A", "snippet": "excerpt" },
        "not a result"
    ] }));
    assert_eq!(citations.len(), 1);
    assert_eq!(citations[0].cited_text.as_deref(), Some("excerpt"));
    assert_eq!(citations[0].source_index, Some(0));
}

// spec: protocols/chat_completions/chat_spec.rb:498 citations > falls back to a root citation URL list
#[test]
fn a_root_citation_url_list_is_the_fallback() {
    let citations = root_citations(json!({ "citations": ["https://a.example", 42] }));
    let urls: Vec<Option<&str>> = citations.iter().map(|c| c.url.as_deref()).collect();
    assert_eq!(urls, vec![Some("https://a.example")]);
}

// spec: protocols/chat_completions/chat_spec.rb:504 citations > is empty when the response carries none
#[test]
fn no_citations_is_empty() {
    assert!(root_citations(json!({})).is_empty());
}

// ---- chat_spec.rb: error and usage handling -----------------------------------------------------

// spec: protocols/chat_completions/chat_spec.rb:510 .parse_completion_body error and usage handling > raises the error the provider reported
#[test]
fn the_reported_error_is_raised() {
    let err = parse_err(json!({ "error": { "message": "model overloaded" } }));
    assert_eq!(err.to_string(), "model overloaded");
}

// spec: protocols/chat_completions/chat_spec.rb:520 .parse_completion_body error and usage handling > raises when the response carries no message
#[test]
fn a_response_without_a_message_is_an_error() {
    let err = parse_err(json!({ "choices": [] }));
    assert!(
        err.to_string()
            .contains("Provider returned no completion message"),
        "{err}"
    );
}

// spec: protocols/chat_completions/chat_spec.rb:526 .parse_completion_body error and usage handling > derives generated tokens from the total when the provider omits them
#[test]
fn generated_tokens_are_derived_from_the_total() {
    let data = json!({
        "choices": [{ "message": { "role": "assistant", "content": "Hi" } }],
        "usage": { "prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 14 }
    });
    assert_eq!(parse(Provider::OpenAI, data).tokens.output, Some(4));
}

// ---- chat_spec.rb: thinking round-trips ---------------------------------------------------------

fn reply(message: Value) -> Message {
    let mut message = message;
    message["role"] = "assistant".into();
    parse(Provider::OpenAI, body("gpt-4.1-nano", message, json!({})))
}

fn thinking_text(message: Value) -> Option<String> {
    reply(message).thinking.and_then(|t| t.text)
}

fn thinking_signature(message: Value) -> Option<String> {
    reply(message).thinking.and_then(|t| t.signature)
}

// spec: protocols/chat_completions/chat_spec.rb:540 thinking round-trips > reads reasoning out of the alternate field names
#[test]
fn reasoning_is_read_from_the_alternate_field_names() {
    assert_eq!(
        thinking_text(json!({ "content": "a", "reasoning": "why" })).as_deref(),
        Some("why")
    );
    assert_eq!(
        thinking_text(json!({ "content": "a", "thinking": "why" })).as_deref(),
        Some("why")
    );
    assert_eq!(
        thinking_text(json!({ "content": "a", "reasoning_content": 42 })),
        None
    );
    // `reasoning_content || reasoning || thinking`: a non-string earlier field still wins, then fails the String check.
    assert_eq!(
        thinking_text(json!({ "content": "a", "reasoning_content": 42, "reasoning": "why" })),
        None
    );
    assert_eq!(
        thinking_signature(json!({ "content": "a", "signature": "sig" })).as_deref(),
        Some("sig")
    );
    assert_eq!(
        thinking_signature(json!({ "content": "a", "reasoning_signature": 42 })),
        None
    );
    assert_eq!(
        thinking_signature(
            json!({ "content": "a", "reasoning_signature": 42, "signature": "sig" })
        ),
        None
    );
}

// spec: protocols/chat_completions/chat_spec.rb:548 thinking round-trips > hands string content back untouched, markup and all
#[test]
fn string_content_is_returned_untouched() {
    let plain = reply(json!({ "content": "plain" }));
    assert_eq!(
        (plain.content.as_deref(), plain.thinking),
        (Some("plain"), None)
    );
    let marked = reply(json!({ "content": "<think>why</think>answer" }));
    assert_eq!(
        (marked.content.as_deref(), marked.thinking),
        (Some("<think>why</think>answer"), None)
    );
}

// spec: protocols/chat_completions/chat_spec.rb:554 thinking round-trips > leaves a content shape it does not understand alone
#[test]
fn nil_content_yields_no_content_and_no_thinking() {
    let m = reply(json!({ "content": null }));
    assert_eq!((m.content, m.thinking), (None, None));
}

// spec: protocols/chat_completions/chat_spec.rb:584 thinking round-trips > sends only the signature when that is all the model returned
#[tokio::test]
async fn a_signature_only_thinking_sends_only_the_signature() {
    let server = serve(vec![]).await;
    let mut chat = openai(&server);
    let mut answer = Message::assistant("done");
    answer.thinking = Some(Thinking {
        text: None,
        signature: Some("sig".into()),
    });
    chat.set_messages(vec![Message::user("hi"), answer]);
    let rendered = chat.render().unwrap()["messages"][1].clone();
    assert_eq!(
        rendered,
        json!({ "role": "assistant", "content": "done", "reasoning_signature": "sig" })
    );
}

// ---- chat_spec.rb: prompt caching and max tokens ------------------------------------------------

fn long_context_boundary(chat: &mut Chat) -> Value {
    let mut message = Message::user("Long context");
    message.cache_until_here = true;
    chat.set_messages(vec![message]);
    chat.render().unwrap()
}

// spec: protocols/chat_completions/chat_spec.rb:615 prompt caching > marks cache boundaries without disabling implicit caching
#[tokio::test]
async fn cache_boundaries_are_marked_without_cache_options() {
    let server = serve(vec![]).await;
    let payload = long_context_boundary(&mut cc_chat(&server, "openai", "gpt-5.6"));
    assert_eq!(
        payload["messages"][0]["content"],
        json!([{ "type": "text", "text": "Long context", "prompt_cache_breakpoint": { "mode": "explicit" } }])
    );
    assert!(payload.get("prompt_cache_options").is_none());
    // `openai_prompt_caching?` is true for the whole wire format, not just OpenAI.
    let payload = long_context_boundary(&mut cc_chat(&server, "deepseek", "deepseek-v4-flash"));
    assert_eq!(
        payload["messages"][0]["content"][0]["prompt_cache_breakpoint"],
        json!({ "mode": "explicit" })
    );
}

// spec: protocols/chat_completions/chat_spec.rb:628 prompt caching > preserves cache options alongside explicit boundaries
#[tokio::test]
async fn cache_options_survive_alongside_boundaries() {
    let server = serve(vec![]).await;
    let mut chat = openai(&server)
        .with_caching(json!({ "ttl": "30m" }))
        .unwrap();
    let payload = long_context_boundary(&mut chat);
    assert_eq!(payload["prompt_cache_options"], json!({ "ttl": "30m" }));
    let content = payload["messages"][0]["content"].as_array().unwrap();
    assert_eq!(
        content.last().unwrap()["prompt_cache_breakpoint"],
        json!({ "mode": "explicit" })
    );
}

// spec: protocols/chat_completions/chat_spec.rb:674 #max_output_tokens_field > always sends max_completion_tokens to OpenAI and Azure
// (Azure is a provider the port leaves out; the OpenAI half is asserted.)
#[tokio::test]
async fn openai_always_gets_max_completion_tokens() {
    let server = serve(vec![]).await;
    for id in [
        "gpt-3.5-turbo",
        "gpt-4o-mini",
        "gpt-5.1",
        "o4-mini",
        "ft:gpt-5.1:acme::abc123",
        "prod-reasoner",
    ] {
        let payload = render_hello(cc_chat(&server, "openai", id).with_max_output_tokens(1000));
        let fields: Vec<&String> = payload
            .as_object()
            .unwrap()
            .keys()
            .filter(|k| k.starts_with("max_"))
            .collect();
        assert_eq!(fields, vec!["max_completion_tokens"], "{id}");
    }
}

// ---- media_spec.rb -------------------------------------------------------------------------------

async fn png(resolution: Resolution) -> Attachment {
    let mut image = Attachment::new(format!(
        "{}/tests/fixtures/ruby.png",
        env!("CARGO_MANIFEST_DIR")
    ))
    .with_resolution(resolution);
    image.content().await.unwrap();
    image
}

// spec: protocols/chat_completions/media_spec.rb:60 .format_content > maps low resolution to low image detail
#[tokio::test]
async fn low_resolution_maps_to_low_detail() {
    let server = serve(vec![]).await;
    let message = render_attachments(
        openai(&server),
        "Describe this",
        vec![png(Resolution::Low).await],
    )
    .unwrap();
    assert_eq!(message["content"][1]["image_url"]["detail"], json!("low"));
}

// UPSTREAM-REMOVED in 2.1 (was spec: protocols/chat_completions/media_spec.rb:68) .format_content > maps higher resolutions to high image detail
#[tokio::test]
async fn higher_resolutions_map_to_high_detail() {
    let server = serve(vec![]).await;
    let message = render_attachments(
        openai(&server),
        "Describe this",
        vec![png(Resolution::Medium).await],
    )
    .unwrap();
    assert_eq!(message["content"][1]["image_url"]["detail"], json!("high"));
}

// ---- streaming_spec.rb ---------------------------------------------------------------------------

// spec: protocols/chat_completions/streaming_spec.rb:41 preserves raw finish reasons on chunks
#[test]
fn chunks_carry_the_finish_reason() {
    let chunk = chat_completions::build_chunk(
        Provider::OpenAI,
        &json!({ "model": "gpt-4.1-nano", "choices": [{ "delta": { "content": "" }, "finish_reason": "tool_calls" }] }),
    );
    assert_eq!(chunk.finish_reason, Some(FinishReason::ToolCalls));
}

fn stream_status(data: &str) -> Option<u16> {
    rust_llm::protocols::streaming_error_status(ProtocolName::ChatCompletions)(data)
}

/// The error an OpenAI Chat Completions stream raises when it carries `events`.
async fn stream_error(events: String) -> Error {
    let server = serve_templates(vec![sse(events)]).await;
    openai(&server).ask_stream("hi", |_| {}).await.unwrap_err()
}

// spec: protocols/chat_completions/streaming_spec.rb:59 #parse_streaming_error > parses typed error objects
#[tokio::test]
async fn typed_rate_limit_stream_errors_are_429s() {
    let data = r#"{"error":{"type":"rate_limit_exceeded","message":"Slow down"}}"#;
    assert_eq!(stream_status(data), Some(429));
    let err = stream_error(format!("data: {data}\n\n")).await;
    assert!(
        matches!(&err, Error::RateLimit(m, Some(r)) if m == "Slow down" && r.status == 429),
        "{err:?}"
    );
}

// spec: protocols/chat_completions/streaming_spec.rb:69 #parse_streaming_error > reports a 500 for server errors
#[tokio::test]
async fn server_stream_errors_are_500s() {
    let data = r#"{"error":{"type":"server_error","message":"Internal error"}}"#;
    assert_eq!(stream_status(data), Some(500));
    let err = stream_error(format!("data: {data}\n\n")).await;
    assert!(
        matches!(&err, Error::Server(m, Some(r)) if m == "Internal error" && r.status == 500),
        "{err:?}"
    );
}

// spec: protocols/chat_completions/streaming_spec.rb:89 #parse_streaming_error > falls back to a 400 for other typed error objects
#[tokio::test]
async fn other_typed_stream_errors_are_400s() {
    let data = r#"{"error":{"type":"invalid_request_error","message":"Bad request"}}"#;
    assert_eq!(stream_status(data), Some(400));
    let err = stream_error(format!("data: {data}\n\n")).await;
    assert!(
        matches!(&err, Error::BadRequest(m, Some(r)) if m == "Bad request" && r.status == 400),
        "{err:?}"
    );
}

// spec: protocols/chat_completions/streaming_spec.rb:99 #parse_streaming_error > handles a body that parses to a bare JSON string
#[tokio::test]
async fn a_bare_json_string_stream_error_has_no_status() {
    let data = r#""The model foo is not available in your region (error).""#;
    assert_eq!(stream_status(data), None);
    // No status, so the stream raises the default 500 carrying the string.
    let err = stream_error(format!("event: error\ndata: {data}\n\n")).await;
    assert!(
        matches!(&err, Error::Server(m, Some(r)) if m == "The model foo is not available in your region (error)." && r.status == 500),
        "{err:?}"
    );
}

// spec: protocols/chat_completions/streaming_spec.rb:109 #parse_streaming_error > handles a string error value
#[tokio::test]
async fn a_string_error_value_has_no_status() {
    let data = r#"{"error":"The model foo is not available in your region."}"#;
    assert_eq!(stream_status(data), None);
    let err = stream_error(format!("data: {data}\n\n")).await;
    assert!(
        matches!(&err, Error::Server(m, Some(r)) if m == "The model foo is not available in your region." && r.status == 500),
        "{err:?}"
    );
}

// spec: protocols/chat_completions/streaming_spec.rb:120 surfaces the provider message for a failed streaming response with a string error value
#[tokio::test]
async fn a_failed_stream_with_a_string_error_surfaces_the_message() {
    let server =
        serve_templates(vec![ResponseTemplate::new(404).set_body_string(
            r#"{"error": "The model foo is not available in your region."}"#,
        )])
        .await;
    let err = openai(&server).ask_stream("hi", |_| {}).await.unwrap_err();
    assert!(
        err.to_string().contains("not available in your region"),
        "{err:?}"
    );
}

// ---- tools_spec.rb -------------------------------------------------------------------------------

/// `parse_tool_calls(tool_calls)` through a completion body.
fn parsed_calls(tool_calls: Value) -> Option<Vec<ToolCall>> {
    let mut message = json!({ "role": "assistant", "content": "" });
    if !tool_calls.is_null() {
        message["tool_calls"] = tool_calls;
    }
    parse(Provider::OpenAI, body("gpt-4.1-nano", message, json!({})))
        .tool_calls
        .map(|c| c.values().cloned().collect())
}

/// `extract_tool_call_thought_signature(tool_call)` for a call carrying `extra`.
fn signature_of(extra: Value) -> Option<String> {
    let mut call = json!({ "id": "call_1", "function": { "name": "weather", "arguments": "{}" } });
    if !extra.is_null() {
        call["extra_content"] = extra;
    }
    parsed_calls(json!([call])).unwrap()[0]
        .thought_signature
        .clone()
}

// spec: protocols/chat_completions/tools_spec.rb:27 .parse_tool_calls > extracts thought signatures from extra_content.google.thought_signature
#[test]
fn thought_signatures_are_parsed_from_extra_content() {
    let calls = parsed_calls(json!([{
        "id": "call_456",
        "function": { "name": "weather", "arguments": "{\"location\":\"Paris\"}" },
        "extra_content": { "google": { "thought_signature": "sig_abc123" } }
    }]))
    .unwrap();
    assert_eq!(calls[0].id, "call_456");
    assert_eq!(calls[0].thought_signature.as_deref(), Some("sig_abc123"));
}

// spec: protocols/chat_completions/tools_spec.rb:48 .parse_tool_calls > handles multiple tool calls with thought signatures
#[test]
fn each_parsed_call_keeps_its_own_signature() {
    let calls = parsed_calls(json!([
        { "id": "call_1", "function": { "name": "tool_a", "arguments": "{}" }, "extra_content": { "google": { "thought_signature": "sig_first" } } },
        { "id": "call_2", "function": { "name": "tool_b", "arguments": "{}" } }
    ]))
    .unwrap();
    assert_eq!(
        (calls[0].id.as_str(), calls[0].thought_signature.as_deref()),
        ("call_1", Some("sig_first"))
    );
    assert_eq!(
        (calls[1].id.as_str(), calls[1].thought_signature.as_deref()),
        ("call_2", None)
    );
}

// spec: protocols/chat_completions/tools_spec.rb:67 .parse_tool_calls > returns nil for empty or nil input
#[test]
fn empty_or_missing_tool_calls_parse_to_none() {
    assert!(parsed_calls(Value::Null).is_none());
    assert!(parsed_calls(json!([])).is_none());
}

/// `format_tool_calls(tool_calls)`: the tool_calls of a rendered assistant turn.
async fn formatted_calls(calls: Vec<ToolCall>) -> Value {
    let server = serve(vec![]).await;
    let mut chat = openai(&server);
    let mut turn = tool_call_message(&[]);
    turn.tool_calls = Some(calls.into_iter().map(|c| (c.id.clone(), c)).collect());
    chat.set_messages(vec![Message::user("Go"), turn]);
    chat.render().unwrap()["messages"][1]["tool_calls"].clone()
}

fn signed_call(id: &str, name: &str, arguments: Value, signature: Option<&str>) -> ToolCall {
    let mut call = ToolCall::new(id, name, arguments.as_object().cloned().unwrap_or_default());
    call.thought_signature = signature.map(str::to_string);
    call
}

// spec: protocols/chat_completions/tools_spec.rb:122 .format_tool_calls > includes extra_content.google.thought_signature when present
#[tokio::test]
async fn formatted_calls_carry_their_thought_signature() {
    let calls = formatted_calls(vec![signed_call(
        "call_456",
        "weather",
        json!({ "location": "Paris" }),
        Some("sig_xyz789"),
    )])
    .await;
    assert_eq!(
        calls,
        json!([{
            "id": "call_456",
            "type": "function",
            "function": { "name": "weather", "arguments": "{\"location\":\"Paris\"}" },
            "extra_content": { "google": { "thought_signature": "sig_xyz789" } }
        }])
    );
}

// spec: protocols/chat_completions/tools_spec.rb:147 .format_tool_calls > formats multiple tool calls preserving their signatures
#[tokio::test]
async fn each_formatted_call_keeps_its_own_signature() {
    let calls = formatted_calls(vec![
        signed_call("call_1", "tool_a", json!({}), Some("sig_first")),
        signed_call("call_2", "tool_b", json!({}), None),
    ])
    .await;
    let find = |id: &str| {
        calls
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["id"] == id)
            .unwrap()
            .clone()
    };
    assert_eq!(
        find("call_1")["extra_content"],
        json!({ "google": { "thought_signature": "sig_first" } })
    );
    assert!(find("call_2").get("extra_content").is_none());
}

// spec: protocols/chat_completions/tools_spec.rb:178 .extract_tool_call_thought_signature > extracts signature from nested structure
#[test]
fn the_nested_thought_signature_is_extracted() {
    assert_eq!(
        signature_of(json!({ "google": { "thought_signature": "test_sig" } })).as_deref(),
        Some("test_sig")
    );
}

// spec: protocols/chat_completions/tools_spec.rb:191 .extract_tool_call_thought_signature > returns nil when extra_content is missing
#[test]
fn no_extra_content_means_no_signature() {
    assert_eq!(signature_of(Value::Null), None);
}

// spec: protocols/chat_completions/tools_spec.rb:198 .extract_tool_call_thought_signature > returns nil when google key is missing
#[test]
fn no_google_key_means_no_signature() {
    assert_eq!(signature_of(json!({})), None);
}

// spec: protocols/chat_completions/tools_spec.rb:205 .extract_tool_call_thought_signature > returns nil when thought_signature is missing
#[test]
fn no_thought_signature_key_means_no_signature() {
    assert_eq!(signature_of(json!({ "google": {} })), None);
}
