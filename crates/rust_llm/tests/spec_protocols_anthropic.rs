//! Anthropic protocol specs ported from RubyLLM 2.0: `protocols/anthropic/{chat,
//! chat_thinking_replay,media,streaming,tools}_spec.rb` and `protocols/anthropic_compaction_spec.rb`.
//! Ruby calls the protocol's private helpers (`format_message`, `parse_completion_body`,
//! `inject_cache_control`); these read the rendered payload, the parsed reply, or the chunk the
//! public `build_chunk` returns, which is what those helpers produce. `// spec:` lines tie each test
//! to its Ruby example.

mod spec_helpers;

use rust_llm::files::UploadedFile;
use rust_llm::message::RawResponse;
use rust_llm::protocols::anthropic::{self, StreamBlocks};
use rust_llm::{
    Attachment, Chat, Error, FinishReason, Message, ProtocolName, Role, Thinking, ThinkingConfig,
    ThinkingDisplay,
};
use serde_json::{Value, json};
use spec_helpers::*;

fn fixture(name: &str) -> String {
    format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
}

/// A fixture attachment with its bytes read, as `Attachment.new(path)` reads them lazily.
async fn loaded(name: &str) -> Attachment {
    let mut a = Attachment::new(fixture(name));
    a.content().await.unwrap();
    a
}

/// `chat.render` over `messages`.
fn render(chat: &mut Chat, messages: Vec<Message>) -> Value {
    chat.set_messages(messages);
    chat.render().unwrap()
}

fn tool_result(id: &str, content: &str) -> Message {
    let mut m = Message::new(Role::Tool, Some(content.to_string()));
    m.tool_call_id = Some(id.into());
    m
}

fn body(content: Value) -> Value {
    json!({ "model": MODEL, "content": content, "stop_reason": "end_turn", "usage": {} })
}

fn parse(data: Value) -> Message {
    anthropic::parse_completion_body(&data, RawResponse::default()).unwrap()
}

fn sse_events(events: &[Value]) -> String {
    events
        .iter()
        .map(|e| {
            format!(
                "event: {}\ndata: {e}\n\n",
                e["type"].as_str().unwrap_or("message")
            )
        })
        .collect()
}

// ---- chat_spec.rb ------------------------------------------------------------------------------

// spec: protocols/anthropic/chat_spec.rb:7 normalizes stop_reason into finish_reason
#[tokio::test]
async fn stop_reason_max_tokens_normalizes_to_max_tokens() {
    let mut response = text_response("Hello");
    response["stop_reason"] = "max_tokens".into();
    let server = serve(vec![response]).await;
    let reply = chat(&server).ask("hi").await.unwrap();
    assert_eq!(reply.finish_reason, Some(FinishReason::MaxTokens));
}

// spec: protocols/anthropic/chat_spec.rb:26 accepts URL schemes regardless of case
#[tokio::test]
async fn citation_url_scheme_is_case_insensitive() {
    let mut response = text_response("Cited.");
    response["content"][0]["citations"] = json!([{ "url": "HTTPS://example.com/source" }]);
    let server = serve(vec![response]).await;
    let reply = chat(&server).ask("hi").await.unwrap();
    assert_eq!(
        reply.citations[0].url.as_deref(),
        Some("HTTPS://example.com/source")
    );
}

// spec: protocols/anthropic/chat_spec.rb:50 returns both text blocks when multiple :system messages are passed
#[tokio::test]
async fn each_system_message_becomes_its_own_block() {
    let server = serve(vec![]).await;
    let payload = render(
        &mut chat(&server),
        vec![
            Message::system("Static prompt."),
            Message::system("Per-session context."),
            Message::user("Hi"),
        ],
    );
    assert_eq!(
        payload["system"],
        json!([{ "type": "text", "text": "Static prompt." }, { "type": "text", "text": "Per-session context." }])
    );
}

// spec: protocols/anthropic/chat_spec.rb:99 drops a turn that rendered no content blocks
#[tokio::test]
async fn a_turn_with_no_content_blocks_is_dropped() {
    let server = serve(vec![]).await;
    let payload = render(
        &mut chat(&server),
        vec![
            Message::user("Hello"),
            Message::assistant(""),
            Message::user("Still there?"),
        ],
    );
    assert_eq!(
        payload["messages"],
        json!([
            { "role": "user", "content": [{ "type": "text", "text": "Hello" }] },
            { "role": "user", "content": [{ "type": "text", "text": "Still there?" }] }
        ])
    );
}

// spec: protocols/anthropic/chat_spec.rb:114 adds cache_control to a tool result marked as a cache boundary
#[tokio::test]
async fn a_tool_result_cache_boundary_carries_cache_control() {
    let server = serve(vec![]).await;
    let mut result = tool_result("tool_1", "result");
    result.cache_until_here = true;
    let mut chat = chat(&server).with_caching(json!({ "ttl": "1h" })).unwrap();
    let payload = render(
        &mut chat,
        vec![
            Message::user("Go"),
            tool_call_message(&[("tool_1", "lookup", json!({}))]),
            result,
        ],
    );
    let last = payload["messages"]
        .as_array()
        .unwrap()
        .last()
        .unwrap()
        .clone();
    assert_eq!(
        last["content"].as_array().unwrap().last().unwrap()["cache_control"],
        json!({ "type": "ephemeral", "ttl": "1h" })
    );
}

// spec: protocols/anthropic/chat_spec.rb:140 formats attachments before tool calls
// spec: protocols/anthropic/tools_spec.rb:94 formats attachments before tool calls
#[tokio::test]
async fn attachments_render_before_tool_calls() {
    let server = serve(vec![]).await;
    let mut call = tool_call_message(&[("tool_123", "test_tool", json!({ "arg1": "value1" }))]);
    call.content = Some("Read this before calling the tool".into());
    call.attachments = vec![loaded("ruby.txt").await];
    let payload = render(&mut chat(&server), vec![Message::user("Go"), call]);
    let content = payload["messages"][1]["content"]
        .as_array()
        .unwrap()
        .clone();
    assert_eq!(
        content[0],
        json!({ "type": "text", "text": "Read this before calling the tool" })
    );
    assert_eq!(content[1]["type"], "text");
    assert!(
        content[1]["text"]
            .as_str()
            .unwrap()
            .contains("<file name='ruby.txt' mime_type='text/plain'>")
    );
    assert_eq!(
        (content[2]["type"].as_str(), content[2]["id"].as_str()),
        (Some("tool_use"), Some("tool_123"))
    );
}

// spec: protocols/anthropic/chat_spec.rb:173 adds cache_control to a user message marked as a cache boundary
#[tokio::test]
async fn a_user_cache_boundary_carries_cache_control() {
    let server = serve(vec![]).await;
    let mut chat = chat(&server);
    chat.ask_later("Long context").unwrap();
    chat.cache_until_here().unwrap();
    let payload = chat.render().unwrap();
    assert_eq!(
        payload["messages"][0]["content"]
            .as_array()
            .unwrap()
            .last()
            .unwrap()["cache_control"],
        json!({ "type": "ephemeral" })
    );
}

// spec: protocols/anthropic/chat_spec.rb:294 strips strict key from schema
#[tokio::test]
async fn schema_strict_is_stripped_from_output_config() {
    let server = serve(vec![]).await;
    let mut chat = chat(&server).with_schema(json!({
        "name": "response",
        "schema": { "type": "object", "strict": true, "properties": { "name": { "type": "string" } } },
        "strict": true
    }));
    chat.ask_later("Hello").unwrap();
    let schema = chat.render().unwrap()["output_config"]["format"]["schema"].clone();
    assert!(schema.is_object());
    assert!(schema.get("strict").is_none());
}

// ---- chat_spec.rb: render_payload with thinking --------------------------------------------------

/// `render_payload(model_id:, thinking:)`: the Ruby spec's model has no max_output_tokens, so
/// max_tokens falls back to DEFAULT_MAX_OUTPUT_TOKENS (4096). The registry entries carry the same
/// reasoning_options the spec passes.
fn thinking_payload(model: &str, thinking: ThinkingConfig) -> Value {
    thinking_payload_with(
        model,
        thinking,
        Some(anthropic::DEFAULT_MAX_OUTPUT_TOKENS),
        None,
    )
}

fn thinking_payload_with(
    model: &str,
    thinking: ThinkingConfig,
    max: Option<i64>,
    schema: Option<Value>,
) -> Value {
    let mut chat = Chat::with_config(render_config(), Some(model), Some("anthropic"), false)
        .unwrap()
        .with_thinking(thinking)
        .with_max_output_tokens(max);
    if let Some(schema) = schema {
        chat = chat.with_schema(schema);
    }
    chat.ask_later("Hello").unwrap();
    chat.render().unwrap()
}

/// Rendering never sends, so any configured base will do.
fn render_config() -> std::sync::Arc<rust_llm::Config> {
    let mut c = rust_llm::Config::default();
    c.set("anthropic_api_key", "test");
    std::sync::Arc::new(c)
}

// spec: protocols/anthropic/chat_spec.rb:438 turns on adaptive thinking beside effort on generations without a budget
#[test]
fn effort_without_a_budget_option_thinks_adaptively() {
    let p = thinking_payload("claude-opus-4-7", ThinkingConfig::effort("xhigh"));
    assert_eq!(p["thinking"], json!({ "type": "adaptive" }));
    assert_eq!(p["output_config"], json!({ "effort": "xhigh" }));
}

// spec: protocols/anthropic/chat_spec.rb:449 sizes a budget from the effort on generations that take one
#[test]
fn effort_sizes_a_budget_on_generations_that_take_one() {
    let p = thinking_payload("claude-opus-4-5", ThinkingConfig::effort("medium"));
    assert_eq!(
        p["thinking"],
        json!({ "type": "enabled", "budget_tokens": 4095 })
    );
    assert_eq!(p["output_config"], json!({ "effort": "medium" }));
}

// spec: protocols/anthropic/chat_spec.rb:460 keeps the effort budget under max_tokens and above the minimum
#[test]
fn effort_budget_stays_above_the_minimum() {
    let p = thinking_payload("claude-sonnet-4-5", ThinkingConfig::effort("low"));
    assert_eq!(
        p["thinking"],
        json!({ "type": "enabled", "budget_tokens": 1024 })
    );
    assert_eq!(p["output_config"], json!({ "effort": "low" }));
}

// spec: protocols/anthropic/chat_spec.rb:471 keeps the effort budget under the max_output_tokens of the request
#[test]
fn effort_budget_stays_under_the_request_max_output_tokens() {
    let p = thinking_payload_with(
        "claude-opus-4-5",
        ThinkingConfig::effort("medium"),
        Some(4000),
        None,
    );
    assert_eq!(p["max_tokens"], json!(4000));
    assert_eq!(
        p["thinking"],
        json!({ "type": "enabled", "budget_tokens": 3999 })
    );
}

// spec: protocols/anthropic/chat_spec.rb:492 resolves a bare with_thinking to a request Claude honors
#[test]
fn bare_with_thinking_resolves_to_adaptive_medium() {
    let p = thinking_payload("claude-opus-4-8", ThinkingConfig::on());
    assert_eq!(p["thinking"], json!({ "type": "adaptive" }));
    assert_eq!(p["output_config"], json!({ "effort": "medium" }));
}

// spec: protocols/anthropic/chat_spec.rb:507 sends a budget the registry does not advertise
#[test]
fn a_budget_goes_out_even_when_the_registry_lists_only_effort() {
    let p = thinking_payload("claude-opus-4-7", ThinkingConfig::budget(2048));
    assert_eq!(
        p["thinking"],
        json!({ "type": "enabled", "budget_tokens": 2048 })
    );
}

// spec: protocols/anthropic/chat_spec.rb:517 sends effort and budget side by side
#[test]
fn effort_and_budget_go_side_by_side() {
    let mut thinking = ThinkingConfig::effort("high");
    thinking.budget = Some(4096);
    let p = thinking_payload("claude-opus-4-5", thinking);
    assert_eq!(
        p["thinking"],
        json!({ "type": "enabled", "budget_tokens": 4096 })
    );
    assert_eq!(p["output_config"], json!({ "effort": "high" }));
}

// spec: protocols/anthropic/chat_spec.rb:539 carries a display on enabled thinking when a budget is set
#[test]
fn a_display_rides_on_budgeted_thinking() {
    let p = thinking_payload(
        "claude-sonnet-4-6",
        ThinkingConfig::budget(4096).with_display(ThinkingDisplay::Summarized),
    );
    assert_eq!(
        p["thinking"],
        json!({ "type": "enabled", "budget_tokens": 4096, "display": "summarized" })
    );
}

// spec: protocols/anthropic/chat_spec.rb:549 merges thinking effort with schema output_config
#[test]
fn effort_merges_with_the_schema_output_config() {
    let schema = json!({ "name": "response", "schema": { "type": "object", "properties": { "name": { "type": "string" } } } });
    let p = thinking_payload_with(
        "claude-opus-4-7",
        ThinkingConfig::effort("high"),
        Some(4096),
        Some(schema),
    );
    assert_eq!(
        p["output_config"],
        json!({
            "effort": "high",
            "format": { "type": "json_schema", "schema": { "type": "object", "properties": { "name": { "type": "string" } } } }
        })
    );
}

// spec: protocols/anthropic/chat_spec.rb:568 omits thinking when effort is none
#[test]
fn effort_none_omits_thinking() {
    let p = thinking_payload("claude-opus-4-7", ThinkingConfig::effort("none"));
    assert!(p.get("thinking").is_none());
    assert!(p.get("output_config").is_none());
}

/// `RubyLLM::Model.new(id: 'claude-3-haiku', provider: 'anthropic')`: no registry entry, no
/// reasoning options.
fn haiku3(thinking: Option<ThinkingConfig>) -> Value {
    let mut chat = Chat::with_config(
        render_config(),
        Some("claude-3-haiku"),
        Some("anthropic"),
        true,
    )
    .unwrap();
    if let Some(t) = thinking {
        chat = chat.with_thinking(t);
    }
    chat.ask_later("Hello").unwrap();
    chat.render().unwrap()
}

// spec: protocols/anthropic/chat_spec.rb:822 is nil when thinking is off or explicitly none
#[test]
fn no_thinking_fields_when_thinking_is_off_or_none() {
    for p in [haiku3(None), haiku3(Some(ThinkingConfig::effort("none")))] {
        assert!(
            p.get("thinking").is_none() && p.get("output_config").is_none(),
            "{p}"
        );
    }
}

// spec: protocols/anthropic/chat_spec.rb:827 sends effort alone when the registry lists no thinking controls
#[test]
fn effort_goes_alone_when_the_registry_lists_no_controls() {
    let p = haiku3(Some(ThinkingConfig::effort("high")));
    assert!(p.get("thinking").is_none());
    assert_eq!(p["output_config"], json!({ "effort": "high" }));
}

// ---- chat_spec.rb: thinking blocks ---------------------------------------------------------------

/// `claude_answer`'s usage entry: the answer Anthropic produced, so its signature is Claude's own.
fn by_claude(mut m: Message) -> Message {
    m.usage_entries = vec![rust_llm::UsageEntry {
        id: rust_llm::UsageEntry::next_id(),
        owner: None,
        operation: rust_llm::message::Operation::Chat,
        provider: "anthropic".into(),
        model: MODEL.into(),
        status: rust_llm::UsageStatus::Succeeded,
        tokens: Default::default(),
        cost: Default::default(),
    }];
    m
}

// spec: protocols/anthropic/chat_spec.rb:745 replays a stored thinking block even when the request asks for no thinking
#[tokio::test]
async fn stored_thinking_replays_on_a_tool_turn_without_thinking_config() {
    let server = serve(vec![]).await;
    let mut call = tool_call_message(&[("toolu_1", "weather", json!({}))]);
    call.thinking = Some(Thinking {
        text: Some("why".into()),
        signature: Some("sig".into()),
    });
    let payload = render(
        &mut chat(&server),
        vec![Message::user("Weather?"), by_claude(call)],
    );
    let types: Vec<&str> = payload["messages"][1]["content"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|b| b["type"].as_str())
        .collect();
    assert_eq!(types, ["thinking", "tool_use"]);
}

// spec: protocols/anthropic/chat_spec.rb:756 keeps a display-omitted thinking block as thinking, not redacted data
#[tokio::test]
async fn display_omitted_thinking_stays_thinking() {
    let reply = parse(body(
        json!([{ "type": "thinking", "thinking": "", "signature": "sig" }, { "type": "text", "text": "hi" }]),
    ));
    assert_eq!(
        reply.thinking,
        Some(Thinking {
            text: Some(String::new()),
            signature: Some("sig".into())
        })
    );
    // `build_thinking_block(thinking)`: rendered from the thinking alone, without the raw blocks.
    let server = serve(vec![]).await;
    let mut replay = Message::assistant("hi");
    replay.thinking = reply.thinking.clone();
    let payload = render(
        &mut chat(&server),
        vec![Message::user("Hi"), by_claude(replay)],
    );
    assert_eq!(
        payload["messages"][1]["content"][0],
        json!({ "type": "thinking", "thinking": "", "signature": "sig" })
    );
}

/// A server-tool assistant turn replayed verbatim from `raw_content`, marked as a cache boundary.
fn raw_boundary(raw: Value) -> Message {
    let mut m = Message::assistant("");
    m.raw_content = Some(raw);
    m.cache_until_here = true;
    m
}

// spec: protocols/anthropic/chat_spec.rb:772 leaves empty blocks alone
#[tokio::test]
async fn cache_control_leaves_empty_blocks_alone() {
    let server = serve(vec![]).await;
    let payload = render(
        &mut chat(&server),
        vec![Message::user("Hi"), raw_boundary(json!([]))],
    );
    assert_eq!(
        payload["messages"],
        json!([{ "role": "user", "content": [{ "type": "text", "text": "Hi" }] }])
    );
}

// spec: protocols/anthropic/chat_spec.rb:776 leaves a block that already carries cache_control alone
#[tokio::test]
async fn cache_control_keeps_an_existing_cache_control() {
    let server = serve(vec![]).await;
    let blocks =
        json!([{ "type": "text", "text": "hi", "cache_control": { "type": "ephemeral" } }]);
    let mut chat = chat(&server).with_caching(json!({ "ttl": "1h" })).unwrap();
    let payload = render(
        &mut chat,
        vec![Message::user("Hi"), raw_boundary(blocks.clone())],
    );
    assert_eq!(payload["messages"][1]["content"], blocks);
}

// spec: protocols/anthropic/chat_spec.rb:782 leaves a trailing block it cannot annotate alone
#[tokio::test]
async fn cache_control_leaves_a_non_object_block_alone() {
    let server = serve(vec![]).await;
    let payload = render(
        &mut chat(&server),
        vec![Message::user("Hi"), raw_boundary(json!(["plain"]))],
    );
    assert_eq!(payload["messages"][1]["content"], json!(["plain"]));
}

// spec: protocols/anthropic/chat_spec.rb:796 reads the data field off a redacted thinking block
#[test]
fn the_signature_comes_from_redacted_thinking_data() {
    let reply = parse(body(
        json!([{ "type": "redacted_thinking", "data": "blob" }]),
    ));
    assert_eq!(
        reply.thinking,
        Some(Thinking {
            text: None,
            signature: Some("blob".into())
        })
    );
}

// spec: protocols/anthropic/chat_spec.rb:802 is nil when no block carries thinking
#[test]
fn no_thinking_without_thinking_blocks() {
    let reply = parse(body(json!([{ "type": "text" }])));
    assert_eq!(reply.thinking, None);
}

// spec: protocols/anthropic/chat_spec.rb:807 falls back to the text field of a thinking block
#[test]
fn thinking_text_falls_back_to_the_text_field() {
    let reply = parse(body(json!([{ "type": "thinking", "text": "why" }])));
    assert_eq!(reply.thinking.and_then(|t| t.text).as_deref(), Some("why"));
}

// ---- chat_thinking_replay_spec.rb ----------------------------------------------------------------

fn thinking_sequence() -> Vec<Value> {
    vec![
        json!({ "type": "thinking", "thinking": "First.", "signature": "sig-one" }),
        json!({ "type": "redacted_thinking", "data": "encrypted" }),
        json!({ "type": "thinking", "thinking": "", "signature": "sig-two" }),
    ]
}

/// The content blocks the reply replays as on the next turn.
fn replayed(chat: &mut Chat) -> Value {
    chat.ask_later("next").unwrap();
    chat.render().unwrap()["messages"][1]["content"].clone()
}

fn with_blocks(content: Vec<Value>, stop_reason: &str) -> Value {
    let mut r = text_response("");
    r["content"] = Value::Array(content);
    r["stop_reason"] = stop_reason.into();
    r
}

// spec: protocols/anthropic/chat_thinking_replay_spec.rb:24 replays every signed and redacted block in order
#[tokio::test]
async fn signed_and_redacted_blocks_replay_in_order() {
    let mut content = thinking_sequence();
    content.push(json!({ "type": "text", "text": "Done." }));
    let server = serve(vec![with_blocks(content.clone(), "end_turn")]).await;
    let mut chat = chat(&server);
    let reply = chat.ask("hi").await.unwrap();
    assert_eq!(
        reply.thinking,
        Some(Thinking {
            text: Some("First.".into()),
            signature: Some("sig-one".into())
        })
    );
    assert_eq!(replayed(&mut chat), Value::Array(content));
}

// spec: protocols/anthropic/chat_thinking_replay_spec.rb:33 keeps the complete thinking sequence before local tool calls
#[tokio::test]
async fn the_thinking_sequence_replays_before_tool_calls() {
    let mut content = thinking_sequence();
    content.push(json!({ "type": "tool_use", "id": "call-1", "name": "lookup", "input": {} }));
    let server = serve(vec![with_blocks(content.clone(), "tool_use")]).await;
    let mut chat = chat(&server);
    chat.ask_later("hi").unwrap();
    chat.step().await.unwrap();
    assert_eq!(
        chat.render().unwrap()["messages"][1]["content"],
        Value::Array(content)
    );
}

// spec: protocols/anthropic/chat_thinking_replay_spec.rb:39 replays thinking assembled from multiple streamed blocks
#[tokio::test]
async fn streamed_thinking_blocks_replay_as_received() {
    let blocks = thinking_sequence();
    let mut events = vec![json!({ "type": "message_start", "message": { "model": MODEL } })];
    for (index, block) in blocks.iter().enumerate() {
        let thinking = block["type"] == "thinking";
        let start = if thinking {
            json!({ "type": "thinking", "thinking": "" })
        } else {
            block.clone()
        };
        events
            .push(json!({ "type": "content_block_start", "index": index, "content_block": start }));
        if thinking {
            events.push(json!({ "type": "content_block_delta", "index": index, "delta": { "type": "thinking_delta", "thinking": block["thinking"] } }));
            events.push(json!({ "type": "content_block_delta", "index": index, "delta": { "type": "signature_delta", "signature": block["signature"] } }));
        }
        events.push(json!({ "type": "content_block_stop", "index": index }));
    }
    events.push(json!({ "type": "message_stop" }));
    let server = serve_templates(vec![sse(sse_events(&events))]).await;
    let mut chat = chat(&server);
    chat.ask_stream("hi", |_| {}).await.unwrap();
    assert_eq!(replayed(&mut chat), Value::Array(blocks));
}

// spec: protocols/anthropic/chat_thinking_replay_spec.rb:59 does not replay another protocol's native thinking blocks
#[tokio::test]
async fn another_protocols_raw_reasoning_is_not_replayed() {
    let server = serve(vec![]).await;
    let mut done = Message::assistant("Done.");
    done.raw_reasoning = Some(json!({ "converse": [{ "reasoningContent": {} }] }));
    let payload = render(&mut chat(&server), vec![Message::user("Hi"), done]);
    assert_eq!(
        payload["messages"][1]["content"],
        json!([{ "type": "text", "text": "Done." }])
    );
}

// spec: protocols/anthropic/chat_thinking_replay_spec.rb:66 clears the previous stream's thinking before a new response
#[test]
fn message_start_clears_the_previous_streams_blocks() {
    let mut state = StreamBlocks::default();
    anthropic::build_chunk(
        &mut state,
        &json!({ "type": "content_block_start", "index": 0, "content_block": thinking_sequence()[0] }),
    );
    anthropic::build_chunk(&mut state, &json!({ "type": "message_start" }));
    assert_eq!(
        anthropic::build_chunk(&mut state, &json!({ "type": "message_stop" })).raw_reasoning,
        None
    );
}

// spec: protocols/anthropic/chat_thinking_replay_spec.rb:73 retains the thinking in the final segment of a paused server-tool turn
#[tokio::test]
async fn a_paused_turn_keeps_the_final_segments_thinking() {
    let blocks = thinking_sequence();
    let first = vec![
        blocks[0].clone(),
        json!({ "type": "server_tool_use", "id": "server-1", "name": "web_search", "input": {} }),
    ];
    let mut last: Vec<Value> = blocks[1..].to_vec();
    last.push(json!({ "type": "text", "text": "Done." }));
    let server = serve(vec![
        with_blocks(first.clone(), "pause_turn"),
        with_blocks(last.clone(), "end_turn"),
    ])
    .await;
    let mut chat = chat(&server);
    chat.ask("hi").await.unwrap();
    assert_eq!(replayed(&mut chat), Value::Array([first, last].concat()));
}

// ---- media_spec.rb -------------------------------------------------------------------------------

// spec: protocols/anthropic/media_spec.rb:22 skips empty content rather than rendering an empty text block
#[tokio::test]
async fn empty_text_is_skipped_beside_attachments() {
    let pdf = loaded("sample.pdf").await;
    let blocks = anthropic::format_content(Some(""), &[pdf]).unwrap();
    assert_eq!(
        blocks
            .iter()
            .filter_map(|b| b["type"].as_str())
            .collect::<Vec<_>>(),
        ["document"]
    );
    assert!(anthropic::format_content(Some(""), &[]).unwrap().is_empty());
}

// spec: protocols/anthropic/media_spec.rb:40 formats provider-managed PDFs as document file sources
#[tokio::test]
async fn a_provider_managed_pdf_is_a_citable_document_file_source() {
    let file = UploadedFile {
        id: "file_123".into(),
        provider: "anthropic".into(),
        filename: Some("proposal.pdf".into()),
        byte_size: None,
        created_at: None,
        expires_at: None,
        status: None,
        mime_type: Some("application/pdf".into()),
        purpose: None,
        uri: None,
        downloadable: None,
        metadata: Value::Null,
    };
    let server = serve(vec![]).await;
    let mut chat = chat(&server).with_citations(true);
    chat.ask_later_with("Summarize this", vec![Attachment::from_uploaded(file)])
        .unwrap();
    assert_eq!(
        chat.render().unwrap()["messages"][0]["content"][1],
        json!({
            "type": "document",
            "source": { "type": "file", "file_id": "file_123" },
            "title": "proposal.pdf",
            "citations": { "enabled": true }
        })
    );
}

// ---- streaming_spec.rb ---------------------------------------------------------------------------

// spec: protocols/anthropic/streaming_spec.rb:14 preserves raw stop_reason from message_delta events
#[test]
fn message_delta_stop_reason_reaches_the_chunk() {
    let chunk = anthropic::build_chunk(
        &mut StreamBlocks::default(),
        &json!({ "type": "message_delta", "delta": { "stop_reason": "end_turn" }, "usage": { "output_tokens": 10 } }),
    );
    assert_eq!(chunk.finish_reason, Some(FinishReason::Stop));
}

// spec: protocols/anthropic/streaming_spec.rb:27 reads thinking token counts from message_delta usage
#[test]
fn message_delta_usage_carries_thinking_tokens() {
    let chunk = anthropic::build_chunk(
        &mut StreamBlocks::default(),
        &json!({
            "type": "message_delta", "delta": { "stop_reason": "end_turn" },
            "usage": { "output_tokens": 10, "output_tokens_details": { "thinking_tokens": 7 } }
        }),
    );
    assert_eq!(chunk.tokens.thinking, Some(7));
}

// spec: protocols/anthropic/streaming_spec.rb:72 sends Accept-Encoding: identity on streaming requests
#[tokio::test]
async fn streaming_requests_ask_for_identity_encoding() {
    let server = serve_templates(vec![sse(text_stream(&["hi"]))]).await;
    chat(&server).ask_stream("hi", |_| {}).await.unwrap();
    let sent = &server.received_requests().await.unwrap()[0];
    assert_eq!(sent.headers.get("accept-encoding").unwrap(), "identity");
}

async fn stream_error(data: &str) -> Error {
    let server = serve_templates(vec![sse(format!("event: error\ndata: {data}\n\n"))]).await;
    chat(&server).ask_stream("hi", |_| {}).await.unwrap_err()
}

// UPSTREAM-REMOVED in 2.1 (was spec: protocols/anthropic/streaming_spec.rb:68) falls back to a 500 for other typed error objects
// 2.1 maps each documented type to its status (`ERROR_STATUSES`), so a typed invalid request is a 400.
#[tokio::test]
async fn other_typed_stream_errors_are_server_errors() {
    let err = stream_error(
        r#"{"type":"error","error":{"type":"invalid_request_error","message":"Bad request"}}"#,
    )
    .await;
    assert!(
        matches!(&err, Error::BadRequest(m, Some(r)) if m == "Bad request" && r.status == 400),
        "{err:?}"
    );
}

// spec: protocols/anthropic/streaming_spec.rb:124 handles a string error value
#[tokio::test]
async fn a_string_stream_error_is_a_server_error() {
    let err = stream_error(r#"{"type":"error","error":"Overloaded"}"#).await;
    assert!(
        matches!(&err, Error::Server(m, Some(r)) if m == "Overloaded" && r.status == 500),
        "{err:?}"
    );
}

// spec: protocols/anthropic/streaming_spec.rb:134 ignores a body that parses to a bare JSON string
// (`parse_streaming_error` is nil; the error is raised as the stream's default 500 with the string.)
#[tokio::test]
async fn a_bare_json_string_stream_error_has_no_parsed_status() {
    let status = rust_llm::protocols::streaming_error_status(ProtocolName::Anthropic);
    assert_eq!(status(r#""model unavailable (type: error)""#), None);
    let err = stream_error(r#""model unavailable (type: error)""#).await;
    assert!(
        matches!(&err, Error::Server(m, Some(r)) if m == "model unavailable (type: error)" && r.status == 500),
        "{err:?}"
    );
}

// ---- tools_spec.rb -------------------------------------------------------------------------------

// spec: protocols/anthropic/tools_spec.rb:181 uses a placeholder when the tool returns no content
#[tokio::test]
async fn an_empty_tool_result_renders_a_placeholder() {
    let server = serve(vec![]).await;
    let payload = render(
        &mut chat(&server),
        vec![
            Message::user("Go"),
            tool_call_message(&[("tool_123", "lookup", json!({}))]),
            tool_result("tool_123", ""),
        ],
    );
    assert_eq!(
        payload["messages"][2],
        json!({
            "role": "user",
            "content": [{ "type": "tool_result", "tool_use_id": "tool_123", "content": [{ "type": "text", "text": "(no output)" }] }]
        })
    );
}

// spec: protocols/anthropic/tools_spec.rb:279 returns nil for empty or nil input
#[test]
fn no_tool_use_blocks_means_no_tool_calls() {
    assert_eq!(
        parse(json!({ "model": MODEL, "usage": {} })).tool_calls,
        None
    );
    assert_eq!(parse(body(json!([]))).tool_calls, None);
}

// ---- anthropic_compaction_spec.rb ----------------------------------------------------------------

fn compaction_body() -> Value {
    json!({
        "content": [
            { "type": "compaction", "content": "Summary of the conversation: earlier turns summarized." },
            { "type": "text", "text": "The answer is 42." }
        ],
        "stop_reason": "end_turn", "model": "claude-sonnet-4-6", "usage": {}
    })
}

// spec: protocols/anthropic_compaction_spec.rb:30 replays the compaction block verbatim in later turns
#[tokio::test]
async fn a_compaction_block_replays_verbatim() {
    let server = serve(vec![compaction_body()]).await;
    let mut chat = chat(&server);
    chat.ask("hi").await.unwrap();
    assert_eq!(replayed(&mut chat)[0], compaction_body()["content"][0]);
}

// spec: protocols/anthropic_compaction_spec.rb:38 accumulates streamed compaction deltas into the reconstructed block
#[test]
fn streamed_compaction_deltas_accumulate() {
    let mut state = StreamBlocks::default();
    for event in [
        json!({ "type": "content_block_start", "index": 0, "content_block": { "type": "compaction" } }),
        json!({ "type": "content_block_delta", "index": 0, "delta": { "type": "compaction_delta", "content": "Summary of " } }),
        json!({ "type": "content_block_delta", "index": 0, "delta": { "type": "compaction_delta", "content": "the conversation." } }),
        json!({ "type": "content_block_stop", "index": 0 }),
    ] {
        anthropic::build_chunk(&mut state, &event);
    }
    let chunk = anthropic::build_chunk(&mut state, &json!({ "type": "message_stop" }));
    let compaction = chunk
        .server_tool_calls
        .iter()
        .find(|c| c.kind == "compaction")
        .unwrap();
    assert_eq!(
        compaction.result,
        Some(json!("Summary of the conversation."))
    );
}

fn compacted_usage() -> Value {
    json!({
        "input_tokens": 187, "output_tokens": 85, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
        "iterations": [
            { "input_tokens": 99_207, "output_tokens": 125, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0, "type": "compaction" },
            { "input_tokens": 187, "output_tokens": 85, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0, "type": "message" }
        ]
    })
}

// spec: protocols/anthropic_compaction_spec.rb:81 sums cache tokens across iterations
#[tokio::test]
async fn cache_tokens_sum_across_iterations() {
    let mut usage = compacted_usage();
    usage["iterations"][0]["cache_read_input_tokens"] = 1_000.into();
    usage["iterations"][1]["cache_creation_input_tokens"] = 20.into();
    let mut response = text_response("Hi.");
    response["usage"] = usage;
    let server = serve(vec![response]).await;
    let reply = chat(&server).ask("hi").await.unwrap();
    assert_eq!(
        (reply.tokens.cache_read, reply.tokens.cache_write),
        (Some(1_000), Some(20))
    );
}

// spec: protocols/anthropic_compaction_spec.rb:114 sums iterations reported mid-stream
#[test]
fn streamed_iterations_sum() {
    let chunk = anthropic::build_chunk(
        &mut StreamBlocks::default(),
        &json!({ "type": "message_delta", "usage": compacted_usage() }),
    );
    assert_eq!(chunk.tokens.output, Some(210));
}
