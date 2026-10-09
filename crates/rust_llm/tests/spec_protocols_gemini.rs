//! Gemini protocol specs ported from RubyLLM 2.0: `protocols/gemini/{chat,media,streaming,tools}_spec.rb`.
//! Ruby calls the protocol's private helpers (`format_messages`, `build_response_content`,
//! `extract_citations`, `parse_streaming_error`); these read the payload `Chat#render` produces, the
//! message the public `gemini::parse_completion_body` returns, the chunk `gemini::build_chunk`
//! returns, or the error a stream raises, which is what those helpers feed. `// spec:` lines tie
//! each test to its Ruby example.

mod spec_helpers;

use async_trait::async_trait;
use rust_llm::files::UploadedFile;
use rust_llm::message::RawResponse;
use rust_llm::model::Model;
use rust_llm::protocols::{StreamState, gemini};
use rust_llm::thinking::ThinkingConfig;
use rust_llm::tool::{Tool, ToolError, ToolResult};
use rust_llm::{
    Attachment, Chat, Error, FinishReason, Message, ProtocolName, Resolution, Role, Thinking,
    ToolCall,
};
use serde_json::{Map, Value, json};
use spec_helpers::*;

const GEMINI: &str = "gemini-2.5-flash";

fn fixture(name: &str) -> String {
    format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
}

/// A fixture attachment with its bytes read, as `Attachment.new(path)` reads them lazily.
async fn loaded(name: &str) -> Attachment {
    let mut a = Attachment::new(fixture(name));
    a.content().await.unwrap();
    a
}

/// `Attachment.new(StringIO.new(bytes), filename:)`.
fn bytes(content: &str, filename: &str) -> Attachment {
    Attachment::from_bytes(content.as_bytes().to_vec(), filename, None)
}

/// `RubyLLM::UploadedFile.new(id:, filename:, mime_type:)`.
fn uploaded(id: &str, filename: &str, mime_type: &str) -> UploadedFile {
    UploadedFile {
        id: id.into(),
        provider: "gemini".into(),
        filename: Some(filename.into()),
        byte_size: None,
        created_at: None,
        expires_at: None,
        status: None,
        mime_type: Some(mime_type.into()),
        purpose: None,
        uri: None,
        downloadable: None,
        metadata: Value::Null,
    }
}

fn gemini_chat(server: &wiremock::MockServer, model: &str) -> Chat {
    Chat::with_config(config(server), Some(model), Some("gemini"), false).unwrap()
}

/// `render_payload(messages, ...)` for `model`.
async fn render(model: &str, messages: Vec<Message>) -> Value {
    let server = serve(vec![]).await;
    let mut chat = gemini_chat(&server, model);
    chat.set_messages(messages);
    chat.render().unwrap()
}

/// `Media.format_content(text, attachments)`: the parts of a rendered user turn.
async fn user_parts(text: Option<&str>, attachments: Vec<Attachment>) -> Vec<Value> {
    let payload = render(
        GEMINI,
        vec![Message::new(Role::User, text.map(str::to_string)).with_attachments(attachments)],
    )
    .await;
    payload["contents"][0]["parts"].as_array().unwrap().clone()
}

fn tool_result(id: &str, content: &str, attachments: Vec<Attachment>) -> Message {
    let mut m = Message::new(Role::Tool, Some(content.to_string()));
    m.tool_call_id = Some(id.into());
    m.attachments = attachments;
    m
}

/// `format_tool_result(message)` for `model`: the parts of the rendered tool turn.
async fn tool_result_parts(model: &str, result: Message) -> Vec<Value> {
    let payload = render(model, vec![Message::user("Go"), result]).await;
    payload["contents"][1]["parts"].as_array().unwrap().clone()
}

fn parse(data: Value) -> Message {
    gemini::parse_completion_body(
        &Model::default_for(GEMINI, "gemini"),
        &data,
        RawResponse::default(),
    )
    .unwrap()
}

/// A response whose single candidate carries `parts`.
fn parts_body(parts: Value) -> Value {
    json!({ "candidates": [{ "content": { "parts": parts } }], "usageMetadata": {} })
}

fn b64(data: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.encode(data)
}

// ---- chat_spec.rb: render_payload ------------------------------------------------------------------

fn schema_payload(schema: Value) -> Value {
    let mut c = rust_llm::Config::default();
    c.set("gemini_api_key", "test");
    let mut chat = Chat::with_config(
        std::sync::Arc::new(c),
        Some("gemini-flash-latest"),
        Some("gemini"),
        false,
    )
    .unwrap()
    .with_schema(schema);
    chat.ask_later("hi").unwrap();
    chat.render().unwrap()["generationConfig"]["responseJsonSchema"].clone()
}

// spec: protocols/gemini/chat_spec.rb:63 strips strict, which Gemini has no field for
#[test]
fn strict_is_stripped_from_the_response_schema() {
    let rendered = schema_payload(json!({
        "name": "PersonSchema",
        "schema": { "type": "object", "properties": { "result": { "type": "string" } }, "strict": true }
    }));
    assert_eq!(
        rendered,
        json!({ "type": "object", "properties": { "result": { "type": "string" } } })
    );
}

// spec: protocols/gemini/chat_spec.rb:86 keeps the JSON Schema keywords the old conversion dropped
#[test]
fn rich_json_schema_keywords_are_kept() {
    let rendered = schema_payload(json!({
        "name": "contact",
        "schema": {
            "type": "object",
            "additionalProperties": false,
            "$defs": { "Tag": { "type": "string" } },
            "properties": {
                "name": { "type": "string", "pattern": "^[A-Z][a-z]+$", "minLength": 2 },
                "kind": { "const": "person" },
                "contact": { "anyOf": [{ "type": "string", "format": "email" }, { "type": "integer", "minimum": 1 }] },
                "tags": { "type": "array", "items": { "$ref": "#/$defs/Tag" }, "default": [] }
            },
            "required": ["name", "kind", "contact"]
        }
    }));
    assert_eq!(rendered["additionalProperties"], json!(false));
    assert_eq!(rendered["$defs"], json!({ "Tag": { "type": "string" } }));
    assert_eq!(
        rendered["properties"]["name"],
        json!({ "type": "string", "pattern": "^[A-Z][a-z]+$", "minLength": 2 })
    );
    assert_eq!(rendered["properties"]["kind"], json!({ "const": "person" }));
    assert_eq!(
        rendered["properties"]["contact"]["anyOf"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    assert_eq!(
        rendered["properties"]["tags"]["items"],
        json!({ "$ref": "#/$defs/Tag" })
    );
    assert_eq!(rendered["properties"]["tags"]["default"], json!([]));
}

// ---- chat_spec.rb: format_messages ---------------------------------------------------------------

// spec: protocols/gemini/chat_spec.rb:172 restores call order when results of the same tool finish out of order
#[tokio::test]
async fn out_of_order_results_are_put_back_in_call_order() {
    let payload = render(
        GEMINI,
        vec![
            Message::user("Question?"),
            tool_call_message(&[
                ("call_1", "weather", json!({ "city": "Berlin" })),
                ("call_2", "weather", json!({ "city": "Paris" })),
            ]),
            tool_result("call_2", "Paris is sunny", vec![]),
            tool_result("call_1", "Berlin is rainy", vec![]),
        ],
    )
    .await;
    let contents: Vec<Value> = payload["contents"][2]["parts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p["functionResponse"]["response"]["content"].clone())
        .collect();
    assert_eq!(
        contents,
        vec![
            json!([{ "text": "Berlin is rainy" }]),
            json!([{ "text": "Paris is sunny" }])
        ]
    );
}

// ---- chat_spec.rb: build_thinking_config, format_system_instruction, build_thought_part ----------

// spec: protocols/gemini/chat_spec.rb:250 sends a numeric budget when one is set
#[tokio::test]
async fn a_numeric_budget_is_sent_as_thinking_budget() {
    let server = serve(vec![]).await;
    let mut chat = gemini_chat(&server, GEMINI).with_thinking(ThinkingConfig::budget(1024));
    chat.ask_later("hi").unwrap();
    assert_eq!(
        chat.render().unwrap()["generationConfig"]["thinkingConfig"],
        json!({ "includeThoughts": true, "thinkingBudget": 1024 })
    );
}

// spec: protocols/gemini/chat_spec.rb:269 skips empty system messages
#[tokio::test]
async fn an_empty_system_message_renders_no_system_instruction() {
    let payload = render(GEMINI, vec![Message::system(""), Message::user("hi")]).await;
    assert!(payload.get("systemInstruction").is_none(), "{payload}");
}

// ---- chat_spec.rb: replaying an answer -------------------------------------------------------------

/// `answer(content, thinking)`: an answer Gemini produced (its usage names the provider).
fn gemini_answer(content: Option<&str>, thinking: Option<Thinking>) -> Message {
    let mut m = Message::new(Role::Assistant, content.map(str::to_string));
    m.thinking = thinking;
    m.usage_entries = vec![rust_llm::UsageEntry {
        id: rust_llm::UsageEntry::next_id(),
        owner: None,
        operation: rust_llm::message::Operation::Chat,
        provider: "gemini".into(),
        model: GEMINI.into(),
        status: rust_llm::UsageStatus::Succeeded,
        tokens: Default::default(),
        cost: Default::default(),
    }];
    m
}

/// `replay(message)`: `format_parts` of the answer, read from the rendered model turn.
async fn replayed(message: Message) -> Value {
    render(GEMINI, vec![Message::user("hi"), message]).await["contents"][1]["parts"].clone()
}

// spec: protocols/gemini/chat_spec.rb:290 replaying an answer > sends a signature without thinking text back on the answer part
#[tokio::test]
async fn sends_a_signature_without_thinking_text_back_on_the_answer_part() {
    let parts = replayed(gemini_answer(
        Some("Done."),
        Thinking::build(None, Some("sig".into())),
    ))
    .await;
    let data_fields = [
        "text",
        "inline_data",
        "file_data",
        "functionCall",
        "functionResponse",
    ];
    assert!(
        parts
            .as_array()
            .unwrap()
            .iter()
            .all(|p| data_fields.iter().any(|f| p.get(f).is_some()))
    );
    assert_eq!(
        parts,
        json!([{ "text": "Done.", "thoughtSignature": "sig" }])
    );
}

// spec: protocols/gemini/chat_spec.rb:297 replaying an answer > keeps the thought summary unsigned ahead of the signed answer
#[tokio::test]
async fn keeps_the_thought_summary_unsigned_ahead_of_the_signed_answer() {
    let parts = replayed(gemini_answer(
        Some("Done."),
        Thinking::build(Some("Adding.".into()), Some("sig".into())),
    ))
    .await;
    assert_eq!(
        parts,
        json!([{ "thought": true, "text": "Adding." }, { "text": "Done.", "thoughtSignature": "sig" }])
    );
}

// spec: protocols/gemini/chat_spec.rb:303 replaying an answer > carries the signature of an answer without content in an empty text part
#[tokio::test]
async fn carries_the_signature_of_an_answer_without_content_in_an_empty_text_part() {
    let parts = replayed(gemini_answer(
        None,
        Thinking::build(None, Some("sig".into())),
    ))
    .await;
    assert_eq!(parts, json!([{ "text": "", "thoughtSignature": "sig" }]));
}

// spec: protocols/gemini/chat_spec.rb:309 replaying an answer > sends an answer back with its parts as Gemini returned them
#[tokio::test]
async fn sends_an_answer_back_with_its_parts_as_gemini_returned_them() {
    let parts = json!([{ "text": "Adding.", "thought": true }, { "text": "5 + 3 = 8", "thoughtSignature": "sig" }]);
    let parsed = parse(parts_body(parts.clone()));
    let replay = replayed(gemini_answer(parsed.content.as_deref(), parsed.thinking)).await;
    assert_eq!(replay, parts);
}

// spec: protocols/gemini/chat_spec.rb:194 #format_messages > signs the first unsigned call of each step in the current turn only
#[tokio::test]
async fn signs_the_first_unsigned_call_of_each_step_in_the_current_turn_only() {
    let step = |ids: &[&str], signature: Option<&str>| {
        let mut m = Message::new(Role::Assistant, Some(String::new()));
        let mut calls = rust_llm::message::indexmap_lite::IndexMap::new();
        for id in ids {
            let mut call = ToolCall::new(*id, "lookup", Map::new());
            call.thought_signature = signature.map(str::to_string);
            calls.insert(id.to_string(), call);
        }
        m.tool_calls = Some(calls);
        m
    };
    let payload = render(
        GEMINI,
        vec![
            Message::user("Question?"),
            step(&["a"], None),
            Message::tool_result("a", "A"),
            Message::assistant("Answer."),
            Message::user("Again?"),
            step(&["b", "c"], None),
            Message::tool_result("b", "B"),
            Message::tool_result("c", "C"),
            step(&["d"], Some("sig")),
            Message::tool_result("d", "D"),
        ],
    )
    .await;
    let signatures: Vec<Vec<Value>> = payload["contents"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["role"] == "model")
        .map(|c| {
            c["parts"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| p.get("thoughtSignature").cloned().unwrap_or(Value::Null))
                .collect()
        })
        .collect();
    assert_eq!(
        signatures,
        vec![
            vec![Value::Null],
            vec![Value::Null],
            vec![json!("skip_thought_signature_validator"), Value::Null],
            vec![json!("sig")],
        ]
    );
}

// ---- chat_spec.rb: extract_citations ---------------------------------------------------------------

fn grounded(text: &str, metadata: Value) -> Value {
    json!({ "candidates": [{ "content": { "parts": [{ "text": text }] }, "groundingMetadata": metadata }], "usageMetadata": {} })
}

// spec: protocols/gemini/chat_spec.rb:321 returns nothing without grounding metadata
#[test]
fn no_grounding_metadata_means_no_citations() {
    assert!(
        parse(parts_body(json!([{ "text": "text" }])))
            .citations
            .is_empty()
    );
}

// spec: protocols/gemini/chat_spec.rb:325 cites every grounding chunk when there are no supports
#[test]
fn every_grounding_chunk_is_cited_without_supports() {
    let citations = parse(grounded(
        "text",
        json!({ "groundingChunks": [
            { "web": { "uri": "https://a.example", "title": "A" } },
            { "retrievedContext": { "uri": "https://b.example", "title": "B" } },
            { "unknown": {} },
            "not a chunk"
        ] }),
    ))
    .citations;
    assert_eq!(
        citations
            .iter()
            .map(|c| c.url.as_deref())
            .collect::<Vec<_>>(),
        [Some("https://a.example"), Some("https://b.example")]
    );
    assert_eq!(
        citations.iter().map(|c| c.source_index).collect::<Vec<_>>(),
        [Some(0), Some(1)]
    );
}

// spec: protocols/gemini/chat_spec.rb:347 anchors supports to character offsets in the response text
#[test]
fn supports_are_anchored_to_character_offsets() {
    let citations = parse(grounded(
        "Café is French",
        json!({
            "groundingChunks": [{ "web": { "uri": "https://a.example", "title": "A" } }],
            "groundingSupports": [{ "segment": { "endIndex": 4, "text": "Café" }, "groundingChunkIndices": [0, 9] }]
        }),
    ))
    .citations;
    assert_eq!(citations.len(), 1);
    assert_eq!(citations[0].start_index, Some(0));
    assert_eq!(citations[0].end_index, Some(4));
}

// spec: protocols/gemini/chat_spec.rb:368 leaves offsets nil when the support carries no segment
#[test]
fn a_support_without_a_segment_has_no_offsets() {
    let citations = parse(grounded(
        "text",
        json!({
            "groundingChunks": [{ "web": { "uri": "https://a.example" } }],
            "groundingSupports": [{ "groundingChunkIndices": [0] }]
        }),
    ))
    .citations;
    assert_eq!(citations[0].start_index, None);
    assert_eq!(citations[0].end_index, None);
}

// ---- chat_spec.rb: parse_content, extract_thought_signature, parse_completion_response -----------

// spec: protocols/gemini/chat_spec.rb:388 returns empty content for a response with no candidate
#[test]
fn no_candidate_is_empty_content() {
    let m = parse(json!({}));
    assert_eq!((m.content.as_deref(), m.attachments.len()), (Some(""), 0));
}

// spec: protocols/gemini/chat_spec.rb:392 returns empty content for a candidate with no parts
#[test]
fn a_candidate_without_parts_is_empty_content() {
    let m = parse(json!({ "candidates": [{ "content": {} }] }));
    assert_eq!((m.content.as_deref(), m.attachments.len()), (Some(""), 0));
}

// spec: protocols/gemini/chat_spec.rb:398 reads the signature off a function call part
#[test]
fn the_signature_is_read_off_a_function_call_part() {
    let m = parse(parts_body(
        json!([{ "functionCall": { "thought_signature": "sig" } }]),
    ));
    assert_eq!(m.thinking.and_then(|t| t.signature).as_deref(), Some("sig"));
}

// spec: protocols/gemini/chat_spec.rb:404 returns nil when no part carries one
#[test]
fn no_part_carries_a_signature() {
    let m = parse(parts_body(json!([{ "text": "hi" }])));
    assert_eq!(m.thinking.and_then(|t| t.signature), None);
}

// spec: protocols/gemini/chat_spec.rb:488 normalizes finishReason
#[test]
fn safety_normalizes_to_content_filter() {
    let m = parse(
        json!({ "candidates": [{ "finishReason": "SAFETY", "content": { "parts": [{ "text": "No" }] } }] }),
    );
    assert_eq!(m.finish_reason, Some(FinishReason::ContentFilter));
}

// spec: protocols/gemini/chat_spec.rb:507 keeps thought-only parts out of assistant content
#[test]
fn thought_only_parts_stay_out_of_content() {
    let m = parse(parts_body(
        json!([{ "thought": true, "text": "Internal reasoning only" }]),
    ));
    assert_eq!(m.content.as_deref(), Some(""));
    assert_eq!(
        m.thinking.and_then(|t| t.text).as_deref(),
        Some("Internal reasoning only")
    );
}

// spec: protocols/gemini/chat_spec.rb:531 keeps non-thought text in content when mixed with thought parts
#[test]
fn mixed_thought_and_text_parts_split() {
    let m = parse(parts_body(
        json!([{ "thought": true, "text": "Reasoning trace" }, { "text": "{\"ok\":true}" }]),
    ));
    assert_eq!(m.content.as_deref(), Some("{\"ok\":true}"));
    assert_eq!(
        m.thinking.and_then(|t| t.text).as_deref(),
        Some("Reasoning trace")
    );
}

// spec: protocols/gemini/chat_spec.rb:556 captures cached token usage when present
#[test]
fn cached_token_usage_is_captured() {
    let m = parse(json!({
        "candidates": [{ "content": { "parts": [{ "text": "Hi" }] } }],
        "usageMetadata": { "promptTokenCount": 42, "candidatesTokenCount": 8, "cachedContentTokenCount": 21 }
    }));
    assert_eq!(
        (m.tokens.input, m.tokens.output, m.tokens.cache_read),
        (Some(21), Some(8), Some(21))
    );
}

// ---- media_spec.rb: format_content -----------------------------------------------------------------

// spec: protocols/gemini/media_spec.rb:7 raises a clear error for unsupported rich documents
#[tokio::test]
async fn a_rich_document_is_rejected() {
    let server = serve(vec![]).await;
    let mut chat = gemini_chat(&server, GEMINI);
    chat.ask_later_with(
        "Summarize this file",
        vec![bytes("docx bytes", "proposal.docx")],
    )
    .unwrap();
    let err = chat.render().unwrap_err();
    assert!(
        matches!(&err, Error::UnsupportedAttachment(m)
            if m.contains("Unsupported attachment type: application/vnd.openxmlformats-officedocument.wordprocessingml.document")),
        "{err:?}"
    );
}

// spec: protocols/gemini/media_spec.rb:59 sends high resolution when ultra high is requested for a PDF
#[tokio::test]
async fn ultra_high_on_a_pdf_sends_high() {
    let parts = user_parts(
        Some("Read this page"),
        vec![bytes("pdf bytes", "page.pdf").with_resolution(Resolution::UltraHigh)],
    )
    .await;
    assert_eq!(
        parts[1]["media_resolution"],
        json!({ "level": "MEDIA_RESOLUTION_HIGH" })
    );
}

// spec: protocols/gemini/media_spec.rb:67 sets media_resolution on provider-managed files
#[tokio::test]
async fn a_provider_managed_file_carries_media_resolution() {
    let file = Attachment::from_uploaded(uploaded("files/abc", "video.mp4", "video/mp4"))
        .with_resolution(Resolution::Low);
    let parts = user_parts(Some("Watch this"), vec![file]).await;
    assert_eq!(
        parts[1]["media_resolution"],
        json!({ "level": "MEDIA_RESOLUTION_LOW" })
    );
}

// spec: protocols/gemini/media_spec.rb:76 sends ultra high resolution on images
#[tokio::test]
async fn ultra_high_on_an_image_is_sent_as_is() {
    let parts = user_parts(
        Some("Read this"),
        vec![
            loaded("ruby.png")
                .await
                .with_resolution(Resolution::UltraHigh),
        ],
    )
    .await;
    assert_eq!(
        parts[1]["media_resolution"],
        json!({ "level": "MEDIA_RESOLUTION_ULTRA_HIGH" })
    );
}

// spec: protocols/gemini/media_spec.rb:84 sends high resolution when ultra high is requested for a video
#[tokio::test]
async fn ultra_high_on_a_video_sends_high() {
    let file = Attachment::from_uploaded(uploaded("files/abc", "video.mp4", "video/mp4"))
        .with_resolution(Resolution::UltraHigh);
    let parts = user_parts(Some("Watch this"), vec![file]).await;
    assert_eq!(
        parts[1]["media_resolution"],
        json!({ "level": "MEDIA_RESOLUTION_HIGH" })
    );
}

// spec: protocols/gemini/media_spec.rb:103 omits media_resolution on audio
#[tokio::test]
async fn audio_carries_no_media_resolution() {
    let parts = user_parts(
        Some("Listen"),
        vec![loaded("ruby.wav").await.with_resolution(Resolution::Low)],
    )
    .await;
    assert!(parts[1].get("inline_data").is_some());
    assert!(parts[1].get("media_resolution").is_none(), "{}", parts[1]);
}

// spec: protocols/gemini/media_spec.rb:111 omits media_resolution from standalone parts such as tool results
#[tokio::test]
async fn a_tool_result_attachment_carries_no_media_resolution() {
    let pdf = bytes("pdf bytes", "page.pdf").with_resolution(Resolution::High);
    let parts = tool_result_parts(GEMINI, tool_result("uuid-123", "Found it", vec![pdf])).await;
    let sibling = parts.last().unwrap();
    assert!(sibling.get("inline_data").is_some(), "{sibling}");
    assert!(sibling.get("media_resolution").is_none(), "{sibling}");
}

// spec: protocols/gemini/media_spec.rb:211 sends attachments without any text
#[tokio::test]
async fn attachments_are_sent_without_text() {
    let parts = user_parts(None, vec![bytes("pdf bytes", "proposal.pdf")]).await;
    assert_eq!(parts.len(), 1);
    assert!(parts[0].get("inline_data").is_some());
}

// ---- media_spec.rb: build_response_content, attachment_filename -----------------------------------

// spec: protocols/gemini/media_spec.rb:127 parses inline image responses as a text and attachments pair
#[tokio::test]
async fn an_inline_image_response_is_an_attachment_without_text() {
    let image = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut m = parse(parts_body(
        json!([{ "inlineData": { "mimeType": "image/png", "data": b64(&image) } }]),
    ));
    assert_eq!(m.content, None);
    assert_eq!(m.attachments.len(), 1);
    let attachment = &mut m.attachments[0];
    assert_eq!(
        attachment.filename.as_deref(),
        Some("gemini_attachment_1.png")
    );
    assert_eq!(attachment.mime_type, "image/png");
    assert_eq!(attachment.content().await.unwrap(), image);
}

// spec: protocols/gemini/media_spec.rb:159 joins text parts and reports no attachments
#[test]
fn text_parts_are_joined() {
    let m = parse(parts_body(json!([{ "text": "one " }, { "text": "two" }])));
    assert_eq!(m.content.as_deref(), Some("one two"));
    assert!(m.attachments.is_empty());
}

// spec: protocols/gemini/media_spec.rb:166 ignores parts it does not recognize
// (A lone functionCall part still yields a tool call, and `Message` makes a tool call's missing
// content `''` in Ruby too; beside text, the part adds nothing to content or attachments.)
#[test]
fn unrecognized_parts_add_no_content_or_attachments() {
    let m = parse(parts_body(json!([{ "functionCall": {} }])));
    assert_eq!((m.content.as_deref(), m.attachments.len()), (Some(""), 0));
    let m = parse(parts_body(
        json!([{ "functionCall": {} }, { "text": "hi" }]),
    ));
    assert_eq!((m.content.as_deref(), m.attachments.len()), (Some("hi"), 0));
}

// spec: protocols/gemini/media_spec.rb:170 builds an attachment from a fileData part
#[test]
fn a_file_data_part_is_an_attachment() {
    let m = parse(parts_body(
        json!([{ "fileData": { "fileUri": "https://files.example/report", "mimeType": "application/pdf" } }]),
    ));
    assert_eq!(m.content, None);
    assert_eq!(
        m.attachments[0].filename.as_deref(),
        Some("gemini_attachment_1.pdf")
    );
    assert_eq!(m.attachments[0].url(), Some("https://files.example/report"));
}

// spec: protocols/gemini/media_spec.rb:179 prefers the filename the response carries
#[test]
fn a_file_data_filename_wins() {
    let m = parse(parts_body(
        json!([{ "fileData": { "fileUri": "https://files.example/x", "filename": "report.pdf" } }]),
    ));
    assert_eq!(m.attachments[0].filename.as_deref(), Some("report.pdf"));
}

// spec: protocols/gemini/media_spec.rb:187 skips a fileData part with no URI
#[test]
fn a_file_data_part_without_a_uri_is_skipped() {
    let m = parse(parts_body(json!([{ "fileData": {} }])));
    assert_eq!((m.content, m.attachments.len()), (None, 0));
}

// spec: protocols/gemini/media_spec.rb:191 skips an inlineData part with no data
#[test]
fn an_inline_data_part_without_data_is_skipped() {
    let m = parse(parts_body(
        json!([{ "inlineData": { "mimeType": "image/png" } }]),
    ));
    assert_eq!((m.content, m.attachments.len()), (None, 0));
}

// spec: protocols/gemini/media_spec.rb:199 falls back to an extensionless name without a mime type
#[test]
fn an_attachment_without_a_mime_type_has_no_extension() {
    let m = parse(parts_body(
        json!([{ "inlineData": { "data": b64(b"bytes") } }]),
    ));
    assert_eq!(
        m.attachments[0].filename.as_deref(),
        Some("gemini_attachment_1")
    );
}

// spec: protocols/gemini/media_spec.rb:203 normalizes the extensions Gemini reports
#[test]
fn reported_extensions_are_normalized() {
    let m = parse(parts_body(json!([
        { "inlineData": { "mimeType": "image/jpeg", "data": b64(b"a") } },
        { "inlineData": { "mimeType": "text/plain", "data": b64(b"b") } },
        { "inlineData": { "mimeType": "image/svg+xml", "data": b64(b"c") } }
    ])));
    let names: Vec<_> = m
        .attachments
        .iter()
        .map(|a| a.filename.as_deref().unwrap_or(""))
        .collect();
    assert_eq!(
        names,
        [
            "gemini_attachment_1.jpg",
            "gemini_attachment_2.txt",
            "gemini_attachment_3.svg.xml"
        ]
    );
}

// ---- streaming_spec.rb -----------------------------------------------------------------------------

// spec: protocols/gemini/streaming_spec.rb:16 captures cached token usage on chunks when present
#[test]
fn a_chunk_captures_cached_token_usage() {
    let chunk = gemini::build_chunk(
        &mut StreamState::default(),
        &json!({
            "candidates": [{ "content": { "parts": [{ "text": "hello" }] } }],
            "usageMetadata": { "promptTokenCount": 10, "candidatesTokenCount": 4, "cachedContentTokenCount": 6 },
            "modelVersion": "gemini-2.5-flash"
        }),
    );
    assert_eq!(
        (
            chunk.tokens.input,
            chunk.tokens.output,
            chunk.tokens.cache_read
        ),
        (Some(4), Some(4), Some(6))
    );
}

// spec: protocols/gemini/streaming_spec.rb:92 preserves raw finishReason on chunks
#[test]
fn a_chunk_normalizes_finish_reason() {
    let chunk = gemini::build_chunk(
        &mut StreamState::default(),
        &json!({ "candidates": [{ "finishReason": "MAX_TOKENS", "content": { "parts": [{ "text": "hello" }] } }] }),
    );
    assert_eq!(chunk.finish_reason, Some(FinishReason::MaxTokens));
}

async fn stream_error(data: &str) -> Error {
    let server = serve_templates(vec![sse(format!("event: error\ndata: {data}\n\n"))]).await;
    gemini_chat(&server, GEMINI)
        .ask_stream("hi", |_| {})
        .await
        .unwrap_err()
}

// spec: protocols/gemini/streaming_spec.rb:108 parses error objects
#[tokio::test]
async fn an_error_object_raises_its_code() {
    let data = r#"{"error":{"code":429,"message":"Quota exceeded"}}"#;
    assert_eq!(
        rust_llm::protocols::streaming_error_status(ProtocolName::Gemini)(data),
        Some(429)
    );
    let err = stream_error(data).await;
    assert!(
        matches!(&err, Error::RateLimit(m, Some(r)) if m == "Quota exceeded" && r.status == 429),
        "{err:?}"
    );
}

// spec: protocols/gemini/streaming_spec.rb:118 handles a body that parses to a bare JSON string
// (`parse_streaming_error` gives no status; the error is raised as the stream's default 500.)
#[tokio::test]
async fn a_bare_json_string_error_has_no_status() {
    let data = r#""model unavailable""#;
    assert_eq!(
        rust_llm::protocols::streaming_error_status(ProtocolName::Gemini)(data),
        None
    );
    let err = stream_error(data).await;
    assert!(
        matches!(&err, Error::Server(m, Some(r)) if m == "model unavailable" && r.status == 500),
        "{err:?}"
    );
}

// spec: protocols/gemini/streaming_spec.rb:125 handles a string error value
#[tokio::test]
async fn a_string_error_value_has_no_status() {
    let data = r#"{"error":"model unavailable"}"#;
    assert_eq!(
        rust_llm::protocols::streaming_error_status(ProtocolName::Gemini)(data),
        None
    );
    let err = stream_error(data).await;
    assert!(
        matches!(&err, Error::Server(m, Some(r)) if m == "model unavailable" && r.status == 500),
        "{err:?}"
    );
}

// ---- tools_spec.rb ---------------------------------------------------------------------------------

// spec: protocols/gemini/tools_spec.rb:37 outputs a functionCall part for each tool call and preserves assistant text
#[tokio::test]
async fn function_call_parts_follow_the_assistant_text() {
    let mut call = tool_call_message(&[
        ("a", "weather", json!({ "latitude": "52.5200" })),
        ("b", "best_language_to_learn", json!({})),
    ]);
    call.content = Some("Working on it...".into());
    let payload = render(GEMINI, vec![Message::user("Go"), call]).await;
    let parts = payload["contents"][1]["parts"].as_array().unwrap();
    assert_eq!(parts.len(), 3);
    assert_eq!(parts[0], json!({ "text": "Working on it..." }));
    assert_eq!(
        parts[1]["functionCall"],
        json!({ "name": "weather", "args": { "latitude": "52.5200" } })
    );
    assert_eq!(
        parts[2]["functionCall"],
        json!({ "name": "best_language_to_learn", "args": {} })
    );
}

// spec: protocols/gemini/tools_spec.rb:68 uses the tool call id for Gemini function responses
#[tokio::test]
async fn an_unmatched_result_is_named_by_its_tool_call_id() {
    let parts = tool_result_parts(GEMINI, tool_result("uuid-123", "Result payload", vec![])).await;
    assert_eq!(
        parts,
        [
            json!({ "functionResponse": { "name": "uuid-123", "response": { "name": "uuid-123", "content": [{ "text": "Result payload" }] } } })
        ]
    );
}

// spec: protocols/gemini/tools_spec.rb:90 uses a placeholder when the tool returns no content
#[tokio::test]
async fn an_empty_result_renders_a_placeholder() {
    let parts = tool_result_parts(GEMINI, tool_result("uuid-123", "", vec![])).await;
    assert_eq!(
        parts,
        [
            json!({ "functionResponse": { "name": "uuid-123", "response": { "name": "uuid-123", "content": [{ "text": "(no output)" }] } } })
        ]
    );
}

// spec: protocols/gemini/tools_spec.rb:137 nests media for the latest aliases, which track the newest release
// spec: protocols/gemini/tools_spec.rb:219 treats the latest aliases as the newest generation
#[tokio::test]
async fn latest_aliases_nest_media_in_the_function_response() {
    for id in [
        "gemini-flash-latest",
        "gemini-pro-latest",
        "gemini-flash-lite-latest",
    ] {
        let parts = tool_result_parts(
            id,
            tool_result("uuid-123", "Found it", vec![loaded("ruby.png").await]),
        )
        .await;
        assert_eq!(parts.len(), 1, "{id}");
        assert_eq!(
            parts[0]["functionResponse"]["parts"][0]["inline_data"]["mime_type"], "image/png",
            "{id}"
        );
    }
}

// spec: protocols/gemini/tools_spec.rb:146 keeps text files as sibling text parts on Gemini 3 models
#[tokio::test]
async fn text_files_stay_sibling_parts_on_gemini_3() {
    let parts = tool_result_parts(
        "gemini-3-flash-preview",
        tool_result("uuid-123", "Found it", vec![loaded("ruby.txt").await]),
    )
    .await;
    assert!(parts[0]["functionResponse"].get("parts").is_none());
    assert!(parts.last().unwrap().get("text").is_some());
}

// spec: protocols/gemini/tools_spec.rb:153 keeps provider-managed files as sibling parts on Gemini 3 models
#[tokio::test]
async fn provider_files_stay_sibling_parts_on_gemini_3() {
    let mut file = uploaded("files/abc123", "ruby.png", "image/png");
    file.byte_size = Some(1234);
    let parts = tool_result_parts(
        "gemini-3-flash-preview",
        tool_result(
            "uuid-123",
            "Found it",
            vec![Attachment::from_uploaded(file)],
        ),
    )
    .await;
    assert!(parts[0]["functionResponse"].get("parts").is_none());
    assert!(parts.last().unwrap().get("file_data").is_some());
}

// spec: protocols/gemini/tools_spec.rb:225 reads no generation out of an id that names none
// (Ruby's `supported?(nil)` has no counterpart: a Rust model always has an id.)
#[tokio::test]
async fn ids_naming_no_generation_keep_media_as_siblings() {
    for id in [
        "gemini-omni-flash-preview",
        "deep-research-max-preview-04-2026",
        "openai/gpt-oss-120b-maas",
        "moonshotai/kimi-k2-thinking-maas",
    ] {
        let server = serve(vec![]).await;
        let mut chat = Chat::with_config(config(&server), Some(id), Some("gemini"), true).unwrap();
        chat.set_messages(vec![
            Message::user("Go"),
            tool_result("uuid-123", "Found it", vec![loaded("ruby.png").await]),
        ]);
        let parts = chat.render().unwrap()["contents"][1]["parts"]
            .as_array()
            .unwrap()
            .clone();
        assert!(parts[0]["functionResponse"].get("parts").is_none(), "{id}");
        assert!(parts.last().unwrap().get("inline_data").is_some(), "{id}");
    }
}

/// `instance_double(RubyLLM::Tool, parameters_schema:, provider_options:)`.
struct Declared {
    schema: Option<Value>,
    options: Map<String, Value>,
}

#[async_trait]
impl Tool for Declared {
    fn name(&self) -> String {
        "lookup".into()
    }
    fn description(&self) -> String {
        "Looks up".into()
    }
    fn parameters_schema(&self) -> Option<Value> {
        self.schema.clone()
    }
    fn provider_options(&self) -> Map<String, Value> {
        self.options.clone()
    }
    async fn execute(
        &self,
        _a: Map<String, Value>,
        _c: &ToolCall,
    ) -> Result<ToolResult, ToolError> {
        Ok("".into())
    }
}

async fn declaration(tool: Declared) -> Value {
    let server = serve(vec![]).await;
    let mut chat = gemini_chat(&server, GEMINI).with_tool(tool);
    chat.ask_later("hi").unwrap();
    chat.render().unwrap()["tools"][0]["functionDeclarations"][0].clone()
}

// spec: protocols/gemini/tools_spec.rb:186 merges provider options into the declaration
#[tokio::test]
async fn provider_options_merge_into_the_declaration() {
    let d = declaration(Declared {
        schema: None,
        options: args(json!({ "behavior": "BLOCKING" })),
    })
    .await;
    assert_eq!(d["behavior"], "BLOCKING");
    assert_eq!(
        (d["name"].as_str(), d["description"].as_str()),
        (Some("lookup"), Some("Looks up"))
    );
}

// spec: protocols/gemini/tools_spec.rb:199 is nil for a response Gemini did not send
#[test]
fn malformed_responses_have_no_tool_calls() {
    for data in [
        Value::Null,
        json!("not a hash"),
        json!({ "candidates": [] }),
        json!({ "candidates": [{ "content": {} }] }),
    ] {
        assert_eq!(parse(data.clone()).tool_calls, None, "{data}");
    }
}

// spec: protocols/gemini/tools_spec.rb:259 keeps type unions, references, and constraints the converter dropped
#[tokio::test]
async fn tool_schemas_keep_unions_references_and_constraints() {
    let schema = json!({
        "type": "object",
        "additionalProperties": false,
        "$defs": { "Tag": { "type": "string", "minLength": 2 } },
        "properties": {
            "count": { "type": ["integer", "null"], "multipleOf": 2 },
            "tag": { "$ref": "#/$defs/Tag" }
        }
    });
    let d = declaration(Declared {
        schema: Some(schema.clone()),
        options: Map::new(),
    })
    .await;
    assert_eq!(d["parametersJsonSchema"], schema);
}

// ---- 2.1: original resolution, grounding usage and search suggestions, parallel signatures -------

// spec: protocols/gemini/media_spec.rb:94 .format_content > maps original resolution to #{level} for #{filename}
#[tokio::test]
async fn maps_original_resolution_to_the_highest_level_each_media_takes() {
    for (filename, level) in [
        ("page.png", "ULTRA_HIGH"),
        ("document.pdf", "HIGH"),
        ("video.mp4", "HIGH"),
    ] {
        let parts = user_parts(
            Some("Read this"),
            vec![bytes("bytes", filename).with_resolution(Resolution::Original)],
        )
        .await;
        assert_eq!(
            parts[1]["media_resolution"],
            json!({ "level": format!("MEDIA_RESOLUTION_{level}") }),
            "{filename}"
        );
    }
}

fn grounded_body(metadata: Value, usage: Value) -> Value {
    json!({ "candidates": [{ "content": { "parts": [{ "text": "Ruby 4.0.7" }] },
                             "groundingMetadata": metadata }],
            "usageMetadata": usage })
}

const SUGGESTIONS: &str =
    r#"<style>.container { display: flex; }</style><div class="container">Ruby 4.0.7</div>"#;

/// The parts of a turn that ran Google Search and code execution, with suggestions on the result.
fn searched_parts() -> Value {
    json!([
        { "thoughtSignature": "sig-1",
          "toolCall": { "toolType": "GOOGLE_SEARCH_WEB", "id": "call_1",
                        "args": { "queries": ["latest stable ruby version"] } } },
        { "thoughtSignature": "sig-2",
          "toolResponse": { "toolType": "GOOGLE_SEARCH_WEB", "id": "call_1",
                            "response": { "search_suggestions": "<style></style>" } } },
        { "executableCode": { "language": "PYTHON", "code": "print(4)" } },
        { "codeExecutionResult": { "outcome": "OUTCOME_OK", "output": "4\n" } },
        { "text": "Ruby 4.0.7 is the latest." }
    ])
}

// spec: protocols/gemini/chat_spec.rb:410 #parse_completion_response > counts each distinct web search query the grounding ran
#[test]
fn counts_each_distinct_web_search_query_the_grounding_ran() {
    let m = parse(grounded_body(
        json!({ "webSearchQueries": ["ruby 4.0.7 released", "ruby releases", "", "ruby releases"] }),
        json!({ "promptTokenCount": 534, "candidatesTokenCount": 313 }),
    ));
    assert_eq!(
        m.tokens.server_tool_use,
        json!({ "web_search_requests": 2 }).as_object().cloned()
    );
}

// spec: protocols/gemini/chat_spec.rb:427 #parse_completion_response > puts the search suggestions on the live search call only
#[test]
fn puts_the_search_suggestions_on_the_live_search_call_only() {
    let m = parse(grounded_body(
        json!({ "searchEntryPoint": { "renderedContent": SUGGESTIONS },
                "webSearchQueries": ["ruby 4.0.7 released"] }),
        json!({}),
    ));
    let search = m
        .server_tool_calls
        .iter()
        .find(|c| c.kind == "google_search")
        .unwrap();
    assert_eq!(search.search_suggestions.as_deref(), Some(SUGGESTIONS));
    assert!(
        !serde_json::to_string(&m.to_h())
            .unwrap()
            .contains("container")
    );
}

// spec: protocols/gemini/chat_spec.rb:447 #parse_completion_response > drops search suggestions from the parts it keeps for replay
#[test]
fn drops_search_suggestions_from_the_parts_it_keeps_for_replay() {
    let parts = searched_parts();
    let m = parse(parts_body(parts.clone()));
    let raw = m.raw_content.unwrap();
    assert_eq!(
        raw[1],
        json!({ "thoughtSignature": "sig-2",
                "toolResponse": { "toolType": "GOOGLE_SEARCH_WEB", "id": "call_1", "response": {} } })
    );
    for i in [0, 2, 3, 4] {
        assert_eq!(raw[i], parts[i]);
    }
}

// spec: protocols/gemini/chat_spec.rb:477 #parse_completion_response > counts no web searches without grounding
#[test]
fn counts_no_web_searches_without_grounding() {
    let m = parse(
        json!({ "candidates": [{ "content": { "parts": [{ "text": "Hi" }] } }],
                          "usageMetadata": { "promptTokenCount": 3, "candidatesTokenCount": 1 } }),
    );
    assert_eq!(m.tokens.server_tool_use, None);
}

// spec: protocols/gemini/streaming_spec.rb:40 counts the web searches the final chunk grounds on
#[test]
fn counts_the_web_searches_the_final_chunk_grounds_on() {
    let mut state = StreamState::default();
    let text = gemini::build_chunk(
        &mut state,
        &json!({ "candidates": [{ "content": { "parts": [{ "text": "Ruby" }] } }] }),
    );
    let last = gemini::build_chunk(
        &mut state,
        &json!({ "candidates": [{ "content": { "parts": [{ "text": " 4.0.7" }] }, "finishReason": "STOP",
                                  "groundingMetadata": { "webSearchQueries": ["\"Ruby 4.0.7\" released"] } }],
                 "usageMetadata": { "promptTokenCount": 408, "candidatesTokenCount": 171 } }),
    );
    assert_eq!(text.tokens.server_tool_use, None);
    assert_eq!(
        last.tokens.server_tool_use,
        json!({ "web_search_requests": 1 }).as_object().cloned()
    );
}

// spec: protocols/gemini/streaming_spec.rb:56 streams the search suggestions on the live search call only
#[test]
fn streams_the_search_suggestions_on_the_live_search_call_only() {
    let chunk = gemini::build_chunk(
        &mut StreamState::default(),
        &json!({ "candidates": [{ "content": { "parts": [{ "text": "Ruby 4.0.7" }] }, "finishReason": "STOP",
                                  "groundingMetadata": { "searchEntryPoint": { "renderedContent": SUGGESTIONS },
                                                         "webSearchQueries": ["\"Ruby 4.0.7\" released"] } }] }),
    );
    let search = chunk
        .server_tool_calls
        .iter()
        .find(|c| c.kind == "google_search")
        .unwrap();
    assert_eq!(search.search_suggestions.as_deref(), Some(SUGGESTIONS));
    assert!(!serde_json::to_string(search).unwrap().contains("container"));
}

// spec: protocols/gemini/streaming_spec.rb:74 drops search suggestions from the streamed parts it keeps for replay
#[test]
fn drops_search_suggestions_from_the_streamed_parts_it_keeps_for_replay() {
    let parts = searched_parts();
    let mut state = StreamState::default();
    gemini::build_chunk(
        &mut state,
        &json!({ "candidates": [{ "content": { "parts": [parts[1].clone(), parts[2].clone()] } }] }),
    );
    let chunk = gemini::build_chunk(
        &mut state,
        &json!({ "candidates": [{ "content": { "parts": [{ "text": "4" }] }, "finishReason": "STOP" }] }),
    );
    let raw = chunk.raw_content.unwrap();
    assert_eq!(raw[0]["toolResponse"]["response"], json!({}));
    assert_eq!(raw.as_array().unwrap().len(), 3);
}

// spec: protocols/gemini/tools_spec.rb:52 #format_tool_call > sends parallel calls back in order with only the first one signed, as Gemini returned them
#[tokio::test]
async fn sends_parallel_calls_back_in_order_with_only_the_first_one_signed() {
    let parts = json!([
        { "functionCall": { "name": "weather", "args": { "city": "Zurich" } }, "thoughtSignature": "sig" },
        { "functionCall": { "name": "local_time", "args": { "city": "Zurich" } } },
        { "functionCall": { "name": "weather", "args": { "city": "Paris" } } },
        { "functionCall": { "name": "local_time", "args": { "city": "Paris" } } }
    ]);
    let message = parse(parts_body(parts.clone()));
    // An earlier turn: the current-turn placeholder never applies, as with `format_tool_call`.
    let payload = render(
        GEMINI,
        vec![Message::user("Weather?"), message, Message::user("Thanks")],
    )
    .await;
    assert_eq!(payload["contents"][1]["parts"], parts);
}
