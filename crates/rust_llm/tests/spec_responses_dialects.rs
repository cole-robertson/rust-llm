//! Responses dialect specs ported from RubyLLM 2.0: `providers/deepseek/responses_spec.rb`,
//! `providers/xai/responses_spec.rb`, and the counting-payload parts of
//! `providers/openai/responses_spec.rb`. Ruby calls the protocol's private `render_payload`,
//! `parse_completion_body`, `build_chunk`, and `parse_usage`; these go through `Chat#render`,
//! `ask`, `ask_stream`, and `count_tokens` against a mock server, which run the same code.

mod spec_helpers;

use std::sync::Arc;

use rust_llm::{
    Attachment, Chat, Config, Error, FnTool, Message, ProtocolName, ProviderTool, Role,
    ThinkingConfig, ToolCalls, ToolChoice, ToolResult,
};
use serde_json::{Value, json};
use spec_helpers::*;
use wiremock::MockServer;

fn fixture(name: &str) -> String {
    format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
}

fn inline(name: &str) -> Attachment {
    Attachment::from_bytes(std::fs::read(fixture(name)).unwrap(), name, None)
}

fn uploaded(id: &str, filename: &str, mime: &str) -> Attachment {
    rust_llm::UploadedFile {
        id: id.into(),
        provider: "deepseek".into(),
        filename: Some(filename.into()),
        byte_size: None,
        created_at: None,
        expires_at: None,
        status: None,
        mime_type: Some(mime.into()),
        purpose: None,
        uri: None,
        downloadable: None,
        metadata: Value::Null,
    }
    .into()
}

/// `include_context 'with configured RubyLLM'`, plus xAI pointed at `server`.
fn config_with_xai(server: &MockServer) -> Arc<Config> {
    let mut c = (*config(server)).clone();
    c.set("xai_api_base", server.uri());
    c.set("xai_api_key", "test");
    Arc::new(c)
}

/// `RubyLLM.chat(model: model_for(:deepseek), provider: :deepseek, protocol: :responses)`.
fn deepseek(server: &MockServer) -> Chat {
    Chat::with_config(
        config(server),
        Some("deepseek-v4-flash"),
        Some("deepseek"),
        false,
    )
    .unwrap()
    .with_protocol(ProtocolName::Responses)
}

fn xai(server: &MockServer) -> Chat {
    Chat::with_config(
        config_with_xai(server),
        Some("grok-4-1-fast-non-reasoning"),
        Some("xai"),
        false,
    )
    .unwrap()
}

fn openai(server: &MockServer) -> Chat {
    Chat::with_config(config(server), Some("gpt-5-nano"), Some("openai"), false).unwrap()
}

fn render_input(chat: &Chat) -> rust_llm::Result<Value> {
    chat.render().map(|p| p["input"].clone())
}

// ---- providers/deepseek/responses_spec.rb ------------------------------------------------------

// spec: providers/deepseek/responses_spec.rb:21 rejects the unsupported web search alias before sending a request
#[tokio::test]
async fn deepseek_rejects_the_web_search_alias_before_sending() {
    let server = serve(vec![]).await;
    let chat = deepseek(&server).with_provider_tools(["web_search".into()]);
    let Err(Error::UnsupportedServerTool(message)) = chat.render() else {
        panic!("expected UnsupportedServerTool")
    };
    let web_search = message.find(":web_search").expect(":web_search in message");
    assert!(message[web_search..].contains(":apply_patch"), "{message}");
    assert_eq!(requests(&server).await, 0);
}

// spec: providers/deepseek/responses_spec.rb:27 keeps the patch tool alias
#[tokio::test]
async fn deepseek_keeps_the_patch_tool_alias() {
    let server = serve(vec![]).await;
    let payload = deepseek(&server)
        .with_provider_tools(["apply_patch".into()])
        .render()
        .unwrap();
    assert_eq!(
        payload["tools"],
        json!([{ "type": "custom", "name": "apply_patch" }])
    );
}

// spec: providers/deepseek/responses_spec.rb:33 passes raw tool definitions through unchanged
#[tokio::test]
async fn deepseek_passes_raw_tool_definitions_through() {
    let server = serve(vec![]).await;
    let definition = json!({ "type": "web_search" });
    let payload = deepseek(&server)
        .with_provider_tools([ProviderTool::raw(definition.clone())])
        .render()
        .unwrap();
    assert_eq!(payload["tools"], json!([definition]));
}

// spec: providers/deepseek/responses_spec.rb:41 sends JSON Schema through the Responses text format
#[tokio::test]
async fn deepseek_sends_json_schema_through_the_text_format() {
    let server = serve(vec![]).await;
    let inner = json!({ "type": "object", "properties": { "name": { "type": "string" } } });
    let mut chat = deepseek(&server).with_schema(json!({ "name": "person", "schema": inner }));
    chat.ask_later("Extract the name: Ruby").unwrap();
    let format = chat.render().unwrap()["text"]["format"].clone();
    assert_eq!(format["type"], json!("json_schema"));
    assert_eq!(format["name"], json!("person"));
    assert_eq!(format["schema"], inner);
}

// spec: providers/deepseek/responses_spec.rb:50 renders inline images as input_image parts
#[tokio::test]
async fn deepseek_renders_inline_images_as_input_image() {
    let server = serve(vec![]).await;
    let mut chat = deepseek(&server);
    chat.ask_later_with("Describe this image", vec![inline("ruby.png")])
        .unwrap();
    let input = render_input(&chat).unwrap();
    let part = input[0]["content"]
        .as_array()
        .unwrap()
        .last()
        .unwrap()
        .clone();
    assert_eq!(part["type"], json!("input_image"));
    assert!(
        part["image_url"]
            .as_str()
            .unwrap()
            .starts_with("data:image/png;base64,")
    );
}

// spec: providers/deepseek/responses_spec.rb:60 renders uploaded images as input_image references
#[tokio::test]
async fn deepseek_renders_uploaded_images_as_input_image_references() {
    let server = serve(vec![]).await;
    let mut chat = deepseek(&server);
    chat.ask_later_with(
        "Read the screenshot",
        vec![uploaded("file-api-image", "screenshot.png", "image/png")],
    )
    .unwrap();
    assert_eq!(
        render_input(&chat).unwrap()[0]["content"],
        json!([
            { "type": "input_text", "text": "Read the screenshot" },
            { "type": "input_image", "file_id": "file-api-image" }
        ])
    );
}

// spec: providers/deepseek/responses_spec.rb:71 keeps image attachments inside their function call output
#[tokio::test]
async fn deepseek_keeps_image_attachments_inside_the_function_call_output() {
    let server = serve(vec![]).await;
    let mut chat = deepseek(&server);
    chat.add_message(
        Message::tool_result("call_1", "Screenshot").with_attachments(vec![uploaded(
            "file-api-image",
            "screenshot.png",
            "image/png",
        )]),
    );
    assert_eq!(
        render_input(&chat).unwrap(),
        json!([{
            "type": "function_call_output", "call_id": "call_1",
            "output": [{ "type": "input_text", "text": "Screenshot" }, { "type": "input_image", "file_id": "file-api-image" }]
        }])
    );
}

// spec: providers/deepseek/responses_spec.rb:83 rejects inline documents that DeepSeek cannot read
#[tokio::test]
async fn deepseek_rejects_inline_documents() {
    let server = serve(vec![]).await;
    let mut chat = deepseek(&server);
    chat.ask_later_with("Read this", vec![inline("sample.pdf")])
        .unwrap();
    let Err(Error::UnsupportedAttachment(message)) = chat.render() else {
        panic!("expected UnsupportedAttachment")
    };
    assert!(message.contains("application/pdf"), "{message}");
}

// spec: providers/deepseek/responses_spec.rb:90 rejects uploaded documents that DeepSeek cannot read
#[tokio::test]
async fn deepseek_rejects_uploaded_documents() {
    let server = serve(vec![]).await;
    let mut chat = deepseek(&server);
    chat.ask_later_with(
        "Read this",
        vec![uploaded("file-api-pdf", "report.pdf", "application/pdf")],
    )
    .unwrap();
    assert!(matches!(
        chat.render(),
        Err(Error::UnsupportedAttachment(_))
    ));
}

// spec: providers/deepseek/responses_spec.rb:118 replays server-tool history without duplicating its reasoning
#[tokio::test]
async fn deepseek_replays_server_tool_history_without_duplicating_reasoning() {
    let output = json!([
        { "type": "reasoning", "content": [{ "type": "reasoning_text", "text": "Search first." }] },
        { "type": "web_search_call", "id": "search_1", "action": { "type": "search", "query": "Ruby" } }
    ]);
    let server = serve(vec![json!({ "status": "completed", "output": output })]).await;
    let mut chat = deepseek(&server);
    let message = chat.ask("Search").await.unwrap();
    assert_eq!(
        message.thinking.as_ref().and_then(|t| t.text.as_deref()),
        Some("Search first.")
    );
    chat.set_messages(vec![message]);
    assert_eq!(render_input(&chat).unwrap(), output);
}

// ---- providers/openai/responses_spec.rb ---------------------------------------------------------

async fn counted_body(server: &MockServer) -> Value {
    let requests = server.received_requests().await.unwrap_or_default();
    let request = requests.last().expect("a count request");
    assert!(
        request.url.path().ends_with("/responses/input_tokens"),
        "{}",
        request.url
    );
    serde_json::from_slice(&request.body).unwrap()
}

// spec: providers/openai/responses_spec.rb:23 counts instructions, function tools, schemas and reasoning without generation-only options
#[tokio::test]
async fn openai_count_payload_keeps_instructions_tools_schema_and_reasoning_only() {
    let server = serve(vec![
        json!({ "object": "response.input_tokens", "input_tokens": 42 }),
    ])
    .await;
    let weather = FnTool::new("weather", "Looks up weather", |_| async {
        Ok(ToolResult::from("sunny"))
    });
    let inner = json!({ "type": "object", "properties": { "answer": { "type": "string" } } });
    let mut chat = openai(&server)
        .with_instructions("Be concise.")
        .with_tool(weather)
        .with_tool_choice(ToolChoice::Required)
        .unwrap()
        .with_tool_calls(ToolCalls::One)
        .with_schema(json!({ "name": "answer", "schema": inner, "strict": true }))
        .with_thinking(ThinkingConfig::effort("low"))
        .with_caching(json!({ "key": "weather" }))
        .unwrap();
    chat.ask_later("Weather?").unwrap();
    assert_eq!(chat.count_tokens(None).await.unwrap(), 42);
    let payload = counted_body(&server).await;
    assert_eq!(payload["model"], json!("gpt-5-nano"));
    assert_eq!(payload["instructions"], json!("Be concise."));
    assert_eq!(
        payload["input"],
        json!([{ "role": "user", "content": "Weather?" }])
    );
    assert_eq!(payload["tools"].as_array().unwrap().len(), 1);
    assert_eq!(payload["tools"][0]["type"], json!("function"));
    assert_eq!(payload["tools"][0]["name"], json!("weather"));
    assert_eq!(payload["tool_choice"], json!("required"));
    assert_eq!(payload["parallel_tool_calls"], json!(false));
    assert_eq!(payload["reasoning"], json!({ "effort": "low" }));
    assert_eq!(
        payload["text"],
        json!({ "format": { "type": "json_schema", "name": "answer", "schema": inner, "strict": true } })
    );
    let mut keys: Vec<&str> = payload
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        [
            "input",
            "instructions",
            "model",
            "parallel_tool_calls",
            "reasoning",
            "text",
            "tool_choice",
            "tools"
        ]
    );
}

// spec: providers/openai/responses_spec.rb:47 keeps file references and image inputs in the counting payload
#[tokio::test]
async fn openai_count_payload_keeps_file_references_and_images() {
    let server = serve(vec![
        json!({ "object": "response.input_tokens", "input_tokens": 7 }),
    ])
    .await;
    let mut chat = openai(&server);
    let file = uploaded("file_123", "proposal.pdf", "application/pdf");
    chat.ask_later_with(
        "Compare these",
        vec![file, Attachment::new(fixture("ruby.png"))],
    )
    .unwrap();
    chat.count_tokens(None).await.unwrap();
    let content = counted_body(&server).await["input"][0]["content"]
        .as_array()
        .unwrap()
        .clone();
    assert_eq!(content.len(), 3);
    assert!(content.contains(&json!({ "type": "input_text", "text": "Compare these" })));
    assert!(content.contains(&json!({ "type": "input_file", "file_id": "file_123" })));
    assert!(content.iter().any(|p| {
        p["type"] == "input_image"
            && p["image_url"]
                .as_str()
                .is_some_and(|u| u.starts_with("data:image/png;base64,"))
    }));
}

// spec: providers/openai/responses_spec.rb:63 does not enable OpenAI token counting on #{dialect}
#[tokio::test]
async fn xai_and_deepseek_responses_do_not_count_tokens() {
    let server = serve(vec![]).await;
    for (chat, name) in [(xai(&server), "XAI"), (deepseek(&server), "DeepSeek")] {
        let Err(Error::Api(message, _)) = chat.count_tokens(Some("Hi")).await else {
            panic!("{name}: expected an error")
        };
        assert_eq!(message, format!("{name} doesn't support token counting"));
    }
    assert_eq!(requests(&server).await, 0);
}

// ---- providers/xai/responses_spec.rb ------------------------------------------------------------

const COLLECTION_SOURCE: &str = "collections://collection_1/files/file_facts";

fn assert_citation(c: &rust_llm::Citation, url: &str, index: i64) {
    assert_eq!((c.url.as_deref(), c.source_index), (Some(url), Some(index)));
}

// spec: providers/xai/responses_spec.rb:15 preserves collection source references alongside web citations
#[tokio::test]
async fn xai_preserves_collection_sources_alongside_web_citations() {
    let server =
        serve(vec![json!({ "status": "completed", "output": [], "citations": [COLLECTION_SOURCE, "https://ruby-lang.org"] })]).await;
    let message = xai(&server).ask("Facts?").await.unwrap();
    assert_eq!(message.citations.len(), 2);
    assert_citation(&message.citations[0], COLLECTION_SOURCE, 0);
    assert_citation(&message.citations[1], "https://ruby-lang.org", 1);
}

// spec: providers/xai/responses_spec.rb:29 preserves collection citations from completed streams
#[tokio::test]
async fn xai_preserves_collection_citations_from_completed_streams() {
    let event = json!({
        "type": "response.completed",
        "response": { "status": "completed", "output": [], "citations": [COLLECTION_SOURCE] }
    });
    let server = serve_templates(vec![sse(format!(
        "event: response.completed\ndata: {event}\n\n"
    ))])
    .await;
    let chunks = log::<Message>();
    let sink = chunks.clone();
    xai(&server)
        .ask_stream("Facts?", move |c| sink.lock().unwrap().push(c.clone()))
        .await
        .unwrap();
    let chunks = chunks.lock().unwrap();
    let chunk = chunks
        .iter()
        .find(|c| !c.citations.is_empty())
        .expect("a chunk with citations");
    assert_citation(&chunk.citations[0], COLLECTION_SOURCE, 0);
}

// spec: providers/xai/responses_spec.rb:45 #parse_usage > converts cost_in_usd_ticks into a reported cost in dollars
#[tokio::test]
async fn xai_converts_cost_in_usd_ticks_into_dollars() {
    let usage = json!({ "input_tokens": 10, "output_tokens": 5, "cost_in_usd_ticks": 2_909_000 });
    let server = serve(vec![
        json!({ "status": "completed", "output": [], "usage": usage }),
    ])
    .await;
    let cost = xai(&server)
        .ask("Hi")
        .await
        .unwrap()
        .tokens
        .reported_cost
        .expect("a reported cost");
    assert!((cost - 0.0002909).abs() <= 1e-12, "{cost}");
}

// spec: providers/xai/responses_spec.rb:52 #parse_usage > leaves reported cost nil when ticks are absent
#[tokio::test]
async fn xai_leaves_reported_cost_nil_without_ticks() {
    let server = serve(vec![
        json!({ "status": "completed", "output": [], "usage": { "input_tokens": 10 } }),
    ])
    .await;
    let message = xai(&server).ask("Hi").await.unwrap();
    assert_eq!(message.role, Role::Assistant);
    assert_eq!(message.tokens.reported_cost, None);
}
