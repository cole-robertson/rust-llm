//! `spec/ruby_llm/protocols/mistral/{conversations,conversations_live,conversations/images,
//! multi_completion}_spec.rb`: Mistral's Conversations protocol and multi-completion responses.
//! Live examples replay RubyLLM's cassettes; stubbed ones run the real render/parse path.

mod support;

use std::sync::{Arc, Mutex};

use rust_llm::protocols::mistral;
use rust_llm::{
    Chat, Config, Error, Message, PaintOptions, ProtocolName, ProviderTool, Role, paint,
};
use serde_json::{Value, json};
use support::{Cassette, config_for};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const MODEL: &str = "mistral-small-latest";

fn offline_config() -> Arc<Config> {
    let mut c = Config::default();
    c.set("mistral_api_key", "test");
    Arc::new(c)
}

/// `RubyLLM.chat(model: model_for(:mistral), provider: :mistral, protocol: :conversations)`.
fn conversations_with(config: Arc<Config>) -> Chat {
    Chat::with_config(config, Some(MODEL), Some("mistral"), false)
        .unwrap()
        .with_protocol(ProtocolName::Conversations)
}

fn conversations() -> Chat {
    conversations_with(offline_config())
}

fn pending() -> Value {
    json!({ "object": "conversation.response", "conversation_id": "conv_test", "outputs": [
        { "type": "function.call", "id": "fc_test", "tool_call_id": "search_test", "name": "web_search",
          "arguments": "{\"query\":\"Ruby\"}", "confirmation_status": "pending" }
    ], "usage": { "prompt_tokens": 20, "completion_tokens": 5 } })
}

fn parse(data: &Value) -> rust_llm::Result<Message> {
    mistral::parse_completion_body(MODEL, data, None)
}

fn rendered(chat: &mut Chat, prompt: &str) -> Value {
    chat.ask_later(prompt).unwrap();
    chat.render().unwrap()
}

// ---- conversations_spec.rb ----------------------------------------------------------------------

// spec: protocols/mistral/conversations_spec.rb:20
#[test]
fn keeps_chat_completions_as_the_default_and_registers_the_explicit_conversations_protocol() {
    assert_eq!(
        rust_llm::Provider::Mistral.default_protocol(),
        ProtocolName::ChatCompletions
    );
    let payload = rendered(&mut conversations(), "Hello");
    assert_eq!(payload["store"], json!(false));
    assert_eq!(
        payload["inputs"],
        json!([{ "type": "message.input", "role": "user", "content": "Hello" }])
    );
}

// spec: protocols/mistral/conversations_spec.rb:28
#[test]
fn renders_instructions_json_schema_limits_and_temperature_in_their_documented_fields() {
    let schema = json!({ "type": "object", "properties": { "count": { "type": "integer" } }, "required": ["count"] });
    let mut chat = conversations()
        .with_instructions("Be concise")
        .with_schema(schema)
        .with_temperature(0.2)
        .with_max_output_tokens(50);
    let payload = rendered(&mut chat, "Count");
    assert_eq!(payload["instructions"], json!("Be concise"));
    assert_eq!(payload["completion_args"]["temperature"], json!(0.2));
    assert_eq!(payload["completion_args"]["max_tokens"], json!(50));
    assert_eq!(
        payload["completion_args"]["response_format"]["type"],
        json!("json_schema")
    );
}

// spec: protocols/mistral/conversations_spec.rb:37
#[test]
fn deduplicates_the_shared_web_search_and_fetch_tool_while_rendering_every_supported_alias() {
    let mut chat = conversations().with_provider_tools([
        "web_search".into(),
        "web_fetch".into(),
        "code_execution".into(),
        "image_generation".into(),
        ProviderTool::with_options("file_search", json!({ "library_ids": ["library_test"] })),
        ProviderTool::with_options("mcp", json!({ "connector_id": "docs" })),
    ]);
    let payload = rendered(&mut chat, "Hello");
    let types: Vec<&str> = payload["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["type"].as_str().unwrap())
        .collect();
    assert_eq!(
        types,
        [
            "web_search",
            "code_interpreter",
            "image_generation",
            "document_library",
            "connector"
        ]
    );
}

// spec: protocols/mistral/conversations_spec.rb:46
#[test]
fn rejects_hosted_confirmations_before_sending_a_request() {
    let mut chat = conversations().with_provider_tools([ProviderTool::with_options(
        "web_search",
        json!({ "tool_configuration": { "requires_confirmation": ["web_search"] } }),
    )]);
    chat.ask_later("Find Ruby documentation").unwrap();
    let err = chat.render().unwrap_err();
    assert!(
        matches!(&err, Error::Argument(m) if m.contains("require provider conversation storage")),
        "{err:?}"
    );
}

// spec: protocols/mistral/conversations_spec.rb:51
#[test]
fn rejects_unexpected_pending_hosted_confirmations_without_executing_them_as_tools() {
    let err = parse(&pending()).unwrap_err();
    assert!(
        matches!(&err, Error::Api(m, _) if m.contains("requires provider conversation storage")),
        "{err:?}"
    );
}

// spec: protocols/mistral/conversations_spec.rb:56
#[test]
fn replays_edited_history_after_serialization_without_continuing_a_remote_conversation() {
    let mut chat = conversations();
    chat.ask_later("Remember violet").unwrap();
    let mut completed = pending();
    completed["outputs"] = json!([{ "type": "message.output", "content": "OK" }]);
    chat.add_message(parse(&completed).unwrap());
    let mut restored = conversations();
    let mut messages: Vec<Message> = chat
        .messages()
        .iter()
        .map(|m| {
            let mut r = Message::new(m.role, m.content.clone());
            r.raw_content = m
                .raw_content
                .as_ref()
                .map(|v| serde_json::from_str(&v.to_string()).unwrap());
            r
        })
        .collect();
    messages[0].content = Some("Remember orange".into());
    restored.set_messages(messages);
    let mut restored = restored.with_instructions("Be concise");
    let payload = rendered(&mut restored, "What color?");
    assert_eq!(payload["model"], json!(MODEL));
    assert_eq!(payload["store"], json!(false));
    assert_eq!(payload["instructions"], json!("Be concise"));
    let inputs = payload["inputs"].as_array().unwrap();
    assert!(inputs.iter().any(|i| i["content"] == "Remember orange"));
    assert!(inputs.iter().any(|i| i["content"] == "What color?"));
    assert!(inputs.contains(&json!({ "type": "message.output", "content": "OK" })));
    let text = payload.to_string();
    assert!(!text.contains("conv_test") && !text.contains("Remember violet"));
}

// spec: protocols/mistral/conversations_spec.rb:74
#[test]
fn keeps_local_function_calls_separate_from_completed_hosted_executions() {
    let entry = pending()["outputs"][0].clone();
    let mut local = entry.clone();
    local.as_object_mut().unwrap().remove("confirmation_status");
    let mut outputs: Vec<Value> = ["allowed", "denied"]
        .iter()
        .map(|status| {
            let mut e = entry.clone();
            e["confirmation_status"] = json!(status);
            e["tool_call_id"] = json!(status);
            e
        })
        .collect();
    outputs.push(local);
    let mut data = pending();
    data["outputs"] = Value::Array(outputs);
    let message = parse(&data).unwrap();
    let calls = message.tool_calls.as_ref().unwrap();
    assert_eq!(calls.keys().collect::<Vec<_>>(), ["search_test"]);
    assert!(calls.values().all(|c| !c.remote));
    let ids: Vec<Option<&str>> = message
        .server_tool_calls
        .iter()
        .map(|c| c.id.as_deref())
        .collect();
    assert_eq!(ids, [Some("fc_test"), Some("fc_test")]);
}

// spec: protocols/mistral/conversations_spec.rb:85
#[test]
fn parses_hosted_steps_citation_markers_and_connector_input_tokens_without_inflating_output() {
    let data = json!({ "outputs": [
        { "type": "tool.execution", "name": "web_search", "id": "search", "arguments": "{}",
          "info": { "result": "Ruby docs" } },
        { "type": "message.output", "content": [
            { "type": "text", "text": "Ruby docs" },
            { "type": "tool_reference", "url": "https://www.ruby-lang.org", "title": "Ruby" }
        ] }
    ], "usage": { "prompt_tokens": 20, "completion_tokens": 5, "connector_tokens": 100,
                  "total_tokens": 125, "connectors": { "web_search": 1 } } });
    let message = parse(&data).unwrap();
    assert_eq!(message.content(), "Ruby docs");
    assert_eq!(
        (message.tokens.input, message.tokens.output),
        (Some(120), Some(5))
    );
    let citation = &message.citations[0];
    assert_eq!(
        (
            citation.title.as_deref(),
            citation.start_index,
            citation.end_index
        ),
        (Some("Ruby"), Some(9), Some(9))
    );
    let step = &message.server_tool_calls[0];
    assert_eq!(
        (step.name.as_deref(), step.result.clone()),
        (Some("web_search"), Some(json!({ "result": "Ruby docs" })))
    );
    assert_eq!(message.raw_content, Some(data["outputs"].clone()));
}

// spec: protocols/mistral/conversations_spec.rb:103
#[test]
fn keeps_generated_provider_files_as_typed_attachments() {
    let data = json!({ "outputs": [{ "type": "message.output", "content": [
        { "type": "tool_file", "tool": "image_generation", "file_id": "image_test", "file_name": "circle", "file_type": "png" }
    ] }] });
    let message = parse(&data).unwrap();
    let a = &message.attachments[0];
    assert_eq!(a.provider_file_id(), Some("image_test"));
    let rust_llm::attachment::Source::ProviderFile(file) = &a.source else {
        panic!("not a provider file")
    };
    assert_eq!(
        (file.provider.as_str(), file.mime_type.as_deref()),
        ("mistral", Some("image/png"))
    );
}

// spec: protocols/mistral/conversations_spec.rb:113
#[test]
fn replays_actual_hosted_results_without_modifying_the_saved_provider_history() {
    let entries = json!([
        { "type": "tool.execution", "id": "first", "function": "lookup", "arguments": "{}", "info": { "result": false } },
        { "type": "tool.execution", "id": "second", "function": "lookup", "arguments": "{}", "info": { "result": { "answer": 42 } } },
        { "type": "message.output", "content": [{ "type": "tool_reference", "title": "Ruby", "url": "https://www.ruby-lang.org" }] }
    ]);
    let mut message = Message::new(Role::Assistant, None);
    message.raw_content = Some(entries.clone());
    let rendered =
        mistral::format_entries(rust_llm::Provider::Mistral, std::slice::from_ref(&message))
            .unwrap();
    let results: Vec<&Value> = rendered
        .iter()
        .filter(|e| e["type"] == "function.result")
        .map(|e| &e["result"])
        .collect();
    assert_eq!(results, [&json!("false"), &json!("{\"answer\":42}")]);
    assert_ne!(rendered[0]["tool_call_id"], rendered[2]["tool_call_id"]);
    assert_eq!(
        rendered.last().unwrap()["content"],
        json!([{ "type": "text", "text": "[Ruby](https://www.ruby-lang.org)" }])
    );
    assert_eq!(message.raw_content, Some(entries.clone()));
    assert_eq!(entries[0]["type"], json!("tool.execution"));
}

// spec: protocols/mistral/conversations_spec.rb:133
#[tokio::test]
async fn rejects_truncated_conversation_streams_and_reports_provider_stream_errors() {
    let (truncated, _) =
        stream_conversation(vec![json!({ "type": "conversation.response.started" })]).await;
    assert!(
        matches!(&truncated, Err(Error::Api(m, _)) if m.contains("ended before completion")),
        "{truncated:?}"
    );
    let mut state = mistral::ConversationStream::default();
    let err = mistral::build_conversation_chunk(
        MODEL,
        &mut state,
        &json!({ "type": "conversation.response.error", "message": "Tool failed" }),
    );
    assert!(
        matches!(&err, Err(Error::Api(m, _)) if m == "Tool failed"),
        "{err:?}"
    );
}

// spec: protocols/mistral/conversations_spec.rb:141
#[test]
fn accumulates_tool_arguments_text_and_final_usage_from_conversations_events() {
    let events = [
        json!({ "type": "conversation.response.started", "conversation_id": "conv_test" }),
        json!({ "type": "tool.execution.started", "output_index": 0, "id": "exec", "name": "code_interpreter", "arguments": "" }),
        json!({ "type": "tool.execution.delta", "output_index": 0, "arguments": "{\"code\":\"1+1\"}" }),
        json!({ "type": "tool.execution.done", "output_index": 0, "info": { "result": "2" } }),
        json!({ "type": "message.output.delta", "output_index": 1, "content_index": 0, "id": "msg", "content": "Tw" }),
        json!({ "type": "message.output.delta", "output_index": 1, "content_index": 0, "id": "msg", "content": "o" }),
        json!({ "type": "conversation.response.done", "usage": { "prompt_tokens": 10, "completion_tokens": 2, "connector_tokens": 1 } }),
    ];
    let mut state = mistral::ConversationStream::default();
    let chunks: Vec<Message> = events
        .iter()
        .map(|e| mistral::build_conversation_chunk(MODEL, &mut state, e).unwrap())
        .collect();
    assert_eq!(
        chunks
            .iter()
            .filter_map(|c| c.content.clone())
            .collect::<String>(),
        "Two"
    );
    let last = chunks.last().unwrap();
    assert_eq!((last.tokens.input, last.tokens.output), (Some(11), Some(2)));
    assert_eq!(
        last.server_tool_calls[0].input,
        Some(json!("{\"code\":\"1+1\"}"))
    );
    let raw = last.raw_content.as_ref().unwrap().as_array().unwrap();
    assert_eq!(
        raw.last().unwrap()["content"],
        json!([{ "type": "text", "text": "Two" }])
    );
}

/// Serves `events` as one SSE stream and runs a streamed `chat` against it.
async fn stream_events(
    chat: impl FnOnce(Arc<Config>) -> Chat,
    events: Vec<Value>,
) -> (rust_llm::Result<Message>, Vec<Message>) {
    let body: String = events
        .iter()
        .map(|e| format!("data: {e}\n\n"))
        .chain(["data: [DONE]\n\n".to_string()])
        .collect();
    let server = MockServer::start().await;
    Mock::given(wiremock::matchers::any())
        .respond_with(ResponseTemplate::new(200).set_body_raw(body, "text/event-stream"))
        .mount(&server)
        .await;
    let mut config = Config::default();
    config.set("mistral_api_key", "test");
    config.set("mistral_api_base", server.uri());
    config.max_retries = 0;
    let mut chat = chat(Arc::new(config));
    let chunks = Arc::new(Mutex::new(Vec::new()));
    let sink = chunks.clone();
    // One request, like the spec's direct `stream_response` call: a tool call is not run.
    chat.ask_later("Go").unwrap();
    let result = chat
        .step_stream(move |c| sink.lock().unwrap().push(c.clone()))
        .await
        .map(|m| m.expect("a response"));
    let chunks = chunks.lock().unwrap().clone();
    (result, chunks)
}

async fn stream_conversation(events: Vec<Value>) -> (rust_llm::Result<Message>, Vec<Message>) {
    stream_events(conversations_with, events).await
}

// ---- conversations/images_spec.rb ---------------------------------------------------------------

// spec: protocols/mistral/conversations/images_spec.rb:12
#[tokio::test]
async fn routes_paint_through_conversations_while_preserving_the_default_chat_protocol() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/conversations"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "outputs": [] })))
        .mount(&server)
        .await;
    let config = mistral_config(&server);
    let chat = Chat::with_config(config.clone(), Some(MODEL), Some("mistral"), false).unwrap();
    assert_eq!(
        chat.provider()
            .resolve_protocol(None, chat.model(), chat.config())
            .unwrap(),
        ProtocolName::ChatCompletions
    );
    let err = paint(
        "Red circle",
        PaintOptions {
            model: Some(MODEL),
            provider: Some("mistral"),
            config: Some(config),
            ..Default::default()
        },
    )
    .await;
    assert!(
        matches!(&err, Err(Error::Api(m, _)) if m == "Mistral returned no generated image"),
        "{err:?}"
    );
    let sent: Value =
        serde_json::from_slice(&server.received_requests().await.unwrap()[0].body).unwrap();
    assert_eq!(
        sent,
        json!({ "model": MODEL, "store": false, "inputs": "Red circle", "tools": [{ "type": "image_generation" }] })
    );
}

fn mistral_config(server: &MockServer) -> Arc<Config> {
    let mut c = Config::default();
    c.set("mistral_api_key", "test");
    c.set("mistral_api_base", server.uri());
    c.max_retries = 0;
    Arc::new(c)
}

// spec: protocols/mistral/conversations/images_spec.rb:20
#[tokio::test]
async fn downloads_generated_files_from_the_documented_content_endpoint_and_detects_their_actual_type()
 {
    let server = MockServer::start().await;
    let data = json!({ "outputs": [{ "type": "message.output", "content": [
        { "type": "tool_file", "tool": "image_generation", "file_id": "generated", "file_type": "png" }
    ] }], "usage": { "prompt_tokens": 10, "completion_tokens": 3, "connector_tokens": 5 } });
    let bytes: Vec<u8> = b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00".to_vec();
    Mock::given(method("POST"))
        .and(path("/conversations"))
        .respond_with(ResponseTemplate::new(200).set_body_json(data))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/files/generated/content"))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(bytes.clone(), "application/octet-stream"),
        )
        .mount(&server)
        .await;
    let options = PaintOptions {
        model: Some(MODEL),
        provider: Some("mistral"),
        config: Some(mistral_config(&server)),
        ..Default::default()
    };
    let image = paint("Circle", options).await.unwrap().into_image();
    assert_eq!(image.to_blob().await.unwrap(), bytes);
    assert_eq!(image.mime_type.as_deref(), Some("image/jpeg"));
    let tokens = image.tokens();
    assert_eq!((tokens.input, tokens.output), (Some(15), Some(3)));
}

// spec: protocols/mistral/conversations/images_spec.rb:34
#[tokio::test]
async fn rejects_image_controls_that_the_hosted_tool_cannot_honor() {
    let options = |size, count| PaintOptions {
        model: Some(MODEL),
        provider: Some("mistral"),
        config: Some(offline_config()),
        size,
        count,
        ..Default::default()
    };
    let size = paint("Circle", options(Some("1024x1024"), None)).await;
    assert!(
        matches!(&size, Err(Error::Argument(m)) if m.contains("size")),
        "{size:?}"
    );
    let count = paint("Circle", options(None, Some(2))).await;
    assert!(
        matches!(&count, Err(Error::Argument(m)) if m.contains("count")),
        "{count:?}"
    );
}

// spec: protocols/mistral/conversations/images_spec.rb:43
#[tokio::test]
async fn generates_and_downloads_an_image_through_paint() {
    let cassette = Cassette::start(
        "protocols_mistral_conversations_images_generates_and_downloads_an_image_through_paint",
    )
    .await
    .unwrap();
    let options = PaintOptions {
        model: Some(MODEL),
        provider: Some("mistral"),
        config: Some(config_for(&cassette, "mistral")),
        ..Default::default()
    };
    let image = paint(
        "Generate one image of a small red circle on a white background.",
        options,
    )
    .await
    .unwrap()
    .into_image();
    assert!(image.to_blob().await.unwrap().len() > 1000);
    assert!(
        image
            .mime_type
            .as_deref()
            .is_some_and(|m| m.starts_with("image/"))
    );
    let tokens = image.tokens();
    assert!(
        tokens.input.is_some_and(|i| i > 0) && tokens.output.is_some_and(|o| o > 0),
        "{tokens:?}"
    );
    cassette.assert_all_matched().await;
}

// ---- conversations_live_spec.rb -----------------------------------------------------------------

async fn live(name: &str) -> (Cassette, Chat) {
    let cassette = Cassette::start(&format!("protocols_mistral_conversations_{name}"))
        .await
        .unwrap_or_else(|| panic!("missing cassette {name}"));
    let chat = conversations_with(config_for(&cassette, "mistral"));
    (cassette, chat)
}

// spec: protocols/mistral/conversations_live_spec.rb:8
#[tokio::test]
async fn searches_the_web_with_citations_and_replays_hosted_results_in_a_stateless_conversation() {
    let (cassette, chat) = live(
        "searches_the_web_with_citations_and_replays_hosted_results_in_a_stateless_conversation",
    )
    .await;
    let mut chat = chat
        .with_provider_tools(["web_search".into()])
        .with_instructions("Search once for the first request. Answer subsequent requests from the existing results.");
    let response = chat
        .ask("Find the official Ruby 3.4.0 release announcement. Give its date and a citation.")
        .await
        .unwrap();
    assert!(
        response
            .server_tool_calls
            .iter()
            .any(|c| c.name.as_deref() == Some("web_search"))
    );
    assert!(response.citations.iter().any(|c| {
        c.url
            .as_deref()
            .is_some_and(|u| u.contains("ruby-lang.org"))
    }));
    assert!(response.tokens().input.is_some_and(|i| i > 0));
    assert!(
        chat.ask("What version was that announcement for?")
            .await
            .unwrap()
            .content()
            .contains("3.4")
    );
    cassette.assert_all_matched().await;
}

// spec: protocols/mistral/conversations_live_spec.rb:18
#[tokio::test]
async fn fetches_a_public_page_through_the_web_fetch_alias() {
    let (cassette, chat) = live("fetches_a_public_page_through_the_web_fetch_alias").await;
    let mut chat = chat.with_provider_tools(["web_fetch".into()]);
    let response = chat
        .ask("Open https://www.ruby-lang.org/en/news/2024/12/25/ruby-3-4-0-released/ with open_url. Which parser does that release make the default?")
        .await
        .unwrap();
    assert!(
        response.content().to_lowercase().contains("prism"),
        "{:?}",
        response.content
    );
    assert!(
        response
            .server_tool_calls
            .iter()
            .any(|c| c.raw["function"] == "open_url")
    );
    cassette.assert_all_matched().await;
}

// spec: protocols/mistral/conversations_live_spec.rb:27
#[tokio::test]
async fn streams_hosted_python_execution_with_complete_tool_history_and_usage() {
    let (cassette, chat) =
        live("streams_hosted_python_execution_with_complete_tool_history_and_usage").await;
    let mut chat = chat.with_provider_tools(["code_execution".into()]);
    let mut text = String::new();
    let response = chat
        .ask_stream("Use Python to multiply 37 by 19.", |c| {
            text.push_str(c.content())
        })
        .await
        .unwrap();
    assert_eq!(text, response.content());
    assert!(response.content().contains("703"));
    assert!(
        response
            .server_tool_calls
            .iter()
            .any(|c| c.name.as_deref() == Some("code_interpreter"))
    );
    let tokens = response.tokens();
    assert_eq!(
        tokens
            .server_tool_use
            .as_ref()
            .and_then(|s| s.get("code_interpreter")),
        Some(&json!(1))
    );
    assert!(tokens.output.is_some_and(|o| o > 0));
    assert!(
        chat.ask("What was the result?")
            .await
            .unwrap()
            .content()
            .contains("703")
    );
    cassette.assert_all_matched().await;
}

// The example's library setup (create, upload, poll, delete) goes through Faraday directly; this
// port has no public Mistral libraries API, so the test replays those exchanges with raw requests.
// spec: protocols/mistral/conversations_live_spec.rb:40
#[tokio::test]
async fn searches_an_uploaded_document_through_the_file_search_alias() {
    let (cassette, chat) =
        live("searches_an_uploaded_document_through_the_file_search_alias").await;
    let base = format!("{}/v1", cassette.server.uri());
    let client = reqwest::Client::new();
    let library: Value = client
        .post(format!("{base}/libraries"))
        .json(&json!({ "name": "RubyLLM file search integration test" }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let library_id = library["id"].as_str().unwrap().to_string();
    let part = reqwest::multipart::Part::bytes(
        b"The fictional test project is codenamed saffron and has seven paper robots.".to_vec(),
    )
    .file_name("facts.txt")
    .mime_str("text/plain")
    .unwrap();
    let document: Value = client
        .post(format!("{base}/libraries/{library_id}/documents"))
        .multipart(reqwest::multipart::Form::new().part("file", part))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let document_id = document["id"].as_str().unwrap();
    loop {
        let status: Value = client
            .get(format!(
                "{base}/libraries/{library_id}/documents/{document_id}/status"
            ))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        if status["process_status"] == "done" {
            break;
        }
    }
    let mut chat = chat.with_provider_tools([ProviderTool::with_options(
        "file_search",
        json!({ "library_ids": [library_id] }),
    )]);
    let response = chat
        .ask("Search facts.txt in the library. What is the fictional project codename and how many paper robots does it have?")
        .await
        .unwrap();
    client
        .delete(format!("{base}/libraries/{library_id}"))
        .send()
        .await
        .unwrap();
    let content = response.content().to_lowercase();
    assert!(content.contains("saffron"), "{content}");
    assert!(
        content.contains("seven") || content.contains('7'),
        "{content}"
    );
    assert!(
        response
            .server_tool_calls
            .iter()
            .any(|c| c.name.as_deref() == Some("document_library"))
    );
    // The multipart upload body is not JSON; the replay compares its path and method only.
    cassette.assert_all_matched().await;
}

// ---- multi_completion_spec.rb -------------------------------------------------------------------

fn call() -> Value {
    json!({ "id": "imagecall", "type": "function", "function": { "name": "generate_image", "arguments": "{}" },
            "metadata": { "tool_type": "image" } })
}

fn multi_messages() -> Vec<Value> {
    vec![
        json!({ "role": "assistant", "content": "", "tool_calls": [call()], "index": 0 }),
        json!({ "role": "tool", "tool_call_id": "imagecall", "content": "{\"url\":\"https://example.com/image.jpg\"}", "index": 1 }),
        json!({ "role": "assistant", "content": [
            { "type": "text", "text": "Your image." },
            { "type": "image_url", "image_url": "https://example.com/image.jpg" }
        ], "index": 2 }),
    ]
}

fn multi_body(messages: &[Value]) -> Value {
    json!({ "model": MODEL, "choices": [{ "messages": messages, "finish_reason": "stop" }], "usage": {} })
}

/// The default (Chat Completions) Mistral chat's response parsing for `data`, through a request.
async fn complete_with(data: Value) -> (rust_llm::Result<Message>, Chat) {
    let server = MockServer::start().await;
    Mock::given(wiremock::matchers::any())
        .respond_with(ResponseTemplate::new(200).set_body_json(data))
        .mount(&server)
        .await;
    let mut chat =
        Chat::with_config(mistral_config(&server), Some(MODEL), Some("mistral"), false).unwrap();
    let result = chat.ask("Draw").await;
    (result, chat)
}

// spec: protocols/mistral/multi_completion_spec.rb:30
#[tokio::test]
async fn parses_completed_hosted_tools_and_images_without_scheduling_a_local_tool() {
    let messages = multi_messages();
    let (message, mut chat) = complete_with(multi_body(&messages)).await;
    let message = message.unwrap();
    assert_eq!(message.content(), "Your image.");
    assert_eq!(
        message.attachments[0].url(),
        Some("https://example.com/image.jpg")
    );
    assert!(!message.is_tool_call());
    let step = &message.server_tool_calls[0];
    assert_eq!(
        (step.name.as_deref(), step.id.as_deref()),
        (Some("generate_image"), Some("imagecall"))
    );
    assert_eq!(message.raw_content, Some(Value::Array(messages.clone())));
    chat.ask_later("Next").unwrap();
    let replayed = chat.render().unwrap()["messages"]
        .as_array()
        .unwrap()
        .clone();
    let expected: Vec<Value> = messages
        .iter()
        .map(|m| {
            let mut m = m.clone();
            m.as_object_mut().unwrap().remove("index");
            m
        })
        .collect();
    assert_eq!(replayed[1..4], expected[..]);
}

// spec: protocols/mistral/multi_completion_spec.rb:40
#[tokio::test]
async fn keeps_an_unanswered_local_function_call_separate_from_completed_hosted_calls() {
    let mut messages = multi_messages();
    messages[2]["tool_calls"] = json!([{ "id": "localcall", "type": "function", "function": { "name": "calculate", "arguments": "{}" } }]);
    let message = mistral::parse_multi_message(&multi_body(&messages), None)
        .unwrap()
        .unwrap();
    let calls = message.tool_calls.as_ref().unwrap();
    assert_eq!(calls.keys().collect::<Vec<_>>(), ["localcall"]);
    assert!(!calls.values().next().unwrap().remote);
}

// spec: protocols/mistral/multi_completion_spec.rb:49
#[test]
fn refuses_to_execute_an_unfinished_hosted_call_as_a_local_tool() {
    let err = mistral::parse_multi_message(&multi_body(&multi_messages()[..1]), None).unwrap_err();
    assert!(
        matches!(&err, Error::Api(m, _) if m.contains("unfinished hosted tool")),
        "{err:?}"
    );
}

// spec: protocols/mistral/multi_completion_spec.rb:54
#[test]
fn renders_only_the_hosted_tools_supported_by_chat_completions_without_an_invented_request_flag() {
    let mut chat = Chat::with_config(offline_config(), Some(MODEL), Some("mistral"), false)
        .unwrap()
        .with_provider_tools([
            "image_generation".into(),
            ProviderTool::with_options("mcp", json!({ "connector_id": "docs" })),
        ]);
    let payload = rendered(&mut chat, "Draw");
    assert_eq!(
        payload["tools"],
        json!([{ "type": "image_generation" }, { "type": "connector", "connector_id": "docs" }])
    );
    assert!(payload.get("multi_completion").is_none());
}

// spec: protocols/mistral/multi_completion_spec.rb:60
#[tokio::test]
async fn sums_separate_streamed_completions_and_does_not_expose_tool_result_text_as_assistant_output()
 {
    let messages = multi_messages();
    let mut first_call = call();
    first_call["index"] = json!(0);
    let events = vec![
        json!({ "id": "first", "choices": [{ "delta": { "index": 0, "role": "assistant", "tool_calls": [first_call] },
                                             "finish_reason": "tool_calls" }],
                "usage": { "prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12 } }),
        json!({ "id": "first", "choices": [{ "delta": messages[1] }] }),
        json!({ "id": "second", "choices": [{ "delta": messages[2], "finish_reason": "stop" }],
                "usage": { "prompt_tokens": 20, "completion_tokens": 3, "total_tokens": 23 } }),
    ];
    let chat = |config| {
        Chat::with_config(config, Some(MODEL), Some("mistral"), false)
            .unwrap()
            .with_provider_tools(["image_generation".into()])
    };
    let (message, chunks) = stream_events(chat, events).await;
    let message = message.unwrap();
    assert_eq!(
        chunks
            .iter()
            .filter_map(|c| c.content.clone())
            .collect::<String>(),
        "Your image."
    );
    assert_eq!(
        (message.tokens.input, message.tokens.output),
        (Some(30), Some(5))
    );
    assert!(!message.is_tool_call());
    assert_eq!(message.attachments.len(), 1);
}

// spec: protocols/mistral/multi_completion_spec.rb:82
#[tokio::test]
async fn streams_and_downloads_a_hosted_image_through_the_default_chat_api() {
    let name = "protocols_mistral_multicompletion_streams_and_downloads_a_hosted_image_through_the_default_chat_api";
    let cassette = Cassette::start(name).await.unwrap();
    let mut chat = Chat::with_config(
        config_for(&cassette, "mistral"),
        Some(MODEL),
        Some("mistral"),
        false,
    )
    .unwrap()
    .with_provider_tools(["image_generation".into()])
    .with_instructions(
        "Generate the requested image once. Answer follow-up questions from the previous result.",
    );
    let mut text = String::new();
    let response = chat
        .ask_stream(
            "Generate one image of a small blue square on a white background.",
            |c| text.push_str(c.content()),
        )
        .await
        .unwrap();
    assert_eq!(text, response.content());
    assert!(
        response
            .server_tool_calls
            .iter()
            .any(|c| c.name.as_deref() == Some("generate_image"))
    );
    assert!(!response.is_tool_call());
    assert!(
        response.attachments[0]
            .url()
            .is_some_and(|u| u.contains("/image.jpg?"))
    );
    // `attachments.first.content` downloads the image from Azure blob storage. VCR recorded that GET
    // with its query string re-sorted, so the replay fetches the recorded URI from this server.
    let recorded = support::load(name).unwrap()[1].uri.clone();
    let path = recorded
        .split_once("://")
        .and_then(|(_, rest)| rest.split_once('/'))
        .map(|(_, p)| p)
        .unwrap();
    let mut image = rust_llm::Attachment::new(format!("{}/{path}", cassette.server.uri()));
    assert!(image.content().await.unwrap().len() > 1000);
    assert!(response.tokens().input.is_some_and(|i| i > 0));
    assert!(
        chat.ask("What color was the square?")
            .await
            .unwrap()
            .content()
            .to_lowercase()
            .contains("blue")
    );
    cassette.assert_all_matched().await;
}
