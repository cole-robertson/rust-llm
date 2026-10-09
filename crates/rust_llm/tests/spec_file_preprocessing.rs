//! Ports of `spec/ruby_llm/protocol_file_preprocessing_spec.rb`: request-time auto-upload of
//! oversized attachments (`Protocol#preprocess_message`). The Ruby specs stub
//! `provider.upload_file` and inflate `byte_size`; here the attachments really are over each
//! protocol's inline threshold and a mock server plays the provider's Files API and chat
//! endpoint, so the requests show what was uploaded, how often, and what the chat referenced.

use std::sync::Arc;

use rust_llm::{Attachment, Chat, Config, Error, Message, ProtocolName};
use serde_json::{Value, json};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

const MB: usize = 1024 * 1024;

fn large(filename: &str, size: usize) -> Attachment {
    Attachment::from_bytes(vec![b' '; size], filename, None)
}

/// Every provider this file uses, pointed at `server`, with `openai_api_key = openai_key`.
fn config(server: &MockServer, openai_key: &str) -> Arc<Config> {
    let mut c = Config::default();
    for (provider, base) in [
        ("anthropic", ""),
        ("openai", "/v1"),
        ("gemini", "/v1beta"),
        ("openrouter", "/api/v1"),
    ] {
        c.set(
            format!("{provider}_api_base"),
            format!("{}{base}", server.uri()),
        );
        c.set(format!("{provider}_api_key"), "test");
    }
    c.set("openai_api_key", openai_key);
    c.max_retries = 0;
    Arc::new(c)
}

fn chat(config: &Arc<Config>, model: &str, provider: &str) -> Chat {
    Chat::with_config(config.clone(), Some(model), Some(provider), false).unwrap()
}

fn body(request: &Request) -> Value {
    serde_json::from_slice(&request.body).unwrap()
}

fn paths(requests: &[Request]) -> Vec<String> {
    requests.iter().map(|r| r.url.path().to_string()).collect()
}

/// The value of a text field in a multipart body.
fn form_field(request: &Request, name: &str) -> Option<String> {
    let marker = format!("name=\"{name}\"\r\n\r\n");
    let text = String::from_utf8_lossy(&request.body);
    let start = text.find(&marker)? + marker.len();
    Some(text[start..start + text[start..].find("\r\n")?].to_string())
}

async fn gemini_answers(server: &MockServer) {
    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-2.5-flash:generateContent"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates": [{ "content": { "role": "model", "parts": [{ "text": "ok" }] }, "finishReason": "STOP" }],
            "usageMetadata": { "promptTokenCount": 1, "candidatesTokenCount": 1 }
        })))
        .mount(server)
        .await;
}

/// Gemini's resumable upload: the start request hands back `session`, which stores `file`.
/// Each start mock answers once, so successive uploads get successive files.
async fn gemini_upload(server: &MockServer, session: &str, file: Value) {
    Mock::given(method("POST"))
        .and(path("/upload/v1beta/files"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("x-goog-upload-url", format!("{}/{session}", server.uri())),
        )
        .up_to_n_times(1)
        .mount(server)
        .await;
    Mock::given(method("POST"))
        .and(path(format!("/{session}")))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "file": file })))
        .mount(server)
        .await;
}

async fn openai_answers(server: &MockServer) {
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "resp_1", "object": "response", "status": "completed", "model": "gpt-5-nano",
            "output": [{ "type": "message", "role": "assistant", "content": [{ "type": "output_text", "text": "ok" }] }],
            "usage": { "input_tokens": 1, "output_tokens": 1 }
        })))
        .mount(server)
        .await;
}

fn openai_file(id: &str, filename: &str) -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(json!({
        "id": id, "object": "file", "filename": filename, "bytes": 1, "purpose": "user_data",
        "created_at": 1_700_000_000, "status": "processed"
    }))
}

/// The `file_id` the Responses request's first user message referenced.
fn responses_file_id(request: &Request) -> Value {
    body(request)["input"][0]["content"][1]["file_id"].clone()
}

// spec: protocol_file_preprocessing_spec.rb:72 replaces an upload past its retention window
#[tokio::test]
async fn an_upload_past_its_retention_window_is_replaced() {
    let server = MockServer::start().await;
    let expiring = chrono::Utc::now() + chrono::Duration::seconds(30);
    let fresh = chrono::Utc::now() + chrono::Duration::hours(48);
    gemini_upload(
        &server,
        "session-old",
        json!({ "name": "files/old", "displayName": "clip.mp4", "mimeType": "video/mp4",
                "expirationTime": expiring.to_rfc3339() }),
    )
    .await;
    gemini_upload(
        &server,
        "session-new",
        json!({ "name": "files/new", "displayName": "clip.mp4", "mimeType": "video/mp4",
                "expirationTime": fresh.to_rfc3339() }),
    )
    .await;
    gemini_answers(&server).await;
    let config = config(&server, "test");
    let mut chat = chat(&config, "gemini-2.5-flash", "gemini");

    chat.ask_with("Watch this", vec![large("clip.mp4", 25 * MB)])
        .await
        .unwrap();
    chat.ask("And again").await.unwrap();

    let requests = server.received_requests().await.unwrap();
    let referenced: Vec<Value> = requests
        .iter()
        .filter(|r| r.url.path().ends_with(":generateContent"))
        .map(|r| body(r)["contents"][0]["parts"][1]["file_data"]["file_uri"].clone())
        .collect();
    assert_eq!(referenced, [json!("files/old"), json!("files/new")]);
    let starts = paths(&requests)
        .iter()
        .filter(|p| *p == "/upload/v1beta/files")
        .count();
    assert_eq!(starts, 2, "uploaded twice");
}

// spec: protocol_file_preprocessing_spec.rb:90 uploads separately for each provider
#[tokio::test]
async fn each_provider_gets_its_own_upload() {
    let server = MockServer::start().await;
    gemini_upload(
        &server,
        "session-abc",
        json!({ "name": "files/abc", "displayName": "report.pdf", "mimeType": "application/pdf",
                "uri": "https://example.test/files/abc" }),
    )
    .await;
    gemini_answers(&server).await;
    Mock::given(method("POST"))
        .and(path("/v1/files"))
        .respond_with(openai_file("file_123", "report.pdf"))
        .mount(&server)
        .await;
    openai_answers(&server).await;
    let config = config(&server, "test");
    let attachment = large("report.pdf", 60 * MB);

    let mut from_gemini = chat(&config, "gemini-2.5-flash", "gemini");
    from_gemini
        .ask_with("Summarize this", vec![attachment.clone()])
        .await
        .unwrap();
    let mut from_openai =
        chat(&config, "gpt-5-nano", "openai").with_protocol(ProtocolName::Responses);
    from_openai
        .ask_with("Summarize this", vec![attachment.clone()])
        .await
        .unwrap();

    let requests = server.received_requests().await.unwrap();
    let gemini_chat = requests
        .iter()
        .find(|r| r.url.path().ends_with(":generateContent"))
        .unwrap();
    assert_eq!(
        body(gemini_chat)["contents"][0]["parts"][1]["file_data"]["file_uri"],
        "https://example.test/files/abc"
    );
    let openai_chat = requests
        .iter()
        .find(|r| r.url.path() == "/v1/responses")
        .unwrap();
    assert_eq!(responses_file_id(openai_chat), "file_123");

    // `attachment.provider_uploads` holds both: asking each provider again uploads nothing.
    from_gemini.ask("Again").await.unwrap();
    from_openai.ask("Again").await.unwrap();
    let requests = server.received_requests().await.unwrap();
    let uploads: Vec<String> = paths(&requests)
        .into_iter()
        .filter(|p| p == "/upload/v1beta/files" || p == "/v1/files")
        .collect();
    assert_eq!(uploads, ["/upload/v1beta/files", "/v1/files"]);
}

// spec: protocol_file_preprocessing_spec.rb:114 uploads again for the same provider under different credentials
#[tokio::test]
async fn the_same_provider_under_other_credentials_uploads_again() {
    let server = MockServer::start().await;
    for (index, key) in ["test", "sk-other-tenant"].iter().enumerate() {
        Mock::given(method("POST"))
            .and(path("/v1/files"))
            .and(header("authorization", format!("Bearer {key}").as_str()))
            .respond_with(openai_file(&format!("file_{index}"), "report.pdf"))
            .expect(1)
            .mount(&server)
            .await;
    }
    openai_answers(&server).await;
    let attachment = large("report.pdf", 60 * MB);

    let mut ids = Vec::new();
    for key in ["test", "sk-other-tenant"] {
        let mut tenant = chat(&config(&server, key), "gpt-5-nano", "openai")
            .with_protocol(ProtocolName::Responses);
        tenant
            .ask_with("Summarize this", vec![attachment.clone()])
            .await
            .unwrap();
        let requests = server.received_requests().await.unwrap();
        let last = requests
            .iter()
            .rev()
            .find(|r| r.url.path() == "/v1/responses")
            .unwrap();
        ids.push(responses_file_id(last));
    }
    assert_eq!(ids, [json!("file_0"), json!("file_1")]);
}

/// A Responses chat that auto-uploads `filename`; returns the upload and chat requests.
async fn responses_auto_upload(filename: &str, id: &str) -> (Request, Request) {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/files"))
        .respond_with(openai_file(id, filename))
        .expect(1)
        .mount(&server)
        .await;
    openai_answers(&server).await;
    let config = config(&server, "test");
    let mut chat = chat(&config, "gpt-5-nano", "openai").with_protocol(ProtocolName::Responses);
    chat.ask_with("Summarize this", vec![large(filename, 60 * MB)])
        .await
        .unwrap();
    let requests = server.received_requests().await.unwrap();
    assert_eq!(paths(&requests), ["/v1/files", "/v1/responses"]);
    (requests[0].clone(), requests[1].clone())
}

// spec: protocol_file_preprocessing_spec.rb:148 uses OpenAI user_data purpose for automatic Responses uploads
#[tokio::test]
async fn responses_auto_uploads_use_the_user_data_purpose() {
    let (upload, chat) = responses_auto_upload("large.pdf", "file_123").await;
    assert_eq!(form_field(&upload, "purpose").as_deref(), Some("user_data"));
    assert_eq!(responses_file_id(&chat), "file_123");
}

// spec: protocol_file_preprocessing_spec.rb:168 uploads oversized Responses documents beyond PDFs
#[tokio::test]
async fn responses_auto_uploads_documents_beyond_pdfs() {
    let (upload, chat) = responses_auto_upload("large.docx", "file_456").await;
    assert_eq!(form_field(&upload, "purpose").as_deref(), Some("user_data"));
    assert_eq!(responses_file_id(&chat), "file_456");
}

// UPSTREAM-REMOVED in 2.1 (was spec: protocol_file_preprocessing_spec.rb:168) raises before uploading files above the provider file limit
#[tokio::test]
async fn files_above_the_provider_limit_raise_before_uploading() {
    let server = MockServer::start().await;
    let config = config(&server, "test");
    let mut chat = chat(&config, "anthropic/claude-haiku-4.5", "openrouter")
        .with_protocol(ProtocolName::ChatCompletions);
    let err = chat
        .ask_with("Summarize this", vec![large("huge.pdf", 101 * MB)])
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Api(..)), "{err:?}");
    assert!(
        err.to_string()
            .starts_with("OpenRouter file uploads support files up to"),
        "{err}"
    );
    assert!(server.received_requests().await.unwrap().is_empty());
}

async fn anthropic(server: &MockServer) {
    Mock::given(method("POST"))
        .and(path("/v1/files"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "type": "file", "id": "file_large", "size_bytes": 25165825, "filename": "large.pdf",
            "mime_type": "application/pdf", "downloadable": false, "created_at": "2026-09-17T22:14:58Z"
        })))
        .mount(server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-haiku-4-5",
            "content": [{ "type": "text", "text": "ok" }], "stop_reason": "end_turn",
            "usage": { "input_tokens": 1, "output_tokens": 1 }
        })))
        .mount(server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/messages/count_tokens"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "input_tokens": 1 })))
        .mount(server)
        .await;
}

// spec: protocol_file_preprocessing_spec.rb:200 preprocesses at request time rather than when messages are added
#[tokio::test]
async fn preprocessing_happens_at_request_time_not_when_adding_messages() {
    let server = MockServer::start().await;
    anthropic(&server).await;
    let config = config(&server, "test");
    let mut chat = chat(&config, "claude-haiku-4-5", "anthropic");

    chat.add_message(Message::user("hi").with_attachments(vec![large("large.pdf", 24 * MB + 1)]));
    assert!(
        server.received_requests().await.unwrap().is_empty(),
        "adding a message uploads nothing"
    );

    chat.complete().await.unwrap();
    let requests = server.received_requests().await.unwrap();
    assert_eq!(paths(&requests), ["/v1/files", "/v1/messages"]);
    assert_eq!(
        body(&requests[1])["messages"][0]["content"][1]["source"],
        json!({ "type": "file", "file_id": "file_large" })
    );
}

// spec: protocol_file_preprocessing_spec.rb:211 preprocesses the messages it counts tokens for
#[tokio::test]
async fn count_tokens_preprocesses_the_messages_it_counts() {
    let server = MockServer::start().await;
    anthropic(&server).await;
    let config = config(&server, "test");
    let mut chat = chat(&config, "claude-haiku-4-5", "anthropic");
    chat.add_message(Message::user("hi").with_attachments(vec![large("large.pdf", 24 * MB + 1)]));

    assert_eq!(chat.count_tokens(None).await.unwrap(), 1);

    let requests = server.received_requests().await.unwrap();
    assert_eq!(paths(&requests), ["/v1/files", "/v1/messages/count_tokens"]);
    assert_eq!(
        body(&requests[1])["messages"][0]["content"][1]["source"],
        json!({ "type": "file", "file_id": "file_large" })
    );
    assert!(
        !chat.messages()[0].attachments[0].is_provider_file(),
        "history keeps the original attachment"
    );
}
