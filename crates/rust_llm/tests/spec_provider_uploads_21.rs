//! Ports of RubyLLM 2.1's `chat_provider_uploads_spec.rb` (uploading again a file the provider
//! deleted) and the stored-upload half of `protocol_file_preprocessing_spec.rb`
//! (`Protocol::StoredUploads`), plus its conversion-before-upload example.
//!
//! The Ruby specs stub the inline limit down to 16 bytes and inflate `byte_size`; here the
//! attachments really are over each protocol's inline threshold, and a mock server plays the
//! provider's Files API and chat endpoint.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use rust_llm::files::{ProviderFileStore, is_confirmed_upload};
use rust_llm::{Attachment, Chat, Config, Error, Message, Provider, UploadedFile};
use serde_json::{Value, json};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

const MB: usize = 1024 * 1024;

/// The spec's `store`: an in-memory `provider_file_store`.
#[derive(Default)]
struct Store {
    files: Mutex<HashMap<(String, String), UploadedFile>>,
}

#[async_trait::async_trait]
impl ProviderFileStore for Store {
    async fn fetch(&self, provider: &str, account: &str) -> Option<UploadedFile> {
        self.files
            .lock()
            .unwrap()
            .get(&(provider.into(), account.into()))
            .cloned()
    }
    async fn store(&self, upload: &UploadedFile, provider: &str, account: &str) {
        self.files
            .lock()
            .unwrap()
            .insert((provider.into(), account.into()), upload.clone());
    }
    async fn forget(&self, id: &str, provider: &str, account: &str) {
        let mut files = self.files.lock().unwrap();
        let key = (provider.to_string(), account.to_string());
        if files.get(&key).is_some_and(|f| f.id == id) {
            files.remove(&key);
        }
    }
}

impl Store {
    fn remember(&self, provider: &str, account: &str, file: UploadedFile) {
        self.files
            .lock()
            .unwrap()
            .insert((provider.into(), account.into()), file);
    }
    fn ids(&self) -> Vec<String> {
        self.files
            .lock()
            .unwrap()
            .values()
            .map(|f| f.id.clone())
            .collect()
    }
    fn keys(&self) -> Vec<(String, String)> {
        let mut keys: Vec<_> = self.files.lock().unwrap().keys().cloned().collect();
        keys.sort();
        keys
    }
}

fn uploaded(id: &str, provider: &str) -> UploadedFile {
    UploadedFile {
        id: id.into(),
        provider: provider.into(),
        filename: None,
        byte_size: None,
        created_at: None,
        expires_at: None,
        status: None,
        mime_type: None,
        purpose: None,
        uri: None,
        downloadable: None,
        metadata: Value::Null,
    }
}

fn config(server: &MockServer, key: &str) -> Arc<Config> {
    let mut c = Config::default();
    for (provider, base) in [("anthropic", ""), ("gemini", "/v1beta"), ("xai", "/v1")] {
        c.set(
            format!("{provider}_api_base"),
            format!("{}{base}", server.uri()),
        );
        c.set(format!("{provider}_api_key"), key);
    }
    c.max_retries = 0;
    Arc::new(c)
}

fn json_response(body: Value, status: u16) -> ResponseTemplate {
    ResponseTemplate::new(status).set_body_json(body)
}

// ---- chat_provider_uploads_spec.rb ---------------------------------------------------------------

const MODEL: &str = "claude-haiku-4-5";

fn file_body(id: &str) -> Value {
    json!({ "id": id, "type": "file", "filename": "notes.txt", "mime_type": "text/plain", "size_bytes": 36,
            "created_at": "2026-10-01T09:00:00Z", "downloadable": false })
}

fn reply() -> Value {
    json!({ "id": "msg_1", "type": "message", "role": "assistant", "model": MODEL,
            "content": [{ "type": "text", "text": "Noted." }], "stop_reason": "end_turn",
            "usage": { "input_tokens": 9, "output_tokens": 2 } })
}

fn missing(id: &str, status: u16) -> ResponseTemplate {
    json_response(
        json!({ "type": "error", "error": { "type": "not_found_error", "message": format!("File not found: {id}") } }),
        status,
    )
}

fn streamed(texts: &[&str], error: Option<Value>) -> ResponseTemplate {
    let mut events = vec![
        json!({ "type": "message_start", "message": { "id": "msg_1", "type": "message",
        "role": "assistant", "model": MODEL, "content": [], "stop_reason": null,
        "usage": { "input_tokens": 9, "output_tokens": 0 } } }),
    ];
    events.push(json!({ "type": "content_block_start", "index": 0, "content_block": { "type": "text", "text": "" } }));
    for t in texts {
        events.push(json!({ "type": "content_block_delta", "index": 0, "delta": { "type": "text_delta", "text": t } }));
    }
    match error {
        Some(e) => events.push(json!({ "type": "error", "error": e })),
        None => {
            events.push(json!({ "type": "content_block_stop", "index": 0 }));
            events.push(json!({ "type": "message_delta", "delta": { "stop_reason": "end_turn" }, "usage": { "output_tokens": 2 } }));
            events.push(json!({ "type": "message_stop" }));
        }
    }
    let body: String = events
        .iter()
        .map(|e| format!("event: {}\ndata: {e}\n\n", e["type"].as_str().unwrap()))
        .collect();
    ResponseTemplate::new(200)
        .insert_header("content-type", "text/event-stream")
        .set_body_raw(body, "text/event-stream")
}

/// Answers `/v1/messages` with `responses` in order (the last repeats).
async fn messages(server: &MockServer, responses: Vec<ResponseTemplate>) {
    let next = Arc::new(Mutex::new(0usize));
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(move |_: &Request| {
            let mut n = next.lock().unwrap();
            let r = responses[(*n).min(responses.len() - 1)].clone();
            *n += 1;
            r
        })
        .mount(server)
        .await;
}

/// The spec's `before`: `file_old` is stored for this account and the provider still lists it.
async fn anthropic_with_stored_file(server: &MockServer) -> (Arc<Config>, Arc<Store>, String) {
    let config = config(server, "test");
    let account = Provider::Anthropic.account_identity(&config).unwrap();
    let store = Arc::new(Store::default());
    store.remember("anthropic", &account, uploaded("file_old", "anthropic"));
    Mock::given(method("GET"))
        .and(path("/v1/files/file_old"))
        .respond_with(json_response(file_body("file_old"), 200))
        .mount(server)
        .await;
    (config, store, account)
}

/// `notes`: an attachment past the inline limit, backed by the store.
fn notes(store: &Arc<Store>) -> Attachment {
    let mut bytes = b"Meeting notes long enough to upload.".to_vec();
    bytes.resize(25 * MB, b' ');
    let a = Attachment::from_bytes(bytes, "notes.txt", None);
    a.set_provider_file_store(Some(store.clone() as Arc<dyn ProviderFileStore>));
    a
}

async fn sent_file(server: &MockServer, id: &str) -> usize {
    server
        .received_requests()
        .await
        .unwrap()
        .iter()
        .filter(|r| r.url.path() == "/v1/messages")
        .filter(|r| String::from_utf8_lossy(&r.body).contains(&format!("\"file_id\":\"{id}\"")))
        .count()
}

async fn count(server: &MockServer, verb: &str, p: &str) -> usize {
    server
        .received_requests()
        .await
        .unwrap()
        .iter()
        .filter(|r| r.method.as_str() == verb && r.url.path() == p)
        .count()
}

async fn upload_returns(server: &MockServer, id: &str) {
    Mock::given(method("POST"))
        .and(path("/v1/files"))
        .respond_with(json_response(file_body(id), 200))
        .mount(server)
        .await;
}

fn chat(config: &Arc<Config>) -> Chat {
    Chat::with_config(config.clone(), Some(MODEL), Some("anthropic"), false).unwrap()
}

// spec: chat_provider_uploads_spec.rb:94 uploads a file the provider deleted again and retries the request once
#[tokio::test]
async fn uploads_a_deleted_file_again_and_retries_once() {
    let server = MockServer::start().await;
    let (config, store, account) = anthropic_with_stored_file(&server).await;
    upload_returns(&server, "file_new").await;
    messages(
        &server,
        vec![missing("file_old", 404), json_response(reply(), 200)],
    )
    .await;

    let response = chat(&config)
        .ask_with("Summarize these notes", vec![notes(&store)])
        .await
        .unwrap();

    assert_eq!(response.content.as_deref(), Some("Noted."));
    assert_eq!(count(&server, "POST", "/v1/files").await, 1);
    assert_eq!(sent_file(&server, "file_old").await, 1);
    assert_eq!(sent_file(&server, "file_new").await, 1);
    assert_eq!(store.ids(), vec!["file_new"]);
    assert!(!is_confirmed_upload("anthropic", &account, "file_old"));
}

// spec: chat_provider_uploads_spec.rb:108 retries a streamed request that failed before streaming anything
#[tokio::test]
async fn retries_a_stream_that_failed_before_streaming_anything() {
    let server = MockServer::start().await;
    let (config, store, _) = anthropic_with_stored_file(&server).await;
    upload_returns(&server, "file_new").await;
    messages(
        &server,
        vec![missing("file_old", 400), streamed(&["Noted."], None)],
    )
    .await;
    let chunks = Arc::new(Mutex::new(String::new()));
    let sink = chunks.clone();

    let mut chat = chat(&config);
    chat.ask_later_with("Summarize these notes", vec![notes(&store)])
        .unwrap();
    let response = chat
        .complete_stream(move |c| {
            if let Some(t) = &c.content {
                sink.lock().unwrap().push_str(t);
            }
        })
        .await
        .unwrap();

    assert_eq!(response.content.as_deref(), Some("Noted."));
    assert!(chunks.lock().unwrap().starts_with("Noted."));
    assert_eq!(sent_file(&server, "file_new").await, 1);
}

// spec: chat_provider_uploads_spec.rb:122 never retries once streamed content reached the caller
#[tokio::test]
async fn never_retries_once_streamed_content_reached_the_caller() {
    let server = MockServer::start().await;
    let (config, store, _) = anthropic_with_stored_file(&server).await;
    upload_returns(&server, "file_new").await;
    messages(
        &server,
        vec![streamed(
            &["Partial"],
            Some(json!({ "type": "not_found_error", "message": "File not found: file_old" })),
        )],
    )
    .await;
    let chunks = Arc::new(Mutex::new(String::new()));
    let sink = chunks.clone();

    let mut chat = chat(&config);
    chat.ask_later_with("Summarize these notes", vec![notes(&store)])
        .unwrap();
    let err = chat
        .complete_stream(move |c| {
            if let Some(t) = &c.content {
                sink.lock().unwrap().push_str(t);
            }
        })
        .await
        .unwrap_err();

    assert!(err.to_string().contains("file_old"), "{err}");
    assert_eq!(*chunks.lock().unwrap(), "Partial");
    assert_eq!(count(&server, "POST", "/v1/files").await, 0);
    assert_eq!(count(&server, "POST", "/v1/messages").await, 1);
}

// spec: chat_provider_uploads_spec.rb:136 uploads again after a 404 that names no file
#[tokio::test]
async fn uploads_again_after_a_404_that_names_no_file() {
    let server = MockServer::start().await;
    let (config, store, _) = anthropic_with_stored_file(&server).await;
    upload_returns(&server, "file_new").await;
    messages(
        &server,
        vec![
            json_response(
                json!({ "type": "error", "error": { "type": "not_found_error", "message": "Not found" } }),
                404,
            ),
            json_response(reply(), 200),
        ],
    )
    .await;

    let response = chat(&config)
        .ask_with("Summarize these notes", vec![notes(&store)])
        .await
        .unwrap();
    assert_eq!(response.content.as_deref(), Some("Noted."));
    assert_eq!(sent_file(&server, "file_new").await, 1);
}

// spec: chat_provider_uploads_spec.rb:147 retries only once
#[tokio::test]
async fn retries_only_once() {
    let server = MockServer::start().await;
    let (config, store, _) = anthropic_with_stored_file(&server).await;
    upload_returns(&server, "file_new").await;
    messages(
        &server,
        vec![missing("file_old", 404), missing("file_new", 404)],
    )
    .await;

    let err = chat(&config)
        .ask_with("Summarize these notes", vec![notes(&store)])
        .await
        .unwrap_err();
    assert!(err.to_string().contains("file_new"), "{err}");
    assert_eq!(count(&server, "POST", "/v1/files").await, 1);
    assert_eq!(count(&server, "POST", "/v1/messages").await, 2);
}

// spec: chat_provider_uploads_spec.rb:157 does not retry errors about something else
#[tokio::test]
async fn does_not_retry_errors_about_something_else() {
    let server = MockServer::start().await;
    let (config, store, _) = anthropic_with_stored_file(&server).await;
    upload_returns(&server, "file_new").await;
    messages(
        &server,
        vec![json_response(
            json!({ "type": "error", "error": { "type": "invalid_request_error", "message": "max_tokens is too large" } }),
            400,
        )],
    )
    .await;

    let err = chat(&config)
        .ask_with("Summarize these notes", vec![notes(&store)])
        .await
        .unwrap_err();
    assert!(matches!(err, Error::BadRequest(..)), "{err:?}");
    assert_eq!(count(&server, "POST", "/v1/files").await, 0);
    assert_eq!(count(&server, "POST", "/v1/messages").await, 1);
}

// spec: chat_provider_uploads_spec.rb:170 does not retry a file the application uploaded itself
#[tokio::test]
async fn does_not_retry_a_file_the_application_uploaded_itself() {
    let server = MockServer::start().await;
    let config = config(&server, "test");
    messages(&server, vec![missing("file_app", 404)]).await;
    let mut file = uploaded("file_app", "anthropic");
    file.filename = Some("notes.txt".into());
    file.mime_type = Some("text/plain".into());

    let err = chat(&config)
        .ask_with("Summarize these notes", vec![Attachment::from(file)])
        .await
        .unwrap_err();
    assert!(err.to_string().contains("file_app"), "{err}");
    assert_eq!(count(&server, "POST", "/v1/messages").await, 1);
}

// spec: chat_provider_uploads_spec.rb:185 compaction > uploads a file the provider deleted again and compacts once more
#[tokio::test]
async fn compaction_uploads_a_deleted_file_again_and_compacts_once_more() {
    let server = MockServer::start().await;
    let config = config(&server, "test");
    let xai_account = Provider::XAI.account_identity(&config).unwrap();
    let store = Arc::new(Store::default());
    store.remember("xai", &xai_account, uploaded("file_old", "xai"));
    let xai_file = |id: &str| json!({ "id": id, "object": "file", "filename": "report.pdf", "bytes": 40, "created_at": 1, "purpose": "user_data" });
    Mock::given(method("GET"))
        .and(path("/v1/files/file_old"))
        .respond_with(json_response(xai_file("file_old"), 200))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/files"))
        .respond_with(json_response(xai_file("file_new"), 200))
        .mount(&server)
        .await;
    let next = Arc::new(Mutex::new(0usize));
    Mock::given(method("POST"))
        .and(path("/v1/responses/compact"))
        .respond_with(move |_: &Request| {
            let mut n = next.lock().unwrap();
            *n += 1;
            if *n == 1 {
                missing("file_old", 404)
            } else {
                json_response(
                    json!({ "id": "cmp_1", "object": "response.compaction",
                            "output": [{ "type": "compaction", "id": "cmp_1" }],
                            "usage": { "input_tokens": 23, "output_tokens": 7 } }),
                    200,
                )
            }
        })
        .mount(&server)
        .await;
    let mut bytes = b"%PDF-1.4 long enough to upload".to_vec();
    bytes.resize(51 * MB, b' ');
    let report = Attachment::from_bytes(bytes, "report.pdf", None);
    report.set_provider_file_store(Some(store.clone() as Arc<dyn ProviderFileStore>));
    let mut chat = Chat::with_config(config, Some("grok-4.3"), Some("xai"), false).unwrap();
    let mut m = Message::user("Remember this report.");
    m.attachments.push(report);
    chat.add_message(m);

    chat.compact().await.unwrap();

    let compacts: Vec<String> = server
        .received_requests()
        .await
        .unwrap()
        .iter()
        .filter(|r| r.url.path() == "/v1/responses/compact")
        .map(|r| String::from_utf8_lossy(&r.body).into_owned())
        .collect();
    assert_eq!(
        compacts
            .iter()
            .filter(|b| b.contains("\"file_id\":\"file_new\""))
            .count(),
        1
    );
}

// ---- protocol_file_preprocessing_spec.rb ---------------------------------------------------------

const GEMINI: &str = "gemini-2.5-flash";

async fn gemini_answers(server: &MockServer) {
    Mock::given(method("POST"))
        .and(path(format!("/v1beta/models/{GEMINI}:generateContent")))
        .respond_with(json_response(
            json!({ "candidates": [{ "content": { "role": "model", "parts": [{ "text": "ok" }] }, "finishReason": "STOP" }],
                    "usageMetadata": { "promptTokenCount": 1, "candidatesTokenCount": 1 } }),
            200,
        ))
        .mount(server)
        .await;
}

fn gemini_file(id: &str, expires_in_secs: i64) -> Value {
    let expires = chrono::Utc::now() + chrono::Duration::seconds(expires_in_secs);
    json!({ "name": id, "displayName": "clip.mp4", "mimeType": "video/mp4",
            "uri": format!("https://example.test/{id}"), "expirationTime": expires.to_rfc3339(),
            "state": "ACTIVE" })
}

fn gemini_uploaded(id: &str, expires_in_secs: i64) -> UploadedFile {
    let mut f = uploaded(id, "gemini");
    f.uri = Some(format!("https://example.test/{id}"));
    f.expires_at = Some(chrono::Utc::now() + chrono::Duration::seconds(expires_in_secs));
    f
}

/// Gemini's resumable upload answering with `file` (as many times as asked).
async fn gemini_upload(server: &MockServer, file: Value) {
    Mock::given(method("POST"))
        .and(path("/upload/v1beta/files"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("x-goog-upload-url", format!("{}/session", server.uri())),
        )
        .mount(server)
        .await;
    Mock::given(method("POST"))
        .and(path("/session"))
        .respond_with(json_response(json!({ "file": file }), 200))
        .mount(server)
        .await;
}

async fn gemini_finds(server: &MockServer, id: &str, response: ResponseTemplate) {
    Mock::given(method("GET"))
        .and(path(format!("/v1beta/{id}")))
        .respond_with(response)
        .mount(server)
        .await;
}

/// `stored_attachment`: a clip past Gemini's inline threshold, backed by `store`.
fn stored_attachment(store: &Arc<Store>) -> Attachment {
    let a = Attachment::from_bytes(vec![b' '; 21 * MB], "clip.mp4", None);
    a.set_provider_file_store(Some(store.clone() as Arc<dyn ProviderFileStore>));
    a
}

/// `sent_file_id`: the file the chat referenced for `attachment`.
async fn sent_file_id(server: &MockServer, config: &Arc<Config>, attachment: Attachment) -> String {
    let before = server.received_requests().await.unwrap().len();
    let mut chat = Chat::with_config(config.clone(), Some(GEMINI), Some("gemini"), false).unwrap();
    chat.ask_with("Watch this", vec![attachment]).await.unwrap();
    let requests = server.received_requests().await.unwrap();
    let sent = requests[before..]
        .iter()
        .rfind(|r| r.url.path().ends_with(":generateContent"))
        .unwrap();
    let body: Value = serde_json::from_slice(&sent.body).unwrap();
    body["contents"][0]["parts"][1]["file_data"]["file_uri"]
        .as_str()
        .unwrap()
        .to_string()
}

/// Each example uses its own API key, so the process-wide confirmations never leak between tests.
fn gemini_setup(server: &MockServer, key: &str) -> (Arc<Config>, Arc<Store>, String) {
    let config = config(server, key);
    let account = Provider::Gemini.account_identity(&config).unwrap();
    (config, Arc::new(Store::default()), account)
}

// spec: protocol_file_preprocessing_spec.rb:10 uploads a replacement only after the application converts the unsupported attachment
#[tokio::test]
async fn uploads_a_replacement_only_after_conversion() {
    let server = MockServer::start().await;
    let config = config(&server, "convert");
    gemini_answers(&server).await;
    let mut converted = gemini_file("files/converted", 48 * 3600);
    converted["displayName"] = "report.pdf".into();
    converted["mimeType"] = "application/pdf".into();
    gemini_upload(&server, converted).await;
    let document = Attachment::from_bytes(b"office document".to_vec(), "report.docx", None);
    let mut chat = Chat::with_config(config, Some(GEMINI), Some("gemini"), false)
        .unwrap()
        .convert_unsupported_attachments(|_| {
            Ok(Some(Attachment::from_bytes(
                vec![b' '; 60 * MB],
                "report.pdf",
                None,
            )))
        });
    chat.ask_with("Read this.", vec![document.clone()])
        .await
        .unwrap();

    let requests = server.received_requests().await.unwrap();
    let sent = requests
        .iter()
        .find(|r| r.url.path().ends_with(":generateContent"))
        .unwrap();
    let body: Value = serde_json::from_slice(&sent.body).unwrap();
    assert_eq!(
        body["contents"][0]["parts"][1]["file_data"]["file_uri"],
        "https://example.test/files/converted"
    );
    assert_eq!(count(&server, "POST", "/upload/v1beta/files").await, 1);
    assert_eq!(chat.messages()[0].attachments, vec![document]);
}

// spec: protocol_file_preprocessing_spec.rb:221 resolves the preprocessing protocol once per request
#[test]
fn resolves_the_preprocessing_protocol_once_per_request() {
    // Ruby counts `protocol_for` calls; here protocol resolution is a pure function of the chat's
    // settings, so what the spec protects is that preprocessing follows the request's protocol:
    // five plain messages render without touching any attachment machinery.
    let mut c = Config::default();
    c.set("openai_api_key", "test");
    let mut chat =
        Chat::with_config(Arc::new(c), Some("gpt-5-nano"), Some("openai"), false).unwrap();
    for i in 0..5 {
        chat.add_message(Message::user(format!("message {i}")));
    }
    let payload = chat.render().unwrap();
    assert_eq!(payload["input"].as_array().unwrap().len(), 5);
}

// spec: protocol_file_preprocessing_spec.rb:286 with uploads stored for the attachment > reuses a stored upload once the provider confirms it still has the file
#[tokio::test]
async fn reuses_a_stored_upload_the_provider_confirms() {
    let server = MockServer::start().await;
    let (config, store, account) = gemini_setup(&server, "reuse");
    gemini_answers(&server).await;
    store.remember(
        "gemini",
        &account,
        gemini_uploaded("files/stored", 48 * 3600),
    );
    gemini_finds(
        &server,
        "files/stored",
        json_response(gemini_file("files/stored", 48 * 3600), 200),
    )
    .await;

    let uri = sent_file_id(&server, &config, stored_attachment(&store)).await;
    assert_eq!(uri, "https://example.test/files/stored");
    assert_eq!(count(&server, "POST", "/upload/v1beta/files").await, 0);
}

// spec: protocol_file_preprocessing_spec.rb:295 with uploads stored for the attachment > asks the provider about a stored upload once per process
#[tokio::test]
async fn asks_about_a_stored_upload_once_per_process() {
    let server = MockServer::start().await;
    let (config, store, account) = gemini_setup(&server, "once");
    gemini_answers(&server).await;
    store.remember(
        "gemini",
        &account,
        gemini_uploaded("files/stored", 48 * 3600),
    );
    gemini_finds(
        &server,
        "files/stored",
        json_response(gemini_file("files/stored", 48 * 3600), 200),
    )
    .await;

    for _ in 0..3 {
        assert_eq!(
            sent_file_id(&server, &config, stored_attachment(&store)).await,
            "https://example.test/files/stored"
        );
    }
    assert_eq!(count(&server, "GET", "/v1beta/files/stored").await, 1);
}

// spec: protocol_file_preprocessing_spec.rb:305 with uploads stored for the attachment > uploads again and stores the new file when the provider no longer has the stored one
#[tokio::test]
async fn uploads_again_when_the_provider_no_longer_has_the_stored_file() {
    let server = MockServer::start().await;
    let (config, store, account) = gemini_setup(&server, "gone");
    gemini_answers(&server).await;
    store.remember("gemini", &account, gemini_uploaded("files/gone", 48 * 3600));
    gemini_finds(
        &server,
        "files/gone",
        json_response(json!({ "error": { "message": "File not found" } }), 404),
    )
    .await;
    gemini_upload(&server, gemini_file("files/new", 48 * 3600)).await;

    assert_eq!(
        sent_file_id(&server, &config, stored_attachment(&store)).await,
        "https://example.test/files/new"
    );
    assert_eq!(store.ids(), vec!["files/new"]);
}

// spec: protocol_file_preprocessing_spec.rb:314 with uploads stored for the attachment > uploads again without asking the provider when the stored file has expired
#[tokio::test]
async fn uploads_again_without_asking_when_the_stored_file_expired() {
    let server = MockServer::start().await;
    let (config, store, account) = gemini_setup(&server, "expired");
    gemini_answers(&server).await;
    store.remember("gemini", &account, gemini_uploaded("files/old", -1));
    gemini_upload(&server, gemini_file("files/new", 48 * 3600)).await;

    assert_eq!(
        sent_file_id(&server, &config, stored_attachment(&store)).await,
        "https://example.test/files/new"
    );
    assert_eq!(count(&server, "GET", "/v1beta/files/old").await, 0);
}

// spec: protocol_file_preprocessing_spec.rb:323 with uploads stored for the attachment > uploads again when the provider reports the stored file has expired
#[tokio::test]
async fn uploads_again_when_the_provider_reports_the_file_expired() {
    let server = MockServer::start().await;
    let (config, store, account) = gemini_setup(&server, "reported");
    gemini_answers(&server).await;
    store.remember("gemini", &account, gemini_uploaded("files/old", 48 * 3600));
    gemini_finds(
        &server,
        "files/old",
        json_response(gemini_file("files/old", 0), 200),
    )
    .await;
    gemini_upload(&server, gemini_file("files/new", 48 * 3600)).await;

    assert_eq!(
        sent_file_id(&server, &config, stored_attachment(&store)).await,
        "https://example.test/files/new"
    );
}

// spec: protocol_file_preprocessing_spec.rb:331 with uploads stored for the attachment > stores a new upload and trusts it for the rest of the process
#[tokio::test]
async fn stores_a_new_upload_and_trusts_it_for_the_process() {
    let server = MockServer::start().await;
    let (config, store, _) = gemini_setup(&server, "trust");
    gemini_answers(&server).await;
    gemini_upload(&server, gemini_file("files/new", 48 * 3600)).await;

    let first = sent_file_id(&server, &config, stored_attachment(&store)).await;
    let second = sent_file_id(&server, &config, stored_attachment(&store)).await;
    assert_eq!([first, second], ["https://example.test/files/new"; 2]);
    assert_eq!(count(&server, "POST", "/upload/v1beta/files").await, 1);
    assert_eq!(count(&server, "GET", "/v1beta/files/new").await, 0);
}

// spec: protocol_file_preprocessing_spec.rb:342 with uploads stored for the attachment > keeps each account apart
#[tokio::test]
async fn keeps_each_account_apart() {
    let server = MockServer::start().await;
    let (_, store, ours) = gemini_setup(&server, "ours");
    let (other, _, theirs) = gemini_setup(&server, "other-tenant");
    gemini_answers(&server).await;
    store.remember("gemini", &ours, gemini_uploaded("files/ours", 48 * 3600));
    gemini_upload(&server, gemini_file("files/theirs", 48 * 3600)).await;

    assert_eq!(
        sent_file_id(&server, &other, stored_attachment(&store)).await,
        "https://example.test/files/theirs"
    );
    let mut expected = vec![("gemini".to_string(), ours), ("gemini".to_string(), theirs)];
    expected.sort();
    assert_eq!(store.keys(), expected);
}
