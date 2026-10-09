//! `RubyLLM.upload` / `UploadedFile`, replayed from RubyLLM's `uploadedfile_*` cassettes.
//! Assertions follow `spec/ruby_llm/uploaded_file_spec.rb`. Multipart bodies can't be JSON-compared,
//! so each upload asserts method, path, headers, and the form parts against the recording.

mod support;

use std::sync::Arc;

use rust_llm::files::{FileOptions, UploadOptions};
use rust_llm::{Attachment, Chat, Config, Error, UploadedFile, upload};
use serde_json::json;
use support::{Cassette, Interaction, config_for};

fn pdf_path() -> String {
    format!("{}/tests/fixtures/sample.pdf", env!("CARGO_MANIFEST_DIR"))
}

fn pdf_bytes() -> Vec<u8> {
    std::fs::read(pdf_path()).expect("sample.pdf fixture")
}

async fn start(name: &str) -> Cassette {
    Cassette::start(name).await.unwrap_or_else(|| {
        panic!("missing cassette {name}; run bin/convert-cassettes 'uploadedfile_*'")
    })
}

/// One part of a multipart body: its disposition name, filename, content type, and bytes.
#[derive(Debug)]
struct Part {
    name: String,
    filename: Option<String>,
    content_type: Option<String>,
    body: Vec<u8>,
}

fn find(haystack: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    haystack
        .get(from..)?
        .windows(needle.len())
        .position(|w| w == needle)
        .map(|p| p + from)
}

/// Splits a `multipart/form-data` body on `boundary`.
fn parse_multipart(body: &[u8], boundary: &str) -> Vec<Part> {
    let delimiter = format!("--{boundary}");
    let mut parts = Vec::new();
    let mut at = find(body, delimiter.as_bytes(), 0).expect("first boundary") + delimiter.len();
    while body.get(at..at + 2) == Some(b"\r\n") {
        let head_end = find(body, b"\r\n\r\n", at).expect("part headers");
        let head = String::from_utf8_lossy(&body[at + 2..head_end]).to_string();
        let next =
            find(body, format!("\r\n{delimiter}").as_bytes(), head_end).expect("next boundary");
        let header = |key: &str| {
            head.lines().find_map(|l| {
                let (k, v) = l.split_once(':')?;
                k.trim()
                    .eq_ignore_ascii_case(key)
                    .then(|| v.trim().to_string())
            })
        };
        let disposition = header("content-disposition").unwrap_or_default();
        let attr = |key: &str| {
            let marker = format!("{key}=\"");
            let start = disposition.find(&marker)? + marker.len();
            Some(disposition[start..start + disposition[start..].find('"')?].to_string())
        };
        parts.push(Part {
            name: attr("name").unwrap_or_default(),
            filename: attr("filename"),
            content_type: header("content-type"),
            body: body[head_end + 4..next].to_vec(),
        });
        at = next + 2 + delimiter.len();
    }
    parts
}

/// The recorded Ruby body names its own boundary on the first line.
fn recorded_parts(interaction: &Interaction) -> Vec<Part> {
    let first = interaction
        .request_body
        .lines()
        .next()
        .expect("recorded multipart body");
    parse_multipart(
        interaction.request_body.as_bytes(),
        first.trim().strip_prefix("--").expect("boundary line"),
    )
}

fn sent_parts(request: &wiremock::Request) -> Vec<Part> {
    let content_type = header(request, "content-type").unwrap_or_default();
    assert!(
        content_type.starts_with("multipart/form-data"),
        "upload must be multipart, sent {content_type}"
    );
    let boundary = content_type
        .split("boundary=")
        .nth(1)
        .expect("multipart boundary")
        .to_string();
    parse_multipart(&request.body, &boundary)
}

/// The sent form matches RubyLLM's: same parts in the same order, same filename, content type,
/// and field values; the file part carries exactly sample.pdf, as long as Ruby's `Content-Length`.
/// (The cassette's copy of the PDF bytes is lossy UTF-8, so the bytes are checked against the fixture.)
fn assert_same_form(recorded: &Interaction, sent: &wiremock::Request) {
    let expected = recorded_parts(recorded);
    let actual = sent_parts(sent);
    let names = |parts: &[Part]| parts.iter().map(|p| p.name.clone()).collect::<Vec<_>>();
    assert_eq!(names(&actual), names(&expected), "multipart field names");
    for (e, a) in expected.iter().zip(&actual) {
        assert_eq!(a.filename, e.filename, "filename of {}", e.name);
        assert_eq!(a.content_type, e.content_type, "content type of {}", e.name);
        if e.name == "file" {
            assert_eq!(a.body, pdf_bytes(), "the file part is the PDF's bytes");
            let recorded_len = recorded
                .request_body
                .lines()
                .find_map(|l| l.strip_prefix("Content-Length: "))
                .expect("recorded length");
            assert_eq!(a.body.len().to_string(), recorded_len.trim());
        } else {
            assert_eq!(
                String::from_utf8_lossy(&a.body),
                String::from_utf8_lossy(&e.body),
                "value of {}",
                e.name
            );
        }
    }
}

fn header(request: &wiremock::Request, name: &str) -> Option<String> {
    request
        .headers
        .get(name)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string)
}

fn upload_options<'a>(
    provider: &'a str,
    purpose: Option<&'a str>,
    config: Arc<Config>,
) -> UploadOptions<'a> {
    UploadOptions {
        provider: Some(provider),
        purpose,
        config: Some(config),
        ..Default::default()
    }
}

/// `it "#{provider} uploads a PDF through the Files API"`: `file.id` is present and
/// `file.byte_size` equals the PDF's size.
async fn uploads_a_pdf(
    provider: &str,
    purpose: Option<&str>,
) -> (UploadedFile, Vec<wiremock::Request>, Vec<Interaction>) {
    let name = format!("uploadedfile_live_uploads_{provider}_uploads_a_pdf_through_the_files_api");
    let cassette = start(&name).await;
    let file = upload(
        pdf_path().as_str(),
        upload_options(provider, purpose, config_for(&cassette, provider)),
    )
    .await
    .expect("upload");
    assert!(!file.id.is_empty(), "file.id is present");
    assert_eq!(
        file.byte_size,
        Some(std::fs::metadata(pdf_path()).unwrap().len())
    );
    assert_eq!(file.provider, provider);
    cassette.assert_all_matched().await;
    let requests = cassette
        .server
        .received_requests()
        .await
        .unwrap_or_default();
    (file, requests, support::load(&name).unwrap())
}

#[tokio::test]
async fn anthropic_uploads_a_pdf_through_the_files_api() {
    let (file, requests, recorded) = uploads_a_pdf("anthropic", None).await;
    assert_eq!(file.id, "file_013WdkzVoWhQXMN86PMMPhjC");
    assert_eq!(file.mime_type.as_deref(), Some("application/pdf"));
    assert_eq!(file.downloadable, Some(false));
    assert_eq!(file.created_at.map(|t| t.timestamp()), Some(1791377967));
    assert_eq!(requests[0].method.as_str(), "POST");
    assert_eq!(requests[0].url.path(), "/v1/files");
    assert_eq!(
        header(&requests[0], "anthropic-beta").as_deref(),
        Some("files-api-2025-04-14")
    );
    assert_eq!(
        header(&requests[0], "x-api-key").as_deref(),
        Some("test-key")
    );
    assert_same_form(&recorded[0], &requests[0]);
}

#[tokio::test]
async fn openai_uploads_a_pdf_through_the_files_api() {
    let (file, requests, recorded) = uploads_a_pdf("openai", Some("user_data")).await;
    assert_eq!(file.id, "file-JxdyqrXagjmVcDLvELUhFU");
    assert_eq!(file.purpose.as_deref(), Some("user_data"));
    assert_eq!(file.status.as_deref(), Some("processed"));
    assert_eq!(file.expires_at, None);
    assert_eq!(requests[0].method.as_str(), "POST");
    assert_eq!(requests[0].url.path(), "/v1/files");
    assert_eq!(
        header(&requests[0], "authorization").as_deref(),
        Some("Bearer test-key")
    );
    assert_same_form(&recorded[0], &requests[0]);
}

#[tokio::test]
async fn mistral_uploads_a_pdf_through_the_files_api() {
    let (file, requests, recorded) = uploads_a_pdf("mistral", Some("ocr")).await;
    assert_eq!(file.id, "bb9e30b1-970f-4a42-b682-8c0079e7c242");
    assert_eq!(file.mime_type.as_deref(), Some("application/pdf"));
    assert_eq!(file.purpose.as_deref(), Some("ocr"));
    assert_eq!(requests[0].url.path(), "/v1/files");
    assert_same_form(&recorded[0], &requests[0]);
}

#[tokio::test]
async fn xai_uploads_a_pdf_through_the_files_api() {
    let (file, requests, recorded) = uploads_a_pdf("xai", None).await;
    assert_eq!(file.id, "file_8e8bc0c2-2e67-45b1-a96b-783f535c443c");
    assert_eq!(requests[0].url.path(), "/v1/files");
    // xAI sends neither purpose nor expires_after when they aren't given: only the file part.
    assert_same_form(&recorded[0], &requests[0]);
}

/// Gemini's resumable upload: a JSON `start` naming the file, then the raw bytes to the URL Gemini returned.
fn assert_gemini_resumable_upload(requests: &[wiremock::Request], recorded: &[Interaction]) {
    let start = &requests[0];
    assert_eq!(start.method.as_str(), "POST");
    assert_eq!(start.url.path(), "/upload/v1beta/files");
    assert_eq!(
        header(start, "x-goog-upload-protocol").as_deref(),
        Some("resumable")
    );
    assert_eq!(
        header(start, "x-goog-upload-command").as_deref(),
        Some("start")
    );
    assert_eq!(
        header(start, "x-goog-upload-header-content-length").as_deref(),
        Some("18810")
    );
    assert_eq!(
        header(start, "x-goog-upload-header-content-type").as_deref(),
        Some("application/pdf")
    );
    assert_eq!(header(start, "x-goog-api-key").as_deref(), Some("test-key"));

    let bytes = &requests[1];
    let recorded_query = recorded[1].uri.split_once('?').map(|(_, q)| q).unwrap();
    assert_eq!(bytes.url.path(), "/upload/v1beta/files");
    assert_eq!(
        bytes.url.query(),
        Some(recorded_query),
        "bytes go to the upload URL Gemini returned"
    );
    assert_eq!(
        header(bytes, "x-goog-upload-command").as_deref(),
        Some("upload, finalize")
    );
    assert_eq!(header(bytes, "x-goog-upload-offset").as_deref(), Some("0"));
    assert_eq!(bytes.body, pdf_bytes());
}

#[tokio::test]
async fn gemini_uploads_a_pdf_through_the_files_api() {
    let (file, requests, recorded) = uploads_a_pdf("gemini", None).await;
    assert_eq!(file.id, "files/aurmd9vp5ypk");
    assert_eq!(file.filename.as_deref(), Some("sample.pdf"));
    assert_eq!(file.status.as_deref(), Some("ACTIVE"));
    assert_eq!(
        file.uri.as_deref(),
        Some("https://generativelanguage.googleapis.com/v1beta/files/aurmd9vp5ypk")
    );
    assert_eq!(file.expires_at.map(|t| t.timestamp()), Some(1791550769));
    assert_gemini_resumable_upload(&requests, &recorded);
}

#[tokio::test]
async fn openai_gpt_5_nano_reuses_an_uploaded_file_in_chat() {
    let name = "uploadedfile_live_uploads_openai_gpt-5-nano_reuses_an_uploaded_file_in_chat";
    let cassette = start(name).await;
    let config = config_for(&cassette, "openai");
    let file = upload(
        pdf_path().as_str(),
        upload_options("openai", Some("user_data"), config.clone()),
    )
    .await
    .unwrap();
    let mut chat = Chat::with_config(config, Some("gpt-5-nano"), Some("openai"), false).unwrap();
    let response = chat
        .ask_with(
            "Summarize this document in one sentence.",
            vec![file.into()],
        )
        .await
        .unwrap();
    let content = response.content().to_lowercase();
    assert!(
        ["pdf", "document", "lorem", "sample"]
            .iter()
            .any(|w| content.contains(w)),
        "{content}"
    );
    // The responses request (input_file with file_id) is JSON-compared to RubyLLM's here.
    cassette.assert_all_matched().await;
    let requests = cassette.server.received_requests().await.unwrap();
    assert_same_form(&support::load(name).unwrap()[0], &requests[0]);
}

#[tokio::test]
async fn gemini_2_5_flash_reuses_an_uploaded_file_in_chat() {
    let name = "uploadedfile_live_uploads_gemini_gemini-2_5-flash_reuses_an_uploaded_file_in_chat";
    let cassette = start(name).await;
    let config = config_for(&cassette, "gemini");
    let file = upload(
        pdf_path().as_str(),
        upload_options("gemini", None, config.clone()),
    )
    .await
    .unwrap();
    let mut chat =
        Chat::with_config(config, Some("gemini-2.5-flash"), Some("gemini"), false).unwrap();
    let response = chat
        .ask_with(
            "Summarize this document in one sentence.",
            vec![file.into()],
        )
        .await
        .unwrap();
    let content = response.content().to_lowercase();
    assert!(
        ["pdf", "document", "lorem", "sample"]
            .iter()
            .any(|w| content.contains(w)),
        "{content}"
    );
    // The start body and the generateContent request (file_data.file_uri) are JSON-compared here.
    cassette.assert_all_matched().await;
    let requests = cassette.server.received_requests().await.unwrap();
    assert_gemini_resumable_upload(&requests, &support::load(name).unwrap());
}

// ---- rendering file references (no cassette records these chat requests) --------------------

fn uploaded(provider: &str, id: &str, mime: &str) -> UploadedFile {
    UploadedFile {
        id: id.into(),
        provider: provider.into(),
        filename: Some("sample.pdf".into()),
        byte_size: Some(18810),
        created_at: None,
        expires_at: None,
        status: None,
        mime_type: Some(mime.into()),
        purpose: None,
        uri: None,
        downloadable: None,
        metadata: json!({}),
    }
}

fn offline_chat(model: &str, provider: &str) -> Chat {
    let mut config = Config::default();
    config.set(format!("{provider}_api_key"), "test-key");
    Chat::with_config(Arc::new(config), Some(model), Some(provider), false).unwrap()
}

#[test]
fn anthropic_references_uploaded_documents_and_images_by_file_id() {
    let mut chat = offline_chat("claude-haiku-4-5", "anthropic");
    let files = vec![
        uploaded("anthropic", "file_pdf", "application/pdf").into(),
        uploaded("anthropic", "file_png", "image/png").into(),
    ];
    chat.ask_later_with("Compare", files).unwrap();
    let payload = chat.render().unwrap();
    assert_eq!(
        payload["messages"][0]["content"],
        json!([
            { "type": "text", "text": "Compare" },
            { "type": "document", "source": { "type": "file", "file_id": "file_pdf" } },
            { "type": "image", "source": { "type": "file", "file_id": "file_png" } }
        ])
    );
}

#[test]
fn openai_chat_completions_references_an_uploaded_file_by_id() {
    let mut chat =
        offline_chat("gpt-5-nano", "openai").with_protocol(rust_llm::ProtocolName::ChatCompletions);
    chat.ask_later_with(
        "Summarize",
        vec![uploaded("openai", "file-1", "application/pdf").into()],
    )
    .unwrap();
    let payload = chat.render().unwrap();
    assert_eq!(
        payload["messages"][0]["content"][1],
        json!({ "type": "file", "file": { "file_id": "file-1" } })
    );
}

#[test]
fn gemini_falls_back_to_the_file_id_without_a_uri() {
    let mut chat = offline_chat("gemini-2.5-flash", "gemini");
    chat.ask_later_with(
        "Summarize",
        vec![uploaded("gemini", "files/abc", "application/pdf").into()],
    )
    .unwrap();
    let payload = chat.render().unwrap();
    assert_eq!(
        payload["contents"][0]["parts"][1],
        json!({ "file_data": { "mime_type": "application/pdf", "file_uri": "files/abc" } })
    );
}

#[test]
fn xai_chat_completions_refuses_a_provider_file() {
    let mut chat = offline_chat("grok-4-1-fast-non-reasoning", "xai")
        .with_protocol(rust_llm::ProtocolName::ChatCompletions);
    chat.ask_later_with(
        "Summarize",
        vec![uploaded("xai", "file_1", "application/pdf").into()],
    )
    .unwrap();
    assert!(matches!(
        chat.render(),
        Err(Error::UnsupportedAttachment(_))
    ));
}

#[test]
fn a_provider_file_has_no_inline_content() {
    let a: Attachment = uploaded("openai", "file-1", "application/pdf").into();
    assert!(a.is_provider_file());
    assert_eq!(a.filename.as_deref(), Some("sample.pdf"));
    assert_eq!(a.byte_size(), Some(18810));
    let err = a.encoded().unwrap_err();
    assert_eq!(
        err.to_string(),
        "Provider-managed file file-1 cannot be read as inline attachment content"
    );
}

// ---- auto-upload of large attachments (Protocol#preprocess_message) -------------------------

/// A PDF one byte over Anthropic's 24 MB inline limit.
fn large_pdf() -> Attachment {
    let mut bytes = pdf_bytes();
    bytes.resize(24 * 1024 * 1024 + 1, b' ');
    Attachment::from_bytes(bytes, "large.pdf", None)
}

async fn mock_anthropic() -> (wiremock::MockServer, Config) {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, ResponseTemplate};
    let server = wiremock::MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/files"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "type": "file", "id": "file_large", "size_bytes": 25165825, "filename": "large.pdf",
            "mime_type": "application/pdf", "downloadable": false, "created_at": "2026-09-17T22:14:58Z"
        })))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-haiku-4-5",
            "content": [{ "type": "text", "text": "A sample PDF." }], "stop_reason": "end_turn",
            "usage": { "input_tokens": 10, "output_tokens": 5 }
        })))
        .mount(&server)
        .await;
    let mut config = Config::default();
    config.set("anthropic_api_base", server.uri());
    config.set("anthropic_api_key", "test-key");
    config.max_retries = 0;
    (server, config)
}

#[tokio::test]
async fn attachments_over_the_inline_limit_upload_once_and_go_as_file_references() {
    let (server, config) = mock_anthropic().await;
    let mut chat = Chat::with_config(
        Arc::new(config),
        Some("claude-haiku-4-5"),
        Some("anthropic"),
        false,
    )
    .unwrap();
    chat.ask_with("Summarize", vec![large_pdf()]).await.unwrap();
    chat.ask("And again").await.unwrap();

    let requests = server.received_requests().await.unwrap();
    let paths: Vec<&str> = requests.iter().map(|r| r.url.path()).collect();
    assert_eq!(
        paths,
        ["/v1/files", "/v1/messages", "/v1/messages"],
        "the second turn reuses the upload"
    );
    for chat_request in &requests[1..] {
        let body: serde_json::Value = serde_json::from_slice(&chat_request.body).unwrap();
        assert_eq!(
            body["messages"][0]["content"][1],
            json!({ "type": "document", "source": { "type": "file", "file_id": "file_large" } })
        );
        assert_eq!(
            header(chat_request, "anthropic-beta").as_deref(),
            Some("files-api-2025-04-14")
        );
    }
    // History keeps the original attachment; only the request carries the reference.
    assert!(!chat.messages()[0].attachments[0].is_provider_file());
}

#[tokio::test]
async fn auto_upload_can_be_turned_off() {
    let (server, mut config) = mock_anthropic().await;
    config.auto_upload_large_files = false;
    let mut chat = Chat::with_config(
        Arc::new(config),
        Some("claude-haiku-4-5"),
        Some("anthropic"),
        false,
    )
    .unwrap();
    chat.ask_with("Summarize", vec![large_pdf()]).await.unwrap();
    let requests = server.received_requests().await.unwrap();
    assert_eq!(
        requests.iter().map(|r| r.url.path()).collect::<Vec<_>>(),
        ["/v1/messages"]
    );
    let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(
        body["messages"][0]["content"][1]["source"]["type"],
        "base64"
    );
    assert_eq!(header(&requests[0], "anthropic-beta"), None);
}

#[tokio::test]
async fn small_attachments_stay_inline() {
    let (server, config) = mock_anthropic().await;
    let mut chat = Chat::with_config(
        Arc::new(config),
        Some("claude-haiku-4-5"),
        Some("anthropic"),
        false,
    )
    .unwrap();
    chat.ask_with("Summarize", vec![Attachment::new(pdf_path())])
        .await
        .unwrap();
    let requests = server.received_requests().await.unwrap();
    assert_eq!(
        requests.iter().map(|r| r.url.path()).collect::<Vec<_>>(),
        ["/v1/messages"]
    );
}

// ---- argument errors, find, and download --------------------------------------------------

/// 2.1 (`Let OpenAI and DeepSeek validate file uploads`): an upload without a purpose goes out as
/// given, and OpenAI's error states what it needs.
#[tokio::test]
async fn openai_uploads_leave_the_purpose_requirement_to_openai() {
    let server = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::path("/v1/files"))
        .respond_with(
            wiremock::ResponseTemplate::new(400).set_body_json(serde_json::json!({
                "error": { "message": "Missing required parameter: 'purpose'." }
            })),
        )
        .expect(1)
        .mount(&server)
        .await;
    let mut config = Config::default();
    config.set("openai_api_key", "test-key");
    config.set("openai_api_base", format!("{}/v1", server.uri()));
    config.max_retries = 0;
    let options = UploadOptions {
        provider: Some("openai"),
        config: Some(Arc::new(config)),
        ..Default::default()
    };
    let err = upload(pdf_path().as_str(), options).await.unwrap_err();
    assert_eq!(err.to_string(), "Missing required parameter: 'purpose'.");
    let requests = server.received_requests().await.unwrap();
    assert!(!String::from_utf8_lossy(&requests[0].body).contains("name=\"purpose\""));
}

#[tokio::test]
async fn unknown_upload_options_are_rejected() {
    let mut config = Config::default();
    config.set("openai_api_key", "test-key");
    let mut options = UploadOptions {
        provider: Some("openai"),
        purpose: Some("batch"),
        config: Some(Arc::new(config)),
        ..Default::default()
    };
    options
        .provider_options
        .insert("unsupported".into(), json!(true));
    let err = upload(pdf_path().as_str(), options).await.unwrap_err();
    assert!(err.to_string().contains("unknown keyword"), "{err}");
}

#[tokio::test]
async fn providers_without_a_files_api_say_so() {
    let mut config = Config::default();
    config.set("ollama_api_base", "http://localhost:11434/v1");
    let err = UploadedFile::find(
        "file_1",
        FileOptions {
            provider: Some("ollama"),
            config: Some(Arc::new(config)),
        },
    )
    .await
    .unwrap_err();
    assert_eq!(err.to_string(), "ollama doesn't support file uploads");
}

#[tokio::test]
async fn find_and_download_use_the_files_endpoints() {
    use wiremock::matchers::{header as has_header, method, path};
    use wiremock::{Mock, ResponseTemplate};
    let server = wiremock::MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/files/file_1"))
        .and(has_header("anthropic-beta", "files-api-2025-04-14"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "file_1", "size_bytes": 5, "filename": "a.txt", "mime_type": "text/plain", "downloadable": true
        })))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/v1/files/file_1/content"))
        .and(has_header("accept", "application/octet-stream"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(b"hello".to_vec()))
        .mount(&server)
        .await;
    let mut config = Config::default();
    config.set("anthropic_api_base", server.uri());
    config.set("anthropic_api_key", "test-key");
    let config = Arc::new(config);
    let options = || FileOptions {
        provider: Some("anthropic"),
        config: Some(config.clone()),
    };

    let file = UploadedFile::find("file_1", options()).await.unwrap();
    assert_eq!(
        (file.id.as_str(), file.byte_size, file.downloadable),
        ("file_1", Some(5), Some(true))
    );
    let downloaded = rust_llm::download("file_1", options()).await.unwrap();
    assert_eq!(downloaded.to_blob(), b"hello");
    let path = std::env::temp_dir().join(format!("rust_llm_download_{}.txt", std::process::id()));
    assert_eq!(downloaded.save(&path).unwrap(), path);
    assert_eq!(std::fs::read(&path).unwrap(), b"hello");
    std::fs::remove_file(&path).ok();
}
