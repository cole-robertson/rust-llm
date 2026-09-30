//! Ports of `spec/ruby_llm/uploaded_file_spec.rb` (provider inferred from the default model, the
//! `Context` shortcuts) and `downloaded_file_spec.rb` (download through a context's default
//! provider). The Ruby specs stub `Models.resolve` to hand back a provider double; here the default
//! model resolves through the real registry and a mock server stands in for that provider, so the
//! request proves which provider was chosen.

use std::sync::Arc;

use rust_llm::files::{FileOptions, UploadOptions};
use rust_llm::{Config, Context, UploadedFile};
use serde_json::json;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn pdf_path() -> String {
    format!("{}/tests/fixtures/sample.pdf", env!("CARGO_MANIFEST_DIR"))
}

/// A configuration whose default model is `model` and whose `provider` points at `server`.
fn config_for(server: &MockServer, provider: &str, base_path: &str, model: &str) -> Config {
    let mut config = Config::default();
    config.default_model = model.into();
    config.set(
        format!("{provider}_api_base"),
        format!("{}{base_path}", server.uri()),
    );
    config.set(format!("{provider}_api_key"), "test-key");
    config.max_retries = 0;
    config
}

/// `model_for(:openai)` against a mock OpenAI.
async fn openai() -> (MockServer, Arc<Config>) {
    let server = MockServer::start().await;
    let config = config_for(&server, "openai", "/v1", "gpt-5-nano");
    (server, Arc::new(config))
}

fn openai_file() -> serde_json::Value {
    json!({ "id": "file_123", "object": "file", "filename": "sample.pdf", "bytes": 18810,
            "created_at": 1_700_000_000, "purpose": "batch", "status": "processed" })
}

// spec: uploaded_file_spec.rb:75 uses the provider of the default model when provider is omitted
#[tokio::test]
async fn upload_uses_the_provider_of_the_default_model() {
    let (server, config) = openai().await;
    Mock::given(method("POST"))
        .and(path("/v1/files"))
        .respond_with(ResponseTemplate::new(200).set_body_json(openai_file()))
        .expect(1)
        .mount(&server)
        .await;
    let file = UploadedFile::upload(
        pdf_path().as_str(),
        UploadOptions {
            purpose: Some("batch"),
            config: Some(config),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(
        (file.id.as_str(), file.provider.as_str()),
        ("file_123", "openai")
    );
}

// spec: uploaded_file_spec.rb:93 uses the provider of the default model when provider is omitted
#[tokio::test]
async fn find_uses_the_provider_of_the_default_model() {
    let (server, config) = openai().await;
    Mock::given(method("GET"))
        .and(path("/v1/files/file_123"))
        .respond_with(ResponseTemplate::new(200).set_body_json(openai_file()))
        .expect(1)
        .mount(&server)
        .await;
    let file = UploadedFile::find(
        "file_123",
        FileOptions {
            config: Some(config),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(
        (file.id.as_str(), file.provider.as_str()),
        ("file_123", "openai")
    );
}

// spec: uploaded_file_spec.rb:111 uses the provider of the default model when provider is omitted
#[tokio::test]
async fn download_uses_the_provider_of_the_default_model() {
    let (server, config) = openai().await;
    Mock::given(method("GET"))
        .and(path("/v1/files/file_123/content"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(b"jsonl\n".to_vec()))
        .expect(1)
        .mount(&server)
        .await;
    let file = UploadedFile::download(
        "file_123",
        FileOptions {
            config: Some(config),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(file.to_blob(), b"jsonl\n");
}

/// `RubyLLM::Context.new(config)` with `default_model = model_for(:anthropic)`.
async fn anthropic_context() -> (MockServer, Context) {
    let server = MockServer::start().await;
    let config = config_for(&server, "anthropic", "", "claude-haiku-4-5");
    (server, Context::new(config))
}

// spec: uploaded_file_spec.rb:121 uploads through the provider of the context default model
#[tokio::test]
async fn context_uploads_through_the_provider_of_its_default_model() {
    let (server, context) = anthropic_context().await;
    Mock::given(method("POST"))
        .and(path("/v1/files"))
        .and(header("anthropic-beta", "files-api-2025-04-14"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "type": "file", "id": "file_doc", "size_bytes": 18810, "filename": "sample.pdf",
            "mime_type": "application/pdf", "downloadable": false, "created_at": "2026-09-17T22:14:58Z"
        })))
        .expect(1)
        .mount(&server)
        .await;
    let file = context
        .upload(pdf_path().as_str(), UploadOptions::default())
        .await
        .unwrap();
    assert_eq!(
        (file.id.as_str(), file.provider.as_str()),
        ("file_doc", "anthropic")
    );
}

// spec: uploaded_file_spec.rb:133 downloads through the provider of the context default model
#[tokio::test]
async fn context_downloads_through_the_provider_of_its_default_model() {
    let (server, context) = anthropic_context().await;
    Mock::given(method("GET"))
        .and(path("/v1/files/file_123/content"))
        .and(header("anthropic-beta", "files-api-2025-04-14"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(b"content\n".to_vec()))
        .expect(1)
        .mount(&server)
        .await;
    let file = context
        .download("file_123", FileOptions::default())
        .await
        .unwrap();
    assert_eq!(file.to_blob(), b"content\n");
}

// spec: downloaded_file_spec.rb:34 uses the context default provider and keeps existing string operations
#[tokio::test]
async fn context_download_uses_its_default_provider_and_keeps_the_bytes() {
    let server = MockServer::start().await;
    let body = b"\x00\xFF\r\n".to_vec();
    Mock::given(method("GET"))
        .and(path("/v1/files/file_123/content"))
        .and(header("authorization", "Bearer download-test-key"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/octet-stream")
                .set_body_bytes(body.clone()),
        )
        .expect(1)
        .mount(&server)
        .await;
    let uri = server.uri();
    // `RubyLLM.context { |config| ... }`
    let context = rust_llm::context(|config| {
        config.set("openai_api_key", "download-test-key");
        config.set("openai_api_base", format!("{uri}/v1"));
        config.default_model = "gpt-5-nano".into();
    });
    let file = context
        .download("file_123", FileOptions::default())
        .await
        .unwrap();
    // `DownloadedFile < String`: here it derefs to the raw bytes, byte for byte.
    assert_eq!(file.to_blob(), body.as_slice());
    assert_eq!(file.to_vec(), body);
    assert_eq!(
        file.split_inclusive(|b| *b == b'\n').collect::<Vec<_>>(),
        body.split_inclusive(|b| *b == b'\n').collect::<Vec<_>>()
    );
}
