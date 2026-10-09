//! Ports of the stubbed-HTTP examples in `spec/ruby_llm/protocols/files_spec.rb` (Gemini) and
//! `protocols/perplexity/files_spec.rb`. The Ruby specs stub the Faraday connection or `find`;
//! here a mock server answers, so the port's real request path runs.

use std::sync::Arc;

use rust_llm::files::{FileOptions, UploadOptions};
use rust_llm::{Config, Error, UploadedFile};
use serde_json::json;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn ruby_txt() -> String {
    format!("{}/tests/fixtures/ruby.txt", env!("CARGO_MANIFEST_DIR"))
}

/// A Gemini configuration (`gemini_api_key = 'test'`) pointed at `server`.
fn gemini(server: &MockServer) -> Arc<Config> {
    let mut config = Config::default();
    config.set("gemini_api_base", format!("{}/v1beta", server.uri()));
    config.set("gemini_api_key", "test");
    config.max_retries = 0;
    Arc::new(config)
}

fn gemini_options(config: &Arc<Config>) -> FileOptions<'static> {
    FileOptions {
        provider: Some("gemini"),
        config: Some(config.clone()),
    }
}

// spec: protocols/files_spec.rb:241 raises when Gemini does not hand back an upload URL
#[tokio::test]
async fn gemini_upload_without_an_upload_url_raises() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/upload/v1beta/files"))
        .respond_with(ResponseTemplate::new(200))
        .expect(1)
        .mount(&server)
        .await;
    let config = gemini(&server);
    let err = UploadedFile::upload(
        ruby_txt().as_str(),
        UploadOptions {
            provider: Some("gemini"),
            config: Some(config),
            ..Default::default()
        },
    )
    .await
    .unwrap_err();
    assert!(matches!(err, Error::Api(..)), "{err:?}");
    assert_eq!(err.to_string(), "gemini did not return an upload URL");
}

// spec: protocols/files_spec.rb:253 refuses to download a file with no download URI
#[tokio::test]
async fn gemini_download_without_a_download_uri_raises() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1beta/files/abc"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "name": "files/abc" })))
        .expect(1)
        .mount(&server)
        .await;
    let config = gemini(&server);
    let err = UploadedFile::download("files/abc", gemini_options(&config))
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Api(..)), "{err:?}");
    assert_eq!(err.to_string(), "gemini file has no download URI");
}

// spec: protocols/files_spec.rb:261 downloads JSON files without parsing their contents
#[tokio::test]
async fn gemini_downloads_json_files_without_parsing_them() {
    let server = MockServer::start().await;
    let download_uri = format!("{}/files-example/abc", server.uri());
    Mock::given(method("GET"))
        .and(path("/v1beta/files/abc"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({ "name": "files/abc", "downloadUri": download_uri })),
        )
        .mount(&server)
        .await;
    let body = "{\"content\":\"Hello\"}\n";
    Mock::given(method("GET"))
        .and(path("/files-example/abc"))
        .and(header("x-goog-api-key", "test"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_bytes(body.as_bytes().to_vec()),
        )
        .expect(1)
        .mount(&server)
        .await;
    let config = gemini(&server);
    let file = UploadedFile::download("files/abc", gemini_options(&config))
        .await
        .unwrap();
    assert_eq!(file.to_blob(), body.as_bytes());
}

// spec: protocols/perplexity/files_spec.rb:19 finds and downloads only the bound generated file through the provider connection
#[tokio::test]
async fn perplexity_finds_and_downloads_the_bound_generated_file() {
    let server = MockServer::start().await;
    let data =
        json!({ "id": "file_123", "filename": "numbers.csv", "bytes": 9, "created_at": 100 });
    Mock::given(method("GET"))
        .and(path("/v1/agent/resp_123/files"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "data": [data] })))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/v1/agent/resp_123/files/file_123/content"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(b"value\n91\n".to_vec()))
        .expect(1)
        .mount(&server)
        .await;
    let mut config = Config::default();
    config.set("perplexity_api_base", server.uri());
    config.set("perplexity_api_key", "test");
    config.max_retries = 0;
    let config = Arc::new(config);
    let options = || FileOptions {
        provider: Some("perplexity"),
        config: Some(config.clone()),
    };

    let file = UploadedFile::find("resp_123/files/file_123", options())
        .await
        .unwrap();
    assert_eq!(file.filename.as_deref(), Some("numbers.csv"));
    let downloaded = UploadedFile::download("resp_123/files/file_123", options())
        .await
        .unwrap();
    assert_eq!(downloaded.to_blob(), b"value\n91\n");
}
