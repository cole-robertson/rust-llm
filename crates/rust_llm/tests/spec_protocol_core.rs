//! Ports of RubyLLM 2.0's `protocol_spec.rb` (the base `Protocol` class). The spec calls private
//! methods on an anonymous `Protocol` subclass; the port has no protocol subclasses, so these run
//! each behavior through the public path that reaches it: `protocols::parse_completion`, a chat
//! auto-upload, and `paint`.

use std::sync::Arc;

use rust_llm::message::RawResponse;
use rust_llm::protocols::parse_completion;
use rust_llm::{
    Attachment, Chat, Config, Error, ErrorKind, Model, PaintOptions, ProtocolName, Provider,
    Resolution, paint,
};
use serde_json::{Value, json};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn raw(body: Value) -> RawResponse {
    RawResponse {
        status: 200,
        body,
        ..Default::default()
    }
}

// spec: protocol_spec.rb:17 #parse_completion_response > raises RubyLLM::Error for empty completion bodies
#[test]
fn raises_for_empty_completion_bodies() {
    let model = Model::default_for("gpt-4.1-nano", "openai");
    for body in [Value::Null, json!({}), json!([]), json!("")] {
        let err = parse_completion(
            ProtocolName::ChatCompletions,
            Provider::OpenAI,
            &model,
            raw(body.clone()),
        )
        .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Api, "{body}: {err:?}");
        assert_eq!(
            err.to_string(),
            "Provider returned an empty response body",
            "{body}"
        );
    }
}

// spec: protocol_spec.rb:37 #parse_completion_response > raises RubyLLM::Error when the protocol finds no completion message in the body
#[test]
fn raises_when_the_protocol_finds_no_completion_message_in_the_body() {
    let model = Model::default_for("gpt-4.1-nano", "openai");
    let err = parse_completion(
        ProtocolName::ChatCompletions,
        Provider::OpenAI,
        &model,
        raw(json!({ "choices": [] })),
    )
    .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Api);
    assert_eq!(err.to_string(), "Provider returned no completion message");
    // `Error.new(..., response:)`.
    assert_eq!(err.response().unwrap().body, r#"{"choices":[]}"#);
}

// spec: protocol_spec.rb:111 provider file defaults > keeps the resolution when it uploads a large attachment
#[tokio::test]
async fn keeps_the_resolution_when_it_uploads_a_large_attachment() {
    let server = MockServer::start().await;
    // Gemini's resumable upload, then the chat.
    Mock::given(method("POST"))
        .and(path("/upload/v1beta/files"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("x-goog-upload-url", format!("{}/session", server.uri())),
        )
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/session"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "file": {
            "name": "files/file_123", "displayName": "a.pdf", "mimeType": "application/pdf",
            "uri": "https://example.test/files/file_123", "state": "ACTIVE"
        }})))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-2.5-flash:generateContent"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates": [{ "content": { "role": "model", "parts": [{ "text": "ok" }] }, "finishReason": "STOP" }],
            "usageMetadata": { "promptTokenCount": 1, "candidatesTokenCount": 1 }
        })))
        .mount(&server)
        .await;
    let mut config = Config::default();
    config.set("gemini_api_base", format!("{}/v1beta", server.uri()));
    config.set("gemini_api_key", "test");
    config.max_retries = 0;
    // Over GEMINI_INLINE_FILE_THRESHOLD (20 MB), so it is uploaded.
    let attachment = Attachment::from_bytes(vec![b' '; 20 * 1024 * 1024 + 1], "a.pdf", None)
        .with_resolution(Resolution::High);
    let mut chat = Chat::with_config(
        Arc::new(config),
        Some("gemini-2.5-flash"),
        Some("gemini"),
        false,
    )
    .unwrap();
    chat.ask_with("Summarize", vec![attachment]).await.unwrap();

    let requests = server.received_requests().await.unwrap();
    let chat_request = requests
        .iter()
        .find(|r| r.url.path().ends_with(":generateContent"))
        .unwrap();
    let body: Value = serde_json::from_slice(&chat_request.body).unwrap();
    let part = &body["contents"][0]["parts"][1];
    // The uploaded file reference still carries `resolution: :high`.
    assert_eq!(
        part["file_data"]["file_uri"],
        json!("https://example.test/files/file_123")
    );
    assert_eq!(
        part["media_resolution"],
        json!({ "level": "MEDIA_RESOLUTION_HIGH" })
    );
}

// spec: protocol_spec.rb:130 #validate_paint_inputs! > refuses image references the protocol cannot send
#[tokio::test]
async fn refuses_image_references_the_protocol_cannot_send() {
    // Gemini's Imagen models keep the base `Protocol#validate_paint_inputs!`.
    let mut config = Config::default();
    config.set("gemini_api_key", "test");
    config.set("gemini_api_base", "http://127.0.0.1:9");
    let err = paint(
        "a ruby",
        PaintOptions {
            model: Some("imagen-4.0-generate-001"),
            provider: Some("gemini"),
            assume_model_exists: true,
            with: vec![Attachment::new("ref.png")],
            config: Some(Arc::new(config)),
            ..Default::default()
        },
    )
    .await
    .unwrap_err();
    assert!(matches!(err, Error::UnsupportedAttachment(_)), "{err:?}");
    assert!(err.to_string().contains("image reference"), "{err}");
}
