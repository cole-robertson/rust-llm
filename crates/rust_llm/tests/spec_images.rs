//! `spec/ruby_llm/protocols/chat_completions/images_spec.rb` and
//! `spec/ruby_llm/protocols/gemini/images_spec.rb`. RubyLLM calls `render_image_payload` and
//! `parse_image_responses` directly; here `paint` runs against a wiremock server, so the request
//! path and body are what the provider would see and the response goes through the real parser.

use std::sync::Arc;

use rust_llm::{Attachment, Config, Error, Image, PaintOptions, paint};
use serde_json::{Value, json};
use wiremock::matchers::any;
use wiremock::{Mock, MockServer, ResponseTemplate};

fn fixture(name: &str) -> String {
    format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
}

async fn server(body: Value) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(any())
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&server)
        .await;
    server
}

fn config(server: &MockServer) -> Arc<Config> {
    let mut c = Config::default();
    c.set("openai_api_base", format!("{}/v1", server.uri()));
    c.set("openai_api_key", "test");
    c.set("gemini_api_base", format!("{}/v1beta", server.uri()));
    c.set("gemini_api_key", "test");
    c.max_retries = 0;
    Arc::new(c)
}

async fn requests(server: &MockServer) -> Vec<wiremock::Request> {
    server.received_requests().await.unwrap_or_default()
}

/// The single request's path and JSON body.
async fn only_request(server: &MockServer) -> (String, Value) {
    let requests = requests(server).await;
    assert_eq!(requests.len(), 1, "expected exactly one request");
    let body = serde_json::from_slice(&requests[0].body).expect("json body");
    (requests[0].url.path().to_string(), body)
}

fn options<'a>(server: &MockServer, provider: &'a str, model: &'a str) -> PaintOptions<'a> {
    PaintOptions {
        model: Some(model),
        provider: Some(provider),
        assume_model_exists: true,
        config: Some(config(server)),
        ..Default::default()
    }
}

async fn paint_all(prompt: &str, options: PaintOptions<'_>) -> rust_llm::Result<Vec<Image>> {
    paint(prompt, options).await.map(rust_llm::Images::into_vec)
}

fn unsupported_mentions<T: std::fmt::Debug>(result: rust_llm::Result<T>, needle: &str) {
    match result {
        Err(Error::UnsupportedAttachment(message)) => {
            assert!(message.contains(needle), "{message}")
        }
        other => panic!("expected UnsupportedAttachmentError mentioning {needle}, got {other:?}"),
    }
}

// ---- chat_completions/images_spec.rb ------------------------------------------------------------

fn openai_images() -> Value {
    json!({ "data": [{ "b64_json": "first-image" }] })
}

// spec: protocols/chat_completions/images_spec.rb:19 #render_image_payload > carries the count into edit requests
#[tokio::test]
async fn openai_carries_the_count_into_edit_requests() {
    let server = server(openai_images()).await;
    paint(
        "make it green",
        PaintOptions {
            size: Some("1024x1024"),
            count: Some(2),
            with: vec![Attachment::new("https://example.com/logo.png")],
            ..options(&server, "openai", "gpt-image-1")
        },
    )
    .await
    .unwrap();
    let (path, body) = only_request(&server).await;
    assert_eq!(path, "/v1/images/edits");
    assert_eq!(body["n"], 2);
}

// spec: protocols/chat_completions/images_spec.rb:74 #parse_image_responses > raises when the response carries no image
#[tokio::test]
async fn openai_raises_when_the_response_carries_no_image() {
    let server = server(json!({ "data": [] })).await;
    let err = paint("a cat", options(&server, "openai", "gpt-image-1"))
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("Unexpected response format"),
        "{err}"
    );
}

// ---- gemini/images_spec.rb ----------------------------------------------------------------------

const IMAGEN: &str = "imagen-4.0-generate-001";
const GEMINI_IMAGE: &str = "gemini-2.5-flash-image";

fn imagen_predictions() -> Value {
    json!({ "predictions": [{ "bytesBase64Encoded": "base64-image", "mimeType": "image/jpeg" }] })
}

fn gemini_image_candidates() -> Value {
    json!({ "candidates": [{ "content": { "parts": [{ "inlineData": { "mimeType": "image/png", "data": "base64-image" } }] } }] })
}

// spec: protocols/gemini/images_spec.rb:17 #render_image_payload > keeps Imagen models on the image API
#[tokio::test]
async fn keeps_imagen_models_on_the_image_api() {
    let server = server(imagen_predictions()).await;
    paint(
        "a cat",
        PaintOptions {
            size: Some("1024x1024"),
            ..options(&server, "gemini", IMAGEN)
        },
    )
    .await
    .unwrap();
    let (path, body) = only_request(&server).await;
    assert_eq!(path, format!("/v1beta/models/{IMAGEN}:predict"));
    assert_eq!(
        body,
        json!({ "instances": [{ "prompt": "a cat" }], "parameters": { "sampleCount": 1 } })
    );
}

// spec: protocols/gemini/images_spec.rb:27 #render_image_payload > asks Imagen for several samples in one request
#[tokio::test]
async fn asks_imagen_for_several_samples_in_one_request() {
    let server = server(imagen_predictions()).await;
    paint(
        "a cat",
        PaintOptions {
            size: Some("1024x1024"),
            count: Some(4),
            ..options(&server, "gemini", IMAGEN)
        },
    )
    .await
    .unwrap();
    let (_, body) = only_request(&server).await;
    assert_eq!(body["parameters"]["sampleCount"], 4);
}

// spec: protocols/gemini/images_spec.rb:33 #render_image_payload > asks Gemini image models for several candidates in one request
#[tokio::test]
async fn asks_gemini_image_models_for_several_candidates() {
    let server = server(gemini_image_candidates()).await;
    paint(
        "a cat",
        PaintOptions {
            size: Some("1024x1024"),
            count: Some(3),
            ..options(&server, "gemini", GEMINI_IMAGE)
        },
    )
    .await
    .unwrap();
    let (_, body) = only_request(&server).await;
    assert_eq!(body["generationConfig"]["candidateCount"], 3);
}

// spec: protocols/gemini/images_spec.rb:95 #render_image_payload > lets provider_options express Gemini-specific output options
#[tokio::test]
async fn provider_options_express_gemini_output_options() {
    let server = server(gemini_image_candidates()).await;
    paint(
        "a cat",
        PaintOptions {
            size: Some("1024x1024"),
            provider_options: json!({
                "generationConfig": {
                    "responseModalities": ["IMAGE"],
                    "candidateCount": 1,
                    "imageConfig": { "aspectRatio": "16:9", "imageSize": "4K" }
                }
            }),
            ..options(&server, "gemini", GEMINI_IMAGE)
        },
    )
    .await
    .unwrap();
    let (_, body) = only_request(&server).await;
    assert_eq!(
        body["generationConfig"],
        json!({
            "responseModalities": ["IMAGE"],
            "candidateCount": 1,
            "imageConfig": { "aspectRatio": "16:9", "imageSize": "4K" }
        })
    );
}

// spec: protocols/gemini/images_spec.rb:116 #render_image_payload > formats image references for Gemini image models
#[tokio::test]
async fn formats_image_references_for_gemini_image_models() {
    let server = server(gemini_image_candidates()).await;
    paint(
        "edit this",
        PaintOptions {
            size: Some("1024x1024"),
            with: vec![Attachment::new(fixture("ruby.png"))],
            ..options(&server, "gemini", GEMINI_IMAGE)
        },
    )
    .await
    .unwrap();
    let (_, body) = only_request(&server).await;
    let image_part = &body["contents"][0]["parts"][1];
    assert_eq!(image_part["inline_data"]["mime_type"], "image/png");
    assert!(
        image_part["inline_data"]["data"]
            .as_str()
            .is_some_and(|d| !d.is_empty())
    );
}

// spec: protocols/gemini/images_spec.rb:125 #render_image_payload > rejects non-image references for Gemini image models
#[tokio::test]
async fn rejects_non_image_references_for_gemini_image_models() {
    let server = server(gemini_image_candidates()).await;
    let result = paint(
        "edit this",
        PaintOptions {
            size: Some("1024x1024"),
            with: vec![Attachment::new(fixture("ruby.wav"))],
            ..options(&server, "gemini", GEMINI_IMAGE)
        },
    )
    .await;
    unsupported_mentions(result, "Unsupported attachment type: audio/wav");
    assert!(requests(&server).await.is_empty());
}

// spec: protocols/gemini/images_spec.rb:134 #parse_image_response > parses Imagen image API responses
#[tokio::test]
async fn parses_imagen_image_api_responses() {
    let server = server(imagen_predictions()).await;
    let image = paint("a cat", options(&server, "gemini", IMAGEN))
        .await
        .unwrap()
        .into_image();
    assert_eq!(image.data.as_deref(), Some("base64-image"));
    assert_eq!(image.mime_type.as_deref(), Some("image/jpeg"));
    assert_eq!(image.model, IMAGEN);
}

// spec: protocols/gemini/images_spec.rb:190 #parse_image_response > returns every Imagen sample the request generated
#[tokio::test]
async fn returns_every_imagen_sample() {
    let server = server(json!({ "predictions": [
        { "bytesBase64Encoded": "first-image" },
        { "bytesBase64Encoded": "second-image" }
    ] }))
    .await;
    let images = paint_all("a cat", options(&server, "gemini", IMAGEN))
        .await
        .unwrap();
    let data: Vec<_> = images.iter().map(|i| i.data.as_deref()).collect();
    assert_eq!(data, vec![Some("first-image"), Some("second-image")]);
}

// spec: protocols/gemini/images_spec.rb:205 #parse_image_response > raises when Gemini generateContent returns no image
#[tokio::test]
async fn raises_when_generate_content_returns_no_image() {
    let server = server(
        json!({ "candidates": [{ "content": { "parts": [{ "text": "No image here." }] } }] }),
    )
    .await;
    let err = paint("a cat", options(&server, "gemini", GEMINI_IMAGE))
        .await
        .unwrap_err();
    assert!(
        matches!(&err, Error::Api(m, _) if m == "Unexpected response format from Gemini image generation API"),
        "{err:?}"
    );
}

// spec: protocols/gemini/images_spec.rb:225 #validate_paint_inputs! > rejects masks for Gemini image models
#[tokio::test]
async fn rejects_masks_for_gemini_image_models() {
    let server = server(gemini_image_candidates()).await;
    let result = paint(
        "edit this",
        PaintOptions {
            with: vec![Attachment::new(fixture("ruby.png"))],
            mask: Some(Attachment::new(fixture("ruby.png"))),
            ..options(&server, "gemini", GEMINI_IMAGE)
        },
    )
    .await;
    unsupported_mentions(result, "Unsupported attachment type: image mask");
    assert!(requests(&server).await.is_empty());
}
