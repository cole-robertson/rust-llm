//! `RubyLLM.paint`, replayed from RubyLLM's `image_*` cassettes. Assertions follow
//! `spec/ruby_llm/image_spec.rb`; `save_and_verify_image` becomes `saves_a_real_image`.

mod support;

use std::sync::Arc;

use rust_llm::message::Operation;
use rust_llm::model::{Pricing, PricingCategory, PricingTier};
use rust_llm::{
    Attachment, Config, Cost, Error, Image, Images, Model, PaintOptions, Tokens, UsageStatus, paint,
};
use serde_json::json;
use support::{Cassette, config_for};

const PROMPT: &str = "turn the logo to green";

fn image_path() -> String {
    format!("{}/tests/fixtures/ruby.png", env!("CARGO_MANIFEST_DIR"))
}

async fn start(name: &str) -> Cassette {
    Cassette::start(name)
        .await
        .unwrap_or_else(|| panic!("missing cassette {name}; run bin/convert-cassettes 'image_*'"))
}

/// `save_and_verify_image`: `save` returns the path it was given and writes more than 1KB.
async fn saves_a_real_image(image: &Image) {
    let path = std::env::temp_dir().join(format!("rust_llm_image_{}.img", uuid::Uuid::new_v4()));
    let saved = image.save(path.clone()).await.expect("save");
    assert_eq!(saved, path);
    let size = std::fs::metadata(&path).expect("saved file").len();
    std::fs::remove_file(&path).ok();
    assert!(
        size > 1000,
        "a real image is larger than 1KB, saved {size} bytes"
    );
}

/// Hosted images are downloaded from the provider's CDN; send that request to the replay server.
fn hosted_on(image: &mut Image, cassette: &Cassette) {
    let url = image.url.clone().expect("hosted image url");
    let path = url
        .split_once("://")
        .and_then(|(_, rest)| rest.split_once('/'))
        .map(|(_, p)| p)
        .unwrap_or("");
    image.url = Some(format!("{}/{path}", cassette.server.uri()));
}

fn options<'a>(model: &'a str, provider: Option<&'a str>, config: Arc<Config>) -> PaintOptions<'a> {
    PaintOptions {
        model: Some(model),
        provider,
        config: Some(config),
        ..Default::default()
    }
}

/// The ledger records one `image` operation for the call, carrying the image's usage.
fn billed_once_as_image(image: &Image, provider: &str, model: &str) {
    assert_eq!(image.usage_entries.len(), 1);
    let entry = &image.usage_entries[0];
    assert_eq!(entry.operation, Operation::Image);
    assert_eq!(entry.operation.as_str(), "image");
    assert_eq!(entry.status, UsageStatus::Succeeded);
    assert_eq!(entry.provider, provider);
    assert_eq!(entry.model, model);
}

// ---- basic functionality ------------------------------------------------------------------

#[tokio::test]
async fn openai_gpt_image_1_can_paint_images() {
    let cassette = start("image_basic_functionality_openai_gpt-image-1_can_paint_images").await;
    let config = config_for(&cassette, "openai");
    let image = paint(
        "a siamese cat",
        options("gpt-image-1", Some("openai"), config),
    )
    .await
    .unwrap()
    .into_image();
    assert!(image.mime_type.as_deref().unwrap().contains("image"));
    assert_eq!(image.model, "gpt-image-1");
    assert!(image.is_base64());
    assert_eq!(image.tokens().input, Some(10));
    assert_eq!(image.tokens().output, Some(4160));
    billed_once_as_image(&image, "openai", "gpt-image-1");
    saves_a_real_image(&image).await;
    cassette.assert_all_matched().await;
}

#[tokio::test]
async fn gemini_can_paint_images() {
    let cassette =
        start("image_basic_functionality_gemini_gemini-3_1-flash-lite-image_can_paint_images")
            .await;
    let config = config_for(&cassette, "gemini");
    let model = "gemini-3.1-flash-lite-image";
    let image = paint("a siamese cat", options(model, Some("gemini"), config))
        .await
        .unwrap()
        .into_image();
    assert_eq!(image.mime_type.as_deref(), Some("image/jpeg"));
    assert_eq!(image.model, model);
    // promptTokenCount 5, candidatesTokenCount 1529.
    assert_eq!(image.tokens().input, Some(5));
    assert_eq!(image.tokens().output, Some(1529));
    billed_once_as_image(&image, "gemini", model);
    saves_a_real_image(&image).await;
    cassette.assert_all_matched().await;
}

#[tokio::test]
async fn openrouter_can_paint_images() {
    let cassette = start(
        "image_basic_functionality_openrouter_google_gemini-3_1-flash-lite-image_can_paint_images",
    )
    .await;
    let config = config_for(&cassette, "openrouter");
    let model = "google/gemini-3.1-flash-lite-image";
    let image = paint("a siamese cat", options(model, Some("openrouter"), config))
        .await
        .unwrap()
        .into_image();
    assert_eq!(image.mime_type.as_deref(), Some("image/jpeg"));
    assert_eq!(image.model, model);
    assert_eq!(image.tokens().reported_cost, Some(0.033601));
    assert_eq!(image.cost().total(), Some(0.033601));
    billed_once_as_image(&image, "openrouter", model);
    saves_a_real_image(&image).await;
    cassette.assert_all_matched().await;
}

#[tokio::test]
async fn xai_can_paint_images_and_downloads_the_hosted_file() {
    let cassette = start("image_basic_functionality_xai_grok-imagine-image_can_paint_images").await;
    let config = config_for(&cassette, "xai");
    let mut image = paint(
        "a siamese cat",
        options("grok-imagine-image", Some("xai"), config),
    )
    .await
    .unwrap()
    .into_image();
    assert_eq!(image.mime_type.as_deref(), Some("image/jpeg"));
    assert_eq!(image.model, "grok-imagine-image");
    assert!(!image.is_base64());
    billed_once_as_image(&image, "xai", "grok-imagine-image");
    hosted_on(&mut image, &cassette);
    saves_a_real_image(&image).await;
    cassette.assert_all_matched().await;
}

#[tokio::test]
async fn openai_paints_several_images_in_one_request() {
    let cassette = start(
        "image_basic_functionality_openai_gpt-image-1_5_paints_several_images_in_one_request",
    )
    .await;
    let config = config_for(&cassette, "openai");
    let images = paint(
        "a siamese cat",
        PaintOptions {
            count: Some(2),
            ..options("gpt-image-1.5", Some("openai"), config)
        },
    )
    .await
    .unwrap();
    assert!(matches!(images, Images::Many(_)));
    let images = images.into_vec();
    assert_eq!(images.len(), 2);
    for image in &images {
        assert!(image.mime_type.as_deref().unwrap().contains("image"));
        saves_a_real_image(image).await;
    }
    // Billed once: only the first image carries the call's usage.
    billed_once_as_image(&images[0], "openai", "gpt-image-1.5");
    assert_eq!(images[0].tokens().output, Some(13178));
    assert!(images[1].usage_entries.is_empty());
    assert_eq!(images[1].tokens(), Tokens::default());
    cassette.assert_all_matched().await;
}

#[tokio::test]
async fn validates_model_existence() {
    let err = paint(
        "a cat",
        PaintOptions {
            model: Some("invalid-model"),
            ..Default::default()
        },
    )
    .await
    .unwrap_err();
    assert!(matches!(err, Error::ModelNotFound(_)), "{err:?}");
}

#[tokio::test]
async fn gpt_image_1_5_supports_image_edits_with_multiple_images() {
    let cassette =
        start("image_basic_functionality_gpt-image-1_5_supports_image_edits_with_multiple_images")
            .await;
    let config = config_for(&cassette, "openai");
    let image = paint(
        PROMPT,
        PaintOptions {
            with: vec![Attachment::new(image_path()), Attachment::new(image_path())],
            ..options("gpt-image-1.5", None, config)
        },
    )
    .await
    .unwrap()
    .into_image();
    assert!(image.is_base64());
    assert_eq!(image.mime_type.as_deref(), Some("image/png"));
    assert_eq!(image.model, "gpt-image-1.5");
    assert!(image.tokens().input.unwrap() > 0);
    saves_a_real_image(&image).await;
    cassette.assert_all_matched().await;
}

// ---- edit functionality -------------------------------------------------------------------

#[tokio::test]
async fn supports_image_edits_with_a_valid_local_png() {
    let cassette = start(
        "image_edit_functionality_with_local_files_supports_image_edits_with_a_valid_local_png",
    )
    .await;
    let config = config_for(&cassette, "openai");
    let image = paint(
        PROMPT,
        PaintOptions {
            with: vec![Attachment::new(image_path())],
            ..options("gpt-image-1.5", None, config)
        },
    )
    .await
    .unwrap()
    .into_image();
    assert!(image.is_base64());
    assert!(!image.data.as_deref().unwrap().is_empty());
    assert_eq!(image.mime_type.as_deref(), Some("image/png"));
    assert_eq!(image.model, "gpt-image-1.5");
    assert!(image.tokens().input.unwrap() > 0);
    assert!(!image.to_blob().await.unwrap().is_empty());
    cassette.assert_all_matched().await;
}

#[tokio::test]
async fn rejects_edits_with_a_non_png_local_file() {
    let wav = Attachment::from_bytes(b"RIFF\0\0\0\0WAVE".to_vec(), "ruby.wav", None);
    let mut config = Config::default();
    config.openai_api_key("test-key");
    let err = paint(
        PROMPT,
        PaintOptions {
            with: vec![wav],
            ..options("gpt-image-1.5", None, Arc::new(config))
        },
    )
    .await
    .unwrap_err();
    assert!(matches!(err, Error::UnsupportedAttachment(_)), "{err:?}");
    assert!(
        err.to_string()
            .contains("Unsupported attachment type: audio/wav"),
        "{err}"
    );
}

#[tokio::test]
async fn customizes_image_output() {
    let cassette = start("image_edit_functionality_with_local_files_customizes_image_output").await;
    let config = config_for(&cassette, "openai");
    let image = paint(
        PROMPT,
        PaintOptions {
            with: vec![Attachment::new(image_path())],
            provider_options: json!({ "size": "1024x1024", "quality": "low" }),
            ..options("gpt-image-1.5", None, config)
        },
    )
    .await
    .unwrap()
    .into_image();
    assert!(image.is_base64());
    assert_eq!(image.mime_type.as_deref(), Some("image/png"));
    assert!(image.tokens().output.unwrap() > 0);
    assert!(!image.to_blob().await.unwrap().is_empty());
    cassette.assert_all_matched().await;
}

#[tokio::test]
async fn openrouter_edits_images_passed_via_with() {
    let cassette = start("image_edit_functionality_with_openrouter_reference_images_openrouter_google_gemini-3_1-flash-lite-image_edits_images_passed_via_with").await;
    let config = config_for(&cassette, "openrouter");
    let model = "google/gemini-3.1-flash-lite-image";
    let image = paint(
        PROMPT,
        PaintOptions {
            with: vec![Attachment::new(image_path())],
            ..options(model, Some("openrouter"), config)
        },
    )
    .await
    .unwrap()
    .into_image();
    assert!(image.is_base64());
    assert!(image.mime_type.as_deref().unwrap().contains("image"));
    let reported = image.tokens().reported_cost.unwrap();
    assert!(reported > 0.0);
    assert_eq!(image.cost().total(), Some(reported));
    saves_a_real_image(&image).await;
    cassette.assert_all_matched().await;
}

#[tokio::test]
async fn rejects_edits_with_a_url_having_invalid_content_type() {
    let cassette = start("image_edit_functionality_with_remote_urls_rejects_edits_with_a_url_having_invalid_content_type").await;
    let config = config_for(&cassette, "openai");
    let with = vec![Attachment::new(
        "https://rubyllm.com/assets/images/logotype.svg",
    )];
    let err = paint(
        PROMPT,
        PaintOptions {
            with,
            ..options("gpt-image-1.5", None, config)
        },
    )
    .await
    .unwrap_err();
    assert!(matches!(err, Error::BadRequest(..)), "{err:?}");
    assert!(err.to_string().contains("Invalid image data"), "{err}");
    cassette.assert_all_matched().await;
}

#[tokio::test]
async fn rejects_edits_with_a_url_that_returns_404() {
    let cassette = start(
        "image_edit_functionality_with_remote_urls_rejects_edits_with_a_url_that_returns_404",
    )
    .await;
    let config = config_for(&cassette, "openai");
    let with = vec![Attachment::new(
        "https://rubyllm.com/some-asset-that-does-not-exist.png",
    )];
    let err = paint(
        PROMPT,
        PaintOptions {
            with,
            ..options("gpt-image-1.5", None, config)
        },
    )
    .await
    .unwrap_err();
    assert!(matches!(err, Error::BadRequest(..)), "{err:?}");
    assert!(err.to_string().contains("404"), "{err}");
    cassette.assert_all_matched().await;
}

#[tokio::test]
async fn gpt_image_2_accepts_a_16px_multiple_size_on_edits() {
    let cassette = start("image_edit_functionality_with_flexible_sizes_gpt-image-2_accepts_a_16px-multiple_size_on_edits").await;
    let config = config_for(&cassette, "openai");
    let image = paint(
        PROMPT,
        PaintOptions {
            with: vec![Attachment::new(image_path())],
            size: Some("1536x864"),
            provider_options: json!({ "quality": "low" }),
            ..options("gpt-image-2", None, config)
        },
    )
    .await
    .unwrap()
    .into_image();
    assert!(image.is_base64());
    assert_eq!(image.mime_type.as_deref(), Some("image/png"));
    assert_eq!(image.model, "gpt-image-2");
    assert!(!image.to_blob().await.unwrap().is_empty());
    cassette.assert_all_matched().await;
}

#[tokio::test]
async fn xai_supports_image_edits_with_reference_images() {
    let cassette = start("image_edit_functionality_with_xai_reference_images_xai_grok-imagine-image-quality_supports_image_edits_with_reference_images").await;
    let config = config_for(&cassette, "xai");
    let model = "grok-imagine-image-quality";
    let mut image = paint(
        PROMPT,
        PaintOptions {
            with: vec![Attachment::new(image_path()), Attachment::new(image_path())],
            ..options(model, Some("xai"), config)
        },
    )
    .await
    .unwrap()
    .into_image();
    assert!(image.url.is_some());
    assert!(image.mime_type.as_deref().unwrap().contains("image"));
    assert_eq!(image.model, model);
    hosted_on(&mut image, &cassette);
    saves_a_real_image(&image).await;
    cassette.assert_all_matched().await;
}

#[tokio::test]
async fn xai_rejects_masks() {
    let mut config = Config::default();
    config.set("xai_api_key", "test-key");
    let err = paint(
        PROMPT,
        PaintOptions {
            with: vec![Attachment::new(image_path())],
            mask: Some(Attachment::new(image_path())),
            ..options("grok-imagine-image-quality", Some("xai"), Arc::new(config))
        },
    )
    .await
    .unwrap_err();
    assert!(err.to_string().contains("mask"), "{err}");
}

// ---- #cost --------------------------------------------------------------------------------

/// image_spec.rb `#cost`: text input tokens at the text price, image input tokens and output at
/// the image price. `Image#cost` calls `Cost::images` with the registry model; the spec stubs
/// the registry, so this passes the model directly.
#[test]
fn cost_splits_image_input_by_modality_like_chat_costs() {
    let mut model = Model::default_for("gpt-image-1.5", "openai");
    let tier = |input: f64, output: Option<f64>| PricingCategory {
        standard: Some(PricingTier {
            input_per_million: Some(input),
            output_per_million: output,
            ..Default::default()
        }),
        ..Default::default()
    };
    model.pricing = Pricing {
        text_tokens: Some(tier(5.0, None)),
        images: Some(tier(10.0, Some(40.0))),
        ..Default::default()
    };
    let image = Image::new(
        "gpt-image-1.5",
        json!({ "input_tokens": 350, "input_tokens_details": { "text_tokens": 100, "image_tokens": 250 }, "output_tokens": 50 }),
    );
    let tokens = image.tokens();
    assert_eq!(tokens.input, Some(350));
    assert_eq!(tokens.output, Some(50));
    let cost = Cost::images(
        &tokens,
        Some(&model),
        Some(&json!({ "text_tokens": 100, "image_tokens": 250 })),
    );
    assert!((cost.input.unwrap() - 0.003).abs() < 1e-10);
    assert!((cost.output.unwrap() - 0.002).abs() < 1e-10);
    assert!((cost.total().unwrap() - 0.005).abs() < 1e-10);
}

/// One multipart part: `(name, filename, content_type, bytes)`.
type Part = (String, Option<String>, Option<String>, Vec<u8>);

/// Multipart parts parsed from a request body.
fn multipart_parts(body: &[u8], content_type: &str) -> Vec<Part> {
    let boundary = content_type
        .split("boundary=")
        .nth(1)
        .expect("boundary")
        .trim_matches('"');
    let delimiter = format!("--{boundary}");
    let text = body;
    let mut parts = Vec::new();
    let find = |hay: &[u8], needle: &[u8], from: usize| {
        hay[from..]
            .windows(needle.len())
            .position(|w| w == needle)
            .map(|p| p + from)
    };
    let mut pos = find(text, delimiter.as_bytes(), 0).expect("first delimiter") + delimiter.len();
    while let Some(next) = find(text, delimiter.as_bytes(), pos) {
        let section = &text[pos..next];
        let section = section.strip_prefix(b"\r\n").unwrap_or(section);
        let split = find(section, b"\r\n\r\n", 0).expect("header end");
        let headers = String::from_utf8_lossy(&section[..split]).to_string();
        let mut data = section[split + 4..].to_vec();
        if data.ends_with(b"\r\n") {
            data.truncate(data.len() - 2);
        }
        let attr = |key: &str| {
            headers.split(['\r', '\n', ';']).find_map(|h| {
                h.trim()
                    .strip_prefix(&format!("{key}=\""))
                    .map(|v| v.trim_end_matches('"').to_string())
            })
        };
        let ctype = headers.lines().find_map(|l| {
            l.to_ascii_lowercase()
                .strip_prefix("content-type:")
                .map(|v| v.trim().to_string())
        });
        parts.push((
            attr("name").unwrap_or_default(),
            attr("filename"),
            ctype,
            data,
        ));
        pos = next + delimiter.len();
        if text[pos..].starts_with(b"--") {
            break;
        }
    }
    parts
}

// Ruby's `render_edit_payload` for a model without JSON image references (dall-e-2): a
// multipart upload of model, prompt, n, the image file, and the mask file.
#[tokio::test]
async fn dall_e_edits_upload_the_image_and_mask_as_multipart_parts() {
    let server = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .and(wiremock::matchers::path("/v1/images/edits"))
        .respond_with(wiremock::ResponseTemplate::new(200).set_body_json(json!({
            "created": 1, "data": [{ "b64_json": "aGVsbG8=" }]
        })))
        .expect(1)
        .mount(&server)
        .await;
    let mut config = Config::default();
    config.set("openai_api_base", format!("{}/v1", server.uri()));
    config.set("openai_api_key", "test");
    config.max_retries = 0;
    let png = std::fs::read(format!(
        "{}/tests/fixtures/ruby.png",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap();

    let image = paint(
        "Add a hat",
        PaintOptions {
            model: Some("dall-e-2"),
            provider: Some("openai"),
            assume_model_exists: true,
            with: vec![Attachment::from_bytes(
                png.clone(),
                "ruby.png",
                Some("image/png"),
            )],
            mask: Some(Attachment::from_bytes(
                png.clone(),
                "mask.png",
                Some("image/png"),
            )),
            provider_options: json!({ "response_format": "b64_json" }),
            config: Some(Arc::new(config)),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    let Images::One(image) = image else {
        panic!("one image")
    };
    assert!(image.is_base64());

    let requests = server.received_requests().await.unwrap();
    let content_type = requests[0]
        .headers
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    assert!(
        content_type.starts_with("multipart/form-data"),
        "{content_type}"
    );
    let parts = multipart_parts(&requests[0].body, &content_type);
    let names: Vec<&str> = parts.iter().map(|p| p.0.as_str()).collect();
    assert_eq!(
        names,
        ["model", "prompt", "n", "image", "mask", "response_format"]
    );
    assert_eq!(parts[0].3, b"dall-e-2");
    assert_eq!(parts[1].3, b"Add a hat");
    assert_eq!(parts[2].3, b"1");
    assert_eq!(parts[3].1.as_deref(), Some("ruby.png"));
    assert_eq!(parts[3].2.as_deref(), Some("image/png"));
    assert_eq!(parts[3].3, png);
    assert_eq!(parts[4].1.as_deref(), Some("mask.png"));
    assert_eq!(parts[5].3, b"b64_json");
}

#[tokio::test]
async fn dall_e_edits_reject_non_image_uploads() {
    let mut config = Config::default();
    config.set("openai_api_base", "http://127.0.0.1:9/v1");
    config.set("openai_api_key", "test");
    let err = paint(
        "Add a hat",
        PaintOptions {
            model: Some("dall-e-2"),
            provider: Some("openai"),
            assume_model_exists: true,
            with: vec![Attachment::from_bytes(
                b"%PDF-1.4".to_vec(),
                "doc.pdf",
                Some("application/pdf"),
            )],
            config: Some(Arc::new(config)),
            ..Default::default()
        },
    )
    .await
    .unwrap_err();
    assert!(matches!(err, Error::UnsupportedAttachment(_)), "{err:?}");
}
