//! Ports of RubyLLM's provider media unit specs (`spec/ruby_llm/providers/{openrouter,xai,gpustack,
//! mistral}/{images,videos,embeddings,speech,transcription,ocr}_spec.rb`). The Ruby specs call the
//! protocol's render/parse seams directly or stub HTTP; here a wiremock server stands in for the
//! provider, the public API (`paint`, `animate_later`, `embed`, `speak_stream`, `transcribe_stream`,
//! `ocr`) runs the real render/parse path, and the tests assert the request bodies Ruby asserts.

use std::sync::Arc;

use base64::Engine;
use rust_llm::files::UploadedFile;
use rust_llm::{
    AnimateOptions, Attachment, Config, EmbedOptions, Error, OcrOptions, PaintOptions, Resolution,
    SpeakOptions, TranscribeOptions, Vectors, Video, VideoSource, VideoStatus, animate_later,
    embed, ocr, paint, speak_stream, transcribe_stream,
};
use serde_json::{Value, json};
use wiremock::matchers::{any, header, method, path, query_param};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

fn fixture(name: &str) -> String {
    format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
}

/// Every provider in this file pointed at `server`, as `include_context 'with configured RubyLLM'`.
fn config(server: &MockServer) -> Arc<Config> {
    let mut c = Config::default();
    for (provider, prefix) in [
        ("openrouter", "/api/v1"),
        ("xai", "/v1"),
        ("mistral", "/v1"),
        ("gpustack", "/v1"),
    ] {
        c.set(
            format!("{provider}_api_base"),
            format!("{}{prefix}", server.uri()),
        );
        c.set(format!("{provider}_api_key"), "test");
    }
    c.max_retries = 0;
    Arc::new(c)
}

/// A server answering every request with `body` as JSON.
async fn json_server(body: Value) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(any())
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&server)
        .await;
    server
}

/// A server answering every request with `body` under `content_type`.
async fn raw_server(content_type: &str, body: impl Into<Vec<u8>>) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(any())
        .respond_with(ResponseTemplate::new(200).set_body_raw(body.into(), content_type))
        .mount(&server)
        .await;
    server
}

async fn received(server: &MockServer) -> Vec<Request> {
    server.received_requests().await.unwrap_or_default()
}

/// The single request's path and JSON body.
async fn only_request(server: &MockServer) -> (String, Value) {
    let requests = received(server).await;
    assert_eq!(requests.len(), 1, "expected exactly one request");
    let body = serde_json::from_slice(&requests[0].body).expect("json body");
    (requests[0].url.path().to_string(), body)
}

fn sse(events: &[Value]) -> String {
    events.iter().map(|e| format!("data: {e}\n\n")).collect()
}

/// A text field of a multipart form body.
fn form_field(body: &[u8], name: &str) -> Option<String> {
    let body = String::from_utf8_lossy(body);
    let marker = format!("name=\"{name}\"\r\n\r\n");
    let start = body.find(&marker)? + marker.len();
    let end = body[start..].find("\r\n--")?;
    Some(body[start..start + end].to_string())
}

fn unsupported_mentions(result: Result<impl std::fmt::Debug, Error>, needle: &str) {
    match result {
        Err(Error::UnsupportedAttachment(message)) => {
            assert!(message.contains(needle), "{message}")
        }
        other => panic!("expected UnsupportedAttachmentError mentioning {needle}, got {other:?}"),
    }
}

fn uploaded(id: &str, provider: &str, filename: &str, mime_type: &str) -> UploadedFile {
    UploadedFile {
        id: id.into(),
        provider: provider.into(),
        filename: Some(filename.into()),
        byte_size: None,
        created_at: None,
        expires_at: None,
        status: None,
        mime_type: Some(mime_type.into()),
        purpose: None,
        uri: None,
        downloadable: None,
        metadata: Value::Null,
    }
}

// ---- OpenRouter images ------------------------------------------------------------------------

const OPENROUTER_IMAGE_MODEL: &str = "google/gemini-2.5-flash-image";

fn openrouter_image_body() -> Value {
    json!({
        "created": 0,
        "data": [{ "b64_json": "/9j/4AAQSkZJRg==", "media_type": "image/jpeg" }],
        "usage": { "prompt_tokens": 7, "completion_tokens": 1120, "total_tokens": 1127, "cost": 0.0336 }
    })
}

fn openrouter_paint<'a>(server: &MockServer) -> PaintOptions<'a> {
    PaintOptions {
        model: Some(OPENROUTER_IMAGE_MODEL),
        provider: Some("openrouter"),
        config: Some(config(server)),
        ..Default::default()
    }
}

// spec: providers/openrouter/images_spec.rb:16 #render_image_payload > renders a generation payload
#[tokio::test]
async fn openrouter_image_payload_drops_the_size() {
    let server = json_server(openrouter_image_body()).await;
    paint(
        "a cute cat",
        PaintOptions {
            size: Some("1024x1024"),
            ..openrouter_paint(&server)
        },
    )
    .await
    .unwrap();
    let (path, body) = only_request(&server).await;
    assert_eq!(path, "/api/v1/images");
    assert_eq!(
        body,
        json!({ "model": OPENROUTER_IMAGE_MODEL, "prompt": "a cute cat" })
    );
}

// spec: providers/openrouter/images_spec.rb:23 #render_image_payload > merges provider options
#[tokio::test]
async fn openrouter_image_payload_merges_provider_options() {
    let server = json_server(openrouter_image_body()).await;
    paint(
        "a cute cat",
        PaintOptions {
            provider_options: json!({ "aspect_ratio": "16:9" }),
            ..openrouter_paint(&server)
        },
    )
    .await
    .unwrap();
    let (_, body) = only_request(&server).await;
    assert_eq!(body["aspect_ratio"], "16:9");
}

// spec: providers/openrouter/images_spec.rb:39 #render_image_payload > passes reference URLs through untouched
#[tokio::test]
async fn openrouter_image_reference_urls_pass_through() {
    let server = json_server(openrouter_image_body()).await;
    let with = vec![Attachment::new("https://example.com/logo.png")];
    paint(
        "make it green",
        PaintOptions {
            with,
            ..openrouter_paint(&server)
        },
    )
    .await
    .unwrap();
    let (_, body) = only_request(&server).await;
    assert_eq!(
        body["input_references"],
        json!([{ "type": "image_url", "image_url": { "url": "https://example.com/logo.png" } }])
    );
}

// spec: providers/openrouter/images_spec.rb:48 #render_image_payload > rejects non-image references
#[tokio::test]
async fn openrouter_image_rejects_non_image_references() {
    let server = json_server(openrouter_image_body()).await;
    let with = vec![Attachment::new(fixture("ruby.wav"))];
    unsupported_mentions(
        paint(
            "make it green",
            PaintOptions {
                with,
                ..openrouter_paint(&server)
            },
        )
        .await,
        "audio/wav",
    );
    assert!(received(&server).await.is_empty());
}

// spec: providers/openrouter/images_spec.rb:63 #validate_paint_inputs! > rejects masks
#[tokio::test]
async fn openrouter_image_rejects_masks() {
    let server = json_server(openrouter_image_body()).await;
    let options = PaintOptions {
        with: vec![Attachment::new(fixture("ruby.png"))],
        mask: Some(Attachment::new(fixture("ruby.png"))),
        ..openrouter_paint(&server)
    };
    unsupported_mentions(paint("make it green", options).await, "image mask");
    assert!(received(&server).await.is_empty());
}

// spec: providers/openrouter/images_spec.rb:100 #parse_image_response > defaults the MIME type when the provider omits it
#[tokio::test]
async fn openrouter_image_defaults_the_mime_type() {
    let mut body = openrouter_image_body();
    body["data"][0]
        .as_object_mut()
        .unwrap()
        .remove("media_type");
    let server = json_server(body).await;
    let image = paint("a cute cat", openrouter_paint(&server))
        .await
        .unwrap()
        .into_image();
    assert_eq!(image.mime_type.as_deref(), Some("image/png"));
}

// spec: providers/openrouter/images_spec.rb:108 #parse_image_response > raises an error when no image data is returned
#[tokio::test]
async fn openrouter_image_without_data_is_an_error() {
    let server = json_server(json!({ "data": [] })).await;
    match paint("a cute cat", openrouter_paint(&server)).await {
        Err(Error::Api(message, _)) => {
            assert!(message.contains("Unexpected response format"), "{message}")
        }
        other => panic!("expected an Unexpected response format error, got {other:?}"),
    }
}

// ---- xAI images -------------------------------------------------------------------------------

fn xai_image_body() -> Value {
    json!({ "data": [{ "b64_json": "aGk=" }] })
}

fn xai_paint<'a>(server: &MockServer, model: &'a str) -> PaintOptions<'a> {
    PaintOptions {
        model: Some(model),
        provider: Some("xai"),
        config: Some(config(server)),
        ..Default::default()
    }
}

// UPSTREAM-REMOVED in 2.1 (was spec: providers/xai/images_spec.rb:7) .render_image_payload > drops the size parameter xAI rejects
#[tokio::test]
async fn xai_image_payload_drops_the_size() {
    let server = json_server(xai_image_body()).await;
    paint(
        "a cute cat",
        PaintOptions {
            size: Some("1024x1024"),
            ..xai_paint(&server, "grok-imagine-image")
        },
    )
    .await
    .unwrap();
    let (path, body) = only_request(&server).await;
    assert_eq!(path, "/v1/images/generations");
    assert_eq!(
        body,
        json!({ "model": "grok-imagine-image", "prompt": "a cute cat" })
    );
}

// spec: providers/xai/images_spec.rb:13 .render_image_payload > merges provider options
#[tokio::test]
async fn xai_image_payload_merges_provider_options() {
    let server =
        json_server(json!({ "data": [{ "b64_json": "aGk=" }, { "b64_json": "aGk=" }] })).await;
    paint(
        "a cute cat",
        PaintOptions {
            provider_options: json!({ "n": 2 }),
            ..xai_paint(&server, "grok-imagine-image")
        },
    )
    .await
    .unwrap();
    let (_, body) = only_request(&server).await;
    assert_eq!(body["n"], 2);
}

// spec: providers/xai/images_spec.rb:30 .render_image_payload > passes remote reference images through as URLs
#[tokio::test]
async fn xai_image_remote_references_pass_through_as_urls() {
    let server = json_server(xai_image_body()).await;
    let with = vec![Attachment::new("https://example.com/logo.png")];
    paint(
        "combine the logos",
        PaintOptions {
            with,
            ..xai_paint(&server, "grok-imagine-image-quality")
        },
    )
    .await
    .unwrap();
    let (path, body) = only_request(&server).await;
    assert_eq!(path, "/v1/images/edits");
    assert_eq!(
        body["images"],
        json!([{ "type": "image_url", "url": "https://example.com/logo.png" }])
    );
}

// ---- OpenRouter videos ------------------------------------------------------------------------

fn openrouter_animate<'a>(server: &MockServer, model: &'a str) -> AnimateOptions<'a> {
    AnimateOptions {
        model: Some(model),
        provider: Some("openrouter"),
        assume_model_exists: true,
        config: Some(config(server)),
        ..Default::default()
    }
}

/// A server that accepts a job with `accepted` and answers `GET videos/abc123` with `status`.
async fn video_job_server(accepted: Value, status: Value) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(accepted))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/api/v1/videos/abc123"))
        .respond_with(ResponseTemplate::new(200).set_body_json(status))
        .mount(&server)
        .await;
    server
}

// spec: providers/openrouter/videos_spec.rb:14 #render_video_payload > sends model, prompt, and provider options
#[tokio::test]
async fn openrouter_video_sends_model_prompt_and_provider_options() {
    let server = json_server(json!({ "id": "abc123", "status": "pending" })).await;
    let options = AnimateOptions {
        provider_options: json!({ "duration": 1, "resolution": "480p" }),
        ..openrouter_animate(&server, "x-ai/grok-imagine-video")
    };
    animate_later(Some("a calm ocean wave at sunset"), options)
        .await
        .unwrap();
    let (path, body) = only_request(&server).await;
    assert_eq!(path, "/api/v1/videos");
    assert_eq!(
        body,
        json!({ "model": "x-ai/grok-imagine-video", "prompt": "a calm ocean wave at sunset", "duration": 1, "resolution": "480p" })
    );
}

// spec: providers/openrouter/videos_spec.rb:30 #render_video_payload > maps reference images to first and last frames
#[tokio::test]
async fn openrouter_video_maps_references_to_first_and_last_frames() {
    let server = json_server(json!({ "id": "abc123", "status": "pending" })).await;
    let options = AnimateOptions {
        with: vec![
            Attachment::new("https://example.com/first.jpg"),
            Attachment::new("https://example.com/last.jpg"),
        ],
        ..openrouter_animate(&server, "google/veo-3.1-lite")
    };
    animate_later(Some("the camera slowly pushes in"), options)
        .await
        .unwrap();
    let (_, body) = only_request(&server).await;
    assert_eq!(
        body["frame_images"],
        json!([
            { "type": "image_url", "image_url": { "url": "https://example.com/first.jpg" }, "frame_type": "first_frame" },
            { "type": "image_url", "image_url": { "url": "https://example.com/last.jpg" }, "frame_type": "last_frame" }
        ])
    );
}

// spec: providers/openrouter/videos_spec.rb:59 #parse_video_job > reads the job id and status from the accepted job
#[tokio::test]
async fn openrouter_video_reads_the_job_id_and_status() {
    let accepted = json!({ "id": "abc123", "polling_url": "https://openrouter.ai/api/v1/videos/abc123", "status": "pending" });
    let server =
        video_job_server(accepted, json!({ "id": "abc123", "status": "in_progress" })).await;
    let mut job = animate_later(
        Some("a wave"),
        openrouter_animate(&server, "x-ai/grok-imagine-video"),
    )
    .await
    .unwrap();
    assert_eq!(job.id, "abc123");
    assert!(job.is_pending());
    // `video_job_url(job)`: the poll goes to videos/abc123.
    job.refresh().await.unwrap();
    let requests = received(&server).await;
    assert_eq!(requests[1].url.path(), "/api/v1/videos/abc123");
}

// spec: providers/openrouter/videos_spec.rb:77 #parse_video_job_status > stays pending while the job is in progress
#[tokio::test]
async fn openrouter_video_stays_pending_while_in_progress() {
    let status = json!({ "id": "abc123", "status": "in_progress" });
    let server = video_job_server(
        json!({ "id": "abc123", "status": "pending" }),
        status.clone(),
    )
    .await;
    let mut job = animate_later(
        Some("a wave"),
        openrouter_animate(&server, "x-ai/grok-imagine-video"),
    )
    .await
    .unwrap();
    job.refresh().await.unwrap();
    assert_eq!(job.status, VideoStatus::Pending);
    assert_eq!(job.raw, status);
}

// spec: providers/openrouter/videos_spec.rb:85 #parse_video_job_status > completes when the job reports completed
#[tokio::test]
async fn openrouter_video_completes_when_reported_completed() {
    let status = json!({
        "id": "abc123",
        "status": "completed",
        "unsigned_urls": ["https://openrouter.ai/api/v1/videos/abc123/content?index=0"],
        "usage": { "cost": 0.05 }
    });
    let server = video_job_server(
        json!({ "id": "abc123", "status": "pending" }),
        status.clone(),
    )
    .await;
    let mut job = animate_later(
        Some("a wave"),
        openrouter_animate(&server, "x-ai/grok-imagine-video"),
    )
    .await
    .unwrap();
    job.refresh().await.unwrap();
    assert_eq!(job.status, VideoStatus::Completed);
    assert_eq!(job.raw, status);
}

// spec: providers/openrouter/videos_spec.rb:113 #parse_video_job_status > fails with the reported error
#[tokio::test]
async fn openrouter_video_fails_with_the_reported_error() {
    let status = json!({ "status": "failed", "error": "provider rejected" });
    let server = video_job_server(json!({ "id": "abc123", "status": "pending" }), status).await;
    let mut job = animate_later(
        Some("a wave"),
        openrouter_animate(&server, "x-ai/grok-imagine-video"),
    )
    .await
    .unwrap();
    job.refresh().await.unwrap();
    assert_eq!(job.status, VideoStatus::Failed);
    assert_eq!(job.error.as_deref(), Some("provider rejected"));
}

// spec: providers/openrouter/videos_spec.rb:123 #download_video > downloads the job content with the API connection
#[tokio::test]
async fn openrouter_video_downloads_content_with_the_api_connection() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({ "id": "abc123", "status": "completed" })),
        )
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/api/v1/videos/abc123/content"))
        .and(query_param("index", "0"))
        .and(header("authorization", "Bearer test"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(b"mp4 bytes".to_vec(), "video/mp4"))
        .expect(1)
        .mount(&server)
        .await;
    let mut job = animate_later(
        Some("a wave"),
        openrouter_animate(&server, "x-ai/grok-imagine-video"),
    )
    .await
    .unwrap();
    let video = job
        .video()
        .await
        .unwrap()
        .expect("completed job has a video");
    assert_eq!(video.data.as_deref(), Some(&b"mp4 bytes"[..]));
    assert_eq!(video.mime_type.as_deref(), Some("video/mp4"));
    assert_eq!(video.model.as_deref(), Some("x-ai/grok-imagine-video"));
}

// ---- xAI videos -------------------------------------------------------------------------------

fn xai_animate<'a>(server: &MockServer) -> AnimateOptions<'a> {
    AnimateOptions {
        model: Some("grok-imagine-video"),
        provider: Some("xai"),
        config: Some(config(server)),
        ..Default::default()
    }
}

fn xai_job() -> Value {
    json!({ "request_id": "4482fadb" })
}

// spec: providers/xai/videos_spec.rb:32 #render_video_payload > references a remote image by URL for image-to-video
#[tokio::test]
async fn xai_video_references_a_remote_image_by_url() {
    let server = json_server(xai_job()).await;
    let options = AnimateOptions {
        with: vec![Attachment::new("https://example.com/waterfall.png")],
        ..xai_animate(&server)
    };
    animate_later(Some("make the water crash down"), options)
        .await
        .unwrap();
    let (path, body) = only_request(&server).await;
    assert_eq!(path, "/v1/videos/generations");
    assert_eq!(
        body["image"],
        json!({ "url": "https://example.com/waterfall.png" })
    );
}

// spec: providers/xai/videos_spec.rb:42 #render_video_payload > inlines a local image as a data URI
#[tokio::test]
async fn xai_video_inlines_a_local_image_as_a_data_uri() {
    let server = json_server(xai_job()).await;
    let options = AnimateOptions {
        with: vec![Attachment::new(fixture("ruby.png"))],
        ..xai_animate(&server)
    };
    animate_later(Some("bring the logo to life"), options)
        .await
        .unwrap();
    let (_, body) = only_request(&server).await;
    assert!(
        body["image"]["url"]
            .as_str()
            .unwrap()
            .starts_with("data:image/png;base64,")
    );
}

// spec: providers/xai/videos_spec.rb:88 video editing and extension > references uploaded images and videos using file_id fields
#[tokio::test]
async fn xai_video_references_uploaded_files_by_file_id() {
    let file = uploaded("file_video", "xai", "clip.mp4", "video/mp4");
    let server = json_server(xai_job()).await;
    let options = AnimateOptions {
        with: vec![Attachment::from_uploaded(file.clone())],
        ..xai_animate(&server)
    };
    animate_later(Some("Turn the background blue"), options)
        .await
        .unwrap();
    let options = AnimateOptions {
        extend: Some(VideoSource::Attachment(Attachment::from_uploaded(file))),
        ..xai_animate(&server)
    };
    animate_later(Some("Continue"), options).await.unwrap();

    let requests = received(&server).await;
    let edit: Value = serde_json::from_slice(&requests[0].body).unwrap();
    let extension: Value = serde_json::from_slice(&requests[1].body).unwrap();
    assert_eq!(requests[0].url.path(), "/v1/videos/edits");
    assert_eq!(edit["video"], json!({ "file_id": "file_video" }));
    assert_eq!(requests[1].url.path(), "/v1/videos/extensions");
    assert_eq!(extension["video"], json!({ "file_id": "file_video" }));
}

// spec: providers/xai/videos_spec.rb:109 video editing and extension > rejects conflicting sources and invalid extension inputs before sending requests
// `extend: [video, video]` cannot be expressed with `VideoSource`; the "exactly one video" error is
// reached by a `Video` carrying no clip, the one source the type admits that is not one video.
#[tokio::test]
async fn xai_video_rejects_conflicting_and_invalid_extension_sources() {
    let server = json_server(xai_job()).await;
    let video = || {
        VideoSource::Attachment(Attachment::from_bytes(
            b"mp4 bytes".to_vec(),
            "clip.mp4",
            None,
        ))
    };

    let conflicting = AnimateOptions {
        with: vec![Attachment::new(fixture("ruby.png"))],
        extend: Some(video()),
        ..xai_animate(&server)
    };
    match animate_later(Some("Continue"), conflicting).await {
        Err(Error::Argument(message)) => {
            assert!(message.contains("cannot be combined"), "{message}")
        }
        other => panic!("expected ArgumentError, got {other:?}"),
    }

    let empty = VideoSource::Video(Video::new(None, Some("video/mp4".into()), Value::Null));
    match animate_later(
        Some("Continue"),
        AnimateOptions {
            extend: Some(empty),
            ..xai_animate(&server)
        },
    )
    .await
    {
        Err(Error::Argument(message)) => {
            assert!(message.contains("exactly one video"), "{message}")
        }
        other => panic!("expected ArgumentError, got {other:?}"),
    }

    let image = VideoSource::from(fixture("ruby.png").as_str());
    unsupported_mentions(
        animate_later(
            Some("Continue"),
            AnimateOptions {
                extend: Some(image),
                ..xai_animate(&server)
            },
        )
        .await,
        "image/png",
    );
    assert!(received(&server).await.is_empty());
}

// ---- GPUStack embeddings ----------------------------------------------------------------------

const GPUSTACK_MODEL: &str = "qwen3";

fn png_image() -> Attachment {
    Attachment::from_bytes(b"png bytes".to_vec(), "logo.png", None)
}

fn gpustack_image_part() -> Value {
    json!({ "type": "image_url", "image_url": { "url": "data:image/png;base64,cG5nIGJ5dGVz", "detail": "auto" } })
}

fn gpustack_embed<'a>(server: &MockServer) -> EmbedOptions<'a> {
    EmbedOptions {
        model: Some(GPUSTACK_MODEL),
        provider: Some("gpustack"),
        config: Some(config(server)),
        ..Default::default()
    }
}

fn one_vector() -> Value {
    json!({ "data": [{ "embedding": [0.1, 0.2] }], "usage": { "prompt_tokens": 12 } })
}

// spec: providers/gpustack/embeddings_spec.rb:17 embeds text and an image through the messages request format
#[tokio::test]
async fn gpustack_embeds_text_and_an_image_as_messages() {
    let server = json_server(one_vector()).await;
    embed(
        "The Ruby logo",
        EmbedOptions {
            dimensions: Some(256),
            with: vec![png_image()],
            ..gpustack_embed(&server)
        },
    )
    .await
    .unwrap();
    let (path, body) = only_request(&server).await;
    assert_eq!(path, "/v1/embeddings");
    assert_eq!(
        body,
        json!({
            "model": GPUSTACK_MODEL,
            "dimensions": 256,
            "messages": [{ "role": "user", "content": [{ "type": "text", "text": "The Ruby logo" }, gpustack_image_part()] }]
        })
    );
}

// spec: providers/gpustack/embeddings_spec.rb:25 embeds an image without text
#[tokio::test]
async fn gpustack_embeds_an_image_without_text() {
    let server = json_server(one_vector()).await;
    embed(
        None::<String>,
        EmbedOptions {
            dimensions: Some(256),
            with: vec![png_image()],
            ..gpustack_embed(&server)
        },
    )
    .await
    .unwrap();
    let (_, body) = only_request(&server).await;
    assert_eq!(
        body,
        json!({ "model": GPUSTACK_MODEL, "dimensions": 256, "messages": [{ "role": "user", "content": [gpustack_image_part()] }] })
    );
}

// spec: providers/gpustack/embeddings_spec.rb:31 preserves text batches and provider options
#[tokio::test]
async fn gpustack_preserves_text_batches_and_provider_options() {
    let server =
        json_server(json!({ "data": [{ "embedding": [0.1] }, { "embedding": [0.2] }] })).await;
    let options = EmbedOptions {
        dimensions: Some(256),
        provider_options: json!({ "dimensions": 128, "encoding_format": "float" }),
        ..gpustack_embed(&server)
    };
    embed(vec!["Ruby".to_string(), "Rails".into()], options)
        .await
        .unwrap();
    let (_, body) = only_request(&server).await;
    assert_eq!(
        body,
        json!({ "model": GPUSTACK_MODEL, "input": ["Ruby", "Rails"], "dimensions": 128, "encoding_format": "float" })
    );
}

// spec: providers/gpustack/embeddings_spec.rb:41 rejects attachments the backend cannot render
#[tokio::test]
async fn gpustack_rejects_attachments_it_cannot_render() {
    let server = json_server(one_vector()).await;
    let document = Attachment::from_bytes(b"docx bytes".to_vec(), "report.docx", None);
    let result = embed(
        "The report",
        EmbedOptions {
            dimensions: Some(256),
            with: vec![document],
            ..gpustack_embed(&server)
        },
    )
    .await;
    assert!(
        matches!(result, Err(Error::UnsupportedAttachment(_))),
        "{result:?}"
    );
    assert!(received(&server).await.is_empty());
}

// spec: providers/gpustack/embeddings_spec.rb:47 accepts image attachments through the public API and returns one vector
#[tokio::test]
async fn gpustack_embeds_an_image_through_the_public_api() {
    let server = json_server(one_vector()).await;
    let result = embed(
        None::<String>,
        EmbedOptions {
            with: vec![png_image()],
            ..gpustack_embed(&server)
        },
    )
    .await
    .unwrap();
    let (_, body) = only_request(&server).await;
    assert_eq!(
        body,
        json!({ "model": GPUSTACK_MODEL, "messages": [{ "role": "user", "content": [gpustack_image_part()] }] })
    );
    assert_eq!(result.vectors, Vectors::Single(vec![0.1, 0.2]));
    assert_eq!(result.tokens().input, Some(12));
}

// ---- OpenRouter embeddings --------------------------------------------------------------------

const OPENROUTER_EMBEDDING_MODEL: &str = "google/gemini-embedding-2";

fn openrouter_embed<'a>(server: &MockServer) -> EmbedOptions<'a> {
    EmbedOptions {
        model: Some(OPENROUTER_EMBEDDING_MODEL),
        provider: Some("openrouter"),
        config: Some(config(server)),
        ..Default::default()
    }
}

// spec: providers/openrouter/embeddings_spec.rb:24 embeds an image without text
#[tokio::test]
async fn openrouter_embeds_an_image_without_text() {
    let server = json_server(one_vector()).await;
    let with = vec![Attachment::new("https://example.com/logo.png")];
    embed(
        None::<String>,
        EmbedOptions {
            with,
            ..openrouter_embed(&server)
        },
    )
    .await
    .unwrap();
    let (_, body) = only_request(&server).await;
    assert_eq!(
        body["input"],
        json!([{ "content": [{ "type": "image_url", "image_url": { "url": "https://example.com/logo.png" } }] }])
    );
}

// spec: providers/openrouter/embeddings_spec.rb:30 leaves image detail out of embedding inputs
#[tokio::test]
async fn openrouter_leaves_image_detail_out_of_embeddings() {
    let server = json_server(one_vector()).await;
    let with =
        vec![Attachment::new("https://example.com/logo.png").with_resolution(Resolution::High)];
    embed(
        None::<String>,
        EmbedOptions {
            with,
            ..openrouter_embed(&server)
        },
    )
    .await
    .unwrap();
    let (_, body) = only_request(&server).await;
    assert_eq!(
        body["input"][0]["content"][0],
        json!({ "type": "image_url", "image_url": { "url": "https://example.com/logo.png" } })
    );
}

// spec: providers/openrouter/embeddings_spec.rb:45 preserves batched text inputs and maps the task type
#[tokio::test]
async fn openrouter_preserves_batches_and_maps_the_task_type() {
    let server =
        json_server(json!({ "data": [{ "embedding": [0.1] }, { "embedding": [0.2] }] })).await;
    embed(
        vec!["Ruby".to_string(), "Rails".into()],
        EmbedOptions {
            task_type: Some("search_document"),
            ..openrouter_embed(&server)
        },
    )
    .await
    .unwrap();
    let (_, body) = only_request(&server).await;
    assert_eq!(
        body,
        json!({ "model": OPENROUTER_EMBEDDING_MODEL, "input": ["Ruby", "Rails"], "input_type": "search_document" })
    );
}

// spec: providers/openrouter/embeddings_spec.rb:61 rejects provider-managed references without inline content
#[tokio::test]
async fn openrouter_embedding_rejects_provider_managed_files() {
    let server = json_server(one_vector()).await;
    let with = vec![Attachment::from_uploaded(uploaded(
        "file_123",
        "openrouter",
        "report.pdf",
        "application/pdf",
    ))];
    let result = embed(
        None::<String>,
        EmbedOptions {
            with,
            ..openrouter_embed(&server)
        },
    )
    .await;
    assert!(
        matches!(result, Err(Error::UnsupportedAttachment(_))),
        "{result:?}"
    );
    assert!(received(&server).await.is_empty());
}

// spec: providers/openrouter/embeddings_spec.rb:68 allows provider options to override rendered fields
#[tokio::test]
async fn openrouter_embedding_provider_options_override_rendered_fields() {
    let server = json_server(one_vector()).await;
    let options = EmbedOptions {
        task_type: Some("search_document"),
        provider_options: json!({ "input_type": "search_query", "dimensions": 256 }),
        ..openrouter_embed(&server)
    };
    embed("Ruby", options).await.unwrap();
    let (_, body) = only_request(&server).await;
    assert_eq!(
        body,
        json!({ "model": OPENROUTER_EMBEDDING_MODEL, "input": "Ruby", "input_type": "search_query", "dimensions": 256 })
    );
}

// ---- GPUStack speech and transcription --------------------------------------------------------

// spec: providers/gpustack/speech_spec.rb:8 requests the vLLM binary stream with PCM as its default format
#[tokio::test]
async fn gpustack_streams_pcm_speech_by_default() {
    let server = raw_server("application/octet-stream", b"audio bytes".to_vec()).await;
    let options = SpeakOptions {
        model: Some(GPUSTACK_MODEL),
        provider: Some("gpustack"),
        voice: Some("Vivian"),
        config: Some(config(&server)),
        ..Default::default()
    };
    let mut chunks = Vec::new();
    let speech = speak_stream("Hello", options, |chunk| chunks.push(chunk.clone()))
        .await
        .unwrap();
    let (path, body) = only_request(&server).await;
    assert_eq!(path, "/v1/audio/speech");
    assert_eq!(
        body,
        json!({ "model": GPUSTACK_MODEL, "input": "Hello", "voice": "Vivian", "stream": true, "response_format": "pcm" })
    );
    assert_eq!(speech.format, "pcm");
    assert_eq!(chunks[0].format, "pcm");
    assert_eq!(
        chunks
            .iter()
            .flat_map(|c| c.data.clone())
            .collect::<Vec<u8>>(),
        speech.data
    );
}

async fn gpustack_transcribe(
    events: &[Value],
) -> (
    MockServer,
    rust_llm::Transcription,
    Vec<rust_llm::TranscriptionChunk>,
) {
    let server = raw_server("text/event-stream", sse(events)).await;
    let options = TranscribeOptions {
        model: Some(GPUSTACK_MODEL),
        provider: Some("gpustack"),
        config: Some(config(&server)),
        ..Default::default()
    };
    let mut chunks = Vec::new();
    let result = transcribe_stream(Attachment::new(fixture("ruby.wav")), options, |c| {
        chunks.push(c.clone())
    })
    .await
    .unwrap();
    // `{ model:, stream: 'true', stream_include_usage: 'true' }` on audio/transcriptions.
    let requests = received(&server).await;
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].url.path(), "/v1/audio/transcriptions");
    assert_eq!(
        form_field(&requests[0].body, "model").as_deref(),
        Some(GPUSTACK_MODEL)
    );
    assert_eq!(
        form_field(&requests[0].body, "stream").as_deref(),
        Some("true")
    );
    assert_eq!(
        form_field(&requests[0].body, "stream_include_usage").as_deref(),
        Some("true")
    );
    (server, result, chunks)
}

// spec: providers/gpustack/transcription_spec.rb:20 streams vLLM text and retains a separate final usage event
#[tokio::test]
async fn gpustack_streams_text_and_keeps_the_final_usage_event() {
    let events = [
        json!({ "choices": [{ "delta": { "role": "assistant", "content": "" }, "finish_reason": null }] }),
        json!({ "choices": [{ "delta": { "content": "Hello, " }, "finish_reason": null }] }),
        json!({ "choices": [{ "delta": { "content": "Ruby." }, "finish_reason": "stop" }] }),
        json!({ "choices": [], "usage": { "prompt_tokens": 20, "completion_tokens": 4 } }),
    ];
    let (_, result, chunks) = gpustack_transcribe(&events).await;
    assert_eq!(
        chunks
            .iter()
            .filter_map(|c| c.delta.as_deref())
            .collect::<String>(),
        "Hello, Ruby."
    );
    let last = chunks.last().unwrap();
    assert!(last.is_done());
    assert_eq!(last.raw, events[3]);
    assert_eq!(result.text.as_deref(), Some("Hello, Ruby."));
    assert_eq!(result.tokens().input, Some(20));
    assert_eq!(result.tokens().output, Some(4));
}

// spec: providers/gpustack/transcription_spec.rb:38 returns the transcript when vLLM does not report usage
#[tokio::test]
async fn gpustack_returns_the_transcript_without_usage() {
    let events = [
        json!({ "choices": [{ "delta": { "content": "Good morning." }, "finish_reason": null }] }),
        json!({ "choices": [{ "delta": {}, "finish_reason": "stop" }] }),
    ];
    let (_, result, chunks) = gpustack_transcribe(&events).await;
    assert_eq!(result.text.as_deref(), Some("Good morning."));
    assert!(chunks.last().unwrap().is_done());
    assert_eq!(result.tokens().input, None);
}

// spec: providers/gpustack/transcription_spec.rb:51 preserves typed transcript and speaker events from compatible backends
#[tokio::test]
async fn gpustack_preserves_typed_transcript_and_speaker_events() {
    let segment = json!({ "type": "transcript.text.segment", "text": "Hello.", "speaker": "S01", "start": 0, "end": 1 });
    let events = [
        segment.clone(),
        json!({ "type": "transcript.text.done", "text": "Hello.", "usage": { "input_tokens": 10, "output_tokens": 2 } }),
    ];
    let (_, result, chunks) = gpustack_transcribe(&events).await;
    assert!(chunks[0].is_segment());
    assert_eq!(chunks[0].raw, segment);
    assert_eq!(result.text.as_deref(), Some("Hello."));
    assert_eq!(
        result.segments,
        Some(vec![
            json!({ "text": "Hello.", "speaker": "S01", "start": 0, "end": 1 })
        ])
    );
    assert_eq!(result.tokens().input, Some(10));
}

// ---- Mistral speech, transcription, and OCR ---------------------------------------------------

fn mistral_speak<'a>(server: &MockServer) -> SpeakOptions<'a> {
    SpeakOptions {
        model: Some("voxtral-mini-tts-latest"),
        provider: Some("mistral"),
        config: Some(config(server)),
        ..Default::default()
    }
}

fn speech_events(events: &[Value]) -> String {
    events
        .iter()
        .map(|e| format!("event: {}\ndata: {e}\n\n", e["type"].as_str().unwrap_or("")))
        .collect()
}

// spec: providers/mistral/speech_spec.rb:35 #stream_speech > rejects an audio stream that ends before completion
#[tokio::test]
async fn mistral_speech_stream_ending_before_completion_is_an_error() {
    let delta = json!({ "type": "speech.audio.delta", "audio_data": base64::engine::general_purpose::STANDARD.encode(b"\xFFaudio") });
    let server = raw_server("text/event-stream", speech_events(&[delta])).await;
    match speak_stream("Hello", mistral_speak(&server), |_| {}).await {
        Err(e) => assert!(e.to_string().contains("before its completion event"), "{e}"),
        Ok(_) => panic!("a stream without its completion event must fail"),
    }
}

// spec: providers/mistral/speech_spec.rb:43 #stream_speech > raises streaming provider errors without delivering them as speech
#[tokio::test]
async fn mistral_speech_stream_errors_are_raised_not_spoken() {
    let server = raw_server(
        "text/event-stream",
        "event: error\ndata: {\"error\":{\"message\":\"Generation failed\"}}\n\n".to_string(),
    )
    .await;
    let mut chunks = 0;
    match speak_stream("Hello", mistral_speak(&server), |_| chunks += 1).await {
        Err(e) => assert!(e.to_string().contains("Generation failed"), "{e}"),
        Ok(_) => panic!("a streamed error must fail"),
    }
    assert_eq!(chunks, 0);
}

async fn mistral_transcribe(
    events: &[Value],
) -> (rust_llm::Transcription, Vec<rust_llm::TranscriptionChunk>) {
    let server = raw_server("text/event-stream", sse(events)).await;
    let options = TranscribeOptions {
        model: Some("voxtral-mini-latest"),
        provider: Some("mistral"),
        config: Some(config(&server)),
        ..Default::default()
    };
    let mut chunks = Vec::new();
    let result = transcribe_stream(Attachment::new(fixture("ruby.wav")), options, |c| {
        chunks.push(c.clone())
    })
    .await
    .unwrap();
    // `{ model:, stream: 'true' }` on audio/transcriptions.
    let requests = received(&server).await;
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].url.path(), "/v1/audio/transcriptions");
    assert_eq!(
        form_field(&requests[0].body, "model").as_deref(),
        Some("voxtral-mini-latest")
    );
    assert_eq!(
        form_field(&requests[0].body, "stream").as_deref(),
        Some("true")
    );
    assert_eq!(form_field(&requests[0].body, "stream_include_usage"), None);
    (result, chunks)
}

// spec: providers/mistral/transcription_spec.rb:19 streams Mistral text and preserves the final language, audio duration, and usage
#[tokio::test]
async fn mistral_streams_text_and_keeps_language_duration_and_usage() {
    let events = [
        json!({ "type": "transcription.language", "audio_language": "en" }),
        json!({ "type": "transcription.text.delta", "text": "Hello, " }),
        json!({ "type": "transcription.text.delta", "text": "Ruby." }),
        json!({ "type": "transcription.done", "text": "Hello, Ruby.", "language": "en",
                "usage": { "prompt_tokens": 8, "completion_tokens": 4, "prompt_audio_seconds": 3 } }),
    ];
    let (result, chunks) = mistral_transcribe(&events).await;
    assert_eq!(
        chunks
            .iter()
            .filter_map(|c| c.delta.as_deref())
            .collect::<String>(),
        "Hello, Ruby."
    );
    let last = chunks.last().unwrap();
    assert!(last.is_done());
    assert_eq!(last.raw, events[3]);
    assert_eq!(result.text.as_deref(), Some("Hello, Ruby."));
    assert_eq!(result.language.as_deref(), Some("en"));
    assert_eq!(result.duration, Some(3.0));
    assert_eq!(result.tokens().input, Some(8));
    assert_eq!(result.tokens().output, Some(4));
}

// spec: providers/mistral/transcription_spec.rb:50 preserves segments reported only by the final event
#[tokio::test]
async fn mistral_keeps_segments_reported_only_by_the_final_event() {
    let segment = json!({ "text": "Hello.", "start": 0.0, "end": 1.0 });
    let events = [
        json!({ "type": "transcription.done", "text": "Hello.", "segments": [segment], "usage": {} }),
    ];
    let (result, _) = mistral_transcribe(&events).await;
    assert_eq!(result.segments, Some(vec![segment]));
}

fn mistral_ocr<'a>(server: &MockServer) -> OcrOptions<'a> {
    OcrOptions {
        model: Some("mistral-ocr-latest"),
        provider: Some("mistral"),
        config: Some(config(server)),
        ..Default::default()
    }
}

fn ocr_body() -> Value {
    json!({ "pages": [{ "index": 0, "markdown": "# Hello" }], "model": "mistral-ocr-latest" })
}

// spec: providers/mistral/ocr_spec.rb:7 .render_ocr_payload > sends remote documents as document_url references
#[tokio::test]
async fn mistral_ocr_sends_remote_documents_as_document_urls() {
    let server = json_server(ocr_body()).await;
    ocr("https://example.com/report.pdf", mistral_ocr(&server))
        .await
        .unwrap();
    let (path, body) = only_request(&server).await;
    assert_eq!(path, "/v1/ocr");
    assert_eq!(
        body,
        json!({ "model": "mistral-ocr-latest", "document": { "type": "document_url", "document_url": "https://example.com/report.pdf" } })
    );
}

// spec: providers/mistral/ocr_spec.rb:25 .render_ocr_payload > sends images through the image_url variant
#[tokio::test]
async fn mistral_ocr_sends_images_as_image_urls() {
    let server = json_server(ocr_body()).await;
    ocr(Attachment::new(fixture("ruby.png")), mistral_ocr(&server))
        .await
        .unwrap();
    let (_, body) = only_request(&server).await;
    assert_eq!(body["document"]["type"], "image_url");
    assert!(
        body["document"]["image_url"]
            .as_str()
            .unwrap()
            .starts_with("data:image/png;base64,")
    );
}

// spec: providers/mistral/ocr_spec.rb:43 .render_ocr_payload > inlines XML documents as base64 data URIs
#[tokio::test]
async fn mistral_ocr_inlines_xml_as_a_data_uri() {
    let server = json_server(ocr_body()).await;
    ocr(Attachment::new(fixture("ruby.xml")), mistral_ocr(&server))
        .await
        .unwrap();
    let (_, body) = only_request(&server).await;
    assert_eq!(body["document"]["type"], "document_url");
    assert!(
        body["document"]["document_url"]
            .as_str()
            .unwrap()
            .starts_with("data:application/xml;base64,")
    );
}

// spec: providers/mistral/ocr_spec.rb:67 .parse_ocr_response > builds an OCR result from pages, model, and usage_info
#[tokio::test]
async fn mistral_ocr_builds_a_result_from_pages_model_and_usage() {
    let body = json!({
        "pages": [
            { "index": 0, "markdown": "# Hello", "images": [], "tables": [] },
            { "index": 1, "markdown": "World", "images": [], "tables": [] }
        ],
        "model": "mistral-ocr-latest",
        "usage_info": { "pages_processed": 2, "doc_size_bytes": 123 }
    });
    let server = json_server(body.clone()).await;
    let result = ocr("https://example.com/report.pdf", mistral_ocr(&server))
        .await
        .unwrap();
    assert_eq!(
        result.pages.iter().map(|p| p.index).collect::<Vec<_>>(),
        vec![0, 1]
    );
    assert_eq!(result.pages[0].markdown.as_deref(), Some("# Hello"));
    assert_eq!(result.markdown(), "# Hello\n\nWorld");
    assert_eq!(result.model, "mistral-ocr-latest");
    assert_eq!(
        result.usage,
        Some(json!({ "pages_processed": 2, "doc_size_bytes": 123 }))
    );
    assert_eq!(result.raw, body);
}

// ---- xAI speech -------------------------------------------------------------------------------

// spec: providers/xai/speech_spec.rb:6 rejects timestamp JSON responses before starting an audio stream
#[tokio::test]
async fn xai_streaming_speech_rejects_with_timestamps() {
    let server = raw_server("audio/mpeg", b"audio".to_vec()).await;
    let options = SpeakOptions {
        model: Some("grok-tts"),
        provider: Some("xai"),
        voice: Some("eve"),
        format: Some("mp3"),
        provider_options: json!({ "with_timestamps": true }),
        config: Some(config(&server)),
        ..Default::default()
    };
    match speak_stream("Hello", options, |chunk| panic!("Unexpected: {chunk:?}")).await {
        Err(Error::Argument(message)) => assert!(
            message.contains("does not accept with_timestamps"),
            "{message}"
        ),
        other => panic!("expected ArgumentError, got {other:?}"),
    }
    assert!(received(&server).await.is_empty());
}
