//! RubyLLM 2.0's `spec/ruby_llm/protocols/gpustack/videos_spec.rb`: vLLM-Omni video jobs through
//! GPUStack's model proxy (`protocols/gpustack/videos.rb`). Ruby's WebMock stubs are a wiremock
//! server here; the multipart fields RubyLLM renders are read back from the recorded request.

use std::time::Duration;

use rust_llm::{AnimateOptions, Attachment, Config, Context, Error, UploadedFile};
use serde_json::{Value, json};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

/// `model_for(:gpustack)`.
const MODEL: &str = "qwen3";
const BASE: &str = "/cluster/model/proxy/42/v1/videos";

fn context(server: &MockServer) -> Context {
    let mut config = Config::default();
    config.set(
        "gpustack_api_base",
        format!("{}/cluster/model/proxy/42/v1/", server.uri()),
    );
    config.set("gpustack_api_key", "isolated-key");
    config.video_generation_poll_interval = Duration::ZERO;
    Context::new(config)
}

fn options(with: Vec<Attachment>) -> AnimateOptions<'static> {
    AnimateOptions {
        model: Some(MODEL),
        provider: Some("gpustack"),
        with,
        ..Default::default()
    }
}

fn state(status: &str) -> Value {
    json!({ "id": "video_gen_1", "model": MODEL, "status": status, "seconds": "4", "media_type": "video/mp4" })
}

fn queued() -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(state("queued"))
}

/// The text of multipart field `name` in the request body.
fn field(request: &Request, name: &str) -> Option<String> {
    let body = String::from_utf8_lossy(&request.body);
    let start = body.find(&format!("name=\"{name}\""))?;
    let rest = &body[start..];
    let value = &rest[rest.find("\r\n\r\n")? + 4..];
    Some(value[..value.find("\r\n--")?].to_string())
}

fn json_field(request: &Request, name: &str) -> Value {
    serde_json::from_str(&field(request, name).unwrap_or_else(|| panic!("no {name} field")))
        .unwrap()
}

async fn posts(server: &MockServer) -> Vec<Request> {
    server
        .received_requests()
        .await
        .unwrap()
        .into_iter()
        .filter(|r| r.method.as_str() == "POST")
        .collect()
}

// spec: protocols/gpustack/videos_spec.rb:26 submits text-only multipart once, polls all states, and downloads bytes with proxy authentication
#[tokio::test]
async fn submits_text_only_multipart_once_polls_all_states_and_downloads_with_proxy_auth() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(BASE))
        .and(|req: &Request| {
            let body = String::from_utf8_lossy(&req.body);
            req.headers
                .get("content-type")
                .and_then(|v| v.to_str().ok())
                .is_some_and(|v| v.starts_with("multipart/form-data; boundary="))
                && req
                    .headers
                    .get("authorization")
                    .is_some_and(|v| v == "Bearer isolated-key")
                && body.contains("A rainy street")
                && body.contains("name=\"model\"")
                && body.contains(MODEL)
        })
        .respond_with(queued())
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path(format!("{BASE}/video_gen_1")))
        .respond_with(ResponseTemplate::new(200).set_body_json(state("in_progress")))
        .up_to_n_times(1)
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path(format!("{BASE}/video_gen_1")))
        .respond_with(ResponseTemplate::new(200).set_body_json(state("completed")))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path(format!("{BASE}/video_gen_1/content")))
        .and(header("Authorization", "Bearer isolated-key"))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(b"mp4\x00bytes".to_vec(), "video/mp4"),
        )
        .expect(1)
        .mount(&server)
        .await;

    let result = context(&server)
        .animate(Some("A rainy street"), options(vec![]))
        .await
        .unwrap();

    assert_eq!(result.data.as_deref(), Some(&b"mp4\x00bytes"[..]));
    assert_eq!(result.mime_type.as_deref(), Some("video/mp4"));
    assert_eq!(result.duration, None);
    assert_eq!(result.model.as_deref(), Some(MODEL));
    assert_eq!(result.raw["status"], "completed");
    assert_eq!(result.to_blob().await.unwrap(), b"mp4\x00bytes");
    // `expect` counts are verified when the server drops: one submit, two polls, one download.
    server.verify().await;
}

// spec: protocols/gpustack/videos_spec.rb:48 serializes image and audio input and nested backend options as JSON multipart fields
#[tokio::test]
async fn serializes_image_and_audio_input_and_nested_options_as_json_fields() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(BASE))
        .respond_with(queued())
        .mount(&server)
        .await;
    let image = Attachment::from_bytes(b"image".to_vec(), "scene.png", None);
    let audio = Attachment::from_bytes(b"audio".to_vec(), "voice.wav", None);
    let options = AnimateOptions {
        provider_options: json!({ "extra_params": { "foo": 1 } }),
        ..options(vec![image, audio])
    };
    context(&server)
        .animate_later(Some("A person singing"), options)
        .await
        .unwrap();

    let request = &posts(&server).await[0];
    assert_eq!(
        json_field(request, "image_reference"),
        json!({ "image_url": "data:image/png;base64,aW1hZ2U=" })
    );
    assert_eq!(
        json_field(request, "audio_reference"),
        json!({ "audio_url": "data:audio/wav;base64,YXVkaW8=" })
    );
    assert_eq!(
        field(request, "extra_params").as_deref(),
        Some(r#"{"foo":1}"#)
    );
}

// spec: protocols/gpustack/videos_spec.rb:57 preserves ordered multiple references and sends remote references inline
#[tokio::test]
async fn preserves_ordered_multiple_references_and_sends_remote_references_inline() {
    let server = MockServer::start().await;
    for (file, body, content_type) in [
        ("first.png", "first image", "image/png"),
        ("last.png", "last image", "image/png"),
        ("clip.mp4", "clip video", "video/mp4"),
    ] {
        Mock::given(method("GET"))
            .and(path(format!("/media/{file}")))
            .respond_with(ResponseTemplate::new(200).set_body_raw(body, content_type))
            .mount(&server)
            .await;
    }
    Mock::given(method("POST"))
        .and(path(BASE))
        .respond_with(queued())
        .mount(&server)
        .await;
    let media = |file: &str| Attachment::new(format!("{}/media/{file}", server.uri()));
    let with = vec![media("first.png"), media("last.png"), media("clip.mp4")];
    context(&server)
        .animate_later(Some("Continue the scene"), options(with))
        .await
        .unwrap();

    let b64 = |s: &str| base64::Engine::encode(&base64::engine::general_purpose::STANDARD, s);
    let request = &posts(&server).await[0];
    assert_eq!(
        json_field(request, "image_reference"),
        json!([
            { "image_url": format!("data:image/png;base64,{}", b64("first image")) },
            { "image_url": format!("data:image/png;base64,{}", b64("last image")) }
        ])
    );
    assert_eq!(
        json_field(request, "video_reference"),
        json!({ "video_url": format!("data:video/mp4;base64,{}", b64("clip video")) })
    );
}

// spec: protocols/gpustack/videos_spec.rb:78 accepts a local video through the public attachment API and sends a JSON reference field
#[tokio::test]
async fn accepts_a_local_video_and_sends_a_json_reference_field() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(BASE))
        .and(|req: &Request| {
            let body = String::from_utf8_lossy(&req.body);
            body.contains("name=\"video_reference\"")
                && body.contains(r#"{"video_url":"data:video/mp4;base64,dmlkZW8="}"#)
        })
        .respond_with(queued())
        .expect(1)
        .mount(&server)
        .await;
    let video = Attachment::from_bytes(b"video".to_vec(), "scene.mp4", None);

    let job = context(&server)
        .animate_later(Some("Change the light to sunset"), options(vec![video]))
        .await
        .unwrap();

    assert!(job.is_pending());
    server.verify().await;
}

// spec: protocols/gpustack/videos_spec.rb:91 preserves provider failure and rejects unknown job states rather than polling forever
#[tokio::test]
async fn preserves_provider_failure_and_rejects_unknown_job_states() {
    let server = MockServer::start().await;
    let mut failed = state("failed");
    failed["error"] = json!({ "code": "render_error", "message": "Out of memory" });
    Mock::given(method("POST"))
        .and(path(BASE))
        .respond_with(ResponseTemplate::new(200).set_body_json(failed))
        .mount(&server)
        .await;
    let context = context(&server);

    let mut job = context
        .animate_later(Some("A street"), options(vec![]))
        .await
        .unwrap();
    assert!(job.is_failed());
    assert_eq!(job.error.as_deref(), Some("Out of memory"));
    let err = job.video().await.unwrap_err();
    assert!(
        matches!(err, Error::Api(..)) && err.to_string().contains("Out of memory"),
        "{err:?}"
    );

    server.reset().await;
    Mock::given(method("POST"))
        .and(path(BASE))
        .respond_with(ResponseTemplate::new(200).set_body_json(state("unrecognized")))
        .mount(&server)
        .await;
    let err = context
        .animate_later(Some("A street"), options(vec![]))
        .await
        .unwrap_err();
    assert!(
        matches!(err, Error::Api(..))
            && err
                .to_string()
                .contains("Unknown GPUStack video status: \"unrecognized\""),
        "{err:?}"
    );
}

// spec: protocols/gpustack/videos_spec.rb:104 does not repeat a video submission after an uncertain transport failure
#[tokio::test]
async fn does_not_repeat_a_video_submission_after_an_uncertain_transport_failure() {
    let server = MockServer::start().await;
    // Faraday::TimeoutError: the response never arrives within the request timeout.
    Mock::given(method("POST"))
        .and(path(BASE))
        .respond_with(queued().set_delay(Duration::from_secs(5)))
        .mount(&server)
        .await;
    let mut config = (**context(&server).config()).clone();
    config.max_retries = 3;
    config.request_timeout = Duration::from_millis(200);
    let context = Context::new(config);

    let err = context
        .animate_later(Some("A street"), options(vec![]))
        .await
        .unwrap_err();

    assert!(matches!(err, Error::Timeout(_)), "{err:?}");
    assert_eq!(posts(&server).await.len(), 1);
}

// spec: protocols/gpustack/videos_spec.rb:112 rejects unsupported gateway routes, references, multi-output requests, and extension before HTTP
#[tokio::test]
async fn rejects_unsupported_inputs_before_http() {
    let server = MockServer::start().await;
    let context = context(&server);
    let file = UploadedFile {
        id: "file_1".into(),
        provider: "gpustack".into(),
        filename: Some("scene.png".into()),
        byte_size: None,
        created_at: None,
        expires_at: None,
        status: None,
        mime_type: Some("image/png".into()),
        purpose: None,
        uri: None,
        downloadable: None,
        metadata: Value::Null,
    };
    let err = context
        .animate_later(Some("A street"), options(vec![file.into()]))
        .await
        .unwrap_err();
    assert!(
        matches!(err, Error::Argument(ref m) if m.contains("uploaded file ids")),
        "{err:?}"
    );

    let multi = AnimateOptions {
        provider_options: json!({ "num_outputs_per_prompt": 2 }),
        ..options(vec![])
    };
    let err = context
        .animate_later(Some("A street"), multi)
        .await
        .unwrap_err();
    assert!(
        matches!(err, Error::Argument(ref m) if m.contains("one video")),
        "{err:?}"
    );

    let err = context
        .animate_later(None, options(vec![]))
        .await
        .unwrap_err();
    assert!(
        matches!(err, Error::Argument(ref m) if m.contains("requires a prompt")),
        "{err:?}"
    );

    let extend = AnimateOptions {
        extend: Some("https://media.test/clip.mp4".into()),
        ..options(vec![])
    };
    let err = context
        .animate_later(Some("Continue"), extend)
        .await
        .unwrap_err();
    assert!(
        matches!(err, Error::Api(ref m, _) if m.contains("doesn't support video extension")),
        "{err:?}"
    );

    let mut gateway = (**context.config()).clone();
    gateway.set("gpustack_api_base", format!("{}/v1", server.uri()));
    let err = Context::new(gateway)
        .animate_later(Some("A street"), options(vec![]))
        .await
        .unwrap_err();
    assert!(
        matches!(err, Error::Api(ref m, _) if m.contains("/model/proxy/ROUTE_ID/v1")),
        "{err:?}"
    );

    assert_eq!(server.received_requests().await.unwrap().len(), 0);
}
