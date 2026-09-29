//! `RubyLLM.animate` / `animate_later` / `VideoJob`, replayed from RubyLLM's `video_*` and
//! `providers_xai_videos_*` cassettes. Assertions follow `spec/ruby_llm/video_spec.rb` and
//! `spec/ruby_llm/providers/xai/videos_spec.rb`. The poll interval is zero, as the Ruby spec sets
//! it on replay, so no test sleeps.

mod support;

use std::sync::Arc;
use std::time::Duration;

use rust_llm::{AnimateOptions, Config, Error, Video, VideoStatus, animate, animate_later};
use serde_json::json;
use support::{Cassette, config_for};

async fn start(name: &str) -> Cassette {
    Cassette::start(name)
        .await
        .unwrap_or_else(|| panic!("missing cassette {name}; run bin/convert-cassettes 'video_*'"))
}

/// The replay server's config with polling that never sleeps.
fn no_wait(cassette: &Cassette, provider: &str) -> Arc<Config> {
    let mut config = (*config_for(cassette, provider)).clone();
    config.video_generation_poll_interval = Duration::ZERO;
    Arc::new(config)
}

/// Hosted videos download from the provider's CDN; send that request to the replay server.
fn hosted_on(video: &mut Video, cassette: &Cassette) {
    let url = video.url.clone().expect("hosted video url");
    let path = url
        .split_once("://")
        .and_then(|(_, rest)| rest.split_once('/'))
        .map(|(_, p)| p)
        .unwrap_or("");
    video.url = Some(format!("{}/{path}", cassette.server.uri()));
}

/// `save_and_verify_video`: `save` returns the path it was given and writes more than 10KB.
async fn saves_a_real_clip(video: &Video) {
    let path = std::env::temp_dir().join(format!("rust_llm_video_{}.mp4", uuid::Uuid::new_v4()));
    let saved = video.save(path.clone()).await.expect("save");
    assert_eq!(saved, path);
    let size = std::fs::metadata(&path).expect("saved file").len();
    std::fs::remove_file(&path).ok();
    assert!(
        size > 10_000,
        "a real clip is larger than 10KB, saved {size} bytes"
    );
}

/// "gemini/veo-3.1-lite-generate-preview can animate videos": predictLongRunning, six pending
/// polls, then an authenticated download of the Files API URI.
#[tokio::test]
async fn gemini_can_animate_videos() {
    let cassette =
        start("video_basic_functionality_gemini_veo-3_1-lite-generate-preview_can_animate_videos")
            .await;
    let base = cassette.server.uri();
    let options = AnimateOptions {
        model: Some("veo-3.1-lite-generate-preview"),
        provider: Some("gemini"),
        provider_options: json!({ "parameters": { "durationSeconds": 4 } }),
        config: Some(no_wait(&cassette, "gemini")),
        ..Default::default()
    };
    // The download URI is absolute; point it at the replay server by rewriting the job's response.
    let mut job = animate_later(Some("a calm ocean wave at sunset"), options)
        .await
        .unwrap();
    assert_eq!(
        job.id,
        "models/veo-3.1-lite-generate-preview/operations/q7m9xli7kryn"
    );
    assert!(job.is_pending());
    job.wait(None, None).await.unwrap();
    assert!(job.is_completed());
    let uri = job
        .raw
        .pointer("/response/generateVideoResponse/generatedSamples/0/video/uri")
        .and_then(|u| u.as_str())
        .unwrap()
        .to_string();
    let path = uri
        .split_once("://")
        .and_then(|(_, rest)| rest.split_once('/'))
        .map(|(_, p)| p)
        .unwrap();
    job.raw["response"]["generateVideoResponse"]["generatedSamples"][0]["video"]["uri"] =
        json!(format!("{base}/{path}"));
    let video = job
        .video()
        .await
        .unwrap()
        .expect("a completed job has a video");

    assert!(video.mime_type.as_deref().unwrap().contains("video"));
    assert!(video.data.is_some());
    assert_eq!(
        video.model.as_deref(),
        Some("veo-3.1-lite-generate-preview")
    );
    saves_a_real_clip(&video).await;
    cassette.assert_all_matched().await;
}

/// "xai/grok-imagine-video can animate videos" through the blocking `animate`.
#[tokio::test]
async fn xai_can_animate_videos() {
    let cassette =
        start("video_basic_functionality_xai_grok-imagine-video_can_animate_videos").await;
    let options = AnimateOptions {
        model: Some("grok-imagine-video"),
        provider: Some("xai"),
        provider_options: json!({ "duration": 1, "resolution": "480p" }),
        config: Some(no_wait(&cassette, "xai")),
        ..Default::default()
    };
    let mut video = animate(Some("a calm ocean wave at sunset"), options)
        .await
        .unwrap();

    assert_eq!(video.mime_type.as_deref(), Some("video/mp4"));
    assert!(video.url.is_some());
    assert_eq!(video.duration, Some(1.0));
    assert_eq!(video.model.as_deref(), Some("grok-imagine-video"));
    assert_eq!(video.raw["status"], "done");
    hosted_on(&mut video, &cassette);
    saves_a_real_clip(&video).await;
    cassette.assert_all_matched().await;
}

/// xai/videos_spec.rb "completes a video edit through the public API": a video URL in `with:`
/// takes the `videos/edits` route.
#[tokio::test]
async fn xai_completes_a_video_edit() {
    let cassette =
        start("providers_xai_videos_completes_a_video_edit_through_the_public_api").await;
    let context = rust_llm::Context::new((*no_wait(&cassette, "xai")).clone());
    let options = AnimateOptions {
        model: Some("grok-imagine-video"),
        provider: Some("xai"),
        with: vec![rust_llm::Attachment::new(
            "https://data.x.ai/docs/video-generation/portrait-wave.mp4",
        )],
        ..Default::default()
    };
    let mut job = context
        .animate_later(Some("Make the background blue"), options)
        .await
        .unwrap();
    assert!(!job.id.is_empty());
    assert!(
        job.wait(Some(Duration::from_secs(180)), None)
            .await
            .unwrap()
            .is_completed()
    );
    let mut video = job.video().await.unwrap().unwrap();
    assert_eq!(video.mime_type.as_deref(), Some("video/mp4"));
    assert_eq!(video.raw["status"], "done");
    hosted_on(&mut video, &cassette);
    assert!(video.to_blob().await.unwrap().len() > 1000);
    cassette.assert_all_matched().await;
}

/// xai/videos_spec.rb "completes a video extension through the public API".
#[tokio::test]
async fn xai_completes_a_video_extension() {
    let cassette =
        start("providers_xai_videos_completes_a_video_extension_through_the_public_api").await;
    let context = rust_llm::Context::new((*no_wait(&cassette, "xai")).clone());
    let options = AnimateOptions {
        model: Some("grok-imagine-video"),
        provider: Some("xai"),
        extend: Some("https://data.x.ai/docs/video-generation/portrait-wave.mp4".into()),
        provider_options: json!({ "duration": 2 }),
        ..Default::default()
    };
    let mut job = context
        .animate_later(Some("Continue the gentle waving motion"), options)
        .await
        .unwrap();
    assert!(
        job.wait(Some(Duration::from_secs(180)), None)
            .await
            .unwrap()
            .is_completed()
    );
    let mut video = job.video().await.unwrap().unwrap();
    assert_eq!(video.raw["status"], "done");
    hosted_on(&mut video, &cassette);
    assert!(video.to_blob().await.unwrap().len() > 1000);
    cassette.assert_all_matched().await;
}

/// "validates model existence".
#[tokio::test]
async fn validates_model_existence() {
    let err = animate(
        Some("a cat"),
        AnimateOptions {
            model: Some("invalid-model"),
            ..Default::default()
        },
    )
    .await
    .unwrap_err();
    assert!(matches!(err, Error::ModelNotFound(_)), "{err}");
}

/// "raises a clear error for providers without video generation".
#[tokio::test]
async fn providers_without_video_generation_fail_clearly() {
    let mut config = Config::default();
    config.set("anthropic_api_key", "test");
    let options = AnimateOptions {
        model: Some("claude-haiku-4-5"),
        config: Some(config.into()),
        ..Default::default()
    };
    let err = animate_later(Some("a cat"), options).await.unwrap_err();
    assert!(
        err.to_string()
            .contains("Anthropic doesn't support video generation"),
        "{err}"
    );
}

/// xai/videos_spec.rb "rejects conflicting sources ... before sending requests".
#[tokio::test]
async fn with_and_extend_cannot_be_combined() {
    let mut config = Config::default();
    config.set("xai_api_key", "test");
    let options = AnimateOptions {
        model: Some("grok-imagine-video"),
        provider: Some("xai"),
        with: vec![rust_llm::Attachment::new("https://example.com/a.png")],
        extend: Some("https://example.com/clip.mp4".into()),
        config: Some(config.into()),
        ..Default::default()
    };
    let err = animate_later(Some("Continue"), options).await.unwrap_err();
    assert!(err.to_string().contains("cannot be combined"), "{err}");
}

/// Jobs a replay server keeps pending: `wait` gives up at the deadline instead of sleeping past it
/// ("raises when the job outlives the timeout", "does not sleep past the timeout deadline"), and
/// a failed job surfaces the provider's message from `wait` and `video`.
#[tokio::test]
async fn wait_honors_the_deadline_and_surfaces_failures() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/videos/generations"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "request_id": "slow" })))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/v1/videos/slow"))
        .respond_with(ResponseTemplate::new(202).set_body_json(json!({ "status": "pending" })))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/v1/videos/bad"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({ "status": "failed", "error": "flagged by moderation" })),
        )
        .mount(&server)
        .await;

    let mut config = Config::default();
    config.set("xai_api_key", "test");
    config.set("xai_api_base", format!("{}/v1", server.uri()));
    config.max_retries = 0;
    let config = Arc::new(config);
    let options = || AnimateOptions {
        model: Some("grok-imagine-video"),
        provider: Some("xai"),
        config: Some(config.clone()),
        ..Default::default()
    };

    let mut job = animate_later(Some("a cat"), options()).await.unwrap();
    assert_eq!(
        job.video().await.unwrap().map(|_| ()),
        None,
        "no video while pending"
    );
    let err = job
        .wait(Some(Duration::ZERO), Some(Duration::ZERO))
        .await
        .unwrap_err();
    assert_eq!(
        err.to_string(),
        "Video generation timed out after 0 seconds"
    );

    let started = std::time::Instant::now();
    let err = job
        .wait(
            Some(Duration::from_millis(200)),
            Some(Duration::from_secs(60)),
        )
        .await
        .unwrap_err();
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "slept past the deadline: {:?}",
        started.elapsed()
    );
    assert!(
        err.to_string().contains("timed out after 0.2 seconds"),
        "{err}"
    );

    job.id = "bad".into();
    let err = job
        .wait(Some(Duration::from_secs(10)), Some(Duration::ZERO))
        .await
        .unwrap_err();
    assert!(err.to_string().contains("flagged by moderation"), "{err}");
    assert_eq!(job.status, VideoStatus::Failed);
    assert!(
        job.video()
            .await
            .unwrap_err()
            .to_string()
            .contains("flagged by moderation")
    );
    // Done jobs stop refreshing.
    job.refresh().await.unwrap();
    let polls = server
        .received_requests()
        .await
        .unwrap()
        .iter()
        .filter(|r| r.url.path() == "/v1/videos/bad")
        .count();
    assert_eq!(polls, 1);
}
