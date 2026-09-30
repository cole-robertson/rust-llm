# Video

Generate, animate, edit, and extend videos with `rust_llm::animate`. This follows RubyLLM's
`video-generation.md`.

## Generating a Video

```ruby
video = RubyLLM.animate "A paper boat sailing down a rainy gutter"
video.save "boat.mp4"
```

```rust,no_run
# async fn run() -> rust_llm::Result<()> {
let video = rust_llm::animate(Some("A paper boat sailing down a rainy gutter"), Default::default()).await?;
video.save("boat.mp4").await?;
# Ok(()) }
```

`animate` submits a job, polls until it finishes, and returns a `Video`. The prompt is an `Option`
because some models animate references without text.

## Generating Without Waiting

```ruby
job = RubyLLM.animate_later("A hummingbird hovering in slow motion")
job.refresh
job.video.save("hummingbird.mp4") if job.completed?
```

```rust,no_run
# async fn run() -> rust_llm::Result<()> {
let mut job = rust_llm::animate_later(Some("A hummingbird hovering in slow motion"), Default::default()).await?;
let id = job.id.clone(); // e.g. to check from a scheduled job

job.refresh().await?;
if job.is_completed() {
    if let Some(video) = job.video().await? {
        video.save("hummingbird.mp4").await?;
    }
}
# Ok(()) }
```

`status` is `VideoStatus::Pending`, `Completed`, or `Failed`, with `is_pending()`, `is_done()`,
`is_completed()`, and `is_failed()`. `video()` is `None` while pending and an error when rendering
failed. `job.wait(timeout, interval)` polls for you.

## Images, Edits, and Extensions

```ruby
RubyLLM.animate("Make the waterfall crash down", model: "grok-imagine-video", with: "waterfall.png",
                provider_options: { duration: 5 })
RubyLLM.animate("Change the background to blue", model: "grok-imagine-video", with: "scene.mp4")
RubyLLM.animate("The boat passes under a bridge", model: "grok-imagine-video", extend: video)
```

```rust,no_run
use rust_llm::{AnimateOptions, Attachment};
use serde_json::json;

# async fn run() -> rust_llm::Result<()> {
let from_image = AnimateOptions {
    model: Some("grok-imagine-video"),
    with: vec![Attachment::new("waterfall.png")],
    provider_options: json!({ "duration": 5 }),
    ..Default::default()
};
let video = rust_llm::animate(Some("Make the waterfall crash down"), from_image).await?;

let edit = AnimateOptions { model: Some("grok-imagine-video"), with: vec![Attachment::new("scene.mp4")], ..Default::default() };
rust_llm::animate(Some("Change the background to blue"), edit).await?;

let extension = AnimateOptions { model: Some("grok-imagine-video"), extend: Some(video.into()), ..Default::default() };
rust_llm::animate(Some("The boat passes under a bridge"), extension).await?;
# Ok(()) }
```

`with` takes reference images or a video to edit. `extend` takes a `Video` returned by an earlier
`animate`, or a path, URL, or `Attachment`, and cannot be combined with `with`. Gemini extends Veo
3.1 videos it generated, passed as the returned `Video`.

## Models and Options

The default model is `config.default_video_model` (`grok-imagine-video-1.5`). xAI, Gemini (Veo),
OpenRouter, and GPUStack generate video. Durations and resolutions go in `provider_options`, in the
provider's vocabulary:

```rust,no_run
use rust_llm::AnimateOptions;
use serde_json::json;

# async fn run() -> rust_llm::Result<()> {
let veo = AnimateOptions {
    model: Some("veo-3.1-lite-generate-preview"),
    provider_options: json!({ "parameters": { "durationSeconds": 8, "resolution": "1080p" } }),
    ..Default::default()
};
rust_llm::animate(Some("A calm ocean wave at sunset"), veo).await?;
# Ok(()) }
```

## Polling and Timeouts

```ruby
RubyLLM.configure do |config|
  config.video_generation_timeout = 600
  config.video_generation_poll_interval = 5
end
job.wait(timeout: 900, interval: 10)
```

```rust,no_run
use std::time::Duration;

# async fn run(mut job: rust_llm::VideoJob) -> rust_llm::Result<()> {
rust_llm::configure(|config| {
    config.video_generation_timeout = Duration::from_secs(600);    // default 600 s
    config.video_generation_poll_interval = Duration::from_secs(5); // default 5 s
});
job.wait(Some(Duration::from_secs(900)), Some(Duration::from_secs(10))).await?;
# Ok(()) }
```

When the timeout elapses, `animate` and `wait` fail with `Error::Api`; the provider keeps
rendering, and only the wait stops. Video jobs cannot be cancelled.

## Working with the Result

```rust,no_run
# async fn run(video: rust_llm::Video) -> rust_llm::Result<()> {
let hosted = video.url.as_deref();    // Some for providers that host the clip
let mime = video.mime_type.as_deref(); // e.g. "video/mp4"
let seconds = video.duration;
let bytes = video.to_blob().await?;   // inline data, or downloaded from the URL
# Ok(()) }
```

Save hosted videos before their URLs expire. `animate` and `animate_later` emit `video.rust_llm`
and `video_job.rust_llm` [instrumentation](instrumentation.md) events, and `AnimateOptions` takes
`metadata` for them.

## Differences from RubyLLM

- Vertex AI Veo, Bedrock, and ElevenLabs video belong to providers RustLLM does not port.
- References are paths, URLs, or `Attachment`s, not IO objects or Active Storage attachments.
