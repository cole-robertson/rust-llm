//! Port of `lib/ruby_llm/video.rb` (`RubyLLM.animate`), `lib/ruby_llm/video_job.rb`
//! (`RubyLLM.animate_later`), `Protocol#animate_later`/`refresh_video_job`, and the video seams
//! in `providers/xai/videos.rb`, `providers/openrouter/videos.rb`, `protocols/gemini/videos.rb`,
//! and `protocols/gpustack/videos.rb`.
//!
//! ```ruby
//! video = RubyLLM.animate("a paper boat sailing down a rainy gutter")
//! video.save("boat.mp4")
//! ```
//!
//! Polling honors `config.video_generation_timeout` and `config.video_generation_poll_interval`,
//! so a test can set the interval to zero instead of sleeping.

use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{Map, Value, json};

use crate::attachment::{Attachment, AttachmentType};
use crate::chat::resolve_model;
use crate::config::Config;
use crate::error::{Error, Result, error_for_status};
use crate::protocols::anthropic::unsupported;
use crate::protocols::deep_merge;
use crate::providers::Provider;
use crate::transport::{Connection, json_response};

/// A generated clip (`RubyLLM::Video`). Save it with [`Video::save`] or read its bytes with
/// [`Video::to_blob`]; both handle hosted URLs and inline data.
#[derive(Debug, Clone)]
pub struct Video {
    /// The URL of the hosted video, for providers that return one.
    pub url: Option<String>,
    /// The video bytes, returned inline or downloaded with the provider's credentials.
    pub data: Option<Vec<u8>>,
    /// The MIME type, such as `"video/mp4"`.
    pub mime_type: Option<String>,
    /// The id of the model that generated the video.
    pub model: Option<String>,
    /// The clip length in seconds, when the provider reports one.
    pub duration: Option<f64>,
    /// The provider's raw job response, for fields such as reported cost.
    pub raw: Value,
    config: Option<Arc<Config>>,
}

impl Video {
    /// `Video.new(url:, data:, mime_type:, model:, duration:, raw:)`.
    pub fn new(url: Option<String>, mime_type: Option<String>, raw: Value) -> Video {
        Video {
            url,
            data: None,
            mime_type,
            model: None,
            duration: None,
            raw,
            config: None,
        }
    }

    /// `Video#config`: the configuration of the context that generated the video, whose
    /// connection settings (`http_proxy`, `request_timeout`) the download uses; the global one otherwise.
    pub fn config(&self) -> Arc<Config> {
        self.config.clone().unwrap_or_else(crate::config)
    }

    /// The video bytes: `data` when present, otherwise downloaded from `url`.
    pub async fn to_blob(&self) -> Result<Vec<u8>> {
        if let Some(data) = &self.data {
            return Ok(data.clone());
        }
        let url = self
            .url
            .as_deref()
            .ok_or_else(|| Error::Argument("video has neither data nor a url".into()))?;
        let client = crate::transport::basic(&self.config())?;
        let response = client
            .get(url)
            .send()
            .await
            .map_err(|e| Error::ConnectionFailed(e.to_string()))?;
        let status = response.status();
        let bytes = response
            .bytes()
            .await
            .map_err(|e| Error::ConnectionFailed(e.to_string()))?;
        if !status.is_success() {
            return Err(error_for_status(
                status.as_u16(),
                &String::from_utf8_lossy(&bytes),
            ));
        }
        Ok(bytes.to_vec())
    }

    /// Writes the video to `path` and returns `path` as given.
    pub async fn save<P: AsRef<std::path::Path>>(&self, path: P) -> Result<P> {
        tokio::fs::write(path.as_ref(), self.to_blob().await?).await?;
        Ok(path)
    }
}

/// `VideoJob#status`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoStatus {
    Pending,
    Completed,
    Failed,
}

/// An in-flight video generation (`RubyLLM::VideoJob`). Poll it with [`VideoJob::refresh`] or
/// [`VideoJob::wait`], then read the clip with [`VideoJob::video`].
#[derive(Debug)]
pub struct VideoJob {
    /// The provider's id for the job.
    pub id: String,
    pub status: VideoStatus,
    /// The id of the model rendering the video.
    pub model: String,
    /// The provider's failure message when failed.
    pub error: Option<String>,
    /// The provider's raw response from the last submit or poll.
    pub raw: Value,
    /// The clip length in seconds the request asked for, or `None` when it left the length to
    /// the provider's default.
    pub duration: Option<f64>,
    /// The resolution the request asked for, as the provider spells it, such as `"720p"` or
    /// `"1280x720"`, or `None` when it left the resolution to the provider's default.
    pub resolution: Option<String>,
    family: Family,
    connection: Connection,
    video: Option<Video>,
    /// `@reported_cost`: what the provider billed, in USD, from the last poll.
    reported_cost: Option<f64>,
    /// `@usage_recorded`.
    usage_recorded: bool,
    /// `@usage_owner`: the owner at submission, who the finished job's usage is attributed to.
    usage_owner: Option<crate::accounting::UsageOwner>,
}

impl VideoJob {
    /// `pending?`: the provider is still rendering.
    pub fn is_pending(&self) -> bool {
        self.status == VideoStatus::Pending
    }

    /// `done?`: finished, successfully or not.
    pub fn is_done(&self) -> bool {
        !self.is_pending()
    }

    /// `completed?`: finished with a video.
    pub fn is_completed(&self) -> bool {
        self.status == VideoStatus::Completed
    }

    /// `failed?`: finished without a video; `error` carries the provider's message.
    pub fn is_failed(&self) -> bool {
        self.status == VideoStatus::Failed
    }

    /// `refresh`: re-fetch the job. Does nothing once the job is done.
    pub async fn refresh(&mut self) -> Result<&mut Self> {
        if self.is_done() {
            return Ok(self);
        }
        let path = match self.family {
            Family::Gemini => self.id.clone(),
            Family::XAI | Family::OpenRouter | Family::GPUStack => format!("videos/{}", self.id),
        };
        let body = self.connection.get(&path, &[]).await?.body;
        let (status, error) = self.family.parse_status(&body)?;
        self.status = status;
        self.error = error;
        self.reported_cost = self.family.reported_cost(&body);
        self.raw = body;
        self.record_usage().await;
        Ok(self)
    }

    /// `cost`: the cost the provider reported for the job. Its total is `None` while the job is
    /// pending or when the provider reports no price (only xAI and OpenRouter do).
    pub fn cost(&self) -> crate::cost::Cost {
        let tokens = crate::tokens::Tokens {
            reported_cost: self.reported_cost,
            ..Default::default()
        };
        crate::cost::Cost::new(&tokens, None, crate::cost::Tier::Standard)
    }

    /// `record_usage`: one `usage.rust_llm` entry once the job finishes, priced at the
    /// provider-reported cost.
    async fn record_usage(&mut self) {
        if self.is_pending() || self.usage_recorded {
            return;
        }
        self.usage_recorded = true;
        let mut entry = crate::message::UsageEntry::new(
            crate::message::Operation::Video,
            self.family.provider().slug(),
            Some(&self.model),
        );
        entry.status = if self.is_completed() {
            crate::message::UsageStatus::Succeeded
        } else {
            crate::message::UsageStatus::Failed
        };
        entry.cost = self.cost();
        entry.owner = self.usage_owner.clone();
        crate::accounting::report(self.connection.config(), &[entry]).await;
    }

    /// `wait(timeout:, interval:)`: poll until the job finishes. `None` uses the configured
    /// `video_generation_timeout` and `video_generation_poll_interval`. Never sleeps past the
    /// deadline. Fails when the job fails or the timeout elapses first.
    pub async fn wait(
        &mut self,
        timeout: Option<Duration>,
        interval: Option<Duration>,
    ) -> Result<&mut Self> {
        let config = self.connection.config().clone();
        let timeout = timeout.unwrap_or(config.video_generation_timeout);
        let interval = interval.unwrap_or(config.video_generation_poll_interval);
        let deadline = Instant::now() + timeout;
        while !self.is_done() {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(Error::Api(
                    format!(
                        "Video generation timed out after {} seconds",
                        seconds(timeout)
                    ),
                    None,
                ));
            }
            tokio::time::sleep(interval.min(remaining)).await;
            self.refresh().await?;
        }
        self.raise_if_failed()?;
        Ok(self)
    }

    /// `video`: the finished clip, downloading it when the provider requires that. `None` while
    /// pending; an error when the job failed.
    pub async fn video(&mut self) -> Result<Option<Video>> {
        self.raise_if_failed()?;
        if !self.is_completed() {
            return Ok(None);
        }
        if self.video.is_none() {
            let mut video = self.family.download(self).await?;
            video.config = Some(self.connection.config().clone());
            self.video = Some(video);
        }
        Ok(self.video.clone())
    }

    fn raise_if_failed(&self) -> Result<()> {
        if self.is_failed() {
            return Err(Error::Api(
                format!(
                    "Video generation failed: {}",
                    self.error.as_deref().unwrap_or("")
                ),
                None,
            ));
        }
        Ok(())
    }
}

/// Ruby prints a whole-second timeout as an Integer.
fn seconds(d: Duration) -> String {
    if d.subsec_nanos() == 0 {
        d.as_secs().to_string()
    } else {
        d.as_secs_f64().to_string()
    }
}

/// `extend:`: the video to continue.
#[derive(Debug, Clone)]
pub enum VideoSource {
    /// A `Video` returned by an earlier `animate`.
    Video(Video),
    /// A path, URL, or uploaded file.
    Attachment(Attachment),
}

impl From<Video> for VideoSource {
    fn from(v: Video) -> Self {
        VideoSource::Video(v)
    }
}

impl From<&str> for VideoSource {
    fn from(s: &str) -> Self {
        VideoSource::Attachment(Attachment::new(s))
    }
}

impl From<Attachment> for VideoSource {
    fn from(a: Attachment) -> Self {
        VideoSource::Attachment(a)
    }
}

/// Options for [`animate`] and [`animate_later`], the keyword arguments of `Video.animate`.
#[derive(Default)]
pub struct AnimateOptions<'a> {
    /// `model:`, defaulting to `config.default_video_model`.
    pub model: Option<&'a str>,
    pub provider: Option<&'a str>,
    pub assume_model_exists: bool,
    /// `with:`: a reference image, or a video to edit.
    pub with: Vec<Attachment>,
    /// `extend:`: a video to continue. Cannot be combined with `with`.
    pub extend: Option<VideoSource>,
    /// `provider_options:`: durations, resolutions, ... in the provider's vocabulary.
    pub provider_options: Value,
    /// `context:`: use this configuration instead of the global one.
    pub config: Option<Arc<Config>>,
    /// `metadata:`: added to the `video.rust_llm` and `video_job.rust_llm` event payloads, never
    /// sent to the provider.
    pub metadata: Option<Value>,
    /// `owner:`: who the job's usage is attributed to once it finishes, such as a user; wins over
    /// [`crate::accounting::with_usage_owner`].
    pub owner: Option<crate::accounting::UsageOwner>,
}

/// `RubyLLM.animate(prompt, ...)`: submit a job and poll until the video is ready, inside a
/// `video.rust_llm` event.
pub async fn animate(prompt: Option<&str>, options: AnimateOptions<'_>) -> Result<Video> {
    let config = options.config.clone().unwrap_or_else(crate::config);
    let mut event = crate::instrumentation::Event::start(&config, "video.rust_llm", || {
        crate::instrumentation::payload([
            ("model", options.model.into()),
            ("prompt", prompt.into()),
            ("provider_options", options.provider_options.clone()),
            (
                "metadata",
                crate::instrumentation::metadata(&options.metadata),
            ),
        ])
    });
    let run = async {
        let mut job = animate_later(prompt, options).await?;
        let (model, id) = (job.model.clone(), job.id.clone());
        let video = async {
            job.wait(None, None).await?;
            job.video()
                .await?
                .ok_or_else(|| Error::Api("Video generation finished without a video".into(), None))
        }
        .await;
        Ok::<_, Error>((model, id, video))
    };
    let (job, result) = match event.instrument(run).await {
        Ok((model, id, video)) => (Some((model, id)), video),
        Err(e) => (None, Err(e)),
    };
    if let Some((model, id)) = job {
        event.set("model", || model.into());
        event.set("job_id", || id.into());
    }
    if let Ok(video) = &result {
        event.set("result", || {
            json!({ "url": video.url, "mime_type": video.mime_type, "model": video.model, "duration": video.duration })
        });
        event.set("response_model", || video.model.clone().into());
    }
    event.finish(result.as_ref().err());
    result
}

/// `RubyLLM.animate_later(prompt, ...)`: submit a job and return it without waiting, inside a
/// `video_job.rust_llm` event.
pub async fn animate_later(prompt: Option<&str>, options: AnimateOptions<'_>) -> Result<VideoJob> {
    let config = options.config.clone().unwrap_or_else(crate::config);
    let model_id = options
        .model
        .unwrap_or(&config.default_video_model)
        .to_string();
    let (model, provider) =
        resolve_model(&model_id, options.provider, options.assume_model_exists)?;
    let owner = options.owner.clone();
    let mut event = crate::instrumentation::Event::start(&config, "video_job.rust_llm", || {
        crate::instrumentation::payload([
            ("provider", provider.slug().into()),
            ("provider_class", provider.display().into()),
            ("model", model.id.clone().into()),
            ("prompt", prompt.into()),
            ("provider_options", options.provider_options.clone()),
            (
                "metadata",
                crate::instrumentation::metadata(&options.metadata),
            ),
        ])
    });
    let result = event
        .instrument(crate::accounting::owned_by(
            owner,
            submit(prompt, options, config.clone(), model, provider),
        ))
        .await;
    if let Ok(job) = &result {
        event.set("job_id", || job.id.clone().into());
    }
    event.finish(result.as_ref().err());
    result
}

async fn submit(
    prompt: Option<&str>,
    options: AnimateOptions<'_>,
    config: Arc<Config>,
    model: crate::model::Model,
    provider: Provider,
) -> Result<VideoJob> {
    let AnimateOptions {
        mut with,
        extend,
        provider_options,
        ..
    } = options;
    // `provider_options: {}` by default; `Null` would replace the payload in a deep merge.
    let provider_options = if provider_options.is_null() {
        json!({})
    } else {
        provider_options
    };
    provider.ensure_configured(&config)?;
    let family = match Family::for_provider(provider) {
        Ok(family) => family,
        // `Protocol#render_video_extension_payload` raises before the missing video seams do.
        Err(_) if extend.is_some() => {
            return Err(Error::Api(
                format!("{} doesn't support video extension", provider.display()),
                None,
            ));
        }
        Err(e) => return Err(e),
    };
    if !with.is_empty() && extend.is_some() {
        return Err(Error::Argument(
            "with: and extend: cannot be combined".into(),
        ));
    }
    let connection = Connection::new(provider, config.clone())?;
    let (path, payload) = match extend {
        Some(source) => {
            family
                .render_extension(prompt, &model.id, source, &provider_options, &connection)
                .await?
        }
        None => {
            family.validate(&with, &config)?;
            for a in with.iter_mut().filter(|a| family.loads(a)) {
                a.load(connection.client()).await?;
            }
            family.render(prompt, &model.id, &with, &provider_options)?
        }
    };
    // `post_video`: submitting creates a job, so a lost response is never retried. GPUStack takes
    // the payload as multipart form fields.
    let resp = connection
        .send(
            reqwest::Method::POST,
            &path,
            &[],
            false,
            &|req| match family {
                Family::GPUStack => req.multipart(multipart_form(&payload)),
                _ => req.json(&payload),
            },
        )
        .await?;
    let body = json_response(resp, payload.clone()).await?.body;
    let (id, status, error) = family.parse_job(&body)?;
    let (duration, resolution) = family.parse_request(&payload);
    let model = match family {
        Family::GPUStack => body.get("model").and_then(Value::as_str),
        _ => None,
    }
    .unwrap_or(&model.id)
    .to_string();
    let mut job = VideoJob {
        id,
        status,
        model,
        error,
        raw: body,
        duration,
        resolution,
        family,
        connection,
        video: None,
        reported_cost: None,
        usage_recorded: false,
        usage_owner: crate::accounting::usage_owner(),
    };
    job.record_usage().await;
    Ok(job)
}

/// Which video seams a provider's protocol includes.
#[derive(Debug, Clone, Copy)]
enum Family {
    XAI,
    OpenRouter,
    Gemini,
    GPUStack,
}

const OPENROUTER_FRAME_TYPES: &[&str] = &["first_frame", "last_frame"];

impl Family {
    fn for_provider(provider: Provider) -> Result<Family> {
        match provider {
            Provider::XAI => Ok(Family::XAI),
            Provider::OpenRouter => Ok(Family::OpenRouter),
            Provider::Gemini => Ok(Family::Gemini),
            Provider::GPUStack => Ok(Family::GPUStack),
            other => Err(Error::Api(
                format!("{} doesn't support video generation", other.display()),
                None,
            )),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Family::XAI => "XAI",
            Family::OpenRouter => "OpenRouter",
            Family::Gemini => "Gemini",
            Family::GPUStack => "GPUStack",
        }
    }

    /// Whether the attachment's bytes are needed to render it. Veo and vLLM-Omni inline every
    /// reference.
    fn loads(self, a: &Attachment) -> bool {
        matches!(self, Family::Gemini | Family::GPUStack) || (!a.is_url() && !a.is_provider_file())
    }

    /// `validate_animate_inputs!`.
    fn validate(self, with: &[Attachment], config: &Config) -> Result<()> {
        let refuse = |a: &Attachment| Err(Error::UnsupportedAttachment(unsupported(&a.mime_type)));
        match self {
            Family::XAI => {
                if with.len() > 1 {
                    return Err(Error::Api(
                        "xAI video generation takes a single reference image or video".into(),
                        None,
                    ));
                }
                for a in with.iter().filter(|a| !a.is_provider_file() && !a.is_url()) {
                    if !matches!(a.kind(), AttachmentType::Image | AttachmentType::Video) {
                        return refuse(a);
                    }
                }
            }
            Family::OpenRouter => {
                if with.len() > 2 {
                    return Err(Error::Api(
                        "OpenRouter video generation takes at most first and last frame images"
                            .into(),
                        None,
                    ));
                }
                for a in with.iter().filter(|a| !a.is_url()) {
                    if a.kind() != AttachmentType::Image {
                        return refuse(a);
                    }
                }
            }
            Family::Gemini => {
                if with.len() > 1 {
                    return Err(Error::Api(
                        "Veo takes a single reference image".into(),
                        None,
                    ));
                }
                for a in with {
                    if a.kind() != AttachmentType::Image {
                        return refuse(a);
                    }
                }
            }
            Family::GPUStack => {
                // `@provider.backend_api_base`: video jobs are only served by the model proxy,
                // whose `/v1/videos` is then this base's `videos`.
                crate::tokenization::gpustack_backend_base(&Provider::GPUStack.api_base(config)?)?;
                for a in with {
                    if a.is_provider_file() {
                        return Err(Error::Argument(
                            "vLLM-Omni video references require media bytes or URLs, not uploaded file ids".into(),
                        ));
                    }
                    if !matches!(
                        a.kind(),
                        AttachmentType::Image | AttachmentType::Video | AttachmentType::Audio
                    ) {
                        return refuse(a);
                    }
                }
            }
        }
        Ok(())
    }

    /// `video_request_url` and `render_video_payload`.
    fn render(
        self,
        prompt: Option<&str>,
        model: &str,
        with: &[Attachment],
        provider_options: &Value,
    ) -> Result<(String, Value)> {
        match self {
            Family::XAI => {
                let mut payload = json!({ "model": model, "prompt": prompt });
                if let Some(a) = with.first() {
                    let key = if a.kind() == AttachmentType::Video {
                        "video"
                    } else {
                        "image"
                    };
                    payload[key] = xai_reference(a)?;
                }
                let payload = merge(payload, provider_options);
                let path = if payload.get("video").is_some() {
                    "videos/edits"
                } else {
                    "videos/generations"
                };
                Ok((path.into(), payload))
            }
            Family::OpenRouter => {
                let mut payload = json!({ "model": model, "prompt": prompt });
                if !with.is_empty() {
                    let frames = with
                        .iter()
                        .zip(OPENROUTER_FRAME_TYPES)
                        .map(|(a, frame)| Ok(json!({ "type": "image_url", "image_url": { "url": a.url_or_data_uri()? }, "frame_type": frame })))
                        .collect::<Result<Vec<_>>>()?;
                    payload["frame_images"] = frames.into();
                }
                Ok(("videos".into(), merge(payload, provider_options)))
            }
            Family::Gemini => {
                let mut instance = json!({ "prompt": prompt });
                if let Some(image) = with.first() {
                    instance["image"] = json!({ "inlineData": { "mimeType": image.mime_type, "data": image.encoded()? } });
                }
                let mut payload = json!({ "instances": [instance] });
                deep_merge(&mut payload, provider_options);
                Ok((format!("models/{model}:predictLongRunning"), payload))
            }
            Family::GPUStack => {
                let prompt = prompt.ok_or_else(|| {
                    Error::Argument("vLLM-Omni video generation requires a prompt".into())
                })?;
                if provider_options
                    .get("num_outputs_per_prompt")
                    .is_some_and(|n| !n.is_null() && n.as_f64() != Some(1.0))
                {
                    return Err(Error::Argument(
                        "animate returns one video; num_outputs_per_prompt must be 1".into(),
                    ));
                }
                let mut payload = json!({ "model": model, "prompt": prompt });
                if let Some(p) = payload.as_object_mut() {
                    p.extend(gpustack_references(with)?);
                }
                // `.compact`, then nested values as JSON: every field is a multipart form field.
                let fields: Map<String, Value> = merge(payload, provider_options)
                    .as_object()
                    .into_iter()
                    .flatten()
                    .filter(|(_, v)| !v.is_null())
                    .map(|(k, v)| {
                        let v = match v {
                            Value::Object(_) | Value::Array(_) => Value::String(v.to_string()),
                            other => other.clone(),
                        };
                        (k.clone(), v)
                    })
                    .collect();
                Ok(("videos".into(), Value::Object(fields)))
            }
        }
    }

    /// `video_extension_url` and `render_video_extension_payload`.
    async fn render_extension(
        self,
        prompt: Option<&str>,
        model: &str,
        source: VideoSource,
        provider_options: &Value,
        connection: &Connection,
    ) -> Result<(String, Value)> {
        match self {
            Family::XAI => {
                let mut video = extension_attachment(source)?;
                if !video.is_url() && !video.is_provider_file() {
                    video.load(connection.client()).await?;
                }
                let payload =
                    json!({ "model": model, "prompt": prompt, "video": xai_reference(&video)? });
                Ok(("videos/extensions".into(), merge(payload, provider_options)))
            }
            Family::Gemini => {
                let uri = match &source {
                    VideoSource::Video(v) => gemini_generated_video(&v.raw)
                        .and_then(|g| g.get("uri"))
                        .and_then(Value::as_str)
                        .map(str::to_string),
                    VideoSource::Attachment(_) => None,
                };
                let uri = match uri {
                    Some(uri) => uri,
                    None => {
                        let video = extension_attachment(source)?;
                        match (video.url(), video.provider_file_uri()) {
                            (Some(url), _) => url.to_string(),
                            (None, Some(uri)) => uri.to_string(),
                            _ => {
                                return Err(Error::Argument(
                                    "Gemini extends generated Veo videos; pass the returned Video or its URI".into(),
                                ));
                            }
                        }
                    }
                };
                let mut payload =
                    json!({ "instances": [{ "prompt": prompt, "video": { "uri": uri } }] });
                deep_merge(&mut payload, provider_options);
                Ok((format!("models/{model}:predictLongRunning"), payload))
            }
            Family::OpenRouter | Family::GPUStack => Err(Error::Api(
                format!("{} doesn't support video extension", self.name()),
                None,
            )),
        }
    }

    /// `parse_video_request`: the duration in seconds and the resolution the rendered request
    /// asked for, read from each protocol's own fields (`video_request_settings`).
    fn parse_request(self, payload: &Value) -> (Option<f64>, Option<String>) {
        let field = |key: &str| payload.get(key).filter(|v| !v.is_null());
        let (duration, resolution) = match self {
            Family::XAI => (field("duration"), field("resolution")),
            Family::OpenRouter => (
                field("duration"),
                field("resolution").or_else(|| field("size")),
            ),
            Family::Gemini => {
                let parameters = payload.get("parameters");
                (
                    parameters.and_then(|p| p.get("durationSeconds")),
                    parameters.and_then(|p| p.get("resolution")),
                )
            }
            Family::GPUStack => (field("seconds"), field("size")),
        };
        (
            duration.and_then(video_seconds),
            resolution.and_then(Value::as_str).map(str::to_string),
        )
    }

    /// The provider whose protocol includes these seams.
    fn provider(self) -> Provider {
        match self {
            Family::XAI => Provider::XAI,
            Family::OpenRouter => Provider::OpenRouter,
            Family::Gemini => Provider::Gemini,
            Family::GPUStack => Provider::GPUStack,
        }
    }

    /// `parse_video_job_status`'s `reported_cost: reported_cost(body['usage'] || {})`: xAI's USD
    /// ticks and OpenRouter's dollars, through their chat protocols' helpers.
    fn reported_cost(self, body: &Value) -> Option<f64> {
        match self {
            Family::XAI | Family::OpenRouter => crate::protocols::chat_completions::reported_cost(
                self.provider(),
                body.get("usage").unwrap_or(&json!({})),
            ),
            Family::Gemini | Family::GPUStack => None,
        }
    }

    /// `parse_video_job`: the job id, its state, and the provider's failure message.
    fn parse_job(self, body: &Value) -> Result<(String, VideoStatus, Option<String>)> {
        let (key, message) = match self {
            Family::XAI => ("request_id", "xAI did not return a video request id"),
            Family::OpenRouter => ("id", "OpenRouter did not return a video job"),
            Family::Gemini => ("name", "Gemini did not return a video generation operation"),
            Family::GPUStack => ("id", "GPUStack did not return a video job id"),
        };
        let id = body
            .get(key)
            .and_then(Value::as_str)
            .ok_or_else(|| Error::Api(message.into(), None))?;
        let (status, error) = match self {
            Family::OpenRouter => (openrouter_status(body), None),
            Family::GPUStack => self.parse_status(body)?,
            _ => (VideoStatus::Pending, None),
        };
        Ok((id.to_string(), status, error))
    }

    /// `parse_video_job_status`: the job's state and, when failed, the provider's message.
    fn parse_status(self, body: &Value) -> Result<(VideoStatus, Option<String>)> {
        let status_or_error = || {
            Some(
                body.get("error")
                    .filter(|e| !e.is_null())
                    .or_else(|| body.get("status"))
                    .map(display)
                    .unwrap_or_default(),
            )
        };
        Ok(match self {
            Family::XAI => match body.get("status").and_then(Value::as_str) {
                Some("done") => (VideoStatus::Completed, None),
                Some("failed" | "expired") => (VideoStatus::Failed, status_or_error()),
                _ => (VideoStatus::Pending, None),
            },
            Family::OpenRouter => match openrouter_status(body) {
                VideoStatus::Failed => (VideoStatus::Failed, status_or_error()),
                status => (status, None),
            },
            Family::Gemini => {
                if let Some(error) = body.get("error").filter(|e| !e.is_null()) {
                    return Ok((VideoStatus::Failed, error.get("message").map(display)));
                }
                if !body.get("done").and_then(Value::as_bool).unwrap_or(false) {
                    return Ok((VideoStatus::Pending, None));
                }
                if gemini_generated_video(body).is_some() {
                    return Ok((VideoStatus::Completed, None));
                }
                let reasons: Vec<String> = body
                    .pointer("/response/generateVideoResponse/raiMediaFilteredReasons")
                    .and_then(Value::as_array)
                    .map(|r| r.iter().map(display).collect())
                    .unwrap_or_default();
                let error = if reasons.is_empty() {
                    "Gemini returned no video".to_string()
                } else {
                    reasons.join(" ")
                };
                (VideoStatus::Failed, Some(error))
            }
            // `video_job_state`: an unknown state is an error rather than a job polled forever.
            Family::GPUStack => {
                let status = match body.get("status").and_then(Value::as_str) {
                    Some("queued" | "in_progress") => VideoStatus::Pending,
                    Some("completed") => VideoStatus::Completed,
                    Some("failed") => VideoStatus::Failed,
                    _ => {
                        let status = body
                            .get("status")
                            .filter(|s| !s.is_null())
                            .map_or_else(|| "nil".to_string(), Value::to_string);
                        return Err(Error::Api(
                            format!("Unknown GPUStack video status: {status}"),
                            None,
                        ));
                    }
                };
                (status, body.pointer("/error/message").map(display))
            }
        })
    }

    /// `download_video`.
    async fn download(self, job: &VideoJob) -> Result<Video> {
        match self {
            Family::XAI => {
                let video = job.raw.get("video").cloned().unwrap_or_else(|| json!({}));
                Ok(Video {
                    url: video.get("url").and_then(Value::as_str).map(str::to_string),
                    data: None,
                    mime_type: Some("video/mp4".into()),
                    model: Some(
                        job.raw
                            .get("model")
                            .and_then(Value::as_str)
                            .unwrap_or(&job.model)
                            .to_string(),
                    ),
                    duration: video.get("duration").and_then(Value::as_f64),
                    raw: job.raw.clone(),
                    config: None,
                })
            }
            Family::OpenRouter => {
                let (data, content_type) = fetch(
                    &job.connection,
                    &format!("videos/{}/content?index=0", job.id),
                )
                .await?;
                Ok(Video {
                    url: None,
                    data: Some(data),
                    mime_type: Some(content_type.unwrap_or_else(|| "video/mp4".into())),
                    model: Some(job.model.clone()),
                    duration: None,
                    raw: job.raw.clone(),
                    config: None,
                })
            }
            Family::Gemini => {
                let generated = gemini_generated_video(&job.raw)
                    .cloned()
                    .unwrap_or_else(|| json!({}));
                let uri = generated
                    .get("uri")
                    .and_then(Value::as_str)
                    .ok_or_else(|| Error::Api("Gemini returned no video".into(), None))?;
                // The file URI redirects to the download host; both hops carry the API key.
                let (data, _) = fetch(&job.connection, uri).await?;
                Ok(Video {
                    url: None,
                    data: Some(data),
                    mime_type: Some(
                        generated
                            .get("mimeType")
                            .and_then(Value::as_str)
                            .unwrap_or("video/mp4")
                            .to_string(),
                    ),
                    model: Some(job.model.clone()),
                    duration: None,
                    raw: job.raw.clone(),
                    config: None,
                })
            }
            Family::GPUStack => {
                let (data, content_type) =
                    fetch(&job.connection, &format!("videos/{}/content", job.id)).await?;
                Ok(Video {
                    url: None,
                    data: Some(data),
                    mime_type: content_type.or_else(|| {
                        job.raw
                            .get("media_type")
                            .and_then(Value::as_str)
                            .map(str::to_string)
                    }),
                    model: Some(job.model.clone()),
                    duration: None,
                    raw: job.raw.clone(),
                    config: None,
                })
            }
        }
    }
}

/// `GPUStack::Videos#video_references`: `<type>_reference` per media type in order of first
/// appearance, one `{ "<type>_url": data URI }` or an array of them.
fn gpustack_references(with: &[Attachment]) -> Result<Map<String, Value>> {
    let mut groups: Vec<(&str, Vec<Value>)> = Vec::new();
    for a in with {
        let kind = match a.kind() {
            AttachmentType::Image => "image",
            AttachmentType::Video => "video",
            _ => "audio",
        };
        let mut reference = Map::new();
        reference.insert(format!("{kind}_url"), a.for_llm()?.into());
        match groups.iter_mut().find(|(k, _)| *k == kind) {
            Some((_, values)) => values.push(reference.into()),
            None => groups.push((kind, vec![reference.into()])),
        }
    }
    Ok(groups
        .into_iter()
        .map(|(kind, mut values)| {
            let value = if values.len() == 1 {
                values.remove(0)
            } else {
                values.into()
            };
            (format!("{kind}_reference"), value)
        })
        .collect())
}

/// Faraday's multipart encoding of a flat payload: each value as a text field (`to_s`).
fn multipart_form(payload: &Value) -> reqwest::multipart::Form {
    payload
        .as_object()
        .into_iter()
        .flatten()
        .fold(reqwest::multipart::Form::new(), |form, (k, v)| {
            form.text(k.clone(), display(v))
        })
}

/// An authenticated GET for a binary body, returning the bytes and the response content type.
async fn fetch(connection: &Connection, path: &str) -> Result<(Vec<u8>, Option<String>)> {
    let resp = connection
        .send(reqwest::Method::GET, path, &[], true, &|req| req)
        .await?;
    let content_type = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let bytes = resp
        .bytes()
        .await
        .map_err(|e| Error::ConnectionFailed(e.to_string()))?;
    Ok((bytes.to_vec(), content_type))
}

fn openrouter_status(body: &Value) -> VideoStatus {
    match body.get("status").and_then(Value::as_str) {
        Some("completed") => VideoStatus::Completed,
        Some("failed") => VideoStatus::Failed,
        _ => VideoStatus::Pending,
    }
}

fn gemini_generated_video(body: &Value) -> Option<&Value> {
    body.pointer("/response/generateVideoResponse/generatedSamples/0/video")
        .filter(|v| !v.is_null())
}

/// `parse_video_seconds`: a number as is; a string like `"8"` or `"8s"` as its number.
fn video_seconds(value: &Value) -> Option<f64> {
    match value {
        // Ruby's `Float()` refuses "inf" and "NaN", which Rust's parser takes.
        Value::String(s) => s
            .strip_suffix('s')
            .unwrap_or(s)
            .parse::<f64>()
            .ok()
            .filter(|n| n.is_finite()),
        other => other.as_f64(),
    }
}

/// Ruby string interpolation of a JSON value: strings bare, everything else as JSON.
fn display(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// `Protocol#video_extension_attachment`: a returned `Video` becomes its URL or bytes; the
/// source must be a video.
fn extension_attachment(source: VideoSource) -> Result<Attachment> {
    let attachment = match source {
        VideoSource::Attachment(a) => a,
        VideoSource::Video(v) => match (v.url, v.data) {
            (Some(url), _) => Attachment::new(url),
            (None, Some(data)) => Attachment::from_bytes(data, "video.mp4", None),
            (None, None) => return Err(Error::Argument("extend: takes exactly one video".into())),
        },
    };
    if attachment.kind() != AttachmentType::Video {
        return Err(Error::UnsupportedAttachment(unsupported(
            &attachment.mime_type,
        )));
    }
    Ok(attachment)
}

/// `XAI::Videos#video_reference`: an uploaded file by id, anything else by URL or data URI.
fn xai_reference(a: &Attachment) -> Result<Value> {
    if let Some(file_id) = a.provider_file_id() {
        return Ok(json!({ "file_id": file_id }));
    }
    Ok(json!({ "url": a.url_or_data_uri()? }))
}

/// Ruby's `Hash#merge`: provider options replace top-level keys.
fn merge(mut payload: Value, options: &Value) -> Value {
    if let (Some(p), Some(o)) = (payload.as_object_mut(), options.as_object()) {
        p.extend(o.clone());
    }
    payload
}

#[cfg(test)]
mod tests {
    use super::*;

    // xai/videos_spec.rb: "sends model, prompt, and provider options".
    #[test]
    fn xai_sends_model_prompt_and_provider_options() {
        let (path, payload) = Family::XAI
            .render(
                Some("make the water crash down"),
                "grok-imagine-video",
                &[],
                &json!({ "duration": 1, "resolution": "480p" }),
            )
            .unwrap();
        assert_eq!(path, "videos/generations");
        assert_eq!(
            payload,
            json!({ "model": "grok-imagine-video", "prompt": "make the water crash down", "duration": 1, "resolution": "480p" })
        );
    }

    // xai/videos_spec.rb: "selects the editing route for local video attachments".
    #[test]
    fn xai_edits_a_local_video_through_a_data_uri() {
        let video = Attachment::from_bytes(b"mp4 bytes".to_vec(), "clip.mp4", None);
        let (path, payload) = Family::XAI
            .render(
                Some("Turn the background blue"),
                "grok-imagine-video",
                &[video],
                &Value::Null,
            )
            .unwrap();
        assert_eq!(path, "videos/edits");
        assert_eq!(
            payload["video"],
            json!({ "url": "data:video/mp4;base64,bXA0IGJ5dGVz" })
        );
    }

    #[test]
    fn gemini_reports_filtered_reasons_as_the_failure() {
        let body = json!({ "done": true, "response": { "generateVideoResponse": { "raiMediaFilteredReasons": ["unsafe", "content"] } } });
        assert_eq!(
            Family::Gemini.parse_status(&body).unwrap(),
            (VideoStatus::Failed, Some("unsafe content".into()))
        );
        assert_eq!(
            Family::Gemini
                .parse_status(&json!({ "done": true }))
                .unwrap(),
            (VideoStatus::Failed, Some("Gemini returned no video".into()))
        );
        assert_eq!(
            Family::Gemini
                .parse_status(&json!({ "name": "op" }))
                .unwrap()
                .0,
            VideoStatus::Pending
        );
    }

    // spec: protocols/gemini/videos_spec.rb:43 #parse_video_request > reads the requested duration and resolution from the parameters
    #[test]
    fn gemini_reads_the_requested_duration_and_resolution_from_the_parameters() {
        let (_, payload) = Family::Gemini
            .render(
                Some("a hummingbird"),
                "veo-3.1-fast-generate-preview",
                &[],
                &json!({ "parameters": { "durationSeconds": 8, "resolution": "1080p" } }),
            )
            .unwrap();

        assert_eq!(
            Family::Gemini.parse_request(&payload),
            (Some(8.0), Some("1080p".into()))
        );
        assert_eq!(
            Family::Gemini.parse_request(&json!({ "instances": [{ "prompt": "a hummingbird" }] })),
            (None, None)
        );
    }

    #[test]
    fn xai_expired_jobs_fail_with_their_status() {
        assert_eq!(
            Family::XAI
                .parse_status(&json!({ "status": "expired" }))
                .unwrap(),
            (VideoStatus::Failed, Some("expired".into()))
        );
    }
}
