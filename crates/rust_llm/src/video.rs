//! Port of `lib/ruby_llm/video.rb` (`RubyLLM.animate`), `lib/ruby_llm/video_job.rb`
//! (`RubyLLM.animate_later`), `Protocol#animate_later`/`refresh_video_job`, and the video seams
//! in `providers/xai/videos.rb`, `providers/openrouter/videos.rb`, and `protocols/gemini/videos.rb`.
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

use serde_json::{Value, json};

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
        Video { url, data: None, mime_type, model: None, duration: None, raw, config: None }
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
        let url = self.url.as_deref().ok_or_else(|| Error::Argument("video has neither data nor a url".into()))?;
        let client = crate::transport::basic(&self.config())?;
        let response = client.get(url).send().await.map_err(|e| Error::ConnectionFailed(e.to_string()))?;
        let status = response.status();
        let bytes = response.bytes().await.map_err(|e| Error::ConnectionFailed(e.to_string()))?;
        if !status.is_success() {
            return Err(error_for_status(status.as_u16(), &String::from_utf8_lossy(&bytes)));
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
    family: Family,
    connection: Connection,
    video: Option<Video>,
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
            Family::XAI | Family::OpenRouter => format!("videos/{}", self.id),
        };
        let body = self.connection.get(&path, &[]).await?.body;
        let (status, error) = self.family.parse_status(&body);
        self.status = status;
        self.error = error;
        self.raw = body;
        Ok(self)
    }

    /// `wait(timeout:, interval:)`: poll until the job finishes. `None` uses the configured
    /// `video_generation_timeout` and `video_generation_poll_interval`. Never sleeps past the
    /// deadline. Fails when the job fails or the timeout elapses first.
    pub async fn wait(&mut self, timeout: Option<Duration>, interval: Option<Duration>) -> Result<&mut Self> {
        let config = self.connection.config().clone();
        let timeout = timeout.unwrap_or(config.video_generation_timeout);
        let interval = interval.unwrap_or(config.video_generation_poll_interval);
        let deadline = Instant::now() + timeout;
        while !self.is_done() {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(Error::Api(format!("Video generation timed out after {} seconds", seconds(timeout)), None));
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
            return Err(Error::Api(format!("Video generation failed: {}", self.error.as_deref().unwrap_or("")), None));
        }
        Ok(())
    }
}

/// Ruby prints a whole-second timeout as an Integer.
fn seconds(d: Duration) -> String {
    if d.subsec_nanos() == 0 { d.as_secs().to_string() } else { d.as_secs_f64().to_string() }
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
}

/// `RubyLLM.animate(prompt, ...)`: submit a job and poll until the video is ready.
pub async fn animate(prompt: Option<&str>, options: AnimateOptions<'_>) -> Result<Video> {
    let mut job = animate_later(prompt, options).await?;
    job.wait(None, None).await?;
    job.video().await?.ok_or_else(|| Error::Api("Video generation finished without a video".into(), None))
}

/// `RubyLLM.animate_later(prompt, ...)`: submit a job and return it without waiting.
pub async fn animate_later(prompt: Option<&str>, options: AnimateOptions<'_>) -> Result<VideoJob> {
    let AnimateOptions { model, provider, assume_model_exists, mut with, extend, provider_options, config } = options;
    let config = config.unwrap_or_else(crate::config);
    let model_id = model.unwrap_or(&config.default_video_model).to_string();
    let (model, provider) = resolve_model(&model_id, provider, assume_model_exists)?;
    provider.ensure_configured(&config)?;
    let family = Family::for_provider(provider)?;
    if !with.is_empty() && extend.is_some() {
        return Err(Error::Argument("with: and extend: cannot be combined".into()));
    }
    let connection = Connection::new(provider, config.clone())?;
    let (path, payload) = match extend {
        Some(source) => family.render_extension(prompt, &model.id, source, &provider_options, &connection).await?,
        None => {
            family.validate(&with)?;
            for a in with.iter_mut().filter(|a| family.loads(a)) {
                a.load(connection.client()).await?;
            }
            family.render(prompt, &model.id, &with, &provider_options)?
        }
    };
    // `post_video`: submitting creates a job, so a lost response is never retried.
    let resp = connection.send(reqwest::Method::POST, &path, &[], false, &|req| req.json(&payload)).await?;
    let body = json_response(resp, payload.clone()).await?.body;
    let (id, status) = family.parse_job(&body)?;
    Ok(VideoJob { id, status, model: model.id.clone(), error: None, raw: body, family, connection, video: None })
}

/// Which video seams a provider's protocol includes.
#[derive(Debug, Clone, Copy)]
enum Family {
    XAI,
    OpenRouter,
    Gemini,
}

const OPENROUTER_FRAME_TYPES: &[&str] = &["first_frame", "last_frame"];

impl Family {
    fn for_provider(provider: Provider) -> Result<Family> {
        match provider {
            Provider::XAI => Ok(Family::XAI),
            Provider::OpenRouter => Ok(Family::OpenRouter),
            Provider::Gemini => Ok(Family::Gemini),
            other => Err(Error::Api(format!("{} doesn't support video generation", other.display()), None)),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Family::XAI => "XAI",
            Family::OpenRouter => "OpenRouter",
            Family::Gemini => "Gemini",
        }
    }

    /// Whether the attachment's bytes are needed to render it. Veo inlines every image.
    fn loads(self, a: &Attachment) -> bool {
        matches!(self, Family::Gemini) || (!a.is_url() && !a.is_provider_file())
    }

    /// `validate_animate_inputs!`.
    fn validate(self, with: &[Attachment]) -> Result<()> {
        let refuse = |a: &Attachment| Err(Error::UnsupportedAttachment(unsupported(&a.mime_type)));
        match self {
            Family::XAI => {
                if with.len() > 1 {
                    return Err(Error::Api("xAI video generation takes a single reference image or video".into(), None));
                }
                for a in with.iter().filter(|a| !a.is_provider_file() && !a.is_url()) {
                    if !matches!(a.kind(), AttachmentType::Image | AttachmentType::Video) {
                        return refuse(a);
                    }
                }
            }
            Family::OpenRouter => {
                if with.len() > 2 {
                    return Err(Error::Api("OpenRouter video generation takes at most first and last frame images".into(), None));
                }
                for a in with.iter().filter(|a| !a.is_url()) {
                    if a.kind() != AttachmentType::Image {
                        return refuse(a);
                    }
                }
            }
            Family::Gemini => {
                if with.len() > 1 {
                    return Err(Error::Api("Veo takes a single reference image".into(), None));
                }
                for a in with {
                    if a.kind() != AttachmentType::Image {
                        return refuse(a);
                    }
                }
            }
        }
        Ok(())
    }

    /// `video_request_url` and `render_video_payload`.
    fn render(self, prompt: Option<&str>, model: &str, with: &[Attachment], provider_options: &Value) -> Result<(String, Value)> {
        match self {
            Family::XAI => {
                let mut payload = json!({ "model": model, "prompt": prompt });
                if let Some(a) = with.first() {
                    let key = if a.kind() == AttachmentType::Video { "video" } else { "image" };
                    payload[key] = xai_reference(a)?;
                }
                let payload = merge(payload, provider_options);
                let path = if payload.get("video").is_some() { "videos/edits" } else { "videos/generations" };
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
                let payload = json!({ "model": model, "prompt": prompt, "video": xai_reference(&video)? });
                Ok(("videos/extensions".into(), merge(payload, provider_options)))
            }
            Family::Gemini => {
                let uri = match &source {
                    VideoSource::Video(v) => gemini_generated_video(&v.raw).and_then(|g| g.get("uri")).and_then(Value::as_str).map(str::to_string),
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
                let mut payload = json!({ "instances": [{ "prompt": prompt, "video": { "uri": uri } }] });
                deep_merge(&mut payload, provider_options);
                Ok((format!("models/{model}:predictLongRunning"), payload))
            }
            Family::OpenRouter => Err(Error::Api(format!("{} doesn't support video extension", self.name()), None)),
        }
    }

    /// `parse_video_job`.
    fn parse_job(self, body: &Value) -> Result<(String, VideoStatus)> {
        let (key, message) = match self {
            Family::XAI => ("request_id", "xAI did not return a video request id"),
            Family::OpenRouter => ("id", "OpenRouter did not return a video job"),
            Family::Gemini => ("name", "Gemini did not return a video generation operation"),
        };
        let id = body.get(key).and_then(Value::as_str).ok_or_else(|| Error::Api(message.into(), None))?;
        let status = match self {
            Family::OpenRouter => openrouter_status(body),
            _ => VideoStatus::Pending,
        };
        Ok((id.to_string(), status))
    }

    /// `parse_video_job_status`: the job's state and, when failed, the provider's message.
    fn parse_status(self, body: &Value) -> (VideoStatus, Option<String>) {
        let status_or_error = || Some(body.get("error").filter(|e| !e.is_null()).or_else(|| body.get("status")).map(display).unwrap_or_default());
        match self {
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
                    return (VideoStatus::Failed, error.get("message").map(display));
                }
                if !body.get("done").and_then(Value::as_bool).unwrap_or(false) {
                    return (VideoStatus::Pending, None);
                }
                if gemini_generated_video(body).is_some() {
                    return (VideoStatus::Completed, None);
                }
                let reasons: Vec<String> = body
                    .pointer("/response/generateVideoResponse/raiMediaFilteredReasons")
                    .and_then(Value::as_array)
                    .map(|r| r.iter().map(display).collect())
                    .unwrap_or_default();
                let error = if reasons.is_empty() { "Gemini returned no video".to_string() } else { reasons.join(" ") };
                (VideoStatus::Failed, Some(error))
            }
        }
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
                    model: Some(job.raw.get("model").and_then(Value::as_str).unwrap_or(&job.model).to_string()),
                    duration: video.get("duration").and_then(Value::as_f64),
                    raw: job.raw.clone(),
                    config: None,
                })
            }
            Family::OpenRouter => {
                let (data, content_type) = fetch(&job.connection, &format!("videos/{}/content?index=0", job.id)).await?;
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
                let generated = gemini_generated_video(&job.raw).cloned().unwrap_or_else(|| json!({}));
                let uri = generated
                    .get("uri")
                    .and_then(Value::as_str)
                    .ok_or_else(|| Error::Api("Gemini returned no video".into(), None))?;
                // The file URI redirects to the download host; both hops carry the API key.
                let (data, _) = fetch(&job.connection, uri).await?;
                Ok(Video {
                    url: None,
                    data: Some(data),
                    mime_type: Some(generated.get("mimeType").and_then(Value::as_str).unwrap_or("video/mp4").to_string()),
                    model: Some(job.model.clone()),
                    duration: None,
                    raw: job.raw.clone(),
                    config: None,
                })
            }
        }
    }
}

/// An authenticated GET for a binary body, returning the bytes and the response content type.
async fn fetch(connection: &Connection, path: &str) -> Result<(Vec<u8>, Option<String>)> {
    let resp = connection.send(reqwest::Method::GET, path, &[], true, &|req| req).await?;
    let content_type = resp.headers().get("content-type").and_then(|v| v.to_str().ok()).map(str::to_string);
    let bytes = resp.bytes().await.map_err(|e| Error::ConnectionFailed(e.to_string()))?;
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
    body.pointer("/response/generateVideoResponse/generatedSamples/0/video").filter(|v| !v.is_null())
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
        return Err(Error::UnsupportedAttachment(unsupported(&attachment.mime_type)));
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
            .render(Some("make the water crash down"), "grok-imagine-video", &[], &json!({ "duration": 1, "resolution": "480p" }))
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
        let (path, payload) = Family::XAI.render(Some("Turn the background blue"), "grok-imagine-video", &[video], &Value::Null).unwrap();
        assert_eq!(path, "videos/edits");
        assert_eq!(payload["video"], json!({ "url": "data:video/mp4;base64,bXA0IGJ5dGVz" }));
    }

    #[test]
    fn gemini_reports_filtered_reasons_as_the_failure() {
        let body = json!({ "done": true, "response": { "generateVideoResponse": { "raiMediaFilteredReasons": ["unsafe", "content"] } } });
        assert_eq!(Family::Gemini.parse_status(&body), (VideoStatus::Failed, Some("unsafe content".into())));
        assert_eq!(Family::Gemini.parse_status(&json!({ "done": true })), (VideoStatus::Failed, Some("Gemini returned no video".into())));
        assert_eq!(Family::Gemini.parse_status(&json!({ "name": "op" })).0, VideoStatus::Pending);
    }

    #[test]
    fn xai_expired_jobs_fail_with_their_status() {
        assert_eq!(Family::XAI.parse_status(&json!({ "status": "expired" })), (VideoStatus::Failed, Some("expired".into())));
    }
}
