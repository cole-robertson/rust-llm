//! Port of `lib/ruby_llm/uploaded_file.rb`, `downloaded_file.rb`, `protocols/files.rb` with
//! `protocols/{openai,anthropic,gemini,mistral,xai,deepseek,openrouter,perplexity}/files.rb`, and
//! the auto-upload half of `protocol.rb` (`preprocess_message`, `preprocess_attachment`,
//! `provider_upload`) with each protocol's thresholds from its `chat.rb`.
//!
//! ```ruby
//! file = RubyLLM.upload("document.pdf", provider: :openai, purpose: "user_data")
//! RubyLLM.chat(model: "gpt-5-nano").ask("Summarize this", with: file)
//! ```
//!
//! ```no_run
//! # async fn run() -> rust_llm::Result<()> {
//! use rust_llm::files::{UploadOptions, upload};
//! let file = upload("document.pdf", UploadOptions { provider: Some("openai"), purpose: Some("user_data"), ..Default::default() }).await?;
//! rust_llm::chat_with("gpt-5-nano")?.ask_with("Summarize this", vec![file.into()]).await?;
//! # Ok(()) }
//! ```

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde_json::{Map, Value, json};

use crate::attachment::{Attachment, AttachmentType};
use crate::chat::resolve_model;
use crate::config::Config;
use crate::error::{Error, Result};
use crate::message::{Message, Role};
use crate::providers::{ProtocolName, Provider};
use crate::transport::Connection;

/// `Anthropic::Files::BETA_HEADER`.
pub const ANTHROPIC_FILES_BETA: &str = "files-api-2025-04-14";

/// `OpenAI::Files::UPLOAD_PURPOSES`.
const OPENAI_UPLOAD_PURPOSES: &[&str] = &[
    "assistants",
    "batch",
    "fine-tune",
    "vision",
    "user_data",
    "evals",
];
/// `DeepSeek::Files::IMAGE_TYPES` and `MAX_FILE_SIZE`.
const DEEPSEEK_IMAGE_TYPES: &[&str] = &["image/jpeg", "image/png", "image/gif", "image/webp"];
const DEEPSEEK_MAX_FILE_SIZE: usize = 64 * 1024 * 1024;
/// `Gemini::Files::PROCESSING_POLL_INTERVAL` and `PROCESSING_TIMEOUT`, in seconds.
const GEMINI_POLL_INTERVAL: u64 = 2;
const GEMINI_PROCESSING_TIMEOUT: i64 = 600;
/// `UploadedFile::EXPIRY_MARGIN`, in seconds.
const EXPIRY_MARGIN: i64 = 60;

/// `RubyLLM::UploadedFile`: the metadata record for a file stored with a provider through its
/// Files API. File ids are provider-owned: persist `provider` alongside `id`.
#[derive(Debug, Clone, PartialEq)]
pub struct UploadedFile {
    pub id: String,
    pub provider: String,
    pub filename: Option<String>,
    pub byte_size: Option<u64>,
    pub created_at: Option<DateTime<Utc>>,
    pub expires_at: Option<DateTime<Utc>>,
    pub status: Option<String>,
    pub mime_type: Option<String>,
    pub purpose: Option<String>,
    pub uri: Option<String>,
    pub downloadable: Option<bool>,
    /// The raw provider response data for the file.
    pub metadata: Value,
}

/// `RubyLLM::DownloadedFile`: a provider file's bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DownloadedFile(pub Vec<u8>);

impl DownloadedFile {
    pub fn to_blob(&self) -> &[u8] {
        &self.0
    }

    /// Writes the bytes to `path` (a leading `~/` expands to `$HOME`, like `File.expand_path`)
    /// and returns `path`.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<PathBuf> {
        let path = path.as_ref();
        let expanded = match (path.strip_prefix("~"), std::env::var_os("HOME")) {
            (Ok(rest), Some(home)) => PathBuf::from(home).join(rest),
            _ => path.to_path_buf(),
        };
        std::fs::write(&expanded, &self.0)?;
        Ok(path.to_path_buf())
    }
}

impl std::ops::Deref for DownloadedFile {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        &self.0
    }
}

/// `UploadedFile.upload(file, provider:, context:, filename:, purpose:, expires_in:, provider_options:)`.
/// `provider_options` carries `visibility` (Mistral) and `display_name` (Gemini).
#[derive(Default, Clone)]
pub struct UploadOptions<'a> {
    pub provider: Option<&'a str>,
    pub filename: Option<&'a str>,
    pub purpose: Option<&'a str>,
    /// Seconds until the provider deletes the file (OpenAI, xAI, Mistral rounded up to hours).
    pub expires_in: Option<u64>,
    pub provider_options: Map<String, Value>,
    /// `context:`: use this configuration instead of the global one.
    pub config: Option<Arc<Config>>,
}

/// `provider:` and `context:` for `find` and `download`.
#[derive(Default, Clone)]
pub struct FileOptions<'a> {
    pub provider: Option<&'a str>,
    pub config: Option<Arc<Config>>,
}

/// `RubyLLM.upload`.
pub async fn upload(
    file: impl Into<Attachment>,
    options: UploadOptions<'_>,
) -> Result<UploadedFile> {
    UploadedFile::upload(file, options).await
}

/// `RubyLLM.download`.
pub async fn download(id: &str, options: FileOptions<'_>) -> Result<DownloadedFile> {
    UploadedFile::download(id, options).await
}

impl UploadedFile {
    /// `expired?`: the retention window has passed or ends within the next minute.
    pub fn is_expired(&self) -> bool {
        self.expires_at
            .is_some_and(|t| t <= Utc::now() + chrono::Duration::seconds(EXPIRY_MARGIN))
    }

    /// `UploadedFile.upload`. Without `provider`, the default model's provider is used.
    pub async fn upload(
        file: impl Into<Attachment>,
        options: UploadOptions<'_>,
    ) -> Result<UploadedFile> {
        let config = options.config.clone().unwrap_or_else(crate::config);
        let provider = provider_for(options.provider, &config)?;
        let connection = Connection::new(provider, config)?;
        upload_file(&connection, provider, file.into(), &options).await
    }

    /// `UploadedFile.find`.
    pub async fn find(id: &str, options: FileOptions<'_>) -> Result<UploadedFile> {
        let config = options.config.clone().unwrap_or_else(crate::config);
        let provider = provider_for(options.provider, &config)?;
        find_file(&Connection::new(provider, config)?, provider, id).await
    }

    /// `UploadedFile.download`.
    pub async fn download(id: &str, options: FileOptions<'_>) -> Result<DownloadedFile> {
        let config = options.config.clone().unwrap_or_else(crate::config);
        let provider = provider_for(options.provider, &config)?;
        download_file(&Connection::new(provider, config)?, provider, id)
            .await
            .map(DownloadedFile)
    }
}

/// `UploadedFile.provider_for` plus `Provider#initialize`'s `ensure_configured!`.
fn provider_for(provider: Option<&str>, config: &Config) -> Result<Provider> {
    let provider = match provider {
        Some(slug) => Provider::resolve_or_err(slug)?,
        None => resolve_model(&config.default_model, None, false)?.1,
    };
    provider.ensure_configured(config)?;
    Ok(provider)
}

/// `Provider#files?`: providers that register a `:files` protocol.
pub fn supports_files(provider: Provider) -> bool {
    matches!(
        provider,
        Provider::OpenAI
            | Provider::Anthropic
            | Provider::Gemini
            | Provider::Mistral
            | Provider::XAI
            | Provider::DeepSeek
            | Provider::OpenRouter
            | Provider::Perplexity
    )
}

fn ensure_files_supported(provider: Provider) -> Result<()> {
    if supports_files(provider) {
        return Ok(());
    }
    Err(Error::Api(
        format!("{} doesn't support file uploads", provider.slug()),
        None,
    ))
}

fn files_url(provider: Provider) -> &'static str {
    if provider == Provider::Anthropic {
        "v1/files"
    } else {
        "files"
    }
}

/// `file_headers` / `upload_headers`: only Anthropic's Files API is a beta.
fn file_headers(provider: Provider) -> Vec<(String, String)> {
    if provider == Provider::Anthropic {
        vec![("anthropic-beta".into(), ANTHROPIC_FILES_BETA.into())]
    } else {
        Vec::new()
    }
}

/// `Files#file_content_type`.
fn file_content_type(attachment: &Attachment) -> String {
    let jsonl = attachment
        .filename
        .as_deref()
        .and_then(|f| Path::new(f).extension())
        .is_some_and(|e| e.eq_ignore_ascii_case("jsonl"));
    if jsonl {
        "application/jsonl".into()
    } else {
        attachment.mime_type.clone()
    }
}

/// `Provider#upload_file` → `Files#upload`.
pub(crate) async fn upload_file(
    connection: &Connection,
    provider: Provider,
    file: Attachment,
    options: &UploadOptions<'_>,
) -> Result<UploadedFile> {
    ensure_files_supported(provider)?;
    if provider == Provider::Perplexity {
        return Err(Error::Api(
            "Perplexity Agent only supports downloading generated response files".into(),
            None,
        ));
    }
    // `file_attachment(file, filename:)`
    let mut attachment = match options.filename {
        Some(name) => file.with_filename(name),
        None => file,
    };
    attachment.load(connection.client()).await?;
    let mut visibility = None;
    let mut display_name = None;
    for (key, value) in &options.provider_options {
        match key.as_str() {
            "visibility" => visibility = value.as_str().map(str::to_string),
            "display_name" => display_name = value.as_str().map(str::to_string),
            // Gemini's `upload` reads only `display_name`; the others splat the rest as keywords.
            _ if provider == Provider::Gemini => {}
            other => return Err(Error::Argument(format!("unknown keyword: :{other}"))),
        }
    }
    if provider == Provider::Gemini {
        return gemini_upload(connection, &attachment, display_name).await;
    }

    let fields = upload_fields(provider, &attachment, options, visibility)?;
    let bytes = attachment.bytes()?.to_vec();
    let filename = attachment.filename.clone().unwrap_or_default();
    let content_type = file_content_type(&attachment);
    reqwest::multipart::Part::bytes(Vec::new())
        .mime_str(&content_type)
        .map_err(|e| Error::Argument(format!("invalid content type {content_type:?}: {e}")))?;
    let form = || {
        let part = reqwest::multipart::Part::bytes(bytes.clone())
            .file_name(filename.clone())
            .mime_str(&content_type)
            .expect("content type validated above");
        fields.iter().fold(
            reqwest::multipart::Form::new().part("file", part),
            |form, (k, v)| form.text(k.clone(), v.clone()),
        )
    };
    let raw = connection
        .post_multipart(files_url(provider), form, &file_headers(provider), false)
        .await?;
    parse_file_response(provider, &raw.body)
}

/// Each provider's `render_upload_payload` after the `file` part, as flat multipart fields
/// (Faraday encodes a nested hash as `key[sub]`).
fn upload_fields(
    provider: Provider,
    attachment: &Attachment,
    options: &UploadOptions<'_>,
    visibility: Option<String>,
) -> Result<Vec<(String, String)>> {
    let mut fields = Vec::new();
    let mut push = |k: &str, v: Option<String>| {
        if let Some(v) = v {
            fields.push((k.to_string(), v));
        }
    };
    let purpose = options.purpose.map(str::to_string);
    match provider {
        Provider::OpenAI | Provider::DeepSeek => {
            let purpose = if provider == Provider::DeepSeek {
                if !DEEPSEEK_IMAGE_TYPES.contains(&attachment.mime_type.as_str()) {
                    return Err(Error::UnsupportedAttachment(
                        crate::protocols::anthropic::unsupported(&attachment.mime_type),
                    ));
                }
                if attachment.bytes()?.len() > DEEPSEEK_MAX_FILE_SIZE {
                    return Err(Error::Argument(
                        "DeepSeek image uploads cannot exceed 64 MiB".into(),
                    ));
                }
                if purpose.as_deref().is_some_and(|p| p != "user_data") {
                    return Err(Error::Argument(
                        "DeepSeek file uploads require purpose: user_data".into(),
                    ));
                }
                Some("user_data".to_string())
            } else {
                purpose
            };
            let Some(purpose) = purpose else {
                return Err(Error::Argument(format!(
                    "{} file uploads require purpose: {}",
                    provider.display(),
                    OPENAI_UPLOAD_PURPOSES.join(", ")
                )));
            };
            push("purpose", Some(purpose));
            if let Some(seconds) = options.expires_in {
                push("expires_after[anchor]", Some("created_at".into()));
                push("expires_after[seconds]", Some(seconds.to_string()));
            }
        }
        Provider::Mistral => {
            push("purpose", purpose);
            push(
                "expiry",
                options.expires_in.map(|s| s.div_ceil(3600).to_string()),
            );
            push("visibility", visibility);
        }
        Provider::XAI => {
            push("expires_after", options.expires_in.map(|s| s.to_string()));
            push("purpose", purpose);
        }
        _ => {}
    }
    Ok(fields)
}

/// `Gemini::Files#upload`: a resumable upload in two requests, then waiting for processing.
async fn gemini_upload(
    connection: &Connection,
    attachment: &Attachment,
    display_name: Option<String>,
) -> Result<UploadedFile> {
    let provider = Provider::Gemini;
    let bytes = attachment.bytes()?.to_vec();
    let display_name = display_name.or_else(|| attachment.filename.clone());
    let base = provider.api_base(connection.config())?;
    let base = base.trim_end_matches('/');
    let upload_base = if let Some(b) = base.strip_suffix("/v1beta") {
        format!("{b}/upload/v1beta")
    } else if let Some(b) = base.strip_suffix("/v1") {
        format!("{b}/upload/v1")
    } else {
        base.to_string()
    };
    let start_headers = vec![
        (
            "X-Goog-Upload-Protocol".to_string(),
            "resumable".to_string(),
        ),
        ("X-Goog-Upload-Command".to_string(), "start".to_string()),
        (
            "X-Goog-Upload-Header-Content-Length".to_string(),
            bytes.len().to_string(),
        ),
        (
            "X-Goog-Upload-Header-Content-Type".to_string(),
            file_content_type(attachment),
        ),
    ];
    let body = json!({ "file": { "display_name": display_name } });
    // `gemini_connection` is a basic Faraday connection: no retries.
    let resp = connection
        .send(
            reqwest::Method::POST,
            &format!("{upload_base}/files"),
            &start_headers,
            false,
            &|req| req.json(&body),
        )
        .await?;
    let upload_url = resp
        .headers()
        .get("x-goog-upload-url")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string)
        .ok_or_else(|| Error::Api("gemini did not return an upload URL".into(), None))?;

    let upload_headers = vec![
        ("X-Goog-Upload-Offset".to_string(), "0".to_string()),
        (
            "X-Goog-Upload-Command".to_string(),
            "upload, finalize".to_string(),
        ),
    ];
    let resp = connection
        .send(
            reqwest::Method::POST,
            &upload_url,
            &upload_headers,
            false,
            &|req| req.body(bytes.clone()),
        )
        .await?;
    let raw = crate::transport::json_response(resp, Value::Null).await?;
    let data = raw
        .body
        .get("file")
        .ok_or_else(|| Error::Api("gemini upload response has no file".into(), None))?;
    let mut file = parse_file_response(provider, data)?;

    // `await_active`: Gemini rejects a file reference until processing finishes.
    let deadline = Utc::now() + chrono::Duration::seconds(GEMINI_PROCESSING_TIMEOUT);
    while file.status.as_deref() == Some("PROCESSING") {
        if Utc::now() >= deadline {
            return Err(Error::Api(
                format!("gemini is still processing {}", file.id),
                None,
            ));
        }
        tokio::time::sleep(Duration::from_secs(GEMINI_POLL_INTERVAL)).await;
        file = find_file(connection, provider, &file.id).await?;
    }
    if file.status.as_deref() == Some("FAILED") {
        return Err(Error::Api(
            format!("gemini failed to process {}", file.id),
            None,
        ));
    }
    Ok(file)
}

fn gemini_file_name(id: &str) -> String {
    if id.starts_with("files/") {
        id.to_string()
    } else {
        format!("files/{id}")
    }
}

/// `Perplexity::Files#split_resource_id`.
fn split_perplexity_id(id: &str) -> Result<(String, String)> {
    let re = regex::Regex::new(r"\A([A-Za-z0-9_-]+)/files/([A-Za-z0-9_-]+)\z")
        .map_err(|e| Error::Argument(e.to_string()))?;
    let caps = re.captures(id).ok_or_else(|| {
        Error::Argument(
            "Perplexity file IDs must include their response: response_id/files/file_id".into(),
        )
    })?;
    Ok((caps[1].to_string(), caps[2].to_string()))
}

/// `Provider#find_file` → `Files#find`.
pub(crate) async fn find_file(
    connection: &Connection,
    provider: Provider,
    id: &str,
) -> Result<UploadedFile> {
    ensure_files_supported(provider)?;
    if provider == Provider::Perplexity {
        let (response_id, file_id) = split_perplexity_id(id)?;
        let raw = connection
            .get(&format!("v1/agent/{response_id}/files"), &[])
            .await?;
        let data = raw
            .body
            .get("data")
            .and_then(Value::as_array)
            .and_then(|items| {
                items
                    .iter()
                    .find(|i| i.get("id").and_then(Value::as_str) == Some(file_id.as_str()))
            })
            .ok_or_else(|| Error::Api("Perplexity response file was not found".into(), None))?;
        return parse_perplexity_file(&response_id, data);
    }
    let path = if provider == Provider::Gemini {
        gemini_file_name(id)
    } else {
        format!("{}/{id}", files_url(provider))
    };
    let raw = connection.get(&path, &file_headers(provider)).await?;
    parse_file_response(provider, &raw.body)
}

/// `Provider#download_file` → `Files#download`.
pub(crate) async fn download_file(
    connection: &Connection,
    provider: Provider,
    id: &str,
) -> Result<Vec<u8>> {
    ensure_files_supported(provider)?;
    match provider {
        Provider::DeepSeek => Err(Error::Api(
            "DeepSeek does not support downloading uploaded files".into(),
            None,
        )),
        Provider::Gemini => {
            let file = find_file(connection, provider, id).await?;
            let uri = file
                .metadata
                .get("downloadUri")
                .and_then(Value::as_str)
                .ok_or_else(|| Error::Api("gemini file has no download URI".into(), None))?;
            let resp = connection
                .send(reqwest::Method::GET, uri, &[], false, &|req| req)
                .await?;
            Ok(resp
                .bytes()
                .await
                .map_err(|e| Error::ConnectionFailed(e.to_string()))?
                .to_vec())
        }
        _ => {
            let path = if provider == Provider::Perplexity {
                split_perplexity_id(id)?;
                format!("v1/agent/{id}/content")
            } else {
                format!("{}/{id}/content", files_url(provider))
            };
            let mut headers = vec![("Accept".to_string(), "application/octet-stream".to_string())];
            headers.extend(file_headers(provider));
            connection.get_bytes(&path, &headers).await
        }
    }
}

/// `Files#timestamp`: epoch seconds (number or digit string) or ISO 8601.
fn timestamp(value: Option<&Value>) -> Result<Option<DateTime<Utc>>> {
    let secs = match value {
        None | Some(Value::Null) => return Ok(None),
        Some(Value::Number(n)) => n.as_i64(),
        Some(Value::String(s)) if !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()) => {
            s.parse().ok()
        }
        Some(Value::String(s)) => {
            return DateTime::parse_from_rfc3339(s)
                .map(|t| Some(t.with_timezone(&Utc)))
                .map_err(|e| Error::Argument(format!("invalid date: {s:?} ({e})")));
        }
        Some(other) => return Err(Error::Argument(format!("invalid date: {other}"))),
    };
    Ok(secs.and_then(|s| DateTime::from_timestamp(s, 0)))
}

fn string(data: &Value, key: &str) -> Option<String> {
    data.get(key).and_then(Value::as_str).map(str::to_string)
}

fn size(data: &Value, key: &str) -> Option<u64> {
    let v = data.get(key)?;
    v.as_u64()
        .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
}

/// Each provider's `parse_file_response`.
fn parse_file_response(provider: Provider, data: &Value) -> Result<UploadedFile> {
    let mut file = UploadedFile {
        id: String::new(),
        provider: provider.slug().to_string(),
        filename: string(data, "filename"),
        byte_size: None,
        created_at: timestamp(data.get("created_at"))?,
        expires_at: None,
        status: None,
        mime_type: None,
        purpose: None,
        uri: None,
        downloadable: None,
        metadata: data.clone(),
    };
    match provider {
        Provider::Anthropic | Provider::OpenRouter => {
            file.id = string(data, "id").unwrap_or_default();
            file.byte_size = size(data, "size_bytes");
            file.mime_type = string(data, "mime_type");
            file.downloadable = data.get("downloadable").and_then(Value::as_bool);
        }
        Provider::Gemini => {
            file.id = string(data, "name").unwrap_or_default();
            file.filename = string(data, "displayName");
            file.byte_size = size(data, "sizeBytes");
            file.created_at = timestamp(data.get("createTime"))?;
            file.expires_at = timestamp(data.get("expirationTime"))?;
            file.mime_type = string(data, "mimeType");
            file.status = string(data, "state");
            file.uri = string(data, "uri");
        }
        Provider::Mistral => {
            file.id = string(data, "id").unwrap_or_default();
            file.byte_size = size(data, "bytes");
            file.expires_at = timestamp(data.get("expires_at"))?;
            file.mime_type = string(data, "mimetype");
            file.purpose = string(data, "purpose");
            file.status = data
                .get("deleted")
                .and_then(Value::as_bool)
                .filter(|d| *d)
                .map(|_| "deleted".to_string());
        }
        // OpenAI, DeepSeek (an OpenAI subclass), and xAI.
        _ => {
            file.id = string(data, "id").unwrap_or_default();
            file.byte_size = size(data, "bytes");
            file.expires_at = timestamp(data.get("expires_at"))?;
            file.purpose = string(data, "purpose");
            if provider != Provider::XAI {
                file.status = string(data, "status");
            }
            if provider == Provider::DeepSeek {
                file.downloadable = Some(false);
            }
        }
    }
    Ok(file)
}

/// `Perplexity::Files#parse_response_file`.
fn parse_perplexity_file(response_id: &str, data: &Value) -> Result<UploadedFile> {
    let file_id = string(data, "file_id")
        .or_else(|| string(data, "id"))
        .unwrap_or_default();
    let id = format!("{response_id}/files/{file_id}");
    split_perplexity_id(&id)?;
    let mut metadata = data.clone();
    if let Some(m) = metadata.as_object_mut() {
        m.insert("response_id".into(), response_id.into());
    }
    Ok(UploadedFile {
        id,
        provider: Provider::Perplexity.slug().to_string(),
        filename: string(data, "filename"),
        byte_size: size(data, "bytes").or_else(|| size(data, "size_bytes")),
        created_at: timestamp(data.get("created_at"))?,
        expires_at: None,
        status: None,
        mime_type: string(data, "content_type").or_else(|| string(data, "mime_type")),
        purpose: None,
        uri: None,
        downloadable: Some(true),
        metadata,
    })
}

/// Uploads memoized on an attachment, keyed by provider and credentials (`provider_uploads`).
/// Clones share the memo, so each request's copy of history reuses an earlier upload.
#[derive(Debug, Clone, Default)]
pub struct ProviderUploads(Arc<Mutex<HashMap<String, UploadedFile>>>);

impl PartialEq for ProviderUploads {
    /// A memo of uploads is not part of an attachment's identity.
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

/// The per-protocol auto-upload settings from each `chat.rb`.
struct AutoUpload {
    threshold: u64,
    limit: u64,
    attachable: fn(&Attachment) -> bool,
    purpose: Option<&'static str>,
}

const MB: u64 = 1024 * 1024;

/// `supports_provider_file_references?`, `default_large_file_upload_threshold`,
/// `provider_file_upload_limit`, `provider_file_attachable?`, `provider_file_upload_options`.
fn auto_upload_rules(protocol: ProtocolName, provider: Provider) -> Option<AutoUpload> {
    use AttachmentType as T;
    match protocol {
        // ANTHROPIC_INLINE_REQUEST_LIMIT, ANTHROPIC_FILE_UPLOAD_LIMIT
        ProtocolName::Anthropic => Some(AutoUpload {
            threshold: 24 * MB,
            limit: 500 * MB,
            attachable: |a| matches!(a.kind(), T::Image | T::Pdf | T::Text),
            purpose: None,
        }),
        // GEMINI_INLINE_FILE_THRESHOLD, GEMINI_FILE_UPLOAD_LIMIT
        ProtocolName::Gemini => Some(AutoUpload {
            threshold: 20 * MB,
            limit: 2 * 1024 * MB,
            attachable: |a| matches!(a.kind(), T::Image | T::Video | T::Audio | T::Pdf | T::Text),
            purpose: None,
        }),
        // Perplexity's Agent API takes no file references.
        ProtocolName::Responses if provider == Provider::Perplexity => None,
        // OPENAI_INLINE_FILE_LIMIT, OPENAI_FILE_UPLOAD_LIMIT
        ProtocolName::Responses => Some(AutoUpload {
            threshold: 50 * MB,
            limit: 512 * MB,
            attachable: |a| matches!(a.kind(), T::Pdf | T::Document),
            purpose: Some("user_data"),
        }),
        // OPENROUTER_INLINE_FILE_THRESHOLD, OPENROUTER_FILE_UPLOAD_LIMIT
        ProtocolName::ChatCompletions if provider == Provider::OpenRouter => Some(AutoUpload {
            threshold: 50 * MB,
            limit: 100 * MB,
            attachable: |a| a.kind() == T::Pdf,
            purpose: None,
        }),
        ProtocolName::ChatCompletions if provider == Provider::OpenAI => Some(AutoUpload {
            threshold: 50 * MB,
            limit: 512 * MB,
            attachable: |a| a.kind() == T::Pdf,
            purpose: Some("user_data"),
        }),
        ProtocolName::ChatCompletions => None,
        // `Protocol#supports_provider_file_references?` is false for these.
        ProtocolName::Interactions
        | ProtocolName::Conversations
        | ProtocolName::RouterChatCompletions => None,
    }
}

/// `provider_upload_scope`: the provider plus a hash of the credentials in play, so a chat moved
/// to another account uploads again. The credentials are hashed, never stored on the attachment.
fn provider_upload_scope(provider: Provider, config: &Config) -> String {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    provider.api_base(config).ok().hash(&mut hasher);
    provider.headers(config).hash(&mut hasher);
    format!("{}:{:x}", provider.slug(), hasher.finish())
}

fn format_bytes(bytes: Option<u64>) -> String {
    match bytes {
        Some(b) => format!("{:.1} MB", ((b as f64 / MB as f64) * 10.0).round() / 10.0),
        None => "unknown size".into(),
    }
}

/// `Protocol#preprocess_message` (the upload half; foreign thinking is dropped in `Chat`): a user
/// message's local attachments over the protocol's inline threshold are uploaded to the
/// provider's Files API and sent as file references. History keeps the original attachments.
pub(crate) async fn preprocess_messages(
    messages: &mut [Message],
    protocol: ProtocolName,
    provider: Provider,
    config: &Config,
    connection: &Connection,
) -> Result<()> {
    if !config.auto_upload_large_files || !supports_files(provider) {
        return Ok(());
    }
    let Some(rules) = auto_upload_rules(protocol, provider) else {
        return Ok(());
    };
    for message in messages.iter_mut().filter(|m| m.role == Role::User) {
        for attachment in message.attachments.iter_mut() {
            if attachment.is_provider_file() {
                continue;
            }
            let size = attachment.byte_size();
            if !size.is_some_and(|s| s > rules.threshold) || !(rules.attachable)(attachment) {
                continue;
            }
            if size.is_some_and(|s| s > rules.limit) {
                return Err(Error::Api(
                    format!(
                        "{} file uploads support files up to {}; {} is {}",
                        provider.display(),
                        format_bytes(Some(rules.limit)),
                        attachment.filename.as_deref().unwrap_or(""),
                        format_bytes(size)
                    ),
                    None,
                ));
            }
            let uploaded =
                provider_upload(attachment, &rules, provider, config, connection).await?;
            let resolution = attachment.resolution;
            let mut replacement = Attachment::from_uploaded(uploaded);
            replacement.resolution = resolution;
            *attachment = replacement;
        }
    }
    Ok(())
}

/// `provider_upload`: once per provider account, replaced when past its retention window.
async fn provider_upload(
    attachment: &Attachment,
    rules: &AutoUpload,
    provider: Provider,
    config: &Config,
    connection: &Connection,
) -> Result<UploadedFile> {
    let scope = provider_upload_scope(provider, config);
    let memo = attachment.provider_uploads().clone();
    let existing = memo
        .0
        .lock()
        .map_err(|_| Error::Api("provider upload memo poisoned".into(), None))?
        .get(&scope)
        .cloned();
    if let Some(upload) = existing.filter(|u| !u.is_expired()) {
        return Ok(upload);
    }
    let options = UploadOptions {
        purpose: rules.purpose,
        ..Default::default()
    };
    let uploaded = upload_file(connection, provider, attachment.clone(), &options).await?;
    memo.0
        .lock()
        .map_err(|_| Error::Api("provider upload memo poisoned".into(), None))?
        .insert(scope, uploaded.clone());
    Ok(uploaded)
}

/// `Anthropic#apply_files_beta`: a request that references an uploaded file (a block whose
/// `source.type` is `file`) carries the Files API beta, joined to any beta already set.
pub(crate) fn apply_files_beta(
    protocol: ProtocolName,
    payload: &Value,
    headers: &mut Vec<(String, String)>,
) {
    if protocol != ProtocolName::Anthropic {
        return;
    }
    let is_file = |b: &Value| b.pointer("/source/type").and_then(Value::as_str) == Some("file");
    let messages = payload
        .get("messages")
        .and_then(Value::as_array)
        .into_iter()
        .flatten();
    let mut blocks = messages
        .filter_map(|m| m.get("content").and_then(Value::as_array))
        .flatten();
    let system = payload
        .get("system")
        .and_then(Value::as_array)
        .into_iter()
        .flatten();
    if !blocks.any(is_file) && !system.clone().any(is_file) {
        return;
    }
    // `join_betas`: one comma-separated header, existing betas first, no duplicates.
    let mut betas: Vec<String> = Vec::new();
    headers.retain(|(k, v)| {
        if !k.eq_ignore_ascii_case("anthropic-beta") {
            return true;
        }
        betas.extend(
            v.split(',')
                .map(str::trim)
                .filter(|b| !b.is_empty())
                .map(str::to_string),
        );
        false
    });
    betas.push(ANTHROPIC_FILES_BETA.to_string());
    let mut unique: Vec<String> = Vec::new();
    for b in betas {
        if !unique.contains(&b) {
            unique.push(b);
        }
    }
    headers.push(("anthropic-beta".into(), unique.join(",")));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestamps_parse_epoch_digit_strings_and_iso8601() {
        assert_eq!(
            timestamp(Some(&json!(1789683302)))
                .unwrap()
                .unwrap()
                .timestamp(),
            1789683302
        );
        assert_eq!(
            timestamp(Some(&json!("1789683302")))
                .unwrap()
                .unwrap()
                .timestamp(),
            1789683302
        );
        let t = timestamp(Some(&json!("2026-09-19T22:15:08.647110897Z")))
            .unwrap()
            .unwrap();
        assert_eq!(
            t.to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
            "2026-09-19T22:15:08Z"
        );
        assert!(timestamp(Some(&Value::Null)).unwrap().is_none());
        assert!(timestamp(Some(&json!("not a date"))).is_err());
    }

    #[test]
    fn a_file_expiring_within_a_minute_counts_as_expired() {
        let mut file = parse_file_response(Provider::OpenAI, &json!({ "id": "file-1" })).unwrap();
        assert!(!file.is_expired(), "no expiry never expires");
        file.expires_at = Some(Utc::now() + chrono::Duration::seconds(30));
        assert!(file.is_expired());
        file.expires_at = Some(Utc::now() + chrono::Duration::seconds(3600));
        assert!(!file.is_expired());
    }

    #[test]
    fn mistral_rounds_expiry_up_to_whole_hours() {
        let a = Attachment::from_bytes(b"x".to_vec(), "a.pdf", None);
        let options = UploadOptions {
            purpose: Some("ocr"),
            expires_in: Some(3601),
            ..Default::default()
        };
        let fields = upload_fields(Provider::Mistral, &a, &options, Some("user".into())).unwrap();
        assert_eq!(
            fields,
            vec![
                ("purpose".into(), "ocr".into()),
                ("expiry".into(), "2".into()),
                ("visibility".into(), "user".into())
            ]
        );
    }

    #[test]
    fn openai_requires_a_purpose_and_nests_expires_after() {
        let a = Attachment::from_bytes(b"{}".to_vec(), "batch.jsonl", None);
        let err = upload_fields(Provider::OpenAI, &a, &UploadOptions::default(), None).unwrap_err();
        assert_eq!(
            err.to_string(),
            "OpenAI file uploads require purpose: assistants, batch, fine-tune, vision, user_data, evals"
        );
        let options = UploadOptions {
            purpose: Some("batch"),
            expires_in: Some(86400),
            ..Default::default()
        };
        let fields = upload_fields(Provider::OpenAI, &a, &options, None).unwrap();
        assert_eq!(
            fields[1],
            ("expires_after[anchor]".into(), "created_at".into())
        );
        assert_eq!(fields[2], ("expires_after[seconds]".into(), "86400".into()));
        assert_eq!(file_content_type(&a), "application/jsonl");
    }

    #[test]
    fn deepseek_only_uploads_images() {
        let pdf = Attachment::from_bytes(b"%PDF".to_vec(), "a.pdf", None);
        assert!(matches!(
            upload_fields(Provider::DeepSeek, &pdf, &UploadOptions::default(), None),
            Err(Error::UnsupportedAttachment(_))
        ));
        let png = Attachment::from_bytes(b"png".to_vec(), "a.png", None);
        let fields =
            upload_fields(Provider::DeepSeek, &png, &UploadOptions::default(), None).unwrap();
        assert_eq!(fields, vec![("purpose".into(), "user_data".into())]);
    }

    #[test]
    fn anthropic_file_references_join_the_files_beta() {
        let payload = json!({ "messages": [{ "role": "user", "content": [
            { "type": "document", "source": { "type": "file", "file_id": "file_1" } }
        ]}]});
        let mut headers = vec![(
            "anthropic-beta".to_string(),
            "compact-2026-01-12".to_string(),
        )];
        apply_files_beta(ProtocolName::Anthropic, &payload, &mut headers);
        assert_eq!(
            headers,
            vec![(
                "anthropic-beta".to_string(),
                format!("compact-2026-01-12,{ANTHROPIC_FILES_BETA}")
            )]
        );

        let inline = json!({ "messages": [{ "role": "user", "content": [{ "type": "text", "text": "hi" }] }] });
        let mut headers = Vec::new();
        apply_files_beta(ProtocolName::Anthropic, &inline, &mut headers);
        assert!(headers.is_empty(), "no file reference, no beta");
    }

    #[test]
    fn perplexity_ids_must_name_their_response() {
        assert_eq!(
            split_perplexity_id("resp_1/files/f_2").unwrap(),
            ("resp_1".into(), "f_2".into())
        );
        assert!(split_perplexity_id("f_2").is_err());
    }
}
