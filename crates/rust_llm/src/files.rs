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
    let mut attachment = file_attachment(file, options.filename);
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

/// `Files#file_attachment(file, filename:)`: the attachment as given, or rewrapped under `filename`.
fn file_attachment(file: Attachment, filename: Option<&str>) -> Attachment {
    match filename {
        Some(name) => file.with_filename(name),
        None => file,
    }
}

/// `Files#file_size`: a path's size on disk, otherwise the length of the loaded content. 2.1's
/// upload protocols no longer check sizes, so only its spec exercises it.
#[cfg_attr(not(test), allow(dead_code))]
fn file_size(attachment: &Attachment) -> Result<u64> {
    match &attachment.source {
        crate::attachment::Source::Path(path) => Ok(std::fs::metadata(path)?.len()),
        _ => Ok(attachment.bytes()?.len() as u64),
    }
}

/// Each provider's `render_upload_payload` after the `file` part, as flat multipart fields
/// (Faraday encodes a nested hash as `key[sub]`).
fn upload_fields(
    provider: Provider,
    _attachment: &Attachment,
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
            // 2.1 leaves purposes, file types, and sizes to the APIs (`openai/files.rb`,
            // `deepseek/files.rb`); DeepSeek still defaults the purpose to user_data.
            let purpose = if provider == Provider::DeepSeek {
                purpose.or_else(|| Some("user_data".to_string()))
            } else {
                purpose
            };
            push("purpose", purpose);
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
pub struct ProviderUploads(Arc<Mutex<HashMap<String, UploadedFile>>>, StoreSlot);

/// `Attachment#provider_file_store`, kept beside the upload memo so clones share it.
#[derive(Clone, Default)]
struct StoreSlot(Arc<Mutex<Option<Arc<dyn ProviderFileStore>>>>);

impl std::fmt::Debug for StoreSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ProviderFileStore")
    }
}

impl ProviderUploads {
    /// Whether two attachments share this memo, i.e. one is a clone of the other.
    pub(crate) fn same(&self, other: &ProviderUploads) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    pub(crate) fn store(&self) -> Option<Arc<dyn ProviderFileStore>> {
        self.1.0.lock().ok().and_then(|s| s.clone())
    }

    pub(crate) fn set_store(&self, store: Option<Arc<dyn ProviderFileStore>>) {
        if let Ok(mut slot) = self.1.0.lock() {
            *slot = store;
        }
    }

    fn get(&self, scope: &str) -> Option<UploadedFile> {
        self.0.lock().ok().and_then(|m| m.get(scope).cloned())
    }

    fn insert(&self, scope: String, upload: UploadedFile) {
        if let Ok(mut m) = self.0.lock() {
            m.insert(scope, upload);
        }
    }

    fn remove(&self, scope: &str) -> Option<UploadedFile> {
        self.0.lock().ok().and_then(|mut m| m.remove(scope))
    }
}

/// `Attachment#provider_file_store`: where uploads of an attachment outlive the process, keyed by
/// provider slug and [`Provider::account_identity`]. The Rails integration keeps one per file in
/// Active Storage (`active_record/provider_file.rb`).
/// Async because a store is usually a database; Ruby's is synchronous Active Record.
#[async_trait::async_trait]
pub trait ProviderFileStore: Send + Sync {
    /// `fetch(provider:, account:)`.
    async fn fetch(&self, provider: &str, account: &str) -> Option<UploadedFile>;
    /// `store(upload, provider:, account:)`.
    async fn store(&self, upload: &UploadedFile, provider: &str, account: &str);
    /// `forget(id, provider:, account:)`.
    async fn forget(&self, id: &str, provider: &str, account: &str);
}

/// `StoredUploads::FILES`: files a provider confirmed (or that this process uploaded), keyed by
/// provider, account, and id, so each stored upload is asked about once per process. A
/// `ProcessCache` of 1024 entries, oldest dropped first.
type ConfirmedFiles = Mutex<Vec<((String, String, String), UploadedFile)>>;

fn confirmed_files() -> &'static ConfirmedFiles {
    static FILES: std::sync::OnceLock<ConfirmedFiles> = std::sync::OnceLock::new();
    FILES.get_or_init(Default::default)
}

const CONFIRMED_FILES_LIMIT: usize = 1024;

/// `StoredUploads.files.fetch(key)`: whether this process already confirmed the file.
#[doc(hidden)]
pub fn is_confirmed_upload(provider: &str, account: &str, id: &str) -> bool {
    let key = (provider.to_string(), account.to_string(), id.to_string());
    confirmed_files()
        .lock()
        .is_ok_and(|f| f.iter().any(|(k, _)| *k == key))
}

/// `StoredUploads.files.clear`: forgets every confirmation, as a fresh process would.
#[doc(hidden)]
pub fn clear_confirmed_uploads() {
    if let Ok(mut files) = confirmed_files().lock() {
        files.clear();
    }
}

fn confirmed(key: &(String, String, String)) -> Option<UploadedFile> {
    let mut files = confirmed_files().lock().ok()?;
    let at = files.iter().position(|(k, _)| k == key)?;
    // `touch`: the most recently used entry moves to the back.
    let entry = files.remove(at);
    let file = entry.1.clone();
    files.push(entry);
    Some(file)
}

fn confirm(key: (String, String, String), file: &UploadedFile) {
    if let Ok(mut files) = confirmed_files().lock() {
        if files.iter().any(|(k, _)| *k == key) {
            return;
        }
        files.push((key, file.clone()));
        while files.len() > CONFIRMED_FILES_LIMIT {
            files.remove(0);
        }
    }
}

fn unconfirm(key: &(String, String, String)) {
    if let Ok(mut files) = confirmed_files().lock() {
        files.retain(|(k, _)| k != key);
    }
}

/// `Protocol::StoredUploads` (`protocol/stored_uploads.rb`): reuses the provider file an
/// attachment's store recorded in an earlier process. Before a process first reuses a file the
/// provider confirms it still has it; a missing or expired file is uploaded again.
struct StoredUploads<'a> {
    provider: Provider,
    connection: &'a Connection,
    store: Option<Arc<dyn ProviderFileStore>>,
    account: Option<String>,
}

impl<'a> StoredUploads<'a> {
    fn new(
        provider: Provider,
        config: &Config,
        connection: &'a Connection,
        store: Option<Arc<dyn ProviderFileStore>>,
    ) -> StoredUploads<'a> {
        let account = store
            .as_ref()
            .and_then(|_| provider.account_identity(config));
        StoredUploads {
            provider,
            connection,
            store,
            account,
        }
    }

    fn key(&self, account: &str, id: &str) -> (String, String, String) {
        (self.provider.slug().into(), account.into(), id.into())
    }

    /// `fetch { upload }`: the stored file when the provider still has it, else a new upload,
    /// remembered in the store.
    async fn fetch(
        &self,
        upload: impl std::future::Future<Output = Result<UploadedFile>>,
    ) -> Result<UploadedFile> {
        let (Some(store), Some(account)) = (&self.store, &self.account) else {
            return upload.await;
        };
        if let Some(file) = self.stored_file(store.as_ref(), account).await {
            return Ok(file);
        }
        let uploaded = upload.await?;
        confirm(self.key(account, &uploaded.id), &uploaded);
        store.store(&uploaded, self.provider.slug(), account).await;
        Ok(uploaded)
    }

    async fn stored_file(
        &self,
        store: &dyn ProviderFileStore,
        account: &str,
    ) -> Option<UploadedFile> {
        let stored = store.fetch(self.provider.slug(), account).await?;
        if stored.is_expired() {
            return None;
        }
        let key = self.key(account, &stored.id);
        let file = match confirmed(&key) {
            Some(file) => file,
            // `confirmed_file`: a provider that cannot find it means upload again.
            None => {
                let file = find_file(self.connection, self.provider, &stored.id)
                    .await
                    .ok()?;
                confirm(key, &file);
                file
            }
        };
        (!file.is_expired()).then_some(file)
    }

    /// `forget(upload)`: drops the process's confirmation and the stored row.
    async fn forget(&self, upload: &UploadedFile) {
        let (Some(store), Some(account)) = (&self.store, &self.account) else {
            return;
        };
        unconfirm(&self.key(account, &upload.id));
        store
            .forget(&upload.id, self.provider.slug(), account)
            .await;
    }
}

impl PartialEq for ProviderUploads {
    /// A memo of uploads is not part of an attachment's identity.
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

/// The per-protocol auto-upload settings from each `chat.rb`.
struct AutoUpload {
    threshold: u64,
    attachable: fn(&Attachment) -> bool,
    purpose: Option<&'static str>,
}

const MB: u64 = 1024 * 1024;

/// `supports_provider_file_references?`, `default_large_file_upload_threshold`,
/// `provider_file_attachable?`, `provider_file_upload_options`. 2.1 leaves the upload size limit
/// to the provider, whose error states it.
fn auto_upload_rules(protocol: ProtocolName, provider: Provider) -> Option<AutoUpload> {
    use AttachmentType as T;
    match protocol {
        // ANTHROPIC_INLINE_REQUEST_LIMIT, ANTHROPIC_FILE_UPLOAD_LIMIT
        ProtocolName::Anthropic => Some(AutoUpload {
            threshold: 24 * MB,
            attachable: |a| matches!(a.kind(), T::Image | T::Pdf | T::Text),
            purpose: None,
        }),
        // GEMINI_INLINE_FILE_THRESHOLD, GEMINI_FILE_UPLOAD_LIMIT
        ProtocolName::Gemini => Some(AutoUpload {
            threshold: 20 * MB,
            attachable: |a| matches!(a.kind(), T::Image | T::Video | T::Audio | T::Pdf | T::Text),
            purpose: None,
        }),
        // Perplexity's Agent API takes no file references.
        ProtocolName::Responses if provider == Provider::Perplexity => None,
        // OPENAI_INLINE_FILE_LIMIT, OPENAI_FILE_UPLOAD_LIMIT
        ProtocolName::Responses => Some(AutoUpload {
            threshold: 50 * MB,
            attachable: |a| matches!(a.kind(), T::Pdf | T::Document),
            purpose: Some("user_data"),
        }),
        // OPENROUTER_INLINE_FILE_THRESHOLD, OPENROUTER_FILE_UPLOAD_LIMIT
        ProtocolName::ChatCompletions if provider == Provider::OpenRouter => Some(AutoUpload {
            threshold: 50 * MB,
            attachable: |a| a.kind() == T::Pdf,
            purpose: None,
        }),
        ProtocolName::ChatCompletions if provider == Provider::OpenAI => Some(AutoUpload {
            threshold: 50 * MB,
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
    let memo = attachment.provider_uploads();
    if let Some(upload) = memo.get(&scope).filter(|u| !u.is_expired()) {
        return Ok(upload);
    }
    let options = UploadOptions {
        purpose: rules.purpose,
        ..Default::default()
    };
    // `StoredUploads.new(@provider, attachment.provider_file_store).fetch { upload_file }`.
    let uploaded = StoredUploads::new(provider, config, connection, memo.store())
        .fetch(upload_file(
            connection,
            provider,
            attachment.clone(),
            &options,
        ))
        .await?;
    memo.insert(scope, uploaded.clone());
    Ok(uploaded)
}

/// `Protocol#discard_missing_uploads`: a provider can delete a file RubyLLM uploaded for an
/// attachment. When a request fails naming such a file, or with a 404 that names none, the
/// attachments forget those uploads (the memo, the stored row, and the process's confirmation)
/// so the next request uploads them again. Returns the uploads it forgot.
pub(crate) async fn discard_missing_uploads(
    messages: &[Message],
    error: &Error,
    provider: Provider,
    config: &Config,
    connection: &Connection,
) -> Vec<UploadedFile> {
    let scope = provider_upload_scope(provider, config);
    let uploads: Vec<(&Attachment, UploadedFile)> = messages
        .iter()
        .flat_map(|m| m.attachments.iter())
        .filter_map(|a| a.provider_uploads().get(&scope).map(|u| (a, u)))
        .collect();
    // `names_file?`: the error message or the response body mentions the file id.
    let names = |id: &str| {
        error.to_string().contains(id) || error.response().is_some_and(|r| r.body.contains(id))
    };
    let named: Vec<(&Attachment, UploadedFile)> = uploads
        .iter()
        .filter(|(_, u)| names(&u.id))
        .cloned()
        .collect();
    let missing = if named.is_empty() && error.response().is_some_and(|r| r.status == 404) {
        uploads
    } else {
        named
    };
    let mut forgotten = Vec::with_capacity(missing.len());
    for (attachment, upload) in missing {
        let memo = attachment.provider_uploads();
        memo.remove(&scope);
        StoredUploads::new(provider, config, connection, memo.store())
            .forget(&upload)
            .await;
        forgotten.push(upload);
    }
    forgotten
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
        // `leaves the purpose requirement to OpenAI`: no purpose field, no local refusal.
        let a = Attachment::from_bytes(b"{}".to_vec(), "batch.jsonl", None);
        let fields = upload_fields(Provider::OpenAI, &a, &UploadOptions::default(), None).unwrap();
        assert!(fields.iter().all(|(k, _)| k != "purpose"), "{fields:?}");
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
        // `leaves file types, sizes, and explicit purposes to DeepSeek`.
        let text = Attachment::from_bytes(b"Ruby".to_vec(), "ruby.txt", None);
        let batch = UploadOptions {
            purpose: Some("batch"),
            ..Default::default()
        };
        assert_eq!(
            upload_fields(Provider::DeepSeek, &text, &batch, None).unwrap(),
            vec![("purpose".into(), "batch".into())]
        );
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

    fn ruby_txt() -> String {
        format!("{}/tests/fixtures/ruby.txt", env!("CARGO_MANIFEST_DIR"))
    }

    // spec: protocols/files_spec.rb:108 passes expires_in as expires_after seconds
    #[test]
    fn xai_passes_expires_in_as_expires_after_seconds() {
        let options = UploadOptions {
            expires_in: Some(3600),
            ..Default::default()
        };
        let fields =
            upload_fields(Provider::XAI, &Attachment::new(ruby_txt()), &options, None).unwrap();
        assert_eq!(fields, vec![("expires_after".into(), "3600".into())]);
    }

    // spec: protocols/files_spec.rb:123 normalizes file metadata
    #[test]
    fn openrouter_normalizes_file_metadata() {
        let file = parse_file_response(
            Provider::OpenRouter,
            &json!({
                "id": "file_123", "filename": "document.pdf", "mime_type": "application/pdf",
                "size_bytes": 1024, "created_at": "2025-01-01T00:00:00Z", "downloadable": false
            }),
        )
        .unwrap();
        assert_eq!(file.id, "file_123");
        assert_eq!(file.provider, "openrouter");
        assert_eq!(file.filename.as_deref(), Some("document.pdf"));
        assert_eq!(file.byte_size, Some(1024));
        assert_eq!(file.mime_type.as_deref(), Some("application/pdf"));
        assert_eq!(file.downloadable, Some(false));
    }

    // spec: protocols/files_spec.rb:211 prefixes a bare file id with the collection name
    #[test]
    fn gemini_prefixes_a_bare_file_id_with_the_collection_name() {
        assert_eq!(gemini_file_name("abc"), "files/abc");
        assert_eq!(gemini_file_name("files/abc"), "files/abc");
    }

    // spec: protocols/files_spec.rb:409 rewraps an attachment when a new filename is given
    #[test]
    fn file_attachment_rewraps_only_for_a_new_filename() {
        let attachment = Attachment::new(ruby_txt());
        assert_eq!(file_attachment(attachment.clone(), None), attachment);
        let renamed = file_attachment(attachment, Some("renamed.txt"));
        assert_eq!(renamed.filename.as_deref(), Some("renamed.txt"));
    }

    // spec: protocols/files_spec.rb:423 sizes a file from disk or from its content
    #[test]
    fn file_size_reads_disk_or_content() {
        let on_disk = std::fs::metadata(ruby_txt()).unwrap().len();
        assert_eq!(file_size(&Attachment::new(ruby_txt())).unwrap(), on_disk);
        let bytes = Attachment::from_bytes(b"12345".to_vec(), "a.txt", None);
        assert_eq!(file_size(&bytes).unwrap(), 5);
    }

    // spec: protocol_file_preprocessing_spec.rb:355 with uploads stored for the attachment > keeps uploads in memory when the provider cannot name the account
    #[tokio::test]
    async fn keeps_uploads_in_memory_when_the_provider_cannot_name_the_account() {
        #[derive(Default)]
        struct Store(Mutex<Vec<String>>);
        #[async_trait::async_trait]
        impl ProviderFileStore for Store {
            async fn fetch(&self, _: &str, _: &str) -> Option<UploadedFile> {
                panic!("a store without an account is never read")
            }
            async fn store(&self, upload: &UploadedFile, _: &str, _: &str) {
                self.0.lock().unwrap().push(upload.id.clone());
            }
            async fn forget(&self, _: &str, _: &str, _: &str) {}
        }
        // Mistral's `account_identity` is nil, as the spec stubs Gemini's.
        let config = Config::default();
        assert_eq!(Provider::Mistral.account_identity(&config), None);
        let connection = Connection::new(Provider::Mistral, Arc::new(config.clone())).unwrap();
        let store = Arc::new(Store::default());
        let stored = StoredUploads::new(
            Provider::Mistral,
            &config,
            &connection,
            Some(store.clone() as Arc<dyn ProviderFileStore>),
        );
        let file = parse_file_response(Provider::Mistral, &json!({ "id": "files/new" })).unwrap();
        let uploaded = stored.fetch(async { Ok(file) }).await.unwrap();
        assert_eq!(uploaded.id, "files/new");
        assert!(store.0.lock().unwrap().is_empty());
    }

    // spec: protocols/deepseek/files_spec.rb:29 reports that stored images cannot be downloaded
    #[tokio::test]
    async fn deepseek_reports_that_stored_images_cannot_be_downloaded() {
        let file = parse_file_response(
            Provider::DeepSeek,
            &json!({ "id": "file-api-image", "filename": "ruby.png", "bytes": 10 }),
        )
        .unwrap();
        assert_eq!(file.downloadable, Some(false));
        let mut config = Config::default();
        config.set("deepseek_api_key", "test");
        let connection = Connection::new(Provider::DeepSeek, Arc::new(config)).unwrap();
        let err = download_file(&connection, Provider::DeepSeek, &file.id)
            .await
            .unwrap_err();
        assert!(matches!(err, Error::Api(..)), "{err:?}");
        assert!(
            err.to_string().contains("does not support downloading"),
            "{err}"
        );
    }

    // spec: protocols/perplexity/files_spec.rb:12 binds a generated file to its response and preserves native metadata
    #[test]
    fn perplexity_binds_a_generated_file_to_its_response() {
        let data =
            json!({ "id": "file_123", "filename": "numbers.csv", "bytes": 9, "created_at": 100 });
        let file = parse_perplexity_file("resp_123", &data).unwrap();
        assert_eq!(file.id, "resp_123/files/file_123");
        assert_eq!(file.filename.as_deref(), Some("numbers.csv"));
        assert_eq!(file.byte_size, Some(9));
        assert_eq!(file.provider, "perplexity");
        assert_eq!(file.created_at, DateTime::from_timestamp(100, 0));
        assert_eq!(file.downloadable, Some(true));
        let mut expected = data.clone();
        expected["response_id"] = json!("resp_123");
        assert_eq!(file.metadata, expected);
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
