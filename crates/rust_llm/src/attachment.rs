//! Port of `lib/ruby_llm/attachment.rb` and `files/mime_type.rb`.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use base64::Engine;

use crate::error::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttachmentType {
    Image,
    Video,
    Audio,
    Pdf,
    Text,
    Document,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Resolution {
    Low,
    Medium,
    High,
    UltraHigh,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Source {
    Path(PathBuf),
    Url(String),
    Bytes(Vec<u8>),
    /// A file already stored with a provider (`Attachment.new(uploaded_file)`).
    ProviderFile(Box<crate::files::UploadedFile>),
}

/// A file sent to (or returned by) a model. Local paths are read when the request is rendered;
/// URLs are passed through to providers that accept them and fetched otherwise.
#[derive(Debug, Clone, PartialEq)]
pub struct Attachment {
    pub source: Source,
    pub filename: Option<String>,
    pub mime_type: String,
    pub resolution: Option<Resolution>,
    /// `@content`, read once and shared by clones, so a later turn's copy of history reuses the
    /// bytes instead of fetching the URL again.
    content: Arc<OnceLock<Vec<u8>>>,
    /// Set when rendering asked for a URL's bytes before they were fetched (Ruby's lazy `content`).
    wanted: Wanted,
    provider_uploads: crate::files::ProviderUploads,
}

/// Shared by clones, so the copy a render reads flags the attachment history keeps.
#[derive(Debug, Clone, Default)]
struct Wanted(Arc<AtomicBool>);

impl PartialEq for Wanted {
    /// Whether a render asked for the bytes is not part of an attachment's identity.
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

const DOCUMENT_EXTENSIONS: &[&str] = &[
    "doc", "docx", "dot", "key", "numbers", "odp", "ods", "odt", "pages", "pot", "pps", "ppt", "pptx", "rtf",
    "xls", "xlsx",
];
const TEXT_SUFFIXES: &[&str] = &["+json", "+xml", "+html", "+yaml", "+csv", "+plain", "+javascript", "+svg"];
const TEXT_MIME_TYPES: &[&str] = &[
    "application/json", "application/xml", "application/javascript", "application/ecmascript",
    "application/rtf", "application/sql", "application/x-sh", "application/x-csh", "application/x-httpd-php",
    "application/sdp", "application/sparql-query", "application/graphql", "application/yang", "application/mbox",
    "application/x-tex", "application/x-latex", "application/x-perl", "application/x-python", "application/x-tcl",
    "application/pgp-signature", "application/pgp-keys", "application/vnd.coffeescript", "application/vnd.dart",
    "application/vnd.oai.openapi", "application/vnd.zul", "application/x-yaml", "application/yaml",
    "application/toml",
];
const DOCUMENT_MIME_TYPES: &[&str] = &[
    "application/msword", "application/rtf", "application/vnd.apple.keynote", "application/vnd.apple.numbers",
    "application/vnd.apple.pages", "application/vnd.google-apps.document",
];
const DOCUMENT_MIME_PREFIXES: &[&str] =
    &["application/vnd.openxmlformats-officedocument.", "application/vnd.oasis.opendocument."];

/// The magic numbers Marcel checks for the media types attachments carry.
fn sniff(bytes: &[u8]) -> Option<&'static str> {
    let at = |offset: usize, magic: &[u8]| bytes.get(offset..offset + magic.len()) == Some(magic);
    if at(0, b"\x89PNG\r\n\x1a\n") {
        Some("image/png")
    } else if at(0, b"\xff\xd8\xff") {
        Some("image/jpeg")
    } else if at(0, b"GIF87a") || at(0, b"GIF89a") {
        Some("image/gif")
    } else if at(0, b"RIFF") && at(8, b"WEBP") {
        Some("image/webp")
    } else if at(0, b"RIFF") && at(8, b"WAVE") {
        Some("audio/wav")
    } else if at(0, b"%PDF-") {
        Some("application/pdf")
    } else if at(4, b"ftyp") {
        Some("video/mp4")
    } else if at(0, b"ID3") {
        Some("audio/mpeg")
    } else {
        None
    }
}

/// Text formats `mime_guess` maps to octet-stream but Marcel knows as text.
const TEXT_EXTENSIONS: &[&str] = &["rb", "rs", "py", "go", "ts", "tsx", "jsx", "md", "yml", "yaml", "toml", "sh"];

fn mime_for_name(name: &str) -> String {
    let ext = Path::new(name).extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
    if TEXT_EXTENSIONS.contains(&ext.as_str()) {
        return "text/plain".into();
    }
    match mime_guess::from_path(name).first() {
        Some(m) if m.essence_str() == "audio/x-wav" => "audio/wav".into(),
        // Marcel types `.xml` as application/xml where mime_guess says text/xml.
        Some(m) if m.essence_str() == "text/xml" => "application/xml".into(),
        Some(m) => m.essence_str().to_string(),
        None => "application/octet-stream".into(),
    }
}

impl From<&str> for Attachment {
    fn from(source: &str) -> Attachment {
        Attachment::new(source)
    }
}

impl From<crate::files::UploadedFile> for Attachment {
    fn from(file: crate::files::UploadedFile) -> Attachment {
        Attachment::from_uploaded(file)
    }
}

impl Attachment {
    /// `Attachment.new(source)`: a local path or an http(s) URL.
    pub fn new(source: impl AsRef<str>) -> Attachment {
        let source = source.as_ref();
        // `url?`: `\Ahttps?://` case-insensitively.
        let scheme = source.get(..8).unwrap_or(source).to_ascii_lowercase();
        if scheme.starts_with("http://") || scheme.starts_with("https://") {
            let path = source.split(['?', '#']).next().unwrap_or(source);
            let filename = path.rsplit('/').next().map(str::to_string);
            let mime = mime_for_name(filename.as_deref().unwrap_or(""));
            Attachment { source: Source::Url(source.to_string()), filename, mime_type: mime, resolution: None, content: Default::default(), wanted: Default::default(), provider_uploads: Default::default() }
        } else {
            let path = PathBuf::from(source);
            let filename = path.file_name().map(|f| f.to_string_lossy().into_owned());
            let mime = mime_for_name(source);
            Attachment { source: Source::Path(path), filename, mime_type: mime, resolution: None, content: Default::default(), wanted: Default::default(), provider_uploads: Default::default() }
        }
    }

    pub fn from_bytes(bytes: Vec<u8>, filename: impl Into<String>, mime_type: Option<&str>) -> Attachment {
        let filename = filename.into();
        let mime = mime_type.map(str::to_string).unwrap_or_else(|| mime_for_name(&filename));
        Attachment {
            source: Source::Bytes(bytes.clone()),
            filename: Some(filename),
            mime_type: mime,
            resolution: None,
            content: Arc::new(OnceLock::from(bytes)),
            wanted: Default::default(),
            provider_uploads: Default::default(),
        }
    }

    pub fn with_resolution(mut self, resolution: Resolution) -> Attachment {
        self.resolution = Some(resolution);
        self
    }

    pub fn is_url(&self) -> bool {
        matches!(self.source, Source::Url(_))
    }

    pub fn url(&self) -> Option<&str> {
        match &self.source {
            Source::Url(u) => Some(u),
            _ => None,
        }
    }

    /// `Attachment.new(uploaded_file)`: the filename and MIME type come from the provider's record.
    pub fn from_uploaded(file: crate::files::UploadedFile) -> Attachment {
        let filename = file.filename.clone();
        let mime = file.mime_type.clone().unwrap_or_else(|| mime_for_name(filename.as_deref().unwrap_or("")));
        Attachment {
            source: Source::ProviderFile(Box::new(file)),
            filename,
            mime_type: mime,
            resolution: None,
            content: Default::default(),
            wanted: Default::default(),
            provider_uploads: Default::default(),
        }
    }

    /// `Attachment.new(source, filename:)`: the same source under another name, typed by that name.
    pub(crate) fn with_filename(&self, filename: &str) -> Attachment {
        let mime_type = match &self.source {
            Source::ProviderFile(f) => f.mime_type.clone().unwrap_or_else(|| mime_for_name(filename)),
            _ => mime_for_name(filename),
        };
        Attachment {
            filename: Some(filename.to_string()),
            mime_type,
            provider_uploads: Default::default(),
            ..self.clone()
        }
    }

    pub fn is_provider_file(&self) -> bool {
        matches!(self.source, Source::ProviderFile(_))
    }

    pub fn provider_file_id(&self) -> Option<&str> {
        match &self.source {
            Source::ProviderFile(f) => Some(&f.id),
            _ => None,
        }
    }

    pub fn provider_file_uri(&self) -> Option<&str> {
        match &self.source {
            Source::ProviderFile(f) => f.uri.as_deref(),
            _ => None,
        }
    }

    /// Files this attachment has been auto-uploaded to, keyed by provider and credentials. Shared
    /// by clones, so the per-request copy of history reuses the upload.
    pub(crate) fn provider_uploads(&self) -> &crate::files::ProviderUploads {
        &self.provider_uploads
    }

    /// `Attachment#byte_size`: the provider's size, the file's size on disk, or the loaded bytes.
    pub fn byte_size(&self) -> Option<u64> {
        match &self.source {
            Source::ProviderFile(f) => f.byte_size,
            Source::Path(p) => std::fs::metadata(p).ok().map(|m| m.len()),
            _ => self.content.get().map(|c| c.len() as u64),
        }
    }

    /// A URL whose bytes a render asked for before they were fetched.
    pub(crate) fn is_wanted(&self) -> bool {
        self.is_url() && self.content.get().is_none() && self.wanted.0.load(Ordering::SeqCst)
    }

    /// Reads the bytes now, so rendering a request never blocks on the network. A source whose
    /// name gives no MIME type is typed from its bytes (`MimeType.for(content)`).
    pub(crate) async fn load(&mut self, client: &reqwest::Client) -> Result<()> {
        if self.content.get().is_some() {
            return Ok(());
        }
        let bytes = match &self.source {
            Source::ProviderFile(_) => return Ok(()),
            Source::Path(p) => tokio::fs::read(p).await?,
            Source::Bytes(b) => b.clone(),
            Source::Url(u) => client
                .get(u)
                .send()
                .await
                .and_then(|r| r.error_for_status())
                .map_err(|e| Error::ConnectionFailed(e.to_string()))?
                .bytes()
                .await
                .map_err(|e| Error::ConnectionFailed(e.to_string()))?
                .to_vec(),
        };
        if self.mime_type == "application/octet-stream"
            && let Some(mime) = sniff(&bytes) {
                self.mime_type = mime.to_string();
            }
        let _ = self.content.set(bytes);
        Ok(())
    }

    /// What `Attachment.new` reads before a request: local files, and a URL whose name leaves
    /// the MIME type unknown (Ruby fetches it to detect the type). Other URLs are fetched only
    /// if the request needs their bytes.
    pub(crate) async fn prepare(&mut self, client: &reqwest::Client) -> Result<()> {
        if !self.is_url() || self.mime_type == "application/octet-stream" {
            self.load(client).await?;
        }
        Ok(())
    }

    pub(crate) fn bytes(&self) -> Result<&[u8]> {
        if let Some(id) = self.provider_file_id() {
            return Err(Error::Api(format!("Provider-managed file {id} cannot be read as inline attachment content"), None));
        }
        if self.is_url() && self.content.get().is_none() {
            self.wanted.0.store(true, Ordering::SeqCst);
        }
        self.content.get().map(Vec::as_slice).ok_or_else(|| {
            Error::Argument(format!("attachment {:?} was not loaded before rendering", self.filename))
        })
    }

    /// `Attachment#content`: the raw bytes, reading or fetching the source on the first call.
    /// Fails for a provider-managed file, which has no local content.
    pub async fn content(&mut self) -> Result<Vec<u8>> {
        if !self.is_provider_file() {
            self.load(&reqwest::Client::new()).await?;
        }
        self.bytes().map(<[u8]>::to_vec)
    }

    pub fn content_text(&self) -> Result<String> {
        Ok(String::from_utf8_lossy(self.bytes()?).into_owned())
    }

    pub fn encoded(&self) -> Result<String> {
        Ok(base64::engine::general_purpose::STANDARD.encode(self.bytes()?))
    }

    pub fn data_uri(&self) -> Result<String> {
        Ok(format!("data:{};base64,{}", self.mime_type, self.encoded()?))
    }

    pub fn url_or_data_uri(&self) -> Result<String> {
        match &self.source {
            Source::Url(u) => Ok(u.clone()),
            _ => self.data_uri(),
        }
    }

    /// `Attachment#for_llm`: text files are wrapped in a `<file>` tag, everything else is a data URI.
    pub fn for_llm(&self) -> Result<String> {
        match self.kind() {
            AttachmentType::Text => Ok(format!(
                "<file name='{}' mime_type='{}'>{}</file>",
                self.filename.as_deref().unwrap_or(""),
                self.mime_type,
                self.content_text()?
            )),
            _ => self.data_uri(),
        }
    }

    pub fn kind(&self) -> AttachmentType {
        let m = self.mime_type.as_str();
        if m.starts_with("image/") {
            AttachmentType::Image
        } else if m.starts_with("video/") {
            AttachmentType::Video
        } else if m.starts_with("audio/") {
            AttachmentType::Audio
        } else if m == "application/pdf" {
            AttachmentType::Pdf
        } else if self.is_text() {
            AttachmentType::Text
        } else if self.is_document() {
            AttachmentType::Document
        } else {
            AttachmentType::Unknown
        }
    }

    fn is_text(&self) -> bool {
        let m = self.mime_type.as_str();
        m.starts_with("text/") || TEXT_SUFFIXES.iter().any(|s| m.ends_with(s)) || TEXT_MIME_TYPES.contains(&m)
    }

    fn is_document(&self) -> bool {
        let m = self.mime_type.as_str();
        if m == "application/pdf" || self.is_text() {
            return false;
        }
        let ext = self
            .filename
            .as_deref()
            .and_then(|f| Path::new(f).extension())
            .and_then(|e| e.to_str())
            .map(str::to_lowercase);
        DOCUMENT_MIME_TYPES.contains(&m)
            || DOCUMENT_MIME_PREFIXES.iter().any(|p| m.starts_with(p))
            || ext.is_some_and(|e| DOCUMENT_EXTENSIONS.contains(&e.as_str()))
    }

    /// `Attachment#format`: the short audio format name providers expect.
    pub fn format(&self) -> String {
        match self.mime_type.as_str() {
            "audio/mpeg" => "mp3".into(),
            "audio/wav" | "audio/wave" | "audio/x-wav" => "wav".into(),
            m => m.rsplit('/').next().unwrap_or(m).to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_by_mime_type() {
        assert_eq!(Attachment::new("ruby.png").kind(), AttachmentType::Image);
        assert_eq!(Attachment::new("contract.pdf").kind(), AttachmentType::Pdf);
        assert_eq!(Attachment::new("app.rb").kind(), AttachmentType::Text);
        assert_eq!(Attachment::new("meeting.wav").kind(), AttachmentType::Audio);
        assert_eq!(Attachment::new("meeting.wav").format(), "wav");
        assert_eq!(Attachment::new("deck.pptx").kind(), AttachmentType::Document);
    }

    #[test]
    fn text_files_are_wrapped_for_the_model() {
        let a = Attachment::from_bytes(b"puts 1".to_vec(), "app.rb", None);
        assert_eq!(a.for_llm().unwrap(), "<file name='app.rb' mime_type='text/plain'>puts 1</file>");
    }
}
