//! Port of `lib/ruby_llm/models/registry.rb`: reading and atomically writing registry JSON files,
//! and fetching the published catalog with ETag revalidation.

use std::io::Write as _;
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::config::Config;
use crate::error::{Error, Result};
use crate::model::Model;

/// `Registry::PUBLISHED_URL`: RubyLLM's published catalog, which RustLLM shares (same format).
/// Set `config.set("model_registry_url", ...)` to read another copy.
pub const PUBLISHED_URL: &str = "https://rubyllm.com/models.json";

/// `Registry.cache_path`: the platform cache file `model_registry_file` defaults to.
pub fn cache_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME").filter(|h| !h.is_empty()).map(PathBuf::from);
    let directory = if cfg!(target_os = "macos") {
        home?.join("Library/Caches/RustLLM")
    } else if cfg!(windows) {
        let local = std::env::var_os("LOCALAPPDATA").filter(|v| !v.is_empty()).map(PathBuf::from);
        local.or_else(|| home.map(|h| h.join("AppData/Local")))?.join("RustLLM/Cache")
    } else {
        let xdg = std::env::var_os("XDG_CACHE_HOME").filter(|v| !v.is_empty()).map(PathBuf::from);
        xdg.or_else(|| home.map(|h| h.join(".cache")))?.join("rust_llm")
    };
    Some(directory.join("models.json"))
}

/// `Registry.read`: `None` when the file does not exist.
pub fn read(path: &Path) -> Result<Option<Vec<Model>>> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => {
            return Err(Error::ModelRegistry(format!("Could not read the model registry from {}: {e}", path.display())));
        }
    };
    let data: Value = serde_json::from_str(&text)
        .map_err(|e| Error::ModelRegistry(format!("Invalid model registry JSON in {}: {e}", path.display())))?;
    models_from_data(data, &path.display().to_string()).map(Some)
}

/// `Registry.models_from_data`.
pub fn models_from_data(data: Value, source: &str) -> Result<Vec<Model>> {
    if !data.is_array() {
        return Err(Error::ModelRegistry(format!("Model registry in {source} must be a JSON array")));
    }
    serde_json::from_value(data).map_err(|e| Error::ModelRegistry(format!("Invalid model registry entry in {source}: {e}")))
}

/// `Registry.pretty_json`.
pub fn pretty_json(models: &[&Model]) -> Result<String> {
    Ok(format!("{}\n", serde_json::to_string_pretty(models)?))
}

/// `Registry::FileStore`: a registry file plus the `<file>.etag` of the catalog it came from.
#[derive(Debug, Clone)]
pub struct FileStore {
    pub path: PathBuf,
}

impl FileStore {
    pub fn new(path: impl Into<PathBuf>) -> Result<FileStore> {
        let path = path.into();
        if path.as_os_str().is_empty() {
            return Err(Error::ModelRegistry("A model registry file path is required".into()));
        }
        Ok(FileStore { path })
    }

    pub fn read(&self) -> Result<Option<Vec<Model>>> {
        read(&self.path)
    }

    fn etag_path(&self) -> PathBuf {
        suffixed(&self.path, ".etag")
    }

    /// `FileStore#etag`: `None` unless the registry file exists and an ETag was saved with it.
    pub fn etag(&self) -> Result<Option<String>> {
        if !self.path.is_file() {
            return Ok(None);
        }
        match std::fs::read_to_string(self.etag_path()) {
            Ok(v) => Ok(Some(v.trim().to_string()).filter(|v| !v.is_empty())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(Error::ModelRegistry(format!(
                "Could not read the model registry ETag from {}: {e}",
                self.etag_path().display()
            ))),
        }
    }

    /// `FileStore#write`: replaces the file atomically, then records or clears the ETag.
    pub fn write(&self, models: &[&Model], etag: Option<&str>) -> Result<()> {
        if let Some(dir) = self.path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir)?;
        }
        atomic_write(&self.path, pretty_json(models)?.as_bytes())?;
        match etag {
            Some(etag) => atomic_write(&self.etag_path(), format!("{etag}\n").as_bytes())?,
            None => match std::fs::remove_file(self.etag_path()) {
                Err(e) if e.kind() != std::io::ErrorKind::NotFound => return Err(e.into()),
                _ => {}
            },
        }
        Ok(())
    }
}

pub(crate) fn suffixed(path: &Path, suffix: &str) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(suffix);
    PathBuf::from(s)
}

/// A temporary file in the destination's directory, fsynced, then renamed over it. The rename
/// keeps the destination's mode, so a registry other users can read stays readable.
fn atomic_write(destination: &Path, contents: &[u8]) -> Result<()> {
    let directory = destination.parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or(Path::new("."));
    let name = destination.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let temporary = directory.join(format!(".{name}.{}.tmp", uuid::Uuid::new_v4()));
    let result = (|| -> std::io::Result<()> {
        let mut file = std::fs::File::create(&temporary)?;
        file.write_all(contents)?;
        file.sync_all()?;
        if let Ok(meta) = std::fs::metadata(destination) {
            std::fs::set_permissions(&temporary, meta.permissions())?;
        }
        std::fs::rename(&temporary, destination)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    Ok(result?)
}

/// `PublishedSource::Result`: `models` is `None` on a 304 Not Modified.
#[derive(Debug, Clone)]
pub struct Published {
    pub models: Option<Vec<Model>>,
    pub etag: Option<String>,
    pub not_modified: bool,
}

/// `PublishedSource#fetch`: GET the catalog, revalidating with `If-None-Match` when `etag` is set.
pub async fn fetch_published(config: &Config, etag: Option<&str>) -> Result<Published> {
    let url = config.get("model_registry_url").unwrap_or(PUBLISHED_URL).to_string();
    let wrap = |message: String| Error::ModelRegistry(format!("Could not refresh the model registry from {url}: {message}"));
    let client = reqwest::Client::builder().timeout(config.request_timeout).build().map_err(|e| wrap(e.to_string()))?;
    let mut request = client.get(&url);
    if let Some(etag) = etag {
        request = request.header("If-None-Match", etag);
    }
    let response = request.send().await.map_err(|e| wrap(e.to_string()))?;
    let status = response.status();
    let returned_etag = response.headers().get("etag").and_then(|v| v.to_str().ok()).map(str::to_string);
    if status == reqwest::StatusCode::NOT_MODIFIED {
        return Ok(Published { models: None, etag: returned_etag.or(etag.map(str::to_string)), not_modified: true });
    }
    if !status.is_success() {
        return Err(wrap(format!("the server responded with status {}", status.as_u16())));
    }
    let body: Value = response.json().await.map_err(|e| wrap(e.to_string()))?;
    let models = models_from_data(body, &url)?;
    if models.is_empty() {
        return Err(Error::ModelRegistry("Published model registry is empty".into()));
    }
    Ok(Published { models: Some(models), etag: returned_etag, not_modified: false })
}
