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
    let home = std::env::var_os("HOME")
        .filter(|h| !h.is_empty())
        .map(PathBuf::from);
    cache_path_for(std::env::consts::OS, home, |name| std::env::var(name).ok())
}

/// `Registry.cache_path` for a given operating system (`std::env::consts::OS`), home directory,
/// and environment, the way Ruby branches on `RbConfig::CONFIG['host_os']`.
pub fn cache_path_for(
    os: &str,
    home: Option<PathBuf>,
    env: impl Fn(&str) -> Option<String>,
) -> Option<PathBuf> {
    let var = |name: &str| env(name).filter(|v| !v.is_empty()).map(PathBuf::from);
    let directory = match os {
        "macos" => home?.join("Library/Caches/RustLLM"),
        "windows" => var("LOCALAPPDATA")
            .or_else(|| home.map(|h| h.join("AppData/Local")))?
            .join("RustLLM/Cache"),
        _ => var("XDG_CACHE_HOME")
            .or_else(|| home.map(|h| h.join(".cache")))?
            .join("rust_llm"),
    };
    Some(directory.join("models.json"))
}

/// `Registry.read`: `None` when the file does not exist.
pub fn read(path: &Path) -> Result<Option<Vec<Model>>> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => {
            return Err(Error::ModelRegistry(format!(
                "Could not read the model registry from {}: {e}",
                path.display()
            )));
        }
    };
    let data: Value = serde_json::from_str(&text).map_err(|e| {
        Error::ModelRegistry(format!(
            "Invalid model registry JSON in {}: {e}",
            path.display()
        ))
    })?;
    models_from_data(data, Some(&path.display().to_string())).map(Some)
}

/// `Registry.models_from_data`: `source` names where the data came from in the error.
pub fn models_from_data(data: Value, source: Option<&str>) -> Result<Vec<Model>> {
    let location = source.map(|s| format!(" in {s}")).unwrap_or_default();
    if !data.is_array() {
        return Err(Error::ModelRegistry(format!(
            "Model registry{location} must be a JSON array"
        )));
    }
    serde_json::from_value(data)
        .map_err(|e| Error::ModelRegistry(format!("Invalid model registry entry{location}: {e}")))
}

/// `Registry.pretty_json`.
pub fn pretty_json(models: &[&Model]) -> Result<String> {
    Ok(format!("{}\n", serde_json::to_string_pretty(models)?))
}

/// `config.model_registry_store`: where the registry lives instead of `model_registry_file`, such
/// as an application's database. A store wins over the file: the registry loads from it when it
/// holds models, and `refresh` saves to it and then adopts what it reads back, so a store that
/// keeps entries the merge dropped (unlisted models still referenced) reports them.
///
/// `write` is optional in RubyLLM (`respond_to?(:write)`); the default here reports the store as
/// read-only, which is the error Ruby raises for one.
pub trait ModelRegistryStore: Send + Sync {
    /// `store.read`: the stored models (empty when there are none).
    fn read(&self) -> Result<Vec<Model>>;

    /// `store.write(models)`: replaces the stored registry.
    fn write(&self, _models: &super::Models) -> Result<()> {
        Err(Error::ModelRegistry(format!(
            "Model registry store {} is read-only",
            std::any::type_name::<Self>()
        )))
    }

    /// `store.description`, else the store's type name: names the store in save errors.
    fn description(&self) -> String {
        std::any::type_name::<Self>().to_string()
    }
}

impl std::fmt::Debug for dyn ModelRegistryStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ModelRegistryStore({})", self.description())
    }
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
            return Err(Error::ModelRegistry(
                "A model registry file path is required".into(),
            ));
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
    let directory = destination
        .parent()
        .filter(|d| !d.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let name = destination
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
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
    let url = config
        .get("model_registry_url")
        .unwrap_or(PUBLISHED_URL)
        .to_string();
    let wrap = |message: String| {
        Error::ModelRegistry(format!(
            "Could not refresh the model registry from {url}: {message}"
        ))
    };
    let client = reqwest::Client::builder()
        .timeout(config.request_timeout)
        .build()
        .map_err(|e| wrap(e.to_string()))?;
    let mut request = client.get(&url);
    if let Some(etag) = etag {
        request = request.header("If-None-Match", etag);
    }
    let response = request.send().await.map_err(|e| wrap(e.to_string()))?;
    let status = response.status();
    let returned_etag = response
        .headers()
        .get("etag")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    if status == reqwest::StatusCode::NOT_MODIFIED {
        return Ok(Published {
            models: None,
            etag: returned_etag.or(etag.map(str::to_string)),
            not_modified: true,
        });
    }
    if !status.is_success() {
        return Err(wrap(format!(
            "the server responded with status {}",
            status.as_u16()
        )));
    }
    let body: Value = response.json().await.map_err(|e| wrap(e.to_string()))?;
    let models = models_from_data(body, Some(&url))?;
    if models.is_empty() {
        return Err(Error::ModelRegistry(
            "Published model registry is empty".into(),
        ));
    }
    Ok(Published {
        models: Some(models),
        etag: returned_etag,
        not_modified: false,
    })
}
