//! Ports of RubyLLM 2.0's registry specs: `models/registry_spec.rb` (FileStore, PublishedSource,
//! ETags, cache_path, the registry store), `models/lookup_spec.rb`, `models_spec.rb`,
//! `models_refresh_spec.rb`, `models_local_refresh_spec.rb`, `models_json_validation_spec.rb`,
//! plus the `Model#initialize`, `Cost#to_h`, and `Support::Utils` examples.
//!
//! Most tests change the process-wide registry or configuration, so every test holds `LOCK` and a
//! `Restore` guard puts the bundled registry and the previous configuration back.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use rust_llm::cost::Tier;
use rust_llm::models::registry::{self, FileStore, ModelRegistryStore};
use rust_llm::models::{Models, refresh};
use rust_llm::tokens::Tokens;
use rust_llm::{Chat, Config, Cost, Error, Model, Provider};
use serde_json::{Value, json};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

static LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
static BUNDLE: &str = include_str!("../data/models.json");

/// Restores the bundled registry and the configuration it saw, and removes its temp directory.
struct Restore {
    config: Arc<Config>,
    dir: PathBuf,
}

impl Restore {
    fn new(name: &str) -> Restore {
        let dir = std::env::temp_dir().join(format!(
            "rust_llm_spec_registry_{}_{name}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Restore {
            config: rust_llm::config(),
            dir,
        }
    }

    fn file(&self, name: &str) -> PathBuf {
        self.dir.join(name)
    }
}

impl Drop for Restore {
    fn drop(&mut self) {
        let saved = (*self.config).clone();
        rust_llm::configure(|c| *c = saved);
        refresh::reset();
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// Replaces the global configuration (`allow(RubyLLM).to receive(:config)`).
fn set_global(config: Config) {
    rust_llm::configure(|c| *c = config);
}

fn model(data: Value) -> Model {
    serde_json::from_value(data).unwrap()
}

/// `let(:model)`: new-model at openai.
fn new_model() -> Model {
    model(json!({ "id": "new-model", "name": "New Model", "provider": "openai" }))
}

/// `let(:old_model)`: old-model at openai.
fn old_model() -> Model {
    model(json!({ "id": "old-model", "name": "Old Model", "provider": "openai" }))
}

fn ids(models: Vec<&Model>) -> Vec<String> {
    models.into_iter().map(|m| m.id.clone()).collect()
}

/// The published catalog: `GET /models.json` answers `body` with `etag`.
async fn catalog(body: Value, etag: Option<&str>) -> MockServer {
    let server = MockServer::start().await;
    let mut response = ResponseTemplate::new(200).set_body_json(body);
    if let Some(etag) = etag {
        response = response.insert_header("etag", etag);
    }
    Mock::given(method("GET"))
        .and(path("/models.json"))
        .respond_with(response)
        .mount(&server)
        .await;
    server
}

/// A configuration with no providers, the published catalog at `server`, and a registry file in
/// the test's directory.
fn refresh_config(server: &MockServer, restore: &Restore) -> Config {
    let mut config = Config::default();
    config.set(
        "model_registry_url",
        format!("{}/models.json", server.uri()),
    );
    config.model_registry_file = Some(restore.file("models.json"));
    config
}

/// A model registry store double (`double('model registry store', ...)`).
struct TestStore {
    stored: Mutex<Vec<Model>>,
    fail_with: Option<String>,
    seen: Mutex<Vec<(Vec<String>, Vec<String>)>>,
}

impl TestStore {
    fn new(stored: Vec<Model>) -> TestStore {
        TestStore {
            stored: Mutex::new(stored),
            fail_with: None,
            seen: Mutex::new(Vec::new()),
        }
    }
}

impl ModelRegistryStore for TestStore {
    fn read(&self) -> rust_llm::Result<Vec<Model>> {
        Ok(self.stored.lock().unwrap().clone())
    }

    fn write(&self, models: &Models) -> rust_llm::Result<()> {
        // What the store was given, next to what the global registry held at that moment.
        self.seen
            .lock()
            .unwrap()
            .push((ids(models.all()), ids(rust_llm::models().all())));
        match &self.fail_with {
            Some(message) => Err(Error::Io(std::io::Error::other(message.clone()))),
            None => Ok(()),
        }
    }

    fn description(&self) -> String {
        "database:Model".into()
    }
}

fn with_store(mut config: Config, store: Arc<TestStore>) -> Config {
    config.model_registry_store = Some(store);
    config
}

// ---- models/registry_spec.rb: .cache_path ------------------------------------------------------

// spec: models/registry_spec.rb:30 .cache_path > uses XDG_CACHE_HOME on Linux
#[test]
fn cache_path_uses_xdg_cache_home_on_linux() {
    let env = |name: &str| (name == "XDG_CACHE_HOME").then(|| "/cache".to_string());
    assert_eq!(
        registry::cache_path_for("linux", Some("/home/me".into()), env),
        Some(PathBuf::from("/cache/rust_llm/models.json"))
    );
}

// spec: models/registry_spec.rb:37 .cache_path > uses the native cache directory on macOS
#[test]
fn cache_path_uses_the_native_cache_directory_on_macos() {
    assert_eq!(
        registry::cache_path_for("macos", Some("/home/me".into()), |_| None),
        Some(PathBuf::from("/home/me/Library/Caches/RustLLM/models.json"))
    );
}

// spec: models/registry_spec.rb:43 .cache_path > uses the native cache directory on Windows
#[test]
fn cache_path_uses_the_native_cache_directory_on_windows() {
    let env =
        |name: &str| (name == "LOCALAPPDATA").then(|| "C:/Users/me/AppData/Local".to_string());
    assert_eq!(
        registry::cache_path_for("windows", Some("/home/me".into()), env),
        Some(PathBuf::from(
            "C:/Users/me/AppData/Local/RustLLM/Cache/models.json"
        ))
    );
}

// ---- .models_from_data -------------------------------------------------------------------------

// spec: models/registry_spec.rb:52 .models_from_data > reads the registry as a top-level array
#[test]
fn models_from_data_reads_a_top_level_array() {
    let data = serde_json::to_value(vec![new_model()]).unwrap();
    let models = registry::models_from_data(data, Some("models.json")).unwrap();
    assert_eq!(ids(models.iter().collect()), ["new-model"]);
}

// spec: models/registry_spec.rb:58 .models_from_data > rejects an object envelope
#[test]
fn models_from_data_rejects_an_object_envelope() {
    let err = registry::models_from_data(json!({ "models": [new_model()] }), Some("models.json"))
        .unwrap_err();
    assert!(
        matches!(&err, Error::ModelRegistry(m) if m.contains("must be a JSON array")),
        "{err}"
    );
}

// spec: models/registry_spec.rb:328 .models_from_data validation > refuses anything but an array
#[test]
fn models_from_data_refuses_anything_but_an_array() {
    let err = registry::models_from_data(json!({}), None).unwrap_err();
    assert!(matches!(&err, Error::ModelRegistry(m) if m == "Model registry must be a JSON array"));
}

// spec: models/registry_spec.rb:334 .models_from_data validation > names the source in the error
#[test]
fn models_from_data_names_the_source_in_the_error() {
    let err = registry::models_from_data(json!({}), Some("/tmp/models.json")).unwrap_err();
    assert!(
        matches!(&err, Error::ModelRegistry(m) if m == "Model registry in /tmp/models.json must be a JSON array")
    );
}

// ---- FileStore -----------------------------------------------------------------------------------

// spec: models/registry_spec.rb:66 RubyLLM::Models::Registry::FileStore > reads UTF-8 model names under an ASCII locale
#[tokio::test]
async fn file_store_reads_utf8_model_names() {
    let _lock = LOCK.lock().await;
    let restore = Restore::new("utf8");
    let file = restore.file("models.json");
    let mut entry = serde_json::to_value(new_model()).unwrap();
    entry["name"] = "Modèle français".into();
    std::fs::write(&file, serde_json::to_vec(&json!([entry])).unwrap()).unwrap();

    let read = FileStore::new(&file).unwrap().read().unwrap().unwrap();
    assert_eq!(read[0].name, "Modèle français");
}

// spec: models/registry_spec.rb:80 RubyLLM::Models::Registry::FileStore > writes a top-level array and an adjacent ETag
#[tokio::test]
async fn file_store_writes_a_top_level_array_and_an_adjacent_etag() {
    let _lock = LOCK.lock().await;
    let restore = Restore::new("etag_write");
    let file = restore.file("models.json");
    let store = FileStore::new(&file).unwrap();

    store
        .write(&[&new_model()], Some("\"registry-1\""))
        .unwrap();

    let saved: Value = serde_json::from_str(&std::fs::read_to_string(&file).unwrap()).unwrap();
    assert!(saved.is_array());
    assert_eq!(
        ids(store.read().unwrap().unwrap().iter().collect()),
        ["new-model"]
    );
    assert_eq!(store.etag().unwrap().as_deref(), Some("\"registry-1\""));
    assert_eq!(
        std::fs::read_to_string(restore.file("models.json.etag"))
            .unwrap()
            .trim(),
        "\"registry-1\""
    );
}

/// The process umask, from `/proc/self/status` (Linux).
#[cfg(target_os = "linux")]
fn umask() -> u32 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    let line = status.lines().find(|l| l.starts_with("Umask:")).unwrap();
    u32::from_str_radix(line.trim_start_matches("Umask:").trim(), 8).unwrap()
}

// spec: models/registry_spec.rb:94 RubyLLM::Models::Registry::FileStore > leaves the registry readable instead of inheriting the tempfile mode
#[cfg(target_os = "linux")]
#[tokio::test]
async fn file_store_leaves_the_registry_readable() {
    use std::os::unix::fs::PermissionsExt;
    let _lock = LOCK.lock().await;
    let restore = Restore::new("mode");
    let file = restore.file("models.json");
    let store = FileStore::new(&file).unwrap();
    let mode = |p: &Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;

    store.write(&[&new_model()], None).unwrap();
    assert_eq!(mode(&file), 0o666 & !umask());

    std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o640)).unwrap();
    store.write(&[&new_model()], None).unwrap();
    assert_eq!(mode(&file), 0o640);
}

// spec: models/registry_spec.rb:284 RubyLLM::Models::Registry::FileStore paths and ETags > requires a path
#[test]
fn file_store_requires_a_path() {
    let err = FileStore::new("").unwrap_err();
    assert!(
        matches!(&err, Error::ModelRegistry(m) if m == "A model registry file path is required")
    );
}

// spec: models/registry_spec.rb:290 RubyLLM::Models::Registry::FileStore paths and ETags > accepts anything that names a path
#[test]
fn file_store_accepts_anything_that_names_a_path() {
    let p = std::env::temp_dir().join("models.json");
    assert_eq!(FileStore::new(p.clone()).unwrap().path, p);
    assert_eq!(FileStore::new(p.as_path()).unwrap().path, p);
    assert_eq!(FileStore::new(p.display().to_string()).unwrap().path, p);
}

// spec: models/registry_spec.rb:297 RubyLLM::Models::Registry::FileStore paths and ETags > #etag > is nil when the registry file does not exist
#[tokio::test]
async fn etag_is_none_without_a_registry_file() {
    let _lock = LOCK.lock().await;
    let restore = Restore::new("etag_none");
    assert_eq!(
        FileStore::new(restore.file("models.json"))
            .unwrap()
            .etag()
            .unwrap(),
        None
    );
}

// spec: models/registry_spec.rb:303 RubyLLM::Models::Registry::FileStore paths and ETags > #etag > is nil when the etag file is empty
#[tokio::test]
async fn etag_is_none_when_the_etag_file_is_empty() {
    let _lock = LOCK.lock().await;
    let restore = Restore::new("etag_empty");
    let file = restore.file("models.json");
    std::fs::write(&file, "[]").unwrap();
    std::fs::write(restore.file("models.json.etag"), "\n").unwrap();
    assert_eq!(FileStore::new(&file).unwrap().etag().unwrap(), None);
}

// spec: models/registry_spec.rb:313 RubyLLM::Models::Registry::FileStore paths and ETags > #etag > reports an unreadable etag file
#[tokio::test]
async fn etag_reports_an_unreadable_etag_file() {
    let _lock = LOCK.lock().await;
    let restore = Restore::new("etag_unreadable");
    let file = restore.file("models.json");
    std::fs::write(&file, "[]").unwrap();
    // A directory where the ETag file should be cannot be read, and is not "missing".
    std::fs::create_dir(restore.file("models.json.etag")).unwrap();
    let err = FileStore::new(&file).unwrap().etag().unwrap_err();
    assert!(
        matches!(&err, Error::ModelRegistry(m) if m.starts_with("Could not read the model registry ETag")),
        "{err}"
    );
}

// spec: models/registry_spec.rb:348 .read > reports an unreadable registry file
#[tokio::test]
async fn read_reports_an_unreadable_registry_file() {
    let _lock = LOCK.lock().await;
    let restore = Restore::new("unreadable");
    let file = restore.file("models.json");
    std::fs::create_dir(&file).unwrap();
    let err = registry::read(&file).unwrap_err();
    let expected = format!("Could not read the model registry from {}", file.display());
    assert!(
        matches!(&err, Error::ModelRegistry(m) if m.starts_with(&expected)),
        "{err}"
    );
}

// ---- PublishedSource -----------------------------------------------------------------------------

// spec: models/registry_spec.rb:110 RubyLLM::Models::Registry::PublishedSource > loads the published top-level array
#[tokio::test]
async fn published_source_loads_the_published_array() {
    let server = catalog(json!([new_model()]), Some("\"registry-1\"")).await;
    let mut config = Config::default();
    config.set(
        "model_registry_url",
        format!("{}/models.json", server.uri()),
    );

    let result = registry::fetch_published(&config, None).await.unwrap();

    assert_eq!(
        ids(result.models.as_ref().unwrap().iter().collect()),
        ["new-model"]
    );
    assert_eq!(result.etag.as_deref(), Some("\"registry-1\""));
    assert!(!result.not_modified);
}

// spec: models/registry_spec.rb:125 RubyLLM::Models::Registry::PublishedSource > sends the cached ETag and handles an unmodified registry
#[tokio::test]
async fn published_source_sends_the_cached_etag() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/models.json"))
        .and(header("If-None-Match", "\"registry-1\""))
        .respond_with(ResponseTemplate::new(304).insert_header("etag", "\"registry-1\""))
        .expect(1)
        .mount(&server)
        .await;
    let mut config = Config::default();
    config.set(
        "model_registry_url",
        format!("{}/models.json", server.uri()),
    );

    let result = registry::fetch_published(&config, Some("\"registry-1\""))
        .await
        .unwrap();

    assert!(result.models.is_none());
    assert_eq!(result.etag.as_deref(), Some("\"registry-1\""));
    assert!(result.not_modified);
}

// spec: models/registry_spec.rb:358 RubyLLM::Models::Registry::PublishedSource failures > raises when the published catalog is empty
#[tokio::test]
async fn published_source_raises_on_an_empty_catalog() {
    let server = catalog(json!([]), None).await;
    let mut config = Config::default();
    config.set(
        "model_registry_url",
        format!("{}/models.json", server.uri()),
    );
    let err = registry::fetch_published(&config, None).await.unwrap_err();
    assert!(matches!(&err, Error::ModelRegistry(m) if m == "Published model registry is empty"));
}

// spec: models/registry_spec.rb:369 RubyLLM::Models::Registry::PublishedSource failures > wraps a transport failure
#[tokio::test]
async fn published_source_wraps_a_transport_failure() {
    let mut config = Config::default();
    config.set("model_registry_url", "http://127.0.0.1:9/models.json");
    let err = registry::fetch_published(&config, None).await.unwrap_err();
    assert!(
        matches!(&err, Error::ModelRegistry(m)
            if m.starts_with("Could not refresh the model registry from http://127.0.0.1:9/models.json")),
        "{err}"
    );
}

// spec: models/registry_spec.rb:380 RubyLLM::Models::Registry::PublishedSource failures > keeps the requested etag when the catalog has not changed
#[tokio::test]
async fn published_source_keeps_the_requested_etag_on_304() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(304))
        .mount(&server)
        .await;
    let mut config = Config::default();
    config.set(
        "model_registry_url",
        format!("{}/models.json", server.uri()),
    );
    let result = registry::fetch_published(&config, Some("etag-1"))
        .await
        .unwrap();
    assert!(result.not_modified);
    assert_eq!(result.etag.as_deref(), Some("etag-1"));
}

// ---- RubyLLM::Models with a registry file or store ------------------------------------------------

// spec: models/registry_spec.rb:152 RubyLLM::Models > uses a valid registry file when one exists
#[tokio::test]
async fn load_models_uses_a_valid_registry_file() {
    let _lock = LOCK.lock().await;
    let restore = Restore::new("load_valid");
    let cache = restore.file("models.json");
    FileStore::new(&cache)
        .unwrap()
        .write(&[&new_model()], None)
        .unwrap();
    let mut config = Config::default();
    config.model_registry_file = Some(cache);
    set_global(config);

    assert_eq!(
        ids(rust_llm::models::load_models().iter().collect()),
        ["new-model"]
    );
}

// spec: models/registry_spec.rb:166 RubyLLM::Models > falls back to the bundle when the registry file is corrupt
#[tokio::test]
async fn load_models_falls_back_to_the_bundle_for_a_corrupt_file() {
    let _lock = LOCK.lock().await;
    let restore = Restore::new("load_corrupt");
    let cache = restore.file("models.json");
    std::fs::write(&cache, "{broken").unwrap();
    let mut config = Config::default();
    config.model_registry_file = Some(cache);
    set_global(config);

    let loaded = rust_llm::models::load_models();
    assert!(!loaded.is_empty());
    assert!(!loaded.iter().any(|m| m.id == "new-model"));
}

// spec: models/registry_spec.rb:182 RubyLLM::Models > persists a successful refresh and reuses its ETag
#[tokio::test]
async fn refresh_persists_and_reuses_its_etag() {
    let _lock = LOCK.lock().await;
    let restore = Restore::new("persist_etag");
    let server = MockServer::start().await;
    // A request carrying the saved ETag gets 304; any other gets the catalog.
    Mock::given(method("GET"))
        .and(path("/models.json"))
        .and(header("If-None-Match", "\"registry-1\""))
        .respond_with(ResponseTemplate::new(304).insert_header("etag", "\"registry-1\""))
        .with_priority(1)
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/models.json"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("etag", "\"registry-1\"")
                .set_body_json(json!([new_model()])),
        )
        .expect(1)
        .mount(&server)
        .await;
    let config = Arc::new(refresh_config(&server, &restore));
    Models::install(vec![old_model()]);

    refresh::refresh_with_config(config.clone(), false)
        .await
        .unwrap();

    let file = restore.file("models.json");
    let saved: Value = serde_json::from_str(&std::fs::read_to_string(&file).unwrap()).unwrap();
    assert!(saved.is_array());
    assert_eq!(
        FileStore::new(&file).unwrap().etag().unwrap().as_deref(),
        Some("\"registry-1\"")
    );
    assert_eq!(ids(rust_llm::models().all()), ["new-model"]);

    refresh::refresh_with_config(config, false).await.unwrap();
    assert_eq!(ids(rust_llm::models().all()), ["new-model"]);
}

// spec: models/registry_spec.rb:210 RubyLLM::Models > raises on a cache write failure without changing the in-memory registry
#[tokio::test]
async fn refresh_raises_on_a_cache_write_failure() {
    let _lock = LOCK.lock().await;
    let restore = Restore::new("write_failure");
    let server = catalog(json!([new_model()]), None).await;
    let mut config = refresh_config(&server, &restore);
    // The registry's directory is a regular file, so nothing can be written under it.
    std::fs::write(restore.file("blocked"), "").unwrap();
    config.model_registry_file = Some(restore.file("blocked").join("models.json"));
    Models::install(vec![old_model()]);

    let err = refresh::refresh_with_config(Arc::new(config), false)
        .await
        .unwrap_err();

    assert!(
        matches!(&err, Error::ModelRegistry(m) if m.contains("Could not save")),
        "{err}"
    );
    assert_eq!(ids(rust_llm::models().all()), ["old-model"]);
}

// spec: models/registry_spec.rb:231 RubyLLM::Models > raises on a database write failure without changing the in-memory registry
#[tokio::test]
async fn refresh_raises_on_a_store_write_failure() {
    let _lock = LOCK.lock().await;
    let restore = Restore::new("store_failure");
    let server = catalog(json!([new_model()]), None).await;
    let store = Arc::new(TestStore {
        fail_with: Some("database unavailable".into()),
        ..TestStore::new(Vec::new())
    });
    let config = with_store(refresh_config(&server, &restore), store);
    Models::install(vec![old_model()]);

    let err = refresh::refresh_with_config(Arc::new(config), false)
        .await
        .unwrap_err();

    assert!(
        matches!(&err, Error::ModelRegistry(m)
            if m == "Could not save the model registry to database:Model: database unavailable"),
        "{err}"
    );
    assert_eq!(ids(rust_llm::models().all()), ["old-model"]);
}

// spec: models/registry_spec.rb:246 RubyLLM::Models > persists a candidate before changing the in-memory registry
#[tokio::test]
async fn refresh_persists_a_candidate_before_changing_the_registry() {
    let _lock = LOCK.lock().await;
    let restore = Restore::new("candidate");
    let server = catalog(json!([new_model()]), None).await;
    let store = Arc::new(TestStore::new(Vec::new()));
    let config = with_store(refresh_config(&server, &restore), store.clone());
    Models::install(vec![old_model()]);

    refresh::refresh_with_config(Arc::new(config), false)
        .await
        .unwrap();

    let seen = store.seen.lock().unwrap().clone();
    assert_eq!(
        seen,
        [(vec!["new-model".to_string()], vec!["old-model".to_string()])]
    );
    assert_eq!(ids(rust_llm::models().all()), ["new-model"]);
    // A store takes precedence over the registry file, which is left alone.
    assert!(!restore.file("models.json").exists());
}

// spec: models/registry_spec.rb:264 RubyLLM::Models > adopts what the store reports, keeping the unlisted models the merge dropped
#[tokio::test]
async fn refresh_adopts_what_the_store_reports() {
    let _lock = LOCK.lock().await;
    let restore = Restore::new("adopt");
    let server = catalog(json!([new_model()]), None).await;
    let unlisted = model(json!({
        "id": "old-model", "name": "Old Model", "provider": "openai",
        "unlisted_at": "2026-09-01 00:00:00 UTC"
    }));
    let store = Arc::new(TestStore::new(vec![new_model(), unlisted]));
    let config = with_store(refresh_config(&server, &restore), store);
    Models::install(vec![old_model()]);

    let registry = refresh::refresh_with_config(Arc::new(config), false)
        .await
        .unwrap();

    assert_eq!(ids(registry.all()), ["new-model"]);
    assert_eq!(ids(registry.unlisted()), ["old-model"]);
    assert_eq!(
        registry.find("old-model", Some("openai")).unwrap().id,
        "old-model"
    );
}

// ---- models/lookup_spec.rb -----------------------------------------------------------------------

fn original() -> Model {
    model(json!({ "id": "gpt-5-nano", "provider": "openai", "name": "Original" }))
}

fn replacement() -> Model {
    model(json!({ "id": "gpt-5-nano", "provider": "openai", "name": "Updated" }))
}

fn retired() -> Model {
    model(json!({ "id": "gpt-4.1", "provider": "openai", "name": "gpt-4.1" }))
}

// spec: models/lookup_spec.rb:68 #load_from_json > replaces previously found entries and removes entries absent from the file
#[tokio::test]
async fn load_from_json_replaces_found_entries() {
    let _lock = LOCK.lock().await;
    let restore = Restore::new("lookup_json");
    let mut registry = Models::new(vec![original(), retired()]);
    assert_eq!(registry.find("gpt-5-nano", None).unwrap(), original());
    assert_eq!(registry.find("gpt-4.1", None).unwrap(), retired());

    let file = restore.file("models.json");
    std::fs::write(&file, registry::pretty_json(&[&replacement()]).unwrap()).unwrap();
    registry.load_from_json(Some(&file));

    assert_eq!(registry.find("gpt-5-nano", None).unwrap().name, "Updated");
    assert_eq!(
        registry.find("gpt-5-nano", Some("openai")).unwrap().name,
        "Updated"
    );
    assert!(matches!(
        registry.find("gpt-4.1", None),
        Err(Error::ModelNotFound(_))
    ));
}

// spec: models/lookup_spec.rb:86 #load_from_store > invalidates previous lookups when a store reuses its array
#[tokio::test]
async fn load_from_store_invalidates_previous_lookups() {
    let _lock = LOCK.lock().await;
    let _restore = Restore::new("lookup_store");
    let store = Arc::new(TestStore::new(vec![original(), retired()]));
    set_global(with_store(Config::default(), store.clone()));
    let mut registry = Models::new(vec![original(), retired()]);

    registry.load_from_store().unwrap();
    assert_eq!(registry.find("gpt-5-nano", None).unwrap(), original());

    *store.stored.lock().unwrap() = vec![replacement()];
    registry.load_from_store().unwrap();
    assert_eq!(registry.find("gpt-5-nano", None).unwrap(), replacement());
    assert!(matches!(
        registry.find("gpt-4.1", Some("openai")),
        Err(Error::ModelNotFound(_))
    ));
}

// spec: models/lookup_spec.rb:98 #load_from_store > does not reuse an index built before a concurrent reload
// Ruby races a lookup against a reload of the same object. Here a registry is immutable once
// shared: a lookup that took the registry before the reload keeps answering from it, and every
// lookup after the reload sees the new entries.
#[tokio::test]
async fn a_lookup_in_flight_keeps_its_registry_across_a_reload() {
    let _lock = LOCK.lock().await;
    let _restore = Restore::new("lookup_concurrent");
    set_global(with_store(
        Config::default(),
        Arc::new(TestStore::new(vec![replacement()])),
    ));
    Models::install(vec![original(), retired()]);
    let in_flight = rust_llm::models();

    let lookup = std::thread::spawn(move || in_flight.find("gpt-5-nano", None).unwrap());
    let mut reloaded = (*rust_llm::models()).clone();
    reloaded.load_from_store().unwrap();
    Models::install(reloaded.all_including_unlisted().to_vec());

    assert_eq!(
        rust_llm::models().find("gpt-5-nano", None).unwrap(),
        replacement()
    );
    assert_eq!(lookup.join().unwrap(), original());
    assert_eq!(
        rust_llm::models().find("gpt-5-nano", None).unwrap(),
        replacement()
    );
}

// spec: models/lookup_spec.rb:133 #refresh > finds refreshed entries and unlisted entries retained by the store
#[tokio::test]
async fn refresh_finds_refreshed_and_retained_unlisted_entries() {
    let _lock = LOCK.lock().await;
    let restore = Restore::new("lookup_refresh");
    let server = catalog(json!([replacement()]), None).await;
    let mut unlisted = retired();
    unlisted.unlisted_at = Some("2026-09-01 00:00:00 UTC".into());
    let store = Arc::new(TestStore::new(vec![replacement(), unlisted.clone()]));
    let config = with_store(refresh_config(&server, &restore), store);
    Models::install(vec![original(), retired()]);
    assert_eq!(
        rust_llm::models().find("gpt-5-nano", None).unwrap(),
        original()
    );
    assert_eq!(rust_llm::models().find("gpt-4.1", None).unwrap(), retired());

    refresh::refresh_with_config(Arc::new(config), false)
        .await
        .unwrap();

    let registry = rust_llm::models();
    assert_eq!(registry.find("gpt-5-nano", None).unwrap(), replacement());
    assert_eq!(registry.find("gpt-4.1", Some("openai")).unwrap(), unlisted);
    assert_eq!(registry.all(), [&replacement()]);
}

// spec: models/lookup_spec.rb:149 #refresh_from_providers > replaces previous lookup results with the new provider catalog
#[tokio::test]
async fn refresh_from_providers_replaces_previous_lookups() {
    let _lock = LOCK.lock().await;
    let _restore = Restore::new("lookup_providers");
    // OpenAI now lists only gpt-5-nano, and models.dev names it "Updated".
    let openai = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(
            json!({ "data": [{ "id": "gpt-5-nano", "object": "model", "owned_by": "system" }] }),
        ))
        .mount(&openai)
        .await;
    let models_dev = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api.json"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "openai": { "models": { "gpt-5-nano": { "id": "gpt-5-nano", "name": "Updated" } } }
        })))
        .mount(&models_dev)
        .await;
    let mut config = Config::default();
    config.set("openai_api_key", "test");
    config.set("openai_api_base", format!("{}/v1", openai.uri()));
    config.set("models_dev_url", format!("{}/api.json", models_dev.uri()));
    set_global(config);
    Models::install(vec![original(), retired()]);
    assert_eq!(
        rust_llm::models().find("gpt-5-nano", None).unwrap(),
        original()
    );

    let registry = refresh::refresh_from_providers(true).await.unwrap();

    assert_eq!(
        registry.find("gpt-5-nano", Some("openai")).unwrap().name,
        "Updated"
    );
    assert!(matches!(
        registry.find("gpt-4.1", None),
        Err(Error::ModelNotFound(_))
    ));
}

// ---- models_spec.rb -------------------------------------------------------------------------------

// spec: models_spec.rb:71 filtering and chaining > filters by vision support
#[tokio::test]
async fn filters_by_vision_support() {
    let _lock = LOCK.lock().await;
    let registry = rust_llm::models();
    let vision: Vec<&Model> = registry
        .all()
        .into_iter()
        .filter(|m| m.supports("vision"))
        .collect();
    assert!(!vision.is_empty());
    assert!(vision.iter().all(|m| m.supports("vision")));
}

// spec: models_spec.rb:77 filtering and chaining > filters by video support
#[tokio::test]
async fn filters_by_video_support() {
    let _lock = LOCK.lock().await;
    let registry = rust_llm::models();
    let video: Vec<&Model> = registry
        .all()
        .into_iter()
        .filter(|m| m.supports("video"))
        .collect();
    assert!(!video.is_empty());
    assert!(video.iter().all(|m| m.supports("video")));
}

// spec: models_spec.rb:82 filtering and chaining > finds transcription support in the bundled registry
#[tokio::test]
async fn finds_transcription_support_in_the_bundled_registry() {
    let _lock = LOCK.lock().await;
    let registry = rust_llm::models();
    assert!(
        registry
            .find("whisper-1", Some("openai"))
            .unwrap()
            .supports("transcription")
    );
    assert!(
        registry
            .find("gemini-2.5-flash", Some("gemini"))
            .unwrap()
            .supports("transcription")
    );
}

// spec: models_spec.rb:187 #refresh > updates models and returns a chainable Models instance
#[tokio::test]
async fn refresh_returns_a_registry_to_chain_on() {
    let _lock = LOCK.lock().await;
    let restore = Restore::new("refresh_chain");
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/models.json"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(BUNDLE, "application/json"))
        .mount(&server)
        .await;
    let config = Arc::new(refresh_config(&server, &restore));

    let chat_models: Vec<Model> = refresh::refresh_with_config(config, false)
        .await
        .unwrap()
        .chat_models()
        .into_iter()
        .cloned()
        .collect();

    assert!(
        chat_models
            .iter()
            .all(|m| m.model_type() == rust_llm::model::ModelType::Chat)
    );
    let providers: Vec<&str> = chat_models.iter().map(|m| m.provider.as_str()).collect();
    assert!(providers.contains(&"openai") && providers.contains(&"anthropic"));
}

// spec: models_spec.rb:200 #refresh > works as a class method too
#[tokio::test]
async fn refresh_works_through_the_global_registry() {
    let _lock = LOCK.lock().await;
    let restore = Restore::new("refresh_global");
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/models.json"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(BUNDLE, "application/json"))
        .mount(&server)
        .await;
    set_global(refresh_config(&server, &restore));
    Models::install(Vec::new());

    rust_llm::models::refresh(false).await.unwrap();

    assert!(!rust_llm::models().all().is_empty());
}

// spec: models_spec.rb:481 #audio_models > filters to models that are audio-capable
#[tokio::test]
async fn audio_models_have_audio_output() {
    let _lock = LOCK.lock().await;
    let registry = rust_llm::models();
    let audio = registry.audio_models();
    assert!(!audio.is_empty());
    assert!(audio.iter().all(|m| {
        m.model_type() == rust_llm::model::ModelType::Audio
            || m.modalities.output.iter().any(|o| o == "audio")
    }));
}

// spec: models_spec.rb:496 #image_models > filters to models that are image-capable
#[tokio::test]
async fn image_models_have_image_output() {
    let _lock = LOCK.lock().await;
    let registry = rust_llm::models();
    let images = registry.image_models();
    assert!(!images.is_empty());
    assert!(images.iter().all(|m| {
        m.model_type() == rust_llm::model::ModelType::Image
            || m.modalities.output.iter().any(|o| o == "image")
    }));
}

// spec: models_spec.rb:511 #by_family > filters models by family
#[tokio::test]
async fn by_family_filters_models_by_family() {
    let _lock = LOCK.lock().await;
    let registry = rust_llm::models();
    let family = registry.all()[0].family.clone().unwrap();
    let in_family = registry.by_family(&family);
    assert!(!in_family.is_empty());
    assert!(
        in_family
            .iter()
            .all(|m| m.family.as_deref() == Some(family.as_str()))
    );
}

// spec: models_spec.rb:595 #resolve > uses registry metadata before dynamic-provider fallback models
// Ollama Cloud is a provider whose models are assumed to exist, like the spec's dynamic provider.
#[tokio::test]
async fn assumed_providers_use_registry_metadata_first() {
    let _lock = LOCK.lock().await;
    let _restore = Restore::new("resolve_dynamic");
    let registry_model = model(json!({
        "id": "remote-dynamic-chat", "name": "Remote Dynamic Chat", "provider": "ollama_cloud",
        "context_window": 128000, "max_output_tokens": 4096,
        "capabilities": ["function_calling", "streaming"],
        "pricing": { "text_tokens": { "standard": { "input_per_million": 0.1, "output_per_million": 0.2 } } },
        "metadata": { "source": "models.dev" }
    }));
    assert!(Provider::OllamaCloud.assume_models_exist());
    Models::install(vec![registry_model.clone()]);
    let mut config = Config::default();
    config.set("ollama_cloud_api_key", "test");

    let chat = Chat::with_config(
        Arc::new(config),
        Some("remote-dynamic-chat"),
        Some("ollama_cloud"),
        false,
    )
    .unwrap();

    assert_eq!(chat.model(), &registry_model);
    assert_eq!(chat.model().context_window, Some(128_000));
    assert_eq!(
        Value::Object(chat.model().metadata.clone()),
        json!({ "source": "models.dev" })
    );
    assert_eq!(chat.provider(), Provider::OllamaCloud);
}

// spec: models_spec.rb:636 #save_to_json > saves models to the models.json file
#[tokio::test]
async fn save_to_json_saves_every_listed_model() {
    let _lock = LOCK.lock().await;
    let restore = Restore::new("save");
    let file = restore.file("models.json");
    let registry = rust_llm::models();

    registry.save_to_json(Some(&file)).unwrap();

    let saved: Value = serde_json::from_str(&std::fs::read_to_string(&file).unwrap()).unwrap();
    assert_eq!(saved.as_array().unwrap().len(), registry.all().len());
}

// spec: models_spec.rb:654 #save_to_json > saves and loads from a custom file path
#[tokio::test]
async fn save_to_json_round_trips_through_a_custom_path() {
    let _lock = LOCK.lock().await;
    let restore = Restore::new("save_custom");
    let file = restore.file("custom_models.json");
    let registry = rust_llm::models();

    registry.save_to_json(Some(&file)).unwrap();
    let reloaded = rust_llm::models::models_from_file(Some(&file)).unwrap();

    assert_eq!(reloaded.len(), registry.all().len());
    assert_eq!(reloaded[0].id, registry.all()[0].id);
}

// ---- models_refresh_spec.rb: refresh models output structure --------------------------------------

/// The bundled catalog, plus OpenAI and Anthropic listing one test model each
/// (`mock_provider_models`).
async fn refresh_with_provider_models(restore: &Restore) -> (Arc<Models>, Vec<MockServer>) {
    let published = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/models.json"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(BUNDLE, "application/json"))
        .mount(&published)
        .await;
    let openai = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(
            json!({ "data": [{ "id": "test-model-1", "object": "model", "owned_by": "system" }] }),
        ))
        .mount(&openai)
        .await;
    let anthropic = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [{ "id": "test-model-2", "display_name": "Test Model 2", "type": "model" }],
            "has_more": false
        })))
        .mount(&anthropic)
        .await;
    let mut config = refresh_config(&published, restore);
    config.set("openai_api_key", "test");
    config.set("openai_api_base", format!("{}/v1", openai.uri()));
    config.set("anthropic_api_key", "test");
    config.set("anthropic_api_base", anthropic.uri());
    let registry = refresh::refresh_with_config(Arc::new(config), false)
        .await
        .unwrap();
    (registry, vec![published, openai, anthropic])
}

// spec: models_refresh_spec.rb:153 refresh models output structure > returns models with consistent structure
#[tokio::test]
async fn refresh_returns_models_with_consistent_structure() {
    let _lock = LOCK.lock().await;
    let restore = Restore::new("structure");
    let (registry, _servers) = refresh_with_provider_models(&restore).await;

    assert!(registry.find("test-model-1", Some("openai")).is_ok());
    assert!(registry.find("test-model-2", Some("anthropic")).is_ok());
    // Every entry survives a round trip through the registry's JSON form unchanged.
    for m in registry.all() {
        let again: Model = serde_json::from_value(serde_json::to_value(m).unwrap()).unwrap();
        assert_eq!(&again, m);
    }
}

// spec: models_refresh_spec.rb:166 refresh models output structure > saves models with correct JSON structure
#[tokio::test]
async fn refresh_saves_models_with_correct_json_structure() {
    let _lock = LOCK.lock().await;
    let restore = Restore::new("structure_saved");
    let (registry, _servers) = refresh_with_provider_models(&restore).await;
    let file = restore.file("test_models.json");

    registry.save_to_json(Some(&file)).unwrap();

    let saved: Value = serde_json::from_str(&std::fs::read_to_string(&file).unwrap()).unwrap();
    let saved = saved.as_array().unwrap();
    assert_eq!(saved.len(), registry.all().len());
    for m in saved {
        assert!(m["capabilities"].is_array(), "{m}");
        assert!(m["modalities"].is_object(), "{m}");
        assert!(m["pricing"].is_object(), "{m}");
    }
}

// ---- models_local_refresh_spec.rb -----------------------------------------------------------------

/// An Ollama server listing `test-model` whose `/api/show` reports tools.
async fn ollama_server() -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(
            json!({ "data": [{ "id": "test-model", "created": 1234567890, "owned_by": "library" }] }),
        ))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/show"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({ "capabilities": ["completion", "tools"] })),
        )
        .mount(&server)
        .await;
    server
}

async fn ollama_listings(server: &MockServer) -> usize {
    server
        .received_requests()
        .await
        .unwrap_or_default()
        .iter()
        .filter(|r| r.url.path() == "/v1/models")
        .count()
}

// spec: models_local_refresh_spec.rb:19 local provider model fetching > .refresh > with default parameters > includes local providers
#[tokio::test]
async fn refresh_includes_local_providers_by_default() {
    let _lock = LOCK.lock().await;
    let restore = Restore::new("local_default");
    let published = catalog(json!([new_model()]), None).await;
    let ollama = ollama_server().await;
    let mut config = refresh_config(&published, &restore);
    config.set("ollama_api_base", format!("{}/v1", ollama.uri()));

    let registry = refresh::refresh_with_config(Arc::new(config), false)
        .await
        .unwrap();

    assert_eq!(ollama_listings(&ollama).await, 1);
    assert!(registry.find("test-model", Some("ollama")).is_ok());
}

// spec: models_local_refresh_spec.rb:30 local provider model fetching > .refresh > with remote_only: true > excludes local providers
#[tokio::test]
async fn remote_only_refresh_never_asks_local_providers() {
    let _lock = LOCK.lock().await;
    let restore = Restore::new("local_remote_only");
    let published = catalog(json!([new_model()]), None).await;
    let ollama = ollama_server().await;
    let mut config = refresh_config(&published, &restore);
    config.set("ollama_api_base", format!("{}/v1", ollama.uri()));

    refresh::refresh_with_config(Arc::new(config), true)
        .await
        .unwrap();

    assert_eq!(ollama_listings(&ollama).await, 0);
}

// spec: models_local_refresh_spec.rb:50 local provider model fetching > .fetch_provider_models > can include local providers with remote_only: false
#[tokio::test]
async fn fetch_provider_models_includes_local_providers_when_asked() {
    let ollama = ollama_server().await;
    let mut config = Config::default();
    config.set("ollama_api_base", format!("{}/v1", ollama.uri()));
    let config = Arc::new(config);

    let local = refresh::fetch_provider_models(&config, false).await;
    assert_eq!(local.configured_names, ["Ollama"]);
    assert_eq!(local.fetched_providers, ["ollama"]);

    let remote = refresh::fetch_provider_models(&config, true).await;
    assert!(remote.configured_names.is_empty());
}

// spec: models_local_refresh_spec.rb:62 local provider model fetching > Ollama models integration > responds to list_models
#[tokio::test]
async fn ollama_lists_its_models() {
    let ollama = ollama_server().await;
    let mut config = Config::default();
    config.set("ollama_api_base", format!("{}/v1", ollama.uri()));
    let models = refresh::list_models(Provider::Ollama, Arc::new(config))
        .await
        .unwrap();
    assert_eq!(ids(models.iter().collect()), ["test-model"]);
}

// spec: models_local_refresh_spec.rb:66 local provider model fetching > Ollama models integration > can parse list models response
#[test]
fn ollama_parses_a_list_models_response() {
    let body = json!({ "data": [{ "id": "llama3:latest", "created": 1234567890, "owned_by": "library" }] });
    let models = refresh::parse_ollama_models(&body, "ollama", &Default::default(), false);
    assert_eq!(models[0].id, "llama3:latest");
    assert_eq!(models[0].provider, "ollama");
    assert_eq!(models[0].capabilities, ["streaming", "structured_output"]);
}

// spec: models_local_refresh_spec.rb:93 local provider model fetching > GPUStack models integration > responds to list_models
#[tokio::test]
async fn gpustack_lists_its_models() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "data": [] })))
        .mount(&server)
        .await;
    let mut config = Config::default();
    config.set("gpustack_api_base", format!("{}/v1", server.uri()));
    let models = refresh::list_models(Provider::GPUStack, Arc::new(config))
        .await
        .unwrap();
    assert!(models.is_empty());
    assert!(!server.received_requests().await.unwrap().is_empty());
}

// spec: models_local_refresh_spec.rb:99 local provider model fetching > local provider model resolution > assumes model exists for Ollama without warning after refresh
#[tokio::test]
async fn ollama_models_come_from_the_registry_after_a_refresh() {
    let _lock = LOCK.lock().await;
    let restore = Restore::new("local_resolution");
    let published = catalog(json!([new_model()]), None).await;
    let ollama = ollama_server().await;
    let mut config = refresh_config(&published, &restore);
    config.set("ollama_api_base", format!("{}/v1", ollama.uri()));
    let config = Arc::new(config);
    refresh::refresh_with_config(config.clone(), false)
        .await
        .unwrap();

    let chat = Chat::with_config(config, Some("test-model"), Some("ollama"), false).unwrap();

    assert_eq!(chat.model().id, "test-model");
    // The refreshed entry, not the "Assuming model exists" stand-in.
    assert!(!chat.model().metadata.contains_key("warning"));
    assert!(chat.model().supports("function_calling"));
}

// spec: models_local_refresh_spec.rb:127 local provider model fetching > local provider model resolution > assumes model exists for GPUStack without checking registry
#[tokio::test]
async fn gpustack_assumes_models_exist() {
    let _lock = LOCK.lock().await;
    let mut config = Config::default();
    config.set("gpustack_api_base", "http://localhost:9/v1");
    let chat =
        Chat::with_config(Arc::new(config), Some("any-model"), Some("gpustack"), false).unwrap();
    assert_eq!(chat.model().id, "any-model");
    assert_eq!(chat.model().provider, "gpustack");
}

// ---- models_json_validation_spec.rb ---------------------------------------------------------------

/// A validator for the JSON Schema keywords `Models::Schema` uses (type, enum, anyOf, minimum,
/// format date, properties, required, additionalProperties, items), reporting data pointers.
fn validate(schema: &Value, data: &Value, pointer: &str, errors: &mut Vec<String>) {
    if let Some(any_of) = schema.get("anyOf").and_then(Value::as_array) {
        let ok = any_of.iter().any(|s| {
            let mut e = Vec::new();
            validate(s, data, pointer, &mut e);
            e.is_empty()
        });
        if !ok {
            errors.push(format!("{pointer}: matches none of anyOf"));
        }
    }
    if let Some(kind) = schema.get("type").and_then(Value::as_str) {
        let ok = match kind {
            "object" => data.is_object(),
            "array" => data.is_array(),
            "string" => data.is_string(),
            "integer" => data.is_i64() || data.is_u64(),
            "number" => data.is_number(),
            "null" => data.is_null(),
            _ => false,
        };
        if !ok {
            errors.push(format!("{pointer}: is not a {kind}"));
            return;
        }
    }
    if let Some(allowed) = schema.get("enum").and_then(Value::as_array)
        && !allowed.contains(data)
    {
        errors.push(format!(
            "{pointer}: {data} is not one of the allowed values"
        ));
    }
    if let (Some(min), Some(n)) = (schema.get("minimum").and_then(Value::as_f64), data.as_f64())
        && n < min
    {
        errors.push(format!("{pointer}: {n} is below {min}"));
    }
    if schema.get("format").and_then(Value::as_str) == Some("date")
        && let Some(s) = data.as_str()
        && chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d").is_err()
    {
        errors.push(format!("{pointer}: {s} is not a date"));
    }
    if let Some(object) = data.as_object() {
        let properties = schema.get("properties").and_then(Value::as_object);
        for key in schema
            .get("required")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
        {
            if !object.contains_key(key) {
                errors.push(format!("{pointer}: is missing {key}"));
            }
        }
        for (key, value) in object {
            match properties.and_then(|p| p.get(key)) {
                Some(s) => validate(s, value, &format!("{pointer}/{key}"), errors),
                None if schema.get("additionalProperties") == Some(&Value::Bool(false)) => {
                    errors.push(format!("{pointer}: has unexpected property {key}"))
                }
                None => {}
            }
        }
    }
    if let (Some(items), Some(array)) = (schema.get("items"), data.as_array()) {
        for (i, item) in array.iter().enumerate() {
            validate(items, item, &format!("{pointer}/{i}"), errors);
        }
    }
}

// spec: models_json_validation_spec.rb:9 validates that models.json conforms to the schema
#[test]
fn bundled_models_json_conforms_to_the_schema() {
    // The schema is RubyLLM's own, as `Models::Schema.json_schema` writes it.
    let ruby: Value =
        serde_json::from_str(include_str!("fixtures/ruby_models_schema.json")).unwrap();
    assert_eq!(rust_llm::models::schema::json_schema(), ruby);

    let registry = json!({ "type": "array", "items": rust_llm::models::schema::json_schema() });
    let data: Value = serde_json::from_str(BUNDLE).unwrap();
    let mut errors = Vec::new();
    validate(&registry, &data, "", &mut errors);
    assert!(
        errors.is_empty(),
        "models.json has validation errors:\n{}",
        errors[..errors.len().min(20)].join("\n")
    );

    // The validator does catch a bad entry.
    let mut bad = data[0].clone();
    bad["capabilities"] = json!(["telepathy"]);
    bad["knowledge_cutoff"] = json!("someday");
    let mut errors = Vec::new();
    validate(&registry, &json!([bad]), "", &mut errors);
    assert_eq!(errors.len(), 2, "{errors:?}");
}

// ---- model_spec.rb ---------------------------------------------------------------------------------

fn model_spec_data() -> Value {
    json!({
        "id": "gpt-5", "name": "GPT-5", "provider": "openai", "family": "gpt",
        "created_at": "2026-02-20 00:00:00 UTC",
        "context_window": 400000, "max_output_tokens": 128000,
        "knowledge_cutoff": "2025-10-01",
        "modalities": { "input": ["text", "image"], "output": ["text"] },
        "capabilities": ["function_calling", "streaming", "vision", "structured_output"],
        "metadata": {
            "description": "A test model",
            "reasoning_options": [
                { "type": "effort", "values": ["low", "medium", "high"] },
                { "type": "budget_tokens", "min": 1024 }
            ]
        }
    })
}

fn merged(overrides: Value) -> Model {
    let mut data = model_spec_data();
    for (k, v) in overrides.as_object().unwrap() {
        data[k] = v.clone();
    }
    model(data)
}

// spec: model_spec.rb:43 #initialize > parses created_at and knowledge_cutoff
#[test]
fn parses_created_at_and_knowledge_cutoff() {
    let m = model(model_spec_data());
    assert_eq!(
        m.created_at_time(),
        chrono::DateTime::parse_from_rfc3339("2026-02-20T00:00:00Z")
            .ok()
            .map(|t| t.to_utc())
    );
    assert_eq!(
        m.knowledge_cutoff_date(),
        chrono::NaiveDate::from_ymd_opt(2025, 10, 1)
    );
}

// spec: model_spec.rb:48 #initialize > normalizes time to UTC
#[test]
fn normalizes_time_to_utc() {
    let m = model(json!({
        "id": "x", "name": "x", "provider": "openai",
        "created_at": "2026-02-20 00:00:00 +0700"
    }));
    assert_eq!(m.created_at.as_deref(), Some("2026-02-19 17:00:00 UTC"));
    assert_eq!(
        m.created_at_time(),
        chrono::DateTime::parse_from_rfc3339("2026-02-19T17:00:00Z")
            .ok()
            .map(|t| t.to_utc())
    );
}

// spec: model_spec.rb:127 #reasoning_options > accepts top-level reasoning options and stores them in metadata
#[test]
fn accepts_top_level_reasoning_options() {
    let m = merged(json!({
        "reasoning_options": [{ "type": "effort", "values": ["low", "high"] }],
        "metadata": {}
    }));
    let expected = json!([{ "type": "effort", "values": ["low", "high"] }]);
    assert_eq!(
        Value::Array(
            m.reasoning_options()
                .into_iter()
                .map(Value::Object)
                .collect()
        ),
        expected
    );
    assert_eq!(m.metadata["reasoning_options"], expected);
}

// spec: model_spec.rb:155 #reasoning_options > prefers top-level reasoning options over metadata when both are present
#[test]
fn prefers_top_level_reasoning_options() {
    let m = merged(json!({
        "reasoning_options": [{ "type": "effort", "values": ["low", "high"] }],
        "metadata": { "reasoning_options": [{ "type": "budget_tokens", "min": 1024 }] }
    }));
    assert_eq!(
        Value::Array(
            m.reasoning_options()
                .into_iter()
                .map(Value::Object)
                .collect()
        ),
        json!([{ "type": "effort", "values": ["low", "high"] }])
    );
}

// spec: model_spec.rb:308 #cost_for > hydrates long_context pricing from metadata.cost when pricing lacks it
#[test]
fn hydrates_long_context_pricing_from_metadata_cost() {
    let m = merged(json!({
        "pricing": { "text_tokens": { "standard": { "input_per_million": 5.0, "output_per_million": 30.0 } } },
        "metadata": { "cost": {
            "input": 5.0, "output": 30.0,
            "tiers": [{ "input": 10.0, "output": 45.0, "tier": { "type": "context", "size": 272000 } }]
        } }
    }));
    let text = serde_json::to_value(&m.pricing).unwrap()["text_tokens"].clone();
    assert_eq!(
        text["long_context"],
        json!({ "input_per_million": 10.0, "output_per_million": 45.0 })
    );
    assert_eq!(text["long_context_threshold"], 272000);
}

// ---- cost_spec.rb ------------------------------------------------------------------------------------

fn priced_model() -> Model {
    model(json!({
        "id": "priced-model", "name": "Priced Model", "provider": "openai",
        "pricing": { "text_tokens": { "standard": {
            "input_per_million": 1.0, "output_per_million": 2.0,
            "cache_read_input_per_million": 0.25, "cache_write_input_per_million": 1.25
        } } }
    }))
}

// spec: cost_spec.rb:283 provider-reported cost > round-trips the reported total through to_h
#[test]
fn round_trips_the_reported_total_through_to_h() {
    let tokens = Tokens {
        input: Some(10),
        output: Some(5),
        reported_cost: Some(0.0042),
        ..Default::default()
    };
    let restored = Cost::from_h(&Cost::new(&tokens, None, Tier::Standard).to_h(), None);
    assert_eq!(restored.total(), Some(0.0042));
}

// spec: cost_spec.rb:348 .from_h > round-trips a live cost through to_h
#[test]
fn round_trips_a_live_cost_through_to_h() {
    let tokens = Tokens {
        input: Some(1_000),
        output: Some(2_000),
        ..Default::default()
    };
    let live = Cost::new(&tokens, Some(&priced_model()), Tier::Standard);
    let restored = Cost::from_h(&live.to_h(), None);
    assert_eq!(restored.to_h(), live.to_h());
    assert_eq!(restored.total(), live.total());
    assert_eq!(
        live.to_h(),
        json!({ "input": 0.001, "output": 0.004, "total": 0.005 })
    );
}

// spec: cost_spec.rb:449 #to_h > omits the components that were never priced
#[test]
fn to_h_omits_components_that_were_never_priced() {
    let tokens = Tokens {
        input: Some(1_000),
        ..Default::default()
    };
    let h = Cost::new(&tokens, Some(&priced_model()), Tier::Standard).to_h();
    let mut keys: Vec<&String> = h.as_object().unwrap().keys().collect();
    keys.sort();
    assert_eq!(keys, ["input", "total"]);
}

// ---- support/utils_spec.rb -------------------------------------------------------------------------

// spec: support/utils_spec.rb:25 .underscore > separates words and keeps acronyms together
#[test]
fn underscore_separates_words_and_keeps_acronyms_together() {
    use rust_llm::tool::underscore;
    assert_eq!(underscore("MyTool"), "my_tool");
    assert_eq!(underscore("HTTPProxyTool"), "http_proxy_tool");
    assert_eq!(underscore("XMLHttpRequest"), "xml_http_request");
    assert_eq!(underscore("Tool2Name"), "tool2_name");
}

// spec: support/utils_spec.rb:102 .deep_merge > merges nested hashes without mutating the originals
#[test]
fn deep_merge_merges_nested_objects_without_touching_the_overrides() {
    let original = json!({ "config": { "retries": 3, "timeout": 5 }, "mode": "safe" });
    let overrides = json!({ "config": { "timeout": 10 }, "verbose": true });
    let mut result = original.clone();

    rust_llm::protocols::deep_merge(&mut result, &overrides);

    assert_eq!(
        result,
        json!({ "config": { "retries": 3, "timeout": 10 }, "mode": "safe", "verbose": true })
    );
    assert_eq!(
        original,
        json!({ "config": { "retries": 3, "timeout": 5 }, "mode": "safe" })
    );
    assert_eq!(
        overrides,
        json!({ "config": { "timeout": 10 }, "verbose": true })
    );
}
