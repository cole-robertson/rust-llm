//! Port of `lib/ruby_llm/models.rb`, `models/lookup.rb`, and `models/aliases.rb`: the bundled
//! registry, alias resolution, and provider preference. Refreshing lives in
//! [`mod@refresh`], registry files in [`registry`].

pub mod refresh;
pub mod registry;
pub mod schema;

pub use refresh::{
    ProviderFailure, last_provider_failures, list_models, refresh, refresh_from_providers,
};

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, RwLock};

use serde_json::{Map, Value};

use crate::error::{Error, Result};
use crate::model::Model;

/// `Models::PROVIDER_PREFERENCE`: which provider wins when a bare model id exists at several.
pub const PROVIDER_PREFERENCE: &[&str] = &[
    "openai",
    "anthropic",
    "gemini",
    "deepseek",
    "mistral",
    "cohere",
    "typesafe",
    "perplexity",
    "xai",
    "vertexai",
    "bedrock",
    "openrouter",
    "azure",
    "hetzner",
    "ollama_cloud",
    "ollama",
    "gpustack",
];

static BUNDLED_MODELS: &str = include_str!("../data/models.json");
static BUNDLED_ALIASES: &str = include_str!("../data/aliases.json");

static REGISTRY: LazyLock<RwLock<Arc<Models>>> =
    LazyLock::new(|| RwLock::new(Arc::new(Models::new(load_models()))));

/// `Models.load_models`: the configured `model_registry_store` when it holds models, else the
/// `model_registry_file` when it does, else the bundled registry.
pub fn load_models() -> Vec<Model> {
    let config = crate::config::try_config();
    config
        .as_ref()
        .and_then(|c| models_from_store(c.model_registry_store.as_deref()))
        .or_else(|| {
            config
                .as_ref()
                .and_then(|c| models_from_file(c.model_registry_file.as_deref()))
        })
        .unwrap_or_else(bundled_models)
}

/// `Models.models_from_store`: `None` without a store, or when it holds no models. A store that
/// cannot be read is logged and skipped, like an empty one.
pub fn models_from_store(store: Option<&dyn registry::ModelRegistryStore>) -> Option<Vec<Model>> {
    let models = match store?.read() {
        Ok(models) => models,
        Err(e) => {
            tracing::warn!("Could not read the model registry store: {e}");
            return None;
        }
    };
    if models.is_empty() {
        tracing::debug!("Model registry store is empty, falling back to the registry file");
        return None;
    }
    Some(models)
}

/// `Models.models_from_file`: `None` when there is no file, it is missing or empty, or it is
/// invalid (logged as a warning).
pub fn models_from_file(file: Option<&std::path::Path>) -> Option<Vec<Model>> {
    let file = file?;
    match registry::read(file) {
        Ok(models) => models.filter(|m| !m.is_empty()),
        Err(e) => {
            tracing::warn!(
                "Ignoring invalid model registry file {}: {e}",
                file.display()
            );
            None
        }
    }
}

/// `Models.models_from_bundle`.
pub fn bundled_models() -> Vec<Model> {
    serde_json::from_str(BUNDLED_MODELS).expect("bundled models.json parses")
}

static ALIASES: LazyLock<HashMap<String, Map<String, Value>>> =
    LazyLock::new(|| serde_json::from_str(BUNDLED_ALIASES).expect("bundled aliases.json parses"));

/// The model registry, `RubyLLM.models`.
#[derive(Debug, Clone)]
pub struct Models {
    models: Vec<Model>,
    by_id: HashMap<String, Vec<usize>>,
}

/// `RubyLLM.models`.
pub fn models() -> Arc<Models> {
    REGISTRY.read().unwrap().clone() // poisoned lock only
}

impl Models {
    pub fn new(models: Vec<Model>) -> Models {
        let mut by_id: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, m) in models.iter().enumerate() {
            by_id.entry(m.id.clone()).or_default().push(i);
        }
        Models { models, by_id }
    }

    /// Replaces the process-wide registry, e.g. after refreshing from providers.
    pub fn install(models: Vec<Model>) {
        *REGISTRY.write().unwrap() = Arc::new(Models::new(models)); // poisoned lock only
    }

    /// `Models#load_from_json`: replaces this registry with the models in `file` (the configured
    /// `model_registry_file` when `None`), or the bundled registry when it is missing or invalid.
    pub fn load_from_json(&mut self, file: Option<&std::path::Path>) -> &mut Self {
        let configured = crate::config().model_registry_file.clone();
        let models =
            models_from_file(file.or(configured.as_deref())).unwrap_or_else(bundled_models);
        *self = Models::new(models);
        self
    }

    /// `Models#load_from_store`: replaces this registry with what the configured
    /// `model_registry_store` holds.
    pub fn load_from_store(&mut self) -> Result<&mut Self> {
        let store = crate::config()
            .model_registry_store
            .clone()
            .ok_or_else(|| Error::ModelRegistry("No model registry store is configured".into()))?;
        *self = Models::new(store.read()?);
        Ok(self)
    }

    /// `Models#save_to_json`: writes the listed models to `file` (the configured
    /// `model_registry_file` when `None`) as pretty-printed JSON.
    pub fn save_to_json(&self, file: Option<&std::path::Path>) -> Result<&Self> {
        let configured = crate::config().model_registry_file.clone();
        let path = file
            .or(configured.as_deref())
            .ok_or_else(|| Error::ModelRegistry("A model registry file path is required".into()))?;
        registry::FileStore::new(path)?.write(&self.all(), None)?;
        Ok(self)
    }

    /// Every model, including unlisted ones (`all_including_unlisted`).
    pub fn all_including_unlisted(&self) -> &[Model] {
        &self.models
    }

    /// Listed models (`Models#all`).
    pub fn all(&self) -> Vec<&Model> {
        self.models.iter().filter(|m| !m.is_unlisted()).collect()
    }

    /// `Models#unlisted`: the models the provider has stopped listing (only a store keeps them).
    pub fn unlisted(&self) -> Vec<&Model> {
        self.models.iter().filter(|m| m.is_unlisted()).collect()
    }

    pub fn chat_models(&self) -> Vec<&Model> {
        self.all()
            .into_iter()
            .filter(|m| m.model_type() == crate::model::ModelType::Chat)
            .collect()
    }

    /// `Models#embedding_models`.
    pub fn embedding_models(&self) -> Vec<&Model> {
        self.all()
            .into_iter()
            .filter(|m| {
                m.model_type() == crate::model::ModelType::Embedding
                    || m.modalities.output.iter().any(|o| o == "embeddings")
            })
            .collect()
    }

    /// `Models#audio_models`: models with audio output.
    pub fn audio_models(&self) -> Vec<&Model> {
        self.all()
            .into_iter()
            .filter(|m| {
                m.model_type() == crate::model::ModelType::Audio
                    || m.modalities.output.iter().any(|o| o == "audio")
            })
            .collect()
    }

    /// `Models#image_models`: models with image output.
    pub fn image_models(&self) -> Vec<&Model> {
        self.all()
            .into_iter()
            .filter(|m| {
                m.model_type() == crate::model::ModelType::Image
                    || m.modalities.output.iter().any(|o| o == "image")
            })
            .collect()
    }

    /// `Models#by_family`.
    pub fn by_family(&self, family: &str) -> Vec<&Model> {
        self.all()
            .into_iter()
            .filter(|m| m.family.as_deref() == Some(family))
            .collect()
    }

    pub fn by_provider(&self, provider: &str) -> Vec<&Model> {
        self.all()
            .into_iter()
            .filter(|m| m.provider == provider)
            .collect()
    }

    fn candidates(&self, id: &str) -> Vec<&Model> {
        self.by_id
            .get(id)
            .map(|ix| ix.iter().map(|&i| &self.models[i]).collect())
            .unwrap_or_default()
    }

    /// `Models#find`.
    pub fn find(&self, model_id: &str, provider: Option<&str>) -> Result<Model> {
        match provider {
            Some(provider) => {
                let resolved = resolve_alias(model_id, Some(provider));
                self.candidates(&resolved)
                    .into_iter()
                    .find(|m| m.provider == provider)
                    .or_else(|| {
                        self.candidates(model_id)
                            .into_iter()
                            .find(|m| m.provider == provider)
                    })
                    .cloned()
                    .ok_or_else(|| not_found(model_id, Some(provider)))
            }
            None => {
                let resolved = resolve_alias(model_id, None);
                let mut matches = self.candidates(model_id);
                if resolved != model_id {
                    matches.extend(self.candidates(&resolved));
                }
                preferred_match(matches)
                    .cloned()
                    .ok_or_else(|| not_found(model_id, None))
            }
        }
    }
}

fn preferred_match(candidates: Vec<&Model>) -> Option<&Model> {
    if candidates.len() == 1 {
        return candidates.into_iter().next();
    }
    candidates.into_iter().min_by_key(|m| {
        let pref = PROVIDER_PREFERENCE
            .iter()
            .position(|p| *p == m.provider)
            .unwrap_or(PROVIDER_PREFERENCE.len());
        (m.is_unlisted() as u8, pref)
    })
}

fn not_found(model_id: &str, provider: Option<&str>) -> Error {
    let mut message = format!("Unknown model: {model_id:?}");
    if let Some(p) = provider {
        message = format!("{message} for provider: {p:?}");
    }
    Error::ModelNotFound(format!(
        "{message}. If the model exists at the provider, refresh the registry with `rust_llm::models::refresh(false).await`."
    ))
}

/// `Models::Aliases.resolve`.
pub fn resolve_alias(model_id: &str, provider: Option<&str>) -> String {
    let Some(entry) = ALIASES.get(model_id) else {
        return model_id.to_string();
    };
    let value = match provider {
        Some(p) => entry.get(p),
        None => entry.values().next(),
    };
    value
        .and_then(Value::as_str)
        .unwrap_or(model_id)
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bare_id_listed_at_several_providers_picks_by_provider_preference() {
        let model = models().find("claude-haiku-4-5", None).unwrap();
        assert_eq!(model.provider, "anthropic");
    }

    // The cassettes record claude-haiku-4-5 going out as claude-haiku-4-5-20251001.
    #[test]
    fn with_a_provider_an_alias_resolves_to_that_providers_own_id() {
        assert_eq!(
            models()
                .find("claude-haiku-4-5", Some("anthropic"))
                .unwrap()
                .id,
            "claude-haiku-4-5-20251001"
        );
        assert_eq!(
            models()
                .find("claude-haiku-4-5", Some("openrouter"))
                .unwrap()
                .id,
            "anthropic/claude-haiku-4.5"
        );
    }

    #[test]
    fn unknown_models_raise_model_not_found() {
        assert!(matches!(
            models().find("no-such-model", None),
            Err(Error::ModelNotFound(_))
        ));
    }
}
