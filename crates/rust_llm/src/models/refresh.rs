//! Port of `Models#refresh`, `Models.refresh_from_providers`, and the registry reconciliation in
//! `lib/ruby_llm/models.rb`, plus each provider's model listing (`Provider#list_models` through
//! `protocols/{chat_completions,anthropic,gemini,system_one}/models.rb` and
//! `providers/{mistral,openrouter,perplexity,xai,ollama,ollama_cloud,gpustack}/models.rb`) and the
//! capability augmenters in `providers/*/capabilities.rb`.
//!
//! ```ruby
//! RubyLLM.models.refresh
//! RubyLLM.models.refresh(remote_only: true).chat_models
//! ```
//!
//! ```ignore
//! rust_llm::models::refresh(false).await?;
//! rust_llm::models::refresh(true).await?.chat_models();
//! ```

use std::sync::{Arc, LazyLock, Mutex};

use serde_json::{Map, Value, json};

use super::{Models, bundled_models, models, registry};
use crate::config::Config;
use crate::error::{Error, Result};
use crate::instrumentation::{Event, payload};
use crate::model::{Modalities, Model, ModelType};
use crate::providers::{ALL, Provider};
use crate::transport::Connection;

/// `Models::MODELS_DEV_PROVIDER_MAP`.
const MODELS_DEV_PROVIDER_MAP: &[(&str, &str)] = &[
    ("openai", "openai"),
    ("anthropic", "anthropic"),
    ("azure", "azure"),
    ("google", "gemini"),
    ("google-vertex", "vertexai"),
    ("amazon-bedrock", "bedrock"),
    ("cohere", "cohere"),
    ("deepseek", "deepseek"),
    ("hetzner", "hetzner"),
    ("mistral", "mistral"),
    ("ollama-cloud", "ollama_cloud"),
    ("openrouter", "openrouter"),
    ("perplexity", "perplexity"),
    ("perplexity-agent", "perplexity"),
    ("xai", "xai"),
];
pub(crate) const MODELS_DEV_INPUT_MODALITIES: &[&str] =
    &["text", "image", "audio", "pdf", "video", "file"];
pub(crate) const MODELS_DEV_OUTPUT_MODALITIES: &[&str] = &[
    "text",
    "image",
    "audio",
    "video",
    "embeddings",
    "moderation",
    "rerank",
    "judgment",
];
/// models.dev's catalog. `config.set("models_dev_url", ...)` points it elsewhere.
pub const MODELS_DEV_URL: &str = "https://models.dev/api.json";

/// A provider whose model list could not be fetched during a refresh (`last_provider_failures`).
#[derive(Debug, Clone)]
pub struct ProviderFailure {
    pub name: String,
    pub slug: String,
    pub error: String,
}

static LAST_PROVIDER_FAILURES: LazyLock<Mutex<Vec<ProviderFailure>>> =
    LazyLock::new(Default::default);

/// `Models.last_provider_failures`: the providers the last refresh could not list. Their previous
/// models were kept.
pub fn last_provider_failures() -> Vec<ProviderFailure> {
    LAST_PROVIDER_FAILURES
        .lock()
        .map(|f| f.clone())
        .unwrap_or_default()
}

/// What `fetch_provider_models` found.
#[derive(Debug, Default)]
pub struct ProviderFetch {
    pub models: Vec<Model>,
    pub fetched_providers: Vec<String>,
    pub configured_names: Vec<String>,
    pub failed: Vec<ProviderFailure>,
    /// Providers that answered with no models; treated like ones that did not answer.
    pub empty: Vec<(String, String)>,
}

/// `Models.refresh(remote_only:)`: replaces the registry with the published catalog merged with
/// the models each configured provider lists, saves it to `config.model_registry_file`, and
/// returns the new registry. `remote_only` skips local providers (Ollama, GPUStack). Fails with
/// `Error::ModelRegistry`, leaving the registry unchanged, when the catalog cannot be fetched or the
/// result cannot be saved.
pub async fn refresh(remote_only: bool) -> Result<Arc<Models>> {
    refresh_with_config(crate::config(), remote_only).await
}

/// `refresh` with an explicit configuration (`context.models.refresh`).
pub async fn refresh_with_config(config: Arc<Config>, remote_only: bool) -> Result<Arc<Models>> {
    let mut event = Event::start(&config, "models.refresh.rust_llm", || {
        payload([("remote_only", remote_only.into())])
    });
    let result = async {
        let file = file_store(&config)?;
        let published = fetch_published_models(&config, file.as_ref()).await?;
        let catalog = published.models.clone().unwrap_or_default();
        let current = models();
        let merged = merge_discovered_models(&config, &current, &catalog, remote_only).await;
        persist(&config, file.as_ref(), &merged, &published)?;
        Models::install(stored_models(&config).unwrap_or(merged));
        Ok((models(), published.not_modified))
    }
    .await;
    if let Ok((registry, not_modified)) = &result {
        event.set("model_count", || registry.all().len().into());
        event.set("not_modified", || (*not_modified).into());
    }
    event.finish(result.as_ref().err());
    result.map(|(registry, _)| registry)
}

/// `Models.refresh_from_providers(remote_only:)`: rebuilds the registry from the providers and
/// models.dev directly, without the published catalog (the maintainer registry builder). Not saved.
pub async fn refresh_from_providers(remote_only: bool) -> Result<Arc<Models>> {
    let config = crate::config();
    let merged = fetch_merged_models(&config, remote_only).await;
    Models::install(merged);
    Ok(models())
}

/// `Models.fetch_merged_models`.
pub async fn fetch_merged_models(config: &Arc<Config>, remote_only: bool) -> Vec<Model> {
    let mut event = Event::start(config, "models.refresh.rust_llm", || {
        payload([("remote_only", remote_only.into())])
    });
    let existing = read_existing_models();
    let provider_fetch = fetch_provider_models(config, remote_only).await;
    record_failures(&provider_fetch);
    log_provider_fetch(&provider_fetch);
    event.set("failed_providers", || {
        json!(
            provider_fetch
                .failed
                .iter()
                .map(|f| f.slug.clone())
                .collect::<Vec<_>>()
        )
    });
    let models_dev = fetch_models_dev_models(config, &existing).await;
    log_models_dev_fetch(&models_dev);
    let merged = merge_with_existing(&existing, &provider_fetch, &models_dev);
    event.set("model_count", || merged.len().into());
    event.finish(None);
    merged
}

/// `Models.read_existing_models`: the registry's models, or the loaded registry when it is empty.
pub fn read_existing_models() -> Vec<Model> {
    let current = models();
    if current.all().is_empty() {
        super::load_models()
    } else {
        current.all().into_iter().cloned().collect()
    }
}

fn record_failures(fetch: &ProviderFetch) {
    if let Ok(mut f) = LAST_PROVIDER_FAILURES.lock() {
        *f = fetch.failed.clone();
    }
}

/// `Models.fetch_provider_models(remote_only:)`: lists every configured provider. A failure or an
/// empty answer is recorded, not raised, so the refresh keeps that provider's previous models.
pub async fn fetch_provider_models(config: &Arc<Config>, remote_only: bool) -> ProviderFetch {
    let providers: Vec<Provider> = ALL
        .iter()
        .copied()
        .filter(|p| p.is_configured(config) && !(remote_only && p.is_local()))
        .collect();
    let mut result = ProviderFetch {
        configured_names: providers.iter().map(|p| p.display().to_string()).collect(),
        ..Default::default()
    };
    for provider in providers {
        match list_models(provider, config.clone()).await {
            Ok(models) if models.is_empty() => result
                .empty
                .push((provider.display().into(), provider.slug().into())),
            Ok(models) => {
                result.models.extend(models);
                result.fetched_providers.push(provider.slug().into());
            }
            Err(e) => result.failed.push(ProviderFailure {
                name: provider.display().into(),
                slug: provider.slug().into(),
                error: e.to_string(),
            }),
        }
    }
    result
}

/// `Models.log_provider_fetch`.
pub fn log_provider_fetch(fetch: &ProviderFetch) {
    tracing::info!(
        "Fetching models from providers: {}",
        fetch.configured_names.join(", ")
    );
    for f in &fetch.failed {
        tracing::warn!(
            "Failed to fetch {} models ({}). Keeping existing.",
            f.name,
            f.error
        );
    }
    for (name, _) in &fetch.empty {
        tracing::warn!("{name} listed no models. Keeping existing.");
    }
}

/// `models.dev` models, and whether they came from a fresh fetch.
#[derive(Debug, Default)]
pub struct ModelsDevFetch {
    pub models: Vec<Model>,
    pub fetched: bool,
}

/// `Models.fetch_models_dev_models`: on any failure, keeps the models.dev entries already held.
pub async fn fetch_models_dev_models(config: &Config, existing: &[Model]) -> ModelsDevFetch {
    tracing::info!("Fetching models from models.dev API...");
    let url = config
        .get("models_dev_url")
        .unwrap_or(MODELS_DEV_URL)
        .to_string();
    let fetched = async {
        let client = reqwest::Client::builder()
            .timeout(config.request_timeout)
            .build()
            .map_err(|e| e.to_string())?;
        let resp = client.get(&url).send().await.map_err(|e| e.to_string())?;
        if !resp.status().is_success() {
            return Err(format!(
                "the server responded with status {}",
                resp.status().as_u16()
            ));
        }
        let body: Value = resp.json().await.map_err(|e| e.to_string())?;
        parse_models_dev_catalog(&body)
    }
    .await;
    match fetched {
        Ok(models) => ModelsDevFetch {
            models,
            fetched: true,
        },
        Err(e) => {
            tracing::warn!("Failed to fetch models.dev ({e}). Keeping existing.");
            ModelsDevFetch {
                models: models_dev_entries(existing),
                fetched: false,
            }
        }
    }
}

/// `Models.log_models_dev_fetch`: quiet on a fresh fetch, a warning when cached data stands in.
pub fn log_models_dev_fetch(fetch: &ModelsDevFetch) {
    if !fetch.fetched {
        tracing::warn!("Using cached models.dev data due to fetch failure.");
    }
}

fn models_dev_entries(models: &[Model]) -> Vec<Model> {
    models
        .iter()
        .filter(|m| m.metadata.get("source").and_then(Value::as_str) == Some("models.dev"))
        .cloned()
        .collect()
}

/// `Models.parse_models_dev_catalog`: only a catalog carrying models may overrule the registry.
pub fn parse_models_dev_catalog(body: &Value) -> std::result::Result<Vec<Model>, String> {
    let Some(catalog) = body.as_object() else {
        return Err(format!(
            "models.dev returned {} instead of a catalog",
            json_type(body)
        ));
    };
    let models: Vec<Model> = catalog
        .iter()
        .flat_map(|(key, data)| models_dev_provider_models(key, data))
        .collect();
    if models.is_empty() {
        return Err("models.dev returned no models RustLLM knows a provider for".into());
    }
    Ok(models)
}

fn json_type(v: &Value) -> &'static str {
    match v {
        Value::Null => "NilClass",
        Value::Array(_) => "Array",
        Value::String(_) => "String",
        Value::Bool(_) => "Boolean",
        Value::Number(_) => "Number",
        Value::Object(_) => "Hash",
    }
}

fn models_dev_provider_models(key: &str, data: &Value) -> Vec<Model> {
    let Some(&(_, slug)) = MODELS_DEV_PROVIDER_MAP.iter().find(|(k, _)| *k == key) else {
        return Vec::new();
    };
    let Some(entries) = data.get("models").and_then(Value::as_object) else {
        return Vec::new();
    };
    entries
        .values()
        .filter_map(|m| models_dev_model_attributes(m, slug, key))
        .collect()
}

/// `Models.models_dev_model_attributes`: one models.dev entry as a registry `Model` for the
/// provider `slug` (`key` is the models.dev provider key). `None` when the entry has no id.
pub fn models_dev_model_attributes(data: &Value, slug: &str, key: &str) -> Option<Model> {
    let raw_id = data.get("id").and_then(Value::as_str)?;
    let modalities = normalize_models_dev_modalities(data.get("modalities"));
    let capabilities = models_dev_capabilities(data, &modalities, slug, raw_id);
    let created = [data.get("release_date"), data.get("last_updated")]
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .find(|v| !v.trim().is_empty());
    let mut model = Model {
        id: models_dev_model_id(raw_id, slug),
        name: data
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or(raw_id)
            .to_string(),
        provider: slug.to_string(),
        family: data
            .get("family")
            .and_then(Value::as_str)
            .map(str::to_string),
        created_at: created.and_then(iso_date_prefix_to_utc_midnight_string),
        context_window: data.pointer("/limit/context").and_then(Value::as_i64),
        max_output_tokens: data.pointer("/limit/output").and_then(Value::as_i64),
        knowledge_cutoff: data
            .get("knowledge")
            .and_then(Value::as_str)
            .and_then(normalize_models_dev_knowledge),
        modalities,
        capabilities,
        pricing: serde_json::from_value(models_dev_pricing(data.get("cost"))).unwrap_or_default(),
        metadata: models_dev_metadata(data, key),
        unlisted_at: None,
    };
    normalize_embedding_modalities(&model.id, &mut model.modalities);
    Some(model)
}

/// `Models.models_dev_model_id` (`Provider.models_dev_model_id`): Vertex AI drops the `@version`
/// pin models.dev adds; every other provider keeps the id as is.
pub fn models_dev_model_id(id: &str, slug: &str) -> String {
    if slug == "vertexai" {
        id.split('@').next().unwrap_or(id).to_string()
    } else {
        id.to_string()
    }
}

/// `Support::Utils.parse_iso_date_prefix`: `YYYY-MM-DD`, `YYYY-MM` (first of the month), or
/// `YYYY` (first of the year); `None` for anything else or an impossible date.
pub fn parse_iso_date_prefix(value: &str) -> Option<chrono::NaiveDate> {
    let v = value.trim();
    let shape: String = v
        .chars()
        .map(|c| if c.is_ascii_digit() { '9' } else { c })
        .collect();
    let full = match shape.as_str() {
        "9999-99-99" => v.to_string(),
        "9999-99" => format!("{v}-01"),
        "9999" => format!("{v}-01-01"),
        _ => return None,
    };
    chrono::NaiveDate::parse_from_str(&full, "%Y-%m-%d").ok()
}

/// `Support::Utils.iso_date_prefix_to_utc_midnight_string`: `"YYYY-MM-DD 00:00:00 UTC"`.
pub fn iso_date_prefix_to_utc_midnight_string(value: &str) -> Option<String> {
    let date = parse_iso_date_prefix(value)?;
    Some(format!("{} 00:00:00 UTC", date.format("%Y-%m-%d")))
}

/// `Models.normalize_models_dev_knowledge`: `Date.parse`, `None` when it cannot be read as a date.
pub fn normalize_models_dev_knowledge(value: &str) -> Option<String> {
    let date = chrono::NaiveDate::parse_from_str(value.get(..10)?, "%Y-%m-%d").ok()?;
    Some(date.format("%Y-%m-%d").to_string())
}

/// `Models.normalize_models_dev_modalities`: only the modalities RubyLLM knows, in its order of
/// appearance.
pub fn normalize_models_dev_modalities(modalities: Option<&Value>) -> Modalities {
    let pick = |key: &str, allowed: &[&str]| -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for v in modalities
            .and_then(|m| m.get(key))
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
        {
            if allowed.contains(&v) && !out.iter().any(|o| o == v) {
                out.push(v.to_string());
            }
        }
        out
    };
    Modalities {
        input: pick("input", MODELS_DEV_INPUT_MODALITIES),
        output: pick("output", MODELS_DEV_OUTPUT_MODALITIES),
    }
}

fn truthy(v: Option<&Value>) -> bool {
    !matches!(v, None | Some(Value::Null) | Some(Value::Bool(false)))
}

fn models_dev_capabilities(
    data: &Value,
    modalities: &Modalities,
    slug: &str,
    raw_id: &str,
) -> Vec<String> {
    let mut caps: Vec<String> = Vec::new();
    if truthy(data.get("tool_call")) {
        caps.push("function_calling".into());
    }
    if truthy(data.get("structured_output")) {
        caps.push("structured_output".into());
    }
    if truthy(data.get("reasoning")) || truthy(data.get("reasoning_options")) {
        caps.push("reasoning".into());
    }
    if modalities
        .input
        .iter()
        .any(|m| ["image", "video", "pdf"].contains(&m.as_str()))
    {
        caps.push("vision".into());
    }
    if modalities.input.iter().any(|m| m == "video") {
        caps.push("video".into());
    }
    augment_capabilities(slug, caps, raw_id, modalities)
}

/// `Models.models_dev_pricing`.
pub fn models_dev_pricing(cost: Option<&Value>) -> Value {
    let Some(cost) = cost.filter(|c| c.is_object()) else {
        return json!({});
    };
    let rates = |pairs: &[(&str, &str)], source: &Value| -> Map<String, Value> {
        pairs
            .iter()
            .filter_map(|(from, to)| {
                source
                    .get(*from)
                    .and_then(Value::as_f64)
                    .map(|v| (to.to_string(), v.into()))
            })
            .collect()
    };
    const TEXT: &[(&str, &str)] = &[
        ("input", "input_per_million"),
        ("output", "output_per_million"),
        ("cache_read", "cache_read_input_per_million"),
        ("cache_write", "cache_write_input_per_million"),
        ("reasoning", "reasoning_output_per_million"),
    ];
    let text = rates(TEXT, cost);
    let audio = rates(
        &[
            ("input_audio", "input_per_million"),
            ("output_audio", "output_per_million"),
        ],
        cost,
    );
    let mut pricing = Map::new();
    // `PricingCategory.long_context_from_cost`: a `context` tier, or the older `context_over_200k`.
    let long = cost
        .get("tiers")
        .and_then(Value::as_array)
        .and_then(|t| {
            t.iter()
                .find(|e| e.pointer("/tier/type").and_then(Value::as_str) == Some("context"))
        })
        .map(|e| (e, e.pointer("/tier/size").and_then(Value::as_i64)))
        .or_else(|| {
            cost.get("context_over_200k")
                .filter(|c| c.is_object())
                .map(|e| (e, Some(200_000)))
        })
        .map(|(e, threshold)| (rates(TEXT, e), threshold))
        .filter(|(r, _)| !r.is_empty());
    if !text.is_empty() || long.is_some() {
        let mut t = Map::new();
        if !text.is_empty() {
            t.insert("standard".into(), Value::Object(text));
        }
        if let Some((rates, threshold)) = long {
            t.insert("long_context".into(), Value::Object(rates));
            if let Some(threshold) = threshold {
                t.insert("long_context_threshold".into(), threshold.into());
            }
        }
        pricing.insert("text_tokens".into(), Value::Object(t));
    }
    if !audio.is_empty() {
        pricing.insert("audio_tokens".into(), json!({ "standard": audio }));
    }
    Value::Object(pricing)
}

fn models_dev_metadata(data: &Value, key: &str) -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("source".into(), "models.dev".into());
    m.insert("provider_id".into(), key.into());
    for field in [
        "open_weights",
        "attachment",
        "temperature",
        "last_updated",
        "status",
        "interleaved",
        "tool_call",
        "structured_output",
        "reasoning",
        "reasoning_options",
        "cost",
        "limit",
        "knowledge",
    ] {
        if let Some(v) = data.get(field).filter(|v| !v.is_null()) {
            m.insert(field.into(), v.clone());
        }
    }
    m
}

fn normalize_embedding_modalities(id: &str, modalities: &mut Modalities) {
    if !id.contains("embedding") {
        return;
    }
    if modalities.input.is_empty() {
        modalities.input = vec!["text".into()];
    }
    modalities.output = vec!["embeddings".into()];
}

// ---- merging -------------------------------------------------------------------------------

/// `Models.merge_models`: one entry per `provider:id`, models.dev data enriched with what the
/// provider reports, sorted by provider and id.
pub fn merge_models(provider_models: &[Model], models_dev_models: &[Model]) -> Vec<Model> {
    let dev_by_key = index_by_key(models_dev_models);
    let provider_by_key = index_by_key(provider_models);
    let mut provider_by_alias: std::collections::HashMap<String, &Model> = Default::default();
    for m in provider_models {
        for alias in m
            .metadata
            .get("aliases")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
        {
            provider_by_alias
                .entry(format!("{}:{alias}", m.provider))
                .or_insert(m);
        }
    }
    let mut keys: Vec<String> = Vec::new();
    for k in dev_by_key.keys().chain(provider_by_key.keys()) {
        if !keys.contains(k) {
            keys.push(k.clone());
        }
    }
    let mut merged: Vec<Model> = keys
        .iter()
        .filter_map(|key| {
            let provider_model = provider_by_key
                .get(key)
                .or_else(|| provider_by_alias.get(key))
                .copied();
            let dev_model =
                find_models_dev_alias(key, |k| dev_by_key.get(k).copied(), provider_model);
            match (dev_model, provider_model) {
                (Some(dev), Some(p)) => Some(add_provider_metadata(&dev, p)),
                (Some(dev), None) => Some(dev),
                (None, Some(p)) => Some(augment_model_capabilities(p)),
                (None, None) => None,
            }
        })
        .collect();
    merged.sort_by(|a, b| (&a.provider, &a.id).cmp(&(&b.provider, &b.id)));
    merged
}

fn index_by_key(models: &[Model]) -> indexed::Map<'_> {
    let mut map = indexed::Map::default();
    for m in models {
        map.insert(format!("{}:{}", m.provider, m.id), m);
    }
    map
}

/// An insertion-ordered `key => model` map (Ruby's `to_h` keeps order; the last duplicate wins).
mod indexed {
    use crate::model::Model;
    #[derive(Default)]
    pub struct Map<'a>(Vec<(String, &'a Model)>);
    impl<'a> Map<'a> {
        pub fn insert(&mut self, k: String, v: &'a Model) {
            match self.0.iter_mut().find(|(key, _)| *key == k) {
                Some(slot) => slot.1 = v,
                None => self.0.push((k, v)),
            }
        }
        pub fn get(&self, k: &str) -> Option<&&'a Model> {
            self.0.iter().find(|(key, _)| key == k).map(|(_, v)| v)
        }
        pub fn keys(&self) -> impl Iterator<Item = &String> {
            self.0.iter().map(|(k, _)| k)
        }
    }
}

/// `Models.find_models_dev_model(key, models_dev_by_key, provider_model)`: a direct hit for
/// `"provider:id"`, else the provider's `models_dev_alias`.
pub fn find_models_dev_model(
    key: &str,
    models_dev_by_key: &std::collections::HashMap<String, Model>,
    provider_model: Option<&Model>,
) -> Option<Model> {
    find_models_dev_alias(key, |k| models_dev_by_key.get(k), provider_model)
}

fn find_models_dev_alias<'a>(
    key: &str,
    dev_by_key: impl Fn(&str) -> Option<&'a Model>,
    provider_model: Option<&Model>,
) -> Option<Model> {
    if let Some(m) = dev_by_key(key) {
        return Some(m.clone());
    }
    let (provider, model_id) = key.split_once(':')?;
    let with_id = |source: &Model| Model {
        id: model_id.to_string(),
        ..source.clone()
    };
    match provider {
        "openai" => {
            // `OpenAI::Models.models_dev_alias`: a snapshot reuses its base entry.
            if let Some((_, base)) = OPENAI_SNAPSHOT_BASES.iter().find(|(s, _)| *s == model_id)
                && let Some(source) = dev_by_key(&format!("openai:{base}"))
            {
                return Some(with_id(source));
            }
            let (base, date) = model_id.rsplit_once('-').and_then(|(rest, day)| {
                let (rest, month) = rest.rsplit_once('-')?;
                let (base, year) = rest.rsplit_once('-')?;
                let dated = year.len() == 4
                    && month.len() == 2
                    && day.len() == 2
                    && [year, month, day]
                        .iter()
                        .all(|p| p.chars().all(|c| c.is_ascii_digit()));
                dated.then(|| (base, format!("{year}-{month}-{day}")))
            })?;
            let source = dev_by_key(&format!("openai:{base}"))?;
            (source.created_at.as_deref()?.get(..10)? == date).then(|| with_id(source))
        }
        "mistral" => provider_model?
            .metadata
            .get("aliases")
            .and_then(Value::as_array)?
            .iter()
            .filter_map(Value::as_str)
            .find_map(|a| dev_by_key(&format!("mistral:{a}")))
            .map(with_id),
        "vertexai" => dev_by_key(&format!("gemini:{model_id}")).map(|s| Model {
            provider: "vertexai".into(),
            ..s.clone()
        }),
        _ => None,
    }
}

const OPENAI_SNAPSHOT_BASES: &[(&str, &str)] = &[
    ("gpt-3.5-turbo-0125", "gpt-3.5-turbo"),
    ("gpt-3.5-turbo-1106", "gpt-3.5-turbo"),
    ("gpt-4-0314", "gpt-4"),
    ("gpt-4-0613", "gpt-4"),
    ("gpt-4-turbo-2024-04-09", "gpt-4-turbo"),
    ("o1-2024-12-17", "o1"),
    ("o3-mini-2025-01-31", "o3-mini"),
];

/// `Models.blank_value?`.
pub fn is_blank(value: &Value) -> bool {
    match value {
        Value::Null => true,
        Value::String(s) => s.is_empty(),
        Value::Array(a) => a.is_empty(),
        Value::Object(o) => o.values().all(is_blank),
        _ => false,
    }
}

fn to_h(model: &Model) -> Map<String, Value> {
    match serde_json::to_value(model) {
        Ok(Value::Object(m)) => m,
        _ => Map::new(),
    }
}

/// `Models.add_provider_metadata`: models.dev data first, blanks filled from the provider.
pub fn add_provider_metadata(dev: &Model, provider_model: &Model) -> Model {
    let mut data = to_h(dev);
    let provided = to_h(provider_model);
    for key in [
        "name",
        "family",
        "created_at",
        "context_window",
        "max_output_tokens",
        "knowledge_cutoff",
        "modalities",
    ] {
        if data.get(key).is_none_or(is_blank) {
            data.insert(
                key.into(),
                provided.get(key).cloned().unwrap_or(Value::Null),
            );
        }
    }
    if dev.model_type() == ModelType::Chat && provider_model.model_type() != ModelType::Chat {
        data.insert("modalities".into(), provided["modalities"].clone());
    }
    let mut pricing = provided.get("pricing").cloned().unwrap_or(json!({}));
    crate::protocols::deep_merge(&mut pricing, data.get("pricing").unwrap_or(&json!({})));
    data.insert("pricing".into(), pricing);
    let mut metadata = provider_model.metadata.clone();
    metadata.extend(dev.metadata.clone());
    data.insert("metadata".into(), Value::Object(metadata));
    let mut merged: Model =
        serde_json::from_value(Value::Object(data)).unwrap_or_else(|_| dev.clone());
    merged.capabilities = merge_capabilities(dev, provider_model, &merged.modalities);
    normalize_embedding_modalities(&merged.id.clone(), &mut merged.modalities);
    merged
}

/// `Models.merge_capabilities`: models.dev only overrules the capabilities it reports on.
fn merge_capabilities(dev: &Model, provider_model: &Model, modalities: &Modalities) -> Vec<String> {
    let reported = models_dev_reported_capabilities(dev);
    let denied: Vec<&String> = reported
        .iter()
        .filter(|c| !dev.capabilities.contains(c))
        .collect();
    let mut caps: Vec<String> = Vec::new();
    for c in dev.capabilities.iter().chain(&provider_model.capabilities) {
        if !caps.contains(c) && !denied.contains(&c) {
            caps.push(c.clone());
        }
    }
    augment_capabilities(
        &provider_model.provider,
        caps,
        &provider_model.id,
        modalities,
    )
}

fn models_dev_reported_capabilities(dev: &Model) -> Vec<String> {
    let m = &dev.metadata;
    let present = |k: &str| m.get(k).is_some_and(|v| !v.is_null());
    let mut out = Vec::new();
    if present("tool_call") {
        out.push("function_calling".to_string());
    }
    if present("structured_output") {
        out.push("structured_output".into());
    }
    if present("reasoning") || present("reasoning_options") {
        out.push("reasoning".into());
    }
    if !dev.modalities.input.is_empty() {
        out.push("vision".into());
    }
    out
}

fn augment_model_capabilities(model: &Model) -> Model {
    let caps = augment_capabilities(
        &model.provider,
        model.capabilities.clone(),
        &model.id,
        &model.modalities,
    );
    Model {
        capabilities: caps,
        ..model.clone()
    }
}

fn union(mut caps: Vec<String>, additions: &[&str]) -> Vec<String> {
    for a in additions {
        if !caps.iter().any(|c| c == a) {
            caps.push(a.to_string());
        }
    }
    caps
}

const OPENAI_CHAT_MODELS: &[&str] = &["gpt-5-chat-latest", "gpt-5.1-chat-latest"];
const OPENAI_CODEX_MODELS: &[&str] = &[
    "gpt-5-codex",
    "gpt-5.1-codex",
    "gpt-5.1-codex-max",
    "gpt-5.1-codex-mini",
    "gpt-5.2-codex",
];
const OPENAI_SEARCH_MODELS: &[&str] = &[
    "gpt-4o-mini-search-preview",
    "gpt-4o-mini-search-preview-2025-03-11",
    "gpt-4o-search-preview",
    "gpt-4o-search-preview-2025-03-11",
    "gpt-5-search-api",
    "gpt-5-search-api-2025-10-14",
];
const OPENAI_DEEP_RESEARCH_MODELS: &[&str] = &[
    "o3-deep-research",
    "o3-deep-research-2025-06-26",
    "o4-mini-deep-research",
    "o4-mini-deep-research-2025-06-26",
];
const OPENAI_MODERATION_MODELS: &[&str] = &["omni-moderation-2024-09-26", "omni-moderation-latest"];
const OPENAI_TRANSCRIPTION_MODELS: &[&str] = &[
    "gpt-live-transcribe",
    "gpt-realtime-whisper",
    "gpt-transcribe",
    "gpt-4o-mini-transcribe",
    "gpt-4o-mini-transcribe-2025-03-20",
    "gpt-4o-mini-transcribe-2025-12-15",
    "gpt-4o-transcribe",
    "gpt-4o-transcribe-diarize",
    "whisper-1",
];

/// `Provider.capabilities.augment`: the narrow per-provider additions from
/// `providers/*/capabilities.rb`. Providers without an augmenter keep what they have.
pub fn augment_capabilities(
    slug: &str,
    caps: Vec<String>,
    model_id: &str,
    modalities: &Modalities,
) -> Vec<String> {
    let has = |c: &Vec<String>, name: &str| c.iter().any(|x| x == name);
    let tools = ["tool_choice", "parallel_tool_calls"];
    match slug {
        "openai" => {
            let groups: [(&str, Vec<&[&str]>); 6] = [
                (
                    "function_calling",
                    vec![OPENAI_CHAT_MODELS, OPENAI_CODEX_MODELS],
                ),
                (
                    "structured_output",
                    vec![
                        OPENAI_CHAT_MODELS,
                        OPENAI_CODEX_MODELS,
                        OPENAI_SEARCH_MODELS,
                    ],
                ),
                (
                    "vision",
                    vec![
                        OPENAI_CHAT_MODELS,
                        OPENAI_CODEX_MODELS,
                        OPENAI_DEEP_RESEARCH_MODELS,
                        OPENAI_MODERATION_MODELS,
                    ],
                ),
                (
                    "reasoning",
                    vec![OPENAI_CODEX_MODELS, OPENAI_DEEP_RESEARCH_MODELS],
                ),
                ("transcription", vec![OPENAI_TRANSCRIPTION_MODELS]),
                ("citations", vec![OPENAI_SEARCH_MODELS]),
            ];
            let additions: Vec<&str> = groups
                .iter()
                .filter(|(_, lists)| lists.iter().any(|l| l.contains(&model_id)))
                .map(|(c, _)| *c)
                .collect();
            let caps = union(caps, &additions);
            if has(&caps, "function_calling") {
                union(caps, &tools)
            } else {
                caps
            }
        }
        "anthropic" if has(&caps, "function_calling") => union(caps, &tools),
        "deepseek" if has(&caps, "function_calling") => union(caps, &["tool_choice"]),
        "mistral" => {
            let caps = if has(&caps, "function_calling") {
                union(caps, &tools)
            } else {
                caps
            };
            if ["mistral-small-2603", "mistral-small-latest"].contains(&model_id) {
                union(caps, &["structured_output"])
            } else {
                caps
            }
        }
        "gemini" => {
            let mut additions = Vec::new();
            if has(&caps, "function_calling") {
                additions.push("tool_choice");
            }
            let audio_in = modalities.input.iter().any(|m| m == "audio");
            let text_out = modalities.output.iter().any(|m| m == "text");
            if !model_id.contains("embedding") && audio_in && text_out {
                additions.push("transcription");
            }
            union(caps, &additions)
        }
        "xai" if modalities.output.iter().any(|m| m == "text") => {
            let caps = union(caps, &["streaming"]);
            if model_id == "grok-4.3" {
                union(caps, &tools)
            } else {
                caps
            }
        }
        "vertexai" if !model_id.contains("embedding") => {
            let mut additions = Vec::new();
            if model_id == "gemini-2.5-flash" {
                additions.push("tool_choice");
            }
            if modalities.input.iter().any(|m| m == "audio")
                && modalities.output.iter().any(|m| m == "text")
            {
                additions.push("transcription");
            }
            union(caps, &additions)
        }
        "azure" if model_id == "grok-4-1-fast-non-reasoning" => union(caps, &tools),
        "bedrock"
            if ["amazon.nova-2-lite-v1:0", "us.amazon.nova-2-lite-v1:0"].contains(&model_id) =>
        {
            union(caps, &["tool_choice"])
        }
        _ => caps,
    }
}

/// `Models.merge_with_existing`: a provider that answered replaces its own models; the rest keep
/// theirs, and a failed models.dev fetch keeps the models.dev entries already held.
pub fn merge_with_existing(
    existing: &[Model],
    provider_fetch: &ProviderFetch,
    models_dev: &ModelsDevFetch,
) -> Vec<Model> {
    let mut provider_models = provider_fetch.models.clone();
    provider_models.extend(
        existing
            .iter()
            .filter(|m| !provider_fetch.fetched_providers.contains(&m.provider))
            .cloned(),
    );
    let dev = if models_dev.fetched {
        models_dev.models.clone()
    } else {
        models_dev_entries(existing)
    };
    merge_models(&provider_models, &dev)
}

/// `Models#merge_discovered_models`: providers that answered replace their own models, the
/// published catalog replaces what it covers, and everything else survives.
async fn merge_discovered_models(
    config: &Arc<Config>,
    current: &Models,
    published: &[Model],
    remote_only: bool,
) -> Vec<Model> {
    let fetch = fetch_provider_models(config, remote_only).await;
    record_failures(&fetch);
    log_provider_fetch(&fetch);
    let covered: Vec<&str> = fetch
        .fetched_providers
        .iter()
        .map(String::as_str)
        .chain(published.iter().map(|m| m.provider.as_str()))
        .collect();
    let failed: Vec<&str> = fetch.failed.iter().map(|f| f.slug.as_str()).collect();
    let preserved = |provider: &str| failed.contains(&provider) || !covered.contains(&provider);
    let mut provider_models = fetch.models.clone();
    provider_models.extend(
        current
            .all()
            .into_iter()
            .filter(|m| preserved(&m.provider))
            .cloned(),
    );
    merge_models(&provider_models, published)
}

/// `Models#file_store`: the registry file, unless a `model_registry_store` takes precedence.
fn file_store(config: &Config) -> Result<Option<registry::FileStore>> {
    if config.model_registry_store.is_some() {
        return Ok(None);
    }
    config
        .model_registry_file
        .as_ref()
        .map(registry::FileStore::new)
        .transpose()
}

/// `Models#fetch_published_models`: revalidates with the saved ETag only while the catalog behind
/// it is still on disk, and fetches everything again when a 304 leaves nothing to use.
async fn fetch_published_models(
    config: &Config,
    file: Option<&registry::FileStore>,
) -> Result<registry::Published> {
    let cached = file
        .and_then(|f| {
            registry::read(&registry::suffixed(&f.path, ".published.json"))
                .ok()
                .flatten()
        })
        .filter(|m| !m.is_empty());
    let etag = match (&cached, file) {
        (Some(_), Some(f)) => f.etag()?,
        _ => None,
    };
    let mut result = registry::fetch_published(config, etag.as_deref()).await?;
    if result.models.is_none() {
        result.models = cached;
    }
    if result.models.is_some() {
        Ok(result)
    } else {
        registry::fetch_published(config, None).await
    }
}

/// `Models#persist_registry!`: writes the merged registry to the configured store, or else the
/// published snapshot and the merged registry with its ETag to the registry file.
fn persist(
    config: &Config,
    file: Option<&registry::FileStore>,
    models: &[Model],
    published: &registry::Published,
) -> Result<()> {
    let wrap = |destination: String| {
        move |e: Error| match e {
            Error::ModelRegistry(_) => e,
            other => Error::ModelRegistry(format!(
                "Could not save the model registry to {destination}: {other}"
            )),
        }
    };
    if let Some(store) = &config.model_registry_store {
        return store
            .write(&Models::new(models.to_vec()))
            .map_err(wrap(store.description()));
    }
    let file = file.ok_or_else(|| {
        Error::ModelRegistry("No writable model registry store is configured".into())
    })?;
    let write = || -> Result<()> {
        if !published.not_modified {
            let snapshot: Vec<&Model> = published.models.iter().flatten().collect();
            registry::FileStore::new(registry::suffixed(&file.path, ".published.json"))?
                .write(&snapshot, None)?;
        }
        let listed: Vec<&Model> = models.iter().filter(|m| !m.is_unlisted()).collect();
        file.write(&listed, published.etag.as_deref())
    };
    write().map_err(wrap(file.path.display().to_string()))
}

/// `Models#stored_models`: what the store holds after a write. A store keeps entries the merge
/// dropped (unlisted models still referenced), so its answer wins over the merge.
fn stored_models(config: &Config) -> Option<Vec<Model>> {
    match config.model_registry_store.as_ref()?.read() {
        Ok(models) if !models.is_empty() => Some(models),
        Ok(_) => None,
        Err(e) => {
            tracing::debug!("Could not re-read the model registry store: {e}");
            None
        }
    }
}

/// Rebuilds the registry from the bundled copy (`load_from_json` after a test changed it).
pub fn reset() {
    Models::install(bundled_models());
}

// ---- provider listings ---------------------------------------------------------------------

/// `Provider#list_models`: the models `provider` lists at its own catalog endpoint.
pub async fn list_models(provider: Provider, config: Arc<Config>) -> Result<Vec<Model>> {
    provider.ensure_configured(&config)?;
    if provider == Provider::TypeSafe {
        return crate::judge::list_judgment_models(Some(config)).await;
    }
    let conn = Connection::new(provider, config)?;
    let slug = provider.slug();
    match provider {
        Provider::Anthropic => {
            let mut models = Vec::new();
            let mut after: Option<String> = None;
            loop {
                let path = match &after {
                    Some(id) => format!("v1/models?limit=1000&after_id={id}"),
                    None => "v1/models?limit=1000".into(),
                };
                let body = conn.get(&path, &[]).await?.body;
                models.extend(parse_anthropic_models(&body, slug));
                if body.get("has_more").and_then(Value::as_bool) != Some(true) {
                    break;
                }
                after = body
                    .get("last_id")
                    .and_then(Value::as_str)
                    .map(str::to_string);
                if after.is_none() {
                    break;
                }
            }
            Ok(models)
        }
        Provider::Gemini => {
            let mut models = Vec::new();
            let mut token: Option<String> = None;
            loop {
                let path = match &token {
                    Some(t) => format!("models?pageSize=1000&pageToken={t}"),
                    None => "models?pageSize=1000".into(),
                };
                let body = conn.get(&path, &[]).await?.body;
                models.extend(parse_gemini_models(&body, slug));
                token = body
                    .get("nextPageToken")
                    .and_then(Value::as_str)
                    .map(str::to_string);
                if token.is_none() {
                    break;
                }
            }
            Ok(models)
        }
        Provider::Mistral => Ok(parse_mistral_models(
            &conn.get("models", &[]).await?.body,
            slug,
        )),
        Provider::XAI => Ok(parse_xai_models(&conn.get("models", &[]).await?.body, slug)),
        Provider::OpenRouter => {
            let mut models: Vec<Model> = Vec::new();
            for url in OPENROUTER_CATALOGS {
                for m in parse_openrouter_models(&conn.get(url, &[]).await?.body, slug) {
                    if !models.iter().any(|x| x.id == m.id) {
                        models.push(m);
                    }
                }
            }
            Ok(models)
        }
        Provider::Perplexity => match conn.get("v1/models", &[]).await {
            Ok(raw) => Ok(parse_perplexity_models(&raw.body, slug)),
            Err(e) if !matches!(e, Error::Timeout(_) | Error::ConnectionFailed(_)) => {
                tracing::warn!(
                    "Perplexity models endpoint failed ({e}). Using the static model list."
                );
                Ok(perplexity_static_models(
                    slug,
                    PERPLEXITY_STATIC_IDS.iter().copied(),
                ))
            }
            Err(e) => Err(e),
        },
        Provider::Ollama | Provider::OllamaCloud => {
            let body = conn.get("models", &[]).await?.body;
            let base = provider.api_base(conn.config())?;
            let show = format!(
                "{}/api/show",
                base.trim_end_matches('/')
                    .rsplit_once('/')
                    .map(|(p, _)| p)
                    .unwrap_or(&base)
            );
            let mut details = std::collections::HashMap::new();
            for id in data(&body)
                .iter()
                .filter_map(|m| m.get("id").and_then(Value::as_str))
            {
                let reported = match conn
                    .post(&show, &json!({ "model": id }), &[], &mut |_| {})
                    .await
                {
                    Ok(raw) => strings(raw.body.get("capabilities")),
                    Err(e) => {
                        tracing::debug!("Ollama did not report capabilities for {id} ({e}).");
                        Vec::new()
                    }
                };
                details.insert(id.to_string(), reported);
            }
            Ok(parse_ollama_models(
                &body,
                slug,
                &details,
                provider == Provider::OllamaCloud,
            ))
        }
        Provider::GPUStack => {
            let mut entries: Vec<(String, Value, Vec<String>)> = Vec::new();
            for category in GPUSTACK_CATEGORIES {
                let body = conn
                    .get(&format!("models?with_meta=true&categories={category}"), &[])
                    .await?
                    .body;
                for m in data(&body) {
                    let Some(id) = m.get("id").and_then(Value::as_str) else {
                        continue;
                    };
                    match entries.iter_mut().find(|(i, _, _)| i == id) {
                        Some(entry) => entry.2.push(category.to_string()),
                        None => {
                            entries.push((id.to_string(), m.clone(), vec![category.to_string()]))
                        }
                    }
                }
            }
            Ok(entries
                .into_iter()
                .map(|(_, m, cats)| gpustack_model(&m, &cats, slug))
                .collect())
        }
        _ => Ok(parse_openai_models(
            &conn.get("models", &[]).await?.body,
            slug,
            false,
        )),
    }
}

fn data(body: &Value) -> Vec<Value> {
    body.get("data")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}

fn strings(v: Option<&Value>) -> Vec<String> {
    v.and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

/// A bare model the way `Model.new` builds one from the few fields a listing gives.
fn bare(id: &str, name: &str, slug: &str) -> Model {
    Model {
        id: id.to_string(),
        name: name.to_string(),
        provider: slug.to_string(),
        family: None,
        created_at: None,
        context_window: None,
        max_output_tokens: None,
        knowledge_cutoff: None,
        modalities: Modalities::default(),
        capabilities: Vec::new(),
        pricing: Default::default(),
        metadata: Map::new(),
        unlisted_at: None,
    }
}

/// `Time.at(seconds)` as RubyLLM's registry writes it.
fn unix_time(v: Option<&Value>) -> Option<String> {
    let secs = v.and_then(Value::as_i64)?;
    chrono::DateTime::from_timestamp(secs, 0).map(|t| t.format("%Y-%m-%d %H:%M:%S UTC").to_string())
}

fn parse_time(v: Option<&Value>) -> Option<String> {
    let t = chrono::DateTime::parse_from_rfc3339(v?.as_str()?).ok()?;
    Some(
        t.with_timezone(&chrono::Utc)
            .format("%Y-%m-%d %H:%M:%S UTC")
            .to_string(),
    )
}

/// `ChatCompletions::Models.parse_list_models_response` (and xAI's, which compacts its metadata).
pub fn parse_openai_models(body: &Value, slug: &str, compact: bool) -> Vec<Model> {
    data(body)
        .iter()
        .filter_map(|m| {
            let id = m.get("id").and_then(Value::as_str)?;
            let mut model = bare(id, id, slug);
            model.created_at = unix_time(m.get("created"));
            for key in ["object", "owned_by"] {
                let v = m.get(key).cloned().unwrap_or(Value::Null);
                if !(compact && v.is_null()) {
                    model.metadata.insert(key.into(), v);
                }
            }
            if let Some(date) = m.get("shutdown_date").filter(|v| !v.is_null() && !compact) {
                model.metadata.insert("shutdown_date".into(), date.clone());
            }
            Some(model)
        })
        .collect()
}

/// `XAI::Models.parse_list_models_response`: the listing, plus the model-less TTS and STT services
/// it omits.
pub fn parse_xai_models(body: &Value, slug: &str) -> Vec<Model> {
    let mut models = parse_openai_models(body, slug, true);
    models.extend(xai_audio_models(slug));
    models
}

fn xai_audio_models(slug: &str) -> Vec<Model> {
    [
        ("grok-tts", "Grok TTS", ["text"], ["audio"], None),
        (
            "grok-stt",
            "Grok STT",
            ["audio"],
            ["text"],
            Some("transcription"),
        ),
    ]
    .into_iter()
    .map(|(id, name, input, output, cap)| {
        let mut m = bare(id, name, slug);
        m.family = Some("grok".into());
        m.modalities = Modalities {
            input: input.map(String::from).to_vec(),
            output: output.map(String::from).to_vec(),
        };
        m.capabilities = cap.map(String::from).into_iter().collect();
        m
    })
    .collect()
}

/// `Anthropic::Models.parse_list_models_response`.
pub fn parse_anthropic_models(body: &Value, slug: &str) -> Vec<Model> {
    const REPORTED: &[(&str, &str)] = &[
        ("batch", "batch"),
        ("citations", "citations"),
        ("image_input", "vision"),
        ("structured_outputs", "structured_output"),
        ("thinking", "reasoning"),
    ];
    data(body)
        .iter()
        .filter_map(|m| {
            let id = m.get("id").and_then(Value::as_str)?;
            let mut model = bare(
                id,
                m.get("display_name").and_then(Value::as_str).unwrap_or(id),
                slug,
            );
            model.created_at = parse_time(m.get("created_at"));
            model.context_window = m.get("max_input_tokens").and_then(Value::as_i64);
            model.max_output_tokens = m.get("max_tokens").and_then(Value::as_i64);
            if let Some(reported) = m.get("capabilities").and_then(Value::as_object) {
                model.capabilities = reported
                    .iter()
                    .filter(|(_, d)| d.get("supported").and_then(Value::as_bool) == Some(true))
                    .filter_map(|(name, _)| {
                        REPORTED
                            .iter()
                            .find(|(k, _)| k == name)
                            .map(|(_, c)| c.to_string())
                    })
                    .collect();
            }
            Some(model)
        })
        .collect()
}

/// `Gemini::Models.parse_list_models_response`.
pub fn parse_gemini_models(body: &Value, slug: &str) -> Vec<Model> {
    body.get("models")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|m| {
            let id = m
                .get("name")
                .and_then(Value::as_str)?
                .replace("models/", "");
            let methods = strings(m.get("supportedGenerationMethods"));
            let has = |x: &str| methods.iter().any(|mm| mm == x);
            let mut model = bare(
                &id,
                m.get("displayName").and_then(Value::as_str).unwrap_or(&id),
                slug,
            );
            model.context_window = m.get("inputTokenLimit").and_then(Value::as_i64);
            model.max_output_tokens = m.get("outputTokenLimit").and_then(Value::as_i64);
            if has("embedContent") {
                model.modalities = Modalities {
                    input: vec!["text".into()],
                    output: vec!["embeddings".into()],
                };
            }
            if has("batchGenerateContent") || has("asyncBatchEmbedContent") {
                model.capabilities.push("batch".into());
            }
            if has("createCachedContent") {
                model.capabilities.push("caching".into());
            }
            if has("bidiGenerateContent") {
                model
                    .capabilities
                    .extend(["streaming".into(), "realtime".into()]);
            }
            model.metadata.insert(
                "version".into(),
                m.get("version").cloned().unwrap_or(Value::Null),
            );
            model.metadata.insert(
                "description".into(),
                m.get("description").cloned().unwrap_or(Value::Null),
            );
            model
                .metadata
                .insert("supported_generation_methods".into(), json!(methods));
            Some(model)
        })
        .collect()
}

/// `Mistral::Models.parse_list_models_response`.
pub fn parse_mistral_models(body: &Value, slug: &str) -> Vec<Model> {
    const FLAGS: &[(&str, &str)] = &[
        ("function_calling", "function_calling"),
        ("reasoning", "reasoning"),
        ("fine_tuning", "fine_tuning"),
        ("vision", "vision"),
        ("ocr", "vision"),
        ("moderation", "moderation"),
        ("audio_transcription", "transcription"),
        ("audio_transcription_realtime", "realtime"),
        ("audio_speech", "speech_generation"),
    ];
    let embed =
        regex::Regex::new(r"(?i)(?:\A|[-_ ])embed(?:ding)?(?:\z|[-_ ])").expect("valid regex");
    data(body)
        .iter()
        .filter_map(|m| {
            let id = m.get("id").and_then(Value::as_str)?;
            let flags = m.get("capabilities").cloned().unwrap_or(json!({}));
            let on = |f: &str| truthy(flags.get(f));
            let mut model = bare(id, id, slug);
            model.context_window = m.get("max_context_length").and_then(Value::as_i64);
            model.capabilities = FLAGS
                .iter()
                .filter(|(f, _)| on(f))
                .map(|(_, c)| c.to_string())
                .collect();
            let described: Vec<String> = [m.get("id"), m.get("description")]
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .chain(strings(m.get("aliases")))
                .collect();
            model.modalities = if described.iter().any(|v| embed.is_match(v)) {
                Modalities {
                    input: vec!["text".into()],
                    output: vec!["embeddings".into()],
                }
            } else if on("audio_transcription") {
                Modalities {
                    input: vec!["audio".into()],
                    output: vec!["text".into()],
                }
            } else {
                let mut input = vec!["text".to_string()];
                if on("vision") || on("ocr") {
                    input.push("image".into());
                }
                let mut output = vec!["text".to_string()];
                if on("audio_speech") {
                    output.push("audio".into());
                }
                Modalities { input, output }
            };
            for key in [
                "object",
                "owned_by",
                "description",
                "aliases",
                "deprecation",
                "deprecation_replacement_model",
            ] {
                if let Some(v) = m
                    .get(key)
                    .filter(|v| !is_blank(v) || matches!(v, Value::Bool(_) | Value::Number(_)))
                {
                    model.metadata.insert(key.into(), v.clone());
                }
            }
            Some(model)
        })
        .collect()
}

const OPENROUTER_CATALOGS: &[&str] = &[
    "models",
    "embeddings/models",
    "models?output_modalities=speech",
    "models?output_modalities=transcription",
    "models?output_modalities=rerank",
    "images/models",
];

/// `OpenRouter::Models.parse_list_models_response`.
pub fn parse_openrouter_models(body: &Value, slug: &str) -> Vec<Model> {
    data(body)
        .iter()
        .filter_map(|m| {
            let id = m.get("id").and_then(Value::as_str)?;
            let outputs = strings(m.pointer("/architecture/output_modalities"));
            let mut mapped: Vec<String> = Vec::new();
            for o in &outputs {
                let v = match o.as_str() {
                    "speech" => "audio",
                    "transcription" => "text",
                    other => other,
                };
                if !mapped.iter().any(|x| x == v) {
                    mapped.push(v.to_string());
                }
            }
            let mut model = bare(
                id,
                m.get("name").and_then(Value::as_str).unwrap_or(id),
                slug,
            );
            model.family = id.split('/').next().map(str::to_string);
            model.created_at = unix_time(m.get("created"));
            model.context_window = m.get("context_length").and_then(Value::as_i64);
            model.max_output_tokens = m
                .pointer("/top_provider/max_completion_tokens")
                .and_then(Value::as_i64);
            model.knowledge_cutoff = m
                .get("knowledge_cutoff")
                .and_then(Value::as_str)
                .map(str::to_string);
            model.modalities = Modalities {
                input: strings(m.pointer("/architecture/input_modalities")),
                output: mapped,
            };
            let mut standard = Map::new();
            for (source, target) in [
                ("prompt", "input_per_million"),
                ("completion", "output_per_million"),
                ("input_cache_read", "cache_read_input_per_million"),
                ("input_cache_write", "cache_write_input_per_million"),
                ("internal_reasoning", "reasoning_output_per_million"),
            ] {
                let v = m
                    .pointer(&format!("/pricing/{source}"))
                    .map(to_f)
                    .unwrap_or(0.0);
                if v > 0.0 {
                    standard.insert(target.into(), (v * 1_000_000.0).into());
                }
            }
            if !standard.is_empty() {
                model.pricing =
                    serde_json::from_value(json!({ "text_tokens": { "standard": standard } }))
                        .unwrap_or_default();
            }
            let caps = supported_parameters_to_capabilities(m.get("supported_parameters"));
            let by_output: Vec<&str> = outputs
                .iter()
                .filter_map(|o| match o.as_str() {
                    "speech" => Some("speech_generation"),
                    "transcription" => Some("transcription"),
                    "image" => Some("image_generation"),
                    _ => None,
                })
                .collect();
            model.capabilities = union(caps, &by_output);
            for (key, pointer) in [
                ("description", "/description"),
                ("architecture", "/architecture"),
                ("top_provider", "/top_provider"),
                ("per_request_limits", "/per_request_limits"),
                ("supported_parameters", "/supported_parameters"),
                ("expiration_date", "/expiration_date"),
            ] {
                model.metadata.insert(
                    key.into(),
                    m.pointer(pointer).cloned().unwrap_or(Value::Null),
                );
            }
            Some(model)
        })
        .collect()
}

/// `OpenRouter::Models#supported_parameters_to_capabilities`. `params` is the listing's array of
/// names, or the image catalog's hash of parameter definitions, whose keys Ruby's `include?` checks.
pub fn supported_parameters_to_capabilities(params: Option<&Value>) -> Vec<String> {
    const PARAMS: &[(&str, &[&str])] = &[
        ("function_calling", &["tools", "tool_choice"]),
        ("tool_choice", &["tool_choice"]),
        ("parallel_tool_calls", &["parallel_tool_calls"]),
        (
            "structured_output",
            &["response_format", "structured_outputs"],
        ),
        ("batch", &["batch"]),
    ];
    let p: Vec<&str> = match params {
        Some(Value::Array(a)) => a.iter().filter_map(Value::as_str).collect(),
        Some(Value::Object(o)) => o.keys().map(String::as_str).collect(),
        _ => return Vec::new(),
    };
    let mut caps = vec!["streaming".to_string()];
    caps.extend(
        PARAMS
            .iter()
            .filter(|(_, ps)| ps.iter().any(|x| p.contains(x)))
            .map(|(c, _)| c.to_string()),
    );
    if p.contains(&"logit_bias") && p.contains(&"top_k") {
        caps.push("predicted_outputs".into());
    }
    caps
}

/// Ruby's `to_f` on a JSON price, which OpenRouter sends as a string.
fn to_f(v: &Value) -> f64 {
    v.as_f64()
        .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
        .unwrap_or(0.0)
}

const PERPLEXITY_SEARCH_IDS: &[&str] = &[
    "sonar",
    "sonar-pro",
    "sonar-reasoning-pro",
    "sonar-deep-research",
];
const PERPLEXITY_PRESET_IDS: &[&str] = &["fast", "low", "medium", "high", "xhigh", "wide-research"];
const PERPLEXITY_EMBEDDING_IDS: &[&str] = &["pplx-embed-v1-0.6b", "pplx-embed-v1-4b"];
static PERPLEXITY_STATIC_IDS: LazyLock<Vec<&'static str>> = LazyLock::new(|| {
    PERPLEXITY_SEARCH_IDS
        .iter()
        .chain(PERPLEXITY_PRESET_IDS)
        .chain(PERPLEXITY_EMBEDDING_IDS)
        .copied()
        .collect()
});

/// One `Perplexity::Models::STATIC_MODEL_DATA` entry: (context window, input, output, reasoning
/// prices, capabilities).
type StaticModel = (
    Option<i64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    &'static [&'static str],
);

fn perplexity_static(id: &str) -> StaticModel {
    const SEARCH: &[&str] = &["streaming", "structured_output", "citations"];
    const REASONING: &[&str] = &["streaming", "structured_output", "citations", "reasoning"];
    match id {
        "sonar" => (Some(128_000), Some(1.0), Some(1.0), None, SEARCH),
        "sonar-pro" => (Some(200_000), Some(3.0), Some(15.0), None, SEARCH),
        "sonar-reasoning-pro" => (Some(128_000), Some(2.0), Some(8.0), None, REASONING),
        "sonar-deep-research" => (Some(128_000), Some(2.0), Some(8.0), Some(3.0), REASONING),
        "pplx-embed-v1-0.6b" => (Some(32_768), Some(0.004), None, None, &[]),
        "pplx-embed-v1-4b" => (Some(32_768), Some(0.03), None, None, &[]),
        p if PERPLEXITY_PRESET_IDS.contains(&p) => (
            None,
            None,
            None,
            None,
            &[
                "streaming",
                "structured_output",
                "citations",
                "function_calling",
            ],
        ),
        _ => (None, None, None, None, &[]),
    }
}

fn perplexity_model(id: &str, slug: &str, pricing: Option<Value>) -> Model {
    let (context, input, output, reasoning, caps) = perplexity_static(id);
    let mut model = bare(id, id, slug);
    model.context_window = context;
    model.capabilities = caps.iter().map(|c| c.to_string()).collect();
    let pricing = pricing.or_else(|| {
        let input = input?;
        let mut standard = json!({ "input_per_million": input });
        if let Some(o) = output {
            standard["output_per_million"] = o.into();
        }
        if let Some(r) = reasoning {
            standard["reasoning_output_per_million"] = r.into();
        }
        Some(json!({ "text_tokens": { "standard": standard } }))
    });
    model.pricing = pricing
        .and_then(|p| serde_json::from_value(p).ok())
        .unwrap_or_default();
    if PERPLEXITY_EMBEDDING_IDS.contains(&id) {
        model.modalities = Modalities {
            input: vec!["text".into()],
            output: vec!["embeddings".into()],
        };
    }
    model
}

fn perplexity_static_models<'a>(slug: &str, ids: impl Iterator<Item = &'a str>) -> Vec<Model> {
    ids.map(|id| perplexity_model(id, slug, None)).collect()
}

/// `Perplexity::Models.parse_list_models_response`: the endpoint's catalog, with the search models,
/// presets, and embedding models it leaves out added from the static list.
pub fn parse_perplexity_models(body: &Value, slug: &str) -> Vec<Model> {
    let listed: Vec<Model> = data(body)
        .iter()
        .filter_map(|m| {
            let id = m.get("id").and_then(Value::as_str)?;
            let standard: Map<String, Value> = [
                ("input", "input_per_million"),
                ("output", "output_per_million"),
                ("cache_read", "cache_read_input_per_million"),
                ("cache_write", "cache_write_input_per_million"),
            ]
            .iter()
            .filter_map(|(k, t)| {
                m.pointer(&format!("/pricing/{k}"))
                    .filter(|v| !v.is_null())
                    .map(|v| (t.to_string(), v.clone()))
            })
            .collect();
            let pricing = (m.get("pricing").is_some_and(Value::is_object) && !standard.is_empty())
                .then(|| json!({ "text_tokens": { "standard": standard } }));
            Some(perplexity_model(id, slug, pricing))
        })
        .collect();
    let missing = PERPLEXITY_STATIC_IDS
        .iter()
        .copied()
        .filter(|id| !listed.iter().any(|m| m.id == *id));
    let mut models = perplexity_static_models(slug, missing);
    models.extend(listed);
    models
}

/// `Ollama::Models.parse_list_models_response`; Ollama Cloud drops `structured_output`. `details`
/// holds what `/api/show` reported for each id.
pub fn parse_ollama_models(
    body: &Value,
    slug: &str,
    details: &std::collections::HashMap<String, Vec<String>>,
    cloud: bool,
) -> Vec<Model> {
    data(body)
        .iter()
        .filter_map(|m| {
            let id = m.get("id").and_then(Value::as_str)?;
            let reported = details.get(id).cloned().unwrap_or_default();
            let has = |x: &str| reported.iter().any(|r| r == x);
            let mut model = bare(id, id, slug);
            model.family = Some("ollama".into());
            model.created_at = unix_time(m.get("created"));
            if has("embedding") {
                model.modalities = Modalities {
                    input: vec!["text".into()],
                    output: vec!["embeddings".into()],
                };
            } else {
                let mut input = vec!["text".to_string()];
                if has("vision") {
                    input.push("image".into());
                }
                model.modalities = Modalities {
                    input,
                    output: vec!["text".into()],
                };
                let base: &[&str] = if cloud {
                    &["streaming"]
                } else {
                    &["streaming", "structured_output"]
                };
                model.capabilities = base.iter().map(|c| c.to_string()).collect();
                for (native, cap) in [
                    ("tools", "function_calling"),
                    ("vision", "vision"),
                    ("thinking", "reasoning"),
                ] {
                    if has(native) {
                        model.capabilities.push(cap.into());
                    }
                }
            }
            model.metadata.insert(
                "owned_by".into(),
                m.get("owned_by").cloned().unwrap_or(Value::Null),
            );
            Some(model)
        })
        .collect()
}

const GPUSTACK_CATEGORIES: &[&str] = &[
    "llm",
    "embedding",
    "image",
    "reranker",
    "speech_to_text",
    "text_to_speech",
    "unknown",
];

/// `GPUStack::Models#parse_list_models_response`: a plain OpenAI list, without categories.
pub fn parse_gpustack_models(body: &Value, slug: &str) -> Vec<Model> {
    data(body)
        .iter()
        .map(|m| gpustack_model(m, &[], slug))
        .collect()
}

/// `GPUStack::Models#build_model`.
pub fn gpustack_model(m: &Value, categories: &[String], slug: &str) -> Model {
    let id = m.get("id").and_then(Value::as_str).unwrap_or_default();
    let meta = m.get("meta").cloned().unwrap_or(json!({}));
    let cat = |c: &str| categories.iter().any(|x| x == c);
    let any = |cs: &[&str]| cs.iter().any(|c| cat(c));
    let flag = |k: &str| truthy(meta.get(k));
    let mut model = bare(id, id, slug);
    model.family = Some("gpustack".into());
    model.created_at = unix_time(m.get("created"));
    let context = meta
        .get("n_ctx")
        .filter(|v| !v.is_null())
        .or_else(|| meta.get("max_model_len"))
        .and_then(Value::as_i64);
    model.context_window = context;
    model.max_output_tokens = context;
    if cat("llm") {
        model.capabilities = ["streaming", "structured_output", "json_mode"]
            .map(String::from)
            .to_vec();
        for (k, c) in [
            ("support_tool_calls", "function_calling"),
            ("support_vision", "vision"),
            ("support_reasoning", "reasoning"),
        ] {
            if flag(k) {
                model.capabilities.push(c.into());
            }
        }
    }
    let mut input = Vec::new();
    if any(&["llm", "embedding", "image", "reranker", "text_to_speech"]) {
        input.push("text".to_string());
    }
    if cat("llm") && flag("support_vision") {
        input.push("image".into());
    }
    if cat("speech_to_text") || (cat("llm") && flag("support_audio")) {
        input.push("audio".into());
    }
    let mut output = Vec::new();
    if any(&["llm", "speech_to_text", "reranker"]) {
        output.push("text".to_string());
    }
    for (c, o) in [
        ("embedding", "embeddings"),
        ("image", "image"),
        ("text_to_speech", "audio"),
    ] {
        if cat(c) {
            output.push(o.into());
        }
    }
    model.modalities = Modalities { input, output };
    model.metadata.insert(
        "owned_by".into(),
        m.get("owned_by").cloned().unwrap_or(Value::Null),
    );
    model
        .metadata
        .insert("categories".into(), json!(categories));
    model
        .metadata
        .insert("meta".into(), m.get("meta").cloned().unwrap_or(Value::Null));
    model
}
