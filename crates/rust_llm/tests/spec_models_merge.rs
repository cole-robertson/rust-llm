//! RubyLLM 2.0's `spec/ruby_llm/models_merge_spec.rb`, plus the `.models_dev_model_attributes`
//! rows of `models_spec.rb`, `models_refresh_spec.rb`'s `.models_dev_model_id`, and
//! `support/utils_spec.rb`'s ISO date prefix helpers.
//!
//! Ruby stubs `Provider.configured_providers` with fake provider classes; here the providers are
//! real ones (Mistral, Anthropic, Ollama) pointed at wiremock servers. `RubyLLM.logger` is a
//! `tracing` collector. Tests that change the process-wide registry or configuration hold
//! `GLOBAL` and restore both when done.

use std::sync::{Arc, Mutex};

use rust_llm::model::{Model, ModelType};
use rust_llm::models::refresh::{
    ModelsDevFetch, ProviderFetch, add_provider_metadata, fetch_merged_models,
    fetch_models_dev_models, fetch_provider_models, find_models_dev_model, is_blank,
    iso_date_prefix_to_utc_midnight_string, log_models_dev_fetch, log_provider_fetch, merge_models,
    merge_with_existing, models_dev_model_attributes, models_dev_model_id, models_dev_pricing,
    normalize_models_dev_knowledge, normalize_models_dev_modalities, parse_iso_date_prefix,
    read_existing_models, refresh_from_providers, refresh_with_config,
};
use rust_llm::models::registry::ModelRegistryStore;
use rust_llm::models::{Models, models_from_store};
use rust_llm::{Chat, Config, Error};
use serde_json::{Value, json};
use wiremock::matchers::{header_exists, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

static GLOBAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// Puts back the global registry and configuration a test changed.
struct Restore(Arc<Config>);

impl Restore {
    fn new() -> Restore {
        Restore(rust_llm::config())
    }
}

impl Drop for Restore {
    fn drop(&mut self) {
        let saved = (*self.0).clone();
        rust_llm::configure(|c| *c = saved);
        rust_llm::models::refresh::reset();
    }
}

/// `model(id:, provider:, **attributes)`. Ruby leaves `name` nil unless given; the port's
/// `name` is a `String`, so nil is the empty string.
fn model(id: &str, provider: &str, extra: Value) -> Model {
    let mut data = json!({ "id": id, "name": "", "provider": provider });
    for (k, v) in extra.as_object().cloned().unwrap_or_default() {
        data[k] = v;
    }
    serde_json::from_value(data).unwrap()
}

fn sorted<S: ToString>(v: impl IntoIterator<Item = S>) -> Vec<String> {
    let mut v: Vec<String> = v.into_iter().map(|s| s.to_string()).collect();
    v.sort();
    v
}

fn ids(models: &[Model]) -> Vec<&str> {
    models.iter().map(|m| m.id.as_str()).collect()
}

// ---- RubyLLM.logger ----------------------------------------------------------------------------

/// Collects `tracing` INFO and WARN events on this thread as `(level, message)`.
struct Logger(Arc<Mutex<Vec<(tracing::Level, String)>>>);

impl tracing::Subscriber for Logger {
    // Tests run in parallel: a callsite first hit with no collector set is cached as "never",
    // so ask on every event instead of caching the interest.
    fn register_callsite(
        &self,
        _: &'static tracing::Metadata<'static>,
    ) -> tracing::subscriber::Interest {
        tracing::subscriber::Interest::sometimes()
    }
    fn max_level_hint(&self) -> Option<tracing::level_filters::LevelFilter> {
        Some(tracing::level_filters::LevelFilter::TRACE)
    }
    fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        struct Message<'a>(&'a mut String);
        impl tracing::field::Visit for Message<'_> {
            fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                if field.name() == "message" {
                    self.0.push_str(&format!("{value:?}"));
                }
            }
        }
        let level = *event.metadata().level();
        if level == tracing::Level::WARN || level == tracing::Level::INFO {
            let mut text = String::new();
            event.record(&mut Message(&mut text));
            self.0.lock().unwrap().push((level, text));
        }
    }
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}

/// A logger for the rest of the current scope; `.lines(level)` reads what it received.
struct Logged {
    lines: Arc<Mutex<Vec<(tracing::Level, String)>>>,
    _guard: tracing::subscriber::DefaultGuard,
}

impl Logged {
    fn start() -> Logged {
        let lines = Arc::new(Mutex::new(Vec::new()));
        let guard =
            tracing::dispatcher::set_default(&tracing::Dispatch::new(Logger(lines.clone())));
        Logged {
            lines,
            _guard: guard,
        }
    }

    fn lines(&self, level: tracing::Level) -> Vec<String> {
        self.lines
            .lock()
            .unwrap()
            .iter()
            .filter(|(l, _)| *l == level)
            .map(|(_, m)| m.clone())
            .collect()
    }
}

// ---- fake providers ----------------------------------------------------------------------------

/// A Mistral endpoint listing `models` (as `{ id, max_context_length }` entries). Mistral stands in
/// for the spec's `fake_provider(slug: 'acme', display_name: 'Acme')`.
async fn mistral(models: Value) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "data": models })))
        .mount(&server)
        .await;
    server
}

fn with_mistral(config: &mut Config, server: &MockServer) {
    config.set("mistral_api_key", "test");
    config.set("mistral_api_base", format!("{}/v1", server.uri()));
}

async fn models_dev(body: Value) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api.json"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&server)
        .await;
    server
}

fn with_models_dev(config: &mut Config, server: &MockServer) {
    config.set("models_dev_url", format!("{}/api.json", server.uri()));
}

/// A configuration with no provider keys (the environment's are not read) and no registry file.
fn bare_config() -> Config {
    let mut config = Config::default();
    config.model_registry_file = None;
    config.max_retries = 0;
    config
}

// ---- .fetch_provider_models --------------------------------------------------------------------

// spec: models_merge_spec.rb:38 .fetch_provider_models collects models from every configured provider
#[tokio::test]
async fn fetch_provider_models_collects_models_from_every_configured_provider() {
    let server = mistral(json!([{ "id": "acme-1" }])).await;
    let mut config = bare_config();
    with_mistral(&mut config, &server);

    let result = fetch_provider_models(&Arc::new(config), true).await;

    assert_eq!(ids(&result.models), ["acme-1"]);
    assert_eq!(result.fetched_providers, ["mistral"]);
    assert_eq!(result.configured_names, ["Mistral"]);
    assert!(result.failed.is_empty());
    assert!(result.empty.is_empty());
}

// spec: models_merge_spec.rb:53 .fetch_provider_models treats a provider that lists nothing as one that did not answer
#[tokio::test]
async fn fetch_provider_models_treats_an_empty_listing_as_no_answer() {
    let server = mistral(json!([])).await;
    let mut config = bare_config();
    with_mistral(&mut config, &server);

    let result = fetch_provider_models(&Arc::new(config), true).await;

    assert!(result.fetched_providers.is_empty());
    assert!(result.failed.is_empty());
    assert_eq!(
        result.empty,
        [("Mistral".to_string(), "mistral".to_string())]
    );
}

// spec: models_merge_spec.rb:64 .fetch_provider_models records a provider that fails instead of aborting the refresh
#[tokio::test]
async fn fetch_provider_models_records_a_failing_provider() {
    let working = mistral(json!([{ "id": "other-1" }])).await;
    let failing = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({ "error": "boom" })))
        .mount(&failing)
        .await;
    let mut config = bare_config();
    with_mistral(&mut config, &working);
    config.set("ollama_api_base", format!("{}/v1", failing.uri()));

    // remote_only: false, so the local Ollama is asked too.
    let result = fetch_provider_models(&Arc::new(config), false).await;

    assert_eq!(ids(&result.models), ["other-1"]);
    assert_eq!(result.fetched_providers, ["mistral"]);
    let failure = &result.failed[0];
    assert_eq!(
        (failure.name.as_str(), failure.slug.as_str()),
        ("Ollama", "ollama")
    );
    // Ruby keeps the exception object; the port keeps its message.
    assert!(failure.error.contains("boom"), "{}", failure.error);
}

// ---- .log_provider_fetch / .log_models_dev_fetch -----------------------------------------------

// spec: models_merge_spec.rb:93 .log_provider_fetch warns once per failed provider
#[test]
fn log_provider_fetch_warns_once_per_failed_provider() {
    let logged = Logged::start();
    log_provider_fetch(&ProviderFetch {
        configured_names: vec!["Acme".into()],
        failed: vec![rust_llm::models::ProviderFailure {
            name: "Acme".into(),
            slug: "acme".into(),
            error: "boom".into(),
        }],
        ..Default::default()
    });

    assert_eq!(
        logged.lines(tracing::Level::INFO),
        ["Fetching models from providers: Acme"]
    );
    // Ruby prints `(ArgumentError: boom)`; a Rust error has no class name, only its message.
    assert_eq!(
        logged.lines(tracing::Level::WARN),
        ["Failed to fetch Acme models (boom). Keeping existing."]
    );
}

// spec: models_merge_spec.rb:106 .log_provider_fetch warns about a provider that listed nothing
#[test]
fn log_provider_fetch_warns_about_a_provider_that_listed_nothing() {
    let logged = Logged::start();
    log_provider_fetch(&ProviderFetch {
        configured_names: vec!["Acme".into()],
        empty: vec![("Acme".into(), "acme".into())],
        ..Default::default()
    });

    assert_eq!(
        logged.lines(tracing::Level::WARN),
        ["Acme listed no models. Keeping existing."]
    );
}

// spec: models_merge_spec.rb:119 .log_models_dev_fetch stays quiet on a successful fetch
#[test]
fn log_models_dev_fetch_stays_quiet_on_success() {
    let logged = Logged::start();
    log_models_dev_fetch(&ModelsDevFetch {
        models: vec![],
        fetched: true,
    });
    assert!(logged.lines(tracing::Level::WARN).is_empty());
}

// spec: models_merge_spec.rb:127 .log_models_dev_fetch warns when falling back to cached data
#[test]
fn log_models_dev_fetch_warns_when_falling_back() {
    let logged = Logged::start();
    log_models_dev_fetch(&ModelsDevFetch {
        models: vec![],
        fetched: false,
    });
    assert_eq!(
        logged.lines(tracing::Level::WARN),
        ["Using cached models.dev data due to fetch failure."]
    );
}

// ---- .fetch_models_dev_models ------------------------------------------------------------------

// spec: models_merge_spec.rb:164 .fetch_models_dev_models maps the models.dev catalog onto Model instances
#[tokio::test]
async fn fetch_models_dev_models_maps_the_catalog_onto_models() {
    let server = models_dev(json!({
        "openai": { "models": { "gpt-test": {
            "id": "gpt-test", "name": "GPT Test", "tool_call": true,
            "modalities": { "input": ["text", "video", "unknown"], "output": ["text", "bogus"] },
            "cost": { "input": 1.0, "output": 2.0, "input_audio": 3.0, "output_audio": 4.0 }
        } } },
        "google-vertex": { "models": { "claude-haiku": { "id": "claude-haiku-4-5@20251001", "name": "Claude Haiku" } } },
        "perplexity-agent": { "models": { "third-party": { "id": "anthropic/claude-test", "name": "Claude Test" } } },
        "unknownprovider": { "models": { "x": { "id": "x" } } }
    }))
    .await;
    let mut config = bare_config();
    with_models_dev(&mut config, &server);

    let result = fetch_models_dev_models(&config, &[]).await;

    assert!(result.fetched);
    let mut pairs: Vec<(&str, &str)> = result
        .models
        .iter()
        .map(|m| (m.provider.as_str(), m.id.as_str()))
        .collect();
    pairs.sort();
    assert_eq!(
        pairs,
        [
            ("openai", "gpt-test"),
            ("perplexity", "anthropic/claude-test"),
            ("vertexai", "claude-haiku-4-5")
        ]
    );
    let openai = result
        .models
        .iter()
        .find(|m| m.provider == "openai")
        .unwrap();
    assert_eq!(openai.modalities.input, ["text", "video"]);
    assert_eq!(openai.modalities.output, ["text"]);
    assert_eq!(
        sorted(&openai.capabilities),
        sorted([
            "function_calling",
            "tool_choice",
            "parallel_tool_calls",
            "vision",
            "video"
        ])
    );
    assert_eq!(
        serde_json::to_value(&openai.pricing).unwrap(),
        json!({
            "text_tokens": { "standard": { "input_per_million": 1.0, "output_per_million": 2.0 } },
            "audio_tokens": { "standard": { "input_per_million": 3.0, "output_per_million": 4.0 } }
        })
    );
}

// spec: models_merge_spec.rb:187 .fetch_models_dev_models keeps the models.dev entries it already had when the answer carries no models
#[tokio::test]
async fn fetch_models_dev_models_keeps_cached_entries_when_the_answer_has_no_models() {
    let logged = Logged::start();
    let cached = model(
        "cached",
        "openai",
        json!({ "metadata": { "source": "models.dev" } }),
    );
    for body in [
        json!({}),
        Value::Null,
        json!({ "openai-inc": { "models": { "a": { "id": "a" } } } }),
    ] {
        let server = models_dev(body).await;
        let mut config = bare_config();
        with_models_dev(&mut config, &server);

        let result = fetch_models_dev_models(&config, std::slice::from_ref(&cached)).await;

        assert!(!result.fetched);
        assert_eq!(result.models, std::slice::from_ref(&cached));
    }
    let returned = logged
        .lines(tracing::Level::WARN)
        .into_iter()
        .filter(|w| w.contains("models.dev returned"))
        .count();
    assert_eq!(returned, 3);
}

// spec: models_merge_spec.rb:204 .fetch_models_dev_models keeps the models.dev entries it already had when the fetch fails
#[tokio::test]
async fn fetch_models_dev_models_keeps_cached_entries_when_the_fetch_fails() {
    let logged = Logged::start();
    let mut config = bare_config();
    config.set("models_dev_url", "http://127.0.0.1:9/api.json");
    let cached = model(
        "cached",
        "openai",
        json!({ "metadata": { "source": "models.dev" } }),
    );
    let provider_owned = model(
        "live",
        "openai",
        json!({ "metadata": { "source": "openai" } }),
    );

    let result = fetch_models_dev_models(&config, &[cached.clone(), provider_owned]).await;

    assert!(!result.fetched);
    assert_eq!(result.models, [cached]);
    assert!(
        logged
            .lines(tracing::Level::WARN)
            .iter()
            .any(|w| w.starts_with("Failed to fetch models.dev"))
    );
}

// ---- models.dev field helpers ------------------------------------------------------------------

// spec: models_merge_spec.rb:222 .models_dev_pricing returns nothing when models.dev reports no cost
#[test]
fn models_dev_pricing_is_empty_without_a_cost() {
    assert_eq!(models_dev_pricing(None), json!({}));
}

// spec: models_merge_spec.rb:228 .normalize_models_dev_modalities returns empty lists when models.dev omits modalities
#[test]
fn normalize_models_dev_modalities_is_empty_without_modalities() {
    let modalities = normalize_models_dev_modalities(None);
    assert!(modalities.input.is_empty());
    assert!(modalities.output.is_empty());
}

// The port keeps dates as `YYYY-MM-DD` strings (what the registry JSON holds), so a value that is
// already a date is one in that form.
// spec: models_merge_spec.rb:234 .normalize_models_dev_knowledge passes a Date through untouched
#[test]
fn normalize_models_dev_knowledge_passes_a_date_through() {
    assert_eq!(
        normalize_models_dev_knowledge("2025-01-01").as_deref(),
        Some("2025-01-01")
    );
}

// spec: models_merge_spec.rb:240 .normalize_models_dev_knowledge parses a string cutoff
#[test]
fn normalize_models_dev_knowledge_parses_a_string_cutoff() {
    assert_eq!(
        normalize_models_dev_knowledge("2025-01-01T00:00:00Z").as_deref(),
        Some("2025-01-01")
    );
}

// spec: models_merge_spec.rb:244 .normalize_models_dev_knowledge ignores an unparseable cutoff
#[test]
fn normalize_models_dev_knowledge_ignores_an_unparseable_cutoff() {
    assert_eq!(normalize_models_dev_knowledge("not a date"), None);
    assert_eq!(normalize_models_dev_knowledge(""), None);
}

// spec: models_merge_spec.rb:260 .blank_value? treats nil, empty collections and all-blank hashes as blank
#[test]
fn blank_value_treats_nil_empty_and_all_blank_hashes_as_blank() {
    assert!(is_blank(&Value::Null));
    assert!(is_blank(&json!("")));
    assert!(is_blank(&json!([])));
    assert!(is_blank(&json!({})));
    assert!(is_blank(&json!({ "input": [], "output": null })));
}

// spec: models_merge_spec.rb:268 .blank_value? treats anything with content as present
#[test]
fn blank_value_treats_content_as_present() {
    assert!(!is_blank(&json!("gpt")));
    assert!(!is_blank(&json!(["text"])));
    assert!(!is_blank(&json!({ "input": ["text"] })));
    assert!(!is_blank(&json!(0)));
}

// ---- .add_provider_metadata --------------------------------------------------------------------

// spec: models_merge_spec.rb:277 .add_provider_metadata fills every blank models.dev field from the provider entry
#[test]
fn add_provider_metadata_fills_every_blank_field() {
    let models_dev_model = model(
        "test",
        "openai",
        json!({ "metadata": { "source": "models.dev" } }),
    );
    let provider_model = model(
        "test",
        "openai",
        json!({
            "name": "Test", "family": "test-family", "created_at": "2025-01-01 00:00:00 UTC",
            "context_window": 1000, "max_output_tokens": 100, "knowledge_cutoff": "2024-10-01",
            "modalities": { "input": ["text"], "output": ["text"] },
            "pricing": { "text_tokens": { "standard": { "input_per_million": 1.0 } } },
            "capabilities": ["streaming", "vision"], "metadata": { "provider_note": "kept" }
        }),
    );

    let merged = add_provider_metadata(&models_dev_model, &provider_model);

    assert_eq!(merged.name, "Test");
    assert_eq!(merged.family.as_deref(), Some("test-family"));
    assert_eq!(
        merged.created_at.as_deref(),
        Some("2025-01-01 00:00:00 UTC")
    );
    assert_eq!(merged.context_window, Some(1000));
    assert_eq!(merged.max_output_tokens, Some(100));
    assert_eq!(merged.knowledge_cutoff.as_deref(), Some("2024-10-01"));
    assert_eq!(merged.modalities.input, ["text"]);
    assert_eq!(
        serde_json::to_value(&merged.pricing).unwrap(),
        json!({ "text_tokens": { "standard": { "input_per_million": 1.0 } } })
    );
    assert_eq!(merged.metadata["source"], "models.dev");
    assert_eq!(merged.metadata["provider_note"], "kept");
    assert_eq!(sorted(&merged.capabilities), ["streaming", "vision"]);
}

// spec: models_merge_spec.rb:308 .add_provider_metadata keeps the models.dev values it does have
#[test]
fn add_provider_metadata_keeps_the_models_dev_values_it_has() {
    let models_dev_model = model(
        "test",
        "openai",
        json!({ "knowledge_cutoff": "2025-06-01", "metadata": { "source": "models.dev" } }),
    );
    let provider_model = model(
        "test",
        "openai",
        json!({ "knowledge_cutoff": "2024-10-01" }),
    );

    let merged = add_provider_metadata(&models_dev_model, &provider_model);

    assert_eq!(merged.knowledge_cutoff.as_deref(), Some("2025-06-01"));
}

// spec: models_merge_spec.rb:319 .add_provider_metadata fills missing pricing rates without replacing models.dev rates
#[test]
fn add_provider_metadata_fills_missing_pricing_rates_only() {
    let models_dev_model = model(
        "test",
        "perplexity",
        json!({ "pricing": { "text_tokens": { "standard": { "input_per_million": 1.0 } } } }),
    );
    let provider_model = model(
        "test",
        "perplexity",
        json!({ "pricing": { "text_tokens": { "standard": {
            "input_per_million": 2.0, "output_per_million": 3.0, "cache_write_input_per_million": 0.5
        } } } }),
    );

    let merged = add_provider_metadata(&models_dev_model, &provider_model);

    assert_eq!(
        serde_json::to_value(&merged.pricing).unwrap()["text_tokens"]["standard"],
        json!({ "input_per_million": 1.0, "output_per_million": 3.0, "cache_write_input_per_million": 0.5 })
    );
}

// spec: models_merge_spec.rb:340 .add_provider_metadata keeps a provider capability models.dev does not report on
#[test]
fn add_provider_metadata_keeps_a_capability_models_dev_does_not_report_on() {
    let models_dev_model = model(
        "test",
        "openai",
        json!({
            "capabilities": ["function_calling"],
            "modalities": { "input": ["text"], "output": ["text"] },
            "metadata": { "source": "models.dev", "tool_call": true }
        }),
    );
    let provider_model = model(
        "test",
        "openai",
        json!({ "capabilities": ["function_calling", "structured_output"] }),
    );

    let merged = add_provider_metadata(&models_dev_model, &provider_model);

    assert_eq!(
        sorted(&merged.capabilities),
        sorted([
            "function_calling",
            "tool_choice",
            "parallel_tool_calls",
            "structured_output"
        ])
    );
}

// spec: models_merge_spec.rb:355 .add_provider_metadata drops a provider capability models.dev reports as absent
#[test]
fn add_provider_metadata_drops_a_capability_models_dev_reports_absent() {
    let models_dev_model = model(
        "test",
        "openai",
        json!({
            "capabilities": [],
            "modalities": { "input": ["text"], "output": ["text"] },
            "metadata": { "source": "models.dev", "tool_call": false, "structured_output": false, "reasoning": false }
        }),
    );
    let provider_model = model(
        "test",
        "openai",
        json!({ "capabilities": ["streaming", "function_calling", "structured_output", "reasoning", "vision"] }),
    );

    let merged = add_provider_metadata(&models_dev_model, &provider_model);

    assert_eq!(merged.capabilities, ["streaming"]);
}

// spec: models_merge_spec.rb:371 .add_provider_metadata normalizes embedding modalities on the merged entry
#[test]
fn add_provider_metadata_normalizes_embedding_modalities() {
    let models_dev_model = model("text-embedding-3-small", "openai", json!({}));
    let provider_model = model("text-embedding-3-small", "openai", json!({}));

    let merged = add_provider_metadata(&models_dev_model, &provider_model);

    assert_eq!(merged.modalities.input, ["text"]);
    assert_eq!(merged.modalities.output, ["embeddings"]);
}

// spec: models_merge_spec.rb:381 .add_provider_metadata prefers a provider-reported non-chat operation over a models.dev chat classification
#[test]
fn add_provider_metadata_prefers_a_provider_reported_non_chat_operation() {
    let models_dev_model = model(
        "mistral-embed",
        "mistral",
        json!({ "modalities": { "input": ["text"], "output": ["text"] } }),
    );
    let provider_model = model(
        "mistral-embed",
        "mistral",
        json!({ "modalities": { "input": ["text"], "output": ["embeddings"] } }),
    );

    let merged = add_provider_metadata(&models_dev_model, &provider_model);

    assert_eq!(merged.model_type(), ModelType::Embedding);
    assert_eq!(merged.modalities.output, ["embeddings"]);
}

// ---- .find_models_dev_model --------------------------------------------------------------------

fn by_key(entries: &[(&str, &Model)]) -> std::collections::HashMap<String, Model> {
    entries
        .iter()
        .map(|(k, m)| (k.to_string(), (*m).clone()))
        .collect()
}

// spec: models_merge_spec.rb:397 .find_models_dev_model prefers a direct hit
#[test]
fn find_models_dev_model_prefers_a_direct_hit() {
    let entry = model("gpt-5", "openai", json!({}));
    assert_eq!(
        find_models_dev_model("openai:gpt-5", &by_key(&[("openai:gpt-5", &entry)]), None),
        Some(entry)
    );
}

// spec: models_merge_spec.rb:429 .find_models_dev_model reuses an exact OpenAI base entry for its release-dated snapshot
#[test]
fn find_models_dev_model_reuses_an_openai_base_entry_for_its_dated_snapshot() {
    let entry = model(
        "gpt-5.4-mini",
        "openai",
        json!({ "created_at": "2026-03-17 00:00:00 UTC", "context_window": 400_000 }),
    );

    let found = find_models_dev_model(
        "openai:gpt-5.4-mini-2026-03-17",
        &by_key(&[("openai:gpt-5.4-mini", &entry)]),
        None,
    )
    .unwrap();

    assert_eq!(found.id, "gpt-5.4-mini-2026-03-17");
    assert_eq!(found.context_window, Some(400_000));
}

// spec: models_merge_spec.rb:442 .find_models_dev_model maps OpenAI snapshots whose public release date differs from their base entry
#[test]
fn find_models_dev_model_maps_listed_openai_snapshots() {
    let entry = model("o1", "openai", json!({ "context_window": 200_000 }));

    let found = find_models_dev_model(
        "openai:o1-2024-12-17",
        &by_key(&[("openai:o1", &entry)]),
        None,
    )
    .unwrap();

    assert_eq!(found.id, "o1-2024-12-17");
    assert_eq!(found.context_window, Some(200_000));
}

// spec: models_merge_spec.rb:453 .find_models_dev_model does not inherit OpenAI metadata when the release date differs
#[test]
fn find_models_dev_model_does_not_inherit_across_release_dates() {
    let entry = model(
        "gpt-4o",
        "openai",
        json!({ "created_at": "2024-05-13 00:00:00 UTC", "context_window": 128_000 }),
    );

    let found = find_models_dev_model(
        "openai:gpt-4o-2024-08-06",
        &by_key(&[("openai:gpt-4o", &entry)]),
        None,
    );

    assert_eq!(found, None);
}

// spec: models_merge_spec.rb:465 .find_models_dev_model uses aliases reported by Mistral to find a models.dev entry
#[test]
fn find_models_dev_model_uses_mistral_aliases() {
    let entry = model(
        "mistral-embed",
        "mistral",
        json!({ "context_window": 8_000 }),
    );
    let provider_model = model(
        "mistral-embed-2312",
        "mistral",
        json!({ "metadata": { "aliases": ["mistral-embed"] } }),
    );

    let found = find_models_dev_model(
        "mistral:mistral-embed-2312",
        &by_key(&[("mistral:mistral-embed", &entry)]),
        Some(&provider_model),
    )
    .unwrap();

    assert_eq!(found.id, "mistral-embed-2312");
    assert_eq!(found.context_window, Some(8_000));
}

// spec: models_merge_spec.rb:490 .find_models_dev_model returns nothing when neither provider has the model
#[test]
fn find_models_dev_model_returns_nothing_when_no_provider_has_it() {
    let empty = std::collections::HashMap::new();
    assert_eq!(
        find_models_dev_model("vertexai:missing", &empty, None),
        None
    );
    assert_eq!(find_models_dev_model("openai:missing", &empty, None), None);
}

// ---- .merge_models / .merge_with_existing ------------------------------------------------------

// spec: models_merge_spec.rb:497 .merge_models merges by provider and id, sorting the result
#[test]
fn merge_models_merges_by_provider_and_id_sorted() {
    let provider_models = [
        model("shared", "openai", json!({ "name": "Provider Name" })),
        model("provider-only", "openai", json!({})),
    ];
    let models_dev_models = [
        model(
            "shared",
            "openai",
            json!({ "metadata": { "source": "models.dev" } }),
        ),
        model(
            "dev-only",
            "anthropic",
            json!({ "metadata": { "source": "models.dev" } }),
        ),
    ];

    let merged = merge_models(&provider_models, &models_dev_models);

    let pairs: Vec<(&str, &str)> = merged
        .iter()
        .map(|m| (m.provider.as_str(), m.id.as_str()))
        .collect();
    assert_eq!(
        pairs,
        [
            ("anthropic", "dev-only"),
            ("openai", "provider-only"),
            ("openai", "shared")
        ]
    );
    assert_eq!(
        merged.iter().find(|m| m.id == "shared").unwrap().name,
        "Provider Name"
    );
}

// spec: models_merge_spec.rb:517 .merge_with_existing keeps models from providers that were not refreshed
#[test]
fn merge_with_existing_keeps_models_from_providers_not_refreshed() {
    let existing = [
        model("kept", "anthropic", json!({})),
        model("replaced", "openai", json!({})),
    ];
    let provider_fetch = ProviderFetch {
        models: vec![model("fresh", "openai", json!({}))],
        fetched_providers: vec!["openai".into()],
        ..Default::default()
    };

    let merged = merge_with_existing(
        &existing,
        &provider_fetch,
        &ModelsDevFetch {
            models: vec![],
            fetched: true,
        },
    );

    assert_eq!(sorted(ids(&merged)), ["fresh", "kept"]);
}

// spec: models_merge_spec.rb:529 .merge_with_existing falls back to the existing models.dev entries when the fetch failed
#[test]
fn merge_with_existing_falls_back_to_existing_models_dev_entries() {
    let cached = model(
        "cached",
        "openai",
        json!({ "metadata": { "source": "models.dev" } }),
    );

    let merged = merge_with_existing(
        &[cached],
        &ProviderFetch::default(),
        &ModelsDevFetch {
            models: vec![],
            fetched: false,
        },
    );

    assert_eq!(ids(&merged), ["cached"]);
}

// spec: models_merge_spec.rb:538 .merge_with_existing keeps the models of a provider that listed nothing
#[tokio::test]
async fn merge_with_existing_keeps_the_models_of_a_silent_provider() {
    let existing = [model("kept", "mistral", json!({}))];
    let server = mistral(json!([])).await;
    let mut config = bare_config();
    with_mistral(&mut config, &server);

    let provider_fetch = fetch_provider_models(&Arc::new(config), true).await;
    let merged = merge_with_existing(
        &existing,
        &provider_fetch,
        &ModelsDevFetch {
            models: vec![],
            fetched: true,
        },
    );

    assert_eq!(ids(&merged), ["kept"]);
}

// ---- .read_existing_models / .fetch_merged_models / #refresh_from_providers --------------------

// spec: models_merge_spec.rb:551 .read_existing_models uses the loaded registry when the instance is empty
#[tokio::test]
async fn read_existing_models_uses_the_loaded_registry_when_empty() {
    let _lock = GLOBAL.lock().await;
    let _restore = Restore::new();
    Models::install(vec![]);

    assert!(!read_existing_models().is_empty());
}

// spec: models_merge_spec.rb:557 .read_existing_models uses the instance when it already has models
#[tokio::test]
async fn read_existing_models_uses_the_instance_when_it_has_models() {
    let _lock = GLOBAL.lock().await;
    let _restore = Restore::new();
    Models::install(vec![model("a", "openai", json!({}))]);

    assert_eq!(ids(&read_existing_models()), ["a"]);
}

/// An Anthropic endpoint listing one model, standing in for the spec's `acme` provider.
async fn anthropic(id: &str, display_name: &str) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [{ "id": id, "display_name": display_name }], "has_more": false
        })))
        .mount(&server)
        .await;
    server
}

fn with_anthropic(config: &mut Config, server: &MockServer) {
    config.set("anthropic_api_key", "test");
    config.set("anthropic_api_base", server.uri());
}

// Ruby stubs read_existing_models to []; here the registry holds only an entry of the refreshed
// provider, which the refresh replaces, so the existing models contribute nothing either way.
// spec: models_merge_spec.rb:565 .fetch_merged_models combines provider and models.dev fetches into one catalog
#[tokio::test]
async fn fetch_merged_models_combines_provider_and_models_dev_fetches() {
    let _lock = GLOBAL.lock().await;
    let _restore = Restore::new();
    Models::install(vec![model("stale", "anthropic", json!({}))]);
    let provider = anthropic("acme-1", "Acme One").await;
    // A nameless models.dev entry, like the spec's `model(id: 'acme-1', ...)`.
    let dev = models_dev(
        json!({ "anthropic": { "models": { "acme-1": { "id": "acme-1", "name": "" } } } }),
    )
    .await;
    let mut config = bare_config();
    with_anthropic(&mut config, &provider);
    with_models_dev(&mut config, &dev);

    let merged = fetch_merged_models(&Arc::new(config), false).await;

    assert_eq!(ids(&merged), ["acme-1"]);
    assert_eq!(merged[0].name, "Acme One");
    assert_eq!(merged[0].metadata["source"], "models.dev");
}

// spec: models_merge_spec.rb:589 #refresh_from_providers replaces the registry with the merged provider catalog
#[tokio::test]
async fn refresh_from_providers_replaces_the_registry() {
    let _lock = GLOBAL.lock().await;
    let _restore = Restore::new();
    Models::install(vec![model("stale", "anthropic", json!({}))]);
    let provider = anthropic("only", "only").await;
    let dev = models_dev(json!({ "anthropic": { "models": { "only": { "id": "only" } } } })).await;
    let mut config = bare_config();
    with_anthropic(&mut config, &provider);
    with_models_dev(&mut config, &dev);
    rust_llm::configure(|c| *c = config);

    let registry = refresh_from_providers(false).await.unwrap();

    let pairs: Vec<(&str, &str)> = registry
        .all()
        .iter()
        .map(|m| (m.provider.as_str(), m.id.as_str()))
        .collect();
    assert_eq!(pairs, [("anthropic", "only")]);
    assert_eq!(
        ids(&rust_llm::models()
            .all()
            .into_iter()
            .cloned()
            .collect::<Vec<_>>()),
        ["only"]
    );
}

// ---- model registry stores ---------------------------------------------------------------------

/// A store answering `read` with `models` and recording each `write`.
#[derive(Default)]
struct RecordingStore {
    models: Vec<Model>,
    writes: Mutex<Vec<Vec<String>>>,
}

impl ModelRegistryStore for RecordingStore {
    fn read(&self) -> rust_llm::Result<Vec<Model>> {
        Ok(self.models.clone())
    }
    fn write(&self, models: &Models) -> rust_llm::Result<()> {
        let ids = models.all().iter().map(|m| m.id.clone()).collect();
        self.writes.lock().unwrap().push(ids);
        Ok(())
    }
}

/// A store with no `write` (Ruby's store that does not `respond_to?(:write)`).
struct ReadOnlyStore;

impl ModelRegistryStore for ReadOnlyStore {
    fn read(&self) -> rust_llm::Result<Vec<Model>> {
        Ok(vec![])
    }
}

struct DescribedStore;

impl ModelRegistryStore for DescribedStore {
    fn read(&self) -> rust_llm::Result<Vec<Model>> {
        Ok(vec![])
    }
    fn write(&self, _: &Models) -> rust_llm::Result<()> {
        Err(std::io::Error::other("disk full").into())
    }
    fn description(&self) -> String {
        "the Rails model table".into()
    }
}

struct AnonymousStore;

impl ModelRegistryStore for AnonymousStore {
    fn read(&self) -> rust_llm::Result<Vec<Model>> {
        Ok(vec![])
    }
    fn write(&self, _: &Models) -> rust_llm::Result<()> {
        Err(std::io::Error::other("disk full").into())
    }
}

// spec: models_merge_spec.rb:600 .models_from_store ignores an empty store
#[test]
fn models_from_store_ignores_an_empty_store() {
    let store = RecordingStore::default();
    assert_eq!(
        models_from_store(Some(&store as &dyn ModelRegistryStore)),
        None
    );
}

// spec: models_merge_spec.rb:607 .models_from_store returns nothing when no store is configured
#[test]
fn models_from_store_returns_nothing_without_a_store() {
    assert_eq!(models_from_store(None), None);
}

// spec: models_merge_spec.rb:649 #load_from_store raises when no store is configured
#[tokio::test]
async fn load_from_store_raises_without_a_store() {
    let _lock = GLOBAL.lock().await;
    let _restore = Restore::new();
    rust_llm::configure(|c| c.model_registry_store = None);

    let error = Models::new(vec![])
        .load_from_store()
        .map(|_| ())
        .unwrap_err();

    assert!(matches!(error, Error::ModelRegistry(_)));
    assert_eq!(error.to_string(), "No model registry store is configured");
}

// spec: models_merge_spec.rb:655 #load_from_store replaces the registry with the store contents
#[tokio::test]
async fn load_from_store_replaces_the_registry_with_the_store_contents() {
    let _lock = GLOBAL.lock().await;
    let _restore = Restore::new();
    let stored = vec![model("stored", "openai", json!({}))];
    let store = RecordingStore {
        models: stored.clone(),
        ..Default::default()
    };
    rust_llm::configure(|c| c.model_registry_store = Some(Arc::new(store)));

    let mut registry = Models::new(vec![]);
    registry.load_from_store().unwrap();

    assert_eq!(
        registry.all().into_iter().cloned().collect::<Vec<_>>(),
        stored
    );
}

/// A published-catalog server answering with `models` and an `etag-1` ETag.
async fn published(models: Value) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/models.json"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("etag", "etag-1")
                .set_body_json(models),
        )
        .mount(&server)
        .await;
    server
}

fn with_published(config: &mut Config, server: &MockServer) {
    config.set(
        "model_registry_url",
        format!("{}/models.json", server.uri()),
    );
}

fn gpt_x() -> Value {
    json!([{ "id": "gpt-x", "name": "GPT X", "provider": "openai", "context_window": 400_000 }])
}

/// `Models.new([]).refresh` with `config`, `persist_registry!` included.
async fn refresh_empty(config: Config) -> rust_llm::Result<Arc<Models>> {
    Models::install(vec![]);
    refresh_with_config(Arc::new(config), false).await
}

// spec: models_merge_spec.rb:669 #persist_registry! writes through a store that supports it
#[tokio::test]
async fn persist_registry_writes_through_a_store_that_supports_it() {
    let _lock = GLOBAL.lock().await;
    let _restore = Restore::new();
    let catalog = published(gpt_x()).await;
    let store = Arc::new(RecordingStore::default());
    let mut config = bare_config();
    with_published(&mut config, &catalog);
    config.model_registry_store = Some(store.clone());

    refresh_empty(config).await.unwrap();

    // The store receives a whole `Models` registry.
    assert_eq!(*store.writes.lock().unwrap(), [vec!["gpt-x".to_string()]]);
}

// spec: models_merge_spec.rb:680 #persist_registry! rejects a read-only store
#[tokio::test]
async fn persist_registry_rejects_a_read_only_store() {
    let _lock = GLOBAL.lock().await;
    let _restore = Restore::new();
    let catalog = published(gpt_x()).await;
    let mut config = bare_config();
    with_published(&mut config, &catalog);
    config.model_registry_store = Some(Arc::new(ReadOnlyStore));

    let error = refresh_empty(config).await.map(|_| ()).unwrap_err();

    assert!(matches!(error, Error::ModelRegistry(_)));
    assert!(error.to_string().ends_with("is read-only"), "{error}");
}

// spec: models_merge_spec.rb:688 #persist_registry! rejects a configuration with nowhere to write
#[tokio::test]
async fn persist_registry_rejects_a_configuration_with_nowhere_to_write() {
    let _lock = GLOBAL.lock().await;
    let _restore = Restore::new();
    let catalog = published(gpt_x()).await;
    let mut config = bare_config();
    with_published(&mut config, &catalog);

    let error = refresh_empty(config).await.map(|_| ()).unwrap_err();

    assert!(matches!(error, Error::ModelRegistry(_)));
    assert_eq!(
        error.to_string(),
        "No writable model registry store is configured"
    );
}

// spec: models_merge_spec.rb:696 #persist_registry! names the store in the error when writing blows up
#[tokio::test]
async fn persist_registry_names_the_store_when_writing_fails() {
    let _lock = GLOBAL.lock().await;
    let _restore = Restore::new();
    let catalog = published(gpt_x()).await;
    let mut config = bare_config();
    with_published(&mut config, &catalog);
    config.model_registry_store = Some(Arc::new(DescribedStore));

    let error = refresh_empty(config).await.map(|_| ()).unwrap_err();

    assert!(matches!(error, Error::ModelRegistry(_)));
    assert_eq!(
        error.to_string(),
        "Could not save the model registry to the Rails model table: disk full"
    );
}

// Ruby's fallback is the store's class name; the port's is its type name, module path included.
// spec: models_merge_spec.rb:707 #persist_registry! falls back to the store class name when it has no description
#[tokio::test]
async fn persist_registry_falls_back_to_the_store_type_name() {
    let _lock = GLOBAL.lock().await;
    let _restore = Restore::new();
    let catalog = published(gpt_x()).await;
    let mut config = bare_config();
    with_published(&mut config, &catalog);
    config.model_registry_store = Some(Arc::new(AnonymousStore));

    let error = refresh_empty(config).await.map(|_| ()).unwrap_err();

    let message = error.to_string();
    assert!(
        message.starts_with("Could not save the model registry to ")
            && message.ends_with("AnonymousStore: disk full"),
        "{message}"
    );
}

// ---- refresh against the published catalog -----------------------------------------------------

/// A scratch directory for `model_registry_file`, removed on drop.
struct TempDir(std::path::PathBuf);

impl TempDir {
    fn new(name: &str) -> TempDir {
        let dir = std::env::temp_dir().join(format!(
            "rust_llm_models_merge_{}_{name}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        TempDir(dir)
    }

    fn file(&self) -> std::path::PathBuf {
        self.0.join("models.json")
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

// Private `published_catalog`: a corrupt snapshot counts as no cache, so no ETag is sent.
// spec: models_merge_spec.rb:720 #published_catalog treats a corrupt snapshot as no cache
#[tokio::test]
async fn published_catalog_treats_a_corrupt_snapshot_as_no_cache() {
    let _lock = GLOBAL.lock().await;
    let _restore = Restore::new();
    let dir = TempDir::new("corrupt");
    std::fs::write(dir.file(), "[]").unwrap();
    std::fs::write(dir.0.join("models.json.etag"), "etag-1\n").unwrap();
    std::fs::write(dir.0.join("models.json.published.json"), "{broken").unwrap();
    let catalog = MockServer::start().await;
    Mock::given(header_exists("If-None-Match"))
        .respond_with(ResponseTemplate::new(304))
        .expect(0)
        .mount(&catalog)
        .await;
    Mock::given(method("GET"))
        .and(path("/models.json"))
        .respond_with(ResponseTemplate::new(200).set_body_json(gpt_x()))
        .mount(&catalog)
        .await;
    let mut config = bare_config();
    with_published(&mut config, &catalog);
    config.model_registry_file = Some(dir.file());

    let registry = refresh_empty(config).await.unwrap();

    assert_eq!(
        registry
            .all()
            .iter()
            .map(|m| m.id.as_str())
            .collect::<Vec<_>>(),
        ["gpt-x"]
    );
    catalog.verify().await;
}

/// Mistral listing `llama` with `context` tokens, standing in for the spec's local provider.
async fn local_provider(server: &MockServer, context: i64) {
    server.reset().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(
                json!({ "data": [{ "id": "llama", "max_context_length": context }] }),
            ),
        )
        .mount(server)
        .await;
}

// spec: models_merge_spec.rb:753 #refresh against the published catalog prunes and refreshes on a not-modified answer just as it does on a fresh one
#[tokio::test]
async fn refresh_prunes_and_refreshes_on_a_not_modified_answer() {
    let _lock = GLOBAL.lock().await;
    let _restore = Restore::new();
    let dir = TempDir::new("not_modified");
    let provider = MockServer::start().await;
    local_provider(&provider, 8192).await;
    let catalog = published(gpt_x()).await;
    let mut config = bare_config();
    with_mistral(&mut config, &provider);
    with_published(&mut config, &catalog);
    config.model_registry_file = Some(dir.file());
    refresh_empty(config.clone()).await.unwrap();

    let mut stale: Vec<Model> = rust_llm::models().all().into_iter().cloned().collect();
    stale.push(model("retired", "openai", json!({})));
    Models::install(stale);
    local_provider(&provider, 131_072).await;
    catalog.reset().await;
    Mock::given(header_exists("If-None-Match"))
        .respond_with(ResponseTemplate::new(304).insert_header("etag", "etag-1"))
        .mount(&catalog)
        .await;
    let registry = refresh_with_config(Arc::new(config), false).await.unwrap();

    assert_eq!(
        sorted(registry.all().iter().map(|m| m.id.as_str())),
        ["gpt-x", "llama"]
    );
    assert_eq!(
        registry
            .find("llama", Some("mistral"))
            .unwrap()
            .context_window,
        Some(131_072)
    );
}

// spec: models_merge_spec.rb:767 #refresh against the published catalog fetches the whole catalog again when the snapshot behind the ETag is gone
#[tokio::test]
async fn refresh_fetches_the_whole_catalog_again_when_the_snapshot_is_gone() {
    let _lock = GLOBAL.lock().await;
    let _restore = Restore::new();
    let dir = TempDir::new("snapshot_gone");
    let provider = MockServer::start().await;
    local_provider(&provider, 8192).await;
    let catalog = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(304).insert_header("etag", "etag-1"))
        .up_to_n_times(1)
        .with_priority(1)
        .mount(&catalog)
        .await;
    Mock::given(method("GET"))
        .and(path("/models.json"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("etag", "etag-1")
                .set_body_json(gpt_x()),
        )
        .mount(&catalog)
        .await;
    let mut config = bare_config();
    with_mistral(&mut config, &provider);
    with_published(&mut config, &catalog);
    config.model_registry_file = Some(dir.file());

    let registry = refresh_empty(config).await.unwrap();

    assert_eq!(
        sorted(registry.all().iter().map(|m| m.id.as_str())),
        ["gpt-x", "llama"]
    );
    assert_eq!(catalog.received_requests().await.unwrap().len(), 2);
}

// Private `file_store`: with a store configured, the registry file is never written.
// spec: models_merge_spec.rb:820 #file_store is nil when a store takes precedence
#[tokio::test]
async fn file_store_is_nil_when_a_store_takes_precedence() {
    let _lock = GLOBAL.lock().await;
    let _restore = Restore::new();
    let dir = TempDir::new("store_precedence");
    let catalog = published(gpt_x()).await;
    let store = Arc::new(RecordingStore::default());
    let mut config = bare_config();
    with_published(&mut config, &catalog);
    config.model_registry_file = Some(dir.file());
    config.model_registry_store = Some(store.clone());

    refresh_empty(config).await.unwrap();

    assert!(!dir.file().exists());
    assert!(!dir.0.join("models.json.published.json").exists());
    assert_eq!(store.writes.lock().unwrap().len(), 1);
}

// Private `file_store`: with no registry file there is nothing to revalidate against or write to.
// spec: models_merge_spec.rb:826 #file_store is nil when no registry file is configured
#[tokio::test]
async fn file_store_is_nil_without_a_registry_file() {
    let _lock = GLOBAL.lock().await;
    let _restore = Restore::new();
    let catalog = MockServer::start().await;
    Mock::given(header_exists("If-None-Match"))
        .respond_with(ResponseTemplate::new(304))
        .expect(0)
        .mount(&catalog)
        .await;
    Mock::given(method("GET"))
        .and(path("/models.json"))
        .respond_with(ResponseTemplate::new(200).set_body_json(gpt_x()))
        .mount(&catalog)
        .await;
    let mut config = bare_config();
    with_published(&mut config, &catalog);
    config.model_registry_file = None;

    let error = refresh_empty(config).await.map(|_| ()).unwrap_err();

    assert_eq!(
        error.to_string(),
        "No writable model registry store is configured"
    );
    catalog.verify().await;
}

// Private `resolve_provider_registry_id`, seen through `find`: a provider RustLLM does not know
// leaves the id as given, so the registry entry filed under it is still found.
// spec: models_merge_spec.rb:834 #resolve_provider_registry_id leaves the id alone for an unknown provider
#[test]
fn resolve_provider_registry_id_leaves_the_id_alone_for_an_unknown_provider() {
    let registry = Models::new(vec![model("some-model", "nowhere", json!({}))]);

    assert_eq!(
        registry.find("some-model", Some("nowhere")).unwrap().id,
        "some-model"
    );
}

// spec: models_merge_spec.rb:842 .resolve requires a provider when assuming the model exists
#[test]
fn resolve_requires_a_provider_when_assuming_the_model_exists() {
    let error = Chat::with_config(Arc::new(bare_config()), Some("anything"), None, true)
        .map(|_| ())
        .unwrap_err();

    assert!(matches!(error, Error::Argument(_)));
    assert_eq!(
        error.to_string(),
        "Provider must be specified if assume_model_exists is true"
    );
}

// ---- models_spec.rb .models_dev_model_attributes -----------------------------------------------

fn model_data() -> Value {
    json!({
        "id": "gpt-test-1",
        "name": "GPT Test 1",
        "family": "gpt-test",
        "last_updated": "2025-02-01",
        "knowledge": "2024-01-01",
        "modalities": { "input": ["text", "image"], "output": ["text", "pdf"] },
        "tool_call": true,
        "structured_output": true,
        "reasoning": false,
        "reasoning_options": [
            { "type": "effort", "values": ["low", "high"] },
            { "type": "budget_tokens", "min": 1024 }
        ],
        "cost": { "input": 1.25, "output": 5.0, "cache_read": 0.5, "cache_write": 2.5, "reasoning": 10.0 },
        "limit": { "context": 128_000, "output": 4096 }
    })
}

fn merged_data(extra: Value) -> Value {
    let mut data = model_data();
    for (k, v) in extra.as_object().cloned().unwrap_or_default() {
        data[k] = v;
    }
    data
}

fn attributes(data: &Value, slug: &str, key: &str) -> Model {
    models_dev_model_attributes(data, slug, key).unwrap()
}

// spec: models_spec.rb:255 .models_dev_model_attributes converts models.dev payload into a Model attributes hash
#[test]
fn models_dev_model_attributes_converts_the_payload() {
    let data = model_data();
    let model = attributes(&data, "openai", "openai");

    assert_eq!(
        (
            model.id.as_str(),
            model.name.as_str(),
            model.provider.as_str()
        ),
        ("gpt-test-1", "GPT Test 1", "openai")
    );
    assert_eq!(model.family.as_deref(), Some("gpt-test"));
    assert_eq!(model.context_window, Some(128_000));
    assert_eq!(model.max_output_tokens, Some(4096));
    assert_eq!(model.knowledge_cutoff.as_deref(), Some("2024-01-01"));
    assert_eq!(model.modalities.input, ["text", "image"]);
    assert_eq!(model.modalities.output, ["text"]);
    assert_eq!(
        sorted(&model.capabilities),
        sorted([
            "function_calling",
            "tool_choice",
            "parallel_tool_calls",
            "reasoning",
            "structured_output",
            "vision"
        ])
    );
    let attrs = serde_json::to_value(&model).unwrap();
    assert!(attrs.get("reasoning_options").is_none());
    assert_eq!(
        serde_json::to_value(&model.pricing).unwrap(),
        json!({ "text_tokens": { "standard": {
            "input_per_million": 1.25,
            "output_per_million": 5.0,
            "cache_read_input_per_million": 0.5,
            "cache_write_input_per_million": 2.5,
            "reasoning_output_per_million": 10.0
        } } })
    );
    assert_eq!(model.metadata["source"], "models.dev");
    assert_eq!(model.metadata["provider_id"], "openai");
    assert_eq!(model.metadata["last_updated"], "2025-02-01");
    assert_eq!(model.metadata["cost"], data["cost"]);
    assert_eq!(model.metadata["limit"], data["limit"]);
    assert_eq!(model.metadata["knowledge"], data["knowledge"]);
    assert_eq!(
        model.metadata["reasoning_options"],
        data["reasoning_options"]
    );
}

// spec: models_spec.rb:300 .models_dev_model_attributes derives transcription from audio input and text output
#[test]
fn models_dev_model_attributes_derives_transcription() {
    let data =
        merged_data(json!({ "modalities": { "input": ["text", "audio"], "output": ["text"] } }));
    assert!(
        attributes(&data, "gemini", "google")
            .capabilities
            .contains(&"transcription".to_string())
    );
}

// spec: models_spec.rb:310 .models_dev_model_attributes does not mark audio-output models as transcription models
#[test]
fn models_dev_model_attributes_skips_transcription_for_audio_output() {
    let data = merged_data(json!({ "modalities": { "input": ["text"], "output": ["audio"] } }));
    assert!(
        !attributes(&data, "gemini", "google")
            .capabilities
            .contains(&"transcription".to_string())
    );
}

// spec: models_spec.rb:320 .models_dev_model_attributes does not infer transcription for other providers
#[test]
fn models_dev_model_attributes_does_not_infer_transcription_for_other_providers() {
    let data = merged_data(json!({ "modalities": { "input": ["audio"], "output": ["text"] } }));
    assert!(
        !attributes(&data, "openrouter", "openrouter")
            .capabilities
            .contains(&"transcription".to_string())
    );
}

// spec: models_spec.rb:330 .models_dev_model_attributes recognizes explicitly named OpenAI transcription models
#[test]
fn models_dev_model_attributes_recognizes_openai_transcription_models() {
    let data = merged_data(json!({
        "id": "gpt-4o-transcribe", "modalities": { "input": ["audio"], "output": ["text"] }
    }));
    assert!(
        attributes(&data, "openai", "openai")
            .capabilities
            .contains(&"transcription".to_string())
    );
}

// spec: models_spec.rb:340 .models_dev_model_attributes does not mark multimodal embedding models as transcription models
#[test]
fn models_dev_model_attributes_skips_transcription_for_embedding_models() {
    let data = merged_data(json!({
        "id": "gemini-embedding-2", "modalities": { "input": ["audio"], "output": ["text"] }
    }));
    assert!(
        !attributes(&data, "gemini", "google")
            .capabilities
            .contains(&"transcription".to_string())
    );
}

// spec: models_spec.rb:350 .models_dev_model_attributes maps models.dev context tiers into text_tokens.long_context
#[test]
fn models_dev_model_attributes_maps_context_tiers() {
    let data = merged_data(json!({ "cost": {
        "input": 5.0, "output": 30.0, "cache_read": 0.5, "cache_write": 6.25,
        "tiers": [{
            "input": 10.0, "output": 45.0, "cache_read": 1.0, "cache_write": 12.5,
            "tier": { "type": "context", "size": 272_000 }
        }]
    } }));

    let model = attributes(&data, "openai", "openai");

    assert_eq!(
        serde_json::to_value(&model.pricing).unwrap(),
        json!({ "text_tokens": {
            "standard": {
                "input_per_million": 5.0, "output_per_million": 30.0,
                "cache_read_input_per_million": 0.5, "cache_write_input_per_million": 6.25
            },
            "long_context": {
                "input_per_million": 10.0, "output_per_million": 45.0,
                "cache_read_input_per_million": 1.0, "cache_write_input_per_million": 12.5
            },
            "long_context_threshold": 272_000
        } })
    );
}

// spec: models_spec.rb:390 .models_dev_model_attributes falls back to context_over_200k when models.dev omits structured tiers
#[test]
fn models_dev_model_attributes_falls_back_to_context_over_200k() {
    let data = merged_data(json!({ "cost": {
        "input": 1.25, "output": 10.0, "cache_read": 0.125,
        "context_over_200k": { "input": 2.5, "output": 15.0, "cache_read": 0.25 }
    } }));

    let text = serde_json::to_value(&attributes(&data, "google", "google").pricing).unwrap()
        ["text_tokens"]
        .clone();

    assert_eq!(
        text["long_context"],
        json!({ "input_per_million": 2.5, "output_per_million": 15.0, "cache_read_input_per_million": 0.25 })
    );
    assert_eq!(text["long_context_threshold"], 200_000);
}

// spec: models_spec.rb:416 .models_dev_model_attributes keeps models.dev authoritative for the capabilities it reports on when merging provider metadata
#[test]
fn add_provider_metadata_keeps_models_dev_authoritative_for_reported_capabilities() {
    let models_dev_model = model(
        "test-model",
        "xai",
        json!({
            "name": "Test Model", "context_window": 1000, "max_output_tokens": 100,
            "modalities": { "input": ["text"], "output": ["text"] },
            "capabilities": ["function_calling"],
            "metadata": { "source": "models.dev", "tool_call": true, "reasoning": false }
        }),
    );
    let provider_model = model(
        "test-model",
        "xai",
        json!({
            "name": "Test Model", "context_window": 1000, "max_output_tokens": 100,
            "capabilities": ["streaming", "function_calling", "reasoning", "vision", "structured_output"]
        }),
    );

    let merged = add_provider_metadata(&models_dev_model, &provider_model);

    assert_eq!(
        sorted(&merged.capabilities),
        sorted(["function_calling", "streaming", "structured_output"])
    );
}

// spec: models_spec.rb:441 .models_dev_model_attributes uses release_date cast to midnight as created_at
#[test]
fn models_dev_model_attributes_uses_release_date_as_created_at() {
    let data = merged_data(json!({ "release_date": "2025-03-01" }));
    assert_eq!(
        attributes(&data, "openai", "openai").created_at.as_deref(),
        Some("2025-03-01 00:00:00 UTC")
    );
}

// spec: models_spec.rb:447 .models_dev_model_attributes normalizes month-only release dates to the first day of the month
#[test]
fn models_dev_model_attributes_normalizes_month_only_release_dates() {
    let data = merged_data(json!({ "release_date": "2025-09" }));
    assert_eq!(
        attributes(&data, "openai", "openai").created_at.as_deref(),
        Some("2025-09-01 00:00:00 UTC")
    );
}

// spec: models_spec.rb:455 .models_dev_model_attributes falls back to last_updated cast to midnight as created_at when release_date is missing
#[test]
fn models_dev_model_attributes_falls_back_to_last_updated() {
    let data = merged_data(json!({ "release_date": null, "last_updated": "2025-03-01" }));
    assert_eq!(
        attributes(&data, "openai", "openai").created_at.as_deref(),
        Some("2025-03-01 00:00:00 UTC")
    );
}

// spec: models_spec.rb:461 .models_dev_model_attributes keeps created_at nil when both release_date and last_updated are missing
#[test]
fn models_dev_model_attributes_keeps_created_at_nil_without_dates() {
    let data = merged_data(json!({ "release_date": null, "last_updated": null }));
    assert_eq!(attributes(&data, "openai", "openai").created_at, None);
}

// ---- models_refresh_spec.rb .models_dev_model_id -----------------------------------------------

// spec: models_refresh_spec.rb:97 .models_dev_model_id leaves bare ids and other providers untouched
#[test]
fn models_dev_model_id_leaves_bare_ids_and_other_providers_untouched() {
    assert_eq!(
        models_dev_model_id("gemini-2.5-flash", "vertexai"),
        "gemini-2.5-flash"
    );
    assert_eq!(models_dev_model_id("some@thing", "gemini"), "some@thing");
}

// ---- support/utils_spec.rb ISO date prefixes ---------------------------------------------------

fn date(y: i32, m: u32, d: u32) -> chrono::NaiveDate {
    chrono::NaiveDate::from_ymd_opt(y, m, d).unwrap()
}

// spec: support/utils_spec.rb:68 .parse_iso_date_prefix parses a full ISO date
#[test]
fn parse_iso_date_prefix_parses_a_full_date() {
    assert_eq!(parse_iso_date_prefix("2025-09-15"), Some(date(2025, 9, 15)));
}

// spec: support/utils_spec.rb:72 .parse_iso_date_prefix normalizes a month-only ISO date to the first day of the month
#[test]
fn parse_iso_date_prefix_normalizes_a_month_only_date() {
    assert_eq!(parse_iso_date_prefix("2025-09"), Some(date(2025, 9, 1)));
}

// spec: support/utils_spec.rb:76 .parse_iso_date_prefix normalizes a year-only ISO date to the first day of the year
#[test]
fn parse_iso_date_prefix_normalizes_a_year_only_date() {
    assert_eq!(parse_iso_date_prefix("2025"), Some(date(2025, 1, 1)));
}

// spec: support/utils_spec.rb:80 .parse_iso_date_prefix returns nil for blank and invalid values
#[test]
fn parse_iso_date_prefix_returns_nothing_for_blank_or_invalid_values() {
    assert_eq!(parse_iso_date_prefix(""), None);
    assert_eq!(parse_iso_date_prefix("2025-13"), None);
}

// spec: support/utils_spec.rb:87 .iso_date_prefix_to_utc_midnight_string formats a full ISO date as a UTC midnight timestamp
#[test]
fn iso_date_prefix_to_utc_midnight_string_formats_a_full_date() {
    assert_eq!(
        iso_date_prefix_to_utc_midnight_string("2025-09-15").as_deref(),
        Some("2025-09-15 00:00:00 UTC")
    );
}

// spec: support/utils_spec.rb:91 .iso_date_prefix_to_utc_midnight_string formats a partial ISO date as a UTC midnight timestamp
#[test]
fn iso_date_prefix_to_utc_midnight_string_formats_a_partial_date() {
    assert_eq!(
        iso_date_prefix_to_utc_midnight_string("2025-09").as_deref(),
        Some("2025-09-01 00:00:00 UTC")
    );
}

// spec: support/utils_spec.rb:95 .iso_date_prefix_to_utc_midnight_string returns nil for blank and invalid values
#[test]
fn iso_date_prefix_to_utc_midnight_string_returns_nothing_for_blank_or_invalid_values() {
    assert_eq!(iso_date_prefix_to_utc_midnight_string(""), None);
    assert_eq!(iso_date_prefix_to_utc_midnight_string("2025-13"), None);
}
