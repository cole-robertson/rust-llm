//! RubyLLM 2.0's `models_refresh_spec.rb`, `models_merge_spec.rb`, `models_local_refresh_spec.rb`,
//! the provider `models_spec.rb` parsers, and `providers/perplexity/models_spec.rb` replayed from
//! its cassette.
//!
//! The refresh tests change the process-wide registry, so they hold `REGISTRY_LOCK` and restore
//! the bundled registry when done.

mod support;

use std::sync::{Arc, Mutex};

use rust_llm::model::{Model, ModelType, Modalities};
use rust_llm::models::refresh::{
    ModelsDevFetch, ProviderFetch, add_provider_metadata, augment_capabilities, fetch_models_dev_models, gpustack_model,
    is_blank, merge_models, merge_with_existing, models_dev_pricing, normalize_models_dev_knowledge,
    normalize_models_dev_modalities, parse_anthropic_models, parse_gemini_models, parse_mistral_models, parse_openai_models,
    parse_openrouter_models,
};
use rust_llm::models::{Models, registry};
use rust_llm::{Config, Error, Provider};
use serde_json::{Value, json};
use support::Cassette;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

static REGISTRY_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

fn model(id: &str, provider: &str, extra: Value) -> Model {
    let mut data = json!({ "id": id, "name": id, "provider": provider });
    for (k, v) in extra.as_object().cloned().unwrap_or_default() {
        data[k] = v;
    }
    serde_json::from_value(data).unwrap()
}

fn sorted(mut v: Vec<String>) -> Vec<String> {
    v.sort();
    v
}

fn caps(m: &Model) -> Vec<String> {
    sorted(m.capabilities.clone())
}

// ---- models_dev ---------------------------------------------------------------------------------

async fn serve_models_dev(body: Value) -> (MockServer, Config) {
    let server = MockServer::start().await;
    Mock::given(method("GET")).and(path("/api.json")).respond_with(ResponseTemplate::new(200).set_body_json(body)).mount(&server).await;
    let mut config = Config::default();
    config.set("models_dev_url", format!("{}/api.json", server.uri()));
    (server, config)
}

// spec: models_merge_spec.rb "maps the models.dev catalog onto Model instances"
#[tokio::test]
async fn maps_the_models_dev_catalog_onto_models() {
    let (_server, config) = serve_models_dev(json!({
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
    let result = fetch_models_dev_models(&config, &[]).await;
    assert!(result.fetched);
    let mut pairs: Vec<(String, String)> = result.models.iter().map(|m| (m.provider.clone(), m.id.clone())).collect();
    pairs.sort();
    assert_eq!(
        pairs,
        [("openai".into(), "gpt-test".into()), ("perplexity".into(), "anthropic/claude-test".into()), ("vertexai".into(), "claude-haiku-4-5".into())]
    );
    let openai = result.models.iter().find(|m| m.provider == "openai").unwrap();
    assert_eq!(openai.modalities.input, ["text", "video"]);
    assert_eq!(openai.modalities.output, ["text"]);
    assert_eq!(caps(openai), sorted(["function_calling", "tool_choice", "parallel_tool_calls", "vision", "video"].map(String::from).to_vec()));
    assert_eq!(
        serde_json::to_value(&openai.pricing).unwrap(),
        json!({
            "text_tokens": { "standard": { "input_per_million": 1.0, "output_per_million": 2.0 } },
            "audio_tokens": { "standard": { "input_per_million": 3.0, "output_per_million": 4.0 } }
        })
    );
    assert_eq!(openai.metadata["source"], "models.dev");
}

// spec: "keeps the models.dev entries it already had when the answer carries no models"
#[tokio::test]
async fn keeps_models_dev_entries_when_the_answer_carries_no_models() {
    let cached = model("cached", "openai", json!({ "metadata": { "source": "models.dev" } }));
    for body in [json!({}), Value::Null, json!({ "openai-inc": { "models": { "a": { "id": "a" } } } })] {
        let (_server, config) = serve_models_dev(body).await;
        let result = fetch_models_dev_models(&config, std::slice::from_ref(&cached)).await;
        assert!(!result.fetched);
        assert_eq!(result.models, std::slice::from_ref(&cached));
    }
}

// spec: "keeps the models.dev entries it already had when the fetch fails"
#[tokio::test]
async fn keeps_models_dev_entries_when_the_fetch_fails() {
    let mut config = Config::default();
    config.set("models_dev_url", "http://127.0.0.1:9/api.json");
    let cached = model("cached", "openai", json!({ "metadata": { "source": "models.dev" } }));
    let live = model("live", "openai", json!({ "metadata": { "source": "openai" } }));
    let result = fetch_models_dev_models(&config, &[cached.clone(), live]).await;
    assert!(!result.fetched);
    assert_eq!(result.models, [cached]);
}

#[test]
fn models_dev_helpers() {
    // spec: ".models_dev_pricing returns nothing when models.dev reports no cost"
    assert_eq!(models_dev_pricing(None), json!({}));
    // spec: ".normalize_models_dev_modalities returns empty lists when models.dev omits modalities"
    assert_eq!(normalize_models_dev_modalities(None), Modalities::default());
    // spec: ".normalize_models_dev_knowledge"
    assert_eq!(normalize_models_dev_knowledge("2025-01-01").as_deref(), Some("2025-01-01"));
    assert_eq!(normalize_models_dev_knowledge("not a date"), None);
    // A context tier becomes long-context pricing with its threshold.
    assert_eq!(
        models_dev_pricing(Some(&json!({ "input": 1.0, "context_over_200k": { "input": 2.0 } }))),
        json!({ "text_tokens": {
            "standard": { "input_per_million": 1.0 },
            "long_context": { "input_per_million": 2.0 },
            "long_context_threshold": 200000
        } })
    );
}

// spec: ".blank_value?"
#[test]
fn blank_values() {
    for v in [Value::Null, json!(""), json!([]), json!({}), json!({ "input": [], "output": null })] {
        assert!(is_blank(&v), "{v}");
    }
    for v in [json!("gpt"), json!(["text"]), json!({ "input": ["text"] }), json!(0)] {
        assert!(!is_blank(&v), "{v}");
    }
}

// ---- add_provider_metadata -----------------------------------------------------------------------

// spec: "fills every blank models.dev field from the provider entry"
#[test]
fn fills_every_blank_models_dev_field_from_the_provider() {
    let dev = model("test", "openai", json!({ "metadata": { "source": "models.dev" } }));
    let mut dev_blank_name = dev.clone();
    dev_blank_name.name = String::new();
    let provider = model("test", "openai", json!({
        "name": "Test", "family": "test-family", "created_at": "2025-01-01 00:00:00 UTC",
        "context_window": 1000, "max_output_tokens": 100, "knowledge_cutoff": "2024-10-01",
        "modalities": { "input": ["text"], "output": ["text"] },
        "pricing": { "text_tokens": { "standard": { "input_per_million": 1.0 } } },
        "capabilities": ["streaming", "vision"], "metadata": { "provider_note": "kept" }
    }));
    let merged = add_provider_metadata(&dev_blank_name, &provider);
    assert_eq!(merged.name, "Test");
    assert_eq!(merged.family.as_deref(), Some("test-family"));
    assert_eq!(merged.created_at.as_deref(), Some("2025-01-01 00:00:00 UTC"));
    assert_eq!(merged.context_window, Some(1000));
    assert_eq!(merged.max_output_tokens, Some(100));
    assert_eq!(merged.knowledge_cutoff.as_deref(), Some("2024-10-01"));
    assert_eq!(merged.modalities.input, ["text"]);
    assert_eq!(serde_json::to_value(&merged.pricing).unwrap(), json!({ "text_tokens": { "standard": { "input_per_million": 1.0 } } }));
    assert_eq!(merged.metadata["source"], "models.dev");
    assert_eq!(merged.metadata["provider_note"], "kept");
    assert_eq!(caps(&merged), ["streaming", "vision"]);
}

// spec: "keeps the models.dev values it does have"
#[test]
fn keeps_the_models_dev_values_it_has() {
    let dev = model("test", "openai", json!({ "knowledge_cutoff": "2025-06-01", "metadata": { "source": "models.dev" } }));
    let provider = model("test", "openai", json!({ "knowledge_cutoff": "2024-10-01" }));
    assert_eq!(add_provider_metadata(&dev, &provider).knowledge_cutoff.as_deref(), Some("2025-06-01"));
}

// spec: "fills missing pricing rates without replacing models.dev rates"
#[test]
fn fills_missing_pricing_rates_only() {
    let dev = model("test", "perplexity", json!({ "pricing": { "text_tokens": { "standard": { "input_per_million": 1.0 } } } }));
    let provider = model("test", "perplexity", json!({ "pricing": { "text_tokens": { "standard": {
        "input_per_million": 2.0, "output_per_million": 3.0, "cache_write_input_per_million": 0.5
    } } } }));
    let merged = add_provider_metadata(&dev, &provider);
    assert_eq!(
        serde_json::to_value(&merged.pricing).unwrap()["text_tokens"]["standard"],
        json!({ "input_per_million": 1.0, "output_per_million": 3.0, "cache_write_input_per_million": 0.5 })
    );
}

// spec: "keeps a provider capability models.dev does not report on"
#[test]
fn keeps_a_capability_models_dev_does_not_report_on() {
    let dev = model("test", "openai", json!({
        "capabilities": ["function_calling"], "modalities": { "input": ["text"], "output": ["text"] },
        "metadata": { "source": "models.dev", "tool_call": true }
    }));
    let provider = model("test", "openai", json!({ "capabilities": ["function_calling", "structured_output"] }));
    assert_eq!(
        caps(&add_provider_metadata(&dev, &provider)),
        sorted(["function_calling", "tool_choice", "parallel_tool_calls", "structured_output"].map(String::from).to_vec())
    );
}

// spec: "drops a provider capability models.dev reports as absent"
#[test]
fn drops_a_capability_models_dev_reports_absent() {
    let dev = model("test", "openai", json!({
        "capabilities": [], "modalities": { "input": ["text"], "output": ["text"] },
        "metadata": { "source": "models.dev", "tool_call": false, "structured_output": false, "reasoning": false }
    }));
    let provider = model("test", "openai", json!({ "capabilities": ["streaming", "function_calling", "structured_output", "reasoning", "vision"] }));
    assert_eq!(add_provider_metadata(&dev, &provider).capabilities, ["streaming"]);
}

// spec: "normalizes embedding modalities on the merged entry"
#[test]
fn normalizes_embedding_modalities() {
    let merged = add_provider_metadata(&model("text-embedding-3-small", "openai", json!({})), &model("text-embedding-3-small", "openai", json!({})));
    assert_eq!(merged.modalities.input, ["text"]);
    assert_eq!(merged.modalities.output, ["embeddings"]);
}

// spec: "prefers a provider-reported non-chat operation over a models.dev chat classification"
#[test]
fn prefers_a_provider_reported_non_chat_operation() {
    let dev = model("mistral-embed", "mistral", json!({ "modalities": { "input": ["text"], "output": ["text"] } }));
    let provider = model("mistral-embed", "mistral", json!({ "modalities": { "input": ["text"], "output": ["embeddings"] } }));
    let merged = add_provider_metadata(&dev, &provider);
    assert_eq!(merged.model_type(), ModelType::Embedding);
    assert_eq!(merged.modalities.output, ["embeddings"]);
}

// ---- find_models_dev_model via merge_models --------------------------------------------------------

fn merged_one(provider_model: Model, dev: Vec<Model>) -> Model {
    let id = provider_model.id.clone();
    merge_models(&[provider_model], &dev).into_iter().find(|m| m.id == id).unwrap()
}

// spec: "reuses an exact OpenAI base entry for its release-dated snapshot"
#[test]
fn openai_dated_snapshots_reuse_their_base_entry() {
    let base = model("gpt-5.4-mini", "openai", json!({ "created_at": "2026-03-17 00:00:00 UTC", "context_window": 400000 }));
    assert_eq!(merged_one(model("gpt-5.4-mini-2026-03-17", "openai", json!({})), vec![base]).context_window, Some(400000));
    // spec: "maps OpenAI snapshots whose public release date differs from their base entry"
    let o1 = model("o1", "openai", json!({ "context_window": 200000 }));
    assert_eq!(merged_one(model("o1-2024-12-17", "openai", json!({})), vec![o1]).context_window, Some(200000));
    // spec: "does not inherit OpenAI metadata when the release date differs"
    let gpt4o = model("gpt-4o", "openai", json!({ "created_at": "2024-05-13 00:00:00 UTC", "context_window": 128000 }));
    assert_eq!(merged_one(model("gpt-4o-2024-08-06", "openai", json!({})), vec![gpt4o]).context_window, None);
}

// spec: "uses aliases reported by Mistral to find a models.dev entry"
#[test]
fn mistral_aliases_find_a_models_dev_entry() {
    let entry = model("mistral-embed", "mistral", json!({ "context_window": 8000 }));
    let provider = model("mistral-embed-2312", "mistral", json!({ "metadata": { "aliases": ["mistral-embed"] } }));
    assert_eq!(merged_one(provider, vec![entry]).context_window, Some(8000));
}

// spec: ".merge_models merges by provider and id, sorting the result"
#[test]
fn merge_models_merges_by_provider_and_id_sorted() {
    let providers = [model("shared", "openai", json!({ "name": "Provider Name" })), model("provider-only", "openai", json!({}))];
    let dev = [
        model("shared", "openai", json!({ "name": "", "metadata": { "source": "models.dev" } })),
        model("dev-only", "anthropic", json!({ "metadata": { "source": "models.dev" } })),
    ];
    let merged = merge_models(&providers, &dev);
    let pairs: Vec<(&str, &str)> = merged.iter().map(|m| (m.provider.as_str(), m.id.as_str())).collect();
    assert_eq!(pairs, [("anthropic", "dev-only"), ("openai", "provider-only"), ("openai", "shared")]);
    assert_eq!(merged.iter().find(|m| m.id == "shared").unwrap().name, "Provider Name");
}

// spec: ".merge_with_existing keeps models from providers that were not refreshed"
#[test]
fn merge_with_existing_keeps_providers_that_were_not_refreshed() {
    let existing = [model("kept", "anthropic", json!({})), model("replaced", "openai", json!({}))];
    let fetch = ProviderFetch { models: vec![model("fresh", "openai", json!({}))], fetched_providers: vec!["openai".into()], ..Default::default() };
    let merged = merge_with_existing(&existing, &fetch, &ModelsDevFetch { models: vec![], fetched: true });
    assert_eq!(sorted(merged.iter().map(|m| m.id.clone()).collect()), ["fresh", "kept"]);
}

// spec: "falls back to the existing models.dev entries when the fetch failed"
#[test]
fn merge_with_existing_keeps_cached_models_dev_entries() {
    let cached = model("cached", "openai", json!({ "metadata": { "source": "models.dev" } }));
    let merged = merge_with_existing(&[cached], &ProviderFetch::default(), &ModelsDevFetch { models: vec![], fetched: false });
    assert_eq!(merged.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(), ["cached"]);
}

// ---- provider list parsers -----------------------------------------------------------------------

// spec: protocols/chat_completions/models_spec.rb
#[test]
fn chat_completions_listing_keeps_only_reported_metadata() {
    let body = json!({ "data": [
        { "id": "gpt-3.5-turbo-0125", "created": 1741110400, "object": "model", "owned_by": "system" },
        { "id": "gpt-5-codex", "created": 1741110403, "object": "model", "owned_by": "system", "shutdown_date": "2026-07-23" }
    ] });
    let models = parse_openai_models(&body, "openai", false);
    let m = &models[0];
    assert_eq!(m.name, "gpt-3.5-turbo-0125");
    assert_eq!((m.family.as_deref(), m.context_window, m.max_output_tokens), (None, None, None));
    assert!(m.capabilities.is_empty());
    assert_eq!(serde_json::to_value(&m.pricing).unwrap(), json!({}));
    assert_eq!(Value::Object(m.metadata.clone()), json!({ "object": "model", "owned_by": "system" }));
    assert_eq!(m.created_at.as_deref(), Some("2025-03-04 17:46:40 UTC"));
    assert_eq!(Value::Object(models[1].metadata.clone()), json!({ "object": "model", "owned_by": "system", "shutdown_date": "2026-07-23" }));
}

// spec: protocols/anthropic/models_spec.rb
#[test]
fn anthropic_listing_reads_limits_and_reported_capabilities() {
    let minimal = parse_anthropic_models(&json!({ "data": [{ "id": "claude-sonnet-4-5", "display_name": "Claude Sonnet 4.5", "created_at": "2026-01-02T03:04:05Z" }] }), "anthropic");
    assert_eq!(minimal[0].name, "Claude Sonnet 4.5");
    assert_eq!(minimal[0].created_at.as_deref(), Some("2026-01-02 03:04:05 UTC"));
    assert!(minimal[0].capabilities.is_empty() && minimal[0].context_window.is_none());
    let full = parse_anthropic_models(&json!({ "data": [{
        "id": "claude-opus-5", "display_name": "Claude Opus 5", "created_at": "2026-07-24T00:00:00Z",
        "max_input_tokens": 1000000, "max_tokens": 128000,
        "capabilities": {
            "batch": { "supported": true }, "citations": { "supported": true }, "code_execution": { "supported": true },
            "image_input": { "supported": true }, "pdf_input": { "supported": true }, "structured_outputs": { "supported": true },
            "thinking": { "supported": true }, "effort": { "supported": false }
        }
    }] }), "anthropic");
    assert_eq!((full[0].context_window, full[0].max_output_tokens), (Some(1000000), Some(128000)));
    assert_eq!(caps(&full[0]), sorted(["citations", "batch", "vision", "structured_output", "reasoning"].map(String::from).to_vec()));
}

// spec: protocols/gemini/models_spec.rb
#[test]
fn gemini_listing_maps_reported_operations() {
    let m = &parse_gemini_models(&json!({ "models": [{
        "name": "models/gemini-2.0-flash-001", "displayName": "Gemini 2.0 Flash", "version": "001",
        "description": "Fast Gemini model", "supportedGenerationMethods": ["generateContent"]
    }] }), "gemini")[0];
    assert_eq!((m.id.as_str(), m.name.as_str()), ("gemini-2.0-flash-001", "Gemini 2.0 Flash"));
    assert!(m.capabilities.is_empty());
    assert_eq!(
        Value::Object(m.metadata.clone()),
        json!({ "version": "001", "description": "Fast Gemini model", "supported_generation_methods": ["generateContent"] })
    );
    let ops = &parse_gemini_models(&json!({ "models": [{ "name": "models/gemini-embedding-test",
        "supportedGenerationMethods": ["embedContent", "asyncBatchEmbedContent", "createCachedContent", "bidiGenerateContent"] }] }), "gemini")[0];
    assert_eq!(ops.model_type(), ModelType::Embedding);
    assert_eq!(caps(ops), sorted(["batch", "caching", "streaming", "realtime"].map(String::from).to_vec()));
}

// spec: providers/mistral/models_spec.rb
#[test]
fn mistral_listing_keeps_what_the_listing_reports() {
    let data = json!({
        "id": "mistral-test", "object": "model", "owned_by": "mistralai", "description": "A test model",
        "max_context_length": 262144, "aliases": ["mistral-test-latest"], "deprecation": "2026-08-31T12:00:00Z",
        "deprecation_replacement_model": "mistral-medium-3-5",
        "capabilities": { "completion_chat": true, "function_calling": true, "vision": false, "audio_speech": true }
    });
    let m = &parse_mistral_models(&json!({ "data": [data.clone()] }), "mistral")[0];
    assert_eq!((m.name.as_str(), m.created_at.as_deref(), m.context_window, m.max_output_tokens), ("mistral-test", None, Some(262144), None));
    assert_eq!(caps(m), ["function_calling", "speech_generation"]);
    assert_eq!(m.metadata["aliases"], json!(["mistral-test-latest"]));
    assert_eq!(m.metadata["deprecation_replacement_model"], "mistral-medium-3-5");

    let mut vision = data.clone();
    vision["capabilities"]["vision"] = true.into();
    let v = &parse_mistral_models(&json!({ "data": [vision] }), "mistral")[0];
    assert_eq!(v.modalities.input, ["text", "image"]);

    let mut embed = data.clone();
    embed["id"] = "codestral-embed".into();
    embed["description"] = "Official Codestral embedding model".into();
    embed["capabilities"] = json!({ "completion_chat": false });
    let e = &parse_mistral_models(&json!({ "data": [embed] }), "mistral")[0];
    assert_eq!(e.model_type(), ModelType::Embedding);

    let mut bare = data;
    for k in ["max_context_length", "capabilities", "deprecation"] {
        bare.as_object_mut().unwrap().remove(k);
    }
    bare["aliases"] = json!([]);
    let b = &parse_mistral_models(&json!({ "data": [bare] }), "mistral")[0];
    assert!(b.context_window.is_none() && b.capabilities.is_empty());
    assert!(!b.metadata.contains_key("deprecation") && !b.metadata.contains_key("aliases"));
}

// spec: providers/openrouter/models_spec.rb "maps the OpenRouter payload onto a Model"
#[test]
fn openrouter_listing_maps_pricing_and_parameters() {
    let m = &parse_openrouter_models(&json!({ "data": [{
        "id": "openai/gpt-test", "name": "GPT Test", "created": 1741110400, "context_length": 128000,
        "architecture": { "input_modalities": ["text", "image"], "output_modalities": ["text"] },
        "pricing": { "prompt": "0.000001", "completion": "0.000002" },
        "top_provider": { "max_completion_tokens": 4096 },
        "supported_parameters": ["tools", "response_format"]
    }] }), "openrouter")[0];
    assert_eq!(m.family.as_deref(), Some("openai"));
    assert_eq!(m.max_output_tokens, Some(4096));
    let standard = &serde_json::to_value(&m.pricing).unwrap()["text_tokens"]["standard"];
    assert!((standard["input_per_million"].as_f64().unwrap() - 1.0).abs() < 1e-9);
    assert!((standard["output_per_million"].as_f64().unwrap() - 2.0).abs() < 1e-9);
    assert_eq!(caps(m), ["function_calling", "streaming", "structured_output"]);
    assert!(parse_openrouter_models(&json!({}), "openrouter").is_empty());
}

// spec: providers/xai/models_spec.rb "does not guess modalities from model ids"
#[test]
fn xai_listing_keeps_only_reported_metadata() {
    let m = &parse_openai_models(&json!({ "data": [{ "id": "grok-4", "owned_by": "xai" }] }), "xai", true)[0];
    assert_eq!(Value::Object(m.metadata.clone()), json!({ "owned_by": "xai" }));
    assert!(m.modalities.input.is_empty());
}

// spec: models_local_refresh_spec.rb "can parse list models response" (Ollama)
// and providers/gpustack/models_spec.rb
#[test]
fn gpustack_maps_categories_onto_modalities() {
    let m = gpustack_model(
        &json!({ "id": "qwen3", "created": 1741110400, "meta": { "n_ctx": 32768, "support_tool_calls": true } }),
        &["llm".into()],
        "gpustack",
    );
    assert_eq!(m.context_window, Some(32768));
    assert_eq!(m.capabilities, ["streaming", "structured_output", "json_mode", "function_calling"]);
    assert_eq!((m.modalities.input.clone(), m.modalities.output.clone()), (vec!["text".to_string()], vec!["text".to_string()]));
    let e = gpustack_model(&json!({ "id": "bge" }), &["embedding".into()], "gpustack");
    assert_eq!(e.modalities.output, ["embeddings"]);
    assert!(e.capabilities.is_empty());
}

#[test]
fn capability_augmenters_match_ruby() {
    let text = Modalities { input: vec!["text".into()], output: vec!["text".into()] };
    assert_eq!(augment_capabilities("anthropic", vec!["function_calling".into()], "x", &text), ["function_calling", "tool_choice", "parallel_tool_calls"]);
    assert_eq!(augment_capabilities("deepseek", vec!["function_calling".into()], "x", &text), ["function_calling", "tool_choice"]);
    assert_eq!(augment_capabilities("xai", vec![], "grok-4.3", &text), ["streaming", "tool_choice", "parallel_tool_calls"]);
    assert_eq!(augment_capabilities("openai", vec![], "whisper-1", &text), ["transcription"]);
    assert!(augment_capabilities("ollama", vec![], "x", &text).is_empty());
}

// ---- cassette: providers/perplexity/models_spec.rb -------------------------------------------------

#[tokio::test]
async fn perplexity_lists_the_endpoint_catalog_with_static_models_added() {
    let cassette = Cassette::start("providers_perplexity_models_list_models_lists_the_models_endpoint_catalog_with_the_search_and_embedding_models_added")
        .await
        .expect("cassette");
    let mut config = Config::default();
    config.set("perplexity_api_base", cassette.server.uri()).set("perplexity_api_key", "test");
    config.max_retries = 0;
    let models = rust_llm::models::list_models(Provider::Perplexity, Arc::new(config)).await.unwrap();
    cassette.assert_all_matched().await;

    let ids: Vec<&str> = models.iter().map(|m| m.id.as_str()).collect();
    for id in ["sonar", "sonar-pro", "sonar-reasoning-pro", "sonar-deep-research", "fast", "wide-research", "pplx-embed-v1-0.6b", "pplx-embed-v1-4b", "perplexity/sonar"] {
        assert!(ids.contains(&id), "missing {id}");
    }
    assert!(!ids.contains(&"sonar-reasoning"));
    assert!(models.iter().all(|m| m.provider == "perplexity"));
    let sonar = models.iter().find(|m| m.id == "sonar").unwrap();
    assert_eq!((sonar.context_window, sonar.max_output_tokens), (Some(128000), None));
    assert_eq!(sonar.capabilities, ["streaming", "structured_output", "citations"]);
    assert_eq!(sonar.pricing.text_tokens().input(), Some(1.0));
    let embedding = models.iter().find(|m| m.id == "pplx-embed-v1-0.6b").unwrap();
    assert_eq!(embedding.model_type(), ModelType::Embedding);
    assert_eq!((embedding.context_window, embedding.pricing.text_tokens().input()), (Some(32768), Some(0.004)));
    let listed = models.iter().find(|m| m.id == "perplexity/sonar").unwrap();
    assert_eq!(listed.pricing.text_tokens().output(), Some(2.5));
}

// spec: "falls back to the static list when the endpoint fails"
#[tokio::test]
async fn perplexity_falls_back_to_the_static_list() {
    let server = MockServer::start().await;
    Mock::given(method("GET")).respond_with(ResponseTemplate::new(500).set_body_json(json!({ "error": { "message": "down" } }))).mount(&server).await;
    let mut config = Config::default();
    config.set("perplexity_api_base", server.uri()).set("perplexity_api_key", "test");
    config.max_retries = 0;
    let models = rust_llm::models::list_models(Provider::Perplexity, Arc::new(config)).await.unwrap();
    assert!(models.iter().any(|m| m.id == "sonar"));
    assert!(!models.iter().any(|m| m.id.contains('/')));
}

// ---- refresh (models_refresh_spec.rb, models_local_refresh_spec.rb) ------------------------------

struct Refresh {
    _servers: Vec<MockServer>,
    dir: std::path::PathBuf,
    config: Arc<Config>,
}

impl Drop for Refresh {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
        rust_llm::models::refresh::reset();
    }
}

/// A published catalog server plus an Ollama server listing `llama` with `context` tokens.
async fn refresh_setup(published: Value, etag: &str) -> Refresh {
    let catalog = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/models.json"))
        .respond_with(ResponseTemplate::new(200).insert_header("etag", etag).set_body_json(published))
        .mount(&catalog)
        .await;
    let ollama = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "data": [{ "id": "llama", "created": 1741110400, "owned_by": "library" }] })))
        .mount(&ollama)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/show"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "capabilities": ["completion", "tools"] })))
        .mount(&ollama)
        .await;
    // `described_class.new([])`: start from an empty registry, so only what the refresh finds remains.
    Models::install(Vec::new());
    let dir = std::env::temp_dir().join(format!("rust_llm_registry_{}_{}", std::process::id(), etag));
    let _ = std::fs::remove_dir_all(&dir);
    let mut config = Config::default();
    config.set("model_registry_url", format!("{}/models.json", catalog.uri()));
    config.set("ollama_api_base", format!("{}/v1", ollama.uri()));
    config.model_registry_file = Some(dir.join("models.json"));
    Refresh { _servers: vec![catalog, ollama], dir, config: Arc::new(config) }
}

#[tokio::test]
async fn refresh_merges_the_published_catalog_with_provider_listings_and_saves_it() {
    let _guard = REGISTRY_LOCK.lock().await;
    let published = json!([{ "id": "gpt-x", "name": "GPT X", "provider": "openai", "context_window": 400000 }]);
    let r = refresh_setup(published, "etag-1").await;
    let registry = rust_llm::models::refresh::refresh_with_config(r.config.clone(), false).await.unwrap();

    // spec: "returns models with consistent structure"
    let ids = sorted(registry.all().iter().map(|m| m.id.clone()).collect());
    assert_eq!(ids, ["gpt-x", "llama"]);
    let llama = registry.find("llama", Some("ollama")).unwrap();
    assert_eq!(llama.capabilities, ["streaming", "structured_output", "function_calling"]);
    assert_eq!(llama.family.as_deref(), Some("ollama"));
    // The global registry now answers for the refreshed models.
    assert!(rust_llm::models().find("gpt-x", None).is_ok());

    // spec: "saves models with correct JSON structure"
    let file = r.config.model_registry_file.clone().unwrap();
    let saved: Value = serde_json::from_str(&std::fs::read_to_string(&file).unwrap()).unwrap();
    for m in saved.as_array().unwrap() {
        assert!(m["capabilities"].is_array() && m["modalities"].is_object() && m["pricing"].is_object());
    }
    assert_eq!(std::fs::read_to_string(file.with_extension("json.etag")).unwrap().trim(), "etag-1");
    assert_eq!(registry::read(&file).unwrap().unwrap().len(), 2);
}

// spec: models_local_refresh_spec.rb "with remote_only: true excludes local providers"
#[tokio::test]
async fn remote_only_refresh_skips_local_providers() {
    let _guard = REGISTRY_LOCK.lock().await;
    let published = json!([{ "id": "gpt-x", "name": "GPT X", "provider": "openai" }]);
    let r = refresh_setup(published, "etag-2").await;
    let registry = rust_llm::models::refresh::refresh_with_config(r.config.clone(), true).await.unwrap();
    assert!(registry.find("llama", Some("ollama")).is_err());
}

// Refresh events: `models.refresh.rust_llm` with the model count.
#[tokio::test]
async fn refresh_emits_its_event() {
    let _guard = REGISTRY_LOCK.lock().await;
    let r = refresh_setup(json!([{ "id": "gpt-x", "name": "GPT X", "provider": "openai" }]), "etag-3").await;
    let events: Arc<Mutex<Vec<(String, Value)>>> = Default::default();
    let sink = events.clone();
    let mut config = (*r.config).clone();
    config.instrumenter = Some(Arc::new(move |name: &str, p: &serde_json::Map<String, Value>, _: Option<std::time::Duration>| {
        sink.lock().unwrap().push((name.into(), Value::Object(p.clone())));
    }));
    rust_llm::models::refresh::refresh_with_config(Arc::new(config), true).await.unwrap();
    let events = events.lock().unwrap();
    let (_, p) = events.iter().find(|(n, _)| n == "models.refresh.rust_llm").unwrap();
    assert_eq!(p["remote_only"], true);
    assert_eq!(p["model_count"], 1);
    assert_eq!(p["not_modified"], false);
}

// A catalog that cannot be fetched leaves the registry unchanged.
#[tokio::test]
async fn a_failed_catalog_fetch_leaves_the_registry_unchanged() {
    let _guard = REGISTRY_LOCK.lock().await;
    let before = rust_llm::models().all().len();
    let mut config = Config::default();
    config.set("model_registry_url", "http://127.0.0.1:9/models.json");
    config.model_registry_file = None;
    let err = rust_llm::models::refresh::refresh_with_config(Arc::new(config), true).await.unwrap_err();
    assert!(matches!(&err, Error::ModelRegistry(m) if m.starts_with("Could not refresh the model registry from")), "{err}");
    assert_eq!(rust_llm::models().all().len(), before);
}

// spec: models_spec.rb "includes provider-specific refresh guidance for unknown models"
#[test]
fn unknown_models_point_at_the_real_refresh_api() {
    let err = rust_llm::models().find("nonexistent-model-12345", Some("openai")).unwrap_err();
    let message = err.to_string();
    assert!(message.contains(r#"Unknown model: "nonexistent-model-12345" for provider: "openai""#));
    assert!(message.contains("rust_llm::models::refresh"));
}

// spec: models_refresh_spec.rb "saves models with correct JSON structure" via save_to_json
#[test]
fn save_to_json_round_trips_the_registry() {
    let dir = std::env::temp_dir().join(format!("rust_llm_save_{}", std::process::id()));
    let file = dir.join("models.json");
    let registry = Models::new(vec![model("a", "openai", json!({ "capabilities": ["streaming"] }))]);
    registry.save_to_json(Some(&file)).unwrap();
    let loaded = Models::load_from_json(Some(&file));
    assert_eq!(loaded.all().len(), 1);
    assert_eq!(loaded.all()[0].capabilities, ["streaming"]);
    // A missing file falls back to the bundled registry.
    assert!(Models::load_from_json(Some(&dir.join("missing.json"))).all().len() > 100);
    let _ = std::fs::remove_dir_all(dir);
}
