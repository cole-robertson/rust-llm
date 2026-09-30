//! Ports of RubyLLM 2.0's provider model-listing and capability-augmenter specs:
//! `provider_capability_augmenters_spec.rb`, `protocols/{anthropic,chat_completions,gemini}/models_spec.rb`,
//! `providers/{gpustack,mistral,openrouter,perplexity,xai}/models_spec.rb`,
//! `providers/{mistral,xai}/capabilities_spec.rb`, and the model-listing examples of
//! `providers/{hetzner,ollama,ollama_cloud}_spec.rb`.
//!
//! Stubbed connections become wiremock servers, so `list_models` sends its real requests.

mod support;

use std::collections::HashMap;
use std::sync::Arc;

use rust_llm::model::{Modalities, Model, ModelType, PricingTier};
use rust_llm::models::refresh::{
    augment_capabilities, parse_anthropic_models, parse_gemini_models, parse_gpustack_models,
    parse_mistral_models, parse_models_dev_catalog, parse_ollama_models, parse_openai_models,
    parse_openrouter_models, parse_perplexity_models, parse_xai_models,
    supported_parameters_to_capabilities,
};
use rust_llm::{Config, Provider};
use serde_json::{Value, json};
use support::Cassette;
use wiremock::matchers::{body_json, method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// `Models::Schema::CAPABILITIES` (`lib/ruby_llm/models/schema.rb`).
const SCHEMA_CAPABILITIES: &[&str] = &[
    "streaming",
    "function_calling",
    "tool_choice",
    "parallel_tool_calls",
    "structured_output",
    "predicted_outputs",
    "distillation",
    "fine_tuning",
    "batch",
    "realtime",
    "image_generation",
    "speech_generation",
    "transcription",
    "translation",
    "citations",
    "reasoning",
    "caching",
    "moderation",
    "json_mode",
    "vision",
    "video",
    "ocr",
    "judgment",
];

fn strs(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

fn sorted(mut v: Vec<String>) -> Vec<String> {
    v.sort();
    v
}

/// RSpec's `contain_exactly`.
fn assert_contain_exactly(actual: &[String], expected: &[&str]) {
    assert_eq!(sorted(actual.to_vec()), sorted(strs(expected)));
}

fn includes(actual: &[String], x: &str) -> bool {
    actual.iter().any(|c| c == x)
}

fn find<'a>(models: &'a [Model], id: &str) -> &'a Model {
    models
        .iter()
        .find(|m| m.id == id)
        .unwrap_or_else(|| panic!("no model {id}"))
}

fn ids(models: &[Model]) -> Vec<&str> {
    models.iter().map(|m| m.id.as_str()).collect()
}

fn modalities(input: &[&str], output: &[&str]) -> Modalities {
    Modalities {
        input: strs(input),
        output: strs(output),
    }
}

fn metadata(v: Value) -> serde_json::Map<String, Value> {
    v.as_object().cloned().unwrap()
}

// ---- provider_capability_augmenters_spec.rb ---------------------------------------------------

/// `augment(provider, capabilities, model_id: 'test', input: ['text'], output: ['text'])`.
fn augment(
    slug: &str,
    caps: &[&str],
    model_id: &str,
    input: &[&str],
    output: &[&str],
) -> Vec<String> {
    augment_capabilities(slug, strs(caps), model_id, &modalities(input, output))
}

fn augment_default(slug: &str, caps: &[&str], model_id: &str) -> Vec<String> {
    augment(slug, caps, model_id, &["text"], &["text"])
}

// spec: provider_capability_augmenters_spec.rb:10 adds tool controls only when the catalog reports function calling
#[test]
fn adds_tool_controls_only_when_the_catalog_reports_function_calling() {
    assert_contain_exactly(
        &augment_default("openai", &["function_calling"], "test"),
        &["function_calling", "tool_choice", "parallel_tool_calls"],
    );
    assert!(augment_default("openai", &[], "test").is_empty());
}

// spec: provider_capability_augmenters_spec.rb:17 keeps provider-specific tool controls narrow
#[test]
fn keeps_provider_specific_tool_controls_narrow() {
    assert_contain_exactly(
        &augment_default("deepseek", &["function_calling"], "test"),
        &["function_calling", "tool_choice"],
    );
    assert_contain_exactly(
        &augment_default("anthropic", &["function_calling"], "test"),
        &["function_calling", "tool_choice", "parallel_tool_calls"],
    );
}

// spec: provider_capability_augmenters_spec.rb:26 recognizes only explicit OpenAI transcription model ids
#[test]
fn recognizes_only_explicit_openai_transcription_model_ids() {
    assert!(includes(
        &augment_default("openai", &[], "gpt-transcribe"),
        "transcription"
    ));
    assert!(!includes(
        &augment("openai", &[], "future-transcribe", &["audio"], &["text"]),
        "transcription"
    ));
}

// spec: provider_capability_augmenters_spec.rb:35 recognizes only explicit OpenAI search model ids as citable
#[test]
fn recognizes_only_explicit_openai_search_model_ids_as_citable() {
    assert_contain_exactly(
        &augment_default("openai", &[], "gpt-5-search-api"),
        &["structured_output", "citations"],
    );
    assert!(!includes(
        &augment_default("openai", &[], "future-search"),
        "citations"
    ));
}

// spec: provider_capability_augmenters_spec.rb:44 restores documented capabilities for exact OpenAI Chat and Codex model ids
#[test]
fn restores_documented_capabilities_for_openai_chat_and_codex_ids() {
    assert_contain_exactly(
        &augment_default("openai", &[], "gpt-5-chat-latest"),
        &[
            "function_calling",
            "tool_choice",
            "parallel_tool_calls",
            "structured_output",
            "vision",
        ],
    );
    let codex = augment_default("openai", &[], "gpt-5.2-codex");
    for c in [
        "function_calling",
        "structured_output",
        "vision",
        "reasoning",
    ] {
        assert!(includes(&codex, c), "missing {c}");
    }
}

// spec: provider_capability_augmenters_spec.rb:53 restores documented capabilities for exact OpenAI research and moderation model ids
#[test]
fn restores_documented_capabilities_for_openai_research_and_moderation_ids() {
    assert_contain_exactly(
        &augment_default("openai", &[], "o3-deep-research"),
        &["vision", "reasoning"],
    );
    assert_contain_exactly(
        &augment_default("openai", &[], "omni-moderation-latest"),
        &["vision"],
    );
}

// spec: provider_capability_augmenters_spec.rb:62 adds streaming to xAI models with text output
#[test]
fn adds_streaming_to_xai_models_with_text_output() {
    assert!(includes(&augment_default("xai", &[], "test"), "streaming"));
    assert!(!includes(
        &augment("xai", &[], "test", &["text"], &["image"]),
        "streaming"
    ));
}

// spec: provider_capability_augmenters_spec.rb:69 uses Google modality facts without classifying embedding operations as transcription
#[test]
fn uses_google_modality_facts_without_classifying_embeddings_as_transcription() {
    assert!(includes(
        &augment("gemini", &[], "gemini-test", &["audio"], &["text"]),
        "transcription"
    ));
    assert!(!includes(
        &augment(
            "gemini",
            &[],
            "gemini-embedding-test",
            &["audio"],
            &["text"]
        ),
        "transcription"
    ));
}

// ---- providers/mistral/capabilities_spec.rb ---------------------------------------------------

// spec: providers/mistral/capabilities_spec.rb:7 .augment includes documented structured output for Small 4 and its current alias
#[test]
fn mistral_includes_structured_output_for_small_4_and_its_alias() {
    for model_id in ["mistral-small-2603", "mistral-small-latest"] {
        let caps = augment_default(
            "mistral",
            &["function_calling", "reasoning", "vision"],
            model_id,
        );
        for c in ["structured_output", "tool_choice", "parallel_tool_calls"] {
            assert!(includes(&caps, c), "{model_id} missing {c}");
        }
    }
}

// spec: providers/mistral/capabilities_spec.rb:15 .augment does not give document models chat structured output
#[test]
fn mistral_does_not_give_document_models_structured_output() {
    assert_eq!(
        augment_default("mistral", &["vision"], "mistral-ocr-latest"),
        ["vision"]
    );
}

// ---- providers/xai/capabilities_spec.rb -------------------------------------------------------

// spec: providers/xai/capabilities_spec.rb:6 includes documented tool controls for Grok 4.3
#[test]
fn xai_includes_tool_controls_for_grok_4_3() {
    let caps = augment(
        "xai",
        &["function_calling", "reasoning", "structured_output"],
        "grok-4.3",
        &["text", "image"],
        &["text"],
    );
    assert!(includes(&caps, "tool_choice") && includes(&caps, "parallel_tool_calls"));
}

// spec: providers/xai/capabilities_spec.rb:15 does not infer tool controls for audio models
#[test]
fn xai_does_not_infer_tool_controls_for_audio_models() {
    assert!(augment("xai", &[], "grok-tts", &[], &["audio"]).is_empty());
}

// ---- protocols/anthropic/models_spec.rb --------------------------------------------------------

// spec: protocols/anthropic/models_spec.rb:25 #parse_list_models_response returns minimal provider metadata for models covered by models.dev
#[test]
fn anthropic_returns_minimal_provider_metadata() {
    let body = json!({ "data": [{
        "id": "claude-sonnet-4-5", "display_name": "Claude Sonnet 4.5", "created_at": "2026-01-02T03:04:05Z"
    }] });
    let model = &parse_anthropic_models(&body, "anthropic")[0];
    assert_eq!(model.id, "claude-sonnet-4-5");
    assert_eq!(model.name, "Claude Sonnet 4.5");
    assert_eq!(model.provider, "anthropic");
    assert_eq!(model.created_at.as_deref(), Some("2026-01-02 03:04:05 UTC"));
    assert_eq!(model.family, None);
    assert_eq!(model.context_window, None);
    assert_eq!(model.max_output_tokens, None);
    assert!(model.capabilities.is_empty());
    assert_eq!(model.pricing, Default::default());
}

// spec: protocols/anthropic/models_spec.rb:39 #parse_list_models_response reads limits and capabilities the API reports
#[test]
fn anthropic_reads_limits_and_capabilities_the_api_reports() {
    let body = json!({ "data": [{
        "id": "claude-opus-5", "display_name": "Claude Opus 5", "created_at": "2026-07-24T00:00:00Z",
        "max_input_tokens": 1_000_000, "max_tokens": 128_000,
        "capabilities": {
            "batch": { "supported": true }, "citations": { "supported": true },
            "code_execution": { "supported": true }, "image_input": { "supported": true },
            "pdf_input": { "supported": true }, "structured_outputs": { "supported": true },
            "thinking": { "supported": true }, "effort": { "supported": false }
        }
    }] });
    let model = &parse_anthropic_models(&body, "anthropic")[0];
    assert_eq!(model.context_window, Some(1_000_000));
    assert_eq!(model.max_output_tokens, Some(128_000));
    assert_contain_exactly(
        &model.capabilities,
        &[
            "citations",
            "batch",
            "vision",
            "structured_output",
            "reasoning",
        ],
    );
    assert!(
        model
            .capabilities
            .iter()
            .all(|c| SCHEMA_CAPABILITIES.contains(&c.as_str()))
    );
}

// ---- protocols/chat_completions/models_spec.rb -------------------------------------------------

fn chat_completions_listing() -> Value {
    json!({ "data": [
        { "id": "gpt-3.5-turbo-0125", "created": 1_741_110_400, "object": "model", "owned_by": "system" },
        { "id": "omni-moderation-latest", "created": 1_741_110_401, "object": "model", "owned_by": "system" }
    ] })
}

// spec: protocols/chat_completions/models_spec.rb:42 .parse_list_models_response keeps only metadata the provider reports
#[test]
fn chat_completions_keeps_only_metadata_the_provider_reports() {
    let models = parse_openai_models(&chat_completions_listing(), "openai", false);
    let model = find(&models, "gpt-3.5-turbo-0125");
    assert_eq!(model.name, "gpt-3.5-turbo-0125");
    assert_eq!(model.family, None);
    assert_eq!(model.context_window, None);
    assert_eq!(model.max_output_tokens, None);
    assert!(model.capabilities.is_empty());
    assert_eq!(model.pricing, Default::default());
    assert_eq!(
        model.metadata,
        metadata(json!({ "object": "model", "owned_by": "system" }))
    );
}

// spec: protocols/chat_completions/models_spec.rb:54 .parse_list_models_response does not guess metadata from a model id
#[test]
fn chat_completions_does_not_guess_metadata_from_a_model_id() {
    let body = json!({ "data": [
        { "id": "gpt-5.4-nano-2026-03-17", "created": 1_741_110_402, "object": "model", "owned_by": "system" }
    ] });
    let model = &parse_openai_models(&body, "openai", false)[0];
    assert_eq!(model.context_window, None);
    assert_eq!(model.max_output_tokens, None);
    assert!(model.capabilities.is_empty());
    assert_eq!(model.pricing, Default::default());
}

// spec: protocols/chat_completions/models_spec.rb:74 .parse_list_models_response keeps the retirement date the provider reports
#[test]
fn chat_completions_keeps_the_retirement_date() {
    let body = json!({ "data": [{
        "id": "gpt-5-codex", "created": 1_741_110_403, "object": "model", "owned_by": "system",
        "shutdown_date": "2026-07-23"
    }] });
    let model = &parse_openai_models(&body, "openai", false)[0];
    assert_eq!(
        model.metadata,
        metadata(json!({ "object": "model", "owned_by": "system", "shutdown_date": "2026-07-23" }))
    );
}

// spec: protocols/chat_completions/models_spec.rb:92 .parse_list_models_response omits the retirement date for providers that do not report one
#[test]
fn chat_completions_omits_an_unreported_retirement_date() {
    let models = parse_openai_models(&chat_completions_listing(), "openai", false);
    assert!(
        !find(&models, "gpt-3.5-turbo-0125")
            .metadata
            .contains_key("shutdown_date")
    );
}

// ---- protocols/gemini/models_spec.rb -----------------------------------------------------------

// spec: protocols/gemini/models_spec.rb:27 #parse_list_models_response keeps only metadata the provider reports
#[test]
fn gemini_keeps_only_metadata_the_provider_reports() {
    let body = json!({ "models": [{
        "name": "models/gemini-2.0-flash-001", "displayName": "Gemini 2.0 Flash", "version": "001",
        "description": "Fast Gemini model", "supportedGenerationMethods": ["generateContent"]
    }] });
    let model = &parse_gemini_models(&body, "gemini")[0];
    assert_eq!(model.id, "gemini-2.0-flash-001");
    assert_eq!(model.name, "Gemini 2.0 Flash");
    assert_eq!(model.provider, "gemini");
    assert_eq!(model.family, None);
    assert_eq!(model.context_window, None);
    assert_eq!(model.max_output_tokens, None);
    assert!(model.capabilities.is_empty());
    assert_eq!(model.pricing, Default::default());
    assert_eq!(
        model.metadata,
        metadata(json!({
            "version": "001", "description": "Fast Gemini model",
            "supported_generation_methods": ["generateContent"]
        }))
    );
}

// spec: protocols/gemini/models_spec.rb:45 #parse_list_models_response maps the operations reported by the provider
#[test]
fn gemini_maps_the_operations_reported_by_the_provider() {
    let body = json!({ "models": [{
        "name": "models/gemini-embedding-test",
        "supportedGenerationMethods": ["embedContent", "asyncBatchEmbedContent", "createCachedContent", "bidiGenerateContent"]
    }] });
    let model = &parse_gemini_models(&body, "gemini")[0];
    assert_eq!(model.model_type(), ModelType::Embedding);
    assert_eq!(model.modalities, modalities(&["text"], &["embeddings"]));
    assert_contain_exactly(
        &model.capabilities,
        &["batch", "caching", "streaming", "realtime"],
    );
}

// ---- providers/gpustack/models_spec.rb ---------------------------------------------------------

const GPUSTACK_CATEGORIES: &[&str] = &[
    "llm",
    "embedding",
    "image",
    "reranker",
    "speech_to_text",
    "text_to_speech",
    "unknown",
];

/// `stub_categories`: each category query answers with its models; unlisted categories answer
/// with an empty list. Every request must ask `models?with_meta=true&categories=<category>`.
async fn gpustack_list(by_category: Value) -> (MockServer, Vec<Model>) {
    let server = MockServer::start().await;
    for category in GPUSTACK_CATEGORIES {
        let data = by_category.get(*category).cloned().unwrap_or(json!([]));
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .and(query_param("with_meta", "true"))
            .and(query_param("categories", *category))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(json!({ "object": "list", "data": data })),
            )
            .expect(1)
            .mount(&server)
            .await;
    }
    let mut config = Config::default();
    config.set("gpustack_api_base", format!("{}/v1", server.uri()));
    config.max_retries = 0;
    let models = rust_llm::models::list_models(Provider::GPUStack, Arc::new(config))
        .await
        .unwrap();
    server.verify().await;
    (server, models)
}

// spec: providers/gpustack/models_spec.rb:27 #models_url asks the catalog for model meta
#[tokio::test]
async fn gpustack_asks_the_catalog_for_model_meta() {
    let (server, _) = gpustack_list(json!({})).await;
    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), GPUSTACK_CATEGORIES.len());
    for (request, category) in requests.iter().zip(GPUSTACK_CATEGORIES) {
        assert_eq!(request.url.path(), "/v1/models");
        assert_eq!(
            request.url.query(),
            Some(format!("with_meta=true&categories={category}").as_str())
        );
    }
}

// spec: providers/gpustack/models_spec.rb:33 #list_models queries each category and maps the OpenAI list format onto Models
#[tokio::test]
async fn gpustack_queries_each_category_and_maps_the_list() {
    let (_server, models) = gpustack_list(json!({
        "llm": [{
            "id": "qwen3", "object": "model", "created": 1_735_770_000, "owned_by": "gpustack",
            "meta": { "n_ctx": 32_768, "support_tool_calls": true, "support_reasoning": true }
        }],
        "embedding": [{ "id": "bge-m3", "object": "model", "owned_by": "gpustack" }]
    }))
    .await;
    assert_eq!(sorted(strs(&ids(&models))), ["bge-m3", "qwen3"]);
    let qwen = find(&models, "qwen3");
    assert_eq!(qwen.provider, "gpustack");
    assert_eq!(qwen.family.as_deref(), Some("gpustack"));
    assert_eq!(qwen.created_at.as_deref(), Some("2025-01-01 22:20:00 UTC"));
    assert_eq!(qwen.context_window, Some(32_768));
    assert_contain_exactly(
        &qwen.capabilities,
        &[
            "streaming",
            "structured_output",
            "json_mode",
            "function_calling",
            "reasoning",
        ],
    );
    assert_eq!(qwen.modalities.input, ["text"]);
    assert_eq!(qwen.modalities.output, ["text"]);
    assert_eq!(qwen.metadata["owned_by"], "gpustack");
    assert_eq!(qwen.metadata["categories"], json!(["llm"]));
}

// spec: providers/gpustack/models_spec.rb:62 #list_models keeps owner-prefixed ids and reads vLLM-style meta
#[tokio::test]
async fn gpustack_keeps_owner_prefixed_ids_and_reads_vllm_meta() {
    let (_server, models) = gpustack_list(json!({
        "llm": [{
            "id": "alice/qwen3-vl", "object": "model", "owned_by": "alice",
            "meta": { "max_model_len": 131_072, "support_vision": true, "support_audio": true }
        }]
    }))
    .await;
    let model = &models[0];
    assert_eq!(model.id, "alice/qwen3-vl");
    assert_eq!(model.context_window, Some(131_072));
    assert!(model.supports("vision"));
    assert_eq!(model.modalities.input, ["text", "image", "audio"]);
}

// spec: providers/gpustack/models_spec.rb:80 #list_models merges models that appear under several categories
#[tokio::test]
async fn gpustack_merges_models_listed_under_several_categories() {
    let (_server, models) = gpustack_list(json!({
        "speech_to_text": [{ "id": "voxbox", "object": "model", "owned_by": "gpustack" }],
        "text_to_speech": [{ "id": "voxbox", "object": "model", "owned_by": "gpustack" }]
    }))
    .await;
    assert_eq!(models.len(), 1);
    assert_eq!(
        models[0].metadata["categories"],
        json!(["speech_to_text", "text_to_speech"])
    );
    assert_eq!(models[0].modalities.input, ["text", "audio"]);
    assert_eq!(models[0].modalities.output, ["text", "audio"]);
}

// spec: providers/gpustack/models_spec.rb:94 #list_models maps image and embedding categories onto their output modality
#[tokio::test]
async fn gpustack_maps_image_and_embedding_categories_onto_outputs() {
    let (_server, models) = gpustack_list(json!({
        "embedding": [{ "id": "bge-m3", "object": "model", "owned_by": "gpustack" }],
        "image": [{ "id": "flux.1", "object": "model", "owned_by": "gpustack" }]
    }))
    .await;
    let embedding = find(&models, "bge-m3");
    assert!(embedding.capabilities.is_empty());
    assert_eq!(embedding.modalities.output, ["embeddings"]);
    assert_eq!(find(&models, "flux.1").modalities.output, ["image"]);
}

// spec: providers/gpustack/models_spec.rb:109 #list_models returns an empty list when no category has models
#[tokio::test]
async fn gpustack_returns_an_empty_list_when_no_category_has_models() {
    let (_server, models) = gpustack_list(json!({})).await;
    assert!(models.is_empty());
}

// spec: providers/gpustack/models_spec.rb:117 #parse_list_models_response parses a plain OpenAI list without meta or categories
#[test]
fn gpustack_parses_a_plain_openai_list() {
    let body = json!({ "object": "list", "data": [{ "id": "qwen3", "object": "model", "owned_by": "gpustack" }] });
    let model = &parse_gpustack_models(&body, "gpustack")[0];
    assert_eq!(model.id, "qwen3");
    assert_eq!(model.created_at, None);
    assert_eq!(model.context_window, None);
    assert!(model.capabilities.is_empty());
    assert_eq!(model.pricing, Default::default());
}

// spec: providers/gpustack/models_spec.rb:130 #parse_list_models_response returns an empty list when the payload has no data
#[test]
fn gpustack_parse_returns_an_empty_list_without_data() {
    assert!(parse_gpustack_models(&json!({}), "gpustack").is_empty());
}

// ---- providers/mistral/models_spec.rb ----------------------------------------------------------

fn mistral_data() -> Value {
    json!({
        "id": "mistral-test", "object": "model", "owned_by": "mistralai", "description": "A test model",
        "max_context_length": 262_144, "aliases": ["mistral-test-latest"],
        "deprecation": "2026-08-31T12:00:00Z", "deprecation_replacement_model": "mistral-medium-3-5",
        "capabilities": { "completion_chat": true, "function_calling": true, "vision": false, "audio_speech": true }
    })
}

fn mistral_model(data: Value) -> Model {
    parse_mistral_models(&json!({ "data": [data] }), "mistral").remove(0)
}

// spec: providers/mistral/models_spec.rb:29 .parse_list_models_response keeps the model identity from the provider
#[test]
fn mistral_keeps_the_model_identity() {
    let model = mistral_model(mistral_data());
    assert_eq!(model.id, "mistral-test");
    assert_eq!(model.name, "mistral-test");
    assert_eq!(model.created_at, None);
}

// spec: providers/mistral/models_spec.rb:35 .parse_list_models_response takes the context window the listing reports
#[test]
fn mistral_takes_the_reported_context_window() {
    let model = mistral_model(mistral_data());
    assert_eq!(model.context_window, Some(262_144));
    assert_eq!(model.max_output_tokens, None);
}

// spec: providers/mistral/models_spec.rb:40 .parse_list_models_response adds only capabilities the listing reports
#[test]
fn mistral_adds_only_reported_capabilities() {
    assert_contain_exactly(
        &mistral_model(mistral_data()).capabilities,
        &["function_calling", "speech_generation"],
    );
}

// spec: providers/mistral/models_spec.rb:44 .parse_list_models_response gives image input to the models the listing marks as vision
#[test]
fn mistral_gives_image_input_to_vision_models() {
    let mut data = mistral_data();
    data["capabilities"]["vision"] = json!(true);
    let model = mistral_model(data);
    assert_eq!(model.modalities.input, ["text", "image"]);
    assert!(model.supports("vision"));
}

// spec: providers/mistral/models_spec.rb:51 .parse_list_models_response recognizes an embedding operation from the provider description
#[test]
fn mistral_recognizes_embeddings_from_the_description() {
    let mut data = mistral_data();
    data["id"] = json!("codestral-embed");
    data["description"] = json!("Official Codestral embedding model");
    data["capabilities"] = json!({ "completion_chat": false });
    let model = mistral_model(data);
    assert_eq!(model.modalities, modalities(&["text"], &["embeddings"]));
    assert_eq!(model.model_type(), ModelType::Embedding);
}

// spec: providers/mistral/models_spec.rb:60 .parse_list_models_response keeps the deprecation, description and aliases in metadata
#[test]
fn mistral_keeps_deprecation_description_and_aliases() {
    let model = mistral_model(mistral_data());
    for (k, v) in [
        ("description", json!("A test model")),
        ("aliases", json!(["mistral-test-latest"])),
        ("deprecation", json!("2026-08-31T12:00:00Z")),
        ("deprecation_replacement_model", json!("mistral-medium-3-5")),
    ] {
        assert_eq!(model.metadata.get(k), Some(&v), "{k}");
    }
}

// spec: providers/mistral/models_spec.rb:69 .parse_list_models_response does not invent metadata the listing omits
#[test]
fn mistral_does_not_invent_omitted_metadata() {
    let mut data = mistral_data();
    let obj = data.as_object_mut().unwrap();
    obj.remove("max_context_length");
    obj.remove("capabilities");
    obj.remove("deprecation");
    obj.insert("aliases".into(), json!([]));
    let model = mistral_model(data);
    assert_eq!(model.context_window, None);
    assert!(model.capabilities.is_empty());
    assert_eq!(model.pricing, Default::default());
    assert!(!model.metadata.contains_key("deprecation"));
    assert!(!model.metadata.contains_key("aliases"));
}

// ---- providers/openrouter/models_spec.rb -------------------------------------------------------

fn openrouter(model: Value) -> Model {
    parse_openrouter_models(&json!({ "data": [model] }), "openrouter").remove(0)
}

// spec: providers/openrouter/models_spec.rb:20 #models_url points at the OpenRouter catalog endpoint
#[tokio::test]
async fn openrouter_points_at_the_catalog_endpoint() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "data": [] })))
        .mount(&server)
        .await;
    let mut config = Config::default();
    config
        .set("openrouter_api_base", format!("{}/api/v1", server.uri()))
        .set("openrouter_api_key", "test");
    config.max_retries = 0;
    rust_llm::models::list_models(Provider::OpenRouter, Arc::new(config))
        .await
        .unwrap();
    let first = &server.received_requests().await.unwrap()[0];
    assert_eq!(first.url.path(), "/api/v1/models");
    assert_eq!(first.url.query(), None);
}

// spec: providers/openrouter/models_spec.rb:26 #parse_list_models_response returns an empty list when the payload has no data
#[test]
fn openrouter_returns_an_empty_list_without_data() {
    assert!(parse_openrouter_models(&json!({}), "openrouter").is_empty());
}

// spec: providers/openrouter/models_spec.rb:30 #parse_list_models_response maps the OpenRouter payload onto a Model
#[test]
fn openrouter_maps_the_payload_onto_a_model() {
    let model = openrouter(json!({
        "id": "anthropic/claude-haiku-4-5", "name": "Anthropic: Claude Haiku 4.5",
        "description": "Fast Claude", "created": 1_700_000_000, "context_length": 200_000,
        "architecture": { "input_modalities": ["text", "image"], "output_modalities": ["text"] },
        "top_provider": { "max_completion_tokens": 8192 },
        "per_request_limits": { "prompt_tokens": "100" },
        "supported_parameters": ["tools", "response_format"],
        "pricing": { "prompt": "0.000001", "completion": "0.000005", "input_cache_read": "0.0000001" }
    }));
    assert_eq!(model.id, "anthropic/claude-haiku-4-5");
    assert_eq!(model.name, "Anthropic: Claude Haiku 4.5");
    assert_eq!(model.provider, "openrouter");
    assert_eq!(model.family.as_deref(), Some("anthropic"));
    assert_eq!(model.created_at.as_deref(), Some("2023-11-14 22:13:20 UTC"));
    assert_eq!(model.context_window, Some(200_000));
    assert_eq!(model.max_output_tokens, Some(8192));
    assert_eq!(model.modalities.input, ["text", "image"]);
    assert_eq!(model.modalities.output, ["text"]);
    assert_eq!(model.metadata["description"], "Fast Claude");
    assert_eq!(
        model.metadata["per_request_limits"],
        json!({ "prompt_tokens": "100" })
    );
    assert_eq!(
        model.metadata["supported_parameters"],
        json!(["tools", "response_format"])
    );
}

// spec: providers/openrouter/models_spec.rb:62 #parse_list_models_response converts per-token prices to per-million and drops the zeroed ones
#[test]
fn openrouter_converts_prices_to_per_million_and_drops_zeros() {
    let model = openrouter(json!({
        "id": "vendor/model",
        "pricing": { "prompt": "0.000001", "completion": "0.000005", "input_cache_read": "0", "internal_reasoning": "0.00001" }
    }));
    assert_eq!(
        serde_json::to_value(&model.pricing).unwrap(),
        json!({ "text_tokens": { "standard": {
            "input_per_million": 1.0, "output_per_million": 5.0, "reasoning_output_per_million": 10.0
        } } })
    );
}

// spec: providers/openrouter/models_spec.rb:86 #parse_list_models_response keeps the cache write price so cached prompts are costed
#[test]
fn openrouter_keeps_the_cache_write_price() {
    let model = openrouter(json!({
        "id": "anthropic/claude-opus-5:batch",
        "pricing": {
            "prompt": "0.0000025", "completion": "0.0000125",
            "input_cache_read": "0.00000025", "input_cache_write": "0.000003125"
        }
    }));
    let text = model.pricing.text_tokens();
    assert_eq!(text.cache_write_input(), Some(3.125));
    assert_eq!(text.cache_read_input(), Some(0.25));
}

// spec: providers/openrouter/models_spec.rb:103 #parse_list_models_response keeps the lifecycle dates OpenRouter reports
#[test]
fn openrouter_keeps_the_lifecycle_dates() {
    let model = openrouter(json!({
        "id": "z-ai/glm-4.5", "knowledge_cutoff": "2024-12-31", "expiration_date": "2026-12-31"
    }));
    assert_eq!(model.knowledge_cutoff.as_deref(), Some("2024-12-31"));
    assert_eq!(model.metadata["expiration_date"], "2026-12-31");
}

// spec: providers/openrouter/models_spec.rb:116 #parse_list_models_response leaves created_at nil when OpenRouter omits the timestamp
#[test]
fn openrouter_leaves_created_at_nil_without_a_timestamp() {
    let model = openrouter(json!({ "id": "vendor/model" }));
    assert_eq!(model.created_at, None);
    assert!(model.capabilities.is_empty());
    assert_eq!(model.knowledge_cutoff, None);
}

// spec: providers/openrouter/models_spec.rb:124 #parse_list_models_response preserves the rerank output modality
#[test]
fn openrouter_preserves_the_rerank_output_modality() {
    let model = openrouter(json!({
        "id": "voyageai/rerank-2.5-lite",
        "architecture": { "input_modalities": ["text"], "output_modalities": ["rerank"] }
    }));
    assert_eq!(model.modalities, modalities(&["text"], &["rerank"]));
    assert_eq!(model.model_type(), ModelType::Rerank);
}

fn params(v: Value) -> Vec<String> {
    supported_parameters_to_capabilities(Some(&v))
}

// spec: providers/openrouter/models_spec.rb:138 #supported_parameters_to_capabilities returns nothing when the model lists no parameters
#[test]
fn openrouter_params_nothing_without_parameters() {
    assert!(supported_parameters_to_capabilities(None).is_empty());
}

// spec: providers/openrouter/models_spec.rb:142 #supported_parameters_to_capabilities always claims streaming
#[test]
fn openrouter_params_always_claims_streaming() {
    assert_eq!(params(json!([])), ["streaming"]);
}

// spec: providers/openrouter/models_spec.rb:146 #supported_parameters_to_capabilities maps tools and tool_choice to function calling
#[test]
fn openrouter_params_maps_tools_to_function_calling() {
    assert!(includes(&params(json!(["tools"])), "function_calling"));
    assert!(includes(
        &params(json!(["tool_choice"])),
        "function_calling"
    ));
}

// spec: providers/openrouter/models_spec.rb:151 #supported_parameters_to_capabilities preserves explicit tool selection and parallel call controls
#[test]
fn openrouter_params_preserves_tool_selection_and_parallel_controls() {
    let caps = params(json!(["tools", "tool_choice", "parallel_tool_calls"]));
    for c in ["function_calling", "tool_choice", "parallel_tool_calls"] {
        assert!(includes(&caps, c), "missing {c}");
    }
}

// spec: providers/openrouter/models_spec.rb:157 #supported_parameters_to_capabilities does not infer parallel controls from tool selection
#[test]
fn openrouter_params_does_not_infer_parallel_controls() {
    let caps = params(json!(["tools", "tool_choice"]));
    assert!(includes(&caps, "tool_choice"));
    assert!(!includes(&caps, "parallel_tool_calls"));
}

// spec: providers/openrouter/models_spec.rb:164 #supported_parameters_to_capabilities recognizes explicit JSON schema support without a response format flag
#[test]
fn openrouter_params_recognizes_structured_outputs() {
    assert!(includes(
        &params(json!(["structured_outputs"])),
        "structured_output"
    ));
}

// spec: providers/openrouter/models_spec.rb:168 #supported_parameters_to_capabilities accepts the parameter definitions returned by the image catalog
#[test]
fn openrouter_params_accepts_image_catalog_definitions() {
    let definitions = json!({
        "aspect_ratio": { "type": "enum", "values": ["1:1", "auto"] },
        "n": { "type": "range", "min": 1, "max": 1 }
    });
    assert_eq!(params(definitions), ["streaming"]);
}

// spec: providers/openrouter/models_spec.rb:177 #supported_parameters_to_capabilities maps the remaining parameters onto capabilities
#[test]
fn openrouter_params_maps_the_remaining_parameters() {
    assert_contain_exactly(
        &params(json!(["response_format", "batch", "logit_bias", "top_k"])),
        &[
            "streaming",
            "structured_output",
            "batch",
            "predicted_outputs",
        ],
    );
}

// spec: providers/openrouter/models_spec.rb:187 #supported_parameters_to_capabilities requires both logit_bias and top_k for predicted outputs
#[test]
fn openrouter_params_requires_logit_bias_and_top_k() {
    assert!(!includes(
        &params(json!(["logit_bias"])),
        "predicted_outputs"
    ));
}

// ---- providers/perplexity/models_spec.rb -------------------------------------------------------

// spec: providers/perplexity/models_spec.rb:12 #list_models lists the models endpoint catalog with the search and embedding models added
#[tokio::test]
async fn perplexity_lists_the_catalog_with_search_and_embedding_models() {
    let cassette = Cassette::start("providers_perplexity_models_list_models_lists_the_models_endpoint_catalog_with_the_search_and_embedding_models_added")
        .await
        .expect("cassette");
    let mut config = Config::default();
    config
        .set("perplexity_api_base", cassette.server.uri())
        .set("perplexity_api_key", "test");
    config.max_retries = 0;
    let models = rust_llm::models::list_models(Provider::Perplexity, Arc::new(config))
        .await
        .unwrap();
    cassette.assert_all_matched().await;

    let ids = ids(&models);
    for id in [
        "sonar",
        "sonar-pro",
        "sonar-reasoning-pro",
        "sonar-deep-research",
        "fast",
        "wide-research",
        "pplx-embed-v1-0.6b",
        "pplx-embed-v1-4b",
        "perplexity/sonar",
    ] {
        assert!(ids.contains(&id), "missing {id}");
    }
    assert!(!ids.contains(&"sonar-reasoning"));
    assert!(models.iter().all(|m| m.provider == "perplexity"));

    let sonar = find(&models, "sonar");
    assert_eq!(sonar.context_window, Some(128_000));
    assert_eq!(sonar.max_output_tokens, None);
    assert_eq!(
        sonar.capabilities,
        ["streaming", "structured_output", "citations"]
    );
    assert_eq!(sonar.pricing.text_tokens().input(), Some(1.0));

    let embedding = find(&models, "pplx-embed-v1-0.6b");
    assert_eq!(embedding.model_type(), ModelType::Embedding);
    assert_eq!(embedding.context_window, Some(32_768));
    assert_eq!(embedding.pricing.text_tokens().input(), Some(0.004));

    let listed = find(&models, "perplexity/sonar");
    assert_eq!(listed.pricing.text_tokens().output(), Some(2.5));
}

// spec: providers/perplexity/models_spec.rb:36 #list_models falls back to the static list when the endpoint fails
#[tokio::test]
async fn perplexity_falls_back_to_the_static_list_when_the_endpoint_fails() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(
            ResponseTemplate::new(500).set_body_json(json!({ "error": { "message": "down" } })),
        )
        .mount(&server)
        .await;
    let mut config = Config::default();
    config
        .set("perplexity_api_base", server.uri())
        .set("perplexity_api_key", "test");
    config.max_retries = 0;
    let models = rust_llm::models::list_models(Provider::Perplexity, Arc::new(config))
        .await
        .unwrap();
    assert_eq!(
        ids(&models),
        [
            "sonar",
            "sonar-pro",
            "sonar-reasoning-pro",
            "sonar-deep-research",
            "fast",
            "low",
            "medium",
            "high",
            "xhigh",
            "wide-research",
            "pplx-embed-v1-0.6b",
            "pplx-embed-v1-4b"
        ]
    );
}

// spec: providers/perplexity/models_spec.rb:64 #parse_list_models_response keeps the endpoint pricing without inventing token limits
#[test]
fn perplexity_keeps_endpoint_pricing_without_inventing_limits() {
    let body = json!({ "data": [{
        "id": "anthropic/claude-opus-5",
        "pricing": { "input": 5, "output": 25, "cache_read": 0.5, "cache_write": 6.25 }
    }] });
    let models = parse_perplexity_models(&body, "perplexity");
    let opus = find(&models, "anthropic/claude-opus-5");
    assert_eq!(opus.context_window, None);
    assert_eq!(opus.max_output_tokens, None);
    assert_eq!(
        opus.pricing.text_tokens().standard,
        Some(PricingTier {
            input_per_million: Some(5.0),
            output_per_million: Some(25.0),
            cache_read_input_per_million: Some(0.5),
            cache_write_input_per_million: Some(6.25),
            reasoning_output_per_million: None,
        })
    );
}

// ---- providers/xai/models_spec.rb --------------------------------------------------------------

// spec: providers/xai/models_spec.rb:7 .parse_list_models_response keeps only metadata the model list reports
#[test]
fn xai_keeps_only_metadata_the_model_list_reports() {
    let body = json!({ "data": [
        { "id": "grok-4.3", "object": "model", "created": 1_777_068_000, "owned_by": "xai" },
        { "id": "grok-4.20-0309-non-reasoning", "object": "model", "created": 1_777_068_000, "owned_by": "xai" }
    ] });
    let models = parse_xai_models(&body, "xai");
    let reasoning = find(&models, "grok-4.3");
    assert_eq!(reasoning.name, "grok-4.3");
    assert!(reasoning.capabilities.is_empty());
    assert!(reasoning.modalities.input.is_empty());
    assert!(reasoning.modalities.output.is_empty());
    assert!(
        find(&models, "grok-4.20-0309-non-reasoning")
            .capabilities
            .is_empty()
    );
    assert_eq!(find(&models, "grok-tts").family.as_deref(), Some("grok"));
}

// spec: providers/xai/models_spec.rb:42 .parse_list_models_response does not guess modalities from model ids
#[test]
fn xai_does_not_guess_modalities_from_model_ids() {
    let body = json!({ "data": [
        { "id": "grok-imagine-image", "object": "model", "owned_by": "xai" },
        { "id": "grok-imagine-video", "object": "model", "owned_by": "xai" }
    ] });
    let models = parse_xai_models(&body, "xai");
    for id in ["grok-imagine-image", "grok-imagine-video"] {
        let m = find(&models, id);
        assert_eq!(m.modalities, modalities(&[], &[]));
        assert!(m.capabilities.is_empty());
    }
}

// ---- providers/hetzner_spec.rb -----------------------------------------------------------------

// spec: providers/hetzner_spec.rb:47 reads the models.dev hetzner catalog
#[test]
fn hetzner_reads_the_models_dev_hetzner_catalog() {
    // `MODELS_DEV_PROVIDER_MAP` includes 'hetzner' => 'hetzner': a models.dev `hetzner` entry
    // becomes a hetzner model.
    let models = parse_models_dev_catalog(&json!({
        "hetzner": { "models": { "qwen": { "id": "Qwen3.8-27B", "name": "Qwen 3.8" } } }
    }))
    .unwrap();
    assert_eq!(models.len(), 1);
    assert_eq!(
        (models[0].provider.as_str(), models[0].id.as_str()),
        ("hetzner", "Qwen3.8-27B")
    );
}

// spec: providers/hetzner_spec.rb:133 model listing reads the OpenAI-compatible model list
#[test]
fn hetzner_reads_the_openai_compatible_model_list() {
    let body = json!({ "object": "list", "data": [
        { "id": "Qwen/Qwen3.6-35B-A3B-FP8", "object": "model", "created": 1_776_384_000, "owned_by": "hetzner" },
        { "id": "Qwen3.8-27B", "object": "model", "created": 1_786_665_600, "owned_by": "hetzner" }
    ] });
    let models = parse_openai_models(&body, "hetzner", false);
    assert_eq!(ids(&models), ["Qwen/Qwen3.6-35B-A3B-FP8", "Qwen3.8-27B"]);
    assert!(models.iter().all(|m| m.provider == "hetzner"));
}

// ---- providers/ollama_spec.rb, providers/ollama_cloud_spec.rb ----------------------------------

fn ollama_listing(ids: &[&str]) -> Value {
    json!({ "data": ids.iter().map(|id| json!({ "id": id })).collect::<Vec<_>>() })
}

fn details(entries: &[(&str, &[&str])]) -> HashMap<String, Vec<String>> {
    entries
        .iter()
        .map(|(id, caps)| (id.to_string(), strs(caps)))
        .collect()
}

// spec: providers/ollama_spec.rb:31 model listing derives capabilities and modalities from what /api/show reports
#[test]
fn ollama_derives_capabilities_and_modalities_from_api_show() {
    let models = parse_ollama_models(
        &ollama_listing(&["llava:7b", "qwen3:8b"]),
        "ollama",
        &details(&[
            ("llava:7b", &["completion", "vision"]),
            ("qwen3:8b", &["completion", "tools", "thinking"]),
        ]),
        false,
    );
    let (vision, text) = (&models[0], &models[1]);
    assert_eq!(
        vision.capabilities,
        ["streaming", "structured_output", "vision"]
    );
    assert_eq!(vision.modalities.input, ["text", "image"]);
    assert_eq!(
        text.capabilities,
        [
            "streaming",
            "structured_output",
            "function_calling",
            "reasoning"
        ]
    );
    assert_eq!(text.modalities.input, ["text"]);
    assert!(!text.supports("vision"));
}

// spec: providers/ollama_spec.rb:45 model listing claims nothing beyond the server defaults when /api/show says nothing
#[test]
fn ollama_claims_only_server_defaults_without_api_show() {
    let model = &parse_ollama_models(
        &ollama_listing(&["mystery:latest"]),
        "ollama",
        &HashMap::new(),
        false,
    )[0];
    assert_eq!(model.capabilities, ["streaming", "structured_output"]);
    assert_eq!(model.modalities, modalities(&["text"], &["text"]));
}

// spec: providers/ollama_spec.rb:52 model listing reads embedding models as embedding models
#[test]
fn ollama_reads_embedding_models_as_embedding_models() {
    let models = parse_ollama_models(
        &ollama_listing(&["nomic-embed-text"]),
        "ollama",
        &details(&[("nomic-embed-text", &["embedding"])]),
        false,
    );
    assert!(models[0].capabilities.is_empty());
    assert_eq!(models[0].modalities, modalities(&["text"], &["embeddings"]));
}

/// An Ollama server listing `llava:7b` at `/v1/models`, with `show` answering `/api/show`.
async fn ollama_server(show: ResponseTemplate) -> (MockServer, Vec<Model>) {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(ollama_listing(&["llava:7b"])))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/show"))
        .and(body_json(json!({ "model": "llava:7b" })))
        .respond_with(show)
        .expect(1)
        .mount(&server)
        .await;
    let mut config = Config::default();
    config.set("ollama_api_base", format!("{}/v1", server.uri()));
    config.max_retries = 0;
    let models = rust_llm::models::list_models(Provider::Ollama, Arc::new(config))
        .await
        .unwrap();
    server.verify().await;
    (server, models)
}

// spec: providers/ollama_spec.rb:61 model listing asks /api/show about every listed model
#[tokio::test]
async fn ollama_asks_api_show_about_every_listed_model() {
    let (_server, models) = ollama_server(
        ResponseTemplate::new(200)
            .set_body_json(json!({ "capabilities": ["completion", "vision"] })),
    )
    .await;
    assert!(models[0].supports("vision"));
}

// spec: providers/ollama_spec.rb:73 model listing survives a server that does not answer /api/show
#[tokio::test]
async fn ollama_survives_a_server_that_does_not_answer_api_show() {
    let (_server, models) =
        ollama_server(ResponseTemplate::new(400).set_body_json(json!({ "error": "bad request" })))
            .await;
    assert!(!models[0].supports("vision"));
}

fn ollama_cloud_listing() -> Value {
    json!({ "data": [{ "id": "gpt-oss:120b", "created": 1_754_352_000, "owned_by": "ollama" }] })
}

// spec: providers/ollama_cloud_spec.rb:61 model listing reads the cloud catalog without the local -cloud suffix
#[test]
fn ollama_cloud_reads_the_catalog_without_the_cloud_suffix() {
    let models = parse_ollama_models(
        &ollama_cloud_listing(),
        "ollama_cloud",
        &HashMap::new(),
        true,
    );
    assert_eq!(ids(&models), ["gpt-oss:120b"]);
    assert_eq!(models[0].provider, "ollama_cloud");
}

// spec: providers/ollama_cloud_spec.rb:68 model listing reports no structured output, which Ollama Cloud does not support
#[test]
fn ollama_cloud_reports_no_structured_output() {
    let models = parse_ollama_models(
        &ollama_cloud_listing(),
        "ollama_cloud",
        &details(&[("gpt-oss:120b", &["completion", "tools", "thinking"])]),
        true,
    );
    assert_eq!(
        models[0].capabilities,
        ["streaming", "function_calling", "reasoning"]
    );
    assert!(!models[0].supports("structured_output"));
}

// spec: providers/ollama_cloud_spec.rb:77 model listing leaves vision to the models that /api/show says have it
#[test]
fn ollama_cloud_leaves_vision_to_models_api_show_reports() {
    let models = parse_ollama_models(
        &ollama_cloud_listing(),
        "ollama_cloud",
        &details(&[("gpt-oss:120b", &["completion", "tools", "thinking"])]),
        true,
    );
    assert!(!models[0].supports("vision"));
    assert_eq!(models[0].modalities.input, ["text"]);
}
