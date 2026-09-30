//! Port of `lib/ruby_llm/models/schema.rb`: the JSON Schema every model registry entry conforms
//! to. RubyLLM builds it with Schematist; this writes out the same Draft 2020-12 document.

use serde_json::{Map, Value, json};

use super::refresh::{MODELS_DEV_INPUT_MODALITIES, MODELS_DEV_OUTPUT_MODALITIES};

/// `Models::Schema::CAPABILITIES`.
pub const CAPABILITIES: &[&str] = &[
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

/// `Model::Pricing::CATEGORIES`.
const PRICING_CATEGORIES: &[&str] = &["text_tokens", "images", "audio_tokens", "embeddings"];
/// `Model::PricingCategory::TIERS`.
const PRICING_TIERS: &[&str] = &["standard", "batch", "long_context"];
/// `Model::PricingTier::ATTRIBUTES`.
const PRICING_ATTRIBUTES: &[&str] = &[
    "input_per_million",
    "output_per_million",
    "cache_read_input_per_million",
    "cache_write_input_per_million",
    "reasoning_output_per_million",
];

fn nullable(description: &str, schema: Value) -> Value {
    json!({ "description": description, "anyOf": [schema, { "type": "null" }] })
}

fn object(properties: Map<String, Value>) -> Value {
    json!({
        "type": "object",
        "properties": properties,
        "required": [],
        "additionalProperties": false
    })
}

/// `Models::Schema.json_schema`: the schema for one model entry. Wrap it in an array schema to
/// validate a whole registry.
pub fn json_schema() -> Value {
    let tier = object(
        PRICING_ATTRIBUTES
            .iter()
            .map(|a| (a.to_string(), json!({ "type": "number", "minimum": 0 })))
            .collect(),
    );
    let category = {
        let mut properties: Map<String, Value> = PRICING_TIERS
            .iter()
            .map(|t| (t.to_string(), tier.clone()))
            .collect();
        properties.insert(
            "long_context_threshold".into(),
            json!({
                "type": "integer",
                "description": "Prompt size above which long_context rates apply",
                "minimum": 0
            }),
        );
        object(properties)
    };
    let mut pricing = object(
        PRICING_CATEGORIES
            .iter()
            .map(|c| (c.to_string(), category.clone()))
            .collect(),
    );
    pricing["description"] = "Pricing information for the model".into();

    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "RubyLLM::Models::Schema",
        "description": "A model entry in the RubyLLM model registry",
        "type": "object",
        "properties": {
            "id": { "type": "string", "description": "Unique identifier for the model" },
            "name": { "type": "string", "description": "Display name of the model" },
            "provider": {
                "type": "string",
                "description": "Provider of the model (e.g., openai, anthropic, mistral)"
            },
            "family": nullable("Model family (e.g., gpt-4, claude-3)", json!({ "type": "string" })),
            "created_at": nullable("Creation date of the model", json!({ "type": "string" })),
            "context_window": nullable(
                "Maximum context window size",
                json!({ "type": "integer", "minimum": 0 })
            ),
            "max_output_tokens": nullable(
                "Maximum output tokens",
                json!({ "type": "integer", "minimum": 0 })
            ),
            "knowledge_cutoff": nullable(
                "Knowledge cutoff date",
                json!({ "type": "string", "format": "date" })
            ),
            "modalities": {
                "type": "object",
                "properties": {
                    "input": {
                        "type": "array",
                        "description": "Supported input modalities",
                        "items": { "type": "string", "enum": MODELS_DEV_INPUT_MODALITIES }
                    },
                    "output": {
                        "type": "array",
                        "description": "Supported output modalities",
                        "items": { "type": "string", "enum": MODELS_DEV_OUTPUT_MODALITIES }
                    }
                },
                "required": ["input", "output"],
                "additionalProperties": false
            },
            "capabilities": {
                "type": "array",
                "description": "Model capabilities",
                "items": { "type": "string", "enum": CAPABILITIES }
            },
            "pricing": pricing,
            "metadata": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": true,
                "description": "Additional metadata about the model"
            },
            "unlisted_at": nullable(
                "When the provider stopped listing the model",
                json!({ "type": "string" })
            )
        },
        "required": ["id", "name", "provider", "context_window", "max_output_tokens"],
        "additionalProperties": false
    })
}
