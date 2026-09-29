//! Port of `lib/ruby_llm/model.rb` and `lib/ruby_llm/model/pricing*.rb`.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PricingTier {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_per_million: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_per_million: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read_input_per_million: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write_input_per_million: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_output_per_million: Option<f64>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PricingCategory {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub standard: Option<PricingTier>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch: Option<PricingTier>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub long_context: Option<PricingTier>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub long_context_threshold: Option<i64>,
}

impl PricingCategory {
    pub fn input(&self) -> Option<f64> {
        self.standard.as_ref()?.input_per_million
    }
    pub fn output(&self) -> Option<f64> {
        self.standard.as_ref()?.output_per_million
    }
    pub fn cache_read_input(&self) -> Option<f64> {
        self.standard.as_ref()?.cache_read_input_per_million
    }
    pub fn cache_write_input(&self) -> Option<f64> {
        self.standard.as_ref()?.cache_write_input_per_million
    }
    pub fn reasoning_output(&self) -> Option<f64> {
        self.standard.as_ref()?.reasoning_output_per_million
    }

    /// `PricingCategory#tier_for`: the long-context tier once the prompt passes its threshold.
    pub fn tier_for(&self, prompt_tokens: i64) -> Option<&PricingTier> {
        match (&self.long_context, self.long_context_threshold) {
            (Some(long), Some(threshold)) if prompt_tokens > threshold => Some(long),
            _ => self.standard.as_ref(),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Pricing {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text_tokens: Option<PricingCategory>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub images: Option<PricingCategory>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_tokens: Option<PricingCategory>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embeddings: Option<PricingCategory>,
}

impl Pricing {
    pub fn text_tokens(&self) -> PricingCategory {
        self.text_tokens.clone().unwrap_or_default()
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Modalities {
    #[serde(default)]
    pub input: Vec<String>,
    #[serde(default)]
    pub output: Vec<String>,
}

/// One entry of the model registry (`models.json`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Model {
    pub id: String,
    pub name: String,
    pub provider: String,
    #[serde(default)]
    pub family: Option<String>,
    #[serde(default)]
    pub created_at: Option<String>,
    #[serde(default)]
    pub context_window: Option<i64>,
    #[serde(default)]
    pub max_output_tokens: Option<i64>,
    #[serde(default)]
    pub knowledge_cutoff: Option<String>,
    #[serde(default)]
    pub modalities: Modalities,
    #[serde(default)]
    pub capabilities: Vec<String>,
    #[serde(default)]
    pub pricing: Pricing,
    #[serde(default)]
    pub metadata: Map<String, Value>,
    #[serde(default)]
    pub unlisted_at: Option<String>,
}

/// `Model::ModelType`: what kind of output a model produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    Chat,
    Embedding,
    Moderation,
    Image,
    Audio,
    Video,
    Rerank,
    Judgment,
}

impl Model {
    /// `Model.default`: what `assume_model_exists` uses when the registry has no entry.
    pub fn default_for(model_id: &str, provider: &str) -> Model {
        let mut name = model_id.replace('-', " ");
        if let Some(first) = name.get(0..1) {
            name = first.to_uppercase() + &name[1..].to_lowercase();
        }
        let mut metadata = Map::new();
        metadata.insert(
            "warning".into(),
            "Assuming model exists, capabilities may not be accurate".into(),
        );
        Model {
            id: model_id.to_string(),
            name,
            provider: provider.to_string(),
            family: None,
            created_at: None,
            context_window: None,
            max_output_tokens: None,
            knowledge_cutoff: None,
            modalities: Modalities {
                input: vec!["text".into(), "image".into()],
                output: vec!["text".into()],
            },
            capabilities: ["function_calling", "streaming", "vision", "structured_output"]
                .map(String::from)
                .to_vec(),
            pricing: Pricing::default(),
            metadata,
            unlisted_at: None,
        }
    }

    pub fn supports(&self, capability: &str) -> bool {
        self.capabilities.iter().any(|c| c == capability)
    }

    pub fn is_unlisted(&self) -> bool {
        self.unlisted_at.is_some()
    }

    pub fn label(&self) -> String {
        format!("{} - {}", crate::providers::display_name(&self.provider), self.name)
    }

    /// Reasoning controls the model accepts (`metadata.reasoning_options`).
    pub fn reasoning_options(&self) -> Vec<Map<String, Value>> {
        self.metadata
            .get("reasoning_options")
            .and_then(Value::as_array)
            .map(|opts| opts.iter().filter_map(|o| o.as_object().cloned()).collect())
            .unwrap_or_default()
    }

    pub fn reasoning_option(&self, kind: &str) -> Option<Map<String, Value>> {
        self.reasoning_options()
            .into_iter()
            .find(|o| o.get("type").and_then(Value::as_str) == Some(kind))
    }

    pub fn reasoning_option_values(&self, kind: &str) -> Vec<String> {
        self.reasoning_option(kind)
            .and_then(|o| o.get("values").and_then(Value::as_array).cloned())
            .map(|v| {
                v.iter()
                    .map(|x| x.as_str().map(str::to_string).unwrap_or_else(|| x.to_string()))
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn model_type(&self) -> ModelType {
        let has = |m: &str| self.modalities.output.iter().any(|o| o == m);
        if has("embeddings") {
            ModelType::Embedding
        } else if has("moderation") {
            ModelType::Moderation
        } else if has("image") {
            ModelType::Image
        } else if has("audio") {
            ModelType::Audio
        } else if has("video") {
            ModelType::Video
        } else if has("rerank") {
            ModelType::Rerank
        } else if has("judgment") {
            ModelType::Judgment
        } else {
            ModelType::Chat
        }
    }

    /// `Model#cost_for`.
    pub fn cost_for(&self, tokens: &crate::Tokens) -> crate::Cost {
        crate::Cost::new(tokens, Some(self), crate::cost::Tier::Standard)
    }
}
