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
    /// `PricingCategory.long_context_from_cost`: models.dev's `context` tier (or the older
    /// `context_over_200k`) as long-context rates and the prompt size they start above.
    pub fn long_context_from_cost(cost: &Value) -> Option<(PricingTier, Option<i64>)> {
        let cost = cost.as_object()?;
        let (entry, threshold) =
            match cost
                .get("tiers")
                .and_then(Value::as_array)
                .and_then(|tiers| {
                    tiers.iter().find(|e| {
                        e.pointer("/tier/type").and_then(Value::as_str) == Some("context")
                    })
                }) {
                Some(tier) => {
                    let size = tier.pointer("/tier/size");
                    let threshold = size.and_then(Value::as_i64).or_else(|| {
                        size.and_then(Value::as_str)
                            .and_then(|s| s.trim().parse().ok())
                    });
                    (tier, threshold)
                }
                None => (
                    cost.get("context_over_200k").filter(|c| c.is_object())?,
                    Some(200_000),
                ),
            };
        let rate = |key: &str| entry.get(key).and_then(Value::as_f64);
        let rates = PricingTier {
            input_per_million: rate("input"),
            output_per_million: rate("output"),
            cache_read_input_per_million: rate("cache_read"),
            cache_write_input_per_million: rate("cache_write"),
            reasoning_output_per_million: rate("reasoning"),
        };
        (rates != PricingTier::default()).then_some((rates, threshold))
    }

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

/// One entry of the model registry (`models.json`). Built through `Model#initialize`'s
/// normalization ([`ModelData`]): times are UTC, reasoning options live in `metadata`, and a
/// long-context price models.dev reported only in `metadata.cost` is copied into `pricing`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(from = "ModelData")]
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

/// The attributes `Model.new(data)` accepts: a registry entry, plus a top-level
/// `reasoning_options` that `Model#initialize` moves into `metadata`.
#[derive(Deserialize)]
pub struct ModelData {
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
    #[serde(default)]
    pub reasoning_options: Option<Value>,
}

/// `Model#initialize`.
impl From<ModelData> for Model {
    fn from(data: ModelData) -> Model {
        let mut metadata = data.metadata;
        let mut pricing = data.pricing;
        // `pricing_data_with_long_context`.
        let mut text = pricing.text_tokens.take().unwrap_or_default();
        if text.long_context.is_none()
            && let Some((rates, threshold)) = metadata
                .get("cost")
                .and_then(PricingCategory::long_context_from_cost)
        {
            text.long_context = Some(rates);
            if threshold.is_some() {
                text.long_context_threshold = threshold;
            }
        }
        pricing.text_tokens = (text != PricingCategory::default()).then_some(text);
        // `reasoning_options_from` + `store_reasoning_options_metadata`.
        let options = normalize_reasoning_options(
            data.reasoning_options
                .as_ref()
                .filter(|v| !v.is_null())
                .or_else(|| metadata.get("reasoning_options")),
        );
        if !options.is_empty() {
            metadata.insert("reasoning_options".into(), Value::Array(options));
        }
        Model {
            id: data.id,
            name: data.name,
            provider: data.provider,
            family: data.family,
            created_at: data.created_at.map(|t| normalize_time(&t)),
            context_window: data.context_window,
            max_output_tokens: data.max_output_tokens,
            knowledge_cutoff: data.knowledge_cutoff.map(|d| normalize_date(&d)),
            modalities: data.modalities,
            capabilities: data.capabilities,
            pricing,
            metadata,
            unlisted_at: data.unlisted_at.map(|t| normalize_time(&t)),
        }
    }
}

/// `Support::Utils.to_time(value)&.utc`, written the way Ruby's `Time#to_s` writes a UTC time
/// (`2026-02-19 17:00:00 UTC`). A value that does not parse as a time is kept as given.
fn normalize_time(value: &str) -> String {
    parse_time(value)
        .map(|t| t.format("%Y-%m-%d %H:%M:%S UTC").to_string())
        .unwrap_or_else(|| value.to_string())
}

fn parse_time(value: &str) -> Option<chrono::DateTime<chrono::Utc>> {
    use chrono::{DateTime, NaiveDate, NaiveDateTime, Utc};
    let v = value.trim();
    if let Some(naive) = v
        .strip_suffix(" UTC")
        .and_then(|t| NaiveDateTime::parse_from_str(t, "%Y-%m-%d %H:%M:%S").ok())
    {
        return Some(naive.and_utc());
    }
    DateTime::parse_from_str(v, "%Y-%m-%d %H:%M:%S %z")
        .or_else(|_| DateTime::parse_from_rfc3339(v))
        .map(|t| t.with_timezone(&Utc))
        .ok()
        .or_else(|| {
            NaiveDate::parse_from_str(v, "%Y-%m-%d")
                .ok()
                .and_then(|d| d.and_hms_opt(0, 0, 0))
                .map(|t| t.and_utc())
        })
}

/// `Support::Utils.to_date`: a `YYYY-MM-DD` date. A value that does not parse is kept as given.
fn normalize_date(value: &str) -> String {
    parse_date(value)
        .map(|d| d.format("%Y-%m-%d").to_string())
        .unwrap_or_else(|| value.to_string())
}

fn parse_date(value: &str) -> Option<chrono::NaiveDate> {
    chrono::NaiveDate::parse_from_str(value.trim().get(..10)?, "%Y-%m-%d").ok()
}

/// `Model#normalize_reasoning_options`: hashes only, with `type`, `values`, and a symbol
/// `default` as strings.
fn normalize_reasoning_options(options: Option<&Value>) -> Vec<Value> {
    let to_s = |v: &Value| match v {
        Value::String(s) => Value::String(s.clone()),
        other => Value::String(other.to_string()),
    };
    let list = match options {
        Some(Value::Array(a)) => a.clone(),
        Some(Value::Null) | None => Vec::new(),
        Some(other) => vec![other.clone()],
    };
    list.into_iter()
        .filter_map(|option| {
            let mut option = option.as_object()?.clone();
            if let Some(t) = option.get("type").filter(|t| !t.is_null()) {
                option.insert("type".into(), to_s(t));
            }
            if let Some(values) = option.get("values") {
                let values = match values {
                    Value::Array(a) => a.iter().map(to_s).collect(),
                    Value::Null => Vec::new(),
                    other => vec![to_s(other)],
                };
                option.insert("values".into(), Value::Array(values));
            }
            Some(Value::Object(option))
        })
        .collect()
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
            capabilities: [
                "function_calling",
                "streaming",
                "vision",
                "structured_output",
            ]
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

    /// `created_at` as a UTC time (Ruby's `Model#created_at` is a `Time`).
    pub fn created_at_time(&self) -> Option<chrono::DateTime<chrono::Utc>> {
        parse_time(self.created_at.as_deref()?)
    }

    /// `knowledge_cutoff` as a date (Ruby's `Model#knowledge_cutoff` is a `Date`).
    pub fn knowledge_cutoff_date(&self) -> Option<chrono::NaiveDate> {
        parse_date(self.knowledge_cutoff.as_deref()?)
    }

    /// `unlisted_at` as a UTC time.
    pub fn unlisted_at_time(&self) -> Option<chrono::DateTime<chrono::Utc>> {
        parse_time(self.unlisted_at.as_deref()?)
    }

    pub fn label(&self) -> String {
        format!(
            "{} - {}",
            crate::providers::display_name(&self.provider),
            self.name
        )
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
                    .map(|x| {
                        x.as_str()
                            .map(str::to_string)
                            .unwrap_or_else(|| x.to_string())
                    })
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
