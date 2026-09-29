//! Port of `lib/ruby_llm/cost.rb` for text tokens.

use serde::{Deserialize, Serialize};

use crate::model::{Model, PricingCategory, PricingTier};
use crate::tokens::Tokens;

const PER_MILLION: f64 = 1_000_000.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    Standard,
    Batch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Component {
    Input,
    Output,
    CacheRead,
    CacheWrite,
    Thinking,
}

const COMPONENTS: [Component; 5] = [
    Component::Input,
    Component::Output,
    Component::CacheRead,
    Component::CacheWrite,
    Component::Thinking,
];

/// USD cost of some tokens. A component is `None` when there were no tokens for it, and it is
/// listed in `missing` when there were tokens but no price, which makes `total()` unknown rather
/// than silently low.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Cost {
    pub input: Option<f64>,
    pub output: Option<f64>,
    pub cache_read: Option<f64>,
    pub cache_write: Option<f64>,
    pub thinking: Option<f64>,
    missing: Vec<Component>,
    reported: bool,
    complete: bool,
    reported_total: Option<f64>,
}

impl Cost {
    pub fn new(tokens: &Tokens, model: Option<&Model>, tier: Tier) -> Cost {
        let text = model.and_then(|m| m.pricing.text_tokens.clone()).unwrap_or_default();
        let pricer = Pricer { tokens, text: &text, tier };
        let mut cost = Cost { complete: true, ..Default::default() };
        for component in COMPONENTS {
            cost.set(component, pricer.amount_for(component));
            if pricer.missing(component) {
                cost.missing.push(component);
            }
        }
        cost.reported_total = tokens.reported_cost;
        cost.reported = COMPONENTS.iter().any(|&c| pricer.tokens_for(c).is_some()) || tokens.reported_cost.is_some();
        cost
    }

    /// `Cost.aggregate`: sum of the parts, and incomplete if any part is.
    pub fn aggregate<'a>(costs: impl IntoIterator<Item = &'a Cost>, complete: bool) -> Cost {
        let costs: Vec<&Cost> = costs.into_iter().filter(|c| c.reported).collect();
        let mut out = Cost { complete, reported: !costs.is_empty(), ..Default::default() };
        for component in COMPONENTS {
            let missing = costs.iter().any(|c| c.missing.contains(&component));
            if missing {
                out.missing.push(component);
                continue;
            }
            let values: Vec<f64> = costs.iter().filter_map(|c| c.get(component)).collect();
            out.set(component, (!values.is_empty()).then(|| values.iter().sum()));
        }
        let totals: Vec<Option<f64>> = costs.iter().map(|c| c.total()).collect();
        if !totals.is_empty() && totals.iter().all(Option::is_some) {
            out.reported_total = Some(totals.iter().flatten().sum());
        }
        out
    }

    /// `Cost.from_h`: a cost as it was recorded (e.g. the stored usage columns), not re-priced.
    /// Without a recorded total, a component with tokens but no amount makes the total unknown.
    pub fn from_recorded(
        amounts: [Option<f64>; 5],
        total: Option<f64>,
        tokens: &Tokens,
    ) -> Cost {
        let mut cost = Cost { complete: true, reported_total: total, ..Default::default() };
        let counts = [tokens.input, tokens.output, tokens.cache_read, tokens.cache_write, tokens.thinking];
        for (i, component) in COMPONENTS.into_iter().enumerate() {
            cost.set(component, amounts[i]);
            if total.is_none() && counts[i].unwrap_or(0) > 0 && amounts[i].is_none() {
                cost.missing.push(component);
            }
        }
        cost.reported = total.is_some() || counts.iter().any(Option::is_some);
        cost
    }

    pub fn get(&self, component: Component) -> Option<f64> {
        match component {
            Component::Input => self.input,
            Component::Output => self.output,
            Component::CacheRead => self.cache_read,
            Component::CacheWrite => self.cache_write,
            Component::Thinking => self.thinking,
        }
    }

    fn set(&mut self, component: Component, value: Option<f64>) {
        match component {
            Component::Input => self.input = value,
            Component::Output => self.output = value,
            Component::CacheRead => self.cache_read = value,
            Component::CacheWrite => self.cache_write = value,
            Component::Thinking => self.thinking = value,
        }
    }

    pub fn missing(&self) -> &[Component] {
        &self.missing
    }

    /// Total USD, or `None` when any priced component is unknown.
    pub fn total(&self) -> Option<f64> {
        if !self.complete || !self.reported {
            return None;
        }
        if let Some(total) = self.reported_total {
            return Some(total);
        }
        if !self.missing.is_empty() {
            return None;
        }
        let amounts: Vec<f64> = COMPONENTS.iter().filter_map(|&c| self.get(c)).collect();
        (!amounts.is_empty()).then(|| amounts.iter().sum())
    }

    /// `Cost.new(amounts:, missing:, reported:)`: already-priced components, e.g. at batch rates.
    pub fn from_amounts(amounts: [Option<f64>; 5], missing: Vec<Component>, reported: bool) -> Cost {
        let mut cost = Cost { complete: true, reported, missing, ..Default::default() };
        for (i, component) in COMPONENTS.into_iter().enumerate() {
            cost.set(component, amounts[i]);
        }
        cost
    }

    /// `tokens?`: whether there was any usage to price.
    pub fn is_reported(&self) -> bool {
        self.reported
    }

    pub(crate) fn mark_incomplete(mut self) -> Self {
        self.complete = false;
        self
    }

    /// `Cost.new(category: :images, input_details:)`: output uses the image output price (text
    /// as fallback); input splits into text and image tokens when the provider reports both.
    pub fn images(tokens: &Tokens, model: Option<&Model>, input_details: Option<&serde_json::Value>) -> Cost {
        let mut cost = Cost::new(tokens, model, Tier::Standard);
        let text = model.and_then(|m| m.pricing.text_tokens.clone()).unwrap_or_default();
        let images = model.and_then(|m| m.pricing.images.clone()).unwrap_or_default();
        let prompt = tokens.input.unwrap_or(0) + tokens.cache_read.unwrap_or(0) + tokens.cache_write.unwrap_or(0);
        let text_input = text.tier_for(prompt).and_then(|t| t.input_per_million).or_else(|| text.input());
        let per = |count: i64, price: Option<f64>| if count == 0 { Some(0.0) } else { price.map(|p| count as f64 * p / PER_MILLION) };

        if let (Some(output), Some(price)) = (tokens.output, images.output()) {
            cost.output = per(output, Some(price));
            cost.missing.retain(|c| *c != Component::Output);
        }
        let detail = |key: &str| input_details.and_then(|d| d.get(key)).and_then(serde_json::Value::as_i64);
        let parts = [(detail("text_tokens"), text_input), (detail("image_tokens"), images.input().or(text_input))];
        if parts.iter().any(|(count, _)| count.is_some()) {
            cost.missing.retain(|c| *c != Component::Input);
            if parts.iter().any(|(count, price)| count.unwrap_or(0) > 0 && price.is_none()) {
                cost.input = None;
                cost.missing.push(Component::Input);
            } else {
                cost.input = Some(parts.iter().filter_map(|(count, price)| per(count.unwrap_or(0), *price)).sum());
            }
        }
        cost
    }

    /// `Cost.new(category: :audio_tokens)` for speech and transcription: input and output use the
    /// model's audio token prices, falling back to its text prices; cache and thinking stay text.
    pub fn audio(tokens: &Tokens, model: Option<&Model>) -> Cost {
        let mut cost = Cost::new(tokens, model, Tier::Standard);
        let audio = model.and_then(|m| m.pricing.audio_tokens.clone()).unwrap_or_default();
        for (component, count, price) in [
            (Component::Input, tokens.input, audio.input()),
            (Component::Output, tokens.output, audio.output()),
        ] {
            let (Some(count), Some(price)) = (count, price) else { continue };
            cost.set(component, Some(if count == 0 { 0.0 } else { count as f64 * price / PER_MILLION }));
            cost.missing.retain(|c| *c != component);
        }
        cost
    }
}

struct Pricer<'a> {
    tokens: &'a Tokens,
    text: &'a PricingCategory,
    tier: Tier,
}

impl Pricer<'_> {
    fn applicable_tier(&self) -> Option<&PricingTier> {
        match self.tier {
            Tier::Batch => self.text.batch.as_ref(),
            Tier::Standard => {
                let prompt = self.tokens.input.unwrap_or(0)
                    + self.tokens.cache_read.unwrap_or(0)
                    + self.tokens.cache_write.unwrap_or(0);
                self.text.tier_for(prompt)
            }
        }
    }

    fn thinking_priced_separately(&self) -> bool {
        let tier = self.applicable_tier();
        let Some(reasoning) = tier
            .and_then(|t| t.reasoning_output_per_million)
            .or_else(|| self.text.reasoning_output())
        else {
            return false;
        };
        match tier.and_then(|t| t.output_per_million).or_else(|| self.text.output()) {
            None => true,
            Some(output) => reasoning != output,
        }
    }

    fn tokens_for(&self, component: Component) -> Option<i64> {
        match component {
            Component::Input => self.tokens.input,
            Component::Output => self.tokens.output,
            Component::CacheRead => self.tokens.cache_read,
            Component::CacheWrite => self.tokens.cache_write,
            Component::Thinking => self.thinking_priced_separately().then_some(self.tokens.thinking).flatten(),
        }
    }

    fn price_for(&self, component: Component) -> Option<f64> {
        let tier = self.applicable_tier();
        let pick = |f: fn(&PricingTier) -> Option<f64>, standard: Option<f64>| tier.and_then(f).or(standard);
        match component {
            Component::Input => pick(|t| t.input_per_million, self.text.input()),
            Component::Output => pick(|t| t.output_per_million, self.text.output()),
            Component::CacheRead => pick(|t| t.cache_read_input_per_million, self.text.cache_read_input()),
            Component::CacheWrite => pick(|t| t.cache_write_input_per_million, self.text.cache_write_input()),
            Component::Thinking => pick(|t| t.reasoning_output_per_million, self.text.reasoning_output()),
        }
    }

    fn amount_for(&self, component: Component) -> Option<f64> {
        let count = self.tokens_for(component)?;
        if count == 0 {
            return Some(0.0);
        }
        Some(count as f64 * self.price_for(component)? / PER_MILLION)
    }

    fn missing(&self, component: Component) -> bool {
        if component == Component::Thinking && !self.thinking_priced_separately() {
            return false;
        }
        self.tokens_for(component).unwrap_or(0) > 0 && self.price_for(component).is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Pricing;

    fn priced(input: f64, output: f64) -> Model {
        let mut m = Model::default_for("priced-model", "openai");
        m.pricing = Pricing {
            text_tokens: Some(PricingCategory {
                standard: Some(PricingTier {
                    input_per_million: Some(input),
                    output_per_million: Some(output),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            ..Default::default()
        };
        m
    }

    // chat_spec.rb: "keeps manually added messages out of the conversation totals" prices 1k in / 2k out at $0.005.
    #[test]
    fn prices_input_and_output_per_million() {
        let tokens = Tokens { input: Some(1_000), output: Some(2_000), ..Default::default() };
        let cost = Cost::new(&tokens, Some(&priced(1.0, 2.0)), Tier::Standard);
        assert!((cost.total().unwrap() - 0.005).abs() < 1e-12);
    }

    #[test]
    fn an_unpriced_component_with_tokens_makes_the_total_unknown() {
        let tokens = Tokens { input: Some(1_000), cache_read: Some(500), ..Default::default() };
        let cost = Cost::new(&tokens, Some(&priced(1.0, 2.0)), Tier::Standard);
        assert_eq!(cost.missing(), &[Component::CacheRead]);
        assert_eq!(cost.total(), None);
    }

    #[test]
    fn no_usage_means_no_total() {
        assert_eq!(Cost::new(&Tokens::default(), None, Tier::Standard).total(), None);
    }
}
