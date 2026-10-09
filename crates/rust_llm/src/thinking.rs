//! Port of `Thinking::Config` and `Thinking::Controls` from `lib/ruby_llm/thinking.rb`:
//! `with_thinking` options resolved against what the model's registry entry accepts.

use serde_json::Value;

use crate::error::{Error, Result};
use crate::model::Model;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Display {
    Summarized,
    Omitted,
    Full,
}

impl Display {
    pub fn as_str(&self) -> &'static str {
        match self {
            Display::Summarized => "summarized",
            Display::Omitted => "omitted",
            Display::Full => "full",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Intent {
    Enable,
    Disable,
}

/// `with_thinking(...)` options. Build with [`ThinkingConfig::on`], [`ThinkingConfig::off`], or the
/// `effort`/`budget`/`display` setters.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ThinkingConfig {
    pub effort: Option<String>,
    pub budget: Option<i64>,
    pub display: Option<Display>,
    pub enabled: Option<bool>,
    intent: Option<Intent>,
}

impl ThinkingConfig {
    /// `with_thinking`: turn thinking on the way the model prefers.
    pub fn on() -> Self {
        ThinkingConfig {
            intent: Some(Intent::Enable),
            ..Default::default()
        }
    }

    /// `with_thinking(false)`.
    pub fn off() -> Self {
        ThinkingConfig {
            intent: Some(Intent::Disable),
            ..Default::default()
        }
    }

    pub fn effort(effort: impl Into<String>) -> Self {
        ThinkingConfig {
            effort: Some(effort.into()),
            ..Default::default()
        }
    }

    pub fn budget(budget: i64) -> Self {
        ThinkingConfig {
            budget: Some(budget),
            ..Default::default()
        }
    }

    pub fn with_display(mut self, display: Display) -> Self {
        self.display = Some(display);
        self
    }

    pub fn is_enabled(&self) -> bool {
        self.intent.is_some()
            || self.enabled.is_some()
            || self.effort.is_some()
            || self.budget.is_some()
            || self.display.is_some()
    }

    pub fn is_disabled(&self) -> bool {
        self.enabled == Some(false) || self.effort.as_deref() == Some("none")
    }

    /// `Config#resolve`: `Ok(None)` when the model reasons anyway and needs no control.
    pub fn resolve(&self, model: &Model) -> Result<Option<ThinkingConfig>> {
        let Some(intent) = self.intent else {
            return Ok(Some(self.clone()));
        };
        let controls = Controls { model };
        let resolved = match intent {
            Intent::Enable => controls.enable(),
            Intent::Disable => controls.disable(),
        };
        match resolved {
            Resolved::AlreadyOn => Ok(None),
            Resolved::Options(options) => Ok(Some(options)),
            // `resolution_error(model)`.
            Resolved::Unsupported => {
                let (verb, hint) = match intent {
                    Intent::Enable => ("enable", "Pass effort:, budget:, or display:."),
                    Intent::Disable => ("disable", "The model registry has no off control."),
                };
                Err(Error::Argument(format!(
                    "RustLLM does not know how to {verb} thinking for {}/{}. {hint}",
                    model.provider, model.id
                )))
            }
        }
    }
}

enum Resolved {
    Options(ThinkingConfig),
    AlreadyOn,
    Unsupported,
}

const PREFERRED_EFFORTS: &[&str] = &["medium", "low", "minimal", "high", "xhigh", "max"];

struct Controls<'a> {
    model: &'a Model,
}

impl Controls<'_> {
    fn enable(&self) -> Resolved {
        if let Some(effort) = self.default_effort() {
            return Resolved::Options(ThinkingConfig::effort(effort));
        }
        if let Some(budget) = self.explicit_budget() {
            return Resolved::Options(ThinkingConfig::budget(budget));
        }
        if self.model.reasoning_option("toggle").is_some() {
            return Resolved::Options(ThinkingConfig {
                enabled: Some(true),
                ..Default::default()
            });
        }
        let values = self.model.reasoning_option_values("effort");
        if let Some(effort) = PREFERRED_EFFORTS
            .iter()
            .find(|e| values.iter().any(|v| v == *e))
        {
            return Resolved::Options(ThinkingConfig::effort(*effort));
        }
        if let Some(budget) = self.minimum_budget() {
            return Resolved::Options(ThinkingConfig::budget(budget));
        }
        if self.reasoning_model() && self.model.reasoning_options().is_empty() {
            return Resolved::AlreadyOn;
        }
        Resolved::Unsupported
    }

    fn disable(&self) -> Resolved {
        if self
            .model
            .reasoning_option_values("effort")
            .iter()
            .any(|v| v == "none")
        {
            return Resolved::Options(ThinkingConfig::effort("none"));
        }
        let budget = self.model.reasoning_option("budget_tokens");
        if let Some(min) = budget
            .as_ref()
            .and_then(|b| b.get("min"))
            .and_then(Value::as_f64)
            && min <= 0.0
        {
            return Resolved::Options(ThinkingConfig::budget(0));
        }
        // `model.provider_class&.thinking_off_control(model.id)`: Anthropic turns Sonnet 5.5's
        // thinking off with `{ enabled: false }`, which renders as `between_tools`.
        let provider_off = self.model.provider == "anthropic"
            && crate::protocols::anthropic::is_between_tools_off(&self.model.id);
        if self.model.reasoning_option("toggle").is_some() || budget.is_some() || provider_off {
            return Resolved::Options(ThinkingConfig {
                enabled: Some(false),
                ..Default::default()
            });
        }
        Resolved::Unsupported
    }

    fn default_effort(&self) -> Option<String> {
        let option = self.model.reasoning_option("effort")?;
        let default = option.get("default")?;
        let default = default
            .as_str()
            .map(str::to_string)
            .unwrap_or_else(|| default.to_string());
        (default != "none").then_some(default)
    }

    fn explicit_budget(&self) -> Option<i64> {
        let d = self
            .model
            .reasoning_option("budget_tokens")?
            .get("default")?
            .as_i64()?;
        (d > 0).then_some(d)
    }

    fn minimum_budget(&self) -> Option<i64> {
        let min = self
            .model
            .reasoning_option("budget_tokens")?
            .get("min")?
            .as_i64()?;
        Some(min.max(1))
    }

    fn reasoning_model(&self) -> bool {
        self.model.supports("reasoning")
            || self
                .model
                .metadata
                .get("reasoning")
                .and_then(Value::as_bool)
                == Some(true)
    }
}
