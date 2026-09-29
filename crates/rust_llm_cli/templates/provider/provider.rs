//! {{class_name}} API integration: a provider scaffold from `rust-llm generate provider`
//! (RubyLLM's `provider/templates/core/provider.rb.erb`). It holds what the Ruby class declares;
//! wiring it into the `Provider` enum is listed in the generator's output.

use crate::config::Config;
use crate::providers::ProtocolName;

/// `Provider.slug`.
pub const SLUG: &str = "{{slug}}";
/// `Provider.name`.
pub const DISPLAY: &str = "{{class_name}}";
/// `protocol :{{protocol_name}}`, the default (first registered) protocol.
pub const PROTOCOL: ProtocolName = ProtocolName::{{protocol_variant}};
pub const DEFAULT_API_BASE: &str = "{{api_base}}";

/// `configuration_options`.
pub const CONFIGURATION_OPTIONS: &[&str] = &["{{slug}}_api_key", "{{slug}}_api_base"];
/// `configuration_requirements`.
pub const CONFIGURATION_REQUIREMENTS: &[&str] = &["{{slug}}_api_key"];
/// `assume_models_exist?`: accept model ids missing from the registry.
pub const ASSUME_MODELS_EXIST: bool = {{dynamic_models}};

/// `api_base`.
pub fn api_base(config: &Config) -> String {
    config.get("{{slug}}_api_base").unwrap_or(DEFAULT_API_BASE).to_string()
}

/// `headers`.
pub fn headers(config: &Config) -> Vec<(String, String)> {
    let key = config.get("{{slug}}_api_key").unwrap_or_default();
    vec![("Authorization".to_string(), format!("Bearer {key}"))]
}
