//! `spec/ruby_llm/providers/{{slug}}_spec.rb` (RubyLLM's `provider_spec.rb.erb`), from
//! `rust-llm generate provider`.

use rust_llm::providers::{ProtocolName, {{slug}}};

fn config() -> rust_llm::Config {
    let mut config = rust_llm::Config::default();
    config.set("{{slug}}_api_key", "test-key");
    config.set("{{slug}}_api_base", "https://example.test/v1");
    config
}

#[test]
fn registers_a_default_protocol() {
    assert_eq!({{slug}}::PROTOCOL, ProtocolName::{{protocol_variant}});
}

#[test]
fn declares_provider_configuration() {
    assert_eq!({{slug}}::CONFIGURATION_OPTIONS, ["{{slug}}_api_key", "{{slug}}_api_base"]);
    assert_eq!({{slug}}::CONFIGURATION_REQUIREMENTS, ["{{slug}}_api_key"]);
}

#[test]
fn uses_configured_api_base_and_bearer_token() {
    let config = config();
    assert_eq!({{slug}}::api_base(&config), "https://example.test/v1");
    assert_eq!(
        {{slug}}::headers(&config),
        vec![("Authorization".to_string(), "Bearer test-key".to_string())]
    );
}

#[test]
fn defaults_the_api_base() {
    assert_eq!({{slug}}::api_base(&rust_llm::Config::default()), "{{api_base}}");
}

#[test]
fn assume_models_exist_matches_dynamic_models() {
    assert_eq!({{slug}}::ASSUME_MODELS_EXIST, {{dynamic_models}});
}
