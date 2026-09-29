//! Port of `lib/generators/ruby_llm/provider/scaffold.rb` in `core` mode (`script/generate-provider`):
//! a provider module and its test inside the `rust_llm` crate. There is no gem split in Rust, so
//! the `provider-gem` mode is not ported.
//!
//! RubyLLM registers the new class by inserting sorted lines into `lib/ruby_llm.rb`,
//! `.env.example`, and the spec/VCR configuration. Here providers are a closed `Provider` enum
//! with a match arm per provider in several methods; the generator registers the module
//! (`pub mod <slug>;` in `providers.rs`) and prints the enum wiring rather than rewriting match
//! arms it cannot parse reliably.

use crate::{Anchor, Generator, render};

const PROVIDER: &str = include_str!("../templates/provider/provider.rs");
const PROVIDER_TEST: &str = include_str!("../templates/provider/provider_test.rs");

/// `Scaffold::SUPPORTED_DIALECTS` that this port has a wire protocol for.
const DIALECTS: &[(&str, &str, &str)] = &[
    ("chat_completions", "chat_completions", "ChatCompletions"),
    ("responses", "responses", "Responses"),
    ("anthropic", "anthropic", "Anthropic"),
    ("gemini", "gemini", "Gemini"),
    // `Ollama::ChatCompletions` is Chat Completions with Ollama's quirks.
    ("ollama", "chat_completions", "ChatCompletions"),
];

pub struct Options<'a> {
    pub dialect: Option<&'a str>,
    pub api_base: Option<&'a str>,
    pub dynamic_models: bool,
}

/// `normalize_provider_name` + `classify` + `slug`.
fn names(name: &str) -> Result<(String, String), String> {
    let name = name.trim();
    let valid = name.chars().next().is_some_and(|c| c.is_ascii_alphabetic())
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
    if !valid {
        return Err(
            "provider name must start with a letter and contain only letters, numbers, - or _"
                .into(),
        );
    }
    let class = if name.chars().next().is_some_and(|c| c.is_ascii_uppercase())
        && name.chars().all(|c| c.is_ascii_alphanumeric())
    {
        name.to_string()
    } else {
        crate::camelize(name)
    };
    let slug = crate::underscore(&class);
    Ok((class, slug))
}

/// `rust-llm generate provider NAME`, run from the rust-llm repository root.
pub fn generate(g: &mut Generator, name: &str, options: &Options) -> Result<(), String> {
    const PROVIDERS: &str = "crates/rust_llm/src/providers.rs";
    if !g.exists(PROVIDERS) {
        return Err(format!(
            "{PROVIDERS} not found: run this from the rust-llm repository root (or pass --destination)."
        ));
    }
    let (class, slug) = names(name)?;
    let dialect = options
        .dialect
        .unwrap_or("chat_completions")
        .replace('-', "_");
    let Some(&(_, protocol_name, protocol_variant)) =
        DIALECTS.iter().find(|(d, _, _)| *d == dialect)
    else {
        let known: Vec<&str> = DIALECTS.iter().map(|(d, _, _)| *d).collect();
        return Err(format!(
            "unsupported dialect: {dialect:?}. Expected one of: {} (converse is not ported)",
            known.join(", ")
        ));
    };
    let api_base = options.api_base.unwrap_or("https://api.example.com/v1");
    let dynamic = if options.dynamic_models {
        "true"
    } else {
        "false"
    };
    let vars = [
        ("class_name", class.as_str()),
        ("slug", slug.as_str()),
        ("protocol_name", protocol_name),
        ("protocol_variant", protocol_variant),
        ("api_base", api_base),
        ("dynamic_models", dynamic),
    ];

    g.file(
        &format!("crates/rust_llm/src/providers/{slug}.rs"),
        &render(PROVIDER, &vars),
    );
    g.file(
        &format!("crates/rust_llm/tests/provider_{slug}.rs"),
        &render(PROVIDER_TEST, &vars),
    );
    g.inject(
        PROVIDERS,
        &format!("pub mod {slug};"),
        Anchor::Before("use crate::config::Config;"),
    );

    let variant = class.as_str();
    g.note(format!(
        "\n  To route chats to it, wire {class} into crates/rust_llm/src/providers.rs:"
    ));
    g.note(format!(
        "     - add `{variant}` to `enum Provider` and to `ALL`"
    ));
    g.note(format!(
        "     - slug: `Provider::{variant} => {slug}::SLUG`, display: `{slug}::DISPLAY`"
    ));
    g.note(format!(
        "     - configuration_requirements: `{slug}::CONFIGURATION_REQUIREMENTS`"
    ));
    g.note(format!("     - api_base: `Provider::{variant} => return Ok({slug}::api_base(config))`, headers: `{slug}::headers(config)`"));
    g.note(format!(
        "     - default_protocol / supports_protocol: `{slug}::PROTOCOL`"
    ));
    if options.dynamic_models {
        g.note(format!(
            "     - assume_models_exist: `{slug}::ASSUME_MODELS_EXIST`"
        ));
    }
    g.note(format!("     - crates/rust_llm/src/config.rs PROVIDER_OPTIONS: \"{slug}_api_key\", \"{slug}_api_base\" (read from {0}_API_KEY / {0}_API_BASE)", slug.to_uppercase()));
    g.note(format!(
        "  Then: bin/fw cargo test -p rust_llm --test provider_{slug}"
    ));
    Ok(())
}
