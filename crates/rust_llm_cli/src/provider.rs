//! Port of `lib/generators/ruby_llm/provider/scaffold.rb` in `core` mode (`script/generate-provider`):
//! a provider module and its test inside the `rust_llm` crate. There is no gem split in Rust, so
//! the `provider-gem` mode is not ported.
//!
//! RubyLLM registers the new class by inserting sorted lines into `lib/ruby_llm.rb`,
//! `.env.example`, the spec/VCR configuration, and (with `--models-dev-provider`) the
//! `MODELS_DEV_PROVIDER_MAP` in `lib/ruby_llm/models.rb`. Here providers are a closed `Provider`
//! enum with a match arm per provider in several methods; the generator registers the module
//! (`pub mod <slug>;` in `providers.rs`, sorted) and the models.dev key (sorted into
//! `MODELS_DEV_PROVIDER_MAP` in `models/refresh.rs`), and prints the enum wiring rather than
//! rewriting match arms it cannot parse reliably. Like Ruby, a core file that is not there is
//! left alone.

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
    /// `--models-dev-provider`: the models.dev catalog key that maps to this provider.
    pub models_dev_provider: Option<&'a str>,
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
    const MODELS_DEV: &str = "crates/rust_llm/src/models/refresh.rs";
    let (class, slug) = names(name)?;
    // `blank_to_nil`
    let models_dev_provider = options
        .models_dev_provider
        .map(str::trim)
        .filter(|key| !key.is_empty());
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
    // `update_core_registration`: sorted among the `pub mod` lines, else above the first `use`.
    if let Some(providers) = g.read(PROVIDERS) {
        let anchor = if providers.lines().any(|l| l.starts_with("pub mod ")) {
            Anchor::Sorted("pub mod ")
        } else {
            Anchor::Before("use crate::config::Config;")
        };
        g.inject(PROVIDERS, &format!("pub mod {slug};"), anchor);
    }
    // `update_core_models_dev_map if models_dev_provider`
    if let Some(key) = models_dev_provider
        && let Some(refresh) = g.read(MODELS_DEV)
    {
        let entry = format!("    (\"{key}\", \"{slug}\"),");
        match models_dev_anchor(&refresh, &entry) {
            Some(anchor) => g.inject(MODELS_DEV, &entry, anchor),
            None => g.note(format!(
                "  {MODELS_DEV}: MODELS_DEV_PROVIDER_MAP not found; add `{}` yourself.",
                entry.trim()
            )),
        }
    }
    if !g.exists(PROVIDERS) {
        g.note(format!(
            "  {PROVIDERS} not found, so nothing was registered (run from the rust-llm repository root or pass --destination)."
        ));
    }

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

/// `insert_sorted_entry` for `MODELS_DEV_PROVIDER_MAP`: before the first entry that sorts after
/// `entry`, else after the last one. `None` when the map is not in the file.
fn models_dev_anchor<'a>(refresh: &'a str, entry: &str) -> Option<Anchor<'a>> {
    let block: Vec<&str> = refresh
        .lines()
        .skip_while(|l| !l.contains("const MODELS_DEV_PROVIDER_MAP"))
        .skip(1)
        .take_while(|l| l.starts_with("    (\""))
        .collect();
    let last = block.last()?;
    let key = entry.trim_end_matches(',');
    Some(match block.iter().find(|l| key < l.trim_end_matches(',')) {
        Some(next) => Anchor::Before(next),
        None => Anchor::After(last),
    })
}
