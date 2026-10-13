//! `lib/generators/ruby_llm/upgrade/upgrade_generator.rb`: moves an app's schema from the previous
//! release to this one. RubyLLM ships the upgrade for the changes since its previous release
//! only, and replaces it each release.
//!
//! RustLLM starts at 2.1.1, so there is no earlier RustLLM schema to upgrade: `rust-llm generate
//! install` creates the 2.1.1 tables. The command stays so the next release can ship its upgrade
//! under the same name.

use crate::Generator;

/// `rust-llm generate upgrade`: nothing to do in this release.
pub fn generate(g: &mut Generator) -> Result<(), String> {
    generate_with(g, &[])
}

/// `rust-llm generate upgrade [message:ChatMessage]`.
pub fn generate_with(g: &mut Generator, mappings: &[&str]) -> Result<(), String> {
    if !g.exists("migration/src/lib.rs") {
        return Err(
            "Run this from the root of a Loco app (migration/src/lib.rs not found).".into(),
        );
    }
    for mapping in mappings {
        match mapping.split_once(':') {
            Some(("message", model)) if !model.is_empty() => {}
            _ => return Err(format!("Unknown model mapping: {mapping}")),
        }
    }
    g.note("\n  Nothing to upgrade: RustLLM 2.1.1 is the first release of this schema.");
    g.note("  `rust-llm generate install` creates it.");
    Ok(())
}
