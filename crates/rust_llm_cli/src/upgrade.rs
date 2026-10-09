//! Port of `lib/generators/ruby_llm/upgrade/upgrade_generator.rb`: the 2.0 -> 2.1 schema upgrade
//! (`upgrade_ruby_llm_to_2_1.rb.tt`) as a Loco migration that adds what 2.1 needs when it is
//! missing (see the template for the list).

use crate::Generator;

const MIGRATION: &str = include_str!("../templates/upgrade/migration.rs");

/// The name the generated migration file ends with (`upgrade_ruby_llm_to_2_1`).
pub const MIGRATION_SUFFIX: &str = "_upgrade_rust_llm_to_2_1";

/// `rust-llm generate upgrade`: `create_migration_file`.
pub fn generate(g: &mut Generator) -> Result<(), String> {
    generate_with(g, &[])
}

/// `rust-llm generate upgrade [message:ChatMessage]`: like the Rails generator's model mappings,
/// `message:` names the app's message model, whose table (`chat_messages`) gets `cache_ttl`.
pub fn generate_with(g: &mut Generator, mappings: &[&str]) -> Result<(), String> {
    if !g.exists("migration/src/lib.rs") {
        return Err(
            "Run this from the root of a Loco app (migration/src/lib.rs not found).".into(),
        );
    }
    let mut migration = MIGRATION.to_string();
    for mapping in mappings {
        match mapping.split_once(':') {
            Some(("message", model)) if !model.is_empty() => {
                migration = migration.replace(
                    "const MESSAGES: &str = \"messages\";",
                    &format!("const MESSAGES: &str = \"{}\";", table_name(model)),
                );
            }
            _ => return Err(format!("Unknown model mapping: {mapping}")),
        }
    }
    crate::install::migration_template(g, MIGRATION_SUFFIX, &migration);
    g.note("\n  Then run: cargo loco db migrate");
    Ok(())
}

/// `model_name.tableize`: `ChatMessage` -> `chat_messages`, `Admin::Note` -> `admin_notes`.
fn table_name(model: &str) -> String {
    let mut snake = String::new();
    for (i, c) in model.replace("::", "").chars().enumerate() {
        if c.is_ascii_uppercase() {
            if i > 0 {
                snake.push('_');
            }
            snake.push(c.to_ascii_lowercase());
        } else {
            snake.push(c);
        }
    }
    if snake.ends_with('s') {
        snake
    } else if let Some(stem) = snake.strip_suffix('y') {
        format!("{stem}ies")
    } else {
        format!("{snake}s")
    }
}
