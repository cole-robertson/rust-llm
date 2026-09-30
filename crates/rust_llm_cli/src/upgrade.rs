//! Port of `lib/generators/ruby_llm/upgrade/upgrade_generator.rb`: the 2.0 -> 2.1 schema upgrade
//! (`upgrade_ruby_llm_to_2_1.rb.tt`) as a Loco migration that adds `rust_llm_mcp_credentials`
//! and `rust_llm_tool_calls.pending_input` when they are missing.

use crate::Generator;

const MIGRATION: &str = include_str!("../templates/upgrade/migration.rs");

/// The name the generated migration file ends with (`upgrade_ruby_llm_to_2_1`).
pub const MIGRATION_SUFFIX: &str = "_upgrade_rust_llm_to_2_1";

/// `rust-llm generate upgrade`: `create_migration_file`.
pub fn generate(g: &mut Generator) -> Result<(), String> {
    if !g.exists("migration/src/lib.rs") {
        return Err(
            "Run this from the root of a Loco app (migration/src/lib.rs not found).".into(),
        );
    }
    crate::install::migration_template(g, MIGRATION_SUFFIX, MIGRATION);
    g.note("\n  Then run: cargo loco db migrate");
    Ok(())
}
