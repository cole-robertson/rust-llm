//! Port of `lib/generators/ruby_llm/upgrade/upgrade_generator.rb`.
//!
//! Upstream's only upgrade is 2.0 -> 2.1 (`upgrade_ruby_llm_to_2_1.rb.tt`: an MCP credentials
//! table and `ruby_llm_tool_calls.pending_input`). This port is 2.0.0, its install migration
//! already creates `pending_input`, and the credentials table only backs MCP OAuth, which is not
//! ported (MCP itself is: `rust_llm::mcp` over stdio and Streamable HTTP, `Chat::with_mcp`), so
//! there is nothing to write.

use crate::Generator;

/// `rust-llm generate upgrade`: writes nothing and says so.
pub fn generate(g: &mut Generator) {
    g.note(format!(
        "  No upgrade migrations exist for rust_llm {} yet; `rust-llm generate install` creates the current schema. Nothing was written.",
        rust_llm::VERSION
    ));
}
