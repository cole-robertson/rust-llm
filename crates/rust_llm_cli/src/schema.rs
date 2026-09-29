//! Port of `lib/generators/ruby_llm/schema/schema_generator.rb`: an empty schema type. RubyLLM's
//! is a `Schematist::Schema` subclass; here it is a `schemars::JsonSchema` struct, which
//! `Chat::with_schema_for` turns into the same JSON schema.

use crate::{Anchor, Generator, class_name, render};

const SCHEMA: &str = include_str!("../templates/schema/schema.rs");

/// `rust-llm generate schema NAME`.
pub fn generate(g: &mut Generator, name: &str) -> Result<(), String> {
    let class = class_name(name, "Schema")?;
    let file = crate::underscore(&class);

    // The same major version rust_llm uses, so `with_schema_for` accepts the derive.
    g.dependency("Cargo.toml", "schemars", "\"1.2\"");
    g.module_dir(
        "schemas",
        "//! Structured-output schemas (`rust-llm generate schema NAME`).\n",
    );
    g.file(
        &format!("src/schemas/{file}.rs"),
        &render(SCHEMA, &[("class_name", &class)]),
    );
    g.inject(
        "src/schemas/mod.rs",
        &format!("pub mod {file};"),
        Anchor::Sorted("pub mod "),
    );
    Ok(())
}
