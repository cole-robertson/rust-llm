//! Port of `lib/generators/ruby_llm/agent/agent_generator.rb`: an agent and its (empty)
//! instructions prompt.
//!
//! RubyLLM renders `instructions.txt.erb` with ERB at runtime. The Rust agent embeds a plain
//! `instructions.txt` with `include_str!`; there is no template language for prompt locals.

use crate::{Anchor, Generator, class_name, render};

const AGENT: &str = include_str!("../templates/agent/agent.rs");

/// `rust-llm generate agent NAME`.
pub fn generate(g: &mut Generator, name: &str) -> Result<(), String> {
    let class = class_name(name, "Agent")?;
    let file = crate::underscore(&class);
    let vars = [("class_name", class.as_str()), ("file_name", file.as_str())];

    g.module_dir(
        "agents",
        "//! RustLLM agents (`rust-llm generate agent NAME`).\n",
    );
    g.file(&format!("src/agents/{file}.rs"), &render(AGENT, &vars));
    g.inject(
        "src/agents/mod.rs",
        &format!("pub mod {file};"),
        Anchor::Sorted("pub mod "),
    );
    // `instructions.txt.erb.tt` is empty upstream too.
    g.file(&format!("src/prompts/{file}/instructions.txt"), "");
    Ok(())
}
