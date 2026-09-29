//! Port of `lib/generators/ruby_llm/tool/tool_generator.rb`: a tool plus its call/result
//! components (the `tool_calls/_<name>` and `tool_results/_<name>` partials).

use crate::{Anchor, Generator, class_name, render};

const TOOL: &str = include_str!("../templates/tool/tool.rs");
const TOOL_CALL: &str = include_str!("../templates/tool/tool_call.tsx");
const TOOL_RESULT: &str = include_str!("../templates/tool/tool_result.tsx");

/// Where the chat UI looks up per-tool components (`import.meta.glob`), like the partial prefix.
pub const MESSAGE_COMPONENTS: &str = "frontend/components/messages";

/// `rust-llm generate tool NAME`.
pub fn generate(g: &mut Generator, name: &str) -> Result<(), String> {
    let class = class_name(name, "Tool")?;
    let file = crate::underscore(&class);
    // Mirrors `Tool::name()`, the key the runtime component lookup uses.
    let tool_name = file.trim_end_matches("_tool").to_string();
    let display = class.strip_suffix("Tool").unwrap_or(&class).to_string();
    if tool_name == "default" {
        return Err("`default` is the fallback component name; pick another tool name".into());
    }
    let vars = [
        ("class_name", class.as_str()),
        ("tool_name", tool_name.as_str()),
        ("display_name", display.as_str()),
    ];

    g.module_dir(
        "tools",
        "//! RustLLM tools (`rust-llm generate tool NAME`).\n",
    );
    g.file(&format!("src/tools/{file}.rs"), &render(TOOL, &vars));
    g.inject(
        "src/tools/mod.rs",
        &format!("pub mod {file};"),
        Anchor::Sorted("pub mod "),
    );
    g.file(
        &format!("{MESSAGE_COMPONENTS}/tool_calls/{tool_name}.tsx"),
        &render(TOOL_CALL, &vars),
    );
    g.file(
        &format!("{MESSAGE_COMPONENTS}/tool_results/{tool_name}.tsx"),
        &render(TOOL_RESULT, &vars),
    );
    if !g.exists(&format!("{MESSAGE_COMPONENTS}/types.ts")) {
        g.note("\n  The tool components import from frontend/components/messages/; run `rust-llm generate chat_ui` to create it.");
    }
    g.note(format!(
        "\n  Use it: chat.with_tool(crate::tools::{file}::{class})"
    ));
    Ok(())
}
