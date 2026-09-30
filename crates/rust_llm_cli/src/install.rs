//! Port of `lib/generators/ruby_llm/install/install_generator.rb`: dependencies, the migration,
//! the initializer, the chat/message models, and the convention directories.

use std::path::Path;

use crate::{Anchor, Generator};

const MIGRATION: &str = include_str!("../templates/install/migration.rs");
const INITIALIZER: &str = include_str!("../templates/install/initializer.rs");
const CHAT_MODEL: &str = include_str!("../templates/install/chat_model.rs");
const MESSAGE_MODEL: &str = include_str!("../templates/install/message_model.rs");

/// The name every generated migration file ends with; an existing one is reused, like
/// `migration_template` refusing to add a second `create_ruby_llm_records`.
pub const MIGRATION_SUFFIX: &str = "_create_rust_llm_records";

/// `rust-llm generate install [--path RUST_LLM_CHECKOUT]`.
pub fn generate(g: &mut Generator, rust_llm_path: Option<&str>) -> Result<(), String> {
    if !g.exists("Cargo.toml") || !g.exists("migration/src/lib.rs") {
        return Err(
            "Run this from the root of a Loco app (Cargo.toml and migration/src/lib.rs not found)."
                .into(),
        );
    }
    add_dependencies(g, rust_llm_path)?;
    create_migration(g);
    create_model_files(g);
    create_initializer(g);
    create_convention_directories(g);
    show_install_info(g);
    Ok(())
}

fn add_dependencies(g: &mut Generator, rust_llm_path: Option<&str>) -> Result<(), String> {
    let spec = |krate: &str| -> Result<String, String> {
        let Some(root) = rust_llm_path else {
            return Ok("\"2.0.0\"".to_string());
        };
        let dir = Path::new(root).join("crates").join(krate);
        let dir = dir
            .canonicalize()
            .map_err(|e| format!("--path {root}: crates/{krate} not found ({e})"))?;
        Ok(format!("{{ path = \"{}\" }}", dir.display()))
    };
    let (core, loco) = (spec("rust_llm")?, spec("rust_llm_loco")?);
    g.dependency("Cargo.toml", "rust_llm", &core);
    g.dependency("Cargo.toml", "rust_llm_loco", &loco);
    g.dependency("migration/Cargo.toml", "rust_llm_loco", &loco);
    if rust_llm_path.is_none() {
        g.note("  Note: rust_llm is not published to crates.io yet; re-run with --path <rust-llm checkout> --force or edit the version to a path/git dependency.");
    }
    Ok(())
}

fn existing_migration(g: &Generator, suffix: &str) -> Option<String> {
    let entries = std::fs::read_dir(g.path("migration/src")).ok()?;
    entries
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter_map(|name| name.strip_suffix(".rs").map(str::to_string))
        .find(|stem| stem.ends_with(suffix))
}

fn create_migration(g: &mut Generator) {
    migration_template(g, MIGRATION_SUFFIX, MIGRATION);
}

/// `migration_template`: writes `migration/src/m<timestamp><suffix>.rs` (reusing an existing one
/// with that suffix) and registers it with the app's `Migrator`.
pub(crate) fn migration_template(g: &mut Generator, suffix: &str, content: &str) {
    let module = existing_migration(g, suffix)
        .unwrap_or_else(|| format!("{}{suffix}", chrono::Utc::now().format("m%Y%m%d_%H%M%S")));
    g.file(&format!("migration/src/{module}.rs"), content);
    // Loco's own model generator injects these two lines the same way.
    g.inject(
        "migration/src/lib.rs",
        &format!("mod {module};"),
        Anchor::Before("pub struct Migrator"),
    );
    g.inject(
        "migration/src/lib.rs",
        &format!("            Box::new({module}::Migration),"),
        Anchor::Before("inject-above"),
    );
}

fn create_model_files(g: &mut Generator) {
    g.file("src/models/chats.rs", CHAT_MODEL);
    g.file("src/models/messages.rs", MESSAGE_MODEL);
    g.inject(
        "src/models/mod.rs",
        "pub mod chats;",
        Anchor::Sorted("pub mod "),
    );
    g.inject(
        "src/models/mod.rs",
        "pub mod messages;",
        Anchor::Sorted("pub mod "),
    );
}

fn create_initializer(g: &mut Generator) {
    g.module_dir(
        "initializers",
        "//! Loco initializers (`Hooks::initializers` in `src/app.rs`).\n",
    );
    g.file("src/initializers/rust_llm.rs", INITIALIZER);
    g.inject(
        "src/initializers/mod.rs",
        "pub mod rust_llm;",
        Anchor::Sorted("pub mod "),
    );
    register_initializer(g);
}

/// Adds the initializer first in the `vec![` that `Hooks::initializers` returns. Like the kit's
/// own scaffold injections, the line may need `cargo fmt` afterwards.
fn register_initializer(g: &mut Generator) {
    let entry = "Box::new(crate::initializers::rust_llm::RustLlm)";
    let nonempty = g.read("src/app.rs").and_then(|app| {
        let f = app.find("fn initializers")?;
        let v = f + app[f..].find("vec![")? + "vec![".len();
        Some(!app[v..].trim_start().starts_with(']'))
    });
    let text = if nonempty == Some(true) {
        format!("{entry}, ")
    } else {
        entry.to_string()
    };
    g.splice("src/app.rs", entry, &text, &["fn initializers", "vec!["]);
}

fn create_convention_directories(g: &mut Generator) {
    g.module_dir(
        "tools",
        "//! RustLLM tools (`rust-llm generate tool NAME`).\n",
    );
    g.module_dir(
        "agents",
        "//! RustLLM agents (`rust-llm generate agent NAME`).\n",
    );
    g.module_dir(
        "schemas",
        "//! Structured-output schemas (`rust-llm generate schema NAME`).\n",
    );
    if !g.exists("src/prompts") {
        g.file("src/prompts/.gitkeep", "");
    }
}

fn show_install_info(g: &mut Generator) {
    g.note("\n  RustLLM installed!");
    g.note("\n  Next steps:");
    g.note("     1. Run: cargo fmt --all && cargo loco db migrate");
    g.note("     2. Set your API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, ...) in the environment or src/initializers/rust_llm.rs");
    g.note("     3. Start chatting: ChatRecord::create(&ctx.db, \"gpt-5.6-luna\", None).await? then record.ask(&ctx.db, &mut chat, \"Hello!\")");
    g.note("     4. Optional UI: rust-llm generate chat_ui");
    g.note("\n  Models come from the bundled registry; refresh it with rust_llm::models::refresh(false),");
    g.note("  and rust_llm_models rows are filled from it on first use.");
}
