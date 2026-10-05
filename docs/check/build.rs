//! Turns the README's ```rust blocks into doctests. The README shows plain snippets (GitHub neither
//! hides rustdoc's `# ` lines nor highlights `rust,no_run`), so the scaffolding is added here:
//! items (`use`, `struct`, `impl`, attributes) stay at the top level, the remaining statements go
//! into an `async fn`, and a stub `Weather` tool and Loco `ctx` are provided when a block uses
//! them without defining them. Blocks with their own `fn main` are kept as they are.

use std::{env, fs, path::Path};

const PRELUDE: &str = r#"
#[allow(dead_code)]
struct Ctx { db: sea_orm::DatabaseConnection }
"#;

const WEATHER_STUB: &str = r#"
struct Weather;
#[async_trait::async_trait]
impl rust_llm::Tool for Weather {
    fn description(&self) -> String { String::new() }
    async fn execute(
        &self,
        _: serde_json::Map<String, serde_json::Value>,
        _: &rust_llm::ToolCall,
    ) -> Result<rust_llm::ToolResult, rust_llm::ToolError> {
        Ok("".into())
    }
}
"#;

fn main() {
    let readme = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../README.md");
    println!("cargo:rerun-if-changed={}", readme.display());
    let text = fs::read_to_string(&readme).expect("README.md");

    let mut out = String::from("Compiled samples from the README.\n\n");
    let mut lines = text.lines();
    while let Some(line) = lines.next() {
        if line.trim_end() != "```rust" {
            continue;
        }
        let block: Vec<&str> = lines
            .by_ref()
            .take_while(|l| l.trim_end() != "```")
            .collect();
        out.push_str("```rust,no_run\n");
        out.push_str(&doctest(&block));
        out.push_str("```\n\n");
    }

    let dest = Path::new(&env::var("OUT_DIR").unwrap()).join("readme.md");
    fs::write(dest, out).expect("write readme.md");
}

fn doctest(block: &[&str]) -> String {
    let source = block.join("\n");
    if source.contains("fn main") {
        return format!("{source}\n");
    }

    let (mut items, mut body) = (String::new(), String::new());
    let mut in_item = false;
    let mut depth = 0i32;
    for line in block {
        let starts_item = ["use ", "struct ", "impl ", "#["]
            .iter()
            .any(|p| line.starts_with(p));
        if in_item || starts_item {
            in_item = true;
            depth += line.matches('{').count() as i32 - line.matches('}').count() as i32;
            items.push_str(line);
            items.push('\n');
            let end = line.trim_end();
            if depth == 0 && (end.ends_with('}') || end.ends_with(';')) {
                in_item = false;
            }
        } else {
            body.push_str("    ");
            body.push_str(line);
            body.push('\n');
        }
    }

    let mut out = String::from(PRELUDE);
    if source.contains("Weather") && !source.contains("struct Weather;") {
        out.push_str(WEATHER_STUB);
    }
    out.push_str(&items);
    out.push_str(
        "\nasync fn run(ctx: &Ctx) -> Result<(), Box<dyn std::error::Error>> {\n    let _ = ctx;\n",
    );
    out.push_str(&body);
    out.push_str("    Ok(())\n}\n");
    out
}
