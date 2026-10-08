//! # rust_llm_cli
//!
//! RubyLLM's Rails generators (`lib/generators/ruby_llm/*`) as a `rust-llm` CLI for Loco + Inertia
//! \+ React apps.
//!
//! Install it from a checkout of the repository, then run it from the root of a Loco app:
//!
//! ```text
//! cargo install --path crates/rust_llm_cli
//! rust-llm generate install --path <rust-llm checkout>
//! ```
//!
//! | RubyLLM | rust-llm |
//! |---|---|
//! | `bin/rails g ruby_llm:install` | `rust-llm generate install [--path RUST_LLM_CHECKOUT]` |
//! | `bin/rails g ruby_llm:tool Weather` | `rust-llm generate tool Weather` |
//! | `bin/rails g ruby_llm:agent Support` | `rust-llm generate agent Support` |
//! | `bin/rails g ruby_llm:schema Product` | `rust-llm generate schema Product` |
//! | `bin/rails g ruby_llm:chat_ui` | `rust-llm generate chat_ui` |
//! | (none) | `rust-llm generate public_chat`: a no-sign-in, rate-limited streaming chat page |
//! | `script/generate-provider NAME` (core mode) | `rust-llm generate provider NAME [--dialect D] [--api-base URL] [--models-dev-provider KEY] [--dynamic-models] [--destination DIR]` |
//! | `bin/rails g ruby_llm:upgrade` | `rust-llm generate upgrade` |
//!
//! `g` is short for `generate`, and every generator takes `--force`. Like Rails generators,
//! existing files are skipped unless `--force`, injections are skipped when already present, and
//! every path is printed with its action (`create`, `identical`, `skip`, `force`, `insert`). An
//! injection whose anchor is missing is reported and fails the run. See the
//! [generators guide](https://github.com/cole-robertson/rust-llm/blob/main/docs/generators.md).

pub mod agent;
pub mod chat_ui;
pub mod install;
pub mod provider;
pub mod public_chat;
pub mod schema;
pub mod tool;
pub mod upgrade;

use std::fs;
use std::path::{Path, PathBuf};

/// Where an injected line goes, like Thor's `inject_into_file before:/after:`.
#[derive(Debug, Clone, Copy)]
pub enum Anchor<'a> {
    /// Before the first line containing this text.
    Before(&'a str),
    /// After the first line containing this text.
    After(&'a str),
    /// In name order among the lines starting with this prefix (above any attribute or doc comment
    /// of the next one), else at the end of the file. rustfmt keeps `mod` lines sorted this way.
    Sorted(&'a str),
    /// Before the first line that is exactly this text, trimmed (`// scaffold:nav`, not
    /// `// scaffold:nav-global`).
    BeforeLine(&'a str),
    /// At the end of the file.
    End,
}

/// Runs one generator against an app directory, recording what it did.
pub struct Generator {
    root: PathBuf,
    force: bool,
    echo: bool,
    /// `(action, relative path)`, in order.
    pub actions: Vec<(String, String)>,
    /// Injections that could not be made; the CLI exits non-zero when this is not empty.
    pub failures: Vec<String>,
    /// Extra lines printed after the actions (next steps, notes).
    pub notes: Vec<String>,
}

impl Generator {
    pub fn new(root: impl Into<PathBuf>, force: bool) -> Generator {
        Generator {
            root: root.into(),
            force,
            echo: false,
            actions: Vec::new(),
            failures: Vec::new(),
            notes: Vec::new(),
        }
    }

    /// Print each action as it happens, like `say_status`.
    pub fn echo(mut self) -> Generator {
        self.echo = true;
        self
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn path(&self, rel: &str) -> PathBuf {
        self.root.join(rel)
    }

    pub fn exists(&self, rel: &str) -> bool {
        self.path(rel).exists()
    }

    pub fn read(&self, rel: &str) -> Option<String> {
        fs::read_to_string(self.path(rel)).ok()
    }

    fn record(&mut self, action: &str, rel: &str) {
        if self.echo {
            println!("{action:>12}  {rel}");
        }
        self.actions.push((action.to_string(), rel.to_string()));
    }

    fn fail(&mut self, message: String) {
        if self.echo {
            eprintln!("{:>12}  {message}", "error");
        }
        self.failures.push(message);
    }

    pub fn note(&mut self, line: impl Into<String>) {
        self.notes.push(line.into());
    }

    /// Thor's `create_file`/`template`: `identical` when unchanged, `skip` when different
    /// (unless `--force`), otherwise `create`/`force`.
    pub fn file(&mut self, rel: &str, content: &str) {
        let path = self.path(rel);
        let action = match fs::read_to_string(&path) {
            Ok(existing) if existing == content => return self.record("identical", rel),
            Ok(_) if !self.force => return self.record("skip", rel),
            Ok(_) => "force",
            Err(_) => "create",
        };
        if let Some(dir) = path.parent()
            && let Err(e) = fs::create_dir_all(dir)
        {
            return self.fail(format!("{rel}: {e}"));
        }
        match fs::write(&path, content) {
            Ok(()) => self.record(action, rel),
            Err(e) => self.fail(format!("{rel}: {e}")),
        }
    }

    /// Thor's `inject_into_file`. `content` is inserted as whole lines; like Thor, it is
    /// `identical` (not inserted twice) when the file already contains it.
    pub fn inject(&mut self, rel: &str, content: &str, anchor: Anchor) {
        let Some(existing) = self.read(rel) else {
            return self.fail(format!(
                "{rel}: file not found; add this yourself:\n{content}"
            ));
        };
        // Whole trimmed lines, so `mod chats;` is not found inside `pub mod chats;`.
        let normalize = |s: &str| {
            let lines: Vec<&str> = s.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
            format!("\n{}\n", lines.join("\n"))
        };
        if normalize(&existing).contains(&normalize(content)) {
            return self.record("identical", rel);
        }
        let first = content
            .lines()
            .find(|l| !l.trim().is_empty())
            .unwrap_or("")
            .trim();
        // A blank file (the kit's `src/initializers/mod.rs` is a lone newline) becomes the content.
        let lines: Vec<&str> = if existing.trim().is_empty() {
            Vec::new()
        } else {
            existing.lines().collect()
        };
        let at = match anchor {
            Anchor::Before(marker) => lines.iter().position(|l| l.contains(marker)),
            Anchor::After(marker) => lines.iter().position(|l| l.contains(marker)).map(|i| i + 1),
            Anchor::Sorted(prefix) => Some(sorted_index(&lines, first, prefix)),
            Anchor::BeforeLine(marker) => lines.iter().position(|l| l.trim() == marker),
            Anchor::End => Some(lines.len()),
        };
        let Some(at) = at else {
            let marker = match anchor {
                Anchor::Before(m)
                | Anchor::After(m)
                | Anchor::Sorted(m)
                | Anchor::BeforeLine(m) => m,
                Anchor::End => "end of file",
            };
            return self.fail(format!(
                "{rel}: anchor `{marker}` not found; add this yourself:\n{content}"
            ));
        };
        let mut out: Vec<&str> = lines[..at].to_vec();
        out.extend(content.lines());
        out.extend(&lines[at..]);
        let mut text = out.join("\n");
        if existing.ends_with('\n') || existing.is_empty() {
            text.push('\n');
        }
        match fs::write(self.path(rel), text) {
            Ok(()) => self.record("insert", rel),
            Err(e) => self.fail(format!("{rel}: {e}")),
        }
    }

    /// Inserts `text` inline right after the last of `markers`, each searched after the previous
    /// one (e.g. `["fn initializers", "vec!["]`). `identical` when the file already has `present`.
    pub fn splice(&mut self, rel: &str, present: &str, text: &str, markers: &[&str]) {
        let Some(existing) = self.read(rel) else {
            return self.fail(format!("{rel}: file not found; add `{present}` yourself"));
        };
        if existing.contains(present) {
            return self.record("identical", rel);
        }
        let mut at = 0;
        for marker in markers {
            match existing[at..].find(marker) {
                Some(i) => at += i + marker.len(),
                None => {
                    return self.fail(format!(
                        "{rel}: `{}` not found; add `{present}` yourself",
                        markers.join(" .. ")
                    ));
                }
            }
        }
        let updated = format!("{}{text}{}", &existing[..at], &existing[at..]);
        match fs::write(self.path(rel), updated) {
            Ok(()) => self.record("insert", rel),
            Err(e) => self.fail(format!("{rel}: {e}")),
        }
    }

    /// Adds `name` in sorted position to a one-line `import { a, c } from "module"`, like the
    /// kit's `scaffold:pages` does for the sidebar.
    pub fn named_import(&mut self, rel: &str, module: &str, name: &str) {
        let Some(existing) = self.read(rel) else {
            return self.fail(format!(
                "{rel}: file not found; import `{name}` from \"{module}\" yourself"
            ));
        };
        let suffix = format!(" }} from \"{module}\"");
        let mut found = false;
        let mut changed = false;
        let lines: Vec<String> = join_wrapped_imports(&existing)
            .iter()
            .map(|line| {
                let Some(names) = line
                    .strip_prefix("import { ")
                    .and_then(|l| l.strip_suffix(&suffix))
                else {
                    return line.to_string();
                };
                found = true;
                let mut names: Vec<&str> = names.split(", ").collect();
                if names.contains(&name) {
                    return line.to_string();
                }
                names.push(name);
                names.sort_unstable();
                changed = true;
                format!("import {{ {} }} from \"{module}\"", names.join(", "))
            })
            .map(|line| wrap_import(&line))
            .collect();
        if !found {
            return self.fail(format!(
                "{rel}: no import {{ ... }} from \"{module}\"; import `{name}` yourself"
            ));
        }
        if !changed {
            return self.record("identical", rel);
        }
        let mut text = lines.join("\n");
        text.push('\n');
        match fs::write(self.path(rel), text) {
            Ok(()) => self.record("insert", rel),
            Err(e) => self.fail(format!("{rel}: {e}")),
        }
    }

    /// `src/<dir>/mod.rs` (created with `doc` when missing) declared in `src/lib.rs`, the Rust
    /// equivalent of an autoloaded `app/<dir>`.
    pub fn module_dir(&mut self, dir: &str, doc: &str) {
        let rel = format!("src/{dir}/mod.rs");
        if !self.exists(&rel) {
            self.file(&rel, doc);
        }
        self.inject(
            "src/lib.rs",
            &format!("pub mod {dir};"),
            Anchor::Sorted("pub mod "),
        );
    }

    /// Adds `name = spec` to a Cargo.toml `[dependencies]` table, after its last entry.
    pub fn dependency(&mut self, rel: &str, name: &str, spec: &str) {
        let Some(existing) = self.read(rel) else {
            return self.fail(format!(
                "{rel}: file not found; add `{name} = {spec}` to [dependencies]"
            ));
        };
        let line = format!("{name} = {spec}");
        let lines: Vec<&str> = existing.lines().collect();
        let Some(start) = lines.iter().position(|l| l.trim() == "[dependencies]") else {
            return self.fail(format!("{rel}: no [dependencies] table; add `{line}`"));
        };
        let end = lines[start + 1..]
            .iter()
            .position(|l| l.trim_start().starts_with('['))
            .map_or(lines.len(), |i| start + 1 + i);
        if lines[start + 1..end]
            .iter()
            .any(|l| l.split('=').next().is_some_and(|k| k.trim() == name))
        {
            return self.record("identical", rel);
        }
        let last = lines[start + 1..end]
            .iter()
            .rposition(|l| !l.trim().is_empty())
            .map_or(start + 1, |i| start + 2 + i);
        let mut out: Vec<&str> = lines[..last].to_vec();
        out.push(&line);
        out.extend(&lines[last..]);
        let mut text = out.join("\n");
        text.push('\n');
        match fs::write(self.path(rel), text) {
            Ok(()) => self.record("insert", rel),
            Err(e) => self.fail(format!("{rel}: {e}")),
        }
    }
}

/// Prettier's print width: an `import { a, b } from "m"` longer than 80 characters gets one
/// name per line.
fn wrap_import(line: &str) -> String {
    let Some((names, module)) = line
        .strip_prefix("import { ")
        .and_then(|l| l.split_once(" } from "))
    else {
        return line.to_string();
    };
    if line.len() <= 80 {
        return line.to_string();
    }
    let list: Vec<String> = names.split(", ").map(|n| format!("  {n},\n")).collect();
    format!("import {{\n{}}} from {module}", list.concat())
}

/// `source`'s lines, with each prettier-wrapped `import {\n  a,\n  b,\n} from "m"` joined
/// back into one `import { a, b } from "m"` line.
fn join_wrapped_imports(source: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut open: Option<Vec<String>> = None;
    for line in source.lines() {
        match open.as_mut() {
            None if line == "import {" => open = Some(Vec::new()),
            None => out.push(line.to_string()),
            Some(names) => match line.strip_prefix("} from ") {
                Some(module) => {
                    out.push(format!("import {{ {} }} from {module}", names.join(", ")));
                    open = None;
                }
                None => names.push(line.trim().trim_end_matches(',').to_string()),
            },
        }
    }
    if let Some(names) = open {
        out.push("import {".to_string());
        out.extend(names.into_iter().map(|n| format!("  {n},")));
    }
    out
}

/// Scaffold's `sorted_line_index`: after the last smaller line, above the attributes and doc
/// comments that belong to the next one.
fn sorted_index(lines: &[&str], line: &str, prefix: &str) -> usize {
    let key = |l: &str| l.trim().trim_end_matches(';').to_string();
    let matching: Vec<usize> = (0..lines.len())
        .filter(|&i| lines[i].starts_with(prefix))
        .collect();
    let Some(&last) = matching.last() else {
        return lines.len();
    };
    let Some(mut at) = matching
        .iter()
        .copied()
        .find(|&i| key(line) < key(lines[i]))
    else {
        return last + 1;
    };
    while at > 0
        && (lines[at - 1].trim_start().starts_with("#[")
            || lines[at - 1].trim_start().starts_with("///"))
    {
        at -= 1;
    }
    at
}

/// Replaces each `{{key}}` in an embedded template. Keys never contain spaces, so JSX `{{ ... }}`
/// and Rust `{{}}` pass through untouched.
pub fn render(template: &str, vars: &[(&str, &str)]) -> String {
    let mut out = template.to_string();
    for (key, value) in vars {
        out = out.replace(&format!("{{{{{key}}}}}"), value);
    }
    out
}

/// ActiveSupport's `camelize` for a generator NAME: `weather_lookup`/`weather-lookup` ->
/// `WeatherLookup`; an already camel-cased name is kept.
pub fn camelize(name: &str) -> String {
    name.split(['_', '-', ' '])
        .filter(|w| !w.is_empty())
        .map(|w| {
            let mut chars = w.chars();
            chars
                .next()
                .map(|c| c.to_ascii_uppercase().to_string() + chars.as_str())
                .unwrap_or_default()
        })
        .collect()
}

/// ActiveSupport's `underscore`, acronym-aware like `Tool.tool_name` (`HTTPProxy` -> `http_proxy`).
pub fn underscore(name: &str) -> String {
    rust_llm::tool::underscore(name).replace('-', "_")
}

/// Rails' NamedBase NAME check. Namespaced names (`admin/weather`, `Admin::Weather`) are not
/// supported: Rust modules would need a `mod.rs` per level.
pub fn class_name(name: &str, suffix: &str) -> Result<String, String> {
    let ok = name.chars().next().is_some_and(|c| c.is_ascii_alphabetic())
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
    if !ok {
        return Err(format!(
            "invalid name {name:?}: use letters, digits, `_` or `-` (namespaces are not supported)"
        ));
    }
    let class = camelize(name);
    Ok(if class.ends_with(suffix) {
        class
    } else {
        format!("{class}{suffix}")
    })
}

pub const USAGE: &str = "Usage:
  rust-llm generate install [--path RUST_LLM_CHECKOUT] [--force]
  rust-llm generate tool NAME [--force]
  rust-llm generate agent NAME [--force]
  rust-llm generate schema NAME [--force]
  rust-llm generate chat_ui [--force]
  rust-llm generate public_chat [--force]
  rust-llm generate provider NAME [--dialect chat_completions|responses|anthropic|gemini|ollama]
                                  [--api-base URL] [--models-dev-provider KEY] [--dynamic-models]
                                  [--destination DIR] [--force]
  rust-llm generate upgrade [--force]

Run app generators from the root of a Loco + Inertia + React app.";

/// Parses `generate <generator> [NAME] [options]` and runs it in `cwd`. Returns the exit code.
pub fn run(args: &[String], cwd: &Path) -> i32 {
    // `CLI#run`: a bare invocation prints help and succeeds.
    if args.is_empty() {
        println!("{USAGE}");
        return 0;
    }
    let mut positional: Vec<&str> = Vec::new();
    let mut force = false;
    let mut dynamic_models = false;
    let mut options: Vec<(&str, &str)> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        match arg {
            "-h" | "--help" | "help" => {
                println!("{USAGE}");
                return 0;
            }
            "-f" | "--force" => force = true,
            "--dynamic-models" => dynamic_models = true,
            "--path" | "--dialect" | "--api-base" | "--destination" | "--models-dev-provider" => {
                let Some(value) = args.get(i + 1) else {
                    eprintln!("{arg} needs a value\n\n{USAGE}");
                    return 1;
                };
                options.push((arg, value));
                i += 1;
            }
            _ if arg.starts_with('-') => {
                eprintln!("Unknown option: {arg}\n\n{USAGE}");
                return 1;
            }
            _ => positional.push(arg),
        }
        i += 1;
    }
    let option = |name: &str| options.iter().find(|(k, _)| *k == name).map(|(_, v)| *v);
    let (command, generator, name) = (
        positional.first(),
        positional.get(1),
        positional.get(2).copied(),
    );
    if !matches!(command, Some(&"generate") | Some(&"g")) {
        eprintln!("{USAGE}");
        return 1;
    }
    // `parse_provider_options`: `Unexpected arguments: ...` for positionals past the NAME.
    let arity = match generator.copied() {
        Some("install" | "chat_ui" | "public_chat" | "upgrade") => 2,
        Some("tool" | "agent" | "schema" | "provider") => 3,
        _ => usize::MAX,
    };
    if positional.len() > arity {
        eprintln!("Unexpected arguments: {}", positional[arity..].join(" "));
        return 1;
    }
    let needs_name = |g: &str| name.ok_or_else(|| format!("`rust-llm generate {g}` needs a NAME"));
    let mut generator_run = Generator::new(cwd, force).echo();
    let result = match generator.copied() {
        Some("install") => install::generate(&mut generator_run, option("--path")),
        Some("tool") => needs_name("tool").and_then(|n| tool::generate(&mut generator_run, n)),
        Some("agent") => needs_name("agent").and_then(|n| agent::generate(&mut generator_run, n)),
        Some("schema") => {
            needs_name("schema").and_then(|n| schema::generate(&mut generator_run, n))
        }
        Some("chat_ui") => chat_ui::generate(&mut generator_run),
        Some("public_chat") => public_chat::generate(&mut generator_run),
        Some("provider") => needs_name("provider").and_then(|n| {
            let root = option("--destination").map_or_else(|| cwd.to_path_buf(), |d| cwd.join(d));
            generator_run = Generator::new(root, force).echo();
            let opts = provider::Options {
                dialect: option("--dialect"),
                api_base: option("--api-base"),
                models_dev_provider: option("--models-dev-provider"),
                dynamic_models,
            };
            provider::generate(&mut generator_run, n, &opts)
        }),
        Some("upgrade") => upgrade::generate(&mut generator_run),
        other => Err(format!(
            "Unknown generator: {}\n\n{USAGE}",
            other.unwrap_or("(none)")
        )),
    };
    if let Err(message) = result {
        eprintln!("{message}");
        return 1;
    }
    for line in &generator_run.notes {
        println!("{line}");
    }
    if generator_run.failures.is_empty() {
        0
    } else {
        1
    }
}
