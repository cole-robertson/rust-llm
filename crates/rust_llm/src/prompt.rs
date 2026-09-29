//! Port of `lib/ruby_llm/prompt.rb` and `lib/ruby_llm/prompt/context.rb`: prompt templates under
//! `app/prompts`, rendered with locals.
//!
//! RubyLLM renders ERB (`name.txt.erb`). ERB evaluates Ruby, so RustLLM uses Jinja through
//! [minijinja](https://docs.rs/minijinja) instead, with files named `name.txt.jinja`. The
//! behavior is RubyLLM's: the same lookup roots, partials named `_name`, bare partial names
//! resolved next to the current template, path names from the roots, no locals leaking into
//! partials, and an error for an unknown variable. Rewrite a template like this:
//!
//! | ERB (`.txt.erb`) | Jinja (`.txt.jinja`) |
//! |---|---|
//! | `<%= name %>` | `{{ name }}` |
//! | `<% if admin %>...<% end %>` | `{% if admin %}...{% endif %}` |
//! | `<% items.each do \|i\| %>...<% end %>` | `{% for i in items %}...{% endfor %}` |
//! | `<%= render "tone", name: name %>` | `{{ render("tone", name=name) }}` |
//! | `<%= render partial: "shared/safety", locals: { name: name } %>` | `{{ render(partial="shared/safety", locals={"name": name}) }}` |
//! | `<%= local_assigns[:name] \|\| "friend" %>` | `{{ local_assigns.name \| default("friend") }}` |
//! | `<%= local_assigns["x-y"] %>` | `{{ local_assigns["x-y"] }}` |
//!
//! Output is not HTML-escaped, like ERB's. A name that is not a valid identifier (`x-y`, `Name`)
//! reaches the template only through `local_assigns`, like Ruby's local-variable rule.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use minijinja::value::{Kwargs, Rest};
use minijinja::{AutoEscape, Environment, ErrorKind, UndefinedBehavior, Value as Jinja};
use serde_json::{Map, Value};

use crate::config::Config;
use crate::error::{Error, Result};

/// The template extension (Ruby's `.txt.erb`).
pub const EXTENSION: &str = ".txt.jinja";

/// `Prompt.root`: `app/prompts` under the working directory.
pub fn root() -> PathBuf {
    std::env::current_dir()
        .unwrap_or_default()
        .join("app/prompts")
}

/// `Prompt.roots`: the application root first, then `config.prompt_roots` in order. An
/// application overrides an engine's prompt by shipping a file at the same relative path.
pub fn roots(config: &Config) -> Vec<PathBuf> {
    let app = config
        .get("prompt_root")
        .map(PathBuf::from)
        .unwrap_or_else(root);
    std::iter::once(app)
        .chain(config.prompt_roots.iter().cloned())
        .collect()
}

/// `RubyLLM::Prompt`: one named template.
#[derive(Debug, Clone)]
pub struct Prompt {
    name: String,
    filename: String,
    config: Arc<Config>,
}

impl Prompt {
    /// `Prompt.new(name)`: `name` without the extension, like `"support/instructions"`.
    pub fn new(name: impl Into<String>) -> Prompt {
        Prompt::with_config(crate::config(), name)
    }

    pub fn with_config(config: Arc<Config>, name: impl Into<String>) -> Prompt {
        let name = name.into();
        let filename = if name.ends_with(EXTENSION) {
            name.clone()
        } else {
            format!("{name}{EXTENSION}")
        };
        Prompt {
            name,
            filename,
            config,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// `Prompt#path`: the first root that has the file, else where the application root would.
    pub fn path(&self) -> PathBuf {
        let candidates: Vec<PathBuf> = roots(&self.config)
            .iter()
            .map(|r| r.join(&self.filename))
            .collect();
        candidates
            .iter()
            .find(|c| c.exists())
            .cloned()
            .unwrap_or_else(|| candidates[0].clone())
    }

    pub fn exists(&self) -> bool {
        self.path().exists()
    }

    /// `Prompt#render(**locals)`: `locals` is a JSON object (or `Null` for none).
    pub fn render(&self, locals: Value) -> Result<String> {
        render_file(&self.config, &self.name, self.path(), locals_map(locals))
    }
}

/// `RubyLLM.render_prompt(name, **locals)`: renders `app/prompts/<name>.txt.jinja` with `locals`
/// (a JSON object). Fails with `Error::PromptNotFound` when no root has the file.
///
/// ```ignore
/// let instructions = rust_llm::render_prompt("support/instructions", json!({ "product_name": "BillingHub" }))?;
/// chat.with_instructions(instructions);
/// ```
pub fn render_prompt(name: &str, locals: Value) -> Result<String> {
    Prompt::new(name).render(locals)
}

fn locals_map(locals: Value) -> Map<String, Value> {
    match locals {
        Value::Object(m) => m,
        _ => Map::new(),
    }
}

/// Ruby's `LOCAL_VARIABLE_NAME`: lowercase or `_` first, then word characters.
fn is_local_name(name: &str) -> bool {
    let mut chars = name.chars();
    chars
        .next()
        .is_some_and(|c| c == '_' || c.is_lowercase() || !c.is_ascii())
        && name.chars().all(|c| c == '_' || c.is_alphanumeric())
}

fn render_file(
    config: &Arc<Config>,
    name: &str,
    path: PathBuf,
    locals: Map<String, Value>,
) -> Result<String> {
    let source = std::fs::read_to_string(&path).map_err(|e| match e.kind() {
        std::io::ErrorKind::NotFound => {
            Error::PromptNotFound(format!("Prompt file not found: {}", path.display()))
        }
        _ => Error::Io(e),
    })?;
    let mut env = Environment::new();
    env.set_undefined_behavior(UndefinedBehavior::Strict);
    env.set_keep_trailing_newline(true);
    env.set_auto_escape_callback(|_| AutoEscape::None);
    let (config_for_render, current) = (config.clone(), name.to_string());
    env.add_function(
        "render",
        move |args: Rest<Jinja>, kwargs: Kwargs| -> std::result::Result<String, minijinja::Error> {
            let (partial, locals) = match args.first() {
                Some(first) => {
                    let mut locals = Map::new();
                    for key in kwargs.args() {
                        let v: Jinja = kwargs.get(key)?;
                        locals.insert(key.to_string(), to_json(&v)?);
                    }
                    (first.to_string(), locals)
                }
                None => {
                    let partial: String = kwargs.get("partial")?;
                    let locals: Option<Jinja> = kwargs.get("locals")?;
                    (
                        partial,
                        locals
                            .map(|v| to_json(&v))
                            .transpose()?
                            .map(locals_map)
                            .unwrap_or_default(),
                    )
                }
            };
            let name = partial_name(&current, &partial);
            let path = Prompt::with_config(config_for_render.clone(), &name).path();
            render_file(&config_for_render, &name, path, locals)
                .map_err(|e| minijinja::Error::new(ErrorKind::InvalidOperation, e.to_string()))
        },
    );
    let template = env
        .template_from_str(&source)
        .map_err(|e| Error::Prompt(describe(&e)))?;
    let mut context: Map<String, Value> = locals
        .iter()
        .filter(|(k, _)| is_local_name(k))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    context.insert("local_assigns".into(), Value::Object(locals));
    template
        .render(Jinja::from_serialize(&context))
        .map_err(|e| match e.kind() {
            ErrorKind::InvalidOperation
                if e.detail()
                    .is_some_and(|d| d.starts_with("Prompt file not found")) =>
            {
                Error::PromptNotFound(e.detail().unwrap_or_default().to_string())
            }
            _ => Error::Prompt(describe(&e)),
        })
}

fn to_json(v: &Jinja) -> std::result::Result<Value, minijinja::Error> {
    serde_json::to_value(v)
        .map_err(|e| minijinja::Error::new(ErrorKind::InvalidOperation, e.to_string()))
}

fn describe(e: &minijinja::Error) -> String {
    let mut message = e.to_string();
    if e.kind() == ErrorKind::UndefinedError {
        message = format!("undefined local variable in prompt: {message}");
    }
    message
}

/// `Context#partial`: `_name` next to the current prompt for a bare name, from the roots for a
/// path.
fn partial_name(current: &str, name: &str) -> String {
    let partial = match name.rsplit_once('/') {
        Some((dir, file)) => format!("{dir}/_{file}"),
        None => format!("_{name}"),
    };
    let directory = Path::new(current)
        .parent()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_default();
    if name.contains('/') || directory.is_empty() {
        partial
    } else {
        format!("{directory}/{partial}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partials_resolve_next_to_the_prompt_or_from_the_root() {
        assert_eq!(
            partial_name("work_assistant/instructions", "tone"),
            "work_assistant/_tone"
        );
        assert_eq!(
            partial_name("work_assistant/instructions", "shared/safety"),
            "shared/_safety"
        );
        assert_eq!(partial_name("instructions", "tone"), "_tone");
    }

    #[test]
    fn only_ruby_local_names_become_variables() {
        assert!(is_local_name("name"));
        assert!(is_local_name("_x1"));
        assert!(!is_local_name("x-y"));
        assert!(!is_local_name("Name"));
        assert!(!is_local_name("1x"));
    }
}
