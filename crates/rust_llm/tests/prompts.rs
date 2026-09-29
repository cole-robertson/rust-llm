//! RubyLLM 2.0's `prompt_spec.rb` and the prompt parts of `agent_instructions_spec.rb`, with the
//! ERB templates rewritten in Jinja (see `rust_llm::prompt` for the mapping).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use rust_llm::prompt::Prompt;
use rust_llm::{Agent, Chat, Config, Error, Role};
use serde_json::{Value, json};

struct Roots {
    app: PathBuf,
    config: Arc<Config>,
    _dirs: Vec<TempDir>,
}

/// A directory removed when dropped.
struct TempDir(PathBuf);

impl TempDir {
    fn new() -> TempDir {
        let dir = std::env::temp_dir().join(format!("rust_llm_prompts_{}", uuid_like()));
        std::fs::create_dir_all(&dir).unwrap();
        TempDir(dir)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn uuid_like() -> String {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    format!(
        "{}_{}",
        std::process::id(),
        N.fetch_add(1, Ordering::SeqCst)
    )
}

fn write(root: &Path, name: &str, content: &str) {
    let path = root.join(format!("{name}.txt.jinja"));
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, content).unwrap();
}

/// An application root, plus engine roots appended with `config.prompt_roots`.
fn roots(engines: usize) -> (Roots, Vec<PathBuf>) {
    let app_dir = TempDir::new();
    let app = app_dir.0.join("app/prompts");
    std::fs::create_dir_all(&app).unwrap();
    let mut dirs = vec![app_dir];
    let mut engine_roots = Vec::new();
    for _ in 0..engines {
        let d = TempDir::new();
        let root = d.0.join("app/prompts");
        std::fs::create_dir_all(&root).unwrap();
        engine_roots.push(root);
        dirs.push(d);
    }
    let mut config = Config::default();
    config.set("prompt_root", app.to_string_lossy().to_string());
    config.prompt_roots = engine_roots.clone();
    (
        Roots {
            app,
            config: Arc::new(config),
            _dirs: dirs,
        },
        engine_roots,
    )
}

impl Roots {
    fn render(&self, name: &str, locals: Value) -> rust_llm::Result<String> {
        Prompt::with_config(self.config.clone(), name).render(locals)
    }
}

fn setup() -> Roots {
    roots(0).0
}

// ---- .render ---------------------------------------------------------------------------------

#[test]
fn renders_a_prompt_with_locals() {
    let r = setup();
    write(&r.app, "friend", "Hello, {{ name }}!");
    assert_eq!(
        r.render("friend", json!({ "name": "Andrey" })).unwrap(),
        "Hello, Andrey!"
    );
}

#[test]
fn renders_a_nested_prompt_path() {
    let r = setup();
    write(
        &r.app,
        "work_assistant/instructions",
        "You assist {{ user }}.",
    );
    assert_eq!(
        r.render("work_assistant/instructions", json!({ "user": "Bob" }))
            .unwrap(),
        "You assist Bob."
    );
}

#[test]
fn renders_without_locals() {
    let r = setup();
    write(&r.app, "simple", "Just a static prompt.");
    assert_eq!(
        r.render("simple", Value::Null).unwrap(),
        "Just a static prompt."
    );
}

#[test]
fn raises_prompt_not_found_for_missing_prompts() {
    let r = setup();
    assert!(matches!(
        r.render("nonexistent", Value::Null),
        Err(Error::PromptNotFound(_))
    ));
}

// ---- #render / #path -------------------------------------------------------------------------

#[test]
fn exposes_name_and_path() {
    let r = setup();
    let prompt = Prompt::with_config(r.config.clone(), "greeting");
    assert_eq!(prompt.name(), "greeting");
    assert_eq!(prompt.path(), r.app.join("greeting.txt.jinja"));
}

// ---- partials ------------------------------------------------------------------------------------

#[test]
fn renders_a_partial_from_the_current_prompt_directory() {
    let r = setup();
    write(&r.app, "work_assistant/_tone", "Be kind to {{ name }}.");
    write(
        &r.app,
        "work_assistant/instructions",
        "Hello.\n{{ render(\"tone\", name=name) }}",
    );
    assert_eq!(
        r.render("work_assistant/instructions", json!({ "name": "Ada" }))
            .unwrap(),
        "Hello.\nBe kind to Ada."
    );
}

#[test]
fn does_not_fall_back_to_the_prompt_root_for_a_bare_name() {
    let r = setup();
    write(&r.app, "_tone", "Root tone.");
    write(
        &r.app,
        "work_assistant/instructions",
        "{{ render(\"tone\") }}",
    );
    let err = r
        .render("work_assistant/instructions", Value::Null)
        .unwrap_err();
    assert!(
        matches!(&err, Error::PromptNotFound(m) if m.contains("work_assistant/_tone.txt.jinja")),
        "{err}"
    );
}

#[test]
fn resolves_a_bare_name_from_the_root_for_a_top_level_prompt() {
    let r = setup();
    write(&r.app, "_tone", "Root tone.");
    write(&r.app, "instructions", "{{ render(\"tone\") }}");
    assert_eq!(r.render("instructions", Value::Null).unwrap(), "Root tone.");
}

#[test]
fn resolves_a_path_name_from_the_roots_not_the_current_directory() {
    let r = setup();
    write(&r.app, "shared/_safety", "Root safety.");
    write(&r.app, "work_assistant/shared/_safety", "Nested safety.");
    write(
        &r.app,
        "work_assistant/instructions",
        "{{ render(\"shared/safety\") }}",
    );
    assert_eq!(
        r.render("work_assistant/instructions", Value::Null)
            .unwrap(),
        "Root safety."
    );
}

#[test]
fn renders_a_partial_with_the_hash_form() {
    let r = setup();
    write(&r.app, "work_assistant/_tone", "Be kind to {{ name }}.");
    write(
        &r.app,
        "work_assistant/instructions",
        "{{ render(partial=\"tone\", locals={\"name\": name}) }}",
    );
    assert_eq!(
        r.render("work_assistant/instructions", json!({ "name": "Ada" }))
            .unwrap(),
        "Be kind to Ada."
    );
}

#[test]
fn renders_a_partial_with_the_hash_form_and_no_locals() {
    let r = setup();
    write(&r.app, "shared/_safety", "Stay safe.");
    write(
        &r.app,
        "instructions",
        "{{ render(partial=\"shared/safety\") }}",
    );
    assert_eq!(r.render("instructions", Value::Null).unwrap(), "Stay safe.");
}

#[test]
fn exposes_local_assigns_in_a_partial() {
    let r = setup();
    write(
        &r.app,
        "_tone",
        "{{ local_assigns.name | default(\"friend\") }}",
    );
    write(
        &r.app,
        "instructions",
        "{{ render(\"tone\") }} {{ render(\"tone\", name=\"Ada\") }}",
    );
    assert_eq!(r.render("instructions", Value::Null).unwrap(), "friend Ada");
}

#[test]
fn exposes_local_assigns_in_a_prompt() {
    let r = setup();
    write(
        &r.app,
        "instructions",
        "{{ local_assigns.name | default(\"friend\") }}",
    );
    assert_eq!(r.render("instructions", Value::Null).unwrap(), "friend");
    assert_eq!(
        r.render("instructions", json!({ "name": "Ada" })).unwrap(),
        "Ada"
    );
}

#[test]
fn keeps_a_local_with_an_invalid_variable_name_in_local_assigns_only() {
    let r = setup();
    write(
        &r.app,
        "instructions",
        "{{ local_assigns[\"x-y\"] }}{{ local_assigns.Name }}",
    );
    assert_eq!(
        r.render("instructions", json!({ "x-y": 1, "Name": 2 }))
            .unwrap(),
        "12"
    );
    write(&r.app, "direct", "{{ Name }}");
    assert!(matches!(
        r.render("direct", json!({ "Name": 2 })),
        Err(Error::Prompt(_))
    ));
}

#[test]
fn treats_null_locals_in_the_hash_form_as_no_locals() {
    let r = setup();
    write(&r.app, "_tone", "Stay calm.");
    write(
        &r.app,
        "instructions",
        "{{ render(partial=\"tone\", locals=none) }}",
    );
    assert_eq!(r.render("instructions", Value::Null).unwrap(), "Stay calm.");
}

#[test]
fn renders_nested_partials() {
    let r = setup();
    write(&r.app, "_inner", "inner");
    write(&r.app, "_outer", "outer {{ render(\"inner\") }}");
    write(&r.app, "instructions", "{{ render(\"outer\") }}");
    assert_eq!(
        r.render("instructions", Value::Null).unwrap(),
        "outer inner"
    );
}

#[test]
fn resolves_a_bare_name_next_to_the_partial_that_renders_it() {
    let r = setup();
    write(&r.app, "shared/_inner", "inner");
    write(&r.app, "shared/_outer", "outer {{ render(\"inner\") }}");
    write(&r.app, "instructions", "{{ render(\"shared/outer\") }}");
    assert_eq!(
        r.render("instructions", Value::Null).unwrap(),
        "outer inner"
    );
}

#[test]
fn does_not_leak_locals_into_a_partial() {
    let r = setup();
    write(&r.app, "_tone", "{{ name }}");
    write(&r.app, "instructions", "{{ render(\"tone\") }}");
    let err = r
        .render("instructions", json!({ "name": "Ada" }))
        .unwrap_err();
    assert!(err.to_string().contains("undefined"), "{err}");
}

#[test]
fn raises_prompt_not_found_for_a_missing_partial() {
    let r = setup();
    write(
        &r.app,
        "work_assistant/instructions",
        "{{ render(\"tone\") }}",
    );
    let err = r
        .render("work_assistant/instructions", Value::Null)
        .unwrap_err();
    assert!(
        matches!(&err, Error::PromptNotFound(m) if m.contains("work_assistant/_tone.txt.jinja")),
        "{err}"
    );
}

#[test]
fn reports_the_root_path_for_a_missing_path_partial() {
    let r = setup();
    write(
        &r.app,
        "work_assistant/instructions",
        "{{ render(\"shared/safety\") }}",
    );
    let err = r
        .render("work_assistant/instructions", Value::Null)
        .unwrap_err();
    assert!(
        matches!(&err, Error::PromptNotFound(m) if m.contains("prompts/shared/_safety.txt.jinja")),
        "{err}"
    );
}

// ---- .roots ------------------------------------------------------------------------------------

#[test]
fn keeps_the_application_root_first() {
    let (r, engines) = roots(1);
    assert_eq!(
        rust_llm::prompt::roots(&r.config),
        vec![r.app.clone(), engines[0].clone()]
    );
}

#[test]
fn resolves_a_prompt_from_an_engine_root_the_application_does_not_ship() {
    let (r, engines) = roots(1);
    write(
        &engines[0],
        "engine_agent/instructions",
        "Engine prompt for {{ name }}.",
    );
    assert_eq!(
        r.render("engine_agent/instructions", json!({ "name": "Ava" }))
            .unwrap(),
        "Engine prompt for Ava."
    );
}

#[test]
fn prefers_the_application_prompt_over_an_engine_prompt() {
    let (r, engines) = roots(1);
    write(&r.app, "engine_agent/instructions", "Application override.");
    write(&engines[0], "engine_agent/instructions", "Engine default.");
    assert_eq!(
        r.render("engine_agent/instructions", Value::Null).unwrap(),
        "Application override."
    );
}

#[test]
fn resolves_path_to_the_engine_file_when_only_the_engine_ships_it() {
    let (r, engines) = roots(1);
    write(&engines[0], "engine_agent/instructions", "Engine default.");
    assert_eq!(
        Prompt::with_config(r.config.clone(), "engine_agent/instructions").path(),
        engines[0].join("engine_agent/instructions.txt.jinja")
    );
}

#[test]
fn renders_a_partial_shipped_by_an_engine() {
    let (r, engines) = roots(1);
    write(&engines[0], "engine_agent/_tone", "Engine tone.");
    write(
        &engines[0],
        "engine_agent/instructions",
        "{{ render(\"tone\") }}",
    );
    assert_eq!(
        r.render("engine_agent/instructions", Value::Null).unwrap(),
        "Engine tone."
    );
}

#[test]
fn falls_back_to_the_application_path_when_no_root_has_the_file() {
    let (r, _) = roots(1);
    let prompt = Prompt::with_config(r.config.clone(), "missing");
    assert_eq!(prompt.path(), r.app.join("missing.txt.jinja"));
    assert!(
        matches!(prompt.render(Value::Null), Err(Error::PromptNotFound(m)) if m.contains("missing.txt.jinja"))
    );
}

// ---- Agent instructions from prompts (agent_instructions_spec.rb) ------------------------------

struct WorkAssistant;
impl Agent for WorkAssistant {
    fn prompt_locals(&self) -> Value {
        json!({ "display_name": "Ava" })
    }
}

struct Declared;
impl Agent for Declared {
    fn instructions(&self) -> Option<String> {
        Some("Inline instructions".into())
    }
}

fn chat_with(config: &Arc<Config>) -> Chat {
    let mut c = (**config).clone();
    c.set("openai_api_key", "test");
    Chat::with_config(Arc::new(c), Some("gpt-5-nano"), Some("openai"), false).unwrap()
}

fn system_messages(chat: &Chat) -> Vec<String> {
    chat.messages()
        .iter()
        .filter(|m| m.role == Role::System)
        .map(|m| m.content().to_string())
        .collect()
}

#[test]
fn an_agent_uses_its_conventional_prompt_with_locals() {
    let r = setup();
    write(
        &r.app,
        "work_assistant/instructions",
        "Child prompt for {{ display_name }}",
    );
    let chat = WorkAssistant.apply(chat_with(&r.config)).unwrap();
    assert_eq!(system_messages(&chat), ["Child prompt for Ava"]);
}

#[test]
fn inline_declarations_win_over_the_prompt() {
    let r = setup();
    write(&r.app, "declared/instructions", "Prompt");
    let chat = Declared.apply(chat_with(&r.config)).unwrap();
    assert_eq!(system_messages(&chat), ["Inline instructions"]);
}

#[test]
fn an_empty_prompt_means_no_instructions() {
    let r = setup();
    write(&r.app, "work_assistant/instructions", "");
    let chat = WorkAssistant.apply(chat_with(&r.config)).unwrap();
    assert!(system_messages(&chat).is_empty());
}

#[test]
fn an_agent_without_a_prompt_file_has_no_instructions() {
    let r = setup();
    let chat = WorkAssistant.apply(chat_with(&r.config)).unwrap();
    assert!(system_messages(&chat).is_empty());
}

#[test]
fn agent_prompt_paths_underscore_the_type_name() {
    assert_eq!(
        rust_llm::agent::prompt_agent_path("WorkAssistant"),
        "work_assistant"
    );
    assert_eq!(
        rust_llm::agent::prompt_agent_path("Admin::HTTPHelper"),
        "admin/http_helper"
    );
    assert_eq!(WorkAssistant.name(), "WorkAssistant");
}
