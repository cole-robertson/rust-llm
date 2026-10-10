//! RubyLLM 2.0's `agent_spec.rb`, `agent_dsl_spec.rb` and `agent_instructions_spec.rb` examples
//! that have a Rust counterpart: class macros are `Agent` trait methods, declared inputs are the
//! agent's fields, and ERB prompts are Jinja (see `rust_llm::prompt`).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use rust_llm::agent::{InstructionDeclaration, prompt_agent_path};
use rust_llm::protocols::Caching;
use rust_llm::{Agent, Chat, Config, Context, Error, Role, ThinkingConfig};
use serde_json::{Map, Value, json};

/// `include_context 'with configured RubyLLM'`: provider keys only. Every test sets the same
/// values, so running them in parallel is safe; prompt roots go through a per-test `Context`.
fn keys() {
    rust_llm::configure(|c| {
        c.set("openai_api_key", "test");
        c.set("anthropic_api_key", "test");
    });
}

/// `model_for(:openai, :temperature)`.
const OPENAI: &str = "gpt-4.1-nano";
/// `model_for(:anthropic, :adaptive_thinking)`.
const ADAPTIVE: &str = "claude-sonnet-5";

fn system_messages(chat: &Chat) -> Vec<String> {
    chat.messages()
        .iter()
        .filter(|m| m.role == Role::System)
        .map(|m| m.content().to_string())
        .collect()
}

// ---- caching, thinking, citations, compaction -------------------------------------------------

/// `caching` with no options.
struct DefaultCaching;
impl Agent for DefaultCaching {
    fn model(&self) -> Option<&str> {
        Some(OPENAI)
    }
    fn provider(&self) -> Option<&str> {
        Some("openai")
    }
    fn caching(&self) -> Option<Value> {
        Some(Value::Bool(true))
    }
}

// spec: agent_dsl_spec.rb:67 enables provider-default caching without options
// spec: agent_spec.rb:122 lets agents enable provider-default prompt caching
#[test]
fn caching_without_options_enables_the_provider_default() {
    keys();
    let chat = DefaultCaching.chat().unwrap();
    assert_eq!(chat.caching(), Some(&Caching::On(Map::new())));
}

struct Uncached;
impl Agent for Uncached {
    fn model(&self) -> Option<&str> {
        Some(OPENAI)
    }
}

// spec: agent_spec.rb:131 does not enable prompt caching unless configured
#[test]
fn caching_stays_unset_unless_declared() {
    keys();
    assert_eq!(Uncached.chat().unwrap().caching(), None);
}

/// `caching { { retention: '24h' } }`.
struct RetainedCaching;
impl Agent for RetainedCaching {
    fn model(&self) -> Option<&str> {
        Some(OPENAI)
    }
    fn caching(&self) -> Option<Value> {
        Some(json!({ "retention": "24h" }))
    }
}

// spec: agent_spec.rb:139 lets agent instances disable prompt caching
#[test]
fn an_agent_chat_can_disable_declared_caching() {
    keys();
    let chat = RetainedCaching.chat().unwrap();
    assert_eq!(
        chat.caching(),
        Some(&Caching::On(
            json!({ "retention": "24h" }).as_object().unwrap().clone()
        ))
    );
    // `agent.with_caching(false)`: an agent instance delegates it to its chat.
    let chat = chat.with_caching(Value::Bool(false)).unwrap();
    assert_eq!(chat.caching(), Some(&Caching::Off));
}

/// `thinking` / `citations` / `compaction`, all without options.
struct FeatureDefaults;
impl Agent for FeatureDefaults {
    fn model(&self) -> Option<&str> {
        Some(ADAPTIVE)
    }
    fn provider(&self) -> Option<&str> {
        Some("anthropic")
    }
    fn thinking(&self) -> Option<ThinkingConfig> {
        Some(ThinkingConfig::on())
    }
    fn citations(&self) -> Option<bool> {
        Some(true)
    }
    fn compaction(&self) -> Option<Value> {
        Some(Value::Bool(true))
    }
}

// spec: agent_dsl_spec.rb:76 enables feature defaults without options
#[test]
fn features_without_options_enable_their_defaults() {
    keys();
    let chat = FeatureDefaults.chat().unwrap();
    assert_eq!(
        chat.render().unwrap()["thinking"],
        json!({ "type": "adaptive" })
    );
    assert!(chat.citations());
    assert_eq!(chat.compaction(), Some(&json!({})));
}

/// `thinking false` / `caching false` / `compaction false` / `citations false`.
struct FeaturesOff;
impl Agent for FeaturesOff {
    fn model(&self) -> Option<&str> {
        Some(ADAPTIVE)
    }
    fn provider(&self) -> Option<&str> {
        Some("anthropic")
    }
    fn thinking(&self) -> Option<ThinkingConfig> {
        Some(ThinkingConfig::off())
    }
    fn caching(&self) -> Option<Value> {
        Some(Value::Bool(false))
    }
    fn compaction(&self) -> Option<Value> {
        Some(Value::Bool(false))
    }
    fn citations(&self) -> Option<bool> {
        Some(false)
    }
}

// spec: agent_dsl_spec.rb:91 disables features with false
#[test]
fn features_declared_false_are_disabled() {
    keys();
    let chat = FeaturesOff.chat().unwrap();
    assert_eq!(
        chat.render().unwrap()["thinking"],
        json!({ "type": "disabled" })
    );
    assert_eq!(chat.caching(), Some(&Caching::Off));
    assert_eq!(chat.compaction(), Some(&Value::Bool(false)));
    assert!(!chat.citations());
}

// ---- context and end_user ---------------------------------------------------------------------

struct InContext(Context);
impl Agent for InContext {
    fn model(&self) -> Option<&str> {
        Some(OPENAI)
    }
    fn provider(&self) -> Option<&str> {
        Some("openai")
    }
    fn context(&self) -> Option<Context> {
        Some(self.0.clone())
    }
}

// spec: agent_dsl_spec.rb:122 binds a configured context to the chat it builds
#[test]
fn an_agent_builds_its_chat_with_the_declared_context() {
    keys();
    let context = rust_llm::context(|c| c.request_timeout = Duration::from_secs(42));
    let agent = InContext(context.clone());
    assert!(Arc::ptr_eq(
        agent.context().unwrap().config(),
        context.config()
    ));
    let chat = agent.chat().unwrap();
    assert_eq!(chat.config().request_timeout, Duration::from_secs(42));
    assert!(Arc::ptr_eq(chat.config(), context.config()));
    assert_ne!(rust_llm::config().request_timeout, Duration::from_secs(42));
}

#[test]
fn applying_an_agent_rebinds_a_chat_to_its_context() {
    keys();
    let context = rust_llm::context(|c| c.request_timeout = Duration::from_secs(42));
    let chat = Chat::new(Some(OPENAI), Some("openai")).unwrap();
    let chat = InContext(context.clone()).apply(chat).unwrap();
    assert!(Arc::ptr_eq(chat.config(), context.config()));
}

/// `inputs :tenant` + `end_user { "tenant-#{tenant}" }`: inputs are the agent's fields.
struct Tenanted {
    tenant: String,
}
impl Agent for Tenanted {
    fn model(&self) -> Option<&str> {
        Some(OPENAI)
    }
    fn provider(&self) -> Option<&str> {
        Some("openai")
    }
    fn end_user(&self) -> Option<String> {
        Some(format!("tenant-{}", self.tenant))
    }
}

// spec: agent_dsl_spec.rb:292 resolves the safety identifier from the agent inputs
#[test]
fn the_end_user_comes_from_the_agent_inputs() {
    keys();
    let chat = Tenanted {
        tenant: "acme".into(),
    }
    .chat()
    .unwrap();
    assert_eq!(chat.end_user(), Some("tenant-acme"));
}

struct Assumed;
impl Agent for Assumed {
    fn model(&self) -> Option<&str> {
        Some("my-private-finetune")
    }
    fn provider(&self) -> Option<&str> {
        Some("openai")
    }
    fn assume_model_exists(&self) -> bool {
        true
    }
}

#[test]
fn assume_model_exists_builds_a_chat_for_an_unregistered_model() {
    keys();
    assert_eq!(Assumed.chat().unwrap().model().id, "my-private-finetune");
}

// ---- prompts ----------------------------------------------------------------------------------

/// A directory removed when dropped.
struct TempDir(PathBuf);

impl TempDir {
    fn new() -> TempDir {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static N: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "rust_llm_spec_agent_{}_{}",
            std::process::id(),
            N.fetch_add(1, Ordering::SeqCst)
        ));
        std::fs::create_dir_all(dir.join("app/prompts")).unwrap();
        TempDir(dir)
    }

    fn prompts(&self) -> PathBuf {
        self.0.join("app/prompts")
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn write(root: &Path, name: &str, content: &str) {
    let path = root.join(format!("{name}.txt.jinja"));
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, content).unwrap();
}

/// `with_prompt_root`: `Prompt.root` stubbed to a temp directory, plus engine `roots`.
fn prompt_context(app: &TempDir, engines: &[&TempDir]) -> Context {
    rust_llm::context(|c| {
        c.set("prompt_root", app.prompts().to_string_lossy().to_string());
        c.prompt_roots = engines.iter().map(|e| e.prompts()).collect();
    })
}

/// A named agent with no `instructions` macro, built in `Context`.
macro_rules! prompt_agent {
    ($name:ident) => {
        struct $name(Context);
        impl Agent for $name {
            fn model(&self) -> Option<&str> {
                Some(OPENAI)
            }
            fn context(&self) -> Option<Context> {
                Some(self.0.clone())
            }
        }
    };
}

prompt_agent!(SpecDefaultPromptMissingAgent);

// spec: agent_spec.rb:159 starts without instructions when the default prompt is missing
#[test]
fn an_agent_without_a_prompt_file_starts_without_instructions() {
    keys();
    let app = TempDir::new();
    let chat = SpecDefaultPromptMissingAgent(prompt_context(&app, &[]))
        .chat()
        .unwrap();
    assert!(chat.messages().is_empty());
}

/// `instructions { prompt('instructions') }`.
struct ExplicitPrompt;
impl Agent for ExplicitPrompt {
    fn model(&self) -> Option<&str> {
        Some(OPENAI)
    }
    fn instruction_declarations(
        &self,
        _config: &Arc<Config>,
        _chat: &Value,
    ) -> rust_llm::Result<Vec<InstructionDeclaration>> {
        Ok(vec![InstructionDeclaration::new(
            self.render_prompt("instructions", json!({}))?,
        )])
    }
}

// spec: agent_spec.rb:167 raises when an explicitly referenced prompt is missing
#[test]
fn an_explicitly_referenced_missing_prompt_is_an_error() {
    keys();
    match ExplicitPrompt.chat() {
        Err(Error::PromptNotFound(message)) => {
            assert!(message.contains("Prompt file not found"), "{message}");
            assert!(
                message.contains("explicit_prompt/instructions.txt.jinja"),
                "{message}"
            );
        }
        other => panic!("expected PromptNotFound, got {:?}", other.map(|_| ())),
    }
}

prompt_agent!(SpecImplicitPromptAgent);

// spec: agent_spec.rb:176 loads conventional instructions prompt automatically for named agents
#[test]
fn a_named_agent_loads_its_conventional_prompt() {
    keys();
    let app = TempDir::new();
    // Ruby renders `chat.class.name`; a plain chat reaches Jinja as `chat`, which is none.
    write(
        &app.prompts(),
        "spec_implicit_prompt_agent/instructions",
        "Hello from {% if chat is none %}a plain chat{% endif %}",
    );
    let chat = SpecImplicitPromptAgent(prompt_context(&app, &[]))
        .chat()
        .unwrap();
    assert_eq!(chat.messages()[0].role, Role::System);
    assert_eq!(chat.messages()[0].content(), "Hello from a plain chat");
}

prompt_agent!(SpecEnginePromptAgent);

// spec: agent_spec.rb:194 loads the conventional instructions prompt from a registered engine root
#[test]
fn a_named_agent_loads_its_prompt_from_an_engine_root() {
    keys();
    let (app, engine) = (TempDir::new(), TempDir::new());
    write(
        &engine.prompts(),
        "spec_engine_prompt_agent/instructions",
        "Shipped by the engine",
    );
    let chat = SpecEnginePromptAgent(prompt_context(&app, &[&engine]))
        .chat()
        .unwrap();
    assert_eq!(chat.messages()[0].role, Role::System);
    assert_eq!(chat.messages()[0].content(), "Shipped by the engine");
}

/// `inputs :display_name` + `instructions display_name: 'Bea'`, built with `display_name: 'Ava'`.
struct SpecInheritedAgent {
    context: Context,
    display_name: String,
}
impl Agent for SpecInheritedAgent {
    fn model(&self) -> Option<&str> {
        Some(OPENAI)
    }
    fn context(&self) -> Option<Context> {
        Some(self.context.clone())
    }
    /// `resolve_prompt_locals`: the inputs, with the explicitly declared locals merged over them.
    fn prompt_locals(&self) -> Value {
        let mut locals = json!({ "display_name": self.display_name });
        locals["display_name"] = json!("Bea");
        locals
    }
}

// spec: agent_instructions_spec.rb:62 renders the child prompt with explicitly declared locals
#[test]
fn the_conventional_prompt_renders_with_declared_locals() {
    keys();
    let app = TempDir::new();
    write(
        &app.prompts(),
        "spec_inherited_agent/instructions",
        "Child prompt for {{ display_name }}",
    );
    let chat = SpecInheritedAgent {
        context: prompt_context(&app, &[]),
        display_name: "Ava".into(),
    }
    .chat()
    .unwrap();
    assert_eq!(system_messages(&chat), ["Child prompt for Bea"]);
}

/// Several `instructions` declarations with `append:` and `cache_until_here:`.
struct Layered;
impl Agent for Layered {
    fn model(&self) -> Option<&str> {
        Some(OPENAI)
    }
    fn instruction_declarations(
        &self,
        _config: &Arc<Config>,
        _chat: &Value,
    ) -> rust_llm::Result<Vec<InstructionDeclaration>> {
        Ok(vec![
            InstructionDeclaration {
                cache_until_here: true,
                ..InstructionDeclaration::new("Base")
            },
            InstructionDeclaration {
                append: true,
                ..InstructionDeclaration::new("  ")
            },
            InstructionDeclaration {
                append: true,
                ..InstructionDeclaration::new("Extra")
            },
        ])
    }
}

#[test]
fn every_declaration_applies_in_order_and_blank_ones_are_skipped() {
    keys();
    let chat = Layered.chat().unwrap();
    assert_eq!(system_messages(&chat), ["Base", "Extra"]);
    assert!(chat.messages()[0].cache_until_here);
    assert!(!chat.messages()[1].cache_until_here);
}

// ---- prompt paths -----------------------------------------------------------------------------

mod support {
    /// `Support::BillingAgent`: a Rust type name carries no namespace, so it declares one.
    pub struct BillingAgent;
    impl rust_llm::Agent for BillingAgent {
        fn name(&self) -> String {
            "Support::BillingAgent".into()
        }
    }
}

// spec: agent_dsl_spec.rb:452 underscores the class name into a prompt directory
#[test]
fn the_class_name_underscores_into_a_prompt_directory() {
    assert_eq!(
        prompt_agent_path(&support::BillingAgent.name()),
        "support/billing_agent"
    );
    assert_eq!(
        prompt_agent_path(&SpecImplicitPromptAgent(rust_llm::context(|_| {})).name()),
        "spec_implicit_prompt_agent"
    );
}

#[test]
fn apply_except_instructions_leaves_the_messages_alone() {
    keys();
    let context = rust_llm::context(|c| c.request_timeout = Duration::from_secs(42));
    let chat = Chat::new(Some(OPENAI), Some("openai")).unwrap();
    let chat = InContext(context.clone())
        .apply_except_instructions(chat)
        .unwrap();
    assert!(Arc::ptr_eq(chat.config(), context.config()));
    let chat = Layered
        .apply_except_instructions(Chat::new(Some(OPENAI), Some("openai")).unwrap())
        .unwrap();
    assert!(chat.messages().is_empty());
}
