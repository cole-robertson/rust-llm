//! Renders each generator into a temp copy of a minimal Loco + Inertia app skeleton and checks
//! the files, injections, idempotency, and `--force`, like RubyLLM's generator specs
//! (`spec/generators/ruby_llm/*_spec.rb`).

use std::fs;
use std::path::Path;

use rust_llm_cli::{
    Generator, agent, chat_ui, install, provider, public_chat, schema, tool, upgrade,
};

mod support;
use support::app;

fn read(root: &Path, rel: &str) -> String {
    fs::read_to_string(root.join(rel)).unwrap_or_else(|e| panic!("{rel}: {e}"))
}

fn count(haystack: &str, needle: &str) -> usize {
    haystack.matches(needle).count()
}

fn actions(g: &Generator) -> Vec<(&str, &str)> {
    g.actions
        .iter()
        .map(|(a, p)| (a.as_str(), p.as_str()))
        .collect()
}

fn installed() -> tempfile::TempDir {
    let dir = app();
    let mut g = Generator::new(dir.path(), false);
    install::generate(&mut g, None).unwrap();
    assert!(g.failures.is_empty(), "{:?}", g.failures);
    dir
}

fn migration_module(root: &Path) -> String {
    let names: Vec<String> = fs::read_dir(root.join("migration/src"))
        .unwrap()
        .filter_map(|e| e.unwrap().file_name().into_string().ok())
        .filter(|n| n.ends_with("_create_rust_llm_records.rs"))
        .collect();
    assert_eq!(names.len(), 1, "exactly one rust_llm migration: {names:?}");
    names[0].trim_end_matches(".rs").to_string()
}

#[test]
fn install_adds_dependencies_to_the_app_and_migration_crates() {
    let dir = installed();
    let cargo = read(dir.path(), "Cargo.toml");
    let deps = cargo.split("[dev-dependencies]").next().unwrap();
    assert!(deps.contains("rust_llm = \"2.0.0\"\n"), "{cargo}");
    assert!(deps.contains("rust_llm_loco = \"2.0.0\"\n"), "{cargo}");
    assert!(read(dir.path(), "migration/Cargo.toml").contains("loco-rs = { workspace = true }\nrust_llm_loco = \"2.0.0\"\n\n[dependencies.sea-orm-migration]"));
}

#[test]
fn install_path_points_dependencies_at_a_checkout() {
    let dir = app();
    let checkout = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut g = Generator::new(dir.path(), false);
    install::generate(&mut g, Some(checkout.to_str().unwrap())).unwrap();
    let crates = checkout.join("crates").canonicalize().unwrap();
    let cargo = read(dir.path(), "Cargo.toml");
    assert!(
        cargo.contains(&format!(
            "rust_llm = {{ path = \"{}\" }}",
            crates.join("rust_llm").display()
        )),
        "{cargo}"
    );
    assert!(
        cargo.contains(&format!(
            "rust_llm_loco = {{ path = \"{}\" }}",
            crates.join("rust_llm_loco").display()
        )),
        "{cargo}"
    );

    let other = app();
    let mut g = Generator::new(other.path(), false);
    let err = install::generate(&mut g, Some("/nonexistent")).unwrap_err();
    assert!(err.contains("crates/rust_llm not found"), "{err}");
}

#[test]
fn install_migration_runs_rust_llm_loco_migrations_and_is_registered_above_the_loco_marker() {
    let dir = installed();
    let module = migration_module(dir.path());
    assert!(
        module.starts_with('m') && module.as_bytes()[1..9].iter().all(u8::is_ascii_digit),
        "{module}"
    );
    let migration = read(dir.path(), &format!("migration/src/{module}.rs"));
    assert!(migration.contains("for migration in rust_llm_loco::migrations()"));
    assert!(
        migration.contains("rust_llm_loco::migrations().iter().rev()"),
        "down runs in reverse"
    );

    let lib = read(dir.path(), "migration/src/lib.rs");
    let registered =
        format!("            Box::new({module}::Migration),\n            // inject-above");
    assert!(lib.contains(&registered), "{lib}");
    // Where Loco's own model generator puts it (`before: "pub struct Migrator"`).
    assert!(
        lib.contains(&format!("mod {module};\npub struct Migrator;")),
        "{lib}"
    );
}

#[test]
fn install_writes_the_initializer_and_registers_it_first() {
    let dir = installed();
    let init = read(dir.path(), "src/initializers/rust_llm.rs");
    assert!(init.contains("rust_llm::configure(|config|"));
    assert!(init.contains("std::env::var(\"OPENAI_API_KEY\")"));
    assert_eq!(
        read(dir.path(), "src/initializers/mod.rs"),
        "pub mod rust_llm;\n",
        "no stray blank line"
    );
    assert!(read(dir.path(), "src/lib.rs").contains("pub mod initializers;\npub mod models;"));
    assert!(read(dir.path(), "src/app.rs").contains(
        "Ok(vec![Box::new(crate::initializers::rust_llm::RustLlm), Box::new(crate::inertia::ssr::SsrSupervisor)])"
    ));
}

#[test]
fn install_writes_models_and_convention_directories() {
    let dir = installed();
    assert!(read(dir.path(), "src/models/chats.rs").contains("pub use rust_llm_loco::ChatRecord;"));
    assert!(read(dir.path(), "src/models/messages.rs").contains("pub async fn create_user("));
    // Sorted among existing `pub mod` lines, above the next module's `#[cfg]`.
    assert_eq!(
        read(dir.path(), "src/models/mod.rs"),
        "pub mod _entities;\n#[cfg(feature = \"bench\")]\npub mod bench_events;\npub mod chats;\npub mod messages;\npub mod sessions;\npub mod users;\n"
    );
    for dir_name in ["tools", "agents", "schemas"] {
        assert!(
            dir.path().join(format!("src/{dir_name}/mod.rs")).exists(),
            "{dir_name}"
        );
        assert!(read(dir.path(), "src/lib.rs").contains(&format!("pub mod {dir_name};")));
    }
    assert!(dir.path().join("src/prompts/.gitkeep").exists());
}

#[test]
fn install_twice_changes_nothing() {
    let dir = installed();
    let before: Vec<(String, String)> = walk(dir.path());
    let mut g = Generator::new(dir.path(), false);
    install::generate(&mut g, None).unwrap();
    assert!(g.failures.is_empty(), "{:?}", g.failures);
    assert!(
        actions(&g).iter().all(|(a, _)| *a == "identical"),
        "{:?}",
        g.actions
    );
    assert_eq!(walk(dir.path()), before);
    migration_module(dir.path());
}

#[test]
fn install_outside_a_loco_app_fails_without_writing() {
    let dir = tempfile::tempdir().unwrap();
    let mut g = Generator::new(dir.path(), false);
    let err = install::generate(&mut g, None).unwrap_err();
    assert!(err.contains("root of a Loco app"), "{err}");
    assert!(walk(dir.path()).is_empty());
}

#[test]
fn a_missing_anchor_is_reported_not_silently_skipped() {
    let dir = app();
    fs::write(
        dir.path().join("migration/src/lib.rs"),
        "pub struct Migrator;\n",
    )
    .unwrap();
    let mut g = Generator::new(dir.path(), false);
    install::generate(&mut g, None).unwrap();
    assert_eq!(g.failures.len(), 1, "{:?}", g.failures);
    assert!(
        g.failures[0].contains("anchor `inject-above` not found"),
        "{}",
        g.failures[0]
    );
}

#[test]
fn tool_writes_the_tool_and_its_components() {
    let dir = installed();
    let mut g = Generator::new(dir.path(), false);
    tool::generate(&mut g, "weather").unwrap();
    let tool = read(dir.path(), "src/tools/weather_tool.rs");
    assert!(tool.contains("pub struct WeatherTool;"));
    assert!(tool.contains("impl Tool for WeatherTool {"));
    assert!(tool.contains("\"TODO: describe what this tool does\""));
    assert!(tool.contains("Ok(\"TODO: implement WeatherTool\".into())"));
    assert!(read(dir.path(), "src/tools/mod.rs").contains("pub mod weather_tool;"));
    // Named for `Tool::name()` (`weather`), the key message-list.tsx looks components up by.
    let call = read(
        dir.path(),
        "frontend/components/messages/tool_calls/weather.tsx",
    );
    assert!(call.contains("export default function ToolCall("));
    assert!(call.contains("const label = \"Weather Call\""));
    let result = read(
        dir.path(),
        "frontend/components/messages/tool_results/weather.tsx",
    );
    assert!(result.contains("export default function ToolResult("));
    assert!(result.contains("const label = \"Weather Result\""));
    assert!(result.contains("message.tool_error_message"));
}

#[test]
fn tool_names_follow_tool_name_rules() {
    let dir = installed();
    let mut g = Generator::new(dir.path(), false);
    tool::generate(&mut g, "HTTPProxyTool").unwrap();
    assert!(read(dir.path(), "src/tools/http_proxy_tool.rs").contains("pub struct HTTPProxyTool;"));
    assert!(
        dir.path()
            .join("frontend/components/messages/tool_calls/http_proxy.tsx")
            .exists()
    );
    assert_eq!(
        rust_llm::tool::tool_name_from_type("app::tools::HTTPProxyTool"),
        "http_proxy"
    );

    tool::generate(&mut g, "weather-lookup").unwrap();
    assert!(
        read(dir.path(), "src/tools/weather_lookup_tool.rs")
            .contains("pub struct WeatherLookupTool;")
    );

    assert!(
        tool::generate(&mut g, "admin/weather").is_err(),
        "namespaces are rejected"
    );
    assert!(
        tool::generate(&mut g, "default").is_err(),
        "`default` is the fallback component"
    );
}

#[test]
fn existing_files_are_skipped_unless_forced() {
    let dir = installed();
    let path = "src/tools/weather_tool.rs";
    tool::generate(&mut Generator::new(dir.path(), false), "Weather").unwrap();
    fs::write(dir.path().join(path), "// edited\n").unwrap();

    let mut g = Generator::new(dir.path(), false);
    tool::generate(&mut g, "Weather").unwrap();
    assert!(actions(&g).contains(&("skip", path)), "{:?}", g.actions);
    assert!(actions(&g).contains(&("identical", "src/tools/mod.rs")));
    assert_eq!(read(dir.path(), path), "// edited\n");
    assert_eq!(
        count(
            &read(dir.path(), "src/tools/mod.rs"),
            "pub mod weather_tool;"
        ),
        1
    );

    let mut g = Generator::new(dir.path(), true);
    tool::generate(&mut g, "Weather").unwrap();
    assert!(actions(&g).contains(&("force", path)), "{:?}", g.actions);
    assert!(read(dir.path(), path).contains("pub struct WeatherTool;"));
}

#[test]
fn agent_writes_the_agent_and_an_empty_instructions_prompt() {
    let dir = installed();
    let mut g = Generator::new(dir.path(), false);
    agent::generate(&mut g, "support").unwrap();
    let agent = read(dir.path(), "src/agents/support_agent.rs");
    assert!(agent.contains("pub struct SupportAgent;"));
    assert!(agent.contains("impl Agent for SupportAgent {"));
    assert!(agent.contains("include_str!(\"../prompts/support_agent/instructions.txt\")"));
    assert!(
        agent.contains("(!text.trim().is_empty()).then"),
        "blank prompt means no instructions"
    );
    assert_eq!(
        read(dir.path(), "src/prompts/support_agent/instructions.txt"),
        ""
    );
    assert!(read(dir.path(), "src/agents/mod.rs").contains("pub mod support_agent;"));
}

#[test]
fn schema_writes_a_json_schema_struct_and_the_schemars_dependency() {
    let dir = installed();
    let mut g = Generator::new(dir.path(), false);
    schema::generate(&mut g, "Product").unwrap();
    let schema = read(dir.path(), "src/schemas/product_schema.rs");
    assert!(
        schema.contains(
            "#[derive(Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]"
        )
    );
    assert!(schema.contains("pub struct ProductSchema {}"));
    assert!(read(dir.path(), "src/schemas/mod.rs").contains("pub mod product_schema;"));
    assert!(read(dir.path(), "Cargo.toml").contains("schemars = \"1.2\""));

    schema::generate(&mut Generator::new(dir.path(), false), "ProductSchema").unwrap();
    assert_eq!(count(&read(dir.path(), "Cargo.toml"), "schemars"), 1);
    assert_eq!(
        count(
            &read(dir.path(), "src/schemas/mod.rs"),
            "pub mod product_schema;"
        ),
        1
    );
}

#[test]
fn chat_ui_requires_install_first() {
    let dir = app();
    let mut g = Generator::new(dir.path(), false);
    let err = chat_ui::generate(&mut g).unwrap_err();
    assert!(err.contains("rust-llm generate install"), "{err}");
    assert!(!dir.path().join("src/controllers/chats.rs").exists());
}

#[test]
fn chat_ui_writes_controllers_channel_worker_pages_and_tests() {
    let dir = installed();
    let mut g = Generator::new(dir.path(), false);
    chat_ui::generate(&mut g).unwrap();
    assert!(g.failures.is_empty(), "{:?}", g.failures);
    let root = dir.path();

    let chats = read(root, "src/controllers/chats.rs");
    for handler in [
        "async fn index(",
        "async fn new(",
        "async fn create(",
        "async fn show(",
        "async fn destroy(",
    ] {
        assert!(chats.contains(handler), "{handler}");
    }
    // Account-scoped: every handler takes CurrentAccount and finds chats in the account.
    assert!(!chats.contains("Authenticated"), "no global handlers");
    assert_eq!(count(&chats, "current: CurrentAccount"), 4);
    assert!(chats.contains("chats::list_in_account(&ctx.db, current.account.id)"));
    assert!(chats.contains("chats::create_in_account(&ctx.db, current.account.id,"));
    assert_eq!(
        count(
            &chats,
            "chats::find_in_account(&ctx.db, current.account.id, id)"
        ),
        2
    );
    let messages = read(root, "src/controllers/messages.rs");
    assert!(messages.contains("chats::find_in_account(&ctx.db, account_id, chat_id)"));
    assert!(messages.contains("ChatResponseWorker::perform_later"));
    assert!(!read(root, "src/controllers/models.rs").contains("Authenticated"));

    // The worker streams and broadcasts on ChatChannel, keyed by account and chat.
    let worker = read(root, "src/workers/chat_response.rs");
    assert!(worker.contains("record\n        .complete_stream(db, &mut chat,"));
    assert!(worker.contains("ChatChannel::broadcast_to(account_id, chat_id, payload)"));
    for event in ["message_start", "chunk", "message_end", "error"] {
        assert!(
            worker.contains(&format!("\"type\": \"{event}\"")),
            "{event}"
        );
    }
    let channel = read(root, "src/channels/chat.rs");
    assert!(channel.contains("format!(\"{account_id}:{id}\")"));
    assert!(channel.contains("chats::find_in_account(&ctx.db, membership.account_id, id)"));

    // The page subscribes with useChannel; no polling.
    let show = read(root, "frontend/pages/chats/show.tsx");
    assert!(show.contains("useChannel<ChatEvent>(\"ChatChannel\", at,"));
    assert!(show.contains("only: [\"messages\", \"awaiting_response\"]"));
    assert!(!show.contains("usePoll"), "streaming, not polling");
    for page in ["chats/index", "chats/new", "models/index", "models/show"] {
        let source = read(root, &format!("frontend/pages/{page}.tsx"));
        assert!(source.contains("useCurrentAccount()"), "{page}");
    }
    let list = read(root, "frontend/components/messages/message-list.tsx");
    assert!(list.contains("import.meta.glob") && list.contains("\"./tool_calls/*.tsx\""));
    assert!(
        read(root, "frontend/components/messages/model-picker.tsx")
            .contains("from \"@/components/ui/select\"")
    );

    // Request tests, importing the app by its package name.
    let test = read(root, "tests/requests/chats.rs");
    assert!(test.contains("use app::{"), "{}", &test[..400]);
    assert!(test.contains("async fn another_accounts_chat_is_a_404()"));
    assert!(test.contains("async fn a_non_member_cannot_subscribe_to_a_chat()"));
    assert!(root.join("tests/requests/anthropic_stub.rs").exists());
    assert!(
        read(root, "tests/requests/mod.rs")
            .contains("mod accounts;\nmod anthropic_stub;\nmod chats;\nmod live;")
    );
}

#[test]
fn chat_ui_adds_the_account_to_chats() {
    let dir = installed();
    chat_ui::generate(&mut Generator::new(dir.path(), false)).unwrap();
    let root = dir.path();
    let migrations: Vec<String> = fs::read_dir(root.join("migration/src"))
        .unwrap()
        .filter_map(|e| e.unwrap().file_name().into_string().ok())
        .filter(|n| n.ends_with(&format!("{}.rs", chat_ui::MIGRATION_SUFFIX)))
        .collect();
    assert_eq!(migrations.len(), 1, "{migrations:?}");
    let module = migrations[0].trim_end_matches(".rs");
    let lib = read(root, "migration/src/lib.rs");
    // After the install migration, so `chats` exists when it runs.
    let install = lib
        .find(&format!("Box::new({}::Migration)", migration_module(root)))
        .unwrap();
    let account = lib.find(&format!("Box::new({module}::Migration)")).unwrap();
    assert!(install < account, "{lib}");
    let model = read(root, "src/models/chats.rs");
    assert!(model.starts_with(include_str!("../templates/install/chat_model.rs")));
    assert!(model.ends_with(chat_ui::CHAT_MODEL_ACCOUNTS));
}

#[test]
fn chat_ui_refuses_an_app_without_accounts_or_live_updates() {
    let dir = installed();
    fs::remove_file(dir.path().join("src/live/mod.rs")).unwrap();
    let mut g = Generator::new(dir.path(), false);
    let err = chat_ui::generate(&mut g).unwrap_err();
    assert!(err.contains("src/live/mod.rs"), "{err}");
    assert!(!dir.path().join("src/controllers/chats.rs").exists());
}

#[test]
fn chat_ui_wires_routes_controllers_channel_worker_and_sidebar() {
    let dir = installed();
    chat_ui::generate(&mut Generator::new(dir.path(), false)).unwrap();
    let root = dir.path();

    assert!(read(root, "src/controllers/mod.rs").contains("pub mod bench;\npub mod chats;\npub mod dashboard;\npub mod messages;\npub mod models;\npub mod users;"));
    let app = read(root, "src/app.rs");
    for controller in ["chats", "messages", "models"] {
        assert_eq!(
            count(
                &app,
                &format!(".add_route(controllers::{controller}::routes())")
            ),
            1
        );
    }
    assert!(app.contains("fn connect_workers(ctx: &AppContext, queue: &Queue) -> Result<()> {\n        queue\n            .register(crate::workers::chat_response::ChatResponseWorker::build(\n"));
    assert!(read(root, "src/workers/mod.rs").contains("pub mod chat_response;"));
    let channels = read(root, "src/channels/mod.rs");
    assert!(channels.contains("pub mod account;\npub mod chat;"));
    assert!(channels.contains("        Arc::new(chat::ChatChannel),\n        // channels-inject"));

    let table = read(root, "src/route_table.rs");
    assert!(
        table.contains(
            "pub const CHAT_MESSAGES: &str = \"/{account_slug}/chats/{chat_id}/messages\";"
        )
    );
    assert!(table.contains("pub fn chat_path(slug: &str, id: i32) -> String {"));
    assert!(table.find("pub fn chat_path").unwrap() < table.find("// scaffold:paths").unwrap());
    assert!(
        table.contains(
            "        route(\n            \"models.show\",\n            Get,\n            MODEL,\n"
        ),
        "rustfmt-shaped"
    );
    assert!(table.find("\"models.show\"").unwrap() < table.find("// scaffold:routes").unwrap());
    // RubyLLM's `refresh` action, for the account's managers.
    assert!(table.contains("pub const MODELS_REFRESH: &str = \"/{account_slug}/models/refresh\";"));
    let models = read(root, "src/controllers/models.rs");
    assert!(models.contains("rust_llm::models::refresh(false).await"));
    assert!(models.contains("if !current.is_manager() {"));
    assert!(models.contains(".add(route_table::MODELS_REFRESH, post(refresh))"));
    assert!(
        read(root, "frontend/pages/models/index.tsx")
            .contains("router.post(routes.refresh(accountSlug).url)")
    );

    // The link goes in the account's nav (`// scaffold:nav`), not the global one.
    let sidebar = read(root, "frontend/components/app-sidebar.tsx");
    assert!(
        sidebar.contains("import { chats, dashboard } from \"@/routes\""),
        "{sidebar}"
    );
    assert!(
        sidebar.contains(
            "import { BookOpen, Folder, LayoutGrid, MessagesSquare } from \"lucide-react\""
        ),
        "{sidebar}"
    );
    assert!(sidebar.contains(
        "        {\n          title: \"Chats\",\n          href: chats.index(account.slug).url,\n          icon: MessagesSquare,\n        },\n        // scaffold:nav\n"
    ), "{sidebar}");
    assert!(sidebar.contains("const globalNavItems: NavItem[] = [\n  // scaffold:nav-global\n]"));
}

#[test]
fn chat_ui_twice_changes_nothing() {
    let dir = installed();
    chat_ui::generate(&mut Generator::new(dir.path(), false)).unwrap();
    let before = walk(dir.path());
    let mut g = Generator::new(dir.path(), false);
    chat_ui::generate(&mut g).unwrap();
    assert!(g.failures.is_empty(), "{:?}", g.failures);
    assert!(
        actions(&g).iter().all(|(a, _)| *a == "identical"),
        "{:?}",
        g.actions
    );
    assert_eq!(walk(dir.path()), before);
}

#[test]
fn tool_components_import_the_chat_ui_modules_that_exist() {
    let dir = installed();
    chat_ui::generate(&mut Generator::new(dir.path(), false)).unwrap();
    tool::generate(&mut Generator::new(dir.path(), false), "Weather").unwrap();
    let components = dir.path().join("frontend/components/messages");
    for file in ["tool_calls/weather.tsx", "tool_results/weather.tsx"] {
        let source = read(&components, file);
        for import in source.lines().filter_map(|l| l.split(" from \"").nth(1)) {
            let module = import.trim_end_matches('"');
            let resolved = components
                .join(Path::new(file).parent().unwrap())
                .join(module);
            let exists = ["ts", "tsx"]
                .iter()
                .any(|ext| resolved.with_extension(ext).exists());
            assert!(
                exists,
                "{file} imports {module}, which chat_ui did not write"
            );
        }
    }
}

#[test]
fn provider_writes_a_module_and_test_and_registers_the_module() {
    let dir = tempfile::tempdir().unwrap();
    let providers = "crates/rust_llm/src/providers.rs";
    fs::create_dir_all(dir.path().join("crates/rust_llm/src")).unwrap();
    fs::write(
        dir.path().join(providers),
        "//! Providers.\n\nuse crate::config::Config;\n",
    )
    .unwrap();
    let mut g = Generator::new(dir.path(), false);
    let options = provider::Options {
        dialect: Some("anthropic"),
        api_base: Some("https://api.acme.test/v1"),
        models_dev_provider: None,
        dynamic_models: true,
    };
    provider::generate(&mut g, "acme-ai", &options).unwrap();

    let module = read(dir.path(), "crates/rust_llm/src/providers/acme_ai.rs");
    assert!(module.contains("pub const SLUG: &str = \"acme_ai\";"));
    assert!(module.contains("pub const DISPLAY: &str = \"AcmeAi\";"));
    assert!(module.contains("pub const PROTOCOL: ProtocolName = ProtocolName::Anthropic;"));
    assert!(module.contains("pub const DEFAULT_API_BASE: &str = \"https://api.acme.test/v1\";"));
    assert!(module.contains("pub const ASSUME_MODELS_EXIST: bool = true;"));
    assert!(module.contains("config.get(\"acme_ai_api_key\")"));
    let test = read(dir.path(), "crates/rust_llm/tests/provider_acme_ai.rs");
    assert!(test.contains("use rust_llm::providers::{ProtocolName, acme_ai};"));
    assert!(test.contains("\"Bearer test-key\""));
    assert_eq!(
        read(dir.path(), providers),
        "//! Providers.\n\npub mod acme_ai;\nuse crate::config::Config;\n"
    );
    assert!(
        g.notes
            .iter()
            .any(|n| n.contains("add `AcmeAi` to `enum Provider`"))
    );

    let bad = provider::Options {
        dialect: Some("converse"),
        api_base: None,
        models_dev_provider: None,
        dynamic_models: false,
    };
    assert!(
        provider::generate(&mut g, "acme", &bad)
            .unwrap_err()
            .contains("converse is not ported")
    );
    let plain = provider::Options {
        dialect: None,
        api_base: None,
        models_dev_provider: None,
        dynamic_models: false,
    };
    assert!(provider::generate(&mut g, "9lives", &plain).is_err());
}

#[test]
fn upgrade_writes_one_migration_and_registers_it() {
    let dir = installed();
    let mut g = Generator::new(dir.path(), false);
    upgrade::generate(&mut g).unwrap();
    assert!(g.failures.is_empty(), "{:?}", g.failures);
    let lib = read(dir.path(), "migration/src/lib.rs");
    let module = lib
        .lines()
        .find_map(|l| {
            l.strip_prefix("mod ")?
                .strip_suffix("_upgrade_rust_llm_to_2_1;")
        })
        .map(|stem| format!("{stem}_upgrade_rust_llm_to_2_1"))
        .expect("upgrade migration declared");
    assert!(lib.contains(&format!(
        "            Box::new({module}::Migration),\n            // inject-above"
    )));
    assert!(
        lib.find(&format!("Box::new({module}")) > lib.find("_create_rust_llm_records::Migration"),
        "runs after the install migration"
    );
    assert_eq!(
        read(dir.path(), &format!("migration/src/{module}.rs")),
        include_str!("../templates/upgrade/migration.rs")
    );

    // A second run reuses the file instead of adding another migration.
    let before = walk(dir.path());
    let mut g = Generator::new(dir.path(), false);
    upgrade::generate(&mut g).unwrap();
    assert!(
        actions(&g).iter().all(|(a, _)| *a == "identical"),
        "{:?}",
        g.actions
    );
    assert_eq!(walk(dir.path()), before);
}

#[test]
fn cli_parses_generators_and_options() {
    let dir = app();
    let args = |a: &[&str]| a.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    assert_eq!(
        rust_llm_cli::run(&args(&["generate", "install"]), dir.path()),
        0
    );
    assert_eq!(
        rust_llm_cli::run(&args(&["g", "tool", "Weather", "--force"]), dir.path()),
        0
    );
    assert_eq!(
        rust_llm_cli::run(&args(&["generate", "tool"]), dir.path()),
        1,
        "NAME is required"
    );
    assert_eq!(
        rust_llm_cli::run(&args(&["generate", "nope"]), dir.path()),
        1
    );
    assert_eq!(
        rust_llm_cli::run(&args(&["generate", "install", "--bogus"]), dir.path()),
        1
    );
    assert_eq!(
        rust_llm_cli::run(&args(&["generate", "upgrade"]), dir.path()),
        0
    );
    assert!(dir.path().join("src/tools/weather_tool.rs").exists());
}

#[test]
fn named_imports_wrap_past_prettiers_print_width_and_stay_idempotent() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("a.tsx");
    fs::write(
        &file,
        "import { BookOpen, Folder, LayoutGrid, Settings, Users } from \"lucide-react\"\nimport { a } from \"@/routes\"\n",
    )
    .unwrap();
    let mut g = Generator::new(dir.path(), false);
    g.named_import("a.tsx", "lucide-react", "MessagesSquare");
    assert_eq!(
        read(dir.path(), "a.tsx"),
        "import {\n  BookOpen,\n  Folder,\n  LayoutGrid,\n  MessagesSquare,\n  Settings,\n  Users,\n} from \"lucide-react\"\nimport { a } from \"@/routes\"\n"
    );
    // A wrapped import is still found: an existing name is `identical`, a new one joins it.
    g.named_import("a.tsx", "lucide-react", "Folder");
    g.named_import("a.tsx", "@/routes", "b");
    assert!(g.failures.is_empty(), "{:?}", g.failures);
    assert_eq!(g.actions[1].0, "identical");
    let text = read(dir.path(), "a.tsx");
    assert!(text.contains("  MessagesSquare,\n"), "{text}");
    assert!(
        text.ends_with("import { a, b } from \"@/routes\"\n"),
        "{text}"
    );
}

#[test]
fn public_chat_needs_the_kit_and_install() {
    let dir = app();
    let err = public_chat::generate(&mut Generator::new(dir.path(), false)).unwrap_err();
    assert!(err.contains("rust-llm generate install"), "{err}");
    fs::remove_file(dir.path().join("src/controllers/rate_limit.rs")).unwrap();
    let err = public_chat::generate(&mut Generator::new(dir.path(), false)).unwrap_err();
    assert!(err.contains("src/controllers/rate_limit.rs"), "{err}");
    assert!(!dir.path().join("src/controllers/public_chat.rs").exists());
}

#[test]
fn public_chat_writes_an_unauthenticated_streaming_page_with_limits() {
    let dir = installed();
    let mut g = Generator::new(dir.path(), false);
    public_chat::generate(&mut g).unwrap();
    assert!(g.failures.is_empty(), "{:?}", g.failures);
    let root = dir.path();

    let controller = read(root, "src/controllers/public_chat.rs");
    // No sign-in, and no shared channel: the reply streams back in the POST's response.
    assert!(!controller.contains("Authenticated") && !controller.contains("CurrentAccount"));
    assert!(!controller.contains("broadcast_to(") && !controller.contains("crate::live"));
    assert!(controller.contains("Sse::new(events)"));
    for (var, default) in [
        ("PUBLIC_CHAT_IP_MESSAGES", "20"),
        ("PUBLIC_CHAT_SESSION_MESSAGES", "10"),
        ("PUBLIC_CHAT_WINDOW_SECS", "600"),
        ("PUBLIC_CHAT_MAX_INPUT_CHARS", "4_000"),
        ("PUBLIC_CHAT_MAX_OUTPUT_TOKENS", "1_024"),
        ("PUBLIC_CHAT_MAX_TURNS", "20"),
    ] {
        assert!(
            controller.contains(&format!("var(\"{var}\", {default})")),
            "{var}"
        );
    }
    assert!(controller.contains(".with_max_output_tokens(limits.max_output_tokens)"));
    assert!(controller.contains("cookie.set_http_only(true);"));
    assert!(root.join("frontend/pages/public_chat/show.tsx").exists());

    let test = read(root, "tests/requests/public_chat.rs");
    assert!(test.contains("use app::{"));
    for name in [
        "guests_never_see_each_others_conversations",
        "one_ip_is_rate_limited_across_conversations",
        "one_conversation_is_rate_limited",
        "a_message_over_the_length_limit_is_refused",
        "a_conversation_stops_at_its_turn_limit",
        "the_output_token_cap_is_sent_to_the_model",
    ] {
        assert!(test.contains(&format!("async fn {name}()")), "{name}");
    }

    let table = read(root, "src/route_table.rs");
    assert!(table.contains("pub const PUBLIC_CHAT: &str = \"/chat\";"));
    assert!(table.contains("pub const PUBLIC_CHAT_MESSAGES: &str = \"/chat/messages\";"));
    assert_eq!(
        count(
            &read(root, "src/app.rs"),
            ".add_route(controllers::public_chat::routes())"
        ),
        1
    );
    // `/chat` can't also be an account's slug.
    assert!(
        read(root, "src/models/accounts.rs")
            .contains("RESERVED_SLUGS: &[&str] = &[\n    \"chat\",\n")
    );
    assert!(
        read(root, "tests/requests/mod.rs")
            .contains("mod anthropic_stub;\nmod live;\nmod public_chat;")
    );
}

#[test]
fn public_chat_and_chat_ui_share_the_stub_and_run_twice_unchanged() {
    let dir = installed();
    chat_ui::generate(&mut Generator::new(dir.path(), false)).unwrap();
    let mut g = Generator::new(dir.path(), false);
    public_chat::generate(&mut g).unwrap();
    assert!(g.failures.is_empty(), "{:?}", g.failures);
    assert!(
        actions(&g).contains(&("identical", "tests/requests/anthropic_stub.rs")),
        "{:?}",
        g.actions
    );
    let before = walk(dir.path());
    let mut g = Generator::new(dir.path(), false);
    public_chat::generate(&mut g).unwrap();
    assert!(
        actions(&g).iter().all(|(a, _)| *a == "identical"),
        "{:?}",
        g.actions
    );
    assert_eq!(walk(dir.path()), before);
}

fn walk(root: &Path) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                stack.push(path);
            } else {
                let rel = path.strip_prefix(root).unwrap().display().to_string();
                out.push((rel, fs::read_to_string(&path).unwrap()));
            }
        }
    }
    out.sort();
    out
}
