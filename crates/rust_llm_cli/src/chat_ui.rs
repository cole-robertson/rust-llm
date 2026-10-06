//! Port of `lib/generators/ruby_llm/chat_ui/chat_ui_generator.rb` for the Loco + Inertia starter
//! kit: Inertia + React + shadcn/ui instead of ERB + Turbo, inside the kit's accounts.
//!
//! | RubyLLM | here |
//! |---|---|
//! | `app/controllers/{chats,messages,models}_controller.rb` | `src/controllers/{chats,messages,models}.rs`, under `/{account_slug}` with `CurrentAccount` |
//! | `app/jobs/chat_response_job.rb` | `src/workers/chat_response.rs` (Loco worker, `ChatRecord::complete_stream`) |
//! | `broadcasts_to` + `broadcast_append_chunk` (Turbo Streams) | `src/channels/chat.rs` (`ChatChannel`, the kit's live updates) |
//! | `app/views/chats/*`, `app/views/models/*` | `frontend/pages/{chats,models}/*.tsx` |
//! | `app/views/messages/_<role>`, `tool_calls/_default`, `tool_results/_default`, `_error` | `frontend/components/messages/*` |
//! | `form.select :model` | `frontend/components/messages/model-picker.tsx` (shadcn `Select`) |
//! | `MessagesHelper#tool_call_partial` | `import.meta.glob` lookup in `message-list.tsx` |
//! | `config/routes.rb` resources | `src/route_table.rs` constants and routes |
//!
//! Streaming: like RubyLLM's job, the worker streams the reply. `complete_stream` creates the
//! assistant row before the first chunk, and the worker broadcasts on `ChatChannel` (stream key
//! `"{account_id}:{chat_id}"`): `message_start`, `chunk` deltas (batched every
//! [`CHUNK_INTERVAL_MS`] ms), and `message_end` once the row is written. The page renders the
//! deltas as they arrive and reloads its `messages` prop on `message_end`.
//!
//! Chats belong to an account (`chats.account_id`, added by this generator's migration), and
//! every query goes through `find_in_account`, so another account's chat is a 404.

use crate::{Anchor, Generator, install};

/// How often the worker sends the deltas it has collected, in milliseconds.
pub const CHUNK_INTERVAL_MS: u64 = 50;

/// The suffix of the migration that adds `chats.account_id`.
pub const MIGRATION_SUFFIX: &str = "_add_account_to_chats";

macro_rules! templates {
    ($($path:literal => $source:literal),* $(,)?) => {
        const FILES: &[(&str, &str)] = &[$(($path, include_str!(concat!("../templates/chat_ui/", $source)))),*];
    };
}

templates! {
    "src/controllers/chats.rs" => "controllers/chats.rs",
    "src/controllers/messages.rs" => "controllers/messages.rs",
    "src/controllers/models.rs" => "controllers/models.rs",
    "src/channels/chat.rs" => "channels/chat.rs",
    "src/workers/chat_response.rs" => "workers/chat_response.rs",
    "tests/requests/chats.rs" => "tests/chats.rs",
    "tests/requests/anthropic_stub.rs" => "tests/anthropic_stub.rs",
    "frontend/pages/chats/index.tsx" => "pages/chats/index.tsx",
    "frontend/pages/chats/new.tsx" => "pages/chats/new.tsx",
    "frontend/pages/chats/show.tsx" => "pages/chats/show.tsx",
    "frontend/pages/models/index.tsx" => "pages/models/index.tsx",
    "frontend/pages/models/show.tsx" => "pages/models/show.tsx",
    "frontend/components/messages/types.ts" => "components/messages/types.ts",
    "frontend/components/messages/format.ts" => "components/messages/format.ts",
    "frontend/components/messages/error.tsx" => "components/messages/error.tsx",
    "frontend/components/messages/message-list.tsx" => "components/messages/message-list.tsx",
    "frontend/components/messages/model-picker.tsx" => "components/messages/model-picker.tsx",
    "frontend/components/messages/tool_calls/default.tsx" => "components/messages/tool_calls/default.tsx",
    "frontend/components/messages/tool_results/default.tsx" => "components/messages/tool_results/default.tsx",
}

const MIGRATION: &str = include_str!("../templates/chat_ui/migration.rs");
/// `find_in_account`, `list_in_account`, `create_in_account`, appended to `src/models/chats.rs`.
pub const CHAT_MODEL_ACCOUNTS: &str = include_str!("../templates/chat_ui/chat_model_accounts.rs");

const PATHS: &str = r#"pub const CHATS: &str = "/{account_slug}/chats";
pub const NEW_CHAT: &str = "/{account_slug}/chats/new";
pub const CHAT: &str = "/{account_slug}/chats/{id}";
pub const CHAT_MESSAGES: &str = "/{account_slug}/chats/{chat_id}/messages";
pub const MODELS: &str = "/{account_slug}/models";
pub const MODELS_REFRESH: &str = "/{account_slug}/models/refresh";
pub const MODEL: &str = "/{account_slug}/models/{id}";

/// `/{account_slug}/chats`.
#[must_use]
pub fn chats_path(slug: &str) -> String {
    CHATS.replace("{account_slug}", &encode_segment(slug))
}

/// `/{account_slug}/chats/new`.
#[must_use]
pub fn new_chat_path(slug: &str) -> String {
    NEW_CHAT.replace("{account_slug}", &encode_segment(slug))
}

/// `/{account_slug}/chats/{id}`.
#[must_use]
pub fn chat_path(slug: &str, id: i32) -> String {
    chats_path(slug) + "/" + &id.to_string()
}

/// `/{account_slug}/models`.
#[must_use]
pub fn models_path(slug: &str) -> String {
    MODELS.replace("{account_slug}", &encode_segment(slug))
}
"#;

// rustfmt-shaped, like the kit's own entries, so `cargo fmt --check` stays clean.
const ROUTES: &str = r#"        route(
            "chats.index",
            Get,
            CHATS,
            ts("ChatsController", "chats", "index", None),
        ),
        route(
            "chats.create",
            Post,
            CHATS,
            ts("ChatsController", "chats", "create", None),
        ),
        route(
            "chats.new",
            Get,
            NEW_CHAT,
            ts("ChatsController", "chats", "new", Some("newChat")),
        ),
        route(
            "chats.show",
            Get,
            CHAT,
            ts("ChatsController", "chats", "show", Some("chat")),
        ),
        route(
            "chats.destroy",
            Delete,
            CHAT,
            ts("ChatsController", "chats", "destroy", None),
        ),
        route(
            "chat_messages.create",
            Post,
            CHAT_MESSAGES,
            ts("ChatMessagesController", "chatMessages", "create", None),
        ),
        route(
            "models.index",
            Get,
            MODELS,
            ts("ModelsController", "models", "index", None),
        ),
        route(
            "models.refresh",
            Post,
            MODELS_REFRESH,
            ts("ModelsController", "models", "refresh", None),
        ),
        route(
            "models.show",
            Get,
            MODEL,
            ts("ModelsController", "models", "show", Some("model")),
        ),"#;

const NAV_ITEM: &str = r#"{ title: "Chats", href: chats.index(account.slug).url, icon: MessagesSquare },"#;

/// What the kit provides that the account-scoped chat UI builds on.
const KIT_FILES: &[(&str, &str)] = &[
    ("src/route_table.rs", "the route table"),
    ("src/models/accounts.rs", "accounts (`CurrentAccount`)"),
    ("src/live/mod.rs", "live updates (`GET /live`)"),
    ("src/channels/mod.rs", "the channel registry"),
];

/// `rust-llm generate chat_ui`.
pub fn generate(g: &mut Generator) -> Result<(), String> {
    // `check_model_exists`
    if !g.exists("src/models/messages.rs") {
        return Err("Model file not found: src/models/messages.rs\n\nPlease run the install generator first:\n  rust-llm generate install".into());
    }
    let missing: Vec<String> = KIT_FILES
        .iter()
        .filter(|(path, _)| !g.exists(path))
        .map(|(path, what)| format!("  {path} ({what})"))
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "chat_ui targets the Loco + Inertia starter kit (github.com/cole-robertson/inertia-rust-starter-kit): its chats belong to an account and stream over live updates. Not found:\n{}",
            missing.join("\n")
        ));
    }
    install::migration_template(g, MIGRATION_SUFFIX, MIGRATION);
    g.inject("src/models/chats.rs", CHAT_MODEL_ACCOUNTS, Anchor::End);
    let pkg = package_name(g)?;
    for (path, content) in FILES {
        g.file(path, &crate::render(content, &[("pkg_name", &pkg)]));
    }
    for controller in ["chats", "messages", "models"] {
        g.inject(
            "src/controllers/mod.rs",
            &format!("pub mod {controller};"),
            Anchor::Sorted("pub mod "),
        );
        g.inject(
            "src/app.rs",
            &format!("            .add_route(controllers::{controller}::routes())"),
            Anchor::After("AppRoutes::empty()"),
        );
    }
    g.inject(
        "src/channels/mod.rs",
        "pub mod chat;",
        Anchor::Sorted("pub mod "),
    );
    g.inject(
        "src/channels/mod.rs",
        "        Arc::new(chat::ChatChannel),",
        Anchor::Before("// channels-inject"),
    );
    g.inject(
        "src/workers/mod.rs",
        "pub mod chat_response;",
        Anchor::Sorted("pub mod "),
    );
    g.inject(
        "src/app.rs",
        "        queue\n            .register(crate::workers::chat_response::ChatResponseWorker::build(\n                ctx,\n            ))\n            .await?;",
        Anchor::After("fn connect_workers"),
    );
    g.inject(
        "src/route_table.rs",
        PATHS,
        Anchor::Before("// scaffold:paths"),
    );
    g.inject(
        "src/route_table.rs",
        ROUTES,
        Anchor::Before("// scaffold:routes"),
    );
    test_modules(g, &["anthropic_stub", "chats"]);
    link_sidebar(g);

    g.note("\n  Chat UI installed, under /{account_slug}/chats!");
    g.note("  Replies stream token by token over ChatChannel (src/channels/chat.rs).");
    g.note("\n  Next steps:");
    g.note("     1. cargo loco task routes:generate   (writes frontend/routes/*.ts for the new routes)");
    g.note("     2. cargo fmt --all && cargo loco db migrate");
    g.note("     3. cargo test --test mod requests::chats");
    g.note("     4. Start the app (bin/dev runs the worker too) and open Chats in the sidebar");
    Ok(())
}

/// The app's library crate name (`[package] name` in Cargo.toml, `-` as `_`): the generated
/// request tests import it, and the kit's `bin/rename` changes it.
pub(crate) fn package_name(g: &Generator) -> Result<String, String> {
    let cargo = g.read("Cargo.toml").ok_or("Cargo.toml not found")?;
    let mut in_package = false;
    for line in cargo.lines() {
        let line = line.trim();
        if line.starts_with('[') {
            in_package = line == "[package]";
        } else if in_package
            && let Some(value) = line.strip_prefix("name").map(str::trim_start)
            && let Some(value) = value.strip_prefix('=')
        {
            return Ok(value.trim().trim_matches('"').replace('-', "_"));
        }
    }
    Err("Cargo.toml has no [package] name".into())
}

/// `mod <name>;` in `tests/requests/mod.rs`, for each generated request test.
pub(crate) fn test_modules(g: &mut Generator, modules: &[&str]) {
    const MOD: &str = "tests/requests/mod.rs";
    if !g.exists(MOD) {
        return g.note(format!(
            "  {MOD} not found; add `mod {};` to your request tests yourself.",
            modules.join(";`, `mod ")
        ));
    }
    for module in modules {
        g.inject(MOD, &format!("mod {module};"), Anchor::Sorted("mod "));
    }
}

/// A "Chats" link in the account's sidebar nav, above `// scaffold:nav`, indented like it.
fn link_sidebar(g: &mut Generator) {
    const SIDEBAR: &str = "frontend/components/app-sidebar.tsx";
    let Some(sidebar) = g.read(SIDEBAR) else {
        return g.note(format!(
            "  {SIDEBAR} not found; link the chats page from your navigation yourself."
        ));
    };
    let indent: String = sidebar
        .lines()
        .find(|l| l.trim() == "// scaffold:nav")
        .map(|l| l.chars().take_while(|c| c.is_whitespace()).collect())
        .unwrap_or_else(|| "  ".to_string());
    g.inject(
        SIDEBAR,
        &format!("{indent}{NAV_ITEM}"),
        Anchor::BeforeLine("// scaffold:nav"),
    );
    g.named_import(SIDEBAR, "@/routes", "chats");
    g.named_import(SIDEBAR, "lucide-react", "MessagesSquare");
}
