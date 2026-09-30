//! Port of `lib/generators/ruby_llm/chat_ui/chat_ui_generator.rb` for Inertia + React +
//! shadcn/ui instead of ERB + Turbo.
//!
//! | RubyLLM | here |
//! |---|---|
//! | `app/controllers/{chats,messages,models}_controller.rb` | `src/controllers/{chats,messages,models}.rs` |
//! | `app/jobs/chat_response_job.rb` | `src/workers/chat_response.rs` (Loco worker) |
//! | `app/views/chats/*`, `app/views/models/*` | `frontend/pages/{chats,models}/*.tsx` |
//! | `app/views/messages/_<role>`, `tool_calls/_default`, `tool_results/_default`, `_error` | `frontend/components/messages/*` |
//! | `form.select :model` | `frontend/components/messages/model-picker.tsx` (shadcn `Select`) |
//! | `MessagesHelper#tool_call_partial` | `import.meta.glob` lookup in `message-list.tsx` |
//! | `config/routes.rb` resources | `src/route_table.rs` constants and routes |
//! | `broadcasts_to` + `broadcast_append_chunk` (Turbo streaming) | polling (see below) |
//!
//! Streaming: RubyLLM appends each chunk over Turbo Streams. This port does not stream tokens.
//! The worker runs `ChatRecord::complete`, which persists every message as it lands, and the chat
//! page polls with Inertia's `usePoll` (a partial reload of `messages`, once a second) while the
//! last message still waits for the model.
//!
//! One UI variant (shadcn/ui) replaces RubyLLM's `tailwind` and `scaffold` variants.

use crate::{Anchor, Generator};

macro_rules! templates {
    ($($path:literal => $source:literal),* $(,)?) => {
        const FILES: &[(&str, &str)] = &[$(($path, include_str!(concat!("../templates/chat_ui/", $source)))),*];
    };
}

templates! {
    "src/controllers/chats.rs" => "controllers/chats.rs",
    "src/controllers/messages.rs" => "controllers/messages.rs",
    "src/controllers/models.rs" => "controllers/models.rs",
    "src/workers/chat_response.rs" => "workers/chat_response.rs",
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

const PATHS: &str = r#"pub const CHATS: &str = "/chats";
pub const NEW_CHAT: &str = "/chats/new";
pub const CHAT: &str = "/chats/{id}";
pub const CHAT_MESSAGES: &str = "/chats/{chat_id}/messages";
pub const MODELS: &str = "/models";
pub const MODELS_REFRESH: &str = "/models/refresh";
pub const MODEL: &str = "/models/{id}";

/// `/chats/{id}` with the id filled in.
#[must_use]
pub fn chat_path(id: i32) -> String {
    CHAT.replace("{id}", &id.to_string())
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

const NAV_ITEM: &str = r#"  { title: "Chats", href: chats.index().url, icon: MessagesSquare },"#;

/// `rust-llm generate chat_ui`.
pub fn generate(g: &mut Generator) -> Result<(), String> {
    // `check_model_exists`
    if !g.exists("src/models/messages.rs") {
        return Err("Model file not found: src/models/messages.rs\n\nPlease run the install generator first:\n  rust-llm generate install".into());
    }
    if !g.exists("src/route_table.rs") {
        return Err("src/route_table.rs not found: chat_ui targets the Loco + Inertia starter kit's route table.".into());
    }
    for (path, content) in FILES {
        g.file(path, content);
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
    link_sidebar(g);

    g.note("\n  Chat UI installed!");
    g.note("  Streaming: not token-by-token. The worker persists each message; the chat page polls every second while a reply is pending.");
    g.note("\n  Next steps:");
    g.note("     1. cargo loco task routes:generate   (writes frontend/routes/*.ts for the new routes)");
    g.note("     2. cargo fmt --all");
    g.note("     3. Start the app and visit /chats");
    Ok(())
}

/// A "Chats" link in the kit's sidebar, above `// scaffold:nav`.
fn link_sidebar(g: &mut Generator) {
    const SIDEBAR: &str = "frontend/components/app-sidebar.tsx";
    if !g.exists(SIDEBAR) {
        return g.note(format!(
            "  {SIDEBAR} not found; link /chats from your navigation yourself."
        ));
    }
    g.inject(SIDEBAR, NAV_ITEM, Anchor::Before("// scaffold:nav"));
    g.named_import(SIDEBAR, "@/routes", "chats");
    g.named_import(SIDEBAR, "lucide-react", "MessagesSquare");
}
