//! `rust-llm generate public_chat`: a chat page anyone can try without signing in, for demos.
//! RubyLLM has no counterpart (its chat UI is for signed-in users of the app).
//!
//! | Writes | |
//! |---|---|
//! | `src/controllers/public_chat.rs` | `GET /chat` (the page), `POST /chat/messages` (the reply, streamed as Server-Sent Events in the response itself), `DELETE /chat` (start over); the limits (`PUBLIC_CHAT_*`) |
//! | `frontend/pages/public_chat/show.tsx` | the page: shadcn/ui cards and a textarea; reads the streamed reply with `fetch` |
//! | `tests/requests/public_chat.rs` | streaming, isolation between guests, and each limit |
//! | `tests/requests/anthropic_stub.rs` | a local stand-in for Anthropic's streaming API (shared with `chat_ui`) |
//!
//! The conversation is kept in memory for the browser session, under an unguessable id in an
//! HttpOnly cookie; nothing is written to the database. Because the reply comes back in the
//! response to the guest's own POST, there is no shared channel one guest could read another's
//! stream from, and the kit's `/live` (signed-in users only) is left alone.

use crate::{Anchor, Generator, chat_ui};

const CONTROLLER: &str = include_str!("../templates/public_chat/controller.rs");
const PAGE: &str = include_str!("../templates/public_chat/page.tsx");
const TEST: &str = include_str!("../templates/public_chat/test.rs");
const STUB: &str = include_str!("../templates/chat_ui/tests/anthropic_stub.rs");

const PATHS: &str = r#"/// The public chat (`rust-llm generate public_chat`): no sign-in.
pub const PUBLIC_CHAT: &str = "/chat";
pub const PUBLIC_CHAT_MESSAGES: &str = "/chat/messages";
"#;

// rustfmt-shaped, like the kit's own entries.
const ROUTES: &str = r#"        route(
            "public_chat.show",
            Get,
            PUBLIC_CHAT,
            ts("PublicChatController", "publicChat", "show", Some("chat")),
        ),
        route(
            "public_chat.destroy",
            Delete,
            PUBLIC_CHAT,
            ts("PublicChatController", "publicChat", "destroy", None),
        ),
        route(
            "public_chat.messages",
            Post,
            PUBLIC_CHAT_MESSAGES,
            ts("PublicChatController", "publicChat", "messages", None),
        ),"#;

/// `rust-llm generate public_chat`.
pub fn generate(g: &mut Generator) -> Result<(), String> {
    if !g.exists("src/route_table.rs") || !g.exists("src/controllers/rate_limit.rs") {
        return Err("public_chat targets the Loco + Inertia starter kit (github.com/cole-robertson/inertia-rust-starter-kit): src/route_table.rs and src/controllers/rate_limit.rs not found.".into());
    }
    if !g
        .read("Cargo.toml")
        .is_some_and(|c| c.contains("rust_llm ="))
    {
        return Err("rust_llm is not a dependency yet. Run the install generator first:\n  rust-llm generate install".into());
    }
    let pkg = chat_ui::package_name(g)?;
    g.file("src/controllers/public_chat.rs", CONTROLLER);
    g.file("frontend/pages/public_chat/show.tsx", PAGE);
    g.file(
        "tests/requests/public_chat.rs",
        &crate::render(TEST, &[("pkg_name", &pkg)]),
    );
    g.file("tests/requests/anthropic_stub.rs", STUB);
    g.inject(
        "src/controllers/mod.rs",
        "pub mod public_chat;",
        Anchor::Sorted("pub mod "),
    );
    g.inject(
        "src/app.rs",
        "            .add_route(controllers::public_chat::routes())",
        Anchor::After("AppRoutes::empty()"),
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
    reserve_slug(g);
    chat_ui::test_modules(g, &["anthropic_stub", "public_chat"]);

    g.note("\n  Public chat installed at /chat (no sign-in)!");
    g.note("  Every reply calls the model with your provider key. Limits (env, defaults):");
    g.note("    PUBLIC_CHAT_IP_MESSAGES=20 and PUBLIC_CHAT_SESSION_MESSAGES=10 per PUBLIC_CHAT_WINDOW_SECS=600,");
    g.note("    PUBLIC_CHAT_MAX_INPUT_CHARS=4000, PUBLIC_CHAT_MAX_OUTPUT_TOKENS=1024, PUBLIC_CHAT_MAX_TURNS=20,");
    g.note("    PUBLIC_CHAT_MODEL (else RUST_LLM_DEFAULT_MODEL), PUBLIC_CHAT_PROVIDER, PUBLIC_CHAT_INSTRUCTIONS.");
    g.note("  Per-IP limits need the real client address: behind a proxy, enable `remote_ip` (production does).");
    g.note("\n  Next steps:");
    g.note("     1. cargo loco task routes:generate");
    g.note("     2. cargo fmt --all");
    g.note("     3. cargo test --test mod requests::public_chat");
    g.note("     4. Start the app and visit /chat");
    Ok(())
}

/// `chat` becomes a reserved account slug, so no account's `/{slug}` pages hide behind `/chat`.
fn reserve_slug(g: &mut Generator) {
    const ACCOUNTS: &str = "src/models/accounts.rs";
    if g.read(ACCOUNTS).is_some_and(|s| s.contains("RESERVED_SLUGS")) {
        g.inject(
            ACCOUNTS,
            "    \"chat\",",
            Anchor::After("pub const RESERVED_SLUGS"),
        );
    }
}
