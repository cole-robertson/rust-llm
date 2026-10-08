# Generators (`rust-llm` CLI)

`rust_llm_cli` ports RubyLLM's Rails generators to a `rust-llm` command for Loco + Inertia + React
(shadcn/ui) apps.

## Installing the CLI

```sh
cargo install rust_llm_cli
# or, from a checkout of this repository: cargo install --path crates/rust_llm_cli
rust-llm --help
```

Run app generators from the root of the Loco app.

## Commands

| RubyLLM | rust-llm |
|---|---|
| `bin/rails g ruby_llm:install` | `rust-llm generate install [--path RUST_LLM_CHECKOUT]` |
| `bin/rails g ruby_llm:chat_ui` | `rust-llm generate chat_ui` |
| (none) | `rust-llm generate public_chat` |
| `bin/rails g ruby_llm:tool Weather` | `rust-llm generate tool Weather` |
| `bin/rails g ruby_llm:agent Support` | `rust-llm generate agent Support` |
| `bin/rails g ruby_llm:schema Product` | `rust-llm generate schema Product` |
| `script/generate-provider NAME` | `rust-llm generate provider NAME [--dialect ...] [--api-base URL] [--dynamic-models] [--destination DIR]` |
| `bin/rails g ruby_llm:upgrade` | `rust-llm generate upgrade` |

`g` is short for `generate`. Every generator takes `--force`. As with Rails generators, existing
files are skipped unless `--force` is given, and injections that are already present are skipped.
Each path is printed with its action (`create`, `identical`, `skip`, `force`, `insert`). An
injection whose anchor is missing is reported with the text to add by hand, and the command exits
non-zero.

## install

```sh
rust-llm generate install --path ../rust-llm
cargo fmt --all && cargo loco db migrate
```

Adds:

- `rust_llm` and `rust_llm_loco` to `Cargo.toml`, and `rust_llm_loco` to `migration/Cargo.toml`,
  at version `2.0.0` from crates.io. Pass `--path` to a checkout to depend on a local copy instead.
- `migration/src/m<timestamp>_create_rust_llm_records.rs`, registered in the migrator. It runs
  `rust_llm_loco::migrations()` (see [Persistence with Loco](persistence-loco.md)).
- `src/models/chats.rs` and `src/models/messages.rs`, re-exporting the `rust_llm_loco` entities
  and `ChatRecord`, with page-prop helpers.
- `src/initializers/rust_llm.rs`, a Loco initializer that calls `rust_llm::configure`, registered
  in `src/app.rs`.
- `src/tools/`, `src/agents/`, `src/schemas/`, and `src/prompts/`.

## chat_ui

```sh
rust-llm generate chat_ui
cargo loco task routes:generate
cargo fmt --all && cargo loco db migrate
```

Targets the [Loco + Inertia starter kit](https://github.com/cole-robertson/inertia-rust-starter-kit)
and its accounts. Adds:

- Inertia + React pages (`frontend/pages/chats/*`, `frontend/pages/models/*`) and message
  components under `frontend/components/messages/` (a shadcn `Select` model picker; per-tool call
  and result components looked up with `import.meta.glob`).
- Controllers (`chats`, `messages`, `models`) under `/{account_slug}`, with routes in
  `src/route_table.rs` and a sidebar link. Chats belong to an account (`chats.account_id`, added by
  a migration), and every query goes through `find_in_account`, so another account's chat is a 404.
- `src/workers/chat_response.rs`, a Loco worker that loads the chat with `to_llm` and runs
  `ChatRecord::complete_stream` (see [Streaming](persistence-loco.md#streaming)).
- `src/channels/chat.rs`, a `ChatChannel` on the kit's live updates. The worker broadcasts
  `message_start`, `chunk` (batched at most every 50 ms), `message_end`, and `error`; the chat page
  shows the reply as it streams and reloads the saved messages when it ends. Only members of the
  chat's account can subscribe.
- Request tests (`tests/requests/chats.rs`) against a local stand-in for Anthropic's streaming API.

To give the chat tools, add them in the worker, where the comment marks the spot.

## public_chat

```sh
rust-llm generate public_chat
cargo loco task routes:generate
```

A chat page at `/chat` that anyone can use without signing in, for a demo or a landing page. A
guest's conversation lives in server memory under an unguessable cookie (no rows), and replies
stream back on the request as Server-Sent Events. Limits come from `PUBLIC_CHAT_*` environment
variables: messages per IP and per conversation in a window (20 and 10 per 10 minutes), input
length (4,000 characters), output tokens per reply (1,024), and turns per conversation (20). The
model is `PUBLIC_CHAT_MODEL`, or RustLLM's `default_model`. RubyLLM has no equivalent. Each reply is
billed to the app's provider key, so keep the limits tight on a public deployment.

## tool

```sh
rust-llm generate tool Weather
```

Writes `src/tools/weather_tool.rs` (a `WeatherTool` implementing `rust_llm::Tool`, called
`weather` by the model) and the chat UI components
`frontend/components/messages/tool_calls/weather.tsx` and `tool_results/weather.tsx`.

## agent

```sh
rust-llm generate agent Support
```

Writes `src/agents/support_agent.rs` (a `SupportAgent` implementing `rust_llm::Agent`) and an
empty `src/prompts/support_agent/instructions.txt`, embedded with `include_str!`. An empty file
means no instructions.

## schema

```sh
rust-llm generate schema Product
```

Writes `src/schemas/product_schema.rs`, an empty `ProductSchema` deriving `schemars::JsonSchema`,
and adds `schemars` to `Cargo.toml`. Use it with `chat.with_schema_for::<ProductSchema>()`.

## provider

```sh
rust-llm generate provider Acme --dialect chat_completions --api-base https://api.acme.ai/v1 --destination ../rust-llm
```

For contributors: run from a RustLLM checkout (or pass `--destination`). Writes
`crates/rust_llm/src/providers/acme.rs` and `crates/rust_llm/tests/provider_acme.rs`, registers the
module, and prints the `Provider` enum wiring to add by hand. `--models-dev-provider KEY` also adds
the provider to `MODELS_DEV_PROVIDER_MAP`; core files that are not there are left alone. Dialects: `chat_completions`,
`responses`, `anthropic`, `gemini`, `ollama`.

## upgrade

Writes `migration/src/m<timestamp>_upgrade_rust_llm_to_2_1.rs` (RubyLLM's
`upgrade_ruby_llm_to_2_1.rb.tt`) and registers it: it adds `rust_llm_mcp_credentials` and
`rust_llm_tool_calls.pending_input` when they are missing, so it is safe on an up-to-date schema.
Then run `cargo loco db migrate`.

## Not Generated

- `ruby_llm:load_models`: models come from the bundled registry. The chat UI's models page has a
  Refresh button (`POST /{account_slug}/models/refresh`); with `rust_llm_loco::ModelStore`
  configured, `rust_llm::models::refresh` saves into `rust_llm_models` (see
  [Persistence with Loco](persistence-loco.md#the-model-registry-in-the-database)).

## Differences from RubyLLM

- The chat UI is one Inertia + React (shadcn/ui) variant that streams over a live channel instead of
  Turbo Streams. There are no ERB `tailwind`/`scaffold` variants. It is scoped to the starter kit's
  accounts, where RubyLLM's is not scoped to a user.
- `--skip-active-storage`, custom model names, and namespaced names (`admin/weather`) are Rails
  generator options; the Loco tables keep RubyLLM's names, and attachments always go to
  `rust_llm_attachments`.
- The agent generator writes a plain `instructions.txt` embedded with `include_str!` instead of an
  ERB template. Agents can also render Jinja prompts at runtime (see [Prompt Templates](prompts.md)).
- The provider generator has no `provider-gem` mode: providers are modules in the `rust_llm` crate.
