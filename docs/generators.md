# Generators (`rust-llm` CLI)

`rust_llm_cli` ports RubyLLM's Rails generators to a `rust-llm` command for Loco + Inertia + React
(shadcn/ui) apps.

## Installing the CLI

```sh
cargo install --path crates/rust_llm_cli   # from a checkout of this repository
rust-llm --help
```

Run app generators from the root of the Loco app.

## Commands

| RubyLLM | rust-llm |
|---|---|
| `bin/rails g ruby_llm:install` | `rust-llm generate install [--path RUST_LLM_CHECKOUT]` |
| `bin/rails g ruby_llm:chat_ui` | `rust-llm generate chat_ui` |
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

- `rust_llm` and `rust_llm_loco` to `Cargo.toml`, and `rust_llm_loco` to `migration/Cargo.toml`.
  The crates are not on crates.io yet, so pass `--path` to a checkout (without it the generator
  writes version `2.0.0` and prints a note).
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
```

Adds Inertia + React pages (`frontend/pages/chats/*`, `frontend/pages/models/*`), message
components under `frontend/components/messages/` (a shadcn `Select` model picker; per-tool call
and result components looked up with `import.meta.glob`), controllers (`chats`, `messages`,
`models`), routes in `src/route_table.rs`, a sidebar link, and a Loco worker,
`src/workers/chat_response.rs`, that loads the chat with `to_llm` and runs `ChatRecord::complete`.

The page does not stream tokens. The worker saves each message as it is produced, and the chat page
polls (Inertia `usePoll`, once a second) while a reply is pending. To give the chat tools, add them
in the worker, where the comment marks the spot.

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
module, and prints the `Provider` enum wiring to add by hand. Dialects: `chat_completions`,
`responses`, `anthropic`, `gemini`, `ollama`.

## upgrade

Writes nothing and says so: `install` already creates the current schema.

## Not ported

- `--skip-active-storage`, custom model names, and namespaced generator names (`admin/weather`).
- `ruby_llm:load_models` and `POST /models/refresh`: models come from the bundled registry.
- Turbo Streams token streaming, and the `tailwind`/`scaffold` UI variants (there is one shadcn
  variant).
- ERB prompt templates (`instructions.txt.erb`).
- The provider generator's `provider-gem` mode.
