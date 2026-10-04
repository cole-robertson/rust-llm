# Contributing to RustLLM

RustLLM is a 1:1 port of [RubyLLM](https://github.com/crmne/ruby_llm) 2.0. The rule for every
change: **do what RubyLLM does**. Same public names (with `?` predicates spelled `is_*`), same
behavior, same wire format. Cite the Ruby file you ported in a doc comment
(`/// Port of lib/ruby_llm/...`).

## Layout

| Path | What |
|---|---|
| `crates/rust_llm` | the library |
| `crates/rust_llm_loco` | `acts_as_chat` for Loco/SeaORM |
| `crates/rust_llm_cli` | the `rust-llm generate` CLI and its templates |
| `docs/` | guides; `docs/check` compiles every Rust sample in them as a doctest |
| `bench/` | benchmarks against RubyLLM (`bench/run.sh`, results in `docs/BENCHMARK.md`) |
| `upstream/` | RubyLLM at the ported commit, read-only (not in the published crates) |

## Building and testing

```sh
cargo test --workspace --all-targets   # unit tests, cassette replays, Loco, generators
cargo test --workspace --doc           # doctests, including every sample in docs/
cargo clippy --workspace --all-targets
cargo doc --workspace --no-deps
```

No test touches the network or needs an API key. `cargo run -p rust_llm --example readme` runs
RubyLLM's README live against Anthropic (`ANTHROPIC_API_KEY`).

Maintainers build on a remote box to keep laptops cool: `bin/fw <command>` rsyncs the tree
(minus `target/`, `upstream/`, `.git`) to `rebulk:~/src/tries/rust-llm` (or `$BUILD_HOST`) and runs
the command there, e.g. `bin/fw cargo test -p rust_llm --test cassette_replay`. The sync uses `--delete`, so
don't keep files you care about only in the remote copy.

## Verifying against RubyLLM's cassettes

RubyLLM's specs record real provider traffic as VCR cassettes
(`upstream/spec/fixtures/vcr_cassettes/*.yml`). RustLLM replays the same cassettes. A test
passes only if every request the port sends has a body JSON-equal to the one RubyLLM recorded, at
the same path, in the same count.

1. **Convert** the cassettes your spec uses to JSON (needs Ruby):

   ```sh
   bin/convert-cassettes 'chat_function_calling_anthropic_*'
   ```

   This writes `crates/rust_llm/tests/cassettes/<name>.json` with `!binary` bodies decoded.
   Only convert providers RustLLM implements.

2. **Replay** in a test, with `tests/support/mod.rs`:

   ```rust,ignore
   let cassette = Cassette::start("chat_function_calling_anthropic_claude-haiku-4-5_can_use_tools").await.unwrap();
   let config = config_for(&cassette, "anthropic");       // points the provider at the replay server
   let mut chat = Chat::with_config(config, Some("claude-haiku-4-5"), Some("anthropic"), false)?;
   let answer = chat.ask("What's the weather in Berlin? (52.5200, 13.4050)").await?;
   cassette.assert_all_matched().await;                   // bodies JSON-equal, request count equal
   ```

   Cassette names follow RSpec's `full_description.parameterize(separator: '_')`, and
   `support::cassette_name` builds them. Mirror the assertions of the matching
   `upstream/spec/ruby_llm/*_spec.rb` example. For multipart bodies, assert path, method, and
   fields.

3. **When a replay diverges, fix the library, not the test.** Make the smallest change that
   matches Ruby. Never loosen the comparison. If a field is truly random (ids, uuids), exclude
   only that field and say why in the test.

The cassettes are 64 MB, so they live in the repository but not in the published crates (see the
`include` lists in each `Cargo.toml`). Tests therefore run from a git checkout, not from the
`.crate` files.

## Docs

Guides live in `docs/*.md`. Every ```rust block there is compiled by `cargo test -p rust_llm_docs
--doc`, so samples can't rot. Use `rust,no_run` for anything that would call a provider.

## Releasing

1. Update `CHANGELOG.md` and `[workspace.package] version` in `Cargo.toml`, and the
   `version = "..."` on the path dependencies to `rust_llm`.
2. Tag `vX.Y.Z` and push the tag. `.github/workflows/release.yml` tests, then publishes
   `rust_llm`, `rust_llm_loco`, `rust_llm_cli` in order with crates.io trusted publishing.
3. The very first publish of each crate has to be manual; see the header of `release.yml`.

## License

MIT. By contributing you agree your contribution is licensed under the MIT License. RustLLM
derives from RubyLLM (MIT, Carmine Paolino); keep `UPSTREAM_LICENSE` intact.
