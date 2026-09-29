# Configuration

Configure only the providers you use. Everything else has a default.

## Quick Start

```ruby
RubyLLM.configure do |config|
  config.openai_api_key = ENV.fetch('OPENAI_API_KEY')
end
```

```rust,no_run
rust_llm::configure(|config| {
    config.openai_api_key(std::env::var("OPENAI_API_KEY").unwrap_or_default());
});
```

`configure` takes a closure over `&mut Config` and replaces the process-wide configuration. Chats
created afterwards see the new values; chats that already exist keep the configuration they were
built with.

## API Keys

The first time the configuration is read, every provider option below is loaded from the
environment variable of the same name in upper case (`openai_api_key` from `OPENAI_API_KEY`,
`ollama_api_base` from `OLLAMA_API_BASE`, ...). So in most apps exporting the keys is all the
configuration you need.

`Config` has typed setters for the common keys (`openai_api_key`, `anthropic_api_key`,
`gemini_api_key`, `deepseek_api_key`, `openrouter_api_key`, `ollama_api_base`). Every other option
is set by its RubyLLM name with `set`:

```ruby
RubyLLM.configure do |config|
  config.anthropic_api_key = ENV['ANTHROPIC_API_KEY']
  config.mistral_api_key = ENV['MISTRAL_API_KEY']
  config.openai_api_base = "https://my-proxy.example.com/v1"
end
```

```rust,no_run
rust_llm::configure(|config| {
    config.anthropic_api_key("sk-ant-...");
    config.set("mistral_api_key", "...");
    config.set("openai_api_base", "https://my-proxy.example.com/v1");
});
```

The options RustLLM reads:

| Provider | Options |
|---|---|
| OpenAI | `openai_api_key`, `openai_api_base`, `openai_organization_id`, `openai_project_id`, `openai_use_system_role` |
| Anthropic | `anthropic_api_key`, `anthropic_api_base` |
| Gemini | `gemini_api_key`, `gemini_api_base` |
| DeepSeek | `deepseek_api_key`, `deepseek_api_base` |
| Mistral | `mistral_api_key`, `mistral_api_base` |
| OpenRouter | `openrouter_api_key`, `openrouter_api_base`, `openrouter_app_url`, `openrouter_app_name` |
| xAI | `xai_api_key`, `xai_api_base` |
| Perplexity | `perplexity_api_key`, `perplexity_api_base` |
| Ollama | `ollama_api_base`, `ollama_api_key` |
| Ollama Cloud | `ollama_cloud_api_key`, `ollama_cloud_api_base` |
| GPUStack | `gpustack_api_base`, `gpustack_api_key` |
| Hetzner | `hetzner_api_key`, `hetzner_api_base` |
| TypeSafe (judgments) | `typesafe_api_key`, `typesafe_api_base` |

Using a provider whose required option is missing fails with `Error::Configuration`, and the error
message includes the `rust_llm::configure` line to add.

`<provider>_protocol` (for example `config.set("openai_protocol", "chat_completions")`) picks the
wire protocol for every chat on that provider, like RubyLLM's `config.openai_protocol`. It is only
read through `set`/`get`; the environment does not supply it.

## Default Models

```ruby
RubyLLM.configure do |config|
  config.default_model = 'claude-haiku-4-5'
  config.default_embedding_model = 'text-embedding-3-large'
  config.default_image_model = 'gpt-image-2'
  config.default_judgment_model = 'jev-latest'
end
```

```rust,no_run
rust_llm::configure(|config| {
    config.default_model = "claude-haiku-4-5".into();                // rust_llm::chat()
    config.default_embedding_model = "text-embedding-3-large".into(); // rust_llm::embed
    config.default_image_model = "gpt-image-2".into();                // rust_llm::paint
    config.default_judgment_model = "jev-latest".into();              // rust_llm::judge
});
```

Defaults: chat `gpt-5.6`, embeddings `text-embedding-3-small`, images `gpt-image-2`, judgments
`jev-latest`. The `RUST_LLM_DEFAULT_MODEL` environment variable overrides the chat default.

## Connection Settings

```rust,no_run
use std::time::Duration;

rust_llm::configure(|config| {
    config.request_timeout = Duration::from_secs(120); // default 300 s
    config.max_retries = 5;                            // default 3
    config.retry_interval = 0.5;                       // seconds, default 0.1
    config.retry_backoff_factor = 2.0;                 // default 2
    config.retry_interval_randomness = 0.5;            // default 0.5
    config.retry_max_interval = 30.0;                  // seconds, default 30
    config.auto_upload_large_files = true;             // default true, see attachments-and-files.md
});
```

See [Errors and Retries](errors-and-retries.md) for what is retried.

## Isolated Configurations

RubyLLM's `RubyLLM.context { |config| ... }` gives one tenant or agent its own keys. In Rust, clone
the global configuration, change it, and pass the `Arc<Config>` wherever RubyLLM takes `context:`:

```ruby
ctx = RubyLLM.context { |config| config.openai_api_key = tenant.openai_key }
ctx.chat.ask "Hello"
```

```rust,no_run
use std::sync::Arc;
use rust_llm::Chat;

# async fn run(tenant_key: String) -> rust_llm::Result<()> {
let mut config = (*rust_llm::config()).clone();
config.openai_api_key(tenant_key);
let config = Arc::new(config);

let mut chat = Chat::with_config(config.clone(), Some("gpt-5.6"), None, false)?;
chat.ask("Hello").await?;

// The one-shot APIs take it in their options struct.
let options = rust_llm::EmbedOptions { config: Some(config), ..Default::default() };
rust_llm::embed("Hello", options).await?;
# Ok(()) }
```

`PaintOptions`, `UploadOptions`, `FileOptions`, `Judge::with_config`, `Batch::find_with_config`,
`McpBuilder::config`, and `ChatRecord::to_llm_with` take a configuration the same way.

## In a Loco App

`rust-llm generate install` writes `src/initializers/rust_llm.rs`, a Loco initializer that calls
`rust_llm::configure` in `before_run`. Put your settings there. See [Generators](generators.md).

## Not ported

- Azure, Bedrock, Vertex AI, Cohere, ElevenLabs, and Deepgram options.
- `default_video_model`, `default_speech_model`, `default_transcription_model`,
  `default_ocr_model`, `default_moderation_model` (their operations are not ported).
- `model_registry_store`. `model_registry_file` is ported: `rust_llm::models::refresh` saves there
  (default: the platform cache, `~/.cache/rust_llm/models.json` on Linux), and the registry loads
  from it before falling back to the bundled `models.json`.
- `http_proxy`, `faraday_adapter`, `logger`, `log_file`, `log_level`, `log_stream_debug`,
  `deprecation_behavior` (`instrumenter` is ported: see [Instrumentation](instrumentation.md)). Retries and parse failures are logged through the
  `tracing` crate at `debug` level.
- `tool_concurrency` is a field on `Config`, but nothing reads it: tools always run one after
  another.
