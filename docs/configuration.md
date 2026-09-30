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
    config.default_speech_model = "gpt-4o-mini-tts".into();           // rust_llm::speak
    config.default_transcription_model = "gpt-4o-transcribe".into();  // rust_llm::transcribe
    config.default_moderation_model = "omni-moderation-latest".into(); // rust_llm::moderate
    config.default_ocr_model = "mistral-ocr-latest".into();           // rust_llm::ocr
    config.default_video_model = "grok-imagine-video".into();         // rust_llm::animate
});
```

Defaults: chat `gpt-5.6`, embeddings `text-embedding-3-small`, images `gpt-image-2`, judgments
`jev-latest`, speech `gpt-4o-mini-tts-2025-12-15`, transcription `gpt-transcribe`, moderation
`omni-moderation-latest`, OCR `mistral-ocr-latest`, video `grok-imagine-video-1.5`. The
`RUST_LLM_DEFAULT_MODEL` environment variable overrides the chat default. `rerank` has no default
model.

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
    config.http_proxy = Some("http://proxy.internal:3128".into());
    config.video_generation_timeout = Duration::from_secs(600);      // default 600 s
    config.video_generation_poll_interval = Duration::from_secs(5);  // default 5 s
});
```

`http_proxy` applies to every HTTP request. WebSocket transcription does not support a proxy and
fails with `Error::Argument` when one is set.

See [Errors and Retries](errors-and-retries.md) for what is retried.

## Isolated Configurations

`rust_llm::context` gives one tenant or agent its own configuration, a copy of the global one with
your changes. It has the same entry points as the crate root, and the global configuration is left
alone:

```ruby
ctx = RubyLLM.context { |config| config.openai_api_key = tenant.openai_key }
ctx.chat.ask "Hello"
ctx.embed "Hello"
```

```rust,no_run
# async fn run(tenant_key: String) -> rust_llm::Result<()> {
let ctx = rust_llm::context(|config| {
    config.openai_api_key(tenant_key);
});
ctx.chat(Some("gpt-5.6"), None)?.ask("Hello").await?;
ctx.embed("Hello", Default::default()).await?;
# Ok(()) }
```

`Context` also has `count_tokens`, `tokenize`, `embed_later`, `mcp`, `paint`, `animate`,
`animate_later`, `speak`, `speak_stream`, `moderate`, `judge`, `ocr`, `rerank`, `upload`, and
`download`. For everything else, pass its `Arc<Config>` (`ctx.config().clone()`) where RubyLLM
takes `context:`:

```rust,no_run
use std::sync::Arc;
use rust_llm::Chat;

# async fn run(ctx: rust_llm::Context) -> rust_llm::Result<()> {
let config: Arc<rust_llm::Config> = ctx.config().clone();
let mut chat = Chat::with_config(config.clone(), Some("gpt-5.6"), None, false)?;
chat.ask("Hello").await?;

let options = rust_llm::TranscribeOptions { config: Some(config), ..Default::default() };
rust_llm::transcribe("meeting.wav", options).await?;
# Ok(()) }
```

Every one-shot options struct has a `config` field, and so do `Judge::with_config`,
`Batch::find_with_config`, `McpBuilder::config`, and `ChatRecord::to_llm_with`.
`chat.with_context(Some(&ctx))` moves an existing chat onto a context, and an agent declares one
with `fn context`.

## In a Loco App

`rust-llm generate install` writes `src/initializers/rust_llm.rs`, a Loco initializer that calls
`rust_llm::configure` in `before_run`. Put your settings there. See [Generators](generators.md).

## Other Options

| Option | Purpose |
|---|---|
| `instrumenter` | receives every `*.rust_llm` event (see [Instrumentation](instrumentation.md)) |
| `tool_concurrency` | run a response's tool calls at once (see [Tools](tools.md)) |
| `model_registry_file` | where `rust_llm::models::refresh` saves the registry and where it loads from (default: the platform cache, `~/.cache/rust_llm/models.json` on Linux), before falling back to the bundled `models.json` |
| `model_registry_store` | keep the registry somewhere else, such as `rust_llm_loco::ModelStore` (see [Persistence with Loco](persistence-loco.md)) |
| `prompt_roots` | extra directories for prompt templates (see [Prompt Templates](prompts.md)) |
| `batch_store` | persist submitted batches, such as `rust_llm_loco::BatchStore` (see [Batches](batches.md)) |
| `mcp_credential_store`, `mcp_client_name`, `mcp_client_id` | MCP OAuth (see [MCP](mcp.md#oauth)) |

## Differences from RubyLLM

- The Azure, Bedrock, Vertex AI, Cohere, ElevenLabs, and Deepgram providers are not ported, so
  neither are their options.
- `faraday_adapter`, `logger`, `log_file`, `log_level`, `log_stream_debug`, and
  `deprecation_behavior` are Ruby-only. RustLLM logs retries and parse failures through the
  `tracing` crate at `debug` level; configure a `tracing` subscriber instead.
