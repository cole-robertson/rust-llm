# Instrumentation and Workflows

Observe model operations, provider attempts, tool calls, workflows, and model refreshes. This
follows RubyLLM's `docs/_advanced/instrumentation.md`.

## Instrumenters

RubyLLM sends events to `config.instrumenter`, an object with `instrument(name, payload)`. In
RustLLM the instrumenter is anything that implements `rust_llm::Instrumenter`, including a closure
taking the event name, its payload as JSON, and how long the instrumented work took:

```ruby
RubyLLM.configure { |config| config.instrumenter = ActiveSupport::Notifications }
ActiveSupport::Notifications.subscribe('chat.ruby_llm') { |event| ... }
```

```rust,no_run
use std::sync::Arc;
use std::time::Duration;
use serde_json::{Map, Value};

rust_llm::configure(|config| {
    config.instrumenter = Some(Arc::new(|name: &str, payload: &Map<String, Value>, duration: Option<Duration>| {
        if name == "chat.rust_llm" {
            println!("{} {} {:?} cost={}", payload["provider"], payload["model"], duration, payload["cost"]["total"]);
        }
    }));
});
```

An event is delivered when its work finishes, so nested events (a request inside a chat) arrive
before the event around them. When the work fails, the payload gets `exception: [kind, message]`,
as `ActiveSupport::Notifications` adds `:exception`. Set `instrumenter` on a configuration you pass
to `Chat::with_config` or a `context` to instrument only that work.

Every event also runs inside a `tracing` span named `rust_llm` with an `event` field, and logs its
duration at `debug` level, so a `tracing` subscriber sees the work without an instrumenter.

## Event Names

RustLLM keeps RubyLLM's event names with `.rust_llm` in place of `.ruby_llm`:

| RubyLLM | RustLLM | Emitted by |
|---|---|---|
| `workflow.ruby_llm` | `workflow.rust_llm` | `rust_llm::workflow`, `Workflow::run` |
| `workflow_step.ruby_llm` | `workflow_step.rust_llm` | `Workflow::step` |
| `request.ruby_llm` | `request.rust_llm` | every HTTP call through a provider connection, retries included |
| `usage.ruby_llm` | `usage.rust_llm` | every finished provider attempt (retries, failures, cancellations), including one-shot operations and results collected from a batch |
| `chat.ruby_llm` | `chat.rust_llm` | each completion (`ask`, `complete`, `step`, `generate`) |
| `compaction.ruby_llm` | `compaction.rust_llm` | `Chat::compact` |
| `tool_call.ruby_llm` | `tool_call.rust_llm` | each local tool execution |
| `embedding.ruby_llm` | `embedding.rust_llm` | `rust_llm::embed` |
| `image.ruby_llm` | `image.rust_llm` | `rust_llm::paint` |
| `speech.ruby_llm` | `speech.rust_llm` | `rust_llm::speak`, `speak_stream` |
| `transcription.ruby_llm` | `transcription.rust_llm` | `rust_llm::transcribe`, `transcribe_stream` |
| `moderation.ruby_llm` | `moderation.rust_llm` | `rust_llm::moderate` |
| `ocr.ruby_llm` | `ocr.rust_llm` | `rust_llm::ocr` |
| `rerank.ruby_llm` | `rerank.rust_llm` | `rust_llm::rerank` |
| `tokenization.ruby_llm` | `tokenization.rust_llm` | `rust_llm::tokenize` |
| `video.ruby_llm` | `video.rust_llm` | `rust_llm::animate` |
| `video_job.ruby_llm` | `video_job.rust_llm` | `rust_llm::animate_later` |
| `judgment.ruby_llm` | `judgment.rust_llm` | `rust_llm::judge`, `Judge::judge` |
| `batch.ruby_llm` | `batch.rust_llm` | `rust_llm::batch` submission |
| `models.refresh.ruby_llm` | `models.refresh.rust_llm` | `rust_llm::models::refresh` |

Payloads carry RubyLLM's keys. Value objects become their `to_h` (`tokens`, `cost`, `response`,
`input_messages`), and Ruby objects without a JSON form (`chat`, `tool`, `model_info`) are left out.
`tokens` and `cost` are always present on chat, embedding, image, and other billed one-shot
events; their fields are missing when the provider did not report them.

The one-shot options structs (`EmbedOptions`, `PaintOptions`, `SpeakOptions`,
`TranscribeOptions`, `ModerateOptions`, `OcrOptions`, `RerankOptions`, `AnimateOptions`,
`JudgeOptions`) take RubyLLM's `metadata:` as `metadata: Option<Value>`. It is added to the event
payload and never sent to the provider:

```rust,no_run
use serde_json::json;

# async fn run() -> rust_llm::Result<()> {
let options = rust_llm::EmbedOptions { metadata: Some(json!({ "document_id": 42 })), ..Default::default() };
rust_llm::embed("Rust", options).await?;
# Ok(()) }
```

Payloads include message content, tool arguments, and provider responses. Only export or log them
when your application policy allows it.

## Workflows and Steps

`rust_llm::workflow` groups the events of a piece of work. Each step adds its own timing and name:

```ruby
RubyLLM.workflow("Summarize meeting", id: "meeting-42") do |workflow|
  workflow.step("Summarize") { RubyLLM.chat.ask(transcript).content }
end
```

```rust,no_run
# async fn run(transcript: String) -> rust_llm::Result<()> {
rust_llm::workflow("Summarize meeting", Some("meeting-42"), None, |wf| async move {
    wf.step("Summarize", None, async { rust_llm::chat()?.ask(transcript).await }).await
})
.await?;
# Ok(()) }
```

Every event inside receives `workflow_id` and `workflow_name`, plus `workflow_step_id` and
`workflow_step_name` inside a step, `workflow_step_parent_id` for a nested step, and
`workflow_metadata` when you pass `metadata`. A workflow inside another keeps its own identity and
records `workflow_parent_id` and `workflow_parent_step_id`. The id defaults to a UUID.

RubyLLM keeps this context in a thread-local; RustLLM keeps it in a tokio task-local that follows
the future. A task you `tokio::spawn` starts outside it, so start its step inside the spawned
future, as RubyLLM's guide says to do for `Async` tasks.

Your own code can emit events that carry the workflow with `rust_llm::instrument`:

```rust,no_run
# async fn run() -> rust_llm::Result<()> {
let config = rust_llm::config();
let payload = serde_json::json!({ "document": 42 }).as_object().cloned().unwrap_or_default();
rust_llm::instrument(&config, "index.my_app", payload, async { Ok(()) }).await?;
# Ok(()) }
```

## Differences from RubyLLM

- Event names end in `.rust_llm` instead of `.ruby_llm`.
- There is no `ActiveSupport::Notifications`: write an instrumenter that forwards to your
  observability stack, or subscribe to the `tracing` spans.
- Hosted research (`research_job.ruby_llm`) belongs to Vertex AI, which is not ported.
