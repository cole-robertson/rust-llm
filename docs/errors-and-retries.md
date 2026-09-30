# Errors and Retries

Handle provider failures, fall back to other models, and understand what is retried.

## The Error Type

Each RubyLLM error class is a variant of `rust_llm::Error`. Every fallible call returns
`rust_llm::Result<T>`.

| RubyLLM | `rust_llm::Error` |
|---|---|
| `RubyLLM::Error` (status without a specific class) | `Api(message, response)` |
| `BadRequestError` (400) | `BadRequest(..)` |
| `UnauthorizedError` (401) | `Unauthorized(..)` |
| `PaymentRequiredError` (402) | `PaymentRequired(..)` |
| `ForbiddenError` (403) | `Forbidden(..)` |
| `RateLimitError` (429) | `RateLimit(..)` |
| `ContextLengthExceededError` (400/429 by message) | `ContextLengthExceeded(..)` |
| `ServerError` (500) | `Server(..)` |
| `ServiceUnavailableError` (502-504) | `ServiceUnavailable(..)` |
| `OverloadedError` (529, or a 400 saying so) | `Overloaded(..)` |
| `ToolCallParseError` | `ToolCallParse { message, finish_reason }` |
| `UnsupportedAttachmentError` | `UnsupportedAttachment(..)` |
| `UnsupportedServerToolError` | `UnsupportedServerTool(..)` |
| `ConfigurationError` | `Configuration(..)` |
| `ModelNotFoundError` | `ModelNotFound(..)` |
| `ModelRegistryError` | `ModelRegistry(..)` |
| `PromptNotFoundError` | `PromptNotFound(..)` |
| an ERB error while rendering a prompt | `Prompt(..)` |
| `InvalidToolChoiceError` | `InvalidToolChoice(..)` |
| `PendingToolCallsError` | `PendingToolCalls(..)` |
| `CancelledError` | `Cancelled` |
| `ArgumentError`, `InvalidRoleError` | `Argument(..)` |
| `Faraday::TimeoutError` | `Timeout(..)` |
| `Faraday::ConnectionFailed` | `ConnectionFailed(..)` |
| an exception raised in a tool's `execute` | `Tool(message)` |
| `RubyLLM::MCP::Error` | `Mcp(..)` |
| `RubyLLM::MCP::InputRequiredError` | `McpInputRequired(..)` |
| `JSON::ParserError`, IO errors | `Json(..)`, `Io(..)` |

`error.kind()` returns an `ErrorKind` with one value per HTTP-level class (plus `Other`), which is
what fallbacks match on.

## Handling Errors

```ruby
begin
  chat.ask "Generate a complex report."
rescue RubyLLM::UnauthorizedError
  puts "Check your API key."
rescue RubyLLM::RateLimitError
  puts "Slow down."
rescue RubyLLM::ContextLengthExceededError
  puts "Too much context."
rescue RubyLLM::Error => e
  puts "API error: #{e.message}"
end
```

```rust,no_run
use rust_llm::Error;

# async fn run(mut chat: rust_llm::Chat) {
match chat.ask("Generate a complex report.").await {
    Ok(response) => println!("{}", response.content()),
    Err(Error::Unauthorized(..)) => println!("Check your API key."),
    Err(Error::RateLimit(..)) => println!("Slow down."),
    Err(Error::ContextLengthExceeded(..)) => println!("Too much context."),
    Err(Error::Configuration(message)) => println!("Configuration missing: {message}"),
    Err(e) => println!("API error: {e}"),
}
# }
```

## The Provider's Response

```ruby
rescue RubyLLM::ForbiddenError => e
  e.response&.status
  e.response&.body
```

```rust,no_run
# fn run(error: rust_llm::Error) {
if let Some(response) = error.response() {
    println!("{} {}", response.status, response.body);
}
# }
```

## Errors During Streaming

The chunk closure runs for everything received before the error; then `ask_stream` returns the
error. Keep what you accumulated if you need the partial text.

## Model Fallbacks

```ruby
chat = RubyLLM.chat(model: "gpt-4.1").with_fallbacks("gpt-4.1-mini", "claude-haiku-4-5")
chat.with_fallbacks("gpt-4.1-mini", on: [RubyLLM::RateLimitError, RubyLLM::ServiceUnavailableError])
```

```rust,no_run
use rust_llm::{ErrorKind, Fallback};

# fn run() -> rust_llm::Result<()> {
let chat = rust_llm::chat_with("gpt-4.1")?
    .with_fallbacks(["gpt-4.1-mini".into(), "claude-haiku-4-5".into()]);

// A fallback pinned to a provider, and a custom set of errors that trigger it.
let chat = rust_llm::chat_with("gpt-4.1")?
    .with_fallbacks([Fallback { model: "claude-haiku-4-5".into(), provider: Some("anthropic".into()) }])
    .with_fallback_errors(vec![ErrorKind::RateLimit, ErrorKind::ServiceUnavailable]);
# Ok(()) }
```

Fallbacks are tried in order for one generation; afterwards the chat returns to its own model. By
default they trigger on `RateLimit`, `Server`, `ServiceUnavailable`, `Overloaded`, `Timeout`, and
`ConnectionFailed`. Call `with_fallback_errors` after `with_fallbacks`, which resets the list.

```ruby
chat.before_fallback { |f| puts "#{f.from.id} -> #{f.to.id}: #{f.error.class}" }
chat.after_fallback { |f| puts f.succeeded? }
```

```rust,no_run
# fn run(chat: rust_llm::Chat) {
let chat = chat
    .before_fallback(|f| println!("{} -> {}: {}", f.from, f.to, f.error))
    .after_fallback(|f| println!("attempt {} succeeded: {:?}", f.attempt, f.succeeded));
# }
```

`FallbackAttempt` has `attempt`, `error` (the message), `from` and `to` (model ids), `streaming`,
`chunks_yielded`, and `succeeded`. When a stream already yielded chunks before failing, those
chunks cannot be taken back; `chunks_yielded` tells you so.

## Errors in Tools

Return `Ok(ToolResult::error("..."))` for problems the model can recover from; return `Err(..)`
to stop the conversation with `Error::Tool`. See [Tools](tools.md).

## Automatic Retries

Requests are retried, with exponential backoff and jitter, on:

- timeouts and connection failures,
- `RateLimit` (429),
- `Server`, `ServiceUnavailable`, `Overloaded` (500, 502-504, 529).

`ContextLengthExceeded` is never retried. `Retry-After` and `retry-after-ms` are honored; a
`Retry-After` longer than `retry_max_interval` stops retrying. A stream that already delivered a
chunk is never retried. Requests that create something at the provider (file uploads, batch
submissions) are sent once.

```rust,no_run
rust_llm::configure(|config| {
    config.max_retries = 5;      // default 3
    config.retry_interval = 0.5; // default 0.1 s
});
```

Every attempt, retried or not, is recorded in the usage ledger (see
[Cost and Usage](cost-and-usage.md)).

## Debugging

RustLLM logs retries and unparseable stream data through the `tracing` crate at `debug` level. It
does not log request or response bodies. Use `chat.render()` or `message.raw` to inspect them.

## Differences from RubyLLM

- Agent-level `rescue_from` is a Ruby exception-class DSL: match on `rust_llm::Error` where you
  call the agent.
- Fallbacks name a model id (`"gpt-4.1-mini".into()`) or a `Fallback { model, provider }`, not a
  `RubyLLM::Model` object.
- `RUBYLLM_DEBUG` request and response logging is Ruby-only (Faraday's logger).
