# Chat

Start a conversation, set instructions, switch models, control the request, and read the
response.

## Starting a Conversation

`rust_llm::chat()` starts a conversation with the configured default model (`RubyLLM.chat`).
`ask` adds a user message, runs the conversation until the model answers, and returns that
answer as a `Message`. `say` is an alias.

```ruby
chat = RubyLLM.chat
response = chat.ask "Explain 'Convention over Configuration' in Rails."
puts response.content
```

```rust,no_run
# async fn run() -> rust_llm::Result<()> {
let mut chat = rust_llm::chat()?;
let response = chat.ask("Explain 'Convention over Configuration' in Loco.").await?;
println!("{}", response.content()); // "" when the model returned no text
# Ok(()) }
```

When the model calls [tools](tools.md), `ask` runs them and asks again until the model answers
without a tool call.

## Continuing the Conversation

```rust,no_run
# async fn run(mut chat: rust_llm::Chat) -> rust_llm::Result<()> {
chat.ask("Can you give a specific example?").await?;

for message in chat.messages() {
    println!("[{}] {}", message.role.as_str(), message.content());
}
# Ok(()) }
```

`chat.messages()` is the transcript sent with every request.

## Instructions

```ruby
chat.with_instructions "Explain Ruby concepts to a beginner."
chat.with_instructions "Use exactly one short paragraph.", append: true
chat.with_instructions(nil)
```

```rust,no_run
# fn run() -> rust_llm::Result<()> {
// Builder style replaces earlier system messages.
let mut chat = rust_llm::chat()?.with_instructions("Explain Rust concepts to a beginner.");

// `set_instructions(text, append, cache_until_here)` covers the keyword forms.
chat.set_instructions(Some("Use exactly one short paragraph.".into()), true, false);
chat.set_instructions(None, false, false); // with_instructions(nil)
# Ok(()) }
```

## Choosing a Model

```ruby
RubyLLM.chat(model: 'claude-haiku-4-5')
RubyLLM.chat(model: 'claude-haiku-4-5', provider: :anthropic)
chat.with_model('gemini-2.5-flash')
RubyLLM.chat(model: 'llama3', provider: :ollama, assume_model_exists: true)
```

```rust,no_run
use rust_llm::Chat;

# fn run() -> rust_llm::Result<()> {
let chat = rust_llm::chat_with("claude-haiku-4-5")?;
let chat = Chat::new(Some("claude-haiku-4-5"), Some("anthropic"))?;
let chat = chat.with_model("gemini-2.5-flash", None)?;
let chat = rust_llm::chat()?.with_assumed_model("my-finetune", "openai")?;
# Ok(()) }
```

Models resolve through the bundled registry (`rust_llm::models()`, the same `models.json` and
`aliases.json` RubyLLM ships). A bare id available from several providers is resolved with
RubyLLM's provider preference, and an unknown id fails with `Error::ModelNotFound`. When you name
Ollama, GPUStack, Ollama Cloud, or Hetzner as the provider, any model id is accepted.

```rust,no_run
# fn run() -> rust_llm::Result<()> {
let model = rust_llm::models().find("claude-haiku-4-5", None)?;
let provider = &model.provider;               // "anthropic"
let window = model.context_window;            // Option<i64>
let structured = model.supports("structured_output");
let chat_models = rust_llm::models().chat_models();
# Ok(()) }
```

## Temperature and Output Length

```ruby
RubyLLM.chat.with_temperature(0.2).with_max_output_tokens(200)
```

```rust,no_run
# fn run() -> rust_llm::Result<()> {
let chat = rust_llm::chat()?.with_temperature(0.2).with_max_output_tokens(200);
# Ok(()) }
```

The value goes on the wire unchanged. Nothing is sent until you set one. A model that rejects a
temperature answers with `Error::BadRequest`.

## Provider Options, Protocols, Headers, and Hooks

```ruby
chat.with_provider_options(text: { format: { type: 'json_object' } })
RubyLLM.chat(model: 'gpt-5.6', protocol: :chat_completions)
chat.with_headers('X-Request-Id' => 'abc')
chat.before_request { |payload| payload[:metadata] = { team: "support" } }
chat.render
```

```rust,no_run
use rust_llm::ProtocolName;
use serde_json::json;

# fn run() -> rust_llm::Result<()> {
let mut chat = rust_llm::chat()?
    .with_provider_options(json!({ "text": { "format": { "type": "json_object" } } }))
    .with_protocol(ProtocolName::ChatCompletions)
    .with_headers([("X-Request-Id".to_string(), "abc".to_string())])
    .before_request(|payload| {
        payload["metadata"] = json!({ "team": "support" });
    });

chat.ask_later("Hi")?;
let payload = chat.render()?; // the request body, without sending it
# Ok(()) }
```

`with_provider_options` is deep-merged into every rendered request and can override anything in
it. `with_params` is the deprecated 1.x name. `render` returns the payload with provider options
and `before_request` hooks applied, exactly as it would be sent.

## Callbacks

```rust,no_run
# fn run() -> rust_llm::Result<()> {
let chat = rust_llm::chat()?
    .before_message(|| print!("Assistant > "))
    .after_message(|message| println!("{}", message.content()));
# Ok(()) }
```

Callbacks are additive. `before_tool_call`, `after_tool_result`, and `after_tool_progress` are in
[Tools](tools.md); `before_fallback` and `after_fallback` are in
[Errors and Retries](errors-and-retries.md).

## Raw Responses and Finish Reasons

```ruby
response.raw.body
response.max_tokens?
response.finish_reason # => :stop
```

```rust,no_run
use rust_llm::FinishReason;

# async fn run(mut chat: rust_llm::Chat) -> rust_llm::Result<()> {
let response = chat.ask("Summarize this in one paragraph").await?;

if let Some(raw) = &response.raw {
    println!("{} {}", raw.status, raw.body);
}

if response.is_max_tokens() {
    println!("hit the token limit");
} else if response.is_content_filtered() {
    println!("filtered");
} else if response.is_tool_call_stop() {
    println!("the model requested a tool call");
} else if response.is_stopped() {
    println!("finished normally");
}
let stopped = response.finish_reason == Some(FinishReason::Stop);
# Ok(()) }
```

`FinishReason` normalizes each provider's value to `Stop`, `MaxTokens`, `ToolCalls`,
`ContentFilter`, `PauseTurn`, or `Other(String)`.

## Driving the Loop Yourself

```ruby
chat.ask_later("Check the weather in every capital")
10.times { chat.step; break if chat.complete? }
```

```rust,no_run
# async fn run(mut chat: rust_llm::Chat) -> rust_llm::Result<()> {
chat.ask_later("Check the weather in every capital")?;
for _ in 0..10 {
    chat.step().await?; // run pending tools, or generate the next response
    if chat.is_complete() {
        break;
    }
}
# Ok(()) }
```

`complete()` runs until `is_complete()` or until the chat waits on an approval. `generate()` makes
one model call without running tools, and `run_tools()` runs the pending tool calls.

## Replacing the Transcript

```ruby
chat.messages = chat.messages.last(4)
```

```rust,no_run
# fn run(mut chat: rust_llm::Chat) {
let keep = chat.messages().iter().rev().take(4).rev().cloned().collect();
chat.set_messages(keep);
chat.add_message(rust_llm::Message::assistant("(earlier context trimmed)"));
# }
```

## Prompt Caching Boundaries

```rust,no_run
# fn run(mut chat: rust_llm::Chat) -> rust_llm::Result<()> {
chat.add_message(rust_llm::Message::user("<a long shared document>"));
chat.cache_until_here()?; // marks the last message as a cache boundary
# Ok(()) }
```

## Not ported

- `with_caching`, `with_citations`, `with_compaction`, `compact`, `with_end_user`, `count_tokens`.
  Citations that a provider returns anyway are still parsed into `message.citations`.
- `with_context`: build the chat with `Chat::with_config` instead.
- `with_model(nil)` is `with_default_model()`; `with_temperature(None)`, `with_max_output_tokens(None)`, `with_provider_options(Value::Null)`, and `with_headers([])` clear a setting.
- Prompt rendering from `app/prompts` (`RubyLLM.render_prompt`).
- Perplexity's `router_chat_completions` protocol. `ProtocolName` has `ChatCompletions`,
  `Responses`, `Anthropic`, and `Gemini`.
- Instrumentation events.
