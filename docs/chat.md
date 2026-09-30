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

## Prompt Caching

```ruby
chat.with_caching                 # provider-default prompt caching
chat.with_caching(ttl: "1h")
chat.with_instructions(policy).cache_until_here
chat.with_caching(id: cache)      # a Gemini explicit cache
chat.with_caching(false)
```

```rust,no_run
use serde_json::json;

# async fn run(policy: String) -> rust_llm::Result<()> {
let mut chat = rust_llm::chat_with("claude-haiku-4-5")?
    .with_caching(json!(true))?
    .with_caching(json!({ "ttl": "1h" }))?
    .with_instructions(policy);
chat.cache_until_here()?; // marks the last message as a cache boundary

// Gemini explicit caches: create the resource once, reference it from chats.
let options = rust_llm::CacheOptions { model: "gemini-2.5-flash", ttl: Some(rust_llm::Ttl::Seconds(3600)), ..Default::default() };
let cache = rust_llm::cache("<the handbook>", options).await?;
let mut gemini = rust_llm::chat_with("gemini-2.5-flash")?.with_caching(json!({ "id": cache.name }))?;
gemini.ask("What does the handbook say about releases?").await?;
# Ok(()) }
```

`with_caching` takes `true`, `false`, or an object (`key`, `ttl`, `mode`, `id`); calling it again
replaces the earlier options. `CachedContent` also has `find`, `renew(ttl)`, and `delete`. Read the
result in `response.tokens().cache_read` and `cache_write`.

## Citations

```ruby
chat = RubyLLM.chat(model: "claude-sonnet-4-6").with_citations
response = chat.ask "Who created Ruby?", with: "facts.txt"
response.citations.first.cited_text
```

```rust,no_run
# async fn run() -> rust_llm::Result<()> {
let mut chat = rust_llm::chat_with("claude-haiku-4-5")?.with_citations(true);
let response = chat.ask_with("Who created Rust?", vec!["facts.txt".into()]).await?;
for citation in &response.citations {
    println!("{:?}: {:?}", citation.title, citation.cited_text);
}
# Ok(()) }
```

Citations from web search and grounding arrive without `with_citations`. A tool can return
`rust_llm::SearchResults` so the model cites its documents (see [Tools](tools.md)).

## End Users and Compaction

```ruby
chat.with_end_user("user-42")
chat.with_compaction(at: 50_000, instructions: "Keep every decision.")
summary = chat.compact
```

```rust,no_run
use serde_json::json;

# async fn run() -> rust_llm::Result<()> {
let mut chat = rust_llm::chat_with("claude-haiku-4-5")?
    .with_end_user(Some("user-42"))
    .with_compaction(json!({ "at": 50_000, "instructions": "Keep every decision." }))?;
chat.ask("Let's go through the whole migration plan.").await?;

// Manual compaction (OpenAI and xAI Responses): later requests send the compacted context,
// while `chat.messages()` keeps every earlier message.
let mut grok = rust_llm::Chat::new(Some("grok-4.3"), Some("xai"))?;
grok.ask("The project codename is Thimble.").await?;
let summary = grok.compact().await?;
println!("{:?}", summary.cost(None).total());
# Ok(()) }
```

`with_end_user` sends an opaque per-user id where the provider has a field for it.
`with_compaction` accepts `true`, `false`, or `at`, `instructions`, and `pause_after`; other keys
fail with `Error::Argument`.

## Counting Tokens

```ruby
RubyLLM.count_tokens("Explain Ruby blocks.", model: "claude-haiku-4-5")
chat.count_tokens("What should I check in a renewal clause?")
```

```rust,no_run
# async fn run(chat: rust_llm::Chat) -> rust_llm::Result<()> {
let count = rust_llm::count_tokens("Explain Rust closures.", Some("claude-haiku-4-5"), None).await?;
let next = chat.count_tokens(Some("What should I check in a renewal clause?")).await?;
let history = chat.count_tokens(None).await?;
# Ok(()) }
```

Anthropic, Gemini, and OpenAI's Responses API count tokens. The count covers instructions, tools,
schema, thinking, and attachments; provider tools, provider options, compaction, and
`before_request` hooks are not included. See [Cost and Usage](cost-and-usage.md) for
`rust_llm::tokenize`.

## Isolated Configuration

```ruby
ctx = RubyLLM.context { |config| config.openai_api_key = tenant.key }
ctx.chat.ask "Hello"
chat.with_context(ctx)
```

```rust,no_run
# async fn run(tenant_key: String, chat: rust_llm::Chat) -> rust_llm::Result<()> {
let ctx = rust_llm::context(|config| {
    config.openai_api_key(tenant_key);
});
ctx.chat(None, None)?.ask("Hello").await?;
let chat = chat.with_context(Some(&ctx))?; // `None` goes back to the global configuration
# Ok(()) }
```

See [Configuration](configuration.md#isolated-configurations).

## Protocols

`ProtocolName` covers every protocol of the ported providers: `ChatCompletions`, `Responses`,
`Anthropic`, `Gemini`, `Interactions` (Gemini's Interactions API), `Conversations` (Mistral's
Conversations API), and `RouterChatCompletions` (Perplexity Router):

```ruby
RubyLLM.chat(model: "perplexity/kimi-k3", provider: :perplexity, protocol: :router_chat_completions)
```

```rust,no_run
use rust_llm::{Chat, ProtocolName};

# fn run() -> rust_llm::Result<()> {
let router = Chat::new(Some("perplexity/kimi-k3"), Some("perplexity"))?
    .with_protocol(ProtocolName::RouterChatCompletions);
let conversations = Chat::new(Some("mistral-small-latest"), Some("mistral"))?
    .with_protocol(ProtocolName::Conversations);
# Ok(()) }
```

## Differences from RubyLLM

- Clearing a setting: `with_model(nil)` is `with_default_model()`; `with_temperature(None)`,
  `with_max_output_tokens(None)`, `with_provider_options(Value::Null)`, and `with_headers([])`
  clear theirs.
- Prompt templates are Jinja, not ERB (see [Prompt Templates](prompts.md)).
- Instrumentation events end in `.rust_llm` (see [Instrumentation](instrumentation.md)).
