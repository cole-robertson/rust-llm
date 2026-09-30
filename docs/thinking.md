# Extended Thinking

Give reasoning models room to deliberate, and read their thinking.

## Controlling Thinking

`with_thinking` takes a `ThinkingConfig`:

| RubyLLM | RustLLM |
|---|---|
| `with_thinking` | `with_thinking(ThinkingConfig::on())` |
| `with_thinking(false)` | `with_thinking(ThinkingConfig::off())` |
| `with_thinking(effort: :high)` | `with_thinking(ThinkingConfig::effort("high"))` |
| `with_thinking(budget: 10_000)` | `with_thinking(ThinkingConfig::budget(10_000))` |
| `with_thinking(effort: :none)` | `with_thinking(ThinkingConfig::effort("none"))` |
| `with_thinking(effort: :high, display: :summarized)` | `ThinkingConfig::effort("high").with_display(ThinkingDisplay::Summarized)` |

```ruby
chat = RubyLLM.chat(model: 'claude-opus-4-5').with_thinking(effort: :high)
response = chat.ask("What is 15 * 23?")
response.thinking&.text
response.thinking&.signature
```

```rust,no_run
use rust_llm::ThinkingConfig;

# async fn run() -> rust_llm::Result<()> {
let mut chat = rust_llm::chat_with("claude-opus-4-5")?.with_thinking(ThinkingConfig::effort("high"));
let response = chat.ask("What is 15 * 23?").await?;

if let Some(thinking) = &response.thinking {
    println!("{:?}", thinking.text);      // the visible thinking
    println!("{:?}", thinking.signature); // opaque provider state, replayed on later turns
}
# Ok(()) }
```

For both an effort and a budget, set the public field:

```rust,no_run
use rust_llm::ThinkingConfig;

let mut config = ThinkingConfig::effort("high");
config.budget = Some(8000);
```

`ThinkingConfig::on()` and `off()` resolve against the model's registry entry when the request is
built: a default effort, then an explicit budget, then a provider toggle, then `medium` effort,
then the smallest budget the model accepts. A model that exposes no matching control fails with
`Error::Argument` ("... does not expose thinking controls in the model registry"). Effort and
budget values are sent exactly as given.

When a chat switches provider (`with_model` or a fallback), earlier turns are replayed without
their thinking, since no provider can verify another's signature.

## Display

```rust,no_run
use rust_llm::{ThinkingConfig, ThinkingDisplay};

# fn run() -> rust_llm::Result<()> {
let chat = rust_llm::chat()?.with_thinking(ThinkingConfig::effort("high").with_display(ThinkingDisplay::Summarized));
# Ok(()) }
```

`Summarized`, `Omitted`, or `Full`. OpenAI only returns thinking text with `Summarized`.

## Streaming with Thinking

```rust,no_run
use rust_llm::{ThinkingConfig, ThinkingDisplay};

# async fn run() -> rust_llm::Result<()> {
let mut chat = rust_llm::chat_with("claude-opus-4-5")?
    .with_thinking(ThinkingConfig::effort("medium").with_display(ThinkingDisplay::Summarized));
chat.ask_stream("Solve step by step: what is 127 * 43?", |chunk| {
    if let Some(text) = chunk.thinking.as_ref().and_then(|t| t.text.as_deref()) {
        print!("{text}");
    }
    print!("{}", chunk.content());
})
.await?;
# Ok(()) }
```

## Thinking Tokens

`response.tokens().thinking` reports reasoning tokens separately. `tokens.output` is already the
billable output, so do not add them. [Persistence with Loco](persistence-loco.md) stores
`thinking_text` and `thinking_signature` on each message row.
