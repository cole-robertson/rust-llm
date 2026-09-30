# Cost and Usage

Read token counts and costs from responses, chats, and one-shot results.

## Reading Tokens and Costs

```ruby
response = chat.ask "Explain Ruby blocks."
response.tokens.input
response.tokens.output
response.cost.total
chat.cost.total
```

```rust,no_run
# async fn run() -> rust_llm::Result<()> {
let mut chat = rust_llm::chat()?;
let response = chat.ask("Explain Rust closures.").await?;

let tokens = response.tokens();        // rust_llm::Tokens
let cost = response.cost(None);        // rust_llm::Cost
println!("{:?} in, {:?} out, ${:?}", tokens.input, tokens.output, cost.total());

let chat_total = chat.cost().total(); // every attempt in the chat
# Ok(()) }
```

Unknown values are `None`, so a missing price never looks like a free request. `cost.total()` is
`None` when any component with tokens has no price.

`Message::cost(Some(&model))` prices the same tokens against another model instead of the one that
answered.

The same readers exist on `Embedding` (`tokens()`, `cost()`), `Image`, `Judgment`, `Speech`,
`Transcription`, `Moderation`, `Ocr`, `Rerank`, `BatchResult`, and `Batch` (`tokens().await`,
`cost().await`, at batch rates).

## Token Buckets

| Field | Counts |
|---|---|
| `tokens.input` | standard input, excluding cache reads and writes |
| `tokens.output` | billable output, including thinking when billed as output |
| `tokens.cache_read` | input served from the prompt cache |
| `tokens.cache_write` | input written to the prompt cache |
| `tokens.thinking` | thinking tokens, when reported; already in `output`, don't add them |

`Cost` has the matching `input`, `output`, `cache_read`, `cache_write`, and `thinking` amounts in
USD, plus `total()` and `missing()` (the components that had tokens but no price). When the
provider reports its own price, `total()` uses it.

## The Usage Ledger

A request can be retried, fall back to another model, or be cancelled mid-stream, so the
transcript is not an accounting record. Every physical attempt becomes a `UsageEntry`
(`operation`, `provider`, `model`, `status`, `tokens`, `cost`), with status `Succeeded`, `Failed`,
or `Cancelled`:

```rust,no_run
# fn run(chat: &rust_llm::Chat, response: &rust_llm::Message) {
for entry in chat.usage_entries() {
    println!("{} {} {:?} {:?}", entry.provider, entry.model, entry.status, entry.cost.total());
}
let attempts_for_this_answer = &response.usage_entries; // retries and fallbacks that produced it
# }
```

`response.tokens()` and `cost()` aggregate every attempt behind that answer. `chat.tokens()` and
`chat.cost()` aggregate the whole ledger, including failed retries and cancelled attempts. A
failed attempt the provider refused (4xx) or that never reached it is billed as zero; any other
failed attempt has unknown usage, which makes the chat's `cost().total()` `None` rather than too
low.

## Pricing Usage Yourself

```ruby
model = RubyLLM.models.find('gpt-5.6')
model.cost_for(response.tokens).total
RubyLLM::Cost.aggregate(messages.map(&:cost)).total
```

```rust,no_run
use rust_llm::Cost;

# fn run(response: &rust_llm::Message, messages: &[rust_llm::Message]) -> rust_llm::Result<()> {
let model = rust_llm::models().find("gpt-5.6", None)?;
let total = model.cost_for(&response.tokens()).total();

let costs: Vec<Cost> = messages.iter().map(|m| m.cost(None)).collect();
let sum = Cost::aggregate(costs.iter(), true).total();
# Ok(()) }
```

When a model has a long-context tier, prices switch once `input + cache_read + cache_write`
exceeds the registry threshold.

## Persistence

`rust_llm_loco` writes each attempt to `rust_llm_usages` as soon as it finishes, with costs frozen
at that moment. `ChatRecord::tokens`, `cost`, and `total_cost` read them back. See
[Persistence with Loco](persistence-loco.md).

## Counting Tokens Before Sending

```ruby
RubyLLM.count_tokens("Explain Ruby blocks.", model: "claude-haiku-4-5")
RubyLLM.tokenize("Ruby makes AI useful.", model: "grok-4.3", provider: :xai).count
```

```rust,no_run
# async fn run() -> rust_llm::Result<()> {
let input = rust_llm::count_tokens("Explain Rust closures.", Some("claude-haiku-4-5"), None).await?;
let options = rust_llm::TokenizeOptions { model: Some("grok-4.3"), provider: Some("xai"), ..Default::default() };
let tokens = rust_llm::tokenize("Rust makes AI useful.", options).await?.count();
# Ok(()) }
```

These inspect input before generation; they do not predict billed usage. See
[Chat](chat.md#counting-tokens) and [Tokenization](moderation-ocr-rerank.md#tokenization).

## Usage Events

Every finished provider attempt, for chats and one-shot operations alike, emits a
`usage.rust_llm` [instrumentation](instrumentation.md) event with `operation`, `provider`,
`model`, `status`, `tokens`, and `cost`. Results collected from a batch emit it once each.

## Keeping Pricing Fresh

Prices come from the bundled `models.json` until you refresh the registry:

```ruby
RubyLLM.models.refresh
```

```rust,no_run
# async fn run() -> rust_llm::Result<()> {
let models = rust_llm::models::refresh(false).await?; // `true` skips local providers (Ollama, GPUStack)
# Ok(()) }
```

`refresh` fetches the published catalog (the same `models.json` RubyLLM publishes), merges each
configured provider's model list, installs the result, and saves it to `model_registry_file` or
`model_registry_store` (see [Configuration](configuration.md#other-options)).
`rust_llm::models::Models::install(models)` replaces the process-wide registry with a list you
built yourself.
