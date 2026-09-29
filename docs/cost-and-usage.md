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

The same readers exist on `Embedding` (`tokens()`, `cost()`), `Image`, `Judgment`, `BatchResult`,
and `Batch` (`tokens().await`, `cost().await`, at batch rates).

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

## Keeping Pricing Fresh

Prices come from the bundled `models.json`. `rust_llm::models::Models::install(models)` replaces
the process-wide registry if you load a fresher list yourself.

## Not ported

- `count_tokens` and `tokenize`.
- `RubyLLM.models.refresh`.
- The `usage.ruby_llm` instrumentation event. `Chat::set_usage_recorder` receives each entry as it
  is recorded, but it has a single slot that `rust_llm_loco` uses for persistence.
