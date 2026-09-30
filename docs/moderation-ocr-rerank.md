# Moderation, OCR, Rerank, and Tokenization

Four single-purpose operations: screen content with `rust_llm::moderate`, extract text from
documents with `rust_llm::ocr`, order documents by relevance with `rust_llm::rerank`, and split
text into tokens with `rust_llm::tokenize`. This follows RubyLLM's `moderation.md`, `ocr.md`,
`rerank.md`, and `tokenization.md`.

## Moderation

```ruby
result = RubyLLM.moderate "I love programming in Ruby."
result.flagged?           # => false
result.flagged_categories
result.category_scores
```

```rust,no_run
# async fn run() -> rust_llm::Result<()> {
let result = rust_llm::moderate("I love programming in Rust.", Default::default()).await?;
if result.is_flagged() {
    println!("{:?} {:?}", result.flagged_categories(), result.category_scores());
}
# Ok(()) }
```

Several texts get one verdict each; images go in `with`, with or without text:

```ruby
moderation = RubyLLM.moderate(["Great explanation!", "Can you add a Rails example?"])
RubyLLM.moderate("Photo for my profile", with: "profile.png")
```

```rust,no_run
use rust_llm::{Attachment, ModerateOptions, ModerationInput};

# async fn run() -> rust_llm::Result<()> {
let comments = vec!["Great explanation!".to_string(), "Can you add a Loco example?".into()];
let moderation = rust_llm::moderate(comments.clone(), Default::default()).await?;
for (text, verdict) in comments.iter().zip(&moderation.results) {
    println!("{text}: {}", verdict.is_flagged());
}

let images = ModerateOptions { with: vec![Attachment::new("profile.png")], ..Default::default() };
rust_llm::moderate(ModerationInput::None, images).await?; // images only
# Ok(()) }
```

The overall result is flagged when any input is; `category_scores()` keeps the highest score per
category, and `raw` holds the provider response. The default model is
`config.default_moderation_model` (`omni-moderation-latest`). Providers on the Chat Completions
or Responses protocol moderate through their `moderations` endpoint (OpenAI, Mistral, and other
OpenAI-compatible providers); the rest return `Error::Api("... doesn't support moderation")`.

## OCR

```ruby
ocr = RubyLLM.ocr("contract.pdf")
ocr.markdown
ocr.pages.each { |page| puts page.index, page.markdown }
RubyLLM.ocr("annual-report.pdf", pages: [0, 1, 2], provider_options: { table_format: "html" })
```

```rust,no_run
use rust_llm::OcrOptions;
use serde_json::json;

# async fn run() -> rust_llm::Result<()> {
let ocr = rust_llm::ocr("contract.pdf", Default::default()).await?;
println!("{}", ocr.markdown()); // every page, joined
for page in &ocr.pages {
    println!("page {}: {:?}", page.index, page.markdown);
}

let options = OcrOptions { pages: Some(vec![0, 1, 2]), provider_options: json!({ "table_format": "html" }), ..Default::default() };
rust_llm::ocr("annual-report.pdf", options).await?;
# Ok(()) }
```

The file is a path, URL, or `Attachment`. Each page has `index` (zero-based), `markdown`,
`images`, `tables`, and `raw`. OCR runs on Mistral; the default model is
`config.default_ocr_model` (`mistral-ocr-latest`). To pull fields out of a document, pass
`ocr.markdown()` to a chat with a [schema](structured-output.md).

## Rerank

```ruby
rerank = RubyLLM.rerank("what is ruby", documents, model: "voyageai/rerank-2.5-lite", provider: :openrouter, top_n: 5)
rerank.results.first.document
```

```rust,no_run
use rust_llm::RerankOptions;

# async fn run(documents: Vec<String>) -> rust_llm::Result<()> {
let docs: Vec<&str> = documents.iter().map(String::as_str).collect();
let options = RerankOptions { provider: Some("openrouter"), top_n: Some(5), ..Default::default() };
let rerank = rust_llm::rerank("what is rust", &docs, "voyageai/rerank-2.5-lite", options).await?;
for result in &rerank.results {
    println!("#{} {:?}: {}", result.index, result.score, result.document);
}
# Ok(()) }
```

`model` is required: rerank has no default. `index` is the document's position in the slice you
passed, so map results back to your records with it. Scores are provider-specific, so use them to
order and to set cutoffs tuned on your data. OpenRouter and GPUStack rerank.

## Tokenization

```ruby
result = RubyLLM.tokenize("Ruby makes AI useful.", model: "grok-4.3", provider: :xai)
result.ids
result.count
```

```rust,no_run
use rust_llm::TokenizeOptions;

# async fn run() -> rust_llm::Result<()> {
let options = TokenizeOptions { model: Some("grok-4.3"), provider: Some("xai"), ..Default::default() };
let result = rust_llm::tokenize("Rust makes AI useful.", options).await?;
println!("{} tokens: {:?}", result.count(), result.ids);
# Ok(()) }
```

xAI and GPUStack tokenize. Tokenization covers the string only: it excludes chat formatting,
instructions, tools, and attachments. To count a whole chat request, use `rust_llm::count_tokens`
or `chat.count_tokens` (see [Chat](chat.md#counting-tokens)).

## Tokens, Cost, and Events

`Moderation`, `Ocr`, and `Rerank` have `tokens()`, `cost()`, and `usage_entries` (see
[Cost and Usage](cost-and-usage.md)); tokenizing records no usage. Each operation emits its
`*.rust_llm` [instrumentation](instrumentation.md) event: `moderation`, `ocr`, `rerank`, and
`tokenization`. Every options struct also takes `config` for an
[isolated configuration](configuration.md#isolated-configurations).

## Differences from RubyLLM

- Bedrock Guardrails moderation, Cohere OCR and rerank, and Bedrock, Azure, and Vertex AI rerank
  belong to providers RustLLM does not port.
- Files are paths, URLs, or `Attachment`s, not IO objects or Active Storage attachments.
