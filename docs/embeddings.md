# Embeddings

Turn text into vectors for similarity search, clustering, and retrieval.

## Basic Embedding

```ruby
embedding = RubyLLM.embed("Ruby is a programmer's best friend")
embedding.vectors # => [0.018, -0.027, ...]
```

```rust,no_run
use rust_llm::Vectors;

# async fn run() -> rust_llm::Result<()> {
let embedding = rust_llm::embed("Rust is a programmer's best friend", Default::default()).await?;
if let Vectors::Single(vector) = &embedding.vectors {
    println!("{} dimensions", vector.len());
}
# Ok(()) }
```

Ruby returns a flat array for one string and an array of arrays for several; `Vectors` makes the
two cases explicit: `Single(Vec<f64>)` or `Batch(Vec<Vec<f64>>)`.

## Several Texts

```ruby
embeddings = RubyLLM.embed(["Ruby", "Python", "JavaScript"])
embeddings.vectors.length # => 3
```

```rust,no_run
use rust_llm::Vectors;

# async fn run() -> rust_llm::Result<()> {
let texts = vec!["Rust".to_string(), "Python".into(), "JavaScript".into()];
let embeddings = rust_llm::embed(texts.clone(), Default::default()).await?;
if let Vectors::Batch(vectors) = embeddings.vectors {
    for (text, vector) in texts.iter().zip(&vectors) {
        println!("{text}: {} dimensions", vector.len());
    }
}
# Ok(()) }
```

## Models, Providers, and Dimensions

`EmbedOptions` carries RubyLLM's keywords:

```ruby
RubyLLM.embed("This is a test sentence", model: "text-embedding-3-small", dimensions: 512)
```

```rust,no_run
use rust_llm::EmbedOptions;

# async fn run() -> rust_llm::Result<()> {
let options = EmbedOptions { model: Some("text-embedding-3-small"), dimensions: Some(512), ..Default::default() };
let embedding = rust_llm::embed("This is a test sentence", options).await?;
# Ok(()) }
```

The fields are `model`, `provider`, `dimensions`, `assume_model_exists`, and `config`. The default
model is `config.default_embedding_model` (`text-embedding-3-small`). OpenAI-compatible providers
and Gemini embed; Anthropic returns `Error::Api("Anthropic doesn't support embeddings")`.

## Usage and Cost

```rust,no_run
# async fn run() -> rust_llm::Result<()> {
let embedding = rust_llm::embed("Rust", Default::default()).await?;
let input = embedding.tokens().input;   // None when the provider does not report it (Gemini)
let total = embedding.cost().total();
let model = &embedding.model;
# Ok(()) }
```

## Measuring Similarity

The crate returns vectors; comparing them is ordinary Rust:

```rust,no_run
fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
    dot / (norm(a) * norm(b))
}
```

Store vectors in a `pgvector` column or any vector store.

## Batching Embeddings

`rust_llm::embed_later` stages an embedding for a provider batch. See [Batches](batches.md).

## Not ported

- `task_type:` and `title:` (Vertex AI), and `with:` media embeddings.
- Sparse vectors and extra `provider_options`.
- Gemini embedding batches.
