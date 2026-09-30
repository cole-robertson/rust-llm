# Batches

Submit many requests to a provider's batch API, often at a lower price, and collect the answers
later.

## Staging and Submitting

`ask_later` stages a question without calling the provider. `rust_llm::batch` submits the staged
chats as one batch.

```ruby
chats = documents.map do |doc|
  RubyLLM.chat(model: "claude-haiku-4-5")
    .with_instructions("Summarize the document in one paragraph.")
    .ask_later(doc.text)
end
batch = RubyLLM.batch(chats)
batch.id     # save this
batch.status # => :pending
```

```rust,no_run
use rust_llm::{BatchStatus, Chat};

# async fn run(documents: Vec<String>) -> rust_llm::Result<()> {
let mut chats: Vec<Chat> = Vec::new();
for text in &documents {
    let mut chat = rust_llm::chat_with("claude-haiku-4-5")?.with_instructions("Summarize the document in one paragraph.");
    chat.ask_later(text.as_str())?;
    chats.push(chat);
}

let batch = rust_llm::batch(chats).await?;
let id = batch.id().to_string(); // save this
assert_eq!(batch.status(), BatchStatus::Pending);
# Ok(()) }
```

The batch takes ownership of the chats; read them back with `batch.chats()` or
`batch.into_chats()`. Chats keep their instructions, history, tools, schemas, and other settings.
Use one provider per batch; Anthropic and xAI allow different models in one batch. Submission is
never retried.

## Collecting the Answers

```ruby
batch = RubyLLM::Batch.find(batch_id, provider: :anthropic)
sleep 60 until batch.refresh.complete?
batch.messages.each { |message| message&.content }
```

```rust,no_run
use rust_llm::Batch;

# async fn run(batch_id: &str) -> rust_llm::Result<()> {
let mut batch = Batch::find(batch_id, Some("anthropic")).await?;
while !batch.refresh().await?.is_complete() {
    tokio::time::sleep(std::time::Duration::from_secs(60)).await;
}

for message in batch.messages().await? {
    match message {
        Some(message) => println!("{}", message.content()),
        None => println!("(this request failed)"),
    }
}
println!("{:?}", batch.statuses()); // per request: Some(Succeeded | Failed | Cancelled)
# Ok(()) }
```

- `is_complete()` reads the last fetched state; `refresh()` asks the provider.
- `status()` is `Pending`, `Succeeded`, `Failed`, or `Cancelled`; `raw_status()` is the provider's
  own string; `is_succeeded()`, `is_failed()`, `is_cancelled()` match it.
- `messages()` / `results()` return answers in submission order, `None` for failed slots. When the
  batch holds its chats, each answer is also appended to its chat. Collected results are cached once
  the batch is complete.
- `Batch::find` needs the provider. To have answers appended to chats you rebuilt in another
  process, pass them with `.with_chats(chats)` in submission order.
- `cancel()` stops unfinished work where the provider supports it.

## Cost and Usage

```rust,no_run
# async fn run(mut batch: rust_llm::Batch) -> rust_llm::Result<()> {
let total = batch.cost().await?.total(); // None until the batch has ended; batch-tier prices
let input = batch.tokens().await?.input;
# Ok(()) }
```

## Tools in Batches

A batch generates one model turn per chat. Run requested tools yourself, then submit the next turn:

```ruby
batch.messages
chats.each(&:run_tools)
pending = chats.reject(&:complete?)
next_batch = RubyLLM.batch(pending) if pending.any?
```

```rust,no_run
# async fn run(mut batch: rust_llm::Batch) -> rust_llm::Result<()> {
batch.messages().await?;
let mut chats = batch.into_chats().unwrap_or_default();
for chat in &mut chats {
    chat.run_tools().await?;
}
let pending: Vec<_> = chats.into_iter().filter(|c| !c.is_complete()).collect();
if !pending.is_empty() {
    let next_batch = rust_llm::batch(pending).await?;
}
# Ok(()) }
```

Record `approve`/`deny` decisions before `run_tools` for tools that require approval.

## Batching Embeddings

```ruby
requests = documents.map { |doc| RubyLLM.embed_later(doc.text, model: "text-embedding-3-small") }
batch = RubyLLM.batch(requests)
sleep 60 until batch.refresh.complete?
batch.results.first.vectors
```

```rust,no_run
use rust_llm::{EmbedOptions, EmbeddingRequest};

# async fn run(documents: Vec<String>) -> rust_llm::Result<()> {
let requests: Vec<EmbeddingRequest> = documents
    .iter()
    .map(|text| rust_llm::embed_later(text.as_str(), EmbedOptions { model: Some("text-embedding-3-small"), ..Default::default() }))
    .collect::<rust_llm::Result<_>>()?;

let mut batch = rust_llm::batch(requests).await?;
while !batch.refresh().await?.is_complete() {
    tokio::time::sleep(std::time::Duration::from_secs(60)).await;
}
for result in batch.results().await? {
    if let Some(embedding) = result.as_ref().and_then(|r| r.as_embedding()) {
        println!("{:?}", embedding.vectors);
    }
}
// batch.requests() holds the EmbeddingRequests with `result` filled in.
# Ok(()) }
```

`embed_later` accepts `model`, `provider`, and `dimensions`, with the same defaults as `embed`. A
batch takes chats or embedding requests, not both. OpenAI, Gemini, Mistral, and OpenRouter batch
embeddings; OpenRouter accepts text only, without `task_type` or provider preferences.

## Persisted Batches

With a `config.batch_store`, submitted chat batches are recorded, and `Batch::find` returns a stored
batch without asking for the provider. `rust_llm_loco::BatchStore` keeps them in the
`rust_llm_batches` table with the chat records they answer:

```ruby
batch = RubyLLM.batch(chats)          # records are persisted by the Railtie's batch store
RubyLLM::Batch.find(batch.id).messages # answers land on the chat records
```

```rust,no_run
use std::sync::Arc;
use rust_llm_loco::{BatchStore, ChatRecord};

# async fn run(db: sea_orm::DatabaseConnection) -> rust_llm_loco::Result<()> {
rust_llm::configure(|c| c.batch_store = Some(Arc::new(BatchStore::new(db.clone()))));

let record = ChatRecord::create(&db, "claude-haiku-4-5", None).await?;
let mut chat = record.to_llm(&db).await?;
record.ask_later(&db, &mut chat, "What is 2 + 2?").await?;
let batch = rust_llm_loco::batch::submit(&db, vec![(record, chat)]).await?;

// Later, in a job: no provider needed.
let mut batch = rust_llm::Batch::find(batch.id(), None).await?;
if batch.refresh().await?.is_complete() {
    rust_llm_loco::batch::collect(&db, &mut batch).await?; // persists each answer on its record
}
# Ok(()) }
```

`collect` appends each answer to its record once, however often it runs. Without a store,
`Batch::find` needs the provider, and you keep the id and provider yourself.
