# Streaming

Show a response as it is generated.

## Basic Streaming

RubyLLM streams when `ask` gets a block; RustLLM has `ask_stream`, which takes a closure called
with each chunk.

```ruby
chat.ask "Write a short story about an adventurous ruby gem." do |chunk|
  print chunk.content
end
```

```rust,no_run
# async fn run() -> rust_llm::Result<()> {
let mut chat = rust_llm::chat()?;
chat.ask_stream("Write a short story about an adventurous crab.", |chunk| {
    print!("{}", chunk.content());
})
.await?;
# Ok(()) }
```

`complete_stream` and `step_stream` are the streaming versions of `complete` and `step`.

## Chunks

A chunk is a `Message` (`Chunk` is a type alias, as `Chunk < Message` in Ruby) holding only what
arrived in that piece of the stream:

- `chunk.content`: the text fragment, `None` for chunks that carry only metadata or tool calls.
  `chunk.content()` returns `""` for `None`.
- `chunk.thinking`: thinking text, when the provider streams it.
- `chunk.tool_calls`: partial tool calls. Arguments arrive as `ToolArguments::Partial` JSON
  fragments; use `before_tool_call` when you need the complete call.
- `chunk.tokens`: usually only set on the last chunk.
- `chunk.finish_reason`: usually only on the last chunk.

## The Returned Message

`ask_stream` returns the accumulated message, with the full text, usage, and cost:

```ruby
response = chat.ask("Write a haiku about programming.") { |chunk| print chunk.content }
response.tokens.output
response.cost.total
```

```rust,no_run
# async fn run(mut chat: rust_llm::Chat) -> rust_llm::Result<()> {
let response = chat.ask_stream("Write a haiku about programming.", |chunk| print!("{}", chunk.content())).await?;
let output = response.tokens().output;
let total = response.cost(None).total();
let reason = response.finish_reason;
# Ok(()) }
```

## Streaming with Tools

The closure keeps receiving text across tool rounds. Use a callback to show tool activity:

```rust,no_run
# use rust_llm::{Tool, ToolCall, ToolError, ToolResult};
# struct Weather;
# #[async_trait::async_trait]
# impl Tool for Weather {
#     fn description(&self) -> String { String::new() }
#     async fn execute(&self, _: serde_json::Map<String, serde_json::Value>, _: &ToolCall) -> Result<ToolResult, ToolError> { Ok("".into()) }
# }
# async fn run() -> rust_llm::Result<()> {
let mut chat = rust_llm::chat()?
    .with_tool(Weather)
    .before_tool_call(|call| println!("\nCalling {}...", call.name));
chat.ask_stream("What's the weather in Berlin? Latitude 52.52, longitude 13.40.", |chunk| {
    print!("{}", chunk.content());
})
.await?;
# Ok(()) }
```

## Streaming to a Web Client

The closure is synchronous (`FnMut(&Message) + Send`). To forward chunks to an SSE or WebSocket
response, push them into a channel and let the handler drain it:

```rust,no_run
# async fn run() -> rust_llm::Result<()> {
let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();

let writer = tokio::spawn(async move {
    while let Some(text) = rx.recv().await {
        // write `data: {text}\n\n` to the response body
        let _ = text;
    }
});

let mut chat = rust_llm::chat()?;
chat.ask_stream("Tell me a fun fact.", move |chunk| {
    if let Some(text) = &chunk.content {
        let _ = tx.send(text.clone());
    }
})
.await?;
let _ = writer.await;
# Ok(()) }
```

The generated Loco chat UI does not stream tokens; it polls for persisted messages (see
[Generators](generators.md)).

## Cancelling a Stream

```ruby
chat.ask("Write a long report") { |chunk| chat.cancel if should_stop? }
rescue RubyLLM::CancelledError
```

`Chat::cancel` takes `&self`, but the chat is mutably borrowed while `ask_stream` runs, so take a
`CancelHandle` first. It can also be sent to another task.

```rust,no_run
# fn should_stop() -> bool { true }
# async fn run() -> rust_llm::Result<()> {
let mut chat = rust_llm::chat()?;
let handle = chat.cancel_handle();

match chat.ask_stream("Write a long report", |chunk| {
    print!("{}", chunk.content());
    if should_stop() {
        handle.cancel();
    }
})
.await
{
    Err(rust_llm::Error::Cancelled) => println!("Generation cancelled"),
    other => { other?; }
}
# Ok(()) }
```

Cancellation is checked before each model request, before each tool runs, and after every chunk.
A cancelled attempt is still recorded in `chat.usage_entries()` with status `Cancelled`.

## Errors During Streaming

The closure runs for every chunk received before an error; then `ask_stream` returns the error.
Keep what you printed if you want the partial text. A stream that already delivered a chunk is
never retried. See [Errors and Retries](errors-and-retries.md).

## Not ported

- Persisted cancellation (`acts_as_chat`'s `cancel` writing the `cancelled` column). The column
  exists, but `ChatRecord` does not read or write it.
- Turbo Streams broadcasting from the chat UI generator.
