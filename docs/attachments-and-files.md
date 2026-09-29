# Attachments and Files

Send images, PDFs, audio, video, and text files with a message, and manage files stored with a
provider.

## Attaching Files

RubyLLM's `ask(msg, with: ...)` is `ask_with(msg, attachments)`. An `Attachment` is built from a
local path or an http(s) URL; `&str` converts with `.into()`.

```ruby
chat.ask "Describe this logo.", with: "path/to/ruby_logo.png"
chat.ask "What kind of architecture is shown here?", with: "https://example.com/eiffel_tower.jpg"
chat.ask "Compare these screenshots.", with: ["screenshot_v1.png", "screenshot_v2.png"]
```

```rust,no_run
use rust_llm::Attachment;

# async fn run() -> rust_llm::Result<()> {
let mut chat = rust_llm::chat_with("gpt-5.6")?;
chat.ask_with("Describe this logo.", vec!["path/to/rust_logo.png".into()]).await?;
chat.ask_with("What kind of architecture is shown here?", vec![Attachment::new("https://example.com/eiffel_tower.jpg")]).await?;
chat.ask_with("Compare these screenshots.", vec!["screenshot_v1.png".into(), "screenshot_v2.png".into()]).await?;
# Ok(()) }
```

Bytes you already hold (an upload, a generated file) use `from_bytes`:

```rust,no_run
use rust_llm::Attachment;

# fn run(bytes: Vec<u8>) {
let invoice = Attachment::from_bytes(bytes, "invoice.pdf", None); // MIME type from the name
let chart = Attachment::from_bytes(vec![], "chart", Some("image/png"));
# }
```

The type (image, video, audio, PDF, text, other document) is detected from the file name. Local
files are read when the request is built. URLs are passed through to providers that accept them
and downloaded otherwise.

Only pass paths and URLs your application trusts. An attachment source is an instruction to read a
local file or fetch a URL; passing a raw user parameter lets the user point it at server files or
internal services.

`ask_later_with(msg, attachments)` stages a message with attachments without sending it.

## Media Resolution

```ruby
RubyLLM::Attachment.new("page-3.png", resolution: :ultra_high)
```

```rust,no_run
use rust_llm::{Attachment, Resolution};

let page = Attachment::new("page-3.png").with_resolution(Resolution::UltraHigh);
```

`Low`, `Medium`, `High`, `UltraHigh`. Gemini gets `media_resolution`; OpenAI protocols get
`detail: "low"` for `Low` and `"high"` otherwise.

## Uploading Files

Upload once and reuse the file across requests:

```ruby
file = RubyLLM.upload("contract.pdf", provider: :openai, purpose: "user_data")
RubyLLM.chat(model: "gpt-5-nano").ask("Summarize this", with: file)
```

```rust,no_run
use rust_llm::UploadOptions;

# async fn run() -> rust_llm::Result<()> {
let options = UploadOptions { provider: Some("openai"), purpose: Some("user_data"), ..Default::default() };
let file = rust_llm::upload("contract.pdf", options).await?;

rust_llm::chat_with("gpt-5-nano")?.ask_with("Summarize this", vec![file.clone().into()]).await?;

println!("{} {:?} {:?}", file.id, file.filename, file.expires_at);
# Ok(()) }
```

Without `provider`, the default chat model's provider is used. `UploadOptions` also takes
`filename`, `expires_in` (seconds), `provider_options` (e.g. Gemini `display_name`, Mistral
`visibility`), and `config`. File uploads work with OpenAI, Anthropic, Gemini, Mistral, xAI,
DeepSeek, OpenRouter, and Perplexity. `file.is_expired()` is true once the retention window has
passed or ends within a minute. Uploads are never retried.

## Large Attachments

With `auto_upload_large_files` (on by default), a local attachment larger than the provider's
inline limit is uploaded to its Files API before the request and sent by reference. The upload is
reused for later requests in the same chat.

## Finding and Downloading

```ruby
file = RubyLLM::UploadedFile.find("file-abc", provider: :openai)
RubyLLM.download("file-abc", provider: :openai).save("out.pdf")
```

```rust,no_run
use rust_llm::{FileOptions, UploadedFile};

# async fn run() -> rust_llm::Result<()> {
let options = || FileOptions { provider: Some("openai"), ..Default::default() };
let file = UploadedFile::find("file-abc", options()).await?;
let downloaded = rust_llm::download("file-abc", options()).await?;
downloaded.save("out.pdf")?;
let bytes: &[u8] = downloaded.to_blob();
# Ok(()) }
```

## Not ported

- Active Storage attachments and IO objects: pass a path, URL, or bytes.
- ElevenLabs media assets and Cohere datasets.
- Attachments on persisted messages in `rust_llm_loco`.
