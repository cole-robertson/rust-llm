# Images

Generate and edit images with `rust_llm::paint`.

## Generating an Image

```ruby
image = RubyLLM.paint "A red panda coding Ruby on a laptop, watercolor"
image.save "red_panda.png"
```

```rust,no_run
# async fn run() -> rust_llm::Result<()> {
let image = rust_llm::paint("A red panda coding Rust on a laptop, watercolor", Default::default())
    .await?
    .into_image();
image.save("red_panda.png").await?;
# Ok(()) }
```

`paint` returns `Images`: `Images::One(Image)` or, when the provider returned several,
`Images::Many(Vec<Image>)` (RubyLLM returns an image or an array). `into_image()` takes the first;
`into_vec()` takes them all. `save` handles hosted URLs and inline Base64 data.

## Options

`PaintOptions` carries the keywords:

```ruby
images = RubyLLM.paint("a siamese cat", model: "gpt-image-2", size: "1024x1024", count: 4)
images.each_with_index { |image, i| image.save("cat-#{i}.png") }
```

```rust,no_run
use rust_llm::PaintOptions;

# async fn run() -> rust_llm::Result<()> {
let options = PaintOptions { model: Some("gpt-image-2"), size: Some("1024x1024"), count: Some(4), ..Default::default() };
let images = rust_llm::paint("a siamese cat", options).await?;
for (i, image) in images.into_vec().into_iter().enumerate() {
    image.save(format!("cat-{i}.png")).await?;
}
# Ok(()) }
```

Fields: `model`, `provider`, `assume_model_exists`, `size`, `count`, `with` (source images to edit),
`mask`, `provider_options` (merged into the request), and `config`. The default model is
`config.default_image_model` (`gpt-image-2`). Some providers ignore `size` or `count` and log that
at `debug` level. For Gemini, `size` may be an aspect ratio such as `"16:9"`.

## Editing Images

```ruby
RubyLLM.paint("Turn the logo green", model: "gpt-image-2", with: "logo.png")
RubyLLM.paint("Replace only the background", with: "portrait.png", mask: "portrait-mask.png")
```

```rust,no_run
use rust_llm::{Attachment, PaintOptions};

# async fn run() -> rust_llm::Result<()> {
let options = PaintOptions {
    model: Some("gpt-image-2"),
    with: vec![Attachment::new("portrait.png")],
    mask: Some(Attachment::new("portrait-mask.png")),
    ..Default::default()
};
let edited = rust_llm::paint("Replace only the background with a sunset sky", options).await?.into_image();
# Ok(()) }
```

## Working with the Result

```rust,no_run
# async fn run(image: rust_llm::Image) -> rust_llm::Result<()> {
let bytes: Vec<u8> = image.to_blob().await?; // decodes data or downloads url
let hosted = image.url.as_deref();            // Some for providers that host the image
let inline = image.is_base64();
let mime = image.mime_type.as_deref();
let revised = image.revised_prompt.as_deref();
let model = &image.model;
# Ok(()) }
```

## Tokens and Cost

```rust,no_run
# async fn run(image: rust_llm::Image) {
let tokens = image.tokens();
let total = image.cost().total(); // None when pricing or usage is unknown
# }
```

Usage for the request is on the first image, so a multi-image call is billed once.

## Not ported

- Multipart image edits for non-`gpt-image` models (dall-e-2).
- ElevenLabs image generation.
- Active Storage integration and IO-object sources.
