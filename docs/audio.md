# Audio

Turn text into speech with `rust_llm::speak`, and speech into text with `rust_llm::transcribe`.
This follows RubyLLM's `text-to-speech.md` and `audio-transcription.md`.

## Generating Speech

```ruby
speech = RubyLLM.speak "Hello, welcome to RubyLLM!"
speech.save "welcome.mp3"
speech.format    # => "mp3"
speech.mime_type # => "audio/mpeg"
```

```rust,no_run
# async fn run() -> rust_llm::Result<()> {
let speech = rust_llm::speak("Hello, welcome to RustLLM!", Default::default()).await?;
speech.save("welcome.mp3")?;
println!("{} {} {:?}", speech.model, speech.mime_type, speech.voice);
let bytes: &[u8] = speech.to_blob();
# Ok(()) }
```

`SpeakOptions` carries the keywords: `model`, `provider`, `assume_model_exists`, `voice`, `format`,
`provider_options`, `config`, and `metadata`. The default model is `config.default_speech_model`.
OpenAI and OpenAI-compatible providers, Gemini, Mistral, xAI, OpenRouter, and GPUStack speak.

```ruby
RubyLLM.speak("The build is green.", voice: "nova", format: "wav",
              provider_options: { instructions: "Speak with calm confidence.", speed: 1.1 })
```

```rust,no_run
use rust_llm::SpeakOptions;
use serde_json::json;

# async fn run() -> rust_llm::Result<()> {
let options = SpeakOptions {
    voice: Some("nova"),
    format: Some("wav"),
    provider_options: json!({ "instructions": "Speak with calm confidence.", "speed": 1.1 }),
    ..Default::default()
};
rust_llm::speak("The build is green.", options).await?.save("voiceover.wav")?;
# Ok(()) }
```

Gemini's speech models return raw PCM (`speech.format == "pcm"`); convert it with a tool such as
`ffmpeg -f s16le -ar 24000 -ac 1 -i out.pcm out.wav`.

## Streaming Speech

```ruby
speech = RubyLLM.speak("Hello!") { |chunk| player.write(chunk.data) }
```

```rust,no_run
# async fn run(mut player: impl std::io::Write + Send) -> rust_llm::Result<()> {
let speech = rust_llm::speak_stream("Hello!", Default::default(), |chunk| {
    let _ = player.write_all(&chunk.data);
})
.await?;
speech.save("hello.mp3")?; // the complete recording
# Ok(()) }
```

Chunks are consecutive bytes of one recording, and each has `data`, `format`, and `mime_type`. A
stream that already delivered a chunk is not retried.

## Transcribing Audio

```ruby
transcription = RubyLLM.transcribe("meeting.wav")
transcription.text
```

```rust,no_run
# async fn run() -> rust_llm::Result<()> {
let transcription = rust_llm::transcribe("meeting.wav", Default::default()).await?;
println!("{}", transcription.text.as_deref().unwrap_or_default());
# Ok(()) }
```

The audio is a path, URL, or `Attachment`. `TranscribeOptions` carries `model`, `provider`,
`assume_model_exists`, `language`, `prompt`, `temperature`, `format`, `timestamps`,
`speaker_names`, `speaker_references`, `provider_options`, `config`, and `metadata`. The default
model is `config.default_transcription_model`. OpenAI and OpenAI-compatible providers, Gemini,
Mistral, xAI, OpenRouter, and GPUStack transcribe.

## Language, Speakers, and Timestamps

```ruby
RubyLLM.transcribe("entrevista.mp3", language: "es", prompt: "Ruby, Rails, PostgreSQL")
RubyLLM.transcribe("meeting.wav", model: "gpt-4o-transcribe-diarize",
                   speaker_names: ["Alice", "Bob"], speaker_references: ["alice.wav", "bob.wav"])
RubyLLM.transcribe("interview.mp3", model: "whisper-1", timestamps: [:word, :segment])
```

```rust,no_run
use rust_llm::{Attachment, TranscribeOptions};

# async fn run() -> rust_llm::Result<()> {
let hints = TranscribeOptions { language: Some("es"), prompt: Some("Rust, Loco, PostgreSQL"), ..Default::default() };
rust_llm::transcribe("entrevista.mp3", hints).await?;

let diarized = TranscribeOptions {
    model: Some("gpt-4o-transcribe-diarize"),
    speaker_names: Some(vec!["Alice".into(), "Bob".into()]),
    speaker_references: Some(vec![Attachment::new("alice.wav"), Attachment::new("bob.wav")]),
    ..Default::default()
};
for segment in rust_llm::transcribe("meeting.wav", diarized).await?.segments.unwrap_or_default() {
    println!("{}: {}", segment["speaker"], segment["text"]);
}

let timed = TranscribeOptions { model: Some("whisper-1"), timestamps: Some(vec!["word", "segment"]), ..Default::default() };
let transcription = rust_llm::transcribe("interview.mp3", timed).await?;
println!("{:?} seconds, {} words", transcription.duration, transcription.words.map_or(0, |w| w.len()));
# Ok(()) }
```

`segments` and `words` keep the provider's field names. Which timestamp granularities a model
accepts depends on the provider; an unsupported combination fails with `Error::Argument`.
`format` (for example `"srt"`) uses the provider's names.

## Streaming Transcripts

```ruby
RubyLLM.transcribe("meeting.wav", model: "gpt-4o-transcribe") do |chunk|
  print chunk.delta if chunk.delta?
end
```

```rust,no_run
use rust_llm::TranscribeOptions;

# async fn run() -> rust_llm::Result<()> {
let options = TranscribeOptions { model: Some("gpt-4o-transcribe"), ..Default::default() };
let transcription = rust_llm::transcribe_stream("meeting.wav", options, |chunk| {
    if chunk.is_delta() {
        print!("{}", chunk.delta.as_deref().unwrap_or_default());
    }
})
.await?;
# Ok(()) }
```

| Predicate | What it carries |
|---|---|
| `is_partial()` | `text`: tentative text that replaces the previous partial |
| `is_delta()` | `delta`: committed text to append |
| `is_segment()` | `segment`: a JSON object with speaker and timing fields |
| `is_done()` | `text`: the complete transcript, when the endpoint sends one |

`chunk.raw` holds the original event. The call still returns the completed `Transcription`.

### WebSocket Transcription

xAI (`grok-stt`) and Gemini Live (`gemini-3.5-transcribe-live`) stream over a WebSocket. The API is
the same; the WebSocket client is built in, with no extra dependency:

```rust,no_run
use rust_llm::TranscribeOptions;

# async fn run() -> rust_llm::Result<()> {
let options = TranscribeOptions { model: Some("grok-stt"), ..Default::default() };
rust_llm::transcribe_stream("meeting.wav", options, |chunk| {
    if chunk.is_partial() {
        println!("~ {}", chunk.text.as_deref().unwrap_or_default());
    }
})
.await?;
# Ok(()) }
```

| Provider | Model | Input |
|---|---|---|
| xAI | `grok-stt` | WAV: 16-bit PCM or 8-bit G.711 |
| Gemini | `gemini-3.5-transcribe-live` | mono 16-bit PCM WAV |

Gemini Live returns text without speaker labels or word timestamps. WebSocket connections do not
support `config.http_proxy`. The stream processes an existing recording, not a live microphone.

## Tokens and Cost

`Speech` and `Transcription` have `tokens()`, `cost()`, and `usage_entries`, like chat messages (see
[Cost and Usage](cost-and-usage.md)). Both operations emit `speech.rust_llm` and
`transcription.rust_llm` [instrumentation](instrumentation.md) events.

## Differences from RubyLLM

- The ElevenLabs, Deepgram, Azure, and Vertex AI speech and transcription models belong to
  providers RustLLM does not port.
- Audio sources are paths, URLs, or `Attachment`s, not IO objects or Active Storage attachments.
