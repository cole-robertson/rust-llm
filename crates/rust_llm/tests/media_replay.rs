//! Multimodal input, replayed from RubyLLM's `chat_{vision,video,audio,pdf,document}_models_*`
//! cassettes (`spec/ruby_llm/chat_content_spec.rb`), `embedding_multimodal_embeddings_*`
//! (`spec/ruby_llm/embedding_spec.rb`), and `chat_prompt_cache_round-trip_*`
//! (`spec/ruby_llm/chat_cache_until_here_spec.rb`). Media bytes travel base64-encoded in the
//! recorded bodies, so every fixture must match RubyLLM's byte for byte.
//!
//! Remote attachments: RubyLLM fetched `httpbin.org`, `pdfobject.com`, and `filesamples.com`
//! while recording. The replay server serves those recorded downloads; a provider that received
//! the URL itself (not its bytes) has the recorded host rewritten to the replay server's.

mod support;

use rust_llm::{Attachment, Chat, EmbedOptions, Error, Message, Vectors, embed};
use support::{Cassette, cassette_name, chat_for, config_for};

fn fixture(name: &str) -> String {
    format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
}

fn check(cond: bool, what: impl Into<String>) -> Result<(), String> {
    if cond { Ok(()) } else { Err(what.into()) }
}

fn matches_any(text: &str, words: &[&str]) -> bool {
    let lower = text.to_lowercase();
    words.iter().any(|w| lower.contains(w))
}

/// The URL RubyLLM fetched, re-hosted on the replay server.
fn served(cassette: &Cassette, url: &str) -> String {
    let path = url.split_once("://").and_then(|(_, rest)| rest.split_once('/')).map(|(_, p)| p).unwrap_or("");
    format!("{}/{path}", cassette.server.uri())
}

fn first_attachment(chat: &Chat, index: usize) -> (Option<String>, String) {
    let a = &chat.messages()[0].attachments[index];
    (a.filename.clone(), a.mime_type.clone())
}

/// Runs `body` for each `(provider, model)` whose cassette exists; returns how many replayed.
async fn each<F, Fut>(describe: &str, it: &str, models: &[(&'static str, &'static str)], hosts: &[&str], body: F) -> usize
where
    F: Fn(Cassette, &'static str, &'static str) -> Fut,
    Fut: std::future::Future<Output = Result<Cassette, String>>,
{
    let mut failures = Vec::new();
    let mut ran = 0;
    for &(provider, model) in models {
        let name = cassette_name(describe, provider, model, it);
        let Some(cassette) = Cassette::start_serving(&name, hosts).await else {
            failures.push(format!("{provider} {model}: missing cassette {name}"));
            continue;
        };
        ran += 1;
        match body(cassette, provider, model).await {
            Ok(cassette) => {
                let r = futures::FutureExt::catch_unwind(std::panic::AssertUnwindSafe(cassette.assert_all_matched())).await;
                if let Err(p) = r {
                    failures.push(format!("{provider} {model}: {}", p.downcast_ref::<String>().cloned().unwrap_or_default()));
                }
            }
            Err(e) => failures.push(format!("{provider} {model}: {e}")),
        }
    }
    assert!(failures.is_empty(), "{} of {} failed:\n{}", failures.len(), models.len(), failures.join("\n\n"));
    eprintln!("{it}: {ran} replayed");
    ran
}

// ---- vision -------------------------------------------------------------------------------------

/// `VISION_MODELS` for the providers this port implements.
const VISION_MODELS: &[(&str, &str)] = &[
    ("anthropic", "claude-haiku-4-5"),
    ("deepseek", "deepseek-flash"),
    ("gemini", "gemini-2.5-flash"),
    ("hetzner", "Qwen3.8-27B"),
    ("mistral", "pixtral-12b"),
    ("ollama", "gemma4"),
    ("openai", "gpt-5-nano"),
    ("openrouter", "claude-haiku-4-5"),
    ("xai", "grok-4-1-fast-non-reasoning"),
];

fn content_ok(response: &Message) -> Result<(), String> {
    check(!response.content().contains("RubyLLM::Content"), "content leaked a Ruby object")
}

#[tokio::test]
async fn vision_models_can_understand_local_images() {
    let ran = each("chat vision models", "can understand local images", VISION_MODELS, &[], |cassette, provider, model| async move {
        let mut chat = chat_for(&cassette, provider, model);
        let response = chat
            .ask_with("What do you see in this image?", vec![Attachment::new(fixture("ruby.png"))])
            .await
            .map_err(|e| e.to_string())?;
        check(matches_any(response.content(), &["ruby", "gem", "red", "crystal", "stone", "logo"]), response.content())?;
        content_ok(&response)?;
        check(chat.messages()[0].content() == "What do you see in this image?", "user content")?;
        check(first_attachment(&chat, 0) == (Some("ruby.png".into()), "image/png".into()), "attachment")?;
        Ok(cassette)
    })
    .await;
    assert_eq!(ran, VISION_MODELS.len());
}

const IMAGE_URL_NO_EXT: &str = "https://httpbin.org/image/jpeg";

#[tokio::test]
async fn vision_models_can_understand_remote_images_without_extension() {
    let describe = "chat vision models";
    let it = "can understand remote images without extension";
    let ran = each(describe, it, VISION_MODELS, &["https://httpbin.org"], |cassette, provider, model| async move {
        let mut chat = chat_for(&cassette, provider, model);
        let url = served(&cassette, IMAGE_URL_NO_EXT);
        let response = chat
            .ask_with("What do you see in this image?", vec![Attachment::new(url)])
            .await
            .map_err(|e| e.to_string())?;
        if provider == "ollama" {
            // Small local vision models cannot reliably describe the fetched image.
            check(!response.content().is_empty(), "empty")?;
        } else {
            check(matches_any(response.content(), &["coyote", "jackal", "canid", "canine"]), response.content())?;
        }
        content_ok(&response)?;
        check(chat.messages()[0].content() == "What do you see in this image?", "user content")?;
        let got = first_attachment(&chat, 0);
        check(got == (Some("jpeg".into()), "image/jpeg".into()), format!("attachment {got:?}"))?;
        Ok(cassette)
    })
    .await;
    assert_eq!(ran, VISION_MODELS.len());
}

#[tokio::test]
async fn vision_returns_errors_when_content_doesnt_exist() {
    let server = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::any())
        .respond_with(wiremock::ResponseTemplate::new(404).set_body_string("Not Found"))
        .mount(&server)
        .await;
    let mut config = rust_llm::Config::default();
    config.set("anthropic_api_key", "test-key");
    config.set("anthropic_api_base", server.uri());
    let config = std::sync::Arc::new(config);

    let mut chat = Chat::with_config(config.clone(), Some("claude-haiku-4-5"), Some("anthropic"), false).unwrap();
    let bad_url = format!("{}/eiffel_tower", server.uri());
    let err = chat.ask_with("What do you see in this image?", vec![Attachment::new(bad_url)]).await.unwrap_err();
    assert!(err.to_string().contains("404"), "{err:?}");

    let mut chat = Chat::with_config(config, Some("claude-haiku-4-5"), Some("anthropic"), false).unwrap();
    let err = chat.ask_with("What do you see in this image?", vec![Attachment::new(fixture("bad_image.png"))]).await.unwrap_err();
    assert!(matches!(err, Error::Io(ref e) if e.kind() == std::io::ErrorKind::NotFound), "{err:?}");
    // Neither attachment could be read, so nothing reached the provider.
    let provider_calls = server.received_requests().await.unwrap().iter().filter(|r| r.url.path() == "/v1/messages").count();
    assert_eq!(provider_calls, 0);
}

// ---- video ------------------------------------------------------------------------------------

const VIDEO_MODELS: &[(&str, &str)] = &[("gemini", "gemini-2.5-flash")];

#[tokio::test]
async fn video_models_can_understand_local_videos() {
    let ran = each("chat video models", "can understand local videos", VIDEO_MODELS, &[], |cassette, provider, model| async move {
        let mut chat = chat_for(&cassette, provider, model);
        let response = chat
            .ask_with("What do you see in this video?", vec![Attachment::new(fixture("ruby.mp4"))])
            .await
            .map_err(|e| e.to_string())?;
        check(matches_any(response.content(), &["beach", "ocean", "sand"]), response.content())?;
        content_ok(&response)?;
        check(first_attachment(&chat, 0) == (Some("ruby.mp4".into()), "video/mp4".into()), "attachment")?;
        Ok(cassette)
    })
    .await;
    assert_eq!(ran, 1);
}

#[tokio::test]
async fn video_models_can_understand_remote_videos_without_extension() {
    let it = "can understand remote videos without extension";
    let ran = each("chat video models", it, VIDEO_MODELS, &[], |cassette, provider, model| async move {
        let mut chat = chat_for(&cassette, provider, model);
        let url = served(&cassette, "https://filesamples.com/samples/video/mp4/sample_640x360.mp4");
        let response =
            chat.ask_with("What do you see in this video?", vec![Attachment::new(url)]).await.map_err(|e| e.to_string())?;
        check(matches_any(response.content(), &["beach", "ocean", "sand"]), response.content())?;
        content_ok(&response)?;
        let got = first_attachment(&chat, 0);
        check(got == (Some("sample_640x360.mp4".into()), "video/mp4".into()), format!("attachment {got:?}"))?;
        Ok(cassette)
    })
    .await;
    assert_eq!(ran, 1);
}

// ---- audio ------------------------------------------------------------------------------------

const AUDIO_MODELS: &[(&str, &str)] =
    &[("openai", "gpt-audio-mini"), ("gemini", "gemini-2.5-flash"), ("mistral", "voxtral-small-latest")];

#[tokio::test]
async fn audio_models_can_understand_audio() {
    let ran = each("chat audio models", "can understand audio", AUDIO_MODELS, &[], |cassette, provider, model| async move {
        let mut chat = chat_for(&cassette, provider, model);
        let response =
            chat.ask_with("What is being said?", vec![Attachment::new(fixture("ruby.wav"))]).await.map_err(|e| e.to_string())?;
        check(matches_any(response.content(), &["ruby"]), response.content())?;
        content_ok(&response)?;
        check(chat.messages()[0].content() == "What is being said?", "user content")?;
        check(first_attachment(&chat, 0) == (Some("ruby.wav".into()), "audio/wav".into()), "attachment")?;
        Ok(cassette)
    })
    .await;
    assert_eq!(ran, AUDIO_MODELS.len());
}

#[tokio::test]
async fn audio_models_can_understand_mp3_audio() {
    let ran = each("chat audio models", "can understand MP3 audio", AUDIO_MODELS, &[], |cassette, provider, model| async move {
        let mut chat = chat_for(&cassette, provider, model);
        let response =
            chat.ask_with("What is being said?", vec![Attachment::new(fixture("ruby.mp3"))]).await.map_err(|e| e.to_string())?;
        check(matches_any(response.content(), &["ruby"]), response.content())?;
        content_ok(&response)?;
        let a = &chat.messages()[0].attachments[0];
        check(a.filename.as_deref() == Some("ruby.mp3") && a.mime_type == "audio/mpeg", "attachment")?;
        check(a.format() == "mp3", "format")?;
        Ok(cassette)
    })
    .await;
    assert_eq!(ran, AUDIO_MODELS.len());
}

// ---- pdf --------------------------------------------------------------------------------------

const PDF_MODELS: &[(&str, &str)] = &[
    ("anthropic", "claude-haiku-4-5"),
    ("gemini", "gemini-2.5-flash"),
    ("openai", "gpt-5-nano"),
    ("openrouter", "gemini-2.5-flash"),
];

const PDF_WORDS: &[&str] = &["pdf", "document", "lorem", "sample"];

#[tokio::test]
async fn pdf_models_understand_pdfs() {
    let ran = each("chat pdf models", "understands PDFs", PDF_MODELS, &[], |cassette, provider, model| async move {
        let mut chat = chat_for(&cassette, provider, model);
        let response =
            chat.ask_with("Summarize this document", vec![Attachment::new(fixture("sample.pdf"))]).await.map_err(|e| e.to_string())?;
        check(matches_any(response.content(), PDF_WORDS), response.content())?;
        content_ok(&response)?;
        check(first_attachment(&chat, 0) == (Some("sample.pdf".into()), "application/pdf".into()), "attachment")?;
        let response = chat.ask("go on").await.map_err(|e| e.to_string())?;
        check(matches_any(response.content(), PDF_WORDS), response.content())?;
        Ok(cassette)
    })
    .await;
    assert_eq!(ran, PDF_MODELS.len());
}

#[tokio::test]
async fn pdf_models_handle_multiple_pdfs() {
    let words = &["pdf", "document", "lorem", "sample", "identical"];
    let ran = each("chat pdf models", "handles multiple PDFs", PDF_MODELS, &["https://pdfobject.com"], |cassette, provider, model| async move {
        let mut chat = chat_for(&cassette, provider, model);
        let url = served(&cassette, "https://pdfobject.com/pdf/sample.pdf");
        let with = vec![Attachment::new(fixture("sample.pdf")), Attachment::new(url)];
        let response = chat.ask_with("Compare these documents", with).await.map_err(|e| e.to_string())?;
        check(matches_any(response.content(), words), response.content())?;
        content_ok(&response)?;
        for i in 0..2 {
            check(first_attachment(&chat, i) == (Some("sample.pdf".into()), "application/pdf".into()), format!("attachment {i}"))?;
        }
        let response = chat.ask("go on").await.map_err(|e| e.to_string())?;
        check(matches_any(response.content(), words), response.content())?;
        Ok(cassette)
    })
    .await;
    assert_eq!(ran, PDF_MODELS.len());
}

#[tokio::test]
async fn pdf_models_can_handle_array_of_mixed_files_with_auto_detection() {
    let it = "can handle array of mixed files with auto-detection";
    let ran = each("chat pdf models", it, PDF_MODELS, &[], |cassette, provider, model| async move {
        let mut chat = chat_for(&cassette, provider, model);
        let prompt = "Describe the image, then summarize the PDF. Cover both files separately.";
        let with = vec![Attachment::new(fixture("ruby.png")), Attachment::new(fixture("sample.pdf"))];
        let response = chat.ask_with(prompt, with).await.map_err(|e| e.to_string())?;
        check(matches_any(response.content(), &["ruby", "gem", "logo"]), response.content())?;
        check(matches_any(response.content(), PDF_WORDS), response.content())?;
        check(chat.messages()[0].content() == prompt, "user content")?;
        check(first_attachment(&chat, 0) == (Some("ruby.png".into()), "image/png".into()), "image attachment")?;
        check(first_attachment(&chat, 1) == (Some("sample.pdf".into()), "application/pdf".into()), "pdf attachment")?;
        Ok(cassette)
    })
    .await;
    assert_eq!(ran, PDF_MODELS.len());
}

// ---- documents ----------------------------------------------------------------------------------

const DOCUMENT_MODELS: &[(&str, &str)] = &[("mistral", "mistral-small-latest"), ("openai", "gpt-5-nano")];

#[tokio::test]
async fn document_models_understand_docx_documents() {
    let ran = each("chat document models", "understands DOCX documents", DOCUMENT_MODELS, &[], |cassette, provider, model| async move {
        let mut chat = chat_for(&cassette, provider, model);
        let response = chat
            .ask_with(
                "What is the project codename in this document? Answer with only the code.",
                vec![Attachment::new(fixture("sample.docx"))],
            )
            .await
            .map_err(|e| e.to_string())?;
        let codename = regex::Regex::new(r"(?i)BLUE[-\s]?LANTERN[-\s]?42").unwrap();
        check(codename.is_match(response.content()), response.content())?;
        let a = &chat.messages()[0].attachments[0];
        check(a.filename.as_deref() == Some("sample.docx"), "filename")?;
        check(a.kind() == rust_llm::attachment::AttachmentType::Document, "document?")?;
        Ok(cassette)
    })
    .await;
    assert_eq!(ran, DOCUMENT_MODELS.len());
}

// ---- multimodal embeddings (embedding_spec.rb) ---------------------------------------------------

const TEST_DIMENSIONS: i64 = 768;

fn assert_floats(vectors: &Vectors) {
    match vectors {
        Vectors::Single(v) => assert_eq!(v.len() as i64, TEST_DIMENSIONS),
        Vectors::Batch(rows) => panic!("expected one vector, got {} rows", rows.len()),
    }
}

async fn embed_cassette(name: &str) -> Cassette {
    Cassette::start(name).await.unwrap_or_else(|| panic!("missing cassette {name}"))
}

#[tokio::test]
async fn openrouter_embeds_an_image_alongside_text() {
    let cassette =
        embed_cassette("embedding_multimodal_embeddings_openrouter_google_gemini-embedding-2_embeds_an_image_alongside_text").await;
    let e = embed(
        "The Ruby logo",
        EmbedOptions {
            model: Some("google/gemini-embedding-2"),
            provider: Some("openrouter"),
            dimensions: Some(TEST_DIMENSIONS),
            with: vec![Attachment::new(fixture("ruby.png"))],
            config: Some(config_for(&cassette, "openrouter")),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_floats(&e.vectors);
    assert!(e.tokens().input.unwrap_or(0) > 0);
    cassette.assert_all_matched().await;
}

#[tokio::test]
async fn openrouter_embeds_pdf_wav_and_mp4_without_text() {
    for (filename, slug) in [("sample.pdf", "sample_pdf"), ("ruby.wav", "ruby_wav"), ("ruby.mp4", "ruby_mp4")] {
        let name = format!("embedding_multimodal_embeddings_openrouter_google_gemini-embedding-2_embeds_{slug}");
        let cassette = embed_cassette(&name).await;
        let e = embed(
            None::<String>,
            EmbedOptions {
                model: Some("google/gemini-embedding-2"),
                provider: Some("openrouter"),
                dimensions: Some(TEST_DIMENSIONS),
                with: vec![Attachment::new(fixture(filename))],
                config: Some(config_for(&cassette, "openrouter")),
                ..Default::default()
            },
        )
        .await
        .unwrap_or_else(|err| panic!("{filename}: {err}"));
        assert_floats(&e.vectors);
        assert!(e.tokens().input.unwrap_or(0) > 0, "{filename}");
        cassette.assert_all_matched().await;
    }
}

#[tokio::test]
async fn gemini_embeds_text_with_custom_dimensions() {
    let cassette = embed_cassette("embedding_multimodal_embeddings_gemini_gemini-embedding-2_embeds_text_with_custom_dimensions").await;
    let e = embed(
        "Ruby is a programmer's best friend",
        EmbedOptions {
            model: Some("gemini-embedding-2"),
            provider: Some("gemini"),
            dimensions: Some(TEST_DIMENSIONS),
            config: Some(config_for(&cassette, "gemini")),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_floats(&e.vectors);
    assert_eq!(e.model, "gemini-embedding-2");
    cassette.assert_all_matched().await;
}

#[tokio::test]
async fn gemini_embeds_an_image_alongside_text() {
    let cassette = embed_cassette("embedding_multimodal_embeddings_gemini_gemini-embedding-2_embeds_an_image_alongside_text").await;
    let e = embed(
        "The Ruby logo",
        EmbedOptions {
            model: Some("gemini-embedding-2"),
            provider: Some("gemini"),
            dimensions: Some(TEST_DIMENSIONS),
            with: vec![Attachment::new(fixture("ruby.png"))],
            config: Some(config_for(&cassette, "gemini")),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_floats(&e.vectors);
    cassette.assert_all_matched().await;
}

#[tokio::test]
async fn raises_unsupported_attachment_error_on_providers_without_multimodal_embeddings() {
    let mut config = rust_llm::Config::default();
    config.set("openai_api_key", "test-key");
    config.set("openai_api_base", "http://127.0.0.1:9");
    let err = embed(
        "Ruby is a programmer's best friend",
        EmbedOptions {
            model: Some("text-embedding-3-small"),
            provider: Some("openai"),
            with: vec![Attachment::new(fixture("ruby.png"))],
            config: Some(std::sync::Arc::new(config)),
            ..Default::default()
        },
    )
    .await
    .unwrap_err();
    assert!(matches!(err, Error::UnsupportedAttachment(ref m) if m.contains("image/png")), "{err:?}");
}

#[tokio::test]
async fn rejects_attachments_alongside_multiple_texts() {
    let mut config = rust_llm::Config::default();
    config.set("gemini_api_key", "test-key");
    config.set("gemini_api_base", "http://127.0.0.1:9");
    let texts = vec!["Ruby is a programmer's best friend".to_string(), "Rails is a web framework".to_string()];
    let err = embed(
        texts,
        EmbedOptions {
            model: Some("gemini-embedding-2"),
            provider: Some("gemini"),
            with: vec![Attachment::new(fixture("ruby.png"))],
            config: Some(std::sync::Arc::new(config)),
            ..Default::default()
        },
    )
    .await
    .unwrap_err();
    assert!(matches!(err, Error::Argument(ref m) if m.contains("one text at a time")), "{err:?}");
}

// ---- prompt cache round-trip (chat_cache_until_here_spec.rb) -----------------------------------

fn cacheable_instructions() -> String {
    "You are a meticulous release engineer for the RubyLLM project. Review every\nchange for backwards compatibility, provider wire-format drift, cassette\nhygiene, and documentation accuracy before approving it for release.\n".repeat(150)
}

#[tokio::test]
async fn anthropic_writes_then_reads_the_prompt_cache() {
    let name = "chat_prompt_cache_round-trip_anthropic_claude-haiku-4-5_writes_then_reads_the_prompt_cache";
    let cassette = Cassette::start(name).await.expect("cassette");
    let ask = |cassette: &Cassette| {
        let mut chat = chat_for(cassette, "anthropic", "claude-haiku-4-5").with_caching(serde_json::json!(true)).unwrap();
        chat = chat.with_instructions(cacheable_instructions());
        chat.cache_until_here().unwrap();
        chat
    };
    let first = ask(&cassette).ask("Reply with exactly: OK").await.unwrap();
    let t = first.tokens();
    assert!(t.cache_write.unwrap_or(0) + t.cache_read.unwrap_or(0) > 0, "{t:?}");
    let second = ask(&cassette).ask("Reply with exactly: OK").await.unwrap();
    assert!(second.tokens().cache_read.unwrap_or(0) > 0, "{:?}", second.tokens());
    cassette.assert_all_matched().await;
}

#[tokio::test]
async fn openai_reuses_the_prompt_cache_with_a_shared_key() {
    let name = "chat_prompt_cache_round-trip_openai_gpt-5_2_reuses_the_prompt_cache_with_a_shared_key";
    let cassette = Cassette::start(name).await.expect("cassette");
    let mut second = None;
    for _ in 0..2 {
        let mut options = serde_json::Map::new();
        options.insert("key".into(), "rubyllm-test".into());
        let mut chat = chat_for(&cassette, "openai", "gpt-5.2").with_caching(serde_json::Value::Object(options)).unwrap().with_instructions(cacheable_instructions());
        second = Some(chat.ask("Reply with exactly: OK").await.unwrap());
    }
    let t = second.unwrap().tokens();
    assert!(t.cache_read.unwrap_or(0) + t.cache_write.unwrap_or(0) > 0, "{t:?}");
    cassette.assert_all_matched().await;
}
