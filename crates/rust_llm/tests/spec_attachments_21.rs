//! Ports of RubyLLM 2.1's attachment and caching examples: `attachment_spec.rb` (original
//! resolution, serialized attributes), `chat_unsupported_attachments_spec.rb`
//! (`convert_unsupported_attachments`), `protocol_supported_attachment_spec.rb`, the original image
//! detail in `protocols/*/media_spec.rb`, and per-boundary cache lifetimes in
//! `chat_cache_until_here_spec.rb`.

mod support;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use rust_llm::protocols::supported_attachment;
use rust_llm::{
    Attachment, Chat, Config, Error, Message, ProtocolName, Provider, Resolution, Role,
};
use serde_json::{Value, json};
use support::{Cassette, config_for};

fn fixture(name: &str) -> String {
    format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
}

/// `include_context 'with configured RubyLLM'`: every provider configured, nothing sent.
fn configured() -> Arc<Config> {
    let mut c = Config::default();
    for p in [
        "anthropic",
        "openai",
        "gemini",
        "openrouter",
        "xai",
        "mistral",
        "deepseek",
        "perplexity",
    ] {
        c.set(format!("{p}_api_key"), "test");
    }
    c.max_retries = 0;
    Arc::new(c)
}

fn chat(model: &str, provider: &str) -> Chat {
    Chat::with_config(configured(), Some(model), Some(provider), false).unwrap()
}

// ---- attachment_spec.rb --------------------------------------------------------------------------

// spec: attachment_spec.rb:266 accepts original resolution
#[test]
fn accepts_original_resolution() {
    let a = Attachment::from_bytes(b"png".to_vec(), "page.png", None)
        .with_resolution(Resolution::Original);
    assert_eq!(a.resolution, Some(Resolution::Original));
    assert_eq!(Resolution::parse("original").unwrap(), Resolution::Original);
}

fn original() -> Attachment {
    Attachment::new(fixture("ruby.png"))
        .with_filename_public("custom.png")
        .with_resolution(Resolution::High)
}

/// `Attachment.new(path, filename: 'custom.png')`.
trait WithFilename {
    fn with_filename_public(self, name: &str) -> Attachment;
}

impl WithFilename for Attachment {
    fn with_filename_public(self, name: &str) -> Attachment {
        Attachment::from_h(&json!({ "source": self.to_h()["source"], "filename": name })).unwrap()
    }
}

// spec: attachment_spec.rb:287 serialized attributes > serializes the source, filename, and resolution
#[test]
fn serializes_the_source_filename_and_resolution() {
    assert_eq!(
        original().to_h(),
        json!({ "type": "image", "source": fixture("ruby.png"), "filename": "custom.png", "resolution": "high" })
    );
}

// spec: attachment_spec.rb:292 serialized attributes > omits an unset resolution
#[test]
fn omits_an_unset_resolution() {
    let h = Attachment::new(fixture("ruby.png")).to_h();
    assert!(h.get("resolution").is_none(), "{h}");
}

// spec: attachment_spec.rb:296 serialized attributes > rebuilds an attachment from Symbol keys
// spec: attachment_spec.rb:304 serialized attributes > rebuilds an attachment from String keys and a String resolution
#[tokio::test]
async fn rebuilds_an_attachment_from_its_attributes() {
    // Rust has one key type: the Hash and its JSON round trip are the same value.
    let attributes: Value = serde_json::from_str(&original().to_h().to_string()).unwrap();
    assert_eq!(attributes["filename"], "custom.png");
    assert_eq!(attributes["resolution"], "high");
    let mut rebuilt = Attachment::from_h(&attributes).unwrap();
    assert_eq!(
        rebuilt.source,
        rust_llm::attachment::Source::Path(fixture("ruby.png").into())
    );
    assert_eq!(rebuilt.filename.as_deref(), Some("custom.png"));
    assert_eq!(rebuilt.mime_type, "image/png");
    assert_eq!(rebuilt.resolution, Some(Resolution::High));
    assert_eq!(
        rebuilt.content().await.unwrap(),
        std::fs::read(fixture("ruby.png")).unwrap()
    );
    // An unknown resolution name is refused (`resolution must be one of`).
    assert!(matches!(
        Attachment::from_h(&json!({ "source": fixture("ruby.png"), "resolution": "huge" })),
        Err(Error::Argument(_))
    ));
}

// spec: attachment_spec.rb:314 serialized attributes > rebuilds a URL attachment from JSON attributes
#[test]
fn rebuilds_a_url_attachment_from_json_attributes() {
    let attributes: Value = serde_json::from_str(
        &Attachment::new("https://example.com/ruby.png")
            .to_h()
            .to_string(),
    )
    .unwrap();
    let rebuilt = Attachment::from_h(&attributes).unwrap();
    assert_eq!(rebuilt.url(), Some("https://example.com/ruby.png"));
    assert_eq!(rebuilt.filename.as_deref(), Some("ruby.png"));
    assert_eq!(rebuilt.mime_type, "image/png");
}

// spec: attachment_spec.rb:323 serialized attributes > passes the configuration to rebuilt attachments
#[test]
fn rebuilt_attachments_in_a_message_keep_their_attributes() {
    // An attachment carries no configuration here (the chat's connection reads it), so the
    // configuration-independent half is what a rebuilt message must keep.
    let mut m = Message::user("Look");
    m.attachments.push(original());
    let restored = Message::from_h(&m.to_h()).unwrap();
    assert_eq!(
        restored.attachments[0].filename.as_deref(),
        Some("custom.png")
    );
    assert_eq!(restored.attachments[0].resolution, Some(Resolution::High));
}

// ---- protocols/*/media_spec.rb: original image detail --------------------------------------------

fn detail(provider: &str, model: &str, protocol: ProtocolName, resolution: Resolution) -> Value {
    let mut chat = chat(model, provider).with_protocol(protocol);
    let image = Attachment::from_bytes(
        std::fs::read(fixture("ruby.png")).unwrap(),
        "ruby.png",
        None,
    )
    .with_resolution(resolution);
    chat.ask_later_with("Read the small print", vec![image])
        .unwrap();
    let payload = chat.render().unwrap();
    match protocol {
        ProtocolName::Responses => payload["input"][0]["content"][1]["detail"].clone(),
        _ => {
            let messages = payload["messages"].as_array().unwrap();
            messages.last().unwrap()["content"][1]["image_url"]["detail"].clone()
        }
    }
}

// spec: protocols/chat_completions/media_spec.rb:69 .format_content > maps #{resolution} resolution to high detail even when original detail is enabled
#[test]
fn chat_completions_maps_other_resolutions_to_high_detail_for_openai() {
    for resolution in [Resolution::Medium, Resolution::High, Resolution::UltraHigh] {
        assert_eq!(
            detail(
                "openai",
                "gpt-5-nano",
                ProtocolName::ChatCompletions,
                resolution
            ),
            json!("high"),
            "{resolution:?}"
        );
    }
}

// spec: protocols/chat_completions/media_spec.rb:83 .format_content > maps original resolution to #{detail} image detail for #{provider_class}
#[test]
fn chat_completions_maps_original_resolution_per_provider() {
    for (provider, model, expected) in [
        ("openai", "gpt-5-nano", "original"),
        ("xai", "grok-4-1-fast-non-reasoning", "high"),
        ("openrouter", "anthropic/claude-haiku-4.5", "high"),
    ] {
        assert_eq!(
            detail(
                provider,
                model,
                ProtocolName::ChatCompletions,
                Resolution::Original
            ),
            json!(expected),
            "{provider}"
        );
    }
}

// spec: protocols/responses/media_spec.rb:76 .format_content > maps original resolution to #{detail} image detail for #{provider_class}
#[test]
fn responses_maps_original_resolution_per_provider() {
    for (provider, model, expected) in [
        ("openai", "gpt-5-nano", "original"),
        ("xai", "grok-4-1-fast-non-reasoning", "high"),
        ("openrouter", "anthropic/claude-haiku-4.5", "high"),
    ] {
        assert_eq!(
            detail(
                provider,
                model,
                ProtocolName::Responses,
                Resolution::Original
            ),
            json!(expected),
            "{provider}"
        );
    }
}

// ---- protocol_supported_attachment_spec.rb -------------------------------------------------------

// spec: protocol_supported_attachment_spec.rb:36 protocol_class.name > matches rendering support for #{extension} attachments
#[test]
fn supported_attachment_matches_rendering_support() {
    // Azure, Bedrock (Converse), and Cohere are providers this port leaves out.
    let cases: &[(ProtocolName, &str, &str, &[&str])] = &[
        (
            ProtocolName::Anthropic,
            "anthropic",
            "claude-haiku-4-5",
            &["png", "pdf", "txt"],
        ),
        (
            ProtocolName::ChatCompletions,
            "openai",
            "gpt-5-nano",
            &["png", "wav", "pdf", "txt"],
        ),
        (
            ProtocolName::Responses,
            "openai",
            "gpt-5-nano",
            &["png", "pdf", "docx", "pptx", "txt"],
        ),
        (
            ProtocolName::Gemini,
            "gemini",
            "gemini-2.5-flash",
            &["png", "wav", "mov", "pdf", "txt"],
        ),
        (
            ProtocolName::Interactions,
            "gemini",
            "gemini-2.5-flash",
            &["png", "wav", "mov", "pdf", "txt"],
        ),
        (
            ProtocolName::ChatCompletions,
            "gpustack",
            "qwen3",
            &["png", "wav", "mov", "txt"],
        ),
        (
            ProtocolName::ChatCompletions,
            "mistral",
            "mistral-small-latest",
            &["png", "wav", "pdf", "docx", "pptx", "txt"],
        ),
        (
            ProtocolName::Conversations,
            "mistral",
            "mistral-small-latest",
            &["png", "wav", "pdf", "docx", "pptx", "txt"],
        ),
        (
            ProtocolName::ChatCompletions,
            "ollama",
            "qwen3",
            &["png", "wav", "txt"],
        ),
        (
            ProtocolName::ChatCompletions,
            "openrouter",
            "anthropic/claude-haiku-4.5",
            &["png", "wav", "mov", "pdf", "txt"],
        ),
        (
            ProtocolName::ChatCompletions,
            "perplexity",
            "sonar",
            &["png", "pdf", "docx", "pptx", "txt"],
        ),
        (
            ProtocolName::Responses,
            "perplexity",
            "openai/gpt-5-mini",
            &["png", "txt"],
        ),
        (
            ProtocolName::ChatCompletions,
            "deepseek",
            "deepseek-v4-flash",
            &["png", "txt"],
        ),
        (
            ProtocolName::Responses,
            "deepseek",
            "deepseek-v4-flash",
            &["png", "txt"],
        ),
        (
            ProtocolName::ChatCompletions,
            "hetzner",
            "Qwen3.8-27B",
            &["png", "txt"],
        ),
        (
            ProtocolName::ChatCompletions,
            "xai",
            "grok-4-1-fast-non-reasoning",
            &["png", "txt"],
        ),
    ];
    for &(protocol, provider, model, supported) in cases {
        let kind = Provider::resolve(provider).unwrap();
        for extension in ["png", "wav", "mov", "pdf", "docx", "pptx", "txt", "bin"] {
            let attachment =
                Attachment::from_bytes(b"file bytes".to_vec(), format!("sample.{extension}"), None);
            let expected = supported.contains(&extension);
            assert_eq!(
                supported_attachment(protocol, kind, &attachment),
                expected,
                "{protocol:?}/{provider} {extension}"
            );
            // Rendering agrees with the predicate.
            let mut config = Config::default();
            config.set(format!("{provider}_api_key"), "test");
            config.set(format!("{provider}_api_base"), "http://localhost:1");
            let mut chat = Chat::with_config(Arc::new(config), Some(model), Some(provider), true)
                .unwrap()
                .with_protocol(protocol);
            chat.ask_later_with("Read this.", vec![attachment]).unwrap();
            let rendered = chat.render();
            if expected {
                assert!(
                    rendered.is_ok(),
                    "{protocol:?}/{provider} {extension}: {rendered:?}"
                );
            } else {
                let err = rendered.unwrap_err();
                let refused = if protocol == ProtocolName::Interactions {
                    matches!(err, Error::Argument(_))
                } else {
                    matches!(err, Error::UnsupportedAttachment(_))
                };
                assert!(refused, "{protocol:?}/{provider} {extension}: {err:?}");
            }
        }
    }
}

// ---- chat_unsupported_attachments_spec.rb --------------------------------------------------------

fn document() -> Attachment {
    Attachment::from_bytes(b"office document".to_vec(), "report.docx", None)
}

fn replacement() -> Attachment {
    Attachment::from_bytes(b"Extracted report".to_vec(), "report.txt", None)
}

/// `chat` with the spec's `before`: one user message carrying the Word document.
fn anthropic_with_document() -> (Chat, Attachment) {
    let doc = document();
    let mut chat = chat("claude-haiku-4-5", "anthropic");
    let mut m = Message::user("Summarize this report.");
    m.attachments.push(doc.clone());
    chat.add_message(m);
    (chat, doc)
}

fn rendered(chat: &Chat) -> String {
    chat.render().unwrap().to_string()
}

// spec: chat_unsupported_attachments_spec.rb:15 keeps the existing error when no callback handles the attachment
#[test]
fn keeps_the_existing_error_without_a_converter() {
    let (chat, _) = anthropic_with_document();
    assert!(matches!(
        chat.render(),
        Err(Error::UnsupportedAttachment(_))
    ));
}

// spec: chat_unsupported_attachments_spec.rb:19 uses a replacement for the request while keeping the original transcript
#[test]
fn uses_a_replacement_while_keeping_the_original_transcript() {
    let (chat, doc) = anthropic_with_document();
    let received = Arc::new(Mutex::new(Vec::new()));
    let seen = received.clone();
    let chat = chat.convert_unsupported_attachments(move |a| {
        seen.lock().unwrap().push(a.clone());
        Ok(Some(replacement()))
    });
    assert!(rendered(&chat).contains("Extracted report"));
    assert_eq!(*received.lock().unwrap(), vec![doc.clone()]);
    assert_eq!(chat.messages().last().unwrap().attachments, vec![doc]);
}

// spec: chat_unsupported_attachments_spec.rb:31 converts an attachment once and reuses the replacement on later requests
#[test]
fn converts_an_attachment_once() {
    let (chat, _) = anthropic_with_document();
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = calls.clone();
    let chat = chat.convert_unsupported_attachments(move |_| {
        counter.fetch_add(1, Ordering::SeqCst);
        Ok(Some(replacement()))
    });
    for _ in 0..2 {
        assert!(rendered(&chat).contains("Extracted report"));
    }
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

// spec: chat_unsupported_attachments_spec.rb:43 keeps the existing error when callbacks return nil
#[test]
fn keeps_the_existing_error_when_converters_return_none() {
    let (chat, _) = anthropic_with_document();
    let chat = chat.convert_unsupported_attachments(|_| Ok(None));
    assert!(matches!(
        chat.render(),
        Err(Error::UnsupportedAttachment(_))
    ));
}

// spec: chat_unsupported_attachments_spec.rb:49 uses the first replacement from callbacks registered in order
#[test]
fn uses_the_first_replacement_in_registration_order() {
    let (chat, _) = anthropic_with_document();
    let calls = Arc::new(Mutex::new(Vec::new()));
    let (first, second) = (calls.clone(), calls.clone());
    let chat = chat
        .convert_unsupported_attachments(move |_| {
            first.lock().unwrap().push("first");
            Ok(None)
        })
        .convert_unsupported_attachments(move |_| {
            second.lock().unwrap().push("second");
            Ok(Some(replacement()))
        })
        .convert_unsupported_attachments(|_| panic!("Already replaced"));
    chat.render().unwrap();
    assert_eq!(*calls.lock().unwrap(), vec!["first", "second"]);
}

// spec: chat_unsupported_attachments_spec.rb:66 does not call the handler when the new provider can render the original
#[test]
fn does_not_convert_when_the_new_provider_renders_the_original() {
    let (chat, doc) = anthropic_with_document();
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = calls.clone();
    let chat = chat.convert_unsupported_attachments(move |_| {
        counter.fetch_add(1, Ordering::SeqCst);
        Ok(Some(replacement()))
    });
    chat.render().unwrap();
    let chat = chat.with_model("gpt-5-nano", Some("openai")).unwrap();
    let payload = rendered(&chat);
    assert!(
        payload.contains("input_file") && payload.contains("report.docx"),
        "{payload}"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(chat.messages().last().unwrap().attachments, vec![doc]);
}

// spec: chat_unsupported_attachments_spec.rb:80 does not infer attachment support from the model catalog
#[test]
fn does_not_infer_support_from_the_model_catalog() {
    // An unlisted model has no catalog modalities; support comes from the protocol alone.
    let mut chat = Chat::with_config(
        configured(),
        Some("claude-unlisted"),
        Some("anthropic"),
        true,
    )
    .unwrap()
    .convert_unsupported_attachments(|_| panic!("Images are renderable"));
    chat.set_messages(vec![{
        let mut m = Message::user("Describe this.");
        m.attachments.push(Attachment::from_bytes(
            std::fs::read(fixture("ruby.png")).unwrap(),
            "ruby.png",
            None,
        ));
        m
    }]);
    assert!(rendered(&chat).contains("image"));
}

// spec: chat_unsupported_attachments_spec.rb:89 rejects a replacement the same protocol cannot render without calling the handler again
#[test]
fn rejects_a_replacement_the_protocol_cannot_render() {
    let (chat, _) = anthropic_with_document();
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = calls.clone();
    let chat = chat.convert_unsupported_attachments(move |_| {
        counter.fetch_add(1, Ordering::SeqCst);
        Ok(Some(document()))
    });
    assert!(matches!(
        chat.render(),
        Err(Error::UnsupportedAttachment(_))
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

// spec: chat_unsupported_attachments_spec.rb:106 propagates errors from the application callback
#[test]
fn propagates_errors_from_the_converter() {
    let (chat, _) = anthropic_with_document();
    let chat =
        chat.convert_unsupported_attachments(|_| Err(Error::Argument("Extraction failed".into())));
    let err = chat.render().unwrap_err();
    assert_eq!(err.to_string(), "Extraction failed");
}

struct Reader;

#[async_trait::async_trait]
impl rust_llm::Agent for Reader {
    fn model(&self) -> Option<&str> {
        Some("claude-haiku-4-5")
    }
    fn provider(&self) -> Option<&str> {
        Some("anthropic")
    }
}

// spec: chat_unsupported_attachments_spec.rb:112 allows the same callback through an agent
#[test]
fn allows_the_same_converter_through_an_agent() {
    // `Agent#convert_unsupported_attachments` delegates to its chat; a Rust agent builds that chat.
    let doc = document();
    let mut chat = rust_llm::Agent::apply(&Reader, chat("claude-haiku-4-5", "anthropic"))
        .unwrap()
        .convert_unsupported_attachments(|_| Ok(Some(replacement())));
    let mut m = Message::user("Summarize this report.");
    m.attachments.push(doc);
    chat.add_message(m);
    assert!(rendered(&chat).contains("Extracted report"));
}

// spec: chat_unsupported_attachments_spec.rb:135 sends converted Office documents through Anthropic
#[tokio::test]
async fn sends_converted_office_documents_through_anthropic() {
    let cassette = Cassette::start("chat_sends_converted_office_documents_through_anthropic")
        .await
        .unwrap();
    let config = config_for(&cassette, "anthropic");
    let doc = document();
    let mut chat = Chat::with_config(config, Some("claude-haiku-4-5"), Some("anthropic"), false)
        .unwrap()
        .convert_unsupported_attachments(|_| {
            Ok(Some(Attachment::from_bytes(
                b"The project codename is CEDAR881.".to_vec(),
                "report.txt",
                None,
            )))
        });
    let response = chat
        .ask_with(
            "Reply with the project codename in this report and nothing else.",
            vec![doc.clone()],
        )
        .await
        .unwrap();
    assert!(response.content.as_deref().unwrap().contains("CEDAR881"));
    assert_eq!(chat.messages()[0].attachments, vec![doc]);
    cassette.assert_all_matched().await;
}

// ---- chat_cache_until_here_spec.rb ---------------------------------------------------------------

// spec: chat_cache_until_here_spec.rb:236 #cache_until_here > gives the boundary its own lifetime
#[test]
fn gives_the_boundary_its_own_lifetime() {
    let mut chat = chat("claude-haiku-4-5", "anthropic").with_instructions("Stable instructions");
    chat.cache_until_here_with(Some("1h")).unwrap();
    let last = chat.messages().last().unwrap();
    assert!(last.cache_until_here);
    assert_eq!(last.cache_ttl.as_deref(), Some("1h"));
}

// spec: chat_cache_until_here_spec.rb:243 #cache_until_here > gives an instruction boundary its own lifetime from with_instructions
#[test]
fn gives_an_instruction_boundary_its_own_lifetime() {
    let mut chat = chat("claude-haiku-4-5", "anthropic");
    chat.set_instructions_with(
        Some("Stable instructions".into()),
        false,
        &json!({ "ttl": "1h" }),
    )
    .unwrap();
    assert_eq!(
        chat.messages().last().unwrap().cache_ttl.as_deref(),
        Some("1h")
    );
}

// spec: chat_cache_until_here_spec.rb:249 #cache_until_here > rejects boundary options it does not know
#[test]
fn rejects_boundary_options_it_does_not_know() {
    let mut chat = chat("claude-haiku-4-5", "anthropic");
    let err = chat
        .set_instructions_with(
            Some("Stable instructions".into()),
            false,
            &json!({ "scope": "user" }),
        )
        .unwrap_err();
    assert!(matches!(err, Error::Argument(_)));
    assert!(
        err.to_string()
            .contains("cache_until_here accepts true, false, or ttl:"),
        "{err}"
    );
}

// spec: chat_cache_until_here_spec.rb:254 #cache_until_here > renders the boundary lifetime ahead of the chat lifetime on Anthropic
#[test]
fn renders_the_boundary_lifetime_ahead_of_the_chat_lifetime_on_anthropic() {
    let mut chat = chat("claude-haiku-4-5", "anthropic")
        .with_caching(json!({}))
        .unwrap()
        .with_instructions("Stable policy");
    chat.cache_until_here_with(Some("1h")).unwrap();
    chat.ask_later("Long context").unwrap();
    chat.cache_until_here().unwrap();

    let payload = chat.render().unwrap();
    let system = payload["system"].as_array().unwrap();
    assert_eq!(
        system.last().unwrap()["cache_control"],
        json!({ "type": "ephemeral", "ttl": "1h" })
    );
    let messages = payload["messages"].as_array().unwrap();
    let content = messages.last().unwrap()["content"].as_array().unwrap();
    assert_eq!(
        content.last().unwrap()["cache_control"],
        json!({ "type": "ephemeral" })
    );
    assert_eq!(payload["cache_control"], json!({ "type": "ephemeral" }));
}

// spec: chat_cache_until_here_spec.rb:266 #cache_until_here > keeps the lifetime when a message round-trips through to_h
#[test]
fn keeps_the_lifetime_through_to_h() {
    let message = Message::new(Role::System, Some("Stable policy".to_string()))
        .with_cache_until_here(Some("1h"));
    let restored = Message::from_h(&message.to_h()).unwrap();
    assert!(restored.cache_until_here);
    assert_eq!(restored.cache_ttl.as_deref(), Some("1h"));
}

/// `cacheable_instructions`: the spec's three-line paragraph, 150 times.
fn cacheable_instructions() -> String {
    "You are a meticulous release engineer for the RubyLLM project. Review every\n\
     change for backwards compatibility, provider wire-format drift, cassette\n\
     hygiene, and documentation accuracy before approving it for release.\n"
        .repeat(150)
}

// spec: chat_cache_until_here_spec.rb:304 prompt cache round-trip > anthropic/#{model_for(:anthropic)} writes a one-hour boundary and prices it as one
#[tokio::test]
async fn anthropic_writes_a_one_hour_boundary_and_prices_it_as_one() {
    let cassette = Cassette::start(
        "chat_prompt_cache_round-trip_anthropic_claude-haiku-4-5_writes_a_one-hour_boundary_and_prices_it_as_one",
    )
    .await
    .unwrap();
    let config = config_for(&cassette, "anthropic");
    let mut chat = Chat::with_config(config, Some("claude-haiku-4-5"), Some("anthropic"), false)
        .unwrap()
        .with_caching(json!({}))
        .unwrap()
        .with_instructions(format!(
            "{}\nKeep this prefix for an hour.",
            cacheable_instructions()
        ));
    chat.cache_until_here_with(Some("1h")).unwrap();

    let response = chat.ask("Reply with exactly: OK").await.unwrap();

    let tokens = response.tokens();
    let by_ttl = tokens.cache_write_by_ttl.clone().unwrap();
    let one_hour_tokens = by_ttl["1h"].as_i64().unwrap();
    assert!(one_hour_tokens > 0);
    let input_price = response
        .model_info()
        .unwrap()
        .pricing
        .text_tokens()
        .standard
        .unwrap()
        .input_per_million
        .unwrap();
    let one_hour = one_hour_tokens as f64 * input_price * 2.0 / 1_000_000.0;
    assert!(response.cost(None).cache_write.unwrap() >= one_hour);
    cassette.assert_all_matched().await;
}
