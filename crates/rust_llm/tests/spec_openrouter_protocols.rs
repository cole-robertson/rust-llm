//! `spec/ruby_llm/protocols/openrouter/responses_spec.rb` "keeps normal chat and individual
//! operations on their existing adapters": OpenRouter chats default to Chat Completions, the
//! `openrouter_protocol` option moves chat to Responses, and single operations (transcription)
//! stay on OpenRouter's Chat Completions adapter regardless.

use std::sync::Arc;

use rust_llm::{Attachment, Config, ProtocolName, Provider, TranscribeOptions, transcribe};
use serde_json::json;
use wiremock::matchers::any;
use wiremock::{Mock, MockServer, ResponseTemplate};

/// `model_for(:openrouter, :provider_tools)`.
const MODEL: &str = "openai/gpt-5.2";

// spec: protocols/openrouter/responses_spec.rb:37 keeps normal chat and individual operations on their existing adapters
#[tokio::test]
async fn keeps_normal_chat_and_individual_operations_on_their_existing_adapters() {
    let server = MockServer::start().await;
    Mock::given(any())
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "text": "hello" })))
        .mount(&server)
        .await;
    let mut config = Config::default();
    config.set("openrouter_api_base", format!("{}/api/v1", server.uri()));
    config.set("openrouter_api_key", "test");
    config.max_retries = 0;
    let model = rust_llm::models().find(MODEL, Some("openrouter")).unwrap();

    assert_eq!(
        Provider::OpenRouter
            .resolve_protocol(None, &model, &config)
            .unwrap(),
        ProtocolName::ChatCompletions
    );

    config.set("openrouter_protocol", "responses");
    assert_eq!(
        Provider::OpenRouter
            .resolve_protocol(None, &model, &config)
            .unwrap(),
        ProtocolName::Responses
    );

    // `resolve_protocol(nil, nil, operation: :transcribe)` stays on Chat Completions: the
    // transcription request goes to its `audio/transcriptions` endpoint, not to `responses`.
    let transcription = transcribe(
        Attachment::new(format!(
            "{}/tests/fixtures/ruby.wav",
            env!("CARGO_MANIFEST_DIR")
        )),
        TranscribeOptions {
            model: Some("openai/whisper-1"),
            provider: Some("openrouter"),
            assume_model_exists: true,
            config: Some(Arc::new(config)),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(transcription.text.as_deref(), Some("hello"));
    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].url.path(), "/api/v1/audio/transcriptions");
}
