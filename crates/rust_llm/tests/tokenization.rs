//! `RubyLLM.tokenize`, replayed from RubyLLM's `tokenization_*` cassettes. Assertions follow
//! `spec/ruby_llm/tokenization_spec.rb`. The Cohere cassette needs a provider RustLLM does not port.

mod support;

use rust_llm::{TokenizeOptions, tokenize};
use serde_json::json;
use support::{Cassette, config_for};

/// "tokenizes text with xai through the public API".
#[tokio::test]
async fn tokenizes_text_with_xai_through_the_public_api() {
    let cassette = Cassette::start("tokenization_tokenizes_text_with_xai_through_the_public_api")
        .await
        .expect("run bin/convert-cassettes 'tokenization_*'");
    let config = config_for(&cassette, "xai");
    let options = TokenizeOptions {
        model: Some("grok-4.3"),
        provider: Some("xai"),
        config: Some(config),
        ..Default::default()
    };
    let result = tokenize("Ruby makes AI useful.", options).await.unwrap();

    assert_eq!(result.model, "grok-4.3");
    assert!(!result.ids.is_empty());
    assert_eq!(result.ids[0], 94804);
    assert_eq!(result.count(), result.ids.len());
    assert!(result.raw.is_object());
    cassette.assert_all_matched().await;
}

/// "uses the context configuration and does not record generation usage": the context's key and
/// default model reach the request.
#[tokio::test]
async fn uses_the_context_configuration() {
    use wiremock::matchers::{body_json, header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/tokenize-text"))
        .and(header("Authorization", "Bearer isolated-key"))
        .and(body_json(json!({ "model": "grok-4.3", "text": "Ruby" })))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(
                json!({ "token_ids": [{ "token_id": 42, "string_token": "Ruby" }] }),
            ),
        )
        .expect(1)
        .mount(&server)
        .await;
    let context = rust_llm::context(|config| {
        config.set("xai_api_key", "isolated-key");
        config.set("xai_api_base", format!("{}/v1", server.uri()));
        config.default_model = "grok-4.3".into();
    });
    let result = context
        .tokenize(
            "Ruby",
            TokenizeOptions {
                provider: Some("xai"),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(result.ids, vec![42]);
    assert_eq!(result.model, "grok-4.3");
}

/// "rejects providers without a text tokenizer before a request".
#[tokio::test]
async fn providers_without_a_tokenizer_fail_before_a_request() {
    let context = rust_llm::context(|config| {
        config.set("openai_api_key", "test");
    });
    let options = TokenizeOptions {
        model: Some("gpt-5-nano"),
        provider: Some("openai"),
        ..Default::default()
    };
    let err = context.tokenize("Ruby", options).await.unwrap_err();
    assert!(
        err.to_string()
            .contains("doesn't support text tokenization"),
        "{err}"
    );
}
