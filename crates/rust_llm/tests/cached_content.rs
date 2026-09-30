//! RubyLLM 2.0's `cached_content_spec.rb` (Gemini explicit caching), replayed from its cassettes.

mod support;

use std::sync::Arc;

use rust_llm::{CacheOptions, CachedContent, Chat, Config, Error, cache};
use serde_json::json;
use support::{Cassette, config_for};

/// gemini-2.5-flash requires at least 2,048 tokens for a cacheable prefix.
fn long_text() -> String {
    "The RubyLLM release engineering handbook describes cassette hygiene, provider wire \
     format drift, backwards compatibility checks, and documentation accuracy reviews \
     that every change must pass before release. "
        .repeat(300)
}

fn options(config: &Arc<Config>) -> CacheOptions<'static> {
    CacheOptions {
        model: "gemini-2.5-flash",
        ttl: Some(300.into()),
        config: Some(config.clone()),
        ..Default::default()
    }
}

// spec: cached_content_spec.rb:14 explicit caching round-trip > gemini/#{model_for(:gemini)} creates, uses, extends, and deletes a cache
#[tokio::test]
async fn creates_uses_extends_and_deletes_a_cache() {
    let cassette = Cassette::start("cachedcontent_explicit_caching_round-trip_gemini_gemini-2_5-flash_creates_uses_extends_and_deletes_a_cache")
        .await
        .expect("cassette");
    let config = config_for(&cassette, "gemini");
    let text = long_text();
    let mut cache = cache(
        &text,
        CacheOptions {
            instructions: Some("You are a meticulous release engineer."),
            ..options(&config)
        },
    )
    .await
    .unwrap();

    assert!(cache.name.starts_with("cachedContents/"));
    assert_eq!(cache.model.as_deref(), Some("gemini-2.5-flash"));
    assert_eq!(cache.provider, "gemini");
    assert!(cache.tokens.unwrap() > 2048);
    let expires_at = cache.expires_at.expect("expires_at");

    let response = Chat::with_config(
        config.clone(),
        Some("gemini-2.5-flash"),
        Some("gemini"),
        false,
    )
    .unwrap()
    .with_caching(json!({ "id": cache.name }))
    .unwrap()
    .ask("In one short sentence, what does the cached handbook describe?")
    .await
    .unwrap();
    assert!(!response.content().is_empty());
    assert_eq!(response.tokens().cache_read, cache.tokens);

    cache.renew(600).await.unwrap();
    assert!(cache.expires_at.unwrap() > expires_at);
    cache.delete().await.unwrap();
    cassette.assert_all_matched().await;
}

// spec: cached_content_spec.rb:60 explicit caching round-trip > finds an existing cache by name
#[tokio::test]
async fn finds_an_existing_cache_by_name() {
    let cassette = Cassette::start(
        "cachedcontent_explicit_caching_round-trip_finds_an_existing_cache_by_name",
    )
    .await
    .expect("cassette");
    let config = config_for(&cassette, "gemini");
    let created = cache(&long_text(), options(&config)).await.unwrap();
    let found = CachedContent::find_with_config(config.clone(), &created.name, Some("gemini"))
        .await
        .unwrap();
    assert_eq!(found.name, created.name);
    assert_eq!(found.tokens, created.tokens);
    created.delete().await.unwrap();
    cassette.assert_all_matched().await;
}

// spec: cached_content_spec.rb:73 unsupported providers > raises a clear error for providers without explicit caching
#[tokio::test]
async fn providers_without_explicit_caching_raise_a_clear_error() {
    let mut config = Config::default();
    config.set("anthropic_api_key", "test");
    let err = cache(
        "Some text",
        CacheOptions {
            model: "claude-haiku-4-5",
            config: Some(Arc::new(config)),
            ..Default::default()
        },
    )
    .await
    .unwrap_err();
    assert!(
        matches!(&err, Error::Api(m, _) if m == "Anthropic doesn't support explicit content caching"),
        "{err}"
    );
}

// spec: cached_content_spec.rb:81 RubyLLM shortcuts > creates caches through RubyLLM.cache
/// Ruby stubs `CachedContent.create` and checks `RubyLLM.cache` hands its arguments through; here
/// both run against one mock Gemini and must send the same request and return the same cache.
#[tokio::test]
async fn creates_caches_through_the_cache_shortcut() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1beta/cachedContents"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "name": "cachedContents/abc123", "model": "models/gemini-2.5-flash",
            "usageMetadata": { "totalTokenCount": 7809 }
        })))
        .mount(&server)
        .await;
    let mut config = Config::default();
    config.set("gemini_api_base", format!("{}/v1beta", server.uri()));
    config.set("gemini_api_key", "test");
    config.max_retries = 0;
    let config = Arc::new(config);
    let opts = || CacheOptions {
        model: "gemini-2.5-flash",
        config: Some(config.clone()),
        ..Default::default()
    };

    let via_shortcut = cache("text", opts()).await.unwrap();
    let via_create = CachedContent::create("text", opts()).await.unwrap();

    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 2);
    let body = |i: usize| serde_json::from_slice::<serde_json::Value>(&requests[i].body).unwrap();
    assert_eq!(body(0), body(1));
    assert_eq!(
        body(0),
        json!({ "model": "models/gemini-2.5-flash", "contents": [{ "role": "user", "parts": [{ "text": "text" }] }] })
    );
    assert_eq!(via_shortcut.name, via_create.name);
    assert_eq!(via_shortcut.tokens, Some(7809));
}
