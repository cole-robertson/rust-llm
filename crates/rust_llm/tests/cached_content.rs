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
    CacheOptions { model: "gemini-2.5-flash", ttl: Some(300.into()), config: Some(config.clone()), ..Default::default() }
}

// spec: "gemini/gemini-2.5-flash creates, uses, extends, and deletes a cache"
#[tokio::test]
async fn creates_uses_extends_and_deletes_a_cache() {
    let cassette = Cassette::start("cachedcontent_explicit_caching_round-trip_gemini_gemini-2_5-flash_creates_uses_extends_and_deletes_a_cache")
        .await
        .expect("cassette");
    let config = config_for(&cassette, "gemini");
    let text = long_text();
    let mut cache = cache(&text, CacheOptions { instructions: Some("You are a meticulous release engineer."), ..options(&config) }).await.unwrap();

    assert!(cache.name.starts_with("cachedContents/"));
    assert_eq!(cache.model.as_deref(), Some("gemini-2.5-flash"));
    assert_eq!(cache.provider, "gemini");
    assert!(cache.tokens.unwrap() > 2048);
    let expires_at = cache.expires_at.expect("expires_at");

    let response = Chat::with_config(config.clone(), Some("gemini-2.5-flash"), Some("gemini"), false)
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

// spec: "finds an existing cache by name"
#[tokio::test]
async fn finds_an_existing_cache_by_name() {
    let cassette = Cassette::start("cachedcontent_explicit_caching_round-trip_finds_an_existing_cache_by_name").await.expect("cassette");
    let config = config_for(&cassette, "gemini");
    let created = cache(&long_text(), options(&config)).await.unwrap();
    let found = CachedContent::find_with_config(config.clone(), &created.name, Some("gemini")).await.unwrap();
    assert_eq!(found.name, created.name);
    assert_eq!(found.tokens, created.tokens);
    created.delete().await.unwrap();
    cassette.assert_all_matched().await;
}

// spec: "raises a clear error for providers without explicit caching"
#[tokio::test]
async fn providers_without_explicit_caching_raise_a_clear_error() {
    let mut config = Config::default();
    config.set("anthropic_api_key", "test");
    let err = cache("Some text", CacheOptions { model: "claude-haiku-4-5", config: Some(Arc::new(config)), ..Default::default() })
        .await
        .unwrap_err();
    assert!(matches!(&err, Error::Api(m, _) if m == "Anthropic doesn't support explicit content caching"), "{err}");
}
