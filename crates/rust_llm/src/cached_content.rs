//! Port of `lib/ruby_llm/cached_content.rb`, `RubyLLM.cache`, and
//! `lib/ruby_llm/protocols/gemini/caches.rb`: provider-side prompt caches (Gemini's
//! `cachedContents`).
//!
//! ```ruby
//! cache = RubyLLM.cache(big_document, model: 'gemini-2.5-flash', ttl: 3600)
//! chat = RubyLLM.chat(model: 'gemini-2.5-flash').with_caching(id: cache)
//! cache.delete
//! ```
//!
//! ```ignore
//! let cache = rust_llm::cache(&big_document, CacheOptions { model: "gemini-2.5-flash", ttl: Some(3600.into()), ..Default::default() }).await?;
//! let chat = rust_llm::chat_with("gemini-2.5-flash")?.with_caching(json!({ "id": cache.name }))?;
//! cache.delete().await?;
//! ```

use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde_json::{Value, json};

use crate::attachment::Attachment;
use crate::chat::resolve_model;
use crate::config::Config;
use crate::error::{Error, Result};
use crate::providers::Provider;
use crate::transport::Connection;

/// A cache lifetime: seconds, or a provider duration string such as `"300s"`.
#[derive(Debug, Clone, PartialEq)]
pub enum Ttl {
    Seconds(i64),
    Duration(String),
}

impl From<i64> for Ttl {
    fn from(s: i64) -> Self {
        Ttl::Seconds(s)
    }
}

impl From<&str> for Ttl {
    fn from(s: &str) -> Self {
        Ttl::Duration(s.to_string())
    }
}

impl Ttl {
    /// `format_cache_ttl`.
    fn render(&self) -> String {
        match self {
            Ttl::Seconds(s) => format!("{s}s"),
            Ttl::Duration(d) => d.clone(),
        }
    }
}

/// Arguments of `CachedContent.create`.
#[derive(Default)]
pub struct CacheOptions<'a> {
    pub model: &'a str,
    pub provider: Option<&'a str>,
    pub ttl: Option<Ttl>,
    /// `instructions:`: a system prompt cached alongside the content.
    pub instructions: Option<&'a str>,
    /// `with:`: attachments cached with the content.
    pub with: Vec<Attachment>,
    /// `context:`: use this configuration instead of the global one.
    pub config: Option<Arc<Config>>,
}

/// `RubyLLM::CachedContent`: a provider-side prompt cache resource.
#[derive(Debug, Clone)]
pub struct CachedContent {
    /// The provider-assigned resource name, such as `"cachedContents/abc123"`.
    pub name: String,
    /// The model the cache was created for.
    pub model: Option<String>,
    /// The slug of the provider that stores the cache.
    pub provider: String,
    pub created_at: Option<DateTime<Utc>>,
    pub expires_at: Option<DateTime<Utc>>,
    /// The number of tokens stored in the cache.
    pub tokens: Option<i64>,
    /// The raw provider response.
    pub metadata: Value,
    connection: Option<Connection>,
}

/// `RubyLLM.cache(content, model:, ttl:, instructions:, provider:, with:)`.
pub async fn cache(content: &str, options: CacheOptions<'_>) -> Result<CachedContent> {
    CachedContent::create(content, options).await
}

impl CachedContent {
    /// `CachedContent.new(name:)`: a handle on an existing Gemini cache, e.g. to pass to
    /// `with_caching` (`json!({ "id": cache.as_ref() })`) without fetching it first.
    pub fn new(name: impl Into<String>) -> CachedContent {
        CachedContent {
            name: name.into(),
            metadata: json!({}),
            ..cache_from(&Value::Null)
        }
    }

    /// `CachedContent.create`. The content must exceed the model's minimum cacheable size.
    pub async fn create(content: &str, options: CacheOptions<'_>) -> Result<CachedContent> {
        let config = options.config.clone().unwrap_or_else(crate::config);
        let (model, provider) = resolve_model(options.model, options.provider, false)?;
        let connection = connection_for(provider, config)?;
        let mut with = options.with;
        for a in &mut with {
            a.load(connection.client()).await?;
        }
        let payload = render_cache_payload(
            content,
            &model.id,
            options.ttl.as_ref(),
            options.instructions,
            &with,
        )?;
        // `post caches_url, payload, idempotent: false`: never retried, since a retry after a lost
        // response would create a second cache.
        let resp = connection
            .send(
                reqwest::Method::POST,
                "cachedContents",
                &[],
                false,
                &|req| req.json(&payload),
            )
            .await?;
        let body = crate::transport::json_response(resp, Value::Null)
            .await?
            .body;
        Ok(parse_cache_response(&body, connection))
    }

    /// `CachedContent.find(name, provider:)`: `provider` defaults to the default model's.
    pub async fn find(name: &str, provider: Option<&str>) -> Result<CachedContent> {
        CachedContent::find_with_config(crate::config(), name, provider).await
    }

    pub async fn find_with_config(
        config: Arc<Config>,
        name: &str,
        provider: Option<&str>,
    ) -> Result<CachedContent> {
        let provider = match provider {
            Some(p) => Provider::resolve_or_err(p)?,
            None => resolve_model(&config.default_model, None, false)?.1,
        };
        let connection = connection_for(provider, config)?;
        let body = connection.get(&cache_name(name), &[]).await?.body;
        Ok(parse_cache_response(&body, connection))
    }

    /// `delete`: removes the cache from the provider.
    pub async fn delete(&self) -> Result<&Self> {
        self.connection()?
            .delete(&cache_name(&self.name), &[])
            .await?;
        Ok(self)
    }

    /// `renew(ttl:)`: extends the lifetime to `ttl` from now and updates `expires_at`.
    pub async fn renew(&mut self, ttl: impl Into<Ttl>) -> Result<&mut Self> {
        let connection = self.connection()?.clone();
        let payload = render_cache_update_payload(&ttl.into());
        let resp = connection
            .send(
                reqwest::Method::PATCH,
                &cache_name(&self.name),
                &[],
                true,
                &|req| req.json(&payload),
            )
            .await?;
        let body = crate::transport::json_response(resp, Value::Null)
            .await?
            .body;
        let refreshed = parse_cache_response(&body, connection);
        self.expires_at = refreshed.expires_at;
        self.metadata = refreshed.metadata;
        Ok(self)
    }

    fn connection(&self) -> Result<&Connection> {
        self.connection
            .as_ref()
            .ok_or_else(|| Error::Argument("This cache has no provider connection".into()))
    }
}

/// Only Gemini's protocol manages cache resources (`Protocol#cache_content` raises otherwise).
fn connection_for(provider: Provider, config: Arc<Config>) -> Result<Connection> {
    if provider != Provider::Gemini {
        return Err(Error::Api(
            format!(
                "{} doesn't support explicit content caching",
                provider.display()
            ),
            None,
        ));
    }
    provider.ensure_configured(&config)?;
    Connection::new(provider, config)
}

/// `Caches#render_cache_payload`.
pub fn render_cache_payload(
    content: &str,
    model: &str,
    ttl: Option<&Ttl>,
    instructions: Option<&str>,
    attachments: &[Attachment],
) -> Result<Value> {
    let parts = crate::protocols::gemini::format_content(Some(content), attachments)?;
    let mut payload = json!({ "model": format!("models/{model}"), "contents": [{ "role": "user", "parts": parts }] });
    if let Some(text) = instructions {
        payload["systemInstruction"] = json!({ "parts": [{ "text": text }] });
    }
    if let Some(ttl) = ttl {
        payload["ttl"] = ttl.render().into();
    }
    Ok(payload)
}

/// `Caches#render_cache_update_payload`: numeric ttls become duration strings.
pub fn render_cache_update_payload(ttl: &Ttl) -> Value {
    json!({ "ttl": ttl.render() })
}

/// `CachedContent` stands in for its name wherever a cache name is taken (`cache_name`).
impl AsRef<str> for CachedContent {
    fn as_ref(&self) -> &str {
        &self.name
    }
}

/// `Caches#cache_name`: bare ids get the collection prefix; a `CachedContent` is unwrapped to its
/// name.
pub fn cache_name(name: impl AsRef<str>) -> String {
    let name = name.as_ref();
    if name.contains('/') {
        name.to_string()
    } else {
        format!("cachedContents/{name}")
    }
}

fn time(v: Option<&Value>) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(v?.as_str()?)
        .ok()
        .map(|t| t.with_timezone(&Utc))
}

/// `Caches#parse_cache_response`.
fn parse_cache_response(data: &Value, connection: Connection) -> CachedContent {
    CachedContent {
        connection: Some(connection),
        ..cache_from(data)
    }
}

/// The resource fields of `parse_cache_response`, before a connection is attached.
fn cache_from(data: &Value) -> CachedContent {
    CachedContent {
        name: data
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        model: data
            .get("model")
            .and_then(Value::as_str)
            .and_then(|m| m.rsplit('/').next())
            .map(str::to_string),
        provider: Provider::Gemini.slug().into(),
        created_at: time(data.get("createTime")),
        expires_at: time(data.get("expireTime")),
        tokens: data
            .pointer("/usageMetadata/totalTokenCount")
            .and_then(Value::as_i64),
        metadata: data.clone(),
        connection: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // spec: protocols/gemini/caches_spec.rb:9 #render_cache_payload > renders the model, contents, system instruction, and ttl
    #[test]
    fn renders_the_model_contents_system_instruction_and_ttl() {
        let payload = render_cache_payload(
            "A long stable prefix.",
            "gemini-2.5-flash",
            Some(&300.into()),
            Some("You are a careful analyst."),
            &[],
        )
        .unwrap();
        assert_eq!(
            payload,
            json!({
                "model": "models/gemini-2.5-flash",
                "contents": [{ "role": "user", "parts": [{ "text": "A long stable prefix." }] }],
                "systemInstruction": { "parts": [{ "text": "You are a careful analyst." }] },
                "ttl": "300s"
            })
        );
    }

    // spec: protocols/gemini/caches_spec.rb:27 #render_cache_payload > omits system instruction and ttl when not given
    #[test]
    fn omits_system_instruction_and_ttl_when_not_given() {
        let payload = render_cache_payload("Prefix.", "gemini-2.5-flash", None, None, &[]).unwrap();
        assert_eq!(
            payload.as_object().unwrap().keys().collect::<Vec<_>>(),
            ["model", "contents"]
        );
    }

    // spec: protocols/gemini/caches_spec.rb:47 #render_cache_payload > passes duration strings through as ttl
    #[test]
    fn passes_duration_strings_through_as_ttl() {
        let payload = render_cache_payload(
            "Prefix.",
            "gemini-2.5-flash",
            Some(&"450s".into()),
            None,
            &[],
        )
        .unwrap();
        assert_eq!(payload["ttl"], "450s");
    }

    // spec: protocols/gemini/caches_spec.rb:91 #cache_name > prefixes bare ids with the collection name
    #[test]
    fn prefixes_bare_ids_with_the_collection_name() {
        assert_eq!(cache_name("abc123"), "cachedContents/abc123");
    }

    // spec: protocols/gemini/caches_spec.rb:95 #cache_name > keeps full resource names unchanged
    #[test]
    fn keeps_full_resource_names_unchanged() {
        assert_eq!(cache_name("cachedContents/abc123"), "cachedContents/abc123");
    }

    // spec: protocols/gemini/caches_spec.rb:99 #cache_name > unwraps CachedContent instances
    #[test]
    fn unwraps_cached_content_instances() {
        let cache = cache_from(&json!({ "name": "cachedContents/abc123" }));
        assert_eq!(cache_name(&cache), "cachedContents/abc123");
    }

    // spec: protocols/gemini/caches_spec.rb:35 #render_cache_payload > formats attachments through the Gemini media handling
    #[test]
    fn formats_attachments_through_the_gemini_media_handling() {
        let attachment = Attachment::from_bytes(b"fake-png".to_vec(), "diagram.png", None);
        let payload = render_cache_payload(
            "See the diagram.",
            "gemini-2.5-flash",
            None,
            None,
            &[attachment],
        )
        .unwrap();
        let parts = payload["contents"][0]["parts"].as_array().unwrap();
        assert_eq!(
            parts.first().unwrap(),
            &json!({ "text": "See the diagram." })
        );
        assert_eq!(
            parts.last().unwrap()["inline_data"]["mime_type"],
            "image/png"
        );
    }

    // spec: protocols/gemini/caches_spec.rb:57 #render_cache_update_payload > converts numeric ttls into duration strings
    #[test]
    fn converts_numeric_ttls_into_duration_strings() {
        assert_eq!(
            render_cache_update_payload(&600.into()),
            json!({ "ttl": "600s" })
        );
    }

    // spec: protocols/gemini/caches_spec.rb:76 #parse_cache_response > builds a CachedContent from the resource
    #[test]
    fn builds_a_cached_content_from_the_resource() {
        let data = json!({
            "name": "cachedContents/abc123",
            "model": "models/gemini-2.5-flash",
            "createTime": "2026-08-11T10:00:00Z",
            "expireTime": "2026-08-11T11:00:00Z",
            "usageMetadata": { "totalTokenCount": 7809 }
        });
        let cache = cache_from(&data);
        let at = |s| DateTime::parse_from_rfc3339(s).unwrap().with_timezone(&Utc);
        assert_eq!(cache.name, "cachedContents/abc123");
        assert_eq!(cache.model.as_deref(), Some("gemini-2.5-flash"));
        assert_eq!(cache.provider, "gemini");
        assert_eq!(cache.created_at, Some(at("2026-08-11T10:00:00Z")));
        assert_eq!(cache.expires_at, Some(at("2026-08-11T11:00:00Z")));
        assert_eq!(cache.tokens, Some(7809));
        assert_eq!(cache.metadata, data);
    }
}
