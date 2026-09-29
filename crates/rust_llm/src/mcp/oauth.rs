//! Port of `lib/ruby_llm/mcp/oauth.rb` and `lib/ruby_llm/mcp/oauth/memory_store.rb`: OAuth for
//! one MCP server and one owner, following the 2026-07-28 authorization spec: protected resource
//! metadata, authorization server discovery, client ID metadata documents or dynamic registration
//! when no client is configured, PKCE, resource indicators, issuer checks, and token refresh.
//! Credentials live in the configured `mcp_credential_store`, keyed by owner and server.
//!
//! ```ruby
//! class Linear < RubyLLM::MCP
//!   url "https://mcp.linear.app/mcp"
//!   inputs :user
//!   oauth owner: :user
//! end
//! redirect_to Linear.new(user: current_user).authorization_url(redirect_uri: mcp_callback_url)
//! Linear.new(user: current_user).authorize(params)
//! ```
//!
//! ```no_run
//! # use rust_llm::mcp::{Mcp, OAuthSettings};
//! # async fn run(user: String, params: Vec<(String, String)>) -> rust_llm::Result<()> {
//! let linear = Mcp::url("https://mcp.linear.app/mcp")
//!     .oauth(OAuthSettings::new().owner(user))
//!     .build()?;
//! let url = linear.authorization_url("https://app.example.com/mcp/callback").await?;
//! // ... the user comes back to the callback with `params` ...
//! linear.authorize(params).await?;
//! # Ok(()) }
//! ```

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use base64::Engine;
use base64::engine::general_purpose::{STANDARD, URL_SAFE_NO_PAD};
use rand::RngCore;
use regex::Regex;
use reqwest::Url;
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use super::http::Authorization;
use super::{Http, McpError};
use crate::config::Config;
use crate::error::{Error, Result};

const PENDING_FOR: i64 = 600;
const REFRESH_EARLY: i64 = 60;
const SERVER_FIELDS: &[&str] =
    &["issuer", "token_endpoint", "token_endpoint_auth_methods_supported", "authorization_response_iss_parameter_supported"];

/// The parameters of a Bearer `WWW-Authenticate` challenge (`OAuth.challenge`).
pub type Challenge = HashMap<String, String>;

/// Names whose credentials these are, evaluated per MCP like `oauth owner: :user`.
pub type OwnerSource = Arc<dyn Fn() -> Option<String> + Send + Sync>;

/// `config.mcp_credential_store`: where MCP OAuth credentials live. A store responds to
/// `read(key)`, `write(key, data, owner:)`, and `delete(key)`. `owner` is the owner's key (a
/// GlobalID string such as `gid://app/Chat/1` for records), or `None` for client registrations.
#[async_trait]
pub trait CredentialStore: Send + Sync {
    async fn read(&self, key: &str) -> Result<Option<Value>>;
    async fn write(&self, key: &str, data: Value, owner: Option<&str>) -> Result<()>;
    async fn delete(&self, key: &str) -> Result<()>;
}

impl std::fmt::Debug for dyn CredentialStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CredentialStore")
    }
}

/// `MCP::OAuth::MemoryStore`: keeps OAuth credentials in memory for the life of the process.
/// Loco applications keep them in the database instead (`rust_llm_loco::McpCredentialStore`).
#[derive(Debug, Default)]
pub struct MemoryStore {
    credentials: Mutex<HashMap<String, Value>>,
}

impl MemoryStore {
    pub fn new() -> MemoryStore {
        MemoryStore::default()
    }
}

#[async_trait]
impl CredentialStore for MemoryStore {
    async fn read(&self, key: &str) -> Result<Option<Value>> {
        Ok(lock(&self.credentials).get(key).cloned())
    }

    async fn write(&self, key: &str, data: Value, _owner: Option<&str>) -> Result<()> {
        lock(&self.credentials).insert(key.to_string(), data);
        Ok(())
    }

    async fn delete(&self, key: &str) -> Result<()> {
        lock(&self.credentials).remove(key);
        Ok(())
    }
}

/// `OAuth.memory_store`: the process-wide store used when `mcp_credential_store` is unset.
static MEMORY_STORE: LazyLock<Arc<MemoryStore>> = LazyLock::new(|| Arc::new(MemoryStore::new()));

/// The settings of `oauth owner:, scopes:, client_id:, client_secret:`.
#[derive(Clone, Default)]
pub struct OAuthSettings {
    /// `owner:`: whose credentials these are. Declared but resolving to `None` is an error.
    pub owner: Option<OwnerSource>,
    /// `scopes:`: overrides the scopes the server asks for.
    pub scopes: Option<Vec<String>>,
    /// `client_id:`/`client_secret:`: an app you registered, which servers such as Slack require.
    pub client_id: Option<String>,
    pub client_secret: Option<String>,
}

impl OAuthSettings {
    pub fn new() -> OAuthSettings {
        OAuthSettings::default()
    }

    /// `owner: "ada"`.
    pub fn owner(self, owner: impl Into<String>) -> Self {
        let owner = owner.into();
        self.owner_with(move || Some(owner.clone()))
    }

    /// `owner: :user`: evaluated when the MCP needs its credentials.
    pub fn owner_with(mut self, owner: impl Fn() -> Option<String> + Send + Sync + 'static) -> Self {
        self.owner = Some(Arc::new(owner));
        self
    }

    /// `scopes: %w[issues:read]`.
    pub fn scopes<S: AsRef<str>>(mut self, scopes: &[S]) -> Self {
        self.scopes = Some(scopes.iter().map(|s| s.as_ref().to_string()).collect());
        self
    }

    /// `client_id:`.
    pub fn client_id(mut self, client_id: impl Into<String>) -> Self {
        self.client_id = Some(client_id.into());
        self
    }

    /// `client_secret:`.
    pub fn client_secret(mut self, client_secret: impl Into<String>) -> Self {
        self.client_secret = Some(client_secret.into());
        self
    }
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn error(message: impl Into<String>) -> Error {
    McpError::new(message).into()
}

fn now() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0)
}

/// Ruby's `value.to_s` for a JSON value: strings bare, `nil` empty.
fn text(value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(other) => other.to_string(),
    }
}

fn truthy(value: Option<&Value>) -> Option<&Value> {
    value.filter(|v| !matches!(v, Value::Null | Value::Bool(false)))
}

/// `Array(value)` for a JSON value of strings.
fn strings(value: Option<&Value>) -> Vec<String> {
    match value {
        Some(Value::Array(items)) => items.iter().map(|i| text(Some(i))).collect(),
        Some(Value::Null) | None => Vec::new(),
        Some(other) => vec![text(Some(other))],
    }
}

/// `SecureRandom.urlsafe_base64(bytes)`.
fn random_token(bytes: usize) -> String {
    let mut buffer = vec![0u8; bytes];
    rand::thread_rng().fill_bytes(&mut buffer);
    URL_SAFE_NO_PAD.encode(buffer)
}

/// `OpenSSL.fixed_length_secure_compare` after the length check.
fn secure_compare(a: &str, b: &str) -> bool {
    a.len() == b.len() && a.bytes().zip(b.bytes()).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// `URI.encode_www_form`: spaces as `+`, everything but `*-._` and alphanumerics escaped.
fn encode_form(pairs: &[(&str, String)]) -> String {
    let Ok(mut url) = Url::parse("http://form.invalid/") else { return String::new() };
    url.query_pairs_mut().extend_pairs(pairs);
    url.query().unwrap_or("").to_string()
}

/// `path.chomp('/')`.
fn chomp(path: &str) -> &str {
    path.strip_suffix('/').unwrap_or(path)
}

fn same_origin(url: &Url, server: &Url) -> bool {
    (url.scheme(), url.host_str(), url.port_or_known_default()) == (server.scheme(), server.host_str(), server.port_or_known_default())
}

fn has_userinfo(url: &Url) -> bool {
    !url.username().is_empty() || url.password().is_some()
}

/// `covers?(resource, server)`: the resource is the server's origin and a prefix of its path.
fn covers(resource: &Url, server: &Url) -> bool {
    if has_userinfo(resource) || has_userinfo(server) || !same_origin(resource, server) {
        return false;
    }
    let path = chomp(resource.path());
    server.path() == path || server.path().starts_with(&format!("{path}/"))
}

/// `RubyLLM::MCP::OAuth`: one server's OAuth for one owner.
pub struct OAuth {
    server_url: String,
    owner: Option<String>,
    scopes: Option<Vec<String>>,
    client_id: Option<String>,
    client_secret: Option<String>,
    config: Arc<Config>,
    http: reqwest::Client,
    credential: Mutex<Option<Map<String, Value>>>,
    challenge: Mutex<Option<Challenge>>,
    resource: Mutex<Option<String>>,
    scopes_supported: Mutex<Option<Value>>,
}

impl OAuth {
    /// `OAuth.challenge(header)`: the parameters of a Bearer `WWW-Authenticate` challenge.
    pub fn challenge(header: Option<&str>) -> Challenge {
        static BEARER: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)\A\s*Bearer\s+").expect("constant pattern"));
        let rest = BEARER.replace(header.unwrap_or(""), "");
        let mut pairs: Vec<&str> = rest.split(',').collect();
        while pairs.last() == Some(&"") {
            pairs.pop();
        }
        pairs
            .into_iter()
            .map(|pair| {
                let mut parts = pair.splitn(2, '=');
                let name = parts.next().unwrap_or("").trim();
                let value = parts.next().unwrap_or("").trim();
                let value = value.strip_prefix('"').unwrap_or(value);
                (name.to_string(), value.strip_suffix('"').unwrap_or(value).to_string())
            })
            .collect()
    }

    /// `OAuth.new(server_url, owner:, scopes:, client_id:, client_secret:, config:)`. Requests
    /// use the configuration's `request_timeout` and never follow redirects, like
    /// `Transport::Connection.basic`.
    pub fn new(
        server_url: impl Into<String>,
        owner: Option<String>,
        scopes: Option<Vec<String>>,
        client_id: Option<String>,
        client_secret: Option<String>,
        config: Arc<Config>,
    ) -> Result<OAuth> {
        let http = reqwest::Client::builder()
            .timeout(config.request_timeout)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|e| Error::Configuration(e.to_string()))?;
        Ok(OAuth {
            server_url: server_url.into(),
            owner,
            scopes,
            client_id,
            client_secret,
            config,
            http,
            credential: Mutex::new(None),
            challenge: Mutex::new(None),
            resource: Mutex::new(None),
            scopes_supported: Mutex::new(None),
        })
    }

    /// `authorized?`.
    pub async fn is_authorized(&self) -> Result<bool> {
        Ok(self.credential().await?.is_some_and(|c| c.contains_key("access_token")))
    }

    /// `access_token`: refreshed first when it expires within a minute.
    pub async fn access_token(&self) -> Result<Option<String>> {
        let Some(credential) = self.credential().await?.filter(|c| c.contains_key("access_token")) else { return Ok(None) };
        if credential.get("expires_at").and_then(Value::as_i64).is_some_and(|at| at - REFRESH_EARLY < now()) {
            self.refresh().await?;
        }
        Ok(self.credential().await?.map(|c| text(c.get("access_token"))))
    }

    /// `refresh`: `false` when there is no refresh token or the authorization server refuses.
    pub async fn refresh(&self) -> Result<bool> {
        let Some(refresh_token) = self.credential().await?.and_then(|c| c.get("refresh_token").cloned()) else { return Ok(false) };
        match self.token_request("refresh_token", None, vec![("refresh_token", Some(text(Some(&refresh_token))))]).await {
            Ok(tokens) => {
                self.store_tokens(&tokens, None).await?;
                Ok(true)
            }
            Err(Error::Mcp(_)) => Ok(false),
            Err(e) => Err(e),
        }
    }

    /// `authorization_url(redirect_uri:, challenge:)`: discovers the authorization server,
    /// registers a client when needed, stores the pending authorization (state, PKCE verifier,
    /// issuer), and returns where to send the user.
    pub async fn authorization_url(&self, redirect_uri: &str, challenge: Option<Challenge>) -> Result<String> {
        *lock(&self.challenge) = challenge;
        let server = self.authorization_server().await?;
        let client = self.client_for(&server, redirect_uri).await?;
        let verifier = random_token(64);
        let state = random_token(32);
        let scope = self.scopes_for(&server).await?;
        let mut pending = client.clone();
        pending.insert("state".into(), state.clone().into());
        pending.insert("verifier".into(), verifier.clone().into());
        pending.insert("redirect_uri".into(), redirect_uri.into());
        pending.insert("issuer".into(), server.get("issuer").cloned().unwrap_or(Value::Null));
        pending.insert("scope".into(), scope.clone().map_or(Value::Null, Value::String));
        pending.insert("expires_at".into(), (now() + PENDING_FOR).into());
        let server_fields: Map<String, Value> =
            SERVER_FIELDS.iter().filter_map(|f| server.get(*f).map(|v| (f.to_string(), v.clone()))).collect();
        pending.insert("server".into(), Value::Object(server_fields));
        let mut data = self.credential().await?.unwrap_or_default();
        data.insert("pending".into(), Value::Object(pending));
        self.write(data).await?;

        let code_challenge = URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()));
        let query: Vec<(&str, String)> = [
            ("response_type", Some("code".to_string())),
            ("client_id", client.get("client_id").map(|v| text(Some(v)))),
            ("redirect_uri", Some(redirect_uri.to_string())),
            ("state", Some(state)),
            ("code_challenge", Some(code_challenge)),
            ("code_challenge_method", Some("S256".to_string())),
            ("resource", Some(self.resource())),
            ("scope", scope),
        ]
        .into_iter()
        .filter_map(|(k, v)| v.map(|v| (k, v)))
        .collect();
        let endpoint = text(server.get("authorization_endpoint"));
        self.endpoint(&endpoint)?;
        Ok(format!("{endpoint}?{}", encode_form(&query)))
    }

    /// `authorize(params)`: checks the callback against the pending authorization, exchanges
    /// the code, and stores the tokens.
    pub async fn authorize(&self, params: &HashMap<String, String>) -> Result<()> {
        let pending = self
            .credential()
            .await?
            .and_then(|c| c.get("pending").and_then(Value::as_object).cloned())
            .ok_or_else(|| error("No authorization in progress"))?;
        check_callback(&pending, params)?;
        let form = vec![
            ("code", params.get("code").cloned()),
            ("redirect_uri", pending.get("redirect_uri").map(|v| text(Some(v)))),
            ("code_verifier", pending.get("verifier").map(|v| text(Some(v)))),
        ];
        let tokens = self.token_request("authorization_code", Some(&pending), form).await?;
        let client: Map<String, Value> = ["client_id", "client_secret", "issuer", "server", "scope"]
            .iter()
            .filter_map(|k| pending.get(*k).map(|v| (k.to_string(), v.clone())))
            .collect();
        self.store_tokens(&tokens, Some(client)).await
    }

    /// `deauthorize`: forgets the owner's credentials for this server.
    pub async fn deauthorize(&self) -> Result<()> {
        *lock(&self.credential) = None;
        self.store().delete(&self.key()).await
    }

    fn store(&self) -> Arc<dyn CredentialStore> {
        self.config.mcp_credential_store.clone().unwrap_or_else(|| MEMORY_STORE.clone())
    }

    /// `"#{owner}@#{server_url}"`.
    fn key(&self) -> String {
        format!("{}@{}", self.owner.as_deref().unwrap_or(""), self.server_url)
    }

    /// `@credential ||= store.read(key)`.
    async fn credential(&self) -> Result<Option<Map<String, Value>>> {
        if let Some(cached) = lock(&self.credential).clone() {
            return Ok(Some(cached));
        }
        let read = match self.store().read(&self.key()).await? {
            Some(Value::Object(map)) => Some(map),
            _ => None,
        };
        if let Some(map) = &read {
            *lock(&self.credential) = Some(map.clone());
        }
        Ok(read)
    }

    async fn write(&self, data: Map<String, Value>) -> Result<()> {
        *lock(&self.credential) = Some(data.clone());
        self.store().write(&self.key(), Value::Object(data), self.owner.as_deref()).await
    }

    async fn store_tokens(&self, tokens: &Value, client: Option<Map<String, Value>>) -> Result<()> {
        let mut data = self.credential().await?.unwrap_or_default();
        data.remove("pending");
        data.extend(client.unwrap_or_default());
        data.insert("access_token".into(), tokens.get("access_token").cloned().unwrap_or(Value::Null));
        let scope = truthy(tokens.get("scope")).or(data.get("scope")).cloned().unwrap_or(Value::Null);
        data.insert("scope".into(), scope);
        let expires_in = truthy(tokens.get("expires_in"))
            .map(|v| v.as_i64().or_else(|| v.as_f64().map(|f| f as i64)).unwrap_or_else(|| text(Some(v)).trim().parse().unwrap_or(0)));
        data.insert("expires_at".into(), expires_in.map_or(Value::Null, |e| (now() + e).into()));
        if let Some(refresh) = truthy(tokens.get("refresh_token")) {
            data.insert("refresh_token".into(), refresh.clone());
        }
        data.retain(|_, v| !v.is_null());
        self.write(data).await
    }

    async fn token_request(&self, grant_type: &str, pending: Option<&Map<String, Value>>, params: Vec<(&str, Option<String>)>) -> Result<Value> {
        let client = match pending {
            Some(pending) => pending.clone(),
            None => self.credential().await?.unwrap_or_default(),
        };
        let server = truthy(client.get("server")).cloned().ok_or_else(|| error("No authorization server known; authorize first"))?;
        let mut form = params;
        form.push(("grant_type", Some(grant_type.to_string())));
        form.push(("client_id", truthy(client.get("client_id")).map(|v| text(Some(v)))));
        form.push(("resource", Some(self.resource())));
        let mut headers = vec![
            ("Content-Type", "application/x-www-form-urlencoded".to_string()),
            ("Accept", "application/json".to_string()),
        ];
        if let Some(secret) = truthy(client.get("client_secret")) {
            let methods = truthy(server.get("token_endpoint_auth_methods_supported"))
                .map(|m| strings(Some(m)))
                .unwrap_or_else(|| vec!["client_secret_basic".to_string()]);
            if methods.iter().any(|m| m == "client_secret_basic") {
                let credentials = STANDARD.encode(format!("{}:{}", text(client.get("client_id")), text(Some(secret))));
                headers.push(("Authorization", format!("Basic {credentials}")));
            } else {
                form.push(("client_secret", Some(text(Some(secret)))));
            }
        }
        let form: Vec<(&str, String)> = form.into_iter().filter_map(|(k, v)| v.map(|v| (k, v))).collect();
        self.post(&text(server.get("token_endpoint")), encode_form(&form), &headers).await
    }

    async fn authorization_server(&self) -> Result<Value> {
        let server = match self.protected_resource_metadata().await? {
            Some(metadata) => self.described_authorization_server(&metadata).await?,
            None => self.legacy_authorization_server().await,
        };
        if !strings(server.get("code_challenge_methods_supported")).iter().any(|m| m == "S256") {
            return Err(error(format!("{} does not support PKCE with S256", text(server.get("issuer")))));
        }
        Ok(server)
    }

    async fn described_authorization_server(&self, metadata: &Value) -> Result<Value> {
        let issuer = strings(metadata.get("authorization_servers"))
            .into_iter()
            .next()
            .ok_or_else(|| error(format!("{} names no authorization server", self.server_url)))?;
        let resource = self.checked_resource(truthy(metadata.get("resource")))?;
        *lock(&self.resource) = resource;
        *lock(&self.scopes_supported) = metadata.get("scopes_supported").cloned();
        self.discover_authorization_server(&issuer).await
    }

    fn server_uri(&self) -> Result<Url> {
        Url::parse(&self.server_url).map_err(|e| Error::Argument(format!("{}: {e}", self.server_url)))
    }

    async fn protected_resource_metadata(&self) -> Result<Option<Value>> {
        let server = self.server_uri()?;
        let challenged = lock(&self.challenge).as_ref().and_then(|c| c.get("resource_metadata").cloned());
        if let Some(url) = challenged
            && Url::parse(&url).is_ok_and(|u| same_origin(&u, &server))
        {
            return self.get_json(&url).await.map(Some);
        }
        let path = chomp(server.path()).to_string();
        let mut candidates = vec![format!("/.well-known/oauth-protected-resource{path}"), "/.well-known/oauth-protected-resource".to_string()];
        candidates.dedup();
        let urls: Vec<String> = candidates.iter().filter_map(|c| server.join(c).ok()).map(String::from).collect();
        Ok(self.first_json(&urls).await)
    }

    /// Servers from the 2025-03-26 revision publish no protected resource metadata: their own
    /// origin is the authorization server, with default endpoints when it publishes no metadata
    /// either.
    async fn legacy_authorization_server(&self) -> Value {
        let origin = self.server_uri().ok().and_then(|u| u.join("/").ok()).map(String::from).unwrap_or_default();
        let origin = chomp(&origin).to_string();
        if let Some(server) = self.first_json(&[format!("{origin}/.well-known/oauth-authorization-server")]).await {
            return server;
        }
        json!({
            "issuer": origin, "authorization_endpoint": format!("{origin}/authorize"),
            "token_endpoint": format!("{origin}/token"), "registration_endpoint": format!("{origin}/register"),
            "code_challenge_methods_supported": ["S256"]
        })
    }

    fn checked_resource(&self, resource: Option<&Value>) -> Result<Option<String>> {
        let Some(resource) = resource else { return Ok(None) };
        let resource = text(Some(resource));
        let server = self.server_uri()?;
        if Url::parse(&resource).is_ok_and(|r| covers(&r, &server)) {
            return Ok(Some(resource));
        }
        Err(error(format!("{} published metadata for another resource: {resource}", self.server_url)))
    }

    async fn discover_authorization_server(&self, issuer: &str) -> Result<Value> {
        let missing = || error(format!("{issuer} publishes no authorization server metadata"));
        let uri = Url::parse(issuer).map_err(|_| missing())?;
        let path = chomp(uri.path()).to_string();
        let mut paths = vec![
            format!("/.well-known/oauth-authorization-server{path}"),
            format!("/.well-known/openid-configuration{path}"),
            format!("{path}/.well-known/openid-configuration"),
        ];
        if path.is_empty() {
            paths.truncate(2);
        }
        let mut urls: Vec<String> = Vec::new();
        for url in paths.iter().filter_map(|p| uri.join(p).ok()).map(String::from) {
            if !urls.contains(&url) {
                urls.push(url);
            }
        }
        let server = self.first_json(&urls).await.ok_or_else(missing)?;
        if server.get("issuer").and_then(Value::as_str) != Some(issuer) {
            return Err(error(format!("{issuer} metadata names a different issuer")));
        }
        Ok(server)
    }

    async fn client_for(&self, server: &Value, redirect_uri: &str) -> Result<Map<String, Value>> {
        if let Some(client_id) = &self.client_id {
            let mut client = Map::new();
            client.insert("client_id".into(), client_id.clone().into());
            if let Some(secret) = &self.client_secret {
                client.insert("client_secret".into(), secret.clone().into());
            }
            return Ok(client);
        }
        if let Some(metadata_client_id) = &self.config.mcp_client_id
            && truthy(server.get("client_id_metadata_document_supported")).is_some()
        {
            let mut client = Map::new();
            client.insert("client_id".into(), metadata_client_id.clone().into());
            return Ok(client);
        }
        self.register(server, redirect_uri).await
    }

    async fn register(&self, server: &Value, redirect_uri: &str) -> Result<Map<String, Value>> {
        let issuer = text(server.get("issuer"));
        let endpoint = truthy(server.get("registration_endpoint")).ok_or_else(|| error(format!("{issuer} does not register clients")))?;
        let registration_key = format!("client:{issuer} {redirect_uri}");
        if let Some(Value::Object(registered)) = self.store().read(&registration_key).await? {
            return Ok(registered);
        }
        let body = json!({
            "client_name": self.config.mcp_client_name, "redirect_uris": [redirect_uri], "response_types": ["code"],
            "grant_types": ["authorization_code", "refresh_token"], "token_endpoint_auth_method": "none",
            "application_type": if Http::is_loopback(redirect_uri) { "native" } else { "web" }
        });
        let response = self.post(&text(Some(endpoint)), body.to_string(), &[("Content-Type", "application/json".into())]).await?;
        let client: Map<String, Value> =
            ["client_id", "client_secret"].iter().filter_map(|k| response.get(*k).map(|v| (k.to_string(), v.clone()))).collect();
        self.store().write(&registration_key, Value::Object(client.clone()), None).await?;
        Ok(client)
    }

    async fn scopes_for(&self, server: &Value) -> Result<Option<String>> {
        let challenged = lock(&self.challenge).as_ref().and_then(|c| c.get("scope").cloned());
        let mut scopes: Vec<String> = match challenged {
            Some(challenged) => {
                let granted = text(self.credential().await?.as_ref().and_then(|c| c.get("scope")));
                let mut scopes = self.scopes.clone().unwrap_or_default();
                scopes.extend(challenged.split_whitespace().map(str::to_string));
                scopes.extend(granted.split_whitespace().map(str::to_string));
                scopes
            }
            None => match &self.scopes {
                Some(scopes) => scopes.clone(),
                None => strings(lock(&self.scopes_supported).as_ref()),
            },
        };
        if strings(server.get("scopes_supported")).iter().any(|s| s == "offline_access") {
            scopes.push("offline_access".into());
        }
        let mut unique: Vec<String> = Vec::new();
        for scope in scopes {
            if !unique.contains(&scope) {
                unique.push(scope);
            }
        }
        Ok((!unique.is_empty()).then(|| unique.join(" ")))
    }

    /// `@resource || @server_url.chomp('/')`.
    fn resource(&self) -> String {
        lock(&self.resource).clone().unwrap_or_else(|| chomp(&self.server_url).to_string())
    }

    async fn first_json(&self, urls: &[String]) -> Option<Value> {
        for url in urls {
            if let Ok(json) = self.get_json(url).await {
                return Some(json);
            }
        }
        None
    }

    async fn get_json(&self, url: &str) -> Result<Value> {
        self.endpoint(url)?;
        let response = self
            .http
            .get(url)
            .header("Accept", "application/json")
            .send()
            .await
            .map_err(|e| Error::ConnectionFailed(e.to_string()))?;
        let status = response.status().as_u16();
        let body = response.text().await.map_err(|e| Error::ConnectionFailed(e.to_string()))?;
        if status >= 400 {
            let host = Url::parse(url).ok().and_then(|u| u.host_str().map(str::to_string)).unwrap_or_default();
            return Err(error(format!("{host} answered HTTP {status}")));
        }
        parse(&body)
    }

    async fn post(&self, url: &str, body: String, headers: &[(&str, String)]) -> Result<Value> {
        self.endpoint(url)?;
        let host = Url::parse(url).ok().and_then(|u| u.host_str().map(str::to_string)).unwrap_or_default();
        let refused = |body: &str| {
            let details: Value = serde_json::from_str(body).unwrap_or_else(|_| json!({}));
            let reason = truthy(details.get("error_description")).or(details.get("error"));
            error(format!("{host} refused the request: {}", text(reason)))
        };
        let mut request = self.http.post(url).body(body);
        for (name, value) in headers {
            request = request.header(*name, value);
        }
        let Ok(response) = request.send().await else { return Err(refused("")) };
        let status = response.status().as_u16();
        let Ok(body) = response.text().await else { return Err(refused("")) };
        if status >= 400 {
            return Err(refused(&body));
        }
        parse(&body)
    }

    /// `endpoint(url)`: OAuth endpoints must be HTTPS without credentials, or plain HTTP to a
    /// loopback address when the MCP server itself is on one.
    fn endpoint(&self, url: &str) -> Result<()> {
        let https = Url::parse(url).is_ok_and(|u| u.scheme() == "https" && !has_userinfo(&u));
        if https || (Http::is_loopback(&self.server_url) && Http::is_secure(url)) {
            return Ok(());
        }
        Err(error(format!("OAuth endpoints must use HTTPS: {url}")))
    }
}

fn parse(body: &str) -> Result<Value> {
    serde_json::from_str(body).map_err(|_| error("The authorization server did not answer with JSON"))
}

fn check_callback(pending: &Map<String, Value>, params: &HashMap<String, String>) -> Result<()> {
    if pending.get("expires_at").and_then(Value::as_i64).unwrap_or(0) < now() {
        return Err(error("The authorization expired; start again"));
    }
    match params.get("iss") {
        Some(issuer) if Some(issuer.as_str()) != pending.get("issuer").and_then(Value::as_str) => {
            return Err(error("The authorization response came from the wrong issuer"));
        }
        None if pending.get("server").and_then(|s| s.get("authorization_response_iss_parameter_supported")) == Some(&Value::Bool(true)) => {
            return Err(error("The authorization server did not identify itself"));
        }
        _ => {}
    }
    let expected = text(pending.get("state"));
    if !secure_compare(params.get("state").map_or("", String::as_str), &expected) {
        return Err(error("The authorization state does not match"));
    }
    if let Some(failure) = params.get("error") {
        let reason = params.get("error_description").unwrap_or(failure);
        return Err(error(format!("Authorization failed: {reason}")));
    }
    Ok(())
}

/// The MCP side of OAuth (`MCP#oauth`, `#request_headers`, `#unauthorized`, and `@challenge`),
/// shared between an [`Mcp`](super::Mcp) and its HTTP transport.
pub(crate) struct Authorizer {
    name: String,
    url: String,
    settings: OAuthSettings,
    config: Arc<Config>,
    oauth: Mutex<Option<Arc<OAuth>>>,
    challenge: Mutex<Option<Challenge>>,
}

impl Authorizer {
    pub(crate) fn new(name: String, url: String, settings: OAuthSettings, config: Arc<Config>) -> Authorizer {
        Authorizer { name, url, settings, config, oauth: Mutex::new(None), challenge: Mutex::new(None) }
    }

    /// `MCP#oauth`: raises when a declared owner resolves to nothing.
    pub(crate) fn oauth(&self) -> Result<Arc<OAuth>> {
        let owner = self.settings.owner.as_ref().map(|owner| owner());
        if matches!(owner, Some(None)) {
            return Err(Error::Argument(format!("{} needs an owner for OAuth credentials", self.name)));
        }
        let mut slot = lock(&self.oauth);
        if let Some(oauth) = slot.as_ref() {
            return Ok(oauth.clone());
        }
        let oauth = Arc::new(OAuth::new(
            self.url.clone(),
            owner.flatten(),
            self.settings.scopes.clone(),
            self.settings.client_id.clone(),
            self.settings.client_secret.clone(),
            self.config.clone(),
        )?);
        *slot = Some(oauth.clone());
        Ok(oauth)
    }

    /// `@challenge`: the last challenge the server sent.
    pub(crate) fn challenge(&self) -> Option<Challenge> {
        lock(&self.challenge).clone()
    }
}

#[async_trait]
impl Authorization for Authorizer {
    async fn authorization(&self) -> Result<Option<String>> {
        Ok(self.oauth()?.access_token().await?.map(|token| format!("Bearer {token}")))
    }

    /// `MCP#unauthorized`: keeps the challenge, and refreshes the token on a 401.
    async fn unauthorized(&self, header: Option<&str>, status: u16) -> bool {
        *lock(&self.challenge) = Some(OAuth::challenge(header));
        let Ok(oauth) = self.oauth() else { return false };
        status == 401 && oauth.is_authorized().await.unwrap_or(false) && oauth.refresh().await.unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // spec: mcp/oauth_spec.rb:182 reads WWW-Authenticate challenges
    #[test]
    fn reads_www_authenticate_challenges() {
        let challenge = OAuth::challenge(Some(r#"Bearer error="insufficient_scope", scope="a b", resource_metadata="https://x.test/m?a=1""#));
        let expected: Challenge = [("error", "insufficient_scope"), ("scope", "a b"), ("resource_metadata", "https://x.test/m?a=1")]
            .into_iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        assert_eq!(challenge, expected);
        assert!(OAuth::challenge(None).is_empty());
    }
}
