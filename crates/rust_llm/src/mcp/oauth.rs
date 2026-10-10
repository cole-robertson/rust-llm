//! Port of `lib/ruby_llm/mcp/oauth.rb` and `lib/ruby_llm/mcp/oauth/memory_store.rb`: OAuth for
//! one MCP server and one owner, following the 2026-07-28 authorization spec: protected resource
//! metadata, authorization server discovery, client ID metadata documents or dynamic registration
//! when no client is configured, PKCE, resource indicators, issuer checks, and token refresh.
//! Credentials live in the configured `mcp_credential_store`, keyed by owner and server.
//!
//! Two grants need no user to authorize: the client credentials grant of the OAuth Client
//! Credentials extension, and the JWT bearer grant (RFC 7523 section 2.1). The latter presents a
//! token the workload's platform issued (Workload Identity Federation) or an identity assertion
//! grant the user's identity provider issued for their ID token (Enterprise-Managed
//! Authorization). A token is requested once the server rejects a request without one, and again
//! before it expires. Pre-registered clients authenticate with their secret or with a
//! `private_key_jwt` assertion (RFC 7523 section 2.2) addressed to the authorization server's
//! issuer.
//!
//! A server that requires DPoP-bound tokens (RFC 9449) gets them: new tokens are bound to a key
//! kept with them in the credentials, and every token request and server request carries a fresh
//! proof with the nonce the server last supplied.
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

mod key;
mod proofs;

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
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

use self::key::Key;
use self::proofs::Proofs;
use super::http::Authorization;
use super::{Http, McpError};
use crate::config::Config;
use crate::error::{Error, ErrorResponse, Result};

const PENDING_FOR: i64 = 600;
const REFRESH_EARLY: i64 = 60;
const ASSERTION_FOR: i64 = 60;
const JWT_BEARER: &str = "urn:ietf:params:oauth:grant-type:jwt-bearer";
const TOKEN_EXCHANGE: &str = "urn:ietf:params:oauth:grant-type:token-exchange";
const ID_JAG: &str = "urn:ietf:params:oauth:token-type:id-jag";
const ID_TOKEN: &str = "urn:ietf:params:oauth:token-type:id_token";
const CLIENT_ASSERTION: &str = "urn:ietf:params:oauth:client-assertion-type:jwt-bearer";
const CLIENT_CREDENTIALS: &str = "io.modelcontextprotocol/oauth-client-credentials";
const ENTERPRISE_MANAGED: &str = "io.modelcontextprotocol/enterprise-managed-authorization";
const SERVER_FIELDS: &[&str] = &[
    "issuer",
    "token_endpoint",
    "token_endpoint_auth_methods_supported",
    "token_endpoint_auth_signing_alg_values_supported",
    "authorization_response_iss_parameter_supported",
];
const CLIENT_FIELDS: &[&str] = &[
    "client_id",
    "client_secret",
    "issuer",
    "server",
    "redirect_uri",
    "dpop_key",
];
const AUTHORIZATION_SERVER_PATHS: &[&str] = &[
    "/.well-known/oauth-authorization-server{path}",
    "/.well-known/openid-configuration{path}",
    "{path}/.well-known/openid-configuration",
];

/// The challenges of a `WWW-Authenticate` header (`OAuth.challenge`): each scheme, lowercase,
/// with its parameters, keyed by lowercase name.
pub type Challenge = HashMap<String, HashMap<String, String>>;

/// The block a [`CredentialStore::synchronize`] runs.
pub type Synchronized<'a> = Pin<Box<dyn Future<Output = Result<bool>> + Send + 'a>>;

/// Names whose credentials these are, evaluated per MCP like `oauth owner: :user`.
pub type OwnerSource = Arc<dyn Fn() -> Option<String> + Send + Sync>;

/// A value read for every token, like a block or method name given to `assertion:` or
/// `identity_provider:` (`resolve(value)`).
pub type ValueSource = Arc<dyn Fn() -> Option<String> + Send + Sync>;

/// `config.mcp_credential_store`: where MCP OAuth credentials live. A store responds to
/// `read(key)`, `write(key, data, owner:)`, and `delete(key)`. `owner` is the owner's key (a
/// GlobalID string such as `gid://app/Chat/1` for records), or `None` for client registrations.
///
/// A store shared between processes also implements `synchronize(key)`, running the block while
/// no other process refreshes the credentials stored under `key`; the default runs it as is.
#[async_trait]
pub trait CredentialStore: Send + Sync {
    async fn read(&self, key: &str) -> Result<Option<Value>>;
    async fn write(&self, key: &str, data: Value, owner: Option<&str>) -> Result<()>;
    async fn delete(&self, key: &str) -> Result<()>;
    /// `synchronize(key) { ... }`.
    async fn synchronize<'a>(&'a self, _key: &'a str, block: Synchronized<'a>) -> Result<bool> {
        block.await
    }
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

/// `@refreshes`: one lock per credentials key, so one task in this process refreshes them at a
/// time.
static REFRESHES: LazyLock<Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// `grant:`, one of `OAuth::GRANTS`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Grant {
    AuthorizationCode,
    ClientCredentials,
    JwtBearer,
}

impl std::str::FromStr for Grant {
    type Err = Error;

    /// `oauth grant:`: raises `Error::Argument` for grants it does not know.
    fn from_str(grant: &str) -> Result<Grant> {
        match grant {
            "authorization_code" => Ok(Grant::AuthorizationCode),
            "client_credentials" => Ok(Grant::ClientCredentials),
            "jwt_bearer" => Ok(Grant::JwtBearer),
            other => Err(Error::Argument(format!("Unknown OAuth grant: {other}"))),
        }
    }
}

/// `identity_provider: { issuer:, client_id:, client_secret:, id_token: }`: the user's identity
/// provider for enterprise-managed authorization. Values are read for every token.
#[derive(Clone, Default)]
pub struct IdentityProvider {
    pub issuer: Option<ValueSource>,
    pub client_id: Option<ValueSource>,
    pub client_secret: Option<ValueSource>,
    pub id_token: Option<ValueSource>,
}

fn source(value: impl Into<String>) -> ValueSource {
    let value = value.into();
    Arc::new(move || Some(value.clone()))
}

impl IdentityProvider {
    pub fn new() -> IdentityProvider {
        IdentityProvider::default()
    }

    /// `issuer:`.
    pub fn issuer(mut self, issuer: impl Into<String>) -> Self {
        self.issuer = Some(source(issuer));
        self
    }

    /// `client_id:`.
    pub fn client_id(mut self, client_id: impl Into<String>) -> Self {
        self.client_id = Some(source(client_id));
        self
    }

    /// `client_secret:`.
    pub fn client_secret(mut self, client_secret: impl Into<String>) -> Self {
        self.client_secret = Some(source(client_secret));
        self
    }

    /// `id_token: -> { user.id_token }`: read for every token, since ID tokens expire.
    pub fn id_token_with(
        mut self,
        id_token: impl Fn() -> Option<String> + Send + Sync + 'static,
    ) -> Self {
        self.id_token = Some(Arc::new(id_token));
        self
    }
}

/// The settings of `oauth owner:, scopes:, client_id:, client_secret:, grant:, private_key:,
/// assertion:, identity_provider:`.
#[derive(Clone, Default)]
pub struct OAuthSettings {
    /// `owner:`: whose credentials these are. Declared but resolving to `None` is an error.
    pub owner: Option<OwnerSource>,
    /// `scopes:`: overrides the scopes the server asks for.
    pub scopes: Option<Vec<String>>,
    /// `client_id:`/`client_secret:`: an app you registered, which servers such as Slack require.
    pub client_id: Option<String>,
    pub client_secret: Option<String>,
    /// `grant:`: `None` is `authorization_code`, or `jwt_bearer` with an assertion or identity
    /// provider.
    pub grant: Option<Grant>,
    /// `private_key:`: a PEM key that signs a short-lived assertion in place of the secret.
    pub private_key: Option<String>,
    /// `assertion:`: a JWT the workload's platform issued, read for every token.
    pub assertion: Option<ValueSource>,
    /// `identity_provider:`: enterprise-managed authorization.
    pub identity_provider: Option<IdentityProvider>,
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
    pub fn owner_with(
        mut self,
        owner: impl Fn() -> Option<String> + Send + Sync + 'static,
    ) -> Self {
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

    /// `grant: :client_credentials`: connects your app as itself, with no user.
    pub fn grant(mut self, grant: Grant) -> Self {
        self.grant = Some(grant);
        self
    }

    /// `private_key: ENV["REPORTS_PRIVATE_KEY"]`: a PEM string.
    pub fn private_key(mut self, pem: impl Into<String>) -> Self {
        self.private_key = Some(pem.into());
        self
    }

    /// `assertion: "..."`.
    pub fn assertion(mut self, assertion: impl Into<String>) -> Self {
        self.assertion = Some(source(assertion));
        self
    }

    /// `assertion: -> { File.read("/var/run/secrets/tokens/mcp-token") }`: read for every
    /// token, as platforms rotate them.
    pub fn assertion_with(
        mut self,
        assertion: impl Fn() -> Option<String> + Send + Sync + 'static,
    ) -> Self {
        self.assertion = Some(Arc::new(assertion));
        self
    }

    /// `identity_provider: { ... }`.
    pub fn identity_provider(mut self, provider: IdentityProvider) -> Self {
        self.identity_provider = Some(provider);
        self
    }

    /// `OAuth.extensions(settings)`: the authorization extensions these settings use, as client
    /// capabilities declare them.
    pub fn extensions(&self) -> Map<String, Value> {
        let mut names = Map::new();
        if self.grant == Some(Grant::ClientCredentials) {
            names.insert(CLIENT_CREDENTIALS.into(), json!({}));
        }
        if self.identity_provider.is_some() {
            names.insert(ENTERPRISE_MANAGED.into(), json!({}));
        }
        names
    }
}

pub(super) fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn error(message: impl Into<String>) -> Error {
    McpError::new(message).into()
}

pub(super) fn now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
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

fn string(value: Option<&Value>) -> Option<String> {
    truthy(value).map(|v| text(Some(v)))
}

/// `Array(value)` for a JSON value of strings.
fn strings(value: Option<&Value>) -> Vec<String> {
    match value {
        Some(Value::Array(items)) => items.iter().map(|i| text(Some(i))).collect(),
        Some(Value::Null) | None => Vec::new(),
        Some(other) => vec![text(Some(other))],
    }
}

fn slice(map: &Map<String, Value>, fields: &[&str]) -> Map<String, Value> {
    fields
        .iter()
        .filter_map(|f| map.get(*f).map(|v| (f.to_string(), v.clone())))
        .collect()
}

/// `SecureRandom.urlsafe_base64(bytes)`.
fn random_token(bytes: usize) -> String {
    let mut buffer = vec![0u8; bytes];
    rand::thread_rng().fill_bytes(&mut buffer);
    URL_SAFE_NO_PAD.encode(buffer)
}

/// `OpenSSL.fixed_length_secure_compare` after the length check.
fn secure_compare(a: &str, b: &str) -> bool {
    a.len() == b.len()
        && a.bytes()
            .zip(b.bytes())
            .fold(0u8, |acc, (x, y)| acc | (x ^ y))
            == 0
}

/// `URI.encode_www_form`: spaces as `+`, everything but `*-._` and alphanumerics escaped.
fn encode_form(pairs: &[(&str, String)]) -> String {
    let Ok(mut url) = Url::parse("http://form.invalid/") else {
        return String::new();
    };
    url.query_pairs_mut().extend_pairs(pairs);
    url.query().unwrap_or("").to_string()
}

/// `URI.encode_www_form_component`.
fn encode_component(value: &str) -> String {
    let encoded = encode_form(&[("", value.to_string())]);
    encoded.strip_prefix('=').unwrap_or(&encoded).to_string()
}

/// `path.chomp('/')`.
fn chomp(path: &str) -> &str {
    path.strip_suffix('/').unwrap_or(path)
}

fn same_origin(url: &Url, server: &Url) -> bool {
    (url.scheme(), url.host_str(), url.port_or_known_default())
        == (
            server.scheme(),
            server.host_str(),
            server.port_or_known_default(),
        )
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

/// `same_issuer?`: RFC 3986 section 6.2.3: an empty path and "/" name the same resource.
fn same_issuer(one: &str, other: &str) -> bool {
    static TRAILING_SLASH: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"(?i)\A([a-z][a-z0-9+.-]*://[^/?#]+)/([?#].*)?\z").expect("constant pattern")
    });
    let normalized = |issuer: &str| TRAILING_SLASH.replace(issuer, "$1$2").into_owned();
    normalized(one) == normalized(other)
}

/// The parts of `Error#data` and `#response` an OAuth refusal carries.
fn oauth_error(error: &Error) -> Option<String> {
    match error {
        Error::Mcp(e) => e
            .data
            .as_ref()
            .and_then(|d| d.get("error"))
            .and_then(Value::as_str)
            .map(str::to_string),
        _ => None,
    }
}

/// `UnauthorizedError.new(e.message, response: e.response)`.
fn unauthorized(error: Error) -> Error {
    match error {
        Error::Mcp(e) => Error::Unauthorized(e.message, e.response),
        other => other,
    }
}

/// `OAuth.challenge(header)`: reads the challenges of a `WWW-Authenticate` header (RFC 9110
/// section 11.6.1) into their parameters, keyed by lowercase scheme.
fn parse_challenge(header: &str) -> Challenge {
    fn is_token(c: char) -> bool {
        c.is_ascii_alphanumeric() || "!#$%&'*+.^_`|~-".contains(c)
    }
    fn skip_separators(s: &str, i: &mut usize) {
        while let Some(c) = s[*i..].chars().next() {
            if c.is_whitespace() || c == ',' {
                *i += c.len_utf8();
            } else {
                break;
            }
        }
    }
    fn token(s: &str, i: usize) -> usize {
        i + s[i..].find(|c: char| !is_token(c)).unwrap_or(s.len() - i)
    }
    fn skip_spaces(s: &str, i: usize) -> usize {
        i + s[i..]
            .find(|c: char| !c.is_whitespace())
            .unwrap_or(s.len() - i)
    }
    let s = header;
    let mut challenges = Challenge::new();
    let mut i = 0;
    loop {
        skip_separators(s, &mut i);
        let end = token(s, i);
        if end == i {
            break;
        }
        let scheme = s[i..end].to_ascii_lowercase();
        i = end;
        let parameters = challenges.entry(scheme).or_default();
        loop {
            let mut j = i;
            skip_separators(s, &mut j);
            let name_end = token(s, j);
            let after = skip_spaces(s, name_end);
            if name_end == j || !s[after..].starts_with('=') {
                i = j;
                break;
            }
            let name = s[j..name_end].to_ascii_lowercase();
            let mut k = skip_spaces(s, after + 1);
            let value = if s[k..].starts_with('"') {
                let mut value = String::new();
                let mut chars = s[k + 1..].char_indices();
                let mut closed = None;
                while let Some((offset, c)) = chars.next() {
                    match c {
                        '\\' => {
                            if let Some((_, escaped)) = chars.next() {
                                value.push(escaped);
                            }
                        }
                        '"' => {
                            closed = Some(k + 1 + offset + 1);
                            break;
                        }
                        c => value.push(c),
                    }
                }
                match closed {
                    Some(end) => {
                        k = end;
                        value
                    }
                    None => {
                        let end = k + s[k..]
                            .find(|c: char| c.is_whitespace() || c == ',')
                            .unwrap_or(s.len() - k);
                        let raw = s[k..end].to_string();
                        k = end;
                        raw
                    }
                }
            } else {
                let end = k + s[k..]
                    .find(|c: char| c.is_whitespace() || c == ',')
                    .unwrap_or(s.len() - k);
                let raw = s[k..end].to_string();
                k = end;
                raw
            };
            parameters.entry(name).or_insert(value);
            i = k;
        }
    }
    challenges
}

/// How to send a rejected request again (`OAuth#recover`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Recovery {
    /// With the nonce the server sent, in a new proof.
    Nonce,
    /// With a new token.
    Token,
}

/// `RubyLLM::MCP::OAuth`: one server's OAuth for one owner.
pub struct OAuth {
    server_url: String,
    owner: Option<String>,
    scopes: Option<Vec<String>>,
    client_id: Option<String>,
    client_secret: Option<String>,
    private_key: Option<String>,
    assertion: Option<ValueSource>,
    identity_provider: Option<IdentityProvider>,
    grant: Grant,
    config: Arc<Config>,
    http: reqwest::Client,
    credential: Mutex<Option<Map<String, Value>>>,
    challenge: Mutex<Option<Challenge>>,
    resource: Mutex<Option<String>>,
    scopes_supported: Mutex<Option<Value>>,
    dpop_required: Mutex<bool>,
    signing_key: Mutex<Option<Arc<Key>>>,
    proofs: Proofs,
}

/// One token request: its grant parameters, the client, and the key that signs its assertion.
struct TokenRequest<'a> {
    grant_type: &'a str,
    client: Map<String, Value>,
    key: Option<Arc<Key>>,
    params: Vec<(&'a str, Option<String>)>,
}

impl OAuth {
    /// `OAuth.challenge(header)`: the challenges of a `WWW-Authenticate` header.
    pub fn challenge(header: Option<&str>) -> Challenge {
        parse_challenge(header.unwrap_or(""))
    }

    /// `OAuth.new(server_url, owner:, scopes:, client_id:, client_secret:, config:)`, with the
    /// rest of the settings left unset. Requests use the configuration's `request_timeout` and
    /// never follow redirects, like `Transport::Connection.basic`.
    pub fn new(
        server_url: impl Into<String>,
        owner: Option<String>,
        scopes: Option<Vec<String>>,
        client_id: Option<String>,
        client_secret: Option<String>,
        config: Arc<Config>,
    ) -> Result<OAuth> {
        let settings = OAuthSettings {
            scopes,
            client_id,
            client_secret,
            ..OAuthSettings::default()
        };
        OAuth::with_settings(server_url, owner, &settings, config)
    }

    /// `OAuth.new(server_url, owner:, **settings, config:)`.
    pub fn with_settings(
        server_url: impl Into<String>,
        owner: Option<String>,
        settings: &OAuthSettings,
        config: Arc<Config>,
    ) -> Result<OAuth> {
        let http = reqwest::Client::builder()
            .timeout(config.request_timeout)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|e| Error::Configuration(e.to_string()))?;
        let grant = settings.grant.unwrap_or(
            if settings.assertion.is_some() || settings.identity_provider.is_some() {
                Grant::JwtBearer
            } else {
                Grant::AuthorizationCode
            },
        );
        Ok(OAuth {
            server_url: server_url.into(),
            owner,
            scopes: settings.scopes.clone(),
            client_id: settings.client_id.clone(),
            client_secret: settings.client_secret.clone(),
            private_key: settings.private_key.clone(),
            assertion: settings.assertion.clone(),
            identity_provider: settings.identity_provider.clone(),
            grant,
            config,
            http,
            credential: Mutex::new(None),
            challenge: Mutex::new(None),
            resource: Mutex::new(None),
            scopes_supported: Mutex::new(None),
            dpop_required: Mutex::new(false),
            signing_key: Mutex::new(None),
            proofs: Proofs::default(),
        })
    }

    /// `authorized?`.
    pub async fn is_authorized(&self) -> Result<bool> {
        Ok(self
            .credential()
            .await?
            .is_some_and(|c| c.contains_key("access_token")))
    }

    /// `access_token`: renewed first when it expires within a minute.
    pub async fn access_token(&self) -> Result<Option<String>> {
        if !self.is_authorized().await? {
            return Ok(None);
        }
        if self.is_expiring() {
            self.renew().await?;
        }
        Ok(self
            .credential()
            .await?
            .map(|c| text(c.get("access_token"))))
    }

    /// `authorization_headers(verb)`: the headers that authorize a `verb` request to the server:
    /// none without a token, and a proof of possession with a bound one.
    pub async fn authorization_headers(&self, verb: &str) -> Result<Vec<(String, String)>> {
        let Some(token) = self.access_token().await? else {
            return Ok(Vec::new());
        };
        let credential = self.credential().await?.unwrap_or_default();
        if credential.get("token_type").and_then(Value::as_str) != Some("DPoP") {
            return Ok(vec![("Authorization".into(), format!("Bearer {token}"))]);
        }
        let proof = self.proofs.sign(
            &text(credential.get("dpop_key")),
            &self.server_url,
            Some(&token),
            verb,
        )?;
        Ok(vec![
            ("Authorization".into(), format!("DPoP {token}")),
            ("DPoP".into(), proof),
        ])
    }

    /// `recover(challenge, nonce:, recovered:)`: answers the server's rejection of a request,
    /// given its challenge, its DPoP nonce, and the ways the request recovered before. Returns
    /// how to send it again, or `None` when it is not worth sending again.
    pub async fn recover(
        &self,
        challenge: Challenge,
        nonce: Option<&str>,
        recovered: &[Recovery],
    ) -> Result<Option<Recovery>> {
        *lock(&self.challenge) = Some(challenge);
        self.remember_nonce(nonce);
        let recovery = if nonce.is_some() && self.is_nonce_requested().await? {
            Recovery::Nonce
        } else {
            Recovery::Token
        };
        if recovered.contains(&recovery) {
            return Ok(None);
        }
        if recovery == Recovery::Nonce || self.is_renewed().await? {
            return Ok(Some(recovery));
        }
        Ok(None)
    }

    /// `remember_nonce(nonce)`: keeps the DPoP nonce the server supplied with a response for the
    /// next proof (RFC 9449 section 9).
    pub fn remember_nonce(&self, nonce: Option<&str>) {
        self.proofs.remember(&self.server_url, nonce);
    }

    /// `refresh`: `false` when there is no refresh token or the authorization server refuses;
    /// `true` without asking when another worker replaced the token since it was used.
    pub async fn refresh(&self) -> Result<bool> {
        let Some(used) = self
            .credential()
            .await?
            .filter(|c| c.contains_key("refresh_token"))
            .map(|c| c.get("access_token").cloned())
        else {
            return Ok(false);
        };
        let refreshed = self
            .synchronize(Box::pin(async move {
                let credential = self.reread().await?;
                if let Some(credential) = &credential
                    && credential.get("access_token").cloned() != used
                {
                    return Ok(true);
                }
                let Some(refresh_token) = credential.and_then(|c| c.get("refresh_token").cloned())
                else {
                    return Ok(false);
                };
                let client = self.credential().await?.unwrap_or_default();
                let request = TokenRequest {
                    grant_type: "refresh_token",
                    key: self.signing_key_for(&client)?,
                    client,
                    params: vec![("refresh_token", Some(text(Some(&refresh_token))))],
                };
                let tokens = self.token_request(request).await?;
                self.store_tokens(&tokens, None).await?;
                Ok(true)
            }))
            .await;
        match refreshed {
            Err(Error::Mcp(_)) => Ok(false),
            other => other,
        }
    }

    /// `authorization_url(redirect_uri:, challenge:)`: discovers the authorization server,
    /// registers a client when needed, stores the pending authorization (state, PKCE verifier,
    /// issuer), and returns where to send the user.
    pub async fn authorization_url(
        &self,
        redirect_uri: &str,
        challenge: Option<Challenge>,
    ) -> Result<String> {
        if self.grant != Grant::AuthorizationCode {
            return Err(Error::Configuration(format!(
                "The {} grant needs no authorization",
                self.grant_name()
            )));
        }
        *lock(&self.challenge) = challenge;
        let server = self.authorization_server().await?;
        let mut client = self.client_for(&server, redirect_uri).await?;
        let verifier = random_token(64);
        let state = random_token(32);
        let scope = self.scopes_for(&server).await?;
        client.insert("state".into(), state.clone().into());
        client.insert("verifier".into(), verifier.clone().into());
        client.insert("redirect_uri".into(), redirect_uri.into());
        client.insert(
            "issuer".into(),
            server.get("issuer").cloned().unwrap_or(Value::Null),
        );
        client.insert(
            "scope".into(),
            scope.clone().map_or(Value::Null, Value::String),
        );
        client.insert(
            "dpop_key".into(),
            self.binding_key()?.map_or(Value::Null, Value::String),
        );
        client.insert("expires_at".into(), (now() + PENDING_FOR).into());
        let server_fields = server
            .as_object()
            .map(|s| slice(s, SERVER_FIELDS))
            .unwrap_or_default();
        client.insert("server".into(), Value::Object(server_fields));
        let mut data = self.credential().await?.unwrap_or_default();
        data.insert("pending".into(), Value::Object(client.clone()));
        self.write(data).await?;

        let code_challenge = URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()));
        let query: Vec<(&str, String)> = [
            ("response_type", Some("code".to_string())),
            ("client_id", string(client.get("client_id"))),
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
        let request = TokenRequest {
            grant_type: "authorization_code",
            key: self.signing_key_for(&pending)?,
            params: vec![
                ("code", params.get("code").cloned()),
                ("redirect_uri", string(pending.get("redirect_uri"))),
                ("code_verifier", string(pending.get("verifier"))),
            ],
            client: pending.clone(),
        };
        let tokens = self.token_request(request).await?;
        let mut fields = CLIENT_FIELDS.to_vec();
        fields.push("scope");
        self.store_tokens(&tokens, Some(slice(&pending, &fields)))
            .await
    }

    /// `deauthorize`: forgets the owner's credentials for this server.
    pub async fn deauthorize(&self) -> Result<()> {
        *lock(&self.credential) = None;
        self.store().delete(&self.key()).await
    }

    fn grant_name(&self) -> &'static str {
        match self.grant {
            Grant::AuthorizationCode => "authorization_code",
            Grant::ClientCredentials => "client_credentials",
            Grant::JwtBearer => "jwt_bearer",
        }
    }

    fn store(&self) -> Arc<dyn CredentialStore> {
        self.config
            .mcp_credential_store
            .clone()
            .unwrap_or_else(|| MEMORY_STORE.clone())
    }

    /// `"#{owner}@#{server_url}"`.
    fn key(&self) -> String {
        format!(
            "{}@{}",
            self.owner.as_deref().unwrap_or(""),
            self.server_url
        )
    }

    /// `@credential ||= store.read(key)`.
    async fn credential(&self) -> Result<Option<Map<String, Value>>> {
        if let Some(cached) = lock(&self.credential).clone() {
            return Ok(Some(cached));
        }
        self.reread().await
    }

    /// `@credential = store.read(key)`.
    async fn reread(&self) -> Result<Option<Map<String, Value>>> {
        let read = match self.store().read(&self.key()).await? {
            Some(Value::Object(map)) => Some(map),
            _ => None,
        };
        *lock(&self.credential) = read.clone();
        Ok(read)
    }

    fn cached(&self) -> Map<String, Value> {
        lock(&self.credential).clone().unwrap_or_default()
    }

    async fn write(&self, data: Map<String, Value>) -> Result<()> {
        *lock(&self.credential) = Some(data.clone());
        self.store()
            .write(&self.key(), Value::Object(data), self.owner.as_deref())
            .await
    }

    /// `synchronize { ... }`: while no other task in this process, nor (through the store) any
    /// other process, refreshes these credentials.
    async fn synchronize<'a>(&'a self, block: Synchronized<'a>) -> Result<bool> {
        let key = self.key();
        let mutex = lock(&REFRESHES).entry(key.clone()).or_default().clone();
        let _guard = mutex.lock().await;
        let store = self.store();
        store.synchronize(&key, block).await
    }

    async fn store_tokens(&self, tokens: &Value, client: Option<Map<String, Value>>) -> Result<()> {
        let mut data = self.credential().await?.unwrap_or_default();
        data.remove("pending");
        if let Some(client) = client {
            data.retain(|k, _| !CLIENT_FIELDS.contains(&k.as_str()));
            data.extend(client);
        }
        data.insert(
            "access_token".into(),
            tokens.get("access_token").cloned().unwrap_or(Value::Null),
        );
        let scope = truthy(tokens.get("scope"))
            .or(data.get("scope"))
            .cloned()
            .unwrap_or(Value::Null);
        data.insert("scope".into(), scope);
        let expires_in = truthy(tokens.get("expires_in")).map(|v| {
            v.as_i64()
                .or_else(|| v.as_f64().map(|f| f as i64))
                .unwrap_or_else(|| text(Some(v)).trim().parse().unwrap_or(0))
        });
        data.insert(
            "expires_at".into(),
            expires_in.map_or(Value::Null, |e| (now() + e).into()),
        );
        let dpop = text(tokens.get("token_type")).eq_ignore_ascii_case("DPoP");
        data.insert(
            "token_type".into(),
            if dpop { "DPoP".into() } else { Value::Null },
        );
        if let Some(refresh) = truthy(tokens.get("refresh_token")) {
            data.insert("refresh_token".into(), refresh.clone());
        }
        data.retain(|_, v| !v.is_null());
        self.write(data).await
    }

    fn is_expiring(&self) -> bool {
        self.cached()
            .get("expires_at")
            .and_then(Value::as_i64)
            .is_some_and(|at| at - REFRESH_EARLY < now())
    }

    async fn renew(&self) -> Result<bool> {
        match self.grant {
            Grant::AuthorizationCode => self.refresh().await,
            _ => self.obtain().await,
        }
    }

    async fn is_renewed(&self) -> Result<bool> {
        match self.grant {
            Grant::AuthorizationCode => Ok(self.is_authorized().await? && self.refresh().await?),
            _ => self.obtain().await,
        }
    }

    /// `obtain`: requests a token with the configured grant, unless another worker has replaced
    /// the one the server rejected. Failures raise `Error::Unauthorized`.
    async fn obtain(&self) -> Result<bool> {
        let rejected = self
            .credential()
            .await?
            .and_then(|c| c.get("access_token").cloned());
        self.synchronize(Box::pin(async move {
            let credential = self.reread().await?;
            if let Some(credential) = &credential
                && credential.contains_key("access_token")
                && credential.get("access_token").cloned() != rejected
                && !self.is_expiring()
            {
                return Ok(true);
            }
            let server = self.authorization_server().await?;
            let client = self.granting_client(&server).await?;
            let scope = self.scopes_for(&server).await?;
            let (grant_type, mut params) = self.grant_parameters(&server, scope.clone()).await?;
            params.push(("scope", scope));
            let request = TokenRequest {
                grant_type,
                key: self.signing_key_for(&client)?,
                client: client.clone(),
                params,
            };
            let tokens = self.token_request(request).await?;
            self.store_tokens(&tokens, Some(client)).await?;
            Ok(true)
        }))
        .await
        .map_err(unauthorized)
    }

    async fn granting_client(&self, server: &Value) -> Result<Map<String, Value>> {
        if self.grant == Grant::ClientCredentials && self.client_id.is_none() {
            return Err(Error::Configuration(
                "The client_credentials grant needs a client_id".into(),
            ));
        }
        let mut client = self.preregistered_client(server).await?;
        client.insert(
            "issuer".into(),
            server.get("issuer").cloned().unwrap_or(Value::Null),
        );
        let server_fields = server
            .as_object()
            .map(|s| slice(s, SERVER_FIELDS))
            .unwrap_or_default();
        client.insert("server".into(), Value::Object(server_fields));
        client.insert(
            "dpop_key".into(),
            self.binding_key()?.map_or(Value::Null, Value::String),
        );
        client.retain(|_, v| !v.is_null());
        Ok(client)
    }

    /// `dpop_required?`: RFC 9449 section 7.1 and RFC 9728 section 2: a server requires
    /// DPoP-bound tokens when it challenges with the DPoP scheme and not Bearer, or when its
    /// metadata says so.
    fn is_dpop_required(&self) -> bool {
        let challenge = lock(&self.challenge).clone().unwrap_or_default();
        *lock(&self.dpop_required)
            || (challenge.contains_key("dpop") && !challenge.contains_key("bearer"))
    }

    fn binding_key(&self) -> Result<Option<String>> {
        if !self.is_dpop_required() {
            return Ok(None);
        }
        Ok(Some(Key::generate()?.to_pem().to_string()))
    }

    /// `nonce_requested?`: RFC 9449 section 9: a server that wants a nonce in proofs answers
    /// with `use_dpop_nonce` and the nonce to use.
    async fn is_nonce_requested(&self) -> Result<bool> {
        let asked = lock(&self.challenge)
            .as_ref()
            .and_then(|c| c.get("dpop"))
            .and_then(|p| p.get("error"))
            .is_some_and(|e| e == "use_dpop_nonce");
        Ok(asked
            && self
                .credential()
                .await?
                .is_some_and(|c| c.get("token_type").and_then(Value::as_str) == Some("DPoP")))
    }

    /// `grant_parameters`: assertions are resolved for every token, since workload platforms
    /// rotate the tokens they issue and ID tokens expire.
    async fn grant_parameters(
        &self,
        server: &Value,
        scope: Option<String>,
    ) -> Result<(&'static str, Vec<(&'static str, Option<String>)>)> {
        if self.grant == Grant::ClientCredentials {
            return Ok(("client_credentials", Vec::new()));
        }
        let assertion = match &self.identity_provider {
            Some(provider) => Some(self.identity_assertion(provider, server, scope).await?),
            None => self.assertion.as_ref().and_then(|a| a()),
        };
        let assertion = assertion.ok_or_else(|| {
            Error::Configuration(
                "The jwt_bearer grant needs an assertion or an identity provider".into(),
            )
        })?;
        Ok((JWT_BEARER, vec![("assertion", Some(assertion))]))
    }

    /// `identity_assertion`: exchanges the user's ID token at the identity provider for an
    /// Identity Assertion JWT Authorization Grant addressed to the server's authorization server
    /// (enterprise-managed authorization, section 4). Only that grant is forwarded: any other
    /// token the identity provider returns is the user's credential there.
    async fn identity_assertion(
        &self,
        provider: &IdentityProvider,
        server: &Value,
        scope: Option<String>,
    ) -> Result<String> {
        let read = |value: &Option<ValueSource>| value.as_ref().and_then(|v| v());
        let (Some(issuer), Some(id_token)) = (read(&provider.issuer), read(&provider.id_token))
        else {
            return Err(Error::Configuration(
                "The identity provider needs an issuer and an id_token".into(),
            ));
        };
        let mut client = Map::new();
        if let Some(id) = read(&provider.client_id) {
            client.insert("client_id".into(), id.into());
        }
        if let Some(secret) = read(&provider.client_secret) {
            client.insert("client_secret".into(), secret.into());
        }
        client.insert(
            "server".into(),
            self.discover_authorization_server(&issuer).await?,
        );
        let request = TokenRequest {
            grant_type: TOKEN_EXCHANGE,
            client,
            key: None,
            params: vec![
                ("requested_token_type", Some(ID_JAG.into())),
                ("audience", string(server.get("issuer"))),
                ("scope", scope),
                ("subject_token", Some(id_token)),
                ("subject_token_type", Some(ID_TOKEN.into())),
            ],
        };
        let grant = self.token_request(request).await?;
        if grant.get("issued_token_type").and_then(Value::as_str) == Some(ID_JAG) {
            return Ok(text(grant.get("access_token")));
        }
        Err(error(format!(
            "{issuer} did not issue an identity assertion grant"
        )))
    }

    /// `key: (signing_key if client['client_id'] == @client_id)`.
    fn signing_key_for(&self, client: &Map<String, Value>) -> Result<Option<Arc<Key>>> {
        if string(client.get("client_id")) != self.client_id {
            return Ok(None);
        }
        self.signing_key()
    }

    fn signing_key(&self) -> Result<Option<Arc<Key>>> {
        let Some(pem) = &self.private_key else {
            return Ok(None);
        };
        let mut slot = lock(&self.signing_key);
        if slot.is_none() {
            *slot = Some(Arc::new(Key::new(pem)?));
        }
        Ok(slot.clone())
    }

    /// `token_request(grant_type, client:, key:, **params)`: forgets the registration the
    /// authorization server no longer knows.
    async fn token_request(&self, request: TokenRequest<'_>) -> Result<Value> {
        let client = request.client;
        let key = request.key;
        let server = truthy(client.get("server"))
            .cloned()
            .ok_or_else(|| error("No authorization server known; authorize first"))?;
        let mut params = request.params;
        params.push(("grant_type", Some(request.grant_type.to_string())));
        let mut result = self
            .send_token_request(&server, &client, key.as_deref(), &params)
            .await;
        if let Err(e) = &result
            && oauth_error(e).as_deref() == Some("use_dpop_nonce")
            && truthy(client.get("dpop_key")).is_some()
        {
            result = self
                .send_token_request(&server, &client, key.as_deref(), &params)
                .await;
        }
        if let Err(e) = &result
            && oauth_error(e).as_deref() == Some("invalid_client")
        {
            self.forget_registration(&client).await?;
        }
        result
    }

    /// `send_token_request`: one attempt, which signs a new proof and a new client assertion.
    async fn send_token_request(
        &self,
        server: &Value,
        client: &Map<String, Value>,
        key: Option<&Key>,
        params: &[(&str, Option<String>)],
    ) -> Result<Value> {
        let url = text(server.get("token_endpoint"));
        let mut form: Vec<(&str, Option<String>)> = params.to_vec();
        form.push(("client_id", string(client.get("client_id"))));
        form.push(("resource", Some(self.resource())));
        let mut headers = vec![
            (
                "Content-Type",
                "application/x-www-form-urlencoded".to_string(),
            ),
            ("Accept", "application/json".to_string()),
        ];
        self.authenticate(client, server, &mut form, &mut headers, key)?;
        if let Some(pem) = string(client.get("dpop_key")) {
            headers.push(("DPoP", self.proofs.sign(&pem, &url, None, "POST")?));
        }
        let form: Vec<(&str, String)> = form
            .into_iter()
            .filter_map(|(k, v)| v.map(|v| (k, v)))
            .collect();
        self.post(&url, encode_form(&form), &headers).await
    }

    async fn forget_registration(&self, client: &Map<String, Value>) -> Result<()> {
        let registration = registration_key(
            &text(client.get("issuer")),
            &text(client.get("redirect_uri")),
        );
        let registered = self.store().read(&registration).await?;
        if registered
            .as_ref()
            .and_then(|r| r.get("client_id"))
            .is_some_and(|id| Some(id) == client.get("client_id"))
        {
            self.store().delete(&registration).await?;
        }
        Ok(())
    }

    fn authenticate(
        &self,
        client: &Map<String, Value>,
        server: &Value,
        form: &mut Vec<(&str, Option<String>)>,
        headers: &mut Vec<(&str, String)>,
        key: Option<&Key>,
    ) -> Result<()> {
        if let Some(key) = key {
            form.retain(|(k, _)| *k != "client_id");
            let assertion = client_assertion(key, &text(client.get("client_id")), server)?;
            form.push(("client_assertion_type", Some(CLIENT_ASSERTION.into())));
            form.push(("client_assertion", Some(assertion)));
        } else if let Some(secret) = string(client.get("client_secret")) {
            let methods = truthy(server.get("token_endpoint_auth_methods_supported"))
                .map(|m| strings(Some(m)))
                .unwrap_or_else(|| vec!["client_secret_basic".to_string()]);
            if methods.iter().any(|m| m == "client_secret_basic") {
                let credentials = basic_credentials(&text(client.get("client_id")), &secret);
                headers.push(("Authorization", format!("Basic {credentials}")));
            } else {
                form.push(("client_secret", Some(secret)));
            }
        }
        Ok(())
    }

    async fn authorization_server(&self) -> Result<Value> {
        let server = match self.protected_resource_metadata().await? {
            Some(metadata) => self.described_authorization_server(&metadata).await?,
            None => self.legacy_authorization_server().await,
        };
        if self.grant == Grant::AuthorizationCode
            && !strings(server.get("code_challenge_methods_supported"))
                .iter()
                .any(|m| m == "S256")
        {
            return Err(error(format!(
                "{} does not support PKCE with S256",
                text(server.get("issuer"))
            )));
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
        *lock(&self.dpop_required) =
            metadata.get("dpop_bound_access_tokens_required") == Some(&Value::Bool(true));
        self.discover_authorization_server(&issuer).await
    }

    fn server_uri(&self) -> Result<Url> {
        Url::parse(&self.server_url)
            .map_err(|e| Error::Argument(format!("{}: {e}", self.server_url)))
    }

    fn challenged(&self, name: &str) -> Option<String> {
        let challenge = lock(&self.challenge).clone().unwrap_or_default();
        let mut schemes: Vec<_> = challenge.into_iter().collect();
        schemes.sort_by_key(|(scheme, _)| scheme != "bearer");
        schemes
            .into_iter()
            .find_map(|(_, parameters)| parameters.get(name).cloned())
    }

    async fn protected_resource_metadata(&self) -> Result<Option<Value>> {
        let server = self.server_uri()?;
        if let Some(url) = self.challenged("resource_metadata")
            && Url::parse(&url).is_ok_and(|u| same_origin(&u, &server))
        {
            return self.get_json(&url).await.map(Some);
        }
        let path = chomp(server.path()).to_string();
        let mut candidates = vec![
            format!("/.well-known/oauth-protected-resource{path}"),
            "/.well-known/oauth-protected-resource".to_string(),
        ];
        candidates.dedup();
        let urls: Vec<String> = candidates
            .iter()
            .filter_map(|c| server.join(c).ok())
            .map(String::from)
            .collect();
        Ok(self.first_json(&urls).await)
    }

    /// Servers from the 2025-03-26 revision publish no protected resource metadata: their own
    /// origin is the authorization server, with default endpoints when it publishes no metadata
    /// either.
    async fn legacy_authorization_server(&self) -> Value {
        let origin = self
            .server_uri()
            .ok()
            .and_then(|u| u.join("/").ok())
            .map(String::from)
            .unwrap_or_default();
        let origin = chomp(&origin).to_string();
        if let Some(server) = self
            .first_json(&[format!("{origin}/.well-known/oauth-authorization-server")])
            .await
        {
            return server;
        }
        json!({
            "issuer": origin, "authorization_endpoint": format!("{origin}/authorize"),
            "token_endpoint": format!("{origin}/token"), "registration_endpoint": format!("{origin}/register"),
            "code_challenge_methods_supported": ["S256"]
        })
    }

    fn checked_resource(&self, resource: Option<&Value>) -> Result<Option<String>> {
        let Some(resource) = resource else {
            return Ok(None);
        };
        let resource = text(Some(resource));
        let server = self.server_uri()?;
        if Url::parse(&resource).is_ok_and(|r| covers(&r, &server)) {
            return Ok(Some(resource));
        }
        Err(error(format!(
            "{} published metadata for another resource: {resource}",
            self.server_url
        )))
    }

    /// `discover_authorization_server(issuer)`: follows metadata naming another issuer once, to
    /// the metadata that issuer publishes about itself.
    async fn discover_authorization_server(&self, issuer: &str) -> Result<Value> {
        let different = || error(format!("{issuer} metadata names a different issuer"));
        let server = self.authorization_server_metadata(issuer).await?;
        let named = server.get("issuer");
        if same_issuer(&text(named), issuer) {
            return Ok(server);
        }
        let named = named
            .and_then(Value::as_str)
            .filter(|n| Url::parse(n).is_ok_and(|u| u.host_str().is_some_and(|h| !h.is_empty())))
            .ok_or_else(different)?
            .to_string();
        let server = self.authorization_server_metadata(&named).await?;
        if !same_issuer(&text(server.get("issuer")), &named) {
            return Err(different());
        }
        Ok(server)
    }

    async fn authorization_server_metadata(&self, issuer: &str) -> Result<Value> {
        let missing = || {
            error(format!(
                "{issuer} publishes no authorization server metadata"
            ))
        };
        let uri = Url::parse(issuer).map_err(|_| missing())?;
        let path = chomp(uri.path()).to_string();
        let mut paths: Vec<String> = AUTHORIZATION_SERVER_PATHS
            .iter()
            .map(|pattern| pattern.replace("{path}", &path))
            .collect();
        if path.is_empty() {
            paths.truncate(2);
        }
        let mut urls: Vec<String> = Vec::new();
        for url in paths
            .iter()
            .filter_map(|p| uri.join(p).ok())
            .map(String::from)
        {
            if !urls.contains(&url) {
                urls.push(url);
            }
        }
        self.first_json(&urls).await.ok_or_else(missing)
    }

    async fn client_for(&self, server: &Value, redirect_uri: &str) -> Result<Map<String, Value>> {
        if self.client_id.is_some() {
            return self.preregistered_client(server).await;
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

    /// `preregistered_client(server)`: keeps a pre-registered client (or a workload) to the
    /// authorization server it was first used with.
    async fn preregistered_client(&self, server: &Value) -> Result<Map<String, Value>> {
        let issuer_key = format!(
            "issuer:{} {}",
            self.client_id.as_deref().unwrap_or(""),
            self.server_url
        );
        let current = text(server.get("issuer"));
        let issuer = self
            .store()
            .read(&issuer_key)
            .await?
            .and_then(|r| string(r.get("issuer")));
        match issuer {
            None => {
                self.store()
                    .write(
                        &issuer_key,
                        json!({ "issuer": server.get("issuer").cloned().unwrap_or(Value::Null) }),
                        None,
                    )
                    .await?
            }
            Some(issuer) if !same_issuer(&issuer, &current) => {
                return Err(error(format!(
                    "{} is registered with {issuer}, but {} now uses {current}",
                    self.client_id.as_deref().unwrap_or("The workload"),
                    self.server_url
                )));
            }
            Some(_) => {}
        }
        let mut client = Map::new();
        if let Some(id) = &self.client_id {
            client.insert("client_id".into(), id.clone().into());
        }
        if let Some(secret) = &self.client_secret {
            client.insert("client_secret".into(), secret.clone().into());
        }
        Ok(client)
    }

    async fn register(&self, server: &Value, redirect_uri: &str) -> Result<Map<String, Value>> {
        let issuer = text(server.get("issuer"));
        let endpoint = truthy(server.get("registration_endpoint"))
            .ok_or_else(|| error(format!("{issuer} does not register clients")))?;
        let registration = registration_key(&issuer, redirect_uri);
        let scope = self.scopes_for(server).await?;
        if let Some(Value::Object(registered)) = self.store().read(&registration).await?
            && registered_for(&text(registered.get("scope")), scope.as_deref())
        {
            return Ok(slice(&registered, &["client_id", "client_secret"]));
        }
        let mut body = json!({
            "client_name": self.config.mcp_client_name, "redirect_uris": [redirect_uri], "response_types": ["code"],
            "grant_types": ["authorization_code", "refresh_token"], "token_endpoint_auth_method": "none",
            "application_type": if Http::is_loopback(redirect_uri) { "native" } else { "web" }
        });
        if let Some(scope) = &scope {
            body["scope"] = scope.clone().into();
        }
        let response = self
            .post(
                &text(Some(endpoint)),
                body.to_string(),
                &[("Content-Type", "application/json".into())],
            )
            .await?;
        let client = response
            .as_object()
            .map(|r| slice(r, &["client_id", "client_secret"]))
            .unwrap_or_default();
        let mut stored = client.clone();
        if let Some(scope) = scope {
            stored.insert("scope".into(), scope.into());
        }
        stored.retain(|_, v| !v.is_null());
        self.store()
            .write(&registration, Value::Object(stored), None)
            .await?;
        Ok(client)
    }

    async fn scopes_for(&self, server: &Value) -> Result<Option<String>> {
        let mut scopes: Vec<String> = match self.challenged("scope") {
            Some(challenged) => {
                let granted = text(
                    self.credential()
                        .await?
                        .as_ref()
                        .and_then(|c| c.get("scope")),
                );
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
        if self.grant == Grant::AuthorizationCode
            && strings(server.get("scopes_supported"))
                .iter()
                .any(|s| s == "offline_access")
        {
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
        lock(&self.resource)
            .clone()
            .unwrap_or_else(|| chomp(&self.server_url).to_string())
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
        let body = response
            .text()
            .await
            .map_err(|e| Error::ConnectionFailed(e.to_string()))?;
        if status >= 400 {
            let host = Url::parse(url)
                .ok()
                .and_then(|u| u.host_str().map(str::to_string))
                .unwrap_or_default();
            return Err(error(format!("{host} answered HTTP {status}")));
        }
        parse(&body)
    }

    async fn post(&self, url: &str, body: String, headers: &[(&str, String)]) -> Result<Value> {
        self.endpoint(url)?;
        let mut request = self.http.post(url).body(body);
        for (name, value) in headers {
            request = request.header(*name, value);
        }
        let Ok(response) = request.send().await else {
            return Err(self.refusal(url, None, ""));
        };
        let status = response.status().as_u16();
        let nonce = response
            .headers()
            .get("dpop-nonce")
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        self.proofs.remember(url, nonce.as_deref());
        let Ok(body) = response.text().await else {
            return Err(self.refusal(url, Some(status), ""));
        };
        if status >= 400 {
            return Err(self.refusal(url, Some(status), &body));
        }
        parse(&body)
    }

    fn refusal(&self, url: &str, status: Option<u16>, body: &str) -> Error {
        let host = Url::parse(url)
            .ok()
            .and_then(|u| u.host_str().map(str::to_string))
            .unwrap_or_default();
        let details = serde_json::from_str::<Value>(body)
            .ok()
            .filter(Value::is_object)
            .unwrap_or_else(|| json!({}));
        let reason = truthy(details.get("error_description")).or(details.get("error"));
        McpError {
            message: format!("{host} refused the request: {}", text(reason)),
            data: Some(details),
            response: status.map(|status| ErrorResponse {
                status,
                body: body.to_string(),
                ..Default::default()
            }),
            ..Default::default()
        }
        .into()
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

/// `registered_for?`: a registration covers the scopes requested when it was made for all of
/// them.
fn registered_for(registered: &str, requested: Option<&str>) -> bool {
    let registered: Vec<&str> = registered.split_whitespace().collect();
    requested
        .unwrap_or("")
        .split_whitespace()
        .all(|scope| registered.contains(&scope))
}

fn registration_key(issuer: &str, redirect_uri: &str) -> String {
    format!("client:{issuer} {redirect_uri}")
}

/// `basic_credentials`: RFC 6749 section 2.3.1: the client ID and secret are form-encoded before
/// they become the user and password of Basic authentication.
fn basic_credentials(client_id: &str, client_secret: &str) -> String {
    STANDARD.encode(format!(
        "{}:{}",
        encode_component(client_id),
        encode_component(client_secret)
    ))
}

/// `client_assertion`: RFC 7523bis section 4: the issuer is the sole audience, so no other
/// authorization server can replay the assertion, and the explicit type tells servers the client
/// follows that rule.
fn client_assertion(key: &Key, client_id: &str, server: &Value) -> Result<String> {
    let now = now();
    let claims = json!({
        "iss": client_id, "sub": client_id, "aud": server.get("issuer").cloned().unwrap_or(Value::Null),
        "iat": now, "exp": now + ASSERTION_FOR, "jti": uuid::Uuid::new_v4().to_string()
    });
    let algorithm = key.algorithm(&strings(
        server.get("token_endpoint_auth_signing_alg_values_supported"),
    ));
    let mut header = Map::new();
    header.insert("typ".into(), "client-authentication+jwt".into());
    key.jwt(&claims, algorithm, header)
}

fn parse(body: &str) -> Result<Value> {
    serde_json::from_str(body)
        .map_err(|_| error("The authorization server did not answer with JSON"))
}

fn check_callback(pending: &Map<String, Value>, params: &HashMap<String, String>) -> Result<()> {
    if pending
        .get("expires_at")
        .and_then(Value::as_i64)
        .unwrap_or(0)
        < now()
    {
        return Err(error("The authorization expired; start again"));
    }
    match params.get("iss") {
        Some(issuer) if Some(issuer.as_str()) != pending.get("issuer").and_then(Value::as_str) => {
            return Err(error(
                "The authorization response came from the wrong issuer",
            ));
        }
        None if pending
            .get("server")
            .and_then(|s| s.get("authorization_response_iss_parameter_supported"))
            == Some(&Value::Bool(true)) =>
        {
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

/// The MCP side of OAuth (`MCP#oauth`, `#request_headers`, `#unauthorized`, `#responded`, and
/// `@challenge`), shared between an [`Mcp`](super::Mcp) and its HTTP transport.
pub(crate) struct Authorizer {
    name: String,
    url: String,
    settings: OAuthSettings,
    config: Arc<Config>,
    oauth: Mutex<Option<Arc<OAuth>>>,
    challenge: Mutex<Option<Challenge>>,
}

impl Authorizer {
    pub(crate) fn new(
        name: String,
        url: String,
        settings: OAuthSettings,
        config: Arc<Config>,
    ) -> Authorizer {
        Authorizer {
            name,
            url,
            settings,
            config,
            oauth: Mutex::new(None),
            challenge: Mutex::new(None),
        }
    }

    /// `MCP#oauth`: raises when a declared owner resolves to nothing.
    pub(crate) fn oauth(&self) -> Result<Arc<OAuth>> {
        let owner = self.settings.owner.as_ref().map(|owner| owner());
        if matches!(owner, Some(None)) {
            return Err(Error::Argument(format!(
                "{} needs an owner for OAuth credentials",
                self.name
            )));
        }
        let mut slot = lock(&self.oauth);
        if let Some(oauth) = slot.as_ref() {
            return Ok(oauth.clone());
        }
        let oauth = Arc::new(OAuth::with_settings(
            self.url.clone(),
            owner.flatten(),
            &self.settings,
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
        Ok(self
            .authorization_headers("POST")
            .await?
            .into_iter()
            .find(|(name, _)| name == "Authorization")
            .map(|(_, value)| value))
    }

    /// `oauth.authorization_headers(verb)`.
    async fn authorization_headers(&self, verb: &str) -> Result<Vec<(String, String)>> {
        self.oauth()?.authorization_headers(verb).await
    }

    /// `MCP#unauthorized` without a nonce: keeps the challenge, and renews the token on a 401.
    async fn unauthorized(&self, header: Option<&str>, status: u16) -> bool {
        matches!(self.recover(header, status, None, &[]).await, Ok(Some(_)))
    }

    /// `MCP#unauthorized(headers, status, recovered)`.
    async fn recover(
        &self,
        header: Option<&str>,
        status: u16,
        nonce: Option<&str>,
        recovered: &[Recovery],
    ) -> Result<Option<Recovery>> {
        let challenge = OAuth::challenge(header);
        *lock(&self.challenge) = Some(challenge.clone());
        if status != 401 {
            return Ok(None);
        }
        self.oauth()?.recover(challenge, nonce, recovered).await
    }

    /// `MCP#responded(headers)`.
    fn responded(&self, nonce: Option<&str>) {
        if let Ok(oauth) = self.oauth() {
            oauth.remember_nonce(nonce);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parameters(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    // spec: mcp/oauth_spec.rb:381 reads WWW-Authenticate challenges
    #[test]
    fn reads_www_authenticate_challenges() {
        let challenge = OAuth::challenge(Some(
            r#"Bearer error="insufficient_scope", scope="a b", resource_metadata="https://x.test/m?a=1""#,
        ));
        let expected: Challenge = HashMap::from([(
            "bearer".to_string(),
            parameters(&[
                ("error", "insufficient_scope"),
                ("scope", "a b"),
                ("resource_metadata", "https://x.test/m?a=1"),
            ]),
        )]);
        assert_eq!(challenge, expected);
        assert!(OAuth::challenge(None).is_empty());
    }

    // spec: mcp/oauth_spec.rb:386 reads every challenge of a WWW-Authenticate header
    #[test]
    fn reads_every_challenge_of_a_www_authenticate_header() {
        let header =
            r#"Bearer realm="a, \"b\"", DPoP algs="ES256 PS256", error=use_dpop_nonce, Basic"#;

        let expected: Challenge = HashMap::from([
            ("bearer".to_string(), parameters(&[("realm", r#"a, "b""#)])),
            (
                "dpop".to_string(),
                parameters(&[("algs", "ES256 PS256"), ("error", "use_dpop_nonce")]),
            ),
            ("basic".to_string(), HashMap::new()),
        ]);
        assert_eq!(OAuth::challenge(Some(header)), expected);
    }

    #[test]
    fn treats_an_issuers_trailing_slash_as_its_empty_path() {
        assert!(same_issuer("https://a.test/", "https://a.test"));
        assert!(same_issuer("https://a.test", "https://a.test/"));
        assert!(!same_issuer("https://a.test/x/", "https://a.test/x"));
        assert!(!same_issuer("https://b.test", "https://a.test"));
    }

    #[test]
    fn form_encodes_basic_credentials() {
        assert_eq!(
            basic_credentials("reports", "p:ss%word"),
            STANDARD.encode("reports:p%3Ass%25word")
        );
    }
}
