//! MCP OAuth, mirroring RubyLLM's `spec/ruby_llm/mcp/oauth_spec.rb` (plus `mcp_spec.rb`'s OAuth
//! settings). WebMock's `mcp.example.com` and `auth.example.com` become two wiremock servers on
//! loopback (distinct origins), which is why plain-HTTP OAuth endpoints are allowed here, exactly
//! as `OAuth#endpoint` allows them for a loopback MCP server. Each test gets its own
//! `MemoryStore` through its MCP's configuration, like the spec's `around` block.

use std::collections::HashMap;
use std::sync::Arc;

use base64::Engine;
use rust_llm::mcp::{CredentialStore, Mcp, MemoryStore, OAuthSettings};
use rust_llm::{Config, Error};
use serde_json::{Value, json};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

const REDIRECT_URI: &str = "https://app.example.com/mcp/callback";

/// The MCP server: answers any JSON-RPC request with `{ tools: [] }` for `access-1`/`access-2`,
/// and challenges everything else.
struct McpServer {
    challenge: String,
}

impl Respond for McpServer {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let authorization = request.headers.get("authorization").and_then(|v| v.to_str().ok());
        if matches!(authorization, Some("Bearer access-1" | "Bearer access-2")) {
            let id = serde_json::from_slice::<Value>(&request.body).ok().and_then(|b| b.get("id").cloned()).unwrap_or(Value::Null);
            return ResponseTemplate::new(200).set_body_json(json!({ "jsonrpc": "2.0", "id": id, "result": { "tools": [] } }));
        }
        ResponseTemplate::new(401).insert_header("WWW-Authenticate", self.challenge.as_str())
    }
}

/// The token endpoint: `access-2` for a refresh, `access-1` for a code.
struct TokenEndpoint;

impl Respond for TokenEndpoint {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let token = if form(request).get("grant_type").map(String::as_str) == Some("refresh_token") { "access-2" } else { "access-1" };
        ResponseTemplate::new(200).set_body_json(json!({ "access_token": token, "refresh_token": "refresh-1", "expires_in": 3600 }))
    }
}

fn form(request: &Request) -> HashMap<String, String> {
    reqwest::Url::parse(&format!("http://x/?{}", String::from_utf8_lossy(&request.body)))
        .map(|u| u.query_pairs().into_owned().collect())
        .unwrap_or_default()
}

fn query(url: &str) -> HashMap<String, String> {
    reqwest::Url::parse(url).unwrap().query_pairs().into_owned().collect()
}

struct World {
    mcp: MockServer,
    auth: MockServer,
    config: Arc<Config>,
    store: Arc<MemoryStore>,
}

impl World {
    fn server_url(&self) -> String {
        format!("{}/mcp", self.mcp.uri())
    }

    fn issuer(&self) -> String {
        self.auth.uri()
    }

    fn authorization_server(&self) -> Value {
        let issuer = self.issuer();
        json!({
            "issuer": issuer, "authorization_endpoint": format!("{issuer}/authorize"),
            "token_endpoint": format!("{issuer}/token"), "registration_endpoint": format!("{issuer}/register"),
            "code_challenge_methods_supported": ["S256"], "authorization_response_iss_parameter_supported": true
        })
    }

    fn challenge(&self) -> String {
        format!(r#"Bearer resource_metadata="{}/.well-known/oauth-protected-resource/mcp", scope="issues:read""#, self.mcp.uri())
    }

    /// `linear_class`: `url`, `inputs :user`, `oauth owner: :user`.
    fn linear(&self, user: &str) -> Mcp {
        self.mcp_with(OAuthSettings::new().owner(user))
    }

    fn mcp_with(&self, settings: OAuthSettings) -> Mcp {
        Mcp::url(self.server_url()).oauth(settings).config(self.config.clone()).build().unwrap()
    }

    /// `callback(url, **overrides)`: `None` drops a parameter, like `iss: nil`.
    fn callback(&self, url: &str, overrides: &[(&str, Option<&str>)]) -> HashMap<String, String> {
        let mut params: HashMap<String, Option<String>> = HashMap::from([
            ("code".into(), Some("code-1".into())),
            ("state".into(), query(url).get("state").cloned()),
            ("iss".into(), Some(self.issuer())),
        ]);
        for (key, value) in overrides {
            params.insert(key.to_string(), value.map(str::to_string));
        }
        params.into_iter().filter_map(|(k, v)| v.map(|v| (k, v))).collect()
    }

    /// `linear.authorize(callback(linear.authorization_url(redirect_uri:)))`.
    async fn authorize(&self, mcp: &Mcp) {
        let url = mcp.authorization_url(REDIRECT_URI).await.unwrap();
        mcp.authorize(self.callback(&url, &[])).await.unwrap();
    }

    async fn requests(&self, server: &MockServer, method: &str, path: &str) -> Vec<Request> {
        server
            .received_requests()
            .await
            .unwrap_or_default()
            .into_iter()
            .filter(|r| r.method.as_str() == method && r.url.path() == path)
            .collect()
    }

    /// Overrides a stub for one test, like a later `stub_request` in the example.
    async fn stub(&self, server: &MockServer, verb: &str, at: &str, response: ResponseTemplate) {
        Mock::given(method(verb)).and(path(at)).respond_with(response).with_priority(1).mount(server).await;
    }
}

/// The spec's `before` block.
async fn world() -> World {
    let (mcp, auth) = (MockServer::start().await, MockServer::start().await);
    let store = Arc::new(MemoryStore::new());
    let mut config = Config::default();
    config.mcp_credential_store = Some(store.clone());
    let world = World { mcp, auth, config: Arc::new(config), store };
    let server_url = world.server_url();
    Mock::given(method("POST")).and(path("/mcp")).respond_with(McpServer { challenge: world.challenge() }).mount(&world.mcp).await;
    Mock::given(method("GET"))
        .and(path("/.well-known/oauth-protected-resource/mcp"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "resource": server_url, "authorization_servers": [world.issuer()] })))
        .mount(&world.mcp)
        .await;
    Mock::given(method("GET"))
        .and(path("/.well-known/oauth-authorization-server"))
        .respond_with(ResponseTemplate::new(200).set_body_json(world.authorization_server()))
        .mount(&world.auth)
        .await;
    Mock::given(method("POST"))
        .and(path("/register"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "client_id": "registered" })))
        .mount(&world.auth)
        .await;
    Mock::given(method("POST")).and(path("/token")).respond_with(TokenEndpoint).mount(&world.auth).await;
    world
}

fn mcp_error(error: Error) -> String {
    match error {
        Error::Mcp(e) => e.message,
        other => panic!("expected MCP::Error, got {other:?}"),
    }
}

// spec: mcp/oauth_spec.rb:63 starts an authorization with PKCE, the resource, and the challenged scope
#[tokio::test]
async fn starts_an_authorization_with_pkce_the_resource_and_the_challenged_scope() {
    let w = world().await;
    let url = w.linear("ada").authorization_url(REDIRECT_URI).await.unwrap();
    let params = query(&url);

    assert!(url.starts_with(&format!("{}/authorize?", w.issuer())), "{url}");
    for (key, value) in [
        ("client_id", "registered"),
        ("code_challenge_method", "S256"),
        ("resource", w.server_url().as_str()),
        ("scope", "issues:read"),
        ("redirect_uri", REDIRECT_URI),
    ] {
        assert_eq!(params.get(key).map(String::as_str), Some(value), "{key}");
    }
    let registrations = w.requests(&w.auth, "POST", "/register").await;
    assert_eq!(registrations.len(), 1);
    let body: Value = serde_json::from_slice(&registrations[0].body).unwrap();
    assert_eq!((&body["application_type"], &body["token_endpoint_auth_method"]), (&json!("web"), &json!("none")));
}

// spec: mcp/oauth_spec.rb:76 exchanges the code and uses the token
#[tokio::test]
async fn exchanges_the_code_and_uses_the_token() {
    let w = world().await;
    let linear = w.linear("ada");
    w.authorize(&linear).await;

    assert!(linear.is_authorized().await.unwrap());
    assert!(linear.tools().await.unwrap().is_empty());
    let exchanges = w.requests(&w.auth, "POST", "/token").await;
    let exchange = exchanges.iter().map(form).find(|f| f.get("grant_type").map(String::as_str) == Some("authorization_code")).unwrap();
    assert_eq!(exchange.get("resource"), Some(&w.server_url()));
    assert!(exchange.get("code_verifier").is_some_and(|v| !v.is_empty()));
}

// spec: mcp/oauth_spec.rb:88 sends the server and OAuth requests through the connection of its context
// Ruby proves it with a recording Faraday adapter; reqwest has no adapter hook, so this checks
// the same requests are made and that they read the context's configuration (its client name and
// credential store) rather than the global one.
#[tokio::test]
async fn sends_the_server_and_oauth_requests_with_the_configuration_of_its_context() {
    let w = world().await;
    let mut config = (*w.config).clone();
    config.mcp_client_name = "Context client".into();
    let store = Arc::new(MemoryStore::new());
    config.mcp_credential_store = Some(store.clone());
    let linear = Mcp::url(w.server_url()).oauth(OAuthSettings::new().owner("ada")).config(Arc::new(config)).build().unwrap();

    w.authorize(&linear).await;
    linear.tools().await.unwrap();

    assert_eq!(w.requests(&w.auth, "GET", "/.well-known/oauth-authorization-server").await.len(), 1);
    let registration = w.requests(&w.auth, "POST", "/register").await;
    assert_eq!(serde_json::from_slice::<Value>(&registration[0].body).unwrap()["client_name"], "Context client");
    assert!(!w.requests(&w.auth, "POST", "/token").await.is_empty());
    assert!(!w.requests(&w.mcp, "POST", "/mcp").await.is_empty());
    assert!(store.read(&format!("ada@{}", w.server_url())).await.unwrap().is_some());
    assert!(w.store.read(&format!("ada@{}", w.server_url())).await.unwrap().is_none());
}

// spec: mcp/oauth_spec.rb:106 keeps credentials per owner
#[tokio::test]
async fn keeps_credentials_per_owner() {
    let w = world().await;
    w.authorize(&w.linear("ada")).await;

    assert!(!w.linear("grace").is_authorized().await.unwrap());
}

// spec: mcp/oauth_spec.rb:112 refreshes an expired token when the server rejects it
#[tokio::test]
async fn refreshes_an_expired_token_when_the_server_rejects_it() {
    let w = world().await;
    w.authorize(&w.linear("ada")).await;
    let key = format!("ada@{}", w.server_url());
    let mut stale = w.store.read(&key).await.unwrap().unwrap();
    stale["access_token"] = json!("stale");
    w.store.write(&key, stale, None).await.unwrap();

    assert!(w.linear("ada").tools().await.unwrap().is_empty());
    assert_eq!(w.store.read(&key).await.unwrap().unwrap()["access_token"], "access-2");
}

// spec: mcp/oauth_spec.rb:122 refuses a callback with the wrong state
#[tokio::test]
async fn refuses_a_callback_with_the_wrong_state() {
    let w = world().await;
    let linear = w.linear("ada");
    let url = linear.authorization_url(REDIRECT_URI).await.unwrap();

    let error = linear.authorize(w.callback(&url, &[("state", Some("forged"))])).await.err().unwrap();
    assert_eq!(mcp_error(error), "The authorization state does not match");
}

// spec: mcp/oauth_spec.rb:129 refuses a callback from another issuer, or none when one is required
#[tokio::test]
async fn refuses_a_callback_from_another_issuer_or_none_when_one_is_required() {
    let w = world().await;
    let linear = w.linear("ada");
    let url = linear.authorization_url(REDIRECT_URI).await.unwrap();

    let error = linear.authorize(w.callback(&url, &[("iss", Some("https://evil.example.com"))])).await.err().unwrap();
    assert!(mcp_error(error).contains("wrong issuer"));
    let error = linear.authorize(w.callback(&url, &[("iss", None)])).await.err().unwrap();
    assert!(mcp_error(error).contains("did not identify"));
}

// spec: mcp/oauth_spec.rb:137 uses a pre-registered client with its secret
#[tokio::test]
async fn uses_a_pre_registered_client_with_its_secret() {
    let w = world().await;
    let slack = w.mcp_with(OAuthSettings::new().client_id("slack-app").client_secret("shh"));

    let url = slack.authorization_url(REDIRECT_URI).await.unwrap();
    slack.authorize(w.callback(&url, &[])).await.unwrap();

    assert!(w.requests(&w.auth, "POST", "/register").await.is_empty());
    let basic = format!("Basic {}", base64::engine::general_purpose::STANDARD.encode("slack-app:shh"));
    let tokens = w.requests(&w.auth, "POST", "/token").await;
    assert!(tokens.iter().any(|r| r.headers.get("authorization").and_then(|v| v.to_str().ok()) == Some(basic.as_str())));
}

// spec: mcp/oauth_spec.rb:151 refuses authorization servers without PKCE
#[tokio::test]
async fn refuses_authorization_servers_without_pkce() {
    let w = world().await;
    let mut server = w.authorization_server();
    server.as_object_mut().unwrap().remove("code_challenge_methods_supported");
    w.stub(&w.auth, "GET", "/.well-known/oauth-authorization-server", ResponseTemplate::new(200).set_body_json(server)).await;

    let error = w.linear("ada").authorization_url(REDIRECT_URI).await.err().unwrap();
    assert!(mcp_error(error).contains("PKCE"));
}

// spec: mcp/oauth_spec.rb:158 refuses metadata for another resource
#[tokio::test]
async fn refuses_metadata_for_another_resource() {
    let w = world().await;
    let metadata = json!({ "resource": "https://other.example.com/mcp", "authorization_servers": [w.issuer()] });
    w.stub(&w.mcp, "GET", "/.well-known/oauth-protected-resource/mcp", ResponseTemplate::new(200).set_body_json(metadata)).await;

    let error = w.linear("ada").authorization_url(REDIRECT_URI).await.err().unwrap();
    assert!(mcp_error(error).contains("another resource"));
}

/// `mcp` at `impostor` (`oauth` with no owner), whose server challenges with metadata on its
/// own origin claiming `resource` is the real server.
async fn impostor_error(w: &World, impostor_server: &MockServer, impostor: &str) -> String {
    let base = impostor.trim_end_matches("/mcp");
    let metadata = format!("{base}/.well-known/oauth-protected-resource/mcp");
    let challenge = format!(r#"Bearer resource_metadata="{metadata}""#);
    w.stub(impostor_server, "POST", "/mcp", ResponseTemplate::new(401).insert_header("WWW-Authenticate", challenge.as_str())).await;
    let claims = json!({ "resource": w.server_url(), "authorization_servers": [w.issuer()] });
    w.stub(impostor_server, "GET", "/.well-known/oauth-protected-resource/mcp", ResponseTemplate::new(200).set_body_json(claims)).await;
    let mcp = Mcp::url(impostor).oauth(OAuthSettings::new()).config(w.config.clone()).build().unwrap();
    mcp_error(mcp.authorization_url(REDIRECT_URI).await.err().unwrap())
}

// spec: mcp/oauth_spec.rb:167 refuses a server at #{impostor} claiming another server's resource
// Both impostors: another host on the same port (`localhost` against `127.0.0.1`, standing in for
// `mcp.example.com.attacker.io`), and the same host on another port (`:8443`).
#[tokio::test]
async fn refuses_a_server_at_another_host_or_port_claiming_another_servers_resource() {
    let w = world().await;
    let port = w.mcp.address().port();
    assert!(impostor_error(&w, &w.mcp, &format!("http://localhost:{port}/mcp")).await.contains("another resource"));

    let other_port = MockServer::start().await;
    let impostor = format!("{}/mcp", other_port.uri());
    assert!(impostor_error(&w, &other_port, &impostor).await.contains("another resource"));
}

// spec: mcp/oauth_spec.rb:187 refuses an authorization endpoint that is not HTTPS
#[tokio::test]
async fn refuses_an_authorization_endpoint_that_is_not_https() {
    let w = world().await;
    let mut server = w.authorization_server();
    server["authorization_endpoint"] = json!("javascript:alert(1)");
    w.stub(&w.auth, "GET", "/.well-known/oauth-authorization-server", ResponseTemplate::new(200).set_body_json(server)).await;

    let error = w.linear("ada").authorization_url(REDIRECT_URI).await.err().unwrap();
    assert!(mcp_error(error).contains("HTTPS"));
}

// spec: mcp/oauth_spec.rb:194 needs the declared owner
#[tokio::test]
async fn needs_the_declared_owner() {
    let w = world().await;
    let nobody = w.mcp_with(OAuthSettings::new().owner_with(|| None));

    let error = nobody.is_authorized().await.err().unwrap();
    assert!(matches!(&error, Error::Argument(m) if m.contains("needs an owner")), "{error:?}");
}

// spec: mcp/oauth_spec.rb:198 keeps refreshing with the token endpoint that issued the token
#[tokio::test]
async fn keeps_refreshing_with_the_token_endpoint_that_issued_the_token() {
    let w = world().await;
    let linear = w.linear("ada");
    w.authorize(&linear).await;
    let evil = MockServer::start().await;
    let mut server = w.authorization_server();
    server["token_endpoint"] = json!(format!("{}/token", evil.uri()));
    w.stub(&w.auth, "GET", "/.well-known/oauth-authorization-server", ResponseTemplate::new(200).set_body_json(server)).await;
    linear.authorization_url(REDIRECT_URI).await.unwrap();

    assert!(w.linear("ada").oauth().unwrap().refresh().await.unwrap());

    assert!(evil.received_requests().await.unwrap_or_default().is_empty());
}

// spec: mcp/oauth_spec.rb:209 ignores metadata URLs on other hosts
#[tokio::test]
async fn ignores_metadata_urls_on_other_hosts() {
    let w = world().await;
    let internal = MockServer::start().await;
    let challenge = format!(r#"Bearer resource_metadata="{}/metadata""#, internal.uri());
    w.stub(&w.mcp, "POST", "/mcp", ResponseTemplate::new(401).insert_header("WWW-Authenticate", challenge.as_str())).await;

    w.linear("ada").authorization_url(REDIRECT_URI).await.unwrap();

    assert!(internal.received_requests().await.unwrap_or_default().is_empty());
    assert_eq!(w.requests(&w.mcp, "GET", "/.well-known/oauth-protected-resource/mcp").await.len(), 1);
}

fn scopes(url: &str) -> Vec<String> {
    let mut scopes: Vec<String> = query(url).get("scope").map(|s| s.split(' ').map(str::to_string).collect()).unwrap_or_default();
    scopes.sort();
    scopes
}

async fn insufficient_scope(w: &World) {
    let challenge = r#"Bearer error="insufficient_scope", scope="issues:write""#;
    w.stub(&w.mcp, "POST", "/mcp", ResponseTemplate::new(403).insert_header("WWW-Authenticate", challenge)).await;
}

// spec: mcp/oauth_spec.rb:220 asks for challenged scopes along with the ones already granted
#[tokio::test]
async fn asks_for_challenged_scopes_along_with_the_ones_already_granted() {
    let w = world().await;
    w.authorize(&w.linear("ada")).await;
    insufficient_scope(&w).await;
    let step_up = w.linear("ada");

    assert!(matches!(step_up.tools().await, Err(Error::Forbidden(..))));
    assert_eq!(scopes(&step_up.authorization_url(REDIRECT_URI).await.unwrap()), ["issues:read", "issues:write"]);
}

// spec: mcp/oauth_spec.rb:232 uses the server origin for servers without protected resource metadata
#[tokio::test]
async fn uses_the_server_origin_for_servers_without_protected_resource_metadata() {
    let w = world().await;
    for at in ["/.well-known/oauth-protected-resource/mcp", "/.well-known/oauth-protected-resource", "/.well-known/oauth-authorization-server"] {
        w.stub(&w.mcp, "GET", at, ResponseTemplate::new(404)).await;
    }
    w.stub(&w.mcp, "POST", "/register", ResponseTemplate::new(200).set_body_json(json!({ "client_id": "legacy" }))).await;
    w.stub(&w.mcp, "POST", "/mcp", ResponseTemplate::new(401)).await;

    let url = w.linear("ada").authorization_url(REDIRECT_URI).await.unwrap();

    assert!(url.starts_with(&format!("{}/authorize?", w.mcp.uri())), "{url}");
    assert_eq!(query(&url).get("client_id").map(String::as_str), Some("legacy"));
}

// spec: mcp/oauth_spec.rb:245 adds challenged scopes to configured ones
#[tokio::test]
async fn adds_challenged_scopes_to_configured_ones() {
    let w = world().await;
    let scoped = w.mcp_with(OAuthSettings::new().owner("ada").scopes(&["issues:read"]));
    w.authorize(&scoped).await;
    insufficient_scope(&w).await;

    assert!(matches!(scoped.tools().await, Err(Error::Forbidden(..))));
    assert_eq!(scopes(&scoped.authorization_url(REDIRECT_URI).await.unwrap()), ["issues:read", "issues:write"]);
}

// spec: mcp/oauth_spec.rb:262 refuses a legacy authorization server without PKCE
#[tokio::test]
async fn refuses_a_legacy_authorization_server_without_pkce() {
    let w = world().await;
    for at in ["/.well-known/oauth-protected-resource/mcp", "/.well-known/oauth-protected-resource"] {
        w.stub(&w.mcp, "GET", at, ResponseTemplate::new(404)).await;
    }
    let origin = w.mcp.uri();
    let legacy = json!({ "issuer": origin, "authorization_endpoint": format!("{origin}/authorize"), "token_endpoint": format!("{origin}/token") });
    w.stub(&w.mcp, "GET", "/.well-known/oauth-authorization-server", ResponseTemplate::new(200).set_body_json(legacy)).await;
    w.stub(&w.mcp, "POST", "/mcp", ResponseTemplate::new(401)).await;

    let error = w.linear("ada").authorization_url(REDIRECT_URI).await.err().unwrap();
    assert!(mcp_error(error).contains("PKCE"));
}

// spec: mcp/oauth_spec.rb:273 refreshes once when the server keeps rejecting the token
#[tokio::test]
async fn refreshes_once_when_the_server_keeps_rejecting_the_token() {
    let w = world().await;
    w.authorize(&w.linear("ada")).await;
    w.stub(&w.mcp, "POST", "/mcp", ResponseTemplate::new(401)).await;

    assert!(matches!(w.linear("ada").tools().await, Err(Error::Unauthorized(..))));
    let refreshes = w.requests(&w.auth, "POST", "/token").await;
    let refreshes = refreshes.iter().filter(|r| String::from_utf8_lossy(&r.body).contains("refresh_token")).count();
    assert_eq!(refreshes, 1);
}

// spec: mcp/oauth_spec.rb:282 forgets credentials
#[tokio::test]
async fn forgets_credentials() {
    let w = world().await;
    w.authorize(&w.linear("ada")).await;

    let linear = w.linear("ada");
    assert!(!linear.deauthorize().await.unwrap().is_authorized().await.unwrap());
}

// spec: mcp_spec.rb:518 accepts a prefix and OAuth settings
#[tokio::test]
async fn accepts_a_prefix_and_oauth_settings() {
    let linear = Mcp::url("https://mcp.linear.app/mcp")
        .prefix("mcp_1")
        .oauth(OAuthSettings::new().owner("owner-1").scopes(&["read"]))
        .build()
        .unwrap();

    let settings = linear.oauth_settings().unwrap();
    assert_eq!(settings.scopes.as_deref(), Some(&["read".to_string()][..]));
    assert_eq!(settings.owner.as_ref().and_then(|owner| owner()).as_deref(), Some("owner-1"));
    assert_eq!(linear.prefix(), Some("mcp_1"));
}
