//! MCP OAuth, mirroring RubyLLM's `spec/ruby_llm/mcp/oauth_spec.rb` (plus `mcp_spec.rb`'s OAuth
//! settings). WebMock's `mcp.example.com` and `auth.example.com` become two wiremock servers on
//! loopback (distinct origins), which is why plain-HTTP OAuth endpoints are allowed here, exactly
//! as `OAuth#endpoint` allows them for a loopback MCP server. Each test gets its own
//! `MemoryStore` through its MCP's configuration, like the spec's `around` block.

use std::collections::HashMap;
use std::sync::Arc;

use base64::Engine;
use rust_llm::mcp::{CredentialStore, Grant, IdentityProvider, Mcp, MemoryStore, OAuthSettings};
use rust_llm::{Config, Error};
use serde_json::{Value, json};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

const REDIRECT_URI: &str = "https://app.example.com/mcp/callback";

/// A fixed RSA key for `private_key_jwt` (ring signs with RSA keys but does not generate them).
const RSA_KEY: &str = "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQCcQATE2MT22nCu\nAckwkyf7wybIAauCza7p08Xw2/71VI2lrzJVMjVi37hh5Kuuqf5u3pxRD5tVQ2ol\nQ+f3vjg3IgI79DHSClY9SEvUz9/oxbHrrLm+RVb0JCZ8uG3LORTqV4PBVsI7JHuZ\nPu4smuPfcy+6LQEOnuL60vjQlSSkIXgAeca6i2TUMIRicOkvd3tJsU2WECIUgxpB\nk6c81EfLBxfFAa65Y9+Pas5dgrfsP/5r2ylvvgkoCUYkPruhGRo35WJqn7l7+tkj\nY+uYdLRIrtaW1LJnHAfNIuCNuL+/mvOdf7EGs+S6xXoZlcE7czeYoKtvoIBwNUx1\n5IN+boTvAgMBAAECggEAB+GpYH36agmLKIPVQtF/KCz6XAcViXYOt0gIm6oFMWKr\nZirDZo14ccZodeuoK1Cp/z6vzgIYvWWkW4okj88ZnGSeIGA25I6eyEtJEafOZBWp\n8Djh3WBcKkczHF2TNMn8iL3cWuGhhmANhW672CZLH9n6XNWyK72OtnvOjw0zcE+n\naCsuLHQNQUFApDka7g4EWBm8C1a2R1a+qeci9um7PVT4B+LUfyLZE9vPIapjTf7R\nXId/KbPixyqs9Q+n+GCGKD5xUH123vd6AFiAKHGyaI1+8W12XfDHa+JCH2fj3ZFC\nmMPJIkOVN1pA/N8S9tETIsotyP/SQLcFaRknI/LMOQKBgQDOCP+w1yDLPsf46Pwb\neortDhKngI4fn4qNP8cVZkfSAYHpr+aqT5PpuLXKdR/XwOPe+ii5WCrtsU8VuVDu\nUAVt8+YQGZu9JvmORB8xYsbZdMmlEKAJGQ1GaYbbqwrEz+xDjyoQR1Ru01EPCEIa\nSHoVVmsyniivVjrPnue9GuPkRwKBgQDCJEe9ViBUdha/odVLANjVM/xcr/bUbQxP\nEj4vNXmUoYzzhA9mNiqXwVepgQqDbtbdf81VV9+rVQck0QkrSZ2KhQ5I+qX9ZNY0\nguNQKO9Upi4PLKotBAZXfQYEb479HnJHAE4JHFV46xBF5dm5fOQKZqii0M0ybf8J\nLIapFif2GQKBgQDHbhZxSgrIMMDHwl0lC/ylcNXFpL3tBjTKfE1r/VDPif4CAO25\nNMXrmYr9qVllMaRgFKyOmzUSVmpCkNoxkutufoLWWrNQ6ATvHClFWGM54b29NNZz\nd/hNi5+pyWnnD4uV6WHB2Al2LL1tW4UAg98IAFpK6KRg84qBpUKS3RBxyQKBgAg8\nOb7SVHTAvZ5LYxzXYFtK5T2ZSUMhjRAdmf2uqwWfBLeftneDfLMLRIiwLJ3+qaaj\nsTYZkCdYaAErzNPFP6WMl1qJJ1lkWaHIm5Pe6KgSlImYP2/BZ/N2Hjc59DrQe9B/\nNtA0H3wNnJcadO3lWlcGm8isSsgE2nitJtktU2yhAoGBAJeAWJPgTXKN1hDkr5cJ\nIdfkbdpFrFWFt9qSo1UXcXBr+1uu8cfYyg+FCel8oiX3lmBeIxHEkqrJPg5cmU9e\nC3BIBej80Y5bwjU7ZRDzSE03oR0poac0EMEip+Exvxb7uS3U0ukj1e98IbvTxj4d\nUgQ6ledPvJO+S+vuq63QNDk8\n-----END PRIVATE KEY-----";

/// The MCP server: answers any JSON-RPC request with `{ tools: [] }` for `access-1`/`access-2`,
/// and challenges everything else.
struct McpServer {
    challenge: String,
}

impl Respond for McpServer {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let authorization = request
            .headers
            .get("authorization")
            .and_then(|v| v.to_str().ok());
        if matches!(authorization, Some("Bearer access-1" | "Bearer access-2")) {
            return ResponseTemplate::new(200).set_body_json(answer(request));
        }
        ResponseTemplate::new(401).insert_header("WWW-Authenticate", self.challenge.as_str())
    }
}

/// The token endpoint: `access-2` for a refresh, `access-1` for a code.
struct TokenEndpoint;

impl Respond for TokenEndpoint {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let token = if form(request).get("grant_type").map(String::as_str) == Some("refresh_token")
        {
            "access-2"
        } else {
            "access-1"
        };
        ResponseTemplate::new(200).set_body_json(
            json!({ "access_token": token, "refresh_token": "refresh-1", "expires_in": 3600 }),
        )
    }
}

/// The spec's answer to an authorized request: `server/discover` gets the supported versions,
/// everything else `{ tools: [] }`.
fn answer(request: &Request) -> Value {
    let message: Value = serde_json::from_slice(&request.body).unwrap_or(Value::Null);
    let result = if message["method"] == "server/discover" {
        json!({ "supportedVersions": ["2026-07-28"] })
    } else {
        json!({ "tools": [] })
    };
    json!({ "jsonrpc": "2.0", "id": message["id"], "result": result })
}

fn header(request: &Request, name: &str) -> Option<String> {
    request
        .headers
        .get(name)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string)
}

fn form(request: &Request) -> HashMap<String, String> {
    reqwest::Url::parse(&format!(
        "http://x/?{}",
        String::from_utf8_lossy(&request.body)
    ))
    .map(|u| u.query_pairs().into_owned().collect())
    .unwrap_or_default()
}

fn query(url: &str) -> HashMap<String, String> {
    reqwest::Url::parse(url)
        .unwrap()
        .query_pairs()
        .into_owned()
        .collect()
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
        format!(
            r#"Bearer resource_metadata="{}/.well-known/oauth-protected-resource/mcp", scope="issues:read""#,
            self.mcp.uri()
        )
    }

    /// `linear_class`: `url`, `inputs :user`, `oauth owner: :user`.
    fn linear(&self, user: &str) -> Mcp {
        self.mcp_with(OAuthSettings::new().owner(user))
    }

    fn mcp_with(&self, settings: OAuthSettings) -> Mcp {
        Mcp::url(self.server_url())
            .oauth(settings)
            .config(self.config.clone())
            .build()
            .unwrap()
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
        params
            .into_iter()
            .filter_map(|(k, v)| v.map(|v| (k, v)))
            .collect()
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
        Mock::given(method(verb))
            .and(path(at))
            .respond_with(response)
            .with_priority(1)
            .mount(server)
            .await;
    }
}

/// The spec's `before` block.
async fn world() -> World {
    let (mcp, auth) = (MockServer::start().await, MockServer::start().await);
    let store = Arc::new(MemoryStore::new());
    let mut config = Config::default();
    config.mcp_credential_store = Some(store.clone());
    let world = World {
        mcp,
        auth,
        config: Arc::new(config),
        store,
    };
    let server_url = world.server_url();
    Mock::given(method("POST"))
        .and(path("/mcp"))
        .respond_with(McpServer {
            challenge: world.challenge(),
        })
        .mount(&world.mcp)
        .await;
    Mock::given(method("GET"))
        .and(path("/.well-known/oauth-protected-resource/mcp"))
        .respond_with(ResponseTemplate::new(200).set_body_json(
            json!({ "resource": server_url, "authorization_servers": [world.issuer()] }),
        ))
        .mount(&world.mcp)
        .await;
    Mock::given(method("GET"))
        .and(path("/.well-known/oauth-authorization-server"))
        .respond_with(ResponseTemplate::new(200).set_body_json(world.authorization_server()))
        .mount(&world.auth)
        .await;
    Mock::given(method("POST"))
        .and(path("/register"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(json!({ "client_id": "registered" })),
        )
        .mount(&world.auth)
        .await;
    Mock::given(method("POST"))
        .and(path("/token"))
        .respond_with(TokenEndpoint)
        .mount(&world.auth)
        .await;
    world
}

fn mcp_error(error: Error) -> String {
    match error {
        Error::Mcp(e) => e.message,
        other => panic!("expected MCP::Error, got {other:?}"),
    }
}

// spec: mcp/oauth_spec.rb:103 starts an authorization with PKCE, the resource, and the challenged scope
#[tokio::test]
async fn starts_an_authorization_with_pkce_the_resource_and_the_challenged_scope() {
    let w = world().await;
    let url = w
        .linear("ada")
        .authorization_url(REDIRECT_URI)
        .await
        .unwrap();
    let params = query(&url);

    assert!(
        url.starts_with(&format!("{}/authorize?", w.issuer())),
        "{url}"
    );
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
    assert_eq!(
        (
            &body["application_type"],
            &body["token_endpoint_auth_method"]
        ),
        (&json!("web"), &json!("none"))
    );
}

// spec: mcp/oauth_spec.rb:135 exchanges the code and uses the token
#[tokio::test]
async fn exchanges_the_code_and_uses_the_token() {
    let w = world().await;
    let linear = w.linear("ada");
    w.authorize(&linear).await;

    assert!(linear.is_authorized().await.unwrap());
    assert!(linear.tools().await.unwrap().is_empty());
    let exchanges = w.requests(&w.auth, "POST", "/token").await;
    let exchange = exchanges
        .iter()
        .map(form)
        .find(|f| f.get("grant_type").map(String::as_str) == Some("authorization_code"))
        .unwrap();
    assert_eq!(exchange.get("resource"), Some(&w.server_url()));
    assert!(exchange.get("code_verifier").is_some_and(|v| !v.is_empty()));
}

// spec: mcp/oauth_spec.rb:155 sends the server and OAuth requests through the connection of its context
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
    let linear = Mcp::url(w.server_url())
        .oauth(OAuthSettings::new().owner("ada"))
        .config(Arc::new(config))
        .build()
        .unwrap();

    w.authorize(&linear).await;
    linear.tools().await.unwrap();

    assert_eq!(
        w.requests(&w.auth, "GET", "/.well-known/oauth-authorization-server")
            .await
            .len(),
        1
    );
    let registration = w.requests(&w.auth, "POST", "/register").await;
    assert_eq!(
        serde_json::from_slice::<Value>(&registration[0].body).unwrap()["client_name"],
        "Context client"
    );
    assert!(!w.requests(&w.auth, "POST", "/token").await.is_empty());
    assert!(!w.requests(&w.mcp, "POST", "/mcp").await.is_empty());
    assert!(
        store
            .read(&format!("ada@{}", w.server_url()))
            .await
            .unwrap()
            .is_some()
    );
    assert!(
        w.store
            .read(&format!("ada@{}", w.server_url()))
            .await
            .unwrap()
            .is_none()
    );
}

// spec: mcp/oauth_spec.rb:173 keeps credentials per owner
#[tokio::test]
async fn keeps_credentials_per_owner() {
    let w = world().await;
    w.authorize(&w.linear("ada")).await;

    assert!(!w.linear("grace").is_authorized().await.unwrap());
}

// spec: mcp/oauth_spec.rb:179 refreshes an expired token when the server rejects it
#[tokio::test]
async fn refreshes_an_expired_token_when_the_server_rejects_it() {
    let w = world().await;
    w.authorize(&w.linear("ada")).await;
    let key = format!("ada@{}", w.server_url());
    let mut stale = w.store.read(&key).await.unwrap().unwrap();
    stale["access_token"] = json!("stale");
    w.store.write(&key, stale, None).await.unwrap();

    assert!(w.linear("ada").tools().await.unwrap().is_empty());
    assert_eq!(
        w.store.read(&key).await.unwrap().unwrap()["access_token"],
        "access-2"
    );
}

// spec: mcp/oauth_spec.rb:232 refuses a callback with the wrong state
#[tokio::test]
async fn refuses_a_callback_with_the_wrong_state() {
    let w = world().await;
    let linear = w.linear("ada");
    let url = linear.authorization_url(REDIRECT_URI).await.unwrap();

    let error = linear
        .authorize(w.callback(&url, &[("state", Some("forged"))]))
        .await
        .err()
        .unwrap();
    assert_eq!(mcp_error(error), "The authorization state does not match");
}

// spec: mcp/oauth_spec.rb:239 refuses a callback from another issuer, or none when one is required
#[tokio::test]
async fn refuses_a_callback_from_another_issuer_or_none_when_one_is_required() {
    let w = world().await;
    let linear = w.linear("ada");
    let url = linear.authorization_url(REDIRECT_URI).await.unwrap();

    let error = linear
        .authorize(w.callback(&url, &[("iss", Some("https://evil.example.com"))]))
        .await
        .err()
        .unwrap();
    assert!(mcp_error(error).contains("wrong issuer"));
    let error = linear
        .authorize(w.callback(&url, &[("iss", None)]))
        .await
        .err()
        .unwrap();
    assert!(mcp_error(error).contains("did not identify"));
}

// spec: mcp/oauth_spec.rb:306 uses a pre-registered client with its secret
#[tokio::test]
async fn uses_a_pre_registered_client_with_its_secret() {
    let w = world().await;
    let slack = w.mcp_with(
        OAuthSettings::new()
            .client_id("slack-app")
            .client_secret("shh"),
    );

    let url = slack.authorization_url(REDIRECT_URI).await.unwrap();
    slack.authorize(w.callback(&url, &[])).await.unwrap();

    assert!(w.requests(&w.auth, "POST", "/register").await.is_empty());
    let basic = format!(
        "Basic {}",
        base64::engine::general_purpose::STANDARD.encode("slack-app:shh")
    );
    let tokens = w.requests(&w.auth, "POST", "/token").await;
    assert!(tokens.iter().any(
        |r| r.headers.get("authorization").and_then(|v| v.to_str().ok()) == Some(basic.as_str())
    ));
}

// spec: mcp/oauth_spec.rb:350 refuses authorization servers without PKCE
#[tokio::test]
async fn refuses_authorization_servers_without_pkce() {
    let w = world().await;
    let mut server = w.authorization_server();
    server
        .as_object_mut()
        .unwrap()
        .remove("code_challenge_methods_supported");
    w.stub(
        &w.auth,
        "GET",
        "/.well-known/oauth-authorization-server",
        ResponseTemplate::new(200).set_body_json(server),
    )
    .await;

    let error = w
        .linear("ada")
        .authorization_url(REDIRECT_URI)
        .await
        .err()
        .unwrap();
    assert!(mcp_error(error).contains("PKCE"));
}

// spec: mcp/oauth_spec.rb:357 refuses metadata for another resource
#[tokio::test]
async fn refuses_metadata_for_another_resource() {
    let w = world().await;
    let metadata = json!({ "resource": "https://other.example.com/mcp", "authorization_servers": [w.issuer()] });
    w.stub(
        &w.mcp,
        "GET",
        "/.well-known/oauth-protected-resource/mcp",
        ResponseTemplate::new(200).set_body_json(metadata),
    )
    .await;

    let error = w
        .linear("ada")
        .authorization_url(REDIRECT_URI)
        .await
        .err()
        .unwrap();
    assert!(mcp_error(error).contains("another resource"));
}

/// `mcp` at `impostor` (`oauth` with no owner), whose server challenges with metadata on its
/// own origin claiming `resource` is the real server.
async fn impostor_error(w: &World, impostor_server: &MockServer, impostor: &str) -> String {
    let base = impostor.trim_end_matches("/mcp");
    let metadata = format!("{base}/.well-known/oauth-protected-resource/mcp");
    let challenge = format!(r#"Bearer resource_metadata="{metadata}""#);
    w.stub(
        impostor_server,
        "POST",
        "/mcp",
        ResponseTemplate::new(401).insert_header("WWW-Authenticate", challenge.as_str()),
    )
    .await;
    let claims = json!({ "resource": w.server_url(), "authorization_servers": [w.issuer()] });
    w.stub(
        impostor_server,
        "GET",
        "/.well-known/oauth-protected-resource/mcp",
        ResponseTemplate::new(200).set_body_json(claims),
    )
    .await;
    let mcp = Mcp::url(impostor)
        .oauth(OAuthSettings::new())
        .config(w.config.clone())
        .build()
        .unwrap();
    mcp_error(mcp.authorization_url(REDIRECT_URI).await.err().unwrap())
}

// spec: mcp/oauth_spec.rb:366 refuses a server at #{impostor} claiming another server's resource
// Both impostors: another host on the same port (`localhost` against `127.0.0.1`, standing in for
// `mcp.example.com.attacker.io`), and the same host on another port (`:8443`).
#[tokio::test]
async fn refuses_a_server_at_another_host_or_port_claiming_another_servers_resource() {
    let w = world().await;
    let port = w.mcp.address().port();
    assert!(
        impostor_error(&w, &w.mcp, &format!("http://localhost:{port}/mcp"))
            .await
            .contains("another resource")
    );

    let other_port = MockServer::start().await;
    let impostor = format!("{}/mcp", other_port.uri());
    assert!(
        impostor_error(&w, &other_port, &impostor)
            .await
            .contains("another resource")
    );
}

// spec: mcp/oauth_spec.rb:394 refuses an authorization endpoint that is not HTTPS
#[tokio::test]
async fn refuses_an_authorization_endpoint_that_is_not_https() {
    let w = world().await;
    let mut server = w.authorization_server();
    server["authorization_endpoint"] = json!("javascript:alert(1)");
    w.stub(
        &w.auth,
        "GET",
        "/.well-known/oauth-authorization-server",
        ResponseTemplate::new(200).set_body_json(server),
    )
    .await;

    let error = w
        .linear("ada")
        .authorization_url(REDIRECT_URI)
        .await
        .err()
        .unwrap();
    assert!(mcp_error(error).contains("HTTPS"));
}

// spec: mcp/oauth_spec.rb:401 needs the declared owner
#[tokio::test]
async fn needs_the_declared_owner() {
    let w = world().await;
    let nobody = w.mcp_with(OAuthSettings::new().owner_with(|| None));

    let error = nobody.is_authorized().await.err().unwrap();
    assert!(
        matches!(&error, Error::Argument(m) if m.contains("needs an owner")),
        "{error:?}"
    );
}

// spec: mcp/oauth_spec.rb:405 keeps refreshing with the token endpoint that issued the token
#[tokio::test]
async fn keeps_refreshing_with_the_token_endpoint_that_issued_the_token() {
    let w = world().await;
    let linear = w.linear("ada");
    w.authorize(&linear).await;
    let evil = MockServer::start().await;
    let mut server = w.authorization_server();
    server["token_endpoint"] = json!(format!("{}/token", evil.uri()));
    w.stub(
        &w.auth,
        "GET",
        "/.well-known/oauth-authorization-server",
        ResponseTemplate::new(200).set_body_json(server),
    )
    .await;
    linear.authorization_url(REDIRECT_URI).await.unwrap();

    assert!(w.linear("ada").oauth().unwrap().refresh().await.unwrap());

    assert!(
        evil.received_requests()
            .await
            .unwrap_or_default()
            .is_empty()
    );
}

// spec: mcp/oauth_spec.rb:416 ignores metadata URLs on other hosts
#[tokio::test]
async fn ignores_metadata_urls_on_other_hosts() {
    let w = world().await;
    let internal = MockServer::start().await;
    let challenge = format!(r#"Bearer resource_metadata="{}/metadata""#, internal.uri());
    w.stub(
        &w.mcp,
        "POST",
        "/mcp",
        ResponseTemplate::new(401).insert_header("WWW-Authenticate", challenge.as_str()),
    )
    .await;

    w.linear("ada")
        .authorization_url(REDIRECT_URI)
        .await
        .unwrap();

    assert!(
        internal
            .received_requests()
            .await
            .unwrap_or_default()
            .is_empty()
    );
    assert_eq!(
        w.requests(&w.mcp, "GET", "/.well-known/oauth-protected-resource/mcp")
            .await
            .len(),
        1
    );
}

fn scopes(url: &str) -> Vec<String> {
    let mut scopes: Vec<String> = query(url)
        .get("scope")
        .map(|s| s.split(' ').map(str::to_string).collect())
        .unwrap_or_default();
    scopes.sort();
    scopes
}

async fn insufficient_scope(w: &World) {
    let challenge = r#"Bearer error="insufficient_scope", scope="issues:write""#;
    w.stub(
        &w.mcp,
        "POST",
        "/mcp",
        ResponseTemplate::new(403).insert_header("WWW-Authenticate", challenge),
    )
    .await;
}

// spec: mcp/oauth_spec.rb:427 asks for challenged scopes along with the ones already granted
#[tokio::test]
async fn asks_for_challenged_scopes_along_with_the_ones_already_granted() {
    let w = world().await;
    w.authorize(&w.linear("ada")).await;
    insufficient_scope(&w).await;
    let step_up = w.linear("ada");

    assert!(matches!(step_up.tools().await, Err(Error::Forbidden(..))));
    assert_eq!(
        scopes(&step_up.authorization_url(REDIRECT_URI).await.unwrap()),
        ["issues:read", "issues:write"]
    );
}

// spec: mcp/oauth_spec.rb:439 uses the server origin for servers without protected resource metadata
#[tokio::test]
async fn uses_the_server_origin_for_servers_without_protected_resource_metadata() {
    let w = world().await;
    for at in [
        "/.well-known/oauth-protected-resource/mcp",
        "/.well-known/oauth-protected-resource",
        "/.well-known/oauth-authorization-server",
    ] {
        w.stub(&w.mcp, "GET", at, ResponseTemplate::new(404)).await;
    }
    w.stub(
        &w.mcp,
        "POST",
        "/register",
        ResponseTemplate::new(200).set_body_json(json!({ "client_id": "legacy" })),
    )
    .await;
    w.stub(&w.mcp, "POST", "/mcp", ResponseTemplate::new(401))
        .await;

    let url = w
        .linear("ada")
        .authorization_url(REDIRECT_URI)
        .await
        .unwrap();

    assert!(
        url.starts_with(&format!("{}/authorize?", w.mcp.uri())),
        "{url}"
    );
    assert_eq!(
        query(&url).get("client_id").map(String::as_str),
        Some("legacy")
    );
}

// spec: mcp/oauth_spec.rb:452 adds challenged scopes to configured ones
#[tokio::test]
async fn adds_challenged_scopes_to_configured_ones() {
    let w = world().await;
    let scoped = w.mcp_with(OAuthSettings::new().owner("ada").scopes(&["issues:read"]));
    w.authorize(&scoped).await;
    insufficient_scope(&w).await;

    assert!(matches!(scoped.tools().await, Err(Error::Forbidden(..))));
    assert_eq!(
        scopes(&scoped.authorization_url(REDIRECT_URI).await.unwrap()),
        ["issues:read", "issues:write"]
    );
}

// spec: mcp/oauth_spec.rb:469 refuses a legacy authorization server without PKCE
#[tokio::test]
async fn refuses_a_legacy_authorization_server_without_pkce() {
    let w = world().await;
    for at in [
        "/.well-known/oauth-protected-resource/mcp",
        "/.well-known/oauth-protected-resource",
    ] {
        w.stub(&w.mcp, "GET", at, ResponseTemplate::new(404)).await;
    }
    let origin = w.mcp.uri();
    let legacy = json!({ "issuer": origin, "authorization_endpoint": format!("{origin}/authorize"), "token_endpoint": format!("{origin}/token") });
    w.stub(
        &w.mcp,
        "GET",
        "/.well-known/oauth-authorization-server",
        ResponseTemplate::new(200).set_body_json(legacy),
    )
    .await;
    w.stub(&w.mcp, "POST", "/mcp", ResponseTemplate::new(401))
        .await;

    let error = w
        .linear("ada")
        .authorization_url(REDIRECT_URI)
        .await
        .err()
        .unwrap();
    assert!(mcp_error(error).contains("PKCE"));
}

// spec: mcp/oauth_spec.rb:480 refreshes once when the server keeps rejecting the token
#[tokio::test]
async fn refreshes_once_when_the_server_keeps_rejecting_the_token() {
    let w = world().await;
    w.authorize(&w.linear("ada")).await;
    w.stub(&w.mcp, "POST", "/mcp", ResponseTemplate::new(401))
        .await;

    assert!(matches!(
        w.linear("ada").tools().await,
        Err(Error::Unauthorized(..))
    ));
    let refreshes = w.requests(&w.auth, "POST", "/token").await;
    let refreshes = refreshes
        .iter()
        .filter(|r| String::from_utf8_lossy(&r.body).contains("refresh_token"))
        .count();
    assert_eq!(refreshes, 1);
}

// spec: mcp/oauth_spec.rb:556 forgets credentials
#[tokio::test]
async fn forgets_credentials() {
    let w = world().await;
    w.authorize(&w.linear("ada")).await;

    let linear = w.linear("ada");
    assert!(
        !linear
            .deauthorize()
            .await
            .unwrap()
            .is_authorized()
            .await
            .unwrap()
    );
}

// spec: mcp_spec.rb:1210 accepts a prefix and OAuth settings
#[tokio::test]
async fn accepts_a_prefix_and_oauth_settings() {
    let linear = Mcp::url("https://mcp.linear.app/mcp")
        .prefix("mcp_1")
        .oauth(OAuthSettings::new().owner("owner-1").scopes(&["read"]))
        .build()
        .unwrap();

    let settings = linear.oauth_settings().unwrap();
    assert_eq!(settings.scopes.as_deref(), Some(&["read".to_string()][..]));
    assert_eq!(
        settings.owner.as_ref().and_then(|owner| owner()).as_deref(),
        Some("owner-1")
    );
    assert_eq!(linear.prefix(), Some("mcp_1"));
}

// ---------------------------------------------------------------------------------------------
// RubyLLM 2.1: registrations per scope, issuer handling, refresh locking, the grants that need no
// user, and DPoP.
// ---------------------------------------------------------------------------------------------

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use base64::engine::general_purpose::{STANDARD, URL_SAFE_NO_PAD};
use ring::signature::{self as ring_signature, UnparsedPublicKey};

fn basic(credentials: &str) -> String {
    format!("Basic {}", STANDARD.encode(credentials))
}

fn decode_part(part: &str) -> Value {
    serde_json::from_slice(&URL_SAFE_NO_PAD.decode(part).unwrap()).unwrap()
}

/// `verified(jwt, key)` with the public key's bytes: the header and claims of a JWT whose
/// signature verifies.
fn verified(
    jwt: &str,
    algorithm: &'static dyn ring_signature::VerificationAlgorithm,
    public_key: &[u8],
) -> (Value, Value) {
    let (input, signature) = jwt.rsplit_once('.').unwrap();
    UnparsedPublicKey::new(algorithm, public_key)
        .verify(
            input.as_bytes(),
            &URL_SAFE_NO_PAD.decode(signature).unwrap(),
        )
        .expect("The signature does not verify");
    let (header, claims) = input.split_once('.').unwrap();
    (decode_part(header), decode_part(claims))
}

/// `proved(proof)`: verifies a DPoP proof with the JWK in its own header.
fn proved(proof: &str) -> (Value, Value) {
    let header = decode_part(proof.split('.').next().unwrap());
    let jwk = &header["jwk"];
    let mut point = vec![4u8];
    point.extend(URL_SAFE_NO_PAD.decode(jwk["x"].as_str().unwrap()).unwrap());
    point.extend(URL_SAFE_NO_PAD.decode(jwk["y"].as_str().unwrap()).unwrap());
    verified(proof, &ring_signature::ECDSA_P256_SHA256_FIXED, &point)
}

/// The public key of an EC key pair ring generates, with its PKCS#8 PEM.
fn ec_key() -> (String, Vec<u8>) {
    use ring::signature::KeyPair;
    let rng = ring::rand::SystemRandom::new();
    let document = ring_signature::EcdsaKeyPair::generate_pkcs8(
        &ring_signature::ECDSA_P256_SHA256_FIXED_SIGNING,
        &rng,
    )
    .unwrap();
    let pair = ring_signature::EcdsaKeyPair::from_pkcs8(
        &ring_signature::ECDSA_P256_SHA256_FIXED_SIGNING,
        document.as_ref(),
        &rng,
    )
    .unwrap();
    let body = STANDARD.encode(document.as_ref());
    let lines: Vec<String> = body
        .as_bytes()
        .chunks(64)
        .map(|c| String::from_utf8_lossy(c).into_owned())
        .collect();
    let pem = format!(
        "-----BEGIN PRIVATE KEY-----\n{}\n-----END PRIVATE KEY-----\n",
        lines.join("\n")
    );
    (pem, pair.public_key().as_ref().to_vec())
}

fn rsa_public_key() -> Vec<u8> {
    use ring::signature::KeyPair;
    let body: String = RSA_KEY
        .lines()
        .filter(|l| !l.starts_with("-----"))
        .collect();
    let pair = ring_signature::RsaKeyPair::from_pkcs8(&STANDARD.decode(body).unwrap()).unwrap();
    pair.public_key().as_ref().to_vec()
}

impl World {
    /// `token_form`: the form of the one token request.
    async fn token_form(&self) -> HashMap<String, String> {
        let tokens = self.requests(&self.auth, "POST", "/token").await;
        assert_eq!(tokens.len(), 1, "token requests");
        form(&tokens[0])
    }

    /// `declared_extensions`: the distinct `extensions` the requests to the server declared.
    async fn declared_extensions(&self) -> Vec<Value> {
        let mut declared: Vec<Value> = Vec::new();
        let requests = self.requests(&self.mcp, "POST", "/mcp").await;
        assert!(!requests.is_empty());
        for request in requests {
            let body: Value = serde_json::from_slice(&request.body).unwrap();
            let extensions = body
                .pointer("/params/_meta/io.modelcontextprotocol~1clientCapabilities/extensions")
                .cloned()
                .unwrap_or(Value::Null);
            if !declared.contains(&extensions) {
                declared.push(extensions);
            }
        }
        declared
    }

    /// `moved_to(issuer)`: the server now names `issuer`, whose metadata is served by `server`.
    async fn moved_to(&self, server: &MockServer) {
        let issuer = server.uri();
        let metadata = json!({ "resource": self.server_url(), "authorization_servers": [issuer] });
        self.stub(
            &self.mcp,
            "GET",
            "/.well-known/oauth-protected-resource/mcp",
            ResponseTemplate::new(200).set_body_json(metadata),
        )
        .await;
        let mut moved = self.authorization_server();
        moved["issuer"] = json!(issuer);
        moved["authorization_endpoint"] = json!(format!("{issuer}/authorize"));
        moved["token_endpoint"] = json!(format!("{issuer}/token"));
        Mock::given(method("GET"))
            .and(path("/.well-known/oauth-authorization-server"))
            .respond_with(ResponseTemplate::new(200).set_body_json(moved))
            .mount(server)
            .await;
    }

    fn key(&self, owner: &str) -> String {
        format!("{owner}@{}", self.server_url())
    }

    async fn merge(&self, key: &str, changes: Value) {
        let mut data = self.store.read(key).await.unwrap().unwrap();
        for (k, v) in changes.as_object().unwrap() {
            data[k] = v.clone();
        }
        self.store.write(key, data, None).await.unwrap();
    }

    /// The token requests whose body includes `refresh_token`.
    async fn refreshes(&self) -> usize {
        self.requests(&self.auth, "POST", "/token")
            .await
            .iter()
            .filter(|r| String::from_utf8_lossy(&r.body).contains("refresh_token"))
            .count()
    }
}

fn client_id_in(url: &str) -> Option<String> {
    query(url).get("client_id").cloned()
}

// spec: mcp/oauth_spec.rb:116 registers for the scopes it requests, and again when it needs more
#[tokio::test]
async fn registers_for_the_scopes_it_requests_and_again_when_it_needs_more() {
    let w = world().await;
    w.linear("ada")
        .authorization_url(REDIRECT_URI)
        .await
        .unwrap();
    let url = w
        .linear("ada")
        .authorization_url(REDIRECT_URI)
        .await
        .unwrap();
    let writer = w.mcp_with(OAuthSettings::new().owner("bob").scopes(&["issues:write"]));
    writer.authorization_url(REDIRECT_URI).await.unwrap();

    let scopes: Vec<Value> = w
        .requests(&w.auth, "POST", "/register")
        .await
        .iter()
        .map(|r| serde_json::from_slice::<Value>(&r.body).unwrap()["scope"].clone())
        .collect();
    assert_eq!(
        scopes,
        [json!("issues:read"), json!("issues:write issues:read")]
    );
    assert_eq!(client_id_in(&url).as_deref(), Some("registered"));
}

// spec: mcp/oauth_spec.rb:147 declares no authorization extension for apps that users authorize
#[tokio::test]
async fn declares_no_authorization_extension_for_apps_that_users_authorize() {
    let w = world().await;
    w.authorize(&w.linear("ada")).await;

    w.linear("ada").tools().await.unwrap();

    assert_eq!(w.declared_extensions().await, [Value::Null]);
}

/// `rotating_token_endpoint`: a refresh token works once; each answer takes 200 ms.
struct RotatingTokenEndpoint {
    used: Mutex<Vec<String>>,
}

impl Respond for RotatingTokenEndpoint {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let token = form(request)
            .get("refresh_token")
            .cloned()
            .unwrap_or_default();
        let reused = {
            let mut used = self.used.lock().unwrap();
            let reused = used.contains(&token);
            used.push(token);
            reused
        };
        let delay = std::time::Duration::from_millis(200);
        if reused {
            return ResponseTemplate::new(400)
                .set_body_json(json!({ "error": "invalid_grant" }))
                .set_delay(delay);
        }
        ResponseTemplate::new(200)
            .set_body_json(json!({ "access_token": "access-2", "refresh_token": "refresh-2", "expires_in": 3600 }))
            .set_delay(delay)
    }
}

async fn rotating_token_endpoint(w: &World) {
    Mock::given(method("POST"))
        .and(path("/token"))
        .respond_with(RotatingTokenEndpoint {
            used: Mutex::new(Vec::new()),
        })
        .with_priority(1)
        .mount(&w.auth)
        .await;
}

// spec: mcp/oauth_spec.rb:210 refreshing a rotating token > refreshes once when threads refresh together
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn refreshes_once_when_threads_refresh_together() {
    let w = world().await;
    w.authorize(&w.linear("ada")).await;
    let before = w.refreshes().await;
    let key = w.key("ada");
    w.merge(&key, json!({ "expires_at": 0 })).await;
    rotating_token_endpoint(&w).await;

    let (one, two) = (
        w.linear("ada").oauth().unwrap(),
        w.linear("ada").oauth().unwrap(),
    );
    let tokens = tokio::join!(
        tokio::spawn(async move { one.access_token().await.unwrap() }),
        tokio::spawn(async move { two.access_token().await.unwrap() })
    );

    assert_eq!(
        [tokens.0.unwrap(), tokens.1.unwrap()],
        [Some("access-2".to_string()), Some("access-2".to_string())]
    );
    assert_eq!(w.refreshes().await - before, 1);
    assert_eq!(
        w.store.read(&key).await.unwrap().unwrap()["refresh_token"],
        "refresh-2"
    );
}

// spec: mcp/oauth_spec.rb:221 refreshing a rotating token > uses the token another worker refreshed since the server rejected its own
#[tokio::test]
async fn uses_the_token_another_worker_refreshed_since_the_server_rejected_its_own() {
    let w = world().await;
    w.authorize(&w.linear("ada")).await;
    rotating_token_endpoint(&w).await;
    let oauth = w.linear("ada").oauth().unwrap();
    oauth.access_token().await.unwrap();
    w.merge(
        &w.key("ada"),
        json!({ "access_token": "access-2", "refresh_token": "refresh-2" }),
    )
    .await;

    assert!(oauth.refresh().await.unwrap());
    assert_eq!(
        oauth.access_token().await.unwrap().as_deref(),
        Some("access-2")
    );
    assert_eq!(w.refreshes().await, 0);
}

/// The listed authorization server serves `metadata`; `listed` is what the protected resource
/// metadata names.
async fn lists(w: &World, listed: &str, published: &str) {
    let metadata = json!({ "resource": w.server_url(), "authorization_servers": [listed] });
    w.stub(
        &w.mcp,
        "GET",
        "/.well-known/oauth-protected-resource/mcp",
        ResponseTemplate::new(200).set_body_json(metadata),
    )
    .await;
    let mut server = w.authorization_server();
    server["issuer"] = json!(published);
    w.stub(
        &w.auth,
        "GET",
        "/.well-known/oauth-authorization-server",
        ResponseTemplate::new(200).set_body_json(server),
    )
    .await;
}

// spec: mcp/oauth_spec.rb:249 accepts issuer #{published} for authorization server #{listed}
// Both pairs: the issuer listed with a trailing slash and published without, and the reverse.
#[tokio::test]
async fn accepts_an_issuer_with_or_without_a_trailing_slash_for_the_listed_authorization_server() {
    for (listed, published) in [("/", ""), ("", "/")] {
        let w = world().await;
        let (listed, published) = (
            format!("{}{listed}", w.issuer()),
            format!("{}{published}", w.issuer()),
        );
        lists(&w, &listed, &published).await;
        let linear = w.linear("ada");

        let url = linear.authorization_url(REDIRECT_URI).await.unwrap();
        linear
            .authorize(w.callback(&url, &[("iss", Some(&published))]))
            .await
            .unwrap();

        assert!(
            linear.is_authorized().await.unwrap(),
            "{listed} / {published}"
        );
    }
}

// spec: mcp/oauth_spec.rb:262 compares the iss parameter exactly, without normalizing a trailing slash
#[tokio::test]
async fn compares_the_iss_parameter_exactly_without_normalizing_a_trailing_slash() {
    let w = world().await;
    let linear = w.linear("ada");
    let url = linear.authorization_url(REDIRECT_URI).await.unwrap();

    let slashed = format!("{}/", w.issuer());
    let error = linear
        .authorize(w.callback(&url, &[("iss", Some(&slashed))]))
        .await
        .err()
        .unwrap();
    assert!(mcp_error(error).contains("wrong issuer"));
}

/// `when the listed issuer serves metadata naming another issuer`: the server lists `api`, whose
/// metadata names the real issuer.
async fn names_another_issuer(w: &World) -> MockServer {
    let api = MockServer::start().await;
    let metadata = json!({ "resource": w.server_url(), "authorization_servers": [api.uri()] });
    w.stub(
        &w.mcp,
        "GET",
        "/.well-known/oauth-protected-resource/mcp",
        ResponseTemplate::new(200).set_body_json(metadata),
    )
    .await;
    Mock::given(method("GET"))
        .and(path("/.well-known/oauth-authorization-server"))
        .respond_with(ResponseTemplate::new(200).set_body_json(w.authorization_server()))
        .mount(&api)
        .await;
    api
}

// spec: mcp/oauth_spec.rb:277 when the listed issuer serves metadata naming another issuer > follows it once to metadata the named issuer publishes about itself
#[tokio::test]
async fn follows_metadata_naming_another_issuer_once_to_what_that_issuer_publishes() {
    let w = world().await;
    let _api = names_another_issuer(&w).await;
    let linear = w.linear("ada");

    let url = linear.authorization_url(REDIRECT_URI).await.unwrap();
    linear.authorize(w.callback(&url, &[])).await.unwrap();

    assert!(
        url.starts_with(&format!("{}/authorize?", w.issuer())),
        "{url}"
    );
    assert!(linear.is_authorized().await.unwrap());
}

// spec: mcp/oauth_spec.rb:285 when the listed issuer serves metadata naming another issuer > compares iss with the issuer it confirmed
#[tokio::test]
async fn compares_iss_with_the_issuer_it_confirmed() {
    let w = world().await;
    let api = names_another_issuer(&w).await;
    let linear = w.linear("ada");
    let url = linear.authorization_url(REDIRECT_URI).await.unwrap();

    let error = linear
        .authorize(w.callback(&url, &[("iss", Some(&api.uri()))]))
        .await
        .err()
        .unwrap();
    assert!(mcp_error(error).contains("wrong issuer"));
}

// spec: mcp/oauth_spec.rb:292 when the listed issuer serves metadata naming another issuer > refuses when the named issuer does not name itself
#[tokio::test]
async fn refuses_when_the_named_issuer_does_not_name_itself() {
    let w = world().await;
    let _api = names_another_issuer(&w).await;
    let mut server = w.authorization_server();
    server["issuer"] = json!("https://other.example.com");
    w.stub(
        &w.auth,
        "GET",
        "/.well-known/oauth-authorization-server",
        ResponseTemplate::new(200).set_body_json(server),
    )
    .await;

    let error = w
        .linear("ada")
        .authorization_url(REDIRECT_URI)
        .await
        .err()
        .unwrap();
    assert!(mcp_error(error).contains("different issuer"));
}

// spec: mcp/oauth_spec.rb:299 when the listed issuer serves metadata naming another issuer > refuses when the named issuer publishes no metadata
#[tokio::test]
async fn refuses_when_the_named_issuer_publishes_no_metadata() {
    let w = world().await;
    let _api = names_another_issuer(&w).await;
    Mock::given(method("GET"))
        .and(wiremock::matchers::path_regex(r"^/\.well-known/"))
        .respond_with(ResponseTemplate::new(404))
        .with_priority(1)
        .mount(&w.auth)
        .await;

    let error = w
        .linear("ada")
        .authorization_url(REDIRECT_URI)
        .await
        .err()
        .unwrap();
    assert!(mcp_error(error).contains("publishes no"));
}

fn pre_registered(w: &World, client_id: &str) -> Mcp {
    w.mcp_with(
        OAuthSettings::new()
            .client_id(client_id)
            .client_secret("shh"),
    )
}

// spec: mcp/oauth_spec.rb:329 a pre-registered client > stays with the authorization server it was first used with
#[tokio::test]
async fn a_pre_registered_client_stays_with_the_authorization_server_it_was_first_used_with() {
    let w = world().await;
    let slack = pre_registered(&w, "slack-app");
    w.authorize(&slack).await;
    let elsewhere = MockServer::start().await;
    w.moved_to(&elsewhere).await;

    let error = pre_registered(&w, "slack-app")
        .authorization_url(REDIRECT_URI)
        .await
        .err()
        .unwrap();
    assert_eq!(
        mcp_error(error),
        format!(
            "slack-app is registered with {}, but {} now uses {}",
            w.issuer(),
            w.server_url(),
            elsewhere.uri()
        )
    );
    assert!(
        elsewhere
            .received_requests()
            .await
            .unwrap()
            .iter()
            .all(|r| r.url.path() != "/token")
    );
}

// spec: mcp/oauth_spec.rb:340 a pre-registered client > binds another client to the authorization server it is used with
#[tokio::test]
async fn binds_another_client_to_the_authorization_server_it_is_used_with() {
    let w = world().await;
    pre_registered(&w, "slack-app")
        .authorization_url(REDIRECT_URI)
        .await
        .unwrap();
    let elsewhere = MockServer::start().await;
    w.moved_to(&elsewhere).await;

    let url = pre_registered(&w, "elsewhere-app")
        .authorization_url(REDIRECT_URI)
        .await
        .unwrap();

    assert!(
        url.starts_with(&format!("{}/authorize?", elsewhere.uri())),
        "{url}"
    );
}

async fn forgotten_client(w: &World) {
    let refusal = json!({ "error": "invalid_client", "error_description": "Unknown client" });
    w.stub(
        &w.auth,
        "POST",
        "/token",
        ResponseTemplate::new(401).set_body_json(refusal),
    )
    .await;
}

/// `registers_as(client)`: a newer stub wins over earlier ones, like WebMock's.
async fn registers_as(w: &World, client: Value, priority: u8) {
    Mock::given(method("POST"))
        .and(path("/register"))
        .respond_with(ResponseTemplate::new(200).set_body_json(client))
        .with_priority(priority)
        .mount(&w.auth)
        .await;
}

// spec: mcp/oauth_spec.rb:506 registrations the authorization server forgets > registers again after a refresh finds the client gone
#[tokio::test]
async fn registers_again_after_a_refresh_finds_the_client_gone() {
    let w = world().await;
    w.authorize(&w.linear("ada")).await;
    w.merge(&w.key("ada"), json!({ "access_token": "stale" }))
        .await;
    forgotten_client(&w).await;
    registers_as(&w, json!({ "client_id": "registered-again" }), 1).await;

    assert!(matches!(
        w.linear("ada").tools().await,
        Err(Error::Unauthorized(..))
    ));
    let url = w
        .linear("ada")
        .authorization_url(REDIRECT_URI)
        .await
        .unwrap();
    assert_eq!(client_id_in(&url).as_deref(), Some("registered-again"));
}

// spec: mcp/oauth_spec.rb:516 registrations the authorization server forgets > registers again after a code exchange finds the client gone
#[tokio::test]
async fn registers_again_after_a_code_exchange_finds_the_client_gone() {
    let w = world().await;
    let linear = w.linear("ada");
    let url = linear.authorization_url(REDIRECT_URI).await.unwrap();
    forgotten_client(&w).await;
    registers_as(&w, json!({ "client_id": "registered-again" }), 1).await;

    match linear.authorize(w.callback(&url, &[])).await.err().unwrap() {
        Error::Mcp(e) => {
            assert!(e.message.contains("Unknown client"), "{}", e.message);
            assert_eq!(e.data.as_ref().unwrap()["error"], "invalid_client");
        }
        other => panic!("expected MCP::Error, got {other:?}"),
    }
    let url = linear.authorization_url(REDIRECT_URI).await.unwrap();
    assert_eq!(client_id_in(&url).as_deref(), Some("registered-again"));
}

// spec: mcp/oauth_spec.rb:527 registrations the authorization server forgets > keeps a registration made since
#[tokio::test]
async fn keeps_a_registration_made_since() {
    let w = world().await;
    w.authorize(&w.linear("ada")).await;
    let registration = format!("client:{} {REDIRECT_URI}", w.issuer());
    w.store
        .write(&registration, json!({ "client_id": "newer" }), None)
        .await
        .unwrap();
    w.merge(&w.key("ada"), json!({ "access_token": "stale" }))
        .await;
    forgotten_client(&w).await;

    assert!(matches!(
        w.linear("ada").tools().await,
        Err(Error::Unauthorized(..))
    ));
    assert_eq!(
        w.store.read(&registration).await.unwrap(),
        Some(json!({ "client_id": "newer" }))
    );
}

// spec: mcp/oauth_spec.rb:538 registrations the authorization server forgets > drops the secret of the registration it replaces
#[tokio::test]
async fn drops_the_secret_of_the_registration_it_replaces() {
    let w = world().await;
    registers_as(
        &w,
        json!({ "client_id": "registered", "client_secret": "old-secret" }),
        2,
    )
    .await;
    w.authorize(&w.linear("ada")).await;
    let key = w.key("ada");
    w.merge(&key, json!({ "access_token": "stale" })).await;
    forgotten_client(&w).await;
    assert!(matches!(
        w.linear("ada").tools().await,
        Err(Error::Unauthorized(..))
    ));
    registers_as(&w, json!({ "client_id": "registered-again" }), 1).await;
    w.auth.reset().await;
    registers_as(&w, json!({ "client_id": "registered-again" }), 1).await;
    Mock::given(method("GET"))
        .and(path("/.well-known/oauth-authorization-server"))
        .respond_with(ResponseTemplate::new(200).set_body_json(w.authorization_server()))
        .mount(&w.auth)
        .await;
    Mock::given(method("POST"))
        .and(path("/token"))
        .respond_with(TokenEndpoint)
        .mount(&w.auth)
        .await;
    let again = w.linear("ada");

    w.authorize(&again).await;

    let stored = w.store.read(&key).await.unwrap().unwrap();
    assert_eq!(stored["client_id"], "registered-again");
    assert!(stored.get("client_secret").is_none(), "{stored}");
}

fn reports(w: &World, settings: OAuthSettings) -> Mcp {
    w.mcp_with(
        settings
            .grant(Grant::ClientCredentials)
            .client_id("reports"),
    )
}

// spec: mcp/oauth_spec.rb:571 the client credentials grant > requests a token when the server asks for one, authenticating with the secret
#[tokio::test]
async fn requests_a_client_credentials_token_when_the_server_asks_authenticating_with_the_secret() {
    let w = world().await;

    assert!(
        reports(&w, OAuthSettings::new().client_secret("shh"))
            .tools()
            .await
            .unwrap()
            .is_empty()
    );

    let form = w.token_form().await;
    assert_eq!(
        form.get("grant_type").map(String::as_str),
        Some("client_credentials")
    );
    assert_eq!(form.get("resource"), Some(&w.server_url()));
    assert_eq!(form.get("scope").map(String::as_str), Some("issues:read"));
    let tokens = w.requests(&w.auth, "POST", "/token").await;
    assert_eq!(
        header(&tokens[0], "authorization"),
        Some(basic("reports:shh"))
    );
    assert!(w.requests(&w.auth, "POST", "/register").await.is_empty());
}

// spec: mcp/oauth_spec.rb:581 the client credentials grant > form-encodes the client credentials before sending them as Basic authentication
#[tokio::test]
async fn form_encodes_the_client_credentials_before_sending_them_as_basic_authentication() {
    let w = world().await;

    reports(&w, OAuthSettings::new().client_secret("p:ss%word"))
        .tools()
        .await
        .unwrap();

    let tokens = w.requests(&w.auth, "POST", "/token").await;
    assert_eq!(
        header(&tokens[0], "authorization"),
        Some(basic("reports:p%3Ass%25word"))
    );
}

// spec: mcp/oauth_spec.rb:589 the client credentials grant > declares the client credentials extension with every request
#[tokio::test]
async fn declares_the_client_credentials_extension_with_every_request() {
    let w = world().await;

    reports(&w, OAuthSettings::new().client_secret("shh"))
        .tools()
        .await
        .unwrap();

    assert_eq!(
        w.declared_extensions().await,
        [json!({ "io.modelcontextprotocol/oauth-client-credentials": {} })]
    );
}

// spec: mcp/oauth_spec.rb:595 the client credentials grant > signs an assertion for the issuer with a private key instead of a secret
#[tokio::test]
async fn signs_an_assertion_for_the_issuer_with_a_private_key_instead_of_a_secret() {
    let w = world().await;
    let (pem, public_key) = ec_key();

    reports(&w, OAuthSettings::new().private_key(pem))
        .tools()
        .await
        .unwrap();

    let form = w.token_form().await;
    let (header, claims) = verified(
        &form["client_assertion"],
        &ring_signature::ECDSA_P256_SHA256_FIXED,
        &public_key,
    );
    assert_eq!(
        form.get("client_assertion_type").map(String::as_str),
        Some("urn:ietf:params:oauth:client-assertion-type:jwt-bearer")
    );
    assert!(!form.contains_key("client_id") && !form.contains_key("client_secret"));
    assert_eq!(
        header,
        json!({ "typ": "client-authentication+jwt", "alg": "ES256" })
    );
    assert_eq!(
        (&claims["iss"], &claims["sub"], &claims["aud"]),
        (&json!("reports"), &json!("reports"), &json!(w.issuer()))
    );
    assert_eq!(
        claims["exp"].as_i64().unwrap() - claims["iat"].as_i64().unwrap(),
        60
    );
}

// spec: mcp/oauth_spec.rb:609 the client credentials grant > signs with an RSA key in an algorithm the authorization server accepts
#[tokio::test]
async fn signs_with_an_rsa_key_in_an_algorithm_the_authorization_server_accepts() {
    let w = world().await;
    let mut server = w.authorization_server();
    server["token_endpoint_auth_signing_alg_values_supported"] = json!(["RS256", "ES256"]);
    w.stub(
        &w.auth,
        "GET",
        "/.well-known/oauth-authorization-server",
        ResponseTemplate::new(200).set_body_json(server),
    )
    .await;

    reports(&w, OAuthSettings::new().private_key(RSA_KEY))
        .tools()
        .await
        .unwrap();

    let form = w.token_form().await;
    let (header, _) = verified(
        &form["client_assertion"],
        &ring_signature::RSA_PKCS1_2048_8192_SHA256,
        &rsa_public_key(),
    );
    assert_eq!(header["alg"], "RS256");
}

// spec: mcp/oauth_spec.rb:620 the client credentials grant > requests a new token before the old one expires
#[tokio::test]
async fn requests_a_new_client_credentials_token_before_the_old_one_expires() {
    let w = world().await;
    reports(&w, OAuthSettings::new().client_secret("shh"))
        .tools()
        .await
        .unwrap();
    w.merge(&w.key(""), json!({ "expires_at": 0 })).await;

    reports(&w, OAuthSettings::new().client_secret("shh"))
        .tools()
        .await
        .unwrap();

    let grants = w
        .requests(&w.auth, "POST", "/token")
        .await
        .iter()
        .filter(|r| form(r).get("grant_type").map(String::as_str) == Some("client_credentials"))
        .count();
    assert_eq!(grants, 2);
}

// spec: mcp/oauth_spec.rb:632 the client credentials grant > raises the authorization server's reason after asking once
#[tokio::test]
async fn raises_the_authorization_servers_reason_after_asking_once() {
    let w = world().await;
    forgotten_client(&w).await;

    let error = reports(&w, OAuthSettings::new().client_secret("wrong"))
        .tools()
        .await
        .err()
        .unwrap();

    let host = reqwest::Url::parse(&w.issuer())
        .unwrap()
        .host_str()
        .unwrap()
        .to_string();
    match error {
        Error::Unauthorized(message, _) => assert_eq!(
            message,
            format!("{host} refused the request: Unknown client")
        ),
        other => panic!("expected UnauthorizedError, got {other:?}"),
    }
    assert_eq!(w.requests(&w.auth, "POST", "/token").await.len(), 1);
}

// spec: mcp/oauth_spec.rb:641 the client credentials grant > has no authorization for a user to complete
#[tokio::test]
async fn has_no_authorization_for_a_user_to_complete() {
    let w = world().await;

    let error = reports(&w, OAuthSettings::new().client_secret("shh"))
        .authorization_url(REDIRECT_URI)
        .await
        .err()
        .unwrap();

    assert!(
        matches!(&error, Error::Configuration(m) if m == "The client_credentials grant needs no authorization"),
        "{error:?}"
    );
}

// spec: mcp/oauth_spec.rb:646 the client credentials grant > signs the code exchange of an app that users authorize
#[tokio::test]
async fn signs_the_code_exchange_of_an_app_that_users_authorize() {
    let w = world().await;
    let (pem, public_key) = ec_key();
    let slack = w.mcp_with(OAuthSettings::new().client_id("slack-app").private_key(pem));

    w.authorize(&slack).await;

    let form = w.token_form().await;
    assert_eq!(
        form.get("grant_type").map(String::as_str),
        Some("authorization_code")
    );
    let (_, claims) = verified(
        &form["client_assertion"],
        &ring_signature::ECDSA_P256_SHA256_FIXED,
        &public_key,
    );
    assert_eq!(
        (&claims["iss"], &claims["sub"]),
        (&json!("slack-app"), &json!("slack-app"))
    );
}

fn workload(w: &World, settings: OAuthSettings) -> Mcp {
    w.mcp_with(settings)
}

// spec: mcp/oauth_spec.rb:671 the JWT bearer grant > presents the workload's assertion when the server asks for a token
#[tokio::test]
async fn presents_the_workloads_assertion_when_the_server_asks_for_a_token() {
    let w = world().await;

    assert!(
        workload(&w, OAuthSettings::new().assertion("workload-jwt"))
            .tools()
            .await
            .unwrap()
            .is_empty()
    );

    let expected: HashMap<String, String> = [
        ("grant_type", "urn:ietf:params:oauth:grant-type:jwt-bearer"),
        ("assertion", "workload-jwt"),
        ("resource", w.server_url().as_str()),
        ("scope", "issues:read"),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect();
    assert_eq!(w.token_form().await, expected);
    let tokens = w.requests(&w.auth, "POST", "/token").await;
    assert!(tokens.iter().all(|r| header(r, "authorization").is_none()));
}

/// Records the assertion of each token request.
struct AssertionRecorder {
    assertions: Arc<Mutex<Vec<String>>>,
}

impl Respond for AssertionRecorder {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        self.assertions
            .lock()
            .unwrap()
            .push(form(request).get("assertion").cloned().unwrap_or_default());
        ResponseTemplate::new(200)
            .set_body_json(json!({ "access_token": "access-1", "expires_in": 3600 }))
    }
}

// spec: mcp/oauth_spec.rb:680 the JWT bearer grant > reads the assertion again for every token
#[tokio::test]
async fn reads_the_assertion_again_for_every_token() {
    let w = world().await;
    let assertions = Arc::new(Mutex::new(Vec::new()));
    Mock::given(method("POST"))
        .and(path("/token"))
        .respond_with(AssertionRecorder {
            assertions: assertions.clone(),
        })
        .with_priority(1)
        .mount(&w.auth)
        .await;
    let rotating = Arc::new(AtomicUsize::new(0));
    let platform = || {
        let rotating = rotating.clone();
        workload(
            &w,
            OAuthSettings::new().assertion_with(move || {
                Some(["first", "second"][rotating.fetch_add(1, Ordering::SeqCst)].to_string())
            }),
        )
    };

    platform().tools().await.unwrap();
    w.merge(&w.key(""), json!({ "expires_at": 0 })).await;
    platform().tools().await.unwrap();

    assert_eq!(*assertions.lock().unwrap(), ["first", "second"]);
}

// spec: mcp/oauth_spec.rb:697 the JWT bearer grant > keeps the assertion from an authorization server the server moves to
#[tokio::test]
async fn keeps_the_assertion_from_an_authorization_server_the_server_moves_to() {
    let w = world().await;
    workload(&w, OAuthSettings::new().assertion("workload-jwt"))
        .tools()
        .await
        .unwrap();
    w.store.delete(&w.key("")).await.unwrap();
    let elsewhere = MockServer::start().await;
    w.moved_to(&elsewhere).await;

    let error = workload(&w, OAuthSettings::new().assertion("workload-jwt"))
        .tools()
        .await
        .err()
        .unwrap();

    match error {
        Error::Unauthorized(message, _) => assert_eq!(
            message,
            format!(
                "The workload is registered with {}, but {} now uses {}",
                w.issuer(),
                w.server_url(),
                elsewhere.uri()
            )
        ),
        other => panic!("expected UnauthorizedError, got {other:?}"),
    }
    assert!(
        elsewhere
            .received_requests()
            .await
            .unwrap()
            .iter()
            .all(|r| r.url.path() != "/token")
    );
}

/// `enterprise-managed authorization`'s `before`: the identity provider publishes OpenID
/// metadata only, and issues an identity assertion grant.
async fn identity_provider(w: &World) -> MockServer {
    let idp = MockServer::start().await;
    let issuer = idp.uri();
    Mock::given(method("GET"))
        .and(path("/.well-known/oauth-authorization-server"))
        .respond_with(ResponseTemplate::new(404))
        .mount(&idp)
        .await;
    Mock::given(method("GET"))
        .and(path("/.well-known/openid-configuration"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "issuer": issuer, "token_endpoint": format!("{issuer}/token"),
            "token_endpoint_auth_methods_supported": ["client_secret_post"]
        })))
        .mount(&idp)
        .await;
    issues_grant(&idp, "urn:ietf:params:oauth:token-type:id-jag", 2).await;
    let _ = w;
    idp
}

async fn issues_grant(idp: &MockServer, kind: &str, priority: u8) {
    Mock::given(method("POST"))
        .and(path("/token"))
        .respond_with(ResponseTemplate::new(200).set_body_json(
            json!({ "issued_token_type": kind, "access_token": "id-jag", "token_type": "N_A" }),
        ))
        .with_priority(priority)
        .mount(idp)
        .await;
}

fn wiki(w: &World, idp: &MockServer, user: &str) -> Mcp {
    let user = user.to_string();
    w.mcp_with(
        OAuthSettings::new()
            .owner(user.clone())
            .client_id("wiki-app")
            .client_secret("wiki-secret")
            .identity_provider(
                IdentityProvider::new()
                    .issuer(idp.uri())
                    .client_id("sso-app")
                    .client_secret("sso-secret")
                    .id_token_with(move || Some(format!("id-token-for-{user}"))),
            ),
    )
}

// spec: mcp/oauth_spec.rb:735 enterprise-managed authorization > exchanges the user's ID token for a grant the authorization server accepts
#[tokio::test]
async fn exchanges_the_users_id_token_for_a_grant_the_authorization_server_accepts() {
    let w = world().await;
    let idp = identity_provider(&w).await;

    assert!(wiki(&w, &idp, "ada").tools().await.unwrap().is_empty());

    let exchanges: Vec<Request> = idp
        .received_requests()
        .await
        .unwrap()
        .into_iter()
        .filter(|r| r.method.as_str() == "POST")
        .collect();
    assert_eq!(exchanges.len(), 1);
    let strings = |pairs: &[(&str, &str)]| -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    };
    assert_eq!(
        form(&exchanges[0]),
        strings(&[
            (
                "grant_type",
                "urn:ietf:params:oauth:grant-type:token-exchange"
            ),
            (
                "requested_token_type",
                "urn:ietf:params:oauth:token-type:id-jag"
            ),
            ("audience", &w.issuer()),
            ("resource", &w.server_url()),
            ("scope", "issues:read"),
            ("subject_token", "id-token-for-ada"),
            (
                "subject_token_type",
                "urn:ietf:params:oauth:token-type:id_token"
            ),
            ("client_id", "sso-app"),
            ("client_secret", "sso-secret"),
        ])
    );
    assert_eq!(
        w.token_form().await,
        strings(&[
            ("grant_type", "urn:ietf:params:oauth:grant-type:jwt-bearer"),
            ("assertion", "id-jag"),
            ("client_id", "wiki-app"),
            ("resource", &w.server_url()),
            ("scope", "issues:read"),
        ])
    );
    let tokens = w.requests(&w.auth, "POST", "/token").await;
    assert_eq!(
        header(&tokens[0], "authorization"),
        Some(basic("wiki-app:wiki-secret"))
    );
}

// spec: mcp/oauth_spec.rb:756 enterprise-managed authorization > declares the enterprise-managed authorization extension with every request
#[tokio::test]
async fn declares_the_enterprise_managed_authorization_extension_with_every_request() {
    let w = world().await;
    let idp = identity_provider(&w).await;

    wiki(&w, &idp, "ada").tools().await.unwrap();

    assert_eq!(
        w.declared_extensions().await,
        [json!({ "io.modelcontextprotocol/enterprise-managed-authorization": {} })]
    );
}

// spec: mcp/oauth_spec.rb:762 enterprise-managed authorization > forwards nothing the identity provider issues but an identity assertion grant
#[tokio::test]
async fn forwards_nothing_the_identity_provider_issues_but_an_identity_assertion_grant() {
    let w = world().await;
    let idp = identity_provider(&w).await;
    issues_grant(&idp, "urn:ietf:params:oauth:token-type:access_token", 1).await;

    let error = wiki(&w, &idp, "ada").tools().await.err().unwrap();

    match error {
        Error::Unauthorized(message, _) => {
            assert_eq!(
                message,
                format!("{} did not issue an identity assertion grant", idp.uri())
            )
        }
        other => panic!("expected UnauthorizedError, got {other:?}"),
    }
    assert!(w.requests(&w.auth, "POST", "/token").await.is_empty());
}

// spec: mcp/oauth_spec.rb:770 enterprise-managed authorization > raises the identity provider's refusal
#[tokio::test]
async fn raises_the_identity_providers_refusal() {
    let w = world().await;
    let idp = identity_provider(&w).await;
    Mock::given(method("POST"))
        .and(path("/token"))
        .respond_with(ResponseTemplate::new(400).set_body_json(
            json!({ "error": "invalid_grant", "error_description": "The ID token expired" }),
        ))
        .with_priority(1)
        .mount(&idp)
        .await;

    let error = wiki(&w, &idp, "ada").tools().await.err().unwrap();

    let host = reqwest::Url::parse(&idp.uri())
        .unwrap()
        .host_str()
        .unwrap()
        .to_string();
    match error {
        Error::Unauthorized(message, _) => assert_eq!(
            message,
            format!("{host} refused the request: The ID token expired")
        ),
        other => panic!("expected UnauthorizedError, got {other:?}"),
    }
}

/// `requires_dpop(nonce:, supplies:, challenge:)`: the server challenges requests without a
/// DPoP-bound token and proof, asks for `nonce` in proofs, and supplies `supplies` with answers.
struct RequiresDpop {
    challenge: String,
    nonce: Option<String>,
    supplies: Option<String>,
}

impl Respond for RequiresDpop {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let proof = header(request, "dpop");
        let bound = header(request, "authorization").is_some_and(|a| a.starts_with("DPoP access-"));
        let Some(proof) = proof.filter(|_| bound) else {
            return ResponseTemplate::new(401)
                .insert_header("WWW-Authenticate", self.challenge.as_str());
        };
        if let Some(nonce) = &self.nonce
            && proved(&proof).1["nonce"].as_str() != Some(nonce.as_str())
        {
            return ResponseTemplate::new(401)
                .insert_header("WWW-Authenticate", r#"DPoP error="use_dpop_nonce""#)
                .insert_header("DPoP-Nonce", nonce.as_str());
        }
        let mut response = ResponseTemplate::new(200).set_body_json(answer(request));
        if let Some(supplies) = &self.supplies {
            response = response.insert_header("DPoP-Nonce", supplies.as_str());
        }
        response
    }
}

/// The DPoP `before`: the token endpoint binds tokens to proofs it receives.
struct DpopTokenEndpoint;

impl Respond for DpopTokenEndpoint {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let refreshing =
            form(request).get("grant_type").map(String::as_str) == Some("refresh_token");
        let token_type = if header(request, "dpop").is_some() {
            "DPoP"
        } else {
            "Bearer"
        };
        ResponseTemplate::new(200).set_body_json(json!({
            "access_token": if refreshing { "access-2" } else { "access-1" }, "refresh_token": "refresh-1",
            "expires_in": 3600, "token_type": token_type
        }))
    }
}

impl World {
    fn metadata_url(&self) -> String {
        format!(
            "{}/.well-known/oauth-protected-resource/mcp",
            self.mcp.uri()
        )
    }

    async fn requires_dpop(
        &self,
        nonce: Option<&str>,
        supplies: Option<&str>,
        challenge: Option<String>,
        priority: u8,
    ) {
        let challenge = challenge
            .unwrap_or_else(|| format!(r#"DPoP resource_metadata="{}""#, self.metadata_url()));
        Mock::given(method("POST"))
            .and(path("/mcp"))
            .respond_with(RequiresDpop {
                challenge,
                nonce: nonce.map(str::to_string),
                supplies: supplies.map(str::to_string),
            })
            .with_priority(priority)
            .mount(&self.mcp)
            .await;
    }

    /// `proofs_sent_to(url)`: the verified proofs of the requests that carried one.
    async fn proofs_sent_to(&self, server: &MockServer, at: &str) -> Vec<(Value, Value)> {
        let requests = self.requests(server, "POST", at).await;
        assert!(!requests.is_empty());
        requests
            .iter()
            .filter_map(|r| header(r, "dpop"))
            .map(|p| proved(&p))
            .collect()
    }
}

/// `servers that require DPoP`'s `before`.
async fn dpop_world() -> World {
    let w = world().await;
    w.requires_dpop(None, None, None, 3).await;
    Mock::given(method("POST"))
        .and(path("/token"))
        .respond_with(DpopTokenEndpoint)
        .with_priority(3)
        .mount(&w.auth)
        .await;
    w
}

fn nonces(proofs: &[(Value, Value)]) -> Vec<Option<String>> {
    proofs
        .iter()
        .map(|(_, claims)| claims["nonce"].as_str().map(str::to_string))
        .collect()
}

// spec: mcp/oauth_spec.rb:843 servers that require DPoP > binds tokens to a key kept with the credentials and proves possession of it on every request
#[tokio::test]
async fn binds_tokens_to_a_key_kept_with_the_credentials_and_proves_possession_on_every_request() {
    let w = dpop_world().await;
    w.authorize(&w.linear("ada")).await;
    w.linear("ada").tools().await.unwrap();

    let token_proofs = w.proofs_sent_to(&w.auth, "/token").await;
    let (token_header, token_claims) = &token_proofs[0];
    assert_eq!(
        (&token_header["typ"], &token_header["alg"]),
        (&json!("dpop+jwt"), &json!("ES256"))
    );
    let mut keys: Vec<&String> = token_header["jwk"].as_object().unwrap().keys().collect();
    keys.sort();
    assert_eq!(keys, ["crv", "kty", "x", "y"]);
    assert_eq!(
        (&token_claims["htm"], &token_claims["htu"]),
        (&json!("POST"), &json!(format!("{}/token", w.issuer())))
    );
    assert!(token_claims.get("ath").is_none() && token_claims.get("nonce").is_none());
    let requests = w.proofs_sent_to(&w.mcp, "/mcp").await;
    assert_eq!(requests.len(), 2);
    assert!(
        requests
            .iter()
            .all(|(header, _)| header["jwk"] == token_header["jwk"])
    );
    let ath = URL_SAFE_NO_PAD.encode(<sha2::Sha256 as sha2::Digest>::digest(b"access-1"));
    for (_, claims) in &requests {
        assert_eq!(
            (&claims["htm"], &claims["htu"], &claims["ath"]),
            (&json!("POST"), &json!(w.server_url()), &json!(ath))
        );
    }
    assert_ne!(requests[0].1["jti"], requests[1].1["jti"]);
    let stored = w.store.read(&w.key("ada")).await.unwrap().unwrap();
    assert_eq!(stored["token_type"], "DPoP");
    assert!(
        stored["dpop_key"]
            .as_str()
            .unwrap()
            .starts_with("-----BEGIN PRIVATE KEY-----")
    );
}

// spec: mcp/oauth_spec.rb:863 servers that require DPoP > refreshes with a proof from the same key
#[tokio::test]
async fn refreshes_with_a_proof_from_the_same_key() {
    let w = dpop_world().await;
    w.authorize(&w.linear("ada")).await;
    w.merge(&w.key("ada"), json!({ "expires_at": 0 })).await;

    w.linear("ada").tools().await.unwrap();

    let proofs = w.proofs_sent_to(&w.auth, "/token").await;
    assert_eq!(proofs.len(), 2);
    assert_eq!(proofs[1].0["jwk"], proofs[0].0["jwk"]);
    let refreshed = w
        .requests(&w.mcp, "POST", "/mcp")
        .await
        .iter()
        .filter(|r| header(r, "authorization").as_deref() == Some("DPoP access-2"))
        .count();
    assert_eq!(refreshed, 2);
}

/// Answers `use_dpop_nonce` once, then a DPoP-bound token.
struct NonceTokenEndpoint {
    requests: AtomicUsize,
}

impl Respond for NonceTokenEndpoint {
    fn respond(&self, _request: &Request) -> ResponseTemplate {
        if self.requests.fetch_add(1, Ordering::SeqCst) == 0 {
            return ResponseTemplate::new(400)
                .insert_header("DPoP-Nonce", "as-nonce")
                .set_body_json(json!({ "error": "use_dpop_nonce" }));
        }
        ResponseTemplate::new(200).set_body_json(
            json!({ "access_token": "access-1", "token_type": "DPoP", "expires_in": 3600 }),
        )
    }
}

// spec: mcp/oauth_spec.rb:875 servers that require DPoP > retries a token request with the authorization server's nonce
#[tokio::test]
async fn retries_a_token_request_with_the_authorization_servers_nonce() {
    let w = dpop_world().await;
    Mock::given(method("POST"))
        .and(path("/token"))
        .respond_with(NonceTokenEndpoint {
            requests: AtomicUsize::new(0),
        })
        .with_priority(1)
        .mount(&w.auth)
        .await;
    let linear = w.linear("ada");

    w.authorize(&linear).await;

    assert_eq!(
        nonces(&w.proofs_sent_to(&w.auth, "/token").await),
        [None, Some("as-nonce".to_string())]
    );
    assert!(linear.is_authorized().await.unwrap());
}

// spec: mcp/oauth_spec.rb:891 servers that require DPoP > retries a request with the server's nonce
#[tokio::test]
async fn retries_a_request_with_the_servers_nonce() {
    let w = dpop_world().await;
    w.authorize(&w.linear("ada")).await;
    w.requires_dpop(Some("rs-nonce"), None, None, 1).await;

    assert!(w.linear("ada").tools().await.unwrap().is_empty());

    let nonce = Some("rs-nonce".to_string());
    assert_eq!(
        nonces(&w.proofs_sent_to(&w.mcp, "/mcp").await),
        [None, nonce.clone(), nonce]
    );
}

/// `keeps_sessions`: a server that predates 2026-07-28, keeps a session, and requires DPoP.
struct KeepsSessions {
    metadata_url: String,
}

impl Respond for KeepsSessions {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        if !header(request, "authorization").is_some_and(|a| a.starts_with("DPoP access-")) {
            return ResponseTemplate::new(401).insert_header(
                "WWW-Authenticate",
                format!(r#"DPoP resource_metadata="{}""#, self.metadata_url).as_str(),
            );
        }
        let message: Value = serde_json::from_slice(&request.body).unwrap_or(Value::Null);
        if message["method"] == "server/discover" {
            return ResponseTemplate::new(404);
        }
        if message.get("id").is_none() {
            return ResponseTemplate::new(202);
        }
        let result = if message["method"] == "initialize" {
            json!({ "protocolVersion": "2025-06-18", "capabilities": { "tools": { "listChanged": true } } })
        } else {
            json!({ "tools": [] })
        };
        ResponseTemplate::new(200)
            .insert_header("Mcp-Session-Id", "session-1")
            .set_body_json(json!({ "jsonrpc": "2.0", "id": message["id"], "result": result }))
    }
}

// spec: mcp/oauth_spec.rb:900 servers that require DPoP > proves possession with the HTTP method of each request
// Ruby's `mcp.listen` starts the listener, whose GET opens the session's event stream; here the
// client's `listen` makes that GET (answered 405, so it returns at once) and `close` the DELETE.
#[tokio::test]
async fn proves_possession_with_the_http_method_of_each_request() {
    let w = dpop_world().await;
    Mock::given(method("POST"))
        .and(path("/mcp"))
        .respond_with(KeepsSessions {
            metadata_url: w.metadata_url(),
        })
        .with_priority(1)
        .mount(&w.mcp)
        .await;
    w.authorize(&w.linear("ada")).await;
    w.stub(&w.mcp, "GET", "/mcp", ResponseTemplate::new(405))
        .await;
    w.stub(&w.mcp, "DELETE", "/mcp", ResponseTemplate::new(204))
        .await;

    let mcp = w.linear("ada");
    let _ = mcp
        .client()
        .listen(&json!({ "toolsListChanged": true }), &mut |_| {})
        .await;
    mcp.close().await;

    for (_, claims) in w.proofs_sent_to(&w.mcp, "/mcp").await {
        assert_eq!(claims["htm"], "POST");
    }
    let stream = w.requests(&w.mcp, "GET", "/mcp").await;
    let (_, claims) = proved(&header(&stream[0], "dpop").unwrap());
    assert_eq!(
        (&claims["htm"], &claims["htu"]),
        (&json!("GET"), &json!(w.server_url()))
    );
    let ended = w.requests(&w.mcp, "DELETE", "/mcp").await;
    assert_eq!(ended.len(), 1);
    assert_eq!(
        proved(&header(&ended[0], "dpop").unwrap()).1["htm"],
        "DELETE"
    );
}

// spec: mcp/oauth_spec.rb:920 servers that require DPoP > signs the next proof with the nonce the server sends with a response
#[tokio::test]
async fn signs_the_next_proof_with_the_nonce_the_server_sends_with_a_response() {
    let w = dpop_world().await;
    w.authorize(&w.linear("ada")).await;
    w.requires_dpop(None, Some("rs-nonce"), None, 1).await;

    w.linear("ada").tools().await.unwrap();

    assert_eq!(
        nonces(&w.proofs_sent_to(&w.mcp, "/mcp").await),
        [None, Some("rs-nonce".to_string())]
    );
}

// spec: mcp/oauth_spec.rb:929 servers that require DPoP > gets a token for the app's first request, then retries it with the server's nonce
#[tokio::test]
async fn gets_a_token_for_the_apps_first_request_then_retries_it_with_the_servers_nonce() {
    let w = dpop_world().await;
    w.requires_dpop(Some("rs-nonce"), None, None, 1).await;

    let reports = reports(&w, OAuthSettings::new().client_secret("shh"));
    assert!(reports.tools().await.unwrap().is_empty());

    let nonce = Some("rs-nonce".to_string());
    assert_eq!(
        nonces(&w.proofs_sent_to(&w.mcp, "/mcp").await),
        [None, nonce.clone(), nonce]
    );
}

// spec: mcp/oauth_spec.rb:942 servers that require DPoP > binds tokens the app requests for itself when the metadata requires it
#[tokio::test]
async fn binds_tokens_the_app_requests_for_itself_when_the_metadata_requires_it() {
    let w = dpop_world().await;
    w.requires_dpop(
        None,
        None,
        Some(format!(
            r#"Bearer resource_metadata="{}""#,
            w.metadata_url()
        )),
        1,
    )
    .await;
    let metadata = json!({
        "resource": w.server_url(), "authorization_servers": [w.issuer()], "dpop_bound_access_tokens_required": true
    });
    w.stub(
        &w.mcp,
        "GET",
        "/.well-known/oauth-protected-resource/mcp",
        ResponseTemplate::new(200).set_body_json(metadata),
    )
    .await;

    let reports = reports(&w, OAuthSettings::new().client_secret("shh"));
    assert!(reports.tools().await.unwrap().is_empty());

    assert_eq!(w.proofs_sent_to(&w.auth, "/token").await.len(), 1);
    assert_eq!(w.proofs_sent_to(&w.mcp, "/mcp").await.len(), 2);
}

// spec: mcp/oauth_spec.rb:960 servers that require DPoP > keeps bearer tokens for servers that also accept them
#[tokio::test]
async fn keeps_bearer_tokens_for_servers_that_also_accept_them() {
    let w = dpop_world().await;
    let challenge = format!(
        r#"Bearer resource_metadata="{}", DPoP algs="ES256""#,
        w.metadata_url()
    );
    w.stub(
        &w.mcp,
        "POST",
        "/mcp",
        ResponseTemplate::new(401).insert_header("WWW-Authenticate", challenge.as_str()),
    )
    .await;

    w.authorize(&w.linear("ada")).await;

    let tokens = w.requests(&w.auth, "POST", "/token").await;
    assert!(tokens.iter().all(|r| header(r, "dpop").is_none()));
}

// spec: mcp/oauth_spec.rb:970 refuses grants it does not know
#[test]
fn refuses_grants_it_does_not_know() {
    let error = "password".parse::<Grant>().err().unwrap();
    assert!(
        matches!(&error, Error::Argument(m) if m == "Unknown OAuth grant: password"),
        "{error:?}"
    );
}
