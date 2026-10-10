//! Port of `lib/ruby_llm/mcp/client.rb`: speaks JSON-RPC to one MCP server over a transport.
//! It speaks the 2026-07-28 revision and falls back to the `initialize` handshake for servers
//! that predate it, declaring only its extensions so those servers never call back. A server
//! that rejects 2026-07-28 while listing it gets the request once more; one that answers the
//! handshake with a revision missing from [`LEGACY_VERSIONS`] is disconnected.
//!
//! Servers that predate 2026-07-28 may announce that their tools, prompts, or resources changed
//! on the stream of any request. Those notifications reach the [`Client::on_change`] callback
//! once the request is answered, so the callback can make requests of its own.
//! [`Client::listen`] opens a subscription for them instead: `subscriptions/listen` on newer
//! servers, `resources/subscribe` and the session's own stream on older ones, which never
//! announce the status of tasks. Their subscription counts as acknowledged once a ping confirms
//! the session, starting a new one if it ended.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use futures::future::BoxFuture;
use serde_json::{Map, Value, json};

use super::listener::Subscribe;
use super::{McpError, OnNotification, Transport};
use crate::error::{Error, Result};

pub const VERSION: &str = "2026-07-28";
/// The revisions before 2026-07-28 the client speaks, newest first.
pub const LEGACY_VERSIONS: [&str; 4] = ["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"];
/// The revision the `initialize` handshake asks for.
pub const LEGACY_VERSION: &str = LEGACY_VERSIONS[0];
const UNSUPPORTED_VERSION: i64 = -32_022;
const MODERN_ERRORS: &[i64] = &[-32_020, -32_021, UNSUPPORTED_VERSION];
pub const METHOD_NOT_FOUND: i64 = -32_601;
const DISCOVERY_TIMEOUT: Duration = Duration::from_secs(10);
/// `ACKNOWLEDGED`: the notification that starts a subscription.
pub const ACKNOWLEDGED: &str = "notifications/subscriptions/acknowledged";
/// `CHANGES`: the notifications a server announces its changes with.
pub const CHANGES: [&str; 4] = [
    "notifications/tools/list_changed",
    "notifications/prompts/list_changed",
    "notifications/resources/list_changed",
    "notifications/resources/updated",
];

/// Receives the changes a server announces on the stream of a request.
pub type ChangeHandler = Arc<dyn Fn(Value) -> BoxFuture<'static, ()> + Send + Sync>;

/// `RubyLLM::MCP::Client`.
pub struct Client {
    transport: Arc<dyn Transport>,
    capabilities: Value,
    server: tokio::sync::Mutex<Option<Value>>,
    version: Mutex<Option<String>>,
    on_change: Mutex<Option<ChangeHandler>>,
    subscribed: Mutex<Vec<String>>,
}

impl Client {
    pub fn new(transport: Arc<dyn Transport>, capabilities: Value) -> Client {
        Client {
            transport,
            capabilities,
            server: tokio::sync::Mutex::new(None),
            version: Mutex::new(None),
            on_change: Mutex::new(None),
            subscribed: Mutex::new(Vec::new()),
        }
    }

    /// `Client.reply`: the answer to a request the server sends. RustLLM only answers pings.
    pub fn reply(request: &Value) -> Value {
        let id = request.get("id").cloned().unwrap_or(Value::Null);
        if request.get("method").and_then(Value::as_str) == Some("ping") {
            json!({ "jsonrpc": "2.0", "id": id, "result": {} })
        } else {
            json!({ "jsonrpc": "2.0", "id": id, "error": { "code": METHOD_NOT_FOUND, "message": "Method not found" } })
        }
    }

    /// The block given to `Client.new`: receives each change a server announces on the stream of
    /// a request, once the request is answered.
    pub fn on_change<F, Fut>(&self, callback: F)
    where
        F: Fn(Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        let handler: ChangeHandler = Arc::new(move |change| Box::pin(callback(change)));
        if let Ok(mut slot) = self.on_change.lock() {
            *slot = Some(handler);
        }
    }

    /// The protocol version the server agreed to, once connected.
    pub fn version(&self) -> Option<String> {
        self.version.lock().ok().and_then(|v| v.clone())
    }

    /// `modern?`: whether the server speaks 2026-07-28.
    pub fn is_modern(&self) -> bool {
        self.version().as_deref() == Some(VERSION)
    }

    fn set_version(&self, version: Option<String>) {
        if let Ok(mut v) = self.version.lock() {
            *v = version;
        }
    }

    /// `server`: what the server said about itself, discovering or shaking hands on first use.
    pub async fn server(&self) -> Result<Value> {
        let mut server = self.server.lock().await;
        if let Some(s) = server.as_ref() {
            return Ok(s.clone());
        }
        let discovered = match self.discover().await? {
            Some(result) => result,
            None => self.handshake().await?,
        };
        *server = Some(discovered.clone());
        Ok(discovered)
    }

    /// `request(method, params, headers:) { |notification| }`. A server whose session ended gets
    /// a new session and the request once more.
    pub async fn request(
        &self,
        method: &str,
        params: Value,
        headers: &[(String, String)],
        on_notification: &mut OnNotification<'_>,
    ) -> Result<Value> {
        self.server().await?;
        match self
            .call(method, params.clone(), None, headers, on_notification)
            .await
        {
            Err(Error::Mcp(e)) if e.session_expired => {
                self.renew_session().await?;
                self.call(method, params, None, headers, on_notification)
                    .await
            }
            other => other,
        }
    }

    /// `list(method, key)`: every item across `nextCursor` pages.
    pub async fn list(&self, method: &str, key: &str) -> Result<Vec<Value>> {
        let items_of = |page: &Value| {
            page.get(key)
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default()
        };
        let mut page = self.request(method, json!({}), &[], &mut |_| {}).await?;
        let mut items = items_of(&page);
        while let Some(cursor) = page.get("nextCursor").filter(|c| !c.is_null()).cloned() {
            page = self
                .request(method, json!({ "cursor": cursor }), &[], &mut |_| {})
                .await?;
            items.extend(items_of(&page));
        }
        Ok(items)
    }

    /// `listen(changes) { |notification| }`: subscribes to `changes`, a `subscriptions/listen`
    /// filter, and passes each notification of the subscription to `on_notification`, starting
    /// with the server's acknowledgment. Returns when the server ends the subscription and fails
    /// when it refuses one or the stream breaks.
    pub async fn listen(
        &self,
        changes: &Value,
        on_notification: &mut OnNotification<'_>,
    ) -> Result<()> {
        self.server().await?;
        if self.is_modern() {
            self.subscribe(changes, on_notification).await
        } else {
            self.listen_to_session(changes, on_notification).await
        }
    }

    /// `close`: releases the transport; the next request connects again.
    pub async fn close(&self) {
        let mut server = self.server.lock().await;
        self.transport.close().await;
        *server = None;
        self.set_version(None);
        self.set_subscribed(Vec::new());
    }

    async fn renew_session(&self) -> Result<()> {
        let mut server = self.server.lock().await;
        *server = Some(self.handshake().await?);
        Ok(())
    }

    fn set_subscribed(&self, uris: Vec<String>) {
        if let Ok(mut subscribed) = self.subscribed.lock() {
            *subscribed = uris;
        }
    }

    async fn discover(&self) -> Result<Option<Value>> {
        self.set_version(Some(VERSION.into()));
        match self
            .call(
                "server/discover",
                json!({}),
                Some(DISCOVERY_TIMEOUT),
                &[],
                &mut |_| {},
            )
            .await
        {
            Ok(result) => {
                let supported = result.get("supportedVersions").and_then(Value::as_array);
                Ok(supported
                    .is_some_and(|v| v.iter().any(|v| v == VERSION))
                    .then_some(result))
            }
            Err(Error::Mcp(e)) if e.code.is_some_and(|c| MODERN_ERRORS.contains(&c)) => {
                Err(Error::Mcp(e))
            }
            Err(Error::Mcp(_)) => Ok(None),
            Err(e) => Err(e),
        }
    }

    async fn handshake(&self) -> Result<Value> {
        self.set_version(None);
        let mut capabilities = Map::new();
        if let Some(extensions) = self.capabilities.get("extensions") {
            capabilities.insert("extensions".into(), extensions.clone());
        }
        let params = json!({ "protocolVersion": LEGACY_VERSION, "capabilities": capabilities, "clientInfo": client_info() });
        let result = self
            .call("initialize", params, None, &[], &mut |_| {})
            .await?;
        let version = result
            .get("protocolVersion")
            .and_then(Value::as_str)
            .map(str::to_string);
        let Some(version) = version.filter(|v| LEGACY_VERSIONS.contains(&v.as_str())) else {
            self.transport.close().await;
            let answered = result
                .get("protocolVersion")
                .map(|v| v.as_str().map_or_else(|| v.to_string(), str::to_string))
                .unwrap_or_default();
            return Err(McpError::new(format!(
                "The server answered with protocol version {answered}, which RustLLM does not speak"
            ))
            .into());
        };
        self.set_version(Some(version.clone()));
        self.set_subscribed(Vec::new());
        self.transport
            .notify(
                &self.message("notifications/initialized", json!({}), None),
                Some(&version),
            )
            .await?;
        Ok(result)
    }

    /// `call(method, params, timeout:, headers:, retried:)`: a modern server that rejects
    /// 2026-07-28 while listing it gets the request once more.
    async fn call(
        &self,
        method: &str,
        params: Value,
        timeout: Option<Duration>,
        headers: &[(String, String)],
        on_notification: &mut OnNotification<'_>,
    ) -> Result<Value> {
        match self
            .call_once(method, params.clone(), timeout, headers, on_notification)
            .await
        {
            Err(Error::Mcp(e)) if self.offers_version(&e) => {
                self.call_once(method, params, timeout, headers, on_notification)
                    .await
            }
            other => other,
        }
    }

    async fn call_once(
        &self,
        method: &str,
        params: Value,
        timeout: Option<Duration>,
        headers: &[(String, String)],
        on_notification: &mut OnNotification<'_>,
    ) -> Result<Value> {
        let id = uuid::Uuid::new_v4().to_string();
        let request = self.message(method, params, Some(&id));
        let response = self
            .exchange(&request, timeout, headers, on_notification)
            .await?;
        answer(&response)
    }

    /// `exchange`: announced changes reach `on_change` once the request is answered (or fails).
    async fn exchange(
        &self,
        request: &Value,
        timeout: Option<Duration>,
        headers: &[(String, String)],
        on_notification: &mut OnNotification<'_>,
    ) -> Result<Value> {
        let version = self.version();
        let mut changes = Vec::new();
        let result = self
            .transport
            .request(
                request,
                version.as_deref(),
                timeout,
                headers,
                &mut |notification: &Value| {
                    let method = notification.get("method").and_then(Value::as_str);
                    if method.is_some_and(|m| CHANGES.contains(&m)) {
                        changes.push(notification.clone());
                    } else {
                        on_notification(notification);
                    }
                },
            )
            .await;
        let result = match result {
            Err(Error::Cancelled) => {
                self.cancel(request).await?;
                Err(Error::Cancelled)
            }
            other => other,
        };
        let handler = self.on_change.lock().ok().and_then(|h| h.clone());
        if let Some(handler) = handler {
            for change in changes {
                handler(change).await;
            }
        }
        result
    }

    async fn subscribe(
        &self,
        changes: &Value,
        on_notification: &mut OnNotification<'_>,
    ) -> Result<()> {
        let id = uuid::Uuid::new_v4().to_string();
        let request = self.message(
            "subscriptions/listen",
            json!({ "notifications": changes }),
            Some(&id),
        );
        let version = self.version();
        match self
            .transport
            .listen(Some(&request), version.as_deref(), on_notification)
            .await
        {
            Err(Error::Cancelled) => {
                self.cancel(&request).await?;
                Err(Error::Cancelled)
            }
            Err(e) => Err(e),
            Ok(ending) => answer(&ending.unwrap_or(Value::Null)).map(|_| ()),
        }
    }

    async fn listen_to_session(
        &self,
        changes: &Value,
        on_notification: &mut OnNotification<'_>,
    ) -> Result<()> {
        match self.session_stream(changes, on_notification).await {
            Err(Error::Mcp(e)) if e.session_expired => {
                self.renew_session().await?;
                Err(Error::Mcp(e))
            }
            other => other,
        }
    }

    async fn session_stream(
        &self,
        changes: &Value,
        on_notification: &mut OnNotification<'_>,
    ) -> Result<()> {
        let uris: Vec<String> = changes
            .get("resourceSubscriptions")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|u| u.as_str().map(str::to_string))
            .collect();
        let subscribed = self
            .subscribed
            .lock()
            .map(|s| s.clone())
            .unwrap_or_default();
        for uri in subscribed.iter().filter(|u| !uris.contains(u)) {
            self.request(
                "resources/unsubscribe",
                json!({ "uri": uri }),
                &[],
                &mut |_| {},
            )
            .await?;
        }
        for uri in &uris {
            self.request(
                "resources/subscribe",
                json!({ "uri": uri }),
                &[],
                &mut |_| {},
            )
            .await?;
        }
        self.set_subscribed(uris);
        self.request("ping", json!({}), &[], &mut |_| {}).await?;
        let mut listened = changes.as_object().cloned().unwrap_or_default();
        listened.remove("taskIds");
        on_notification(
            &json!({ "method": ACKNOWLEDGED, "params": { "notifications": listened } }),
        );
        let version = self.version();
        self.transport
            .listen(None, version.as_deref(), on_notification)
            .await
            .map(|_| ())
    }

    async fn cancel(&self, request: &Value) -> Result<()> {
        let id = request.get("id").cloned().unwrap_or(Value::Null);
        let cancelled = self.message("notifications/cancelled", json!({ "requestId": id }), None);
        let version = self.version();
        self.transport.cancel(&cancelled, version.as_deref()).await
    }

    /// `offers_version?`: a modern server rejected 2026-07-28 while listing it as supported.
    fn offers_version(&self, error: &McpError) -> bool {
        let supported = error.data.as_ref().and_then(|d| d.get("supported"));
        let listed = match supported {
            Some(Value::Array(versions)) => versions.iter().any(|v| v == VERSION),
            Some(version) => version == VERSION,
            None => false,
        };
        self.is_modern() && error.code == Some(UNSUPPORTED_VERSION) && listed
    }

    fn message(&self, method: &str, params: Value, id: Option<&str>) -> Value {
        let mut params = match params {
            Value::Object(map) => map,
            _ => Map::new(),
        };
        if self.is_modern() {
            let mut meta = self.meta();
            if let Some(Value::Object(extra)) = params.get("_meta") {
                meta.extend(extra.clone());
            }
            params.insert("_meta".into(), Value::Object(meta));
        }
        let mut message = json!({ "jsonrpc": "2.0" });
        if let Some(id) = id {
            message["id"] = id.into();
        }
        message["method"] = method.into();
        message["params"] = Value::Object(params);
        message
    }

    fn meta(&self) -> Map<String, Value> {
        let mut meta = Map::new();
        meta.insert(
            "io.modelcontextprotocol/protocolVersion".into(),
            VERSION.into(),
        );
        meta.insert("io.modelcontextprotocol/clientInfo".into(), client_info());
        meta.insert(
            "io.modelcontextprotocol/clientCapabilities".into(),
            self.capabilities.clone(),
        );
        meta
    }
}

#[async_trait]
impl Subscribe for Client {
    async fn listen(
        &self,
        changes: &Value,
        on_notification: &mut OnNotification<'_>,
    ) -> Result<()> {
        Client::listen(self, changes, on_notification).await
    }
}

/// `answer(response)`: the result, or the JSON-RPC error as `Error::Mcp`.
fn answer(response: &Value) -> Result<Value> {
    if let Some(error) = response.get("error").filter(|e| !e.is_null()) {
        return Err(McpError {
            message: error
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            code: error.get("code").and_then(Value::as_i64),
            data: error.get("data").cloned(),
            ..Default::default()
        }
        .into());
    }
    Ok(response.get("result").cloned().unwrap_or(Value::Null))
}

/// `client_info`. RubyLLM sends `ruby_llm`; the port identifies itself as `rust_llm`.
pub fn client_info() -> Value {
    json!({ "name": "rust_llm", "version": crate::VERSION })
}
