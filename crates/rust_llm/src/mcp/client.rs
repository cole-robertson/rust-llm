//! Port of `lib/ruby_llm/mcp/client.rb`: speaks JSON-RPC to one MCP server over a transport.
//! It speaks the 2026-07-28 revision and falls back to the `initialize` handshake for servers
//! that predate it, declaring no client capabilities so those servers never call back.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{Map, Value, json};

use super::{McpError, OnNotification, Transport};
use crate::error::{Error, Result};

pub const VERSION: &str = "2026-07-28";
pub const LEGACY_VERSION: &str = "2025-11-25";
const MODERN_ERRORS: &[i64] = &[-32_020, -32_021, -32_022];
const DISCOVERY_TIMEOUT: Duration = Duration::from_secs(10);

/// `RubyLLM::MCP::Client`.
pub struct Client {
    transport: Arc<dyn Transport>,
    capabilities: Value,
    server: tokio::sync::Mutex<Option<Value>>,
    version: Mutex<Option<String>>,
}

impl Client {
    pub fn new(transport: Arc<dyn Transport>, capabilities: Value) -> Client {
        Client { transport, capabilities, server: tokio::sync::Mutex::new(None), version: Mutex::new(None) }
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

    /// `request(method, params, headers:) { |notification| }`.
    pub async fn request(
        &self,
        method: &str,
        params: Value,
        headers: &[(String, String)],
        on_notification: &mut OnNotification<'_>,
    ) -> Result<Value> {
        self.server().await?;
        self.call(method, params, None, headers, on_notification).await
    }

    /// `list(method, key)`: every item across `nextCursor` pages.
    pub async fn list(&self, method: &str, key: &str) -> Result<Vec<Value>> {
        let items_of = |page: &Value| page.get(key).and_then(Value::as_array).cloned().unwrap_or_default();
        let mut page = self.request(method, json!({}), &[], &mut |_| {}).await?;
        let mut items = items_of(&page);
        while let Some(cursor) = page.get("nextCursor").filter(|c| !c.is_null()).cloned() {
            page = self.request(method, json!({ "cursor": cursor }), &[], &mut |_| {}).await?;
            items.extend(items_of(&page));
        }
        Ok(items)
    }

    /// `close`: releases the transport; the next request connects again.
    pub async fn close(&self) {
        let mut server = self.server.lock().await;
        self.transport.close().await;
        *server = None;
        self.set_version(None);
    }

    async fn discover(&self) -> Result<Option<Value>> {
        self.set_version(Some(VERSION.into()));
        match self.call("server/discover", json!({}), Some(DISCOVERY_TIMEOUT), &[], &mut |_| {}).await {
            Ok(result) => {
                let supported = result.get("supportedVersions").and_then(Value::as_array);
                Ok(supported.is_some_and(|v| v.iter().any(|v| v == VERSION)).then_some(result))
            }
            Err(Error::Mcp(e)) if e.code.is_some_and(|c| MODERN_ERRORS.contains(&c)) => Err(Error::Mcp(e)),
            Err(Error::Mcp(_)) => Ok(None),
            Err(e) => Err(e),
        }
    }

    async fn handshake(&self) -> Result<Value> {
        self.set_version(None);
        let params = json!({ "protocolVersion": LEGACY_VERSION, "capabilities": {}, "clientInfo": client_info() });
        let result = self.call("initialize", params, None, &[], &mut |_| {}).await?;
        self.set_version(result.get("protocolVersion").and_then(Value::as_str).map(str::to_string));
        let version = self.version();
        self.transport.notify(&self.message("notifications/initialized", json!({}), None), version.as_deref()).await?;
        Ok(result)
    }

    async fn call(
        &self,
        method: &str,
        params: Value,
        timeout: Option<Duration>,
        headers: &[(String, String)],
        on_notification: &mut OnNotification<'_>,
    ) -> Result<Value> {
        let id = uuid::Uuid::new_v4().to_string();
        let request = self.message(method, params, Some(&id));
        let version = self.version();
        let response = match self.transport.request(&request, version.as_deref(), timeout, headers, on_notification).await {
            Err(Error::Cancelled) => {
                let cancelled = self.message("notifications/cancelled", json!({ "requestId": id }), None);
                self.transport.cancel(&cancelled, version.as_deref()).await?;
                return Err(Error::Cancelled);
            }
            other => other?,
        };
        if let Some(error) = response.get("error").filter(|e| !e.is_null()) {
            return Err(McpError {
                message: error.get("message").and_then(Value::as_str).unwrap_or("").to_string(),
                code: error.get("code").and_then(Value::as_i64),
                data: error.get("data").cloned(),
                response: None,
            }
            .into());
        }
        Ok(response.get("result").cloned().unwrap_or(Value::Null))
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
        meta.insert("io.modelcontextprotocol/protocolVersion".into(), VERSION.into());
        meta.insert("io.modelcontextprotocol/clientInfo".into(), client_info());
        meta.insert("io.modelcontextprotocol/clientCapabilities".into(), self.capabilities.clone());
        meta
    }
}

/// `client_info`. RubyLLM sends `ruby_llm`; the port identifies itself as `rust_llm`.
pub fn client_info() -> Value {
    json!({ "name": "rust_llm", "version": crate::VERSION })
}
