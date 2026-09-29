//! Port of `lib/ruby_llm/mcp/http.rb`: Streamable HTTP. Every message is its own POST, answered
//! with a JSON body or with an event stream that carries the request's notifications before its
//! response. Plain HTTP is only allowed on loopback addresses, URLs cannot carry credentials, and
//! redirects are never followed. A cancelled chat closes the stream, which is how 2026-07-28
//! cancels a request.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use base64::Engine;
use futures::StreamExt;
use serde_json::Value;

use super::{McpError, OnNotification, Transport, client};
use crate::error::{Error, ErrorResponse, Result};
use crate::transport::SseParser;

const LOOPBACK_HOSTS: &[&str] = &["localhost", "127.0.0.1", "[::1]", "::1"];

/// Headers sent with every request, resolved per request (`headers: -> { ... }`).
pub type HeaderSource = Arc<dyn Fn() -> Vec<(String, String)> + Send + Sync>;

/// `RubyLLM::MCP::HTTP`.
pub struct Http {
    url: reqwest::Url,
    headers: HeaderSource,
    client: reqwest::Client,
    session: Mutex<Option<String>>,
}

impl Http {
    /// `HTTP.secure?`: HTTPS, or plain HTTP to a loopback address, and no credentials in the URL.
    pub fn is_secure(url: &str) -> bool {
        let Ok(url) = reqwest::Url::parse(url) else { return false };
        if !url.username().is_empty() || url.password().is_some() {
            return false;
        }
        url.scheme() == "https" || (url.scheme() == "http" && url.host_str().is_some_and(|h| LOOPBACK_HOSTS.contains(&h)))
    }

    /// Raises `Error::Argument` for insecure URLs, like `HTTP.new`.
    pub fn new(url: &str, headers: HeaderSource, timeout: Duration) -> Result<Http> {
        if !Http::is_secure(url) {
            return Err(Error::Argument(format!("MCP servers must use HTTPS without credentials in the URL: {url}")));
        }
        let url = reqwest::Url::parse(url).map_err(|e| Error::Argument(e.to_string()))?;
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|e| Error::Configuration(e.to_string()))?;
        Ok(Http { url, headers, client, session: Mutex::new(None) })
    }

    fn host(&self) -> String {
        self.url.host_str().unwrap_or("").to_string()
    }

    fn session(&self) -> Option<String> {
        self.session.lock().ok().and_then(|s| s.clone())
    }

    fn set_session(&self, session: Option<String>) {
        if let Ok(mut s) = self.session.lock() {
            *s = session;
        }
    }

    async fn post(
        &self,
        message: &Value,
        version: Option<&str>,
        timeout: Option<Duration>,
        params: &[(String, String)],
        on_notification: &mut OnNotification<'_>,
    ) -> Result<Vec<Value>> {
        let method = message.get("method").and_then(Value::as_str).unwrap_or("");
        let mut request = self
            .client
            .post(self.url.clone())
            .header("Content-Type", "application/json")
            .header("Accept", "application/json, text/event-stream")
            .header("Mcp-Method", method)
            .body(message.to_string());
        if let Some(version) = version {
            request = request.header("MCP-Protocol-Version", version);
        }
        if let Some(session) = self.session() {
            request = request.header("Mcp-Session-Id", session);
        }
        let name = message.pointer("/params/name").or_else(|| message.pointer("/params/uri"));
        if let Some(name) = name.and_then(|n| header_value(&text_of(n))) {
            request = request.header("Mcp-Name", name);
        }
        for (key, value) in (self.headers)() {
            request = request.header(key, value);
        }
        for (key, value) in params {
            if let Some(value) = header_value(value) {
                request = request.header(format!("Mcp-Param-{key}"), value);
            }
        }
        if let Some(timeout) = timeout {
            request = request.timeout(timeout);
        }
        let response = request.send().await.map_err(|e| {
            if e.is_timeout() { Error::Timeout(e.to_string()) } else { Error::ConnectionFailed(e.to_string()) }
        })?;
        let status = response.status().as_u16();
        if method == "initialize" {
            let session = response.headers().get("mcp-session-id").and_then(|v| v.to_str().ok()).map(str::to_string);
            self.set_session(session);
        }
        let mut stream = Stream::default();
        let mut body = response.bytes_stream();
        while let Some(chunk) = body.next().await {
            if crate::progress::is_cancelled() {
                return Err(Error::Cancelled);
            }
            let chunk = chunk.map_err(|e| Error::ConnectionFailed(e.to_string()))?;
            stream.feed(&String::from_utf8_lossy(&chunk), on_notification);
        }
        let replies = stream.replies(on_notification);
        if (200..300).contains(&status) || answered(&replies, message) {
            return Ok(replies);
        }
        Err(self.failure(status, &stream.body, &replies))
    }

    fn failure(&self, status: u16, body: &str, replies: &[Value]) -> Error {
        let response = Some(ErrorResponse { status, body: body.to_string() });
        match status {
            401 => Error::Unauthorized(format!("{} requires authorization", self.host()), response),
            403 => Error::Forbidden(format!("{} refused the request", self.host()), response),
            _ => {
                let error = replies.first().and_then(|r| r.get("error")).cloned().unwrap_or(Value::Null);
                McpError {
                    message: error
                        .get("message")
                        .and_then(Value::as_str)
                        .map(str::to_string)
                        .unwrap_or_else(|| format!("{} answered HTTP {status}", self.host())),
                    code: error.get("code").and_then(Value::as_i64),
                    data: error.get("data").cloned(),
                    response,
                }
                .into()
            }
        }
    }
}

/// Some servers, such as Google's Drive preview, send a complete JSON-RPC result with an error
/// status. The result is the answer.
fn answered(replies: &[Value], message: &Value) -> bool {
    replies.iter().any(|r| r.get("id") == message.get("id") && r.get("result").is_some())
}

fn text_of(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// `header_value`: printable ASCII passes through; anything else is sent as `=?base64?...?=`.
fn header_value(value: &str) -> Option<String> {
    let safe = value.is_empty()
        || (value.bytes().all(|b| (0x20..=0x7E).contains(&b))
            && !value.starts_with(' ')
            && !value.ends_with(' ')
            && !(value.starts_with("=?base64?") && value.ends_with("?=")));
    Some(if safe { value.to_string() } else { format!("=?base64?{}?=", base64::engine::general_purpose::STANDARD.encode(value)) })
}

/// Collects the JSON-RPC messages of one response, yielding notifications as they arrive. The
/// body is either a single JSON value or a server-sent event stream; the first character tells
/// them apart.
#[derive(Default)]
struct Stream {
    parser: SseParser,
    body: String,
    events: Option<bool>,
    replies: Vec<Value>,
}

impl Stream {
    fn feed(&mut self, chunk: &str, on_notification: &mut OnNotification<'_>) {
        self.body.push_str(chunk);
        if self.events == Some(false) {
            return;
        }
        let chunk = if self.events.is_none() {
            let trimmed = self.body.trim_start();
            if trimmed.is_empty() {
                return;
            }
            self.events = Some(!(trimmed.starts_with('{') || trimmed.starts_with('[')));
            if self.events == Some(false) {
                return;
            }
            self.body.clone()
        } else {
            chunk.to_string()
        };
        for event in self.parser.feed(&chunk) {
            self.receive(&event.data, on_notification);
        }
    }

    fn replies(&mut self, on_notification: &mut OnNotification<'_>) -> Vec<Value> {
        if self.events == Some(true) {
            for event in self.parser.finish() {
                self.receive(&event.data, on_notification);
            }
        } else if self.events == Some(false) && self.replies.is_empty() {
            let body = self.body.clone();
            self.receive(&body, on_notification);
        }
        self.replies.clone()
    }

    fn receive(&mut self, data: &str, on_notification: &mut OnNotification<'_>) {
        let Ok(parsed) = serde_json::from_str::<Value>(data) else {
            tracing::debug!("MCP server sent a message that is not JSON");
            return;
        };
        let replies = match parsed {
            Value::Array(items) => items,
            other => vec![other],
        };
        for reply in replies {
            if reply.get("method").is_some() && reply.get("id").is_none() {
                on_notification(&reply);
            }
            self.replies.push(reply);
        }
    }
}

#[async_trait]
impl Transport for Http {
    async fn request(
        &self,
        message: &Value,
        version: Option<&str>,
        timeout: Option<Duration>,
        headers: &[(String, String)],
        on_notification: &mut OnNotification<'_>,
    ) -> Result<Value> {
        let replies = self.post(message, version, timeout, headers, on_notification).await?;
        replies.into_iter().find(|r| r.get("id") == message.get("id")).ok_or_else(|| {
            let method = message.get("method").and_then(Value::as_str).unwrap_or("");
            McpError::new(format!("{} did not answer {method}", self.host())).into()
        })
    }

    async fn notify(&self, message: &Value, version: Option<&str>) -> Result<()> {
        self.post(message, version, None, &[], &mut |_| {}).await.map(|_| ())
    }

    async fn cancel(&self, notification: &Value, version: Option<&str>) -> Result<()> {
        if version == Some(client::VERSION) {
            return Ok(());
        }
        self.notify(notification, version).await
    }

    async fn close(&self) {
        self.set_session(None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refuses_plain_http_outside_loopback_and_urls_with_credentials() {
        assert!(!Http::is_secure("http://mcp.example.com/mcp"));
        assert!(Http::is_secure("http://localhost:3000/mcp"));
        assert!(Http::is_secure("http://127.0.0.1:3000/mcp"));
        assert!(Http::is_secure("https://mcp.example.com/mcp"));
        assert!(!Http::is_secure("https://mcp.example.com@attacker.io/mcp"));
    }

    #[test]
    fn encodes_header_values_that_are_not_plain_ascii() {
        assert_eq!(header_value("search").as_deref(), Some("search"));
        let encoded = |v: &str| format!("=?base64?{}?=", base64::engine::general_purpose::STANDARD.encode(v));
        assert_eq!(header_value(" us-west1"), Some(encoded(" us-west1")));
        assert_eq!(header_value("file:///Überblick.md"), Some(encoded("file:///Überblick.md")));
        assert_eq!(header_value("=?base64?abc?="), Some(encoded("=?base64?abc?=")));
    }
}
