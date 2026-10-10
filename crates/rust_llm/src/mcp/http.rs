//! Port of `lib/ruby_llm/mcp/http.rb`: Streamable HTTP. Every message is its own POST, answered
//! with a JSON body or with an event stream that carries the request's notifications before its
//! response. Plain HTTP is only allowed on loopback addresses, URLs cannot carry credentials, and
//! redirects are never followed. A cancelled chat closes the stream, which is how 2026-07-28
//! cancels a request. Servers that predate it may keep a session, which ends with a DELETE when
//! the transport closes.
//!
//! When a stream ends before the response, 2026-07-28 sends the request again with a new ID,
//! while older servers resume the stream from its last event ID after the wait they ask for, a
//! few times at most.
//!
//! A subscription's stream stays open; older servers send changes on the session's event stream
//! instead. A server's own requests, on any stream, are answered right away: pings with a
//! result, the rest with method not found, so a server never waits on RustLLM.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use base64::Engine;
use futures::StreamExt;
use serde_json::Value;

use super::oauth::Recovery;
use super::{McpError, OnNotification, Transport, client};
use crate::error::{Error, ErrorResponse, Result};
use crate::transport::SseParser;

const LOOPBACK_HOSTS: &[&str] = &["localhost", "127.0.0.1", "[::1]", "::1"];
const CLOSE_TIMEOUT: Duration = Duration::from_secs(5);
const RECONNECTS: usize = 3;
const RECONNECT_DELAY: Duration = Duration::from_secs(1);
const CHECK_INTERVAL: Duration = Duration::from_millis(100);

/// Headers sent with every request, resolved per request for its HTTP verb (`headers: ->(verb)
/// { ... }`).
pub type HeaderSource = Arc<dyn Fn(&str) -> Vec<(String, String)> + Send + Sync>;

/// The OAuth side of `HTTP.new(headers:, unauthorized:)`: the `Authorization` header, resolved
/// per request (refreshing an expiring token), and the `unauthorized` callback, which receives
/// the `WWW-Authenticate` header and status of a 401 or 403 and returns whether a 401 is worth
/// one retry.
///
/// DPoP (RFC 9449) needs more than one header and a proof per HTTP verb: `authorization_headers`
/// returns every header that authorizes a `verb` request, `recover` also receives the response's
/// `DPoP-Nonce` and the ways the request already recovered, and `responded` receives the
/// `DPoP-Nonce` of every response. Their defaults use the first two methods.
#[async_trait]
pub trait Authorization: Send + Sync {
    async fn authorization(&self) -> Result<Option<String>>;
    async fn unauthorized(&self, www_authenticate: Option<&str>, status: u16) -> bool;

    /// `headers.call(verb)`'s OAuth part: `oauth.authorization_headers(verb)`.
    async fn authorization_headers(&self, _verb: &str) -> Result<Vec<(String, String)>> {
        Ok(self
            .authorization()
            .await?
            .map(|value| vec![("Authorization".to_string(), value)])
            .unwrap_or_default())
    }

    /// `unauthorized.call(headers, status, recovered)`: how to send a rejected request again, or
    /// `None`. Fails when getting a new token fails, such as when a grant is refused.
    async fn recover(
        &self,
        www_authenticate: Option<&str>,
        status: u16,
        _nonce: Option<&str>,
        _recovered: &[Recovery],
    ) -> Result<Option<Recovery>> {
        Ok(self
            .unauthorized(www_authenticate, status)
            .await
            .then_some(Recovery::Token))
    }

    /// `responded.call(headers)`: the `DPoP-Nonce` a response carried.
    fn responded(&self, _nonce: Option<&str>) {}
}

fn dpop_nonce(response: &reqwest::Response) -> Option<String> {
    response
        .headers()
        .get("dpop-nonce")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string)
}

/// A failed response: its status, `WWW-Authenticate` header, and the stream of its body.
struct Failure {
    status: u16,
    challenge: Option<String>,
    stream: Stream,
    nonce: Option<String>,
}

/// `RubyLLM::MCP::HTTP`.
pub struct Http {
    url: reqwest::Url,
    headers: HeaderSource,
    client: reqwest::Client,
    timeout: Duration,
    session: Mutex<Option<String>>,
    version: Mutex<Option<String>>,
    authorization: Option<Arc<dyn Authorization>>,
}

impl Http {
    /// `HTTP.secure?`: HTTPS, or plain HTTP to a loopback address, and no credentials in the URL.
    pub fn is_secure(url: &str) -> bool {
        let Ok(url) = reqwest::Url::parse(url) else {
            return false;
        };
        if !url.username().is_empty() || url.password().is_some() {
            return false;
        }
        url.scheme() == "https"
            || (url.scheme() == "http"
                && url.host_str().is_some_and(|h| LOOPBACK_HOSTS.contains(&h)))
    }

    /// `HTTP.loopback?`.
    pub fn is_loopback(url: &str) -> bool {
        reqwest::Url::parse(url)
            .ok()
            .is_some_and(|u| u.host_str().is_some_and(|h| LOOPBACK_HOSTS.contains(&h)))
    }

    /// Raises `Error::Argument` for insecure URLs, like `HTTP.new`. `timeout` bounds connecting
    /// and each read, as Faraday's `options.timeout` does, so a stream that keeps sending stays
    /// open.
    pub fn new(url: &str, headers: HeaderSource, timeout: Duration) -> Result<Http> {
        if !Http::is_secure(url) {
            return Err(Error::Argument(format!(
                "MCP servers must use HTTPS without credentials in the URL: {url}"
            )));
        }
        let url = reqwest::Url::parse(url).map_err(|e| Error::Argument(e.to_string()))?;
        let client = reqwest::Client::builder()
            .connect_timeout(timeout)
            .read_timeout(timeout)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|e| Error::Configuration(e.to_string()))?;
        Ok(Http {
            url,
            headers,
            client,
            timeout,
            session: Mutex::new(None),
            version: Mutex::new(None),
            authorization: None,
        })
    }

    /// `unauthorized:`: OAuth for this server (see [`Authorization`]).
    pub fn with_authorization(mut self, authorization: Arc<dyn Authorization>) -> Http {
        self.authorization = Some(authorization);
        self
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

    fn take_session(&self) -> Option<String> {
        self.session.lock().ok().and_then(|mut s| s.take())
    }

    fn last_version(&self) -> Option<String> {
        self.version.lock().ok().and_then(|v| v.clone())
    }

    /// `custom_headers(verb)` plus the OAuth `Authorization` header.
    async fn custom_headers(
        &self,
        mut request: reqwest::RequestBuilder,
        verb: &str,
    ) -> Result<reqwest::RequestBuilder> {
        for (key, value) in (self.headers)(verb) {
            request = request.header(key, value);
        }
        if let Some(authorization) = &self.authorization {
            for (key, value) in authorization.authorization_headers(verb).await? {
                request = request.header(key, value);
            }
        }
        Ok(request)
    }

    /// `post(..., recovered:)`: a 401 the `unauthorized` callback fixes is sent once more; a 404
    /// for a request in a session means the session ended.
    async fn post(
        &self,
        message: &Value,
        version: Option<&str>,
        timeout: Option<Duration>,
        params: &[(String, String)],
        on_notification: &mut OnNotification<'_>,
        subscription: bool,
    ) -> Result<Stream> {
        let method = message.get("method").and_then(Value::as_str);
        let session = if method == Some("initialize") {
            None
        } else {
            self.session()
        };
        let mut recovered: Vec<Recovery> = Vec::new();
        loop {
            let failure = match self
                .post_once(
                    message,
                    version,
                    timeout,
                    params,
                    session.as_deref(),
                    on_notification,
                    subscription,
                )
                .await?
            {
                Ok(stream) => return Ok(stream),
                Err(failure) => failure,
            };
            if failure.stream.is_answered() {
                return Ok(failure.stream);
            }
            if session.is_some() && message.get("id").is_some() && failure.status == 404 {
                return Err(McpError::session_expired(format!(
                    "{} ended the session",
                    self.host()
                ))
                .into());
            }
            let Some(recovery) = self.recover(&failure, &recovered).await? else {
                return Err(self.failure(failure.status, &failure.stream));
            };
            recovered.push(recovery);
        }
    }

    /// `recover(response, recovered)`: a 403 only reports its challenge; a 401 is sent again
    /// once per way it recovers, such as with a new token or a nonce for its proof.
    async fn recover(&self, failure: &Failure, recovered: &[Recovery]) -> Result<Option<Recovery>> {
        let Some(authorization) = &self.authorization else {
            return Ok(None);
        };
        if !matches!(failure.status, 401 | 403) {
            return Ok(None);
        }
        let recovery = authorization
            .recover(
                failure.challenge.as_deref(),
                failure.status,
                failure.nonce.as_deref(),
                recovered,
            )
            .await?;
        Ok(recovery.filter(|r| failure.status == 401 && !recovered.contains(r)))
    }

    /// `responded(headers)`.
    fn responded(&self, nonce: Option<&str>) {
        if let Some(authorization) = &self.authorization {
            authorization.responded(nonce);
        }
    }

    /// One POST: its stream, or the failure of an error status.
    #[allow(clippy::too_many_arguments)]
    async fn post_once(
        &self,
        message: &Value,
        version: Option<&str>,
        timeout: Option<Duration>,
        params: &[(String, String)],
        session: Option<&str>,
        on_notification: &mut OnNotification<'_>,
        subscription: bool,
    ) -> Result<std::result::Result<Stream, Failure>> {
        let method = message.get("method").and_then(Value::as_str);
        let mut request = self
            .client
            .post(self.url.clone())
            .header("Content-Type", "application/json")
            .header("Accept", "application/json, text/event-stream")
            .body(message.to_string());
        if let Some(version) = version {
            request = request.header("MCP-Protocol-Version", version);
        }
        if let Some(session) = session {
            request = request.header("Mcp-Session-Id", session);
        }
        if let Some(method) = method {
            request = request.header("Mcp-Method", method);
        }
        let name = message
            .pointer("/params/name")
            .or_else(|| message.pointer("/params/uri"))
            .or_else(|| message.pointer("/params/taskId"));
        if let Some(name) = name.and_then(|n| header_value(&text_of(n))) {
            request = request.header("Mcp-Name", name);
        }
        request = self.custom_headers(request, "POST").await?;
        for (key, value) in params {
            if let Some(value) = header_value(value) {
                request = request.header(format!("Mcp-Param-{key}"), value);
            }
        }
        if let Some(timeout) = timeout {
            request = request.timeout(timeout);
        }
        let mut stream = Stream::new(message.get("id").cloned());
        stream.ends_when_cancelled = subscription;
        let response = request.send().await.map_err(request_error)?;
        let status = response.status().as_u16();
        let challenge = response
            .headers()
            .get("www-authenticate")
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        let session_id = response
            .headers()
            .get("mcp-session-id")
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        let nonce = dpop_nonce(&response);
        self.responded(nonce.as_deref());
        self.read(&mut stream, response, version, on_notification)
            .await?;
        if !(200..300).contains(&status) {
            return Ok(Err(Failure {
                status,
                challenge,
                stream,
                nonce,
            }));
        }
        // `remember`
        if let Some(version) = version
            && let Ok(mut v) = self.version.lock()
        {
            *v = Some(version.to_string());
        }
        if method == Some("initialize") {
            self.set_session(session_id);
        }
        Ok(Ok(stream))
    }

    /// `get`: reads the session's event stream, from `last_event_id` when resuming.
    async fn get(
        &self,
        stream: &mut Stream,
        version: Option<&str>,
        timeout: Option<Duration>,
        last_event_id: Option<&str>,
        on_notification: &mut OnNotification<'_>,
    ) -> Result<std::result::Result<(), Failure>> {
        let mut request = self
            .client
            .get(self.url.clone())
            .header("Accept", "text/event-stream");
        if let Some(version) = version {
            request = request.header("MCP-Protocol-Version", version);
        }
        if let Some(session) = self.session() {
            request = request.header("Mcp-Session-Id", session);
        }
        if let Some(last_event_id) = last_event_id {
            request = request.header("Last-Event-ID", last_event_id);
        }
        request = self.custom_headers(request, "GET").await?;
        if let Some(timeout) = timeout {
            request = request.timeout(timeout);
        }
        let response = request.send().await.map_err(request_error)?;
        let status = response.status().as_u16();
        let challenge = response
            .headers()
            .get("www-authenticate")
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        let nonce = dpop_nonce(&response);
        self.responded(nonce.as_deref());
        if !(200..300).contains(&status) {
            let mut failed = Stream::new(None);
            self.read(&mut failed, response, version, on_notification)
                .await?;
            return Ok(Err(Failure {
                status,
                challenge,
                stream: failed,
                nonce,
            }));
        }
        self.read(stream, response, version, on_notification)
            .await?;
        Ok(Ok(()))
    }

    /// `Stream#read`: feeds the body to `stream` until it carries its answer, answering what the
    /// server asks on the way. An event stream that breaks midway counts as ended.
    async fn read(
        &self,
        stream: &mut Stream,
        response: reqwest::Response,
        version: Option<&str>,
        on_notification: &mut OnNotification<'_>,
    ) -> Result<()> {
        let mut body = response.bytes_stream();
        loop {
            let chunk = tokio::select! {
                chunk = body.next() => chunk,
                () = tokio::time::sleep(CHECK_INTERVAL) => {
                    if crate::progress::is_cancelled() {
                        return Err(Error::Cancelled);
                    }
                    continue;
                }
            };
            let Some(chunk) = chunk else {
                stream.finish(on_notification);
                break;
            };
            if crate::progress::is_cancelled() {
                return Err(Error::Cancelled);
            }
            let chunk = match chunk {
                Ok(chunk) => chunk,
                Err(_) if stream.is_events() => break,
                Err(e) => return Err(request_error(e)),
            };
            stream.feed(&chunk, on_notification);
            for request in std::mem::take(&mut stream.asked) {
                self.answer(request, version).await;
            }
            if stream.is_done() {
                break;
            }
        }
        for request in std::mem::take(&mut stream.asked) {
            self.answer(request, version).await;
        }
        Ok(())
    }

    /// `answer`: replies to a request the server sent, without failing the call when the server
    /// refuses the reply.
    fn answer<'a>(
        &'a self,
        request: Value,
        version: Option<&'a str>,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
        Box::pin(async move {
            let reply = super::Client::reply(&request);
            if let Err(e) = self
                .post(&reply, version, None, &[], &mut |_| {}, false)
                .await
            {
                let method = request.get("method").and_then(Value::as_str).unwrap_or("");
                tracing::debug!("{} did not take the answer to {method}: {e}", self.host());
            }
        })
    }

    /// `resume`: waits as long as the server asked, then reads the stream on from its last
    /// event. A resumption that fails ends like a stream without the answer.
    async fn resume(
        &self,
        stream: &Stream,
        version: Option<&str>,
        timeout: Option<Duration>,
        on_notification: &mut OnNotification<'_>,
    ) -> Result<Stream> {
        wait(stream.retry_after().unwrap_or(RECONNECT_DELAY)).await?;
        let mut resumed = Stream::new(stream.id.clone());
        let last_event_id = stream.last_event_id();
        match self
            .get(
                &mut resumed,
                version,
                timeout,
                last_event_id.as_deref(),
                on_notification,
            )
            .await
        {
            Err(Error::Cancelled) => Err(Error::Cancelled),
            Ok(Err(_)) => Ok(Stream::new(stream.id.clone())),
            _ => Ok(resumed),
        }
    }

    /// `resumable?`.
    fn is_resumable(&self, stream: &Stream, limit: Duration) -> bool {
        stream.is_interrupted()
            && stream.last_event_id().is_some()
            && stream.retry_after().unwrap_or_default() <= limit
    }

    /// `listen_to_session`: an older server's own event stream.
    async fn listen_to_session(
        &self,
        version: Option<&str>,
        on_notification: &mut OnNotification<'_>,
    ) -> Result<Option<Value>> {
        let mut stream = Stream::new(None);
        let failure = match self
            .get(&mut stream, version, None, None, on_notification)
            .await
        {
            Ok(Ok(())) => return Ok(None),
            Ok(Err(failure)) => failure,
            Err(Error::Cancelled) => return Err(Error::Cancelled),
            Err(e @ (Error::Timeout(_) | Error::ConnectionFailed(_))) => {
                return Err(McpError::new(e.to_string()).into());
            }
            Err(e) => return Err(e),
        };
        if failure.status == 404 && self.session().is_some() {
            return Err(
                McpError::session_expired(format!("{} ended the session", self.host())).into(),
            );
        }
        if failure.status == 405 {
            return Err(McpError {
                message: format!("{} sends no events", self.host()),
                code: Some(client::METHOD_NOT_FOUND),
                ..Default::default()
            }
            .into());
        }
        Err(self.failure(failure.status, &failure.stream))
    }

    fn failure(&self, status: u16, stream: &Stream) -> Error {
        let response = Some(ErrorResponse {
            status,
            body: stream.body.clone(),
            ..Default::default()
        });
        match status {
            401 => Error::Unauthorized(format!("{} requires authorization", self.host()), response),
            403 => Error::Forbidden(format!("{} refused the request", self.host()), response),
            _ => {
                let error = stream
                    .replies
                    .first()
                    .and_then(|r| r.get("error"))
                    .cloned()
                    .unwrap_or(Value::Null);
                McpError {
                    message: error
                        .get("message")
                        .and_then(Value::as_str)
                        .map(str::to_string)
                        .unwrap_or_else(|| format!("{} answered HTTP {status}", self.host())),
                    code: error.get("code").and_then(Value::as_i64),
                    data: error.get("data").cloned(),
                    response,
                    ..Default::default()
                }
                .into()
            }
        }
    }
}

fn request_error(e: reqwest::Error) -> Error {
    if e.is_timeout() {
        Error::Timeout(e.to_string())
    } else {
        Error::ConnectionFailed(e.to_string())
    }
}

/// `wait`: sleeps, checking for cancellation every `CHECK_INTERVAL`.
async fn wait(duration: Duration) -> Result<()> {
    let deadline = tokio::time::Instant::now() + duration;
    loop {
        if crate::progress::is_cancelled() {
            return Err(Error::Cancelled);
        }
        let now = tokio::time::Instant::now();
        if now >= deadline {
            return Ok(());
        }
        tokio::time::sleep((deadline - now).min(CHECK_INTERVAL)).await;
    }
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
    Some(if safe {
        value.to_string()
    } else {
        format!(
            "=?base64?{}?=",
            base64::engine::general_purpose::STANDARD.encode(value)
        )
    })
}

/// `HTTP::Stream`: collects the answers in one response and yields the messages the server sends
/// of its own, notifications and requests, as they arrive. The body is either a single JSON value
/// or a server-sent event stream; the first character tells them apart. An event stream stops
/// being read once it carries the answer to request `id`, since servers may keep it open. Only
/// answers are kept, so a stream that stays open does not grow.
struct Stream {
    id: Option<Value>,
    parser: SseParser,
    body: String,
    events: Option<bool>,
    replies: Vec<Value>,
    /// Requests the server sent, waiting to be answered.
    asked: Vec<Value>,
    /// A subscription also ends with the server's `notifications/cancelled` for it (Ruby's
    /// `Client#subscribe` throws on it).
    ends_when_cancelled: bool,
    cancelled: Option<Value>,
}

impl Stream {
    fn new(id: Option<Value>) -> Stream {
        Stream {
            id,
            parser: SseParser::default(),
            body: String::new(),
            events: None,
            replies: Vec::new(),
            asked: Vec::new(),
            ends_when_cancelled: false,
            cancelled: None,
        }
    }

    fn feed(&mut self, chunk: &[u8], on_notification: &mut OnNotification<'_>) {
        if self.events == Some(false) {
            self.body.push_str(&String::from_utf8_lossy(chunk));
            return;
        }
        let chunk = if self.events.is_none() {
            self.body.push_str(&String::from_utf8_lossy(chunk));
            let trimmed = self.body.trim_start();
            if trimmed.is_empty() {
                return;
            }
            self.events = Some(!(trimmed.starts_with('{') || trimmed.starts_with('[')));
            if self.events == Some(false) {
                return;
            }
            std::mem::take(&mut self.body).into_bytes()
        } else {
            chunk.to_vec()
        };
        for event in self.parser.feed(&chunk) {
            if !event.data.is_empty() {
                self.receive(&event.data, on_notification);
            }
        }
    }

    /// The body has ended: a bare JSON body is read now.
    fn finish(&mut self, on_notification: &mut OnNotification<'_>) {
        if self.events == Some(false) && self.replies.is_empty() {
            let body = self.body.clone();
            self.receive(&body, on_notification);
        }
    }

    fn receive(&mut self, data: &str, on_notification: &mut OnNotification<'_>) {
        let Ok(parsed) = serde_json::from_str::<Value>(data) else {
            tracing::debug!("MCP server sent a message that is not JSON");
            return;
        };
        let messages = match parsed {
            Value::Array(items) => items,
            other => vec![other],
        };
        for message in messages.into_iter().filter(Value::is_object) {
            if message.get("method").is_none_or(Value::is_null) {
                self.replies.push(message);
            } else if message.get("id").is_some_and(|id| !id.is_null()) {
                self.asked.push(message);
            } else if self.cancels(&message) {
                self.cancelled = Some(message);
            } else {
                on_notification(&message);
            }
        }
    }

    fn cancels(&self, message: &Value) -> bool {
        self.ends_when_cancelled
            && message.get("method").and_then(Value::as_str) == Some("notifications/cancelled")
            && message.pointer("/params/requestId") == self.id.as_ref()
    }

    fn is_events(&self) -> bool {
        self.events == Some(true)
    }

    /// Whether to stop reading: the answer (or the cancellation of a subscription) arrived.
    fn is_done(&self) -> bool {
        self.is_events() && (self.answer().is_some() || self.cancelled.is_some())
    }

    /// `answer`: the reply to request `id`.
    fn answer(&self) -> Option<&Value> {
        let id = self.id.as_ref()?;
        self.replies.iter().find(|r| r.get("id") == Some(id))
    }

    /// `answered?`: some servers, such as Google's Drive preview, send a complete JSON-RPC
    /// result with an error status. The result is the answer.
    fn is_answered(&self) -> bool {
        self.answer().is_some_and(|a| a.get("result").is_some())
    }

    fn is_interrupted(&self) -> bool {
        self.is_events() && self.answer().is_none()
    }

    fn last_event_id(&self) -> Option<String> {
        let id = self.parser.last_event_id();
        (!id.is_empty()).then(|| id.to_string())
    }

    fn retry_after(&self) -> Option<Duration> {
        self.parser.reconnection_time().map(Duration::from_millis)
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
        let mut stream = self
            .post(message, version, timeout, headers, on_notification, false)
            .await?;
        for _ in 0..RECONNECTS {
            if version == Some(client::VERSION) {
                if !stream.is_interrupted() {
                    break;
                }
                let mut again = message.clone();
                again["id"] = uuid::Uuid::new_v4().to_string().into();
                stream = self
                    .post(&again, version, timeout, headers, on_notification, false)
                    .await?;
            } else {
                if !self.is_resumable(&stream, timeout.unwrap_or(self.timeout)) {
                    break;
                }
                stream = self
                    .resume(&stream, version, timeout, on_notification)
                    .await?;
            }
        }
        stream.answer().cloned().ok_or_else(|| {
            let method = message.get("method").and_then(Value::as_str).unwrap_or("");
            McpError::new(format!("{} did not answer {method}", self.host())).into()
        })
    }

    async fn notify(&self, message: &Value, version: Option<&str>) -> Result<()> {
        self.post(message, version, None, &[], &mut |_| {}, false)
            .await
            .map(|_| ())
    }

    async fn cancel(&self, notification: &Value, version: Option<&str>) -> Result<()> {
        if version == Some(client::VERSION) {
            return Ok(());
        }
        self.notify(notification, version).await
    }

    async fn listen(
        &self,
        message: Option<&Value>,
        version: Option<&str>,
        on_notification: &mut OnNotification<'_>,
    ) -> Result<Option<Value>> {
        let Some(message) = message else {
            return self.listen_to_session(version, on_notification).await;
        };
        let stream = match self
            .post(message, version, None, &[], on_notification, true)
            .await
        {
            Ok(stream) => stream,
            Err(e @ (Error::Timeout(_) | Error::ConnectionFailed(_))) => {
                return Err(McpError::new(e.to_string()).into());
            }
            Err(e) => return Err(e),
        };
        match stream.answer().cloned().or(stream.cancelled) {
            Some(ending) => Ok(Some(ending)),
            None => Err(McpError::new(format!("{} closed the subscription", self.host())).into()),
        }
    }

    /// `close`: ends the session of a server that predates 2026-07-28 with a DELETE.
    async fn close(&self) {
        let Some(session) = self.take_session() else {
            return;
        };
        let mut request = self
            .client
            .delete(self.url.clone())
            .header("Mcp-Session-Id", session)
            .timeout(CLOSE_TIMEOUT);
        if let Some(version) = self.last_version() {
            request = request.header("MCP-Protocol-Version", version);
        }
        let Ok(request) = self.custom_headers(request, "DELETE").await else {
            return;
        };
        if let Ok(response) = request.send().await {
            self.responded(dpop_nonce(&response).as_deref());
        }
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
        let encoded = |v: &str| {
            format!(
                "=?base64?{}?=",
                base64::engine::general_purpose::STANDARD.encode(v)
            )
        };
        assert_eq!(header_value(" us-west1"), Some(encoded(" us-west1")));
        assert_eq!(
            header_value("file:///Überblick.md"),
            Some(encoded("file:///Überblick.md"))
        );
        assert_eq!(
            header_value("=?base64?abc?="),
            Some(encoded("=?base64?abc?="))
        );
    }

    // spec: mcp/http_spec.rb:800 keeps only the answers of a stream, so one that stays open does not grow
    #[test]
    fn keeps_only_the_answers_of_a_stream() {
        let mut stream = Stream::new(Some(Value::from("listen-1")));
        let event = format!(
            "data: {}\n\n",
            serde_json::json!({ "jsonrpc": "2.0", "method": "notifications/tools/list_changed" })
        );
        for _ in 0..100 {
            stream.feed(event.as_bytes(), &mut |_| {});
        }
        assert!(stream.replies.is_empty());
        assert!(stream.body.is_empty());
    }
}
