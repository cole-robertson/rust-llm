//! Port of `lib/ruby_llm/transport/websocket_connection.rb` (`RubyLLM::Transport::WebsocketConnection`):
//! the client WebSocket that streaming transcription (Gemini Live, xAI) runs on. RubyLLM drives
//! `websocket-driver` over a raw socket; this port uses tokio-tungstenite over rustls with the
//! webpki roots, and keeps RubyLLM's rules: `ws`/`wss` only, no URL credentials or fragment, no
//! HTTP proxy, reads and writes bounded by `request_timeout`, 16 MiB frames, a close code other
//! than 1000 is an error, and a peer that drops the socket without a close frame is an error.
//!
//! ```ruby
//! Transport::WebsocketConnection.open(url, headers:, config:) do |socket|
//!   socket.send_text(JSON.generate(setup))
//!   socket.each_message(write: ->(s) { send_audio(s) }) { |message| handle(message) }
//! end
//! ```

use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use futures::stream::{SplitSink, SplitStream};
use futures::{FutureExt, SinkExt, StreamExt};
use tokio::net::TcpStream;
use tokio::sync::{Mutex, watch};
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::{HeaderName, HeaderValue};
use tokio_tungstenite::tungstenite::protocol::WebSocketConfig;
use tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode;
use tokio_tungstenite::tungstenite::{self, Message};
use tokio_tungstenite::{Connector, MaybeTlsStream, WebSocketStream};

use crate::config::Config;
use crate::error::{Error, Result};

type Socket = WebSocketStream<MaybeTlsStream<TcpStream>>;

/// `MAX_FRAME_BYTES`.
pub const MAX_FRAME_BYTES: usize = 16 * 1024 * 1024;

const NO_CLOSE_FRAME: &str = "WebSocket connection ended without a close frame";

/// A client WebSocket (`RubyLLM::Transport::WebsocketConnection`). Messages are the frame
/// payloads: text frames as their UTF-8 bytes, binary frames as-is.
pub struct WebsocketConnection {
    url: String,
    headers: Vec<(String, String)>,
    timeout: Duration,
    sink: Mutex<Option<SplitSink<Socket, Message>>>,
    stream: Mutex<Option<SplitStream<Socket>>>,
    opened: AtomicBool,
    closed: watch::Sender<bool>,
    close_sent: AtomicBool,
    error: StdMutex<Option<String>>,
}

impl std::fmt::Debug for WebsocketConnection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WebsocketConnection")
            .field("url", &self.url)
            .finish()
    }
}

fn failure(message: impl Into<String>) -> Error {
    Error::Api(message.into(), None)
}

fn timed_out() -> Error {
    failure("WebSocket operation timed out")
}

impl WebsocketConnection {
    /// `WebsocketConnection.open(url, headers:, config:) { |connection| }`: connects, runs `f`,
    /// and closes the connection however `f` ends.
    pub async fn open<T>(
        url: &str,
        headers: &[(String, String)],
        config: &Config,
        f: impl AsyncFnOnce(&WebsocketConnection) -> Result<T>,
    ) -> Result<T> {
        let connection = WebsocketConnection::new(url, headers, config)?;
        connection.connect().await?;
        let result = f(&connection).await;
        connection.close();
        connection.send_close_frame().await;
        result
    }

    /// `WebsocketConnection.new(url, headers:, config:)`: validates the URL and configuration
    /// without opening a socket.
    pub fn new(
        url: &str,
        headers: &[(String, String)],
        config: &Config,
    ) -> Result<WebsocketConnection> {
        let (scheme, rest) = url.split_once("://").unwrap_or(("", url));
        if !scheme.eq_ignore_ascii_case("ws") && !scheme.eq_ignore_ascii_case("wss") {
            return Err(Error::Argument("WebSocket URL must use ws or wss".into()));
        }
        let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
        if authority.contains('@') || url.contains('#') {
            return Err(Error::Argument(
                "WebSocket URL must not contain credentials or a fragment".into(),
            ));
        }
        if config.http_proxy.is_some() {
            return Err(Error::Argument(
                "WebSocket connections do not support HTTP proxies".into(),
            ));
        }
        Ok(WebsocketConnection {
            url: url.to_string(),
            headers: headers.to_vec(),
            timeout: config.request_timeout,
            sink: Mutex::new(None),
            stream: Mutex::new(None),
            opened: AtomicBool::new(false),
            closed: watch::Sender::new(false),
            close_sent: AtomicBool::new(false),
            error: StdMutex::new(None),
        })
    }

    pub fn url(&self) -> &str {
        &self.url
    }

    /// `connect`: the TCP (and TLS) connection and the opening handshake, within the timeout.
    pub async fn connect(&self) -> Result<&WebsocketConnection> {
        let mut request = self
            .url
            .as_str()
            .into_client_request()
            .map_err(|e| Error::Argument(format!("invalid WebSocket URL: {e}")))?;
        for (name, value) in &self.headers {
            let name = HeaderName::from_bytes(name.as_bytes())
                .map_err(|e| Error::Argument(format!("invalid header {name:?}: {e}")))?;
            let value = HeaderValue::from_str(value)
                .map_err(|e| Error::Argument(format!("invalid value for header {name}: {e}")))?;
            request.headers_mut().insert(name, value);
        }
        let limits = WebSocketConfig::default()
            .max_message_size(Some(MAX_FRAME_BYTES))
            .max_frame_size(Some(MAX_FRAME_BYTES));
        let connecting = tokio_tungstenite::connect_async_tls_with_config(
            request,
            Some(limits),
            false,
            Some(Connector::Rustls(tls_config()?)),
        );
        let socket = match tokio::time::timeout(self.timeout, connecting).await {
            Err(_) => return Err(timed_out()),
            Ok(Err(e)) => return Err(handshake_error(e)),
            Ok(Ok((socket, _response))) => socket,
        };
        let (sink, stream) = socket.split();
        *self.sink.lock().await = Some(sink);
        *self.stream.lock().await = Some(stream);
        self.opened.store(true, Ordering::SeqCst);
        Ok(self)
    }

    /// `send_text`.
    pub async fn send_text(&self, text: impl Into<String>) -> Result<()> {
        self.send(Message::text(text.into())).await
    }

    /// `send_binary`.
    pub async fn send_binary(&self, data: impl Into<Vec<u8>>) -> Result<()> {
        self.send(Message::binary(data.into())).await
    }

    async fn send(&self, message: Message) -> Result<()> {
        self.ensure_open()?;
        let mut sink = self.sink.lock().await;
        let sink = sink.as_mut().ok_or_else(closed_error)?;
        match tokio::time::timeout(self.timeout, sink.send(message)).await {
            Err(_) => Err(timed_out()),
            Ok(Err(e)) => Err(self.fail(e)),
            Ok(Ok(())) => Ok(()),
        }
    }

    /// `read(timeout:)`: the next message, or `None` once the connection has closed. A read
    /// blocked here ends as soon as [`close`](Self::close) is called.
    pub async fn read(&self, timeout: Option<Duration>) -> Result<Option<Vec<u8>>> {
        let deadline = tokio::time::Instant::now() + timeout.unwrap_or(self.timeout);
        let mut closed = self.closed.subscribe();
        let mut stream = self.stream.lock().await;
        loop {
            self.raise_error()?;
            if *closed.borrow() {
                return Ok(None);
            }
            let Some(socket) = stream.as_mut() else {
                return Ok(None);
            };
            let next = tokio::select! {
                next = tokio::time::timeout_at(deadline, socket.next()) => next,
                _ = closed.wait_for(|c| *c) => return self.raise_error().map(|()| None),
            };
            match next {
                Err(_) => return Err(timed_out()),
                Ok(next) => {
                    if let Some(message) = self.receive(next) {
                        return Ok(Some(message));
                    }
                }
            }
        }
    }

    /// `read_available`: a message that has already arrived in full, or `None` without waiting.
    /// Bytes of a partial frame stay buffered for the next call.
    pub async fn read_available(&self) -> Result<Option<Vec<u8>>> {
        self.ensure_open()?;
        let mut stream = self.stream.lock().await;
        let mut message = None;
        while !self.is_closed() && self.error().is_none() {
            let Some(socket) = stream.as_mut() else { break };
            let Some(next) = socket.next().now_or_never() else {
                break;
            };
            if let Some(received) = self.receive(next) {
                message = Some(received);
                break;
            }
        }
        if message.is_none() {
            // `rescue EOFError`: a dropped peer reports why before the closed state does.
            self.raise_error()?;
            self.ensure_open()?;
        }
        Ok(message)
    }

    /// `each_message(write:) { |message| }`: runs `write` alongside the reads and hands every
    /// message to `on_message` until the connection closes. `write` then gets one second to
    /// finish; a writer that fails closes the connection and its error is returned.
    pub async fn each_message(
        &self,
        write: impl Future<Output = Result<()>>,
        mut on_message: impl FnMut(Vec<u8>) -> Result<()>,
    ) -> Result<()> {
        let writer = async {
            let result = write.await;
            if result.is_err() {
                self.close();
            }
            result
        };
        let reader = async {
            while let Some(message) = self.read(None).await? {
                on_message(message)?;
            }
            Ok::<(), Error>(())
        };
        let mut writer = std::pin::pin!(writer);
        let mut reader = std::pin::pin!(reader);
        let mut written: Option<Result<()>> = None;
        let read = loop {
            tokio::select! {
                result = &mut writer, if written.is_none() => written = Some(result),
                result = &mut reader => break result,
            }
        };
        // `raise writer_error || e`: a failed writer explains why the reads stopped.
        let result = match (read, written) {
            (Err(_), Some(Err(writer_error))) => Err(writer_error),
            (Err(e), _) => Err(e),
            (Ok(()), Some(written)) => written,
            // `finish_writer`: `writer.join(1)`.
            (Ok(()), None) => match tokio::time::timeout(Duration::from_secs(1), &mut writer).await
            {
                Ok(written) => written,
                Err(_) => Err(failure(
                    "WebSocket closed before outgoing messages finished",
                )),
            },
        };
        self.close();
        result
    }

    /// `close`: marks the connection closed and wakes any blocked read. The close frame goes
    /// out when [`open`](Self::open) finishes.
    pub fn close(&self) {
        self.closed.send_replace(true);
    }

    fn is_closed(&self) -> bool {
        *self.closed.borrow()
    }

    /// The close frame `@driver.close` writes, sent once and best-effort.
    async fn send_close_frame(&self) {
        if !self.opened.load(Ordering::SeqCst) || self.close_sent.swap(true, Ordering::SeqCst) {
            return;
        }
        let mut sink = self.sink.lock().await;
        if let Some(sink) = sink.as_mut() {
            let _ = tokio::time::timeout(self.timeout, sink.close()).await;
        }
    }

    fn error(&self) -> Option<String> {
        self.error.lock().ok().and_then(|e| e.clone())
    }

    fn set_error(&self, message: String) {
        if let Ok(mut error) = self.error.lock() {
            error.get_or_insert(message);
        }
    }

    fn raise_error(&self) -> Result<()> {
        match self.error() {
            Some(message) => Err(failure(message)),
            None => Ok(()),
        }
    }

    /// `ensure_open`.
    fn ensure_open(&self) -> Result<()> {
        if !self.opened.load(Ordering::SeqCst) || self.is_closed() {
            return Err(closed_error());
        }
        self.raise_error()
    }

    /// Records a transport failure (`@error`), unless the connection was already closed.
    fn fail(&self, error: tungstenite::Error) -> Error {
        let message = match &error {
            tungstenite::Error::Protocol(
                tungstenite::error::ProtocolError::ResetWithoutClosingHandshake,
            ) => NO_CLOSE_FRAME.to_string(),
            tungstenite::Error::Io(e) if ends_without_close(e) => NO_CLOSE_FRAME.to_string(),
            tungstenite::Error::ConnectionClosed | tungstenite::Error::AlreadyClosed => {
                return closed_error();
            }
            other => other.to_string(),
        };
        self.close();
        self.set_error(message.clone());
        failure(message)
    }

    /// The driver's `:message` and `:close` events for one read. Returns a data message; control
    /// frames and closes return `None` after updating the state.
    fn receive(&self, next: Option<tungstenite::Result<Message>>) -> Option<Vec<u8>> {
        match next {
            None => {
                if !self.is_closed() {
                    self.set_error(NO_CLOSE_FRAME.into());
                }
                self.close();
                None
            }
            Some(Err(e)) => {
                if !self.is_closed() {
                    self.fail(e);
                }
                self.close();
                None
            }
            Some(Ok(Message::Text(text))) => Some(text.as_str().as_bytes().to_vec()),
            Some(Ok(Message::Binary(data))) => Some(data.to_vec()),
            Some(Ok(Message::Close(frame))) => {
                if let Some(frame) = frame.filter(|f| f.code != CloseCode::Normal) {
                    self.set_error(format!(
                        "WebSocket closed with code {}: {}",
                        u16::from(frame.code),
                        frame.reason.as_str()
                    ));
                }
                self.close();
                None
            }
            Some(Ok(_)) => None,
        }
    }
}

fn closed_error() -> Error {
    failure("WebSocket connection is closed")
}

fn ends_without_close(e: &std::io::Error) -> bool {
    matches!(
        e.kind(),
        std::io::ErrorKind::UnexpectedEof
            | std::io::ErrorKind::ConnectionReset
            | std::io::ErrorKind::ConnectionAborted
            | std::io::ErrorKind::BrokenPipe
    )
}

/// Errors while opening: a dropped socket is `ended without a close frame`, a refused upgrade is
/// the driver's `Unexpected response code`, and TLS failures keep rustls's reason.
fn handshake_error(error: tungstenite::Error) -> Error {
    match error {
        tungstenite::Error::Io(e) if ends_without_close(&e) => failure(NO_CLOSE_FRAME),
        tungstenite::Error::Protocol(
            tungstenite::error::ProtocolError::HandshakeIncomplete
            | tungstenite::error::ProtocolError::ResetWithoutClosingHandshake,
        ) => failure(NO_CLOSE_FRAME),
        tungstenite::Error::Http(response) => failure(format!(
            "Unexpected response code: {}",
            response.status().as_u16()
        )),
        tungstenite::Error::Io(e) => Error::ConnectionFailed(e.to_string()),
        other => Error::ConnectionFailed(other.to_string()),
    }
}

/// `OpenSSL::SSL::SSLContext#set_params`: verify the peer against the webpki roots.
fn tls_config() -> Result<Arc<rustls::ClientConfig>> {
    let roots = rustls::RootCertStore {
        roots: webpki_roots::TLS_SERVER_ROOTS.to_vec(),
    };
    let config = rustls::ClientConfig::builder_with_provider(Arc::new(
        rustls::crypto::ring::default_provider(),
    ))
    .with_safe_default_protocol_versions()
    .map_err(|e| Error::ConnectionFailed(e.to_string()))?
    .with_root_certificates(roots)
    .with_no_client_auth();
    Ok(Arc::new(config))
}
