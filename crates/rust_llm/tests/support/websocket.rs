//! WebSocket counterparts of RubyLLM's spec support: a local tokio-tungstenite server for the
//! transport specs, and a replay server for `spec/fixtures/websocket_cassettes/*.json`
//! (`spec/support/websocket_cassette.rb`).
//!
//! A WebSocket cassette records the URL (credentials filtered), every frame the client sent, and
//! every event the server sent. Ruby's replay stubs the socket: it hands the recorded events to
//! the protocol while the writer runs to completion, then requires the sent frames to equal the
//! recording. This replay does the same over a real socket. It records the handshake path and
//! headers, sends the first event on connect, and sends the rest once the client has sent as
//! many frames as the recording holds (or has gone quiet), which is where the live server sent
//! its final events. It then reads until the client closes, closing normally itself if the
//! client stays quiet. Frames are sanitized the way `RealtimeCredentials.sanitize(redact_audio:
//! true)` does: binary frames by SHA-256 and byte count, Base64 audio in JSON frames by the
//! SHA-256 of its bytes. Every accepted connection is counted; only the first is served.

#![allow(dead_code)]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use base64::Engine;
use futures::{SinkExt, StreamExt};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tokio::net::{TcpListener, TcpStream};
use tokio::task::JoinHandle;
use tokio_tungstenite::WebSocketStream;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::handshake::server::{Request, Response};

/// A recorded WebSocket session.
pub struct WebsocketCassette {
    pub url: String,
    pub incoming: Vec<Value>,
    pub outgoing: Vec<Value>,
}

impl WebsocketCassette {
    pub fn load(name: &str) -> WebsocketCassette {
        let path = format!(
            "{}/tests/cassettes/websocket/{name}.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let data: Value = serde_json::from_str(&text).expect("cassette JSON");
        let list = |key: &str| data[key].as_array().cloned().unwrap_or_default();
        WebsocketCassette {
            url: data["url"].as_str().unwrap_or_default().to_string(),
            incoming: list("incoming"),
            outgoing: list("outgoing"),
        }
    }
}

/// What the replay server saw.
#[derive(Debug, Default, Clone)]
pub struct Session {
    /// The request path and query of the handshake.
    pub path: String,
    /// The handshake's headers, lowercased.
    pub headers: Vec<(String, String)>,
    /// Every frame the client sent, sanitized like the recording.
    pub outgoing: Vec<Value>,
    /// Whether the client sent a close frame.
    pub closed_cleanly: bool,
    /// How many connections the server accepted.
    pub connections: usize,
    /// How many frames the client sent before the first event.
    pub before_first_event: usize,
}

pub struct Replay {
    pub port: u16,
    pub session: Arc<Mutex<Session>>,
    task: JoinHandle<()>,
}

/// How long the client may stay quiet before the replay moves on.
const QUIET: Duration = Duration::from_millis(300);

impl Replay {
    /// Serves one recorded session on a local port.
    pub async fn start(cassette: &WebsocketCassette) -> Replay {
        Replay::start_holding(cassette, Duration::ZERO).await
    }

    /// Like `start`, but holds the first event for `hold`, recording what the client sends
    /// meanwhile in [`Session::before_first_event`].
    pub async fn start_holding(cassette: &WebsocketCassette, hold: Duration) -> Replay {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let session = Arc::new(Mutex::new(Session::default()));
        let incoming = cassette.incoming.clone();
        let expected = cassette.outgoing.len();
        let seen = session.clone();
        let task = tokio::spawn(async move {
            let Ok((tcp, _)) = listener.accept().await else {
                return;
            };
            seen.lock().unwrap().connections += 1;
            let counter = seen.clone();
            let extra = tokio::spawn(async move {
                while listener.accept().await.is_ok() {
                    counter.lock().unwrap().connections += 1;
                }
            });
            let record = seen.clone();
            // tungstenite's `Callback` fixes the `Result<Response, ErrorResponse>` signature.
            #[allow(clippy::result_large_err)]
            let callback = move |request: &Request, response: Response| {
                let mut s = record.lock().unwrap();
                s.path = request.uri().to_string();
                s.headers = request
                    .headers()
                    .iter()
                    .map(|(k, v)| {
                        (
                            k.as_str().to_ascii_lowercase(),
                            v.to_str().unwrap_or_default().to_string(),
                        )
                    })
                    .collect();
                Ok(response)
            };
            if let Ok(mut socket) = tokio_tungstenite::accept_hdr_async(tcp, callback).await {
                serve(&mut socket, incoming, expected, hold, &seen).await;
            }
            // Keep counting reconnects briefly after the session.
            let _ = tokio::time::timeout(QUIET, extra).await;
        });
        Replay {
            port,
            session,
            task,
        }
    }

    /// `http://127.0.0.1:PORT`, for an `*_api_base`.
    pub fn base(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    /// Waits for the session to end and returns what the server saw.
    pub async fn finish(self) -> Session {
        let _ = tokio::time::timeout(std::time::Duration::from_secs(5), self.task).await;
        self.session.lock().unwrap().clone()
    }
}

async fn serve(
    socket: &mut WebSocketStream<TcpStream>,
    incoming: Vec<Value>,
    expected: usize,
    hold: Duration,
    seen: &Arc<Mutex<Session>>,
) {
    let mut events = incoming.into_iter();
    let held_until = tokio::time::Instant::now() + hold;
    while tokio::time::Instant::now() < held_until {
        match tokio::time::timeout_at(held_until, socket.next()).await {
            Ok(Some(Ok(frame))) => record(frame, seen),
            Ok(_) => return,
            Err(_) => break,
        }
    }
    {
        let mut s = seen.lock().unwrap();
        s.before_first_event = s.outgoing.len();
    }
    if let Some(first) = events.next() {
        let _ = socket.send(Message::text(first.to_string())).await;
    }
    let mut rest: Option<Vec<Value>> = Some(events.collect());
    loop {
        let sent = seen.lock().unwrap().outgoing.len();
        if expected > 0
            && sent >= expected
            && let Some(rest) = rest.take()
        {
            for event in rest {
                let _ = socket.send(Message::text(event.to_string())).await;
            }
        }
        match tokio::time::timeout(QUIET, socket.next()).await {
            Ok(Some(Ok(Message::Close(_)))) => {
                seen.lock().unwrap().closed_cleanly = true;
                let _ = socket.close(None).await;
                return;
            }
            Ok(Some(Ok(frame))) => record(frame, seen),
            Ok(_) => return,
            // Quiet: send what is left, or end the session once everything was sent.
            Err(_) => match rest.take() {
                Some(rest) => {
                    for event in rest {
                        let _ = socket.send(Message::text(event.to_string())).await;
                    }
                }
                None => {
                    let _ = socket.close(None).await;
                    return;
                }
            },
        }
    }
}

fn record(frame: Message, seen: &Arc<Mutex<Session>>) {
    let value = match frame {
        Message::Text(text) => sanitize(serde_json::from_str(text.as_str()).unwrap_or(Value::Null)),
        Message::Binary(data) => json!({
            "binary_sha256": hex(&Sha256::digest(&data)),
            "bytes": data.len(),
        }),
        _ => return,
    };
    seen.lock().unwrap().outgoing.push(value);
}

/// The recorded URL's path and query (`RealtimeCredentials.sanitize_url` filtered only `key`).
pub fn recorded_path(cassette: &WebsocketCassette) -> String {
    let url = reqwest::Url::parse(&cassette.url).expect("recorded URL");
    match url.query() {
        Some(query) if !query.is_empty() => format!("{}?{query}", url.path()),
        _ => url.path().to_string(),
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// `RealtimeCredentials.sanitize(value, redact_audio: true)` for the frames these cassettes hold:
/// Gemini `realtimeInput.audio.data` becomes `{ "sha256": ... }` of the decoded bytes.
pub fn sanitize(mut value: Value) -> Value {
    if let Some(data) = value.pointer_mut("/realtimeInput/audio/data")
        && let Some(encoded) = data.as_str()
    {
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .expect("strict Base64 audio");
        *data = json!({ "sha256": hex(&Sha256::digest(&bytes)) });
    }
    value
}

/// A local WebSocket peer for the transport specs (`with_server(echo:)`): echoes text and binary
/// frames when `echo`, and closes the TCP connection when the client sends a close frame.
pub struct EchoServer {
    pub url: String,
    pub task: JoinHandle<()>,
}

impl EchoServer {
    pub async fn start(echo: bool) -> EchoServer {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("ws://127.0.0.1:{}", listener.local_addr().unwrap().port());
        let task = tokio::spawn(async move {
            let Ok((tcp, _)) = listener.accept().await else {
                return;
            };
            let Ok(mut socket) = tokio_tungstenite::accept_async(tcp).await else {
                return;
            };
            while let Some(Ok(message)) = socket.next().await {
                match message {
                    Message::Text(_) | Message::Binary(_) if echo => {
                        if socket.send(message).await.is_err() {
                            break;
                        }
                    }
                    Message::Close(_) => break,
                    _ => {}
                }
            }
        });
        EchoServer { url, task }
    }

    /// `worker.join(1)`: whether the server finished within a second.
    pub async fn finished(self) -> bool {
        tokio::time::timeout(std::time::Duration::from_secs(1), self.task)
            .await
            .is_ok()
    }
}
