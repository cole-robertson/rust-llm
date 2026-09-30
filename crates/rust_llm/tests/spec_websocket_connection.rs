//! `spec/ruby_llm/transport/websocket_connection_spec.rb`: the WebSocket transport against real
//! local peers (tokio-tungstenite servers, a raw TCP peer that writes frames byte by byte, and a
//! rustls server with a self-signed certificate).

#[path = "support/websocket.rs"]
mod websocket_support;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use rust_llm::transport::WebsocketConnection;
use rust_llm::{Config, Error};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use websocket_support::EchoServer;

fn config() -> Config {
    Config::default()
}

fn message(error: &Error) -> String {
    error.to_string()
}

// spec: transport/websocket_connection_spec.rb:41 performs a real WebSocket handshake and exchanges frames
#[tokio::test]
async fn performs_a_real_websocket_handshake_and_exchanges_frames() {
    let server = EchoServer::start(true).await;
    let echoed = WebsocketConnection::open(&server.url, &[], &config(), async |connection| {
        connection.send_text("Hello Rust").await?;
        connection.read(None).await
    })
    .await
    .unwrap();
    assert_eq!(echoed.as_deref(), Some(&b"Hello Rust"[..]));
}

// spec: transport/websocket_connection_spec.rb:50 closes the socket when the caller raises
#[tokio::test]
async fn closes_the_socket_when_the_caller_raises() {
    let server = EchoServer::start(true).await;
    let url = server.url.clone();
    let result: rust_llm::Result<()> =
        WebsocketConnection::open(&url, &[], &config(), async |_connection| {
            Err(Error::Api("Stopped".into(), None))
        })
        .await;
    assert_eq!(message(&result.unwrap_err()), "Stopped");
    assert!(server.finished().await, "the peer saw the connection close");
}

// spec: transport/websocket_connection_spec.rb:59 reads available frames without waiting when the socket has no complete message
#[tokio::test]
async fn reads_available_frames_without_waiting() {
    let server = EchoServer::start(true).await;
    WebsocketConnection::open(&server.url, &[], &config(), async |connection| {
        let started = Instant::now();
        assert_eq!(connection.read_available().await?, None);
        assert!(started.elapsed() < Duration::from_millis(100));

        connection.send_text("Queued cancellation").await?;
        let deadline = Instant::now() + Duration::from_secs(1);
        let mut received = None;
        while received.is_none() && Instant::now() < deadline {
            received = connection.read_available().await?;
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(received.as_deref(), Some(&b"Queued cancellation"[..]));
        assert_eq!(connection.read_available().await?, None);
        Ok(())
    })
    .await
    .unwrap();
}

/// A peer that completes the handshake by hand and then writes the bytes it is given, so a
/// frame can arrive in pieces.
async fn raw_peer() -> (String, tokio::sync::mpsc::Sender<Vec<u8>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("ws://127.0.0.1:{}", listener.local_addr().unwrap().port());
    let (tx, mut rx) = tokio::sync::mpsc::channel::<Vec<u8>>(8);
    tokio::spawn(async move {
        let (mut tcp, _) = listener.accept().await.unwrap();
        let mut request = Vec::new();
        let mut buf = [0u8; 1024];
        while !request.windows(4).any(|w| w == b"\r\n\r\n") {
            let n = tcp.read(&mut buf).await.unwrap();
            request.extend_from_slice(&buf[..n]);
        }
        let request = String::from_utf8_lossy(&request).to_string();
        let key = request
            .lines()
            .find_map(|l| {
                let (name, value) = l.split_once(':')?;
                name.eq_ignore_ascii_case("sec-websocket-key")
                    .then(|| value.trim().to_string())
            })
            .unwrap();
        let accept = tokio_tungstenite::tungstenite::handshake::derive_accept_key(key.as_bytes());
        let response = format!(
            "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: {accept}\r\n\r\n"
        );
        tcp.write_all(response.as_bytes()).await.unwrap();
        while let Some(bytes) = rx.recv().await {
            tcp.write_all(&bytes).await.unwrap();
            tcp.flush().await.unwrap();
        }
        let mut sink = Vec::new();
        let _ = tokio::time::timeout(Duration::from_millis(200), tcp.read_to_end(&mut sink)).await;
    });
    (url, tx)
}

// spec: transport/websocket_connection_spec.rb:75 keeps partial frames buffered without blocking or losing their bytes
#[tokio::test]
async fn keeps_partial_frames_buffered_without_blocking_or_losing_their_bytes() {
    let (url, peer) = raw_peer().await;
    WebsocketConnection::open(&url, &[], &config(), async |connection| {
        peer.send(b"\x81\x05he".to_vec()).await.unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        let started = Instant::now();
        assert_eq!(connection.read_available().await?, None);
        assert!(started.elapsed() < Duration::from_millis(100));

        peer.send(b"llo".to_vec()).await.unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(
            connection.read_available().await?.as_deref(),
            Some(&b"hello"[..])
        );
        Ok(())
    })
    .await
    .unwrap();
}

// spec: transport/websocket_connection_spec.rb:88 preserves arbitrary binary frame bytes
#[tokio::test]
async fn preserves_arbitrary_binary_frame_bytes() {
    let server = EchoServer::start(true).await;
    let echoed = WebsocketConnection::open(&server.url, &[], &config(), async |connection| {
        connection.send_binary(b"\xFF\x00\x80".to_vec()).await?;
        connection.read(None).await
    })
    .await
    .unwrap();
    assert_eq!(echoed.as_deref(), Some(&b"\xFF\x00\x80"[..]));
}

/// Sets its flag when dropped: the writer future is gone.
struct Alive(Arc<AtomicBool>);

impl Drop for Alive {
    fn drop(&mut self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

// spec: transport/websocket_connection_spec.rb:97 stops the writer when the connection closes before an acknowledgement
#[tokio::test]
async fn stops_the_writer_when_the_connection_closes_before_an_acknowledgement() {
    let server = EchoServer::start(true).await;
    let alive = Arc::new(AtomicBool::new(false));
    let flag = alive.clone();
    let result = WebsocketConnection::open(&server.url, &[], &config(), async |connection| {
        let write = async {
            flag.store(true, Ordering::SeqCst);
            let _guard = Alive(flag.clone());
            connection
                .send_text("Waiting for an acknowledgement")
                .await?;
            std::future::pending::<()>().await;
            Ok(())
        };
        connection
            .each_message(write, |_message| {
                connection.close();
                Ok(())
            })
            .await
    })
    .await;
    let error = result.unwrap_err();
    assert!(
        message(&error).contains("before outgoing messages finished"),
        "{error}"
    );
    assert!(!alive.load(Ordering::SeqCst), "the writer is stopped");
}

// spec: transport/websocket_connection_spec.rb:114 rejects an untrusted TLS certificate
#[tokio::test]
async fn rejects_an_untrusted_tls_certificate() {
    let fixture = |name: &str| {
        std::fs::read(format!(
            "{}/tests/fixtures/websocket/{name}",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap()
    };
    let cert = rustls::pki_types::CertificateDer::from(fixture("untrusted-cert.der"));
    let key = rustls::pki_types::PrivateKeyDer::Pkcs8(fixture("untrusted-key.der").into());
    let tls = rustls::ServerConfig::builder_with_provider(Arc::new(
        rustls::crypto::ring::default_provider(),
    ))
    .with_safe_default_protocol_versions()
    .unwrap()
    .with_no_client_auth()
    .with_single_cert(vec![cert], key)
    .unwrap();
    let acceptor = tokio_rustls::TlsAcceptor::from(Arc::new(tls));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let server = tokio::spawn(async move {
        let (tcp, _) = listener.accept().await.unwrap();
        acceptor.accept(tcp).await.is_ok()
    });

    let result: rust_llm::Result<()> = WebsocketConnection::open(
        &format!("wss://127.0.0.1:{port}"),
        &[],
        &config(),
        async |_| Ok(()),
    )
    .await;
    let error = result.unwrap_err();
    assert!(matches!(error, Error::ConnectionFailed(_)), "{error:?}");
    assert!(
        message(&error).contains("invalid peer certificate: UnknownIssuer"),
        "{error}"
    );
    assert!(!server.await.unwrap(), "the TLS handshake never completed");
}

// spec: transport/websocket_connection_spec.rb:144 bounds a read when the peer sends no frames
#[tokio::test]
async fn bounds_a_read_when_the_peer_sends_no_frames() {
    let server = EchoServer::start(false).await;
    let mut config = config();
    config.request_timeout = Duration::from_millis(50);
    let result: rust_llm::Result<()> =
        WebsocketConnection::open(&server.url, &[], &config, async |connection| {
            connection.read(None).await.map(|_| ())
        })
        .await;
    let error = result.unwrap_err();
    assert!(message(&error).contains("timed out"), "{error}");
}

// spec: transport/websocket_connection_spec.rb:153 ends a blocked read when another thread closes the connection
#[tokio::test]
async fn ends_a_blocked_read_when_another_task_closes_the_connection() {
    let server = EchoServer::start(false).await;
    WebsocketConnection::open(&server.url, &[], &config(), async |connection| {
        let reader = connection.read(None);
        let closer = async {
            tokio::time::sleep(Duration::from_millis(50)).await;
            connection.close();
        };
        let (read, ()) = tokio::time::timeout(Duration::from_secs(1), async {
            tokio::join!(reader, closer)
        })
        .await
        .expect("the blocked read ends within a second");
        assert_eq!(read?, None);
        Ok(())
    })
    .await
    .unwrap();
}

// spec: transport/websocket_connection_spec.rb:173 rejects a configured proxy before opening a socket
#[tokio::test]
async fn rejects_a_configured_proxy_before_opening_a_socket() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("ws://127.0.0.1:{}", listener.local_addr().unwrap().port());
    let mut config = config();
    config.http_proxy = Some("socks5://localhost:1080".into());

    let result: rust_llm::Result<()> =
        WebsocketConnection::open(&url, &[], &config, async |_| Ok(())).await;
    let error = result.unwrap_err();
    assert!(
        matches!(&error, Error::Argument(m) if m.contains("do not support HTTP proxies")),
        "{error:?}"
    );
    let accepted = tokio::time::timeout(Duration::from_millis(100), listener.accept()).await;
    assert!(accepted.is_err(), "no socket was opened");
}

// spec: transport/websocket_connection_spec.rb:189 rejects unsupported URL schemes and URL credentials
#[test]
fn rejects_unsupported_url_schemes_and_url_credentials() {
    let error = WebsocketConnection::new("https://example.com", &[], &config()).unwrap_err();
    assert!(
        matches!(&error, Error::Argument(m) if m.contains("must use ws or wss")),
        "{error:?}"
    );
    let error = WebsocketConnection::new("wss://secret@example.com", &[], &config()).unwrap_err();
    assert!(
        matches!(&error, Error::Argument(m) if m.contains("must not contain credentials")),
        "{error:?}"
    );
    assert!(WebsocketConnection::new("wss://example.com/path?x=1", &[], &config()).is_ok());
}

// spec: transport/websocket_connection_spec.rb:196 fails if the peer closes without a WebSocket close frame
#[tokio::test]
async fn fails_if_the_peer_closes_without_a_websocket_close_frame() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("ws://127.0.0.1:{}", listener.local_addr().unwrap().port());
    tokio::spawn(async move {
        let (tcp, _) = listener.accept().await.unwrap();
        drop(tcp);
    });

    let result: rust_llm::Result<()> =
        WebsocketConnection::open(&url, &[], &config(), async |_| Ok(())).await;
    let error = result.unwrap_err();
    assert!(message(&error).contains("without a close frame"), "{error}");
}

#[tokio::test]
async fn reports_a_peer_that_drops_an_open_connection_without_a_close_frame() {
    let (url, peer) = raw_peer().await;
    let result: rust_llm::Result<()> =
        WebsocketConnection::open(&url, &[], &config(), async |connection| {
            drop(peer);
            connection.read(None).await.map(|_| ())
        })
        .await;
    let error = result.unwrap_err();
    assert!(message(&error).contains("without a close frame"), "{error}");
}
