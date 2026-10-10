//! Ports of RubyLLM 2.1's connection specs: `transport/connection_keep_alive_spec.rb`,
//! `transport/connection_reuse_spec.rb`, `transport/connection_release_spec.rb`, the TLS rows of
//! `transport/connection_retry_spec.rb`, and `protocol/streaming_retry_spec.rb`.
//!
//! Ruby caches one Faraday connection per settings and counts sockets with a keep-alive adapter.
//! The port caches one `reqwest::Client` (whose pool keeps sockets alive) per provider, API base,
//! timeout, proxy, and tokio runtime; `KeepAlive` below is the spec's loopback server, numbering
//! the socket that served each request.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use rust_llm::message::UsageStatus;
use rust_llm::transport::{Connection, SseEvent};
use rust_llm::{Config, Context, EmbedOptions, Error, ErrorKind, JudgeOptions, Provider};
use serde_json::{Value, json};

// ---- KeepAlive::Server -----------------------------------------------------------------------

#[derive(Default)]
struct Log {
    requests: Vec<usize>,
    authorizations: Vec<String>,
}

/// A loopback HTTP/1.1 server that keeps connections open and records which one, numbered in
/// order of acceptance, served each request.
struct KeepAlive {
    url: String,
    log: Arc<Mutex<Log>>,
}

impl KeepAlive {
    fn start() -> KeepAlive {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let log = Arc::new(Mutex::new(Log::default()));
        let connections = Arc::new(AtomicUsize::new(0));
        let shared = log.clone();
        std::thread::spawn(move || {
            for socket in listener.incoming().flatten() {
                let number = connections.fetch_add(1, Ordering::SeqCst) + 1;
                let log = shared.clone();
                std::thread::spawn(move || serve(socket, number, &log));
            }
        });
        KeepAlive { url, log }
    }

    fn requests(&self) -> Vec<usize> {
        self.log.lock().unwrap().requests.clone()
    }

    fn authorizations(&self) -> Vec<String> {
        self.log.lock().unwrap().authorizations.clone()
    }
}

fn serve(socket: TcpStream, number: usize, log: &Mutex<Log>) {
    let mut writer = socket.try_clone().unwrap();
    let mut reader = BufReader::new(socket);
    let mut request_line = String::new();
    while reader.read_line(&mut request_line).unwrap_or(0) > 0 {
        let mut length = 0;
        let mut authorization = String::new();
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                break;
            }
            let (name, value) = line.split_once(':').unwrap_or((&line, ""));
            match name.trim().to_ascii_lowercase().as_str() {
                "content-length" => length = value.trim().parse().unwrap_or(0),
                "authorization" => authorization = value.trim().to_string(),
                _ => {}
            }
        }
        let mut body = vec![0; length];
        if reader.read_exact(&mut body).is_err() {
            return;
        }
        let body: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
        {
            let mut log = log.lock().unwrap();
            log.requests.push(number);
            log.authorizations.push(authorization);
        }
        let path = request_line.split_whitespace().nth(1).unwrap_or_default();
        let response = if path.ends_with("/systemone") {
            let answers: serde_json::Map<String, Value> = body["questions"]
                .as_object()
                .map(|q| {
                    q.keys()
                        .map(|id| (id.clone(), json!({ "type": "noul", "noul": 0.79 })))
                        .collect()
                })
                .unwrap_or_default();
            json!({ "model": body["model"], "answers": answers,
                    "usage": { "input_tokens": 9, "output_tokens": 1 } })
        } else {
            json!({ "data": [{ "index": 0, "embedding": [0.1, 0.2] }], "model": body["model"],
                    "usage": { "prompt_tokens": 2, "total_tokens": 2 } })
        }
        .to_string();
        let head = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
            response.len()
        );
        if writer.write_all(head.as_bytes()).is_err()
            || writer.write_all(response.as_bytes()).is_err()
        {
            return;
        }
        request_line.clear();
    }
}

fn context(server: &KeepAlive, options: &[(&str, &str)]) -> Context {
    let mut config = Config::default();
    config.set("openai_api_key", "test");
    config.set("openai_api_base", format!("{}/v1", server.url));
    config.set("typesafe_api_key", "test");
    config.set("typesafe_api_base", server.url.clone());
    for (option, value) in options {
        config.set(*option, *value);
    }
    Context::new(config)
}

async fn embed(context: &Context) {
    context
        .embed(
            "Ruby",
            EmbedOptions {
                model: Some("text-embedding-3-small"),
                ..Default::default()
            },
        )
        .await
        .unwrap();
}

/// The spec's `hold_until_all_arrive`: a server answering each connection on its own thread,
/// where every request waits until `together` requests have arrived, so they are in flight at
/// once. `respond` gives the content type and body for a request body.
fn in_flight_together(
    together: usize,
    respond: fn(&str, &[u8]) -> (&'static str, String),
) -> (String, Arc<Mutex<Vec<String>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let barrier = Arc::new(std::sync::Barrier::new(together));
    let authorizations = Arc::new(Mutex::new(Vec::new()));
    let seen = authorizations.clone();
    std::thread::spawn(move || {
        for socket in listener.incoming().flatten() {
            let (barrier, seen) = (barrier.clone(), seen.clone());
            std::thread::spawn(move || {
                let mut writer = socket.try_clone().unwrap();
                let mut reader = BufReader::new(socket);
                let mut request_line = String::new();
                reader.read_line(&mut request_line).unwrap();
                let (mut length, mut authorization) = (0, String::new());
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                        break;
                    }
                    let (name, value) = line.split_once(':').unwrap_or((&line, ""));
                    match name.trim().to_ascii_lowercase().as_str() {
                        "content-length" => length = value.trim().parse().unwrap_or(0),
                        "authorization" => authorization = value.trim().to_string(),
                        _ => {}
                    }
                }
                let mut body = vec![0; length];
                reader.read_exact(&mut body).unwrap();
                seen.lock().unwrap().push(authorization);
                barrier.wait();
                let path = request_line.split_whitespace().nth(1).unwrap_or_default();
                let (content_type, response) = respond(path, &body);
                let head = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    response.len()
                );
                let _ = writer.write_all(head.as_bytes());
                let _ = writer.write_all(response.as_bytes());
            });
        }
    });
    (url, authorizations)
}

// ---- transport/connection_keep_alive_spec.rb -------------------------------------------------

// spec: transport/connection_keep_alive_spec.rb:27 with a keep-alive adapter > reuses one socket across consecutive calls
#[tokio::test]
async fn reuses_one_socket_across_consecutive_calls() {
    let server = KeepAlive::start();
    let llm = context(&server, &[]);
    for _ in 0..3 {
        embed(&llm).await;
    }
    assert_eq!(server.requests(), [1, 1, 1]);
}

// spec: transport/connection_keep_alive_spec.rb:35 with a keep-alive adapter > reuses one socket across consecutive judgments
#[tokio::test]
async fn reuses_one_socket_across_consecutive_judgments() {
    let server = KeepAlive::start();
    let llm = context(&server, &[]);
    for _ in 0..2 {
        llm.judge(
            "RubyLLM calls AI APIs.",
            json!({ "keep_alive_expected": { "type": "probability", "instructions": "Is the socket reused?" } }),
            JudgeOptions {
                model: Some(Some("jev-latest".into())),
                provider: Some("typesafe".into()),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    }
    assert_eq!(server.requests(), [1, 1]);
}

// spec: transport/connection_keep_alive_spec.rb:43 with a keep-alive adapter > shares the socket between contexts that only differ in credentials
#[tokio::test]
async fn shares_the_socket_between_contexts_that_only_differ_in_credentials() {
    let server = KeepAlive::start();
    let tenant_a = context(&server, &[("openai_api_key", "tenant-a")]);
    let tenant_b = context(&server, &[("openai_api_key", "tenant-b")]);
    for llm in [&tenant_a, &tenant_b, &tenant_a] {
        embed(llm).await;
    }
    assert_eq!(server.requests(), [1, 1, 1]);
    assert_eq!(
        server.authorizations(),
        ["Bearer tenant-a", "Bearer tenant-b", "Bearer tenant-a"]
    );
}

// spec: transport/connection_keep_alive_spec.rb:53 with a keep-alive adapter > opens another socket for a context with other connection settings
#[tokio::test]
async fn opens_another_socket_for_a_context_with_other_connection_settings() {
    let server = KeepAlive::start();
    embed(&context(&server, &[])).await;
    let mut other = Config::clone(context(&server, &[]).config());
    other.request_timeout = std::time::Duration::from_secs(30);
    embed(&Context::new(other)).await;
    assert_eq!(server.requests(), [1, 2]);
}

// spec: transport/connection_keep_alive_spec.rb:60 with a keep-alive adapter > shares the adapter across threads
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn shares_the_client_across_threads() {
    let server = KeepAlive::start();
    let llm = Arc::new(context(&server, &[]));
    let tasks: Vec<_> = (0..4)
        .map(|_| {
            let llm = llm.clone();
            tokio::spawn(async move {
                for _ in 0..2 {
                    embed(&llm).await;
                }
            })
        })
        .collect();
    for task in tasks {
        task.await.unwrap();
    }
    // Ruby's adapter serializes requests over its one socket. reqwest's pool opens a socket per
    // request in flight at once, and a thread's second call can start just before its first
    // socket is back in the pool, so the exact count varies with scheduling. What the spec
    // protects is that the threads share one client: sockets are reused across the eight
    // calls (one client per call would open eight), and a later call reuses a pooled socket.
    let requests = server.requests();
    assert_eq!(requests.len(), 8);
    let opened = *requests.iter().max().unwrap();
    assert!(opened < 8, "every call opened its own socket: {requests:?}");
    embed(&llm).await;
    let after = server.requests();
    assert!(
        *after.last().unwrap() <= opened,
        "a later call opened a new socket: {after:?}"
    );
}

// spec: transport/connection_keep_alive_spec.rb:68 with a keep-alive adapter > shares the adapter across fibers
#[tokio::test]
async fn shares_the_client_across_tasks_on_one_thread() {
    let server = KeepAlive::start();
    let llm = context(&server, &[]);
    for _ in 0..4 {
        embed(&llm).await;
    }
    let tasks = futures::future::join_all((0..4).map(|_| embed(&llm)));
    tasks.await;
    let requests = server.requests();
    assert_eq!(&requests[..4], [1, 1, 1, 1]);
    // In flight together, they share the pool: the kept-alive socket plus at most three more.
    assert!(
        requests[4..].iter().all(|n| (1..=4).contains(n)),
        "{requests:?}"
    );
}

// spec: transport/connection_keep_alive_spec.rb:76 with a keep-alive adapter > never sends a forked child through the parent socket
// The port's counterpart of a forked child is another tokio runtime: a pooled socket belongs to
// the runtime that opened it, so the cache keys clients by runtime and a new runtime opens its
// own socket; the first runtime keeps reusing its own.
#[test]
fn never_sends_another_runtime_through_a_socket_of_the_first() {
    let server = KeepAlive::start();
    let runtime = || {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
    };
    let llm = context(&server, &[]);
    let parent = runtime();
    parent.block_on(embed(&llm));
    let child = runtime();
    child.block_on(async {
        embed(&llm).await;
        embed(&llm).await;
    });
    drop(child);
    parent.block_on(embed(&llm));
    assert_eq!(server.requests(), [1, 2, 2, 1]);
}

// ---- transport/connection_reuse_spec.rb -------------------------------------------------------

fn openai(config: &Config) -> Connection {
    Connection::new(Provider::OpenAI, Arc::new(config.clone())).unwrap()
}

/// Ruby compares the cached Faraday connections with `be`; the port's client is opaque, so the
/// keep-alive server tells: connections that share a client send consecutive requests over one
/// socket, and separate clients open their own.
async fn sockets_used(connections: &[Connection]) {
    for connection in connections {
        connection
            .post("embeddings", &json!({ "model": "m" }), &[], &mut |_| {})
            .await
            .unwrap();
    }
}

fn keep_alive_config(server: &KeepAlive) -> Config {
    let mut config = Config::default();
    config.set("openai_api_base", format!("{}/v1", server.url));
    config.set("xai_api_base", format!("{}/v1", server.url));
    config
}

// spec: transport/connection_reuse_spec.rb:21 keying > shares one Faraday connection between providers built from equal settings
#[tokio::test]
async fn shares_one_client_between_providers_built_from_equal_settings() {
    let server = KeepAlive::start();
    let config = keep_alive_config(&server);
    sockets_used(&[openai(&config), openai(&config.clone())]).await;
    assert_eq!(server.requests(), [1, 1]);
}

// spec: transport/connection_reuse_spec.rb:25 keying > shares it between configurations that only differ in credentials
#[tokio::test]
async fn shares_it_between_configurations_that_only_differ_in_credentials() {
    let server = KeepAlive::start();
    let mut tenant_a = keep_alive_config(&server);
    tenant_a.set("openai_api_key", "tenant-a");
    let mut tenant_b = keep_alive_config(&server);
    tenant_b.set("openai_api_key", "tenant-b");
    sockets_used(&[openai(&tenant_a), openai(&tenant_b)]).await;
    assert_eq!(server.requests(), [1, 1]);
    assert_eq!(
        server.authorizations(),
        ["Bearer tenant-a", "Bearer tenant-b"]
    );
}

// spec: transport/connection_reuse_spec.rb:31 keying > builds a separate connection for each setting the connection depends on
// The settings `Connection.basic` builds a client from are the API base, `request_timeout`, and
// `http_proxy` (retries run in the port's own loop and read each connection's config). The
// keep-alive server doubles as the proxy, which receives absolute-form request targets.
/// A setting's name and how to change it on a config, given the test server's URL.
type SettingChange = (&'static str, fn(&mut Config, &str));

#[tokio::test]
async fn builds_a_separate_client_for_each_setting_the_client_depends_on() {
    let changes: [SettingChange; 3] = [
        ("openai_api_base", |c, url| {
            c.set("openai_api_base", format!("{url}/other/v1"));
        }),
        ("request_timeout", |c, _| {
            c.request_timeout = std::time::Duration::from_secs(7);
        }),
        ("http_proxy", |c, url| c.http_proxy = Some(url.to_string())),
    ];
    for (option, change) in changes {
        let server = KeepAlive::start();
        let config = keep_alive_config(&server);
        let mut changed = config.clone();
        change(&mut changed, &server.url);
        sockets_used(&[openai(&config), openai(&changed)]).await;
        assert_eq!(
            server.requests(),
            [1, 2],
            "expected {option} to separate clients"
        );
    }
}

// spec: transport/connection_reuse_spec.rb:43 keying > builds a separate connection for each provider
#[tokio::test]
async fn builds_a_separate_client_for_each_provider() {
    let server = KeepAlive::start();
    let same_base = keep_alive_config(&server);
    let xai = Connection::new(Provider::XAI, Arc::new(same_base.clone())).unwrap();
    sockets_used(&[openai(&same_base), xai]).await;
    assert_eq!(server.requests(), [1, 2]);
}

/// A stream and a buffered request, one after the other, over two connections built from the
/// same settings.
async fn buffered_then_streamed() -> Vec<Value> {
    let server = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .respond_with(
            wiremock::ResponseTemplate::new(200)
                .set_body_raw("data: {\"x\":1}\n\n", "text/event-stream"),
        )
        .mount(&server)
        .await;
    let mut config = Config::default();
    config.set("openai_api_base", server.uri());
    openai(&config)
        .post("chat/completions", &json!({}), &[], &mut |_| {})
        .await
        .unwrap();
    let mut seen = Vec::new();
    openai(&config.clone())
        .stream(
            "chat/completions",
            &json!({}),
            &[],
            &mut |_| {},
            &mut |_e: SseEvent, data: Value| {
                seen.push(data);
                Ok(())
            },
            |_| None,
        )
        .await
        .unwrap();
    seen
}

// spec: transport/connection_reuse_spec.rb:56 keying > gives streaming requests a shared connection of their own
// Ruby caches a second connection for streams because some Faraday adapters (httpx) only stream
// if their first request did. A reqwest client streams any response, so streams share the one
// client; this checks what the split protects: a stream works on a client that already served a
// buffered request.
#[tokio::test]
async fn streams_share_the_client_with_buffered_requests() {
    assert_eq!(buffered_then_streamed().await, [json!({ "x": 1 })]);
}

// spec: transport/connection_reuse_spec.rb:63 keying > streams after buffered requests on adapters that settle on streaming at their first request
#[tokio::test]
async fn streams_after_buffered_requests_over_the_shared_client() {
    assert_eq!(buffered_then_streamed().await, [json!({ "x": 1 })]);
}

// spec: transport/connection_reuse_spec.rb:183 requests sharing a connection > parse errors with the provider that sent them
#[tokio::test]
async fn parse_errors_with_the_provider_that_sent_them() {
    let server = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .respond_with(wiremock::ResponseTemplate::new(400).set_body_raw(
            r#"{"error":{"message":"Invalid input"}}"#,
            "application/json",
        ))
        .mount(&server)
        .await;
    // Ruby subclasses OpenAI to prefix each tenant's key; the port's providers are a closed enum,
    // so two providers that parse errors differently share one base instead: OpenRouter reads
    // `error.metadata.raw`, OpenAI does not.
    let mut config = Config::default();
    config.set("openai_api_base", server.uri());
    config.set("openrouter_api_base", server.uri());
    config.max_retries = 0;
    let openai = openai(&config);
    let openrouter = Connection::new(Provider::OpenRouter, Arc::new(config)).unwrap();
    for connection in [&openai, &openrouter] {
        let error = connection
            .post("embeddings", &json!({}), &[], &mut |_| {})
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::BadRequest);
        assert_eq!(error.to_string(), "Invalid input");
    }
}

// spec: transport/connection_reuse_spec.rb:199 requests sharing a connection > keep credentials and usage apart while they are in flight together
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn keep_credentials_and_usage_apart_while_they_are_in_flight_together() {
    let (url, authorizations) = in_flight_together(2, |_, body| {
        let input = serde_json::from_slice::<Value>(body).unwrap()["input"]
            .as_str()
            .unwrap()
            .len();
        let response = json!({ "data": [{ "embedding": [input as f64] }],
                               "usage": { "prompt_tokens": input, "total_tokens": input } });
        ("application/json", response.to_string())
    });
    let tasks: Vec<_> = [("tenant-a", "Ruby"), ("tenant-bb", "Rails on Ruby")]
        .into_iter()
        .map(|(key, text)| {
            let mut config = Config::default();
            config.set("openai_api_key", key);
            config.set("openai_api_base", url.clone());
            let llm = Context::new(config);
            tokio::spawn(async move {
                llm.embed(
                    text,
                    EmbedOptions {
                        model: Some("text-embedding-3-small"),
                        ..Default::default()
                    },
                )
                .await
                .unwrap()
            })
        })
        .collect();
    let mut embeddings = Vec::new();
    for task in tasks {
        embeddings.push(task.await.unwrap());
    }
    assert_eq!(
        embeddings
            .iter()
            .map(|e| e.input_tokens)
            .collect::<Vec<_>>(),
        [Some(4), Some(13)]
    );
    assert_eq!(
        embeddings
            .iter()
            .map(|e| e.vectors.clone())
            .collect::<Vec<_>>(),
        [
            rust_llm::embedding::Vectors::Single(vec![4.0]),
            rust_llm::embedding::Vectors::Single(vec![13.0])
        ]
    );
    let mut authorizations = authorizations.lock().unwrap().clone();
    authorizations.sort();
    assert_eq!(authorizations, ["Bearer tenant-a", "Bearer tenant-bb"]);
}

// spec: transport/connection_reuse_spec.rb:219 requests sharing a connection > keep the responses of many fibers apart in one reactor
#[tokio::test]
async fn keep_the_responses_of_many_tasks_apart_on_one_thread() {
    let server = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::path("/embeddings"))
        .respond_with(|request: &wiremock::Request| {
            let input = serde_json::from_slice::<Value>(&request.body).unwrap()["input"]
                .as_str()
                .unwrap()
                .len();
            wiremock::ResponseTemplate::new(200)
                .set_delay(std::time::Duration::from_millis(rand::random::<u64>() % 5))
                .set_body_json(json!({
                    "data": [{ "embedding": [input as f64] }],
                    "usage": { "prompt_tokens": input, "total_tokens": input }
                }))
        })
        .mount(&server)
        .await;
    wiremock::Mock::given(wiremock::matchers::path("/chat/completions"))
        .respond_with(|request: &wiremock::Request| {
            let body = serde_json::from_slice::<Value>(&request.body).unwrap();
            let content = body["messages"].as_array().unwrap().last().unwrap()["content"].clone();
            let delta = json!({ "choices": [{ "delta": { "content": content } }] });
            wiremock::ResponseTemplate::new(200)
                .set_delay(std::time::Duration::from_millis(rand::random::<u64>() % 5))
                .set_body_raw(
                    format!("data: {delta}\n\ndata: [DONE]\n\n"),
                    "text/event-stream",
                )
        })
        .mount(&server)
        .await;
    let mut config = Config::default();
    config.set("openai_api_key", "test");
    config.set("openai_api_base", server.uri());
    let llm = Context::new(config);
    let results = futures::future::join_all((0..20).map(|index| {
        let llm = &llm;
        async move {
            let text = format!("fiber {index}");
            if index % 2 == 1 {
                let embedding = llm
                    .embed(
                        text.as_str(),
                        EmbedOptions {
                            model: Some("text-embedding-3-small"),
                            ..Default::default()
                        },
                    )
                    .await
                    .unwrap();
                return json!(embedding.input_tokens);
            }
            let mut chunks = String::new();
            llm.chat(Some("gpt-4.1-nano"), Some("openai"))
                .unwrap()
                .with_protocol(rust_llm::ProtocolName::ChatCompletions)
                .ask_stream(text, |chunk| chunks.push_str(chunk.content()))
                .await
                .unwrap();
            json!(chunks)
        }
    }))
    .await;
    let expected: Vec<Value> = (0..20)
        .map(|i| {
            let text = format!("fiber {i}");
            if i % 2 == 1 {
                json!(text.len())
            } else {
                json!(text)
            }
        })
        .collect();
    assert_eq!(results, expected);
}

// spec: transport/connection_reuse_spec.rb:251 requests sharing a connection > stream to their own handlers while they are in flight together
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stream_to_their_own_handlers_while_they_are_in_flight_together() {
    let (url, _) = in_flight_together(2, |_, body| {
        (
            "text/event-stream",
            format!("data: {}\n\n", String::from_utf8_lossy(body)),
        )
    });
    let mut config = Config::default();
    config.set("openai_api_base", url);
    let transport = openai(&config);
    let tasks: Vec<_> = ["alpha", "beta"]
        .into_iter()
        .map(|name| {
            let transport = transport.clone();
            tokio::spawn(async move {
                let mut chunks = Vec::new();
                transport
                    .stream(
                        "embeddings",
                        &json!({ "name": name }),
                        &[],
                        &mut |_| {},
                        &mut |_e: SseEvent, data: Value| {
                            chunks.push(data);
                            Ok(())
                        },
                        |_| None,
                    )
                    .await
                    .unwrap();
                chunks
            })
        })
        .collect();
    let mut streams = Vec::new();
    for task in tasks {
        streams.push(task.await.unwrap());
    }
    assert_eq!(
        streams,
        [
            vec![json!({ "name": "alpha" })],
            vec![json!({ "name": "beta" })]
        ]
    );
}

// ---- transport/connection_release_spec.rb -----------------------------------------------------

// spec: transport/connection_release_spec.rb:23 returns a response that keeps no copy of the request body
#[tokio::test]
async fn returns_a_response_that_keeps_no_copy_of_the_request_body() {
    let server = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .respond_with(
            wiremock::ResponseTemplate::new(200).set_body_raw(r#"{"ok":true}"#, "application/json"),
        )
        .mount(&server)
        .await;
    let mut config = Config::default();
    config.set("openai_api_base", server.uri());
    let history = "history ".repeat(1000);
    let response = openai(&config)
        .post(
            "chat/completions",
            &json!({ "messages": [{ "role": "user", "content": history }] }),
            &[],
            &mut |_| {},
        )
        .await
        .unwrap();
    assert!(response.request_body.is_empty());
    assert_eq!(response.request_body_json(), Value::Null);
    assert_eq!(response.status, 200);
    assert_eq!(response.body, json!({ "ok": true }));
}

// ---- transport/connection_retry_spec.rb (TLS) -------------------------------------------------

/// A TLS server whose first `failures` connections are closed before the handshake (the client
/// sees an unexpected EOF during the handshake, as Ruby's stubbed `OpenSSL::SSL::SSLError`
/// reports), then answers. Returns its base URL and the attempt counter.
fn flaky_tls(failures: usize, body: &'static str) -> (String, Arc<AtomicUsize>) {
    let dir = format!("{}/tests/fixtures/websocket", env!("CARGO_MANIFEST_DIR"));
    let cert_der = std::fs::read(format!("{dir}/untrusted-cert.der")).unwrap();
    let key_der = std::fs::read(format!("{dir}/untrusted-key.der")).unwrap();
    let tls = rustls::ServerConfig::builder_with_provider(Arc::new(
        rustls::crypto::ring::default_provider(),
    ))
    .with_safe_default_protocol_versions()
    .unwrap()
    .with_no_client_auth()
    .with_single_cert(
        vec![rustls::pki_types::CertificateDer::from(cert_der)],
        rustls::pki_types::PrivateKeyDer::Pkcs8(key_der.into()),
    )
    .unwrap();
    let tls = Arc::new(tls);
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let attempts = Arc::new(AtomicUsize::new(0));
    let counter = attempts.clone();
    std::thread::spawn(move || {
        for socket in listener.incoming().flatten() {
            let attempt = counter.fetch_add(1, Ordering::SeqCst);
            if attempt < failures {
                drop(socket);
                continue;
            }
            let tls = tls.clone();
            std::thread::spawn(move || {
                let mut connection = rustls::ServerConnection::new(tls).unwrap();
                let mut socket = socket;
                let mut stream = rustls::Stream::new(&mut connection, &mut socket);
                let length = {
                    let mut reader = BufReader::new(&mut stream);
                    let mut length = 0;
                    loop {
                        let mut line = String::new();
                        if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                            break;
                        }
                        if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                            length = v.trim().parse().unwrap_or(0);
                        }
                    }
                    let mut request = vec![0; length];
                    let _ = reader.read_exact(&mut request);
                    length
                };
                let _ = length;
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(response.as_bytes());
                let _ = stream.flush();
                connection.send_close_notify();
                let _ = connection.complete_io(&mut socket);
            });
        }
    });
    (format!("https://127.0.0.1:{port}"), attempts)
}

/// A connection to `base` over a client that accepts the fixture's self-signed certificate (a
/// trust setting `Config` does not carry, so the client comes in through `with_client`).
fn tls_connection(provider: Provider, base: &str) -> Connection {
    let mut config = Config::default();
    config.set(format!("{}_api_base", provider.slug()), base);
    config.set(format!("{}_api_key", provider.slug()), "test-key");
    config.max_retries = 3;
    config.retry_interval = 0.0;
    config.retry_interval_randomness = 0.0;
    let client = reqwest::Client::builder()
        .danger_accept_invalid_certs(true)
        .build()
        .unwrap();
    Connection::with_client(provider, Arc::new(config), client)
}

// spec: transport/connection_retry_spec.rb:132 rate limit retry timing > retries a chat completion whose TLS connection fails
#[tokio::test]
async fn retries_a_chat_completion_whose_tls_connection_fails() {
    let (base, attempts) = flaky_tls(1, "{}");
    let connection = tls_connection(Provider::OpenAI, &base);
    let response = connection
        .post("chat/completions", &json!({}), &[], &mut |_| {})
        .await
        .unwrap();
    assert_eq!(response.status, 200);
    assert_eq!(attempts.load(Ordering::SeqCst), 2);
}

// spec: transport/connection_retry_spec.rb:202 job-creating requests > submits a batch once when the first attempt fails with a TLS error
#[tokio::test]
async fn submits_a_job_once_when_the_first_attempt_fails_with_a_tls_error() {
    let (base, attempts) = flaky_tls(
        1,
        r#"{"id":"msgbatch_01","processing_status":"in_progress"}"#,
    );
    let connection = tls_connection(Provider::Anthropic, &base);
    // `create_batch` posts with `idempotent: false`; the batch builder takes a whole Chat, so the
    // non-idempotent send is driven directly, as `Connection#post(..., idempotent: false)`.
    let error = connection
        .send(
            reqwest::Method::POST,
            "v1/messages/batches",
            &[],
            false,
            &|req| req.json(&json!({ "requests": [] })),
        )
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::ConnectionFailed, "{error:?}");
    assert_eq!(attempts.load(Ordering::SeqCst), 1);
}

// ---- protocol/streaming_retry_spec.rb ---------------------------------------------------------

/// One scripted attempt: the reads the server sends, then whether it cuts the connection.
struct Attempt {
    reads: Vec<String>,
    cut: bool,
}

fn successful_reads() -> Vec<String> {
    let completed = json!({ "type": "response.completed", "response": {
        "model": "gpt-4.1-nano", "status": "completed", "error": null,
        "usage": { "input_tokens": 10, "output_tokens": 7 } } });
    vec![
        format!(
            "data: {}\n\n",
            json!({ "type": "response.output_text.delta", "delta": "Hello" })
        ),
        "event: response.completed\ndata: ".into(),
        format!("{completed}\n\n"),
    ]
}

/// The spec's scripted adapter: a server that answers each attempt with its reads as separate
/// chunks of a chunked 200, cutting the connection mid-body when the attempt says so (Ruby
/// raises `Faraday::ConnectionFailed`/`TimeoutError` after the reads).
fn scripted(attempts: Vec<Attempt>) -> (String, Arc<AtomicUsize>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let count = Arc::new(AtomicUsize::new(0));
    let counter = count.clone();
    std::thread::spawn(move || {
        let mut attempts = attempts.into_iter();
        for mut socket in listener.incoming().flatten() {
            let Some(attempt) = attempts.next() else {
                return;
            };
            counter.fetch_add(1, Ordering::SeqCst);
            let mut reader = BufReader::new(socket.try_clone().unwrap());
            let mut length = 0;
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                    break;
                }
                if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                    length = v.trim().parse().unwrap_or(0);
                }
            }
            let mut body = vec![0; length];
            let _ = reader.read_exact(&mut body);
            let _ = socket.write_all(
                b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\nconnection: close\r\n\r\n",
            );
            for read in &attempt.reads {
                let _ = socket.write_all(format!("{:x}\r\n{read}\r\n", read.len()).as_bytes());
                let _ = socket.flush();
                std::thread::sleep(std::time::Duration::from_millis(30));
            }
            if !attempt.cut {
                let _ = socket.write_all(b"0\r\n\r\n");
            }
            let _ = socket.shutdown(std::net::Shutdown::Both);
        }
    });
    (base, count)
}

fn interrupted(reads: &[&str]) -> Attempt {
    Attempt {
        reads: reads.iter().map(|r| r.to_string()).collect(),
        cut: true,
    }
}

fn complete(reads: Vec<String>) -> Attempt {
    Attempt { reads, cut: false }
}

fn streaming_chat(base: &str, attempts: usize) -> rust_llm::Chat {
    let mut config = Config::default();
    config.set("openai_api_key", "test");
    config.set("openai_api_base", base);
    config.max_retries = attempts as u32 - 1;
    config.retry_interval = 0.0;
    config.retry_interval_randomness = 0.0;
    Context::new(config)
        .chat(Some("gpt-4.1-nano"), Some("openai"))
        .unwrap()
        .with_protocol(rust_llm::ProtocolName::Responses)
}

// spec: protocol/streaming_retry_spec.rb:63 when streaming #{mode} > delivers text and usage after an interrupted JSON body
#[tokio::test]
async fn delivers_text_and_usage_after_an_interrupted_json_body() {
    let (base, attempts) = scripted(vec![interrupted(&["{\n"]), complete(successful_reads())]);
    let mut chunks = Vec::new();
    let message = streaming_chat(&base, 2)
        .ask_stream("Hello", |chunk| {
            if !chunk.content().is_empty() {
                chunks.push(chunk.content().to_string());
            }
        })
        .await
        .unwrap();
    assert_eq!(attempts.load(Ordering::SeqCst), 2);
    assert_eq!(chunks, ["Hello"]);
    assert_eq!(message.content(), "Hello");
    assert_eq!(message.tokens().output, Some(7));
    assert_eq!(
        message
            .usage_entries
            .iter()
            .map(|e| e.status)
            .collect::<Vec<_>>(),
        [UsageStatus::Failed, UsageStatus::Succeeded]
    );
    // `STREAM_RESET_KEY` released: the port's raw response keeps no request (connection_release).
    assert!(message.raw.as_ref().unwrap().request_body.is_empty());
}

// spec: protocol/streaming_retry_spec.rb:76 when streaming #{mode} > discards a partially parsed SSE event before retrying
#[tokio::test]
async fn discards_a_partially_parsed_sse_event_before_retrying() {
    let (base, attempts) = scripted(vec![
        interrupted(&[": keepalive\n\ndata: {\"type\":"]),
        complete(successful_reads()),
    ]);
    let message = streaming_chat(&base, 2)
        .ask_stream("Hello", |_| {})
        .await
        .unwrap();
    assert_eq!(attempts.load(Ordering::SeqCst), 2);
    assert_eq!(message.content(), "Hello");
    assert_eq!(message.tokens().output, Some(7));
}

fn service_busy() -> Attempt {
    complete(vec![
        r#"{"error":{"message":"Service busy","type":"server_error"}}"#.into(),
    ])
}

// spec: protocol/streaming_retry_spec.rb:86 when streaming #{mode} > raises a JSON error returned after an interrupted SSE event
#[tokio::test]
async fn raises_a_json_error_returned_after_an_interrupted_sse_event() {
    let (base, attempts) = scripted(vec![
        interrupted(&[": keepalive\n\ndata: {\"type\":"]),
        service_busy(),
    ]);
    let error = streaming_chat(&base, 2)
        .ask_stream("Hello", |_| panic!("No chunks should be delivered"))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Server, "{error:?}");
    assert_eq!(error.to_string(), "Service busy");
    assert_eq!(attempts.load(Ordering::SeqCst), 2);
}

// spec: protocol/streaming_retry_spec.rb:95 when streaming #{mode} > raises a JSON error returned after an interrupted JSON body
#[tokio::test]
async fn raises_a_json_error_returned_after_an_interrupted_json_body() {
    let (base, attempts) = scripted(vec![interrupted(&["{\"error\":"]), service_busy()]);
    let error = streaming_chat(&base, 2)
        .ask_stream("Hello", |_| panic!("No chunks should be delivered"))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Server, "{error:?}");
    assert_eq!(error.to_string(), "Service busy");
    assert_eq!(attempts.load(Ordering::SeqCst), 2);
}

// spec: protocol/streaming_retry_spec.rb:104 when streaming #{mode} > starts fresh after successive interrupted bodies
#[tokio::test]
async fn starts_fresh_after_successive_interrupted_bodies() {
    let (base, attempts) = scripted(vec![
        interrupted(&["{\n"]),
        interrupted(&[": keepalive\n\ndata: {\"type\":"]),
        complete(successful_reads()),
    ]);
    let message = streaming_chat(&base, 3)
        .ask_stream("Hello", |_| {})
        .await
        .unwrap();
    assert_eq!(attempts.load(Ordering::SeqCst), 3);
    assert_eq!(message.content(), "Hello");
    assert_eq!(message.tokens().output, Some(7));
    assert_eq!(
        message
            .usage_entries
            .iter()
            .map(|e| e.status)
            .collect::<Vec<_>>(),
        [
            UsageStatus::Failed,
            UsageStatus::Failed,
            UsageStatus::Succeeded
        ]
    );
}

// spec: protocol/streaming_retry_spec.rb:116 when streaming #{mode} > does not retry a connection failure after delivering text
#[tokio::test]
async fn does_not_retry_a_connection_failure_after_delivering_text() {
    let first = successful_reads().remove(0);
    let (base, attempts) = scripted(vec![
        Attempt {
            reads: vec![first],
            cut: true,
        },
        complete(successful_reads()),
    ]);
    let mut chunks = Vec::new();
    let error = streaming_chat(&base, 2)
        .ask_stream("Hello", |chunk| chunks.push(chunk.content().to_string()))
        .await
        .unwrap_err();
    assert!(matches!(error, Error::ConnectionFailed(_)), "{error:?}");
    assert_eq!(attempts.load(Ordering::SeqCst), 1);
    assert_eq!(chunks, ["Hello"]);
}
