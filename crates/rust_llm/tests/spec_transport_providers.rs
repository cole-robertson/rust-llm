//! Ports of RubyLLM's transport and provider-configuration specs: `transport/connection_retry_spec.rb`,
//! `transport/error_middleware_spec.rb`, `context_connection_spec.rb`, `provider_spec.rb`, and the
//! provider-config rows of `providers/{openai,openrouter,perplexity,anthropic,hetzner,ollama,ollama_cloud}`.
//! Ruby stubs HTTP with webmock; here a wiremock server answers, and for the `http_proxy` examples
//! it stands in as the proxy itself (a plain-HTTP proxy receives absolute-form request targets, so
//! the recorded URL keeps the original host).

mod spec_helpers;

use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use rust_llm::providers::{self, ProtocolName, Provider};
use rust_llm::transport::{Connection, apply_retry_delay};
use rust_llm::{
    AnimateOptions, Attachment, Chat, Config, Context, Error, ErrorKind, FileOptions, PaintOptions,
    UploadOptions, files::UploadedFile,
};
use serde_json::{Value, json};
use spec_helpers::Sequence;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn headers(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

fn responses_text(text: &str) -> Value {
    json!({ "id": "resp_1", "object": "response", "status": "completed", "model": "gpt-4.1-nano",
            "output": [{ "type": "message", "role": "assistant", "content": [{ "type": "output_text", "text": text }] }],
            "usage": { "input_tokens": 1, "output_tokens": 1 } })
}

/// Everything the context-connection examples fetch: chat, image and video endpoints (OpenAI and
/// xAI under `/v1`), and the hosted files they point at under `cdn`.
async fn mount_media(server: &MockServer, cdn: &str) {
    let mocks = [
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_json(responses_text("ok"))),
        Mock::given(method("GET"))
            .and(path("/notes.txt"))
            .respond_with(ResponseTemplate::new(200).set_body_string("hello notes")),
        Mock::given(method("POST"))
            .and(path("/v1/images/generations"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(json!({ "data": [{ "url": format!("{cdn}/out.png") }] })),
            ),
        Mock::given(method("GET"))
            .and(path("/out.png"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(b"png-bytes".to_vec())),
        Mock::given(method("POST"))
            .and(path("/v1/videos/generations"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(json!({ "request_id": "vid_1" })),
            ),
        Mock::given(method("GET"))
            .and(path("/v1/videos/vid_1"))
            .respond_with(ResponseTemplate::new(200).set_body_json(
                json!({ "status": "done", "video": { "url": format!("{cdn}/clip.mp4") } }),
            )),
        Mock::given(method("GET"))
            .and(path("/clip.mp4"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(b"mp4-bytes".to_vec())),
    ];
    for mock in mocks {
        mock.mount(server).await;
    }
}

/// `RubyLLM.context { |c| c.http_proxy = ...; c.request_timeout = 7 }`, with the wiremock server
/// as the proxy. Provider bases and hosted files use hosts that only the proxy can answer for.
async fn proxied_context() -> (Context, MockServer) {
    let server = MockServer::start().await;
    mount_media(&server, "http://cdn.example.test").await;
    let mut config = Config::default();
    config.http_proxy = Some(server.uri());
    config.request_timeout = Duration::from_secs(7);
    for provider in ["openai", "xai"] {
        config.set(format!("{provider}_api_key"), "test");
        config.set(format!("{provider}_api_base"), "http://api.example.test/v1");
    }
    (Context::new(config), server)
}

/// The global configuration for the "falls back to the global configuration" examples: set once
/// for this test binary (so `rust_llm::config()` stays the same `Arc`), pointed at a server that
/// lives on its own thread for the whole run.
fn global_server() -> &'static str {
    static URI: OnceLock<String> = OnceLock::new();
    URI.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("runtime");
            runtime.block_on(async move {
                let server = MockServer::start().await;
                mount_media(&server, &server.uri()).await;
                tx.send(server.uri()).expect("send uri");
                futures::future::pending::<()>().await;
            });
        });
        let uri = rx.recv().expect("global server");
        rust_llm::configure(|c| {
            for provider in ["openai", "xai"] {
                c.set(format!("{provider}_api_key"), "test");
                c.set(format!("{provider}_api_base"), format!("{uri}/v1"));
            }
        });
        uri
    })
}

/// Whether the proxy server saw a request for `host` + `path` (absolute-form, so it was proxied).
async fn proxied(server: &MockServer, host: &str, path: &str) -> bool {
    server
        .received_requests()
        .await
        .unwrap_or_default()
        .iter()
        .any(|r| r.url.host_str() == Some(host) && r.url.path() == path)
}

// ---- context_connection_spec.rb --------------------------------------------------------------

// spec: context_connection_spec.rb:23 RubyLLM::Transport::Connection > builds a basic connection from the configuration it is given
#[tokio::test]
async fn a_basic_connection_uses_the_proxy_and_timeout_of_its_configuration() {
    let (context, server) = proxied_context().await;
    let client = rust_llm::transport::basic(context.config()).expect("client");

    assert!(format!("{client:?}").contains(": 7s"), "{client:?}");
    let body = client
        .get("http://files.example.test/notes.txt")
        .send()
        .await
        .expect("proxied")
        .text()
        .await
        .unwrap();
    assert_eq!(body, "hello notes");
    assert!(proxied(&server, "files.example.test", "/notes.txt").await);
    assert_eq!(
        context.config().http_proxy.as_deref(),
        Some(server.uri().as_str())
    );
}

// spec: context_connection_spec.rb:27 RubyLLM::Transport::Connection > falls back to the global configuration
#[tokio::test]
async fn a_basic_connection_falls_back_to_the_global_configuration() {
    let global = rust_llm::config();
    let client = rust_llm::transport::basic(&global).expect("client");

    assert!(
        format!("{client:?}").contains(&format!(": {:?}", global.request_timeout)),
        "{client:?}"
    );
    assert_eq!(global.request_timeout, Config::default().request_timeout);
    assert_eq!(global.http_proxy, None);
}

// spec: context_connection_spec.rb:53 RubyLLM::Chat > gives a URL attachment the configuration of the context it was built from
#[tokio::test]
async fn a_context_chat_downloads_url_attachments_through_the_context_proxy() {
    let (context, server) = proxied_context().await;
    let mut chat = context
        .chat(Some("gpt-4.1-nano"), Some("openai"))
        .expect("chat");
    chat.ask_later_with(
        "What is this?",
        vec![Attachment::new("http://files.example.test/notes.txt")],
    )
    .expect("stage");

    chat.complete().await.expect("complete");

    assert!(
        proxied(&server, "files.example.test", "/notes.txt").await,
        "the attachment was fetched through the proxy"
    );
    assert!(Arc::ptr_eq(chat.config(), context.config()));
}

// spec: context_connection_spec.rb:60 RubyLLM::Chat > leaves a chat without a context on the global configuration
#[tokio::test]
async fn a_plain_chat_downloads_url_attachments_with_the_global_configuration() {
    let uri = global_server();
    let mut chat = Chat::new(Some("gpt-4.1-nano"), Some("openai")).expect("chat");
    chat.ask_later_with(
        "What is this?",
        vec![Attachment::new(format!("{uri}/notes.txt"))],
    )
    .expect("stage");

    chat.complete().await.expect("complete");

    assert!(Arc::ptr_eq(chat.config(), &rust_llm::config()));
    let sent = chat.messages()[0].attachments[0]
        .content_text()
        .expect("downloaded");
    assert_eq!(sent, "hello notes");
}

// spec: context_connection_spec.rb:78 RubyLLM::Image > downloads a hosted image through the configuration that generated it
#[tokio::test]
async fn a_hosted_image_downloads_through_the_generating_context() {
    let (context, server) = proxied_context().await;
    let options = PaintOptions {
        model: Some("gpt-image-1"),
        provider: Some("openai"),
        size: Some("1024x1024"),
        ..Default::default()
    };
    let image = context
        .paint("a small watercolor robot", options)
        .await
        .expect("paint")
        .into_image();

    assert_eq!(
        image.config().http_proxy.as_deref(),
        Some(server.uri().as_str())
    );
    assert_eq!(image.config().request_timeout, Duration::from_secs(7));
    assert_eq!(image.to_blob().await.expect("download"), b"png-bytes");
    assert!(proxied(&server, "cdn.example.test", "/out.png").await);
}

// spec: context_connection_spec.rb:82 RubyLLM::Image > downloads through the global configuration by default
#[tokio::test]
async fn a_hosted_image_downloads_through_the_global_configuration_by_default() {
    global_server();
    let options = PaintOptions {
        model: Some("gpt-image-1"),
        provider: Some("openai"),
        size: Some("1024x1024"),
        ..Default::default()
    };
    let image = rust_llm::paint("a small watercolor robot", options)
        .await
        .expect("paint")
        .into_image();

    assert!(Arc::ptr_eq(&image.config(), &rust_llm::config()));
    assert_eq!(image.to_blob().await.expect("download"), b"png-bytes");
}

// spec: context_connection_spec.rb:95 RubyLLM::Video > downloads a hosted video through the configuration that generated it
#[tokio::test]
async fn a_hosted_video_downloads_through_the_generating_context() {
    let (context, server) = proxied_context().await;
    let options = AnimateOptions {
        model: Some("grok-imagine-video"),
        provider: Some("xai"),
        ..Default::default()
    };
    let mut job = context
        .animate_later(Some("a wave"), options)
        .await
        .expect("submit");
    job.refresh().await.expect("refresh");
    let video = job.video().await.expect("video").expect("completed");

    assert_eq!(
        video.config().http_proxy.as_deref(),
        Some(server.uri().as_str())
    );
    assert_eq!(video.config().request_timeout, Duration::from_secs(7));
    assert_eq!(video.to_blob().await.expect("download"), b"mp4-bytes");
    assert!(proxied(&server, "cdn.example.test", "/clip.mp4").await);
}

// spec: context_connection_spec.rb:99 RubyLLM::Video > downloads through the global configuration by default
#[tokio::test]
async fn a_hosted_video_downloads_through_the_global_configuration_by_default() {
    global_server();
    let options = AnimateOptions {
        model: Some("grok-imagine-video"),
        provider: Some("xai"),
        ..Default::default()
    };
    let mut job = rust_llm::animate_later(Some("a wave"), options)
        .await
        .expect("submit");
    job.refresh().await.expect("refresh");
    let video = job.video().await.expect("video").expect("completed");

    assert!(Arc::ptr_eq(&video.config(), &rust_llm::config()));
    assert_eq!(video.to_blob().await.expect("download"), b"mp4-bytes");
}

// ---- providers/openai_spec.rb #retry_delay ---------------------------------------------------

fn openai_delay(pairs: &[(&str, &str)]) -> Option<f64> {
    Provider::OpenAI.retry_delay(&headers(pairs))
}

// spec: providers/openai_spec.rb:48 #retry_delay > parses duration-formatted reset headers
#[test]
fn openai_retry_delay_parses_duration_formatted_reset_headers() {
    assert_eq!(
        openai_delay(&[("x-ratelimit-reset-requests", "6m0s")]),
        Some(360.0)
    );
}

// spec: providers/openai_spec.rb:54 #retry_delay > parses fractional seconds
#[test]
fn openai_retry_delay_parses_fractional_seconds() {
    assert_eq!(
        openai_delay(&[("x-ratelimit-reset-requests", "7.66s")]),
        Some(7.66)
    );
}

// spec: providers/openai_spec.rb:60 #retry_delay > parses milliseconds
#[test]
fn openai_retry_delay_parses_milliseconds() {
    assert_eq!(
        openai_delay(&[("x-ratelimit-reset-tokens", "76ms")]),
        Some(0.076)
    );
}

// spec: providers/openai_spec.rb:66 #retry_delay > parses hours
#[test]
fn openai_retry_delay_parses_hours() {
    assert_eq!(
        openai_delay(&[("x-ratelimit-reset-tokens", "1h2m3s")]),
        Some(3723.0)
    );
}

// spec: providers/openai_spec.rb:72 #retry_delay > returns the longer wait when both limits are hit
#[test]
fn openai_retry_delay_returns_the_longer_wait() {
    assert_eq!(
        openai_delay(&[
            ("x-ratelimit-reset-requests", "1s"),
            ("x-ratelimit-reset-tokens", "2m30s")
        ]),
        Some(150.0)
    );
}

// spec: providers/openai_spec.rb:81 #retry_delay > returns nil without rate limit headers
#[test]
fn openai_retry_delay_is_none_without_rate_limit_headers() {
    assert_eq!(openai_delay(&[("content-type", "application/json")]), None);
}

// spec: providers/openai_spec.rb:85 #retry_delay > returns nil for unparseable values
#[test]
fn openai_retry_delay_is_none_for_unparseable_values() {
    assert_eq!(
        openai_delay(&[("x-ratelimit-reset-requests", "soon")]),
        None
    );
}

// spec: providers/openai_spec.rb:91 #retry_delay > returns nil when a duration contains trailing text
#[test]
fn openai_retry_delay_is_none_with_trailing_text() {
    assert_eq!(
        openai_delay(&[("x-ratelimit-reset-requests", "1s later")]),
        None
    );
}

// spec: providers/openai_spec.rb:97 #retry_delay > returns nil when the response has no headers
#[test]
fn openai_retry_delay_is_none_without_headers() {
    assert_eq!(Provider::OpenAI.retry_delay(&[]), None);
}

// ---- providers/openrouter/parse_error_spec.rb ------------------------------------------------

fn openrouter(body: Value) -> Option<String> {
    Provider::OpenRouter.parse_error(&body.to_string())
}

// spec: providers/openrouter/parse_error_spec.rb:13 #parse_error > appends nested provider message from metadata.raw when present
#[test]
fn openrouter_appends_the_upstream_message_from_metadata_raw() {
    let raw = json!({ "error": { "code": "unsupported_country_region_territory",
                                 "message": "Country, region, or territory not supported", "type": "request_forbidden" } });
    let body = json!({ "error": { "message": "Provider returned error", "code": 403,
                                  "metadata": { "raw": raw.to_string(), "provider_name": "OpenAI" } },
                       "user_id": "user_2" });
    assert_eq!(
        openrouter(body).as_deref(),
        Some("Provider returned error - Country, region, or territory not supported")
    );
}

// spec: providers/openrouter/parse_error_spec.rb:53 #parse_error > joins a list of errors
#[test]
fn openrouter_joins_a_list_of_errors() {
    let body = json!([{ "error": { "message": "first" } }, { "error": { "message": "second" } }]);
    assert_eq!(openrouter(body).as_deref(), Some("first. second"));
}

// spec: providers/openrouter/parse_error_spec.rb:65 #parse_error > skips empty list entries
#[test]
fn openrouter_skips_empty_list_entries() {
    let body = json!([null, "", { "error": [] }, { "error": { "message": "second" } }]);
    assert_eq!(openrouter(body).as_deref(), Some("second"));
}

// spec: providers/openrouter/parse_error_spec.rb:74 #parse_error > passes a body it cannot interpret through
#[test]
fn openrouter_passes_an_uninterpretable_body_through() {
    assert_eq!(openrouter(json!(42)).as_deref(), Some("42"));
}

// spec: providers/openrouter/parse_error_spec.rb:83 #parse_error > ignores a raw payload that is not a JSON object
#[test]
fn openrouter_ignores_a_raw_payload_that_is_not_json() {
    let body = json!({ "error": { "message": "Provider returned error", "metadata": { "raw": "not json" } } });
    assert_eq!(openrouter(body).as_deref(), Some("Provider returned error"));
}

// spec: providers/openrouter/parse_error_spec.rb:98 #parse_error > joins messages from an array error value
#[test]
fn openrouter_joins_messages_from_an_array_error_value() {
    let body = json!({ "error": [{ "message": "first failure" }, "second failure"] });
    assert_eq!(
        openrouter(body).as_deref(),
        Some("first failure. second failure")
    );
}

// spec: providers/openrouter/parse_error_spec.rb:107 #parse_error > ignores non-object metadata
#[test]
fn openrouter_ignores_non_object_metadata() {
    let body = json!({ "error": { "message": "Provider returned error", "metadata": "raw text" } });
    assert_eq!(openrouter(body).as_deref(), Some("Provider returned error"));
}

// spec: providers/openrouter/parse_error_spec.rb:116 #parse_error > handles a string error nested in metadata.raw
#[test]
fn openrouter_handles_a_string_error_nested_in_metadata_raw() {
    let body = json!({ "error": { "message": "Provider returned error",
                                  "metadata": { "raw": json!({ "error": "upstream detail" }).to_string() } } });
    assert_eq!(
        openrouter(body).as_deref(),
        Some("Provider returned error - upstream detail")
    );
}

// ---- providers/perplexity/models_spec.rb error parsing ---------------------------------------

// spec: providers/perplexity/models_spec.rb:88 error parsing > reads the title out of the HTML Perplexity returns for auth failures
#[test]
fn perplexity_reads_the_html_title() {
    let html = "<html>\n<head><title>401 Authorization Required</title></head>\n</html>";
    assert_eq!(
        Provider::Perplexity.parse_error(html).as_deref(),
        Some("Authorization Required")
    );
}

// spec: providers/perplexity/models_spec.rb:94 error parsing > falls back to the shared parser for HTML without a title match
#[test]
fn perplexity_falls_back_for_html_without_a_title() {
    let html = "<html><title></title>no title here</html>";
    assert_eq!(
        Provider::Perplexity.parse_error(html).as_deref(),
        Some(html)
    );
}

// spec: providers/perplexity/models_spec.rb:104 error parsing > is nil for an empty body
#[test]
fn perplexity_parse_error_is_none_for_an_empty_body() {
    assert_eq!(Provider::Perplexity.parse_error(""), None);
}

// ---- transport/connection_retry_spec.rb ------------------------------------------------------

/// `rate limit retry timing`'s config: one retry, no backoff interval, no jitter.
fn retry_config(server: &MockServer) -> Config {
    let mut config = Config::default();
    config.set("openai_api_key", "test-key");
    config.set("openai_api_base", format!("{}/v1", server.uri()));
    config.max_retries = 1;
    config.retry_interval = 0.0;
    config.retry_interval_randomness = 0.0;
    config
}

async fn serve_in_order(templates: Vec<ResponseTemplate>) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(Sequence(Mutex::new(templates.into())))
        .mount(&server)
        .await;
    server
}

fn ok() -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(json!({}))
}

async fn post(config: Config) -> rust_llm::Result<rust_llm::message::RawResponse> {
    let connection = Connection::new(Provider::OpenAI, Arc::new(config)).expect("connection");
    connection
        .post("chat/completions", &json!({}), &[], &mut |_| {})
        .await
}

async fn requests(server: &MockServer) -> usize {
    server.received_requests().await.unwrap_or_default().len()
}

// spec: transport/connection_retry_spec.rb:51 retry middleware configuration > does not retry a POST the caller marked non-idempotent
#[tokio::test]
async fn a_non_idempotent_post_is_not_retried() {
    let server = serve_in_order(vec![
        ResponseTemplate::new(500).set_body_string(r#"{"error":{"message":"boom"}}"#),
        ok(),
    ])
    .await;
    let mut config = retry_config(&server);
    config.max_retries = 3;
    let connection = Connection::new(Provider::OpenAI, Arc::new(config)).expect("connection");

    let result = connection
        .send(
            reqwest::Method::POST,
            "chat/completions",
            &[],
            false,
            &|req| req.json(&json!({})),
        )
        .await;

    assert_eq!(result.unwrap_err().kind(), ErrorKind::Server);
    assert_eq!(requests(&server).await, 1);
}

// spec: transport/connection_retry_spec.rb:61 retry middleware configuration > caps retry delays at retry_max_interval
#[tokio::test]
async fn retry_backoff_is_capped_at_retry_max_interval() {
    assert_eq!(Config::default().retry_max_interval, 30.0);
    let server = serve_in_order(vec![ResponseTemplate::new(500).set_body_string("{}"), ok()]).await;
    let mut config = retry_config(&server);
    config.retry_interval = 10.0;
    config.retry_max_interval = 0.2;
    let started = Instant::now();

    post(config).await.expect("retried");

    let waited = started.elapsed();
    assert!(
        waited >= Duration::from_millis(200) && waited < Duration::from_secs(5),
        "waited {waited:?}"
    );
    assert_eq!(requests(&server).await, 2);
}

// spec: transport/connection_retry_spec.rb:91 rate limit retry timing > retries after the provider supplies a delay
#[tokio::test]
async fn a_rate_limit_retries_after_the_provider_reset_delay() {
    let limited = ResponseTemplate::new(429)
        .insert_header("x-ratelimit-reset-requests", "0ms")
        .set_body_string(r#"{"error":{"message":"Rate limit reached"}}"#);
    let server = serve_in_order(vec![limited, ok()]).await;

    let response = post(retry_config(&server)).await.expect("retried");

    assert_eq!(response.status, 200);
    assert_eq!(requests(&server).await, 2);
}

// spec: transport/connection_retry_spec.rb:106 rate limit retry timing > gives up immediately when the provider asks to wait longer than retry_max_interval
#[tokio::test]
async fn a_provider_reset_beyond_retry_max_interval_is_not_waited_out() {
    let limited = ResponseTemplate::new(429)
        .insert_header("x-ratelimit-reset-requests", "6m0s")
        .set_body_string(r#"{"error":{"message":"Rate limit reached"}}"#);
    let server = serve_in_order(vec![limited, ok()]).await;
    let mut config = retry_config(&server);
    config.retry_max_interval = 5.0;

    let error = post(config).await.unwrap_err();

    assert_eq!(error.kind(), ErrorKind::RateLimit);
    assert_eq!(requests(&server).await, 1);
}

// spec: transport/connection_retry_spec.rb:119 rate limit retry timing > retries a chat completion that fails with a server error
#[tokio::test]
async fn a_chat_completion_is_retried_after_a_server_error() {
    let failed = ResponseTemplate::new(500)
        .set_body_string(r#"{"error":{"message":"Internal server error"}}"#);
    let server = serve_in_order(vec![failed, ok()]).await;

    let response = post(retry_config(&server)).await.expect("retried");

    assert_eq!(response.status, 200);
    assert_eq!(requests(&server).await, 2);
}

// spec: transport/connection_retry_spec.rb:133 rate limit retry timing > honors millisecond retry delays for HTTP #{status}
#[tokio::test]
async fn millisecond_retry_delays_are_honored() {
    for status in [429, 500, 503, 529] {
        let failed = ResponseTemplate::new(status)
            .insert_header("retry-after-ms", "1500")
            .set_body_string(r#"{"error":{"message":"Try again later"}}"#);
        let server = serve_in_order(vec![failed, ok()]).await;
        let started = Instant::now();

        assert_eq!(
            post(retry_config(&server)).await.expect("retried").status,
            200
        );

        // Ruby records the sleep as exactly [1.5]; here the wait is measured.
        let waited = started.elapsed();
        assert!(
            waited >= Duration::from_millis(1500) && waited < Duration::from_secs(4),
            "{status}: waited {waited:?}"
        );
        assert_eq!(requests(&server).await, 2);
    }
}

// spec: transport/connection_retry_spec.rb:146 rate limit retry timing > does not retry before an excessive millisecond delay has elapsed
#[tokio::test]
async fn an_excessive_millisecond_delay_is_not_retried() {
    let failed = ResponseTemplate::new(529)
        .insert_header("retry-after-ms", "6000")
        .set_body_string(r#"{"error":{"message":"Overloaded"}}"#);
    let server = serve_in_order(vec![failed, ok()]).await;
    let mut config = retry_config(&server);
    config.retry_max_interval = 5.0;

    let error = post(config).await.unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Overloaded);
    assert_eq!(requests(&server).await, 1);
}

fn anthropic_batch_config(server: &MockServer) -> Arc<Config> {
    let mut config = Config::default();
    config.set("anthropic_api_key", "test-key");
    config.set("anthropic_api_base", server.uri());
    config.max_retries = 3;
    config.retry_interval = 0.0;
    config.retry_interval_randomness = 0.0;
    Arc::new(config)
}

fn created_batch() -> ResponseTemplate {
    ResponseTemplate::new(200)
        .set_body_json(json!({ "id": "msgbatch_01", "processing_status": "in_progress" }))
}

async fn submit_one(config: Arc<Config>) -> rust_llm::Result<rust_llm::Batch> {
    let mut chat = Chat::with_config(config, Some("claude-haiku-4-5"), Some("anthropic"), false)
        .expect("chat");
    chat.ask_later("hi").expect("stage");
    rust_llm::batch(vec![chat]).await
}

// spec: transport/connection_retry_spec.rb:175 job-creating requests > submits a batch once when the first attempt fails with a server error
#[tokio::test]
async fn a_batch_is_submitted_once_after_a_server_error() {
    let failed = ResponseTemplate::new(500)
        .set_body_string(r#"{"error":{"message":"Internal server error"}}"#);
    let server = serve_in_order(vec![failed, created_batch()]).await;

    let error = submit_one(anthropic_batch_config(&server))
        .await
        .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Server);
    assert_eq!(requests(&server).await, 1);
}

// spec: transport/connection_retry_spec.rb:183 job-creating requests > submits a batch once when the first attempt times out
#[tokio::test]
async fn a_batch_is_submitted_once_after_a_timeout() {
    let server = serve_in_order(vec![
        created_batch().set_delay(Duration::from_secs(3)),
        created_batch(),
    ])
    .await;
    let mut config = (*anthropic_batch_config(&server)).clone();
    config.request_timeout = Duration::from_millis(500);

    let error = submit_one(Arc::new(config)).await.unwrap_err();

    // webmock's `to_timeout` surfaces as Faraday::ConnectionFailed; a real timeout here is Timeout.
    // Both are transport failures the retry middleware would otherwise retry.
    assert_eq!(error.kind(), ErrorKind::Timeout, "{error:?}");
    assert_eq!(requests(&server).await, 1);
}

// ---- transport/error_middleware_spec.rb ------------------------------------------------------

fn retry_after(provider: Option<Provider>, status: u16, pairs: &[(&str, &str)]) -> Option<String> {
    let mut headers = headers(pairs);
    apply_retry_delay(provider, status, &mut headers);
    headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("retry-after"))
        .map(|(_, v)| v.clone())
}

/// The Ruby examples stub `retry_delay` on a provider double; OpenAI's reset header is the port's
/// way to make a provider report a delay (`"12.5s"` => 12.5, `"2m"` => 120.0).
const RESET: &str = "x-ratelimit-reset-requests";

// spec: transport/error_middleware_spec.rb:84 retry delay normalization > copies the provider retry delay into Retry-After for the retry middleware
#[test]
fn the_provider_retry_delay_becomes_retry_after() {
    assert_eq!(
        retry_after(Some(Provider::OpenAI), 429, &[(RESET, "12.5s")]).as_deref(),
        Some("12.5")
    );
}

// spec: transport/error_middleware_spec.rb:92 retry delay normalization > keeps a Retry-After already sent by the provider
#[test]
fn a_retry_after_sent_by_the_provider_is_kept() {
    let date = "Wed, 21 Oct 2099 07:28:00 GMT";
    let headers = &[
        ("Retry-After", date),
        ("retry-after-ms", "1500"),
        (RESET, "2m"),
    ];
    assert_eq!(
        retry_after(Some(Provider::OpenAI), 429, headers).as_deref(),
        Some(date)
    );
}

// spec: transport/error_middleware_spec.rb:101 retry delay normalization > normalizes millisecond retry delays for HTTP #{status} without a provider
#[test]
fn millisecond_retry_delays_are_normalized_without_a_provider() {
    for status in [429, 500, 503, 529] {
        assert_eq!(
            retry_after(None, status, &[("retry-after-ms", "1500.5")]).as_deref(),
            Some("1.5005"),
            "{status}"
        );
    }
}

// spec: transport/error_middleware_spec.rb:110 retry delay normalization > preserves a zero millisecond delay ahead of provider reset hints
#[test]
fn a_zero_millisecond_delay_wins_over_provider_reset_hints() {
    assert_eq!(
        retry_after(
            Some(Provider::OpenAI),
            429,
            &[("retry-after-ms", "0"), (RESET, "2m")]
        )
        .as_deref(),
        Some("0.0")
    );
}

// spec: transport/error_middleware_spec.rb:119 retry delay normalization > falls back to provider timing when retry-after-ms is #{value}
#[test]
fn invalid_millisecond_delays_fall_back_to_provider_timing() {
    for value in ["invalid", "-1000", "NaN", "Infinity", "1e999"] {
        let delay = retry_after(
            Some(Provider::OpenAI),
            429,
            &[("retry-after-ms", value), (RESET, "12.5s")],
        );
        assert_eq!(delay.as_deref(), Some("12.5"), "{value}");
    }
}

/// `ErrorMiddleware.parse_error` for a response with `status` and `{"error":{"message": message}}`.
async fn failure(status: u16, message: &str) -> Error {
    let server = MockServer::start().await;
    let body = json!({ "error": { "message": message } }).to_string();
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(status).set_body_string(body))
        .mount(&server)
        .await;
    let mut config = retry_config(&server);
    config.max_retries = 0;
    post(config).await.unwrap_err()
}

// spec: transport/error_middleware_spec.rb:171 .parse_error > maps context-length-like 429 errors to ContextLengthExceededError
#[tokio::test]
async fn a_request_too_large_429_is_context_length_exceeded() {
    assert_eq!(
        failure(429, "Request too large for model").await.kind(),
        ErrorKind::ContextLengthExceeded
    );
}

// spec: transport/error_middleware_spec.rb:189 .parse_error > maps context-length-like 400 errors to ContextLengthExceededError
#[tokio::test]
async fn a_maximum_context_length_400_is_context_length_exceeded() {
    let error = failure(400, "This model's maximum context length is 8192 tokens.").await;
    assert_eq!(error.kind(), ErrorKind::ContextLengthExceeded);
}

// spec: transport/error_middleware_spec.rb:209 .parse_error > keeps an invalid context-size setting as BadRequestError
#[tokio::test]
async fn an_invalid_context_size_setting_stays_a_bad_request() {
    let msg = "Invalid context size: must be a positive integer";
    let error = failure(400, msg).await;
    assert_eq!(error.kind(), ErrorKind::BadRequest);
    assert_eq!(error.to_string(), msg);
}

// spec: transport/error_middleware_spec.rb:229 .parse_error > maps Anthropic's 'input length and max_tokens exceed context limit' 400 error to ContextLengthExceededError
#[tokio::test]
async fn an_exceeded_context_limit_400_is_context_length_exceeded() {
    let error = failure(
        400,
        "input length and max_tokens exceed context limit: 900000 + 128000 > 1000000",
    )
    .await;
    assert_eq!(error.kind(), ErrorKind::ContextLengthExceeded);
}

// spec: transport/error_middleware_spec.rb:239 .parse_error > maps a currently overloaded 400 to OverloadedError
#[tokio::test]
async fn a_currently_overloaded_400_is_overloaded() {
    let msg = "Our servers are currently overloaded. Please try again later.";
    let error = failure(400, msg).await;
    assert_eq!(error.kind(), ErrorKind::Overloaded);
    assert_eq!(error.to_string(), msg);
}

// spec: transport/error_middleware_spec.rb:249 .parse_error > maps 'the engine is currently overloaded' 400 errors to OverloadedError
#[tokio::test]
async fn an_engine_overloaded_400_is_overloaded() {
    let msg = "The engine is currently overloaded, please try again later.";
    let error = failure(400, msg).await;
    assert_eq!(error.kind(), ErrorKind::Overloaded);
    assert_eq!(error.to_string(), msg);
}

// spec: transport/error_middleware_spec.rb:259 .parse_error > keeps a 400 that only mentions overloaded as BadRequestError
#[tokio::test]
async fn a_400_that_only_mentions_overloaded_stays_a_bad_request() {
    let msg = "Unknown parameter: overloaded";
    let error = failure(400, msg).await;
    assert_eq!(error.kind(), ErrorKind::BadRequest);
    assert_eq!(error.to_string(), msg);
}

// ---- provider_spec.rb ------------------------------------------------------------------------

/// `config_for(slug)`: the settings each provider needs to be constructed.
fn config_for(provider: Provider) -> Config {
    let mut config = Config::default();
    match provider {
        Provider::Ollama => {
            config.set("ollama_api_base", "https://ollama.example.com/v1");
            config.set("ollama_api_key", "ollama-key");
        }
        Provider::GPUStack => {
            config.set("gpustack_api_base", "https://gpustack.example.com/v1");
        }
        other => {
            config.set(
                format!("{}_api_key", other.slug()),
                format!("{}-key", other.slug()),
            );
        }
    }
    config
}

// spec: provider_spec.rb:174 #parse_error > returns nil for empty error bodies on every provider
#[test]
fn empty_error_bodies_parse_to_nothing_on_every_provider() {
    for provider in providers::ALL {
        for body in ["", "{}", "[]"] {
            assert_eq!(
                provider.parse_error(body),
                None,
                "{} {body:?}",
                provider.slug()
            );
        }
    }
}

// spec: provider_spec.rb:494 with API base configuration > keeps existing defaults for providers with built-in endpoints
#[test]
fn providers_with_built_in_endpoints_keep_their_default_api_base() {
    let defaults = [
        (Provider::Anthropic, "https://api.anthropic.com"),
        (Provider::DeepSeek, "https://api.deepseek.com"),
        (
            Provider::Gemini,
            "https://generativelanguage.googleapis.com/v1beta",
        ),
        (Provider::Hetzner, "https://inference.hetzner.com/api/v1"),
        (Provider::Mistral, "https://api.mistral.ai/v1"),
        (Provider::OllamaCloud, "https://ollama.com/v1"),
        (Provider::OpenAI, "https://api.openai.com/v1"),
        (Provider::OpenRouter, "https://openrouter.ai/api/v1"),
        (Provider::Perplexity, "https://api.perplexity.ai"),
        (Provider::TypeSafe, "https://api.typesafe.ai"),
        (Provider::XAI, "https://api.x.ai/v1"),
    ];
    for (provider, default) in defaults {
        assert_eq!(
            provider.api_base(&config_for(provider)).unwrap(),
            default,
            "{}",
            provider.slug()
        );
    }
}

fn gpt_5_4() -> rust_llm::Model {
    rust_llm::Model::default_for("gpt-5.4", "openai")
}

// spec: provider_spec.rb:522 protocol resolution > prefers the configured protocol over routing
#[test]
fn the_configured_protocol_beats_routing() {
    let mut config = config_for(Provider::OpenAI);
    config.set("openai_protocol", "chat_completions");
    assert_eq!(
        Provider::OpenAI
            .resolve_protocol(None, &gpt_5_4(), &config)
            .unwrap(),
        ProtocolName::ChatCompletions
    );
}

// spec: provider_spec.rb:531 protocol resolution > prefers an explicit protocol over the configured one
#[test]
fn an_explicit_protocol_beats_the_configured_one() {
    let mut config = config_for(Provider::OpenAI);
    config.set("openai_protocol", "chat_completions");
    let protocol = Provider::OpenAI
        .resolve_protocol(Some(ProtocolName::Responses), &gpt_5_4(), &config)
        .unwrap();
    assert_eq!(protocol, ProtocolName::Responses);
}

// spec: provider_spec.rb:609 protocol resolution > raises on protocols the provider does not speak
#[test]
fn a_protocol_the_provider_does_not_speak_is_refused() {
    let error = Provider::OpenAI
        .resolve_protocol(
            Some(ProtocolName::Gemini),
            &gpt_5_4(),
            &config_for(Provider::OpenAI),
        )
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Api);
    let pattern = regex::Regex::new(
        r"gemini is not a protocol of OpenAI\. Available: responses, chat_completions",
    )
    .unwrap();
    assert!(pattern.is_match(&error.to_string()), "{error}");
}

// spec: provider_spec.rb:642 files protocol registration > exposes provider-managed files only where implemented
#[test]
fn provider_managed_files_exist_only_where_implemented() {
    let file_providers = [
        "anthropic",
        "deepseek",
        "gemini",
        "mistral",
        "openai",
        "openrouter",
        "perplexity",
        "xai",
    ];
    for provider in providers::ALL {
        assert_eq!(
            rust_llm::files::supports_files(*provider),
            file_providers.contains(&provider.slug()),
            "{}",
            provider.slug()
        );
    }
}

// spec: provider_spec.rb:693 providers without file support > refuses every file operation
#[tokio::test]
async fn a_provider_without_files_refuses_every_file_operation() {
    let config = Arc::new(config_for(Provider::Ollama));
    let options = || FileOptions {
        provider: Some("ollama"),
        config: Some(config.clone()),
    };
    let upload = UploadedFile::upload(
        Attachment::from_bytes(b"hello".to_vec(), "file.txt", Some("text/plain")),
        UploadOptions {
            provider: Some("ollama"),
            config: Some(config.clone()),
            ..Default::default()
        },
    )
    .await;
    let results = [
        upload.map(|_| ()),
        UploadedFile::find("id", options()).await.map(|_| ()),
        UploadedFile::download("id", options()).await.map(|_| ()),
    ];
    for result in results {
        let error = result.unwrap_err();
        assert!(
            error.to_string().contains("doesn't support file uploads"),
            "{error}"
        );
    }
}

fn openai_error(body: Value) -> Option<String> {
    Provider::OpenAI.parse_error(&body.to_string())
}

// spec: provider_spec.rb:723 #parse_error body shapes > joins a list of errors
#[test]
fn a_list_of_errors_is_joined() {
    assert_eq!(
        openai_error(json!([{ "error": "first" }, { "error": { "message": "second" } }]))
            .as_deref(),
        Some("first. second")
    );
}

// spec: provider_spec.rb:729 #parse_error body shapes > ignores empty error shapes and stringifies scalar list entries
#[test]
fn empty_error_shapes_are_ignored_and_scalar_entries_stringified() {
    let body = json!([null, "", { "error": [] }, "second", { "message": "third" }]);
    assert_eq!(openai_error(body).as_deref(), Some("second. third"));
    assert_eq!(openai_error(json!({ "error": [] })), None);
}

// spec: provider_spec.rb:736 #parse_error body shapes > passes a body it cannot interpret through
#[test]
fn an_uninterpretable_body_is_passed_through() {
    assert_eq!(openai_error(json!(42)).as_deref(), Some("42"));
}

// spec: provider_spec.rb:744 #parse_error body shapes > keeps an unparseable string body as text
#[test]
fn an_unparseable_string_body_is_kept_as_text() {
    assert_eq!(
        Provider::OpenAI
            .parse_error("plain text failure")
            .as_deref(),
        Some("plain text failure")
    );
}

// spec: provider_spec.rb:750 provider registry partitions > splits providers into local and remote
#[test]
fn providers_split_into_local_and_remote() {
    let local = providers::local_providers();
    let remote = providers::remote_providers();
    assert_eq!(local, vec![Provider::Ollama, Provider::GPUStack]);
    assert!(!remote.contains(&Provider::Ollama) && !remote.contains(&Provider::GPUStack));
    let mut all: Vec<&str> = local.iter().chain(&remote).map(Provider::slug).collect();
    let mut expected: Vec<&str> = providers::ALL.iter().map(Provider::slug).collect();
    all.sort();
    expected.sort();
    assert_eq!(all, expected);
}

// spec: provider_spec.rb:758 provider registry partitions > lists only providers the configuration can reach
#[test]
fn only_reachable_providers_are_listed_as_configured() {
    let config = config_for(Provider::OpenAI);
    assert!(providers::configured_providers(&config).contains(&Provider::OpenAI));
    assert!(providers::configured_remote_providers(&config).contains(&Provider::OpenAI));
    assert!(!providers::configured_remote_providers(&config).contains(&Provider::Ollama));
}

// ---- providers/{anthropic,hetzner,ollama_cloud,ollama}_spec.rb --------------------------------

fn keyed(option: &str) -> Config {
    let mut config = Config::default();
    config.set(option, "test-key");
    config
}

fn bearer(key: &str) -> Vec<(String, String)> {
    vec![("Authorization".to_string(), format!("Bearer {key}"))]
}

// spec: providers/anthropic_spec.rb:28 #api_base > when anthropic_api_base is not set > returns the default Anthropic API URL
#[test]
fn anthropic_defaults_to_its_api_url() {
    assert_eq!(
        Provider::Anthropic
            .api_base(&keyed("anthropic_api_key"))
            .unwrap(),
        "https://api.anthropic.com"
    );
}

// spec: providers/hetzner_spec.rb:20 declares provider configuration
#[test]
fn hetzner_declares_its_configuration() {
    assert_eq!(
        Provider::Hetzner.configuration_options(),
        ["hetzner_api_key", "hetzner_api_base"]
    );
    assert_eq!(
        Provider::Hetzner.configuration_requirements(),
        ["hetzner_api_key"]
    );
}

// spec: providers/hetzner_spec.rb:25 defaults to the Hetzner Inference endpoint and sends a bearer token
#[test]
fn hetzner_defaults_to_its_endpoint_with_a_bearer_token() {
    let config = keyed("hetzner_api_key");
    assert_eq!(
        Provider::Hetzner.api_base(&config).unwrap(),
        "https://inference.hetzner.com/api/v1"
    );
    assert_eq!(Provider::Hetzner.headers(&config), bearer("test-key"));
}

// spec: providers/hetzner_spec.rb:36 is a remote provider that requires an API key
#[test]
fn hetzner_is_remote_and_requires_an_api_key() {
    assert!(!Provider::Hetzner.is_local());
    assert!(providers::remote_providers().contains(&Provider::Hetzner));
    assert!(!Provider::Hetzner.is_configured(&Config::default()));
    assert!(Provider::Hetzner.is_configured(&keyed("hetzner_api_key")));
}

// spec: providers/hetzner_spec.rb:43 allows model ids missing from the registry
#[test]
fn hetzner_allows_unregistered_model_ids() {
    assert!(Provider::Hetzner.assume_models_exist());
}

// spec: providers/hetzner_spec.rb:109 attachments > rejects documents and audio, which the models do not accept
#[test]
fn hetzner_rejects_documents_and_audio() {
    let fixtures = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
    for (prompt, file) in [("Read this", "sample.pdf"), ("Hear this", "ruby.wav")] {
        let config = Arc::new(keyed("hetzner_api_key"));
        let mut chat =
            Chat::with_config(config, Some("Qwen3.8-27B"), Some("hetzner"), false).expect("chat");
        chat.ask_later_with(prompt, vec![Attachment::new(format!("{fixtures}/{file}"))])
            .expect("stage");
        let result = chat.render();
        assert!(
            matches!(result, Err(Error::UnsupportedAttachment(_))),
            "{file}: {result:?}"
        );
    }
}

// spec: providers/ollama_cloud_spec.rb:19 declares provider configuration
#[test]
fn ollama_cloud_declares_its_configuration() {
    assert_eq!(
        Provider::OllamaCloud.configuration_options(),
        ["ollama_cloud_api_key", "ollama_cloud_api_base"]
    );
    assert_eq!(
        Provider::OllamaCloud.configuration_requirements(),
        ["ollama_cloud_api_key"]
    );
}

// spec: providers/ollama_cloud_spec.rb:24 defaults to the ollama.com endpoint and sends a bearer token
#[test]
fn ollama_cloud_defaults_to_ollama_com_with_a_bearer_token() {
    let config = keyed("ollama_cloud_api_key");
    assert_eq!(
        Provider::OllamaCloud.api_base(&config).unwrap(),
        "https://ollama.com/v1"
    );
    assert_eq!(Provider::OllamaCloud.headers(&config), bearer("test-key"));
}

// spec: providers/ollama_cloud_spec.rb:35 is a remote provider, unlike local Ollama
#[test]
fn ollama_cloud_is_remote_unlike_local_ollama() {
    assert!(!Provider::OllamaCloud.is_local());
    assert!(providers::remote_providers().contains(&Provider::OllamaCloud));
}

// spec: providers/ollama_cloud_spec.rb:40 requires an API key
#[test]
fn ollama_cloud_requires_an_api_key() {
    assert!(!Provider::OllamaCloud.is_configured(&Config::default()));
    assert!(Provider::OllamaCloud.is_configured(&keyed("ollama_cloud_api_key")));
}

// spec: providers/ollama_cloud_spec.rb:45 allows model ids missing from the registry
#[test]
fn ollama_cloud_allows_unregistered_model_ids() {
    assert!(Provider::OllamaCloud.assume_models_exist());
}

// spec: providers/ollama_spec.rb:9 #headers > returns empty headers when no API key is configured
#[test]
fn ollama_sends_no_headers_without_an_api_key() {
    let mut config = Config::default();
    config.set("ollama_api_base", "http://localhost:11434/v1");
    assert_eq!(
        Provider::Ollama.headers(&config),
        Vec::<(String, String)>::new()
    );
}

// spec: providers/ollama_spec.rb:16 #headers > returns Authorization header when API key is configured
#[test]
fn ollama_sends_a_bearer_token_with_an_api_key() {
    let mut config = Config::default();
    config.set("ollama_api_base", "http://localhost:11434/v1");
    config.set("ollama_api_key", "test-ollama-key");
    assert_eq!(Provider::Ollama.headers(&config), bearer("test-ollama-key"));
}
