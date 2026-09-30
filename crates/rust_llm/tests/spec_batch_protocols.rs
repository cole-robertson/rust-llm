//! Batch protocol unit specs: `protocols/chat_completions/batches_spec.rb`,
//! `protocols/chat_completions/embedding_batches_spec.rb`, `protocols/responses/batches_spec.rb`,
//! and `protocols/gemini/batches_spec.rb`. RubyLLM stubs `@connection` and `upload_file`; here the
//! provider talks to a wiremock server, and `validate_batch_requests!` is reached through its
//! `#[doc(hidden)]` port in `rust_llm::batch`.

use std::sync::{Arc, Mutex};

use rust_llm::batch::openai_validate_batch_requests;
use rust_llm::{Batch, BatchStatus, Chat, Config, Error, ProtocolName};
use serde_json::{Value, json};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn config(server: &MockServer) -> Arc<Config> {
    let mut c = Config::default();
    c.set("openai_api_base", format!("{}/v1", server.uri()));
    c.set("openai_api_key", "test");
    c.set("gemini_api_base", format!("{}/v1beta", server.uri()));
    c.set("gemini_api_key", "test");
    c.max_retries = 0;
    Arc::new(c)
}

async fn mount(server: &MockServer, verb: &str, at: &str, body: Value) {
    Mock::given(method(verb))
        .and(path(at))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(server)
        .await;
}

async fn received(server: &MockServer, verb: &str, at: &str) -> Vec<wiremock::Request> {
    server
        .received_requests()
        .await
        .unwrap_or_default()
        .into_iter()
        .filter(|r| r.method.as_str() == verb && r.url.path() == at)
        .collect()
}

/// A text field of a multipart form body.
fn form_field(body: &[u8], name: &str) -> Option<String> {
    let body = String::from_utf8_lossy(body);
    let marker = format!("name=\"{name}\"");
    let start = body.find(&marker)? + marker.len();
    let rest = &body[start..];
    let value_start = rest.find("\r\n\r\n")? + 4;
    let end = rest[value_start..].find("\r\n--")?;
    Some(rest[value_start..value_start + end].to_string())
}

fn assert_api_error(result: rust_llm::Result<()>, needle: &str) {
    match result {
        Err(Error::Api(message, _)) => assert!(message.contains(needle), "{message}"),
        other => panic!("expected an Error mentioning {needle}, got {other:?}"),
    }
}

// ---- chat_completions/batches_spec.rb -----------------------------------------------------------

// spec: protocols/chat_completions/batches_spec.rb:31 #create_batch > uploads JSONL through the provider files API and creates a Chat Completions batch
/// `gpt-4o-mini-search-preview` only works over Chat Completions, so its staged chat renders a
/// Chat Completions payload and the batch goes to `/v1/chat/completions`.
#[tokio::test]
async fn uploads_jsonl_and_creates_a_chat_completions_batch() {
    let server = MockServer::start().await;
    mount(&server, "POST", "/v1/files", json!({ "id": "file_123" })).await;
    mount(
        &server,
        "POST",
        "/v1/batches",
        json!({ "id": "batch_123", "status": "validating" }),
    )
    .await;
    let mut chat = Chat::with_config(
        config(&server),
        Some("gpt-4o-mini-search-preview"),
        Some("openai"),
        false,
    )
    .unwrap();
    chat.ask_later("Hi").unwrap();

    let batch = rust_llm::batch(chat).await.unwrap();

    assert_eq!(batch.id(), "batch_123");
    let uploads = received(&server, "POST", "/v1/files").await;
    assert_eq!(uploads.len(), 1);
    assert_eq!(
        form_field(&uploads[0].body, "purpose").as_deref(),
        Some("batch")
    );
    let upload = String::from_utf8_lossy(&uploads[0].body);
    assert!(
        upload.contains("filename=\"ruby_llm_batch.jsonl\""),
        "{upload}"
    );
    let line: Value = serde_json::from_str(
        form_field(&uploads[0].body, "file")
            .expect("file part")
            .lines()
            .next()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(line["url"], "/v1/chat/completions");
    assert!(line["body"].get("messages").is_some());
    assert!(line["body"].get("stream").is_none());
    let creates = received(&server, "POST", "/v1/batches").await;
    assert_eq!(creates.len(), 1);
    assert_eq!(
        serde_json::from_slice::<Value>(&creates[0].body).unwrap(),
        json!({ "input_file_id": "file_123", "endpoint": "/v1/chat/completions", "completion_window": "24h" })
    );
}

// spec: protocols/chat_completions/batches_spec.rb:59 #validate_batch_requests! > rejects unsupported payload shapes
#[test]
fn chat_completions_batches_reject_unsupported_payload_shapes() {
    assert_api_error(
        openai_validate_batch_requests("chat_completions", &[json!({ "input": "hi" })]),
        "chat completion payloads",
    );
}

// spec: protocols/chat_completions/batches_spec.rb:96 #parse_batch_completion_response > parses chat completion rows with the chat completions parser
#[tokio::test]
async fn parses_chat_completion_rows_with_the_chat_completions_parser() {
    let server = MockServer::start().await;
    let body = json!({
        "model": "gpt-5-nano",
        "choices": [{ "message": { "role": "assistant", "content": "Hello" } }],
        "usage": { "prompt_tokens": 2, "completion_tokens": 1 }
    });
    mount(
        &server,
        "GET",
        "/v1/batches/batch_1",
        json!({ "id": "batch_1", "status": "completed", "endpoint": "/v1/chat/completions", "output_file_id": "file-out", "request_counts": { "total": 1 } }),
    )
    .await;
    let line = json!({ "custom_id": "0", "response": { "status_code": 200, "body": body } });
    Mock::given(method("GET"))
        .and(path("/v1/files/file-out/content"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(format!("{line}\n"), "application/octet-stream"),
        )
        .mount(&server)
        .await;

    let mut batch = Batch::find_with_config(config(&server), "batch_1", Some("openai"))
        .await
        .unwrap();
    let messages = batch.messages().await.unwrap();

    let message = messages[0].as_ref().expect("a parsed message");
    assert_eq!(message.content(), "Hello");
    assert_eq!(message.model.as_deref(), Some("gpt-5-nano"));
    assert_eq!(message.tokens.input, Some(2));
    assert_eq!(message.raw.as_ref().map(|r| &r.body), Some(&body));
}

// ---- chat_completions/embedding_batches_spec.rb -------------------------------------------------

// spec: protocols/chat_completions/embedding_batches_spec.rb:59 #validate_batch_requests! > rejects unsupported payload shapes
#[test]
fn embedding_batches_reject_unsupported_payload_shapes() {
    assert_api_error(
        openai_validate_batch_requests("embeddings", &[json!({ "messages": [] })]),
        "embedding payloads",
    );
}

// ---- responses/batches_spec.rb ------------------------------------------------------------------

// spec: protocols/responses/batches_spec.rb:64 #validate_batch_requests! > rejects unsupported payload shapes
#[test]
fn responses_batches_reject_unsupported_payload_shapes() {
    assert_api_error(
        openai_validate_batch_requests("responses", &[json!({ "messages": [] })]),
        "responses payloads",
    );
}

/// The Chat Completions protocol is what an explicit `with_protocol` picks; a batch of such chats
/// must pass validation and reach the provider.
#[tokio::test]
async fn validated_chat_completions_chats_still_submit() {
    let server = MockServer::start().await;
    mount(&server, "POST", "/v1/files", json!({ "id": "file_123" })).await;
    mount(
        &server,
        "POST",
        "/v1/batches",
        json!({ "id": "batch_123", "status": "validating" }),
    )
    .await;
    let mut chat = Chat::with_config(config(&server), Some("gpt-5-nano"), Some("openai"), false)
        .unwrap()
        .with_protocol(ProtocolName::ChatCompletions);
    chat.ask_later("hi").unwrap();
    rust_llm::batch(chat).await.unwrap();
    assert_eq!(received(&server, "POST", "/v1/batches").await.len(), 1);
}

// ---- gemini/batches_spec.rb ---------------------------------------------------------------------

/// A Gemini batch `batches/abc` whose GET answers `batch`. Keep the server bound while the batch
/// is used: a dropped `MockServer` goes back to wiremock's pool for another test to reuse.
async fn gemini_batch(batch: Value) -> (MockServer, Batch) {
    let server = MockServer::start().await;
    mount(&server, "GET", "/v1beta/batches/abc", batch).await;
    let found = Batch::find_with_config(config(&server), "batches/abc", Some("gemini"))
        .await
        .unwrap();
    (server, found)
}

// spec: protocols/gemini/batches_spec.rb:99 #parse_batch_response > marks terminal states completed regardless of the enum prefix
#[tokio::test]
async fn marks_terminal_gemini_states_completed_regardless_of_the_enum_prefix() {
    for state in [
        "BATCH_STATE_SUCCEEDED",
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
    ] {
        let (_server, batch) = gemini_batch(json!({ "name": "batches/abc", "state": state })).await;
        assert!(batch.is_complete(), "{state}");
    }
}

fn completed_gemini_batch(inlined: Value) -> Value {
    json!({
        "name": "batches/abc",
        "metadata": { "model": "models/gemini-2.5-flash", "state": "BATCH_STATE_SUCCEEDED" },
        "response": { "inlinedResponses": { "inlinedResponses": inlined } }
    })
}

// spec: protocols/gemini/batches_spec.rb:127 #parse_inline_response > falls back to the old metadata key and then position
#[tokio::test]
async fn falls_back_to_the_old_metadata_key_and_then_position() {
    let answer = |text: &str| json!({ "candidates": [{ "content": { "parts": [{ "text": text }] } }], "modelVersion": "gemini-2.5-flash" });
    let (_server, mut batch) = gemini_batch(completed_gemini_batch(json!([
        { "metadata": { "key": "2" }, "response": answer("by key") },
        { "response": answer("by position") }
    ])))
    .await;
    let messages = batch.messages().await.unwrap();
    assert_eq!(messages.len(), 3);
    assert!(messages[0].is_none());
    assert_eq!(
        messages[1].as_ref().map(|m| m.content().to_string()),
        Some("by position".into())
    );
    assert_eq!(
        messages[2].as_ref().map(|m| m.content().to_string()),
        Some("by key".into())
    );
}

// spec: protocols/gemini/batches_spec.rb:135 #parse_inline_response > logs and skips a per-item error
#[tokio::test]
async fn logs_and_skips_a_per_item_error() {
    let (_server, mut batch) = gemini_batch(completed_gemini_batch(json!([
        { "metadata": { "custom_id": "0" }, "error": { "code": 400, "message": "too long" } }
    ])))
    .await;
    let warnings = Arc::new(Mutex::new(Vec::new()));
    let messages = {
        let _guard = tracing::dispatcher::set_default(&tracing::Dispatch::new(WarnCollector(
            warnings.clone(),
        )));
        batch.messages().await.unwrap()
    };
    assert_eq!(messages.len(), 1);
    assert!(messages[0].is_none());
    assert_eq!(batch.statuses(), &[Some(BatchStatus::Failed)]);
    let warnings = warnings.lock().unwrap();
    assert!(
        warnings.iter().any(|w| w.contains("failed: too long")),
        "{warnings:?}"
    );
}

/// Collects WARN messages, as the Ruby spec's `RubyLLM.logger` spy does.
struct WarnCollector(Arc<Mutex<Vec<String>>>);

impl tracing::Subscriber for WarnCollector {
    fn register_callsite(
        &self,
        _: &'static tracing::Metadata<'static>,
    ) -> tracing::subscriber::Interest {
        tracing::subscriber::Interest::sometimes()
    }
    fn max_level_hint(&self) -> Option<tracing::level_filters::LevelFilter> {
        Some(tracing::level_filters::LevelFilter::TRACE)
    }
    fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        struct Message<'a>(&'a mut String);
        impl tracing::field::Visit for Message<'_> {
            fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                if field.name() == "message" {
                    self.0.push_str(&format!("{value:?}"));
                }
            }
        }
        if *event.metadata().level() == tracing::Level::WARN {
            let mut message = String::new();
            event.record(&mut Message(&mut message));
            self.0.lock().unwrap().push(message);
        }
    }
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}
