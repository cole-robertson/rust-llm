//! `spec/ruby_llm/batch_helpers_spec.rb` and the stubbed `batch_spec.rb` examples not covered by
//! `batches.rs`. The Ruby specs build `Batch.new(provider:, id:, raw_status:, completed:)` and stub
//! `provider.batch_results`/`find_batch`; here `Batch::from_attributes` builds the same batch and a
//! mock server serves the provider's batch endpoints, so the real parse path runs in between.
//! `// spec:` lines tie each test to its Ruby example.

mod spec_helpers;

use std::sync::{Arc, Mutex};

use rust_llm::batch::{BatchAttributes, batch_error_message, batch_failure};
use rust_llm::{Batch, BatchStatus, Config, Cost, EmbedOptions, Role, Tokens, embed_later};
use serde_json::{Value, json};
use spec_helpers::*;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn attributes(id: &str, raw_status: &str, completed: bool) -> BatchAttributes {
    BatchAttributes {
        id: id.into(),
        raw_status: Some(raw_status.into()),
        completed,
        ..Default::default()
    }
}

/// `RubyLLM.chat(model: model_for(:anthropic)).ask_later(text)` against `server`.
fn staged(server: &MockServer, text: &str) -> rust_llm::Chat {
    let mut chat = chat(server);
    chat.ask_later(text).expect("ask_later");
    chat
}

fn roles(chat: &rust_llm::Chat) -> Vec<Role> {
    chat.messages().iter().map(|m| m.role).collect()
}

async fn get(server: &MockServer, at: &str, response: ResponseTemplate) {
    Mock::given(method("GET"))
        .and(path(at))
        .respond_with(response)
        .mount(server)
        .await;
}

fn jsonl(lines: &[Value]) -> ResponseTemplate {
    let body: Vec<String> = lines.iter().map(Value::to_string).collect();
    ResponseTemplate::new(200).set_body_raw(body.join("\n"), "application/octet-stream")
}

/// An Anthropic batch results line answering request `custom_id` with `text`.
fn anthropic_line(custom_id: &str, text: &str) -> Value {
    json!({ "custom_id": custom_id, "result": { "type": "succeeded", "message": text_response(text) } })
}

/// An OpenAI batch output line: a Responses answer from gpt-5-nano using `input` tokens.
fn openai_line(custom_id: &str, text: &str, input: i64) -> Value {
    json!({ "custom_id": custom_id, "response": { "status_code": 200, "body": {
        "id": "resp_1", "object": "response", "status": "completed", "model": "gpt-5-nano",
        "output": [{ "type": "message", "role": "assistant", "content": [{ "type": "output_text", "text": text }] }],
        "usage": { "input_tokens": input, "output_tokens": 0 } } } })
}

/// An OpenAI batch as `GET /v1/batches/:id` returns it.
fn openai_batch(
    id: &str,
    status: &str,
    endpoint: &str,
    output_file_id: Option<&str>,
) -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(json!({ "id": id, "status": status, "endpoint": endpoint, "output_file_id": output_file_id }))
}

/// Collects `tracing` WARN events on this thread, standing in for `RubyLLM.logger.warn`.
struct WarnCollector(Arc<Mutex<Vec<String>>>);

impl tracing::Subscriber for WarnCollector {
    // Tests run in parallel: a callsite first hit with no collector set is cached as "never",
    // so ask on every event instead of caching the interest.
    fn register_callsite(
        &self,
        _: &'static tracing::Metadata<'static>,
    ) -> tracing::subscriber::Interest {
        tracing::subscriber::Interest::sometimes()
    }
    // Without this the global max level is recomputed from other threads' (absent) collectors and
    // can drop WARN events before they reach this one.
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
            let mut text = String::new();
            event.record(&mut Message(&mut text));
            self.0.lock().unwrap().push(text);
        }
    }
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}

fn warnings_of(f: impl FnOnce()) -> Vec<String> {
    let warnings = Arc::new(Mutex::new(Vec::new()));
    {
        let _guard = tracing::dispatcher::set_default(&tracing::Dispatch::new(WarnCollector(
            warnings.clone(),
        )));
        f();
    }
    warnings.lock().unwrap().clone()
}

// ---- #status ------------------------------------------------------------------------------------

/// The Bedrock and Vertex AI rows are left out: those providers are not ported (BRIEF.md).
// spec: batch_helpers_spec.rb:9 #status normalizes provider lifecycle statuses
#[test]
fn normalizes_provider_lifecycle_statuses() {
    let cases = [
        ("anthropic", "in_progress", false, BatchStatus::Pending),
        ("anthropic", "ended", true, BatchStatus::Succeeded),
        ("openai", "failed", true, BatchStatus::Failed),
        ("openai", "expired", true, BatchStatus::Failed),
        ("openai", "cancelled", true, BatchStatus::Cancelled),
        (
            "gemini",
            "JOB_STATE_SUCCEEDED",
            true,
            BatchStatus::Succeeded,
        ),
        ("gemini", "JOB_STATE_FAILED", true, BatchStatus::Failed),
        (
            "gemini",
            "JOB_STATE_CANCELLED",
            true,
            BatchStatus::Cancelled,
        ),
        ("mistral", "TIMEOUT_EXCEEDED", true, BatchStatus::Failed),
        ("mistral", "CANCELLED", true, BatchStatus::Cancelled),
        ("xai", "completed", true, BatchStatus::Succeeded),
    ];
    for (provider, raw_status, completed, status) in cases {
        let batch = Batch::from_attributes(
            Arc::new(Config::default()),
            provider,
            attributes("batch_1", raw_status, completed),
        )
        .unwrap();
        assert_eq!(
            batch.status(),
            status,
            "expected {provider} {raw_status} to be {status:?}"
        );
    }
}

// spec: batch_helpers_spec.rb:36 #status exposes the provider status separately
#[test]
fn exposes_the_provider_status_separately() {
    let batch = Batch::from_attributes(
        Arc::new(Config::default()),
        "openai",
        attributes("batch_1", "cancelled", true),
    )
    .unwrap();

    assert_eq!(batch.status(), BatchStatus::Cancelled);
    assert_eq!(batch.raw_status(), Some("cancelled"));
    assert!(batch.is_complete());
    assert!(batch.is_cancelled());
    assert!(!batch.is_succeeded());
    assert!(!batch.is_failed());
}

// ---- #cost --------------------------------------------------------------------------------------

// spec: batch_helpers_spec.rb:52 #cost preserves a zero invoice, waits for completion, and retains it across missing metadata
#[tokio::test]
async fn preserves_a_zero_invoice_waits_for_completion_and_retains_it_across_missing_metadata() {
    let server = MockServer::start().await;
    // find_batch reports the batch completed, with no invoice this time.
    get(
        &server,
        "/v1/batches/batch_cost",
        openai_batch("batch_cost", "completed", "/v1/responses", None),
    )
    .await;
    let zero = Cost::from_recorded([None; 5], Some(0.0), &Tokens::default());
    let pending = BatchAttributes {
        reported_cost: Some(zero.clone()),
        ..attributes("batch_cost", "in_progress", false)
    };
    let mut batch = Batch::from_attributes(config(&server), "openai", pending).unwrap();

    assert_eq!(batch.reported_cost().unwrap().total(), Some(0.0));
    assert_eq!(batch.cost().await.unwrap().total(), None);

    batch.refresh().await.unwrap();
    let cost = batch.cost().await.unwrap();
    assert_eq!(cost.total(), Some(0.0));
    assert_eq!(batch.reported_cost(), Some(&zero));
    assert_eq!(cost.input, None);
}

/// The two answers cost $0.02 and $0.03: gpt-5-nano input is $0.05 per million, half in a batch.
// spec: batch_helpers_spec.rb:76 #cost keeps the total unknown until all processing ends when results arrive early
#[tokio::test]
async fn keeps_the_total_unknown_until_all_processing_ends_when_results_arrive_early() {
    let server = MockServer::start().await;
    // The first collection (protocol lookup, then results) sees the batch still running with one
    // answer in; after that it has completed with both.
    Mock::given(method("GET"))
        .and(path("/v1/batches/batch_cost"))
        .respond_with(openai_batch(
            "batch_cost",
            "in_progress",
            "/v1/responses",
            Some("file-early"),
        ))
        .up_to_n_times(2)
        .mount(&server)
        .await;
    get(
        &server,
        "/v1/batches/batch_cost",
        openai_batch(
            "batch_cost",
            "completed",
            "/v1/responses",
            Some("file-full"),
        ),
    )
    .await;
    get(
        &server,
        "/v1/files/file-early/content",
        jsonl(&[openai_line("0", "Ruby", 800_000)]),
    )
    .await;
    get(
        &server,
        "/v1/files/file-full/content",
        jsonl(&[
            openai_line("0", "Ruby", 800_000),
            openai_line("1", "Rails", 1_200_000),
        ]),
    )
    .await;
    let pending = BatchAttributes {
        request_count: Some(2),
        ..attributes("batch_cost", "in_progress", false)
    };
    let mut batch = Batch::from_attributes(config(&server), "openai", pending).unwrap();

    let early = batch.messages().await.unwrap();
    assert_eq!(
        early.iter().map(Option::is_some).collect::<Vec<_>>(),
        vec![true, false]
    );
    assert_eq!(batch.cost().await.unwrap().total(), None);

    batch.refresh().await.unwrap();
    let messages = batch.messages().await.unwrap();
    assert_eq!(
        messages
            .iter()
            .map(|m| m.as_ref().unwrap().content().to_string())
            .collect::<Vec<_>>(),
        vec!["Ruby", "Rails"]
    );
    assert!((batch.cost().await.unwrap().total().unwrap() - 0.05).abs() < 1e-12);
}

// spec: batch_helpers_spec.rb:92 #cost returns an unknown cost without requesting unavailable results while pending
#[tokio::test]
async fn returns_an_unknown_cost_without_requesting_unavailable_results_while_pending() {
    let server = serve(vec![]).await;
    let mut batch = Batch::from_attributes(
        config(&server),
        "anthropic",
        attributes("batch_pending", "in_progress", false),
    )
    .unwrap();

    assert_eq!(batch.cost().await.unwrap().total(), None);
    assert_eq!(
        requests(&server).await,
        0,
        "no batch results were requested"
    );
}

// spec: batch_helpers_spec.rb:102 #cost is empty for a batch that collected nothing
#[tokio::test]
async fn is_empty_for_a_batch_that_collected_nothing() {
    let server = MockServer::start().await;
    get(
        &server,
        "/v1/messages/batches/msgbatch_123/results",
        jsonl(&[]),
    )
    .await;
    let ended = BatchAttributes {
        request_count: Some(1),
        ..attributes("msgbatch_123", "ended", true)
    };
    let mut batch = Batch::from_attributes(config(&server), "anthropic", ended).unwrap();

    assert!(batch.messages().await.unwrap().iter().all(Option::is_none));
    assert_eq!(batch.cost().await.unwrap().total(), None);
}

// ---- Batch::Helpers ---------------------------------------------------------------------------

// spec: batch_helpers_spec.rb:123 #batch_error_message reads a flat string error
#[test]
fn batch_error_message_reads_a_flat_string_error() {
    assert_eq!(
        batch_error_message(&json!({ "error": "boom" })).as_deref(),
        Some("boom")
    );
}

// spec: batch_helpers_spec.rb:127 #batch_error_message reads every nested shape providers use
#[test]
fn batch_error_message_reads_every_nested_shape_providers_use() {
    assert_eq!(
        batch_error_message(&json!({ "error": { "message": "nested" } })).as_deref(),
        Some("nested")
    );
    assert_eq!(
        batch_error_message(&json!({ "error_message": "flat field" })).as_deref(),
        Some("flat field")
    );
    assert_eq!(
        batch_error_message(&json!({ "response": { "body": { "error": { "message": "body" } } } }))
            .as_deref(),
        Some("body")
    );
    assert_eq!(
        batch_error_message(&json!({ "response": { "error": { "message": "response" } } }))
            .as_deref(),
        Some("response")
    );
}

// spec: batch_helpers_spec.rb:138 #batch_error_message reads nested string errors
#[test]
fn batch_error_message_reads_nested_string_errors() {
    assert_eq!(
        batch_error_message(&json!({ "response": { "body": { "error": "body" } } })).as_deref(),
        Some("body")
    );
    assert_eq!(
        batch_error_message(&json!({ "response": { "error": "response" } })).as_deref(),
        Some("response")
    );
}

// spec: batch_helpers_spec.rb:143 #batch_error_message is nil when the line carries no error
#[test]
fn batch_error_message_is_none_when_the_line_carries_no_error() {
    assert_eq!(batch_error_message(&json!({})), None);
}

// spec: batch_helpers_spec.rb:185 #batch_failure warns with the detail when there is one
#[test]
fn batch_failure_warns_with_the_detail_when_there_is_one() {
    let warnings = warnings_of(|| {
        batch_failure("7", Some("rate limited".into()), "expired");
    });
    assert_eq!(warnings, vec!["Batch request 7 expired: rate limited"]);
}

/// Ruby's `status:` defaults to 'failed'; Rust passes it explicitly.
// spec: batch_helpers_spec.rb:193 #batch_failure warns without a detail
#[test]
fn batch_failure_warns_without_a_detail() {
    let warnings = warnings_of(|| {
        batch_failure("7", None, "failed");
    });
    assert_eq!(warnings, vec!["Batch request 7 failed"]);
}

// ---- .find --------------------------------------------------------------------------------------

// spec: batch_helpers_spec.rb:223 .find refuses a provider that has no batch API
#[tokio::test]
async fn find_refuses_a_provider_that_has_no_batch_api() {
    let err = Batch::find("batch_123", Some("perplexity"))
        .await
        .unwrap_err();
    assert!(matches!(err, rust_llm::Error::Api(..)), "{err:?}");
    assert_eq!(err.to_string(), "perplexity doesn't support batch requests");
}

// ---- #messages ----------------------------------------------------------------------------------

// spec: batch_helpers_spec.rb:231 #messages delivers an answer once even when the chat stages another question
#[tokio::test]
async fn delivers_an_answer_once_even_when_the_chat_stages_another_question() {
    let server = MockServer::start().await;
    get(
        &server,
        "/v1/messages/batches/msgbatch_123/results",
        jsonl(&[anthropic_line("0", "First answer")]),
    )
    .await;
    let chat = staged(&server, "First question");
    let mut batch = Batch::from_attributes(
        config(&server),
        "anthropic",
        attributes("msgbatch_123", "in_progress", false),
    )
    .unwrap()
    .with_chats(vec![chat]);

    batch.messages().await.unwrap();
    batch.chats_mut().unwrap()[0]
        .ask_later("Second question")
        .unwrap();
    let before = batch.chats().unwrap()[0].messages().to_vec();
    batch.messages().await.unwrap();

    let chat = &batch.chats().unwrap()[0];
    assert_eq!(chat.messages(), before.as_slice());
    assert_eq!(roles(chat), vec![Role::User, Role::Assistant, Role::User]);
}

// spec: batch_helpers_spec.rb:247 #messages retains every missing slot when a reloaded provider batch fails
#[tokio::test]
async fn retains_every_missing_slot_when_a_reloaded_provider_batch_fails() {
    let server = MockServer::start().await;
    get(
        &server,
        "/v1/batches/batch_1",
        openai_batch("batch_1", "failed", "/v1/responses", None),
    )
    .await;
    let failed = BatchAttributes {
        request_count: Some(3),
        request_counts: Some(json!({ "total": 3 })),
        ..attributes("batch_1", "failed", true)
    };
    let mut batch = Batch::from_attributes(config(&server), "openai", failed).unwrap();

    assert!(
        batch
            .messages()
            .await
            .unwrap()
            .iter()
            .map(Option::is_none)
            .eq([true, true, true])
    );
    assert_eq!(batch.statuses(), &[Some(BatchStatus::Failed); 3]);
}

// spec: batch_helpers_spec.rb:263 #messages marks unreturned slots cancelled on a cancelled provider batch
#[tokio::test]
async fn marks_unreturned_slots_cancelled_on_a_cancelled_provider_batch() {
    let server = MockServer::start().await;
    get(
        &server,
        "/v1/batches/batch_1",
        openai_batch("batch_1", "cancelled", "/v1/responses", None),
    )
    .await;
    let cancelled = BatchAttributes {
        request_count: Some(2),
        request_counts: Some(json!({ "total": 2 })),
        ..attributes("batch_1", "cancelled", true)
    };
    let mut batch = Batch::from_attributes(config(&server), "openai", cancelled).unwrap();

    assert!(
        batch
            .messages()
            .await
            .unwrap()
            .iter()
            .map(Option::is_none)
            .eq([true, true])
    );
    assert_eq!(batch.statuses(), &[Some(BatchStatus::Cancelled); 2]);
}

/// One Ruby example per kind: a duplicate, a negative, and a past-the-end index.
// spec: batch_helpers_spec.rb:292 #messages with malformed result indices rejects #{kind} index without delivering any answer
#[tokio::test]
async fn rejects_a_malformed_result_index_without_delivering_any_answer() {
    for (kind, index, error) in [
        ("a duplicate", "0", "Duplicate batch result index: 0"),
        ("a negative", "-1", "Invalid batch result index: -1"),
        ("a past-the-end", "2", "Invalid batch result index: 2"),
    ] {
        let server = MockServer::start().await;
        get(
            &server,
            "/v1/messages/batches/msgbatch_123/results",
            jsonl(&[anthropic_line("0", "Hello"), anthropic_line(index, "Hello")]),
        )
        .await;
        let chats = vec![staged(&server, "Hi"), staged(&server, "Hi")];
        let mut batch = Batch::from_attributes(
            config(&server),
            "anthropic",
            attributes("msgbatch_123", "ended", true),
        )
        .unwrap()
        .with_chats(chats);

        let err = batch.messages().await.unwrap_err();
        assert!(matches!(err, rust_llm::Error::Api(..)), "{kind}: {err:?}");
        assert_eq!(err.to_string(), error, "{kind}");
        let chats = batch.chats().unwrap();
        assert_eq!(
            chats.iter().map(roles).collect::<Vec<_>>(),
            vec![vec![Role::User], vec![Role::User]],
            "{kind}"
        );
        assert!(batch.statuses().is_empty(), "{kind}");
    }
}

// spec: batch_helpers_spec.rb:302 #messages rejects a duplicate embedding index without hydrating any request
#[tokio::test]
async fn rejects_a_duplicate_embedding_index_without_hydrating_any_request() {
    let server = MockServer::start().await;
    get(
        &server,
        "/v1/batches/batch_1",
        openai_batch("batch_1", "completed", "/v1/embeddings", Some("file-out")),
    )
    .await;
    let embedding = |value: f64| {
        json!({ "custom_id": "0", "response": { "status_code": 200, "body": {
            "object": "list", "model": "text-embedding-3-small",
            "data": [{ "object": "embedding", "embedding": [value] }], "usage": { "prompt_tokens": 1 } } } })
    };
    get(
        &server,
        "/v1/files/file-out/content",
        jsonl(&[embedding(0.1), embedding(0.2)]),
    )
    .await;
    let options = || EmbedOptions {
        model: Some("text-embedding-3-small"),
        config: Some(config(&server)),
        ..Default::default()
    };
    let requests = vec![
        embed_later("Hi", options()).unwrap(),
        embed_later("Hi", options()).unwrap(),
    ];
    let mut batch = Batch::from_attributes(
        config(&server),
        "openai",
        attributes("batch_1", "completed", true),
    )
    .unwrap()
    .with_requests(requests);

    let err = batch.results().await.unwrap_err();
    assert!(matches!(err, rust_llm::Error::Api(..)), "{err:?}");
    assert_eq!(err.to_string(), "Duplicate batch result index: 0");
    assert!(batch.requests().unwrap().iter().all(|r| r.result.is_none()));
}

// spec: batch_helpers_spec.rb:313 #messages keeps sparse results for a reloaded batch without a request count
#[tokio::test]
async fn keeps_sparse_results_for_a_reloaded_batch_without_a_request_count() {
    let server = MockServer::start().await;
    get(
        &server,
        "/v1/batches/batch_1",
        openai_batch("batch_1", "completed", "/v1/responses", Some("file-out")),
    )
    .await;
    get(
        &server,
        "/v1/files/file-out/content",
        jsonl(&[openai_line("3", "Hello", 1)]),
    )
    .await;
    let mut batch = Batch::from_attributes(
        config(&server),
        "openai",
        attributes("batch_1", "completed", true),
    )
    .unwrap();

    let messages = batch.messages().await.unwrap();

    assert_eq!(messages.len(), 4);
    assert!(messages[..3].iter().all(Option::is_none));
    let answer = messages[3].as_ref().unwrap();
    assert_eq!(answer.content(), "Hello");
    assert_eq!(answer.model.as_deref(), Some("gpt-5-nano"));
}

// ---- batch_spec.rb ------------------------------------------------------------------------------

// spec: batch_spec.rb:93 .submit with a single chat wraps it without decomposing the conversation
#[tokio::test]
async fn submit_wraps_a_single_chat_without_decomposing_the_conversation() {
    let server = serve(vec![
        json!({ "id": "msgbatch_test", "processing_status": "in_progress" }),
    ])
    .await;
    let chat = staged(&server, "Hi");
    let conversation = chat.messages().to_vec();

    let batch = rust_llm::batch(chat).await.unwrap();

    let chats = batch.chats().unwrap();
    assert_eq!(chats.len(), 1);
    assert_eq!(chats[0].messages(), conversation.as_slice());
    assert_eq!(requests(&server).await, 1);
}
