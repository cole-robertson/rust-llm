//! `spec/ruby_llm/protocols/openrouter/batches_spec.rb`: OpenRouter's inline Batch API
//! (`protocols/openrouter/batches.rb`), with WebMock's stubs served by wiremock at
//! `/api/beta/batches` (`OpenRouter#batch_api_base` of the `/api/v1` API base).
//! `// spec:` lines tie each test to its Ruby example.

mod spec_helpers;

use std::sync::Arc;

use rust_llm::embedding::EmbedInput;
use rust_llm::{Attachment, Batch, BatchStatus, Config, EmbedOptions, Error, FinishReason, Vectors, embed_later};
use serde_json::{Value, json};
use spec_helpers::config;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// `model_for(:openrouter)`.
const MODEL: &str = "claude-haiku-4-5";
/// `model_for(:openrouter, :embedding)`.
const EMBEDDING_MODEL: &str = "openai/text-embedding-3-small";
const ENDPOINT: &str = "/api/beta/batches";

fn batch_data(model: &str, results: Value, api: &str, status: &str, count: usize) -> Value {
    json!({ "id": "batch-ruby", "status": status, "endpoint": api, "model": model, "request_counts": { "total": count }, "results": results })
}

fn response_row(id: &str, body: Value) -> Value {
    json!({ "custom_id": id, "response": { "status_code": 200, "body": body }, "error": null })
}

/// `embedding_row(id, vectors, positions:)`: records in reverse order, each billed $0.0001 for 4 tokens.
fn embedding_row(id: &str, vectors: &[Vec<f64>], positions: Option<Vec<Value>>) -> Value {
    let positions = positions.unwrap_or_else(|| (0..vectors.len()).map(|i| json!(i)).collect());
    let mut rows: Vec<Value> = vectors.iter().zip(positions).map(|(v, index)| json!({ "index": index, "embedding": v })).collect();
    rows.reverse();
    response_row(id, json!({ "model": EMBEDDING_MODEL, "data": rows, "usage": { "prompt_tokens": 4, "cost": 0.0001 } }))
}

async fn stub(server: &MockServer, verb: &str, at: &str, body: Value) {
    Mock::given(method(verb)).and(path(at)).respond_with(ResponseTemplate::new(200).set_body_json(body)).mount(server).await;
}

fn embed_options(config: &Arc<Config>) -> EmbedOptions<'static> {
    EmbedOptions { model: Some(EMBEDDING_MODEL), provider: Some("openrouter"), config: Some(config.clone()), ..Default::default() }
}

fn staged_chat(config: &Arc<Config>, text: &str, attachments: Vec<Attachment>) -> rust_llm::Chat {
    let mut chat = rust_llm::Chat::with_config(config.clone(), Some(MODEL), Some("openrouter"), false).unwrap();
    chat.ask_later_with(text, attachments).unwrap();
    chat
}

fn vectors(results: &[Option<rust_llm::BatchResult>]) -> Vec<Option<Vectors>> {
    results.iter().map(|r| r.as_ref().and_then(|r| r.as_embedding()).map(|e| e.vectors.clone())).collect()
}

async fn requests_to(server: &MockServer, verb: &str, at: &str) -> Vec<wiremock::Request> {
    let all = server.received_requests().await.unwrap_or_default();
    all.into_iter().filter(|r| r.method.as_str() == verb && r.url.path() == at).collect()
}

// spec: protocols/openrouter/batches_spec.rb:24 submits inline text requests in the required key order and preserves the resolved model ID
#[tokio::test]
async fn submits_inline_text_requests_in_the_required_key_order_and_preserves_the_resolved_model_id() {
    let server = MockServer::start().await;
    let config = config(&server);
    let chat = staged_chat(&config, "Reply Ruby.", Vec::new());
    let model_id = chat.model().id.clone();
    stub(&server, "POST", ENDPOINT, batch_data(&model_id, json!([]), "/v1/chat/completions", "validating", 1)).await;

    let batch = rust_llm::batch(chat).await.unwrap();

    assert_eq!((batch.id(), batch.status(), batch.raw_status()), ("batch-ruby", BatchStatus::Pending, Some("validating")));
    let posts = requests_to(&server, "POST", ENDPOINT).await;
    assert_eq!(posts.len(), 1);
    let data: Value = serde_json::from_slice(&posts[0].body).unwrap();
    assert_eq!(data.as_object().unwrap().keys().collect::<Vec<_>>(), ["endpoint", "model", "requests"]);
    assert_eq!(data["endpoint"], "/v1/chat/completions");
    assert_eq!(data["model"], json!(model_id));
    assert_eq!(data["requests"][0]["custom_id"], "0");
    assert!(data["requests"][0]["body"].get("stream").is_none());
    let all = server.received_requests().await.unwrap();
    assert!(!all.iter().any(|r| r.url.path().contains("files")), "no file upload");
}

// spec: protocols/openrouter/batches_spec.rb:41 restores scalar and array embedding results in request and vector order
#[tokio::test]
async fn restores_scalar_and_array_embedding_results_in_request_and_vector_order() {
    let server = MockServer::start().await;
    let config = config(&server);
    let texts: [EmbedInput; 3] = ["Ruby".into(), vec!["Rails".to_string()].into(), vec!["AI".to_string(), "Ruby".to_string()].into()];
    let requests: Vec<_> = texts.into_iter().map(|t| embed_later(t, embed_options(&config)).unwrap()).collect();
    let rows = json!([
        embedding_row("2:array", &[vec![5.0, 6.0], vec![7.0, 8.0]], None),
        embedding_row("0", &[vec![1.0, 2.0]], None),
        embedding_row("1:array", &[vec![3.0, 4.0]], None)
    ]);
    let data = batch_data(EMBEDDING_MODEL, rows, "/v1/embeddings", "completed", 3);
    stub(&server, "POST", ENDPOINT, data.clone()).await;
    stub(&server, "GET", &format!("{ENDPOINT}/batch-ruby"), data).await;

    let mut batch = rust_llm::batch(requests).await.unwrap();
    let posts = requests_to(&server, "POST", ENDPOINT).await;
    let sent: Value = serde_json::from_slice(&posts[0].body).unwrap();
    let ids: Vec<&Value> = sent["requests"].as_array().unwrap().iter().map(|r| &r["custom_id"]).collect();
    assert_eq!(ids, [&json!("0"), &json!("1:array"), &json!("2:array")]);

    let expected = vec![
        Some(Vectors::Single(vec![1.0, 2.0])),
        Some(Vectors::Batch(vec![vec![3.0, 4.0]])),
        Some(Vectors::Batch(vec![vec![5.0, 6.0], vec![7.0, 8.0]])),
    ];
    assert_eq!(vectors(&batch.results().await.unwrap()), expected);
    let mut restored = Batch::find_with_config(config.clone(), batch.id(), Some("openrouter")).await.unwrap();
    assert_eq!(vectors(&restored.results().await.unwrap()), expected);
    assert_eq!(restored.tokens().await.unwrap().input, Some(12));
    let total = restored.cost().await.unwrap().total().unwrap();
    assert!((total - 0.0003).abs() < 0.0000001, "{total}");
}

// spec: protocols/openrouter/batches_spec.rb:61 fails only the embedding result with invalid positions (all 7 cases)
#[tokio::test]
async fn fails_only_the_embedding_result_with_invalid_positions() {
    let cases: Vec<Vec<Value>> = vec![
        vec![json!(0), json!(0)],
        vec![json!(0), json!(2)],
        vec![json!(-1), json!(0)],
        vec![Value::Null, json!(0)],
        vec![json!("0"), json!(1)],
        vec![json!(0.0), json!(1)],
        vec![json!(1)],
    ];
    for positions in cases {
        let server = MockServer::start().await;
        let config = config(&server);
        let id = if positions.len() == 1 { "0" } else { "0:array" };
        let invalid = embedding_row(id, &vec![vec![1.0, 2.0]; positions.len()], Some(positions.clone()));
        let rows = json!([invalid, embedding_row("1:array", &[vec![3.0, 4.0], vec![5.0, 6.0]], None)]);
        stub(&server, "GET", &format!("{ENDPOINT}/batch-ruby"), batch_data(EMBEDDING_MODEL, rows, "/v1/embeddings", "completed", 2)).await;
        let mut batch = Batch::find_with_config(config, "batch-ruby", Some("openrouter")).await.unwrap();

        let results = batch.results().await.unwrap();
        assert!(results[0].is_none(), "{positions:?}");
        assert_eq!(vectors(&results)[1], Some(Vectors::Batch(vec![vec![3.0, 4.0], vec![5.0, 6.0]])), "{positions:?}");
        assert_eq!(batch.statuses(), [Some(BatchStatus::Failed), Some(BatchStatus::Succeeded)], "{positions:?}");
        assert_eq!(batch.tokens().await.unwrap().input, Some(4), "{positions:?}");
        assert_eq!(batch.cost().await.unwrap().total(), Some(0.0001), "{positions:?}");
    }
}

// spec: protocols/openrouter/batches_spec.rb:76 fails only the embedding result with a missing position
#[tokio::test]
async fn fails_only_the_embedding_result_with_a_missing_position() {
    let server = MockServer::start().await;
    let config = config(&server);
    let mut invalid = embedding_row("0", &[vec![1.0, 2.0]], None);
    invalid["response"]["body"]["data"][0].as_object_mut().unwrap().remove("index");
    let rows = json!([invalid, embedding_row("1", &[vec![3.0, 4.0]], None)]);
    stub(&server, "GET", &format!("{ENDPOINT}/batch-ruby"), batch_data(EMBEDDING_MODEL, rows, "/v1/embeddings", "completed", 2)).await;
    let mut batch = Batch::find_with_config(config, "batch-ruby", Some("openrouter")).await.unwrap();

    let results = batch.results().await.unwrap();
    assert!(results[0].is_none());
    assert_eq!(vectors(&results)[1], Some(Vectors::Single(vec![3.0, 4.0])));
    assert_eq!(batch.statuses(), [Some(BatchStatus::Failed), Some(BatchStatus::Succeeded)]);
}

// spec: protocols/openrouter/batches_spec.rb:89 normalizes Responses results and preserves per-request failures after a fresh find
#[tokio::test]
async fn normalizes_responses_results_and_preserves_per_request_failures_after_a_fresh_find() {
    let server = MockServer::start().await;
    let config = config(&server);
    let body = json!({ "model": MODEL, "status": "completed",
        "output": [{ "type": "message", "role": "assistant", "content": [{ "type": "output_text", "text": "Ruby." }] }],
        "usage": { "input_tokens": 2, "output_tokens": 3, "cost": 0.0001 } });
    let rows = json!([{ "custom_id": "1", "response": null, "error": { "message": "Unavailable" } }, response_row("0", body)]);
    stub(&server, "GET", &format!("{ENDPOINT}/batch-ruby"), batch_data(MODEL, rows, "/v1/responses", "completed", 2)).await;
    let mut batch = Batch::find_with_config(config, "batch-ruby", Some("openrouter")).await.unwrap();

    let messages = batch.messages().await.unwrap();
    let first = messages[0].as_ref().expect("first");
    assert_eq!((first.content(), first.finish_reason.clone()), ("Ruby.", Some(FinishReason::Stop)));
    assert!(messages[1].is_none());
    assert_eq!(batch.statuses(), [Some(BatchStatus::Succeeded), Some(BatchStatus::Failed)]);
    let tokens = batch.tokens().await.unwrap();
    assert_eq!((tokens.input, tokens.output), (Some(2), Some(3)));
}

// spec: protocols/openrouter/batches_spec.rb:104 rejects multimodal requests and unsupported embedding options before submitting
// (the embedding half stubs `EmbeddingRequest#render` to add `input_type`, which `embed_later` never
// renders; it is the `openrouter_rejects_embedding_options_before_submitting` unit test in batch.rs)
#[tokio::test]
async fn rejects_multimodal_requests_before_submitting() {
    let server = MockServer::start().await;
    let config = config(&server);
    let image = Attachment::new(format!("{}/tests/fixtures/ruby.png", env!("CARGO_MANIFEST_DIR")));
    let chat = staged_chat(&config, "Describe.", vec![image]);

    let err = rust_llm::batch(chat).await.unwrap_err();

    assert!(matches!(err, Error::Argument(_)), "{err:?}");
    assert_eq!(err.to_string(), "OpenRouter batches accept text input and output only");
    assert!(requests_to(&server, "POST", ENDPOINT).await.is_empty());
}

// spec: protocols/openrouter/batches_spec.rb:114 keeps missing results pending and rejects duplicate IDs and unavailable cancellation
#[tokio::test]
async fn keeps_missing_results_pending_and_rejects_duplicate_ids_and_unavailable_cancellation() {
    let server = MockServer::start().await;
    let config = config(&server);
    let at = format!("{ENDPOINT}/batch-ruby");
    stub(&server, "GET", &at, batch_data(MODEL, Value::Null, "/v1/chat/completions", "in_progress", 2)).await;
    let mut batch = Batch::find_with_config(config, "batch-ruby", Some("openrouter")).await.unwrap();

    assert!(batch.results().await.unwrap().iter().all(Option::is_none));
    assert_eq!(batch.results().await.unwrap().len(), 2);
    assert_eq!(batch.status(), BatchStatus::Pending);
    let err = batch.cancel().await.unwrap_err();
    assert!(matches!(err, Error::Api(..)) && err.to_string().contains("does not expose batch cancellation"), "{err:?}");

    server.reset().await;
    let rows = json!([embedding_row("0", &[vec![1.0, 2.0]], None), embedding_row("0", &[vec![3.0, 4.0]], None)]);
    stub(&server, "GET", &at, batch_data(EMBEDDING_MODEL, rows, "/v1/embeddings", "completed", 2)).await;
    let err = batch.results().await.unwrap_err();
    assert!(matches!(err, Error::Api(..)), "{err:?}");
    assert_eq!(err.to_string(), "Duplicate batch result index: 0");
}

// spec: protocols/openrouter/batches_spec.rb:127 does not repeat a submission whose outcome is uncertain
#[tokio::test]
async fn does_not_repeat_a_submission_whose_outcome_is_uncertain() {
    let server = MockServer::start().await;
    let mut retrying = (*config(&server)).clone();
    retrying.max_retries = 2;
    let config = Arc::new(retrying);
    Mock::given(method("POST"))
        .and(path(ENDPOINT))
        .respond_with(ResponseTemplate::new(502).set_body_raw(r#"{"error":{"message":"Unavailable"}}"#, "application/json"))
        .mount(&server)
        .await;
    let chat = staged_chat(&config, "Ruby.", Vec::new());

    let err = rust_llm::batch(chat).await.unwrap_err();

    assert!(err.to_string().contains("Unavailable"), "{err:?}");
    assert_eq!(requests_to(&server, "POST", ENDPOINT).await.len(), 1);
}

// spec: protocols/openrouter/batches_spec.rb:136 uses the reported aggregate invoice without assigning that amount to individual results
#[tokio::test]
async fn uses_the_reported_aggregate_invoice_without_assigning_that_amount_to_individual_results() {
    let server = MockServer::start().await;
    let config = config(&server);
    let mut data = batch_data(EMBEDDING_MODEL, json!([embedding_row("0", &[vec![1.0, 2.0]], None)]), "/v1/embeddings", "completed", 1);
    data["usage"] = json!({ "cost": 0.00004, "prompt_tokens": 99 });
    stub(&server, "GET", &format!("{ENDPOINT}/batch-ruby"), data).await;
    let mut batch = Batch::find_with_config(config, "batch-ruby", Some("openrouter")).await.unwrap();

    assert!(batch.reported_cost().is_some());
    assert_eq!(batch.cost().await.unwrap().total(), Some(0.00004));
    let results = batch.results().await.unwrap();
    assert_eq!(results[0].as_ref().unwrap().cost().total(), Some(0.0001));
    assert_eq!(batch.tokens().await.unwrap().input, Some(4));
}
