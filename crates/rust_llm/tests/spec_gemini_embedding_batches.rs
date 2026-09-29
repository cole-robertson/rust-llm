//! `spec/ruby_llm/protocols/gemini/embedding_batches_spec.rb`: Gemini embedding batches
//! (`protocols/gemini/embedding_batches.rb`) through the public `Batch` API, with the stubbed
//! connection served by wiremock, plus the live example replayed from its cassette. The examples
//! that call private protocol methods directly (`protocol.send(:embedding_batch_requests, ...)`)
//! are `#[cfg(test)]` unit tests in `src/batch.rs` citing the same spec lines.

mod spec_helpers;
mod support;

use std::sync::Arc;

use rust_llm::embedding::EmbedInput;
use rust_llm::{Batch, Config, EmbedOptions, Vectors, embed_later};
use serde_json::{Value, json};
use spec_helpers::config;
use support::{Cassette, config_for};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// `model_for(:gemini, :embedding)`.
const MODEL: &str = "gemini-embedding-001";

fn options(config: &Arc<Config>, dimensions: Option<i64>) -> EmbedOptions<'static> {
    EmbedOptions {
        model: Some(MODEL),
        provider: Some("gemini"),
        dimensions,
        config: Some(config.clone()),
        ..Default::default()
    }
}

/// `inline_response(request, vector)`: an answer carrying the request's metadata, 2 prompt tokens.
fn inline_response(request: &Value, vector: &[f64]) -> Value {
    json!({ "metadata": request["metadata"], "response": { "embedding": { "values": vector }, "usageMetadata": { "promptTokenCount": 2 } } })
}

fn vectors(results: &[Option<rust_llm::BatchResult>]) -> Vec<Option<Vectors>> {
    results
        .iter()
        .map(|r| {
            r.as_ref()
                .and_then(|r| r.as_embedding())
                .map(|e| e.vectors.clone())
        })
        .collect()
}

// spec: protocols/gemini/embedding_batches_spec.rb:23 stages scalar input as embedContent and arrays as batchEmbedContents payloads
#[test]
fn stages_scalar_input_as_embed_content_and_arrays_as_batch_embed_contents_payloads() {
    let mut offline = Config::default();
    offline.set("gemini_api_key", "test");
    let config = Arc::new(offline);
    let scalar = embed_later("Ruby", options(&config, Some(64)))
        .unwrap()
        .render()
        .unwrap();
    let array = embed_later(vec!["Ruby".to_string()], options(&config, Some(64)))
        .unwrap()
        .render()
        .unwrap();

    assert_eq!(scalar["content"], json!({ "parts": [{ "text": "Ruby" }] }));
    assert_eq!(scalar["outputDimensionality"], 64);
    assert!(scalar.get("requests").is_none());
    assert_eq!(array["requests"], json!([scalar]));
}

// spec: protocols/gemini/embedding_batches_spec.rb:32 submits an asynchronous embedding batch with explicit result correlation metadata
#[tokio::test]
async fn submits_an_asynchronous_embedding_batch_with_explicit_result_correlation_metadata() {
    let server = MockServer::start().await;
    let config = config(&server);
    let at = format!("/v1beta/models/{MODEL}:asyncBatchEmbedContent");
    Mock::given(method("POST"))
        .and(path(at.as_str()))
        .respond_with(ResponseTemplate::new(200).set_body_json(
            json!({ "name": "batches/abc", "metadata": { "state": "BATCH_STATE_PENDING" } }),
        ))
        .mount(&server)
        .await;
    let requests = vec![
        embed_later("Ruby", options(&config, Some(64))).unwrap(),
        embed_later(
            vec!["Python".to_string(), "Rust".to_string()],
            options(&config, Some(64)),
        )
        .unwrap(),
    ];

    let batch = rust_llm::batch(requests).await.unwrap();

    assert_eq!((batch.id(), batch.is_complete()), ("batches/abc", false));
    let sent = server.received_requests().await.unwrap();
    assert_eq!(sent.len(), 1);
    assert_eq!(sent[0].url.path(), at);
    let body: Value = serde_json::from_slice(&sent[0].body).unwrap();
    assert!(
        body["batch"]["displayName"]
            .as_str()
            .unwrap()
            .starts_with("ruby_llm_")
    );
    let metadata: Vec<&Value> = body["batch"]["inputConfig"]["requests"]["requests"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| &r["metadata"])
        .collect();
    assert_eq!(
        metadata,
        [
            &json!({ "custom_id": "0", "model": MODEL, "array_input": false, "embedding_index": 0, "embedding_count": 1 }),
            &json!({ "custom_id": "1", "model": MODEL, "array_input": true, "embedding_index": 0, "embedding_count": 2 }),
            &json!({ "custom_id": "1", "model": MODEL, "array_input": true, "embedding_index": 1, "embedding_count": 2 }),
        ]
    );
}

// spec: protocols/gemini/embedding_batches_spec.rb:106 hydrates staged embedding requests and restores a completed batch by id
#[tokio::test]
async fn hydrates_staged_embedding_requests_and_restores_a_completed_batch_by_id() {
    let server = MockServer::start().await;
    let config = config(&server);
    let texts: [EmbedInput; 3] = [
        "Ruby".into(),
        vec!["Rails".to_string()].into(),
        vec!["Python".to_string(), "Rust".to_string()].into(),
    ];
    let requests: Vec<_> = texts
        .into_iter()
        .map(|t| embed_later(t, options(&config, None)).unwrap())
        .collect();
    // `protocol.send(:embedding_batch_requests, ...)`: the inline requests as Ruby stages them.
    let mut output = Vec::new();
    for (index, (count, array)) in [(1, false), (1, true), (2, true)].into_iter().enumerate() {
        for position in 0..count {
            let input = json!({ "metadata": { "custom_id": index.to_string(), "model": MODEL, "array_input": array, "embedding_index": position, "embedding_count": count } });
            output.push(inline_response(&input, &[0.1, 0.2]));
        }
    }
    output.reverse();
    let metadata = json!({ "@type": "type.googleapis.com/google.ai.generativelanguage.v1main.EmbedContentBatch", "batchStats": { "requestCount": "4" } });
    let mut pending = json!({ "name": "batches/abc", "metadata": metadata.clone() });
    pending["metadata"]["state"] = "BATCH_STATE_PENDING".into();
    let mut completed = json!({ "name": "batches/abc", "metadata": metadata, "response": { "inlinedResponses": { "inlinedResponses": output } } });
    completed["metadata"]["state"] = "BATCH_STATE_SUCCEEDED".into();
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(pending))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/v1beta/batches/abc"))
        .respond_with(ResponseTemplate::new(200).set_body_json(completed))
        .mount(&server)
        .await;

    let mut batch = rust_llm::batch(requests).await.unwrap();
    batch.refresh().await.unwrap();

    assert!(batch.is_succeeded());
    let results = batch.results().await.unwrap();
    let expected = vec![
        Some(Vectors::Single(vec![0.1, 0.2])),
        Some(Vectors::Batch(vec![vec![0.1, 0.2]])),
        Some(Vectors::Batch(vec![vec![0.1, 0.2], vec![0.1, 0.2]])),
    ];
    assert_eq!(vectors(&results), expected);
    let hydrated: Vec<Option<Vectors>> = batch
        .requests()
        .unwrap()
        .iter()
        .map(|r| r.result.as_ref().map(|e| e.vectors.clone()))
        .collect();
    assert_eq!(hydrated, expected);
    let mut found = Batch::find_with_config(config, batch.id(), Some("gemini"))
        .await
        .unwrap();
    assert_eq!(vectors(&found.results().await.unwrap()), expected);
}

// spec: protocols/gemini/embedding_batches_spec.rb:140 submits, retrieves, and cancels an asynchronous embedding batch
// The recorded `displayName` is `ruby_llm_#{SecureRandom.hex(8)}`, so that one field is excluded
// from the body comparison (as `tests/batches.rs` does for Gemini chat batches); all else must match.
#[tokio::test]
async fn submits_retrieves_and_cancels_an_asynchronous_embedding_batch() {
    let cassette = Cassette::start("protocols_gemini_embeddingbatches_submits_retrieves_and_cancels_an_asynchronous_embedding_batch")
        .await
        .expect("cassette");
    let config = config_for(&cassette, "gemini");
    let request = embed_later("Ruby", options(&config, Some(64))).unwrap();

    let batch = rust_llm::batch(vec![request]).await.unwrap();
    let mut restored = Batch::find_with_config(config, batch.id(), Some("gemini"))
        .await
        .unwrap();

    assert_eq!(restored.id(), batch.id());
    let id = restored.cancel().await.unwrap().id().to_string();
    assert_eq!(id, batch.id());
    assert!(!restored.is_failed());
    assert!(restored.is_cancelled());

    let mismatches: Vec<String> = cassette
        .mismatches
        .lock()
        .unwrap()
        .iter()
        .filter(|m| !m.contains("/batch/displayName"))
        .cloned()
        .collect();
    assert!(
        mismatches.is_empty(),
        "request bodies differ from RubyLLM's:\n  {}",
        mismatches.join("\n  ")
    );
    let received = cassette
        .server
        .received_requests()
        .await
        .unwrap_or_default()
        .len();
    assert_eq!(received, cassette.count);
}
