//! `spec/ruby_llm/protocols/chat_completions/embeddings_spec.rb` (sparse vectors) and
//! `spec/ruby_llm/protocols/gemini/embeddings_spec.rb` (`taskType`, `title`, provider options).
//! RubyLLM calls `parse_embedding_response`/`render_embedding_payload` directly; here a wiremock
//! server answers `embed`, so the real render and parse paths run and the request body is asserted.

use std::collections::BTreeMap;
use std::sync::Arc;

use rust_llm::{Config, EmbedOptions, Embedding, SparseVectors, Vectors, embed};
use serde_json::{Value, json};
use wiremock::matchers::any;
use wiremock::{Mock, MockServer, ResponseTemplate};

async fn server(body: Value) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(any())
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&server)
        .await;
    server
}

fn config(server: &MockServer) -> Arc<Config> {
    let mut c = Config::default();
    c.set("openai_api_base", format!("{}/v1", server.uri()));
    c.set("openai_api_key", "test");
    c.set("gemini_api_base", format!("{}/v1beta", server.uri()));
    c.set("gemini_api_key", "test");
    c.max_retries = 0;
    Arc::new(c)
}

async fn only_body(server: &MockServer) -> Value {
    let requests = server.received_requests().await.unwrap_or_default();
    assert_eq!(requests.len(), 1, "expected exactly one request");
    serde_json::from_slice(&requests[0].body).expect("json body")
}

// ---- chat_completions/embeddings_spec.rb --------------------------------------------------------

/// `parse(rows, text:)`: an OpenAI-compatible embeddings response for `bge-m3`.
async fn parse(rows: Value, text: impl Into<rust_llm::embedding::EmbedInput>) -> Embedding {
    let server = server(json!({ "data": rows, "usage": { "prompt_tokens": 8 } })).await;
    embed(
        text,
        EmbedOptions {
            model: Some("bge-m3"),
            provider: Some("openai"),
            assume_model_exists: true,
            config: Some(config(&server)),
            ..Default::default()
        },
    )
    .await
    .unwrap()
}

fn weights(pairs: &[(i64, f64)]) -> BTreeMap<i64, f64> {
    pairs.iter().copied().collect()
}

fn texts(ts: &[&str]) -> Vec<String> {
    ts.iter().map(|t| t.to_string()).collect()
}

// spec: protocols/chat_completions/embeddings_spec.rb:25 #parse_embedding_response > reports no sparse vectors when the server returns dense ones only
#[tokio::test]
async fn reports_no_sparse_vectors_for_dense_only_responses() {
    let embedding = parse(json!([{ "embedding": [0.1, 0.2] }]), "Ruby").await;
    assert_eq!(embedding.vectors, Vectors::Single(vec![0.1, 0.2]));
    assert_eq!(embedding.sparse_vectors, None);
}

// spec: protocols/chat_completions/embeddings_spec.rb:32 #parse_embedding_response > reads the sparse vector BGE-M3 returns as lexical_weights
#[tokio::test]
async fn reads_lexical_weights() {
    let embedding = parse(
        json!([{ "embedding": [0.1], "lexical_weights": { "1037": 0.25, "2003": 0.5 } }]),
        "Ruby",
    )
    .await;
    assert_eq!(
        embedding.sparse_vectors,
        Some(SparseVectors::Single(weights(&[(1037, 0.25), (2003, 0.5)])))
    );
}

// spec: protocols/chat_completions/embeddings_spec.rb:39 #parse_embedding_response > reads the sparse vector other servers return as sparse_embedding
#[tokio::test]
async fn reads_sparse_embedding() {
    let embedding = parse(
        json!([{ "embedding": [0.1], "sparse_embedding": { "42": 1 } }]),
        "Ruby",
    )
    .await;
    assert_eq!(
        embedding.sparse_vectors,
        Some(SparseVectors::Single(weights(&[(42, 1.0)])))
    );
}

// spec: protocols/chat_completions/embeddings_spec.rb:45 #parse_embedding_response > shapes sparse vectors like dense ones when an array of texts was embedded
#[tokio::test]
async fn shapes_sparse_vectors_like_dense_ones_for_arrays() {
    let embedding = parse(
        json!([
            { "embedding": [0.1], "lexical_weights": { "1": 0.5 } },
            { "embedding": [0.2], "lexical_weights": { "2": 0.75 } }
        ]),
        texts(&["Ruby", "Python"]),
    )
    .await;
    assert_eq!(
        embedding.vectors,
        Vectors::Batch(vec![vec![0.1], vec![0.2]])
    );
    assert_eq!(
        embedding.sparse_vectors,
        Some(SparseVectors::Batch(vec![
            Some(weights(&[(1, 0.5)])),
            Some(weights(&[(2, 0.75)]))
        ]))
    );
}

// spec: protocols/chat_completions/embeddings_spec.rb:58 #parse_embedding_response > keeps the array shape when only some rows carry a sparse vector
#[tokio::test]
async fn keeps_the_array_shape_when_only_some_rows_are_sparse() {
    let embedding = parse(
        json!([
            { "embedding": [0.1], "lexical_weights": { "1": 0.5 } },
            { "embedding": [0.2] }
        ]),
        texts(&["Ruby", "Python"]),
    )
    .await;
    assert_eq!(
        embedding.sparse_vectors,
        Some(SparseVectors::Batch(vec![Some(weights(&[(1, 0.5)])), None]))
    );
}

// ---- gemini/embeddings_spec.rb -----------------------------------------------------------------

fn gemini_embeddings(n: usize) -> Value {
    json!({ "embeddings": (0..n).map(|_| json!({ "values": [0.1] })).collect::<Vec<_>>() })
}

fn gemini_options<'a>(server: &MockServer) -> EmbedOptions<'a> {
    EmbedOptions {
        model: Some("gemini-embedding-001"),
        provider: Some("gemini"),
        config: Some(config(server)),
        ..Default::default()
    }
}

// spec: protocols/gemini/embeddings_spec.rb:20 #render_embedding_payload > adds taskType and title to each request
#[tokio::test]
async fn gemini_adds_task_type_and_title_to_each_request() {
    let server = server(gemini_embeddings(2)).await;
    embed(
        texts(&["one", "two"]),
        EmbedOptions {
            task_type: Some("RETRIEVAL_DOCUMENT"),
            title: Some("Docs"),
            ..gemini_options(&server)
        },
    )
    .await
    .unwrap();
    assert_eq!(
        only_body(&server).await,
        json!({
            "requests": [
                {
                    "model": "models/gemini-embedding-001",
                    "content": { "parts": [{ "text": "one" }] },
                    "taskType": "RETRIEVAL_DOCUMENT",
                    "title": "Docs"
                },
                {
                    "model": "models/gemini-embedding-001",
                    "content": { "parts": [{ "text": "two" }] },
                    "taskType": "RETRIEVAL_DOCUMENT",
                    "title": "Docs"
                }
            ]
        })
    );
}

// spec: protocols/gemini/embeddings_spec.rb:48 #render_embedding_payload > lets provider options override the rendered requests
#[tokio::test]
async fn gemini_provider_options_override_the_rendered_requests() {
    let server = server(gemini_embeddings(1)).await;
    embed(
        "one",
        EmbedOptions {
            task_type: Some("RETRIEVAL_DOCUMENT"),
            provider_options: json!({ "requests": [{ "model": "models/gemini-embedding-001", "taskType": "RETRIEVAL_QUERY" }] }),
            ..gemini_options(&server)
        },
    )
    .await
    .unwrap();
    assert_eq!(
        only_body(&server).await,
        json!({ "requests": [{ "model": "models/gemini-embedding-001", "taskType": "RETRIEVAL_QUERY" }] })
    );
}
