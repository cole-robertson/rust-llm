//! `RubyLLM.rerank`, replayed from RubyLLM's `rerank_*` cassettes. Assertions follow
//! `spec/ruby_llm/rerank_spec.rb`. The Cohere cassettes need the Cohere provider, which RustLLM
//! does not port.

mod support;

use rust_llm::message::Operation;
use rust_llm::{RerankOptions, rerank};
use support::{Cassette, config_for};

/// "orders documents by relevance and reports the exact cost".
#[tokio::test]
async fn openrouter_orders_documents_by_relevance_and_reports_the_exact_cost() {
    let name = "rerank_reranking_with_openrouter_voyageai_rerank-2_5-lite_orders_documents_by_relevance_and_reports_the_exact_cost";
    let cassette = Cassette::start(name).await.expect("run bin/convert-cassettes 'rerank_*'");
    let config = config_for(&cassette, "openrouter");
    let documents = ["Paris is the capital of France", "Ruby is a programming language created by Matz"];
    let options = RerankOptions { provider: Some("openrouter"), assume_model_exists: true, config: Some(config), ..Default::default() };
    let result = rerank("what is ruby", &documents, "voyageai/rerank-2.5-lite", options).await.unwrap();

    assert!(result.results[0].document.contains("Ruby"));
    assert!(result.results[0].score > result.results[1].score);
    let mut indexes: Vec<usize> = result.results.iter().map(|r| r.index).collect();
    indexes.sort();
    assert_eq!(indexes, vec![0, 1]);
    assert_eq!(result.tokens().reported_cost, Some(3.4e-7));
    assert_eq!(result.tokens().input, Some(17));
    assert_eq!(result.cost().total(), Some(3.4e-7));
    assert_eq!(result.usage_entries.len(), 1);
    assert_eq!(result.usage_entries[0].operation, Operation::Rerank);
    cassette.assert_all_matched().await;
}

/// "raises clearly for providers without a rerank endpoint".
#[tokio::test]
async fn providers_without_a_rerank_endpoint_fail_clearly() {
    let mut config = rust_llm::Config::default();
    config.set("anthropic_api_key", "test");
    let options = RerankOptions { provider: Some("anthropic"), config: Some(config.into()), ..Default::default() };
    let err = rerank("query", &["doc"], "claude-haiku-4-5", options).await.unwrap_err();
    assert!(err.to_string().contains("doesn't support reranking"), "{err}");
}
