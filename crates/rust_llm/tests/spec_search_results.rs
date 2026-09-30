//! Ports of the remaining examples in RubyLLM 2.0's `search_results_spec.rb`.

use rust_llm::SearchResults;
use serde_json::{Value, json};

// spec: search_results_spec.rb:12 accepts multiple results
#[test]
fn accepts_multiple_results() {
    let results = SearchResults::new(vec![
        json!({ "title": "A", "text": "one" }),
        json!({ "title": "B", "text": "two" }),
    ])
    .unwrap();
    let titles: Vec<&Value> = results.results.iter().map(|r| &r["title"]).collect();
    assert_eq!(titles, [&json!("A"), &json!("B")]);
}

// spec: search_results_spec.rb:51 .from_content > returns nil for unrelated JSON
#[test]
fn from_content_returns_nil_for_unrelated_json() {
    assert_eq!(
        SearchResults::from_content(Some(r#"{"weather":"sunny"}"#)),
        None
    );
    assert_eq!(
        SearchResults::from_content(Some(r#"{"search_results":[]}"#)),
        None
    );
}
