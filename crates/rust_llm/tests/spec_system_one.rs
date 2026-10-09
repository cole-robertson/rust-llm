//! Ports of the protocol-level examples in RubyLLM 2.0's `protocols/system_one_spec.rb`. The
//! spec calls the protocol's private `render_question` / `parse_judgment_response` directly; the
//! port's System One protocol lives inside `judge.rs`, so these drive it through `Judge` against
//! a local stub that answers with the spec's `body` (or a modified one).

use std::sync::Arc;

use rust_llm::{Answer, Config, Error, ErrorKind, Judge, list_judgment_models};
use serde_json::{Value, json};
use wiremock::{Mock, MockServer, ResponseTemplate, matchers};

/// `model_for(:typesafe, :judgment)`.
const MODEL: &str = "jev-latest";

fn config(server: &MockServer) -> Arc<Config> {
    let mut c = Config::default();
    c.set("typesafe_api_base", server.uri());
    c.set("typesafe_api_key", "test");
    c.max_retries = 0;
    Arc::new(c)
}

/// The spec's `body`.
fn body() -> Value {
    json!({
        "model": "jev-1.13.0",
        "answers": {
            "urgent": { "type": "noul", "noul": 0.9 },
            "team": { "type": "choice", "choice": "Billing & payments", "confidence": 0.8,
                      "probabilities": { "Billing & payments": 0.9, "other": 0.1 } },
            "severity": { "type": "score", "score": 0.1, "confidence": 0.8,
                          "legend": { "1": ["Major", "Blocking"], "0": "Minor" },
                          "probabilities": { "1": 0.1, "0": 0.9 } }
        },
        "usage": { "input_tokens": 100, "output_tokens": 20 }
    })
}

async fn answering(body: Value) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(matchers::method("POST"))
        .and(matchers::path("/v1/systemone"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&server)
        .await;
    server
}

/// The spec's `questions`: urgent (probability), team (choice), severity (score).
fn judge(server: &MockServer) -> Judge {
    Judge::new()
        .with_config(config(server))
        .model(MODEL)
        .probability_with("urgent", None, Value::Null, Value::Null)
        .unwrap()
        .choice(
            "team",
            None,
            json!({ "Billing & payments": null, "other": "Other" }),
        )
        .unwrap()
        .score("severity", None, json!(["Minor", ["Major", "Blocking"]]))
        .unwrap()
}

// UPSTREAM-REMOVED in 2.1 (was spec: protocols/system_one_spec.rb:49) enforces provider limits without putting them in the domain
#[tokio::test]
async fn enforces_provider_limits_without_putting_them_in_the_domain() {
    let server = answering(body()).await;
    let options: serde_json::Map<String, Value> =
        (0..256).map(|n| (n.to_string(), Value::Null)).collect();
    // The questions themselves are valid (the domain has no limit); only rendering refuses them.
    let choice = Judge::new()
        .with_config(config(&server))
        .model(MODEL)
        .choice("team", None, Value::Object(options))
        .unwrap();
    let score = Judge::new()
        .with_config(config(&server))
        .model(MODEL)
        .score("score", None, json!(vec!["A level"; 11]))
        .unwrap();

    let err = choice.judge("Help").await.unwrap_err();
    assert!(matches!(err, Error::Argument(_)), "{err:?}");
    assert!(err.to_string().contains("255"), "{err}");
    let err = score.judge("Help").await.unwrap_err();
    assert!(matches!(err, Error::Argument(_)), "{err:?}");
    assert!(err.to_string().contains("10"), "{err}");
    assert!(server.received_requests().await.unwrap().is_empty());
}

// spec: protocols/system_one_spec.rb:65 parses all result fields and preserves declared key types
#[tokio::test]
async fn parses_all_result_fields_and_preserves_declared_key_types() {
    let server = answering(body()).await;
    let result = judge(&server).judge("Help").await.unwrap();

    assert_eq!(result.model, "jev-1.13.0");
    // `result.raw` is the response it parsed.
    assert_eq!(result.raw.as_ref().unwrap().body, body());
    let Answer::Choice {
        choice,
        probabilities,
        ..
    } = result.get("team").unwrap()
    else {
        panic!("team is a choice")
    };
    assert_eq!(choice, "Billing & payments");
    assert_eq!(
        probabilities,
        &vec![
            ("Billing & payments".to_string(), 0.9),
            ("other".to_string(), 0.1)
        ]
    );
    let Answer::Score {
        levels,
        probabilities,
        ..
    } = result.get("severity").unwrap()
    else {
        panic!("severity is a score")
    };
    assert_eq!(levels, &vec![json!("Minor"), json!(["Major", "Blocking"])]);
    // Integer level keys, in declared order.
    let keys: Vec<usize> = probabilities.iter().map(|(k, _)| *k).collect();
    assert_eq!(keys, [0, 1]);
    assert_eq!(result.tokens().output, Some(20));
}

// UPSTREAM-REMOVED in 2.1 (was spec: protocols/system_one_spec.rb:77) rejects missing and unexpected answers rather than returning partial results
#[tokio::test]
async fn rejects_missing_and_unexpected_answers_rather_than_returning_partial_results() {
    let mut body = body();
    body["answers"].as_object_mut().unwrap().remove("urgent");
    let server = answering(body).await;
    let err = judge(&server).judge("Help").await.unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Api);
    assert!(err.to_string().contains("different question IDs"), "{err}");
}

// UPSTREAM-REMOVED in 2.1 (was spec: protocols/system_one_spec.rb:84) rejects incorrect answer types, out-of-range probabilities, and unrecognized options
#[tokio::test]
async fn rejects_incorrect_answer_types_out_of_range_probabilities_and_unrecognized_options() {
    let modifications: [fn(&mut Value); 6] = [
        |b| b["answers"]["urgent"]["type"] = json!("choice"),
        |b| b["answers"]["urgent"]["noul"] = json!(1.1),
        |b| b["answers"]["team"]["choice"] = json!("unknown"),
        |b| b["answers"]["team"]["confidence"] = json!("0.9"),
        |b| b["answers"]["severity"]["score"] = json!(-1),
        |b| {
            b["answers"]["severity"]["legend"]
                .as_object_mut()
                .unwrap()
                .remove("0");
        },
    ];
    for change in modifications {
        let mut body = body();
        change(&mut body);
        let server = answering(body.clone()).await;
        let err = judge(&server).judge("Help").await.unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Api, "{err:?}");
        // `error.response` is the response that failed to parse.
        let response = err.response().expect("the error keeps the response");
        assert_eq!(serde_json::from_str::<Value>(&response.body).unwrap(), body);
    }
}

// spec: protocols/system_one_spec.rb:96 preserves unknown usage rather than replacing it with zero
#[tokio::test]
async fn preserves_unknown_usage_rather_than_replacing_it_with_zero() {
    let mut body = body();
    body["usage"] = json!({});
    let server = answering(body).await;
    let result = judge(&server).judge("Help").await.unwrap();
    assert_eq!(result.tokens().input, None);
}

// spec: protocols/system_one_spec.rb:102 normalizes validation error details
#[tokio::test]
async fn normalizes_validation_error_details() {
    let server = MockServer::start().await;
    Mock::given(matchers::method("POST"))
        .respond_with(ResponseTemplate::new(422).set_body_json(json!({
            "detail": [{ "loc": ["body", "questions", "urgent"], "msg": "Invalid question" }]
        })))
        .mount(&server)
        .await;
    let err = judge(&server).judge("Help").await.unwrap_err();
    assert_eq!(err.to_string(), "body.questions.urgent: Invalid question");
}

// spec: protocols/system_one_spec.rb:126 parses catalog facts without inventing limits or pricing
#[tokio::test]
async fn parses_catalog_facts_without_inventing_limits_or_pricing() {
    let server = MockServer::start().await;
    Mock::given(matchers::method("GET"))
        .and(matchers::path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "models": [{ "name": MODEL, "description": "System One model",
                         "release_date": "2026-09-10T18:38:01Z" }]
        })))
        .mount(&server)
        .await;
    let entry = list_judgment_models(Some(config(&server)))
        .await
        .unwrap()
        .remove(0);

    assert_eq!(entry.id, MODEL);
    assert_eq!(entry.model_type(), rust_llm::model::ModelType::Judgment);
    assert!(entry.supports("judgment"));
    assert_eq!(entry.context_window, None);
    // `pricing.to_h == {}`: no tier carries a price.
    assert_eq!(entry.pricing, rust_llm::model::Pricing::default());
}
