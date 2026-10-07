//! Judgments with TypeSafe's Jev: RubyLLM 2.0's `providers/typesafe_spec.rb` and `judge_spec.rb`.
//! The live examples replay RubyLLM's cassettes (request bodies must be JSON-equal); the stubbed
//! examples run against a local mock like the spec's WebMock stubs.

mod support;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use rust_llm::{
    Answer, Dynamic, ErrorKind, Judge, JudgeOptions, UsageStatus, list_judgment_models,
};
use serde_json::{Value, json};
use support::Cassette;
use wiremock::{Mock, MockServer, Request, ResponseTemplate, matchers};

fn config_at(base: &str, retries: u32) -> Arc<rust_llm::Config> {
    let mut c = rust_llm::Config::default();
    c.set("typesafe_api_base", base);
    c.set("typesafe_api_key", "test-key");
    c.max_retries = retries;
    c.retry_interval = 0.001;
    Arc::new(c)
}

fn opts(config: Arc<rust_llm::Config>) -> Judge {
    Judge::new().with_config(config)
}

// ---- live cassettes -----------------------------------------------------------------------------

#[tokio::test]
async fn judges_all_three_question_types_through_the_compact_dsl() {
    let cassette = Cassette::start("providers_typesafe_with_the_typesafe_api_judges_all_three_question_types_through_the_compact_dsl").await.unwrap();
    let triage = opts(config_at(&cassette.server.uri(), 0))
        .model("jev-latest")
        .probability_with(
            "urgent",
            Some("Does the customer explicitly need action today?".into()),
            "Explicitly asks for action today",
            "No deadline or a later deadline",
        )
        .unwrap()
        .choice(
            "department",
            Some("Which team should handle this message?".into()),
            json!({ "billing": "Payments and refunds", "technical": "Bugs and integrations", "other": null }),
        )
        .unwrap()
        .score(
            "frustration",
            Some("How frustrated is the customer?".into()),
            json!(["Calm and polite", "Expresses frustration", "Angry or hostile"]),
        )
        .unwrap();

    let result = triage
        .judge(
            json!({ "message": "I was charged twice. Please refund the duplicate charge today." }),
        )
        .await
        .unwrap();

    let urgent = result.probability("urgent").unwrap();
    assert!((0.0..=1.0).contains(&urgent));
    assert!(["billing", "technical", "other"].contains(&result.choice("department").unwrap()));
    let Answer::Choice { probabilities, .. } = result.get("department").unwrap() else {
        panic!()
    };
    assert_eq!(
        probabilities
            .iter()
            .map(|(k, _)| k.as_str())
            .collect::<Vec<_>>(),
        ["billing", "technical", "other"]
    );
    let score = result.score("frustration").unwrap();
    assert!((0.0..=2.0).contains(&score));
    let Answer::Score { probabilities, .. } = result.get("frustration").unwrap() else {
        panic!()
    };
    assert_eq!(
        probabilities.iter().map(|(k, _)| *k).collect::<Vec<_>>(),
        [0, 1, 2]
    );
    assert!(result.tokens().input.unwrap() > 0);
    assert!(result.tokens().output.unwrap() > 0);
    assert!(result.model.starts_with("jev-"));
    cassette.assert_all_matched().await;
}

#[tokio::test]
async fn preserves_structured_descriptions_and_arbitrary_choice_names() {
    let cassette = Cassette::start("providers_typesafe_with_the_typesafe_api_preserves_structured_descriptions_and_arbitrary_choice_names").await.unwrap();
    let questions = json!({
        "urgent": { "type": "probability", "instructions": { "question": "Is action needed today?" },
                    "criteria": { "yes": { "deadline": "today" }, "no": ["No deadline", "Later"] } },
        "department": { "type": "choice", "instructions": ["Which team?", "Pick the most relevant team"],
                        "options": { "Billing & payments": { "handles": ["Charges", "Refunds"] }, "Other": null } },
        "frustration": { "type": "score", "instructions": "How frustrated is the customer?",
                         "levels": [{ "description": "Calm and polite" }, ["Frustrated", "Still civil"], "Angry or hostile"] }
    });
    let result = opts(config_at(&cassette.server.uri(), 0))
        .model("jev-latest")
        .judge_with(
            json!(["Please refund the duplicate charge today."]),
            JudgeOptions {
                questions: questions.as_object().unwrap().clone(),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    assert!(["Billing & payments", "Other"].contains(&result.choice("department").unwrap()));
    let Answer::Score { levels, .. } = result.get("frustration").unwrap() else {
        panic!()
    };
    assert_eq!(
        levels,
        &vec![
            json!({ "description": "Calm and polite" }),
            json!(["Frustrated", "Still civil"]),
            json!("Angry or hostile")
        ]
    );
    cassette.assert_all_matched().await;
}

#[tokio::test]
async fn accepts_omitted_instructions_when_criteria_describe_the_judgment() {
    let cassette = Cassette::start("providers_typesafe_with_the_typesafe_api_accepts_omitted_instructions_when_criteria_describe_the_judgment").await.unwrap();
    let mut config = (*config_at(&cassette.server.uri(), 0)).clone();
    config.default_judgment_model = "jev-latest".into();
    let result = Judge::new()
        .with_config(Arc::new(config))
        .judge_with("Please refund my duplicate charge.", JudgeOptions {
            questions: json!({
                "department": { "type": "choice", "options": { "billing": "Payments and refunds", "technical": "Bugs" } },
                "urgent": { "type": "probability", "criteria": { "yes": "Needs action today", "no": null } }
            })
            .as_object()
            .unwrap()
            .clone(),
            ..Default::default()
        })
        .await
        .unwrap();
    assert!(["billing", "technical"].contains(&result.choice("department").unwrap()));
    assert!((0.0..=1.0).contains(&result.probability("urgent").unwrap()));
    cassette.assert_all_matched().await;
}

#[tokio::test]
async fn lists_the_available_judgment_models() {
    let cassette = Cassette::start(
        "providers_typesafe_with_the_typesafe_api_lists_the_available_judgment_models",
    )
    .await
    .unwrap();
    let models = list_judgment_models(Some(config_at(&cassette.server.uri(), 0)))
        .await
        .unwrap();
    assert!(models.iter().any(|m| m.id == "jev-latest"));
    assert!(
        models
            .iter()
            .all(|m| m.model_type() == rust_llm::model::ModelType::Judgment)
    );
    cassette.assert_all_matched().await;
}

// ---- stubbed (spec WebMock examples) ------------------------------------------------------------

fn ok_body(model: &str, probability: f64) -> Value {
    json!({ "model": model, "answers": { "urgent": { "type": "noul", "noul": probability } },
            "usage": { "input_tokens": 100, "output_tokens": 10 } })
}

fn urgent() -> Value {
    json!({ "urgent": { "type": "probability", "instructions": "Does this need attention today?" } })
}

fn questions(v: Value) -> JudgeOptions {
    JudgeOptions {
        questions: v.as_object().unwrap().clone(),
        ..Default::default()
    }
}

#[test]
fn the_bundled_catalog_resolves_jev_without_an_explicit_provider() {
    let model = rust_llm::models().find("jev-latest", None).unwrap();
    assert_eq!(model.provider, "typesafe");
    assert_eq!(model.model_type(), rust_llm::model::ModelType::Judgment);
    assert!(model.supports("judgment"));
    assert!(
        !rust_llm::models()
            .chat_models()
            .iter()
            .any(|m| m.id == "jev-latest")
    );
    assert_eq!(
        rust_llm::Config::default().default_judgment_model,
        "jev-latest"
    );
}

#[tokio::test]
async fn judges_an_unlisted_local_model_on_a_jev_compatible_server() {
    let server = MockServer::start().await;
    Mock::given(matchers::method("POST"))
        .and(matchers::path("/v1/systemone"))
        .and(matchers::header("authorization", "Bearer local"))
        .and(matchers::body_json(json!({
            "model": "english", "state": "Please help today.",
            "questions": { "urgent": { "type": "noul", "instructions": "Does this need attention today?" } }
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model": "english", "answers": { "urgent": { "type": "noul", "noul": 0.9 } },
            "usage": { "input_tokens": 40, "output_tokens": 0 }
        })))
        .expect(1)
        .mount(&server)
        .await;
    let mut local = rust_llm::Config::default();
    local.set("typesafe_api_base", server.uri());
    local.set("typesafe_api_key", "local");
    local.default_judgment_model = "english".into();

    let result = Judge::new()
        .with_config(Arc::new(local))
        .provider("typesafe")
        .assume_model_exists()
        .judge_with("Please help today.", questions(urgent()))
        .await
        .unwrap();

    assert_eq!(result.probability("urgent"), Some(0.9));
    assert_eq!(result.model, "english");
    assert_eq!(
        (result.tokens().input, result.tokens().output),
        (Some(40), Some(0))
    );
    assert_eq!(
        result.cost().total(),
        None,
        "the catalog has no pricing, so cost stays unknown"
    );
    assert_eq!(
        rust_llm::Provider::TypeSafe
            .api_base(&rust_llm::Config::default())
            .unwrap(),
        "https://api.typesafe.ai"
    );
}

// RustLLM-only: RubyLLM's bundled models.json has `"pricing": {}` for Jev, so its judgments cost
// `nil`. The bundled registry here carries TypeSafe's published price
// (https://docs.typesafe.ai/models.md): $0.042 per million input tokens, output free.
#[tokio::test]
async fn a_jev_judgment_is_priced_at_typesafes_published_rate() {
    let server = MockServer::start().await;
    Mock::given(matchers::method("POST"))
        .and(matchers::path("/v1/systemone"))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_body("jev-1.13.0", 0.9)))
        .expect(1)
        .mount(&server)
        .await;

    for model in ["jev-latest", "jev-preview"] {
        let pricing = rust_llm::models().find(model, None).unwrap().pricing;
        let standard = pricing.text_tokens.unwrap().standard.unwrap();
        assert_eq!(standard.input_per_million, Some(0.042), "{model}");
        assert_eq!(standard.output_per_million, Some(0.0), "{model}");
    }

    let result = opts(config_at(&server.uri(), 0))
        .model("jev-latest")
        .judge_with("Please help today.", questions(urgent()))
        .await
        .unwrap();

    // 100 input tokens at $0.042 per million; the 10 output tokens are free.
    let cost = result.cost();
    assert_eq!(cost.input, Some(100.0 * 0.042 / 1_000_000.0));
    assert_eq!(cost.output, Some(0.0));
    assert_eq!(cost.total(), Some(100.0 * 0.042 / 1_000_000.0));
    assert_eq!(
        result.to_value()["cost"]["total"],
        json!(100.0 * 0.042 / 1_000_000.0)
    );
}

#[tokio::test]
async fn retries_overloads_and_accounts_for_both_attempts() {
    let server = MockServer::start().await;
    Mock::given(matchers::method("POST"))
        .respond_with(ResponseTemplate::new(529).set_body_json(json!({ "detail": "Overloaded" })))
        .up_to_n_times(1)
        .with_priority(1)
        .mount(&server)
        .await;
    Mock::given(matchers::method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_body("jev-latest", 0.9)))
        .with_priority(2)
        .mount(&server)
        .await;
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = calls.clone();
    let result = opts(config_at(&server.uri(), 1))
        .model("jev-latest")
        .probability(
            "urgent",
            Dynamic::from_fn(move |_| {
                counter.fetch_add(1, Ordering::SeqCst);
                json!("Does this need attention today?")
            }),
        )
        .unwrap()
        .judge("Please help today.")
        .await
        .unwrap();

    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "dynamic values resolve once, before retries"
    );
    let statuses: Vec<UsageStatus> = result.usage_entries.iter().map(|e| e.status).collect();
    assert_eq!(statuses, [UsageStatus::Failed, UsageStatus::Succeeded]);
    assert_eq!(result.tokens().input, Some(100));
    assert_eq!(result.cost().total(), None);
    assert_eq!(
        result.usage_entries[1].operation,
        rust_llm::message::Operation::Judgment
    );
}

#[tokio::test]
async fn reports_normalized_errors_with_the_detail_message() {
    let server = MockServer::start().await;
    Mock::given(matchers::method("POST"))
        .respond_with(
            ResponseTemplate::new(401).set_body_json(json!({ "detail": "Invalid API key" })),
        )
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(matchers::method("POST"))
        .respond_with(ResponseTemplate::new(402).set_body_json(json!({
            "detail": { "message": "Your organization has no available TypeSafe API credits" }
        })))
        .mount(&server)
        .await;
    let j = opts(config_at(&server.uri(), 0)).model("jev-latest");

    let err = j.judge_with("Help", questions(urgent())).await.unwrap_err();
    assert_eq!(
        (err.kind(), err.to_string().as_str()),
        (ErrorKind::Unauthorized, "Invalid API key")
    );
    let err = j.judge_with("Help", questions(urgent())).await.unwrap_err();
    assert_eq!(err.kind(), ErrorKind::PaymentRequired);
    assert_eq!(
        err.to_string(),
        "Your organization has no available TypeSafe API credits"
    );
}

async fn capture() -> (MockServer, Arc<std::sync::Mutex<Vec<Value>>>) {
    let server = MockServer::start().await;
    let seen: Arc<std::sync::Mutex<Vec<Value>>> = Arc::default();
    let sink = seen.clone();
    Mock::given(matchers::method("POST"))
        .respond_with(move |req: &Request| {
            let body: Value = serde_json::from_slice(&req.body).unwrap();
            let names: Vec<String> = body["questions"].as_object().unwrap().keys().cloned().collect();
            sink.lock().unwrap().push(body);
            let answers: serde_json::Map<String, Value> =
                names.into_iter().map(|n| (n, json!({ "type": "noul", "noul": 0.9 }))).collect();
            ResponseTemplate::new(200).set_body_json(json!({
                "model": "jev-latest", "answers": answers, "usage": { "input_tokens": 1, "output_tokens": 1 }
            }))
        })
        .mount(&server)
        .await;
    (server, seen)
}

#[tokio::test]
async fn resolves_declared_inputs_into_questions_and_criteria() {
    let (server, seen) = capture().await;
    let configured = opts(config_at(&server.uri(), 0))
        .model("jev-latest")
        .inputs(["ticket"])
        .probability_with(
            "urgent",
            Some(Dynamic::from_fn(
                |i| json!({ "question": "Is this urgent?", "deadline": i["ticket"] }),
            )),
            Dynamic::from_fn(|i| json!(format!("Due {}", i["ticket"].as_str().unwrap()))),
            "No deadline",
        )
        .unwrap();
    let mut inputs = serde_json::Map::new();
    inputs.insert("ticket".into(), json!("today"));
    configured
        .judge_with(
            json!({ "message": "Please help today" }),
            JudgeOptions {
                inputs,
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let body = &seen.lock().unwrap()[0];
    assert_eq!(
        body["questions"]["urgent"]["instructions"],
        json!({ "question": "Is this urgent?", "deadline": "today" })
    );
    assert_eq!(
        body["questions"]["urgent"]["criteria"],
        json!({ "true": "Due today", "false": "No deadline" })
    );
    assert!(
        body.get("ticket").is_none(),
        "inputs are not sent unless used"
    );
}

#[tokio::test]
async fn validates_before_sending_anything() {
    let (server, seen) = capture().await;
    let j = opts(config_at(&server.uri(), 0)).model("jev-latest");

    let missing = j
        .clone()
        .inputs(["ticket"])
        .probability("urgent", "?")
        .unwrap()
        .judge("Help")
        .await
        .unwrap_err();
    assert!(missing.to_string().contains("Missing judge inputs: ticket"));
    assert!(
        j.clone()
            .judge("Help")
            .await
            .unwrap_err()
            .to_string()
            .contains("at least one question")
    );
    let dup = j
        .clone()
        .probability("urgent", "?")
        .unwrap()
        .probability("urgent", "again");
    assert!(dup.unwrap_err().to_string().contains("Duplicate question"));
    let non_input = j
        .clone()
        .probability("urgent", "?")
        .unwrap()
        .judge(json!(42))
        .await
        .unwrap_err();
    assert!(
        non_input
            .to_string()
            .contains("Judgment input must be text")
    );
    let bad_score = j
        .clone()
        .score("s", None, json!(["only one"]))
        .unwrap()
        .judge("x")
        .await
        .unwrap_err();
    assert!(
        bad_score
            .to_string()
            .contains("at least two non-nil levels")
    );
    let reserved = j
        .clone()
        .provider_options(json!({ "state": "sneaky" }))
        .probability("urgent", "?")
        .unwrap()
        .judge("x")
        .await
        .unwrap_err();
    assert!(
        reserved
            .to_string()
            .contains("instead of provider_options for state")
    );
    assert!(seen.lock().unwrap().is_empty(), "nothing was sent");
}

#[tokio::test]
async fn chat_models_are_rejected_for_judgments_and_jev_for_chat() {
    let err = Judge::new()
        .with_config(config_at("http://127.0.0.1:9", 0))
        .model("gpt-5-nano")
        .probability("urgent", "?")
        .unwrap()
        .judge("x")
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("doesn't support judgments"),
        "{err}"
    );

    let mut chat = rust_llm::Chat::with_config(
        config_at("http://127.0.0.1:9", 0),
        Some("jev-latest"),
        None,
        false,
    )
    .unwrap();
    let err = chat.ask("hi").await.unwrap_err();
    assert!(
        err.to_string().contains("TypeSafe doesn't support chat"),
        "{err}"
    );
}
