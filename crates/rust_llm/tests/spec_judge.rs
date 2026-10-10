//! Ports of the remaining stubbed examples in RubyLLM 2.0's `judge_spec.rb`, `judgment_spec.rb`, and
//! `judge/question_spec.rb`. The spec's WebMock stub of `POST /v1/systemone` is a wiremock server
//! that records each request body and answers every asked question with `noul: 0.9`.

use std::sync::{Arc, Mutex};

use rust_llm::judge::Question;
use rust_llm::{Answer, Config, Error, Judge, JudgeOptions, Judgment, Tokens};
use serde_json::{Map, Value, json};
use wiremock::{Mock, MockServer, Request, ResponseTemplate, matchers};

/// `model_for(:typesafe, :judgment)`.
const MODEL: &str = "jev-latest";

/// Tests that change the global configuration (`RubyLLM.config`) run one at a time.
static GLOBAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

type Seen = Arc<Mutex<Vec<(Value, Option<String>)>>>;

/// The spec's `before { stub_request(:post, .../v1/systemone) }`, keeping each body and its
/// Authorization header.
async fn stub() -> (MockServer, Seen) {
    let server = MockServer::start().await;
    let seen: Seen = Arc::default();
    let sink = seen.clone();
    Mock::given(matchers::method("POST"))
        .and(matchers::path("/v1/systemone"))
        .respond_with(move |req: &Request| {
            let body: Value = serde_json::from_slice(&req.body).unwrap();
            let auth = req.headers.get("authorization").and_then(|v| v.to_str().ok()).map(str::to_string);
            let answers: Map<String, Value> = body["questions"]
                .as_object()
                .unwrap()
                .keys()
                .map(|n| (n.clone(), json!({ "type": "noul", "noul": 0.9 })))
                .collect();
            let model = body["model"].clone();
            sink.lock().unwrap().push((body, auth));
            ResponseTemplate::new(200).set_body_json(json!({
                "model": model, "answers": answers, "usage": { "input_tokens": 100, "output_tokens": 10 }
            }))
        })
        .mount(&server)
        .await;
    (server, seen)
}

fn bodies(seen: &Seen) -> Vec<Value> {
    seen.lock()
        .unwrap()
        .iter()
        .map(|(b, _)| b.clone())
        .collect()
}

fn field(seen: &Seen, path: &[&str]) -> Vec<Value> {
    bodies(seen)
        .iter()
        .map(|b| path.iter().fold(b, |v, k| &v[*k]).clone())
        .collect()
}

/// `include_context 'with configured RubyLLM'`: TypeSafe pointed at the stub, no retries.
fn configured(server: &MockServer, default_model: &str) -> Config {
    let mut c = Config::default();
    c.set("typesafe_api_base", server.uri());
    c.set("typesafe_api_key", "test-key");
    c.max_retries = 0;
    c.default_judgment_model = default_model.into();
    c
}

/// `Class.new(RubyLLM::Judge) { model id, provider: :typesafe, assume_model_exists: true; probability :urgent, ... }`.
fn judge_class(config: Config) -> Judge {
    Judge::new()
        .with_config(Arc::new(config))
        .model(MODEL)
        .provider("typesafe")
        .assume_model_exists()
        .probability("urgent", "Does this need attention today?")
        .unwrap()
}

/// The `with a default judgment model` context's judge: no model declaration.
fn undeclared_judge() -> Judge {
    Judge::new()
        .probability("urgent", "Does this need attention today?")
        .unwrap()
}

fn one_off() -> Value {
    json!({ "urgent": { "type": "probability", "instructions": "Is this urgent?" } })
}

fn questions(v: Value) -> Map<String, Value> {
    v.as_object().unwrap().clone()
}

// ---- judge_spec.rb ----------------------------------------------------------------------------

// spec: judge_spec.rb:82 keeps successive judgments independent
#[tokio::test]
async fn keeps_successive_judgments_independent() {
    let (server, seen) = stub().await;
    let judge = judge_class(configured(&server, MODEL));
    judge.judge("First").await.unwrap();
    judge.judge("Second").await.unwrap();

    assert_eq!(field(&seen, &["state"]), [json!("First"), json!("Second")]);
}

// spec: judge_spec.rb:121 inherits configuration while allowing a subclass to replace a question
// A Ruby subclass is a clone of the Judge; redeclaring an inherited question goes through
// `replacing` (a plain redeclaration is a duplicate, as within one Ruby class).
#[tokio::test]
async fn a_derived_judge_replaces_an_inherited_question_and_keeps_the_model() {
    let (server, seen) = stub().await;
    let parent = judge_class(configured(&server, "jev-preview"));
    let child = parent
        .clone()
        .replacing("urgent")
        .probability("urgent", "Does this require an immediate response?")
        .unwrap();
    child.judge("Help").await.unwrap();
    parent.judge("Help").await.unwrap();

    assert_eq!(
        field(&seen, &["questions", "urgent", "instructions"]),
        [
            json!("Does this require an immediate response?"),
            json!("Does this need attention today?")
        ]
    );
    // `child.model == judge_class.model`: both send the declared model, not the configured default.
    assert_eq!(field(&seen, &["model"]), [json!(MODEL), json!(MODEL)]);
}

// spec: judge_spec.rb:149 supports isolated contexts and forwards provider options and instrumentation metadata
// Ports the context (tenant key) and provider_options halves; the `metadata:` instrumentation
// event half belongs to the instrumentation lane.
#[tokio::test]
async fn an_isolated_context_uses_its_own_key_and_forwards_provider_options() {
    let (server, seen) = stub().await;
    let mut base = configured(&server, MODEL);
    base.set("typesafe_api_key", "global-key");
    let mut tenant = base.clone();
    tenant.set("typesafe_api_key", "tenant-key");
    let context = rust_llm::Context::new(tenant);

    context
        .judge(
            "Help",
            one_off(),
            JudgeOptions {
                model: Some(Some(MODEL.into())),
                provider: Some("typesafe".into()),
                assume_model_exists: Some(true),
                provider_options: Some(json!({ "extension": { "enabled": true } })),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0].1.as_deref(), Some("Bearer tenant-key"));
    assert_eq!(seen[0].0["extension"], json!({ "enabled": true }));
}

// spec: judge_spec.rb:175 rejects missing or unknown runtime inputs
#[tokio::test]
async fn rejects_missing_or_unknown_runtime_inputs() {
    let (server, seen) = stub().await;
    let configured_judge = judge_class(configured(&server, MODEL)).inputs(["ticket"]);

    let missing = configured_judge.judge("Help").await.unwrap_err();
    assert!(matches!(missing, Error::Argument(_)));
    assert!(
        missing.to_string().contains("Missing judge inputs: ticket"),
        "{missing}"
    );

    let inputs = questions(json!({ "ticket": "Help", "extra": true }));
    let extra = configured_judge
        .judge_with(
            "Help",
            JudgeOptions {
                inputs,
                ..Default::default()
            },
        )
        .await
        .unwrap_err();
    assert!(matches!(extra, Error::Argument(_)));
    assert!(
        extra.to_string().contains("Unknown judge inputs: extra"),
        "{extra}"
    );
    assert!(bodies(&seen).is_empty());
}

// spec: judge_spec.rb:232 with a default judgment model > resolves the global default at each call, including inherited judges
#[tokio::test]
async fn resolves_the_global_default_at_each_call_including_derived_judges() {
    let _lock = GLOBAL.lock().await;
    let (server, seen) = stub().await;
    let previous = rust_llm::config();
    let child = undeclared_judge().clone();

    rust_llm::configure(|c| *c = configured(&server, MODEL));
    let first = child.judge("First").await;
    rust_llm::configure(|c| c.default_judgment_model = "jev-preview".into());
    let second = child.judge("Second").await;
    rust_llm::configure(|c| *c = (*previous).clone());

    first.unwrap();
    second.unwrap();
    assert_eq!(
        field(&seen, &["model"]),
        [json!(MODEL), json!("jev-preview")]
    );
}

// spec: judge_spec.rb:250 with a default judgment model > uses an isolated context default for classes and one-off questions
#[tokio::test]
async fn uses_an_isolated_context_default_for_judges_and_one_off_questions() {
    let _lock = GLOBAL.lock().await;
    let (server, seen) = stub().await;
    let previous = rust_llm::config();
    rust_llm::configure(|c| *c = configured(&server, MODEL));
    let context = rust_llm::context(|c| c.default_judgment_model = "jev-preview".into());

    let class_call = undeclared_judge()
        .judge_with(
            "Help",
            JudgeOptions {
                config: Some(context.config().clone()),
                ..Default::default()
            },
        )
        .await;
    let one_off_call = context
        .judge("Help", one_off(), JudgeOptions::default())
        .await;
    let global_default = rust_llm::config().default_judgment_model.clone();
    rust_llm::configure(|c| *c = (*previous).clone());

    class_call.unwrap();
    one_off_call.unwrap();
    assert_eq!(
        field(&seen, &["model"]),
        [json!("jev-preview"), json!("jev-preview")]
    );
    assert_eq!(global_default, MODEL);
}

// spec: judge_spec.rb:261 with a default judgment model > prefers a class model over the default and a call model over both
#[tokio::test]
async fn prefers_a_judge_model_over_the_default_and_a_call_model_over_both() {
    let (server, seen) = stub().await;
    let judge = undeclared_judge()
        .with_config(Arc::new(configured(&server, "jev-preview")))
        .model(MODEL);
    judge.judge("First").await.unwrap();
    judge
        .judge_with(
            "Second",
            JudgeOptions {
                model: Some(Some("jev-preview".into())),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    assert_eq!(
        field(&seen, &["model"]),
        [json!(MODEL), json!("jev-preview")]
    );
}

// spec: judge_spec.rb:270 with a default judgment model > uses the default when a call explicitly resets the model to nil
#[tokio::test]
async fn uses_the_default_when_a_call_explicitly_resets_the_model() {
    let (server, seen) = stub().await;
    let judge = undeclared_judge()
        .with_config(Arc::new(configured(&server, "jev-preview")))
        .model(MODEL);
    judge
        .judge_with(
            "Help",
            JudgeOptions {
                model: Some(None),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    assert_eq!(field(&seen, &["model"]), [json!("jev-preview")]);
}

// spec: judge_spec.rb:278 with a default judgment model > requires a model when the default is unset
#[tokio::test]
async fn requires_a_model_when_the_default_is_unset() {
    let (server, seen) = stub().await;
    let judge = undeclared_judge().with_config(Arc::new(configured(&server, "")));

    let err = judge.judge("Help").await.unwrap_err();
    assert!(matches!(err, Error::Argument(_)));
    assert!(err.to_string().contains("model"), "{err}");
    assert!(bodies(&seen).is_empty());
    let result = judge
        .judge_with(
            "Help",
            JudgeOptions {
                model: Some(Some(MODEL.into())),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(result.probability("urgent"), Some(0.9));
}

// ---- judgment_spec.rb -------------------------------------------------------------------------

// spec: judgment_spec.rb:21 serializes answers without discarding their uncertainty
#[test]
fn serializes_answers_without_discarding_their_uncertainty() {
    let probability = Answer::Probability { probability: 0.9 };
    let choice = Answer::Choice {
        choice: "Billing & payments".into(),
        probabilities: vec![("Billing & payments".into(), 1.0)],
        confidence: 1.0,
    };
    let result = Judgment::new(
        vec![
            ("urgent".into(), probability),
            ("department".into(), choice),
        ],
        MODEL,
        Tokens {
            input: Some(20),
            output: Some(10),
            ..Default::default()
        },
    );

    let h = result.to_value();
    assert_eq!(h["model"], json!(MODEL));
    assert_eq!(
        h["answers"],
        json!({
            "urgent": { "type": "probability", "probability": 0.9 },
            "department": { "type": "choice", "choice": "Billing & payments",
                            "probabilities": { "Billing & payments": 1.0 }, "confidence": 1.0 }
        })
    );
    assert_eq!(result.tokens().input, Some(20));
    assert_eq!(result.cost().total(), None);
}

// ---- judge/question_spec.rb -------------------------------------------------------------------

// spec: judge/question_spec.rb:23 rejects invalid question definitions
// Ruby's `{ type: :choice, criteria: { true => 'Boolean option name' } }` has no JSON form (object
// keys are always strings, and the string "true" is a valid option name in Ruby too); the other
// eight definitions are checked.
#[tokio::test]
async fn rejects_invalid_question_definitions() {
    let (server, seen) = stub().await;
    let judge = Judge::new().with_config(Arc::new(configured(&server, MODEL)));
    let definitions = [
        json!({ "type": "probability", "instructions": 42 }),
        json!({ "type": "probability", "criteria": { "maybe": "Uncertain" } }),
        json!({ "type": "probability", "criteria": { "yes": "Yes", "true": "Also yes" } }),
        json!({ "type": "choice", "options": {} }),
        json!({ "type": "choice", "options": { "": "Blank" } }),
        json!({ "type": "choice", "options": { "first": 42 } }),
        json!({ "type": "score", "levels": [] }),
        json!({ "type": "score", "levels": { "first": "First", "second": "Second" } }),
    ];

    for definition in definitions {
        let options = JudgeOptions {
            questions: questions(json!({ "question": definition })),
            ..Default::default()
        };
        let err = judge.judge_with("Help", options).await.unwrap_err();
        assert!(matches!(err, Error::Argument(_)), "{definition}: {err}");
    }
    assert!(bodies(&seen).is_empty());
}

// spec: judge/question_spec.rb:53 rejects unknown types and misspelled Hash fields
#[test]
fn rejects_unknown_types_and_misspelled_hash_fields() {
    for (definition, inspected) in [
        (json!({ "type": null }), "nil"),
        (json!({ "type": "text" }), ":text"),
        (json!({ "type": 42 }), "42"),
        (json!({}), "nil"),
    ] {
        let err = Question::from_value("question", &definition).unwrap_err();
        assert!(matches!(err, Error::Argument(_)));
        assert_eq!(
            err.to_string(),
            format!("Unknown judgment type: {inspected}")
        );
    }
    let err = Question::from_value(
        "question",
        &json!({ "type": "choice", "option": { "billing": null } }),
    )
    .unwrap_err();
    assert!(matches!(err, Error::Argument(_)));
    assert!(
        err.to_string().contains("Unknown question options: option"),
        "{err}"
    );
}
