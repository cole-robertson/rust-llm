//! RubyLLM 2.1 judgments: images through `with:`, OpenAI Decisions (`protocols/openai/decisions_spec.rb`,
//! `providers/openai_judgments_spec.rb`), local decision models through Ollama
//! (`providers/ollama_spec.rb`), and the relaxed score levels (`judge/question_spec.rb`). WebMock
//! stubs are wiremock servers; the live examples replay RubyLLM's cassettes.

mod support;

use std::sync::{Arc, Mutex};

use rust_llm::files::UploadedFile;
use rust_llm::judge::Question;
use rust_llm::judge::decisions::{parse_judgment_response, render_judgment_payload};
use rust_llm::model::ModelType;
use rust_llm::models::refresh::parse_ollama_models;
use rust_llm::{
    Answer, Attachment, Config, Error, ErrorKind, Judge, JudgeOptions, ProtocolName, Provider,
    Resolution,
};
use serde_json::{Map, Value, json};
use support::Cassette;
use wiremock::{Mock, MockServer, Request, ResponseTemplate, matchers};

/// `model_for(:openai, :judgment)`.
const OPENAI_MODEL: &str = "gpt-6-luna";

fn image_path() -> String {
    format!("{}/tests/fixtures/ruby.png", env!("CARGO_MANIFEST_DIR"))
}

fn data_uri() -> String {
    use base64::Engine;
    let bytes = std::fs::read(image_path()).unwrap();
    format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(bytes)
    )
}

fn question(name: &str, definition: Value) -> Question {
    Question::from_value(name, &definition).unwrap()
}

fn questions(v: Value) -> Map<String, Value> {
    v.as_object().unwrap().clone()
}

/// The spec's `questions`: a predicate, a choice with a String option name, and a score.
fn decision_questions() -> Vec<Question> {
    vec![
        question(
            "urgent",
            json!({ "type": "probability", "instructions": "Is this urgent?" }),
        ),
        question(
            "team",
            json!({ "type": "choice", "instructions": "Which team?",
                    "options": { "Billing & payments": null, "other": "Other" } }),
        ),
        question(
            "severity",
            json!({ "type": "score", "instructions": "How severe?",
                    "levels": ["Minor", ["Major", "Blocking"]] }),
        ),
    ]
}

/// `protocol.render_judgment_payload(input, questions:, model:, with:, provider_options:)`.
async fn render(input: Value, questions: &[Question], with: Vec<Attachment>) -> Value {
    render_judgment_payload(input, questions, OPENAI_MODEL, with, json!({}))
        .await
        .unwrap()
}

/// The spec's `body`: answers in question order, with cached input in the usage.
fn decision_body() -> Value {
    json!({
        "model": "gpt-6-luna-2026-09-22",
        "answers": [
            { "type": "predicate", "name": "urgent", "probability": 0.9 },
            { "type": "choice", "name": "team", "choice": "Billing & payments", "confidence": 0.8,
              "probabilities": [{ "value": "other", "probability": 0.1 },
                                { "value": "Billing & payments", "probability": 0.9 }] },
            { "type": "score", "name": "severity", "score": 0.2, "confidence": 0.7,
              "probabilities": [{ "value": 0, "label": "0", "probability": 0.8 },
                                { "value": 1, "label": "1", "probability": 0.2 }] }
        ],
        "usage": { "input_tokens": 120, "output_tokens": 3, "input_tokens_details": { "cached_tokens": 20 } }
    })
}

fn openai_model() -> rust_llm::Model {
    rust_llm::models()
        .find(OPENAI_MODEL, Some("openai"))
        .unwrap()
}

fn parse(body: Value) -> rust_llm::Result<rust_llm::Judgment> {
    parse_judgment_response(body, &decision_questions(), &openai_model())
}

// ---- judge/question_spec.rb ---------------------------------------------------------------------

// spec: judge/question_spec.rb:41 leaves the number of levels and their descriptions to the provider
#[tokio::test]
async fn leaves_the_number_of_levels_and_their_descriptions_to_the_provider() {
    for levels in [json!(["Only level"]), json!(["Calm", null])] {
        let severity = question("severity", json!({ "type": "score", "levels": levels }));
        let payload = render(json!("Help"), &[severity], vec![]).await;
        let rendered: Vec<Value> = payload["questions"][0]["levels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|l| l["description"].clone())
            .collect();
        assert_eq!(Value::Array(rendered), levels);
    }
}

// ---- judge_spec.rb ------------------------------------------------------------------------------

type Seen = Arc<Mutex<Vec<Value>>>;

/// `stub_request(:post, 'https://api.typesafe.ai/v1/systemone')`, keeping each body.
async fn typesafe_stub() -> (MockServer, Seen) {
    let server = MockServer::start().await;
    let seen: Seen = Arc::default();
    let sink = seen.clone();
    Mock::given(matchers::method("POST"))
        .and(matchers::path("/v1/systemone"))
        .respond_with(move |req: &Request| {
            sink.lock()
                .unwrap()
                .push(serde_json::from_slice(&req.body).unwrap());
            ResponseTemplate::new(200).set_body_json(json!({
                "model": "jev-latest",
                "answers": { "urgent": { "type": "noul", "noul": 0.9 } },
                "usage": { "input_tokens": 100, "output_tokens": 10 }
            }))
        })
        .mount(&server)
        .await;
    (server, seen)
}

fn judge_class(config: Config) -> Judge {
    Judge::new()
        .with_config(Arc::new(config))
        .model("jev-latest")
        .provider("typesafe")
        .assume_model_exists()
        .probability("urgent", "Does this need attention today?")
        .unwrap()
}

fn typesafe_config(server: &MockServer) -> Config {
    let mut c = Config::default();
    c.set("typesafe_api_base", server.uri());
    c.set("typesafe_api_key", "test-key");
    c.max_retries = 0;
    c
}

// spec: judge_spec.rb:166 accepts images with or without input and rejects them on text-only protocols before sending
#[tokio::test]
async fn accepts_images_with_or_without_input_and_rejects_them_on_text_only_protocols() {
    let (server, seen) = typesafe_stub().await;
    let judge = judge_class(typesafe_config(&server));
    let image = || vec![Attachment::new("https://example.com/receipt.png")];

    let with_text = judge
        .judge_with(
            "Help",
            JudgeOptions {
                with: image(),
                ..Default::default()
            },
        )
        .await
        .unwrap_err();
    assert!(
        matches!(with_text, Error::UnsupportedAttachment(_)),
        "{with_text}"
    );
    assert!(with_text.to_string().contains("image/png"), "{with_text}");

    let image_only = judge
        .judge_with(
            Value::Null,
            JudgeOptions {
                with: image(),
                ..Default::default()
            },
        )
        .await
        .unwrap_err();
    assert!(
        matches!(image_only, Error::UnsupportedAttachment(_)),
        "{image_only}"
    );

    let nothing = judge
        .judge_with(Value::Null, JudgeOptions::default())
        .await
        .unwrap_err();
    assert!(matches!(nothing, Error::Argument(_)));
    assert!(nothing.to_string().contains("Judgment input"), "{nothing}");
    assert!(seen.lock().unwrap().is_empty());
}

// spec: judge_spec.rb:293 lets OpenAI reject chat models through Decisions
#[tokio::test]
async fn lets_openai_reject_chat_models_through_decisions() {
    let (typesafe, _) = typesafe_stub().await;
    let openai = MockServer::start().await;
    let model = "gpt-5-nano";
    Mock::given(matchers::method("POST"))
        .and(matchers::path("/v1/decisions"))
        .respond_with(ResponseTemplate::new(404).set_body_json(json!({
            "error": { "message": format!("The model `{model}` does not exist or you do not have access to it.") }
        })))
        .expect(1)
        .mount(&openai)
        .await;
    let mut config = typesafe_config(&typesafe);
    config.set("openai_api_base", format!("{}/v1", openai.uri()));
    config.set("openai_api_key", "test-key");

    let err = judge_class(config)
        .judge_with(
            "Help",
            JudgeOptions {
                model: Some(Some(model.into())),
                provider: Some("openai".into()),
                ..Default::default()
            },
        )
        .await
        .unwrap_err();
    assert!(err.to_string().contains("does not exist"), "{err}");
}

// ---- protocols/openai/decisions_spec.rb ---------------------------------------------------------

// spec: protocols/openai/decisions_spec.rb:40 renders ordered questions with the Decisions vocabulary
#[tokio::test]
async fn renders_ordered_questions_with_the_decisions_vocabulary() {
    let payload = render(json!("Help"), &decision_questions(), vec![]).await;

    assert_eq!(payload["model"], "gpt-6-luna");
    assert_eq!(payload["input"], "Help");
    assert_eq!(
        payload["questions"],
        json!([
            { "type": "predicate", "name": "urgent", "instructions": "Is this urgent?" },
            { "type": "choice", "name": "team", "instructions": "Which team?",
              "choices": [{ "value": "Billing & payments" }, { "value": "other", "description": "Other" }] },
            { "type": "score", "name": "severity", "instructions": "How severe?",
              "levels": [{ "label": "0", "description": "Minor" },
                         { "label": "1", "description": "[\"Major\",\"Blocking\"]" }] }
        ])
    );
}

// spec: protocols/openai/decisions_spec.rb:56 sends structured input and descriptions as JSON text
#[tokio::test]
async fn sends_structured_input_and_descriptions_as_json_text() {
    let team = question(
        "team",
        json!({ "type": "choice", "instructions": { "ask": "Which team?" },
                "options": { "billing": { "handles": ["Refunds"] }, "other": null } }),
    );

    let payload = render(json!({ "message": "Help" }), &[team], vec![]).await;

    assert_eq!(payload["input"], "{\"message\":\"Help\"}");
    assert_eq!(
        payload["questions"][0]["instructions"],
        "{\"ask\":\"Which team?\"}"
    );
    assert_eq!(
        payload["questions"][0]["choices"],
        json!([{ "value": "billing", "description": "{\"handles\":[\"Refunds\"]}" }, { "value": "other" }])
    );
}

// spec: protocols/openai/decisions_spec.rb:69 adds yes and no descriptions to the predicate instructions
#[tokio::test]
async fn adds_yes_and_no_descriptions_to_the_predicate_instructions() {
    let explicit = question(
        "urgent",
        json!({ "type": "probability", "instructions": "Is this urgent?",
                "criteria": { "yes": "A deadline today", "false": null } }),
    );
    let implicit = question(
        "urgent",
        json!({ "type": "probability", "criteria": { "no": "No deadline" } }),
    );

    let rendered = render(json!("Help"), &[explicit], vec![]).await;
    assert_eq!(
        rendered["questions"][0]["instructions"],
        "Is this urgent?\nYes: A deadline today"
    );
    let rendered = render(json!("Help"), &[implicit], vec![]).await;
    assert_eq!(rendered["questions"][0]["instructions"], "No: No deadline");
}

// spec: protocols/openai/decisions_spec.rb:79 sends images inline in a message with the input text
#[tokio::test]
async fn sends_images_inline_in_a_message_with_the_input_text() {
    let qs = decision_questions();
    let image = || vec![Attachment::new(image_path())];
    let content = |payload: Value| payload["input"][0]["content"].clone();

    assert_eq!(
        render(json!("Classify this page"), &qs, image()).await["input"],
        json!([{ "type": "message", "role": "user", "content": [
            { "type": "input_text", "text": "Classify this page" },
            { "type": "input_image", "image_url": data_uri() }
        ] }])
    );
    let bare = json!([{ "type": "input_image", "image_url": data_uri() }]);
    assert_eq!(content(render(Value::Null, &qs, image()).await), bare);
    assert_eq!(content(render(json!(""), &qs, image()).await), bare);
    let low = vec![Attachment::new(image_path()).with_resolution(Resolution::Low)];
    assert_eq!(
        content(render(Value::Null, &qs, low).await),
        json!([{ "type": "input_image", "image_url": data_uri(), "detail": "low" }])
    );
    let original = vec![Attachment::new(image_path()).with_resolution(Resolution::Original)];
    assert_eq!(
        content(render(Value::Null, &qs, original).await),
        json!([{ "type": "input_image", "image_url": data_uri(), "detail": "original" }])
    );
}

// spec: protocols/openai/decisions_spec.rb:96 downloads image URLs to send them inline
#[tokio::test]
async fn downloads_image_urls_to_send_them_inline() {
    let server = MockServer::start().await;
    Mock::given(matchers::method("GET"))
        .and(matchers::path("/receipt.png"))
        .respond_with(ResponseTemplate::new(200).set_body_raw("png", "image/png"))
        .expect(1)
        .mount(&server)
        .await;
    let url = format!("{}/receipt.png", server.uri());

    let payload = render(
        Value::Null,
        &decision_questions(),
        vec![Attachment::new(url)],
    )
    .await;

    assert_eq!(
        payload["input"][0]["content"],
        json!([{ "type": "input_image", "image_url": "data:image/png;base64,cG5n" }])
    );
}

// spec: protocols/openai/decisions_spec.rb:104 sends uploaded images by file ID and rejects attachments Decisions cannot express
#[tokio::test]
async fn sends_uploaded_images_by_file_id_and_rejects_other_attachments() {
    let uploaded = Attachment::from_uploaded(UploadedFile {
        id: "file-123".into(),
        provider: "openai".into(),
        filename: Some("receipt.png".into()),
        byte_size: None,
        created_at: None,
        expires_at: None,
        status: None,
        mime_type: Some("image/png".into()),
        purpose: None,
        uri: None,
        downloadable: None,
        metadata: Value::Null,
    });
    let qs = decision_questions();

    let payload = render(Value::Null, &qs, vec![uploaded]).await;
    assert_eq!(
        payload["input"][0]["content"],
        json!([{ "type": "input_image", "file_id": "file-123" }])
    );

    let document = Attachment::new("https://example.com/contract.pdf");
    let err = render_judgment_payload(json!("Help"), &qs, OPENAI_MODEL, vec![document], json!({}))
        .await
        .unwrap_err();
    assert!(matches!(err, Error::UnsupportedAttachment(_)), "{err}");
    assert!(err.to_string().contains("application/pdf"), "{err}");
}

// spec: protocols/openai/decisions_spec.rb:114 leaves question limits and missing instructions to the API
#[tokio::test]
async fn leaves_question_limits_and_missing_instructions_to_the_api() {
    let many: Vec<Question> = (0..65)
        .map(|n| {
            question(
                &n.to_string(),
                json!({ "type": "probability", "instructions": "Is this urgent?" }),
            )
        })
        .collect();
    let single = question(
        "team",
        json!({ "type": "choice", "instructions": "Which team?", "options": { "a": null } }),
    );
    let bare = question("urgent", json!({ "type": "probability" }));

    let rendered = render(json!("Help"), &many, vec![]).await;
    assert_eq!(rendered["questions"].as_array().unwrap().len(), 65);
    let rendered = render(json!("Help"), &[single], vec![]).await;
    assert_eq!(
        rendered["questions"][0]["choices"],
        json!([{ "value": "a" }])
    );
    let rendered = render(json!("Help"), &[bare], vec![]).await;
    assert_eq!(
        rendered["questions"],
        json!([{ "type": "predicate", "name": "urgent" }])
    );
}

// spec: protocols/openai/decisions_spec.rb:126 prevents provider options from replacing the questions, model, or input behind the parser
#[tokio::test]
async fn prevents_provider_options_from_replacing_reserved_fields() {
    let qs = decision_questions();
    for key in ["questions", "input", "model"] {
        let err =
            render_judgment_payload(json!("Help"), &qs, OPENAI_MODEL, vec![], json!({ key: {} }))
                .await
                .unwrap_err();
        assert!(matches!(err, Error::Argument(_)), "{err}");
        assert!(err.to_string().contains(key), "{err}");
    }
    let payload = render_judgment_payload(
        json!("Help"),
        &qs,
        OPENAI_MODEL,
        vec![],
        json!({ "service_tier": "flex" }),
    )
    .await
    .unwrap();
    assert_eq!(payload["service_tier"], "flex");
}

// spec: protocols/openai/decisions_spec.rb:133 parses answers in question order into declared names and option types
#[test]
fn parses_answers_in_question_order_into_declared_names_and_option_types() {
    let judgment = parse(decision_body()).unwrap();

    let names: Vec<&str> = judgment.answers.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, ["urgent", "team", "severity"]);
    assert_eq!(judgment.probability("urgent"), Some(0.9));
    assert_eq!(judgment.choice("team"), Some("Billing & payments"));
    let Some(Answer::Choice { probabilities, .. }) = judgment.get("team") else {
        panic!("team is a choice");
    };
    // Ruby compares Hashes without order; the entries keep the order the API listed them.
    assert_eq!(
        probabilities,
        &vec![
            ("other".to_string(), 0.1),
            ("Billing & payments".to_string(), 0.9)
        ]
    );
    assert_eq!(judgment.score("severity"), Some(0.2));
    let Some(Answer::Score {
        levels,
        probabilities,
        ..
    }) = judgment.get("severity")
    else {
        panic!("severity is a score");
    };
    assert_eq!(levels, &vec![json!("Minor"), json!(["Major", "Blocking"])]);
    assert_eq!(probabilities, &vec![(0, 0.8), (1, 0.2)]);
    assert_eq!(judgment.model, "gpt-6-luna-2026-09-22");
    let tokens = judgment.tokens();
    assert_eq!(
        [tokens.input, tokens.output, tokens.cache_read],
        [Some(100), Some(3), Some(20)]
    );
}

// spec: protocols/openai/decisions_spec.rb:147 records reasoning tokens and tolerates missing usage
#[test]
fn records_reasoning_tokens_and_tolerates_missing_usage() {
    let mut reasoning = decision_body();
    reasoning["usage"]["output_tokens_details"] = json!({ "reasoning_tokens": 2 });
    let mut without_usage = decision_body();
    without_usage.as_object_mut().unwrap().remove("usage");

    assert_eq!(parse(reasoning).unwrap().tokens().thinking, Some(2));
    assert_eq!(parse(without_usage).unwrap().tokens().input, None);
}

// spec: protocols/openai/decisions_spec.rb:155 turns a malformed body into a RubyLLM error
#[test]
fn turns_a_malformed_body_into_a_ruby_llm_error() {
    let body = decision_body();
    let without = |key: &str| {
        let mut b = body.clone();
        b.as_object_mut().unwrap().remove(key);
        b
    };
    let mut truncated = body.clone();
    truncated["answers"].as_array_mut().unwrap().truncate(2);
    let mut unknown_choice = body.clone();
    unknown_choice["answers"][1]["choice"] = json!("sales");

    for invalid in [
        json!("Internal error"),
        without("answers"),
        without("model"),
        truncated,
        unknown_choice,
    ] {
        let err = parse(invalid.clone()).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Api, "{invalid}");
        assert!(
            err.to_string()
                .contains("OpenAI Decisions returned an invalid judgment"),
            "{err}"
        );
    }
}

// ---- providers/openai_judgments_spec.rb ---------------------------------------------------------

fn openai_config(server: &MockServer) -> Config {
    let mut c = Config::default();
    c.set("openai_api_base", format!("{}/v1", server.uri()));
    c.set("openai_api_key", "test-key");
    c.max_retries = 0;
    c
}

fn urgent() -> Map<String, Value> {
    questions(
        json!({ "urgent": { "type": "probability", "instructions": "Does this need attention today?" } }),
    )
}

async fn openai_judge(config: Config, input: &str) -> rust_llm::Result<rust_llm::Judgment> {
    rust_llm::judge(
        input,
        Value::Object(urgent()),
        JudgeOptions {
            model: Some(Some(OPENAI_MODEL.into())),
            provider: Some("openai".into()),
            config: Some(Arc::new(config)),
            ..Default::default()
        },
    )
    .await
}

// spec: providers/openai_judgments_spec.rb:12 routes judgments to Decisions even when chat uses a configured protocol
#[tokio::test]
async fn routes_judgments_to_decisions_even_when_chat_uses_a_configured_protocol() {
    let server = MockServer::start().await;
    Mock::given(matchers::method("POST"))
        .and(matchers::path("/v1/decisions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model": OPENAI_MODEL,
            "answers": [{ "type": "predicate", "name": "urgent", "probability": 0.2 }],
            "usage": { "input_tokens": 10, "output_tokens": 1 }
        })))
        .expect(1)
        .mount(&server)
        .await;
    let mut config = openai_config(&server);
    config.set("openai_protocol", "chat_completions");

    let result = openai_judge(config, "Help").await.unwrap();
    assert_eq!(result.probability("urgent"), Some(0.2));
}

// spec: providers/openai_judgments_spec.rb:24 judges through the OpenAI connection and prices usage from the model registry
#[tokio::test]
async fn judges_through_the_openai_connection_and_prices_usage_from_the_registry() {
    let server = MockServer::start().await;
    Mock::given(matchers::method("POST"))
        .and(matchers::path("/v1/decisions"))
        .and(matchers::header("authorization", "Bearer test-key"))
        .and(matchers::body_json(json!({
            "model": OPENAI_MODEL, "input": "Please help today.",
            "questions": [{ "type": "predicate", "name": "urgent", "instructions": "Does this need attention today?" }]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model": OPENAI_MODEL,
            "answers": [{ "type": "predicate", "name": "urgent", "probability": 0.9 }],
            "usage": { "input_tokens": 40, "output_tokens": 1 }
        })))
        .expect(1)
        .mount(&server)
        .await;

    let result = openai_judge(openai_config(&server), "Please help today.")
        .await
        .unwrap();

    assert_eq!(result.probability("urgent"), Some(0.9));
    assert_eq!(result.tokens().input, Some(40));
    let expected = openai_model().cost_for(&result.tokens()).total();
    assert!(expected.is_some(), "the registry prices {OPENAI_MODEL}");
    assert_eq!(result.cost().total(), expected);
}

// spec: providers/openai_judgments_spec.rb:44 reports normalized OpenAI errors
#[tokio::test]
async fn reports_normalized_openai_errors() {
    let server = MockServer::start().await;
    Mock::given(matchers::method("POST"))
        .and(matchers::path("/v1/decisions"))
        .respond_with(
            ResponseTemplate::new(401)
                .set_body_json(json!({ "error": { "message": "Incorrect API key provided" } })),
        )
        .mount(&server)
        .await;

    let err = openai_judge(openai_config(&server), "Help")
        .await
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Unauthorized);
    assert_eq!(err.to_string(), "Incorrect API key provided");
}

fn cassette_config(cassette: &Cassette) -> Arc<Config> {
    support::config_for(cassette, "openai")
}

// spec: providers/openai_judgments_spec.rb:56 with the Decisions API judges all three question types through the compact DSL
#[tokio::test]
async fn decisions_judges_all_three_question_types_through_the_compact_dsl() {
    let cassette = Cassette::start(
        "providers_openai_with_the_decisions_api_judges_all_three_question_types_through_the_compact_dsl",
    )
    .await
    .unwrap();
    let triage = Judge::new()
        .with_config(cassette_config(&cassette))
        .model(OPENAI_MODEL)
        .provider("openai")
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
    cassette.assert_all_matched().await;

    assert!((0.0..=1.0).contains(&result.probability("urgent").unwrap()));
    assert!(["billing", "technical", "other"].contains(&result.choice("department").unwrap()));
    let Some(Answer::Choice { probabilities, .. }) = result.get("department") else {
        panic!("department is a choice");
    };
    let keys: Vec<&str> = probabilities.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(keys, ["billing", "technical", "other"]);
    assert!((0.0..=2.0).contains(&result.score("frustration").unwrap()));
    let Some(Answer::Score { probabilities, .. }) = result.get("frustration") else {
        panic!("frustration is a score");
    };
    let levels: Vec<usize> = probabilities.iter().map(|(k, _)| *k).collect();
    assert_eq!(levels, [0, 1, 2]);
    assert!(result.tokens().input.unwrap() > 0);
    let raw = result.raw.as_ref().unwrap();
    assert_eq!(
        result.tokens().output,
        raw.body["usage"]["output_tokens"].as_i64()
    );
    assert!(result.model.starts_with("gpt-6-luna"));
}

// spec: providers/openai_judgments_spec.rb:87 with the Decisions API judges an image without text input
#[tokio::test]
async fn decisions_judges_an_image_without_text_input() {
    let cassette = Cassette::start(
        "providers_openai_with_the_decisions_api_judges_an_image_without_text_input",
    )
    .await
    .unwrap();

    let result = rust_llm::judge(
        Value::Null,
        json!({
            "logo": { "type": "probability", "instructions": "Is this image a logo?" },
            "color": { "type": "choice", "instructions": "What is the dominant color?",
                       "options": { "red": null, "blue": null, "green": null } }
        }),
        JudgeOptions {
            model: Some(Some(OPENAI_MODEL.into())),
            provider: Some("openai".into()),
            config: Some(cassette_config(&cassette)),
            with: vec![Attachment::new(image_path())],
            ..Default::default()
        },
    )
    .await
    .unwrap();
    cassette.assert_all_matched().await;

    assert!((0.0..=1.0).contains(&result.probability("logo").unwrap()));
    assert!(["red", "blue", "green"].contains(&result.choice("color").unwrap()));
    assert!(result.tokens().input.unwrap() > 0);
}

// ---- providers/ollama_spec.rb -------------------------------------------------------------------

// spec: providers/ollama_spec.rb:61 model listing reads decision models as judgment models
#[test]
fn ollama_reads_decision_models_as_judgment_models() {
    let details = [(
        "clef-flash:latest".to_string(),
        vec!["decision".to_string()],
    )]
    .into_iter()
    .collect();
    let model = &parse_ollama_models(
        &json!({ "data": [{ "id": "clef-flash:latest" }] }),
        "ollama",
        &details,
        false,
    )[0];

    assert_eq!(model.model_type(), ModelType::Judgment);
    assert_eq!(model.capabilities, ["judgment"]);
}

/// An Ollama server answering `POST /v1/systemone` (System One under the OpenAI-compatible
/// base), keeping each body.
async fn ollama_stub() -> (MockServer, Seen, Judge) {
    let server = MockServer::start().await;
    let seen: Seen = Arc::default();
    let sink = seen.clone();
    Mock::given(matchers::method("POST"))
        .and(matchers::path("/v1/systemone"))
        .respond_with(move |req: &Request| {
            sink.lock()
                .unwrap()
                .push(serde_json::from_slice(&req.body).unwrap());
            ResponseTemplate::new(200).set_body_json(json!({
                "model": "clef-flash",
                "answers": { "urgent": { "type": "noul", "noul": 0.7 } },
                "usage": { "input_tokens": 12, "output_tokens": 1 }
            }))
        })
        .mount(&server)
        .await;
    let mut config = Config::default();
    config.set("ollama_api_base", format!("{}/v1", server.uri()));
    config.max_retries = 0;
    let judge = Judge::new()
        .with_config(Arc::new(config))
        .model("clef-flash")
        .provider("ollama")
        .probability("urgent", "Urgent?")
        .unwrap();
    (server, seen, judge)
}

// spec: providers/ollama_spec.rb:101 judgments routes judgments through System One and chats through Chat Completions
#[tokio::test]
async fn ollama_routes_judgments_through_system_one_and_chats_through_chat_completions() {
    let (_server, seen, judge) = ollama_stub().await;

    let result = judge.judge("Help").await.unwrap();
    assert_eq!(result.probability("urgent"), Some(0.7));
    assert_eq!(
        seen.lock().unwrap()[0]["questions"]["urgent"]["type"],
        "noul"
    );

    let model = rust_llm::Model::default_for("clef-flash", "ollama");
    let mut config = Config::default();
    config.set("ollama_api_base", "http://localhost:11434/v1");
    assert_eq!(
        Provider::Ollama
            .resolve_protocol(None, &model, &config)
            .unwrap(),
        ProtocolName::ChatCompletions
    );
}

// spec: providers/ollama_spec.rb:106 judgments posts to systemone under the OpenAI-compatible base
#[tokio::test]
async fn ollama_posts_to_systemone_under_the_openai_compatible_base() {
    let (server, _seen, judge) = ollama_stub().await;

    judge.judge("Help").await.unwrap();

    let requests = server.received_requests().await.unwrap();
    let paths: Vec<&str> = requests.iter().map(|r| r.url.path()).collect();
    assert_eq!(paths, ["/v1/systemone"]);
}

// spec: providers/ollama_spec.rb:110 judgments sends images as base64 alongside the state
#[tokio::test]
async fn ollama_sends_images_as_base64_alongside_the_state() {
    let (_server, seen, judge) = ollama_stub().await;
    let image = Attachment::new(image_path());
    let encoded = data_uri()
        .trim_start_matches("data:image/png;base64,")
        .to_string();

    judge
        .judge_with(
            "Look",
            JudgeOptions {
                with: vec![image],
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let payload = seen.lock().unwrap()[0].clone();
    assert_eq!(payload["state"], "Look");
    assert_eq!(payload["images"], json!([encoded]));
    assert_eq!(
        payload["questions"]["urgent"],
        json!({ "type": "noul", "instructions": "Urgent?" })
    );
}

// spec: providers/ollama_spec.rb:119 judgments leaves images out when there are none
#[tokio::test]
async fn ollama_leaves_images_out_when_there_are_none() {
    let (_server, seen, judge) = ollama_stub().await;

    judge.judge("Help").await.unwrap();

    assert!(seen.lock().unwrap()[0].get("images").is_none());
}

// spec: providers/ollama_spec.rb:125 judgments rejects attachments that are not images
#[tokio::test]
async fn ollama_rejects_attachments_that_are_not_images() {
    let (_server, seen, judge) = ollama_stub().await;
    let text = Attachment::from_bytes(b"notes".to_vec(), "notes.txt", None);

    let err = judge
        .judge_with(
            "Help",
            JudgeOptions {
                with: vec![text],
                ..Default::default()
            },
        )
        .await
        .unwrap_err();

    assert!(matches!(err, Error::UnsupportedAttachment(_)), "{err}");
    assert!(seen.lock().unwrap().is_empty());
}
