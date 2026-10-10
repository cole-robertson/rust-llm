//! RubyLLM 2.1's `evaluation_defaults_spec.rb`, `evaluation/evaluator_spec.rb`, and
//! `evaluation_accounting_spec.rb`: evaluations graded by a reviewer model or a decision model,
//! and the usage each trial counts. Ruby stubs the provider endpoints with WebMock; here a
//! wiremock server answers per path, so the port's real request rendering runs in between.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use rust_llm::evaluation::{
    Case, Evaluation, Evaluator, Failure, Measurement, Outcome, RunOptions, Status,
};
use rust_llm::{Agent, Answer, Config, EmbedOptions, Judge, ProtocolName, SharedTool};
use serde_json::{Map, Value, json};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

const TASK_MODEL: &str = "claude-haiku-4-5";
const REVIEW_MODEL: &str = "gpt-5-nano";
const EMBEDDING_MODEL: &str = "text-embedding-3-small";
const JUDGMENT_MODEL: &str = "jev-latest";

type Handler = Box<dyn Fn(&Value) -> ResponseTemplate + Send + Sync>;
/// A path, the responses queued for it, and the handler for every request after them.
type Route = (String, VecDeque<Handler>, Option<Handler>);

/// Routes each request by path to a handler, logging bodies per path.
#[derive(Default)]
struct Routes {
    handlers: Mutex<Vec<Route>>,
    log: Mutex<Vec<(String, Value)>>,
}

impl Routes {
    /// Every request to `path` answered by `handler`.
    fn on(&self, path: &str, handler: impl Fn(&Value) -> ResponseTemplate + Send + Sync + 'static) {
        let mut h = self.handlers.lock().unwrap();
        h.retain(|(p, _, _)| p != path);
        h.push((path.into(), VecDeque::new(), Some(Box::new(handler))));
    }

    /// The next requests to `path` answered in order, then `fallback`.
    fn sequence(&self, path: &str, responses: Vec<Value>) {
        let mut h = self.handlers.lock().unwrap();
        h.retain(|(p, _, _)| p != path);
        let queue: VecDeque<Handler> = responses
            .into_iter()
            .map(|r| Box::new(move |_: &Value| json_response(&r)) as Handler)
            .collect();
        h.push((path.into(), queue, None));
    }

    fn bodies(&self, path: &str) -> Vec<Value> {
        self.log
            .lock()
            .unwrap()
            .iter()
            .filter(|(p, _)| p == path)
            .map(|(_, b)| b.clone())
            .collect()
    }
}

struct Server(Arc<Routes>);

impl Respond for Server {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let path = request.url.path().to_string();
        let body: Value = serde_json::from_slice(&request.body).unwrap_or(Value::Null);
        self.0
            .log
            .lock()
            .unwrap()
            .push((path.clone(), body.clone()));
        let mut handlers = self.0.handlers.lock().unwrap();
        match handlers.iter_mut().find(|(p, _, _)| *p == path) {
            Some((_, queue, fallback)) => match queue.pop_front() {
                Some(h) => h(&body),
                None => fallback
                    .as_ref()
                    .map_or_else(|| ResponseTemplate::new(599), |h| h(&body)),
            },
            None => ResponseTemplate::new(404).set_body_string(format!("no stub for {path}")),
        }
    }
}

fn json_response(body: &Value) -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(body)
}

const RESPONSES: &str = "/v1/responses";
const COMPLETIONS: &str = "/v1/chat/completions";
const MESSAGES: &str = "/v1/messages";
const EMBEDDINGS: &str = "/v1/embeddings";
const SYSTEMONE: &str = "/v1/systemone";

struct Stub {
    _server: MockServer,
    routes: Arc<Routes>,
    config: Arc<Config>,
}

async fn stub(default_model: &str) -> Stub {
    let server = MockServer::start().await;
    let routes = Arc::new(Routes::default());
    Mock::given(wiremock::matchers::any())
        .respond_with(Server(routes.clone()))
        .mount(&server)
        .await;
    let mut c = Config::default();
    for (provider, path) in [("anthropic", ""), ("openai", "/v1"), ("typesafe", "")] {
        c.set(
            format!("{provider}_api_base"),
            format!("{}{path}", server.uri()),
        );
        c.set(format!("{provider}_api_key"), "test");
    }
    c.max_retries = 0;
    c.default_model = default_model.into();
    Stub {
        _server: server,
        routes,
        config: Arc::new(c),
    }
}

/// A Responses reply passing every criterion the request's schema requires.
fn responses_verdicts(body: &Value) -> ResponseTemplate {
    let verdicts: Map<String, Value> = body
        .pointer("/text/format/schema/required")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(|n| {
            (
                n.to_string(),
                json!({ "verdict": "pass", "reason": "The answer satisfies the criterion." }),
            )
        })
        .collect();
    json_response(&json!({
        "id": "resp_evaluation", "model": REVIEW_MODEL, "status": "completed",
        "output": [{ "type": "message", "role": "assistant",
                     "content": [{ "type": "output_text", "text": Value::Object(verdicts).to_string(), "annotations": [] }] }],
        "usage": { "input_tokens": 100, "output_tokens": 20 }
    }))
}

/// A System One reply answering every question with `noul`.
fn systemone(noul: f64, input_tokens: i64) -> impl Fn(&Value) -> ResponseTemplate {
    move |body: &Value| {
        let answers: Map<String, Value> = body["questions"]
            .as_object()
            .into_iter()
            .flatten()
            .map(|(k, _)| (k.clone(), json!({ "type": "noul", "noul": noul })))
            .collect();
        json_response(&json!({
            "model": JUDGMENT_MODEL, "answers": answers,
            "usage": { "input_tokens": input_tokens, "output_tokens": 1 }
        }))
    }
}

fn required(body: &Value, pointer: &str) -> Vec<String> {
    body.pointer(pointer)
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Value::as_str)
                .map(String::from)
                .collect()
        })
        .unwrap_or_default()
}

fn hello() -> Vec<Case> {
    vec![
        Case::new("greeting", "Hello")
            .unwrap()
            .with_expected_output("HELLO"),
    ]
}

/// The defaults spec's `evaluation`: `perform` returns its input.
fn identity(config: &Arc<Config>) -> Evaluation {
    let mut e = Evaluation::new();
    e.with_config(config.clone());
    e.perform(|i| {
        let input = i.input().clone();
        async move { Ok(Outcome::Value(input)) }
    });
    e
}

fn argument_error<T: std::fmt::Debug>(result: rust_llm::Result<T>, pattern: &str) {
    match result {
        Err(rust_llm::Error::Argument(m)) => assert!(
            regex::Regex::new(pattern).unwrap().is_match(&m),
            "{m:?} does not match {pattern:?}"
        ),
        other => panic!("expected an ArgumentError matching {pattern:?}, got {other:?}"),
    }
}

// ---- evaluation_defaults_spec.rb --------------------------------------------------------------

async fn defaults() -> Stub {
    let s = stub(REVIEW_MODEL).await;
    s.routes.on(RESPONSES, responses_verdicts);
    s
}

fn criteria(s: &Stub) -> Vec<Vec<String>> {
    s.routes
        .bodies(RESPONSES)
        .iter()
        .map(|b| required(b, "/text/format/schema/required"))
        .collect()
}

fn run(dataset: Vec<Case>) -> RunOptions {
    RunOptions::default().dataset(dataset)
}

// spec: evaluation_defaults_spec.rb:40
#[tokio::test]
async fn grades_a_perform_only_evaluation_against_the_reference_using_the_configured_default_model()
{
    let s = defaults().await;
    let report = identity(&s.config).run_with(run(hello())).await.unwrap();
    assert!(report.is_passed(), "{report}");
    assert_eq!(report.first().unwrap().assertion_count(), 0);
    assert_eq!(criteria(&s), [["correctness"]]);
    assert_eq!(report.definitions().len(), 1);
    assert_eq!(report.definitions()[0]["name"], "correctness");
    assert_eq!(
        report.definitions()[0]["instructions"],
        "The answer agrees with the expected output"
    );
    let first = &s.routes.bodies(RESPONSES)[0];
    let input = first["input"].as_array().unwrap();
    let evidence: Value =
        serde_json::from_str(input.last().unwrap()["content"].as_str().unwrap()).unwrap();
    assert_eq!(evidence["inputs"], "Hello");
    assert_eq!(evidence["actual"], "Hello");
    assert_eq!(evidence["expected_output"], "HELLO");
    assert_eq!(report.tokens().input, Some(100));
    assert_eq!(
        report.cost().total(),
        report.first().unwrap().evaluator_cost().total()
    );
}

// spec: evaluation_defaults_spec.rb:55
#[tokio::test]
async fn keeps_default_correctness_when_ruby_assertions_are_added() {
    let s = defaults().await;
    let mut e = identity(&s.config);
    e.assertions(|a| Ok(a.assert_equal("Goodbye", a.output().clone())?));
    let report = e.run_with(run(hello())).await.unwrap();
    let first = report.first().unwrap();
    assert_eq!(first.status(), Status::Failed);
    assert!(first.assertion_failure().is_some());
    assert!(first.evaluations()[0].is_passed());
    assert_eq!(criteria(&s), [["correctness"]]);
}

// spec: evaluation_defaults_spec.rb:65
#[tokio::test]
async fn uses_explicitly_declared_criteria_as_the_complete_set_without_requiring_references() {
    let s = defaults().await;
    let mut e = identity(&s.config);
    e.evaluation("grounded", Some("Every claim has a citation"))
        .unwrap();
    let report = e
        .run_with(run(vec![Case::new("citation", "See [1]").unwrap()]))
        .await
        .unwrap();
    assert!(report.is_passed());
    assert_eq!(criteria(&s), [["grounded"]]);
    assert_eq!(report.definitions().len(), 1);
    assert_eq!(report.definitions()[0]["name"], "grounded");
    assert_eq!(
        report.definitions()[0]["instructions"],
        "Every claim has a citation"
    );
}

// spec: evaluation_defaults_spec.rb:74
#[tokio::test]
async fn allows_custom_correctness_instructions_without_adding_the_built_in_definition() {
    let s = defaults().await;
    let mut e = identity(&s.config);
    e.evaluation(
        "correctness",
        Some("The response contains the order number"),
    )
    .unwrap();
    let report = e
        .run_with(run(vec![Case::new("order", "Order 42").unwrap()]))
        .await
        .unwrap();
    assert!(report.is_passed());
    assert_eq!(criteria(&s), [["correctness"]]);
    assert_eq!(report.definitions().len(), 1);
    assert_eq!(
        report.definitions()[0]["instructions"],
        "The response contains the order number"
    );
}

// spec: evaluation_defaults_spec.rb:83
#[tokio::test]
async fn includes_built_in_correctness_explicitly_alongside_another_criterion_in_one_request() {
    let s = defaults().await;
    let mut e = identity(&s.config);
    e.evaluation("correctness", None).unwrap();
    e.evaluation("concise", Some("The answer contains no unnecessary detail"))
        .unwrap();
    let report = e.run_with(run(hello())).await.unwrap();
    assert!(report.is_passed());
    assert_eq!(criteria(&s), [["correctness", "concise"]]);
    let names: Vec<&str> = report
        .first()
        .unwrap()
        .evaluations()
        .iter()
        .map(|r| r.name())
        .collect();
    assert_eq!(names, ["correctness", "concise"]);
    assert_eq!(report.tokens().input, Some(100));
}

// spec: evaluation_defaults_spec.rb:94
#[tokio::test]
async fn validates_all_selected_references_before_starting_any_case() {
    let s = defaults().await;
    let performed = Arc::new(AtomicUsize::new(0));
    let mut e = identity(&s.config);
    let p = performed.clone();
    e.perform(move |i| {
        p.fetch_add(1, Ordering::SeqCst);
        let input = i.input().clone();
        async move { Ok(Outcome::Value(input)) }
    });
    let mut cases = hello();
    cases.push(Case::new("missing", "No reference").unwrap());
    argument_error(
        e.run_with(run(cases.clone())).await,
        "expected_output.*missing",
    );
    assert_eq!(performed.load(Ordering::SeqCst), 0);
    assert!(s.routes.bodies(RESPONSES).is_empty());
    assert!(
        e.run_with(run(cases).only(["greeting"]))
            .await
            .unwrap()
            .is_passed()
    );
}

// spec: evaluation_defaults_spec.rb:104
#[tokio::test]
async fn requires_references_for_explicitly_requested_built_in_correctness_too() {
    let s = defaults().await;
    let mut e = identity(&s.config);
    e.evaluation("correctness", None).unwrap();
    argument_error(
        e.run_with(run(vec![Case::new("missing", "Hello").unwrap()]))
            .await,
        "expected_output.*missing",
    );
    assert!(s.routes.bodies(RESPONSES).is_empty());
}

// spec: evaluation_defaults_spec.rb:112
#[tokio::test]
async fn accepts_false_zero_and_explicit_null_as_reference_values() {
    let s = defaults().await;
    let dataset: Vec<Case> = [json!(false), json!(0), Value::Null]
        .into_iter()
        .map(|v| {
            Case::new(v.to_string(), v.clone())
                .unwrap()
                .with_expected_output(v)
        })
        .collect();
    let report = identity(&s.config).run_with(run(dataset)).await.unwrap();
    assert!(report.is_passed());
    let references: Vec<Value> = s
        .routes
        .bodies(RESPONSES)
        .iter()
        .map(|b| {
            let content = b["input"].as_array().unwrap().last().unwrap()["content"].clone();
            serde_json::from_str::<Value>(content.as_str().unwrap()).unwrap()["expected_output"]
                .clone()
        })
        .collect();
    assert_eq!(references, [json!(false), json!(0), Value::Null]);
}

// spec: evaluation_defaults_spec.rb:123
#[tokio::test]
async fn disables_model_grading_explicitly_while_keeping_ruby_assertions_and_reports() {
    let s = defaults().await;
    let mut e = identity(&s.config);
    e.without_evaluator();
    e.assertions(|a| Ok(a.refute_empty(a.output())?));
    let report = e
        .run_with(run(vec![Case::new("no_reference", "Hello").unwrap()]))
        .await
        .unwrap();
    assert!(e.current_evaluator().is_none());
    assert!(report.is_passed());
    assert!(report.definitions().is_empty());
    assert!(report.first().unwrap().evaluations().is_empty());
    assert!(report.first().unwrap().assertion_count() > 0);
    assert_eq!(report.cost().total(), None);
    assert!(s.routes.bodies(RESPONSES).is_empty());
}

// spec: evaluation_defaults_spec.rb:137
#[tokio::test]
async fn does_not_pass_an_evaluation_with_grading_disabled_and_no_assertions() {
    let s = defaults().await;
    let mut e = identity(&s.config);
    e.without_evaluator();
    let report = e.run_with(run(hello())).await.unwrap();
    assert_eq!(report.first().unwrap().status(), Status::Measured);
    assert!(s.routes.bodies(RESPONSES).is_empty());
}

// spec: evaluation_defaults_spec.rb:144
#[tokio::test]
async fn inherits_disabled_grading_and_lets_a_child_reenable_it_without_changing_its_parent() {
    let s = defaults().await;
    let mut parent = identity(&s.config);
    parent.without_evaluator();
    let disabled = parent.subclass();
    let mut enabled = disabled.subclass();
    enabled.evaluator(Evaluator::model(REVIEW_MODEL).provider("openai"));
    assert!(disabled.current_evaluator().is_none());
    assert!(enabled.run_with(run(hello())).await.unwrap().is_passed());
    assert!(parent.current_evaluator().is_none());
}

// spec: evaluation_defaults_spec.rb:155
#[tokio::test]
async fn does_not_turn_resolved_defaults_into_inherited_explicit_criteria() {
    let s = defaults().await;
    let e = identity(&s.config);
    e.run_with(run(hello())).await.unwrap();
    let mut child = e.subclass();
    child
        .evaluation("concise", Some("The answer is brief"))
        .unwrap();
    child.run_with(run(hello())).await.unwrap();
    e.run_with(run(hello())).await.unwrap();
    assert_eq!(
        criteria(&s),
        [vec!["correctness"], vec!["concise"], vec!["correctness"]]
    );
    assert!(e.definitions().is_empty());
}

// spec: evaluation_defaults_spec.rb:166
#[tokio::test]
async fn allows_inherited_correctness_instructions_to_be_replaced_without_modifying_the_parent() {
    let s = defaults().await;
    let mut parent = identity(&s.config);
    parent
        .evaluation(
            "correctness",
            Some("The answer preserves every reference fact"),
        )
        .unwrap();
    let mut child = parent.subclass();
    child.evaluation("correctness", None).unwrap();
    assert_eq!(
        child.run_with(run(hello())).await.unwrap().definitions()[0]["instructions"],
        "The answer agrees with the expected output"
    );
    assert_eq!(
        parent.run_with(run(hello())).await.unwrap().definitions()[0]["instructions"],
        "The answer preserves every reference fact"
    );
}

// spec: evaluation_defaults_spec.rb:177
#[tokio::test]
async fn rejects_contradictory_disabled_grading_configurations() {
    // `evaluator(false, model:)`: `without_evaluator` takes no model options, so the first
    // contradiction cannot be written; the second is checked as in Ruby.
    let s = defaults().await;
    let mut e = identity(&s.config);
    e.without_evaluator();
    e.evaluation("correctness", None).unwrap();
    argument_error(
        e.run_with(run(hello())).await,
        "semantic evaluations with evaluator false",
    );
    assert!(s.routes.bodies(RESPONSES).is_empty());
}

async fn decisions() -> Stub {
    let s = defaults().await;
    s.routes.on(SYSTEMONE, systemone(0.9, 80));
    s
}

fn judge_config(s: &Stub) -> Judge {
    Judge::new().with_config(s.config.clone())
}

// spec: evaluation_defaults_spec.rb:198
#[tokio::test]
async fn uses_a_configured_judge_own_questions_including_correctness_without_imposing_reference_requirements()
 {
    let s = decisions().await;
    let judge = judge_config(&s)
        .probability(
            "correctness",
            "The answer follows the policy in its citations",
        )
        .unwrap()
        .probability("grounded", "The answer cites its sources")
        .unwrap()
        .model(JUDGMENT_MODEL)
        .provider("typesafe");
    let mut e = identity(&s.config);
    e.evaluator(Evaluator::judge(judge));
    e.evaluation_with("correctness", None, Some(0.8), None)
        .unwrap();
    let report = e
        .run_with(run(vec![Case::new("policy", "See [1]").unwrap()]))
        .await
        .unwrap();
    let names: Vec<&str> = report
        .first()
        .unwrap()
        .evaluations()
        .iter()
        .map(|r| r.name())
        .collect();
    assert_eq!(names, ["correctness", "grounded"]);
    assert!(report.first().unwrap().evaluations()[0].is_passed());
    assert_eq!(
        s.routes.bodies(SYSTEMONE)[0].pointer("/questions/correctness/instructions"),
        Some(&json!("The answer follows the policy in its citations"))
    );
}

// spec: evaluation_defaults_spec.rb:214
#[tokio::test]
async fn retains_native_correctness_probabilities_and_requires_an_explicit_threshold_to_count_them_as_passes()
 {
    let s = decisions().await;
    let mut e = identity(&s.config);
    e.evaluator(Evaluator::model(JUDGMENT_MODEL).provider("typesafe"));
    let measured = e.run_with(run(hello())).await.unwrap();
    e.evaluation_with("correctness", None, Some(0.8), None)
        .unwrap();
    let accepted = e.run_with(run(hello())).await.unwrap();
    assert_eq!(measured.first().unwrap().status(), Status::Measured);
    assert!(matches!(
        measured.first().unwrap().evaluations()[0].value(),
        Some(Measurement::Answer(Answer::Probability { .. }))
    ));
    assert!(accepted.is_passed());
    assert_eq!(
        s.routes
            .bodies(SYSTEMONE)
            .last()
            .unwrap()
            .pointer("/questions/correctness/instructions"),
        Some(&json!("The answer agrees with the expected output"))
    );
}

// ---- evaluation/evaluator_spec.rb -------------------------------------------------------------

fn completion(model: &str, content: &str, prompt: i64, completion: i64) -> Value {
    json!({
        "model": model,
        "choices": [{ "index": 0, "message": { "role": "assistant", "content": content }, "finish_reason": "stop" }],
        "usage": { "prompt_tokens": prompt, "completion_tokens": completion }
    })
}

/// The evaluator spec's setup: `evaluator(model:, provider: :openai, protocol: :chat_completions)`,
/// one `:correct` criterion, and Chat Completions answering with `verdicts`.
async fn reviewer(verdicts: Value) -> (Stub, Evaluation, Arc<Mutex<Value>>) {
    let s = stub(REVIEW_MODEL).await;
    let verdicts = Arc::new(Mutex::new(verdicts));
    let v = verdicts.clone();
    s.routes.on(COMPLETIONS, move |_| {
        json_response(&completion(
            REVIEW_MODEL,
            &v.lock().unwrap().to_string(),
            100,
            20,
        ))
    });
    let mut e = Evaluation::new();
    e.with_config(s.config.clone());
    e.perform(|i| {
        let out = i.input().as_str().unwrap_or_default().to_uppercase();
        async move { Ok(Outcome::Value(out.into())) }
    });
    e.assertions(|a| Ok(a.refute_empty(a.output())?));
    e.evaluator(
        Evaluator::model(REVIEW_MODEL)
            .provider("openai")
            .protocol(ProtocolName::ChatCompletions),
    );
    e.evaluation("correct", Some("Agrees with the expected output"))
        .unwrap();
    (s, e, verdicts)
}

fn answer_cases() -> Vec<Case> {
    vec![
        Case::new("answer", "hello")
            .unwrap()
            .with_expected_output("HELLO"),
    ]
}

fn pass() -> Value {
    json!({ "correct": { "verdict": "pass", "reason": "Matches the reference." } })
}

// spec: evaluation/evaluator_spec.rb:36
#[tokio::test]
async fn uses_the_built_in_reviewer_with_structured_output_and_separated_evidence() {
    let (s, e, _) = reviewer(pass()).await;
    let report = e.run_with(run(answer_cases())).await.unwrap();
    assert!(report.is_passed());
    assert_eq!(
        report.first().unwrap().evaluations()[0].reason(),
        Some("Matches the reference.")
    );
    let first = &s.routes.bodies(COMPLETIONS)[0];
    assert_eq!(
        required(first, "/response_format/json_schema/schema/required"),
        ["correct"]
    );
    let messages = first["messages"].as_array().unwrap();
    let instructions: String = messages[..messages.len() - 1]
        .iter()
        .filter_map(|m| m["content"].as_str())
        .collect();
    assert!(instructions.contains("untrusted data"));
    assert!(instructions.contains("Agrees with the expected output"));
    let evidence: Value =
        serde_json::from_str(messages.last().unwrap()["content"].as_str().unwrap()).unwrap();
    assert_eq!(evidence["actual"], "HELLO");
    assert_eq!(evidence["expected_output"], "HELLO");
    assert!(evidence.get("name").is_none());
    assert!(report.first().unwrap().evaluator_cost().total().unwrap() > 0.0);
}

// spec: evaluation/evaluator_spec.rb:50
#[tokio::test]
async fn accepts_an_instantiated_registry_model() {
    let (_s, mut e, _) = reviewer(pass()).await;
    let model = rust_llm::models().find(REVIEW_MODEL, None).unwrap();
    e.evaluator(Evaluator::registry_model(model).protocol(ProtocolName::ChatCompletions));
    assert!(e.run_with(run(answer_cases())).await.unwrap().is_passed());
}

struct PolicyReviewer(Arc<Config>);

impl Agent for PolicyReviewer {
    fn model(&self) -> Option<&str> {
        Some(REVIEW_MODEL)
    }
    fn protocol(&self) -> Option<ProtocolName> {
        Some(ProtocolName::ChatCompletions)
    }
    fn instructions(&self) -> Option<String> {
        Some("You are the policy reviewer.".into())
    }
    fn context(&self) -> Option<rust_llm::Context> {
        Some(rust_llm::Context::new((*self.0).clone()))
    }
}

// spec: evaluation/evaluator_spec.rb:56
#[tokio::test]
async fn preserves_a_supplied_agent_prompt_and_uses_a_fresh_conversation_for_every_case() {
    let (s, mut e, _) = reviewer(pass()).await;
    e.evaluator(Evaluator::agent(PolicyReviewer(s.config.clone())));
    let report = e
        .run_with(run(answer_cases()).repetitions(2))
        .await
        .unwrap();
    assert!(report.is_passed());
    let bodies = s.routes.bodies(COMPLETIONS);
    let sizes: Vec<usize> = bodies
        .iter()
        .map(|b| b["messages"].as_array().unwrap().len())
        .collect();
    assert_eq!(sizes, [3, 3]);
    let system = bodies[0]["messages"][0]["content"].as_str().unwrap();
    assert!(system.contains("You are the policy reviewer."));
    assert!(!system.contains("Accept equivalent"));
}

// spec: evaluation/evaluator_spec.rb:70
#[tokio::test]
async fn records_abstentions_without_counting_them_as_passes_or_failures() {
    let (_s, e, _) =
        reviewer(json!({ "correct": { "verdict": "unknown", "reason": "Missing." } })).await;
    let report = e.run_with(run(answer_cases())).await.unwrap();
    assert_eq!(report.first().unwrap().status(), Status::Unassessed);
    assert_eq!(report.count(Status::Unassessed), 1);
    assert_eq!(report.pass_rate(), Some(0.0));
}

// spec: evaluation/evaluator_spec.rb:79
#[tokio::test]
async fn does_not_hide_missing_extra_or_invalid_evaluator_answers() {
    let (_s, e, verdicts) = reviewer(pass()).await;
    for answers in [
        json!({}),
        json!({ "unexpected": { "verdict": "pass" } }),
        json!({ "correct": { "verdict": "maybe", "reason": "Unclear" } }),
    ] {
        *verdicts.lock().unwrap() = answers.clone();
        let report = e.run_with(run(answer_cases())).await.unwrap();
        assert_eq!(report.first().unwrap().status(), Status::Error, "{answers}");
        assert!(report.first().unwrap().evaluations()[0].error().is_some());
    }
}

// spec: evaluation/evaluator_spec.rb:89
#[tokio::test]
async fn records_evaluator_request_failures_and_still_assesses_later_cases() {
    let (s, e, _) = reviewer(pass()).await;
    // `to_timeout`: the provider never answers in time.
    s.routes.on(COMPLETIONS, |_| {
        ResponseTemplate::new(200).set_delay(std::time::Duration::from_secs(5))
    });
    let mut config = (*s.config).clone();
    config.request_timeout = std::time::Duration::from_millis(100);
    let mut e = e;
    e.with_config(Arc::new(config));
    let report = e
        .run_with(run(answer_cases()).repetitions(2))
        .await
        .unwrap();
    let statuses: Vec<Status> = report.iter().map(|t| t.status()).collect();
    assert_eq!(statuses, [Status::Error, Status::Error]);
    assert_eq!(report.first().unwrap().evaluations()[0].name(), "correct");
}

struct SchemaReviewer;

impl Agent for SchemaReviewer {
    fn model(&self) -> Option<&str> {
        Some(REVIEW_MODEL)
    }
    fn schema(&self) -> Option<Value> {
        Some(json!({ "type": "object", "properties": { "unrelated": { "type": "string" } } }))
    }
}

// spec: evaluation/evaluator_spec.rb:97
#[tokio::test]
async fn does_not_overwrite_an_agent_output_schema() {
    let (s, mut e, _) = reviewer(pass()).await;
    e.evaluator(Evaluator::agent(SchemaReviewer));
    let report = e.run_with(run(answer_cases())).await.unwrap();
    let error = report.first().unwrap().evaluations()[0].error().unwrap();
    assert!(error.message.contains("schema"));
    assert!(s.routes.bodies(COMPLETIONS).is_empty());
}

// spec: evaluation/evaluator_spec.rb:111
#[tokio::test]
async fn keeps_evaluator_errors_distinct_from_failed_application_assertions() {
    let (_s, mut e, _) =
        reviewer(json!({ "correct": { "verdict": "fail", "reason": "Wrong." } })).await;
    e.assertions(|a| Ok(a.assert_equal("WRONG", a.output().clone())?));
    let report = e.run_with(run(answer_cases())).await.unwrap();
    let first = report.first().unwrap();
    assert_eq!(first.status(), Status::Failed);
    assert!(first.assertion_failure().is_some());
    assert_eq!(first.evaluations()[0].status(), Status::Failed);
    assert!(first.error().is_none());
}

/// The decision-model context: `jev-latest` on TypeSafe answering 0.85.
async fn decision_reviewer() -> (Stub, Evaluation, Judge) {
    let (s, e, _) = reviewer(pass()).await;
    s.routes.on(SYSTEMONE, systemone(0.85, 80));
    let judge = judge_config(&s)
        .probability("correct", "Agrees with the reference")
        .unwrap()
        .model(JUDGMENT_MODEL)
        .provider("typesafe");
    (s, e, judge)
}

// spec: evaluation/evaluator_spec.rb:142
#[tokio::test]
async fn routes_a_decision_model_to_native_judgments_and_retains_its_probability() {
    let (s, mut e, _) = decision_reviewer().await;
    e.evaluator(Evaluator::registry_model(
        rust_llm::models().find(JUDGMENT_MODEL, None).unwrap(),
    ));
    let report = e.run_with(run(answer_cases())).await.unwrap();
    let value = report.first().unwrap().evaluations()[0].value();
    assert_eq!(
        value,
        Some(&Measurement::Answer(Answer::Probability {
            probability: 0.85
        }))
    );
    assert_eq!(report.first().unwrap().status(), Status::Measured);
    assert!(!report.is_passed());
    assert_eq!(s.routes.bodies(SYSTEMONE)[0]["state"]["actual"], "HELLO");
}

/// `Class.new(self.evaluation)` with the Judge as evaluator and `evaluation :correct, minimum:`,
/// which replaces the inherited `:correct` with a policy on the Judge's own question.
fn judged(e: &Evaluation, judge: Judge, minimum: f64) -> Evaluation {
    let mut child = e.subclass();
    child.evaluator(Evaluator::judge(judge));
    child
        .evaluation_with("correct", None, Some(minimum), None)
        .unwrap();
    child
}

// spec: evaluation/evaluator_spec.rb:153
#[tokio::test]
async fn uses_a_judge_class_questions_without_repeating_their_definitions() {
    let (s, e, judge) = decision_reviewer().await;
    let child = judged(&e, judge, 0.8);
    assert!(
        child
            .run_with(run(answer_cases()))
            .await
            .unwrap()
            .is_passed()
    );
    assert_eq!(
        s.routes.bodies(SYSTEMONE)[0].pointer("/questions/correct/instructions"),
        Some(&json!("Agrees with the reference"))
    );
}

// spec: evaluation/evaluator_spec.rb:162
#[tokio::test]
async fn does_not_treat_probabilities_below_the_explicit_threshold_as_passes() {
    let (_s, e, judge) = decision_reviewer().await;
    let child = judged(&e, judge, 0.9);
    let report = child.run_with(run(answer_cases())).await.unwrap();
    assert_eq!(report.first().unwrap().status(), Status::Failed);
}

// spec: evaluation/evaluator_spec.rb:170
#[tokio::test]
async fn rejects_conflicting_question_declarations_before_making_a_request() {
    let (s, mut e, judge) = decision_reviewer().await;
    e.evaluator(Evaluator::judge(judge));
    argument_error(
        e.run_with(run(answer_cases())).await,
        "Duplicate Judge question",
    );
    assert!(s.routes.bodies(SYSTEMONE).is_empty());
    assert!(s.routes.bodies(COMPLETIONS).is_empty());
}

// spec: evaluation/evaluator_spec.rb:177
#[tokio::test]
async fn supports_a_separate_evaluator_for_an_individual_criterion() {
    let (s, mut e, _) = decision_reviewer().await;
    e.evaluation_with(
        "correct_decision",
        Some("Matches reference"),
        None,
        Some(Evaluator::registry_model(
            rust_llm::models().find(JUDGMENT_MODEL, None).unwrap(),
        )),
    )
    .unwrap();
    let report = e.run_with(run(answer_cases())).await.unwrap();
    let evaluations = report.first().unwrap().evaluations();
    let names: Vec<&str> = evaluations.iter().map(|r| r.name()).collect();
    assert_eq!(names, ["correct", "correct_decision"]);
    assert!(evaluations[0].is_passed());
    assert_eq!(evaluations[1].status(), Status::Measured);
    assert_eq!(
        s.routes.bodies(COMPLETIONS).len() + s.routes.bodies(SYSTEMONE).len(),
        2
    );
}

// ---- evaluation_accounting_spec.rb ------------------------------------------------------------

fn task_reply() -> Value {
    json!({
        "id": "msg_1", "type": "message", "role": "assistant", "model": TASK_MODEL,
        "content": [{ "type": "text", "text": "Hello" }], "stop_reason": "end_turn",
        "usage": { "input_tokens": 9, "output_tokens": 2,
                   "cache_creation_input_tokens": 4, "cache_read_input_tokens": 6 }
    })
}

fn review_body(verdicts: &Value) -> Value {
    json!({
        "model": REVIEW_MODEL,
        "choices": [{ "index": 0, "message": { "role": "assistant", "content": verdicts.to_string() }, "finish_reason": "stop" }],
        "usage": { "prompt_tokens": 100, "completion_tokens": 20,
                   "prompt_tokens_details": { "cached_tokens": 10 },
                   "completion_tokens_details": { "reasoning_tokens": 3 } }
    })
}

fn greets() -> Value {
    json!({ "correct": { "verdict": "pass", "reason": "A greeting." } })
}

/// The accounting spec's setup: an Anthropic task, a Chat Completions reviewer, embeddings.
async fn accounting(verdicts: Value) -> (Stub, Evaluation, Arc<Mutex<Value>>) {
    let s = stub(REVIEW_MODEL).await;
    s.routes.on(MESSAGES, |_| json_response(&task_reply()));
    let verdicts = Arc::new(Mutex::new(verdicts));
    let v = verdicts.clone();
    s.routes.on(COMPLETIONS, move |_| {
        json_response(&review_body(&v.lock().unwrap()))
    });
    s.routes.on(EMBEDDINGS, |_| {
        json_response(&json!({ "model": EMBEDDING_MODEL, "data": [{ "embedding": [0.1, 0.2] }], "usage": { "prompt_tokens": 3 } }))
    });
    let mut e = Evaluation::new();
    e.with_config(s.config.clone());
    e.evaluation("correct", Some("The output greets the user"))
        .unwrap();
    let config = s.config.clone();
    e.perform(move |i| {
        let config = config.clone();
        async move {
            let mut chat =
                rust_llm::Chat::with_config(config, Some(TASK_MODEL), Some("anthropic"), false)?;
            let answer = chat.ask(i.input().as_str().unwrap_or_default()).await?;
            Ok(Outcome::Value(answer.content().into()))
        }
    });
    e.assertions(|a| Ok(a.refute_empty(a.output())?));
    e.evaluator(
        Evaluator::model(REVIEW_MODEL)
            .provider("openai")
            .protocol(ProtocolName::ChatCompletions),
    );
    (s, e, verdicts)
}

fn greeting_cases() -> Vec<Case> {
    vec![Case::new("greeting", "Hello").unwrap()]
}

/// `Tokens#to_h`: only the buckets that were reported.
fn tokens_h(tokens: &rust_llm::Tokens) -> Value {
    let mut h = Map::new();
    for (key, value) in [
        ("input_tokens", tokens.input),
        ("output_tokens", tokens.output),
        ("cache_read_tokens", tokens.cache_read),
        ("cache_write_tokens", tokens.cache_write),
        ("thinking_tokens", tokens.thinking),
    ] {
        if let Some(v) = value {
            h.insert(key.into(), v.into());
        }
    }
    Value::Object(h)
}

fn close(a: Option<f64>, b: Option<f64>) {
    let (a, b) = (a.expect("a total"), b.expect("a total"));
    assert!((a - b).abs() < 1e-12, "{a} != {b}");
}

// spec: evaluation_accounting_spec.rb:50
#[tokio::test]
async fn accounts_for_plain_text_returns_every_repetition_and_the_standard_token_buckets() {
    let (_s, e, _) = accounting(greets()).await;
    let report = e
        .run_with(run(greeting_cases()).repetitions(2))
        .await
        .unwrap();
    let trial = report.first().unwrap();
    assert!(report.is_passed(), "{report}");
    assert_eq!(trial.result().unwrap().value(), Some(&json!("Hello")));
    assert_eq!(
        tokens_h(&report.tokens()),
        json!({ "input_tokens": 198, "output_tokens": 44, "cache_read_tokens": 32,
                "cache_write_tokens": 8, "thinking_tokens": 6 })
    );
    assert_eq!(trial.task_tokens().input, Some(9));
    assert_eq!(trial.evaluator_tokens().input, Some(90));
    close(
        trial.cost().total(),
        Some(trial.task_cost().total().unwrap() + trial.evaluator_cost().total().unwrap()),
    );
    close(
        report.cost().total(),
        Some(trial.cost().total().unwrap() * 2.0),
    );
    let h = report.to_h();
    assert_eq!(h["tokens"], tokens_h(&report.tokens()));
    assert!(h.get("cost").is_some());
    let t = trial.to_h();
    assert_eq!(t["task_tokens"], tokens_h(&trial.task_tokens()));
    assert_eq!(t["evaluator_tokens"], tokens_h(&trial.evaluator_tokens()));
}

fn embed_hook(
    config: &Arc<Config>,
    text: &'static str,
) -> impl Fn(rust_llm::evaluation::Instance) -> futures::future::BoxFuture<'static, Result<(), Failure>>
+ Send
+ Sync
+ 'static {
    let config = config.clone();
    move |_| {
        let config = config.clone();
        Box::pin(async move {
            rust_llm::embed(
                text,
                EmbedOptions {
                    model: Some(EMBEDDING_MODEL),
                    provider: Some("openai"),
                    config: Some(config),
                    ..Default::default()
                },
            )
            .await?;
            Ok(())
        })
    }
}

// spec: evaluation_accounting_spec.rb:68
#[tokio::test]
async fn counts_calls_in_setup_and_teardown_even_when_their_results_are_discarded() {
    let (s, mut e, _) = accounting(greets()).await;
    e.setup(embed_hook(&s.config, "Setup"));
    e.teardown(embed_hook(&s.config, "Cleanup"));
    let report = e.run_with(run(greeting_cases())).await.unwrap();
    let trial = report.first().unwrap();
    assert_eq!(trial.task_tokens().input, Some(15));
    assert_eq!(trial.tokens().input, Some(105));
}

// spec: evaluation_accounting_spec.rb:79
#[tokio::test]
async fn keeps_task_usage_when_model_grading_is_disabled() {
    let (s, _, _) = accounting(greets()).await;
    let mut e = Evaluation::new();
    e.without_evaluator();
    let config = s.config.clone();
    e.perform(move |i| {
        let config = config.clone();
        async move {
            let mut chat =
                rust_llm::Chat::with_config(config, Some(TASK_MODEL), Some("anthropic"), false)?;
            Ok(Outcome::from(
                chat.ask(i.input().as_str().unwrap_or_default()).await?,
            ))
        }
    });
    e.assertions(|a| Ok(a.refute_empty(a.output())?));
    let report = e.run_with(run(greeting_cases())).await.unwrap();
    let first = report.first().unwrap();
    assert!(report.is_passed());
    assert!(first.evaluations().is_empty());
    assert_eq!(first.evaluator_tokens().input, None);
    assert_eq!(tokens_h(&report.tokens()), tokens_h(&first.task_tokens()));
    assert_eq!(report.cost().total(), first.task_cost().total());
    assert_eq!(report.tokens().input, Some(9));
    assert!(s.routes.bodies(COMPLETIONS).is_empty());
}

// spec: evaluation_accounting_spec.rb:103
#[tokio::test]
async fn excludes_a_returned_conversation_history_that_was_billed_before_the_run() {
    let (s, mut e, _) = accounting(greets()).await;
    let mut chat =
        rust_llm::Chat::with_config(s.config.clone(), Some(TASK_MODEL), Some("anthropic"), false)
            .unwrap();
    chat.ask("Earlier question").await.unwrap();
    let shared = Arc::new(tokio::sync::Mutex::new(Some(chat)));
    let slot = shared.clone();
    e.perform(move |i| {
        let slot = slot.clone();
        async move {
            let mut chat = slot.lock().await.take().expect("the chat");
            chat.ask(i.input().as_str().unwrap_or_default()).await?;
            Ok(Outcome::from(chat))
        }
    });
    let report = e.run_with(run(greeting_cases())).await.unwrap();
    let trial = report.first().unwrap();
    assert_eq!(
        trial.result().unwrap().chat().unwrap().tokens().input,
        Some(18)
    );
    assert_eq!(trial.task_tokens().input, Some(9));
}

// spec: evaluation_accounting_spec.rb:117
#[tokio::test]
async fn sends_every_turn_to_the_evaluator_when_perform_returns_a_multi_turn_chat() {
    let (s, mut e, _) = accounting(greets()).await;
    let config = s.config.clone();
    e.perform(move |i| {
        let config = config.clone();
        async move {
            let mut chat =
                rust_llm::Chat::with_config(config, Some(TASK_MODEL), Some("anthropic"), false)?;
            for question in i.input().as_array().into_iter().flatten() {
                chat.ask(question.as_str().unwrap_or_default()).await?;
            }
            Ok(Outcome::from(chat))
        }
    });
    let dataset = vec![Case::new("conversation", json!(["First question", "Follow-up"])).unwrap()];
    let report = e.run_with(run(dataset)).await.unwrap();
    let trial = report.first().unwrap();
    let body = s.routes.bodies(COMPLETIONS).pop().unwrap();
    let content = body["messages"].as_array().unwrap().last().unwrap()["content"].clone();
    let evidence: Value = serde_json::from_str(content.as_str().unwrap()).unwrap();
    let contents: Vec<Value> = evidence["actual"]["messages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["content"].clone())
        .collect();
    assert!(trial.is_passed());
    assert_eq!(contents, ["First question", "Hello", "Follow-up", "Hello"]);
    assert_eq!(trial.output(), &json!("Hello"));
    assert_eq!(trial.task_tokens().input, Some(18));
}

// spec: evaluation_accounting_spec.rb:140
#[tokio::test]
async fn keeps_billed_usage_when_application_code_fails_after_receiving_a_response() {
    let (s, mut e, _) = accounting(greets()).await;
    let config = s.config.clone();
    e.perform(move |i| {
        let config = config.clone();
        async move {
            let mut chat =
                rust_llm::Chat::with_config(config, Some(TASK_MODEL), Some("anthropic"), false)?;
            chat.ask(i.input().as_str().unwrap_or_default()).await?;
            Err(Failure::error("Application failed"))
        }
    });
    let report = e.run_with(run(greeting_cases())).await.unwrap();
    let trial = report.first().unwrap();
    assert_eq!(trial.status(), Status::Error);
    assert_eq!(trial.tokens().input, Some(9));
    assert!(trial.cost().total().unwrap() > 0.0);
    assert_eq!(tokens_h(&trial.evaluator_tokens()), json!({}));
}

// spec: evaluation_accounting_spec.rb:155
#[tokio::test]
async fn keeps_the_known_evaluator_cost_when_the_returned_assessment_is_malformed() {
    let (_s, e, _) = accounting(json!({})).await;
    let report = e.run_with(run(greeting_cases())).await.unwrap();
    let trial = report.first().unwrap();
    assert_eq!(trial.status(), Status::Error);
    assert_eq!(trial.evaluator_tokens().input, Some(90));
    assert!(trial.evaluator_cost().total().unwrap() > 0.0);
    assert!(trial.cost().total().unwrap() > 0.0);
}

// spec: evaluation_accounting_spec.rb:165
#[tokio::test]
async fn preserves_unknown_totals_when_a_failed_evaluator_attempt_may_have_been_billed() {
    let (s, e, _) = accounting(greets()).await;
    s.routes.on(COMPLETIONS, |_| {
        ResponseTemplate::new(500)
            .set_body_json(json!({ "error": { "message": "Failed after acceptance" } }))
    });
    let report = e.run_with(run(greeting_cases())).await.unwrap();
    let trial = report.first().unwrap();
    assert_eq!(trial.status(), Status::Error);
    assert!(trial.task_cost().total().unwrap() > 0.0);
    assert_eq!(trial.evaluator_cost().total(), None);
    assert_eq!(trial.cost().total(), None);
    assert_eq!(trial.tokens().input, Some(9));
}

// spec: evaluation_accounting_spec.rb:178
#[tokio::test]
async fn counts_a_grouped_evaluator_request_once_for_all_its_criteria() {
    let (_s, mut e, verdicts) = accounting(greets()).await;
    e.evaluation("concise", Some("The output is brief"))
        .unwrap();
    verdicts.lock().unwrap()["concise"] = json!({ "verdict": "pass", "reason": "One word." });
    let report = e.run_with(run(greeting_cases())).await.unwrap();
    let trial = report.first().unwrap();
    assert_eq!(trial.evaluations().len(), 2);
    assert_eq!(trial.evaluator_tokens().input, Some(90));
}

struct PolicyLookup(Arc<Config>);

#[async_trait::async_trait]
impl rust_llm::Tool for PolicyLookup {
    fn name(&self) -> String {
        "evaluation_policy_lookup".into()
    }
    fn description(&self) -> String {
        "Look up the greeting policy.".into()
    }
    async fn execute(
        &self,
        _: Map<String, Value>,
        _: &rust_llm::ToolCall,
    ) -> Result<rust_llm::ToolResult, rust_llm::ToolError> {
        let e = rust_llm::embed(
            "Policy",
            EmbedOptions {
                model: Some(EMBEDDING_MODEL),
                provider: Some("openai"),
                config: Some(self.0.clone()),
                ..Default::default()
            },
        )
        .await?;
        Ok(format!("{:?}", e.vectors).into())
    }
}

struct ToolReviewer(Arc<Config>);

impl Agent for ToolReviewer {
    fn model(&self) -> Option<&str> {
        Some(REVIEW_MODEL)
    }
    fn provider(&self) -> Option<&str> {
        Some("openai")
    }
    fn protocol(&self) -> Option<ProtocolName> {
        Some(ProtocolName::ChatCompletions)
    }
    fn tools(&self) -> Vec<SharedTool> {
        vec![Arc::new(PolicyLookup(self.0.clone()))]
    }
    fn context(&self) -> Option<rust_llm::Context> {
        Some(rust_llm::Context::new((*self.0).clone()))
    }
}

// spec: evaluation_accounting_spec.rb:187
#[tokio::test]
async fn includes_model_calls_made_inside_an_evaluator_tool() {
    let (s, mut e, _) = accounting(greets()).await;
    e.evaluator(Evaluator::agent(ToolReviewer(s.config.clone())));
    let tool_reply = json!({
        "model": REVIEW_MODEL,
        "choices": [{ "index": 0, "finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": null,
            "tool_calls": [{ "id": "call_1", "type": "function", "function": { "name": "evaluation_policy_lookup", "arguments": "{}" } }]
        } }],
        "usage": { "prompt_tokens": 12, "completion_tokens": 4 }
    });
    s.routes
        .sequence(COMPLETIONS, vec![tool_reply, review_body(&greets())]);
    let report = e.run_with(run(greeting_cases())).await.unwrap();
    let trial = report.first().unwrap();
    assert!(trial.is_passed(), "{report}");
    assert_eq!(trial.evaluator_tokens().input, Some(105));
    assert_eq!(trial.evaluator_tokens().output, Some(24));
}

// spec: evaluation_accounting_spec.rb:212
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn includes_child_threads_and_fibers_without_mixing_concurrent_evaluations_or_unrelated_calls()
 {
    // Ruby's threads and fibers inherit the trial's usage scope; here a spawned task joins it
    // with `in_current_scope`, and a nested future (Ruby's fiber) runs inside it already.
    let (s, mut e, _) = accounting(greets()).await;
    let ready = Arc::new(tokio::sync::Barrier::new(3));
    let proceed = Arc::new(tokio::sync::Notify::new());
    let config = s.config.clone();
    let (r, p) = (ready.clone(), proceed.clone());
    e.perform(move |i| {
        let (config, ready, proceed) = (config.clone(), r.clone(), p.clone());
        async move {
            let input = i.input().as_str().unwrap_or_default().to_string();
            let notified = proceed.notified();
            ready.wait().await;
            notified.await;
            let (c, q) = (config.clone(), input.clone());
            tokio::spawn(rust_llm::evaluation::in_current_scope(async move {
                let mut chat =
                    rust_llm::Chat::with_config(c, Some(TASK_MODEL), Some("anthropic"), false)?;
                chat.ask(q).await
            }))
            .await
            .map_err(|e| Failure::error(e.to_string()))??;
            let mut chat =
                rust_llm::Chat::with_config(config, Some(TASK_MODEL), Some("anthropic"), false)?;
            Ok(Outcome::Value(chat.ask(input).await?.content().into()))
        }
    });
    let runs: Vec<_> = (0..2)
        .map(|_| {
            let e = e.clone();
            tokio::spawn(async move { e.run_with(run(greeting_cases())).await })
        })
        .collect();
    ready.wait().await;
    rust_llm::embed(
        "Unrelated",
        EmbedOptions {
            model: Some(EMBEDDING_MODEL),
            provider: Some("openai"),
            config: Some(s.config.clone()),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    proceed.notify_waiters();
    let mut reports = Vec::new();
    for r in runs {
        reports.push(r.await.unwrap().unwrap());
    }
    assert!(reports.iter().all(|r| r.is_passed()));
    let task: Vec<Option<i64>> = reports
        .iter()
        .map(|r| r.first().unwrap().task_tokens().input)
        .collect();
    assert_eq!(task, [Some(18), Some(18)]);
    let total: Vec<Option<i64>> = reports.iter().map(|r| r.tokens().input).collect();
    assert_eq!(total, [Some(108), Some(108)]);
}

// spec: evaluation_accounting_spec.rb:233
#[tokio::test]
async fn counts_nested_evaluations_as_task_work_and_restores_the_outer_scope() {
    let (_s, outer, _) = accounting(greets()).await;
    let inner = outer.subclass();
    let mut outer = outer;
    outer.perform(move |_| {
        let inner = inner.clone();
        async move {
            inner.run_with(run(greeting_cases())).await?;
            Ok(Outcome::Value("Hello".into()))
        }
    });
    let report = outer.run_with(run(greeting_cases())).await.unwrap();
    let trial = report.first().unwrap();
    assert_eq!(trial.task_tokens().input, Some(99));
    assert_eq!(trial.evaluator_tokens().input, Some(90));
    assert_eq!(trial.tokens().input, Some(189));
}

// spec: evaluation_accounting_spec.rb:248
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn keeps_a_finished_report_unchanged_when_detached_work_completes_later() {
    let (s, mut e, _) = accounting(greets()).await;
    let proceed = Arc::new(tokio::sync::Notify::new());
    let handle: Arc<Mutex<Option<tokio::task::JoinHandle<rust_llm::Result<rust_llm::Message>>>>> =
        Default::default();
    let (config, p, h) = (s.config.clone(), proceed.clone(), handle.clone());
    e.perform(move |i| {
        let (config, proceed, handle) = (config.clone(), p.clone(), h.clone());
        async move {
            let input = i.input().as_str().unwrap_or_default().to_string();
            // Detached work that stays in the trial's scope (Ruby's `Thread.new`).
            let task = tokio::spawn(rust_llm::evaluation::in_current_scope(async move {
                proceed.notified().await;
                let mut chat = rust_llm::Chat::with_config(
                    config,
                    Some(TASK_MODEL),
                    Some("anthropic"),
                    false,
                )?;
                chat.ask(input).await
            }));
            *handle.lock().unwrap() = Some(task);
            Ok(Outcome::Value("Hello".into()))
        }
    });
    let report = e.run_with(run(greeting_cases())).await.unwrap();
    proceed.notify_waiters();
    let task = handle.lock().unwrap().take().unwrap();
    assert_eq!(task.await.unwrap().unwrap().content(), "Hello");
    assert_eq!(report.tokens().input, Some(90));
    assert_eq!(tokens_h(&report.first().unwrap().task_tokens()), json!({}));
}
