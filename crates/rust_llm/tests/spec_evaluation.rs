//! RubyLLM 2.1's `evaluation_spec.rb`, `evaluation/assertions_spec.rb`,
//! `evaluation/result_spec.rb`, and `evaluation_progress_spec.rb`: evaluations that run without a
//! model (`evaluator false`), their datasets, reports, and progress events.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use rust_llm::evaluation::{
    Case, Dataset, Evaluation, Failure, Measurement, Outcome, Result as Verdict, RunOptions,
    Status, TrialError,
};
use rust_llm::{Answer, Config, ErrorKind};
use serde_json::{Map, Value, json};

fn case(name: &str, inputs: impl Into<Value>, expected: impl Into<Value>) -> Case {
    Case::new(name, inputs)
        .unwrap()
        .with_expected_output(expected)
}

fn greeting() -> Vec<Case> {
    vec![case("greeting", "hello", "HELLO")]
}

fn upcase(input: &Value) -> Value {
    input
        .as_str()
        .map_or(Value::Null, |s| s.to_uppercase().into())
}

/// The spec's `evaluation`: `evaluator false`, `perform` upcases, assertions compare.
fn evaluation() -> Evaluation {
    upcase_evaluation(Evaluation::new())
}

fn upcase_evaluation(mut e: Evaluation) -> Evaluation {
    e.without_evaluator();
    e.perform(|i| async move {
        let s = i
            .input()
            .as_str()
            .ok_or_else(|| Failure::error("undefined method 'upcase'"))?;
        Ok(Outcome::Value(s.to_uppercase().into()))
    });
    e.assertions(|a| {
        a.assert_equal(a.expected_output().clone(), a.output().clone())?;
        a.refute_empty(a.output())?;
        Ok(())
    });
    e
}

fn statuses(report: &rust_llm::evaluation::Report) -> Vec<Status> {
    report.iter().map(|t| t.status()).collect()
}

fn argument_error<T: std::fmt::Debug>(result: rust_llm::Result<T>, pattern: &str) {
    match result {
        Err(rust_llm::Error::Argument(m)) => {
            assert!(m.contains(pattern), "{m:?} does not mention {pattern:?}")
        }
        other => panic!("expected an ArgumentError matching {pattern:?}, got {other:?}"),
    }
}

// spec: evaluation_spec.rb:26
#[tokio::test]
async fn runs_ordinary_assertions_without_calling_a_model() {
    let report = evaluation()
        .run_with(RunOptions::default().dataset(greeting()))
        .await
        .unwrap();
    assert!(report.is_passed());
    let first = report.first().unwrap();
    assert_eq!(first.output(), &json!("HELLO"));
    assert_eq!(first.result().unwrap().value(), Some(&json!("HELLO")));
    assert!(first.assertion_count() >= 3);
    assert_eq!(report.pass_rate(), Some(1.0));
    assert!(report.to_string().contains("greeting [1]: passed"));
}

// spec: evaluation_spec.rb:37
#[tokio::test]
async fn records_failed_assertions_and_continues_to_later_cases() {
    let mut cases = greeting();
    cases.push(case("bad", "wrong", "RIGHT"));
    let report = evaluation()
        .run_with(RunOptions::default().dataset(cases))
        .await
        .unwrap();
    assert_eq!(statuses(&report), [Status::Passed, Status::Failed]);
    assert!(
        report
            .trials()
            .last()
            .unwrap()
            .assertion_failure()
            .is_some()
    );
    assert_eq!(report.pass_rate(), Some(0.5));
    assert!(!report.is_passed());
    let text = report.to_string();
    assert!(text.contains("RIGHT") && text.contains("WRONG"), "{text}");
}

// spec: evaluation_spec.rb:48
#[tokio::test]
async fn lists_and_selects_named_cases_without_executing_the_application_during_discovery() {
    let mut cases = greeting();
    cases.push(case("second", "bye", "BYE"));
    let e = evaluation();
    let dataset = Dataset::from(cases.clone());
    let selected = e
        .cases(Some(&dataset), Some(&["second".to_string()]))
        .unwrap();
    assert_eq!(selected, vec![cases[1].clone()]);
    let report = e
        .run_with(
            RunOptions::default()
                .dataset(cases.clone())
                .only(["second"]),
        )
        .await
        .unwrap();
    let names: Vec<&str> = report.iter().map(|t| t.test_case().name()).collect();
    assert_eq!(names, ["second"]);
    assert!(report.is_passed());
    argument_error(
        e.run_with(RunOptions::default().dataset(cases.clone()).only(["typo"]))
            .await,
        "Unknown evaluation cases",
    );
    argument_error(
        e.run_with(
            RunOptions::default()
                .dataset(cases)
                .only(Vec::<String>::new()),
        )
        .await,
        "Unknown evaluation cases",
    );
}

// spec: evaluation_spec.rb:59
#[tokio::test]
async fn isolates_input_mutations_and_instance_state_across_cases_and_repetitions() {
    let mut e = Evaluation::new();
    e.without_evaluator();
    e.perform(|i| async move {
        let counter = i.get("counter").and_then(|v| v.as_i64()).unwrap_or(0) + 1;
        i.set("counter", counter);
        let mut input = i.input().clone();
        input.as_array_mut().unwrap().push(counter.into());
        Ok(Outcome::Value(input))
    });
    e.assertions(|a| Ok(a.assert_equal(json!([1]), a.output().clone())?));
    let original = Case::new("mutable", json!([])).unwrap();
    let report = e
        .run_with(
            RunOptions::default()
                .dataset(vec![original.clone()])
                .repetitions(3),
        )
        .await
        .unwrap();
    let outputs: Vec<&Value> = report.iter().map(|t| t.output()).collect();
    assert_eq!(outputs, [&json!([1]), &json!([1]), &json!([1])]);
    let reps: Vec<usize> = report.iter().map(|t| t.repetition()).collect();
    assert_eq!(reps, [1, 2, 3]);
    assert_eq!(original.inputs(), &json!([]));
}

// spec: evaluation_spec.rb:80
#[tokio::test]
async fn records_task_errors_and_runs_teardown() {
    let cleanups = Arc::new(AtomicUsize::new(0));
    let mut e = evaluation();
    e.perform(|_| async { Err(Failure::error("task failed")) });
    let seen = cleanups.clone();
    e.teardown(move |_| {
        let seen = seen.clone();
        async move {
            seen.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    });
    let report = e
        .run_with(RunOptions::default().dataset(greeting()).repetitions(2))
        .await
        .unwrap();
    assert_eq!(statuses(&report), [Status::Error, Status::Error]);
    assert_eq!(
        report.first().unwrap().error().unwrap().message,
        "task failed"
    );
    assert_eq!(cleanups.load(Ordering::SeqCst), 2);
    assert_eq!(report.pass_rate(), Some(0.0));
}

// spec: evaluation_spec.rb:92
#[tokio::test]
async fn records_cleanup_errors_without_discarding_the_trial() {
    let mut e = evaluation();
    e.teardown(|_| async { Err(Failure::error("cleanup failed")) });
    let report = e
        .run_with(RunOptions::default().dataset(greeting()))
        .await
        .unwrap();
    let first = report.first().unwrap();
    assert_eq!(first.status(), Status::Error);
    assert_eq!(first.error().unwrap().message, "cleanup failed");
}

// spec: evaluation_spec.rb:100
#[tokio::test]
async fn does_not_pass_a_case_that_assessed_nothing() {
    let mut e = evaluation();
    e.assertions(|_| Ok(()));
    let report = e
        .run_with(RunOptions::default().dataset(greeting()))
        .await
        .unwrap();
    assert_eq!(report.first().unwrap().status(), Status::Measured);
}

#[derive(Debug)]
struct Invoice {
    total: i64,
}

// spec: evaluation_spec.rb:106
#[tokio::test]
async fn inherits_criteria_and_adapters_without_modifying_the_parent() {
    let mut parent = evaluation();
    parent
        .evaluation("correct", Some("Correct answer"))
        .unwrap();
    let mut child = parent.subclass();
    child
        .evaluation("correct", Some("A more specific criterion"))
        .unwrap();
    child.evaluation("short", Some("Short answer")).unwrap();
    child.adapt(|invoice: &Invoice| json!({ "total": invoice.total }));

    let names: Vec<String> = parent.definitions().into_iter().map(|d| d.0).collect();
    assert_eq!(names, ["correct"]);
    assert_eq!(parent.definitions()[0].1.as_deref(), Some("Correct answer"));
    assert_eq!(parent.adapter_count(), 0);
    argument_error(
        child.evaluation("short", Some("Duplicate")).map(|_| ()),
        "Duplicate",
    );
}

// spec: evaluation_spec.rb:119
#[tokio::test]
async fn rejects_invalid_runs_and_incomplete_definitions_before_executing() {
    let mut e = evaluation();
    argument_error(
        e.run_with(RunOptions::default().dataset(Vec::new()).repetitions(0))
            .await,
        "Repetitions",
    );
    argument_error(
        e.run_with(RunOptions::default().dataset(Vec::new())).await,
        "empty",
    );
    let twice = [greeting(), greeting()].concat();
    argument_error(
        e.run_with(RunOptions::default().dataset(twice)).await,
        "unique",
    );
    e.evaluation("correct", None).unwrap();
    e.evaluator(rust_llm::evaluation::Evaluator::model("gpt-5-nano"));
    argument_error(
        e.run_with(RunOptions::default().dataset(greeting())).await,
        "Missing instructions",
    );
}

/// `allow(RubyLLM::Prompt).to receive(:root).and_return(<dir>/app/prompts)`.
fn rooted_at(e: &mut Evaluation, dir: &std::path::Path) {
    let mut config = Config::default();
    config.set(
        "prompt_root",
        dir.join("app/prompts").to_string_lossy().into_owned(),
    );
    e.with_config(Arc::new(config));
}

fn rows(cases: &[Case]) -> Value {
    Value::Array(cases.iter().map(Case::to_h).collect())
}

fn tempdir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "rust_llm_eval_{tag}_{}",
        uuid::Uuid::new_v4().simple()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

// spec: evaluation_spec.rb:128
#[tokio::test]
async fn discovers_yaml_by_class_name_under_app_evals_and_rejects_ambiguous_matches() {
    let dir = tempdir("discover");
    let root = dir.join("app/evals");
    std::fs::create_dir_all(&root).unwrap();
    // `stub_const('GreetingEvaluation', evaluation)`: the same definition under that name.
    let mut e = upcase_evaluation(Evaluation::named("GreetingEvaluation"));
    rooted_at(&mut e, &dir);
    let yaml = serde_yaml::to_string(&json!({ "cases": rows(&greeting()) })).unwrap();
    std::fs::write(root.join("greeting_evaluation.yml"), yaml).unwrap();
    assert!(e.run().await.unwrap().is_passed());
    std::fs::write(
        root.join("greeting_evaluation.json"),
        rows(&greeting()).to_string(),
    )
    .unwrap();
    argument_error(e.run().await, "Ambiguous");
    std::fs::remove_dir_all(dir).ok();
}

// spec: evaluation_spec.rb:142
#[tokio::test]
async fn loads_yaml_json_and_jsonl_with_the_same_data_and_explicit_paths() {
    let dir = tempdir("formats");
    let files = [
        (
            "yml",
            serde_yaml::to_string(&json!({ "cases": rows(&greeting()) })).unwrap(),
        ),
        ("json", rows(&greeting()).to_string()),
        ("jsonl", greeting()[0].to_h().to_string()),
    ];
    let mut e = evaluation();
    for (extension, body) in files {
        let path = dir.join(format!("cases.{extension}"));
        std::fs::write(&path, body).unwrap();
        e.dataset(path);
        assert!(e.run().await.unwrap().is_passed(), "{extension}");
    }
    std::fs::remove_dir_all(dir).ok();
}

// spec: evaluation_spec.rb:155
#[tokio::test]
async fn loads_enumerable_cases_from_a_dataset_block_once_per_run() {
    let runs = Arc::new(AtomicUsize::new(0));
    let counter = runs.clone();
    let mut e = evaluation();
    e.dataset(Dataset::block(move || {
        counter.fetch_add(1, Ordering::SeqCst);
        Ok(Dataset::Cases(greeting()))
    }));
    assert!(
        e.run_with(RunOptions::default().repetitions(2))
            .await
            .unwrap()
            .is_passed()
    );
    assert_eq!(runs.load(Ordering::SeqCst), 1);
}

// spec: evaluation_spec.rb:167
#[tokio::test]
async fn rejects_unsafe_yaml_instead_of_constructing_arbitrary_objects() {
    let dir = tempdir("unsafe");
    let path = dir.join("evaluation.yml");
    std::fs::write(&path, "--- !ruby/object:Object {}\n").unwrap();
    // `Psych::DisallowedClass` ("Tried to load unspecified class: Object").
    argument_error(
        evaluation()
            .run_with(RunOptions::default().dataset(path))
            .await,
        "Tried to load unspecified class",
    );
    std::fs::remove_dir_all(dir).ok();
}

// spec: evaluation_spec.rb:175
#[tokio::test]
async fn preserves_false_zero_and_explicit_nil_reference_answers() {
    for expected in [json!(false), json!(0), Value::Null] {
        let test_case = case("value", expected.clone(), expected.clone());
        let mut e = evaluation();
        e.perform(|i| {
            let input = i.input().clone();
            async move { Ok(Outcome::Value(input)) }
        });
        e.assertions(|a| {
            if a.expected_output().is_null() {
                a.assert_nil(a.output())?;
            } else {
                a.assert_equal(a.expected_output().clone(), a.output().clone())?;
            }
            Ok(())
        });
        let report = e
            .run_with(RunOptions::default().dataset(vec![test_case.clone()]))
            .await
            .unwrap();
        assert!(report.is_passed(), "{expected}");
        assert!(test_case.is_expected_output());
        assert!(test_case.to_h().get("expected_output").is_some());
    }
    assert!(
        !Case::new("no_reference", Value::Null)
            .unwrap()
            .is_expected_output()
    );
}

// spec: evaluation_spec.rb:189
#[tokio::test]
async fn saves_reports_with_failed_cases_and_serializable_evidence() {
    let report = evaluation()
        .run_with(RunOptions::default().dataset(greeting()))
        .await
        .unwrap();
    let dir = tempdir("report");
    let path = dir.join("report.json");
    assert_eq!(report.save(&path).unwrap(), path);
    let data: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let first = &data["trials"][0];
    assert_eq!(first["status"], "passed");
    assert_eq!(first["evidence"], "HELLO");
    assert_eq!(data["id"], report.id());
    std::fs::remove_dir_all(dir).ok();
}

// spec: evaluation_spec.rb:199
#[tokio::test]
async fn rejects_imported_executable_evaluators_instead_of_silently_ignoring_their_checks() {
    let data = json!({ "cases": rows(&greeting()), "evaluators": ["EqualsExpected"] });
    argument_error(
        evaluation()
            .run_with(RunOptions::default().dataset(data))
            .await,
        "declare evaluators in Rust",
    );
}

// spec: evaluation_spec.rb:205
#[tokio::test]
async fn runs_teardown_when_setup_fails_and_keeps_the_original_error() {
    let cleanups = Arc::new(AtomicUsize::new(0));
    let mut e = evaluation();
    e.setup(|_| async { Err(Failure::error("setup failed")) });
    let seen = cleanups.clone();
    e.teardown(move |_| {
        seen.fetch_add(1, Ordering::SeqCst);
        async { Err(Failure::error("cleanup also failed")) }
    });
    let report = e
        .run_with(RunOptions::default().dataset(greeting()))
        .await
        .unwrap();
    assert_eq!(
        report.first().unwrap().error().unwrap().message,
        "setup failed"
    );
    assert_eq!(cleanups.load(Ordering::SeqCst), 1);
}

// spec: evaluation_spec.rb:218
#[tokio::test]
async fn rejects_an_evaluation_without_perform_before_starting_cases() {
    argument_error(
        Evaluation::new()
            .run_with(RunOptions::default().dataset(greeting()))
            .await,
        "Define perform",
    );
}

// spec: evaluation_spec.rb:222
#[tokio::test]
async fn rejects_conflicting_dataset_arguments_and_invalid_acceptance_thresholds() {
    // `dataset(cases) { cases }`: Rust takes one `Dataset` value, so a path and a block together
    // cannot be expressed; the thresholds and names are validated as in Ruby.
    let mut e = evaluation();
    argument_error(
        e.evaluation_with("correct", Some("Correct"), Some(f64::NAN), None)
            .map(|_| ()),
        "finite",
    );
    argument_error(e.evaluation("", Some("Correct")).map(|_| ()), "empty");
    // `evaluator(Object.new)`: `Evaluator` only accepts a model, a registry model, a Judge, or
    // an Agent, which the type system enforces.
}

// ---- evaluation/assertions_spec.rb ------------------------------------------------------------

fn upcasing(assertions: bool) -> Evaluation {
    let mut e = Evaluation::new();
    e.without_evaluator();
    e.perform(|i| {
        let out = upcase(i.input());
        async move { Ok(Outcome::Value(out)) }
    });
    if assertions {
        e.assertions(|a| Ok(a.assert(true, None)?));
    }
    e
}

// spec: evaluation/assertions_spec.rb:22
#[tokio::test]
async fn does_not_load_minitest_when_an_evaluation_has_no_assertions() {
    let cases = vec![Case::new("greeting", "hello").unwrap()];
    let report = upcasing(false)
        .run_with(RunOptions::default().dataset(cases))
        .await
        .unwrap();
    assert_eq!(report.first().unwrap().assertion_count(), 0);
}

// spec: evaluation/assertions_spec.rb:45
#[tokio::test]
async fn matches_only_minitest_assertion_failures() {
    // `Assertions::Failure === error`: only an assertion failure is recorded as one; any other
    // error is an execution error.
    let assertion: Failure = rust_llm::evaluation::AssertionFailure::new("Expected").into();
    assert!(matches!(assertion, Failure::Assertion(_)));
    let other: Failure = std::io::Error::other("boom").into();
    assert!(matches!(other, Failure::Error(_)));

    let mut e = upcasing(false);
    e.assertions(|_| Err(Failure::error("not an assertion")));
    let report = e
        .run_with(RunOptions::default().dataset(vec![Case::new("greeting", "hello").unwrap()]))
        .await
        .unwrap();
    let first = report.first().unwrap();
    assert!(first.assertion_failure().is_none());
    assert_eq!(first.error().unwrap().message, "not an assertion");
}

// ---- evaluation/result_spec.rb ----------------------------------------------------------------

fn probability(p: f64) -> Option<Measurement> {
    Some(Measurement::Answer(Answer::Probability { probability: p }))
}

// spec: evaluation/result_spec.rb:6
#[test]
fn preserves_boolean_verdicts_without_turning_a_failure_into_an_error() {
    assert!(
        Verdict::new("correct", Some(Measurement::Verdict(true)), None)
            .unwrap()
            .is_passed()
    );
    assert_eq!(
        Verdict::new("correct", Some(Measurement::Verdict(false)), None)
            .unwrap()
            .status(),
        Status::Failed
    );
    assert_eq!(
        Verdict::new("correct", None, None).unwrap().status(),
        Status::Unassessed
    );
}

// spec: evaluation/result_spec.rb:12
#[test]
fn requires_an_explicit_policy_before_counting_probabilities_as_passes() {
    let measurement = Verdict::new("correct", probability(0.8), None).unwrap();
    assert_eq!(measurement.status(), Status::Measured);
    assert!(!measurement.is_passed());
    assert!(
        Verdict::new("correct", probability(0.8), Some(0.8))
            .unwrap()
            .is_passed()
    );
    assert_eq!(
        Verdict::new("correct", probability(0.8), Some(0.9))
            .unwrap()
            .status(),
        Status::Failed
    );
}

// spec: evaluation/result_spec.rb:22
#[test]
fn preserves_score_distributions_and_their_native_scale() {
    let score = Answer::Score {
        score: 1.2,
        levels: vec!["Bad".into(), "Fair".into(), "Good".into()],
        probabilities: vec![(0, 0.1), (1, 0.6), (2, 0.3)],
        confidence: 0.5,
    };
    let result = Verdict::new("quality", Some(Measurement::Answer(score)), Some(1.0)).unwrap();
    assert!(result.is_passed());
    let h = result.to_h();
    assert_eq!(
        h["value"]["probabilities"],
        json!({ "0": 0.1, "1": 0.6, "2": 0.3 })
    );
    assert_eq!(h["value"]["score"], json!(1.2));
}

// spec: evaluation/result_spec.rb:32
#[test]
fn does_not_reinterpret_a_choice_confidence_as_quality() {
    let choice = Answer::Choice {
        choice: "wrong".into(),
        probabilities: vec![("wrong".into(), 1.0), ("right".into(), 0.0)],
        confidence: 1.0,
    };
    let result = Verdict::new("correct", Some(Measurement::Answer(choice.clone())), None).unwrap();
    assert_eq!(result.status(), Status::Measured);
    argument_error(
        Verdict::new("correct", Some(Measurement::Answer(choice)), Some(0.8)),
        "Minimum",
    );
}

// spec: evaluation/result_spec.rb:40
#[test]
fn serializes_errors_distinctly_from_verdicts_and_missing_evidence() {
    let error = rust_llm::Error::Api("Evaluator unavailable".into(), None);
    let result = Verdict::failed(
        "correct",
        TrialError::new(error.class_name(), error.to_string()),
    );
    assert_eq!(result.status(), Status::Error);
    assert_eq!(
        result.to_h()["error"],
        json!({ "class": "rust_llm::Error::Api", "message": "Evaluator unavailable" })
    );
    assert!(!result.is_passed());
    let _ = ErrorKind::Api;
}

// ---- evaluation_progress_spec.rb --------------------------------------------------------------

type Events = Arc<Mutex<Vec<(String, Map<String, Value>)>>>;

fn capture() -> (Arc<Config>, Events) {
    let events: Events = Default::default();
    let sink = events.clone();
    let mut config = Config::default();
    config.instrumenter = Some(Arc::new(
        move |name: &str, payload: &Map<String, Value>, _: Option<Duration>| {
            sink.lock()
                .unwrap()
                .push((name.to_string(), payload.clone()));
        },
    ));
    (Arc::new(config), events)
}

fn progress_cases() -> Vec<Case> {
    vec![
        case("greeting", "hello", "HELLO"),
        case("mismatch", "bye", "HELLO"),
        Case::new("broken", Value::Null).unwrap(),
    ]
}

/// The progress spec's `GreetingEvaluation`, reporting to `config`.
fn progress_evaluation(config: &Arc<Config>) -> Evaluation {
    let mut e = Evaluation::named("GreetingEvaluation");
    e.without_evaluator();
    e.with_config(config.clone());
    e.perform(|i| async move {
        let s = i
            .input()
            .as_str()
            .ok_or_else(|| Failure::error("undefined method 'upcase' for nil"))?;
        Ok(Outcome::Value(s.to_uppercase().into()))
    });
    e.assertions(|a| Ok(a.assert_equal(a.expected_output().clone(), a.output().clone())?));
    e
}

// spec: evaluation_progress_spec.rb:33
#[tokio::test]
async fn yields_completed_trials_after_cleanup_and_before_starting_the_next_case() {
    let (config, _) = capture();
    let timeline: Arc<Mutex<Vec<(String, Value)>>> = Default::default();
    let mut e = progress_evaluation(&config);
    let t = timeline.clone();
    e.setup(move |i| {
        t.lock().unwrap().push(("setup".into(), i.input().clone()));
        async { Ok(()) }
    });
    let t = timeline.clone();
    e.teardown(move |i| {
        t.lock()
            .unwrap()
            .push(("teardown".into(), i.input().clone()));
        async { Ok(()) }
    });
    let delivered: Arc<Mutex<Vec<Value>>> = Default::default();
    let (t, d) = (timeline.clone(), delivered.clone());
    let report = e
        .run_each(
            RunOptions::default()
                .dataset(progress_cases())
                .repetitions(2),
            move |trial| {
                t.lock()
                    .unwrap()
                    .push(("delivered".into(), trial.test_case().inputs().clone()));
                // `JSON.generate(trial.to_h)` does not raise.
                serde_json::to_string(&trial.to_h()).unwrap();
                d.lock().unwrap().push(trial.to_h());
                async { Ok(()) }
            },
        )
        .await
        .unwrap();
    let reported: Vec<Value> = report.iter().map(|t| t.to_h()).collect();
    let delivered = delivered.lock().unwrap().clone();
    assert_eq!(
        reported.iter().map(|t| &t["case"]).collect::<Vec<_>>(),
        delivered.iter().map(|t| &t["case"]).collect::<Vec<_>>()
    );
    assert_eq!(
        delivered
            .iter()
            .map(|t| t["status"].clone())
            .collect::<Vec<_>>(),
        ["passed", "passed", "failed", "failed", "error", "error"]
    );
    let expected: Vec<(String, Value)> = progress_cases()
        .iter()
        .flat_map(|c| {
            let i = c.inputs().clone();
            let once = [
                ("setup".to_string(), i.clone()),
                ("teardown".to_string(), i.clone()),
                ("delivered".to_string(), i),
            ];
            [once.clone(), once].concat()
        })
        .collect();
    assert_eq!(*timeline.lock().unwrap(), expected);
}

fn evaluation_events(events: &Events) -> Vec<(String, Map<String, Value>)> {
    events
        .lock()
        .unwrap()
        .iter()
        .filter(|(n, _)| n.starts_with("evaluation"))
        .cloned()
        .collect()
}

// spec: evaluation_progress_spec.rb:54
#[tokio::test]
async fn publishes_progress_and_the_final_report_through_the_existing_instrumenter() {
    let (config, events) = capture();
    let report = progress_evaluation(&config)
        .run_with(
            RunOptions::default()
                .dataset(progress_cases())
                .only(["mismatch", "broken"])
                .repetitions(2)
                .id("run-42"),
        )
        .await
        .unwrap();
    let evaluation = evaluation_events(&events);
    let names: Vec<&str> = evaluation.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(
        names,
        [
            "evaluation_trial.rust_llm",
            "evaluation_trial.rust_llm",
            "evaluation_trial.rust_llm",
            "evaluation_trial.rust_llm",
            "evaluation.rust_llm"
        ]
    );
    let trials: Vec<&Map<String, Value>> = evaluation[..4].iter().map(|(_, p)| p).collect();
    assert_eq!(
        trials
            .iter()
            .map(|p| p["completed"].clone())
            .collect::<Vec<_>>(),
        [1, 2, 3, 4]
    );
    assert_eq!(
        trials
            .iter()
            .map(|p| p["repetition"].clone())
            .collect::<Vec<_>>(),
        [1, 2, 1, 2]
    );
    assert_eq!(
        trials.iter().map(|p| p["case"].clone()).collect::<Vec<_>>(),
        ["mismatch", "mismatch", "broken", "broken"]
    );
    // `payload[:trial] == report.trials`: the payload carries each trial's `to_h` (durations differ
    // only by the clock, so compare everything else).
    for (payload, trial) in trials.iter().zip(report.iter()) {
        let mut sent = payload["trial"].clone();
        let mut kept = trial.to_h();
        sent["duration"] = Value::Null;
        kept["duration"] = Value::Null;
        assert_eq!(sent, kept);
    }
    for p in &trials {
        assert_eq!(p["evaluation_id"], "run-42");
        assert_eq!(p["evaluation_name"], "GreetingEvaluation");
        assert_eq!(p["total"], 4);
    }
    let last = &evaluation[4].1;
    assert_eq!(last["report"]["id"], report.id());
    assert_eq!(last["completed"], 4);
    assert_eq!(last["total"], 4);
    assert_eq!(
        last["started_at"],
        report
            .started_at()
            .to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
    );
    assert_eq!(report.id(), "run-42");
    let workflows: Vec<Map<String, Value>> = events
        .lock()
        .unwrap()
        .iter()
        .filter(|(n, _)| n == "workflow.rust_llm")
        .map(|(_, p)| p.clone())
        .collect();
    assert!(!workflows.is_empty());
    for w in workflows {
        assert_eq!(w["workflow_id"], report.id());
    }
}

// spec: evaluation_progress_spec.rb:71
#[tokio::test]
async fn reports_an_interrupted_run_without_reclassifying_the_completed_trial_as_an_error() {
    let (config, events) = capture();
    let cleanups: Arc<Mutex<Vec<Value>>> = Default::default();
    let mut e = progress_evaluation(&config);
    let c = cleanups.clone();
    e.teardown(move |i| {
        c.lock().unwrap().push(i.input().clone());
        async { Ok(()) }
    });
    let result = e
        .run_each(RunOptions::default().dataset(progress_cases()), |_| async {
            Err(rust_llm::Error::Tool("UI storage failed".into()))
        })
        .await;
    assert_eq!(result.unwrap_err().to_string(), "UI storage failed");
    assert_eq!(*cleanups.lock().unwrap(), [json!("hello")]);
    let evaluation = evaluation_events(&events);
    let names: Vec<&str> = evaluation.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, ["evaluation_trial.rust_llm", "evaluation.rust_llm"]);
    assert_eq!(evaluation[0].1["trial"]["status"], "passed");
    let last = &evaluation[1].1;
    assert_eq!(last["completed"], 1);
    assert_eq!(last["total"], 3);
    assert_eq!(last["exception"], json!(["Other", "UI storage failed"]));
    assert!(last.get("report").is_none());
}

// spec: evaluation_progress_spec.rb:93
#[tokio::test]
async fn keeps_nested_run_identities_and_progress_independent() {
    let (config, events) = capture();
    let e = progress_evaluation(&config);
    let inner_id: Arc<Mutex<Option<String>>> = Default::default();
    let (inner_e, slot) = (e.clone(), inner_id.clone());
    let outer = e
        .run_each(
            RunOptions::default()
                .dataset(progress_cases())
                .only(["greeting"])
                .id("outer"),
            move |_| {
                let (inner_e, slot) = (inner_e.clone(), slot.clone());
                async move {
                    let inner = inner_e
                        .run_with(
                            RunOptions::default()
                                .dataset(progress_cases())
                                .only(["greeting"])
                                .repetitions(2)
                                .id("inner"),
                        )
                        .await?;
                    *slot.lock().unwrap() = Some(inner.id().to_string());
                    Ok(())
                }
            },
        )
        .await
        .unwrap();
    let payloads: Vec<Map<String, Value>> = events
        .lock()
        .unwrap()
        .iter()
        .filter(|(n, _)| n == "evaluation.rust_llm")
        .map(|(_, p)| p.clone())
        .collect();
    assert_eq!(payloads.len(), 2);
    let inner = payloads
        .iter()
        .find(|p| p["evaluation_id"] == "inner")
        .unwrap();
    assert_eq!(
        (inner["completed"].clone(), inner["total"].clone()),
        (json!(2), json!(2))
    );
    assert_eq!(inner["report"]["id"], "inner");
    let outer_p = payloads
        .iter()
        .find(|p| p["evaluation_id"] == "outer")
        .unwrap();
    assert_eq!(
        (outer_p["completed"].clone(), outer_p["total"].clone()),
        (json!(1), json!(1))
    );
    assert_eq!(outer_p["report"]["id"], outer.id());
    assert_eq!(inner_id.lock().unwrap().as_deref(), Some("inner"));
}

// spec: evaluation_progress_spec.rb:106
#[tokio::test]
async fn generates_distinct_run_identities_and_normalizes_supplied_application_ids() {
    let (config, _) = capture();
    let e = progress_evaluation(&config);
    let options = || {
        RunOptions::default()
            .dataset(progress_cases())
            .only(["greeting"])
    };
    let first = e.run_with(options()).await.unwrap();
    let second = e.run_with(options()).await.unwrap();
    assert_ne!(first.id(), second.id());
    assert_eq!(e.run_with(options().id(42)).await.unwrap().id(), "42");
}

// spec: evaluation_progress_spec.rb:114
#[tokio::test]
async fn rejects_invalid_configuration_before_publishing_progress_or_yielding_trials() {
    let (config, events) = capture();
    let performed = Arc::new(AtomicUsize::new(0));
    let mut e = progress_evaluation(&config);
    let p = performed.clone();
    e.perform(move |_| {
        p.fetch_add(1, Ordering::SeqCst);
        async { Ok(Outcome::Value(Value::Null)) }
    });
    let unexpected = |_: &rust_llm::evaluation::Trial| async { panic!("unexpected progress") };
    argument_error(
        e.run_each(
            RunOptions::default().dataset(progress_cases()).id(""),
            unexpected,
        )
        .await,
        "id cannot be empty",
    );
    argument_error(
        e.run_each(
            RunOptions::default()
                .dataset(progress_cases())
                .only(["missing"]),
            unexpected,
        )
        .await,
        "Unknown evaluation cases",
    );
    assert!(events.lock().unwrap().is_empty());
    assert_eq!(performed.load(Ordering::SeqCst), 0);
}
