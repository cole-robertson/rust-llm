//! RubyLLM 2.1's `evaluation_integration_spec.rb`: running evaluations from the test suite
//! (`Evaluation::RSpec` / `Evaluation::Minitest` are [`assert_case`] and `evaluates!` here) and
//! from the `ruby_llm:eval` task ([`tasks::run_to`]).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use rust_llm::Config;
use rust_llm::evaluation::{Evaluation, Outcome, assert_case, tasks};
use serde_json::Value;

const DATASET: &str = "cases:
  - name: correct
    inputs: hello
    expected_output: HELLO
  - name: regression
    inputs: goodbye
    expected_output: BONJOUR
";

/// The spec's app directory: `app/evals/formatting_evaluation.yml` beside `app/prompts`.
fn app() -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "rust_llm_eval_app_{}",
        uuid::Uuid::new_v4().simple()
    ));
    std::fs::create_dir_all(dir.join("app/evals")).unwrap();
    std::fs::write(dir.join("app/evals/formatting_evaluation.yml"), DATASET).unwrap();
    dir
}

/// `class FormattingEvaluation < RubyLLM::Evaluation` with `evaluator false`, its dataset
/// discovered under `dir`.
fn formatting(dir: &Path) -> Evaluation {
    let mut config = Config::default();
    config.set(
        "prompt_root",
        dir.join("app/prompts").to_string_lossy().into_owned(),
    );
    let mut e = Evaluation::named("FormattingEvaluation");
    e.with_config(Arc::new(config));
    e.without_evaluator();
    e.perform(|i| {
        let out = i.input().as_str().unwrap_or_default().to_uppercase();
        async move { Ok(Outcome::Value(out.into())) }
    });
    e.assertions(|a| Ok(a.assert_equal(a.expected_output().clone(), a.output().clone())?));
    e
}

fn panic_message(result: std::thread::Result<()>) -> Option<String> {
    let payload = result.err()?;
    Some(
        payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default(),
    )
}

// spec: evaluation_integration_spec.rb:47
#[test]
fn runs_separate_test_cases_and_includes_the_actual_failed_assertion_in_its_report() {
    let dir = app();
    let evaluation = formatting(&dir);
    let cases = evaluation.cases(None, None).unwrap();
    // One test per case, as `evaluates described_class` defines one example per case.
    let failures: Vec<String> = cases
        .iter()
        .filter_map(|case| {
            let e = evaluation.clone();
            let name = case.name().to_string();
            panic_message(std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                move || assert_case(&e, &name),
            )))
        })
        .collect();
    assert_eq!(
        (cases.len(), failures.len()),
        (2, 1),
        "2 examples, 1 failure"
    );
    let report = &failures[0];
    for expected in ["regression", "BONJOUR", "GOODBYE"] {
        assert!(report.contains(expected), "{report}");
    }
    std::fs::remove_dir_all(dir).ok();
}

mod generated {
    use super::*;

    fn evaluation() -> Evaluation {
        let dir = app();
        formatting(&dir)
    }

    // spec: evaluation_integration_spec.rb:61 (Minitest's `evaluates` is the same per-case test as
    // RSpec's; the failing case and its report are asserted in the :47 port above)
    // `evaluates!` defines a `#[test]` per named case; only the passing one is generated here,
    // since a generated failure would fail this suite.
    rust_llm::evaluates!(evaluation(); formatting_correct => "correct");
}

// spec: evaluation_integration_spec.rb:75
#[tokio::test]
async fn discovers_conventional_evaluations_from_the_task_saves_reports_and_fails_on_regressions() {
    let dir = app();
    let out_dir = dir.join("tmp/evaluations");
    let mut out = Vec::new();
    let result = tasks::run_to(&[formatting(&dir)], None, None, 1, &out_dir, &mut out).await;
    let output = String::from_utf8(out).unwrap();
    match result {
        Err(rust_llm::Error::Argument(m)) => assert!(m.contains("Evaluations failed"), "{m}"),
        other => panic!("expected the task to fail, got {other:?}"),
    }
    for expected in ["correct [1]: passed", "regression [1]: failed", "BONJOUR"] {
        assert!(output.contains(expected), "{output}");
    }
    let report: Value = serde_json::from_str(
        &std::fs::read_to_string(out_dir.join("FormattingEvaluation.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(report["counts"]["passed"], 1);
    assert_eq!(report["counts"]["failed"], 1);
    std::fs::remove_dir_all(dir).ok();
}

// spec: evaluation_integration_spec.rb:89
#[tokio::test]
async fn selects_an_evaluation_and_case_from_the_task() {
    // `task(:environment)` is the Rails app boot; a Rust binary has already built its
    // evaluations before calling the task.
    let dir = app();
    let mut out = Vec::new();
    tasks::run_to(
        &[formatting(&dir)],
        Some("FormattingEvaluation"),
        Some("correct"),
        1,
        &dir.join("tmp/evaluations"),
        &mut out,
    )
    .await
    .unwrap();
    let output = String::from_utf8(out).unwrap();
    assert!(output.contains("1 passed, 0 failed"), "{output}");
    assert!(!output.contains("regression"), "{output}");
    std::fs::remove_dir_all(dir).ok();
}

// spec: evaluation_integration_spec.rb:102
#[tokio::test]
async fn rejects_a_misspelled_evaluation_instead_of_passing_an_empty_run() {
    let dir = app();
    let mut out = Vec::new();
    let result = tasks::run_to(
        &[formatting(&dir)],
        Some("MissingEvaluation"),
        None,
        1,
        &dir.join("tmp/evaluations"),
        &mut out,
    )
    .await;
    match result {
        Err(rust_llm::Error::Argument(m)) => {
            assert!(m.contains("No evaluations found"), "{m}");
            assert!(m.contains("MissingEvaluation"), "{m}");
        }
        other => panic!("expected an error, got {other:?}"),
    }
    std::fs::remove_dir_all(dir).ok();
}
