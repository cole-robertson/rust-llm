//! Evaluations: run your application against a dataset of cases and grade each result. Port of
//! `lib/ruby_llm/evaluation.rb` and `lib/ruby_llm/evaluation/*` (case, dataset, evaluator,
//! evidence, report, result, reviewer, runner, trial, assertions).
//!
//! ```ruby
//! class SupportEvaluation < RubyLLM::Evaluation
//!   def perform(input)
//!     SupportAgent.new.ask(input)
//!   end
//! end
//! SupportEvaluation.run.save("tmp/support.json")
//! ```
//!
//! ```no_run
//! # use rust_llm::evaluation::Evaluation;
//! # async fn run() -> rust_llm::Result<()> {
//! let mut support = Evaluation::named("SupportEvaluation");
//! support.perform(|case| async move {
//!     let mut chat = rust_llm::chat()?;
//!     Ok(chat.ask(case.input().as_str().unwrap_or_default()).await?.into())
//! });
//! support.run().await?.save("tmp/support.json")?;
//! # Ok(()) }
//! ```
//!
//! Ruby's class DSL becomes a builder: `evaluator`, `evaluation`, `dataset`, and `adapt` are
//! methods, and `perform`, `assertions`, `setup`, and `teardown` are closures. Every case and
//! repetition gets a fresh [`Instance`], as Ruby builds a fresh instance. A subclass is
//! [`Evaluation::subclass`]. Datasets default to `app/evals/<name>.{yml,yaml,json,jsonl}`
//! beside the prompt root. Without declared criteria, each case is graded for correctness
//! against its `expected_output` by the configured default model.
//!
//! Usage made while a trial runs is counted on the trial: the task's (setup, perform,
//! assertions, teardown) and the evaluators'. Ruby threads inherit the trial's scope; a task you
//! `tokio::spawn` needs [`in_current_scope`].
//!
//! Tests: [`assert_case`] and [`evaluates!`](crate::evaluates) run one case per `#[test]`, the
//! counterpart of `Evaluation::RSpec` and `Evaluation::Minitest`; [`tasks::run`] is the
//! `ruby_llm:eval` task.

mod assertions;
mod case;
mod dataset;
mod evaluator;
mod evidence;
mod result;

use std::any::Any;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use futures::future::BoxFuture;
use serde_json::{Map, Value, json};

pub use assertions::{AssertionFailure, Assertions, Failure, TrialError};
pub use case::Case;
pub use dataset::Dataset;
pub use evaluator::{Evaluator, REVIEWER_INSTRUCTIONS, Target};
pub use evidence::{Adapter, Evidence, EvidenceAttachment, Outcome};
pub use result::{Measurement, Result, Status};

use crate::config::Config;
use crate::cost::Cost;
use crate::error::Error;
use crate::message::UsageEntry;
use crate::tokens::Tokens;
use evaluator::Criterion;

type CrateResult<T> = crate::Result<T>;

/// `CORRECTNESS`: the built-in reference comparison.
const CORRECTNESS: &str = "The answer agrees with the expected output";

/// `perform(input)`.
pub type PerformFn = Arc<
    dyn Fn(Instance) -> BoxFuture<'static, std::result::Result<Outcome, Failure>> + Send + Sync,
>;
/// `setup` / `teardown`.
pub type HookFn =
    Arc<dyn Fn(Instance) -> BoxFuture<'static, std::result::Result<(), Failure>> + Send + Sync>;
/// `assertions`.
pub type AssertionsFn =
    Arc<dyn Fn(&mut Assertions<'_>) -> std::result::Result<(), Failure> + Send + Sync>;

/// One criterion as declared (`evaluation name, instructions, minimum:, evaluator:`).
#[derive(Clone)]
struct Definition {
    name: String,
    instructions: Option<String>,
    minimum: Option<f64>,
    evaluator: Option<Evaluator>,
}

/// `RubyLLM::Evaluation`: a reusable evaluation. Clone it to reuse a definition; use
/// [`Evaluation::subclass`] for Ruby's `Class.new(parent)`.
#[derive(Clone)]
pub struct Evaluation {
    name: Option<String>,
    dataset: Option<Dataset>,
    /// `None` is `evaluator false`.
    evaluator: Option<Evaluator>,
    definitions: Vec<Definition>,
    declared: Vec<String>,
    adapters: Vec<Adapter>,
    perform: Option<PerformFn>,
    assertions: Option<AssertionsFn>,
    setup: Option<HookFn>,
    teardown: Option<HookFn>,
    config: Option<Arc<Config>>,
}

impl Default for Evaluation {
    fn default() -> Evaluation {
        Evaluation::new()
    }
}

impl Evaluation {
    /// An anonymous evaluation (`Class.new(RubyLLM::Evaluation)`): it needs an explicit dataset.
    pub fn new() -> Evaluation {
        Evaluation {
            name: None,
            dataset: None,
            evaluator: Some(Evaluator::default()),
            definitions: Vec::new(),
            declared: Vec::new(),
            adapters: Vec::new(),
            perform: None,
            assertions: None,
            setup: None,
            teardown: None,
            config: None,
        }
    }

    /// `class SupportEvaluation < RubyLLM::Evaluation`: the name reports carry and dataset
    /// discovery uses (`SupportEvaluation` reads `app/evals/support_evaluation.yml`).
    pub fn named(name: impl Into<String>) -> Evaluation {
        let mut e = Evaluation::new();
        e.name = Some(name.into());
        e
    }

    /// `Class.new(parent)`: an anonymous child inheriting the dataset, evaluator, criteria,
    /// adapters, and hooks. The child may redeclare an inherited criterion; the parent is unchanged.
    pub fn subclass(&self) -> Evaluation {
        let mut child = self.clone();
        child.name = None;
        child.declared.clear();
        child
    }

    /// The evaluation's name, if it has one.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Use this configuration for events, the default evaluator model, and its requests.
    pub fn with_config(&mut self, config: Arc<Config>) -> &mut Self {
        self.config = Some(config);
        self
    }

    fn config(&self) -> Arc<Config> {
        self.config.clone().unwrap_or_else(crate::config)
    }

    /// `dataset(source)`: a path, cases, data, or [`Dataset::block`].
    pub fn dataset(&mut self, source: impl Into<Dataset>) -> &mut Self {
        self.dataset = Some(source.into());
        self
    }

    /// `dataset` with no argument: the configured source, if any.
    pub fn dataset_source(&self) -> Option<&Dataset> {
        self.dataset.as_ref()
    }

    /// `evaluator target, **options`: the default evaluator for every criterion.
    pub fn evaluator(&mut self, evaluator: Evaluator) -> &mut Self {
        self.evaluator = Some(evaluator);
        self
    }

    /// `evaluator false`: no model grading; only assertions run.
    pub fn without_evaluator(&mut self) -> &mut Self {
        self.evaluator = None;
        self
    }

    /// `evaluator`: the class evaluator, or `None` when grading is disabled.
    pub fn current_evaluator(&self) -> Option<&Evaluator> {
        self.evaluator.as_ref()
    }

    /// `evaluation name, instructions`: a semantic criterion. Declared criteria replace implicit
    /// correctness; `evaluation(:correctness)` without instructions includes the built-in
    /// comparison. `None` instructions set a policy on a question a Judge already defines.
    pub fn evaluation(
        &mut self,
        name: impl Into<String>,
        instructions: Option<&str>,
    ) -> CrateResult<&mut Self> {
        self.evaluation_with(name, instructions, None, None)
    }

    /// `evaluation name, instructions, minimum:, evaluator:`. A minimum applies to a native
    /// probability or score; `evaluator` overrides the class evaluator for this criterion.
    pub fn evaluation_with(
        &mut self,
        name: impl Into<String>,
        instructions: Option<&str>,
        minimum: Option<f64>,
        evaluator: Option<Evaluator>,
    ) -> CrateResult<&mut Self> {
        let name = name.into();
        if name.is_empty() {
            return Err(Error::Argument("An evaluation name cannot be empty".into()));
        }
        if minimum.is_some_and(|m| !m.is_finite()) {
            return Err(Error::Argument("Minimum must be a finite number".into()));
        }
        if self.declared.contains(&name) {
            return Err(Error::Argument(format!("Duplicate evaluation: {name}")));
        }
        self.declared.push(name.clone());
        let definition = Definition {
            name: name.clone(),
            instructions: instructions.map(str::to_string),
            minimum,
            evaluator,
        };
        match self.definitions.iter_mut().find(|d| d.name == name) {
            Some(existing) => *existing = definition,
            None => self.definitions.push(definition),
        }
        Ok(self)
    }

    /// `definitions`: `(name, instructions, minimum)` for every declared criterion.
    pub fn definitions(&self) -> Vec<(String, Option<String>, Option<f64>)> {
        self.definitions
            .iter()
            .map(|d| (d.name.clone(), d.instructions.clone(), d.minimum))
            .collect()
    }

    /// `adapt Type { |value| ... }`: converts a custom result into evaluation evidence. The
    /// original stays available as the trial's `result`.
    pub fn adapt<T: Any + Send + Sync>(
        &mut self,
        f: impl Fn(&T) -> Value + Send + Sync + 'static,
    ) -> &mut Self {
        self.adapters
            .push(Arc::new(move |value: &(dyn Any + Send + Sync)| {
                value.downcast_ref::<T>().map(&f)
            }));
        self
    }

    /// `adapters`: how many are declared.
    pub fn adapter_count(&self) -> usize {
        self.adapters.len()
    }

    /// `def perform(input)`: runs the application under evaluation.
    pub fn perform<F, Fut>(&mut self, f: F) -> &mut Self
    where
        F: Fn(Instance) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = std::result::Result<Outcome, Failure>> + Send + 'static,
    {
        self.perform = Some(Arc::new(move |i| Box::pin(f(i))));
        self
    }

    /// `def assertions`: runs after `perform`, with the Minitest-style assertion family.
    pub fn assertions(
        &mut self,
        f: impl Fn(&mut Assertions<'_>) -> std::result::Result<(), Failure> + Send + Sync + 'static,
    ) -> &mut Self {
        self.assertions = Some(Arc::new(f));
        self
    }

    /// Removes the `assertions` hook (`def assertions; end`).
    pub fn clear_assertions(&mut self) -> &mut Self {
        self.assertions = None;
        self
    }

    /// `def setup`: runs before `perform` on each fresh case instance.
    pub fn setup<F, Fut>(&mut self, f: F) -> &mut Self
    where
        F: Fn(Instance) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = std::result::Result<(), Failure>> + Send + 'static,
    {
        self.setup = Some(Arc::new(move |i| Box::pin(f(i))));
        self
    }

    /// `def teardown`: runs after each case, even when setup, perform, or an assertion fails.
    pub fn teardown<F, Fut>(&mut self, f: F) -> &mut Self
    where
        F: Fn(Instance) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = std::result::Result<(), Failure>> + Send + 'static,
    {
        self.teardown = Some(Arc::new(move |i| Box::pin(f(i))));
        self
    }

    /// `cases(dataset:, only:)`: the dataset cases, without running anything. `only` selects
    /// case names and fails when any is missing (or none is given).
    pub fn cases(
        &self,
        dataset: Option<&Dataset>,
        only: Option<&[String]>,
    ) -> CrateResult<Vec<Case>> {
        let source = dataset.or(self.dataset.as_ref());
        let loaded = dataset::load(source, self.name.as_deref(), &self.config())?;
        let Some(names) = only else {
            return Ok(loaded);
        };
        let missing: Vec<&str> = names
            .iter()
            .map(String::as_str)
            .filter(|n| !loaded.iter().any(|c| c.name() == *n))
            .collect();
        if !missing.is_empty() || names.is_empty() {
            return Err(Error::Argument(format!(
                "Unknown evaluation cases: {}",
                missing.join(", ")
            )));
        }
        Ok(loaded
            .into_iter()
            .filter(|c| names.iter().any(|n| n == c.name()))
            .collect())
    }

    /// `run`: every case once, with the configured dataset.
    pub async fn run(&self) -> CrateResult<Report> {
        self.run_with(RunOptions::default()).await
    }

    /// `run(dataset:, only:, repetitions:, id:)`.
    pub async fn run_with(&self, options: RunOptions) -> CrateResult<Report> {
        self.run_each(options, |_| async { Ok(()) }).await
    }

    /// `run(...) { |trial| ... }`: `on_trial` receives each completed trial before the next case
    /// starts, and the future it returns runs to completion first; its error ends the run and is
    /// returned. Configuration errors return before any case runs; task, assertion, and
    /// evaluator failures are recorded per trial.
    pub async fn run_each<F, Fut>(
        &self,
        options: RunOptions,
        mut on_trial: F,
    ) -> CrateResult<Report>
    where
        F: FnMut(&Trial) -> Fut + Send,
        Fut: Future<Output = CrateResult<()>> + Send,
    {
        let Some(perform) = self.perform.clone() else {
            return Err(Error::Argument(
                "Define perform(input) in your evaluation".into(),
            ));
        };
        if self.evaluator.is_none() && !self.definitions.is_empty() {
            return Err(Error::Argument(
                "Cannot declare semantic evaluations with evaluator false".into(),
            ));
        }
        if options.repetitions == 0 {
            return Err(Error::Argument(
                "Repetitions must be a positive Integer".into(),
            ));
        }
        let id = options
            .id
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        if id.is_empty() {
            return Err(Error::Argument(
                "An evaluation run id cannot be empty".into(),
            ));
        }
        let cases = self.cases(options.dataset.as_ref(), options.only.as_deref())?;
        let groups = self.evaluation_groups(&cases)?;
        let definitions: Vec<Value> = groups
            .iter()
            .flat_map(|(backend, criteria)| {
                criteria.iter().map(move |c| {
                    json!({
                        "name": c.name, "instructions": c.instructions, "minimum": c.minimum,
                        "evaluator": backend.description(),
                    })
                })
            })
            .collect();
        let started_at = chrono::Utc::now();
        let name = self
            .name
            .clone()
            .unwrap_or_else(|| "Anonymous evaluation".into());
        let total = cases.len() * options.repetitions;
        let config = self.config();
        let mut event =
            crate::instrumentation::Event::start(&config, "evaluation.rust_llm", || {
                crate::instrumentation::payload([
                    ("evaluation_id", id.clone().into()),
                    ("evaluation_name", name.clone().into()),
                    ("total", total.into()),
                    ("completed", 0.into()),
                    ("started_at", iso8601(&started_at).into()),
                ])
            });
        let run = RunContext {
            evaluation: self,
            perform,
            groups: &groups,
            id: &id,
            name: &name,
            total,
            config: &config,
        };
        let body = async {
            let mut trials = Vec::new();
            let mut completed = 0;
            for case in &cases {
                for repetition in 1..=options.repetitions {
                    let trial = run.trial(case, repetition, &mut completed).await;
                    if let Err(e) = on_trial(&trial).await {
                        return (trials, completed, Err(e));
                    }
                    trials.push(trial);
                }
            }
            (trials, completed, Ok(()))
        };
        let (trials, completed, outcome) = event.instrument(body).await;
        event.set("completed", || completed.into());
        if let Err(e) = outcome {
            event.finish(Some(&e));
            return Err(e);
        }
        let report = Report {
            name,
            id,
            trials,
            started_at,
            definitions,
        };
        event.set("report", || report.to_h());
        event.finish(None);
        Ok(report)
    }

    /// `evaluation_groups`: criteria grouped by the backend that grades them, prepared and
    /// validated before any case runs.
    fn evaluation_groups(&self, cases: &[Case]) -> CrateResult<Vec<(Evaluator, Vec<Criterion>)>> {
        let Some(class_evaluator) = &self.evaluator else {
            return Ok(Vec::new());
        };
        // `question_names.to_h { ... }.merge(definitions)`: a Judge's questions first, then
        // declarations, each replacing a question of the same name in place.
        let mut all: Vec<Definition> = class_evaluator
            .question_names()
            .into_iter()
            .map(|name| Definition {
                name,
                instructions: None,
                minimum: None,
                evaluator: None,
            })
            .collect();
        for d in &self.definitions {
            match all.iter_mut().find(|a| a.name == d.name) {
                Some(existing) => *existing = d.clone(),
                None => all.push(d.clone()),
            }
        }
        if all.is_empty() {
            all.push(Definition {
                name: "correctness".into(),
                instructions: None,
                minimum: None,
                evaluator: None,
            });
        }
        // `group_by { definition[:evaluator] || evaluator }`: each declared evaluator is its own
        // backend; the rest share the class evaluator, in first-appearance order.
        let mut groups: Vec<(Option<usize>, Evaluator, Vec<Criterion>)> = Vec::new();
        for (index, d) in all.iter().enumerate() {
            let key = d.evaluator.as_ref().map(|_| index);
            let backend = d
                .evaluator
                .clone()
                .unwrap_or_else(|| class_evaluator.clone());
            let criterion = prepare_criterion(&backend, d, cases)?;
            match groups.iter_mut().find(|(k, _, _)| *k == key) {
                Some((_, _, criteria)) => criteria.push(criterion),
                None => groups.push((key, backend, vec![criterion])),
            }
        }
        Ok(groups.into_iter().map(|(_, b, c)| (b, c)).collect())
    }
}

/// `prepare_criterion`.
fn prepare_criterion(
    backend: &Evaluator,
    d: &Definition,
    cases: &[Case],
) -> CrateResult<Criterion> {
    let existing = backend.question_names().contains(&d.name);
    if existing && d.instructions.is_some() {
        return Err(Error::Argument(format!(
            "Duplicate Judge question: {}",
            d.name
        )));
    }
    let criterion = |instructions: Option<String>| Criterion {
        name: d.name.clone(),
        instructions,
        minimum: d.minimum,
    };
    if existing || d.instructions.as_deref().is_some_and(|i| !i.is_empty()) {
        return Ok(criterion(d.instructions.clone()));
    }
    if d.name == "correctness" && d.instructions.is_none() {
        let missing: Vec<&str> = cases
            .iter()
            .filter(|c| !c.is_expected_output())
            .map(Case::name)
            .collect();
        if !missing.is_empty() {
            return Err(Error::Argument(format!(
                "Correctness requires expected_output for cases: {}",
                missing.join(", ")
            )));
        }
        return Ok(criterion(Some(CORRECTNESS.into())));
    }
    Err(Error::Argument(format!("Missing instructions: {}", d.name)))
}

/// `Time#iso8601`.
fn iso8601(t: &chrono::DateTime<chrono::Utc>) -> String {
    t.to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

/// Options for [`Evaluation::run_with`].
#[derive(Debug, Clone)]
pub struct RunOptions {
    /// `dataset:`: overrides discovery for this run.
    pub dataset: Option<Dataset>,
    /// `only:`: case names to run.
    pub only: Option<Vec<String>>,
    /// `repetitions:`: runs per case (at least 1).
    pub repetitions: usize,
    /// `id:`: correlates reports and events with an application record or job; a UUID by default.
    pub id: Option<String>,
}

impl Default for RunOptions {
    fn default() -> RunOptions {
        RunOptions {
            dataset: None,
            only: None,
            repetitions: 1,
            id: None,
        }
    }
}

impl RunOptions {
    pub fn dataset(mut self, dataset: impl Into<Dataset>) -> RunOptions {
        self.dataset = Some(dataset.into());
        self
    }

    pub fn only<S: Into<String>>(mut self, names: impl IntoIterator<Item = S>) -> RunOptions {
        self.only = Some(names.into_iter().map(Into::into).collect());
        self
    }

    pub fn repetitions(mut self, repetitions: usize) -> RunOptions {
        self.repetitions = repetitions;
        self
    }

    pub fn id(mut self, id: impl ToString) -> RunOptions {
        self.id = Some(id.to_string());
        self
    }
}

/// The fresh evaluation instance one case and repetition runs with: the case's input,
/// reference, and metadata, plus state the hooks share (Ruby's instance variables).
#[derive(Clone)]
pub struct Instance(Arc<InstanceInner>);

struct InstanceInner {
    input: Value,
    expected_output: Option<Value>,
    metadata: Map<String, Value>,
    state: Mutex<Map<String, Value>>,
}

impl Instance {
    fn new(case: &Case) -> Instance {
        Instance(Arc::new(InstanceInner {
            input: case.inputs().clone(),
            expected_output: case.expected_output().cloned(),
            metadata: case.metadata().clone(),
            state: Mutex::new(Map::new()),
        }))
    }

    /// `input`: a copy of the case's inputs, so changes never reach the dataset.
    pub fn input(&self) -> &Value {
        &self.0.input
    }

    /// `expected_output`: the reference answer, or `null` when none was supplied.
    pub fn expected_output(&self) -> &Value {
        self.0.expected_output.as_ref().unwrap_or(&Value::Null)
    }

    /// `metadata`.
    pub fn metadata(&self) -> &Map<String, Value> {
        &self.0.metadata
    }

    /// An instance variable (`@counter`).
    pub fn get(&self, key: &str) -> Option<Value> {
        self.0
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(key)
            .cloned()
    }

    /// `@counter = value`.
    pub fn set(&self, key: impl Into<String>, value: impl Into<Value>) {
        self.0
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(key.into(), value.into());
    }
}

// ---- usage accounting (`Runner#instrument`, `with_usage`) -------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Task,
    Evaluator,
}

#[derive(Default)]
struct Collector {
    usage: Mutex<CollectedUsage>,
}

#[derive(Default)]
struct CollectedUsage {
    task: Vec<UsageEntry>,
    evaluator: Vec<UsageEntry>,
    closed: bool,
}

impl Collector {
    fn lock(&self) -> std::sync::MutexGuard<'_, CollectedUsage> {
        self.usage
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

type Scope = Vec<(Arc<Collector>, Phase)>;

tokio::task_local! {
    static SCOPE: Scope;
}

/// `Runner#instrument` for `usage.ruby_llm`: every trial whose scope is active records the
/// attempt in its current phase. A finished trial records nothing more.
pub(crate) fn capture_usage(entry: &UsageEntry) {
    let _ = SCOPE.try_with(|scope| {
        for (collector, phase) in scope {
            let mut usage = collector.lock();
            if usage.closed {
                continue;
            }
            match phase {
                Phase::Task => usage.task.push(entry.clone()),
                Phase::Evaluator => usage.evaluator.push(entry.clone()),
            }
        }
    });
}

/// `with_usage(phase)`: `collector` counts usage in `phase` while `future` runs; enclosing
/// trials keep their own phase (a nested evaluation is task work to the outer one).
async fn with_phase<F: Future>(collector: &Arc<Collector>, phase: Phase, future: F) -> F::Output {
    let mut scope: Scope = SCOPE.try_with(Clone::clone).unwrap_or_default();
    scope.retain(|(c, _)| !Arc::ptr_eq(c, collector));
    scope.push((collector.clone(), phase));
    SCOPE.scope(scope, future).await
}

/// `future` counted by the trials running the current task, for a task spawned from `perform`
/// (Ruby's threads and fibers inherit the trial's usage scope; a `tokio::spawn` does not).
pub fn in_current_scope<F: Future>(future: F) -> impl Future<Output = F::Output> {
    let scope: Scope = SCOPE.try_with(Clone::clone).unwrap_or_default();
    SCOPE.scope(scope, future)
}

// ---- the runner (`evaluation/runner.rb`) ------------------------------------------------------

struct RunContext<'a> {
    evaluation: &'a Evaluation,
    perform: PerformFn,
    groups: &'a [(Evaluator, Vec<Criterion>)],
    id: &'a str,
    name: &'a str,
    total: usize,
    config: &'a Arc<Config>,
}

/// What `perform` produced, carried from the perform step to the evaluate step.
#[derive(Default)]
struct Steps {
    result: Option<Outcome>,
    evidence: Option<Evidence>,
    assertion_count: usize,
    assertion_failure: Option<AssertionFailure>,
    error: Option<TrialError>,
    evaluations: Vec<Result>,
}

impl RunContext<'_> {
    /// `run_trial`: one case and repetition inside `evaluation_trial.rust_llm`.
    async fn trial(&self, case: &Case, repetition: usize, completed: &mut usize) -> Trial {
        let mut event =
            crate::instrumentation::Event::start(self.config, "evaluation_trial.rust_llm", || {
                crate::instrumentation::payload([
                    ("evaluation_id", self.id.into()),
                    ("evaluation_name", self.name.into()),
                    ("total", self.total.into()),
                    ("case", case.name().into()),
                    ("repetition", repetition.into()),
                ])
            });
        // A `dyn` future, so callers (and runs nested inside `perform`) do not re-prove `Send`
        // through the whole provider stack.
        let run: BoxFuture<'_, Trial> = Box::pin(self.run_case(case, repetition));
        let trial = event.instrument(run).await;
        *completed += 1;
        event.set("completed", || (*completed).into());
        event.set("trial", || trial.to_h());
        event.finish(None);
        trial
    }

    /// `Runner#run`.
    async fn run_case(&self, case: &Case, repetition: usize) -> Trial {
        let collector = Arc::new(Collector::default());
        let started = Instant::now();
        let instance = Instance::new(case);
        let mut steps = Steps::default();
        with_phase(&collector, Phase::Task, async {
            self.run_steps(case, repetition, &instance, &collector, &mut steps)
                .await;
            if let Some(teardown) = &self.evaluation.teardown
                && let Err(e) = teardown(instance.clone()).await
            {
                steps.error.get_or_insert(failure_error(e));
            }
        })
        .await;
        let (task_usage, evaluator_usage) = {
            let mut usage = collector.lock();
            usage.closed = true;
            (
                std::mem::take(&mut usage.task),
                std::mem::take(&mut usage.evaluator),
            )
        };
        let duration = started.elapsed().as_secs_f64();
        Trial {
            test_case: case.clone(),
            repetition,
            output: steps.evidence.as_ref().map(|e| e.output.clone()),
            evidence: steps.evidence.as_ref().map(|e| e.data.clone()),
            result: steps.result,
            evaluations: steps.evaluations,
            assertion_count: steps.assertion_count,
            assertion_failure: steps.assertion_failure,
            error: steps.error,
            duration,
            task_usage,
            evaluator_usage,
        }
    }

    /// `run_steps`: `perform` and `evaluate` as steps of a workflow named after the evaluation.
    async fn run_steps(
        &self,
        case: &Case,
        repetition: usize,
        instance: &Instance,
        collector: &Arc<Collector>,
        steps: &mut Steps,
    ) {
        let name = self.evaluation.name.as_deref().unwrap_or("Evaluation");
        let metadata = json!({ "case": case.name(), "repetition": repetition });
        let Ok(workflow) = crate::workflow::Workflow::new(
            name,
            Some(self.id),
            Some(metadata),
            self.config.clone(),
        ) else {
            return;
        };
        let _ = workflow
            .run(|wf| async move {
                wf.step("perform", None, async {
                    self.perform(instance, steps).await;
                    Ok(())
                })
                .await?;
                if steps.evidence.is_some() {
                    with_phase(
                        collector,
                        Phase::Evaluator,
                        wf.step("evaluate", None, async {
                            self.evaluate(case, steps).await;
                            Ok(())
                        }),
                    )
                    .await?;
                }
                Ok(())
            })
            .await;
    }

    /// `perform`: setup, the application, its evidence, then assertions.
    async fn perform(&self, instance: &Instance, steps: &mut Steps) {
        let outcome = async {
            if let Some(setup) = &self.evaluation.setup {
                setup(instance.clone()).await?;
            }
            let result = (self.perform)(instance.clone()).await?;
            let evidence = Evidence::with_adapters(&result, &self.evaluation.adapters);
            steps.result = Some(result);
            steps.evidence = Some(evidence?);
            if let (Some(assertions), Some(evidence), Some(result)) =
                (&self.evaluation.assertions, &steps.evidence, &steps.result)
            {
                let mut context = Assertions {
                    input: instance.input(),
                    expected_output: instance.0.expected_output.as_ref(),
                    metadata: instance.metadata(),
                    result,
                    output: &evidence.output,
                    messages: &evidence.messages,
                    tool_calls: &evidence.tool_calls,
                    count: 0,
                };
                let outcome = assertions(&mut context);
                steps.assertion_count = context.count;
                outcome?;
            }
            Ok::<(), Failure>(())
        }
        .await;
        match outcome {
            Ok(()) => {}
            Err(Failure::Assertion(f)) => steps.assertion_failure = Some(f),
            Err(Failure::Error(e)) => steps.error = Some(e),
        }
    }

    /// `evaluate`: each backend grades its criteria; a failing backend records an error for each.
    async fn evaluate(&self, case: &Case, steps: &mut Steps) {
        let Some(evidence) = &steps.evidence else {
            return;
        };
        // `test_case.to_h.except(:name).merge(actual:)`, keeping Ruby's key order.
        let mut data = case.to_h();
        if let Some(h) = data.as_object_mut() {
            h.shift_remove("name");
            h.insert("actual".into(), evidence.data.clone());
        }
        let attachments: Vec<crate::Attachment> = evidence
            .attachments
            .iter()
            .map(EvidenceAttachment::to_attachment)
            .collect();
        for (backend, criteria) in self.groups {
            let call: BoxFuture<'_, CrateResult<Vec<Result>>> =
                Box::pin(backend.call(&data, criteria, &attachments, self.config));
            match call.await {
                Ok(results) => steps.evaluations.extend(results),
                Err(e) => steps.evaluations.extend(
                    criteria
                        .iter()
                        .map(|c| Result::failed(c.name.clone(), TrialError::from_error(&e))),
                ),
            }
        }
    }
}

fn failure_error(failure: Failure) -> TrialError {
    match failure {
        Failure::Error(e) => e,
        Failure::Assertion(f) => TrialError::new("Minitest::Assertion", f.message()),
    }
}

// ---- Trial and Report -------------------------------------------------------------------------

fn tokens_of(entries: &[UsageEntry]) -> Tokens {
    Tokens::aggregate(entries.iter().map(|e| &e.tokens))
}

fn cost_of(entries: &[UsageEntry]) -> Cost {
    Cost::aggregate(
        entries.iter().map(|e| &e.cost),
        entries.iter().all(UsageEntry::cost_available),
    )
}

/// `RubyLLM::Evaluation::Trial`: the recorded outcome of one case and repetition.
#[derive(Debug)]
pub struct Trial {
    test_case: Case,
    repetition: usize,
    result: Option<Outcome>,
    output: Option<Value>,
    evidence: Option<Value>,
    evaluations: Vec<Result>,
    assertion_count: usize,
    assertion_failure: Option<AssertionFailure>,
    error: Option<TrialError>,
    duration: f64,
    task_usage: Vec<UsageEntry>,
    evaluator_usage: Vec<UsageEntry>,
}

impl Trial {
    /// The case whose inputs were executed.
    pub fn test_case(&self) -> &Case {
        &self.test_case
    }

    /// The one-based repetition number.
    pub fn repetition(&self) -> usize {
        self.repetition
    }

    /// The value `perform` returned, when it completed.
    pub fn result(&self) -> Option<&Outcome> {
        self.result.as_ref()
    }

    /// The primary output extracted from the returned value (`null` when there was none).
    pub fn output(&self) -> &Value {
        self.output.as_ref().unwrap_or(&Value::Null)
    }

    /// The evidence supplied to evaluators.
    pub fn evidence(&self) -> Option<&Value> {
        self.evidence.as_ref()
    }

    /// Each criterion's [`Result`].
    pub fn evaluations(&self) -> &[Result] {
        &self.evaluations
    }

    /// How many assertions ran.
    pub fn assertion_count(&self) -> usize {
        self.assertion_count
    }

    pub fn assertion_failure(&self) -> Option<&AssertionFailure> {
        self.assertion_failure.as_ref()
    }

    /// An execution or evidence-conversion error.
    pub fn error(&self) -> Option<&TrialError> {
        self.error.as_ref()
    }

    /// Elapsed wall-clock seconds, including evaluators.
    pub fn duration(&self) -> f64 {
        self.duration
    }

    fn entries(&self) -> Vec<UsageEntry> {
        self.task_usage
            .iter()
            .chain(&self.evaluator_usage)
            .cloned()
            .collect()
    }

    /// Every provider attempt made during this trial.
    pub fn tokens(&self) -> Tokens {
        tokens_of(&self.entries())
    }

    /// The trial's cost, task and evaluators together; `None` total when any attempt is unpriced.
    pub fn cost(&self) -> Cost {
        cost_of(&self.entries())
    }

    /// The task's tokens, including setup, assertions, and teardown requests.
    pub fn task_tokens(&self) -> Tokens {
        tokens_of(&self.task_usage)
    }

    /// The tokens used by the evaluators and their tools.
    pub fn evaluator_tokens(&self) -> Tokens {
        tokens_of(&self.evaluator_usage)
    }

    pub fn task_cost(&self) -> Cost {
        cost_of(&self.task_usage)
    }

    pub fn evaluator_cost(&self) -> Cost {
        cost_of(&self.evaluator_usage)
    }

    /// `status`: empty trials are never passes.
    pub fn status(&self) -> Status {
        let mut statuses: Vec<Status> = self.evaluations.iter().map(Result::status).collect();
        if self.error.is_some() {
            statuses.push(Status::Error);
        }
        if self.assertion_failure.is_some() {
            statuses.push(Status::Failed);
        }
        if self.evaluations.is_empty() && self.assertion_count == 0 {
            statuses.push(Status::Measured);
        }
        [
            Status::Error,
            Status::Failed,
            Status::Unassessed,
            Status::Measured,
        ]
        .into_iter()
        .find(|s| statuses.contains(s))
        .unwrap_or(Status::Passed)
    }

    pub fn is_passed(&self) -> bool {
        self.status() == Status::Passed
    }

    /// `to_h`: the portable trial record, without the live application object.
    pub fn to_h(&self) -> Value {
        json!({
            "case": self.test_case.to_h(),
            "repetition": self.repetition,
            "status": self.status().as_str(),
            "evidence": self.evidence,
            "evaluations": self.evaluations.iter().map(Result::to_h).collect::<Vec<_>>(),
            "assertion_count": self.assertion_count,
            "assertion_failure": self.assertion_failure.as_ref().map(AssertionFailure::message),
            "duration": self.duration,
            "error": self.error.as_ref().map(TrialError::to_h),
            "tokens": crate::instrumentation::tokens_h(&self.tokens()),
            "cost": crate::instrumentation::cost_h(&self.cost()),
            "task_tokens": crate::instrumentation::tokens_h(&self.task_tokens()),
            "evaluator_tokens": crate::instrumentation::tokens_h(&self.evaluator_tokens()),
            "task_cost": crate::instrumentation::cost_h(&self.task_cost()),
            "evaluator_cost": crate::instrumentation::cost_h(&self.evaluator_cost()),
        })
    }
}

/// `RubyLLM::Evaluation::Report`: a run's trials and summary. Persist with [`Report::save`].
#[derive(Debug)]
pub struct Report {
    name: String,
    id: String,
    trials: Vec<Trial>,
    started_at: chrono::DateTime<chrono::Utc>,
    definitions: Vec<Value>,
}

impl Report {
    /// The evaluation name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The unique run identifier.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// The trials, in dataset and repetition order.
    pub fn trials(&self) -> &[Trial] {
        &self.trials
    }

    /// `first`.
    pub fn first(&self) -> Option<&Trial> {
        self.trials.first()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Trial> {
        self.trials.iter()
    }

    /// The UTC time the run started.
    pub fn started_at(&self) -> chrono::DateTime<chrono::Utc> {
        self.started_at
    }

    /// The criterion definitions and evaluator identities captured for this run.
    pub fn definitions(&self) -> &[Value] {
        &self.definitions
    }

    fn entries(&self) -> Vec<UsageEntry> {
        self.trials.iter().flat_map(Trial::entries).collect()
    }

    /// Tokens across every case and repetition.
    pub fn tokens(&self) -> Tokens {
        tokens_of(&self.entries())
    }

    /// Cost across every case and repetition.
    pub fn cost(&self) -> Cost {
        cost_of(&self.entries())
    }

    /// `counts`: trials per status, in `Status::ALL` order.
    pub fn counts(&self) -> Vec<(Status, usize)> {
        Status::ALL
            .iter()
            .map(|s| (*s, self.trials.iter().filter(|t| t.status() == *s).count()))
            .collect()
    }

    /// `counts[status]`.
    pub fn count(&self, status: Status) -> usize {
        self.trials.iter().filter(|t| t.status() == status).count()
    }

    /// The passing fraction of all trials, including errors and ungraded measurements.
    pub fn pass_rate(&self) -> Option<f64> {
        (!self.trials.is_empty())
            .then(|| self.count(Status::Passed) as f64 / self.trials.len() as f64)
    }

    /// `passed?`: the run is nonempty and every trial passed.
    pub fn is_passed(&self) -> bool {
        !self.trials.is_empty() && self.trials.iter().all(Trial::is_passed)
    }

    /// `to_h`: all trial evidence, measurements, and run identity.
    pub fn to_h(&self) -> Value {
        json!({
            "id": self.id,
            "name": self.name,
            "started_at": iso8601(&self.started_at),
            "definitions": self.definitions,
            "counts": self.counts().iter().map(|(s, n)| (s.as_str().to_string(), json!(n))).collect::<Map<_, _>>(),
            "pass_rate": self.pass_rate(),
            "tokens": crate::instrumentation::tokens_h(&self.tokens()),
            "cost": crate::instrumentation::cost_h(&self.cost()),
            "trials": self.trials.iter().map(Trial::to_h).collect::<Vec<_>>(),
        })
    }

    /// `save(path)`: writes a JSON report and returns the path.
    pub fn save(&self, path: impl AsRef<Path>) -> CrateResult<PathBuf> {
        let path = path.as_ref();
        std::fs::write(path, serde_json::to_string_pretty(&self.to_h())?)?;
        Ok(path.to_path_buf())
    }

    fn trial_description(trial: &Trial) -> String {
        let details: Vec<String> = trial
            .evaluations
            .iter()
            .map(|e| format!("{}={}", e.name(), e.status()))
            .collect();
        let mut heading = format!(
            "{} [{}]: {}",
            trial.test_case.name(),
            trial.repetition,
            trial.status()
        );
        if !details.is_empty() {
            heading.push_str(&format!(" ({})", details.join(", ")));
        }
        let mut failures: Vec<String> = Vec::new();
        if let Some(e) = &trial.error {
            failures.push(e.message.clone());
        }
        if let Some(f) = &trial.assertion_failure {
            failures.push(f.message().to_string());
        }
        failures.extend(
            trial
                .evaluations
                .iter()
                .filter(|e| !e.is_passed())
                .map(|e| format!("{}: {}", e.name(), e.failure_message())),
        );
        std::iter::once(heading)
            .chain(failures.into_iter().map(|f| format!("  {f}")))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

impl<'a> IntoIterator for &'a Report {
    type Item = &'a Trial;
    type IntoIter = std::slice::Iter<'a, Trial>;
    fn into_iter(self) -> Self::IntoIter {
        self.trials.iter()
    }
}

/// `to_s`: a readable per-case report followed by counts.
impl std::fmt::Display for Report {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut lines = vec![self.name.clone()];
        lines.extend(self.trials.iter().map(Report::trial_description));
        lines.push(
            self.counts()
                .iter()
                .map(|(s, n)| format!("{n} {s}"))
                .collect::<Vec<_>>()
                .join(", "),
        );
        f.write_str(&lines.join("\n"))
    }
}

// ---- test integration (`Evaluation::RSpec`, `Evaluation::Minitest`) ---------------------------

/// `evaluates evaluation`: runs one case the way [`Evaluation::run_with`] does and panics with
/// the report when it does not pass, for use in a `#[test]`. It runs its own Tokio runtime, so
/// call it from a plain `#[test]`, not from inside an async runtime.
pub fn assert_case(evaluation: &Evaluation, case: &str) {
    assert_case_with(evaluation, case, RunOptions::default());
}

/// [`assert_case`] with run options (`evaluates evaluation, dataset:, repetitions:`).
pub fn assert_case_with(evaluation: &Evaluation, case: &str, options: RunOptions) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap_or_else(|e| panic!("could not start a runtime for the evaluation: {e}"));
    let report = runtime.block_on(async {
        let selected = evaluation.cases(options.dataset.as_ref(), Some(&[case.to_string()]))?;
        evaluation
            .run_with(RunOptions {
                dataset: Some(Dataset::Cases(selected)),
                only: None,
                ..options
            })
            .await
    });
    match report {
        Ok(report) => assert!(report.is_passed(), "{report}"),
        Err(e) => panic!("{e}"),
    }
}

/// One `#[test]` per case: `evaluates!(formatting_evaluation(); correct => "correct", regression => "regression");`
/// where the first expression builds the [`Evaluation`]. Each failing test prints the report.
#[macro_export]
macro_rules! evaluates {
    ($evaluation:expr; $($test:ident => $case:expr),+ $(,)?) => {
        $(
            #[test]
            fn $test() {
                $crate::evaluation::assert_case(&$evaluation, $case);
            }
        )+
    };
}

/// `rake ruby_llm:eval`: run registered evaluations from a binary or task.
pub mod tasks {
    use std::io::Write;
    use std::path::PathBuf;

    use super::{Evaluation, Report, RunOptions};
    use crate::error::{Error, Result};

    /// `Tasks::Evaluations.run(name, only:)`: runs every evaluation in `evaluations` (or the one
    /// named `name`, optionally only case `only`), `EVAL_REPETITIONS` times each, saves each
    /// report to `EVAL_OUTPUT` (default `tmp/evaluations`) as `<Name>.json`, and prints it to
    /// `out`. Fails with "Evaluations failed" when any did not pass, after saving them all.
    pub async fn run(
        evaluations: &[Evaluation],
        name: Option<&str>,
        only: Option<&str>,
        out: &mut (dyn Write + Send),
    ) -> Result<Vec<Report>> {
        let repetitions = match std::env::var("EVAL_REPETITIONS") {
            Ok(v) => v
                .parse()
                .map_err(|_| Error::Argument(format!("invalid value for Integer(): {v:?}")))?,
            Err(_) => 1,
        };
        let root = PathBuf::from(
            std::env::var("EVAL_OUTPUT").unwrap_or_else(|_| "tmp/evaluations".into()),
        );
        run_to(evaluations, name, only, repetitions, &root, out).await
    }

    /// [`run`] with the repetitions and output directory given instead of read from the
    /// environment.
    pub async fn run_to(
        evaluations: &[Evaluation],
        name: Option<&str>,
        only: Option<&str>,
        repetitions: usize,
        root: &std::path::Path,
        out: &mut (dyn Write + Send),
    ) -> Result<Vec<Report>> {
        let selected: Vec<&Evaluation> = evaluations
            .iter()
            .filter(|e| name.is_none_or(|n| e.name() == Some(n)))
            .collect();
        if selected.is_empty() {
            let matching = name.map(|n| format!(" matching {n}")).unwrap_or_default();
            return Err(Error::Argument(format!("No evaluations found{matching}")));
        }
        let mut reports = Vec::new();
        for evaluation in selected {
            let mut options = RunOptions::default().repetitions(repetitions);
            if let Some(case) = only {
                options = options.only([case]);
            }
            let report = evaluation.run_with(options).await?;
            let path = root.join(format!(
                "{}.json",
                evaluation.name().unwrap_or("Evaluation").replace("::", "/")
            ));
            if let Some(dir) = path.parent() {
                std::fs::create_dir_all(dir)?;
            }
            report.save(&path)?;
            writeln!(out, "{report}")?;
            writeln!(out, "Report: {}", path.display())?;
            reports.push(report);
        }
        if !reports.iter().all(Report::is_passed) {
            return Err(Error::Argument("Evaluations failed".into()));
        }
        Ok(reports)
    }
}
