//! Port of `lib/ruby_llm/support/instrumentation.rb`, `RubyLLM.instrument`, and
//! `config.instrumenter`.
//!
//! ```ruby
//! RubyLLM.configure { |config| config.instrumenter = ActiveSupport::Notifications }
//! ActiveSupport::Notifications.subscribe("chat.ruby_llm") { |event| ... }
//! ```
//!
//! ```ignore
//! rust_llm::configure(|config| {
//!     config.instrumenter = Some(Arc::new(|name: &str, payload: &Map<String, Value>, duration: Option<Duration>| {
//!         tracing::info!(name, ?duration, %Value::Object(payload.clone()));
//!     }));
//! });
//! ```
//!
//! Every event keeps RubyLLM's name with the `.ruby_llm` suffix swapped for `.rust_llm`
//! (`chat.ruby_llm` is `chat.rust_llm`). Payloads carry the same keys as Ruby's, as JSON: value
//! objects become their `to_h` (`tokens`, `cost`, `response`), and Ruby-only objects (`chat`,
//! `tool`, `model_info`) are left out. An event fires when its work finishes, so nested events
//! arrive before the event around them, the order Ruby's `CaptureInstrumenter` records. A failed
//! block adds `exception: [kind, message]`, like `ActiveSupport::Notifications`.
//!
//! Every event also runs inside a `tracing` span named `rust_llm` with an `event` field, and logs
//! its duration at debug level, so a `tracing` subscriber sees the work without an instrumenter.
//! With [`crate::open_telemetry::enable`], traced events also get an OpenTelemetry span
//! (`Support::Instrumentation.subscribe`), which [`Event::instrument`] makes current.

use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{Map, Value, json};

use crate::config::Config;
use crate::cost::Cost;
use crate::error::Error;
use crate::message::UsageEntry;
use crate::tokens::Tokens;

/// `config.instrumenter`: receives every event with its payload. `duration` is the time the
/// instrumented work took, or `None` for events that mark a moment (`usage.rust_llm`).
pub trait Instrumenter: Send + Sync {
    fn instrument(&self, name: &str, payload: &Map<String, Value>, duration: Option<Duration>);
}

impl<F> Instrumenter for F
where
    F: Fn(&str, &Map<String, Value>, Option<Duration>) + Send + Sync,
{
    fn instrument(&self, name: &str, payload: &Map<String, Value>, duration: Option<Duration>) {
        self(name, payload, duration)
    }
}

impl std::fmt::Debug for dyn Instrumenter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Instrumenter")
    }
}

tokio::task_local! {
    static WORKFLOW: Arc<Map<String, Value>>;
}

/// `Instrumentation.current_workflow`: the workflow and step identity of the running task.
pub fn current_workflow() -> Option<Map<String, Value>> {
    WORKFLOW.try_with(|context| (**context).clone()).ok()
}

/// `Instrumentation.with_workflow`: runs `future` with `context` as the current workflow. The
/// previous context comes back when it finishes, fails, or is dropped.
pub(crate) async fn with_workflow<F: Future>(context: Map<String, Value>, future: F) -> F::Output {
    WORKFLOW.scope(Arc::new(context), future).await
}

/// One instrumented block: `RubyLLM.instrument(name, payload) { |event| ... }`. The payload is
/// only built when an instrumenter is configured or OpenTelemetry traces the event.
pub struct Event {
    name: String,
    payload: Option<Map<String, Value>>,
    config: Arc<Config>,
    started: Instant,
    span: tracing::Span,
    /// The OpenTelemetry span (`OpenTelemetry#instrument`), ended on finish or drop.
    otel: crate::open_telemetry::Span,
}

impl Event {
    /// Starts an event. `payload` runs only when `config.instrumenter` is set or OpenTelemetry
    /// traces the event; the current workflow context is merged over it, as Ruby merges it at the
    /// start of the block.
    pub fn start(
        config: &Arc<Config>,
        name: &str,
        payload: impl FnOnce() -> Map<String, Value>,
    ) -> Event {
        let payload = (config.instrumenter.is_some() || crate::open_telemetry::is_traced(name))
            .then(|| {
                let mut payload = payload();
                payload.extend(current_workflow().unwrap_or_default());
                payload
            });
        let otel = crate::open_telemetry::Span::start(name, payload.as_ref());
        Event {
            name: name.to_string(),
            payload,
            config: config.clone(),
            started: Instant::now(),
            span: tracing::debug_span!("rust_llm", event = name),
            otel,
        }
    }

    /// Whether this event's payload is built (an instrumenter or OpenTelemetry will read it).
    pub fn is_enabled(&self) -> bool {
        self.payload.is_some()
    }

    /// `event[key] = value`; `value` runs only when the event is enabled.
    pub fn set(&mut self, key: &str, value: impl FnOnce() -> Value) {
        if let Some(payload) = &mut self.payload {
            payload.insert(key.to_string(), value());
        }
    }

    /// The `tracing` span the instrumented work should run in.
    pub fn span(&self) -> tracing::Span {
        self.span.clone()
    }

    /// Runs `future` as the instrumented work: inside [`Event::span`], and with this event's
    /// OpenTelemetry span as the current context on every poll (`Trace.with_span`).
    pub fn instrument<F: Future>(&self, future: F) -> impl Future<Output = F::Output> + use<F> {
        self.otel
            .instrument(tracing::Instrument::instrument(future, self.span.clone()))
    }

    /// Ends the block and delivers the event, with `exception` added when it failed.
    pub fn finish(mut self, error: Option<&Error>) {
        if let Some(e) = error {
            self.set("exception", || {
                json!([format!("{:?}", e.kind()), e.to_string()])
            });
        }
        self.otel.finish(self.payload.as_ref(), error);
        let elapsed = self.started.elapsed();
        self.emit(Some(elapsed));
    }

    fn emit(self, duration: Option<Duration>) {
        tracing::debug!(parent: &self.span, event = %self.name, duration_ms = duration.map(|d| d.as_secs_f64() * 1000.0), "rust_llm event");
        if let (Some(instrumenter), Some(payload)) = (&self.config.instrumenter, &self.payload) {
            Instrumenter::instrument(&**instrumenter, &self.name, payload, duration);
        }
    }
}

/// `RubyLLM.instrument(name, payload) { ... }`: runs `future` as an instrumented block and returns
/// its output. For events of your own that should carry workflow context.
pub async fn instrument<T>(
    config: &Arc<Config>,
    name: &str,
    payload: Map<String, Value>,
    future: impl Future<Output = crate::Result<T>>,
) -> crate::Result<T> {
    let event = Event::start(config, name, || payload);
    let result = event.instrument(future).await;
    event.finish(result.as_ref().err());
    result
}

/// `RubyLLM.instrument(name, payload)` without a block: an event that marks a moment.
pub fn instrument_event(config: &Arc<Config>, name: &str, payload: Map<String, Value>) {
    Event::start(config, name, || payload).emit(None);
}

/// `Accounting::Usage.instrument`: `usage.rust_llm` for one finished provider attempt.
pub(crate) fn usage(config: &Arc<Config>, entry: &UsageEntry) {
    crate::evaluation::capture_usage(entry);
    if config.instrumenter.is_none() {
        return;
    }
    let mut payload = Map::new();
    payload.insert("operation".into(), entry.operation.as_str().into());
    payload.insert("provider".into(), entry.provider.clone().into());
    payload.insert("model".into(), entry.model.clone().into());
    payload.insert("status".into(), entry.status.as_str().into());
    payload.insert("tokens".into(), tokens_h(&entry.tokens));
    payload.insert("cost".into(), cost_h(&entry.cost));
    payload.insert(
        "owner".into(),
        entry
            .owner
            .as_ref()
            .map_or(Value::Null, crate::accounting::UsageOwner::to_value),
    );
    instrument_event(config, "usage.rust_llm", payload);
}

/// `metadata:`: per-call data for the event payload, never sent to the provider.
pub(crate) fn metadata(metadata: &Option<Value>) -> Value {
    metadata.clone().unwrap_or(Value::Null)
}

/// `Tokens#to_h`.
pub(crate) fn tokens_h(tokens: &Tokens) -> Value {
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
    if let Some(s) = &tokens.server_tool_use {
        h.insert("server_tool_use".into(), Value::Object(s.clone()));
    }
    Value::Object(h)
}

/// `Cost#to_h`.
pub(crate) fn cost_h(cost: &Cost) -> Value {
    let mut h = Map::new();
    for (key, value) in [
        ("input", cost.input),
        ("output", cost.output),
        ("cache_read", cost.cache_read),
        ("cache_write", cost.cache_write),
        ("thinking", cost.thinking),
        ("total", cost.total()),
    ] {
        if let Some(v) = value {
            h.insert(key.into(), v.into());
        }
    }
    Value::Object(h)
}

/// Builds a payload map from `(key, value)` pairs.
pub(crate) fn payload<const N: usize>(pairs: [(&str, Value); N]) -> Map<String, Value> {
    pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}
