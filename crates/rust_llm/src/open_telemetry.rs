//! Port of `lib/ruby_llm/open_telemetry.rb` and `lib/ruby_llm/open_telemetry/attributes.rb`:
//! optional OpenTelemetry tracing through the instrumentation events.
//!
//! ```ruby
//! RubyLLM::OpenTelemetry.enable
//! ```
//!
//! ```ignore
//! // Cargo.toml: rust_llm = { version = "2", features = ["opentelemetry"] }
//! opentelemetry::global::set_tracer_provider(my_sdk_provider); // the application's SDK
//! rust_llm::open_telemetry::enable()?;
//! ```
//!
//! Model calls (`chat`, `embeddings`, `generate_content`, ...), tool calls (`execute_tool`), and
//! workflows (`invoke_workflow`, `ruby_llm.workflow_step`) become spans of the global tracer
//! provider's `rust_llm` tracer, children of the OpenTelemetry context current when they start.
//! The work runs with its span as the current context, so nested calls (a chat inside a tool,
//! a tool inside a workflow step) are its children. Spans carry the GenAI semantic-convention
//! attributes and never content: no prompts, messages, metadata, vectors, or error messages.
//!
//! RustLLM never installs a tracer provider or SDK: configure your own, then call [`enable`].
//! The configured instrumenter keeps receiving every event as before. Tracing failures (a panic
//! in an exporter or provider) are logged at WARN and never fail the operation.
//!
//! The tracer is the `opentelemetry` API crate, behind this crate's optional `opentelemetry`
//! feature (Ruby's optional `opentelemetry-api` gem); without it [`enable`] returns an error.

use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(feature = "opentelemetry")]
use serde_json::{Map, Value};

/// Whether new operations are traced (`Support::Instrumentation.subscribe(@subscriber)`).
static ENABLED: AtomicBool = AtomicBool::new(false);

/// `RubyLLM::OpenTelemetry.enable`: traces new operations through the global tracer provider,
/// without replacing the configured instrumenter. Repeated calls are harmless. The application
/// owns the tracer provider, SDK, and exporters. Requires the `opentelemetry` feature.
pub fn enable() -> crate::Result<()> {
    #[cfg(feature = "opentelemetry")]
    {
        ENABLED.store(true, Ordering::SeqCst);
        Ok(())
    }
    // `rescue LoadError`: "requires the 'opentelemetry-api' gem. Add it to your Gemfile."
    #[cfg(not(feature = "opentelemetry"))]
    Err(crate::Error::Configuration(
        "OpenTelemetry tracing requires the `opentelemetry` feature. Add `features = [\"opentelemetry\"]` to rust_llm in your Cargo.toml."
            .into(),
    ))
}

/// `RubyLLM::OpenTelemetry.disable`: stops tracing new operations without shutting down the
/// SDK. Operations already running finish their spans normally.
pub fn disable() {
    ENABLED.store(false, Ordering::SeqCst);
}

/// Whether an event named `name` starting now gets a span (its payload must then be built).
pub(crate) fn is_traced(name: &str) -> bool {
    #[cfg(feature = "opentelemetry")]
    {
        ENABLED.load(Ordering::SeqCst) && attributes::operation(name).is_some()
    }
    #[cfg(not(feature = "opentelemetry"))]
    {
        let _ = name;
        false
    }
}

/// The span of one instrumented block (`OpenTelemetry#instrument`), or nothing when tracing is
/// off or the event is not traced. Ends when finished, or when dropped unfinished (a cancelled
/// future, Ruby's nonlocal return).
#[cfg(not(feature = "opentelemetry"))]
pub(crate) struct Span;

#[cfg(not(feature = "opentelemetry"))]
impl Span {
    pub(crate) fn start(
        _name: &str,
        _payload: Option<&serde_json::Map<String, serde_json::Value>>,
    ) -> Span {
        Span
    }

    pub(crate) fn instrument<F: std::future::Future>(
        &self,
        future: F,
    ) -> impl std::future::Future<Output = F::Output> + use<F> {
        future
    }

    pub(crate) fn finish(
        &mut self,
        _payload: Option<&serde_json::Map<String, serde_json::Value>>,
        _error: Option<&crate::Error>,
    ) {
    }
}

#[cfg(feature = "opentelemetry")]
pub(crate) struct Span {
    /// The context current at the start, with this span added; `None` once ended.
    cx: Option<opentelemetry::Context>,
}

#[cfg(feature = "opentelemetry")]
impl Span {
    /// `tracer.start_span(Attributes.span_name(...), kind:, attributes: Attributes.request(...))`,
    /// a child of the current context.
    pub(crate) fn start(name: &str, payload: Option<&Map<String, Value>>) -> Span {
        use opentelemetry::trace::{TraceContextExt, Tracer, TracerProvider};
        let operation = attributes::operation(name).filter(|_| ENABLED.load(Ordering::SeqCst));
        let cx = match (operation, payload) {
            (Some(operation), Some(payload)) => safely(|| {
                let scope = opentelemetry::InstrumentationScope::builder("rust_llm")
                    .with_version(crate::VERSION)
                    .build();
                let tracer = opentelemetry::global::tracer_provider().tracer_with_scope(scope);
                let span = tracer
                    .span_builder(attributes::span_name(operation, payload))
                    .with_kind(attributes::span_kind(operation))
                    .with_attributes(attributes::request(operation, payload, name))
                    .start(&tracer);
                opentelemetry::Context::current().with_span(span)
            }),
            _ => None,
        };
        Span { cx }
    }

    /// `Trace.with_span(span) { yield }`: runs `future` with this span as the current context on
    /// every poll. A panic unwinding out of it is recorded on the span (`record_error`).
    pub(crate) fn instrument<F: std::future::Future>(
        &self,
        future: F,
    ) -> impl std::future::Future<Output = F::Output> + use<F> {
        use futures::future::Either;
        let Some(cx) = self.cx.clone() else {
            return Either::Left(future);
        };
        let span = cx.clone();
        // Combinators rather than an `async` block, which would hold `future` twice and double
        // the size of every instrumented future it nests in.
        let guarded = futures::FutureExt::map(
            futures::FutureExt::catch_unwind(std::panic::AssertUnwindSafe(future)),
            move |result| match result {
                Ok(output) => output,
                Err(panic) => {
                    record_error(&span, "panic");
                    std::panic::resume_unwind(panic)
                }
            },
        );
        Either::Right(opentelemetry::context::FutureExt::with_context(guarded, cx))
    }

    /// The `rescue` and `ensure` of `OpenTelemetry#instrument`: records a failure, adds the
    /// response attributes, and ends the span.
    pub(crate) fn finish(
        &mut self,
        payload: Option<&Map<String, Value>>,
        error: Option<&crate::Error>,
    ) {
        use opentelemetry::trace::TraceContextExt;
        let Some(cx) = self.cx.take() else {
            return;
        };
        if let Some(e) = error {
            record_error(&cx, &e.class_name());
        }
        if let Some(payload) = payload {
            safely(|| cx.span().set_attributes(attributes::response(payload)));
        }
        safely(|| cx.span().end());
    }
}

#[cfg(feature = "opentelemetry")]
impl Drop for Span {
    fn drop(&mut self) {
        use opentelemetry::trace::TraceContextExt;
        if let Some(cx) = self.cx.take() {
            safely(|| cx.span().end());
        }
    }
}

/// `record_error`: the error's class, never its message.
#[cfg(feature = "opentelemetry")]
fn record_error(cx: &opentelemetry::Context, class: &str) {
    use opentelemetry::trace::TraceContextExt;
    safely(|| {
        let span = cx.span();
        span.set_attribute(opentelemetry::KeyValue::new(
            "error.type",
            class.to_string(),
        ));
        span.set_status(opentelemetry::trace::Status::error(""));
    });
}

/// `safely`: a tracing failure is logged and never reaches the application. Rust's tracing
/// failures are panics in the provider, SDK, or exporter; like Ruby, only the kind is logged.
#[cfg(feature = "opentelemetry")]
fn safely<T>(f: impl FnOnce() -> T) -> Option<T> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(value) => Some(value),
        Err(_) => {
            tracing::warn!("OpenTelemetry instrumentation failed (panic)");
            None
        }
    }
}

/// Port of `lib/ruby_llm/open_telemetry/attributes.rb`.
#[cfg(feature = "opentelemetry")]
mod attributes {
    use opentelemetry::trace::SpanKind;
    use opentelemetry::{Array, KeyValue, StringValue};
    use serde_json::{Map, Value};

    /// `OPERATIONS`, keyed by RustLLM's event names.
    pub(super) fn operation(event: &str) -> Option<&'static str> {
        Some(match event {
            "chat.rust_llm" => "chat",
            "embedding.rust_llm" => "embeddings",
            "image.rust_llm" | "speech.rust_llm" => "generate_content",
            "transcription.rust_llm" => "transcription",
            "ocr.rust_llm" => "ocr",
            "rerank.rust_llm" => "rerank",
            "moderation.rust_llm" => "moderation",
            "judgment.rust_llm" => "judgment",
            "tool_call.rust_llm" => "execute_tool",
            "workflow.rust_llm" => "invoke_workflow",
            "workflow_step.rust_llm" => "ruby_llm.workflow_step",
            _ => return None,
        })
    }

    /// `PROVIDERS`: the semantic-convention name where it differs from the slug.
    fn provider(slug: &str) -> &str {
        match slug {
            "bedrock" => "aws.bedrock",
            "azure" => "azure.ai.openai",
            "gemini" => "gcp.gen_ai",
            "vertexai" => "gcp.vertex_ai",
            "mistral" => "mistral_ai",
            "xai" => "x_ai",
            other => other,
        }
    }

    /// `INTERNAL_OPERATIONS`.
    const INTERNAL_OPERATIONS: &[&str] =
        &["execute_tool", "invoke_workflow", "ruby_llm.workflow_step"];

    /// `span_name`: the operation and its target, when there is one.
    pub(super) fn span_name(operation: &str, payload: &Map<String, Value>) -> String {
        let target = match operation {
            "execute_tool" => "tool_name",
            "invoke_workflow" => "workflow_name",
            "ruby_llm.workflow_step" => "workflow_step_name",
            _ => "model",
        };
        match payload.get(target).and_then(Value::as_str) {
            Some(target) => format!("{operation} {target}"),
            None => operation.to_string(),
        }
    }

    pub(super) fn span_kind(operation: &str) -> SpanKind {
        if INTERNAL_OPERATIONS.contains(&operation) {
            SpanKind::Internal
        } else {
            SpanKind::Client
        }
    }

    /// A payload value as an attribute value; `nil` (absent or null) is compacted away.
    fn value(v: Option<&Value>) -> Option<opentelemetry::Value> {
        match v? {
            Value::String(s) => Some(s.clone().into()),
            Value::Bool(b) => Some((*b).into()),
            Value::Number(n) => match n.as_i64() {
                Some(i) => Some(i.into()),
                None => n.as_f64().map(Into::into),
            },
            _ => None,
        }
    }

    fn compact(pairs: Vec<(&'static str, Option<opentelemetry::Value>)>) -> Vec<KeyValue> {
        pairs
            .into_iter()
            .filter_map(|(k, v)| v.map(|v| KeyValue::new(k, v)))
            .collect()
    }

    /// `request(operation, payload, event:)`.
    pub(super) fn request(
        operation: &str,
        payload: &Map<String, Value>,
        event: &str,
    ) -> Vec<KeyValue> {
        if matches!(operation, "invoke_workflow" | "ruby_llm.workflow_step") {
            return workflow(operation, payload);
        }
        if operation == "execute_tool" {
            return tool(payload);
        }
        compact(vec![
            ("gen_ai.operation.name", Some(operation.to_string().into())),
            (
                "gen_ai.provider.name",
                payload
                    .get("provider")
                    .and_then(Value::as_str)
                    .map(|p| provider(p).to_string().into()),
            ),
            ("gen_ai.request.model", value(payload.get("model"))),
            (
                "gen_ai.request.temperature",
                value(payload.get("temperature")),
            ),
            (
                "gen_ai.request.max_tokens",
                value(payload.get("max_output_tokens")),
            ),
            (
                "gen_ai.request.stream",
                (payload.get("streaming") == Some(&Value::Bool(true))).then(|| true.into()),
            ),
            (
                "gen_ai.embeddings.dimension.count",
                value(payload.get("dimensions")),
            ),
            (
                "gen_ai.output.type",
                output_type(event, payload).map(Into::into),
            ),
        ])
    }

    /// `output_type(event, payload)`.
    fn output_type(event: &str, payload: &Map<String, Value>) -> Option<&'static str> {
        if payload.get("schema").is_some_and(|s| !s.is_null()) {
            return Some("json");
        }
        match event {
            "image.rust_llm" => Some("image"),
            "speech.rust_llm" => Some("speech"),
            _ => None,
        }
    }

    /// `response(payload)`: the answering model, finish reason, and tokens (`response_tokens`,
    /// the attempt's own, over the event's `tokens`).
    pub(super) fn response(payload: &Map<String, Value>) -> Vec<KeyValue> {
        let finish_reason = payload
            .get("response")
            .and_then(|r| r.get("finish_reason"))
            .and_then(Value::as_str)
            .map(|r| {
                opentelemetry::Value::Array(Array::String(vec![StringValue::from(r.to_string())]))
            });
        let mut attributes = compact(vec![
            (
                "gen_ai.response.model",
                value(payload.get("response_model")),
            ),
            ("gen_ai.response.finish_reasons", finish_reason),
        ]);
        if payload.get("tokens").is_some_and(|t| !t.is_null()) {
            let tokens = payload
                .get("response_tokens")
                .filter(|t| !t.is_null())
                .or_else(|| payload.get("tokens"));
            attributes.extend(self::tokens(tokens.and_then(Value::as_object)));
        }
        attributes
    }

    /// `tokens(tokens)`: input counts cache reads and writes; unknown counts are left out and
    /// reported zeroes kept.
    fn tokens(tokens: Option<&Map<String, Value>>) -> Vec<KeyValue> {
        let Some(tokens) = tokens else {
            return Vec::new();
        };
        let count = |key: &str| tokens.get(key).and_then(Value::as_i64);
        let input: Vec<i64> = ["input_tokens", "cache_read_tokens", "cache_write_tokens"]
            .into_iter()
            .filter_map(count)
            .collect();
        compact(vec![
            (
                "gen_ai.usage.input_tokens",
                (!input.is_empty()).then(|| input.iter().sum::<i64>().into()),
            ),
            (
                "gen_ai.usage.output_tokens",
                count("output_tokens").map(Into::into),
            ),
            (
                "gen_ai.usage.cache_read.input_tokens",
                count("cache_read_tokens").map(Into::into),
            ),
            (
                "gen_ai.usage.cache_write.input_tokens",
                count("cache_write_tokens").map(Into::into),
            ),
            (
                "gen_ai.usage.reasoning.output_tokens",
                count("thinking_tokens").map(Into::into),
            ),
        ])
    }

    /// `tool(payload)`.
    fn tool(payload: &Map<String, Value>) -> Vec<KeyValue> {
        compact(vec![
            ("gen_ai.operation.name", Some("execute_tool".into())),
            ("gen_ai.tool.name", value(payload.get("tool_name"))),
            ("gen_ai.tool.call.id", value(payload.get("tool_call_id"))),
            ("gen_ai.tool.type", Some("function".into())),
        ])
    }

    /// `workflow(operation, payload)`.
    fn workflow(operation: &str, payload: &Map<String, Value>) -> Vec<KeyValue> {
        compact(vec![
            (
                "gen_ai.operation.name",
                (operation != "ruby_llm.workflow_step").then(|| operation.to_string().into()),
            ),
            ("gen_ai.workflow.name", value(payload.get("workflow_name"))),
            ("ruby_llm.workflow.id", value(payload.get("workflow_id"))),
            (
                "ruby_llm.workflow.step.name",
                value(payload.get("workflow_step_name")),
            ),
            (
                "ruby_llm.workflow.step.id",
                value(payload.get("workflow_step_id")),
            ),
        ])
    }
}
