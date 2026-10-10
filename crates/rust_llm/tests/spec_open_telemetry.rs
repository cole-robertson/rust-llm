//! RubyLLM 2.1's `open_telemetry_spec.rb` and `open_telemetry_live_spec.rb`: optional
//! OpenTelemetry tracing (`rust_llm::open_telemetry`). Ruby's shared context
//! (`spec/support/open_telemetry.rb`) installs an SDK tracer provider with an in-memory exporter
//! and enables tracing; [`traced`] does the same with `opentelemetry_sdk`. The global provider and
//! the enabled flag are process-wide, so every test here holds [`OTEL`].
//!
//! Ruby stubs `provider.complete`; here a wiremock server answers in the provider's wire format.
//! The live examples replay RubyLLM's recorded cassettes. Ruby's thread and fiber variants map to
//! tokio: concurrent tools run in one task (`FuturesUnordered`), spawned tasks carry the
//! `opentelemetry::Context` they were given.

mod spec_helpers;
mod support;

/// `does not load OpenTelemetry when requiring or eager loading RubyLLM`: the API crate is an
/// optional dependency outside the default features, and no SDK is ever a dependency.
// spec: open_telemetry_spec.rb:25 does not load OpenTelemetry when requiring or eager loading RubyLLM
#[test]
fn opentelemetry_is_an_optional_dependency_outside_the_default_features() {
    let manifest = include_str!("../Cargo.toml");
    let dependencies = manifest
        .split("[dependencies]")
        .nth(1)
        .and_then(|s| s.split("\n[").next())
        .expect("[dependencies]");
    let api = dependencies
        .lines()
        .find(|l| l.starts_with("opentelemetry ="))
        .expect("opentelemetry dependency");
    assert!(api.contains("optional = true"), "{api}");
    assert!(!dependencies.contains("opentelemetry_sdk"));
    let features = manifest
        .split("[features]")
        .nth(1)
        .and_then(|s| s.split("\n[").next())
        .expect("[features]");
    assert!(
        !features
            .lines()
            .any(|l| l.starts_with("default") && l.contains("opentelemetry")),
        "{features}"
    );
}

// spec: open_telemetry_spec.rb:50 reports a useful error when the optional API is missing
#[cfg(not(feature = "opentelemetry"))]
#[test]
fn enable_reports_the_missing_feature() {
    let error = rust_llm::open_telemetry::enable().unwrap_err();
    let message = error.to_string();
    assert!(
        message.contains("`opentelemetry` feature") && message.contains("Cargo.toml"),
        "{message}"
    );
    assert!(matches!(error, rust_llm::Error::Configuration(_)));
}

#[cfg(feature = "opentelemetry")]
mod traced {
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use async_trait::async_trait;
    use futures::FutureExt as _;
    use opentelemetry::trace::{
        FutureExt as _, Span as _, SpanId, SpanKind, Status, TraceContextExt, Tracer as _,
        TracerProvider as _,
    };
    use opentelemetry::{Context, KeyValue, global};
    use opentelemetry_sdk::trace::{InMemorySpanExporter, Sampler, SdkTracerProvider, SpanData};
    use rust_llm::open_telemetry;
    use rust_llm::{
        Agent, Chat, Config, EmbedOptions, Message, Parameter, ProtocolName, Tool, ToolCall,
        ToolCalls, ToolChoice, ToolError, ToolResult,
    };
    use serde_json::{Map, Value, json};
    use wiremock::matchers::{body_json, method, path};
    use wiremock::{Mock, MockServer, Request, ResponseTemplate};

    use super::spec_helpers::{Sequence, config, sse};
    use super::support;

    /// `model_for(:openai, :temperature)`.
    const MODEL: &str = "gpt-4.1-nano";

    /// The global tracer provider and enabled flag are process-wide.
    static OTEL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    /// `include_context 'with OpenTelemetry tracing'`: an SDK provider with an in-memory exporter
    /// as the global provider, and tracing enabled. Dropping it disables tracing, shuts the
    /// provider down, and puts the no-op provider back.
    struct Traced {
        exporter: InMemorySpanExporter,
        provider: SdkTracerProvider,
        _lock: tokio::sync::MutexGuard<'static, ()>,
    }

    async fn traced() -> Traced {
        let lock = OTEL.lock().await;
        let exporter = InMemorySpanExporter::default();
        let provider = SdkTracerProvider::builder()
            .with_sampler(Sampler::AlwaysOn)
            .with_simple_exporter(exporter.clone())
            .build();
        global::set_tracer_provider(provider.clone());
        open_telemetry::enable().unwrap();
        Traced {
            exporter,
            provider,
            _lock: lock,
        }
    }

    impl Traced {
        fn spans(&self) -> Vec<SpanData> {
            self.exporter.get_finished_spans().unwrap()
        }

        /// `provider.tracer('application')`.
        fn application(&self) -> opentelemetry_sdk::trace::SdkTracer {
            self.provider.tracer("application")
        }
    }

    impl Drop for Traced {
        fn drop(&mut self) {
            open_telemetry::disable();
            let _ = self.provider.shutdown();
            global::set_tracer_provider(opentelemetry::trace::noop::NoopTracerProvider::new());
        }
    }

    fn attr(span: &SpanData, key: &str) -> Option<opentelemetry::Value> {
        span.attributes
            .iter()
            .find(|kv| kv.key.as_str() == key)
            .map(|kv| kv.value.clone())
    }

    fn int(span: &SpanData, key: &str) -> Option<i64> {
        match attr(span, key)? {
            opentelemetry::Value::I64(i) => Some(i),
            other => panic!("{key} is {other:?}"),
        }
    }

    fn string(span: &SpanData, key: &str) -> Option<String> {
        attr(span, key).map(|v| v.as_str().into_owned())
    }

    fn is_error(span: &SpanData) -> bool {
        matches!(span.status, Status::Error { .. })
    }

    fn current_span_id() -> SpanId {
        Context::current().span().span_context().span_id()
    }

    fn payload(value: Value) -> Map<String, Value> {
        value.as_object().cloned().unwrap()
    }

    /// The spec's `response`: a Responses API answer with every token count.
    fn response_body(text: &str) -> Value {
        json!({
            "id": "resp_1", "object": "response", "status": "completed", "model": MODEL,
            "output": [{ "type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                         "content": [{ "type": "output_text", "text": text, "annotations": [] }] }],
            "usage": { "input_tokens": 15, "output_tokens": 5,
                       "input_tokens_details": { "cached_tokens": 3, "cache_write_tokens": 2 },
                       "output_tokens_details": { "reasoning_tokens": 1 } }
        })
    }

    async fn server_answering(text: &str) -> MockServer {
        let server = MockServer::start().await;
        let body = response_body(text);
        Mock::given(path("/v1/responses"))
            .respond_with(move |_: &Request| ResponseTemplate::new(200).set_body_json(body.clone()))
            .mount(&server)
            .await;
        server
    }

    /// `stub_chat(context)`.
    fn stub_chat(config: Arc<Config>) -> Chat {
        Chat::with_config(config, Some(MODEL), Some("openai"), false).unwrap()
    }

    type Events = Arc<Mutex<Vec<(String, Map<String, Value>)>>>;

    /// `CaptureInstrumenter` as the configured instrumenter.
    fn capturing(server: &MockServer) -> (Arc<Config>, Events) {
        let events: Events = Default::default();
        let sink = events.clone();
        let mut c = (*config(server)).clone();
        c.instrumenter = Some(Arc::new(
            move |name: &str, payload: &Map<String, Value>, _: Option<Duration>| {
                sink.lock()
                    .unwrap()
                    .push((name.to_string(), payload.clone()));
            },
        ));
        (Arc::new(c), events)
    }

    // spec: open_telemetry_spec.rb:35 works with only the API without configuring an SDK
    #[tokio::test]
    async fn works_with_only_the_api_without_configuring_an_sdk() {
        let _lock = OTEL.lock().await;
        global::set_tracer_provider(opentelemetry::trace::noop::NoopTracerProvider::new());
        open_telemetry::enable().unwrap();

        let result = rust_llm::instrument(
            &Arc::new(Config::default()),
            "chat.rust_llm",
            payload(json!({ "provider": "openai" })),
            async { Ok("ok") },
        )
        .await;
        open_telemetry::disable();

        assert_eq!(result.unwrap(), "ok", "changed return value");
        // `replaced provider`: the global provider is still the no-op one.
        let probe = global::tracer("probe").start("probe");
        assert!(!probe.span_context().is_valid(), "replaced provider");
    }

    // spec: open_telemetry_spec.rb:56 traces a chat under the application span without capturing content
    #[tokio::test]
    async fn traces_a_chat_under_the_application_span_without_capturing_content() {
        let otel = traced().await;
        let server = server_answering("private response").await;
        let mut chat = stub_chat(config(&server))
            .with_temperature(0.2)
            .with_max_output_tokens(100);

        let request = otel.application().start("request");
        let cx = Context::current_with_span(request);
        let result = chat.ask("private prompt").with_context(cx.clone()).await;
        cx.span().end();

        assert_eq!(result.unwrap().content(), "private response");
        let spans = otel.spans();
        let (chat_span, parent) = (&spans[0], &spans[1]);
        assert_eq!(chat_span.name, format!("chat {MODEL}"));
        assert_eq!(chat_span.span_kind, SpanKind::Client);
        assert_eq!(chat_span.parent_span_id, parent.span_context.span_id());
        assert_eq!(chat_span.instrumentation_scope.name(), "rust_llm");
        assert_eq!(
            chat_span.instrumentation_scope.version(),
            Some(rust_llm::VERSION)
        );
        for (key, value) in [
            ("gen_ai.operation.name", "chat".into()),
            ("gen_ai.provider.name", "openai".into()),
            ("gen_ai.request.model", MODEL.into()),
            ("gen_ai.response.model", MODEL.into()),
            ("gen_ai.request.temperature", 0.2.into()),
            ("gen_ai.request.max_tokens", 100i64.into()),
            (
                "gen_ai.response.finish_reasons",
                opentelemetry::Value::Array(vec![opentelemetry::StringValue::from("stop")].into()),
            ),
            ("gen_ai.usage.input_tokens", 15i64.into()),
            ("gen_ai.usage.output_tokens", 5i64.into()),
            ("gen_ai.usage.cache_read.input_tokens", 3i64.into()),
            ("gen_ai.usage.cache_write.input_tokens", 2i64.into()),
            ("gen_ai.usage.reasoning.output_tokens", 1i64.into()),
        ] {
            assert_eq!(
                attr(chat_span, key),
                Some(value),
                "{key} in {:?}",
                chat_span.attributes
            );
        }
        let exported = format!("{:?}", chat_span.attributes);
        for absent in ["private", "gen_ai.system", "conversation.id"] {
            assert!(!exported.contains(absent), "{absent} in {exported}");
        }
        assert_eq!(chat_span.status, Status::Unset);
    }

    // spec: open_telemetry_spec.rb:79 preserves Rails subscribers, their payloads, and event-only notifications
    #[tokio::test]
    async fn preserves_the_configured_instrumenter_its_payloads_and_event_only_notifications() {
        let otel = traced().await;
        let server = server_answering("private response").await;
        let (config, events) = capturing(&server);
        stub_chat(config.clone())
            .ask("private prompt")
            .await
            .unwrap();

        assert!(config.instrumenter.is_some());
        let events = events.lock().unwrap().clone();
        // Ruby stubs `provider.complete`, below the transport; here the HTTP request also fires
        // `request.rust_llm`, which Ruby's stub skips.
        let names: Vec<&str> = events
            .iter()
            .map(|(n, _)| n.as_str())
            .filter(|n| *n != "request.rust_llm")
            .collect();
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(sorted, ["chat.rust_llm", "usage.rust_llm"]);
        let chat_event = &events.iter().find(|(n, _)| n == "chat.rust_llm").unwrap().1;
        assert_eq!(chat_event["response"]["content"], "private response");
        assert_eq!(chat_event["response"]["model"], MODEL);
        assert_eq!(otel.spans().len(), 1);
    }

    // spec: open_telemetry_spec.rb:93 enables once and disables without shutting down the application provider
    #[tokio::test]
    async fn enables_once_and_disables_without_shutting_down_the_application_provider() {
        let otel = traced().await;
        let server = server_answering("ok").await;
        open_telemetry::enable().unwrap();
        stub_chat(config(&server)).ask("first").await.unwrap();
        open_telemetry::disable();
        stub_chat(config(&server)).ask("second").await.unwrap();
        otel.application().start("still running").end();

        let names: Vec<String> = otel.spans().iter().map(|s| s.name.to_string()).collect();
        assert_eq!(names, [format!("chat {MODEL}"), "still running".into()]);
    }

    /// A Responses stream that yields `text`, then finishes with the spec's token counts.
    fn response_stream(text: &str) -> ResponseTemplate {
        let completed = json!({ "type": "response.completed", "response": response_body(text) });
        sse(format!(
            "event: response.output_text.delta\ndata: {}\n\nevent: response.completed\ndata: {completed}\n\n",
            json!({ "type": "response.output_text.delta", "delta": text })
        ))
    }

    // spec: open_telemetry_spec.rb:103 keeps the span active through streaming callbacks and the final response
    #[tokio::test]
    async fn keeps_the_span_active_through_streaming_callbacks_and_the_final_response() {
        let otel = traced().await;
        let server = MockServer::start().await;
        Mock::given(path("/v1/responses"))
            .respond_with(response_stream("private chunk"))
            .mount(&server)
            .await;
        let exporter = otel.exporter.clone();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let sink = seen.clone();
        stub_chat(config(&server))
            .ask_stream("private prompt", move |chunk: &Message| {
                sink.lock().unwrap().push(chunk.content().to_string());
                assert!(exporter.get_finished_spans().unwrap().is_empty());
                assert!(Context::current().span().is_recording());
            })
            .await
            .unwrap();

        assert_eq!(seen.lock().unwrap().first().unwrap(), "private chunk");
        let spans = otel.spans();
        assert_eq!(spans.len(), 1);
        assert_eq!(attr(&spans[0], "gen_ai.request.stream"), Some(true.into()));
        assert_eq!(int(&spans[0], "gen_ai.usage.output_tokens"), Some(5));
    }

    // spec: open_telemetry_spec.rb:124 finishes failed spans and restores context without exporting exception messages
    #[tokio::test]
    async fn finishes_failed_spans_and_restores_context_without_exporting_exception_messages() {
        let otel = traced().await;
        let server = MockServer::start().await;
        // An unmapped status raises the base error class (`RubyLLM::Error`, `Error::Api`).
        Mock::given(path("/v1/responses"))
            .respond_with(
                ResponseTemplate::new(418)
                    .set_body_json(json!({ "error": { "message": "private error detail" } })),
            )
            .mount(&server)
            .await;
        let previous = current_span_id();

        let error = stub_chat(config(&server))
            .ask("private prompt")
            .await
            .unwrap_err();

        assert!(matches!(error, rust_llm::Error::Api(..)), "{error:?}");
        assert!(error.to_string().contains("private error detail"));
        assert_eq!(current_span_id(), previous);
        let spans = otel.spans();
        assert!(is_error(&spans[0]));
        assert_eq!(
            string(&spans[0], "error.type").as_deref(),
            Some("rust_llm::Error::Api")
        );
        assert!(spans[0].events.is_empty());
        assert!(!format!("{:?} {:?}", spans[0].attributes, spans[0].status).contains("private"));
    }

    // spec: open_telemetry_spec.rb:138 finishes spans on cancellation
    #[tokio::test]
    async fn finishes_spans_on_cancellation() {
        let otel = traced().await;
        let result: rust_llm::Result<()> = rust_llm::instrument(
            &Arc::new(Config::default()),
            "chat.rust_llm",
            payload(json!({ "model": MODEL, "provider": "openai" })),
            async { Err(rust_llm::Error::Cancelled) },
        )
        .await;

        assert!(matches!(result, Err(rust_llm::Error::Cancelled)));
        assert_eq!(
            string(&otel.spans()[0], "error.type").as_deref(),
            Some("rust_llm::Error::Cancelled")
        );
    }

    /// Ruby's `throw` out of the block is a future dropped before it finishes.
    // spec: open_telemetry_spec.rb:146 finishes spans on nonlocal returns
    #[tokio::test]
    async fn finishes_spans_on_nonlocal_returns() {
        let otel = traced().await;
        let config = Arc::new(Config::default());
        let result: rust_llm::Result<&str> = tokio::select! {
            biased;
            r = rust_llm::instrument(
                &config,
                "chat.rust_llm",
                payload(json!({ "model": MODEL, "provider": "openai" })),
                std::future::pending(),
            ) => r,
            _ = async {} => Ok("ok"),
        };

        assert_eq!(result.unwrap(), "ok");
        assert_eq!(otel.spans().len(), 1);
    }

    /// Collects `tracing` WARN events on this thread, standing in for `RubyLLM.logger.warn`.
    struct WarnCollector(Arc<Mutex<Vec<String>>>);

    impl tracing::Subscriber for WarnCollector {
        fn register_callsite(
            &self,
            _: &'static tracing::Metadata<'static>,
        ) -> tracing::subscriber::Interest {
            tracing::subscriber::Interest::sometimes()
        }
        fn max_level_hint(&self) -> Option<tracing::level_filters::LevelFilter> {
            Some(tracing::level_filters::LevelFilter::TRACE)
        }
        fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
            true
        }
        fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
            tracing::span::Id::from_u64(1)
        }
        fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
        fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
        fn event(&self, event: &tracing::Event<'_>) {
            struct Text<'a>(&'a mut String);
            impl tracing::field::Visit for Text<'_> {
                fn record_debug(
                    &mut self,
                    field: &tracing::field::Field,
                    value: &dyn std::fmt::Debug,
                ) {
                    if field.name() == "message" {
                        self.0.push_str(&format!("{value:?}"));
                    }
                }
            }
            if *event.metadata().level() == tracing::Level::WARN {
                let mut text = String::new();
                event.record(&mut Text(&mut text));
                self.0.lock().unwrap().push(text);
            }
        }
        fn enter(&self, _: &tracing::span::Id) {}
        fn exit(&self, _: &tracing::span::Id) {}
    }

    /// A tracer provider whose `tracer` fails, like the spec's stubbed `provider.tracer`.
    struct FailingProvider;

    impl opentelemetry::trace::TracerProvider for FailingProvider {
        type Tracer = opentelemetry::trace::noop::NoopTracer;
        fn tracer_with_scope(&self, _: opentelemetry::InstrumentationScope) -> Self::Tracer {
            panic!("private exporter failure")
        }
    }

    // spec: open_telemetry_spec.rb:155 isolates a tracing failure from the application operation
    #[tokio::test]
    async fn isolates_a_tracing_failure_from_the_application_operation() {
        let _otel = traced().await;
        global::set_tracer_provider(FailingProvider);
        let server = server_answering("private response").await;
        let warnings = Arc::new(Mutex::new(Vec::new()));
        let result = {
            let _guard = tracing::dispatcher::set_default(&tracing::Dispatch::new(WarnCollector(
                warnings.clone(),
            )));
            stub_chat(config(&server)).ask("hello").await
        };

        assert_eq!(result.unwrap().content(), "private response");
        // Ruby logs the exception class; a Rust tracing failure is a panic.
        assert!(
            warnings
                .lock()
                .unwrap()
                .contains(&"OpenTelemetry instrumentation failed (panic)".to_string()),
            "{:?}",
            warnings.lock().unwrap()
        );
    }

    // spec: open_telemetry_spec.rb:163 traces embeddings without exporting inputs or vectors
    #[tokio::test]
    async fn traces_embeddings_without_exporting_inputs_or_vectors() {
        let otel = traced().await;
        let server = MockServer::start().await;
        Mock::given(path("/v1/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "object": "list", "model": "text-embedding-3-small",
                "data": [{ "object": "embedding", "index": 0, "embedding": [0.1, 0.2] }],
                "usage": { "prompt_tokens": 8, "total_tokens": 8 }
            })))
            .mount(&server)
            .await;

        let embedding = rust_llm::embed(
            "private input",
            EmbedOptions {
                model: Some("text-embedding-3-small"),
                dimensions: Some(2),
                config: Some(config(&server)),
                ..Default::default()
            },
        )
        .await
        .unwrap();

        assert_eq!(embedding.vectors, rust_llm::Vectors::Single(vec![0.1, 0.2]));
        let spans = otel.spans();
        assert_eq!(spans[0].name, "embeddings text-embedding-3-small");
        assert_eq!(int(&spans[0], "gen_ai.embeddings.dimension.count"), Some(2));
        assert_eq!(int(&spans[0], "gen_ai.usage.input_tokens"), Some(8));
        let exported = format!("{:?}", spans[0].attributes);
        for absent in ["private", "0.1", "output_tokens"] {
            assert!(!exported.contains(absent), "{absent} in {exported}");
        }
    }

    // spec: open_telemetry_spec.rb:177 keeps transport retries inside a single model span
    #[tokio::test]
    async fn keeps_transport_retries_inside_a_single_model_span() {
        let otel = traced().await;
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(Sequence(Mutex::new(
                vec![
                    ResponseTemplate::new(500).set_body_json(json!({ "error": { "message": "retry" } })),
                    ResponseTemplate::new(200).set_body_json(json!({
                        "model": MODEL,
                        "choices": [{ "message": { "role": "assistant", "content": "done" }, "finish_reason": "stop" }],
                        "usage": { "prompt_tokens": 5, "completion_tokens": 2 }
                    })),
                ]
                .into(),
            )))
            .mount(&server)
            .await;
        let mut c = (*config(&server)).clone();
        c.max_retries = 1;
        c.retry_interval = 0.0;

        Chat::with_config(Arc::new(c), Some(MODEL), Some("openai"), false)
            .unwrap()
            .with_protocol(ProtocolName::ChatCompletions)
            .ask("Hello")
            .await
            .unwrap();

        let spans = otel.spans();
        assert_eq!(spans.len(), 1);
        assert_eq!(int(&spans[0], "gen_ai.usage.input_tokens"), Some(5));
        assert_eq!(spans[0].status, Status::Unset);
    }

    /// The spec's primary model records a failed attempt billed 7 input tokens, then raises
    /// `ServiceUnavailableError`. A provider bills a failed attempt only when it reported usage
    /// before failing, which over HTTP is a stream that sends usage and then a 503 error event,
    /// so this asks with a block.
    // spec: open_telemetry_spec.rb:197 keeps fallback usage on the model span that incurred it
    #[tokio::test]
    async fn keeps_fallback_usage_on_the_model_span_that_incurred_it() {
        let otel = traced().await;
        let server = MockServer::start().await;
        let failing = format!(
            "data: {}\n\ndata: {}\n\n",
            json!({ "id": "c1", "object": "chat.completion.chunk", "model": MODEL,
                    "choices": [{ "index": 0, "delta": { "role": "assistant", "content": "partial" } }],
                    "usage": { "prompt_tokens": 7, "completion_tokens": 1 } }),
            json!({ "error": { "code": 503, "message": "try another model" } })
        );
        Mock::given(path("/v1/chat/completions"))
            .respond_with(sse(failing))
            .mount(&server)
            .await;
        let answer = concat!(
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"model\":\"claude-haiku-4-5\",\"usage\":{\"input_tokens\":4,\"output_tokens\":1}}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"done\"}}\n\n",
            "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n\n",
            "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
        );
        Mock::given(path("/v1/messages"))
            .respond_with(sse(answer.to_string()))
            .mount(&server)
            .await;
        let mut chat = stub_chat(config(&server))
            .with_protocol(ProtocolName::ChatCompletions)
            .with_fallbacks(["claude-haiku-4-5".into()]);

        let result = chat.ask_stream("Hello", |_| {}).await.unwrap();

        assert_eq!(result.tokens().input, Some(11));
        let spans = otel.spans();
        let inputs: Vec<Option<i64>> = spans
            .iter()
            .map(|s| int(s, "gen_ai.usage.input_tokens"))
            .collect();
        assert_eq!(inputs, [Some(7), Some(4)]);
        let providers: Vec<Option<String>> = spans
            .iter()
            .map(|s| string(s, "gen_ai.provider.name"))
            .collect();
        assert_eq!(
            providers,
            [Some("openai".to_string()), Some("anthropic".to_string())]
        );
    }

    // spec: open_telemetry_spec.rb:223 omits unknown token counts and preserves reported zeroes
    #[tokio::test]
    async fn omits_unknown_token_counts_and_preserves_reported_zeroes() {
        let otel = traced().await;
        rust_llm::instrument(
            &Arc::new(Config::default()),
            "embedding.rust_llm",
            payload(json!({ "provider": "openai", "tokens": { "input_tokens": 0 } })),
            async { Ok(()) },
        )
        .await
        .unwrap();

        let spans = otel.spans();
        assert_eq!(int(&spans[0], "gen_ai.usage.input_tokens"), Some(0));
        assert_eq!(attr(&spans[0], "gen_ai.usage.output_tokens"), None);
    }

    // spec: open_telemetry_spec.rb:230 uses conventional provider names and output types without copying arbitrary metadata
    #[tokio::test]
    async fn uses_conventional_provider_names_and_output_types_without_copying_arbitrary_metadata()
    {
        let otel = traced().await;
        let config = Arc::new(Config::default());
        rust_llm::instrument(
            &config,
            "image.rust_llm",
            payload(
                json!({ "provider": "vertexai", "model": MODEL, "prompt": "private prompt",
                            "metadata": { "secret": "private metadata" } }),
            ),
            async { Ok(()) },
        )
        .await
        .unwrap();
        rust_llm::instrument(
            &config,
            "speech.rust_llm",
            payload(json!({ "provider": "azure", "model": MODEL, "input": "private speech" })),
            async { Ok(()) },
        )
        .await
        .unwrap();

        let spans = otel.spans();
        let of =
            |key: &str| -> Vec<Option<String>> { spans.iter().map(|s| string(s, key)).collect() };
        assert_eq!(
            of("gen_ai.provider.name"),
            [Some("gcp.vertex_ai".into()), Some("azure.ai.openai".into())]
        );
        assert_eq!(
            of("gen_ai.output.type"),
            [Some("image".into()), Some("speech".into())]
        );
        let names: Vec<String> = spans.iter().map(|s| s.name.to_string()).collect();
        let expected = format!("generate_content {MODEL}");
        assert_eq!(names, [expected.clone(), expected]);
        let exported = format!(
            "{:?}",
            spans.iter().map(|s| &s.attributes).collect::<Vec<_>>()
        );
        assert!(!exported.contains("private") && !exported.contains("secret"));
    }

    /// Two operations interleaving on one thread (Ruby's fibers): `join!` polls both in one task.
    // spec: open_telemetry_spec.rb:241 keeps distinct operations isolated when fibers interleave
    #[tokio::test]
    async fn keeps_distinct_operations_isolated_when_operations_interleave() {
        let otel = traced().await;
        let config = Arc::new(Config::default());
        let barrier = tokio::sync::Barrier::new(2);
        let run = || {
            rust_llm::instrument(
                &config,
                "chat.rust_llm",
                payload(json!({ "provider": "openai", "model": MODEL })),
                async {
                    let current = current_span_id();
                    barrier.wait().await;
                    assert_eq!(current_span_id(), current);
                    Ok(current)
                },
            )
        };
        let (first, second) = tokio::join!(run(), run());

        assert_ne!(first.unwrap(), second.unwrap());
        let spans = otel.spans();
        assert_eq!(spans.len(), 2);
        assert!(spans.iter().all(|s| s.parent_span_id == SpanId::INVALID));
    }

    /// A Rust streaming callback cannot return an error; it fails by panicking, which unwinds
    /// through the chat like Ruby's raise. A panic has no class, so `error.type` is `panic`.
    // spec: open_telemetry_spec.rb:260 restores the parent when a streaming callback raises
    #[tokio::test]
    async fn restores_the_parent_when_a_streaming_callback_panics() {
        let otel = traced().await;
        let server = MockServer::start().await;
        Mock::given(path("/v1/responses"))
            .respond_with(response_stream("hello"))
            .mount(&server)
            .await;
        let mut chat = stub_chat(config(&server));
        let parent = otel.application().start("request");
        let parent_id = parent.span_context().span_id();
        let cx = Context::current_with_span(parent);
        async {
            let raised = std::panic::AssertUnwindSafe(
                chat.ask_stream("hello", |_| panic!("callback failed")),
            )
            .catch_unwind()
            .await;
            assert!(raised.is_err());
            assert_eq!(current_span_id(), parent_id);
        }
        .with_context(cx.clone())
        .await;
        cx.span().end();

        let spans = otel.spans();
        assert_eq!(string(&spans[0], "error.type").as_deref(), Some("panic"));
        assert!(is_error(&spans[0]));
    }

    // spec: open_telemetry_spec.rb:274 finishes an active span after tracing is disabled
    #[tokio::test]
    async fn finishes_an_active_span_after_tracing_is_disabled() {
        let otel = traced().await;
        let server = server_answering("ok").await;
        let exporter = otel.exporter.clone();
        rust_llm::instrument(
            &Arc::new(Config::default()),
            "chat.rust_llm",
            payload(json!({ "model": MODEL, "provider": "openai" })),
            async {
                open_telemetry::disable();
                stub_chat(config(&server)).ask("untraced").await?;
                assert!(exporter.get_finished_spans().unwrap().is_empty());
                Ok(())
            },
        )
        .await
        .unwrap();

        assert_eq!(otel.spans().len(), 1);
    }

    // spec: open_telemetry_spec.rb:284 continues to notify a custom instrumenter after tracing is disabled
    #[tokio::test]
    async fn continues_to_notify_a_custom_instrumenter_after_tracing_is_disabled() {
        let otel = traced().await;
        let server = server_answering("ok").await;
        let (config, events) = capturing(&server);
        let mut chat = stub_chat(config);
        chat.ask("traced").await.unwrap();
        open_telemetry::disable();
        chat.ask("untraced").await.unwrap();

        let chats = events
            .lock()
            .unwrap()
            .iter()
            .filter(|(n, _)| n == "chat.rust_llm")
            .count();
        assert_eq!(chats, 2);
        assert_eq!(otel.spans().len(), 1);
    }

    // spec: open_telemetry_spec.rb:296 allows application work to run when the sampler drops the trace
    #[tokio::test]
    async fn allows_application_work_to_run_when_the_sampler_drops_the_trace() {
        let otel = traced().await;
        let dropped = SdkTracerProvider::builder()
            .with_sampler(Sampler::AlwaysOff)
            .build();
        global::set_tracer_provider(dropped.clone());
        let server = server_answering("private response").await;

        let result = stub_chat(config(&server)).ask("hello").await;

        assert_eq!(result.unwrap().content(), "private response");
        assert!(otel.spans().is_empty());
        let _ = dropped.shutdown();
    }

    /// The SDK's tracer, with spans whose attribute writes fail (`add_attributes` raising).
    struct FailingAttributes(SdkTracerProvider);
    struct FailingTracer(opentelemetry_sdk::trace::SdkTracer);
    struct FailingSpan(opentelemetry_sdk::trace::Span);

    impl opentelemetry::trace::TracerProvider for FailingAttributes {
        type Tracer = FailingTracer;
        fn tracer_with_scope(&self, scope: opentelemetry::InstrumentationScope) -> FailingTracer {
            FailingTracer(self.0.tracer_with_scope(scope))
        }
    }

    impl opentelemetry::trace::Tracer for FailingTracer {
        type Span = FailingSpan;
        fn build_with_context(
            &self,
            builder: opentelemetry::trace::SpanBuilder,
            parent_cx: &Context,
        ) -> FailingSpan {
            FailingSpan(self.0.build_with_context(builder, parent_cx))
        }
    }

    impl opentelemetry::trace::Span for FailingSpan {
        fn add_event_with_timestamp<T>(
            &mut self,
            name: T,
            timestamp: std::time::SystemTime,
            attributes: Vec<KeyValue>,
        ) where
            T: Into<std::borrow::Cow<'static, str>>,
        {
            self.0.add_event_with_timestamp(name, timestamp, attributes)
        }
        fn span_context(&self) -> &opentelemetry::trace::SpanContext {
            self.0.span_context()
        }
        fn is_recording(&self) -> bool {
            self.0.is_recording()
        }
        fn set_attribute(&mut self, _: KeyValue) {
            panic!("exporter failure")
        }
        fn set_status(&mut self, status: Status) {
            self.0.set_status(status)
        }
        fn update_name<T>(&mut self, new_name: T)
        where
            T: Into<std::borrow::Cow<'static, str>>,
        {
            self.0.update_name(new_name)
        }
        fn add_link(
            &mut self,
            span_context: opentelemetry::trace::SpanContext,
            attributes: Vec<KeyValue>,
        ) {
            self.0.add_link(span_context, attributes)
        }
        fn end_with_timestamp(&mut self, timestamp: std::time::SystemTime) {
            self.0.end_with_timestamp(timestamp)
        }
    }

    // spec: open_telemetry_spec.rb:306 closes a span even when recording its response attributes fails
    #[tokio::test]
    async fn closes_a_span_even_when_recording_its_response_attributes_fails() {
        let otel = traced().await;
        global::set_tracer_provider(FailingAttributes(otel.provider.clone()));
        let server = server_answering("private response").await;

        let result = stub_chat(config(&server)).ask("hello").await;

        assert_eq!(result.unwrap().content(), "private response");
        assert_eq!(otel.spans().len(), 1);
    }

    #[derive(Debug, Clone, PartialEq)]
    struct TestContext(&'static str);

    type Observed = Arc<Mutex<Vec<(SpanId, SpanId, Option<TestContext>)>>>;

    /// `TelemetryNestedChat`: asks a nested chat, noting the span around it before and after,
    /// and the application's context value.
    struct TelemetryNestedChat {
        config: Arc<Config>,
        observed: Observed,
    }

    #[async_trait]
    impl Tool for TelemetryNestedChat {
        fn name(&self) -> String {
            "telemetry_nested_chat".into()
        }
        fn description(&self) -> String {
            String::new()
        }
        fn parameters(&self) -> Vec<Parameter> {
            vec![Parameter::new("label")]
        }
        async fn execute(
            &self,
            args: Map<String, Value>,
            _: &ToolCall,
        ) -> Result<ToolResult, ToolError> {
            let label = args["label"].as_str().unwrap_or_default().to_string();
            let before = current_span_id();
            tokio::time::sleep(Duration::from_millis(10)).await;
            Chat::with_config(self.config.clone(), Some(MODEL), Some("openai"), false)?
                .with_protocol(ProtocolName::ChatCompletions)
                .ask(label.clone())
                .await?;
            self.observed.lock().unwrap().push((
                before,
                current_span_id(),
                Context::current().get::<TestContext>().cloned(),
            ));
            Ok(label.into())
        }
    }

    /// Ruby's `threads` and `fibers` modes are one mode here: the calls of a response run at once
    /// in the chat's task.
    // spec: open_telemetry_spec.rb:320 traces a complete HTTP chat and overlapping nested tools using #{mode}
    #[tokio::test]
    async fn traces_a_complete_http_chat_and_overlapping_nested_tools() {
        let otel = traced().await;
        let server = MockServer::start().await;
        Mock::given(path("/v1/chat/completions"))
            .respond_with(|request: &Request| {
                let body: Value = serde_json::from_slice(&request.body).unwrap();
                let mut message = json!({ "role": "assistant", "content": "done" });
                let last_is_user = body["messages"].as_array().and_then(|m| m.last()).map(|m| m["role"] == "user");
                if body.get("tools").is_some() && last_is_user == Some(true) {
                    message["tool_calls"] = ["alpha", "beta"]
                        .iter()
                        .map(|label| json!({ "id": label, "type": "function",
                            "function": { "name": "telemetry_nested_chat", "arguments": json!({ "label": label }).to_string() } }))
                        .collect();
                }
                let finish_reason = if message.get("tool_calls").is_some() { "tool_calls" } else { "stop" };
                ResponseTemplate::new(200).set_body_json(json!({
                    "model": MODEL, "choices": [{ "message": message, "finish_reason": finish_reason }],
                    "usage": { "prompt_tokens": 4, "completion_tokens": 2 }
                }))
            })
            .mount(&server)
            .await;
        let observed: Observed = Default::default();
        let tool = TelemetryNestedChat {
            config: config(&server),
            observed: observed.clone(),
        };
        let mut chat = stub_chat(config(&server))
            .with_protocol(ProtocolName::ChatCompletions)
            .with_tool(tool)
            .with_tool_concurrency(true);
        let application = Context::current().with_value(TestContext("preserved"));

        rust_llm::workflow("nested tools", None, None, |_| async {
            assert_eq!(chat.ask("Run both tools").await?.content(), "done");
            Ok(())
        })
        .with_context(application)
        .await
        .unwrap();

        let spans = otel.spans();
        let workflow = spans
            .iter()
            .find(|s| s.name == "invoke_workflow nested tools")
            .unwrap();
        let tools: Vec<&SpanData> = spans
            .iter()
            .filter(|s| s.name == "execute_tool telemetry_nested_chat")
            .collect();
        assert_eq!(
            spans.len(),
            7,
            "{:?}",
            spans.iter().map(|s| &s.name).collect::<Vec<_>>()
        );
        assert!(
            tools
                .iter()
                .all(|t| t.parent_span_id == workflow.span_context.span_id())
        );
        assert_eq!(tools.len(), 2);
        let latest_start = tools.iter().map(|t| t.start_time).max().unwrap();
        let earliest_end = tools.iter().map(|t| t.end_time).min().unwrap();
        assert!(latest_start < earliest_end, "tools overlap");
        for tool in &tools {
            let children = spans
                .iter()
                .filter(|s| s.parent_span_id == tool.span_context.span_id())
                .count();
            assert_eq!(children, 1);
        }
        let mut observations = observed.lock().unwrap().clone();
        let mut expected: Vec<_> = tools
            .iter()
            .map(|t| {
                let id = t.span_context.span_id();
                (id, id, Some(TestContext("preserved")))
            })
            .collect();
        observations.sort_by_key(|o| o.0.to_string());
        expected.sort_by_key(|o| o.0.to_string());
        assert_eq!(observations, expected);
        assert_eq!(Context::current().get::<TestContext>(), None);
    }

    /// `ToolConcurrency.run(mode, calls)` with instrumented calls inside a workflow step.
    async fn concurrent_tools_in_a_step(spawned: bool) {
        let otel = traced().await;
        let config = Arc::new(Config::default());
        let previous = current_span_id();
        rust_llm::workflow("answer", None, None, |workflow| {
            let config = config.clone();
            async move {
                workflow
                    .step("lookup", None, async {
                        let calls = (0..2).map(|index| {
                            let config = config.clone();
                            async move {
                                let tool = payload(json!({ "tool_name": "lookup", "tool_call_id": format!("call_{index}") }));
                                rust_llm::instrument(&config.clone(), "tool_call.rust_llm", tool, async move {
                                    rust_llm::instrument(
                                        &config,
                                        "chat.rust_llm",
                                        payload(json!({ "model": MODEL, "provider": "openai" })),
                                        async { Ok("ok") },
                                    )
                                    .await
                                })
                                .await
                            }
                        });
                        if spawned {
                            // Ruby's threads: each call on its own task, given the caller's context.
                            let handles: Vec<_> = calls
                                .map(|call| tokio::spawn(call.with_context(Context::current())))
                                .collect();
                            for handle in handles {
                                handle.await.unwrap()?;
                            }
                        } else {
                            for result in futures::future::join_all(calls).await {
                                result?;
                            }
                        }
                        Ok(())
                    })
                    .await
            }
        })
        .await
        .unwrap();

        let spans = otel.spans();
        let find = |name: &str| spans.iter().find(|s| s.name == name).unwrap();
        let workflow = find("invoke_workflow answer");
        let step = find("ruby_llm.workflow_step lookup");
        let tools: Vec<&SpanData> = spans
            .iter()
            .filter(|s| s.name == "execute_tool lookup")
            .collect();
        let chats: Vec<&SpanData> = spans
            .iter()
            .filter(|s| s.name == format!("chat {MODEL}"))
            .collect();
        assert_eq!(step.parent_span_id, workflow.span_context.span_id());
        assert_eq!(tools.len(), 2);
        assert!(
            tools
                .iter()
                .all(|t| t.parent_span_id == step.span_context.span_id())
        );
        let mut chat_parents: Vec<SpanId> = chats.iter().map(|c| c.parent_span_id).collect();
        let mut tool_ids: Vec<SpanId> = tools.iter().map(|t| t.span_context.span_id()).collect();
        chat_parents.sort_by_key(|id| id.to_string());
        tool_ids.sort_by_key(|id| id.to_string());
        assert_eq!(chat_parents, tool_ids);
        assert!(tools.iter().all(|t| t.span_kind == SpanKind::Internal));
        assert_eq!(current_span_id(), previous);
    }

    // spec: open_telemetry_spec.rb:370 parents concurrent tools and nested calls to their workflow step using #{mode}
    #[tokio::test]
    async fn parents_concurrent_tools_and_nested_calls_to_their_workflow_step_in_one_task() {
        concurrent_tools_in_a_step(false).await;
    }

    // spec: open_telemetry_spec.rb:370 parents concurrent tools and nested calls to their workflow step using #{mode}
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn parents_concurrent_tools_and_nested_calls_to_their_workflow_step_on_spawned_tasks() {
        concurrent_tools_in_a_step(true).await;
    }

    // ---- open_telemetry_live_spec.rb ------------------------------------------------------------

    // spec: open_telemetry_live_spec.rb:11 traces #{slug} #{protocol} #{streaming ? 'streaming' : 'buffered'} chats
    #[tokio::test]
    async fn traces_recorded_chats_buffered_and_streaming() {
        for (slug, protocol, model) in [
            ("openai", Some(ProtocolName::Responses), MODEL),
            ("openai", Some(ProtocolName::ChatCompletions), MODEL),
            ("anthropic", None, "claude-haiku-4-5"),
            ("gemini", None, "gemini-2.5-flash"),
        ] {
            for streaming in [false, true] {
                let otel = traced().await;
                let protocol_name = match protocol {
                    Some(ProtocolName::Responses) => "responses_",
                    Some(_) => "chat_completions_",
                    None => "",
                };
                let mode = if streaming { "streaming" } else { "buffered" };
                let name = format!("opentelemetry_traces_{slug}_{protocol_name}{mode}_chats");
                let cassette = support::Cassette::start(&name).await.expect(&name);
                let mut chat = Chat::with_config(
                    support::config_for(&cassette, slug),
                    Some(model),
                    Some(slug),
                    false,
                )
                .unwrap()
                .with_max_output_tokens(128);
                if let Some(p) = protocol {
                    chat = chat.with_protocol(p);
                }
                let observed = Arc::new(Mutex::new(Vec::new()));
                let result = if streaming {
                    let (sink, exporter) = (observed.clone(), otel.exporter.clone());
                    chat.ask_stream("Reply with just the word hello.", move |_| {
                        sink.lock().unwrap().push(current_span_id());
                        assert!(exporter.get_finished_spans().unwrap().is_empty());
                    })
                    .await
                } else {
                    chat.ask("Reply with just the word hello.").await
                }
                .unwrap();
                cassette.assert_all_matched().await;

                assert!(!result.content().is_empty(), "{name}");
                let spans = otel.spans();
                assert_eq!(spans.len(), 1, "{name}");
                let span = &spans[0];
                assert_eq!(span.name, format!("chat {}", chat.model().id), "{name}");
                assert_eq!(span.span_kind, SpanKind::Client);
                assert_eq!(
                    string(span, "gen_ai.response.model"),
                    result.model,
                    "{name}"
                );
                assert!(
                    int(span, "gen_ai.usage.input_tokens").unwrap() > 0,
                    "{name}"
                );
                assert!(
                    int(span, "gen_ai.usage.output_tokens").unwrap() > 0,
                    "{name}"
                );
                assert_eq!(span.status, Status::Unset, "{name}");
                if streaming {
                    let mut ids = observed.lock().unwrap().clone();
                    assert!(!ids.is_empty(), "{name}");
                    ids.dedup();
                    assert_eq!(ids, [span.span_context.span_id()], "{name}");
                }
            }
        }
    }

    /// Serves a recorded cassette whose requests run concurrently: each recorded request is
    /// answered once, matched by path and its exact JSON body, in whatever order they arrive.
    async fn serve_concurrent(name: &str) -> (MockServer, Arc<Config>) {
        let interactions = support::load(name).expect(name);
        let server = MockServer::start().await;
        for interaction in interactions {
            let route = interaction
                .uri
                .split_once("://")
                .and_then(|(_, rest)| rest.split_once('/'))
                .map(|(_, p)| format!("/{p}"))
                .unwrap();
            let body: Value = serde_json::from_str(&interaction.request_body).unwrap();
            let content_type = interaction
                .response_headers
                .iter()
                .find(|(k, _)| k.eq_ignore_ascii_case("content-type"))
                .and_then(|(_, v)| v.as_str())
                .unwrap_or("application/json")
                .to_string();
            Mock::given(method("POST"))
                .and(path(route))
                .and(body_json(body))
                .respond_with(
                    ResponseTemplate::new(interaction.status)
                        .set_body_raw(interaction.response_body.into_bytes(), &content_type),
                )
                .up_to_n_times(1)
                .expect(1)
                .mount(&server)
                .await;
        }
        let mut c = Config::default();
        c.set("openai_api_base", format!("{}/v1", server.uri()));
        c.set("openai_api_key", "test-key");
        c.max_retries = 0;
        (server, Arc::new(c))
    }

    /// Ruby's fiber reactor: three branches polled concurrently in one task.
    // spec: open_telemetry_live_spec.rb:40 keeps simultaneous streaming chats and embeddings isolated in a fiber reactor
    #[tokio::test]
    async fn keeps_simultaneous_streaming_chats_and_embeddings_isolated() {
        let otel = traced().await;
        let (_server, config) =
            serve_concurrent("opentelemetry_keeps_simultaneous_streaming_chats_and_embeddings_isolated_in_a_fiber_reactor")
                .await;
        let tracer = otel.application();
        let branch = |index: usize| {
            let parent = tracer.start(format!("request {index}"));
            let trace_id = parent.span_context().trace_id();
            let cx = Context::current_with_span(parent);
            let config = config.clone();
            let end = cx.clone();
            async move {
                let name = format!("branch {index}");
                let run = rust_llm::workflow(&name, None, None, move |_| async move {
                    Chat::with_config(config.clone(), Some(MODEL), None, false)?
                        .ask_stream(format!("Reply with just the number {index}."), move |_| {
                            assert_eq!(
                                Context::current().span().span_context().trace_id(),
                                trace_id
                            );
                        })
                        .await?;
                    rust_llm::embed(
                        format!("Synthetic document {index}"),
                        EmbedOptions {
                            model: Some("text-embedding-3-small"),
                            config: Some(config),
                            ..Default::default()
                        },
                    )
                    .await
                });
                let result = run.with_context(cx).await;
                end.span().end();
                result
            }
        };
        let (a, b, c) = tokio::join!(branch(0), branch(1), branch(2));
        for result in [a, b, c] {
            result.unwrap();
        }

        let spans = otel.spans();
        assert_eq!(spans.len(), 12);
        for index in 0..3 {
            let parent = spans
                .iter()
                .find(|s| s.name == format!("request {index}"))
                .unwrap();
            let workflow = spans
                .iter()
                .find(|s| s.name == format!("invoke_workflow branch {index}"))
                .unwrap();
            let children: Vec<&SpanData> = spans
                .iter()
                .filter(|s| s.parent_span_id == workflow.span_context.span_id())
                .collect();
            assert_eq!(workflow.parent_span_id, parent.span_context.span_id());
            let mut operations: Vec<Option<String>> = children
                .iter()
                .map(|s| string(s, "gen_ai.operation.name"))
                .collect();
            operations.sort();
            assert_eq!(operations, [Some("chat".into()), Some("embeddings".into())]);
            assert!(
                children
                    .iter()
                    .all(|s| s.span_context.trace_id() == parent.span_context.trace_id())
            );
        }
        assert!(!Context::current().span().span_context().is_valid());
    }

    /// The spec's agent class: a model and a strict schema.
    struct AnswerAgent(Arc<Config>);

    impl Agent for AnswerAgent {
        fn model(&self) -> Option<&str> {
            Some(MODEL)
        }
        fn schema(&self) -> Option<Value> {
            Some(
                json!({ "type": "object", "properties": { "answer": { "type": "integer" } },
                         "required": ["answer"], "additionalProperties": false }),
            )
        }
        fn context(&self) -> Option<rust_llm::Context> {
            Some(rust_llm::Context::new((*self.0).clone()))
        }
    }

    // spec: open_telemetry_live_spec.rb:71 traces structured output from an agent
    #[tokio::test]
    async fn traces_structured_output_from_an_agent() {
        let otel = traced().await;
        let cassette =
            support::Cassette::start("opentelemetry_traces_structured_output_from_an_agent")
                .await
                .unwrap();

        let result = AnswerAgent(support::config_for(&cassette, "openai"))
            .chat()
            .unwrap()
            .ask("What is six times seven?")
            .await
            .unwrap();
        cassette.assert_all_matched().await;

        assert_eq!(result.parsed().unwrap(), Some(json!({ "answer": 42 })));
        let spans = otel.spans();
        assert_eq!(spans.len(), 1);
        assert_eq!(
            string(&spans[0], "gen_ai.output.type").as_deref(),
            Some("json")
        );
    }

    // spec: open_telemetry_live_spec.rb:87 reports a provider validation error without leaking the error body
    #[tokio::test]
    async fn reports_a_provider_validation_error_without_leaking_the_error_body() {
        let otel = traced().await;
        let cassette = support::Cassette::start(
            "opentelemetry_reports_a_provider_validation_error_without_leaking_the_error_body",
        )
        .await
        .unwrap();

        let error = Chat::with_config(
            support::config_for(&cassette, "openai"),
            Some(MODEL),
            None,
            false,
        )
        .unwrap()
        .with_max_output_tokens(-1)
        .ask("hello")
        .await
        .unwrap_err();
        cassette.assert_all_matched().await;

        assert!(
            matches!(error, rust_llm::Error::BadRequest(..)),
            "{error:?}"
        );
        let spans = otel.spans();
        assert_eq!(spans.len(), 1);
        assert_eq!(
            string(&spans[0], "error.type").as_deref(),
            Some("rust_llm::Error::BadRequest")
        );
        assert!(is_error(&spans[0]));
        assert!(spans[0].events.is_empty());
        let exported = format!("{:?} {:?}", spans[0].attributes, spans[0].status);
        assert!(!exported.contains(&error.to_string()), "{exported}");
        assert!(!Context::current().span().span_context().is_valid());
    }

    /// `TelemetryEcho`: embeds its label, then echoes it.
    struct TelemetryEcho(Arc<Config>);

    #[async_trait]
    impl Tool for TelemetryEcho {
        fn name(&self) -> String {
            "telemetry_echo".into()
        }
        fn description(&self) -> String {
            "Echoes a label".into()
        }
        fn parameters(&self) -> Vec<Parameter> {
            vec![Parameter::new("label").kind("string")]
        }
        async fn execute(
            &self,
            args: Map<String, Value>,
            _: &ToolCall,
        ) -> Result<ToolResult, ToolError> {
            let label = args["label"].as_str().unwrap_or_default().to_string();
            tokio::time::sleep(Duration::from_millis(10)).await;
            rust_llm::embed(
                label.clone(),
                EmbedOptions {
                    model: Some("text-embedding-3-small"),
                    config: Some(self.0.clone()),
                    ..Default::default()
                },
            )
            .await?;
            Ok(label.into())
        }
    }

    /// Both recordings (Ruby's threads and fibers modes) replay through the one concurrent mode.
    // spec: open_telemetry_live_spec.rb:100 traces a real model tool round using #{mode}
    #[tokio::test]
    async fn traces_a_real_model_tool_round() {
        for mode in ["threads", "fibers"] {
            let otel = traced().await;
            let (_server, config) = serve_concurrent(&format!(
                "opentelemetry_traces_a_real_model_tool_round_using_{mode}"
            ))
            .await;
            let chat = Chat::with_config(config.clone(), Some(MODEL), None, false)
                .unwrap()
                .with_protocol(ProtocolName::ChatCompletions)
                .with_tool(TelemetryEcho(config.clone()))
                .with_tool_choice(ToolChoice::Required)
                .unwrap()
                .with_tool_calls(ToolCalls::Many)
                .with_tool_concurrency(true);

            let chat = rust_llm::workflow(&format!("tools {mode}"), None, None, |_| async move {
                let mut chat = chat;
                chat.ask_later(
                    "Call telemetry_echo twice in parallel with label alpha and label beta. Both calls are independent.",
                )?;
                let generated = chat.generate().await?;
                assert_eq!(generated.tool_calls.as_ref().map(|c| c.len()), Some(2));
                chat.run_tools().await?;
                let mut chat = chat.with_tool_choice(ToolChoice::None)?;
                chat.generate().await?;
                Ok(chat)
            })
            .await
            .unwrap();

            let spans = otel.spans();
            let workflow = spans
                .iter()
                .find(|s| s.name == format!("invoke_workflow tools {mode}"))
                .unwrap();
            let tools: Vec<&SpanData> = spans
                .iter()
                .filter(|s| s.name == "execute_tool telemetry_echo")
                .collect();
            let embeddings: Vec<&SpanData> = spans
                .iter()
                .filter(|s| string(s, "gen_ai.operation.name").as_deref() == Some("embeddings"))
                .collect();
            assert_eq!(tools.len(), 2);
            assert!(
                tools
                    .iter()
                    .all(|t| t.parent_span_id == workflow.span_context.span_id())
            );
            let mut parents: Vec<SpanId> = embeddings.iter().map(|e| e.parent_span_id).collect();
            let mut tool_ids: Vec<SpanId> =
                tools.iter().map(|t| t.span_context.span_id()).collect();
            parents.sort_by_key(|id| id.to_string());
            tool_ids.sort_by_key(|id| id.to_string());
            assert_eq!(parents, tool_ids);
            let mut results: Vec<&str> = chat
                .messages()
                .iter()
                .filter(|m| m.is_tool_result())
                .map(Message::content)
                .collect();
            results.sort();
            assert_eq!(results, ["alpha", "beta"]);
            assert!(!Context::current().span().span_context().is_valid());
        }
    }
}
