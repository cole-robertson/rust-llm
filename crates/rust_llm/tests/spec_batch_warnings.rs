//! `spec/ruby_llm/protocols/gemini/batches_spec.rb:135` on its own: it reads `tracing` WARN events,
//! whose per-callsite interest cache races with other tests registering callsites concurrently,
//! so it runs alone in this binary (like `spec_mcp_http_warnings.rs`).

use std::sync::{Arc, Mutex};

use rust_llm::{Batch, BatchStatus, Config};
use serde_json::{Value, json};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn config(server: &MockServer) -> Arc<Config> {
    let mut c = Config::default();
    c.set("openai_api_base", format!("{}/v1", server.uri()));
    c.set("openai_api_key", "test");
    c.set("gemini_api_base", format!("{}/v1beta", server.uri()));
    c.set("gemini_api_key", "test");
    c.max_retries = 0;
    Arc::new(c)
}

async fn mount(server: &MockServer, verb: &str, at: &str, body: Value) {
    Mock::given(method(verb))
        .and(path(at))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(server)
        .await;
}

async fn gemini_batch(batch: Value) -> (MockServer, Batch) {
    let server = MockServer::start().await;
    mount(&server, "GET", "/v1beta/batches/abc", batch).await;
    let found = Batch::find_with_config(config(&server), "batches/abc", Some("gemini"))
        .await
        .unwrap();
    (server, found)
}

fn completed_gemini_batch(inlined: Value) -> Value {
    json!({
        "name": "batches/abc",
        "metadata": { "model": "models/gemini-2.5-flash", "state": "BATCH_STATE_SUCCEEDED" },
        "response": { "inlinedResponses": { "inlinedResponses": inlined } }
    })
}

// spec: protocols/gemini/batches_spec.rb:135 #parse_inline_response > logs and skips a per-item error
#[tokio::test]
async fn logs_and_skips_a_per_item_error() {
    let (_server, mut batch) = gemini_batch(completed_gemini_batch(json!([
        { "metadata": { "custom_id": "0" }, "error": { "code": 400, "message": "too long" } }
    ])))
    .await;
    let warnings = Arc::new(Mutex::new(Vec::new()));
    let messages = {
        let _guard = tracing::dispatcher::set_default(&tracing::Dispatch::new(WarnCollector(
            warnings.clone(),
        )));
        batch.messages().await.unwrap()
    };
    assert_eq!(messages.len(), 1);
    assert!(messages[0].is_none());
    assert_eq!(batch.statuses(), &[Some(BatchStatus::Failed)]);
    let warnings = warnings.lock().unwrap();
    assert!(
        warnings.iter().any(|w| w.contains("failed: too long")),
        "{warnings:?}"
    );
}

/// Collects WARN messages, as the Ruby spec's `RubyLLM.logger` spy does.
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
        struct Message<'a>(&'a mut String);
        impl tracing::field::Visit for Message<'_> {
            fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                if field.name() == "message" {
                    self.0.push_str(&format!("{value:?}"));
                }
            }
        }
        if *event.metadata().level() == tracing::Level::WARN {
            let mut message = String::new();
            event.record(&mut Message(&mut message));
            self.0.lock().unwrap().push(message);
        }
    }
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}
