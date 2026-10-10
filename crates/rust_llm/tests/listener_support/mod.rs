//! Shared by `spec_mcp_listener.rs` and `spec_mcp_listener_backoff.rs`: a scripted client whose
//! subscriptions each play out a script with their number, counted from 1.
#![allow(dead_code)]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use futures::future::BoxFuture;
use rust_llm::mcp::{Listener, ListenerCallback, OnNotification, Subscribe};
use rust_llm::{Error, Result};
use serde_json::{Value, json};
use tokio::sync::mpsc;

/// What a subscription does, given its number and a way to send the listener notifications.
pub type Script = Arc<dyn Fn(usize, Sender) -> BoxFuture<'static, Result<()>> + Send + Sync>;

#[derive(Clone)]
pub struct Sender(mpsc::UnboundedSender<Value>);

impl Sender {
    pub fn notify(&self, notification: Value) {
        let _ = self.0.send(notification);
    }
}

#[derive(Default)]
pub struct Record {
    attempts: Vec<Value>,
    cancelled: Vec<usize>,
}

/// `scripted_client(script)`.
pub struct Scripted {
    script: Script,
    record: Mutex<Record>,
}

impl Scripted {
    pub fn new(script: Script) -> Arc<Scripted> {
        Arc::new(Scripted {
            script,
            record: Mutex::new(Record::default()),
        })
    }

    pub fn attempts(&self) -> Vec<Value> {
        self.record.lock().unwrap().attempts.clone()
    }

    pub fn cancelled(&self) -> Vec<usize> {
        self.record.lock().unwrap().cancelled.clone()
    }
}

#[async_trait]
impl Subscribe for Scripted {
    async fn listen(
        &self,
        changes: &Value,
        on_notification: &mut OnNotification<'_>,
    ) -> Result<()> {
        let attempt = {
            let mut record = self.record.lock().unwrap();
            record.attempts.push(changes.clone());
            record.attempts.len()
        };
        let (sender, mut notifications) = mpsc::unbounded_channel();
        let mut script = (self.script)(attempt, Sender(sender));
        // The listener cancels a subscription through the progress cancellation flag, the way a
        // transport notices it (`CancelledError` in Ruby).
        loop {
            tokio::select! {
                Some(notification) = notifications.recv() => on_notification(&notification),
                ended = &mut script => {
                    while let Ok(notification) = notifications.try_recv() {
                        on_notification(&notification);
                    }
                    return ended;
                }
                _ = tokio::time::sleep(Duration::from_millis(10)) => {
                    if rust_llm::progress::is_cancelled() {
                        self.record.lock().unwrap().cancelled.push(attempt);
                        return Err(Error::Cancelled);
                    }
                }
            }
        }
    }
}

pub fn tools() -> Value {
    json!({ "toolsListChanged": true })
}

pub fn acknowledgment(honored: Value) -> Value {
    json!({ "method": "notifications/subscriptions/acknowledged", "params": { "notifications": honored } })
}

/// `sleep`: a script that never ends on its own.
pub async fn forever() -> Result<()> {
    std::future::pending::<()>().await;
    Ok(())
}

pub fn script<F, Fut>(f: F) -> Script
where
    F: Fn(usize, Sender) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<()>> + Send + 'static,
{
    Arc::new(move |attempt, notify| Box::pin(f(attempt, notify)))
}

pub fn acknowledging() -> Script {
    script(|_, notify| async move {
        notify.notify(acknowledgment(tools()));
        forever().await
    })
}

pub fn quiet() -> ListenerCallback {
    rust_llm::mcp::listener_callback(|_| async {})
}

pub fn listener(client: &Arc<Scripted>, timeout: Duration, callback: ListenerCallback) -> Listener {
    Listener::new(client.clone(), "files", timeout, None, callback)
}

pub async fn eventually(check: impl Fn() -> bool) {
    tokio::time::timeout(Duration::from_secs(5), async {
        while !check() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("timed out");
}

/// Counts `tracing` WARN events (`RubyLLM.logger.warn`) on this thread.
pub struct Warnings(pub Arc<Mutex<usize>>);

impl tracing::Subscriber for Warnings {
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
        if *event.metadata().level() == tracing::Level::WARN {
            *self.0.lock().unwrap() += 1;
        }
    }
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}

/// `allow(listener).to receive(:sleep) { |seconds| delays << seconds }`.
pub fn recorded_sleep(delays: Arc<Mutex<Vec<Duration>>>) -> rust_llm::mcp::Sleep {
    Arc::new(move |duration| {
        delays.lock().unwrap().push(duration);
        Box::pin(async {})
    })
}
