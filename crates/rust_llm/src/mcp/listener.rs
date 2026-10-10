//! Port of `lib/ruby_llm/mcp/listener.rb`: keeps a subscription to a server's changes open in a
//! task of its own, so the notifications reach the MCP while chats and requests go on.
//! [`Listener::start`] returns once the server acknowledges the subscription. When a stream
//! ends, the listener subscribes again after a delay that starts at a second and doubles up to
//! a minute, and gives up when the server agrees to send nothing or does not know the method.
//! Changes made in between are lost, so once the server acknowledges again, the `resumed`
//! callback receives what the listener listens to.
//!
//! Stopping cancels the subscription. Callbacks run outside its cancellation, one at a time and
//! in order, so a stop never cuts one short. The task reports no progress to a chat that started
//! it.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use futures::FutureExt;
use futures::future::BoxFuture;
use serde_json::{Value, json};
use tokio::sync::{Notify, mpsc, oneshot};
use tokio::task::JoinHandle;

use super::client::{ACKNOWLEDGED, METHOD_NOT_FOUND};
use super::{McpError, OnNotification};
use crate::error::{Error, Result};

const RETRY_DELAY: Duration = Duration::from_secs(1);
const MAX_RETRY_DELAY: Duration = Duration::from_secs(60);
const STOP_TIMEOUT: Duration = Duration::from_secs(5);

/// What a [`Listener`] subscribes through: `Client#listen`.
#[async_trait]
pub trait Subscribe: Send + Sync {
    /// Subscribes to `changes`, passing each notification of the subscription to
    /// `on_notification`, starting with the acknowledgment. Returns when the server ends the
    /// subscription, and fails when it refuses one or the stream breaks.
    async fn listen(&self, changes: &Value, on_notification: &mut OnNotification<'_>)
    -> Result<()>;
}

/// A callback the listener runs with a notification, or with what it listens to on resuming.
pub type ListenerCallback = Arc<dyn Fn(Value) -> BoxFuture<'static, ()> + Send + Sync>;

/// How the listener waits before subscribing again (`sleep`).
pub type Sleep = Arc<dyn Fn(Duration) -> BoxFuture<'static, ()> + Send + Sync>;

/// Wraps an async closure as a [`ListenerCallback`].
pub fn callback<F, Fut>(f: F) -> ListenerCallback
where
    F: Fn(Value) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = ()> + Send + 'static,
{
    Arc::new(move |value| Box::pin(f(value)))
}

enum State {
    Pending,
    Acknowledged(Value),
    Failed(Option<Error>),
}

struct Shared {
    name: String,
    state: Mutex<State>,
    settled: Notify,
    stopping: Arc<AtomicBool>,
    stopped: Notify,
}

impl Shared {
    fn settle(&self, state: State) {
        if let Ok(mut s) = self.state.lock() {
            *s = state;
        }
        self.settled.notify_waiters();
    }

    fn acknowledged(&self) -> Option<Value> {
        match &*self.state.lock().ok()? {
            State::Acknowledged(listened) => Some(listened.clone()),
            _ => None,
        }
    }

    fn is_stopping(&self) -> bool {
        self.stopping.load(Ordering::SeqCst)
    }
}

struct Running {
    changes: Value,
    shared: Arc<Shared>,
    task: JoinHandle<()>,
}

enum Item {
    Notification(Value),
    Barrier(oneshot::Sender<()>),
}

/// `RubyLLM::MCP::Listener`.
#[doc(hidden)]
pub struct Listener {
    client: Arc<dyn Subscribe>,
    name: String,
    timeout: Duration,
    resumed: Option<ListenerCallback>,
    on_notification: ListenerCallback,
    sleep: Sleep,
    control: tokio::sync::Mutex<Option<Running>>,
}

impl Listener {
    pub fn new(
        client: Arc<dyn Subscribe>,
        name: impl Into<String>,
        timeout: Duration,
        resumed: Option<ListenerCallback>,
        on_notification: ListenerCallback,
    ) -> Listener {
        Listener {
            client,
            name: name.into(),
            timeout,
            resumed,
            on_notification,
            sleep: Arc::new(|duration| Box::pin(tokio::time::sleep(duration))),
            control: tokio::sync::Mutex::new(None),
        }
    }

    /// Replaces how the listener waits before subscribing again, as RubyLLM's specs stub
    /// `sleep`.
    pub fn with_sleep(mut self, sleep: Sleep) -> Listener {
        self.sleep = sleep;
        self
    }

    /// `start(changes)`: listens for `changes`, a `subscriptions/listen` filter, instead of
    /// whatever the listener heard before. Returns the changes the server agreed to send, or
    /// `None` when there is nothing to listen for.
    pub async fn start(&self, changes: Value) -> Result<Option<Value>> {
        let mut control = self.control.lock().await;
        if let Some(running) = control.as_ref().filter(|r| !r.task.is_finished())
            && running.changes == changes
        {
            return Ok(running.shared.acknowledged());
        }
        halt(control.take()).await;
        if changes.as_object().is_none_or(|c| c.is_empty()) {
            return Ok(None);
        }
        let running = self.launch(changes);
        match acknowledgment(&running.shared, self.timeout).await {
            Ok(listened) => {
                *control = Some(running);
                Ok(Some(listened))
            }
            Err(e) => {
                halt(Some(running)).await;
                Err(e)
            }
        }
    }

    /// `stop`.
    pub async fn stop(&self) {
        let mut control = self.control.lock().await;
        halt(control.take()).await;
    }

    fn launch(&self, changes: Value) -> Running {
        let shared = Arc::new(Shared {
            name: self.name.clone(),
            state: Mutex::new(State::Pending),
            settled: Notify::new(),
            stopping: Arc::new(AtomicBool::new(false)),
            stopped: Notify::new(),
        });
        let (sender, receiver) = mpsc::unbounded_channel();
        let subscriptions = subscribe_loop(
            self.client.clone(),
            changes.clone(),
            shared.clone(),
            sender,
            self.sleep.clone(),
        );
        let dispatcher = dispatch_loop(
            receiver,
            shared.clone(),
            self.on_notification.clone(),
            self.resumed.clone(),
        );
        let flag = shared.stopping.clone();
        let task = tokio::spawn(crate::progress::listen(None, async move {
            futures::join!(crate::progress::watch(flag, subscriptions), dispatcher);
        }));
        Running {
            changes,
            shared,
            task,
        }
    }
}

/// `acknowledgment`: what the server agreed to send, or why it did not.
async fn acknowledgment(shared: &Shared, timeout: Duration) -> Result<Value> {
    let deadline = Instant::now() + timeout;
    loop {
        let settled = shared.settled.notified();
        tokio::pin!(settled);
        settled.as_mut().enable();
        if let Ok(mut state) = shared.state.lock() {
            match &mut *state {
                State::Acknowledged(listened) => return Ok(listened.clone()),
                State::Failed(error) => {
                    return Err(error.take().unwrap_or_else(|| {
                        McpError::new(format!("{} ended the subscription", shared.name)).into()
                    }));
                }
                State::Pending => {}
            }
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() || tokio::time::timeout(remaining, settled).await.is_err() {
            return Err(McpError::new(format!(
                "{} did not acknowledge the subscription",
                shared.name
            ))
            .into());
        }
    }
}

/// `halt`: cancels the subscription and waits for a running callback to finish. A callback that
/// stops its own listener does not wait for itself.
async fn halt(running: Option<Running>) {
    let Some(running) = running else {
        return;
    };
    running.shared.stopping.store(true, Ordering::SeqCst);
    running.shared.stopped.notify_waiters();
    if tokio::task::try_id() == Some(running.task.id()) {
        return;
    }
    let mut task = running.task;
    if tokio::time::timeout(STOP_TIMEOUT, &mut task).await.is_err() {
        task.abort();
    }
}

/// `run`: subscribes again after each ending until `again?` says otherwise.
async fn subscribe_loop(
    client: Arc<dyn Subscribe>,
    changes: Value,
    shared: Arc<Shared>,
    sender: mpsc::UnboundedSender<Item>,
    sleep: Sleep,
) {
    let mut delay = RETRY_DELAY;
    loop {
        let opened = Instant::now();
        let ending = {
            let mut on_notification = |notification: &Value| {
                let _ = sender.send(Item::Notification(notification.clone()));
            };
            client.listen(&changes, &mut on_notification).await
        };
        if matches!(ending, Err(Error::Cancelled)) || shared.is_stopping() {
            return;
        }
        let (done, processed) = oneshot::channel();
        if sender.send(Item::Barrier(done)).is_err() || processed.await.is_err() {
            return;
        }
        let ending = ending.err();
        if !again(&shared, ending.as_ref()) {
            if let Some(ending) = ending.filter(|_| shared.acknowledged().is_none()) {
                shared.settle(State::Failed(Some(ending)));
            }
            return;
        }
        if opened.elapsed() >= MAX_RETRY_DELAY {
            delay = RETRY_DELAY;
        }
        let jittered = delay.mul_f64((1.0 + rand::random::<f64>()) / 2.0);
        let reason = match &ending {
            Some(e) => format!("stopped sending changes: {e}"),
            None => "ended the subscription".to_string(),
        };
        tracing::warn!(
            "{} {reason}; listening again in {:.1}s",
            shared.name,
            jittered.as_secs_f64()
        );
        let stopped = shared.stopped.notified();
        tokio::pin!(stopped);
        stopped.as_mut().enable();
        if shared.is_stopping() {
            return;
        }
        tokio::select! {
            _ = sleep(jittered) => {}
            _ = stopped => return,
        }
        delay = (delay * 2).min(MAX_RETRY_DELAY);
    }
}

/// `again?(ending)`: a subscription that never was acknowledged settles with why it ended.
fn again(shared: &Shared, ending: Option<&Error>) -> bool {
    let Some(listened) = shared.acknowledged() else {
        if ending.is_none() {
            shared.settle(State::Failed(Some(
                McpError::new(format!(
                    "{} ended the subscription before acknowledging it",
                    shared.name
                ))
                .into(),
            )));
        }
        return false;
    };
    let method_not_found =
        matches!(ending, Some(Error::Mcp(e)) if e.code == Some(METHOD_NOT_FOUND));
    listened.as_object().is_some_and(|l| !l.is_empty()) && !method_not_found
}

/// `receive`: runs the callbacks for each notification in order, then settles the
/// acknowledgment, which starts the listener or, once it was acknowledged, resumes it.
async fn dispatch_loop(
    mut receiver: mpsc::UnboundedReceiver<Item>,
    shared: Arc<Shared>,
    on_notification: ListenerCallback,
    resumed: Option<ListenerCallback>,
) {
    while let Some(item) = receiver.recv().await {
        let notification = match item {
            Item::Barrier(done) => {
                let _ = done.send(());
                continue;
            }
            Item::Notification(notification) => notification,
        };
        if shared.is_stopping() {
            continue;
        }
        let acknowledged = notification.get("method").and_then(Value::as_str) == Some(ACKNOWLEDGED);
        let listened = notification
            .pointer("/params/notifications")
            .cloned()
            .unwrap_or_else(|| json!({}));
        let resuming = acknowledged && shared.acknowledged().is_some();
        dispatch(&shared.name, &on_notification, notification).await;
        if let Some(resumed) = resumed.as_ref().filter(|_| resuming) {
            dispatch(&shared.name, resumed, listened.clone()).await;
        }
        if acknowledged {
            shared.settle(State::Acknowledged(listened));
        }
    }
}

/// `dispatch`: a callback that panics is logged, and the listener keeps listening.
async fn dispatch(name: &str, callback: &ListenerCallback, argument: Value) {
    let run = std::panic::AssertUnwindSafe(callback(argument)).catch_unwind();
    if let Err(panic) = run.await {
        let message = panic
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| panic.downcast_ref::<String>().cloned())
            .unwrap_or_default();
        tracing::error!("{name} after_change callback failed: {message}");
    }
}
