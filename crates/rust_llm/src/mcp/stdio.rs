//! Port of `lib/ruby_llm/mcp/stdio.rb`. The server is a child process that reads one JSON-RPC
//! message per line on stdin and writes one per line on stdout. Reads never block past the
//! deadline, even on a partial line. It starts on the first request, restarts after it exits,
//! and handles one request at a time. Its stderr is the parent's. A server that predates
//! 2026-07-28 keeps its session for the life of its process, so after a restart its requests fail
//! with a session-expired error until the client initializes it again.
//!
//! Subscriptions share the channel, so whichever task reads a message that belongs to one hands
//! it to the subscription's listener: a request waiting for its answer, or the listener itself
//! while no request reads.

use std::path::PathBuf;
use std::process::Stdio as ProcessStdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::mpsc;

use super::client::{CHANGES, Client, VERSION};
use super::{McpError, OnNotification, Transport};
use crate::error::{Error, Result};

const SHUTDOWN_GRACE: Duration = Duration::from_secs(2);
const CHECK_INTERVAL: Duration = Duration::from_millis(500);
const LISTEN_INTERVAL: Duration = Duration::from_millis(50);
const SUBSCRIPTION_ID: &str = "io.modelcontextprotocol/subscriptionId";

/// `RubyLLM::MCP::Stdio`.
pub struct Stdio {
    command: Vec<String>,
    env: Vec<(String, String)>,
    directory: Option<PathBuf>,
    timeout: Duration,
    process: tokio::sync::Mutex<Option<Process>>,
    subscriptions: Mutex<Vec<Arc<Subscription>>>,
    started: AtomicU64,
    current: AtomicU64,
    session: AtomicU64,
}

struct Process {
    child: Child,
    stdin: ChildStdin,
    stdout: ChildStdout,
    buffer: Vec<u8>,
}

/// The messages of the subscription opened by request `id`, or with no `id`, the changes a
/// server that predates subscriptions announces.
struct Subscription {
    id: Option<Value>,
    messages: mpsc::UnboundedSender<Value>,
}

impl Subscription {
    fn claims(&self, message: &Value) -> bool {
        let method = message.get("method").and_then(Value::as_str);
        let Some(id) = &self.id else {
            return message.get("id").is_none() && method.is_some_and(|m| CHANGES.contains(&m));
        };
        let params = message.get("params").filter(|p| p.is_object());
        params.and_then(|p| p.pointer("/_meta").and_then(|m| m.get(SUBSCRIPTION_ID))) == Some(id)
            || self.is_answer(message)
            || self.is_cancelled_by(message)
    }

    fn is_answer(&self, message: &Value) -> bool {
        self.id.is_some()
            && message.get("id") == self.id.as_ref()
            && message.get("method").is_none()
    }

    fn is_cancelled_by(&self, message: &Value) -> bool {
        self.id.is_some()
            && message.get("method").and_then(Value::as_str) == Some("notifications/cancelled")
            && message.pointer("/params/requestId") == self.id.as_ref()
    }
}

fn is_alive(slot: &mut Option<Process>) -> bool {
    slot.as_mut()
        .is_some_and(|p| matches!(p.child.try_wait(), Ok(None)))
}

impl Stdio {
    pub fn new(
        command: Vec<String>,
        env: Vec<(String, String)>,
        directory: Option<PathBuf>,
        timeout: Duration,
    ) -> Stdio {
        Stdio {
            command,
            env,
            directory,
            timeout,
            process: tokio::sync::Mutex::new(None),
            subscriptions: Mutex::new(Vec::new()),
            started: AtomicU64::new(0),
            current: AtomicU64::new(0),
            session: AtomicU64::new(0),
        }
    }

    fn name(&self) -> String {
        let first = self.command.first().map(String::as_str).unwrap_or("");
        first.rsplit('/').next().unwrap_or(first).to_string()
    }

    fn exited(&self) -> Error {
        McpError::new(format!("{} exited", self.name())).into()
    }

    /// `check_session`: an older server's requests need the process its session started on.
    fn check_session(&self, slot: &mut Option<Process>, version: Option<&str>) -> Result<()> {
        if version.is_none_or(|v| v == VERSION) {
            return Ok(());
        }
        let current = self.current.load(Ordering::SeqCst);
        if is_alive(slot) && current != 0 && current == self.session.load(Ordering::SeqCst) {
            return Ok(());
        }
        Err(McpError::session_expired(format!("{} exited", self.name())).into())
    }

    fn start(&self) -> Result<Process> {
        let (program, args) = self
            .command
            .split_first()
            .ok_or_else(|| Error::Configuration("MCP command is empty".into()))?;
        let mut command = Command::new(program);
        command
            .args(args)
            .envs(self.env.iter().cloned())
            .stdin(ProcessStdio::piped())
            .stdout(ProcessStdio::piped())
            .kill_on_drop(true);
        if let Some(dir) = &self.directory {
            command.current_dir(dir);
        }
        let mut child = command
            .spawn()
            .map_err(|e| McpError::new(format!("{} could not start: {e}", self.name())))?;
        let stdin = child.stdin.take().ok_or_else(|| self.exited())?;
        let stdout = child.stdout.take().ok_or_else(|| self.exited())?;
        let generation = self.started.fetch_add(1, Ordering::SeqCst) + 1;
        self.current.store(generation, Ordering::SeqCst);
        Ok(Process {
            child,
            stdin,
            stdout,
            buffer: Vec::new(),
        })
    }

    async fn stop(&self, slot: &mut Option<Process>) {
        self.current.store(0, Ordering::SeqCst);
        stop(slot).await;
    }

    async fn write(&self, slot: &mut Option<Process>, message: &Value) -> Result<()> {
        if !is_alive(slot) {
            *slot = Some(self.start()?);
        }
        let Some(process) = slot.as_mut() else {
            return Err(self.exited());
        };
        let mut line = message.to_string();
        line.push('\n');
        let written = async {
            process.stdin.write_all(line.as_bytes()).await?;
            process.stdin.flush().await
        }
        .await;
        if written.is_err() {
            self.stop(slot).await;
            return Err(self.exited());
        }
        Ok(())
    }

    async fn read(&self, slot: &mut Option<Process>, deadline: Instant) -> Result<Value> {
        loop {
            let line = self.next_line(slot, deadline).await?;
            if let Some(message) = self.parse(&line) {
                return Ok(message);
            }
        }
    }

    fn parse(&self, line: &[u8]) -> Option<Value> {
        let text = String::from_utf8_lossy(line);
        if text.trim().is_empty() {
            return None;
        }
        match serde_json::from_str::<Value>(&text) {
            Ok(value) if value.is_object() => Some(value),
            _ => {
                tracing::debug!("{} wrote a line that is not JSON", self.name());
                None
            }
        }
    }

    async fn next_line(&self, slot: &mut Option<Process>, deadline: Instant) -> Result<Vec<u8>> {
        loop {
            let Some(process) = slot.as_mut() else {
                return Err(self.exited());
            };
            if let Some(pos) = process.buffer.iter().position(|b| *b == b'\n') {
                return Ok(process.buffer.drain(..=pos).collect());
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(
                    McpError::new(format!("{} did not answer in time", self.name())).into(),
                );
            }
            if crate::progress::is_cancelled() {
                return Err(Error::Cancelled);
            }
            let mut chunk = vec![0u8; 65_536];
            let read = tokio::time::timeout(
                remaining.min(CHECK_INTERVAL),
                process.stdout.read(&mut chunk),
            )
            .await;
            match read {
                Err(_) => continue,
                Ok(Ok(0)) | Ok(Err(_)) => {
                    self.stop(slot).await;
                    return Err(self.exited());
                }
                Ok(Ok(n)) => process.buffer.extend_from_slice(&chunk[..n]),
            }
        }
    }

    /// `route`: hands a message to the subscription it belongs to.
    fn route(&self, message: &Value) -> bool {
        let Ok(subscriptions) = self.subscriptions.lock() else {
            return false;
        };
        match subscriptions.iter().find(|s| s.claims(message)) {
            Some(subscription) => {
                let _ = subscription.messages.send(message.clone());
                true
            }
            None => false,
        }
    }

    /// `handled?`: routes a message, or answers a request the server sends.
    async fn handle(&self, slot: &mut Option<Process>, message: &Value) -> Result<bool> {
        if self.route(message) {
            return Ok(true);
        }
        if message.get("method").is_none() || message.get("id").is_none() {
            return Ok(false);
        }
        self.write(slot, &Client::reply(message)).await?;
        Ok(true)
    }

    /// `drain`: reads what the server wrote while no request reads, for the subscriptions.
    async fn drain(&self, slot: &mut Option<Process>) -> Result<()> {
        loop {
            let message = {
                let Some(process) = slot.as_mut() else {
                    return Err(self.exited());
                };
                match process.buffer.iter().position(|b| *b == b'\n') {
                    Some(pos) => Some(process.buffer.drain(..=pos).collect::<Vec<u8>>()),
                    None => None,
                }
            };
            if let Some(line) = message {
                if let Some(message) = self.parse(&line) {
                    self.handle(slot, &message).await?;
                }
                continue;
            }
            let Some(process) = slot.as_mut() else {
                return Err(self.exited());
            };
            let mut chunk = vec![0u8; 65_536];
            match tokio::time::timeout(LISTEN_INTERVAL, process.stdout.read(&mut chunk)).await {
                Err(_) => return Ok(()),
                Ok(Ok(0)) | Ok(Err(_)) => {
                    self.stop(slot).await;
                    return Err(self.exited());
                }
                Ok(Ok(n)) => process.buffer.extend_from_slice(&chunk[..n]),
            }
        }
    }

    async fn subscribe(
        &self,
        subscription: &Arc<Subscription>,
        message: Option<&Value>,
    ) -> Result<u64> {
        let mut slot = self.process.lock().await;
        if message.is_none() && !is_alive(&mut slot) {
            return Err(self.exited());
        }
        if let Ok(mut subscriptions) = self.subscriptions.lock() {
            subscriptions.push(subscription.clone());
        }
        if let Some(message) = message {
            self.write(&mut slot, message).await?;
        }
        Ok(self.current.load(Ordering::SeqCst))
    }

    async fn follow(
        &self,
        subscription: &Subscription,
        messages: &mut mpsc::UnboundedReceiver<Value>,
        generation: u64,
        on_notification: &mut OnNotification<'_>,
    ) -> Result<Option<Value>> {
        loop {
            while let Ok(reply) = messages.try_recv() {
                if subscription.is_answer(&reply) || subscription.is_cancelled_by(&reply) {
                    return Ok(Some(reply));
                }
                on_notification(&reply);
            }
            if crate::progress::is_cancelled() {
                return Err(Error::Cancelled);
            }
            if self.current.load(Ordering::SeqCst) != generation {
                return Err(self.exited());
            }
            match self.process.try_lock() {
                Ok(mut slot) => self.drain(&mut slot).await?,
                Err(_) => {
                    if let Ok(Some(reply)) =
                        tokio::time::timeout(LISTEN_INTERVAL, messages.recv()).await
                    {
                        if subscription.is_answer(&reply) || subscription.is_cancelled_by(&reply) {
                            return Ok(Some(reply));
                        }
                        on_notification(&reply);
                    }
                }
            }
        }
    }
}

async fn stop(slot: &mut Option<Process>) {
    let Some(Process {
        mut child,
        stdin,
        stdout,
        ..
    }) = slot.take()
    else {
        return;
    };
    drop(stdin);
    drop(stdout);
    if tokio::time::timeout(SHUTDOWN_GRACE, child.wait())
        .await
        .is_err()
    {
        let _ = child.start_kill();
        let _ = tokio::time::timeout(SHUTDOWN_GRACE, child.wait()).await;
    }
}

#[async_trait]
impl Transport for Stdio {
    async fn request(
        &self,
        message: &Value,
        version: Option<&str>,
        timeout: Option<Duration>,
        _headers: &[(String, String)],
        on_notification: &mut OnNotification<'_>,
    ) -> Result<Value> {
        let mut slot = self.process.lock().await;
        self.check_session(&mut slot, version)?;
        self.write(&mut slot, message).await?;
        if message.get("method").and_then(Value::as_str) == Some("initialize") {
            self.session
                .store(self.current.load(Ordering::SeqCst), Ordering::SeqCst);
        }
        let deadline = Instant::now() + timeout.unwrap_or(self.timeout);
        let id = message.get("id");
        loop {
            let reply = self.read(&mut slot, deadline).await?;
            let has_method = reply.get("method").is_some();
            if reply.get("id") == id && !has_method {
                return Ok(reply);
            }
            if !self.handle(&mut slot, &reply).await? && has_method {
                on_notification(&reply);
            }
        }
    }

    async fn notify(&self, message: &Value, _version: Option<&str>) -> Result<()> {
        let mut slot = self.process.lock().await;
        self.write(&mut slot, message).await
    }

    async fn cancel(&self, notification: &Value, version: Option<&str>) -> Result<()> {
        self.notify(notification, version).await
    }

    async fn close(&self) {
        let mut slot = self.process.lock().await;
        self.stop(&mut slot).await;
    }

    async fn listen(
        &self,
        message: Option<&Value>,
        _version: Option<&str>,
        on_notification: &mut OnNotification<'_>,
    ) -> Result<Option<Value>> {
        let (sender, mut messages) = mpsc::unbounded_channel();
        let subscription = Arc::new(Subscription {
            id: message.and_then(|m| m.get("id").cloned()),
            messages: sender,
        });
        let result = match self.subscribe(&subscription, message).await {
            Ok(generation) => {
                self.follow(&subscription, &mut messages, generation, on_notification)
                    .await
            }
            Err(e) => Err(e),
        };
        if let Ok(mut subscriptions) = self.subscriptions.lock() {
            subscriptions.retain(|s| !Arc::ptr_eq(s, &subscription));
        }
        result
    }
}
