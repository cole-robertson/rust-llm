//! Port of `lib/ruby_llm/mcp/stdio.rb`. The server is a child process that reads one JSON-RPC
//! message per line on stdin and writes one per line on stdout. Reads never block past the
//! deadline, even on a partial line. It starts on the first request, restarts after it exits,
//! and handles one request at a time. Its stderr is the parent's.

use std::path::PathBuf;
use std::process::Stdio as ProcessStdio;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};

use super::{McpError, OnNotification, Transport};
use crate::error::{Error, Result};

const SHUTDOWN_GRACE: Duration = Duration::from_secs(2);
const CHECK_INTERVAL: Duration = Duration::from_millis(500);

/// `RubyLLM::MCP::Stdio`.
pub struct Stdio {
    command: Vec<String>,
    env: Vec<(String, String)>,
    directory: Option<PathBuf>,
    timeout: Duration,
    process: tokio::sync::Mutex<Option<Process>>,
}

struct Process {
    child: Child,
    stdin: ChildStdin,
    stdout: ChildStdout,
    buffer: Vec<u8>,
}

impl Stdio {
    pub fn new(command: Vec<String>, env: Vec<(String, String)>, directory: Option<PathBuf>, timeout: Duration) -> Stdio {
        Stdio { command, env, directory, timeout, process: tokio::sync::Mutex::new(None) }
    }

    fn name(&self) -> String {
        let first = self.command.first().map(String::as_str).unwrap_or("");
        first.rsplit('/').next().unwrap_or(first).to_string()
    }

    fn exited(&self) -> Error {
        McpError::new(format!("{} exited", self.name())).into()
    }

    fn start(&self) -> Result<Process> {
        let (program, args) = self.command.split_first().ok_or_else(|| Error::Configuration("MCP command is empty".into()))?;
        let mut command = Command::new(program);
        command.args(args).envs(self.env.iter().cloned()).stdin(ProcessStdio::piped()).stdout(ProcessStdio::piped()).kill_on_drop(true);
        if let Some(dir) = &self.directory {
            command.current_dir(dir);
        }
        let mut child = command.spawn().map_err(|e| McpError::new(format!("{} could not start: {e}", self.name())))?;
        let stdin = child.stdin.take().ok_or_else(|| self.exited())?;
        let stdout = child.stdout.take().ok_or_else(|| self.exited())?;
        Ok(Process { child, stdin, stdout, buffer: Vec::new() })
    }

    async fn write(&self, slot: &mut Option<Process>, message: &Value) -> Result<()> {
        let alive = match slot.as_mut() {
            Some(p) => matches!(p.child.try_wait(), Ok(None)),
            None => false,
        };
        if !alive {
            *slot = Some(self.start()?);
        }
        let Some(process) = slot.as_mut() else { return Err(self.exited()) };
        let mut line = message.to_string();
        line.push('\n');
        let written = async {
            process.stdin.write_all(line.as_bytes()).await?;
            process.stdin.flush().await
        }
        .await;
        if written.is_err() {
            stop(slot).await;
            return Err(self.exited());
        }
        Ok(())
    }

    async fn read(&self, slot: &mut Option<Process>, deadline: Instant) -> Result<Value> {
        loop {
            let line = self.next_line(slot, deadline).await?;
            let text = String::from_utf8_lossy(&line);
            if text.trim().is_empty() {
                continue;
            }
            match serde_json::from_str(&text) {
                Ok(value) => return Ok(value),
                Err(_) => tracing::debug!("{} wrote a line that is not JSON", self.name()),
            }
        }
    }

    async fn next_line(&self, slot: &mut Option<Process>, deadline: Instant) -> Result<Vec<u8>> {
        loop {
            let Some(process) = slot.as_mut() else { return Err(self.exited()) };
            if let Some(pos) = process.buffer.iter().position(|b| *b == b'\n') {
                return Ok(process.buffer.drain(..=pos).collect());
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(McpError::new(format!("{} did not answer in time", self.name())).into());
            }
            if crate::progress::is_cancelled() {
                return Err(Error::Cancelled);
            }
            let mut chunk = vec![0u8; 65_536];
            let read = tokio::time::timeout(remaining.min(CHECK_INTERVAL), process.stdout.read(&mut chunk)).await;
            match read {
                Err(_) => continue,
                Ok(Ok(0)) | Ok(Err(_)) => {
                    stop(slot).await;
                    return Err(self.exited());
                }
                Ok(Ok(n)) => process.buffer.extend_from_slice(&chunk[..n]),
            }
        }
    }

    /// Answers a request the server sends while it works: `ping`, or "method not found".
    async fn answer(&self, slot: &mut Option<Process>, request: &Value) -> Result<()> {
        let id = request.get("id").cloned().unwrap_or(Value::Null);
        let reply = if request.get("method").and_then(Value::as_str) == Some("ping") {
            json!({ "jsonrpc": "2.0", "id": id, "result": {} })
        } else {
            json!({ "jsonrpc": "2.0", "id": id, "error": { "code": -32_601, "message": "Method not found" } })
        };
        self.write(slot, &reply).await
    }
}

async fn stop(slot: &mut Option<Process>) {
    let Some(Process { mut child, stdin, stdout, .. }) = slot.take() else { return };
    drop(stdin);
    drop(stdout);
    if tokio::time::timeout(SHUTDOWN_GRACE, child.wait()).await.is_err() {
        let _ = child.start_kill();
        let _ = tokio::time::timeout(SHUTDOWN_GRACE, child.wait()).await;
    }
}

#[async_trait]
impl Transport for Stdio {
    async fn request(
        &self,
        message: &Value,
        _version: Option<&str>,
        timeout: Option<Duration>,
        _headers: &[(String, String)],
        on_notification: &mut OnNotification<'_>,
    ) -> Result<Value> {
        let mut slot = self.process.lock().await;
        self.write(&mut slot, message).await?;
        let deadline = Instant::now() + timeout.unwrap_or(self.timeout);
        let id = message.get("id");
        loop {
            let reply = self.read(&mut slot, deadline).await?;
            let has_method = reply.get("method").is_some();
            if reply.get("id") == id && !has_method {
                return Ok(reply);
            }
            if has_method && reply.get("id").is_some() {
                self.answer(&mut slot, &reply).await?;
            } else if has_method {
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
        stop(&mut slot).await;
    }
}
