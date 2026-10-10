//! Port of `lib/ruby_llm/mcp/task.rb`: a tool call that an MCP server runs in the background,
//! as the Tasks extension describes. Declare `McpBuilder::extension(Extension::Tasks, ...)`, and
//! a server may answer a long call with a task instead of a result.
//!
//! A chat never waits for a task. It pauses the tool call, and `Chat::pending_tasks` returns the
//! tasks it waits on. [`Task::refresh`] checks on a task once; `Chat::complete` checks on every
//! one again and resumes the chat once they are done. `Mcp::call` waits for the task of a tool
//! you call directly.

use std::time::Duration;

use chrono::{DateTime, Utc};
use serde_json::{Value, json};

use super::{Mcp, McpError, McpResult};
use crate::error::{Error, Result};
use crate::message::ToolCall;

/// A task's state (`Task#status`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskStatus {
    Working,
    InputRequired,
    Completed,
    Failed,
    Cancelled,
    /// A status the protocol does not define, as the server sent it.
    Other(String),
}

impl TaskStatus {
    fn parse(status: &str) -> TaskStatus {
        match status {
            "working" => TaskStatus::Working,
            "input_required" => TaskStatus::InputRequired,
            "completed" => TaskStatus::Completed,
            "failed" => TaskStatus::Failed,
            "cancelled" => TaskStatus::Cancelled,
            other => TaskStatus::Other(other.to_string()),
        }
    }

    fn as_str(&self) -> &str {
        match self {
            TaskStatus::Working => "working",
            TaskStatus::InputRequired => "input_required",
            TaskStatus::Completed => "completed",
            TaskStatus::Failed => "failed",
            TaskStatus::Cancelled => "cancelled",
            TaskStatus::Other(other) => other,
        }
    }
}

/// `RubyLLM::MCP::Task`.
#[derive(Clone)]
pub struct Task {
    mcp: Option<Mcp>,
    data: Value,
    /// The task's id on its server.
    pub id: String,
    answered: Vec<String>,
    /// The tool call paused on the task, when it came from a chat.
    pub tool_call: Option<ToolCall>,
}

impl std::fmt::Debug for Task {
    /// `#<RubyLLM::MCP::Task id: "task-1", status: :working, status_message: "Queued">`
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Task")
            .field("id", &self.id)
            .field("status", &self.status())
            .field("status_message", &self.status_message())
            .finish()
    }
}

impl Task {
    pub(crate) fn new(mcp: Option<Mcp>, data: Value) -> Task {
        let id = data
            .get("taskId")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        Task {
            mcp,
            data,
            id,
            answered: Vec::new(),
            tool_call: None,
        }
    }

    /// `Task.load(mcp, state, tool_call:)`: the task a paused call saved.
    pub(crate) fn load(mcp: Option<Mcp>, state: &Value, tool_call: Option<ToolCall>) -> Task {
        let mut task = Task::new(mcp, state.get("task").cloned().unwrap_or(Value::Null));
        task.answered = state
            .get("answered")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|k| k.as_str().map(str::to_string))
            .collect();
        task.tool_call = tool_call;
        task
    }

    /// The name of the MCP that runs the task, or `"MCP"` when none is connected.
    pub fn server_name(&self) -> String {
        self.mcp.as_ref().map_or_else(|| "MCP".into(), Mcp::name)
    }

    /// The task as the server last described it.
    pub fn data(&self) -> &Value {
        &self.data
    }

    pub(crate) fn answered(&self) -> &[String] {
        &self.answered
    }

    /// `status`.
    pub fn status(&self) -> TaskStatus {
        TaskStatus::parse(
            self.data
                .get("status")
                .and_then(Value::as_str)
                .unwrap_or(""),
        )
    }

    /// `status_message`: what the server says about the task's state, such as how far it got.
    pub fn status_message(&self) -> Option<&str> {
        self.data.get("statusMessage").and_then(Value::as_str)
    }

    /// `poll_interval`: how long the server asks you to wait before checking on the task again.
    pub fn poll_interval(&self) -> Option<Duration> {
        let ms = self.data.get("pollIntervalMs").and_then(Value::as_f64)?;
        Some(Duration::from_secs_f64(ms / 1000.0))
    }

    /// `expires_at`: when the server may forget the task, or `None` when it keeps it for good.
    pub fn expires_at(&self) -> Option<DateTime<Utc>> {
        let ttl = self.data.get("ttlMs").and_then(Value::as_f64)?;
        let created = self.data.get("createdAt").and_then(Value::as_str)?;
        let created = DateTime::parse_from_rfc3339(created).ok()?;
        Some(created.with_timezone(&Utc) + chrono::Duration::milliseconds(ttl as i64))
    }

    /// `done?`: the task completed, failed, or was cancelled.
    pub fn is_done(&self) -> bool {
        matches!(
            self.status(),
            TaskStatus::Completed | TaskStatus::Failed | TaskStatus::Cancelled
        )
    }

    /// `completed?`: the task finished with a result.
    pub fn is_completed(&self) -> bool {
        self.status() == TaskStatus::Completed
    }

    /// `failed?`.
    pub fn is_failed(&self) -> bool {
        self.status() == TaskStatus::Failed
    }

    /// `cancelled?`.
    pub fn is_cancelled(&self) -> bool {
        self.status() == TaskStatus::Cancelled
    }

    /// `result`: the result once the task completed, `None` until then. Fails with `Error::Mcp`
    /// when the task failed or was cancelled.
    pub fn result(&self) -> Result<Option<McpResult>> {
        if self.is_failed() || self.is_cancelled() {
            return Err(self.error().into());
        }
        Ok(self.is_completed().then(|| {
            McpResult::new(
                self.data
                    .get("result")
                    .cloned()
                    .unwrap_or_else(|| json!({})),
                None,
            )
        }))
    }

    /// `refresh`: checks on the task once. Does nothing once the task is done.
    pub async fn refresh(&mut self) -> Result<&mut Self> {
        if !self.is_done() {
            self.data = self.connected()?.poll_task(&self.id).await?;
        }
        Ok(self)
    }

    /// `wait(timeout:, interval:)`: checks on the task until it is done, sleeping
    /// `poll_interval` in between, answering the server's requests for input with
    /// `before_input_request` callbacks. `timeout` defaults to the MCP's timeout. Fails with
    /// `Error::Mcp` when the task fails or the time runs out, and `Error::McpInputRequired` when
    /// no callback answers a request.
    pub async fn wait(
        &mut self,
        timeout: Option<Duration>,
        interval: Option<Duration>,
    ) -> Result<&mut Self> {
        let mcp = self.connected()?;
        mcp.await_task(self, timeout, interval).await?;
        Ok(self)
    }

    /// `cancel`: asks the server to cancel the task. The server may still finish it, so check
    /// with [`Task::refresh`].
    pub async fn cancel(&mut self) -> Result<&mut Self> {
        if !self.is_done() {
            self.connected()?.cancel_task(&self.id).await?;
        }
        Ok(self)
    }

    /// `error`: why the task failed or was cancelled.
    pub fn error(&self) -> McpError {
        let details = self.data.get("error");
        let message = details
            .and_then(|d| d.get("message"))
            .and_then(Value::as_str)
            .or(self.status_message())
            .map(str::to_string)
            .unwrap_or_else(|| format!("Task {} was {}", self.id, self.status().as_str()));
        McpError {
            message,
            code: details.and_then(|d| d.get("code")).and_then(Value::as_i64),
            data: details.and_then(|d| d.get("data")).cloned(),
            ..Default::default()
        }
    }

    pub(crate) fn record_answers(&mut self, keys: impl IntoIterator<Item = String>) {
        for key in keys {
            if !self.answered.contains(&key) {
                self.answered.push(key);
            }
        }
    }

    /// `to_h`: what a paused call saves.
    pub fn to_h(&self) -> Value {
        let mut h = json!({ "task": self.data });
        if !self.answered.is_empty() {
            h["answered"] = json!(self.answered);
        }
        h
    }

    fn connected(&self) -> Result<Mcp> {
        self.mcp.clone().ok_or_else(|| {
            Error::Configuration(format!(
                "Connect the MCP that runs task {} with with_mcp to reach it",
                self.id
            ))
        })
    }
}
