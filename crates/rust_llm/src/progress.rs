//! Port of `lib/ruby_llm/progress.rb`, `support/progress_reporter.rb`, and
//! `support/cancellation.rb`.
//!
//! RubyLLM keeps the chat's progress listener and cancellation checkpoint in fiber/thread storage
//! while a tool runs, so code deep inside the tool (an MCP transport waiting on a server) can
//! report progress and notice a cancelled chat. Here the same state lives in tokio task-locals
//! scoped around the tool's future.

use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// `RubyLLM::Progress`: what a tool reports while it works.
#[derive(Debug, Clone, PartialEq)]
pub struct Progress {
    pub value: Option<f64>,
    pub total: Option<f64>,
    pub message: Option<String>,
}

impl Progress {
    /// `Progress#fraction`: `value / total` when both are known and `total` is positive.
    pub fn fraction(&self) -> Option<f64> {
        match (self.value, self.total) {
            (Some(value), Some(total)) if total > 0.0 => Some(value / total),
            _ => None,
        }
    }
}

/// Receives progress reports (`Chat#after_tool_progress`, `MCP.after_progress`).
pub type Listener = Arc<dyn Fn(&Progress) + Send + Sync>;

tokio::task_local! {
    static LISTENER: Option<Listener>;
    static CANCELLATION: Arc<AtomicBool>;
}

/// `ProgressReporter.listen`: runs `future` with `listener` receiving its progress reports.
pub async fn listen<F: Future>(listener: Option<Listener>, future: F) -> F::Output {
    LISTENER.scope(listener, future).await
}

/// `ProgressReporter.listener`: the listener of the running tool call, if any.
pub fn listener() -> Option<Listener> {
    LISTENER.try_with(Clone::clone).ok().flatten()
}

/// `ProgressReporter.report` (`Tool#progress`): sends a report to the running tool call's listener.
pub fn report(progress: Progress) {
    if let Some(listener) = listener() {
        listener(&progress);
    }
}

/// `Cancellation.watch`: runs `future` with `flag` as its cancellation checkpoint.
pub async fn watch<F: Future>(flag: Arc<AtomicBool>, future: F) -> F::Output {
    CANCELLATION.scope(flag, future).await
}

/// `Cancellation.check`: whether the surrounding chat has been cancelled. Long waits (an MCP
/// server that has not answered yet) poll this and stop with `Error::Cancelled`.
pub fn is_cancelled() -> bool {
    CANCELLATION
        .try_with(|flag| flag.load(Ordering::SeqCst))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fraction_needs_a_positive_total() {
        let progress = |value, total| Progress {
            value,
            total,
            message: None,
        };
        assert_eq!(progress(Some(1.0), Some(2.0)).fraction(), Some(0.5));
        assert_eq!(progress(Some(1.0), Some(0.0)).fraction(), None);
        assert_eq!(progress(None, Some(2.0)).fraction(), None);
    }
}
