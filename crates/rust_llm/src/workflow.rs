//! Port of `lib/ruby_llm/workflow.rb` and `RubyLLM.workflow`: named, instrumented regions of
//! ordinary async code.
//!
//! ```ruby
//! RubyLLM.workflow("Write article", id: "article-42") do |workflow|
//!   notes = workflow.step("Research") { ResearchAgent.new.ask(topic).content }
//!   workflow.step("Draft") { WriterAgent.new.ask(notes).content }
//! end
//! ```
//!
//! ```ignore
//! rust_llm::workflow("Write article", Some("article-42"), None, |wf| async move {
//!     let notes = wf.step("Research", None, async { researcher.ask(topic).await }).await?;
//!     wf.step("Draft", None, async { writer.ask(notes.content()).await }).await
//! }).await?;
//! ```
//!
//! Ruby keeps the context in a thread-local; here it is a tokio task-local, so it follows the
//! future. Work you `tokio::spawn` starts outside it: wrap the spawned future in its own
//! `step`, as Ruby's docs say to do for `Async` tasks.

use std::future::Future;
use std::sync::Arc;

use serde_json::{Map, Value};

use crate::config::Config;
use crate::error::{Error, Result};
use crate::instrumentation::{self, current_workflow, with_workflow};

/// `RubyLLM::Workflow`. Cheap to clone; clones share the identity.
#[derive(Debug, Clone)]
pub struct Workflow {
    id: String,
    name: String,
    metadata: Option<Value>,
    config: Arc<Config>,
}

/// `RubyLLM.workflow(name, id:, metadata:) { |workflow| ... }`: runs `body` as a workflow and
/// returns its result. `id` defaults to a UUID. Every `*.rust_llm` event inside carries
/// `workflow_id` and `workflow_name` (and `workflow_metadata` when given).
pub async fn workflow<T, F, Fut>(
    name: &str,
    id: Option<&str>,
    metadata: Option<Value>,
    body: F,
) -> Result<T>
where
    F: FnOnce(Workflow) -> Fut,
    Fut: Future<Output = Result<T>>,
{
    Workflow::new(name, id, metadata, crate::config())?
        .run(body)
        .await
}

impl Workflow {
    /// `Workflow.new(name, id:, metadata:, config:)`.
    pub fn new(
        name: &str,
        id: Option<&str>,
        metadata: Option<Value>,
        config: Arc<Config>,
    ) -> Result<Workflow> {
        let id = id
            .map(str::to_string)
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        Ok(Workflow {
            name: normalize(name, "name")?,
            id: normalize(&id, "id")?,
            metadata,
            config,
        })
    }

    /// The identifier shared by the workflow's events.
    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    fn context(&self) -> Map<String, Value> {
        let mut context = Map::new();
        context.insert("workflow_id".into(), self.id.clone().into());
        context.insert("workflow_name".into(), self.name.clone().into());
        if let Some(m) = &self.metadata {
            context.insert("workflow_metadata".into(), m.clone());
        }
        context
    }

    /// `Workflow#run`: links to the enclosing workflow (and step) when there is a different one,
    /// recomputed on every run, then emits `workflow.rust_llm` around `body`.
    pub async fn run<T, F, Fut>(&self, body: F) -> Result<T>
    where
        F: FnOnce(Workflow) -> Fut,
        Fut: Future<Output = Result<T>>,
    {
        let mut context = self.context();
        if let Some(current) = current_workflow()
            .filter(|c| c.get("workflow_id").and_then(Value::as_str) != Some(&self.id))
        {
            if let Some(parent) = current.get("workflow_id") {
                context.insert("workflow_parent_id".into(), parent.clone());
            }
            if let Some(step) = current.get("workflow_step_id") {
                context.insert("workflow_parent_step_id".into(), step.clone());
            }
        }
        let config = self.config.clone();
        let this = self.clone();
        // Boxed so nested workflows do not inline into one ever-larger future on the stack.
        Box::pin(with_workflow(context, async move {
            instrumentation::instrument(
                &config,
                "workflow.rust_llm",
                Map::new(),
                Box::pin(body(this)),
            )
            .await
        }))
        .await
    }

    /// `Workflow#step(name, id:)`: runs `body` as a named step and returns its result. Steps nest:
    /// an inner step records `workflow_step_parent_id`.
    pub async fn step<T>(
        &self,
        name: &str,
        id: Option<&str>,
        body: impl Future<Output = Result<T>>,
    ) -> Result<T> {
        let step_id = id
            .map(str::to_string)
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let current = current_workflow();
        let mut context = current
            .clone()
            .filter(|c| c.get("workflow_id").and_then(Value::as_str) == Some(&self.id))
            .unwrap_or_else(|| self.context());
        let parent = current
            .filter(|c| c.get("workflow_id").and_then(Value::as_str) == Some(&self.id))
            .and_then(|c| c.get("workflow_step_id").cloned());
        context.remove("workflow_step_parent_id");
        context.insert(
            "workflow_step_id".into(),
            normalize(&step_id, "step id")?.into(),
        );
        context.insert(
            "workflow_step_name".into(),
            normalize(name, "step name")?.into(),
        );
        if let Some(parent) = parent {
            context.insert("workflow_step_parent_id".into(), parent);
        }
        let config = self.config.clone();
        Box::pin(with_workflow(context, async move {
            instrumentation::instrument(
                &config,
                "workflow_step.rust_llm",
                Map::new(),
                Box::pin(body),
            )
            .await
        }))
        .await
    }
}

fn normalize(value: &str, attribute: &str) -> Result<String> {
    if value.is_empty() {
        return Err(Error::Argument(format!(
            "workflow {attribute} cannot be empty"
        )));
    }
    Ok(value.to_string())
}
