//! Port of `lib/ruby_llm/mcp/result.rb`.

use serde_json::Value;

use super::content;
use crate::attachment::Attachment;
use crate::tool::ToolResult;

/// `RubyLLM::MCP::Result`: what an MCP tool returned. Text, files, and structured data each have
/// their own reader.
#[derive(Debug, Clone, PartialEq)]
pub struct McpResult {
    /// The text of the result, with text blocks joined by blank lines.
    pub text: String,
    /// Images, audio, and embedded files.
    pub attachments: Vec<Attachment>,
    /// The structured content, or `None`.
    pub structured: Option<Value>,
    data: Value,
}

impl McpResult {
    pub fn new(data: Value) -> McpResult {
        let blocks = data.get("content").and_then(Value::as_array).cloned().unwrap_or_default();
        let (text, attachments) = content::read(&blocks);
        McpResult { text, attachments, structured: data.get("structuredContent").cloned(), data }
    }

    /// `error?`: whether the tool reported a failure.
    pub fn is_error(&self) -> bool {
        self.data.get("isError") == Some(&Value::Bool(true))
    }

    /// `content`: what a chat sends to the model, the text or, when there is none, the structured
    /// content as JSON, followed by any attachments.
    pub fn content(&self) -> ToolResult {
        let body = match &self.structured {
            Some(structured) if self.text.is_empty() => structured.to_string(),
            _ => self.text.clone(),
        };
        ToolResult::with_attachments(body, self.attachments.clone())
    }

    /// `to_h`: the result as the server sent it.
    pub fn to_h(&self) -> &Value {
        &self.data
    }
}
