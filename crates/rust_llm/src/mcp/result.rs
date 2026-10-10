//! Port of `lib/ruby_llm/mcp/result.rb`.

use serde_json::{Map, Value, json};

use super::content;
use crate::attachment::Attachment;
use crate::tool::ToolResult;

/// `RubyLLM::MCP::Result`: what an MCP tool returned. Text, files, and structured data each have
/// their own reader. In a chat, the result of a tool with a UI stays on its tool result message
/// as `Message::mcp_result`, so your app can render the UI again.
#[derive(Debug, Clone, PartialEq)]
pub struct McpResult {
    /// The text of the result, with text blocks joined by blank lines.
    pub text: String,
    /// Images, audio, and embedded files.
    pub attachments: Vec<Attachment>,
    /// The structured content, or `None`.
    pub structured: Option<Value>,
    /// The result's `_meta` as the server sent it, empty when there is none.
    pub meta: Map<String, Value>,
    /// The URI of the UI that renders the result, from its tool's `McpTool::ui_uri`, or `None`
    /// for a tool without one.
    pub ui_uri: Option<String>,
    data: Value,
}

impl McpResult {
    /// `Result.new(data, ui_uri)`.
    pub fn new(data: Value, ui_uri: Option<String>) -> McpResult {
        let blocks = data
            .get("content")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let (text, attachments) = content::read(&blocks);
        McpResult {
            text,
            attachments,
            structured: data.get("structuredContent").cloned(),
            meta: data
                .get("_meta")
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default(),
            ui_uri,
            data,
        }
    }

    /// `Result.load(data)`: a result saved with [`McpResult::dump`].
    pub fn load(data: &Value) -> McpResult {
        McpResult::new(
            data.get("result").cloned().unwrap_or_else(|| json!({})),
            data.get("ui_uri")
                .and_then(Value::as_str)
                .map(str::to_string),
        )
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

    /// `dump`: what a message keeps across serialization.
    pub fn dump(&self) -> Value {
        json!({ "ui_uri": self.ui_uri, "result": self.data })
    }
}
