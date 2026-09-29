//! Port of `lib/ruby_llm/mcp/error.rb`.

use serde_json::Value;

use crate::error::ErrorResponse;

/// `RubyLLM::MCP::Error`: an MCP server answered with a JSON-RPC error, an unexpected HTTP
/// status, or not at all. Surfaces as `rust_llm::Error::Mcp`.
#[derive(Debug, Clone, Default)]
pub struct McpError {
    pub message: String,
    /// The JSON-RPC error code, or `None` when the failure had none.
    pub code: Option<i64>,
    /// Additional information the server attached to the error.
    pub data: Option<Value>,
    pub response: Option<ErrorResponse>,
}

impl McpError {
    pub fn new(message: impl Into<String>) -> McpError {
        McpError { message: message.into(), ..Default::default() }
    }
}

impl std::fmt::Display for McpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for McpError {}

impl From<McpError> for crate::Error {
    fn from(error: McpError) -> Self {
        crate::Error::Mcp(Box::new(error))
    }
}
