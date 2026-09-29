//! Port of `lib/ruby_llm/mcp/prompt.rb`: messages written by the server and filled in with your
//! arguments. Ask a chat with one like a question (`Chat::ask_prompt`).

use serde_json::{Value, json};

use super::Mcp;
use crate::error::Result;
use crate::message::Message;

/// An argument a prompt takes (`Prompt::Argument`).
#[derive(Debug, Clone, PartialEq)]
pub struct PromptArgument {
    pub name: String,
    pub description: Option<String>,
    /// `required?`.
    pub required: bool,
}

/// `RubyLLM::MCP::Prompt`.
#[derive(Clone)]
pub struct Prompt {
    pub name: String,
    pub title: Option<String>,
    pub description: Option<String>,
    pub arguments: Vec<PromptArgument>,
    /// The filled-in messages. A prompt from `Mcp::prompts` has none until `Mcp::prompt`.
    pub messages: Vec<Message>,
    mcp: Mcp,
}

impl std::fmt::Debug for Prompt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Prompt").field("name", &self.name).field("messages", &self.messages.len()).finish()
    }
}

impl Prompt {
    pub(crate) fn new(mcp: Mcp, data: &Value, messages: Vec<Message>) -> Prompt {
        let s = |key: &str| data.get(key).and_then(Value::as_str).map(str::to_string);
        let arguments = data
            .get("arguments")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .map(|a| PromptArgument {
                name: a.get("name").and_then(Value::as_str).unwrap_or("").to_string(),
                description: a.get("description").and_then(Value::as_str).map(str::to_string),
                required: a.get("required") == Some(&Value::Bool(true)),
            })
            .collect();
        Prompt { name: s("name").unwrap_or_default(), title: s("title"), description: s("description"), arguments, messages, mcp }
    }

    /// `suggest(**arguments)`: asks the server to complete the first argument's partial value,
    /// with the rest as context.
    pub async fn suggest(&self, arguments: &[(&str, &str)]) -> Result<Vec<String>> {
        self.mcp.suggest(json!({ "type": "ref/prompt", "name": self.name }), arguments).await
    }
}
