//! Port of `lib/ruby_llm/mcp/apps.rb`: reads what the MCP Apps extension keeps in a tool's
//! `_meta`, the UI that renders the tool's results and who may call the tool. Servers on older
//! SDKs name the UI with the flat `ui/resourceUri` key the spec deprecated.

use serde_json::{Map, Value};

pub const EXTENSION: &str = "io.modelcontextprotocol/ui";
pub const MIME_TYPE: &str = "text/html;profile=mcp-app";
const VISIBILITY: [&str; 2] = ["model", "app"];

/// `Apps.uri(meta)`.
pub fn uri(meta: &Map<String, Value>) -> Option<String> {
    meta.get("ui")
        .and_then(|ui| ui.get("resourceUri"))
        .or_else(|| meta.get("ui/resourceUri"))
        .and_then(Value::as_str)
        .map(str::to_string)
}

/// `Apps.visibility(meta)`: `["model", "app"]` unless the tool says otherwise.
pub fn visibility(meta: &Map<String, Value>) -> Vec<String> {
    match meta.get("ui").and_then(|ui| ui.get("visibility")) {
        Some(Value::Array(values)) => values
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect(),
        Some(Value::String(value)) => vec![value.clone()],
        _ => VISIBILITY.iter().map(|v| v.to_string()).collect(),
    }
}
