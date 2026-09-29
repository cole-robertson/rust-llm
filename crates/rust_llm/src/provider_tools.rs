//! Port of `lib/ruby_llm/tools/provider_tools.rb`, `Protocol#apply_provider_tools`, and the
//! `SERVER_TOOL_ALIASES` tables of the protocols this crate speaks: tools that run on the
//! provider's servers, such as web search, code execution, or a remote MCP server.
//!
//! ```ruby
//! chat.with_provider_tools(:web_search)
//! chat.with_provider_tools(mcp: { name: "docs", url: "https://learn.microsoft.com/api/mcp" })
//! chat.with_provider_tools({ type: "web_search_20260318", name: "web_search" })
//! ```
//!
//! ```no_run
//! # use rust_llm::ProviderTool;
//! # use serde_json::json;
//! # fn run() -> rust_llm::Result<()> {
//! # let chat = rust_llm::chat()?;
//! let chat = chat.with_provider_tools(["web_search".into()]);
//! let chat = chat.with_provider_tools([ProviderTool::with_options("mcp", json!({ "name": "docs", "url": "..." }))]);
//! let chat = chat.with_provider_tools([ProviderTool::raw(json!({ "type": "web_search_20260318", "name": "web_search" }))]);
//! # let _ = chat; Ok(()) }
//! ```

use serde_json::{Map, Value, json};

use crate::error::{Error, Result};
use crate::protocols::deep_merge;
use crate::providers::{ProtocolName, Provider};

/// One `with_provider_tools` entry: a portable alias with options in the provider's own
/// vocabulary, or the provider's tool definition passed verbatim.
#[derive(Debug, Clone, PartialEq)]
pub enum ProviderTool {
    Alias { name: String, options: Map<String, Value> },
    Raw(Value),
}

impl ProviderTool {
    /// `with_provider_tools(:web_search)`.
    pub fn alias(name: impl Into<String>) -> ProviderTool {
        ProviderTool::Alias { name: name.into(), options: Map::new() }
    }

    /// `with_provider_tools(web_search: { allowed_domains: [...] })`.
    pub fn with_options(name: impl Into<String>, options: Value) -> ProviderTool {
        let options = match options {
            Value::Object(map) => map,
            _ => Map::new(),
        };
        ProviderTool::Alias { name: name.into(), options }
    }

    /// `with_provider_tools({ type: "...", ... })`: a tool RubyLLM has no alias for yet.
    pub fn raw(definition: Value) -> ProviderTool {
        ProviderTool::Raw(definition)
    }
}

impl From<&str> for ProviderTool {
    fn from(name: &str) -> Self {
        ProviderTool::alias(name)
    }
}

/// What an alias contributes (`{ tool:, payload:, headers: }`).
#[derive(Default)]
struct Entry {
    tool: Option<Value>,
    payload: Map<String, Value>,
    headers: Vec<(String, String)>,
}

enum Spec {
    Tool(Value),
    Build(fn(&Map<String, Value>) -> Result<Entry>),
}

/// `ProviderTools::Resolution`: tool entries for the payload's tools slot, extra payload fields,
/// and request headers.
#[derive(Default)]
pub(crate) struct Resolution {
    tools: Vec<Value>,
    payload: Map<String, Value>,
    pub(crate) headers: Vec<(String, String)>,
}

impl Resolution {
    fn add(&mut self, spec: Spec, options: &Map<String, Value>) -> Result<()> {
        let entry = match spec {
            Spec::Build(build) => build(options)?,
            Spec::Tool(mut tool) => {
                deep_merge(&mut tool, &Value::Object(options.clone()));
                Entry { tool: Some(tool), ..Default::default() }
            }
        };
        self.tools.extend(entry.tool);
        // Payload arrays accumulate: every entry contributes its own MCP server.
        for (key, addition) in entry.payload {
            match (self.payload.get_mut(&key), &addition) {
                (Some(Value::Array(current)), Value::Array(more)) => {
                    for item in more {
                        if !current.contains(item) {
                            current.push(item.clone());
                        }
                    }
                }
                (Some(current @ Value::Object(_)), Value::Object(_)) => deep_merge(current, &addition),
                _ => {
                    self.payload.insert(key, addition);
                }
            }
        }
        // Anthropic-style beta headers combine as comma-separated values.
        for (key, value) in entry.headers {
            match self.headers.iter_mut().find(|(k, _)| *k == key) {
                Some((_, existing)) if !existing.split(',').any(|v| v == value) => *existing = format!("{existing},{value}"),
                Some(_) => {}
                None => self.headers.push((key, value)),
            }
        }
        Ok(())
    }
}

/// `Protocol#resolve_provider_tools_for_request`. `None` when the chat enabled none.
pub(crate) fn resolve(protocol: ProtocolName, provider: Provider, entries: &[ProviderTool]) -> Result<Option<Resolution>> {
    if entries.is_empty() {
        return Ok(None);
    }
    let Some(table) = aliases(protocol, provider) else {
        return Err(Error::UnsupportedServerTool(format!(
            "{} has no provider-tool support through RubyLLM yet. Request options in the provider vocabulary can be set with with_provider_options.",
            provider.display()
        )));
    };
    let mut resolution = Resolution::default();
    for entry in entries {
        match entry {
            ProviderTool::Raw(tool) => resolution.tools.push(tool.clone()),
            ProviderTool::Alias { name, options } => {
                let Some(spec) = table(name) else {
                    let known: Vec<String> = ALIAS_NAMES.iter().filter(|n| table(n).is_some()).map(|n| format!(":{n}")).collect();
                    return Err(Error::UnsupportedServerTool(format!(
                        "{} has no server tool alias :{name}. Known aliases: {}. New or unlisted tools work by passing the provider's tool definition as a Hash.",
                        provider.display(),
                        known.join(", ")
                    )));
                };
                resolution.add(spec, options)?;
            }
        }
    }
    if protocol == ProtocolName::Responses && provider == Provider::OpenRouter {
        check_openrouter_mcp(&resolution.tools)?;
    }
    if protocol == ProtocolName::Responses && provider == Provider::GPUStack {
        resolution.tools = gpustack_mcp_tools(std::mem::take(&mut resolution.tools))?;
    }
    Ok(Some(resolution))
}

/// `Protocol#apply_provider_tools`: extra payload fields deep-merge in, and provider tools join
/// function tools in the payload's tools array.
pub(crate) fn apply(payload: &mut Value, resolution: &Resolution) {
    if !resolution.payload.is_empty() {
        deep_merge(payload, &Value::Object(resolution.payload.clone()));
    }
    if resolution.tools.is_empty() {
        return;
    }
    let mut tools = match payload.get("tools") {
        Some(Value::Array(existing)) => existing.clone(),
        Some(Value::Null) | None => Vec::new(),
        Some(other) => vec![other.clone()],
    };
    tools.extend(resolution.tools.iter().cloned());
    payload["tools"] = Value::Array(tools);
}

/// Every alias name any table defines, in a stable order for error messages.
const ALIAS_NAMES: &[&str] = &[
    "web_search", "web_fetch", "url_context", "code_execution", "code_interpreter", "file_search", "image_generation",
    "x_search", "collections_search", "google_search", "google_maps", "datetime", "apply_patch", "shell", "mcp",
];

type Table = fn(&str) -> Option<Spec>;

/// Which protocol class's `server_tool_aliases` applies to this provider.
fn aliases(protocol: ProtocolName, provider: Provider) -> Option<Table> {
    match (protocol, provider) {
        (ProtocolName::Anthropic, _) => Some(anthropic),
        (ProtocolName::Gemini, _) => Some(gemini),
        (ProtocolName::Responses, Provider::XAI) => Some(xai_responses),
        (ProtocolName::Responses, Provider::DeepSeek) => Some(deepseek_responses),
        (ProtocolName::Responses, Provider::OpenRouter) => Some(openrouter_responses),
        (ProtocolName::Responses, Provider::GPUStack) => Some(gpustack_responses),
        (ProtocolName::Responses, _) => Some(responses),
        (ProtocolName::ChatCompletions, Provider::OpenRouter) => Some(openrouter_chat_completions),
        (ProtocolName::ChatCompletions, Provider::Mistral) => Some(mistral_multi_completion),
        (ProtocolName::ChatCompletions, _) => None,
        (ProtocolName::Interactions, _) => Some(interactions),
        (ProtocolName::Conversations, _) => Some(mistral_conversations),
        (ProtocolName::RouterChatCompletions, _) => None,
    }
}

fn tool(value: Value) -> Option<Spec> {
    Some(Spec::Tool(value))
}

fn slice(options: &Map<String, Value>, keys: &[&str]) -> Map<String, Value> {
    keys.iter().filter_map(|k| options.get(*k).map(|v| (k.to_string(), v.clone()))).collect()
}

/// `Protocols::Anthropic::SERVER_TOOL_ALIASES`. web_search/web_fetch pin `allowed_callers` to
/// direct invocation.
fn anthropic(name: &str) -> Option<Spec> {
    match name {
        "web_search" => tool(json!({ "type": "web_search_20260318", "name": "web_search", "allowed_callers": ["direct"] })),
        "web_fetch" | "url_context" => tool(json!({ "type": "web_fetch_20260318", "name": "web_fetch", "allowed_callers": ["direct"] })),
        "code_execution" => tool(json!({ "type": "code_execution_20260521", "name": "code_execution" })),
        "mcp" => Some(Spec::Build(anthropic_mcp)),
        _ => None,
    }
}

fn anthropic_mcp(options: &Map<String, Value>) -> Result<Entry> {
    let name = options.get("name").cloned().unwrap_or_else(|| "mcp".into());
    let mut server = Map::new();
    server.insert("type".into(), "url".into());
    server.insert("name".into(), name.clone());
    server.extend(slice(options, &["url", "authorization_token"]));
    let mut toolset = Map::new();
    toolset.insert("type".into(), "mcp_toolset".into());
    toolset.insert("mcp_server_name".into(), name);
    toolset.extend(slice(options, &["default_config", "configs"]));
    let mut payload = Map::new();
    payload.insert("mcp_servers".into(), json!([server]));
    Ok(Entry {
        tool: Some(Value::Object(toolset)),
        payload,
        headers: vec![("anthropic-beta".into(), "mcp-client-2025-11-20".into())],
    })
}

/// `Protocols::Responses::SERVER_TOOL_ALIASES`.
fn responses(name: &str) -> Option<Spec> {
    match name {
        "web_search" => tool(json!({ "type": "web_search" })),
        "file_search" => tool(json!({ "type": "file_search" })),
        "code_execution" | "code_interpreter" => tool(json!({ "type": "code_interpreter", "container": { "type": "auto" } })),
        "image_generation" => tool(json!({ "type": "image_generation" })),
        "mcp" => Some(Spec::Build(responses_mcp)),
        _ => None,
    }
}

/// The Responses `mcp` alias: `url` becomes `server_url` and `name` becomes `server_label`.
fn responses_mcp(options: &Map<String, Value>) -> Result<Entry> {
    let mut definition = Map::new();
    definition.insert("type".into(), "mcp".into());
    for (key, value) in options {
        let key = match key.as_str() {
            "url" => "server_url",
            "name" => "server_label",
            other => other,
        };
        definition.insert(key.into(), value.clone());
    }
    Ok(Entry { tool: Some(Value::Object(definition)), ..Default::default() })
}

/// `Providers::XAI::Responses::SERVER_TOOL_ALIASES`.
fn xai_responses(name: &str) -> Option<Spec> {
    match name {
        "web_search" => tool(json!({ "type": "web_search" })),
        "x_search" => tool(json!({ "type": "x_search" })),
        "code_execution" => tool(json!({ "type": "code_execution" })),
        "code_interpreter" => tool(json!({ "type": "code_interpreter" })),
        "file_search" | "collections_search" => tool(json!({ "type": "file_search" })),
        "image_generation" => tool(json!({ "type": "image_generation" })),
        "mcp" => Some(Spec::Build(responses_mcp)),
        _ => None,
    }
}

/// `Providers::DeepSeek::Responses::SERVER_TOOL_ALIASES`.
fn deepseek_responses(name: &str) -> Option<Spec> {
    match name {
        "apply_patch" => tool(json!({ "type": "custom", "name": "apply_patch" })),
        _ => None,
    }
}

fn openrouter_hosted(name: &str) -> Option<Spec> {
    match name {
        "web_search" | "web_fetch" | "datetime" | "image_generation" | "apply_patch" | "shell" => {
            tool(json!({ "type": format!("openrouter:{name}") }))
        }
        "url_context" => tool(json!({ "type": "openrouter:web_fetch" })),
        "code_execution" => tool(json!({ "type": "openrouter:shell" })),
        _ => None,
    }
}

/// `Providers::OpenRouter::ChatCompletions::SERVER_TOOL_ALIASES`.
fn openrouter_chat_completions(name: &str) -> Option<Spec> {
    openrouter_hosted(name)
}

/// `Protocols::OpenRouter::Responses::SERVER_TOOL_ALIASES`.
fn openrouter_responses(name: &str) -> Option<Spec> {
    match name {
        "mcp" => Some(Spec::Build(responses_mcp)),
        other => openrouter_hosted(other),
    }
}

/// `Protocols::OpenRouter::Responses#merge_server_tool_entries`: OpenRouter returns no approval
/// events, so a remote MCP server must say `require_approval: "never"`.
fn check_openrouter_mcp(tools: &[Value]) -> Result<()> {
    for tool in tools {
        let is_mcp = tool.get("type").and_then(Value::as_str) == Some("mcp");
        if is_mcp && tool.get("require_approval").and_then(Value::as_str) != Some("never") {
            return Err(Error::Argument(
                "OpenRouter MCP requires explicit require_approval: 'never'; approval events are not returned".into(),
            ));
        }
    }
    Ok(())
}

/// `Protocols::GPUStack::Responses::SERVER_TOOL_ALIASES`: vLLM's MCP servers configured on the
/// deployment.
fn gpustack_responses(name: &str) -> Option<Spec> {
    match name {
        "web_search" => Some(Spec::Build(|o| gpustack_mcp_alias(o, "web_search_preview", Some(&["search"])))),
        "web_fetch" => Some(Spec::Build(|o| gpustack_mcp_alias(o, "web_search_preview", Some(&["open"])))),
        "code_execution" => Some(Spec::Build(|o| gpustack_mcp_alias(o, "code_interpreter", None))),
        "mcp" => Some(Spec::Build(responses_mcp)),
        _ => None,
    }
}

fn gpustack_mcp_alias(options: &Map<String, Value>, label: &str, tools: Option<&[&str]>) -> Result<Entry> {
    if options.keys().any(|k| k != "require_approval") {
        return Err(Error::Argument("GPUStack server tool aliases accept only require_approval; use mcp for custom filters".into()));
    }
    let mut definition = json!({ "type": "mcp", "server_label": label });
    if let Some(approval) = options.get("require_approval") {
        definition["require_approval"] = approval.clone();
    }
    if let Some(tools) = tools {
        definition["allowed_tools"] = json!(tools);
    }
    Ok(Entry { tool: Some(definition), ..Default::default() })
}

const GPUSTACK_MCP_LABELS: &[&str] = &["web_search_preview", "code_interpreter", "container"];

/// `Protocols::GPUStack::Responses#merge_server_tool_entries`: every MCP entry is validated
/// against the servers vLLM has configured, then entries for the same server combine their
/// `allowed_tools`, so vLLM cannot let one overwrite the other.
fn gpustack_mcp_tools(tools: Vec<Value>) -> Result<Vec<Value>> {
    let is_mcp = |t: &Value| t.get("type").and_then(Value::as_str) == Some("mcp");
    for tool in tools.iter().filter(|t| is_mcp(t)) {
        validate_gpustack_mcp(tool)?;
    }
    let mut merged: Vec<Value> = Vec::new();
    for tool in tools {
        let label = tool.get("server_label");
        let existing = if is_mcp(&tool) { merged.iter_mut().find(|e| is_mcp(e) && e.get("server_label") == label) } else { None };
        match existing {
            None => merged.push(tool),
            Some(existing) if *existing == tool => {}
            Some(existing) => merge_gpustack_mcp_filter(existing, &tool)?,
        }
    }
    Ok(merged)
}

/// `merge_mcp_filter`: only entries that differ in nothing but explicit tool-name lists combine.
fn merge_gpustack_mcp_filter(existing: &mut Value, tool: &Value) -> Result<()> {
    let without_filter = |t: &Value| {
        let mut t = t.clone();
        if let Some(o) = t.as_object_mut() {
            o.remove("allowed_tools");
        }
        t
    };
    let explicit = |t: &Value| {
        t.get("allowed_tools").and_then(Value::as_array).is_some_and(|names| !names.iter().any(|n| n.as_str() == Some("*")))
    };
    if without_filter(existing) != without_filter(tool) || !explicit(existing) || !explicit(tool) {
        return Err(Error::Argument("Combine GPUStack MCP settings for each server in one entry with explicit tool names".into()));
    }
    if let (Some(Value::Array(names)), Some(Value::Array(more))) = (existing.get_mut("allowed_tools"), tool.get("allowed_tools")) {
        for name in more {
            if !names.contains(name) {
                names.push(name.clone());
            }
        }
    }
    Ok(())
}

/// `validate_mcp_tool`.
fn validate_gpustack_mcp(tool: &Value) -> Result<()> {
    let present = |k: &str| tool.get(k).is_some_and(|v| !v.is_null() && *v != Value::Bool(false));
    if tool.get("require_approval").and_then(Value::as_str) != Some("never") {
        return Err(Error::Argument("GPUStack MCP requires explicit require_approval: 'never'; vLLM has no approval events".into()));
    }
    if present("server_url") || present("connector_id") || present("authorization") {
        return Err(Error::Argument("GPUStack MCP uses servers configured on vLLM, not per-request URLs or connectors".into()));
    }
    if !tool.get("server_label").and_then(Value::as_str).is_some_and(|l| GPUSTACK_MCP_LABELS.contains(&l)) {
        return Err(Error::Argument(format!(
            "GPUStack MCP name must match a configured vLLM label: {}",
            GPUSTACK_MCP_LABELS.join(", ")
        )));
    }
    if tool.pointer("/allowed_tools/read_only").is_some_and(|v| !v.is_null() && *v != Value::Bool(false)) {
        return Err(Error::Argument("vLLM filters MCP tools by name, not by read_only; use allowed_tools: [name]".into()));
    }
    Ok(())
}

/// `Protocols::Mistral::MultiCompletion::SERVER_TOOL_ALIASES`.
fn mistral_multi_completion(name: &str) -> Option<Spec> {
    match name {
        "image_generation" => tool(json!({ "type": "image_generation" })),
        "mcp" => tool(json!({ "type": "connector" })),
        _ => None,
    }
}

/// `Protocols::Interactions::SERVER_TOOL_ALIASES`: options merge into the tool entry.
fn interactions(name: &str) -> Option<Spec> {
    match name {
        "mcp" => tool(json!({ "type": "mcp_server" })),
        "web_search" => tool(json!({ "type": "google_search" })),
        "web_fetch" => tool(json!({ "type": "url_context" })),
        "code_execution" => tool(json!({ "type": "code_execution" })),
        "file_search" => tool(json!({ "type": "file_search" })),
        "google_maps" => tool(json!({ "type": "google_maps" })),
        _ => None,
    }
}

/// `Protocols::Mistral::Conversations::SERVER_TOOL_ALIASES`.
fn mistral_conversations(name: &str) -> Option<Spec> {
    match name {
        "web_search" | "web_fetch" => tool(json!({ "type": "web_search" })),
        "code_execution" => tool(json!({ "type": "code_interpreter" })),
        "file_search" => tool(json!({ "type": "document_library" })),
        "image_generation" => tool(json!({ "type": "image_generation" })),
        "mcp" => tool(json!({ "type": "connector" })),
        _ => None,
    }
}

/// `Protocols::Gemini::SERVER_TOOL_ALIASES`: Gemini nests each tool's options inside its key.
fn gemini(name: &str) -> Option<Spec> {
    match name {
        "google_search" | "web_search" => Some(Spec::Build(|o| gemini_tool("google_search", o))),
        "url_context" | "web_fetch" => Some(Spec::Build(|o| gemini_tool("url_context", o))),
        "code_execution" => Some(Spec::Build(|o| gemini_tool("code_execution", o))),
        "file_search" => Some(Spec::Build(|o| gemini_tool("file_search", o))),
        "google_maps" => Some(Spec::Build(|o| gemini_tool("google_maps", o))),
        _ => None,
    }
}

fn gemini_tool(key: &str, options: &Map<String, Value>) -> Result<Entry> {
    Ok(Entry { tool: Some(json!({ key: options })), ..Default::default() })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn applied(protocol: ProtocolName, provider: Provider, entries: &[ProviderTool]) -> (Value, Vec<(String, String)>) {
        let resolution = resolve(protocol, provider, entries).unwrap().unwrap();
        let mut payload = json!({ "model": "m", "tools": [{ "type": "function", "name": "weather" }] });
        apply(&mut payload, &resolution);
        (payload, resolution.headers)
    }

    #[test]
    fn anthropic_mcp_adds_a_server_a_toolset_and_the_beta_header() {
        let options = json!({ "name": "docs", "url": "https://learn.microsoft.com/api/mcp", "default_config": { "enabled": false } });
        let (payload, headers) = applied(ProtocolName::Anthropic, Provider::Anthropic, &[ProviderTool::with_options("mcp", options)]);
        assert_eq!(payload["mcp_servers"], json!([{ "type": "url", "name": "docs", "url": "https://learn.microsoft.com/api/mcp" }]));
        assert_eq!(payload["tools"][1], json!({ "type": "mcp_toolset", "mcp_server_name": "docs", "default_config": { "enabled": false } }));
        assert_eq!(headers, vec![("anthropic-beta".to_string(), "mcp-client-2025-11-20".to_string())]);
    }

    #[test]
    fn two_anthropic_servers_accumulate_and_share_one_beta_header() {
        let a = ProviderTool::with_options("mcp", json!({ "name": "a", "url": "https://a.example/mcp" }));
        let b = ProviderTool::with_options("mcp", json!({ "name": "b", "url": "https://b.example/mcp" }));
        let (payload, headers) = applied(ProtocolName::Anthropic, Provider::Anthropic, &[a, b]);
        assert_eq!(payload["mcp_servers"].as_array().unwrap().len(), 2);
        assert_eq!(headers.len(), 1);
    }

    #[test]
    fn aliases_merge_options_and_raw_tools_pass_through() {
        let search = ProviderTool::with_options("web_search", json!({ "allowed_domains": ["ruby-lang.org"] }));
        let raw = ProviderTool::raw(json!({ "type": "shell" }));
        let (payload, _) = applied(ProtocolName::Responses, Provider::OpenAI, &[search, raw]);
        assert_eq!(payload["tools"][1], json!({ "type": "web_search", "allowed_domains": ["ruby-lang.org"] }));
        assert_eq!(payload["tools"][2], json!({ "type": "shell" }));
    }

    #[test]
    fn unknown_aliases_and_unsupported_providers_are_refused() {
        let err = resolve(ProtocolName::Anthropic, Provider::Anthropic, &["x_search".into()]).err().unwrap();
        assert!(matches!(&err, Error::UnsupportedServerTool(m) if m.contains("no server tool alias :x_search") && m.contains(":mcp")));
        let err = resolve(ProtocolName::ChatCompletions, Provider::Ollama, &["web_search".into()]).err().unwrap();
        assert!(matches!(&err, Error::UnsupportedServerTool(m) if m.contains("has no provider-tool support")));
    }

    #[test]
    fn openrouter_mcp_must_opt_out_of_approval() {
        let mcp = ProviderTool::with_options("mcp", json!({ "url": "https://x.example/mcp" }));
        assert!(resolve(ProtocolName::Responses, Provider::OpenRouter, &[mcp]).is_err());
    }
}
