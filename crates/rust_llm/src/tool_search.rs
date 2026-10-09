//! Tool search with deferred tool loading: `lib/ruby_llm/chat/tool_search.rb`, the
//! `apply_tool_search`/`tool_search_block?` halves of `protocols/anthropic/tools.rb` and
//! `protocols/responses/tools.rb`, and each protocol's `portable_raw_content`
//! (`protocols/anthropic/chat.rb`, `protocols/responses/chat.rb`).
//!
//! A deferred tool's definition stays out of the model's context until the provider's tool search
//! loads it. Anthropic and the OpenAI Responses API search natively: deferred tools go out with
//! `defer_loading: true` next to a search tool. Every other protocol sends them as ordinary
//! tools. Loaded tools stay deferred on the wire, so the tools array is the same every turn and
//! the prompt cache survives.

use serde_json::{Value, json};

use crate::message::Message;
use crate::providers::ProtocolName;

/// `Anthropic::Tools::NATIVE_TOOL_SEARCH`.
pub fn anthropic_native_tool_search() -> Value {
    json!({ "type": "tool_search_tool_bm25_20251119", "name": "tool_search_tool_bm25" })
}

/// `Responses::Tools::NATIVE_TOOL_SEARCH`.
pub fn responses_native_tool_search() -> Value {
    json!({ "type": "tool_search" })
}

/// `Responses::Chat::TOOL_SEARCH_ITEM_TYPES`.
pub const TOOL_SEARCH_ITEM_TYPES: &[&str] = &["tool_search_call", "tool_search_output"];

/// `Responses::Chat::PORTABLE_OUTPUT_ITEM_TYPES`.
const PORTABLE_OUTPUT_ITEM_TYPES: &[&str] = &[
    "message",
    "function_call",
    "tool_search_call",
    "tool_search_output",
];

/// `Anthropic::Chat::PORTABLE_BLOCK_TYPES`.
const PORTABLE_BLOCK_TYPES: &[&str] = &["text", "tool_use"];

/// `Anthropic::Chat::THINKING_BLOCK_TYPES`.
const THINKING_BLOCK_TYPES: &[&str] = &["thinking", "redacted_thinking"];

fn kind(value: &Value) -> &str {
    value.get("type").and_then(Value::as_str).unwrap_or("")
}

/// `Anthropic::Tools.tool_search_block?`: the search's `server_tool_use` and its result.
pub fn is_tool_search_block(block: &Value) -> bool {
    let k = kind(block);
    if k == "tool_search_tool_result" {
        return true;
    }
    k == "server_tool_use"
        && block
            .get("name")
            .and_then(Value::as_str)
            .is_some_and(|n| n.starts_with("tool_search_tool_"))
}

fn defers_loading(tools: &[Value]) -> bool {
    tools.iter().any(|t| {
        t.get("defer_loading")
            .is_some_and(|v| !matches!(v, Value::Null | Value::Bool(false)))
    })
}

/// `Protocol#apply_tool_search`: the API requires a search tool next to deferred tools, and
/// rejects search history in a request that declares none. A configured search tool is kept.
pub(crate) fn apply(protocol: ProtocolName, payload: &mut Value) {
    match protocol {
        ProtocolName::Anthropic => apply_anthropic(payload),
        ProtocolName::Responses => apply_responses(payload),
        _ => {}
    }
}

/// `Anthropic::Tools.apply_tool_search`.
pub fn apply_anthropic(payload: &mut Value) {
    let tools: Vec<Value> = payload
        .get("tools")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if tools
        .iter()
        .any(|t| kind(t).starts_with("tool_search_tool_"))
    {
        return;
    }
    if defers_loading(&tools) {
        let mut tools = tools;
        tools.push(anthropic_native_tool_search());
        payload["tools"] = Value::Array(tools);
        return;
    }
    // `without_tool_search_blocks`: a message left with no content goes too.
    let Some(messages) = payload.get_mut("messages").and_then(Value::as_array_mut) else {
        return;
    };
    messages.retain_mut(|message| {
        let Some(content) = message.get_mut("content").and_then(Value::as_array_mut) else {
            return true;
        };
        content.retain(|block| !is_tool_search_block(block));
        !content.is_empty()
    });
}

/// `Responses::Tools.apply_tool_search`.
pub fn apply_responses(payload: &mut Value) {
    let tools: Vec<Value> = payload
        .get("tools")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if tools.iter().any(|t| kind(t) == "tool_search") {
        return;
    }
    if defers_loading(&tools) {
        let mut tools = tools;
        tools.push(responses_native_tool_search());
        payload["tools"] = Value::Array(tools);
        return;
    }
    if let Some(input) = payload.get_mut("input").and_then(Value::as_array_mut) {
        input.retain(|item| !TOOL_SEARCH_ITEM_TYPES.contains(&kind(item)));
    }
}

fn is_namespaced_call(item: &Value) -> bool {
    kind(item) == "function_call" && item.get("namespace").is_some_and(|n| !n.is_null())
}

/// `portable_raw_content`: the raw content another model of the same provider reads, such as a
/// tool search, without the producing model's own reasoning. `None` when the turn holds anything
/// else provider-shaped.
pub(crate) fn portable_raw_content(protocol: ProtocolName, message: &Message) -> Option<Value> {
    let items = message.raw_content.as_ref()?.as_array()?;
    match protocol {
        ProtocolName::Anthropic => {
            if !items.iter().any(is_tool_search_block) {
                return None;
            }
            let kept: Vec<Value> = items
                .iter()
                .filter(|b| !THINKING_BLOCK_TYPES.contains(&kind(b)))
                .cloned()
                .collect();
            kept.iter()
                .all(|b| PORTABLE_BLOCK_TYPES.contains(&kind(b)) || is_tool_search_block(b))
                .then_some(Value::Array(kept))
        }
        ProtocolName::Responses => {
            if !items
                .iter()
                .any(|i| TOOL_SEARCH_ITEM_TYPES.contains(&kind(i)) || is_namespaced_call(i))
            {
                return None;
            }
            let kept: Vec<Value> = items
                .iter()
                .filter(|i| kind(i) != "reasoning")
                .cloned()
                .collect();
            kept.iter()
                .all(|i| PORTABLE_OUTPUT_ITEM_TYPES.contains(&kind(i)))
                .then_some(Value::Array(kept))
        }
        _ => None,
    }
}
