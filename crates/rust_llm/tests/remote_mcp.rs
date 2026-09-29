//! Provider-side remote MCP (`with_provider_tools(mcp: {...})`), replayed from RubyLLM's
//! `chat_remote_mcp_*` cassettes with the assertions of `spec/ruby_llm/chat_mcp_spec.rb`. Every
//! request body must be JSON-equal to the one RubyLLM recorded.

mod support;

use rust_llm::{Chat, ProviderTool};
use serde_json::{Value, json};
use support::{Cassette, config_for};

const TOOLS: &str = "chat_remote_mcp_tools_openai_";

fn docs_with_approval() -> ProviderTool {
    ProviderTool::with_options(
        "mcp",
        json!({
            "name": "docs", "url": "https://learn.microsoft.com/api/mcp",
            "allowed_tools": ["microsoft_docs_search"], "require_approval": "always"
        }),
    )
}

async fn openai_chat(cassette: &Cassette) -> Chat {
    Chat::with_config(config_for(cassette, "openai"), Some("gpt-5-nano"), Some("openai"), false)
        .unwrap()
        .with_provider_tools([docs_with_approval()])
        .with_instructions("Use microsoft_docs_search exactly once when asked for documentation. Do not retry a denied call.")
}

const QUESTION: &str = "Search Microsoft documentation for Azure Blob Storage, then give one short sentence about it.";

fn has_mcp_call(calls: &[rust_llm::message::ServerToolCall], with_result: bool) -> bool {
    calls.iter().any(|c| {
        c.kind == "mcp_call" && c.name.as_deref() == Some("microsoft_docs_search") && (!with_result || c.result.as_ref().is_some_and(Value::is_string))
    })
}

#[tokio::test]
async fn openai_executes_a_remote_read_only_tool_after_approval_using_stateless_continuation() {
    let cassette = Cassette::start(&format!("{TOOLS}executes_a_remote_read-only_tool_after_approval_using_stateless_continuation")).await.unwrap();
    let mut chat = openai_chat(&cassette).await;
    chat.ask(QUESTION).await.unwrap();

    assert!(chat.is_awaiting_approval());
    let pending = chat.pending_approvals();
    assert_eq!(pending.len(), 1);
    assert!(pending[0].remote);
    assert_eq!(pending[0].name, "microsoft_docs_search");

    chat.approve(&pending[0].id);
    let response = chat.complete().await.unwrap();
    assert!(chat.is_complete());
    assert!(has_mcp_call(&response.server_tool_calls, true), "{:?}", response.server_tool_calls);
    assert!(response.content().to_lowercase().contains("blob"), "{}", response.content());
    cassette.assert_all_matched().await;
}

#[tokio::test]
async fn openai_denies_a_remote_call_without_executing_it() {
    let cassette = Cassette::start(&format!("{TOOLS}denies_a_remote_call_without_executing_it")).await.unwrap();
    let mut chat = openai_chat(&cassette).await;
    chat.ask(QUESTION).await.unwrap();
    assert!(chat.is_awaiting_approval());
    let id = chat.pending_approvals()[0].id.clone();
    chat.deny(&id);

    chat.complete().await.unwrap();
    assert!(chat.is_complete());
    assert!(!chat.messages().iter().any(|m| m.server_tool_calls.iter().any(|c| c.kind == "mcp_call")));
    let denial = chat.messages().iter().find(|m| m.is_tool_result()).unwrap();
    assert_eq!(denial.raw_content.as_ref().unwrap()[0]["approve"], json!(false));
    cassette.assert_all_matched().await;
}

#[tokio::test]
async fn openai_streams_a_remote_approval_and_its_completed_result() {
    let cassette = Cassette::start(&format!("{TOOLS}streams_a_remote_approval_and_its_completed_result")).await.unwrap();
    let mut chat = openai_chat(&cassette).await;
    let mut text = String::new();
    chat.ask_stream(QUESTION, |chunk| text.push_str(chunk.content())).await.unwrap();
    let pending = chat.pending_approvals();
    assert_eq!(pending.len(), 1);
    assert!(pending[0].remote);
    chat.approve(&pending[0].id);

    let response = chat.complete_stream(|chunk| text.push_str(chunk.content())).await.unwrap();
    assert!(chat.is_complete());
    assert!(has_mcp_call(&response.server_tool_calls, false), "{:?}", response.server_tool_calls);
    assert!(text.to_lowercase().contains("blob"), "{text}");
    cassette.assert_all_matched().await;
}

const EXECUTION_INSTRUCTIONS: &str =
    "Call microsoft_docs_search once to answer the first question. For later questions use only those results.";

/// `remote MCP execution`: the provider runs the tool itself and the history replays verbatim.
async fn executes_and_replays(provider: &str, model: &str, options: Value) -> Cassette {
    let cassette = Cassette::start(&format!("chat_remote_mcp_execution_{provider}_executes_a_read-only_tool_and_replays_its_history"))
        .await
        .unwrap();
    let mut chat = Chat::with_config(config_for(&cassette, provider), Some(model), Some(provider), false)
        .unwrap()
        .with_provider_tools([ProviderTool::with_options("mcp", options)])
        .with_instructions(EXECUTION_INSTRUCTIONS);

    let response = chat.ask("Search Microsoft documentation for Azure Blob Storage, then summarize it in one sentence.").await.unwrap();
    assert!(chat.is_complete());
    assert!(chat.pending_approvals().is_empty());
    let calls = &response.server_tool_calls;
    assert!(calls.iter().any(|c| c.name.as_deref().is_some_and(|n| n.contains("microsoft_docs_search"))), "{calls:?}");
    assert!(calls.iter().any(|c| c.result.is_some()), "{calls:?}");
    assert!(response.content().to_lowercase().contains("blob"));
    let followup = chat.ask("Which cloud service did you just look up?").await.unwrap();
    assert!(followup.content().to_lowercase().contains("blob"), "{}", followup.content());
    cassette
}

#[tokio::test]
async fn anthropic_executes_a_read_only_tool_and_replays_its_history() {
    let options = json!({
        "name": "docs", "url": "https://learn.microsoft.com/api/mcp",
        "default_config": { "enabled": false }, "configs": { "microsoft_docs_search": { "enabled": true } }
    });
    let cassette = executes_and_replays("anthropic", "claude-haiku-4-5", options).await;
    cassette.assert_all_matched().await;
    // The recorded requests carry the MCP connector beta.
    for request in cassette.server.received_requests().await.unwrap() {
        assert_eq!(request.headers.get("anthropic-beta").and_then(|v| v.to_str().ok()), Some("mcp-client-2025-11-20"));
    }
}

#[tokio::test]
async fn xai_executes_a_read_only_tool_and_replays_its_history() {
    let options = json!({ "name": "docs", "url": "https://learn.microsoft.com/api/mcp", "allowed_tools": ["microsoft_docs_search"] });
    let cassette = executes_and_replays("xai", "grok-4.3", options).await;
    cassette.assert_all_matched().await;
}
