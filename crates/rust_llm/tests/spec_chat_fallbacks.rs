//! `spec/ruby_llm/chat_fallbacks_spec.rb` and `chat_tool_approval_spec.rb`. The Ruby specs stub
//! `provider.complete` per model; here one mock server answers per request path in order (OpenAI
//! Responses for the primary, Anthropic Messages for the fallback). `// spec:` lines name the
//! Ruby example each test ports.

mod spec_helpers;

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use rust_llm::{
    Chat, Error, ErrorKind, Fallback, FallbackAttempt, ProtocolName, Tool, ToolCall, ToolError,
    ToolResult, UsageStatus,
};
use serde_json::{Map, Value, json};
use spec_helpers::*;
use wiremock::{Mock, MockServer, ResponseTemplate, matchers};

const PRIMARY: &str = "gpt-5-nano";
const FALLBACK: &str = "claude-haiku-4-5";
const SECOND: &str = "deepseek-v4-flash";

/// Mounts `responses` on `route`, answered in order.
async fn on(server: &MockServer, route: &str, responses: Vec<ResponseTemplate>) {
    Mock::given(matchers::path(route))
        .respond_with(Sequence(Mutex::new(responses.into())))
        .mount(server)
        .await;
}

fn status(code: u16, message: &str) -> ResponseTemplate {
    ResponseTemplate::new(code).set_body_json(json!({ "error": { "message": message } }))
}

fn ok(text: &str) -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(text_response(text))
}

fn deepseek_ok(text: &str) -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(json!({
        "id": "c1", "model": SECOND, "object": "chat.completion",
        "choices": [{ "index": 0, "message": { "role": "assistant", "content": text }, "finish_reason": "stop" }],
        "usage": { "prompt_tokens": 1, "completion_tokens": 1 }
    }))
}

async fn count(server: &MockServer, route: &str) -> usize {
    server
        .received_requests()
        .await
        .unwrap_or_default()
        .iter()
        .filter(|r| r.url.path() == route)
        .count()
}

fn primary(server: &MockServer) -> Chat {
    Chat::with_config(config(server), Some(PRIMARY), Some("openai"), false).unwrap()
}

fn to(model: &str, provider: &str) -> Fallback {
    Fallback {
        model: model.into(),
        provider: Some(provider.into()),
    }
}

// spec: chat_fallbacks_spec.rb:77 stores ordered fallback models
#[tokio::test]
async fn stores_ordered_fallback_models() {
    let server = serve(vec![]).await;
    let chat = primary(&server).with_fallbacks([FALLBACK.into(), to(SECOND, "deepseek")]);
    let ids: Vec<&str> = chat.fallbacks().iter().map(|f| f.model.as_str()).collect();
    assert_eq!(ids, [FALLBACK, SECOND]);
    assert_eq!(
        chat.fallbacks().last().unwrap().provider.as_deref(),
        Some("deepseek")
    );
    assert_eq!(
        chat.fallback_errors(),
        rust_llm::error::DEFAULT_FALLBACK_ERRORS
    );
}

// spec: chat_fallbacks_spec.rb:87 clears fallback models with with_fallbacks(nil)
#[tokio::test]
async fn with_no_fallbacks_clears_them_and_resets_the_errors() {
    let server = serve(vec![]).await;
    let chat = primary(&server)
        .with_fallbacks([FALLBACK.into()])
        .with_fallback_errors(vec![ErrorKind::RateLimit]);
    let chat = chat.with_fallbacks([]);
    assert!(chat.fallbacks().is_empty());
    assert_eq!(
        chat.fallback_errors(),
        rust_llm::error::DEFAULT_FALLBACK_ERRORS
    );
}

// spec: chat_fallbacks_spec.rb:97 falls back on transient errors and restores the primary model
#[tokio::test]
async fn falls_back_on_transient_errors_and_restores_the_primary_model() {
    let server = MockServer::start().await;
    on(&server, "/v1/responses", vec![status(503, "primary down")]).await;
    on(&server, "/v1/messages", vec![ok("from fallback")]).await;
    let mut chat = primary(&server).with_fallbacks([FALLBACK.into()]);
    chat.ask_later("Hello").unwrap();
    let response = chat.generate().await.unwrap();
    assert_eq!(response.content(), "from fallback");
    assert_eq!(
        (chat.model().id.as_str(), chat.provider().slug()),
        (PRIMARY, "openai")
    );
    assert_eq!(
        chat.messages().last().unwrap().model.as_deref(),
        Some(FALLBACK)
    );
}

// spec: chat_fallbacks_spec.rb:113 links failed fallback attempts to the response they ultimately produce
#[tokio::test]
async fn links_failed_attempts_to_the_response_they_produce() {
    let server = MockServer::start().await;
    on(&server, "/v1/responses", vec![status(503, "primary down")]).await;
    let mut answer = text_response("from fallback");
    answer["usage"] = json!({ "input_tokens": 4, "output_tokens": 2 });
    on(
        &server,
        "/v1/messages",
        vec![ResponseTemplate::new(200).set_body_json(answer)],
    )
    .await;
    let mut chat = primary(&server).with_fallbacks([FALLBACK.into()]);
    let response = chat.ask("Hello").await.unwrap();
    let entries = &response.usage_entries;
    assert_eq!(
        entries.iter().map(|e| e.status).collect::<Vec<_>>(),
        [UsageStatus::Failed, UsageStatus::Succeeded]
    );
    assert_eq!(
        entries.iter().map(|e| e.model.as_str()).collect::<Vec<_>>(),
        [PRIMARY, FALLBACK]
    );
    assert_eq!(chat.usage_entries(), entries.as_slice());
    assert_eq!(
        (response.tokens().input, response.tokens().output),
        (Some(4), Some(2))
    );
    assert_eq!(
        (chat.tokens().input, chat.tokens().output),
        (Some(4), Some(2))
    );
}

// spec: chat_fallbacks_spec.rb:147 tries fallback models in order
#[tokio::test]
async fn tries_fallback_models_in_order() {
    let server = MockServer::start().await;
    on(
        &server,
        "/v1/responses",
        vec![status(429, "primary rate limited")],
    )
    .await;
    on(
        &server,
        "/v1/messages",
        vec![status(529, "fallback overloaded")],
    )
    .await;
    on(
        &server,
        "/chat/completions",
        vec![deepseek_ok("second fallback")],
    )
    .await;
    let mut chat = primary(&server).with_fallbacks([FALLBACK.into(), to(SECOND, "deepseek")]);
    chat.ask_later("Hello").unwrap();
    assert_eq!(chat.generate().await.unwrap().content(), "second fallback");
    for route in ["/v1/responses", "/v1/messages", "/chat/completions"] {
        assert_eq!(count(&server, route).await, 1, "{route}");
    }
}

// spec: chat_fallbacks_spec.rb:165 does not fallback on non-transient errors
#[tokio::test]
async fn does_not_fall_back_on_non_transient_errors() {
    let server = MockServer::start().await;
    on(&server, "/v1/responses", vec![status(400, "bad request")]).await;
    on(&server, "/v1/messages", vec![ok("unused")]).await;
    let mut chat = primary(&server).with_fallbacks([FALLBACK.into()]);
    chat.ask_later("Hello").unwrap();
    assert_eq!(
        chat.generate().await.unwrap_err().kind(),
        ErrorKind::BadRequest
    );
    assert_eq!(count(&server, "/v1/messages").await, 0);
}

// spec: chat_fallbacks_spec.rb:177 falls back on configured errors
#[tokio::test]
async fn falls_back_on_configured_errors() {
    let server = MockServer::start().await;
    on(&server, "/v1/responses", vec![status(400, "bad request")]).await;
    on(&server, "/v1/messages", vec![ok("ok")]).await;
    let mut chat = primary(&server)
        .with_fallbacks([FALLBACK.into()])
        .with_fallback_errors(vec![ErrorKind::BadRequest]);
    chat.ask_later("Hello").unwrap();
    assert_eq!(chat.generate().await.unwrap().content(), "ok");
    assert_eq!(chat.fallback_errors(), [ErrorKind::BadRequest]);
}

type Events = Arc<Mutex<Vec<FallbackAttempt>>>;

// spec: chat_fallbacks_spec.rb:192 runs fallback callbacks before retrying
#[tokio::test]
async fn runs_fallback_callbacks_before_retrying() {
    let server = MockServer::start().await;
    on(
        &server,
        "/v1/responses",
        vec![status(500, "primary failed")],
    )
    .await;
    on(&server, "/v1/messages", vec![ok("ok")]).await;
    let (before, after): (Events, Events) = Default::default();
    let (b, a) = (before.clone(), after.clone());
    let mut chat = primary(&server)
        .with_fallbacks([to(FALLBACK, "anthropic")])
        .before_fallback(move |e| b.lock().unwrap().push(e.clone()))
        .after_fallback(move |e| a.lock().unwrap().push(e.clone()));
    chat.ask_later("Hello").unwrap();
    chat.generate().await.unwrap();

    let before = before.lock().unwrap();
    assert_eq!(before.len(), 1);
    let event = &before[0];
    assert_eq!(event.error_kind, ErrorKind::Server);
    // `to` is the resolved model: with a provider given, the alias resolves to Anthropic's dated id
    // (the Ruby spec stubs resolution, so it sees the bare id).
    let resolved = rust_llm::models()
        .find(FALLBACK, Some("anthropic"))
        .unwrap()
        .id;
    assert_eq!(
        (
            event.from.as_str(),
            event.to.as_str(),
            event.to_provider.as_str()
        ),
        (PRIMARY, resolved.as_str(), "anthropic")
    );
    assert_eq!(event.attempt, 1);
    assert!(!event.streaming && !event.chunks_yielded);

    let after = after.lock().unwrap();
    assert_eq!(after.len(), 1);
    assert_eq!(
        after[0]
            .response
            .as_ref()
            .map(|m| m.content().to_string())
            .as_deref(),
        Some("ok")
    );
    assert_eq!(after[0].succeeded, Some(true));
    assert!(after[0].fallback_error.is_none(), "not failed?");
}

// spec: chat_fallbacks_spec.rb:223 runs fallback callbacks when the attempt fails with a non-fallback error
#[tokio::test]
async fn runs_fallback_callbacks_when_the_fallback_fails_with_a_non_fallback_error() {
    let server = MockServer::start().await;
    on(
        &server,
        "/v1/responses",
        vec![status(500, "primary failed")],
    )
    .await;
    on(&server, "/v1/messages", vec![status(400, "bad request")]).await;
    let after: Events = Default::default();
    let a = after.clone();
    let mut chat = primary(&server)
        .with_fallbacks([FALLBACK.into()])
        .after_fallback(move |e| a.lock().unwrap().push(e.clone()));
    chat.ask_later("Hello").unwrap();
    assert_eq!(
        chat.generate().await.unwrap_err().kind(),
        ErrorKind::BadRequest
    );
    let after = after.lock().unwrap();
    assert_eq!(after.len(), 1);
    assert_eq!(after[0].succeeded, Some(false));
    assert_eq!(
        after[0].fallback_error.as_ref().map(|(k, _)| *k),
        Some(ErrorKind::BadRequest)
    );
}

// spec: chat_fallbacks_spec.rb:240 drops an explicit protocol override when the fallback changes provider
#[tokio::test]
async fn drops_an_explicit_protocol_when_the_fallback_changes_provider() {
    let server = MockServer::start().await;
    on(
        &server,
        "/v1/responses",
        vec![status(500, "primary failed")],
    )
    .await;
    on(&server, "/v1/messages", vec![ok("ok")]).await;
    let mut chat = primary(&server)
        .with_protocol(ProtocolName::Responses)
        .with_fallbacks([FALLBACK.into()]);
    chat.ask_later("Hello").unwrap();
    // The Anthropic fallback is reached at all only because the Responses override was dropped.
    assert_eq!(chat.generate().await.unwrap().content(), "ok");
    // ...and restored afterwards: the next render is a Responses payload again.
    assert!(
        chat.render().unwrap().get("input").is_some(),
        "primary protocol restored"
    );
}

// spec: chat_fallbacks_spec.rb:257 starts a new streaming message lifecycle when fallback follows yielded chunks
#[tokio::test]
async fn a_fallback_after_yielded_chunks_starts_a_new_message_lifecycle() {
    let server = MockServer::start().await;
    let partial = concat!(
        "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"primary partial\"}\n\n",
        "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"server_error\",\"code\":\"server_error\",\"message\":\"stream failed\"}}\n\n",
    );
    on(&server, "/v1/responses", vec![sse(partial.to_string())]).await;
    on(
        &server,
        "/v1/messages",
        vec![sse(text_stream(&["fallback chunk"]))],
    )
    .await;
    let lifecycle = log::<String>();
    let (l1, l2, l3, l4) = (
        lifecycle.clone(),
        lifecycle.clone(),
        lifecycle.clone(),
        lifecycle.clone(),
    );
    let mut chat = primary(&server)
        .with_fallbacks([FALLBACK.into()])
        .before_message(move || l1.lock().unwrap().push("before_message".into()))
        .after_message(move |m| {
            l2.lock()
                .unwrap()
                .push(format!("after_message {}", m.content()))
        })
        .before_fallback(move |e| {
            l3.lock().unwrap().push(format!(
                "before_fallback {} {} {}",
                e.from, e.to, e.chunks_yielded
            ))
        })
        .after_fallback(move |e| {
            l4.lock()
                .unwrap()
                .push(format!("after_fallback {} {:?}", e.to, e.succeeded))
        });
    chat.ask_later("Hello").unwrap();
    let chunks = log::<String>();
    let seen = chunks.clone();
    let response = chat
        .complete_stream(move |c| {
            if let Some(text) = c.content.clone().filter(|t| !t.is_empty()) {
                seen.lock().unwrap().push(text)
            }
        })
        .await
        .unwrap();
    assert_eq!(response.content(), "fallback chunk");
    assert_eq!(
        *chunks.lock().unwrap(),
        ["primary partial", "fallback chunk"]
    );
    assert_eq!(
        *lifecycle.lock().unwrap(),
        [
            "before_message".to_string(),
            format!("before_fallback {PRIMARY} {FALLBACK} true"),
            "before_message".into(),
            "after_message fallback chunk".into(),
            format!("after_fallback {FALLBACK} Some(true)"),
        ]
    );
}

// ---- chat_tool_approval_spec.rb ---------------------------------------------------------------

type Runs = Arc<Mutex<Vec<&'static str>>>;

/// `dangerous_tool` (requires approval) and `harmless_tool`, recording executions.
struct Probe {
    name: &'static str,
    approval: bool,
    runs: Runs,
}

#[async_trait]
impl Tool for Probe {
    fn name(&self) -> String {
        self.name.into()
    }
    fn description(&self) -> String {
        format!("The {} tool", self.name)
    }
    fn requires_approval(&self) -> bool {
        self.approval
    }
    async fn execute(
        &self,
        _args: Map<String, Value>,
        _call: &ToolCall,
    ) -> Result<ToolResult, ToolError> {
        self.runs.lock().unwrap().push(self.name);
        Ok(if self.name == "dangerous" {
            "done"
        } else {
            "fine"
        }
        .into())
    }
}

fn dangerous(runs: &Runs) -> Probe {
    Probe {
        name: "dangerous",
        approval: true,
        runs: runs.clone(),
    }
}

fn harmless(runs: &Runs) -> Probe {
    Probe {
        name: "harmless",
        approval: false,
        runs: runs.clone(),
    }
}

fn calls(pairs: &[(&str, &str)]) -> Value {
    let calls: Vec<(&str, &str, Value)> = pairs
        .iter()
        .map(|(id, name)| (*id, *name, json!({})))
        .collect();
    tool_use_response(&calls)
}

async fn stubbed(responses: Vec<Value>, tools: Vec<Probe>) -> (Chat, MockServer) {
    let server = serve(responses).await;
    let mut chat = chat(&server);
    for t in tools {
        chat = chat.with_tool(t);
    }
    (chat, server)
}

// spec: chat_tool_approval_spec.rb:52 parks the loop until a decision is recorded
#[tokio::test]
async fn parks_the_loop_until_a_decision_is_recorded() {
    let runs: Runs = Default::default();
    let (mut chat, _s) = stubbed(
        vec![calls(&[("call_1", "dangerous")]), text_response("All done")],
        vec![dangerous(&runs)],
    )
    .await;
    let response = chat.ask("Do the thing").await.unwrap();
    assert!(chat.is_awaiting_approval());
    assert!(!chat.is_complete());
    assert!(runs.lock().unwrap().is_empty());
    assert_eq!(
        response
            .tool_calls
            .unwrap()
            .keys()
            .cloned()
            .collect::<Vec<_>>(),
        ["call_1"]
    );
}

// spec: chat_tool_approval_spec.rb:63 resumes and executes after approve
// spec: chat_tool_approval_spec.rb:76 accepts a ToolCall for approve (Rust takes the call's id)
#[tokio::test]
async fn resumes_and_executes_after_approve() {
    let runs: Runs = Default::default();
    let (mut chat, _s) = stubbed(
        vec![calls(&[("call_1", "dangerous")]), text_response("All done")],
        vec![dangerous(&runs)],
    )
    .await;
    chat.ask("Do the thing").await.unwrap();
    chat.approve("call_1");
    let response = chat.complete().await.unwrap();
    assert_eq!(*runs.lock().unwrap(), ["dangerous"]);
    assert_eq!(response.content(), "All done");
    assert!(chat.is_complete());
    assert!(!chat.is_awaiting_approval());
}

// spec: chat_tool_approval_spec.rb:86 appends a structured denial result after deny
#[tokio::test]
async fn appends_a_structured_denial_after_deny() {
    let runs: Runs = Default::default();
    let (mut chat, _s) = stubbed(
        vec![calls(&[("call_1", "dangerous")]), text_response("All done")],
        vec![dangerous(&runs)],
    )
    .await;
    chat.ask("Do the thing").await.unwrap();
    chat.deny("call_1");
    let response = chat.complete().await.unwrap();
    assert!(runs.lock().unwrap().is_empty());
    let denial = chat.messages().iter().find(|m| m.is_tool_result()).unwrap();
    assert!(
        denial.content().contains("denied the dangerous tool call"),
        "{}",
        denial.content()
    );
    assert_eq!(denial.tool_call_id.as_deref(), Some("call_1"));
    assert_eq!(response.content(), "All done");
}

// spec: chat_tool_approval_spec.rb:150 refuses a new question while the round is parked
#[tokio::test]
async fn refuses_a_new_question_while_the_round_is_parked() {
    let runs: Runs = Default::default();
    let (mut chat, _s) = stubbed(
        vec![calls(&[("call_1", "dangerous")]), text_response("All done")],
        vec![dangerous(&runs)],
    )
    .await;
    chat.ask("Do the thing").await.unwrap();
    let err = chat.ask("Write an essay instead").await.unwrap_err();
    assert!(
        matches!(&err, Error::PendingToolCalls(m) if m.contains("dangerous")),
        "{err}"
    );
    assert_eq!(
        chat.messages()
            .iter()
            .filter(|m| m.role == rust_llm::Role::User)
            .count(),
        1
    );
}

// spec: chat_tool_approval_spec.rb:158 refuses a new question while any tool call is unanswered, approval or not
#[tokio::test]
async fn refuses_a_new_question_while_any_call_is_unanswered() {
    let runs: Runs = Default::default();
    let (mut chat, _s) = stubbed(
        vec![calls(&[("call_1", "harmless")])],
        vec![harmless(&runs)],
    )
    .await;
    chat.ask_later("Do it").unwrap();
    chat.generate().await.unwrap();
    let err = chat.ask_later("Another thing").unwrap_err();
    assert!(
        matches!(&err, Error::PendingToolCalls(m) if m.contains("harmless")),
        "{err}"
    );
}

// spec: chat_tool_approval_spec.rb:166 lists the pending approvals as ToolCall objects
#[tokio::test]
async fn lists_pending_approvals_as_tool_calls() {
    let runs: Runs = Default::default();
    let (mut chat, _s) = stubbed(
        vec![calls(&[("call_1", "dangerous")]), text_response("All done")],
        vec![dangerous(&runs)],
    )
    .await;
    chat.ask("Do the thing").await.unwrap();
    let pending = chat.pending_approvals();
    assert_eq!(
        pending.iter().map(|c| c.id.as_str()).collect::<Vec<_>>(),
        ["call_1"]
    );
    assert_eq!(pending[0].name, "dangerous");
    chat.approve(&pending[0].id);
    chat.complete().await.unwrap();
    assert!(chat.pending_approvals().is_empty());
}

// spec: chat_tool_approval_spec.rb:179 shows what the chat is waiting for in inspect
#[tokio::test]
async fn debug_output_shows_pending_approvals() {
    let runs: Runs = Default::default();
    let (mut chat, _s) = stubbed(
        vec![calls(&[("call_1", "dangerous")]), text_response("All done")],
        vec![dangerous(&runs)],
    )
    .await;
    chat.ask("Do the thing").await.unwrap();
    assert!(
        format!("{chat:?}").contains(r#"awaiting_approval: ["dangerous"]"#),
        "{chat:?}"
    );
    chat.approve("call_1");
    chat.complete().await.unwrap();
    assert!(
        !format!("{chat:?}").contains("awaiting_approval"),
        "{chat:?}"
    );
}

// spec: chat_tool_approval_spec.rb:191 runs tools that need no approval and parks only the one that does
#[tokio::test]
async fn runs_unapproved_tools_and_parks_only_the_one_that_needs_approval() {
    let runs: Runs = Default::default();
    let (mut chat, _s) = stubbed(
        vec![
            calls(&[("call_a", "harmless"), ("call_b", "dangerous")]),
            text_response("All done"),
        ],
        vec![harmless(&runs), dangerous(&runs)],
    )
    .await;
    chat.ask("Do both things").await.unwrap();
    assert_eq!(*runs.lock().unwrap(), ["harmless"]);
    assert!(chat.is_awaiting_approval());
    chat.approve("call_b");
    assert_eq!(chat.complete().await.unwrap().content(), "All done");
    assert_eq!(*runs.lock().unwrap(), ["harmless", "dangerous"]);
}

/// A `dangerous` tool whose `requires_approval { |tool_call| ... }` resolver returns `decision`,
/// logging each consultation.
struct Resolved {
    runs: Runs,
    decision: Option<bool>,
    consulted: Arc<Mutex<Vec<String>>>,
}

#[async_trait]
impl Tool for Resolved {
    fn name(&self) -> String {
        "dangerous".into()
    }
    fn description(&self) -> String {
        "The dangerous tool".into()
    }
    fn requires_approval(&self) -> bool {
        true
    }
    fn approval(&self, call: &ToolCall) -> Option<Option<bool>> {
        self.consulted.lock().unwrap().push(call.id.clone());
        Some(self.decision)
    }
    async fn execute(
        &self,
        _args: Map<String, Value>,
        _call: &ToolCall,
    ) -> Result<ToolResult, ToolError> {
        self.runs.lock().unwrap().push("dangerous");
        Ok("done".into())
    }
}

async fn resolved(decision: Option<bool>) -> (Chat, Runs, Arc<Mutex<Vec<String>>>, MockServer) {
    let runs: Runs = Default::default();
    let consulted: Arc<Mutex<Vec<String>>> = Default::default();
    let server = serve(vec![
        calls(&[("call_1", "dangerous")]),
        text_response("All done"),
    ])
    .await;
    let chat = chat(&server).with_tool(Resolved {
        runs: runs.clone(),
        decision,
        consulted: consulted.clone(),
    });
    (chat, runs, consulted, server)
}

// spec: chat_tool_approval_spec.rb:106 executes immediately when the resolver returns true
#[tokio::test]
async fn an_approving_resolver_executes_immediately() {
    let (mut chat, runs, _, _s) = resolved(Some(true)).await;
    assert_eq!(
        chat.ask("Do the thing").await.unwrap().content(),
        "All done"
    );
    assert_eq!(*runs.lock().unwrap(), ["dangerous"]);
}

// spec: chat_tool_approval_spec.rb:118 denies without executing when the resolver returns false
#[tokio::test]
async fn a_denying_resolver_denies_without_executing() {
    let (mut chat, runs, _, _s) = resolved(Some(false)).await;
    assert_eq!(
        chat.ask("Do the thing").await.unwrap().content(),
        "All done"
    );
    assert!(runs.lock().unwrap().is_empty());
    assert!(
        chat.messages()
            .iter()
            .find(|m| m.is_tool_result())
            .unwrap()
            .content()
            .contains("denied")
    );
}

// spec: chat_tool_approval_spec.rb:131 consults the resolver lazily, never at definition or registration
#[tokio::test]
async fn the_resolver_is_consulted_lazily() {
    let (mut chat, runs, consulted, _s) = resolved(None).await;
    assert!(
        consulted.lock().unwrap().is_empty(),
        "not consulted at registration"
    );
    chat.ask("Do the thing").await.unwrap();
    let mut ids = consulted.lock().unwrap().clone();
    ids.dedup();
    assert_eq!(ids, ["call_1"]);
    assert!(chat.is_awaiting_approval());
    assert!(runs.lock().unwrap().is_empty());
}

// ---- chat_server_tool_approval_spec.rb --------------------------------------------------------

/// A Responses turn carrying a remote MCP approval request plus any local function calls.
fn server_round(local: &[(&str, &str)]) -> Value {
    let mut output = vec![json!({
        "type": "mcp_approval_request", "id": "approval_1", "name": "search",
        "arguments": "{\"query\":\"Ruby\"}", "server_label": "docs"
    })];
    for (id, name) in local {
        output.push(
            json!({ "type": "function_call", "call_id": id, "name": name, "arguments": "{}" }),
        );
    }
    json!({ "id": "resp_1", "object": "response", "status": "completed", "model": "gpt-5-nano", "output": output,
            "usage": { "input_tokens": 1, "output_tokens": 1 } })
}

fn responses_text(text: &str) -> Value {
    json!({ "id": "resp_2", "object": "response", "status": "completed", "model": "gpt-5-nano",
            "output": [{ "type": "message", "role": "assistant", "content": [{ "type": "output_text", "text": text }] }],
            "usage": { "input_tokens": 1, "output_tokens": 1 } })
}

async fn server_chat(responses: Vec<Value>, runs: &Runs) -> (Chat, MockServer) {
    let server = serve(responses).await;
    let chat = Chat::with_config(config(&server), Some("gpt-5-nano"), Some("openai"), false)
        .unwrap()
        .with_protocol(ProtocolName::Responses)
        .with_tool(Probe {
            name: "search",
            approval: false,
            runs: runs.clone(),
        });
    (chat, server)
}

// spec: chat_server_tool_approval_spec.rb:33 parks without trying to execute a remote approval request locally
#[tokio::test]
async fn a_remote_approval_request_parks_without_running_a_same_name_local_tool() {
    let runs: Runs = Default::default();
    let (mut chat, server) = server_chat(
        vec![server_round(&[]), responses_text("Ruby documentation")],
        &runs,
    )
    .await;
    chat.ask("Look up Ruby").await.unwrap();
    assert!(chat.is_awaiting_approval());
    let pending = chat.pending_approvals();
    assert_eq!(
        (pending.len(), pending[0].id.as_str(), pending[0].remote),
        (1, "approval_1", true)
    );
    assert!(runs.lock().unwrap().is_empty());
    assert_eq!(requests(&server).await, 1);
}

// spec: chat_server_tool_approval_spec.rb:44 records a server #{approved ? 'approval' : 'denial'} exactly once without executing a same-name local tool
#[tokio::test]
async fn a_server_decision_is_recorded_once_without_running_the_local_tool() {
    for approved in [true, false] {
        let runs: Runs = Default::default();
        let (mut chat, _server) = server_chat(
            vec![server_round(&[]), responses_text("Ruby documentation")],
            &runs,
        )
        .await;
        chat.ask("Look up Ruby").await.unwrap();
        if approved {
            chat.approve("approval_1")
        } else {
            chat.deny("approval_1")
        };
        chat.run_tools().await.unwrap();
        chat.run_tools().await.unwrap();
        let decisions: Vec<&rust_llm::Message> = chat
            .messages()
            .iter()
            .filter(|m| m.is_tool_result())
            .collect();
        assert_eq!(decisions.len(), 1, "approved={approved}");
        assert_eq!(
            decisions[0].raw_content,
            Some(
                json!([{ "type": "mcp_approval_response", "approval_request_id": "approval_1", "approve": approved }])
            )
        );
        assert!(runs.lock().unwrap().is_empty());
        assert_eq!(
            chat.complete().await.unwrap().content(),
            "Ruby documentation"
        );
    }
}

// spec: chat_server_tool_approval_spec.rb:69 runs local calls in a mixed round before parking for the server decision
#[tokio::test]
async fn local_calls_in_a_mixed_round_run_before_parking() {
    let runs: Runs = Default::default();
    let (mut chat, _server) = server_chat(
        vec![
            server_round(&[("local_1", "search")]),
            responses_text("Ruby documentation"),
        ],
        &runs,
    )
    .await;
    chat.ask("Look up Ruby").await.unwrap();
    assert_eq!(*runs.lock().unwrap(), ["search"]);
    assert!(chat.is_awaiting_approval());
    chat.approve("approval_1");
    assert_eq!(
        chat.complete().await.unwrap().content(),
        "Ruby documentation"
    );
}
