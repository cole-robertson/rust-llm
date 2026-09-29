//! `spec/ruby_llm/chat_loop_spec.rb`, `chat_before_request_spec.rb`, and `chat_callbacks_spec.rb`:
//! the agentic loop's state machine (`ask_later`, `complete?`, `step`, `generate`, `run_tools`,
//! `complete`, `add_completion`, `cancel`), request hooks, and callback ordering. The Ruby specs
//! stub `provider.complete`; here a mock server returns canned Anthropic responses, so each test
//! also runs the real render and parse path. `// spec:` lines tie each test to its Ruby example.

mod spec_helpers;

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use rust_llm::{Error, Message, Parameter, Progress, Role, Tool, ToolCall, ToolError, ToolResult};
use serde_json::{Map, Value, json};
use spec_helpers::*;

/// `EchoTool`: echoes `text`.
struct Echo;

#[async_trait]
impl Tool for Echo {
    fn name(&self) -> String {
        "echo".into()
    }
    fn description(&self) -> String {
        "Echoes the given text".into()
    }
    fn parameters(&self) -> Vec<Parameter> {
        vec![Parameter::new("text").description("Text to echo")]
    }
    async fn execute(&self, args: Map<String, Value>, _call: &ToolCall) -> Result<ToolResult, ToolError> {
        Ok(args["text"].as_str().unwrap_or_default().into())
    }
}

/// `AttributedEchoTool`: tools receive the executing `ToolCall`.
struct AttributedEcho;

#[async_trait]
impl Tool for AttributedEcho {
    fn description(&self) -> String {
        "Echoes the given text".into()
    }
    fn parameters(&self) -> Vec<Parameter> {
        vec![Parameter::new("text").description("Text to echo")]
    }
    async fn execute(&self, args: Map<String, Value>, call: &ToolCall) -> Result<ToolResult, ToolError> {
        Ok(format!("{} via {}", args["text"].as_str().unwrap_or_default(), call.id).into())
    }
}

fn echo_call(name: &str) -> Message {
    tool_call_message(&[("call_1", name, json!({ "text": "hello" }))])
}

fn two_echo_calls() -> Message {
    tool_call_message(&[("call_1", "echo", json!({ "text": "first" })), ("call_2", "echo", json!({ "text": "second" }))])
}

fn roles(chat: &rust_llm::Chat) -> Vec<Role> {
    chat.messages().iter().map(|m| m.role).collect()
}

// ---- #ask_later / #complete? ------------------------------------------------------------------

// spec: chat_loop_spec.rb:48 stages the question without asking the model
#[tokio::test]
async fn ask_later_stages_the_question_without_asking_the_model() {
    let server = serve(vec![]).await;
    let mut chat = chat(&server).with_tool(Echo);
    chat.ask_later("Echo \"hello\" back to me.").unwrap();
    assert_eq!(requests(&server).await, 0);
    assert!(!chat.is_complete());
    assert_eq!(chat.messages().last().unwrap().role, Role::User);
}

// spec: chat_loop_spec.rb:60 walks the agentic-loop state machine
#[tokio::test]
async fn complete_walks_the_agentic_loop_state_machine() {
    let server = serve(vec![]).await;
    let mut chat = chat(&server).with_tool(Echo);
    assert!(chat.is_complete(), "nothing staged yet");
    chat.ask_later("Echo \"hello\" back to me.").unwrap();
    assert!(!chat.is_complete(), "model owes a response");
    chat.add_message(echo_call("echo"));
    assert!(!chat.is_complete(), "tools owe results");
    chat.run_tools().await.unwrap();
    assert!(!chat.is_complete(), "model owes a response again");
    chat.add_completion(answer_message("hello"), false);
    assert!(chat.is_complete(), "answered, no tools");
}

// spec: chat_loop_spec.rb:76 is not complete while a tool round is partially answered
#[tokio::test]
async fn not_complete_while_a_tool_round_is_partially_answered() {
    let server = serve(vec![]).await;
    let mut chat = chat(&server).with_tool(Echo);
    chat.ask_later("Echo twice.").unwrap();
    chat.add_message(two_echo_calls());
    chat.add_message(Message::tool_result("call_1", "first"));
    assert!(!chat.is_complete());
}

// spec: chat_loop_spec.rb:84 is complete on a chat with only instructions
#[tokio::test]
async fn complete_on_a_chat_with_only_instructions() {
    let server = serve(vec![]).await;
    assert!(chat(&server).with_tool(Echo).with_instructions("Be terse.").is_complete());
}

// spec: chat_loop_spec.rb:88 ignores trailing instructions when deciding whether the model owes a response
#[tokio::test]
async fn trailing_instructions_do_not_hide_a_staged_question() {
    let server = serve(vec![]).await;
    let mut chat = chat(&server).with_tool(Echo);
    chat.ask_later("Echo \"hello\" back to me.").unwrap();
    let chat = chat.with_instructions("Be terse.");
    assert!(!chat.is_complete());
}

// ---- #cancel ----------------------------------------------------------------------------------

// spec: chat_loop_spec.rb:97 marks the chat for one-shot cancellation
#[tokio::test]
async fn cancel_marks_the_chat_for_one_shot_cancellation() {
    let server = serve(vec![]).await;
    let chat = chat(&server);
    chat.cancel();
    assert!(chat.is_cancelled());
}

// spec: chat_loop_spec.rb:102 raises before the next model request and clears the cancellation flag
#[tokio::test]
async fn cancel_raises_before_the_next_request_and_clears_the_flag() {
    let server = serve(vec![text_response("hello")]).await;
    let mut chat = chat(&server).with_tool(Echo);
    chat.ask_later("Echo \"hello\" back to me.").unwrap();
    chat.cancel();
    let err = chat.step().await.unwrap_err();
    assert!(matches!(err, Error::Cancelled));
    assert_eq!(err.to_string(), "Chat generation cancelled");
    assert!(!chat.is_cancelled());
    assert_eq!(requests(&server).await, 0);
}

// spec: chat_loop_spec.rb:112 raises while streaming when the block cancels the chat
#[tokio::test]
async fn cancel_from_the_stream_block_stops_the_stream() {
    let server = serve_templates(vec![sse(text_stream(&["one", "two"]))]).await;
    let mut chat = chat(&server).with_tool(Echo);
    chat.ask_later("Count slowly.").unwrap();
    let handle = chat.cancel_handle();
    let chunks = Arc::new(Mutex::new(Vec::new()));
    let seen = chunks.clone();
    let err = chat
        .complete_stream(move |chunk| {
            if let Some(text) = &chunk.content {
                seen.lock().unwrap().push(text.clone());
                handle.cancel();
            }
        })
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Cancelled), "{err}");
    assert_eq!(*chunks.lock().unwrap(), ["one"]);
    assert!(!chat.is_cancelled());
    assert_eq!(roles(&chat), [Role::User]);
}

// spec: chat_loop_spec.rb:134 raises before appending a non-streaming response when cancelled during the request
#[tokio::test]
async fn cancel_during_a_request_drops_the_response_but_keeps_its_cost() {
    // Ruby cancels inside the stubbed `provider.complete`; here a before_request hook cancels,
    // which runs after the entry checkpoint and before the answer comes back.
    let server = serve(vec![text_response("hello")]).await;
    let handle_later = Arc::new(Mutex::new(None::<rust_llm::CancelHandle>));
    let slot = handle_later.clone();
    let mut chat = spec_helpers::chat(&server).with_tool(Echo).before_request(move |_| {
        if let Some(h) = slot.lock().unwrap().as_ref() {
            h.cancel();
        }
    });
    *handle_later.lock().unwrap() = Some(chat.cancel_handle());
    chat.ask_later("Echo \"hello\" back to me.").unwrap();
    let err = chat.generate().await.unwrap_err();
    assert!(matches!(err, Error::Cancelled), "{err}");
    assert!(!chat.is_cancelled());
    assert_eq!(roles(&chat), [Role::User]);
    assert!(chat.cost().total().is_some_and(|t| t > 0.0), "the billed attempt still counts: {:?}", chat.cost());
}

// spec: chat_loop_spec.rb:148 raises before executing pending tool calls
#[tokio::test]
async fn cancel_raises_before_executing_pending_tool_calls() {
    let server = serve(vec![]).await;
    let mut chat = chat(&server).with_tool(Echo);
    chat.ask_later("Echo \"hello\" back to me.").unwrap();
    chat.add_message(echo_call("echo"));
    chat.cancel();
    let err = chat.run_tools().await.unwrap_err();
    assert_eq!(err.to_string(), "Chat generation cancelled");
    assert!(chat.messages().last().unwrap().is_tool_call());
}

// ---- #generate / #run_tools -------------------------------------------------------------------

// spec: chat_loop_spec.rb:160 calls the model once and appends the response
#[tokio::test]
async fn generate_calls_the_model_once_and_appends_the_response() {
    let server = serve(vec![text_response("hello")]).await;
    let mut chat = chat(&server).with_tool(Echo);
    chat.ask_later("Echo \"hello\" back to me.").unwrap();
    let result = chat.generate().await.unwrap();
    assert_eq!(result.content(), "hello");
    assert_eq!(chat.messages().last().unwrap().content(), "hello");
    assert_eq!(requests(&server).await, 1);
}

// spec: chat_loop_spec.rb:173 executes pending tool calls without asking the model, and returns self
#[tokio::test]
async fn run_tools_executes_pending_calls_without_asking_the_model() {
    let server = serve(vec![]).await;
    let mut chat = chat(&server).with_tool(Echo);
    chat.ask_later("Echo \"hello\" back to me.").unwrap();
    chat.add_message(echo_call("echo"));
    chat.run_tools().await.unwrap();
    assert_eq!(requests(&server).await, 0);
    let last = chat.messages().last().unwrap();
    assert_eq!((last.role, last.content()), (Role::Tool, "hello"));
}

// spec: chat_loop_spec.rb:184 does nothing when no tool calls are pending
#[tokio::test]
async fn run_tools_does_nothing_without_pending_calls() {
    let server = serve(vec![]).await;
    let mut chat = chat(&server).with_tool(Echo);
    chat.ask_later("Echo \"hello\" back to me.").unwrap();
    chat.run_tools().await.unwrap();
    assert_eq!(chat.messages().len(), 1);
}

// spec: chat_loop_spec.rb:190 executes only the tool calls that have no results yet
#[tokio::test]
async fn run_tools_executes_only_unanswered_calls() {
    let server = serve(vec![]).await;
    let mut chat = chat(&server).with_tool(Echo);
    chat.ask_later("Echo twice.").unwrap();
    chat.add_message(two_echo_calls());
    chat.add_message(Message::tool_result("call_1", "first"));
    chat.run_tools().await.unwrap();
    let results: Vec<&Message> = chat.messages().iter().filter(|m| m.is_tool_result()).collect();
    let ids: Vec<&str> = results.iter().map(|m| m.tool_call_id.as_deref().unwrap()).collect();
    assert_eq!(ids, ["call_1", "call_2"]);
    assert_eq!(results.last().unwrap().content(), "second");
}

// spec: chat_loop_spec.rb:203 does nothing when every tool call in the round is answered
#[tokio::test]
async fn run_tools_does_nothing_when_the_round_is_answered() {
    let server = serve(vec![]).await;
    let mut chat = chat(&server).with_tool(Echo);
    chat.ask_later("Echo twice.").unwrap();
    chat.add_message(two_echo_calls());
    chat.add_message(Message::tool_result("call_1", "first"));
    chat.add_message(Message::tool_result("call_2", "second"));
    chat.run_tools().await.unwrap();
    assert_eq!(chat.messages().len(), 4);
}

// spec: chat_loop_spec.rb:212 passes the executing ToolCall to tools that declare tool_call:
#[tokio::test]
async fn run_tools_passes_the_executing_tool_call() {
    let server = serve(vec![]).await;
    let mut chat = chat(&server).with_tool(AttributedEcho);
    chat.ask_later("Echo \"hello\" back to me.").unwrap();
    chat.add_message(echo_call("attributed_echo"));
    chat.run_tools().await.unwrap();
    assert_eq!(chat.messages().last().unwrap().content(), "hello via call_1");
}

// ---- #step ------------------------------------------------------------------------------------

// spec: chat_loop_spec.rb:232 generates when the model owes a response
#[tokio::test]
async fn step_generates_when_the_model_owes_a_response() {
    let server = serve(vec![text_response("hello")]).await;
    let mut chat = chat(&server).with_tool(Echo);
    chat.ask_later("Echo \"hello\" back to me.").unwrap();
    assert_eq!(chat.step().await.unwrap().unwrap().content(), "hello");
}

// spec: chat_loop_spec.rb:239 runs tools when the model asked for them
#[tokio::test]
async fn step_runs_tools_when_the_model_asked_for_them() {
    let server = serve(vec![]).await;
    let mut chat = chat(&server).with_tool(Echo);
    chat.ask_later("Echo \"hello\" back to me.").unwrap();
    chat.add_message(echo_call("echo"));
    chat.step().await.unwrap();
    assert_eq!(requests(&server).await, 0);
    assert_eq!(chat.messages().last().unwrap().role, Role::Tool);
}

// spec: chat_loop_spec.rb:250 returns nil once the conversation is complete
#[tokio::test]
async fn step_returns_none_once_complete() {
    let server = serve(vec![]).await;
    let mut chat = chat(&server).with_tool(Echo);
    chat.ask_later("Echo \"hello\" back to me.").unwrap();
    chat.add_completion(answer_message("hello"), false);
    assert!(chat.step().await.unwrap().is_none());
}

// spec: chat_loop_spec.rb:257 finishes an interrupted tool round before generating
#[tokio::test]
async fn step_finishes_an_interrupted_tool_round_before_generating() {
    let server = serve(vec![]).await;
    let mut chat = chat(&server).with_tool(Echo);
    chat.ask_later("Echo twice.").unwrap();
    chat.add_message(two_echo_calls());
    chat.add_message(Message::tool_result("call_1", "first"));
    chat.step().await.unwrap();
    assert_eq!(requests(&server).await, 0);
    assert_eq!(chat.messages().iter().filter(|m| m.is_tool_result()).count(), 2);
}

// spec: chat_loop_spec.rb:269 generates once every tool call in the round is answered
#[tokio::test]
async fn step_generates_once_every_call_is_answered() {
    let server = serve(vec![text_response("hello")]).await;
    let mut chat = chat(&server).with_tool(Echo);
    chat.ask_later("Echo twice.").unwrap();
    chat.add_message(two_echo_calls());
    chat.add_message(Message::tool_result("call_1", "first"));
    chat.add_message(Message::tool_result("call_2", "second"));
    assert_eq!(chat.step().await.unwrap().unwrap().content(), "hello");
}

// ---- #complete / #add_completion --------------------------------------------------------------

// spec: chat_loop_spec.rb:281 resumes a chat whose last message is an unanswered tool call
#[tokio::test]
async fn complete_resumes_an_unanswered_tool_call() {
    let server = serve(vec![text_response("hello")]).await;
    let mut chat = chat(&server).with_tool(Echo);
    chat.ask_later("Echo \"hello\" back to me.").unwrap();
    chat.add_message(echo_call("echo"));
    let response = chat.complete().await.unwrap();
    assert_eq!(response.content(), "hello");
    assert_eq!(roles(&chat), [Role::User, Role::Assistant, Role::Tool, Role::Assistant]);
    assert_eq!(requests(&server).await, 1);
}

// spec: chat_loop_spec.rb:293 resumes a chat interrupted between tool executions
#[tokio::test]
async fn complete_resumes_between_tool_executions() {
    let server = serve(vec![text_response("hello")]).await;
    let mut chat = chat(&server).with_tool(Echo);
    chat.ask_later("Echo twice.").unwrap();
    chat.add_message(two_echo_calls());
    chat.add_message(Message::tool_result("call_1", "first"));
    let response = chat.complete().await.unwrap();
    assert_eq!(response.content(), "hello");
    assert_eq!(roles(&chat), [Role::User, Role::Assistant, Role::Tool, Role::Tool, Role::Assistant]);
    assert_eq!(requests(&server).await, 1);
}

// spec: chat_loop_spec.rb:306 is a no-op on an already-complete chat
#[tokio::test]
async fn complete_is_a_no_op_on_a_complete_chat() {
    let server = serve(vec![]).await;
    let mut chat = chat(&server).with_tool(Echo);
    chat.ask_later("Echo \"hello\" back to me.").unwrap();
    chat.add_completion(answer_message("hello"), false);
    assert_eq!(chat.complete().await.unwrap().content(), "hello");
    assert_eq!(requests(&server).await, 0);
}

// spec: chat_loop_spec.rb:317 appends the response and runs message callbacks
// spec: chat_options_spec.rb:236 runs the message callbacks for a response produced out of band
#[tokio::test]
async fn add_completion_appends_and_runs_message_callbacks() {
    let server = serve(vec![]).await;
    let received = log::<String>();
    let (before, after) = (received.clone(), received.clone());
    let mut chat = chat(&server)
        .with_tool(Echo)
        .before_message(move || before.lock().unwrap().push("before".into()))
        .after_message(move |m| after.lock().unwrap().push(m.content().to_string()));
    chat.ask_later("Echo \"hello\" back to me.").unwrap();
    chat.add_completion(answer_message("hello"), false);
    assert_eq!(chat.messages().last().unwrap().content(), "hello");
    assert_eq!(*received.lock().unwrap(), ["before", "hello"]);
}

// spec: chat_loop_spec.rb:330 parses JSON content when the chat has a schema
#[tokio::test]
async fn add_completion_keeps_raw_json_and_parsed_reads_it() {
    let server = serve(vec![]).await;
    let mut chat = chat(&server).with_schema(json!({ "type": "object", "properties": { "answer": { "type": "string" } } }));
    let message = chat.add_completion(Message::assistant(r#"{"answer":"hello"}"#), false).clone();
    assert_eq!(message.content(), r#"{"answer":"hello"}"#);
    assert_eq!(message.parsed().unwrap(), Some(json!({ "answer": "hello" })));
}

// ---- chat_before_request_spec.rb --------------------------------------------------------------

fn staged(server: &wiremock::MockServer, hook: impl FnMut(&mut Value) + Send + Sync + 'static) -> rust_llm::Chat {
    let mut chat = chat(server).before_request(hook);
    chat.ask_later("Hello").unwrap();
    chat
}

// spec: chat_before_request_spec.rb:11 lets hooks mutate the rendered payload in place
#[tokio::test]
async fn before_request_hooks_mutate_the_rendered_payload() {
    let server = serve(vec![]).await;
    let chat = staged(&server, |p| p["metadata"] = json!({ "user_id": "u-1" }));
    assert_eq!(chat.render().unwrap()["metadata"], json!({ "user_id": "u-1" }));
}

// spec: chat_before_request_spec.rb:18 lets hooks add provider-native content blocks
#[tokio::test]
async fn before_request_hooks_add_provider_native_blocks() {
    let server = serve(vec![]).await;
    let chat = staged(&server, |p| {
        let last = p["messages"].as_array_mut().unwrap().last_mut().unwrap();
        last["content"].as_array_mut().unwrap().push(json!({ "type": "custom_context", "data": "x" }));
    });
    let payload = chat.render().unwrap();
    let content = payload["messages"].as_array().unwrap().last().unwrap()["content"].as_array().unwrap().clone();
    assert_eq!(content.last().unwrap(), &json!({ "type": "custom_context", "data": "x" }));
}

// spec: chat_before_request_spec.rb:27 supports wholesale replacement via payload.replace
#[tokio::test]
async fn before_request_hooks_replace_the_payload_wholesale() {
    let server = serve(vec![]).await;
    let chat = staged(&server, |p| {
        let mut replaced = p.clone();
        replaced["stream"] = json!(true);
        *p = replaced;
    });
    assert_eq!(chat.render().unwrap()["stream"], json!(true));
}

// spec: chat_before_request_spec.rb:34 ignores hook return values
// N/A-ish in Rust: hooks return `()`, so there is no return value to ignore. The assertion that
// the payload stays a normal payload still holds.
#[tokio::test]
async fn before_request_hooks_cannot_return_a_replacement() {
    let server = serve(vec![]).await;
    let chat = staged(&server, |_| {});
    let payload = chat.render().unwrap();
    assert!(payload.get("replacement").is_none());
    assert!(payload["messages"].is_array());
}

// spec: chat_before_request_spec.rb:42 runs hooks in registration order after params merging
#[tokio::test]
async fn before_request_hooks_run_after_provider_options_merge() {
    let server = serve(vec![]).await;
    let mut chat = chat(&server)
        .with_provider_options(json!({ "metadata": { "user_id": "from-params" } }))
        .before_request(|p| p["metadata"]["user_id"] = json!("from-hook"));
    chat.ask_later("Hello").unwrap();
    assert_eq!(chat.render().unwrap()["metadata"], json!({ "user_id": "from-hook" }));
}

// A request actually sent carries the hook's edit too (Ruby applies hooks inside `Protocol#render`).
#[tokio::test]
async fn before_request_hooks_apply_to_the_sent_request() {
    let server = serve(vec![text_response("ok")]).await;
    let mut chat = chat(&server).before_request(|p| p["metadata"] = json!({ "user_id": "u-1" }));
    chat.ask("Hello").await.unwrap();
    let sent: Value = serde_json::from_slice(&server.received_requests().await.unwrap()[0].body).unwrap();
    assert_eq!(sent["metadata"], json!({ "user_id": "u-1" }));
}

// ---- chat_callbacks_spec.rb -------------------------------------------------------------------

// spec: chat_callbacks_spec.rb:32 runs additive message callbacks in order
#[tokio::test]
async fn message_callbacks_are_additive_and_ordered() {
    let server = serve(vec![text_response("done")]).await;
    let calls = log::<String>();
    let (a, b, c, d) = (calls.clone(), calls.clone(), calls.clone(), calls.clone());
    let mut chat = chat(&server)
        .before_message(move || a.lock().unwrap().push("before_one".into()))
        .before_message(move || b.lock().unwrap().push("before_two".into()))
        .after_message(move |m| c.lock().unwrap().push(format!("after_one {}", m.content())))
        .after_message(move |m| d.lock().unwrap().push(format!("after_two {}", m.content())));
    chat.ask("Hello").await.unwrap();
    assert_eq!(*calls.lock().unwrap(), ["before_one", "before_two", "after_one done", "after_two done"]);
}

/// `CallbackProbeTool`.
struct CallbackProbe;

#[async_trait]
impl Tool for CallbackProbe {
    fn description(&self) -> String {
        "Returns a callback probe result".into()
    }
    async fn execute(&self, _args: Map<String, Value>, _call: &ToolCall) -> Result<ToolResult, ToolError> {
        Ok("tool result".into())
    }
}

// spec: chat_callbacks_spec.rb:52 runs additive tool callbacks in order
#[tokio::test]
async fn tool_callbacks_are_additive_and_ordered() {
    let server = serve(vec![tool_use_response(&[("call_1", "callback_probe", json!({}))]), text_response("complete")]).await;
    let calls = log::<String>();
    let (a, b) = (calls.clone(), calls.clone());
    let mut chat = chat(&server)
        .with_tool(CallbackProbe)
        .before_tool_call(move |call| a.lock().unwrap().push(format!("before_tool_call {}", call.name)))
        .after_tool_result(move |result| b.lock().unwrap().push(format!("after_tool_result {}", result.content)));
    chat.ask("Use the tool").await.unwrap();
    assert_eq!(*calls.lock().unwrap(), ["before_tool_call callback_probe", "after_tool_result tool result"]);
}

/// `ProgressProbeTool`: reports twice, then answers.
struct ProgressProbe;

#[async_trait]
impl Tool for ProgressProbe {
    fn description(&self) -> String {
        "Reports progress while it works".into()
    }
    fn parameters(&self) -> Vec<Parameter> {
        vec![Parameter::new("label")]
    }
    async fn execute(&self, args: Map<String, Value>, _call: &ToolCall) -> Result<ToolResult, ToolError> {
        let label = args["label"].as_str().unwrap_or_default().to_string();
        rust_llm::progress::report(Progress { value: None, total: None, message: Some(format!("Starting {label}")) });
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        rust_llm::progress::report(Progress { value: Some(2.0), total: Some(2.0), message: Some(format!("Finishing {label}")) });
        Ok(format!("{label} done").into())
    }
}

fn progress_calls(labels: &[&str]) -> Value {
    let calls: Vec<(String, Value)> = labels.iter().map(|l| (format!("call_{l}"), json!({ "label": l }))).collect();
    let calls: Vec<(&str, &str, Value)> = calls.iter().map(|(id, a)| (id.as_str(), "progress_probe", a.clone())).collect();
    tool_use_response(&calls)
}

// spec: chat_callbacks_spec.rb:86 passes each report to after_tool_progress with its tool call, before the result
#[tokio::test]
async fn tool_progress_reaches_after_tool_progress_before_the_result() {
    let server = serve(vec![progress_calls(&["a"]), text_response("complete")]).await;
    let calls = log::<String>();
    let (a, b) = (calls.clone(), calls.clone());
    let mut chat = chat(&server)
        .with_tool(ProgressProbe)
        .after_tool_progress(move |call, p| {
            a.lock().unwrap().push(format!("{} {} {:?}", call.id, p.message.clone().unwrap_or_default(), p.fraction()))
        })
        .after_tool_result(move |r| b.lock().unwrap().push(format!("result {}", r.content)));
    chat.ask("Use the tool").await.unwrap();
    assert_eq!(*calls.lock().unwrap(), ["call_a Starting a None", "call_a Finishing a Some(1.0)", "result a done"]);
}

// spec: chat_callbacks_spec.rb:98 keeps progress with its own tool call when tools run with #{mode}
#[tokio::test]
async fn concurrent_tool_progress_stays_with_its_own_call() {
    let server = serve(vec![progress_calls(&["a", "b"]), text_response("complete")]).await;
    let calls = log::<(String, String)>();
    let seen = calls.clone();
    let mut chat = chat(&server)
        .with_tool(ProgressProbe)
        .with_tool_concurrency(true)
        .after_tool_progress(move |call, p| seen.lock().unwrap().push((call.id.clone(), p.message.clone().unwrap_or_default())));
    chat.ask("Use the tools").await.unwrap();
    let calls = calls.lock().unwrap();
    for id in ["a", "b"] {
        let mine: Vec<&str> = calls.iter().filter(|(c, _)| *c == format!("call_{id}")).map(|(_, m)| m.as_str()).collect();
        assert_eq!(mine, [format!("Starting {id}"), format!("Finishing {id}")]);
    }
}

/// `FanOutTool`: reports from a task it spawns.
struct FanOut;

#[async_trait]
impl Tool for FanOut {
    fn description(&self) -> String {
        "Reports from a background task".into()
    }
    async fn execute(&self, _args: Map<String, Value>, _call: &ToolCall) -> Result<ToolResult, ToolError> {
        let listener = rust_llm::progress::listener();
        tokio::spawn(rust_llm::progress::listen(listener, async {
            rust_llm::progress::report(Progress { value: None, total: None, message: Some("Reading in the background".into()) });
        }))
        .await?;
        Ok("done".into())
    }
}

// spec: chat_callbacks_spec.rb:111 passes reports from threads the tool starts
// Tokio task-locals don't cross `spawn` on their own (Ruby's fiber storage does); a tool hands
// the listener on with `progress::listener()` + `progress::listen`, which this checks.
#[tokio::test]
async fn progress_from_a_task_the_tool_spawns_reaches_the_chat() {
    let server = serve(vec![tool_use_response(&[("call_1", "fan_out", json!({}))]), text_response("complete")]).await;
    let calls = log::<(String, String)>();
    let seen = calls.clone();
    let mut chat = chat(&server)
        .with_tool(FanOut)
        .after_tool_progress(move |call, p| seen.lock().unwrap().push((call.id.clone(), p.message.clone().unwrap_or_default())));
    chat.ask("Use the tool").await.unwrap();
    assert_eq!(*calls.lock().unwrap(), [("call_1".to_string(), "Reading in the background".to_string())]);
}

// spec: chat_callbacks_spec.rb:131 runs tools that report progress without a callback
#[tokio::test]
async fn tools_report_progress_without_a_listener() {
    let server = serve(vec![progress_calls(&["a"]), text_response("complete")]).await;
    let mut chat = chat(&server).with_tool(ProgressProbe);
    chat.ask("Use the tool").await.unwrap();
    assert_eq!(chat.messages().iter().find(|m| m.is_tool_result()).unwrap().content(), "a done");
}

/// A tool that fails, to check what concurrent execution does with the error.
struct BlowsUp;

#[async_trait]
impl Tool for BlowsUp {
    fn name(&self) -> String {
        "blows_up".into()
    }
    fn description(&self) -> String {
        "Fails".into()
    }
    async fn execute(&self, _args: Map<String, Value>, _call: &ToolCall) -> Result<ToolResult, ToolError> {
        Err("tool blew up".into())
    }
}

// spec: chat/tool_concurrency_spec.rb:130 raises the first error a threaded tool call produced
#[tokio::test]
async fn concurrent_tool_errors_escape_after_every_call_ends() {
    let server = serve(vec![]).await;
    let mut chat = chat(&server).with_tool(BlowsUp).with_tool(Echo).with_tool_concurrency(true);
    chat.ask_later("go").unwrap();
    chat.add_message(tool_call_message(&[("call_1", "blows_up", json!({})), ("call_2", "echo", json!({ "text": "ok" }))]));
    let err = chat.run_tools().await.unwrap_err();
    assert_eq!(err.to_string(), "tool blew up");
    // The call that succeeded still recorded its result before the error escaped.
    assert_eq!(chat.messages().iter().filter(|m| m.is_tool_result()).map(|m| m.content()).collect::<Vec<_>>(), ["ok"]);
}
