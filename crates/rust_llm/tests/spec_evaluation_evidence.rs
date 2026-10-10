//! RubyLLM 2.1's `evaluation/evidence_spec.rb`: what a returned value becomes as evaluation
//! evidence. Ruby builds the chat with `add_message`; so does this port.

use std::sync::Arc;

use rust_llm::evaluation::{Evidence, Outcome};
use rust_llm::message::indexmap_lite::IndexMap;
use rust_llm::{
    Answer, Attachment, Chat, Config, Embedding, FinishReason, Message, ModerationResult, Role,
    Speech, ToolCall, Transcription, Vectors,
};
use serde_json::{Map, Value, json};

fn call() -> ToolCall {
    let mut args = Map::new();
    args.insert("order_id".into(), json!(42));
    ToolCall::new("call-1", "lookup_order", args)
}

fn config() -> Arc<Config> {
    let mut c = Config::default();
    c.set("openai_api_key", "test");
    Arc::new(c)
}

/// The spec's `chat`: instructions, a refund request, a verified tool call, and the answer.
fn chat() -> Chat {
    let mut chat = Chat::with_config(config(), Some("gpt-5-nano"), Some("openai"), false).unwrap();
    chat = chat.with_instructions("Verify before refunding.");
    chat.add_message(Message::user("Refund order 42"));
    let mut asking = Message::assistant("");
    let calls: IndexMap<ToolCall> = [("call-1".to_string(), call())].into_iter().collect();
    asking.tool_calls = Some(calls);
    chat.add_message(asking);
    let mut result = Message::new(Role::Tool, Some("Order verified".to_string()));
    result.tool_call_id = Some("call-1".into());
    chat.add_message(result);
    let mut answer = Message::assistant("Your refund is approved.");
    answer.finish_reason = Some(FinishReason::Stop);
    chat.add_message(answer);
    chat
}

fn evidence(outcome: Outcome) -> Evidence {
    Evidence::new(&outcome).unwrap()
}

fn argument_error(result: rust_llm::Result<Evidence>, pattern: &str) {
    match result {
        Err(rust_llm::Error::Argument(m)) => assert!(m.contains(pattern), "{m}"),
        other => panic!("expected an ArgumentError matching {pattern}, got {other:?}"),
    }
}

// spec: evaluation/evidence_spec.rb:19
#[test]
fn preserves_the_conversation_and_tool_results_without_raw_provider_payloads() {
    let e = evidence(chat().into());
    assert_eq!(e.output, json!("Your refund is approved."));
    let roles: Vec<Role> = e.messages.iter().map(|m| m.role).collect();
    assert_eq!(
        roles,
        [
            Role::System,
            Role::User,
            Role::Assistant,
            Role::Tool,
            Role::Assistant
        ]
    );
    assert_eq!(e.tool_calls, vec![call()]);
    assert_eq!(e.data["messages"][3]["content"], "Order verified");
    assert_eq!(e.data["messages"][3]["tool_call_id"], "call-1");
    assert_eq!(e.data["complete"], true);
    assert!(e.data["messages"][4].get("raw_reasoning").is_none());
}

// spec: evaluation/evidence_spec.rb:30
#[test]
fn normalizes_an_agent_through_its_existing_chat() {
    let agent = evidence(Outcome::Agent(Box::new(chat())));
    assert_eq!(agent.data, evidence(chat().into()).data);
}

// spec: evaluation/evidence_spec.rb:35
#[test]
fn includes_only_supplied_evidence_for_a_standalone_message() {
    let last = chat().messages().last().unwrap().clone();
    let e = evidence(last.into());
    assert_eq!(e.messages.len(), 1);
    assert_eq!(e.data["content"], "Your refund is approved.");
}

// spec: evaluation/evidence_spec.rb:42
#[test]
fn evaluates_requested_tool_calls_without_executing_them() {
    let mut with_signature = call();
    with_signature.thought_signature = Some("sig".into());
    let e = evidence(with_signature.into());
    assert_eq!(e.tool_calls.len(), 1);
    assert_eq!(e.data["name"], "lookup_order");
    assert_eq!(e.data["arguments"], json!({ "order_id": 42 }));
    assert!(e.data.get("thought_signature").is_none());
}

// spec: evaluation/evidence_spec.rb:50
#[test]
fn recursively_handles_structured_values_and_typed_results() {
    let e = evidence(Outcome::List(vec![
        json!(false).into(),
        json!(0).into(),
        Value::Null.into(),
        Outcome::Map(vec![
            ("call".into(), call().into()),
            (
                "verdict".into(),
                Answer::Probability { probability: 0.8 }.into(),
            ),
        ]),
    ]));
    assert_eq!(
        e.data[3]["verdict"],
        json!({ "type": "probability", "probability": 0.8 })
    );
    assert_eq!(
        e.data.as_array().unwrap()[..3],
        [json!(false), json!(0), Value::Null]
    );
    // `be_frozen`: evidence is an owned snapshot (`Value`), never shared with the result.
}

// spec: evaluation/evidence_spec.rb:58
#[test]
fn does_not_reinterpret_an_ordinary_string_as_a_file_or_url() {
    let e = evidence("https://example.com/private".into());
    assert_eq!(e.data, json!("https://example.com/private"));
    assert!(e.attachments.is_empty());
}

// spec: evaluation/evidence_spec.rb:65
#[test]
fn preserves_attachments_separately_from_textual_evidence() {
    let attachment = Attachment::new("https://example.com/receipt.png");
    let message = Message::assistant("Receipt").with_attachments(vec![attachment.clone()]);
    let e = evidence(message.into());
    assert_eq!(e.attachments.len(), 1);
    assert!(e.attachments[0] == attachment);
    assert_eq!(
        e.data["attachments"],
        json!([{ "attachment": 1, "filename": "receipt.png", "content_type": "image/png" }])
    );
}

struct Opaque;

// spec: evaluation/evidence_spec.rb:74
#[test]
fn requires_explicit_conversion_for_unsupported_objects_and_detects_cycles() {
    argument_error(Evidence::new(&Outcome::other(Opaque)), "adapt");
    let adapter: rust_llm::evaluation::Adapter =
        Arc::new(|v: &(dyn std::any::Any + Send + Sync)| {
            v.downcast_ref::<Opaque>().map(|_| json!("Converted"))
        });
    let e = Evidence::with_adapters(&Outcome::other(Opaque), &[adapter]).unwrap();
    assert_eq!(e.data, json!("Converted"));
    // A cycle (`cycle << cycle`) cannot be built from owned Rust values, so there is nothing to
    // detect: `Outcome` is a tree.
}

// spec: evaluation/evidence_spec.rb:84
#[test]
fn snapshots_text_and_evidence_before_the_returned_chat_changes() {
    let mut outcome = Outcome::Chat(Box::new(chat()));
    let e = Evidence::new(&outcome).unwrap();
    let Outcome::Chat(chat) = &mut outcome else {
        unreachable!()
    };
    chat.messages_mut().last_mut().unwrap().content = Some("Changed later".into());
    chat.add_message(Message::user("Another question"));
    assert_eq!(e.data["messages"][4]["content"], "Your refund is approved.");
    assert_eq!(e.messages.len(), 5);
}

// spec: evaluation/evidence_spec.rb:93
#[test]
fn preserves_incomplete_conversation_state_instead_of_inventing_a_final_answer() {
    let mut chat = chat();
    chat.messages_mut().pop();
    let e = evidence(chat.into());
    assert_eq!(e.data["complete"], false);
    assert_eq!(
        e.data["messages"].as_array().unwrap().last().unwrap()["role"],
        "tool"
    );
}

// spec: evaluation/evidence_spec.rb:101
#[test]
fn distinguishes_a_change_waiting_for_approval_from_an_executed_tool() {
    let tool = rust_llm::FnTool::new("lookup_order", "Look up an order", |_| async {
        panic!("This tool must not run while collecting evidence")
    })
    .requires_approval();
    let mut chat = chat().with_tool(tool);
    chat.messages_mut().pop();
    chat.messages_mut().pop();
    let e = evidence(chat.into());
    assert_eq!(e.data["complete"], false);
    assert_eq!(e.data["waiting"], true);
    assert_eq!(e.data["pending_approvals"], json!(["call-1"]));
    let messages = e.data["messages"].as_array().unwrap();
    assert_eq!(messages.last().unwrap()["role"], "assistant");
    assert!(messages.iter().all(|m| m["role"] != "tool"));
}

// spec: evaluation/evidence_spec.rb:122
#[test]
fn keeps_transcripts_of_nested_agents_distinct() {
    let e = evidence(Outcome::List(vec![
        Outcome::Agent(Box::new(chat())),
        Outcome::Map(vec![("second".into(), chat().into())]),
    ]));
    assert_eq!(e.data[0]["output"], "Your refund is approved.");
    assert_eq!(e.data[1]["second"]["output"], "Your refund is approved.");
}

// spec: evaluation/evidence_spec.rb:129
#[test]
fn adapts_public_operation_results_without_provider_payloads() {
    let transcription = Transcription::new(Some("Good morning".into()), "whisper-1");
    let embedding = Embedding::new(
        Vectors::Single(vec![0.25, -0.5]),
        "text-embedding-3-small".into(),
        None,
    );
    let moderation = ModerationResult::new(false, Vec::new(), Map::new());
    assert_eq!(evidence(transcription.into()).data["text"], "Good morning");
    assert_eq!(
        evidence(embedding.into()).data["vectors"],
        json!([0.25, -0.5])
    );
    assert_eq!(evidence(moderation.into()).data["flagged"], false);
}

// spec: evaluation/evidence_spec.rb:139
#[test]
fn carries_media_bytes_as_attachments_instead_of_putting_binary_data_in_the_prompt() {
    let mut image = rust_llm::Image::new("", Value::Null);
    image.data = Some("aGVsbG8=".into());
    image.mime_type = Some("image/png".into());
    let speech = Speech::new(b"hello".to_vec(), "tts-1", None, None, Some("audio/mpeg"));
    let image_e = evidence(image.into());
    assert_eq!(image_e.attachments.len(), 1);
    assert!(image_e.attachments[0] == "data:image/png;base64,aGVsbG8=");
    let speech_e = evidence(speech.into());
    assert!(speech_e.attachments[0] == "data:audio/mpeg;base64,aGVsbG8=");
    assert!(speech_e.data.get("data").is_none());
}

// spec: evaluation/evidence_spec.rb:148
#[test]
fn rejects_non_finite_numbers_and_conflicting_json_keys() {
    let embedding = Embedding::new(
        Vectors::Single(vec![f64::NAN]),
        "text-embedding-3-small".into(),
        None,
    );
    argument_error(Evidence::new(&embedding.into()), "finite");
    argument_error(
        Evidence::new(&Outcome::Map(vec![
            ("value".into(), json!(1).into()),
            ("value".into(), json!(2).into()),
        ])),
        "Duplicate",
    );
}

// spec: evaluation/evidence_spec.rb:153
#[test]
fn preserves_native_score_distributions_with_integer_level_indexes() {
    let score = Answer::Score {
        score: 0.8,
        levels: vec!["Bad".into(), "Good".into()],
        probabilities: vec![(0, 0.2), (1, 0.8)],
        confidence: 0.6,
    };
    let e = evidence(score.into());
    assert_eq!(e.data["probabilities"], json!({ "0": 0.2, "1": 0.8 }));
    argument_error(
        Evidence::new(&Outcome::Map(vec![
            ("0".into(), json!("First").into()),
            ("0".into(), json!("Second").into()),
        ])),
        "Duplicate",
    );
}
