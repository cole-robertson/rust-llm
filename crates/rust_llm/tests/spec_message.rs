//! `spec/ruby_llm/message_spec.rb` (`.new from #to_h`, `#attachments` in `to_h`, supplied `#cost`,
//! `#tool_results`) and `chat_spec.rb`'s `#tool_results`, ported against `Message::from_h`,
//! `Message::to_h`, `Message::with_cost`, and `Message::tool_results` (`message.rb`).
//! `// spec:` lines tie each test to its Ruby example.

use std::sync::Arc;

use rust_llm::message::indexmap_lite::IndexMap;
use rust_llm::message::{Operation, ServerToolCall};
use rust_llm::model::{PricingCategory, PricingTier};
use rust_llm::{
    Attachment, Chat, Citation, Config, Cost, FinishReason, Message, Model, Role, Thinking, Tokens,
    ToolCall, UsageEntry, UsageStatus,
};
use serde_json::{Map, Value, json};

/// message_spec.rb's `priced-model`: $1 in, $2 out per million.
fn priced() -> Model {
    let mut m = Model::default_for("priced-model", "openai");
    m.name = "Priced Model".into();
    m.pricing.text_tokens = Some(PricingCategory {
        standard: Some(PricingTier {
            input_per_million: Some(1.0),
            output_per_million: Some(2.0),
            ..Default::default()
        }),
        ..Default::default()
    });
    m
}

fn args(value: Value) -> Map<String, Value> {
    value.as_object().cloned().unwrap_or_default()
}

fn calls(list: Vec<ToolCall>) -> Option<IndexMap<ToolCall>> {
    Some(list.into_iter().map(|c| (c.id.clone(), c)).collect())
}

fn report(input: i64, output: i64) -> Message {
    let mut m = Message::assistant("Report");
    m.tokens = Tokens {
        input: Some(input),
        output: Some(output),
        ..Default::default()
    };
    m
}

fn close(actual: Option<f64>, expected: f64) {
    let a = actual.unwrap_or_else(|| panic!("expected {expected}, got None"));
    assert!((a - expected).abs() < 1e-12, "expected {expected}, got {a}");
}

// spec: message_spec.rb:71 .new from #to_h attributes > preserves a remote approval through JSON serialization
#[test]
fn a_remote_approval_survives_json_serialization() {
    let mut call = ToolCall::new("approval_1", "search", args(json!({ "query": "Ruby" })));
    call.remote = true;
    let mut original = Message::assistant("");
    original.tool_calls = calls(vec![call]);

    let attributes: Value = serde_json::from_str(&original.to_h().to_string()).unwrap();
    let rebuilt = Message::from_h(&attributes).unwrap();

    let call = rebuilt
        .tool_calls
        .as_ref()
        .and_then(|c| c.get("approval_1"))
        .expect("approval_1");
    assert_eq!(
        (call.id.as_str(), call.remote, call.arguments()),
        ("approval_1", true, args(json!({ "query": "Ruby" })))
    );
    assert_eq!(rebuilt.to_h(), original.to_h());
}

// spec: message_spec.rb:84 .new from #to_h attributes > rebuilds tool calls, thinking, and citations as value objects
#[test]
fn tool_calls_thinking_and_citations_come_back_as_value_objects() {
    let mut original = Message::assistant("Berlin is sunny.");
    original.tool_calls = calls(vec![ToolCall::new(
        "call_1",
        "weather",
        args(json!({ "city": "Berlin" })),
    )]);
    original.thinking = Some(Thinking {
        text: Some("Check the forecast.".into()),
        signature: Some("sig".into()),
    });
    original.citations = vec![Citation {
        url: Some("https://example.com".into()),
        title: Some("Forecast".into()),
        ..Default::default()
    }];
    original.server_tool_calls = vec![ServerToolCall {
        kind: "web_search".into(),
        name: None,
        id: None,
        input: None,
        result: None,
        raw: json!({ "query": "Berlin" }),
    }];
    original.finish_reason = Some(FinishReason::Stop);

    let rebuilt = Message::from_h(&original.to_h()).unwrap();

    let call = rebuilt
        .tool_calls
        .as_ref()
        .and_then(|c| c.get("call_1"))
        .expect("call_1");
    assert_eq!(
        (call.name.as_str(), call.arguments()),
        ("weather", args(json!({ "city": "Berlin" })))
    );
    assert_eq!(rebuilt.thinking, original.thinking);
    assert_eq!(rebuilt.citations, original.citations);
    assert_eq!(rebuilt.server_tool_calls, original.server_tool_calls);
    assert_eq!(rebuilt.finish_reason, Some(FinishReason::Stop));
    assert_eq!(rebuilt.to_h(), original.to_h());
}

// spec: message_spec.rb:128 #attachments > appears in to_h only when present
#[test]
fn attachments_appear_in_to_h_only_when_present() {
    let image = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/ruby.png");
    let with_files = Message::user("look").with_attachments(vec![Attachment::new(image)]);
    let without_files = Message::user("look");

    let listed = with_files.to_h()["attachments"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0], json!({ "type": "image", "source": image }));
    assert!(without_files.to_h().get("attachments").is_none());
}

// spec: message_spec.rb:154 #cost > preserves an explicitly unknown cost with zero usage through serialization
#[test]
fn an_explicitly_unknown_cost_survives_serialization() {
    let message = report(0, 0).with_cost(Cost::from_h(&json!({}), None));

    assert_eq!(message.cost(None).total(), None);
    assert_eq!(message.to_h()["cost"], json!({}));
    assert_eq!(
        Message::from_h(&message.to_h()).unwrap().cost(None).total(),
        None
    );
}

// spec: message_spec.rb:163 #cost > preserves a supplied cost while allowing explicit model repricing
#[test]
fn a_supplied_cost_is_kept_but_an_explicit_model_reprices() {
    let message = report(1_000, 2_000).with_cost(Cost::from_h(&json!({ "total": 0.02 }), None));

    assert_eq!(message.cost(None).total(), Some(0.02));
    assert_eq!(
        Message::from_h(&message.to_h()).unwrap().cost(None).total(),
        Some(0.02)
    );
    close(message.cost(Some(&priced())).total(), 0.005);
}

// spec: message_spec.rb:172 #cost > uses actual attempt accounting before a supplied cost
#[test]
fn recorded_attempts_win_over_a_supplied_cost() {
    let mut entry = UsageEntry::new(Operation::Chat, "openai", Some("priced-model"));
    entry.status = UsageStatus::Succeeded;
    entry.tokens = Tokens {
        input: Some(1_000),
        output: Some(2_000),
        ..Default::default()
    };
    entry.cost = Cost::from_h(&json!({ "total": 0.03 }), None);
    let mut message =
        Message::assistant("Report").with_cost(Cost::from_h(&json!({ "total": 0.02 }), None));
    message.usage_entries = vec![entry];

    assert_eq!(message.cost(None).total(), Some(0.03));
    assert_eq!(
        Message::from_h(&message.to_h()).unwrap().cost(None).total(),
        Some(0.03)
    );
    close(message.cost(Some(&priced())).total(), 0.005);
}

/// `describe '#tool_results'`: an assistant turn calling two tools, then both results.
fn conversation() -> Vec<Message> {
    let mut call = Message::assistant("");
    call.tool_calls = calls(vec![
        ToolCall::new("call_1", "weather", Map::new()),
        ToolCall::new("call_2", "time", Map::new()),
    ]);
    vec![
        call,
        Message::tool_result("call_1", "sunny"),
        Message::tool_result("call_2", "noon"),
    ]
}

// spec: message_spec.rb:346 #tool_results > returns the tool result messages answering the calls
#[test]
fn tool_results_are_the_messages_answering_the_calls() {
    let messages = conversation();
    let [call, weather, time] = [&messages[0], &messages[1], &messages[2]];

    assert_eq!(call.tool_results(&messages), vec![weather, time]);
}

// spec: message_spec.rb:350 #tool_results > returns an empty array for messages that made no tool calls
#[test]
fn a_message_without_tool_calls_has_no_tool_results() {
    let messages = conversation();

    assert!(messages[1].tool_results(&messages).is_empty());
}

// spec: chat_spec.rb:157 #tool_results > links added messages so a call resolves its result messages
#[test]
fn added_messages_resolve_a_calls_results() {
    let mut config = Config::default();
    config.set("openai_api_key", "test");
    let config = Arc::new(config);
    let default_model = config.default_model.clone();
    let mut chat = Chat::with_config(config, Some(&default_model), None, false).unwrap();
    let mut call = Message::new(Role::Assistant, Some(String::new()));
    call.tool_calls = calls(vec![ToolCall::new("call_1", "weather", Map::new())]);
    let call = chat.add_message(call).clone();
    let result = chat
        .add_message(Message::tool_result("call_1", "sunny"))
        .clone();

    assert_eq!(call.tool_results(chat.messages()), vec![&result]);
}
