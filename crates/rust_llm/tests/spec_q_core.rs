//! RubyLLM 2.1 core value types and the Anthropic protocol: `tokens_spec.rb`, `message_spec.rb`,
//! `cost_spec.rb` (cache writes by lifetime), `tool_call_spec.rb`, `chat_functions_spec.rb`
//! (`#messages=`), `provider_spec.rb` (FastAPI errors), `provider_account_identity_spec.rb`,
//! `chat_server_tool_approval_spec.rb`, and the 2.1 rows of `protocols/anthropic/{chat,streaming}`
//! and `anthropic_compaction_spec.rb`. `// spec:` lines tie each test to its Ruby example.

mod spec_helpers;

use std::sync::Arc;

use rust_llm::message::RawResponse;
use rust_llm::message::indexmap_lite::IndexMap;
use rust_llm::model::{PricingCategory, PricingTier};
use rust_llm::protocols::anthropic::{self, StreamBlocks};
use rust_llm::tokens::Tokens;
use rust_llm::{
    Attachment, Chat, Config, Cost, Error, Message, Model, ProtocolName, Provider, Resolution,
    Thinking, ThinkingConfig, ToolCall, UsageEntry, UsageStatus,
};
use serde_json::{Map, Value, json};
use spec_helpers::*;

fn fixture(name: &str) -> String {
    format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
}

fn counts(value: Value) -> Option<Map<String, Value>> {
    value.as_object().cloned()
}

/// `JSON.parse(JSON.generate(message.to_h))`, then `Message.new`.
fn json_round_trip(message: &Message) -> Message {
    let text = serde_json::to_string(&message.to_h()).unwrap();
    Message::from_h(&serde_json::from_str(&text).unwrap()).unwrap()
}

fn close(actual: Option<f64>, expected: f64, tolerance: f64) {
    let a = actual.unwrap_or_else(|| panic!("expected {expected}, got None"));
    assert!(
        (a - expected).abs() < tolerance,
        "expected {expected}, got {a}"
    );
}

// ---- tokens_spec.rb -----------------------------------------------------------------------------

// spec: tokens_spec.rb:7 #server_tool_use > counts each tool that ran under a String key
#[test]
fn server_tool_use_counts_each_tool_under_a_string_key() {
    let tokens = Tokens::default()
        .with_server_tool_use(&json!({ "web_search_requests": 2, "web_fetch_requests": 1 }));
    assert_eq!(
        tokens.server_tool_use,
        counts(json!({ "web_search_requests": 2, "web_fetch_requests": 1 }))
    );
}

// spec: tokens_spec.rb:13 #server_tool_use > leaves out tools that did not run
#[test]
fn server_tool_use_leaves_out_tools_that_did_not_run() {
    let tokens = Tokens::default()
        .with_server_tool_use(&json!({ "web_search_requests": 1, "web_fetch_requests": 0 }));
    assert_eq!(
        tokens.server_tool_use,
        counts(json!({ "web_search_requests": 1 }))
    );
}

// spec: tokens_spec.rb:19 #server_tool_use > is nil when no tool ran
#[test]
fn server_tool_use_is_none_when_no_tool_ran() {
    let none_ran = Tokens::default().with_server_tool_use(&json!({ "web_search_requests": 0 }));
    assert_eq!(none_ran.server_tool_use, None);
    let input_only = Tokens {
        input: Some(10),
        ..Default::default()
    };
    assert_eq!(input_only.server_tool_use, None);
}

// spec: tokens_spec.rb:26 .aggregate > sums server tool counts across attempts
#[test]
fn aggregate_sums_server_tool_counts() {
    let a = Tokens::default().with_server_tool_use(&json!({ "web_search_requests": 1 }));
    let b = Tokens {
        input: Some(10),
        ..Default::default()
    };
    let c = Tokens::default()
        .with_server_tool_use(&json!({ "web_search_requests": 2, "web_fetch_requests": 1 }));
    assert_eq!(
        Tokens::aggregate([&a, &b, &c]).server_tool_use,
        counts(json!({ "web_search_requests": 3, "web_fetch_requests": 1 }))
    );
}

// spec: tokens_spec.rb:38 .aggregate > sums cache writes by lifetime across attempts
#[test]
fn aggregate_sums_cache_writes_by_lifetime() {
    let a = Tokens {
        cache_write: Some(10),
        ..Default::default()
    }
    .with_cache_write_by_ttl(&json!({ "1h": 10 }));
    let b = Tokens {
        cache_write: Some(5),
        ..Default::default()
    }
    .with_cache_write_by_ttl(&json!({ "5m": 2, "1h": 3 }));
    assert_eq!(
        Tokens::aggregate([&a, &b]).cache_write_by_ttl,
        counts(json!({ "1h": 13, "5m": 2 }))
    );
}

// spec: tokens_spec.rb:51 #cache_write_by_ttl > leaves out lifetimes with no writes
#[test]
fn cache_write_by_ttl_leaves_out_lifetimes_with_no_writes() {
    let tokens = Tokens::default().with_cache_write_by_ttl(&json!({ "5m": 0, "1h": 20 }));
    assert_eq!(tokens.cache_write_by_ttl, counts(json!({ "1h": 20 })));
    let none = Tokens::default().with_cache_write_by_ttl(&json!({ "5m": 0 }));
    assert_eq!(none.cache_write_by_ttl, None);
}

// spec: tokens_spec.rb:58 #cache_write_by_ttl > survives a round trip through a serialized message
#[test]
fn cache_write_by_ttl_survives_a_message_round_trip() {
    let mut message = Message::assistant("Hi");
    message.tokens = Tokens {
        cache_write: Some(20),
        ..Default::default()
    }
    .with_cache_write_by_ttl(&json!({ "1h": 20 }));
    let restored = Message::from_h(&message.to_h()).unwrap();
    assert_eq!(
        restored.tokens().cache_write_by_ttl,
        counts(json!({ "1h": 20 }))
    );
}

// ---- message_spec.rb ----------------------------------------------------------------------------

fn signature_only() -> Message {
    let mut m = Message::assistant("Done.");
    m.thinking = Some(Thinking {
        text: None,
        signature: Some("opaque-signature".into()),
    });
    m
}

fn assert_signature_only(m: &Message) {
    assert_eq!(
        m.thinking,
        Some(Thinking {
            text: None,
            signature: Some("opaque-signature".into())
        })
    );
}

// spec: message_spec.rb:90 .new from #to_h attributes > with signature-only thinking > preserves thinking without text through a Hash round trip
#[test]
fn signature_only_thinking_survives_a_hash_round_trip() {
    let original = signature_only();
    let rebuilt = Message::from_h(&original.to_h()).unwrap();
    assert_signature_only(&rebuilt);
    assert_eq!(rebuilt.to_h(), original.to_h());
}

// spec: message_spec.rb:97 .new from #to_h attributes > with signature-only thinking > preserves thinking without text through a JSON round trip
#[test]
fn signature_only_thinking_survives_a_json_round_trip() {
    let original = signature_only();
    let rebuilt = json_round_trip(&original);
    assert_signature_only(&rebuilt);
    assert_eq!(rebuilt.to_h(), original.to_h());
}

// spec: message_spec.rb:106 .new from #to_h attributes > preserves an empty thinking text with its signature
#[test]
fn an_empty_thinking_text_keeps_its_signature() {
    let mut original = Message::assistant("Done.");
    original.thinking = Some(Thinking {
        text: Some(String::new()),
        signature: Some("opaque-signature".into()),
    });
    assert_eq!(
        Message::from_h(&original.to_h()).unwrap().thinking,
        Some(Thinking {
            text: Some(String::new()),
            signature: Some("opaque-signature".into())
        })
    );
}

// spec: message_spec.rb:114 .new from #to_h attributes > preserves thinking text without a signature
#[test]
fn thinking_text_without_a_signature_survives() {
    let mut original = Message::assistant("Done.");
    original.thinking = Some(Thinking {
        text: Some("Checking.".into()),
        signature: None,
    });
    assert_eq!(
        Message::from_h(&original.to_h()).unwrap().thinking,
        Some(Thinking {
            text: Some("Checking.".into()),
            signature: None
        })
    );
}

fn image_message() -> Message {
    let image = Attachment::new(fixture("ruby.png"));
    let mut h = image.to_h();
    h["filename"] = "custom.png".into();
    h["resolution"] = "high".into();
    Message::user("Look").with_attachments(vec![Attachment::from_h(&h).unwrap()])
}

// spec: message_spec.rb:151 .new from #to_h attributes > with attachments > rebuilds attachments as usable value objects
#[tokio::test]
async fn attachments_rebuild_as_usable_value_objects() {
    let original = image_message();
    let mut rebuilt = Message::from_h(&original.to_h()).unwrap();
    assert_eq!(rebuilt.attachments.len(), 1);
    let a = &mut rebuilt.attachments[0];
    assert_eq!(a.filename.as_deref(), Some("custom.png"));
    assert_eq!(a.mime_type, "image/png");
    assert_eq!(a.resolution, Some(Resolution::High));
    assert_eq!(
        a.content().await.unwrap(),
        std::fs::read(fixture("ruby.png")).unwrap()
    );
    assert_eq!(rebuilt.to_h(), original.to_h());
}

// spec: message_spec.rb:161 .new from #to_h attributes > with attachments > rebuilds attachments through JSON serialization
#[test]
fn attachments_rebuild_through_json() {
    let original = image_message();
    assert_eq!(json_round_trip(&original).to_h(), original.to_h());
}

// spec: message_spec.rb:172 #thinking > builds thinking from a signature without text
#[test]
fn thinking_builds_from_a_signature_without_text() {
    let m = Message::from_h(&json!({
        "role": "assistant", "content": "Done.", "thinking_signature": "opaque-signature"
    }))
    .unwrap();
    assert_signature_only(&m);
}

// spec: message_spec.rb:179 #thinking > keeps thinking absent when its text is nil and its signature is #{signature.inspect}
#[test]
fn thinking_stays_absent_without_text_or_signature() {
    for signature in [Value::Null, json!("")] {
        let m = Message::from_h(&json!({
            "role": "assistant", "content": "Done.", "thinking": null, "thinking_signature": signature
        }))
        .unwrap();
        assert_eq!(m.thinking, None, "signature {signature}");
    }
}

// spec: message_spec.rb:187 #thinking > keeps an existing thinking object and its signature
// (`Message.new(thinking: Thinking, thinking_signature:)`: the Rust port sets the `Thinking` value
// directly, so the object's own signature is the one kept.)
#[test]
fn an_existing_thinking_keeps_its_signature() {
    let thinking = Thinking {
        text: None,
        signature: Some("original-signature".into()),
    };
    let mut m = Message::assistant("Done.");
    m.thinking = Some(thinking.clone());
    assert_eq!(m.thinking, Some(thinking));
    assert_eq!(
        m.thinking.and_then(|t| t.signature).as_deref(),
        Some("original-signature")
    );
}

// spec: message_spec.rb:196 #thinking > keeps the signature supplied inside a thinking Hash
#[test]
fn a_thinking_hash_keeps_its_own_signature() {
    let m = Message::from_h(&json!({
        "role": "assistant", "content": "Done.",
        "thinking": { "signature": "original-signature" },
        "thinking_signature": "other-signature"
    }))
    .unwrap();
    assert_eq!(
        m.thinking,
        Some(Thinking {
            text: None,
            signature: Some("original-signature".into())
        })
    );
}

/// message_spec.rb's `priced-model`: $1 in, $2 out per million.
fn priced() -> Model {
    let mut m = Model::default_for("priced-model", "openai");
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

// spec: message_spec.rb:283 #cost > preserves the provider-reported cost through a JSON round trip
#[test]
fn the_reported_cost_survives_a_json_round_trip() {
    let mut message = Message::assistant("Report");
    message.model = Some("priced-model".into());
    message.tokens = Tokens {
        input: Some(1_000),
        output: Some(2_000),
        reported_cost: Some(0.015),
        ..Default::default()
    };
    let rebuilt = json_round_trip(&message);
    assert_eq!(message.to_h()["reported_cost"], json!(0.015));
    assert_eq!(rebuilt.cost(Some(&priced())).total(), Some(0.015));
    assert_eq!(rebuilt.cost(None).total(), Some(0.015));
    assert_eq!(rebuilt.to_h(), message.to_h());
}

// spec: message_spec.rb:297 #cost > serializes the cost each attempt reported
#[test]
fn each_attempts_reported_cost_is_serialized() {
    let model = priced();
    let mut message = Message::assistant("Report");
    message.usage_entries = [0.01, 0.02]
        .into_iter()
        .map(|reported| {
            let tokens = Tokens {
                input: Some(1_000),
                output: Some(2_000),
                reported_cost: Some(reported),
                ..Default::default()
            };
            UsageEntry {
                id: UsageEntry::next_id(),
                owner: None,
                operation: rust_llm::message::Operation::Chat,
                provider: model.provider.clone(),
                model: model.id.clone(),
                status: UsageStatus::Succeeded,
                cost: Cost::new(&tokens, Some(&model), rust_llm::cost::Tier::Standard),
                tokens,
            }
        })
        .collect();
    close(message.to_h()["reported_cost"].as_f64(), 0.03, 1e-12);
    close(
        Message::from_h(&message.to_h()).unwrap().cost(None).total(),
        0.03,
        1e-12,
    );
}

// spec: message_spec.rb:310 #cost > omits a reported cost the provider did not send
#[test]
fn an_unreported_cost_is_omitted() {
    let mut message = Message::assistant("Hello");
    message.tokens.input = Some(1_000);
    assert!(message.to_h().get("reported_cost").is_none());
}

// ---- chat_functions_spec.rb #messages= ----------------------------------------------------------

fn openai_chat() -> Chat {
    let mut config = Config::default();
    config.set("openai_api_key", "test");
    let config = Arc::new(config);
    let model = config.default_model.clone();
    Chat::with_config(config, Some(&model), None, false).unwrap()
}

/// `chat.messages = attributes`: each serialized message rebuilt through `Message.new`.
fn restore(chat: &mut Chat, attributes: &[Value]) {
    chat.set_messages(
        attributes
            .iter()
            .map(|h| Message::from_h(h).unwrap())
            .collect(),
    );
}

// spec: chat_functions_spec.rb:427 #messages= > restores attachments from serialized messages
#[test]
fn messages_assignment_restores_attachments() {
    let mut source = openai_chat();
    source.add_message(image_message());
    let mut chat = openai_chat();
    let attributes: Vec<Value> = source.messages().iter().map(Message::to_h).collect();
    restore(&mut chat, &attributes);
    let a = &chat.messages()[0].attachments;
    assert_eq!(a.len(), 1);
    assert_eq!(
        (
            a[0].filename.as_deref(),
            a[0].mime_type.as_str(),
            a[0].resolution
        ),
        (Some("custom.png"), "image/png", Some(Resolution::High))
    );
}

// spec: chat_functions_spec.rb:441 #messages= > restores signature-only thinking from a JSON transcript
#[test]
fn messages_assignment_restores_signature_only_thinking() {
    let mut source = openai_chat();
    source.add_message(Message::user("Hello"));
    source.add_message(signature_only());
    let text = serde_json::to_string(
        &source
            .messages()
            .iter()
            .map(Message::to_h)
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let attributes: Vec<Value> = serde_json::from_str(&text).unwrap();
    let mut chat = openai_chat();
    restore(&mut chat, &attributes);
    assert_signature_only(chat.messages().last().unwrap());
    let restored: Vec<Value> = chat.messages().iter().map(Message::to_h).collect();
    let original: Vec<Value> = source.messages().iter().map(Message::to_h).collect();
    assert_eq!(restored, original);
}

// ---- cost_spec.rb: cache writes by lifetime ---------------------------------------------------

fn claude() -> Model {
    rust_llm::models()
        .find("claude-sonnet-4-6", Some("anthropic"))
        .unwrap()
}

fn input_price(m: &Model) -> f64 {
    m.pricing
        .text_tokens
        .as_ref()
        .and_then(|t| t.standard.as_ref())
        .and_then(|t| t.input_per_million)
        .unwrap()
}

fn cache_write_price(m: &Model) -> f64 {
    m.pricing
        .text_tokens
        .as_ref()
        .and_then(|t| t.standard.as_ref())
        .and_then(|t| t.cache_write_input_per_million)
        .unwrap()
}

fn writes(input: Option<i64>, total: i64, by_ttl: Value) -> Tokens {
    Tokens {
        input,
        cache_write: Some(total),
        ..Default::default()
    }
    .with_cache_write_by_ttl(&by_ttl)
}

fn cost_of(tokens: &Tokens, model: &Model) -> Cost {
    Cost::new(tokens, Some(model), rust_llm::cost::Tier::Standard)
}

// spec: cost_spec.rb:238 cache writes by lifetime > prices one-hour Anthropic cache writes at twice the input price
#[test]
fn one_hour_anthropic_writes_cost_twice_the_input_price() {
    let cost = cost_of(
        &writes(Some(0), 100_000, json!({ "1h": 100_000 })),
        &claude(),
    );
    close(cost.cache_write, 0.6, 1e-7);
}

// spec: cost_spec.rb:245 cache writes by lifetime > prices five-minute writes in the same request at the registry cache-write price
#[test]
fn five_minute_writes_keep_the_registry_price() {
    let model = claude();
    let cost = cost_of(
        &writes(Some(0), 3_000, json!({ "5m": 1_000, "1h": 2_000 })),
        &model,
    );
    let expected =
        (1_000.0 * cache_write_price(&model) + 2_000.0 * input_price(&model) * 2.0) / 1_000_000.0;
    close(cost.cache_write, expected, 1e-7);
    close(cost.total(), expected, 1e-7);
}

/// cost_spec.rb's `priced-model` (openai): $1 in, $2 out, $0.25 cache read, $1.25 cache write.
fn openai_priced() -> Model {
    let mut m = Model::default_for("priced-model", "openai");
    m.pricing.text_tokens = Some(PricingCategory {
        standard: Some(PricingTier {
            input_per_million: Some(1.0),
            output_per_million: Some(2.0),
            cache_read_input_per_million: Some(0.25),
            cache_write_input_per_million: Some(1.25),
            ..Default::default()
        }),
        ..Default::default()
    });
    m
}

// spec: cost_spec.rb:254 cache writes by lifetime > keeps the registry cache-write price for lifetimes the provider does not price separately
#[test]
fn unpriced_lifetimes_keep_the_registry_cache_write_price() {
    let cost = cost_of(&writes(None, 100, json!({ "1h": 100 })), &openai_priced());
    close(cost.cache_write, 0.000125, 1e-10);
}

// spec: cost_spec.rb:261 cache writes by lifetime > leaves the cache-write cost unknown when the input price is missing
#[test]
fn a_missing_input_price_leaves_cache_writes_unknown() {
    let mut unpriced = Model::default_for("claude-unpriced", "anthropic");
    unpriced.pricing.text_tokens = Some(PricingCategory {
        standard: Some(PricingTier {
            cache_write_input_per_million: Some(3.75),
            ..Default::default()
        }),
        ..Default::default()
    });
    let cost = cost_of(&writes(None, 100, json!({ "1h": 100 })), &unpriced);
    assert_eq!(cost.cache_write, None);
    assert_eq!(cost.total(), None);
}

// ---- tool_call_spec.rb --------------------------------------------------------------------------

fn from_model() -> Map<String, Value> {
    // serde_json keeps insertion order (`preserve_order`), like a Ruby Hash.
    let mut m = Map::new();
    m.insert("start_date".into(), "2026-10-05".into());
    m.insert("sort".into(), "start_date".into());
    m.insert("direction".into(), "asc".into());
    m
}

fn from_jsonb() -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("sort".into(), "start_date".into());
    m.insert("direction".into(), "asc".into());
    m.insert("start_date".into(), "2026-10-05".into());
    m
}

// spec: tool_call_spec.rb:9 orders argument keys the way jsonb stores them
#[test]
fn argument_keys_are_ordered_like_jsonb() {
    let call = ToolCall::new("call_1", "search_trips", from_model());
    let keys: Vec<String> = call.arguments().keys().cloned().collect();
    assert_eq!(keys, ["sort", "direction", "start_date"]);
}

// spec: tool_call_spec.rb:15 orders nested hashes, including those inside arrays
#[test]
fn nested_argument_hashes_are_ordered_too() {
    let mut filter = Map::new();
    filter.insert("value".into(), 1.into());
    filter.insert("op".into(), "eq".into());
    let mut b = Map::new();
    b.insert("zz".into(), 1.into());
    b.insert("a".into(), 2.into());
    let mut arguments = Map::new();
    arguments.insert("filters".into(), json!([Value::Object(filter)]));
    arguments.insert("b".into(), Value::Object(b));
    let call = ToolCall::new("call_1", "search", arguments);
    assert_eq!(
        serde_json::to_string(&call.arguments()).unwrap(),
        r#"{"b":{"a":2,"zz":1},"filters":[{"op":"eq","value":1}]}"#
    );
}

// spec: tool_call_spec.rb:24 leaves streamed argument fragments untouched
#[test]
fn streamed_argument_fragments_are_untouched() {
    let mut call = ToolCall::opening("call_1".into(), "search".into(), r#"{"q":"#.into());
    if let rust_llm::message::ToolArguments::Partial(s) = &mut call.arguments {
        s.push_str(r#""ruby"}"#);
    }
    assert_eq!(
        call.arguments,
        rust_llm::message::ToolArguments::Partial(r#"{"q":"ruby"}"#.into())
    );
}

// spec: tool_call_spec.rb:33 renders the same #{provider} request before and after a jsonb round trip
#[test]
fn the_same_request_renders_before_and_after_a_jsonb_round_trip() {
    for (provider, model) in [("anthropic", MODEL), ("openai", "gpt-5-nano")] {
        let render = |arguments: Map<String, Value>| {
            let mut config = Config::default();
            config.set("anthropic_api_key", "test");
            config.set("openai_api_key", "test");
            let mut chat =
                Chat::with_config(Arc::new(config), Some(model), Some(provider), false).unwrap();
            chat.add_message(Message::user("What trips do I have coming up?"));
            let mut call = Message::assistant("");
            let mut calls = IndexMap::new();
            calls.insert(
                "call_1".into(),
                ToolCall::new("call_1", "search_trips", arguments),
            );
            call.tool_calls = Some(calls);
            chat.add_message(call);
            chat.add_message(Message::tool_result("call_1", "[]"));
            serde_json::to_string(&chat.render().unwrap()).unwrap()
        };
        assert_eq!(render(from_jsonb()), render(from_model()), "{provider}");
    }
}

// ---- provider_spec.rb: FastAPI detail arrays ---------------------------------------------------

// spec: provider_spec.rb:723 #parse_error body shapes > reads a FastAPI detail array of validation errors
#[test]
fn a_fastapi_detail_array_is_read() {
    let body = json!({ "detail": [{ "loc": ["body", "messages"], "msg": "field required" }] });
    let message = Provider::OpenAI.parse_error(&body.to_string()).unwrap();
    assert!(
        message.contains("body.messages: field required"),
        "{message}"
    );
}

// spec: provider_spec.rb:729 #parse_error body shapes > stringifies a numeric msg in a FastAPI detail array
#[test]
fn a_numeric_fastapi_msg_is_stringified() {
    let body = json!({ "detail": [{ "loc": ["body", "messages"], "msg": 422 }] });
    assert_eq!(
        Provider::OpenAI.parse_error(&body.to_string()).as_deref(),
        Some("body.messages: 422")
    );
}

// ---- provider_account_identity_spec.rb ---------------------------------------------------------

fn identity(provider: Provider, settings: &[(&str, &str)]) -> Option<String> {
    let mut config = Config::default();
    for (option, value) in settings {
        config.set(*option, *value);
    }
    provider.account_identity(&config)
}

// spec: provider_account_identity_spec.rb:20 identifies a #{slug} account by its API key and endpoint
#[test]
fn an_account_is_identified_by_its_key_and_endpoint() {
    let rows: [(Provider, &str, &[&str]); 6] = [
        (
            Provider::OpenAI,
            "openai_api_key",
            &[
                "openai_api_base",
                "openai_organization_id",
                "openai_project_id",
            ],
        ),
        (
            Provider::Anthropic,
            "anthropic_api_key",
            &["anthropic_api_base"],
        ),
        (Provider::Gemini, "gemini_api_key", &["gemini_api_base"]),
        (
            Provider::OpenRouter,
            "openrouter_api_key",
            &["openrouter_api_base"],
        ),
        (Provider::XAI, "xai_api_key", &["xai_api_base"]),
        (
            Provider::DeepSeek,
            "deepseek_api_key",
            &["deepseek_api_base"],
        ),
    ];
    for (provider, key, scope) in rows {
        let account = identity(provider, &[(key, "sk-one")]).unwrap();
        assert!(
            account.len() == 64 && account.chars().all(|c| c.is_ascii_hexdigit()),
            "{account}"
        );
        assert!(!account.contains("sk-one"));
        assert_eq!(
            identity(provider, &[(key, "sk-one")]),
            Some(account.clone())
        );
        assert_ne!(
            identity(provider, &[(key, "sk-two")]),
            Some(account.clone())
        );
        for option in scope {
            assert_ne!(
                identity(provider, &[(key, "sk-one"), (option, "https://other.test")]),
                Some(account.clone()),
                "{} {option}",
                provider.slug()
            );
        }
    }
}

// spec: provider_account_identity_spec.rb:66 leaves the account unnamed for providers that do not reuse stored uploads
#[test]
fn providers_that_do_not_reuse_uploads_name_no_account() {
    assert_eq!(
        identity(Provider::Mistral, &[("mistral_api_key", "sk-one")]),
        None
    );
}

// ---- chat_server_tool_approval_spec.rb ---------------------------------------------------------

// spec: chat_server_tool_approval_spec.rb:60 raises a RubyLLM error for a server #{approved ? 'approval' : 'denial'} after a move to another provider
#[tokio::test]
async fn a_server_decision_after_moving_providers_raises() {
    for approved in [true, false] {
        let server = serve(vec![json!({
            "id": "resp_1", "object": "response", "status": "completed", "model": "gpt-5-nano",
            "output": [{ "type": "mcp_approval_request", "id": "approval_1", "name": "search",
                         "arguments": "{\"query\":\"Ruby\"}", "server_label": "docs" }],
            "usage": { "input_tokens": 1, "output_tokens": 1 }
        })])
        .await;
        let mut c = (*config(&server)).clone();
        c.set("anthropic_api_key", "test");
        let chat = Chat::with_config(Arc::new(c), Some("gpt-5-nano"), Some("openai"), false)
            .unwrap()
            .with_protocol(ProtocolName::Responses);
        let mut chat = chat;
        chat.ask("Look up Ruby").await.unwrap();
        let mut chat = chat.with_model(MODEL, Some("anthropic")).unwrap();
        if approved {
            chat.approve("approval_1")
        } else {
            chat.deny("approval_1")
        };
        let Err(err) = chat.run_tools().await else {
            panic!("expected an error");
        };
        assert!(
            err.to_string()
                .contains("Anthropic doesn't support remote tool approvals"),
            "{err}"
        );
    }
}

// ---- protocols/anthropic/chat_spec.rb -----------------------------------------------------------

fn anthropic_config() -> Arc<Config> {
    let mut c = Config::default();
    c.set("anthropic_api_key", "test");
    Arc::new(c)
}

/// `render_payload` with `thinking` for an Anthropic model the registry may not know.
fn thinking_payload(model_id: &str, thinking: ThinkingConfig) -> Value {
    let mut chat = Chat::with_config(anthropic_config(), Some(model_id), Some("anthropic"), true)
        .unwrap()
        .with_thinking(thinking);
    chat.ask_later("Hello").unwrap();
    chat.render().unwrap()
}

/// `Thinking::Config.new(enabled: false)`.
fn off() -> ThinkingConfig {
    let mut thinking = ThinkingConfig::default();
    thinking.enabled = Some(false);
    thinking
}

// spec: protocols/anthropic/chat_spec.rb:579 .render_payload with thinking > sends disabled thinking when the model accepts it
#[test]
fn disabled_thinking_goes_to_models_that_accept_it() {
    let p = thinking_payload("claude-sonnet-5", off());
    assert_eq!(p["thinking"], json!({ "type": "disabled" }));
}

// spec: protocols/anthropic/chat_spec.rb:589 .render_payload with thinking > sends between_tools when turning thinking off on Sonnet 5.5
#[test]
fn sonnet_5_5_turns_thinking_off_between_tools() {
    let p = thinking_payload("claude-sonnet-5-5", off());
    assert_eq!(p["thinking"], json!({ "type": "between_tools" }));
    assert!(p.get("output_config").is_none());
}

// spec: protocols/anthropic/chat_spec.rb:614 .render_payload with thinking > does not treat accidental claude-sonnet-5-5 suffixes as Sonnet 5.5
#[test]
fn accidental_sonnet_5_5_suffixes_are_not_sonnet_5_5() {
    let p = thinking_payload("evilclaude-sonnet-5-5", off());
    assert_eq!(p["thinking"], json!({ "type": "disabled" }));
    assert!(!anthropic::is_between_tools_off("evilclaude-sonnet-5-5"));
    assert!(anthropic::is_between_tools_off(
        "anthropic.claude-sonnet-5-5"
    ));
}

// spec: protocols/anthropic/chat_spec.rb:620 .render_payload with thinking > resolves with_thinking(false) to between_tools on Sonnet 5.5
#[test]
fn with_thinking_false_resolves_to_between_tools_on_sonnet_5_5() {
    let mut model = Model::default_for("claude-sonnet-5-5", "anthropic");
    model.metadata.insert(
        "reasoning_options".into(),
        json!([{ "type": "effort", "values": ["low", "medium", "high", "xhigh", "max"] }]),
    );
    let resolved = ThinkingConfig::off().resolve(&model).unwrap().unwrap();
    assert_eq!(resolved.enabled, Some(false));
    let p = thinking_payload("claude-sonnet-5-5", resolved);
    assert_eq!(p["thinking"], json!({ "type": "between_tools" }));
}

// spec: protocols/anthropic/chat_spec.rb:845 .build_thinking_payload > sends disabled for models that still accept it
#[test]
fn build_thinking_payload_sends_disabled() {
    let p = thinking_payload(MODEL, off());
    assert_eq!(p["thinking"], json!({ "type": "disabled" }));
}

// spec: protocols/anthropic/chat_spec.rb:851 .build_thinking_payload > sends between_tools for Claude Sonnet 5.5
#[test]
fn build_thinking_payload_sends_between_tools_for_sonnet_5_5() {
    let p = thinking_payload("claude-sonnet-5-5", off());
    assert_eq!(p["thinking"], json!({ "type": "between_tools" }));
}

// spec: protocols/anthropic/chat_spec.rb:658 #parse_completion_response > splits cache writes by lifetime so one-hour writes are priced as such
#[test]
fn cache_writes_split_by_lifetime_are_priced_as_such() {
    let mut message = anthropic::parse_completion_body(
        &json!({
            "model": "claude-sonnet-4-6",
            "content": [{ "type": "text", "text": "Hi!" }],
            "usage": {
                "input_tokens": 3, "output_tokens": 5, "cache_creation_input_tokens": 100_000,
                "cache_creation": { "ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 100_000 }
            }
        }),
        RawResponse::default(),
    )
    .unwrap();
    assert_eq!(message.tokens.cache_write, Some(100_000));
    assert_eq!(
        message.tokens.cache_write_by_ttl,
        counts(json!({ "1h": 100_000 }))
    );
    message.model = Some("claude-sonnet-4-6".into());
    close(message.cost(Some(&claude())).cache_write, 0.6, 1e-7);
}

fn claude_answer(thinking: Thinking, producer: Option<&str>) -> Message {
    let mut m = Message::assistant("hi");
    m.thinking = Some(thinking);
    if let Some(provider) = producer {
        m.usage_entries = vec![UsageEntry {
            id: UsageEntry::next_id(),
            owner: None,
            operation: rust_llm::message::Operation::Chat,
            provider: provider.into(),
            model: MODEL.into(),
            status: UsageStatus::Succeeded,
            tokens: Tokens::default(),
            cost: Cost::default(),
        }];
    }
    m
}

// spec: protocols/anthropic/chat_spec.rb:713 .build_thinking_block > sends no block without a signature Claude issued
#[test]
fn no_thinking_block_goes_without_a_signature_claude_issued() {
    let unsigned = claude_answer(
        Thinking {
            text: Some("why".into()),
            signature: None,
        },
        Some("anthropic"),
    );
    let unknown = claude_answer(
        Thinking {
            text: Some("why".into()),
            signature: Some("gemini-signature".into()),
        },
        None,
    );
    assert_eq!(anthropic::build_thinking_block(&unsigned), None);
    assert_eq!(anthropic::build_thinking_block(&unknown), None);
}

// ---- protocols/anthropic/streaming_spec.rb ------------------------------------------------------

// spec: protocols/anthropic/streaming_spec.rb:40 reads the cache write lifetimes from message_start usage
#[test]
fn message_start_usage_carries_cache_write_lifetimes() {
    let chunk = anthropic::build_chunk(
        &mut StreamBlocks::default(),
        &json!({
            "type": "message_start",
            "message": { "model": "claude-sonnet-4-5", "usage": {
                "input_tokens": 3, "cache_creation_input_tokens": 300,
                "cache_creation": { "ephemeral_5m_input_tokens": 100, "ephemeral_1h_input_tokens": 200 }
            } }
        }),
    );
    assert_eq!(
        chunk.tokens.cache_write_by_ttl,
        counts(json!({ "5m": 100, "1h": 200 }))
    );
}

// spec: protocols/anthropic/streaming_spec.rb:59 appends streamed text to its content block in place
// (Ruby checks object identity of the String; the port appends with `String::push_str` on the
// block's own value, which this checks through the reconstructed block text.)
#[test]
fn streamed_text_appends_to_its_block() {
    let mut state = StreamBlocks::default();
    let delta = |text: &str| json!({ "type": "content_block_delta", "index": 0, "delta": { "type": "text_delta", "text": text } });
    anthropic::build_chunk(
        &mut state,
        &json!({ "type": "content_block_start", "index": 0, "content_block": { "type": "text", "text": "" } }),
    );
    anthropic::build_chunk(&mut state, &delta("Hello"));
    anthropic::build_chunk(&mut state, &delta(", world"));
    anthropic::build_chunk(
        &mut state,
        &json!({ "type": "content_block_start", "index": 1,
                 "content_block": { "type": "server_tool_use", "id": "s1", "name": "web_search", "input": {} } }),
    );
    anthropic::build_chunk(
        &mut state,
        &json!({ "type": "content_block_stop", "index": 1 }),
    );
    let done = anthropic::build_chunk(&mut state, &json!({ "type": "message_stop" }));
    assert_eq!(
        done.raw_content
            .as_ref()
            .and_then(|r| r[0]["text"].as_str()),
        Some("Hello, world"),
        "{:?}",
        done.raw_content
    );
}

fn stream_status(error_type: &str) -> Option<u16> {
    rust_llm::protocols::streaming_error_status(ProtocolName::Anthropic)(
        &json!({ "type": "error", "error": { "type": error_type, "message": "Failed" } })
            .to_string(),
    )
}

// spec: protocols/anthropic/streaming_spec.rb:100 #parse_streaming_error > gives each documented error type its HTTP status
#[test]
fn each_documented_stream_error_type_gets_its_status() {
    for (kind, status) in [
        ("invalid_request_error", 400),
        ("authentication_error", 401),
        ("billing_error", 402),
        ("permission_error", 403),
        ("not_found_error", 404),
        ("request_too_large", 413),
        ("rate_limit_error", 429),
        ("api_error", 500),
        ("timeout_error", 504),
        ("overloaded_error", 529),
    ] {
        assert_eq!(stream_status(kind), Some(status), "{kind}");
    }
}

// spec: protocols/anthropic/streaming_spec.rb:114 #parse_streaming_error > falls back to a 500 for an error type it does not know
#[test]
fn an_unknown_stream_error_type_is_a_500() {
    assert_eq!(stream_status("brand_new_error"), Some(500));
}

async fn stream_failing_with(kind: &str, message: &str) -> Error {
    let events = format!(
        "event: message_start\ndata: {}\n\nevent: error\ndata: {}\n\n",
        json!({ "type": "message_start", "message": { "id": "msg_1", "type": "message", "role": "assistant",
                "content": [], "model": MODEL, "usage": { "input_tokens": 12, "output_tokens": 1 } } }),
        json!({ "type": "error", "error": { "type": kind, "message": message } })
    );
    let server = serve_templates(vec![sse(events)]).await;
    chat(&server).ask_stream("Hello", |_| {}).await.unwrap_err()
}

// spec: protocols/anthropic/streaming_spec.rb:155 stream errors > raises a rate limit reported in an error event
#[tokio::test]
async fn a_streamed_rate_limit_raises_a_rate_limit() {
    let message = "This request would exceed the rate limit for your organization of 50,000 input tokens per minute.";
    let err = stream_failing_with("rate_limit_error", message).await;
    assert!(
        matches!(&err, Error::RateLimit(m, _) if m == message),
        "{err:?}"
    );
}

// spec: protocols/anthropic/streaming_spec.rb:161 stream errors > raises a refused request reported in an error event as a bad request
#[tokio::test]
async fn a_streamed_refusal_raises_a_bad_request() {
    let message = "messages: at least one message is required";
    let err = stream_failing_with("invalid_request_error", message).await;
    assert!(
        matches!(&err, Error::BadRequest(m, _) if m == message),
        "{err:?}"
    );
}

// ---- protocols/anthropic_compaction_spec.rb -----------------------------------------------------

// spec: protocols/anthropic_compaction_spec.rb:93 usage on a compacted turn > sums cache write lifetimes across iterations
#[test]
fn cache_write_lifetimes_sum_across_iterations() {
    let iteration = |kind: &str, input: i64, output: i64, breakdown: Value| {
        json!({ "input_tokens": input, "output_tokens": output, "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0, "type": kind, "cache_creation": breakdown })
    };
    let usage = json!({
        "input_tokens": 187, "output_tokens": 85, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
        "iterations": [
            iteration("compaction", 99_207, 125, json!({ "ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 30 })),
            iteration("message", 187, 85, json!({ "ephemeral_5m_input_tokens": 5, "ephemeral_1h_input_tokens": 10 })),
        ]
    });
    let message = anthropic::parse_completion_body(
        &json!({ "content": [{ "type": "text", "text": "Hi." }], "usage": usage }),
        RawResponse::default(),
    )
    .unwrap();
    assert_eq!(
        message.tokens.cache_write_by_ttl,
        counts(json!({ "5m": 5, "1h": 40 }))
    );
}

// ---- chat_provider_tools_spec.rb: Anthropic MCP --------------------------------------------------

fn anthropic_mcp(options: Value) -> rust_llm::Result<Value> {
    Chat::with_config(anthropic_config(), Some(MODEL), Some("anthropic"), false)
        .unwrap()
        .with_provider_tools([rust_llm::ProviderTool::with_options("mcp", options)])
        .render()
}

// spec: chat_provider_tools_spec.rb:81 request rendering > translates allowed_tools on the Anthropic MCP alias into toolset configs
#[test]
fn anthropic_mcp_allowed_tools_become_toolset_configs() {
    let payload = anthropic_mcp(json!({
        "url": "https://mcp.example.com", "name": "example",
        "allowed_tools": ["search"], "require_approval": "never"
    }))
    .unwrap();
    assert!(
        payload["tools"].as_array().unwrap().contains(&json!({
            "type": "mcp_toolset", "mcp_server_name": "example", "default_config": { "enabled": false },
            "configs": { "search": { "enabled": true } }
        })),
        "{payload}"
    );
}

// spec: chat_provider_tools_spec.rb:93 request rendering > rejects MCP approval on Anthropic, which cannot pause for it
#[test]
fn anthropic_mcp_rejects_approval() {
    let err =
        anthropic_mcp(json!({ "url": "https://mcp.example.com", "require_approval": "always" }))
            .unwrap_err();
    assert!(
        matches!(&err, Error::Argument(m) if m.contains("require_approval: 'never'")),
        "{err:?}"
    );
}

// spec: chat_provider_tools_spec.rb:100 request rendering > rejects MCP tool filters Anthropic cannot express
#[test]
fn anthropic_mcp_rejects_filters_it_cannot_express() {
    let err = anthropic_mcp(json!({
        "url": "https://mcp.example.com", "allowed_tools": { "read_only": true }
    }))
    .unwrap_err();
    assert!(
        matches!(&err, Error::Argument(m) if m.contains("allowed_tools: [name]")),
        "{err:?}"
    );
}

// ---- models_spec.rb -----------------------------------------------------------------------------

// spec: models_spec.rb:15 .instance > loads the registry once when threads ask for it together
// (`RubyLLM.models` is `REGISTRY`, a `LazyLock`: every thread gets the same registry `Arc`.)
#[test]
fn threads_asking_together_share_one_registry() {
    let registries: Vec<_> = (0..8)
        .map(|_| std::thread::spawn(rust_llm::models))
        .collect::<Vec<_>>()
        .into_iter()
        .map(|h| h.join().unwrap())
        .collect();
    assert!(registries.windows(2).all(|w| Arc::ptr_eq(&w[0], &w[1])));
}

// ---- server_tool_call_spec.rb -------------------------------------------------------------------

// spec: server_tool_call_spec.rb:6 keeps search suggestions out of the hash it is stored as
// (`ServerToolCall#to_h` is private to `Message#to_h` in the port; read it from there.)
#[test]
fn search_suggestions_stay_out_of_the_stored_hash() {
    let call = rust_llm::message::ServerToolCall {
        kind: "google_search".into(),
        name: None,
        id: None,
        input: Some(json!({ "queries": ["ruby"] })),
        result: None,
        raw: json!({ "webSearchQueries": ["ruby"] }),
        search_suggestions: Some("<div>Ruby</div>".into()),
    };
    assert_eq!(call.search_suggestions.as_deref(), Some("<div>Ruby</div>"));
    let mut message = Message::assistant("Ruby");
    message.server_tool_calls = vec![call];
    let h = message.to_h();
    assert_eq!(
        h["server_tool_calls"],
        json!([{ "type": "google_search", "input": { "queries": ["ruby"] },
                 "raw": { "webSearchQueries": ["ruby"] } }])
    );
    let restored = Message::from_h(&h).unwrap();
    assert_eq!(restored.server_tool_calls[0].search_suggestions, None);
}
