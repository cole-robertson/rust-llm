//! Value-object specs ported from RubyLLM 2.0: `cost_spec.rb`, `message_spec.rb`,
//! `citation_spec.rb`, `progress_spec.rb`, `error_spec.rb`, `configuration_spec.rb`,
//! `model_spec.rb`, `models_spec.rb`, `models/lookup_spec.rb`, and `tool_spec.rb`. Examples
//! whose Ruby shape has no Rust counterpart (Hash coercion, `from_h`, Ruby class hierarchy,
//! Faraday adapters) are marked N/A in docs/PARITY.md instead of being faked here.
//! `// spec:` lines tie each test to its Ruby example.

use async_trait::async_trait;
use rust_llm::cost::{Component, Tier};
use rust_llm::message::indexmap_lite::IndexMap;
use rust_llm::model::{Modalities, ModelType, PricingCategory, PricingTier};
use rust_llm::{
    Citation, Config, Cost, Error, ErrorKind, FinishReason, Message, Model, Parameter, Progress,
    Role, Tokens, Tool, ToolCall, ToolError, ToolResult,
};
use serde_json::{Map, Value, json};

const EPS: f64 = 1e-10;

fn close(actual: Option<f64>, expected: f64) {
    let a = actual.unwrap_or_else(|| panic!("expected {expected}, got None"));
    assert!((a - expected).abs() < EPS, "expected {expected}, got {a}");
}

fn tier(input: Option<f64>, output: Option<f64>) -> PricingTier {
    PricingTier {
        input_per_million: input,
        output_per_million: output,
        ..Default::default()
    }
}

fn model_with(id: &str, provider: &str, text: PricingCategory) -> Model {
    let mut m = Model::default_for(id, provider);
    m.name = id.into();
    m.pricing.text_tokens = Some(text);
    m
}

fn standard(t: PricingTier) -> PricingCategory {
    PricingCategory {
        standard: Some(t),
        ..Default::default()
    }
}

/// cost_spec.rb's `priced-model`: $1 in, $2 out, $0.25 cache read, $1.25 cache write.
fn priced() -> Model {
    model_with(
        "priced-model",
        "openai",
        standard(PricingTier {
            input_per_million: Some(1.0),
            output_per_million: Some(2.0),
            cache_read_input_per_million: Some(0.25),
            cache_write_input_per_million: Some(1.25),
            ..Default::default()
        }),
    )
}

fn tokens(input: Option<i64>, output: Option<i64>) -> Tokens {
    Tokens {
        input,
        output,
        ..Default::default()
    }
}

fn cost(t: &Tokens, m: Option<&Model>) -> Cost {
    Cost::new(t, m, Tier::Standard)
}

// ---- cost_spec.rb -----------------------------------------------------------------------------

// spec: cost_spec.rb:25 calculates input, output, cache read, and cache write costs from normalized token buckets
#[test]
fn cost_prices_every_normalized_bucket() {
    let t = Tokens {
        input: Some(1_000),
        output: Some(2_000),
        cache_read: Some(300),
        cache_write: Some(100),
        ..Default::default()
    };
    let c = cost(&t, Some(&priced()));
    close(c.input, 0.001);
    close(c.output, 0.004);
    close(c.cache_read, 0.000075);
    close(c.cache_write, 0.000125);
    close(c.total(), 0.0052);
}

// spec: cost_spec.rb:36 trusts input tokens as the standard input bucket
#[test]
fn cost_trusts_input_tokens_as_the_standard_bucket() {
    let t = Tokens {
        input: Some(700),
        cache_read: Some(300),
        ..Default::default()
    };
    let c = cost(&t, Some(&priced()));
    close(c.input, 0.0007);
    close(c.cache_read, 0.000075);
    close(c.total(), 0.000775);
}

// spec: cost_spec.rb:45 calculates image costs from text and image input details
#[test]
fn cost_prices_images_from_text_and_image_input_details() {
    let mut m = model_with("image-model", "openai", standard(tier(Some(5.0), None)));
    m.pricing.images = Some(standard(tier(Some(10.0), Some(40.0))));
    let c = Cost::images(
        &tokens(Some(350), Some(50)),
        Some(&m),
        Some(&json!({ "text_tokens": 100, "image_tokens": 250 })),
    );
    close(c.input, 0.003);
    close(c.output, 0.002);
    close(c.total(), 0.005);
}

// spec: cost_spec.rb:80 does not price thinking tokens separately when output already includes them
#[test]
fn cost_does_not_price_thinking_inside_output_twice() {
    let t = Tokens {
        input: Some(50),
        output: Some(1306),
        thinking: Some(1087),
        ..Default::default()
    };
    let c = cost(&t, Some(&priced()));
    close(c.output, 0.002612);
    assert_eq!(c.thinking, None);
    close(c.total(), 0.002662);
}

// spec: cost_spec.rb:89 prices thinking tokens separately when the model has distinct reasoning pricing
#[test]
fn cost_prices_thinking_separately_with_distinct_reasoning_pricing() {
    let m = model_with(
        "reasoning-priced-model",
        "perplexity",
        standard(PricingTier {
            reasoning_output_per_million: Some(3.0),
            ..tier(Some(2.0), Some(8.0))
        }),
    );
    let t = Tokens {
        input: Some(33),
        output: Some(11_395),
        thinking: Some(193_947),
        ..Default::default()
    };
    let c = cost(&t, Some(&m));
    close(c.input, 0.000066);
    close(c.output, 0.09116);
    close(c.thinking, 0.581841);
    close(c.total(), 0.673067);
}

// spec: cost_spec.rb:113 does not double-count thinking tokens when reasoning pricing matches output pricing
#[test]
fn cost_does_not_double_count_thinking_at_the_output_price() {
    let m = model_with(
        "inclusive",
        "openrouter",
        standard(PricingTier {
            reasoning_output_per_million: Some(12.0),
            ..tier(None, Some(12.0))
        }),
    );
    let c = cost(
        &Tokens {
            output: Some(1_000),
            thinking: Some(800),
            ..Default::default()
        },
        Some(&m),
    );
    assert_eq!(c.output, Some(0.012));
    assert_eq!(c.thinking, None);
    assert_eq!(c.total(), Some(0.012));
}

fn long_context_model() -> Model {
    model_with(
        "gpt-5.6-sol",
        "openai",
        PricingCategory {
            standard: Some(PricingTier {
                cache_read_input_per_million: Some(0.5),
                ..tier(Some(5.0), Some(30.0))
            }),
            long_context: Some(PricingTier {
                cache_read_input_per_million: Some(1.0),
                ..tier(Some(10.0), Some(45.0))
            }),
            long_context_threshold: Some(272_000),
            ..Default::default()
        },
    )
}

// spec: cost_spec.rb:135 uses long-context rates when the prompt exceeds the model threshold
#[test]
fn cost_uses_long_context_rates_past_the_threshold() {
    let m = long_context_model();
    close(
        cost(&tokens(Some(100_000), Some(10_000)), Some(&m)).total(),
        0.8,
    );
    close(
        cost(&tokens(Some(500_000), Some(10_000)), Some(&m)).total(),
        5.45,
    );
}

// spec: cost_spec.rb:170 counts cache tokens toward the long-context prompt threshold
#[test]
fn cost_counts_cache_tokens_toward_the_long_context_threshold() {
    let t = Tokens {
        input: Some(100_000),
        output: Some(1_000),
        cache_read: Some(200_000),
        ..Default::default()
    };
    let c = cost(&t, Some(&long_context_model()));
    close(c.input, 1.0);
    close(c.cache_read, 0.2);
    close(c.output, 0.045);
}

// spec: cost_spec.rb:201 returns nil when pricing is missing for tokens that were used
#[test]
fn cost_total_is_unknown_when_used_tokens_have_no_price() {
    let m = model_with(
        "incomplete-model",
        "openai",
        standard(tier(Some(1.0), None)),
    );
    let c = cost(&tokens(Some(10), Some(5)), Some(&m));
    assert_eq!(c.input, Some(0.00001));
    assert_eq!(c.output, None);
    assert_eq!(c.total(), None);
}

// spec: cost_spec.rb:216 does not require pricing for token buckets that were not used
#[test]
fn cost_needs_no_price_for_unused_buckets() {
    let m = model_with(
        "input-only-model",
        "openai",
        standard(tier(Some(1.0), None)),
    );
    let c = cost(&tokens(Some(10), None), Some(&m));
    assert_eq!(c.output, None);
    assert_eq!(c.total(), Some(0.00001));
}

// spec: cost_spec.rb:230 returns nil when there is no token usage
#[test]
fn cost_total_is_unknown_without_usage() {
    assert_eq!(cost(&Tokens::default(), Some(&priced())).total(), None);
}

fn reported(input: Option<i64>, output: Option<i64>, amount: f64) -> Tokens {
    Tokens {
        reported_cost: Some(amount),
        ..tokens(input, output)
    }
}

// spec: cost_spec.rb:236 prefers the reported cost over the registry estimate
#[test]
fn cost_prefers_the_reported_cost() {
    let c = cost(&reported(Some(1_000), Some(2_000), 0.0042), Some(&priced()));
    close(c.input, 0.001);
    assert_eq!(c.total(), Some(0.0042));
}

// spec: cost_spec.rb:244 returns the reported cost when registry pricing is missing
#[test]
fn cost_returns_the_reported_cost_without_registry_pricing() {
    let c = cost(&reported(Some(10), Some(5), 0.0042), None);
    assert_eq!(c.input, None);
    assert_eq!(c.total(), Some(0.0042));
}

// spec: cost_spec.rb:252 reports usage even when the provider returned only a cost
#[test]
fn cost_reports_usage_from_a_cost_alone() {
    let c = cost(&reported(None, None, 0.0042), None);
    assert!(c.is_reported());
    assert_eq!(c.total(), Some(0.0042));
}

// spec: cost_spec.rb:260 estimates from the registry when no cost was reported
#[test]
fn cost_estimates_from_the_registry_without_a_reported_cost() {
    close(
        cost(&tokens(Some(1_000), Some(2_000)), Some(&priced())).total(),
        0.005,
    );
}

// spec: cost_spec.rb:267 sums reported costs across aggregated attempts
#[test]
fn cost_aggregate_sums_reported_costs() {
    let a = cost(&reported(Some(10), None, 0.001), None);
    let b = cost(&reported(Some(20), None, 0.002), None);
    close(Cost::aggregate([&a, &b], true).total(), 0.003);
}

// spec: cost_spec.rb:275 mixes reported and estimated totals in an aggregate
#[test]
fn cost_aggregate_mixes_reported_and_estimated_totals() {
    let a = cost(&reported(Some(10), None, 0.001), None);
    let b = cost(&tokens(Some(1_000), Some(2_000)), Some(&priced()));
    close(Cost::aggregate([&a, &b], true).total(), 0.006);
}

// spec: cost_spec.rb:292 sums costs while preserving nil for missing pricing
#[test]
fn cost_aggregate_keeps_missing_pricing_missing() {
    let a = cost(&tokens(Some(10), None), Some(&priced()));
    let b = cost(&tokens(None, Some(10)), None);
    let agg = Cost::aggregate([&a, &b], true);
    assert_eq!(agg.input, Some(0.00001));
    assert_eq!(agg.output, None);
    assert_eq!(agg.total(), None);
}

// spec: cost_spec.rb:302 ignores entries without token usage
#[test]
fn cost_aggregate_ignores_entries_without_usage() {
    let empty = cost(&Tokens::default(), Some(&priced()));
    let a = cost(&tokens(Some(10), None), Some(&priced()));
    assert_eq!(Cost::aggregate([&empty, &a], true).total(), Some(0.00001));
}

// spec: cost_spec.rb:312 reads component amounts and total from a stored breakdown
// (`Cost.from_h` is `Cost::from_recorded(amounts, total, tokens)`: the stored usage columns.)
#[test]
fn cost_from_recorded_reads_amounts_and_total() {
    let c = Cost::from_recorded(
        [Some(0.001), Some(0.004), None, None, None],
        Some(0.005),
        &Tokens::default(),
    );
    assert_eq!(
        (c.input, c.output, c.cache_read),
        (Some(0.001), Some(0.004), None)
    );
    assert_eq!(c.total(), Some(0.005));
}

// spec: cost_spec.rb:327 preserves a recorded total when component costs were not stored
// spec: cost_spec.rb:333 preserves a recorded total when token counts were not stored
// spec: cost_spec.rb:464 trusts a recorded total even when components are missing
#[test]
fn cost_from_recorded_trusts_a_recorded_total() {
    let c = Cost::from_recorded([None; 5], Some(0.005), &Tokens::default());
    assert_eq!(c.total(), Some(0.005));
    assert!(c.is_reported());
}

// spec: cost_spec.rb:339 keeps missing historical pricing missing when token usage is known
#[test]
fn cost_from_recorded_keeps_unpriced_usage_missing() {
    let c = Cost::from_recorded([None; 5], None, &tokens(Some(10), None));
    assert_eq!(c.input, None);
    assert_eq!(c.total(), None);
    assert!(c.missing().contains(&Component::Input));
}

// spec: cost_spec.rb:356 returns a nil total when the stored breakdown recorded no total
// (the stored breakdown has no token counts, so nothing proves the input amount is the whole bill)
#[test]
fn cost_from_recorded_without_a_total_and_without_tokens_has_no_total() {
    let c = Cost::from_recorded(
        [Some(0.001), None, None, None, None],
        None,
        &Tokens::default(),
    );
    assert_eq!(c.input, Some(0.001));
    assert_eq!(c.total(), None);
}

// spec: cost_spec.rb:363 aggregates several stored costs
#[test]
fn cost_aggregate_of_recorded_costs() {
    let a = Cost::from_recorded(
        [Some(0.001), Some(0.004), None, None, None],
        Some(0.005),
        &Tokens::default(),
    );
    let b = Cost::from_recorded(
        [Some(0.0005), Some(0.002), None, None, None],
        Some(0.0025),
        &Tokens::default(),
    );
    let agg = Cost::aggregate([&a, &b], true);
    close(agg.input, 0.0015);
    close(agg.output, 0.006);
    close(agg.total(), 0.0075);
}

// spec: cost_spec.rb:373 aggregates a stored cost mixed with a live cost
#[test]
fn cost_aggregate_of_a_recorded_and_a_live_cost() {
    let stored = Cost::from_recorded(
        [Some(0.001), Some(0.004), None, None, None],
        Some(0.005),
        &Tokens::default(),
    );
    let live = cost(&tokens(Some(1_000), None), Some(&priced()));
    let agg = Cost::aggregate([&stored, &live], true);
    close(agg.input, 0.002);
    assert_eq!(agg.output, Some(0.004));
    close(agg.total(), 0.006);
}

// spec: cost_spec.rb:385 reports no total when one attempt is still unpriced
#[test]
fn cost_incomplete_aggregate_has_no_total() {
    let a = cost(&tokens(Some(10), None), Some(&priced()));
    let agg = Cost::aggregate([&a], false);
    close(agg.input, 0.00001);
    assert_eq!(agg.total(), None);
}

// spec: cost_spec.rb:395 prices against a model looked up by id
#[test]
fn cost_prices_against_a_registry_model() {
    let m = rust_llm::models().find("gpt-4.1-nano", None).unwrap();
    assert!(
        cost(&tokens(Some(1_000_000), None), Some(&m))
            .input
            .unwrap()
            > 0.0
    );
}

// spec: cost_spec.rb:429 prices a named category
#[test]
fn cost_prices_the_audio_category() {
    let mut m = Model::default_for("audio-model", "openai");
    m.pricing.audio_tokens = Some(standard(tier(Some(4.0), Some(8.0))));
    let c = Cost::audio(&tokens(Some(1_000_000), Some(1_000_000)), Some(&m));
    assert_eq!((c.input, c.output), (Some(4.0), Some(8.0)));
}

// spec: cost_spec.rb:457 reports nothing when the stored breakdown is empty
#[test]
fn cost_from_an_empty_record_reports_nothing() {
    let c = Cost::from_recorded([None; 5], None, &Tokens::default());
    assert_eq!(c.total(), None);
    assert!(!c.is_reported());
}

// spec: cost_spec.rb:471 flags components that had tokens but no recorded cost
#[test]
fn cost_from_recorded_flags_components_with_tokens_but_no_cost() {
    let c = Cost::from_recorded(
        [Some(0.001), None, None, None, None],
        None,
        &tokens(Some(10), Some(5)),
    );
    assert!(c.missing().contains(&Component::Output));
    assert_eq!(c.total(), None);
}

// ---- message_spec.rb --------------------------------------------------------------------------

fn call(id: &str, name: &str) -> ToolCall {
    ToolCall::new(id, name, Map::new())
}

fn calling(calls: &[(&str, &str)]) -> Message {
    let mut m = Message::new(Role::Assistant, None::<String>);
    m.tool_calls = Some(
        calls
            .iter()
            .map(|(id, name)| (id.to_string(), call(id, name)))
            .collect::<IndexMap<_>>(),
    );
    m
}

// spec: message_spec.rb:30 keeps nil content for messages without tool calls
#[test]
fn message_keeps_no_content_without_tool_calls() {
    assert_eq!(Message::new(Role::Assistant, None::<String>).content, None);
}

// spec: message_spec.rb:44 parses JSON content
#[test]
fn message_parsed_reads_json() {
    assert_eq!(
        Message::assistant(r#"{"name":"Alice","age":30}"#)
            .parsed()
            .unwrap(),
        Some(json!({ "name": "Alice", "age": 30 }))
    );
}

// spec: message_spec.rb:50 returns nil for nil content
#[test]
fn message_parsed_is_none_without_content() {
    assert_eq!(
        Message::new(Role::Assistant, None::<String>)
            .parsed()
            .unwrap(),
        None
    );
}

// spec: message_spec.rb:56 raises for non-JSON content
#[test]
fn message_parsed_fails_for_plain_text() {
    assert!(matches!(
        Message::assistant("plain text").parsed(),
        Err(Error::Json(_))
    ));
}

// spec: message_spec.rb:62 returns nil for a tool-call turn without text
#[test]
fn message_parsed_is_none_for_a_tool_call_turn() {
    assert_eq!(calling(&[("call_1", "weather")]).parsed().unwrap(), None);
}

// spec: message_spec.rb:110 defaults to an empty array
#[test]
fn message_attachments_default_to_empty() {
    assert!(Message::user("hello").attachments.is_empty());
}

// spec: message_spec.rb:185 calculates cost from the supplied model
#[test]
fn message_cost_prices_against_a_supplied_model() {
    let mut m = Message::assistant("Hello");
    m.tokens = tokens(Some(1_000), Some(2_000));
    let c = m.cost(Some(&priced()));
    assert_eq!(c.total(), Some(0.005));
    assert_eq!((c.input, c.output), (Some(0.001), Some(0.004)));
}

// spec: message_spec.rb:208 returns nil when the message model cannot be found
#[test]
fn message_cost_is_unknown_for_an_unknown_model() {
    let mut m = Message::assistant("Hello");
    m.tokens = tokens(Some(1_000), None);
    m.model = Some("missing-model".into());
    assert_eq!(m.cost(None).total(), None);
}

// spec: message_spec.rb:221 always returns token and cost value objects
#[test]
fn message_tokens_and_cost_are_empty_values_by_default() {
    let m = Message::user("Hello");
    assert!(m.tokens().is_empty());
    assert_eq!(m.cost(None).total(), None);
}

// spec: message_spec.rb:230 exposes every bucket through the token value only
#[test]
fn message_exposes_every_bucket_through_tokens() {
    let mut m = Message::assistant("Hello");
    m.tokens = Tokens {
        input: Some(10),
        output: Some(4),
        cache_read: Some(42),
        cache_write: Some(7),
        thinking: Some(2),
        ..Default::default()
    };
    let t = m.tokens();
    assert_eq!(
        (t.input, t.output, t.cache_read, t.cache_write, t.thinking),
        (Some(10), Some(4), Some(42), Some(7), Some(2))
    );
}

// spec: message_spec.rb:253 does not substitute another provider when the recorded model is missing
#[test]
fn message_model_info_does_not_substitute_another_provider() {
    let mut m = Message::assistant("ok");
    m.model = Some("gpt-5-nano".into());
    m.usage_entries = vec![rust_llm::UsageEntry {
        id: rust_llm::UsageEntry::next_id(),
        operation: rust_llm::message::Operation::Chat,
        provider: "custom".into(),
        model: "gpt-5-nano".into(),
        status: rust_llm::UsageStatus::Succeeded,
        tokens: Tokens::default(),
        cost: Cost::default(),
    }];
    assert!(m.model_info().is_none());
}

// spec: message_spec.rb:266 includes finish_reason when present
#[test]
fn message_to_h_includes_the_finish_reason() {
    let mut m = Message::assistant("Hello");
    m.finish_reason = Some(FinishReason::from_symbol("length"));
    assert_eq!(m.to_h()["finish_reason"], json!("length"));
}

// spec: message_spec.rb:287 returns true for #{predicate} on the normalized #{finish_reason} reason
#[test]
fn message_finish_reason_predicates() {
    let with = |r: &str| {
        let mut m = Message::assistant("Hello");
        m.finish_reason = Some(FinishReason::from_symbol(r));
        m
    };
    assert!(with("stop").is_stopped());
    assert!(with("max_tokens").is_max_tokens());
    assert!(with("tool_calls").is_tool_call_stop());
    assert!(with("content_filter").is_content_filtered());
}

// spec: message_spec.rb:294 leaves provider spellings to the protocols
#[test]
fn message_keeps_provider_finish_reasons_verbatim() {
    let mut m = Message::assistant("Hello");
    m.finish_reason = Some(FinishReason::from_symbol("end_turn"));
    assert!(!m.is_stopped());
    assert_eq!(m.finish_reason.unwrap().as_str(), "end_turn");
}

// spec: message_spec.rb:301 returns false when finish_reason is nil or unknown
#[test]
fn message_predicates_are_false_for_unknown_reasons() {
    for reason in [
        None,
        Some(FinishReason::from_symbol("weird_provider_value")),
    ] {
        let mut m = Message::assistant("Hello");
        m.finish_reason = reason;
        assert!(
            !m.is_stopped()
                && !m.is_max_tokens()
                && !m.is_tool_call_stop()
                && !m.is_content_filtered()
        );
    }
}

// spec: message_spec.rb:311 is inherited by streaming chunks (`Chunk` is `Message` here)
#[test]
fn chunks_have_the_finish_reason_predicates() {
    let mut chunk: rust_llm::Chunk = Message::new(Role::Assistant, None::<String>);
    chunk.finish_reason = Some(FinishReason::MaxTokens);
    assert!(chunk.is_max_tokens());
}

// spec: message_spec.rb:317 reports a tool-call stop even when the provider says the turn completed
#[test]
fn message_tool_call_stop_even_when_the_provider_says_stop() {
    let mut m = calling(&[("call_1", "weather")]);
    m.finish_reason = Some(FinishReason::Stop);
    assert!(m.is_tool_call_stop());
    assert!(!m.is_stopped());
}

// ---- citation_spec.rb -------------------------------------------------------------------------

fn full_citation() -> Citation {
    Citation {
        url: Some("https://example.com".into()),
        title: Some("Example".into()),
        cited_text: Some("The grass is green.".into()),
        text: Some("the grass is green".into()),
        start_index: Some(28),
        end_index: Some(46),
        source_id: Some("file_facts".into()),
        source_index: Some(0),
        start_page: Some(5),
        end_page: Some(5),
    }
}

// spec: citation_spec.rb:21 round-trips through to_h and from_h
#[test]
fn citation_round_trips_through_json() {
    let c = full_citation();
    let h = serde_json::to_value(&c).unwrap();
    assert_eq!(h.as_object().unwrap().len(), 10);
    assert_eq!(serde_json::from_value::<Citation>(h).unwrap(), c);
}

// spec: citation_spec.rb:28 builds from string-keyed hashes
#[test]
fn citation_builds_from_a_string_keyed_hash() {
    let c: Citation =
        serde_json::from_value(json!({ "url": "https://example.com", "start_index": 3 })).unwrap();
    assert_eq!(
        (c.url.as_deref(), c.start_index),
        (Some("https://example.com"), Some(3))
    );
}

// spec: citation_spec.rb:35 preserves file identities through JSON persistence
#[test]
fn citation_keeps_file_identities_through_json() {
    let c = Citation {
        source_id: Some("file_facts".into()),
        title: Some("facts.pdf".into()),
        ..Default::default()
    };
    let restored: Citation = serde_json::from_str(&serde_json::to_string(&c).unwrap()).unwrap();
    assert_eq!(
        (
            restored.source_id.as_deref(),
            restored.title.as_deref(),
            restored.url.as_deref()
        ),
        (Some("file_facts"), Some("facts.pdf"), None)
    );
    assert_ne!(
        restored,
        Citation {
            source_id: Some("file_other".into()),
            ..c
        }
    );
}

// spec: citation_spec.rb:43 omits missing fields from to_h
#[test]
fn citation_omits_missing_fields() {
    let c = Citation {
        url: Some("https://example.com".into()),
        ..Default::default()
    };
    assert_eq!(
        serde_json::to_value(&c).unwrap(),
        json!({ "url": "https://example.com" })
    );
}

// spec: citation_spec.rb:49 compares by value
#[test]
fn citation_compares_by_value() {
    assert_eq!(full_citation(), full_citation().clone());
    assert_ne!(
        full_citation(),
        Citation {
            url: Some("https://other.com".into()),
            ..full_citation()
        }
    );
}

// ---- progress_spec.rb -------------------------------------------------------------------------

fn progress(value: Option<f64>, total: Option<f64>, message: Option<&str>) -> Progress {
    Progress {
        value,
        total,
        message: message.map(str::to_string),
    }
}

// spec: progress_spec.rb:6 reads the share of work done
#[test]
fn progress_reads_the_share_of_work_done() {
    assert_eq!(
        progress(Some(3.0), Some(12.0), Some("Reading page 3 of 12")).fraction(),
        Some(0.25)
    );
}

// spec: progress_spec.rb:10 has no share without a value and a total
#[test]
fn progress_has_no_share_without_value_and_total() {
    assert_eq!(progress(None, None, Some("Downloading")).fraction(), None);
    assert_eq!(progress(Some(120.0), None, None).fraction(), None);
    assert_eq!(progress(None, Some(12.0), None).fraction(), None);
}

// ---- error_spec.rb ----------------------------------------------------------------------------

// spec: error_spec.rb:8 uses the message and leaves response nil
// spec: error_spec.rb:53 accepts a plain message
#[test]
fn errors_carry_their_message_and_no_response() {
    let e = Error::Api("something went wrong".into(), None);
    assert_eq!(e.to_string(), "something went wrong");
    assert!(e.response().is_none());
    let e = Error::BadRequest("bad request".into(), None);
    assert_eq!(
        (e.to_string().as_str(), e.response().is_none()),
        ("bad request", true)
    );
}

// spec: error_spec.rb:23 stores the response
// spec: error_spec.rb:28 uses the provided message
#[test]
fn errors_keep_the_response_they_came_from() {
    let response = rust_llm::error::ErrorResponse {
        status: 500,
        body: r#"{"error":"server error"}"#.into(),
    };
    let e = Error::Server("server error".into(), Some(response));
    assert_eq!(e.to_string(), "server error");
    assert_eq!(
        e.response().map(|r| (r.status, r.body.as_str())),
        Some((500, r#"{"error":"server error"}"#))
    );
}

// spec: error_spec.rb:59 keeps local setup and programming errors outside RubyLLM::Error
// Rust has one `Error` enum; the equivalent fact is that local errors are never provider
// errors: they have no response and are not retried or fallen back on.
#[test]
fn local_errors_are_not_provider_errors() {
    for e in [
        Error::Configuration("x".into()),
        Error::InvalidToolChoice("x".into()),
        Error::ModelNotFound("x".into()),
        Error::Argument("x".into()),
    ] {
        assert!(e.response().is_none());
        assert_eq!(e.kind(), ErrorKind::Other);
        assert!(!rust_llm::error::DEFAULT_FALLBACK_ERRORS.contains(&e.kind()));
    }
}

// spec: error_spec.rb:74 stores the finish reason when available
#[tokio::test]
async fn tool_call_parse_errors_keep_the_finish_reason() {
    // A truncated tool call from the wire: arguments cut off by the length limit.
    let server = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::any())
        .respond_with(wiremock::ResponseTemplate::new(200).set_body_json(json!({
            "id": "c1", "model": "deepseek-v4-flash", "object": "chat.completion",
            "choices": [{ "index": 0, "finish_reason": "length", "message": { "role": "assistant", "content": null,
                "tool_calls": [{ "id": "call_1", "type": "function", "function": { "name": "echo", "arguments": "{\"text\": \"hel" } }] } }],
            "usage": { "prompt_tokens": 1, "completion_tokens": 1 }
        })))
        .mount(&server)
        .await;
    let mut config = Config::default();
    config.set("deepseek_api_base", server.uri());
    config.set("deepseek_api_key", "test");
    config.max_retries = 0;
    let mut chat = rust_llm::Chat::with_config(
        std::sync::Arc::new(config),
        Some("deepseek-v4-flash"),
        Some("deepseek"),
        false,
    )
    .unwrap();
    match chat.ask("hi").await.unwrap_err() {
        Error::ToolCallParse {
            message,
            finish_reason,
            ..
        } => {
            // Ruby raises with the normalized reason: `length` is `:max_tokens` by then.
            assert_eq!(finish_reason.as_deref(), Some("max_tokens"));
            assert!(message.contains("finish_reason: max_tokens"), "{message}");
        }
        other => panic!("expected ToolCallParse, got {other:?}"),
    }
}

// spec: error_spec.rb:90 uses a simple standard message with the unsupported type and guidance
// spec: error_spec.rb:128 names the type when there is one
#[test]
fn unsupported_attachment_errors_name_the_type_and_guide() {
    let m = rust_llm::Chat::with_config(
        std::sync::Arc::new({
            let mut c = Config::default();
            c.set("anthropic_api_key", "test");
            c
        }),
        Some("claude-haiku-4-5"),
        Some("anthropic"),
        false,
    )
    .unwrap();
    let mut m = m;
    m.ask_later_with(
        "read this",
        vec![rust_llm::Attachment::from_bytes(
            b"PK".to_vec(),
            "a.docx",
            Some("application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        )],
    )
    .unwrap();
    let err = m.render().unwrap_err();
    assert!(matches!(err, Error::UnsupportedAttachment(_)));
    assert_eq!(
        err.to_string(),
        "Unsupported attachment type: application/vnd.openxmlformats-officedocument.wordprocessingml.document. Consider using a model that supports this attachment type."
    );
}

// spec: error_spec.rb:120 explains #{error_class} when the provider says nothing
#[tokio::test]
async fn every_error_class_has_its_default_message() {
    for (status, kind, message) in [
        (
            400,
            ErrorKind::BadRequest,
            "Invalid request - please check your input",
        ),
        (
            403,
            ErrorKind::Forbidden,
            "Forbidden - you do not have permission to access this resource",
        ),
        (
            529,
            ErrorKind::Overloaded,
            "Service overloaded - please try again later",
        ),
        (
            402,
            ErrorKind::PaymentRequired,
            "Payment required - please top up your account",
        ),
        (
            429,
            ErrorKind::RateLimit,
            "Rate limit exceeded - please wait a moment",
        ),
        (
            500,
            ErrorKind::Server,
            "API server error - please try again",
        ),
        (
            503,
            ErrorKind::ServiceUnavailable,
            "API server unavailable - please try again later",
        ),
        (
            401,
            ErrorKind::Unauthorized,
            "Invalid API key - check your credentials",
        ),
    ] {
        // An empty error body: the provider "says nothing".
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::any())
            .respond_with(wiremock::ResponseTemplate::new(status))
            .mount(&server)
            .await;
        let mut config = Config::default();
        config.set("anthropic_api_key", "test");
        config.set("anthropic_api_base", server.uri());
        config.max_retries = 0;
        let mut chat = rust_llm::Chat::with_config(
            std::sync::Arc::new(config),
            Some("claude-haiku-4-5"),
            Some("anthropic"),
            false,
        )
        .unwrap();
        let e = chat.ask("hi").await.unwrap_err();
        assert_eq!(
            (e.kind(), e.to_string()),
            (kind, message.to_string()),
            "{status}"
        );
    }
}

/// `ErrorMiddleware.parse_error` through a real request: `status` with `body`, no retries.
async fn error_for(status: u16, body: &str) -> Error {
    let server = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::any())
        .respond_with(wiremock::ResponseTemplate::new(status).set_body_string(body))
        .mount(&server)
        .await;
    let mut config = Config::default();
    config.set("anthropic_api_key", "test");
    config.set("anthropic_api_base", server.uri());
    config.max_retries = 0;
    let mut chat = rust_llm::Chat::with_config(
        std::sync::Arc::new(config),
        Some("claude-haiku-4-5"),
        Some("anthropic"),
        false,
    )
    .unwrap();
    chat.ask("hi").await.unwrap_err()
}

// spec: transport/error_middleware_spec.rb:147 maps 502 to ServiceUnavailableError
// spec: transport/error_middleware_spec.rb:163 maps 504 to ServiceUnavailableError
#[tokio::test]
async fn gateway_errors_are_service_unavailable() {
    for status in [502, 504] {
        assert_eq!(
            error_for(status, r#"{"error":{"message":"down"}}"#)
                .await
                .kind(),
            ErrorKind::ServiceUnavailable,
            "{status}"
        );
    }
}

// spec: transport/error_middleware_spec.rb:285 raises the base error for a status it does not map
#[tokio::test]
async fn an_unmapped_status_is_the_base_error_with_the_provider_message() {
    let e = error_for(418, r#"{"error":{"message":"teapot"}}"#).await;
    assert_eq!(
        (e.kind(), e.to_string().as_str()),
        (ErrorKind::Api, "teapot")
    );
}

// spec: error_spec.rb:37 falls back to the response body for the message
#[tokio::test]
async fn an_unparseable_error_body_becomes_the_message() {
    let e = error_for(418, "raw body").await;
    assert_eq!(e.to_string(), "raw body");
    assert_eq!(e.response().map(|r| r.body.as_str()), Some("raw body"));
}

// ---- configuration_spec.rb --------------------------------------------------------------------

// spec: configuration_spec.rb:9 applies core default values
#[test]
fn config_defaults_match_rubyllm() {
    let c = Config::default();
    assert_eq!(c.request_timeout, std::time::Duration::from_secs(300));
    assert_eq!(c.max_retries, 3);
    assert_eq!(c.retry_interval, 0.1);
    assert_eq!(c.retry_backoff_factor, 2.0);
    assert_eq!(c.retry_max_interval, 30.0);
    assert_eq!(c.retry_interval_randomness, 0.5);
    assert!(!c.tool_concurrency);
    assert_eq!(c.default_judgment_model, "jev-latest");
}

// spec: configuration_spec.rb:36 normalizes blank strings to nil
// spec: configuration_spec.rb:44 preserves non-blank strings
#[test]
fn config_treats_blank_strings_as_unset() {
    let mut c = Config::default();
    c.set("openai_api_base", "");
    c.set("anthropic_api_key", " \t\n");
    assert_eq!(c.get("openai_api_base"), None);
    assert_eq!(c.get("anthropic_api_key"), None);
    c.set(
        "openai_api_base",
        "https://openai-compatible.example.com/v1",
    );
    assert_eq!(
        c.get("openai_api_base"),
        Some("https://openai-compatible.example.com/v1")
    );
}

// ---- model_spec.rb ----------------------------------------------------------------------------

fn gpt5() -> Model {
    serde_json::from_value(json!({
        "id": "gpt-5", "name": "GPT-5", "provider": "openai", "family": "gpt",
        "created_at": "2026-02-20 00:00:00 UTC", "context_window": 400_000, "max_output_tokens": 128_000,
        "knowledge_cutoff": "2025-10-01",
        "modalities": { "input": ["text", "image"], "output": ["text"] },
        "capabilities": ["function_calling", "streaming", "vision", "structured_output"],
        "pricing": { "text_tokens": { "standard": { "input_per_million": 2.5, "output_per_million": 10.0 } } },
        "metadata": { "description": "A test model", "reasoning_options": [
            { "type": "effort", "values": ["low", "medium", "high"] }, { "type": "budget_tokens", "min": 1024 }
        ] }
    }))
    .unwrap()
}

// spec: model_spec.rb:32 assigns basic attributes
// spec: model_spec.rb:54 builds modalities
#[test]
fn model_reads_registry_attributes() {
    let m = gpt5();
    assert_eq!(
        (
            m.id.as_str(),
            m.name.as_str(),
            m.provider.as_str(),
            m.family.as_deref()
        ),
        ("gpt-5", "GPT-5", "openai", Some("gpt"))
    );
    assert_eq!(
        (m.context_window, m.max_output_tokens),
        (Some(400_000), Some(128_000))
    );
    assert_eq!(
        m.modalities,
        Modalities {
            input: vec!["text".into(), "image".into()],
            output: vec!["text".into()]
        }
    );
}

// spec: model_spec.rb:64 defaults missing optional fields
#[test]
fn model_defaults_missing_optional_fields() {
    let m: Model =
        serde_json::from_value(json!({ "id": "test", "name": "Test", "provider": "openai" }))
            .unwrap();
    assert!(
        m.capabilities.is_empty()
            && m.metadata.is_empty()
            && m.reasoning_options().is_empty()
            && m.modalities.input.is_empty()
    );
}

// spec: model_spec.rb:77 creates a model with assumed capabilities
#[test]
fn model_default_assumes_capabilities() {
    let m = Model::default_for("my-custom-model", "openai");
    assert_eq!(
        (m.id.as_str(), m.provider.as_str()),
        ("my-custom-model", "openai")
    );
    assert!(m.supports("function_calling") && m.supports("streaming"));
    assert!(m.metadata.contains_key("warning"));
}

// spec: model_spec.rb:88 returns true for included capabilities, as symbol or string
// spec: model_spec.rb:94 returns false for capabilities absent from the registry data
#[test]
fn model_supports_reads_capabilities() {
    let m = gpt5();
    assert!(m.supports("function_calling") && m.supports("streaming") && m.supports("vision"));
    assert!(!m.supports("batch"));
}

// spec: model_spec.rb:103 returns false for a model the provider still lists
// spec: model_spec.rb:108 returns true once the registry reports the time the provider stopped listing it
#[test]
fn model_unlisted_follows_unlisted_at() {
    assert!(!gpt5().is_unlisted());
    let gone = Model {
        unlisted_at: Some("2026-02-20 00:00:00 +0700".into()),
        ..gpt5()
    };
    assert!(gone.is_unlisted());
}

// spec: model_spec.rb:118 normalizes metadata reasoning options
// spec: model_spec.rb:186 returns option values by type
#[test]
fn model_reads_reasoning_options() {
    let m = gpt5();
    assert_eq!(m.reasoning_options().len(), 2);
    assert_eq!(
        m.reasoning_option_values("effort"),
        ["low", "medium", "high"]
    );
    assert!(m.reasoning_option_values("budget_tokens").is_empty());
}

// spec: model_spec.rb:193 returns chat for text output models (and the rest of `#type`)
#[test]
fn model_type_follows_output_modalities() {
    let typed = |out: &[&str]| {
        Model {
            modalities: Modalities {
                input: vec!["text".into()],
                output: out.iter().map(|s| s.to_string()).collect(),
            },
            ..gpt5()
        }
        .model_type()
    };
    assert_eq!(gpt5().model_type(), ModelType::Chat);
    assert_eq!(typed(&["embeddings"]), ModelType::Embedding);
    assert_eq!(typed(&["image"]), ModelType::Image);
    assert_eq!(typed(&["text", "image"]), ModelType::Image);
    assert_eq!(typed(&["text", "audio"]), ModelType::Audio);
    assert_eq!(typed(&["text", "embeddings"]), ModelType::Embedding);
    assert_eq!(typed(&["text", "moderation"]), ModelType::Moderation);
    assert_eq!(typed(&["video"]), ModelType::Video);
    assert_eq!(typed(&["rerank"]), ModelType::Rerank);
}

// spec: model_spec.rb:239 returns the provider and model name
#[test]
fn model_label_names_provider_and_model() {
    assert_eq!(gpt5().label(), "OpenAI - GPT-5");
}

// spec: model_spec.rb:274 builds a Cost for the supplied tokens
#[test]
fn model_cost_for_prices_tokens() {
    assert_eq!(
        gpt5().cost_for(&tokens(Some(1_000), Some(2_000))).total(),
        Some(0.0225)
    );
}

// spec: model_spec.rb:292 builds a Cost from batch pricing when requested
#[test]
fn model_batch_pricing() {
    let mut m = gpt5();
    m.pricing.text_tokens.as_mut().unwrap().batch = Some(tier(Some(1.25), Some(5.0)));
    assert_eq!(
        Cost::new(&tokens(Some(1_000), Some(2_000)), Some(&m), Tier::Batch).total(),
        Some(0.01125)
    );
}

// ---- models_spec.rb / models/lookup_spec.rb ---------------------------------------------------

// spec: models_spec.rb:15 filters models by provider
#[test]
fn models_filter_by_provider() {
    let registry = rust_llm::models();
    let openai = registry.by_provider("openai");
    assert!(!openai.is_empty());
    assert!(openai.iter().all(|m| m.provider == "openai"));
}

// spec: models_spec.rb:48 leaves unlisted models out of every listing method but still finds them
#[test]
fn models_leave_unlisted_models_out_of_listings_but_find_them() {
    let gone = Model {
        id: "gone-model".into(),
        unlisted_at: Some("2026-01-01".into()),
        ..Model::default_for("gone-model", "openai")
    };
    let listed = Model::default_for("kept-model", "openai");
    let registry = rust_llm::models::Models::new(vec![gone, listed]);
    let ids: Vec<&str> = registry.all().iter().map(|m| m.id.as_str()).collect();
    assert_eq!(ids, ["kept-model"]);
    assert!(registry.chat_models().iter().all(|m| m.id != "gone-model"));
    assert!(
        registry
            .by_provider("openai")
            .iter()
            .all(|m| m.id != "gone-model")
    );
    assert_eq!(registry.find("gone-model", None).unwrap().id, "gone-model");
}

// spec: models_spec.rb:64 prefers a listed model over an unlisted one when no provider is given
#[test]
fn models_prefer_a_listed_model_over_an_unlisted_one() {
    let gone = Model {
        unlisted_at: Some("2026-01-01".into()),
        ..Model::default_for("same-id", "openai")
    };
    let listed = Model::default_for("same-id", "openrouter");
    let registry = rust_llm::models::Models::new(vec![gone, listed]);
    assert_eq!(
        registry.find("same-id", None).unwrap().provider,
        "openrouter"
    );
}

// spec: models_spec.rb:89 finds models by ID
// spec: models_spec.rb:104 raises ModelNotFoundError for unknown models
#[test]
fn models_find_by_id_and_raise_for_unknown_ids() {
    let m = rust_llm::models()
        .find("gpt-5-nano", Some("openai"))
        .unwrap();
    assert_eq!(
        (m.id.as_str(), m.provider.as_str()),
        ("gpt-5-nano", "openai")
    );
    assert!(matches!(
        rust_llm::models().find("no-such-model-12345", None),
        Err(Error::ModelNotFound(_))
    ));
}

// spec: models_spec.rb:141 prefers the first-party provider when an aggregator serves the same name
#[test]
fn models_prefer_the_first_party_provider() {
    assert_eq!(
        rust_llm::models()
            .find("claude-haiku-4-5", None)
            .unwrap()
            .provider,
        "anthropic"
    );
}

// spec: models_spec.rb:453 filters to models that are embedding-capable
// spec: models_spec.rb:468 excludes models with non-text output modalities
#[test]
fn models_split_chat_and_embedding_models() {
    let registry = rust_llm::models();
    assert!(!registry.embedding_models().is_empty());
    assert!(
        registry
            .embedding_models()
            .iter()
            .all(|m| m.model_type() == ModelType::Embedding)
    );
    assert!(
        registry
            .chat_models()
            .iter()
            .all(|m| m.model_type() == ModelType::Chat)
    );
}

// spec: models/lookup_spec.rb:28 keeps exact matches first when provider preferences tie
// spec: models/lookup_spec.rb:59 preserves catalog order for duplicate ids from the same provider
#[test]
fn lookup_keeps_catalog_order_for_same_provider_duplicates() {
    let mut first = Model::default_for("dup", "openai");
    first.name = "first".into();
    let mut second = Model::default_for("dup", "openai");
    second.name = "second".into();
    let registry = rust_llm::models::Models::new(vec![first, second]);
    assert_eq!(registry.find("dup", None).unwrap().name, "first");
    assert_eq!(registry.find("dup", Some("openai")).unwrap().name, "first");
}

// ---- tool_spec.rb -----------------------------------------------------------------------------

// spec: tool_spec.rb:64 converts class name to snake_case and removes _tool suffix
// spec: tool_spec.rb:92 handles class names without Tool suffix
// spec: tool_spec.rb:97 strips :: for class in module namespace (Rust paths drop the module)
#[test]
fn tool_names_derive_from_the_type_name() {
    assert_eq!(rust_llm::tool::tool_name_from_type("SampleTool"), "sample");
    assert_eq!(
        rust_llm::tool::tool_name_from_type("AnotherSample"),
        "another_sample"
    );
    assert_eq!(
        rust_llm::tool::tool_name_from_type("app::tools::SampleTool"),
        "sample"
    );
}

// spec: tool_spec.rb:80 normalizes class name Unicode characters to ASCII
// spec: tool_spec.rb:85 handles class names with unsupported characters
#[test]
fn tool_names_normalize_unicode_to_ascii() {
    assert_eq!(rust_llm::tool::tool_name_from_type("SàmpleTòol"), "sample");
    assert_eq!(rust_llm::tool::tool_name_from_type("SampleΨTool"), "sample");
}

/// `SignatureTool`: `execute(questions:)`.
struct Signature;

#[async_trait]
impl Tool for Signature {
    fn description(&self) -> String {
        "Signature".into()
    }
    fn parameters(&self) -> Vec<Parameter> {
        vec![Parameter::new("questions").kind("array")]
    }
    async fn execute(
        &self,
        args: Map<String, Value>,
        _call: &ToolCall,
    ) -> Result<ToolResult, ToolError> {
        Ok(args["questions"].clone().into())
    }
}

/// `NoArgumentTool`: `execute` takes no keywords.
struct NoArgument;

#[async_trait]
impl Tool for NoArgument {
    fn description(&self) -> String {
        "No arguments".into()
    }
    async fn execute(
        &self,
        _args: Map<String, Value>,
        _call: &ToolCall,
    ) -> Result<ToolResult, ToolError> {
        Ok("ok".into())
    }
}

/// Runs `tool` through a chat's tool loop with `arguments`, returning the tool result's text.
async fn call_tool(tool: impl Tool + 'static, name: &str, arguments: Value) -> String {
    let server = wiremock::MockServer::start().await;
    let mut config = Config::default();
    config.set("anthropic_api_key", "test");
    config.set("anthropic_api_base", server.uri());
    let mut chat = rust_llm::Chat::with_config(
        std::sync::Arc::new(config),
        Some("claude-haiku-4-5"),
        Some("anthropic"),
        false,
    )
    .unwrap()
    .with_tool(tool);
    chat.ask_later("go").unwrap();
    let mut m = Message::new(Role::Assistant, Some(String::new()));
    m.tool_calls = Some(
        [(
            "call_1".to_string(),
            ToolCall::new(
                "call_1",
                name,
                arguments.as_object().cloned().unwrap_or_default(),
            ),
        )]
        .into_iter()
        .collect(),
    );
    chat.add_message(m);
    chat.run_tools().await.unwrap();
    chat.messages().last().unwrap().content().to_string()
}

// spec: tool_spec.rb:112 returns an error hash for unknown keyword arguments
#[tokio::test]
async fn tools_answer_unknown_arguments_with_an_error() {
    let content = call_tool(
        Signature,
        "signature",
        json!({ "questions": [], "isOther": true }),
    )
    .await;
    assert_eq!(
        content,
        json!({ "error": "Invalid tool arguments: unknown keyword: isOther" }).to_string()
    );
}

// spec: tool_spec.rb:124 returns an error hash for missing required keyword arguments
#[tokio::test]
async fn tools_answer_missing_arguments_with_an_error() {
    let content = call_tool(Signature, "signature", json!({})).await;
    assert_eq!(
        content,
        json!({ "error": "Invalid tool arguments: missing keyword: questions" }).to_string()
    );
}

// spec: tool_spec.rb:160 returns an error hash for unknown arguments when execute takes no keywords
#[tokio::test]
async fn tools_without_parameters_reject_unexpected_arguments() {
    let content = call_tool(NoArgument, "no_argument", json!({ "unexpected": true })).await;
    assert_eq!(
        content,
        json!({ "error": "Invalid tool arguments: unknown keyword: unexpected" }).to_string()
    );
}

// spec: tool_spec.rb:254 uses an empty object schema for tools without keyword arguments
#[tokio::test]
async fn tools_without_parameters_render_an_empty_object_schema() {
    let mut config = Config::default();
    config.set("openai_api_key", "test");
    let mut chat = rust_llm::Chat::with_config(
        std::sync::Arc::new(config),
        Some("gpt-5-nano"),
        Some("openai"),
        false,
    )
    .unwrap()
    .with_tool(NoArgument);
    chat.ask_later("go").unwrap();
    let payload = chat.render().unwrap();
    assert_eq!(
        payload["tools"][0]["parameters"],
        json!({ "type": "object", "properties": {}, "required": [], "additionalProperties": false, "strict": true })
    );
}

// spec: tool_spec.rb:312 stringifies a result that is neither text nor structured data
// spec: tool_spec.rb:316 serializes structured results as JSON
#[test]
fn tool_results_serialize_structured_data_as_json() {
    assert_eq!(ToolResult::from(json!(42)).content, "42");
    assert_eq!(
        ToolResult::from(json!({ "ok": true })).content,
        r#"{"ok":true}"#
    );
    assert_eq!(ToolResult::from(json!([1, 2])).content, "[1,2]");
}

// spec: tool_spec.rb:351 reports nowhere outside a chat
#[test]
fn progress_reports_nowhere_outside_a_chat() {
    assert!(rust_llm::progress::listener().is_none());
    rust_llm::progress::report(progress(Some(1.0), Some(2.0), None)); // must not panic
}

// spec: tool_spec.rb:365 returns nothing without parameters
// spec: tool_spec.rb:370 gives array parameters a default item type
#[test]
fn parameter_schemas_default_array_items_to_strings() {
    assert!(rust_llm::tool::schema_from_parameters(&[]).is_none());
    let schema =
        rust_llm::tool::schema_from_parameters(&[Parameter::new("tags").kind("array")]).unwrap();
    assert_eq!(
        schema["properties"]["tags"],
        json!({ "type": "array", "items": { "type": "string" } })
    );
}

// spec: tool_spec.rb:391 maps #{declared} to #{expected}
#[test]
fn parameter_types_map_like_rubyllm() {
    for (declared, expected) in [
        ("integer", "integer"),
        ("int", "integer"),
        ("number", "number"),
        ("float", "number"),
        ("double", "number"),
        ("boolean", "boolean"),
        ("array", "array"),
        ("object", "object"),
        ("anything else", "string"),
    ] {
        let schema =
            rust_llm::tool::schema_from_parameters(&[Parameter::new("p").kind(declared)]).unwrap();
        assert_eq!(
            schema["properties"]["p"]["type"],
            json!(expected),
            "{declared}"
        );
    }
}

// ---- attachment_spec.rb -----------------------------------------------------------------------

use rust_llm::Attachment;
use rust_llm::attachment::AttachmentType;

fn fixture(name: &str) -> String {
    format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
}

fn uploaded(
    id: &str,
    provider: &str,
    filename: &str,
    size: u64,
    mime: Option<&str>,
) -> rust_llm::UploadedFile {
    rust_llm::UploadedFile {
        id: id.into(),
        provider: provider.into(),
        filename: Some(filename.into()),
        byte_size: Some(size),
        created_at: None,
        expires_at: None,
        status: None,
        mime_type: mime.map(str::to_string),
        purpose: None,
        uri: None,
        downloadable: None,
        metadata: Value::Null,
    }
}

// spec: attachment_spec.rb:8 supports path attachments from the public API
#[test]
fn path_attachments_read_their_name_and_type() {
    let a = Attachment::new(fixture("ruby.txt"));
    assert_eq!(
        (a.filename.as_deref(), a.mime_type.as_str()),
        (Some("ruby.txt"), "text/plain")
    );
}

// spec: attachment_spec.rb:49 classifies rich document files semantically
#[test]
fn attachments_classify_rich_documents() {
    let a = Attachment::from_bytes(b"docx bytes".to_vec(), "proposal.docx", None);
    assert_eq!(
        a.mime_type,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    );
    assert_eq!(a.kind(), AttachmentType::Document);
}

// spec: attachment_spec.rb:58 keeps text files in one attachment category
#[test]
fn text_attachments_are_text_not_documents() {
    assert_eq!(
        Attachment::from_bytes(b"notes".to_vec(), "notes.txt", None).kind(),
        AttachmentType::Text
    );
}

// spec: attachment_spec.rb:66 wraps provider-managed files without reading inline content
#[test]
fn provider_files_are_wrapped_without_inline_content() {
    let a = Attachment::from_uploaded(uploaded(
        "file_123",
        "anthropic",
        "proposal.pdf",
        1234,
        Some("application/pdf"),
    ));
    assert!(a.is_provider_file());
    assert_eq!(a.provider_file_id(), Some("file_123"));
    assert_eq!(
        (a.filename.as_deref(), a.mime_type.as_str(), a.byte_size()),
        (Some("proposal.pdf"), "application/pdf", Some(1234))
    );
    let err = a.encoded().unwrap_err();
    assert!(
        err.to_string().contains("cannot be read as inline"),
        "{err}"
    );
}

// spec: attachment_spec.rb:85 does not fetch URL content to determine byte size
#[test]
fn url_attachments_have_no_size_until_fetched() {
    assert_eq!(
        Attachment::new("https://example.com/report.pdf").byte_size(),
        None
    );
}

// spec: attachment_spec.rb:92 recognizes URL schemes regardless of case
#[test]
fn url_schemes_are_case_insensitive() {
    let a = Attachment::new("HTTPS://example.com/report.pdf");
    assert!(a.is_url());
    assert_eq!(a.filename.as_deref(), Some("report.pdf"));
}

// spec: attachment_spec.rb:100 passes a remote URL through without fetching it
#[test]
fn url_or_data_uri_passes_urls_through() {
    assert_eq!(
        Attachment::new("https://example.com/ruby.png")
            .url_or_data_uri()
            .unwrap(),
        "https://example.com/ruby.png"
    );
}

// spec: attachment_spec.rb:114 inlines an IO as a base64 data URI
#[test]
fn url_or_data_uri_inlines_bytes() {
    let a = Attachment::from_bytes(b"%PDF-1.4".to_vec(), "report.pdf", None);
    assert_eq!(
        a.url_or_data_uri().unwrap(),
        "data:application/pdf;base64,JVBERi0xLjQ="
    );
}

// spec: attachment_spec.rb:120 inlines text as a data URI rather than a file tag
#[test]
fn url_or_data_uri_inlines_text_as_a_data_uri() {
    let a = Attachment::from_bytes(b"notes".to_vec(), "notes.txt", None);
    assert_eq!(
        a.url_or_data_uri().unwrap(),
        "data:text/plain;base64,bm90ZXM="
    );
}

// spec: attachment_spec.rb:138 reports no provider id or URI for ordinary sources
#[test]
fn ordinary_attachments_have_no_provider_identity() {
    let a = Attachment::new(fixture("ruby.png"));
    assert!(!a.is_provider_file());
    assert_eq!((a.provider_file_id(), a.provider_file_uri()), (None, None));
}

// spec: attachment_spec.rb:146 derives the mime type from the provider filename when the record has none
#[test]
fn provider_files_derive_the_type_from_the_filename() {
    let a = Attachment::from_uploaded(uploaded("file_1", "openai", "notes.pdf", 1, None));
    assert_eq!(a.mime_type, "application/pdf");
}

// spec: attachment_spec.rb:164 reads the size off an IO that reports one
// spec: attachment_spec.rb:168 falls back to the file stat
#[test]
fn byte_size_reads_bytes_or_the_file_stat() {
    assert_eq!(
        Attachment::from_bytes(b"12345".to_vec(), "a.txt", None).byte_size(),
        Some(5)
    );
    assert!(
        Attachment::new(fixture("ruby.txt"))
            .byte_size()
            .is_some_and(|s| s > 0)
    );
}

// spec: attachment_spec.rb:229 takes the basename of a URL path
#[test]
fn url_filenames_are_the_path_basename() {
    assert_eq!(
        Attachment::new("https://example.com/docs/report.pdf")
            .filename
            .as_deref(),
        Some("report.pdf")
    );
}

// spec: attachment_spec.rb:235 is not a document when it is a PDF or plain text
// spec: attachment_spec.rb:241 reports unknown for a type it cannot place
#[test]
fn document_classification_excludes_pdf_and_text() {
    let kind = |name: &str| Attachment::from_bytes(b"x".to_vec(), name, None).kind();
    assert_eq!(kind("a.pdf"), AttachmentType::Pdf);
    assert_eq!(kind("a.txt"), AttachmentType::Text);
    assert_eq!(kind("a.docx"), AttachmentType::Document);
    assert_eq!(kind("a.bin"), AttachmentType::Unknown);
}

// spec: attachment_spec.rb:247 takes the size and filename off the record
#[test]
fn provider_files_take_size_name_and_type_off_the_record() {
    let a = Attachment::from_uploaded(uploaded(
        "file_1",
        "openai",
        "batch.jsonl",
        42,
        Some("application/jsonl"),
    ));
    assert_eq!(
        (
            a.byte_size(),
            a.filename.as_deref(),
            a.mime_type.as_str(),
            a.provider_file_id()
        ),
        (
            Some(42),
            Some("batch.jsonl"),
            "application/jsonl",
            Some("file_1")
        )
    );
}

// spec: attachment_spec.rb:260 accepts a media resolution
#[test]
fn attachments_accept_a_media_resolution() {
    let a = Attachment::from_bytes(b"png".to_vec(), "page.png", None)
        .with_resolution(rust_llm::Resolution::UltraHigh);
    assert_eq!(a.resolution, Some(rust_llm::Resolution::UltraHigh));
}

// ---- remaining message/chat/model examples ------------------------------------------------------

// spec: message_spec.rb:23 normalizes nil content to empty string for assistant tool-call messages
#[tokio::test]
async fn a_parsed_tool_call_turn_has_empty_content() {
    // Ruby normalizes at construction; the port normalizes where messages are built from the wire.
    let server = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::any())
        .respond_with(wiremock::ResponseTemplate::new(200).set_body_json(json!({
            "id": "c1", "model": "deepseek-v4-flash", "object": "chat.completion",
            "choices": [{ "index": 0, "finish_reason": "tool_calls", "message": { "role": "assistant", "content": null,
                "tool_calls": [{ "id": "call_1", "type": "function", "function": { "name": "weather", "arguments": "{}" } }] } }],
            "usage": { "prompt_tokens": 1, "completion_tokens": 1 }
        })))
        .mount(&server)
        .await;
    let mut config = Config::default();
    config.set("deepseek_api_base", server.uri());
    config.set("deepseek_api_key", "test");
    let mut chat = rust_llm::Chat::with_config(
        std::sync::Arc::new(config),
        Some("deepseek-v4-flash"),
        Some("deepseek"),
        false,
    )
    .unwrap();
    chat.ask_later("weather?").unwrap();
    let message = chat.generate().await.unwrap();
    assert_eq!(message.content, Some(String::new()));
}

// spec: message_spec.rb:138 marks the message as a cache boundary
// spec: message_spec.rb:273 includes cache_until_here when marked
#[test]
fn cache_until_here_marks_the_message() {
    let mut m = Message::user("hello");
    assert!(!m.cache_until_here);
    m.cache_until_here = true;
    assert!(m.cache_until_here);
}

// spec: models/lookup_spec.rb:35 prefers the resolved alias when a provider is specified
#[test]
fn lookup_prefers_the_resolved_alias_for_a_provider() {
    let exact = Model::default_for("claude-haiku-4-5", "anthropic");
    let aliased = Model::default_for("claude-haiku-4-5-20251001", "anthropic");
    let registry = rust_llm::models::Models::new(vec![exact, aliased]);
    assert_eq!(
        registry
            .find("claude-haiku-4-5", Some("anthropic"))
            .unwrap()
            .id,
        "claude-haiku-4-5-20251001"
    );
}

// spec: models/lookup_spec.rb:42 falls back to the exact id when the resolved alias belongs to another provider
#[test]
fn lookup_falls_back_to_the_exact_id() {
    let exact = Model::default_for("claude-haiku-4-5", "anthropic");
    let other = Model::default_for("claude-haiku-4-5-20251001", "azure");
    let registry = rust_llm::models::Models::new(vec![other, exact]);
    let found = registry
        .find("claude-haiku-4-5", Some("anthropic"))
        .unwrap();
    assert_eq!(
        (found.id.as_str(), found.provider.as_str()),
        ("claude-haiku-4-5", "anthropic")
    );
}

// spec: models/lookup_spec.rb:50 prefers a first-party alias over another provider with the exact id
#[test]
fn lookup_prefers_a_first_party_alias() {
    let other = Model::default_for("claude-haiku-4-5", "vertexai");
    let aliased = Model::default_for("claude-haiku-4-5-20251001", "anthropic");
    let registry = rust_llm::models::Models::new(vec![other, aliased]);
    assert_eq!(
        registry.find("claude-haiku-4-5", None).unwrap().provider,
        "anthropic"
    );
    assert_eq!(
        registry
            .find("claude-haiku-4-5", Some("vertexai"))
            .unwrap()
            .provider,
        "vertexai"
    );
}

// spec: models_spec.rb:116 includes provider-specific refresh guidance for unknown models
#[test]
fn unknown_models_name_the_provider() {
    let err = rust_llm::models()
        .find("nonexistent-model-12345", Some("openai"))
        .unwrap_err();
    assert!(
        err.to_string()
            .contains(r#"Unknown model: "nonexistent-model-12345" for provider: "openai""#),
        "{err}"
    );
}

// spec: models_spec.rb:129 prioritizes exact matches over aliases
#[test]
fn exact_ids_win_over_aliases() {
    assert_eq!(
        rust_llm::models()
            .find("gemini-2.5-flash", None)
            .unwrap()
            .id,
        "gemini-2.5-flash"
    );
    assert_eq!(
        rust_llm::models()
            .find("gemini-2.5-flash", Some("gemini"))
            .unwrap()
            .id,
        "gemini-2.5-flash"
    );
    assert_eq!(
        rust_llm::models().find("gemini-flash", None).unwrap().id,
        "gemini-flash-latest"
    );
}

// spec: tool_spec.rb:7 sets and returns the tool description
// spec: tool_spec.rb:18 accepts a description for the parameter
#[test]
fn tools_describe_themselves_and_their_parameters() {
    assert_eq!(Signature.description(), "Signature");
    let p = Parameter::new("latitude").description("Latitude");
    assert_eq!(p.description.as_deref(), Some("Latitude"));
}

// spec: tool_spec.rb:69 keeps an instance-level override working
#[test]
fn tool_name_overrides_win() {
    struct InstanceNamed;
    #[async_trait]
    impl Tool for InstanceNamed {
        fn name(&self) -> String {
            "custom".into()
        }
        fn description(&self) -> String {
            String::new()
        }
        async fn execute(
            &self,
            _a: Map<String, Value>,
            _c: &ToolCall,
        ) -> Result<ToolResult, ToolError> {
            Ok("".into())
        }
    }
    assert_eq!(InstanceNamed.name(), "custom");
}
