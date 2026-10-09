//! `spec/ruby_llm/accounting/usage_spec.rb` ported against `rust_llm::accounting::Tracker`
//! (`accounting/usage.rb`). `// spec:` lines tie each test to its Ruby example.
//!
//! Ruby stubs `RubyLLM.models` per example; the Rust registry is process-wide, so every test here
//! holds `Registry::lock` and the guard restores the original registry when it drops.

use std::sync::{Arc, Mutex, MutexGuard};

use rust_llm::accounting::{Tracker, UsageResult};
use rust_llm::error::ErrorResponse;
use rust_llm::message::Operation;
use rust_llm::model::{PricingCategory, PricingTier};
use rust_llm::models::Models;
use rust_llm::{
    Chat, Config, Cost, Embedding, Error, Image, Message, Model, Moderation, Rerank, Role, Speech,
    Tokens, Transcription, UsageEntry, UsageStatus, Vectors,
};
use serde_json::json;

static REGISTRY: Mutex<()> = Mutex::new(());

struct Registry {
    original: Vec<Model>,
    _lock: MutexGuard<'static, ()>,
}

impl Registry {
    /// Serializes registry access; `Some(models)` stands in for `allow(RubyLLM).to receive(:models)`.
    fn lock(models: Option<Vec<Model>>) -> Registry {
        let lock = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
        let original = rust_llm::models().all_including_unlisted().to_vec();
        if let Some(models) = models {
            Models::install(models);
        }
        Registry {
            original,
            _lock: lock,
        }
    }
}

impl Drop for Registry {
    fn drop(&mut self) {
        Models::install(std::mem::take(&mut self.original));
    }
}

fn config() -> Arc<Config> {
    let mut c = Config::default();
    c.set("openai_api_key", "test");
    Arc::new(c)
}

/// `let(:model) { RubyLLM::Model.new(id: 'test-model', name: 'Test Model', provider: 'openai') }`.
fn test_model() -> Model {
    Model::default_for("test-model", "openai")
}

/// `build_tracker`: a chat tracker for the `openai` provider.
fn build_tracker(model: Model) -> Tracker {
    Tracker::new(Operation::Chat, "openai", Some(model), config(), None)
}

/// `model_for(:openai, :temperature)`.
fn temperature_model() -> Model {
    rust_llm::models()
        .find("gpt-4.1-nano", Some("openai"))
        .expect("gpt-4.1-nano in the registry")
}

fn assistant(
    content: &str,
    model: Option<&str>,
    input: Option<i64>,
    output: Option<i64>,
) -> Message {
    let mut m = Message::new(Role::Assistant, Some(content.to_string()));
    m.model = model.map(str::to_string);
    m.tokens = Tokens {
        input,
        output,
        ..Default::default()
    };
    m
}

fn response(status: u16) -> Option<ErrorResponse> {
    Some(ErrorResponse {
        status,
        body: String::new(),
        ..Default::default()
    })
}

fn priced(id: &str, provider: &str, input: f64, output: f64) -> Model {
    let mut m = Model::default_for(id, provider);
    m.pricing.text_tokens = Some(PricingCategory {
        standard: Some(PricingTier {
            input_per_million: Some(input),
            output_per_million: Some(output),
            ..Default::default()
        }),
        ..Default::default()
    });
    m
}

/// `describe 'provider-specific message pricing'`: `other_model`, `model`, and `registry`.
fn other_model() -> Model {
    priced("z-ai/glm-5.3-flash", "openrouter", 0.075, 0.25)
}

fn custom_model() -> Model {
    priced("z-ai/glm-5.3-flash", "custom", 0.2, 0.5)
}

fn custom_tracker(model: Model) -> Tracker {
    Tracker::new(Operation::Chat, "custom", Some(model), config(), None)
}

fn custom_result() -> Message {
    assistant("ok", Some("z-ai/glm-5.3-flash"), Some(19), Some(17))
}

fn close(actual: Option<f64>, expected: f64) {
    let a = actual.unwrap_or_else(|| panic!("expected {expected}, got None"));
    assert!((a - expected).abs() < 1e-12, "expected {expected}, got {a}");
}

// spec: accounting/usage_spec.rb:31 retains absent model identity and unknown token accounting for model-free operations
#[test]
fn a_model_free_operation_keeps_no_model_and_unknown_tokens() {
    let _r = Registry::lock(None);
    let mut tracker = Tracker::new(Operation::Moderation, "openai", None, config(), None);
    let entry = tracker.start();
    let mut result = Moderation {
        id: Some("request".into()),
        model: String::new(),
        results: vec![],
        raw: json!({}),
        usage_entries: vec![],
    };
    tracker.succeed(&mut result);

    let h = tracker.entry(entry).unwrap().to_h();
    assert_eq!(
        (h["model"].clone(), h["status"].clone(), h["tokens"].clone()),
        (json!(null), json!("succeeded"), json!({}))
    );
    assert_eq!(
        result.usage_entries,
        vec![tracker.entry(entry).unwrap().clone()]
    );
    assert_eq!(result.cost().total(), None);

    let refused = tracker.start();
    tracker.fail_attempt(
        Some(refused),
        &Error::Forbidden("Not allowed".into(), response(403)),
    );
    let h = tracker.entry(refused).unwrap().to_h();
    assert_eq!(
        (h["model"].clone(), h["status"].clone(), h["tokens"].clone()),
        (json!(null), json!("failed"), json!({}))
    );
}

// spec: accounting/usage_spec.rb:92 prices against the requested model when the provider echoes an unregistered id
#[test]
fn an_unregistered_echoed_id_is_priced_against_the_requested_model() {
    let _r = Registry::lock(None);
    let mut tracker = build_tracker(temperature_model());
    let entry = tracker.start();
    let mut result = assistant("hi", Some("gpt-4.1-nano-2099-01-01"), Some(10), Some(4));

    tracker.succeed(&mut result);

    let entry_total = tracker.entry(entry).unwrap().cost.total();
    assert!(entry_total.is_some_and(|t| t > 0.0), "{entry_total:?}");
    assert_eq!(result.cost(None).total(), entry_total);
}

// spec: accounting/usage_spec.rb:109 keeps tokens unknown for attempts that may have been billed
#[test]
fn a_possibly_billed_failure_keeps_tokens_unknown() {
    let _r = Registry::lock(None);
    let mut tracker = build_tracker(test_model());
    let id = tracker.start();

    tracker.fail_attempt(Some(id), &Error::Timeout("execution expired".into()));

    let entry = tracker.entry(id).unwrap();
    assert_eq!(entry.status, UsageStatus::Failed);
    assert_eq!(entry.to_h()["tokens"], json!({}));
    assert!(!entry.usage_available());
}

// spec: accounting/usage_spec.rb:141 provider-specific message pricing > replaces a providerless lookup before recording the cost
#[test]
fn a_providerless_lookup_is_replaced_before_pricing() {
    let _r = Registry::lock(Some(vec![other_model(), custom_model()]));
    let mut result = custom_result();
    assert_eq!(result.model_info(), Some(other_model()));
    let mut tracker = custom_tracker(custom_model());
    let id = tracker.start();

    tracker.succeed(&mut result);

    let entry = tracker.entry(id).unwrap();
    assert_eq!(result.model_info(), Some(custom_model()));
    assert_eq!(entry.provider, "custom");
    close(entry.cost.total(), 0.0000123);
    assert_eq!(result.cost(None).total(), entry.cost.total());
}

// spec: accounting/usage_spec.rb:154 provider-specific message pricing > falls back to the requested model when only another provider knows the echoed id
#[test]
fn an_echoed_id_known_only_to_another_provider_falls_back_to_the_requested_model() {
    let mut elsewhere = other_model();
    elsewhere.id = "gpt-4.1".into();
    let _r = Registry::lock(Some(vec![other_model(), custom_model(), elsewhere]));
    let mut result = assistant("ok", Some("gpt-4.1"), Some(19), Some(17));
    let mut tracker = custom_tracker(custom_model());
    let id = tracker.start();

    tracker.succeed(&mut result);

    assert_eq!(result.model.as_deref(), Some("gpt-4.1"));
    assert_eq!(result.model_info(), Some(custom_model()));
    close(tracker.entry(id).unwrap().cost.total(), 0.0000123);
}

// spec: accounting/usage_spec.rb:168 provider-specific message pricing > uses a different echoed model when it belongs to the same provider
#[test]
fn an_echoed_model_of_the_same_provider_prices_the_call() {
    let mut echoed = other_model();
    echoed.id = "gpt-4.1".into();
    echoed.provider = "custom".into();
    let _r = Registry::lock(Some(vec![other_model(), custom_model(), echoed.clone()]));
    let mut result = assistant("ok", Some("gpt-4.1"), Some(19), Some(17));
    let mut tracker = custom_tracker(custom_model());
    let id = tracker.start();

    tracker.succeed(&mut result);

    assert_eq!(result.model_info(), Some(echoed.clone()));
    let expected = echoed.cost_for(&result.tokens()).total();
    assert!(expected.is_some());
    assert_eq!(tracker.entry(id).unwrap().cost.total(), expected);
}

// spec: accounting/usage_spec.rb:183 provider-specific message pricing > preserves a provider-reported cost of #{amount}
#[test]
fn a_provider_reported_cost_is_preserved() {
    let _r = Registry::lock(Some(vec![other_model(), custom_model()]));
    for amount in [0.0, 0.0042] {
        let mut result = custom_result();
        result.tokens.reported_cost = Some(amount);
        let mut tracker = custom_tracker(custom_model());
        let id = tracker.start();

        tracker.succeed(&mut result);

        assert_eq!(result.model_info(), Some(custom_model()));
        assert_eq!(tracker.entry(id).unwrap().cost.total(), Some(amount));
        assert_eq!(result.cost(None).total(), Some(amount));
    }
}

// spec: accounting/usage_spec.rb:197 provider-specific message pricing > preserves an explicitly supplied cost
#[test]
fn an_explicitly_supplied_cost_is_preserved() {
    let _r = Registry::lock(Some(vec![other_model(), custom_model()]));
    let mut result = custom_result().with_cost(Cost::from_h(&json!({ "total": 0.003 }), None));
    let mut tracker = custom_tracker(custom_model());
    let id = tracker.start();

    tracker.succeed(&mut result);

    assert_eq!(tracker.entry(id).unwrap().cost.total(), Some(0.003));
    assert_eq!(result.cost(None).total(), Some(0.003));
}

// spec: accounting/usage_spec.rb:209 provider-specific message pricing > leaves missing provider pricing unknown
#[test]
fn missing_provider_pricing_stays_unknown() {
    let mut unpriced = custom_model();
    unpriced.pricing = Default::default();
    let _r = Registry::lock(Some(vec![other_model(), unpriced.clone()]));
    let mut result = custom_result();
    let mut tracker = custom_tracker(unpriced.clone());
    let id = tracker.start();

    tracker.succeed(&mut result);

    assert_eq!(result.model_info(), Some(unpriced));
    assert_eq!(tracker.entry(id).unwrap().cost.total(), None);
    assert_eq!(result.cost(None).total(), None);
}

fn recorded_entry(status: UsageStatus, cost: Option<Cost>) -> UsageEntry {
    let mut entry = UsageEntry::new(Operation::Chat, "openrouter", Some("test-model"));
    entry.status = status;
    if let Some(cost) = cost {
        entry.cost = cost;
    }
    entry
}

// spec: accounting/usage_spec.rb:301 recognizes an exact cost even when token counts are unavailable
#[test]
fn an_exact_cost_counts_without_token_counts() {
    let _r = Registry::lock(None);
    let entry = recorded_entry(
        UsageStatus::Pending,
        Some(Cost::from_h(&json!({ "total": 0.0042 }), None)),
    );
    let mut message = Message::new(Role::Assistant, Some("hi".to_string()));
    message.usage_entries = vec![entry.clone()];
    let mut chat =
        Chat::with_config(config(), Some("gpt-4.1-nano"), Some("openai"), false).unwrap();
    chat.set_usage_entries(vec![entry.clone()]);

    assert!(entry.cost_available());
    assert!(!entry.usage_available());
    assert_eq!(message.cost(None).total(), Some(0.0042));
    assert_eq!(chat.cost().total(), Some(0.0042));
}

// spec: accounting/usage_spec.rb:318 keeps aggregate cost unknown when any potentially billed attempt is unknown
#[test]
fn one_unknown_attempt_keeps_the_aggregate_unknown() {
    let _r = Registry::lock(None);
    let known = recorded_entry(
        UsageStatus::Pending,
        Some(Cost::from_h(&json!({ "total": 0.0042 }), None)),
    );
    let unknown = recorded_entry(UsageStatus::Failed, None);
    let mut chat =
        Chat::with_config(config(), Some("gpt-4.1-nano"), Some("openai"), false).unwrap();
    chat.set_usage_entries(vec![known, unknown]);

    assert_eq!(chat.cost().total(), None);
}

// spec: accounting/usage_spec.rb:387 ignores a second failure for an attempt that already finished
#[test]
fn a_second_failure_is_ignored() {
    let _r = Registry::lock(None);
    let mut tracker = build_tracker(test_model());
    let id = tracker.start();
    tracker.fail_attempt(Some(id), &Error::ConnectionFailed("reset".into()));

    tracker.fail_attempt(Some(id), &Error::Timeout("late".into()));
    let entry = tracker.entry(id).unwrap();
    assert_eq!(entry.status, UsageStatus::Failed);
    // Still the never-sent zero, not the timeout's unknown.
    assert_eq!(
        entry.to_h()["tokens"],
        json!({ "input_tokens": 0, "output_tokens": 0 })
    );
    tracker.fail_attempt(None, &Error::Timeout("late".into()));
}

// spec: accounting/usage_spec.rb:396 fails every attempt still in flight
#[test]
fn every_attempt_in_flight_fails() {
    let _r = Registry::lock(None);
    let mut tracker = build_tracker(test_model());
    let first = tracker.start();
    let second = tracker.start();

    tracker.fail_pending(&Error::ConnectionFailed("reset".into()));

    assert!(tracker.entry(first).unwrap().is_failed());
    assert!(tracker.entry(second).unwrap().is_failed());
}

// spec: accounting/usage_spec.rb:406 ignores a chunk when nothing is in flight
#[test]
fn a_chunk_with_nothing_in_flight_is_ignored() {
    let _r = Registry::lock(None);
    let mut tracker = build_tracker(test_model());
    let mut chunk = Message::new(Role::Assistant, Some("x".to_string()));
    chunk.tokens.input = Some(3);

    tracker.observe(&chunk);

    assert!(tracker.entries().is_empty());
}

// spec: accounting/usage_spec.rb:412 ignores an observation that carries no tokens
// Ruby observes `Object.new` (no `tokens` method); the Rust counterpart is a chunk reporting none.
#[test]
fn an_observation_without_tokens_leaves_the_attempt_unknown() {
    let _r = Registry::lock(None);
    let mut tracker = build_tracker(test_model());
    let id = tracker.start();

    tracker.observe(&Message::new(Role::Assistant, Some("x".to_string())));

    assert_eq!(tracker.entry(id).unwrap().to_h()["tokens"], json!({}));
    assert!(tracker.entry(id).unwrap().is_pending());
}

// spec: accounting/usage_spec.rb:421 attaches the ledger to a result even when no attempt was recorded
#[test]
fn the_ledger_is_attached_without_attempts() {
    let _r = Registry::lock(None);
    let mut tracker = build_tracker(test_model());
    let mut result = Message::new(Role::Assistant, Some("hi".to_string()));
    result.usage_entries = vec![recorded_entry(UsageStatus::Succeeded, None)];

    tracker.succeed(&mut result);

    assert!(result.usage_entries.is_empty());
    assert_eq!(result.content(), "hi");
}

// spec: accounting/usage_spec.rb:429 credits only the last attempt with the tokens the call used
#[test]
fn only_the_last_attempt_is_credited() {
    let _r = Registry::lock(None);
    let mut tracker = build_tracker(test_model());
    let retried = tracker.start();
    let last = tracker.start();
    let mut result = assistant("hi", None, Some(10), Some(4));

    tracker.succeed(&mut result);

    assert_eq!(tracker.entry(retried).unwrap().to_h()["tokens"], json!({}));
    assert_eq!(tracker.entry(last).unwrap().tokens.input, Some(10));
    let ids: Vec<u64> = result.usage_entries.iter().map(|e| e.id).collect();
    assert_eq!(ids, [retried, last]);
    assert_eq!(result.usage_entries, tracker.entries());
}

// spec: accounting/usage_spec.rb:49 keeps the usage a response reported when the attempt then fails
#[test]
fn a_failed_attempt_keeps_the_usage_its_response_reported() {
    let _r = Registry::lock(None);
    let mut tracker = build_tracker(test_model());
    let id = tracker.start();
    tracker.observe_tokens(&Tokens {
        input: Some(133),
        output: Some(0),
        ..Default::default()
    });
    assert!(tracker.entry(id).unwrap().is_pending());

    tracker.fail_attempt(
        Some(id),
        &Error::ContentFilter("Blocked".into(), response(200)),
    );

    let entry = tracker.entry(id).unwrap();
    assert_eq!(entry.status, UsageStatus::Failed);
    assert_eq!(
        entry.to_h()["tokens"],
        json!({ "input_tokens": 133, "output_tokens": 0 })
    );
}

/// `describe 'provider-specific operation pricing'`: `priced_model(id, provider, rate)` prices
/// text, audio, embeddings, and images at `rate` in and `rate * 4` out.
fn priced_model(id: &str, provider: &str, rate: f64) -> Model {
    let price = PricingCategory {
        standard: Some(PricingTier {
            input_per_million: Some(rate),
            output_per_million: Some(rate * 4.0),
            ..Default::default()
        }),
        ..Default::default()
    };
    let mut m = Model::default_for(id, provider);
    m.pricing.text_tokens = Some(price.clone());
    m.pricing.audio_tokens = Some(price.clone());
    m.pricing.embeddings = Some(price.clone());
    m.pricing.images = Some(price);
    m
}

/// `track(operation, result, provider:, other:)`: the registry holds the result's model id at
/// `other` (rate 0.5, preferred by a providerless lookup) and at `provider` (rate 1.0); the tracker
/// runs for `provider`. Returns the entry and the `provider` model. The tracker takes a provider
/// slug, so slugs the port has no provider for (`vertexai`, `azure`, `custom`) work as in Ruby.
fn track<R: UsageResult>(
    operation: Operation,
    result: &mut R,
    id: &str,
    provider: &str,
    other: &str,
) -> (UsageEntry, Model) {
    let model = priced_model(id, provider, 1.0);
    Models::install(vec![priced_model(id, other, 0.5), model.clone()]);
    let mut tracker = Tracker::new(operation, provider, Some(model.clone()), config(), None);
    let entry = tracker.start();
    tracker.succeed(result);
    (tracker.entry(entry).unwrap().clone(), model)
}

// spec: accounting/usage_spec.rb:244 provider-specific operation pricing > prices a transcription with the model of the provider that transcribed it
#[test]
fn a_transcription_is_priced_with_the_transcribing_providers_model() {
    let _r = Registry::lock(None);
    // `model_for(:vertexai, :transcription)`.
    let id = "gemini-2.5-flash";
    let mut result =
        Transcription::new(Some("ok".into()), id).with_token_counts(Some(1000), Some(500));

    let (entry, model) = track(
        Operation::Transcription,
        &mut result,
        id,
        "vertexai",
        "gemini",
    );

    assert_eq!(result.model_info(), Some(model));
    close(entry.cost.total(), 0.003);
    assert_eq!(result.cost().total(), entry.cost.total());
}

// spec: accounting/usage_spec.rb:255 provider-specific operation pricing > prices an embedding with the model of the provider that embedded it
#[test]
fn an_embedding_is_priced_with_the_embedding_providers_model() {
    let _r = Registry::lock(None);
    // `model_for(:gemini, :embedding)`.
    let id = "gemini-embedding-001";
    let mut result = Embedding::new(Vectors::Single(vec![0.1, 0.2]), id.into(), Some(1000));

    let (entry, model) = track(Operation::Embedding, &mut result, id, "vertexai", "gemini");

    assert_eq!(result.model_info(), Some(model));
    close(entry.cost.total(), 0.001);
    assert_eq!(result.cost().total(), entry.cost.total());
}

// spec: accounting/usage_spec.rb:265 provider-specific operation pricing > prices speech with the model of the provider that generated it
#[test]
fn speech_is_priced_with_the_generating_providers_model() {
    let _r = Registry::lock(None);
    // `model_for(:azure, :azure_speech)`.
    let id = "gpt-4o-mini-tts";
    let mut result = Speech::new(b"audio".to_vec(), id, None, None, None)
        .with_token_counts(Some(1000), Some(500));

    let (entry, model) = track(Operation::Speech, &mut result, id, "azure", "openai");

    assert_eq!(result.model_info(), Some(model));
    close(entry.cost.total(), 0.003);
    assert_eq!(result.cost().total(), entry.cost.total());
}

// spec: accounting/usage_spec.rb:276 provider-specific operation pricing > prices every image with the model of the provider that generated it
#[test]
fn every_image_is_priced_with_the_generating_providers_model() {
    let _r = Registry::lock(None);
    // `model_for(:vertexai, :image)`.
    let id = "gemini-3.1-flash-lite-image";
    let image = |usage| {
        let mut image = Image::new(id, usage);
        image.data = Some("aW1hZ2U=".into());
        image
    };
    let mut images = vec![
        image(json!({ "input_tokens": 1000, "output_tokens": 500 })),
        image(json!({})),
    ];

    let (entry, model) = track(Operation::Image, &mut images, id, "vertexai", "gemini");

    let infos: Vec<Option<Model>> = images.iter().map(Image::model_info).collect();
    assert_eq!(infos, [Some(model.clone()), Some(model)]);
    close(entry.cost.total(), 0.003);
    assert_eq!(images[0].cost().total(), entry.cost.total());
}

// spec: accounting/usage_spec.rb:290 provider-specific operation pricing > prices a rerank with the model of the provider that ranked it
#[test]
fn a_rerank_is_priced_with_the_ranking_providers_model() {
    let _r = Registry::lock(None);
    // `model_for(:cohere, :rerank)`.
    let id = "rerank-v3.5";
    let mut result = Rerank::new(vec![], id, Some(1000));

    let (entry, model) = track(Operation::Rerank, &mut result, id, "custom", "cohere");

    assert_eq!(result.model_info(), Some(model));
    close(entry.cost.total(), 0.001);
    assert_eq!(result.cost().total(), entry.cost.total());
}

// spec: accounting/usage_spec.rb:347 prices a streaming attempt only when its cost is read
// Ruby spies on `Cost.new` to show `Entry#cost` is built lazily on first read. `UsageEntry.cost` is
// a plain field here, so `observe` reprices each chunk (an arithmetic-only `Cost::new` with no
// registry lookup); what the test can observe is the same: the merged tokens and, for the
// unpriced `test-model`, an unknown total.
#[test]
fn a_streaming_attempt_is_unpriced_for_an_unpriced_model() {
    let _r = Registry::lock(None);
    let mut tracker = build_tracker(test_model());
    let id = tracker.start();

    for _ in 0..3 {
        tracker.observe(&assistant("part", None, None, Some(4)));
    }

    let entry = tracker.entry(id).unwrap();
    assert_eq!(entry.cost.total(), None);
    assert_eq!(entry.to_h()["tokens"], json!({ "output_tokens": 4 }));
}
