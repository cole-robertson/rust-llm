//! Provider batch specs: `provider_spec.rb` `#batch_cost` and OpenAI batch protocol routing,
//! `providers/openai_batches_spec.rb`, `providers/mistral/chat_completions/batches_spec.rb`, and
//! `providers/xai/chat_completions/batches_spec.rb`. RubyLLM stubs `@connection` with
//! `instance_double`s; here the provider talks to a wiremock server instead, and the private
//! routing helpers are reached through `#[doc(hidden)]` ports in `rust_llm::batch`.

mod support;

use std::sync::Arc;
use std::time::{Duration, Instant};

use rust_llm::batch::{
    batch_cost, mistral_batch_endpoint, openai_batch_protocol_for_endpoint,
    openai_batch_protocol_name_for,
};
use rust_llm::cost::Component;
use rust_llm::model::{Pricing, PricingCategory, PricingTier};
use rust_llm::{
    Batch, BatchStatus, Chat, Config, EmbedOptions, Error, Model, ProtocolName, Provider, Tokens,
    Vectors, embed_later,
};
use serde_json::{Value, json};
use support::{Cassette, config_for};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

// ---- helpers ------------------------------------------------------------------------------------

fn tier(input: Option<f64>, output: Option<f64>) -> PricingTier {
    PricingTier {
        input_per_million: input,
        output_per_million: output,
        ..Default::default()
    }
}

/// `RubyLLM::Model.new(id:, name:, provider:, pricing: { text_tokens: category })`.
fn model_with(id: &str, provider: &str, category: PricingCategory) -> Model {
    let mut model = Model::default_for(id, provider);
    model.pricing = Pricing {
        text_tokens: Some(category),
        ..Default::default()
    };
    model
}

/// The spec's `let(:pricing)`: $1 in / $5 out per million.
fn standard_pricing() -> PricingCategory {
    PricingCategory {
        standard: Some(tier(Some(1.0), Some(5.0))),
        ..Default::default()
    }
}

fn tokens(input: i64, output: i64) -> Tokens {
    Tokens {
        input: Some(input),
        output: Some(output),
        ..Default::default()
    }
}

fn close(actual: Option<f64>, expected: f64) -> bool {
    actual.is_some_and(|a| (a - expected).abs() < 1e-12)
}

/// Every batch provider this lane touches pointed at `server`, no retries.
fn config(server: &MockServer) -> Arc<Config> {
    let mut c = Config::default();
    for provider in ["openai", "mistral", "xai"] {
        c.set(
            format!("{provider}_api_base"),
            format!("{}/v1", server.uri()),
        );
        c.set(format!("{provider}_api_key"), "test-key");
    }
    c.max_retries = 0;
    Arc::new(c)
}

async fn get(server: &MockServer, at: &str, body: Value) {
    Mock::given(method("GET"))
        .and(path(at))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(server)
        .await;
}

async fn received(server: &MockServer, verb: &str, at: &str) -> Vec<wiremock::Request> {
    server
        .received_requests()
        .await
        .unwrap_or_default()
        .into_iter()
        .filter(|r| r.method.as_str() == verb && r.url.path() == at)
        .collect()
}

fn content(message: &Option<rust_llm::Message>) -> String {
    message
        .as_ref()
        .map(|m| m.content().to_string())
        .unwrap_or_default()
}

// ---- provider_spec.rb #batch_cost ---------------------------------------------------------------

// spec: provider_spec.rb:211 #batch_cost > prices thinking billed as output at the batch rate
#[test]
fn prices_thinking_billed_as_output_at_the_batch_rate() {
    let thinking_tokens = Tokens {
        thinking: Some(500),
        ..tokens(1_000, 2_000)
    };
    let model = model_with("test-model", "openai", standard_pricing());

    let cost = batch_cost(Provider::OpenAI, &thinking_tokens, &model);

    assert_eq!(cost.thinking, None);
    assert!(close(cost.total(), 0.0055), "{:?}", cost.total());
}

// spec: provider_spec.rb:222 #batch_cost > prices thinking billed as output at an explicit batch rate
#[test]
fn prices_thinking_billed_as_output_at_an_explicit_batch_rate() {
    let explicit = PricingCategory {
        batch: Some(tier(Some(0.4), Some(2.0))),
        ..standard_pricing()
    };
    let model = model_with("test-model", "openai", explicit);
    let thinking_tokens = Tokens {
        thinking: Some(500),
        ..tokens(1_000, 2_000)
    };

    assert!(close(
        batch_cost(Provider::OpenAI, &thinking_tokens, &model).total(),
        0.0044
    ));
}

// spec: provider_spec.rb:236 #batch_cost > prices separately billed thinking at the batch rate
#[test]
fn prices_separately_billed_thinking_at_the_batch_rate() {
    let reasoning = PricingCategory {
        standard: Some(PricingTier {
            reasoning_output_per_million: Some(10.0),
            ..tier(Some(1.0), Some(5.0))
        }),
        ..Default::default()
    };
    let model = model_with("test-model", "openai", reasoning);
    let thinking_tokens = Tokens {
        thinking: Some(500),
        ..tokens(1_000, 2_000)
    };

    let cost = batch_cost(Provider::OpenAI, &thinking_tokens, &model);

    assert!(close(cost.thinking, 0.0025), "{:?}", cost.thinking);
    assert!(close(cost.total(), 0.008), "{:?}", cost.total());
}

// spec: provider_spec.rb:262 #batch_cost > does not count unused components as missing when no batch rate applies
#[test]
fn does_not_count_unused_components_as_missing_when_no_batch_rate_applies() {
    let model = model_with("test-model", "xai", standard_pricing());
    let tokens = Tokens {
        cache_read: Some(0),
        ..tokens(1_000, 2_000)
    };

    let cost = batch_cost(Provider::XAI, &tokens, &model);

    assert!(cost.missing().contains(&Component::Input));
    assert!(!cost.missing().contains(&Component::CacheRead));
}

// spec: provider_spec.rb:273 #batch_cost > does not present a partial component sum as a complete batch cost
#[test]
fn does_not_present_a_partial_component_sum_as_a_complete_batch_cost() {
    let partial = PricingCategory {
        standard: Some(tier(Some(1.0), None)),
        ..Default::default()
    };
    let model = model_with("test-model", "openai", partial);

    let cost = batch_cost(Provider::OpenAI, &tokens(1_000, 2_000), &model);

    assert_eq!(cost.input, Some(0.0005));
    assert_eq!(cost.output, None);
    assert_eq!(cost.total(), None);
}

// spec: provider_spec.rb:291 #batch_cost > does not combine Gemini batch and context-cache discounts
#[test]
fn does_not_combine_gemini_batch_and_context_cache_discounts() {
    let cached_tokens = Tokens {
        cache_read: Some(3_000),
        ..tokens(1_000, 2_000)
    };
    let cached = PricingCategory {
        standard: Some(PricingTier {
            cache_read_input_per_million: Some(0.1),
            ..tier(Some(1.0), Some(5.0))
        }),
        ..Default::default()
    };
    let model = model_with("gemini-test", "gemini", cached);

    let cost = batch_cost(Provider::Gemini, &cached_tokens, &model);

    assert_eq!(cost.input, Some(0.0005));
    assert_eq!(cost.output, Some(0.005));
    assert!(close(cost.cache_read, 0.0003), "{:?}", cost.cache_read);
    assert!(close(cost.total(), 0.0058), "{:?}", cost.total());
}

// spec: provider_spec.rb:332 #batch_cost > applies the batch modifier to long-context rates
#[test]
fn applies_the_batch_modifier_to_long_context_rates() {
    let long_context = PricingCategory {
        standard: Some(tier(Some(1.0), Some(5.0))),
        batch: Some(tier(Some(0.5), Some(2.5))),
        long_context: Some(tier(Some(2.0), Some(8.0))),
        long_context_threshold: Some(1_000),
    };
    let model = model_with("test-model", "anthropic", long_context);

    let cost = batch_cost(Provider::Anthropic, &tokens(2_000, 1_000), &model);

    assert_eq!(cost.input, Some(0.002));
    assert_eq!(cost.output, Some(0.004));
    assert_eq!(cost.total(), Some(0.006));
}

// ---- provider_spec.rb protocol resolution + openai_batches_spec.rb --------------------------------

/// An OpenAI batch output file line answering `custom_id` with a Responses body.
fn responses_line(custom_id: &str, text: &str) -> Value {
    json!({ "custom_id": custom_id, "response": { "status_code": 200, "body": {
        "id": "resp_1", "object": "response", "status": "completed", "model": "gpt-5-nano",
        "output": [{ "type": "message", "role": "assistant", "content": [{ "type": "output_text", "text": text }] }],
        "usage": { "input_tokens": 1, "output_tokens": 1 } } } })
}

/// An OpenAI batch output file line answering `custom_id` with a Chat Completions body.
fn chat_completions_line(custom_id: &str, text: &str) -> Value {
    json!({ "custom_id": custom_id, "response": { "status_code": 200, "body": {
        "id": "chatcmpl_1", "object": "chat.completion", "model": "gpt-5-nano",
        "choices": [{ "index": 0, "message": { "role": "assistant", "content": text }, "finish_reason": "stop" }],
        "usage": { "prompt_tokens": 1, "completion_tokens": 1 } } } })
}

/// A completed OpenAI batch `batch_1` whose stored `endpoint` is `endpoint`, with one output line.
async fn stored_openai_batch(endpoint: Option<&str>, line: Value) -> MockServer {
    let server = MockServer::start().await;
    let mut batch = json!({ "id": "batch_1", "status": "completed", "output_file_id": "file-out", "request_counts": { "total": 1 } });
    if let Some(endpoint) = endpoint {
        batch["endpoint"] = endpoint.into();
    }
    get(&server, "/v1/batches/batch_1", batch).await;
    Mock::given(method("GET"))
        .and(path("/v1/files/file-out/content"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(format!("{line}\n"), "application/octet-stream"),
        )
        .mount(&server)
        .await;
    server
}

// spec: provider_spec.rb:615 protocol resolution > uses Responses as the default batch protocol
#[tokio::test]
async fn uses_responses_as_the_default_batch_protocol() {
    let server = stored_openai_batch(None, responses_line("0", "From Responses")).await;
    let mut config = (*config(&server)).clone();
    config.set("openai_protocol", "chat_completions");

    let mut batch = Batch::find_with_config(Arc::new(config), "batch_1", Some("openai"))
        .await
        .unwrap();

    // A batch with no endpoint on record reads its results as Responses, not the chat default.
    assert_eq!(
        content(&batch.messages().await.unwrap()[0]),
        "From Responses"
    );
}

// spec: provider_spec.rb:625 protocol resolution > routes OpenAI batches by rendered payload shape
#[tokio::test]
async fn routes_openai_batches_by_rendered_payload_shape() {
    assert_eq!(
        openai_batch_protocol_name_for(&json!({ "input": [{ "role": "user", "content": "hi" }] }))
            .unwrap(),
        "responses"
    );
    assert_eq!(
        openai_batch_protocol_name_for(&json!({ "messages": [] })).unwrap(),
        "chat_completions"
    );
    assert_eq!(
        openai_batch_protocol_name_for(
            &json!({ "model": "text-embedding-3-small", "input": "hi" })
        )
        .unwrap(),
        "embeddings"
    );

    // A Responses chat and a Chat Completions chat in one submission.
    let server = MockServer::start().await;
    let config = config(&server);
    let mut responses =
        Chat::with_config(config.clone(), Some("gpt-5-nano"), Some("openai"), false).unwrap();
    responses.ask_later("hi").unwrap();
    let mut chat_completions = Chat::with_config(config, Some("gpt-5-nano"), Some("openai"), false)
        .unwrap()
        .with_protocol(ProtocolName::ChatCompletions);
    chat_completions.ask_later("hi").unwrap();

    let err = rust_llm::batch(vec![responses, chat_completions])
        .await
        .unwrap_err();

    assert!(err.to_string().contains("one endpoint"), "{err}");
    assert!(
        server
            .received_requests()
            .await
            .unwrap_or_default()
            .is_empty()
    );
}

// spec: providers/openai_batches_spec.rb:16 #batch_protocol_name_for > reads a Chat Completions payload
#[test]
fn reads_a_chat_completions_payload() {
    assert_eq!(
        openai_batch_protocol_name_for(&json!({ "messages": [] })).unwrap(),
        "chat_completions"
    );
}

// spec: providers/openai_batches_spec.rb:28 #batch_protocol_name_for > refuses a payload it cannot route
#[test]
fn refuses_a_payload_it_cannot_route() {
    let err = openai_batch_protocol_name_for(&json!({ "prompt": "hi" })).unwrap_err();
    assert!(matches!(&err, Error::Api(..)), "{err:?}");
    assert_eq!(
        err.to_string(),
        "openai batch requests only support chat, responses, or embedding payloads"
    );
}

// spec: providers/openai_batches_spec.rb:36 #batch_protocol_for_endpoint > routes both spellings of each batch endpoint
#[test]
fn routes_both_spellings_of_each_batch_endpoint() {
    for (endpoint, protocol) in [
        ("/v1/responses", "responses"),
        ("responses", "responses"),
        ("/v1/chat/completions", "chat_completions"),
        ("chat/completions", "chat_completions"),
        ("/v1/embeddings", "embeddings"),
        ("embeddings", "embeddings"),
    ] {
        assert_eq!(
            openai_batch_protocol_for_endpoint(Some(endpoint)),
            Some(protocol),
            "{endpoint}"
        );
    }
}

// spec: providers/openai_batches_spec.rb:49 #batch_protocol_for_endpoint > is nil for an endpoint it does not know
#[test]
fn is_none_for_an_endpoint_it_does_not_know() {
    assert_eq!(
        openai_batch_protocol_for_endpoint(Some("/v1/moderations")),
        None
    );
    assert_eq!(openai_batch_protocol_for_endpoint(None), None);
}

// spec: providers/openai_batches_spec.rb:56 #batch_protocol_for_stored_batch > falls back to the default protocol when the endpoint is unknown
#[tokio::test]
async fn falls_back_to_the_default_protocol_when_the_endpoint_is_unknown() {
    let server = stored_openai_batch(
        Some("/v1/moderations"),
        responses_line("0", "Read as Responses"),
    )
    .await;

    let mut batch = Batch::find_with_config(config(&server), "batch_1", Some("openai"))
        .await
        .unwrap();

    assert_eq!(batch.batch_protocol(), None);
    assert_eq!(
        content(&batch.messages().await.unwrap()[0]),
        "Read as Responses"
    );
}

// spec: providers/openai_batches_spec.rb:66 #batch_protocol_for_stored_batch > reads the protocol out of the stored endpoint
#[tokio::test]
async fn reads_the_protocol_out_of_the_stored_endpoint() {
    let server = stored_openai_batch(
        Some("/v1/chat/completions"),
        chat_completions_line("0", "Read as Chat Completions"),
    )
    .await;

    let mut batch = Batch::find_with_config(config(&server), "batch_1", Some("openai"))
        .await
        .unwrap();

    assert_eq!(batch.batch_protocol(), Some("chat_completions"));
    assert_eq!(
        content(&batch.messages().await.unwrap()[0]),
        "Read as Chat Completions"
    );
}

// ---- providers/mistral/chat_completions/batches_spec.rb -----------------------------------------

fn mistral_options(config: &Arc<Config>) -> EmbedOptions<'static> {
    EmbedOptions {
        model: Some("mistral-embed"),
        provider: Some("mistral"),
        config: Some(config.clone()),
        ..Default::default()
    }
}

// spec: providers/mistral/chat_completions/batches_spec.rb:32 #create_batch > sends embedding jobs to the embeddings endpoint
#[tokio::test]
async fn sends_embedding_jobs_to_the_embeddings_endpoint() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/batch/jobs"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({ "id": "job_123", "status": "QUEUED" })),
        )
        .mount(&server)
        .await;
    let config = config(&server);
    let request = embed_later(vec!["Ruby".to_string()], mistral_options(&config)).unwrap();

    let batch = rust_llm::batch(request).await.unwrap();

    assert_eq!(batch.id(), "job_123");
    assert!(!batch.is_complete());
    let posts = received(&server, "POST", "/v1/batch/jobs").await;
    assert_eq!(posts.len(), 1);
    let body: Value = serde_json::from_slice(&posts[0].body).unwrap();
    assert_eq!(
        body,
        json!({ "endpoint": "/v1/embeddings", "model": "mistral-embed",
                "requests": [{ "custom_id": "0:array", "body": { "input": ["Ruby"] } }] })
    );
}

// spec: providers/mistral/chat_completions/batches_spec.rb:50 #create_batch > rejects mixed chat and embedding requests
#[test]
fn rejects_mixed_chat_and_embedding_requests() {
    let err = mistral_batch_endpoint(&[json!({ "input": "Ruby" }), json!({ "messages": [] })])
        .unwrap_err();
    assert!(matches!(&err, Error::Api(..)), "{err:?}");
    assert!(err.to_string().contains("cannot mix"), "{err}");
}

// spec: providers/mistral/chat_completions/batches_spec.rb:60 #create_batch > rejects mixed-model jobs
#[tokio::test]
async fn rejects_mixed_model_mistral_jobs() {
    let server = MockServer::start().await;
    let config = config(&server);
    let chats = ["mistral-small-latest", "mistral-large-latest"].map(|model| {
        let mut chat =
            Chat::with_config(config.clone(), Some(model), Some("mistral"), false).unwrap();
        chat.ask_later("Hi").unwrap();
        chat
    });

    let err = rust_llm::batch(Vec::from(chats)).await.unwrap_err();

    assert!(matches!(&err, Error::Api(..)), "{err:?}");
    assert!(err.to_string().contains("one model"), "{err}");
    assert!(
        server
            .received_requests()
            .await
            .unwrap_or_default()
            .is_empty()
    );
}

// spec: providers/mistral/chat_completions/batches_spec.rb:96 #parse_batch_response > leaves cancellation requests running
#[tokio::test]
async fn leaves_cancellation_requests_running() {
    let server = MockServer::start().await;
    get(
        &server,
        "/v1/batch/jobs/job_123",
        json!({ "id": "job_123", "status": "CANCELLATION_REQUESTED" }),
    )
    .await;

    let batch = Batch::find_with_config(config(&server), "job_123", Some("mistral"))
        .await
        .unwrap();

    assert!(!batch.is_complete());
    assert_eq!(batch.raw_status(), Some("CANCELLATION_REQUESTED"));
}

async fn mistral_job_responses(server: &MockServer, responses: Vec<ResponseTemplate>) {
    for (i, response) in responses.into_iter().enumerate() {
        Mock::given(method("GET"))
            .and(path("/v1/batch/jobs/job_123"))
            .respond_with(response)
            .up_to_n_times(1)
            .with_priority(1 + i as u8)
            .mount(server)
            .await;
    }
}

fn missing_job() -> ResponseTemplate {
    ResponseTemplate::new(404).set_body_json(json!({ "detail": "No batch job matches" }))
}

// spec: providers/mistral/chat_completions/batches_spec.rb:110 #find_batch > retries an initial missing job while a new submission becomes visible
#[tokio::test]
async fn retries_an_initial_missing_job_while_a_new_submission_becomes_visible() {
    let server = MockServer::start().await;
    let found =
        ResponseTemplate::new(200).set_body_json(json!({ "id": "job_123", "status": "QUEUED" }));
    mistral_job_responses(&server, vec![missing_job(), found]).await;
    let started = Instant::now();

    let batch = Batch::find_with_config(config(&server), "job_123", Some("mistral"))
        .await
        .unwrap();

    assert_eq!(batch.id(), "job_123");
    assert!(!batch.is_complete());
    assert_eq!(
        received(&server, "GET", "/v1/batch/jobs/job_123")
            .await
            .len(),
        2
    );
    // `sleep(0.5)` before the one retry.
    let waited = started.elapsed();
    assert!(
        waited >= Duration::from_millis(500) && waited < Duration::from_millis(1_500),
        "{waited:?}"
    );
}

// spec: providers/mistral/chat_completions/batches_spec.rb:126 #find_batch > raises when the job remains missing after bounded retries
#[tokio::test]
async fn raises_when_the_job_remains_missing_after_bounded_retries() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/batch/jobs/job_123"))
        .respond_with(missing_job())
        .mount(&server)
        .await;

    let err = Batch::find_with_config(config(&server), "job_123", Some("mistral"))
        .await
        .unwrap_err();

    assert_eq!(err.response().map(|r| r.status), Some(404));
    assert_eq!(
        received(&server, "GET", "/v1/batch/jobs/job_123")
            .await
            .len(),
        3
    );
}

// spec: providers/mistral/chat_completions/batches_spec.rb:134 #find_batch > does not retry permission errors
#[tokio::test]
async fn does_not_retry_permission_errors() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/batch/jobs/job_123"))
        .respond_with(ResponseTemplate::new(403).set_body_json(json!({ "detail": "Forbidden" })))
        .mount(&server)
        .await;
    let started = Instant::now();

    let err = Batch::find_with_config(config(&server), "job_123", Some("mistral"))
        .await
        .unwrap_err();

    assert!(matches!(&err, Error::Forbidden(..)), "{err:?}");
    assert_eq!(
        received(&server, "GET", "/v1/batch/jobs/job_123")
            .await
            .len(),
        1
    );
    assert!(
        started.elapsed() < Duration::from_millis(500),
        "no sleep before giving up"
    );
}

// spec: providers/mistral/chat_completions/batches_spec.rb:144 #parse_batch_result > preserves scalar and one-element array embedding shapes after reloading a job
#[tokio::test]
async fn preserves_scalar_and_one_element_array_embedding_shapes_after_reloading_a_job() {
    let server = MockServer::start().await;
    let body = json!({ "model": "mistral-embed", "data": [{ "embedding": [0.1, 0.2] }], "usage": { "prompt_tokens": 3 } });
    let job = json!({ "id": "job_123", "status": "SUCCESS", "total_requests": 2, "outputs": [
        { "custom_id": "0", "response": { "body": body } },
        { "custom_id": "1:array", "response": { "body": body } }
    ] });
    get(&server, "/v1/batch/jobs/job_123", job).await;

    // Reloaded by id: the job's own custom ids are all that say which input was an array.
    let mut batch = Batch::find_with_config(config(&server), "job_123", Some("mistral"))
        .await
        .unwrap();
    let results = batch.results().await.unwrap();

    let scalar = results[0].as_ref().and_then(|r| r.as_embedding()).unwrap();
    let array = results[1].as_ref().and_then(|r| r.as_embedding()).unwrap();
    assert_eq!(scalar.vectors, Vectors::Single(vec![0.1, 0.2]));
    assert_eq!(array.vectors, Vectors::Batch(vec![vec![0.1, 0.2]]));
    assert_eq!(array.tokens().input, Some(3));
}

// spec: providers/mistral/chat_completions/batches_spec.rb:158 #parse_batch_result > keeps a failed embedding request in its original slot
#[tokio::test]
async fn keeps_a_failed_embedding_request_in_its_original_slot() {
    let server = MockServer::start().await;
    // Still running, so only the reported row gets a status.
    let job = json!({ "id": "job_123", "status": "RUNNING", "total_requests": 3, "outputs": [
        { "custom_id": "2:array", "error": { "message": "Invalid input" } }
    ] });
    get(&server, "/v1/batch/jobs/job_123", job).await;

    let mut batch = Batch::find_with_config(config(&server), "job_123", Some("mistral"))
        .await
        .unwrap();
    let results = batch.results().await.unwrap();

    assert!(results.iter().all(Option::is_none));
    assert_eq!(batch.statuses(), &[None, None, Some(BatchStatus::Failed)]);
}

// spec: providers/mistral/chat_completions/batches_spec.rb:184 submits, reloads, and cancels an embedding batch
#[tokio::test]
async fn submits_reloads_and_cancels_an_embedding_batch() {
    let cassette = Cassette::start(
        "providers_mistral_chatcompletions_batches_submits_reloads_and_cancels_an_embedding_batch",
    )
    .await
    .expect("cassette");
    let config = config_for(&cassette, "mistral");
    let requests = vec![
        embed_later("Ruby", mistral_options(&config)).unwrap(),
        embed_later(vec!["Rails".to_string()], mistral_options(&config)).unwrap(),
    ];
    let mut batch = rust_llm::batch(requests).await.expect("submit");

    let found = Batch::find_with_config(config, batch.id(), Some("mistral"))
        .await
        .expect("find");

    assert_eq!(found.id(), batch.id());
    assert_eq!(
        found.request_counts().and_then(|c| c.get("total")),
        Some(&json!(2))
    );
    if !batch.is_complete() {
        batch.cancel().await.expect("cancel");
    }
    cassette.assert_all_matched().await;
}

// ---- providers/xai/chat_completions/batches_spec.rb ---------------------------------------------

async fn xai_batch(server: &MockServer, state: Value) {
    get(
        server,
        "/v1/batches/batch_1",
        json!({ "batch_id": "batch_1", "state": state }),
    )
    .await;
}

// spec: providers/xai/chat_completions/batches_spec.rb:43 #xai_batch_request > keeps the model per request so mixed-model batches remain request-scoped
#[tokio::test]
async fn keeps_the_model_per_request_so_mixed_model_batches_remain_request_scoped() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/batches"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "batch_id": "batch_1" })))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/batches/batch_1/requests"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({})))
        .mount(&server)
        .await;
    xai_batch(&server, json!({ "num_requests": 2, "num_pending": 2 })).await;
    let config = config(&server);
    let chats = ["grok-4.3", "grok-4.1"].map(|model| {
        let mut chat = Chat::with_config(config.clone(), Some(model), Some("xai"), true)
            .unwrap()
            .with_protocol(ProtocolName::ChatCompletions);
        chat.ask_later("Hi").unwrap();
        chat
    });

    rust_llm::batch(Vec::from(chats)).await.unwrap();

    let posts = received(&server, "POST", "/v1/batches/batch_1/requests").await;
    let body: Value = serde_json::from_slice(&posts[0].body).unwrap();
    let models: Vec<&str> = body["batch_requests"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            r.pointer("/batch_request/chat_get_completion/model")
                .and_then(Value::as_str)
                .unwrap_or_default()
        })
        .collect();
    assert_eq!(models, ["grok-4.3", "grok-4.1"]);
}

// spec: providers/xai/chat_completions/batches_spec.rb:122 #parse_batch_response state shapes > reports a state carrying an error as failed and still running
#[tokio::test]
async fn reports_a_state_carrying_an_error_as_failed_and_still_running() {
    let server = MockServer::start().await;
    get(&server, "/v1/batches/batch_1", json!({ "id": "batch_1", "state": { "num_requests": 2, "num_pending": 1, "error": "boom" } })).await;

    let batch = Batch::find_with_config(config(&server), "batch_1", Some("xai"))
        .await
        .unwrap();

    assert_eq!(batch.raw_status(), Some("failed"));
    assert!(!batch.is_complete());
}

// spec: providers/xai/chat_completions/batches_spec.rb:128 #parse_batch_response state shapes > falls back to a plain status field
#[tokio::test]
async fn falls_back_to_a_plain_status_field() {
    let server = MockServer::start().await;
    get(
        &server,
        "/v1/batches/batch_1",
        json!({ "id": "batch_1", "status": "queued" }),
    )
    .await;

    let batch = Batch::find_with_config(config(&server), "batch_1", Some("xai"))
        .await
        .unwrap();

    assert_eq!(batch.raw_status(), Some("queued"));
    assert!(!batch.is_complete());
}

/// A completed xAI batch of `count` requests whose results page is `results`.
async fn finished_xai_batch(count: i64, results: Value) -> MockServer {
    let server = MockServer::start().await;
    xai_batch(&server, json!({ "num_requests": count, "num_pending": 0 })).await;
    get(
        &server,
        "/v1/batches/batch_1/results",
        json!({ "results": results }),
    )
    .await;
    server
}

// spec: providers/xai/chat_completions/batches_spec.rb:136 #parse_batch_result response shapes > reads a result nested directly under response
#[tokio::test]
async fn reads_a_result_nested_directly_under_response() {
    let server = finished_xai_batch(4, json!([{ "custom_id": "3", "response": { "chat_get_completion": {
        "model": "grok-4.3", "choices": [{ "message": { "role": "assistant", "content": "Hi" } }] } } }]))
    .await;

    let mut batch = Batch::find_with_config(config(&server), "batch_1", Some("xai"))
        .await
        .unwrap();
    let messages = batch.messages().await.unwrap();

    assert_eq!(content(&messages[3]), "Hi");
    assert_eq!(batch.statuses()[3], Some(BatchStatus::Succeeded));
}

// spec: providers/xai/chat_completions/batches_spec.rb:153 #parse_batch_result response shapes > warns and returns no message for a failed row
// The row goes through `batch_failure`, whose warning text is asserted in spec_batch.rs
// (batch_helpers_spec.rb:185); asserting it here too raced with parallel tests' tracing state.
#[tokio::test]
async fn warns_and_returns_no_message_for_a_failed_row() {
    let server = finished_xai_batch(
        5,
        json!([{ "batch_request_id": "4", "error": "rate limited" }]),
    )
    .await;
    let mut batch = Batch::find_with_config(config(&server), "batch_1", Some("xai"))
        .await
        .unwrap();

    let messages = batch.messages().await.unwrap();

    assert!(messages[4].is_none());
    assert_eq!(batch.statuses()[4], Some(BatchStatus::Failed));
    assert_eq!(
        rust_llm::batch::batch_error_message(
            &json!({ "batch_request_id": "4", "error": "rate limited" })
        )
        .as_deref(),
        Some("rate limited"),
        "the failure detail the warning reports"
    );
}
