//! `spec/ruby_llm/batch_spec.rb`: provider batches, replayed from RubyLLM's `batch_*` cassettes,
//! plus the spec's stubbed-provider examples served by a local mock provider.

mod support;

use std::sync::Arc;

use rust_llm::batch::batch_cost;
use rust_llm::cost::Tier;
use rust_llm::model::{Pricing, PricingCategory, PricingTier};
use rust_llm::{Batch, BatchStatus, Chat, Config, Cost, EmbedOptions, Error, Message, Model, Provider, Role, Tokens, embed_later};
use serde_json::{Value, json};
use support::{Cassette, config_for};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// `RubyLLM.chat(model:, provider:).ask_later(text)`. The Anthropic examples pass no provider, so
/// `claude-haiku-4-5` stays the bare registry id rather than its dated alias.
fn staged(config: &Arc<Config>, model: &str, provider: Option<&str>, text: &str) -> Chat {
    let mut chat = Chat::with_config(config.clone(), Some(model), provider, false).expect("chat");
    chat.ask_later(text).expect("ask_later");
    chat
}

/// `wait_for(batch)`: poll with `refresh` until complete. Replay needs no sleeping.
async fn wait_for(batch: &mut Batch) {
    for _ in 0..120 {
        if batch.refresh().await.expect("refresh").is_complete() {
            return;
        }
    }
    panic!("batch never completed");
}

/// Like `Cassette::assert_all_matched`, but tolerates the fields RubyLLM fills with
/// `SecureRandom.hex` (Gemini's `displayName`, xAI's batch `name`).
async fn assert_matched(cassette: &Cassette, random: &[&str]) {
    let mismatches: Vec<String> = cassette
        .mismatches
        .lock()
        .unwrap()
        .iter()
        .filter(|m| !random.iter().any(|r| m.contains(r)))
        .cloned()
        .collect();
    assert!(mismatches.is_empty(), "request bodies differ from RubyLLM's:\n  {}", mismatches.join("\n  "));
    let received = cassette.server.received_requests().await.unwrap_or_default().len();
    assert_eq!(received, cassette.count, "expected {} requests like RubyLLM made, sent {received}", cassette.count);
}

fn content(message: &Option<Message>) -> String {
    message.as_ref().map(|m| m.content().to_string()).unwrap_or_default()
}

fn jsonl_lines(body: &str) -> Vec<Value> {
    body.lines().filter(|l| l.starts_with("{\"custom_id\"")).map(|l| serde_json::from_str(l).expect("jsonl line")).collect()
}

/// The JSONL file upload is multipart, so it can't be JSON-compared: check the form fields and
/// that each JSONL line is JSON-equal to the one RubyLLM uploaded.
async fn assert_uploaded_jsonl_matches(cassette: &Cassette, name: &str) {
    let requests = cassette.server.received_requests().await.unwrap_or_default();
    let upload = &requests[0];
    assert_eq!(upload.method.as_str(), "POST");
    assert_eq!(upload.url.path(), "/v1/files");
    let content_type = upload.headers.get("content-type").and_then(|v| v.to_str().ok()).unwrap_or_default();
    assert!(content_type.starts_with("multipart/form-data; boundary="), "{content_type}");
    let body = String::from_utf8_lossy(&upload.body);
    assert!(body.contains("name=\"file\"; filename=\"ruby_llm_batch.jsonl\""), "{body}");
    assert!(body.contains("name=\"purpose\"\r\n\r\nbatch\r\n"), "{body}");
    let recorded = support::load(name).expect("cassette");
    assert_eq!(jsonl_lines(&body), jsonl_lines(&recorded[0].request_body));
}

// ---- cassettes ----------------------------------------------------------------------------------

#[tokio::test]
async fn anthropic_answers_staged_chats_and_appends_the_answers_to_their_conversations() {
    let cassette = Cassette::start("batch_with_anthropic_claude-haiku-4-5_answers_staged_chats_and_appends_the_answers_to_their_conversations")
        .await
        .expect("cassette");
    let config = config_for(&cassette, "anthropic");
    let mut first = Chat::with_config(config.clone(), Some("claude-haiku-4-5"), None, false).unwrap().with_instructions("Be terse.");
    first.ask_later("What is 2 + 2?").unwrap();
    let chats = vec![first, staged(&config, "claude-haiku-4-5", None, "Name the largest planet in our solar system. One word.")];

    let mut batch = rust_llm::batch(chats).await.expect("submit");

    assert!(batch.id().starts_with("msgbatch_"));
    assert_eq!(batch.status(), BatchStatus::Pending);
    assert_eq!(batch.raw_status(), Some("in_progress"));

    wait_for(&mut batch).await;

    assert!(batch.is_complete());
    assert!(batch.is_succeeded());
    let messages = batch.messages().await.expect("messages");
    assert!(content(&messages[0]).contains('4'));
    assert!(content(&messages[1]).to_lowercase().contains("jupiter"));
    assert!(messages[0].as_ref().unwrap().tokens().input.unwrap() > 0);
    let roles: Vec<Role> = batch.chats().unwrap()[0].messages().iter().map(|m| m.role).collect();
    assert_eq!(roles, vec![Role::System, Role::User, Role::Assistant]);
    assert_matched(&cassette, &[]).await;
}

#[tokio::test]
async fn anthropic_reloads_a_batch_by_id_and_collects_messages_without_the_chats() {
    let cassette = Cassette::start("batch_with_anthropic_claude-haiku-4-5_reloads_a_batch_by_id_and_collects_messages_without_the_chats")
        .await
        .expect("cassette");
    let config = config_for(&cassette, "anthropic");
    let mut submitted = rust_llm::batch(staged(&config, "claude-haiku-4-5", None, "What is 3 + 3? Just the number."))
        .await
        .expect("submit");
    wait_for(&mut submitted).await;

    let mut batch = Batch::find_with_config(config, submitted.id(), Some("anthropic")).await.expect("find");

    assert!(batch.is_complete());
    assert!(batch.chats().is_none());
    assert!(content(&batch.messages().await.unwrap()[0]).contains('6'));
    assert_matched(&cassette, &[]).await;
}

#[tokio::test]
async fn anthropic_cancels_a_running_batch() {
    let cassette = Cassette::start("batch_with_anthropic_claude-haiku-4-5_cancels_a_running_batch").await.expect("cassette");
    let config = config_for(&cassette, "anthropic");
    let mut batch = rust_llm::batch(vec![staged(&config, "claude-haiku-4-5", None, "What is 5 + 5?")]).await.expect("submit");

    batch.cancel().await.expect("cancel");

    assert!(matches!(batch.status(), BatchStatus::Pending | BatchStatus::Succeeded));
    assert!(matches!(batch.raw_status(), Some("canceling" | "ended")));
    assert_matched(&cassette, &[]).await;
}

/// `with #{provider}/#{model}`: gemini, mistral, openai, and xai answer two staged chats.
#[tokio::test]
async fn providers_answer_staged_chats_and_append_the_answers_to_their_conversations() {
    let cases: &[(&str, &str, &str, &[&str])] = &[
        ("gemini", "gemini-2.5-flash", "gemini-2_5-flash", &["/batch/displayName"]),
        ("mistral", "mistral-small-latest", "mistral-small-latest", &[]),
        ("openai", "gpt-5-nano", "gpt-5-nano", &[]),
        ("xai", "grok-4-1-fast-non-reasoning", "grok-4-1-fast-non-reasoning", &["request 0: /name"]),
    ];
    for (provider, model, slug, random) in cases {
        let name = format!("batch_with_{provider}_{slug}_answers_staged_chats_and_appends_the_answers_to_their_conversations");
        let cassette = Cassette::start(&name).await.expect("cassette");
        let config = config_for(&cassette, provider);
        let chats = vec![
            staged(&config, model, Some(provider), "What is 2 + 2? Just the number."),
            staged(&config, model, Some(provider), "Name the largest planet in our solar system. One word."),
        ];

        let mut batch = rust_llm::batch(chats).await.unwrap_or_else(|e| panic!("{provider}: {e}"));
        assert!(!batch.id().is_empty(), "{provider}");

        wait_for(&mut batch).await;

        assert!(batch.is_complete(), "{provider}");
        let messages = batch.messages().await.unwrap_or_else(|e| panic!("{provider}: {e}"));
        assert!(content(&messages[0]).contains('4'), "{provider}: {:?}", content(&messages[0]));
        assert!(content(&messages[1]).to_lowercase().contains("jupiter"), "{provider}");
        let roles: Vec<Role> = batch.chats().unwrap()[1].messages().iter().map(|m| m.role).collect();
        assert_eq!(roles, vec![Role::User, Role::Assistant], "{provider}");
        assert_eq!(batch.statuses(), &[Some(BatchStatus::Succeeded), Some(BatchStatus::Succeeded)], "{provider}");
        assert_matched(&cassette, random).await;
        if *provider == "openai" {
            assert_eq!(batch.batch_protocol(), Some("responses"));
            assert_uploaded_jsonl_matches(&cassette, &name).await;
        }
    }
}

#[tokio::test]
async fn openai_embeds_staged_texts_and_hydrates_each_request_result() {
    let name = "batch_with_openai_text-embedding-3-small_embeddings_embeds_staged_texts_and_hydrates_each_request_result";
    let cassette = Cassette::start(name).await.expect("cassette");
    let config = config_for(&cassette, "openai");
    let options = |dimensions| EmbedOptions {
        model: Some("text-embedding-3-small"),
        dimensions,
        config: Some(config.clone()),
        ..Default::default()
    };
    let requests = vec![
        embed_later("Ruby is a programmer best friend", options(None)).unwrap(),
        embed_later("Batches come back within a day", options(Some(256))).unwrap(),
    ];

    let mut batch = rust_llm::batch(requests).await.expect("submit");
    assert!(!batch.id().is_empty());

    wait_for(&mut batch).await;

    assert!(batch.is_complete());
    let results = batch.results().await.expect("results");
    let first = results[0].as_ref().and_then(|r| r.as_embedding()).expect("first");
    let second = results[1].as_ref().and_then(|r| r.as_embedding()).expect("second");
    assert!(matches!(&first.vectors, rust_llm::Vectors::Single(v) if v.len() == 1536));
    assert!(matches!(&second.vectors, rust_llm::Vectors::Single(v) if v.len() == 256));
    let hydrated = batch.requests().unwrap()[0].result.as_ref().expect("hydrated");
    assert_eq!(hydrated.vectors, first.vectors);
    assert!(hydrated.tokens().input.unwrap() > 0);
    assert_eq!(batch.batch_protocol(), Some("embeddings"));
    assert_matched(&cassette, &[]).await;
    assert_uploaded_jsonl_matches(&cassette, name).await;
}

// ---- .submit ------------------------------------------------------------------------------------

fn offline_config() -> Arc<Config> {
    let mut config = Config::default();
    for provider in ["anthropic", "openai", "deepseek", "mistral"] {
        config.set(format!("{provider}_api_key"), "test-key");
        config.set(format!("{provider}_api_base"), "http://127.0.0.1:9");
    }
    config.max_retries = 0;
    Arc::new(config)
}

#[tokio::test]
async fn rejects_an_empty_batch() {
    let err = rust_llm::batch(Vec::<Chat>::new()).await.unwrap_err();
    assert!(matches!(&err, Error::Argument(m) if m.contains("empty batch")), "{err}");
}

#[tokio::test]
async fn rejects_chats_that_are_not_awaiting_the_model() {
    let chat = Chat::with_config(offline_config(), Some("claude-haiku-4-5"), None, false).unwrap();
    let err = rust_llm::batch(vec![chat]).await.unwrap_err();
    assert!(matches!(&err, Error::Argument(m) if m.contains("awaiting the model")), "{err}");
}

#[tokio::test]
async fn rejects_mixed_providers() {
    let config = offline_config();
    let chats = vec![staged(&config, "claude-haiku-4-5", None, "Hi"), staged(&config, "gpt-5-nano", Some("openai"), "Hi")];
    let err = rust_llm::batch(chats).await.unwrap_err();
    assert!(matches!(&err, Error::Argument(m) if m.contains("one provider")), "{err}");
}

#[tokio::test]
async fn rejects_mixed_models_for_model_scoped_providers() {
    let config = offline_config();
    let chats = vec![staged(&config, "gpt-5-nano", Some("openai"), "Hi"), staged(&config, "gpt-5-mini", Some("openai"), "Hi")];
    let err = rust_llm::batch(chats).await.unwrap_err();
    assert!(err.to_string().contains("one model"), "{err}");
}

#[tokio::test]
async fn rejects_providers_without_batch_support() {
    let mut chat = Chat::with_config(offline_config(), Some("deepseek-v4-flash"), Some("deepseek"), true).unwrap();
    chat.ask_later("Hi").unwrap();
    let err = rust_llm::batch(chat).await.unwrap_err();
    assert!(err.to_string().contains("doesn't support batch requests"), "{err}");
}

#[tokio::test]
async fn find_requires_a_provider() {
    let err = Batch::find("msgbatch_123", None).await.unwrap_err();
    assert!(matches!(&err, Error::Argument(m) if m.contains("Provider")), "{err}");
}

// ---- #messages against a mock provider ----------------------------------------------------------

/// An Anthropic batch that has ended, whose results are `jsonl`.
async fn ended_anthropic_batch(jsonl: String) -> (MockServer, Arc<Config>) {
    let server = MockServer::start().await;
    let status = json!({ "id": "msgbatch_test", "processing_status": "ended",
        "request_counts": { "processing": 0, "succeeded": 1, "errored": 1, "canceled": 0, "expired": 0 } });
    Mock::given(method("GET")).and(path("/v1/messages/batches/msgbatch_test")).respond_with(ResponseTemplate::new(200).set_body_json(status)).mount(&server).await;
    Mock::given(method("GET"))
        .and(path("/v1/messages/batches/msgbatch_test/results"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(jsonl, "application/x-jsonl"))
        .mount(&server)
        .await;
    let mut config = Config::default();
    config.set("anthropic_api_key", "test-key");
    config.set("anthropic_api_base", server.uri());
    config.max_retries = 0;
    (server, Arc::new(config))
}

fn anthropic_success(custom_id: &str, content: Value, input: i64, output: i64) -> String {
    json!({ "custom_id": custom_id, "result": { "type": "succeeded", "message": {
        "model": "claude-haiku-4-5", "id": "msg_1", "type": "message", "role": "assistant",
        "content": content, "stop_reason": "end_turn",
        "usage": { "input_tokens": input, "output_tokens": output } } } })
    .to_string()
}

async fn results_requests(server: &MockServer) -> usize {
    server.received_requests().await.unwrap_or_default().iter().filter(|r| r.url.path().ends_with("/results")).count()
}

#[tokio::test]
async fn leaves_failed_slots_empty_and_their_chats_awaiting_a_response() {
    let failed = json!({ "custom_id": "0", "result": { "type": "errored", "error": { "error": { "message": "overloaded" } } } }).to_string();
    let jsonl = format!("{failed}\n{}\n", anthropic_success("1", json!([{ "type": "text", "text": "4" }]), 1, 1));
    let (server, config) = ended_anthropic_batch(jsonl).await;
    let chats = vec![
        staged(&config, "claude-haiku-4-5", None, "This one fails"),
        staged(&config, "claude-haiku-4-5", None, "This one succeeds"),
    ];

    let mut batch = Batch::find_with_config(config, "msgbatch_test", Some("anthropic")).await.unwrap().with_chats(chats);
    let messages = batch.messages().await.unwrap();

    assert!(messages[0].is_none());
    assert_eq!(content(&messages[1]), "4");
    assert_eq!(batch.statuses(), &[Some(BatchStatus::Failed), Some(BatchStatus::Succeeded)]);
    let tokens = batch.tokens().await.unwrap();
    assert_eq!((tokens.input, tokens.output), (Some(1), Some(1)));
    assert!(batch.cost().await.unwrap().total().is_some());
    let chats = batch.chats().unwrap();
    assert!(!chats[0].is_complete());
    assert_eq!(chats[1].messages().last().unwrap().content(), "4");

    batch.messages().await.unwrap();
    assert_eq!(results_requests(&server).await, 1, "results are cached once the batch is complete");
}

#[tokio::test]
async fn applies_the_provider_batch_discount_when_the_model_has_no_batch_tier() {
    let jsonl = anthropic_success("0", json!([{ "type": "text", "text": "Hello" }]), 1_000, 2_000);
    let (_server, config) = ended_anthropic_batch(jsonl).await;
    let chat = staged(&config, "claude-haiku-4-5", None, "Hi");
    let standard = Cost::new(&Tokens { input: Some(1_000), output: Some(2_000), ..Default::default() }, Some(chat.model()), Tier::Standard)
        .total()
        .unwrap();

    let mut batch = Batch::find_with_config(config, "msgbatch_test", Some("anthropic")).await.unwrap().with_chats(vec![chat]);
    let message = batch.messages().await.unwrap().remove(0).unwrap();

    // claude-haiku-4-5 lists $1/$5 per million and no batch tier: half of $0.011.
    assert!((message.cost(None).total().unwrap() - standard * 0.5).abs() < 1e-12);
    assert!((message.cost(None).total().unwrap() - 0.0055).abs() < 1e-12);
    let chat = &batch.chats().unwrap()[0];
    assert!((chat.cost().total().unwrap() - 0.0055).abs() < 1e-12, "the chat records the batch-priced usage");
    assert_eq!(chat.usage_entries(), message.usage_entries.as_slice());
    assert_eq!(batch.cost().await.unwrap().total(), message.cost(None).total());
}

#[tokio::test]
async fn delivers_each_answer_once_and_does_not_redeliver_a_tool_call_answer_after_its_tools_ran() {
    let tool_use = json!([{ "type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {} }]);
    let (_server, config) = ended_anthropic_batch(anthropic_success("0", tool_use, 1, 1)).await;
    let chat = staged(&config, "claude-haiku-4-5", None, "Look it up.");

    let collect = |chat: Chat| {
        let config = config.clone();
        async move {
            let mut batch = Batch::find_with_config(config, "msgbatch_test", Some("anthropic")).await.unwrap().with_chats(vec![chat]);
            batch.messages().await.unwrap();
            batch.into_chats().unwrap().remove(0)
        }
    };

    let mut chat = collect(chat).await; // first delivery appends the tool-call answer
    assert_eq!(chat.messages().iter().filter(|m| m.is_tool_call()).count(), 1);
    chat.add_message(Message::tool_result("toolu_1", "done")); // the app runs the tool
    let chat = collect(chat).await; // a redelivered poll re-collects the same batch

    assert_eq!(chat.messages().iter().filter(|m| m.is_tool_call()).count(), 1);
    assert_eq!(chat.messages().len(), 3);
}

#[tokio::test]
async fn does_not_append_a_plain_answer_that_is_already_in_the_chat() {
    let jsonl = anthropic_success("0", json!([{ "type": "text", "text": "Hello" }]), 1, 1);
    let (_server, config) = ended_anthropic_batch(jsonl).await;
    let chat = staged(&config, "claude-haiku-4-5", None, "Hi");

    let mut batch = Batch::find_with_config(config.clone(), "msgbatch_test", Some("anthropic")).await.unwrap().with_chats(vec![chat]);
    batch.messages().await.unwrap();
    let chat = batch.into_chats().unwrap().remove(0);
    let mut again = Batch::find_with_config(config, "msgbatch_test", Some("anthropic")).await.unwrap().with_chats(vec![chat]);
    let second = again.messages().await.unwrap().remove(0).unwrap();

    let chat = &again.chats().unwrap()[0];
    assert_eq!(chat.messages().len(), 2);
    assert_eq!(chat.usage_entries().len(), 1, "usage is recorded once");
    assert!(!second.usage_entries.is_empty(), "the re-collected answer is still priced");
}

#[tokio::test]
async fn hydrates_embeddings_into_their_staged_requests_and_leaves_failed_slots_empty() {
    let server = MockServer::start().await;
    let batch_json = json!({ "id": "batch_test", "status": "completed", "endpoint": "/v1/embeddings",
        "output_file_id": "file-out", "error_file_id": "", "request_counts": { "total": 2, "completed": 1, "failed": 1 } });
    Mock::given(method("POST")).and(path("/v1/files")).respond_with(ResponseTemplate::new(200).set_body_json(json!({ "id": "file-in" }))).mount(&server).await;
    Mock::given(method("POST")).and(path("/v1/batches")).respond_with(ResponseTemplate::new(200).set_body_json(batch_json.clone())).mount(&server).await;
    Mock::given(method("GET")).and(path("/v1/batches/batch_test")).respond_with(ResponseTemplate::new(200).set_body_json(batch_json)).mount(&server).await;
    let failed = json!({ "custom_id": "0", "response": { "status_code": 400, "body": { "error": { "message": "bad input" } } } });
    let ok = json!({ "custom_id": "1", "response": { "status_code": 200, "body": {
        "object": "list", "model": "text-embedding-3-small",
        "data": [{ "object": "embedding", "embedding": [0.1, 0.2] }], "usage": { "prompt_tokens": 3 } } } });
    Mock::given(method("GET"))
        .and(path("/v1/files/file-out/content"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(format!("{failed}\n{ok}\n"), "application/octet-stream"))
        .mount(&server)
        .await;
    let mut config = Config::default();
    config.set("openai_api_key", "test-key");
    config.set("openai_api_base", format!("{}/v1", server.uri()));
    let config = Arc::new(config);
    let options = || EmbedOptions { model: Some("text-embedding-3-small"), config: Some(config.clone()), ..Default::default() };
    let requests = vec![embed_later("This one fails", options()).unwrap(), embed_later("This one succeeds", options()).unwrap()];

    let mut batch = rust_llm::batch(requests).await.unwrap();
    let results = batch.results().await.unwrap();

    assert!(results[0].is_none());
    let embedding = results[1].as_ref().and_then(|r| r.as_embedding()).unwrap();
    assert_eq!(embedding.vectors, rust_llm::Vectors::Single(vec![0.1, 0.2]));
    assert_eq!(batch.statuses(), &[Some(BatchStatus::Failed), Some(BatchStatus::Succeeded)]);
    assert!(batch.requests().unwrap()[0].result.is_none());
    assert_eq!(batch.requests().unwrap()[1].result.as_ref().unwrap().vectors, embedding.vectors);
    assert_eq!(batch.tokens().await.unwrap().input, Some(3));
    assert_eq!(embedding.usage_entries.len(), 1);
    assert_eq!(embedding.usage_entries[0].operation, rust_llm::message::Operation::Embedding);
    // text-embedding-3-small is $0.02 per million input tokens; batches are half price.
    let standard = 3.0 * 0.02 / 1_000_000.0;
    assert!((embedding.cost().total().unwrap() - standard * 0.5).abs() < 1e-12);
    assert_eq!(batch.cost().await.unwrap().total(), embedding.cost().total());
}

// ---- Provider#batch_cost ------------------------------------------------------------------------

fn priced(batch: Option<PricingTier>) -> Model {
    let mut model = Model::default_for("priced-model", "anthropic");
    model.pricing = Pricing {
        text_tokens: Some(PricingCategory {
            standard: Some(PricingTier { input_per_million: Some(1.0), output_per_million: Some(5.0), ..Default::default() }),
            batch,
            ..Default::default()
        }),
        ..Default::default()
    };
    model
}

#[test]
fn batch_cost_uses_the_model_batch_tier_over_the_provider_discount() {
    let tokens = Tokens { input: Some(1_000), output: Some(2_000), ..Default::default() };
    let tier = PricingTier { input_per_million: Some(0.1), output_per_million: Some(1.0), ..Default::default() };
    let cost = batch_cost(Provider::Anthropic, &tokens, &priced(Some(tier)));
    // 1k * $0.1/M + 2k * $1/M, not half of the $0.011 standard price.
    assert!((cost.total().unwrap() - 0.0021).abs() < 1e-12);
}

#[test]
fn batch_cost_leaves_the_total_unknown_for_providers_without_a_discount_or_tier() {
    let tokens = Tokens { input: Some(1_000), output: Some(2_000), ..Default::default() };
    assert_eq!(batch_cost(Provider::XAI, &tokens, &priced(None)).total(), None);
}

#[test]
fn batch_cost_keeps_a_provider_reported_cost() {
    let tokens = Tokens { input: Some(1_000), output: Some(2_000), reported_cost: Some(0.42), ..Default::default() };
    assert_eq!(batch_cost(Provider::Anthropic, &tokens, &priced(None)).total(), Some(0.42));
}
