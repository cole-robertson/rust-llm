//! Port of `spec/ruby_llm/accounting/usage_owner_spec.rb`: `RubyLLM.with_usage_owner`, the
//! operation's `owner:` keyword, and the `owner` in each `usage.rust_llm` payload.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use rust_llm::accounting::{UsageOwner, usage_owner, with_current_owner, with_usage_owner};
use rust_llm::{Config, EmbedOptions, embed};
use serde_json::{Map, Value, json};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

type Events = Arc<Mutex<Vec<(String, Map<String, Value>)>>>;

/// `CaptureInstrumenter` in a context whose OpenAI embeddings answer with a canned vector.
async fn setup() -> (MockServer, Arc<Config>, Events) {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model": "text-embedding-3-small",
            "data": [{ "embedding": [0.1, 0.2] }],
            "usage": { "prompt_tokens": 3, "total_tokens": 3 }
        })))
        .mount(&server)
        .await;
    let events: Events = Arc::default();
    let sink = events.clone();
    let mut config = Config::default();
    config.set("openai_api_base", format!("{}/v1", server.uri()));
    config.set("openai_api_key", "test");
    config.max_retries = 0;
    config.instrumenter = Some(Arc::new(
        move |name: &str, payload: &Map<String, Value>, _: Option<Duration>| {
            sink.lock()
                .unwrap()
                .push((name.to_string(), payload.clone()));
        },
    ));
    (server, Arc::new(config), events)
}

async fn embed_with(config: &Arc<Config>, owner: Option<UsageOwner>) {
    embed(
        "Ruby",
        EmbedOptions {
            model: Some("text-embedding-3-small"),
            provider: Some("openai"),
            config: Some(config.clone()),
            owner,
            ..Default::default()
        },
    )
    .await
    .unwrap();
}

fn usage_events(events: &Events) -> Vec<Map<String, Value>> {
    events
        .lock()
        .unwrap()
        .iter()
        .filter(|(n, _)| n == "usage.rust_llm")
        .map(|(_, p)| p.clone())
        .collect()
}

fn usage_owners(events: &Events) -> Vec<Value> {
    usage_events(events)
        .iter()
        .map(|p| p["owner"].clone())
        .collect()
}

// spec: accounting/usage_owner_spec.rb:30 includes the keyword owner in the usage instrumentation payload
#[tokio::test]
async fn the_keyword_owner_is_in_the_usage_payload() {
    let (_server, config, events) = setup().await;

    embed_with(&config, Some("account-1".into())).await;

    let event = &usage_events(&events)[0];
    assert_eq!(event["operation"], "embedding");
    assert_eq!(event["status"], "succeeded");
    assert_eq!(event["owner"], "account-1");
}

// spec: accounting/usage_owner_spec.rb:37 reports no owner outside a block and without the keyword
#[tokio::test]
async fn no_owner_outside_a_block_and_without_the_keyword() {
    let (_server, config, events) = setup().await;

    embed_with(&config, None).await;

    let event = &usage_events(&events)[0];
    assert!(event.contains_key("owner"));
    assert_eq!(event["owner"], Value::Null);
}

// spec: accounting/usage_owner_spec.rb:43 applies the ambient owner, lets the keyword win, and restores outer owners
#[tokio::test]
async fn the_ambient_owner_applies_the_keyword_wins_and_outer_owners_return() {
    let (_server, config, events) = setup().await;

    with_usage_owner(UsageOwner::from("outer"), async {
        embed_with(&config, None).await;
        embed_with(&config, Some("keyword".into())).await;
        with_usage_owner(UsageOwner::from("inner"), embed_with(&config, None)).await;
        embed_with(&config, None).await;
    })
    .await;
    embed_with(&config, None).await;

    assert_eq!(
        usage_owners(&events),
        [
            json!("outer"),
            json!("keyword"),
            json!("inner"),
            json!("outer"),
            Value::Null
        ]
    );
}

// spec: accounting/usage_owner_spec.rb:55 returns the block value and restores the owner when the block raises
#[tokio::test]
async fn the_block_value_returns_and_the_owner_is_restored_after_a_failure() {
    let (_server, config, events) = setup().await;

    assert_eq!(
        with_usage_owner(UsageOwner::from("account-1"), async { "done" }).await,
        "done"
    );
    let failed: Result<(), &str> =
        with_usage_owner(UsageOwner::from("account-1"), async { Err("boom") }).await;
    assert_eq!(failed, Err("boom"));
    // A panicking block unwinds the task-local scope too.
    let panicked = tokio::spawn(with_usage_owner(UsageOwner::from("account-1"), async {
        panic!("boom")
    }))
    .await;
    assert!(panicked.is_err());

    assert_eq!(usage_owner(), None);
    embed_with(&config, None).await;
    assert_eq!(usage_owners(&events), [Value::Null]);
}

// spec: accounting/usage_owner_spec.rb:63 keeps each thread and fiber to its own owner
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn each_task_keeps_its_own_owner() {
    let (_server, config, events) = setup().await;

    // Two tasks hold their owners while both are inside their blocks, as Ruby's threads do.
    let barrier = Arc::new(tokio::sync::Barrier::new(2));
    let tasks: Vec<_> = ["thread-1", "thread-2"]
        .into_iter()
        .map(|owner| {
            let (config, barrier) = (config.clone(), barrier.clone());
            tokio::spawn(with_usage_owner(UsageOwner::from(owner), async move {
                barrier.wait().await;
                embed_with(&config, None).await;
            }))
        })
        .collect();
    for task in tasks {
        task.await.unwrap();
    }

    // Ruby's fiber: suspended inside its block while the caller embeds outside it, then resumed.
    let (resume, resumed) = tokio::sync::oneshot::channel::<()>();
    let fiber_config = config.clone();
    let fiber = tokio::spawn(with_usage_owner(UsageOwner::from("fiber"), async move {
        resumed.await.unwrap();
        embed_with(&fiber_config, None).await;
    }));
    tokio::task::yield_now().await;
    embed_with(&config, None).await;
    resume.send(()).unwrap();
    fiber.await.unwrap();

    let owners = usage_owners(&events);
    let mut sorted: Vec<String> = owners.iter().map(Value::to_string).collect();
    sorted.sort();
    assert_eq!(
        sorted,
        ["\"fiber\"", "\"thread-1\"", "\"thread-2\"", "null"]
    );
    assert_eq!(owners[2..], [Value::Null, json!("fiber")]);
}

// spec: accounting/usage_owner_spec.rb:93 passes the owner to threads started inside the block
#[tokio::test]
async fn tasks_started_inside_the_block_get_its_owner() {
    let (_server, config, events) = setup().await;

    // Ruby threads inherit fiber storage; a spawned task inherits it through
    // `with_current_owner`, since task-locals do not cross `tokio::spawn` on their own.
    with_usage_owner(UsageOwner::from("parent"), async {
        let config = config.clone();
        tokio::spawn(with_current_owner(async move {
            embed_with(&config, None).await
        }))
        .await
        .unwrap();
    })
    .await;

    assert_eq!(usage_owners(&events), [json!("parent")]);
}
