//! RubyLLM 2.0's `support/instrumentation_spec.rb` and `workflow_spec.rb`: `*.rust_llm` events,
//! their payloads, and workflow/step context. Chat, tool, request, usage, and embedding events are
//! checked against replayed RubyLLM cassettes.

mod support;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use rust_llm::workflow::Workflow;
use rust_llm::{
    Chat, Config, EmbedOptions, Error, Parameter, Tool, ToolCall, ToolError, ToolResult, instrument,
};
use serde_json::{Map, Value, json};
use support::{Cassette, config_for};

type Events = Arc<Mutex<Vec<(String, Map<String, Value>, Option<Duration>)>>>;

/// `CaptureInstrumenter`: records `[name, payload]` as each event finishes.
fn capture(config: &mut Config) -> Events {
    let events: Events = Default::default();
    let sink = events.clone();
    config.instrumenter = Some(Arc::new(
        move |name: &str, payload: &Map<String, Value>, duration: Option<Duration>| {
            sink.lock()
                .unwrap()
                .push((name.to_string(), payload.clone(), duration));
        },
    ));
    events
}

fn capturing() -> (Arc<Config>, Events) {
    let mut config = Config::default();
    let events = capture(&mut config);
    (Arc::new(config), events)
}

fn payload(events: &Events, name: &str) -> Map<String, Value> {
    events
        .lock()
        .unwrap()
        .iter()
        .find(|(n, _, _)| n == name)
        .map(|(_, p, _)| p.clone())
        .unwrap_or_else(|| panic!("no {name} event"))
}

fn names(events: &Events) -> Vec<String> {
    events
        .lock()
        .unwrap()
        .iter()
        .map(|(n, _, _)| n.clone())
        .collect()
}

// spec: support/instrumentation_spec.rb:8 emits structured events through a Rails-compatible instrumenter
#[tokio::test]
async fn emits_structured_events_and_returns_the_block_result() {
    let (config, events) = capturing();
    let result = instrument(
        &config,
        "example.rust_llm",
        json!({ "provider": "openai" }).as_object().unwrap().clone(),
        async { Ok("ok") },
    )
    .await;
    assert_eq!(result.unwrap(), "ok");
    let recorded = events.lock().unwrap().clone();
    assert_eq!(recorded.len(), 1);
    assert_eq!(recorded[0].0, "example.rust_llm");
    assert_eq!(
        Value::Object(recorded[0].1.clone()),
        json!({ "provider": "openai" })
    );
    assert!(
        recorded[0].2.is_some(),
        "a block event carries its duration"
    );
}

// spec: support/instrumentation_spec.rb:19 allows no-op events when no instrumenter is configured
#[tokio::test]
async fn no_instrumenter_is_a_no_op() {
    let config = Arc::new(Config::default());
    assert_eq!(
        instrument(&config, "example.rust_llm", Map::new(), async { Ok(1) })
            .await
            .unwrap(),
        1
    );
}

// spec: support/instrumentation_spec.rb:31 does not swallow errors from the instrumented block
#[tokio::test]
async fn errors_pass_through_and_are_recorded() {
    let (config, events) = capturing();
    let result: rust_llm::Result<()> = instrument(&config, "example.rust_llm", Map::new(), async {
        Err(Error::Argument("boom".into()))
    })
    .await;
    assert!(matches!(result, Err(Error::Argument(m)) if m == "boom"));
    assert_eq!(
        payload(&events, "example.rust_llm")["exception"],
        json!(["Other", "boom"])
    );
}

// spec: support/instrumentation_spec.rb:45 emits rich chat events around the whole completion flow
// (against a replayed OpenAI cassette, with its usage event)
#[tokio::test]
async fn a_completed_chat_reports_response_tokens_cost_and_usage() {
    let cassette = Cassette::start(
        "chat_basic_chat_functionality_openai_gpt-5-nano_can_have_a_basic_conversation",
    )
    .await
    .expect("cassette");
    let mut config = (*config_for(&cassette, "openai")).clone();
    let events = capture(&mut config);
    let mut chat =
        Chat::with_config(Arc::new(config), Some("gpt-5-nano"), Some("openai"), false).unwrap();
    let response = chat.ask("What's 2 + 2?").await.unwrap();
    cassette.assert_all_matched().await;

    assert_eq!(
        names(&events).last().map(String::as_str),
        Some("chat.rust_llm")
    );
    let chat_event = payload(&events, "chat.rust_llm");
    assert_eq!(chat_event["provider"], "openai");
    assert_eq!(chat_event["provider_class"], "OpenAI");
    assert_eq!(chat_event["model"], "gpt-5-nano");
    assert_eq!(chat_event["streaming"], false);
    assert!(!chat_event.contains_key("operation") && !chat_event.contains_key("result"));
    assert_eq!(chat_event["input_messages"][0]["content"], "What's 2 + 2?");
    assert_eq!(chat_event["response"]["content"], response.content());
    assert_eq!(chat_event["response_role"], "assistant");
    assert_eq!(chat_event["response_model"], json!(response.model));
    assert_eq!(
        chat_event["messages_after"]
            .as_array()
            .unwrap()
            .last()
            .unwrap()["content"],
        response.content()
    );
    assert_eq!(
        chat_event["tokens"]["input_tokens"],
        json!(response.tokens().input)
    );
    assert_eq!(
        chat_event["cost"]["total"],
        json!(response.cost(None).total())
    );

    let usage = payload(&events, "usage.rust_llm");
    assert_eq!(usage["operation"], "chat");
    assert_eq!(usage["provider"], "openai");
    assert_eq!(usage["model"], "gpt-5-nano");
    assert_eq!(usage["status"], "succeeded");
    assert_eq!(usage["tokens"], chat_event["tokens"]);
    assert!(!usage.contains_key("usage_status") && !usage.contains_key("usage"));

    let request = payload(&events, "request.rust_llm");
    assert_eq!(request["provider"], "openai");
    assert_eq!(request["method"], "post");
    assert!(request["url"].as_str().unwrap().ends_with("/v1/responses"));
    assert_eq!(request["status"], 200);
}

// spec: support/instrumentation_spec.rb:86 marks streaming chat events when a block is passed
#[tokio::test]
async fn streaming_chats_are_marked() {
    let cassette =
        Cassette::start("chat_streaming_responses_openai_gpt-5-nano_supports_streaming_responses")
            .await
            .expect("cassette");
    let mut config = (*config_for(&cassette, "openai")).clone();
    let events = capture(&mut config);
    let mut chat =
        Chat::with_config(Arc::new(config), Some("gpt-5-nano"), Some("openai"), false).unwrap();
    chat.ask_stream("Count from 1 to 3", |_| {}).await.unwrap();
    cassette.assert_all_matched().await;
    assert_eq!(payload(&events, "chat.rust_llm")["streaming"], true);
}

// spec: support/instrumentation_spec.rb:116 emits one usage event for every transport attempt
#[tokio::test]
async fn one_usage_event_per_transport_attempt() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(500)
                .set_body_json(json!({ "error": { "message": "try again" } })),
        )
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model": "gpt-4.1-nano",
            "choices": [{ "message": { "role": "assistant", "content": "done" }, "finish_reason": "stop" }],
            "usage": { "prompt_tokens": 5, "completion_tokens": 2 }
        })))
        .mount(&server)
        .await;
    let mut config = Config::default();
    config
        .set("openai_api_base", format!("{}/v1", server.uri()))
        .set("openai_api_key", "test");
    config.max_retries = 1;
    config.retry_interval = 0.0;
    let events = capture(&mut config);
    let chat = Chat::with_config(
        Arc::new(config),
        Some("gpt-4.1-nano"),
        Some("openai"),
        false,
    )
    .unwrap()
    .with_protocol(rust_llm::ProtocolName::ChatCompletions);
    let mut chat = chat.with_temperature(0.2);
    chat.ask("Hello").await.unwrap();
    assert_eq!(payload(&events, "chat.rust_llm")["temperature"], 0.2);

    let usage: Vec<Map<String, Value>> = events
        .lock()
        .unwrap()
        .iter()
        .filter(|(n, _, _)| n == "usage.rust_llm")
        .map(|(_, p, _)| p.clone())
        .collect();
    assert_eq!(
        usage
            .iter()
            .map(|p| p["status"].clone())
            .collect::<Vec<_>>(),
        [json!("failed"), json!("succeeded")]
    );
    let last = usage.last().unwrap();
    assert_eq!(last["operation"], "chat");
    assert_eq!(last["model"], "gpt-4.1-nano");
    assert_eq!(
        last["tokens"],
        json!({ "input_tokens": 5, "output_tokens": 2, "cache_write_tokens": 0 })
    );
    assert!(last["cost"]["total"].is_number());
    // Retries happen inside one request event, as Faraday's retry middleware does.
    assert_eq!(
        names(&events)
            .iter()
            .filter(|n| *n == "request.rust_llm")
            .count(),
        1
    );
}

struct Weather;

#[async_trait]
impl Tool for Weather {
    fn description(&self) -> String {
        "Gets current weather for a location".into()
    }
    fn parameters(&self) -> Vec<Parameter> {
        vec![
            Parameter::new("latitude").description("Latitude (e.g., 52.5200)"),
            Parameter::new("longitude").description("Longitude (e.g., 13.4050)"),
        ]
    }
    async fn execute(
        &self,
        args: Map<String, Value>,
        _: &ToolCall,
    ) -> Result<ToolResult, ToolError> {
        let lat = args["latitude"]
            .as_str()
            .map(str::to_string)
            .unwrap_or_else(|| args["latitude"].to_string());
        let lon = args["longitude"]
            .as_str()
            .map(str::to_string)
            .unwrap_or_else(|| args["longitude"].to_string());
        Ok(format!("Current weather at {lat}, {lon}: 15°C, Wind: 10 km/h").into())
    }
}

// spec: support/instrumentation_spec.rb:156 emits tool call events with arguments and result
#[tokio::test]
async fn tool_call_events_carry_arguments_and_result() {
    let cassette = Cassette::start("chat_function_calling_openai_gpt-5-nano_can_use_tools")
        .await
        .expect("cassette");
    let mut config = (*config_for(&cassette, "openai")).clone();
    let events = capture(&mut config);
    let mut chat = Chat::with_config(Arc::new(config), Some("gpt-5-nano"), Some("openai"), false)
        .unwrap()
        .with_tool(Weather);
    chat.ask("What's the weather in Berlin? (52.5200, 13.4050)")
        .await
        .unwrap();
    cassette.assert_all_matched().await;

    let tool = payload(&events, "tool_call.rust_llm");
    assert_eq!(tool["provider"], "openai");
    assert_eq!(tool["model"], "gpt-5-nano");
    assert_eq!(tool["tool_name"], "weather");
    assert!(
        tool["tool_call_id"]
            .as_str()
            .is_some_and(|id| !id.is_empty())
    );
    assert!(tool["tool_arguments"].get("latitude").is_some());
    let result = tool["result"].as_str().unwrap();
    assert!(result.starts_with("Current weather at 52.52"), "{result}");
    assert_eq!(tool["result_content"], tool["result"]);
    assert!(!tool.contains_key("operation") && !tool.contains_key("tool_object"));
    // The tool ran inside the conversation: two chat events, the tool between them.
    let order = names(&events)
        .into_iter()
        .filter(|n| n == "chat.rust_llm" || n == "tool_call.rust_llm")
        .collect::<Vec<_>>();
    assert_eq!(
        order,
        ["chat.rust_llm", "tool_call.rust_llm", "chat.rust_llm"]
    );
}

// spec: support/instrumentation_spec.rb:191 emits embedding events with usage and vector dimensions
#[tokio::test]
async fn embedding_events_carry_usage_and_dimensions() {
    let cassette = Cassette::start(
        "embedding_basic_functionality_openai_text-embedding-3-small_can_handle_a_single_text",
    )
    .await
    .expect("cassette");
    let mut config = (*config_for(&cassette, "openai")).clone();
    let events = capture(&mut config);
    let text = "Ruby is a programmer's best friend";
    let options = EmbedOptions {
        model: Some("text-embedding-3-small"),
        config: Some(Arc::new(config)),
        ..Default::default()
    };
    let embedding = rust_llm::embed(text, options).await.unwrap();
    cassette.assert_all_matched().await;

    let (name, event, _) = events.lock().unwrap().last().cloned().unwrap();
    assert_eq!(name, "embedding.rust_llm");
    assert_eq!(event["provider"], "openai");
    assert_eq!(event["provider_class"], "OpenAI");
    assert_eq!(event["model"], "text-embedding-3-small");
    assert_eq!(event["input"], text);
    assert_eq!(event["response_model"], json!(embedding.model));
    assert_eq!(event["embedding_count"], 1);
    let rust_llm::Vectors::Single(v) = &embedding.vectors else {
        panic!("single vector")
    };
    assert_eq!(event["embedding_dimensions"], v.len());
    assert_eq!(
        event["tokens"]["input_tokens"],
        json!(embedding.input_tokens)
    );
    assert!(!event.contains_key("operation"));
}

// ---- workflow_spec.rb --------------------------------------------------------------------------

fn workflow(
    config: &Arc<Config>,
    name: &str,
    id: Option<&str>,
    metadata: Option<Value>,
) -> Workflow {
    Workflow::new(name, id, metadata, config.clone()).unwrap()
}

fn example(
    config: &Arc<Config>,
    name: &str,
    payload: Map<String, Value>,
) -> impl std::future::Future<Output = rust_llm::Result<()>> {
    let config = config.clone();
    let name = name.to_string();
    async move { instrument(&config, &name, payload, async { Ok(()) }).await }
}

// spec: workflow_spec.rb:11 adds workflow and step identity to every nested instrumentation event
#[tokio::test]
async fn workflow_and_step_identity_reach_nested_events() {
    let (config, events) = capturing();
    let result = workflow(&config, "Write article", Some("article-42"), None)
        .run(|wf| async move {
            let inner = config.clone();
            wf.step("Research", Some("research-1"), async move {
                instrument(&inner, "example.rust_llm", Map::new(), async {
                    Ok("notes")
                })
                .await
            })
            .await
        })
        .await
        .unwrap();
    assert_eq!(result, "notes");
    let identity = json!({ "workflow_id": "article-42", "workflow_name": "Write article", "workflow_step_id": "research-1", "workflow_step_name": "Research" });
    for name in ["example.rust_llm", "workflow_step.rust_llm"] {
        let p = payload(&events, name);
        for (k, v) in identity.as_object().unwrap() {
            assert_eq!(&p[k], v, "{name} {k}");
        }
    }
    let wf = payload(&events, "workflow.rust_llm");
    assert_eq!(wf["workflow_id"], "article-42");
    assert_eq!(wf["workflow_name"], "Write article");
    assert!(!wf.contains_key("workflow_step_id") && !wf.contains_key("workflow_metadata"));
}

// spec: workflow_spec.rb:39 attaches workflow metadata as workflow_metadata alongside per-call metadata
#[tokio::test]
async fn workflow_metadata_sits_beside_per_call_metadata() {
    let (config, events) = capturing();
    let c = config.clone();
    workflow(
        &config,
        "With metadata",
        Some("meta-1"),
        Some(json!({ "account_id": 7 })),
    )
    .run(|wf| async move {
        let per_call = json!({ "metadata": { "request": "r-1" } })
            .as_object()
            .unwrap()
            .clone();
        wf.step("Only step", None, example(&c, "example.rust_llm", per_call))
            .await
    })
    .await
    .unwrap();
    let p = payload(&events, "example.rust_llm");
    assert_eq!(p["workflow_metadata"], json!({ "account_id": 7 }));
    assert_eq!(p["metadata"], json!({ "request": "r-1" }));
    assert_eq!(
        payload(&events, "workflow.rust_llm")["workflow_metadata"],
        json!({ "account_id": 7 })
    );
    assert_eq!(
        payload(&events, "workflow_step.rust_llm")["workflow_metadata"],
        json!({ "account_id": 7 })
    );
}

// spec: workflow_spec.rb:54 generates IDs when they are omitted
#[tokio::test]
async fn ids_are_generated_when_omitted() {
    let (config, events) = capturing();
    let wf = workflow(&config, "Generated IDs", None, None);
    let id = wf.id().to_string();
    wf.run(|wf| async move { wf.step("First", None, async { Ok(()) }).await })
        .await
        .unwrap();
    let uuid = regex::Regex::new(r"\A[0-9a-f-]{36}\z").unwrap();
    assert!(uuid.is_match(&id));
    assert!(
        uuid.is_match(
            payload(&events, "workflow_step.rust_llm")["workflow_step_id"]
                .as_str()
                .unwrap()
        )
    );
}

fn step_event(events: &Events, id: &str) -> Map<String, Value> {
    events
        .lock()
        .unwrap()
        .iter()
        .find(|(n, p, _)| n == "workflow_step.rust_llm" && p["workflow_step_id"] == id)
        .map(|(_, p, _)| p.clone())
        .unwrap()
}

// spec: workflow_spec.rb:66 records parent identity for nested steps
#[tokio::test]
async fn nested_steps_record_their_parent() {
    let (config, events) = capturing();
    workflow(&config, "Nested", Some("nested-1"), None)
        .run(|wf| async move {
            let inner = wf.clone();
            wf.step("Outer", Some("outer-1"), async move {
                inner.step("Inner", Some("inner-1"), async { Ok(()) }).await
            })
            .await
        })
        .await
        .unwrap();
    assert_eq!(
        step_event(&events, "inner-1")["workflow_step_parent_id"],
        "outer-1"
    );
    assert!(!step_event(&events, "outer-1").contains_key("workflow_step_parent_id"));
}

fn workflow_event(events: &Events, id: &str) -> Map<String, Value> {
    events
        .lock()
        .unwrap()
        .iter()
        .find(|(n, p, _)| n == "workflow.rust_llm" && p["workflow_id"] == id)
        .map(|(_, p, _)| p.clone())
        .unwrap()
}

// spec: workflow_spec.rb:84 links nested workflows to their parent workflow and step
#[tokio::test]
async fn nested_workflows_link_to_their_parent() {
    let (config, events) = capturing();
    let c = config.clone();
    workflow(&config, "Outer", Some("outer-wf"), None)
        .run(|outer| async move {
            outer
                .step("Compose", Some("compose-1"), async move {
                    let c2 = c.clone();
                    workflow(&c, "Inner", Some("inner-wf"), None)
                        .run(|inner| async move {
                            inner
                                .step(
                                    "Summarize",
                                    Some("inner-step-1"),
                                    example(&c2, "example.rust_llm", Map::new()),
                                )
                                .await
                        })
                        .await?;
                    example(&c, "resumed.rust_llm", Map::new()).await
                })
                .await
        })
        .await
        .unwrap();
    let inner = workflow_event(&events, "inner-wf");
    assert_eq!(inner["workflow_parent_id"], "outer-wf");
    assert_eq!(inner["workflow_parent_step_id"], "compose-1");
    let example_event = payload(&events, "example.rust_llm");
    assert_eq!(example_event["workflow_id"], "inner-wf");
    assert_eq!(example_event["workflow_step_id"], "inner-step-1");
    assert_eq!(example_event["workflow_parent_id"], "outer-wf");
    assert!(!workflow_event(&events, "outer-wf").contains_key("workflow_parent_id"));
    let resumed = payload(&events, "resumed.rust_llm");
    assert_eq!(resumed["workflow_id"], "outer-wf");
    assert_eq!(resumed["workflow_step_id"], "compose-1");
    assert!(!resumed.contains_key("workflow_parent_id"));
}

// spec: workflow_spec.rb:108 recomputes parent links each time a workflow runs
#[tokio::test]
async fn parent_links_are_recomputed_per_run() {
    let (config, events) = capturing();
    let reusable = workflow(&config, "Reusable", Some("reusable"), None);
    let r = reusable.clone();
    workflow(&config, "Outer", Some("outer"), None)
        .run(|outer| async move {
            outer
                .step("Nested", Some("nested"), async move {
                    r.run(|_| async { Ok(()) }).await
                })
                .await
        })
        .await
        .unwrap();
    reusable.run(|_| async { Ok(()) }).await.unwrap();
    let runs: Vec<Map<String, Value>> = events
        .lock()
        .unwrap()
        .iter()
        .filter(|(n, p, _)| n == "workflow.rust_llm" && p["workflow_id"] == "reusable")
        .map(|(_, p, _)| p.clone())
        .collect();
    assert_eq!(runs[0]["workflow_parent_id"], "outer");
    assert_eq!(runs[0]["workflow_parent_step_id"], "nested");
    assert!(
        !runs[1].contains_key("workflow_parent_id")
            && !runs[1].contains_key("workflow_parent_step_id")
    );
}

// spec: workflow_spec.rb:124 restores the previous instrumentation context after errors
#[tokio::test]
async fn context_is_restored_after_errors() {
    let (config, events) = capturing();
    let result: rust_llm::Result<()> = workflow(&config, "Failing", Some("failing-1"), None)
        .run(|wf| async move {
            wf.step("Explode", None, async {
                Err(Error::Argument("boom".into()))
            })
            .await
        })
        .await;
    assert!(matches!(result, Err(Error::Argument(m)) if m == "boom"));
    example(&config, "after.rust_llm", Map::new())
        .await
        .unwrap();
    assert!(!payload(&events, "after.rust_llm").contains_key("workflow_id"));
    assert_eq!(
        payload(&events, "workflow_step.rust_llm")["exception"],
        json!(["Other", "boom"])
    );
}

// spec: workflow_spec.rb:136 rejects empty names and missing blocks (a Rust closure is always given)
#[tokio::test]
async fn empty_names_are_rejected() {
    let config = Arc::new(Config::default());
    let err = Workflow::new("", None, None, config.clone()).unwrap_err();
    assert!(err.to_string().contains("name cannot be empty"));
    let err = workflow(&config, "Ok", None, None)
        .step("", None, async { Ok(()) })
        .await
        .unwrap_err();
    assert!(err.to_string().contains("step name cannot be empty"));
}

// Events emitted by chats inside a workflow step carry the step (Ruby's thread-local, here a
// task-local that follows the future).
#[tokio::test]
async fn chats_inside_a_step_carry_the_workflow() {
    let cassette = Cassette::start(
        "chat_basic_chat_functionality_openai_gpt-5-nano_can_have_a_basic_conversation",
    )
    .await
    .expect("cassette");
    let mut config = (*config_for(&cassette, "openai")).clone();
    let events = capture(&mut config);
    let config = Arc::new(config);
    let chat_config = config.clone();
    workflow(&config, "Answer", Some("wf-1"), None)
        .run(|wf| async move {
            wf.step("Ask", Some("ask-1"), async move {
                let mut chat =
                    Chat::with_config(chat_config, Some("gpt-5-nano"), Some("openai"), false)?;
                chat.ask("What's 2 + 2?").await.map(|_| ())
            })
            .await
        })
        .await
        .unwrap();
    cassette.assert_all_matched().await;
    for name in ["chat.rust_llm", "usage.rust_llm", "request.rust_llm"] {
        let p = payload(&events, name);
        assert_eq!(p["workflow_id"], "wf-1", "{name}");
        assert_eq!(p["workflow_step_id"], "ask-1", "{name}");
    }
}
