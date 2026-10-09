//! RubyLLM's `spec/ruby_llm/protocols/*` and `spec/ruby_llm/providers/*` live examples, plus the
//! embedding specs' provider-specific cases, replayed from its own cassettes.

mod support;

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use rust_llm::{
    Chat, EmbedOptions, Parameter, ProtocolName, ProviderTool, Tool, ToolCall, ToolError,
    ToolResult, UploadOptions, Vectors, embed,
};
use serde_json::{Map, Value, json};
use support::{Cassette, config_for};

async fn start(name: &str) -> Cassette {
    Cassette::start(name)
        .await
        .unwrap_or_else(|| panic!("missing cassette {name}"))
}

// ---- providers/perplexity_spec.rb: Agent API ---------------------------------------------------

fn preset(cassette: &Cassette) -> Chat {
    Chat::with_config(
        config_for(cassette, "perplexity"),
        Some("fast"),
        Some("perplexity"),
        false,
    )
    .unwrap()
}

fn agent(cassette: &Cassette) -> Chat {
    Chat::with_config(
        config_for(cassette, "perplexity"),
        Some("openai/gpt-5-mini"),
        Some("perplexity"),
        false,
    )
    .unwrap()
}

const RAILS: &str = "In one sentence: who created Ruby on Rails?";

#[tokio::test]
async fn perplexity_answers_through_a_preset_reporting_the_billed_cost() {
    let cassette =
        start("providers_perplexity_agent_api_answers_through_a_preset_reporting_the_billed_cost")
            .await;
    let mut chat = preset(&cassette);
    let response = chat.ask(RAILS).await.unwrap();
    assert!(
        response.content().contains("Heinemeier Hansson"),
        "{:?}",
        response.content
    );
    assert!(
        response.cost(None).total().is_some_and(|t| t > 0.0),
        "{:?}",
        response.cost(None)
    );
    cassette.assert_all_matched().await;
}

#[tokio::test]
async fn perplexity_streams_through_a_preset_without_running_searches_as_tools() {
    let cassette = start("providers_perplexity_agent_api_streams_through_a_preset_without_running_its_web_searches_as_tools").await;
    let mut chat = preset(&cassette);
    let mut chunks = 0;
    let response = chat.ask_stream(RAILS, |_| chunks += 1).await.unwrap();
    assert!(chunks > 0);
    assert!(!response.is_tool_call());
    assert!(
        response.content().contains("Heinemeier Hansson"),
        "{:?}",
        response.content
    );
    cassette.assert_all_matched().await;
}

#[tokio::test]
async fn perplexity_continues_a_searched_conversation() {
    let cassette = start("providers_perplexity_agent_api_continues_a_searched_conversation").await;
    let mut chat = preset(&cassette);
    chat.ask("Which company created the Ruby on Rails framework?")
        .await
        .unwrap();
    let followup = chat
        .ask("In which year was that company founded? Answer with the year only.")
        .await
        .unwrap();
    let digits = regex::Regex::new(r"\d{4}").unwrap();
    assert!(
        digits.is_match(followup.content()),
        "{:?}",
        followup.content
    );
    cassette.assert_all_matched().await;
}

/// The spec's anonymous `add` tool.
struct Add;

#[async_trait]
impl Tool for Add {
    fn name(&self) -> String {
        "add".into()
    }
    fn description(&self) -> String {
        "Add two integers.".into()
    }
    fn parameters(&self) -> Vec<Parameter> {
        vec![
            Parameter::new("left").kind("integer"),
            Parameter::new("right").kind("integer"),
        ]
    }
    async fn execute(
        &self,
        args: Map<String, Value>,
        _: &ToolCall,
    ) -> Result<ToolResult, ToolError> {
        let sum = args["left"].as_i64().unwrap_or(0) + args["right"].as_i64().unwrap_or(0);
        Ok(json!(sum).into())
    }
}

#[tokio::test]
async fn perplexity_calls_a_local_tool_while_streaming() {
    let cassette = start("providers_perplexity_agent_api_calls_a_local_tool_while_streaming").await;
    let calls = Arc::new(Mutex::new(Vec::new()));
    let seen = calls.clone();
    let mut chat = agent(&cassette)
        .with_tool(Add)
        .before_tool_call(move |c| seen.lock().unwrap().push(Value::Object(c.arguments())));
    let response = chat
        .ask_stream("Use the add tool to add 17 and 25.", |_| {})
        .await
        .unwrap();
    assert_eq!(
        *calls.lock().unwrap(),
        vec![json!({ "left": 17, "right": 25 })]
    );
    assert!(response.content().contains("42"), "{:?}", response.content);
    cassette.assert_all_matched().await;
}

#[tokio::test]
async fn perplexity_returns_json_schema_output() {
    let cassette =
        start("providers_perplexity_agent_api_returns_json_schema_output_through_with_schema")
            .await;
    let mut chat = agent(&cassette).with_schema(json!({
        "type": "object",
        "properties": { "name": { "type": "string" }, "year": { "type": "integer" } },
        "required": ["name", "year"],
        "additionalProperties": false
    }));
    let message = chat
        .ask(
            "Extract these two facts: Ruby was released in 1995. Return its name and release year.",
        )
        .await
        .unwrap();
    assert_eq!(
        message.parsed().unwrap(),
        Some(json!({ "name": "Ruby", "year": 1995 }))
    );
    cassette.assert_all_matched().await;
}

// ---- protocols/openrouter/responses_spec.rb -----------------------------------------------------

fn openrouter_responses(cassette: &Cassette) -> Chat {
    Chat::with_config(
        config_for(cassette, "openrouter"),
        Some("openai/gpt-5.2"),
        Some("openrouter"),
        false,
    )
    .unwrap()
    .with_protocol(ProtocolName::Responses)
    .with_max_output_tokens(700)
}

#[tokio::test]
async fn openrouter_executes_a_hosted_shell() {
    let cassette = start("protocols_openrouter_responses_executes_a_hosted_shell_and_returns_its_real_output_and_billed_usage").await;
    let mut chat =
        openrouter_responses(&cassette).with_provider_tools([ProviderTool::with_options(
            "code_execution",
            json!({ "parameters": { "engine": "openrouter" } }),
        )]);
    let message = chat
        .ask(
            r#"Use the hosted shell to run python3 -c "print(17*23)". Reply with the result only."#,
        )
        .await
        .unwrap();
    assert!(message.content().contains("391"), "{:?}", message.content);
    let call = message
        .server_tool_calls
        .iter()
        .find(|c| c.kind == "openrouter:shell")
        .expect("shell call");
    let first = call.result.as_ref().and_then(|r| r.get(0)).expect("result");
    assert!(
        first["stdout"].as_str().is_some_and(|s| s.contains("391")),
        "{first}"
    );
    assert_eq!(first.pointer("/outcome/exit_code"), Some(&json!(0)));
    assert!(message.tokens().input.unwrap_or(0) > 0);
    assert!(message.tokens().output.unwrap_or(0) > 0);
    assert!(message.cost(None).total().is_some_and(|t| t > 0.0));
    // `usage.cost`; `server_tool_use_details` reports only OpenRouter's totals spanning tools
    // (`tool_calls_requested`/`tool_calls_executed`), which 2.1 leaves out (d21bf001).
    assert_eq!(message.tokens().reported_cost, Some(0.0046345));
    assert_eq!(message.tokens().server_tool_use, None);
    cassette.assert_all_matched().await;
}

#[tokio::test]
async fn openrouter_records_remote_mcp_arguments_without_inventing_names_or_results() {
    let cassette = start("protocols_openrouter_responses_records_remote_mcp_arguments_without_inventing_omitted_tool_names_or_results").await;
    let mut chat = openrouter_responses(&cassette).with_provider_tools([ProviderTool::with_options(
        "mcp",
        json!({ "name": "learn", "url": "https://learn.microsoft.com/api/mcp", "allowed_tools": ["microsoft_docs_search"], "require_approval": "never" }),
    )]);
    let mut chunks = Vec::new();
    let prompt = "Use microsoft_docs_search to find Microsoft documentation about Ruby. Summarize in one sentence.";
    let message = chat
        .ask_stream(prompt, |c| chunks.push(c.clone()))
        .await
        .unwrap();
    let calls: Vec<_> = chunks
        .iter()
        .flat_map(|c| c.server_tool_calls.clone())
        .filter(|c| c.kind == "mcp_call")
        .collect();
    let first = calls.first().expect("mcp_call");
    let input = first
        .input
        .as_ref()
        .map(|i| {
            i.as_str()
                .map(str::to_string)
                .unwrap_or_else(|| i.to_string())
        })
        .unwrap_or_default();
    assert!(input.contains("Ruby"), "{input}");
    assert_eq!(first.name, None);
    assert_eq!(first.result, None);
    assert!(!message.content().is_empty());
    assert!(message.tokens().input.unwrap_or(0) > 0);
    assert!(message.tokens().output.unwrap_or(0) > 0);
    cassette.assert_all_matched().await;
}

// ---- protocols/deepseek/files_spec.rb ------------------------------------------------------------

/// The upload is multipart: assert path, method, and form fields, then the JSON-free GET/DELETE.
#[tokio::test]
async fn deepseek_uploads_and_retrieves_an_image_with_an_expiry() {
    let cassette =
        start("protocols_deepseek_files_uploads_and_retrieves_an_image_with_an_expiry").await;
    let config = config_for(&cassette, "deepseek");
    let image = format!("{}/tests/fixtures/ruby.png", env!("CARGO_MANIFEST_DIR"));
    let file = rust_llm::upload(
        image.as_str(),
        UploadOptions {
            provider: Some("deepseek"),
            expires_in: Some(3600),
            config: Some(config.clone()),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    let stored = rust_llm::UploadedFile::find(
        &file.id,
        rust_llm::FileOptions {
            provider: Some("deepseek"),
            config: Some(config),
        },
    )
    .await
    .unwrap();
    assert_eq!(stored.id, file.id);
    assert_eq!(stored.filename.as_deref(), Some("ruby.png"));
    assert_eq!(stored.purpose.as_deref(), Some("user_data"));
    assert_eq!(
        stored.byte_size,
        Some(std::fs::metadata(&image).unwrap().len())
    );
    assert!(
        stored.expires_at > stored.created_at && stored.created_at.is_some(),
        "{stored:?}"
    );
    // The spec's `ensure provider.connection.delete("files/#{file.id}")` is a raw connection call.
    let deleted = reqwest::Client::new()
        .delete(format!("{}/files/{}", cassette.server.uri(), file.id))
        .send()
        .await
        .unwrap();
    assert!(deleted.status().is_success());

    let requests = cassette.server.received_requests().await.unwrap();
    assert_eq!(requests.len(), cassette.count);
    assert_eq!(
        (requests[0].method.as_str(), requests[0].url.path()),
        ("POST", "/files")
    );
    let body = String::from_utf8_lossy(&requests[0].body);
    assert!(
        body.contains("name=\"file\"; filename=\"ruby.png\""),
        "file part"
    );
    assert!(
        body.contains("name=\"purpose\"\r\n\r\nuser_data\r\n"),
        "purpose"
    );
    assert!(
        body.contains("name=\"expires_after[anchor]\"\r\n\r\ncreated_at\r\n"),
        "anchor"
    );
    assert!(
        body.contains("name=\"expires_after[seconds]\"\r\n\r\n3600\r\n"),
        "seconds"
    );
    assert_eq!(
        (requests[1].method.as_str(), requests[1].url.path()),
        ("GET", format!("/files/{}", file.id).as_str())
    );
    assert_eq!(
        (requests[2].method.as_str(), requests[2].url.path()),
        ("DELETE", format!("/files/{}", file.id).as_str())
    );
}

// providers/mistral/chat_completions/batches_spec.rb 'submits, reloads, and cancels an embedding
// batch' is replayed in tests/spec_batches.rs.

// ---- providers/xai/chat_completions/batches_live_spec.rb ----------------------------------------

#[tokio::test]
async fn xai_collects_structured_responses_and_chat_completions_results_in_order() {
    let name = "providers_xai_chatcompletions_batches_collects_completed_structured_responses_and_chat_completions_results_in_submission_order";
    let cassette = start(name).await;
    let config = config_for(&cassette, "xai");
    let schema = json!({ "type": "object", "properties": { "language": { "type": "string" } }, "required": ["language"], "additionalProperties": false });
    let mut structured = Chat::with_config(config.clone(), Some("grok-4.3"), Some("xai"), false)
        .unwrap()
        .with_schema(schema)
        .with_max_output_tokens(256);
    structured.ask_later("Return language Ruby.").unwrap();
    let mut plain = Chat::with_config(config.clone(), Some("grok-4.3"), Some("xai"), false)
        .unwrap()
        .with_protocol(ProtocolName::ChatCompletions)
        .with_max_output_tokens(256);
    plain
        .ask_later("Reply with exactly one word: Rails")
        .unwrap();
    let mut batch = rust_llm::batch(vec![structured, plain]).await.unwrap();
    for _ in 0..10 {
        if batch.refresh().await.unwrap().is_complete() {
            break;
        }
    }
    assert!(batch.is_complete());
    let results = batch.messages().await.unwrap();
    let results: Vec<_> = results.into_iter().map(|m| m.expect("message")).collect();
    assert_eq!(
        results[0].parsed().unwrap(),
        Some(json!({ "language": "Ruby" }))
    );
    assert!(
        results[1].content().contains("Rails"),
        "{:?}",
        results[1].content
    );
    assert!(results[0].tokens().input.unwrap_or(0) > 0);
    assert!(
        results[0].tokens().reported_cost.is_some_and(|c| c >= 0.0),
        "{:?}",
        results[0].tokens()
    );
    // The batch `name` is `SecureRandom.hex`, so request 0's body is the one random field.
    let mismatches: Vec<String> = cassette
        .mismatches
        .lock()
        .unwrap()
        .iter()
        .filter(|m| !m.starts_with("request 0: /name"))
        .cloned()
        .collect();
    assert!(mismatches.is_empty(), "{mismatches:#?}");
    assert_eq!(
        cassette.server.received_requests().await.unwrap().len(),
        cassette.count
    );
}

// ---- embedding_spec.rb: provider-specific cases --------------------------------------------------

const TEXT: &str = "Ruby is a programmer's best friend";

fn perplexity_embed<'a>(cassette: &Cassette) -> EmbedOptions<'a> {
    EmbedOptions {
        model: Some("pplx-embed-v1-0.6b"),
        provider: Some("perplexity"),
        config: Some(config_for(cassette, "perplexity")),
        ..Default::default()
    }
}

#[tokio::test]
async fn perplexity_decodes_a_single_text_into_int8_vectors() {
    let cassette = start("embedding_perplexity_int8_embeddings_perplexity_pplx-embed-v1-0_6b_decodes_a_single_text_into_int8_vectors").await;
    let e = embed(TEXT, perplexity_embed(&cassette)).await.unwrap();
    let Vectors::Single(v) = &e.vectors else {
        panic!("expected one vector")
    };
    assert_eq!(v.len(), 1024);
    assert!(
        v.iter()
            .all(|x| x.fract() == 0.0 && (-128.0..=127.0).contains(x))
    );
    assert!(
        v.iter().any(|x| *x < 0.0),
        "signed bytes decode as negatives"
    );
    assert_eq!(e.model, "pplx-embed-v1-0.6b");
    assert!(e.tokens().input.unwrap_or(0) > 0);
    cassette.assert_all_matched().await;
}

#[tokio::test]
async fn perplexity_embeds_multiple_texts_with_custom_dimensions() {
    let cassette = start("embedding_perplexity_int8_embeddings_perplexity_pplx-embed-v1-0_6b_handles_multiple_texts_with_custom_dimensions").await;
    let texts = vec!["Ruby".to_string(), "Python".into(), "JavaScript".into()];
    let e = embed(
        texts,
        EmbedOptions {
            dimensions: Some(256),
            ..perplexity_embed(&cassette)
        },
    )
    .await
    .unwrap();
    let Vectors::Batch(rows) = &e.vectors else {
        panic!("expected a batch")
    };
    assert_eq!(rows.len(), 3);
    assert!(rows.iter().all(|r| r.len() == 256));
    cassette.assert_all_matched().await;
}

#[tokio::test]
async fn openrouter_embedding_reports_the_exact_provider_cost() {
    let cassette = start("embedding_provider-reported_cost_openrouter_openai_text-embedding-3-small_returns_the_exact_cost_the_provider_reported").await;
    let e = embed(
        TEXT,
        EmbedOptions {
            model: Some("openai/text-embedding-3-small"),
            provider: Some("openrouter"),
            config: Some(config_for(&cassette, "openrouter")),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    let reported = e.tokens().reported_cost.expect("reported cost");
    assert!(reported > 0.0);
    assert_eq!(e.cost().total(), Some(reported));
    cassette.assert_all_matched().await;
}
