//! The rest of RubyLLM's chat specs, replayed from its own cassettes: text attachments
//! (`chat_content_spec.rb`), provider options (`chat_request_options_spec.rb`), real error
//! scenarios (`chat_error_spec.rb`), a failing tool (`chat_tools_spec.rb`), citations
//! (`chat_citations_spec.rb`), and web search, code execution, and Responses dialects
//! (`chat_provider_tools_spec.rb`).

mod support;

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use rust_llm::{
    Attachment, Chat, ErrorKind, Message, Parameter, ProtocolName, ProviderTool, Role, SearchResults, ThinkingConfig,
    Tool, ToolCall, ToolError, ToolResult,
};
use serde_json::{Map, Value, json};
use support::{CHAT_MODELS, Cassette, cassette_name, config_for};

fn fixture(name: &str) -> String {
    format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
}

async fn start(name: &str) -> Cassette {
    Cassette::start(name).await.unwrap_or_else(|| panic!("missing cassette {name}"))
}

fn chat_for(cassette: &Cassette, provider: &str, model: &str) -> Chat {
    let assume = matches!(provider, "ollama" | "gpustack" | "ollama_cloud" | "hetzner");
    Chat::with_config(config_for(cassette, provider), Some(model), Some(provider), assume).expect("chat")
}

fn check(cond: bool, what: impl Into<String>) -> Result<(), String> {
    if cond { Ok(()) } else { Err(what.into()) }
}

/// Runs `body` against every `(provider, model)` whose cassette exists, reporting all failures
/// together. Returns how many cassettes replayed.
async fn each<F, Fut>(rows: Vec<(String, &'static str, &'static str)>, body: F) -> usize
where
    F: Fn(Cassette, &'static str, &'static str) -> Fut,
    Fut: std::future::Future<Output = Result<Cassette, String>>,
{
    let mut failures = Vec::new();
    let mut ran = 0;
    for (name, provider, model) in rows {
        let Some(cassette) = Cassette::start(&name).await else { continue };
        ran += 1;
        match body(cassette, provider, model).await {
            Ok(cassette) => {
                let r = futures::FutureExt::catch_unwind(std::panic::AssertUnwindSafe(cassette.assert_all_matched())).await;
                if let Err(p) = r {
                    failures.push(format!("{provider} {model}: {}", p.downcast_ref::<String>().cloned().unwrap_or_default()));
                }
            }
            Err(e) => failures.push(format!("{provider} {model}: {e}")),
        }
    }
    assert!(ran > 0, "no cassettes");
    assert!(failures.is_empty(), "{} of {ran} failed:\n{}", failures.len(), failures.join("\n\n"));
    eprintln!("{ran} replayed");
    ran
}

fn chat_rows(describe: &str, it: &str, providers: &[&str]) -> Vec<(String, &'static str, &'static str)> {
    CHAT_MODELS
        .iter()
        .filter(|(p, _)| providers.is_empty() || providers.contains(p))
        .map(|&(p, m)| (cassette_name(describe, p, m, it), p, m))
        .collect()
}

// ---- chat_content_spec.rb: text models --------------------------------------------------------

#[tokio::test]
async fn text_models_can_understand_text() {
    let ran = each(chat_rows("chat text models", "can understand text", &[]), |cassette, provider, model| async move {
        let mut chat = chat_for(&cassette, provider, model);
        let response = chat.ask_with("What's in this file?", vec![Attachment::new(fixture("ruby.txt"))]).await.map_err(|e| e.to_string())?;
        check(response.content().to_lowercase().contains("ruby"), format!("content {:?}", response.content))?;
        check(!response.content().contains("RubyLLM::Content"), "content object leaked")?;
        let first = &chat.messages()[0];
        check(first.content() == "What's in this file?", "first content")?;
        check(first.attachments[0].filename.as_deref() == Some("ruby.txt"), "filename")?;
        check(first.attachments[0].mime_type == "text/plain", "mime")?;

        let response = chat.ask_with("and in this one?", vec![Attachment::new(fixture("ruby.xml"))]).await.map_err(|e| e.to_string())?;
        check(response.content().to_lowercase().contains("ruby"), format!("second content {:?}", response.content))?;
        let third = &chat.messages()[2];
        check(third.content() == "and in this one?", "third content")?;
        check(third.attachments[0].filename.as_deref() == Some("ruby.xml"), "xml filename")?;
        check(third.attachments[0].mime_type == "application/xml", format!("xml mime {}", third.attachments[0].mime_type))?;
        Ok(cassette)
    })
    .await;
    assert_eq!(ran, 12);
}

#[tokio::test]
async fn text_models_can_understand_remote_text() {
    let ran = each(chat_rows("chat text models", "can understand remote text", &[]), |cassette, provider, model| async move {
        let mut chat = chat_for(&cassette, provider, model);
        // The recorded GET of https://www.ruby-lang.org/en/about/license.txt, served by the replay.
        let url = format!("{}/en/about/license.txt", cassette.server.uri());
        let response = chat.ask_with("What's in this file?", vec![Attachment::new(url)]).await.map_err(|e| e.to_string())?;
        let lower = response.content().to_lowercase();
        check(["ruby", "license", "copyright", "bsd"].iter().any(|w| lower.contains(w)), format!("content {lower:?}"))?;
        let first = &chat.messages()[0];
        check(first.attachments[0].filename.as_deref() == Some("license.txt"), "filename")?;
        check(first.attachments[0].mime_type == "text/plain", "mime")?;
        Ok(cassette)
    })
    .await;
    assert_eq!(ran, 12);
}

// ---- chat_request_options_spec.rb: with params ------------------------------------------------

fn expect_result_8(content: &str) -> Result<(), String> {
    let parsed: Value = serde_json::from_str(content).map_err(|e| format!("{e}: {content:?}"))?;
    check(parsed == json!({ "result": 8 }), format!("parsed {parsed}"))
}

const SQRT_PROMPT: &str = "What is the square root of 64? Answer with a JSON object with the key `result`.";

#[tokio::test]
async fn params_response_format() {
    let rows = chat_rows("chat with params", "supports response_format param", &["openai", "ollama", "deepseek", "mistral", "xai"]);
    let ran = each(rows, |cassette, provider, model| async move {
        let params = if matches!(provider, "openai" | "xai") {
            json!({ "text": { "format": { "type": "json_object" } } })
        } else {
            json!({ "response_format": { "type": "json_object" } })
        };
        let mut chat = chat_for(&cassette, provider, model).with_provider_options(params);
        let response = chat.ask(SQRT_PROMPT).await.map_err(|e| e.to_string())?;
        expect_result_8(response.content())?;
        Ok(cassette)
    })
    .await;
    assert_eq!(ran, 5);
}

#[tokio::test]
async fn params_gemini_response_schema() {
    let ran = each(chat_rows("chat with params", "supports responseSchema param", &["gemini"]), |cassette, provider, model| async move {
        let mut chat = chat_for(&cassette, provider, model).with_provider_options(json!({
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": { "type": "OBJECT", "properties": { "result": { "type": "NUMBER" } } }
            }
        }));
        let response = chat.ask(SQRT_PROMPT).await.map_err(|e| e.to_string())?;
        expect_result_8(response.content())?;
        Ok(cassette)
    })
    .await;
    assert_eq!(ran, 1);
}

#[tokio::test]
async fn params_perplexity_json_schema() {
    let rows = chat_rows("chat with params", "supports json_schema response_format param", &["perplexity"]);
    let ran = each(rows, |cassette, provider, model| async move {
        let mut chat = chat_for(&cassette, provider, model).with_provider_options(json!({
            "response_format": { "type": "json_schema", "json_schema": { "schema": {
                "type": "object", "properties": { "result": { "type": "number" } }, "required": ["result"]
            } } }
        }));
        let response = chat.ask(SQRT_PROMPT).await.map_err(|e| e.to_string())?;
        expect_result_8(response.content())?;
        Ok(cassette)
    })
    .await;
    assert_eq!(ran, 1);
}

/// Anthropic `service_tier` and OpenRouter `top_k`, steered to JSON by a leading `{` turn.
#[tokio::test]
async fn params_prefilled_json() {
    let mut rows = chat_rows("chat with params", "supports service_tier param", &["anthropic"]);
    rows.extend(chat_rows("chat with params", "supports top_k param", &["openrouter"]));
    let ran = each(rows, |cassette, provider, model| async move {
        let params = if provider == "anthropic" { json!({ "service_tier": "standard_only" }) } else { json!({ "top_k": 5 }) };
        let mut chat = chat_for(&cassette, provider, model).with_provider_options(params);
        chat.add_message(Message::user(SQRT_PROMPT));
        chat.add_message(Message::assistant("{"));
        let response = chat.generate().await.map_err(|e| e.to_string())?;
        expect_result_8(&format!("{{{}", response.content()))?;
        Ok(cassette)
    })
    .await;
    assert_eq!(ran, 2);
}

// ---- chat_error_spec.rb: real error scenarios -------------------------------------------------

fn human_readable(message: &str) -> Result<(), String> {
    let t = message.trim();
    check(!(t.starts_with('{') || t.starts_with('[')), format!("looks like JSON: {message}"))?;
    check(message.chars().next().is_some_and(|c| c.is_ascii_alphabetic()), format!("not capitalized text: {message}"))
}

#[tokio::test]
async fn real_errors_context_length_exceeded() {
    let ran = each(chat_rows("chat real error scenarios", "handles context length exceeded errors", &[]), |cassette, provider, model| async move {
        let mut chat = chat_for(&cassette, provider, model);
        if provider == "gpustack" {
            chat = chat.with_max_output_tokens(1);
            chat.add_message(Message::user("context ".repeat(10_000)));
        } else {
            // RubyLLM sends 'a' * 1_000_000 and VCR records it as `<MASSIVE_TEXT>` (the spec's
            // `filter_sensitive_data`), so the placeholder is what reproduces the recorded body.
            for _ in 0..5 {
                chat.add_message(Message::user("<MASSIVE_TEXT>"));
                chat.add_message(Message::assistant("<MASSIVE_TEXT>"));
            }
        }
        let err = match chat.ask("Hi").await {
            Ok(m) => return Err(format!("expected an error, got {:?}", m.content)),
            Err(e) => e,
        };
        if provider == "gpustack" {
            check(err.kind() == ErrorKind::ContextLengthExceeded, format!("expected ContextLengthExceeded, got {err:?}"))?;
        }
        human_readable(&err.to_string())?;
        Ok(cassette)
    })
    .await;
    assert_eq!(ran, 9);
}

// ---- chat_tools_spec.rb: error handling --------------------------------------------------------

struct Broken;

#[async_trait]
impl Tool for Broken {
    fn name(&self) -> String {
        "broken".into()
    }
    fn description(&self) -> String {
        "Gets current weather".into()
    }
    async fn execute(&self, _: Map<String, Value>, _: &ToolCall) -> Result<ToolResult, ToolError> {
        Err("This tool is broken".into())
    }
}

#[tokio::test]
async fn a_failing_tool_raises() {
    let cassette = start("chat_error_handling_raises_an_error_when_tool_execution_fails").await;
    // `RubyLLM.chat` with the default model.
    let mut chat = Chat::with_config(config_for(&cassette, "openai"), None, None, false)
        .unwrap()
        .with_tool(Broken)
        .with_tool_choice(rust_llm::ToolChoice::Required)
        .unwrap();
    let err = chat.ask("What is the weather?").await.unwrap_err();
    assert!(err.to_string().contains("This tool is broken"), "{err:?}");
    cassette.assert_all_matched().await;
}

// ---- chat_citations_spec.rb --------------------------------------------------------------------

fn anthropic(cassette: &Cassette) -> Chat {
    chat_for(cassette, "anthropic", "claude-haiku-4-5")
}

/// `response.content[citation.start_index...citation.end_index] == citation.text`.
fn span_matches(response: &Message, citation: &rust_llm::Citation) -> bool {
    let (Some(s), Some(e)) = (citation.start_index, citation.end_index) else { return false };
    let span: String = response.content().chars().skip(s as usize).take((e - s) as usize).collect();
    Some(span) == citation.text
}

#[tokio::test]
async fn with_citations_toggles() {
    let mut config = rust_llm::Config::default();
    config.set("anthropic_api_key", "test-key");
    let chat = Chat::with_config(Arc::new(config), Some("claude-haiku-4-5"), Some("anthropic"), false).unwrap();
    assert!(!chat.citations());
    let chat = chat.with_citations(true);
    assert!(chat.citations());
    let chat = chat.with_citations(false);
    assert!(!chat.citations());
}

#[tokio::test]
async fn citations_cite_text_documents() {
    let cassette = start("chat_citations_with_anthropic_claude-haiku-4-5_cites_text_documents_in_responses").await;
    let mut chat = anthropic(&cassette).with_citations(true);
    let response = chat.ask_with("Who created Ruby and when? Use the document.", vec![Attachment::new(fixture("facts.txt"))]).await.unwrap();
    let citation = response.citations.first().expect("citations");
    assert!(citation.cited_text.as_deref().is_some_and(|t| !t.is_empty()));
    assert_eq!(citation.title.as_deref(), Some("facts.txt"));
    assert_eq!(citation.source_index, Some(0));
    assert!(span_matches(&response, citation), "{citation:?} vs {:?}", response.content);
    cassette.assert_all_matched().await;
}

#[tokio::test]
async fn citations_cite_pdf_pages() {
    let cassette = start("chat_citations_with_anthropic_claude-haiku-4-5_cites_pdf_documents_with_page_numbers").await;
    let mut chat = anthropic(&cassette).with_citations(true);
    let response = chat.ask_with("What does the document say? Use the document.", vec![Attachment::new(fixture("sample.pdf"))]).await.unwrap();
    let citation = response.citations.first().expect("citations");
    assert!(citation.cited_text.as_deref().is_some_and(|t| !t.is_empty()));
    assert!(citation.start_page.is_some_and(|p| p >= 1), "{citation:?}");
    cassette.assert_all_matched().await;
}

/// `KnowledgeBase`: returns `RubyLLM::SearchResults`.
struct KnowledgeBase;

#[async_trait]
impl Tool for KnowledgeBase {
    fn description(&self) -> String {
        "Searches the company knowledge base".into()
    }
    fn parameters(&self) -> Vec<Parameter> {
        vec![Parameter::new("query").description("What to look for")]
    }
    async fn execute(&self, _: Map<String, Value>, _: &ToolCall) -> Result<ToolResult, ToolError> {
        let results = SearchResults::new(vec![json!({
            "title": "Ruby Facts",
            "url": "https://example.com/ruby-facts",
            "text": "The Ruby programming language was created by Yukihiro Matsumoto in 1993."
        })])?;
        Ok(results.into())
    }
}

#[tokio::test]
async fn citations_cite_search_result_tool_results() {
    let cassette = start("chat_citations_with_anthropic_claude-haiku-4-5_cites_tool_results_returned_as_search_results").await;
    let mut chat = anthropic(&cassette).with_tool(KnowledgeBase);
    let response = chat.ask("Who created Ruby? Search the knowledge base first and cite your sources.").await.unwrap();
    let citation = response.citations.first().expect("citations");
    assert_eq!(citation.url.as_deref(), Some("https://example.com/ruby-facts"));
    assert_eq!(citation.title.as_deref(), Some("Ruby Facts"));
    assert!(citation.cited_text.as_deref().is_some_and(|t| !t.is_empty()));
    assert!(span_matches(&response, citation), "{citation:?} vs {:?}", response.content);
    cassette.assert_all_matched().await;
}

#[tokio::test]
async fn citations_stream() {
    let cassette = start("chat_citations_with_anthropic_claude-haiku-4-5_streams_citations").await;
    let mut chat = anthropic(&cassette).with_citations(true);
    let mut chunks = Vec::new();
    chat.ask_later_with("Who created Ruby? Use the document.", vec![Attachment::new(fixture("facts.txt"))]).unwrap();
    let response = chat.complete_stream(|c| chunks.push(c.clone())).await.unwrap();
    assert!(chunks.iter().any(|c| !c.citations.is_empty()));
    assert!(response.citations.first().and_then(|c| c.cited_text.as_deref()).is_some_and(|t| !t.is_empty()));
    cassette.assert_all_matched().await;
}

#[tokio::test]
async fn citations_perplexity_search_results() {
    let cassette = start("chat_citations_with_perplexity_fast_returns_search_result_citations").await;
    let mut chat = chat_for(&cassette, "perplexity", "fast");
    let response = chat.ask("What is the Ruby programming language?").await.unwrap();
    assert!(response.citations.first().and_then(|c| c.url.as_deref()).is_some_and(|u| !u.is_empty()), "{:?}", response.citations);
    cassette.assert_all_matched().await;
}

#[tokio::test]
async fn citations_perplexity_search_results_streaming() {
    let cassette = start("chat_citations_with_perplexity_fast_returns_search_result_citations_when_streaming").await;
    let mut chat = chat_for(&cassette, "perplexity", "fast");
    let mut chunks = 0;
    let response = chat.ask_stream("What is the Ruby programming language?", |_| chunks += 1).await.unwrap();
    assert!(chunks > 0);
    assert!(response.citations.first().and_then(|c| c.url.as_deref()).is_some_and(|u| !u.is_empty()), "{:?}", response.citations);
    cassette.assert_all_matched().await;
}

#[tokio::test]
async fn citations_gemini_grounding() {
    let cassette = start("chat_citations_with_gemini_gemini-2_5-flash_returns_grounding_citations_when_search_is_enabled").await;
    let mut chat = chat_for(&cassette, "gemini", "gemini-2.5-flash").with_provider_options(json!({ "tools": [{ "google_search": {} }] }));
    let response = chat.ask("What is the latest stable version of Ruby?").await.unwrap();
    assert!(response.citations.first().and_then(|c| c.url.as_deref()).is_some_and(|u| !u.is_empty()), "{:?}", response.citations);
    cassette.assert_all_matched().await;
}

#[tokio::test]
async fn citations_openrouter_sonar() {
    let cassette = start("chat_citations_with_openrouter_perplexity_sonar_returns_search_result_citations").await;
    let mut chat = Chat::with_config(config_for(&cassette, "openrouter"), Some("perplexity/sonar"), Some("openrouter"), true).unwrap();
    let response = chat.ask("What is the Ruby programming language?").await.unwrap();
    assert!(response.citations.first().and_then(|c| c.url.as_deref()).is_some_and(|u| !u.is_empty()), "{:?}", response.citations);
    cassette.assert_all_matched().await;
}

#[tokio::test]
async fn citations_openai_search_model() {
    let cassette = start("chat_citations_with_openai_gpt-5-search-api_returns_url_citations_when_web_search_is_enabled").await;
    let mut chat = chat_for(&cassette, "openai", "gpt-5-search-api").with_provider_options(json!({ "web_search_options": {} }));
    let response = chat.ask("What is the latest stable version of Ruby? Cite your sources.").await.unwrap();
    let citation = response.citations.first().expect("citations");
    assert!(citation.url.as_deref().is_some_and(|u| !u.is_empty()));
    if citation.text.is_some() {
        assert!(span_matches(&response, citation), "{citation:?}");
    }
    cassette.assert_all_matched().await;
}

/// Collects `tracing` WARN events on this thread, standing in for `RubyLLM.logger.warn`.
struct WarnCollector(Arc<Mutex<Vec<String>>>);

impl tracing::Subscriber for WarnCollector {
    // Tests run in parallel: a callsite first hit with no collector set is cached as "never",
    // so ask on every event instead of caching the interest.
    fn register_callsite(&self, _: &'static tracing::Metadata<'static>) -> tracing::subscriber::Interest {
        tracing::subscriber::Interest::sometimes()
    }
    // Without this the global max level is recomputed from other threads' (absent) collectors and
    // can drop WARN events before they reach this one.
    fn max_level_hint(&self) -> Option<tracing::level_filters::LevelFilter> {
        Some(tracing::level_filters::LevelFilter::TRACE)
    }
    fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        struct Message<'a>(&'a mut String);
        impl tracing::field::Visit for Message<'_> {
            fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                if field.name() == "message" {
                    self.0.push_str(&format!("{value:?}"));
                }
            }
        }
        if *event.metadata().level() == tracing::Level::WARN {
            let mut text = String::new();
            event.record(&mut Message(&mut text));
            self.0.lock().unwrap().push(text);
        }
    }
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}

#[tokio::test]
async fn citations_warn_when_the_model_does_not_support_them() {
    let cassette = start("chat_citations_with_a_model_that_does_not_support_citations_warns_when_citations_are_requested").await;
    let warnings = Arc::new(Mutex::new(Vec::new()));
    let _guard = tracing::dispatcher::set_default(&tracing::Dispatch::new(WarnCollector(warnings.clone())));
    let mut chat = chat_for(&cassette, "openai", "gpt-5-nano").with_citations(true);
    let response = chat.ask("Say hi").await.unwrap();
    assert!(response.citations.is_empty());
    let warnings = warnings.lock().unwrap().clone();
    assert!(warnings.iter().any(|w| w.contains("does not support citations")), "{warnings:?}");
    cassette.assert_all_matched().await;
}

// ---- chat_provider_tools_spec.rb: web search ---------------------------------------------------

fn kinds(m: &Message) -> Vec<String> {
    m.server_tool_calls.iter().map(|c| c.kind.clone()).collect()
}

fn searching(cassette: &Cassette, provider: &str, model: &str) -> Chat {
    chat_for(cassette, provider, model).with_provider_tools([ProviderTool::alias("web_search")])
}

const SEARCH_AND_CITE: &str = "Search the web: what is the latest stable Ruby version? Cite your source.";
const SEARCH: &str = "Search the web: what is the latest stable Ruby version?";
const SAY_OK: &str = "Thanks. Now just say OK.";

#[tokio::test]
async fn web_search_anthropic_searches_cites_and_reports_usage() {
    let cassette = start("chat_web_search_with_anthropic_claude-haiku-4-5_searches_cites_and_reports_tool_usage").await;
    let mut chat = searching(&cassette, "anthropic", "claude-haiku-4-5");
    let response = chat.ask(SEARCH_AND_CITE).await.unwrap();
    let k = kinds(&response);
    assert!(k.contains(&"server_tool_use".into()) && k.contains(&"web_search_tool_result".into()), "{k:?}");
    assert!(!response.citations.is_empty());
    assert!(response.tokens().server_tool_use.is_some_and(|u| u.contains_key("web_search_requests")));
    assert!(response.raw_content.as_ref().is_some_and(Value::is_array));
    cassette.assert_all_matched().await;
}

#[tokio::test]
async fn web_search_anthropic_replays_search_turns() {
    let cassette = start("chat_web_search_with_anthropic_claude-haiku-4-5_replays_search_turns_so_the_conversation_can_continue").await;
    let mut chat = searching(&cassette, "anthropic", "claude-haiku-4-5");
    chat.ask(SEARCH).await.unwrap();
    let followup = chat.ask(SAY_OK).await.unwrap();
    assert!(!followup.content().is_empty());
    cassette.assert_all_matched().await;
}

#[tokio::test]
async fn web_search_anthropic_streams_and_reconstructs() {
    let cassette = start("chat_web_search_with_anthropic_claude-haiku-4-5_streams_search_turns_and_reconstructs_them").await;
    let mut chat = searching(&cassette, "anthropic", "claude-haiku-4-5");
    let mut chunks = 0;
    let response = chat.ask_stream(SEARCH, |_| chunks += 1).await.unwrap();
    assert!(chunks > 0);
    assert!(kinds(&response).contains(&"server_tool_use".into()), "{:?}", kinds(&response));
    assert!(response.raw_content.as_ref().is_some_and(Value::is_array));
    let followup = chat.ask(SAY_OK).await.unwrap();
    assert!(!followup.content().is_empty());
    cassette.assert_all_matched().await;
}

#[tokio::test]
async fn web_search_openai_records_tool_call_items() {
    let cassette = start("chat_web_search_with_openai_gpt-5_2_searches_and_records_the_tool_call_items").await;
    let mut chat = searching(&cassette, "openai", "gpt-5.2");
    let response = chat.ask(SEARCH_AND_CITE).await.unwrap();
    assert!(kinds(&response).contains(&"web_search_call".into()), "{:?}", kinds(&response));
    assert!(response.raw_content.as_ref().is_some_and(Value::is_array));
    let followup = chat.ask(SAY_OK).await.unwrap();
    assert!(!followup.content().is_empty());
    cassette.assert_all_matched().await;
}

#[tokio::test]
async fn web_search_gemini_grounds_and_exposes_queries() {
    let cassette = start("chat_web_search_with_gemini_gemini-3_5-flash_grounds_the_answer_and_exposes_the_queries_it_ran").await;
    let mut chat = searching(&cassette, "gemini", "gemini-3.5-flash");
    let response = chat.ask(SEARCH_AND_CITE).await.unwrap();
    assert!(kinds(&response).contains(&"google_search".into()), "{:?}", kinds(&response));
    assert!(!response.citations.is_empty());
    cassette.assert_all_matched().await;
}

#[tokio::test]
async fn web_search_openrouter_returns_citations_and_counters() {
    let cassette = start("chat_web_search_with_openrouter_openai_gpt-5_2_searches_transparently_returning_citations_and_usage_counters").await;
    let mut chat = searching(&cassette, "openrouter", "openai/gpt-5.2");
    let response = chat.ask(SEARCH_AND_CITE).await.unwrap();
    assert!(!response.citations.is_empty());
    assert!(response.tokens().server_tool_use.is_some_and(|u| u.contains_key("web_search_requests")), "{:?}", response.tokens());
    cassette.assert_all_matched().await;
}

#[tokio::test]
async fn web_search_xai_searches_cites_and_counts() {
    let cassette = start("chat_web_search_with_xai_grok-4_3_searches_cites_and_counts_the_sources_it_used").await;
    let mut chat = searching(&cassette, "xai", "grok-4.3");
    let response = chat.ask(SEARCH_AND_CITE).await.unwrap();
    assert!(!response.server_tool_calls.is_empty());
    assert!(!response.citations.is_empty());
    assert!(response.tokens().server_tool_use.is_some_and(|u| u.contains_key("num_server_side_tools_used")), "{:?}", response.tokens());
    assert!(response.raw_content.as_ref().is_some_and(Value::is_array));
    cassette.assert_all_matched().await;
}

#[tokio::test]
async fn web_search_xai_replays_search_turns() {
    let cassette = start("chat_web_search_with_xai_grok-4_3_replays_search_turns_so_the_conversation_can_continue").await;
    let mut chat = searching(&cassette, "xai", "grok-4.3");
    chat.ask(SEARCH).await.unwrap();
    let followup = chat.ask(SAY_OK).await.unwrap();
    assert!(!followup.content().is_empty());
    cassette.assert_all_matched().await;
}

// ---- chat_provider_tools_spec.rb: code execution -----------------------------------------------

const COMPUTE: &str = "Use code execution to compute 123456789 * 987654321 and report the exact product.";

#[tokio::test]
async fn code_execution_anthropic_returns_result_blocks() {
    let cassette = start("chat_code_execution_with_anthropic_claude-haiku-4-5_runs_code_server-side_and_returns_the_result_blocks").await;
    let mut chat = chat_for(&cassette, "anthropic", "claude-haiku-4-5").with_provider_tools([ProviderTool::alias("code_execution")]);
    let response = chat.ask(COMPUTE).await.unwrap();
    assert!(!response.server_tool_calls.is_empty());
    assert!(response.content().replace(',', "").contains("121932631112635269"), "{:?}", response.content);
    cassette.assert_all_matched().await;
}

#[tokio::test]
async fn code_execution_gemini_replays_the_turn() {
    let cassette = start("chat_code_execution_with_gemini_gemini-3_5-flash_runs_code_server-side_and_replays_the_turn").await;
    let mut chat = chat_for(&cassette, "gemini", "gemini-3.5-flash").with_provider_tools([ProviderTool::alias("code_execution")]);
    let response = chat.ask(COMPUTE).await.unwrap();
    assert!(kinds(&response).contains(&"executable_code".into()), "{:?}", kinds(&response));
    assert!(response.content().replace(',', "").contains("121932631112635269"), "{:?}", response.content);
    let followup = chat.ask(SAY_OK).await.unwrap();
    assert!(!followup.content().is_empty());
    cassette.assert_all_matched().await;
}

// ---- chat_provider_tools_spec.rb: responses protocol dialects ----------------------------------
// `response.raw.env.url.path` ends with `/responses`: the replay asserts every request path.

#[tokio::test]
async fn xai_chats_on_responses_by_default() {
    let cassette = start("chat_responses_protocol_dialects_with_xai_grok-4_3_chats_on_the_responses_endpoint_by_default").await;
    let mut chat = chat_for(&cassette, "xai", "grok-4.3");
    let response = chat.ask("What is 2 + 2? Just the number.").await.unwrap();
    assert!(response.content().contains('4'));
    assert!(response.thinking.is_some());
    let followup = chat.ask("Now multiply that by 3. Just the number.").await.unwrap();
    assert!(followup.content().contains("12"));
    cassette.assert_all_matched().await;
}

fn deepseek_responses(cassette: &Cassette) -> Chat {
    chat_for(cassette, "deepseek", "deepseek-v4-flash").with_protocol(ProtocolName::Responses)
}

#[tokio::test]
async fn deepseek_responses_reasoning_text() {
    let cassette = start("chat_responses_protocol_dialects_with_deepseek_deepseek-v4-flash_chats_with_reasoning_text_on_the_opt-in_responses_protocol").await;
    let mut chat = deepseek_responses(&cassette);
    let response = chat.ask("What is 2 + 2? Just the number.").await.unwrap();
    assert!(response.content().contains('4'));
    assert!(response.thinking.as_ref().and_then(|t| t.text.as_deref()).is_some_and(|t| !t.is_empty()));
    let followup = chat.ask("Multiply that answer by 3. Just the number.").await.unwrap();
    assert!(followup.content().contains("12"));
    cassette.assert_all_matched().await;
}

#[tokio::test]
async fn deepseek_responses_streams_reasoning() {
    let cassette = start("chat_responses_protocol_dialects_with_deepseek_deepseek-v4-flash_streams_reasoning_deltas").await;
    let mut chat = deepseek_responses(&cassette);
    let mut chunks = Vec::new();
    let response = chat.ask_stream("What is 2 + 2? Just the number.", |c| chunks.push(c.clone())).await.unwrap();
    assert!(chunks.iter().any(|c| c.thinking.is_some()));
    assert!(response.content().contains('4'));
    cassette.assert_all_matched().await;
}

/// `LookupNumber` from `providers/deepseek/responses_spec.rb`.
struct LookupNumber;

#[async_trait]
impl Tool for LookupNumber {
    fn description(&self) -> String {
        "Returns the current reference number".into()
    }
    async fn execute(&self, _: Map<String, Value>, _: &ToolCall) -> Result<ToolResult, ToolError> {
        Ok(json!(137).into())
    }
}

#[tokio::test]
async fn deepseek_responses_continues_after_a_tool_result() {
    let cassette = start("providers_deepseek_responses_continues_a_streamed_reasoning_conversation_after_a_tool_result").await;
    let mut chat = deepseek_responses(&cassette).with_thinking(ThinkingConfig::effort("low")).with_tool(LookupNumber);
    let mut chunks = Vec::new();
    let response = chat.ask_stream("Call lookup_number and tell me the reference number it returns.", |c| chunks.push(c.clone())).await.unwrap();
    assert!(response.content().contains("137"));
    let call = chat.messages().iter().find(|m| m.is_tool_call()).expect("tool call");
    assert!(call.thinking.as_ref().and_then(|t| t.text.as_deref()).is_some_and(|t| !t.is_empty()));
    assert!(chunks.iter().any(|c| c.thinking.is_some()));
    let types: Vec<String> = chat.render().unwrap()["input"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|i| i.get("type").and_then(Value::as_str).map(str::to_string))
        .collect();
    assert!(types.contains(&"reasoning".into()) && types.contains(&"function_call_output".into()), "{types:?}");
    assert!(chat.messages().iter().any(|m| m.role == Role::Tool));
    cassette.assert_all_matched().await;
}
