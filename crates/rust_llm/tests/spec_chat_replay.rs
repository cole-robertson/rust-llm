//! Cassette replays for `chat_streaming_spec.rb` (token usage with and without streaming) and
//! `embedding_spec.rb` (multiple texts with custom dimensions, single-string arrays). Each replay
//! asserts RubyLLM's recorded request bodies (JSON-equal) and request counts.

mod support;

use rust_llm::{EmbedOptions, Message, ThinkingConfig, Tokens, Vectors, embed};
use support::{Cassette, config_for};

/// `CHAT_MODELS` rows this port implements, with `token_model` swapped in for OpenAI and
/// Perplexity (`model_for(provider, :temperature)`), as the Ruby example does.
const TOKEN_MODELS: &[(&str, &str)] = &[
    ("anthropic", "claude-haiku-4-5"),
    ("deepseek", "deepseek-v4-flash"),
    ("gemini", "gemini-2.5-flash"),
    ("gpustack", "qwen3"),
    ("hetzner", "Qwen3.8-27B"),
    ("mistral", "mistral-small-latest"),
    ("ollama", "qwen3"),
    ("ollama_cloud", "gpt-oss:120b"),
    ("openai", "gpt-4.1-nano"),
    ("openrouter", "claude-haiku-4-5"),
    ("perplexity", "perplexity/sonar"),
    ("xai", "grok-4-1-fast-non-reasoning"),
];

/// `Tokens#to_h`: the components RubyLLM reports (it has no `reported_cost` key).
fn to_h(tokens: &Tokens) -> Tokens {
    Tokens { reported_cost: None, ..tokens.clone() }
}

/// `chunks.reduce({}) { |usage, chunk| usage.merge(chunk.tokens.to_h) }`: each component's last
/// reported value.
fn merged(chunks: &[Message]) -> Tokens {
    let mut usage = Tokens::default();
    for t in chunks.iter().map(|c| &c.tokens) {
        usage.input = t.input.or(usage.input);
        usage.output = t.output.or(usage.output);
        usage.cache_read = t.cache_read.or(usage.cache_read);
        usage.cache_write = t.cache_write.or(usage.cache_write);
        usage.thinking = t.thinking.or(usage.thinking);
        usage.server_tool_use = t.server_tool_use.clone().or(usage.server_tool_use);
    }
    usage
}

fn prompt_token_count(t: &Tokens) -> i64 {
    t.input.unwrap_or(0) + t.cache_read.unwrap_or(0) + t.cache_write.unwrap_or(0)
}

/// `visible_output_token_count`: `nil` when thinking came back without a thinking count.
fn visible_output_token_count(m: &Message) -> Option<i64> {
    let t = m.tokens();
    if m.thinking.is_some() && t.thinking.is_none() {
        return None;
    }
    Some(t.output? - t.thinking.unwrap_or(0))
}

fn check(ok: bool, what: impl Into<String>) -> Result<(), String> {
    if ok { Ok(()) } else { Err(what.into()) }
}

/// One replay of the example: a streamed ask, then the same ask on a fresh chat without streaming.
async fn token_usage_example(cassette: &Cassette, provider: &str, model: &str) -> Result<(), String> {
    let basic = || {
        let chat = support::chat_for(cassette, provider, model).with_temperature(0.0);
        // DeepSeek ignores temperature while thinking is enabled.
        if provider == "deepseek" { chat.with_thinking(ThinkingConfig::off()) } else { chat }
    };
    let prompt = "Reply with exactly: 1, 2, 3";
    let mut chunks = Vec::new();
    let stream_message = basic().ask_stream(prompt, |c| chunks.push(c.clone())).await.map_err(|e| e.to_string())?;
    let sync_message = basic().ask(prompt).await.map_err(|e| e.to_string())?;

    check(stream_message.content().trim() == "1, 2, 3", format!("stream content {:?}", stream_message.content()))?;
    check(sync_message.content().trim() == stream_message.content().trim(), format!("sync content {:?}", sync_message.content()))?;
    for message in [&stream_message, &sync_message] {
        let t = message.tokens();
        for (component, count) in [("input", t.input), ("cache_read", t.cache_read), ("cache_write", t.cache_write), ("thinking", t.thinking)] {
            check(count.is_none_or(|c| c >= 0), format!("{component} {count:?}"))?;
        }
        check(prompt_token_count(&t) > 0, format!("prompt tokens {t:?}"))?;
        check(t.output.is_some_and(|o| o > 0), format!("output {t:?}"))?;
        check(t.output.unwrap_or(0) >= t.thinking.unwrap_or(0), format!("output < thinking {t:?}"))?;
    }
    let chunk_usage = merged(&chunks);
    check(to_h(&stream_message.tokens()) == chunk_usage, format!("stream {:?} vs chunks {chunk_usage:?}", stream_message.tokens()))?;
    if let (Some(sync), Some(stream)) = (visible_output_token_count(&sync_message), visible_output_token_count(&stream_message)) {
        check((sync - stream).abs() <= 2, format!("visible output {sync} vs {stream}"))?;
    }
    Ok(())
}

// spec: chat_streaming_spec.rb:39 #{provider}/#{token_model} reports token usage with and without streaming
#[tokio::test]
async fn reports_token_usage_with_and_without_streaming() {
    let it = "reports token usage with and without streaming";
    let mut failures = Vec::new();
    let mut ran = 0;
    for &(provider, model) in TOKEN_MODELS {
        let name = support::cassette_name("chat streaming responses", provider, model, it);
        let Some(cassette) = Cassette::start(&name).await else { continue };
        ran += 1;
        if let Err(e) = token_usage_example(&cassette, provider, model).await {
            failures.push(format!("{provider} {model}: {e}"));
            continue;
        }
        let replay = futures::FutureExt::catch_unwind(std::panic::AssertUnwindSafe(cassette.assert_all_matched())).await;
        if let Err(p) = replay {
            failures.push(format!("{provider} {model}: {}", p.downcast_ref::<String>().cloned().unwrap_or_default()));
        }
    }
    assert!(ran > 0, "no cassettes for {it}");
    assert!(failures.is_empty(), "{} of {ran} providers failed:\n{}", failures.len(), failures.join("\n\n"));
    eprintln!("{it}: {ran} providers replayed");
}

/// `EMBEDDING_MODELS` rows this port implements: (provider, model, cassette slug, dimensions).
/// `model_info.fetch(:dimensions, 768)`: Mistral declares `nil`, so it skips the dimensions case.
const EMBEDDING_MODELS: &[(&str, &str, &str, Option<i64>)] = &[
    ("gemini", "gemini-embedding-001", "gemini-embedding-001", Some(768)),
    ("mistral", "mistral-embed", "mistral-embed", None),
    ("openai", "text-embedding-3-small", "text-embedding-3-small", Some(768)),
    ("openrouter", "openai/text-embedding-3-small", "openai_text-embedding-3-small", Some(768)),
];

async fn replay_embeddings<F, Fut>(it: &str, rows: impl Iterator<Item = &'static (&'static str, &'static str, &'static str, Option<i64>)>, body: F)
where
    F: Fn(Cassette, &'static str, &'static str, Option<i64>) -> Fut,
    Fut: std::future::Future<Output = Result<Cassette, String>>,
{
    let mut failures = Vec::new();
    let mut ran = 0;
    for &(provider, model, slug, dimensions) in rows {
        let name = format!("embedding_basic_functionality_{provider}_{slug}_{it}");
        let Some(cassette) = Cassette::start(&name).await else { continue };
        ran += 1;
        match body(cassette, provider, model, dimensions).await {
            Ok(cassette) => {
                let replay = futures::FutureExt::catch_unwind(std::panic::AssertUnwindSafe(cassette.assert_all_matched())).await;
                if let Err(p) = replay {
                    failures.push(format!("{provider}: {}", p.downcast_ref::<String>().cloned().unwrap_or_default()));
                }
            }
            Err(e) => failures.push(format!("{provider} {model}: {e}")),
        }
    }
    assert!(ran > 0, "no cassettes for {it}");
    assert!(failures.is_empty(), "{} of {ran} failed:\n{}", failures.len(), failures.join("\n\n"));
    eprintln!("{it}: {ran} replayed");
}

// spec: embedding_spec.rb:39 #{provider}/#{model} can handle multiple texts with custom dimensions
#[tokio::test]
async fn embeds_multiple_texts_with_custom_dimensions() {
    let rows = EMBEDDING_MODELS.iter().filter(|(_, _, _, d)| d.is_some());
    replay_embeddings("can_handle_multiple_texts_with_custom_dimensions", rows, |cassette, provider, model, dimensions| async move {
        let texts = vec!["Ruby".to_string(), "Python".into(), "JavaScript".into()];
        let options = EmbedOptions {
            model: Some(model),
            provider: Some(provider),
            dimensions,
            config: Some(config_for(&cassette, provider)),
            ..Default::default()
        };
        let e = embed(texts, options).await.map_err(|e| e.to_string())?;
        let Vectors::Batch(rows) = &e.vectors else { return Err("expected an array of vectors".into()) };
        let expected = dimensions.unwrap_or_default() as usize;
        check(rows.iter().all(|v| v.len() == expected), format!("lengths {:?}", rows.iter().map(Vec::len).collect::<Vec<_>>()))?;
        Ok(cassette)
    })
    .await;
}

// spec: embedding_spec.rb:50 #{provider}/#{model} handles single-string arrays consistently
#[tokio::test]
async fn handles_single_string_arrays_consistently() {
    replay_embeddings("handles_single-string_arrays_consistently", EMBEDDING_MODELS.iter(), |cassette, provider, model, _| async move {
        let options = EmbedOptions { model: Some(model), provider: Some(provider), config: Some(config_for(&cassette, provider)), ..Default::default() };
        let e = embed(vec!["Ruby is great".to_string()], options).await.map_err(|e| e.to_string())?;
        let Vectors::Batch(rows) = &e.vectors else { return Err("a one-string array must still give an array of vectors".into()) };
        check(rows.len() == 1 && !rows[0].is_empty(), format!("{} vectors", rows.len()))?;
        Ok(cassette)
    })
    .await;
}
