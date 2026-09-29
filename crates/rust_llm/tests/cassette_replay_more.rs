//! Structured output, embeddings, and auth errors, replayed from RubyLLM's cassettes.

mod support;

use rust_llm::{Chat, EmbedOptions, Vectors, embed};
use serde_json::json;
use support::{Cassette, config_for};

fn person_schema() -> serde_json::Value {
    json!({
        "type": "object",
        "properties": { "name": { "type": "string" }, "age": { "type": "integer" } },
        "required": ["name", "age"],
        "additionalProperties": false
    })
}

const SCHEMA_MODELS: &[(&str, &str, &str)] = &[
    ("anthropic", "claude-haiku-4-5", "claude-haiku-4-5"),
    ("gemini", "gemini-3-flash-preview", "gemini-3-flash-preview"),
    ("mistral", "mistral-small-latest", "mistral-small-latest"),
    ("openai", "gpt-5-nano", "gpt-5-nano"),
    ("openrouter", "claude-haiku-4-5", "claude-haiku-4-5"),
    ("xai", "grok-4-1-fast-non-reasoning", "grok-4-1-fast-non-reasoning"),
];

async fn run<F, Fut>(names: Vec<(String, &'static str, &'static str)>, body: F)
where
    F: Fn(Cassette, &'static str, &'static str) -> Fut,
    Fut: std::future::Future<Output = Result<Cassette, String>>,
{
    let mut failures = Vec::new();
    let mut ran = 0;
    for (name, provider, model) in names {
        let Some(cassette) = Cassette::start(&name).await else { continue };
        ran += 1;
        match body(cassette, provider, model).await {
            Ok(cassette) => {
                let r = futures::FutureExt::catch_unwind(std::panic::AssertUnwindSafe(cassette.assert_all_matched())).await;
                if let Err(p) = r {
                    failures.push(format!("{provider}: {}", p.downcast_ref::<String>().cloned().unwrap_or_default()));
                }
            }
            Err(e) => failures.push(format!("{provider} {model}: {e}")),
        }
    }
    assert!(ran > 0, "no cassettes");
    assert!(failures.is_empty(), "{} of {ran} failed:\n{}", failures.len(), failures.join("\n\n"));
    eprintln!("{ran} replayed");
}

#[tokio::test]
async fn schema_returns_structured_output() {
    let names = SCHEMA_MODELS
        .iter()
        .map(|(p, m, slug)| {
            let slug = slug.replace('.', "_");
            (format!("chat_with_schema_with_{p}_{slug}_accepts_a_json_schema_and_returns_structured_output"), *p, *m)
        })
        .collect();
    run(names, |cassette, provider, model| async move {
        let mut chat = Chat::with_config(config_for(&cassette, provider), Some(model), Some(provider), false)
            .map_err(|e| e.to_string())?
            .with_schema(person_schema());
        let response = chat.ask("Generate a person named John who is 30 years old").await.map_err(|e| e.to_string())?;
        let parsed = response.parsed().map_err(|e| e.to_string())?.ok_or("no parsed")?;
        if parsed["name"] != "John" || parsed["age"] != 30 {
            return Err(format!("parsed {parsed}"));
        }
        Ok(cassette)
    })
    .await;
}

#[tokio::test]
async fn schema_can_be_removed_mid_conversation() {
    let names = SCHEMA_MODELS
        .iter()
        .map(|(p, m, slug)| {
            let slug = slug.replace('.', "_");
            (format!("chat_with_schema_with_{p}_{slug}_allows_removing_schema_mid-conversation"), *p, *m)
        })
        .collect();
    run(names, |cassette, provider, model| async move {
        let mut chat = Chat::with_config(config_for(&cassette, provider), Some(model), Some(provider), false)
            .map_err(|e| e.to_string())?
            .with_schema(person_schema());
        let r1 = chat.ask("Generate a person named Bob").await.map_err(|e| e.to_string())?;
        let parsed = r1.parsed().map_err(|e| e.to_string())?.ok_or("no parsed")?;
        if !parsed["age"].is_i64() {
            return Err(format!("age {parsed}"));
        }
        chat = chat.with_schema(serde_json::Value::Null);
        let r2 = chat.ask("Now just tell me about Ruby").await.map_err(|e| e.to_string())?;
        if !r2.content().contains("Ruby") {
            return Err("no Ruby".into());
        }
        Ok(cassette)
    })
    .await;
}

const EMBEDDING_MODELS: &[(&str, &str, &str, Option<i64>)] = &[
    ("gemini", "gemini-embedding-001", "gemini-embedding-001", Some(768)),
    ("mistral", "mistral-embed", "mistral-embed", None),
    ("openai", "text-embedding-3-small", "text-embedding-3-small", Some(768)),
];

#[tokio::test]
async fn embeds_a_single_text() {
    let names = EMBEDDING_MODELS
        .iter()
        .map(|(p, m, slug, _)| (format!("embedding_basic_functionality_{p}_{slug}_can_handle_a_single_text"), *p, *m))
        .collect();
    run(names, |cassette, provider, model| async move {
        let e = embed(
            "Ruby is a programmer's best friend",
            EmbedOptions { model: Some(model), provider: Some(provider), config: Some(config_for(&cassette, provider)), ..Default::default() },
        )
        .await
        .map_err(|e| e.to_string())?;
        match &e.vectors {
            Vectors::Single(v) if !v.is_empty() => {}
            other => return Err(format!("vectors {:?}", std::mem::discriminant(other))),
        }
        if e.model != model {
            return Err(e.model);
        }
        Ok(cassette)
    })
    .await;
}

#[tokio::test]
async fn embeds_multiple_texts() {
    let names = EMBEDDING_MODELS
        .iter()
        .map(|(p, m, slug, _)| (format!("embedding_basic_functionality_{p}_{slug}_can_handle_multiple_texts"), *p, *m))
        .collect();
    run(names, |cassette, provider, model| async move {
        let texts = vec!["Ruby".to_string(), "Python".into(), "JavaScript".into()];
        // The Ruby spec omits provider: here, so the registry picks it.
        let e = embed(texts, EmbedOptions { model: Some(model), config: Some(config_for(&cassette, provider)), ..Default::default() })
            .await
            .map_err(|e| e.to_string())?;
        match &e.vectors {
            Vectors::Batch(rows) if rows.len() == 3 => Ok(cassette),
            _ => Err("expected 3 vectors".into()),
        }
    })
    .await;
}

#[tokio::test]
async fn embeds_with_custom_dimensions() {
    let names = EMBEDDING_MODELS
        .iter()
        .filter(|(_, _, _, d)| d.is_some())
        .map(|(p, m, slug, _)| {
            (format!("embedding_basic_functionality_{p}_{slug}_can_handle_a_single_text_with_custom_dimensions"), *p, *m)
        })
        .collect();
    run(names, |cassette, provider, model| async move {
        let e = embed(
            "Ruby is a programmer's best friend",
            EmbedOptions {
                model: Some(model),
                provider: Some(provider),
                dimensions: Some(768),
                config: Some(config_for(&cassette, provider)),
                ..Default::default()
            },
        )
        .await
        .map_err(|e| e.to_string())?;
        match &e.vectors {
            Vectors::Single(v) if v.len() == 768 => Ok(cassette),
            Vectors::Single(v) => Err(format!("{} dims", v.len())),
            _ => Err("batch".into()),
        }
    })
    .await;
}

#[tokio::test]
async fn auth_errors_are_human_readable() {
    let models: &[(&str, &str, &str)] = &[
        ("anthropic", "claude-haiku-4-5", "claude-haiku-4-5"),
        ("deepseek", "deepseek-v4-flash", "deepseek-v4-flash"),
        ("gemini", "gemini-2.5-flash", "gemini-2_5-flash"),
        ("mistral", "mistral-small-latest", "mistral-small-latest"),
        ("openai", "gpt-5-nano", "gpt-5-nano"),
        ("openrouter", "claude-haiku-4-5", "claude-haiku-4-5"),
        ("perplexity", "openai/gpt-5-mini", "openai_gpt-5-mini"),
        ("xai", "grok-4-1-fast-non-reasoning", "grok-4-1-fast-non-reasoning"),
        ("hetzner", "Qwen3.8-27B", "qwen3_8-27b"),
        ("ollama_cloud", "gpt-oss:120b", "gpt-oss_120b"),
    ];
    let names = models
        .iter()
        .map(|(p, m, slug)| (format!("chat_error_handling_with_{p}_{slug}_raises_appropriate_auth_error"), *p, *m))
        .collect();
    run(names, |cassette, provider, model| async move {
        let assume = matches!(provider, "hetzner" | "ollama_cloud");
        let mut chat = Chat::with_config(config_for(&cassette, provider), Some(model), Some(provider), assume).map_err(|e| e.to_string())?;
        let err = match chat.ask("Hello").await {
            Ok(_) => return Err("expected an error".into()),
            Err(e) => e,
        };
        let message = err.to_string();
        if err.response().is_none() {
            return Err(format!("no response on {err:?}"));
        }
        if message.trim_start().starts_with('{') || !message.chars().next().is_some_and(|c| c.is_ascii_alphabetic()) {
            return Err(format!("not human readable: {message}"));
        }
        Ok(cassette)
    })
    .await;
}

/// `class PersonSchemaClass < Schematist::Schema; string :name; number :age; end`
#[derive(schemars::JsonSchema)]
#[allow(dead_code)]
struct PersonSchemaClass {
    name: String,
    age: f64,
}

#[tokio::test]
async fn typed_schema_matches_rubyllm_schema_dsl() {
    let names = SCHEMA_MODELS
        .iter()
        .map(|(p, m, slug)| {
            (format!("chat_with_schema_with_{p}_{slug}_accepts_schematist_schema_class_instances_and_returns_structured_output"), *p, *m)
        })
        .collect();
    run(names, |cassette, provider, model| async move {
        let mut chat = Chat::with_config(config_for(&cassette, provider), Some(model), Some(provider), false)
            .map_err(|e| e.to_string())?
            .with_schema_for::<PersonSchemaClass>();
        let response = chat.ask("Generate a person named Alice who is 28 years old").await.map_err(|e| e.to_string())?;
        let parsed = response.parsed().map_err(|e| e.to_string())?.ok_or("no parsed")?;
        if parsed["name"] != "Alice" {
            return Err(format!("parsed {parsed}"));
        }
        Ok(cassette)
    })
    .await;
}
