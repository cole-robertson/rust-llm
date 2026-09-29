//! Replays RubyLLM's own VCR cassettes (converted to JSON by `bin/convert-cassettes`) against the
//! Rust port. Each recorded request is served only if the port sends a JSON-equal body to the
//! same path, so a passing test means the port speaks exactly the wire format RubyLLM 2.0 does.

#![allow(dead_code)]

use std::sync::{Arc, Mutex};

use serde_json::Value;
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Interaction {
    pub method: String,
    pub uri: String,
    pub request_body: String,
    pub status: u16,
    pub response_headers: serde_json::Map<String, Value>,
    pub response_body: String,
    /// The exact bytes of a binary download (an image a URL attachment fetches).
    #[serde(default)]
    pub response_body_base64: Option<String>,
}

pub fn load(name: &str) -> Option<Vec<Interaction>> {
    let path = format!("{}/tests/cassettes/{name}.json", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(path).ok()?;
    // docs/PARITY.md's "replayed" column comes from this log (`bin/parity`).
    if let Ok(log) = std::env::var("RUST_LLM_CASSETTE_LOG") {
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(log)
        {
            let bin = std::env::current_exe()
                .ok()
                .and_then(|p| p.file_stem().map(|s| s.to_string_lossy().into_owned()))
                .unwrap_or_default();
            let bin = bin
                .rsplit_once('-')
                .map_or(bin.as_str(), |(b, _)| b)
                .to_string();
            let _ = f.write_all(format!("{name}\t{bin}\n").as_bytes()); // one write, so parallel tests don't interleave
        }
    }
    Some(serde_json::from_str(&text).expect("cassette json"))
}

/// Differences between the recorded and sent bodies, as JSON pointers.
pub fn diff(expected: &Value, actual: &Value, path: &str, out: &mut Vec<String>) {
    match (expected, actual) {
        (Value::Object(e), Value::Object(a)) => {
            for (k, ev) in e {
                match a.get(k) {
                    Some(av) => diff(ev, av, &format!("{path}/{k}"), out),
                    None => out.push(format!("missing {path}/{k} (expected {})", short(ev))),
                }
            }
            for k in a.keys().filter(|k| !e.contains_key(*k)) {
                out.push(format!("unexpected {path}/{k} = {}", short(&a[k])));
            }
        }
        (Value::Array(e), Value::Array(a)) => {
            if e.len() != a.len() {
                out.push(format!(
                    "{path}: expected {} items, got {}",
                    e.len(),
                    a.len()
                ));
            }
            for (i, (ev, av)) in e.iter().zip(a).enumerate() {
                diff(ev, av, &format!("{path}/{i}"), out);
            }
        }
        (Value::Number(e), Value::Number(a)) if e.as_f64() == a.as_f64() => {}
        (e, a) if e == a => {}
        (e, a) => out.push(format!("{path}: expected {} got {}", short(e), short(a))),
    }
}

fn short(v: &Value) -> String {
    let s = v.to_string();
    if s.len() > 160 {
        format!("{}…", &s[..160])
    } else {
        s
    }
}

/// Serves the cassette's interactions in order, recording any body mismatches.
pub struct Replay {
    interactions: Vec<Interaction>,
    next: Mutex<usize>,
    pub mismatches: Arc<Mutex<Vec<String>>>,
}

impl Respond for Replay {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let mut next = self.next.lock().unwrap();
        let Some(interaction) = self.interactions.get(*next) else {
            self.mismatches
                .lock()
                .unwrap()
                .push(format!("unexpected extra request to {}", request.url));
            return ResponseTemplate::new(599);
        };
        *next += 1;
        let recorded_path = interaction
            .uri
            .split_once("://")
            .map(|(_, r)| r.split_once('/').map(|(_, p)| p).unwrap_or(""))
            .unwrap_or("");
        let sent = format!(
            "{}{}",
            request.url.path().trim_start_matches('/'),
            request
                .url
                .query()
                .map(|q| format!("?{q}"))
                .unwrap_or_default()
        );
        let recorded = recorded_path.trim_start_matches('/');
        if !recorded.ends_with(&sent) && !sent.ends_with(recorded) {
            self.mismatches.lock().unwrap().push(format!(
                "request {}: path {sent} != recorded {recorded}",
                *next - 1
            ));
        }
        let expected: Value =
            serde_json::from_str(&interaction.request_body).unwrap_or(Value::Null);
        let actual: Value = serde_json::from_slice(&request.body).unwrap_or(Value::Null);
        let mut out = Vec::new();
        diff(&expected, &actual, "", &mut out);
        for d in out {
            self.mismatches
                .lock()
                .unwrap()
                .push(format!("request {}: {d}", *next - 1));
        }
        let content_type = interaction
            .response_headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case("content-type"))
            .and_then(|(_, v)| v.as_str())
            .unwrap_or("application/json")
            .to_string();
        let mut response = ResponseTemplate::new(interaction.status)
            .insert_header("content-type", content_type.as_str());
        // Gemini's resumable upload returns the URL to send the bytes to; point it at this server.
        if let Some(url) = interaction
            .response_headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case("x-goog-upload-url"))
            .and_then(|(_, v)| v.as_str())
        {
            let host = request
                .headers
                .get("host")
                .and_then(|v| v.to_str().ok())
                .unwrap_or_default();
            let here = format!("http://{host}");
            let rewritten = match url
                .split_once("://")
                .and_then(|(_, rest)| rest.split_once('/'))
            {
                Some((_, path)) => format!("{here}/{path}"),
                None => url.to_string(),
            };
            response = response.insert_header("x-goog-upload-url", rewritten.as_str());
        }
        let body = match &interaction.response_body_base64 {
            Some(b64) => base64::Engine::decode(&base64::engine::general_purpose::STANDARD, b64)
                .unwrap_or_default(),
            None => interaction.response_body.clone().into_bytes(),
        };
        response.set_body_raw(body, &content_type)
    }
}

pub struct Cassette {
    pub server: MockServer,
    pub mismatches: Arc<Mutex<Vec<String>>>,
    pub count: usize,
}

impl Cassette {
    pub async fn start(name: &str) -> Option<Cassette> {
        Cassette::start_serving(name, &[]).await
    }

    /// Like `start`, for a test that sends this server's URL where RubyLLM sent a real one (a
    /// URL attachment passed to the model as a link): `hosts` in the recorded bodies are
    /// rewritten to this server, so the comparison stays exact.
    pub async fn start_serving(name: &str, hosts: &[&str]) -> Option<Cassette> {
        let mut interactions = load(name)?;
        let count = interactions.len();
        let server = MockServer::start().await;
        for interaction in &mut interactions {
            for host in hosts {
                interaction.request_body = interaction.request_body.replace(host, &server.uri());
            }
        }
        let mismatches = Arc::new(Mutex::new(Vec::new()));
        Mock::given(wiremock::matchers::any())
            .respond_with(Replay {
                interactions,
                next: Mutex::new(0),
                mismatches: mismatches.clone(),
            })
            .mount(&server)
            .await;
        Some(Cassette {
            server,
            mismatches,
            count,
        })
    }

    /// Points every provider this test might use at the replay server.
    pub fn configure(&self, config: &mut rust_llm::Config, provider: &str) {
        let base = self.server.uri();
        let base = match provider {
            "openai" | "mistral" | "xai" | "ollama_cloud" => format!("{base}/v1"),
            "openrouter" => format!("{base}/api/v1"),
            "gemini" => format!("{base}/v1beta"),
            "hetzner" => format!("{base}/api/v1"),
            "ollama" | "gpustack" => format!("{base}/v1"),
            _ => base,
        };
        config.set(format!("{provider}_api_base"), base);
        config.set(format!("{provider}_api_key"), "test-key");
        config.max_retries = 0;
    }

    pub async fn assert_all_matched(&self) {
        let mismatches = self.mismatches.lock().unwrap().clone();
        assert!(
            mismatches.is_empty(),
            "request bodies differ from RubyLLM's:\n  {}",
            mismatches.join("\n  ")
        );
        let received = self
            .server
            .received_requests()
            .await
            .unwrap_or_default()
            .len();
        assert_eq!(
            received, self.count,
            "expected {} requests like RubyLLM made, sent {received}",
            self.count
        );
    }
}

pub fn config_for(cassette: &Cassette, provider: &str) -> Arc<rust_llm::Config> {
    let mut config = rust_llm::Config::default();
    cassette.configure(&mut config, provider);
    Arc::new(config)
}

/// The chat models `spec/support/models_to_test.rb` covers that this port implements.
pub const CHAT_MODELS: &[(&str, &str)] = &[
    ("anthropic", "claude-haiku-4-5"),
    ("deepseek", "deepseek-v4-flash"),
    ("gemini", "gemini-2.5-flash"),
    ("gpustack", "qwen3"),
    ("hetzner", "Qwen3.8-27B"),
    ("mistral", "mistral-small-latest"),
    ("ollama", "qwen3"),
    ("ollama_cloud", "gpt-oss:120b"),
    ("openai", "gpt-5-nano"),
    ("openrouter", "claude-haiku-4-5"),
    ("perplexity", "openai/gpt-5-mini"),
    ("xai", "grok-4-1-fast-non-reasoning"),
];

/// `example.full_description.parameterize(separator: '_')`.
pub fn cassette_name(describe: &str, provider: &str, model: &str, it: &str) -> String {
    let raw = format!("{describe} {provider}/{model} {it}");
    let mut out = String::new();
    let mut sep = false;
    for c in raw.to_lowercase().chars() {
        if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
            if sep && !out.is_empty() {
                out.push('_');
            }
            sep = false;
            out.push(c);
        } else {
            sep = true;
        }
    }
    out
}

/// `RubyLLM.chat(model:, provider:)` against the replay server. Local and self-hosted providers
/// aren't in the bundled registry, so they assume the model exists, as the Ruby specs do.
pub fn chat_for(cassette: &Cassette, provider: &str, model: &str) -> rust_llm::Chat {
    let assume = matches!(provider, "ollama" | "gpustack" | "ollama_cloud" | "hetzner");
    rust_llm::Chat::with_config(
        config_for(cassette, provider),
        Some(model),
        Some(provider),
        assume,
    )
    .expect("chat")
}

/// `each_model(MODELS) { it "#{provider}/#{model} ..." }`: runs `body` for each model in
/// `$models` that has a recorded cassette, and reports every failure together. Opt in with
/// `#[macro_use] mod support;`.
#[allow(unused_macros)]
macro_rules! each_model {
    ($models:expr, $describe:expr, $it:expr, |$chat:ident, $provider:ident, $model:ident| $body:block) => {{
        let mut failures = Vec::new();
        let mut ran = 0;
        for &($provider, $model) in $models {
            let name = crate::support::cassette_name($describe, $provider, $model, $it);
            let Some(cassette) = crate::support::Cassette::start(&name).await else {
                continue;
            };
            ran += 1;
            #[allow(unused_mut)]
            let mut $chat = crate::support::chat_for(&cassette, $provider, $model);
            let outcome: Result<(), String> = async { $body }.await;
            let replay = std::panic::AssertUnwindSafe(cassette.assert_all_matched());
            let replay = futures::FutureExt::catch_unwind(replay).await;
            if let Err(e) = outcome {
                failures.push(format!("{} {}: {e}", $provider, $model));
            } else if let Err(p) = replay {
                let msg = p.downcast_ref::<String>().cloned().unwrap_or_default();
                failures.push(format!("{} {}: {msg}", $provider, $model));
            }
        }
        assert!(ran > 0, "no cassettes found for {}", $it);
        assert!(
            failures.is_empty(),
            "{} of {ran} providers failed:\n{}",
            failures.len(),
            failures.join("\n\n")
        );
        eprintln!("{}: {ran} providers replayed", $it);
    }};
}
