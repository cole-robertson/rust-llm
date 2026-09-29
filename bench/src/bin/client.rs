//! The RustLLM side of the benchmark. `bench/ruby/client.rb` is the same program in RubyLLM;
//! keep the two in step. Every case prints one JSON line; `bench/run.sh` collects them.
//!
//!   client <case> [key=value ...]     (base=http://127.0.0.1:8765)
//!
//! Cases:
//! - `first`: process start to the first answer (run as a fresh process each time)
//! - `overhead n=2000 delay=0`: sequential `ask`s on fresh chats; per-request latency minus the delay
//! - `concurrent chats=100 rounds=5 delay=50`: `chats` concurrent chats, each asking `rounds` times
//! - `stream chunks=1000 n=50`: streamed `ask`s with `chunks` deltas; time per chunk
//! - `tools n=200`: one `ask` that runs a 3-round tool loop (4 requests)
//! - `memory chats=100 delay=2000`: RSS with `chats` chats in flight at once
//! - `render messages=200 n=2000`: `Chat#render` of a 200-message history

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use rust_llm::{Chat, Config, Message, Parameter, Tool, ToolCall, ToolError, ToolResult};
use serde_json::{Map, Value, json};

struct Args(Vec<(String, String)>);

impl Args {
    fn get(&self, key: &str, default: u64) -> u64 {
        self.0.iter().find(|(k, _)| k == key).and_then(|(_, v)| v.parse().ok()).unwrap_or(default)
    }
    fn base(&self) -> String {
        self.0.iter().find(|(k, _)| k == "base").map(|(_, v)| v.clone()).unwrap_or_else(|| "http://127.0.0.1:8765".into())
    }
}

/// A config pointing Anthropic at the mock; the path carries the mock's delay and chunk knobs.
fn config(args: &Args, delay: u64, chunks: u64) -> Arc<Config> {
    let mut config = Config::default();
    config.set("anthropic_api_base", format!("{}/d/{delay}/n/{chunks}", args.base()));
    config.set("anthropic_api_key", "bench");
    config.max_retries = 0;
    Arc::new(config)
}

fn chat(config: &Arc<Config>) -> Chat {
    Chat::with_config(config.clone(), Some("claude-haiku-4-5"), Some("anthropic"), false).expect("chat")
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
    async fn execute(&self, args: Map<String, Value>, _: &ToolCall) -> Result<ToolResult, ToolError> {
        Ok(format!("Current weather at {}, {}: 15°C, Wind: 10 km/h", args["latitude"], args["longitude"]).into())
    }
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[rank.min(sorted.len() - 1)]
}

fn summary(mut samples_ms: Vec<f64>) -> Value {
    samples_ms.sort_by(|a, b| a.total_cmp(b));
    let mean = samples_ms.iter().sum::<f64>() / samples_ms.len().max(1) as f64;
    json!({
        "n": samples_ms.len(),
        "p50_ms": percentile(&samples_ms, 50.0),
        "p99_ms": percentile(&samples_ms, 99.0),
        "mean_ms": mean,
    })
}

fn rss_kib() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| s.lines().find(|l| l.starts_with("VmRSS:")).and_then(|l| l.split_whitespace().nth(1)?.parse().ok()))
        .unwrap_or(0)
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

#[tokio::main]
async fn main() {
    let started = Instant::now();
    let mut argv = std::env::args().skip(1);
    let case = argv.next().unwrap_or_default();
    let args = Args(argv.filter_map(|a| a.split_once('=').map(|(k, v)| (k.to_string(), v.to_string()))).collect());

    let result = match case.as_str() {
        "first" => {
            let config = config(&args, 0, 1);
            let answer = chat(&config).ask("What's 2 + 2?").await.expect("ask");
            assert_eq!(answer.content(), "2 + 2 = 4");
            json!({ "first_answer_ms": ms(started.elapsed()), "rss_kib": rss_kib() })
        }
        "overhead" => {
            let delay = args.get("delay", 0);
            let config = config(&args, delay, 1);
            let n = args.get("n", 2000) as usize;
            for _ in 0..50 {
                chat(&config).ask("What's 2 + 2?").await.expect("warmup");
            }
            let mut samples = Vec::with_capacity(n);
            for _ in 0..n {
                let t = Instant::now();
                let answer = chat(&config).ask("What's 2 + 2?").await.expect("ask");
                samples.push(ms(t.elapsed()) - delay as f64);
                assert_eq!(answer.content(), "2 + 2 = 4");
            }
            let mut s = summary(samples);
            s["rss_kib"] = rss_kib().into();
            s
        }
        "concurrent" => {
            let delay = args.get("delay", 50);
            let chats = args.get("chats", 100) as usize;
            let rounds = args.get("rounds", 5) as usize;
            // strip_raw=1 drops each message's raw HTTP response after it arrives (diagnostic for
            // the O(n^2) `RawResponse::request_body` growth; see docs/BENCHMARK.md).
            let strip_raw = args.get("strip_raw", 0) == 1;
            chat(&config(&args, 0, 1)).ask("What's 2 + 2?").await.expect("warmup");
            let config = config(&args, delay, 1);
            let t = Instant::now();
            let tasks: Vec<_> = (0..chats)
                .map(|_| {
                    let config = config.clone();
                    tokio::spawn(async move {
                        let mut chat = chat(&config);
                        for _ in 0..rounds {
                            chat.ask("What's 2 + 2?").await.expect("ask");
                            if strip_raw {
                                chat.messages_mut().iter_mut().for_each(|m| m.raw = None);
                            }
                        }
                        // Cost of rendering the finished conversation, to compare with `render`.
                        let t = Instant::now();
                        std::hint::black_box(chat.render().expect("render"));
                        (chat.messages().len(), ms(t.elapsed()))
                    })
                })
                .collect();
            let mut messages = 0;
            let mut final_render_ms = 0.0;
            for task in tasks {
                let (count, render_ms) = task.await.expect("task");
                messages += count;
                final_render_ms = render_ms;
            }
            assert_eq!(messages, chats * rounds * 2);
            let elapsed = t.elapsed().as_secs_f64();
            let requests = chats * rounds;
            json!({
                "requests": requests,
                "rounds": rounds,
                "elapsed_s": elapsed,
                "req_per_s": requests as f64 / elapsed,
                "ideal_req_per_s": chats as f64 * 1000.0 / delay.max(1) as f64,
                "rss_kib": rss_kib(),
                "strip_raw": strip_raw,
                "final_render_ms": final_render_ms,
            })
        }
        "stream" => {
            let chunks = args.get("chunks", 1000);
            let n = args.get("n", 50) as usize;
            let config = config(&args, 0, chunks);
            for _ in 0..3 {
                chat(&config).ask_stream("Count from 1 to 3", |_| {}).await.expect("warmup");
            }
            let mut samples = Vec::with_capacity(n);
            let mut seen = 0;
            for _ in 0..n {
                let t = Instant::now();
                let answer = chat(&config)
                    .ask_stream("Count from 1 to 3", |chunk| {
                        if !chunk.content().is_empty() {
                            seen += 1;
                        }
                    })
                    .await
                    .expect("stream");
                samples.push(ms(t.elapsed()) * 1000.0 / chunks as f64);
                assert_eq!(answer.content().len(), 5 * chunks as usize);
            }
            assert_eq!(seen, (n as u64 * chunks) as usize);
            let mut s = summary(samples);
            s["unit"] = "us_per_chunk".into();
            s
        }
        "tools" => {
            let n = args.get("n", 200) as usize;
            let config = config(&args, 0, 1);
            let calls = Arc::new(AtomicUsize::new(0));
            let run = |config: Arc<Config>, calls: Arc<AtomicUsize>| async move {
                let mut chat = chat(&config).with_tool(Weather).before_tool_call(move |_| {
                    calls.fetch_add(1, Ordering::Relaxed);
                });
                let answer = chat.ask("What's the weather in Berlin? (52.5200, 13.4050)").await.expect("ask");
                assert!(answer.content().starts_with("The current weather in Berlin"));
                assert_eq!(chat.messages().len(), 8);
            };
            for _ in 0..20 {
                run(config.clone(), calls.clone()).await;
            }
            let mut samples = Vec::with_capacity(n);
            for _ in 0..n {
                let t = Instant::now();
                run(config.clone(), calls.clone()).await;
                samples.push(ms(t.elapsed()));
            }
            assert_eq!(calls.load(Ordering::Relaxed), (n + 20) * 3);
            summary(samples)
        }
        "memory" => {
            let chats = args.get("chats", 100) as usize;
            let delay = args.get("delay", 2000);
            // Warm the registry and HTTP stack, then measure with every chat in flight.
            chat(&config(&args, 0, 1)).ask("What's 2 + 2?").await.expect("warmup");
            let config = config(&args, delay, 1);
            let baseline = rss_kib();
            let tasks: Vec<_> = (0..chats)
                .map(|_| {
                    let config = config.clone();
                    tokio::spawn(async move { chat(&config).ask("What's 2 + 2?").await.expect("ask").content().len() })
                })
                .collect();
            tokio::time::sleep(Duration::from_millis(delay / 2)).await;
            let in_flight = rss_kib();
            for task in tasks {
                task.await.expect("task");
            }
            json!({ "chats": chats, "baseline_rss_kib": baseline, "in_flight_rss_kib": in_flight })
        }
        "render" => {
            let n = args.get("n", 2000) as usize;
            let count = args.get("messages", 200) as usize;
            let config = config(&args, 0, 1);
            let mut chat = chat(&config).with_instructions("You are a helpful assistant.").with_tool(Weather);
            for i in 0..count / 2 {
                chat.add_message(Message::user(format!("Question {i}: what's the weather like in city number {i} today?")));
                chat.add_message(Message::assistant(format!(
                    "Answer {i}: it is sunny with a light breeze, around {} degrees Celsius.",
                    10 + i % 20
                )));
            }
            let bytes = serde_json::to_vec(&chat.render().expect("render")).expect("json").len();
            for _ in 0..50 {
                std::hint::black_box(serde_json::to_vec(&chat.render().expect("render")).expect("json"));
            }
            let mut samples = Vec::with_capacity(n);
            for _ in 0..n {
                let t = Instant::now();
                std::hint::black_box(serde_json::to_vec(&chat.render().expect("render")).expect("json"));
                samples.push(ms(t.elapsed()));
            }
            let mut s = summary(samples);
            s["payload_bytes"] = bytes.into();
            s
        }
        other => panic!("unknown case {other:?}"),
    };
    println!("{}", json!({ "impl": "rust", "case": case, "result": result }));
}
