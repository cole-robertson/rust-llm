# Benchmark: RustLLM vs RubyLLM 2.0

Both libraries talk to the same **local mock Anthropic server**, which serves the responses RubyLLM
recorded in its own VCR cassettes, so the network and the model are not variables. Each case runs
the same program in both languages (`bench/src/bin/client.rs`, `bench/ruby/client.rb`): same
model, same prompts, same tool, and the same checks on every answer.

**Headline:** once a request is in flight, the library's own work is small in both languages.
RustLLM adds **0.08 ms** per request vs **0.25 ms** (RubyLLM + YJIT). It handles a streamed chunk
**7×** faster, and a 3-round tool loop **2.7×** faster. In a Rust program a chat is ready in
**12 ms** from exec; `bundle exec ruby` with RubyLLM takes **411 ms**. The gap only matters for
throughput when many chats share one process. At 1,000 concurrent chats behind a 50 ms model,
RustLLM completes **13,295 req/s** vs **2,680** (Async fibers) and **2,008** (threads).

**Where Ruby is close or wins:**
- Rendering a 200-message payload: 0.17 vs 0.22 ms (1.3×).
- A handful of chats: with one chat, all three reach 85–95% of the mock's ideal rate, within 12%
  of each other. That's where most applications live: one LLM call takes seconds, and 0.2 ms of
  library time is noise.
- **Memory at 1,000 concurrent chats: RubyLLM on fibers uses less (91 MiB vs 148 MiB)**. See
  [Memory](#memory).
- Compile time, which Ruby doesn't have. See [Build cost](#build-cost).

The benchmark also found a real RustLLM bug: long conversations were quadratic in time and memory.
It was fixed before these numbers were taken; see [Long conversations](#long-conversations).

## Setup

| | RustLLM | RubyLLM |
|---|---|---|
| Version | 2.0.0 (this repo, 2026-09-29) | 2.0.0 (`upstream/` = `crmne/ruby_llm@1e91b30`) |
| Runtime | rustc 1.98.1, `--release`, default allocator | Ruby 3.4.10, YJIT on (and off, as `ruby` below) |
| HTTP | reqwest 0.12 (hyper), one `Client` per `Chat` | Faraday 2.14.4 + net_http adapter, one connection per `Chat` |
| Concurrency | tokio multi-thread runtime, one task per chat | `async` 2.46 fibers (RubyLLM's [recommended model](../upstream/docs/_advanced/async.md)), or one `Thread` per chat |
| Gems | | json 3.0.2, faraday-retry 2.4.0, zeitwerk 2.8.3 (`bench/ruby/Gemfile.lock`) |

- **Host:** `framework`, AMD Ryzen AI 9 HX 370 (4 Zen 5 cores at 5.2 GHz + 8 Zen 5c cores at
  3.3 GHz, SMT), 93 GiB RAM, Arch Linux, kernel 7.2.3. No other benchmark ran at the same time.
  Other agents' builds on the box are the likely cause of the 1-minute load average of about 4.
- **Pinning:** each client runs on the four fast cores (`taskset -c 0-3`), and the mock on cores
  4–7, so they never share a core.
- **Mock** (`bench/src/bin/mock.rs`, axum): the path carries a delay and a chunk count. Plain
  requests get the recorded `2 + 2 = 4` message. Requests with tools get a recorded `tool_use`
  until they carry 3 tool results, then the recorded final answer. `stream: true` gets the
  recorded SSE stream, with its text delta repeated N times.
- **Retries** are off in both (`max_retries = 0`), and neither logs requests at the default level.
- **Repeats:** the whole suite ran 5 times. Every cell is the **median across the 5 runs, with
  (min–max)**. Cold start is 50 fresh processes per implementation.
- **Correctness:** every sample checks its answer (content, message count, tool call count,
  chunk count), so a case that returned the wrong thing would abort rather than report a time.
- Raw output: `bench/results/2026-09-29/raw.jsonl`. Reproduce with `bench/run.sh`
  (`QUICK=1` for a 2-minute smoke run); it prints `summary.md`.

## Latency

"Overhead" is the full `ask` round trip against a mock that answers at once. That covers
building the chat, rendering and serializing the request, HTTP over loopback, parsing, and
updating the message list and usage ledger. p50/p99 are over 2,000 sequential asks per run, each
on a fresh chat.

| Case | RustLLM | RubyLLM + YJIT | RubyLLM (interpreter) | Rust ÷ Ruby+YJIT |
|---|---:|---:|---:|---:|
| Cold start: process exec → first answer, inside the program | **10.1 ms** (p90 10.7) | 251 ms (p90 257) | 215 ms (p90 224) | 25× |
| Cold start: same, wall clock incl. `bundle exec` (10 runs) | **12 ms** | 411 ms | | 34× |
| Per-request overhead, p50 | **0.080 ms** (0.078–0.081) | 0.246 ms (0.242–0.258) | 0.410 ms | 3.1× |
| Per-request overhead, p99 | **0.119 ms** (0.112–0.131) | 0.742 ms (0.716–0.912) | 0.866 ms | 6.2× |
| Streaming, per chunk (1,000-chunk stream), p50 | **2.7 µs** (2.70–2.78) | 20.0 µs (19.97–21.64) | 32.8 µs | 7.4× |
| Tool loop: 3 rounds (4 requests, 3 tool runs), p50 | **0.38 ms** (0.35–0.39) | 1.05 ms (1.03–1.09) | 1.68 ms | 2.7× |
| Tool loop, p99 | **0.53 ms** (0.42–0.55) | 2.85 ms (2.55–4.16) | 3.78 ms | 5.4× |
| Render a 200-message history to JSON (25 KB), p50 | **0.171 ms** (0.170–0.174) | 0.221 ms (0.220–0.228) | 0.432 ms | **1.3×** |
| Render, p99 | **0.183 ms** | 0.480 ms | 1.080 ms | 2.6× |

- **Cold start.** The in-program clock starts at the first line of each client: before
  `require "bundler/setup"` in Ruby, and at `main` in Rust. Most of Ruby's 251 ms is `require "ruby_llm"`
  (Zeitwerk, Faraday, the model registry); the first request itself is fast. Starting `bundle
  exec` adds the rest of the 411 ms wall clock. YJIT costs ~35 ms here, which is why the
  interpreter wins this row. A long-running Rails process pays this once at boot, so it matters
  for CLIs, scripts, and serverless.
- **Render is the closest case.** It is JSON building, and Ruby's C `json` generator is fast.
  RustLLM builds a `serde_json::Value` tree and then serializes it; RubyLLM builds Hashes and
  serializes them in C.
- **Ruby's p99s** are 3–5× its p50s: GC pauses. Rust's p99 stays within 1.5× of p50.
- **Streaming** is where per-chunk work adds up: a 4,000-chunk answer costs 11 ms of CPU
  in Rust and 80 ms in Ruby. That's still small next to the seconds the model takes to write it.

## Throughput: concurrent chats

N chats run at the same time. Each asks 5 times in sequence, and the mock waits 50 ms before
every answer (a fast model). "Ideal" is N ÷ 50 ms: the rate if the library took no time.

| Chats | Ideal req/s | RustLLM (tokio) | RubyLLM + YJIT, Async fibers | RubyLLM + YJIT, threads | Rust ÷ fibers |
|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 19 (19–19) | 17 (17–17) | 17 (17–18) | 1.1× |
| 10 | 200 | 191 (189–192) | 147 (145–148) | 132 (129–137) | 1.3× |
| 100 | 2,000 | 1,775 (1,755–1,792) | 1,090 (981–1,146) | 939 (937–957) | 1.6× |
| 1,000 | 20,000 | **13,295** (13,200–13,716) | 2,680 (2,659–2,749) | 2,008 (2,000–2,044) | **5.0×** |

- With one chat all three are within 12% of each other: the 50 ms "model" is the bottleneck,
  as it is in production. At 10 chats Ruby is already 23% (fibers) and 31% (threads) behind.
- At 100 chats Ruby on fibers reaches 55% of ideal; at 1,000 it plateaus at ~2,700 req/s
  (~0.37 ms per request). The GVL keeps Ruby code on one core whether it runs on fibers or
  threads, and fibers beat threads by 30%. RustLLM spreads the work over the 4 pinned cores and
  reaches 66% of ideal at 1,000. It doesn't reach 100% because each `Chat` sets up its own
  `reqwest::Client` and connection. That's likely the remaining cost, but it wasn't profiled.
- A Rails app can run several Puma or Solid Queue processes to use more cores. This is
  per-process throughput.

## Memory

RSS (MiB) of the client process. "In flight" is sampled halfway through a 2-second mock delay,
with every chat waiting on its answer at once.

| Chats in flight | RustLLM | RubyLLM, fibers | RubyLLM, threads |
|---:|---:|---:|---:|
| 0 (after one warm-up ask) | **23.3** | 59.3 | 59.3 |
| 1 | **23.7** | 59.5 | 59.5 |
| 100 | **30.0** (+6.7) | 66.7 (+7.4) | 69.4 (+10.1) |
| 1,000 (end of the throughput run, above) | 148 | **91** | 165 |

- Baseline: RustLLM's 23 MiB includes the bundled model registry (`models.json`, 2.6 MB of JSON,
  parsed once). RubyLLM's 59 MiB is the Ruby VM plus the gems.
- **Per chat, the two are similar: ~70 KiB in Rust, ~75 KiB on fibers, ~100 KiB on threads.**
  At 1,000 concurrent chats RubyLLM on fibers ends lower in total than RustLLM. Part of Rust's
  cost is a `reqwest::Client` per `Chat` (19 KiB each, measured with `bench/src/bin/clients.rs`)
  plus its connection pool. The rest is allocator retention after a burst: glibc malloc keeps
  freed memory in per-thread arenas. Sharing one HTTP client across chats that use the same
  provider would cut this. That's a candidate library change, not made here: RubyLLM also creates
  a connection per chat.

## Long conversations

One chat, N sequential asks, with the mock answering at once. Every ask re-sends the whole
history, so some growth is inherent to both.

| Asks in one chat | RustLLM before fix | RustLLM after first fix | **RustLLM now** | RubyLLM + YJIT |
|---:|---:|---:|---:|---:|
| 100 | 0.34 s, 50 MiB | 0.05 s, 37 MiB | **0.03 s, 26 MiB** | 0.16 s, 66 MiB |
| 200 | 2.3 s, 119 MiB | 0.23 s, 73 MiB | **0.12 s, 29 MiB** | 0.27 s, 68 MiB |
| 300 (5-run median) | | | **0.26 s, 35 MiB** | 0.44 s, 72 MiB |
| 400 | 16.4 s, 392 MiB | 0.99 s, 211 MiB | **0.49 s, 40 MiB** | 0.66 s, 76 MiB |
| 800 | | 4.4 s, 755 MiB | **2.1 s, 81 MiB** | 2.1 s, 114 MiB |
| Render the finished 800-ask chat | | 4.2 ms | **1.6 ms** | 2.2 ms |

What the benchmark found, and what changed in the library:

1. **Each response stored a deep copy of the request it answered.** `message.raw.request_body`
   was a `serde_json::Value`, and request k contains the whole history, so a chat held O(n²)
   JSON trees. On top of that, every request deep-cloned every message, including `raw`, to
   render the next payload (O(n³) time). It's now `Arc<str>`: the exact serialized bytes sent,
   like Faraday's `env.request_body`, parsed on demand with `request_body_json()`.
2. **Every request cloned every message in full.** It cloned the model info with its pricing
   maps, the usage entries, and `raw`, just to render. Per-request copies now use
   `Message::for_request()`, which copies only the fields protocols render.

The 100/200/400/800 rows were measured by hand on the same host with the same harness (3 runs
each, medians). The 300 row is from the full 5-run suite. **What's left:** at 800 asks both take
2.1 s, since re-sending an 800-message history dominates in both languages. RustLLM's memory is
now linear. The `rust strip_raw` rows in the raw data clear `message.raw` after every ask, as a
diagnostic. They take the same time as normal (0.26 s at 300 asks) and use 7 MiB less (28 vs 35
MiB). That 7 MiB is the linear cost of keeping one response per message. `raw` no longer drives
the growth.

## Build cost

| | RustLLM | RubyLLM |
|---|---:|---:|
| Cold `cargo build --release -p rust_llm` (~164 crates, 24 threads) | 32 s | `bundle install`: prebuilt gems, seconds |
| Rebuild after editing `chat.rs`, release / `cargo check` | 8.8 s / 0.6 s | 0 (reload) |
| Benchmark client binary, release | 46 MB (with line tables) | n/a |

## What this does not measure

- Real providers: a real model takes 0.5–30 s per answer, which dwarfs every library number
  above. These results matter for many concurrent chats per process, streaming-heavy UIs, tool
  loops with fast tools, CLIs, and short-lived processes.
- TLS: the mock is plain HTTP on loopback. Real traffic adds a TLS handshake per new connection
  on both sides (rustls vs OpenSSL).
- Persistence: `rust_llm_loco` vs `acts_as_chat` (SeaORM vs Active Record writes) is not
  benchmarked here.
- Other providers: Anthropic only. The OpenAI Responses path parses larger bodies (the
  recorded one is 4.6 KB vs 0.5 KB), so JSON parsing weighs more there.
