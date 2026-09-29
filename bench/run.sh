#!/usr/bin/env bash
# RustLLM vs RubyLLM benchmark. Run from Archie (or anywhere with ssh to framework):
#
#   bench/run.sh                 # full run, results in bench/results/<date>/
#   QUICK=1 bench/run.sh         # small counts, to check the harness
#
# Everything executes on the framework box, from a private copy of the repo (other agents'
# `bin/fw` syncs use --delete on the shared dir). Both clients talk to the same mock Anthropic
# server (bench/src/bin/mock.rs), pinned to its own cores so it is never the bottleneck.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAMP="$(date +%Y-%m-%d)"
OUT="$ROOT/bench/results/$STAMP${QUICK:+-quick}"
mkdir -p "$OUT"

ssh framework "mkdir -p ~/.cache/rust-llm-bench/src"
rsync -a --delete --exclude target --exclude .git --exclude 'bench/results' \
  --exclude 'upstream/spec' --exclude 'upstream/docs' --exclude 'crates/rust_llm/tests/cassettes' \
  "$ROOT/" "framework:.cache/rust-llm-bench/src/"

ssh framework "QUICK=${QUICK:-} bash -s" > "$OUT/raw.jsonl" 2> "$OUT/stderr.log" <<'EOF'
set -euo pipefail
cd "$HOME/.cache/rust-llm-bench/src/bench"
export PATH="$HOME/.local/share/gem/ruby/3.4.0/bin:$PATH"
export BUNDLE_PATH="$HOME/.cache/rust-llm-bench/bundle" BUNDLE_GEMFILE="$PWD/ruby/Gemfile"
export CARGO_TARGET_DIR="$HOME/.cache/rust-llm-bench/target"
cargo build --release --quiet >&2
bundle install --quiet >&2
BIN="$CARGO_TARGET_DIR/release"

# Cores 0-3 are the fast Zen 5 cores (5.2 GHz), 4-11 the Zen 5c cores (3.3 GHz); 12-23 are
# their SMT siblings. Clients get cores 0-3, the mock gets 4-7, so they never share a core.
CLIENT_CPUS=0-3
taskset -c 4-7 "$BIN/mock" 8765 >&2 &
MOCK=$!
trap 'kill $MOCK' EXIT
sleep 0.5

q() { if [ -n "${QUICK:-}" ]; then echo "$2"; else echo "$1"; fi; }
rust() { taskset -c $CLIENT_CPUS "$BIN/client" "$@"; }
ruby_yjit() { taskset -c $CLIENT_CPUS bundle exec ruby --yjit ruby/client.rb "$@"; }
ruby_interp() { taskset -c $CLIENT_CPUS bundle exec ruby --disable-yjit ruby/client.rb "$@"; }

python3 - <<PY
import json, platform, subprocess as sp
sh = lambda c: sp.run(c, shell=True, capture_output=True, text=True).stdout.strip()
print(json.dumps({"impl": "meta", "case": "env", "result": {
  "host": platform.node(), "cpu": sh("lscpu | sed -n 's/^Model name: *//p'"), "kernel": platform.release(),
  "mem": sh("free -g | awk '/Mem:/{print \$2\" GiB\"}'"), "rustc": sh("rustc -V"), "ruby": sh("ruby -v"),
  "gems": sh("cd ruby && bundle list 2>/dev/null | grep -E 'ruby_llm|faraday |async |json '"),
  "loadavg": open("/proc/loadavg").read().split()[:3]}}))
PY

REPEATS=$(q 5 1)
for rep in $(seq 1 "$REPEATS"); do
  # Cold start: a fresh process per sample (the OS page cache is warm for both).
  for i in $(seq 1 "$(q 10 2)"); do rust first; ruby_yjit first; ruby_interp first; done
  for impl in rust ruby_yjit ruby_interp; do
    $impl overhead n="$(q 2000 100)" delay=0
    $impl stream chunks=1000 n="$(q 50 5)"
    $impl tools n="$(q 300 20)"
    $impl render messages=200 n="$(q 2000 100)"
  done
  for chats in 1 10 100 "$(q 1000 200)"; do
    rust concurrent chats="$chats" rounds=5 delay=50
    ruby_yjit concurrent chats="$chats" rounds=5 delay=50
    ruby_yjit concurrent chats="$chats" rounds=5 delay=50 mode=threads
  done
  # One long conversation: 300 asks in one chat, 0 ms delay (history re-sent every time).
  rust concurrent chats=1 rounds="$(q 300 100)" delay=0
  rust concurrent chats=1 rounds="$(q 300 100)" delay=0 strip_raw=1
  ruby_yjit concurrent chats=1 rounds="$(q 300 100)" delay=0
  for chats in 1 100; do
    rust memory chats="$chats" delay=2000
    ruby_yjit memory chats="$chats" delay=2000
    ruby_yjit memory chats="$chats" delay=2000 mode=threads
  done
done
EOF

echo "raw results: $OUT/raw.jsonl ($(wc -l < "$OUT/raw.jsonl") lines)"
python3 "$ROOT/bench/summarize.py" "$OUT/raw.jsonl" | tee "$OUT/summary.md"
