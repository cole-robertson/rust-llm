#!/usr/bin/env bash
# Starts a generated app (cwd) against the Anthropic replay: fresh DB, migrate, seed, then
# `cargo loco start --server-and-worker` on $PORT. Used for the chat_ui end-to-end check.
set -euo pipefail
CASSETTE="$1"
PORT="${PORT:-5188}"
rm -f ./*.sqlite ./*.sqlite-*
cargo loco db migrate 2>&1 | grep -oE "Migration '[a-z0-9_]+' has been applied"
cargo loco db seed >/dev/null 2>&1
nohup python3 "$(dirname "$0")/anthropic_replay.py" "$CASSETTE" 18765 >/tmp/replay.log 2>&1 &
ANTHROPIC_API_KEY=test ANTHROPIC_API_BASE=http://127.0.0.1:18765 RUST_LLM_DEFAULT_MODEL=claude-haiku-4-5 PORT="$PORT" \
  nohup cargo loco start --server-and-worker >/tmp/genapp.log 2>&1 &
for _ in $(seq 1 60); do
  curl -sf -o /dev/null "http://localhost:$PORT/up" && { echo "up on $PORT"; exit 0; }
  sleep 1
done
echo "app did not start:" >&2
tail -5 /tmp/genapp.log >&2
exit 1
