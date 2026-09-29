#!/usr/bin/env bash
# Drives a generated chat UI with curl, as the kit's seeded user: sign in, list models, open
# /chats/new, create a chat, wait for the worker's answer, post a follow-up, and delete the chat.
# Prints each step; exits non-zero on the first failed check.
set -euo pipefail
BASE="http://localhost:${PORT:-5188}"
JAR="$(mktemp)"
VERSION=""
fail() { echo "FAIL: $*" >&2; exit 1; }

xsrf() { awk '$6 == "XSRF-TOKEN" { print $7 }' "$JAR" | python3 -c 'import sys,urllib.parse;print(urllib.parse.unquote(sys.stdin.read().strip()))'; }

# An Inertia visit: the page object as JSON.
page() {
  curl -s -b "$JAR" -c "$JAR" -H "X-Inertia: true" -H "X-Inertia-Version: $VERSION" "$BASE$1"
}

# A form post the way Inertia's <Form> sends it; prints "status location".
post() {
  curl -s -o /dev/null -w "%{http_code} %{redirect_url}" -b "$JAR" -c "$JAR" -X "${3:-POST}" \
    -H "Content-Type: application/json" -H "X-XSRF-TOKEN: $(xsrf)" -d "$2" "$BASE$1"
}

prop() { python3 -c "import json,sys;p=json.load(sys.stdin);print(eval(sys.argv[1]))" "$1"; }

curl -s -o /dev/null -b "$JAR" -c "$JAR" "$BASE/sign_in"
VERSION="$(curl -s -b "$JAR" -c "$JAR" "$BASE/sign_in" | python3 -c '
import json,re,sys
m=re.search(r"<script data-page=\"app\"[^>]*>(.*?)</script>", sys.stdin.read(), re.S) or sys.exit("no page object on /sign_in")
print(json.loads(m.group(1))["version"] or "")')"

r="$(post /sign_in '{"email":"one@example.com","password":"Secret1*3*5*"}')"
[[ "$r" == 30[23]* ]] || fail "sign in: $r"
echo "signed in: $r"

anon="$(curl -s -o /dev/null -w '%{http_code} %{redirect_url}' "$BASE/chats")"
[[ "$anon" == 30[23]*sign_in ]] || fail "/chats without a session should redirect to sign in: $anon"
echo "GET /chats signed out -> $anon"

models="$(page /models)"
echo "GET /models: component=$(prop 'p["component"]' <<<"$models") models=$(prop 'len(p["props"]["models"])' <<<"$models")"
[[ "$(prop 'p["component"]' <<<"$models")" == models/index ]] || fail "/models"
show="$(page '/models/claude-haiku-4-5?provider=anthropic')"
echo "GET /models/claude-haiku-4-5: $(prop 'p["props"]["model"]["label"]' <<<"$show")"

new="$(page '/chats/new?model=anthropic:claude-haiku-4-5')"
[[ "$(prop 'p["component"]' <<<"$new")" == chats/new ]] || fail "/chats/new"
echo "GET /chats/new: selected=$(prop 'p["props"]["selected_model"]' <<<"$new") default=$(prop 'p["props"]["default_model_label"]' <<<"$new")"

blank="$(post /chats '{"model":"anthropic:claude-haiku-4-5","prompt":""}')"
errors="$(page '/chats/new' | prop 'p["props"].get("errors")')"
echo "POST /chats blank prompt -> $blank, errors=$errors"
[[ "$errors" == *"can't be blank"* ]] || fail "blank prompt should come back with an error"

created="$(post /chats '{"model":"anthropic:claude-haiku-4-5","prompt":"What'"'"'s 2 + 2?"}')"
echo "POST /chats -> $created"
chat_path="/${created#*://*/}"
[[ "$chat_path" =~ ^/chats/[0-9]+$ ]] || fail "create should redirect to the chat: $created"

for _ in $(seq 1 30); do
  shown="$(page "$chat_path")"
  [[ "$(prop 'p["props"]["awaiting_response"]' <<<"$shown")" == False ]] && break
  sleep 1
done
echo "GET $chat_path: model=$(prop 'p["props"]["chat"]["model_label"]' <<<"$shown") awaiting=$(prop 'p["props"]["awaiting_response"]' <<<"$shown")"
echo "  messages: $(prop '[(m["role"], m["content"]) for m in p["props"]["messages"]]' <<<"$shown")"
[[ "$(prop '[m["role"] for m in p["props"]["messages"]]' <<<"$shown")" == "['user', 'assistant']" ]] || fail "expected a user and an assistant message"

partial="$(curl -s -b "$JAR" -H "X-Inertia: true" -H "X-Inertia-Version: $VERSION" \
  -H "X-Inertia-Partial-Component: chats/show" -H "X-Inertia-Partial-Data: messages,awaiting_response" "$BASE$chat_path")"
echo "  poll (partial reload) props: $(prop 'sorted(k for k in p["props"] if k not in ("auth","errors","flash"))' <<<"$partial")"

followup="$(post "$chat_path/messages" '{"content":"And 3 + 3?"}')"
echo "POST $chat_path/messages -> $followup"
for _ in $(seq 1 30); do
  shown="$(page "$chat_path")"
  [[ "$(prop 'p["props"]["awaiting_response"]' <<<"$shown")" == False ]] && break
  sleep 1
done
echo "  roles now: $(prop '[m["role"] for m in p["props"]["messages"]]' <<<"$shown")"
[[ "$(prop 'len(p["props"]["messages"])' <<<"$shown")" == 4 ]] || fail "follow-up should add a user and an assistant message"

index="$(page /chats)"
echo "GET /chats: $(prop '[(c["id"], c["message_count"]) for c in p["props"]["chats"]]' <<<"$index")"

deleted="$(post "$chat_path" '' DELETE)"
echo "DELETE $chat_path -> $deleted"
missing="$(curl -s -o /dev/null -w '%{http_code}' -b "$JAR" "$BASE$chat_path")"
echo "GET $chat_path after delete -> $missing"
[[ "$missing" == 404 ]] || fail "deleted chat should 404"
echo "OK"
