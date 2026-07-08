#!/usr/bin/env bash
# Live smoke test for a RUNNING CAG container. YOU run this (the project rule keeps
# Claude out of docker); it drives the published HTTP API with curl.
#
#   BASE=http://localhost:8000 ./smoke.sh                 # API-only run
#   CONTAINER=cag ./smoke.sh                              # also tests restart persistence
#
# Sequence: status -> ingest sample -> ask (EN, grounded) -> ask (multi-turn follow-up) ->
#   ask (FR, grounded) -> [restart container] -> ASSERT status.restored==true ->
#   ask again (served from the restored cache). Grounding + persistence are ASSERTED
#   (grep, non-zero exit on miss), not just printed, so regressions fail loudly.
set -euo pipefail

BASE="${BASE:-http://localhost:8000}"
CONTAINER="${CONTAINER:-}"

say() { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }
req() { curl -sS -m 120 "$@"; }

# A short, fictional document so a correct answer proves grounding (not model memory).
read -r -d '' DOC <<'EOF' || true
Acme Cloud Storage — Service Facts (internal).
The Standard plan includes 500 GB of storage and costs 9 euros per month.
The Pro plan includes 2 TB of storage and costs 19 euros per month.
Support is available Monday to Friday, 9:00 to 18:00 CET.
Data is stored in data centers located in Frankfurt and Paris.
The free trial lasts 14 days and requires no credit card.
EOF

# Reconstruct the plain-text answer from an SSE token stream (joins {"token": …} frames)
# so a fact split across several tokens still matches a substring assert.
extract_answer() {
  python3 -c '
import sys, json
out = []
for line in sys.stdin:
    line = line.strip()
    if not line.startswith("data:"):
        continue
    data = line[5:].strip()
    if not data:
        continue
    try:
        j = json.loads(data)
    except ValueError:
        continue
    if j.get("token"):
        out.append(j["token"])
sys.stdout.write("".join(out))
'
}

# Fail the whole smoke loudly if a needle is absent (a grounding / persistence regression
# must not pass silently).
assert_contains() {  # assert_contains <haystack> <needle> <label>
  if printf '%s' "$1" | grep -qF -- "$2"; then
    printf '   \033[1;32mPASS\033[0m %s — found "%s"\n' "$3" "$2"
  else
    printf '   \033[1;31mFAIL\033[0m %s — "%s" not found\n' "$3" "$2" >&2
    exit 1
  fi
}

# Ask a question, echo the streamed answer, and expose it via the ANSWER global.
ask() {  # ask <question-text>
  local raw payload
  payload=$(Q="$1" python3 -c 'import json,os;print(json.dumps({"question":os.environ["Q"]}))')
  raw=$(req -N -X POST "$BASE/api/ask" -H 'Content-Type: application/json' --data "$payload")
  ANSWER=$(printf '%s' "$raw" | extract_answer)
  printf '   A: %s\n' "$ANSWER"
}

say "1. status (before ingest)"
req "$BASE/api/status"; echo

say "2. ingest sample document"
# jq-free JSON: build the payload with python so quotes/UTF-8 are escaped correctly.
INGEST_PAYLOAD=$(DOC="$DOC" python3 -c 'import json,os;print(json.dumps({"text":os.environ["DOC"]}))')
req -X POST "$BASE/api/ingest" -H 'Content-Type: application/json' \
    --data "$INGEST_PAYLOAD"; echo

say "3. ask (English) — streamed, must be grounded in the document"
echo '  Q: How much does the Pro plan cost and how much storage does it include?'
ask "How much does the Pro plan cost and how much storage does it include?"
assert_contains "$ANSWER" "19" "Pro plan price grounded"

say "4. ask (multi-turn follow-up) — relies on chat history + the ChatML prefix-cache seam"
echo '  Q: And the Standard plan?'
# Elliptical follow-up: only answerable if the PREVIOUS turn is still in context. It also
# live-exercises the ChatML-seam additivity the cached per-pair token counts assume — a
# real-tokenizer miscount would drift history_tokens and eventually overflow the budget.
ask "And the Standard plan?"
assert_contains "$ANSWER" "500" "Standard plan grounded via chat history"

say "5. ask (French) — grounded answer in the question's language"
echo '  Q: Combien de temps dure l’essai gratuit ?'
ask "Combien de temps dure l’essai gratuit ?"
assert_contains "$ANSWER" "14" "French trial length grounded"

if [ -n "$CONTAINER" ]; then
  say "6. docker restart $CONTAINER (tests KV-cache slot restore)"
  docker restart "$CONTAINER"
  echo -n "   waiting for app to come back"
  for _ in $(seq 1 120); do
    if curl -sf -m 5 "$BASE/api/status" >/dev/null 2>&1; then echo " -> up"; break; fi
    sleep 1; echo -n "."
  done

  say "7. status after restart — ASSERT \"restored\": true (the persistence guarantee)"
  STATUS=$(req "$BASE/api/status"); printf '%s\n' "$STATUS"
  # Regex tolerates compact ("restored":true) or pretty ("restored": true) JSON.
  if printf '%s' "$STATUS" | grep -Eq '"restored" *: *true'; then
    printf '   \033[1;32mPASS\033[0m KV slot restored from disk (no ~5-min re-prefill)\n'
  else
    printf '   \033[1;31mFAIL\033[0m restored:true missing — restore regressed to a re-prefill\n' >&2
    exit 1
  fi

  say "8. ask again after restart — must stream a grounded answer from the restored cache"
  echo '  Q: Where are the data centers located?'
  ask "Where are the data centers located?"
  assert_contains "$ANSWER" "Frankfurt" "grounded answer served from restored cache"

  say "smoke PASSED (with restart persistence)"
else
  say "6. restart persistence — set CONTAINER=<name> to automate"
  cat <<EOF
   Manual check:
     docker restart <your-container>
     # wait for it to come back, then:
     curl -sS $BASE/api/status        # expect "restored": true, no ~5-min re-prefill
     curl -N -X POST $BASE/api/ask -H 'Content-Type: application/json' \\
          --data '{"question":"Where are the data centers located?"}'
EOF
  say "smoke PASSED (API-only; restart persistence not exercised)"
fi
