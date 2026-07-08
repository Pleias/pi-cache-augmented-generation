#!/usr/bin/env bash
# Q4 throughput bench against the RUNNING cag-app container on the Pi.
#
# Measures the baked Qwen2.5-1.5B Q4_K_M at the SAME context sizes as the Q8_0
# benchmark (docs/throughput.md: answer ~1956 tok / answer_exercises ~2455 / full
# ~3180), in the same three modes (nocontext / naive / cag), talking to the
# INTERNAL llama-server (127.0.0.1:8080) from inside the container — same
# endpoints, same manual ChatML, same greedy sampling as bench/cag_bench.py.
#
# Usage:
#   CONTAINER=cag ./bench/q4_pi_bench.sh            # full run (three context sizes)
#   CONTAINER=cag TARGETS="1956" ./bench/q4_pi_bench.sh   # quick single-size run
#   CONTAINER=cag N_PREDICT=200 QUESTIONS=3 ./bench/q4_pi_bench.sh
#
# NOTES
# - Don't chat in the app while this runs (single slot; the lock is app-side only).
# - The run overwrites slot 0's LIVE KV with bench prompts. Saved slot FILES are
#   untouched; `docker restart $CONTAINER` afterwards restores the app's document
#   cache from disk in seconds.
# - Expect ~10-20 min for the full three-size run on a Pi 4 (naive prefills dominate).
set -euo pipefail

CONTAINER="${CONTAINER:-cag}"
TARGETS="${TARGETS:-1956 2455 3180}"
N_PREDICT="${N_PREDICT:-200}"
QUESTIONS="${QUESTIONS:-3}"

docker exec -i \
  -e TARGETS="$TARGETS" -e N_PREDICT="$N_PREDICT" -e QUESTIONS="$QUESTIONS" \
  "$CONTAINER" /opt/venv/bin/python - <<'PY'
import json, os, sys, time

sys.path.insert(0, "/app")           # reuse the app's exact prompt construction
from server import cag                # build_system_prefix / build_prompt / STOP

import httpx

BASE = "http://127.0.0.1:8080"
TARGETS = [int(t) for t in os.environ["TARGETS"].split()]
N_PREDICT = int(os.environ["N_PREDICT"])
QUESTIONS = int(os.environ["QUESTIONS"])

# Generic French filler in the domain of the original fiches; repeated + trimmed
# to hit each target token count for the SYSTEM PREFIX (same anchor the app and
# the Q8 bench used).
PARA = (
    "L'eau est indispensable à la vie du village. La mare se remplit pendant la "
    "saison des pluies et sert à abreuver les animaux. Le puits, creusé par les "
    "habitants, donne une eau plus propre pour la cuisine. Le forage, équipé d'une "
    "pompe, fournit l'eau potable de l'école et du dispensaire. Les enfants "
    "apprennent à protéger ces points d'eau en les gardant propres. "
)
QS = [
    "Résume le document en deux phrases.",
    "Quels sont les points d'eau mentionnés et à quoi servent-ils ?",
    "Propose un exercice simple pour des élèves de CE1 à partir de ce document.",
    "Pourquoi faut-il protéger les points d'eau ?",
][:QUESTIONS]

http = httpx.Client(timeout=None)

def tokens(text: str) -> int:
    r = http.post(f"{BASE}/tokenize", json={"content": text})
    r.raise_for_status()
    return len(r.json()["tokens"])

def completion(prompt: str, cache: bool, n_predict: int):
    r = http.post(f"{BASE}/completion", json={
        "prompt": prompt, "cache_prompt": cache, "id_slot": 0,
        "n_predict": n_predict, "stream": False,
        "temperature": 0.0, "stop": cag.STOP,   # greedy — matches the Q8 bench
    })
    r.raise_for_status()
    return r.json()

def build_fiche(target: int) -> str:
    """Filler text whose SYSTEM PREFIX tokenizes to ~target (±10)."""
    lo, hi = 1, 200
    while tokens(cag.build_system_prefix(PARA * hi)) < target:
        hi *= 2
    while lo < hi:
        mid = (lo + hi) // 2
        if tokens(cag.build_system_prefix(PARA * mid)) < target:
            lo = mid + 1
        else:
            hi = mid
    text = PARA * lo
    # fine trim by sentence to land within ±10 tokens
    while tokens(cag.build_system_prefix(text)) > target + 10 and ". " in text:
        text = text.rsplit(". ", 2)[0] + ". "
    return text

def row(label, mode, t):
    pn, pms = t["prompt_n"], t["prompt_ms"]
    dn, dms = t["predicted_n"], t["predicted_ms"]
    print(f"| {label} | {mode} | {pn} | {pms/1000:8.1f}s | "
          f"{(pn/pms*1000) if pms else 0:6.1f} | {dn} | "
          f"{(dn/dms*1000) if dms else 0:5.2f} | {(pms+dms)/1000:7.1f}s |")

props = http.get(f"{BASE}/props").json()
print(f"model: {props.get('model_path', '?')}")
print(f"targets={TARGETS} n_predict={N_PREDICT} questions={QUESTIONS} temp=0 (greedy)")
print()
print("| ctx target | mode | prompt_n | prefill | pf tok/s | out | dec tok/s | total |")
print("|---|---|---:|---:|---:|---:|---:|---:|")

for target in TARGETS:
    fiche = build_fiche(target)

    # --- nocontext: question only, no document (floor) --------------------------
    for q in QS[:1]:
        p = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
        row(target, "nocontext", completion(p, False, N_PREDICT)["timings"])

    # --- naive: full context re-prefilled EVERY question (the RAG baseline) -----
    for q in QS:
        p = cag.build_prompt(fiche, [], q)
        row(target, "naive", completion(p, False, N_PREDICT)["timings"])

    # --- cag: prefill once (untimed row printed for reference), then reuse ------
    warm = completion(cag.build_system_prefix(fiche), True, 1)
    row(target, "cag-warmup", warm["timings"])
    for q in QS:
        p = cag.build_prompt(fiche, [], q)
        row(target, "cag", completion(p, True, N_PREDICT)["timings"])

print()
print("read: 'naive' prefill tok/s = real prefill speed; 'cag' prompt_n is tiny")
print("(question only, prefix reused) so its prefill col is overhead, not speed;")
print("decode tok/s is comparable across all modes. Q8 reference: docs/throughput.md.")
print("Restore the app's document cache afterwards with: docker restart <container>")
PY
