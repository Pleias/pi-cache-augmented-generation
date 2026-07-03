# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx>=0.27"]
# ///
"""
CAG benchmark runner for Baguettotron on teacher fiches.

Talks to an already-running llama-server (see run_server.sh). For one fiche it runs
each question in three modes and logs the server's own prefill/decode timings:

  cag        : fiche kept as stable prefix, KV cache REUSED across questions
               (cache_prompt=true). Q1 prefills the fiche; Q2..N skip it.
  naive      : full fiche re-sent and RE-PREFILLED every question (cache_prompt=false).
  nocontext  : question only, no fiche. Validity gate + quality/leak baseline.

Primary metric = prompt_ms (prefill time). CAG's whole value is killing repeated prefill.
Answer quality across cag vs naive should match (correctness check); nocontext that
answers correctly = a leaky question.

Usage:
  uv run bench/cag_bench.py                      # entry 32, all modes
  uv run bench/cag_bench.py --tokens-only        # just report exact fiche token count
  uv run bench/cag_bench.py --modes cag naive    # subset
  uv run bench/cag_bench.py --entry 32 --max-tokens 150
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import httpx

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "extensions_geographie_deuxieme_etape_ce1.json"
RESULTS_DIR = ROOT / "bench" / "results"

SYSTEM_PROMPT = (
    "Tu es un assistant pédagogique. Réponds à la question en te basant "
    "UNIQUEMENT sur la fiche pédagogique fournie ci-dessous. Sois bref et précis.\n\n"
    "=== FICHE PÉDAGOGIQUE ===\n{fiche}\n=== FIN DE LA FICHE ==="
)
SYSTEM_NOCTX = (
    "Tu es un assistant pédagogique. Réponds à la question de manière brève et précise."
)


def build_fiche(entry: dict) -> str:
    """The context we feed the model — the answer-bearing fields, no JSON wrapper."""
    parts = []
    for k in ("introductory_text", "answer", "situations", "exercises"):
        v = entry.get(k)
        if v:
            parts.append(str(v).strip())
    return "\n\n".join(parts)


def server_tokenize(client: httpx.Client, base: str, text: str) -> int:
    r = client.post(f"{base}/tokenize", json={"content": text})
    r.raise_for_status()
    return len(r.json().get("tokens", []))


def ask(client: httpx.Client, base: str, system: str, question: str,
        *, cache_prompt: bool, max_tokens: int, seed: int) -> dict:
    """One chat completion. Returns {answer, prompt_ms, predicted_ms, prompt_n,
    predicted_n, wall_ms}. prompt_ms is the server's prefill time (the CAG metric)."""
    body = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
        "temperature": 0.0,
        "seed": seed,
        "max_tokens": max_tokens,
        "cache_prompt": cache_prompt,   # llama.cpp extension: reuse matching KV prefix
        "stream": False,
    }
    t0 = time.perf_counter()
    r = client.post(f"{base}/v1/chat/completions", json=body, timeout=600)
    wall_ms = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    d = r.json()
    answer = d["choices"][0]["message"]["content"].strip()
    t = d.get("timings", {})  # llama.cpp adds this to the OpenAI-compat response
    usage = d.get("usage", {})
    return {
        "answer": answer,
        "prompt_ms": t.get("prompt_ms"),
        "predicted_ms": t.get("predicted_ms"),
        "prompt_n": t.get("prompt_n", usage.get("prompt_tokens")),
        "predicted_n": t.get("predicted_n", usage.get("completion_tokens")),
        "wall_ms": round(wall_ms, 1),
    }


def run_mode(client, base, mode, fiche, questions, *, max_tokens, seed):
    """Run all questions in one mode, in order. For cag, order matters: Q1 primes
    the fiche prefix, Q2..N should show prompt_ms collapse."""
    system = SYSTEM_NOCTX if mode == "nocontext" else SYSTEM_PROMPT.format(fiche=fiche)
    cache_prompt = (mode == "cag")
    rows = []
    for q in questions:
        res = ask(client, base, system, q["question"],
                  cache_prompt=cache_prompt, max_tokens=max_tokens, seed=seed)
        res.update({"id": q["id"], "type": q.get("type"),
                    "leak_risk": q.get("leak_risk"),
                    "expected": q.get("expected_answer")})
        rows.append(res)
        pm = res["prompt_ms"]
        print(f"  [{mode:9}] Q{q['id']} prefill={pm:8.1f}ms "
              f"({res['prompt_n']} tok)  decode={res['predicted_ms']:.0f}ms "
              f"-> {res['answer'][:60]!r}")
    return rows


def summarize(results: dict):
    print("\n=== SUMMARY: prefill (prompt_ms) per question ===")
    modes = list(results.keys())
    ids = [r["id"] for r in next(iter(results.values()))]
    hdr = "Q  | " + " | ".join(f"{m:>10}" for m in modes)
    print(hdr); print("-" * len(hdr))
    for i, qid in enumerate(ids):
        cells = []
        for m in modes:
            pm = results[m][i]["prompt_ms"]
            cells.append(f"{pm:10.1f}" if pm is not None else f"{'--':>10}")
        print(f"Q{qid} | " + " | ".join(cells))
    # amortized totals: what you actually pay for N questions
    print("\n=== TOTAL prefill ms over all questions (the CAG story) ===")
    for m in modes:
        tot = sum((r["prompt_ms"] or 0) for r in results[m])
        print(f"  {m:9}: {tot:9.1f} ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--entry", type=int, default=32)
    ap.add_argument("--questions", default=str(ROOT / "bench" / "questions_entry32.json"))
    ap.add_argument("--modes", nargs="+", default=["nocontext", "naive", "cag"],
                    choices=["nocontext", "naive", "cag"])
    ap.add_argument("--max-tokens", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tokens-only", action="store_true",
                    help="just report exact tokenized length of the fiche and exit")
    args = ap.parse_args()

    entry = json.loads(DATA.read_text())[args.entry]
    fiche = build_fiche(entry)

    with httpx.Client() as client:
        # sanity: server up?
        try:
            client.get(f"{args.base}/health", timeout=5).raise_for_status()
        except Exception as e:
            sys.exit(f"Server not reachable at {args.base} — start it with "
                     f"./bench/run_server.sh  ({e})")

        n_tok = server_tokenize(client, args.base, fiche)
        print(f"Fiche entry {args.entry}: {len(fiche)} chars = {n_tok} tokens "
              f"(exact, model tokenizer)")
        budget = 4000
        print(f"Context budget check: {n_tok} fiche + ~50 question + {args.max_tokens} "
              f"answer = {n_tok + 50 + args.max_tokens} / {budget} ctx "
              f"{'OK' if n_tok + 50 + args.max_tokens <= budget else '!! OVERFLOW — reduce max-tokens or trim fiche'}")
        if args.tokens_only:
            return

        questions = json.loads(Path(args.questions).read_text())["questions"]
        print(f"\nRunning {len(questions)} questions x {len(args.modes)} modes "
              f"(max_tokens={args.max_tokens}, seed={args.seed})\n")

        results = {}
        for mode in args.modes:
            print(f"--- mode: {mode} ---")
            results[mode] = run_mode(client, args.base, mode, fiche, questions,
                                     max_tokens=args.max_tokens, seed=args.seed)

    summarize(results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"entry{args.entry}_results.json"
    out.write_text(json.dumps(
        {"entry": args.entry, "fiche_tokens": n_tok, "max_tokens": args.max_tokens,
         "seed": args.seed, "results": results}, ensure_ascii=False, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
