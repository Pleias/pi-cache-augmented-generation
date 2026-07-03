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


# Context variants. Baguettotron is a verbose reasoning model in a 4k window, so we
# feed a trimmed fiche (not the whole entry) to leave room for the <think> block +
# answer. See docs §5.
#   answer          : lesson content only (~1.9k tok)
#   answer_exercises : lesson + exercises-with-corrigés (~2.3k tok). Lets the model
#                      answer corrigé-lookup questions and propose grounded exercises.
CONTEXT_VARIANTS = ("answer", "answer_exercises")


def build_fiche(entry: dict, variant: str) -> str:
    ans = str(entry.get("answer", "")).strip()
    if variant == "answer":
        return ans
    if variant == "answer_exercises":
        ex = str(entry.get("exercises", "")).strip()
        return ans + ("\n\n" + ex if ex else "")
    raise ValueError(f"unknown context variant: {variant}")


def split_think(raw: str) -> tuple[str, str, bool]:
    """Reasoning model output = <think>…</think> then the answer. The template opens
    the think block, so `raw` starts INSIDE it. Returns (reasoning, answer, truncated).
    truncated=True means </think> never appeared -> ran out of tokens mid-reasoning,
    no answer produced (score as 'no-answer', not wrong)."""
    tag = "</think>"
    if tag in raw:
        think, answer = raw.split(tag, 1)
        return think.strip(), answer.strip(), False
    return raw.strip(), "", True


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
    raw = d["choices"][0]["message"]["content"] or ""
    finish = d["choices"][0].get("finish_reason")
    reasoning, answer, truncated = split_think(raw)  # reasoning model: strip <think>
    t = d.get("timings", {})  # llama.cpp adds this to the OpenAI-compat response
    usage = d.get("usage", {})
    return {
        "answer": answer,            # text AFTER </think> — what we score
        "reasoning": reasoning,      # the <think> block (kept for inspection)
        "truncated": truncated,      # </think> never emitted -> no answer
        "finish_reason": finish,
        "prompt_ms": t.get("prompt_ms"),
        "predicted_ms": t.get("predicted_ms"),
        "prompt_n": t.get("prompt_n", usage.get("prompt_tokens")),
        "predicted_n": t.get("predicted_n", usage.get("completion_tokens")),
        "cache_n": t.get("cache_n"),  # cached-prefix tokens -> verifies CAG reuse
        "wall_ms": round(wall_ms, 1),
    }


def run_mode(client, base, mode, fiche, questions, *, max_tokens, seed):
    """Run all questions in one mode, in order. For cag, order matters: Q1 primes
    the fiche prefix, Q2..N should show prompt_ms collapse (and cache_n jump)."""
    system = SYSTEM_NOCTX if mode == "nocontext" else SYSTEM_PROMPT.format(fiche=fiche)
    cache_prompt = (mode == "cag")
    rows = []
    for q in questions:
        res = ask(client, base, system, q["question"],
                  cache_prompt=cache_prompt, max_tokens=max_tokens, seed=seed)
        res.update({"id": q["id"], "type": q.get("type"),
                    "leak_risk": q.get("leak_risk"),
                    "requires_context": q.get("requires_context"),
                    "expected": q.get("expected_answer"),
                    "expected_elements": q.get("expected_elements")})
        rows.append(res)
        pm = res["prompt_ms"] or 0
        flag = " [TRUNC:no-answer]" if res["truncated"] else ""
        shown = res["answer"] or res["reasoning"]
        print(f"  [{mode:9}] Q{q['id']:<2} prefill={pm:8.1f}ms cache_n={res['cache_n']} "
              f"decode={res['predicted_ms'] or 0:.0f}ms{flag} -> {shown[:60]!r}")
    return rows


def summarize(variant: str, results: dict):
    print(f"\n=== [{variant}] prefill (prompt_ms) per question ===")
    modes = list(results.keys())
    ids = [r["id"] for r in next(iter(results.values()))]
    hdr = "Q   | " + " | ".join(f"{m:>10}" for m in modes)
    print(hdr); print("-" * len(hdr))
    for i, qid in enumerate(ids):
        cells = []
        for m in modes:
            pm = results[m][i]["prompt_ms"]
            cells.append(f"{pm:10.1f}" if pm is not None else f"{'--':>10}")
        print(f"Q{qid:<3} | " + " | ".join(cells))
    print(f"--- [{variant}] TOTAL prefill ms over all questions (the CAG story) ---")
    for m in modes:
        tot = sum((r["prompt_ms"] or 0) for r in results[m])
        trunc = sum(1 for r in results[m] if r["truncated"])
        print(f"  {m:9}: {tot:9.1f} ms   ({trunc}/{len(results[m])} truncated/no-answer)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--entry", type=int, default=31)
    ap.add_argument("--questions", default=str(ROOT / "bench" / "questions_entry31.json"))
    ap.add_argument("--modes", nargs="+", default=["nocontext", "naive", "cag"],
                    choices=["nocontext", "naive", "cag"])
    ap.add_argument("--context", nargs="+", default=list(CONTEXT_VARIANTS),
                    choices=CONTEXT_VARIANTS,
                    help="which context variant(s) to run (default: both)")
    ap.add_argument("--max-tokens", type=int, default=512,
                    help="must fit reasoning (<think>) + answer; see docs §5")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tokens-only", action="store_true",
                    help="just report exact tokenized length of each variant and exit")
    args = ap.parse_args()

    entry = json.loads(DATA.read_text())[args.entry]

    with httpx.Client() as client:
        # sanity: server up?
        try:
            client.get(f"{args.base}/health", timeout=5).raise_for_status()
        except Exception as e:
            sys.exit(f"Server not reachable at {args.base} — start it with "
                     f"./bench/run_server.sh  ({e})")

        budget = 4096
        fiche_tok = {}
        for v in args.context:
            fiche = build_fiche(entry, v)
            n_tok = server_tokenize(client, args.base, fiche)
            fiche_tok[v] = n_tok
            used = n_tok + 50 + args.max_tokens
            print(f"[{v:16}] {len(fiche)} chars = {n_tok} tok  | "
                  f"+50 q +{args.max_tokens} gen = {used}/{budget} "
                  f"{'OK' if used <= budget else '!! OVERFLOW'}")
        if args.tokens_only:
            return

        questions = json.loads(Path(args.questions).read_text())["questions"]
        print(f"\nRunning {len(questions)} questions x {len(args.modes)} modes "
              f"x {len(args.context)} context variants "
              f"(max_tokens={args.max_tokens}, seed={args.seed})")

        all_results = {}  # variant -> mode -> rows
        for v in args.context:
            fiche = build_fiche(entry, v)
            print(f"\n########## CONTEXT VARIANT: {v} ##########")
            per_mode = {}
            for mode in args.modes:
                print(f"--- [{v}] mode: {mode} ---")
                per_mode[mode] = run_mode(client, args.base, mode, fiche, questions,
                                          max_tokens=args.max_tokens, seed=args.seed)
            all_results[v] = per_mode

    for v in args.context:
        summarize(v, all_results[v])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"entry{args.entry}_results.json"
    out.write_text(json.dumps(
        {"entry": args.entry, "fiche_tokens": fiche_tok, "max_tokens": args.max_tokens,
         "seed": args.seed, "context_variants": args.context,
         "results": all_results}, ensure_ascii=False, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
