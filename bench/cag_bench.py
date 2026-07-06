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

Two context variants (answer, answer_exercises) x three modes x N questions.
Results are run-tagged (never overwrite) and saved incrementally to a JSONL as each
call completes, so a killed run loses at most the in-flight call.

Usage:
  uv run bench/cag_bench.py                       # entry 31, all modes+variants
  uv run bench/cag_bench.py --tokens-only         # budget check only, no generation
  uv run bench/cag_bench.py --limit 2 --tag pi    # quick smoke, tagged output
  uv run bench/cag_bench.py --modes cag naive --context answer
"""
from __future__ import annotations
import argparse, json, re, sys, threading, time
from pathlib import Path
import httpx

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "extensions_geographie_deuxieme_etape_ce1.json"
RESULTS_DIR = ROOT / "bench" / "results"

# Fiche goes in the USER turn, not system: small models (Baguettotron ~320M) weight the
# system role weakly and ignored the fiche when it lived there (hallucinated Warang ->
# India). Putting it right before the question in the user message fixes grounding.
# CAG note: "FICHE…QUESTION: " is a stable prefix across questions -> cache_prompt still
# reuses the fiche KV; only the question tail differs.
# Baguettotron's native RAG format (per model card + HF discussions):
#   - context wrapped in <source_1>…</source_1> (real special tokens); the model grounds
#     on it and cites with <ref>[quote]</ref>. Any other framing -> it ignores the context
#     and free-associates (hallucination / lesson-plan mimicry / repetition).
#   - NO system message (the card's RAG example uses none).
#   - <|im_end|> is NOT the model's EOS, so generation runs on into an invented next turn
#     (the "fake conversation" confabulation) unless we pass explicit STOP strings.
STOP = ["<|im_end|>", "<|im_start|>"]

# Card format is question-FIRST, source-after. But that breaks CAG prefix caching (fiche
# no longer a stable prefix). So --source-pos lets us test both:
#   after  = card-faithful (best grounding odds); CAG prefix reuse does NOT apply
#   before = source-first (fiche is a stable prefix -> cache_prompt reuses it); needs
#            verifying the model still grounds when the source precedes the question
USER_CTX_AFTER = "{question}\n\n<source_1>{fiche}</source_1>"
USER_CTX_BEFORE = "<source_1>{fiche}</source_1>\n\n{question}"
SYSTEM_NOCTX = (
    "Tu es un assistant pédagogique. Réponds à la question de manière brève et précise."
)

# --- ChatML / instruct-model format (Qwen2.5, LFM2, SmolLM2, …) -----------------------
# Standard instruct models have a real `system` role rendered BEFORE the user turn, so the
# fiche goes in the system message -> it's a stable, cacheable prefix (context-first) and
# the question varies in the user turn. This is the CAG-friendly layout PleIAs lacks.
# These models answer DIRECTLY (no <think> scaffold), so we take the content verbatim.
# ⚠️ LFM2 caveat: its hybrid conv+attention layers break llama.cpp automatic prefix reuse
#    (cache_prompt/--prompt-cache can crash: ggml-org/llama.cpp#19041). Grounding (nocontext
#    /naive) is fine; the `cag` mode's prefix reuse is unreliable on LFM2 — use a pure-
#    attention model (Qwen2.5-0.5B) for a clean CAG-via-prefix-cache measurement.
CHATML_SYS_CTX = (
    "Tu es un assistant pédagogique. Réponds à la question en français, de manière brève "
    "et précise, en te basant UNIQUEMENT sur le document ci-dessous. Si le document ne "
    "contient pas la réponse, dis-le.\n\n=== DOCUMENT ===\n{fiche}\n=== FIN DU DOCUMENT ==="
)
MODEL_FORMATS = ("baguettotron", "chatml")


# Context variants. Baguettotron is a verbose reasoning model in a 4k window, so we
# feed a trimmed fiche (not the whole entry) to leave room for the <think> block +
# answer. See docs §5.
#   answer          : lesson content only (~1.9k tok)
#   answer_exercises : lesson + exercises-with-corrigés (~2.3k tok). Lets the model
#                      answer corrigé-lookup questions and propose grounded exercises.
#   full             : whole fiche (intro + answer + situations + exercises ≈ 3.5k tok).
#                      Needs a context-first model with a big window (Qwen 32k); overflows
#                      Baguettotron's 4k. Bigger cache -> bigger CAG prefill saving.
CONTEXT_VARIANTS = ("answer", "answer_exercises", "full")


def build_fiche(entry: dict, variant: str) -> str:
    ans = str(entry.get("answer", "")).strip()
    if variant == "answer":
        return ans
    if variant == "answer_exercises":
        ex = str(entry.get("exercises", "")).strip()
        return ans + ("\n\n" + ex if ex else "")
    if variant == "full":
        parts = [str(entry.get(k, "")).strip()
                 for k in ("introductory_text", "answer", "situations", "exercises")]
        return "\n\n".join(p for p in parts if p)
    raise ValueError(f"unknown context variant: {variant}")


def parse_indices(spec: str) -> list[int]:
    """Parse a --corpus spec into a sorted list of unique entry indices. Accepts an
    inclusive range 'a-b' (e.g. '29-38') OR a comma list 'a,b,c' (e.g. '29,31,33')."""
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        a, b = spec.split("-", 1)
        lo, hi = int(a), int(b)
        if lo > hi:
            lo, hi = hi, lo
        return list(range(lo, hi + 1))
    return sorted({int(x) for x in spec.split(",") if x.strip() != ""})


def build_corpus(entries: list, indices: list[int], field: str = "answer") -> str:
    """Concatenate several fiches into ONE labeled context. Each fiche is reused via
    build_fiche(entry, field) so its formatting matches the single-entry path, and gets a
    header:  === FICHE {n}: {titre} (idx {i}) ===. Fiches are joined by a blank line."""
    blocks = []
    for n, i in enumerate(indices, start=1):
        entry = entries[i]
        titre = str((entry.get("lesson_meta") or {}).get("titre", "")).strip()
        body = build_fiche(entry, field)
        blocks.append(f"=== FICHE {n}: {titre} (idx {i}) ===\n{body}\n")
    return "\n".join(blocks)


_SPECIAL = ("<|im_end|>", "<|im_start|>", "<think>", "</think>", "<|endoftext|>")
# Baguettotron's CoT uses a fixed markdown scaffold and reasoning glyphs. Trailing prose
# blocks that contain NONE of these are treated as the final answer.
_SCAFFOLD = ("###", "∴", "Query", "Decomposition", "Retrieval", "Synthesis",
             "Response should", "Response Construction", "Response Structure",
             "Context Analysis", "Knowledge Synthesis", "Requirements", "Parsing",
             "addresses", "both parts", "conversational", "maintaining")


# Baguettotron cites grounded claims with <ref name="…">quote</ref> and wraps context in
# <source_N> tags — strip these tag markers from the displayed answer (keep the quote text).
_TAG_RE = re.compile(r"</?ref[^>]*>|</?source_\d+>")


def _clean(s: str) -> str:
    for tok in _SPECIAL:
        s = s.replace(tok, "")
    s = _TAG_RE.sub("", s)
    return s.strip()


def _tail_answer(text: str) -> str:
    """No </think> delimiter: the model writes a '### 1..5' scaffold then the final
    answer as plain prose at the end. Collect trailing blank-line-separated blocks that
    are NOT scaffold until we hit a scaffold/reasoning block."""
    blocks = re.split(r"\n\s*\n", text)
    picked = []
    for b in reversed(blocks):
        bs = b.strip()
        if not bs:
            continue
        if any(m in bs for m in _SCAFFOLD):
            break
        picked.append(bs)
    return "\n\n".join(reversed(picked)).strip()


def parse_output(content: str, reasoning_content: str, finish: str | None,
                 strip_scaffold: bool = True) -> dict:
    """Split model output into reasoning vs answer, robust to the model's format.
    Truncation is judged by finish_reason == 'length'. `strip_scaffold=False` for direct-
    answer instruct models (Qwen/LFM2/ChatML): the content IS the answer, no <think>."""
    truncated = (finish == "length")
    # direct-answer instruct model: whole content is the answer, no reasoning scaffold
    if not strip_scaffold:
        return {"answer": _clean(content or ""), "reasoning": "", "truncated": truncated}
    # A) server already separated reasoning (some llama.cpp builds) -> content is answer
    if reasoning_content and reasoning_content.strip():
        return {"answer": _clean(content), "reasoning": _clean(reasoning_content),
                "truncated": truncated}
    raw = content or ""
    # B) explicit </think> present (other builds/models)
    if "</think>" in raw:
        think, ans = raw.split("</think>", 1)
        return {"answer": _clean(ans), "reasoning": _clean(think), "truncated": truncated}
    # C) no delimiter -> heuristic: trailing prose = answer, scaffold = reasoning
    cleaned = _clean(raw)
    ans = _tail_answer(cleaned)
    reasoning = cleaned[: len(cleaned) - len(ans)].strip() if ans else cleaned
    return {"answer": ans, "reasoning": reasoning, "truncated": truncated}


def server_tokenize(client: httpx.Client, base: str, text: str) -> int:
    r = client.post(f"{base}/tokenize", json={"content": text})
    r.raise_for_status()
    return len(r.json().get("tokens", []))


def _gguf_kv_dims(path: str) -> dict | None:
    """Read the GGUF metadata needed to size the KV cache: block_count (layers),
    attention.head_count_kv, attention.key_length, attention.value_length. Returns None
    on any failure (then cache size is reported as unknown)."""
    import struct
    try:
        f = open(path, "rb")
    except OSError:
        return None
    with f:
        if f.read(4) != b"GGUF":
            return None
        struct.unpack("<I", f.read(4))  # version
        struct.unpack("<Q", f.read(8))  # n_tensors
        n_kv = struct.unpack("<Q", f.read(8))[0]

        def rstr():
            n = struct.unpack("<Q", f.read(8))[0]
            return f.read(n).decode("utf-8", "replace")

        def rval(t):
            if t == 0: return struct.unpack("<b", f.read(1))[0]
            if t == 1: return struct.unpack("<B", f.read(1))[0]
            if t == 2: return struct.unpack("<h", f.read(2))[0]
            if t == 3: return struct.unpack("<H", f.read(2))[0]
            if t == 4: return struct.unpack("<i", f.read(4))[0]
            if t == 5: return struct.unpack("<I", f.read(4))[0]
            if t == 6: return struct.unpack("<f", f.read(4))[0]
            if t == 7: return struct.unpack("<?", f.read(1))[0]
            if t == 8: return rstr()
            if t == 10: return struct.unpack("<q", f.read(8))[0]
            if t == 11: return struct.unpack("<Q", f.read(8))[0]
            if t == 12: return struct.unpack("<d", f.read(8))[0]
            if t == 9:  # array — read & discard, we don't need any array keys
                et = struct.unpack("<I", f.read(4))[0]
                ln = struct.unpack("<Q", f.read(8))[0]
                for _ in range(ln):
                    rval(et)
                return None
            raise ValueError(f"gguf type {t}")

        want = ("block_count", "attention.head_count_kv",
                "attention.key_length", "attention.value_length",
                "embedding_length", "attention.head_count")
        out = {}
        try:
            for _ in range(n_kv):
                k = rstr()
                t = struct.unpack("<I", f.read(4))[0]
                v = rval(t)
                for w in want:
                    if k.endswith(w):
                        out[w] = v
        except Exception:
            return out or None
        return out


def kv_bytes_per_token(dims: dict, kv_byte: int = 2) -> int | None:
    """KV cache bytes per token = n_kv_heads * (key_len + val_len) * n_layers * kv_byte.
    kv_byte defaults to 2 (f16, llama.cpp default unless -ctk/-ctv quantized)."""
    if not dims:
        return None
    layers = dims.get("block_count")
    kvh = dims.get("attention.head_count_kv")
    kl = dims.get("attention.key_length")
    vl = dims.get("attention.value_length")
    if kl is None or vl is None:  # derive head_dim = embedding_length / head_count
        emb = dims.get("embedding_length"); heads = dims.get("attention.head_count")
        if emb and heads:
            kl = vl = emb // heads
    if not (layers and kvh and kl and vl):
        return None
    return kvh * (kl + vl) * layers * kv_byte


def _status(msg: str):
    """Rewrite a single live status line in place (no newline)."""
    sys.stdout.write("\r\033[K" + msg)
    sys.stdout.flush()


def ask(client: httpx.Client, base: str, system: str | None, user: str,
        *, cache_prompt: bool, max_tokens: int, seed: int,
        temperature: float = 0.0, repeat_penalty: float = 1.0,
        top_p: float | None = None, top_k: int | None = None,
        chat_template_kwargs: dict | None = None,
        strip_scaffold: bool = True, label: str = "") -> dict:
    """One STREAMING chat completion with a live progress line, so a slow box never
    looks hung. Shows a prefill heartbeat (elapsed seconds until the first token), then
    a decode counter (tokens + elapsed). Returns answer/reasoning/timings.
    prompt_ms is the server's prefill time (the CAG metric).

    repeat_penalty > 1.0 breaks greedy repetition loops. chat_template_kwargs forwards
    e.g. {"enable_thinking": false} to disable Qwen3 reasoning (direct answer)."""
    messages = ([{"role": "system", "content": system}] if system else []) \
        + [{"role": "user", "content": user}]
    body = {
        "messages": messages,
        "temperature": temperature,
        "repeat_penalty": repeat_penalty,    # llama.cpp: anti-repetition
        "seed": seed,
        "max_tokens": max_tokens,
        "stop": STOP,                        # <|im_end|> is NOT eos -> must stop explicitly
        "cache_prompt": cache_prompt,        # llama.cpp: reuse matching KV prefix
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if top_p is not None:
        body["top_p"] = top_p
    if top_k is not None:
        body["top_k"] = top_k
    if chat_template_kwargs:
        body["chat_template_kwargs"] = chat_template_kwargs  # e.g. enable_thinking=false
    t0 = time.perf_counter()
    first = threading.Event()

    # heartbeat: until the first token lands (i.e. during prefill) tick every 1s so the
    # user sees the box is alive even when prefill takes a minute on a starved CPU.
    def _heartbeat():
        while not first.wait(1.0):
            _status(f"  {label} prefilling… {time.perf_counter()-t0:5.1f}s")
    hb = threading.Thread(target=_heartbeat, daemon=True)
    hb.start()

    chunks, reason_chunks, timings, usage = [], [], {}, {}
    ttft_ms = None
    ntok = 0
    finish = None
    try:
        with client.stream("POST", f"{base}/v1/chat/completions",
                            json=body, timeout=600) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    j = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if j.get("timings"):
                    timings = j["timings"]      # llama.cpp puts this in the final chunk
                if j.get("usage"):
                    usage = j["usage"]
                ch = (j.get("choices") or [{}])[0]
                finish = ch.get("finish_reason") or finish
                delta = ch.get("delta") or {}
                rc = delta.get("reasoning_content")  # some builds separate reasoning
                if rc:
                    reason_chunks.append(rc)
                content = delta.get("content")
                if content or rc:
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - t0) * 1000
                        first.set()
                    if content:
                        chunks.append(content)
                    ntok += 1
                    if ntok % 8 == 0:
                        _status(f"  {label} decoding… {ntok} tok  "
                                f"{time.perf_counter()-t0:5.1f}s")
    finally:
        first.set()
        hb.join(timeout=1.0)
        _status("")  # clear the live line

    wall_ms = (time.perf_counter() - t0) * 1000
    parsed = parse_output("".join(chunks), "".join(reason_chunks), finish,
                          strip_scaffold=strip_scaffold)
    return {
        "answer": parsed["answer"],       # best-effort final answer (what we score)
        "reasoning": parsed["reasoning"], # the CoT scaffold (kept for inspection)
        "truncated": parsed["truncated"], # finish_reason == 'length'
        "finish_reason": finish,
        "prompt_ms": timings.get("prompt_ms"),
        "predicted_ms": timings.get("predicted_ms"),
        "prompt_n": timings.get("prompt_n", usage.get("prompt_tokens")),
        "predicted_n": timings.get("predicted_n", usage.get("completion_tokens")),
        "cache_n": timings.get("cache_n"),  # cached-prefix tokens -> verifies CAG reuse
        "ttft_ms": round(ttft_ms, 1) if ttft_ms else None,
        "wall_ms": round(wall_ms, 1),
    }


def run_mode(client, base, mode, variant, fiche, questions,
             *, max_tokens, seed, temperature=0.0, repeat_penalty=1.0,
             top_p=None, top_k=None, chat_template_kwargs=None,
             source_pos="before", model_format="baguettotron",
             progress=None, sink=None):
    """Run all questions in one mode, in order. For cag, order matters: Q1 primes
    the fiche prefix, Q2..N should show prompt_ms collapse (and cache_n jump).
    `progress` = {"k": int, "total": int} for a global counter.
    `sink(row)` (optional) persists each row the instant it completes (crash-safe)."""
    cache_prompt = (mode == "cag")
    # chatml/instruct models answer directly; Baguettotron emits a <think> scaffold to strip
    strip_scaffold = (model_format == "baguettotron")
    rows = []
    for q in questions:
        if progress:
            progress["k"] += 1
        gp = f"[{progress['k']}/{progress['total']}] " if progress else ""
        label = f"{gp}{variant}/{mode}/Q{q['id']}"
        # build the turn per model format
        if mode == "nocontext":
            system, user = SYSTEM_NOCTX, q["question"]
        elif model_format == "chatml":
            # context-first: fiche in the SYSTEM message (stable, cacheable prefix),
            # question in the user turn. This is the CAG-friendly layout.
            system = CHATML_SYS_CTX.format(fiche=fiche)
            user = q["question"]
        else:  # baguettotron: <source_1> tags, NO system message, --source-pos ordering
            system = None
            tmpl = USER_CTX_AFTER if source_pos == "after" else USER_CTX_BEFORE
            user = tmpl.format(fiche=fiche, question=q["question"])
        # announce BEFORE the (possibly minutes-long) call, so it's never silent
        print(f"  {label}  {q.get('type','?')} "
              f"({q.get('question','')[:50]}…) ", flush=True)
        res = ask(client, base, system, user, cache_prompt=cache_prompt,
                  max_tokens=max_tokens, seed=seed, temperature=temperature,
                  repeat_penalty=repeat_penalty, top_p=top_p, top_k=top_k,
                  chat_template_kwargs=chat_template_kwargs,
                  strip_scaffold=strip_scaffold, label=label)
        res.update({"variant": variant, "mode": mode, "id": q["id"],
                    "type": q.get("type"), "leak_risk": q.get("leak_risk"),
                    "requires_context": q.get("requires_context"),
                    "expected": q.get("expected_answer"),
                    "expected_elements": q.get("expected_elements")})
        rows.append(res)
        if sink:
            sink(res)  # persist immediately
        pm = res["prompt_ms"] or 0
        flag = " [TRUNC]" if res["truncated"] else ""
        shown = res["answer"] or "(no answer extracted — see reasoning)"
        print(f"    -> prefill={pm:7.1f}ms cache_n={res['cache_n']} "
              f"decode={res['predicted_ms'] or 0:5.0f}ms/{res['predicted_n']}tok "
              f"wall={res['wall_ms']/1000:.1f}s finish={res['finish_reason']}{flag}\n"
              f"       ans: {shown[:90]!r}", flush=True)
    return rows


def summarize(variant: str, results: dict, per_token_bytes: int | None = None):
    modes = list(results.keys())
    n = len(next(iter(results.values())))

    def col(m, key):
        return [(r.get(key) or 0) for r in results[m]]

    # --- latency table: prefill / decode / total wall, per mode (totals + per-question avg)
    print(f"\n=== [{variant}] latency (ms) — prefill + generation = total wait ===")
    print(f"{'mode':10} | {'prefill':>10} {'decode':>10} {'total':>10}  |"
          f" {'/Q prefill':>11} {'/Q decode':>10} {'/Q total':>9}")
    print("-" * 82)
    for m in modes:
        pf, dc, wl = sum(col(m, "prompt_ms")), sum(col(m, "predicted_ms")), sum(col(m, "wall_ms"))
        print(f"{m:10} | {pf:10.0f} {dc:10.0f} {wl:10.0f}  |"
              f" {pf/n:11.0f} {dc/n:10.0f} {wl/n:9.0f}")

    # --- cache size: cache_n tokens actually reused (from cag mode) x KV bytes/token
    cag = results.get("cag")
    if cag:
        cache_n = max((r.get("cache_n") or 0) for r in cag)
        line = f"--- [{variant}] KV cache: {cache_n} tokens reused"
        if per_token_bytes and cache_n:
            mb = cache_n * per_token_bytes / 1024 / 1024
            line += (f"  x {per_token_bytes/1024:.1f} KB/tok = {mb:.1f} MB"
                     f"  ({per_token_bytes} B/tok, f16 KV)")
        else:
            line += "  (KV bytes/token unknown)"
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--entry", type=int, default=31)
    ap.add_argument("--questions", default=str(ROOT / "bench" / "questions_entry31.json"))
    ap.add_argument("--modes", nargs="+", default=["nocontext", "naive", "cag"],
                    choices=["nocontext", "naive", "cag"])
    ap.add_argument("--context", nargs="+", default=list(CONTEXT_VARIANTS),
                    choices=CONTEXT_VARIANTS,
                    help="context variant(s): answer, answer_exercises, full (default: all "
                         "three; full ~3.5k tok needs a big-window model like Qwen)")
    ap.add_argument("--max-tokens", type=int, default=1024,
                    help="reasoning CoT + answer (model finishes ~600 tok); see docs §5")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temp", type=float, default=0.0,
                    help="sampling temperature (0 = greedy; >0 helps break repetition loops)")
    ap.add_argument("--repeat-penalty", type=float, default=1.0,
                    help="llama.cpp anti-repetition; try 1.2-1.3 if the model loops")
    ap.add_argument("--top-p", type=float, default=None,
                    help="nucleus sampling (Qwen3 non-thinking wants 0.8)")
    ap.add_argument("--top-k", type=int, default=None,
                    help="top-k sampling (Qwen3 wants 20)")
    ap.add_argument("--no-think", action="store_true",
                    help="Qwen3: send chat_template_kwargs {enable_thinking:false} for a "
                         "direct answer (also launch server with --reasoning-budget 0)")
    ap.add_argument("--model-format", choices=MODEL_FORMATS, default="baguettotron",
                    help="baguettotron: <source_1> tags, no system, reasoning scaffold. "
                         "chatml: fiche in system message, direct answer (Qwen/LFM2/SmolLM)")
    ap.add_argument("--source-pos", choices=["before", "after"], default="before",
                    help="[baguettotron only] place <source_1> BEFORE the question (CAG "
                         "prefix reuse) or AFTER it (card-faithful RAG format)")
    ap.add_argument("--corpus", default="",
                    help="MULTI-FICHE CORPUS mode: concatenate several fiches (by entry "
                         "index) into ONE big labeled context, cache it once, and answer "
                         "cross-fiche questions. Accepts an inclusive range '29-38' OR a "
                         "comma list '29,31,33'. Bypasses --context/--entry; the single "
                         "context variant is named 'corpus'. Pass a corpus questions file "
                         "via --questions.")
    ap.add_argument("--corpus-field", choices=["answer", "full"], default="answer",
                    help="[corpus only] how much of each fiche to include: 'answer' "
                         "(lesson content, default) or 'full' (whole fiche).")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap number of questions (0 = all); handy for a quick smoke test")
    ap.add_argument("--tag", default="",
                    help="label for this run; goes in the output filename (e.g. pi, brief)")
    ap.add_argument("--tokens-only", action="store_true",
                    help="just report exact tokenized length of each variant and exit")
    args = ap.parse_args()

    data = json.loads(DATA.read_text())

    # MULTI-FICHE CORPUS mode: one big labeled context named "corpus" (bypasses --context/
    # --entry). Otherwise the usual per-single-entry context variants.
    corpus_mode = bool(args.corpus)
    corpus_indices = parse_indices(args.corpus) if corpus_mode else []
    if corpus_mode:
        context_texts = {"corpus": build_corpus(data, corpus_indices, args.corpus_field)}
    else:
        entry = data[args.entry]
        context_texts = {v: build_fiche(entry, v) for v in args.context}
    context_names = list(context_texts.keys())

    ctk = {"enable_thinking": False} if args.no_think else None

    with httpx.Client() as client:
        # sanity: server up?
        try:
            client.get(f"{args.base}/health", timeout=5).raise_for_status()
        except Exception as e:
            sys.exit(f"Server not reachable at {args.base} — start it with "
                     f"./bench/run_server.sh  ({e})")

        # KV-cache sizing + real context budget: read from /props once.
        #   model_path        -> GGUF dims -> KV bytes/token
        #   default_generation_settings.n_ctx -> the server's REAL context window (the true
        #   budget). Using it (not a hardcoded 4096) stops big corpora from falsely flagging
        #   OVERFLOW on a large-window model.
        per_token_bytes = None
        budget = 4096
        try:
            props = client.get(f"{args.base}/props", timeout=5).json()
            mp = props.get("model_path")
            if mp:
                per_token_bytes = kv_bytes_per_token(_gguf_kv_dims(mp) or {})
            n_ctx = (props.get("default_generation_settings") or {}).get("n_ctx")
            if n_ctx:
                budget = int(n_ctx)
        except Exception:
            pass

        fiche_tok = {}
        for v in context_names:
            fiche = context_texts[v]
            n_tok = server_tokenize(client, args.base, fiche)
            fiche_tok[v] = n_tok
            used = n_tok + 50 + args.max_tokens
            kv = (f" | KV ~{n_tok*per_token_bytes/1024/1024:.1f} MB"
                  if per_token_bytes else "")
            print(f"[{v:16}] {len(fiche)} chars = {n_tok} tok  | "
                  f"+50 q +{args.max_tokens} gen = {used}/{budget} "
                  f"{'OK' if used <= budget else '!! OVERFLOW'}{kv}")
        if args.tokens_only:
            return

        questions = json.loads(Path(args.questions).read_text())["questions"]
        if args.limit:
            questions = questions[:args.limit]
        total = len(questions) * len(args.modes) * len(context_names)
        progress = {"k": 0, "total": total}
        print(f"\nRunning {len(questions)} questions x {len(args.modes)} modes "
              f"x {len(context_names)} context variants = {total} calls "
              f"(max_tokens={args.max_tokens}, seed={args.seed})")
        print("NOTE: streaming live status shown per call; on a loaded CPU each call "
              "can take minutes — the prefill/decode heartbeat means it's working.")

        # --- run-tagged output, never overwrites; JSONL saved incrementally ---
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        run_ts = time.strftime("%Y%m%d-%H%M%S")
        if corpus_mode:
            base_stem = "corpus" + args.corpus.replace(",", "-")
        else:
            base_stem = f"entry{args.entry}"
        stem = base_stem + (f"_{args.tag}" if args.tag else "") + f"_{run_ts}"
        jsonl_path = RESULTS_DIR / f"{stem}.jsonl"
        json_path = RESULTS_DIR / f"{stem}.json"
        meta = {"_meta": True, "entry": args.entry, "tag": args.tag,
                "fiche_tokens": fiche_tok, "max_tokens": args.max_tokens,
                "seed": args.seed, "temp": args.temp,
                "repeat_penalty": args.repeat_penalty, "top_p": args.top_p,
                "top_k": args.top_k, "no_think": args.no_think,
                "source_pos": args.source_pos, "model_format": args.model_format,
                "modes": args.modes, "context_variants": context_names,
                "corpus": bool(corpus_mode), "corpus_indices": corpus_indices,
                "corpus_field": args.corpus_field if corpus_mode else None,
                "budget": budget, "base": args.base, "run_ts": run_ts}
        print(f"Saving live to {jsonl_path.name} (crash-safe; tail -f to watch)\n")

        all_results = {}  # variant -> mode -> rows
        with jsonl_path.open("w", encoding="utf-8") as jf:
            jf.write(json.dumps(meta, ensure_ascii=False) + "\n"); jf.flush()

            def sink(row):
                jf.write(json.dumps(row, ensure_ascii=False) + "\n"); jf.flush()

            for v in context_names:
                fiche = context_texts[v]
                print(f"\n########## CONTEXT VARIANT: {v} ##########")
                per_mode = {}
                for mode in args.modes:
                    print(f"--- [{v}] mode: {mode} ---")
                    per_mode[mode] = run_mode(client, args.base, mode, v, fiche,
                                              questions, max_tokens=args.max_tokens,
                                              seed=args.seed, temperature=args.temp,
                                              repeat_penalty=args.repeat_penalty,
                                              top_p=args.top_p, top_k=args.top_k,
                                              chat_template_kwargs=ctk,
                                              source_pos=args.source_pos,
                                              model_format=args.model_format,
                                              progress=progress, sink=sink)
                all_results[v] = per_mode

    for v in context_names:
        summarize(v, all_results[v], per_token_bytes)

    json_path.write_text(json.dumps(
        {**meta, "results": all_results}, ensure_ascii=False, indent=2))
    print(f"\nWrote {json_path}\n  (+ incremental {jsonl_path.name})")


if __name__ == "__main__":
    main()
