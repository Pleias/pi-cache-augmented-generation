# Raspberry Pi — setup status & how to launch

**Date:** 2026-07-03
**Purpose:** the Pi is the real CAG target (slow prefill → CAG payoff shows). This records
what is already on the Pi so we don't lose it. See `cag_benchmark_design.md` for the why.

## Host
- User/host: `pleias@pleias`
- llama.cpp: **built from source** at `~/llama.cpp` → binary `~/llama.cpp/build/bin/llama`
  (new unified CLI — subcommand `llama serve`, **not** the standalone `llama-server`).
- Model: `/home/pleias/custom_models/Baguettotron-Q8_0.gguf`
  (343,656,288 bytes — byte-identical to the x86 box; same Q8_0 quant → valid cross-device
  comparison).

## Confirmed working
`llama serve` loads the model and serves:
```
srv  load_model: initializing, n_slots = 4, n_ctx_slot = 4096, kv_unified = 'true'
srv  llama_server: model loaded
srv  llama_server: listening on http://127.0.0.1:8080
```
Same `n_ctx = 4096`, same port `8080` as the x86 host → the Python runner
(`bench/cag_bench.py`) points at it unchanged (`--base http://<pi>:8080`).

**Measured performance (2026-07-04, dedicated):** prefill **~290 tok/s**, decode **~28 tok/s**.
Per question ≈ 6.5 s fiche prefill + ~18 s decode. This is the clean, reproducible
measurement environment — unlike the shared x86 Xeon (load ~55, ~1 tok/s = contention noise).
Fiche prefill (~6.5 s) is what `cag` mode skips on cached questions.

## Gotcha already hit
First launch failed with `failed to open GGUF file ... (No such file or directory)` — the
path had a stray **curly quote** `’` appended (`...Baguettotron-Q8_0.gguf’`). Copy-paste
smart-quote artifact. Re-running with a clean straight path worked. If load fails with
"No such file", check for a trailing `’`/`'`.

## Launch for the benchmark (flags matter)
The bare `llama serve -m ...` above uses defaults — but our benchmark needs the same flags
as the x86 launcher (`bench/run_server.sh`):
```bash
~/llama.cpp/build/bin/llama serve \
  -m /home/pleias/custom_models/Baguettotron-Q8_0.gguf \
  --ctx-size 4096 \
  --jinja \
  --slot-save-path ~/kv_cache \
  --host 0.0.0.0 --port 8080
```
- `--jinja` — apply the ChatML template (reasoning `<think>` behaviour; see design §5).
  Without it the model output format may differ.
- `--slot-save-path` — enables disk KV save/restore (the Pi cold-start CAG scenario).
- `--host 0.0.0.0` — only if driving it from another machine; else `127.0.0.1`.
- Verify the flag names against this build's `llama serve --help` (source build may differ
  slightly from the `b9864` prebuilt used on x86).

## Run the benchmark against the Pi
From the dev machine (or on the Pi with the repo synced):
```bash
uv run bench/cag_bench.py --base http://<pi-ip>:8080 --tokens-only   # confirm fit
uv run bench/cag_bench.py --base http://<pi-ip>:8080                 # full run
```
Same quant, fiche (entry 31), and questions as x86 → the prefill-time delta (naive vs cag)
on the Pi is the **headline result**. Unlike the shared x86 host (contended, load ~17), the
Pi should give clean, reproducible timings.

## TODO / open
- [ ] Relaunch `llama serve` on the Pi with the benchmark flags above.
- [ ] Confirm `--slot-save-path` / `--jinja` flag names on this source build.
- [ ] Get the Pi's IP / reachability from the dev box (or run the repo on the Pi).
- [ ] Record baseline `--tokens-only` + a first `--limit 2` smoke run.
