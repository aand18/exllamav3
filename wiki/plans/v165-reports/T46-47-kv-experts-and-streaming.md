# T46/T47 — the KV→experts move fits when sized correctly, and the streaming gates lose

Campaign of 2026-11-05/06, one session, server stopped, production regime otherwise unchanged
(DFlash2 w4, `-cq 4,4`, `-mcs`/`-mct` as noted), one pass per arm, `-dr 1`.

## Why these arms exist

The earlier KV→experts arm (`t37-B1`) and every streaming arm (`t36`) died on the **VRAM floor** at 23 MB
and ~110 MB free. Both tried to spend the KV that was freed *and* keep it. This campaign splits the freed
headroom between the two experiments instead, and caps the ladder at 32 k because a 32 k cache cannot
hold a longer sequence.

## Results, same session

| arm | configuration | PP 32768 | TG 32512 | min free VRAM | min free RAM |
|---|---|---|---|---|---|
| `t46-A-ref32k` | cache 262 144, split 381 (reference at this ladder) | 1787.10 | 38.42 | 2 267 MB | 5 120 MB |
| `t46-B-kv2exp` | cache **32 768**, split **360** (+21 GPU experts/layer) | **1844.60 (+3.2 %)** | **40.55 (+5.5 %)** | **2 943 MB** | 8 900 MB |
| `t46-C-stream` | cache 32 768, split 381, `STREAM_MIN_ROWS=2` + `STREAM_T=2` | 1770.59 (−0.9 %) | 36.72 (**−4.4 %**) | 4 899 MB | 5 697 MB |
| `t47-kv2exp-ccs` | B + `-ccs 2.0` (host KV tier), full-ladder flags | 1857.18 (+3.9 %) | 39.60 (+3.1 %) | 2 943 MB | 7 294 MB |

All four `status=ok`, **no floor breach**.

## What each says

- **KV→experts fits and is faster at ≤32 k.** `-cs 32768 -mcs 360` is +3.2 % prefill and +5.5 % decode
  against the same-session reference, and it has **more** VRAM headroom (2 943 MB vs 2 267 MB) because the
  freed KV exceeds the expert move. The earlier breach was arithmetic, not physics.
- **The M3 streaming gates are now tested and lose.** With 4.9 GB of headroom they run cleanly and are
  **−4.4 % decode, −0.9 % prefill**. Decode-size streaming of multi-hit misses costs more than it saves on
  this box.
- **`-ccs` buys nothing at 32 k** (B+ccs is 2.3 % *slower* decode than B): there is no spill to serve at
  this length, so the host tier is overhead.

## The limit that decides deployability

**A 32 k cache cannot serve 130 k context, and `perf.py` caps its ladders at the cache size** — every arm
here stops at 32 768 prefill and 32 512 decode, including `t47` with `-MaxLength 131072`. So B is a
**short-context** optimisation by construction. Serving the long requests that dominate this box's traffic
(300 of 731 at ≥128 k, T14) with a 32 k cache requires the **server-side** host KV tier
(`sysmem_kv_cache` in `tabbyAPI/config.yml`), which no `perf.py` arm exercises — that is a server test, not
another pass.

## Caveats

One draw per arm, ladder capped, `-dr 1`. B's +5.5 % needs a repeat before it is quoted as a configuration,
and the long-context case is unmeasured rather than good or bad.

## Run ids

| arm | run id | status |
|---|---|---|
| `t46-A-ref32k` | `t46-A-ref32k-20261105-225430` | ok |
| `t46-B-kv2exp` | `t46-B-kv2exp-20261105-225745` | ok |
| `t46-C-stream` | `t46-C-stream-20261105-230052` | ok |
| `t47-kv2exp-ccs` | `t47-kv2exp-ccs-20261105-230746` | ok |
