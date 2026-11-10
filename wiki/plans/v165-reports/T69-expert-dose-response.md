# T69 — dose-response on "more GPU experts": 3, then 8, both zero. 2c's prize collapses.

The question this answers: does buying GPU expert slots with freed VRAM reduce decode time? It was
asked three times at three doses. Basis: `spec_decode`, 131 072 decode, exl3 drafter, static
placement, pin 24 GiB, slots 4.

## The three doses

| dose | configuration | result |
|---|---|---|
| +3 experts/layer | T62: cache 262144, split 381 → 378 | **+0.28 %** (inside spread) |
| +8 experts/layer | T69: cache 40960, split 381 → 373 | **0.00 %**, mechanism flat |
| +16 and +26 | T68: split 365 / 355 at cache 40960 | **VRAM floor breach, killed** — not measurable |

### T69 detail (the usable dose)

| split | draft tok/s | rows=5 distinct (median) | `cmb` median | µs/expert |
|---|---:|---:|---:|---:|
| 381 (`t68E`, `t68H`, `t68J`) | 57.06 / 58.17 / 58.19 | **20.0** | **0.676** | 33.8 |
| **373** (`t69A`, `t69B`) | **58.60 / 59.67** | **20.0** | **0.679** | 33.9 |

Agentic-weighted primary **1.0185 against a 1.039 floor → `INCONCLUSIVE`**, and acceptance is
identical (2.80 → 2.80) with Agentic-curl +3.61 % and Coding −0.17 %. **No mechanism moved** —
distinct experts unchanged at 20.0, `cmb` unchanged within 3 µs — so the primary movement is noise,
not an effect.

## Why this is more informative than "another null"

The prediction going in was explicit: the routing profile says ranks 131..159 (the next 29 GPU
slots) hold **5.24 %** of stationary mass, so +8 slots should cut distinct misses by ~0.7 and `cmb`
by ~25 µs/layer. **Measured: 0.0 experts and 3 µs.** The profile arithmetic is falsified in the same
direction T65 found — per-conversation routing is far broader than the stationary profile, so the
experts sitting at the GPU/CPU boundary are worth even less than the profile says.

## What that does to 2c

2c (KV residency → ~2.5 GB of VRAM → ~29 expert slots) was priced at **+5–8 %** by profile
arithmetic. That arithmetic has now failed its own prediction at one eighth of the dose, in the same
direction, on a measurement of the mechanism itself rather than of tok/s. The honest estimate for
+29 slots is **~1–2 %, inside the spread**, and the box cannot reach the full dose to check
(`split 355` and `365` both broke the 200 MB VRAM floor; only ~1.4 GiB of the claimed 2.4 GiB
headroom at `cs 40960` is real).

**Recommendation: do not implement 2c.** It is weeks of attention-backend work whose measured
first step (a third of the dose) returns zero, and whose full dose this machine cannot execute.

## The floor this leaves

Decode time is `20 CPU-resident experts/layer × 33.9 µs × 48 layers ≈ 0.69 ms/layer` of pure DDR5
read, against `ker` 0.118 + `mixer` 0.101 + `sub` 0.047 ms of GPU work. Nothing measured in T57–T69
changes either factor:

- experts **per layer**: placement (T66), split with 3 (T62) or 8 (T69) more slots, pinning (T67) —
  all zero on `cmb`;
- bytes **per expert**: 2 bpw is an operator decision; the engine holds one K per layer.

The one lever the arithmetic still admits is **accepted tokens per round** — fewer CPU experts per
token — i.e. the drafting side, not the MoE.

## Runs

| tag | run id | status |
|---|---|---|
| `t68E-cs40-mcs381` | `t68E-cs40-mcs381-20261110-211419` | ok |
| `t68F-cs40-mcs355` | `t68F-cs40-mcs355-20261110-213100` | breach |
| `t68H-cs40-mcs381` | `t68H-cs40-mcs381-20261110-215158` | ok |
| `t68I-cs40-mcs365` | `t68I-cs40-mcs365-20261110-220852` | breach |
| `t68J-cs40-mcs381` | `t68J-cs40-mcs381-20261110-221041` | ok |
| `t69A-cs40-mcs373` | `t69A-cs40-mcs373-20261110-223004` | ok |
| `t69B-cs40-mcs373` | `t69B-cs40-mcs373-20261110-224657` | ok |

`t68A-cs262-mcs381-20261110-203436` and `t68G-cs262-mcs381-20261110-213300` are the 262 144 cache
references (57.03 / 57.47 tok/s) used for the cache-effect line. The two `breach` rows are cited only
as evidence that the dose is unexecutable on this box; they carry no measurement.
