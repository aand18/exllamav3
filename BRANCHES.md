# Branches

| Branch | Upstream issue | Status | PR |
| --- | --- | --- | --- |
| `wip/kvarn-cache` | — (feature port, no upstream issue) | WIP: 4090-validated, 64.5 tok/s decode @8192 (27B dense 1.4bpw, fp16 86–87 same runs); micro-KLD identical same-top 100%, PARITY=1 clean, CPU 77 + triton twins 10 green; optimizing the generation wall | [#2 (draft)](https://github.com/aand18/exllamav3/pull/2) |
| `feat/specter-feasibility` | — (feasibility study, no upstream issue) | Study only, no runtime change: paper's artifact not portable (INT4 draft 61.8 GiB, premise mismatch, recurrent-draft assert), its look-ahead idea does transfer; measurement plan written, port plan phased | [#3 (draft)](https://github.com/aand18/exllamav3/pull/3) |

Delete rows when the branch is merged or deleted.

---

## `wip/kvarn-cache`

**Goal:** Port Huawei KVarN variance-normalized KV-cache quantization
(arXiv:2606.03458, as adapted by BeeLlama) into exllamav3 as a pluggable
`CacheLayer`: longer context in same VRAM, better quality at same bits.
Targets: Qwen 3.6/3.8 dense, Qwen 3.8-Flash-Next (Qwen4Exp).

**Upstream issue:** none — feature port, not a fix. No upstream PR planned
until hardware validation lands.

**Non-goals (yet):** production-quality fused online CUDA/Triton kernels
beyond the current serve path; model-level KLD numbers; upstream PR.

**Status:** CPU-complete through M5 (see above) plus 4090-validated
generation work, tracked in
[`doc/kvarn-4090.md`](https://github.com/aand18/exllamav3/blob/wip/kvarn-cache/doc/kvarn-4090.md)
on the branch: prefill 134.6s → ~5.0s, decode 8.9 → 64.5 tok/s @8192
(27B dense 1.4bpw, fp16 86–87 same runs) via fused Triton serve/store
kernels, sync-free evict scan, global dirty sweep + in-place WHT,
steady-path call trimming, serve-from-image stash-first overlay,
store write-through + dirty flag, and tick-gated touch. Micro-KLD vs fp16 identical (same-top
100%); `EXL3_KVARN_TRITON_PARITY=1` clean at 8k; CPU suite 77 passed
6 skipped; triton twins 10 passed. Current wall is per-step launch count
(Kineto: host-starved, Self CPU ≫ Self CUDA), not syncs.

**Test plan:**
- CPU unit/parity/accounting suite: `tests/test_kvarn_*.py` — 77 green
  6 skipped (TRITON unset; triton tests CUDA-gated).
- `eval/kvarn_microkld.py` 8k KLD + 64-step decode bench on the 4090 box
  — identical digits, same-top 100% (gates every perf commit).
- Triton parity mode (`EXL3_KVARN_TRITON_PARITY=1`) at 8k — clean.
- 4090 report: [doc/kvarn-4090.md](./doc/kvarn-4090.md) (lives on the
  branch, updated with each perf commit).

**Verify locally:**
`kvarn-venv\Scripts\python.exe -m pytest tests/test_kvarn_cpu.py tests/test_kvarn_tail_cpu.py tests/test_kvarn_widths_cpu.py tests/test_kvarn_m4_cpu.py tests/test_kvarn_m5_cpu.py tests/test_kvarn_triton.py -q`

**Key commits (bottom-up):** review bundle (`379ae7a`…`957a01e`), dispatch
CPU fix (`8f3b8ea`), micro-KLD harness (`b649766`), Triton bootstrap
(`e27e247`), workflow docs.

**Blocked on:** nothing — autonomous 4090 optimization loop running
(decode 64.5 tok/s @8192, next target: per-step launch count).

**Branch history note:** renamed `kvarn` -> `wip/kvarn-cache` per naming
rule; draft PR #1 superseded by #2. 4090 perf commits (bottom-up):
fused serve `52ba530` (28.8 tok/s), sync-free evict `8142fc0`,
dirty sweep + in-place WHT `7596288` (32.0 tok/s), steady-path
trimming `f1d93ee` (35.2 tok/s), serve-from-image stash-first
`981b99a` (36.2 tok/s), store write-through + dirty flag `02787f7`
 (49.3 tok/s), tick-gated touch `2c4fa21` (64.5 tok/s). 2026-09-28:
 rebased onto `origin/fork-overview` (house base rule changed off stale
 Sep-10 master; zero content conflicts in kvarn paths, CPU 90+7 green).

---

## `feat/specter-feasibility`

**Goal:** decide whether the Specter paper (arXiv:2511.14102, ASPLOS'27) can be
implemented in exllamav3 for Qwen3.8-Flash-Next, and write down a plan that
survives the answer. Deliverable is the study plus a budget calculator, not a
port.

**Upstream issue:** none - feasibility study. The CPU MoE offload path and the
Qwen3.8-Flash-Next support it studies are already upstream; the fork adds only
docs and the study.

**Non-goals:** no exllamav3 runtime change in this branch; no literal port of
the paper's artifact (INT4 draft of the target, ELB/top-M GPU expert cache);
no upstream PR until a measurement justifies the work.

**Status:** study complete, no runtime change. Three findings: (1) the paper
optimizes a bottleneck this engine does not have - exllamav3 ships activations
not weights, 480 KiB/token vs 1.17 GiB/token, 2550x less traffic, so there is
no PCIe stall to hide; the paper's draft is 61.8 GiB and the AR-draft path
rejects recurrent architectures (`generator.py:162`), which every Gated DeltaNet
model is; (2) at 512 experts/layer the paper's cache-centric design does not
fit - the union over K=16 is ~67 experts/layer, 7.8 GiB/step against a ~13 GiB
expert budget, so only a per-layer streaming design survives; (3) the idea that
does transfer is a predictor of future routing, and the right predictor is not a
quantized copy of the model: the fork's placement is purely retrospective, and
the router fires a whole layer before the weights are needed, so the slow
host-computed layer can serve as the prefetch window. Plan is measurement
first (host cost per offloaded expert vs bus cost of the same expert decides
between two designs ~2x apart in TPOT), then a decode-time streaming path,
then the look-ahead on top.

**Test plan:** no GPU run needed for the study. `python eval/specter_budget.py
--preset qwen3.8-flash-next --vram-gb 24` reproduces every number in the
study; the paper's three models are available as presets for comparison
(`--preset dsv2-lite|qwen1.5-moe|phi3.5-moe`). The study's host-cost
constants are assumptions and are replaced by `--cpu-us-per-expert` once
measured on the 4090 box.

**Verify locally:**
`python eval/specter_budget.py --preset qwen3.8-flash-next --vram-gb 24` and
read `doc/specter-feasibility.md`.

**Key commits (bottom-up):** budget calculator (`a39e677`), feasibility study
(`a2748ce`).

**Blocked on:** a 4090 box for the measurements listed in the study (host cost
per expert, link bandwidth, baseline TPOT, routing statistics); a
Qwen3.8-Flash-Next EXL3 checkpoint, which means converting the 125 B FP16
release plus a separate n-gram table file; `eval/smi_guard.py`, which this
repo's rules require for CUDA runs and which does not exist in-tree.

**Branch history note:** cut from `origin/fork-overview` (54177b5), which is
byte-identical to upstream `turboderp-org/exllamav3` master apart from the
fork's docs - the offload path and the Qwen3.8 support under study are
upstream code, so a successful phase 2 would be an upstream PR.
