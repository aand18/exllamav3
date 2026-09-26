# Branches

| Branch | Upstream issue | Status | PR |
| --- | --- | --- | --- |
| `wip/kvarn-cache` | — (feature port, no upstream issue) | WIP: 4090-validated, 36.2 tok/s decode @8192 (27B dense 1.4bpw, fp16 85–86 same runs); micro-KLD identical same-top 100%, PARITY=1 clean, CPU 77 + triton twins 10 green; optimizing the generation wall | [#2 (draft)](https://github.com/aand18/exllamav3/pull/2) |

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
on the branch: prefill 134.6s → ~5.0s, decode 8.9 → 36.2 tok/s @8192
(27B dense 1.4bpw, fp16 85–86 same runs) via fused Triton serve/store
kernels, sync-free evict scan, global dirty sweep + in-place WHT,
steady-path call trimming, and serve-from-image stash-first overlay.
Micro-KLD vs fp16 identical (same-top
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
(decode 36.2 tok/s @8192, next target: per-step launch count).

**Branch history note:** renamed `kvarn` -> `wip/kvarn-cache` per naming
rule; draft PR #1 superseded by #2. 4090 perf commits (bottom-up):
fused serve `52ba530` (28.8 tok/s), sync-free evict `8142fc0`,
dirty sweep + in-place WHT `7596288` (32.0 tok/s), steady-path
trimming `f1d93ee` (35.2 tok/s), serve-from-image stash-first
`981b99a` (36.2 tok/s).
