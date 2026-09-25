# Branches

| Branch | Upstream issue | Status | PR |
| --- | --- | --- | --- |
| `wip/kvarn-cache` | — (feature port, no upstream issue) | WIP: CPU-complete (M1–M5, 77 tests green); Triton dequant bootstrap untested, blocked on GPU hardware | [#2 (draft)](https://github.com/aand18/exllamav3/pull/2) |

Delete rows when the branch is merged or deleted.

---

## `wip/kvarn-cache`

**Goal:** Port Huawei KVarN variance-normalized KV-cache quantization
(arXiv:2606.03458, as adapted by BeeLlama) into exllamav3 as a pluggable
`CacheLayer`: longer context in same VRAM, better quality at same bits.
Targets: Qwen 3.6/3.8 dense, Qwen 3.8-Flash-Next (Qwen4Exp).

**Upstream issue:** none — feature port, not a fix. No upstream PR planned
until hardware validation lands.

**Non-goals (yet):** fused online CUDA/Triton kernels at production quality
(bootstrap present, untested); model-level KLD numbers (needs GPU);
upstream PR.

**Status:** CPU-complete through M5. Full 36-width table, Bee sink + tail
policy (`--kv-tail-tokens/--kv-tail-type`), SWA window tail + per-side
overrides (`--kv-swa-k/--kv-swa-v`), compact arenas (0.34–0.40× fp16),
dense + QSA coverage, state versioning, autosplit/BC handling. Triton
dequant bootstrap added behind `EXL3_KVARN_TRITON=1` (default off), never
launched — `EXL3_KVARN_TRITON_PARITY=1` is its acceptance test.

**Test plan:**
- CPU unit/parity/accounting suite: `tests/test_kvarn_*.py` — 77 green in
  triton-free and triton-present venvs.
- `eval/kvarn_microkld.py -m <dir> -cq kvarn4 -ntok 200` on a GPU box.
- Triton parity mode on the first 4090 run.
- 4090 handoff (clone, build, validate, report): [doc/kvarn-4090.md](./doc/kvarn-4090.md).

**Verify locally:**
`kvarn-venv\Scripts\python.exe -m pytest tests/test_kvarn_cpu.py tests/test_kvarn_tail_cpu.py tests/test_kvarn_widths_cpu.py tests/test_kvarn_m4_cpu.py tests/test_kvarn_m5_cpu.py tests/test_kvarn_triton.py -q`

**Key commits (bottom-up):** review bundle (`379ae7a`…`957a01e`), dispatch
CPU fix (`8f3b8ea`), micro-KLD harness (`b649766`), Triton bootstrap
(`e27e247`), workflow docs.

**Blocked on:** GPU hardware (4090 box) for Triton parity, micro-KLD
numbers, and any perf work. `ext` is CUDA-only — no CPU model forward is
possible on this machine.

**Branch history note:** renamed `kvarn` -> `wip/kvarn-cache` per naming
rule; draft PR #1 superseded by #2.
