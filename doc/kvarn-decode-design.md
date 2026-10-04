# KVarN generation: compile-friendly backend (design)

## Goal

Close generation 12.1 -> ~86 tok/s (fp16 parity) on the 27B-dense @8192
with zero quality change: KLD digits identical, same-top 100%,
79-test suite green. Current state is the torch-eager ceiling; this doc
specifies the structural work that gets past it.

## Evidence (why this design and nothing smaller)

- Kineto, 5 decode steps @8192: ~3000 aten calls/step, Self CPU 112ms
  vs Self CUDA 19ms -- host-starved. Top CPU: index 29ms (448 calls x
  ~65us dispatch each), copy_ 18ms, nonzero/unique 18ms, to-casts 9ms,
  109 DtoH syncs/step. Per-token per-layer: store ~3.7ms, serve
  ~1.1ms, attention ~0.7ms (x16 cached layers).
- Micro-opts landed (+36%: fast-path store, touch vectorize, stacked
  WHT, evict cadence, ping-pong WHT, Triton row-WHT) then stopped
  moving the number. Two ideas measured ~zero or negative and were
  reverted (in-place overlay + stash restore; full-clone theory).
- torch.compile probe (`eval/kvarn_compile_probe.py`): 585 dynamo
  calls into 63 unique graphs, recompile limit hit on id-keyed
  `params['dev_cache']` in `attn.py`. A flag flip cannot fuse this
  code; only static structure can.

## Constraints (both paths below)

- Bit-exact vs current torch path, proven by parity asserts + KLD +
  suite -- never by reasoning alone.
- Gated development (opt-in env flags, stock behavior untouched).
- VRAM discipline: 100MB-free rule every run; persistent allocations
  sized and documented.
- Triton-only preferred; C++ extension changes must build with
  MSVC + nvcc 13.3 + torch 2.11/cu130.

## Path A -- fused store/serve Triton kernels (~2x, NOT parity)

Per token per layer is ~40 launches + ~10 syncs. Kernels collapse
launches, not syncs: fused store (row WHT + stage/exact write +
present flags, bails to torch on policy events) + fused serve
(refresh + overlay gather + scatter). Seal path stays torch
(Sinkhorn, 1/128 tokens). Expected ceiling ~25 tok/s. Do this only
if a quick ~2x is needed before Path B lands; B subsumes it.

## Path B -- compile-friendly backend (the actual fix)

Replace every dynamism source so `torch.compile` of the decode step
produces O(1) graphs with zero fallbacks:

### Phase 1: densify storage (this doc's first work item)

- `stage_blocks: dict[group, [k, v]]` -> static `stage_k/stage_v`
  `(num_groups, 128, kvh, hd)` half + residency from existing
  `present` mask. Cost: ~33MB/layer @8k/27B (same order as the
  persistent image; budgeted, documented).
- `exact_blocks: dict[group, [k, v]]` -> static `exact_k/exact_v`
  `(num_groups, 128, kvh, hd)` tail_dtype + `exact_present`
  `(num_groups, 128)` bool. Same cost again.
- Readers/writers migrated one function at a time, suite green after
  each: `_store_rows` + fast path, `_seal_*`, `_evict_exact_all`,
  `_apply_exact_overlay`, `_overlay_index`, `copy_page`,
  `_staging_from_records`, `get_tensors`, storage accounting.
- Tests asserting dict internals (`exact_blocks.keys()`,
  `stage_blocks.keys()`) rewritten to mask equivalents with identical
  logical assertions (resident sets, seal counts).

### Phase 2: static control flow

- Zero `.item()`/`.tolist()` in steady state: group/base/sealed/
  owner decisions as masked tensor ops; seals padded to a fixed
  max-per-call; eviction already cadenced.
- `copy_page`, page-reuse reset, unseal: slow paths allowed Python,
  but must reset any shape guards (recompile on policy events is
  acceptable: they are rare; steady decode must never recompile).
- Framework cooperation: `params['dev_cache']` id-keyed guard in
  `attn.py` recompiles per call -- fix the key (or exclude that
  region from the compiled unit). Kvarn-only changes cannot fix this
  alone; coordinate, don't work around.

### Phase 3: compile + prove

- `torch.compile` the decode step (default inductor, static shapes).
  Acceptance, all required: KLD digits identical, suite green,
  `suppress_errors=False` (no silent fallback), dynamo counters show
  O(1) unique graphs and no recompile-limit hits across steps,
  decode >= 60 tok/s (70% of fp16). Validate 27B-dense first,
  Flash-Next (QSA/MoE) second.
- Keep the Kineto + `--decode` + KLD + VRAM-guard harness as the
  acceptance suite for every phase.

## Risks

- Silent divergence (masked-op rewrite of policy logic). Mitigation:
  parity mode during dev (run old vs new side by side, assert equal),
  then the standard suite + KLD + decode-quality spot checks.
- Memory story changes (dense exact/stage ~1GB @8k/27B): re-measure
  and re-document the compact-memory claims; the `storage_size` /
  `overhead_size` accounting must include the new tensors honestly.
- Compile time / RAM for inductor on the full step: stage it
  (per-layer functions first if full-step fails to trace).

## Out of scope

- Prefill (done at 1.3x fp16, 32x total).
- Speculative (draft-target) decoding: orthogonal multiplier,
  product decision.
- BeeLlama timing parity: no shared harness exists; Bee is mined for
  architectural ideas only (final todo), not numbers.

## Perf history pointer

Prefill 134.6s -> 4.2s and generation 8.9 -> 12.1 tok/s step tables
live in `doc/kvarn-4090.md` (regression baseline, plain decimals).
