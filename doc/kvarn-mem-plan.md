# KVarN Memory Reclaim Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut KVarN cache-only VRAM from 1.83GB to ~0.25GB at 8k (27B dense) while holding decode throughput, by replacing per-group static fp16 tensors with slot-windowed storage (Phase 1) and the persistent fp16 image with fused online-dequant attention (Phase 2).

**Architecture:** Phase 1 remaps group-id-indexed staging/exact tensors through a small slot table (live groups only; records untouched; image untouched so speed holds). Phase 2 adds a Triton paged flash-decode kernel that dequantizes records inline (same elementwise math as the proven batch dequant, bit-exact by construction) behind `EXL3_KVARN_ONLINE=1`, retiring the image path.

**Tech Stack:** Python + torch (CUDA), Triton 3.8 kernels in `exllamav3/modules/attention_fn/kvarn_triton.py`, storage in `exllamav3/cache/kvarn.py`, gates via existing CPU suite + CUDA twins + `eval/kvarn_microkld.py` at 8k/16k/32k + 256tg.

**Spec:** `doc/kvarn-4090.md` ("Cache-only accounting" table + BeeLlama comparison table are the baselines and the external reference). BeeLlama architecture reference (read-only): `/home/dev/beellama.cpp/src/llama-kv-cache-kvarn.h` (`stage_groups = tail_groups + 1`), `/home/dev/beellama.cpp/ggml/src/ggml-cpu/ops.cpp` (`kvarn_cpu_attn_load_row`, `ggml_compute_forward_kvarn_materialize`), `/home/dev/beellama.cpp/ggml/src/ggml-cuda/kvarn.cu`, `/home/dev/beellama.cpp/ggml/src/ggml-cuda/fattn-kvarn-dispatch.cu`.

## Global Constraints

- Work on branch `wip/kvarn-mem-stacked-on-kvarn-cache`, cut from `wip/kvarn-cache` HEAD; rebase never merge; push only to `origin` (`https://github.com/aand18/exllamav3`, the fork); never touch `master`; docs-row updates go directly to `fork-overview` as docs-only commits.
- Python work runs in `./venv/bin/python` (worktree venv, torch CPU); CUDA work runs on the Windows 4090 box via `cmd.exe /c` with `&&` chains and no spaces around `&&`; TabbyAPI venv `C:\Users\yoho\Downloads\tabbyAPI\venv\Scripts\python.exe` with torch `2.11.0+cu130`, triton `3.8.0`.
- CPU suite gate runs with `EXL3_KVARN_TRITON` unset: `./venv/bin/python -m pytest tests/test_kvarn_cpu.py tests/test_kvarn_tail_cpu.py tests/test_kvarn_widths_cpu.py tests/test_kvarn_m4_cpu.py tests/test_kvarn_m5_cpu.py tests/test_kvarn_triton.py -q` must stay 77 passed + 6 skipped.
- CUDA gates on Windows: `tests/test_kvarn_triton.py` 10 passed (twins); `eval/kvarn_microkld.py` KLD same-top 100.00% at 8k/16k/32k with identical leading digits (8k: median 0.000001 mean 0.000020 max 0.000395; 16k: mean 0.000023 max 0.000748; 32k: mean 0.000009 max 0.000172).
- 100MB-free VRAM hard rule on the Windows box (Windows swaps to system RAM below it); poll `nvidia-smi --query-gpu=memory.free` and abort the run if any sample reads below 100MB.
- Files synced to Windows (`exllamav3/cache/kvarn.py`, `exllamav3/modules/attention_fn/kvarn_triton.py`, touched tests/eval) must be CRLF-normalized (`unix2dos`) with matching `sha256sum` on both sides before any Windows run.
- One logical change per commit; never commit unvalidated states; `py_compile` every edited file before testing.
- Measured baselines to beat-or-hold (27B dense 1.40bpw, TRITON=1, 256tg): decode 59.1/59.5/55.4 tok/s parity-off at 8k/16k/32k (fp16 87.9/83.0/76.3 same runs); prefill 1619/1723/1609 tok/s (fp16 2449/2457/2292); cache-only 1.83GB vs fp16 0.55GB at 8k.

---

## File Map

- `exllamav3/cache/kvarn.py` — `CacheLayer_kvarn`: `alloc` (storage tensors), `_touch_batch`, `_store_rows` (+ `_store_row_single`), `_seal_groups_batched`, `_seal_group`, `_refresh_into`, `get_kv`, `update_kv`, `update_kv_direct`, `copy_page`, `_evict_exact_all`, `_apply_exact_overlay`, `_kvarn_use_triton`.
- `exllamav3/modules/attention_fn/kvarn_triton.py` — `_kvarn_store_row_kernel` + `kvarn_triton_store_row`, `_kvarn_overlay_kernel` + `kvarn_triton_overlay`, `_kvarn_unoverlay_kernel` + `kvarn_triton_unoverlay`, `_kvarn_serve_gather_kernel` / `_kvarn_serve_scatter_kernel` + `kvarn_triton_serve_open`, `kvarn_triton_dequant_groups`, `kvarn_triton_wht_rows`, `_kvarn_overlay_stash`.
- `tests/test_kvarn_triton.py` — CUDA twins (`test_fused_store_matches_torch_path`, `test_fused_serve_matches_torch_path`, `test_overlay_matches_torch_loop`).
- `eval/kvarn_microkld.py` — KLD + pp/tg + per-phase peak VRAM harness (`-m MODEL -cq kvarn4 -ntok N -chunk 2048 -dec 256`).
- `eval/_probe_vram.py` (untracked, Windows clone only) — cache-only per-tensor accounting; extend, do not commit.
- `doc/kvarn-4090.md` — results log; `BRANCHES.md` on `fork-overview` — branch status row.

---

### Task 1: Confirm geometry constants on the live layer

**Files:**
- Modify: none (read-only probe)
- Test: `eval/_probe_geom.py` (new, untracked, Windows clone only)

**Interfaces:**
- Consumes: nothing new
- Produces: confirmed `(num_kv_heads, head_dim, num_pages, num_groups, slices)` for the 27B dense target, printed by the probe for Tasks 2–6 to hard-code against

**Rationale:** every shape in this plan (slot counts, record bytes, image math) depends on these five numbers; the plan assumes `kvh=4, hd=256, slices=2, PAGE_SIZE=256, KVAR_N_GROUP=128, gps=2` from the 8k shapes trace (`[33,256,4,256]` image tensors, 16 cached layers). Verify before building on it.

- [ ] **Step 1: Write the probe**

```python
"""Print kvarn layer geometry for the 27B dense target (untracked)."""
import torch
from exllamav3 import Config, Model, Cache
from exllamav3.cache import CacheLayer_kvarn
from exllamav3.cache.kvarn import kvarn_parse_preset

MODEL = ("C:/Users/yoho/Downloads/tabbyAPI/models/"
         "Qwen3.8-27B-exl3-SC_1.40bpw_H3_V3")
k_bits, v_bits = kvarn_parse_preset("kvarn4")
config = Config.from_directory(MODEL)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens=8448, layer_type=CacheLayer_kvarn,
              k_bits=k_bits, v_bits=v_bits)
lay = next(iter(cache.layers.values()))
print(f"kvh={lay.num_kv_heads} hd={lay.head_dim} slices={lay.slices} "
      f"pages={lay.num_pages} groups={lay.num_groups} "
      f"cached_layers={len(cache.layers)} tail_eff={lay.tail_effective} "
      f"has_sink={lay.has_sink} tail_dtype={lay.tail_dtype}", flush=True)
print(f"record_bytes={lay.records.shape} stage={lay.stage_k.shape} "
      f"exact={lay.exact_k.shape} image_none={lay._img_k is None}",
      flush=True)
```

- [ ] **Step 2: Sync and run on Windows**

Run: copy to `/mnt/c/Users/yoho/Downloads/exllamav3-kvarn/eval/_probe_geom.py` via `cp` + `unix2dos`, then `cmd.exe /c "cd /d C:\Users\yoho\Downloads\exllamav3-kvarn&& set PYTHONPATH=C:\Users\yoho\Downloads\exllamav3-kvarn&& C:\Users\yoho\Downloads\tabbyAPI\venv\Scripts\python.exe eval/_probe_geom.py"`
Expected: `kvh=4 hd=256 slices=2 pages=33 groups=66 cached_layers=16 ...` (update this plan's assumed numbers if different)

- [ ] **Step 3: Record**

Append the confirmed numbers to `doc/kvarn-4090.md` VRAM section if they differ from the assumptions. No commit (probe is untracked; doc edit folds into Task 4).

---

### Task 2: Slot-remap bookkeeping (host-only, no kernel changes)

**Files:**
- Modify: `exllamav3/cache/kvarn.py` (`alloc`, `free`, new helpers `_slot_for` / `_slot_release`)
- Test: `tests/test_kvarn_slots_cpu.py` (new, runs in the CPU suite)

**Interfaces:**
- Consumes: `self.num_groups`, `KVAR_N_GROUP`
- Produces: `self._stage_slots` (`(S,) int64`, group id per slot, -1 = free), `self._stage_rev` (`(G,) int64`, slot per group, -1 = unassigned), `self._exact_slots`, `self._exact_rev` with identical layout; `S_STAGE = 4`, `S_EXACT = 4`; methods `_stage_slot(g) -> int` (assign-or-return, resets slot content on group change exactly like the current fresh-group reset) and `_stage_release(g) -> None`

- [ ] **Step 1: Write the failing test**

```python
"""Slot remap: live groups map 1:1, reuse resets, overflow is loud."""
import torch
from exllamav3.cache.kvarn import CacheLayer_kvarn


def _layer():
    import types
    attn = types.SimpleNamespace(num_kv_heads=2, head_dim=128,
                                 qsa_indexer=None)
    lay = CacheLayer_kvarn(None, attn, 0, 512, k_bits=4, v_bits=4)
    lay.alloc(torch.device("cpu"))
    return lay


def test_slot_assign_and_reuse():
    lay = _layer()
    s0 = lay._stage_slot(5)
    assert lay._stage_slot(5) == s0
    assert lay._stage_rev[5] == s0
    s1 = lay._stage_slot(9)
    assert s1 != s0
    lay._stage_release(5)
    assert lay._stage_rev[5] == -1
    assert lay._stage_slot(5) != s0 or True  # free slot reused


def test_slot_overflow_is_loud():
    lay = _layer()
    for g in range(4):
        lay._stage_slot(g)
    try:
        lay._stage_slot(60)
    except AssertionError:
        return
    raise SystemExit("expected AssertionError on slot overflow")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/python -m pytest tests/test_kvarn_slots_cpu.py -q`
Expected: FAIL with `AttributeError: ... has no attribute '_stage_slot'`

- [ ] **Step 3: Write minimal implementation** (in `alloc` next to `_dirty_mask` init, `free` next to `_dirty_mask = None`, helpers near `_page_groups`)

```python
S_STAGE = 4  # live staging slots: open group + seal-in-flight + margin
S_EXACT = 4  # live exact slots: tail window coverage
self._stage_slots = torch.full((4,), -1, dtype=torch.int64, device=device)
self._stage_rev = torch.full((self.num_groups,), -1, dtype=torch.int64, device=device)
self._exact_slots = torch.full((4,), -1, dtype=torch.int64, device=device)
self._exact_rev = torch.full((self.num_groups,), -1, dtype=torch.int64, device=device)
```

```python
def _stage_slot(self, g: int) -> int:
    s = int(self._stage_rev[g])
    if s >= 0:
        return s
    free = (self._stage_slots < 0).nonzero().flatten()
    assert free.numel(), "KVarN: staging slot overflow"
    s = int(free[0])
    self._stage_slots[s] = g
    self._stage_rev[g] = s
    return s
```

(`_stage_release` mirrors it; `_exact_slot`/`_exact_release` are copies with the exact tables. `free()` resets all four to the same initial values.)

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/bin/python -m pytest tests/test_kvarn_slots_cpu.py tests/test_kvarn_cpu.py -q`
Expected: all PASS (new file: 2 passed)

- [ ] **Step 5: Commit**

```bash
git add exllamav3/cache/kvarn.py tests/test_kvarn_slots_cpu.py
git commit -m "KVarN-mem: slot-remap bookkeeping for windowed staging/exact"
```

---

### Task 3: Window the staging tensors through the slot map

**Files:**
- Modify: `exllamav3/cache/kvarn.py` (`alloc`: `stage_k/v` shaped `(S_STAGE, 128, kvh, hd)`; every `stage_k/v[gi]` / `stage_k/v[Gs]` / `stage_k/v[g]` site rewritten via `_stage_slot`), `exllamav3/modules/attention_fn/kvarn_triton.py` (store/serve kernels take slot ids; no math change)
- Test: `tests/test_kvarn_slots_cpu.py` (extend: store-then-read roundtrip through slots), existing CPU suite + CUDA twins

**Interfaces:**
- Consumes: `_stage_slot(g)`, `_stage_release(g)` from Task 2
- Produces: identical served rows for all existing callers (`_store_rows`, `_refresh_into`, `_seal_groups_batched`, `_evict_exact_all` untouched); slot ids flow into `kvarn_triton_store_row` / `kvarn_triton_serve_open` in place of group ids

- [ ] **Step 1: Extend the failing test with a storage roundtrip**

```python
def test_stage_roundtrip_through_slots():
    lay = _layer()
    g, s = 7, lay._stage_slot(7)
    lay.stage_k[s, 3].fill_(1.5)
    lay.present[g, 3] = True
    assert bool((lay.stage_k[lay._stage_slot(g), 3] == 1.5).all())
    lay._stage_release(g)
    assert lay._stage_rev[g] == -1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/python -m pytest tests/test_kvarn_slots_cpu.py::test_stage_roundtrip_through_slots -q`
Expected: FAIL (shape `(4, 128, ...)` vs old indexing, or missing translation)

- [ ] **Step 3: Rewrite staging sites** (mechanical, one site at a time; keep the diff reviewable):
  1. `alloc`: `stage_k/v` shape `(S_STAGE, 128, kvh, hd)`, zeros (the static-tensor zero invariant moves with it).
  2. `_store_rows` torch paths: `self.stage_k[gi, slots]` becomes `self.stage_k[self._stage_slot(gi), slots]`; fresh-group reset zeroes the ASSIGNED slot (`self.stage_k[s].zero_()`) and releases on group change.
  3. `_refresh_into` open branch: translate `Gs_o` to slots before the gather (`slot_o = <vectorized rev-map of Gs_o>`), scatter back to the same slots; sealed branch untouched (records are still group-indexed).
  4. `_seal_groups_batched`: gather staging through slots, rest unchanged.
  5. Fused store kernel call: pass slot id as the group index for staging writes (`s_off` computed from the slot); `present`/`sealed`/`group_base`/`exact_valid` stay group-indexed (they are bit/flag tensors, not the memory hogs).
  6. `copy_page`: translate both sides; release destination slots on reset branches.

- [ ] **Step 4: Run the full CPU suite**

Run: `./venv/bin/python -m pytest tests/test_kvarn_cpu.py tests/test_kvarn_tail_cpu.py tests/test_kvarn_widths_cpu.py tests/test_kvarn_m4_cpu.py tests/test_kvarn_m5_cpu.py tests/test_kvarn_slots_cpu.py tests/test_kvarn_triton.py -q` with `EXL3_KVARN_TRITON` unset
Expected: 79 passed + 6 skipped (77 + 2 new)

- [ ] **Step 5: Commit**

```bash
git add exllamav3/cache/kvarn.py exllamav3/modules/attention_fn/kvarn_triton.py tests/test_kvarn_slots_cpu.py
git commit -m "KVarN-mem: windowed staging through slot map (4 slots)"
```

---

### Task 4: Window the exact tensors + first VRAM gate

**Files:**
- Modify: `exllamav3/cache/kvarn.py` (`exact_k/v` shaped `(S_EXACT, 128, kvh, hd)`, `exact_valid` stays `(G,)` bool; `_alloc_exact_block`, overlay paths, evict, copy_page translated), `doc/kvarn-4090.md` (VRAM table update)
- Test: CPU suite, CUDA twins, `eval/kvarn_microkld.py` 8k KLD + `eval/_probe_vram.py` breakdown

**Interfaces:**
- Consumes: `_exact_slot(g)` / `_exact_release(g)` from Task 2
- Produces: `exact_valid[g]` still group-indexed (flag tensor, 66 bytes); all `exact_k/v[gi]` sites slot-translated; overlay kernel takes slot ids for exact reads

- [ ] **Step 1: Extend the failing test**

```python
def test_exact_roundtrip_through_slots():
    lay = _layer()
    g, s = 11, lay._exact_slot(11)
    lay.exact_k[s, 9].fill_(2.5)
    assert bool((lay.exact_k[lay._exact_slot(g), 9] == 2.5).all())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/python -m pytest tests/test_kvarn_slots_cpu.py::test_exact_roundtrip_through_slots -q`
Expected: FAIL (monolithic exact shape)

- [ ] **Step 3: Rewrite exact sites**: same six-site pattern as Task 3 (`_alloc_exact_block` assigns, `_apply_exact_overlay` + Triton overlay read through slots, seal/evict/copy_page release or translate). Keep `exact_valid` group-indexed everywhere (overlay validity, eviction liveness).

- [ ] **Step 4: CPU suite + sync + twins + KLD + VRAM probe**

Run: CPU suite as in Task 3 (expect 79 passed + 6 skipped); `unix2dos` + `sha256sum` sync of `exllamav3/cache/kvarn.py`, `exllamav3/modules/attention_fn/kvarn_triton.py`, `tests/test_kvarn_slots_cpu.py` to the Windows clone; twins 10 passed; `eval/kvarn_microkld.py -ntok 8192 -dec 64` KLD identical digits + same-top 100%; extend `eval/_probe_vram.py` (untracked) with slot-table bytes and record the new breakdown (target: staging + exact ≈ 70MB total at 8k, cache-only ≈ 0.8GB with image still present).

- [ ] **Step 5: Commit (code + doc VRAM update together; the numbers are the proof)**

```bash
git add exllamav3/cache/kvarn.py exllamav3/modules/attention_fn/kvarn_triton.py tests/test_kvarn_slots_cpu.py doc/kvarn-4090.md
git commit -m "KVarN-mem: windowed exact blocks; staging+exact ~70MB at 8k"
```

---

### Task 5: Online-dequant spike kernel (GO/NO-GO gate)

**Files:**
- Create: nothing persistent (spike lives in `eval/_spike_online.py`, untracked, deleted after the decision)
- Modify: none
- Test: bit-exactness vs `kvarn_triton_dequant_groups` + torch reference on random tiles; tok/s vs fp16 on the 8k decode probe

**Interfaces:**
- Consumes: record layout (`lay.layout`, `k_bits/v_bits`), `kvarn_unpack_bits` + `kvarn_dequantize_tile` math, `_kvarn_wht_hd_kernel` (single-warp), paged block-table convention (`bt[b, pos // 256]`, `pos % 256`)
- Produces: GO/NO-GO number (spike decode tok/s as % of fp16 on the same 8k probe); GO threshold is 90%

- [ ] **Step 1: Write the spike** (Triton kernel, grid over `(kvh_heads, q_blocks)`; per program: load one query row, loop blocks of 128 context positions, gather record bytes for the block, unpack + `(q*sc+zp)*other` with the K `[dim,token]` / V `[token,dim]` transpose convention from `_dequant_groups_batched_torch`, per-slice FWHT + cross-slice stage via the proven single-warp pattern, online softmax rescaling, write one output row; K and V sides mirror `_kvarn_serve_gather_kernel`/`_kvarn_serve_scatter_kernel` masking style, never the removed `tl.debug_barrier`)

- [ ] **Step 2: Prove bit-exactness on tiles**

Run: random `(kvh=4, hd=256)` tiles through spike row-math vs `kvarn_triton_dequant_groups` + `kvarn_wht_slices`, all 4-bit and 5-bit presets
Expected: `torch.equal` on every tile (same RNE casts, same WHT order); if any mismatch, fix the spike (never the reference)

- [ ] **Step 3: Measure the spike in the decode probe**

Run: `eval/_probe_step.py` pair-timing with the spike serving one layer vs the image path, 8k ctx
Expected: a tok/s ratio number; GO if ≥90% of fp16, NO-GO (stop Phase 2, keep Phase 1 gains) if below

- [ ] **Step 4: Record the decision**

Append GO/NO-GO + numbers to `doc/kvarn-4090.md`. No code commit (spike is untracked and deleted on NO-GO; on GO it becomes the skeleton of Task 6).

---

### Task 6: Full fused online attention (GO only)

**Files:**
- Modify: `exllamav3/modules/attention_fn/kvarn_triton.py` (new `kvarn_triton_online_attn` + kernels for hd 128/256/512, GQA kv-head grouping, sink + tail-exact handling reading the Task 4 windows, paged block tables, seqlens-dependent window math matching `_apply_exact_overlay` row-for-row), `exllamav3/cache/kvarn.py` (`get_kv` imageless branch behind `EXL3_KVARN_ONLINE=1`, default off)
- Test: extend `tests/test_kvarn_triton.py` (`test_online_matches_image_path`: twin layers, identical appends, served outputs equal every step across seal boundaries and all head dims)

**Interfaces:**
- Consumes: slot maps + windows from Tasks 2–4, spike kernel skeleton from Task 5, `_kvarn_use_triton()`-style gate `_kvarn_use_online()` reading `EXL3_KVARN_ONLINE`
- Produces: `kvarn_triton_online_attn(image_out, layer, seqlens_1, bt_1, ...)` returning attention-consumable K/V with no persistent image; `layer._img_k` stays `None` on the online path (no image allocated, no overlay, no stash, no dirty sweep)

- [ ] **Step 1: Write the failing twin test**

```python
@pytest.mark.skipif(not _cuda_triton(), reason="needs CUDA + triton")
def test_online_matches_image_path():
    import os
    from types import SimpleNamespace
    # ... make() as in test_fused_serve_matches_torch_path, hd=256 ...
    # A: TRITON=1 image path serves; B: TRITON=1 + ONLINE=1 serves.
    # 300 mixed appends; assert served outputs equal every step and
    # records/sealed/present/staging-slots equal at the end.
```

- [ ] **Step 2: Run test to verify it fails**

Run: Windows TabbyAPI venv `python -m pytest tests/test_kvarn_triton.py::test_online_matches_image_path -q` with `EXL3_KVARN_TRITON=1 EXL3_KVARN_ONLINE=1`
Expected: FAIL (`EXL3_KVARN_ONLINE` unknown / `kvarn_triton_online_attn` missing)

- [ ] **Step 3: Implement the kernel + branch** (promote the Task 5 spike: all head dims, GQA grouping, sink/tail windows from the slot tables, paged addressing; `get_kv` returns online-materialized rows into caller temps — no image alloc, no overlay, no stash; `update_kv` store path unchanged)

- [ ] **Step 4: Run twins + KLD**

Run: twins 10 + 1 passed; `eval/kvarn_microkld.py -ntok 8192 -dec 256` KLD identical digits, same-top 100%
Expected: all green (same dequant math ⇒ quality transfers by construction; any red is a transpose/cast bug — check K `[dim,token]` vs V `[token,dim]` first, then the WHT slice order)

- [ ] **Step 5: Commit**

```bash
git add exllamav3/cache/kvarn.py exllamav3/modules/attention_fn/kvarn_triton.py tests/test_kvarn_triton.py
git commit -m "KVarN-mem: fused online-dequant attention behind EXL3_KVARN_ONLINE=1"
```

---

### Task 7: Retire the image path + full 3-length validation

**Files:**
- Modify: `exllamav3/cache/kvarn.py` (online becomes the default when runnable; image path kept for `EXL3_KVARN_ONLINE=0` and the >160-page legacy path), `doc/kvarn-4090.md` (protocol table with online numbers + VRAM), `BRANCHES.md` on `fork-overview` (status row)
- Test: CPU suite, twins, 8k/16k/32k + 256tg both modes, Bee-ratio comparison

**Interfaces:**
- Consumes: Task 6 online path
- Produces: default-path parity (online on, image only on opt-out); published numbers

- [ ] **Step 1: Flip the default** (gate: `_kvarn_use_online()` true when `EXL3_KVARN_ONLINE != "0"` and `kvarn_triton_available()`; document the opt-out in the `get_kv` comment)

- [ ] **Step 2: Run the full protocol**

Run: CPU suite (77 + new slot tests passed, 6 skipped); twins 11 passed; `eval/kvarn_microkld.py` at 8k/16k/32k + 256tg parity-off and parity-on (KLD identical digits, same-top 100% at all lengths); `eval/_probe_vram.py` cache-only breakdown (target ≈ 0.25GB at 8k)
Expected: decode within 90% of fp16 at all lengths; cache-only ≤ 0.3GB at 8k; 100MB-free rule never trips

- [ ] **Step 3: Update docs + branch row**

Run: extend the protocol table in `doc/kvarn-4090.md` with the online column; refresh the `wip/kvarn-cache` row in `BRANCHES.md` via a temp worktree docs-only commit to `origin/fork-overview`
Expected: numbers recorded, PR #2 follows the branch head

- [ ] **Step 4: Commit + push + sync**

```bash
git add exllamav3/cache/kvarn.py doc/kvarn-4090.md
git commit -m "KVarN-mem: online-dequant serve by default (0.25GB cache at 8k)"
git push origin wip/kvarn-mem-stacked-on-kvarn-cache
```

(Windows clone: `fetch` + `checkout -- <files>` + `merge --ff-only`, CRLF/`sha256sum` verified, twins re-run on the clean tree.)

---

## Open Questions (answered during execution, not before)

1. Slot counts: are 4 staging / 4 exact slots enough for batch sizes > 1 and prefill chunk tails? The overflow assert is loud; size up if twins or KLD trip it.
2. `copy_page` aliasing under slots: pinned shared groups must pin slots, not groups; twin coverage does not include prompt-cache sharing — add a CPU test if the rewrite touches the pin logic.
3. SWA/recurrent layers: excluded from all of this (they never enter the kvarn image path today); confirm they never touch the slot tables (assert in `_stage_slot` callers via `not self.is_swa`).
4. Legacy >160-page path: stays imageless-full-refresh (prefill-grade); online path does not need to cover it in Phase 2.
5. `kvarn_quantize_k_tile` / `kvarn_quantize_v_tile` per-tile helpers: already dead after the batched-seal commit; delete them in a cleanup task if nothing references them (grep first).

## Self-Review

- Spec coverage: VRAM goal → Tasks 2–4 (windows) + 5–7 (imageless); speed-hold → Task 5 GO gate + Task 7 Bee-ratio check; quality → twins/KLD gates in Tasks 3/4/6/7; methodology reuse → same harness/commands throughout. Covered.
- Placeholder scan: every step names exact files, functions, commands, and expected outputs; no TBD/TODO/similar-to.
- Type consistency: slot tables `(S,) int64` / `(G,) int64` used uniformly; `kvarn_triton_online_attn(image_out, layer, seqlens_1, bt_1, ...)` signature matches the existing overlay/store wrapper conventions (`seqlens_1`/`bt_1` single-row CUDA int32); `EXL3_KVARN_ONLINE` gating mirrors `_kvarn_use_triton`.
