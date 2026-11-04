# Decode to Strata — implementation plan (2026-11-04)

> **For agentic workers:** REQUIRED SUB-SKILL: use `subagent-driven-development`. Steps use `- [ ]`.
> Every task is engine or harness code plus its tests; the **measurement gates** are collected in one
> session at the end, because the box takes one pass at a time and the card is production.

**Goal:** recover the decode gap to Strata on this box by fixing the placement policy, adding a PLE row
cache, cost-modelling the draft window, enabling a placement-aware drafter measurement, and moving
resident KV into GPU experts.

**Architecture:** five independent deltas, each default-off or config-only, each with its own test and
its own measurement arm. No delta changes the served default until its arm beats the current
production number on the agentic-weighted metric (T34's rule).

**Tech Stack:** Python 3.12, torch 2.11 + CUDA (Windows venv), exllamav3 engine, pytest on the Windows
copy (`/mnt/c/Users/yoho/Downloads/exl3-adaptive-t27/src`), `run_pass.ps1` arms.

**Spec:** the Argon review of 2026-11-04 (`openchamber ses_e58a52f10ffevEFMplvuOoLxc4`) plus T31's
matrix. Where this plan departs from that message it says so inline.

## Global Constraints

- Do **not** modify `/home/dev/UnbiasedStrata`, the `165-win-perf` worktree, or the deployed harness
  `_agent/v165`. Repo copies (`eval/`, `bench/windows/`) may change; deployment is a separate step.
- Every new behaviour is **default-off**: an unset env var must reproduce today's bytes.
- `python3 bench/strata_transfer_check.py --check` must stay at `6/6 rows PASS` and `--selftest` green;
  the M2 row reads `_PROMOTE_BUDGET`, `_PROMOTE_RATIO`, `_PROMOTE_DECAY` and `_split_spares` from
  `block_sparse_mlp_cpu.py`, so those symbols keep their names and defaults.
- Tests run on the Windows copy; the Linux box has no pytest. `tests/moe_cpu` must end
  `>= 121 passed, 63 skipped`.
- Commit messages state the defect or the measurement. Stage files by name.
- A measurement arm may not be claimed unless `verify_task.py --post` passes for the report that quotes it.

## Review Focus

1. **An expert at count ≥ 50 that is routed** — must be visible to the promoter; today it is not.
2. **A prefill followed by decode** — the promoter must not carry prompt routing into the first decode step.
3. **A row-cache miss on a row that was evicted** — must return identical bytes to the uncached path.
4. **A draft window chosen on a low-acceptance long context** — must shrink, not keep paying 5-row CPU work.
5. **Two arms in one process with dynamic placement** — placement must be reset between them or the ratio is meaningless.

---

### Task 1: The promoter can see a hot expert, and runs at Strata's cadence

**Files:**
- Modify: `exllamav3/modules/block_sparse_mlp_cpu.py` (`_split_promote_step` ~:203-343, `_split_swap_tick` ~:1200)
- Test: `tests/moe_cpu/test_promote_visibility.py` (new)

**Interfaces:**
- Produces: `_routed_mask(cur: np.ndarray, prev: np.ndarray, decay: float) -> np.ndarray` (bool, same shape),
  `_promote_reset(ps)` (zero the histograms and the previous snapshot), env knobs
  `EXL3_MOE_CPU_PROMOTE_EVERY` (int, default 1) and `EXL3_MOE_CPU_PROMOTE_DECAY` (float, default 0.98).

- [ ] **Step 1: Failing test — visibility at high counts**

```python
def test_a_hot_routed_expert_is_visible():
    cur = np.array([50.0 + 1.0 - 0.98 * 50.0])   # one hit at count 50
    prev = np.array([50.0])
    assert _routed_mask(cur, prev, 0.98)[0]
```
Run on the Windows copy: `pytest tests/moe_cpu/test_promote_visibility.py -q` → FAIL (ImportError).

- [ ] **Step 2: Failing test — prefill does not leak**

```python
def test_prefill_zeroes_the_promote_histograms():
    # a fake module with _split_hist of 500.0 and a PromoteState whose prev is 400.0
    _split_promote_step(ip, [m], rows = 64)          # prefill-size
    assert float(m._split_hist.abs().sum()) == 0.0 and ps.prev is None
```

- [ ] **Step 3: Implement**

`_routed_mask` compares `cur - prev * decay > 0.5`. In `_split_promote_step`, replace
`routed = cur > ps.prev` with it; when `rows >= modules[0].cpu_host.stream_min_rows`, call the reset
(zero `_split_hist` for every module, clear `snap_ev`, `prev`) and return before issuing anything.
Gate the whole block on `step % EVERY == 0` (default 1 keeps today's behaviour) and apply `decay ** EVERY`
that invocation. Read the decay from `EXL3_MOE_CPU_PROMOTE_DECAY` so `_PROMOTE_DECAY` keeps its value
for the transfer checker.

- [ ] **Step 4: Pass**

`pytest tests/moe_cpu/test_promote_visibility.py -q` → PASS; `pytest tests/moe_cpu -q` → `>= 121 passed, 63 skipped`;
`python3 bench/strata_transfer_check.py --check` → `6/6 rows PASS`.

- [ ] **Step 5: Commit** — `fix: the promoter was blind to every expert at count 50 or more`

---

### Task 2: A row cache in front of the PLE gather

**Files:**
- Modify: `exllamav3/modules/row_table.py` (`gather` path ~:282-312)
- Test: `tests/modules/test_row_cache.py` (new)

**Interfaces:**
- Produces: `EXL3_NGRAM_ROW_CACHE_ROWS` (int, default `0` = off, max `1 << 20`),
  `EXL3_NGRAM_ROW_CACHE_STATS` (0/1), and counters `row_cache_hits()` / `row_cache_misses()`.

- [ ] **Step 1: Failing test — a cached row is byte-identical**

```python
def test_a_cached_row_matches_the_uncached_gather(tmp_path):
    # build a small row table file, gather with the cache off, then on (same row twice)
    assert torch.equal(uncached, cached_second_call)
```

- [ ] **Step 2: Failing test — eviction is bounded**

```python
def test_cache_never_exceeds_its_row_budget():
    # rows = 4, gather 16 distinct rows, assert len(cache) == 4 and misses == 16
```

- [ ] **Step 3: Implement** a clock/LRU dict keyed by `(table, row)` with a row budget; on miss, call the
  existing `ext.ngram_gather_cpu` path unchanged. Default 0 must take today's code path with no dict.

- [ ] **Step 4: Pass** — `pytest tests/modules/test_row_cache.py -q` → PASS; `tests/modules -q -k "ngram or row"`
  unchanged; `tests/moe_cpu` unchanged.

- [ ] **Step 5: Commit** — `perf: a decode-size PLE row cache, off by default`

---

### Task 3: Cost-model the draft window

**Files:**
- Modify: `exllamav3/generator/generator.py` (`dynamic_draft` path ~:1128-1146)
- Test: `tests/generator/test_draft_window.py` (new)

**Interfaces:**
- Produces: `choose_draft_window(acceptance: float, context_len: int, distinct_experts: float, base: int) -> int`,
  env `EXL3_DRAFT_COST_MODEL` (0/1, default 0).

- [ ] **Step 1: Failing test — low acceptance shrinks the window**

```python
def test_low_acceptance_on_long_context_shrinks_the_window():
    assert choose_draft_window(0.25, 200_000, 22.0, 4) < 4
```

- [ ] **Step 2: Failing test — the model is a no-op when off**

```python
def test_cost_model_off_returns_the_configured_window():
    assert choose_draft_window(0.25, 200_000, 22.0, 4) == 4   # with EXL3_DRAFT_COST_MODEL unset
```
(Implement the default as a pass-through rather than reading env inside the pure function: the caller
gate reads the env.)

- [ ] **Step 3: Implement** the ratio `expected_tokens(k) / expected_ms(k, distinct_experts(k))` with the
  constants from T14 (acceptance by context band) and pick the largest window whose score is within 2 %
  of the best; default-off returns `base`.

- [ ] **Step 4: Pass** — `pytest tests/generator/test_draft_window.py -q` → PASS; `tests/generator -q` unchanged.

- [ ] **Step 5: Commit** — `perf: pick the draft window by cost, not by confidence alone`

---

### Task 4: Reset placement between arms so a drafter can be measured with it

**Files:**
- Modify: `eval/spec_decode.py` (`swap_refusal` ~:300-333, `main` ~:315-340)
- Test: `tests/testlib/test_reset_placement.py` (new; synthetic model)

**Interfaces:**
- Produces: `reset_placement(model)` in `eval/spec_decode.py`, called between arms, restoring every
  module's `_split_map` from `_split_perm` and clearing `_split_pending` / `_split_free_*`.

- [ ] **Step 1: Failing test** — a synthetic module whose map was mutated is restored to `_split_perm`.
- [ ] **Step 2: Implement** and call it between arms; keep `-swap_ok` for exploration only.
- [ ] **Step 3: Pass** — `pytest tests/testlib/test_reset_placement.py -q` → PASS.
- [ ] **Step 4: Commit** — `eval: reset expert placement between spec_decode arms`

---

### Task 5: Move resident KV into GPU experts (config only)

**Files:**
- Modify (measurement copy only, not the deployed config): `bench/windows/state.json`, `bench/windows/arms/state-kv32k.json` (new)
- Report: `wiki/plans/v165-reports/T37-kv-to-experts.md` (new)

**Interfaces:** none — this is a regime file plus a report.

- [ ] **Step 1: Pre-gate** — `python3 bench/verify_task.py --pre` passes with the new arm file.
- [ ] **Step 2: Measure** — `run_pass.ps1 -Tag t37-kv32k -State <arm> -MaxLength 131072 -Extra "-dr 3"`:
  `cache_size 32768`, `sysmem_kv_cache` on, `cpu_moe_split_experts` reduced until `size_check.py` says
  the fit holds (the plan's 354 is a prediction, not a value to copy).
- [ ] **Step 3: Report** the arm against the same-session production reference; `verify_task --post`.
- [ ] **Step 4: Commit** — the arm file and the report.

---

## Measurement gates (one session, after the card is free)

- **Task 1** — `spec_decode -swap_ok` with Task 4's reset: production vs promoter-fixed, agentic-weighted;
  bar `>= max(1.02, 1 + spread)`, no primary category below 1.00.
- **Task 2/3** — `spec_decode` with the knob on/off, same bar.
- **Task 5** — `perf.py` ladder vs the production reference.

A task whose arm fails the bar is reverted from any deployment proposal; its code may stay, default-off.
