# Strata → exllamav3 transfer review: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a machine-checked verdict on every Strata mechanism this fork claims to have transferred for decode performance — faithful, diverged, absent, or incompatible — with the arm that measured it or the statement that nothing did.

**Architecture:** One analysis script (`bench/strata_transfer_check.py`) extracts the placement constants and cadences from both engines' sources and compares them against a recorded matrix, so the review's numbers fail loudly when either engine's defaults drift. The matrix and its per-row verdicts live in one report (`wiki/plans/v165-reports/T31-strata-transfer.md`), each row carrying the source line in both engines and the run id that measured it, or an explicit "unmeasured".

**Tech Stack:** Python 3.12 stdlib for the checker (no torch, no GPU), the Windows harness's `perf` runner for the one arm this plan adds, `bench/verify_task.py` for the gates.

**Spec:** the two independent research notes and the sibling engine's source —
- `/home/dev/research/strata-vs-exllamav3-2026-11-02.md` (the seven mechanisms M1-M5, M7 and the 4/3/2 portability verdict)
- `/tmp/opencode/strata-vs-exllama-research.md` (the coverage attribution and its gate)
- `/home/dev/UnbiasedStrata` at `8234d0e`-era `main` (Strata's own source: `src/program/generate.cpp`, `include/strata/core/expert_cache.hpp`)
- this branch: `exllamav3/modules/block_sparse_mlp_cpu.py`, `exllamav3/model/moe_cpu_host.py`, `bench/windows/state.json`
- measurements already in hand: `wiki/plans/v165-reports/T16-adaptive-tier.md` §7 (four `t27-*` passes) and `/tmp/opencode/t29-pin-unset.md` (four `t29-*` passes)

## Global Constraints

- Do not modify `/home/dev/.local/share/opencode/worktree/3a6c9fa6df7b22161dba7be0a7f274f329396456/165-win-perf`, the deployed harness `/mnt/c/Users/yoho/Downloads/tabbyAPI/_agent/v165/`, `/mnt/c/Users/yoho/Downloads/tabbyAPI/_agent/adaptive-t27/backup/`, or `/home/dev/UnbiasedStrata` (read-only: it belongs to the sibling investigation).
- The checker runs on the dev box: no GPU, no torch, no processes started.
- Any arm is `-Runner perf -MaxLength 131072` from the deployed harness with **env deltas only**; no regime flag may appear in `-Env` (`-mcs`, `-cs`, `-cq`, `-chunk_size`, `-ambs`, draft flags) — the harness aborts, naming the flag.
- A pass is judged by its `results/passes.csv` row (`status=ok`, `breach=0`) and its parsed `PASS/FLOORS/PP/TG_S1` block, never by a process exit code (the venv `python.exe` is a uv launcher).
- The box serves live traffic: a GPU task stops the served model, and restores the pristine engine files (`ec103eb9522d162e86e872de8ea42d289452060f0f73ffd7179c2acdff7bc952`, `f7342c9d57f767af68425b60ae98b911528e9e60481de5caeb71793541d31d71`), restarts it with `C:\Users\yoho\Downloads\tabbyAPI\start_tuned.ps1`, and proves `/v1/models` answers 200.
- Numbers already measured are quoted with their run ids in the same sentence; a number with no run id is "unmeasured", never "approximately".
- Commit messages state the defect or the measurement, not the activity; one logical change per commit.
- The checker must pass on a CPU-only box (`pytest tests/tools -q` collects nothing here — the checker's tests live in `bench/` and run under `python3 -m pytest bench` only if pytest exists; otherwise `python3 bench/strata_transfer_check.py --selftest`).

## Review Focus

Each line is a failure mode this review exists to catch, and the task that pins it is named.

1. **A constant copied with the wrong unit or scope** — Strata's `--adapt-swaps 96` is per *adaptation round over the whole tier*; the fork's nearest knob was 64 swaps *total across all 48 layers per sweep*. Same name, 1.5x different unit, and it made the campaign's first churn estimate 75x too small. Pinned in Task 3.
2. **A mechanism whose name matches but whose semantics differ** — "swap" in Strata is a RAM↔VRAM exchange with no file reads; in the fork's sweep it is two checkpoint reads per swap. A reader who matches the names concludes the port exists. Pinned in Task 2.
3. **A performance claim carried over without a same-session arm** — the sibling engine's 79-95 % tier coverage was the port's justification, and the fork's own arm reads 31.7 % (control) / 72.3 % (pin). Pinned in Task 4.
4. **The instrument changing the thing it measures** — the coverage line costs 1.03-2.32 ms/step and was on in every arm of both campaigns; no arm priced it. Pinned in Task 4.
5. **A divergence that is justified but undocumented** — the fork has no file tier and no per-window adaptation, so the transfer is *allowed* to diverge; a reader who finds the divergence with no recorded reason will "fix" it back. Pinned in Task 3.

---

### Task 1: The Strata side of the transfer, extracted

**Files:**
- Create: `bench/strata_transfer_check.py`
- Create: `bench/test_strata_transfer_check.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `extract_strata(root: pathlib.Path) -> dict` returning
  `{"adapt_every": int, "adapt_swaps": int, "adapt_decay": float, "slot_pool": "global" | "per-layer",
    "swap_source": "ram-copy" | "checkpoint" | "unknown", "kv_resident_tokens": int | None,
    "prefill_chunk": int | None, "source_lines": dict[str, str]}`; and a `--selftest` CLI that runs the
  module's own assertions without pytest.

- [ ] **Step 1: Write the failing test** — `bench/test_strata_transfer_check.py`

```python
def test_extract_strata_reads_the_defaults_from_source_text():
    tree = {  # the shape of the real file, not the file
        "src/program/generate.cpp": (
            "    int adapt_every = 4;\n"
            "    float adapt_decay = 0.7f;\n"
            "    int adapt_swaps = 96;\n"
            '    "exchange them with the GPU cache (no file reads)"\n'),
    }
    got = extract_strata(_fake_root(tree))
    assert got["adapt_every"] == 4 and got["adapt_swaps"] == 96
    assert got["adapt_decay"] == 0.7
    assert got["swap_source"] == "ram-copy"
    assert got["slot_pool"] == "global"
```

- [ ] **Step 2: Run it, expect failure** — `python3 -m pytest bench/test_strata_transfer_check.py -q` → `ModuleNotFoundError`/`AttributeError`.

- [ ] **Step 3: Implement `extract_strata`** in `bench/strata_transfer_check.py`. Parse by regex over the named files only (`src/program/generate.cpp`, `include/strata/core/expert_cache.hpp`), never by importing Strata. `slot_pool` is `"global"` unless `--expert-cache-per-layer` is the default in that file; `swap_source` is `"ram-copy"` when the exchange comment is present, `"checkpoint"` when the only swap path names a checkpoint read, `"unknown"` otherwise. Every value carries the `file:line` it came from in `source_lines`.

- [ ] **Step 4: Run it, expect pass** — same command → `1 passed`.

- [ ] **Step 5: Commit** — `git add bench/strata_transfer_check.py bench/test_strata_transfer_check.py && git commit -m "bench: extract Strata's placement constants from source, not from prose"`

---

### Task 2: The exllamav3 side, extracted

**Files:**
- Modify: `bench/strata_transfer_check.py`
- Modify: `bench/test_strata_transfer_check.py`

**Interfaces:**
- Consumes: `extract_strata`'s dict shape from Task 1.
- Produces: `extract_exllama(root: pathlib.Path) -> dict` returning
  `{"promote_budget": int, "promote_ratio": float, "promote_decay": float, "spares": int,
    "sweep_interval": int, "sweep_max": int, "sweep_source": "checkpoint" | "ram-copy",
    "stream_min_rows": int, "cache_mode": str, "cache_size": int, "source_lines": dict[str, str]}`
  with the same `source_lines` contract.

- [ ] **Step 1: Write the failing test** — assert on the real tree:

```python
def test_extract_exllama_matches_the_branch():
    got = extract_exllama(REPO)
    assert (got["promote_budget"], got["promote_ratio"]) == (6, 1.5)
    assert got["sweep_interval"] == 128 and got["sweep_max"] == 64
    assert got["sweep_source"] == "checkpoint"      # _split_swap_experts re-reads via stc.get_tensor
    assert got["stream_min_rows"] == 32
```

- [ ] **Step 2: Run it, expect failure.**

- [ ] **Step 3: Implement `extract_exllama`** — regex/defaults over `exllamav3/modules/block_sparse_mlp_cpu.py` (the four `_PROMOTE_*`/`EXL3_MOE_CPU_*` defaults, `_split_swap_experts`' source), `exllamav3/model/moe_cpu_host.py` (`EXL3_MOE_STREAM_MIN_ROWS`), and `bench/windows/state.json` (`cache_quant`, `cache_size`). `sweep_source` is `"checkpoint"` iff `_split_swap_experts` calls `stc.get_tensor`, `"ram-copy"` iff it copies from `arena_views` only.

- [ ] **Step 4: Run it, expect pass.**

- [ ] **Step 5: Commit** — `git commit -am "bench: extract this fork's placement constants the same way"`

---

### Task 3: The matrix and its verdicts, machine-checked

**Files:**
- Modify: `bench/strata_transfer_check.py`
- Create: `wiki/plans/v165-reports/T31-strata-transfer.md`

**Interfaces:**
- Consumes: both extractors.
- Produces: `MATRIX: list[dict]` in the script — one row per mechanism with `id`, `strata`, `fork`,
  `verdict` in `{"faithful", "diverged", "absent", "incompatible"}`, `justification` (required for
  `diverged`/`absent`/`incompatible`), and `measured` (run id or `"unmeasured"`) — plus
  `check(strata: dict, fork: dict) -> list[str]` returning one line per row: `PASS/FAIL <id>: <what was compared>`.

- [ ] **Step 1: Write the failing test** — a fixture matrix with one row whose recorded constant is wrong must FAIL, and the real tree must PASS:

```python
def test_a_wrong_recorded_constant_fails():
    bad = [dict(MATRIX[0], strata=dict(MATRIX[0]["strata"], adapt_swaps=64))]
    fails = [l for l in check(extract_strata(STRATA), extract_exllama(REPO), matrix=bad) if l.startswith("FAIL")]
    assert fails and "adapt_swaps" in fails[0]
```

- [ ] **Step 2: Run it, expect failure.**

- [ ] **Step 3: Write the matrix** in the script with these rows, each carrying the exact values read in Tasks 1-2 and the record of what the earlier campaigns concluded:
  `M2-churn` (Strata `adapt_every 4` / `adapt_swaps 96` per round, decay 0.7, RAM↔VRAM → fork spares + `PROMOTE_BUDGET 6`/step, `RATIO 1.5`, `DECAY 0.98`, sweep 64/128 steps with two checkpoint reads): verdict `diverged`, justification = the churn-rate arithmetic (Strata ≈ 15 swaps/round vs the fork's ≈ 0.2/step at the t27 operating point) and `measured = "t29-C1-pin-unset-20261103"`; `M1-budget` (Strata 173 experts/layer with 32k KV resident → fork 131/layer at 262144 KV): `absent`, `justification` names the three levers and the arithmetic from the research note (≈38 GB needed for 80 % on 24 GiB), `measured = "unmeasured"`; `M3-miss-split` (Strata DMA-splits ≈55 % of misses → fork `stream_min_rows 32` sends every 4-row window to the CPU): `absent`, `justification` = the env delta that would test it (`EXL3_MOE_STREAM_MIN_ROWS=2`), `measured = "unmeasured"`; `M4-spec-yield` (2.5-3.5 vs 2.2-2.4 tokens/window): `diverged`, `measured = "unmeasured"` (the arms carry no draft flags); `M5-ple` and `M7-deploy`: `diverged`/`absent` with the same discipline.
  `check()` compares the extractors' values with the matrix's and prints one PASS/FAIL line per row.

- [ ] **Step 4: Run it, expect pass** — `python3 bench/strata_transfer_check.py --check` prints `8/8 rows PASS` (or the exact failures).

- [ ] **Step 5: Write `wiki/plans/v165-reports/T31-strata-transfer.md`** — the same matrix as a table with columns `mechanism | Strata (value, file:line) | fork (value, file:line) | verdict | justification | measured`, one section per row for the divergences' arithmetic, and a `## How to re-run` section quoting the exact commands. Commit both together.

---

### Task 4: Which rows have performance evidence, and what the instrument cost

**Files:**
- Modify: `wiki/plans/v165-reports/T31-strata-transfer.md`

**Interfaces:**
- Consumes: Task 3's matrix and the run ids in `T16-adaptive-tier.md` §7 and `/tmp/opencode/t29-pin-unset.md`.
- Produces: the `measured` column filled for every row, with either a run id plus the number it produced or the literal `unmeasured`.

- [ ] **Step 1: Write the evidence table** with these facts, each with its run id (recompute from `results/*.csv` and `logs/perf-t29-*.log`, do not copy from prose): `t29-A1-pin-off-20261103` 31.760 % pooled / 30.989 % mean coverage, `t29-C1-pin-unset-20261103` 72.347 / 70.200, `t29-C2-pin-unset-20261103` 72.331 / 70.186, `t29-A2-pin-off-20261103` 31.735 / 30.975; C's 8,669 promotions by step 3,584, hook 2.318 ms/step; the pin's 30 of 34 chunks (30,720 MiB, driver limit) and 42 of 48 layers promotable; PP 131072 +17.99 %, PP 4096 +15.71 %, TG 130816 +1.13 % against a 4.6 % repeat spread.
- [ ] **Step 2: Price the instrument, or mark it unmeasured** — the campaigns enabled `EXL3_MOE_TIER_STATS=1` in every arm. If no `EXL3_MOE_TIER_STATS=0` arm exists in `passes.csv`, write `instrument cost: unmeasured (all arms carried it)` and add it to the report's next-arm list; do not estimate it.
- [ ] **Step 3: Gate the report** — `cd <165-win-perf worktree> && python3 bench/verify_task.py --post --report <abs path>/T31-strata-transfer.md` must pass; paste its verdict line into the report.
- [ ] **Step 4: Commit.**

---

### Task 5: The one arm this review needs, and the portability verdicts it does not

**Files:**
- Modify: `wiki/plans/v165-reports/T31-strata-transfer.md`

**Interfaces:**
- Consumes: Task 4's `unmeasured` rows.
- Produces: for each `absent` row, a one-line "the port is <env delta | <N> lines in <file> | impossible because <reason>>" and, if the row is worth an arm, the exact command.

- [ ] **Step 1: Add the transfer-cost table** — for `M1-budget` (needs `-CacheSize`/`-MaxLength` at 32768 plus the `-Split` that follows, `sysmem_kv_cache` in the served config), `M3-miss-split` (`EXL3_MOE_STREAM_MIN_ROWS=2`, an env delta), `M4-spec-yield` (draft flags belong to `-Runner spec_decode`), `M5-ple` (the n-gram row cache exists in this fork; name the knob or say the port would be code).
- [ ] **Step 2: Run the one arm the review cannot answer from existing data**, if the operator's consent is in hand: `-Tag t31-M3-rows2 -Runner perf -MaxLength 131072 -Env "EXL3_MOE_TIER_STATS=1,EXL3_MOE_STREAM_MIN_ROWS=2"`, stop the served model first and restore the box afterwards exactly as the Global Constraints state. Record tag, run id, `status`, `breach`, TG 130816 and the coverage line. If consent is not in hand, mark the row `unmeasured — needs an arm` and say so in the handoff.
- [ ] **Step 3: Commit and hand off** — the report's closing section lists what a reader must re-run, in order, and which rows are already settled.

---

## Self-review

- **Spec coverage:** M1, M2, M3, M4, M5, M7 each get a matrix row (Task 3), and M2's two halves (policy and rate) get separate justifications. The two research notes' gate ("measure the hit rate, then raise the budget") is answered by Task 4's evidence table.
- **Step scan:** every step names a command and its expected output; no step says "handle edge cases" or "add validation". The one place a body would be needed — `check()`'s comparison — is specified by its signature and its two tests (a wrong constant must FAIL, the real tree must PASS), so the implementer writes it.
- **Type consistency:** `extract_strata`/`extract_exllama` return the same `source_lines` contract; `check` takes both plus an optional `matrix`; `--check` and `--selftest` are the only CLI verbs named anywhere.
- **Review Focus:** all five lines have a task — units/scope (Task 3), name-vs-semantics (Task 2), carried claims (Task 4), instrument cost (Task 4), undocumented divergence (Task 3's `justification` requirement).
- **Proportion:** the plan specifies two function signatures, one matrix schema and the exact values; the code is left to the implementer except the regex-source contract. Task 6 is the only task that edits production code, and it exists because leaving a known-defective phrase and four self-contradicting report lines is worse than the two commits it costs.

---

### Task 6: The two findings the branch's final review left open (carry-over)

**Files:**
- Modify: `exllamav3/model/moe_cpu_host.py` (`arena_pin_report`, ~line 1836)
- Modify: `tests/moe_cpu/` (the pin-report tests)
- Modify: `wiki/plans/v165-reports/T16-adaptive-tier.md`

**Interfaces:**
- Consumes: `pinned_layers()` (Task 1 of the superseded t29 plan, already merged in `cd39e08b`).
- Produces: `arena_pin_report()` whose output names the promotable layer count in all three cases, and a T16 with no claim its own later sections refute.

- [ ] **Step 1: Write the failing test** — a host with zero promotable layers must print a phrase, and a host with every layer promotable must print a different one:

```python
def test_pin_report_states_zero_and_all_layers_explicitly():
    assert "0 of 2 layer(s) promotable" in _arena_host(blocks=2, pinned_chunks=()).[1]
    assert "all 2 layer(s) promotable" in _arena_host(blocks=2, pinned_chunks=(0, 1)).[1]
```

- [ ] **Step 2: Run it, expect failure** (both extremes currently print no phrase).

- [ ] **Step 3: Implement the three-way branch** in `arena_pin_report`: real total (`sum(1 for b in self.layer_blocks if b)`), `0 of {total}`, `{n} of {total} ({first}..{last})`, `all {total}`.

- [ ] **Step 4: Run `pytest tests/moe_cpu -q`, expect pass** (rsync the tree to a Windows path first, `--delete`, as in T16 §9).

- [ ] **Step 5: Fix T16's four stale claims** — the "nothing here has been run / unrun" lines, the "hot end of the CPU tier" claim (contradicted by its own §7/§10), the `generator.py` hook reference, and the line claiming C5 was fixed with `Event(device = d)`.

- [ ] **Step 6: Gate and commit** — `bench/verify_task.py --post --report <abs>/T16-adaptive-tier.md --cites-failure t27-A-base-20261103-121148` prints `verdict: all gates passed`; commit the code+test and the report as two commits.
