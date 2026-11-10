# T64 — Senior Campaign Review (Argon)

**Reviewed:** `feat/adaptive-tier` (`3b2a43b3`), reports `T00`–`T63`, `v165-STATE.md`, and the MoE handoff/profiler paths (`block_sparse_mlp.py`, `block_sparse_mlp_cpu.py`, `moe_cpu_host.py`, `round_prof.py`).

---

## 1. Is the analysis sound?

**No.** The closing conclusion—*"the binding constraint is the CPU handoff wait, and no remaining lever reaches Strata"*—commits a physical category error: it mistakes **irreducible CPU DDR5 DRAM read time (`~0.60 ms/layer`)** for **synchronization/handoff latency**, and relies on three broken intermediate measurements that were never reconciled.

### A. `cmb` (`0.696 ms`) is DDR5 bandwidth time, not "handoff wait"
In a 5-row drafted verify round (`T59`, `rows=5`), per-layer median stream times inside `mlp.forward` (`0.883 ms`) are:
- `sub` (`cpu_split_submit`): `0.014 ms` in `T56` (inflated to `0.047 ms` in `T58`/`T59` by an instrumentation bug; see §1.C).
- `ker` (`self.bc.run_bszN`, fused GPU experts + shared expert): `0.117 ms`.
- `cmb` (`cpu_split_combine`, stream wait on `slot["done"]` + add): `0.696 ms`.
- `rest` (`router + latents + post-norms`): `~0.056–0.064 ms`, plus `mixer` (`attention/GDN`) outside MoE at `0.099–0.143 ms`.

Because `sub` signals `data_ready` before `ker` launches (`block_sparse_mlp.py:1167, 1460`), the CPU worker runs **concurrently** with `ker`. From `T52` (`EXL3_MOE_HANDOFF_PROF` + `EXL3_MOE_CPU_PROF`) and `T58`:
- At `rows=5`, a layer routes `5 × 10 = 50` picks and misses the GPU on a median of **8 (mean 9.3) distinct CPU experts** (`T58`).
- At `~1.95 MB/expert`, the CPU reads **`~18.1 MB/layer`** from DDR5 at **`42.9–45.1 GB/s`** (~92 % of the box's ~48 GB/s bus), taking **`0.58–0.69 ms` of pure GEMV compute** plus **`0.21 ms` of PCIe/doorbell spin** = **`~0.81 ms/layer`** total worker elapsed time.
- Subtracting the concurrent GPU kernel (`0.813 ms - 0.117 ms`) leaves **`0.696 ms`** exposed in `cmb`—matching `T59` to the microsecond.

### B. Why `T57`–`T60` were doomed by elementary pipeline physics
Across a 5-row drafted layer, total GPU compute is only **`T_GPU ≈ 0.34 ms/layer`** (`mixer 0.143 + sub 0.014 + ker 0.117 + rest 0.064` = `~16.3 ms/round`), while serial CPU work is **`T_CPU ≈ 0.81 ms/layer`** (`~38.9 ms/round`, with the DDR5 bus 92 % busy).
- **You cannot hide `0.81 ms` of 92 %-saturated DDR5 reads behind `0.34 ms` of GPU compute.**
- Even perfect full-round overlap across layer boundaries could hide at most `0.34 ms/layer`, leaving `~0.47 ms/layer` (`~22.5 ms/round`) exposed.
- Therefore, per-expert completion flags (`T57`, −44 %), ring slots 4→8 (`T58`, null), grouped GPU kernels (`T59`, attacking a `0.117 ms` non-bottleneck), and routing prefetch (`T60`, prefetching on a DDR5 bus with only 8 % idle time) could never reach Strata. **The only way to shrink `cmb` is to cut CPU DRAM bytes per layer.**

### C. Four measurement and basis errors in the chain
1. **The `T51` → `T52` → `T56` whipsaw:** `T52` subtracted a **1-row** undrafted `SPLIT_PROF` wait (`0.25 ms`, measured in `perf.py`) from a **5-row** drafted round budget (`1.08 ms/layer`, from `spec_decode.py`)—violating Measurement Rule #1—to claim the round was only 23 % CPU-late and 77 % (`0.82 ms/layer`) GPU-bound. When `T56`/`T59` measured the actual 5-row drafted round and proved the reverse (`0.117 ms` GPU kernel vs `0.696 ms` CPU wait), the campaign **never revoked `T51`'s dismissal of expert coverage**.
2. **`T51` tested coverage in the wrong runner and wrong shape:** `t51-A` ran in `perf.py` (`bsz=1` undrafted decode, 128-token bands), where pooled coverage moved only **`31.76 % → 33.94 %`** (hitting `57.5 %` only at the end of the pass), while paying inline checkpoint-swap overhead (`~386 µs/swap`) and unmeasured `TIER_STATS` overhead, with **13 of 48 layers unpinned** (unable to adapt under the 24 GiB pin cap). Worse, `-Runner spec_decode` **hardcodes `EXL3_MOE_CPU_SWAP=0`** (`v165-STATE.md:133`), so adaptive coverage was **never tested on the 5-row drafted path** where 50 picks/layer create the DDR5 bottleneck.
3. **`T46` (`t46-C-stream`) never executed decode streaming:** It tested `EXL3_MOE_STREAM_MIN_ROWS=2` and `STREAM_T=2` using `perf.py`, where `TG` runs at **`bsz = 1`**. In `block_sparse_mlp_cpu.py:1118`, `if bsz < self.cpu_host.stream_min_rows:` (`1 < 2`) is **always true** during `perf.py` decode (and an expert can only be hit once at `bsz = 1 < STREAM_T=2`). The streaming path was dead code during `TG`; the reported `−4.4 %` was single-draw `perf.py` noise, yet it was used to close `M3-miss-split`.
4. **Instrumentation self-interference (`T58`–`T60`):**
   - `T58` inserted `buf[:bsz].copy_(selected_experts, non_blocking=True)` **inside** the `sub` CUDA event bracket (`block_sparse_mlp_cpu.py:1108`), inflating `sub` median 3.5× (`0.014 ms → 0.049 ms`).
   - `T60` measured "previous-round routing autocorrelation" with `SAMPLE=8` and filtered to `rows==5` (`15,888` of `124,991` layer-rounds, or 12.7 %, because `spec_decode.py` logs the `rows=1` baseline arm into the same CSV and `-dds` shortens draft windows). Consecutive `rows==5` samples were **24 to ~150 tokens apart**, not adjacent rounds.

---

## 2. Did we misunderstand what Strata does?

**Yes—both what drives Strata's speed and what numbers are being compared.**

### A. Strata cuts CPU DDR5 bytes per layer by ~4–5× (`M1 + M2 + M3 + M7`)
Strata does not win by handoff micro-optimizations or grouped-GEMM weight reuse at 31 % coverage. It shrinks the `0.70 ms/layer` CPU miss window to `<0.15 ms/layer` (cutting a 48-layer round from `~54 ms` to **`~25.4 ms`**, `T51:77`) by stacking three byte-reduction mechanisms (`T31`, `T50`):
1. **82.77 % live GPU expert hit rate (`M1 + M2 + M7`):** Strata runs single-slot (`batch 1`), pages cold KV to host RAM (`--kv-resident 32768`, keeping 0.23 GB of KV in VRAM vs the fork's 2.82 GB at 262k), and uses the built-in MTP head + zero-VRAM prompt-lookup instead of a separate drafter model. It spends that freed VRAM on **8,301 expert slots (`173/layer`, +32 % over the fork's 6,288)**, managed as **one global non-uniform pool across all 48 layers** (forcing per-layer uniformity inside Strata itself costs `−15.5 %` throughput, `T50 §4`), updated asynchronously in the background (`96 swaps / 4 rounds`, decay `0.7`, RAM→VRAM, no carved-out spares). Raising coverage from `31.8 %` to `82.8 %` drops distinct misses from `~9.3` to `~2.5` experts/layer.
2. **Hybrid PCIe + CPU miss splitting (`M3: pcie_frac = 0.55`):** For the remaining `~2.5` missed experts/layer, Strata streams ~55 % of the expert weights over PCIe 4.0 x16 (`~24 GB/s`, which sits idle during `cmb` in the fork) to the GPU while the CPU computes the other ~45 % over DDR5 (`~44 GB/s`).
3. **Zero-weight-cost prompt lookup + MTP (`M4`):** On agentic/coding prompts with 100k+ tokens of repo context, n-gram prompt lookup drafts multi-token spans with zero model-forward cost (`2.5–3.5 tokens/window`).

### B. The `55–58` vs `93.6 tok/s` headline compares incompatible workloads
- **The fork's `55–58 tok/s` corpus figure** is an unweighted mean across 8 categories, 4 of which are Creative and Translation (`42–46 tok/s`, acceptance `1.0–1.4`) where `v165-STATE.md:78` notes DFlash2 loses. On **Agentic and Coding categories (`T63`)**, the fork already runs at **70.1–83.1 tok/s** (`spec_primary.py` agentic-weighted score **71.5–72.5 tok/s**).
- **The live `56.9 tok/s` figure (`T14`)** is a 2-day multi-client (`max_batch_size: 2`) median serving the degraded 4-bit exl3 drafter (`39.6 %` live acceptance = `1.58/4`) under static placement, whereas **Strata's `93.6 tok/s`** is from one 88-request single-stream (`batch 1`) log across a warm conversation sweep (`82.8 %` coverage).

### C. What to check (read-only, no build)
1. **Stratify Strata's `server-20261029-172352.err.log` (88 requests):** Check cold-request (requests 1–5) vs warm-sweep coverage and tok/s, and check the split between prompt-lookup n-gram drafts and MTP drafts.
2. **Check per-layer miss skew in `roundprofT58*.distinct` and `50ac90eb5104cb75.json`:** Tabulate mean distinct CPU misses by layer index `0..47`. How much of the `9.3` distinct misses/layer comes from high-entropy layers starving under the uniform `131/layer` cap?
3. **Verify `t46-C-stream`'s log:** Confirm that `STREAM_MIN_ROWS=2` never fired during `perf.py` `TG` (`bsz=1`).

---

## 3. What we did wrong procedurally, and what should have come first

1. **Profiled inside the MoE at Task 56 instead of Task 1:** Weeks went into sweeps and ports before three CUDA events (`sub`, `ker`, `cmb`) were placed inside `mlp.forward`.
2. **Used `perf.py` (`bsz=1`, undrafted, swap on) as a proxy for `spec_decode.py` (`bsz=5`, drafted, `CPU_SWAP=0`):** Mechanisms whose physics depend on a 5-row verify window (`T46` streaming gates, `T51` adaptive coverage, `T52` `SPLIT_PROF`) were tested in `perf.py` or mixed across runners (`T52`).
3. **Never propagated falsifications backward:** When `T56` proved `T52`'s GPU-bound diagnosis was backwards, `T51`'s dismissal of coverage was left standing.
4. **Let production drift from `state.json` and judged agentic performance on an 8-domain unweighted mean (`T61`–`T63`):** The 4-bit exl3 drafter ran in production from Nov 2 to Nov 10 before being tested on `spec_primary.py`, where it `WITHDRAW`s (`Agentic-curl −5.74 %`, acceptance `2.67 → 2.19`).

**What I would have done on Day 1:** Run the 5-bracket phase profiler (`mixer`, `sub`, `ker`, `cmb`, `drafter`) + `EXL3_MOE_CPU_PROF` + `distinct` counter on `spec_decode.py` (scored by `spec_primary.py`). Seeing `T_GPU = 0.34 ms/layer` vs `T_CPU_DRAM = 0.70 ms/layer` (`9.3` distinct CPU experts × `1.95 MB` at `44 GB/s`) on Day 1 proves that **only levers that increase accepted tokens/round or decrease CPU DDR5 bytes/layer can move decode speed**.

---

## 4. Single recommended experiment for one final week of GPU time

**Experiment: Two-gated test of Strata's byte-reduction path on `spec_decode` (262k Q4, split 381, BF16 drafter restored).**

- **Gate 1 (2 hours, CPU-only offline trace simulation — zero GPU time):** Replay the recorded router selections (`selected_experts` from `spec_decode` on the 3 agentic categories) through a 50-line Python simulator of Strata's cache policy (`6,288` global non-uniform slots across 48 layers, `96 swaps / 4 rounds`, `decay 0.7`, no carved spares) and `M3` (`pcie_frac = 0.50` streaming of multi-row misses).
  - **Gate 1 falsifier:** If the global non-uniform pool + `M3` split does not cut CPU-handled distinct experts at `rows=5` from **`9.3` to `≤ 4.5` per layer** (`≥ 0.30 ms/layer` = `14.4 ms/round` less DDR5 read time), **stop**—`6,288` slots are physically insufficient without host-paged KV (`M1/M7: kv_resident 32768`) or a 2-bit cold CPU tail.
- **Gate 2 (the GPU experiment, if Gate 1 passes):**
  1. Raise `EXL3_MOE_PINNED_ARENA_GIB=34` so **all 48 layers** (not 35) reside in pinned host RAM (fits easily in 63.6 GB RAM).
  2. Enable the non-uniform global slot table (`first[l]`) + async background-thread RAM↔VRAM swaps (`96 / 4 rounds`, decay `0.7`, zero carved spares) inside `-Runner spec_decode` (removing the forced `EXL3_MOE_CPU_SWAP=0`).
  3. Run a 4-pass **ABBA** on `spec_decode` at 262k (`bench/spec_primary.py` + `EXL3_ROUND_PROF=1`).
- **Pre-registered pass / falsification criteria:**
  - **Pass:** `rows=5` distinct CPU experts/layer drops from `9.3` to `≤ 5.0`, `cmb` median drops from `0.696 ms` to `≤ 0.42 ms`, and `spec_primary.py` agentic-weighted tok/s rises **`≥ +15 %`** (`~71.5 → ≥ 82 tok/s`).
  - **Falsified if:** (a) distinct CPU experts drop `≤ 5.0` but `cmb` stays `> 0.55 ms` (proving `cmb` is dominated by worker/PCIe wakeup latency rather than DDR5 bytes), or (b) background PCIe/RAM swap traffic steals DDR5 bandwidth from the CPU worker (`EXL3_MOE_CPU_PROF` GEMV bandwidth drops `< 38 GB/s`), neutralizing the fewer misses.
