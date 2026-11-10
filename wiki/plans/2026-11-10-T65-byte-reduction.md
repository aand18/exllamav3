# T65 — plan: test the byte-reduction lever (Argon's recommendation), and what we already have

Answers two questions and turns the review into two gated experiments. No build, no kernel, no
engine change: every mechanism named here already exists in the tree.

## 1. Do we have cold KV paged to host? Yes.

- `exllamav3/generator/cpu_cache.py` — `CPUPageCache`, "second-tier page cache in pinned system
  memory": hashed K/V pages evicted from the GPU tier, restored on the stream instead of paying a
  prefill pass. Class docstring at `cpu_cache.py:14-39`.
- `exllamav3/generator/disk_cache.py` — `DiskPageCache`, an optional third tier behind it.
- Wired in `generator/generator.py:530-540`: `cpu_cache_size` → `self.cpu_page_cache`, and
  `disk_cache_dir` requires it.
- CLI: `-ccs / --cpu_cache_size` (GB), `-dcd / --disk_cache_dir` (`model_init.py:123`).

So Strata's `--kv-resident 32768` shape — a **small GPU cache plus a host tier holding the
overflow** — is expressible today as `-cs 32768 -ccs <GB>`. What has *not* been measured is that
combination at a 262 k working set; the earlier T46/T47 arm ran a 32 k cache **without** a host tier
and measured **+5.5 % decode / +3.2 % prefill** from the VRAM it freed (`T46-47-kv-experts-and-streaming.md`),
and the note that "the host KV tier does not extend context" is about *extending*, not about
*paging a fixed 262 k working set*.

The VRAM arithmetic: KV at 262144 Q4 is 2911 MiB (`size_check`); at 32768 it is ~360 MiB, so
**~2551 MiB frees = ~29 experts = 410/layer uniform (20 094 slots)**, against today's 18 288
(381 × 48). That is the M1+M7 lever, and it needs no new code.

## 2. What Argon got right, wrong, and unverified

**Accepted (verified in the tree):**
- `-Runner spec_decode` forces `EXL3_MOE_CPU_SWAP=0` (`run_pass.ps1:72`), so **T51's coverage
  closure was measured in `perf.py` at bsz=1 and never on the drafted path**. The dismissal of
  expert coverage therefore rests on a runner with 1 pick/layer instead of 50.
- The T58 instrument inflated `sub` (median 0.014 → 0.049 ms) because the selection copy sat inside
  the `sub` event bracket. Ours, and it propagated into T59's table.
- T60's cover used sampled rounds 8 apart (lag ≈ 24 tokens), not adjacent rounds.
- The 55–58 vs 93.6 headline mixes bases: agentic categories already run 70–83 t/s (T63).

**Wrong (arithmetic does not survive):** the review prices an expert at **1.95 MB**; the arena says
**5.57 MB** (33375 MiB ÷ 48 layers ÷ 131 CPU experts). Its byte budget is ~2.9× light, and its
`pcie_frac`/`96 swaps` numbers come from Strata's config, not from this box.

**Unverified, and it decides everything:** that `cmb` is *DDR5 bytes* rather than wait. The
distinguishing test is cheap and is Gate 1's second half — if CPU bytes were the whole story,
`cmb` would scale with distinct misses; if it is wake-up latency, it would not.

## 3. Gate 1 — zero-build simulation, then one instrumented pass

**1a (done, negative, and it taught us something).** `bench/pool_sim.py` replays the engine's saved
routing profile (`engine-state/master/exllamav3/expert_stats`, per-layer decayed frequencies)
through uniform and global pools. Its calibration step — reproduce T58's median 8 / mean 9.35 at
381 resident — **failed at 0.49 misses**, by 19×. The stationary profile says the coldest 131
experts take 4–6 % of picks; the measurement says they take ~20 %. **Conclusion: the long-run
decayed profile is far more concentrated than per-conversation routing, so no pool can be sized
from it.** Any capital allocation for this lever needs real per-round selections.

**1b (the actual gate).** Dump `selected_experts` per layer per sampled round during one
`spec_decode` pass on the agentic workloads, then simulate pools against *that*. Falsifier, per
Argon: if a global non-uniform pool of 20 094–25 000 slots cannot cut distinct CPU misses from
9.3 to ≤4.5/layer, stop — the lever is dead and the campaign closes on the CPU-bytes explanation.

**1c (the deciding measurement, same pass).** Correlate `cmb` against per-layer distinct misses
across all sampled layer-rounds. Bytes ⇒ strong positive slope; wake-up latency ⇒ flat.

## 4. Gate 2 — only if 1b and 1c both pass

`-Runner spec_decode` at 262144 with `-cs 32768 -ccs 8` (host tier holding the working set), all 48
layers pinned (`EXL3_MOE_PINNED_ARENA_GIB=34`), placement allowed on the drafted path
(`-StaticSwap` removed), 4-pass ABBA scored by `spec_primary.py`.

Pre-registered pass: distinct CPU misses ≤5.0/layer, `cmb` median ≤0.42 ms, agentic primary
**≥ +15 %** (71.5 → ≥82 tok/s), acceptance unchanged within 0.05.
Falsified if `cmb` stays >0.55 ms at ≤5 misses (latency-dominated), or if page restores cost more
than the freed experts buy (measure restore bytes and their PCIe time in the same pass).

## 5. Order and stopping rule

1. 1b + 1c in one instrumented pass (~17 min). Stop if 1b's falsifier fires.
2. Gate 2 only on a green 1b **and** 1c. One campaign, ABBA, `spec_primary.py` as the verdict.
3. If Gate 2 fails, the campaign closes with the measured statement that the remaining gap is
   bytes-per-token that this engine cannot remove at 262 k with 512 experts and Q4 KV.
