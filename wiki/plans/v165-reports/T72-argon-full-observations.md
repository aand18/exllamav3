# T72 — Argon's four observations and the methodology critique: one reopens the byte lever

Argon's second review was truncated by my 400-word ceiling. This is the **complete** follow-up, with
each claim checked against the tree. One of them overturns a constraint this campaign has been
treating as an engine invariant since the 2 bpw work was postponed.

## 1. `T66`'s dynamic arm was handicapped — the null stands, but on `cmb` only

Verified: with `EXL3_MOE_CPU_SWAP=1` and `-mcpa`, `EXL3_MOE_CPU_PROMOTE` defaults to **4**
(`block_sparse_mlp_cpu.py:1388-1393`), so `gpu_first = first - 4` leaves four GPU slots per layer
empty at load, and `_split_promote_step` (`:361`) returns `promote_ready == False` for experts in
**unpinned** arena chunks — which the WDDM budget (§T67) makes the last 18–24 layers. Those layers
therefore ran at **127 GPU experts instead of 131**.

So T66 tested a degraded configuration. Its conclusion is **not** withdrawn, because it rests on
`cmb` being flat to ±3 µs (0.4 %) and `distinct` unchanged — not on the tok/s gate. But the arm was
not the clean test it claimed, and any future dynamic-placement arm must set
`EXL3_MOE_CPU_PROMOTE=0` explicitly.

## 2. **"One K per layer" is false across the GPU/CPU split boundary** — the 2 bpw tail is legal

Verified in the code:

- `dims_of()` returns the K of *that tensor* (`trellis_rate(s[2])`), not a layer constant
  (`block_sparse_mlp_cpu.py:1425-1427`).
- `register_layer(...)` passes `proj_dims = dict(g=gd, u=ud, d=dd)` taken from
  **`self.gates[first]` — the first CPU-resident expert** (`:1428-1429`, `:1464`), and the worker
  turns those into its own byte offsets (`moe_cpu_host.py:1240-1247`).
- After registration the module shrinks to the GPU slice (`self.gates = self.gates[:first]`, `:1477`)
  and the tail Linears leave the module tree entirely — the two halves **never share tensors,
  pointers or kernels**.

So the requirement is **uniform K within the GPU slice and uniform K within the CPU slice**, not one
K per layer. The earlier "`multilinear.py:26` requires one K per layer" finding applies to the GPU
`MultiLinear` path and blocks *per-expert* K, which is what the tail work was cancelled on. It does
**not** block a **region** split: hot 131 at 3.18 bpw on the GPU, cold 381 at 2 bpw on the CPU.

What that buys, from this campaign's own law:

| quantity | today | 2 bpw CPU tail |
|---|---:|---:|
| bytes per CPU expert | 1.83 MB | ~1.15 MB |
| µs per distinct expert | 33.9 | **~21** |
| `cmb` per layer (20.5 misses) | 0.69 ms | **~0.44 ms** |
| round (48 layers) | 47.2 ms | **~35 ms** |

**≈ +25–30 % decode**, the largest single lever this campaign has found, needing no kernel change —
an offline splice plus the existing `bench/tail_*.py` tooling (`tail_variant_plan.py`,
`tail_variant_write.py`, `tail_requantize.py`, all built for this and shelved when the per-expert
route was cancelled).

Quality exposure, stated up front: the CPU slice carries **22.4 %** of stationary routing mass
(mean over 49 profiles, ranks 131+). So this quantises ~22 % of the model's routed compute more
coarsely, not 74 %. That is measurable (perplexity / logits on a held-out set) and is an operator
decision before it is an engineering one.

## 3. Why `M3` streaming would fail as coded — recorded, not pursued

Three mechanisms, all verified in structure: `cpu_split_submit` calls `submit_prefill` synchronously
and returns `(routed_sum, None)`, running the whole CPU+PCIe pass **before** `ker` instead of
concurrently (`block_sparse_mlp_cpu.py:1137`); `counts1.tolist()` forces a **host sync per layer**
(`moe_cpu_host.py:2119`); and PCIe DMA pulls from the same DDR5 controller the worker already
saturates at 50–60 GB/s. So streaming cannot add an independent pipe — it competes with the worker.

## 4. The `aux` tensors explain the T68 floor breaches

`fetch_aux` (`:1433-1446`) loads `suh`/`svh` (and biases) onto **VRAM** for **all 381 CPU-resident
experts on every layer** — 54 864 small tensors — to serve streamed-prefill reconstruction. That,
plus caching-allocator rounding, is the ~950 MiB `size_check` never sees. It explains the
`split 365`/`355` breaches at `cs 40960` and is a fixable waste if that path is ever needed.

## 5. The methodology critique — accepted, with a caveat on our verdicts

Argon's case against `bench/spec_primary.py`'s gate:

- **`arm_spread` uses the sample range** (`max − min`, `:117-128`), which *grows* with N, so adding
  replicates raises the pass floor (1.039 → 1.045 → 1.050 in T66–T70) instead of shrinking it by
  1/√N.
- **`rate()` at `:121` is an unweighted category mean** while the primary at `:108` is 5:4:3
  weighted — two different metrics under one name.
- **Per-category veto** (`:140`, `:143`) over 3–5-prompt categories with 3.3–6.5 % natural jitter
  makes a true +2.5 % uniform gain fail ~40–50 % of the time by trajectory noise alone.
- **`abs(acc_t − acc_c) ≤ 0.05` (`:144`)** bars any genuine acceptance improvement from passing —
  which is why T61/T63/T70 could only ever be judged on their losses.

Where that leaves our verdicts: **unaffected in direction, weakened in precision.** T66/T69/T70 were
dismissed on `cmb` flat to ±3 µs and `distinct` flat — a 0.4 %-resolution hardware timer, not the
tok/s gate. Argon's own recommended fix matches what we already trust: gate systems changes on
per-round latency at fixed rows, not free-running generation.

The `-dds` closing ABBA, and any future drafter arm, **cannot be judged by `spec_primary.py` as
written**. That is a real defect to fix before another drafting experiment runs.
