# T11 — cache staging pages, recurrent checkpoints, Q2 sanity

Date: 2026-10-30. Engine win-perf `1.6.5`. Regime: 3.18bpw, split 376, `-cs 262144` where the task
asks for long context, `-cq 2,2` (operator-fixed; no mode ladder), `-NoSpecDec`, frozen profile per
run. Cache mode is a decision, not a candidate: Q2 is operator-fixed, so Step 1–2 measure the knobs
that make that mode affordable rather than remeasuring the mode itself.

## Step 1 — `EXL3_QC_STAGING_PAGES`: inert at this chunk size (mechanism, not a null)

| arm (run id) | PP 65536 | PP 131072 | PP 262144 | TG 261888 | peak_alloc | peak_reserved | min free VRAM |
|---|---|---|---|---|---|---|---|
| `t11-sp64-…` | **1886.42** | **1867.22** | **1850.08** | 39.08 | 20618 | 20794 | 2883 MB |
| `t11-sp256-…` (default) | 1843.58 | 1830.45 | 1828.03 | 39.35 | 20618 | 20794 | 2883 MB |
| `t11-sp512-…` | 1847.04 | 1845.80 | 1834.54 | 38.75 | 20618 | 20794 | 2883 MB |
| `t11-sp1024-…` | 1839.30 | 1832.45 | 1822.84 | 39.09 | 20618 | 20794 | 2883 MB |

The four arms' allocator watermarks are **byte-identical** (20618 allocated, 20794 reserved, 2883 MB
free), and the pp ordering is a ~1 % spread across a 16× change in the knob. The engine says why:

```python
npps_w = block_table.shape[1]                      # pages the window needs, per call
w = min(npps_w, max(1, max(_qc_staging_pages, staging_pages or 0)))   # the knob is a CAP
kd = torch.empty((bsz * w, page_size, n_kvh, head_dim), ...)          # per-call transient
```

At `-chunk_size 4096` a prefill call spans 4096 tokens = **16 pages of 256**, so `npps_w` is 16 and
every tested cap (64…1024) is above it: `w = npps_w` in all four arms, the same scratch, the same
kernels. The knob only binds when the per-call window exceeds the cap — a chunk size of 64k or more.

**The repeat confirms it.** Two arms re-run with the knob at the two ends:

| arm | PP 131072 | PP 262144 | mean (two passes) |
|---|---|---|---|
| `t11-sp256` and `t11-sp256-r2-…` | 1830.45 / 1856.29 | 1828.03 / 1848.12 | 1843.4 / 1838.1 |
| `t11-sp64` and `t11-sp64-r2-…` | 1867.22 / 1837.92 | 1850.08 / 1819.35 | 1852.6 / 1834.7 |

The 16× knob moves the mean by +0.5 % at 131072 and −0.2 % at 262144, while the same arm varies
0.7 % peak-to-peak between its own two passes. The ordering in the first pass does not survive.

**Verdict: the knob is inert at the campaign's chunk size, and the ~1 % pp differences are noise.**
It is not a lever to ship at `-chunk_size 4096`, and the three-page "PP 64 is fastest" reading that
the raw numbers invite would be a fabrication. If chunk size is ever raised, the cap becomes
relevant and this table is the calibration for it.

*(Consequence worth carrying: a bigger `-chunk_size` is the knob that would make both the staging cap
and the per-call overhead matter. It is not in T11's scope and the regime fixes chunk 4096; recorded
as an unexplored lever rather than a result.)*

## Step 2 — recurrent state in fp32 (`-rs32`)

| arm (run id) | PP 262144 | TG 261888 | peak_alloc | peak_reserved | min free VRAM |
|---|---|---|---|---|---|
| `t12-rs32-…` | pending | | | | |
| `t13-262k-perf-20261030-080210` (fp16 default) | 1839.54 | 39.32 | 20618 | 20794 | 2855 MB |

## Step 3 — Q2 retrieval sanity

Not run in this session: it needs `eval/longctx.py`, which the harness does not ship (it is not a
`perf.py`/`spec_decode.py` runner), so it would have to be deployed and driven outside the watcher's
row contract — a small harness addition rather than a measurement this session could make honestly.
Carried forward with that note: the question is real (Q2 is the most aggressive mode), the tool
exists, the gap is plumbing.
