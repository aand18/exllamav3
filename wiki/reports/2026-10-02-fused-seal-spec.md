# Subagent report: fused seal Triton kernel spec (2026-10-02)

## Provenance
RECOVERED from chat output (pre-convention dispatch: no report file
was written; this is the complete returned text, verbatim). Session
`ses_f05ae0402ffeL7mhzEomF8PH1U`. Intermediate reasoning not
preserved. Verdict: DEFER (prize is ALU-only, ULP-fragile) -- spec
kept for later. See `doc/perf-strats.md` #2.

## Original report (verbatim)

## Fused K4V4 seal kernel spec (analysis only)

**1. Exact op-order, one launch, `grid=(T,)` with `T=G*C*2`**

One program per tile (K and V as separate pids; K transposed on
store). Single-program-per-tile with row paging (16-row chunks; keep
only `log_c/log_r[128]` + scalars on chip) -- 128x128 fp32=64KBx2
buffers exceeds SRAM, so never materialize full `cur`; stream
passes. `BLOCK_C=128, BLOCK_R=16`. Per-program sequence mirrors
torch exactly:

- Init `kvarn.py:770-774`: `log_c=log_r=0`, `s_best=1`,
  `imb_best=imbalance(tile)`.
- `rebuild` `kvarn.py:777`: `cur=tile/(exp(log_c)[None,:]*
  exp(log_r)[:,None])` -- exp both, multiply first, single divide,
  in that order.
- Per iter `kvarn.py:791-796`: col `std(correction=1)`
  (`kvarn.py:748`); `clamp(1e-3,1e3)`;
  `log_c=(log_c+log(std_c)).clamp(-0.3,10.0)`; `rebuild`; row
  `std(correction=1)` (`kvarn.py:752`); same clamp/log/clamp;
  `rebuild`.
- Imbalance `kvarn.py:755-759`: `amax/amin(clamp_min(1e-8))` col
  ratio + row ratio.
- Best `kvarn.py:798-802`: `better=imb<=imb_best`;
  `where`-select `imb_best,s_col_best=exp(log_c),
  s_row_best=exp(log_r)` (exp after compare).
- Window `kvarn.py:803-807`: check every 4th iter or last;
  stall-break.
- Final `kvarn.py:809`: `balanced=tile/(s_col[None,:]*s_row[:,None])`.
- RTN `kvarn.py:823-832` (K4: `qmax=15`): `lo=amin(-1)`,
  `hi=amax(-1)`, `scale=(hi-lo)/15 .clamp_min(1e-10)`;
  `q=floor((x-lo)/scale+0.5).clamp(0,15).u8` (half-away-zero, valid
  as arg>=0); out `s_row*scale, s_row*lo, s_col`.
- Pack `kvarn.py:656-666`: LSB-first `stream=(q>>b)&1, b=0..3`; no
  pad (16384*4%8==0); `byte=sum(stream*2**arange(8))`. K payload
  stores `q.T` (`kvarn.py:857`), V stores `q` direct (`kvarn.py:873`).
  Meta `kvarn.py:2336-2347`: six `half()` rows (sc/zp/other per
  side).

Input tiles per `kvarn.py:2317-2320`: K `permute(0,2,3,4,1)` ->
rows=dim,cols=token; V `permute(0,2,3,1,4)` -> rows=token,cols=dim.

**2. Early-break: recommend fixed 4 iters**

Torch break `kvarn.py:803-807` is batch-global (`.any()`): one slow
tile keeps all tiles iterating. Per-tile independent exit is *not*
bit-identical (a stalled tile torch would have carried could improve
at iter 9). True ballot needs grid-sync (Triton lacks it) -> split-
kernel-per-window or spin-flag, reintroducing launches/syncs to
save only ALU (launch count already 1 after fusion). Typical tiles
plateau by iter ~4 (`kvarn.py:785-786`); each iter ~=6 tile passes,
so fixed-4 costs 25% of worst-case 16. Recommend **fixed 4**
(fallback 8 if KLD-8k/64k regresses): deterministic, no divergence,
twin-testable.

**3. Layout I/O + coalescing**

In: `(G,128,kvh,hd)` fp32 blocks `bk/bv` (`kvarn.py:2297-2298`,
`1894-1897`); each program streams 16x128 fp16/fp32 row-chunks --
fully coalesced 256B rows. Out per tile: 8192B payload
(`kvarn.py:570-573`: 16384*4/8) written sequentially (coalesced),
+3x128 f16 meta rows=768B/side (`kvarn.py:636-637` shapes).
Traffic/group/head-slice ~= 128KB read (K+V fp32, or 64KB if fp16
in) + ~9KB write vs today: same bytes but ~100 DRAM passes from
per-op temps (`rebuild` allocs each half-iter) + ~300
launches/layer/chunk. Fused: same logical passes, no temps,
1 launch.

**4. Numerics risks + twin gates**

- `log/exp` lib (torch vs Triton/libdevice): 1-ulp -> gate
  `torch.equal` on `s_col/s_row_best` per tile; `allclose`
  insufficient.
- `tile/(a*b)` vs `(tile/a)/b` fusion: gate `torch.equal` on
  `balanced`.
- `std(correction=1)` reduction order (two-pass vs Welford): gate
  `torch.equal` on iter-1 `std_c/std_r`.
- `clamp` fusion order (`clamp->log->clamp`): gate `torch.equal` on
  `log_c` after iter 1.
- `floor(x+0.5)` vs `round` on .5 + fp32 div order: gate
  `torch.equal` on `q` bytes.
- fp32->fp16 RNE + NaN/Inf propagation in `where`: gate
  `torch.equal` on meta f16 bits; `allclose` only allowed
  end-to-end dequant error (KLD), never on payload/meta.
- Pack reduction `sum(stream*w)` vs OR: gate `torch.equal` on
  payload bytes.

**5. Verdict: DEFER fused Sinkhorn; does not conflict with fast path**

Complementary, not subsuming: `_store_rows_fast`
(`kvarn.py:1818-1921`) removes staging/slot/sync overhead and feeds
the shared core (`kvarn.py:1919->2302`); the fused kernel would
replace core internals (`kvarn.py:2321-2334`), accelerating all
three callers (fast/batched/spilled). But seal runs once per 128
tokens/layer while decode/store/WHT (already Tritonized) dominate;
torch.compile `maxdiff 16.0` on this exact early-break graph proves
hand-replication is ULP-fragile across ~6 risk sites for at most
ALU savings (launches already batched). Defer until profiles
attribute >10% prefill time to seal math; then build fixed-4
K4V4-only kernel behind `PARITY=1` twin gates above.
