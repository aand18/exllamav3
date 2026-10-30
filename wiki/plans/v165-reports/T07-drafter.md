# T07 — drafter comparison (resumed 2026-10-30)

Engine: win-perf `1.6.5` wheel (`1422a0a1…`, installed 2026-10-30 after T8's null verdict; stock
`1.6.5` for the A/B itself). Model: production `3.18bpw_h6_ng8` at split 376, `-cs 40960`
(`draft.spec_decode_cache_size`), `EXL3_MOE_CPU_SWAP=0` (automatic for drafting runs), frozen
expert profile restored per arm. Runner `run_pass.ps1 -Runner spec_decode -Extra "<delta> -nbl"`.
Harness `fba75185…`, `spec_decode.py` `333c4f24…`.

## Step 0 — the session's decode level is not yesterday's

Before any drafter arm: `t8-stock-a1` measured TG 38.38 @130816 where the archive has 49.79 for the
same regime. The drift is characterised in `v165-STATE.md` (cache size, frozen profile, GPU clocks,
CPU clocks, host load, RAM speed all ruled out; PP unchanged). Consequence here: every arm in this
report is compared **only against another arm measured today**, and the reference was re-measured
for this task (`t7-ref`), never taken from the T6 numbers of 2026-10-29.

## Step 1 — reference (stock MTP head, `-ndt 2`, win-perf wheel)

`t7-ref-20261030-062125`: **46.48 tok/s**, accepted **1.35/2.00** per round, 8 categories,
`breach=0`. Consistent with today's stock-wheel drafting references (`sdhoff` 45.86,
`sdhoff-r2` 45.49) — the wheel is a null for drafting too, as T8 found for pp/tg.

## Step 2 — the `-or` sidecar path is broken in the installed wheel (found, fixed, not rebuilt)

`t7-sc26-20261030-063149` (`-Extra "-or mtp-sidecar-2.6bpw.yaml -nbl"`) died in **6 s** at load:

```
File ".../exllamav3/modules/block_sparse_mlp_cpu.py", line 805, in bias_keys
    has = [(l.key + ".bias") in stc.tensor_file_map for l in ls[first:]]
AttributeError: 'VariantSafetensorsCollection' object has no attribute 'tensor_file_map'
```

`bias_keys()` asks the base class's `tensor_file_map` directly; under `-or` the collection is a
`VariantSafetensorsCollection`, which exposes the same information through the polymorphic
`has_tensor()` (it delegates via `find_stc`). Both bias sites (`:688`, `:805`) now call
`stc.has_tensor(...)` — commit `330ce0cd`, **load-path only, not in the installed wheel**, so this
run used the fallback path below.

The bug is worth carrying: `-or` plus a CPU split could never have loaded a model with expert
biases, which is every Qwen4Exp checkpoint on this box. It is the first real defect the new gates
did not catch, because no gate loads a model under `-or`.

## Step 2b — sidecar 2.6 bpw via the dir-copy path (the production shape)

`D:\llms\Qwen3.8-Flash-Next-exl3-3.18bpw_h6_ng8-mtpopt` is a **hardlink copy** of the base
directory (`cp -al`: 0 extra bytes, the 54.5 GB of shards and the 51.8 GB ngram table are shared
inodes) plus `mtp-2.6bpw.safetensors`. The copy has its own expert-profile key, so the base
profile was copied to it (`50ac90eb…` → `50a2b423…`) in both the default cache and the frozen
master; the arm reports **49/49 `placement from saved profile`**, i.e. as warm as the reference.

The shadow is proven twice: the load log prints `!! Overriding mtp.fc_hidden.suh … with
mtp-2.6bpw.safetensors` for the head's tensors, and the generated text diverges from the reference
at character 2233 ("find **any** Python files" → "find **the** Python files").

| arm (run id) | window | tok/s | accepted/round | note |
|---|---|---|---|---|
| `t7-ref-20261030-062125` | 2 | **46.48** | 1.35/2.00 | stock head, reference |
| `t7-sc26-dir-20261030-063352` | 2 | 42.90 | 1.35/2.00 | sidecar 2.6 bpw |
| `t7-sc26-dir-r2-20261030-064537` | 2 | 45.53 | 1.36/2.00 | sidecar repeat — the arm's own spread is 2.6 tok/s (6 %) |
| `t7-ref-ndt4-20261030-065614` | 4 | **40.45** | 1.73/4.00 | stock head |
| `t7-sc26-ndt4-20261030-070733` | 4 | 38.67 | 1.69/4.00 | sidecar |

**Verdict: the optimized sidecar does not beat the stock head — do not deploy it.** At the deployed
window (2 tokens) its acceptance is *identical* (1.35/2.00 and 1.36/2.00 against 1.35/2.00) while its
own two runs span 42.90–45.53, i.e. the tok/s difference is inside the arm's spread and the head buys
nothing. At window 4 — where the card's "uniform-K8 acceptance" claim would have to show — it is
**slower and accepts fewer**: 38.67 vs 40.45 tok/s, 1.69 vs 1.73 accepted of 4. The stock head also
confirms the T6 result independently: window 4 loses ~13 % to window 2 (40.45 vs 46.48) today, on
the fork wheel, in a fresh process.

Per the plan's own rule ("2.3 bpw and 3.1 bpw only if 2.6 bpw beats the reference"), the two smaller
sidecars are **not run** — spending 40 minutes on heads bracketing a loser would be waste, not
coverage.

## Step 4 — DFlash2 BF16, window 4

`t7-dflash2-w4-20261030-071928`: **56.57 tok/s**, accepted **2.40/4.00**, floors clean
(min free VRAM 1473 MB, RAM 11060 MB), `breach=0`. State variant `state-dflash.json`
(`draft.mode: none` so no `-mtp`, `-dm D:\llms\Qwen3.8-Flash-Next-DFlash2`).

**1.22× the incumbent** (56.57 vs 46.48 at window 2) — the first drafter in the campaign that beats
the stock MTP head, and it does so at a *wider* window, where the MTP head loses 13 %.

The mean hides the shape, so per category against `t7-ref` (MTP, window 2):

| category | MTP w2 | DFlash2 w4 | ratio | DFlash2 acceptance |
|---|---|---|---|---|
| Agentic; code | 53.0 | 73.5 | **1.39** | 3.09/4 (77 %) |
| Agentic; curl | 50.5 | 66.9 | **1.32** | 2.62/4 (66 %) |
| Coding | 59.0 | 80.5 | **1.36** | 2.82/4 (70 %) |
| Creative | 37.2 | 44.4 | 1.19 | 1.87/4 (47 %) |
| Creative (reasoning) | 38.5 | 45.6 | 1.19 | 1.88/4 (47 %) |
| Translation | 42.2 | 37.9 | **0.90** | 1.47/4 (37 %) |
| Translation (reasoning) | 42.3 | 37.1 | **0.88** | 1.47/4 (37 %) |
| Trivial repetition | 49.2 | 66.8 | 1.36 | 3.99/4 (100 %) |
| **median** | | | **1.258** | |

**Reading:** DFlash2's block-diffusion drafter is excellent where the next tokens follow a
structural pattern — code, tool-call JSON, repeated text (66–100 % acceptance) — and below
break-even on translation (37 %), where its block prediction disagrees with the target and the
verification of four tokens costs more than the one it lands. This is a *workload-dependent*
drafter, not a uniform upgrade: deploying it helps a coding/agentic mix by ~35 % and hurts a
translation mix by ~11 %. The window ladder (6, 8) and the QSA block-size question follow; the
deployment decision belongs to T13 with the operator's traffic mix in hand.

## Step 4b — window ladder 6 and 8 (and the drafter's real limit)

| arm (run id) | window | tok/s | accepted/round | min free VRAM | verdict |
|---|---|---|---|---|---|
| `t7-dflash2-w4-20261030-071928` | 4 | **56.57** | 2.40/4.00 | 1473 MB | best |
| `t7-dflash2-w6-20261030-073115` | 6 | 55.45 | 2.94/6.00 | 745 MB | 2 % slower than w4, margin thin |
| `t7-dflash2-w8-20261030-073945` | 8 | — | — | — | **crashed at 175 s**, `status=fail` |

The w8 crash is an engine contract problem, not a measurement:

```
generator.py:1387  self.draft_ids_pinned[:batch_size, :window].copy_(new_ids[:batch_size, :window])
RuntimeError: The size of tensor a (8) must match the size of tensor b (7) at non-singleton dimension 1
```

`dflash_config.block_size` in the DFlash2 config is **8**: the drafter emits one block of 8, the
generator crops the first column (`new_ids[:, 1:]`) expecting `window + 1` columns, so any window
`≥ block_size` asks for one column more than the model can emit. Windows 4 and 6 work; 8 is out of
contract and should be rejected with a message rather than a shape error. (Recorded as an engine
defect; the fix would clamp `window = min(window, block_size - 1)` where the drafter's block size is
known, or validate at load.)

**This answers the plan's QSA/block-size question with numbers:** the drafter's block is 8, the
engine's usable window is ≤ 7, and the ladder is monotone from w4 down — widening the window does
not buy acceptance on this box (w4 60 % per token, w6 49 %, and the verification cost per round
grows with it).

## Step 6 — exl3 conversion of the drafter

Not run. The BF16 drafter fits with 1473 MB of headroom at `-cs 40960` and split 376, so the
conversion would buy VRAM margin only, and the 262144 question it would inform is measured directly
in T13 (`size_check` puts 262144 + MTP at 1664 MiB free after reserve; DFlash2 costs 1150 MiB of
weights + 248 MiB of draft cache on top of that). If T13 finds 262144 + DFlash2 does not fit, the
conversion becomes the first thing the next session runs — not a hidden omission.

## Step 7 — `DRAFT_WINNER`

**`DFlash2` BF16, window 4.** It is the only drafter measured that beats the stock MTP head
(56.57 vs 46.48 tok/s, +22 %; median per-category ratio 1.258), it fits at the current split with
1.4 GB free, and its one regression is confined to translation (−11 %, acceptance 37 % there vs
77 % on agentic code). The stock MTP head remains the fallback for a translation-heavy mix and for
any deployment where a second draft model cannot be configured. The sidecar is closed as a negative.
