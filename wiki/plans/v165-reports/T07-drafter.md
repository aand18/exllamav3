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

| arm (run id) | tok/s | accepted/round | min free VRAM | note |
|---|---|---|---|---|
| `t7-ref-…062147` | 46.48 | 1.35/2.00 | 2343 MB | stock head |
| `t7-sc26-dir-…063352` | 42.90 | 1.35/2.00 | 2343 MB | sidecar 2.6 bpw |
| `t7-sc26-dir-r2` | pending | pending | pending | repeat, for the spread |
| `t7-ref-ndt4` | pending | pending | pending | stock head, window 4 |
| `t7-sc26-ndt4` | pending | pending | pending | sidecar, window 4 |

**Reading so far:** acceptance is *identical* to the reference (1.35 of 2.00) while the head is
demonstrably different, and tok/s is 7.7 % lower. The card's claim for this checkpoint is
uniform-K8 acceptance — a property of a *window of 8*, which a 2-token window cannot express. The
window-4 pair is the fair test; it decides whether the head is neutral (no reason to deploy it) or
only helps at a window this campaign already rejected on the stock head (`-ndt 4` lost to `-ndt 2`
by 1.30× in T6).
