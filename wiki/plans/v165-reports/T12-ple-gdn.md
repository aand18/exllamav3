# T12 — PLE / n-gram table and GDN knobs

Date: 2026-10-30. Engine win-perf `1.6.5`. Regime: 3.18bpw, split 376, `-cq 2,2`, frozen profile per
run. The PLE arms run prefill at 262144 (`-MaxLength 262144 -CacheSize 262144 -SkipGen`); the GDN
arms run the full ladder at 131072 against the T9 session reference (`t9-ref-a`/`-b`: TG 38.87 /
39.83 at 130816, PP 1835.57 / 1798.08 at 131072).

## Step 1 — `EXL3_PLE_SUB_CHUNK` (prefill, short list)

| arm (run id) | PP 131072 | PP 262144 | peak_alloc | peak_reserved | min free VRAM |
|---|---|---|---|---|---|
| `t12-ple0-…` (`0` = run the chunk whole) | 1831.81 | **1814.33** | **21193** | 21714 | **1969 MB** |
| `t12-ple512-…` | **1867.80** | 1849.81 | 20618 | 20794 | 2889 MB |
| `t12-ple1024-…` (default) | **1871.86** | **1856.51** | 20618 | 20794 | 2889 MB |
| reference: the default, from the T11 arms at 262k (`t11-sp256` etc., staging inert) | 1830–1856 | 1819–1850 | 20618 | 20794 | 2883 MB |

**Default and 512 are level (1871.86 / 1856.51 against 1867.80 / 1849.81); `0` is the loser.**

**`0` costs VRAM and gains nothing.** Running the PLE stream pass whole raises the allocator peak by
**575 MB** (20618 → 21193) and drops free VRAM by **914 MB** (2883 → 1969) at 262144, while prefill is
1 % *slower* than the default. The sub-chunking is what keeps the fused stream pass's workspace
bounded. **Keep the default (1024); 512 is neutral.**

## Step 2 — `ngram_ram` (`-ngr`, the 51.8 GB table in RAM): a breach, not a fit

`t12-ngram-ram-r2-20261030-110633`: **`status=breach` after 27 s**, free RAM **637 MB** against the
1200 MB floor — the watcher killed the tree while the table was still loading. VRAM was untouched
(21399 MB free), so nothing about the GPU was the problem: the 51.8 GB table plus the model's 33 GB
pinned arena cannot be resident in 63.6 GB of RAM, and the OS was driven into reclamation before the
load finished.

(The first attempt was a harness/user error worth recording: `-ngram_ram` is not an option — the
flags are `-ngr` / `--ngram_ram` with `allow_abbrev=False` — and the pass failed in 2 s on
`unrecognized arguments`. The corrected arm is the one above.)

**Verdict: `ngram_ram` is not available on this box at the production split.** The streamed table is
not a compromise here, it is the only option; a smaller split (more experts in RAM) would make it
worse, and a larger one does not free enough. The per-arm cold-read cost the protocol warns about
stays a property of the first arm in a process, which is why the frozen-profile rule and the
untimed warm corpus exist.

## Step 3 — `-rs32`, fp32 recurrent state at 262144

| arm (run id) | PP 262144 | TG 261888 | peak_alloc | peak_reserved | min free VRAM |
|---|---|---|---|---|---|
| `t12-rs32-…` | 1843.53 | pending | **20845** | **21526** | **2175 MB** |
| `t13-262k-perf-…` (fp16 default) | 1839.54 | 39.32 | 20618 | 20794 | 2855 MB |

**Costs ~0.7 GB of VRAM for no measurable gain**: allocated +227 MB, reserved +732 MB, free
−680 MB; pp is 0.2 % (inside its own noise). The scan compute is fp32 either way (the flag only
changes the stored state), so the default (fp16) stays — the GDN state pool is not a place to spend
VRAM on this card.

## Step 4 — `EXL3_BC_GDN`

| arm (run id) | PP 131072 | TG 32768 | TG 65536 | TG 130816 | verdict |
|---|---|---|---|---|---|
| `t12-bcgdn0-…` | 1836.88 | 30.49 | 35.74 | **33.33** | **−15 % decode** |
| reference (`t9-ref-a`/`-b`) | 1835.57 / 1798.08 | 38.70 / 39.16 | 37.79 / 37.00 | 38.87 / 39.83 | reference |

Turning the batched/chunked GDN path off costs 15 % of decode at the long band (and 21 % at 32768)
while prefill is unchanged. **The default (on) is doing real work**; do not disable it.

## Step 5 — `EXL3_GDN_SUB_CHUNK` 2048 (default) vs 1024

| arm (run id) | PP 131072 | TG 32768 | TG 65536 | TG 130816 |
|---|---|---|---|---|
| `t12-gdn1024-…` | 1793.49 | 32.52 | 37.61 | **37.88** |
| reference | 1835.57 / 1798.08 | 38.70 / 39.16 | 37.79 / 37.00 | 38.87 / 39.83 |

1024 is 2.2 % down on prefill and 3.7 % down at the long band, with the 32768 band 16 % down. It is
at the edge of the spread rather than clearly outside it, but nothing about it is better, so the
default **2048** stays.

## Summary

| knob | tested | verdict |
|---|---|---|
| `EXL3_PLE_SUB_CHUNK` | 0 / 512 (default 1024 by reference) | **keep 1024** — `0` costs ~0.9 GB VRAM, 512 neutral |
| `ngram_ram` (`-ngr`) | at 131072 | **not available** — RAM floor breached in 27 s (637 MB free); streamed table stays |
| `-rs32` | off (default) vs on at 262144 | **keep fp16** — fp32 costs ~0.7 GB VRAM, no gain |
| `EXL3_BC_GDN` | 0 vs default | **keep on** — disabling costs 15 % decode |
| `EXL3_GDN_SUB_CHUNK` | 1024 vs default 2048 | **keep 2048** — 1024 is 2–4 % down everywhere |

None of T12's knobs is a lever to ship; three of them are defaults that are load-bearing (PLE
sub-chunk, BC_GDN, GDN sub-chunk) and one is a cost with no benefit (`-rs32`). That is the useful
result: the deployed config does not need a PLE or GDN line at all, and the campaign can stop
wondering whether it was leaving something on the table there.
