# T01 — Code base verification (2026-10-29)

Read-only verification of the branches/tags/worktrees created 2026-10-29 and of the
regime-relevant defaults at `v1.6.5`. No GPU, no builds. Run from the plan-branch
worktree `/home/dev/.local/share/opencode/worktree/.../165-win-perf`.

## Step 1 — branches, tags, worktrees, knob wiring

### `git -C /home/dev/exllamav3 worktree list | grep v165`

```
/home/dev/exllamav3/.worktrees/v165-stock                                                              cdc785db [v165-stock]
/home/dev/exllamav3/.worktrees/v165-win-perf                                                           d0f55f71 (detached HEAD)
```

Matches the plan: stock worktree at `v1.6.5` (`cdc785db`), win-perf build worktree
detached at the tag commit `d0f55f71`.

### `git log --oneline v1.6.5..win-perf-1.6.5`

```
d0f55f71 plan: Q2 cache operator-fixed; drop the cache-mode ladder
069e2b76 plan: model swap to turboderp 3.68bpw/1.66bpw branches; drafting settled early
9062f655 plan: add T9b drafter comparison (optimized MTP sidecar, DFlash2)
7030053f plan: v1.6.5 retune for Qwen3.8-Flash-Next on Windows
68eea9dc bench: commit Windows measurement harness for the v1.6.4 retune
21c2dfda Docs: Specter plan on the v1.6.4 base (arms, gates, seams, hazards)
1206f590 Ignore .worktrees directory (bare + slash)
5cef00e1 CPU MoE: Windows worker thread and timer knobs
```

Matches the plan's list: worker knobs `5cef00e1`, .gitignore `1206f590`, Specter docs
`21c2dfda`, harness `68eea9dc`, plans.

### Tag vs branch tip (per spawn instruction — the tag was NOT moved to the tip)

```
$ git rev-parse win-perf-1.6.5^{commit}
d0f55f71c9875e4ec6870f3f9f71c2d304a8b195
$ git rev-parse 165-win-perf
5835bb6b1f92588139e7447f4d80bc7ce49b7631
$ git rev-parse v1.6.5^{commit}
cdc785db89e263e65e5849cb53e1ada156411414
```

**tag ≠ branch tip.** `win-perf-1.6.5` is an annotated tag (`objectname c7e5800f`,
`*objectname d0f55f71`) pointing at `d0f55f71`; branch `165-win-perf` is 3 commits
ahead:

```
$ git log --oneline win-perf-1.6.5..165-win-perf
5835bb6b plan: T0 findings folded in; T2 downloads+hardlink done
6ac23d1e plan: T0 preconditions done
09923498 bench: -Model override and Q2 cache in the production regime
```

`git diff --stat win-perf-1.6.5 165-win-perf` touches only `bench/windows/run_pass.ps1`
and `wiki/plans/*` — **no engine code**. The engine source at the tag is identical to
the branch tip; the detached build worktree (at the tag) is fine for T3/T8 builds, but
the deployed harness comes from the repo checkout (branch), which carries the T0
`run_pass.ps1` fix.

### `git grep -n "EXL3_MOE_TIMER_RES\|EXL3_MOE_THREAD_PRIO\|EXL3_MOE_BLOCKING_SYNC\|EXL3_MOE_CPU_PIN_OFFSET" win-perf-1.6.5 -- exllamav3`

```
win-perf-1.6.5:exllamav3/exllamav3_ext/bindings.cpp:103:    if (const char* e = std::getenv("EXL3_MOE_BLOCKING_SYNC"))
win-perf-1.6.5:exllamav3/exllamav3_ext/cpu/moe_mul1.cpp:2792:        const char* e = std::getenv("EXL3_MOE_THREAD_PRIO");
win-perf-1.6.5:exllamav3/exllamav3_ext/cpu/moe_mul1.cpp:2798:// Rotate the start index into the physical-core order (EXL3_MOE_CPU_PIN_OFFSET,
win-perf-1.6.5:exllamav3/exllamav3_ext/cpu/moe_mul1.cpp:2805:        const char* e = std::getenv("EXL3_MOE_CPU_PIN_OFFSET");
win-perf-1.6.5:exllamav3/model/moe_cpu_host.py:177:        # (EXL3_MOE_THREAD_PRIO in moe_mul1.cpp, EXL3_MOE_BLOCKING_SYNC in bindings.cpp)
win-perf-1.6.5:exllamav3/model/moe_cpu_host.py:178:        self.timer_res = os.environ.get("EXL3_MOE_TIMER_RES", "0") != "0"
```

All four fork knobs present. `Pool::pin_self` (the rewritten pinning path) confirmed:

```
$ git grep -n "pin_self" win-perf-1.6.5 -- exllamav3
win-perf-1.6.5:exllamav3/exllamav3_ext/cpu/moe_mul1.cpp:2653: ... decoded by Pool::pin_self -- SetThreadAffinityMask
win-perf-1.6.5:exllamav3/exllamav3_ext/cpu/moe_mul1.cpp:2828:    void pin_self(int idx)
win-perf-1.6.5:exllamav3/exllamav3_ext/cpu/moe_mul1.cpp:2866:        pin_self(idx);
win-perf-1.6.5:exllamav3/exllamav3_ext/cpu/moe_mul1.cpp:2936:        pin_self(0);   // worker 0 is the calling thread itself, ...
```

## Step 2 — defaults at v1.6.5, verified against the tree (not the plan's list)

Verified with `git grep -n <knob> v1.6.5 -- exllamav3` and `git show v1.6.5:<file>`.

| Knob | Plan says | Actual at v1.6.5 | Evidence | Verdict |
|---|---|---|---|---|
| `EXL3_MOE_MEMOPS` | 1 | `"1"` (`!= "0"`) | `model/moe_cpu_host.py:174` | ✅ |
| `EXL3_MOE_STREAM_T` | per-device | base `8`; when unset, scaled per-device by a pinned→device bandwidth probe: `max(8, round(8*(25/bw)**0.5))`; explicit env overrides | `moe_cpu_host.py:138–139`, `:1663–1671` | ✅ (per-device scaling from base 8) |
| `EXL3_MOE_STREAM_BATCH_EXPERTS` | 24 | `24` (clamped to `MOE_JOB_MAX_EXPERTS`) | `moe_cpu_host.py:143` | ✅ |
| `EXL3_MOE_CPU_PIN` | 1 | on unless env is exactly `"0"` (`!(e && *e=='0')`) | `exllamav3_ext/cpu/moe_mul1.cpp:2783–2785` | ✅ |
| `EXL3_MOE_CPU_SWIZZLE` | 1 | `"1"` (`!= "0"`) | `moe_cpu_host.py:130` | ✅ |
| `EXL3_MOE_PINNED_ARENA` | 0 | `"0"` (`!= "0"`) | `model/config.py:54`, `moe_cpu_host.py:157` | ✅ |
| `EXL3_MOE_CPU_PROMOTE` | 4 | `4` (0 = off) | `modules/block_sparse_mlp_cpu.py:842` | ✅ |
| `EXL3_MOE_HOST_CORES` | 1 | `1` | `moe_cpu_host.py:105` | ✅ |
| `EXL3_PLE_SUB_CHUNK` | 1024 | `1024` | `modules/ple.py:125` | ✅ |
| `EXL3_BC_GDN` | 1 | `"1"` (`!= "0"`) | `modules/gated_delta_net.py:23` | ✅ |
| `EXL3_GDN_SUB_CHUNK` | 2048 | `2048` | `modules/gated_delta_net_fn/gated_delta_rule.py:95` | ✅ |
| `EXL3_QC_STAGING` | 1 | `1` | `modules/attention_fn/triton_paged.py:737` | ✅ |
| `EXL3_QC_STAGING_PAGES` | 256 | `256` | `triton_paged.py:743` | ✅ |
| `EXL3_NO_BATCH_READBACK` | 0 | **⚠️ NOT AN ENV KNOB at v1.6.5** — no `getenv` anywhere in v1.6.5, v1.6.4, or `165-win-perf`. It is a hardcoded module constant `_NO_BATCH_READBACK = False` (test hook) in `generator/generator.py:32`, used at `:1867` | `git grep EXL3_NO_BATCH_READBACK v1.6.5` → no match | ❌ **MISMATCH** |

### ⚠️ Mismatch — EXL3_NO_BATCH_READBACK

The plan (knob table line 165 and T6 Step 1: "A/B `EXL3_NO_BATCH_READBACK=1` vs
unset") treats this as an env knob. It does not exist as one in any tree checked
(v1.6.4, v1.6.5, `165-win-perf`); the only occurrences are in the plan document
itself. The batched-readback path IS the default (the test hook is `False`), so the
*state* the plan wants matches, but **T6 cannot A/B it via env** — it would need a
source patch flipping `_NO_BATCH_READBACK` in `generator.py`, or T6's readback arm
must be dropped. Flag for the operator before T6.

## Verdict

**PASS with one flag.** Step 1 matches the plan block (worktrees, 8-commit range,
four fork knobs wired through `Pool::pin_self`). Tag `win-perf-1.6.5` = `d0f55f71`
is **not** the branch tip (`165-win-perf` = `5835bb6b`, 3 commits ahead, bench/wiki
only — engine code identical). 13/14 defaults confirmed at v1.6.5;
`EXL3_NO_BATCH_READBACK` is not an env knob (hardcoded test-hook constant, default
batched-ON) — **T6 must be adjusted before it runs.**
