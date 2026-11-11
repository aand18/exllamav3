#!/usr/bin/env python3
"""Gate 1 (T65): how many GPU expert slots are needed to cut distinct CPU misses?

Zero GPU. Reads the engine's saved routing-frequency profile (per layer, per expert decayed
selection sums) and simulates a drafted verify window: `rows x topk` picks per layer, drawn from
that layer's own distribution. An expert is a CPU miss when it is not in the GPU resident set.

Two policies:
  uniform  - every layer holds the same number of slots (today's design: split 381 of 512)
  global   - one pool ranked across all layers, allocated by each layer's own entropy/coverage
             demand (Strata's non-uniform pool)

Calibration is the whole point of this file, and it has already failed once:

  * `-mcs N` names the **CPU** side (model_init.py: "run the TAIL N routed experts on the CPU"), so at
    split 381 the GPU holds 512-381 = **131** experts per layer, not 381. The first version of this
    simulator passed 381 as the GPU resident set and produced 0.49 misses/layer where the harness
    measures 21 - a 40x error that looked like a plausible "bytes don't matter" result.
  * the measured target was also misread: 8 was an all-rows median dominated by the undrafted rows=1
    arm. The drafted window is median 21, mean 20.5.

Both mistakes produced confident, wrong output. That is why the calibration step runs first, prints
the measured target beside the simulated one, and declares every number below it void when they
disagree - and why `uniform_resident()` takes the number of GPU slots, not the split value.
"""

from __future__ import annotations

import glob
import json
import math
import os
import random
import statistics as st
import sys

PROFILE_GLOB = "/mnt/c/Users/yoho/Downloads/tabbyAPI/_agent/v165/engine-state/master/exllamav3/expert_stats/*.json"
ROWS = 5
TOPK = 10
N_EXPERTS = 512
CPU_SPLIT = 381          # -mcs 381 = the TAIL 381 experts run on the CPU
GPU_SLOTS = N_EXPERTS - CPU_SPLIT   # 131 GPU-resident experts per layer today
TRIALS = 400          # sampled windows per layer
SEED = 12345


def load_profile():
    """The richest saved profile: {layer_key: [freq per expert]}"""
    best = None
    for p in glob.glob(PROFILE_GLOB):
        d = json.load(open(p))
        layers = {k: v for k, v in d.get("layers", {}).items() if isinstance(v, list) and len(v) > 100}
        if layers and (best is None or len(layers) > len(best[1])):
            best = (p, layers)
    if best is None:
        sys.exit("no usable expert profile found")
    path, layers = best
    layers = {k: v for k, v in layers.items() if ".layers." in k}
    return path, layers


def cumulative(weights):
    out, acc = [], 0.0
    for w in weights:
        acc += w
        out.append(acc)
    return out, acc


def sample_picks(weights, cum, total, rng, n):
    picks = []
    for _ in range(n):
        r = rng.random() * total
        lo, hi = 0, len(cum) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if cum[mid] < r:
                lo = mid + 1
            else:
                hi = mid
        picks.append(lo)
    return picks


def simulate(layers, resident_for_layer, rng):
    """distinct CPU misses per layer, over TRIALS windows"""
    per_layer = {}
    for key in sorted(layers, key=lambda s: int(s.split(".")[-2])):
        freqs = layers[key]
        cum, total = cumulative(freqs)
        if total <= 0:
            continue
        resident = resident_for_layer(key, freqs)
        misses = []
        for _ in range(TRIALS):
            picks = sample_picks(freqs, cum, total, rng, ROWS * TOPK)
            misses.append(len({e for e in picks if e not in resident}))
        per_layer[key] = (st.median(misses), st.mean(misses))
    return per_layer


def uniform_resident(n_slots):
    """Hot-to-cold static split: the layer's n_slots most frequent experts"""
    def f(key, freqs):
        order = sorted(range(len(freqs)), key=lambda e: -freqs[e])
        return set(order[:n_slots])
    return f


def global_resident(total_slots, layers):
    """One pool ranked across layers; each layer gets slots by greedy marginal coverage.

    Greedy: start every layer empty, repeatedly give the next slot to the (layer, expert) pair
    that removes the most expected CPU picks. This is the optimal allocation for a fixed total,
    so it is an upper bound on what any global pool can do.
    """
    keys = [k for k in sorted(layers, key=lambda s: int(s.split(".")[-2]))]
    demand = {}
    for k in keys:
        freqs = layers[k]
        cum, total = cumulative(freqs)
        if total > 0:
            demand[k] = [f / total for f in freqs]
    alloc = {k: set() for k in keys}
    for _ in range(total_slots):
        best_key, best_exp, best_gain = None, None, -1.0
        for k, probs in demand.items():
            if len(alloc[k]) >= len(probs):
                continue
            for e, p in enumerate(probs):
                if e in alloc[k]:
                    continue
                # expected picks removed by holding this expert for one window
                gain = p * ROWS * TOPK
                if gain > best_gain:
                    best_key, best_exp, best_gain = k, e, gain
        if best_key is None:
            break
        alloc[best_key].add(best_exp)
                    # greedy is O(slots x layers x experts); cap the search with a pruning list
    return alloc


def global_resident_fast(total_slots, layers):
    """Same greedy allocation, computed by sorting on expected picks (equivalent for a single
    token of greed, which is exact while gains stay independent -- true for distinct-count)."""
    ranked = []
    for k, freqs in layers.items():
        s = sum(freqs)
        if s <= 0:
            continue
        for e, f in enumerate(freqs):
            ranked.append((f / s * ROWS * TOPK, k, e))
    ranked.sort(reverse=True)
    alloc = {k: set() for k in layers}
    for gain, k, e in ranked[:total_slots]:
        alloc[k].add(e)
    return alloc


def main():
    path, layers = load_profile()
    n_layers = len(layers)
    n_experts = max(len(v) for v in layers.values())
    print(f"profile     {os.path.basename(path)}: {n_layers} layers x {n_experts} experts")
    print(f"window      rows={ROWS} topk={TOPK} -> {ROWS*TOPK} picks/layer, {TRIALS} windows/layer, seed {SEED}")
    print()

    rng = random.Random(SEED)

    # --- calibration: today's static split (381 of 512 resident) ---
    cal = simulate(layers, uniform_resident(GPU_SLOTS), rng)
    med = st.median(v[0] for v in cal.values())
    mean = st.mean(v[1] for v in cal.values())
    print(f"CALIBRATION (split {CPU_SPLIT}: {GPU_SLOTS} GPU slots/layer, uniform)")
    print(f"  distinct CPU misses/layer: median {med:.1f}  mean {mean:.2f}")
    print(f"  measured (T66 correction): median 21   mean 20.5  (drafted window, rows=5)")
    ok = abs(med - 21) <= 2.0 and abs(mean - 20.5) <= 2.0
    print(f"  calibration {'OK' if ok else 'FAILED - model is wrong, numbers below are void'}")
    print()

    # --- the question: how many slots buy what ---
    # uniform policies at increasing per-layer residency
    print("UNIFORM pool, GPU slots per layer (today: %d)" % GPU_SLOTS)
    for per_layer in (GPU_SLOTS, 140, 160, 180, 200, 220, 240):
        res = uniform_resident(per_layer)
        sim = simulate(layers, res, random.Random(SEED))
        m = st.median(v[0] for v in sim.values())
        mn = st.mean(v[1] for v in sim.values())
        print(f"  {per_layer:3d}/layer ({per_layer*n_layers:5d} total): median {m:4.1f}  mean {mn:5.2f}")
    print()

    print("GLOBAL non-uniform pool (Strata shape)")
    for total in (GPU_SLOTS * n_layers, 7000, 7700, 8301, 9000, 10000):
        alloc = global_resident_fast(total, layers)
        hold = lambda k, f, a=alloc: a[k]
        sim = simulate(layers, hold, random.Random(SEED))
        m = st.median(v[0] for v in sim.values())
        mn = st.mean(v[1] for v in sim.values())
        held = [len(alloc[k]) for k in alloc]
        print(f"  {total:5d} total ({total/n_layers:5.1f}/layer avg, {min(held)}..{max(held)}): "
              f"median {m:4.1f}  mean {mn:5.2f}")
    print()

    # --- what pool size reaches Argon's gate (mean <= 4.5, or <= 5.0 for the pass) ---
    print("POOL SIZE NEEDED")
    for target in (5.0, 4.5, 3.0):
        lo, hi = GPU_SLOTS * n_layers, N_EXPERTS * n_layers
        while lo < hi:
            mid = (lo + hi) // 2
            alloc = global_resident_fast(mid, layers)
            hold = lambda k, f, a=alloc: a[k]
            sim = simulate(layers, hold, random.Random(SEED))
            mn = st.mean(v[1] for v in sim.values())
            if mn <= target:
                hi = mid
            else:
                lo = mid + 1
        print(f"  mean <= {target:4.1f} misses/layer needs ~{lo} global slots "
              f"({lo/n_layers:.0f}/layer avg)")

    # --- where the VRAM would come from ---
    print()
    print("VRAM BUDGET (from size_check, Q4 at 262144)")
    kv_full, kv_32k, expert_mib = 2911, 360, 87.7
    freed = kv_full - kv_32k
    print(f"  KV at 262144: {kv_full} MiB; at 32768: ~{kv_32k} MiB -> frees {freed} MiB")
    print(f"  at {expert_mib} MiB/expert that is ~{freed/expert_mib:.0f} experts "
          f"-> {GPU_SLOTS + freed/expert_mib:.0f}/layer uniform ({int((GPU_SLOTS + freed/expert_mib)*n_layers)} total)")


if __name__ == "__main__":
    main()
