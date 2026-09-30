"""Standalone inductor exactness+speed probe for the seal math
(Sinkhorn normalize + RTN quantize + bit-pack). No model load.

Loads the real exllamav3.cache.kvarn math with stubbed parents (same
pattern as tests/test_kvarn_cpu.py), then compares eager vs
torch.compile(fullgraph=False) on realistic tile batches, bit-exact
(torch.equal on every output) + wall time.
Usage: python eval/_probe_seal_compile.py
"""
import importlib.util
import sys
import time
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
EXL = ROOT / "exllamav3"


def _stub(name):
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, EXL / rel)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    parent, _, attr = name.rpartition(".")
    if parent and parent in sys.modules:
        setattr(sys.modules[parent], attr, m)
    spec.loader.exec_module(m)
    return m


def main():
    _pkg = _stub("exllamav3")
    _pkg.__path__ = [str(EXL)]
    _cache_pkg = _stub("exllamav3.cache")
    _cache_pkg.__path__ = [str(EXL / "cache")]
    _constants = _load("exllamav3.constants", "constants.py")
    _cache_mod = _load("exllamav3.cache.cache", "cache/cache.py")
    _fp16m = _stub("exllamav3.cache.fp16")
    _fp16m.CacheLayer_fp16 = type("CacheLayer_fp16",
                                  (_cache_mod.CacheLayer,), {})
    _quantm = _stub("exllamav3.cache.quant")
    _quantm.CacheLayer_quant = type("CacheLayer_quant",
                                    (_cache_mod.CacheLayer,), {})
    _qsa = _load("exllamav3.cache.qsa", "cache/qsa.py")
    kvarn = _load("exllamav3.cache.kvarn", "cache/kvarn.py")

    dev = torch.device("cuda:0")
    iters = kvarn.KVAR_N_SINKHORN_ITERS

    def seal_math(k_tiles, v_tiles, k_bits, v_bits):
        qk, sck, zpk, otk = kvarn.kvarn_quantize_tile(
            k_tiles, k_bits, iters)
        qv, scv, zpv, otv = kvarn.kvarn_quantize_tile(
            v_tiles, v_bits, iters)
        pk = kvarn.kvarn_pack_bits(
            qk.transpose(1, 2).contiguous().reshape(-1), k_bits)
        pv = kvarn.kvarn_pack_bits(qv.reshape(-1), v_bits)
        return qk, sck, zpk, otk, qv, scv, zpv, otv, pk, pv

    compiled = None
    try:
        compiled = torch.compile(seal_math, fullgraph=False)
        have_compile = True
    except Exception as e:
        print(f"COMPILE-UNAVAILABLE: {e}", flush=True)
        have_compile = False

    torch.manual_seed(0)
    for tag, ntiles, kb, vb in (("G232k4v4", 232, 4, 4),
                                ("G232k4v2", 232, 4, 2),
                                ("G8k4v4", 8, 4, 4)):
        kt = torch.randn(ntiles, 128, 128, dtype=torch.float32,
                         device=dev)
        vt = torch.randn(ntiles, 128, 128, dtype=torch.float32,
                         device=dev)
        ref = seal_math(kt, vt, kb, vb)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(3):
            ref = seal_math(kt, vt, kb, vb)
        torch.cuda.synchronize()
        t_eager = (time.perf_counter() - t0) / 3 * 1e3
        if not have_compile:
            print(f"{tag}: eager={t_eager:.2f}ms (no compiled run)",
                  flush=True)
            continue
        try:
            got = compiled(kt, vt, kb, vb)  # compile here
            torch.cuda.synchronize()
        except Exception as e:
            print(f"{tag}: COMPILE-FAILED: {type(e).__name__}: {e}",
                  flush=True)
            have_compile = False
            continue
        t0 = time.perf_counter()
        for _ in range(5):
            got = compiled(kt, vt, kb, vb)
        torch.cuda.synchronize()
        t_comp = (time.perf_counter() - t0) / 5 * 1e3
        exact = all(torch.equal(a, b) for a, b in zip(ref, got))
        maxdiff = max(float((a.float() - b.float()).abs().max())
                      for a, b in zip(ref, got))
        print(f"{tag}: eager={t_eager:.2f}ms compiled={t_comp:.2f}ms "
              f"exact={exact} maxdiff={maxdiff:.3e}", flush=True)
    print("SURVIVED", flush=True)


if __name__ == "__main__":
    main()
