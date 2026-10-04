"""Spike v1: can the serve+combine kernels CUDA-graph-capture cleanly?

Synthetic, no model load. Warms up, captures
_kvarn_online_serve_kernel + _kvarn_online_combine_kernel in one
torch.cuda.CUDAGraph, replays, and compares vs eager (correctness +
per-replay time). Answers the core mechanical unknown for decode
graphs (masked loads? tl.debug_barrier in combine? flag stores?).
Static shapes only (graphs need them); dynamic-shape recapture,
memory pools, and dispatch integration are v2 problems.
Usage: python eval/_spike_graph_serve.py
"""
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from exllamav3.modules.attention_fn import kvarn_triton as kt
from exllamav3.cache import kvarn
from _spike2_online import _make_records


def main():
    assert torch.cuda.is_available()
    dev = torch.device("cuda:0")
    torch.manual_seed(15)
    kvh, sl, hd, qpk = 2, 1, 128, 2
    qh = kvh * qpk
    qpad = 1 << (qpk - 1).bit_length()
    layout = kvarn.kvarn_make_layout(128, 128, 4, 4)
    G, n = 8, 1024
    gps, scale = 2, hd ** -0.5
    nbpad = 1 << (G - 1).bit_length()

    Q = torch.randn(qh, hd, dtype=torch.float16, device=dev)
    Qf = Q.float()
    Qw = kt.kvarn_triton_wht_rows(Q.float(), hd)
    records = _make_records(G, kvh, sl, layout, 4, 4)
    rec_f16 = records.view(torch.float16)
    exact_k = torch.zeros(G, 128, kvh, hd, dtype=torch.float16, device=dev)
    exact_v_w = torch.zeros(G, 128, kvh, hd, dtype=torch.float32, device=dev)
    exrev = torch.arange(G, dtype=torch.int64, device=dev)
    sealed = torch.ones(G, dtype=torch.bool, device=dev)
    bt = torch.arange((n + 255) // 256, dtype=torch.int32, device=dev)
    n_0d = torch.tensor([n], dtype=torch.int32, device=dev)
    m = torch.empty((kvh, qpad, G), dtype=torch.float32, device=dev)
    l = torch.empty((kvh, qpad, G), dtype=torch.float32, device=dev)
    acc = torch.empty((kvh, qpad, G, hd), dtype=torch.float32, device=dev)
    out = torch.empty((qh, hd), dtype=torch.float32, device=dev)
    flag = torch.zeros((1,), dtype=torch.uint8, device=dev)

    L = layout
    sargs = (Qw, Qf, records, rec_f16, exact_k, exact_v_w, exrev, sealed,
             bt, n_0d, flag, m, l, acc,
             L.k_payload_off, L.k_s_col_off // 2, L.k_zp_off // 2,
             L.k_s_row_off // 2, 4,
             L.v_payload_off, L.v_s_row_off // 2, L.v_zp_off // 2,
             L.v_s_col_off // 2, 4,
             records.shape[1], records.shape[2], sl, gps,
             kvh, qpk, qpad, hd, G, scale, 0, 0, 1, G)
    cargs = (m, l, acc, out, kvh, qpk, qpad, G, nbpad, hd, sl, 1.0)

    def eager():
        flag.zero_()
        kt._kvarn_online_serve_kernel[(kvh, G,)](*sargs,
                                                 num_warps=4, num_stages=1)
        kt._kvarn_online_combine_kernel[(qh,)](*cargs, num_warps=1)
        return out

    # Warmup (triton compile happens here, outside capture).
    for _ in range(10):
        eager()
    torch.cuda.synchronize()
    ref = eager().clone()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(100):
        eager()
    torch.cuda.synchronize()
    t_eager = (time.perf_counter() - t0) / 100 * 1e3

    # Capture.
    try:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            flag.zero_()
            kt._kvarn_online_serve_kernel[(kvh, G,)](*sargs,
                                                     num_warps=4,
                                                     num_stages=1)
            kt._kvarn_online_combine_kernel[(qh,)](*cargs, num_warps=1)
        print("CAPTURE: ok", flush=True)
    except Exception as e:
        print(f"CAPTURE-FAILED: {type(e).__name__}: {e}", flush=True)
        return

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        g.replay()
    torch.cuda.synchronize()
    t_graph = (time.perf_counter() - t0) / 100 * 1e3

    d = (out.float() - ref.float()).abs()
    print(f"EAGER: {t_eager:.3f}ms GRAPH: {t_graph:.3f}ms "
          f"speedup={t_eager / t_graph:.2f}x", flush=True)
    print(f"maxabs={float(d.max()):.3e} meanabs={float(d.mean()):.3e}",
          flush=True)
    print("SURVIVED", flush=True)


if __name__ == "__main__":
    main()
