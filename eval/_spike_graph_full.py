"""Spike v2: capture the FULL per-layer decode sublattice in one graph.

qwht + serve + combine + tail_gather + bmm + tail_reduce + merge,
synthetic all-assigned setup (exrev>=0 everywhere, so the tail block
is numerically zero and merge == body -- degenerate but exercises
every launch, mask, and temp). Static shapes, persistent buffers;
views (Qh, Kt, Vt) precomputed outside capture. Compares replay vs
eager (correctness + per-replay time). No model load.
Usage: python eval/_spike_graph_full.py
"""
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from types import SimpleNamespace
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
    R, MAXW = 128, 128
    rpad = 1 << (R - 1).bit_length()

    Q = torch.randn(qh, hd, dtype=torch.float16, device=dev)
    qs = torch.empty((qh, hd), dtype=torch.float32, device=dev)
    qw = torch.empty((qh, hd), dtype=torch.float32, device=dev)
    Qf = torch.empty((qh, hd), dtype=torch.float32, device=dev)
    Qf.copy_(Q)
    Qh = Qf.reshape(kvh, qpk, hd)
    records = _make_records(G, kvh, sl, layout, 4, 4)
    rec_f16 = records.view(torch.float16)
    exact_k = torch.zeros(G, 128, kvh, hd, dtype=torch.float16, device=dev)
    exact_v = torch.zeros(G, 128, kvh, hd, dtype=torch.float16, device=dev)
    exact_v_w = torch.zeros(G, 128, kvh, hd, dtype=torch.float32, device=dev)
    exrev = torch.arange(G, dtype=torch.int64, device=dev)
    valid = torch.ones(G, dtype=torch.bool, device=dev)
    sealed = torch.ones(G, dtype=torch.bool, device=dev)
    bt = torch.arange((n + 255) // 256, dtype=torch.int32, device=dev)
    n_0d = torch.tensor([n], dtype=torch.int32, device=dev)
    m = torch.empty((kvh, qpad, G), dtype=torch.float32, device=dev)
    l = torch.empty((kvh, qpad, G), dtype=torch.float32, device=dev)
    acc = torch.empty((kvh, qpad, G, hd), dtype=torch.float32, device=dev)
    out_b = torch.empty((qh, hd), dtype=torch.float32, device=dev)
    flag = torch.zeros((1,), dtype=torch.uint8, device=dev)
    tpos = torch.arange(n - R, n, dtype=torch.int64, device=dev)
    K = torch.zeros((MAXW, kvh, hd), dtype=torch.float32, device=dev)
    V = torch.zeros((MAXW, kvh, hd), dtype=torch.float32, device=dev)
    ev = torch.empty((R,), dtype=torch.bool, device=dev)
    gg = torch.empty((R,), dtype=torch.int64, device=dev)
    ss = torch.empty((R,), dtype=torch.int64, device=dev)
    Kt = K[:R].permute(1, 2, 0)
    Vt = V[:R]
    st = torch.empty((kvh, qpk, R), dtype=torch.float32, device=dev)
    tail_m = torch.empty((kvh, qpk), dtype=torch.float32, device=dev)
    tail_den = torch.empty((kvh, qpk), dtype=torch.float32, device=dev)
    tail_num = torch.empty((kvh, qpk, hd), dtype=torch.float32, device=dev)
    out = torch.empty((qh, hd), dtype=torch.float16, device=dev)
    lay = SimpleNamespace(
        records=records, layout=layout, k_bits=4, v_bits=4,
        num_kv_heads=kvh, head_dim=hd, slices=sl, exact_k=exact_k,
        exact_v=exact_v, _exact_rev=exrev, exact_valid=valid)

    L = layout
    sargs = (qw, Qf, records, rec_f16, exact_k, exact_v_w, exrev, sealed,
             bt, n_0d, flag, m, l, acc,
             L.k_payload_off, L.k_s_col_off // 2, L.k_zp_off // 2,
             L.k_s_row_off // 2, 4,
             L.v_payload_off, L.v_s_row_off // 2, L.v_zp_off // 2,
             L.v_s_col_off // 2, 4,
             records.shape[1], records.shape[2], sl, gps,
             kvh, qpk, qpad, hd, G, scale, 0, 0, 1, G)
    cargs = (m, l, acc, out_b, kvh, qpk, qpad, G, nbpad, hd, sl, 1.0)

    def eager():
        kt.kvarn_triton_qwht(Q, qs, qw, sl, 1.0)
        flag.zero_()
        kt._kvarn_online_serve_kernel[(kvh, G,)](*sargs,
                                                 num_warps=4, num_stages=1)
        kt._kvarn_online_combine_kernel[(qh,)](*cargs, num_warps=1)
        kt._kvarn_online_tail_gather_kernel[(R, kvh)](
            tpos, bt, exact_k, exact_v, exrev, valid, K, V, ev, gg, ss,
            gps, kvh, hd, num_warps=4)
        torch.bmm(Qh, Kt, out=st)
        torch.mul(st, scale, out=st)
        kt._kvarn_online_tail_kernel[(kvh * qpk,)](
            st, Vt, gg, exrev, tail_m, tail_den, tail_num,
            kvh, qpk, R, rpad, hd, num_warps=4)
        kt._kvarn_online_merge_kernel[(qh,)](
            m, l, out_b,
            tail_m.reshape(qh), tail_den.reshape(qh),
            tail_num.reshape(qh, hd), out,
            kvh, qpk, qpad, G, nbpad, hd, num_warps=4)
        return out

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

    try:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            kt.kvarn_triton_qwht(Q, qs, qw, sl, 1.0)
            flag.zero_()
            kt._kvarn_online_serve_kernel[(kvh, G,)](*sargs,
                                                     num_warps=4,
                                                     num_stages=1)
            kt._kvarn_online_combine_kernel[(qh,)](*cargs, num_warps=1)
            kt._kvarn_online_tail_gather_kernel[(R, kvh)](
                tpos, bt, exact_k, exact_v, exrev, valid, K, V, ev, gg,
                ss, gps, kvh, hd, num_warps=4)
            torch.bmm(Qh, Kt, out=st)
            torch.mul(st, scale, out=st)
            kt._kvarn_online_tail_kernel[(kvh * qpk,)](
                st, Vt, gg, exrev, tail_m, tail_den, tail_num,
                kvh, qpk, R, rpad, hd, num_warps=4)
            kt._kvarn_online_merge_kernel[(qh,)](
                m, l, out_b,
                tail_m.reshape(qh), tail_den.reshape(qh),
                tail_num.reshape(qh, hd), out,
                kvh, qpk, qpad, G, nbpad, hd, num_warps=4)
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
