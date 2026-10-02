"""Spike v2b: capture via WRAPPERS (not bare kernels).

Same sublattice as _spike_graph_full.py but dispatch-style: qwht,
serve (incl. combine), tail_gather, bmm, tail_reduce, merge wrappers
with persistent bufs + precomputed views. Decides whether dispatch
v2 can capture wrapper calls as-is (rec_f16 view? shape checks? env
reads?) or must drop to bare kernels. Compares wrapper-graph replay
vs wrapper-eager (bar: exact) and wrapper-eager vs kernel-eager.
Usage: python eval/_spike_graph_wrap.py
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


def run_spike():
    assert torch.cuda.is_available()
    dev = torch.device("cuda:0")
    torch.manual_seed(15)
    kvh, sl, hd, qpk = 2, 1, 128, 2
    qh = kvh * qpk
    layout = kvarn.kvarn_make_layout(128, 128, 4, 4)
    G, n = 8, 1024
    gps, scale = 2, hd ** -0.5
    R = 128

    Q = torch.randn(qh, hd, dtype=torch.float16, device=dev)
    qs = torch.empty((qh, hd), dtype=torch.float32, device=dev)
    qw = torch.empty((qh, hd), dtype=torch.float32, device=dev)
    Qf = torch.empty((qh, hd), dtype=torch.float32, device=dev)
    Qf.copy_(Q)
    Qh = Qf.reshape(kvh, qpk, hd)
    records = _make_records(G, kvh, sl, layout, 4, 4)
    exact_k = torch.zeros(G, 128, kvh, hd, dtype=torch.float16, device=dev)
    exact_v = torch.zeros(G, 128, kvh, hd, dtype=torch.float16, device=dev)
    exact_v_w = torch.zeros(G, 128, kvh, hd, dtype=torch.float32, device=dev)
    exrev = torch.arange(G, dtype=torch.int64, device=dev)
    valid = torch.ones(G, dtype=torch.bool, device=dev)
    sealed = torch.ones(G, dtype=torch.bool, device=dev)
    bt = torch.arange((n + 255) // 256, dtype=torch.int32, device=dev)
    n_0d = torch.tensor([n], dtype=torch.int32, device=dev)
    K = torch.zeros((R, kvh, hd), dtype=torch.float32, device=dev)
    V = torch.zeros((R, kvh, hd), dtype=torch.float32, device=dev)
    ev = torch.empty((R,), dtype=torch.bool, device=dev)
    gg = torch.empty((R,), dtype=torch.int64, device=dev)
    ss = torch.empty((R,), dtype=torch.int64, device=dev)
    Kt = K.permute(1, 2, 0)
    Vt = V
    st = torch.empty((kvh, qpk, R), dtype=torch.float32, device=dev)
    tail_m = torch.empty((kvh, qpk), dtype=torch.float32, device=dev)
    tail_den = torch.empty((kvh, qpk), dtype=torch.float32, device=dev)
    tail_num = torch.empty((kvh, qpk, hd), dtype=torch.float32, device=dev)
    tm_v = tail_m.reshape(qh)
    td_v = tail_den.reshape(qh)
    tn_v = tail_num.reshape(qh, hd)
    out = torch.empty((qh, hd), dtype=torch.float16, device=dev)
    lay = SimpleNamespace(
        records=records, layout=layout, k_bits=4, v_bits=4,
        num_kv_heads=kvh, head_dim=hd, slices=sl, exact_k=exact_k,
        exact_v=exact_v, exact_v_w=exact_v_w, _exact_rev=exrev,
        exact_valid=valid, exrev=exrev, sealed=sealed)
    tpos = torch.arange(n - R, n, dtype=torch.int64, device=dev)

    def eager_w():
        kt.kvarn_triton_qwht(Q, qs, qw, sl, 1.0)
        out_bw, _ = kt.kvarn_triton_online_serve(
            lay, qw, Qf, exact_k, exact_v_w, exrev, sealed, bt, n_0d,
            qpk, scale, 0, 0, gps, gc=G, sync_flag=False)
        kt.kvarn_triton_online_tail_gather(
            lay, tpos, bt, K, V, gps, (ev, gg, ss))
        torch.bmm(Qh, Kt, out=st)
        torch.mul(st, scale, out=st)
        kt.kvarn_triton_online_tail_reduce(
            st, Vt, gg, exrev, (tail_m, tail_den, tail_num))
        return kt.kvarn_triton_online_merge(
            lay._ov_serve_m, lay._ov_serve_l, out_bw,
            tm_v, td_v, tn_v, qpk, G, out)

    for _ in range(10):
        eager_w()
    torch.cuda.synchronize()
    ref = eager_w().clone()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(100):
        eager_w()
    torch.cuda.synchronize()
    t_eager = (time.perf_counter() - t0) / 100 * 1e3

    try:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            kt.kvarn_triton_qwht(Q, qs, qw, sl, 1.0)
            out_bw, _ = kt.kvarn_triton_online_serve(
                lay, qw, Qf, exact_k, exact_v_w, exrev, sealed, bt,
                n_0d, qpk, scale, 0, 0, gps, gc=G, sync_flag=False)
            kt.kvarn_triton_online_tail_gather(
                lay, tpos, bt, K, V, gps, (ev, gg, ss))
            torch.bmm(Qh, Kt, out=st)
            torch.mul(st, scale, out=st)
            kt.kvarn_triton_online_tail_reduce(
                st, Vt, gg, exrev, (tail_m, tail_den, tail_num))
            kt.kvarn_triton_online_merge(
                lay._ov_serve_m, lay._ov_serve_l, out_bw,
                tm_v, td_v, tn_v, qpk, G, out)
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
    return float(d.max()), t_eager / t_graph


def main():
    maxabs, speedup = run_spike()
    assert maxabs == 0.0, maxabs


if __name__ == "__main__":
    main()
