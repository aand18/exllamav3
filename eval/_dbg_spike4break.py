"""Breakdown of spike4 full path (UNTRACKED)."""
import sys
import torch
sys.path.insert(0, 'eval')
import _spike4_online as S4
from exllamav3.cache.kvarn import kvarn_make_layout
from exllamav3.modules.attention_fn.kvarn_triton import kvarn_triton_wht_rows

torch.manual_seed(3)
kvh, sl, hd, qpk, qh, Gg, nb = 4, 2, 256, 6, 24, 63, 63
bits = (4, 4)
layout = kvarn_make_layout(128, 128, bits[0], bits[1])
t0 = torch.cuda.Event(enable_timing=True)
t1 = torch.cuda.Event(enable_timing=True)

def timeit(fn, iters=100, warm=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0.record()
    for _ in range(iters):
        fn()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) / iters

print("building records...", flush=True)
records = S4._tail_partials  # touch import
import _spike2_online as S2
records = S2._make_records(Gg, kvh, sl, layout, bits[0], bits[1])
rec_f16 = records.view(torch.float16)
Q = torch.randn(qh, hd, dtype=torch.float16, device="cuda")
qw = torch.empty((qh, hd), dtype=torch.float32, device="cuda")
qscratch = torch.empty_like(qw)
gids = torch.arange(nb, dtype=torch.int64, device="cuda")
qpad = 8
dev = "cuda"
m = torch.empty((kvh, qpad, nb), dtype=torch.float32, device=dev)
l = torch.empty_like(m)
acc = torch.empty((kvh, qpad, nb, hd), dtype=torch.float32, device=dev)
out = torch.empty((qh, hd), dtype=torch.float32, device=dev)
C, B = records.shape[1], records.shape[2]
L = layout
n_0 = torch.tensor([nb * 128], dtype=torch.int32, device=dev)
R = 256
Kt = torch.randn(R, kvh, hd, dtype=torch.float32, device=dev)
Vt = torch.randn_like(Kt)
tm = torch.empty((kvh, qpad, 2), dtype=torch.float32, device=dev)
tll = torch.empty_like(tm)
tacc = torch.empty((kvh, qpad, 2, hd), dtype=torch.float32, device=dev)

def do_qwht():
    return _s3qwht(Q)
import _spike3_online as _s3
def do_qwht():
    return _s3.qwht_fused(Q, qscratch, qw, sl, 0.7071067811865475)
def do_stage1():
    return _s3._block_kernel_h2[(kvh, nb,)](
        qw, records, rec_f16, gids, m, l, acc,
        L.k_payload_off, L.k_s_col_off // 2, L.k_zp_off // 2, L.k_s_row_off // 2, bits[0],
        L.v_payload_off, L.v_s_row_off // 2, L.v_zp_off // 2, L.v_s_col_off // 2, bits[1],
        C, B, kvh, qpk, qpad, hd, nb, 0.0625, num_warps=4)
def do_tailsplit():
    return S4._tail_block_kernel[(kvh, 2,)](
        Q, Kt, Vt, tm, tll, tacc, n_0, 256, kvh, qpk, qpad, hd, 0.0625, num_warps=4)
def do_merge():
    return S4._mega_merge_kernel[(qh,)](
        m, l, acc, tm, tll, tacc, out, kvh, qpk, qpad, nb, 64, hd, sl, 0.7071067811865475, num_warps=1)
qw_ = do_qwht(); do_stage1(); do_tailsplit()
print(f"qwht:      {timeit(do_qwht) * 1e3:7.1f} us", flush=True)
print(f"stage1:    {timeit(do_stage1) * 1e3:7.1f} us", flush=True)
print(f"tailsplit: {timeit(do_tailsplit) * 1e3:7.1f} us", flush=True)
print(f"megamerge: {timeit(do_merge) * 1e3:7.1f} us", flush=True)
