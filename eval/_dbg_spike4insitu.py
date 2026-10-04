"""In-situ breakdown of the spike4 probe path (UNTRACKED)."""
import sys, torch
sys.path.insert(0, 'eval')
import _spike4_online as S4
from exllamav3.cache.kvarn import kvarn_make_layout
from exllamav3.modules.attention_fn.kvarn_triton import kvarn_triton_wht_rows
import _spike3_online as _s3
from exllamav3.modules.attention_fn.kvarn_triton import _kvarn_online_block_kernel
torch.manual_seed(3)
kvh, sl, hd, qpk, qh, Gg, nb = 4, 2, 256, 6, 24, 63, 63
bits = (4, 4)
layout = kvarn_make_layout(128, 128, bits[0], bits[1])
import _spike2_online as S2
records = S2._make_records(Gg, kvh, sl, layout, bits[0], bits[1])
rec_f16 = records.view(torch.float16)
dev = "cuda"
Q = torch.randn(qh, hd, dtype=torch.float16, device=dev)
qw = torch.empty((qh, hd), dtype=torch.float32, device=dev)
qscratch = torch.empty_like(qw)
gids = torch.arange(nb, dtype=torch.int64, device=dev)
qpad = 8
m = torch.empty((kvh, qpad, nb), dtype=torch.float32, device=dev)
l = torch.empty_like(m)
acc = torch.empty((kvh, qpad, nb, hd), dtype=torch.float32, device=dev)
out = torch.empty((qh, hd), dtype=torch.float32, device=dev)
C, B = records.shape[1], records.shape[2]
R = 256
Kt = torch.randn(R, kvh, hd, dtype=torch.float32, device=dev)
Vt = torch.randn_like(Kt)
tm = torch.empty((kvh, qpad, 2), dtype=torch.float32, device=dev)
tll = torch.empty_like(tm)
tacc = torch.empty((kvh, qpad, 2, hd), dtype=torch.float32, device=dev)
n_0 = torch.tensor([nb * 128], dtype=torch.int32, device=dev)
L = layout
acc_t = {"qwht": 0.0, "stage1": 0.0, "tailsplit": 0.0, "merge": 0.0}
EV0 = torch.cuda.Event(enable_timing=True)
EV1 = torch.cuda.Event(enable_timing=True)
EV2 = torch.cuda.Event(enable_timing=True)
EV3 = torch.cuda.Event(enable_timing=True)
EV4 = torch.cuda.Event(enable_timing=True)

def do_spike():
    _s3.qwht_fused(Q, qscratch, qw, sl, 0.7071067811865475)
    EV0.record()
    _kvarn_online_block_kernel[(kvh, nb,)](
        qw, records, rec_f16, gids, m, l, acc,
        L.k_payload_off, L.k_s_col_off // 2, L.k_zp_off // 2, L.k_s_row_off // 2, bits[0],
        L.v_payload_off, L.v_s_row_off // 2, L.v_zp_off // 2, L.v_s_col_off // 2, bits[1],
        C, B, sl, kvh, qpk, qpad, hd, nb, 0.0625, n_0, 128, 128, num_warps=4)
    EV1.record()
    S4._tail_block_kernel[(kvh, 2,)](
        Q, Kt, Vt, tm, tll, tacc, n_0, 256, kvh, qpk, qpad, hd, 0.0625, num_warps=4)
    EV2.record()
    S4._mega_merge_kernel[(qh,)](
        m, l, acc, tm, tll, tacc, out, kvh, qpk, qpad, nb, 64, hd, sl, 0.7071067811865475, num_warps=1)
    EV3.record()
    return out

for _ in range(10):
    do_spike()
torch.cuda.synchronize()
N = 100
evs = [torch.cuda.Event(enable_timing=True) for _ in range(4 * N + 1)]
for i in range(N):
    _s3.qwht_fused(Q, qscratch, qw, sl, 0.7071067811865475)
    evs[4 * i].record()
    _kvarn_online_block_kernel[(kvh, nb,)](
        qw, records, rec_f16, gids, m, l, acc,
        L.k_payload_off, L.k_s_col_off // 2, L.k_zp_off // 2, L.k_s_row_off // 2, bits[0],
        L.v_payload_off, L.v_s_row_off // 2, L.v_zp_off // 2, L.v_s_col_off // 2, bits[1],
        C, B, sl, kvh, qpk, qpad, hd, nb, 0.0625, n_0, 128, 128, num_warps=4)
    evs[4 * i + 1].record()
    S4._tail_block_kernel[(kvh, 2,)](
        Q, Kt, Vt, tm, tll, tacc, n_0, 256, kvh, qpk, qpad, hd, 0.0625, num_warps=4)
    evs[4 * i + 2].record()
    S4._mega_merge_kernel[(qh,)](
        m, l, acc, tm, tll, tacc, out, kvh, qpk, qpad, nb, 64, hd, sl, 0.7071067811865475, num_warps=1)
    evs[4 * i + 3].record()
torch.cuda.synchronize()
import statistics as st
qw = [evs[4*i].elapsed_time(evs[4*i+1]) for i in range(N)]
s1 = [evs[4*i+1].elapsed_time(evs[4*i+2]) for i in range(N)]
ts = [evs[4*i+2].elapsed_time(evs[4*i+3]) for i in range(N)]
mg = [evs[4*i+3].elapsed_time(evs[4*i+4]) if 4*i+4 < len(evs) else 0 for i in range(N-1)]
print(f"qwht med: {st.median(qw):.1f}us  stage1 med: {st.median(s1):.1f}us  tailsplit med: {st.median(ts):.1f}us  merge med: {st.median(mg):.1f}us", flush=True)
print(f"sum med: {(st.median(qw)+st.median(s1)+st.median(ts)+st.median(mg)):.1f}us", flush=True)
