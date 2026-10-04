"""Raw partial inspection (UNTRACKED)."""
import sys
import torch
sys.path.insert(0, 'eval')
import _spike5_single as S5
from exllamav3.cache.kvarn import kvarn_make_layout
from exllamav3.modules.attention_fn.kvarn_triton import kvarn_triton_wht_rows
import _spike2_online as S2
torch.manual_seed(5)
kvh, sl, hd, qpk, Gg = 4, 2, 256, 6, 1
bits = (4, 4)
layout = kvarn_make_layout(128, 128, bits[0], bits[1])
records = S2._make_records(Gg, kvh, sl, layout, bits[0], bits[1])
qh = kvh * qpk
Q = torch.randn(qh, hd, dtype=torch.float16, device="cuda")
qw = kvarn_triton_wht_rows(Q.float(), hd)
n = 100
exact_k = torch.zeros((Gg, 128, kvh, hd), dtype=torch.float16, device="cuda")
exact_v = torch.full((Gg, 128, kvh, hd), 2.0, dtype=torch.float16, device="cuda")
exrev = torch.zeros(Gg, dtype=torch.int64, device="cuda")
sealed = torch.zeros(Gg, dtype=torch.bool, device="cuda")
bt = torch.zeros(1, dtype=torch.int32, device="cuda")
n_0d = torch.tensor([n], dtype=torch.int32, device="cuda")
dev = "cuda"
C, B = records.shape[1], records.shape[2]
m = torch.empty((qh, Gg), dtype=torch.float32, device=dev)
l = torch.empty_like(m)
acc = torch.empty((qh, Gg, hd), dtype=torch.float32, device=dev)
flag = torch.zeros((1,), dtype=torch.uint8, device=dev)
rec_f16 = records.view(torch.float16)
S5._serve_kernel[(qh, Gg,)](
    qw, records, rec_f16, exact_k, exact_v, exrev, sealed, bt, n_0d,
    flag, m, l, acc,
    layout.k_payload_off,
    layout.k_s_col_off // 2, layout.k_zp_off // 2, layout.k_s_row_off // 2, bits[0],
    layout.v_payload_off,
    layout.v_s_row_off // 2, layout.v_zp_off // 2, layout.v_s_col_off // 2, bits[1],
    C, B, sl, 2,
    kvh, qpk, qh, hd, Gg, 0.0625, 512, 128,
    num_warps=4)
print("m[0]:", m[0].tolist(), "(expect ~0)")
print("l[0]:", l[0].tolist(), "(expect ~100)")
print("acc[0,0,:6]:", acc[0, 0, :6].tolist(), "(expect ~200s)")
print("flag:", int(flag[0]), flush=True)
