"""Tail-only isolation for spike5 (UNTRACKED)."""
import sys
import torch
sys.path.insert(0, 'eval')
import _spike5_single as S5
from exllamav3.cache.kvarn import kvarn_make_layout
from exllamav3.modules.attention_fn.kvarn_triton import kvarn_triton_wht_rows
import _spike2_online as S2
torch.manual_seed(5)
kvh, sl, hd, qpk, Gg = 4, 2, 256, 6, 4
bits = (4, 4)
layout = kvarn_make_layout(128, 128, bits[0], bits[1])
records = S2._make_records(Gg, kvh, sl, layout, bits[0], bits[1])
qh = kvh * qpk
Q = torch.randn(qh, hd, dtype=torch.float16, device="cuda")
qw = kvarn_triton_wht_rows(Q.float(), hd)
n = 400
exact_k = torch.randn(Gg, 128, kvh, hd, dtype=torch.float16, device="cuda")
exact_v = torch.randn_like(exact_k)
exrev = torch.arange(Gg, dtype=torch.int64, device="cuda")
sealed = torch.zeros(Gg, dtype=torch.bool, device="cuda")
bt = torch.arange(2, dtype=torch.int32, device="cuda")
n_0d = torch.tensor([n], dtype=torch.int32, device="cuda")
# sink covers everything -> all rows tail.
from exllamav3.modules.attention_fn.kvarn_triton import (
        kvarn_triton_wht_rows as _wht2)
Qf = Q.float()
exact_v_w = _wht2(exact_v.float(), hd)
out, flag = S5.serve_online(qw, Qf, records, layout, bits[0], bits[1],
                            exact_k, exact_v_w, exrev, sealed, bt, n_0d,
                            kvh, qpk, sl, hd, 2, 512, 128)
print("flag:", flag, flush=True)
g = torch.arange(n, device="cuda") // 128
s = torch.arange(n, device="cuda") % 128
Ke = exact_k[g, s].float().reshape(-1, kvh, hd)
Ve = exact_v[g, s].float().reshape(-1, kvh, hd)
outs = []
for h in range(kvh):
    q = Q[h * qpk:(h + 1) * qpk].float()
    p = torch.softmax((q @ Ke[:, h, :].T) * 0.0625, dim=-1)
    outs.append(p @ Ve[:, h, :])
ref = torch.cat(outs)
d = (out - ref).abs()
print(f"tail-only maxdiff={float(d.max()):.3e} RMSE={float((d**2).mean().sqrt()):.3e}", flush=True)
