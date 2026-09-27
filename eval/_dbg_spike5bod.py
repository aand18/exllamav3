"""Body-only isolation for spike5 (UNTRACKED)."""
import sys
import torch
sys.path.insert(0, 'eval')
import _spike5_single as S5
from exllamav3.cache.kvarn import kvarn_make_layout, kvarn_wht_head
from exllamav3.modules.attention_fn.kvarn_triton import (
    kvarn_triton_dequant_groups, kvarn_triton_wht_rows)
import _spike2_online as S2
torch.manual_seed(5)
kvh, sl, hd, qpk, Gg = 4, 2, 256, 6, 4
bits = (4, 4)
layout = kvarn_make_layout(128, 128, bits[0], bits[1])
records = S2._make_records(Gg, kvh, sl, layout, bits[0], bits[1])
qh = kvh * qpk
Q = torch.randn(qh, hd, dtype=torch.float16, device="cuda")
qw = kvarn_triton_wht_rows(Q.float(), hd)
n = 512
# sink=0, tail=0 -> everything body. All 4 groups covered.
exact_k = torch.zeros(Gg, 128, kvh, hd, dtype=torch.float16, device="cuda")
exact_v = torch.zeros_like(exact_k)
exrev = torch.full((Gg,), -1, dtype=torch.int64, device="cuda")
sealed = torch.ones(Gg, dtype=torch.bool, device="cuda")
bt = torch.arange(2, dtype=torch.int32, device="cuda")
n_0d = torch.tensor([n], dtype=torch.int32, device="cuda")
from exllamav3.modules.attention_fn.kvarn_triton import (
        kvarn_triton_wht_rows as _wht2)
Qf = Q.float()
exact_v_w = _wht2(exact_v.float(), hd)
out, flag = S5.serve_online(qw, Qf, records, layout, bits[0], bits[1],
                            exact_k, exact_v_w, exrev, sealed, bt, n_0d,
                            kvh, qpk, sl, hd, 2, 0, 0)
print("flag:", flag, flush=True)
bk, bv = kvarn_triton_dequant_groups(
    records, layout, bits[0], bits[1], kvh, sl, do_wht=False)
K = kvarn_wht_head(bk, hd).reshape(Gg * 128, kvh, hd)
V = kvarn_wht_head(bv, hd).reshape(Gg * 128, kvh, hd)
outs = []
for h in range(kvh):
    q = Q[h * qpk:(h + 1) * qpk].float()
    s = (q @ K[:, h, :].T) * 0.0625
    p = torch.softmax(s, dim=-1)
    outs.append(p @ V[:, h, :])
ref = torch.cat(outs)
d = (out - ref).abs()
print(f"body-only maxdiff={float(d.max()):.3e} RMSE={float((d**2).mean().sqrt()):.3e}", flush=True)
