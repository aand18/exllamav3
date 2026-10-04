"""Constant-exact probe: exact rows all C -> uniform scores -> out must be C."""
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
exact_k = torch.zeros((Gg, 128, kvh, hd), dtype=torch.float16, device="cuda")
exact_v = torch.full((Gg, 128, kvh, hd), 2.0, dtype=torch.float16, device="cuda")
exrev = torch.arange(Gg, dtype=torch.int64, device="cuda")
sealed = torch.zeros(Gg, dtype=torch.bool, device="cuda")
bt = torch.arange(2, dtype=torch.int32, device="cuda")
n_0d = torch.tensor([n], dtype=torch.int32, device="cuda")
from exllamav3.modules.attention_fn.kvarn_triton import (
        kvarn_triton_wht_rows as _wht2)
Qf = Q.float()
exact_v_w = _wht2(exact_v.float(), hd)
out, flag = S5.serve_online(qw, Qf, records, layout, bits[0], bits[1],
                            exact_k, exact_v_w, exrev, sealed, bt, n_0d,
                            kvh, qpk, sl, hd, 2, 512, 128)
print("flag:", flag, flush=True)
print("out mean (expect ~2.0):", float(out.mean()), "std:", float(out.std()), flush=True)
print("out[0,:6]:", out[0, :6].tolist(), flush=True)
