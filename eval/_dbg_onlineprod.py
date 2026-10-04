"""Validate production kvarn_triton_online_decode (masked block kernel).
UNTRACKED. Run: TRITON=1, no PARITY needed (production import path).
Gates: (1) body-masked RMSE vs torch reference with tail-overlapping
groups; (2) fully-masked call returns zeros without NaN.
"""
import sys
import torch

sys.path.insert(0, 'eval')
from _spike2_online import _make_records
from exllamav3.cache.kvarn import kvarn_make_layout, kvarn_wht_head
from exllamav3.modules.attention_fn.kvarn_triton import (
    kvarn_triton_dequant_groups,
    kvarn_triton_online_decode,
    kvarn_triton_qwht,
)

torch.manual_seed(5)
kvh, sl, hd, qpk = 4, 2, 256, 6
bits = (4, 4)
layout = kvarn_make_layout(128, 128, bits[0], bits[1])


class FakeLayer:
    pass


def make_layer(records):
    lay = FakeLayer()
    lay.num_kv_heads = kvh
    lay.num_groups = int(records.shape[0])
    lay.head_dim = hd
    lay.slices = sl
    lay.layout = layout
    lay.records = records
    lay.k_bits, lay.v_bits = bits
    return lay


def torch_body_ref(records, Q, body_rows, scale=0.0625):
    """Torch attention restricted to body positions (sorted)."""
    bk, bv = kvarn_triton_dequant_groups(
        records, layout, bits[0], bits[1], kvh, sl, do_wht=False)
    K = kvarn_wht_head(bk, hd).reshape(-1, kvh, hd)
    V = kvarn_wht_head(bv, hd).reshape(-1, kvh, hd)
    outs = []
    for h in range(kvh):
        q = Q[h * qpk:(h + 1) * qpk].float()
        s = (q @ K[body_rows][:, h, :].T) * scale
        p = torch.softmax(s, dim=-1)
        outs.append(p @ V[body_rows][:, h, :])
    return torch.cat(outs)


# Case 1: 4 sealed groups, n=400 -> tail [272,400), sink [0,128).
# Body rows: [128,272). Group 2 (256-383) straddles the tail edge.
Gg = 4
records = _make_records(Gg, kvh, sl, layout, bits[0], bits[1])
lay = make_layer(records)
qh = kvh * qpk
Q = torch.randn(qh, hd, dtype=torch.float16, device="cuda")
qw = kvarn_triton_qwht(
    Q, torch.empty_like(Q.float()), torch.empty_like(Q.float()), sl,
    0.7071067811865475)
ids = torch.arange(Gg, dtype=torch.int64, device="cuda")
n_new = torch.tensor([400], dtype=torch.int32, device="cuda")
out = kvarn_triton_online_decode(lay, qw, ids, qpk, 0.0625, n_new, 128, 128)
body_rows = list(range(128, 272))
ref = torch_body_ref(records, Q, body_rows)
d = (out - ref).abs()
print(f"masked body: maxdiff={float(d.max()):.3e} "
      f"RMSE={float((d**2).mean().sqrt()):.3e}", flush=True)
assert float((d**2).mean().sqrt()) < 1e-6

# Case 2: fully masked (n=100 < sink+tail) -> zeros, no NaN.
n_small = torch.tensor([100], dtype=torch.int32, device="cuda")
out2 = kvarn_triton_online_decode(lay, qw, ids, qpk, 0.0625, n_small, 128,
                                  128)
print(f"fully masked: has_nan={bool(torch.isnan(out2).any())} "
      f"maxabs={float(out2.abs().max()):.3e}", flush=True)
assert not bool(torch.isnan(out2).any())
assert float(out2.abs().max()) == 0.0
print("PRODUCTION MASKED KERNEL PASS", flush=True)
