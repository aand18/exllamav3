"""Isolate exact_k/v loads (UNTRACKED)."""
import torch, triton, triton.language as tl
torch.manual_seed(0)
kvh, hd = 4, 256
Gg = 4
exact_k = torch.full((Gg, 128, kvh, hd), 0.25, dtype=torch.float16, device="cuda")
exact_v = torch.full((Gg, 128, kvh, hd), 2.0, dtype=torch.float16, device="cuda")

@triton.jit
def _loadrow(k_ptr, v_ptr, out_ptr, g, s, h,
             KVH: tl.constexpr, HD: tl.constexpr):
    lane = tl.arange(0, HD)
    e_off = (g * 128 + s) * KVH * HD + h * HD + lane
    k = tl.load(k_ptr + e_off).to(tl.float32)
    v = tl.load(v_ptr + e_off).to(tl.float32)
    tl.store(out_ptr + lane, k)
    tl.store(out_ptr + HD + lane, v)

out = torch.empty((2, hd), dtype=torch.float32, device="cuda")
_loadrow[(1,)](exact_k, exact_v, out, 2, 17, 3, kvh, hd)
print("k[2,17,3,:6]:", out[0, :6].tolist(), "(expect 0.25s)")
print("v[2,17,3,:6]:", out[1, :6].tolist(), "(expect 2.0s)")
