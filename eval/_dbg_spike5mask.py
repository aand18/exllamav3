"""Masked-load isolation (UNTRACKED)."""
import torch, triton, triton.language as tl
torch.manual_seed(0)
kvh, hd = 4, 256
Gg = 4
exact = torch.full((Gg, 128, kvh, hd), 2.0, dtype=torch.float16, device="cuda")

@triton.jit
def _loadmask(k_ptr, out_ptr, g, s, h, flag,
              KVH: tl.constexpr, HD: tl.constexpr):
    lane = tl.arange(0, HD)
    e_off = (g * 128 + s) * KVH * HD + h * HD + lane
    ok = tl.load(flag) > 0
    k = tl.load(k_ptr + e_off, mask=ok, other=0.0).to(tl.float32)
    tl.store(out_ptr + lane, k)

for flag_val, tag in ((1, "mask-true"), (0, "mask-false")):
    out = torch.empty((hd,), dtype=torch.float32, device="cuda")
    flag = torch.tensor([flag_val], dtype=torch.uint8, device="cuda")
    _loadmask[(1,)](exact, out, 2, 17, 3, flag, kvh, hd)
    print(f"{tag}: out[:4]={out[:4].tolist()} sum={float(out.sum())} (expect {2.0*256 if flag_val else 0.0})", flush=True)
