"""Spike7-pre: coalescing microbench for K-payload row reads (UNTRACKED).

Thesis: K payload is dim-major (v = dd*128+s). A (16,256) row-tile read
touches consecutive-dd bytes 64B apart -> ~64x L2-sector amplification.
V payload is slot-major (v = s*128+dd) -> rows contiguous (128B/row).

This bench isolates the pattern: same (16,256) nibble tile, same grid as
spike6 serve (264 programs), strided (dd*128+s) vs transposed (s*128+dd)
indexing, over a 16MB L2-resident buffer. Prints both timings + ratio.

Usage: python eval/_dbg_coal.py
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _read_strided(buf_ptr, out_ptr, NCH: tl.constexpr, HD: tl.constexpr):
    pid = tl.program_id(0)
    lane = tl.arange(0, HD)
    dd = lane % 128
    s = (pid * 16 + tl.arange(0, 16)) % 128  # (16,) slots
    vv = dd[None, :] * 128 + s[:, None]  # dim-major like K
    nb = vv // 2
    acc = tl.zeros([16, HD], dtype=tl.float32)
    for c in tl.static_range(8):
        b = tl.load(buf_ptr + c * 2 * 1024 * 1024 + nb).to(tl.int32)
        acc += ((b >> ((vv % 2) * 4)) & 0xF).to(tl.float32)
    tl.store(out_ptr + pid * 16 * HD
             + tl.arange(0, 16)[:, None] * HD + lane[None, :], acc)


@triton.jit
def _read_coal(buf_ptr, out_ptr, NCH: tl.constexpr, HD: tl.constexpr):
    pid = tl.program_id(0)
    lane = tl.arange(0, HD)
    dd = lane % 128
    s = (pid * 16 + tl.arange(0, 16)) % 128
    vv = s[:, None] * 128 + dd[None, :]  # slot-major like V
    nb = vv // 2
    acc = tl.zeros([16, HD], dtype=tl.float32)
    for c in tl.static_range(8):
        b = tl.load(buf_ptr + c * 2 * 1024 * 1024 + nb).to(tl.int32)
        acc += ((b >> ((vv % 2) * 4)) & 0xF).to(tl.float32)
    tl.store(out_ptr + pid * 16 * HD
             + tl.arange(0, 16)[:, None] * HD + lane[None, :], acc)


def main():
    torch.manual_seed(1)
    buf = torch.randint(0, 256, (16 * 1024 * 1024,), dtype=torch.uint8,
                        device="cuda")
    out = torch.empty((264, 16, 256), dtype=torch.float32, device="cuda")

    def hot(fn, iters=200):
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(iters):
            fn()
        t1.record()
        torch.cuda.synchronize()
        return t0.elapsed_time(t1) / iters

    t_s = hot(lambda: _read_strided[(264,)](buf, out, 8, 256))
    t_c = hot(lambda: _read_coal[(264,)](buf, out, 8, 256))
    print(f"strided (K-like): {t_s * 1e3:.1f} us", flush=True)
    print(f"coalesced (V-like): {t_c * 1e3:.1f} us", flush=True)
    print(f"ratio strided/coal: {t_s / t_c:.2f}", flush=True)


if __name__ == "__main__":
    main()
