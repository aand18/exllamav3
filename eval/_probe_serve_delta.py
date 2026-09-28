"""Micro A/B: serve kernel with cached Ew vs fresh full-refresh Ew (GPU).

Alternates kvarn_triton_online_serve calls (same args except the Ew
source) 200x to cancel drift. If cached-Ew serve is slower, the gap is
a residency/aliasing effect in the serve read; if equal, look elsewhere.
Usage: python eval/_probe_serve_delta.py
"""
import time
import torch

from exllamav3 import Config, Model, Tokenizer, Cache
from exllamav3.cache import CacheLayer_kvarn
from exllamav3.cache.kvarn import (
    kvarn_parse_preset, KVAR_N_SINK_TOKENS, KVAR_N_GROUP)
from exllamav3.constants import PAGE_SIZE
from exllamav3.modules.attention_fn.kvarn_triton import (
    kvarn_triton_qwht, kvarn_triton_wht_rows,
    kvarn_triton_online_serve, _kvarn_online_buffers)
from kvarn_microkld import SAMPLER_TEXT, populate

MODEL = ("C:/Users/yoho/Downloads/tabbyAPI/models/"
         "Qwen3.8-27B-exl3-SC_1.40bpw_H3_V3")
NTOK, CHUNK, ITERS = 8192, 4096, 200


def main():
    k_bits, v_bits = kvarn_parse_preset("kvarn4")
    config = Config.from_directory(MODEL)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=NTOK + 512,
                  layer_type=CacheLayer_kvarn, k_bits=k_bits, v_bits=v_bits)
    model.load("cuda:0", progressbar=False)
    tokenizer = Tokenizer.from_config(config)
    reps = max(16, (NTOK // 24) + 2)
    ids = tokenizer.encode(SAMPLER_TEXT * reps)[:, :NTOK]
    n = int(ids.shape[1])
    states, _ = populate(model, cache, ids, CHUNK, n)
    del states
    torch.cuda.synchronize()
    lay0 = next(iter(cache.layers.values()))
    kvh, hd, sl = lay0.num_kv_heads, lay0.head_dim, lay0.slices
    qh, qpk = 24, 6
    assert kvh * qpk == qh and sl == 2
    max_tok = NTOK + 512
    bt = torch.arange(max_tok // 256, dtype=torch.int32,
                      device="cuda").unsqueeze(0).expand(1, -1).contiguous()
    gps = PAGE_SIZE // KVAR_N_GROUP
    scale, sink_n, tail_eff = 0.0625, 128, int(lay0.tail_effective)
    qpad = 1 << (qpk - 1).bit_length()
    sscale = 0.7071067811865475
    gc_eff = (n + 127) // 128
    torch.manual_seed(11)
    Q1 = torch.randn(qh, hd, dtype=torch.float16, device="cuda")
    Qf = Q1.float()
    seqlens = torch.tensor([n], dtype=torch.int32, device="cuda")
    n_0d = seqlens[:1] + 1
    _mb, _lb, _ab, qw, qs, _o = _kvarn_online_buffers(
        lay0, qh, qpad, hd, torch.device("cuda"))
    kvarn_triton_qwht(Q1, qs, qw, sl, sscale)
    torch.cuda.synchronize()

    with torch.inference_mode():
        Ew_cache = lay0._eref_ensure()
        Ew_fresh = kvarn_triton_wht_rows(lay0.exact_v.float(), hd)
        assert torch.equal(Ew_cache, Ew_fresh), "cache != fresh"
        print("cache==fresh: True", flush=True)

        def do_serve(Ew):
            return kvarn_triton_online_serve(
                lay0, qw, Qf, lay0.exact_k, Ew, lay0._exact_rev,
                lay0.sealed, bt[0], n_0d, qpk, scale, sink_n,
                tail_eff, gps, gc=gc_eff)[0]

        do_serve(Ew_cache)
        do_serve(Ew_fresh)
        tc, tf = [], []
        for _ in range(ITERS):
            t0 = time.perf_counter()
            do_serve(Ew_cache)
            torch.cuda.synchronize()
            tc.append((time.perf_counter() - t0) * 1e6)
            t0 = time.perf_counter()
            do_serve(Ew_fresh)
            torch.cuda.synchronize()
            tf.append((time.perf_counter() - t0) * 1e6)
        tc, tf = torch.tensor(tc), torch.tensor(tf)
        print(f"serve_cached: mean {float(tc.mean()):.1f}us median "
              f"{float(tc.median()):.1f}us", flush=True)
        print(f"serve_fresh : mean {float(tf.mean()):.1f}us median "
              f"{float(tf.median()):.1f}us", flush=True)


if __name__ == "__main__":
    main()
