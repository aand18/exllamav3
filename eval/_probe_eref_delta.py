"""Micro A/B: cost of the in-kernel eref write-through (needs GPU).

Advancing raw kvarn_triton_store_row calls over two adjacent 256-step
ranges on lay0: range 1 with the eref cache built (DO_EREF on), range 2
with it dropped (DO_EREF off). Reports mean/median per call + code
histogram. Isolates the extra masked store + args from all host logic
(no evict/seal/touch: raw kernel call only).
Usage: python eval/_probe_eref_delta.py
"""
import time
import torch

from exllamav3 import Config, Model, Tokenizer, Cache
from exllamav3.cache import CacheLayer_kvarn
from exllamav3.cache.kvarn import kvarn_parse_preset, KVAR_N_GROUP
from exllamav3.constants import PAGE_SIZE
from exllamav3.modules.attention_fn.kvarn_triton import (
    kvarn_triton_store_row)
from kvarn_microkld import SAMPLER_TEXT, populate

MODEL = ("C:/Users/yoho/Downloads/tabbyAPI/models/"
         "Qwen3.8-27B-exl3-SC_1.40bpw_H3_V3")
NTOK, CHUNK = 8192, 4096
SPAN = 256


def main():
    k_bits, v_bits = kvarn_parse_preset("kvarn4")
    config = Config.from_directory(MODEL)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=NTOK + 512 + 2 * SPAN,
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
    kvh, hd = lay0.num_kv_heads, lay0.head_dim
    dev = torch.device("cuda")
    max_tok = NTOK + 512 + 2 * SPAN
    bt = torch.arange(max_tok // 256, dtype=torch.int32,
                      device="cuda").unsqueeze(0).expand(1, -1).contiguous()
    gps = PAGE_SIZE // KVAR_N_GROUP
    torch.manual_seed(5)

    def prefill(p0, count):
        for i in range(p0, p0 + count):
            k = torch.randn(1, 1, kvh, hd, dtype=torch.float16, device="cuda")
            v = torch.randn_like(k)
            se = torch.tensor([i], dtype=torch.int32, device="cuda")
            lay0.update_kv_direct(se, bt, k, v, 1)
        torch.cuda.synchronize()

    HALF = SPAN // 4  # 64: prefill half a group (open), time the rest

    def timed_range(p0, label):
        ts, codes = [], {}
        for i in range(p0, p0 + HALF):
            k = torch.randn(1, 1, kvh, hd, dtype=torch.float16, device="cuda")
            v = torch.randn_like(k)
            se = torch.tensor([i], dtype=torch.int64, device="cuda")
            pages_1 = bt[0, se // 256].long()
            offs_1 = (se % 256).long()
            t0 = time.perf_counter()
            code, _g = kvarn_triton_store_row(
                lay0, k[0, 0], v[0, 0], pages_1, offs_1, se,
                gps, 128, 128, None, None)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1e6
            codes[code] = codes.get(code, 0) + 1
            if code == 0:
                ts.append(dt)
        ts = torch.tensor(ts)
        print(f"{label}: n0={len(ts)} mean {float(ts.mean()):.1f}us "
              f"median {float(ts.median()):.1f}us codes {codes}", flush=True)

    with torch.inference_mode():
        lay0._eref_ensure()
        assert lay0._ov_eref_w is not None
        prefill(n, HALF)
        timed_range(n + HALF, "do_eref_ON ")
        lay0._ov_eref_w = None
        prefill(n + SPAN, HALF)
        timed_range(n + SPAN + HALF, "do_eref_OFF")


if __name__ == "__main__":
    main()
