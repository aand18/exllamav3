"""Prefill cost attribution at 8k (needs GPU).

Populates 4k, then profiles ONE 4k prefill chunk with stack traces.
Prints wall time plus CUDA time attributed to kvarn frames
(seal/dequant/store/refresh/get_kv), identifying the top piece.
Usage: python eval/_probe_prefill.py
"""
import torch

from exllamav3 import Config, Model, Tokenizer, Cache
from exllamav3.cache import CacheLayer_kvarn
from exllamav3.cache.kvarn import kvarn_parse_preset
from kvarn_microkld import SAMPLER_TEXT, _bshape

MODEL = ("C:/Users/yoho/Downloads/tabbyAPI/models/"
         "Qwen3.8-27B-exl3-SC_1.40bpw_H3_V3")
NTOK, CHUNK = 8192, 4096


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
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    bs = _bshape(ids.shape[1])
    states, past = None, 0
    # Populate first chunk unprofiled (warmup + state).
    p = {"cache": cache, "attn_mode": "flash_attn",
         "batch_shape": (1, bs), "past_len": 0}
    out = model.forward(ids[:, :CHUNK], p)
    states = p.get("recurrent_states")
    del out
    torch.cuda.synchronize()

    # Profile the second chunk.
    import time as _time
    p2 = {"cache": cache, "attn_mode": "flash_attn",
          "batch_shape": (1, bs), "past_len": CHUNK,
          "recurrent_states": states}
    t0 = _time.perf_counter()
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True) as prof:
        out2 = model.forward(ids[:, CHUNK:2 * CHUNK], p2)
        torch.cuda.synchronize()
        del out2
    dt = (_time.perf_counter() - t0) * 1e3
    print(f"profiled chunk wall: {dt:.0f}ms", flush=True)
    av = prof.key_averages(group_by_input_shape=True)
    print(str(av.table(sort_by="self_device_time_total", row_limit=40)),
          flush=True)
    print("SURVIVED", flush=True)


if __name__ == "__main__":
    main()
