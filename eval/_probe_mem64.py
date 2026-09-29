"""Per-phase VRAM trace for 64k KLD order (needs GPU).

Replicates kvarn_microkld phase order: fp16 prefill -> fp16 decode
(manual loop) -> del ref + empty_cache -> kvarn prefill, printing
allocated/reserved at each boundary plus per-chunk during kvarn
prefill. Shows what the fp16 phase leaves behind.
Usage: python eval/_probe_mem64.py
"""
import gc
import torch
import torch.nn.functional as F

from exllamav3 import Config, Model, Tokenizer, Cache
from exllamav3.cache import CacheLayer_fp16, CacheLayer_kvarn
from exllamav3.cache.kvarn import kvarn_parse_preset
from kvarn_microkld import SAMPLER_TEXT, _bshape, populate

MODEL = ("C:/Users/yoho/Downloads/tabbyAPI/models/"
         "Qwen3.8-27B-exl3-SC_1.40bpw_H3_V3")
NTOK, CHUNK, DEC = 65536, 4096, 64


def gb(x):
    return x / 1e9


def main():
    k_bits, v_bits = kvarn_parse_preset("kvarn4")
    config = Config.from_directory(MODEL)
    model = Model.from_config(config)
    max_tok = ((NTOK + 511) // 256) * 256
    c_ref = Cache(model, max_num_tokens=max_tok,
                  layer_type=CacheLayer_fp16)
    c_kvarn = Cache(model, max_num_tokens=max_tok,
                    layer_type=CacheLayer_kvarn,
                    k_bits=k_bits, v_bits=v_bits)
    model.load("cuda:0", progressbar=False)
    tokenizer = Tokenizer.from_config(config)
    reps = max(16, (NTOK // 24) + 2)
    ids = tokenizer.encode(SAMPLER_TEXT * reps)
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    ids = ids[:, :NTOK]
    n = int(ids.shape[1])
    print(f"n={n} grad={torch.is_grad_enabled()} "
          f"infer={torch.is_inference_mode_enabled()}", flush=True)

    def mem(tag):
        a, r = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
        print(f"{tag}: alloc={gb(a):.2f}GB reserved={gb(r):.2f}GB",
              flush=True)

    mem("after load")
    states, past, bs = None, 0, _bshape(ids.shape[1])
    ic = 0
    while past < n:
        c = min(CHUNK, n - past)
        p = {"cache": c_ref, "attn_mode": "flash_attn",
             "batch_shape": (1, bs), "past_len": past}
        if states is not None:
            p["recurrent_states"] = states
        out = model.forward(ids[:, past:past + c], p)
        states = p.get("recurrent_states")
        past += c
        del out
        ic += 1
    mem("after fp16 prefill")
    from kvarn_microkld import bench_decode
    # NOTE: bench_decode rebinds its own states; s2 keeps the original
    # list alive like s_ref in the harness.
    s2 = states
    bench_decode(model, c_ref, ids, 256, states, "fp16")
    states = s2
    mem("after fp16 decode")
    for _lay in c_ref.layers.values():
        _free = getattr(_lay, "free", None)
        if callable(_free):
            _free()
    mem("after layer.free loop")
    del c_ref, states, p
    gc.collect()
    torch.cuda.empty_cache()
    mem("after del ref + empty_cache")
    lays = [o for o in gc.get_objects() if isinstance(o, CacheLayer_fp16)]
    print(f"live CacheLayer_fp16 objects: {len(lays)}", flush=True)
    if lays:
        rs = sorted({type(r).__name__ for r in gc.get_referrers(lays[0])})
        print(f"  layer kept-by={rs[:10]}", flush=True)
        for r in gc.get_referrers(lays[0]):
            if isinstance(r, dict):
                keys = [k for k in list(r.keys())[:50]
                        if getattr(k, '__class__', None) is not None]
                import itertools
                hits = [k for k in itertools.islice(r.keys(), 200)
                        if r[k] is lays[0]]
                print(f"  dict-holder keys={hits[:5]} "
                      f"dict-id={id(r)}", flush=True)
                break
    # kvarn prefill per chunk
    states, past = None, 0
    ic = 0
    n0 = int(ids.shape[1])
    while past < n0:
        c = min(CHUNK, n0 - past)
        p = {"cache": c_kvarn, "attn_mode": "flash_attn",
             "batch_shape": (1, bs), "past_len": past}
        if states is not None:
            p["recurrent_states"] = states
        out = model.forward(ids[:, past:past + c], p)
        states = p.get("recurrent_states")
        past += c
        del out
        ic += 1
        a, r = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
        print(f"chunk {ic}: past={past} alloc={gb(a):.2f}GB "
              f"reserved={gb(r):.2f}GB", flush=True)
    mem("after kvarn prefill")
    from kvarn_microkld import bench_decode
    bench_decode(model, c_kvarn, ids, 16, states, "kvarn16")
    mem("after kvarn decode16")
    # Per-step real-path timing (model.forward decode, like bench_decode
    # but per step): distinguishes slow-arm from legacy fallback. Legacy
    # full-remat allocates full-context temps per step (reserved jumps);
    # the arm does not.
    import time as _time
    tok = ids[:, -1:]
    total = ((n0 + 16 + 255) // 256) * 256
    # Allocator-storm check: retry/OOM counters around one step.
    st0 = torch.cuda.memory_stats()
    print(f"pre-step: alloc_retries={st0.get('num_alloc_retries')} "
          f"ooms={st0.get('num_ooms')} "
          f"active_reserved={st0.get('active.reserved_bytes', 0) / 1e9:.2f}GB "
          f"active_alloc={st0.get('active.allocated_bytes', 0) / 1e9:.2f}GB",
          flush=True)
    for i in range(4):
        pp = {"cache": c_kvarn, "attn_mode": "flash_attn",
              "batch_shape": (1, total), "past_len": n0 + 16 + i,
              "recurrent_states": states}
        r0 = torch.cuda.memory_reserved()
        t0 = _time.perf_counter()
        lg = model.forward(tok, pp)
        states = pp.get("recurrent_states")
        torch.cuda.synchronize()
        dt = (_time.perf_counter() - t0) * 1e3
        dr = (torch.cuda.memory_reserved() - r0) / 1e9
        tok = lg.argmax(dim=-1)[:, -1:]
        del lg
        print(f"real step {i}: {dt:.0f}ms dReserved={dr:+.2f}GB",
              flush=True)
    st1 = torch.cuda.memory_stats()
    print(f"post-steps: alloc_retries={st1.get('num_alloc_retries')} "
          f"ooms={st1.get('num_ooms')}", flush=True)
    # Kineto on one more step: attribute the 6.6s definitively.
    pp = {"cache": c_kvarn, "attn_mode": "flash_attn",
          "batch_shape": (1, total), "past_len": n0 + 20,
          "recurrent_states": states}
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True) as prof:
        lg = model.forward(tok, pp)
        torch.cuda.synchronize()
        del lg
    print(prof.key_averages().table(sort_by="self_device_time_total",
                                    row_limit=25),
          flush=True)
    print(prof.key_averages(group_by_input_shape=True).table(
        sort_by="self_device_time_total", row_limit=15), flush=True)
    print("SURVIVED", flush=True)


if __name__ == "__main__":
    main()
