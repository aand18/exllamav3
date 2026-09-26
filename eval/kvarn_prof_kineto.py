"""Kineto op-level profile of one kvarn decode step (CPU+CUDA, shapes).

Loads the model, populates ctx, warms up, then profiles 5 single-token
decode steps. Prints top ops by CUDA time and Python-side sync points.
CPU-only analysis afterwards via the exported table (no GPU needed).
Saves a chrome trace to /tmp/kvarn_step_trace.json.
"""

import argparse
import torch
from torch.profiler import profile, ProfilerActivity

from exllamav3 import Config, Model, Tokenizer, Cache
from exllamav3.cache import CacheLayer_kvarn
from exllamav3.cache.kvarn import kvarn_parse_preset
from kvarn_microkld import SAMPLER_TEXT, populate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model_dir", required=True)
    ap.add_argument("-cq", "--cache_quant", default="kvarn4")
    ap.add_argument("-ntok", "--ntok", type=int, default=2048)
    ap.add_argument("-chunk", "--chunk", type=int, default=2048)
    args = ap.parse_args()

    k_bits, v_bits = kvarn_parse_preset(args.cache_quant)
    config = Config.from_directory(args.model_dir)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=args.ntok, layer_type=CacheLayer_kvarn,
                  k_bits=k_bits, v_bits=v_bits)
    model.load("cuda:0", progressbar=False)
    tokenizer = Tokenizer.from_config(config)
    reps = max(16, (args.ntok // 24) + 2)
    ids = tokenizer.encode(SAMPLER_TEXT * reps)[:, :args.ntok]
    n = int(ids.shape[1])
    states, _ = populate(model, cache, ids, args.chunk, n)

    tok = ids[:, -1:]
    past = n
    # Warmup (compile inductor bits, settle allocator).
    for _ in range(3):
        p = {"cache": cache, "attn_mode": "flash_attn",
             "batch_shape": (1, n), "past_len": past}
        if states is not None:
            p["recurrent_states"] = states
        logits = model.forward(tok, p)
        states = p.get("recurrent_states")
        tok = logits.argmax(dim=-1)[:, -1:]
        past += 1
        del logits
    torch.cuda.synchronize()

    acts = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(activities=acts, record_shapes=True, with_stack=False) as prof:
        for _ in range(5):
            p = {"cache": cache, "attn_mode": "flash_attn",
                 "batch_shape": (1, n), "past_len": past}
            if states is not None:
                p["recurrent_states"] = states
            logits = model.forward(tok, p)
            states = p.get("recurrent_states")
            tok = logits.argmax(dim=-1)[:, -1:]
            past += 1
            del logits
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25),
          flush=True)
    prof.export_chrome_trace("/tmp/kvarn_step_trace.json")
    print("trace: /tmp/kvarn_step_trace.json", flush=True)


if __name__ == "__main__":
    main()
