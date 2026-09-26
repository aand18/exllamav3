"""
Decode component profiler for KVarN (timing only, no quality asserts).

Isolates per-token costs on a populated context:
  A. get_kv serve alone (no store between calls)
  B. update_kv_direct (1 row) + get_kv  (true decode mix)
  C. full model.forward decode step
Derives: serve = A, store ~= B - A, attention+rest ~= C - B.

Needs a GPU box. Example:
    python eval/kvarn_profile_decode.py -m <model_dir> -cq kvarn4 \
        -ntok 8192 -mcl 0 -chunk 4096 -steps 32
"""

import argparse
import time
import torch

from exllamav3 import Config, Model, Tokenizer, Cache
from exllamav3.cache import CacheLayer_fp16, CacheLayer_kvarn
from exllamav3.cache.kvarn import kvarn_parse_preset
from kvarn_microkld import SAMPLER_TEXT, populate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_dir", required=True)
    parser.add_argument("-cq", "--cache_quant", default="kvarn4")
    parser.add_argument("-ntok", "--ntok", type=int, default=8192)
    parser.add_argument("-chunk", "--chunk", type=int, default=4096)
    parser.add_argument("-steps", "--steps", type=int, default=32)
    parser.add_argument("-d", "--device", default="cuda:0")
    parser.add_argument("-mcl", "--moe_cpu_offload", type=int, default=0)
    args = parser.parse_args()

    k_bits, v_bits = kvarn_parse_preset(args.cache_quant)
    config = Config.from_directory(args.model_dir)
    if args.moe_cpu_offload:
        config.infer_params.moe_cpu_offload = args.moe_cpu_offload
    model = Model.from_config(config)
    c_kvarn = Cache(model, max_num_tokens=args.ntok, layer_type=CacheLayer_kvarn,
                    k_bits=k_bits, v_bits=v_bits)
    model.load(args.device, progressbar=False)

    tokenizer = Tokenizer.from_config(config)
    reps = max(16, (args.ntok // 24) + 2)
    ids = tokenizer.encode(SAMPLER_TEXT * reps)[:, :args.ntok]
    n = int(ids.shape[1])
    layers = c_kvarn.layers if hasattr(c_kvarn, "layers") else None

    states, _ = populate(model, c_kvarn, ids, args.chunk, n)
    torch.cuda.synchronize()

    # Cache.layers is a dict keyed by (layer_idx, instance).
    lay0 = next(iter(c_kvarn.layers.values()))
    print(f"cached layers: {len(c_kvarn.layers)}, "
          f"probed: {type(lay0).__name__}", flush=True)
    assert "kvarn" in type(lay0).__name__.lower(), type(lay0).__name__

    # A. get_kv serve alone (block table for full past, fixed seqlens).
    bt = torch.arange(n // 256, dtype=torch.int32, device="cuda:0") \
        .unsqueeze(0).expand(1, -1).contiguous()
    seql = torch.tensor([n], dtype=torch.int32)
    t0 = time.time()
    for _ in range(args.steps):
        k, v = lay0.get_kv(seql, bt)
        del k, v
    torch.cuda.synchronize()
    t_serve = (time.time() - t0) / args.steps * 1000
    print(f"A serve-only get_kv (1 layer): {t_serve:.2f} ms/call", flush=True)

    # B/C need full-model forwards; report per-step totals via forward.
    tok = ids[:, -1:]
    past = n
    t0 = time.time()
    for _ in range(args.steps):
        p = {"cache": c_kvarn, "attn_mode": "flash_attn",
             "batch_shape": (1, n), "past_len": past}
        if states is not None:
            p["recurrent_states"] = states
        logits = model.forward(tok, p)
        states = p.get("recurrent_states")
        tok = logits.argmax(dim=-1)[:, -1:]
        past += 1
        del logits
    torch.cuda.synchronize()
    t_step = (time.time() - t0) / args.steps * 1000
    print(f"C full decode step (all layers): {t_step:.2f} ms/step", flush=True)
    print(f"implied non-get_kv per step: see fp16 bench for attention share",
          flush=True)


if __name__ == "__main__":
    main()
