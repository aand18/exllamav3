"""Feasibility probe: torch.compile over an exllamav3 decode step.

Wraps model.forward with torch.compile (reduce-overhead, static shapes,
suppress_errors=True so unsupported ops fall back to eager instead of
failing). Measures steady-state decode tok/s vs the eager baseline from
kvarn_microkld --decode. A win here means the per-op dispatch overhead
Kineto blamed is fusible; flat means the graph breaks everywhere and the
backend needs static-structure surgery first.

Needs a GPU box. Example:
    python eval/kvarn_compile_probe.py -m <model> -cq kvarn4 -ntok 512 -steps 20
"""

import argparse
import time
import torch

from exllamav3 import Config, Model, Tokenizer, Cache
from exllamav3.cache import CacheLayer_kvarn
from exllamav3.cache.kvarn import kvarn_parse_preset
from kvarn_microkld import SAMPLER_TEXT, populate, _bshape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model_dir", required=True)
    ap.add_argument("-cq", "--cache_quant", default="kvarn4")
    ap.add_argument("-ntok", "--ntok", type=int, default=512)
    ap.add_argument("-chunk", "--chunk", type=int, default=512)
    ap.add_argument("-steps", "--steps", type=int, default=20)
    ap.add_argument("-d", "--device", default="cuda:0")
    ap.add_argument("-mcl", "--moe_cpu_offload", type=int, default=0)
    args = ap.parse_args()

    torch._dynamo.config.suppress_errors = True
    k_bits, v_bits = kvarn_parse_preset(args.cache_quant)
    config = Config.from_directory(args.model_dir)
    if args.moe_cpu_offload:
        config.infer_params.moe_cpu_offload = args.moe_cpu_offload
    model = Model.from_config(config)
    max_tok = _bshape(args.ntok + args.steps + 8)
    cache = Cache(model, max_num_tokens=max_tok, layer_type=CacheLayer_kvarn,
                  k_bits=k_bits, v_bits=v_bits)
    model.load(args.device, progressbar=False)
    tokenizer = Tokenizer.from_config(config)
    reps = max(16, (args.ntok // 24) + 2)
    ids = tokenizer.encode(SAMPLER_TEXT * reps)[:, :args.ntok]
    n = int(ids.shape[1])
    bs = _bshape(n + args.steps + 8)
    states, _ = populate(model, cache, ids, args.chunk, n)

    # Default inductor (no cudagraphs): fuses elementwise chains without
    # choking on the host->device copies in the forward. reduce-overhead
    # graphs trip on those copies (cudaErrorStreamCaptureUnsupported).
    opt_forward = torch.compile(model.forward, mode="default",
                                dynamic=False)
    tok = ids[:, -1:]
    past = n
    # Warmup: first calls compile (or fall back); not timed.
    for _ in range(3):
        p = {"cache": cache, "attn_mode": "flash_attn",
             "batch_shape": (1, bs), "past_len": past}
        if states is not None:
            p["recurrent_states"] = states
        try:
            logits = opt_forward(tok, p)
        except Exception as e:  # noqa: BLE001 - report, keep easing
            print(f"compiled forward failed, eager fallback: {e}", flush=True)
            logits = model.forward(tok, p)
        states = p.get("recurrent_states")
        tok = logits.argmax(dim=-1)[:, -1:]
        past += 1
        del logits
    torch.cuda.synchronize()
    t0 = time.time()
    fallback = 0
    for _ in range(args.steps):
        p = {"cache": cache, "attn_mode": "flash_attn",
             "batch_shape": (1, bs), "past_len": past}
        if states is not None:
            p["recurrent_states"] = states
        try:
            logits = opt_forward(tok, p)
        except Exception:  # noqa: BLE001 - runtime fallback, keep timing
            fallback += 1
            logits = model.forward(tok, p)
        states = p.get("recurrent_states")
        tok = logits.argmax(dim=-1)[:, -1:]
        past += 1
        del logits
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"compiled decode: {args.steps / dt:.1f} tok/s "
          f"({args.steps} steps from {n} ctx, {fallback} fallbacks)",
          flush=True)
    try:
        from torch._dynamo.utils import counters
        print(f"dynamo counters: {dict(counters['stats'])}", flush=True)
    except Exception:  # noqa: BLE001 - stats are best-effort
        pass


if __name__ == "__main__":
    main()
