"""
Micro-KLD: KVarN cache vs fp16 cache over one prefill on a real model.

Compares per-position next-token KLD between a KVarN-cache forward and an
fp16-cache forward of the same prompt. Two-phase (populate past, then score
a continuation at past_len>0): a single past_len=0 prefill only attends
over fresh exact projections, so scoring it measures nothing. NTOK is the
total token count and must exceed ~320 so the populated past actually seals
a 128-group; the tail floor keeps the suffix exact on both sides, so this
measures body-format difference, not suffix policy.

Needs a GPU box (the C++ extension is CUDA-only; CPU forward is unsupported).
Example:
    python eval/kvarn_microkld.py -m <model_dir> -cq kvarn4 -ntok 200
"""

import argparse
import time
import torch
import torch.nn.functional as F

from exllamav3 import Config, Model, Tokenizer, Cache
from exllamav3.cache import CacheLayer_fp16, CacheLayer_kvarn
from exllamav3.cache.kvarn import kvarn_parse_preset

SAMPLER_TEXT = ("The capital of France is Paris. It is known for the Eiffel "
                "Tower, the Louvre, and its arrondissements along the Seine. ")


def run(model, cache, ids):
    # Two-phase: a past_len=0 prefill only attends over fresh (exact)
    # projections -- the cache seals afterwards, so scoring it measures
    # nothing. Instead populate the past first, then score a continuation
    # at past_len>0, which actually attends over the sealed body.
    n = int(ids.shape[1])
    assert n > 256 + 8, \
        f"need >264 tokens to seal a group and score a continuation, got {n}"
    split = n - 64
    p1 = {"cache": cache, "attn_mode": "flash_attn",
          "batch_shape": (1, ids.shape[1]), "past_len": 0}
    model.forward(ids[:, :split], p1)
    states = p1.get("recurrent_states")
    p2 = {"cache": cache, "attn_mode": "flash_attn",
          "batch_shape": (1, ids.shape[1]), "past_len": split}
    if states is not None:
        p2["recurrent_states"] = states
    logits = model.forward(ids[:, split:], p2)
    return logits.float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_dir", required=True)
    parser.add_argument("-cq", "--cache_quant", default="kvarn4",
                        help="KVarN preset, e.g. kvarn4 or kvarn5,kvarn4")
    parser.add_argument("-ntok", "--ntok", type=int, default=200)
    parser.add_argument("-maxtok", "--max_tokens", type=int, default=0,
                        help="Cache max tokens (defaults to max(512, ntok))")
    parser.add_argument("-d", "--device", default="cuda:0")
    parser.add_argument("-mcl", "--moe_cpu_offload", type=int, default=0,
                        help="Offload first N block-sparse MoE layers to CPU "
                             "(e.g. 38 for Qwen3.8-Flash-Next 3.05bpw on 24GB VRAM)")
    args = parser.parse_args()

    k_bits, v_bits = kvarn_parse_preset(args.cache_quant)

    config = Config.from_directory(args.model_dir)
    if args.moe_cpu_offload:
        config.infer_params.moe_cpu_offload = args.moe_cpu_offload
    model = Model.from_config(config)
    # Caches must be attached before load: the loader allocates cache and
    # recurrent-state tensors for attached caches only (see model_init.init).
    # Creating them after load leaves recurrent state on meta, which fails
    # hybrid (GDN) forwards with "conv_state is on meta".
    max_tok = args.max_tokens or max(512, args.ntok)
    assert max_tok >= args.ntok, f"max_tokens {max_tok} < ntok {args.ntok}"
    c_fp16 = Cache(model, max_num_tokens=max_tok, layer_type=CacheLayer_fp16)
    c_kvarn = Cache(model, max_num_tokens=max_tok, layer_type=CacheLayer_kvarn,
                    k_bits=k_bits, v_bits=v_bits)
    t0 = time.time()
    model.load(args.device, progressbar=False)
    print(f"load ok ({time.time() - t0:.1f}s)", flush=True)

    tokenizer = Tokenizer.from_config(config)
    reps = max(16, (args.ntok // 24) + 2)
    ids = tokenizer.encode(SAMPLER_TEXT * reps)[:, :args.ntok]
    print("tokens:", tuple(ids.shape), flush=True)

    t0 = time.time()
    l_fp16 = run(model, c_fp16, ids)
    print(f"fp16 prefill: {time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    l_kvarn = run(model, c_kvarn, ids)
    print(f"kvarn{k_bits}/kvarn{v_bits} prefill: {time.time() - t0:.1f}s",
          flush=True)

    p = F.log_softmax(l_kvarn, dim=-1)
    q = F.log_softmax(l_fp16, dim=-1)
    kld = (q.exp() * (q - p)).sum(-1).squeeze(0)
    print(f"KLD kvarn{k_bits}/kvarn{v_bits} vs fp16-cache "
          f"over {l_kvarn.shape[1]} scored continuation positions:", flush=True)
    print(f"  median {kld.median().item():.9f}  mean {kld.mean().item():.9f}  "
          f"max {kld.max().item():.9f}", flush=True)
    agree = (l_kvarn.argmax(-1) == l_fp16.argmax(-1)).float().mean().item()
    print(f"  same-top {agree * 100:.2f}%", flush=True)


if __name__ == "__main__":
    main()
