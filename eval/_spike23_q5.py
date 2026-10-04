"""Q5-rule check (untracked one-off): kvarn4-vs-fp16 KLD vs q5-vs-fp16
KLD on the same prompt. Reuses kvarn_microkld.run() so the scoring
forward is identical; only the ref cache differs (CacheLayer_quant
5,5 vs fp16). Frees each cache after its logits are saved (tiny),
so peak is one cache at a time. Verdict: kvarn4 mean+max <= q5?
"""
import argparse
import gc
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kvarn_microkld import run, SAMPLER_TEXT, _peak_gb
from exllamav3 import Config, Model, Tokenizer, Cache
from exllamav3.cache import CacheLayer_fp16, CacheLayer_kvarn
from exllamav3.cache.quant import CacheLayer_quant
from exllamav3.cache.kvarn import kvarn_parse_preset


def _free_cache(c):
    for lay in c.layers.values():
        free = getattr(lay, "free", None)
        if callable(free):
            free()
    del c
    gc.collect()
    torch.cuda.empty_cache()


def _kld(a_logits, b_logits, tag):
    p = F.log_softmax(a_logits, dim=-1)
    q = F.log_softmax(b_logits, dim=-1)
    kld = (q.exp() * (q - p)).sum(-1).squeeze(0)
    same = float((p.argmax(-1) == q.argmax(-1)).float().mean()) * 100.0
    print(f"KLD {tag} over {a_logits.shape[1]} positions:", flush=True)
    print(f"  median {kld.median().item():.6f}  "
          f"mean {kld.mean().item():.6f}  max {kld.max().item():.6f}",
          flush=True)
    print(f"  same-top {same:.2f}%", flush=True)
    return float(kld.mean()), float(kld.max()), same


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model_dir", required=True)
    ap.add_argument("-cq", "--cache_quant", default="kvarn4")
    ap.add_argument("-ntok", "--ntok", type=int, default=2048)
    ap.add_argument("-chunk", "--chunk", type=int, default=2048)
    ap.add_argument("-mcl", "--moe_cpu_offload", type=int, default=0)
    args = ap.parse_args()
    k_bits, v_bits = kvarn_parse_preset(args.cache_quant)
    torch.set_grad_enabled(False)
    config = Config.from_directory(args.model_dir)
    if args.moe_cpu_offload:
        config.infer_params.moe_cpu_offload = args.moe_cpu_offload
    model = Model.from_config(config)
    max_tok = ((args.ntok + 255) // 256) * 256
    c_fp16 = Cache(model, max_num_tokens=max_tok,
                   layer_type=CacheLayer_fp16)
    c_q5 = Cache(model, max_num_tokens=max_tok,
                 layer_type=CacheLayer_quant, k_bits=5, v_bits=5)
    c_kv = Cache(model, max_num_tokens=max_tok,
                 layer_type=CacheLayer_kvarn, k_bits=k_bits, v_bits=v_bits)
    t0 = time.time()
    model.load("cuda:0", progressbar=False)
    print(f"load ok ({time.time() - t0:.1f}s)", flush=True)
    tokenizer = Tokenizer.from_config(config)
    reps = max(16, (args.ntok // 24) + 2)
    ids = tokenizer.encode(SAMPLER_TEXT * reps)[:, :args.ntok]
    print("tokens:", tuple(ids.shape), flush=True)

    torch.cuda.reset_peak_memory_stats()
    l_fp16, _ = run(model, c_fp16, ids, args.chunk)
    print(f"fp16 prefill peak {_peak_gb():.1f}GB", flush=True)
    _free_cache(c_fp16)
    l_q5, _ = run(model, c_q5, ids, args.chunk)
    print(f"q5 prefill peak {_peak_gb():.1f}GB", flush=True)
    _free_cache(c_q5)
    l_kv, _ = run(model, c_kv, ids, args.chunk)
    print(f"kvarn prefill peak {_peak_gb():.1f}GB", flush=True)

    m1, x1, s1 = _kld(l_q5, l_fp16, "q5 vs fp16")
    m2, x2, s2 = _kld(l_kv, l_fp16, f"kvarn{k_bits}/kvarn{v_bits} vs fp16")
    ok = (m2 <= m1) and (x2 <= x1) and (s2 == 100.0)
    print(f"RULE kvarn4<=q5: mean {m2:.6f}<={m1:.6f}, max {x2:.6f}<={x1:.6f}, "
          f"same-top {s2:.2f}% -> {'HOLDS' if ok else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
