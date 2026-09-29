"""Differential long-context torture: kvarn4 vs q4 (+fp16) on adversarial corpora.

Targets OUR machinery (seals/evict/slots/retention), not general
capability: every metric compares caches against each other, so no
ground truth or judge is needed. Bit-similar parity (kvarn4 vs q4)
is the primary axis; fp16 is gospel-secondary (fits <=64k).

Corpus A (repetitive code ~32k): near-identical functions differing
in one constant each (seal/dedup/alias stress). Metric: mutual KLD
+ same-top agreement, all pairs.

Corpus B (planted markers): unique markers at 0/25/50/75% depth of
a filler, then a repeat query; teacher-forced prob of the correct
marker tokens per depth per cache. A collapse at depth 0% with
recovery later = eviction dropping live rows (our specific risk).

Usage: python eval/kvarn_torture.py -m MODEL [-ntok 32768] [-chunk 4096]
"""
import argparse
import gc
import time
import torch
import torch.nn.functional as F

from exllamav3 import Config, Model, Tokenizer, Cache
from exllamav3.cache import CacheLayer_fp16, CacheLayer_kvarn
from exllamav3.cache.quant import CacheLayer_quant
from exllamav3.cache.kvarn import kvarn_parse_preset
from kvarn_microkld import populate, _bshape

TEMPLATE = (
    "def compute_{name}(inputs):\n"
    "    total = {c0}\n"
    "    for idx, val in enumerate(inputs):\n"
    "        total += val * {c1} + {c2}\n"
    "    return total // {c3}\n\n"
)
MARKERS = ["alpha", "bravo", "charlie", "delta"]
SENTENCES = [
    "The lighthouse keeper filed seventeen reports on Tuesday.",
    "A copper kettle whistled twice above the mountain cabin.",
    "The quarterly ledger showed a surplus of forty-two crates.",
    "Purple finches nested behind the old boathouse that spring.",
]


def build_repetitive(tokenizer, target):
    import random
    rng = random.Random(7)
    names = ["alpha", "beta", "gamma", "delta", "eps", "zeta", "eta",
             "theta", "iota", "kappa", "lamb", "mu", "nu", "xi", "om",
             "pi", "rho", "sig", "tau", "ups", "phi", "chi", "psi", "omg"]
    parts = []
    i = 0
    while True:
        parts.append(TEMPLATE.format(
            name=f"{rng.choice(names)}_{i}",
            c0=rng.randint(0, 9999), c1=rng.randint(0, 999),
            c2=rng.randint(0, 9999), c3=rng.randint(1, 99)))
        i += 1
        if i % 50 == 0:
            ids = tokenizer.encode("".join(parts))
            ids = ids if ids.dim() == 2 else ids.unsqueeze(0)
            if int(ids.shape[1]) >= target:
                return ids[:, :target]
    raise AssertionError("unreachable")


def build_markers(tokenizer, target):
    filler = ("The quick brown fox jumps over the lazy dog. " * 4000)
    words = filler.split(" ")
    n = len(words)
    spots = [int(n * f) for f in (0.02, 0.27, 0.52, 0.77)]
    for pos, sent in zip(spots, SENTENCES):
        words[pos] = sent
    text = " ".join(words)
    # Query does NOT contain the sentences (else no recall is needed);
    # the reference answer (scored teacher-forced) repeats them.
    query = (" Repeat these four sentences back in order. First, "
             "the lighthouse keeper sentence:")
    answer = " " + " ".join(SENTENCES)
    ids = tokenizer.encode(text + query + answer)
    ids = ids if ids.dim() == 2 else ids.unsqueeze(0)
    a = tokenizer.encode(answer)
    a = a if a.dim() == 2 else a.unsqueeze(0)
    return ids[:, :target], int(a.shape[1])


def run_cache(model, cache, ids, chunk):
    n = int(ids.shape[1])
    split = n - 64
    states, _ = populate(model, cache, ids, chunk, split)
    p2 = {"cache": cache, "attn_mode": "flash_attn",
          "batch_shape": (1, _bshape(ids.shape[1])), "past_len": split}
    if states is not None:
        p2["recurrent_states"] = states
    logits = model.forward(ids[:, split:], p2)
    return logits.float()


def kl_stats(a, b):
    p = F.log_softmax(a, dim=-1)
    q = F.log_softmax(b, dim=-1)
    kld = (q.exp() * (q - p)).sum(-1).squeeze(0)
    agree = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
    return (kld.median().item(), kld.mean().item(), kld.max().item(),
            agree * 100.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_dir", required=True)
    parser.add_argument("-ntok", "--ntok", type=int, default=32768)
    parser.add_argument("-chunk", "--chunk", type=int, default=4096)
    parser.add_argument("-d", "--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_grad_enabled(False)

    config = Config.from_directory(args.model_dir)
    model = Model.from_config(config)
    max_tok = ((args.ntok + 511) // 256) * 256
    caches = {
        "fp16": Cache(model, max_num_tokens=max_tok,
                      layer_type=CacheLayer_fp16),
        "q4": Cache(model, max_num_tokens=max_tok,
                    layer_type=CacheLayer_quant, k_bits=4, v_bits=4),
        "kvarn4": Cache(model, max_num_tokens=max_tok,
                        layer_type=CacheLayer_kvarn,
                        k_bits=4, v_bits=4),
    }
    model.load(args.device, progressbar=False)
    print("load ok", flush=True)
    tokenizer = Tokenizer.from_config(config)

    out = {}
    ids_a = build_repetitive(tokenizer, args.ntok)
    print(f"corpus A: {tuple(ids_a.shape)}", flush=True)
    for name, cache in caches.items():
        t0 = time.time()
        out[name] = run_cache(model, cache, ids_a, args.chunk)
        print(f"  {name}: {time.time() - t0:.1f}s", flush=True)
    print("KLD mutual (median / mean / max / same-top%):", flush=True)
    for x, y in (("kvarn4", "fp16"), ("q4", "fp16"), ("kvarn4", "q4")):
        m = kl_stats(out[x], out[y])
        print(f"  {x} vs {y}: {m[0]:.6f} / {m[1]:.6f} / {m[2]:.6f} / "
              f"{m[3]:.2f}%", flush=True)
    for v in out.values():
        del v
    gc.collect()
    torch.cuda.empty_cache()

    ids_b, ans_len = build_markers(tokenizer, args.ntok)
    print(f"corpus B: {tuple(ids_b.shape)} answer_len={ans_len}",
          flush=True)
    tops = {}
    for name, cache in caches.items():
        t0 = time.time()
        logits = run_cache(model, cache, ids_b, args.chunk)
        tgt = ids_b[0, -ans_len:].to(logits.device)
        # Rank of each answer token (argsort once per position):
        # discriminates where raw probs floor at ~0 for all caches.
        # Logits cover the last 64 positions; the answer is the tail.
        off = 64 - ans_len
        order = logits.float().argsort(dim=-1, descending=True)
        ar = torch.arange(ans_len, device=logits.device)
        tops[name] = order[0, off + ar, :100].cpu()
        hit = (order[0, off + ar, :] == tgt.unsqueeze(1))
        rank = hit.float().argmax(dim=1).float().mean().item()
        top1 = float((order[0, off + ar, 0] == tgt).float().mean())
        print(f"  {name}: {time.time() - t0:.1f}s meanrank={rank:.0f} "
              f"top1={top1:.3f}", flush=True)
        del logits, order
    # Differential signal: top-100 rank-order overlap between caches
    # (faithful retention = same ordering as fp16, even when absolute
    # recall fails for every cache).
    for x, y in (("kvarn4", "fp16"), ("q4", "fp16"), ("kvarn4", "q4")):
        ov = [(set(tops[x][i].tolist()) & set(tops[y][i].tolist()))
              for i in range(ans_len)]
        j = sum(len(o) for o in ov) / (ans_len * 100.0)
        print(f"  top100-overlap {x} vs {y}: {j:.3f}", flush=True)


if __name__ == "__main__":
    main()
