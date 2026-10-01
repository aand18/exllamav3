"""Quick reasoning/long-ctx smoke + needle benchmark (minutes, not hours).

Item 0 is always the smoke test ("capital of France" -> Paris); a smoke
failure aborts immediately (fail fast: the model path is broken, not the
cache). Then needle-in-haystack items at several depths exercise longctx
KV quality: filler + one passcode sentence, ask for the passcode back.

Usage: python eval/kvarn_needle.py -m <model> -cq kvarn4 [-ntok 12288]
Compares presets across runs (kvarn4 vs kvarn5 anchor). Greedy decoding
(temp 0) for determinism. Per-item + total timing; --budget (default
840s) stops gracefully with partial results. One fresh generator (+
clean KVarN slots with --fresh) per item: single-sequence isolation.
"""
from __future__ import annotations
import os, sys, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
except Exception:
    pass
from exllamav3 import model_init, Generator, Job
import argparse
import torch

FILLER = (
    "The municipal archive maintains records of every charter, survey, and "
    "ordinance filed since the founding of the riverside settlement. Clerks "
    "copy each entry by hand into bound ledgers, which are then shelved by "
    "year and district. Visitors may consult the reading room on weekdays. "
)

NEEDLE = "The secret passcode for the midnight launch is BLUEBERRY-7429."
QUESTION = "Question: What is the secret passcode for the midnight launch? Reply with only the passcode and nothing else."
SMOKE_Q = "What is the capital of France? Reply with only the city name."

# Multi-needle extension: interference + recency. Same filler/question
# style; keys are distinct per mission so substring checks are exact.
NEEDLES_MULTI = (
    ("The secret passcode for the midnight launch is BLUEBERRY-7429.",
     "midnight launch", "BLUEBERRY-7429"),
    ("The secret passcode for the dawn patrol is COPPER-3110.",
     "dawn patrol", "COPPER-3110"),
    ("The secret passcode for the harbor manifest is JUNIPER-5567.",
     "harbor manifest", "JUNIPER-5567"),
)
QUESTION_MULTI = (
    "Question: What are the secret passcodes for the midnight launch, "
    "the dawn patrol, and the harbor manifest? Reply with only the "
    "three passcodes separated by commas and nothing else.")
NEEDLE_OLD = "The secret passcode for the midnight launch is BLUEBERRY-7429."
NEEDLE_NEW = "Note: the passcode was changed. The secret passcode for " \
    "the midnight launch is now RASPBERRY-1083."


def build_haystack(tokenizer, ntok, depth, seed=7):
    base = FILLER * 40
    fids = tokenizer.encode(base)[0].tolist()
    needle_ids = tokenizer.encode(" " + NEEDLE + " ")[0].tolist()
    q_ids = tokenizer.encode(" " + QUESTION)[0].tolist()
    budget = max(512, ntok - len(needle_ids) - len(q_ids))
    rep = (budget + len(fids) - 1) // len(fids)
    hay = (fids * rep)[:budget]
    at = min(int(len(hay) * depth), len(hay))
    ids = hay[:at] + needle_ids + hay[at:]
    text = tokenizer.decode(torch.tensor(ids))
    return text + "\n\n" + QUESTION, "BLUEBERRY-7429"


def build_haystack_multi(tokenizer, ntok, seed=7):
    # Three needles at 5/50/95% depths, one conjunction question.
    # Insert deepest-first so earlier insertions don't shift planted
    # positions. Returns (prompt, [keys]).
    base = FILLER * 40
    fids = tokenizer.encode(base)[0].tolist()
    q_ids = tokenizer.encode(" " + QUESTION_MULTI)[0].tolist()
    needles = [(0.05, NEEDLES_MULTI[0]), (0.5, NEEDLES_MULTI[1]),
               (0.95, NEEDLES_MULTI[2])]
    total_needle = sum(len(tokenizer.encode(" " + t + " ")[0])
                       for _, (t, _, _) in needles)
    budget = max(512, ntok - total_needle - len(q_ids))
    rep = (budget + len(fids) - 1) // len(fids)
    hay = (fids * rep)[:budget]
    for depth, (text, _, _) in sorted(needles, reverse=True):
        nids = tokenizer.encode(" " + text + " ")[0].tolist()
        at = min(int(len(hay) * depth), len(hay))
        hay = hay[:at] + nids + hay[at:]
    prompt = tokenizer.decode(torch.tensor(hay))
    return prompt + "\n\n" + QUESTION_MULTI, \
        [k for _, _, k in NEEDLES_MULTI]


def build_haystack_update(tokenizer, ntok, seed=7):
    # Recency: old code at 30%, superseding note at 70%. Hit = new code.
    base = FILLER * 40
    fids = tokenizer.encode(base)[0].tolist()
    q_ids = tokenizer.encode(" " + QUESTION)[0].tolist()
    oids = tokenizer.encode(" " + NEEDLE_OLD + " ")[0].tolist()
    nids = tokenizer.encode(" " + NEEDLE_NEW + " ")[0].tolist()
    budget = max(512, ntok - len(oids) - len(nids) - len(q_ids))
    rep = (budget + len(fids) - 1) // len(fids)
    hay = (fids * rep)[:budget]
    at_new = min(int(len(hay) * 0.70), len(hay))
    hay = hay[:at_new] + nids + hay[at_new:]
    at_old = min(int(len(hay) * 0.30), len(hay))
    hay = hay[:at_old] + oids + hay[at_old:]
    prompt = tokenizer.decode(torch.tensor(hay))
    return prompt + "\n\n" + QUESTION, "RASPBERRY-1083"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick smoke + needle longctx probe", allow_abbrev=False)
    model_init.add_args(
        parser,
        add_sampling_args=True,
        default_cache_size=65536,
        default_sampling_args={
            "temperature": 0.0,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "penalty_range": 1024,
            "min_p": 0.0,
            "top_k": 0,
            "top_p": 1.0,
            "adaptive_target": 1.0,
            "adaptive_decay": 0.9,
        },
    )
    parser.add_argument("-o", "--output", type=str, default=None)
    # 256 (was 64): single-needle items EOS in ~30-64 tok, but the
    # multi/update items reason first (preamble) and need headroom past
    # it; EOS-capped items cost nothing extra, rambles cost ~10s more.
    parser.add_argument("-mt", "--max_tokens", type=int, default=256)
    parser.add_argument("-ntok", "--ntok", type=int, default=12288)
    parser.add_argument("-budget", "--budget", type=float, default=840,
                        help="Stop gracefully after this many seconds")
    parser.add_argument("-fresh", "--fresh", action="store_true",
                        help="Fresh generator + clean KVarN slots per item")
    parser.add_argument("-classic", "--classic", action="store_true",
                        help="Only the original 4 items (smoke + 3 depths)")
    parser.add_argument("-no-multi", "--no_multi", action="store_true",
                        help="Skip the multi-needle conjunction item")
    parser.add_argument("-no-update", "--no_update", action="store_true",
                        help="Skip the recency-update item")
    args = parser.parse_args()

    t_all = time.time()
    model, config, cache, tokenizer = model_init.init(args)
    sampler = model_init.get_arg_sampler(args)

    def fresh_generator():
        gen = Generator(model=model, cache=cache, max_batch_size=1,
                        tokenizer=tokenizer, show_visualizer=False)
        if args.fresh:
            from exllamav3.cache import CacheLayer_kvarn
            for layer in list(cache.layers.values()):
                if isinstance(layer, CacheLayer_kvarn) and \
                        getattr(layer, "device", None) is not None:
                    dev = layer.device
                    layer.free()
                    layer.alloc(dev)
        return gen

    items = [("smoke", SMOKE_Q, "paris")]
    for d in (0.05, 0.5, 0.95):
        prompt, key = build_haystack(tokenizer, args.ntok, d)
        items.append((f"needle@{d}", prompt, key))
    if not args.classic:
        if not args.no_multi:
            prompt, keys = build_haystack_multi(tokenizer, args.ntok)
            items.append(("needlemulti", prompt, keys))
        if not args.no_update:
            prompt, key = build_haystack_update(tokenizer, args.ntok)
            items.append(("needleupdate", prompt, key))

    results = []
    ok = True
    for name, prompt, key in items:
        if time.time() - t_all > args.budget:
            print(f" -- TIME BUDGET ({args.budget:.0f}s) exceeded, "
                  f"stopping with {len(results)}/{len(items)} items")
            break
        ids = tokenizer.hf_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True)
        n_prompt = int(ids.shape[1])
        gen = fresh_generator()
        t0 = time.time()
        gen.enqueue(Job(input_ids=ids, max_new_tokens=args.max_tokens,
                        stop_conditions=config.eos_token_id_list,
                        sampler=sampler, identifier=name,
                        max_rq_tokens=512, stop_on_loop=(300, 3)))
        completion, error = "", None
        while gen.num_remaining_jobs():
            for r in gen.iterate():
                if r.get("eos"):
                    if "error" in r:
                        error = repr(r["error"])
                    else:
                        completion = r.get("full_completion", "")
        dt = time.time() - t0
        n_gen = int(tokenizer.encode(completion).shape[1]) if completion else 0
        # Key may be a single string or a list (multi-needle: all must hit).
        keys = [key] if isinstance(key, str) else list(key)
        hit = bool(completion) and all(
            k.lower() in completion.lower() for k in keys)
        status = "HIT" if hit else ("ERROR" if error else "MISS")
        print(f" [{name:12s}] {status:5s} prompt={n_prompt:6d}tok "
              f"gen={n_gen:4d}tok {dt:6.1f}s "
              f"{(n_prompt + n_gen) / max(dt, 1e-3):7.1f}tok/s :: "
              f"{completion.strip()[:120]!r}", flush=True)
        results.append({"item": name, "hit": hit, "error": error,
                        "seconds": round(dt, 1),
                        "completion": completion.strip()[:500]})
        if name == "smoke" and not hit:
            print(" !! SMOKE FAILED -- model path broken, aborting "
                  "(not a cache verdict)")
            ok = False
            break

    n_hit = sum(1 for r in results if r["hit"])
    dt_all = time.time() - t_all
    print(f" -- NEEDLE RESULT: {n_hit}/{len(results)} hits "
          f"in {dt_all:.0f}s total (budget {args.budget:.0f}s)")
    if args.output:
        import json
        with open(args.output, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f" -- Responses written to {args.output}")
    sys.exit(0 if ok else 2)
