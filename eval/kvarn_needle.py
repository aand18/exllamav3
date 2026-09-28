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

FILLER = (
    "The municipal archive maintains records of every charter, survey, and "
    "ordinance filed since the founding of the riverside settlement. Clerks "
    "copy each entry by hand into bound ledgers, which are then shelved by "
    "year and district. Visitors may consult the reading room on weekdays. "
)

NEEDLE = "The secret passcode for the midnight launch is BLUEBERRY-7429."
QUESTION = "Question: What is the secret passcode for the midnight launch? Reply with only the passcode and nothing else."
SMOKE_Q = "What is the capital of France? Reply with only the city name."


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
    text = tokenizer.decode(ids)
    return text + "\n\n" + QUESTION, "BLUEBERRY-7429"


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
    parser.add_argument("-mt", "--max_tokens", type=int, default=64)
    parser.add_argument("-ntok", "--ntok", type=int, default=12288)
    parser.add_argument("-budget", "--budget", type=float, default=840,
                        help="Stop gracefully after this many seconds")
    parser.add_argument("-fresh", "--fresh", action="store_true",
                        help="Fresh generator + clean KVarN slots per item")
    args = parser.parse_args()

    import torch
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
        hit = (key.lower() in completion.lower()) if completion else False
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
