#!/usr/bin/env python3
"""Multi-key needle retrieval at length, for cache-precision questions.

The `longctx` suite asks eleven questions about a long document, and every cache mode answers all
eleven: a probe that cannot fail cannot separate Q2 from FP16. RULER's own tables say the same thing
about single-needle retrieval, and its multi-key variant is where models start to lose facts.

This runner is that variant, small enough to live in this repo: one haystack of a chosen length, K
unique "the access code for <NAME> is <CODE>" statements at hashed depths, and one prompt that asks
for every code. The answer is scored by machine (`bench/needle_facts.py`), so a run is a number out
of K instead of a reader's opinion.

Same model, cache and drafting configuration as every other runner: everything comes from
model_init, so an arm is the same argv as the corpus and longctx arms with a different runner.

Usage (through the harness, which supplies the regime):
  run_pass.ps1 -Tag t15-q2-needles -Runner needles -CacheSize 262144 -Extra "-nl 262144 -nk 20"

Machine-readable output, parsed by run_pass.ps1 into results/needles.csv:
  NEEDLE name=<NAME> code=<CODE>      one per inserted fact, so the scorer knows the truth
  NEEDLE_DONE keys=<K> prompt_tokens=<N> tests=<T>
"""

import sys, os, random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from exllamav3 import Config, Model, Cache, Tokenizer, model_init, Generator, Job, GreedySampler
import torch

# Invented, unambiguous, and absent from the haystack: a name the model has never seen cannot be
# recalled from its weights, so a hit proves the cache carried it.
NAMES = [
    "Zorbath", "Quellin", "Mavren", "Tolwick", "Drasko", "Pellow", "Kestrin", "Yorval",
    "Brindel", "Ostmund", "Vantix", "Harrowg", "Nimbley", "Craddock", "Fennwell", "Greyholt",
    "Ilvarin", "Jorund", "Kelvane", "Lormund", "Marrowby", "Norville", "Oakhurst", "Pemberly",
    "Quarrie", "Ravensby", "Silvane", "Thornby", "Ulverston", "Vexley", "Wrenfield", "Yardley",
]


def load_source(texts_dir: str) -> str:
    path = os.path.join(texts_dir, "pride_prejudice_mod.txt")
    with open(path, encoding="utf-8") as f:
        return f.read()


def build_haystack(source: str, tokenizer, target_tokens: int) -> str:
    """Repeat the source until the token count reaches the target, then trim to it."""
    text = source
    while token_count(tokenizer, text) < target_tokens:
        text += "\n\n" + source
    # Trim by characters until the encoded length is just under the target. Exactness is not needed:
    # the job reports the real prompt length, and the runner prints it.
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if token_count(tokenizer, text[:mid]) <= target_tokens:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo]


def token_count(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_bos = False)
    return int(ids.shape[-1]) if hasattr(ids, "shape") else len(ids)


def insert_needles(text: str, keys: int, seed: int) -> tuple:
    """Spread K statements through the haystack at hashed depths, and return the pairs."""
    rng = random.Random(seed)
    names = NAMES[:keys]
    codes = []
    used = set()
    for _ in names:
        while True:
            code = f"{rng.randrange(10000, 99999)}"
            if code not in used:
                used.add(code)
                break
        codes.append(code)
    # Depth i lands inside its own slot so the needles never collide, with a jitter that changes per
    # seed: a cache that loses a middle band and a cache that loses a random band are different bugs.
    out, cursor = [], 0
    cuts = []
    for i in range(keys):
        frac = (i + 0.1 + 0.8 * rng.random()) / keys
        cuts.append(int(frac * len(text)))
    cuts.sort()
    for pos, name, code in zip(cuts, names, codes):
        out.append(text[cursor:pos])
        out.append(f"\n\nThe access code for {name} is {code}.\n\n")
        cursor = pos
    out.append(text[cursor:])
    return "".join(out), list(zip(names, codes))


def prompt_for(haystack: str, names: list) -> str:
    listing = "\n".join(f"- {n}" for n in names)
    return (
        "Read the document below. It contains statements of the form "
        "\"The access code for NAME is CODE.\"\n\n"
        "<DOCUMENT>\n" + haystack + "\n</DOCUMENT>\n\n"
        "Give the access code for each of these names:\n" + listing + "\n\n"
        "Answer with one line per name, in the form NAME: CODE. Do not explain.\n"
    )


@torch.inference_mode()
def main(args):
    model, config, cache, tokenizer, draft_model, draft_config, draft_cache = model_init.init(args)
    generator = Generator(
        model, cache, tokenizer,
        draft_model = draft_model,
        draft_cache = draft_cache,
        max_chunk_size = args.chunk_size,
        cpu_cache_size = int(args.cpu_cache_size * 1024 ** 3),
        recurrent_cache_size = int(args.recurrent_cache_size * 1024 ** 3),
    )
    texts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_texts")
    source = load_source(texts_dir)

    jobs, total = [], 0
    for t in range(args.tests):
        seed = args.seed + t
        haystack = build_haystack(source, tokenizer, args.length)
        body, pairs = insert_needles(haystack, args.keys, seed)
        chat = [{"role": "user", "content": prompt_for(body, [n for n, _ in pairs])}]
        # No thinking: the answer is a list of codes, and a reasoning trace would spend the token
        # budget before the list and add numbers to the text the scorer has to read.
        input_ids = tokenizer.hf_chat_template(chat, add_generation_prompt = True, enable_thinking = False)
        job = Job(
            input_ids = input_ids,
            max_new_tokens = args.max_new_tokens,
            stop_conditions = config.eos_token_id_list,
            sampler = GreedySampler(),
        )
        jobs.append((t, job, pairs, int(input_ids.shape[-1])))
        total += len(pairs)

    generator.enqueue([job for _, job, _, _ in jobs])
    while generator.num_remaining_jobs():
        generator.iterate()

    for t, job, pairs, plen in jobs:
        out = (job.full_completion or "").strip()
        hits = sum(1 for _, code in pairs if code in out)
        print(f"\n--- test {t + 1}/{args.tests}: {len(pairs)} keys, {hits} codes present, prompt {plen} tokens ---")
        print(out[:4000])
        for name, code in pairs:
            print(f"NEEDLE name={name} code={code}")
    print(f"NEEDLE_DONE keys={total} prompt_tokens={jobs[-1][3]} tests={args.tests}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev = False)
    model_init.add_args(
        parser,
        add_draft_model_args = True,
        default_cache_size = 262144,
        default_autosplit_max_batch_size = 2,
    )
    parser.add_argument("-nl", "--length", type = int, default = 131072,
                        help = "haystack length in tokens (default 131072)")
    parser.add_argument("-nk", "--keys", type = int, default = 20,
                        help = "number of inserted access codes (default 20, max 32)")
    parser.add_argument("-ns", "--seed", type = int, default = 1,
                        help = "seed for depths and codes (default 1)")
    parser.add_argument("-nt", "--tests", type = int, default = 1,
                        help = "number of haystacks (default 1)")
    parser.add_argument("-ntok", "--max_new_tokens", type = int, default = 400,
                        help = "generation budget for the answer (default 400)")
    args = parser.parse_args()
    args.keys = max(1, min(args.keys, len(NAMES)))
    main(args)
