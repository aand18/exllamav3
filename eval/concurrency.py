#!/usr/bin/env python3
"""Two jobs in flight at once: does batching buy throughput on this model?

Every other runner here sends one job at a time, so `max_batch_size` has never been measured on this
box. The engine batches what the cache and the slot budget allow, and the open question is what two
long prompts actually cost against one: the same wall time per request (real parallelism) or double
(serialized), and what happens to each request's own latency.

Runs N jobs at a chosen prompt length, all enqueued before the first token is generated, and reports
per-job prefill and decode times plus the aggregate. `-nj 1` is the baseline the pair is read
against; the prompts differ, so no job rides the other's prefix cache.

Machine-readable output, parsed by run_pass.ps1 into results/concurrent.csv:
  CONC jobs=<N> prompt_tokens=<min>..<max> new_tokens=<total> wall_s=<T> agg_tps=<R>
"""

import sys, os, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from exllamav3 import model_init, Generator, Job, GreedySampler
import torch

PROMPT_TAIL = "\n\n---\n\nSummarize the passage above in about 250 words. Use plain English."


def load_source(texts_dir: str) -> str:
    with open(os.path.join(texts_dir, "pride_prejudice_mod.txt"), encoding="utf-8") as f:
        return f.read()


def slice_for(source: str, tokenizer, tokens: int, offset: int) -> str:
    """A distinct window of the source with at least `tokens` tokens.

    The window wraps around the end of the text: a plain offset ran out of source for the second job
    and handed it 90 k tokens where the first had 131 k, which made the pair look like an overlap
    test when it was a size test (measured 2026-11-02).
    """
    start = offset % len(source)
    rotated = source[start:] + source[:start]
    if tokenizer_count(tokenizer, rotated) < tokens:
        rotated = rotated + source
    lo, hi = 0, len(rotated)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if tokenizer_count(tokenizer, rotated[:mid]) <= tokens:
            lo = mid
        else:
            hi = mid - 1
    return rotated[:lo]


def tokenizer_count(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_bos = False)
    return int(ids.shape[-1])


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

    jobs = []
    for i in range(args.jobs):
        # Distinct offsets, so job 2 cannot ride job 1's prefix: the comparison is about compute,
        # not about a shared cache.
        text = slice_for(source, tokenizer, args.length, offset = i * (args.length * 5))
        chat = [{"role": "user", "content": text + PROMPT_TAIL}]
        input_ids = tokenizer.hf_chat_template(chat, add_generation_prompt = True, enable_thinking = False)
        job = Job(
            input_ids = input_ids,
            max_new_tokens = args.max_new_tokens,
            stop_conditions = config.eos_token_id_list,
            sampler = GreedySampler(),
        )
        jobs.append((job, int(input_ids.shape[-1])))

    t0 = time.time()
    generator.enqueue([j for j, _ in jobs])
    while generator.num_remaining_jobs():
        generator.iterate()
    wall = time.time() - t0

    total_new = sum(j.new_tokens for j, _ in jobs)
    for i, (job, plen) in enumerate(jobs):
        print(f"job{i + 1}: prompt {plen} tokens, new {job.new_tokens} tokens, "
              f"prefill {job.time_prefill:.1f} s, decode {job.time_generate:.1f} s, "
              f"decode {job.new_tokens / job.time_generate if job.time_generate else 0:.1f} T/s")
    print(f"CONC jobs={len(jobs)} prompt_tokens={min(p for _, p in jobs)}..{max(p for _, p in jobs)} "
          f"new_tokens={total_new} wall_s={wall:.1f} agg_tps={total_new / wall if wall else 0:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev = False)
    model_init.add_args(
        parser,
        add_draft_model_args = True,
        default_cache_size = 262144,
        default_autosplit_max_batch_size = 2,
    )
    parser.add_argument("-nj", "--jobs", type = int, default = 2, help = "concurrent jobs (default 2)")
    parser.add_argument("-nl", "--length", type = int, default = 131072,
                        help = "tokens per prompt (default 131072; two of these fill a 262144 cache)")
    parser.add_argument("-ntok", "--max_new_tokens", type = int, default = 256,
                        help = "generation budget per job (default 256)")
    args = parser.parse_args()
    main(args)
