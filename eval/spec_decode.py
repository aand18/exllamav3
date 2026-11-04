import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3 import Generator, Job, model_init, GreedySampler, TopPSampler
import exllamav3.generator.generator as exl_gen
import argparse
import torch
import json
from tabulate import tabulate

# ANSI codes
col_default = "\u001b[0m"
col_yellow = "\u001b[33;1m"
col_blue = "\u001b[34;1m"
col_magenta = "\u001b[35;1m"
col_red = "\u001b[31;1m"
col_green = "\u001b[32;1m"  # Green

prompt_files = [
    ("Trivial repetition", "trivial.json", False),
    ("Agentic, code", "agentic_code_01.json", True),
    ("Agentic, code", "agentic_code_05.json", True),
    ("Agentic, code", "agentic_code_10.json", True),
    ("Agentic, code", "agentic_code_20.json", True),
    ("Agentic, code", "agentic_code_29.json", True),
    ("Agentic, curl", "agentic_curl_05.json", True),
    ("Agentic, curl", "agentic_curl_10.json", True),
    ("Agentic, curl", "agentic_curl_15.json", True),
    ("Agentic, curl", "agentic_curl_16.json", True),
    ("Creative", "creative_01.json", False),
    ("Creative", "creative_02.json", False),
    ("Creative (reasoning)", "creative_01.json", True),
    ("Creative (reasoning)", "creative_02.json", True),
    ("Creative (reasoning)", "creative_03.json", True),
    ("Translation", "translate_01.json", False),
    ("Translation", "translate_02.json", False),
    ("Translation (reasoning)", "translate_01.json", True),
    ("Translation (reasoning)", "translate_02.json", True),
    ("Coding", "coding_01.json", False),
    ("Coding", "coding_02.json", False),
    ("Coding", "coding_03.json", False),
]


def read_prompt(filename):
    """Read one workload prompt, with tool-call arguments decoded to dicts.

    The corpus stores `tool_calls[i].function.arguments` as the JSON *string* the API sends, but
    the chat template renders an object, so the string has to be decoded before tokenizing. Doing
    it here means the untimed warm pass and the timed pass hand the template the same input: they
    used to differ, and on the `Agentic, *` workloads warm_corpus() templated a string where
    measure() templated a dict.
    """
    path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(path, "prompts", filename), "r") as f:
        data = json.load(f)

    for msg in data["messages"]:
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                func = tc.get("function", {})
                args = func.get("arguments")
                if isinstance(args, str):
                    func["arguments"] = json.loads(args)

    return data


def measure(generator, tokenizer, sampler, max_new_tokens, stats_sink = None):
    all_results = {}

    for category, filename, think in prompt_files:
        data = read_prompt(filename)

        prompt_ids = tokenizer.hf_chat_template(
            messages = data["messages"],
            add_generation_prompt = True,
            functions = data.get("functions"),
            tools = data.get("tools"),
            enable_thinking = think
        )

        job = Job(
            input_ids = prompt_ids,
            max_new_tokens = max_new_tokens,
            stop_conditions = generator.model.config.eos_token_id_list,
            sampler = sampler,
        )
        generator.enqueue(job)

        while generator.num_remaining_jobs():
            results = generator.iterate()
            for result in results:
                if result["stage"] == "error":
                    raise result["error"]
                if result["stage"] == "prefill":
                    curr_progress = result["curr_progress"]
                    max_progress = result["max_progress"]
                    print(f"Prefill: {curr_progress:,} / {max_progress:,}...")
                text = result.get("text", "")
                print(text, end = "", flush = True)
                if result.get("eos"):
                    print("\n--------")
                    break

        new_tokens = result["new_tokens"]
        gen_tps = new_tokens / result["time_generate"]
        if "accepted_draft_tokens" in result:
            dacc = result["accepted_draft_tokens"]
            drej = result["rejected_draft_tokens"]
            # Every verification round emits exactly one token that is not an accepted draft
            # token (the bonus token or the mismatch)
            rounds = new_tokens - dacc
        else:
            dacc = None
            drej = None
            rounds = None

        if category not in all_results:
            all_results[category] = []
        all_results[category].append({
            "new_tokens": new_tokens,
            "gen_tps": gen_tps,
            "dacc": dacc,
            "drej": drej,
            "rounds": rounds,
        })

        if stats_sink is not None and job.draft_stats:
            stats_sink.append({
                "category": category,
                "file": filename,
                "stats": job.draft_stats,  # (position, window, accepted) per verification round
            })

    aggregated = {}
    for category, cat_results in all_results.items():
        tps = (
            sum(m["gen_tps"] * m["new_tokens"] for m in cat_results) /
            sum(m["new_tokens"] for m in cat_results)
        )
        rounds = sum(m["rounds"] for m in cat_results if m["rounds"] is not None)
        dacc = sum(m["dacc"] for m in cat_results if m["dacc"] is not None)
        drej = sum(m["drej"] for m in cat_results if m["drej"] is not None)
        aggregated[category] = {
            "tps": tps,
            # Per verification round (including rounds that drafted nothing): mean accepted
            # draft tokens and mean drafted window. Every fed draft position is counted as
            # either accepted or rejected, so fed = dacc + drej
            "acc_len": dacc / rounds if rounds else None,
            "draft_len": (dacc + drej) / rounds if rounds else None,
            "acc_rate": dacc / (dacc + drej) if dacc + drej else None,
        }
    return aggregated


def reset_arm(generator):
    """Release everything one arm allocated before the next arm starts.

    A Generator owns a RecurrentCache whose host pool holds the GDN checkpoints for the arm's
    speculative passes, plus a thread pool and a reference cycle (Generator -> recurrent_cache ->
    page table -> Generator). Building a second Generator without clearing the first kept ~700 MB
    of host RAM alive per arm - the growth this session first misread as an engine leak
    (critique C6). Closing and collecting restores the process to its pre-arm footprint.
    """
    import gc
    if generator is None:
        return
    try:
        rc = getattr(generator, "recurrent_cache", None)
        if rc is not None and hasattr(rc, "clear"):
            rc.clear()
    except Exception:
        pass
    try:
        from exllamav3.cache.recurrent import host_pool
        host_pool.release()
    except Exception:
        pass
    try:
        generator.close()
    except Exception:
        pass
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def warm_corpus(generator, tokenizer, tokens = 64):
    """Touch every workload once before timing.

    The PLE ngram table is streamed from disk when `ngram_ram` is off, so whoever runs first pays
    the cold reads for exactly these prompt/generation streams. Timing that run as a baseline is
    what produced the 6.99 t/s outlier in `mtp-dnt3-368` (critique C3). 64 tokens per workload is
    enough to fault the rows in; nothing here is measured.
    """
    for category, filename, think in prompt_files:
        data = read_prompt(filename)
        prompt_ids = tokenizer.hf_chat_template(
            messages = data["messages"], add_generation_prompt = True,
            functions = data.get("functions"), tools = data.get("tools"),
            enable_thinking = think,
        )
        job = Job(input_ids = prompt_ids, max_new_tokens = tokens,
                  stop_conditions = generator.model.config.eos_token_id_list,
                  sampler = GreedySampler())
        generator.enqueue(job)
        while generator.num_remaining_jobs():
            for result in generator.iterate():
                if result["stage"] == "error":
                    raise result["error"]


def arm_line(label, aggregated):
    """One machine-readable line per arm: the harness parses these into results/arms.csv."""
    tps = [v["tps"] for v in aggregated.values() if v.get("tps")]
    accs = [v["acc_len"] for v in aggregated.values() if v.get("acc_len") is not None]
    drafts = [v["draft_len"] for v in aggregated.values() if v.get("draft_len") is not None]
    if not tps:
        return
    acc = sum(accs) / len(accs) if accs else 0.0
    draft = sum(drafts) / len(drafts) if drafts else 0.0
    print(f"ARM {label} tok_s={sum(tps) / len(tps):.2f} acc={acc:.2f}/{draft:.2f} rounds={len(tps)}")


def plot_stats(stats_sink):
    """
    Show accepted draft tokens per verification round vs generation position, with the chosen window
    overlaid, one panel per trace.
    """
    import matplotlib.pyplot as plt

    n = len(stats_sink)
    fig, axes = plt.subplots(n, 1, figsize = (14, min(3.2 * n, 18)), squeeze = False)
    axes = axes[:, 0]

    for ax, trace in zip(axes, stats_sink):
        stats = trace["stats"]
        pos      = [s[0] for s in stats]
        window   = [s[1] for s in stats]
        accepted = [s[2] for s in stats]

        ax.scatter(pos, accepted, s = 6, color = "tab:blue", alpha = 0.45, label = "accepted / window", zorder = 3)
        ax.step(pos, window, where = "post", color = "tab:red", lw = 1.0, alpha = 0.8, label = "window (chosen)")
        mean_acc = sum(accepted) / len(accepted) if accepted else 0.0
        mean_win = sum(window) / len(window) if window else 0.0
        ax.set_title(f"{trace['category']} - {trace['file']} - "
                     f"mean accepted {mean_acc:.2f} / window {mean_win:.2f} per round", fontsize = 10)
        ax.set_ylabel("draft tokens")
        ax.set_ylim(-0.4, max(window) + 1.4)
        ax.grid(alpha = 0.25)
        ax.legend(loc = "upper right", fontsize = 8)

    axes[-1].set_xlabel("generation position (tokens)")
    fig.tight_layout()
    plt.show()


def print_stats(stats_sink):
    headers = [
        f"{col_yellow}from_pos{col_default}",
        f"{col_yellow}window{col_default}",
        f"{col_yellow}accepted{col_default}",
    ]
    for trace in stats_sink:
        print(f"\n{trace['category']} - {trace['file']}\n")
        rows = [[r[0] - r[2], r[1], r[2]] for r in trace["stats"]]
        print(tabulate(rows, headers = headers, tablefmt = "pipe", colalign = ("right",) * len(headers)))


def count_arms(args):
    """How many arms this run will measure, following the arm blocks in main()."""
    arms = 0 if args.no_baseline else 1
    if args.s_ngram_match_min:
        arms += 1
    if args.draft_model_dir:
        arms += 4 if args.readback_ab else 1
    return arms


def dynamic_placement(model, config):
    """Whether expert placement can move during a generation.

    Placement is decided at load: BlockSparseMLP_CPU.load_cpu_split() sets `_split_dynamic` from
    EXL3_MOE_CPU_SWAP and registers the module in `config.infer_params.moe_cpu_swap_modules` only
    when placement is dynamic, so a static split (EXL3_MOE_CPU_SWAP=0) registers nothing. Every
    attribute is guarded: a run with no CPU split has no registry at all.
    """
    reg = getattr(getattr(config, "infer_params", None), "moe_cpu_swap_modules", None) or []
    return any(getattr(m, "_split_dynamic", False)
               for m in list(reg) + list(getattr(model, "modules", None) or []))


def reset_placement(model):
    """Put every expert of `model` back where it was loaded, so the next arm starts from the same
    placement the last one did. Returns the number of slots whose weights had to be moved back.

    The defect: this runner measures two or more arms in one process, and with dynamic expert
    placement on (EXL3_MOE_CPU_SWAP) the first arm promotes and demotes experts as it generates. The
    second arm then starts from a *different* placement - more or fewer experts of each layer on the
    card - so the arm-to-arm ratio compares two placements and not two drafters, which is what the
    refusal below is about. Resetting between the arms removes that confound; it cannot remove the
    one the refusal names (an arm that moves experts mid-generation generates different tokens), so a
    multi-arm run with dynamic placement still needs -swap_ok.

    The engine owns the mechanics (block_sparse_mlp_cpu.reset_placement): the map the load path
    built, the weights in the slots it names, the transfers in flight and the promote bookkeeping.
    Imported here rather than at module scope because the offline gates in bench/test_gates.py stub
    the engine to import this file without a GPU stack.

    main() calls it at every arm boundary, and only when the run has more than one arm: a single-arm
    run has nothing to compare, and there the placement the warm pass settled is part of what the
    arm measures.
    """
    from exllamav3.modules.block_sparse_mlp_cpu import reset_placement as engine_reset_placement
    moved = engine_reset_placement(model)
    if moved:
        print(f" -- expert placement reset: {moved} slot(s) returned to their load-time map")
    return moved


def swap_refusal(arms: int, dynamic: bool) -> str | None:
    """Why a multi-arm comparison must not run with dynamic expert placement, or None.

    A placement change re-reads experts from the checkpoint mid-stream, and the engine's own note
    in block_sparse_mlp_cpu.py says it "visibly derails a generation in progress". With it on, an
    arm that moves experts generates different tokens from one that does not - measured in
    wiki/plans/v165-reports/artifacts/logs/perf-val-spec-20261029-233352.log:61 vs :103, where
    Baseline and MTP (greedy) produced different greedy text - so the arm-to-arm ratio compares two
    different workloads. One arm cannot be miscompared, and static placement keeps every arm on the
    same tokens.

    reset_placement() between the arms fixes the other half of the problem (the second arm used to
    start from whatever placement the first one ended on), so both arms now start from the load-time
    placement. The tokens still diverge once an arm's own placement moves, so this stays a refusal:
    the reset makes -swap_ok's exploration comparable, not the ratio canonical."""
    if arms <= 1 or not dynamic:
        return None
    return (f"dynamic expert placement is active (EXL3_MOE_CPU_SWAP) and this run compares {arms} "
            f"arms. Placement moves mid-generation, so they do not generate the same tokens and the "
            f"arm-to-arm ratio is meaningless.\n"
            f"  remedy: re-run with EXL3_MOE_CPU_SWAP=0 for static placement.\n"
            f"  deliberate exploration only: pass -swap_ok to measure it anyway (placement is reset "
            f"between arms, so both start from the same experts in the same slots).")


@torch.inference_mode()
def main(args):
    if args.ngram_corpus and args.s_ngram_match_min <= 0:
        raise ValueError("-ngram_corpus requires -ngram_min > 0")
    model, config, cache, tokenizer, draft_model, draft_config, draft_cache = model_init.init(
        args,
        min_draft_len = args.s_ngram_draft_length
    )

    # Hard gate: a ratio between arms only means something when every arm generates the same
    # tokens. Placement is fixed at load, so this is the first moment it can be checked - and
    # checking it beats reporting a number that measures two different generations. Between the arms
    # below, reset_placement() puts the placement back to that load-time state (see its docstring).
    arms = count_arms(args)
    refusal = swap_refusal(arms, dynamic_placement(model, config))
    if refusal:
        if args.swap_ok:
            print(f"{col_yellow}WARNING: {refusal}{col_default}", file = sys.stderr)
        else:
            print(f"{col_red}ERROR: {refusal}{col_default}", file = sys.stderr)
            sys.exit(2)
    # ... and only between arms: a single-arm run has nothing to compare, and there the placement the
    # warm pass settled is part of what the arm measures, i.e. today's number
    reset_between_arms = arms > 1

    # Optionlly limit scope
    if sw := args.single_workload:
        global prompt_files
        if sw.endswith("*"):
            sw = sw[:-1]
            prompt_files = [p for p in prompt_files if p[0].lower().startswith(sw.lower())]
        else:
            prompt_files = [p for p in prompt_files if p[0].lower() == sw.lower()]

    stats_sink = [] if (args.draft_stats or args.plot_stats or args.print_stats) else None

    # Warm-up: pay the cold ngram/disk and expert-placement cost outside any measurement
    if not args.no_warmup_corpus:
        wg = Generator(model = model, cache = cache, tokenizer = tokenizer,
                       max_chunk_size = args.chunk_size)
        warm_corpus(wg, tokenizer)
        reset_arm(wg)
        del wg
        # The warm pass moved experts; every measured arm starts from the load-time placement so the
        # arms differ by their drafter and not by what the pass before them left on the card
        if reset_between_arms:
            reset_placement(model)

    # Baseline
    result_baseline = None
    if not args.no_baseline:
        generator = Generator(
            model = model,
            cache = cache,
            tokenizer = tokenizer,
            max_chunk_size = args.chunk_size,
        )
        result_baseline = measure(generator, tokenizer, GreedySampler(), args.max_new_tokens)
        arm_line("baseline", result_baseline)
        reset_arm(generator)
        del generator
        if reset_between_arms:
            reset_placement(model)

    # N-gram draft
    result_ngram = None
    result_ngram_temp = None
    if args.s_ngram_match_min:
        generator = Generator(
            model = model,
            cache = cache,
            tokenizer = tokenizer,
            ngram_match_min = args.s_ngram_match_min,
            ngram_corpus = args.ngram_corpus,
            num_draft_tokens = args.s_ngram_draft_length,
            hybrid_draft_tokens = args.hybrid_draft_tokens,
            dynamic_draft_tokens = args.dynamic_draft,
            draft_confidence = args.draft_confidence,
            record_draft_stats = stats_sink is not None,
            max_chunk_size = args.chunk_size,
        )
        result_ngram = measure(generator, tokenizer, GreedySampler(), args.max_new_tokens, stats_sink)
        if args.temperature:
            result_ngram_temp = measure(generator, tokenizer, TopPSampler(0.9, 1), args.max_new_tokens)
        # This arm moves placement too, and it has no reset_arm() of its own: the drafter's arm must
        # still start from the load-time placement
        if reset_between_arms:
            reset_placement(model)

    # SD with draft model. With -rb_ab the arm is measured four times, ABBA over the module
    # flag generator._NO_BATCH_READBACK (batched vs. serial speculative-window readback);
    # the flag is read per round, so toggling between measure() windows is a clean in-process
    # A/B (the EXL3_NO_BATCH_READBACK env of the original PR is gone at v1.6.5).
    result_draft = None
    result_draft_temp = None
    rb_results = []
    if args.draft_model_dir:
        rb_arms = [("", None)] if not args.readback_ab else \
                  [("batched", False), ("serial", True), ("serial", True), ("batched", False)]
        # One Generator for the whole arm set: the readback flag is read per verification round
        # (generator.py), so toggling it between measure() windows is a clean in-process A/B and
        # four Generator instances were never needed (critique C6 - they cost ~700 MB each).
        generator = Generator(
            model = model,
            cache = cache,
            draft_model = draft_model,
            draft_cache = draft_cache,
            tokenizer = tokenizer,
            num_draft_tokens = args.num_draft_tokens,
            sparse_draft_head = getattr(args, 'sparse_draft_head', 0),
            hybrid_draft_tokens = args.hybrid_draft_tokens,
            draft_confidence = args.draft_confidence,
            dynamic_draft_tokens = args.dynamic_draft,
            record_draft_stats = stats_sink is not None,
            max_chunk_size = args.chunk_size,
        )
        for rb_i, (rb_label, rb_flag) in enumerate(rb_arms):
            if rb_flag is not None:
                exl_gen._NO_BATCH_READBACK = rb_flag
            # ABBA repeats are arms too: each one starts from the load-time placement of both models,
            # or the four runs differ by where the placement had drifted as well as by the flag
            if reset_between_arms:
                reset_placement(model)
                if draft_model is not None:
                    reset_placement(draft_model)
            res_draft = measure(generator, tokenizer, GreedySampler(), args.max_new_tokens, stats_sink if rb_i == 0 else None)
            arm_line(rb_label if rb_label else "draft", res_draft)
            if generator.sparse_head is not None:
                st = generator.sparse_head.stats()
                print(f" -- sparse draft head: {st['active']} columns, mode sparse {st['mode_sparse']} ({st['mode_switches']} switches), "
                      f"miss rate {st['miss_rate'] * 100:.2f}% (break-even {(st['breakeven_miss_rate'] or 0) * 100:.2f}%), "
                      f"rounds {st['rounds']}, probes {st['probes']} ({st['probe_disagree_inside']} inside / {st['probe_disagree_outside']} outside)")
            if args.readback_ab:
                rb_results.append((rb_label, rb_i // 2 + 1, res_draft))
            else:
                result_draft = res_draft
                if args.temperature:
                    result_draft_temp = measure(generator, tokenizer, TopPSampler(0.9, 1), args.max_new_tokens)
        reset_arm(generator)
        del generator
        exl_gen._NO_BATCH_READBACK = False

    # Print results
    draft_mode = "Draft model"
    if args.draft_model_dir and draft_model.caps.get("dflash_draft"):
        draft_mode = "DFlash"
    if args.draft_model_dir and draft_model.caps.get("mtp_draft"):
        draft_mode = "MTP"
    if args.hybrid_draft_tokens:
        draft_mode += f" + hybrid {args.hybrid_draft_tokens}"
    r = {
        "Baseline": result_baseline,
        "N-gram (greedy)": result_ngram,
        "N-gram (temp 1)": result_ngram_temp,
        f"{draft_mode} (greedy)": result_draft,
        f"{draft_mode} (temp 1)": result_draft_temp,
    }
    for rb_label, rb_run, res in rb_results:
        r[f"{draft_mode} (rb={rb_label} r{rb_run}, greedy)"] = res
    r = {k: v for k, v in r.items() if v is not None}
    if not r:
        print("No results to display")
        return

    categories = sorted({ category for result in r.values() for category in result.keys() })
    headers = [
        f"{col_yellow}Category{col_default}",
        *[f"{col_yellow}{name}{col_default}" for name in r.keys()],
    ]
    rows = []
    for category in categories:
        row = [f"{category}"]
        baseline = None
        for name, result in r.items():
            res_cat = result.get(category)
            if res_cat is not None:
                speedup = None
                if name == "Baseline":
                    baseline = res_cat["tps"]
                elif baseline is not None:
                    speedup = res_cat["tps"] / baseline
                s = f"{col_magenta}{res_cat['tps']:.2f}{col_default} t/s"
                if speedup is not None:
                    s += f", {col_green}{speedup:.2f}{col_default}x"
                if res_cat["acc_len"] is not None:
                    s += (f", {col_blue}{res_cat['acc_len']:.2f}{col_default}/"
                          f"{col_blue}{res_cat['draft_len']:.2f}{col_default} acc/draft")
            else:
                s = "-"
            row.append(s)
        rows.append(row)

    print()
    print(tabulate(rows, headers = headers, tablefmt = "pipe", colalign=("left", *["right"] * (len(headers) - 1)),))
    print()

    if args.draft_stats and stats_sink:
        with open(args.draft_stats, "w") as f:
            json.dump(stats_sink, f)
        print(f"Draft stats written to {args.draft_stats}")

    if args.plot_stats:
        if not stats_sink:
            print("No draft stats recorded, nothing to plot")
        else:
            plot_stats(stats_sink)

    if args.print_stats:
        if not stats_sink:
            print("No draft stats recorded, nothing to plot")
        else:
            print_stats(stats_sink)

    # Last line of the run: the harness reads these watermarks into passes.csv, and generation is
    # part of their peak, so they can only be reported once every arm has finished.
    print(f" -- alloc peak allocated={torch.cuda.max_memory_allocated() / 1024**2:.0f} MiB "
          f"reserved={torch.cuda.max_memory_reserved() / 1024**2:.0f} MiB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev = False)
    model_init.add_args(
        parser,
        default_cache_size = 49152,
        cache = True,
        add_draft_model_args = True,
        default_autosplit_max_batch_size = 1,
        default_chunk_size = 4096,
    )
    parser.add_argument("-nbl", "--no_baseline", action = "store_true", help = "Skip baseline measurement")
    parser.add_argument("-ngram_min", "--s_ngram_match_min", type = int, help = "N-gram minimum match length, default = 0 (disabled)", default = 0)
    parser.add_argument("-ngram_len", "--s_ngram_draft_length", type = int, help = "N-gram draft length, default = 4", default = 4)
    parser.add_argument("-tokens", "--max_new_tokens", type = int, help = "Max sampled tokens per round", default = 1024)
    parser.add_argument("-temp", "--temperature", action = "store_true", help = "Also sample with temperature")
    parser.add_argument("-swap_ok", "--swap_ok", action = "store_true",
                        help = "Measure a multi-arm comparison with dynamic expert placement (EXL3_MOE_CPU_SWAP) still on. Placement is reset between arms, but an arm that moves placement mid-generation generates different tokens, so the ratios are exploration only")
    parser.add_argument("-single", "--single_workload", type = str, help = "Limit to single workload", default = None)
    parser.add_argument("-dstats", "--draft_stats", type = str, help = "Write per-round (position, window, accepted) records to JSON file", default = None)
    parser.add_argument("-nwarm", "--no_warmup_corpus", action = "store_true",
                        help = "Skip the untimed warm pass over the workload corpus (keeps the cold-IO bias)")
    parser.add_argument("-rb_ab", "--readback_ab", action = "store_true", help = "Measure the draft-model arm ABBA over generator._NO_BATCH_READBACK (batched vs. serial speculative-window readback) in one process")
    parser.add_argument("-plot", "--plot_stats", action = "store_true", help = "Plot per-round draft stats in a matplotlib window after the run")
    parser.add_argument("-print", "--print_stats", action = "store_true", help = "Print per-round draft stats to the console after the run")
    _args = parser.parse_args()
    main(_args)
