import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from exllamav3.util.progress import ProgressBar
from exllamav3 import Config, Model, Cache, Tokenizer, model_init, Generator, Job, GreedySampler
import torch

# ANSI codes
ESC = "\u001b"
col_default = "\u001b[0m"
col_yellow = "\u001b[33;1m"
col_blue = "\u001b[34;1m"
col_green = "\u001b[32;1m"
col_red = "\u001b[31;1m"
col_gray = "\u001b[37;1m"

@torch.inference_mode()
def main(args):

    # Load model
    model, config, cache, tokenizer, draft_model, draft_config, draft_cache = model_init.init(args)
    generator = Generator(
        model,
        cache,
        tokenizer,
        draft_model = draft_model,
        draft_cache = draft_cache,
        show_visualizer = args.visualize_cache,
        max_chunk_size = args.chunk_size,
        cpu_cache_size = int(args.cpu_cache_size * 1024 ** 3),
        recurrent_cache_size = int(args.recurrent_cache_size * 1024 ** 3),
    )
    bpw_layer, bpw_head, vram_bits = model.get_storage_info()

    print(f" -- Model: {args.model_dir}")
    print(f" -- Bitrate: {bpw_layer:.2f} bpw / {bpw_head:.2f} bpw (head)")

    # Get
    texts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_texts")
    with open(os.path.join(texts_dir, "illustrious_client.txt"), "r") as file:
        text_ic_orig = file.read()
    with open(os.path.join(texts_dir, "illustrious_client_c1.txt"), "r") as file:
        text_ic_french = file.read()
    with open(os.path.join(texts_dir, "illustrious_client_c2.txt"), "r") as file:
        text_ic_zoomer = file.read()
    with open(os.path.join(texts_dir, "illustrious_client_sum.txt"), "r") as file:
        text_ic_sum = file.read()
    with open(os.path.join(texts_dir, "variable_man_mod.txt"), "r") as file:
        text_vm_mod = file.read()
    with open(os.path.join(texts_dir, "variable_man_mod_c1.txt"), "r") as file:
        text_vm_pony = file.read()
    with open(os.path.join(texts_dir, "variable_man_sum.txt"), "r") as file:
        text_vm_sum = file.read()
    with open(os.path.join(texts_dir, "variable_man_char.txt"), "r") as file:
        text_vm_char = file.read()
    if args.extra_long:
        with open(os.path.join(texts_dir, "pride_prejudice_mod.txt"), "r") as file:
            text_pp_mod = file.read()
        with open(os.path.join(texts_dir, "pride_prejudice_chap.txt"), "r") as file:
            text_pp_chap = file.read()

    # Template
    # The harness parses the LONGCTX lines at the end, so each job carries the name it is reported
    # under and the prompt length it actually exercised. A retrieval answer is prose - no script
    # can grade it - but the run has to prove which context length each test really saw.
    job_lens = {}
    def make_job(name, instruction):
        chat = [{
            "role": "user",
            "content": instruction
        }]
        input_ids = tokenizer.hf_chat_template(chat, add_generation_prompt = True, enable_thinking = True)
        job = Job(
            input_ids = input_ids,
            max_new_tokens = 16384 if args.extra_long else 1024,
            stop_conditions = config.eos_token_id_list,
            # sampler = GreedySampler()
        )
        job_lens[name] = int(input_ids.shape[-1])
        return job, input_ids.shape[-1]

    # Tests
    # TODO: Find some original source material that models are sure to be entirely unfamiliar with
    job_ic_sum, len_ic_sum = make_job("ic_sum", text_ic_orig + "\n\n---\n\nProvide an extremely short summary of the story.")
    job_ic_french, _ = make_job("ic_french", text_ic_french + "\n\n---\n\nOne paragraph in this story has been translated to a different language. Translate it back.")
    job_ic_zoomer, _ = make_job("ic_zoomer", text_ic_zoomer + "\n\n---\n\nTwo paragraphs have been rewritten in a zoomer slang style. Identify them.")
    job_vm_sum, len_vm_sum = make_job("vm_sum", text_vm_mod + "\n\n---\n\nProvide an extremely short summary of the story.")
    vm_q1 = "Why do the SRB computers stop giving reliable war-odds after Edward Milsom arrives in the 22nd century?"
    vm_a1 = "Milsom’s behavior is unpredictable to the machines because he comes from a different era and doesn’t fit their statistical patterns, so his presence introduces a “variable” they cannot factor."
    vm_q2 = "What does Edward Milsom secretly do to the Icarus bomb’s control turret?"
    vm_a2 = "Instead of wiring it to trigger an explosion, he rewires it so the craft can decelerate safely from faster-than-light speed, turning it from a bomb into a workable FTL drive."
    vm_q3 = "How does humanity ultimately benefit from Milsom’s interference, even though Earth loses the war against Jorblax?"
    vm_a3 = "Milsom’s solution delivers a practical faster-than-light return method, giving Earth true interstellar travel and opening the entire universe for exploration and colonization, making the war’s outcome irrelevant."
    job_vm_q1, _ = make_job("vm_q1", text_vm_mod + f"\n\n---\n\nAnswer in one paragraph: {vm_q1}")
    job_vm_q2, _ = make_job("vm_q2", text_vm_mod + f"\n\n---\n\nAnswer in one paragraph: {vm_q2}")
    job_vm_q3, _ = make_job("vm_q3", text_vm_mod + f"\n\n---\n\nAnswer in one paragraph: {vm_q3}")
    job_vm_char, _ = make_job("vm_char", text_vm_mod + "\n\n---\n\nList all the named characters in the story.")
    job_vm_pony, _ = make_job("vm_pony", text_vm_pony+ "\n\n---\n\nA passage from an unrelated story is inserted in the middle of the text. Can you find it?")
    if args.extra_long:
        job_pp_mod, len_pp_mod = make_job("pp_mod", text_pp_mod + "\n\n---\n\nA passage from an unrelated story is inserted in the middle of the text. Can you find it?")
        job_pp_chap, _ = make_job("pp_chap", text_pp_mod + "\n\n---\n\nCan you provide a short, succinct title for each chapter in the story?")

    # Inference
    jobs = [
        ("ic_sum", job_ic_sum),
        ("ic_french", job_ic_french),
        ("ic_zoomer", job_ic_zoomer),
        ("vm_sum", job_vm_sum),
        ("vm_q1", job_vm_q1),
        ("vm_q2", job_vm_q2),
        ("vm_q3", job_vm_q3),
        ("vm_pony", job_vm_pony),
        ("vm_char", job_vm_char),
    ]
    if args.extra_long:
        jobs += [
            ("pp_mod", job_pp_mod),
            ("pp_chap", job_pp_chap),
        ]
    generator.enqueue([j for _, j in jobs])

    with ProgressBar("Inference", len(jobs)) as pb:
        while j := generator.num_remaining_jobs():
            generator.iterate()
            pb.update(len(jobs) - j)

    # Results
    print()
    print(f"{col_green}------------{col_default}")
    print(f"{col_green}SUMMARY TEST{col_default}")
    print(f"{col_green}------------{col_default}")
    print(f"{col_blue}Short summary of 'The Illustrious Client', {len_ic_sum} tokens.\nReference summary:{col_default}")
    print(f"{col_gray}{text_ic_sum.strip()}{col_default}")
    print()
    print(job_ic_sum.full_completion.strip())
    print()

    print()
    print(f"{col_green}-----------{col_default}")
    print(f"{col_green}FRENCH TEST{col_default}")
    print(f"{col_green}-----------{col_default}")
    print(f"{col_blue}One paragraph in this story has been translated to a different language. Translate it back.{col_default}")
    print()
    print(job_ic_french.full_completion.strip())
    print()

    print()
    print(f"{col_green}-----------{col_default}")
    print(f"{col_green}ZOOMER TEST{col_default}")
    print(f"{col_green}-----------{col_default}")
    print(f"{col_blue}A zoomer has edited the text. Identify the edited passages.{col_default}")
    print()
    print(job_ic_zoomer.full_completion.strip())
    print()

    print()
    print(f"{col_green}------------{col_default}")
    print(f"{col_green}SUMMARY TEST{col_default}")
    print(f"{col_green}------------{col_default}")
    print(f"{col_blue}Short summary of a version of 'The Variable Man' with some names replaced, {len_vm_sum} tokens.\nReference summary:{col_default}")
    print(f"{col_gray}{text_vm_sum.strip()}{col_default}")
    print()
    print(job_vm_sum.full_completion.strip())
    print()

    print()
    print(f"{col_green}--------{col_default}")
    print(f"{col_green}Q&A TEST{col_default}")
    print(f"{col_green}--------{col_default}")
    print(f"{col_blue}{vm_q1} Reference answer:{col_default}")
    print(f"{col_gray}{vm_a1}{col_default}")
    print()
    print(job_vm_q1.full_completion.strip())
    print()
    print(f"{col_blue}{vm_q2} Reference answer:{col_default}")
    print(f"{col_gray}{vm_a2}{col_default}")
    print()
    print(job_vm_q2.full_completion.strip())
    print()
    print(f"{col_blue}{vm_q3} Reference answer:{col_default}")
    print(f"{col_gray}{vm_a3}{col_default}")
    print()
    print(job_vm_q3.full_completion.strip())
    print()

    print()
    print(f"{col_green}---------------{col_default}")
    print(f"{col_green}CORRUPTION TEST{col_default}")
    print(f"{col_green}---------------{col_default}")
    print(f"{col_blue}Some MLP fan fiction has made it into the story. Can we detect it?{col_default}")
    print()
    print(job_vm_pony.full_completion.strip())
    print()

    print()
    print(f"{col_green}--------------------{col_default}")
    print(f"{col_green}NAME EXTRACTION TEST{col_default}")
    print(f"{col_green}--------------------{col_default}")
    print(f"{col_blue}List all the named characters in the story.\nReference:{col_default}")
    print(f"{col_gray}{text_vm_char}{col_default}")
    print()
    print(job_vm_char.full_completion.strip())
    print()

    if args.extra_long:
        print()
        print(f"{col_green}---------------{col_default}")
        print(f"{col_green}EXTRA LONG TEST{col_default}")
        print(f"{col_green}---------------{col_default}")
        print(f"{col_blue}An unrelated passage has snuck into Pride & Prejudice, {len_pp_mod} tokens. Can we find it?{col_default}")
        print()
        print(job_pp_mod.full_completion.strip())
        print()
        print(f"{col_blue}Coming up with a title for each chapter. Reference answer:{col_default}")
        print(f"{col_gray}{text_pp_chap}{col_default}")
        print()
        print(job_pp_chap.full_completion.strip())
        print()

    # Machine-readable summary: the harness turns these into results/longctx.csv and marks a run
    # without them as failed. Retrieval quality is a reader's verdict - no script grades a summary -
    # so this proves the tests ran, names the context length each one exercised, and gives the
    # reader the transcript to score.
    print()
    for name, job in jobs:
        out = (job.full_completion or "").strip()
        print(f"LONGCTX job={name} prompt_tokens={job_lens.get(name, 0)} out_chars={len(out)}")
    print(f"LONGCTX jobs={len(jobs)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev = False)
    model_init.add_args(
        parser,
        add_draft_model_args = True,
        default_cache_size = 65536,
        default_autosplit_max_batch_size = 9
    )
    parser.add_argument("-vis", "--visualize_cache", action = "store_true", help = "Show cache visualizer (slow)")
    parser.add_argument("-xl", "--extra_long", action = "store_true", help = "Run harder test with longer inputs and room for long reasoning traces (increase --cache_size to ~200k tokens when using this)")
    _args = parser.parse_args()
    main(_args)
