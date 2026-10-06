#!/usr/bin/env python
# _kb_mklongctx.py -- synthesise long-context prompts for plan §0.7.
#
# Why synthesise instead of pointing at a flag: eval/spec_decode.py and
# eval/perf.py select workloads by CATEGORY, not by length, and the longest
# real conversation available is agentic_code_29 (~30k prompt tokens). There is
# nothing to point at for 35k / 128k / 260k, so the prompts are built here.
#
# Two facts that drive the design, both measured rather than assumed:
#   - all five agentic_code files hold only 36,326 text tokens combined, so
#     reaching 260k means recycling the histories several times over;
#   - agentic_code_10 is 7,808 text tokens but the SERVER logged 17,830 prompt
#     tokens -- the 11 tool schemas cost ~10k. Tools are per-request, so they
#     are attached once, not per message.
#
# Prefill measurement needs COLD prefixes: repeating one long prompt means every
# rep after the first is ~100% prefix-cached and measures nothing. So --variants
# emits DISTINCT conversations (different start offsets into the cycle), and each
# variant is a single-request prompt rather than a progressive replay.
#
# Output is OAI request-body shaped, exactly like the agentic_code_*.json files.
#
# Usage:
#   python _kb_mklongctx.py --target 35000 --variants 3 --outdir <dir>
#   python _kb_mklongctx.py --target 128000 --variants 2 --outdir <dir>

import argparse
import json
import os
import sys

PROMPT_DIR = r"C:\Users\yoho\Downloads\exllamav3-kvarn\eval\prompts"
MODEL_DIR = r"D:\llms\Qwen3.8-Flash-Next-exl3-3.05bpw"
SOURCES = ["agentic_code_01.json", "agentic_code_05.json", "agentic_code_10.json",
           "agentic_code_20.json", "agentic_code_29.json"]

# Roles that may legally end a prompt: the model has to be asked something.
ASK_ROLES = ("user", "tool")


def load_sources():
    convs = []
    for name in SOURCES:
        with open(os.path.join(PROMPT_DIR, name), encoding="utf-8") as fh:
            d = json.load(fh)
        convs.append({"name": name, "messages": d["messages"], "tools": d.get("tools"),
                      "tool_choice": d.get("tool_choice")})
    if not convs or not convs[0]["tools"]:
        sys.exit("no tools in source prompts; cannot mirror the production request shape")
    return convs


def text_tokens(tok, messages):
    txt = "".join(str(m.get("content") or "") for m in messages)
    return len(tok(txt).input_ids)


def build(convs, tok, target_text_tokens, variant):
    """Concatenate histories, cycling, starting at `variant` so each variant is a
    genuinely different conversation. Trimmed to end on an ask role.

    Recycling reuses the source assistant messages verbatim, which duplicates
    their tool_call ids (a 25k build had 56 calls sharing 26 unique ids).
    Duplicated ids across turns risk confusing tool-result matching, and the
    ids are structurally irrelevant for prefill measurement -- so every
    occurrence is re-stamped, keeping each assistant call and its tool result
    consistent."""
    msgs = []
    order = [(variant + i) % len(convs) for i in range(len(convs))]
    rounds = 0
    while True:
        for idx in order:
            if text_tokens(tok, msgs) >= target_text_tokens:
                break
            # A leading system message would be nonsense mid-conversation, and
            # repeating one verbatim is fine for prefill work but keep it out of
            # the middle to stay closer to a real transcript.
            chunk = [m for m in convs[idx]["messages"]]
            for m in chunk:
                if m.get("role") == "system":
                    continue
                m2 = dict(m)
                # Deep-ish copy of tool_calls: a shallow dict(m) shares the SAME
                # tool_call dicts as the source message, so re-stamping ids
                # mutated the SOURCE files. Pass 2 then read already-stamped ids
                # while its tool messages still held raw ones, orphaning every
                # tool result (60 of 60 in testing).
                if m.get("tool_calls"):
                    m2["tool_calls"] = [dict(tc) for tc in m["tool_calls"]]
                if rounds > 0:
                    c = m2.get("content")
                    if isinstance(c, str):
                        m2["content"] = f"[pass {rounds}] {c}"
                    else:
                        m2["content"] = f"[pass {rounds}]"
                msgs.append(m2)
        rounds += 1
        if text_tokens(tok, msgs) >= target_text_tokens or rounds > 40:
            break

    # End on an ask so the model generates. Walk back to the last user/tool.
    while msgs and msgs[-1].get("role") not in ASK_ROLES:
        msgs.pop()

    # Re-stamp tool_call ids so every occurrence is unique, keeping each
    # assistant-call -> tool-result pairing intact.
    #
    # Must be sequential with a PENDING map, not a global one: recycling
    # repeats the same source ids in later passes, so a global map aliases
    # distinct calls onto one id. And the tool results answering an assistant
    # turn appear after it, possibly several, so they must be resolved against
    # the map built by the PRECEDING assistant message rather than whatever
    # happens to be current. (A global map left 60 of 56 tool messages
    # orphaned in testing; a naive last-id scheme mis-pairs parallel calls.)
    n = 0
    pending = {}
    for m in msgs:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                old = tc.get("id")      # MUST be read before tc["id"] is overwritten
                n += 1
                new = f"call_{n:05d}"
                pending[old] = new
                tc["id"] = new
        elif m.get("role") == "tool":
            old = m.get("tool_call_id")
            if old in pending:
                m["tool_call_id"] = pending[old]
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, required=True,
                    help="target TEXT tokens (tools add ~10k on top; the server "
                         "log reports the true prompt count, so calibrate there)")
    ap.add_argument("--variants", type=int, default=1)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    convs = load_sources()
    os.makedirs(args.outdir, exist_ok=True)

    for v in range(args.variants):
        msgs = build(convs, tok, args.target, v)
        body = {
            "model": "Qwen3.8-Flash-Next-exl3-3.05bpw",
            "max_tokens": 256,
            "top_p": 1.0,
            "messages": msgs,
            "tools": convs[v % len(convs)]["tools"],
            "tool_choice": convs[v % len(convs)]["tool_choice"],
            "stream": False,
        }
        name = f"synth_{args.target // 1000}k_v{v}.json"
        path = os.path.join(args.outdir, name)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(body, fh, ensure_ascii=False)
        print(f"{name}: {len(msgs)} msgs, {text_tokens(tok, msgs)} text tokens, "
              f"{os.path.getsize(path)} bytes", flush=True)
    print(f"tools add ~10k prompt tokens per request; expect server-reported "
          f"prompt ~= text + ~10000")


if __name__ == "__main__":
    main()