#!/usr/bin/env python3
"""Emit an (original, early-edited) prompt pair for checkpoint-replay measurement.

`recurrent_checkpoint_interval_pp` does not move tok/s on a cold prefill. Its
doc (config_models.py) says it governs what a mid-conversation *edit* costs:
with the 32768 default an early edit costs a full re-prefill, with a denser grid
replay is proportional to distance from the edit to the end.

So the measurement needs two turns in one boot:
  turn 1 -> original   (checkpoints written for this prefix)
  turn 2 -> edited     (an EARLY message changed, so everything after it must be
                        replayed; the distance to the nearest usable checkpoint
                        is what the knob controls)

Editing early is the whole point -- a late edit is cheap under either setting.
The edit is a small wording change, not a structural one, so token count stays
close and any wall-clock difference is replay cost rather than more prefill.

Usage:
  _kb_mkedit.py --src <synth.json> --outdir <dir> [--edit-frac 0.10]
Writes <outdir>/edit_a.json and <outdir>/edit_b.json.
"""

import argparse
import copy
import json
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument(
        "--edit-frac",
        type=float,
        default=0.10,
        help="edit at this fraction into the conversation (0.10 = early)",
    )
    args = ap.parse_args()

    with open(args.src, encoding="utf-8") as fh:
        data = json.load(fh)

    msgs = data["messages"]
    if len(msgs) < 8:
        print(f"source too short to edit meaningfully ({len(msgs)} messages)")
        return 1

    # Find the first user message at/after the target fraction. Editing a user
    # message is realistic (the operator restates a requirement); editing a tool
    # result would desync the tool_call/tool_result pairing that the plan already
    # had to fix once.
    target = int(len(msgs) * args.edit_frac)
    idx = next(
        (i for i in range(target, len(msgs)) if msgs[i].get("role") == "user"),
        None,
    )
    if idx is None:
        print("no user message at/after the edit fraction")
        return 1

    edited = copy.deepcopy(data)  # deep: never mutate the shared tool_calls
    orig_txt = str(edited["messages"][idx].get("content") or "")
    edited["messages"][idx]["content"] = orig_txt + (
        "\n\n[EDIT] Also make sure the patch is minimal and does not "
        "reformat unrelated lines."
    )

    os.makedirs(args.outdir, exist_ok=True)
    a_path = os.path.join(args.outdir, "edit_a.json")
    b_path = os.path.join(args.outdir, "edit_b.json")
    with open(a_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False)
    with open(b_path, "w", encoding="utf-8") as fh:
        json.dump(edited, fh, ensure_ascii=False)

    added = len(edited["messages"][idx]["content"]) - len(orig_txt)
    print(f"edit at message {idx}/{len(msgs)} (role={msgs[idx]['role']})")
    print(f"  +{added} chars appended -> token delta must stay small")
    print(f"wrote {a_path}")
    print(f"wrote {b_path}")
    print(
        "send edit_a then edit_b in ONE boot (no -Rotate) so turn 2 exercises "
        "replay from turn 1's checkpoints"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())