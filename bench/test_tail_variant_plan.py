#!/usr/bin/env python3
"""The tail-plan arithmetic, pinned on synthetic checkpoints (no model, no GPU)."""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tail_variant_plan import build_plan, is_scale, main, parse_expert  # noqa: E402

LAYER = "model.language_model.layers.{l}.mlp.experts.{e}.{p}.{s}"


def _hdr(layer=0, n=4, trellis=1000, scale=100):
    h = {}
    for e in range(n):
        for p in ("gate_proj", "up_proj", "down_proj"):
            h[LAYER.format(l=layer, e=e, p=p, s="trellis")] = {"nbytes": trellis, "shape": [1], "dtype": "I16"}
            h[LAYER.format(l=layer, e=e, p=p, s="suh")] = {"nbytes": scale, "shape": [1], "dtype": "F16"}
    return h


def test_parse_expert_reads_layer_expert_projection_suffix():
    got = parse_expert(LAYER.format(l=7, e=42, p="down_proj", s="trellis"))
    assert got == (7, 42, "down_proj", "trellis")
    assert parse_expert("model.language_model.layers.0.mlp.gate.weight") is None


def test_only_scales_are_exempt_from_the_rate():
    assert is_scale("down_proj.suh") and is_scale("down_proj.svh") and is_scale("down_proj.bias")
    assert not is_scale("down_proj.trellis")


def test_the_tail_is_the_experts_beyond_the_gpu_slice_and_scales_do_not_shrink():
    plan = build_plan(_hdr(n=4), mcs=2, tail_bpw=2.0, source_bits=4.0)
    layer = plan["layers"][0]
    assert layer["gpu_experts"] == 2 and layer["tail_experts"] == 2
    # per tail expert: 3 trellis at 1000 B halve to 1500, 3 scales at 100 B stay 300
    assert layer["bytes_now"] == 2 * (3 * 1000 + 3 * 100)
    assert layer["bytes_at_tail_bpw"] == 2 * (1500 + 300)


def test_the_hot_first_profile_moves_which_experts_are_the_tail():
    counts = {"0": [0, 10, 5, 1]}          # expert 1 is the hottest
    plan = build_plan(_hdr(n=4), mcs=2, tail_bpw=2.0, source_bits=4.0, profile=counts)
    tail = plan["layers"][0]["expert_prefixes"]
    assert not any(t.endswith(".experts.1") for t in tail)
    assert any(t.endswith(".experts.0") for t in tail)


def test_a_checkpoint_with_no_experts_is_refused():
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"quantization_config": {"bits": 3.18}}, f)
        assert main(["--model", d, "--mcs", "381", "--tail-bpw", "2.0"]) == 2


def test_more_spares_than_the_gpu_slice_does_not_wrap_the_slice():
    """first = 4 - 3 = 1 and spares = 2: the worker holds the whole layer, not one expert."""
    plan = build_plan(_hdr(n=4), mcs=3, tail_bpw=2.0, source_bits=4.0, spares=2)
    assert plan["layers"][0]["tail_experts"] == 4


TESTS = [
    test_more_spares_than_the_gpu_slice_does_not_wrap_the_slice,
    test_parse_expert_reads_layer_expert_projection_suffix,
    test_only_scales_are_exempt_from_the_rate,
    test_the_tail_is_the_experts_beyond_the_gpu_slice_and_scales_do_not_shrink,
    test_the_hot_first_profile_moves_which_experts_are_the_tail,
    test_a_checkpoint_with_no_experts_is_refused,
]

if __name__ == "__main__":
    failures = 0
    for t in TESTS:
        try:
            t()
            print(f"PASS {t.__name__}")
        except BaseException as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL {t.__name__}: {e}")
    sys.exit(1 if failures else 0)
