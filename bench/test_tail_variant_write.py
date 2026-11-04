#!/usr/bin/env python3
"""The variant writer, pinned on synthetic shards (no torch, no checkpoint, no GPU)."""

import json
import os
import struct
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tail_variant_write import expert_globs, main, override_yaml, write_variant  # noqa: E402

PREFIX = "model.language_model.layers.{l}.mlp.experts.{e}.trellis"


def write_shard(path: str, tensors: dict[str, bytes]):
    """A minimal safetensors file: 8-byte header length, JSON header, payload."""
    header, payload, off = {}, bytearray(), 0
    for name, data in tensors.items():
        pad = (-len(payload)) % 8
        payload.extend(b"\0" * pad)
        off = len(payload)
        payload.extend(data)
        header[name] = {"dtype": "I16", "shape": [len(data) // 2], "data_offsets": [off, off + len(data)]}
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * ((-len(blob)) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        f.write(bytes(payload))


def read_shard(path: str) -> dict[str, bytes]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            f.seek(base + meta["data_offsets"][0])
            out[name] = f.read(meta["data_offsets"][1] - meta["data_offsets"][0])
    return out


def _model(tmp) -> str:
    d = os.path.join(tmp, "model")
    os.makedirs(d)
    t = {PREFIX.format(l=0, e=e): bytes([e, 0] * 8) for e in range(3)}
    t[PREFIX.format(l=1, e=0)] = bytes([9, 9] * 8)
    t["model.language_model.layers.0.mlp.gate.weight"] = bytes(16)
    write_shard(os.path.join(d, "model-00001-of-00001.safetensors"), t)
    return d


def test_the_variant_carries_exactly_the_planned_tensors_byte_for_byte():
    with tempfile.TemporaryDirectory() as tmp:
        model = _model(tmp)
        wanted = [PREFIX.format(l=0, e=1), PREFIX.format(l=1, e=0)]
        header = write_variant(model, wanted, os.path.join(tmp, "out"))
        assert sorted(header) == sorted(wanted)
        got = read_shard(os.path.join(tmp, "out", "tail.safetensors"))
        base = read_shard(os.path.join(model, "model-00001-of-00001.safetensors"))
        for name in wanted:
            assert got[name] == base[name]
        assert "model.language_model.layers.0.mlp.gate.weight" not in got


def test_the_override_spec_names_only_the_planned_experts():
    plan = {"layers": {"0": {"expert_prefixes": ["model.language_model.layers.0.mlp.experts.131",
                                                 "model.language_model.layers.0.mlp.experts.132"]}}}
    globs = expert_globs(plan)
    assert globs == ["model.language_model.layers.0.mlp.experts.131.*",
                     "model.language_model.layers.0.mlp.experts.132.*"]
    # a broad glob would make every GPU-resident expert's load raise: the collection has no fallback
    assert all(g.count("experts.*") == 0 for g in globs)
    yml = override_yaml(plan, "D:\\llms\\variant")
    assert yml.count("- key:") == 2 and "model_dir: D:\\llms\\variant" in yml
    assert "source: tail" in yml


def test_a_plan_naming_a_tensor_that_is_not_there_exits_two():
    with tempfile.TemporaryDirectory() as tmp:
        model = _model(tmp)
        plan = {"layers": {"0": {"expert_prefixes": ["model.language_model.layers.0.mlp.experts.99"]}}}
        plan_path = os.path.join(tmp, "plan.json")
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(plan, f)
        assert main(["--plan", plan_path, "--model", model, "--out", os.path.join(tmp, "o")]) == 2


def test_the_written_shard_round_trips_through_the_writer_reader():
    with tempfile.TemporaryDirectory() as tmp:
        model = _model(tmp)
        write_variant(model, [PREFIX.format(l=0, e=2)], os.path.join(tmp, "o"))
        got = read_shard(os.path.join(tmp, "o", "tail.safetensors"))
        assert list(got) == [PREFIX.format(l=0, e=2)]
        assert len(got[PREFIX.format(l=0, e=2)]) == 16


TESTS = [
    test_the_variant_carries_exactly_the_planned_tensors_byte_for_byte,
    test_the_override_spec_names_only_the_planned_experts,
    test_a_plan_naming_a_tensor_that_is_not_there_exits_two,
    test_the_written_shard_round_trips_through_the_writer_reader,
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
