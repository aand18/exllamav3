#!/usr/bin/env python3
"""The tail requantizer, pinned on synthetic shards and the real engine source (no torch, no GPU).

The dry run is this file's main subject: it is the half of `--apply` a Linux box with no card and no
torch can still prove, and it proves the two things that would otherwise be guesses - that every
planned tensor is in the checkpoint, and that the engine call it prints is the one the engine source
carries today.
"""

import contextlib
import io
import json
import os
import struct
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tail_requantize import (Refusal, analyse, engine_calls, main, source_rate,  # noqa: E402
                             trellis_words)
from tail_variant_plan import read_headers  # noqa: E402
from tail_variant_write import write_requantized_shard  # noqa: E402

LAYER = "model.language_model.layers.0.mlp.experts.{e}"
PROJ = ("gate_proj", "up_proj", "down_proj")


def write_shard(path: str, tensors: dict[str, tuple[str, list, bytes]]) -> None:
    """name -> (dtype, shape, payload), written as a real safetensors file."""
    header, payload = {}, bytearray()
    for name, (dtype, shape, data) in tensors.items():
        pad = (-len(payload)) % 8
        payload.extend(b"\0" * pad)
        off = len(payload)
        payload.extend(data)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + len(data)]}
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


def expert_tensors(expert: int, words: int = 48, drop: str | None = None):
    """One expert's tensors at a given trellis width (48 words = 3.0 bpw, 32 = 2.0 bpw).

    The shape is a real EXL3 tile grid: k and n are both 128-multiples, which is what the encoder
    asserts (quantize.py:835-842) and what the dry run checks before --apply would reach it.
    """
    t = {}
    for proj in ("gate_proj", "up_proj", "down_proj"):
        k = n = 128
        base = f"{LAYER.format(e=expert)}.{proj}"
        t[f"{base}.trellis"] = ("I16", [k // 16, n // 16, words],
                                bytes(range(256)) * (k // 16 * n // 16 * words * 2 // 256))
        t[f"{base}.suh"] = ("F16", [k], bytes([expert, 1] * k))
        t[f"{base}.svh"] = ("F16", [n], bytes([expert, 2] * n))
        t[f"{base}.mul1"] = ("I32", [], b"\x2d\x12\xdc\x83")
    if drop:
        for name in [n for n in t if n.endswith(drop)]:
            del t[name]
    return t


def model_dir(tmp: str, experts=(131, 132), drop: str | None = None) -> str:
    d = os.path.join(tmp, "model")
    os.makedirs(d, exist_ok=True)
    tensors = {}
    for e in experts:
        tensors.update(expert_tensors(e, drop=drop))
    tensors["model.language_model.layers.0.mlp.gate.weight"] = ("F16", [4], bytes(8))
    write_shard(os.path.join(d, "model-00001-of-00001.safetensors"), tensors)
    return d


def plan_file(tmp: str, experts=(131, 132), source_bits=3.0, tail_bpw=2.0) -> str:
    plan = {"mcs": 381, "tail_bpw": tail_bpw, "source_bits": source_bits, "spares": 0,
            "layers": {"0": {"num_experts": 133, "gpu_experts": 131, "tail_experts": len(experts),
                             "expert_prefixes": [LAYER.format(e=e) for e in experts]}},
            "tail_bytes_now": 0, "tail_bytes_at_tail_bpw": 0}
    path = os.path.join(tmp, "plan.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(plan, f)
    return path


def run(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


# --------------------------------------------------------------------------------------------------


def test_the_dry_run_prints_the_call_it_read_from_the_engine_source():
    with tempfile.TemporaryDirectory() as tmp:
        code, out, err = run(["--plan", plan_file(tmp), "--model", model_dir(tmp),
                              "--out", os.path.join(tmp, "out"), "--bpw", "2.0"])
        assert code == 0, f"exit {code}: {err}"
        # the call, the module and the file:line, all parsed out of exllamav3/
        assert "exllamav3.modules.quant.exl3_lib.quantize.quantize_exl3(" in out
        assert "exllamav3/modules/quant/exl3_lib/quantize.py:1652" in out
        assert "get_weight_tensor(self)" in out and "ext.reconstruct" in out
        assert "H_data, quant_args, return_weight_q, progress_str, verbose, swap_to_device, save_reg" \
            in out
        # per projection, and in total
        assert out.count("gate_proj") >= 1 and "total" in out
        assert "33" not in err and "refusing" not in err
        # nothing was written: the default mode is a dry run
        assert not os.path.exists(os.path.join(tmp, "out", "tail.safetensors"))


def test_the_dry_run_counts_tiles_not_scaled_bytes():
    # 128x128 weights at 2.0 bpw: 8 x 8 tiles of 32 int16 words
    assert trellis_words(2.0) == 32
    assert trellis_words(3.5) == 56
    for bad in (2.25, 0.1, -1.0):
        try:
            trellis_words(bad)
        except Refusal:
            pass
        else:
            raise AssertionError(f"trellis_words({bad}) was accepted: the encoder cannot express it")
    assert source_rate([8, 8, 48]) == 3.0
    assert source_rate([8, 8, 56]) == 3.5
    assert source_rate([8, 8, 50]) is None          # not a whole number of 32-bit words
    assert source_rate([8, 8]) is None              # not a trellis
    with tempfile.TemporaryDirectory() as tmp:
        model = model_dir(tmp)
        plan = json.load(open(plan_file(tmp), encoding="utf-8"))
        problems, rows = analyse(read_headers(model), plan, 2.0)
        assert not problems, problems
        row = rows["gate_proj"]
        # two experts, 8x8 tiles of 48 words -> 32 words; the scales are counted, never scaled
        assert row["trellis_now"] == 2 * 8 * 8 * 48 * 2 == 12288
        assert row["trellis_then"] == 2 * 8 * 8 * 32 * 2 == 8192
        assert row["scales"] == 2 * (128 + 128) * 2 + 2 * 4       # suh + svh + the mul1 marker
        assert row["rates"] == {3.0: 2}


def test_a_target_above_the_source_rate_exits_two_with_both_numbers():
    with tempfile.TemporaryDirectory() as tmp:
        code, _, err = run(["--plan", plan_file(tmp), "--model", model_dir(tmp),
                            "--out", os.path.join(tmp, "out"), "--bpw", "4.0"])
        assert code == 2, f"exit {code}"
        assert "target 4 bpw is not below the source 3 bpw" in err
        assert ".gate_proj.trellis" in err and "[8, 8, 48]" in err
        # gated on the tensor's own width, so the plan's mean rate cannot wave it through
        assert "4" in err and "3" in err


def test_a_plan_naming_a_missing_tensor_exits_two_naming_it():
    with tempfile.TemporaryDirectory() as tmp:
        # (a) the expert is not in the checkpoint at all
        code, _, err = run(["--plan", plan_file(tmp, experts=(131, 999)), "--model", model_dir(tmp),
                            "--out", os.path.join(tmp, "out"), "--bpw", "2.0"])
        assert code == 2, f"exit {code}"
        assert LAYER.format(e=999) in err and "not in the checkpoint" in err
    with tempfile.TemporaryDirectory() as tmp:
        # (b) the expert is there but a tensor the loader needs is not
        model = model_dir(tmp, drop=".suh")
        code, _, err = run(["--plan", plan_file(tmp), "--model", model,
                            "--out", os.path.join(tmp, "out"), "--bpw", "2.0"])
        assert code == 2, f"exit {code}"
        assert f"{LAYER.format(e=131)}.gate_proj.suh is missing" in err
        assert "trellis, suh and svh" in err


def test_apply_refuses_where_the_engine_or_torch_is_missing():
    with tempfile.TemporaryDirectory() as tmp:
        code, _, err = run(["--plan", plan_file(tmp), "--model", model_dir(tmp),
                            "--out", os.path.join(tmp, "out"), "--bpw", "2.0", "--apply"])
        try:
            import torch  # noqa: F401
        except ImportError:
            assert code == 2, f"exit {code}"
            assert "torch is not importable" in err and "--apply needs" in err
            assert not os.path.exists(os.path.join(tmp, "out", "tail.safetensors"))


def test_the_substitution_helper_writes_the_stub_payloads_and_copies_the_rest():
    with tempfile.TemporaryDirectory() as tmp:
        model = model_dir(tmp)
        plan = json.load(open(plan_file(tmp), encoding="utf-8"))
        stub_trellis = b"\x07\x01\x02\x03\x04\x05"

        def stub(prefix, source):
            assert set(source) >= {"gate_proj.trellis", "gate_proj.suh", "gate_proj.svh"}
            return {"gate_proj.trellis": stub_trellis} if prefix.endswith(".131") else {}

        out = os.path.join(tmp, "out")
        header = write_requantized_shard(model, plan, out, stub)
        got = read_shard(os.path.join(out, "tail.safetensors"))
        base = read_shard(os.path.join(model, "model-00001-of-00001.safetensors"))
        trellis = f"{LAYER.format(e=131)}.gate_proj.trellis"
        assert got[trellis] == stub_trellis                     # the re-encoded payload
        assert len(got[trellis]) == len(stub_trellis)
        for suffix in ("suh", "svh"):
            name = f"{LAYER.format(e=131)}.gate_proj.{suffix}"
            assert got[name] == base[name], f"{suffix} was not copied byte for byte"
        # everything the stub did not name is still the original bytes
        for name in got:
            if name != trellis:
                assert got[name] == base[name], name
        assert set(header) == set(got)
        assert not any(n == "model.language_model.layers.0.mlp.gate.weight" for n in got)


def test_a_tensor_payload_carries_its_own_shape():
    """The real encoder returns tensors, and a re-encoded trellis is narrower than the stored one."""
    with tempfile.TemporaryDirectory() as tmp:
        model = model_dir(tmp)
        plan = json.load(open(plan_file(tmp), encoding="utf-8"))
        name = f"{LAYER.format(e=131)}.gate_proj.trellis"

        class FakeArray:
            def __init__(self, data):
                self.data = data

            def tobytes(self):
                return self.data

        class FakeTensor:
            dtype = "torch.int16"
            shape = (8, 8, 32)                        # 32 words: 2.0 bpw, not the stored 48

            def detach(self):
                return self

            def cpu(self):
                return self

            def contiguous(self):
                return self

            def numpy(self):
                return FakeArray(bytes(8 * 8 * 32 * 2))

        out = os.path.join(tmp, "out")
        header = write_requantized_shard(model, plan, out,
                                        lambda prefix, source: {"gate_proj.trellis": FakeTensor()})
        assert header[name]["shape"] == [8, 8, 32]
        assert len(read_shard(os.path.join(out, "tail.safetensors"))[name]) == 8 * 8 * 32 * 2


def test_the_engine_source_is_where_the_signatures_come_from():
    calls = engine_calls(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert calls["quantize_exl3"]["args"][:4] == ["weight", "H_data", "quant_args",
                                                 "return_weight_q"]
    assert calls["LinearEXL3.get_weight_tensor"]["args"] == ["self"]
    assert calls["out_tensors"][:2] == ["suh", "svh"]
    assert calls["out_tensors_line"] > 0
    assert any(c["function"] == "ext.reconstruct" for c in calls["ext_calls"])


TESTS = [
    test_the_dry_run_prints_the_call_it_read_from_the_engine_source,
    test_the_dry_run_counts_tiles_not_scaled_bytes,
    test_a_target_above_the_source_rate_exits_two_with_both_numbers,
    test_a_plan_naming_a_missing_tensor_exits_two_naming_it,
    test_apply_refuses_where_the_engine_or_torch_is_missing,
    test_the_substitution_helper_writes_the_stub_payloads_and_copies_the_rest,
    test_a_tensor_payload_carries_its_own_shape,
    test_the_engine_source_is_where_the_signatures_come_from,
]

if __name__ == "__main__":
    failures = 0
    for t in TESTS:
        try:
            t()
            print(f"PASS {t.__name__}")
        except BaseException as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    sys.exit(1 if failures else 0)
