#!/usr/bin/env python3
"""`bench/ngram_pin_manifest.py`: the manifest is read off the checkpoint, or nothing is written.

The script's whole reason to exist is that a pin file is only correct for the table whose hash
parameters built it, so every field has to come out of the checkpoint and every absent field has to
stop the run. Neither needs a model, a GPU or torch: the table file is read through its header, and
that is what these tests build - a synthetic `config.json` and a synthetic `ngram_embedding.safetensors`
whose three small tensors hold known values.

The third test is the contract with the sibling tool: `ngram_hot_rows.py` reads the manifest back with
its own loader and hashes a corpus with it, so a manifest that this script considers well formed but
that tool cannot consume fails here rather than on the box.

Run on a box with no pytest (the Linux side of this repo has neither pytest nor torch):

    python3 bench/test_ngram_pin_manifest.py

or under pytest where it exists (the Windows box, in its venv):

    python3 -m pytest bench/test_ngram_pin_manifest.py -q
"""

import contextlib
import io
import json
import pathlib
import struct
import sys
import traceback

# `pytest bench/...` puts bench/ on sys.path only in the default import mode; this keeps a direct
# `python3 bench/test_ngram_pin_manifest.py` working too.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import ngram_hot_rows  # noqa: E402
import ngram_pin_manifest  # noqa: E402


# ---- the checkpoint these tests read ---------------------------------------------------------------

NGRAM_SIZE = 3
HEADS_PER_NGRAM = 2
EOS = 2
VOCAB = 64
ROW_DIM = 160
SIZES = [101, 293, 499, 601]
OFFSETS = [0, 101, 394, 893]                      # cumsum of SIZES, which is how the reference lays
MULTIPLIERS = [977, 1931, 3907]                   # them out; the values are what has to survive
ROWS = OFFSETS[-1] + SIZES[-1]                    # 1494
PREFIX = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"


def base_config() -> dict:
    """A config shaped like the served checkpoint's: the n-gram knobs under text_config, with the
    reference implementation's derivation knobs present as well, so a test can show they are not the
    source of the numbers."""
    return {
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "text_config": {
            "eos_token_id": EOS,
            "heads_per_ngram": HEADS_PER_NGRAM,
            "ngram_size": NGRAM_SIZE,
            "ngram_vocab_size_base": 20000000,
            "split_ngram_parts": 128,
            "make_ngram_vocab_size_divisible_by": 128,
            "ple_embed_dim": (NGRAM_SIZE - 1) * HEADS_PER_NGRAM * ROW_DIM,
            "vocab_size": VOCAB,
        },
        "tie_word_embeddings": False,
    }


def write_safetensors(path: pathlib.Path, metadata: dict, tensors: dict) -> pathlib.Path:
    """A safetensors file from `{name: (dtype, shape, bytes)}`. The header is JSON padded with
    spaces to an 8-byte boundary, then the data, which is the layout the reader has to parse."""
    header = {"__metadata__": metadata}
    blobs, at = [], 0
    for name, (dtype, shape, blob) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [at, at + len(blob)]}
        at += len(blob)
        blobs.append(blob)
    raw = json.dumps(header).encode("utf-8")
    raw += b" " * ((8 - len(raw) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(blobs))
    return path


def i64(values) -> bytes:
    return struct.pack(f"<{len(values)}q", *values)


def write_checkpoint(tmp_path, *, config = None, drop_tensor: str | None = None) -> pathlib.Path:
    """A checkpoint directory with the synthetic config and table file. `drop_tensor` removes one
    aux tensor, for the refusal test."""
    model = tmp_path / "checkpoint"
    model.mkdir(exist_ok = True)
    (model / "config.json").write_text(json.dumps(config or base_config()), encoding = "utf-8")
    shard_rows = ROWS // 2
    num_heads = (NGRAM_SIZE - 1) * HEADS_PER_NGRAM
    tensors = {
        f"{PREFIX}.head_bias": ("F16", [num_heads, ROW_DIM], b"\0" * (2 * num_heads * ROW_DIM)),
        f"{PREFIX}.head_offsets": ("I64", [len(OFFSETS)], i64(OFFSETS)),
        f"{PREFIX}.head_vocab_sizes": ("I64", [len(SIZES)], i64(SIZES)),
        f"{PREFIX}.layer_multipliers": ("I64", [len(MULTIPLIERS)], i64(MULTIPLIERS)),
        f"{PREFIX}.shard_0.trellis": ("I16", [shard_rows, 81], b"\0" * (2 * shard_rows * 81)),
        f"{PREFIX}.shard_1.trellis": ("I16", [ROWS - shard_rows, 81], b"\0" * (2 * (ROWS - shard_rows) * 81)),
    }
    if drop_tensor:
        assert drop_tensor in tensors, f"no such tensor to drop: {drop_tensor}"
        del tensors[drop_tensor]
    write_safetensors(
        model / ngram_pin_manifest.NGRAM_FILE,
        {"format": "exl3_ngram_trellis", "version": "1", "K": "8", "codebook": "mul1",
         "row_dim": str(ROW_DIM), "rows": str(ROWS), "shard_rows": str(shard_rows)},
        tensors,
    )
    return model


def run_main(argv):
    """(exit code, stdout, stderr) of the script's own entry point."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = ngram_pin_manifest.main(argv)
    return code, out.getvalue(), err.getvalue()


# ---- 1 -------------------------------------------------------------------------------------------

def test_the_manifest_carries_every_field_the_tool_requires_from_the_checkpoint(tmp_path):
    """The manifest holds exactly the six fields `ngram_hot_rows.load_params` demands, and the three
    list-valued ones hold the header's values - not the config's derivation knobs, which would give a
    plausible and completely different table."""
    model = write_checkpoint(tmp_path)
    out_path = tmp_path / "ngram_hash.json"
    code, stdout, stderr = run_main(["--model", str(model), "--out-params", str(out_path)])
    assert code == 0, stderr
    manifest = json.loads(out_path.read_text(encoding = "utf-8"))

    assert set(manifest) == {"ngram_size", "heads_per_ngram", "eos_token_id", "layer_multipliers",
                             "head_offsets", "head_vocab_sizes"}
    assert manifest["ngram_size"] == NGRAM_SIZE
    assert manifest["heads_per_ngram"] == HEADS_PER_NGRAM
    assert manifest["eos_token_id"] == EOS

    # the numbers come from the header's tensors, in head order, not from config.json
    assert manifest["layer_multipliers"] == MULTIPLIERS
    assert manifest["head_offsets"] == OFFSETS
    assert manifest["head_vocab_sizes"] == SIZES
    assert len(manifest["head_offsets"]) == (NGRAM_SIZE - 1) * HEADS_PER_NGRAM

    # ... and the summary says where they came from, so the operator can check the pairing by eye
    assert str(model / ngram_pin_manifest.NGRAM_FILE) in stdout
    assert "layer_multipliers" in stdout and "head_vocab_sizes" in stdout


def test_a_config_missing_a_required_field_exits_nonzero_and_names_it(tmp_path):
    """A checkpoint that cannot yield the manifest stops the run: exit code 2, the field's own name
    on stderr, and no manifest file left behind for a later run to mistake for a good one."""
    config = base_config()
    del config["text_config"]["heads_per_ngram"]
    model = write_checkpoint(tmp_path, config = config)
    out_path = tmp_path / "ngram_hash.json"
    code, stdout, stderr = run_main(["--model", str(model), "--out-params", str(out_path)])
    assert code != 0
    assert "heads_per_ngram" in stderr
    assert not out_path.exists()


def test_a_header_missing_a_tensor_exits_nonzero_and_names_it(tmp_path):
    """The other half of the same rule: with the header gone quiet about a per-head size, there is no
    derivation this script is willing to substitute, so it says which tensor is absent."""
    model = write_checkpoint(tmp_path, drop_tensor = f"{PREFIX}.head_vocab_sizes")
    out_path = tmp_path / "ngram_hash.json"
    code, stdout, stderr = run_main(["--model", str(model), "--out-params", str(out_path)])
    assert code != 0
    assert "head_vocab_sizes" in stderr
    assert "ngram_embedding.head_vocab_sizes" in stderr
    assert not out_path.exists()


# ---- 3 -------------------------------------------------------------------------------------------

def test_the_written_manifest_round_trips_through_the_tools_own_loader(tmp_path):
    """`ngram_hot_rows.py` reads the manifest with `load_params`, builds its hasher from it and writes
    a pin file: the manifest this script writes is that tool's input, checked by that tool."""
    model = write_checkpoint(tmp_path)
    params_path = tmp_path / "ngram_hash.json"
    assert run_main(["--model", str(model), "--out-params", str(params_path), "--quiet"])[0] == 0

    params = ngram_hot_rows.load_params(str(params_path))
    hasher = ngram_hot_rows.NGramHash(params["ngram_size"], params["heads_per_ngram"],
                                      params["eos_token_id"], params["layer_multipliers"],
                                      params["head_offsets"], params["head_vocab_sizes"])
    assert hasher.num_heads == len(OFFSETS)

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("3 4 5 6 7\n3 4 5 6 7\n3 4 8 9 10\n", encoding = "utf-8")
    pin = tmp_path / "pin.txt"
    code = ngram_hot_rows.main(["--corpus", str(corpus), "--params", str(params_path),
                                "--out", str(pin), "--top-n", "8", "--quiet"])
    assert code == 0
    ids = [int(line) for line in pin.read_text(encoding = "utf-8").splitlines()
           if line.strip() and not line.startswith("#")]
    assert ids
    assert len(set(ids)) == len(ids)
    assert all(0 <= i < ROWS for i in ids)

    # the same rows the engine would hash to: the hasher built from the manifest is the one that
    # wrote them, so the count of lookups matches the corpus's positions x heads
    counts, documents = hasher.count(ngram_hot_rows.read_corpus(str(corpus)))
    assert documents == 3
    assert sum(counts.values()) == 3 * 5 * hasher.num_heads


# ---- the corpus writer (the one part that needs the tokenizer) ------------------------------------

class _StubIds:
    """What `tokenizer.encode(...)` returns, for the format test: the engine returns a tensor, and
    `flatten().tolist()` is how the ids come out of it."""

    def __init__(self, ids):
        self.ids = ids

    def flatten(self):
        return self

    def tolist(self):
        return self.ids


class _StubTokenizer:
    """An ord() tokenizer: no engine, no torch, and the ids are checkable by eye."""

    def encode(self, text, add_bos = False, add_eos = False):
        assert not add_bos and not add_eos, "the corpus is tokenized without bos/eos"
        return _StubIds([ord(c) for c in text])


def test_the_tokenize_helper_writes_a_corpus_the_pin_tool_reads(tmp_path):
    """The corpus format is the contract with `ngram_hot_rows.read_corpus`: one document per line,
    whitespace-separated decimal ids, and a '#' line reserved for comments. The tokenizer is stubbed
    because the real one needs the venv; the writing does not."""
    model = write_checkpoint(tmp_path)
    text_dir = tmp_path / "text"
    text_dir.mkdir()
    (text_dir / "a.txt").write_text("abc\n\nXYZ 01\n", encoding = "utf-8")
    (text_dir / "b.txt").write_text("# a note to the reader\nabc\n", encoding = "utf-8")

    tokenizer = _StubTokenizer()
    original = ngram_pin_manifest.load_tokenizer
    ngram_pin_manifest.load_tokenizer = lambda model_dir: tokenizer
    try:
        out = tmp_path / "corpus.txt"
        stats = ngram_pin_manifest.tokenize(str(text_dir), str(model), str(out), [".txt"], 0)
    finally:
        ngram_pin_manifest.load_tokenizer = original

    documents = list(ngram_hot_rows.read_corpus(str(out)))
    assert documents == [[97, 98, 99], [88, 89, 90, 32, 48, 49], [97, 98, 99]]
    assert stats["documents"] == 3
    assert stats["comments"] == 1                  # the '#' line: the corpus format owns that character
    assert stats["files"] == 2
    assert stats["ids"] == 12
    # the tool that consumes it counts the same positions x heads the corpus holds
    params = ngram_pin_manifest.build_manifest(str(model))[0]
    hasher = ngram_hot_rows.NGramHash(params["ngram_size"], params["heads_per_ngram"],
                                      params["eos_token_id"], params["layer_multipliers"],
                                      params["head_offsets"], params["head_vocab_sizes"])
    counts, read = hasher.count(documents)
    assert read == 3 and sum(counts.values()) == 12 * hasher.num_heads


TESTS = [
    test_the_manifest_carries_every_field_the_tool_requires_from_the_checkpoint,
    test_a_config_missing_a_required_field_exits_nonzero_and_names_it,
    test_a_header_missing_a_tensor_exits_nonzero_and_names_it,
    test_the_written_manifest_round_trips_through_the_tools_own_loader,
    test_the_tokenize_helper_writes_a_corpus_the_pin_tool_reads,
]


if __name__ == "__main__":
    failures = 0
    for test in TESTS:
        try:
            import tempfile
            with tempfile.TemporaryDirectory() as tmp:
                test(pathlib.Path(tmp))
        except BaseException:
            failures += 1
            print(f"FAIL {test.__name__}")
            traceback.print_exc()
        else:
            print(f"PASS {test.__name__}")
    print(f"{len(TESTS) - failures}/{len(TESTS)} passed")
    sys.exit(1 if failures else 0)
