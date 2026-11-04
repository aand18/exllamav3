"""
bench/ngram_hot_rows.py: the offline pin-set builder for EXL3_NGRAM_ROW_CACHE_PIN.

The tool counts table row ids over a tokenized corpus with its own copy of the n-gram hash, because
it has to run with the standard library alone (no torch, no model load). A second copy of a hash is
a divergence waiting to happen, so test_the_tool_counts_the_rows_the_engine_hashes hands the same
corpus to the engine's own ngram_hash_cpu and compares the ordered pin set: the tool is only worth
its docstring while that test passes.

test_the_pin_file_is_deterministic_and_readable_back needs no engine at all.
"""

import importlib.util
import json
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
TOOL = REPO / "bench" / "ngram_hot_rows.py"

# A pin file the engine's own reader accepts: decimal ids, one per line, '#' comments skipped
def load_tool():
    assert TOOL.is_file(), f"{TOOL} is missing"
    spec = importlib.util.spec_from_file_location("ngram_hot_rows", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def params_json(path, ngram_size, hpn, eos, multipliers, sizes):
    """The manifest the tool reads: the model's own hash parameters, as JSON. Offsets are the
    per-head starts the checkpoint stores, here derived from the sizes the way the reference does."""
    offsets, at = [], 0
    for size in sizes:
        offsets.append(at)
        at += size
    path.write_text(json.dumps({
        "ngram_size": ngram_size,
        "heads_per_ngram": hpn,
        "eos_token_id": eos,
        "layer_multipliers": list(multipliers),
        "head_offsets": offsets,
        "head_vocab_sizes": list(sizes),
    }), encoding = "utf-8")
    return path


def read_ids(path):
    """The pin file as the engine reads it: ids in file order, comments and blanks ignored."""
    ids = []
    for line in path.read_text(encoding = "utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            ids.append(int(line))
    return ids


def write_corpus(path, documents):
    """One document per line, whitespace-separated decimal token ids."""
    path.write_text("".join(" ".join(str(t) for t in doc) + "\n" for doc in documents), encoding = "utf-8")
    return path


def test_the_pin_file_is_deterministic_and_readable_back(tmp_path):
    """Ran twice on the same corpus and parameters, the tool writes the same bytes; the file is a pin
    file, so the engine's reader sees exactly the ids the tool chose, in frequency order."""
    tool = load_tool()
    # stdlib-only parameters: any primes will do, the shape is what this test checks
    sizes = [101, 293, 499, 601]
    params = params_json(tmp_path / "params.json", 3, 2, 2, [97, 193, 389], sizes)
    corpus = write_corpus(tmp_path / "corpus.txt", [
        [11, 12, 13, 14],
        [11, 12, 13, 14],
        [11, 12, 15, 16],
    ])

    first = tmp_path / "pin-a.txt"
    second = tmp_path / "pin-b.txt"
    assert tool.main(["--corpus", str(corpus), "--params", str(params), "--out", str(first),
                      "--top-n", "8"]) == 0
    assert tool.main(["--corpus", str(corpus), "--params", str(params), "--out", str(second),
                      "--top-n", "8"]) == 0
    assert first.read_bytes() == second.read_bytes()

    ids = read_ids(first)
    assert 0 < len(ids) <= 8
    assert len(set(ids)) == len(ids)                 # a pin set is a set
    assert any(line.startswith("#") for line in first.read_text(encoding = "utf-8").splitlines())
    # the column the engine hashes into: every id inside the table the parameters describe
    assert all(0 <= i < sum(sizes) for i in ids)


def test_the_tool_counts_the_rows_the_engine_hashes(tmp_path):
    """The pin set is compared against ext.ngram_hash_cpu over the same three documents. Any drift in
    the tool's transcription - the wrap, the xor, the sign of the remainder, the eos segmentation -
    reorders or renames the ids and fails here."""
    torch = pytest.importorskip("torch")
    try:
        # the form the engine itself uses: exllamav3/ext.py binds the compiled module as an attribute
        from exllamav3.ext import exllamav3_ext as ext
    except ImportError as e:
        pytest.skip(f"the compiled extension is not built here ({e})")
    tool = load_tool()

    from exllamav3.modules.ngram_embedding import _find_nth_prime_after

    EOS = 248044
    K, HPN = 4, 3                                   # a K-gram table with three heads per order
    H = (K - 1) * HPN
    g = torch.Generator().manual_seed(7)
    multipliers = (torch.randint(-2 ** 62, 2 ** 62, (K,), generator = g, dtype = torch.int64) * 2 + 1)
    sizes = torch.tensor([_find_nth_prime_after(100003 + 37 * h, 1) for h in range(H)], dtype = torch.int64)
    offsets = torch.cat([torch.zeros(1, dtype = torch.int64), torch.cumsum(sizes, 0)[:-1]])

    documents = [
        [101, 102, 103, 104, 105],
        [101, 102, 103, 104, 105],                  # the same document twice: every row count 2
        [101, 102, 201, 202, 203],
    ]
    corpus = write_corpus(tmp_path / "corpus.txt", documents)
    params = params_json(tmp_path / "params.json", K, HPN, EOS,
                         multipliers.tolist(), sizes.tolist())

    # What the engine hashes: one call per document, eos-padded history, the row ids deduped with
    # their counts (inverse maps every (position, head) slot to its row)
    counts = {}
    for document in documents:
        history = torch.tensor([[EOS] * (K - 1) + document], dtype = torch.int64)
        n = len(document) * H
        uids = torch.empty(n, dtype = torch.int64)
        inverse = torch.empty(n, dtype = torch.int64)
        heads = torch.empty(n, dtype = torch.int32)
        U = ext.ngram_hash_cpu(history, len(document), multipliers, offsets, sizes, HPN, EOS,
                               uids, inverse, heads)
        for row, count in zip(uids[:U].tolist(), torch.bincount(inverse, minlength = U).tolist()):
            counts[row] = counts.get(row, 0) + count
    assert counts and max(counts.values()) > 1       # the repeated document is visible in the counts

    out = tmp_path / "pin.txt"
    top_n = 12
    assert tool.main(["--corpus", str(corpus), "--params", str(params), "--out", str(out),
                      "--top-n", str(top_n)]) == 0
    expected = [row for row, _ in sorted(counts.items(), key = lambda kv: (-kv[1], kv[0]))][:top_n]
    assert read_ids(out) == expected
