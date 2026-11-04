#!/usr/bin/env python3
"""Count the n-gram table rows a tokenized corpus reads, and write the hottest ones as a pin set.

The engine's embedding lookup hashes every token position to (K - 1) * heads_per_ngram table rows
(see exllamav3/modules/ngram_embedding.py). With the table streamed from disk, each row a decode
step needs that is not in the row cache is an unbuffered NVMe read, and the row cache is a recency
policy: it keeps what was looked up lately, and the rows a corpus shares across unrelated prompts
are the ones a four-slot ring cannot be relied on to hold. EXL3_NGRAM_ROW_CACHE_PIN takes a file of
row ids the cache loads at open and never evicts, and this tool writes that file:

    python3 bench/ngram_hot_rows.py --corpus corpus.txt --params hash.json --out pin.txt --top-n 4096

**Where the frequencies come from.** They come from the corpus handed to this tool, not from live
traffic. A pin set built here is an estimate of what a workload reads, nothing more: it cannot know
which documents an operator actually sends, how prompt caching turns repeated prefixes into fewer
lookups, or which rows a sampler's token distribution over-represents. Judge it by the row cache's
own counters (`EXL3_NGRAM_ROW_CACHE_STATS=1` prints hits, misses and the pin count at close), not by
the coverage this tool prints.

**Inputs.** A tokenized corpus, one document per line, whitespace-separated decimal token ids ('#'
starts a comment line). Tokenization is the caller's job: this tool loads no model and no tokenizer,
so a plain-text corpus has to be tokenized first by whatever tokenizer the checkpoint was built for.
Hash parameters come from a JSON manifest - the model's own, since the pin file is only correct for
the table those parameters belong to:

    {"ngram_size": 4, "heads_per_ngram": 2, "eos_token_id": 248044,
     "layer_multipliers": [...], "head_offsets": [...], "head_vocab_sizes": [...]}

The three lists are the checkpoint's `<parent>.layer_multipliers`, `.ngram_heads_offsets` and
`.ngram_heads_vocab_sizes` tensors, as decimal ints.

**Why the hash is transcribed here.** The engine's hashing lives in the compiled extension
(ext.ngram_hash_cpu), which needs torch and a built CUDA extension; this tool runs with the standard
library alone, on a box that may not have either, over a corpus that never enters the GPU's memory.
So the mixing loop below is a transcription of ngram.cu's, and
tests/tools/test_ngram_hot_rows.py holds it to the engine's own hashing on a three-document corpus:
the two must produce the same rows, in the same order, or the test fails.
"""

import argparse
import json
import sys
from collections import defaultdict

U64 = 1 << 64
SIGN = 1 << 63


def _u64(x: int) -> int:
    """The low 64 bits: C++'s uint64_t multiply wraps here, and Python's ints do not."""
    return x & (U64 - 1)


def _s64(x: int) -> int:
    """The same bits read as a signed int64: the remainder the hash takes is C's on a signed value,
    so the reinterpretation decides the sign of the modulus."""
    x &= U64 - 1
    return x - U64 if x >= SIGN else x


class NGramHash:
    """The engine's n-gram hashing, fed one document at a time.

    Parameters are per table: `multipliers` has one entry per n-gram order (K), `offsets` and
    `sizes` one per head, in head order, H = (K - 1) * heads_per_ngram of them.
    """

    def __init__(self, ngram_size: int, heads_per_ngram: int, eos_token_id: int,
                 multipliers: list, offsets: list, sizes: list):
        self.ngram_size = ngram_size
        self.heads_per_ngram = heads_per_ngram
        self.eos = eos_token_id
        self.multipliers = multipliers
        self.offsets = offsets
        self.sizes = sizes
        H = (ngram_size - 1) * heads_per_ngram
        if ngram_size < 2:
            raise ValueError(f"ngram_size must be at least 2, got {ngram_size}")
        if len(multipliers) != ngram_size:
            raise ValueError(f"layer_multipliers has {len(multipliers)} entries for ngram_size {ngram_size}")
        if len(offsets) != H or len(sizes) != H:
            raise ValueError(f"head_offsets / head_vocab_sizes need {H} entries "
                             f"(ngram_size {ngram_size} x heads_per_ngram {heads_per_ngram} - 1 order)")
        if any(s <= 0 for s in sizes):
            raise ValueError("every head_vocab_sizes entry must be positive")

    @property
    def num_heads(self) -> int:
        return (self.ngram_size - 1) * self.heads_per_ngram

    def rows(self, tokens: list):
        """The table rows one document reads, in (position, head) order.

        The document is hashed as a sequence start: its (K - 1) positions of history are eos, and an
        eos inside the document segments it, as the engine's eos-segmented hashing does. No position
        is skipped for want of history - the first one hashes against eos - so a document of D
        tokens yields D * num_heads row lookups.
        """
        K = self.ngram_size
        ctx = K - 1
        eos = self.eos
        mult = self.multipliers
        hpn = self.heads_per_ngram
        sizes = self.sizes
        offsets = self.offsets
        row = [eos] * ctx + list(tokens)
        prev_eos = -1                       # last eos position strictly before p
        for p in range(ctx, len(row)):
            if row[p - 1] == eos:
                prev_eos = p - 1
            seg_start = prev_eos + 1
            mixed = _u64(row[p] * mult[0])
            for s in range(1, K):
                src = row[p - s] if p - s >= seg_start else eos
                mixed ^= _u64(src * mult[s])
                lo = (s - 1) * hpn
                for h in range(lo, lo + hpn):
                    yield _s64(mixed) % sizes[h] + offsets[h]

    def count(self, documents) -> tuple:
        """({row id: number of (position, head) slots that hashed to it}, documents read) over the
        documents. Counts, not ids: the 20M-row table this is for would not fit as a list."""
        counts = defaultdict(int)
        n_documents = 0
        for tokens in documents:
            n_documents += 1
            for row in self.rows(tokens):
                counts[row] += 1
        return dict(counts), n_documents


def load_params(path: str) -> dict:
    with open(path, "r", encoding = "utf-8") as f:
        params = json.load(f)
    missing = [k for k in ("ngram_size", "heads_per_ngram", "eos_token_id",
                           "layer_multipliers", "head_offsets", "head_vocab_sizes")
               if k not in params]
    if missing:
        raise ValueError(f"{path}: missing {', '.join(missing)}")
    return params


def read_corpus(path: str):
    """Documents as token-id lists, one per line. A '#' line is a comment; the whole file is read
    lazily so a corpus larger than RAM is fine - the count, not the text, is what is kept."""
    with open(path, "r", encoding = "utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            try:
                yield [int(t) for t in line.split()]
            except ValueError:
                raise ValueError(f"{path}:{n}: expected whitespace-separated token ids") from None


def rank(counts: dict, top_n: int) -> list:
    """The hottest rows: count descending, then row id ascending, so one corpus and one top-N give
    one file, byte for byte, on any run and any box."""
    return sorted(counts.items(), key = lambda kv: (-kv[1], kv[0]))[:top_n]


def write_pin(path: str, hot: list, stats: dict):
    """The pin file: '#' header lines the engine's reader skips, then one row id per line."""
    share = stats["kept"] / stats["ngrams"] if stats["ngrams"] else 0.0
    with open(path, "w", encoding = "utf-8", newline = "\n") as f:
        f.write(f"# ngram_hot_rows: top {len(hot)} of {stats['rows']} rows over "
                f"{stats['documents']} documents of {stats['corpus']}\n")
        f.write(f"# {stats['ngrams']} row lookups counted; these {len(hot)} rows cover "
                f"{100 * share:.1f}% of them\n")
        f.write("# frequencies come from this corpus, not from live traffic: validate the pin set "
                "against the row cache's counters\n")
        for row, _ in hot:
            f.write(f"{row}\n")


def main(argv = None) -> int:
    parser = argparse.ArgumentParser(
        description = "Count the n-gram table rows a tokenized corpus reads and write the hottest "
                      "as an EXL3_NGRAM_ROW_CACHE_PIN file (offline, standard library only).")
    parser.add_argument("--corpus", required = True,
                        help = "tokenized corpus: one document per line, whitespace-separated decimal "
                               "token ids, '#' starts a comment line")
    parser.add_argument("--params", required = True,
                        help = "JSON manifest of the table's hash parameters (ngram_size, "
                               "heads_per_ngram, eos_token_id, layer_multipliers, head_offsets, "
                               "head_vocab_sizes)")
    parser.add_argument("--out", required = True, help = "pin file to write, for EXL3_NGRAM_ROW_CACHE_PIN")
    parser.add_argument("--top-n", type = int, default = 4096,
                        help = "how many of the hottest rows to keep (default: 4096)")
    parser.add_argument("--quiet", action = "store_true", help = "no summary on stdout")
    args = parser.parse_args(argv)
    if args.top_n < 1:
        parser.error("--top-n must be at least 1")

    try:
        params = load_params(args.params)
        hasher = NGramHash(params["ngram_size"], params["heads_per_ngram"], params["eos_token_id"],
                           params["layer_multipliers"], params["head_offsets"],
                           params["head_vocab_sizes"])
    except (OSError, ValueError, KeyError, TypeError) as e:
        print(f"ngram_hot_rows: {e}", file = sys.stderr)
        return 2
    try:
        counts, n_documents = hasher.count(read_corpus(args.corpus))
    except (OSError, ValueError) as e:
        print(f"ngram_hot_rows: {e}", file = sys.stderr)
        return 2

    hot = rank(counts, args.top_n)
    stats = {
        "documents": n_documents,
        "ngrams": sum(counts.values()),
        "rows": len(counts),
        "kept": sum(count for _, count in hot),
        "corpus": args.corpus.replace("\\", "/").rsplit("/", 1)[-1],
    }
    write_pin(args.out, hot, stats)
    if not args.quiet:
        share = 100 * stats["kept"] / stats["ngrams"] if stats["ngrams"] else 0.0
        print(f"ngram_hot_rows: {stats['ngrams']} lookups over {stats['documents']} documents, "
              f"{stats['rows']} distinct rows; wrote the top {len(hot)} to {args.out} "
              f"({share:.1f}% of the lookups)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
