#!/usr/bin/env python3
"""The hash-parameter manifest `ngram_hot_rows.py` needs, derived from the checkpoint it describes.

`EXL3_NGRAM_ROW_CACHE_PIN` takes a file of table row ids, and `bench/ngram_hot_rows.py` builds that
file from a tokenized corpus plus a JSON manifest of the table's hash parameters. This script writes
the manifest, from the checkpoint alone:

    python3 bench/ngram_pin_manifest.py --model /mnt/d/llms/<checkpoint> --out-params ngram_hash.json

**Why every field is read and none is computed.** A head's vocabulary size decides which row a token
position hashes to. Derive it from a plausible-looking formula and you get a well-formed pin file for
a table that does not exist: the engine loads the ids, they are all inside `0..num_rows`, and every
one of them is the wrong row. There is no error to see - only a row cache that misses. So the three
values that live in the checkpoint's `ngram_embedding.safetensors` are read out of it, and the three
that live in `config.json` are read out of that. If a value is not in one of the two places, this
script prints which one is missing and exits nonzero; it does not fill the gap.

Where each field comes from:

    ngram_size          config.json   text_config.ngram_size
    heads_per_ngram     config.json   text_config.heads_per_ngram
    eos_token_id        config.json   text_config.eos_token_id, else the top-level key
    layer_multipliers   header        *layer_multipliers            (I64, K entries)
    head_offsets        header        *ngram_embedding.head_offsets (I64, (K-1)*H entries)
    head_vocab_sizes    header        *ngram_embedding.head_vocab_sizes (I64, (K-1)*H entries)

Those are the same tensors the engine's own loader reads for a quantized table
(`exllamav3/modules/ngram_embedding.py`, `_load_aux`: `head_offsets`, `head_vocab_sizes`,
`layer_multipliers`), so the manifest and the served table cannot disagree about the names. The
reference implementation's `split_ngram_parts` / `ngram_vocab_size_base` / prime-step derivation is
deliberately not reimplemented here: exllamav3 reads the stored tensors instead, and a size this
script guessed would be indistinguishable from the measured one in the pin file that followed.

**What is checked before a manifest is written.** The two halves have to describe the same table:
`len(layer_multipliers) == ngram_size`, both per-head lists of length `(ngram_size - 1) *
heads_per_ngram`, every size positive, offsets strictly ascending from 0 (the engine maps a row to
its head with `searchsorted`, which reads a gap or a shuffle as a different head), the last head's
end inside the table's row count, `eos_token_id` inside the vocabulary, and `ple_embed_dim` equal to
`num_heads * row_dim` from the file's own metadata. A failure names both numbers.

**Reading is cheap.** Only the 8-byte header length, the JSON header (21 KB on the served checkpoint)
and the 200 bytes of the three small tensors are read, so this finishes in under a second - measured
0.035 s on WSL - and never touches the checkpoint's 51.8 GB of table data. It can be run while a
measurement campaign owns the card.

**Tokenizing (`--tokenize`) is the step that needs the venv.** A text corpus has to become token ids
before `ngram_hot_rows.py` can count rows, and only the checkpoint's own tokenizer produces the ids
the served model will see. So `--tokenize <text-dir> --out <ids-file>` loads that tokenizer through
the engine (`exllamav3.tokenizer`), which imports torch and `tokenizers`, and refuses - naming what is
absent - when the engine is not importable or the checkpoint carries no tokenizer files. It is the one
part of this script that cannot run with the standard library alone:

    # Windows, from the engine's tree, with the venv run_pass.ps1 uses
    <venv>\\Scripts\\python.exe bench\\ngram_pin_manifest.py --model D:\\llms\\<checkpoint> ^
        --tokenize <text-dir> --out corpus.txt --out-params ngram_hash.json

**Input text layout.** One document per line of the text files under `<text-dir>` (subdirectories are
walked, files are taken in sorted order, `--text-suffix` defaults to `.txt`). A line that starts with
`#` is skipped: the corpus format reserves `#` for comments, and a text line beginning with one would
otherwise be truncated at that character by `read_corpus`.
"""

import argparse
import json
import os
import struct
import sys

# The tensor suffix the converter writes for each manifest field, in the order the loader lists them.
# `endswith` matching covers both the model-prefixed keys of current converter output and the bare
# "ngram_embedding.*" keys of earlier files, exactly as exllamav3/conversion/ngram.py resolves them.
AUX_SUFFIXES = {
    "layer_multipliers": ("ngram_embedding.layer_multipliers",),
    "head_offsets": ("ngram_embedding.head_offsets", "ngram_heads_offsets"),
    "head_vocab_sizes": ("ngram_embedding.head_vocab_sizes", "ngram_heads_vocab_sizes"),
}

MANIFEST_FIELDS = ("ngram_size", "heads_per_ngram", "eos_token_id",
                   "layer_multipliers", "head_offsets", "head_vocab_sizes")

NGRAM_FILE = "ngram_embedding.safetensors"

# int64 is what the converter writes for all three, uint64 what a re-save on some paths produces, and
# the 32-bit forms are accepted because the value is the value whatever the width
INT_DTYPES = {"I64": ("<q", 8), "U64": ("<Q", 8), "I32": ("<i", 4), "U32": ("<I", 4)}

TOKENIZER_NAMES = ("tokenizer.json", "vocab.json", "tokenizer.model", "spiece.model")


class Refusal(ValueError):
    """A field that is not in the checkpoint, or two halves of it that disagree. The message is
    printed as written: it names the absent thing, never a substitute for it."""


# ---- config.json ----------------------------------------------------------------------------------

def read_config(model_dir: str) -> dict:
    path = os.path.join(model_dir, "config.json")
    try:
        with open(path, "r", encoding = "utf-8") as f:
            config = json.load(f)
    except FileNotFoundError:
        raise Refusal(f"{path}: no config.json; --model must be the checkpoint directory") from None
    except json.JSONDecodeError as e:
        raise Refusal(f"{path}: not JSON ({e})") from None
    if not isinstance(config, dict):
        raise Refusal(f"{path}: expected a JSON object at the top level")
    return config


def config_field(config: dict, key: str):
    """(value, where) for a language-model knob. `text_config` is where qwen4_exp keeps them
    (Qwen4ExpConfig reads every one through `text_config->`); the top level is the fallback the other
    architectures use. `None` is not a value: a config that spells `"pad_token_id": null` has not
    given a field."""
    for holder, label in ((config.get("text_config"), "text_config"), (config, None)):
        if not isinstance(holder, dict):
            continue
        value = holder.get(key)
        if value is not None:
            return value, (f"{label}.{key}" if label else key)
    raise Refusal(f"config.json has no {key} (looked in text_config and at the top level), "
                  f"so {key} is not derivable from this checkpoint")


def config_int(config: dict, key: str) -> tuple:
    value, where = config_field(config, key)
    try:
        return int(value), where
    except (TypeError, ValueError):
        raise Refusal(f"{where} = {value!r} is not an integer") from None


# ---- the table file's header ----------------------------------------------------------------------

def read_header(path: str) -> tuple:
    """(metadata, tensors, data_base) of a safetensors file, from its header alone.

    safetensors is 8 little-endian bytes of header length, the JSON header padded with spaces to an
    8-byte boundary, then the tensor data; `data_offsets` in each entry are relative to the data."""
    with open(path, "rb") as f:
        head = f.read(8)
        if len(head) != 8:
            raise Refusal(f"{path}: {len(head)} bytes, too short to be a safetensors file")
        (length,) = struct.unpack("<Q", head)
        if not 0 < length < 1 << 32:
            raise Refusal(f"{path}: header length {length} is not plausible for a safetensors file")
        raw = f.read(length)
    if len(raw) != length:
        raise Refusal(f"{path}: header is {len(raw)} of {length} bytes, the file is truncated")
    try:
        header = json.loads(raw)
    except UnicodeDecodeError as e:
        raise Refusal(f"{path}: header is not UTF-8 ({e})") from None
    except json.JSONDecodeError as e:
        raise Refusal(f"{path}: header is not JSON ({e})") from None
    if not isinstance(header, dict):
        raise Refusal(f"{path}: header is not a JSON object")
    metadata = header.pop("__metadata__", None) or {}
    return metadata, header, 8 + length


def find_aux(header: dict, field: str, path: str) -> str:
    """The one tensor name that holds `field`, or a refusal naming every suffix that was tried."""
    suffixes = AUX_SUFFIXES[field]
    for suffix in suffixes:
        keys = [k for k in header if k.endswith(suffix)]
        if len(keys) > 1:
            raise Refusal(f"{path}: {len(keys)} tensors end in {suffix} ({', '.join(sorted(keys))}), "
                          f"expected exactly one")
        if keys:
            return keys[0]
    raise Refusal(
        f"{path}: no tensor name ends in {' or '.join(suffixes)}, so {field} is not in this checkpoint. "
        f"The engine's loader reads {field} from there (ngram_embedding.py, _load_aux); this script "
        f"will not derive it from config.json's split_ngram_parts / ngram_vocab_size_base, because a "
        f"guessed size hashes every position to a different row and the pin file built from it would "
        f"still look well formed")


def read_int_tensor(path: str, data_base: int, name: str, info: dict) -> list:
    """The tensor's values as ints. The header gives dtype, shape and a byte range; nothing here
    needs an array library for a list of 16 numbers."""
    dtype = info.get("dtype")
    shape = info.get("shape")
    offsets = info.get("data_offsets")
    if dtype not in INT_DTYPES:
        raise Refusal(f"{path}: {name} has dtype {dtype}, expected one of "
                      f"{', '.join(sorted(INT_DTYPES))} (the engine reads these as .long())")
    if not isinstance(shape, list) or not shape or not isinstance(offsets, list) or len(offsets) != 2:
        raise Refusal(f"{path}: {name} has no usable shape/data_offsets in the header")
    fmt, width = INT_DTYPES[dtype]
    start, end = offsets
    count = 1
    for dim in shape:
        count *= dim
    if end - start != count * width:
        raise Refusal(f"{path}: {name} spans {end - start} bytes, but shape {shape} of {dtype} "
                      f"({width} B) is {count * width}")
    with open(path, "rb") as f:
        f.seek(data_base + start)
        raw = f.read(end - start)
    if len(raw) != end - start:
        raise Refusal(f"{path}: {name} is {len(raw)} of {end - start} bytes, the file is truncated")
    return list(struct.unpack(f"<{count}{fmt[1]}", raw))


def table_rows(metadata: dict, header: dict) -> int:
    """Rows the pinned table spans: the trellis shards summed, as the engine counts them
    (`RowTable`), or the metadata count for the older single-tensor layout."""
    shards = [k for k in header if k.endswith(".trellis")]
    if shards:
        return sum(header[k]["shape"][0] for k in shards)
    try:
        return int(metadata["rows"])
    except (KeyError, TypeError, ValueError):
        raise Refusal(f"the header has no .trellis tensor and no usable 'rows' metadata, so the "
                      f"table's row count is unknown") from None


# ---- the manifest ---------------------------------------------------------------------------------

def build_manifest(model_dir: str) -> tuple:
    """(manifest, facts) for the checkpoint at `model_dir`. Every field is read from the checkpoint
    or the call raises `Refusal` naming the missing one."""
    config = read_config(model_dir)
    ngram_size, size_where = config_int(config, "ngram_size")
    heads_per_ngram, heads_where = config_int(config, "heads_per_ngram")
    eos_token_id, eos_where = config_int(config, "eos_token_id")
    text = config.get("text_config")
    text = text if isinstance(text, dict) else {}
    ple_embed_dim = text.get("ple_embed_dim", config.get("ple_embed_dim"))
    vocab_size = text.get("vocab_size", config.get("vocab_size"))

    table_path = os.path.join(model_dir, NGRAM_FILE)
    if not os.path.isfile(table_path):
        raise Refusal(
            f"{table_path}: no n-gram table file. The three hash lists this manifest carries live in "
            f"its auxiliary tensors, so a manifest cannot be written without it - and it is needed: "
            f"a table held in RAM cannot take a pin set (row_table.py refuses EXL3_NGRAM_ROW_CACHE_PIN "
            f"for it), so a checkpoint without this file has no pin file to build")
    metadata, header, data_base = read_header(table_path)

    names = {field: find_aux(header, field, table_path) for field in
             ("layer_multipliers", "head_offsets", "head_vocab_sizes")}
    values = {field: read_int_tensor(table_path, data_base, name, header[name])
              for field, name in names.items()}

    manifest = {
        "ngram_size": ngram_size,
        "heads_per_ngram": heads_per_ngram,
        "eos_token_id": eos_token_id,
        "layer_multipliers": values["layer_multipliers"],
        "head_offsets": values["head_offsets"],
        "head_vocab_sizes": values["head_vocab_sizes"],
    }

    num_heads = (ngram_size - 1) * heads_per_ngram
    rows = table_rows(metadata, header)
    row_dim = metadata.get("row_dim")
    check_manifest(manifest, num_heads, rows, row_dim, ple_embed_dim, vocab_size, table_path)

    facts = {
        "table": table_path,
        "rows": rows,
        "num_heads": num_heads,
        "row_dim": int(row_dim) if row_dim is not None else None,
        "tensors": {field: names[field] for field in names},
        "config": {"ngram_size": size_where, "heads_per_ngram": heads_where,
                   "eos_token_id": eos_where},
        "metadata": dict(metadata),
    }
    return manifest, facts


def check_manifest(manifest: dict, num_heads: int, rows: int, row_dim, ple_embed_dim,
                   vocab_size, where: str):
    """Everything that has to hold for the manifest to describe the table in `where`. Each failure
    names both numbers, so the message says which side is wrong."""
    K = manifest["ngram_size"]
    hpn = manifest["heads_per_ngram"]
    if K < 2:
        raise Refusal(f"ngram_size is {K}: the table hashes (K - 1) orders, so K is at least 2")
    if hpn < 1:
        raise Refusal(f"heads_per_ngram is {hpn}: every order has at least one head")
    multipliers = manifest["layer_multipliers"]
    if len(multipliers) != K:
        raise Refusal(f"layer_multipliers holds {len(multipliers)} entries, but ngram_size is {K}: "
                      f"the hash reads one multiplier per order")
    offsets = manifest["head_offsets"]
    sizes = manifest["head_vocab_sizes"]
    if len(offsets) != len(sizes):
        raise Refusal(f"head_offsets holds {len(offsets)} entries and head_vocab_sizes "
                      f"{len(sizes)}: one of each per head")
    if len(offsets) != num_heads:
        raise Refusal(f"head_offsets holds {len(offsets)} entries, but ngram_size {K} x "
                      f"heads_per_ngram {hpn} needs {num_heads} = (K - 1) * heads_per_ngram")
    if any(size <= 0 for size in sizes):
        bad = next(i for i, size in enumerate(sizes) if size <= 0)
        raise Refusal(f"head_vocab_sizes[{bad}] is {sizes[bad]}: every head has at least one row")
    if offsets[0] != 0:
        raise Refusal(f"head_offsets[0] is {offsets[0]}, not 0: the engine finds a row's head with "
                      f"searchsorted over these offsets, so rows below the first head's start would "
                      f"be read as that head's")
    for h in range(1, len(offsets)):
        if offsets[h] <= offsets[h - 1]:
            raise Refusal(f"head_offsets is not strictly ascending: [{h - 1}] = {offsets[h - 1]}, "
                          f"[{h}] = {offsets[h]}")
    end = offsets[-1] + sizes[-1]
    if rows and end > rows:
        raise Refusal(f"the heads cover rows 0..{end - 1}, but {where} holds {rows} rows: this "
                      f"manifest does not describe this table")
    if vocab_size is not None:
        eos = manifest["eos_token_id"]
        if not 0 <= eos < int(vocab_size):
            raise Refusal(f"eos_token_id is {eos}, outside the vocabulary of {vocab_size} tokens: "
                          f"the n-gram history is padded with it, so it has to be a real token")
    if row_dim is not None and ple_embed_dim is not None:
        expected = num_heads * int(row_dim)
        if int(ple_embed_dim) != expected:
            raise Refusal(f"ple_embed_dim is {ple_embed_dim}, but ngram_size {K} x "
                          f"heads_per_ngram {hpn} = {num_heads} heads of {row_dim} dims is {expected}")


# ---- tokenizing (the step that needs the venv) ----------------------------------------------------

def load_tokenizer(model_dir: str):
    """The checkpoint's own tokenizer, through the engine. Says which half is missing rather than
    raising a ModuleNotFoundError: on this box the Linux side has no torch, and that is a supported
    way to run everything except this."""
    found = [name for name in TOKENIZER_NAMES if os.path.isfile(os.path.join(model_dir, name))]
    if not found:
        raise Refusal(f"{model_dir}: no tokenizer file ({', '.join(TOKENIZER_NAMES)}). --tokenize has "
                      f"to use the checkpoint's own tokenizer - a corpus tokenized by anything else "
                      f"produces ids the served model never sees")
    try:
        from exllamav3.model.config import Config
        from exllamav3.tokenizer import Tokenizer
    except Exception as e:
        raise Refusal(f"--tokenize needs the engine's venv (torch, tokenizers and exllamav3 "
                      f"importable), and importing it here failed: {type(e).__name__}: {e}. Run this "
                      f"with the venv's python, from the engine's own tree") from None
    try:
        config = Config.from_directory(model_dir)
        return Tokenizer.from_config(config)
    except Exception as e:
        raise Refusal(f"{model_dir}: the engine could not build this checkpoint's tokenizer: "
                      f"{type(e).__name__}: {e}") from None


def text_files(text_dir: str, suffixes: list) -> list:
    """Every file under `text_dir` with one of `suffixes`, sorted, so one directory gives one corpus
    byte for byte on any run."""
    if not os.path.isdir(text_dir):
        raise Refusal(f"{text_dir}: not a directory of text documents")
    paths = []
    for root, dirs, files in os.walk(text_dir):
        dirs.sort()
        for name in sorted(files):
            if os.path.splitext(name)[1].lower() in suffixes:
                paths.append(os.path.join(root, name))
    if not paths:
        raise Refusal(f"{text_dir}: no file with suffix {' or '.join(suffixes)}: nothing to tokenize")
    return paths


def tokenize(text_dir: str, model_dir: str, out_path: str, suffixes: list, max_tokens: int) -> dict:
    """Write the token-id corpus: one document per line, as `ngram_hot_rows.read_corpus` reads it.
    Returns the counts for the summary."""
    # the tokenizer first: a box without the engine's venv should hear that, not a complaint about
    # which text files it does or does not have
    tokenizer = load_tokenizer(model_dir)
    paths = text_files(text_dir, suffixes)
    stats = {"files": len(paths), "documents": 0, "ids": 0, "empty": 0, "comments": 0,
             "truncated": 0}
    with open(out_path, "w", encoding = "utf-8", newline = "\n") as out:
        out.write(f"# ngram_pin_manifest: token ids from {len(paths)} text files under "
                  f"{text_dir.replace(chr(92), '/')}\n")
        out.write("# one document per line; a line of a text file is one document\n")
        for path in paths:
            with open(path, "r", encoding = "utf-8", errors = "replace") as source:
                for line in source:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("#"):
                        # the corpus format reserves '#' for comments: a text line starting with one
                        # would be truncated at that character by read_corpus, so it is skipped here
                        stats["comments"] += 1
                        continue
                    try:
                        ids = tokenizer.encode(line, add_bos = False,
                                               add_eos = False).flatten().tolist()
                    except Exception as e:
                        raise Refusal(f"{path}: the tokenizer failed on a {len(line)}-character "
                                      f"line: {type(e).__name__}: {e}") from None
                    if not ids:
                        stats["empty"] += 1
                        continue
                    if max_tokens and len(ids) > max_tokens:
                        ids = ids[:max_tokens]
                        stats["truncated"] += 1
                    out.write(" ".join(str(i) for i in ids) + "\n")
                    stats["documents"] += 1
                    stats["ids"] += len(ids)
    if not stats["documents"]:
        raise Refusal(f"{text_dir}: {len(paths)} files produced no documents; {out_path} holds its "
                      f"header only")
    stats["out"] = out_path
    return stats


# ---- CLI ------------------------------------------------------------------------------------------

def main(argv = None) -> int:
    parser = argparse.ArgumentParser(
        description = "Write the hash-parameter manifest bench/ngram_hot_rows.py needs, from the "
                      "checkpoint's config.json and ngram_embedding.safetensors header (standard "
                      "library only; reads the header and three small tensors, not the table).",
        epilog = "With neither --out-params nor --tokenize the manifest goes to stdout. --tokenize is "
                 "the one part that needs the engine's venv.",
    )
    parser.add_argument("--model", required = True,
                        help = "checkpoint directory: config.json plus ngram_embedding.safetensors")
    parser.add_argument("--out-params", metavar = "PATH",
                        help = "where to write the JSON manifest (ngram_hot_rows.py --params)")
    parser.add_argument("--tokenize", metavar = "TEXT_DIR",
                        help = "also write the token-id corpus for that tool, from this directory of "
                               "text files: needs the checkpoint's tokenizer and the engine's venv")
    parser.add_argument("--out", metavar = "IDS_FILE",
                        help = "the token-id corpus to write (with --tokenize)")
    parser.add_argument("--text-suffix", action = "append", metavar = "SUFFIX",
                        help = "which text files under --tokenize to read (repeatable, default: .txt)")
    parser.add_argument("--max-tokens", type = int, default = 0, metavar = "N",
                        help = "cap a document at N token ids (default: 0, no cap)")
    parser.add_argument("--quiet", action = "store_true", help = "no summary on stdout")
    args = parser.parse_args(argv)
    if args.out and not args.tokenize:
        parser.error("--out is the token-id corpus and needs --tokenize")
    if args.tokenize and not args.out:
        parser.error("--tokenize needs --out (the corpus file to write)")
    if args.max_tokens < 0:
        parser.error("--max-tokens cannot be negative")

    try:
        manifest, facts = build_manifest(args.model)
    except (Refusal, OSError) as e:
        print(f"ngram_pin_manifest: {e}", file = sys.stderr)
        return 2

    if args.out_params:
        with open(args.out_params, "w", encoding = "utf-8", newline = "\n") as f:
            f.write(json.dumps({field: manifest[field] for field in MANIFEST_FIELDS}, indent = 4)
                    + "\n")
    else:
        print(json.dumps(manifest, indent = 4))
    # the manifest is the only thing on stdout when it goes there, so `> ngram_hash.json` captures
    # JSON and nothing else; the summary moves to stderr in that case
    say = (lambda line: print(line, file = sys.stderr)) if not args.out_params else print
    if not args.quiet:
        cfg = facts["config"]
        say(f"ngram_pin_manifest: {facts['table']}: {facts['rows']} rows"
            + (f", row_dim {facts['row_dim']}" if facts["row_dim"] is not None else "")
            + f", {facts['num_heads']} heads; metadata {facts['metadata'].get('format', '?')}")
        say(f"ngram_pin_manifest: ngram_size {manifest['ngram_size']} ({cfg['ngram_size']}), "
            f"heads_per_ngram {manifest['heads_per_ngram']} ({cfg['heads_per_ngram']}), "
            f"eos_token_id {manifest['eos_token_id']} ({cfg['eos_token_id']})")
        say("ngram_pin_manifest: from the header: "
            + ", ".join(f"{field} = {name}" for field, name in facts["tensors"].items()))
        if args.out_params:
            say(f"ngram_pin_manifest: wrote {args.out_params}")

    if args.tokenize:
        suffixes = [s.lower() if s.startswith(".") else f".{s.lower()}"
                    for s in (args.text_suffix or [".txt"])]
        try:
            stats = tokenize(args.tokenize, args.model, args.out, suffixes, args.max_tokens)
        except (Refusal, OSError) as e:
            print(f"ngram_pin_manifest: {e}", file = sys.stderr)
            return 2
        if not args.quiet:
            say(f"ngram_pin_manifest: {stats['documents']} documents, {stats['ids']} token ids "
                f"from {stats['files']} text files -> {stats['out']}")
            if stats["comments"] or stats["empty"] or stats["truncated"]:
                say(f"ngram_pin_manifest: skipped {stats['comments']} '#' lines and "
                    f"{stats['empty']} empty documents; truncated {stats['truncated']} documents "
                    f"at --max-tokens {args.max_tokens}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
